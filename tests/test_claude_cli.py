"""The claude_cli provider: shells out to the locally installed Claude Code
CLI, so proofreading uses the machine's Claude subscription instead of an API
key. No network code of our own — just argv construction and process handling.
"""

import subprocess
from types import SimpleNamespace

import pytest

from conftest import make_cfg
from aiproof.llm.client import CLAUDE_CLI_SYSTEM, LLMClient, find_claude_cli
from aiproof.llm.errors import LLMError


def cli_cfg(**over):
    base = dict(provider="claude-cli", model="haiku", endpoint="")
    base.update(over)
    return make_cfg(**base)


@pytest.fixture
def fake_binary(tmp_path):
    path = tmp_path / "claude"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


class TestDiscovery:
    def test_configured_path_wins(self, fake_binary):
        assert find_claude_cli(fake_binary) == fake_binary

    def test_configured_path_missing_is_none(self, tmp_path):
        assert find_claude_cli(str(tmp_path / "nope")) is None

    def test_falls_back_to_path_lookup(self, monkeypatch, fake_binary):
        monkeypatch.setattr("aiproof.llm.client.shutil.which",
                            lambda name: fake_binary)
        assert find_claude_cli() == fake_binary

    def test_none_when_not_installed(self, monkeypatch):
        monkeypatch.setattr("aiproof.llm.client.shutil.which", lambda name: None)
        monkeypatch.setattr("aiproof.llm.client._CLAUDE_CLI_PATHS", ())
        assert find_claude_cli() is None


class TestRequestShape:
    def test_argv_and_stdin(self, fake_binary):
        req = LLMClient(cli_cfg(endpoint=fake_binary)).build_request("PROMPT")
        assert req["stdin"] == "PROMPT"
        argv = req["argv"]
        assert argv[0] == fake_binary
        assert "--print" in argv
        assert argv[argv.index("--model") + 1] == "haiku"
        # The big startup win: skips loading the user's MCP servers (~2.7s).
        assert "--strict-mcp-config" in argv
        assert argv[argv.index("--system-prompt") + 1] == CLAUDE_CLI_SYSTEM

    def test_system_override_is_used(self, fake_binary):
        req = LLMClient(cli_cfg(endpoint=fake_binary)).build_request(
            "PROMPT", system="EDIT MODE SYSTEM")
        assert req["argv"][req["argv"].index("--system-prompt") + 1] == \
            "EDIT MODE SYSTEM"

    def test_needs_no_api_key(self):
        from aiproof.llm.providers import PROVIDERS
        assert PROVIDERS["claude-cli"]["needs_key"] is False


class TestExecution:
    def _client(self, monkeypatch, fake_binary, result=None, raises=None):
        client = LLMClient(cli_cfg(endpoint=fake_binary))

        def fake_run(argv, **kwargs):
            if raises is not None:
                raise raises
            fake_run.kwargs = kwargs
            return result

        monkeypatch.setattr("aiproof.llm.client.subprocess.run", fake_run)
        self.fake_run = fake_run
        return client

    def test_stdout_returned(self, monkeypatch, fake_binary):
        client = self._client(monkeypatch, fake_binary, result=SimpleNamespace(
            returncode=0, stdout=b"corrected text\n", stderr=b""))
        assert client.query("p") == "corrected text\n"

    def test_prompt_goes_via_stdin_not_argv(self, monkeypatch, fake_binary):
        """User text must never be an argv element (escaping, length limits)."""
        client = self._client(monkeypatch, fake_binary, result=SimpleNamespace(
            returncode=0, stdout=b"ok", stderr=b""))
        client.query("secret text")
        assert self.fake_run.kwargs["input"] == b"secret text"

    def test_runs_in_neutral_cwd(self, monkeypatch, fake_binary):
        """A project CLAUDE.md must not leak into every proofread."""
        client = self._client(monkeypatch, fake_binary, result=SimpleNamespace(
            returncode=0, stdout=b"ok", stderr=b""))
        client.query("p")
        import tempfile
        assert self.fake_run.kwargs["cwd"] == tempfile.gettempdir()

    def test_nonzero_exit_raises_with_stderr(self, monkeypatch, fake_binary):
        client = self._client(monkeypatch, fake_binary, result=SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"something broke"))
        with pytest.raises(LLMError, match="something broke"):
            client.query("p")

    def test_auth_failure_gets_a_hint(self, monkeypatch, fake_binary):
        client = self._client(monkeypatch, fake_binary, result=SimpleNamespace(
            returncode=1, stdout=b"", stderr=b"Invalid API key / not logged in"))
        with pytest.raises(LLMError, match="sign in"):
            client.query("p")

    def test_timeout_becomes_llm_error(self, monkeypatch, fake_binary):
        client = self._client(
            monkeypatch, fake_binary,
            raises=subprocess.TimeoutExpired(cmd="claude", timeout=60))
        with pytest.raises(LLMError, match="timed out"):
            client.query("p")

    def test_missing_binary_is_actionable(self, monkeypatch):
        monkeypatch.setattr("aiproof.llm.client.shutil.which", lambda n: None)
        monkeypatch.setattr("aiproof.llm.client._CLAUDE_CLI_PATHS", ())
        with pytest.raises(LLMError, match="Claude CLI not found"):
            LLMClient(cli_cfg()).query("p")


class TestFallbackIntegration:
    def test_can_be_a_fallback_target(self, monkeypatch, fake_binary):
        """A dead local provider should be able to fail over to the CLI."""
        from aiproof.llm.client import proofread_with_fallback

        cfg = make_cfg(fallback={"provider": "claude-cli", "model": "haiku",
                                 "endpoint": fake_binary})

        def factory(c):
            client = LLMClient(c)
            if c["provider"] == "claude-cli":
                monkeypatch.setattr(client, "query", lambda p: "fixed text")
            else:
                monkeypatch.setattr(
                    client, "query",
                    lambda p: (_ for _ in ()).throw(LLMError("host down")))
            return client

        out, _, client, used_fb = proofread_with_fallback(
            cfg, "teh text", _client_factory=factory)
        assert used_fb is True
        assert client.cfg["provider"] == "claude-cli"

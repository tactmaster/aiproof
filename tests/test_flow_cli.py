"""Flow routes through the CLI: `aiproof proofread` (stdin -> stdout) and
`aiproof trigger[-clipboard]` in both oneshot and daemon-call modes.
"""

from types import SimpleNamespace

import pytest

from conftest import make_cfg
from aiproof.cli import main
from aiproof.llm.client import LLMError

pytestmark = pytest.mark.flow


class FakeCLIProofread:
    def __init__(self):
        self.corrected = "the text"
        self.path = "ok"
        self.error = None
        self.primary_error = None  # fires on_fallback, marks used_fallback
        self.calls = []

    def __call__(self, cfg, text, on_fallback=None, _client_factory=None):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        if self.primary_error is not None and on_fallback:
            on_fallback(self.primary_error)
        client = SimpleNamespace(cfg={"provider": "ollama"},
                                 model="test-model", last_path=self.path)
        return self.corrected, 0.2, client, self.primary_error is not None


@pytest.fixture
def fake_cli_llm(monkeypatch):
    fake = FakeCLIProofread()
    monkeypatch.setattr("aiproof.llm.client.proofread_with_fallback", fake)
    monkeypatch.setattr("aiproof.config.load", lambda: make_cfg())
    return fake


def test_proofread_empty_stdin(fake_stdin, fake_cli_llm, capsys, caplog):
    """cli: empty stdin -> exit 1, nothing sent to the model."""
    fake_stdin("   \n")
    assert main(["proofread"]) == 1
    assert "no input on stdin" in capsys.readouterr().err
    assert fake_cli_llm.calls == []
    assert "no input on stdin" in caplog.text


def test_proofread_strips_and_readds_trailing_newline(fake_stdin, fake_cli_llm,
                                                      capsys, caplog):
    """cli: the single stdin trailing newline is stripped for the model and
    re-added on stdout."""
    fake_stdin("teh text\n")
    assert main(["proofread"]) == 0
    out, err = capsys.readouterr()
    assert fake_cli_llm.calls == ["teh text"]
    assert out == "the text\n"
    assert "[ollama/test-model in 0.2s, path=ok]" in err
    assert "stripped trailing newline=True" in caplog.text
    assert "proofread done: path=ok, used_fallback=False" in caplog.text


def test_proofread_no_trailing_newline(fake_stdin, fake_cli_llm, capsys):
    """cli: input without a trailing newline gets none added."""
    fake_stdin("teh text")
    assert main(["proofread"]) == 0
    assert fake_cli_llm.calls == ["teh text"]
    assert capsys.readouterr().out == "the text"


def test_proofread_double_newline_kept(fake_stdin, fake_cli_llm, capsys):
    """cli: a blank-line ending is part of the text, not stdin noise."""
    fake_stdin("teh text\n\n")
    assert main(["proofread"]) == 0
    assert fake_cli_llm.calls == ["teh text\n\n"]
    assert capsys.readouterr().out == "the text"


def test_proofread_llm_error(fake_stdin, fake_cli_llm, capsys, caplog):
    """cli: an LLMError -> exit 1 with the message on stderr."""
    fake_stdin("teh text\n")
    fake_cli_llm.error = LLMError("HTTP 401 (check your API key)")
    assert main(["proofread"]) == 1
    assert "aiproof: HTTP 401 (check your API key)" in capsys.readouterr().err
    assert "proofread failed: HTTP 401 (check your API key)" in caplog.text


def test_proofread_unexpected_error_propagates(fake_stdin, fake_cli_llm):
    """cli: non-LLMError exceptions escape as a traceback. Known quirk —
    see docs/flow.md."""
    fake_stdin("teh text\n")
    fake_cli_llm.error = ValueError("Unknown provider: typo")
    with pytest.raises(ValueError, match="Unknown provider"):
        main(["proofread"])


def test_proofread_guards_rejected_warns_but_exit_zero(fake_stdin,
                                                       fake_cli_llm, capsys,
                                                       caplog):
    """cli: path=fallback prints the NOT-verified warning but still exits 0.
    Known quirk (the Nautilus script reads only the exit code) — see
    docs/flow.md."""
    fake_stdin("teh text\n")
    fake_cli_llm.corrected = "teh text"
    fake_cli_llm.path = "fallback"
    assert main(["proofread"]) == 0
    err = capsys.readouterr().err
    assert "NOT verified clean" in err
    assert "rejected by the guards; output is the unverified original" \
        in caplog.text


def test_proofread_reports_provider_failover(fake_stdin, fake_cli_llm, capsys):
    """cli: a provider failover is announced on stderr."""
    fake_stdin("teh text\n")
    fake_cli_llm.primary_error = LLMError("dead host")
    assert main(["proofread"]) == 0
    assert "primary provider failed (dead host); trying fallback…" \
        in capsys.readouterr().err


@pytest.mark.parametrize("argv,flow,result,rc", [
    (["trigger", "--oneshot"], "paste", True, 0),
    (["trigger", "--oneshot"], "paste", False, 1),
    (["trigger-clipboard", "--oneshot"], "clipboard", True, 0),
])
def test_trigger_oneshot(monkeypatch, caplog, argv, flow, result, rc):
    """cli: --oneshot runs the flow in-process; exit code mirrors it."""
    ran = []

    class FakeOrch:
        def __init__(self, cfg, notifier):
            pass

        def run_paste_flow(self):
            ran.append("paste")
            return result

        def run_clipboard_flow(self):
            ran.append("clipboard")
            return result

    monkeypatch.setattr("aiproof.orchestrator.Orchestrator", FakeOrch)
    monkeypatch.setattr("aiproof.config.load", lambda: make_cfg())
    assert main(argv) == rc
    assert ran == [flow]
    assert f"oneshot {flow} flow -> {result}" in caplog.text


def test_trigger_daemon_unreachable(gi_stub, capsys, caplog):
    """cli: without --oneshot the daemon is called over D-Bus; an unreachable
    daemon -> exit 1 with a hint."""
    assert main(["trigger"]) == 1
    err = capsys.readouterr().err
    assert "could not reach the daemon (no daemon)" in err
    assert "aiproof daemon" in err
    assert "daemon unreachable over DBus: no daemon" in caplog.text

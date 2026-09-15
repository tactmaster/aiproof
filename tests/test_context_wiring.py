"""Context-profile wiring: prompt splicing, LLMClient threading, and the
config flag. The profile *mining* itself lives in tests/test_context.py — here
load_words_and_summary is either monkeypatched or relied on for its
([], None) contract when context_aware is off.
"""

import pytest

from conftest import make_cfg
from aiproof import config
from aiproof.llm.client import LLMClient
from aiproof.llm.prompts import (
    _line_note,
    PROOFREAD_PROMPT,
    _CONSTRAINT_ANCHOR,
    build_proofread_prompt,
)

HEADER = "CONTEXT (background"


# -- prompt splicing ----------------------------------------------------------


class TestPromptWithoutContext:
    def test_no_args_is_unchanged(self):
        """No context args: byte-identical to the plain formatted prompt."""
        prompt = build_proofread_prompt("x")
        assert prompt == PROOFREAD_PROMPT.format(text="x", line_note=_line_note("x"))
        assert prompt == build_proofread_prompt("x", None, None)
        assert HEADER not in prompt
        assert _CONSTRAINT_ANCHOR in prompt
        assert prompt.count("INPUT TEXT:") == 1

    def test_empty_context_is_unchanged(self):
        """Empty list / empty summary are falsy: still no block."""
        assert build_proofread_prompt("x", [], "") == PROOFREAD_PROMPT.format(text="x", line_note=_line_note("x"))
        assert HEADER not in build_proofread_prompt("x", [], None)


class TestPromptWithContext:
    def test_words_only(self):
        prompt = build_proofread_prompt("x", ["Mender", "ydotool"])
        assert HEADER in prompt
        assert "Known user vocabulary" in prompt
        assert "Mender, ydotool" in prompt
        assert "Typical texts:" not in prompt

    def test_summary_only(self):
        prompt = build_proofread_prompt("x", None, "support emails")
        assert HEADER in prompt
        assert "Typical texts: support emails" in prompt
        assert "Known user vocabulary" not in prompt

    def test_both(self):
        prompt = build_proofread_prompt("x", ["Mender", "ydotool"], "support emails")
        assert "Typical texts: support emails" in prompt
        assert "Mender, ydotool" in prompt

    def test_block_sits_between_examples_and_anchor(self):
        prompt = build_proofread_prompt("x", ["Mender"], "support emails")
        assert prompt.index("EXAMPLES:") < prompt.index(HEADER)
        assert prompt.index(HEADER) < prompt.index(_CONSTRAINT_ANCHOR)
        assert prompt.count("INPUT TEXT:") == 1
        assert prompt.count(_CONSTRAINT_ANCHOR) == 1
        # Blank line between the block and the anchor line.
        assert "\n\n" + _CONSTRAINT_ANCHOR in prompt

    def test_input_text_still_carries_the_text(self):
        prompt = build_proofread_prompt("teh text", ["Mender"], "notes")
        assert '"teh text"' in prompt

    def test_braces_in_context_do_not_crash_or_corrupt(self):
        prompt = build_proofread_prompt(
            "x", ["{brace}", "f{oo}"], "templates like {x} and {{y}}"
        )
        assert "Typical texts: templates like {x} and {{y}}" in prompt
        assert "{brace}, f{oo}" in prompt
        assert '"x"' in prompt

    def test_braces_in_text_still_work_with_context(self):
        prompt = build_proofread_prompt("a {x} b", ["Mender"], None)
        assert '"a {x} b"' in prompt

    def test_block_prepended_when_anchor_missing(self, monkeypatch):
        """Anchor gone (prompt edited): the block is prepended, not dropped."""
        monkeypatch.setattr("aiproof.llm.prompts.PROOFREAD_PROMPT",
                            'INPUT TEXT:\n"{text}"')
        prompt = build_proofread_prompt("x", ["Mender"])
        assert prompt.startswith(HEADER)
        assert '"x"' in prompt


# -- client wiring ------------------------------------------------------------


def make_client(**cfg_over):
    return LLMClient(make_cfg(**cfg_over))


@pytest.fixture
def captured(monkeypatch):
    """LLMClient factory whose query() records prompts instead of calling out."""
    prompts = []

    def build(words=None, summary=None, **cfg_over):
        monkeypatch.setattr(
            "aiproof.llm.client.context.load_words_and_summary",
            lambda cfg: (words if words is not None else [], summary),
        )
        client = make_client(**cfg_over)
        monkeypatch.setattr(client, "query", lambda p: prompts.append(p) or p)
        return client

    build.prompts = prompts
    return build


class TestClientWiring:
    def test_init_loads_the_profile(self, monkeypatch):
        monkeypatch.setattr(
            "aiproof.llm.client.context.load_words_and_summary",
            lambda cfg: (["Mender"], "support emails"),
        )
        client = make_client(context_aware=True)
        assert client.context_words == ["Mender"]
        assert client.context_summary == "support emails"

    def test_init_passes_the_full_cfg(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "aiproof.llm.client.context.load_words_and_summary",
            lambda cfg: seen.append(cfg) or ([], None),
        )
        cfg = make_cfg(context_aware=True)
        LLMClient(cfg)
        assert seen == [cfg]

    def test_proofread_prompt_carries_context(self, captured, monkeypatch):
        client = captured(["Mender", "ydotool"], "support emails",
                          context_aware=True)
        monkeypatch.setattr("aiproof.llm.client.full_clean",
                            lambda resp, orig: "the text")
        monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                            lambda orig, cand, q: (cand, "ok"))
        client.proofread("teh text")
        assert len(captured.prompts) == 1
        prompt = captured.prompts[0]
        assert HEADER in prompt
        assert "Mender, ydotool" in prompt
        assert "Typical texts: support emails" in prompt

    def test_correct_line_prompt_carries_context(self, captured, monkeypatch):
        client = captured(["Mender", "ydotool"], "support emails",
                          context_aware=True)
        monkeypatch.setattr("aiproof.llm.client.full_clean",
                            lambda resp, orig: "the line")
        assert client._correct_line("teh line") == "the line"
        assert len(captured.prompts) == 1
        prompt = captured.prompts[0]
        assert HEADER in prompt
        assert "Mender, ydotool" in prompt
        assert "Typical texts: support emails" in prompt

    def test_context_aware_off_sends_no_block(self, monkeypatch):
        """Default cfg (context_aware False): the real loader returns ([], None)."""
        client = make_client()
        assert (client.context_words, client.context_summary) == ([], None)
        prompts = []
        monkeypatch.setattr(client, "query", lambda p: prompts.append(p) or "the text")
        monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                            lambda orig, cand, q: (cand, "ok"))
        client.proofread("teh text")
        assert prompts and all(HEADER not in p for p in prompts)


# -- config flag --------------------------------------------------------------


@pytest.fixture
def temp_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


class TestConfigFlag:
    def test_default_is_off(self):
        assert config.DEFAULTS["context_aware"] is False

    def test_roundtrips(self, temp_config_home):
        cfg = config.load()
        assert cfg["context_aware"] is False
        cfg["context_aware"] = True
        config.save(cfg)
        assert config.load()["context_aware"] is True


class TestLineNote:
    def test_single_line_note(self):
        prompt = build_proofread_prompt("one line here")
        assert "THIS INPUT IS ONE SINGLE LINE." in prompt

    def test_multiline_note_counts_lines(self):
        prompt = build_proofread_prompt("a\nb\nc\nd")
        assert "THIS INPUT HAS EXACTLY 4 LINES." in prompt
        assert "EXACTLY 4 LINES, EACH LINE CORRECTED IN PLACE." in prompt

    def test_note_sits_before_input_text(self):
        prompt = build_proofread_prompt("a\nb")
        assert prompt.index("EXACTLY 2 LINES") < prompt.index('INPUT TEXT:')

    def test_constrained_retry_inherits_note(self):
        from aiproof.llm.prompts import build_constrained_prompt
        prompt = build_constrained_prompt("a\nb\nc", ["Semicolons added"])
        assert "THIS INPUT HAS EXACTLY 3 LINES." in prompt

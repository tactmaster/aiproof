"""Edit-list prompt construction."""

from aiproof.llm.prompts import (
    EDIT_PROMPT_PREFIX,
    EDIT_SYSTEM_PROMPT,
    build_edit_prompt,
)


def test_prefix_is_constant_across_texts():
    """The fixed prefix must be byte-identical for every request so Ollama's
    prompt-prefix cache covers it."""
    a = build_edit_prompt("first text")
    b = build_edit_prompt("a completely different text\nwith two lines")
    assert a.startswith(EDIT_PROMPT_PREFIX) and b.startswith(EDIT_PROMPT_PREFIX)
    assert a.endswith('Input: "first text"\nOutput:')


def test_variable_parts_come_after_the_prefix():
    p = build_edit_prompt("teh", context_words=["Mender"],
                          context_summary="support emails",
                          app_setting="a Slack message")
    assert p.startswith(EDIT_PROMPT_PREFIX)
    tail = p[len(EDIT_PROMPT_PREFIX):]
    assert "Mender" in tail and "support emails" in tail
    assert "WHERE THIS TEXT IS BEING WRITTEN: a Slack message" in tail
    assert tail.endswith('Input: "teh"\nOutput:')


def test_braces_in_text_survive():
    assert build_edit_prompt("set {x} = {y}").endswith('Input: "set {x} = {y}"\nOutput:')


def test_minimal_style_is_the_bare_text():
    assert build_edit_prompt("teh text", context_words=["X"], style="minimal") == "teh text"


def test_system_prompt_teaches_the_format():
    assert "wrong => right" in EDIT_SYSTEM_PROMPT
    assert "NO CHANGES" in EDIT_SYSTEM_PROMPT
    assert "\n" not in EDIT_SYSTEM_PROMPT

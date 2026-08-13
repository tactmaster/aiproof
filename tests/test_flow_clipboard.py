"""Flow routes through Orchestrator.run_clipboard_flow (primary selection ->
clipboard) and the degraded variants reached via run_paste_flow.
"""

import pytest

from aiproof.llm.client import LLMError

pytestmark = pytest.mark.flow


def test_empty_selection(orch, fake_clipboard, fake_llm, notifier, caplog):
    """clipboard: no primary selection -> 'Select some text first'."""
    fake_clipboard.primary_text = None
    o = orch()

    assert o.run_clipboard_flow() is False
    assert notifier.summaries[-1] == "Select some text first"
    assert fake_llm.calls == []
    assert o.states == ["busy", "error"]
    assert "no primary selection; nothing to proofread" in caplog.text


def test_whitespace_selection(orch, fake_clipboard, fake_llm, notifier):
    """clipboard: whitespace-only primary selection is rejected."""
    fake_clipboard.primary_text = "   \n\t"
    o = orch()

    assert o.run_clipboard_flow() is False
    assert notifier.summaries[-1] == "Select some text first"
    assert fake_llm.calls == []


def test_happy_path(orch, fake_clipboard, fake_keystroke, fake_llm, notifier,
                    caplog):
    """clipboard: correct the primary selection onto the clipboard — no
    keystrokes, no degraded note."""
    fake_clipboard.primary_text = "teh text"
    fake_llm.corrected = "the text"
    o = orch()

    assert o.run_clipboard_flow() is True
    assert ("set_text", "the text") in fake_clipboard.calls
    assert fake_keystroke.calls == []
    assert notifier.summaries == ["Proofreading…", "Corrected in 0.3s"]
    assert "press Ctrl+V to paste" in notifier.last[1]
    assert "Automatic paste unavailable" not in notifier.last[1]
    assert o.states == ["busy", "idle"]
    assert "correction copied to clipboard: 8 -> 8 chars (degraded=False)" \
        in caplog.text


def test_degraded_note_only_when_ydotool_missing(orch, fake_clipboard,
                                                 fake_keystroke, fake_llm,
                                                 notifier):
    """clipboard: the ydotool note appears when degraded by a missing tool,
    not when the user chose clipboard-only mode."""
    fake_keystroke.available_flag = False
    fake_clipboard.primary_text = "teh text"
    o = orch()

    assert o.run_paste_flow() is True
    assert "Automatic paste unavailable — ydotool not set up." in notifier.last[1]


def test_path_note_in_summary(orch, fake_clipboard, fake_llm, notifier):
    """clipboard: salvage paths are surfaced in the summary note."""
    fake_clipboard.primary_text = "teh text"
    fake_llm.path = "segments"
    o = orch()

    assert o.run_clipboard_flow() is True
    assert notifier.summaries[-1] == "Corrected in 0.3s — segments mode"


def test_llm_error(orch, fake_clipboard, fake_llm, notifier, caplog):
    """clipboard: an LLMError is reported and logged; nothing is copied."""
    fake_clipboard.primary_text = "teh text"
    fake_llm.error = LLMError("Cannot reach http://gpu:11434: connection failed")
    o = orch()

    assert o.run_clipboard_flow() is False
    assert notifier.last == (
        "Proofreading failed",
        "Cannot reach http://gpu:11434: connection failed",
    )
    assert ("set_text", "teh text") not in fake_clipboard.calls
    assert "proofread failed: Cannot reach" in caplog.text


def test_unchanged_clean_via_fallback_provider(orch, fake_clipboard, fake_llm,
                                               notifier, caplog):
    """clipboard: a clean result served by the fallback provider says so."""
    fake_clipboard.primary_text = "clean text"
    fake_llm.corrected = "clean text"
    fake_llm.used_fallback = True
    o = orch(fallback={"provider": "ollama", "model": "local-model"})

    assert o.run_clipboard_flow() is True
    assert notifier.summaries[-1] == "No changes needed"
    assert "via fallback local-model" in notifier.last[1]
    assert "no changes needed (path=ok, used_fallback=True)" in caplog.text


def test_unchanged_guards_rejected(orch, fake_clipboard, fake_llm, notifier,
                                   caplog):
    """clipboard: unchanged + path=fallback -> honest unsafe report."""
    fake_clipboard.primary_text = "teh text"
    fake_llm.corrected = "teh text"
    fake_llm.path = "fallback"
    o = orch()

    assert o.run_clipboard_flow() is False
    assert notifier.summaries[-1] == "Couldn't apply corrections safely"
    assert o.states == ["busy", "error"]
    assert "rejected by the guards; reporting as unsafe" in caplog.text


def test_set_text_failure(orch, fake_clipboard, fake_llm, notifier, caplog):
    """clipboard: a wl-copy failure is caught and reported."""
    fake_clipboard.primary_text = "teh text"
    fake_clipboard.set_text_error = RuntimeError("wl-copy exploded")
    o = orch()

    assert o.run_clipboard_flow() is False
    assert notifier.last == ("Proofreading failed", "wl-copy exploded")
    assert "clipboard flow failed" in caplog.text

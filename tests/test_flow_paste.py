"""Flow routes through Orchestrator.run_paste_flow (in-place correction).

Every test drives the real orchestrator with the LLM, clipboard, keystrokes,
notifications, and time.sleep all faked (see conftest.py seam rules).
"""

import pytest

from aiproof.clipboard import NON_TEXT
from aiproof.llm.client import LLMError

pytestmark = pytest.mark.flow


def test_disabled(orch, fake_llm, notifier, caplog):
    """gate: disabled orchestrator refuses the trigger and logs it."""
    o = orch()
    o.enabled = False

    assert o.run_paste_flow() is False
    assert notifier.summaries == ["aiproof is disabled"]
    assert o.states == []  # never went busy
    assert fake_llm.calls == []
    assert "trigger ignored: aiproof is disabled" in caplog.text


def test_lock_contention(orch, fake_llm, notifier, caplog):
    """gate: a second trigger while one is running is rejected."""
    o = orch()
    assert o._lock.acquire(blocking=False)  # simulate a run in progress
    try:
        assert o.run_paste_flow() is False
    finally:
        o._lock.release()
    assert notifier.summaries == ["Already proofreading"]
    assert fake_llm.calls == []
    assert "a run is already in progress" in caplog.text


def test_chosen_clipboard_only_mode(orch, fake_clipboard, fake_keystroke,
                                    fake_llm, notifier, caplog):
    """gate: paste_mode=clipboard-only routes to the clipboard flow WITHOUT
    the 'ydotool not set up' note (the user chose this mode)."""
    fake_clipboard.primary_text = "teh text"
    o = orch(paste_mode="clipboard-only")

    assert o.run_paste_flow() is True
    assert "paste_mode=clipboard-only; using clipboard flow" in caplog.text
    assert fake_keystroke.calls == []  # ydotool never consulted
    assert "Automatic paste unavailable" not in notifier.last[1]
    assert ("set_text", "corrected text") in fake_clipboard.calls


def test_ydotool_unavailable_degrades(orch, fake_clipboard, fake_keystroke,
                                      fake_llm, notifier, caplog):
    """gate: missing ydotool degrades to the clipboard flow WITH the note."""
    fake_keystroke.available_flag = False
    fake_clipboard.primary_text = "teh text"
    o = orch()

    assert o.run_paste_flow() is True
    assert "ydotool unavailable; degrading to clipboard-only" in caplog.text
    assert "Automatic paste unavailable — ydotool not set up." in notifier.last[1]


def test_happy_path(orch, fake_clipboard, fake_keystroke, fake_llm, notifier,
                    no_sleep, caplog):
    """paste: capture via ctrl+c, correct, paste in place, restore clipboard."""
    fake_clipboard.poll_results.append("teh text")
    fake_llm.corrected = "the text"
    o = orch()

    assert o.run_paste_flow() is True
    assert fake_keystroke.calls == ["available", "release_modifiers", "ctrl_c",
                                    "release_modifiers", "ctrl_v"]
    assert fake_clipboard.call_names == \
        ["save", "clear", "poll_for_text", "set_text", "restore"]
    assert fake_clipboard.calls[-1] == ("restore", fake_clipboard.saved_state)
    assert no_sleep == [0.2, 0.05, 0.1, 0.5]  # restore_delay_ms default 500
    assert fake_llm.calls == ["teh text"]
    assert notifier.summaries == ["Proofreading…", "Text corrected"]
    assert "Replaced in place in 0.3s (ollama/phi3.5:latest)." == notifier.last[1]
    assert o.states == ["busy", "idle"]
    assert "captured selection: 8 chars, 1 lines, 0 blank lines" in caplog.text
    assert "correction pasted in place: 8 -> 8 chars" in caplog.text
    assert "flow done: Text corrected" in caplog.text


def test_restore_delay_configurable(orch, fake_clipboard, fake_llm, no_sleep):
    """paste: restore_delay_ms config drives the pre-restore sleep."""
    fake_clipboard.poll_results.append("teh text")
    o = orch(restore_delay_ms=100)

    assert o.run_paste_flow() is True
    assert no_sleep == [0.2, 0.05, 0.1, 0.1]


def test_ctrl_c_retry_then_success(orch, fake_clipboard, fake_keystroke,
                                   fake_llm, caplog):
    """paste: empty clipboard after the first ctrl+c retries once, then works."""
    fake_clipboard.poll_results.extend([None, "teh text"])
    o = orch()

    assert o.run_paste_flow() is True
    assert fake_keystroke.calls == ["available", "release_modifiers", "ctrl_c",
                                    "release_modifiers", "ctrl_c",
                                    "release_modifiers", "ctrl_v"]
    assert "clipboard still empty after ctrl+c; retrying once" in caplog.text


def test_nothing_copied_after_retry(orch, fake_clipboard, fake_llm, notifier,
                                    caplog):
    """paste: both ctrl+c attempts yield nothing -> restore + notify."""
    o = orch()  # poll_results empty -> None both times

    assert o.run_paste_flow() is False
    assert notifier.summaries[-1] == "Select some text first"
    assert ("restore", fake_clipboard.saved_state) in fake_clipboard.calls
    assert fake_llm.calls == []
    assert o.states == ["busy", "error"]
    assert "nothing copied after ctrl+c" in caplog.text


def test_non_text_selection(orch, fake_clipboard, fake_llm, notifier, caplog):
    """paste: clipboard fills with non-text content -> restore + notify."""
    fake_clipboard.poll_results.append(NON_TEXT)
    o = orch()

    assert o.run_paste_flow() is False
    assert notifier.summaries[-1] == "Selection is not text"
    assert ("restore", fake_clipboard.saved_state) in fake_clipboard.calls
    assert fake_llm.calls == []
    assert "selection is not text" in caplog.text


def test_whitespace_only_selection(orch, fake_clipboard, fake_llm, notifier,
                                   caplog):
    """paste: a whitespace-only selection is rejected before the LLM
    (previously it was sent to the model, unlike the clipboard flow)."""
    fake_clipboard.poll_results.append("   \n ")
    o = orch()

    assert o.run_paste_flow() is False
    assert notifier.summaries[-1] == "Select some text first"
    assert fake_llm.calls == []
    assert ("restore", fake_clipboard.saved_state) in fake_clipboard.calls
    assert "empty or whitespace only" in caplog.text


def test_llm_error(orch, fake_clipboard, fake_llm, notifier, caplog):
    """paste: an LLMError becomes 'Proofreading failed' + restore, and is
    logged (previously the biggest logging gap)."""
    fake_clipboard.poll_results.append("teh text")
    fake_llm.error = LLMError("HTTP 401 (check your API key)")
    o = orch()

    assert o.run_paste_flow() is False
    assert notifier.last == ("Proofreading failed",
                             "HTTP 401 (check your API key)")
    assert ("restore", fake_clipboard.saved_state) in fake_clipboard.calls
    assert o.states == ["busy", "error"]
    assert "proofread failed: HTTP 401 (check your API key)" in caplog.text


def test_unexpected_error(orch, fake_clipboard, fake_llm, notifier, caplog):
    """paste: a non-LLMError from the pipeline is reported as unexpected."""
    fake_clipboard.poll_results.append("teh text")
    fake_llm.error = ValueError("Unknown provider: typo")
    o = orch()

    assert o.run_paste_flow() is False
    assert notifier.last == ("Proofreading failed",
                             "Unexpected error: Unknown provider: typo")
    assert "unexpected proofread failure" in caplog.text


def test_unchanged_clean(orch, fake_clipboard, fake_keystroke, fake_llm,
                         notifier, caplog):
    """paste: unchanged text on a clean path -> 'No changes needed', restore,
    no paste."""
    fake_clipboard.poll_results.append("clean text")
    fake_llm.corrected = "clean text"
    o = orch()

    assert o.run_paste_flow() is True
    assert notifier.summaries[-1] == "No changes needed"
    assert "“clean text” is clean" in notifier.last[1]
    assert "ctrl_v" not in fake_keystroke.calls
    assert fake_clipboard.call_names == \
        ["save", "clear", "poll_for_text", "restore"]  # no set_text
    assert "no changes needed (path=ok, used_fallback=False)" in caplog.text


def test_unchanged_guards_rejected(orch, fake_clipboard, fake_llm, notifier,
                                   caplog):
    """paste: unchanged because every correction was rejected by the guards
    (path=fallback) -> honest 'Couldn't apply corrections safely' failure."""
    fake_clipboard.poll_results.append("teh text")
    fake_llm.corrected = "teh text"
    fake_llm.path = "fallback"
    o = orch()

    assert o.run_paste_flow() is False
    assert notifier.summaries[-1] == "Couldn't apply corrections safely"
    assert o.states == ["busy", "error"]
    assert "rejected by the guards; reporting as unsafe" in caplog.text


def test_ctrl_v_failure_restores_clipboard(orch, fake_clipboard,
                                           fake_keystroke, fake_llm, notifier,
                                           caplog):
    """paste: a ydotool failure during ctrl+v restores the clipboard and
    fails (the corrected text is lost — inherent to the crash point)."""
    fake_clipboard.poll_results.append("teh text")
    fake_keystroke.raise_on = {"ctrl_v"}
    o = orch()

    assert o.run_paste_flow() is False
    assert notifier.last == ("Proofreading failed",
                             "ydotool failed: scripted ctrl_v")
    assert fake_clipboard.calls[-1] == ("restore", fake_clipboard.saved_state)
    assert "paste flow failed" in caplog.text


def test_clipboard_save_crash_is_reported(orch, fake_clipboard, fake_llm,
                                          notifier, caplog):
    """paste: clipboard.save() blowing up (e.g. wl-clipboard missing) now
    notifies instead of escaping the flow, and releases the lock."""
    fake_clipboard.save_error = FileNotFoundError("wl-paste: not found")
    fake_clipboard.poll_results.append("teh text")
    o = orch()

    assert o.run_paste_flow() is False
    assert notifier.last == ("Proofreading failed", "wl-paste: not found")
    assert o.states == ["busy", "error"]
    # nothing to restore — save never returned a snapshot
    assert "restore" not in fake_clipboard.call_names
    assert "paste flow failed" in caplog.text

    # the lock was released: a second run goes through
    fake_clipboard.save_error = None
    assert o.run_paste_flow() is True

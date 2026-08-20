"""The proofread flows: copy -> LLM -> paste (in place), or primary-selection
-> LLM -> clipboard. Designed to run in a worker thread; UI updates go through
the on_state callback (marshalled to the GLib loop by the daemon).
"""

import logging
import threading
import time

from . import clipboard, history, keystroke
from .llm.client import LLMError, proofread_with_fallback
from .notify import Notifier

log = logging.getLogger(__name__)

# States reported to on_state (drive the tray icon)
IDLE = "idle"
BUSY = "busy"
ERROR = "error"


class Orchestrator:
    def __init__(self, cfg: dict, notifier: Notifier, on_state=None):
        self.cfg = cfg
        self.notifier = notifier
        self.on_state = on_state or (lambda state: None)
        self.enabled = True
        self.last_path = ""
        self.used_fallback = False
        self._flow_source = ""
        self._lock = threading.Lock()

    def set_config(self, cfg: dict) -> None:
        self.cfg = cfg

    # -- public entry points (called from DBus / tray / CLI) ----------------

    def run_paste_flow(self) -> bool:
        """Copy the selection, proofread, paste the correction in place."""
        if not self._begin():
            return False
        try:
            if self.cfg.get("paste_mode") == "clipboard-only":
                log.info("paste_mode=clipboard-only; using clipboard flow")
                return self._clipboard_flow_locked(degraded=False)
            if not keystroke.available():
                log.warning("ydotool unavailable; degrading to clipboard-only")
                return self._clipboard_flow_locked(degraded=True)
            return self._paste_flow_locked()
        finally:
            self._end()

    def run_clipboard_flow(self) -> bool:
        """Proofread the primary selection; leave the result on the clipboard."""
        if not self._begin():
            return False
        try:
            return self._clipboard_flow_locked(degraded=False)
        finally:
            self._end()

    # -- internals -----------------------------------------------------------

    def _begin(self) -> bool:
        if not self.enabled:
            log.info("trigger ignored: aiproof is disabled")
            self.notifier.notify("aiproof is disabled",
                                 "Enable it from the tray menu.")
            return False
        if not self._lock.acquire(blocking=False):
            log.info("trigger ignored: a run is already in progress")
            self.notifier.notify("Already proofreading",
                                 "Please wait for the current run to finish.")
            return False
        self.notifier.clear_replace_chain()
        self.on_state(BUSY)
        return True

    def _end(self) -> None:
        self._lock.release()

    def _fail(self, summary: str, body: str = "") -> bool:
        log.debug("flow failed: %s", summary)
        self.on_state(ERROR)
        self.notifier.notify(summary, body)
        return False

    def _done(self, summary: str, body: str = "") -> bool:
        log.debug("flow done: %s", summary)
        self.on_state(IDLE)
        self.notifier.notify(summary, body)
        return True

    def _proofread(self, text: str):
        """Returns (corrected, elapsed) or an error string."""
        self.last_path = ""
        self.used_fallback = False
        log.info(
            "captured selection: %d chars, %d lines, %d blank lines",
            len(text), text.count("\n") + 1,
            sum(1 for ln in text.split("\n") if not ln.strip()),
        )
        # Content only at DEBUG (run `aiproof -v daemon` to diagnose capture
        # mismatches without leaking selections into the default journal).
        log.debug("captured content: %r", text[:120])
        self.notifier.notify(
            "Proofreading…",
            f"{len(text):,} characters with "
            f"{self.cfg['provider']}/{self.cfg['model']}",
        )

        def on_fallback(error):
            fb = self.cfg.get("fallback") or {}
            self.notifier.notify(
                "Primary provider unavailable — trying fallback…",
                f"{error}\nFalling back to "
                f"{fb.get('provider', self.cfg['provider'])}/"
                f"{fb.get('model', '?')}",
            )

        try:
            corrected, elapsed, client, used_fb = proofread_with_fallback(
                self.cfg, text, on_fallback=on_fallback
            )
            self.last_path = client.last_path
            self.used_fallback = used_fb
            self._record_history(text, corrected, elapsed)
            return corrected, elapsed
        except LLMError as e:
            log.warning("proofread failed: %s", e)
            return str(e)
        except Exception as e:
            log.exception("unexpected proofread failure")
            return f"Unexpected error: {e}"

    def _record_history(self, original: str, corrected: str,
                        elapsed: float) -> None:
        fb = (self.cfg.get("fallback") or {}) if self.used_fallback else {}
        history.record(
            self.cfg,
            source=self._flow_source,
            original=original,
            corrected=corrected,
            elapsed=elapsed,
            path=self.last_path,
            provider=fb.get("provider") or self.cfg["provider"],
            model=fb.get("model") or self.cfg["model"],
            used_fallback=self.used_fallback,
        )

    def _path_note(self) -> str:
        note = ""
        if self.last_path in ("line-by-line", "segments"):
            note += f" — {self.last_path} mode"
        if self.used_fallback:
            fb = self.cfg.get("fallback") or {}
            note += f" — via fallback {fb.get('model', '')}".rstrip()
        return note

    def _report_unchanged(self, elapsed: float, text: str) -> bool:
        """Distinguish 'genuinely clean' from 'every correction was rejected
        by the safety guards' — reporting the latter as clean hides real
        errors from the user. Show WHAT was checked, so a capture mismatch
        (wrong selection, stale clipboard) is immediately visible."""
        if self.last_path == "fallback":
            log.warning("result unchanged because every correction was "
                        "rejected by the guards; reporting as unsafe")
            return self._fail(
                "Couldn't apply corrections safely",
                "The model kept restyling the text instead of minimally "
                "fixing it, so your original was left untouched. Try "
                "selecting a smaller piece, or a different model.",
            )
        log.info("no changes needed (path=%s, used_fallback=%s)",
                 self.last_path or "ok", self.used_fallback)
        snippet = text.replace("\n", " ⏎ ")
        if len(snippet) > 60:
            snippet = snippet[:57] + "…"
        return self._done(
            "No changes needed",
            f"Checked in {elapsed:.1f}s{self._path_note()} — "
            f"“{snippet}” is clean.",
        )

    def _paste_flow_locked(self) -> bool:
        self._flow_source = "paste"
        saved = None
        try:
            saved = clipboard.save()
            clipboard.clear()
            # Let the user's hotkey chord clear before typing, then neutralize
            # any modifiers still physically held.
            time.sleep(0.2)
            keystroke.release_modifiers()
            time.sleep(0.05)
            keystroke.ctrl_c()

            text = clipboard.poll_for_text(timeout=1.5)
            if text is None:
                log.info("clipboard still empty after ctrl+c; retrying once")
                keystroke.release_modifiers()
                keystroke.ctrl_c()
                text = clipboard.poll_for_text(timeout=1.5)
            if text is clipboard.NON_TEXT:
                log.info("selection is not text; restoring clipboard and aborting")
                clipboard.restore(saved)
                return self._fail("Selection is not text")
            if text is None or not text.strip():
                log.info("nothing copied after ctrl+c (empty or whitespace only); "
                         "aborting")
                clipboard.restore(saved)
                return self._fail(
                    "Select some text first",
                    "Nothing was copied — highlight the text you want proofread.",
                )

            result = self._proofread(text)
            if isinstance(result, str):
                clipboard.restore(saved)
                return self._fail("Proofreading failed", result)
            corrected, elapsed = result

            if corrected == text:
                clipboard.restore(saved)
                return self._report_unchanged(elapsed, text)

            clipboard.set_text(corrected)
            time.sleep(0.1)
            keystroke.release_modifiers()
            keystroke.ctrl_v()

            # Give the target app time to read the clipboard before restoring.
            time.sleep(self.cfg.get("restore_delay_ms", 500) / 1000)
            clipboard.restore(saved)
            log.info("correction pasted in place: %d -> %d chars%s",
                     len(text), len(corrected), self._path_note())
            return self._done(
                "Text corrected",
                f"Replaced in place in {elapsed:.1f}s{self._path_note()} "
                f"({self.cfg['provider']}/{self.cfg['model']}).",
            )
        except Exception as e:
            log.exception("paste flow failed")
            if saved is not None:
                clipboard.restore(saved)
            return self._fail("Proofreading failed", str(e))

    def _clipboard_flow_locked(self, degraded: bool) -> bool:
        self._flow_source = "clipboard"
        try:
            text = clipboard.get_primary_text()
            if not text or not text.strip():
                log.info("no primary selection; nothing to proofread")
                return self._fail(
                    "Select some text first",
                    "Highlight the text you want proofread, then try again.",
                )

            result = self._proofread(text)
            if isinstance(result, str):
                return self._fail("Proofreading failed", result)
            corrected, elapsed = result

            if corrected == text:
                return self._report_unchanged(elapsed, text)

            clipboard.set_text(corrected)
            log.info("correction copied to clipboard: %d -> %d chars%s "
                     "(degraded=%s)",
                     len(text), len(corrected), self._path_note(), degraded)
            note = "Corrected text copied — press Ctrl+V to paste " \
                   "(Ctrl+Shift+V in terminals)."
            if degraded:
                note += "\n(Automatic paste unavailable — ydotool not set up.)"
            return self._done(
                f"Corrected in {elapsed:.1f}s{self._path_note()}", note
            )
        except Exception as e:
            log.exception("clipboard flow failed")
            return self._fail("Proofreading failed", str(e))

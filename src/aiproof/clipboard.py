"""Wayland clipboard access via wl-clipboard (wl-copy / wl-paste).

Every subprocess call has a hard timeout: wl-paste can hang indefinitely when
the clipboard owner is unresponsive.
"""

import logging
import subprocess
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

_TIMEOUT = 2  # seconds, per wl-clipboard call

#: poll_for_text() result when the clipboard holds only non-text content
NON_TEXT = object()


@dataclass
class ClipboardState:
    kind: str  # "empty" | "text" | "image"
    mime: str = ""
    data: bytes = b""


def _run(args: list[str], input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    """Run wl-paste (output captured)."""
    return subprocess.run(
        args, input=input_bytes, capture_output=True, timeout=_TIMEOUT
    )


def _run_copy(args: list[str], input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    """Run wl-copy. Its forked child serves the clipboard and would keep
    captured pipes open forever, so send output to /dev/null instead."""
    return subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=_TIMEOUT,
    )


def list_types() -> list[str]:
    try:
        proc = _run(["wl-paste", "--list-types"])
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    return proc.stdout.decode("utf-8", "replace").split()


def _has_text_type(types: list[str]) -> bool:
    return any(
        t.startswith("text/") or t in ("UTF8_STRING", "STRING", "TEXT", "COMPOUND_TEXT")
        for t in types
    )


def save() -> ClipboardState:
    """Snapshot the clipboard (text or a single image) for later restore."""
    types = list_types()
    if not types:
        return ClipboardState("empty")
    if _has_text_type(types):
        try:
            proc = _run(["wl-paste", "--no-newline", "--type", "text"])
            if proc.returncode == 0:
                return ClipboardState("text", "text/plain;charset=utf-8", proc.stdout)
        except subprocess.TimeoutExpired:
            pass
        return ClipboardState("empty")
    image_mime = next((t for t in types if t.startswith("image/")), None)
    if image_mime:
        try:
            proc = _run(["wl-paste", "--type", image_mime])
            if proc.returncode == 0:
                return ClipboardState("image", image_mime, proc.stdout)
        except subprocess.TimeoutExpired:
            pass
    return ClipboardState("empty")


def restore(state: ClipboardState) -> None:
    try:
        if state.kind == "text":
            _run_copy(["wl-copy"], input_bytes=state.data)
        elif state.kind == "image":
            _run_copy(["wl-copy", "--type", state.mime], input_bytes=state.data)
        else:
            _run_copy(["wl-copy", "--clear"])
    except subprocess.TimeoutExpired:
        log.warning("clipboard restore timed out")


def clear() -> None:
    try:
        _run_copy(["wl-copy", "--clear"])
    except subprocess.TimeoutExpired:
        log.warning("clipboard clear timed out")


def set_text(text: str) -> None:
    _run_copy(["wl-copy"], input_bytes=text.encode("utf-8"))


def get_text() -> str | None:
    try:
        proc = _run(["wl-paste", "--no-newline", "--type", "text"])
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def get_primary_text() -> str | None:
    """Read the primary selection (highlighted text) without any keystrokes."""
    try:
        proc = _run(["wl-paste", "--primary", "--no-newline", "--type", "text"])
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace")


def poll_for_text(timeout: float = 1.5, interval: float = 0.1):
    """Wait for text to appear on the clipboard (after a synthetic Ctrl+C).

    Returns the text, NON_TEXT if only non-text content appeared, or None on
    timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = get_text()
        if text:
            return text
        types = list_types()
        if types and not _has_text_type(types):
            return NON_TEXT
        time.sleep(interval)
    return None

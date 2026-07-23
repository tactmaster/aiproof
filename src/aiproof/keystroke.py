"""Synthetic keystrokes via ydotool (uinput) — the only way to inject keys
into arbitrary apps on GNOME/Mutter Wayland (no virtual-keyboard protocol).

Requires ydotoold running as the user (systemd user unit `ydotool.service`,
shipped by Ubuntu's ydotool package) and /dev/uinput access (udev rule grants
group `input`).
"""

import logging
import os
import shutil
import subprocess

log = logging.getLogger(__name__)

# Linux input-event-codes
KEY_LEFTCTRL = 29
KEY_C = 46
KEY_V = 47
_MODIFIERS = (29, 97, 125, 126, 42, 54, 56, 100)  # L/R ctrl, meta, shift, alt


def socket_path() -> str:
    if os.environ.get("YDOTOOL_SOCKET"):
        return os.environ["YDOTOOL_SOCKET"]
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return f"{runtime}/.ydotool_socket"


def available() -> bool:
    return shutil.which("ydotool") is not None and os.path.exists(socket_path())


def _key(*events: str) -> None:
    env = dict(os.environ, YDOTOOL_SOCKET=socket_path())
    proc = subprocess.run(
        ["ydotool", "key", *events],
        env=env, capture_output=True, timeout=3,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ydotool failed: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )


def release_modifiers() -> None:
    """Release every modifier key. The user is often still holding the hotkey
    chord when we type; synthetic release events neutralize held modifiers so
    our Ctrl+C isn't turned into Ctrl+Super+C. Harmless for unpressed keys."""
    _key(*[f"{code}:0" for code in _MODIFIERS])


def ctrl_c() -> None:
    _key(f"{KEY_LEFTCTRL}:1", f"{KEY_C}:1", f"{KEY_C}:0", f"{KEY_LEFTCTRL}:0")


def ctrl_v() -> None:
    _key(f"{KEY_LEFTCTRL}:1", f"{KEY_V}:1", f"{KEY_V}:0", f"{KEY_LEFTCTRL}:0")

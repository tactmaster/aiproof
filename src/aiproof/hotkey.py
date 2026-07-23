"""GNOME global hotkeys via gsettings custom keybindings
(org.gnome.settings-daemon.plugins.media-keys). Zero-interaction and
persistent; the binding invokes the daemon over DBus with gdbus for a fast
(~30 ms) trigger.
"""

import logging

from gi.repository import Gio

from . import DBUS_IFACE, DBUS_NAME, DBUS_PATH

log = logging.getLogger(__name__)

MEDIA_KEYS_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
KEYBINDING_SCHEMA = MEDIA_KEYS_SCHEMA + ".custom-keybinding"
BASE_PATH = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/"

PATH_TRIGGER = BASE_PATH + "aiproof/"
PATH_CLIPBOARD = BASE_PATH + "aiproof-clipboard/"


def _gdbus_command(method: str) -> str:
    return (
        f"gdbus call -e -d {DBUS_NAME} -o {DBUS_PATH} -m {DBUS_IFACE}.{method}"
    )


def _write_binding(path: str, name: str, command: str, binding: str) -> None:
    settings = Gio.Settings.new_with_path(KEYBINDING_SCHEMA, path)
    settings.set_string("name", name)
    settings.set_string("command", command)
    settings.set_string("binding", binding)
    settings.sync()


def register(cfg: dict) -> None:
    """Idempotently register (or update) both hotkeys."""
    media = Gio.Settings.new(MEDIA_KEYS_SCHEMA)
    paths = list(media.get_strv("custom-keybindings"))

    conflicts = _find_conflicts(cfg, paths)
    if conflicts:
        log.warning("hotkey conflicts with existing bindings: %s", conflicts)

    changed = False
    for path in (PATH_TRIGGER, PATH_CLIPBOARD):
        if path not in paths:
            paths.append(path)
            changed = True
    if changed:
        media.set_strv("custom-keybindings", paths)
        media.sync()

    _write_binding(
        PATH_TRIGGER,
        "AI proofread selection (aiproof)",
        _gdbus_command("Trigger"),
        cfg["hotkey"],
    )
    _write_binding(
        PATH_CLIPBOARD,
        "AI proofread to clipboard (aiproof)",
        _gdbus_command("TriggerClipboardOnly"),
        cfg["hotkey_clipboard"],
    )
    log.info("hotkeys registered: %s, %s", cfg["hotkey"], cfg["hotkey_clipboard"])


def unregister() -> None:
    media = Gio.Settings.new(MEDIA_KEYS_SCHEMA)
    paths = [
        p for p in media.get_strv("custom-keybindings")
        if p not in (PATH_TRIGGER, PATH_CLIPBOARD)
    ]
    media.set_strv("custom-keybindings", paths)
    for path in (PATH_TRIGGER, PATH_CLIPBOARD):
        settings = Gio.Settings.new_with_path(KEYBINDING_SCHEMA, path)
        for key in ("name", "command", "binding"):
            settings.reset(key)
    media.sync()


def is_registered() -> bool:
    media = Gio.Settings.new(MEDIA_KEYS_SCHEMA)
    paths = media.get_strv("custom-keybindings")
    if PATH_TRIGGER not in paths:
        return False
    settings = Gio.Settings.new_with_path(KEYBINDING_SCHEMA, PATH_TRIGGER)
    return settings.get_string("binding") != ""


def _find_conflicts(cfg: dict, paths: list[str]) -> list[str]:
    """Other custom keybindings already using our accelerators."""
    ours = {cfg["hotkey"], cfg["hotkey_clipboard"]}
    conflicts = []
    for path in paths:
        if path in (PATH_TRIGGER, PATH_CLIPBOARD):
            continue
        try:
            other = Gio.Settings.new_with_path(KEYBINDING_SCHEMA, path)
            if other.get_string("binding") in ours:
                conflicts.append(other.get_string("name") or path)
        except Exception:
            continue
    return conflicts

"""Per-application proofreading profiles.

A Slack message, a Zendesk customer reply, and a commit message deserve
different treatment: casual chat keeps its lowercase starts and slang, a
support reply gets full capitalization and punctuation, editor text keeps its
identifiers alone. The focused window (via the focused-window-dbus GNOME
extension) tells us which world the text lives in at trigger time — one
~20 ms DBus call, no extra LLM cost.

Detection is best-effort: no extension, no match, or the flag off all mean
"default profile", which is exactly today's behavior.
"""

import json
import logging
import re

log = logging.getLogger(__name__)

_FOCUSED_WINDOW_BUS = "org.gnome.Shell"
_FOCUSED_WINDOW_PATH = "/org/gnome/shell/extensions/FocusedWindow"
_FOCUSED_WINDOW_IFACE = "org.gnome.shell.extensions.FocusedWindow"

# Order matters: first match wins. Title rules catch browser tabs and PWAs
# (WhatsApp Web, Zendesk); wm_class rules catch native apps.
PROFILES = [
    {
        "id": "slack",
        "wm_class": re.compile(r"slack", re.IGNORECASE),
        "title": re.compile(r"\bslack\b", re.IGNORECASE),
        "casual": True,
        "setting": (
            "The text is a Slack message (casual team chat). This OVERRIDES "
            "wording rule 3: do NOT capitalize a sentence that the writer "
            "started lowercase, and do NOT add a full stop the writer did "
            "not type. Slang, abbreviations, and emoji are intentional. Fix "
            "only genuine misspellings and broken grammar. (\"i\" still "
            "becomes \"I\".)"
        ),
    },
    {
        "id": "whatsapp",
        "wm_class": re.compile(r"whatsapp", re.IGNORECASE),
        "title": re.compile(r"whatsapp", re.IGNORECASE),
        "casual": True,
        "setting": (
            "The text is a WhatsApp message (personal chat). This OVERRIDES "
            "wording rule 3: do NOT capitalize a sentence that the writer "
            "started lowercase, and do NOT add a full stop the writer did "
            "not type. Informal style and slang are intentional. Fix only "
            "genuine misspellings. (\"i\" still becomes \"I\".)"
        ),
    },
    {
        "id": "support",
        "wm_class": re.compile(r"zendesk", re.IGNORECASE),
        "title": re.compile(r"zendesk|freshdesk|intercom", re.IGNORECASE),
        "casual": False,
        "setting": (
            "The text is a customer support reply. It must read "
            "professionally: fix capitalization and end-of-sentence "
            "punctuation as well as spelling and grammar — but still do "
            "not rephrase or reword."
        ),
    },
    {
        "id": "email",
        "wm_class": re.compile(r"thunderbird|evolution|geary", re.IGNORECASE),
        "title": re.compile(r"\bgmail\b|\boutlook\b", re.IGNORECASE),
        "casual": False,
        "setting": (
            "The text is an email. It should read professionally: fix "
            "capitalization and punctuation as well as spelling and "
            "grammar — but still do not rephrase or reword."
        ),
    },
    {
        "id": "technical",
        "wm_class": re.compile(
            r"^code$|jetbrains|terminal|ptyxis|console|kitty|alacritty",
            re.IGNORECASE,
        ),
        "title": re.compile(r"$^"),  # wm_class only — titles are too noisy
        "casual": True,
        "setting": (
            "The text comes from a code editor or terminal (likely a commit "
            "message, code comment, or command). Identifiers, flags, paths, "
            "and lowercase conventions are intentional — never alter them, "
            "and do not capitalize or add punctuation the writer omitted. "
            "Fix only clear prose typos."
        ),
    },
]


def get_focused_app() -> "tuple[str, str]":
    """(wm_class, title) of the focused window, or ("", "") when the
    focused-window-dbus extension isn't available."""
    try:
        from gi.repository import Gio, GLib

        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        reply = bus.call_sync(
            _FOCUSED_WINDOW_BUS, _FOCUSED_WINDOW_PATH, _FOCUSED_WINDOW_IFACE,
            "Get", None, GLib.VariantType("(s)"),
            Gio.DBusCallFlags.NONE, 1000, None,
        )
        info = json.loads(reply.unpack()[0])
        return info.get("wm_class", "") or "", info.get("title", "") or ""
    except Exception as e:
        log.debug("focused-window lookup unavailable: %s", e)
        return "", ""


def match_profile(wm_class: str, title: str) -> "dict | None":
    for profile in PROFILES:
        if (wm_class and profile["wm_class"].search(wm_class)) or (
            title and profile["title"].search(title)
        ):
            return profile
    return None


def profile_by_id(profile_id: str) -> "dict | None":
    for profile in PROFILES:
        if profile["id"] == profile_id:
            return profile
    return None


def get_profile(cfg: dict) -> "dict | None":
    """The profile for the currently focused app, honoring the config flag.
    None means: behave exactly as before."""
    if not cfg.get("app_profiles", True):
        return None
    wm_class, title = get_focused_app()
    profile = match_profile(wm_class, title)
    if profile:
        log.info("app profile: %s (wm_class=%r)", profile["id"], wm_class)
    else:
        log.debug("no app profile for wm_class=%r title=%r", wm_class, title)
    return profile

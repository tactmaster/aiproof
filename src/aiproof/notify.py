"""Desktop notifications via org.freedesktop.Notifications (Gio DBus).

A Notifier keeps the last notification id and replaces it, so the
progress -> result sequence mutates a single bubble instead of stacking.
Falls back to notify-send if the DBus call fails.
"""

import logging
import subprocess

from . import APP_ID

log = logging.getLogger(__name__)


class Notifier:
    def __init__(self):
        self._last_id = 0
        self._proxy = None

    def _get_proxy(self):
        if self._proxy is None:
            from gi.repository import Gio

            self._proxy = Gio.DBusProxy.new_for_bus_sync(
                Gio.BusType.SESSION,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.Notifications",
                "/org/freedesktop/Notifications",
                "org.freedesktop.Notifications",
                None,
            )
        return self._proxy

    def notify(self, summary: str, body: str = "", *, replace: bool = True,
               transient: bool = True, icon: str = APP_ID) -> None:
        try:
            from gi.repository import GLib

            proxy = self._get_proxy()
            hints = {"transient": GLib.Variant("b", transient)}
            params = GLib.Variant(
                "(susssasa{sv}i)",
                (
                    "aiproof",
                    self._last_id if replace else 0,
                    icon,
                    summary,
                    body,
                    [],
                    hints,
                    -1,
                ),
            )
            result = proxy.call_sync(
                "Notify", params, 0, 2000, None
            )
            self._last_id = result.unpack()[0]
        except Exception:
            log.debug("DBus notification failed, falling back to notify-send",
                      exc_info=True)
            try:
                subprocess.run(
                    ["notify-send", "--app-name=aiproof", summary, body],
                    timeout=2, capture_output=True,
                )
            except Exception:
                log.warning("notify-send fallback failed too")

    def clear_replace_chain(self) -> None:
        """Start a fresh bubble for the next notify()."""
        self._last_id = 0

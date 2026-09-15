"""The aiproof daemon: GLib/GTK3 main loop, single-instance DBus service,
tray icon, global hotkey registration, config live-reload.
"""

import logging
import shutil
import subprocess
import sys
import threading

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gio, GLib, Gtk

from . import DBUS_NAME, config, hotkey, keystroke
from .dbus_service import DBusService
from .notify import Notifier
from .orchestrator import Orchestrator
from .tray import Tray

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self):
        self.cfg = config.load()
        self.notifier = Notifier()
        self.state = "idle"
        self.orchestrator = Orchestrator(
            self.cfg, self.notifier, on_state=self._on_state_from_worker
        )
        self.tray = None
        self._dbus = DBusService(self)
        self._config_monitor = None
        self._acquired = False

    # -- lifecycle ------------------------------------------------------------

    def run(self) -> int:
        Gio.bus_own_name(
            Gio.BusType.SESSION,
            DBUS_NAME,
            Gio.BusNameOwnerFlags.DO_NOT_QUEUE,
            self._on_bus_acquired,
            self._on_name_acquired,
            self._on_name_lost,
        )
        Gtk.main()
        return 0

    def quit(self) -> None:
        Gtk.main_quit()

    def _on_bus_acquired(self, connection, name) -> None:
        self._dbus.register(connection)

    def _on_name_acquired(self, connection, name) -> None:
        self._acquired = True
        log.info("aiproof daemon started (bus name %s)", name)
        self.tray = Tray(self)
        self._register_hotkeys()
        self._watch_config()
        threading.Thread(target=self._startup_checks, daemon=True).start()

    def _on_name_lost(self, connection, name) -> None:
        if not self._acquired:
            print("aiproof daemon is already running — exiting.", file=sys.stderr)
            self.quit()
        else:
            log.error("lost DBus name %s — exiting", name)
            self.quit()

    # -- actions (called from tray/DBus, main thread) --------------------------

    def trigger_async(self, clipboard_only: bool = False) -> None:
        target = (
            self.orchestrator.run_clipboard_flow
            if clipboard_only
            else self.orchestrator.run_paste_flow
        )
        threading.Thread(target=target, daemon=True).start()

    def set_enabled(self, enabled: bool) -> None:
        self.orchestrator.enabled = enabled
        self.state = "idle" if enabled else "disabled"
        if self.tray:
            self.tray.set_state(self.state)
            self.tray.sync_enabled(enabled)

    def show_settings(self) -> None:
        exe = shutil.which("aiproof")
        cmd = [exe, "settings"] if exe else [sys.executable, "-m", "aiproof.cli",
                                             "settings"]
        subprocess.Popen(cmd)

    # -- internals --------------------------------------------------------------

    def _on_state_from_worker(self, state: str) -> None:
        def apply():
            if not self.orchestrator.enabled:
                state_now = "disabled"
            else:
                state_now = state
            self.state = state_now
            if self.tray:
                self.tray.set_state(state_now)
            if state == "error":
                # fall back to idle after a few seconds
                GLib.timeout_add_seconds(5, self._error_to_idle)
            return False

        GLib.idle_add(apply)

    def _error_to_idle(self) -> bool:
        if self.state == "error":
            self.state = "idle"
            if self.tray:
                self.tray.set_state("idle")
        return False  # don't repeat

    def _register_hotkeys(self) -> None:
        try:
            hotkey.register(self.cfg)
        except Exception:
            log.exception("hotkey registration failed")
            self.notifier.notify(
                "aiproof: hotkey registration failed",
                "Set a shortcut manually in GNOME Settings → Keyboard.",
            )

    def _watch_config(self) -> None:
        cfg_file = Gio.File.new_for_path(str(config.config_path()))
        self._config_monitor = cfg_file.monitor_file(Gio.FileMonitorFlags.NONE, None)
        self._config_monitor.connect("changed", self._on_config_changed)

    def _on_config_changed(self, monitor, f, other, event) -> None:
        if event not in (
            Gio.FileMonitorEvent.CHANGES_DONE_HINT,
            Gio.FileMonitorEvent.CREATED,
        ):
            return
        log.info("config changed — reloading")
        old = self.cfg
        self.cfg = config.load()
        self.orchestrator.set_config(self.cfg)
        if (
            old["hotkey"] != self.cfg["hotkey"]
            or old["hotkey_clipboard"] != self.cfg["hotkey_clipboard"]
        ):
            self._register_hotkeys()

    def _startup_checks(self) -> None:
        # Already on a background thread: catch up the learned-context
        # profile (including the LLM domain summary when due).
        if self.cfg.get("context_aware") and self.cfg.get("save_history"):
            from . import context
            context.refresh_if_stale(self.cfg, allow_summary=True)
        if not keystroke.available():
            self.notifier.notify(
                "aiproof: automatic paste unavailable",
                "ydotool is not set up — falling back to clipboard-only mode.\n"
                "Run 'aiproof setup' in a terminal to fix this.",
                replace=False,
            )
        if not self.cfg.get("setup_complete"):
            from .setup_check import run_all

            results = run_all(self.cfg)
            if all(ok for _, ok, _ in results):
                config.update(setup_complete=True)
            else:
                failed = [title for title, ok, _ in results if not ok]
                self.notifier.notify(
                    "aiproof needs setup",
                    "Issues: " + ", ".join(failed) +
                    ".\nRun 'aiproof setup' in a terminal for guided fixes.",
                    replace=False, transient=False,
                )


def run_daemon() -> int:
    return Daemon().run()

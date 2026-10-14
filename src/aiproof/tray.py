"""Tray icon via AyatanaAppIndicator3 (StatusNotifierItem). Requires GTK3 in
this process — which is why the settings window lives in a separate GTK4
process. Degrades to no-tray if the typelib is missing.
"""

import importlib.resources
import logging

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk

log = logging.getLogger(__name__)

try:
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator
    HAVE_APPINDICATOR = True
except (ValueError, ImportError):
    HAVE_APPINDICATOR = False
    log.warning("AyatanaAppIndicator3 not available — no tray icon")


def icon_dir() -> str:
    return str(importlib.resources.files("aiproof") / "data" / "icons")


BUSY_FRAMES = [f"aiproof-busy-{i}-symbolic" for i in range(1, 9)]
SPIN_INTERVAL_MS = 125   # 8 frames -> one rotation per second


class Tray:
    def __init__(self, daemon):
        self.daemon = daemon
        self.indicator = None
        self._spin_id = None
        self._spin_frame = 0
        self._enabled_item = None
        if not HAVE_APPINDICATOR:
            return
        self.indicator = AppIndicator.Indicator.new_with_path(
            "aiproof",
            "aiproof-idle-symbolic",
            AppIndicator.IndicatorCategory.APPLICATION_STATUS,
            icon_dir(),
        )
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self.indicator.set_title("AI Proofreader")
        self.indicator.set_menu(self._build_menu())

    def set_state(self, state: str) -> None:
        """state: idle | busy | error | disabled

        Busy spins the tray icon instead of raising a notification banner:
        a proofread takes a couple of seconds, and a popup for every one of
        them is more interruption than information."""
        if not self.indicator:
            return
        self._stop_spinner()
        if state == "busy":
            self._start_spinner()
            return
        self.indicator.set_icon_full(f"aiproof-{state}-symbolic", state)

    def _start_spinner(self) -> None:
        self._spin_frame = 0

        def tick() -> bool:
            if self._spin_id is None:
                return False
            name = BUSY_FRAMES[self._spin_frame % len(BUSY_FRAMES)]
            self.indicator.set_icon_full(name, "working")
            self._spin_frame += 1
            return True

        tick()
        self._spin_id = GLib.timeout_add(SPIN_INTERVAL_MS, tick)

    def _stop_spinner(self) -> None:
        if getattr(self, "_spin_id", None) is not None:
            GLib.source_remove(self._spin_id)
            self._spin_id = None

    def sync_enabled(self, enabled: bool) -> None:
        if self._enabled_item:
            self._enabled_item.handler_block_by_func(self._on_enabled_toggled)
            self._enabled_item.set_active(enabled)
            self._enabled_item.handler_unblock_by_func(self._on_enabled_toggled)

    # -- menu ----------------------------------------------------------------

    def _build_menu(self) -> Gtk.Menu:
        menu = Gtk.Menu()

        item = Gtk.MenuItem(label="Proofread selection now")
        item.connect("activate", lambda *_: self.daemon.trigger_async())
        menu.append(item)

        item = Gtk.MenuItem(label="Proofread selection to clipboard")
        item.connect("activate",
                     lambda *_: self.daemon.trigger_async(clipboard_only=True))
        menu.append(item)

        menu.append(Gtk.SeparatorMenuItem())

        self._enabled_item = Gtk.CheckMenuItem(label="Enabled")
        self._enabled_item.set_active(True)
        self._enabled_item.connect("toggled", self._on_enabled_toggled)
        menu.append(self._enabled_item)

        item = Gtk.MenuItem(label="Settings…")
        item.connect("activate", lambda *_: self.daemon.show_settings())
        menu.append(item)

        item = Gtk.MenuItem(label="About")
        item.connect("activate", self._on_about)
        menu.append(item)

        menu.append(Gtk.SeparatorMenuItem())

        item = Gtk.MenuItem(label="Quit")
        item.connect("activate", lambda *_: self.daemon.quit())
        menu.append(item)

        menu.show_all()
        return menu

    def _on_enabled_toggled(self, item) -> None:
        self.daemon.set_enabled(item.get_active())

    def _on_about(self, _item) -> None:
        from . import __version__

        dialog = Gtk.AboutDialog(
            program_name="AI Proofreader",
            version=__version__,
            comments=(
                "OS-wide AI text proofreader.\n"
                "Select text anywhere, press the hotkey, get it corrected "
                "in place."
            ),
            website="https://github.com/tactmaster/aiproof",
            logo_icon_name="io.github.edwatson.aiproof",
            license_type=Gtk.License.GPL_3_0,
        )
        dialog.connect("response", lambda d, r: d.destroy())
        dialog.present()

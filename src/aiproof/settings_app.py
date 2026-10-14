"""Settings window: GTK4 + libadwaita, runs as its own process (the daemon is
GTK3 because of AppIndicator; the two toolkits can't share a process).
Settings are written straight to config.json — the daemon live-reloads it.
"""

import logging
import threading

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from . import APP_ID, config, secrets
from .llm.providers import PROVIDERS

log = logging.getLogger(__name__)

PROVIDER_IDS = list(PROVIDERS.keys())


class SettingsWindow(Adw.PreferencesWindow):
    def __init__(self, app, page: str | None = None):
        super().__init__(application=app, title="AI Proofreader Settings")
        self.set_default_size(640, 640)
        self.set_search_enabled(False)
        self.cfg = config.load()
        self._loading = True

        self.add(self._build_provider_page())
        self.add(self._build_behavior_page())
        self.add(self._build_setup_page())
        self._load_values()
        self._loading = False
        self.connect("close-request", self._flush_api_key)

        if page == "behavior":
            self.set_visible_page_name("behavior")
        elif page == "setup":
            self.set_visible_page_name("setup")

    # -- provider page --------------------------------------------------------

    def _build_provider_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(
            title="Provider", icon_name="network-server-symbolic", name="provider"
        )
        group = Adw.PreferencesGroup(title="AI Provider")

        self.provider_row = Adw.ComboRow(
            title="Provider",
            model=Gtk.StringList.new(
                [PROVIDERS[pid]["display_name"] for pid in PROVIDER_IDS]
            ),
        )
        self.provider_row.connect("notify::selected", self._on_provider_changed)
        group.add(self.provider_row)

        self.endpoint_row = Adw.EntryRow(title="Endpoint URL")
        self.endpoint_row.connect("changed", self._save_text_row)
        group.add(self.endpoint_row)

        self.model_row = Adw.EntryRow(title="Model")
        self.model_row.connect("changed", self._save_text_row)
        # Model names (phi3.5:latest, gpt-4o-mini, ...) aren't dictionary
        # words — without this GNOME's inline spellcheck underlines them.
        delegate = self.model_row.get_delegate()
        if delegate is not None:
            delegate.set_input_hints(Gtk.InputHints.NO_SPELLCHECK)
        self.model_popover = Gtk.Popover()
        list_btn = Gtk.MenuButton(
            label="List", valign=Gtk.Align.CENTER, popover=self.model_popover
        )
        list_btn.connect("notify::active", self._on_list_menu_toggled)
        self.model_row.add_suffix(list_btn)
        group.add(self.model_row)

        self.api_key_row = Adw.PasswordEntryRow(
            title="API key (press Enter to save)", show_apply_button=True)
        self.api_key_row.connect("apply", self._on_api_key_apply)
        focus = Gtk.EventControllerFocus()
        focus.connect("leave", self._flush_api_key)
        self.api_key_row.add_controller(focus)
        group.add(self.api_key_row)

        # Whether the selected provider keeps text on this machine, and for
        # the Claude CLI whether the binary is actually installed.
        self.provider_note_row = Adw.ActionRow(title="", subtitle="")
        group.add(self.provider_note_row)

        page.add(group)

        fb_group = Adw.PreferencesGroup(
            title="Fallback provider",
            description="Used automatically when the provider above is "
                        "unreachable or fails",
        )
        self.fb_enabled_row = Adw.SwitchRow(title="Enable fallback")
        self.fb_enabled_row.connect("notify::active", self._save_fallback)
        fb_group.add(self.fb_enabled_row)

        self.fb_provider_row = Adw.ComboRow(
            title="Fallback provider",
            model=Gtk.StringList.new(
                [PROVIDERS[pid]["display_name"] for pid in PROVIDER_IDS]
            ),
        )
        self.fb_provider_row.connect("notify::selected", self._on_fb_provider_changed)
        fb_group.add(self.fb_provider_row)

        self.fb_endpoint_row = Adw.EntryRow(title="Fallback endpoint URL")
        self.fb_endpoint_row.connect("changed", self._save_fallback)
        fb_group.add(self.fb_endpoint_row)

        self.fb_model_row = Adw.EntryRow(title="Fallback model")
        self.fb_model_row.connect("changed", self._save_fallback)
        fb_group.add(self.fb_model_row)
        page.add(fb_group)

        test_group = Adw.PreferencesGroup()
        self.test_row = Adw.ActionRow(
            title="Test connection", subtitle="Sends a tiny prompt to the model"
        )
        test_btn = Gtk.Button(label="Test", valign=Gtk.Align.CENTER)
        test_btn.connect("clicked", self._on_test_connection)
        self.test_row.add_suffix(test_btn)
        test_group.add(self.test_row)
        page.add(test_group)

        if not secrets.keyring_available():
            warn = Adw.PreferencesGroup()
            warn.add(Adw.ActionRow(
                title="⚠ Keyring unavailable",
                subtitle="API keys will be stored as plain text in config.json",
            ))
            page.add(warn)

        return page

    def _update_provider_note(self, provider_id: str) -> None:
        """Say plainly where the text goes, and flag a missing Claude CLI."""
        from .llm.client import find_claude_cli

        provider = PROVIDERS.get(provider_id, {})
        fmt = provider.get("format", "")
        endpoint = self.endpoint_row.get_text().strip()
        if fmt == "claude_cli":
            binary = find_claude_cli(endpoint)
            if binary:
                title = "Using the Claude Code CLI on this machine"
                subtitle = (f"{binary} — no API key needed (your Claude "
                            "subscription). Text is sent to Anthropic.")
            else:
                title = "⚠ Claude CLI not found"
                subtitle = ("Install Claude Code, or put the full path to the "
                            "'claude' binary in the endpoint field above.")
        elif fmt == "ollama" or "127.0.0.1" in endpoint or "localhost" in endpoint:
            title = "Local provider"
            subtitle = "Text stays on this machine."
        else:
            title = "Cloud provider"
            subtitle = "Text is sent to the provider over the internet."
        self.provider_note_row.set_title(title)
        self.provider_note_row.set_subtitle(subtitle)

    def _on_provider_changed(self, row, _pspec) -> None:
        if self._loading:
            return
        old_pid = self.cfg.get("provider")
        pid = PROVIDER_IDS[row.get_selected()]
        preset = PROVIDERS[pid]

        overrides = dict(self.cfg.get("provider_overrides") or {})
        if old_pid and old_pid != pid:
            overrides[old_pid] = {
                "model": self.model_row.get_text().strip(),
                "endpoint": self.endpoint_row.get_text().strip(),
            }
        remembered = overrides.get(pid) or {}
        endpoint = remembered.get("endpoint") or preset["endpoint"]
        model = remembered.get("model") or preset["model"]

        self._loading = True
        self.endpoint_row.set_text(endpoint)
        self.model_row.set_text(model)
        self.api_key_row.set_text(secrets.get_api_key(pid) or "")
        self._loading = False
        config.update(
            provider=pid, endpoint=endpoint, model=model, provider_overrides=overrides
        )
        self.cfg = config.load()
        self._update_provider_note(pid)

    def _save_text_row(self, _row) -> None:
        if self._loading:
            return
        config.update(
            endpoint=self.endpoint_row.get_text().strip(),
            model=self.model_row.get_text().strip(),
        )

    def _flush_api_key(self, *_args) -> bool:
        """Commit a typed-but-not-applied key. Returns False so it can also
        serve as a ::close-request handler (False = let the window close)."""
        if self._loading:
            return False
        pid = PROVIDER_IDS[self.provider_row.get_selected()]
        typed = self.api_key_row.get_text().strip()
        if typed and typed != (secrets.get_api_key(pid) or ""):
            self._on_api_key_apply(self.api_key_row)
        return False

    def _on_api_key_apply(self, row) -> None:
        pid = PROVIDER_IDS[self.provider_row.get_selected()]
        key = row.get_text().strip()
        if secrets.set_api_key(pid, key):
            config.update(api_key_plaintext=None)
            self._flash(row, "Saved to keyring")
        else:
            config.update(api_key_plaintext=key or None)
            self._flash(row, "Keyring unavailable — saved as plain text")

    def _on_fb_provider_changed(self, *_args) -> None:
        """Repopulate the fallback endpoint/model with the new provider's
        defaults. Without this the fields kept the previous provider's values
        and produced impossible pairings (claude-cli with an Ollama URL)."""
        if self._loading:
            return
        preset = PROVIDERS[PROVIDER_IDS[self.fb_provider_row.get_selected()]]
        self._loading = True
        self.fb_endpoint_row.set_text(preset["endpoint"])
        self.fb_model_row.set_text(preset["model"])
        self._loading = False
        self._save_fallback()

    def _save_fallback(self, *_args) -> None:
        if self._loading:
            return
        if not self.fb_enabled_row.get_active():
            config.update(fallback=None)
            return
        pid = PROVIDER_IDS[self.fb_provider_row.get_selected()]
        config.update(fallback={
            "provider": pid,
            "endpoint": self.fb_endpoint_row.get_text().strip()
            or PROVIDERS[pid]["endpoint"],
            "model": self.fb_model_row.get_text().strip()
            or PROVIDERS[pid]["model"],
        })

    def _on_list_menu_toggled(self, button, _pspec) -> None:
        if button.get_active():
            self._on_list_models()

    def _on_list_models(self) -> None:
        from .llm.client import client_from_config

        spinner = Gtk.Spinner(spinning=True)
        spinner.set_margin_top(12)
        spinner.set_margin_bottom(12)
        spinner.set_margin_start(12)
        spinner.set_margin_end(12)
        self.model_popover.set_child(spinner)

        def work():
            error = None
            try:
                models = client_from_config(config.load()).list_models()
            except Exception as e:
                models, error = [], str(e)
            GLib.idle_add(self._show_model_list, models, error)

        threading.Thread(target=work, daemon=True).start()

    def _show_model_list(self, models: list[str], error: str | None) -> None:
        if error:
            label = Gtk.Label(label=error, wrap=True, max_width_chars=40)
            label.set_margin_top(12)
            label.set_margin_bottom(12)
            label.set_margin_start(12)
            label.set_margin_end(12)
            self.model_popover.set_child(label)
            return
        if not models:
            label = Gtk.Label(label="No models found")
            label.set_margin_top(12)
            label.set_margin_bottom(12)
            label.set_margin_start(12)
            label.set_margin_end(12)
            self.model_popover.set_child(label)
            return
        listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        listbox.add_css_class("boxed-list")
        for name in models:
            row = Adw.ActionRow(title=name, activatable=True)
            row.connect("activated", self._on_model_selected, name)
            listbox.append(row)
        scroller = Gtk.ScrolledWindow(
            min_content_height=150, max_content_height=300, propagate_natural_width=True
        )
        scroller.set_child(listbox)
        self.model_popover.set_child(scroller)

    def _on_model_selected(self, row, name: str) -> None:
        self.model_row.set_text(name)
        self.model_popover.popdown()

    def _on_test_connection(self, button) -> None:
        from .llm.client import client_from_config

        self.test_row.set_subtitle("Testing…")

        def work():
            ok, msg = client_from_config(config.load()).test_connection()
            GLib.idle_add(self.test_row.set_subtitle, ("✔ " if ok else "✘ ") + msg)

        threading.Thread(target=work, daemon=True).start()

    # -- behavior page --------------------------------------------------------

    def _build_behavior_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(
            title="Behavior", icon_name="preferences-system-symbolic", name="behavior"
        )

        keys = Adw.PreferencesGroup(
            title="Hotkeys",
            description="GNOME accelerator syntax, e.g. &lt;Control&gt;&lt;Super&gt;p",
        )
        self.hotkey_row = Adw.EntryRow(title="Proofread selection in place")
        self.hotkey_row.connect("changed", self._save_hotkeys)
        keys.add(self.hotkey_row)
        self.hotkey_clip_row = Adw.EntryRow(title="Proofread to clipboard")
        self.hotkey_clip_row.connect("changed", self._save_hotkeys)
        keys.add(self.hotkey_clip_row)
        page.add(keys)

        flow = Adw.PreferencesGroup(title="Replace behavior")
        self.paste_mode_row = Adw.ComboRow(
            title="Mode",
            subtitle="Clipboard-only never types into the focused app",
            model=Gtk.StringList.new(
                ["Auto — replace selection in place", "Clipboard only"]
            ),
        )
        self.paste_mode_row.connect("notify::selected", self._save_behavior)
        flow.add(self.paste_mode_row)

        self.delay_row = Adw.SpinRow.new_with_range(100, 3000, 50)
        self.delay_row.set_title("Clipboard restore delay (ms)")
        self.delay_row.set_subtitle(
            "How long the corrected text stays on the clipboard after pasting"
        )
        self.delay_row.connect("notify::value", self._save_behavior)
        flow.add(self.delay_row)
        page.add(flow)

        output = Adw.PreferencesGroup(
            title="Correction output",
            description="How the model reports corrections. The edit list is "
                        "much faster on small local models and can never "
                        "break line breaks or bullets.",
        )
        self.output_mode_row = Adw.ComboRow(
            title="Model output",
            model=Gtk.StringList.new(
                ["Full rewrite of the text", "Edit list (wrong => right)"]
            ),
        )
        self.output_mode_row.connect("notify::selected", self._save_output_mode)
        output.add(self.output_mode_row)
        self.prompt_style_row = Adw.ComboRow(
            title="Prompt style",
            subtitle="Minimal: bare text plus a one-line system prompt, for a "
                     "model fine-tuned on the edit-list task",
            model=Gtk.StringList.new(
                ["Full instructions", "Minimal (fine-tuned model)"]
            ),
        )
        self.prompt_style_row.connect("notify::selected", self._save_output_mode)
        output.add(self.prompt_style_row)
        page.add(output)

        startup = Adw.PreferencesGroup(title="Startup")
        self.autostart_row = Adw.SwitchRow(
            title="Start automatically at login",
        )
        self.autostart_row.connect("notify::active", self._save_autostart)
        startup.add(self.autostart_row)
        page.add(startup)

        from .history import history_path

        hist_group = Adw.PreferencesGroup(
            title="History",
            description="Saves every proofread (your text included) to a "
                        "private local file — view with: aiproof history",
        )
        self.history_row = Adw.SwitchRow(
            title="Save proofread history",
            subtitle=str(history_path()),
        )
        self.history_row.connect("notify::active", self._save_history_toggle)
        hist_group.add(self.history_row)

        self.context_row = Adw.SwitchRow(
            title="Context-aware proofreading",
            subtitle="Learns your domain vocabulary and typical writing "
                     "from saved history — view with: aiproof context",
        )
        self.context_row.connect("notify::active", self._save_context_toggle)
        hist_group.add(self.context_row)
        page.add(hist_group)

        return page

    def _save_hotkeys(self, _row) -> None:
        if self._loading:
            return
        config.update(
            hotkey=self.hotkey_row.get_text().strip(),
            hotkey_clipboard=self.hotkey_clip_row.get_text().strip(),
        )

    def _save_behavior(self, *_args) -> None:
        if self._loading:
            return
        config.update(
            paste_mode="clipboard-only" if self.paste_mode_row.get_selected() == 1
            else "auto",
            restore_delay_ms=int(self.delay_row.get_value()),
        )

    def _save_output_mode(self, *_args) -> None:
        if self._loading:
            return
        config.update(
            output_mode="edits" if self.output_mode_row.get_selected() == 1
            else "rewrite",
            prompt_style="minimal" if self.prompt_style_row.get_selected() == 1
            else "full",
        )

    def _save_history_toggle(self, *_args) -> None:
        if self._loading:
            return
        config.update(save_history=self.history_row.get_active())

    def _save_context_toggle(self, *_args) -> None:
        if self._loading:
            return
        config.update(context_aware=self.context_row.get_active())

    def _save_autostart(self, *_args) -> None:
        if self._loading:
            return
        enabled = self.autostart_row.get_active()
        config.update(autostart=enabled)
        _write_autostart_override(enabled)

    # -- setup page ------------------------------------------------------------

    def _build_setup_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(
            title="Setup", icon_name="emblem-ok-symbolic", name="setup"
        )
        self.setup_group = Adw.PreferencesGroup(
            title="System checks",
            description="Everything aiproof needs for seamless in-place replacement",
        )
        refresh = Gtk.Button(label="Re-run checks", halign=Gtk.Align.START)
        refresh.connect("clicked", lambda *_: self._refresh_checks())
        self.setup_group.set_header_suffix(refresh)
        page.add(self.setup_group)
        self._check_rows: list[Adw.ActionRow] = []
        self._refresh_checks()
        return page

    def _refresh_checks(self) -> None:
        from .setup_check import run_all

        for row in self._check_rows:
            self.setup_group.remove(row)
        self._check_rows.clear()

        def work():
            results = run_all(config.load())
            GLib.idle_add(self._show_checks, results)

        threading.Thread(target=work, daemon=True).start()

    def _show_checks(self, results) -> None:
        for title, ok, hint in results:
            row = Adw.ActionRow(
                title=("✔ " if ok else "✘ ") + title,
                subtitle=hint if not ok else "",
            )
            self.setup_group.add(row)
            self._check_rows.append(row)

    # -- shared ---------------------------------------------------------------

    def _load_values(self) -> None:
        cfg = self.cfg
        try:
            idx = PROVIDER_IDS.index(cfg["provider"])
        except ValueError:
            idx = 0
        self.provider_row.set_selected(idx)
        self.endpoint_row.set_text(cfg["endpoint"] or "")
        self.model_row.set_text(cfg["model"] or "")
        self.api_key_row.set_text(secrets.get_api_key(cfg["provider"]) or "")
        self._update_provider_note(cfg["provider"])
        self.hotkey_row.set_text(cfg["hotkey"])
        self.hotkey_clip_row.set_text(cfg["hotkey_clipboard"])
        self.paste_mode_row.set_selected(
            1 if cfg.get("paste_mode") == "clipboard-only" else 0
        )
        self.delay_row.set_value(cfg.get("restore_delay_ms", 500))
        self.output_mode_row.set_selected(
            1 if cfg.get("output_mode") == "edits" else 0
        )
        self.prompt_style_row.set_selected(
            1 if cfg.get("prompt_style") == "minimal" else 0
        )
        self.autostart_row.set_active(bool(cfg.get("autostart", True)))
        self.history_row.set_active(bool(cfg.get("save_history", False)))
        self.context_row.set_active(bool(cfg.get("context_aware", False)))

        fallback = cfg.get("fallback") or {}
        self.fb_enabled_row.set_active(bool(fallback))
        try:
            fb_idx = PROVIDER_IDS.index(fallback.get("provider", "ollama"))
        except ValueError:
            fb_idx = 0
        self.fb_provider_row.set_selected(fb_idx)
        self.fb_endpoint_row.set_text(fallback.get("endpoint", ""))
        self.fb_model_row.set_text(fallback.get("model", ""))

    def _flash(self, row, text: str) -> None:
        toast = Adw.Toast(title=text, timeout=3)
        self.add_toast(toast)


def _write_autostart_override(enabled: bool) -> None:
    """Per-user autostart control. The deb ships /etc/xdg/autostart/…-daemon
    .desktop; a user file with the same name overrides it (Hidden=true
    disables). When running from source there may be no system file, so the
    enabled case writes a complete launcher."""
    import shutil
    from pathlib import Path

    autostart_dir = Path(config.config_dir()).parent / "autostart"
    autostart_dir.mkdir(parents=True, exist_ok=True)
    path = autostart_dir / f"{APP_ID}-daemon.desktop"
    if enabled:
        exe = shutil.which("aiproof") or "aiproof"
        path.write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=AI Proofreader daemon\n"
            f"Exec={exe} daemon\n"
            f"Icon={APP_ID}\n"
            "OnlyShowIn=GNOME;\n"
            "X-GNOME-Autostart-enabled=true\n"
            "X-GNOME-Autostart-Delay=3\n"
        )
    else:
        path.write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=AI Proofreader daemon\n"
            "Hidden=true\n"
        )


class SettingsApp(Adw.Application):
    def __init__(self, page: str | None):
        super().__init__(
            application_id=f"{APP_ID}.Settings",
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        self.page = page

    def do_activate(self):
        win = self.get_active_window()
        if not win:
            win = SettingsWindow(self, page=self.page)
        win.present()


def run_settings(page: str | None = None) -> int:
    app = SettingsApp(page)
    return app.run([])

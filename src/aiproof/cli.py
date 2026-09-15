"""Command-line entry point: aiproof <daemon|trigger|trigger-clipboard|settings|setup|proofread>."""

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="aiproof",
        description="OS-wide AI text proofreader for GNOME/Wayland",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("daemon", help="run the background daemon (tray icon, DBus, hotkeys)")

    p_trigger = sub.add_parser("trigger", help="proofread the current selection in place")
    p_trigger.add_argument(
        "--oneshot", action="store_true",
        help="run the flow in-process instead of calling the daemon",
    )

    p_trigclip = sub.add_parser(
        "trigger-clipboard",
        help="proofread the primary selection, result to clipboard (no paste)",
    )
    p_trigclip.add_argument("--oneshot", action="store_true",
                            help="run in-process instead of calling the daemon")

    p_settings = sub.add_parser("settings", help="open the settings window")
    p_settings.add_argument("--page", default=None, choices=["provider", "behavior", "setup"])

    sub.add_parser("setup", help="run first-time setup checks in the terminal")

    sub.add_parser("proofread", help="proofread stdin to stdout (debug)")

    p_context = sub.add_parser(
        "context",
        help="show the learned writing context (vocabulary + domain summary)",
    )
    p_context.add_argument("--refresh", action="store_true",
                           help="rebuild now (includes the LLM domain summary)")
    p_context.add_argument("--clear", action="store_true",
                           help="delete the learned context")

    p_history = sub.add_parser(
        "history", help="show saved proofreads (enable in settings first)"
    )
    p_history.add_argument("--tail", type=int, default=10, metavar="N",
                           help="show the last N entries (default 10)")
    p_history.add_argument("--clear", action="store_true",
                           help="delete the history file")
    p_history.add_argument("--path", action="store_true",
                           help="print the history file path")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.cmd == "proofread":
        return _cmd_proofread()
    if args.cmd == "history":
        return _cmd_history(args)
    if args.cmd == "context":
        return _cmd_context(args)
    if args.cmd == "daemon":
        from .daemon import run_daemon
        return run_daemon()
    if args.cmd == "trigger":
        return _cmd_trigger("Trigger", args.oneshot, clipboard_only=False)
    if args.cmd == "trigger-clipboard":
        return _cmd_trigger("TriggerClipboardOnly", args.oneshot, clipboard_only=True)
    if args.cmd == "settings":
        from .settings_app import run_settings
        return run_settings(page=args.page)
    if args.cmd == "setup":
        from .setup_check import run_cli
        return run_cli()
    parser.error(f"unknown command {args.cmd}")
    return 2


def _cmd_proofread() -> int:
    from . import config
    from .llm.client import LLMError, proofread_with_fallback

    text = sys.stdin.read()
    if not text.strip():
        log.info("no input on stdin")
        print("aiproof: no input on stdin", file=sys.stderr)
        return 1
    # stdin usually ends with a newline that isn't part of the "selection"
    stripped_trailing = text.endswith("\n") and not text.endswith("\n\n")
    if stripped_trailing:
        text = text[:-1]
    log.debug("proofread: %d chars from stdin (stripped trailing newline=%s)",
              len(text), stripped_trailing)

    def on_fallback(error):
        print(f"aiproof: primary provider failed ({error}); "
              "trying fallback…", file=sys.stderr)

    cfg = config.load()
    try:
        corrected, elapsed, client, used_fb = proofread_with_fallback(
            cfg, text, on_fallback=on_fallback
        )
    except LLMError as e:
        log.warning("proofread failed: %s", e)
        print(f"aiproof: {e}", file=sys.stderr)
        return 1

    from . import history
    fb = (cfg.get("fallback") or {}) if used_fb else {}
    history.record(
        cfg, source="cli", original=text, corrected=corrected,
        elapsed=elapsed, path=client.last_path,
        provider=fb.get("provider") or cfg["provider"],
        model=fb.get("model") or cfg["model"], used_fallback=used_fb,
    )
    sys.stdout.write(corrected)
    if stripped_trailing:
        sys.stdout.write("\n")
    log.debug("proofread done: path=%s, used_fallback=%s",
              client.last_path, used_fb)
    print(
        f"[{client.cfg['provider']}/{client.model} in {elapsed:.1f}s, "
        f"path={client.last_path}]",
        file=sys.stderr,
    )
    if client.last_path == "fallback":
        log.warning("every correction was rejected by the guards; "
                    "output is the unverified original")
        print(
            "aiproof: WARNING — every correction was rejected by the safety "
            "guards; output is your original text, NOT verified clean.",
            file=sys.stderr,
        )

    # Vocab-only refresh, synchronous and AFTER all output (a background
    # thread would die at CLI exit; the LLM summary is daemon-side only).
    # min_new_entries=5: don't reload the dictionary on every single run.
    from . import context
    if cfg.get("context_aware") and cfg.get("save_history"):
        context.refresh_if_stale(cfg, min_new_entries=5)
    return 0


def _cmd_context(args) -> int:
    from . import config, context

    cfg = config.load()
    if args.clear:
        if context.clear():
            print("Learned context cleared.")
        else:
            print("No learned context to clear.")
        return 0
    if args.refresh:
        if not cfg.get("save_history"):
            print("aiproof: history is disabled — nothing to learn from. "
                  "Enable 'Save proofread history' in settings first.",
                  file=sys.stderr)
            return 1
        context.refresh(cfg, allow_summary=True)
        print("Context refreshed.")

    words, summary = context.load_words_and_summary(
        dict(cfg, context_aware=True)  # show the profile even when injection is off
    )
    enabled = "enabled" if cfg.get("context_aware") else "DISABLED (not injected)"
    print(f"Context-aware proofreading: {enabled}")
    print(f"Typical texts: {summary or '(no summary yet)'}")
    if words:
        print(f"Known vocabulary ({len(words)}): " + ", ".join(words))
    else:
        print("Known vocabulary: (none learned yet — needs a few saved "
              "proofreads)")
    return 0


def _cmd_history(args) -> int:
    from . import context, history

    if args.path:
        print(history.history_path())
        return 0
    if args.clear:
        cleared = history.clear()
        context.clear()  # a deleted history must stop influencing prompts
        if cleared:
            print("History cleared (learned context too).")
        else:
            print("No history file to clear.")
        return 0

    entries = history.read_recent(args.tail)
    if not entries:
        from . import config
        if config.load().get("save_history"):
            print("No proofreads recorded yet.")
        else:
            print("History is disabled — enable it in aiproof settings "
                  "(Behavior page) or set \"save_history\": true in "
                  "~/.config/aiproof/config.json")
        return 0
    for e in entries:
        mark = "changed" if e.get("changed") else "clean"
        print(f"— {e.get('timestamp', '?')}  [{e.get('source', '?')}, "
              f"{e.get('provider', '?')}/{e.get('model', '?')}, "
              f"{e.get('pipeline_path', '?')}, {mark}"
              f"{', fallback' if e.get('used_fallback') else ''}]")
        original = str(e.get("original", "")).replace("\n", "\n    ")
        print(f"  < {original}")
        if e.get("changed"):
            corrected = str(e.get("corrected", "")).replace("\n", "\n    ")
            print(f"  > {corrected}")
    return 0


def _cmd_trigger(method: str, oneshot: bool, clipboard_only: bool) -> int:
    if oneshot:
        from . import config
        from .notify import Notifier
        from .orchestrator import Orchestrator

        orch = Orchestrator(config.load(), Notifier())
        ok = orch.run_clipboard_flow() if clipboard_only else orch.run_paste_flow()
        log.debug("oneshot %s flow -> %s",
                  "clipboard" if clipboard_only else "paste", ok)
        return 0 if ok else 1

    import gi  # noqa: F401
    from gi.repository import Gio, GLib

    from . import DBUS_IFACE, DBUS_NAME, DBUS_PATH

    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        bus.call_sync(
            DBUS_NAME, DBUS_PATH, DBUS_IFACE, method,
            None, None, Gio.DBusCallFlags.NONE, 5000, None,
        )
        log.debug("DBus %s dispatched to the daemon", method)
        return 0
    except GLib.Error as e:
        log.warning("daemon unreachable over DBus: %s", e.message)
        print(
            f"aiproof: could not reach the daemon ({e.message}).\n"
            "Is it running? Start it with: aiproof daemon",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

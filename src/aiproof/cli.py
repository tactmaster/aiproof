"""Command-line entry point: aiproof <daemon|trigger|trigger-clipboard|settings|setup|proofread>."""

import argparse
import logging
import sys


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

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.cmd == "proofread":
        return _cmd_proofread()
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
    from .llm.client import LLMError, client_from_config

    text = sys.stdin.read()
    if not text.strip():
        print("aiproof: no input on stdin", file=sys.stderr)
        return 1
    # stdin usually ends with a newline that isn't part of the "selection"
    stripped_trailing = text.endswith("\n") and not text.endswith("\n\n")
    if stripped_trailing:
        text = text[:-1]

    client = client_from_config(config.load())
    try:
        corrected, elapsed = client.proofread(text)
    except LLMError as e:
        print(f"aiproof: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(corrected)
    if stripped_trailing:
        sys.stdout.write("\n")
    print(
        f"[{client.cfg['provider']}/{client.model} in {elapsed:.1f}s, "
        f"path={client.last_path}]",
        file=sys.stderr,
    )
    if client.last_path == "fallback":
        print(
            "aiproof: WARNING — every correction was rejected by the safety "
            "guards; output is your original text, NOT verified clean.",
            file=sys.stderr,
        )
    return 0


def _cmd_trigger(method: str, oneshot: bool, clipboard_only: bool) -> int:
    if oneshot:
        from . import config
        from .notify import Notifier
        from .orchestrator import Orchestrator

        orch = Orchestrator(config.load(), Notifier())
        ok = orch.run_clipboard_flow() if clipboard_only else orch.run_paste_flow()
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
        return 0
    except GLib.Error as e:
        print(
            f"aiproof: could not reach the daemon ({e.message}).\n"
            "Is it running? Start it with: aiproof daemon",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

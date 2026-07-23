"""First-run / diagnostic checks with guided fixes.

run_all() -> [(title, ok, fix_hint)] — used by the daemon (notification) and
the settings window. run_cli() prints a friendly report and applies the safe
fixes automatically.
"""

import grp
import os
import shutil
import subprocess

import requests

from . import keystroke
from .llm.providers import get_provider


NAUTILUS_SCRIPT_NAME = "Proofread with AI"

# Installed to ~/.local/share/nautilus/scripts/ — Nautilus (Files) shows it
# under right-click → Scripts. This is the only OS-level right-click hook
# GNOME offers; in-app text context menus are not extensible on Wayland.
NAUTILUS_SCRIPT = """\
#!/bin/sh
# aiproof: proofread selected text files in place (backup kept as *.orig)
IFS='
'
for f in $NAUTILUS_SCRIPT_SELECTED_FILE_PATHS; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    case "$(file -b --mime-type "$f")" in
        text/*|application/json|application/x-yaml) ;;
        *)  notify-send --app-name=aiproof "Skipped: $name" \\
                "Not a text file."
            continue ;;
    esac
    notify-send --app-name=aiproof "Proofreading: $name" "…"
    if aiproof proofread < "$f" > "$f.aiproof-tmp" 2>/dev/null \\
            && [ -s "$f.aiproof-tmp" ]; then
        cp -p "$f" "$f.orig"
        mv "$f.aiproof-tmp" "$f"
        notify-send --app-name=aiproof "Proofread: $name" \\
            "Original kept as $name.orig"
    else
        rm -f "$f.aiproof-tmp"
        notify-send --app-name=aiproof "Proofread failed: $name" \\
            "Run: aiproof proofread < '$f'  to see the error."
    fi
done
"""


def nautilus_scripts_dir() -> str:
    data_home = os.environ.get(
        "XDG_DATA_HOME", os.path.expanduser("~/.local/share")
    )
    return os.path.join(data_home, "nautilus", "scripts")


def nautilus_script_installed() -> bool:
    return os.path.exists(
        os.path.join(nautilus_scripts_dir(), NAUTILUS_SCRIPT_NAME)
    )


def install_nautilus_script() -> bool:
    """Install the Files right-click entry. Returns True if (now) installed."""
    if not shutil.which("nautilus"):
        return False
    path = os.path.join(nautilus_scripts_dir(), NAUTILUS_SCRIPT_NAME)
    os.makedirs(nautilus_scripts_dir(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(NAUTILUS_SCRIPT)
    os.chmod(path, 0o755)
    return True


def _service_active() -> bool:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "is-active", "ydotool.service"],
            capture_output=True, timeout=5,
        )
        return proc.stdout.decode().strip() == "active"
    except Exception:
        return False


def _in_input_group() -> bool:
    try:
        input_gid = grp.getgrnam("input").gr_gid
        return input_gid in os.getgroups()
    except KeyError:
        return False


def _uinput_accessible() -> bool:
    return os.access("/dev/uinput", os.R_OK | os.W_OK)


def _hotkeys_registered() -> bool:
    try:
        from . import hotkey
        return hotkey.is_registered()
    except Exception:
        return False


def _provider_reachable(cfg: dict) -> tuple[bool, str]:
    provider = get_provider(cfg["provider"])
    if provider["format"] != "ollama":
        return True, ""  # remote providers only verifiable with a paid call
    endpoint = (cfg.get("endpoint") or provider["endpoint"]).rstrip("/")
    try:
        resp = requests.get(f"{endpoint}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return False, f"Ollama not reachable at {endpoint} — is it running?"
    model = cfg.get("model", "")
    if model and model not in models:
        return False, (
            f"Model {model!r} not pulled — run: ollama pull {model}"
        )
    return True, ""


def run_all(cfg: dict) -> list[tuple[str, bool, str]]:
    results = []

    ok = shutil.which("ydotool") is not None
    results.append((
        "ydotool installed", ok,
        "" if ok else "sudo apt install ydotool",
    ))

    ok = _service_active() and os.path.exists(keystroke.socket_path())
    results.append((
        "ydotoold service running", ok,
        "" if ok else "systemctl --user enable --now ydotool.service",
    ))

    group_ok = _in_input_group()
    access_ok = _uinput_accessible()
    ok = group_ok or access_ok
    results.append((
        "/dev/uinput permission", ok,
        "" if ok else
        "sudo usermod -aG input $USER  (then log out and back in)",
    ))
    if group_ok and not access_ok:
        # group is set but this session predates it
        results[-1] = (
            "/dev/uinput permission", False,
            "log out and back in (group change not yet active)",
        )

    ok = _hotkeys_registered()
    results.append((
        "global hotkeys registered", ok,
        "" if ok else "start the daemon once: aiproof daemon",
    ))

    ok, hint = _provider_reachable(cfg)
    results.append(("AI provider reachable", ok, hint))

    return results


def run_cli() -> int:
    from . import config

    cfg = config.load()

    # Safe automatic fixes: enable the ydotoold user service if merely
    # inactive; install the Files right-click script if Nautilus is present.
    if shutil.which("ydotool") and not _service_active():
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "ydotool.service"],
            capture_output=True, timeout=10,
        )
    if install_nautilus_script():
        print(f' installed Files right-click entry: Scripts → "{NAUTILUS_SCRIPT_NAME}"')

    results = run_all(cfg)
    all_ok = True
    for title, ok, hint in results:
        mark = "\033[32m✔\033[0m" if ok else "\033[31m✘\033[0m"
        print(f" {mark} {title}")
        if not ok:
            all_ok = False
            if hint:
                print(f"    fix: {hint}")
    if all_ok:
        config.update(setup_complete=True)
        print("\nEverything looks good. Select text anywhere and press "
              f"{cfg['hotkey'].replace('<', '').replace('>', '+')} !")
    else:
        print("\nFix the items above, then run 'aiproof setup' again.")
    return 0 if all_ok else 1

"""Configuration: ~/.config/aiproof/config.json with defaults and atomic writes."""

import json
import os
import tempfile
from pathlib import Path

DEFAULTS = {
    "schema_version": 1,
    "provider": "ollama",
    "model": "phi3.5:latest",
    "endpoint": "http://127.0.0.1:11434",
    "google_api_version": "v1beta",
    # "auto": copy selection + paste result via ydotool; "clipboard-only": read
    # primary selection, leave result on the clipboard for a manual paste.
    "paste_mode": "auto",
    "hotkey": "<Control><Super>p",
    "hotkey_clipboard": "<Control><Shift><Super>p",
    "restore_delay_ms": 500,
    "max_chars": 10000,
    "request_timeout_s": 60,
    "autostart": True,
    "setup_complete": False,
    # Fallback only when the Secret Service keyring is unavailable.
    "api_key_plaintext": None,
    # Per-provider {"model": ..., "endpoint": ...}, so the Settings window can
    # restore what you had before when you switch providers and back.
    "provider_overrides": {},
    # Optional {"provider": ..., "endpoint": ..., "model": ...} tried when the
    # primary provider is unreachable or fails (e.g. local Ollama for when
    # the remote GPU box is off the network). None disables failover.
    "fallback": None,
    # Save every proofread (original + corrected + metadata) to
    # ~/.local/share/aiproof/history.jsonl. Off by default: it's your text.
    "save_history": False,
    # Inject learned context (domain vocabulary + summary from saved history)
    # into the proofreading prompt. Requires save_history for data.
    "context_aware": False,
}


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(base) / "aiproof"


def config_path() -> Path:
    return config_dir() / "config.json"


def load() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(config_path(), encoding="utf-8") as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            for key in DEFAULTS:
                if key in stored:
                    cfg[key] = stored[key]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return cfg


def save(cfg: dict) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {key: cfg[key] for key in DEFAULTS if key in cfg}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def update(**changes) -> dict:
    cfg = load()
    cfg.update(changes)
    save(cfg)
    return cfg

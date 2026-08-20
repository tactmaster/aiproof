"""Opt-in proofread history.

Every completed proofread (changed or clean) is appended as one JSON line to
~/.local/share/aiproof/history.jsonl — original text, corrected text, and the
run metadata (source flow, provider/model, pipeline path, timing).

This contains the user's own text, so it is OFF by default
(config "save_history"), the file is created 0600, and content never goes
anywhere except this local file.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def history_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    return Path(base) / "aiproof"


def history_path() -> Path:
    return history_dir() / "history.jsonl"


def record(cfg: dict, *, source: str, original: str, corrected: str,
           elapsed: float, path: str, provider: str, model: str,
           used_fallback: bool = False) -> None:
    """Append one proofread to the history. No-op unless cfg['save_history'].
    Never raises — history must not break the proofread itself."""
    if not cfg.get("save_history"):
        return
    entry = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "provider": provider,
        "model": model,
        "pipeline_path": path,
        "used_fallback": used_fallback,
        "elapsed_s": round(elapsed, 2),
        "changed": corrected != original,
        "original": original,
        "corrected": corrected,
    }
    try:
        history_dir().mkdir(parents=True, exist_ok=True)
        target = history_path()
        existed = target.exists()
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if not existed:
            os.chmod(target, 0o600)
    except Exception:
        log.warning("could not write proofread history", exc_info=True)


def read_recent(count: int = 10) -> list[dict]:
    """Last `count` entries, oldest first. Malformed lines are skipped."""
    try:
        lines = history_path().read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    entries = []
    for line in lines[-count:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def clear() -> bool:
    """Delete the history file. Returns True if there was one."""
    try:
        history_path().unlink()
        return True
    except FileNotFoundError:
        return False

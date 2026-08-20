import json
import os
import stat

import pytest

from aiproof import history
from aiproof.config import DEFAULTS


@pytest.fixture(autouse=True)
def temp_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    return tmp_path


def make_cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    return cfg


def record(cfg, **over):
    kwargs = dict(
        source="paste", original="teh text", corrected="the text",
        elapsed=0.31, path="ok", provider="ollama", model="mistral:latest",
        used_fallback=False,
    )
    kwargs.update(over)
    history.record(cfg, **kwargs)


class TestRecord:
    def test_disabled_by_default_writes_nothing(self):
        record(make_cfg())
        assert not history.history_path().exists()

    def test_enabled_writes_jsonl_entry(self):
        record(make_cfg(save_history=True))
        lines = history.history_path().read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["original"] == "teh text"
        assert entry["corrected"] == "the text"
        assert entry["changed"] is True
        assert entry["source"] == "paste"
        assert entry["provider"] == "ollama"
        assert entry["pipeline_path"] == "ok"
        assert entry["elapsed_s"] == 0.31
        assert "timestamp" in entry

    def test_unchanged_marked_clean(self):
        record(make_cfg(save_history=True), corrected="teh text")
        entry = json.loads(history.history_path().read_text())
        assert entry["changed"] is False

    def test_file_is_private(self):
        record(make_cfg(save_history=True))
        mode = stat.S_IMODE(os.stat(history.history_path()).st_mode)
        assert mode == 0o600

    def test_appends(self):
        cfg = make_cfg(save_history=True)
        record(cfg)
        record(cfg, original="secnd", corrected="second")
        assert len(history.history_path().read_text().splitlines()) == 2

    def test_never_raises_on_write_failure(self, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", "/proc/definitely-not-writable")
        record(make_cfg(save_history=True))  # must not raise


class TestReadRecent:
    def test_missing_file_empty(self):
        assert history.read_recent() == []

    def test_tail_and_order(self):
        cfg = make_cfg(save_history=True)
        for i in range(5):
            record(cfg, original=f"text {i}")
        entries = history.read_recent(2)
        assert [e["original"] for e in entries] == ["text 3", "text 4"]

    def test_malformed_lines_skipped(self):
        cfg = make_cfg(save_history=True)
        record(cfg)
        with open(history.history_path(), "a") as f:
            f.write("{not json\n")
        record(cfg)
        assert len(history.read_recent(10)) == 2


class TestClear:
    def test_clear_removes_file(self):
        record(make_cfg(save_history=True))
        assert history.clear() is True
        assert not history.history_path().exists()
        assert history.clear() is False

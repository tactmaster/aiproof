import json

import pytest

from aiproof import config


@pytest.fixture(autouse=True)
def temp_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


class TestConfig:
    def test_load_defaults_when_missing(self):
        cfg = config.load()
        assert cfg == config.DEFAULTS
        assert cfg is not config.DEFAULTS

    def test_save_load_roundtrip(self):
        cfg = config.load()
        cfg["provider"] = "openai"
        cfg["model"] = "gpt-4o-mini"
        config.save(cfg)
        loaded = config.load()
        assert loaded["provider"] == "openai"
        assert loaded["model"] == "gpt-4o-mini"
        assert loaded["restore_delay_ms"] == 500

    def test_update_merges(self):
        config.update(model="other")
        assert config.load()["model"] == "other"
        assert config.load()["provider"] == config.DEFAULTS["provider"]

    def test_unknown_keys_dropped(self):
        config.config_dir().mkdir(parents=True)
        config.config_path().write_text(
            json.dumps({"provider": "google", "bogus": 1})
        )
        cfg = config.load()
        assert cfg["provider"] == "google"
        assert "bogus" not in cfg

    def test_corrupt_file_falls_back_to_defaults(self):
        config.config_dir().mkdir(parents=True)
        config.config_path().write_text("{not json")
        assert config.load() == config.DEFAULTS

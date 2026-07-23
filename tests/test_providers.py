import pytest

from aiproof.config import DEFAULTS
from aiproof.llm.client import LLMClient, TextTooLongError
from aiproof.llm.providers import PROVIDERS, get_provider


def make_cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    return cfg


class TestRegistry:
    def test_all_formats_known(self):
        formats = ("ollama", "openai", "anthropic", "google")
        for pid, p in PROVIDERS.items():
            assert p["format"] in formats, pid

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            get_provider("nope")


class TestRequestShapes:
    def test_ollama(self):
        client = LLMClient(make_cfg())
        req = client.build_request("PROMPT")
        assert req["url"] == "http://127.0.0.1:11434/api/generate"
        assert req["body"] == {
            "model": "phi3.5:latest",
            "prompt": "PROMPT",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 2000},
        }

    def test_ollama_custom_endpoint_with_api_path_kept(self):
        cfg = make_cfg(endpoint="http://10.0.0.5:11434/api/generate")
        req = LLMClient(cfg).build_request("P")
        assert req["url"] == "http://10.0.0.5:11434/api/generate"

    def test_openai(self):
        cfg = make_cfg(
            provider="openai",
            endpoint="https://api.openai.com/v1/chat/completions",
            model="gpt-4o-mini",
        )
        req = LLMClient(cfg, api_key="sk-test").build_request("PROMPT")
        assert req["headers"]["Authorization"] == "Bearer sk-test"
        body = req["body"]
        assert body["messages"] == [{"role": "user", "content": "PROMPT"}]
        assert body["temperature"] == 0.1
        assert body["max_tokens"] == 2000

    def test_openai_no_key_omits_auth_header(self):
        cfg = make_cfg(
            provider="lmstudio",
            endpoint="http://127.0.0.1:1234/v1/chat/completions",
            model="local",
        )
        req = LLMClient(cfg).build_request("P")
        assert "Authorization" not in req["headers"]

    def test_anthropic(self):
        cfg = make_cfg(
            provider="anthropic",
            endpoint="https://api.anthropic.com/v1/messages",
            model="claude-3-5-haiku-latest",
        )
        req = LLMClient(cfg, api_key="sk-ant-x").build_request("PROMPT")
        assert req["headers"]["x-api-key"] == "sk-ant-x"
        assert req["headers"]["anthropic-version"] == "2023-06-01"
        assert req["body"]["max_tokens"] == 2000

    def test_google(self):
        cfg = make_cfg(
            provider="google",
            endpoint="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-1.5-flash",
            google_api_version="v1beta",
        )
        req = LLMClient(cfg, api_key="AIza-x").build_request("PROMPT")
        assert req["url"] == (
            "https://generativelanguage.googleapis.com"
            "/v1beta/models/gemini-1.5-flash:generateContent"
        )
        assert req["params"] == {"key": "AIza-x"}
        gen = req["body"]["generationConfig"]
        assert gen == {
            "temperature": 0,
            "maxOutputTokens": 2000,
            "candidateCount": 1,
            "topP": 1,
            "topK": 1,
        }


class TestLimits:
    def test_text_too_long(self):
        client = LLMClient(make_cfg(max_chars=10))
        with pytest.raises(TextTooLongError):
            client.proofread("x" * 11)

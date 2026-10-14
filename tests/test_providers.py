import pytest

from aiproof.config import DEFAULTS
from aiproof.llm.client import LLMClient, TextTooLongError
from aiproof.llm.providers import API_FORMATS, PROVIDERS, get_provider


def make_cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    return cfg


class TestRegistry:
    def test_all_formats_known(self):
        for pid, p in PROVIDERS.items():
            assert p["format"] in API_FORMATS, pid

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
            "keep_alive": "30m",
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


class TestGenerationOptions:
    """Edit mode sends its own token budget / temperature / system turn; the
    default request body must stay byte-identical."""

    def test_ollama_options_and_system(self):
        req = LLMClient(make_cfg()).build_request(
            "P", max_tokens=100, temperature=0.0, system="S")
        assert req["body"]["options"] == {"temperature": 0.0, "num_predict": 100}
        assert req["body"]["system"] == "S"
        assert "system" not in LLMClient(make_cfg()).build_request("P")["body"]

    def test_openai_system_message_first(self):
        cfg = make_cfg(provider="openai",
                       endpoint="https://api.openai.com/v1/chat/completions",
                       model="gpt-4o-mini")
        body = LLMClient(cfg).build_request(
            "P", max_tokens=100, temperature=0.0, system="S")["body"]
        assert body["messages"] == [{"role": "system", "content": "S"},
                                    {"role": "user", "content": "P"}]
        assert body["max_tokens"] == 100 and body["temperature"] == 0.0

    def test_anthropic_system_field(self):
        cfg = make_cfg(provider="anthropic",
                       endpoint="https://api.anthropic.com/v1/messages",
                       model="claude-3-5-haiku-latest")
        body = LLMClient(cfg).build_request("P", max_tokens=100, system="S")["body"]
        assert body["system"] == "S" and body["max_tokens"] == 100
        assert body["messages"] == [{"role": "user", "content": "P"}]

    def test_google_system_instruction(self):
        cfg = make_cfg(provider="google",
                       endpoint="https://generativelanguage.googleapis.com/v1beta",
                       model="gemini-1.5-flash")
        body = LLMClient(cfg).build_request("P", max_tokens=100, system="S")["body"]
        assert body["generationConfig"]["maxOutputTokens"] == 100
        assert body["systemInstruction"] == {"parts": [{"text": "S"}]}

    def test_query_forwards_only_given_options(self, monkeypatch):
        client = LLMClient(make_cfg())
        seen = {}

        def fake_dispatch(prompt, **options):
            seen.update(prompt=prompt, **options)
            return "ok"

        monkeypatch.setattr(client, "_dispatch", fake_dispatch)
        assert client.query("p", max_tokens=5, temperature=None) == "ok"
        assert seen == {"prompt": "p", "max_tokens": 5}


class TestOpenAIEndpointNormalization:
    """Local OpenAI-compatible servers (Lemonade, LM Studio, vLLM, LiteLLM)
    document a *base* URL like http://host:8020/v1. Pasting that in used to
    POST to /v1 itself and return 404."""

    @pytest.mark.parametrize("given,expected", [
        ("http://host:8020/v1", "http://host:8020/v1/chat/completions"),
        ("http://host:8020/v1/", "http://host:8020/v1/chat/completions"),
        ("http://host:8020", "http://host:8020/v1/chat/completions"),
        ("http://host:8020/v1/chat/completions",
         "http://host:8020/v1/chat/completions"),
        ("https://api.openai.com/v1/chat/completions",
         "https://api.openai.com/v1/chat/completions"),
    ])
    def test_chat_url(self, given, expected):
        from aiproof.llm.client import openai_chat_url
        assert openai_chat_url(given) == expected

    def test_models_url_pairs_with_chat_url(self):
        from aiproof.llm.client import openai_models_url
        assert openai_models_url("http://host:8020/v1") == \
            "http://host:8020/v1/models"
        assert openai_models_url("http://host:8020/v1/chat/completions") == \
            "http://host:8020/v1/models"

    def test_request_uses_the_normalized_url(self):
        cfg = make_cfg(provider="openai", endpoint="http://host:8020/v1",
                       model="Qwen3-4B")
        req = LLMClient(cfg).build_request("PROMPT")
        assert req["url"] == "http://host:8020/v1/chat/completions"


class TestOpenAIModelListing:
    def _client(self, monkeypatch, payload, status=200):
        from aiproof.llm import client as client_mod

        class Resp:
            status_code = status
            def json(self): return payload
            def raise_for_status(self):
                if status >= 400:
                    raise AssertionError("should have been handled")

        captured = {}

        def fake_get(url, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers or {}
            return Resp()

        monkeypatch.setattr(client_mod.requests, "get", fake_get)
        self.captured = captured
        return LLMClient(make_cfg(provider="openai",
                                  endpoint="http://host:8020/v1"),
                         api_key="secret-key")

    def test_lists_openai_compatible_models(self, monkeypatch):
        client = self._client(monkeypatch, {"data": [{"id": "Qwen3-4B"},
                                                     {"id": "llama-3"}]})
        assert client.list_models() == ["Qwen3-4B", "llama-3"]
        assert self.captured["url"] == "http://host:8020/v1/models"
        assert self.captured["headers"]["Authorization"] == "Bearer secret-key"

    def test_auth_failure_says_to_set_the_key(self, monkeypatch):
        from aiproof.llm.errors import LLMError
        client = self._client(monkeypatch, {}, status=401)
        with pytest.raises(LLMError, match="API key"):
            client.list_models()

    def test_unsupported_format_is_explicit(self):
        from aiproof.llm.errors import LLMError
        client = LLMClient(make_cfg(provider="anthropic"))
        with pytest.raises(LLMError, match="not supported"):
            client.list_models()

import pytest

from aiproof.config import DEFAULTS
from aiproof.llm.client import (
    LLMError,
    TextTooLongError,
    proofread_with_fallback,
)


class FakeClient:
    def __init__(self, cfg, result=None, error=None):
        self.cfg = cfg
        self.model = cfg["model"]
        self.last_path = "ok"
        self._result = result
        self._error = error

    def proofread(self, text):
        if self._error:
            raise self._error
        return self._result, 0.1


def make_cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    return cfg


FALLBACK = {"provider": "ollama", "endpoint": "http://127.0.0.1:11434",
            "model": "local-model"}


class TestProofreadWithFallback:
    def test_primary_success_no_fallback(self):
        cfg = make_cfg(fallback=FALLBACK)
        factory_calls = []

        def factory(c):
            factory_calls.append(c["model"])
            return FakeClient(c, result="fixed")

        out, _, client, used_fb = proofread_with_fallback(
            cfg, "text", _client_factory=factory
        )
        assert (out, used_fb) == ("fixed", False)
        assert factory_calls == [cfg["model"]]

    def test_primary_fails_fallback_used(self):
        cfg = make_cfg(fallback=FALLBACK)

        def factory(c):
            if c["model"] == FALLBACK["model"]:
                return FakeClient(c, result="fixed by fallback")
            return FakeClient(c, error=LLMError("Cannot reach host"))

        out, _, client, used_fb = proofread_with_fallback(
            cfg, "text", _client_factory=factory
        )
        assert (out, used_fb) == ("fixed by fallback", True)
        assert client.cfg["endpoint"] == FALLBACK["endpoint"]

    def test_on_fallback_called_with_primary_error(self):
        cfg = make_cfg(fallback=FALLBACK)
        seen = []

        def factory(c):
            if c["model"] == FALLBACK["model"]:
                return FakeClient(c, result="ok")
            return FakeClient(c, error=LLMError("boom"))

        proofread_with_fallback(cfg, "t", on_fallback=seen.append,
                                _client_factory=factory)
        assert len(seen) == 1 and "boom" in str(seen[0])

    def test_no_fallback_configured_raises(self):
        cfg = make_cfg(fallback=None)

        def factory(c):
            return FakeClient(c, error=LLMError("down"))

        with pytest.raises(LLMError):
            proofread_with_fallback(cfg, "t", _client_factory=factory)

    def test_text_too_long_never_falls_back(self):
        cfg = make_cfg(fallback=FALLBACK)

        def factory(c):
            return FakeClient(c, error=TextTooLongError(20000, 10000))

        with pytest.raises(TextTooLongError):
            proofread_with_fallback(cfg, "t", _client_factory=factory)

    def test_fallback_also_failing_raises(self):
        cfg = make_cfg(fallback=FALLBACK)

        def factory(c):
            return FakeClient(c, error=LLMError("everything is down"))

        with pytest.raises(LLMError):
            proofread_with_fallback(cfg, "t", _client_factory=factory)

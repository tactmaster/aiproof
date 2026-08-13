"""The provider-failover routes and the used_fallback x last_path matrix.

Vocabulary collision pinned here: last_path == "fallback" means "the safety
guards rejected every correction, original kept", while used_fallback == True
means "the secondary PROVIDER served the request". They are independent axes.

Unreachable combinations (asserted nowhere, documented here):
- last_path in ("line-by-line", "segments") with unchanged text — those paths
  are only assigned when the salvage changed something.
- last_path == "fallback" with changed text — that path always returns the
  original.
"""

import pytest

from conftest import make_cfg
from aiproof.llm.client import LLMError, TextTooLongError, proofread_with_fallback
from aiproof.llm.errors import LLMError as ErrorsLLMError

pytestmark = pytest.mark.flow

FB = {"provider": "ollama", "endpoint": "http://127.0.0.1:11434",
      "model": "local-model"}


class FakeClient:
    def __init__(self, cfg, result=None, error=None, path="ok"):
        self.cfg = cfg
        self.model = cfg.get("model")
        self.last_path = path
        self._result = result
        self._error = error

    def proofread(self, text):
        if self._error is not None:
            raise self._error
        return self._result, 0.1


def make_factory(*clients):
    """A _client_factory returning the given fakes in order, recording cfgs."""
    cfgs = []

    def factory(cfg):
        cfgs.append(cfg)
        client = clients[len(cfgs) - 1]
        client.cfg = cfg
        client.model = cfg.get("model")
        return client

    factory.cfgs = cfgs
    return factory


def test_errors_module_reexport():
    """mechanics: client re-exports the errors-module exception classes."""
    assert LLMError is ErrorsLLMError


# -- proofread_with_fallback mechanics ----------------------------------------


def test_primary_success_no_fallback(caplog):
    """failover: primary success returns used_fallback=False, no callback."""
    factory = make_factory(FakeClient({}, result="fixed"))
    fired = []
    corrected, elapsed, client, used = proofread_with_fallback(
        make_cfg(fallback=FB), "text", on_fallback=fired.append,
        _client_factory=factory)
    assert (corrected, used) == ("fixed", False)
    assert fired == []
    assert "primary provider failed" not in caplog.text


def test_failover_engages(caplog):
    """failover: primary LLMError -> callback fires, fallback client serves."""
    primary_error = LLMError("Cannot reach http://gpu:11434")
    factory = make_factory(FakeClient({}, error=primary_error),
                           FakeClient({}, result="fixed"))
    fired = []
    corrected, elapsed, client, used = proofread_with_fallback(
        make_cfg(fallback=FB), "text", on_fallback=fired.append,
        _client_factory=factory)
    assert (corrected, used) == ("fixed", True)
    assert fired == [primary_error]
    assert client.model == "local-model"
    assert "primary provider failed (Cannot reach http://gpu:11434)" in caplog.text


def test_fallback_cfg_overrides_only_truthy_keys():
    """failover: fallback cfg overrides provider/endpoint/model, inherits the
    rest (timeouts, max_chars, api key)."""
    factory = make_factory(FakeClient({}, error=LLMError("down")),
                           FakeClient({}, result="fixed"))
    cfg = make_cfg(fallback={"provider": "openai", "model": "gpt"},
                   max_chars=1234, request_timeout_s=7)
    proofread_with_fallback(cfg, "text", _client_factory=factory)
    fb_cfg = factory.cfgs[1]
    assert fb_cfg["provider"] == "openai"
    assert fb_cfg["model"] == "gpt"
    assert fb_cfg["endpoint"] == cfg["endpoint"]  # not in fallback -> inherited
    assert fb_cfg["max_chars"] == 1234
    assert fb_cfg["request_timeout_s"] == 7


def test_text_too_long_never_fails_over(caplog):
    """failover: TextTooLongError re-raises even with a fallback configured
    (the fallback inherits max_chars, so retrying is pointless)."""
    factory = make_factory(FakeClient({}, error=TextTooLongError(11, 10)))
    fired = []
    with pytest.raises(TextTooLongError):
        proofread_with_fallback(make_cfg(fallback=FB), "text",
                                on_fallback=fired.append,
                                _client_factory=factory)
    assert fired == []
    assert "not trying the fallback provider" in caplog.text


@pytest.mark.parametrize("fallback", [None, {}, {"model": "only-a-model"}])
def test_no_usable_fallback_reraises(fallback, caplog):
    """failover: no fallback / model-only fallback re-raises the primary
    error and logs why."""
    factory = make_factory(FakeClient({}, error=LLMError("down")))
    with pytest.raises(LLMError, match="down"):
        proofread_with_fallback(make_cfg(fallback=fallback), "text",
                                _client_factory=factory)
    assert "no usable fallback configured" in caplog.text


def test_fallback_also_fails_is_logged(caplog):
    """failover: the fallback provider's own failure propagates AND is logged
    (previously silent)."""
    factory = make_factory(FakeClient({}, error=LLMError("primary down")),
                           FakeClient({}, error=LLMError("fallback down too")))
    with pytest.raises(LLMError, match="fallback down too"):
        proofread_with_fallback(make_cfg(fallback=FB), "text",
                                _client_factory=factory)
    assert "primary provider failed (primary down)" in caplog.text
    assert "fallback provider also failed: fallback down too" in caplog.text


def test_on_fallback_exception_aborts_retry():
    """failover: an exception inside on_fallback propagates and aborts the
    retry. Known quirk (callbacks must not raise) — see docs/flow.md."""
    factory = make_factory(FakeClient({}, error=LLMError("down")),
                           FakeClient({}, result="never reached"))

    def bad_callback(error):
        raise RuntimeError("callback exploded")

    with pytest.raises(RuntimeError, match="callback exploded"):
        proofread_with_fallback(make_cfg(fallback=FB), "text",
                                on_fallback=bad_callback,
                                _client_factory=factory)


def test_unknown_primary_provider_never_falls_back():
    """failover: the primary client is built OUTSIDE the try, so an unknown
    provider id (ValueError) escapes without trying the fallback. Known
    quirk — see docs/flow.md."""

    def factory(cfg):
        raise ValueError("Unknown provider: typo")

    with pytest.raises(ValueError, match="Unknown provider"):
        proofread_with_fallback(make_cfg(fallback=FB), "text",
                                _client_factory=factory)


# -- the repair-call laundering fix (postprocess) ------------------------------


def test_repair_llm_error_propagates(caplog):
    """laundering fix: an LLMError from the repair *call* propagates (so
    provider failover can engage) instead of masquerading as path=fallback."""
    from aiproof.llm.postprocess import enforce_formatting_preservation_ex

    def dead_query(prompt):
        raise LLMError("Cannot reach host")

    with pytest.raises(LLMError, match="Cannot reach host"):
        # candidate merges the two lines -> formatting repair path -> query dies
        enforce_formatting_preservation_ex("one\ntwo", "one two", dead_query)


def test_repair_non_llm_error_still_degrades(caplog):
    """laundering fix: non-LLM crashes in the retry still degrade safely to
    (original, 'fallback') rather than losing the text."""
    from aiproof.llm.postprocess import enforce_formatting_preservation_ex

    def broken_query(prompt):
        raise RuntimeError("bug in prompt builder")

    final, path = enforce_formatting_preservation_ex("one\ntwo", "one two",
                                                     broken_query)
    assert (final, path) == ("one\ntwo", "fallback")
    assert "Formatting repair failed" in caplog.text


# -- used_fallback x last_path matrix at the orchestrator level ----------------


@pytest.mark.parametrize("path,used_fb,expected_summary", [
    ("ok", False, "Corrected in 0.3s"),
    ("repaired", False, "Corrected in 0.3s"),  # repair is invisible by design
    ("line-by-line", False, "Corrected in 0.3s — line-by-line mode"),
    ("segments", False, "Corrected in 0.3s — segments mode"),
    ("ok", True, "Corrected in 0.3s — via fallback local-model"),
    ("line-by-line", True,
     "Corrected in 0.3s — line-by-line mode — via fallback local-model"),
])
def test_changed_matrix_notes(orch, fake_clipboard, fake_llm, notifier,
                              path, used_fb, expected_summary):
    """matrix: every reachable changed-text combination and its user note."""
    fake_clipboard.primary_text = "teh text"
    fake_llm.path = path
    fake_llm.used_fallback = used_fb
    o = orch(fallback=FB)

    assert o.run_clipboard_flow() is True
    assert notifier.summaries[-1] == expected_summary
    assert (o.last_path, o.used_fallback) == (path, used_fb)


def test_guards_fallback_plus_provider_fallback(orch, fake_clipboard, fake_llm,
                                                notifier, caplog):
    """matrix: BOTH fallbacks at once — guards rejected everything AND the
    fallback provider served the (unchanged) result -> honest unsafe report;
    the provider detail lives in the logs."""
    fake_clipboard.primary_text = "teh text"
    fake_llm.corrected = "teh text"
    fake_llm.path = "fallback"
    fake_llm.used_fallback = True
    o = orch(fallback=FB)

    assert o.run_clipboard_flow() is False
    assert notifier.summaries[-1] == "Couldn't apply corrections safely"
    assert (o.last_path, o.used_fallback) == ("fallback", True)
    assert "rejected by the guards; reporting as unsafe" in caplog.text


# -- full chain: orchestrator -> real proofread_with_fallback -> fake clients --


def test_full_chain_failover(monkeypatch, notifier, fake_clipboard, no_sleep,
                             caplog):
    """integration: dead primary -> fallback client serves -> notification
    trio + 'via fallback' note, through the real failover code."""
    from aiproof.orchestrator import Orchestrator

    factory = make_factory(FakeClient({}, error=LLMError("dead host")),
                           FakeClient({}, result="the text"))
    monkeypatch.setattr("aiproof.llm.client.client_from_config", factory)
    fake_clipboard.primary_text = "teh text"
    o = Orchestrator(make_cfg(fallback=FB), notifier)

    assert o.run_clipboard_flow() is True
    assert notifier.summaries == [
        "Proofreading…",
        "Primary provider unavailable — trying fallback…",
        "Corrected in 0.1s — via fallback local-model",
    ]
    assert "dead host" in notifier.notes[1][1]
    assert (o.last_path, o.used_fallback) == ("ok", True)
    assert "primary provider failed (dead host)" in caplog.text


def test_full_chain_both_providers_down(monkeypatch, notifier, fake_clipboard,
                                        no_sleep, caplog):
    """integration: both providers down -> 'Proofreading failed' with the
    fallback's error, both failures logged."""
    from aiproof.orchestrator import Orchestrator

    factory = make_factory(FakeClient({}, error=LLMError("primary down")),
                           FakeClient({}, error=LLMError("fallback down too")))
    monkeypatch.setattr("aiproof.llm.client.client_from_config", factory)
    fake_clipboard.primary_text = "teh text"
    o = Orchestrator(make_cfg(fallback=FB), notifier)

    assert o.run_clipboard_flow() is False
    assert notifier.last == ("Proofreading failed", "fallback down too")
    assert "fallback provider also failed: fallback down too" in caplog.text
    assert "proofread failed: fallback down too" in caplog.text

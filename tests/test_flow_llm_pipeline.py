"""Flow routes through LLMClient.proofread: guards, salvage routing, and
transport error mapping. The HTTP layer is cut at client._dispatch (or
client.query); guard *content* is covered by test_postprocess.py — here we
pin which path the pipeline takes.
"""

import pytest
import requests

from conftest import make_cfg
from aiproof.llm.client import LLMClient, LLMError, TextTooLongError

pytestmark = pytest.mark.flow


def make_client(**cfg_over):
    return LLMClient(make_cfg(**cfg_over))


def test_text_too_long(caplog):
    """pipeline: over max_chars raises TextTooLongError before any request."""
    client = make_client(max_chars=10)
    with pytest.raises(TextTooLongError, match="Selection too long"):
        client.proofread("x" * 11)
    assert "selection too long: 11 > 10 characters" in caplog.text


def test_empty_response(monkeypatch, caplog):
    """pipeline: an empty model response raises and is logged."""
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "")
    with pytest.raises(LLMError, match="empty response"):
        client.proofread("hello world")
    assert "model returned an empty response" in caplog.text


def test_wrapper_only_response(monkeypatch, caplog):
    """pipeline: a boilerplate-only response raises and is logged."""
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "Some lead-in that ends with a colon:")
    with pytest.raises(LLMError, match="boilerplate"):
        client.proofread("hello world")
    assert "model returned only boilerplate" in caplog.text


def test_path_ok(monkeypatch, caplog):
    """pipeline: a valid correction passes straight through (path=ok)."""
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "the text")
    monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                        lambda orig, cand, q: (cand, "ok"))
    final, elapsed = client.proofread("teh text")
    assert final == "the text"
    assert client.last_path == "ok"
    assert "proofread done: path=ok" in caplog.text


def test_path_repaired(monkeypatch, caplog):
    """pipeline: a repaired correction reports path=repaired, no salvage."""
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "draft")
    monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                        lambda orig, cand, q: ("the text", "repaired"))
    final, _ = client.proofread("teh text\ntwo lines")
    assert final == "the text"
    assert client.last_path == "repaired"
    assert "salvaging" not in caplog.text


def test_fallback_multiline_salvages_line_by_line(monkeypatch, caplog):
    """pipeline: guards keep rejecting a 2-40 line text -> line-by-line."""
    text = "lne one\nlne two\nlne three"
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "draft")
    monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                        lambda orig, cand, q: (orig, "fallback"))
    monkeypatch.setattr(client, "_correct_line",
                        lambda line: line.replace("lne", "line"))
    final, _ = client.proofread(text)
    assert final == "line one\nline two\nline three"
    assert client.last_path == "line-by-line"
    assert "salvaging line-by-line (3 lines)" in caplog.text


def test_fallback_salvage_no_change(monkeypatch, caplog):
    """pipeline: salvage that fixes nothing keeps path=fallback and says so."""
    text = "lne one\nlne two"
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "draft")
    monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                        lambda orig, cand, q: (orig, "fallback"))
    monkeypatch.setattr(client, "_correct_line", lambda line: None)
    final, _ = client.proofread(text)
    assert final == text
    assert client.last_path == "fallback"
    assert "line-by-line salvage made no changes" in caplog.text


def test_fallback_single_line_salvages_segments(monkeypatch, caplog):
    """pipeline: guards keep rejecting a single line -> segment salvage."""
    text = "teh one, teh two"
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "draft")
    monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                        lambda orig, cand, q: (orig, "fallback"))
    monkeypatch.setattr(client, "_correct_line",
                        lambda line: line.replace("teh", "the"))
    final, _ = client.proofread(text)
    assert final == "the one, the two"
    assert client.last_path == "segments"
    assert "salvaging sentence segments" in caplog.text


def test_fallback_over_salvage_limit_skips(monkeypatch, caplog):
    """pipeline: >40 lines skips salvage, keeps the original, and logs why."""
    text = "\n".join(f"lne {i}" for i in range(41))
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "draft")
    monkeypatch.setattr("aiproof.llm.client.enforce_formatting_preservation_ex",
                        lambda orig, cand, q: (orig, "fallback"))
    final, _ = client.proofread(text)
    assert final == text
    assert client.last_path == "fallback"
    assert "41 lines exceeds the 40-line salvage limit" in caplog.text


def test_correct_line_llm_error_keeps_line(monkeypatch, caplog):
    """pipeline: a per-line LLM failure keeps the original line (None)."""
    client = make_client()

    def boom(prompt):
        raise LLMError("HTTP 500")

    monkeypatch.setattr(client, "query", boom)
    assert client._correct_line("teh line") is None
    assert "line correction failed: HTTP 500" in caplog.text


def test_correct_line_boilerplate_keeps_line(monkeypatch, caplog):
    """pipeline: a per-line boilerplate response keeps the original line."""
    client = make_client()
    monkeypatch.setattr(client, "query", lambda p: "Some lead-in with a colon:")
    assert client._correct_line("teh line") is None
    assert "line correction rejected (empty or boilerplate)" in caplog.text


def test_query_read_timeout(monkeypatch, caplog):
    """transport: a read timeout maps to a 'timed out' LLMError."""
    client = make_client()

    def boom(prompt):
        raise requests.Timeout()

    monkeypatch.setattr(client, "_dispatch", boom)
    with pytest.raises(LLMError, match="timed out after 60s"):
        client.query("p")
    assert "read timed out after 60s" in caplog.text


def test_query_connection_error(monkeypatch, caplog):
    """transport: a connection failure maps to 'Cannot reach'."""
    client = make_client()

    def boom(prompt):
        raise requests.ConnectionError()

    monkeypatch.setattr(client, "_dispatch", boom)
    with pytest.raises(LLMError, match="Cannot reach"):
        client.query("p")
    assert "(ConnectionError)" in caplog.text


def test_query_connect_timeout_is_unreachable(monkeypatch, caplog):
    """transport: ConnectTimeout (subclass of both) now reports the host as
    unreachable instead of as a 60s read timeout."""
    client = make_client()

    def boom(prompt):
        raise requests.ConnectTimeout()

    monkeypatch.setattr(client, "_dispatch", boom)
    with pytest.raises(LLMError, match="Cannot reach"):
        client.query("p")
    assert "(ConnectTimeout)" in caplog.text


def test_query_other_requests_errors_escape(monkeypatch):
    """transport: non-Timeout/ConnectionError exceptions (e.g. bad JSON) are
    NOT wrapped in LLMError, so provider failover never engages and the
    orchestrator reports 'Unexpected error'. Known quirk — see docs/flow.md."""
    client = make_client()

    def boom(prompt):
        raise requests.exceptions.JSONDecodeError("Expecting value", "", 0)

    monkeypatch.setattr(client, "_dispatch", boom)
    with pytest.raises(requests.exceptions.JSONDecodeError):
        client.query("p")

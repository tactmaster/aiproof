"""LLM client: builds requests for the four API formats, non-streaming,
temperature 0.1, max 2000 output tokens (shapes ported from the extension's
queryLocalProvider/queryRemoteProvider in background/background.js).
"""

import logging
import time
from typing import Callable, Optional

import requests

from .postprocess import (
    enforce_formatting_preservation_ex,
    full_clean,
    looks_like_wrapper_only,
    proofread_line_by_line,
)

# Don't burn one model call per line on huge selections.
MAX_SALVAGE_LINES = 40
from .prompts import build_proofread_prompt
from .providers import get_provider

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 2000
TEMPERATURE = 0.1


class LLMError(Exception):
    """User-presentable LLM failure."""


class TextTooLongError(LLMError):
    def __init__(self, length: int, limit: int):
        super().__init__(f"Selection too long ({length:,} > {limit:,} characters)")
        self.length = length
        self.limit = limit


class LLMClient:
    def __init__(self, cfg: dict, api_key: str = "",
                 on_config_update: Optional[Callable[..., None]] = None):
        self.cfg = cfg
        self.api_key = api_key or cfg.get("api_key_plaintext") or ""
        # Called with keyword updates worth persisting (google_api_version fallback).
        self.on_config_update = on_config_update
        provider = get_provider(cfg["provider"])
        self.format = provider["format"]
        self.endpoint = cfg.get("endpoint") or provider["endpoint"]
        self.model = cfg.get("model") or provider["model"]
        self.timeout = cfg.get("request_timeout_s", 60)
        self.max_chars = cfg.get("max_chars", 10000)
        # Which pipeline path produced the last proofread() result:
        # "ok" | "repaired" | "line-by-line" | "fallback"
        self.last_path = ""

    # -- public API ---------------------------------------------------------

    def proofread(self, text: str) -> tuple[str, float]:
        """Return (corrected_text, elapsed_seconds)."""
        if len(text) > self.max_chars:
            raise TextTooLongError(len(text), self.max_chars)
        start = time.monotonic()
        response = self.query(build_proofread_prompt(text))
        cleaned = full_clean(response, text)
        if not cleaned:
            raise LLMError("Model returned an empty response")
        if looks_like_wrapper_only(cleaned, text):
            raise LLMError(
                "Model returned only boilerplate, no corrected text "
                f"({cleaned[:60]!r})"
            )
        final, path = enforce_formatting_preservation_ex(text, cleaned, self.query)
        line_count = text.count("\n") + 1
        if path == "fallback" and 1 < line_count <= MAX_SALVAGE_LINES:
            log.info(
                "whole-text correction kept mangling the layout; "
                "salvaging line-by-line (%d lines)", line_count,
            )
            salvaged = proofread_line_by_line(text, self._correct_line)
            if salvaged != text:
                final, path = salvaged, "line-by-line"
        self.last_path = path
        log.info(
            "proofread done: path=%s, %d chars, %d lines, %.1fs",
            path, len(text), line_count, time.monotonic() - start,
        )
        return final, time.monotonic() - start

    def _correct_line(self, line: str) -> str | None:
        """One line through the model + cleanup; None means keep the
        original line."""
        try:
            response = self.query(build_proofread_prompt(line))
        except LLMError as e:
            log.warning("line correction failed: %s", e)
            return None
        cleaned = full_clean(response, line)
        if not cleaned or looks_like_wrapper_only(cleaned, line):
            return None
        return cleaned

    def query(self, prompt: str) -> str:
        try:
            return self._dispatch(prompt)
        except requests.Timeout:
            raise LLMError(f"Request timed out after {self.timeout}s") from None
        except requests.ConnectionError as e:
            raise LLMError(f"Cannot reach {self.endpoint}: connection failed") from e

    def test_connection(self) -> tuple[bool, str]:
        old_timeout = self.timeout
        self.timeout = 10
        try:
            reply = self.query('Reply with the single word "OK".')
            return True, f"Connected — model replied: {reply.strip()[:60]}"
        except Exception as e:
            return False, str(e)
        finally:
            self.timeout = old_timeout

    def list_models(self) -> list[str]:
        """Model listing (Ollama only)."""
        if self.format != "ollama":
            raise LLMError("Model listing is only supported for Ollama")
        base = self.endpoint.rstrip("/")
        resp = requests.get(f"{base}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    # -- request builders ---------------------------------------------------

    def _dispatch(self, prompt: str) -> str:
        if self.format == "ollama":
            return self._query_ollama(prompt)
        if self.format == "openai":
            return self._query_openai(prompt)
        if self.format == "anthropic":
            return self._query_anthropic(prompt)
        if self.format == "google":
            return self._query_google(prompt)
        raise LLMError(f"Unknown API format: {self.format}")

    def build_request(self, prompt: str) -> dict:
        """Request description (url/headers/body) — used by tests and debug."""
        if self.format == "ollama":
            base = self.endpoint.rstrip("/")
            url = base if "/api/" in base else f"{base}/api/generate"
            return {
                "url": url,
                "headers": {},
                "body": {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": TEMPERATURE, "num_predict": MAX_OUTPUT_TOKENS},
                },
            }
        if self.format == "openai":
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            return {
                "url": self.endpoint,
                "headers": headers,
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                },
            }
        if self.format == "anthropic":
            return {
                "url": self.endpoint,
                "headers": {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                "body": {
                    "model": self.model,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": TEMPERATURE,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        if self.format == "google":
            version = self.cfg.get("google_api_version", "v1beta")
            base = self.endpoint.rstrip("/")
            # The stored endpoint may already name a version (the default is
            # .../v1beta); strip it so the configured version wins.
            for suffix in ("/v1beta", "/v1"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            return {
                "url": f"{base}/{version}/models/{self.model}:generateContent",
                "headers": {},
                "params": {"key": self.api_key},
                "body": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": MAX_OUTPUT_TOKENS,
                        "candidateCount": 1,
                        "topP": 1,
                        "topK": 1,
                    },
                },
            }
        raise LLMError(f"Unknown API format: {self.format}")

    # -- per-format execution ------------------------------------------------

    def _post(self, req: dict) -> requests.Response:
        return requests.post(
            req["url"],
            json=req["body"],
            headers=req["headers"],
            params=req.get("params"),
            timeout=self.timeout,
        )

    def _query_ollama(self, prompt: str) -> str:
        resp = self._post(self.build_request(prompt))
        self._raise_for_status(resp)
        return resp.json().get("response", "")

    def _query_openai(self, prompt: str) -> str:
        resp = self._post(self.build_request(prompt))
        self._raise_for_status(resp)
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from None

    def _query_anthropic(self, prompt: str) -> str:
        resp = self._post(self.build_request(prompt))
        self._raise_for_status(resp)
        data = resp.json()
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from None

    def _query_google(self, prompt: str) -> str:
        resp = self._post(self.build_request(prompt))
        if resp.status_code == 404:
            # v1beta <-> v1 fallback, persisted when it works
            # (port of background.js:1682-1708).
            current = self.cfg.get("google_api_version", "v1beta")
            other = "v1" if current == "v1beta" else "v1beta"
            log.info("Gemini %s returned 404, retrying with %s", current, other)
            self.cfg["google_api_version"] = other
            resp = self._post(self.build_request(prompt))
            if resp.ok and self.on_config_update:
                self.on_config_update(google_api_version=other)
        self._raise_for_status(resp)
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from None

    def _raise_for_status(self, resp: requests.Response) -> None:
        if resp.ok:
            return
        detail = ""
        try:
            body = resp.json()
            detail = (
                body.get("error", {}).get("message")
                if isinstance(body.get("error"), dict)
                else body.get("error")
            ) or ""
        except Exception:
            detail = resp.text[:200]
        hints = {
            401: "check your API key",
            403: "API key lacks access",
            404: "model or endpoint not found",
            429: "rate limit exceeded — wait and retry",
        }
        hint = hints.get(resp.status_code)
        msg = f"HTTP {resp.status_code}"
        if detail:
            msg += f": {detail}"
        if hint:
            msg += f" ({hint})"
        raise LLMError(msg)


def client_from_config(cfg: dict) -> LLMClient:
    """Build a client with the API key resolved from the keyring (or config
    fallback)."""
    api_key = ""
    provider = get_provider(cfg["provider"])
    if provider["needs_key"] or cfg.get("api_key_plaintext"):
        try:
            from .. import secrets
            api_key = secrets.get_api_key(cfg["provider"]) or ""
        except Exception:
            api_key = ""
        if not api_key:
            api_key = cfg.get("api_key_plaintext") or ""

    def persist(**changes):
        from .. import config
        config.update(**changes)

    return LLMClient(cfg, api_key=api_key, on_config_update=persist)

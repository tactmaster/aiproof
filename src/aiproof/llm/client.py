"""LLM client: builds requests for the four API formats, non-streaming,
temperature 0.1, max 2000 output tokens (shapes ported from the extension's
queryLocalProvider/queryRemoteProvider in background/background.js).
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Callable, Optional

import requests

from .. import context
# Re-exported for existing importers (orchestrator, cli, tests, evals).
from .errors import LLMError, TextTooLongError  # noqa: F401
from .postprocess import (
    enforce_formatting_preservation_ex,
    full_clean,
    preserve_casual_style,
    looks_like_wrapper_only,
    proofread_line_by_line,
    proofread_segments,
)

# Don't burn one model call per line on huge selections.
MAX_SALVAGE_LINES = 40
from .edits import apply_edits, looks_like_rewrite, parse_edits
from .prompts import EDIT_SYSTEM_PROMPT, build_edit_prompt, build_proofread_prompt
from .providers import get_provider

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 2000
TEMPERATURE = 0.1
# Edit-list mode: deterministic, and the output is bounded by the number of
# mistakes, not the text length.
EDITS_TEMPERATURE = 0.0
EDITS_MAX_OUTPUT_TOKENS = 768
OUTPUT_MODES = ("rewrite", "edits")
PROMPT_STYLES = ("full", "minimal")


def edits_max_tokens(text: str) -> int:
    """Token budget for an edit list: ~6 tokens per fix plus headroom. Bounds
    a model that ignores the format and rewrites everything to seconds
    (45 chars -> 63 tokens, 330 -> 158, 10k -> 768) instead of 44 s."""
    return min(EDITS_MAX_OUTPUT_TOKENS, 48 + len(text) // 3)


# The Claude Code CLI is usually a per-user install that isn't on a daemon's
# PATH, so look in the standard spots too.
_CLAUDE_CLI_PATHS = (
    "~/.local/bin/claude",
    "~/.claude/local/claude",
    "/usr/local/bin/claude",
)

# Replaces Claude Code's agentic system prompt: we want a text tool, not a
# coding assistant that might reach for file tools.
CLAUDE_CLI_SYSTEM = (
    "You are a text proofreading tool, not a conversational assistant and "
    "not a coding agent. Follow the instructions in the message exactly and "
    "output only the resulting text, with no preamble, commentary, or tool use."
)


def openai_chat_url(endpoint: str) -> str:
    """Full chat-completions URL for an OpenAI-compatible endpoint.

    Local servers (Lemonade, LM Studio, llama.cpp, vLLM, LiteLLM) document a
    *base* URL like http://host:8020/v1, and pasting that in used to POST
    straight to /v1 and get a 404. Accept either form, the way the Ollama
    branch already accepts a bare host."""
    base = endpoint.rstrip("/")
    if not base:
        return base
    if base.endswith("/chat/completions"):
        return base
    if re.search(r"/v\d+(beta)?$", base):      # .../v1, .../v1beta
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def openai_models_url(endpoint: str) -> str:
    """The /models listing URL that pairs with an OpenAI-compatible endpoint."""
    chat = openai_chat_url(endpoint)
    return chat[: -len("/chat/completions")] + "/models"


def find_claude_cli(configured: str = "") -> "str | None":
    """Path to the Claude Code CLI, or None when it isn't installed."""
    # A URL here means the field was carried over from an HTTP provider;
    # treat it as "not configured" rather than failing with a confusing
    # "CLI not found" for a path that was never a path.
    if configured and not re.match(r"^[a-z][a-z0-9+.-]*://", configured):
        path = os.path.expanduser(configured)
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    found = shutil.which("claude")
    if found:
        return found
    for candidate in _CLAUDE_CLI_PATHS:
        path = os.path.expanduser(candidate)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


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
        # Learned domain vocabulary/summary spliced into the prompt; ([], None)
        # unless context_aware is on and a cached profile exists.
        self.context_words, self.context_summary = context.load_words_and_summary(cfg)
        # Per-application register ("this is a Slack message"), set by the
        # orchestrator from the focused window before the proofread starts.
        self.app_setting = cfg.get("app_setting") or None
        # Chat registers keep lowercase starts / missing full stops;
        # prompts alone do not hold a 3B model to that.
        self.app_casual = bool(cfg.get("app_casual"))
        # "edits": the model lists wrong => right pairs that are applied
        # deterministically (fewer output tokens, layout can't break);
        # "rewrite": the model returns the whole corrected text.
        self.output_mode = cfg.get("output_mode") or "rewrite"
        if self.output_mode not in OUTPUT_MODES:
            log.warning("unknown output_mode %r; using rewrite", self.output_mode)
            self.output_mode = "rewrite"
        # "full": instructions + examples in every request; "minimal": bare
        # text with EDIT_SYSTEM_PROMPT as the system turn (fine-tuned models).
        self.prompt_style = cfg.get("prompt_style") or "full"
        if self.prompt_style not in PROMPT_STYLES:
            log.warning("unknown prompt_style %r; using full", self.prompt_style)
            self.prompt_style = "full"
        # Which pipeline path produced the last proofread() result:
        # "ok" | "repaired" | "line-by-line" | "segments" | "fallback"
        # | "edits" | "edits-partial" (some edits were rejected)
        self.last_path = ""
        # True when edit mode produced nothing usable and the rewrite prompt
        # was used instead (visible in the CLI/eval output for tuning).
        self.last_edits_fallback = False

    # -- public API ---------------------------------------------------------

    def proofread(self, text: str) -> tuple[str, float]:
        """Return (corrected_text, elapsed_seconds)."""
        if len(text) > self.max_chars:
            log.info("selection too long: %d > %d characters",
                     len(text), self.max_chars)
            raise TextTooLongError(len(text), self.max_chars)
        start = time.monotonic()
        self.last_edits_fallback = False
        result = None
        if self.output_mode == "edits":
            result = self._proofread_edits(text)
            if result is None:
                self.last_edits_fallback = True
        if result is None:
            response = self.query(build_proofread_prompt(
                text,
                context_words=self.context_words,
                context_summary=self.context_summary,
                app_setting=self.app_setting,
            ))
            result = self._finish_rewrite(text, response)
        final, path = result
        self.last_path = path
        log.info(
            "proofread done: path=%s, %d chars, %d lines, %.1fs",
            path, len(text), text.count("\n") + 1, time.monotonic() - start,
        )
        return final, time.monotonic() - start

    def _finish_rewrite(self, text: str, response: str) -> tuple[str, str]:
        """The full-rewrite pipeline on a model response: cleanup, guards,
        repair retry, line/segment salvage. Returns (final_text, path)."""
        cleaned = full_clean(response, text)
        if self.app_casual:
            cleaned = preserve_casual_style(cleaned, text)
        if not cleaned:
            log.warning("model returned an empty response")
            raise LLMError("Model returned an empty response")
        if looks_like_wrapper_only(cleaned, text):
            log.warning("model returned only boilerplate: %r", cleaned[:60])
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
            else:
                log.info("line-by-line salvage made no changes; "
                         "keeping original (path=fallback)")
        elif path == "fallback" and line_count == 1:
            log.info(
                "whole-text correction kept getting rejected; "
                "salvaging sentence segments"
            )
            salvaged = proofread_segments(text, self._correct_line)
            if salvaged != text:
                final, path = salvaged, "segments"
            else:
                log.info("segment salvage made no changes; "
                         "keeping original (path=fallback)")
        elif path == "fallback":
            log.info(
                "layout kept breaking but %d lines exceeds the %d-line "
                "salvage limit; returning original", line_count, MAX_SALVAGE_LINES,
            )
        return final, path

    def _proofread_edits(self, text: str):
        """Edit-list mode. Returns (final_text, path), or None when the model
        produced nothing usable and the rewrite prompt should be tried."""
        prompt = build_edit_prompt(
            text,
            context_words=self.context_words,
            context_summary=self.context_summary,
            app_setting=self.app_setting,
            style=self.prompt_style,
        )
        system = EDIT_SYSTEM_PROMPT if self.prompt_style == "minimal" else None
        response = self.query(prompt, max_tokens=edits_max_tokens(text),
                              temperature=EDITS_TEMPERATURE, system=system)
        parsed = parse_edits(response)
        if not parsed.edits:
            if parsed.no_changes:
                log.info("edit mode: model reports no changes")
                return text, "edits"
            cleaned = full_clean(response, text)
            if cleaned and looks_like_rewrite(text, cleaned):
                log.info("edit mode: model answered with a full rewrite; "
                         "using it (no second call)")
                self.last_edits_fallback = True
                return self._finish_rewrite(text, response)
            log.info("edit mode: no parsable edits (%d noise lines); "
                     "falling back to the rewrite prompt", parsed.noise_lines)
            return None
        result = apply_edits(text, parsed.edits)
        for edit, reason in result.rejected:
            log.info("edit rejected (%s): %r => %r",
                     reason, edit.wrong[:40], edit.right[:40])
        if not result.applied:
            log.info("edit mode: all %d edits rejected; falling back to the "
                     "rewrite prompt", len(parsed.edits))
            return None
        final = result.text
        if self.app_casual:
            final = preserve_casual_style(final, text)
        log.debug("edit mode: applied %d edit(s), rejected %d",
                  len(result.applied), len(result.rejected))
        return final, ("edits-partial" if result.rejected else "edits")

    def _correct_line(self, line: str) -> str | None:
        """One line through the model + cleanup; None means keep the
        original line."""
        try:
            response = self.query(build_proofread_prompt(
                line,
                context_words=self.context_words,
                context_summary=self.context_summary,
                app_setting=self.app_setting,
            ))
        except LLMError as e:
            log.warning("line correction failed: %s", e)
            return None
        cleaned = full_clean(response, line)
        if self.app_casual:
            cleaned = preserve_casual_style(cleaned, line)
        if not cleaned or looks_like_wrapper_only(cleaned, line):
            log.debug("line correction rejected (empty or boilerplate); "
                      "keeping original line")
            return None
        return cleaned

    def query(self, prompt: str, **options) -> str:
        """One model call. ``options`` (max_tokens, temperature, system) go
        to the request builder; omitted or None keeps the defaults."""
        options = {k: v for k, v in options.items() if v is not None}
        try:
            return self._dispatch(prompt, **options)
        except requests.ConnectionError as e:
            # Caught first: ConnectTimeout subclasses both ConnectionError and
            # Timeout, and a dead host is "unreachable", not "slow".
            log.debug("cannot reach %s (%s)", self.endpoint, type(e).__name__)
            raise LLMError(f"Cannot reach {self.endpoint}: connection failed") from e
        except requests.Timeout:
            log.debug("read timed out after %ss", self.timeout)
            raise LLMError(f"Request timed out after {self.timeout}s") from None

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
        """Available model names, for Ollama and any OpenAI-compatible server
        (they all expose /v1/models). Other formats have no listing API."""
        if self.format == "ollama":
            base = self.endpoint.rstrip("/")
            resp = requests.get(f"{base}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        if self.format == "openai":
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            resp = requests.get(openai_models_url(self.endpoint),
                                headers=headers, timeout=10)
            if resp.status_code in (401, 403):
                raise LLMError(
                    "The server rejected the request (HTTP "
                    f"{resp.status_code}) — set this provider's API key."
                )
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", data if isinstance(data, list) else [])
            return [m.get("id") or m.get("name") for m in models if isinstance(m, dict)]
        raise LLMError(
            f"Listing models is not supported for {self.format} providers"
        )

    # -- request builders ---------------------------------------------------

    def _dispatch(self, prompt: str, **options) -> str:
        if self.format == "ollama":
            return self._query_ollama(self.build_request(prompt, **options))
        if self.format == "openai":
            return self._query_openai(self.build_request(prompt, **options))
        if self.format == "anthropic":
            return self._query_anthropic(self.build_request(prompt, **options))
        if self.format == "google":
            return self._query_google(lambda: self.build_request(prompt, **options))
        if self.format == "claude_cli":
            return self._query_claude_cli(self.build_request(prompt, **options))
        raise LLMError(f"Unknown API format: {self.format}")

    def build_request(self, prompt: str, *, max_tokens=None, temperature=None,
                      system=None) -> dict:
        """Request description (url/headers/body) — used by tests and debug.
        ``system`` adds a system turn (edit mode with prompt_style=minimal);
        the model's own Modelfile SYSTEM applies when it is omitted."""
        if max_tokens is None:
            max_tokens = MAX_OUTPUT_TOKENS
        if temperature is None:
            temperature = TEMPERATURE
        if self.format == "ollama":
            base = self.endpoint.rstrip("/")
            url = base if "/api/" in base else f"{base}/api/generate"
            req = {
                "url": url,
                "headers": {},
                "body": {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    # Keep the model resident between proofreads — a cold
                    # load costs 9-40s, dwarfing the correction itself.
                    "keep_alive": self.cfg.get("ollama_keep_alive", "30m"),
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            }
            if system:
                req["body"]["system"] = system
            return req
        if self.format == "openai":
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            messages = [{"role": "user", "content": prompt}]
            if system:
                messages.insert(0, {"role": "system", "content": system})
            return {
                "url": openai_chat_url(self.endpoint),
                "headers": headers,
                "body": {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            }
        if self.format == "anthropic":
            req = {
                "url": self.endpoint,
                "headers": {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                "body": {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
            if system:
                req["body"]["system"] = system
            return req
        if self.format == "google":
            version = self.cfg.get("google_api_version", "v1beta")
            base = self.endpoint.rstrip("/")
            # The stored endpoint may already name a version (the default is
            # .../v1beta); strip it so the configured version wins.
            for suffix in ("/v1beta", "/v1"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            req = {
                "url": f"{base}/{version}/models/{self.model}:generateContent",
                "headers": {},
                "params": {"key": self.api_key},
                "body": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": max_tokens,
                        "candidateCount": 1,
                        "topP": 1,
                        "topK": 1,
                    },
                },
            }
            if system:
                req["body"]["systemInstruction"] = {"parts": [{"text": system}]}
            return req
        if self.format == "claude_cli":
            # --strict-mcp-config skips loading the user's MCP servers, which
            # measured ~2.7s of the CLI's startup. Temperature/max_tokens are
            # not exposed by the CLI; the prompt does that work instead.
            binary = find_claude_cli(self.cfg.get("endpoint") or "")
            return {
                "binary": binary,
                "argv": [
                    binary or "claude",
                    "--print",
                    "--model", self.model,
                    "--strict-mcp-config",
                    "--system-prompt", system or CLAUDE_CLI_SYSTEM,
                ],
                "stdin": prompt,
            }
        raise LLMError(f"Unknown API format: {self.format}")

    # -- per-format execution ------------------------------------------------

    def _query_claude_cli(self, req: dict) -> str:
        """Run the locally installed Claude Code CLI in print mode. Uses the
        machine's existing Claude subscription, so there is no API key — but
        the text does leave the device."""
        if not req["binary"]:
            raise LLMError(
                "Claude CLI not found. Install Claude Code, or set the "
                "endpoint to the full path of the 'claude' binary."
            )
        try:
            proc = subprocess.run(
                req["argv"],
                input=req["stdin"].encode("utf-8"),
                capture_output=True,
                timeout=self.timeout,
                # Neutral directory: a project CLAUDE.md would otherwise be
                # pulled into the context of every proofread.
                cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired:
            log.debug("claude CLI timed out after %ss", self.timeout)
            raise LLMError(f"Claude CLI timed out after {self.timeout}s") from None
        except OSError as e:
            raise LLMError(f"Could not run the Claude CLI: {e}") from e
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", "replace").strip()
            log.warning("claude CLI exited %s: %s", proc.returncode, stderr[:200])
            hint = ""
            lowered = stderr.lower()
            if any(marker in lowered for marker in
                   ("login", "log in", "logged in", "auth", "api key",
                    "credential", "subscription")):
                hint = " — run 'claude' once in a terminal to sign in"
            raise LLMError(
                f"Claude CLI failed (exit {proc.returncode}): "
                f"{stderr[:200] or 'no error output'}{hint}"
            )
        return proc.stdout.decode("utf-8", "replace")

    def _post(self, req: dict) -> requests.Response:
        # Short connect timeout so an unreachable host fails over to the
        # fallback provider in seconds, not after the full read timeout.
        return requests.post(
            req["url"],
            json=req["body"],
            headers=req["headers"],
            params=req.get("params"),
            timeout=(5, self.timeout),
        )

    def _query_ollama(self, req: dict) -> str:
        resp = self._post(req)
        self._raise_for_status(resp)
        return resp.json().get("response", "")

    def _query_openai(self, req: dict) -> str:
        resp = self._post(req)
        self._raise_for_status(resp)
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from None

    def _query_anthropic(self, req: dict) -> str:
        resp = self._post(req)
        self._raise_for_status(resp)
        data = resp.json()
        try:
            return data["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from None

    def _query_google(self, build_request) -> str:
        """``build_request`` is a thunk: the v1beta <-> v1 fallback below
        needs to rebuild the URL after switching versions."""
        resp = self._post(build_request())
        if resp.status_code == 404:
            # v1beta <-> v1 fallback, persisted when it works
            # (port of background.js:1682-1708).
            current = self.cfg.get("google_api_version", "v1beta")
            other = "v1" if current == "v1beta" else "v1beta"
            log.info("Gemini %s returned 404, retrying with %s", current, other)
            self.cfg["google_api_version"] = other
            resp = self._post(build_request())
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


def proofread_with_fallback(cfg: dict, text: str, on_fallback=None,
                            _client_factory=None):
    """Proofread with the configured provider; if it fails for any reason
    other than the text being too long, retry once with the configured
    fallback provider.

    Returns (corrected, elapsed_seconds, client, used_fallback). on_fallback,
    if given, is called with the primary's error before the retry."""
    factory = _client_factory or client_from_config
    client = factory(cfg)
    try:
        corrected, elapsed = client.proofread(text)
        return corrected, elapsed, client, False
    except TextTooLongError:
        # The fallback cfg inherits max_chars, so retrying cannot help.
        log.info("selection too long; not trying the fallback provider")
        raise
    except LLMError as primary_error:
        fallback = cfg.get("fallback") or {}
        if not (fallback.get("endpoint") or fallback.get("provider")):
            log.info("no usable fallback configured (%r); giving up", fallback or None)
            raise
        log.warning("primary provider failed (%s); trying fallback %s",
                    primary_error, fallback)
        if on_fallback:
            on_fallback(primary_error)
        fb_cfg = dict(cfg)
        for key in ("provider", "endpoint", "model"):
            if fallback.get(key):
                fb_cfg[key] = fallback[key]
        fb_client = factory(fb_cfg)
        try:
            corrected, elapsed = fb_client.proofread(text)
        except LLMError as fb_error:
            log.warning("fallback provider also failed: %s", fb_error)
            raise
        return corrected, elapsed, fb_client, True


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

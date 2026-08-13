"""Shared fixtures for the flow-test suite (tests/test_flow_*.py).

Seam rules
----------
- Orchestrator-level tests cut at ``aiproof.orchestrator.proofread_with_fallback``
  (the name is imported into the orchestrator module, so it is patched there)
  via the ``fake_llm`` fixture.
- Failover-mechanics tests call ``proofread_with_fallback(..., _client_factory=…)``
  directly with FakeClient-style fakes (see tests/test_fallback.py).
- Full-chain integration tests patch ``aiproof.llm.client.client_from_config``
  (the default factory, resolved from module globals at call time).
- CLI tests patch ``aiproof.llm.client.proofread_with_fallback`` (cli imports it
  lazily) and ALWAYS patch ``aiproof.config.load`` so no test reads
  ~/.config/aiproof/config.json.
- ``aiproof.clipboard`` / ``aiproof.keystroke`` are patched function-by-function
  on the real modules (never via sys.modules), so identity checks like
  ``text is clipboard.NON_TEXT`` keep working.

Flow report
-----------
Tests marked ``@pytest.mark.flow`` get their "aiproof" log records captured;
pytest_sessionfinish writes a route-by-route report to build/flow-report.md.
"""

import io
import logging
import sys
import time
import types
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiproof import clipboard as clip_mod
from aiproof import keystroke as key_mod
from aiproof.config import DEFAULTS


def make_cfg(**over):
    cfg = dict(DEFAULTS)
    cfg.update(over)
    return cfg


# -- fakes --------------------------------------------------------------------


class FakeNotifier:
    def __init__(self):
        self.notes = []  # (summary, body)
        self.chain_clears = 0

    def notify(self, summary, body="", **kwargs):
        self.notes.append((summary, body))

    def clear_replace_chain(self):
        self.chain_clears += 1

    @property
    def summaries(self):
        return [s for s, _ in self.notes]

    @property
    def last(self):
        return self.notes[-1]


class FakeClipboard:
    """Stateful stand-in whose methods are patched onto the real
    aiproof.clipboard module, one function at a time."""

    def __init__(self):
        self.calls = []  # ("save",), ("set_text", txt), ("restore", state)...
        self.saved_state = clip_mod.ClipboardState(
            "text", "text/plain;charset=utf-8", b"previous"
        )
        self.poll_results = deque()  # str | None | clip_mod.NON_TEXT per poll
        self.primary_text = None
        self.save_error = None
        self.set_text_error = None

    def save(self):
        self.calls.append(("save",))
        if self.save_error:
            raise self.save_error
        return self.saved_state

    def clear(self):
        self.calls.append(("clear",))

    def restore(self, state):
        self.calls.append(("restore", state))

    def set_text(self, text):
        self.calls.append(("set_text", text))
        if self.set_text_error:
            raise self.set_text_error

    def get_text(self):
        self.calls.append(("get_text",))
        return None

    def get_primary_text(self):
        self.calls.append(("get_primary_text",))
        return self.primary_text

    def poll_for_text(self, timeout=1.5, interval=0.1):
        self.calls.append(("poll_for_text", timeout))
        return self.poll_results.popleft() if self.poll_results else None

    @property
    def call_names(self):
        return [c[0] for c in self.calls]


class FakeKeystroke:
    def __init__(self):
        self.available_flag = True
        self.calls = []
        self.raise_on = set()  # e.g. {"ctrl_v"} -> RuntimeError

    def _rec(self, name):
        self.calls.append(name)
        if name in self.raise_on:
            raise RuntimeError(f"ydotool failed: scripted {name}")

    def available(self):
        self.calls.append("available")
        return self.available_flag

    def release_modifiers(self):
        self._rec("release_modifiers")

    def ctrl_c(self):
        self._rec("ctrl_c")

    def ctrl_v(self):
        self._rec("ctrl_v")


class FakeLLM:
    """Stands in for aiproof.orchestrator.proofread_with_fallback. Returns
    (corrected, elapsed, client-with-last_path, used_fallback)."""

    def __init__(self):
        self.corrected = "corrected text"
        self.path = "ok"
        self.used_fallback = False
        self.elapsed = 0.3
        self.error = None          # raised instead of returning
        self.primary_error = None  # fires on_fallback, then used_fallback=True
        self.calls = []            # texts received

    def __call__(self, cfg, text, on_fallback=None, _client_factory=None):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        if self.primary_error is not None:
            if on_fallback:
                on_fallback(self.primary_error)
            self.used_fallback = True
        return (self.corrected, self.elapsed,
                SimpleNamespace(last_path=self.path), self.used_fallback)


# -- fixtures -----------------------------------------------------------------


@pytest.fixture
def notifier():
    return FakeNotifier()


@pytest.fixture
def fake_clipboard(monkeypatch):
    fake = FakeClipboard()
    for name in ("save", "clear", "restore", "set_text",
                 "get_text", "get_primary_text", "poll_for_text"):
        monkeypatch.setattr(clip_mod, name, getattr(fake, name))
    return fake


@pytest.fixture
def fake_keystroke(monkeypatch):
    fake = FakeKeystroke()
    for name in ("available", "release_modifiers", "ctrl_c", "ctrl_v"):
        monkeypatch.setattr(key_mod, name, getattr(fake, name))
    return fake


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("aiproof.orchestrator.proofread_with_fallback", fake)
    return fake


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


@pytest.fixture
def orch(notifier, fake_clipboard, fake_keystroke, fake_llm, no_sleep):
    """Factory for an Orchestrator wired to every fake; .states records
    on_state transitions."""
    from aiproof.orchestrator import Orchestrator

    def build(**cfg_over):
        states = []
        o = Orchestrator(make_cfg(**cfg_over), notifier, on_state=states.append)
        o.states = states
        return o

    return build


@pytest.fixture
def fake_stdin(monkeypatch):
    def set_stdin(text):
        monkeypatch.setattr(sys, "stdin", io.StringIO(text))

    return set_stdin


@pytest.fixture
def gi_stub(monkeypatch):
    """Minimal gi/gi.repository stub so gi-importing modules (dbus_service,
    cli's non-oneshot trigger) can run headless."""

    class GLibError(Exception):
        def __init__(self, message="scripted"):
            super().__init__(message)
            self.message = message

    glib = types.SimpleNamespace(
        Error=GLibError,
        Variant=lambda sig, val: ("variant", sig, val),
        idle_add=lambda fn, *a: fn(*a),  # run callbacks inline
    )

    def bus_get_sync(*args):
        raise GLibError("no daemon")

    gio = types.SimpleNamespace(
        DBusNodeInfo=types.SimpleNamespace(
            new_for_xml=lambda xml: types.SimpleNamespace(interfaces=[object()])
        ),
        BusType=types.SimpleNamespace(SESSION=2),
        DBusCallFlags=types.SimpleNamespace(NONE=0),
        bus_get_sync=bus_get_sync,
    )

    gi = types.ModuleType("gi")
    gi.require_version = lambda *a, **k: None
    repository = types.ModuleType("gi.repository")
    repository.GLib = glib
    repository.Gio = gio
    gi.repository = repository

    monkeypatch.setitem(sys.modules, "gi", gi)
    monkeypatch.setitem(sys.modules, "gi.repository", repository)
    # Force any gi-importing aiproof module to re-import under the stub, and
    # drop the stub-built module again afterwards.
    monkeypatch.delitem(sys.modules, "aiproof.dbus_service", raising=False)
    yield types.SimpleNamespace(Gio=gio, GLib=glib)
    sys.modules.pop("aiproof.dbus_service", None)


# -- flow log report (build/flow-report.md) -----------------------------------

_FLOW_ENTRIES = []


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(
            f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        )


@pytest.fixture(autouse=True)
def _flow_log_capture(request):
    if request.node.get_closest_marker("flow") is None:
        yield
        return
    logger = logging.getLogger("aiproof")
    handler = _ListHandler()
    # The makereport hook for the "call" phase runs before fixture teardown,
    # so it reads the live handler off the item rather than a teardown copy.
    request.node._flow_handler = handler
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    marker = item.get_closest_marker("flow")
    if report.when != "call" or marker is None:
        return
    doc = (item.function.__doc__ or "").strip()
    handler = getattr(item, "_flow_handler", None)
    _FLOW_ENTRIES.append({
        "module": item.nodeid.split("::")[0],
        "route": marker.kwargs.get("route") or item.name,
        "desc": doc.splitlines()[0] if doc else "",
        "outcome": report.outcome.upper(),
        "lines": list(handler.lines) if handler else [],
    })


def pytest_sessionfinish(session, exitstatus):
    if not _FLOW_ENTRIES:
        return
    passed = sum(1 for e in _FLOW_ENTRIES if e["outcome"] == "PASSED")
    out = [
        "# aiproof flow report",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} — "
        f"{len(_FLOW_ENTRIES)} flow routes, {passed} passed.",
        "",
        "| Route | Result | Log lines |",
        "|---|---|---|",
    ]
    for e in _FLOW_ENTRIES:
        out.append(f"| {e['module']}::{e['route']} | {e['outcome']} | "
                   f"{len(e['lines'])} |")
    module = None
    for e in _FLOW_ENTRIES:
        if e["module"] != module:
            module = e["module"]
            out += ["", f"## {module}"]
        desc = f" — {e['desc']}" if e["desc"] else ""
        out += ["", f"### {e['outcome']} — {e['route']}{desc}", "", "```"]
        out += e["lines"] or ["(no log records)"]
        out += ["```"]
    path = Path(__file__).resolve().parent.parent / "build" / "flow-report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")

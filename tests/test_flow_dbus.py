"""Flow routes through DBusService method dispatch (daemon entry points),
run headless against the gi stub from conftest.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.flow


class FakeInvocation:
    def __init__(self):
        self.returned = None
        self.error = None

    def return_value(self, value):
        self.returned = ("value", value)

    def return_dbus_error(self, name, message):
        self.error = (name, message)


@pytest.fixture
def svc(gi_stub, monkeypatch):
    from aiproof import dbus_service

    # run Trigger handlers inline instead of on a worker thread
    monkeypatch.setattr(dbus_service.DBusService, "_spawn",
                        staticmethod(lambda target: target()))
    calls = []
    daemon = SimpleNamespace(
        orchestrator=SimpleNamespace(
            run_paste_flow=lambda: calls.append("paste"),
            run_clipboard_flow=lambda: calls.append("clipboard"),
        ),
        show_settings=lambda: calls.append("settings"),
        set_enabled=lambda enabled: calls.append(("enabled", enabled)),
        state="idle",
        quit=lambda: calls.append("quit"),
    )
    service = dbus_service.DBusService(daemon)
    service.test_calls = calls
    return service


def call(svc, method, params=None):
    inv = FakeInvocation()
    svc._on_method_call(None, ":1.1", "/io/github/edwatson/aiproof",
                        "io.github.edwatson.aiproof", method, params, inv)
    return inv


def test_trigger(svc, caplog):
    """dbus: Trigger dispatches the paste flow and returns immediately."""
    inv = call(svc, "Trigger")
    assert svc.test_calls == ["paste"]
    assert inv.returned == ("value", None)
    assert "DBus call: Trigger" in caplog.text


def test_trigger_clipboard_only(svc):
    """dbus: TriggerClipboardOnly dispatches the clipboard flow."""
    inv = call(svc, "TriggerClipboardOnly")
    assert svc.test_calls == ["clipboard"]
    assert inv.returned == ("value", None)


def test_show_settings(svc):
    """dbus: ShowSettings delegates to the daemon."""
    inv = call(svc, "ShowSettings")
    assert svc.test_calls == ["settings"]
    assert inv.returned == ("value", None)


def test_set_enabled(svc):
    """dbus: SetEnabled unpacks its boolean argument."""
    params = SimpleNamespace(unpack=lambda: (False,))
    inv = call(svc, "SetEnabled", params)
    assert svc.test_calls == [("enabled", False)]
    assert inv.returned == ("value", None)


def test_get_state(svc):
    """dbus: GetState returns the daemon state as a (s) variant."""
    inv = call(svc, "GetState")
    assert inv.returned == ("value", ("variant", "(s)", ("idle",)))


def test_quit(svc):
    """dbus: Quit replies first, then quits via the main loop."""
    inv = call(svc, "Quit")
    assert inv.returned == ("value", None)
    assert svc.test_calls == ["quit"]  # gi stub runs idle_add inline


def test_unknown_method(svc):
    """dbus: an unknown method returns the DBus UnknownMethod error."""
    inv = call(svc, "Bogus")
    assert inv.error == ("org.freedesktop.DBus.Error.UnknownMethod",
                         "No method Bogus")
    assert svc.test_calls == []

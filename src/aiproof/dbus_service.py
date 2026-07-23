"""DBus service: io.github.edwatson.aiproof on the session bus."""

import logging
import threading

from gi.repository import Gio, GLib

from . import DBUS_IFACE, DBUS_PATH

log = logging.getLogger(__name__)

INTROSPECTION_XML = f"""
<node>
  <interface name="{DBUS_IFACE}">
    <method name="Trigger"/>
    <method name="TriggerClipboardOnly"/>
    <method name="ShowSettings"/>
    <method name="SetEnabled">
      <arg type="b" name="enabled" direction="in"/>
    </method>
    <method name="GetState">
      <arg type="s" name="state" direction="out"/>
    </method>
    <method name="Quit"/>
  </interface>
</node>
"""


class DBusService:
    """Registers the object on an existing connection; delegates to the daemon."""

    def __init__(self, daemon):
        self.daemon = daemon
        self._node = Gio.DBusNodeInfo.new_for_xml(INTROSPECTION_XML)

    def register(self, connection: Gio.DBusConnection) -> None:
        connection.register_object(
            DBUS_PATH,
            self._node.interfaces[0],
            self._on_method_call,
            None,
            None,
        )

    def _on_method_call(self, connection, sender, path, iface, method, params,
                        invocation):
        log.debug("DBus call: %s", method)
        if method == "Trigger":
            self._spawn(self.daemon.orchestrator.run_paste_flow)
            invocation.return_value(None)
        elif method == "TriggerClipboardOnly":
            self._spawn(self.daemon.orchestrator.run_clipboard_flow)
            invocation.return_value(None)
        elif method == "ShowSettings":
            self.daemon.show_settings()
            invocation.return_value(None)
        elif method == "SetEnabled":
            (enabled,) = params.unpack()
            self.daemon.set_enabled(enabled)
            invocation.return_value(None)
        elif method == "GetState":
            invocation.return_value(GLib.Variant("(s)", (self.daemon.state,)))
        elif method == "Quit":
            invocation.return_value(None)
            GLib.idle_add(self.daemon.quit)
        else:
            invocation.return_dbus_error(
                "org.freedesktop.DBus.Error.UnknownMethod", f"No method {method}"
            )

    @staticmethod
    def _spawn(target) -> None:
        threading.Thread(target=target, daemon=True).start()

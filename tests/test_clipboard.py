import subprocess
from unittest import mock

from aiproof import clipboard


def completed(returncode=0, stdout=b""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=b"")


class TestSave:
    def test_empty_clipboard(self):
        with mock.patch.object(clipboard, "_run") as run:
            run.return_value = completed(returncode=1)
            state = clipboard.save()
        assert state.kind == "empty"

    def test_text_clipboard(self):
        def fake_run(args, input_bytes=None):
            if "--list-types" in args:
                return completed(stdout=b"text/plain;charset=utf-8\ntext/plain\n")
            return completed(stdout=b"hello")

        with mock.patch.object(clipboard, "_run", side_effect=fake_run):
            state = clipboard.save()
        assert state.kind == "text"
        assert state.data == b"hello"

    def test_image_clipboard(self):
        def fake_run(args, input_bytes=None):
            if "--list-types" in args:
                return completed(stdout=b"image/png\n")
            # Real wl-paste fails the text read when the clipboard holds an
            # image; save() only then falls back to listing types.
            if "text" in args:
                return completed(returncode=1)
            return completed(stdout=b"\x89PNG...")

        with mock.patch.object(clipboard, "_run", side_effect=fake_run):
            state = clipboard.save()
        assert state.kind == "image"
        assert state.mime == "image/png"

    def test_timeout_returns_empty(self):
        with mock.patch.object(
            clipboard, "_run",
            side_effect=subprocess.TimeoutExpired(["wl-paste"], 2),
        ):
            assert clipboard.save().kind == "empty"


class TestRestore:
    def test_restore_text(self):
        with mock.patch.object(clipboard, "_run_copy") as run:
            run.return_value = completed()
            clipboard.restore(clipboard.ClipboardState("text", "text/plain", b"hi"))
        args, kwargs = run.call_args
        assert args[0] == ["wl-copy"]
        assert kwargs["input_bytes"] == b"hi"

    def test_restore_empty_clears(self):
        with mock.patch.object(clipboard, "_run_copy") as run:
            run.return_value = completed()
            clipboard.restore(clipboard.ClipboardState("empty"))
        assert run.call_args[0][0] == ["wl-copy", "--clear"]


class TestPoll:
    def test_returns_text_when_available(self):
        with mock.patch.object(clipboard, "get_text", return_value="found"):
            assert clipboard.poll_for_text(timeout=0.3) == "found"

    def test_non_text_sentinel(self):
        with (
            mock.patch.object(clipboard, "get_text", return_value=None),
            mock.patch.object(clipboard, "list_types", return_value=["image/png"]),
        ):
            assert clipboard.poll_for_text(timeout=0.3) is clipboard.NON_TEXT

    def test_timeout_returns_none(self):
        with (
            mock.patch.object(clipboard, "get_text", return_value=None),
            mock.patch.object(clipboard, "list_types", return_value=[]),
        ):
            assert clipboard.poll_for_text(timeout=0.25, interval=0.05) is None


class TestSaveSpawnCount:
    """save() runs on every hotkey press and each wl-paste spawn costs
    200-600 ms on a busy desktop, so the common path must stay at one."""

    def test_text_clipboard_uses_a_single_spawn(self):
        calls = []

        def fake_run(args, input_bytes=None):
            calls.append(args)
            return completed(stdout=b"hello")

        with mock.patch.object(clipboard, "_run", side_effect=fake_run):
            state = clipboard.save()
        assert state.kind == "text" and state.data == b"hello"
        assert len(calls) == 1, f"expected 1 wl-paste spawn, got {calls}"
        assert "--list-types" not in calls[0]

    def test_empty_clipboard_falls_back_to_listing(self):
        def fake_run(args, input_bytes=None):
            if "--list-types" in args:
                return completed(returncode=1)
            return completed(returncode=1)

        with mock.patch.object(clipboard, "_run", side_effect=fake_run):
            assert clipboard.save().kind == "empty"

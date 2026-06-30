"""Unit tests for ``main._close_mcp_stdio``.

The MCP stdio shutdown step has three observable paths:

1. **Clean cancel.** The SDK honors the cancellation and the task ends
   well within the probe window. ``_force_close_stdin`` MUST NOT be
   called and the function returns without logging a warning. This is
   the production path: an MCP client closes its end of the pipe and
   the SDK's reader sees EOF immediately.

2. **TTY fast path.** The reader thread is parked in a blocking
   ``read(2)`` on a controlling terminal (interactive ``Ctrl+C``).
   The probe times out after ``_MCP_SHUTDOWN_TTY_PROBE_SECONDS``, the
   shutdown then closes stdin, and the task finishes unwinding. This
   is the path the graceful-stop fix exists to support.

3. **Deadlocked task.** The task refuses to acknowledge cancellation
   even after stdin is closed (a worst-case bug in the SDK or a
   wrapper). ``_close_mcp_stdio`` MUST log a warning and return
   normally so the rest of :func:`_shutdown_in_order` can still run;
   it MUST NOT propagate the timeout.

Every test is hermetic: ``stdin`` is never actually closed.
``_force_close_stdin`` is monkey-patched to a recorder so the tests
neither depend on POSIX behavior for ``close()`` on a file descriptor
held by another thread's blocking read nor leak file descriptors.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from project_knowledge_mcp import main as main_module
from project_knowledge_mcp.main import _close_mcp_stdio

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StdinCloseRecorder:
    """Stand-in for :func:`_force_close_stdin` that records invocations.

    Optionally fires a callback so a TTY-blocked test task can be
    released the moment the production code attempts to close stdin
    (modeling the kernel returning the blocked ``read(2)`` syscall).
    """

    def __init__(self, on_close: object = None) -> None:
        self.call_count = 0
        self._on_close = on_close

    def __call__(self) -> None:
        self.call_count += 1
        if callable(self._on_close):
            self._on_close()


@pytest.fixture
def shrink_shutdown_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the production timing constants so tests run in <1s.

    The constants are intentionally generous in production so a slow
    SDK shutdown is not falsely interrupted. The shutdown logic does
    not care about absolute durations; it only cares about relative
    ordering, so shrinking them here keeps the test suite fast while
    exercising the same control flow.
    """
    monkeypatch.setattr(main_module, "_MCP_SHUTDOWN_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(main_module, "_MCP_SHUTDOWN_TTY_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(
        main_module, "_MCP_SHUTDOWN_POST_CLOSE_TIMEOUT_SECONDS", 0.1
    )


# ---------------------------------------------------------------------------
# Path 1: clean cancel
# ---------------------------------------------------------------------------


async def test_clean_cancellation_does_not_touch_stdin(
    monkeypatch: pytest.MonkeyPatch,
    shrink_shutdown_timeouts: None,
) -> None:
    """A task that honors cancellation exits without closing stdin."""
    recorder = _StdinCloseRecorder()
    monkeypatch.setattr(main_module, "_force_close_stdin", recorder)

    async def clean_reader() -> None:
        # Hangs until cancelled; CancelledError propagates out so the
        # task ends in the ``cancelled`` state.
        await asyncio.sleep(60)

    task = asyncio.create_task(clean_reader())
    # Let the task reach its first await before the shutdown tries to
    # cancel it; otherwise the cancel fires before the task is even
    # scheduled and the test becomes a no-op.
    await asyncio.sleep(0)

    await _close_mcp_stdio(task)

    assert recorder.call_count == 0, "clean cancellation must not close stdin"
    assert task.done()


# ---------------------------------------------------------------------------
# Path 2: TTY fast path (graceful Ctrl+C)
# ---------------------------------------------------------------------------


async def test_tty_probe_times_out_then_closes_stdin_to_unblock_reader(
    monkeypatch: pytest.MonkeyPatch,
    shrink_shutdown_timeouts: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A TTY-blocked reader is freed by the stdin-close fallback.

    Models the interactive ``Ctrl+C`` scenario: cancellation alone
    does not unblock the reader, but the subsequent close releases
    the simulated blocked read so the task can unwind.
    """
    monkeypatch.setattr(main_module, "_stdin_is_tty", lambda: True)
    unblock = asyncio.Event()
    recorder = _StdinCloseRecorder(on_close=unblock.set)
    monkeypatch.setattr(main_module, "_force_close_stdin", recorder)

    async def tty_blocked_reader() -> None:
        # First cancellation: arrives while we're blocking in
        # ``unblock.wait()``; we cannot exit yet because the
        # "syscall" (modeled as the event) has not returned. Catch,
        # loop, and block again — the next ``await`` is no longer
        # cancellable because ``_close_mcp_stdio`` only issues one
        # cancel. The task exits cleanly when the production code
        # calls ``_force_close_stdin``, which sets ``unblock``.
        while True:
            try:
                await unblock.wait()
            except asyncio.CancelledError:
                continue
            return

    task = asyncio.create_task(tty_blocked_reader())
    await asyncio.sleep(0)

    with caplog.at_level(logging.INFO, logger=main_module.__name__):
        await _close_mcp_stdio(task)

    assert recorder.call_count == 1, "TTY path must close stdin exactly once"
    assert task.done()
    assert any(
        "closing stdin to unblock the reader" in record.message.lower()
        for record in caplog.records
    ), "INFO log line about closing stdin is missing"


# ---------------------------------------------------------------------------
# Path 3: deadlocked task
# ---------------------------------------------------------------------------


async def test_deadlocked_task_logs_warning_and_returns_normally(
    monkeypatch: pytest.MonkeyPatch,
    shrink_shutdown_timeouts: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A task that refuses to die does not poison the shutdown sequence.

    The function must still return so :func:`_shutdown_in_order` can
    run the remaining steps (abort the in-flight ingestion, join the
    scheduler, close the store). The test releases the deadlocked
    task explicitly at the end so pytest does not warn about a
    pending task at teardown.
    """
    monkeypatch.setattr(main_module, "_stdin_is_tty", lambda: False)
    recorder = _StdinCloseRecorder()  # no on_close: stdin "close" is a no-op
    monkeypatch.setattr(main_module, "_force_close_stdin", recorder)

    release = asyncio.Event()

    async def deadlocked_reader() -> None:
        # Swallow the cancellation that ``_close_mcp_stdio`` issues at
        # the top so the task does not exit during the probe window
        # nor during the post-close wait. The test releases the task
        # at the very end by setting ``release``.
        try:
            await asyncio.Event().wait()  # never set; cancellable
        except asyncio.CancelledError:
            pass
        # After the first cancellation is swallowed the task keeps
        # itself alive until the test releases it. A second
        # cancellation would re-raise out of ``release.wait`` and end
        # the task, but ``_close_mcp_stdio`` only cancels once.
        await release.wait()

    task = asyncio.create_task(deadlocked_reader())
    await asyncio.sleep(0)

    with caplog.at_level(logging.WARNING, logger=main_module.__name__):
        await _close_mcp_stdio(task)

    assert recorder.call_count == 1, (
        "non-TTY timeout must still attempt the stdin-close fallback"
    )
    assert any(
        "still alive" in record.message.lower()
        for record in caplog.records
    ), "WARNING about a still-alive task after stdin close is missing"

    # Release the deadlocked task so it ends cleanly before the test
    # tears down the event loop.
    release.set()
    await asyncio.wait_for(task, timeout=1.0)


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


def test_force_close_stdin_closes_the_stdin_file_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper closes whatever fd ``sys.stdin`` is currently pointing at.

    Patches ``os.close`` so the test does not actually close stdin
    (which would break pytest's own logging on every later test), and
    substitutes a fake :data:`sys.stdin` with a known ``fileno()``
    because pytest's capture machinery redirects stdin to a pseudo-
    file whose ``fileno`` is not callable.
    """
    closed_fds: list[int] = []
    monkeypatch.setattr(main_module.os, "close", closed_fds.append)

    class _FakeStdin:
        def fileno(self) -> int:
            return 9999

    monkeypatch.setattr(main_module.sys, "stdin", _FakeStdin())

    main_module._force_close_stdin()

    assert closed_fds == [9999]


@pytest.mark.parametrize("exc", [OSError("EBADF"), ValueError("closed")])
def test_force_close_stdin_swallows_documented_failures(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
) -> None:
    """A previously-closed or replaced stdin must not raise from shutdown.

    Both :class:`OSError` (closed fd) and :class:`ValueError`
    (``sys.stdin`` replaced by an object whose ``fileno()`` raises)
    are observed in test harnesses that redirect stdin to
    :data:`subprocess.DEVNULL` and then close it during teardown.
    """
    def fake_close(_fd: int) -> None:
        raise exc

    monkeypatch.setattr(main_module.os, "close", fake_close)

    # The bare call must not propagate the exception.
    main_module._force_close_stdin()


def test_stdin_is_tty_returns_false_when_isatty_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_stdin_is_tty`` treats a broken ``sys.stdin.isatty`` as "not TTY".

    Some test harnesses replace ``sys.stdin`` with an object whose
    ``isatty`` raises on a closed underlying stream; the helper must
    fall back to the conservative non-TTY production timing rather
    than letting the exception poison the shutdown path.
    """
    class _BrokenStdin:
        def isatty(self) -> bool:
            raise ValueError("closed")

    monkeypatch.setattr(main_module.sys, "stdin", _BrokenStdin())

    assert main_module._stdin_is_tty() is False

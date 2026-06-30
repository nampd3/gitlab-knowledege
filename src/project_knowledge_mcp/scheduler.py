"""Scheduler: triggers Ingestion_Jobs at the configured refresh interval.

This module implements the scheduled-refresh component described in the
design's "Scheduled refresh" section: when ``refresh.interval`` is
configured, a ``Scheduler`` periodically invokes
``IngestionCoordinator.start_full_refresh()`` (Requirement 8.3). If the
previous ``Ingestion_Job`` is still running when a tick fires, the
coordinator's single-flight gate raises
:class:`IngestionInProgressError` (Requirement 8.6); the scheduler logs
a single rejection line carrying the canonical
``"Ingestion_Job is already in progress"`` wording (the same wording the
MCP layer surfaces) plus a fragment naming the dropped tick, then
schedules the next tick anyway. Other unexpected exceptions raised by
``start_full_refresh`` are caught and logged with a stack trace; they
never kill the scheduler thread, so a transient analyzer or GitLab
failure cannot freeze periodic refreshes.

The implementation is a daemon thread driven by a ``threading.Event``:

- :meth:`Scheduler.start` spawns the thread and returns immediately.
- The thread loops, sleeping for ``refresh_interval`` between ticks.
  The sleep is implemented as a ``Event.wait`` so that
  :meth:`Scheduler.stop` can interrupt the wait and shut the thread
  down cleanly within milliseconds.
- :meth:`Scheduler.stop` sets the stop event and joins the thread.

The wall-clock and the sleep primitive are exposed as constructor
parameters so task 8.8's integration test can drive the scheduler with
a virtual clock without monkey-patching :mod:`time`. The
``sleep_until_stop`` seam returns a boolean: ``True`` when the stop
signal was raised during the wait (so the loop should exit), ``False``
when the timeout elapsed naturally (so the loop should fire the next
tick). The default implementation delegates to
``self._stop_event.wait(seconds)``.

Implements Requirement 8.3 (with Requirement 8.6 wording for
rejection logs).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from .errors import IngestionInProgressError

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import timedelta

    from .ingestion_coordinator import IngestionCoordinator


_LOGGER = logging.getLogger(__name__)


#: Log line emitted when a scheduled tick fires while a prior
#: ``Ingestion_Job`` is still running. The first half of the message is
#: the canonical Requirement 8.6 wording (matching
#: :attr:`IngestionInProgressError.MESSAGE`) so that operators searching
#: logs for the documented phrase find both the MCP-side and
#: scheduler-side surfaces. The second half names the dropped event so
#: the line is self-explanatory in isolation.
_REJECTED_TICK_LOG_MESSAGE: str = (
    f"{IngestionInProgressError.MESSAGE}; skipping scheduled tick"
)


class Scheduler:
    """Periodically invokes ``IngestionCoordinator.start_full_refresh``.

    The scheduler is a small daemon-thread + ``threading.Event`` loop
    that fires every ``refresh_interval`` until :meth:`stop` is called.
    Each tick calls ``coordinator.start_full_refresh()``; rejection by
    the single-flight gate (Requirement 8.6) and other exceptions are
    both swallowed after being logged so that one bad tick cannot
    silently freeze future ticks (Requirement 8.3).

    The thread is created daemonized so that an unstopped scheduler
    cannot prevent process exit, but :meth:`stop` is still the
    supported way to shut down because it joins the thread and gives
    any in-flight tick a chance to finish before returning.

    Args:
        coordinator: The :class:`IngestionCoordinator` whose
            ``start_full_refresh`` method is called every tick.
        refresh_interval: The wall-clock period between ticks. Must be
            strictly positive; the configuration validator already
            rejects intervals shorter than one minute, so this
            constructor only guards against the obvious zero/negative
            programmer-error case.
        clock: Optional monotonic clock used by tests (task 8.8) that
            want to assert the scheduler advances time correctly.
            Defaults to :func:`time.monotonic`. Production code does
            not need to supply this.
        sleep_until_stop: Optional sleep primitive. Called as
            ``sleep_until_stop(seconds)`` and must return ``True`` if
            the scheduler should stop (i.e. the stop signal arrived
            during the wait) or ``False`` if the wait completed
            naturally and the next tick should fire. Defaults to a
            wrapper around :meth:`threading.Event.wait` on the
            scheduler's internal stop event so that :meth:`stop` can
            interrupt sleeps cleanly. Tests may inject a virtual-clock
            implementation here.

    Raises:
        ValueError: When ``refresh_interval`` is not strictly positive.
    """

    def __init__(
        self,
        coordinator: IngestionCoordinator,
        refresh_interval: timedelta,
        *,
        clock: Callable[[], float] | None = None,
        sleep_until_stop: Callable[[float], bool] | None = None,
    ) -> None:
        interval_seconds = refresh_interval.total_seconds()
        if interval_seconds <= 0:
            # The config validator enforces ``>= 1 minute`` so this
            # branch only fires on programmer error (e.g. a unit test
            # constructing the scheduler directly with a zero
            # interval). A non-positive interval would spin the loop
            # without yielding, which is never the desired behavior.
            raise ValueError(
                f"refresh_interval must be strictly positive, got {refresh_interval!r}"
            )

        self._coordinator = coordinator
        self._interval_seconds = interval_seconds
        self._clock: Callable[[], float] = (
            clock if clock is not None else time.monotonic
        )
        self._stop_event = threading.Event()
        # Bind the default ``sleep_until_stop`` to the stop event after
        # the event exists so that tests injecting a custom callable
        # can also observe / advance the scheduler's notion of time
        # without our default leaking through.
        self._sleep_until_stop: Callable[[float], bool] = (
            sleep_until_stop
            if sleep_until_stop is not None
            else self._default_sleep_until_stop
        )
        # ``_lock`` guards :meth:`start` / :meth:`stop` so two callers
        # cannot race to spawn or join the worker thread.
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _default_sleep_until_stop(self, seconds: float) -> bool:
        """Default sleep: wait on the stop event with a timeout.

        ``Event.wait`` returns ``True`` when the event has been set
        (i.e. :meth:`stop` was called) and ``False`` on timeout. That
        matches the ``sleep_until_stop`` contract exactly.
        """

        return self._stop_event.wait(timeout=seconds)

    def start(self) -> None:
        """Spawn the scheduler thread.

        Returns immediately; the first tick fires after one
        ``refresh_interval`` of wall-clock time has elapsed (matching
        the design's "every time the configured refresh interval
        elapses" wording in Requirement 8.3 — i.e. the first refresh
        happens after one full interval, not at startup).

        Raises:
            RuntimeError: When the scheduler is already running. Stop
                it first via :meth:`stop` if you need to restart.
        """

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Scheduler is already running")
            # Reset the stop event in case a prior run left it set so
            # restarting after a clean stop is well-defined.
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                name="project-knowledge-mcp-scheduler",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def stop(self, *, timeout: float | None = None) -> None:
        """Signal the scheduler to stop and wait for the thread to exit.

        Idempotent: calling :meth:`stop` on a scheduler that was never
        started, or calling it twice, is a no-op.

        Args:
            timeout: Optional join timeout (seconds). ``None`` means
                wait indefinitely. The thread should exit within a
                handful of milliseconds because the stop event
                interrupts the inter-tick sleep, but a tick that is
                currently inside ``coordinator.start_full_refresh()``
                will run to completion before the thread can observe
                the stop signal.
        """

        with self._lock:
            thread = self._thread
            self._stop_event.set()

        if thread is None:
            return

        thread.join(timeout=timeout)

        with self._lock:
            # Only clear ``_thread`` if the join succeeded so that a
            # caller passing a too-short timeout can call ``stop``
            # again without leaking the reference.
            if not thread.is_alive():
                self._thread = None

    def is_running(self) -> bool:
        """Return ``True`` while the scheduler thread is alive."""

        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Main loop body. Sleeps then ticks until stopped.

        The order is "sleep, then tick" rather than "tick, then sleep"
        so the first refresh happens after one full interval has
        elapsed. The MCP_Server's startup sequence (design step 7
        of "Startup") performs an initial full refresh independently;
        the scheduler is responsible only for the periodic cadence
        thereafter.
        """

        while True:
            stopped = self._sleep_until_stop(self._interval_seconds)
            if stopped:
                return
            self._tick()

    def _tick(self) -> None:
        """Trigger one scheduled refresh.

        On :class:`IngestionInProgressError` (Requirement 8.6
        rejection from the single-flight gate), log the canonical
        rejection line and continue. On any other exception, log it
        with a stack trace and continue — the scheduler thread must
        never die from a tick failure or future ticks would silently
        stop firing.
        """

        try:
            self._coordinator.start_full_refresh()
        except IngestionInProgressError:
            # Requirement 8.6: a rejected start emits a log line and
            # leaves coordinator state and Knowledge_Store unchanged.
            # We use ``warning`` (rather than ``info``) so the line is
            # visible at the default log level operators run with.
            _LOGGER.warning(_REJECTED_TICK_LOG_MESSAGE)
        except Exception:
            # Defensive: any other exception (GitLabAuthError raised
            # because the token rotated, transient SQLite errors,
            # bugs in ``Project_Analyzer``) must not kill the
            # scheduler thread. Log with a stack trace and let the
            # next tick fire — operators will still see the failure
            # in logs and the periodic cadence stays intact
            # (Requirement 8.3).
            _LOGGER.exception("Scheduled refresh raised an unexpected exception")

    def now(self) -> float:
        """Return the scheduler's current monotonic time.

        Exposed so tests with a virtual clock can read the same time
        source the scheduler does. Production code rarely needs this.
        """

        return self._clock()


__all__ = ["Scheduler"]

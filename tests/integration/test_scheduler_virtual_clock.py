"""Integration tests for the Scheduler driven by a virtual clock.

These tests exercise :class:`project_knowledge_mcp.scheduler.Scheduler`
end-to-end -- the real daemon-thread loop, the real ``_tick`` body, the
real exception handlers -- but replace two seams with deterministic
test doubles so the suite is fast and free of wall-clock flakiness:

* The ``clock`` parameter is replaced with a :class:`VirtualClock`
  whose ``advance()`` method bumps a monotonic-style counter under a
  lock. Tests can read ``clock()`` to assert exactly how much
  simulated time elapsed.
* The ``sleep_until_stop`` parameter is replaced with a fake that
  advances the virtual clock by the requested duration and returns
  ``False`` for the first N calls (so N ticks fire) before returning
  ``True`` (the stop signal) so the scheduler thread exits naturally.

Together these let the tests assert:

* Requirement 8.3: a tick fires every ``refresh.interval`` and the
  cadence is preserved across many intervals.
* Requirement 8.6: when the underlying single-flight gate raises
  :class:`IngestionInProgressError`, the scheduler emits a single
  warning log line carrying the canonical Requirement 8.6 wording and
  schedules the next tick anyway -- the thread does not die on
  rejection.
* Robustness: any other unexpected exception raised by
  ``start_full_refresh`` is logged but does not kill the scheduler
  thread either; subsequent ticks still fire.

Implements Requirement 8.3 (with Requirement 8.6 wording for
rejection logs) -- task 8.8.
"""

from __future__ import annotations

import logging
import threading
from datetime import timedelta

import pytest

from project_knowledge_mcp.errors import IngestionInProgressError
from project_knowledge_mcp.scheduler import Scheduler

pytestmark = pytest.mark.integration


# Refresh interval used by every test in this module. Chosen as ten
# minutes (600 s) so that the per-tick advance value is large and
# unambiguous when comparing ``VirtualClock.now`` against expectations.
_REFRESH_INTERVAL: timedelta = timedelta(minutes=10)
_REFRESH_INTERVAL_SECONDS: float = _REFRESH_INTERVAL.total_seconds()

# Logger name the scheduler emits records on. The scheduler module
# uses ``logging.getLogger(__name__)`` so this string must match the
# module path exactly; capturing on this name keeps the test from
# accidentally swallowing unrelated records emitted from other modules
# during the test run.
_SCHEDULER_LOGGER: str = "project_knowledge_mcp.scheduler"

# The exact wording the scheduler emits when a scheduled tick collides
# with an already-running ``Ingestion_Job``. This duplicates the
# constant in :mod:`project_knowledge_mcp.scheduler` on purpose: the
# test asserts the canonical Requirement 8.6 phrase verbatim, so a
# refactor that reworded the log line should fail this test rather
# than silently track the change.
_REJECTED_TICK_LOG_MESSAGE: str = (
    "Ingestion_Job is already in progress; skipping scheduled tick"
)

# Bound for ``threading.Event.wait`` calls used to synchronize the
# test thread with the scheduler's daemon thread. Generous enough that
# CI variability never causes a false negative; small enough that a
# real bug (e.g. the scheduler thread deadlocking) surfaces quickly.
_THREAD_JOIN_TIMEOUT_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class VirtualClock:
    """A monotonic-style clock backed by a counter advanced explicitly.

    The clock value starts at ``0.0``; tests (and the
    :class:`FakeSleepUntilStop` injected into the scheduler) advance
    it by a known duration on each scheduled tick. Reads and writes
    are guarded by a lock because the scheduler's daemon thread reads
    via ``Scheduler.now()`` while the test thread may also read for
    assertions.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now = 0.0

    def __call__(self) -> float:
        """Return the current simulated time in seconds."""

        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        """Advance simulated time by ``seconds``."""

        with self._lock:
            self._now += seconds


class FakeSleepUntilStop:
    """A virtual-clock-driven replacement for ``Scheduler._default_sleep_until_stop``.

    Every call advances the supplied :class:`VirtualClock` by the
    requested duration and records that duration in
    :attr:`nap_durations`. The first ``allowed_naps`` calls return
    ``False`` (so the scheduler proceeds to fire a tick); subsequent
    calls return ``True``, which the scheduler interprets as the stop
    signal and breaks out of its run loop on. ``exited`` is set as
    soon as we return ``True`` so the test thread can wait on the
    daemon thread completing without a sleep.
    """

    def __init__(self, clock: VirtualClock, allowed_naps: int) -> None:
        if allowed_naps < 0:
            raise ValueError("allowed_naps must be non-negative")
        self._clock = clock
        self._allowed_naps = allowed_naps
        self._call_count = 0
        self.nap_durations: list[float] = []
        self.exited = threading.Event()

    def __call__(self, seconds: float) -> bool:
        self.nap_durations.append(seconds)
        self._clock.advance(seconds)
        self._call_count += 1
        if self._call_count > self._allowed_naps:
            self.exited.set()
            return True
        return False


class FakeIngestionCoordinator:
    """Records every ``start_full_refresh`` call; optionally raises on chosen ticks.

    ``raise_on_calls`` is the set of 1-indexed call numbers on which
    ``start_full_refresh`` should raise ``error``. Calls outside that
    set return normally. The :attr:`call_count` and :attr:`call_times`
    attributes let the tests assert both the cadence and the total
    number of triggered jobs.
    """

    def __init__(
        self,
        clock: VirtualClock,
        *,
        raise_on_calls: set[int] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._clock = clock
        self._raise_on_calls = raise_on_calls or set()
        self._error = error
        self.call_count = 0
        self.call_times: list[float] = []

    def start_full_refresh(self) -> None:
        self.call_count += 1
        self.call_times.append(self._clock())
        if self.call_count in self._raise_on_calls and self._error is not None:
            raise self._error


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scheduler_fires_tick_after_each_interval() -> None:
    """A tick fires after every ``refresh_interval`` of simulated time.

    The fake sleep returns ``False`` five times before signaling stop,
    so the scheduler should run ``start_full_refresh`` exactly five
    times. We also assert each nap consumed the configured interval
    and that the virtual clock advanced by the expected total --
    together these prove the scheduler's "sleep one interval, then
    tick" cadence (Requirement 8.3).
    """

    expected_ticks = 5
    clock = VirtualClock()
    coordinator = FakeIngestionCoordinator(clock)
    sleep = FakeSleepUntilStop(clock, allowed_naps=expected_ticks)

    scheduler = Scheduler(
        coordinator,  # type: ignore[arg-type]
        _REFRESH_INTERVAL,
        clock=clock,
        sleep_until_stop=sleep,
    )
    scheduler.start()
    try:
        signaled = sleep.exited.wait(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
        assert signaled, "Scheduler thread did not complete five ticks in time"
    finally:
        scheduler.stop(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    assert not scheduler.is_running(), (
        "Scheduler thread is still alive after stop(); the test "
        "would leak a daemon thread between tests"
    )
    assert coordinator.call_count == expected_ticks, (
        f"Scheduler should have fired {expected_ticks} ticks, "
        f"got {coordinator.call_count}"
    )
    # The scheduler calls sleep one extra time after the last tick
    # (that final call is the one that returns True and ends the
    # loop). So the total number of nap durations is ticks + 1.
    assert sleep.nap_durations == [_REFRESH_INTERVAL_SECONDS] * (expected_ticks + 1)
    # Each tick fired after exactly one full interval of simulated
    # time -- the cadence Requirement 8.3 mandates.
    assert coordinator.call_times == [
        _REFRESH_INTERVAL_SECONDS * (i + 1) for i in range(expected_ticks)
    ]
    # And the scheduler observes the same virtual time the test does.
    assert clock() == _REFRESH_INTERVAL_SECONDS * (expected_ticks + 1)
    assert scheduler.now() == _REFRESH_INTERVAL_SECONDS * (expected_ticks + 1)


def test_scheduler_skip_tick_when_previous_job_still_running(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the gate raises ``IngestionInProgressError``, the scheduler logs and continues.

    Requirement 8.6: a tick that fires while a prior ``Ingestion_Job``
    is still running is rejected with the canonical "Ingestion_Job is
    already in progress" message, and the scheduler MUST NOT die on
    that rejection -- the next tick still has to fire on schedule.

    We arrange the fake coordinator to raise
    :class:`IngestionInProgressError` on tick 3 only. After five total
    ticks the scheduler should have called ``start_full_refresh``
    five times (the rejection does not count as a missed call), and
    exactly one warning record should carry the canonical wording.
    """

    expected_ticks = 5
    rejected_tick = 3
    clock = VirtualClock()
    coordinator = FakeIngestionCoordinator(
        clock,
        raise_on_calls={rejected_tick},
        error=IngestionInProgressError(),
    )
    sleep = FakeSleepUntilStop(clock, allowed_naps=expected_ticks)

    scheduler = Scheduler(
        coordinator,  # type: ignore[arg-type]
        _REFRESH_INTERVAL,
        clock=clock,
        sleep_until_stop=sleep,
    )
    # Capture WARNING records from the scheduler's logger only --
    # this avoids accidentally also matching unrelated WARNING
    # records emitted by collaborator code under test.
    with caplog.at_level(logging.WARNING, logger=_SCHEDULER_LOGGER):
        scheduler.start()
        try:
            signaled = sleep.exited.wait(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
            assert signaled, "Scheduler thread did not complete five ticks in time"
        finally:
            scheduler.stop(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    # Subsequent ticks fired despite the rejection -- the scheduler
    # thread did not die on the in-progress error.
    assert coordinator.call_count == expected_ticks, (
        f"Scheduler should have attempted {expected_ticks} ticks even "
        f"after a rejection on tick {rejected_tick}, got "
        f"{coordinator.call_count}"
    )
    # Exactly one rejection record was emitted, with the canonical
    # Requirement 8.6 wording. Filter to the scheduler's logger so
    # we are not influenced by warnings from elsewhere.
    rejection_records = [
        record
        for record in caplog.records
        if record.name == _SCHEDULER_LOGGER
        and record.levelno == logging.WARNING
        and record.getMessage() == _REJECTED_TICK_LOG_MESSAGE
    ]
    assert len(rejection_records) == 1, (
        f"Expected exactly one rejection log line carrying the "
        f"Requirement 8.6 wording {_REJECTED_TICK_LOG_MESSAGE!r}; "
        f"got {[record.getMessage() for record in caplog.records]}"
    )
    # The rejected tick fires at the same cadence as the others -- the
    # rejection does not slip the schedule.
    assert coordinator.call_times == [
        _REFRESH_INTERVAL_SECONDS * (i + 1) for i in range(expected_ticks)
    ]


def test_scheduler_continues_after_unexpected_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unexpected exception on a tick is logged but does not kill the thread.

    The scheduler's ``_tick`` swallows non-``IngestionInProgressError``
    exceptions through a broad ``except Exception`` arm so that one
    bad tick (a transient analyzer bug, an unexpected SQLite error,
    a pyflakes-evading typo in upstream code) cannot silently freeze
    future ticks. This test injects a generic :class:`RuntimeError`
    on tick 2 of a five-tick run and asserts (a) all five ticks were
    attempted and (b) an ERROR-level record with a stack trace was
    emitted by the scheduler logger.
    """

    expected_ticks = 5
    failing_tick = 2
    raised_error = RuntimeError("synthetic upstream failure")
    clock = VirtualClock()
    coordinator = FakeIngestionCoordinator(
        clock,
        raise_on_calls={failing_tick},
        error=raised_error,
    )
    sleep = FakeSleepUntilStop(clock, allowed_naps=expected_ticks)

    scheduler = Scheduler(
        coordinator,  # type: ignore[arg-type]
        _REFRESH_INTERVAL,
        clock=clock,
        sleep_until_stop=sleep,
    )
    with caplog.at_level(logging.ERROR, logger=_SCHEDULER_LOGGER):
        scheduler.start()
        try:
            signaled = sleep.exited.wait(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
            assert signaled, "Scheduler thread did not complete five ticks in time"
        finally:
            scheduler.stop(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)

    # Every tick after the failing one still fired, proving the
    # scheduler thread is alive after the broad-exception handler
    # ran.
    assert coordinator.call_count == expected_ticks, (
        f"Scheduler should have attempted {expected_ticks} ticks even "
        f"after a generic exception on tick {failing_tick}, got "
        f"{coordinator.call_count}"
    )
    # Each tick still fired after exactly one full interval of
    # simulated time.
    assert coordinator.call_times == [
        _REFRESH_INTERVAL_SECONDS * (i + 1) for i in range(expected_ticks)
    ]
    # The unexpected exception was logged via ``_LOGGER.exception``,
    # which is an ERROR-level record carrying the original
    # ``RuntimeError`` as ``exc_info``. We assert both so a future
    # refactor that downgraded the level or dropped the traceback
    # would fail this test.
    error_records = [
        record
        for record in caplog.records
        if record.name == _SCHEDULER_LOGGER and record.levelno == logging.ERROR
    ]
    assert len(error_records) == 1, (
        f"Expected exactly one ERROR-level record from the scheduler "
        f"logger when start_full_refresh raises an unexpected "
        f"exception; got {[record.getMessage() for record in caplog.records]}"
    )
    record = error_records[0]
    assert record.exc_info is not None, (
        "The unexpected-exception log line should carry exc_info so "
        "operators see the stack trace"
    )
    assert record.exc_info[1] is raised_error, (
        "The logged exception should be the one raised by the "
        "coordinator, not a wrapped or different instance"
    )

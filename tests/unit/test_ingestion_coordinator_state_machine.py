"""Unit tests for the ``IngestionCoordinator`` single-flight state machine.

These tests cover the state-machine skeleton implemented in task 8.1:

* ``try_start`` performs the ``idle → running`` CAS and exposes
  ``snapshot_id``, ``trigger``, ``started_at`` while the job is running.
* A second concurrent start raises ``IngestionInProgressError`` with the
  canonical Requirement 8.6 message and leaves the coordinator state
  unchanged.
* ``complete`` and ``abort`` both release the running slot so the next
  start succeeds; the handle's context-manager protocol picks
  ``complete`` on clean exit and ``abort`` on exception.
* ``current_state`` returns idle/running with the documented fields and
  is read-only.
* Concurrent ``try_start`` from many threads always elects exactly one
  winner.

Property-based coverage of the stronger invariant (Property 12: at most
one Ingestion_Job in running state at any moment, rejected starts leave
state and store unchanged, idle starts always succeed) is task 8.6.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from project_knowledge_mcp.errors import IngestionInProgressError
from project_knowledge_mcp.ingestion_coordinator import (
    CoordinatorState,
    CoordinatorStatus,
    IngestionCoordinator,
    JobHandle,
)
from project_knowledge_mcp.models import SnapshotTrigger

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fresh coordinator: idle state
# ---------------------------------------------------------------------------


def test_fresh_coordinator_is_idle() -> None:
    coord = IngestionCoordinator()
    status = coord.current_state()
    assert status.state is CoordinatorState.IDLE
    assert status.snapshot_id is None
    assert status.trigger is None
    assert status.started_at is None
    assert coord.is_idle() is True


# ---------------------------------------------------------------------------
# try_start: success path
# ---------------------------------------------------------------------------


def test_try_start_transitions_to_running_with_supplied_metadata() -> None:
    coord = IngestionCoordinator()
    started_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)

    handle = coord.try_start(
        trigger=SnapshotTrigger.FULL,
        snapshot_id=42,
        started_at=started_at,
    )

    assert isinstance(handle, JobHandle)
    assert handle.snapshot_id == 42
    assert handle.trigger is SnapshotTrigger.FULL
    assert handle.started_at == started_at
    assert handle.is_active is True


def test_running_state_exposes_in_flight_metadata() -> None:
    coord = IngestionCoordinator()
    started_at = datetime(2025, 6, 1, tzinfo=UTC)

    coord.try_start(
        trigger=SnapshotTrigger.SCHEDULED,
        snapshot_id=7,
        started_at=started_at,
    )

    status = coord.current_state()
    assert status.state is CoordinatorState.RUNNING
    assert status.snapshot_id == 7
    assert status.trigger is SnapshotTrigger.SCHEDULED
    assert status.started_at == started_at
    assert coord.is_idle() is False


def test_started_at_defaults_to_now_utc() -> None:
    coord = IngestionCoordinator()
    before = datetime.now(UTC)
    handle = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=1)
    after = datetime.now(UTC)

    # The default timestamp must be tz-aware UTC and within the bracketed
    # wall-clock window.
    assert handle.started_at.tzinfo is not None
    assert before - timedelta(seconds=1) <= handle.started_at <= after + timedelta(seconds=1)


# ---------------------------------------------------------------------------
# try_start: rejection path (Requirement 8.6 / Property 12)
# ---------------------------------------------------------------------------


def test_second_start_while_running_raises_ingestion_in_progress() -> None:
    coord = IngestionCoordinator()
    coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=1)

    with pytest.raises(IngestionInProgressError) as excinfo:
        coord.try_start(trigger=SnapshotTrigger.SINGLE_PROJECT, snapshot_id=2)

    # Canonical Requirement 8.6 message.
    assert str(excinfo.value) == "Ingestion_Job is already in progress"
    assert excinfo.value.message == "Ingestion_Job is already in progress"


def test_rejected_start_leaves_coordinator_state_unchanged() -> None:
    coord = IngestionCoordinator()
    started_at = datetime(2025, 1, 1, tzinfo=UTC)
    coord.try_start(
        trigger=SnapshotTrigger.FULL,
        snapshot_id=100,
        started_at=started_at,
    )
    before = coord.current_state()

    with pytest.raises(IngestionInProgressError):
        coord.try_start(
            trigger=SnapshotTrigger.SCHEDULED,
            snapshot_id=999,
            started_at=datetime(2099, 12, 31, tzinfo=UTC),
        )

    after = coord.current_state()
    assert after == before
    assert after.snapshot_id == 100
    assert after.trigger is SnapshotTrigger.FULL
    assert after.started_at == started_at


# ---------------------------------------------------------------------------
# Releasing the slot via complete / abort / context manager
# ---------------------------------------------------------------------------


def test_complete_releases_slot_so_next_start_succeeds() -> None:
    coord = IngestionCoordinator()
    h1 = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=1)
    h1.complete()

    assert h1.is_active is False
    assert coord.current_state().state is CoordinatorState.IDLE

    # Next start now succeeds.
    h2 = coord.try_start(trigger=SnapshotTrigger.SINGLE_PROJECT, snapshot_id=2)
    assert h2.snapshot_id == 2
    assert coord.current_state().state is CoordinatorState.RUNNING


def test_abort_releases_slot_so_next_start_succeeds() -> None:
    coord = IngestionCoordinator()
    h1 = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=1)
    h1.abort()

    assert h1.is_active is False
    assert coord.current_state().state is CoordinatorState.IDLE

    h2 = coord.try_start(trigger=SnapshotTrigger.SCHEDULED, snapshot_id=2)
    assert h2.snapshot_id == 2


def test_complete_is_idempotent() -> None:
    coord = IngestionCoordinator()
    handle = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=1)
    handle.complete()
    handle.complete()  # no error
    handle.abort()  # also no error
    assert coord.current_state().state is CoordinatorState.IDLE


def test_abort_is_idempotent() -> None:
    coord = IngestionCoordinator()
    handle = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=1)
    handle.abort()
    handle.abort()
    assert coord.current_state().state is CoordinatorState.IDLE


def test_stale_handle_release_does_not_clear_fresh_running_state() -> None:
    """A stale handle's release must not steal a different job's slot."""

    coord = IngestionCoordinator()
    stale = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=1)
    stale.complete()

    fresh = coord.try_start(trigger=SnapshotTrigger.SINGLE_PROJECT, snapshot_id=2)

    # Calling complete() on the stale handle a second time must be a
    # no-op; it must not release the fresh job's slot.
    stale.complete()

    assert coord.current_state().state is CoordinatorState.RUNNING
    assert coord.current_state().snapshot_id == 2

    fresh.complete()
    assert coord.current_state().state is CoordinatorState.IDLE


def test_context_manager_completes_on_clean_exit() -> None:
    coord = IngestionCoordinator()
    handle = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=11)

    with handle as h:
        assert h is handle
        assert coord.current_state().state is CoordinatorState.RUNNING

    assert handle.is_active is False
    assert coord.current_state().state is CoordinatorState.IDLE


def test_context_manager_aborts_on_exception_and_does_not_swallow_it() -> None:
    coord = IngestionCoordinator()
    handle = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=12)

    class _BoomError(RuntimeError):
        pass

    with pytest.raises(_BoomError), handle:
        raise _BoomError("boom")

    assert handle.is_active is False
    assert coord.current_state().state is CoordinatorState.IDLE


# ---------------------------------------------------------------------------
# CoordinatorStatus is a frozen, read-only dataclass
# ---------------------------------------------------------------------------


def test_coordinator_status_is_frozen() -> None:
    status = CoordinatorStatus(state=CoordinatorState.IDLE)
    with pytest.raises((AttributeError, Exception)):
        status.state = CoordinatorState.RUNNING  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Multi-threaded CAS: exactly one winner
# ---------------------------------------------------------------------------


def test_concurrent_try_start_elects_exactly_one_winner() -> None:
    """Many threads racing on try_start must produce exactly one success."""

    coord = IngestionCoordinator()
    barrier = threading.Barrier(8)
    successes: list[JobHandle] = []
    rejections: list[IngestionInProgressError] = []
    success_lock = threading.Lock()
    reject_lock = threading.Lock()

    def worker(snapshot_id: int) -> None:
        # Synchronize the start so all threads CAS as close together as
        # the OS scheduler allows.
        barrier.wait()
        try:
            handle = coord.try_start(
                trigger=SnapshotTrigger.FULL,
                snapshot_id=snapshot_id,
            )
        except IngestionInProgressError as exc:
            with reject_lock:
                rejections.append(exc)
        else:
            with success_lock:
                successes.append(handle)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 1
    assert len(rejections) == 7
    for exc in rejections:
        assert str(exc) == "Ingestion_Job is already in progress"

    # The winner's snapshot_id is whichever thread won the CAS race; it
    # must be one of the contestants and must match what current_state
    # reports.
    winner = successes[0]
    assert winner.snapshot_id in range(8)
    status = coord.current_state()
    assert status.state is CoordinatorState.RUNNING
    assert status.snapshot_id == winner.snapshot_id

    winner.complete()
    assert coord.current_state().state is CoordinatorState.IDLE


def test_serial_start_complete_cycles_alternate_states() -> None:
    coord = IngestionCoordinator()
    for i in range(5):
        assert coord.current_state().state is CoordinatorState.IDLE
        handle = coord.try_start(trigger=SnapshotTrigger.FULL, snapshot_id=i)
        assert coord.current_state().state is CoordinatorState.RUNNING
        assert coord.current_state().snapshot_id == i
        handle.complete()
        assert coord.current_state().state is CoordinatorState.IDLE

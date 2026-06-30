"""Unit tests for ``main._fire_initial_refresh``.

The initial-refresh helper closes the documented gap between a
configured ``REFRESH_INTERVAL`` and the moment the first project
profile appears in the visualization. Three behaviors must hold:

1. The helper invokes ``coordinator.start_full_refresh`` exactly once.
2. An :class:`IngestionInProgressError` (a scheduler tick that raced
   the initial refresh to the single-flight gate) is logged at
   ``INFO`` level and does not propagate to the caller — startup
   must not be blocked by a benign race.
3. Any other exception is logged with a traceback and does not
   propagate — the visualization must still come up so the operator
   can investigate via the visualization or MCP tools.

Each test substitutes a recording coordinator double and joins the
daemon thread the helper returns so the assertions run after the
runner has finished.
"""

from __future__ import annotations

import logging
import threading

import pytest

from project_knowledge_mcp.errors import IngestionInProgressError
from project_knowledge_mcp.main import _fire_initial_refresh

pytestmark = pytest.mark.unit

# A generous join timeout that covers a slow CI worker without
# silently hiding a genuinely hung runner.
_JOIN_TIMEOUT_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Recording coordinator double
# ---------------------------------------------------------------------------


class _RecordingCoordinator:
    """Captures the number of ``start_full_refresh`` calls.

    Optionally raises an exception on each call to exercise the
    error-handling branches of :func:`_fire_initial_refresh`.
    """

    def __init__(self, raises: BaseException | None = None) -> None:
        self.call_count = 0
        self._raises = raises

    def start_full_refresh(self) -> None:
        self.call_count += 1
        if self._raises is not None:
            raise self._raises


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_initial_refresh_invokes_start_full_refresh_exactly_once() -> None:
    """The helper calls ``start_full_refresh`` once on the spawned thread."""
    coordinator = _RecordingCoordinator()

    thread = _fire_initial_refresh(coordinator)  # type: ignore[arg-type]

    thread.join(timeout=_JOIN_TIMEOUT_SECONDS)
    assert not thread.is_alive(), "initial refresh thread did not finish"
    assert coordinator.call_count == 1


def test_initial_refresh_thread_is_daemonic() -> None:
    """The thread is daemonic so it cannot block process exit at shutdown.

    The shutdown sequence (Requirement 12.9) aborts any in-flight
    ``Ingestion_Job`` it owns; a non-daemon initial-refresh thread
    would still keep the interpreter alive after ``main()`` returns.
    """
    coordinator = _RecordingCoordinator()

    thread = _fire_initial_refresh(coordinator)  # type: ignore[arg-type]
    try:
        assert thread.daemon is True
        assert thread.name == "project-knowledge-mcp-initial-refresh"
    finally:
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Single-flight race
# ---------------------------------------------------------------------------


def test_initial_refresh_swallows_ingestion_in_progress_at_info_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scheduler tick that won the single-flight race logs INFO, not WARNING.

    This is the documented benign race: an operator who sets
    ``REFRESH_INTERVAL=1m`` may see the scheduler's first tick win
    the lock by microseconds. The initial refresh is a no-op in that
    case and MUST NOT propagate the rejection to startup.
    """
    coordinator = _RecordingCoordinator(raises=IngestionInProgressError())

    with caplog.at_level(logging.INFO, logger="project_knowledge_mcp.main"):
        thread = _fire_initial_refresh(coordinator)  # type: ignore[arg-type]
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    assert not thread.is_alive()
    assert coordinator.call_count == 1
    info_messages = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO
    ]
    assert any(
        "Initial refresh skipped" in msg for msg in info_messages
    ), f"INFO line about skipped refresh missing; observed: {info_messages!r}"


# ---------------------------------------------------------------------------
# Unexpected failure
# ---------------------------------------------------------------------------


def test_initial_refresh_swallows_unexpected_exception_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unrelated exception must be logged with traceback, never raised.

    The visualization surface and the MCP stdio surface must both
    come up regardless of an initial-refresh failure: operators rely
    on the visualization's empty-state message to see *that* an
    ingestion failed, not on the process crashing at startup.
    """
    coordinator = _RecordingCoordinator(raises=RuntimeError("boom"))

    with caplog.at_level(
        logging.ERROR, logger="project_knowledge_mcp.main"
    ):
        thread = _fire_initial_refresh(coordinator)  # type: ignore[arg-type]
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    assert not thread.is_alive()
    assert coordinator.call_count == 1
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(
        "Initial refresh raised" in record.message for record in error_records
    ), "ERROR line about unexpected exception is missing"
    assert any(
        record.exc_info is not None for record in error_records
    ), "ERROR record must include a traceback (exc_info)"


# ---------------------------------------------------------------------------
# Concurrency: the helper returns immediately
# ---------------------------------------------------------------------------


def test_initial_refresh_does_not_block_the_caller() -> None:
    """The helper returns to the caller before ``start_full_refresh`` runs.

    Startup wiring in :func:`main` must hand control to
    :func:`_serve_surfaces` immediately so the visualization and MCP
    surfaces start serving traffic in parallel with the first
    ingestion. Verify by holding the coordinator inside
    ``start_full_refresh`` on a barrier the test releases only after
    confirming the helper has already returned.
    """
    barrier = threading.Event()
    released = threading.Event()

    class _BlockingCoordinator:
        def start_full_refresh(self) -> None:
            # Signal that the runner thread has entered the
            # ingestion call, then wait for the test to release us.
            released.set()
            barrier.wait(timeout=_JOIN_TIMEOUT_SECONDS)

    coordinator = _BlockingCoordinator()

    thread = _fire_initial_refresh(coordinator)  # type: ignore[arg-type]
    try:
        # If the helper had been blocking, ``thread`` would not yet
        # be in ``start_full_refresh`` (i.e. ``released`` would be
        # clear). Allow up to ``_JOIN_TIMEOUT_SECONDS`` for the
        # daemon thread to be scheduled, then assert.
        assert released.wait(
            timeout=_JOIN_TIMEOUT_SECONDS
        ), "runner thread never entered start_full_refresh"
    finally:
        barrier.set()
        thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    assert not thread.is_alive()

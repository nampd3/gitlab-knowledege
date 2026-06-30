"""Unit tests for the shutdown ordering at the function level.

Task 11.4 pins the call ordering performed by
:func:`project_knowledge_mcp.main._shutdown_in_order`. The shutdown
sequence (Requirement 12.9, design's *Shutdown* section) is the load-
bearing teardown contract for the single-process MCP_Server: it
guarantees that the Visualization_Server stops accepting new HTTP
connections before anything else, that an in-flight ``Ingestion_Job``
is signalled to abort *before* the scheduler thread is joined (so the
worker can unwind on the aborted snapshot rather than blocking the
join), and that the ``Knowledge_Store`` is closed last so neither
surface is asked to read from a closed database.

These tests substitute a recording double for each component the
shutdown function touches and assert the *exact* call order matches
the documented six steps:

1. ``Visualization_Server`` stops accepting new HTTP connections
   (``uvicorn.Server.should_exit = True``).
2. Scheduler is signalled to stop firing new ticks
   (``Scheduler.stop(timeout=0)``).
3. MCP stdio handlers are closed (mcp serve task is cancelled).
4. In-progress ``Ingestion_Job`` snapshot is marked failed
   (``KnowledgeStore.abort_snapshot(snapshot_id)``).
5. Scheduler thread is joined (``Scheduler.stop(timeout=>0)`` via
   ``asyncio.to_thread``).
6. ``GitLab_Connector`` and ``Knowledge_Store`` are closed.

Each test exercises the production code path in
:func:`_shutdown_in_order` directly, so no integration-style fixture
(real sockets, real Uvicorn, real SQLite) is involved — the recording
doubles capture the call order while all timing constants and
``contextlib.suppress`` arms behave exactly as in production. The
tests cover the documented branches in the design's *Shutdown* and
*Shutdown failures* sections:

* Happy path with every component present and an in-flight job.
* No scheduler (refresh interval unset).
* No in-flight ``Ingestion_Job`` (coordinator is idle).
* No ``GitLab_Connector`` (close is skipped).
* One step raises mid-shutdown (subsequent steps still run, per the
  design's *Shutdown failures* "best-effort" rule).

Implements Requirement 12.9.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from project_knowledge_mcp.ingestion_coordinator import (
    CoordinatorState,
    CoordinatorStatus,
)
from project_knowledge_mcp.main import _shutdown_in_order
from project_knowledge_mcp.models import SnapshotTrigger

if TYPE_CHECKING:
    import uvicorn

    from project_knowledge_mcp.gitlab_connector import GitLabConnector
    from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator
    from project_knowledge_mcp.knowledge_store import KnowledgeStore
    from project_knowledge_mcp.scheduler import Scheduler

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Step labels
# ---------------------------------------------------------------------------
#
# Every recording double appends a ``(component, action, args)`` tuple
# to a single shared :class:`_CallLog` so the tests can assert the
# total order across components. The labels mirror the design's six
# steps verbatim:

VIZ_STOP_ACCEPTING = ("viz_server", "stop_accepting")
SCHEDULER_SIGNAL = ("scheduler", "signal_stop")
MCP_STDIO_CLOSE = ("mcp_task", "stdio_close")
COORDINATOR_QUERY = ("coordinator", "current_state")
STORE_ABORT_SNAPSHOT = ("store", "abort_snapshot")
SCHEDULER_JOIN = ("scheduler", "join")
CONNECTOR_CLOSE = ("gitlab_connector", "close")
STORE_CLOSE = ("store", "close")


# ---------------------------------------------------------------------------
# Recording doubles
# ---------------------------------------------------------------------------


@dataclass
class _CallLog:
    """A shared, totally-ordered log of method calls across recording doubles.

    The tests assert against the sequence of ``(component, action)``
    pairs the shutdown function produces. The ``args`` payload is kept
    so individual tests can also assert that, for example,
    :meth:`_RecordingStore.abort_snapshot` was invoked with the
    correct ``snapshot_id``.
    """

    entries: list[tuple[str, str, tuple[Any, ...]]] = field(default_factory=list)

    def record(self, component: str, action: str, *args: Any) -> None:
        self.entries.append((component, action, args))

    def steps(self) -> list[tuple[str, str]]:
        """Return the call log without the args payload, for ordering asserts."""

        return [(component, action) for component, action, _ in self.entries]


class _RecordingVizServer:
    """Stand-in for :class:`uvicorn.Server`.

    The real ``_stop_visualization_server`` helper sets
    :attr:`uvicorn.Server.should_exit` to ``True`` to ask the accept
    loop to drain. We expose ``should_exit`` as a property so the
    setter records the event into the shared log; the underlying
    :class:`asyncio.Event` is then set so the paired ``viz_task``
    coroutine wakes up and the bounded ``await`` inside
    ``_stop_visualization_server`` returns immediately.
    """

    def __init__(self, log: _CallLog) -> None:
        self._log = log
        self._should_exit = False
        self._exit_event = asyncio.Event()

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        # The first transition from False to True is the documented
        # "Visualization_Server stops accepting new HTTP connections"
        # step. Subsequent assignments (which the production code does
        # not perform) are no-ops on the log so the ordering stays
        # one-to-one with the design's six steps.
        if value and not self._should_exit:
            self._log.record(*VIZ_STOP_ACCEPTING)
            self._exit_event.set()
        self._should_exit = value

    async def serve(self) -> None:
        """Block until ``should_exit`` is set, modelling the accept loop."""

        await self._exit_event.wait()


class _RecordingScheduler:
    """Stand-in for :class:`Scheduler`.

    Production code calls :meth:`stop` twice: once with ``timeout=0``
    (step 2: signal-only, no join) and once with the bounded
    ``_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS`` (step 5: join). The double
    distinguishes the two by the ``timeout`` value so the call log
    differentiates "signal" from "join" without smuggling extra state
    into the production interface.
    """

    def __init__(
        self,
        log: _CallLog,
        *,
        running: bool = True,
        signal_raises: BaseException | None = None,
        join_raises: BaseException | None = None,
    ) -> None:
        self._log = log
        self._running = running
        self._signal_raises = signal_raises
        self._join_raises = join_raises
        self.stop_calls: list[float | None] = []

    def stop(self, *, timeout: float | None = None) -> None:
        self.stop_calls.append(timeout)
        if timeout == 0:
            # Step 2: signal-only.
            self._log.record(*SCHEDULER_SIGNAL, timeout)
            if self._signal_raises is not None:
                raise self._signal_raises
        else:
            # Step 5: actual join.
            self._log.record(*SCHEDULER_JOIN, timeout)
            if self._join_raises is not None:
                raise self._join_raises
            # Mirror real Scheduler: once joined, ``is_running`` flips.
            self._running = False

    def is_running(self) -> bool:
        return self._running


class _RecordingCoordinator:
    """Stand-in for :class:`IngestionCoordinator` for shutdown purposes.

    The shutdown helper only ever calls :meth:`current_state` on the
    coordinator (the in-process abort path lives on the
    ``Knowledge_Store``), so the double surfaces only that method.
    """

    def __init__(self, log: _CallLog, status: CoordinatorStatus) -> None:
        self._log = log
        self._status = status

    def current_state(self) -> CoordinatorStatus:
        self._log.record(*COORDINATOR_QUERY)
        return self._status


class _RecordingStore:
    """Stand-in for :class:`KnowledgeStore`.

    Records :meth:`abort_snapshot` and :meth:`close` so the tests can
    assert (a) the order of those two calls relative to every other
    step and (b) that ``abort_snapshot`` is invoked exactly when the
    coordinator reports a running job.
    """

    def __init__(
        self,
        log: _CallLog,
        *,
        abort_raises: BaseException | None = None,
        close_raises: BaseException | None = None,
    ) -> None:
        self._log = log
        self._abort_raises = abort_raises
        self._close_raises = close_raises

    def abort_snapshot(self, snapshot_id: int) -> None:
        self._log.record(*STORE_ABORT_SNAPSHOT, snapshot_id)
        if self._abort_raises is not None:
            raise self._abort_raises

    def close(self) -> None:
        self._log.record(*STORE_CLOSE)
        if self._close_raises is not None:
            raise self._close_raises


class _RecordingConnector:
    """Stand-in for :class:`GitLabConnector`."""

    def __init__(
        self,
        log: _CallLog,
        *,
        close_raises: BaseException | None = None,
    ) -> None:
        self._log = log
        self._close_raises = close_raises

    def close(self) -> None:
        self._log.record(*CONNECTOR_CLOSE)
        if self._close_raises is not None:
            raise self._close_raises


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


async def _make_viz_task(viz_server: _RecordingVizServer) -> asyncio.Task[None]:
    """Wrap ``viz_server.serve`` in an :class:`asyncio.Task`.

    The task models the live Uvicorn accept loop: it stays pending
    until ``should_exit`` flips, at which point it returns. We yield
    once with ``asyncio.sleep(0)`` so the task is registered with the
    event loop before the shutdown sequence runs.
    """

    task = asyncio.create_task(viz_server.serve(), name="viz-server")
    await asyncio.sleep(0)
    return task


async def _make_mcp_task(log: _CallLog) -> asyncio.Task[None]:
    """Build an MCP stdio task that records its own cancellation.

    The shutdown helper closes MCP stdio by cancelling the task. The
    coroutine catches :class:`asyncio.CancelledError`, records the
    "stdio_close" event into the shared log, then re-raises so the
    task settles in the ``cancelled`` state. Recording inside the
    cancellation handler guarantees the log entry lands while the
    shutdown helper is still awaiting the task — i.e. before step 4
    can begin.
    """

    async def _serve() -> None:
        try:
            # An ``asyncio.Event`` that nobody sets is the simplest way
            # to model a coroutine that runs forever until cancelled.
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            log.record(*MCP_STDIO_CLOSE)
            raise

    task = asyncio.create_task(_serve(), name="mcp-stdio-server")
    await asyncio.sleep(0)
    return task


async def _drain(task: asyncio.Task[Any]) -> None:
    """Await ``task`` ignoring cancellation so test cleanup is quiet."""

    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_happy_path_runs_six_steps_in_documented_order() -> None:
    """Every component present and a job in flight: full sequence in order.

    This is the canonical Requirement 12.9 path. The design's
    *Shutdown* section enumerates the six steps and the implementation
    in :func:`_shutdown_in_order` performs them in that order; this
    test pins the order at the function level so a future refactor
    cannot silently re-arrange the steps.
    """
    log = _CallLog()
    viz = _RecordingVizServer(log)
    viz_task = await _make_viz_task(viz)
    mcp_task = await _make_mcp_task(log)
    scheduler = _RecordingScheduler(log, running=True)
    snapshot_id = 42
    coordinator = _RecordingCoordinator(
        log,
        CoordinatorStatus(
            state=CoordinatorState.RUNNING,
            snapshot_id=snapshot_id,
            trigger=SnapshotTrigger.FULL,
            started_at=datetime.now(UTC),
        ),
    )
    store = _RecordingStore(log)
    connector = _RecordingConnector(log)

    await _shutdown_in_order(
        viz_server=cast("uvicorn.Server", viz),
        viz_task=viz_task,
        mcp_task=mcp_task,
        scheduler=cast("Scheduler", scheduler),
        coordinator=cast("IngestionCoordinator", coordinator),
        store=cast("KnowledgeStore", store),
        gitlab_connector=cast("GitLabConnector", connector),
    )

    assert log.steps() == [
        VIZ_STOP_ACCEPTING,
        SCHEDULER_SIGNAL,
        MCP_STDIO_CLOSE,
        COORDINATOR_QUERY,
        STORE_ABORT_SNAPSHOT,
        SCHEDULER_JOIN,
        CONNECTOR_CLOSE,
        STORE_CLOSE,
    ]

    # Step 4 must abort exactly the snapshot the coordinator reported.
    abort_entries = [e for e in log.entries if (e[0], e[1]) == STORE_ABORT_SNAPSHOT]
    assert abort_entries == [("store", "abort_snapshot", (snapshot_id,))]

    # Step 2 signals with timeout=0 and step 5 joins with a positive
    # bounded timeout. Pinning these arguments documents the design's
    # "signal-now / join-later" two-phase approach.
    assert scheduler.stop_calls[0] == 0
    assert scheduler.stop_calls[1] is not None
    assert scheduler.stop_calls[1] > 0

    await _drain(viz_task)
    await _drain(mcp_task)


async def test_no_scheduler_skips_signal_and_join_steps() -> None:
    """Refresh interval unset: scheduler steps are absent but order holds.

    When ``config.refresh_interval`` is ``None`` the scheduler is
    never constructed (per :func:`main.main`). The shutdown helper
    must therefore skip steps 2 and 5 while preserving the relative
    order of every other step.
    """
    log = _CallLog()
    viz = _RecordingVizServer(log)
    viz_task = await _make_viz_task(viz)
    mcp_task = await _make_mcp_task(log)
    coordinator = _RecordingCoordinator(
        log,
        CoordinatorStatus(
            state=CoordinatorState.RUNNING,
            snapshot_id=7,
            trigger=SnapshotTrigger.FULL,
            started_at=datetime.now(UTC),
        ),
    )
    store = _RecordingStore(log)
    connector = _RecordingConnector(log)

    await _shutdown_in_order(
        viz_server=cast("uvicorn.Server", viz),
        viz_task=viz_task,
        mcp_task=mcp_task,
        scheduler=None,
        coordinator=cast("IngestionCoordinator", coordinator),
        store=cast("KnowledgeStore", store),
        gitlab_connector=cast("GitLabConnector", connector),
    )

    assert log.steps() == [
        VIZ_STOP_ACCEPTING,
        MCP_STDIO_CLOSE,
        COORDINATOR_QUERY,
        STORE_ABORT_SNAPSHOT,
        CONNECTOR_CLOSE,
        STORE_CLOSE,
    ]

    await _drain(viz_task)
    await _drain(mcp_task)


async def test_no_in_flight_ingestion_job_skips_abort_snapshot() -> None:
    """Coordinator idle: ``abort_snapshot`` is not invoked.

    Step 4 of the shutdown sequence is conditional on the coordinator
    reporting a running job. When the coordinator is idle the
    snapshot pointer is already consistent; calling
    :meth:`KnowledgeStore.abort_snapshot` would raise because there is
    no in-progress snapshot. The helper must therefore skip
    ``abort_snapshot`` while still calling ``current_state`` (so the
    branch decision is observable) and while preserving the order of
    every other step.
    """
    log = _CallLog()
    viz = _RecordingVizServer(log)
    viz_task = await _make_viz_task(viz)
    mcp_task = await _make_mcp_task(log)
    scheduler = _RecordingScheduler(log, running=True)
    coordinator = _RecordingCoordinator(
        log,
        CoordinatorStatus(state=CoordinatorState.IDLE),
    )
    store = _RecordingStore(log)
    connector = _RecordingConnector(log)

    await _shutdown_in_order(
        viz_server=cast("uvicorn.Server", viz),
        viz_task=viz_task,
        mcp_task=mcp_task,
        scheduler=cast("Scheduler", scheduler),
        coordinator=cast("IngestionCoordinator", coordinator),
        store=cast("KnowledgeStore", store),
        gitlab_connector=cast("GitLabConnector", connector),
    )

    assert log.steps() == [
        VIZ_STOP_ACCEPTING,
        SCHEDULER_SIGNAL,
        MCP_STDIO_CLOSE,
        COORDINATOR_QUERY,
        SCHEDULER_JOIN,
        CONNECTOR_CLOSE,
        STORE_CLOSE,
    ]
    # No abort_snapshot call occurred even though the coordinator was
    # queried in step 4.
    assert all((e[0], e[1]) != STORE_ABORT_SNAPSHOT for e in log.entries)

    await _drain(viz_task)
    await _drain(mcp_task)


async def test_no_gitlab_connector_skips_connector_close() -> None:
    """No connector wired: the connector close step is omitted.

    :func:`_serve_surfaces` accepts ``gitlab_connector=None`` so unit
    tests and degraded-startup paths can drive the shutdown sequence
    without a real HTTP client. The store close (step 6's second
    half) must still run last.
    """
    log = _CallLog()
    viz = _RecordingVizServer(log)
    viz_task = await _make_viz_task(viz)
    mcp_task = await _make_mcp_task(log)
    scheduler = _RecordingScheduler(log, running=True)
    coordinator = _RecordingCoordinator(
        log,
        CoordinatorStatus(
            state=CoordinatorState.RUNNING,
            snapshot_id=11,
            trigger=SnapshotTrigger.SINGLE_PROJECT,
            started_at=datetime.now(UTC),
        ),
    )
    store = _RecordingStore(log)

    await _shutdown_in_order(
        viz_server=cast("uvicorn.Server", viz),
        viz_task=viz_task,
        mcp_task=mcp_task,
        scheduler=cast("Scheduler", scheduler),
        coordinator=cast("IngestionCoordinator", coordinator),
        store=cast("KnowledgeStore", store),
        gitlab_connector=None,
    )

    assert log.steps() == [
        VIZ_STOP_ACCEPTING,
        SCHEDULER_SIGNAL,
        MCP_STDIO_CLOSE,
        COORDINATOR_QUERY,
        STORE_ABORT_SNAPSHOT,
        SCHEDULER_JOIN,
        STORE_CLOSE,
    ]

    await _drain(viz_task)
    await _drain(mcp_task)


async def test_failure_in_one_step_does_not_abort_subsequent_steps() -> None:
    """Best-effort shutdown: a step that raises is logged and skipped.

    The design's *Shutdown failures* section requires that the helper
    log the failure and proceed with ``Knowledge_Store`` close and
    process exit even when an earlier step raises. We pick the step-2
    scheduler signal as the failing step because it is wrapped in
    :func:`contextlib.suppress` in production and is therefore the
    canonical example of a step that is allowed to fail.
    """
    log = _CallLog()
    viz = _RecordingVizServer(log)
    viz_task = await _make_viz_task(viz)
    mcp_task = await _make_mcp_task(log)
    scheduler = _RecordingScheduler(
        log,
        running=True,
        signal_raises=RuntimeError("scheduler signal exploded"),
    )
    coordinator = _RecordingCoordinator(
        log,
        CoordinatorStatus(
            state=CoordinatorState.RUNNING,
            snapshot_id=99,
            trigger=SnapshotTrigger.FULL,
            started_at=datetime.now(UTC),
        ),
    )
    store = _RecordingStore(log)
    connector = _RecordingConnector(log)

    # The helper must NOT propagate the scheduler signal failure.
    await _shutdown_in_order(
        viz_server=cast("uvicorn.Server", viz),
        viz_task=viz_task,
        mcp_task=mcp_task,
        scheduler=cast("Scheduler", scheduler),
        coordinator=cast("IngestionCoordinator", coordinator),
        store=cast("KnowledgeStore", store),
        gitlab_connector=cast("GitLabConnector", connector),
    )

    # The scheduler signal step recorded its attempt before raising,
    # and every subsequent step still ran in the documented order.
    assert log.steps() == [
        VIZ_STOP_ACCEPTING,
        SCHEDULER_SIGNAL,
        MCP_STDIO_CLOSE,
        COORDINATOR_QUERY,
        STORE_ABORT_SNAPSHOT,
        SCHEDULER_JOIN,
        CONNECTOR_CLOSE,
        STORE_CLOSE,
    ]

    await _drain(viz_task)
    await _drain(mcp_task)

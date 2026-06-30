"""Process entry point: wires startup, lifecycle, and shutdown.

This module is the top-level wiring step described in the design's
*Process composition* section. :func:`main` performs the documented
start-up sequence in order:

1. :func:`config.load_and_validate` — validate every configuration key
   and exit non-zero on any failure (Requirements 1.4, 1.5, 12.5,
   15.6). The shared ``configuration error for '{key}': {reason}``
   line is emitted by :mod:`config` so this module never duplicates
   it.
2. :meth:`KnowledgeStore.open` — open the SQLite-backed store at the
   path resolved by :func:`_resolve_store_path` so the snapshot
   pointer set by the previous run is available before either surface
   accepts traffic (Requirement 7.2). On any failure to open the
   store, :func:`_open_store_or_exit` emits a ``"startup error: ..."``
   line and exits non-zero before either surface starts.
3. Construct the in-process collaborators —
   :class:`ProjectCatalog`, :class:`GitLabConnector`, and
   :class:`IngestionCoordinator` (in the ``idle`` state). The
   coordinator is wired with the live ``Project_Analyzer.analyze``
   function so MCP refresh tools and the scheduler can run jobs
   end-to-end.
4. :func:`visualization_server.bind_or_exit` — bind both loopback
   sockets up-front (Requirement 12.2) so a port-in-use or other
   :class:`OSError` is reported as the documented start-up error and
   the process exits non-zero before any traffic is served
   (Requirements 12.6, 12.8). On success
   :func:`visualization_server.emit_ready_log` writes the documented
   ``"Visualization_Server ready at http://127.0.0.1:{port}"`` line
   (Requirement 12.7).
5. Construct :class:`MCPServer`. The MCP transport is stdio so there
   is no separate "bind" step; the SDK opens the stdio transport
   inside :meth:`MCPServer.serve` when the surface is started below.
6. Start the :class:`Scheduler` if and only if
   ``config.refresh_interval`` is configured (Requirement 8.3). When
   the interval is unset the scheduler is never constructed, matching
   the design's "no schedule" branch.
7. Hand control to :func:`_serve_surfaces`, which runs the Uvicorn
   server (over the pre-bound sockets) and the MCP stdio server
   concurrently in the same event loop (Requirement 12.1). When a
   shutdown signal arrives or either surface exits, :func:`_serve_surfaces`
   drives :func:`_shutdown_in_order` which performs the documented
   shutdown ordering (Requirement 12.9).

The single termination path on start-up failure is consolidated with
the exit lines emitted by :mod:`config` (task 2.3) and
:func:`visualization_server.bind_or_exit` (task 10.1) so Requirements
1.4, 1.5, 12.5, 12.6, 12.7, 12.8, and 15.6 share one implementation.

The shutdown ordering (Requirement 12.9, design's *Shutdown* section)
is performed by :func:`_shutdown_in_order` in this exact order:

1. Stop the ``Visualization_Server`` from accepting new HTTP
   connections (set ``uvicorn.Server.should_exit`` so the accept loop
   exits and in-flight responses drain).
2. Signal the scheduler (if any) to stop firing new ticks. The join
   is deferred until after the in-flight ``Ingestion_Job`` has been
   aborted so a tick currently inside ``start_full_refresh`` can
   unwind quickly when its next ``Knowledge_Store`` mutation fails on
   the now-aborted snapshot.
3. Close MCP stdio handlers (cancel the MCP serve task; the SDK
   tears down the stdio streams as the cancellation propagates).
4. If an ``Ingestion_Job`` is still in flight, mark its in-progress
   snapshot ``failed`` via ``Knowledge_Store.abort_snapshot``. The
   ``current_snapshot`` pointer is intentionally left untouched so the
   next startup serves the last successfully committed snapshot
   (consistent with Property 11 and the design's *Shutdown* section).
5. Join the scheduler thread (now safe; the in-flight tick's next
   write will fail on the aborted snapshot and the worker will
   unwind).
6. Close the ``GitLab_Connector`` (releases the underlying HTTP
   client) and then flush + close the ``Knowledge_Store``.

Implements Requirements 1.4, 1.5, 7.2, 12.1, 12.5, 12.6, 12.7, 12.8,
12.9, 15.6.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .config import load_and_validate
from .conflict_detector import classify_pair, find_all_conflicts
from .errors import IngestionInProgressError
from .gitlab_connector import GitLabConnector
from .ingestion_coordinator import CoordinatorState, IngestionCoordinator
from .knowledge_store import KnowledgeStore
from .mcp_server import MCPServer
from .project_analyzer import analyze
from .project_catalog import ProjectCatalog
from .scheduler import Scheduler
from .visualization_server import (
    bind_or_exit,
    build_visualization_app,
    create_server,
    emit_ready_log,
)

if TYPE_CHECKING:
    import socket as _socket
    from collections.abc import Mapping
    from typing import TextIO

    import uvicorn


# ---------------------------------------------------------------------------
# Knowledge_Store path resolution
# ---------------------------------------------------------------------------

#: Environment variable that overrides the default ``Knowledge_Store``
#: path. Operators that want to relocate the SQLite database (for
#: example onto a persistent volume in a container deployment) point
#: this at the desired path.
ENV_KNOWLEDGE_STORE_PATH: Final[str] = "KNOWLEDGE_STORE_PATH"

#: Default location of the ``Knowledge_Store`` SQLite database. The
#: path is relative so the default deployment writes the database
#: under the process's working directory; production deployments are
#: expected to override this via :data:`ENV_KNOWLEDGE_STORE_PATH`.
DEFAULT_KNOWLEDGE_STORE_PATH: Final[Path] = Path("data") / "knowledge_store.db"


# ---------------------------------------------------------------------------
# Shutdown timing constants (Requirement 12.9)
# ---------------------------------------------------------------------------

#: Maximum time, in seconds, to wait for the ``Visualization_Server``
#: accept loop to drain after :attr:`uvicorn.Server.should_exit` has
#: been raised. The :class:`_DeadlineMiddleware` in
#: :mod:`visualization_server` already bounds individual handlers at
#: :data:`HANDLER_DEADLINE_SECONDS`, so the drain is bounded in
#: practice; this is a defensive upper bound for the case where
#: Uvicorn's internal shutdown bookkeeping itself stalls.
_VISUALIZATION_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 10.0

#: Maximum time, in seconds, to wait for the MCP stdio task to
#: acknowledge cancellation when stdin is NOT a TTY (i.e. a real MCP
#: client is talking over a pipe). The SDK's ``Server.run`` propagates
#: ``CancelledError`` through ``stdio_server`` immediately on a clean
#: pipe-close, so this is only a defensive upper bound for the case
#: where the SDK's shutdown bookkeeping itself stalls.
_MCP_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 5.0

#: Quick probe window, in seconds, when stdin is a TTY (interactive
#: development). The MCP SDK's blocking reader thread is parked in a
#: ``read(2)`` syscall on the controlling terminal that Python cannot
#: interrupt, so waiting the full :data:`_MCP_SHUTDOWN_TIMEOUT_SECONDS`
#: just makes a single ``Ctrl+C`` look frozen. After this brief probe
#: we close stdin to unblock the syscall and let the SDK unwind.
_MCP_SHUTDOWN_TTY_PROBE_SECONDS: Final[float] = 0.25

#: Maximum time, in seconds, to wait for the MCP stdio task to drain
#: *after* stdin has been force-closed. The blocking ``read(2)``
#: returns immediately once its fd is closed, so the SDK only needs a
#: brief window to unwind its asyncio frames. Kept generous so a
#: slow logger or unusual SDK shutdown path still completes.
_MCP_SHUTDOWN_POST_CLOSE_TIMEOUT_SECONDS: Final[float] = 2.0

#: Maximum time, in seconds, to wait for the scheduler thread to join
#: during shutdown. A scheduler tick that is currently inside
#: ``start_full_refresh`` will unwind quickly once its in-progress
#: snapshot is aborted (the next ``Knowledge_Store`` mutation will
#: raise on the aborted snapshot), so 30 s is a generous upper bound.
_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 30.0


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

#: Logger used to emit progress messages from the wiring path. The
#: dedicated, named logger keeps the start-up trace easy to find when
#: running the server alongside other Python processes.
_LOG: Final[logging.Logger] = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve the Knowledge_Store path
# ---------------------------------------------------------------------------


def _resolve_store_path(env: Mapping[str, str] | None = None) -> Path:
    """Return the configured ``Knowledge_Store`` path or the default.

    Reads :data:`ENV_KNOWLEDGE_STORE_PATH` from ``env`` (defaulting to
    :data:`os.environ`). When the variable is unset or contains only
    whitespace the design's :data:`DEFAULT_KNOWLEDGE_STORE_PATH` is
    returned. ``env`` is exposed as a parameter so unit tests can
    inject a deterministic mapping without mutating
    :data:`os.environ`.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    raw = source.get(ENV_KNOWLEDGE_STORE_PATH)
    if raw is None or raw.strip() == "":
        return DEFAULT_KNOWLEDGE_STORE_PATH
    return Path(raw)


# ---------------------------------------------------------------------------
# Helper: open the Knowledge_Store, terminating non-zero on failure
# ---------------------------------------------------------------------------


def _open_store_or_exit(
    path: Path,
    *,
    stderr: TextIO | None = None,
) -> KnowledgeStore:
    """Open the ``Knowledge_Store`` at ``path`` or exit non-zero on failure.

    On any exception raised by :meth:`KnowledgeStore.open`, this
    helper writes a single ``"startup error: ..."`` line to ``stderr``
    that names the offending path and the underlying failure, then
    calls :func:`sys.exit` with a non-zero status. This mirrors the
    failure-mode contract of :func:`config.load_and_validate` (task
    2.3) and :func:`visualization_server.bind_or_exit` (task 10.1):
    every documented start-up failure consolidates onto a single
    "single line to stderr; non-zero exit" path so neither MCP nor
    the Visualization_Server is started when initialization cannot
    complete (Requirements 1.4, 1.5, 7.2, 12.5, 12.6, 12.8, 15.6).
    """
    err_stream = sys.stderr if stderr is None else stderr
    try:
        return KnowledgeStore.open(path)
    except Exception as exc:
        print(
            f"startup error: failed to open Knowledge_Store at {path}: {exc}",
            file=err_stream,
            flush=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Shutdown ordering (Requirement 12.9)
# ---------------------------------------------------------------------------


async def _stop_visualization_server(
    viz_server: uvicorn.Server,
    viz_task: asyncio.Task[None],
) -> None:
    """Step 1 of the shutdown ordering: stop accepting new HTTP connections.

    ``should_exit`` is Uvicorn's documented graceful-shutdown signal:
    the accept loop returns and in-flight requests drain (bounded by
    the per-handler deadline middleware). Setting it is a no-op if
    Uvicorn has already returned (for example because stdio closed
    and ``_serve_surfaces`` woke on ``viz_task``); we always set it
    for clarity.

    The wait on ``viz_task`` is bounded by
    :data:`_VISUALIZATION_SHUTDOWN_TIMEOUT_SECONDS` so a misbehaving
    handler cannot stall the rest of the shutdown sequence; if the
    drain does not complete in time the task is cancelled outright.
    """
    viz_server.should_exit = True
    if viz_task.done():
        return
    try:
        await asyncio.wait_for(
            asyncio.shield(viz_task),
            timeout=_VISUALIZATION_SHUTDOWN_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        _LOG.warning(
            "Visualization_Server did not drain within %.1fs; cancelling.",
            _VISUALIZATION_SHUTDOWN_TIMEOUT_SECONDS,
        )
        viz_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await viz_task
    except asyncio.CancelledError:
        # ``shield`` raises CancelledError on the outer ``wait_for``
        # without cancelling the inner task; let the task settle.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await viz_task
    except Exception:
        _LOG.exception("Visualization_Server raised during shutdown.")


async def _close_mcp_stdio(mcp_task: asyncio.Task[None]) -> None:
    """Step 3 of the shutdown ordering: close MCP stdio handlers.

    Cancelling ``mcp_task`` propagates :class:`asyncio.CancelledError`
    into :func:`mcp.server.stdio.stdio_server`, which tears down the
    stdin/stdout streams as the cancellation unwinds — the documented
    "MCP transport closes its stdio handlers" step.

    The shutdown happens in two stages so a single ``Ctrl+C`` from an
    interactive terminal returns quickly without changing the
    behavior seen by an MCP client talking over a pipe:

    1. **Probe.** Cancel the task and wait briefly for the SDK to
       acknowledge. The probe window is
       :data:`_MCP_SHUTDOWN_TTY_PROBE_SECONDS` when stdin is a TTY
       (interactive development; the SDK's reader thread is parked
       in a blocking ``read(2)`` that Python cannot interrupt) and
       :data:`_MCP_SHUTDOWN_TIMEOUT_SECONDS` otherwise.
    2. **Force-close stdin.** If the probe times out, close stdin's
       file descriptor so the blocked ``read(2)`` returns and the
       SDK's worker thread terminates. Then wait up to
       :data:`_MCP_SHUTDOWN_POST_CLOSE_TIMEOUT_SECONDS` for the task
       to finish unwinding.
    """
    if not mcp_task.done():
        mcp_task.cancel()

    probe_timeout = (
        _MCP_SHUTDOWN_TTY_PROBE_SECONDS
        if _stdin_is_tty()
        else _MCP_SHUTDOWN_TIMEOUT_SECONDS
    )
    try:
        await asyncio.wait_for(asyncio.shield(mcp_task), timeout=probe_timeout)
        return
    except asyncio.CancelledError:
        # The SDK honored the cancellation cleanly (the normal path
        # both for pipe-driven clients and for the rare case where a
        # TTY's blocking read happened to be between syscalls).
        return
    except TimeoutError:
        pass  # Fall through to the stdin-close stage below.
    except Exception:
        _LOG.exception("MCP server raised during shutdown.")
        return

    _LOG.info(
        "MCP stdio task did not exit within %.1fs; closing stdin "
        "to unblock the reader.",
        probe_timeout,
    )
    _force_close_stdin()
    try:
        await asyncio.wait_for(
            asyncio.shield(mcp_task),
            timeout=_MCP_SHUTDOWN_POST_CLOSE_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        return
    except TimeoutError:
        _LOG.warning(
            "MCP stdio task still alive %.1fs after stdin close; "
            "proceeding with the rest of the shutdown sequence.",
            _MCP_SHUTDOWN_POST_CLOSE_TIMEOUT_SECONDS,
        )
    except Exception:
        _LOG.exception("MCP server raised during stdin-close shutdown.")


#: Environment variable that overrides the root logging level used
#: by :func:`_configure_logging`. Accepts any standard Python logging
#: level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
#: ``CRITICAL``). Defaults to ``WARNING``: the application's
#: operationally important lines (visualization "ready" banner,
#: refresh-progress per-project lines, refresh-complete summary) are
#: emitted at WARNING level on purpose so they surface under this
#: default without dragging in third-party libraries' per-request
#: chatter. Set ``LOG_LEVEL=INFO`` to additionally see less
#: essential application INFO lines, or ``LOG_LEVEL=DEBUG`` for
#: full diagnostic verbosity. Unknown values fall back to WARNING.
ENV_LOG_LEVEL: Final[str] = "LOG_LEVEL"


def _configure_logging() -> None:
    """Configure root logging to stderr at the requested level.

    Default level is ``WARNING`` because the application's
    operationally important progress lines are emitted at WARNING
    level on purpose (see
    :mod:`project_knowledge_mcp.ingestion_coordinator` and
    :func:`project_knowledge_mcp.visualization_server.emit_ready_log`).
    Third-party libraries (httpx, httpcore, uvicorn, the MCP SDK)
    inherit the same threshold, which keeps their per-request chatter
    out of the operator's terminal.

    The threshold is overridable via the :data:`ENV_LOG_LEVEL`
    environment variable. Setting ``LOG_LEVEL=INFO`` reveals the
    application's diagnostic INFO lines without changing where the
    operational signals come from.

    Idempotent: when called more than once (e.g. by a test harness
    that has already configured logging) the additional call is a
    no-op so existing :class:`logging.Handler` setups — including
    pytest's ``caplog`` capture — are preserved.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    requested = os.environ.get(ENV_LOG_LEVEL, "WARNING").strip().upper()
    level = getattr(logging, requested, logging.WARNING)
    if not isinstance(level, int):
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _stdin_is_tty() -> bool:
    """Return ``True`` when stdin is attached to a terminal.

    Wrapped in a single helper because :meth:`io.IOBase.isatty` can
    raise :class:`ValueError` on a closed stream or :class:`OSError`
    on a stream whose underlying file descriptor was replaced by a
    test harness. Either failure is treated as "not a TTY" so the
    shutdown path falls back to the production timing.
    """
    try:
        return sys.stdin.isatty()
    except (OSError, ValueError):
        return False


def _force_close_stdin() -> None:
    """Close stdin's file descriptor so blocking reads return.

    Wraps the failure modes in a single ``suppress`` so the shutdown
    sequence cannot raise on a process that started without a real
    stdin (test harnesses, ``stdin=DEVNULL`` invocations) or whose
    stdin has already been closed by an earlier shutdown phase.
    Operates on the OS file descriptor rather than ``sys.stdin``
    because the MCP SDK's reader thread is blocked on the raw
    descriptor and does not consult the Python-level stream object.
    """
    with contextlib.suppress(OSError, ValueError):
        os.close(sys.stdin.fileno())


def _fire_initial_refresh(coordinator: IngestionCoordinator) -> threading.Thread:
    """Run one full refresh on a daemon thread, immediately after startup.

    Implements the documented "initial full refresh" referenced by
    :class:`Scheduler` ("the MCP_Server's startup sequence performs
    an initial full refresh independently; the scheduler is
    responsible only for the periodic cadence thereafter"). Without
    this, an operator who configures ``REFRESH_INTERVAL`` waits the
    full interval before any project profile appears in the
    visualization — for a long interval that is "no projects" from
    the operator's perspective.

    The thread is daemonic so it does not delay process exit during
    shutdown: an in-flight initial refresh shares fate with the
    Knowledge_Store snapshot it opened, which
    :func:`_shutdown_in_order` aborts at step 4. Returns the thread
    so unit tests can ``join`` it; production callers ignore the
    return value.
    """

    def runner() -> None:
        try:
            coordinator.start_full_refresh()
        except IngestionInProgressError:
            # A scheduler tick raced the initial refresh to the
            # single-flight gate (only possible for very short
            # ``REFRESH_INTERVAL`` values, e.g. ``1m``). The tick is
            # already doing what the initial refresh was going to do,
            # so log an info line and let the tick proceed.
            _LOG.info(
                "Initial refresh skipped: ingestion already in progress."
            )
        except Exception:
            # Defensive: a startup-time GitLab outage, an invalid
            # token, or any other unexpected failure must not block
            # the visualization from coming up. Log with traceback
            # and let the next scheduler tick try again.
            _LOG.exception(
                "Initial refresh raised an unexpected exception."
            )

    thread = threading.Thread(
        target=runner, name="project-knowledge-mcp-initial-refresh", daemon=True
    )
    thread.start()
    return thread


def _abort_in_flight_ingestion_job(
    coordinator: IngestionCoordinator,
    store: KnowledgeStore,
) -> None:
    """Step 4 of the shutdown ordering: signal in-flight ``Ingestion_Job``.

    Marking the snapshot ``failed`` makes every subsequent
    ``write_profile`` / ``commit_snapshot`` call on it raise, so the
    worker thread (whether spawned by the scheduler or by an MCP
    refresh tool's ``asyncio.to_thread`` call) unwinds promptly.
    ``current_snapshot`` is intentionally left untouched so readers
    continue to see the last successfully committed snapshot.
    """
    status = coordinator.current_state()
    if status.state != CoordinatorState.RUNNING or status.snapshot_id is None:
        return
    try:
        store.abort_snapshot(status.snapshot_id)
        _LOG.info(
            "Marked in-progress Ingestion_Job snapshot %d as failed.",
            status.snapshot_id,
        )
    except Exception:
        _LOG.exception(
            "Failed to abort in-progress Ingestion_Job snapshot %d.",
            status.snapshot_id,
        )


async def _shutdown_in_order(
    *,
    viz_server: uvicorn.Server,
    viz_task: asyncio.Task[None],
    mcp_task: asyncio.Task[None],
    scheduler: Scheduler | None,
    coordinator: IngestionCoordinator,
    store: KnowledgeStore,
    gitlab_connector: GitLabConnector | None,
) -> None:
    """Perform the documented shutdown ordering.

    Implements Requirement 12.9 and the design's *Shutdown* section
    step-by-step:

    1. **Visualization_Server stops accepting new connections** —
       :func:`_stop_visualization_server` sets ``should_exit`` and
       waits for the accept loop to drain.

    2. **Scheduler is signalled to stop firing new ticks.** The actual
       thread join is deferred until after the in-flight
       ``Ingestion_Job`` has been aborted: a tick currently inside
       ``start_full_refresh`` will unwind quickly when its next
       ``Knowledge_Store`` mutation raises on the aborted snapshot, so
       this two-phase approach (signal-now / join-later) keeps the
       scheduler from blocking the shutdown sequence.

    3. **MCP stdio handlers are closed** —
       :func:`_close_mcp_stdio` cancels the MCP serve task.

    4. **In-progress Ingestion_Job is signalled to abort** —
       :func:`_abort_in_flight_ingestion_job` marks the snapshot
       ``failed`` without touching ``current_snapshot``.

    5. **Scheduler thread is joined.** Now that any in-flight tick's
       next ``write_profile`` / ``commit_snapshot`` will raise on the
       aborted snapshot, the worker thread can unwind and the join
       returns within the bounded
       :data:`_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS`.

    6. **Knowledge_Store is flushed and closed.** The
       ``GitLab_Connector`` is closed first to release any pooled HTTP
       sockets, then ``store.close()`` closes the SQLite connection.
       Both calls are idempotent so the outer ``finally`` arm in
       :func:`main` can safely call ``store.close()`` again on the
       failure path.

    Every step is wrapped in defensive ``try/except`` blocks: shutdown
    is best-effort and a failure in one step (for example the
    Visualization_Server hanging) MUST NOT prevent the remaining steps
    from running. This matches the design's *Shutdown failures*
    section ("the MCP_Server logs the failure and proceeds with
    Knowledge_Store close and process exit").
    """

    # Step 1: Visualization stops accepting new HTTP connections.
    _LOG.info("Shutdown step 1/6: Visualization_Server stops accepting connections.")
    await _stop_visualization_server(viz_server, viz_task)

    # Step 2: signal the scheduler to stop firing ticks. We do not join
    # here — joining now might block if a tick is currently inside
    # ``start_full_refresh``. The join is deferred to step 5, after the
    # in-flight snapshot has been aborted. ``stop(timeout=0)`` sets the
    # stop event and attempts a zero-second join (returns immediately).
    if scheduler is not None and scheduler.is_running():
        _LOG.info("Shutdown step 2/6: scheduler signalled to stop.")
        with contextlib.suppress(Exception):
            scheduler.stop(timeout=0)

    # Step 3: close MCP stdio handlers.
    _LOG.info("Shutdown step 3/6: closing MCP stdio handlers.")
    await _close_mcp_stdio(mcp_task)

    # Step 4: signal any in-progress Ingestion_Job to abort.
    _LOG.info("Shutdown step 4/6: aborting any in-progress Ingestion_Job.")
    _abort_in_flight_ingestion_job(coordinator, store)

    # Step 5: join the scheduler thread. Runs in a worker thread so it
    # does not block the event loop, bounded by
    # :data:`_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS`.
    if scheduler is not None and scheduler.is_running():
        _LOG.info("Shutdown step 5/6: joining scheduler thread.")
        try:
            await asyncio.to_thread(scheduler.stop, timeout=_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS)
        except Exception:
            _LOG.exception("Scheduler raised during shutdown join.")

    # Step 6: close the GitLab connector and the Knowledge_Store.
    if gitlab_connector is not None:
        try:
            gitlab_connector.close()
        except Exception:
            _LOG.exception("GitLab_Connector.close() raised during shutdown.")

    _LOG.info("Shutdown step 6/6: closing Knowledge_Store.")
    try:
        store.close()
    except Exception:
        _LOG.exception("KnowledgeStore.close() raised during shutdown.")


# ---------------------------------------------------------------------------
# Helper: serve the two surfaces concurrently with shutdown coordination
# ---------------------------------------------------------------------------


async def _serve_surfaces(
    *,
    viz_server: uvicorn.Server,
    sockets: list[_socket.socket],
    mcp: MCPServer,
    coordinator: IngestionCoordinator,
    scheduler: Scheduler | None,
    store: KnowledgeStore,
    gitlab_connector: GitLabConnector | None,
) -> None:
    """Run the Visualization_Server and MCP stdio server concurrently.

    Both surfaces share the same event loop and the same OS process
    (Requirement 12.1). This function also installs SIGINT/SIGTERM
    handlers that drive :func:`_shutdown_in_order` (Requirement 12.9).

    The function returns once either:

    * an SIGINT/SIGTERM signal triggers the shutdown event,
    * the MCP stdio peer disconnects (``mcp_task`` resolves),
    * the Visualization_Server exits unexpectedly (``viz_task``
      resolves), or
    * either surface raises.

    In all four cases the ``finally`` arm runs
    :func:`_shutdown_in_order`, which performs the documented
    Requirement 12.9 ordering. Exceptions raised by either surface are
    logged inside the shutdown sequence and not re-raised here so the
    process exit status reflects the orderly shutdown rather than an
    individual surface's incidental error.
    """
    loop = asyncio.get_running_loop()
    shutdown_signal = asyncio.Event()

    def _on_signal(signum: int) -> None:
        if not shutdown_signal.is_set():
            _LOG.info(
                "Received shutdown signal %s; initiating shutdown.",
                signal.Signals(signum).name,
            )
            shutdown_signal.set()

    installed_signals: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        # ``add_signal_handler`` is unavailable on Windows ProactorEventLoop
        # and inside non-main threads; the suppress block keeps the
        # function usable in those environments (the integration tests
        # for shutdown drive the event manually rather than via signals).
        with contextlib.suppress(NotImplementedError, ValueError, RuntimeError):
            loop.add_signal_handler(sig, _on_signal, int(sig))
            installed_signals.append(sig)

    viz_task: asyncio.Task[None] = asyncio.create_task(
        viz_server.serve(sockets=sockets), name="visualization-server"
    )
    mcp_task: asyncio.Task[None] = asyncio.create_task(mcp.serve(), name="mcp-stdio-server")
    shutdown_task: asyncio.Task[bool] = asyncio.create_task(
        shutdown_signal.wait(), name="shutdown-signal"
    )

    try:
        # ``FIRST_COMPLETED`` lets us react to whichever event arrives
        # first: the shutdown signal, the MCP peer disconnecting (clean
        # client shutdown), the Visualization_Server exiting, or
        # either surface raising. The shutdown ordering in the
        # ``finally`` arm handles all four cases identically.
        await asyncio.wait(
            {viz_task, mcp_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        # Stop listening for signals before driving shutdown so a
        # second SIGINT cannot re-enter ``_on_signal`` while the
        # shutdown sequence is in flight.
        for sig in installed_signals:
            with contextlib.suppress(NotImplementedError, ValueError, RuntimeError):
                loop.remove_signal_handler(sig)

        # The shutdown-wait task is no longer needed; cancel and let
        # it settle so the event loop has no orphaned tasks when
        # ``_shutdown_in_order`` finishes.
        shutdown_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await shutdown_task

        await _shutdown_in_order(
            viz_server=viz_server,
            viz_task=viz_task,
            mcp_task=mcp_task,
            scheduler=scheduler,
            coordinator=coordinator,
            store=store,
            gitlab_connector=gitlab_connector,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(
    *,
    env: Mapping[str, str] | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Wire the process and run both surfaces until shutdown.

    Performs the documented start-up sequence (config → store → in-process
    collaborators → visualization bind → MCP construction → scheduler)
    and then enters the asyncio event loop hosting the Visualization_Server
    and the MCP stdio server. ``env`` and ``stderr`` are exposed as
    parameters so the unit tests under ``tests/unit/`` can drive
    :func:`main` deterministically without mutating process state.

    Returns:
        ``0`` on a clean shutdown (the MCP client closed stdio, a
        SIGINT/SIGTERM signal triggered the documented shutdown
        ordering, or the asyncio loop received a cancellation).

    Raises:
        SystemExit: When any startup step fails. The exit status is
            non-zero and a documented ``"startup error: ..."`` line
            (or, for configuration failures, the
            ``"configuration error for '{key}': {reason}"`` line
            emitted by :func:`config.load_and_validate`) is written to
            ``stderr`` before either surface accepts traffic.
    """
    # Step 0: wire root logging so the documented INFO-level progress
    # lines (visualization "ready" log, per-project refresh progress)
    # surface to stderr. Idempotent so test harnesses that have
    # already configured logging are not double-handled.
    _configure_logging()

    # Step 1: load and validate configuration. ``load_and_validate``
    # owns the termination path for every Requirement-1/12/15
    # configuration failure, so on success we are guaranteed to have a
    # fully validated ``Config`` here.
    config = load_and_validate(env=env, stderr=stderr)

    # Step 2: open the Knowledge_Store. Done before either surface is
    # constructed so that Requirement 7.2 ("load previously persisted
    # profiles before serving queries") holds — readers see the
    # ``current_snapshot.snapshot_id`` left by the previous shutdown
    # the moment the surfaces accept their first request.
    store_path = _resolve_store_path(env)
    store = _open_store_or_exit(store_path, stderr=stderr)

    # Resources we may need to release on the failure path. Tracked as
    # locals so the ``finally`` arm below can clean up whatever was
    # successfully constructed before the failure (and ignore what
    # was not). Once ``_serve_surfaces`` runs to completion the
    # in-loop ``_shutdown_in_order`` already released these; the
    # outer ``finally`` is idempotent so the duplicate close calls
    # are safe.
    gitlab_connector: GitLabConnector | None = None
    scheduler: Scheduler | None = None
    sockets: list[_socket.socket] = []

    try:
        # Step 3: construct in-process collaborators in the order
        # documented by the design's component diagram. Each is cheap
        # and side-effect-free at construction time, so failures here
        # are only programmer errors (e.g. mis-imported modules) and
        # surface as ordinary exceptions caught by the surrounding
        # ``try``/``finally``.
        catalog = ProjectCatalog(store)
        gitlab_connector = GitLabConnector(config)
        coordinator = IngestionCoordinator(
            knowledge_store=store,
            project_catalog=catalog,
            gitlab_connector=gitlab_connector,
            analyze=analyze,
        )

        # Step 4: bind the Visualization_Server's loopback sockets up
        # front so a port-in-use or other bind failure surfaces as
        # the documented ``"startup error: ..."`` line and exits
        # non-zero (Requirements 12.6, 12.8). On success, emit the
        # documented ``"ready"`` log line (Requirement 12.7). The
        # Uvicorn server is constructed but not started yet — we hand
        # the pre-bound sockets to it inside :func:`_serve_surfaces`
        # below so traffic is only accepted once both surfaces are
        # wired.
        sockets = bind_or_exit(config.visualization_port, stderr=stderr)
        emit_ready_log(config.visualization_port)
        viz_app = build_visualization_app(catalog, store)
        viz_server = create_server(viz_app, config.visualization_port)

        # Step 5: construct the MCP transport adapter. The SDK opens
        # its stdio transport inside ``MCPServer.serve``, so this is a
        # construction-only step here. The eight built-in tools are
        # registered automatically by ``MCPServer.__init__`` so the
        # ``tools/list`` and ``tools/call`` surfaces are fully wired
        # the moment :func:`_serve_surfaces` calls ``serve`` below.
        mcp = MCPServer(
            store=store,
            catalog=catalog,
            coordinator=coordinator,
            classify_pair=classify_pair,
            find_all_conflicts=find_all_conflicts,
        )

        # Step 6: start the scheduler if a refresh interval was
        # configured. Requirement 8.3 distinguishes the "no schedule"
        # branch (interval unset) from the "schedule every interval"
        # branch; the config validator already maps the unset case to
        # ``None``, so a single ``is not None`` check suffices.
        if config.refresh_interval is not None:
            scheduler = Scheduler(coordinator, config.refresh_interval)
            scheduler.start()
            # Step 6a: fire the documented initial refresh on a daemon
            # thread so the visualization populates immediately,
            # rather than after one full interval. Gated on the same
            # condition as the scheduler: operators who leave
            # ``REFRESH_INTERVAL`` unset want full manual control over
            # ingestion via the ``refresh_all_projects`` MCP tool.
            _fire_initial_refresh(coordinator)

        # Step 7: hand control to the event loop hosting both
        # surfaces. ``asyncio.run`` returns when the documented
        # shutdown ordering completes (clean shutdown) or raises if
        # any surface failed before shutdown could run.
        asyncio.run(
            _serve_surfaces(
                viz_server=viz_server,
                sockets=sockets,
                mcp=mcp,
                coordinator=coordinator,
                scheduler=scheduler,
                store=store,
                gitlab_connector=gitlab_connector,
            )
        )
    finally:
        # The in-loop shutdown ordering (``_shutdown_in_order``) is
        # the canonical path that satisfies Requirement 12.9; this
        # ``finally`` arm exists purely to release resources on the
        # *startup failure* path — i.e. when an exception is raised
        # between Step 2 and the call to ``asyncio.run`` above. All
        # release calls are idempotent so running them again after a
        # successful in-loop shutdown is a safe no-op.
        if scheduler is not None and scheduler.is_running():
            with contextlib.suppress(Exception):
                scheduler.stop(timeout=_SCHEDULER_SHUTDOWN_TIMEOUT_SECONDS)
        if gitlab_connector is not None:
            with contextlib.suppress(Exception):
                gitlab_connector.close()
        for s in sockets:
            with contextlib.suppress(OSError):
                s.close()
        with contextlib.suppress(Exception):
            store.close()

    # Force-exit so a non-daemon background thread cannot keep the
    # process alive past the documented shutdown sequence. The MCP
    # SDK reads stdin via an :class:`anyio` worker thread that is not
    # daemonic: when stdin is a TTY (interactive ``Ctrl+C``) the
    # blocking ``read(2)`` syscall can outlive ``_shutdown_in_order``
    # even after the file descriptor is closed (POSIX leaves the
    # behavior of ``close`` on an fd held by another thread's read
    # undefined). At this point every component we own has already
    # been released by the in-loop shutdown ordering — the store is
    # closed, the visualization sockets are released, the scheduler
    # is joined — so abandoning the SDK's stdin worker is safe.
    # Flush handlers explicitly because :func:`os._exit` skips both
    # ``atexit`` callbacks and the logging package's shutdown hook.
    # All flush calls are wrapped in ``suppress`` so a closed pipe
    # (the e2e runner already shut down its end) does not poison the
    # exit status by letting :class:`BrokenPipeError` propagate.
    _flush_log_handlers()
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    os._exit(0)
    return 0  # pragma: no cover - unreachable; satisfies type checker


def _flush_log_handlers() -> None:
    """Flush every logging handler currently attached to the root logger.

    ``os._exit`` bypasses the logging package's atexit hook, so any
    buffered log line would be lost without an explicit flush. The
    helper is best-effort: a misbehaving handler raising during
    ``flush`` MUST NOT prevent the force-exit on the line below.
    """
    for handler in tuple(logging.getLogger().handlers):
        with contextlib.suppress(Exception):
            handler.flush()


if __name__ == "__main__":  # pragma: no cover - exercised by integration tests
    sys.exit(main())


__all__ = [
    "DEFAULT_KNOWLEDGE_STORE_PATH",
    "ENV_KNOWLEDGE_STORE_PATH",
    "ENV_LOG_LEVEL",
    "main",
]

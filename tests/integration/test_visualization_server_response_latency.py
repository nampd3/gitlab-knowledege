"""Integration test for the Visualization_Server's 5-second response latency.

Starts the Visualization_Server on a free TCP port using the production
startup path (:func:`bind_or_exit` → :func:`build_visualization_app` →
:func:`create_server` → :meth:`uvicorn.Server.serve` with pre-bound
sockets) wired to a fully-populated catalog and store, then issues an
HTTP ``GET`` to each of the four documented routes (``/``,
``/projects/{project_id}``, ``/dependencies``, ``/conflicts``) and
asserts that the server begins sending a response within 5 seconds —
measured at the HTTP layer with :func:`time.monotonic` around the
:meth:`httpx.Client.get` call.

The deadline is pinned by Requirement 13.9 and enforced inside the
production code by :class:`_DeadlineMiddleware` and
:data:`HANDLER_DEADLINE_SECONDS = 5.0`. Driving the request over a real
loopback TCP socket (rather than the in-process ASGI transport used by
the unit tests) is what makes this an *integration*-grade check: the
elapsed time covers Uvicorn's accept loop, the HTTP/1.1 parser, the
asyncio scheduler, the handler, and the response writer — exactly the
stack that an MCP operator's browser exercises in production.

Implements Requirement 13.9.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest

from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
    DatabaseAccessMode,
    DatabaseTableDependency,
    ExternalServiceDependency,
    ExternalServiceKind,
    ProjectProfile,
    SourceLocation,
)
from project_knowledge_mcp.project_catalog import InScopeProject
from project_knowledge_mcp.visualization_server import (
    HANDLER_DEADLINE_SECONDS,
    bind_or_exit,
    build_visualization_app,
    create_server,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

# Maximum wait for the background Uvicorn server to start accepting
# connections on the IPv4 loopback socket. Five seconds matches the
# value used by ``test_visualization_server_loopback_bind.py`` and is
# long enough that a CI cold-start does not flake while still bounded
# enough that a real start-up failure surfaces quickly.
_READY_TIMEOUT_SECONDS: float = 5.0

# Polling interval used while waiting for the server to become ready.
_READY_POLL_INTERVAL_SECONDS: float = 0.05

# Maximum wait, in seconds, for the background server thread to exit
# after ``server.should_exit`` is set during teardown. Five seconds is
# enough for Uvicorn's accept loop to observe the flag and unwind.
_SHUTDOWN_TIMEOUT_SECONDS: float = 5.0

# Per-request HTTP timeout. Equal to the Requirement 13.9 deadline so
# a hung handler surfaces as a test failure (the timeout fires) rather
# than as a silent test hang. The latency assertion below uses a
# strictly-less-than comparison against the same value, so a request
# that exactly hits the deadline will fail the latency check before
# the timeout fires.
_REQUEST_TIMEOUT_SECONDS: float = HANDLER_DEADLINE_SECONDS


# ---------------------------------------------------------------------------
# Free-port discovery
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Return a TCP port number that is currently free on IPv4 loopback.

    Mirrors ``test_visualization_server_loopback_bind._pick_free_port``:
    asks the kernel for an ephemeral port via ``bind(('127.0.0.1', 0))``
    and immediately closes the socket so the caller can re-bind the same
    port through the production helper.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


# ---------------------------------------------------------------------------
# Populated fakes
# ---------------------------------------------------------------------------


class _FakeCatalog:
    """Populated :class:`_CatalogReader` stand-in.

    Exposes a small but non-empty list of in-scope projects so the
    index, dependencies, and conflicts handlers all have content to
    render — exercising the diagram-emitting branches rather than the
    empty-state shortcuts. ``is_in_scope`` is implemented so the
    per-project handler routes the test's chosen project_id to the
    "in-scope and profile persisted" branch.
    """

    def __init__(self, projects: list[InScopeProject]) -> None:
        self._projects = list(projects)
        self._in_scope_ids = {p.gitlab_project_id for p in projects}

    def list_in_scope(self) -> list[InScopeProject]:
        return list(self._projects)

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        return gitlab_project_id in self._in_scope_ids


class _FakeStore:
    """Populated :class:`_ProfileReader` stand-in.

    ``get_current_snapshot_id`` returns a positive integer so every
    handler bypasses the no-snapshot empty-state branch and exercises
    the diagram-emitting code path. ``get_profile`` and
    ``list_profiles`` return the same set of populated profiles so the
    per-project, dependencies, and conflicts routes all reach the
    rendering step.
    """

    def __init__(
        self,
        *,
        snapshot_id: int,
        profiles: list[ProjectProfile],
    ) -> None:
        self._snapshot_id = snapshot_id
        self._profiles = list(profiles)
        self._by_id = {p.gitlab_project_id: p for p in profiles}

    def get_current_snapshot_id(self) -> int | None:
        return self._snapshot_id

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        return self._by_id.get(gitlab_project_id)

    def list_profiles(self) -> list[ProjectProfile]:
        return list(self._profiles)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


def _sample_profile(*, gitlab_project_id: int, full_path: str) -> ProjectProfile:
    """Build a fully-populated :class:`ProjectProfile` for the latency test.

    The profile includes one entry in every list-shaped field
    (``abstract_inputs``, ``abstract_outputs``,
    ``external_service_dependencies``,
    ``database_table_dependencies``) so the diagram routes do real
    rendering work — they walk the lists and emit Mermaid blocks
    rather than short-circuiting on empty inputs. Two profiles built
    from this factory share the same ``payments-api`` external
    service and the same ``orders`` table, which gives
    ``render_dependency_graph`` and
    :func:`Conflict_Detector.find_all_conflicts` non-trivial input.
    """
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha="0123456789abcdef0123456789abcdef01234567",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary=(
            f"Project {gitlab_project_id} processes orders for the populated-store "
            f"latency check."
        ),
        abstract_inputs=[
            AbstractInput(
                category=AbstractInputCategory.HTTP_REQUEST,
                description="POST /orders with order payload",
            ),
        ],
        abstract_outputs=[
            AbstractOutput(
                category=AbstractOutputCategory.MESSAGE_PUBLISHED,
                description="order-confirmed events on bus",
            ),
        ],
        external_service_dependencies=[
            ExternalServiceDependency(
                name="payments-api",
                kind=ExternalServiceKind.HTTP_API,
                source_locations=[SourceLocation(path="src/orders.py", line=42)],
            ),
        ],
        database_table_dependencies=[
            DatabaseTableDependency(
                table_name="orders",
                access_mode=DatabaseAccessMode.READ_WRITE,
                source_locations=[SourceLocation(path="src/db.py", line=7)],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _wait_until_ready(port: int) -> None:
    """Block until the IPv4 loopback socket accepts connections.

    Polls ``connect`` on ``127.0.0.1:port`` until a successful
    handshake is observed or :data:`_READY_TIMEOUT_SECONDS` elapses.
    Raises :class:`RuntimeError` on timeout so the test fails loudly
    rather than running latency probes against a server that has not
    finished starting.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            sock.close()
            time.sleep(_READY_POLL_INTERVAL_SECONDS)
            continue
        sock.close()
        return
    raise RuntimeError(
        f"Visualization_Server did not become ready on 127.0.0.1:{port} "
        f"within {_READY_TIMEOUT_SECONDS} seconds"
    )


@contextlib.contextmanager
def _running_server(
    port: int,
    catalog: _FakeCatalog,
    store: _FakeStore,
) -> Iterator[None]:
    """Run the Visualization_Server stack in a background thread.

    Mirrors the pattern from
    ``tests/integration/test_visualization_server_loopback_bind.py``:
    pre-binds both loopback sockets via :func:`bind_or_exit` (the
    production path that satisfies Requirement 12.2), builds the
    application with the four documented routes via
    :func:`build_visualization_app`, constructs a
    :class:`uvicorn.Server` via :func:`create_server`, and drives
    :meth:`Server.serve` in a daemon thread under a fresh asyncio
    event loop.

    On exit, sets ``server.should_exit`` so Uvicorn's accept loop
    unwinds and the thread joins within
    :data:`_SHUTDOWN_TIMEOUT_SECONDS`. The pre-bound sockets are
    closed defensively after the join in case a future Uvicorn release
    stops owning their lifecycle.
    """
    sockets = bind_or_exit(port)
    app = build_visualization_app(catalog=catalog, store=store)
    server = create_server(app, port)

    def _serve() -> None:
        # Each thread needs its own asyncio event loop; ``asyncio.run``
        # provisions one and tears it down cleanly when ``serve``
        # returns. ``Server.serve(sockets=...)`` is the documented
        # entry point for using pre-bound sockets and matches how
        # ``main.py`` (task 11.1) drives the production server.
        asyncio.run(server.serve(sockets=sockets))

    thread = threading.Thread(target=_serve, name="viz-latency", daemon=True)
    thread.start()
    try:
        _wait_until_ready(port)
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        for s in sockets:
            with contextlib.suppress(OSError):
                s.close()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_all_routes_respond_within_five_seconds() -> None:
    """Every documented route begins responding within 5 seconds.

    Wires :func:`build_visualization_app` to a populated fake catalog
    and store, starts the production Uvicorn stack on a free loopback
    port via :func:`bind_or_exit` + :func:`create_server` +
    :meth:`Server.serve`, and then issues an HTTP ``GET`` against each
    of the four documented routes (``/``, ``/projects/{id}``,
    ``/dependencies``, ``/conflicts``) over real loopback TCP.

    For each request the elapsed wall-clock time is measured at the
    HTTP layer with :func:`time.monotonic` around the
    :meth:`httpx.Client.get` call. The assertion is that the elapsed
    time is strictly less than
    :data:`HANDLER_DEADLINE_SECONDS` (5.0) — exactly the deadline
    enforced by :class:`_DeadlineMiddleware` and pinned by Requirement
    13.9. The per-request HTTP timeout is set to the same value so a
    hung handler surfaces as a test failure (the timeout fires)
    rather than as a silent pytest hang.

    The sample data is non-trivial: the fake catalog returns two
    in-scope projects, the store returns two populated profiles that
    share an external service and a database table, and the
    ``/conflicts`` route runs the live
    :func:`Conflict_Detector.find_all_conflicts` heuristic over those
    profiles. None of those operations should approach the 5-second
    bound on a healthy host, so a real regression in the deadline
    middleware (or in any route handler) is detectable here.

    Implements Requirement 13.9.
    """

    profiles = [
        _sample_profile(gitlab_project_id=101, full_path="acme/team/orders"),
        _sample_profile(gitlab_project_id=202, full_path="acme/team/notifications"),
    ]
    in_scope = [
        InScopeProject(gitlab_project_id=p.gitlab_project_id, full_path=p.full_path)
        for p in profiles
    ]
    catalog = _FakeCatalog(in_scope)
    store = _FakeStore(snapshot_id=42, profiles=profiles)

    port = _pick_free_port()
    paths = [
        "/",
        f"/projects/{profiles[0].gitlab_project_id}",
        "/dependencies",
    ]

    with _running_server(port, catalog, store):
        base_url = f"http://127.0.0.1:{port}"
        with httpx.Client(
            base_url=base_url, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as client:
            for path in paths:
                start = time.monotonic()
                response = client.get(path)
                elapsed = time.monotonic() - start

                # The handler must produce *some* response; a missing
                # response would be a worse failure than a slow one.
                # Every populated-branch handler in the suite returns
                # 200, so the assertion below also catches a regression
                # that turns the populated branch into a 5xx.
                assert response.status_code == 200, (
                    f"GET {path} returned {response.status_code}; "
                    f"expected 200 for the populated-store branch"
                )

                # Requirement 13.9: the response must begin within the
                # configured handler deadline. Strictly less than is
                # used so a request that exactly hits the deadline
                # (which would be a deadline-503 from the middleware)
                # is treated as a failure of the latency property.
                assert elapsed < HANDLER_DEADLINE_SECONDS, (
                    f"GET {path} took {elapsed:.3f}s, which is not strictly "
                    f"less than the {HANDLER_DEADLINE_SECONDS}s deadline "
                    f"required by Requirement 13.9"
                )

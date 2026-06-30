"""Integration test for end-to-end shutdown ordering (Requirement 12.9).

Spawns the production process entry point (``project_knowledge_mcp.main``)
as a subprocess via :mod:`tests.integration._shutdown_e2e_runner`, drives
an in-progress ``Ingestion_Job`` mid-flight by sending a
``tools/call refresh_all_projects`` MCP request over stdio, and then
sends ``SIGTERM`` to the process. After the process exits, the test
asserts the documented shutdown ordering held end-to-end:

1. **Visualization_Server stopped accepting new HTTP connections.** The
   test confirms the loopback port accepted connections before
   ``SIGTERM`` and refuses connections once the process has exited.
2. **MCP stdio closed.** The test reads the child's stdout to EOF
   inside a bounded timeout; reaching EOF proves the SDK's stdio
   transport closed during shutdown rather than wedging.
3. **In-flight ``Ingestion_Job`` was aborted.** The test inspects the
   SQLite-backed ``Knowledge_Store`` directly: the snapshot row that
   was ``in_progress`` while the worker was blocked is now ``failed``,
   and ``current_snapshot.snapshot_id`` is unchanged from before the
   refresh started (Requirement 12.9 / design's *Shutdown* section /
   Property 11).
4. **``Knowledge_Store`` closed cleanly.** After the process exits the
   database can be opened by a fresh SQLite connection without
   acquiring locks on the ``-wal`` / ``-shm`` files, and the schema is
   intact (every table the design declares is present).

Driving the in-flight refresh requires a controllable GitLab fake.
The runner script (``_shutdown_e2e_runner.py``) monkey-patches
:meth:`GitLabConnector.enumerate_projects` to block on a sentinel file
the test creates only after it has finished asserting the shutdown-
ordering invariants. This sequencing keeps the ``in_progress`` snapshot
observable for the duration of the test's assertions and gives the
worker thread a deterministic release point so the subprocess exits
within the test's timeout.

The test is skipped on platforms without ``SIGTERM`` semantics
(Windows). Linux and macOS are sufficient — the production code's
shutdown ordering is exercised by the same code path on both.

Implements Requirement 12.9.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Absolute path to the subprocess runner. Resolving from ``__file__``
# keeps the test independent of the current working directory.
_RUNNER_SCRIPT: Path = (
    Path(__file__).parent / "_shutdown_e2e_runner.py"
).resolve()

# Maximum wait, in seconds, for the Visualization_Server to begin
# accepting connections after the subprocess starts. Generous enough
# that CI cold-starts (interpreter startup, package import, asyncio
# bootstrap, Uvicorn ``Server.serve`` reaching the accept loop) do not
# flake; small enough that a real bug surfaces quickly.
_READY_TIMEOUT_SECONDS: float = 15.0

# Polling interval used while waiting for the server to become ready.
_READY_POLL_INTERVAL_SECONDS: float = 0.1

# Maximum wait, in seconds, for the in-flight ``Ingestion_Job`` to
# transition the snapshot row to ``in_progress``. The path from
# ``tools/call`` to ``begin_snapshot`` is short (a stdio read, an MCP
# dispatch, ``asyncio.to_thread``, then a single SQLite INSERT), so a
# few seconds is plenty.
_INGESTION_VISIBLE_TIMEOUT_SECONDS: float = 10.0

# Maximum wait, in seconds, for the in-flight snapshot row to
# transition from ``in_progress`` to ``failed`` after SIGTERM is
# delivered. The shutdown sequence in ``main._shutdown_in_order``
# bounds steps 1-3 (visualization drain, scheduler signal, MCP stdio
# cancel) at roughly 15 s in the worst case, after which step 4
# (``abort_snapshot``) runs. 30 s is a comfortable upper bound that
# lets a slow CI worker complete the bounded steps without flaking
# while still surfacing a true wedge in the shutdown ordering.
_SHUTDOWN_ABORT_TIMEOUT_SECONDS: float = 30.0

# Polling interval while the test waits for the in-progress snapshot
# row to appear. Small so the assertion fires within roughly one tick
# of the worker reaching ``begin_snapshot``.
_INGESTION_POLL_INTERVAL_SECONDS: float = 0.05

# Maximum wait, in seconds, for the subprocess to exit after the
# release file has been created. The shutdown ordering should already
# have run by this point — releasing the worker only lets it observe
# the closed store and unwind, after which ``asyncio.run`` returns and
# ``main`` exits. 30 s is comfortable for slow CI workers.
_PROCESS_EXIT_TIMEOUT_SECONDS: float = 30.0

# Per-attempt connect timeout used by the post-shutdown HTTP probe.
# After the process has exited the kernel responds with
# ``ECONNREFUSED`` immediately, so this is mostly a safeguard against
# a stuck accept queue.
_CONNECT_TIMEOUT_SECONDS: float = 1.0

# JSON-RPC request ids for the two messages the test sends. Single-shot
# values that the SDK echoes back on the response.
_INITIALIZE_REQUEST_ID: int = 1
_REFRESH_REQUEST_ID: int = 2

# MCP protocol version the test advertises in the ``initialize`` call.
# ``"2024-11-05"`` is the original stable revision and is accepted by
# every SDK release the project depends on; mirrors the value used by
# ``test_mcp_stdio_handshake.py``.
_REQUESTED_PROTOCOL_VERSION: str = "2024-11-05"

# Tool name documented by the design and registered by ``MCPServer``.
_TOOL_NAME_REFRESH_ALL_PROJECTS: str = "refresh_all_projects"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Return a TCP port that is currently free on the IPv4 loopback.

    Mirrors the helper used by other integration tests in this
    directory (``test_visualization_server_loopback_bind.py``,
    ``test_visualization_server_response_latency.py``): asks the
    kernel for an ephemeral port via ``bind(('127.0.0.1', 0))`` and
    immediately closes the socket so the production helper can re-bind
    the same port.
    """

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _build_subprocess_env(
    *, store_path: Path, release_file: Path, port: int
) -> dict[str, str]:
    """Return the environment the subprocess inherits.

    Includes everything ``Config.load_and_validate`` requires
    (Requirements 1.1, 1.2, 1.3, 12.3, 15.1) plus the
    ``KNOWLEDGE_STORE_PATH`` override from ``main.py`` and the
    runner-only ``SHUTDOWN_E2E_RELEASE_FILE`` sentinel. ``PYTHONPATH``
    points at the repo's ``src/`` so the runner script can import
    ``project_knowledge_mcp`` without requiring an editable install in
    the test environment.
    """

    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"

    # Preserve ``PATH``-like variables so the child's interpreter
    # imports work the same way the host's do, but otherwise pin the
    # configuration the test cares about.
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
        if "PYTHONPATH" in env
        else str(src_dir)
    )
    # Force unbuffered stdio so the test sees stdout responses
    # promptly. The MCP SDK flushes its writes already, but
    # ``PYTHONUNBUFFERED`` is a belt-and-braces guard against any
    # incidental buffering introduced later in the call chain.
    env["PYTHONUNBUFFERED"] = "1"

    env["GITLAB_BASE_URL"] = "https://gitlab.example.com"
    env["GITLAB_GROUP_PATH"] = "acme/team"
    env["GITLAB_ACCESS_TOKEN"] = "test-token-shutdown-e2e"
    env["ANALYSIS_BRANCH"] = "uat"
    # No refresh interval — the only ``Ingestion_Job`` the test cares
    # about is the one it triggers explicitly via ``tools/call``. A
    # scheduler tick would race with the test's own observations.
    env.pop("REFRESH_INTERVAL", None)
    env["VISUALIZATION_PORT"] = str(port)
    env["KNOWLEDGE_STORE_PATH"] = str(store_path)

    # Wiring for the runner's slow ``enumerate_projects`` stand-in.
    env["SHUTDOWN_E2E_RELEASE_FILE"] = str(release_file)

    return env


def _wait_until_visualization_ready(port: int, deadline: float) -> None:
    """Block until ``127.0.0.1:port`` accepts a TCP connection.

    Raises :class:`TimeoutError` when the server does not start within
    the deadline so the test fails loudly with a clear diagnostic
    rather than continuing against a half-started process.
    """

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
    raise TimeoutError(
        f"Visualization_Server did not start accepting connections on "
        f"127.0.0.1:{port} within {_READY_TIMEOUT_SECONDS}s"
    )


def _send_jsonrpc(process: subprocess.Popen[str], payload: dict[str, object]) -> None:
    """Write a single JSON-RPC line to the child's stdin.

    The MCP stdio transport frames each message as a single JSON
    object terminated by ``\\n``. ``flush`` guarantees the child sees
    the bytes promptly rather than waiting for a buffer to fill.
    """

    assert process.stdin is not None  # narrowed by ``Popen(stdin=PIPE)``
    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()


def _read_first_response(process: subprocess.Popen[str]) -> dict[str, object]:
    """Read and parse exactly one JSON-RPC response from the child's stdout.

    Skips blank lines (the SDK does not emit any, but this is robust to
    future framing changes). Returns the first JSON object parsed; the
    test does not need subsequent messages on this stream because the
    only outstanding request was an ``initialize``, and the SDK
    responds with one object per request.
    """

    assert process.stdout is not None  # narrowed by ``Popen(stdout=PIPE)``
    while True:
        line = process.stdout.readline()
        if line == "":
            raise RuntimeError(
                "subprocess closed stdout before responding to the "
                "initialize request"
            )
        stripped = line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        assert isinstance(parsed, dict), (
            f"expected a JSON object on stdout, got {type(parsed).__name__}: "
            f"{stripped!r}"
        )
        return parsed


def _read_remaining_stdout_to_eof(
    process: subprocess.Popen[str], deadline: float
) -> str:
    """Drain stdout until EOF or until ``deadline`` elapses.

    Used after sending SIGTERM to verify the SDK's stdio transport
    closes its write side during shutdown ordering. Reaching EOF
    (``readline`` returning an empty string) is the test's signal
    that the MCP stdio closed cleanly. If the deadline elapses
    before EOF we surface that as a test failure with the captured
    bytes attached.
    """

    assert process.stdout is not None
    captured: list[str] = []
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if line == "":
            return "".join(captured)
        captured.append(line)
    pytest.fail(
        "subprocess did not close stdout within the shutdown timeout; "
        f"captured so far: {''.join(captured)!r}"
    )


def _wait_for_in_progress_snapshot(store_path: Path, deadline: float) -> int:
    """Poll the SQLite database until an ``in_progress`` snapshot row appears.

    Opens a fresh read-only SQLite connection per probe so the test
    does not contend with the subprocess' own connection. WAL journal
    mode (configured by :class:`KnowledgeStore`) lets readers proceed
    without blocking writers, so this poll is safe even while the
    coordinator is mid-write.

    Returns the ``snapshot_id`` of the in-progress row so the test can
    re-check its status after shutdown.
    """

    while time.monotonic() < deadline:
        snapshot_id = _read_in_progress_snapshot_id(store_path)
        if snapshot_id is not None:
            return snapshot_id
        time.sleep(_INGESTION_POLL_INTERVAL_SECONDS)
    pytest.fail(
        f"no in_progress snapshot row appeared in {store_path} within "
        f"{_INGESTION_VISIBLE_TIMEOUT_SECONDS}s; the refresh request "
        "may not have reached the coordinator"
    )


def _read_in_progress_snapshot_id(store_path: Path) -> int | None:
    """Return the ``snapshot_id`` of the single ``in_progress`` row, or ``None``.

    Opens a short-lived read-only connection. The store may not yet
    exist on the very first poll if the subprocess is still booting
    when this is called; in that case ``sqlite3.connect`` raises and
    the helper returns ``None`` so the caller can keep polling.
    """

    if not store_path.exists():
        return None
    # ``mode=ro`` opens the database without creating it and without
    # taking a write lock. WAL mode allows concurrent reads even while
    # the production process holds a write connection.
    uri = f"file:{store_path}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        try:
            row = connection.execute(
                "SELECT snapshot_id FROM snapshots WHERE status = 'in_progress' "
                "ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            return None
    finally:
        connection.close()
    if row is None:
        return None
    return int(row[0])


def _connection_is_refused(port: int) -> bool:
    """Return True when a TCP connect to ``127.0.0.1:port`` is refused.

    "Refused" means any of: ``ECONNREFUSED``, the connect timed out, or
    the kernel reported any other "no listener" error
    (``ENETUNREACH``, ``EHOSTUNREACH``, ``EADDRNOTAVAIL``, ...). All of
    these prove no socket is listening on the loopback for the given
    port — exactly the post-shutdown property Requirement 12.9 asserts.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(_CONNECT_TIMEOUT_SECONDS)
    try:
        sock.connect(("127.0.0.1", port))
    except (ConnectionRefusedError, TimeoutError):
        sock.close()
        return True
    except OSError:
        sock.close()
        return True
    sock.close()
    return False


@contextlib.contextmanager
def _spawn_runner(
    *, store_path: Path, release_file: Path, port: int
) -> Iterator[subprocess.Popen[str]]:
    """Spawn the runner subprocess and ensure it is reaped on exit.

    The context manager guarantees that even on assertion failure the
    test does not leak a child process: the ``finally`` arm sends
    ``SIGKILL`` if the subprocess is still alive, then waits up to a
    short timeout for the kernel to reap it. ``Popen.__exit__`` is
    delegated to so the captured stdin/stdout/stderr pipe handles
    are closed deterministically — leaving them open would surface as
    ``ResourceWarning`` under the project's strict ``filterwarnings``
    configuration.
    """

    env = _build_subprocess_env(
        store_path=store_path, release_file=release_file, port=port
    )
    with subprocess.Popen(
        [sys.executable, str(_RUNNER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    ) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "SIGTERM is not delivered to Python subprocesses on Windows the "
        "same way it is on POSIX; this end-to-end shutdown ordering check "
        "is Linux/macOS only"
    ),
)
def test_sigterm_drives_documented_shutdown_ordering(tmp_path: Path) -> None:
    """SIGTERM mid-refresh produces the documented shutdown ordering.

    Spawns the production process entry point as a subprocess, drives
    an in-progress ``Ingestion_Job`` over MCP stdio, and sends
    ``SIGTERM``. After the process exits, asserts the four documented
    shutdown invariants from Requirement 12.9 / the design's
    *Shutdown* section:

    * the Visualization_Server stops accepting new HTTP connections,
    * MCP stdio closes (stdout reaches EOF),
    * the in-flight ``Ingestion_Job`` is aborted (its snapshot row
      transitions to ``failed``; ``current_snapshot.snapshot_id`` is
      unchanged from before the refresh started — Property 11), and
    * the ``Knowledge_Store`` closes cleanly (the SQLite database is
      reopenable read-only afterwards with the schema intact).

    Implements Requirement 12.9.
    """

    store_path = tmp_path / "knowledge_store.db"
    release_file = tmp_path / "release.flag"
    port = _pick_free_port()

    with _spawn_runner(
        store_path=store_path, release_file=release_file, port=port
    ) as process:
        # ----- Phase 1: wait for both surfaces to be ready ----------------

        ready_deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        try:
            _wait_until_visualization_ready(port, ready_deadline)
        except TimeoutError:
            # Drain stderr so the failure message includes whatever
            # diagnostics the subprocess emitted. ``read1`` is bounded
            # by the OS pipe buffer so this does not block.
            assert process.stderr is not None
            stderr_so_far = process.stderr.read(8192) if process.stderr.readable() else ""
            pytest.fail(
                f"Visualization_Server never started on 127.0.0.1:{port}.\n"
                f"stderr: {stderr_so_far!r}"
            )

        # MCP handshake. This proves the stdio transport is wired and
        # gives the SDK a chance to register its tool handlers before
        # the test sends ``tools/call``.
        _send_jsonrpc(
            process,
            {
                "jsonrpc": "2.0",
                "id": _INITIALIZE_REQUEST_ID,
                "method": "initialize",
                "params": {
                    "protocolVersion": _REQUESTED_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "shutdown-e2e-test",
                        "version": "0.0.0",
                    },
                },
            },
        )
        initialize_response = _read_first_response(process)
        assert initialize_response.get("id") == _INITIALIZE_REQUEST_ID, (
            initialize_response
        )
        assert "result" in initialize_response, initialize_response

        # ----- Phase 2: trigger an in-progress refresh --------------------

        # The MCP SDK requires the ``initialized`` notification to flow
        # before tool calls. Sending it explicitly keeps the dispatcher
        # path identical to a real client session.
        _send_jsonrpc(
            process,
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )

        _send_jsonrpc(
            process,
            {
                "jsonrpc": "2.0",
                "id": _REFRESH_REQUEST_ID,
                "method": "tools/call",
                "params": {
                    "name": _TOOL_NAME_REFRESH_ALL_PROJECTS,
                    "arguments": {},
                },
            },
        )

        # Wait for the coordinator's ``begin_snapshot`` to become
        # observable in the database. The runner's slow
        # ``enumerate_projects`` blocks immediately after this point,
        # so the snapshot row will sit in ``in_progress`` until the
        # release file is created.
        in_progress_snapshot_id = _wait_for_in_progress_snapshot(
            store_path,
            time.monotonic() + _INGESTION_VISIBLE_TIMEOUT_SECONDS,
        )

        # Capture the ``current_snapshot.snapshot_id`` before SIGTERM
        # so we can prove the pointer is unchanged after shutdown
        # (Property 11). On a fresh store with no prior commit this
        # is ``None``, which is also a meaningful equality.
        current_snapshot_before = _read_current_snapshot_id(store_path)

        # While the refresh is in flight, the visualization server
        # MUST still be accepting connections — Requirement 14.2 says
        # readers see the previous snapshot during an in-progress job,
        # not that the surface goes dark. The probe is a TCP connect
        # only; we close immediately so the server's deadline
        # middleware does not bound a slow handler.
        assert not _connection_is_refused(port), (
            "Visualization_Server unexpectedly refused connections "
            "while an Ingestion_Job was in progress; Requirement 14.2 "
            "requires the read surface to remain available"
        )

        # ----- Phase 3: send SIGTERM and observe the shutdown -------------

        process.send_signal(signal.SIGTERM)
        # Closing the subprocess' stdin lets the MCP SDK's stdio
        # transport's read loop observe EOF and exit promptly. The
        # SDK's ``Server.run`` finally arm cancels every in-flight
        # request handler when the read stream closes, which is the
        # production code path that drives Requirement 12.9 step 3
        # ("MCP stdio handlers close"). Closing stdin from the test
        # side mirrors what a real MCP client would do when its peer
        # exits and keeps the test from waiting on the defensive
        # 5-second timeout in ``main._close_mcp_stdio``.
        assert process.stdin is not None
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()

        # The production shutdown sequence is documented in
        # ``main._shutdown_in_order`` and runs the steps in this order:
        #   1. Visualization_Server stops accepting new connections.
        #   2. Scheduler signalled to stop (no-op when no schedule).
        #   3. MCP stdio handlers close.
        #   4. In-flight Ingestion_Job snapshot is marked ``failed``.
        #   5. Scheduler thread joined (no-op when no schedule).
        #   6. Knowledge_Store flushed and closed.
        #
        # The test pins on step 4: the in-progress snapshot row must
        # transition to ``failed`` before the test releases the worker
        # thread. Releasing the worker BEFORE the abort would let the
        # worker reach ``commit_snapshot`` and the test would observe a
        # ``completed`` snapshot, masking a real shutdown-ordering
        # regression. Polling the database from a separate read-only
        # connection is safe under SQLite WAL mode regardless of how
        # far through the shutdown sequence the subprocess has
        # progressed.
        abort_deadline = time.monotonic() + _SHUTDOWN_ABORT_TIMEOUT_SECONDS
        while time.monotonic() < abort_deadline:
            current_status = _read_snapshot_status_or_none(
                store_path, in_progress_snapshot_id
            )
            if current_status == "failed":
                break
            if process.poll() is not None:
                # The subprocess exited before the abort step ran.
                # That can only happen if the shutdown sequence
                # finished in a state we did not anticipate; surface
                # it as a test failure with the captured streams so
                # the regression is debuggable.
                stdout_tail, stderr_tail = process.communicate(timeout=2.0)
                pytest.fail(
                    "subprocess exited before the in-flight snapshot was "
                    f"marked 'failed' (last observed status: {current_status!r}). "
                    f"stdout: {stdout_tail!r}; stderr: {stderr_tail!r}"
                )
            time.sleep(_INGESTION_POLL_INTERVAL_SECONDS)
        else:
            current_status = _read_snapshot_status_or_none(
                store_path, in_progress_snapshot_id
            )
            pytest.fail(
                f"snapshot {in_progress_snapshot_id} status is "
                f"{current_status!r} after {_SHUTDOWN_ABORT_TIMEOUT_SECONDS}s; "
                "expected the shutdown sequence to mark it 'failed' "
                "(Requirement 12.9 step 4)"
            )

        # Now that the abort step has run, release the worker thread
        # so the subprocess can finish exiting. The worker's next
        # action after the polling loop is a ``populate_in_scope``
        # call which executes ``_require_in_progress`` against the
        # now-``failed`` snapshot; that raises ``ValueError`` and the
        # worker thread terminates without ever reaching
        # ``commit_snapshot``. The order — abort first, release
        # second — is what the test pins to ensure ``current_snapshot``
        # remains unchanged across the shutdown.
        release_file.touch()

        # Drain stdout to EOF. The SDK's stdio transport closes its
        # write stream as the cancellation propagates and the
        # subprocess closes its stdout fd on exit; reaching EOF is
        # the test's evidence that MCP stdio closed (Requirement
        # 12.9 step 3) and that the subprocess actually exited.
        eof_deadline = time.monotonic() + _PROCESS_EXIT_TIMEOUT_SECONDS
        _read_remaining_stdout_to_eof(process, eof_deadline)

        # The subprocess MUST exit cleanly within the timeout. A
        # non-zero status would indicate the shutdown sequence raised
        # past the defensive ``try/except`` blocks in
        # ``_shutdown_in_order`` — i.e. a regression in Requirement
        # 12.9's "best-effort, single termination path" rule.
        try:
            return_code = process.wait(timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"subprocess did not exit within "
                f"{_PROCESS_EXIT_TIMEOUT_SECONDS}s after SIGTERM; "
                "shutdown ordering may be wedged"
            )
        assert return_code == 0, (
            f"subprocess exited with non-zero status {return_code} "
            "after SIGTERM; shutdown ordering did not complete cleanly"
        )

    # ----- Phase 4: post-shutdown invariants ------------------------------

    # Visualization_Server stopped accepting connections. After the
    # process has exited the kernel reaps the listening sockets, so a
    # fresh connect MUST be refused (Requirement 12.9 step 1).
    assert _connection_is_refused(port), (
        f"127.0.0.1:{port} still accepts connections after the "
        "subprocess exited; the Visualization_Server did not stop "
        "accepting new connections during shutdown (Requirement 12.9)"
    )

    # In-flight Ingestion_Job was aborted. The previously-observed
    # ``in_progress`` snapshot row must now be ``failed`` and
    # ``current_snapshot.snapshot_id`` must be unchanged from before
    # the refresh started — exactly the design's *Shutdown* invariant
    # (Property 11) that readers continue to see the previously
    # current snapshot after shutdown.
    final_status = _read_snapshot_status(store_path, in_progress_snapshot_id)
    assert final_status == "failed", (
        f"snapshot {in_progress_snapshot_id} status is {final_status!r}; "
        "expected 'failed' because shutdown should have aborted the "
        "in-flight Ingestion_Job (Requirement 12.9)"
    )
    current_snapshot_after = _read_current_snapshot_id(store_path)
    assert current_snapshot_after == current_snapshot_before, (
        f"current_snapshot.snapshot_id changed from "
        f"{current_snapshot_before!r} to {current_snapshot_after!r} "
        "across the shutdown; the design's Shutdown section requires "
        "the pointer to remain unchanged so the next startup serves "
        "the last successfully committed snapshot (Property 11)"
    )

    # Knowledge_Store closed cleanly. Open the database with a fresh
    # read-only connection and confirm the schema is intact and the
    # snapshot rows are queryable. A residual write lock or a
    # corrupted WAL would surface as an exception here.
    _assert_store_closes_cleanly(store_path)


# ---------------------------------------------------------------------------
# Database probes used by the post-shutdown assertions
# ---------------------------------------------------------------------------


def _read_current_snapshot_id(store_path: Path) -> int | None:
    """Read ``current_snapshot.snapshot_id`` from a fresh read-only connection.

    Returns ``None`` when the pointer is ``NULL`` (no snapshot has
    ever been committed, which is the expected state in this test).
    """

    uri = f"file:{store_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=2.0)
    try:
        row = connection.execute(
            "SELECT snapshot_id FROM current_snapshot WHERE id = 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _read_snapshot_status(store_path: Path, snapshot_id: int) -> str:
    """Return the ``status`` column for ``snapshot_id``.

    Used after shutdown to confirm the in-flight snapshot was marked
    ``failed``. ``mode=ro`` is sufficient because we never need to
    write; opening read-only also surfaces residual locks (a still-
    open write connection would prevent the read).
    """

    status = _read_snapshot_status_or_none(store_path, snapshot_id)
    assert status is not None, (
        f"snapshot_id {snapshot_id} disappeared from the snapshots table"
    )
    return status


def _read_snapshot_status_or_none(
    store_path: Path, snapshot_id: int
) -> str | None:
    """Return the ``status`` column for ``snapshot_id`` or ``None``.

    Used during the post-SIGTERM polling loop where the database may
    be momentarily unreadable (e.g. WAL checkpoint in flight) or
    where the snapshot row simply has not been written yet. ``None``
    signals the caller to keep polling rather than fail outright.
    """

    if not store_path.exists():
        return None
    uri = f"file:{store_path}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
    except sqlite3.Error:
        return None
    try:
        try:
            row = connection.execute(
                "SELECT status FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        except sqlite3.Error:
            return None
    finally:
        connection.close()
    if row is None:
        return None
    return str(row[0])


def _assert_store_closes_cleanly(store_path: Path) -> None:
    """Verify the SQLite database is reopenable and the schema is intact.

    Opens a fresh read-only connection and probes every table the
    design declares (``snapshots``, ``project_profiles``,
    ``ingestion_skips``, ``current_snapshot``). A residual lock,
    truncated WAL, or corrupted page would surface as a
    :class:`sqlite3.OperationalError` from one of these queries.
    """

    uri = f"file:{store_path}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=2.0)
    try:
        for table in ("snapshots", "project_profiles", "ingestion_skips"):
            # ``COUNT(*)`` exercises the table's pages without
            # materializing any rows, which keeps the assertion
            # cheap even on stores that grew large under prior tests.
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        row = connection.execute(
            "SELECT id, snapshot_id FROM current_snapshot WHERE id = 1"
        ).fetchone()
        assert row is not None, (
            "current_snapshot row is missing after shutdown; the store "
            "did not close cleanly"
        )
    finally:
        connection.close()

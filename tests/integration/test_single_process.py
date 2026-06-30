"""Integration test for single-process verification.

Spawns the Project Knowledge MCP Server using the documented process
entry point (``python -m project_knowledge_mcp.main``) and asserts
that the three runtime surfaces -- the MCP stdio transport, the
``Visualization_Server``, and the ``Knowledge_Store`` SQLite
connection -- all live in the **same** OS process (and therefore
share that process' memory, asyncio event loop, and file
descriptors).

Requirement 12.1 phrases this as: "WHEN the MCP_Server starts, THE
MCP_Server SHALL start the Visualization_Server in the same OS
process as the MCP_Server, so that the Visualization_Server shares
the Knowledge_Store handle and lifecycle with the MCP_Server." A
deployment that forked the visualization server into a worker
process, or that opened the SQLite database from a side process,
would still let MCP tools and ``GET /`` succeed in isolation; this
test therefore asserts process *identity* directly rather than
inferring it from per-surface health.

The assertion is the conjunction of three facts about a single
``subprocess.Popen.pid``:

1. The pid responds to an HTTP ``GET /`` on the configured
   ``visualization.port``. The Visualization_Server bound that port
   inside its ASGI lifespan, so a 200 / 503 response over loopback
   from the bound socket proves the Starlette app is running inside
   that pid.
2. The pid responds to an MCP ``initialize`` request over its own
   stdin / stdout pipes. Only the spawned process owns those pipes,
   so a well-formed JSON-RPC reply on stdout proves the MCP stdio
   transport is running inside that pid.
3. The pid holds the ``Knowledge_Store`` SQLite database file open
   on a file descriptor. The check inspects ``/proc/{pid}/fd/`` for
   a symlink whose target is the configured ``KNOWLEDGE_STORE_PATH``
   (or one of its WAL / SHM siblings), proving the SQLite connection
   was opened by that pid.

Together these three facts show that the same OS process owns all
three surfaces, which is the property Requirement 12.1 mandates.

Implements Requirement 12.1.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Skip when the host has no /proc filesystem (non-Linux). The portable
# fallback (``lsof``) is not always available in minimal sandboxes, and
# the design's deployment target is Linux, so a clean skip-with-reason
# is preferable to a flaky lsof probe.
# ---------------------------------------------------------------------------

if not Path("/proc/self/fd").exists():  # pragma: no cover - host-dependent
    pytest.skip(
        "single-process verification requires /proc/{pid}/fd to inspect "
        "open file descriptors; this host does not expose /proc",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

#: Maximum wall-clock wait, in seconds, for the spawned subprocess to
#: bring its Visualization_Server up and start accepting HTTP traffic.
#: Generous enough that a CI cold-start (interpreter startup, full
#: package import, asyncio loop bootstrap, both loopback sockets bound
#: and handed to Uvicorn) does not flake; small enough that a real bug
#: (e.g. the bind step crashing) surfaces the test quickly.
_STARTUP_TIMEOUT_SECONDS: Final[float] = 15.0

#: Polling interval used while waiting for the visualization port to
#: become reachable. Small enough that the observed start-up latency
#: is dominated by the subprocess' own bootstrap rather than by this
#: poll cadence.
_STARTUP_POLL_INTERVAL_SECONDS: Final[float] = 0.1

#: Per-attempt connect / request timeout used by every HTTP probe.
#: Loopback HTTP responses on a healthy host complete in well under a
#: millisecond; one second is far more than required.
_HTTP_PROBE_TIMEOUT_SECONDS: Final[float] = 2.0

#: Maximum wait, in seconds, for the spawned subprocess to reply to
#: the MCP ``initialize`` request on stdout. The MCP SDK sends the
#: handshake reply synchronously upon receiving the request, so this
#: only bounds the case where the stdio transport is wedged.
_MCP_HANDSHAKE_TIMEOUT_SECONDS: Final[float] = 10.0

#: Maximum wait, in seconds, for the subprocess to exit cleanly after
#: receiving SIGTERM during teardown. The documented shutdown
#: ordering (Requirement 12.9) drains the visualization server,
#: cancels the MCP serve task, and closes the Knowledge_Store within
#: a few seconds; ten seconds is a generous upper bound.
_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 10.0


# ---------------------------------------------------------------------------
# JSON-RPC handshake
# ---------------------------------------------------------------------------

#: Protocol version sent in the ``initialize`` request. Pinned to a
#: historically stable revision so this test does not drift when the
#: SDK introduces a new "latest" version. The SDK echoes back any
#: supported version it recognizes, so the test's assertion is only
#: that the field is non-empty -- not that it matches this exact
#: value.
_REQUESTED_PROTOCOL_VERSION: Final[str] = "2024-11-05"

#: JSON-RPC request id for the ``initialize`` round-trip. Single-shot
#: handshake, so 1 is sufficient. The SDK echoes the id verbatim,
#: which lets the test assert the response is the reply to *this*
#: request rather than to some unrelated message.
_INITIALIZE_REQUEST_ID: Final[int] = 1


def _build_initialize_request() -> dict[str, object]:
    """Return the JSON-RPC ``initialize`` request payload.

    The payload mirrors the shape an MCP client sends at session
    start: a ``jsonrpc: "2.0"`` envelope wrapping the ``initialize``
    method with ``protocolVersion``, ``capabilities``, and
    ``clientInfo`` parameters per the MCP specification.
    """

    return {
        "jsonrpc": "2.0",
        "id": _INITIALIZE_REQUEST_ID,
        "method": "initialize",
        "params": {
            "protocolVersion": _REQUESTED_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "project-knowledge-mcp-single-process-test",
                "version": "0.0.0",
            },
        },
    }


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Return a TCP port number that is currently free on IPv4 loopback.

    Asks the kernel for an ephemeral port via ``bind(('127.0.0.1', 0))``
    and immediately closes the socket so the spawned subprocess can
    re-bind the same port via the production helper. There is a small
    race window in which another process could grab the port before
    the subprocess re-binds it; for a single-test integration scenario
    in a sandboxed environment the trade-off is acceptable.
    """

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _http_get(port: int, path: str) -> tuple[int, bytes]:
    """Issue a loopback HTTP ``GET`` and return ``(status, body)``.

    Uses :mod:`http.client` rather than a third-party HTTP library so
    the test has no extra dependencies and the connection lifecycle
    is fully controlled (closed immediately on the test's side, no
    keep-alive sockets lingering at process exit).
    """

    conn = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=_HTTP_PROBE_TIMEOUT_SECONDS
    )
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read()
        return response.status, body
    finally:
        conn.close()


def _wait_until_visualization_ready(port: int, process: subprocess.Popen[bytes]) -> None:
    """Block until the subprocess' Visualization_Server accepts HTTP.

    Polls ``GET /`` on the configured port until any HTTP response is
    received (200 for an empty catalog per Requirement 13.1, or 503
    if the catalog read transiently raised) or the timeout elapses.
    A successful response proves the Visualization_Server's bound
    socket has been handed to Uvicorn and ``Server.serve`` is running.

    Also fails fast if the subprocess exits before the deadline, so
    the test surfaces a startup error rather than waiting the full
    timeout against a dead process.
    """

    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr_bytes = process.stderr.read() if process.stderr else b""
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            pytest.fail(
                f"server subprocess exited with status {process.returncode} "
                f"before the Visualization_Server became ready.\n"
                f"stderr:\n{stderr_text}"
            )
        try:
            status, _ = _http_get(port, "/")
        except (ConnectionRefusedError, OSError):
            time.sleep(_STARTUP_POLL_INTERVAL_SECONDS)
            continue
        # Any HTTP status code proves the server's socket is bound and
        # the ASGI app is dispatching requests inside the spawned pid.
        if 200 <= status < 600:
            return
        time.sleep(_STARTUP_POLL_INTERVAL_SECONDS)

    pytest.fail(
        f"Visualization_Server on 127.0.0.1:{port} did not start within "
        f"{_STARTUP_TIMEOUT_SECONDS} seconds"
    )


# ---------------------------------------------------------------------------
# stdout reader (single line with timeout)
# ---------------------------------------------------------------------------


def _read_stdout_line_with_timeout(
    process: subprocess.Popen[bytes],
    timeout: float,
) -> bytes:
    """Read one ``\\n``-terminated line from ``process.stdout`` with a deadline.

    ``Popen.stdout.readline`` is a blocking call; on its own it can
    hang the test if the subprocess never replies. Driving the
    blocking read from a worker thread and signalling completion via
    a queue lets the main thread enforce a deadline without resorting
    to non-blocking I/O on a pipe (which is awkward to do portably).
    """

    if process.stdout is None:
        pytest.fail("subprocess has no stdout pipe")

    result: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)

    def _reader() -> None:
        # ``readline`` on a closed pipe returns ``b""`` rather than
        # raising; the caller treats that as EOF below.
        try:
            line = process.stdout.readline()  # type: ignore[union-attr]
            result.put(("ok", line))
        except Exception as exc:
            result.put(("err", exc))

    thread = threading.Thread(target=_reader, name="mcp-stdout-reader", daemon=True)
    thread.start()
    try:
        kind, value = result.get(timeout=timeout)
    except queue.Empty:
        pytest.fail(
            f"MCP server did not produce a stdout line within {timeout} seconds"
        )
    if kind == "err":
        raise value  # type: ignore[misc]
    assert isinstance(value, bytes)
    if value == b"":
        pytest.fail("MCP server closed stdout without sending a response")
    return value


# ---------------------------------------------------------------------------
# /proc inspection: is the SQLite database file held open by ``pid``?
# ---------------------------------------------------------------------------


def _open_descriptor_targets(pid: int) -> list[str]:
    """Return the resolved targets of every entry in ``/proc/{pid}/fd``.

    File descriptors that disappear between ``iterdir`` and
    ``readlink`` are silently skipped. Permission errors are also
    swallowed: the test process and the subprocess share the same
    user, so any unreadable entry is an external race rather than a
    test failure.
    """

    fd_dir = Path(f"/proc/{pid}/fd")
    targets: list[str] = []
    for entry in fd_dir.iterdir():
        try:
            targets.append(str(entry.readlink()))
        except OSError:
            continue
    return targets


def _is_sqlite_db_held_by(pid: int, db_path: Path) -> bool:
    """Return ``True`` when ``pid`` has the ``Knowledge_Store`` DB open.

    SQLite in WAL mode opens the database file plus two siblings
    (``{db}-wal`` and ``{db}-shm``); any of the three resolving from
    a ``/proc/{pid}/fd/*`` symlink is sufficient evidence that the
    SQLite connection lives in ``pid``. The check resolves both the
    expected path and each fd target with :meth:`Path.resolve` so a
    relative ``KNOWLEDGE_STORE_PATH`` does not produce a spurious
    mismatch.
    """

    expected = str(db_path.resolve())
    expected_prefixes = (
        expected,  # the database file itself
        f"{expected}-",  # ``-wal``, ``-shm``, ``-journal`` siblings
    )
    for target in _open_descriptor_targets(pid):
        if target == expected or target.startswith(expected_prefixes[1]):
            return True
    return False


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------


def _server_environment(port: int, store_path: Path) -> dict[str, str]:
    """Return the env-var mapping passed to the spawned subprocess.

    The subprocess does not need to reach a real GitLab instance: the
    ``initialize`` handshake and the ``GET /`` response on an empty
    catalog never invoke the GitLab connector. Pointing
    ``GITLAB_BASE_URL`` at a non-routable hostname makes the
    configuration valid (an ``http://`` URL with a host) without
    risking accidental network traffic during the test.
    """

    env = os.environ.copy()
    env.update(
        {
            "GITLAB_BASE_URL": "http://gitlab.invalid",
            "GITLAB_GROUP_PATH": "test-group",
            "GITLAB_ACCESS_TOKEN": "test-token-not-used-by-handshake",
            "VISUALIZATION_PORT": str(port),
            "KNOWLEDGE_STORE_PATH": str(store_path),
            # ``REFRESH_INTERVAL`` is intentionally unset so the
            # scheduler is never started; the test asserts the
            # always-on surfaces (MCP, viz, store) live in one
            # process, which does not depend on the optional
            # scheduler.
        }
    )
    # Remove any stale value the host shell may have set so the test
    # is reproducible across environments.
    env.pop("REFRESH_INTERVAL", None)
    return env


@contextlib.contextmanager
def _spawn_server(port: int, store_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """Spawn the documented process entry point and yield the subprocess.

    On exit the subprocess is sent SIGTERM and then SIGKILL if it
    does not exit within :data:`_SHUTDOWN_TIMEOUT_SECONDS`. The MCP
    stdio pipes are kept in binary mode so the test can frame the
    JSON-RPC request line itself without juggling text encoding
    (Python's text-mode pipes apply universal newlines, which is
    undesirable for a length-sensitive line protocol).
    """

    process = subprocess.Popen(
        [sys.executable, "-m", "project_knowledge_mcp.main"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_server_environment(port, store_path),
        # Binary mode: the MCP stdio transport is a length-agnostic
        # newline-delimited JSON framing on stdout, and the test
        # writes a single JSON object terminated by ``\n`` on stdin.
        text=False,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        # Drain remaining pipes so the subprocess does not linger as a
        # zombie and the test does not leak file descriptors.
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_mcp_visualization_and_knowledge_store_share_one_os_process(
    tmp_path: Path,
) -> None:
    """All three surfaces live in the same OS process (Requirement 12.1).

    Spawns the production process entry point with a temporary
    ``Knowledge_Store`` and a free visualization port, then asserts:

    * the ``Visualization_Server`` accepts a loopback HTTP ``GET /``
      via the spawned pid's bound socket;
    * the MCP stdio transport replies to an ``initialize`` request
      on the spawned pid's stdin / stdout pipes;
    * ``/proc/{pid}/fd/`` lists a symlink resolving to the
      configured ``KNOWLEDGE_STORE_PATH`` (or one of its WAL / SHM
      siblings), proving the SQLite connection was opened by that
      same pid.

    The conjunction of these three facts about one ``Popen.pid``
    establishes that the MCP transport, the Visualization_Server,
    and the ``Knowledge_Store`` connection all live in the same OS
    process -- exactly what Requirement 12.1 mandates.

    Implements Requirement 12.1.
    """

    port = _pick_free_port()
    store_path = tmp_path / "knowledge_store.db"

    with _spawn_server(port, store_path) as process:
        pid = process.pid

        # Step 1: wait until the Visualization_Server is reachable
        # over loopback HTTP. A successful response proves the
        # bound socket is being served from inside ``pid`` (only
        # ``pid`` was given the loopback sockets at start-up).
        _wait_until_visualization_ready(port, process)

        # Step 2: probe the Visualization_Server explicitly so the
        # assertion is a first-class check rather than a side-effect
        # of the readiness wait. Requirement 13.1 specifies that
        # ``GET /`` returns 200 even when no projects are in scope;
        # 503 is also acceptable here because the
        # ``KnowledgeStoreUnavailableError`` middleware can produce
        # it when the empty store has no committed snapshot yet --
        # either way the response originates from inside ``pid``.
        viz_status, viz_body = _http_get(port, "/")
        assert viz_status in (200, 503), (
            f"Visualization_Server on 127.0.0.1:{port} returned an "
            f"unexpected status {viz_status}; body={viz_body!r}"
        )

        # Step 3: drive the MCP ``initialize`` handshake over the
        # subprocess' own stdin / stdout pipes. Only ``pid`` owns
        # those pipes, so a well-formed JSON-RPC reply on stdout
        # proves the MCP stdio transport is also running inside
        # ``pid``.
        assert process.stdin is not None
        request_line = (json.dumps(_build_initialize_request()) + "\n").encode("utf-8")
        process.stdin.write(request_line)
        process.stdin.flush()

        response_line = _read_stdout_line_with_timeout(
            process, _MCP_HANDSHAKE_TIMEOUT_SECONDS
        )
        response = json.loads(response_line.decode("utf-8"))
        assert isinstance(response, dict), response
        assert response.get("jsonrpc") == "2.0", response
        assert response.get("id") == _INITIALIZE_REQUEST_ID, response
        assert "error" not in response, response
        result = response.get("result")
        assert isinstance(result, dict), result
        server_info = result.get("serverInfo")
        assert isinstance(server_info, dict), result
        assert server_info.get("name") == "project-knowledge-mcp", server_info

        # Step 4: confirm the ``Knowledge_Store`` SQLite connection
        # was opened by the same pid. ``/proc/{pid}/fd/`` enumerates
        # every open file descriptor; a symlink resolving to the
        # configured database path (or one of its WAL / SHM
        # siblings) is the direct, unambiguous evidence that the
        # connection lives in ``pid``.
        held = _is_sqlite_db_held_by(pid, store_path)
        if not held:
            # Surface the descriptor table on failure so the
            # diagnostic includes which files the subprocess *did*
            # have open -- often the clearest signal that the
            # database was opened from a side process or never
            # opened at all.
            targets = _open_descriptor_targets(pid)
            pytest.fail(
                f"Knowledge_Store database {store_path} is not held open "
                f"by the spawned MCP server pid {pid}.\n"
                f"open fd targets: {targets!r}"
            )

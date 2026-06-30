"""Integration test for real loopback bind verification.

Starts the Visualization_Server skeleton on a free TCP port using the
production bind path (:func:`bind_or_exit` →
:func:`create_server` → :meth:`uvicorn.Server.serve` with pre-bound
sockets) and then attempts TCP connections from ``127.0.0.1``, ``::1``,
and (when available) a non-loopback interface address. The assertions
prove the server's listening socket set, when exercised against the
live operating-system network stack, accepts connections only on the
two loopback addresses -- exactly as Requirement 12.2 mandates.

This complements the property-based test in task 10.12 (Property 20),
which enumerates candidate interface addresses against the in-memory
binding helper. The integration test here verifies the real network
behavior end-to-end: a pre-bound IPv4 / IPv6 loopback socket pair
under Uvicorn's ``Server.serve`` does in fact refuse connections on
non-loopback addresses, so the bind-helper unit test results
generalize to the running server.

The non-loopback connection check is skipped gracefully when the host
exposes no non-loopback IPv4 interface (for example, isolated CI
sandboxes). The IPv4 and IPv6 loopback connection checks are always
executed because the design treats both as mandatory acceptance
addresses.

Implements Requirement 12.2.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from typing import TYPE_CHECKING

import pytest

from project_knowledge_mcp.visualization_server import (
    bind_or_exit,
    build_app,
    create_server,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

# Maximum wait, in seconds, for the background Uvicorn server to begin
# accepting connections on the IPv4 loopback socket. Five seconds is
# generous enough that CI cold-starts (interpreter import of Starlette /
# Uvicorn, asyncio loop bootstrap, pre-bound socket handover) do not
# flake; small enough that a real failure (e.g. ``serve`` crashing on
# import) surfaces the test quickly rather than freezing the suite.
_READY_TIMEOUT_SECONDS: float = 5.0

# Per-attempt connect timeout used by every TCP probe in this test. A
# successful loopback ``connect()`` returns in well under a millisecond
# on a healthy host, so one second is far more than required for the
# positive cases. For the negative case the kernel responds with
# ``ECONNREFUSED`` immediately when no listener is bound to the
# probed (interface, port) pair, so this same timeout also bounds the
# negative path against a hung accept queue.
_CONNECT_TIMEOUT_SECONDS: float = 1.0

# Maximum wait, in seconds, for the background server thread to exit
# after ``server.should_exit`` is set during teardown. Five seconds is
# enough for Uvicorn's accept loop to observe the flag and unwind all
# tasks; if it is exceeded the test still completes (the thread is a
# daemon) but the assertion that the thread joined is allowed to fail
# loudly so a stuck shutdown is caught.
_SHUTDOWN_TIMEOUT_SECONDS: float = 5.0

# Polling interval used by :func:`_wait_until_ready` while waiting for
# the server to start accepting connections. Small enough that the
# observed start-up latency is dominated by Uvicorn's own bootstrap
# rather than by the polling cadence.
_READY_POLL_INTERVAL_SECONDS: float = 0.05


# ---------------------------------------------------------------------------
# Network discovery helpers
# ---------------------------------------------------------------------------


def _pick_free_port() -> int:
    """Return a TCP port number that is currently free on IPv4 loopback.

    Asks the kernel for an ephemeral port via ``bind(('127.0.0.1', 0))``
    and immediately closes the socket so the caller can re-bind the
    same port via the production helper. There is a small race window
    in which another process could grab the port before the caller
    re-binds it; for a single-test integration scenario in a sandboxed
    environment the trade-off is acceptable, and avoiding it would
    require duplicating the bind helper's IPv4 + IPv6 logic locally.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def _discover_non_loopback_ipv4() -> str | None:
    """Return a non-loopback IPv4 address on this host, or ``None``.

    Tries two strategies in order so the test runs on as many
    environments as possible:

    1. The "UDP-connect" trick: open an ``AF_INET`` UDP socket and call
       ``connect`` with a public IPv4 destination. No packets are sent
       (UDP ``connect`` only sets the kernel's routing decision), but
       the kernel binds the socket's local address to whichever
       interface it would route the destination through. ``getsockname``
       then exposes that interface's IP. This is the most reliable
       way to obtain "the IP a remote host would see" without privileged
       access to the routing table.
    2. ``getaddrinfo(gethostname())``: looks up every address bound to
       the host's primary name. On hosts with ``/etc/hosts`` mapping
       the local hostname to ``127.0.1.1`` (Debian default) this is
       still loopback and is filtered out below; on cloud / corporate
       hosts the hostname usually resolves to the actual interface
       address.

    The first non-loopback candidate is returned. ``None`` indicates
    the host has only loopback interfaces and the negative case of
    Requirement 12.2 cannot be exercised on this environment; the
    caller should ``pytest.skip`` that part of the test gracefully.
    """
    candidates: list[str] = []

    # Strategy 1: UDP-connect trick. Use a TEST-NET-2 address
    # (RFC 5737, 198.51.100.0/24) as the destination so we never
    # actually contact a third party. The kernel still picks an
    # outbound interface for the destination even when no packet is
    # sent, which is exactly the information we need.
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        udp.connect(("198.51.100.1", 80))
        # ``getsockname`` returns a sockaddr whose shape varies across
        # address families; for the ``AF_INET`` socket above the first
        # element is the dotted-decimal host string. Narrow with
        # ``isinstance`` so the static type-check is satisfied without
        # blindly trusting the runtime shape.
        local = udp.getsockname()
        if isinstance(local[0], str):
            candidates.append(local[0])
    except OSError:
        # No default route, fully air-gapped, or otherwise no IPv4
        # routing decision possible; fall through to strategy 2.
        pass
    finally:
        udp.close()

    # Strategy 2: hostname resolution. ``gethostname`` may return a
    # name that resolves only to 127.0.0.1 on minimal sandboxes, in
    # which case we silently move on.
    with contextlib.suppress(OSError):
        for info in socket.getaddrinfo(
            socket.gethostname(),
            None,
            socket.AF_INET,
            socket.SOCK_STREAM,
        ):
            sockaddr = info[4]
            # ``AF_INET`` ``getaddrinfo`` results carry a sockaddr
            # whose first element is the host string; narrow for
            # static analysis as above.
            if isinstance(sockaddr[0], str):
                candidates.append(sockaddr[0])

    for candidate in candidates:
        # Anything in 127.0.0.0/8 is loopback per RFC 1122.
        if not candidate.startswith("127."):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Connect probe
# ---------------------------------------------------------------------------


# A minimal HTTP/1.1 ``GET /`` request with ``Connection: close``. When
# the connection is accepted by Uvicorn, sending a complete request and
# reading the response to EOF lets Uvicorn close its server-side
# transport cleanly before the test moves on. This matters because the
# event loop running ``Server.serve`` is torn down at the end of the
# test, and any transports still in flight at that point surface as
# ``ResourceWarning``s which the project's strict ``filterwarnings``
# config converts to errors. Driving each accepted connection through
# a full request / response cycle eliminates that source of noise
# without changing the property under test (which is whether the
# kernel accepted the connection in the first place).
_HTTP_PROBE_REQUEST: bytes = (
    b"GET / HTTP/1.1\r\n"
    b"Host: probe\r\n"
    b"Connection: close\r\n"
    b"\r\n"
)


def _probe_connect(
    family: int,
    address: tuple[str, int] | tuple[str, int, int, int],
) -> str:
    """TCP-connect to ``address`` and classify the outcome.

    Returns one of:

    * ``"ok"``: the TCP three-way handshake completed. After the
      handshake the probe sends a single HTTP request and reads the
      response to EOF so Uvicorn closes its server-side transport
      cleanly; the bytes of the response are otherwise ignored
      because the assertion is only about whether the connection
      was accepted.
    * ``"refused"``: the kernel returned ``ECONNREFUSED`` because no
      socket is listening on the (interface, port) pair. This is the
      expected outcome for the non-loopback interface address: the
      Visualization_Server is bound to ``127.0.0.1`` and ``::1`` only,
      so a connection to any other interface address on the same port
      reaches a port with no listener.
    * ``"timeout"``: the connect attempt did not complete within
      :data:`_CONNECT_TIMEOUT_SECONDS`. Some Linux configurations drop
      packets to non-loopback addresses silently (e.g. reverse-path
      filtering rejecting the SYN); treating timeout as "did not
      accept" keeps the negative assertion stable across hosts.
    * ``"unreachable:{errno}"``: any other ``OSError`` from connect
      (``ENETUNREACH`` / ``EHOSTUNREACH`` / ``EADDRNOTAVAIL`` / ...).
      These all imply the connection did not reach a listener, so the
      negative case treats them as "not ok" too.

    The socket is always closed before return so the test does not
    leak file descriptors regardless of which branch is taken.
    """
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(_CONNECT_TIMEOUT_SECONDS)
    try:
        sock.connect(address)
    except ConnectionRefusedError:
        sock.close()
        return "refused"
    except TimeoutError:
        # ``socket.timeout`` is an alias for ``TimeoutError`` on
        # Python 3.10+; catching the canonical name covers both.
        sock.close()
        return "timeout"
    except OSError as exc:
        sock.close()
        return f"unreachable:{exc.errno}"

    # Connection accepted. Drive the request / response cycle to EOF
    # so Uvicorn fully closes its server-side transport before the
    # event loop is torn down. Errors during the exchange are
    # swallowed because the only property under test is whether the
    # connection was accepted in the first place; a half-open
    # exchange does not change that answer.
    try:
        sock.sendall(_HTTP_PROBE_REQUEST)
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
    except OSError:
        pass
    finally:
        sock.close()
    return "ok"


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _wait_until_ready(port: int) -> None:
    """Block until the IPv4 loopback socket accepts connections.

    Polls ``connect`` on ``127.0.0.1:port`` until a successful
    handshake is observed or the timeout elapses. Raises
    :class:`RuntimeError` on timeout so the test fails loudly rather
    than proceeding against a server that never finished starting.
    """
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _probe_connect(socket.AF_INET, ("127.0.0.1", port)) == "ok":
            return
        time.sleep(_READY_POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"Visualization_Server did not become ready on 127.0.0.1:{port} "
        f"within {_READY_TIMEOUT_SECONDS} seconds"
    )


@contextlib.contextmanager
def _running_server(port: int) -> Iterator[None]:
    """Run the Visualization_Server skeleton in a background thread.

    Calls the production startup sequence -- :func:`bind_or_exit`
    pre-binds both loopback sockets (Requirement 12.2);
    :func:`build_app` constructs the empty Starlette application;
    :func:`create_server` builds the Uvicorn :class:`uvicorn.Server`
    -- and then drives :meth:`Server.serve` in a daemon thread via
    :func:`asyncio.run`.

    On exit the context manager sets ``server.should_exit`` so
    Uvicorn's accept loop unwinds and the thread joins within
    :data:`_SHUTDOWN_TIMEOUT_SECONDS`. The pre-bound sockets are
    closed defensively after the join in case Uvicorn ever fails to
    own their lifecycle on a future version.
    """
    sockets = bind_or_exit(port)
    app = build_app()
    server = create_server(app, port)

    def _serve() -> None:
        # Each thread needs its own asyncio event loop; ``asyncio.run``
        # provisions one and tears it down cleanly when ``serve``
        # returns. Uvicorn's ``Server.serve`` is the documented entry
        # point for using pre-bound sockets, so we drive it directly.
        asyncio.run(server.serve(sockets=sockets))

    thread = threading.Thread(target=_serve, name="viz-server", daemon=True)
    thread.start()
    try:
        _wait_until_ready(port)
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        # Defensive cleanup: ``Server.serve`` closes the listening
        # sockets on its own when it exits, but if the thread did not
        # finish in time we close them here so the next test run is
        # not blocked by a lingering listener on the same port.
        for s in sockets:
            with contextlib.suppress(OSError):
                s.close()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_visualization_server_accepts_only_loopback_connections() -> None:
    """Only ``127.0.0.1`` and ``::1`` accept TCP connections to the bound port.

    Starts the production bind path on a free TCP port, then probes
    the running server with three TCP connect attempts:

    * ``127.0.0.1`` (IPv4 loopback): MUST be accepted.
    * ``::1`` (IPv6 loopback): MUST be accepted.
    * A non-loopback interface address (when available): MUST NOT be
      accepted -- the kernel must respond with ``ECONNREFUSED``,
      ``ENETUNREACH`` / ``EHOSTUNREACH`` / ``EADDRNOTAVAIL``, or a
      connect timeout. Each of these outcomes proves no socket is
      listening on the non-loopback interface for the bound port,
      which is exactly the Requirement 12.2 negative-direction
      property.

    The third probe is skipped (with a clear reason) when the host
    has no non-loopback IPv4 interface address available, which is
    the case for some minimal sandboxes. The two loopback probes are
    always executed because both addresses are mandatory acceptance
    addresses per the requirement.

    Implements Requirement 12.2.
    """

    port = _pick_free_port()

    with _running_server(port):
        # IPv4 loopback: positive direction of Requirement 12.2.
        ipv4_outcome = _probe_connect(socket.AF_INET, ("127.0.0.1", port))
        assert ipv4_outcome == "ok", (
            f"127.0.0.1:{port} should accept connections "
            f"(IPv4 loopback bind, Requirement 12.2); got {ipv4_outcome!r}"
        )

        # IPv6 loopback: positive direction of Requirement 12.2. The
        # 4-tuple form ``(host, port, flowinfo, scopeid)`` is the
        # canonical IPv6 sockaddr accepted by ``connect``; ``flowinfo``
        # and ``scopeid`` are zero for global-scope traffic to ``::1``.
        ipv6_outcome = _probe_connect(socket.AF_INET6, ("::1", port, 0, 0))
        assert ipv6_outcome == "ok", (
            f"[::1]:{port} should accept connections "
            f"(IPv6 loopback bind, Requirement 12.2); got {ipv6_outcome!r}"
        )

        # Non-loopback: negative direction of Requirement 12.2.
        non_loopback_ip = _discover_non_loopback_ipv4()
        if non_loopback_ip is None:
            pytest.skip(
                "no non-loopback IPv4 interface available on this host; "
                "skipping the negative-direction check for Requirement 12.2"
            )
        non_loopback_outcome = _probe_connect(
            socket.AF_INET, (non_loopback_ip, port)
        )
        assert non_loopback_outcome != "ok", (
            f"{non_loopback_ip}:{port} should NOT accept connections "
            f"(server is bound to loopback only, Requirement 12.2); "
            f"got {non_loopback_outcome!r}"
        )

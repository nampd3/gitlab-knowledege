# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 20: For all network interface addresses, the Visualization_Server SHALL accept HTTP connections on 127.0.0.1 and ::1 and SHALL NOT accept HTTP connections on any other address.
"""Property test for loopback-only binding.

**Validates Requirement 12.2** (Property 20 in the design).

For every candidate network-interface address ``addr``, the
Visualization_Server SHALL accept TCP connections on its configured
port if and only if ``addr`` is the IPv4 loopback (``127.0.0.1``) or
the IPv6 loopback (``::1``). Any other address -- other 127/8 IPv4
variants, the host's discovered non-loopback IPv4 address, RFC 5737
IPv4 documentation addresses, IPv4 link-local, RFC 3849 IPv6
documentation addresses, IPv6 link-local, or IPv6 site-local --
SHALL NOT accept a connection on the bound port.

The test invokes
:func:`project_knowledge_mcp.visualization_server._bind_loopback_sockets`
on a kernel-assigned free TCP port and then drives a
Hypothesis-enumerated candidate address through ``socket.connect``.
``_bind_loopback_sockets`` returns sockets in the LISTEN state, so
the kernel completes the TCP handshake on the loopback addresses
without any userspace ``accept`` call. The test therefore observes
the binding decision directly at the TCP layer (which is the layer
the property constrains) and does not need to spin up an HTTP
application or HTTP client.

The candidate pool is drawn from a *controlled* set of addresses
(per the task brief) so the test is hermetic enough for CI:

* **Loopback variants** -- ``127.0.0.1`` (IPv4) and ``::1`` (IPv6):
  the only addresses on which the server is permitted to accept.
* **Other 127/8 IPv4 addresses** -- e.g. ``127.0.0.2``, ``127.0.1.1``:
  Linux routes the entire 127/8 range through the loopback interface,
  so these addresses *are* reachable from this process; however,
  ``_bind_loopback_sockets`` binds specifically to ``127.0.0.1``,
  so the kernel returns ``ECONNREFUSED`` for any other 127/8
  address. This is the strongest "rejection" signal in the pool
  because the connection actually reaches the binding layer.
* **The host's discovered non-loopback IPv4 address** -- discovered
  at runtime via the standard "outbound UDP getsockname" trick.
  Same role as the other-127/8 case at the external-interface
  boundary: the kernel knows the address is local but no socket is
  bound there. Skipped if discovery fails (e.g. a sandboxed CI
  runner with only the loopback interface).
* **RFC 5737 IPv4 documentation address** -- ``192.0.2.1`` (TEST-NET-1):
  guaranteed never to be routed on the public Internet; on a
  typical CI host the connection fails with ``EHOSTUNREACH`` /
  ``ENETUNREACH`` or times out. Either outcome counts as
  "rejected" for the property.
* **IPv4 link-local** -- ``169.254.1.1``: same role as RFC 5737.
* **RFC 3849 IPv6 documentation prefix** -- ``2001:db8::1``: same
  role for IPv6.
* **IPv6 link-local / site-local** -- ``fe80::1``, ``fc00::1``:
  same role for IPv6.

A small ``socket.settimeout`` is used on each connection attempt so
that addresses with no route fail in bounded time on CI rather than
blocking the test indefinitely.
"""

from __future__ import annotations

import contextlib
import socket
from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.visualization_server import (
    LOOPBACK_IPV4,
    LOOPBACK_IPV6,
    _bind_loopback_sockets,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Connection-attempt knobs
# ---------------------------------------------------------------------------

#: Per-attempt timeout for ``socket.connect``. Loopback connections
#: complete the TCP handshake in well under a millisecond, so this
#: bound only ever fires for non-loopback addresses with no route
#: (RFC 5737, link-local). Choosing 0.5s keeps any single failure
#: bounded and the whole 100-example run well under a typical
#: pytest deadline.
_CONNECT_TIMEOUT_SECONDS: Final[float] = 0.5


# ---------------------------------------------------------------------------
# Candidate address pool
# ---------------------------------------------------------------------------

# The two loopback addresses Property 20 requires the server to accept
# (Requirement 12.2). Encoded as ``(family, address)`` tuples so the
# strategy can generate IPv4 and IPv6 candidates uniformly.
_ACCEPTED_CANDIDATES: Final[tuple[tuple[int, str], ...]] = (
    (socket.AF_INET, LOOPBACK_IPV4),
    (socket.AF_INET6, LOOPBACK_IPV6),
)

# IPv4 addresses that must NOT accept a connection on the bound port.
# These are the "strong rejection" signals (kernel routes them to the
# loopback interface but no socket is bound there) plus a representative
# unrouteable address from each of RFC 5737 and IPv4 link-local.
_REJECTED_IPV4_STATIC: Final[tuple[str, ...]] = (
    # Other 127/8 addresses: routed to loopback by Linux but no socket
    # is bound at this address+port.
    "127.0.0.2",
    "127.0.1.1",
    "127.42.42.42",
    "127.255.255.254",
    # RFC 5737 TEST-NET-1: documentation range, must not be routed.
    "192.0.2.1",
    # IPv4 link-local (RFC 3927).
    "169.254.1.1",
)

# IPv6 addresses that must NOT accept a connection on the bound port.
_REJECTED_IPV6_STATIC: Final[tuple[str, ...]] = (
    # RFC 3849 documentation prefix.
    "2001:db8::1",
    # Link-local (RFC 4291): connect typically fails with EINVAL on
    # Linux because no scope id is supplied.
    "fe80::1",
    # Unique-local (RFC 4193).
    "fc00::1",
)


def _discover_host_non_loopback_ipv4() -> str | None:
    """Return one of the host's non-loopback IPv4 addresses, or None.

    Uses the standard "outbound UDP getsockname" trick: a UDP socket
    is ``connect``ed to a public address (no packet is actually sent
    by ``connect`` on a UDP socket) and the local address the kernel
    would route through is read back via ``getsockname``. The chosen
    public destination (``198.51.100.1`` from RFC 5737 TEST-NET-2) is
    documented to be unreachable, which is fine: the kernel still
    picks an outbound interface address based on its routing table
    without sending anything.

    Returns ``None`` if the host has no IPv4 default route or only
    the loopback interface (typical of some sandboxed CI runners),
    in which case the property is exercised against the static
    rejected pool only.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("198.51.100.1", 9))
            local = probe.getsockname()[0]
    except OSError:
        return None
    # Defensive: if getsockname returns a 127/8 address (e.g. the
    # runner has no default route and the kernel falls back to
    # loopback), skip — it's already covered by ``_REJECTED_IPV4_STATIC``
    # and would otherwise mistakenly land in the *accepted* set.
    if local.startswith("127."):
        return None
    return local


def _build_rejected_candidates() -> tuple[tuple[int, str], ...]:
    """Assemble the full rejected candidate pool.

    Combines the two static IPv4 / IPv6 lists with the runtime-
    discovered host non-loopback IPv4 address (when available). The
    resulting tuple is sampled from by Hypothesis below.
    """
    rejected: list[tuple[int, str]] = [
        (socket.AF_INET, addr) for addr in _REJECTED_IPV4_STATIC
    ]
    rejected.extend((socket.AF_INET6, addr) for addr in _REJECTED_IPV6_STATIC)
    discovered = _discover_host_non_loopback_ipv4()
    if discovered is not None:
        rejected.append((socket.AF_INET, discovered))
    return tuple(rejected)


_REJECTED_CANDIDATES: Final[tuple[tuple[int, str], ...]] = _build_rejected_candidates()

# Full candidate pool that Hypothesis samples from. Both arms of the
# property's "iff" are present so a single test invocation exercises
# both directions: loopback addresses must accept, non-loopback
# addresses must not.
_ALL_CANDIDATES: Final[tuple[tuple[int, str], ...]] = (
    _ACCEPTED_CANDIDATES + _REJECTED_CANDIDATES
)


# ---------------------------------------------------------------------------
# Bound-server fixture
# ---------------------------------------------------------------------------


def _find_free_loopback_port() -> int:
    """Return a TCP port that is currently free on ``127.0.0.1``.

    Found by binding a throwaway IPv4 loopback socket to port ``0``
    (kernel-assigned), reading the assigned port via ``getsockname``,
    and closing the socket. The TOCTOU window between this function
    returning and ``_bind_loopback_sockets`` re-binding the port is
    negligible for a single-process test.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK_IPV4, 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def bound_loopback_port() -> Iterator[int]:
    """Bind both loopback sockets to a free port for the test session.

    Module-scoped so the bind cost is paid exactly once across the
    100-example Hypothesis run. Yields the bound port; teardown
    closes both sockets so the port is released for follow-up tests.
    """
    port = _find_free_loopback_port()
    sockets = _bind_loopback_sockets(port)
    try:
        yield port
    finally:
        for s in sockets:
            with contextlib.suppress(OSError):
                s.close()


# ---------------------------------------------------------------------------
# Connection probe
# ---------------------------------------------------------------------------


def _connection_succeeds(family: int, address: str, port: int) -> bool:
    """Return True iff a TCP handshake to ``(address, port)`` completes.

    A bounded ``socket.settimeout`` is applied so unreachable
    addresses (RFC 5737, link-local, etc.) fail in bounded time
    rather than blocking. Any :class:`OSError` -- ``ECONNREFUSED``,
    ``EHOSTUNREACH``, ``ENETUNREACH``, ``EINVAL``, ``timed out``,
    etc. -- is treated as "the server did not accept the connection",
    which is exactly the rejection half of the property's "iff".
    """
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(_CONNECT_TIMEOUT_SECONDS)
    try:
        try:
            sock.connect((address, port))
        except OSError:
            return False
        return True
    finally:
        with contextlib.suppress(OSError):
            sock.close()


# ---------------------------------------------------------------------------
# Property 20
# ---------------------------------------------------------------------------


@given(candidate=st.sampled_from(_ALL_CANDIDATES))
@settings(max_examples=100)
def test_loopback_only_binding(
    bound_loopback_port: int,
    candidate: tuple[int, str],
) -> None:
    """Property 20: TCP handshake succeeds iff address is loopback.

    For every candidate ``(family, address)`` drawn from the
    controlled pool, ``socket.connect((address, port))`` to the port
    bound by ``_bind_loopback_sockets`` succeeds if and only if
    ``address`` is one of the two loopback addresses. The property is
    expressed as a single biconditional so a regression in *either*
    direction (a loopback address being rejected, or a non-loopback
    address being accepted) is caught by the same test.

    **Validates Requirement 12.2.**
    """
    family, address = candidate
    is_loopback_address = candidate in _ACCEPTED_CANDIDATES

    accepted = _connection_succeeds(family, address, bound_loopback_port)

    if is_loopback_address:
        assert accepted, (
            f"Visualization_Server failed to accept a connection on the "
            f"loopback address {address!r} (family {family}); Requirement "
            f"12.2 requires both 127.0.0.1 and ::1 to accept on the bound port."
        )
    else:
        assert not accepted, (
            f"Visualization_Server unexpectedly accepted a connection on "
            f"the non-loopback address {address!r} (family {family}); "
            f"Requirement 12.2 forbids accepting on any address other "
            f"than 127.0.0.1 and ::1."
        )

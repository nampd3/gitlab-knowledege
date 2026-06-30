"""Unit test for non-port-in-use bind failures.

When :func:`project_knowledge_mcp.visualization_server.bind_or_exit`
fails to bind a loopback socket for any reason other than
``EADDRINUSE`` (for example ``EACCES`` or ``EADDRNOTAVAIL``), it SHALL
write exactly the documented start-up error line to stderr and SHALL
terminate the process with a non-zero exit status (Requirement 12.8).
The wording is fixed by :data:`GENERIC_BIND_FAILURE_TEMPLATE` and is
reproduced verbatim by this test so a regression in the line is caught
at unit-test time.

Sibling test :mod:`test_visualization_server_port_in_use` pins the
companion ``EADDRINUSE`` line (Requirement 12.6); the two tests
together cover both branches of the ``OSError`` triage in
:func:`project_knowledge_mcp.visualization_server._format_startup_error`.

The test patches the :mod:`socket` seam used by
:func:`_bind_loopback_sockets` with a stand-in whose ``bind`` raises a
non-``EADDRINUSE`` :class:`OSError`. This drives :func:`bind_or_exit`
down the Requirement 12.8 branch deterministically and without
depending on host configuration that would actually return e.g.
``EACCES`` or ``EADDRNOTAVAIL`` for a loopback bind.

Implements Requirement 12.8.
"""

from __future__ import annotations

import errno
import io
from typing import Any
from unittest.mock import patch

import pytest

from project_knowledge_mcp.visualization_server import (
    GENERIC_BIND_FAILURE_TEMPLATE,
    bind_or_exit,
)

pytestmark = pytest.mark.unit


class _BindFailingSocket:
    """Minimal stand-in for ``socket.socket`` whose ``bind`` raises.

    Implements only the subset of the socket API that
    :func:`_bind_loopback_sockets` exercises before the bind call: the
    constructor accepts the address-family / socket-type arguments and
    discards them, ``setsockopt`` is a no-op, ``bind`` raises the
    configured :class:`OSError`, and ``close`` is a no-op so the
    cleanup path in the real function (``s.close()`` for the
    partially-initialised IPv4 socket) does not itself fail.
    """

    def __init__(self, exc: OSError) -> None:
        self._exc = exc

    def setsockopt(self, *args: Any, **kwargs: Any) -> None:
        return None

    def bind(self, address: Any) -> None:
        raise self._exc

    def listen(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    ("oserror_errno", "oserror_message"),
    [
        # ``EACCES`` is the canonical failure when a non-root process
        # tries to bind a privileged port; explicitly listed by the
        # task brief as a representative non-EADDRINUSE failure.
        (errno.EACCES, "Permission denied"),
        # ``EADDRNOTAVAIL`` is the canonical failure when the address
        # is not assignable on this host; also explicitly listed by
        # the task brief (``OSError(99, "Cannot assign requested address")``).
        (errno.EADDRNOTAVAIL, "Cannot assign requested address"),
    ],
)
def test_bind_or_exit_emits_documented_line_for_non_eaddrinuse_oserror(
    oserror_errno: int,
    oserror_message: str,
) -> None:
    """``bind_or_exit`` writes the documented line and exits non-zero.

    For any :class:`OSError` raised by the underlying ``bind`` whose
    ``errno`` is not :data:`errno.EADDRINUSE`,
    :func:`bind_or_exit` must:

    1. emit exactly ``"startup error: visualization server failed to
       start: {os_error}"`` to the supplied ``stderr`` stream, and
    2. terminate the process with a non-zero status via
       :func:`sys.exit`.

    The OS error string interpolated into the message is the
    ``str()`` of the underlying :class:`OSError` (e.g.
    ``"[Errno 13] Permission denied"``), so the operator sees the
    failure reason reported by the operating system.

    Implements Requirement 12.8.
    """
    injected = OSError(oserror_errno, oserror_message)
    # Sanity-check the test set-up: this OSError must not be
    # ``EADDRINUSE``, otherwise we'd be re-testing Requirement 12.6.
    assert injected.errno != errno.EADDRINUSE

    port = 12345
    captured = io.StringIO()

    def _factory(*_args: Any, **_kwargs: Any) -> _BindFailingSocket:
        return _BindFailingSocket(injected)

    with (
        patch(
            "project_knowledge_mcp.visualization_server.socket.socket",
            new=_factory,
        ),
        pytest.raises(SystemExit) as exit_info,
    ):
        bind_or_exit(port=port, stderr=captured)

    # Non-zero termination status, as required by the design's
    # *Startup errors* contract: the process must not appear to have
    # started successfully when the bind fails.
    assert exit_info.value.code is not None
    assert exit_info.value.code != 0

    expected_line = GENERIC_BIND_FAILURE_TEMPLATE.format(os_error=injected)
    # The documented wording is fixed verbatim by the design's
    # *Startup errors* table; reproduce it literally so a regression
    # in :data:`GENERIC_BIND_FAILURE_TEMPLATE` is caught here too.
    assert expected_line == (
        f"startup error: visualization server failed to start: {injected}"
    )

    # ``print(..., flush=True)`` appends exactly one trailing newline;
    # the captured stderr must therefore equal the documented line plus
    # that single newline, with no additional output.
    assert captured.getvalue() == expected_line + "\n"

"""Unit test for the port-already-in-use start-up error.

When the configured ``visualization.port`` is already in use,
:func:`project_knowledge_mcp.visualization_server.bind_or_exit` SHALL
write exactly the documented start-up error line to stderr and SHALL
terminate the process with a non-zero exit status (Requirement 12.6).
The wording is fixed by the design's *Startup errors* table and is
reproduced verbatim by :data:`PORT_IN_USE_TEMPLATE`; this test pins the
wording so a regression in the line is caught at unit-test time.

The test occupies a real loopback TCP port with a listening socket and
then calls :func:`bind_or_exit` with the same port. The implementation
binds the IPv4 loopback socket first, so holding a listening IPv4
socket on the chosen port is sufficient to drive
:func:`bind_or_exit` down the ``EADDRINUSE`` branch deterministically.

Implements Requirement 12.6.
"""

from __future__ import annotations

import io
import socket

import pytest

from project_knowledge_mcp.visualization_server import (
    PORT_IN_USE_TEMPLATE,
    bind_or_exit,
)

pytestmark = pytest.mark.unit


def _occupy_loopback_port() -> tuple[socket.socket, int]:
    """Bind and listen on a free IPv4 loopback port; return (socket, port).

    Asks the kernel for any free port via ``bind(('127.0.0.1', 0))`` and
    leaves the socket in the LISTEN state so that any subsequent attempt
    to bind ``127.0.0.1:<port>`` fails with ``EADDRINUSE`` regardless of
    whether the new socket sets ``SO_REUSEADDR``. (On Linux,
    ``SO_REUSEADDR`` only permits rebinding while the previous socket
    is in ``TIME_WAIT`` and never while it is actively listening.)
    """
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    return blocker, port


def test_bind_or_exit_emits_documented_line_when_port_in_use() -> None:
    """``bind_or_exit`` writes the documented line and exits non-zero.

    The captured stderr contains exactly the documented Requirement 12.6
    line, and :func:`bind_or_exit` terminates the process with a
    non-zero status code via :func:`sys.exit`.

    Implements Requirement 12.6.
    """
    blocker, port = _occupy_loopback_port()
    captured = io.StringIO()
    try:
        with pytest.raises(SystemExit) as exit_info:
            bind_or_exit(port=port, stderr=captured)
    finally:
        blocker.close()

    # Non-zero termination status as required by the design's *Startup
    # errors* contract: the process must not appear to have started
    # successfully.
    assert exit_info.value.code != 0

    expected_line = PORT_IN_USE_TEMPLATE.format(port=port)
    # The documented wording is fixed verbatim by the design.
    assert expected_line == f"startup error: visualization.port {port} is already in use"

    # ``print(..., flush=True)`` appends exactly one trailing newline;
    # the captured stderr must therefore equal the documented line plus
    # that single newline, with no additional output.
    assert captured.getvalue() == expected_line + "\n"

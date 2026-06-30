"""Integration test for the stdio MCP handshake.

Spawns the Project Knowledge MCP Server as a subprocess, sends an
``initialize`` request over stdin, parses the response from stdout, and
asserts the handshake completes -- proving the server speaks the Model
Context Protocol over standard input and standard output as Requirement
11.1 mandates.

The production process entry point (:mod:`project_knowledge_mcp.main`,
task 11.1) is not yet wired, and :mod:`project_knowledge_mcp.mcp_server`
does not have a ``__main__`` block of its own. This test therefore
spawns a tiny adjacent runner script (:mod:`_mcp_stdio_runner`) that
does the minimum required to drive the real :meth:`MCPServer.serve`
over the inheriting process' stdin/stdout: it constructs the server
with stub collaborators (the ``initialize`` handshake never invokes
them) and runs the asyncio loop. The wire surface under test is the
real stdio transport bound by ``MCPServer.serve``.

Implements Requirement 11.1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Absolute path to the subprocess runner. Resolving from ``__file__``
# keeps the test independent of the current working directory; pytest
# may be invoked from anywhere in the repository.
_RUNNER_SCRIPT: Path = (Path(__file__).parent / "_mcp_stdio_runner.py").resolve()

# Wall-clock bound for the full handshake: process spawn, single
# request/response round-trip over the stdio pipes, and clean
# shutdown. Generous enough that CI cold-starts (interpreter startup,
# import of the package, asyncio loop bootstrap) do not flake; small
# enough that a real bug (e.g. the server hanging on the read stream)
# surfaces quickly rather than freezing the suite.
_HANDSHAKE_TIMEOUT_SECONDS: float = 10.0

# Protocol version sent by this test in the ``initialize`` request.
# The MCP SDK echoes this value back in the response when it appears
# in ``SUPPORTED_PROTOCOL_VERSIONS``; ``"2024-11-05"`` is the original
# stable revision and is supported by every MCP SDK release the
# project depends on. Hard-coding it (rather than importing
# ``LATEST_PROTOCOL_VERSION`` from the SDK) keeps the test stable
# across minor SDK bumps that introduce a new latest version while
# continuing to support the historical one.
_REQUESTED_PROTOCOL_VERSION: str = "2024-11-05"

# Expected ``serverInfo.name`` in the handshake response. Pinned by
# Requirement 11.2 and the design's ``SERVER_NAME`` constant; if the
# advertised name ever drifts, this test should fail rather than
# silently track the change.
_EXPECTED_SERVER_NAME: str = "project-knowledge-mcp"

# JSON-RPC request id used by this test. Single-shot handshake, so
# ``1`` is sufficient; the server SDK echoes the id back verbatim,
# which lets the test assert the response is the reply to *this*
# request rather than to some unrelated message that might appear on
# stdout in the future.
_INITIALIZE_REQUEST_ID: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_initialize_request() -> dict[str, object]:
    """Return the JSON-RPC ``initialize`` request payload sent to the server.

    The request mirrors the shape an MCP client sends at session
    start: a ``jsonrpc: "2.0"`` envelope wrapping the ``initialize``
    method with ``protocolVersion``, ``capabilities``, and
    ``clientInfo`` parameters per the MCP specification. The
    capabilities object is empty because this test exercises only the
    handshake response, not subsequent feature negotiation.
    """

    return {
        "jsonrpc": "2.0",
        "id": _INITIALIZE_REQUEST_ID,
        "method": "initialize",
        "params": {
            "protocolVersion": _REQUESTED_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "project-knowledge-mcp-stdio-handshake-test",
                "version": "0.0.0",
            },
        },
    }


def _run_handshake() -> tuple[str, str, int]:
    """Spawn the runner, exchange the handshake, and return the captured streams.

    Sends the ``initialize`` request line on the child's stdin, then
    closes stdin so the SDK's stdio transport observes EOF and shuts
    the server down cleanly after answering the request. ``communicate``
    drains stdout and stderr to EOF before returning, so the captured
    text is the complete output the server emitted during the
    handshake -- exactly what the assertions need to inspect.

    Returns:
        ``(stdout, stderr, returncode)``. ``stdout`` contains the
        server's line-delimited JSON-RPC response (one JSON object per
        line, terminated by ``\\n``); ``stderr`` is captured for
        inclusion in the failure message when the handshake does not
        produce a usable response.

    Raises:
        pytest.fail.Exception: when the subprocess does not complete
            the handshake within ``_HANDSHAKE_TIMEOUT_SECONDS``.
    """

    request_line = json.dumps(_build_initialize_request()) + "\n"

    with subprocess.Popen(
        [sys.executable, str(_RUNNER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ) as process:
        try:
            stdout, stderr = process.communicate(
                input=request_line,
                timeout=_HANDSHAKE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(
                "MCP server did not complete the stdio handshake within "
                f"{_HANDSHAKE_TIMEOUT_SECONDS}s.\n"
                f"stdout (so far): {stdout!r}\n"
                f"stderr (so far): {stderr!r}"
            )

    return stdout, stderr, process.returncode


def _parse_first_response(stdout: str, stderr: str) -> dict[str, object]:
    """Parse the first JSON-RPC response line emitted on stdout.

    The MCP stdio transport frames each message as a single JSON
    object terminated by ``\\n`` (no Content-Length headers). The
    initialize handshake produces exactly one server-to-client
    message, so the first non-empty line on stdout is the response
    we want to validate.

    ``stderr`` is included in the failure message so that when the
    server crashes during startup the test surfaces the underlying
    traceback rather than a bare "no output" error.
    """

    first_line = ""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    assert first_line, (
        "Server did not send any data to stdout in response to the "
        "initialize request; the stdio transport is not wired.\n"
        f"stderr: {stderr!r}"
    )

    parsed = json.loads(first_line)
    assert isinstance(parsed, dict), (
        f"Expected a JSON object on stdout, got {type(parsed).__name__}: "
        f"{first_line!r}"
    )
    return parsed


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_stdio_initialize_handshake_completes() -> None:
    """The server completes the MCP ``initialize`` handshake over stdio.

    Spawns the server as a subprocess, sends a single ``initialize``
    JSON-RPC request on stdin, and asserts the response on stdout is:

    * a well-formed JSON-RPC reply (``jsonrpc == "2.0"``, the same
      ``id`` echoed back, no ``error`` field);
    * a successful ``initialize`` result that carries the configured
      ``serverInfo.name`` and a non-empty ``serverInfo.version``;
    * advertises ``capabilities.tools`` so the client knows
      ``tools/list`` and ``tools/call`` are available;
    * echoes a non-empty ``protocolVersion`` string back to the
      client.

    Together these prove the server implements the MCP server role
    over the standard input and standard output transport per
    Requirement 11.1 -- if any assertion fails, either the transport
    is not wired (no response at all) or the response payload does
    not match the contract Requirement 11.2 sets up.

    Implements Requirement 11.1.
    """

    stdout, stderr, returncode = _run_handshake()

    response = _parse_first_response(stdout, stderr)

    # JSON-RPC envelope: this is the reply to *our* initialize request.
    assert response.get("jsonrpc") == "2.0", response
    assert response.get("id") == _INITIALIZE_REQUEST_ID, response

    # The handshake completed successfully -- no JSON-RPC error.
    assert "error" not in response, (
        f"Server returned a JSON-RPC error in response to initialize: "
        f"{response['error']!r}\nstderr: {stderr!r}"
    )

    result = response.get("result")
    assert isinstance(result, dict), (
        f"Expected a result object in the initialize response, got "
        f"{result!r}"
    )

    # Per Requirement 11.2, serverInfo carries name and a non-empty version.
    server_info = result.get("serverInfo")
    assert isinstance(server_info, dict), server_info
    assert server_info.get("name") == _EXPECTED_SERVER_NAME, server_info
    version = server_info.get("version")
    assert isinstance(version, str) and version != "", server_info

    # capabilities.tools advertises the tool surface to the client.
    capabilities = result.get("capabilities")
    assert isinstance(capabilities, dict), capabilities
    assert "tools" in capabilities, capabilities

    # The server echoes a non-empty protocol version. The SDK
    # negotiates this value: when we request a supported version it
    # is echoed back verbatim; otherwise the SDK substitutes its
    # latest. Either way the field MUST be a non-empty string.
    protocol_version = result.get("protocolVersion")
    assert isinstance(protocol_version, str) and protocol_version != "", result

    # The subprocess shut down cleanly after the handshake. ``stdin``
    # was closed by ``communicate``; the SDK's stdio transport sees
    # EOF, ends the read loop, and ``MCPServer.serve`` returns,
    # which lets ``asyncio.run`` exit with a zero status.
    assert returncode == 0, (
        f"Server exited with non-zero status {returncode} after the "
        f"handshake.\nstderr: {stderr!r}"
    )

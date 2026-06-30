"""Unit test for the MCP ``initialize`` response.

This test pins down the surface contract of the stdio MCP server skeleton
implemented in task 9.1: when an MCP client opens the connection and sends
``initialize``, the server's response payload SHALL carry the configured
``name`` and ``version`` and SHALL advertise ``capabilities.tools`` so the
client knows ``tools/list`` is available.

The Python MCP SDK does not expose a direct "build the initialize result"
helper. Instead it builds an :class:`mcp.server.models.InitializationOptions`
record from the underlying :class:`mcp.server.lowlevel.Server` and the
``initialize`` response is constructed from those options at request time.
We therefore exercise the same code path the wire response uses
(:meth:`Server.create_initialization_options`), which guarantees the
on-the-wire ``initialize`` response will carry the same ``name``,
``version``, and ``capabilities.tools`` that we assert here.

The test does not call any of the injected collaborators; the
``initialize`` response is determined entirely by the SDK ``Server``'s
configured name/version and by the request handlers registered at
``MCPServer`` construction time. We therefore pass minimal stand-in
collaborators that satisfy the typing protocol but are never invoked.

Implements Requirement 11.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types
import pytest

from project_knowledge_mcp.mcp_server import SERVER_NAME, MCPServer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_knowledge_mcp.conflict_detector import ConflictPair
    from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator
    from project_knowledge_mcp.knowledge_store import KnowledgeStore
    from project_knowledge_mcp.models import ConflictResult, ProjectProfile
    from project_knowledge_mcp.project_catalog import ProjectCatalog

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

# The ``initialize`` response is shaped by the SDK ``Server``'s ``name``,
# ``version``, and the set of registered request handlers. None of the
# injected collaborators participate, so we pass plain ``object()``
# instances cast to the appropriate type. ``cast`` keeps the static
# typing strict without forcing a real implementation.


def _unused_classify_pair(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
) -> ConflictResult:
    raise AssertionError(
        "classify_pair must not be called while building the initialize response"
    )


def _unused_find_all_conflicts(
    profiles: Sequence[ProjectProfile],
) -> list[ConflictPair]:
    raise AssertionError(
        "find_all_conflicts must not be called while building the initialize response"
    )


def _build_server(*, version: str | None = None) -> MCPServer:
    """Construct an ``MCPServer`` with stand-in collaborators.

    The collaborators are typed via ``cast`` to the protocol/class the
    constructor expects. They are never used: the test only inspects the
    ``initialize`` response payload, which is determined by the SDK
    ``Server`` configuration alone.
    """
    return MCPServer(
        store=cast("KnowledgeStore", object()),
        catalog=cast("ProjectCatalog", object()),
        coordinator=cast("IngestionCoordinator", object()),
        classify_pair=_unused_classify_pair,
        find_all_conflicts=_unused_find_all_conflicts,
        version=version,
    )


def _initialize_options(server: MCPServer) -> Any:
    """Return the ``InitializationOptions`` the SDK uses to answer ``initialize``.

    The on-the-wire ``initialize`` response is built from the same
    ``InitializationOptions`` record produced by
    :meth:`mcp.server.lowlevel.Server.create_initialization_options` (see the
    SDK's ``Server.run`` flow), so asserting against this record is
    equivalent to asserting against the wire response without paying the
    cost of running the stdio loop.

    The underlying SDK ``Server`` instance is held privately on
    :class:`MCPServer` and is exercised here, in a unit test, to keep the
    assertion close to the actual code path the design relies on.
    """
    sdk_server = server._server  # white-box access into the SDK Server
    return sdk_server.create_initialization_options()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initialize_response_carries_configured_name() -> None:
    """The ``initialize`` response carries the configured server ``name``.

    Implements Requirement 11.2.
    """
    server = _build_server(version="1.2.3")

    options = _initialize_options(server)

    assert options.server_name == SERVER_NAME
    assert options.server_name == "project-knowledge-mcp"
    # The public property and the wire payload agree.
    assert options.server_name == server.name


def test_initialize_response_carries_configured_version() -> None:
    """The ``initialize`` response carries a non-empty ``version`` string.

    A specific version is injected to pin down both that the field is
    present and that it is wired through verbatim from the constructor
    argument; the public property is asserted to agree with the wire
    payload so the two never drift.

    Implements Requirement 11.2.
    """
    server = _build_server(version="9.8.7")

    options = _initialize_options(server)

    assert isinstance(options.server_version, str)
    assert options.server_version != ""
    assert options.server_version == "9.8.7"
    assert options.server_version == server.version


def test_initialize_response_advertises_tools_capability() -> None:
    """The ``initialize`` response advertises ``capabilities.tools``.

    Per Requirement 11.2, the server must report ``tools`` as a server
    capability so MCP clients know they may issue ``tools/list`` and
    ``tools/call``. The SDK turns this bit on whenever a ``list_tools``
    handler is registered, which the skeleton does at construction.

    Implements Requirement 11.2.
    """
    server = _build_server(version="1.0.0")

    options = _initialize_options(server)
    capabilities = options.capabilities

    assert isinstance(capabilities, mcp_types.ServerCapabilities)
    # The ``tools`` capability is present (not ``None``); its concrete
    # shape is the SDK's ``ToolsCapability`` record. The design's
    # ``capabilities: {tools: {}}`` JSON shape is produced by Pydantic's
    # default-omitting serialization of this record.
    assert capabilities.tools is not None
    assert isinstance(capabilities.tools, mcp_types.ToolsCapability)

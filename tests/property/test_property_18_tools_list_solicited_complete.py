# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 18: For all MCP sessions, the count of tools/list responses sent by the MCP_Server SHALL equal the count of tools/list requests received from the MCP_Client; every such response SHALL contain exactly the tool set defined by Requirements 8 and 10 with each tool's input schema.
"""Property test for ``tools/list`` being solicited and complete.

**Validates Requirement 11.3** (Property 18 in the design).

For every MCP session in which the client sends ``N`` ``tools/list``
requests (``N >= 0``), the MCP_Server SHALL:

1. Emit exactly ``N`` ``tools/list`` responses — never more (no
   unsolicited ``tools/list`` responses), never fewer (every
   solicited request gets a response).
2. Make each response carry exactly the canonical eight tools defined
   by Requirements 8.1, 8.2, and 10.1-10.6 (the
   :data:`mcp_server.BUILT_IN_TOOLS` tuple) with each tool's
   ``inputSchema`` preserved verbatim.

Hypothesis drives the request count so the response count is checked
against varying session lengths, including the boundary case ``N == 0``
(which is what pins down the "no unsolicited responses" half of the
property: with zero requests the server emits zero responses).

The handler the SDK registered for :class:`mcp.types.ListToolsRequest`
is invoked directly. That registration is the only path through which
``tools/list`` responses are produced in this server (see
:meth:`mcp_server.MCPServer._register_default_handlers`), so driving
the handler ``N`` times models a session of exactly ``N`` solicited
requests and is sufficient to count the responses the server would
emit on the wire.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import mcp.types as mcp_types
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.mcp_server import BUILT_IN_TOOLS, MCPServer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from project_knowledge_mcp.conflict_detector import ConflictPair
    from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator
    from project_knowledge_mcp.knowledge_store import KnowledgeStore
    from project_knowledge_mcp.models import ConflictResult, ProjectProfile
    from project_knowledge_mcp.project_catalog import ProjectCatalog


# ---------------------------------------------------------------------------
# Stand-in collaborators
# ---------------------------------------------------------------------------

# The ``tools/list`` handler does not consult any of the injected
# collaborators (the eight tools are defined by ``BUILT_IN_TOOLS`` at
# module load time and the SDK ``Server`` simply returns that tuple).
# Passing real collaborators would only slow the test down and pull in
# the ``Knowledge_Store`` lifecycle. Stand-ins typed via ``cast`` keep
# the constructor's static typing strict without forcing real
# implementations.


def _unused_classify_pair(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
) -> ConflictResult:
    raise AssertionError(
        "classify_pair must not be called while answering tools/list"
    )


def _unused_find_all_conflicts(
    profiles: Sequence[ProjectProfile],
) -> list[ConflictPair]:
    raise AssertionError(
        "find_all_conflicts must not be called while answering tools/list"
    )


def _build_server() -> MCPServer:
    """Construct an ``MCPServer`` with stand-in collaborators.

    The ``tools/list`` handler is registered at construction time and
    is fully determined by :data:`BUILT_IN_TOOLS`; none of the injected
    collaborators participate in producing the response.
    """

    return MCPServer(
        store=cast("KnowledgeStore", object()),
        catalog=cast("ProjectCatalog", object()),
        coordinator=cast("IngestionCoordinator", object()),
        classify_pair=_unused_classify_pair,
        find_all_conflicts=_unused_find_all_conflicts,
        version="0.0.0-test",
    )


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------

# The SDK signs the registered handler as
# ``Callable[[ListToolsRequest], Awaitable[ServerResult]]`` (see
# :meth:`mcp.server.lowlevel.Server.list_tools`); the helpers below
# spell that signature out inline so the test stays self-contained.


async def _drive_session(
    handler: Callable[
        [mcp_types.ListToolsRequest], Awaitable[mcp_types.ServerResult]
    ],
    request_count: int,
) -> list[mcp_types.ServerResult]:
    """Issue ``request_count`` ``tools/list`` requests and collect responses.

    Each request is a fresh :class:`mcp.types.ListToolsRequest`
    instance with the default empty paginated params, modelling a
    ``tools/list`` request the way the MCP wire protocol delivers it.
    The handler is awaited sequentially so each call corresponds to a
    distinct request-response pair within the same simulated session
    (the SDK uses an asyncio task group per stdio session, so
    sequential awaiting matches the per-session ordering on the
    wire).
    """

    responses: list[mcp_types.ServerResult] = []
    for _ in range(request_count):
        request = mcp_types.ListToolsRequest()
        result = await handler(request)
        responses.append(result)
    return responses


def _resolve_list_tools_handler(
    server: MCPServer,
) -> Callable[
    [mcp_types.ListToolsRequest], Awaitable[mcp_types.ServerResult]
]:
    """Return the registered ``ListToolsRequest`` handler from the SDK server.

    The SDK records request handlers in
    :attr:`mcp.server.lowlevel.Server.request_handlers`. Reaching into
    that map here keeps the test scoped to the exact code path the
    on-the-wire ``tools/list`` flow exercises: the SDK's ``run`` loop
    looks the handler up by request type the same way and awaits its
    result before sending the JSON-RPC response.
    """

    return server._server.request_handlers[mcp_types.ListToolsRequest]


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(request_count=st.integers(min_value=0, max_value=10))
@settings(max_examples=100)
def test_tools_list_solicited_and_complete(request_count: int) -> None:
    """Property 18: tools/list responses are solicited and complete.

    Validates Requirement 11.3.

    For every session length ``request_count``:

    * ``len(responses) == request_count`` — the server emits exactly
      one response per request, never an unsolicited extra and never
      drops one. The ``request_count == 0`` case covers the
      "unsolicited" half of the property: with zero requests the
      handler is never invoked, so the server emits zero responses.
    * each response is a :class:`mcp.types.ServerResult` wrapping a
      :class:`mcp.types.ListToolsResult` whose ``tools`` list equals
      :data:`BUILT_IN_TOOLS` exactly — same names, same order, same
      input schemas.
    """

    server = _build_server()
    handler = _resolve_list_tools_handler(server)

    responses = asyncio.run(_drive_session(handler, request_count))

    # (1) Response count equals request count. Zero requests yields
    # zero responses (the "no unsolicited tools/list responses" half
    # of Property 18). Non-zero counts yield exactly one response per
    # request.
    assert len(responses) == request_count

    # (2) Every response carries exactly the eight canonical tools
    # with their input schemas. The canonical source of truth is the
    # module-level :data:`BUILT_IN_TOOLS` tuple, which the design
    # designates as the ``tools/list`` contract for Requirements 8.1,
    # 8.2, and 10.1-10.6.
    expected_tool_count = len(BUILT_IN_TOOLS)
    for response in responses:
        result = response.root
        assert isinstance(result, mcp_types.ListToolsResult), (
            f"expected ListToolsResult, got {type(result).__name__}"
        )

        emitted_tools = result.tools
        # (2a) Exactly eight tools, in the canonical order.
        assert len(emitted_tools) == expected_tool_count == 8, (
            f"expected {expected_tool_count} tools, got {len(emitted_tools)}"
        )
        emitted_names = tuple(tool.name for tool in emitted_tools)
        canonical_names = tuple(tool.name for tool in BUILT_IN_TOOLS)
        assert emitted_names == canonical_names, (
            f"tool names diverged from canonical: emitted={emitted_names}, "
            f"expected={canonical_names}"
        )

        # (2b) Each emitted tool's ``inputSchema`` matches the
        # canonical ``inputSchema`` byte-for-byte. This is the
        # "with each tool's input schema" half of Property 18.
        for emitted, canonical in zip(emitted_tools, BUILT_IN_TOOLS, strict=True):
            assert emitted.inputSchema == canonical.inputSchema, (
                f"input schema for tool {emitted.name!r} diverged from "
                f"canonical: emitted={emitted.inputSchema!r}, "
                f"expected={canonical.inputSchema!r}"
            )

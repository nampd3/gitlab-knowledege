"""Unit tests for ``tools/list`` and end-to-end happy paths (task 9.5).

Task 9.5 pins down two complementary parts of the MCP transport
surface:

1. ``tools/list`` SHALL return *exactly* the eight tools defined by
   Requirements 8.1, 8.2, and 10.1-10.6, with each tool's
   ``inputSchema`` matching the design's argument shape (no-args for
   ``list_projects`` / ``list_purpose_conflicts`` /
   ``refresh_all_projects``; an integer ``gitlab_project_id`` for the
   four read tools and ``refresh_project``).
2. Every one of those eight tools SHALL produce the documented happy
   path response shape when invoked through the actual ``tools/call``
   dispatcher (i.e. via :class:`mcp.types.CallToolRequest`), not just
   via direct handler lookup. This complements the per-tool wiring
   coverage in ``test_mcp_built_in_tools.py`` by exercising the SDK
   request handler that production traffic travels through.

Implements Requirements 8.1, 8.2, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types
import pytest

from project_knowledge_mcp.conflict_detector import classify_pair, find_all_conflicts
from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.mcp_server import (
    BUILT_IN_TOOLS,
    TOOL_NAME_GET_PROJECT_DEPENDENCIES,
    TOOL_NAME_GET_PROJECT_IO,
    TOOL_NAME_GET_PROJECT_PROFILE,
    TOOL_NAME_GET_PROJECT_PURPOSE,
    TOOL_NAME_LIST_PROJECTS,
    TOOL_NAME_LIST_PURPOSE_CONFLICTS,
    TOOL_NAME_REFRESH_ALL_PROJECTS,
    TOOL_NAME_REFRESH_PROJECT,
    MCPServer,
)
from project_knowledge_mcp.models import (
    EnumeratedProject,
    ProjectProfile,
    SnapshotTrigger,
)
from project_knowledge_mcp.project_catalog import ProjectCatalog

if TYPE_CHECKING:
    from pathlib import Path

    from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# The eight tool names mandated by Requirements 8.1, 8.2 and 10.1-10.6.
# Listed explicitly (rather than derived from ``BUILT_IN_TOOLS``) so that
# any drift between the design and the registry is caught by this test
# rather than silently ratified.
EXPECTED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        TOOL_NAME_LIST_PROJECTS,
        TOOL_NAME_GET_PROJECT_PURPOSE,
        TOOL_NAME_GET_PROJECT_IO,
        TOOL_NAME_GET_PROJECT_DEPENDENCIES,
        TOOL_NAME_GET_PROJECT_PROFILE,
        TOOL_NAME_LIST_PURPOSE_CONFLICTS,
        TOOL_NAME_REFRESH_ALL_PROJECTS,
        TOOL_NAME_REFRESH_PROJECT,
    }
)

# The three tools whose ``inputSchema`` is the empty-object schema
# (no arguments). Used to check schemas in the parametrized test below.
NO_ARGS_TOOL_NAMES: frozenset[str] = frozenset(
    {
        TOOL_NAME_LIST_PROJECTS,
        TOOL_NAME_LIST_PURPOSE_CONFLICTS,
        TOOL_NAME_REFRESH_ALL_PROJECTS,
    }
)

# The five tools whose ``inputSchema`` requires an integer
# ``gitlab_project_id``.
PROJECT_ID_TOOL_NAMES: frozenset[str] = frozenset(
    {
        TOOL_NAME_GET_PROJECT_PURPOSE,
        TOOL_NAME_GET_PROJECT_IO,
        TOOL_NAME_GET_PROJECT_DEPENDENCIES,
        TOOL_NAME_GET_PROJECT_PROFILE,
        TOOL_NAME_REFRESH_PROJECT,
    }
)


# ---------------------------------------------------------------------------
# Test doubles + fixture helpers
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    """Records refresh calls so the ``refresh_*`` happy paths can be observed.

    The real :class:`IngestionCoordinator` orchestrates the
    ``GitLab_Connector`` + ``Project_Analyzer`` pipeline; this fake
    captures invocations without performing any work so the
    happy-path assertions can verify that the MCP refresh tools
    actually dispatch into the coordinator (Requirements 8.1, 8.2)
    and produce the documented ``status: completed`` payload.
    """

    def __init__(self) -> None:
        self.full_refresh_calls: int = 0
        self.single_refresh_calls: list[int] = []

    def start_full_refresh(self) -> None:
        self.full_refresh_calls += 1

    def start_single_project_refresh(self, gitlab_project_id: int) -> None:
        self.single_refresh_calls.append(gitlab_project_id)


def _build_server(
    *,
    store: KnowledgeStore,
    catalog: ProjectCatalog,
    coordinator: _FakeCoordinator | None = None,
) -> tuple[MCPServer, _FakeCoordinator]:
    """Construct an :class:`MCPServer` wired with the production helpers.

    The conflict-detector callables are the real production functions
    (they are pure and have no external dependencies); the coordinator
    is the local fake so refresh-tool calls can be observed.
    """

    actual_coordinator = coordinator if coordinator is not None else _FakeCoordinator()
    server = MCPServer(
        store=store,
        catalog=catalog,
        coordinator=cast("IngestionCoordinator", actual_coordinator),
        classify_pair=classify_pair,
        find_all_conflicts=find_all_conflicts,
        version="0.0.0-test",
    )
    return server, actual_coordinator


def _populate_single_project(
    store: KnowledgeStore,
    catalog: ProjectCatalog,
    *,
    gitlab_project_id: int,
    full_path: str,
) -> ProjectProfile:
    """Commit a single in-scope project + minimal profile to the store.

    Returns the persisted :class:`ProjectProfile` so the happy-path
    assertions can compare structured payloads against the canonical
    ``model_dump(mode="json")`` form.
    """

    profile = ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha="deadbeef" * 5,
        produced_at=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
        purpose_summary="owns the customer onboarding workflow",
        purpose_summary_reason=None,
        abstract_inputs=[],
        abstract_outputs=[],
        external_service_dependencies=[],
        database_table_dependencies=[],
    )
    snapshot_id = store.begin_snapshot(SnapshotTrigger.FULL)
    catalog.populate_in_scope(
        snapshot_id,
        [
            EnumeratedProject(
                gitlab_project_id=gitlab_project_id,
                full_path=full_path,
                analysis_branch_name="uat",
                analysis_branch_commit_sha="deadbeef" * 5,
                branch_missing=False,
            ),
        ],
    )
    store.write_profile(
        snapshot_id,
        profile,
        produced_at=profile.produced_at,
        commit_sha=profile.analysis_branch_commit_sha,
    )
    store.commit_snapshot(snapshot_id)
    return profile


# ---------------------------------------------------------------------------
# tools/list helpers
# ---------------------------------------------------------------------------


async def _invoke_tools_list(server: MCPServer) -> mcp_types.ListToolsResult:
    """Invoke the SDK-registered ``tools/list`` handler and return the result.

    Reaches into :attr:`MCPServer._server.request_handlers` to call the
    handler the SDK wired at construction time. This is the same code
    path the on-the-wire ``tools/list`` request travels: the SDK
    parses the JSON-RPC request into a :class:`mcp.types.ListToolsRequest`,
    looks the request type up in ``request_handlers``, and awaits the
    handler. By replicating that lookup here the test is exercising
    the production handler, not a separate code path.
    """

    handler = server._server.request_handlers[mcp_types.ListToolsRequest]
    request = mcp_types.ListToolsRequest()
    server_result = await handler(request)
    assert isinstance(server_result, mcp_types.ServerResult)
    inner = server_result.root
    assert isinstance(inner, mcp_types.ListToolsResult)
    return inner


async def _invoke_tools_call(
    server: MCPServer, *, name: str, arguments: dict[str, Any] | None = None
) -> mcp_types.CallToolResult:
    """Invoke the SDK-registered ``tools/call`` dispatcher and return the result.

    Mirrors :func:`_invoke_tools_list` for ``tools/call``: looks up the
    handler in :attr:`request_handlers` and awaits it with a
    :class:`CallToolRequest`. Going through the dispatcher (rather than
    the ``_tool_handlers`` registry directly) exercises the schema
    validation step alongside the happy path so the schemas declared
    by ``tools/list`` and the dispatcher's validator stay in agreement.
    """

    handler = server._server.request_handlers[mcp_types.CallToolRequest]
    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(name=name, arguments=arguments),
    )
    server_result = await handler(request)
    assert isinstance(server_result, mcp_types.ServerResult)
    inner = server_result.root
    assert isinstance(inner, mcp_types.CallToolResult)
    return inner


def _structured(result: mcp_types.CallToolResult) -> dict[str, Any]:
    """Return ``result.structuredContent`` after asserting non-None."""

    assert result.structuredContent is not None, (
        "happy-path tool results must carry structuredContent"
    )
    return result.structuredContent


def _text(result: mcp_types.CallToolResult) -> str:
    """Return the text of the single ``TextContent`` block in ``result``."""

    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


# ---------------------------------------------------------------------------
# tools/list returns exactly the eight tools by name
# ---------------------------------------------------------------------------


async def test_tools_list_returns_exactly_eight_tools_by_name(tmp_path: Path) -> None:
    """``tools/list`` SHALL return exactly the eight tool names from the design.

    Implements Requirements 8.1, 8.2, 10.1, 10.2, 10.3, 10.4, 10.5,
    10.6.
    """

    store = KnowledgeStore.open(tmp_path / "tools_list.db")
    try:
        server, _ = _build_server(store=store, catalog=ProjectCatalog(store))

        result = await _invoke_tools_list(server)

        returned_names = [tool.name for tool in result.tools]
        # Exactly eight tools, no duplicates, exactly the eight names
        # mandated by the design.
        assert len(returned_names) == 8
        assert len(set(returned_names)) == 8
        assert set(returned_names) == EXPECTED_TOOL_NAMES
    finally:
        store.close()


async def test_tools_list_response_matches_built_in_tools_registry(
    tmp_path: Path,
) -> None:
    """``tools/list`` SHALL return the canonical ``BUILT_IN_TOOLS`` records.

    The response advertises exactly the same :class:`Tool` objects
    (name + description + inputSchema) that the module-level
    ``BUILT_IN_TOOLS`` constant carries, so the wire surface and the
    in-process registry never disagree.

    Implements Requirement 11.3 (and indirectly Requirements 8.1,
    8.2, 10.1-10.6).
    """

    store = KnowledgeStore.open(tmp_path / "tools_list_match.db")
    try:
        server, _ = _build_server(store=store, catalog=ProjectCatalog(store))

        result = await _invoke_tools_list(server)

        # Compare on a per-tool basis so the failure message names
        # the offending tool. Index by name because order is part of
        # the canonical surface but the content equality is what
        # matters per requirement.
        returned_by_name = {tool.name: tool for tool in result.tools}
        canonical_by_name = {tool.name: tool for tool in BUILT_IN_TOOLS}
        assert returned_by_name.keys() == canonical_by_name.keys()
        for name, canonical in canonical_by_name.items():
            returned = returned_by_name[name]
            assert returned.name == canonical.name
            assert returned.description == canonical.description
            assert returned.inputSchema == canonical.inputSchema
    finally:
        store.close()


@pytest.mark.parametrize("tool_name", sorted(NO_ARGS_TOOL_NAMES))
async def test_tools_list_no_args_tool_has_empty_object_schema(
    tmp_path: Path, tool_name: str
) -> None:
    """The three no-argument tools advertise the empty-object input schema.

    The schema shape ``{"type": "object", "properties": {}}`` is the
    contract used by the dispatcher's :mod:`jsonschema` validator to
    accept the empty arguments mapping for these tools (Requirements
    8.1, 10.4, 10.6 + Requirement 11.3 / 11.6).
    """

    store = KnowledgeStore.open(tmp_path / f"schema_noargs_{tool_name}.db")
    try:
        server, _ = _build_server(store=store, catalog=ProjectCatalog(store))

        result = await _invoke_tools_list(server)
        tool = next(t for t in result.tools if t.name == tool_name)

        assert tool.inputSchema == {"type": "object", "properties": {}}
    finally:
        store.close()


@pytest.mark.parametrize("tool_name", sorted(PROJECT_ID_TOOL_NAMES))
async def test_tools_list_project_id_tool_schema_requires_integer(
    tmp_path: Path, tool_name: str
) -> None:
    """The five project-id-typed tools advertise the integer-id schema.

    Each project-id-typed tool's ``inputSchema`` requires an integer
    ``gitlab_project_id`` field — the contract clients use to decide
    whether their argument is well-formed before issuing a
    ``tools/call`` request.

    Implements Requirements 8.2, 10.1, 10.2, 10.3, 10.5.
    """

    store = KnowledgeStore.open(tmp_path / f"schema_id_{tool_name}.db")
    try:
        server, _ = _build_server(store=store, catalog=ProjectCatalog(store))

        result = await _invoke_tools_list(server)
        tool = next(t for t in result.tools if t.name == tool_name)

        schema = tool.inputSchema
        assert schema["type"] == "object"
        assert schema["required"] == ["gitlab_project_id"]
        properties = schema["properties"]
        assert set(properties.keys()) == {"gitlab_project_id"}
        assert properties["gitlab_project_id"]["type"] == "integer"
    finally:
        store.close()


async def test_tools_list_each_tool_carries_a_non_empty_description(
    tmp_path: Path,
) -> None:
    """Every advertised tool carries a non-empty human-readable description.

    Clients render the description in tool pickers; an empty string
    would be a regression even though the protocol does not strictly
    require it.
    """

    store = KnowledgeStore.open(tmp_path / "descriptions.db")
    try:
        server, _ = _build_server(store=store, catalog=ProjectCatalog(store))

        result = await _invoke_tools_list(server)

        for tool in result.tools:
            assert tool.description is not None
            assert tool.description.strip() != ""
    finally:
        store.close()


# ---------------------------------------------------------------------------
# End-to-end happy paths through the ``tools/call`` dispatcher
# ---------------------------------------------------------------------------
#
# These tests exercise each tool's happy path through the SDK
# ``tools/call`` dispatcher (i.e. the same ``request_handlers`` lookup
# the on-the-wire flow uses). Per-tool registration and direct handler
# happy paths are covered by ``test_mcp_built_in_tools.py``; the
# distinction here is that these tests run through the schema validator
# step in :meth:`MCPServer._dispatch_tool_call` so the ``tools/list``
# schemas and the dispatcher's validator stay in agreement.


async def test_list_projects_happy_path_via_dispatcher(tmp_path: Path) -> None:
    """``tools/call list_projects`` SHALL return the in-scope catalog rows.

    Implements Requirement 10.6.
    """

    store = KnowledgeStore.open(tmp_path / "hp_list_projects.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_single_project(store, catalog, gitlab_project_id=101, full_path="group/alpha")
        server, _ = _build_server(store=store, catalog=catalog)

        result = await _invoke_tools_call(server, name=TOOL_NAME_LIST_PROJECTS)

        assert result.isError is False
        payload = _structured(result)
        assert payload == {
            "projects": [
                {"gitlab_project_id": 101, "full_path": "group/alpha"},
            ],
        }
        # Text and structured surfaces stay aligned.
        assert json.loads(_text(result)) == payload
    finally:
        store.close()


async def test_get_project_purpose_happy_path_via_dispatcher(tmp_path: Path) -> None:
    """``tools/call get_project_purpose`` SHALL return the purpose summary.

    Implements Requirement 10.1.
    """

    store = KnowledgeStore.open(tmp_path / "hp_purpose.db")
    try:
        catalog = ProjectCatalog(store)
        profile = _populate_single_project(
            store, catalog, gitlab_project_id=101, full_path="group/alpha"
        )
        server, _ = _build_server(store=store, catalog=catalog)

        result = await _invoke_tools_call(
            server,
            name=TOOL_NAME_GET_PROJECT_PURPOSE,
            arguments={"gitlab_project_id": 101},
        )

        assert result.isError is False
        payload = _structured(result)
        assert payload == {
            "gitlab_project_id": 101,
            "purpose_summary": profile.purpose_summary,
            "purpose_summary_reason": None,
        }
    finally:
        store.close()


async def test_get_project_io_happy_path_via_dispatcher(tmp_path: Path) -> None:
    """``tools/call get_project_io`` SHALL return Abstract_Inputs / Abstract_Outputs.

    Per Requirements 4.5 and 4.6 the lists are present even when
    empty, so an in-scope project with no detected I/O still produces
    two empty lists.

    Implements Requirement 10.2.
    """

    store = KnowledgeStore.open(tmp_path / "hp_io.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_single_project(store, catalog, gitlab_project_id=101, full_path="group/alpha")
        server, _ = _build_server(store=store, catalog=catalog)

        result = await _invoke_tools_call(
            server,
            name=TOOL_NAME_GET_PROJECT_IO,
            arguments={"gitlab_project_id": 101},
        )

        assert result.isError is False
        payload = _structured(result)
        assert payload == {
            "gitlab_project_id": 101,
            "abstract_inputs": [],
            "abstract_outputs": [],
        }
    finally:
        store.close()


async def test_get_project_dependencies_happy_path_via_dispatcher(
    tmp_path: Path,
) -> None:
    """``tools/call get_project_dependencies`` SHALL return services + tables.

    Implements Requirement 10.3.
    """

    store = KnowledgeStore.open(tmp_path / "hp_deps.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_single_project(store, catalog, gitlab_project_id=101, full_path="group/alpha")
        server, _ = _build_server(store=store, catalog=catalog)

        result = await _invoke_tools_call(
            server,
            name=TOOL_NAME_GET_PROJECT_DEPENDENCIES,
            arguments={"gitlab_project_id": 101},
        )

        assert result.isError is False
        payload = _structured(result)
        assert payload == {
            "gitlab_project_id": 101,
            "external_service_dependencies": [],
            "database_table_dependencies": [],
        }
    finally:
        store.close()


async def test_get_project_profile_happy_path_via_dispatcher(tmp_path: Path) -> None:
    """``tools/call get_project_profile`` SHALL return the full Project_Profile.

    Implements Requirement 10.5.
    """

    store = KnowledgeStore.open(tmp_path / "hp_profile.db")
    try:
        catalog = ProjectCatalog(store)
        profile = _populate_single_project(
            store, catalog, gitlab_project_id=101, full_path="group/alpha"
        )
        server, _ = _build_server(store=store, catalog=catalog)

        result = await _invoke_tools_call(
            server,
            name=TOOL_NAME_GET_PROJECT_PROFILE,
            arguments={"gitlab_project_id": 101},
        )

        assert result.isError is False
        payload = _structured(result)
        assert payload == profile.model_dump(mode="json")
    finally:
        store.close()


async def test_list_purpose_conflicts_happy_path_via_dispatcher(
    tmp_path: Path,
) -> None:
    """``tools/call list_purpose_conflicts`` SHALL return the conflict list.

    With a single profile present there is no pair to classify, so
    the documented happy-path response is the empty conflict list.

    Implements Requirement 10.4.
    """

    store = KnowledgeStore.open(tmp_path / "hp_conflicts.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_single_project(store, catalog, gitlab_project_id=101, full_path="group/alpha")
        server, _ = _build_server(store=store, catalog=catalog)

        result = await _invoke_tools_call(server, name=TOOL_NAME_LIST_PURPOSE_CONFLICTS)

        assert result.isError is False
        payload = _structured(result)
        assert payload == {"conflicts": []}
    finally:
        store.close()


async def test_refresh_all_projects_happy_path_via_dispatcher(tmp_path: Path) -> None:
    """``tools/call refresh_all_projects`` SHALL drive a full refresh.

    The MCP layer dispatches into
    ``Ingestion_Coordinator.start_full_refresh`` and produces the
    documented ``status: completed`` payload (Requirement 8.1).
    """

    store = KnowledgeStore.open(tmp_path / "hp_refresh_all.db")
    try:
        server, coordinator = _build_server(store=store, catalog=ProjectCatalog(store))

        result = await _invoke_tools_call(server, name=TOOL_NAME_REFRESH_ALL_PROJECTS)

        assert coordinator.full_refresh_calls == 1
        assert coordinator.single_refresh_calls == []
        assert result.isError is False
        payload = _structured(result)
        assert payload == {"status": "completed", "trigger": "full"}
    finally:
        store.close()


async def test_refresh_project_happy_path_via_dispatcher(tmp_path: Path) -> None:
    """``tools/call refresh_project`` SHALL drive a single-project refresh.

    The MCP layer dispatches into
    ``Ingestion_Coordinator.start_single_project_refresh`` with the
    requested ``gitlab_project_id`` and produces the documented
    ``status: completed`` payload (Requirement 8.2).
    """

    store = KnowledgeStore.open(tmp_path / "hp_refresh_one.db")
    try:
        server, coordinator = _build_server(store=store, catalog=ProjectCatalog(store))

        result = await _invoke_tools_call(
            server,
            name=TOOL_NAME_REFRESH_PROJECT,
            arguments={"gitlab_project_id": 101},
        )

        assert coordinator.single_refresh_calls == [101]
        assert coordinator.full_refresh_calls == 0
        assert result.isError is False
        payload = _structured(result)
        assert payload == {
            "status": "completed",
            "trigger": "single_project",
            "gitlab_project_id": 101,
        }
    finally:
        store.close()

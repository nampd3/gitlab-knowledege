"""Unit tests for the built-in tool registration and handlers (task 9.4).

These tests pin down the behaviour added in task 9.4: that
:meth:`MCPServer.register_built_in_tools` registers a handler for every
one of the eight tools defined by Requirements 8 and 10, and that each
handler dispatches to the injected collaborator and surfaces errors per
the design's MCP error mapping table.

Coverage:

* All eight tools advertised by ``BUILT_IN_TOOLS`` are wired with a
  handler at construction time (Requirements 8.1, 8.2, 10.1-10.6).
* ``list_projects`` returns the catalog rows ordered by
  ``gitlab_project_id`` ascending (Requirement 10.6).
* ``get_project_purpose`` / ``get_project_io`` /
  ``get_project_dependencies`` / ``get_project_profile`` each return
  the relevant subset of the persisted ``Project_Profile`` for an
  in-scope, analyzed project (Requirements 10.1-10.3, 10.5).
* The same four read tools surface the canonical
  ``"project {id} is not in scope"`` message via ``isError: true`` for
  any GitLab project ID not present in the current
  ``Project_Catalog`` (Requirement 10.7 / Property 17).
* The same four read tools surface a non-error "not yet analyzed"
  message when the project is in scope but the current snapshot has
  no profile for it (mirroring Requirement 14.3 from the
  Visualization_Server side).
* ``list_purpose_conflicts`` runs the injected
  ``find_all_conflicts`` over the current snapshot's profiles and
  returns the resulting :class:`ConflictPair` records (Requirement
  10.4).
* ``refresh_all_projects`` and ``refresh_project`` dispatch to
  ``Ingestion_Coordinator.start_full_refresh`` and
  ``start_single_project_refresh`` respectively, surface
  ``IngestionInProgressError`` with the canonical Requirement 8.6
  wording, and ``refresh_project`` surfaces
  ``ProjectNotInScopeError`` with the canonical Requirement 10.7
  wording.

Implements Requirements 8.1, 8.2, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6,
10.7. Targets Property 17.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types
import pytest

from project_knowledge_mcp.conflict_detector import (
    ConflictPair,
    classify_pair,
    find_all_conflicts,
)
from project_knowledge_mcp.errors import (
    IngestionInProgressError,
    ProjectNotInScopeError,
)
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
    ToolHandler,
)
from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
    DatabaseAccessMode,
    DatabaseTableDependency,
    EnumeratedProject,
    ExternalServiceDependency,
    ExternalServiceKind,
    ProjectProfile,
    SnapshotTrigger,
    SourceLocation,
)
from project_knowledge_mcp.project_catalog import ProjectCatalog

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    """Records refresh calls and replays scripted outcomes.

    Mirrors the public surface of ``IngestionCoordinator`` that the
    MCP refresh tools depend on. Tests configure
    ``full_refresh_raises`` / ``single_refresh_raises`` to force the
    coordinator to raise a specific exception (e.g.
    :class:`IngestionInProgressError`) on the next call, simulating a
    rejected start without having to drive the real coordinator
    through a concurrent transition.
    """

    def __init__(self) -> None:
        self.full_refresh_calls: int = 0
        self.single_refresh_calls: list[int] = []
        self.full_refresh_raises: BaseException | None = None
        self.single_refresh_raises: BaseException | None = None

    def start_full_refresh(self) -> None:
        self.full_refresh_calls += 1
        if self.full_refresh_raises is not None:
            raise self.full_refresh_raises

    def start_single_project_refresh(self, gitlab_project_id: int) -> None:
        self.single_refresh_calls.append(gitlab_project_id)
        if self.single_refresh_raises is not None:
            raise self.single_refresh_raises


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _enumerated(
    gitlab_project_id: int,
    full_path: str,
    *,
    branch: str = "uat",
    commit_sha: str = "deadbeef" * 5,
) -> EnumeratedProject:
    return EnumeratedProject(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch_name=branch,
        analysis_branch_commit_sha=commit_sha,
        branch_missing=False,
    )


def _build_profile(
    *,
    gitlab_project_id: int,
    full_path: str,
    purpose_summary: str = "owns the customer onboarding workflow",
    purpose_summary_reason: str | None = None,
    abstract_inputs: list[AbstractInput] | None = None,
    abstract_outputs: list[AbstractOutput] | None = None,
    external_service_dependencies: list[ExternalServiceDependency] | None = None,
    database_table_dependencies: list[DatabaseTableDependency] | None = None,
) -> ProjectProfile:
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha="deadbeef" * 5,
        produced_at=datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
        purpose_summary=purpose_summary,
        purpose_summary_reason=purpose_summary_reason,
        abstract_inputs=abstract_inputs or [],
        abstract_outputs=abstract_outputs or [],
        external_service_dependencies=external_service_dependencies or [],
        database_table_dependencies=database_table_dependencies or [],
    )


def _build_server(
    *,
    store: KnowledgeStore,
    catalog: ProjectCatalog,
    coordinator: _FakeCoordinator | None = None,
) -> MCPServer:
    """Construct an ``MCPServer`` with a real store/catalog and a fake coordinator.

    The injected ``classify_pair`` and ``find_all_conflicts`` are the
    real production functions (they are pure and have no external
    dependencies), so the MCP layer's behaviour is exercised against
    the same conflict-detection logic the production process uses.
    """

    actual_coordinator = coordinator if coordinator is not None else _FakeCoordinator()
    return MCPServer(
        store=store,
        catalog=catalog,
        coordinator=cast("IngestionCoordinator", actual_coordinator),
        classify_pair=classify_pair,
        find_all_conflicts=find_all_conflicts,
        version="0.0.0-test",
    )


def _populate_catalog_and_profiles(
    store: KnowledgeStore,
    catalog: ProjectCatalog,
    enumerated: list[EnumeratedProject],
    profiles: list[ProjectProfile],
) -> None:
    """Run a minimal full-refresh-shaped write/commit sequence."""

    snapshot_id = store.begin_snapshot(SnapshotTrigger.FULL)
    catalog.populate_in_scope(snapshot_id, enumerated)
    for profile in profiles:
        store.write_profile(
            snapshot_id,
            profile,
            produced_at=profile.produced_at,
            commit_sha=profile.analysis_branch_commit_sha,
        )
    store.commit_snapshot(snapshot_id)


# ---------------------------------------------------------------------------
# Helpers for inspecting CallToolResult
# ---------------------------------------------------------------------------


def _text_of(result: mcp_types.CallToolResult) -> str:
    """Return the text of the single ``TextContent`` block in ``result``."""

    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


def _structured_payload(result: mcp_types.CallToolResult) -> dict[str, Any]:
    """Return ``result.structuredContent`` after asserting non-None."""

    assert result.structuredContent is not None
    return result.structuredContent


def _resolve_handler(server: MCPServer, tool_name: str) -> ToolHandler:
    """Resolve the registered handler for ``tool_name`` on ``server``."""

    # The handler registry is private; the dispatcher consults it
    # the same way. Reaching into it here keeps the test focused on
    # task 9.4's wiring (Requirement 11 dispatch behaviour is
    # exercised separately in the dispatcher tests).
    handler = server._tool_handlers.get(tool_name)
    assert handler is not None, f"tool {tool_name!r} is not registered"
    return handler


# ---------------------------------------------------------------------------
# Registration coverage
# ---------------------------------------------------------------------------


def test_register_built_in_tools_registers_all_eight_tools(tmp_path: Path) -> None:
    """All eight tool names from ``BUILT_IN_TOOLS`` have a registered handler.

    Implements Requirements 8.1, 8.2, 10.1, 10.2, 10.3, 10.4, 10.5,
    10.6.
    """

    store = KnowledgeStore.open(tmp_path / "registration.db")
    try:
        server = _build_server(store=store, catalog=ProjectCatalog(store))

        registered_names = set(server._tool_handlers.keys())
        expected_names = {tool.name for tool in BUILT_IN_TOOLS}
        assert registered_names == expected_names
        assert len(registered_names) == 8
    finally:
        store.close()


# ---------------------------------------------------------------------------
# list_projects (Requirement 10.6)
# ---------------------------------------------------------------------------


async def test_list_projects_returns_in_scope_projects_in_ascending_order(
    tmp_path: Path,
) -> None:
    """``list_projects`` mirrors ``ProjectCatalog.list_in_scope``.

    Implements Requirement 10.6.
    """

    store = KnowledgeStore.open(tmp_path / "list.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[
                _enumerated(303, "group/charlie"),
                _enumerated(101, "group/alpha"),
                _enumerated(202, "group/bravo"),
            ],
            profiles=[],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, TOOL_NAME_LIST_PROJECTS)
        result = await handler({})

        assert result.isError is False
        payload = _structured_payload(result)
        assert payload == {
            "projects": [
                {"gitlab_project_id": 101, "full_path": "group/alpha"},
                {"gitlab_project_id": 202, "full_path": "group/bravo"},
                {"gitlab_project_id": 303, "full_path": "group/charlie"},
            ],
        }
        # The text content is the JSON form of the same payload, so
        # text-mode and structured-mode clients agree.
        assert json.loads(_text_of(result)) == payload
    finally:
        store.close()


async def test_list_projects_returns_empty_list_on_fresh_store(tmp_path: Path) -> None:
    """``list_projects`` returns ``{"projects": []}`` before any commit."""

    store = KnowledgeStore.open(tmp_path / "fresh.db")
    try:
        server = _build_server(store=store, catalog=ProjectCatalog(store))

        handler = _resolve_handler(server, TOOL_NAME_LIST_PROJECTS)
        result = await handler({})

        assert result.isError is False
        assert _structured_payload(result) == {"projects": []}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Read tools: in-scope + analyzed (Requirements 10.1-10.3, 10.5)
# ---------------------------------------------------------------------------


async def test_get_project_purpose_returns_summary_for_in_scope_project(
    tmp_path: Path,
) -> None:
    """``get_project_purpose`` returns purpose_summary + reason.

    Implements Requirement 10.1.
    """

    store = KnowledgeStore.open(tmp_path / "purpose.db")
    try:
        catalog = ProjectCatalog(store)
        profile = _build_profile(
            gitlab_project_id=101,
            full_path="group/alpha",
            purpose_summary="owns customer onboarding",
            purpose_summary_reason=None,
        )
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[_enumerated(101, "group/alpha")],
            profiles=[profile],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, TOOL_NAME_GET_PROJECT_PURPOSE)
        result = await handler({"gitlab_project_id": 101})

        assert result.isError is False
        payload = _structured_payload(result)
        assert payload == {
            "gitlab_project_id": 101,
            "purpose_summary": "owns customer onboarding",
            "purpose_summary_reason": None,
        }
    finally:
        store.close()


async def test_get_project_io_returns_inputs_and_outputs(tmp_path: Path) -> None:
    """``get_project_io`` returns Abstract_Inputs and Abstract_Outputs.

    Implements Requirement 10.2.
    """

    store = KnowledgeStore.open(tmp_path / "io.db")
    try:
        catalog = ProjectCatalog(store)
        profile = _build_profile(
            gitlab_project_id=101,
            full_path="group/alpha",
            abstract_inputs=[
                AbstractInput(
                    category=AbstractInputCategory.HTTP_REQUEST,
                    description="POST /orders carrying purchase requests",
                ),
            ],
            abstract_outputs=[
                AbstractOutput(
                    category=AbstractOutputCategory.MESSAGE_PUBLISHED,
                    description="publishes order-confirmed events",
                ),
            ],
        )
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[_enumerated(101, "group/alpha")],
            profiles=[profile],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, TOOL_NAME_GET_PROJECT_IO)
        result = await handler({"gitlab_project_id": 101})

        assert result.isError is False
        payload = _structured_payload(result)
        assert payload["gitlab_project_id"] == 101
        assert payload["abstract_inputs"] == [
            {
                "category": "http_request",
                "description": "POST /orders carrying purchase requests",
            }
        ]
        assert payload["abstract_outputs"] == [
            {
                "category": "message_published",
                "description": "publishes order-confirmed events",
            }
        ]
    finally:
        store.close()


async def test_get_project_dependencies_returns_services_and_tables(
    tmp_path: Path,
) -> None:
    """``get_project_dependencies`` returns external services + db tables.

    Implements Requirement 10.3.
    """

    store = KnowledgeStore.open(tmp_path / "deps.db")
    try:
        catalog = ProjectCatalog(store)
        profile = _build_profile(
            gitlab_project_id=101,
            full_path="group/alpha",
            external_service_dependencies=[
                ExternalServiceDependency(
                    name="payments-api",
                    kind=ExternalServiceKind.HTTP_API,
                    source_locations=[SourceLocation(path="src/clients/payments.py")],
                ),
            ],
            database_table_dependencies=[
                DatabaseTableDependency(
                    table_name="orders",
                    access_mode=DatabaseAccessMode.READ_WRITE,
                    source_locations=[SourceLocation(path="src/repositories/orders.py")],
                ),
            ],
        )
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[_enumerated(101, "group/alpha")],
            profiles=[profile],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, TOOL_NAME_GET_PROJECT_DEPENDENCIES)
        result = await handler({"gitlab_project_id": 101})

        assert result.isError is False
        payload = _structured_payload(result)
        assert payload["gitlab_project_id"] == 101
        assert payload["external_service_dependencies"] == [
            {
                "name": "payments-api",
                "kind": "http_api",
                "source_locations": [{"path": "src/clients/payments.py", "line": None}],
            }
        ]
        assert payload["database_table_dependencies"] == [
            {
                "table_name": "orders",
                "access_mode": "read_write",
                "source_locations": [
                    {"path": "src/repositories/orders.py", "line": None}
                ],
            }
        ]
    finally:
        store.close()


async def test_get_project_profile_returns_full_profile(tmp_path: Path) -> None:
    """``get_project_profile`` returns the full Project_Profile payload.

    Implements Requirement 10.5.
    """

    store = KnowledgeStore.open(tmp_path / "profile.db")
    try:
        catalog = ProjectCatalog(store)
        profile = _build_profile(
            gitlab_project_id=101,
            full_path="group/alpha",
        )
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[_enumerated(101, "group/alpha")],
            profiles=[profile],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, TOOL_NAME_GET_PROJECT_PROFILE)
        result = await handler({"gitlab_project_id": 101})

        assert result.isError is False
        payload = _structured_payload(result)
        assert payload == profile.model_dump(mode="json")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Read tools: out-of-scope (Requirement 10.7 / Property 17)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_name",
    [
        TOOL_NAME_GET_PROJECT_PURPOSE,
        TOOL_NAME_GET_PROJECT_IO,
        TOOL_NAME_GET_PROJECT_DEPENDENCIES,
        TOOL_NAME_GET_PROJECT_PROFILE,
    ],
)
async def test_read_tool_returns_canonical_not_in_scope_error(
    tmp_path: Path, tool_name: str
) -> None:
    """Every read tool surfaces out-of-scope IDs with the canonical message.

    Implements Requirement 10.7. Targets Property 17.
    """

    store = KnowledgeStore.open(tmp_path / f"oos_{tool_name}.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[_enumerated(101, "group/alpha")],
            profiles=[_build_profile(gitlab_project_id=101, full_path="group/alpha")],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, tool_name)
        result = await handler({"gitlab_project_id": 999})

        assert result.isError is True
        assert _text_of(result) == "project 999 is not in scope"
    finally:
        store.close()


@pytest.mark.parametrize(
    "tool_name",
    [
        TOOL_NAME_GET_PROJECT_PURPOSE,
        TOOL_NAME_GET_PROJECT_IO,
        TOOL_NAME_GET_PROJECT_DEPENDENCIES,
        TOOL_NAME_GET_PROJECT_PROFILE,
    ],
)
async def test_read_tool_returns_not_yet_analyzed_when_in_scope_without_profile(
    tmp_path: Path, tool_name: str
) -> None:
    """Read tools surface a non-error message when the project lacks a profile.

    Mirrors Requirement 14.3 (the "in scope but not yet analyzed"
    state) on the MCP side.
    """

    store = KnowledgeStore.open(tmp_path / f"nya_{tool_name}.db")
    try:
        catalog = ProjectCatalog(store)
        # Catalog row exists but no profile is written for project 101.
        snapshot_id = store.begin_snapshot(SnapshotTrigger.FULL)
        catalog.populate_in_scope(snapshot_id, [_enumerated(101, "group/alpha")])
        store.commit_snapshot(snapshot_id)
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, tool_name)
        result = await handler({"gitlab_project_id": 101})

        assert result.isError is False
        assert "101" in _text_of(result)
        assert "not yet been analyzed" in _text_of(result)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# list_purpose_conflicts (Requirement 10.4)
# ---------------------------------------------------------------------------


async def test_list_purpose_conflicts_returns_conflict_pairs(tmp_path: Path) -> None:
    """``list_purpose_conflicts`` returns the symmetric closure of conflicts.

    Implements Requirement 10.4.
    """

    store = KnowledgeStore.open(tmp_path / "conflicts.db")
    try:
        catalog = ProjectCatalog(store)
        # Two profiles with substantially-the-same primary
        # responsibility; the real classifier must find them.
        profile_a = _build_profile(
            gitlab_project_id=101,
            full_path="group/alpha",
            purpose_summary=(
                "owns user authentication and session management for the platform"
            ),
        )
        profile_b = _build_profile(
            gitlab_project_id=202,
            full_path="group/bravo",
            purpose_summary=(
                "owns user authentication and session management for all clients"
            ),
        )
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[
                _enumerated(101, "group/alpha"),
                _enumerated(202, "group/bravo"),
            ],
            profiles=[profile_a, profile_b],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, TOOL_NAME_LIST_PURPOSE_CONFLICTS)
        result = await handler({})

        assert result.isError is False
        payload = _structured_payload(result)
        # The real classifier produces at least one ConflictPair for
        # this near-identical pair; assert the structural shape and
        # the canonical id ordering.
        assert isinstance(payload["conflicts"], list)
        assert len(payload["conflicts"]) == 1
        entry = payload["conflicts"][0]
        assert entry["project_id_a"] == 101
        assert entry["project_id_b"] == 202
        assert entry["justification"]
    finally:
        store.close()


async def test_list_purpose_conflicts_returns_empty_list_when_none_found(
    tmp_path: Path,
) -> None:
    """``list_purpose_conflicts`` returns ``{"conflicts": []}`` when none."""

    store = KnowledgeStore.open(tmp_path / "no_conflicts.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_catalog_and_profiles(
            store,
            catalog,
            enumerated=[_enumerated(101, "group/alpha")],
            profiles=[
                _build_profile(
                    gitlab_project_id=101,
                    full_path="group/alpha",
                    purpose_summary="renders billing PDFs for accounting",
                )
            ],
        )
        server = _build_server(store=store, catalog=catalog)

        handler = _resolve_handler(server, TOOL_NAME_LIST_PURPOSE_CONFLICTS)
        result = await handler({})

        assert result.isError is False
        assert _structured_payload(result) == {"conflicts": []}
    finally:
        store.close()


def test_find_all_conflicts_directly_for_baseline() -> None:
    """Sanity-check: the production classifier produces a conflict for our fixture.

    Failing this test would mean the conflict-detector heuristic
    changed and the ``test_list_purpose_conflicts_returns_conflict_pairs``
    expectations need to be updated. Keeping this baseline here makes
    that linkage explicit.
    """

    profile_a = _build_profile(
        gitlab_project_id=101,
        full_path="group/alpha",
        purpose_summary=(
            "owns user authentication and session management for the platform"
        ),
    )
    profile_b = _build_profile(
        gitlab_project_id=202,
        full_path="group/bravo",
        purpose_summary=(
            "owns user authentication and session management for all clients"
        ),
    )
    pairs: Sequence[ConflictPair] = find_all_conflicts([profile_a, profile_b])
    assert len(pairs) == 1


# ---------------------------------------------------------------------------
# Refresh tools (Requirements 8.1, 8.2, 8.6, 10.7)
# ---------------------------------------------------------------------------


async def test_refresh_all_projects_dispatches_to_coordinator(tmp_path: Path) -> None:
    """``refresh_all_projects`` calls ``Ingestion_Coordinator.start_full_refresh``.

    Implements Requirement 8.1.
    """

    store = KnowledgeStore.open(tmp_path / "refresh_all.db")
    try:
        coordinator = _FakeCoordinator()
        server = _build_server(
            store=store,
            catalog=ProjectCatalog(store),
            coordinator=coordinator,
        )

        handler = _resolve_handler(server, TOOL_NAME_REFRESH_ALL_PROJECTS)
        result = await handler({})

        assert coordinator.full_refresh_calls == 1
        assert result.isError is False
        payload = _structured_payload(result)
        assert payload["status"] == "completed"
        assert payload["trigger"] == "full"
    finally:
        store.close()


async def test_refresh_all_projects_surfaces_ingestion_in_progress(
    tmp_path: Path,
) -> None:
    """``refresh_all_projects`` surfaces ``IngestionInProgressError``.

    Implements Requirement 8.6.
    """

    store = KnowledgeStore.open(tmp_path / "refresh_all_busy.db")
    try:
        coordinator = _FakeCoordinator()
        coordinator.full_refresh_raises = IngestionInProgressError()
        server = _build_server(
            store=store,
            catalog=ProjectCatalog(store),
            coordinator=coordinator,
        )

        handler = _resolve_handler(server, TOOL_NAME_REFRESH_ALL_PROJECTS)
        result = await handler({})

        assert coordinator.full_refresh_calls == 1
        assert result.isError is True
        # The wording is fixed by the design's error-mapping table for
        # Requirement 8.6.
        assert _text_of(result) == "Ingestion_Job is already in progress"
    finally:
        store.close()


async def test_refresh_project_dispatches_to_coordinator(tmp_path: Path) -> None:
    """``refresh_project`` calls ``start_single_project_refresh`` with the id.

    Implements Requirement 8.2.
    """

    store = KnowledgeStore.open(tmp_path / "refresh_one.db")
    try:
        coordinator = _FakeCoordinator()
        server = _build_server(
            store=store,
            catalog=ProjectCatalog(store),
            coordinator=coordinator,
        )

        handler = _resolve_handler(server, TOOL_NAME_REFRESH_PROJECT)
        result = await handler({"gitlab_project_id": 101})

        assert coordinator.single_refresh_calls == [101]
        assert result.isError is False
        payload = _structured_payload(result)
        assert payload["status"] == "completed"
        assert payload["trigger"] == "single_project"
        assert payload["gitlab_project_id"] == 101
    finally:
        store.close()


async def test_refresh_project_surfaces_canonical_not_in_scope(tmp_path: Path) -> None:
    """``refresh_project`` surfaces ``ProjectNotInScopeError`` with canonical wording.

    Implements Requirement 10.7. Targets Property 17.
    """

    store = KnowledgeStore.open(tmp_path / "refresh_one_oos.db")
    try:
        coordinator = _FakeCoordinator()
        coordinator.single_refresh_raises = ProjectNotInScopeError(999)
        server = _build_server(
            store=store,
            catalog=ProjectCatalog(store),
            coordinator=coordinator,
        )

        handler = _resolve_handler(server, TOOL_NAME_REFRESH_PROJECT)
        result = await handler({"gitlab_project_id": 999})

        assert coordinator.single_refresh_calls == [999]
        assert result.isError is True
        assert _text_of(result) == "project 999 is not in scope"
    finally:
        store.close()


async def test_refresh_project_surfaces_ingestion_in_progress(tmp_path: Path) -> None:
    """``refresh_project`` surfaces ``IngestionInProgressError``.

    Implements Requirement 8.6.
    """

    store = KnowledgeStore.open(tmp_path / "refresh_one_busy.db")
    try:
        coordinator = _FakeCoordinator()
        coordinator.single_refresh_raises = IngestionInProgressError()
        server = _build_server(
            store=store,
            catalog=ProjectCatalog(store),
            coordinator=coordinator,
        )

        handler = _resolve_handler(server, TOOL_NAME_REFRESH_PROJECT)
        result = await handler({"gitlab_project_id": 101})

        assert coordinator.single_refresh_calls == [101]
        assert result.isError is True
        assert _text_of(result) == "Ingestion_Job is already in progress"
    finally:
        store.close()

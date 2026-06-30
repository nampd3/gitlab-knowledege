# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 17: For all MCP tools that accept a gitlab_project_id argument and any gitlab_project_id value not present in the current Project_Catalog, tools/call SHALL return a tool result with isError: true whose message states that the project is not in scope.
"""Property 17 - project-id-typed tools reject out-of-scope IDs.

**Validates Requirement 10.7** (Property 17 in the design).

For every MCP tool that accepts a ``gitlab_project_id`` argument
(``get_project_purpose``, ``get_project_io``,
``get_project_dependencies``, ``get_project_profile``, and
``refresh_project``) and every ``gitlab_project_id`` value *not*
present in the current ``Project_Catalog``, ``tools/call`` SHALL
return a :class:`mcp.types.CallToolResult` with ``isError: true``
whose human-readable message states that the project is not in scope
and includes the offending ID verbatim.

The test drives the property end-to-end through
:meth:`MCPServer._dispatch_tool_call` (the wire-shape entry point for
``tools/call``) so the result observed here is exactly the result an
MCP client would see. The dispatch path is invoked via a real
:class:`mcp.types.CallToolRequest` to keep the assertion grounded in
the SDK's contract rather than in any private handler-table detail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import mcp.types as mcp_types
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.conflict_detector import classify_pair, find_all_conflicts
from project_knowledge_mcp.errors import ProjectNotInScopeError
from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.mcp_server import (
    TOOL_NAME_GET_PROJECT_DEPENDENCIES,
    TOOL_NAME_GET_PROJECT_IO,
    TOOL_NAME_GET_PROJECT_PROFILE,
    TOOL_NAME_GET_PROJECT_PURPOSE,
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


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Test fixture data
# ---------------------------------------------------------------------------

# The five MCP tools that accept a ``gitlab_project_id`` argument.
# Property 17 ranges over exactly this set; ``list_projects``,
# ``list_purpose_conflicts``, and ``refresh_all_projects`` take no
# arguments and are therefore out of scope for this property.
_PROJECT_ID_TYPED_TOOL_NAMES: tuple[str, ...] = (
    TOOL_NAME_GET_PROJECT_PURPOSE,
    TOOL_NAME_GET_PROJECT_IO,
    TOOL_NAME_GET_PROJECT_DEPENDENCIES,
    TOOL_NAME_GET_PROJECT_PROFILE,
    TOOL_NAME_REFRESH_PROJECT,
)

# A fixed timestamp keeps generated profiles trivially valid without
# expanding the search space along an axis Property 17 does not care about.
_PRODUCED_AT = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)

# A 40-char SHA-1 lookalike satisfies the non-empty
# ``analysis_branch_commit_sha`` invariant on ``ProjectProfile`` and
# ``EnumeratedProject``.
_COMMIT_SHA = "deadbeef" * 5


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    """Minimal stand-in for :class:`IngestionCoordinator`.

    Property 17 only exercises the out-of-scope rejection path, which
    short-circuits before any coordinator method is invoked for the
    four read tools. The ``refresh_project`` handler does call into
    the coordinator: when the catalog already rejects the ID, the
    real coordinator would raise :class:`ProjectNotInScopeError`
    after re-checking the parent snapshot's catalog. We mirror that
    behaviour here so the test stays faithful to production while
    avoiding the cost of running a real ingestion job.
    """

    def __init__(self, in_scope_ids: frozenset[int]) -> None:
        self._in_scope_ids = in_scope_ids

    def start_full_refresh(self) -> None:  # pragma: no cover - unused here
        raise AssertionError("start_full_refresh must not be called")

    def start_single_project_refresh(self, gitlab_project_id: int) -> None:
        # The MCP-layer catalog check should have already rejected
        # any out-of-scope ID before this method is reached. If the
        # MCP layer were to forward an out-of-scope ID anyway, the
        # real coordinator would raise ``ProjectNotInScopeError``;
        # we mirror that here so the wire shape stays identical.
        if gitlab_project_id not in self._in_scope_ids:
            raise ProjectNotInScopeError(gitlab_project_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enumerated(gitlab_project_id: int, full_path: str) -> EnumeratedProject:
    """Build a minimal valid :class:`EnumeratedProject` row."""
    return EnumeratedProject(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch_name="uat",
        analysis_branch_commit_sha=_COMMIT_SHA,
        branch_missing=False,
    )


def _build_profile(gitlab_project_id: int, full_path: str) -> ProjectProfile:
    """Build a minimal valid :class:`ProjectProfile` row."""
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha=_COMMIT_SHA,
        produced_at=_PRODUCED_AT,
        purpose_summary="owns a stable, well-defined responsibility",
        purpose_summary_reason=None,
    )


def _populate_catalog(
    store: KnowledgeStore,
    catalog: ProjectCatalog,
    in_scope_ids: list[int],
) -> None:
    """Commit a snapshot containing exactly ``in_scope_ids`` in the catalog.

    Each in-scope ID also receives a persisted :class:`ProjectProfile`
    so the read tools' "in-scope but not yet analyzed" branch
    (Requirement 14.3) is *not* what would be triggered if the test
    accidentally passed an in-scope ID. The full-refresh-shaped
    write/commit sequence mirrors the production
    :class:`IngestionCoordinator` order.
    """
    snapshot_id = store.begin_snapshot(SnapshotTrigger.FULL)
    enumerated = [_enumerated(pid, f"group/p{pid}") for pid in in_scope_ids]
    catalog.populate_in_scope(snapshot_id, enumerated)
    for pid in in_scope_ids:
        store.write_profile(
            snapshot_id,
            _build_profile(pid, f"group/p{pid}"),
            produced_at=_PRODUCED_AT,
            commit_sha=_COMMIT_SHA,
        )
    store.commit_snapshot(snapshot_id)


def _build_server(
    *, store: KnowledgeStore, catalog: ProjectCatalog, in_scope_ids: frozenset[int]
) -> MCPServer:
    """Wire an :class:`MCPServer` over a real store/catalog and the fake coordinator.

    ``classify_pair`` and ``find_all_conflicts`` are the production
    pure functions; Property 17 does not exercise them but the
    constructor expects callable collaborators of those signatures.
    """
    coordinator = _FakeCoordinator(in_scope_ids=in_scope_ids)
    return MCPServer(
        store=store,
        catalog=catalog,
        coordinator=cast("IngestionCoordinator", coordinator),
        classify_pair=classify_pair,
        find_all_conflicts=find_all_conflicts,
        version="0.0.0-test",
    )


async def _call_tool(
    server: MCPServer, tool_name: str, gitlab_project_id: int
) -> mcp_types.CallToolResult:
    """Drive a ``tools/call`` request through the real dispatch entry point.

    Constructs a :class:`mcp.types.CallToolRequest` (the same payload
    type the SDK delivers to :meth:`MCPServer._dispatch_tool_call`)
    and unwraps the :class:`mcp.types.ServerResult` envelope returned
    by the dispatcher. Returning the inner :class:`CallToolResult`
    keeps the assertion sites focused on the wire shape Property 17
    governs.
    """
    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(
            name=tool_name,
            arguments={"gitlab_project_id": gitlab_project_id},
        ),
    )
    server_result = await server._dispatch_tool_call(request)
    inner = server_result.root
    assert isinstance(inner, mcp_types.CallToolResult), (
        f"dispatcher returned non-CallToolResult envelope for {tool_name!r}: "
        f"{type(inner).__name__}"
    )
    return inner


def _text_of(result: mcp_types.CallToolResult) -> str:
    """Return the text of the single ``TextContent`` block in ``result``."""
    assert len(result.content) == 1, (
        f"expected exactly one content block, got {len(result.content)}"
    )
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent), (
        f"expected TextContent, got {type(block).__name__}"
    )
    return block.text


# ---------------------------------------------------------------------------
# Generator: an in-scope catalog plus an out-of-scope ID and a tool name
# ---------------------------------------------------------------------------

# GitLab project IDs are positive integers. We bound the range
# generously enough to admit plenty of non-overlapping picks for
# in-scope vs out-of-scope while keeping examples easy to print.
_PROJECT_ID = st.integers(min_value=1, max_value=1_000_000)


@st.composite
def _scenarios(
    draw: st.DrawFn,
) -> tuple[list[int], int, str]:
    """Generate ``(in_scope_ids, out_of_scope_id, tool_name)``.

    * ``in_scope_ids``: a (possibly empty) list of *unique* project
      IDs to populate the current ``Project_Catalog`` with.
    * ``out_of_scope_id``: a positive integer guaranteed not to appear
      in ``in_scope_ids`` - this is the value the property's
      antecedent ranges over.
    * ``tool_name``: one of the five project-id-typed tool names; the
      property quantifies over every such tool.

    The empty-catalog case is intentionally part of the search space:
    Requirement 10.7 holds whether or not any project is in scope, and
    the ``current_snapshot.snapshot_id is None`` branch
    (no-Ingestion_Job-yet) must surface the same canonical message.
    """
    in_scope_ids = draw(
        st.lists(_PROJECT_ID, min_size=0, max_size=8, unique=True)
    )
    in_scope_set = set(in_scope_ids)
    # Draw an out-of-scope ID and reject collisions; the rejection
    # rate is bounded by the in-scope set size (<= 8) so termination
    # is guaranteed within a single ``filter`` step in practice.
    out_of_scope_id = draw(_PROJECT_ID.filter(lambda pid: pid not in in_scope_set))
    tool_name = draw(st.sampled_from(_PROJECT_ID_TYPED_TOOL_NAMES))
    return in_scope_ids, out_of_scope_id, tool_name


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(scenario=_scenarios())
@settings(max_examples=100)
async def test_project_id_typed_tools_reject_out_of_scope_ids(
    tmp_path_factory: pytest.TempPathFactory,
    scenario: tuple[list[int], int, str],
) -> None:
    """Property 17: every project-id-typed tool rejects out-of-scope IDs."""
    in_scope_ids, out_of_scope_id, tool_name = scenario

    # Each Hypothesis example needs a fresh on-disk SQLite store so
    # snapshot state from a previous example cannot leak into this
    # one. ``tmp_path_factory`` is the supported way to obtain unique
    # paths inside a Hypothesis-driven test.
    db_path: Path = tmp_path_factory.mktemp("prop17") / "store.db"

    store = KnowledgeStore.open(db_path)
    try:
        catalog = ProjectCatalog(store)
        if in_scope_ids:
            # Populate a committed snapshot whose catalog contains
            # every ``in_scope_ids`` value (and no others).
            _populate_catalog(store, catalog, in_scope_ids)

        # Sanity-check the generator: the ID under test must not be
        # in scope, otherwise the property's antecedent isn't met
        # and the test would be vacuously true.
        assert not catalog.is_in_scope(out_of_scope_id)

        server = _build_server(
            store=store,
            catalog=catalog,
            in_scope_ids=frozenset(in_scope_ids),
        )

        result = await _call_tool(server, tool_name, out_of_scope_id)

        # (1) The result MUST flag itself as an error.
        assert result.isError is True, (
            f"tool {tool_name!r} returned isError=False for out-of-scope "
            f"id {out_of_scope_id}; result={result!r}"
        )

        # (2) The message MUST state that the project is not in scope.
        # The design fixes the canonical wording at
        # ``"project {gitlab_project_id} is not in scope"`` (see the
        # MCP error mapping table in design.md and the helper
        # ``_not_in_scope_message`` in mcp_server.py).
        message = _text_of(result)
        assert message == f"project {out_of_scope_id} is not in scope", (
            f"tool {tool_name!r} returned non-canonical not-in-scope "
            f"message for id {out_of_scope_id}: {message!r}"
        )
    finally:
        store.close()

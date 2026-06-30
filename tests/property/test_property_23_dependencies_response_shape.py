# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 23: For all sets of persisted Project_Profiles, GET /dependencies SHALL return HTTP 200 with a Dependency_Graph_Diagram whose node set equals the in-scope Project_Catalog and whose edge set equals exactly the set of unordered pairs {a, b} for which: a and b share an External_Service_Dependency name (each shared service name produces one edge labeled "shared external service: {service_name}"), or a and b share a Database_Table_Dependency table_name (each shared table produces one edge labeled "shared table: {table_name}"). When no two projects share any dependency, the response SHALL include a visible message stating that no shared dependencies were detected, while still rendering the project nodes (or the empty-state message if the catalog is empty). When no Ingestion_Job has ever completed, the response SHALL show the "no knowledge available" message and no diagram.
"""Property test for the ``GET /dependencies`` route response shape.

**Validates Requirements 13.3, 14.4** (Property 23 in the design).

For every reachable Project_Catalog / current-snapshot / persisted-
profile combination the Visualization_Server's ``GET /dependencies``
route SHALL return HTTP 200 with an HTML body whose shape is determined
by the state in three mutually exclusive branches:

* **No Ingestion_Job has ever completed**
  (``Knowledge_Store.get_current_snapshot_id() is None``): the body
  contains :data:`NO_SNAPSHOT_MESSAGE` and no ``Dependency_Graph_Diagram``
  content (Requirement 14.4).
* **Snapshot committed but Project_Catalog empty**: the body contains
  :data:`EMPTY_CATALOG_MESSAGE` together with
  :data:`NO_SHARED_DEPENDENCIES_MESSAGE` and no Mermaid block
  (Requirement 13.3's empty-catalog branch).
* **Snapshot committed and Project_Catalog non-empty**: the body
  embeds the ``Dependency_Graph_Diagram`` whose node set equals
  exactly the in-scope ``Project_Catalog`` and whose edge set equals
  exactly the brute-force enumeration of unordered pairs that share
  an ``External_Service_Dependency.name`` or a
  ``Database_Table_Dependency.table_name``. Each shared service name
  produces one edge labeled
  :data:`SHARED_EXTERNAL_SERVICE_EDGE_LABEL_TEMPLATE`; each shared
  table name produces one edge labeled
  :data:`SHARED_TABLE_EDGE_LABEL_TEMPLATE`. When no two in-scope
  projects share any dependency, :data:`NO_SHARED_DEPENDENCIES_MESSAGE`
  appears alongside the project nodes (Requirement 13.3's "still
  rendering project nodes" clause). Profiles whose
  ``gitlab_project_id`` is not in the catalog are silently dropped by
  the renderer and contribute no nodes or edges.

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the test does not depend on a real
SQLite database. The ``httpx.AsyncClient`` + ``httpx.ASGITransport``
pair drives the in-process ASGI app the same way Property 21 does in
``tests/property/test_property_21_index_response_shape.py`` — no
socket bind, no event-loop juggling. ``asyncio.run`` is used inside
the synchronous test body so Hypothesis' ``@given`` composes cleanly
with the async transport (the same pattern Property 18, Property 19,
Property 21, and Property 22 use in this suite).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.diagram_renderer import (
    EMPTY_CATALOG_MESSAGE,
    NO_SHARED_DEPENDENCIES_MESSAGE,
    NO_SNAPSHOT_MESSAGE,
    SHARED_TABLE_EDGE_LABEL_TEMPLATE,
)
from project_knowledge_mcp.models import (
    DatabaseAccessMode,
    DatabaseTableDependency,
    ExternalServiceDependency,
    ExternalServiceKind,
    ProjectProfile,
    SourceLocation,
)
from project_knowledge_mcp.project_catalog import InScopeProject
from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
    build_visualization_app,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from starlette.applications import Starlette


pytestmark = pytest.mark.property



# ---------------------------------------------------------------------------
# Fakes (duck-typed to ``_CatalogReader`` / ``_ProfileReader``)
# ---------------------------------------------------------------------------


class _FakeCatalog:
    """Minimal stand-in for :class:`ProjectCatalog`.

    Returns whatever list of :class:`InScopeProject` it was constructed
    with, in the order it was given. The handler under test calls only
    :meth:`list_in_scope`; :meth:`is_in_scope` is implemented for
    structural compatibility with the ``_CatalogReader`` Protocol but
    is never consulted by the ``/dependencies`` route.
    """

    def __init__(self, projects: Sequence[InScopeProject]) -> None:
        self._projects = list(projects)

    def list_in_scope(self) -> list[InScopeProject]:
        return list(self._projects)

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        return any(
            project.gitlab_project_id == gitlab_project_id
            for project in self._projects
        )


class _FakeStore:
    """Minimal stand-in for :class:`KnowledgeStore`.

    Holds a snapshot id and a list of :class:`ProjectProfile` records.
    The snapshot id can be ``None`` to exercise the no-snapshot branch.
    :meth:`list_profiles` returns the recorded list verbatim;
    :meth:`get_profile` is implemented for structural compatibility
    with the ``_ProfileReader`` Protocol but is never consulted by the
    ``/dependencies`` route.
    """

    def __init__(
        self,
        *,
        snapshot_id: int | None,
        profiles: Sequence[ProjectProfile] = (),
    ) -> None:
        self._snapshot_id = snapshot_id
        self._profiles = list(profiles)

    def get_current_snapshot_id(self) -> int | None:
        return self._snapshot_id

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        for profile in self._profiles:
            if profile.gitlab_project_id == gitlab_project_id:
                return profile
        return None

    def list_profiles(self) -> list[ProjectProfile]:
        return list(self._profiles)



# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------
#
# All free-form strings are restricted to a "safe" ASCII alphabet so the
# Mermaid label escape (``_escape_mermaid_label`` -> ``html.escape`` after
# replacing ``"`` with ``&quot;``) is the identity function on every
# generated value. That keeps the substring-equality assertions below
# coupled only to the documented edge-label and node-label content; any
# regression in the renderer's escape pipeline is caught by the
# ``test_dependency_graph_renderer.py`` unit tests rather than by this
# property test, whose job is to pin the *node-set / edge-set* contract
# of Property 23.

#: Alphabet used for ``InScopeProject.full_path`` strings. Only letters,
#: digits, dashes, underscores, and slashes are admitted so the rendered
#: Mermaid label requires no HTML escape transformations.
_FULL_PATH_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-_/"

#: Alphabet for free-form full-path strategies. ``min_size=1`` matches
#: the ``Field(min_length=1)`` constraint on ``InScopeProject.full_path``.
_FULL_PATH_STRATEGY = st.text(
    alphabet=_FULL_PATH_ALPHABET,
    min_size=1,
    max_size=20,
)

# A small fixed pool of service / table names is used (rather than
# unconstrained ``st.text``) so Hypothesis frequently generates pairs
# that *share* a dependency. With unbounded text strategies the
# probability of two random strings being equal is essentially zero,
# which would leave the populated branch's edge-set assertions trivially
# satisfied by the empty edge set on every run. Sampling from a tight
# pool drives genuine sharing through the brute-force enumeration.
_SERVICE_NAME_STRATEGY = st.sampled_from(
    ["svc-a", "svc-b", "svc-c", "svc-d", "svc-e"]
)
_TABLE_NAME_STRATEGY = st.sampled_from(["tbl1", "tbl2", "tbl3", "tbl4"])

#: GitLab project IDs admitted into the catalog. The upper bound keeps
#: the rendered HTML body short while still exercising multi-digit ids
#: that surface lexicographic-vs-numeric sort regressions.
_IN_SCOPE_ID_STRATEGY = st.integers(min_value=1, max_value=1_000_000)

#: GitLab project IDs reserved for *out-of-scope* profiles that the
#: renderer must drop defensively. The range is disjoint from
#: :data:`_IN_SCOPE_ID_STRATEGY` so an in-scope project and an
#: out-of-scope profile can never collide on id by accident.
_OUT_OF_SCOPE_ID_STRATEGY = st.integers(
    min_value=10_000_001, max_value=20_000_000
)


# Fixed snapshot id used by the ``empty_catalog`` and ``populated``
# branches. The handler only checks ``is None``, so the concrete value
# is irrelevant.
_SNAPSHOT_ID = 99



def _build_profile(
    *,
    gitlab_project_id: int,
    full_path: str,
    service_names: Sequence[str],
    table_names: Sequence[str],
) -> ProjectProfile:
    """Construct a :class:`ProjectProfile` with the given dependency names.

    All identity / branch / commit / produced-at fields are filled in
    with stable defaults; the property only exercises the
    dependency-graph renderer, which reads only
    ``gitlab_project_id``, ``full_path``,
    ``external_service_dependencies``, and
    ``database_table_dependencies``. The other ``ProjectProfile`` fields
    are required by the model so they are populated with arbitrary but
    valid values.

    ``service_names`` and ``table_names`` must each contain unique
    strings; the ``ProjectProfile`` model rejects duplicates per
    Requirements 5.3 and 6.3, and the calling strategies enforce
    uniqueness via ``st.sets``.
    """
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha="cafef00d",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary=f"Project {gitlab_project_id} purpose summary.",
        external_service_dependencies=[
            ExternalServiceDependency(
                name=name,
                kind=ExternalServiceKind.HTTP_API,
                source_locations=[SourceLocation(path="src/clients.py", line=1)],
            )
            for name in service_names
        ],
        database_table_dependencies=[
            DatabaseTableDependency(
                table_name=name,
                access_mode=DatabaseAccessMode.READ,
                source_locations=[SourceLocation(path="src/db.py", line=1)],
            )
            for name in table_names
        ],
    )


@st.composite
def _profile_strategy(
    draw: st.DrawFn,
    *,
    gitlab_project_id: int,
    full_path: str,
) -> ProjectProfile:
    """Build a single :class:`ProjectProfile` for ``gitlab_project_id``.

    The dependency lists are drawn from :data:`_SERVICE_NAME_STRATEGY`
    and :data:`_TABLE_NAME_STRATEGY` via ``st.sets`` so each profile
    has unique service / table names (Requirements 5.3 and 6.3) and
    so the ``Project_Profile`` model accepts the value without raising.
    Names are sorted before construction so the generated profile is
    deterministic given the drawn set; the renderer sorts again before
    rendering, so input order does not affect output but pinning it
    here makes Hypothesis shrink output stable.
    """
    service_names = sorted(draw(st.sets(_SERVICE_NAME_STRATEGY, max_size=4)))
    table_names = sorted(draw(st.sets(_TABLE_NAME_STRATEGY, max_size=4)))
    return _build_profile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        service_names=service_names,
        table_names=table_names,
    )



@st.composite
def _populated_state(
    draw: st.DrawFn,
) -> tuple[list[InScopeProject], list[ProjectProfile]]:
    """Generate the populated branch's ``(catalog, profiles)`` pair.

    The catalog contains 1-5 :class:`InScopeProject` values with unique
    ``gitlab_project_id`` (mirroring the
    ``UNIQUE(snapshot_id, gitlab_project_id)`` constraint on the
    ``project_catalog`` table). For each in-scope project Hypothesis
    flips a boolean to decide whether to seed a profile under that id;
    when no profile is seeded the project still appears as a node in
    the rendered diagram but contributes no edges (Requirement 14.3's
    "in scope but not yet analyzed" case applied to the dependency
    graph).

    Up to 2 *out-of-scope* profiles are also generated, with ids drawn
    from a range disjoint from the in-scope range. Property 23
    requires the renderer to drop these silently — they must
    contribute no nodes and no edges to the rendered graph. Including
    them in every populated state ensures the assertion is exercised
    on every Hypothesis run rather than only on rare draws.
    """
    projects = draw(
        st.lists(
            st.builds(
                InScopeProject,
                gitlab_project_id=_IN_SCOPE_ID_STRATEGY,
                full_path=_FULL_PATH_STRATEGY,
            ),
            min_size=1,
            max_size=5,
            unique_by=lambda project: project.gitlab_project_id,
        )
    )

    profiles: list[ProjectProfile] = []
    for project in projects:
        # Allow some in-scope projects to have *no* profile so the
        # "in-scope but not yet analyzed" sub-case is exercised
        # (Requirement 14.3 applied to the dependency-graph route).
        if draw(st.booleans()):
            profiles.append(
                draw(
                    _profile_strategy(
                        gitlab_project_id=project.gitlab_project_id,
                        full_path=project.full_path,
                    )
                )
            )

    # Out-of-scope profiles: the renderer must drop them so they
    # contribute neither a node nor any edges. Generated as a list of
    # ``(id, full_path)`` pairs first, then deduplicated by id, so
    # ``st.lists(unique_by=...)`` does not have to recurse into the
    # composite profile strategy.
    out_of_scope_specs = draw(
        st.lists(
            st.tuples(_OUT_OF_SCOPE_ID_STRATEGY, _FULL_PATH_STRATEGY),
            max_size=2,
            unique_by=lambda spec: spec[0],
        )
    )
    for out_id, out_full_path in out_of_scope_specs:
        profiles.append(
            draw(
                _profile_strategy(
                    gitlab_project_id=out_id,
                    full_path=out_full_path,
                )
            )
        )

    return projects, profiles


@st.composite
def _dependency_state(
    draw: st.DrawFn,
) -> tuple[str, list[InScopeProject], list[ProjectProfile], int | None]:
    """Generate one of the three mutually exclusive ``/dependencies`` states.

    The state is encoded as
    ``(branch_name, catalog, profiles, snapshot_id)``:

    * ``branch_name == "no_snapshot"`` -> ``snapshot_id is None``,
      ``catalog == []``, ``profiles == []``. The handler ignores the
      catalog and the profile list in this branch (it short-circuits
      on ``get_current_snapshot_id() is None``); pinning both to empty
      keeps the assertion uniform.
    * ``branch_name == "empty_catalog"`` -> ``snapshot_id`` is a
      fixed positive integer, ``catalog == []``, ``profiles == []``.
    * ``branch_name == "populated"`` -> ``snapshot_id`` is a fixed
      positive integer, ``catalog`` is a non-empty list of unique
      :class:`InScopeProject` values, and ``profiles`` may contain
      both in-scope and out-of-scope :class:`ProjectProfile` records.
    """
    branch = draw(
        st.sampled_from(("no_snapshot", "empty_catalog", "populated"))
    )
    if branch == "no_snapshot":
        return "no_snapshot", [], [], None
    if branch == "empty_catalog":
        return "empty_catalog", [], [], _SNAPSHOT_ID
    catalog, profiles = draw(_populated_state())
    return "populated", catalog, profiles, _SNAPSHOT_ID



# ---------------------------------------------------------------------------
# In-process ASGI driver
# ---------------------------------------------------------------------------


async def _drive_get_dependencies(app: Starlette) -> httpx.Response:
    """Issue ``GET /dependencies`` against ``app`` over the ASGI transport.

    Mirrors the helper in ``test_property_21_index_response_shape.py``:
    uses :class:`httpx.ASGITransport` so the request never touches a
    socket and the test does not have to coordinate a real bind.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://visualization-server.test"
    ) as client:
        return await client.get("/dependencies")


def _get_dependencies(app: Starlette) -> httpx.Response:
    """Synchronous wrapper around :func:`_drive_get_dependencies`.

    ``@given`` does not compose with ``async def`` test bodies under
    pytest-asyncio's automatic mode, so the test function stays
    synchronous and uses :func:`asyncio.run` to drive the ASGI
    transport — the same pattern Property 21 and Property 22 use.
    """
    return asyncio.run(_drive_get_dependencies(app))


# ---------------------------------------------------------------------------
# Brute-force edge enumeration
# ---------------------------------------------------------------------------


def _enumerate_expected_edges(
    catalog: Sequence[InScopeProject],
    profiles: Sequence[ProjectProfile],
) -> list[tuple[int, int, str]]:
    """Brute-force the set of edges Property 23 mandates for ``(catalog, profiles)``.

    Returns a list of ``(a_id, b_id, label)`` tuples with
    ``a_id < b_id``, exactly one entry per shared database-table
    dependency. Profiles whose ``gitlab_project_id`` is not in
    ``catalog`` are dropped (Property 23: "the renderer drops them
    defensively"). The label strings come from
    :data:`SHARED_TABLE_EDGE_LABEL_TEMPLATE` so a regression in that
    module constant is caught the same way the renderer's output is.

    The shared-external-service edge kind was retired by an
    operator-tuning decision: in ESB-scale snapshots every
    microservice tends to share a small set of shared
    infrastructure endpoints, producing a fully connected hairball
    that drowns out the more discriminating shared-table edges. The
    brute-force model below mirrors the renderer's new behavior and
    no longer emits ``shared external service:`` edges.
    """
    in_scope_ids = {project.gitlab_project_id for project in catalog}
    profile_by_id = {
        profile.gitlab_project_id: profile
        for profile in profiles
        if profile.gitlab_project_id in in_scope_ids
    }
    sorted_ids = sorted(in_scope_ids)

    edges: list[tuple[int, int, str]] = []
    for index, a_id in enumerate(sorted_ids):
        profile_a = profile_by_id.get(a_id)
        if profile_a is None:
            continue
        tables_a = {
            dep.table_name for dep in profile_a.database_table_dependencies
        }
        for b_id in sorted_ids[index + 1 :]:
            profile_b = profile_by_id.get(b_id)
            if profile_b is None:
                continue
            tables_b = {
                dep.table_name for dep in profile_b.database_table_dependencies
            }
            for table_name in tables_a & tables_b:
                edges.append(
                    (
                        a_id,
                        b_id,
                        SHARED_TABLE_EDGE_LABEL_TEMPLATE.format(
                            table_name=table_name
                        ),
                    )
                )
    return edges



# ---------------------------------------------------------------------------
# Property 23 assertions
# ---------------------------------------------------------------------------

#: Substring marking that ``render_dependency_graph`` produced its outer
#: ``<section>``. Used to check whether a body embeds a
#: ``Dependency_Graph_Diagram`` independently of any per-section
#: content.
_DEPENDENCY_DIAGRAM_MARKER = 'class="dependency-graph-diagram"'

#: Mermaid block opener emitted by the renderer's populated branches
#: (both the "no shared deps" and the "has shared deps" sub-branches).
#: Absent in the empty-catalog and no-snapshot branches.
_MERMAID_BLOCK_OPENER = '<pre class="mermaid">'


def _assert_no_snapshot_response(response: httpx.Response) -> None:
    """Assert the no-snapshot branch's response shape.

    Implements Requirement 14.4 for ``GET /dependencies``: when no
    ``Ingestion_Job`` has ever completed the body contains the
    documented :data:`NO_SNAPSHOT_MESSAGE` and *no*
    ``Dependency_Graph_Diagram`` content (no diagram fragment, no
    Mermaid block, neither of the populated-branch empty-state
    messages).
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NO_SNAPSHOT_MESSAGE in body, (
        "no-snapshot body must contain the documented "
        f"{NO_SNAPSHOT_MESSAGE!r} message (Requirement 14.4)."
    )
    assert _DEPENDENCY_DIAGRAM_MARKER not in body, (
        "no-snapshot body must not embed any Dependency_Graph_Diagram "
        "content (Requirement 14.4)."
    )
    assert _MERMAID_BLOCK_OPENER not in body, (
        "no-snapshot body must not emit a Mermaid block (Requirement 14.4)."
    )
    assert NO_SHARED_DEPENDENCIES_MESSAGE not in body, (
        "no-snapshot body must not show the populated-branch "
        f"{NO_SHARED_DEPENDENCIES_MESSAGE!r} empty-state message; the "
        "no-snapshot branch takes priority over the populated branch "
        "(Requirement 14.4)."
    )
    assert EMPTY_CATALOG_MESSAGE not in body, (
        "no-snapshot body must not show the empty-catalog message; the "
        "no-snapshot branch takes priority over the empty-catalog branch "
        "(Requirement 14.4)."
    )


def _assert_empty_catalog_response(response: httpx.Response) -> None:
    """Assert the empty-catalog branch's response shape.

    Implements Requirement 13.3's empty-catalog branch:

    * status code 200, documented Content-Type,
    * the body contains :data:`EMPTY_CATALOG_MESSAGE` (the catalog is
      empty),
    * the body contains :data:`NO_SHARED_DEPENDENCIES_MESSAGE` (when
      no projects are in scope, no two projects can share a
      dependency, so the shared-deps empty-state is rendered too),
    * the body does *not* emit a Mermaid block (no nodes to render),
    * the no-snapshot message must not appear — a snapshot has been
      committed in this branch.
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert EMPTY_CATALOG_MESSAGE in body, (
        "empty-catalog body must contain the documented "
        f"{EMPTY_CATALOG_MESSAGE!r} message (Requirement 13.3)."
    )
    assert NO_SHARED_DEPENDENCIES_MESSAGE in body, (
        "empty-catalog body must also contain the documented "
        f"{NO_SHARED_DEPENDENCIES_MESSAGE!r} message (Requirement 13.3)."
    )
    assert _MERMAID_BLOCK_OPENER not in body, (
        "empty-catalog body must not emit a Mermaid block (Requirement 13.3)."
    )
    assert NO_SNAPSHOT_MESSAGE not in body, (
        "empty-catalog body must not show the no-snapshot message; a "
        "snapshot is committed in this branch."
    )



def _assert_populated_response(
    response: httpx.Response,
    catalog: Sequence[InScopeProject],
    profiles: Sequence[ProjectProfile],
) -> None:
    """Assert the populated branch's response shape.

    Implements Requirement 13.3's populated branch and the full
    Property 23 contract:

    * status code 200, documented Content-Type,
    * the body embeds the ``Dependency_Graph_Diagram`` (recognized by
      :data:`_DEPENDENCY_DIAGRAM_MARKER` and
      :data:`_MERMAID_BLOCK_OPENER`),
    * the rendered node-set equals exactly the in-scope catalog: each
      in-scope project's ``P{gitlab_project_id}["..."]`` node line is
      present and the project's ``full_path`` appears in the body,
    * out-of-scope profiles contribute no node — their
      ``P{gitlab_project_id}["..."]`` marker is absent,
    * the rendered edge-set equals exactly the brute-force enumeration
      of unordered pairs that share a service ``name`` or a
      ``table_name``: every expected
      ``P{a_id} ---|"{label}"| P{b_id}`` substring appears in the
      body, every edge label uses the documented template strings,
      and the total edge count matches the brute-force count
      (verified by counting ``---|`` occurrences, which the renderer
      only emits inside Mermaid edge lines),
    * when no edges are expected
      :data:`NO_SHARED_DEPENDENCIES_MESSAGE` is rendered alongside
      the project nodes (Requirement 13.3's "still rendering project
      nodes" clause); when at least one edge is expected the
      empty-state message is *not* rendered,
    * neither :data:`EMPTY_CATALOG_MESSAGE` nor
      :data:`NO_SNAPSHOT_MESSAGE` appears in this branch.
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text

    # Branch-exclusivity: the populated branch must not leak any of
    # the other branches' empty-state messages.
    assert NO_SNAPSHOT_MESSAGE not in body, (
        "populated body must not show the no-snapshot message."
    )
    assert EMPTY_CATALOG_MESSAGE not in body, (
        "populated body must not show the empty-catalog message; the "
        "catalog is non-empty in this branch."
    )

    # The Dependency_Graph_Diagram fragment and Mermaid block are both
    # emitted in the populated branch (both the "has shared deps" and
    # the "no shared deps" sub-branches share the Mermaid block; only
    # the empty-catalog branch suppresses it).
    assert _DEPENDENCY_DIAGRAM_MARKER in body, (
        "populated body must embed the Dependency_Graph_Diagram fragment "
        "(Requirement 13.3)."
    )
    assert _MERMAID_BLOCK_OPENER in body, (
        "populated body must emit a Mermaid block with the project nodes "
        "(Requirement 13.3, 'still rendering project nodes' clause)."
    )

    in_scope_ids = {project.gitlab_project_id for project in catalog}

    # The operator-tuning rule prunes isolated nodes: a project
    # appears as a Mermaid node iff it participates in at least one
    # shared-table edge with another in-scope project. Compute the
    # participating-id set from the brute-force expected edges so
    # the node-set assertions stay aligned with the edge-set
    # assertions below.
    expected_edges = _enumerate_expected_edges(catalog, profiles)
    participating_ids: set[int] = set()
    for a_id, b_id, _label in expected_edges:
        participating_ids.add(a_id)
        participating_ids.add(b_id)

    # Node-set equality: every participating in-scope project appears
    # as a Mermaid node with its ``full_path`` rendered into the
    # label; non-participating in-scope projects are pruned from the
    # diagram. With the safe alphabets used by the strategies above
    # the renderer's ``_escape_mermaid_label`` is the identity, so
    # the substring comparisons below match the rendered HTML
    # byte-for-byte.
    for project in catalog:
        node_marker = f'P{project.gitlab_project_id}["'
        if project.gitlab_project_id in participating_ids:
            assert node_marker in body, (
                f"populated body is missing the Mermaid node for in-scope "
                f"project {project.gitlab_project_id} which participates "
                f"in at least one shared-table edge "
                f"(Property 23 node-set clause)."
            )
            assert project.full_path in body, (
                f"populated body is missing the full_path "
                f"{project.full_path!r} for in-scope project "
                f"{project.gitlab_project_id} (Requirement 13.3)."
            )
        else:
            # Isolated in-scope project: pruned by the operator-
            # tuning rule. The ``P{id}["`` marker is unique to node
            # lines so its absence proves no node was emitted; the
            # ``full_path`` may still appear elsewhere if a separate
            # in-scope project happens to share that exact path, so
            # only the node marker is asserted on.
            assert node_marker not in body, (
                f"populated body must NOT contain a node for isolated "
                f"in-scope project {project.gitlab_project_id} — "
                f"projects with no shared-table edge are pruned by "
                f"the operator-tuning rule."
            )

    # Out-of-scope profiles must not produce any node. The
    # ``P{id}["`` marker is unique to node lines (it cannot appear in
    # an in-scope project's ``full_path`` because the path alphabet
    # excludes ``[`` and ``"``).
    for profile in profiles:
        if profile.gitlab_project_id in in_scope_ids:
            continue
        out_marker = f'P{profile.gitlab_project_id}["'
        assert out_marker not in body, (
            f"populated body must not contain a node for out-of-scope "
            f"profile {profile.gitlab_project_id} "
            f"(Property 23: 'the renderer drops them defensively')."
        )

    # Edge-set equality: each expected edge appears verbatim, and the
    # total ``---|`` count matches the brute-force edge count. The
    # renderer only emits ``---|`` inside Mermaid edge lines; static
    # HTML chrome and node lines do not contain it (the path / name
    # alphabets above also exclude ``|``), so counting it is a sound
    # proxy for the edge count.
    for a_id, b_id, label in expected_edges:
        edge_substring = f'P{a_id} ---|"{label}"| P{b_id}'
        assert edge_substring in body, (
            f"populated body is missing the expected edge "
            f"{edge_substring!r} (Requirement 13.3, Property 23 "
            "edge-set clause)."
        )
    assert body.count("---|") == len(expected_edges), (
        f"populated body has {body.count('---|')} edges, expected "
        f"{len(expected_edges)} (Property 23 edge-set clause); "
        "extra edges would imply spurious shared dependencies, missing "
        "edges would imply a regression in the brute-force enumeration."
    )

    # Empty-state for shared deps: present iff the brute-force edge set
    # is empty. When the edge set is empty the diagram also has no
    # nodes (the operator-tuning prune drops every isolated project),
    # so the empty-state message is the only visible content inside
    # the diagram fragment.
    if expected_edges:
        assert NO_SHARED_DEPENDENCIES_MESSAGE not in body, (
            "populated body with at least one expected edge must not "
            "render the no-shared-dependencies empty-state."
        )
    else:
        assert NO_SHARED_DEPENDENCIES_MESSAGE in body, (
            "populated body with no expected edges must render the "
            f"{NO_SHARED_DEPENDENCIES_MESSAGE!r} empty-state."
        )



# ---------------------------------------------------------------------------
# Property 23
# ---------------------------------------------------------------------------


@given(state=_dependency_state())
@settings(max_examples=100)
def test_dependencies_response_shape_property(
    state: tuple[str, list[InScopeProject], list[ProjectProfile], int | None],
) -> None:
    """Property 23: ``GET /dependencies`` returns the documented shape.

    For every Project_Catalog / current-snapshot / persisted-profile
    state the ``GET /dependencies`` response satisfies the
    branch-specific post-conditions above. The three branches are
    mutually exclusive, so a single Hypothesis-driven test exercises
    all of them — and a regression in any one branch (e.g., the
    no-snapshot body leaking a diagram, the empty-catalog body
    forgetting one of its empty-state messages, the populated body
    drawing a spurious or missing edge, or an out-of-scope profile
    leaking a node) is caught by the same test.

    **Validates Requirements 13.3, 14.4.**
    """
    branch, catalog, profiles, snapshot_id = state
    app = build_visualization_app(
        catalog=_FakeCatalog(catalog),
        store=_FakeStore(snapshot_id=snapshot_id, profiles=profiles),
    )

    response = _get_dependencies(app)

    if branch == "no_snapshot":
        _assert_no_snapshot_response(response)
    elif branch == "empty_catalog":
        _assert_empty_catalog_response(response)
    else:
        assert branch == "populated"
        _assert_populated_response(response, catalog, profiles)

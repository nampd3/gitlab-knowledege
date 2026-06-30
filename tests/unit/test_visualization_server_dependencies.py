"""Unit tests for the ``GET /dependencies`` handler.

Two tests, one per branch of the diagram-route decision tree, drive
the Visualization_Server's ``/dependencies`` route through the
in-process Starlette ASGI transport and assert the response shape
required by Requirements 13.3, 14.1, and 14.4:

* When no ``Ingestion_Job`` has ever completed
  (``Knowledge_Store.get_current_snapshot_id() is None``),
  ``/dependencies`` returns HTTP 200 with the documented "no
  project knowledge available; run an Ingestion_Job" message and
  *no* diagram content (Requirement 14.4 / Property 21 applied to
  the diagram route).
* When a snapshot has been committed, ``/dependencies`` returns
  HTTP 200 with the ``Dependency_Graph_Diagram`` rendered from the
  catalog and the persisted profile list, both read at request
  time (Requirements 13.3 and 14.1).

The handler is wired with two duck-typed collaborators:

* a fake :class:`_CatalogReader` that returns a fixed list of
  :class:`InScopeProject` entries, and
* a fake :class:`_ProfileReader` that returns a fixed snapshot id
  and a fixed list of :class:`ProjectProfile` records.

The previous ``/conflicts`` route and its corresponding tests were
retired by an operator-tuning decision: the conflict graph proved
unreadable at scale and was removed from the visualization
surface. ``Conflict_Detector`` itself remains available for
programmatic callers.

Implements Requirements 13.3, 14.1, 14.4.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest

from project_knowledge_mcp.diagram_renderer import (
    EMPTY_CATALOG_MESSAGE,
    NO_SHARED_DEPENDENCIES_MESSAGE,
    NO_SNAPSHOT_MESSAGE,
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
    from starlette.applications import Starlette


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get(app: Starlette, path: str) -> httpx.Response:
    """Issue ``GET path`` against ``app`` over the in-process ASGI transport.

    Mirrors the helper in ``test_visualization_server_index.py`` and
    ``test_visualization_server_per_project.py``: uses
    :class:`httpx.ASGITransport` so the test does not need a real
    socket bind and so the suite-wide ``filterwarnings = ["error"]``
    rule does not turn the deprecated
    ``starlette.testclient.TestClient`` warning into a collection
    error.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://visualization-server.test"
    ) as client:
        return await client.get(path)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCatalog:
    """Minimal :class:`_CatalogReader` stand-in for the diagram routes.

    Records the number of times :meth:`list_in_scope` is invoked so
    the populated-branch tests can prove that the handler reads from
    the catalog *at request time* (Requirement 14.1's "no in-memory
    caching of profile data" rule), rather than reading once at
    application-build time.
    """

    def __init__(self, projects: Sequence[InScopeProject]) -> None:
        self._projects = list(projects)
        self.list_in_scope_calls = 0

    def list_in_scope(self) -> list[InScopeProject]:
        self.list_in_scope_calls += 1
        return list(self._projects)

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        # The diagram routes never call ``is_in_scope`` — that is the
        # per-project handler's contract — but the method exists so
        # the fake structurally satisfies ``_CatalogReader``.
        return any(
            project.gitlab_project_id == gitlab_project_id
            for project in self._projects
        )


class _FakeStore:
    """Minimal :class:`_ProfileReader` stand-in for the diagram routes.

    Holds a snapshot id and a list of :class:`ProjectProfile` records.
    The snapshot id can be ``None`` to exercise the no-snapshot
    branch. :meth:`list_profiles` returns the recorded list and tracks
    its call count so the populated-branch tests can prove that the
    handler reads from the store at request time.
    """

    def __init__(
        self,
        *,
        snapshot_id: int | None,
        profiles: Sequence[ProjectProfile] = (),
    ) -> None:
        self._snapshot_id = snapshot_id
        self._profiles = list(profiles)
        self.list_profiles_calls = 0

    def get_current_snapshot_id(self) -> int | None:
        return self._snapshot_id

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        for profile in self._profiles:
            if profile.gitlab_project_id == gitlab_project_id:
                return profile
        return None

    def list_profiles(self) -> list[ProjectProfile]:
        self.list_profiles_calls += 1
        return list(self._profiles)


# ---------------------------------------------------------------------------
# Sample profile factory
# ---------------------------------------------------------------------------


def _profile_with_shared_deps(
    *, gitlab_project_id: int, full_path: str
) -> ProjectProfile:
    """Build a :class:`ProjectProfile` with a single external service and a single
    table.

    Two profiles produced from this factory always share the same
    ``payments-api`` external service and the same ``orders`` table,
    so :func:`render_dependency_graph` produces exactly two edges
    between them (one per shared dependency). That gives the populated
    ``/dependencies`` test a deterministic target — both the
    documented edge labels — without the test having to enumerate
    every possible catalog/profile shape.
    """
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha="0123456789abcdef0123456789abcdef01234567",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary=f"Project {gitlab_project_id} processes orders.",
        abstract_inputs=[],
        abstract_outputs=[],
        external_service_dependencies=[
            ExternalServiceDependency(
                name="payments-api",
                kind=ExternalServiceKind.HTTP_API,
                source_locations=[
                    SourceLocation(path="src/orders.py", line=42),
                ],
            ),
        ],
        database_table_dependencies=[
            DatabaseTableDependency(
                table_name="orders",
                access_mode=DatabaseAccessMode.READ_WRITE,
                source_locations=[SourceLocation(path="src/db.py", line=7)],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# /dependencies tests
# ---------------------------------------------------------------------------


async def test_dependencies_returns_no_knowledge_message_when_no_snapshot() -> None:
    """No ``Ingestion_Job`` has ever completed → no-knowledge message branch.

    Verifies Requirement 14.4 for ``GET /dependencies``:

    * status code is 200,
    * Content-Type is the documented ``text/html; charset=utf-8``
      (Requirement 13.5),
    * the body contains :data:`NO_SNAPSHOT_MESSAGE` verbatim,
    * the body contains *no* diagram content — the
      ``Dependency_Graph_Diagram`` outer ``<section>`` and its inline
      Mermaid block are both suppressed (Requirement 14.4 forbids
      diagram-derived content before the first successful
      ``Ingestion_Job``),
    * the handler never reads the catalog or the profile list — the
      no-snapshot branch must not invoke ``list_in_scope`` or
      ``list_profiles``, both because there is nothing useful to
      read and because Requirement 14.1's "no in-memory caching"
      rule is moot when the response is the no-snapshot empty state.
    """
    catalog = _FakeCatalog([])
    store = _FakeStore(snapshot_id=None)
    app = build_visualization_app(catalog=catalog, store=store)

    response = await _get(app, "/dependencies")

    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NO_SNAPSHOT_MESSAGE in body
    # No diagram content: the dependency-graph fragment uses the
    # ``dependency-graph-diagram`` class on its outer <section>.
    assert "dependency-graph-diagram" not in body
    assert '<pre class="mermaid">' not in body
    # Read counters prove the no-snapshot branch short-circuited.
    assert catalog.list_in_scope_calls == 0
    assert store.list_profiles_calls == 0


async def test_dependencies_renders_diagram_when_snapshot_committed() -> None:
    """Snapshot committed → 200 + ``Dependency_Graph_Diagram`` rendered fresh.

    Verifies Requirements 13.3 and 14.1 for ``GET /dependencies``:

    * status code is 200 and the documented Content-Type is set,
    * the body embeds the ``Dependency_Graph_Diagram`` — recognized
      by the ``dependency-graph-diagram`` class on the rendered
      ``<section>`` and the inline Mermaid block,
    * the shared-table edge label documented by Requirement 13.3
      appears in the body for the chosen profile fixture
      (``"shared table: orders"``); the shared-external-service
      edge kind was retired by an operator-tuning decision and
      must not appear,
    * neither empty-state message appears: the catalog is non-empty
      (so :data:`EMPTY_CATALOG_MESSAGE` does not show) and the two
      profiles share dependencies (so
      :data:`NO_SHARED_DEPENDENCIES_MESSAGE` does not show either),
    * the catalog and profile list are read *exactly once per
      request* — the call counters on the fakes prove the route
      reads through to the live store and catalog at request time
      (Requirement 14.1's "no in-memory caching of profile data"
      rule).
    """
    auth = InScopeProject(gitlab_project_id=7, full_path="acme/auth/login")
    payments = InScopeProject(gitlab_project_id=42, full_path="acme/payments/checkout")
    profiles = [
        _profile_with_shared_deps(gitlab_project_id=7, full_path="acme/auth/login"),
        _profile_with_shared_deps(
            gitlab_project_id=42, full_path="acme/payments/checkout"
        ),
    ]
    catalog = _FakeCatalog([auth, payments])
    store = _FakeStore(snapshot_id=99, profiles=profiles)
    app = build_visualization_app(catalog=catalog, store=store)

    response = await _get(app, "/dependencies")

    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NO_SNAPSHOT_MESSAGE not in body
    # Dependency-graph fragment is embedded.
    assert "dependency-graph-diagram" in body
    assert '<pre class="mermaid">' in body
    # The shared-table edge label documented by Requirement 13.3
    # appears in the body. The shared-external-service edge kind
    # was retired by an operator-tuning decision and must NOT
    # appear, even though the two profiles share an external
    # service name.
    assert "shared external service:" not in body
    assert "shared table: orders" in body
    # Catalog is non-empty and at least one shared dependency exists,
    # so neither empty-state message should appear.
    assert EMPTY_CATALOG_MESSAGE not in body
    assert NO_SHARED_DEPENDENCIES_MESSAGE not in body
    # Read at request time, exactly once each (Requirement 14.1).
    assert catalog.list_in_scope_calls == 1
    assert store.list_profiles_calls == 1



"""Unit tests for the ``GET /`` index handler (task 10.5).

Three tests, one per branch of the index handler decision tree, drive the
Visualization_Server's index route through the in-process Starlette
``TestClient`` and assert the response shape required by Requirement 13.1
and Requirement 14.4 (Property 21):

* When no ``Ingestion_Job`` has ever completed
  (``Knowledge_Store.get_current_snapshot_id() is None``), the body
  contains the documented "no project knowledge available; run an
  Ingestion_Job" message and no per-project list or diagram links.
* When a snapshot has been committed but the ``Project_Catalog`` is
  empty, the body contains exactly the message "No Projects are in
  scope" and no per-project list entries.
* When a snapshot has been committed and the catalog is non-empty, the
  body lists exactly one entry per in-scope project ordered by
  ``gitlab_project_id`` ascending; each entry includes the project's
  GitLab id, the GitLab full path, and links to the
  ``Project_Profile_Diagram``, the ``Dependency_Graph_Diagram``, and
  the ``Conflict_Overview_Diagram``.

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_SnapshotReader`` Protocols defined in
``visualization_server.py`` so the tests do not depend on a real SQLite
database. The property test in task 10.13 will exercise the same
contract under arbitrary catalog states.

Implements Requirements 13.1, 14.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from project_knowledge_mcp.diagram_renderer import (
    EMPTY_CATALOG_MESSAGE,
    NO_SNAPSHOT_MESSAGE,
)
from project_knowledge_mcp.project_catalog import InScopeProject
from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
    build_visualization_app,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from starlette.applications import Starlette

    from project_knowledge_mcp.models import ProjectProfile


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get(app: Starlette, path: str) -> httpx.Response:
    """Issue ``GET path`` against ``app`` over the in-process ASGI transport.

    Uses :class:`httpx.ASGITransport` rather than the (deprecated, in
    Starlette 0.49+) ``starlette.testclient.TestClient`` so the suite's
    ``filterwarnings = ["error"]`` configuration does not turn the
    deprecation warning into a collection error. The transport drives
    the ASGI app directly inside the test process; no socket bind, no
    event-loop juggling.

    Returns an :class:`httpx.Response` so callers can assert on
    ``status_code``, ``headers``, and ``text`` with the same idioms used
    by the rest of the test suite.
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
    """Minimal stand-in for :class:`ProjectCatalog`.

    Returns whatever list of :class:`InScopeProject` it was constructed
    with, in the order it was given. The handler under test sorts
    defensively, so passing an unsorted list is fine — and is in fact
    what the populated-branch test exercises to verify the sort.

    Implements both :meth:`list_in_scope` and :meth:`is_in_scope` so the
    fake structurally satisfies ``_CatalogReader`` even though the index
    handler under test only calls :meth:`list_in_scope`. The
    :meth:`is_in_scope` stub is used by sibling per-project handler
    tests (task 10.6); here it is a no-op that returns ``False`` so the
    fake remains a faithful Protocol implementation.
    """

    def __init__(self, projects: Sequence[InScopeProject]) -> None:
        self._projects = list(projects)

    def list_in_scope(self) -> list[InScopeProject]:
        return list(self._projects)

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        del gitlab_project_id  # unused in index-handler tests
        return False


class _FakeStore:
    """Minimal stand-in for :class:`KnowledgeStore`.

    Implements :meth:`get_current_snapshot_id` (consulted by the index
    handler), :meth:`get_profile` (consulted by the per-project
    handler), and :meth:`list_profiles` (consulted by the
    ``/dependencies`` and ``/conflicts`` handlers) so the fake
    structurally satisfies ``_ProfileReader``. The index-handler tests
    in this file never trigger the per-project, dependency, or
    conflicts branches, so :meth:`get_profile` and
    :meth:`list_profiles` are no-ops that return an empty result.
    """

    def __init__(self, snapshot_id: int | None) -> None:
        self._snapshot_id = snapshot_id

    def get_current_snapshot_id(self) -> int | None:
        return self._snapshot_id

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        del gitlab_project_id  # unused in index-handler tests
        return None

    def list_profiles(self) -> list[ProjectProfile]:
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_index_returns_no_knowledge_message_when_no_snapshot_committed() -> None:
    """No ``Ingestion_Job`` has ever completed → no-knowledge message branch.

    Verifies Requirement 14.4 via Property 21's "no Ingestion_Job has
    ever completed" branch:

    * status code is 200,
    * Content-Type is the documented ``text/html; charset=utf-8``
      (Requirement 13.5),
    * the body contains the documented no-knowledge message verbatim,
    * the body contains no per-project list entries (no link to
      ``/projects/{id}``) and no diagram-route links — the
      ``Project_Profile_Diagram``, ``Dependency_Graph_Diagram``, and
      ``Conflict_Overview_Diagram`` content is suppressed entirely
      (Requirement 14.4),
    * the empty-catalog message is *not* shown — the no-snapshot
      branch takes priority.
    """
    app = build_visualization_app(
        catalog=_FakeCatalog([]),
        store=_FakeStore(snapshot_id=None),
    )

    response = await _get(app, "/")

    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NO_SNAPSHOT_MESSAGE in body
    assert EMPTY_CATALOG_MESSAGE not in body
    # No per-project, dependency, or conflict links in the no-snapshot body.
    assert 'href="/projects/' not in body
    assert 'href="/dependencies"' not in body


async def test_index_returns_no_projects_message_when_catalog_is_empty() -> None:
    """Snapshot committed but catalog empty → "No Projects are in scope" branch.

    Verifies Requirement 13.1's "zero in-scope Projects" branch:

    * status code is 200 and the documented Content-Type is set,
    * the body contains exactly "No Projects are in scope",
    * the body has no per-project list entries (Requirement 13.1
      requires the response to "omit per-Project list entries" when the
      catalog is empty),
    * the no-snapshot message is *not* shown — a snapshot has been
      committed, so the index is past the no-knowledge branch.
    """
    app = build_visualization_app(
        catalog=_FakeCatalog([]),
        store=_FakeStore(snapshot_id=42),
    )

    response = await _get(app, "/")

    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert EMPTY_CATALOG_MESSAGE in body
    assert NO_SNAPSHOT_MESSAGE not in body
    assert 'href="/projects/' not in body


async def test_index_lists_in_scope_projects_sorted_with_diagram_links() -> None:
    """Populated catalog → sorted project list with all three diagram links.

    Verifies Requirement 13.1's populated branch:

    * status code is 200 and the documented Content-Type is set,
    * each in-scope project appears in the body with its GitLab id and
      its full path,
    * the projects appear in ``gitlab_project_id`` ascending order even
      though the fake catalog returned them out of order,
    * each entry includes a link to ``/projects/{id}`` (the per-project
      ``Project_Profile_Diagram``), and the body includes links to
      ``/dependencies`` (``Dependency_Graph_Diagram``) and ``/conflicts``
      (``Conflict_Overview_Diagram``) as required by Requirement 13.1
      ("each list entry includes ... a link to the
      Dependency_Graph_Diagram, and a link to the
      Conflict_Overview_Diagram"),
    * neither empty-state message appears.
    """
    # Out-of-order on purpose so the assertion below proves the handler
    # sorts by gitlab_project_id ascending rather than just trusting the
    # catalog.
    projects = [
        InScopeProject(gitlab_project_id=42, full_path="acme/payments/checkout"),
        InScopeProject(gitlab_project_id=7, full_path="acme/auth/login"),
        InScopeProject(gitlab_project_id=15, full_path="acme/notify/email"),
    ]
    app = build_visualization_app(
        catalog=_FakeCatalog(projects),
        store=_FakeStore(snapshot_id=99),
    )

    response = await _get(app, "/")

    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NO_SNAPSHOT_MESSAGE not in body
    assert EMPTY_CATALOG_MESSAGE not in body

    # Each project's id, full path, and per-project link are present.
    for project in projects:
        assert str(project.gitlab_project_id) in body
        assert project.full_path in body
        assert f'href="/projects/{project.gitlab_project_id}"' in body

    # Order check: the per-project link for id=7 must precede the link
    # for id=15, which must precede the link for id=42, regardless of
    # the order the fake catalog returned them in.
    positions = [
        body.index(f'href="/projects/{project_id}"') for project_id in (7, 15, 42)
    ]
    assert positions == sorted(positions)

    # Each list entry includes a Dependency_Graph_Diagram link, so
    # the body contains at least one occurrence per project. ``count``
    # >= len(projects) is sufficient because the design only requires
    # that each entry have that link; it does not forbid additional
    # occurrences.
    assert body.count('href="/dependencies"') >= len(projects)

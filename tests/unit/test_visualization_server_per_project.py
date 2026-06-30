"""Unit tests for the ``GET /projects/{project_id}`` per-project handler (task 10.6).

Three tests, one per branch of the per-project handler decision tree, drive
the Visualization_Server's per-project route through the in-process
Starlette ASGI transport and assert the response shape required by
Requirements 13.2, 13.6, 14.3, and 14.5:

* When the project is in scope and a ``Project_Profile`` is persisted in
  the current snapshot, the handler responds with HTTP 200, the
  documented ``text/html; charset=utf-8`` Content-Type, and an HTML body
  embedding the ``Project_Profile_Diagram`` for the project (Requirements
  13.2, 13.6).
* When the project is in scope but no ``Project_Profile`` is persisted
  in the current snapshot, the handler responds with HTTP 200 and an
  HTML body that contains the documented "Project has not yet been
  analyzed; run an Ingestion_Job" message and *no*
  ``Project_Profile_Diagram`` (Requirement 14.3).
* When the project is not in scope, the handler responds with HTTP 404
  and an HTML body that names the requested ``project_id`` so the
  operator (or a test) can identify which id was rejected (Requirements
  13.6, 14.5).

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the tests do not depend on a real SQLite
database. The property test in task 10.14 will exercise the same
contract under arbitrary catalog and snapshot states.

Implements Requirements 13.2, 13.6, 14.3, 14.5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest

from project_knowledge_mcp.diagram_renderer import (
    NOT_IN_SCOPE_MESSAGE_TEMPLATE,
    NOT_YET_ANALYZED_MESSAGE,
)
from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
    DatabaseAccessMode,
    DatabaseTableDependency,
    ExternalServiceDependency,
    ExternalServiceKind,
    ProjectProfile,
    SourceLocation,
)
from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
    build_visualization_app,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from project_knowledge_mcp.project_catalog import InScopeProject


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get(app: Starlette, path: str) -> httpx.Response:
    """Issue ``GET path`` against ``app`` over the in-process ASGI transport.

    Mirrors the helper in ``test_visualization_server_index.py``: uses
    :class:`httpx.ASGITransport` so the test does not need a real socket
    bind and so the suite-wide ``filterwarnings = ["error"]`` rule does
    not turn the deprecated ``starlette.testclient.TestClient`` warning
    into a collection error.
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
    """Minimal :class:`_CatalogReader` stand-in for the per-project handler.

    The handler only ever calls :meth:`is_in_scope` for this route, so
    :meth:`list_in_scope` returns an empty list — the index handler is
    never exercised in this file. The constructor accepts a set of
    in-scope GitLab project ids so each test can express the in-scope
    universe declaratively.
    """

    def __init__(self, in_scope_ids: set[int]) -> None:
        self._in_scope_ids = set(in_scope_ids)

    def list_in_scope(self) -> list[InScopeProject]:
        # Not consulted by the per-project handler under test, but kept
        # implemented so the fake structurally satisfies ``_CatalogReader``.
        return []

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        return gitlab_project_id in self._in_scope_ids


class _FakeStore:
    """Minimal :class:`_ProfileReader` stand-in for the per-project handler.

    Holds a mapping from ``gitlab_project_id`` to ``ProjectProfile``
    instances so each test can express the persisted-profile universe
    declaratively. :meth:`get_current_snapshot_id` and
    :meth:`list_profiles` are implemented for Protocol completeness
    even though the per-project handler does not consult them
    directly.
    """

    def __init__(
        self,
        *,
        snapshot_id: int | None = 1,
        profiles: dict[int, ProjectProfile] | None = None,
    ) -> None:
        self._snapshot_id = snapshot_id
        self._profiles = dict(profiles) if profiles is not None else {}

    def get_current_snapshot_id(self) -> int | None:
        return self._snapshot_id

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        return self._profiles.get(gitlab_project_id)

    def list_profiles(self) -> list[ProjectProfile]:
        return sorted(
            self._profiles.values(), key=lambda profile: profile.gitlab_project_id
        )


# ---------------------------------------------------------------------------
# Sample profile factory
# ---------------------------------------------------------------------------


def _sample_profile(gitlab_project_id: int) -> ProjectProfile:
    """Build a fully-populated :class:`ProjectProfile` for handler tests.

    The factory deliberately exercises every field the
    ``Project_Profile_Diagram`` renders so the assertions in the
    "profile present" test verify that the diagram fragment was
    embedded in the page (rather than the not-yet-analyzed empty
    state). Values are chosen to be visually distinctive so substring
    assertions on the rendered body are unambiguous.
    """
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=f"acme/team/project-{gitlab_project_id}",
        analysis_branch="uat",
        analysis_branch_commit_sha="abcdef0123456789",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary=f"Project {gitlab_project_id} processes orders.",
        abstract_inputs=[
            AbstractInput(
                category=AbstractInputCategory.HTTP_REQUEST,
                description="POST /orders with order payload",
            ),
        ],
        abstract_outputs=[
            AbstractOutput(
                category=AbstractOutputCategory.MESSAGE_PUBLISHED,
                description="order-confirmed events on bus",
            ),
        ],
        external_service_dependencies=[
            ExternalServiceDependency(
                name="payments-api",
                kind=ExternalServiceKind.HTTP_API,
                source_locations=[SourceLocation(path="src/orders.py", line=42)],
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
# Tests
# ---------------------------------------------------------------------------


async def test_per_project_returns_diagram_when_in_scope_and_profile_persisted() -> None:
    """In-scope id with a persisted profile → 200 + Project_Profile_Diagram.

    Verifies Requirements 13.2 and 13.6:

    * status code is 200,
    * Content-Type is the documented ``text/html; charset=utf-8``
      (Requirement 13.5),
    * the rendered body embeds the ``Project_Profile_Diagram`` for the
      project — recognized by the diagram fragment's distinctive
      ``data-project-id`` attribute on the outer ``<section>`` and the
      diagram-only "Project Profile:" heading,
    * the body does not show the not-yet-analyzed empty-state message
      (the diagram branch must not also render the empty-state
      message),
    * the body does not show the not-in-scope message.
    """
    profile = _sample_profile(123)
    app = build_visualization_app(
        catalog=_FakeCatalog(in_scope_ids={123}),
        store=_FakeStore(profiles={123: profile}),
    )

    response = await _get(app, "/projects/123")

    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    # The Project_Profile_Diagram fragment carries data-project-id="123"
    # on its outer <section>. The full HTML page also carries that id
    # only when the diagram is embedded, so the assertion proves the
    # diagram was rendered into the body.
    assert 'data-project-id="123"' in body
    assert "Project Profile:" in body
    # The full path from the profile is rendered in the diagram header.
    assert profile.full_path in body
    # Empty-state messages from the other two branches must not appear.
    assert NOT_YET_ANALYZED_MESSAGE not in body
    assert "is not in scope" not in body


async def test_per_project_returns_not_yet_analyzed_when_in_scope_without_profile() -> None:
    """In-scope id with no profile persisted → 200 + "not yet analyzed" message.

    Verifies Requirement 14.3:

    * status code is 200 (not 404 — the project is in scope),
    * Content-Type is the documented ``text/html; charset=utf-8``,
    * the body contains the documented
      "Project has not yet been analyzed; run an Ingestion_Job"
      message verbatim,
    * the body does *not* embed a ``Project_Profile_Diagram`` — the
      "not yet analyzed" branch must suppress diagram content
      entirely so the operator sees only the empty-state message.
    """
    app = build_visualization_app(
        catalog=_FakeCatalog(in_scope_ids={42}),
        store=_FakeStore(profiles={}),
    )

    response = await _get(app, "/projects/42")

    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NOT_YET_ANALYZED_MESSAGE in body
    # No diagram content: the diagram template's outer <section> is
    # absent, and the "Project Profile:" heading from the diagram is
    # not rendered.
    assert 'class="project-profile-diagram"' not in body
    assert "Project Profile:" not in body
    assert "is not in scope" not in body


async def test_per_project_returns_404_when_not_in_scope_with_requested_id_in_body() -> None:
    """Out-of-scope digit-only id → 404 + body naming the requested project_id.

    Verifies Requirements 13.6 and 14.5:

    * status code is 404,
    * Content-Type is the documented ``text/html; charset=utf-8``,
    * the body contains the documented "Project {id} is not in scope"
      message with the requested ``project_id`` interpolated,
    * the body does *not* embed a ``Project_Profile_Diagram`` (out-of-
      scope ids must not leak any profile-derived content), and
    * the body does *not* show the in-scope "not yet analyzed"
      message — the out-of-scope branch must not be confused with
      the in-scope-but-not-analyzed branch.
    """
    app = build_visualization_app(
        catalog=_FakeCatalog(in_scope_ids={1, 2, 3}),
        # The store happens to have a profile for id 999 — the handler
        # must not reach for it once is_in_scope returns False, since
        # leaking out-of-scope profile content would violate
        # Requirements 13.6 and 14.5.
        store=_FakeStore(profiles={999: _sample_profile(999)}),
    )

    response = await _get(app, "/projects/999")

    assert response.status_code == 404
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NOT_IN_SCOPE_MESSAGE_TEMPLATE.format(project_id=999) in body
    # The requested id appears in the body so an operator can identify it.
    assert "999" in body
    # No diagram content; in particular the "Project Profile:" heading
    # from a possibly-existing profile must not bleed through.
    assert 'class="project-profile-diagram"' not in body
    assert "Project Profile:" not in body
    assert NOT_YET_ANALYZED_MESSAGE not in body

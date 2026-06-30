# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 22: For all digit-only project_id values, GET /projects/{project_id} SHALL return: HTTP 200 with a Project_Profile_Diagram rendering the project's purpose summary, Abstract_Inputs grouped by category, Abstract_Outputs grouped by category, External_Service_Dependencies labeled by service kind, Database_Table_Dependencies labeled by access mode, and a section-specific empty-state message naming each empty section, when the project is in scope and a profile is persisted, HTTP 200 with a "Project has not yet been analyzed; run an Ingestion_Job" message and no Project_Profile_Diagram, when the project is in scope but no profile is persisted, HTTP 404 with HTML stating the project is not in scope and including the requested project_id value, when the project is not in scope.
"""Property test for the ``GET /projects/{project_id}`` per-project response shape.

**Validates Requirements 13.2, 13.6, 14.3, 14.5** (Property 22 in the design).

For every digit-only ``project_id`` and every reachable Project_Catalog /
Knowledge_Store state the per-project route's response is determined by
three mutually exclusive branches:

* **In scope and a profile is persisted** (Requirements 13.2, 13.6) —
  HTTP 200 with the documented HTML content type, and a body that
  embeds the ``Project_Profile_Diagram`` for the requested project
  (recognized by the diagram fragment's outer
  ``<section class="project-profile-diagram">`` and the
  ``data-project-id="{project_id}"`` attribute the renderer writes
  there).
* **In scope but no profile persisted** (Requirement 14.3) — HTTP 200
  with the documented content type, and a body that contains the
  documented ``"Project has not yet been analyzed; run an
  Ingestion_Job"`` message verbatim and *no* ``Project_Profile_Diagram``
  (the not-yet-analyzed branch must suppress diagram content
  entirely so the operator only sees the empty-state message).
* **Not in scope** (Requirements 13.6, 14.5) — HTTP 404 with the
  documented content type, and a body that contains the documented
  ``"Project {project_id} is not in scope"`` message with the
  requested ``project_id`` interpolated verbatim, and *no*
  ``Project_Profile_Diagram`` — even when the store happens to hold a
  profile under that id (an out-of-scope id must not leak any
  profile-derived content).

Hypothesis generates the branch and the requested digit-only
``project_id`` directly so all three branches are exercised within a
single ``@settings(max_examples=100)`` run. The ``project_id`` is drawn
from :data:`_PROJECT_ID_STRATEGY` (positive integers) and rendered into
the URL via ``str(int)``, which by construction yields a digit-only
path matching the ``[0-9]+`` regex of Starlette's ``:int`` path
converter that the per-project route is registered under.

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the test does not depend on a real
SQLite database. The ``httpx.AsyncClient`` + ``httpx.ASGITransport``
pair drives the in-process ASGI app the same way Property 21 and the
unit tests in ``tests/unit/test_visualization_server_per_project.py``
do — no socket bind, no event-loop juggling. ``asyncio.run`` is used
inside the synchronous test body so Hypothesis' ``@given`` composes
cleanly with the async transport (the same pattern Property 18,
Property 19, and Property 21 use in this suite).
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
    NOT_IN_SCOPE_MESSAGE_TEMPLATE,
    NOT_YET_ANALYZED_MESSAGE,
)
from project_knowledge_mcp.models import ProjectProfile
from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
    build_visualization_app,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from project_knowledge_mcp.project_catalog import InScopeProject


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Fakes (duck-typed to ``_CatalogReader`` / ``_ProfileReader``)
# ---------------------------------------------------------------------------


class _FakeCatalog:
    """Minimal stand-in for :class:`ProjectCatalog`.

    The per-project handler only calls :meth:`is_in_scope` on this
    route, so :meth:`list_in_scope` returns an empty list — it exists
    purely to satisfy the structural ``_CatalogReader`` Protocol so a
    single fake can be passed where a real ``ProjectCatalog`` is
    expected.
    """

    def __init__(self, in_scope_ids: set[int]) -> None:
        self._in_scope_ids = set(in_scope_ids)

    def list_in_scope(self) -> list[InScopeProject]:
        # Not consulted by the per-project handler under test; returning
        # an empty list keeps the fake structurally compatible with
        # ``_CatalogReader`` without inventing arbitrary catalog rows
        # that the property is not asserting against.
        return []

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        return gitlab_project_id in self._in_scope_ids


class _FakeStore:
    """Minimal stand-in for :class:`KnowledgeStore`.

    Holds a mapping from ``gitlab_project_id`` to ``ProjectProfile``
    instances so the test can express the persisted-profile universe
    declaratively. :meth:`get_current_snapshot_id` is implemented so
    the fake satisfies the ``_ProfileReader`` Protocol's
    ``_SnapshotReader`` parent, even though the per-project handler
    does not consult it directly (it relies on
    :meth:`KnowledgeStore.get_profile` filtering by current snapshot
    internally — Property 11).
    """

    def __init__(
        self,
        *,
        profiles: dict[int, ProjectProfile] | None = None,
        snapshot_id: int | None = 1,
    ) -> None:
        self._snapshot_id = snapshot_id
        self._profiles = dict(profiles) if profiles is not None else {}

    def get_current_snapshot_id(self) -> int | None:
        return self._snapshot_id

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        return self._profiles.get(gitlab_project_id)


# ---------------------------------------------------------------------------
# Profile factory
# ---------------------------------------------------------------------------


def _make_profile(gitlab_project_id: int) -> ProjectProfile:
    """Build a minimal :class:`ProjectProfile` for the in-scope+profile branch.

    The property only asserts that the ``Project_Profile_Diagram`` is
    *embedded* in the response body — not the per-section contents of
    that diagram (Property 23 / Property 24 cover those). A minimal
    profile (no inputs, no outputs, no service or table dependencies)
    therefore exercises the diagram-rendered branch with the smallest
    possible payload, which keeps Hypothesis shrinking fast and the
    failure messages short. The empty-section empty-state messages
    are still rendered by ``render_profile_diagram`` but the property
    does not assert against them here.

    The constant ``analysis_branch_commit_sha``, ``produced_at``, and
    ``analysis_branch`` values are arbitrary but valid; the
    ``ProjectProfile`` model rejects empty values for these fields, so
    keeping them fixed avoids growing the strategy without reason.
    """
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=f"acme/project-{gitlab_project_id}",
        analysis_branch="uat",
        analysis_branch_commit_sha="abcdef0123456789",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary=f"Project {gitlab_project_id} purpose summary.",
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# GitLab project IDs are positive integers in the GitLab API. An upper
# bound of 1,000,000 keeps the rendered URL and HTML body short while
# still exercising multi-digit ids; the lower bound of 1 matches the
# real GitLab API's id range. ``str(int)`` over this range always
# yields a digit-only string that matches the ``[0-9]+`` regex behind
# Starlette's ``:int`` path converter, so by construction every URL
# generated here is "digit-only" in the sense Property 22 requires.
_PROJECT_ID_STRATEGY = st.integers(min_value=1, max_value=1_000_000)


@st.composite
def _branch_state(
    draw: st.DrawFn,
) -> tuple[str, int, set[int], dict[int, ProjectProfile]]:
    """Generate one of the three mutually exclusive per-project states.

    The state is encoded as
    ``(branch_name, requested_id, in_scope_ids, profiles)``:

    * ``branch_name == "in_scope_with_profile"`` -> ``in_scope_ids ==
      {requested_id}`` and ``profiles == {requested_id: ...}``.
    * ``branch_name == "in_scope_no_profile"`` -> ``in_scope_ids ==
      {requested_id}`` and ``profiles == {}``.
    * ``branch_name == "not_in_scope"`` -> ``in_scope_ids == {}`` and
      ``profiles`` deliberately holds a profile under ``requested_id``
      to ensure the handler does *not* leak that content. Requirements
      13.6 and 14.5 require the out-of-scope branch to take priority
      over any profile lookup, so seeding a same-id profile here is
      what catches a regression in the priority of the
      :meth:`is_in_scope` check.
    """
    branch = draw(
        st.sampled_from(
            ("in_scope_with_profile", "in_scope_no_profile", "not_in_scope")
        )
    )
    requested_id = draw(_PROJECT_ID_STRATEGY)
    if branch == "in_scope_with_profile":
        return (
            branch,
            requested_id,
            {requested_id},
            {requested_id: _make_profile(requested_id)},
        )
    if branch == "in_scope_no_profile":
        return branch, requested_id, {requested_id}, {}
    # "not_in_scope": the catalog excludes ``requested_id`` but the
    # store carries a profile under that id. The handler must respond
    # 404 without leaking any profile-derived content (Requirements
    # 13.6, 14.5).
    return branch, requested_id, set(), {requested_id: _make_profile(requested_id)}


# ---------------------------------------------------------------------------
# In-process ASGI driver
# ---------------------------------------------------------------------------


async def _drive_get(app: Starlette, path: str) -> httpx.Response:
    """Issue ``GET path`` against ``app`` over the in-process ASGI transport.

    Uses :class:`httpx.ASGITransport` so the request never touches a
    socket. The transport is constructed per call (rather than once
    per session) because it is cheap and because :class:`httpx.AsyncClient`
    closes the transport on ``__aexit__``, which makes a per-call
    transport the simplest correct lifetime.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://visualization-server.test"
    ) as client:
        return await client.get(path)


def _get(app: Starlette, path: str) -> httpx.Response:
    """Synchronous wrapper around :func:`_drive_get`.

    ``@given`` does not compose with ``async def`` test bodies under
    pytest-asyncio's automatic mode, so the test function stays
    synchronous and uses :func:`asyncio.run` to drive the ASGI
    transport — the same pattern Property 18, Property 19, and
    Property 21 use.
    """
    return asyncio.run(_drive_get(app, path))


# ---------------------------------------------------------------------------
# Property 22 assertions
# ---------------------------------------------------------------------------

#: Substring marking that ``render_profile_diagram`` produced its outer
#: ``<section>``. Used to check whether a body embeds a
#: ``Project_Profile_Diagram`` without depending on any per-section
#: content (which is the territory of Properties 23 / 24).
_PROFILE_DIAGRAM_MARKER = 'class="project-profile-diagram"'


def _assert_in_scope_with_profile_response(
    response: httpx.Response, project_id: int
) -> None:
    """Assert the in-scope + profile-persisted branch's response shape.

    Implements Requirements 13.2 and 13.6:

    * status code 200,
    * Content-Type is the documented :data:`HTML_CONTENT_TYPE`
      (Requirement 13.5),
    * the body embeds the ``Project_Profile_Diagram`` for the
      requested project (the diagram's outer ``<section>`` carries the
      class :data:`_PROFILE_DIAGRAM_MARKER` and a
      ``data-project-id="{project_id}"`` attribute),
    * neither the not-yet-analyzed empty-state message nor the
      not-in-scope message appears — those are reserved for the other
      two branches.
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert _PROFILE_DIAGRAM_MARKER in body, (
        "in-scope+profile body must embed the Project_Profile_Diagram "
        "(Requirements 13.2, 13.6)."
    )
    assert f'data-project-id="{project_id}"' in body, (
        "in-scope+profile body must carry the requested project_id on "
        "the diagram fragment (Requirements 13.2, 13.6)."
    )
    assert NOT_YET_ANALYZED_MESSAGE not in body, (
        "in-scope+profile body must not also show the not-yet-analyzed "
        "empty-state message (Property 22 branch-exclusivity)."
    )
    assert NOT_IN_SCOPE_MESSAGE_TEMPLATE.format(project_id=project_id) not in body, (
        "in-scope+profile body must not also show the not-in-scope "
        "message (Property 22 branch-exclusivity)."
    )


def _assert_in_scope_no_profile_response(
    response: httpx.Response, project_id: int
) -> None:
    """Assert the in-scope + no-profile branch's response shape.

    Implements Requirement 14.3:

    * status code 200 (the project is in scope, so this is *not* a
      404 branch),
    * Content-Type is the documented :data:`HTML_CONTENT_TYPE`,
    * the body contains the documented :data:`NOT_YET_ANALYZED_MESSAGE`
      verbatim,
    * the body does *not* embed a ``Project_Profile_Diagram`` — the
      not-yet-analyzed branch must suppress diagram content entirely,
    * the body does *not* show the not-in-scope message.
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NOT_YET_ANALYZED_MESSAGE in body, (
        "in-scope+no-profile body must contain the documented "
        f"{NOT_YET_ANALYZED_MESSAGE!r} message (Requirement 14.3)."
    )
    assert _PROFILE_DIAGRAM_MARKER not in body, (
        "in-scope+no-profile body must not embed a Project_Profile_Diagram "
        "(Requirement 14.3)."
    )
    assert NOT_IN_SCOPE_MESSAGE_TEMPLATE.format(project_id=project_id) not in body, (
        "in-scope+no-profile body must not also show the not-in-scope "
        "message (Property 22 branch-exclusivity)."
    )


def _assert_not_in_scope_response(
    response: httpx.Response, project_id: int
) -> None:
    """Assert the not-in-scope branch's response shape.

    Implements Requirements 13.6 and 14.5:

    * status code 404,
    * Content-Type is the documented :data:`HTML_CONTENT_TYPE`,
    * the body contains the documented
      :data:`NOT_IN_SCOPE_MESSAGE_TEMPLATE` with the requested
      ``project_id`` interpolated verbatim,
    * the requested ``project_id`` value appears in the body so an
      operator (or this test) can identify which id was rejected,
    * the body does *not* embed a ``Project_Profile_Diagram`` — even
      though the fake store deliberately holds a profile under the
      same id, the handler must short-circuit on
      :meth:`is_in_scope` returning ``False`` and never reach for
      the profile (Requirements 13.6, 14.5),
    * the body does *not* show the in-scope not-yet-analyzed message.
    """
    assert response.status_code == 404
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NOT_IN_SCOPE_MESSAGE_TEMPLATE.format(project_id=project_id) in body, (
        "not-in-scope body must contain the documented "
        f"{NOT_IN_SCOPE_MESSAGE_TEMPLATE!r} message with project_id "
        f"{project_id} interpolated (Requirements 13.6, 14.5)."
    )
    assert str(project_id) in body, (
        "not-in-scope body must include the requested project_id value "
        "(Requirements 13.6, 14.5)."
    )
    assert _PROFILE_DIAGRAM_MARKER not in body, (
        "not-in-scope body must not leak any Project_Profile_Diagram "
        "content even when the store holds a profile under the same id "
        "(Requirements 13.6, 14.5)."
    )
    assert NOT_YET_ANALYZED_MESSAGE not in body, (
        "not-in-scope body must not also show the not-yet-analyzed "
        "message (Property 22 branch-exclusivity)."
    )


# ---------------------------------------------------------------------------
# Property 22
# ---------------------------------------------------------------------------


@given(state=_branch_state())
@settings(max_examples=100)
def test_per_project_response_shape_property(
    state: tuple[str, int, set[int], dict[int, ProjectProfile]],
) -> None:
    """Property 22: ``GET /projects/{project_id}`` returns the documented shape.

    For every digit-only ``project_id`` and every reachable
    Project_Catalog / Knowledge_Store state, the per-project route's
    response satisfies the branch-specific post-conditions above. The
    three branches are mutually exclusive, so a single
    Hypothesis-driven test exercises all of them — and a regression in
    any one branch (e.g., the in-scope branch losing its diagram, the
    not-yet-analyzed branch leaking a diagram, or the not-in-scope
    branch leaking profile content for an out-of-scope id) is caught
    by the same test.

    **Validates Requirements 13.2, 13.6, 14.3, 14.5.**
    """
    branch, requested_id, in_scope_ids, profiles = state
    app = build_visualization_app(
        catalog=_FakeCatalog(in_scope_ids),
        store=_FakeStore(profiles=profiles),
    )

    response = _get(app, f"/projects/{requested_id}")

    if branch == "in_scope_with_profile":
        _assert_in_scope_with_profile_response(response, requested_id)
    elif branch == "in_scope_no_profile":
        _assert_in_scope_no_profile_response(response, requested_id)
    else:
        assert branch == "not_in_scope"
        _assert_not_in_scope_response(response, requested_id)

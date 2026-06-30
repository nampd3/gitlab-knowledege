# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 25: For all HTTP 200 responses from the four Visualization_Server diagram routes (/, /projects/{project_id}, /dependencies, /conflicts), the Content-Type response header SHALL equal exactly "text/html; charset=utf-8".
"""Property test for the fixed ``Content-Type`` on every diagram-route 200 response.

**Validates Requirement 13.5** (Property 25 in the design).

For all HTTP 200 responses from the four Visualization_Server diagram
routes (``/``, ``/projects/{project_id}``, ``/dependencies``,
``/conflicts``), the ``Content-Type`` response header SHALL equal
exactly ``"text/html; charset=utf-8"``.

The property is deliberately scoped to *200* responses: the four
non-200 surfaces (404 not-in-scope on ``/projects/{project_id}``, 404
unknown-path, 405 non-GET, 503 store-unavailable) are covered by the
fallback handlers and are the subject of Properties 26 / 27 / 28
respectively. To keep this test focused on Property 25's contract, the
strategies below generate states that only produce 200 responses for
each route:

* ``/`` — the index handler returns 200 in every catalog/snapshot state
  (no-snapshot, empty-catalog, populated). All three branches are
  exercised.
* ``/projects/{project_id}`` — only the in-scope branches return 200
  (with or without a persisted profile). The not-in-scope branch
  returns 404 and is excluded from this property's universe.
* ``/dependencies`` and ``/conflicts`` — both routes return 200 in
  every catalog/snapshot state (no-snapshot, empty-catalog, populated).
  All three branches are exercised.

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the test does not depend on a real
SQLite database. The ``httpx.AsyncClient`` + ``httpx.ASGITransport``
pair drives the in-process ASGI app the same way Properties 21-24 do —
no socket bind, no event-loop juggling. ``asyncio.run`` is used inside
the synchronous test body so Hypothesis' ``@given`` composes cleanly
with the async transport.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import ProjectProfile
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
    with. Both :meth:`list_in_scope` (used by ``/``, ``/dependencies``,
    ``/conflicts``) and :meth:`is_in_scope` (used by
    ``/projects/{project_id}``) are implemented so a single fake can
    serve every route under test.
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

    Carries an optional ``snapshot_id`` (``None`` exercises the
    no-snapshot branch on the three non-per-project routes) plus a
    mapping from ``gitlab_project_id`` to :class:`ProjectProfile` for
    the per-project route's "in-scope with profile" branch.
    :meth:`list_profiles` returns the recorded profiles in the order
    they were inserted; :meth:`get_profile` looks them up by id.
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
        return list(self._profiles.values())



# ---------------------------------------------------------------------------
# Profile factory
# ---------------------------------------------------------------------------


def _make_profile(gitlab_project_id: int) -> ProjectProfile:
    """Build a minimal :class:`ProjectProfile` for the per-project 200 branch.

    Property 25 only asserts on the response's ``Content-Type`` header,
    not on the body content, so a minimal profile (no inputs, no
    outputs, no service or table dependencies) is sufficient to drive
    the ``in-scope + profile`` branch through ``HTMLResponse``. The
    other ``ProjectProfile`` fields are required by the model; constant
    but valid values keep the strategy small and Hypothesis shrinking
    fast.
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

# GitLab project IDs are positive integers in the GitLab API. The upper
# bound of 1,000,000 mirrors Properties 21-24 and keeps the rendered
# body short while still exercising multi-digit ids. ``str(int)`` over
# this range always yields a digit-only path matching the ``[0-9]+``
# regex of Starlette's ``:int`` path converter, so by construction every
# ``/projects/{id}`` URL generated here matches the per-project route.
_PROJECT_ID_STRATEGY = st.integers(min_value=1, max_value=1_000_000)

#: Alphabet for ``InScopeProject.full_path`` strings. Restricted to
#: printable ASCII excluding the four characters that the dependency-
#: graph and conflict-overview renderers' label escapes treat
#: specially. Property 25 does not assert on body content, but using a
#: safe alphabet keeps the underlying renderers from raising on a
#: pathological glyph.
_FULL_PATH_ALPHABET = st.characters(
    min_codepoint=0x20,
    max_codepoint=0x7E,
    exclude_characters='<>&"',
)

_FULL_PATH_STRATEGY = st.text(
    alphabet=_FULL_PATH_ALPHABET,
    min_size=1,
    max_size=20,
)


def _in_scope_project_strategy() -> st.SearchStrategy[InScopeProject]:
    """Build a single :class:`InScopeProject`."""
    return st.builds(
        InScopeProject,
        gitlab_project_id=_PROJECT_ID_STRATEGY,
        full_path=_FULL_PATH_STRATEGY,
    )


# Lists of projects with unique ``gitlab_project_id``, mirroring the
# ``UNIQUE(snapshot_id, gitlab_project_id)`` constraint on the
# ``project_catalog`` table. ``max_size=4`` keeps Hypothesis fast since
# Property 25 only cares about the response header, not the body.
_POPULATED_PROJECTS_STRATEGY = st.lists(
    _in_scope_project_strategy(),
    min_size=1,
    max_size=4,
    unique_by=lambda project: project.gitlab_project_id,
)


#: Snapshot id used by every populated and empty-catalog state. The
#: handlers only check ``is None``, so the concrete value is
#: irrelevant.
_SNAPSHOT_ID = 99



@st.composite
def _request_state(
    draw: st.DrawFn,
) -> tuple[str, list[InScopeProject], dict[int, ProjectProfile], int | None]:
    """Generate one ``(path, catalog, profiles, snapshot_id)`` request scenario.

    The strategy enumerates the four diagram routes and, for each one,
    samples the catalog/snapshot/profile state branches that produce
    HTTP 200. The result tuple is consumed by
    :func:`test_diagram_routes_fixed_content_type_property` below.

    Branches per route:

    * ``"/"``, ``"/dependencies"``, ``"/conflicts"`` — three branches:
      ``no_snapshot`` (``snapshot_id is None``), ``empty_catalog``
      (snapshot committed, catalog empty), ``populated`` (snapshot
      committed, catalog non-empty). All three return 200.
    * ``"/projects/{id}"`` — two branches: ``in_scope_with_profile``
      and ``in_scope_no_profile``. Both return 200. The third
      not-in-scope branch returns 404 and is excluded so the
      assertion universe matches Property 25's "for all 200 responses"
      framing.
    """
    route = draw(
        st.sampled_from(("/", "/projects/{id}", "/dependencies"))
    )

    if route == "/projects/{id}":
        # Both 200 branches are in-scope; sample whether a profile is
        # also persisted.
        project_id = draw(_PROJECT_ID_STRATEGY)
        has_profile = draw(st.booleans())
        catalog = [
            InScopeProject(
                gitlab_project_id=project_id,
                full_path=f"acme/project-{project_id}",
            )
        ]
        profiles: dict[int, ProjectProfile] = (
            {project_id: _make_profile(project_id)} if has_profile else {}
        )
        return f"/projects/{project_id}", catalog, profiles, _SNAPSHOT_ID

    branch = draw(st.sampled_from(("no_snapshot", "empty_catalog", "populated")))
    if branch == "no_snapshot":
        return route, [], {}, None
    if branch == "empty_catalog":
        return route, [], {}, _SNAPSHOT_ID
    catalog = draw(_POPULATED_PROJECTS_STRATEGY)
    # For ``/dependencies`` and ``/conflicts`` we leave ``profiles``
    # empty: Property 25 is about the ``Content-Type`` header, which is
    # set unconditionally on every populated-branch response regardless
    # of whether any profile is persisted. Empty profiles also keep the
    # injected ``find_all_conflicts`` (the live one) operating on a
    # trivial input — the response is still 200 with an HTML body.
    return route, catalog, {}, _SNAPSHOT_ID



# ---------------------------------------------------------------------------
# In-process ASGI driver
# ---------------------------------------------------------------------------


async def _drive_get(app: Starlette, path: str) -> httpx.Response:
    """Issue ``GET path`` against ``app`` over the in-process ASGI transport.

    Uses :class:`httpx.ASGITransport` so the request never touches a
    socket. Mirrors the helpers in Properties 21-24.
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
    transport — the same pattern Properties 21-24 use.
    """
    return asyncio.run(_drive_get(app, path))



# ---------------------------------------------------------------------------
# Property 25
# ---------------------------------------------------------------------------


@given(state=_request_state())
@settings(max_examples=100)
def test_diagram_routes_fixed_content_type_property(
    state: tuple[str, list[InScopeProject], dict[int, ProjectProfile], int | None],
) -> None:
    """Property 25: every 200 response from the four diagram routes carries the documented Content-Type.

    For every reachable ``(route, catalog, profiles, snapshot_id)``
    state that the four documented diagram routes can return HTTP 200
    on, the response's ``Content-Type`` header equals exactly
    :data:`HTML_CONTENT_TYPE` (``"text/html; charset=utf-8"``).

    The state generator is constrained to scenarios that produce 200
    responses (Properties 26 / 27 / 28 cover the 404 / 405 / 503
    surfaces); a status-code assertion is therefore included as a
    sanity check so a regression that turned a documented 200 surface
    into a non-200 response would be caught here too rather than
    silently bypassing the ``Content-Type`` assertion.

    **Validates Requirement 13.5.**
    """
    path, catalog, profiles, snapshot_id = state
    app = build_visualization_app(
        catalog=_FakeCatalog(catalog),
        store=_FakeStore(snapshot_id=snapshot_id, profiles=profiles),
    )

    response = _get(app, path)

    assert response.status_code == 200, (
        f"strategy generated a non-200 state for {path!r}; Property 25 "
        "is scoped to 200 responses (status code "
        f"{response.status_code} returned)."
    )
    assert response.headers["content-type"] == HTML_CONTENT_TYPE, (
        f"diagram route {path!r} returned 200 with Content-Type "
        f"{response.headers['content-type']!r}, expected "
        f"{HTML_CONTENT_TYPE!r} (Requirement 13.5)."
    )

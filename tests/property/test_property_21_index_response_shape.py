# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 21: For all Project_Catalog states (including empty), GET / SHALL return HTTP 200 with an HTML body that: if the catalog is non-empty and a snapshot is current, contains exactly one list entry per in-scope project ordered by gitlab_project_id ascending, where each entry includes the project's ID, full path, a link to its Project_Profile_Diagram, a link to the Dependency_Graph_Diagram, and a link to the Conflict_Overview_Diagram, if the catalog is empty, contains the empty-state message "No Projects are in scope" and no per-project list entries, if no Ingestion_Job has ever completed, contains the "no project knowledge available; run an Ingestion_Job" message and no diagram content.
"""Property test for the ``GET /`` index route response shape.

**Validates Requirements 13.1, 14.4** (Property 21 in the design).

For every reachable Project_Catalog / current-snapshot combination, the
Visualization_Server's ``GET /`` route SHALL return HTTP 200 with an
HTML body whose shape is determined by the state in three mutually
exclusive branches:

* **No Ingestion_Job has ever completed**
  (``Knowledge_Store.get_current_snapshot_id() is None``): the body
  contains the documented "no project knowledge available; run an
  Ingestion_Job" message and no per-project list entries, no
  ``/projects/{id}`` link, and no ``/dependencies`` or ``/conflicts``
  links (Requirement 14.4 forbids any
  ``Project_Profile_Diagram``-, ``Dependency_Graph_Diagram``-, or
  ``Conflict_Overview_Diagram``-derived content before the first
  successful ``Ingestion_Job``).
* **Snapshot committed but Project_Catalog empty**: the body contains
  the documented "No Projects are in scope" message and no
  per-project list entries (Requirement 13.1's empty-catalog branch).
* **Snapshot committed and Project_Catalog non-empty**: the body
  contains exactly one list entry per in-scope project ordered by
  ``gitlab_project_id`` ascending; each entry includes the project's
  GitLab id, full path, a link to its ``Project_Profile_Diagram``
  (``/projects/{id}``), a link to the ``Dependency_Graph_Diagram``
  (``/dependencies``), and a link to the ``Conflict_Overview_Diagram``
  (``/conflicts``) (Requirement 13.1's populated branch).

Hypothesis generates the catalog state directly so all three branches
are exercised within a single ``@settings(max_examples=100)`` run. The
populated branch is driven with a list of unique
:class:`InScopeProject` values (uniqueness on ``gitlab_project_id``
mirrors the catalog's ``UNIQUE(snapshot_id, gitlab_project_id)``
constraint) returned by the catalog *out of order* so the assertion
on ascending ordering is exercised against the handler's defensive
sort rather than against the (already-sorted) real catalog reader.

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_SnapshotReader`` Protocols defined in
``visualization_server.py`` so the test does not depend on a real
SQLite database. The ``httpx.AsyncClient`` + ``httpx.ASGITransport``
pair drives the in-process ASGI app the same way the unit tests in
``tests/unit/test_visualization_server_index.py`` do — no socket
bind, no event loop juggling. ``asyncio.run`` is used inside the
synchronous test body so Hypothesis' ``@given`` composes cleanly
with the async transport (the same pattern is used by Property 18
and Property 19 in this suite).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from markupsafe import escape as markup_escape

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


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Fakes (duck-typed to ``_CatalogReader`` / ``_SnapshotReader``)
# ---------------------------------------------------------------------------


class _FakeCatalog:
    """Minimal stand-in for :class:`ProjectCatalog`.

    Returns whatever list of :class:`InScopeProject` it was constructed
    with, in the order it was given. The handler under test sorts
    defensively, so passing an unsorted list is the right way to
    exercise Property 21's "ordered by ``gitlab_project_id`` ascending"
    clause: a regression in the handler's sort would surface even when
    the real catalog reader already orders rows.
    """

    def __init__(self, projects: Sequence[InScopeProject]) -> None:
        self._projects = list(projects)

    def list_in_scope(self) -> list[InScopeProject]:
        return list(self._projects)


class _FakeStore:
    """Minimal stand-in for :class:`KnowledgeStore`.

    Only :meth:`get_current_snapshot_id` is implemented because that
    is the sole ``Knowledge_Store`` method the index handler consults
    (the per-project handler added in task 10.6 will read profiles
    too, but that is out of scope for Property 21).
    """

    def __init__(self, snapshot_id: int | None) -> None:
        self._snapshot_id = snapshot_id

    def get_current_snapshot_id(self) -> int | None:
        return self._snapshot_id


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# ``InScopeProject.full_path`` is rendered into the response body by a
# Jinja2 template with ``autoescape=True``, so the body contains the
# HTML-escaped form of the path (``&`` -> ``&amp;``, ``"`` -> ``&#34;``,
# etc.). To keep the substring assertions honest we generate paths from
# a printable-ASCII alphabet and compare against the escaped form via
# :func:`markupsafe.escape` — which is exactly what Jinja2 calls under
# ``autoescape``, so the comparison matches the renderer's output
# byte-for-byte (including its idiosyncratic ``&#34;`` for ``"`` and
# ``&#39;`` for ``'``, neither of which Python's :func:`html.escape`
# would produce). The alphabet excludes the C0 control characters and
# DEL so the rendered HTML stays printable; the
# ``Field(min_length=1)`` constraint on ``InScopeProject.full_path`` is
# enforced by ``min_size=1``.
_FULL_PATH_STRATEGY = st.text(
    alphabet=st.characters(
        min_codepoint=0x20,
        max_codepoint=0x7E,
    ),
    min_size=1,
    max_size=40,
)

# GitLab project IDs are positive integers in the GitLab API; an upper
# bound of 1,000,000 keeps the rendered body short while still
# exercising multi-digit ids that surface ordering bugs (lexicographic
# vs. numeric sort).
_GITLAB_ID_STRATEGY = st.integers(min_value=1, max_value=1_000_000)


def _in_scope_project_strategy() -> st.SearchStrategy[InScopeProject]:
    """Build a single :class:`InScopeProject`."""
    return st.builds(
        InScopeProject,
        gitlab_project_id=_GITLAB_ID_STRATEGY,
        full_path=_FULL_PATH_STRATEGY,
    )


# Lists of projects with unique ``gitlab_project_id`` — this mirrors the
# ``UNIQUE(snapshot_id, gitlab_project_id)`` constraint on the
# ``project_catalog`` table, so any state Hypothesis generates is
# reachable in production. The list is unconstrained on order so the
# handler's defensive sort is what produces the ascending order in the
# rendered body.
_POPULATED_PROJECTS_STRATEGY = st.lists(
    _in_scope_project_strategy(),
    min_size=1,
    max_size=8,
    unique_by=lambda project: project.gitlab_project_id,
)


# A snapshot id used by the populated and empty-catalog branches. The
# concrete value is irrelevant to the handler (it only checks
# ``is None``), so a fixed positive integer keeps the strategy small.
_SNAPSHOT_ID = 99


@st.composite
def _catalog_state(
    draw: st.DrawFn,
) -> tuple[str, list[InScopeProject], int | None]:
    """Generate one of the three mutually exclusive index-page states.

    The state is encoded as ``(branch_name, projects, snapshot_id)``:

    * ``branch_name == "no_snapshot"`` -> ``snapshot_id is None``,
      ``projects == []``. Note that the catalog list is irrelevant in
      this branch (the handler ignores it when no snapshot is
      committed); pinning it to ``[]`` keeps the post-condition
      assertions uniform.
    * ``branch_name == "empty_catalog"`` -> ``snapshot_id`` is a
      fixed positive integer, ``projects == []``.
    * ``branch_name == "populated"`` -> ``snapshot_id`` is a fixed
      positive integer, ``projects`` is a non-empty list of unique
      :class:`InScopeProject` values whose ``gitlab_project_id`` is
      drawn from :data:`_GITLAB_ID_STRATEGY`.
    """
    branch = draw(st.sampled_from(("no_snapshot", "empty_catalog", "populated")))
    if branch == "no_snapshot":
        return "no_snapshot", [], None
    if branch == "empty_catalog":
        return "empty_catalog", [], _SNAPSHOT_ID
    return "populated", draw(_POPULATED_PROJECTS_STRATEGY), _SNAPSHOT_ID


# ---------------------------------------------------------------------------
# In-process ASGI driver
# ---------------------------------------------------------------------------


async def _drive_get_index(app: Starlette) -> httpx.Response:
    """Issue ``GET /`` against ``app`` over the in-process ASGI transport.

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
        return await client.get("/")


def _get_index(app: Starlette) -> httpx.Response:
    """Synchronous wrapper around :func:`_drive_get_index`.

    ``@given`` does not compose with ``async def`` test bodies under
    pytest-asyncio's automatic mode, so the test function stays
    synchronous and uses :func:`asyncio.run` to drive the ASGI
    transport — the same pattern Property 18 and Property 19 use.
    """
    return asyncio.run(_drive_get_index(app))


# ---------------------------------------------------------------------------
# Property 21 assertions
# ---------------------------------------------------------------------------


def _assert_no_snapshot_response(response: httpx.Response) -> None:
    """Assert the no-snapshot branch's response shape.

    Implements Requirement 14.4: when no ``Ingestion_Job`` has ever
    completed the body contains the documented no-snapshot message and
    omits all ``Project_Profile_Diagram``, ``Dependency_Graph_Diagram``,
    and ``Conflict_Overview_Diagram`` content.
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert NO_SNAPSHOT_MESSAGE in body, (
        "no-snapshot body must contain the documented "
        f"{NO_SNAPSHOT_MESSAGE!r} message (Requirement 14.4)."
    )
    assert EMPTY_CATALOG_MESSAGE not in body, (
        "no-snapshot body must not also show the empty-catalog message; "
        "the no-snapshot branch takes priority over the empty-catalog "
        "branch (Property 21)."
    )
    # Requirement 14.4 forbids any per-project, dependency, or
    # conflict diagram content before the first Ingestion_Job
    # completes.
    assert 'href="/projects/' not in body
    assert 'href="/dependencies"' not in body


def _assert_empty_catalog_response(response: httpx.Response) -> None:
    """Assert the empty-catalog branch's response shape.

    Implements Requirement 13.1's "zero in-scope Projects" branch:
    the body contains the documented empty-catalog message and no
    per-project list entries. The no-snapshot message must not be
    shown — a snapshot has been committed, so the index is past the
    no-knowledge branch.
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert EMPTY_CATALOG_MESSAGE in body, (
        "empty-catalog body must contain the documented "
        f"{EMPTY_CATALOG_MESSAGE!r} message (Requirement 13.1)."
    )
    assert NO_SNAPSHOT_MESSAGE not in body, (
        "empty-catalog body must not show the no-snapshot message; a "
        "snapshot is committed in this branch (Property 21)."
    )
    # Requirement 13.1's empty-catalog branch: omit per-project list
    # entries.
    assert 'href="/projects/' not in body


def _assert_populated_response(
    response: httpx.Response,
    projects: Sequence[InScopeProject],
) -> None:
    """Assert the populated branch's response shape.

    Implements Requirement 13.1's populated branch:

    * Each in-scope project appears in the body with its GitLab id and
      full path, and with a link to its
      ``Project_Profile_Diagram`` (``/projects/{id}``).
    * The body contains exactly one ``/projects/{id}`` link per
      project — duplicates would violate the "exactly one list entry
      per in-scope project" clause of Property 21.
    * Projects appear in ``gitlab_project_id`` ascending order
      regardless of the order the catalog returned them in (the
      handler sorts defensively).
    * Each list entry includes a link to ``/dependencies`` (the
      ``Dependency_Graph_Diagram``) and a link to ``/conflicts`` (the
      ``Conflict_Overview_Diagram``); the body therefore contains at
      least ``len(projects)`` of each.
    * Neither empty-state message appears.
    """
    assert response.status_code == 200
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text

    assert NO_SNAPSHOT_MESSAGE not in body
    assert EMPTY_CATALOG_MESSAGE not in body

    # (1) Every project's id, full_path, and per-project link are
    # present, and the per-project link appears exactly once. The
    # ``full_path`` is HTML-escaped by the Jinja2 environment's
    # ``autoescape=True`` setting, so the assertion compares against
    # the escaped form.
    for project in projects:
        per_project_href = f'href="/projects/{project.gitlab_project_id}"'
        assert per_project_href in body, (
            f"populated body is missing the per-project link "
            f"{per_project_href!r} for project "
            f"{project.gitlab_project_id} (Requirement 13.1)."
        )
        assert body.count(per_project_href) == 1, (
            f"populated body must contain exactly one entry per in-scope "
            f"project; project {project.gitlab_project_id} has "
            f"{body.count(per_project_href)} entries (Property 21)."
        )
        assert str(project.gitlab_project_id) in body
        assert str(markup_escape(project.full_path)) in body, (
            f"populated body is missing the full path for project "
            f"{project.gitlab_project_id} (Requirement 13.1)."
        )

    # (2) Ascending order on ``gitlab_project_id``. Compare the
    # positions of the per-project ``href`` substrings against the
    # positions you would get if the projects were already sorted by
    # id ascending.
    sorted_ids = sorted(project.gitlab_project_id for project in projects)
    actual_positions = [
        body.index(f'href="/projects/{project_id}"') for project_id in sorted_ids
    ]
    assert actual_positions == sorted(actual_positions), (
        "populated body must list projects in gitlab_project_id ascending "
        "order regardless of the order the catalog returned them "
        "(Requirement 13.1)."
    )

    # (3) Each entry includes a Dependency_Graph_Diagram link, so
    # the body contains at least one occurrence per project.
    assert body.count('href="/dependencies"') >= len(projects), (
        "populated body must include a Dependency_Graph_Diagram link in "
        "every list entry (Requirement 13.1)."
    )


# ---------------------------------------------------------------------------
# Property 21
# ---------------------------------------------------------------------------


@given(state=_catalog_state())
@settings(max_examples=100)
def test_index_response_shape_property(
    state: tuple[str, list[InScopeProject], int | None],
) -> None:
    """Property 21: ``GET /`` returns the documented shape for every state.

    For every Project_Catalog / current-snapshot state, the
    ``GET /`` response satisfies the branch-specific post-conditions
    above. The three branches are mutually exclusive, so a single
    Hypothesis-driven test exercises all of them — and a regression in
    any one branch (e.g., the no-snapshot body leaking diagram links,
    the empty-catalog body forgetting its message, or the populated
    body losing a project) is caught by the same test.

    **Validates Requirements 13.1, 14.4.**
    """
    branch, projects, snapshot_id = state
    app = build_visualization_app(
        catalog=_FakeCatalog(projects),
        store=_FakeStore(snapshot_id=snapshot_id),
    )

    response = _get_index(app)

    if branch == "no_snapshot":
        _assert_no_snapshot_response(response)
    elif branch == "empty_catalog":
        _assert_empty_catalog_response(response)
    else:
        assert branch == "populated"
        _assert_populated_response(response, projects)

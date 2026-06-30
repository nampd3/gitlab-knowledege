# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 28: For all HTTP GET requests to /, /projects/{project_id} (digit-only), /dependencies, or /conflicts issued while Knowledge_Store reads raise KnowledgeStoreUnavailableError, the Visualization_Server SHALL respond with HTTP 503 and an HTML body stating that project knowledge is temporarily unavailable, and SHALL NOT include any Project_Profile-derived content drawn from caches or in-memory state.
"""Property test for the 503 fallback across every diagram route.

**Validates Requirement 14.6** (Property 28 in the design).

For every HTTP GET request to one of the four documented diagram
routes (``/``, ``/projects/{digit-only-id}``, ``/dependencies``,
``/conflicts``) issued while the ``Knowledge_Store`` reader interface
raises :class:`KnowledgeStoreUnavailableError`, the
``Visualization_Server`` SHALL:

* respond with HTTP status 503,
* set the documented ``text/html; charset=utf-8`` Content-Type
  (Requirement 13.5 also applies here so the operator-facing 503 is
  recognizably HTML),
* include the documented :data:`STORE_UNAVAILABLE_MESSAGE` in the
  body, and
* contain *no* ``Project_Profile``-derived content drawn from caches
  or in-memory state — i.e. the body must not embed any of the four
  diagram class names (``project-profile-diagram``,
  ``dependency-graph-diagram``, ``conflict-overview-diagram``,
  ``project-list``) and must not echo any of the catalog ids or full
  paths the in-process fakes carry. The catalog ids are deliberately
  injected by the test as a *negative oracle*: a regression that
  begins serving cached or partial state would surface those ids in
  the body, and the assertion below catches it.

Hypothesis drives the test along two axes:

1. **Route under test** — sampled from the four documented routes.
   The per-project route is parameterized by a Hypothesis-generated
   digit-only ``project_id`` so the route is exercised across the
   full ``[0-9]+`` shape rather than a single canonical id. The
   ``project_id`` is rendered into the URL via ``str(int)`` over
   positive integers, which by construction yields a digit-only
   path matching the ``[0-9]+`` regex behind Starlette's ``:int``
   path converter.
2. **Failure-injection seed** — Hypothesis chooses a snapshot id and
   an in-scope catalog id for the wrapper fakes to expose to the
   handler *if* a read happened to succeed. Both reader interfaces
   raise unconditionally (they are the failure-injection wrapper),
   so these values must never appear in the response body. Asserting
   they don't is what gives Property 28 its "no profile-derived
   content leaks" teeth.

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the test does not depend on a real
SQLite database. Every reader method on both fakes raises
:class:`KnowledgeStoreUnavailableError`, so each of the four routes
is forced down its 503 exception-handler path regardless of which
read it would otherwise issue first. The ``httpx.AsyncClient`` +
``httpx.ASGITransport`` pair drives the in-process ASGI app the same
way Property 21 does — no socket bind, no event-loop juggling.
``asyncio.run`` is used inside the synchronous test body so
Hypothesis' ``@given`` composes cleanly with the async transport.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.errors import KnowledgeStoreUnavailableError
from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
    STORE_UNAVAILABLE_MESSAGE,
    build_visualization_app,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from project_knowledge_mcp.models import ProjectProfile
    from project_knowledge_mcp.project_catalog import InScopeProject


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Failure-injection fakes (duck-typed to ``_CatalogReader`` / ``_ProfileReader``)
# ---------------------------------------------------------------------------


class _UnavailableCatalog:
    """Failure-injection wrapper for the ``_CatalogReader`` Protocol.

    Every read method raises :class:`KnowledgeStoreUnavailableError`
    so any handler that touches the catalog hits the documented 503
    path. The ``leak_id`` and ``leak_full_path`` fields are *not*
    consulted by these methods — they are stored only so the test
    body can assert they never bleed into the 503 response. If a
    regression begins serving cached state, the body would echo
    these values and the negative oracle below would fire.
    """

    def __init__(self, *, leak_id: int, leak_full_path: str) -> None:
        self.leak_id = leak_id
        self.leak_full_path = leak_full_path

    def list_in_scope(self) -> list[InScopeProject]:
        raise KnowledgeStoreUnavailableError("catalog list_in_scope failed")

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        del gitlab_project_id
        raise KnowledgeStoreUnavailableError("catalog is_in_scope failed")


class _UnavailableStore:
    """Failure-injection wrapper for the ``_ProfileReader`` Protocol.

    Every read method raises :class:`KnowledgeStoreUnavailableError`
    so the per-project, dependencies, and conflicts handlers all
    hit the documented 503 path on their first store read. The
    ``leak_snapshot_id`` field carries the snapshot id the wrapper
    *would* have returned if the read had succeeded; the test body
    asserts that value never appears in the response, which is what
    proves the body is not derived from cached snapshot state.
    """

    def __init__(self, *, leak_snapshot_id: int) -> None:
        self.leak_snapshot_id = leak_snapshot_id

    def get_current_snapshot_id(self) -> int | None:
        raise KnowledgeStoreUnavailableError("snapshot pointer read failed")

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        del gitlab_project_id
        raise KnowledgeStoreUnavailableError("profile read failed")

    def list_profiles(self) -> list[ProjectProfile]:
        raise KnowledgeStoreUnavailableError("profile list read failed")


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# GitLab project IDs are positive integers in the GitLab API. Drawing
# the requested ``project_id`` from this range yields a digit-only
# URL path via ``str(int)``, matching the ``[0-9]+`` shape of the
# per-project route's ``:int`` converter. The lower bound of 1 mirrors
# real GitLab ids; the upper bound of 1,000,000 keeps generated bodies
# small.
_PROJECT_ID_STRATEGY = st.integers(min_value=1, max_value=1_000_000)

# A separate id range for the *leak* id the catalog would expose if a
# read had succeeded. Using a distinct, larger range (10**7 - 10**9)
# guarantees the leak id has at least 7 digits, so its decimal
# representation cannot accidentally appear as a substring of the
# requested ``project_id`` (which lives in 1 - 10**6) — making the
# "leak id never appears in body" negative-oracle assertion
# unambiguous.
_LEAK_ID_STRATEGY = st.integers(min_value=10_000_000, max_value=999_999_999)

# A distinctive, easy-to-spot full path the catalog would expose if a
# read had succeeded. A printable-ASCII alphabet keeps the rendered
# substring assertion exact (no escaping ambiguity); the
# ``leak-`` prefix ensures the chosen string is highly unlikely to
# collide with any boilerplate in the static 503 HTML body. The
# alphabet excludes ``<``, ``>``, ``&``, ``"``, and ``'`` so a path
# that did leak would appear in the body literally rather than
# HTML-escaped, which would otherwise complicate the negative
# assertion.
_LEAK_FULL_PATH_STRATEGY = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="-_/",
    ),
    min_size=8,
    max_size=24,
).map(lambda suffix: f"leak-canary/{suffix}")

# A snapshot id used purely as a leak canary on the store fake.
# A distinctive 6-digit value is large enough that its decimal
# representation cannot accidentally appear in either the requested
# ``project_id`` (1 - 10**6) or the static 503 body chrome.
_LEAK_SNAPSHOT_ID_STRATEGY = st.integers(min_value=100_000, max_value=999_999)


@st.composite
def _request_state(
    draw: st.DrawFn,
) -> tuple[str, int, str, int]:
    """Generate one HTTP request + leak-canary tuple.

    Returns ``(path, leak_id, leak_full_path, leak_snapshot_id)``:

    * ``path`` is one of the four documented routes. The per-project
      route is parameterized by a Hypothesis-generated digit-only
      ``project_id``; the other three are fixed strings.
    * ``leak_id`` is a 7- to 9-digit integer the catalog would expose
      via :meth:`list_in_scope` if a read had succeeded. It is
      asserted absent from the body.
    * ``leak_full_path`` is a distinctive ``leak-canary/...`` string
      the catalog would expose via :meth:`list_in_scope`. It is
      asserted absent from the body.
    * ``leak_snapshot_id`` is a 6-digit integer the store would
      expose via :meth:`get_current_snapshot_id` if a read had
      succeeded. It is asserted absent from the body.
    """
    route = draw(
        st.sampled_from(
            ("index", "per_project", "dependencies")
        )
    )
    if route == "per_project":
        project_id = draw(_PROJECT_ID_STRATEGY)
        path = f"/projects/{project_id}"
    elif route == "index":
        path = "/"
    elif route == "dependencies":
        path = "/dependencies"
    else:
        raise AssertionError(f"unknown route key: {route!r}")

    return (
        path,
        draw(_LEAK_ID_STRATEGY),
        draw(_LEAK_FULL_PATH_STRATEGY),
        draw(_LEAK_SNAPSHOT_ID_STRATEGY),
    )


# ---------------------------------------------------------------------------
# In-process ASGI driver
# ---------------------------------------------------------------------------


async def _drive_get(app: Starlette, path: str) -> httpx.Response:
    """Issue ``GET path`` against ``app`` over the in-process ASGI transport.

    Uses :class:`httpx.ASGITransport` so the request never touches a
    socket. The transport is constructed per call (rather than once
    per session) because it is cheap and because
    :class:`httpx.AsyncClient` closes the transport on ``__aexit__``,
    which makes a per-call transport the simplest correct lifetime.
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
    transport — the same pattern Property 18, Property 19, Property
    21, and Property 22 use.
    """
    return asyncio.run(_drive_get(app, path))


# ---------------------------------------------------------------------------
# Diagram class names that must NOT appear in a 503 body
# ---------------------------------------------------------------------------

#: The four diagram fragment / list class names that the renderers in
#: ``diagram_renderer.py`` write into the response body when they run.
#: The 503 body is fully derived from a static string and must not
#: embed any of them — Requirement 14.6 forbids serving cached
#: profile data via the unavailable-store fallback. The values are
#: pinned as bare class strings so the assertion catches both
#: ``class="..."`` attributes (where the renderers use them) and any
#: future change in HTML quoting style.
_FORBIDDEN_DIAGRAM_CLASSES: tuple[str, ...] = (
    "project-profile-diagram",
    "dependency-graph-diagram",
    "conflict-overview-diagram",
    "project-list",
)


# ---------------------------------------------------------------------------
# Property 28
# ---------------------------------------------------------------------------


@given(state=_request_state())
@settings(max_examples=100)
def test_store_unavailable_503_across_diagram_routes_property(
    state: tuple[str, int, str, int],
) -> None:
    """Property 28: every diagram route serves a clean 503 when the store is unavailable.

    For every (route, leak-canary) tuple the response satisfies four
    post-conditions:

    1. ``response.status_code == 503``,
    2. ``response.headers["content-type"] == "text/html; charset=utf-8"``,
    3. :data:`STORE_UNAVAILABLE_MESSAGE` appears in the body, and
    4. the body contains *no* ``Project_Profile``-derived content:
       neither any of the four diagram class names
       (:data:`_FORBIDDEN_DIAGRAM_CLASSES`) nor any of the leak
       canaries (the catalog id, the catalog full path, the
       snapshot id) injected into the failure-injection wrappers.

    The negative oracle in (4) is what proves the body is *not*
    derived from cached or in-memory state. If a regression begins
    serving stale snapshot data alongside the 503, the catalog or
    snapshot canary would surface in the body and this assertion
    would fail with the offending value.

    **Validates Requirement 14.6.**
    """
    path, leak_id, leak_full_path, leak_snapshot_id = state

    app = build_visualization_app(
        catalog=_UnavailableCatalog(
            leak_id=leak_id, leak_full_path=leak_full_path
        ),
        store=_UnavailableStore(leak_snapshot_id=leak_snapshot_id),
    )

    response = _get(app, path)

    # (1) HTTP 503 — Requirement 14.6.
    assert response.status_code == 503, (
        f"GET {path} must return HTTP 503 when Knowledge_Store reads "
        f"raise KnowledgeStoreUnavailableError; got {response.status_code} "
        "(Requirement 14.6)."
    )

    # (2) Documented HTML Content-Type — Requirements 13.5 + 14.6.
    assert response.headers["content-type"] == HTML_CONTENT_TYPE, (
        f"GET {path} 503 response must carry "
        f"Content-Type {HTML_CONTENT_TYPE!r}; got "
        f"{response.headers.get('content-type')!r} (Requirement 13.5)."
    )

    body = response.text

    # (3) Documented "temporarily unavailable" message — Requirement 14.6.
    assert STORE_UNAVAILABLE_MESSAGE in body, (
        f"GET {path} 503 body must contain the documented "
        f"{STORE_UNAVAILABLE_MESSAGE!r} message (Requirement 14.6)."
    )

    # (4a) No diagram class names — Requirement 14.6 forbids serving
    # any Project_Profile-derived content via the unavailable-store
    # fallback.
    for class_name in _FORBIDDEN_DIAGRAM_CLASSES:
        assert class_name not in body, (
            f"GET {path} 503 body must not embed the {class_name!r} "
            "diagram class name; that would mean cached profile data "
            "leaked through the 503 fallback (Requirement 14.6)."
        )

    # (4b) No leak canaries — the failure-injection wrapper carries
    # values the body would expose if a read had succeeded. Their
    # absence proves the body is not derived from cached state.
    assert str(leak_id) not in body, (
        f"GET {path} 503 body must not contain the leak-canary catalog "
        f"id {leak_id}; its presence would mean cached catalog data "
        "leaked through the 503 fallback (Requirement 14.6)."
    )
    assert leak_full_path not in body, (
        f"GET {path} 503 body must not contain the leak-canary catalog "
        f"full_path {leak_full_path!r}; its presence would mean cached "
        "catalog data leaked through the 503 fallback (Requirement 14.6)."
    )
    assert str(leak_snapshot_id) not in body, (
        f"GET {path} 503 body must not contain the leak-canary snapshot "
        f"id {leak_snapshot_id}; its presence would mean cached snapshot "
        "state leaked through the 503 fallback (Requirement 14.6)."
    )

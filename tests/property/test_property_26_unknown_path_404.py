# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 26: For all HTTP GET request paths that do not match /, /projects/{digits}, /dependencies, or /conflicts, the Visualization_Server SHALL respond with HTTP 404 and an HTML body that includes the requested path verbatim and that states the requested page does not exist.
"""Property test for the unknown-path 404 fallback handler.

**Validates Requirement 13.7** (Property 26 in the design).

For every HTTP GET request whose path does not match one of the four
documented Visualization_Server routes (``/``, ``/projects/{digits}``,
``/dependencies``, ``/conflicts``), the server SHALL respond with
HTTP 404 and an HTML body that:

* contains the requested HTTP path verbatim — after the renderer's
  documented HTML-escape pass — so an operator can identify which
  path was rejected, and
* states that the requested page does not exist.

Hypothesis generates off-route paths from a printable-ASCII alphabet
across three shapes that together cover the off-route space:

* **General multi-segment paths** drawn from the broad alphabet, e.g.
  ``/foo/bar``, ``/static/style.css``, ``/foo<script>``.
* **``/projects/{non-digit-segment}`` paths** — the per-project route
  is registered with Starlette's ``:int`` converter (regex
  ``[0-9]+``), so any non-digit segment under ``/projects/`` falls
  through to the 404 fallback (Requirement 13.7), as opposed to the
  digit-only ``/projects/{N}`` "not in scope" 404 that the
  per-project handler emits inline (Requirement 13.6).
* **Bounded paths under one of the route prefixes** (e.g.
  ``/dependencies/extra``, ``/conflicts/x``, ``/projects/123/foo``)
  — Starlette routes by exact path, so any extension after a
  documented prefix falls through.

The body assertion compares against ``html.escape(path, quote=False)``,
which is exactly the form :func:`_render_not_found_html` interpolates
into the visible message line and into the ``<title>``. Using a wide
printable-ASCII alphabet (excluding only the URL-special characters
``/``, ``?``, ``#``, and ``%``) exercises the escape behavior on
characters like ``<``, ``>``, ``&``, ``"``, and ``'`` that would
otherwise break the surrounding HTML chrome.

The handler is wired with two duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the test does not depend on a real
SQLite database; the 404 fallback is dispatched by Starlette's
exception machinery before either fake is ever consulted. The
``httpx.AsyncClient`` + ``httpx.ASGITransport`` pair drives the
in-process ASGI app the same way Property 21 and the unit tests in
``tests/unit/test_visualization_server_404_405_503.py`` do — no
socket bind, no event-loop juggling. ``asyncio.run`` is used inside
the synchronous test body so Hypothesis' ``@given`` composes cleanly
with the async transport (the same pattern Properties 18, 19, 21,
and 22 use in this suite).
"""

from __future__ import annotations

import asyncio
import html
import re
from typing import TYPE_CHECKING

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
    NOT_FOUND_MESSAGE_TEMPLATE,
    build_visualization_app,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from project_knowledge_mcp.models import ProjectProfile
    from project_knowledge_mcp.project_catalog import InScopeProject


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Fakes (duck-typed to ``_CatalogReader`` / ``_ProfileReader``)
# ---------------------------------------------------------------------------


class _FakeCatalog:
    """Minimal :class:`_CatalogReader` stand-in for the 404 fallback test.

    Starlette raises ``HTTPException(404)`` from the routing layer
    before any handler runs on an off-route GET, so the catalog is
    never consulted on the path under test. ``list_in_scope`` and
    ``is_in_scope`` therefore exist purely to satisfy the structural
    ``_CatalogReader`` Protocol.
    """

    def list_in_scope(self) -> list[InScopeProject]:
        return []

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        del gitlab_project_id
        return False


class _FakeStore:
    """Minimal :class:`_ProfileReader` stand-in for the 404 fallback test.

    Like :class:`_FakeCatalog`, this fake is never consulted on the
    404 fallback path; the methods exist to satisfy the structural
    ``_ProfileReader`` Protocol so a single ``build_visualization_app``
    call can wire all four documented routes (whose handlers are not
    reached on an off-route request).
    """

    def get_current_snapshot_id(self) -> int | None:
        return None

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        del gitlab_project_id
        return None

    def list_profiles(self) -> list[ProjectProfile]:
        return []


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Printable ASCII (codepoints 0x20-0x7E) excluding the URL-special
# characters ``/`` (segment separator), ``?`` (query separator),
# ``#`` (fragment separator), and ``%`` (percent-encoding introducer).
# Excluding ``%`` keeps the path from accidentally encoding a literal
# ``%xx`` triplet that ASGI would decode into a different byte than
# the test sent. The remaining alphabet still includes every character
# that needs HTML escaping (``<``, ``>``, ``&``, ``"``, ``'``) so the
# assertion exercises the renderer's ``html.escape(path, quote=False)``
# behavior.
_SEGMENT_ALPHABET = st.characters(
    min_codepoint=0x20,
    max_codepoint=0x7E,
    exclude_characters="/?#%",
)

# A single non-empty path segment, with the literal ``"."`` and
# ``".."`` segments filtered out. Both would be collapsed by
# httpx's RFC 3986 path normalization (``/foo/.`` -> ``/foo``,
# ``/foo/..`` -> ``/``), changing the requested path before it
# reached the handler and breaking the round-trip the property
# relies on.
_PATH_SEGMENT = st.text(
    alphabet=_SEGMENT_ALPHABET, min_size=1, max_size=12
).filter(lambda s: s not in (".", ".."))


def _is_off_route(path: str) -> bool:
    """Return True iff ``path`` does not match a documented route.

    Mirrors the routing decisions :func:`build_visualization_app`
    makes: ``/`` is the index route, ``/dependencies`` and
    ``/conflicts`` are exact-match routes,
    ``/projects/{project_id:int}`` matches exactly the ``[0-9]+``
    regex behind Starlette's ``:int`` converter (Requirements 13.2,
    13.6), and ``/static/<filename>`` is a static-file mount that
    serves the package's ``static/`` directory (added so the
    ``/dependencies`` and ``/conflicts`` pages can load the Mermaid
    JavaScript bundle). Every other path is off-route by
    construction and SHALL produce the 404 fallback
    (Requirement 13.7).
    """
    if path in ("/", "/dependencies"):
        return False
    if path.startswith("/static/"):
        return False
    return re.fullmatch(r"/projects/[0-9]+", path) is None


def _join_segments(segments: list[str]) -> str:
    """Build a path from non-empty segments joined by single slashes."""
    return "/" + "/".join(segments)


# Strategy 1 — random multi-segment paths from the broad alphabet,
# filtered to drop anything that happens to match a documented route.
# The filter rate is essentially zero in practice (matching
# ``/dependencies``, ``/conflicts``, or ``/projects/{N}`` from a
# random alphabet of ~95 characters is vanishingly unlikely) but the
# filter is the contract that keeps the strategy correct under any
# alphabet narrowing.
_GENERAL_PATH = (
    st.lists(_PATH_SEGMENT, min_size=1, max_size=4)
    .map(_join_segments)
    .filter(_is_off_route)
)

# Strategy 2 — ``/projects/{non-digit-segment}`` paths. The
# per-project route only matches when the project_id segment is one
# or more digits (its registered ``:int`` converter has regex
# ``[0-9]+``); any non-digit segment must fall through to the 404
# fallback (Requirement 13.7), distinct from the digit-only
# "not in scope" 404 emitted inline by the per-project handler
# (Requirement 13.6).
_NON_DIGIT_PROJECT_PATH = _PATH_SEGMENT.filter(
    lambda s: not s.isdigit()
).map(lambda s: f"/projects/{s}")

# Strategy 3 — extra trailing segments under one of the documented
# prefixes (``/dependencies/x``, ``/conflicts/x/y``,
# ``/projects/123/foo``). Starlette routes by exact path, so any
# extension after the prefix falls through. The filter handles the
# edge case where the random prefix is ``/projects`` and the trailing
# segment list happens to be exactly ``[digits]`` (which would match
# the per-project route).
_BOUNDED_PREFIX = st.sampled_from(("/dependencies", "/projects"))
_BOUNDED_PATH = st.builds(
    lambda prefix, segs: prefix + _join_segments(segs),
    _BOUNDED_PREFIX,
    st.lists(_PATH_SEGMENT, min_size=1, max_size=3),
).filter(_is_off_route)

# Final strategy: any of the three off-route shapes.
_OFF_ROUTE_PATH = st.one_of(_GENERAL_PATH, _NON_DIGIT_PROJECT_PATH, _BOUNDED_PATH)


# ---------------------------------------------------------------------------
# In-process ASGI driver
# ---------------------------------------------------------------------------


async def _drive_get(app: Starlette, path: str) -> httpx.Response:
    """Issue ``GET path`` against ``app`` over the in-process ASGI transport.

    Uses :class:`httpx.ASGITransport` so the request never touches a
    socket. The transport is constructed per call (rather than once
    per session) because it is cheap and because
    :class:`httpx.AsyncClient` closes the transport on ``__aexit__``,
    which makes a per-call transport the simplest correct lifetime —
    the same lifetime Property 21's driver uses.
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
    transport — the same pattern Properties 18, 19, 21, and 22 use.
    """
    return asyncio.run(_drive_get(app, path))


# ---------------------------------------------------------------------------
# Property 26
# ---------------------------------------------------------------------------


@given(path=_OFF_ROUTE_PATH)
@settings(max_examples=100)
def test_unknown_path_returns_404_with_path_in_body_property(path: str) -> None:
    """Property 26: any off-route GET → 404 + path-in-body.

    For every off-route path, the response satisfies four
    post-conditions drawn directly from Requirement 13.7:

    * the status code is 404,
    * the Content-Type is the documented :data:`HTML_CONTENT_TYPE`
      — Requirement 13.5 fixes the surface's Content-Type and the 404
      body shares it,
    * the body contains the documented
      :data:`NOT_FOUND_MESSAGE_TEMPLATE` message with the
      *escaped* form of the requested path interpolated (the renderer
      escapes via ``html.escape(path, quote=False)`` before
      formatting), and
    * the body contains the escaped path verbatim — used by an
      operator to identify which path was rejected.

    The escape comparison is what keeps the property honest under a
    wide alphabet: a regression that wrote the path raw into the body
    would break the page chrome on a path like ``/foo<script>`` and
    is caught by the substring assertion against the escaped form
    being absent.

    **Validates Requirement 13.7.**
    """
    app = build_visualization_app(catalog=_FakeCatalog(), store=_FakeStore())

    response = _get(app, path)

    assert response.status_code == 404, (
        f"off-route path {path!r} must produce HTTP 404 (Requirement 13.7)."
    )
    assert response.headers["content-type"] == HTML_CONTENT_TYPE, (
        "404 fallback must use the documented HTML Content-Type "
        "(Requirements 13.5, 13.7)."
    )

    body = response.text

    # The renderer interpolates ``html.escape(path, quote=False)``
    # into both the visible message and the ``<title>``. Comparing
    # against the escaped form keeps the property correct on paths
    # containing ``<``, ``>``, ``&``, ``"``, or ``'``.
    escaped_path = html.escape(path, quote=False)

    expected_message = NOT_FOUND_MESSAGE_TEMPLATE.format(path=escaped_path)
    assert expected_message in body, (
        "404 body must contain the documented "
        f"{NOT_FOUND_MESSAGE_TEMPLATE!r} message with the escaped form "
        f"{escaped_path!r} of the requested path interpolated "
        f"(Requirement 13.7); body was {body!r}."
    )
    assert "does not exist" in body, (
        "404 body must state that the requested page does not exist "
        f"(Requirement 13.7); body was {body!r}."
    )
    assert escaped_path in body, (
        "404 body must include the requested path verbatim (after the "
        "documented HTML escape) so an operator can identify which "
        f"path was rejected; expected {escaped_path!r} in body "
        f"(Requirement 13.7); body was {body!r}."
    )

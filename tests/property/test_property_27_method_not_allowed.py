# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 27: For all HTTP methods other than GET against any of the routes /, /projects/{digits}, /dependencies, /conflicts, the Visualization_Server SHALL respond with HTTP 405 and an Allow header whose value is exactly the string "GET".
"""Property test for the non-GET → 405 fallback (Property 27).

**Validates Requirement 13.8** (Property 27 in the design).

For every HTTP method other than ``GET`` issued against any of the four
documented Visualization_Server routes (``/``,
``/projects/{digit-only-id}``, ``/dependencies``, ``/conflicts``), the
server SHALL respond with HTTP ``405`` and an ``Allow`` response header
whose value is exactly the string ``"GET"`` (no commas, no spaces, no
``HEAD``). The response also carries the documented HTML
Content-Type (Requirement 13.5).

This property is implemented by the 405 branch of
:func:`project_knowledge_mcp.visualization_server._http_exception_handler`,
which Starlette dispatches when each documented :class:`Route` is
registered with ``methods={"GET"}`` (HEAD is removed in
:func:`build_visualization_app` so Starlette does not auto-widen
``methods=["GET"]`` to ``{"GET", "HEAD"}``). Hypothesis enumerates the
``(method, path)`` combinations within a single
``@settings(max_examples=100)`` run so a regression in any one of the
four routes — or in the handler's ``Allow`` header value — is caught
by this single test.

The handler is wired with duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the test does not depend on a real
SQLite database. The ``httpx.AsyncClient`` + ``httpx.ASGITransport``
pair drives the in-process ASGI app the same way Property 21 (and
Property 22) do — no socket bind, no event-loop juggling.
``asyncio.run`` is used inside the synchronous test body so
Hypothesis' ``@given`` composes cleanly with the async transport (the
same pattern Property 18, Property 19, Property 21, and Property 22
use in this suite).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
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
    """Minimal :class:`_CatalogReader` stand-in for the 405 property.

    The 405 fallback never reaches into the catalog (Starlette raises
    :class:`HTTPException(status_code=405)` from the routing layer
    before any handler is invoked) so the methods below are simple
    deterministic stubs whose return values do not affect the
    property's assertions. They exist only to satisfy the structural
    ``_CatalogReader`` Protocol so :func:`build_visualization_app` can
    wire all four routes.
    """

    def list_in_scope(self) -> list[InScopeProject]:
        return []

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        # The digit-only project_id used in the URL is irrelevant to
        # the 405 path: the routing layer rejects the non-GET method
        # before per-project handler logic runs. Returning ``False``
        # here is a defensive default that would still produce the
        # documented out-of-scope 404 if a regression somehow caused
        # the GET handler to run; the property's 405 assertions would
        # then fail loudly rather than silently.
        del gitlab_project_id
        return False


class _FakeStore:
    """Minimal :class:`_ProfileReader` stand-in for the 405 property.

    Like :class:`_FakeCatalog`, the 405 fallback never invokes any
    of these methods because the routing layer rejects the request
    before any GET handler runs. The methods exist purely to satisfy
    the structural ``_ProfileReader`` Protocol so the application
    can be built.
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

# The full set of non-GET HTTP methods the in-process httpx + Starlette
# transport accepts. POST, PUT, DELETE, PATCH, and OPTIONS are the
# methods a programmatic client or a browser-side script might send;
# HEAD is included because Starlette's ``Route`` constructor would
# normally auto-widen ``methods=["GET"]`` to ``{"GET", "HEAD"}`` and
# Requirement 13.8 explicitly forbids that — every non-GET method,
# *including HEAD*, must produce 405 with ``Allow: GET`` (no
# ``"GET, HEAD"``). ``GET`` is deliberately excluded; the GET handlers
# are validated by Properties 21-24.
_NON_GET_METHODS: tuple[str, ...] = (
    "POST",
    "PUT",
    "DELETE",
    "PATCH",
    "HEAD",
    "OPTIONS",
)
_NON_GET_METHOD_STRATEGY = st.sampled_from(_NON_GET_METHODS)

# GitLab project IDs are positive integers in the GitLab API; the
# upper bound of 1,000,000 keeps URLs short while still exercising
# multi-digit ids. ``str(int)`` over this range always yields a
# digit-only string that matches the ``[0-9]+`` regex behind
# Starlette's ``:int`` path converter, so the generated path is
# guaranteed to land on the per-project route's path pattern (the
# 405 fallback only fires for *path matches* that disagree on the
# method — non-digit project_id paths would fall through to the 404
# path validated by Property 26).
_PROJECT_ID_STRATEGY = st.integers(min_value=1, max_value=1_000_000)


@st.composite
def _path_strategy(draw: st.DrawFn) -> str:
    """Generate one of the four documented routes' paths.

    The four-way ``sampled_from`` ensures every documented route is
    exercised (a regression in any one route's ``methods={"GET"}``
    constraint would surface as a 200 / 404 / 405-without-Allow
    response on that route alone). The per-project branch draws a
    fresh digit-only id from :data:`_PROJECT_ID_STRATEGY` so the
    property exercises the ``[0-9]+`` path-converter regex with a
    range of widths (1, 2, 3, …, 7 digits) rather than only a fixed
    sentinel value.
    """
    branch = draw(
        st.sampled_from(("root", "per_project", "dependencies"))
    )
    if branch == "root":
        return "/"
    if branch == "per_project":
        return f"/projects/{draw(_PROJECT_ID_STRATEGY)}"
    if branch == "dependencies":
        return "/dependencies"
    raise AssertionError(f"unknown branch key: {branch!r}")


# ---------------------------------------------------------------------------
# In-process ASGI driver
# ---------------------------------------------------------------------------


async def _drive_request(
    app: Starlette, method: str, path: str
) -> httpx.Response:
    """Issue ``method path`` against ``app`` over the in-process ASGI transport.

    Uses :class:`httpx.ASGITransport` so the request never touches a
    socket. The transport is constructed per call (rather than once
    per session) because it is cheap and because :class:`httpx.AsyncClient`
    closes the transport on ``__aexit__``, which makes a per-call
    transport the simplest correct lifetime — the same pattern
    Property 21 and Property 22 use.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://visualization-server.test"
    ) as client:
        return await client.request(method, path)


def _request(app: Starlette, method: str, path: str) -> httpx.Response:
    """Synchronous wrapper around :func:`_drive_request`.

    ``@given`` does not compose with ``async def`` test bodies under
    pytest-asyncio's automatic mode, so the test function stays
    synchronous and uses :func:`asyncio.run` to drive the ASGI
    transport — the same pattern Property 18, Property 19, Property
    21, and Property 22 use.
    """
    return asyncio.run(_drive_request(app, method, path))


# ---------------------------------------------------------------------------
# Property 27
# ---------------------------------------------------------------------------


@given(method=_NON_GET_METHOD_STRATEGY, path=_path_strategy())
@settings(max_examples=100)
def test_non_get_methods_produce_405_with_allow_get_property(
    method: str, path: str,
) -> None:
    """Property 27: non-GET on any documented route → 405 + ``Allow: GET``.

    For every ``(method, path)`` combination drawn from the non-GET
    method set and the four documented routes, the Visualization_Server
    must produce:

    * ``response.status_code == 405``,
    * ``response.headers["allow"] == "GET"`` exactly (no commas, no
      spaces, no ``HEAD``),
    * ``response.headers["content-type"] == "text/html; charset=utf-8"``
      (Requirement 13.5; the 405 body is HTML so the Content-Type
      matches the rest of the surface).

    The fakes do not need to model any catalog or store state because
    Starlette's routing layer raises ``HTTPException(status_code=405)``
    *before* any GET handler runs; the 405 response is fully determined
    by the per-route ``methods`` constraint and by the
    :func:`_http_exception_handler` registered for status 405.

    **Validates Requirement 13.8.**
    """
    app = build_visualization_app(catalog=_FakeCatalog(), store=_FakeStore())

    response = _request(app, method, path)

    assert response.status_code == 405, (
        f"non-GET method {method!r} on documented route {path!r} must "
        f"produce HTTP 405 (Requirement 13.8); got {response.status_code}."
    )
    # The Allow header value MUST be exactly the string "GET"
    # (Requirement 13.8). No commas, no spaces, no HEAD; in
    # particular, Starlette's auto-HEAD-on-GET widening must remain
    # disabled by ``build_visualization_app``.
    assert response.headers["allow"] == "GET", (
        f"non-GET method {method!r} on documented route {path!r} must "
        f"produce an Allow header whose value is exactly the string "
        f'"GET" (Requirement 13.8); got '
        f"{response.headers['allow']!r}."
    )
    assert response.headers["content-type"] == HTML_CONTENT_TYPE, (
        f"405 response on documented route {path!r} must use the "
        f"documented HTML Content-Type {HTML_CONTENT_TYPE!r} "
        f"(Requirement 13.5); got "
        f"{response.headers['content-type']!r}."
    )

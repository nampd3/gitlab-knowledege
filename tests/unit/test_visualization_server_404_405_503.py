"""Unit tests for the 404 / 405 / 503 fallback handlers (task 10.8).

These tests pin the four behaviors documented by Requirements 13.5,
13.7, 13.8, and 14.6:

* **404 fallback** — A request for any path that is not ``/``,
  ``/projects/{digits}``, ``/dependencies``, or ``/conflicts`` returns
  HTTP 404 with an HTML body that contains the requested HTTP path
  verbatim and states the page does not exist (Requirement 13.7).
* **405 fallback** — A non-GET request (POST, PUT, DELETE, PATCH,
  HEAD, ...) on each of the four documented routes returns HTTP 405
  with an ``Allow`` header whose value is exactly the string ``"GET"``
  (Requirement 13.8).
* **503 store-unavailable fallback** — When a handler raises
  :class:`KnowledgeStoreUnavailableError`, the response is HTTP 503
  with an HTML body stating that project knowledge is temporarily
  unavailable. No cached or in-memory profile data is served as a
  fallback (Requirement 14.6).
* **Content-Type** — All HTML responses (including the three above)
  use ``text/html; charset=utf-8`` (Requirement 13.5).

The handlers are wired with duck-typed fakes that satisfy the
``_CatalogReader`` and ``_ProfileReader`` Protocols defined in
``visualization_server.py`` so the tests do not depend on a real
SQLite database.

Implements Requirements 13.5, 13.7, 13.8, 14.6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from project_knowledge_mcp.errors import KnowledgeStoreUnavailableError
from project_knowledge_mcp.visualization_server import (
    HTML_CONTENT_TYPE,
    METHOD_NOT_ALLOWED_MESSAGE,
    STORE_UNAVAILABLE_MESSAGE,
    build_visualization_app,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from project_knowledge_mcp.models import ProjectProfile
    from project_knowledge_mcp.project_catalog import InScopeProject


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _request(
    app: Starlette, method: str, path: str
) -> httpx.Response:
    """Issue ``method path`` against ``app`` over the in-process ASGI transport.

    Mirrors the helpers in the other Visualization_Server unit tests:
    uses :class:`httpx.ASGITransport` so the test does not need a real
    socket bind and so the suite-wide ``filterwarnings = ["error"]``
    rule does not turn the deprecated
    ``starlette.testclient.TestClient`` warning into a collection
    error.
    """
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://visualization-server.test"
    ) as client:
        return await client.request(method, path)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCatalog:
    """Minimal :class:`_CatalogReader` stand-in for the fallback tests.

    The 404 / 405 fallbacks never reach into the catalog, so the
    methods below are simple stubs with deterministic return values.
    The 503 test injects ``raise_on_calls=True`` to make every catalog
    read raise :class:`KnowledgeStoreUnavailableError` so the route
    handler hits the documented exception path.
    """

    def __init__(
        self, *, in_scope_ids: set[int] | None = None, raise_on_calls: bool = False
    ) -> None:
        self._in_scope_ids = set(in_scope_ids) if in_scope_ids is not None else set()
        self._raise_on_calls = raise_on_calls

    def list_in_scope(self) -> list[InScopeProject]:
        if self._raise_on_calls:
            raise KnowledgeStoreUnavailableError("catalog read failed")
        return []

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        if self._raise_on_calls:
            raise KnowledgeStoreUnavailableError("catalog read failed")
        return gitlab_project_id in self._in_scope_ids


class _FakeStore:
    """Minimal :class:`_ProfileReader` stand-in for the fallback tests.

    Holds a ``snapshot_id`` (default ``None``) and a
    ``raise_on_calls`` flag. When ``raise_on_calls`` is set, every
    reader method raises :class:`KnowledgeStoreUnavailableError` so
    the route handler hits the documented exception path. Otherwise
    the reader methods return safe default values; the 404 / 405
    tests never reach them.
    """

    def __init__(
        self, *, snapshot_id: int | None = None, raise_on_calls: bool = False
    ) -> None:
        self._snapshot_id = snapshot_id
        self._raise_on_calls = raise_on_calls

    def get_current_snapshot_id(self) -> int | None:
        if self._raise_on_calls:
            raise KnowledgeStoreUnavailableError("snapshot pointer read failed")
        return self._snapshot_id

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        del gitlab_project_id
        if self._raise_on_calls:
            raise KnowledgeStoreUnavailableError("profile read failed")
        return None

    def list_profiles(self) -> list[ProjectProfile]:
        if self._raise_on_calls:
            raise KnowledgeStoreUnavailableError("profile list read failed")
        return []


# ---------------------------------------------------------------------------
# 404 fallback (Requirement 13.7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "requested_path",
    [
        "/not/a/route",
        "/projects/abc",  # non-digit project id falls through to the 404 fallback
        "/static/styles.css",
        "/projects",  # bare /projects is not a registered route
    ],
)
async def test_404_returns_html_with_requested_path_verbatim(
    requested_path: str,
) -> None:
    """Non-matching path → HTTP 404 + HTML body that includes the path verbatim.

    Verifies Requirement 13.7 across four representative path shapes:
    a deeply-nested miss, a digit-shaped id with non-digit characters
    (``/projects/abc``), a static-asset-style miss, and the bare
    ``/projects`` prefix without a project id. All four must produce
    the documented HTML body.
    """
    app = build_visualization_app(catalog=_FakeCatalog(), store=_FakeStore())

    response = await _request(app, "GET", requested_path)

    assert response.status_code == 404
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    # The requested path appears verbatim in the body so an operator
    # (or a test) can identify which path was rejected.
    assert requested_path in body
    # The body states the page does not exist (Requirement 13.7).
    assert "does not exist" in body


# ---------------------------------------------------------------------------
# 405 fallback (Requirement 13.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    ["POST", "PUT", "DELETE", "PATCH", "HEAD"],
)
@pytest.mark.parametrize(
    "path",
    ["/", "/projects/123", "/dependencies"],
)
async def test_405_returns_allow_get_header_on_documented_routes(
    method: str, path: str
) -> None:
    """Non-GET method on each documented route → 405 + ``Allow: GET``.

    Drives every (method, path) combination across the five non-GET
    methods that a browser or programmatic client might send and the
    four documented routes. Each combination must produce HTTP 405
    with an ``Allow`` header whose value is exactly the string
    ``"GET"`` (Requirement 13.8) and the documented HTML
    Content-Type (Requirement 13.5). HEAD is included to assert that
    Starlette's automatic-HEAD-on-GET behavior is suppressed; the
    documented Allow value forbids ``"GET, HEAD"``.
    """
    app = build_visualization_app(
        catalog=_FakeCatalog(in_scope_ids={123}),
        store=_FakeStore(snapshot_id=1),
    )

    response = await _request(app, method, path)

    assert response.status_code == 405
    # The Allow header value MUST be exactly the string "GET"
    # (Requirement 13.8). No commas, no spaces, no HEAD.
    assert response.headers["allow"] == "GET"
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    # HEAD responses by HTTP convention have no body, so only assert
    # the body's content for the methods that carry one.
    if method != "HEAD":
        assert METHOD_NOT_ALLOWED_MESSAGE in response.text


# ---------------------------------------------------------------------------
# 503 fallback (Requirement 14.6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/", "/projects/123", "/dependencies"],
)
async def test_503_when_store_raises_knowledge_store_unavailable(path: str) -> None:
    """Store raises ``KnowledgeStoreUnavailableError`` → 503 + documented body.

    Verifies Requirement 14.6 for every documented route: when the
    injected store raises :class:`KnowledgeStoreUnavailableError`
    while serving the request, the response is HTTP 503 with the
    documented HTML body and the documented Content-Type. The body
    is fully derived from a static template — no cached or
    in-memory profile data is served as a fallback.
    """
    app = build_visualization_app(
        catalog=_FakeCatalog(in_scope_ids={123}, raise_on_calls=True),
        store=_FakeStore(snapshot_id=1, raise_on_calls=True),
    )

    response = await _request(app, "GET", path)

    assert response.status_code == 503
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    assert STORE_UNAVAILABLE_MESSAGE in body


async def test_503_body_contains_no_cached_profile_data() -> None:
    """503 body never contains profile-derived content (Requirement 14.6).

    Even if the catalog returns in-scope ids and the store nominally
    has a snapshot id, raising :class:`KnowledgeStoreUnavailableError`
    must short-circuit the render path so no profile-derived
    fragment leaks into the body. This test asserts the negative
    case: the 503 body contains no diagram class names, no
    in-scope project ids, and no full paths from the (fake) catalog.
    """
    app = build_visualization_app(
        catalog=_FakeCatalog(in_scope_ids={42}, raise_on_calls=True),
        store=_FakeStore(snapshot_id=99, raise_on_calls=True),
    )

    response = await _request(app, "GET", "/")

    assert response.status_code == 503
    assert response.headers["content-type"] == HTML_CONTENT_TYPE
    body = response.text
    # No diagram fragments leak into the 503 body. Any of these
    # appearing would imply a render path was reached despite the
    # store raising.
    assert "project-profile-diagram" not in body
    assert "dependency-graph-diagram" not in body
    assert "conflict-overview-diagram" not in body
    assert "project-list" not in body
    # The fake catalog's in-scope id MUST NOT bleed into the body.
    assert "42" not in body

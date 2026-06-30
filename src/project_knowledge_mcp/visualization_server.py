"""Visualization_Server: loopback-only HTTP application skeleton.

This module wires the Starlette + Uvicorn skeleton on top of which the
Visualization_Server's four diagram routes (tasks 10.2-10.7), the 404 / 405 /
503 fallbacks (task 10.8), and the renderer plumbing live. The skeleton
itself is responsible for three behaviors that the design treats as
process-level invariants regardless of which routes are registered:

* **Loopback-only binding.** The server is bound to ``127.0.0.1`` and
  ``::1`` simultaneously and never to any other interface address. Both
  sockets are bound up-front (before Uvicorn is started) so that bind
  failures are surfaced as the documented start-up errors and never as
  runtime failures observable by an MCP client (Requirement 12.2).

* **Request-handler-level deadline.** Every request passes through
  :class:`_DeadlineMiddleware`, which fails the handler after
  :data:`HANDLER_DEADLINE_SECONDS` seconds and replies with HTTP 503 so
  the response *begins* within five seconds at the HTTP layer
  (Requirement 13.9). This is enforced as middleware, not as a per-route
  concern, because the requirement applies to every route.

* **Documented start-up errors.** When the bind fails because the
  configured port is in use, this module emits exactly
  ``"startup error: visualization.port {port} is already in use"`` to
  stderr and exits non-zero (Requirement 12.6). When the bind fails for
  any other reason, this module emits exactly
  ``"startup error: visualization server failed to start: {os_error}"``
  and exits non-zero (Requirement 12.8). On a successful bind, it emits
  the ``"Visualization_Server ready at http://127.0.0.1:{port}"`` log
  line described by Requirement 12.7. The wording of all three lines is
  fixed by the design's *Startup errors* table and is reproduced
  verbatim from this module's constants so a regression in the wording
  is caught by the unit tests in tasks 10.9, 10.10, 10.11.

The module deliberately does **not** start the event loop on import:
:func:`run` is the caller-driven entry point that ``main.py`` (task 11.1)
invokes after :func:`config.load_and_validate` succeeds. ``main.py``
remains free to compose the ready log line, the MCP stdio bind, and the
scheduler start; this module only owns the visualization surface.

Implements Requirements 12.2, 12.3, 12.4, 12.6, 12.7, 12.8, 13.9.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import html
import logging
import socket
import sys
from typing import TYPE_CHECKING, Final, Protocol

import uvicorn
from pathlib import Path
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import HTMLResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .diagram_renderer import (
    NO_SNAPSHOT_MESSAGE,
    render_dependency_graph,
    render_index_page,
    render_project_not_in_scope_page,
    render_project_not_yet_analyzed_page,
    render_project_profile_page,
)
from .errors import BindError, KnowledgeStoreUnavailableError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, TextIO

    from starlette.requests import Request
    from starlette.routing import BaseRoute
    from starlette.types import ASGIApp, ExceptionHandler

    from .project_catalog import InScopeProject


# ---------------------------------------------------------------------------
# Loopback binding constants (Requirement 12.2)
# ---------------------------------------------------------------------------

#: IPv4 loopback address; the only IPv4 address the Visualization_Server
#: ever accepts connections on (Requirement 12.2).
LOOPBACK_IPV4: Final[str] = "127.0.0.1"

#: IPv6 loopback address; the only IPv6 address the Visualization_Server
#: ever accepts connections on (Requirement 12.2).
LOOPBACK_IPV6: Final[str] = "::1"


# ---------------------------------------------------------------------------
# Request-handler-level deadline (Requirement 13.9)
# ---------------------------------------------------------------------------

#: Maximum time, in seconds, between the HTTP layer receiving the request
#: and the response *beginning* to be written (Requirement 13.9). When a
#: handler exceeds this deadline, :class:`_DeadlineMiddleware` cancels it
#: and returns a 503 response so the client always sees a response start
#: within this bound.
HANDLER_DEADLINE_SECONDS: Final[float] = 5.0


# ---------------------------------------------------------------------------
# Stable HTML/Content-Type strings used by the skeleton (Requirement 13.5)
# ---------------------------------------------------------------------------

#: The fixed Content-Type for every HTML response from this surface
#: (Requirement 13.5). Defined here so the skeleton's fallback
#: responses use the same value as the future per-route handlers.
HTML_CONTENT_TYPE: Final[str] = "text/html; charset=utf-8"

#: HTML body served when :class:`_DeadlineMiddleware` cancels a handler
#: that overran :data:`HANDLER_DEADLINE_SECONDS`. The body intentionally
#: avoids any reference to persisted ``Project_Profile`` data so it can
#: also be served during start-up before the store is wired in.
_DEADLINE_EXCEEDED_HTML: Final[str] = (
    "<!doctype html>"
    "<html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<title>Visualization_Server: response deadline exceeded</title></head>"
    "<body><h1>503 Service Unavailable</h1>"
    "<p>The Visualization_Server did not produce a response within "
    "the configured handler deadline.</p>"
    "</body></html>"
)


# ---------------------------------------------------------------------------
# Fallback handler bodies (Requirements 13.5, 13.7, 13.8, 14.6)
# ---------------------------------------------------------------------------
#
# Task 10.8 adds three fallback responses: 404 for any path that does not
# match one of the four registered routes (Requirement 13.7), 405 for any
# non-GET method on those four routes (Requirement 13.8), and 503 when a
# handler raises :class:`KnowledgeStoreUnavailableError` (Requirement
# 14.6). All three responses share the documented
# ``text/html; charset=utf-8`` Content-Type (Requirement 13.5) so this
# module owns the wording and structure of the bodies in one place.
#
# The 404 body is built from a template at request time because it must
# *include the requested HTTP path verbatim* (Requirement 13.7) so an
# operator can tell which path was rejected. The 405 and 503 bodies are
# fixed strings — neither depends on per-request inputs — and are pinned
# here so a regression in the wording is caught by the unit tests for
# this task.

#: Visible message line embedded in every 404 body. The ``{path}``
#: placeholder is filled (after HTML-escaping) with the requested HTTP
#: path so the operator sees which path was rejected. Pinned to the
#: design's "the requested page does not exist" wording.
NOT_FOUND_MESSAGE_TEMPLATE: Final[str] = (
    "The requested page {path} does not exist."
)

#: Visible message line embedded in every 405 body. The wording is
#: deliberately short so callers can substring-match without coupling
#: to the surrounding HTML chrome.
METHOD_NOT_ALLOWED_MESSAGE: Final[str] = (
    "Method not allowed; this route only accepts GET."
)

#: Visible message line embedded in every 503 body served because a
#: handler raised :class:`KnowledgeStoreUnavailableError`
#: (Requirement 14.6). Distinct from the deadline-503 wording above
#: so the two 503 paths remain distinguishable from the response body
#: alone (Requirement 13.9 vs Requirement 14.6).
STORE_UNAVAILABLE_MESSAGE: Final[str] = (
    "Project knowledge is temporarily unavailable."
)

#: Status code for the 404 fallback handler. Named so the
#: ``isinstance`` dispatch in :func:`_http_exception_handler` does not
#: trip ``ruff PLR2004`` ("magic value in comparison").
_HTTP_STATUS_NOT_FOUND: Final[int] = 404

#: Status code for the 405 fallback handler. Named for the same
#: reason as :data:`_HTTP_STATUS_NOT_FOUND`.
_HTTP_STATUS_METHOD_NOT_ALLOWED: Final[int] = 405

#: Status code returned by the store-unavailable handler.
_HTTP_STATUS_SERVICE_UNAVAILABLE: Final[int] = 503

#: Fixed full-HTML body served by the 405 fallback handler. The page
#: chrome mirrors the other Visualization_Server pages
#: (``<!doctype html>``, ``<head>``, ``<body>`` with a back-link to
#: ``/``) so an operator landing on the route via a non-GET request
#: can still navigate back to the index.
_METHOD_NOT_ALLOWED_HTML: Final[str] = (
    "<!doctype html>"
    '<html lang="en">'
    '<head><meta charset="utf-8">'
    "<title>Visualization_Server: 405 Method Not Allowed</title>"
    "</head>"
    "<body>"
    '<nav class="page-nav"><a class="page-nav__index-link" href="/">Index</a></nav>'
    "<main>"
    "<h1>405 Method Not Allowed</h1>"
    f'<p class="empty-state empty-state--method-not-allowed">'
    f"{html.escape(METHOD_NOT_ALLOWED_MESSAGE)}</p>"
    "</main>"
    "</body>"
    "</html>"
)

#: Fixed full-HTML body served when a handler raises
#: :class:`KnowledgeStoreUnavailableError`. Distinct from
#: :data:`_DEADLINE_EXCEEDED_HTML` so the two 503 surfaces (timeout vs
#: unavailable store) remain distinguishable from the response body
#: alone, even though both share status code 503 and the documented
#: HTML Content-Type.
_STORE_UNAVAILABLE_HTML: Final[str] = (
    "<!doctype html>"
    '<html lang="en">'
    '<head><meta charset="utf-8">'
    "<title>Visualization_Server: 503 project knowledge unavailable</title>"
    "</head>"
    "<body>"
    "<main>"
    "<h1>503 Service Unavailable</h1>"
    f'<p class="empty-state empty-state--store-unavailable">'
    f"{html.escape(STORE_UNAVAILABLE_MESSAGE)}</p>"
    "</main>"
    "</body>"
    "</html>"
)


def _render_not_found_html(requested_path: str) -> str:
    """Render the HTML body for a 404 response.

    The body must *include the requested HTTP path verbatim*
    (Requirement 13.7) so the operator (or a test) can identify which
    path was rejected. ``requested_path`` is passed through
    :func:`html.escape` before being interpolated into the page so a
    pathological path containing ``<script>`` tags cannot break out of
    the page chrome. The escape preserves the visible glyphs of the
    path (slashes, digits, ASCII letters) so the rendered HTML still
    shows the path "verbatim" to an operator viewing it in a browser.

    Implements Requirement 13.7.
    """
    escaped_path = html.escape(requested_path, quote=False)
    message = NOT_FOUND_MESSAGE_TEMPLATE.format(path=escaped_path)
    return (
        "<!doctype html>"
        '<html lang="en">'
        '<head><meta charset="utf-8">'
        f"<title>Visualization_Server: 404 {escaped_path} not found</title>"
        "</head>"
        "<body>"
        '<nav class="page-nav"><a class="page-nav__index-link" href="/">Index</a></nav>'
        "<main>"
        "<h1>404 Not Found</h1>"
        f'<p class="empty-state empty-state--not-found" '
        f'data-requested-path="{html.escape(requested_path)}">{message}</p>'
        "</main>"
        "</body>"
        "</html>"
    )


# ---------------------------------------------------------------------------
# Documented start-up log / error lines (Requirements 12.6, 12.7, 12.8)
# ---------------------------------------------------------------------------

#: Format string for the ``"ready"`` log line (Requirement 12.7). The
#: ``{port}`` placeholder is filled with the bound TCP port. The wording
#: ("Visualization_Server ready at http://127.0.0.1:{port}") is fixed by
#: the design and the task 10.10 unit test pins it verbatim.
READY_LOG_TEMPLATE: Final[str] = "Visualization_Server ready at http://127.0.0.1:{port}"

#: Format string for the port-already-in-use start-up error
#: (Requirement 12.6). The wording is fixed by the design's *Startup errors*
#: table and the task 10.9 unit test pins it verbatim.
PORT_IN_USE_TEMPLATE: Final[str] = (
    "startup error: visualization.port {port} is already in use"
)

#: Format string for any non-EADDRINUSE bind failure (Requirement 12.8).
#: ``{os_error}`` is filled with the underlying OS error string so the
#: operator sees the failure reason reported by the operating system or
#: HTTP runtime. The task 10.11 unit test pins this wording verbatim.
GENERIC_BIND_FAILURE_TEMPLATE: Final[str] = (
    "startup error: visualization server failed to start: {os_error}"
)


# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

#: Logger used to emit the ``"ready"`` line. A dedicated, named logger
#: makes the line easy to capture in tests via :class:`logging.handlers`
#: and keeps unrelated Uvicorn / Starlette log records routed elsewhere.
_LOG: Final[logging.Logger] = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deadline middleware (Requirement 13.9)
# ---------------------------------------------------------------------------


class _DeadlineMiddleware(BaseHTTPMiddleware):
    """Cancel any handler that exceeds the configured response deadline.

    The Visualization_Server's response-time guarantee (Requirement 13.9)
    applies to every route. Implementing it as middleware keeps the per-
    route handlers free of timeout plumbing and ensures that even routes
    added by future tasks inherit the bound automatically.

    On timeout the middleware cancels the in-flight handler task (which
    :func:`asyncio.wait_for` does for free) and returns a 503 HTML
    response. The 503 body deliberately contains no ``Project_Profile``-
    derived content; it is generated entirely from a static string so it
    can also be served before the ``Knowledge_Store`` is wired up.
    """

    def __init__(
        self,
        app: ASGIApp,
        deadline_seconds: float = HANDLER_DEADLINE_SECONDS,
    ) -> None:
        super().__init__(app)
        self._deadline_seconds = deadline_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await asyncio.wait_for(
                call_next(request), timeout=self._deadline_seconds
            )
        except TimeoutError:
            return HTMLResponse(
                _DEADLINE_EXCEEDED_HTML,
                status_code=503,
                media_type=HTML_CONTENT_TYPE,
            )


# ---------------------------------------------------------------------------
# Application skeleton (extension point for tasks 10.2-10.8)
# ---------------------------------------------------------------------------


def build_app(
    routes: Sequence[BaseRoute] | None = None,
    *,
    exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
) -> Starlette:
    """Construct the Starlette application skeleton.

    The skeleton wires :class:`_DeadlineMiddleware` once and accepts a
    sequence of :class:`starlette.routing.BaseRoute` objects so that
    tasks 10.5-10.7 can register their per-route handlers without
    re-wiring middleware. Passing ``None`` (the default) yields an
    application with no routes; every request will then fall through to
    Starlette's default 404 handler, which the task 10.8 fallback
    handlers will replace.

    ``exception_handlers`` is forwarded verbatim to the Starlette
    constructor so :func:`build_visualization_app` (task 10.8) can
    register its 404, 405, and ``KnowledgeStoreUnavailableError``
    handlers in a single place. The ``ExceptionHandler`` value type
    matches Starlette's own — async or sync ``(Request, Exception) ->
    Response`` callables — and may be keyed by either an integer status
    code or an exception class.
    """
    return Starlette(
        routes=list(routes) if routes is not None else [],
        middleware=[Middleware(_DeadlineMiddleware)],
        exception_handlers=(
            dict(exception_handlers) if exception_handlers is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Catalog / store reader Protocols (task 10.5+)
# ---------------------------------------------------------------------------
#
# Every diagram route reads from the ``Project_Catalog`` and
# ``Knowledge_Store`` *at request time* (Requirement 14.1) so the responses
# always reflect the latest committed snapshot. The route factories take
# those collaborators as arguments rather than touching globals so the unit
# tests in ``tests/unit/`` can wire in lightweight fakes that return
# canned in-scope lists and snapshot ids without spinning up a real SQLite
# database. The Protocols below capture exactly the methods each route
# handler calls, so a fake only needs to implement those methods.

class _CatalogReader(Protocol):
    """Subset of :class:`~project_knowledge_mcp.project_catalog.ProjectCatalog`.

    The index handler calls :meth:`list_in_scope`; the per-project
    handler (task 10.6) calls :meth:`is_in_scope` to decide between the
    in-scope branches (200 with a profile, or 200 with the
    not-yet-analyzed message) and the out-of-scope branch (HTTP 404).
    The Protocol exposes both methods so a single ``ProjectCatalog``
    instance can satisfy every handler in the application.
    """

    def list_in_scope(self) -> list[InScopeProject]:  # pragma: no cover - protocol
        ...

    def is_in_scope(self, gitlab_project_id: int) -> bool:  # pragma: no cover - protocol
        ...


class _SnapshotReader(Protocol):
    """Subset of :class:`~project_knowledge_mcp.knowledge_store.KnowledgeStore`.

    The index handler only needs the current-snapshot pointer to decide
    between the no-snapshot empty-state branch (Requirement 14.4) and
    the populated / empty-catalog branches (Requirement 13.1). It never
    reads profiles directly; the per-project handler added in task 10.6
    relies on the richer :class:`_ProfileReader` Protocol for
    ``get_profile``.
    """

    def get_current_snapshot_id(self) -> int | None:  # pragma: no cover - protocol
        ...


class _ProfileReader(_SnapshotReader, Protocol):
    """Superset of :class:`_SnapshotReader` adding ``get_profile`` / ``list_profiles``.

    The per-project handler (task 10.6) needs to look up a single
    ``Project_Profile`` by ``gitlab_project_id`` from the current
    snapshot. ``KnowledgeStore.get_profile`` already filters by
    ``current_snapshot.snapshot_id`` internally (Property 11), so the
    handler does not need to consult ``get_current_snapshot_id``
    itself; it only needs to ask "is there a profile for this id under
    the snapshot the reader sees right now?".

    The ``/dependencies`` handler calls :meth:`list_profiles` instead —
    it needs every profile in the current snapshot to compute the
    ``Dependency_Graph_Diagram``. ``list_profiles`` likewise consults
    ``current_snapshot.snapshot_id`` internally (Property 11) so the
    handlers do not have to. Both methods read at
    request time with no in-memory caching, satisfying Requirement 14.1.

    Inheriting from :class:`_SnapshotReader` keeps the per-project
    handler's Protocol a strict superset of the index handler's, so
    one ``KnowledgeStore`` instance can satisfy both at the same time.
    """

    def get_profile(  # pragma: no cover - protocol
        self, gitlab_project_id: int
    ) -> ProjectProfile | None:
        ...

    def list_profiles(self) -> list[ProjectProfile]:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# GET / index handler (task 10.5)
# ---------------------------------------------------------------------------


def _build_index_route(
    catalog: _CatalogReader,
    store: _SnapshotReader,
) -> Route:
    """Return the ``Route`` for ``GET /``.

    The route handler is constructed as a closure over ``catalog`` and
    ``store`` so each request reads from the live ``Project_Catalog``
    and ``Knowledge_Store`` rather than from a cached snapshot
    (Requirement 14.1). Sorting by ``gitlab_project_id`` ascending is
    performed defensively here even though
    :meth:`ProjectCatalog.list_in_scope` already returns sorted rows;
    the redundancy keeps the contract local to the handler so that a
    test using a fake catalog returning unsorted data still observes
    the documented order (Requirement 13.1).

    The handler distinguishes the three branches in the order required
    by Property 21: the no-snapshot branch takes priority (Requirement
    14.4), then the empty-catalog branch (Requirement 13.1), then the
    populated branch.

    Implements Requirements 13.1, 14.4.
    """

    async def index(request: Request) -> Response:
        # Unused; included for the Starlette signature contract. The
        # handler intentionally has no per-request input — the index
        # body is fully determined by the catalog + snapshot state.
        del request

        snapshot_id = store.get_current_snapshot_id()
        if snapshot_id is None:
            # Requirement 14.4: no Ingestion_Job has ever completed,
            # so the response must omit any per-project list, any
            # diagram links, and any other Project_Profile-derived
            # content. ``render_index_page`` enforces this by branching
            # on ``has_snapshot`` before consulting ``in_scope_projects``.
            entries: list[InScopeProject] = []
            has_snapshot = False
        else:
            # Requirement 13.1: list every in-scope project ordered by
            # GitLab project ID ascending. Sort here (rather than rely
            # on the catalog) so the contract is local to the handler;
            # ``ProjectCatalog.list_in_scope`` already sorts, so this
            # is a no-op cost in production.
            entries = sorted(
                catalog.list_in_scope(),
                key=lambda project: project.gitlab_project_id,
            )
            has_snapshot = True

        body = render_index_page(
            in_scope_projects=entries,
            has_snapshot=has_snapshot,
        )
        return HTMLResponse(body, media_type=HTML_CONTENT_TYPE)

    return Route("/", index, methods=["GET"])


# ---------------------------------------------------------------------------
# GET /projects/{project_id} per-project handler (task 10.6)
# ---------------------------------------------------------------------------


def _build_per_project_route(
    catalog: _CatalogReader,
    store: _ProfileReader,
) -> Route:
    """Return the ``Route`` for ``GET /projects/{project_id}``.

    The route uses Starlette's built-in ``int`` path converter
    (``{project_id:int}``) whose underlying regex is ``[0-9]+`` —
    matching exactly the *digit-only* shape required by Requirements
    13.2 and 13.6 — and converting the captured group to a Python
    ``int`` so the handler can pass it straight to
    :meth:`ProjectCatalog.is_in_scope` and
    :meth:`KnowledgeStore.get_profile`. Non-digit paths (for example
    ``/projects/abc``) do not match this route at all and fall through
    to the 404 fallback that task 10.8 will register; that separation
    is what keeps Requirements 13.6 (digit-only "not in scope" 404)
    and 13.7 (any other path 404) from collapsing into a single
    branch.

    The handler walks the three branches mandated by Property 22 in
    the order required by the spec:

    1. **Out of scope** — :meth:`is_in_scope` returns ``False``. The
       handler responds with HTTP 404 and a body that names the
       requested ``project_id`` (Requirements 13.6, 14.5).
    2. **In scope but no profile persisted** —
       :meth:`get_profile` returns ``None`` (the project was
       enumerated into the catalog but has not been analyzed yet,
       e.g. its ``Analysis_Branch`` was missing per Requirement 15.5,
       or the snapshot was committed before per-project analysis ran).
       The handler responds with HTTP 200 and the documented
       "Project has not yet been analyzed; run an Ingestion_Job"
       message and *no* diagram (Requirement 14.3).
    3. **In scope and profile persisted** — the handler responds with
       HTTP 200 and the rendered ``Project_Profile_Diagram`` page
       (Requirements 13.2, 13.6).

    Both reads (``is_in_scope`` and ``get_profile``) happen at request
    time on the closures' bound ``catalog`` and ``store``; nothing is
    cached in process between requests, satisfying Requirement 14.1's
    "no in-memory caching of profile data" rule.

    Implements Requirements 13.2, 13.6, 14.3, 14.5.
    """

    async def per_project(request: Request) -> Response:
        # Starlette's ``:int`` converter has already validated the
        # digit-only regex at routing time, so ``project_id`` is always
        # a non-negative ``int`` here.
        project_id = int(request.path_params["project_id"])

        # 1) Out-of-scope branch (Requirements 13.6, 14.5). Checked
        #    first so an out-of-scope id does not leak the existence of
        #    a (potentially in-scope) profile under it.
        if not catalog.is_in_scope(project_id):
            body = render_project_not_in_scope_page(project_id=project_id)
            return HTMLResponse(
                body, status_code=404, media_type=HTML_CONTENT_TYPE
            )

        # 2 & 3) In-scope branches. ``get_profile`` consults the
        #    current snapshot internally (Property 11), so the read is
        #    snapshot-isolated without the handler having to re-read
        #    ``get_current_snapshot_id`` itself.
        profile = store.get_profile(project_id)
        if profile is None:
            body = render_project_not_yet_analyzed_page(project_id=project_id)
            return HTMLResponse(
                body, status_code=200, media_type=HTML_CONTENT_TYPE
            )

        body = render_project_profile_page(profile)
        return HTMLResponse(body, status_code=200, media_type=HTML_CONTENT_TYPE)

    return Route("/projects/{project_id:int}", per_project, methods=["GET"])


# ---------------------------------------------------------------------------
# GET /dependencies handler
# ---------------------------------------------------------------------------
#
# Both diagram routes share the same two structural concerns:
#
#   * The no-snapshot branch (Requirement 14.4): when no Ingestion_Job has
#     ever completed, the response must contain the documented
#     :data:`NO_SNAPSHOT_MESSAGE` and *no* diagram content. This branch
#     is checked first — before any other read against the catalog or
#     store — so the request never observes diagram-derived data while
#     a snapshot is still missing.
#   * The populated branch (Requirements 13.3 / 13.4): when a snapshot
#     exists, each handler reads the catalog and the profile list at
#     request time (Requirement 14.1: "no in-memory caching of profile
#     data"), feeds the raw inputs through the corresponding pure
#     renderer in ``diagram_renderer.py``, and wraps the resulting
#     ``<section>`` fragment in a minimal full-HTML envelope.
#
# The fragment renderers are pure and already escape every user-
# controllable value through the shared Jinja2 environment (see
# :mod:`project_knowledge_mcp.diagram_renderer`); the envelope below
# only contributes static markup of its own, so embedding the
# rendered fragment via plain string concatenation does not bypass
# the trust boundary.

#: Fixed full-HTML body served by the no-snapshot branch of
#: ``GET /dependencies`` (Requirement 14.4).
#: The body wraps :data:`NO_SNAPSHOT_MESSAGE` in a minimal page chrome
#: with a back-link to ``/`` so an operator landing on the route via a
#: bookmark can still navigate to the index. The message is
#: HTML-escaped defensively even though its current value contains no
#: special characters; pinning the escape here keeps the constant
#: stable if the literal in :data:`NO_SNAPSHOT_MESSAGE` is ever
#: extended with characters that would need escaping.
_NO_SNAPSHOT_HTML: Final[str] = (
    "<!doctype html>"
    '<html lang="en">'
    '<head><meta charset="utf-8">'
    "<title>Project Knowledge — no project knowledge available</title>"
    "</head>"
    "<body>"
    '<nav class="page-nav"><a class="page-nav__index-link" href="/">Index</a></nav>'
    "<main>"
    f'<p class="empty-state empty-state--no-snapshot">{html.escape(NO_SNAPSHOT_MESSAGE)}</p>'
    "</main>"
    "</body>"
    "</html>"
)


def _wrap_diagram_fragment(*, title: str, fragment: str) -> str:
    """Wrap a rendered diagram ``<section>`` in a full HTML page.

    Mirrors the chrome rendered by
    :func:`render_project_profile_page` (``<!doctype html>``,
    ``<head>``, ``<nav>`` back-link to ``/``, ``<main>`` containing the
    diagram) so the three diagram-page routes feel like one application
    rather than three disjoint pages.

    The page includes a ``<script src="/static/mermaid.min.js">`` tag
    followed by a single ``mermaid.initialize`` call. The script is
    served from the package's ``static/`` directory (see
    :data:`_STATIC_DIR_PATH` and ``static/README.md``). When the file
    is not present, the browser logs a 404, ``mermaid`` is undefined,
    and the page degrades to showing the raw ``graph LR`` source
    inside the ``<pre class="mermaid">`` blocks — exactly the
    behaviour observed before this script tag was added. Dropping a
    Mermaid release into ``static/mermaid.min.js`` is enough to turn
    the text back into rendered SVG diagrams without any code change.

    The ``mermaid.initialize`` call bumps Mermaid's built-in
    ``maxTextSize`` and ``maxEdges`` ceilings well above the public
    Live Editor defaults. Those defaults (50 KB of source, 500
    edges) are easily exceeded by a realistic ESB-scale catalog: a
    dependency graph for 100+ projects already overruns
    Mermaid's edge cap, producing the in-page error
    ``Maximum text size in diagram exceeded``. The bumped values
    here (1 MiB source, 100k edges) still leave a real safety net
    for pathological inputs while letting any realistic snapshot
    render.

    ``fragment`` is expected to be the ``<section>``-shaped output of
    one of the pure renderers in ``diagram_renderer.py``; those
    renderers already pass every user-controllable value through the
    shared Jinja2 environment's ``autoescape=True`` pipeline, so
    embedding the fragment via plain string concatenation does not
    bypass the trust boundary. ``title`` is HTML-escaped here because
    it could in principle vary per call (today it is a fixed literal,
    but pinning the escape avoids a regression if a future caller
    interpolates a project path or commit SHA into the title).
    """
    return (
        "<!doctype html>"
        '<html lang="en">'
        '<head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        # Inline CSS for the Cytoscape container and the Mermaid
        # fallback. Default state: Cytoscape container is visible
        # and the Mermaid ``<pre>`` is hidden. If Cytoscape fails
        # to load (the static bundle is missing), the JS guard
        # below adds ``body.no-cytoscape`` which inverts the two
        # rules: Mermaid block becomes visible and is initialized
        # as before. This preserves the spec's Mermaid fallback
        # without making the operator look at a tiny diagram when
        # Cytoscape is available.
        "<style>"
        ".graph-container { width: 100%; height: 80vh; "
        "border: 1px solid #ccc; background: #fafafa; "
        "margin-top: 0.5em; }"
        "pre.mermaid { display: none; }"
        "body.no-cytoscape pre.mermaid { display: block; }"
        "body.no-cytoscape .graph-container { display: none; }"
        # Static-SVG container — used when the server pre-rendered
        # the diagram with Graphviz. The SVG is embedded inline; the
        # ``overflow:hidden`` plus ``cursor:grab`` style hints at
        # the vanilla-JS pan/zoom binding installed below. When a
        # ``.static-graph`` element is present the JS guard hides
        # the Cytoscape and Mermaid containers because the static
        # image already represents the diagram.
        ".static-graph { width: 100%; height: 80vh; "
        "border: 1px solid #ccc; background: #fafafa; "
        "margin-top: 0.5em; overflow: hidden; cursor: grab; "
        "position: relative; }"
        ".static-graph.is-panning { cursor: grabbing; }"
        ".static-graph svg { width: 100%; height: 100%; "
        "display: block; user-select: none; "
        "transform-origin: 0 0; }"
        "body.has-static-graph .graph-container { display: none; }"
        "body.has-static-graph pre.mermaid { display: none; }"
        "</style>"
        "</head>"
        "<body>"
        '<nav class="page-nav"><a class="page-nav__index-link" href="/">Index</a></nav>'
        f"<main>{fragment}</main>"
        # Load Cytoscape first; Mermaid is the documented fallback
        # used when Cytoscape is not present. Both are served from
        # ``/static/`` (the package directory shipped with the
        # release); the ``static/README.md`` inside that directory
        # explains how to drop in the bundles.
        '<script src="/static/cytoscape.min.js"></script>'
        '<script src="/static/mermaid.min.js"></script>'
        "<script>"
        "(function() {"
        # Static-SVG short-circuit: when the server pre-rendered the
        # diagram (Graphviz path), bind vanilla pan/zoom on the SVG
        # and skip both the Cytoscape and the Mermaid initialization
        # — there is no source to render.
        "  var staticGraph = document.querySelector('.static-graph');"
        "  if (staticGraph) {"
        "    document.body.classList.add('has-static-graph');"
        "    var svg = staticGraph.querySelector('svg');"
        "    if (svg) {"
        # Drop any width/height attributes Graphviz emitted so the
        # SVG honours our 100%/100% container sizing (the inline
        # attributes otherwise pin a fixed pixel size).
        "      svg.removeAttribute('width');"
        "      svg.removeAttribute('height');"
        "      var tx = 0, ty = 0, scale = 1;"
        "      var dragging = false, lastX = 0, lastY = 0;"
        "      function apply() {"
        "        svg.style.transform = "
        "          'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';"
        "      }"
        "      staticGraph.addEventListener('wheel', function(ev) {"
        "        ev.preventDefault();"
        "        var rect = staticGraph.getBoundingClientRect();"
        "        var mx = ev.clientX - rect.left, my = ev.clientY - rect.top;"
        "        var factor = ev.deltaY < 0 ? 1.1 : 1 / 1.1;"
        # Zoom around the cursor position by adjusting the pan so
        # the world point under the cursor stays fixed.
        "        tx = mx - (mx - tx) * factor;"
        "        ty = my - (my - ty) * factor;"
        "        scale *= factor;"
        "        apply();"
        "      }, { passive: false });"
        "      staticGraph.addEventListener('mousedown', function(ev) {"
        "        dragging = true; lastX = ev.clientX; lastY = ev.clientY;"
        "        staticGraph.classList.add('is-panning');"
        "      });"
        "      window.addEventListener('mousemove', function(ev) {"
        "        if (!dragging) { return; }"
        "        tx += ev.clientX - lastX; ty += ev.clientY - lastY;"
        "        lastX = ev.clientX; lastY = ev.clientY;"
        "        apply();"
        "      });"
        "      window.addEventListener('mouseup', function() {"
        "        dragging = false;"
        "        staticGraph.classList.remove('is-panning');"
        "      });"
        # Double-click resets the view to the fit-on-load state.
        "      staticGraph.addEventListener('dblclick', function() {"
        "        tx = 0; ty = 0; scale = 1; apply();"
        "      });"
        "    }"
        "    return;"
        "  }"
        "  var hasCy = (typeof cytoscape !== 'undefined');"
        "  if (!hasCy) {"
        "    document.body.classList.add('no-cytoscape');"
        "    if (typeof mermaid !== 'undefined') {"
        "      mermaid.initialize({"
        "        startOnLoad: true,"
        "        maxTextSize: 1048576,"
        "        maxEdges: 100000,"
        "        flowchart: { useMaxWidth: false, nodeSpacing: 80,"
        "                     rankSpacing: 100, htmlLabels: true }"
        "      });"
        "    }"
        "    return;"
        "  }"
        "  var blocks = document.querySelectorAll('.graph-data');"
        "  blocks.forEach(function(scriptEl) {"
        "    var container = scriptEl.parentElement"
        "                            .querySelector('.graph-container');"
        "    if (!container) { return; }"
        "    try {"
        "      var data = JSON.parse(scriptEl.textContent);"
        "      var elements = [];"
        "      if (data.nodes) { data.nodes.forEach(function(n) {"
        "        elements.push(n); }); }"
        "      if (data.edges) { data.edges.forEach(function(e) {"
        "        elements.push(e); }); }"
        # Layout choice — ``cose`` is decent for force-directed
        # visuals but defaults to 1000 iterations of an O(N^2)
        # physics simulation. On a 145-node graph with hundreds of
        # edges that takes several seconds and feels frozen. The
        # tuning below caps iterations at 100, disables animation,
        # and reuses existing positions where possible — enough to
        # converge on a usable layout in a fraction of a second.
        # Viewport-time options (``hideEdgesOnViewport``,
        # ``hideLabelsOnViewport``, ``textureOnViewport``,
        # ``motionBlur``) keep pan/zoom smooth on commodity laptops
        # by trading transient visual fidelity for frame rate. Edge
        # labels are hidden in the steady state (drawn on
        # ``:hover``/``:selected`` instead) because rendering text
        # over hundreds of overlapping edges is the dominant cost.
        "      cytoscape({"
        "        container: container,"
        "        elements: elements,"
        "        layout: { name: 'cose', animate: false, fit: true,"
        "                  padding: 30, nodeRepulsion: 4000,"
        "                  idealEdgeLength: 80, nodeOverlap: 16,"
        "                  numIter: 100, randomize: false,"
        "                  componentSpacing: 100 },"
        "        hideEdgesOnViewport: true,"
        "        hideLabelsOnViewport: true,"
        "        textureOnViewport: true,"
        "        motionBlur: true,"
        "        motionBlurOpacity: 0.2,"
        "        pixelRatio: 1.0,"
        "        wheelSensitivity: 0.3,"
        "        style: ["
        "          { selector: 'node', style: {"
        "            'label': 'data(label)',"
        "            'background-color': '#6FB1FC',"
        "            'text-valign': 'center',"
        "            'text-halign': 'center',"
        "            'color': '#1a1a1a',"
        "            'font-size': '10px',"
        "            'min-zoomed-font-size': 6,"
        "            'text-wrap': 'wrap',"
        "            'text-max-width': '140px',"
        "            'shape': 'roundrectangle',"
        "            'padding': '6px',"
        "            'width': 'label',"
        "            'height': 'label',"
        "            'border-width': 1,"
        "            'border-color': '#3779C7'"
        "          } },"
        "          { selector: 'edge', style: {"
        "            'label': '',"
        "            'curve-style': 'bezier',"
        "            'line-color': '#aaa',"
        "            'width': 1,"
        "            'opacity': 0.4,"
        "            'target-arrow-shape': 'none'"
        "          } },"
        "          { selector: 'edge:selected, edge.hover', style: {"
        "            'label': 'data(label)',"
        "            'font-size': '10px',"
        "            'color': '#000',"
        "            'text-rotation': 'autorotate',"
        "            'text-background-color': '#fff',"
        "            'text-background-opacity': 0.95,"
        "            'text-background-padding': '3px',"
        "            'line-color': '#3779C7',"
        "            'width': 2,"
        "            'opacity': 1,"
        "            'z-index': 10"
        "          } },"
        "          { selector: 'node:selected', style: {"
        "            'background-color': '#FCB16F',"
        "            'border-color': '#C77137',"
        "            'border-width': 2"
        "          } }"
        "        ]"
        "      }).on('mouseover', 'edge', function(evt) {"
        "        evt.target.addClass('hover');"
        "      }).on('mouseout', 'edge', function(evt) {"
        "        evt.target.removeClass('hover');"
        "      });"
        "    } catch (err) {"
        "      console.error('Failed to render graph:', err);"
        "    }"
        "  });"
        "})();"
        "</script>"
        "</body>"
        "</html>"
    )


def _build_dependencies_route(
    catalog: _CatalogReader,
    store: _ProfileReader,
) -> Route:
    """Return the ``Route`` for ``GET /dependencies``.

    The handler walks two branches in the order required by Property
    21's no-snapshot priority and Requirement 14.4:

    1. **No snapshot committed** —
       :meth:`_SnapshotReader.get_current_snapshot_id` returns
       ``None``. The handler responds with HTTP 200 and the fixed
       :data:`_NO_SNAPSHOT_HTML` body, which contains
       :data:`NO_SNAPSHOT_MESSAGE` and *no* diagram content
       (Requirement 14.4).
    2. **Snapshot committed** — the handler reads
       :meth:`ProjectCatalog.list_in_scope` and
       :meth:`KnowledgeStore.list_profiles` *at request time*
       (Requirement 14.1's "no in-memory caching of profile data"
       rule), feeds them to :func:`render_dependency_graph`, and
       returns the resulting fragment wrapped in the standard page
       envelope.

    The snapshot read precedes the catalog and profile reads so the
    no-snapshot branch never invokes ``list_profiles``; this matches
    the per-project handler's "check authority first" idiom and keeps
    the no-snapshot response cheap. ``render_dependency_graph`` itself
    is pure and snapshot-isolated through ``KnowledgeStore``'s
    ``current_snapshot.snapshot_id`` lookup (Property 11), so the
    catalog and profile reads observe a single consistent snapshot
    even though they are issued sequentially.

    Implements Requirements 13.3, 14.1, 14.4.
    """

    async def dependencies(request: Request) -> Response:
        del request  # Unused; the response is fully determined by store + catalog.

        snapshot_id = store.get_current_snapshot_id()
        if snapshot_id is None:
            return HTMLResponse(
                _NO_SNAPSHOT_HTML, status_code=200, media_type=HTML_CONTENT_TYPE
            )

        in_scope = catalog.list_in_scope()
        profiles = store.list_profiles()
        fragment = render_dependency_graph(in_scope, profiles)
        body = _wrap_diagram_fragment(
            title="Project Knowledge — Dependency Graph",
            fragment=fragment,
        )
        return HTMLResponse(body, status_code=200, media_type=HTML_CONTENT_TYPE)

    return Route("/dependencies", dependencies, methods=["GET"])


# ---------------------------------------------------------------------------
# Fallback exception handlers (Requirements 13.5, 13.7, 13.8, 14.6)
# ---------------------------------------------------------------------------
#
# These handlers are registered with the Starlette application by
# :func:`build_visualization_app` (task 10.8). They cover three failure
# modes that the four documented diagram routes do not handle inline:
#
#   * **404** — the request matched no registered route. Starlette's
#     :class:`Router` raises ``HTTPException(status_code=404)`` and the
#     status_handler below renders an HTML body that names the requested
#     path verbatim (Requirement 13.7). Routing layers that match the
#     path *and* the method (e.g. ``/projects/{project_id:int}``) but
#     fail to find a profile send their own 404 — those bypass this
#     fallback because they raise no exception, they return a
#     ``Response`` directly (per-project handler in task 10.6).
#   * **405** — the request matched the path of a registered route but
#     used a non-GET method. Each documented route is registered with
#     ``methods={"GET"}`` (HEAD removed via post-construction mutation
#     in :func:`build_visualization_app`) so Starlette raises
#     ``HTTPException(status_code=405, headers={"Allow": "GET"})`` for
#     every non-GET request — including HEAD — and the status_handler
#     below replaces the default plain-text body with the documented
#     HTML body while keeping the ``Allow`` header value exactly equal
#     to the string ``"GET"`` (Requirement 13.8).
#   * **503** — a handler raised
#     :class:`KnowledgeStoreUnavailableError` because the underlying
#     store could not be read at request time. The class-keyed
#     exception handler below converts the failure into the documented
#     503 HTML body. The body is *fully derived from the static
#     :data:`_STORE_UNAVAILABLE_HTML` constant*; no cached profile data
#     is ever served as a fallback (Requirement 14.6).
#
# Both 503 paths (this one and the deadline-503 in
# :class:`_DeadlineMiddleware`) share the documented HTML
# Content-Type (Requirement 13.5) but use different body strings so
# the operator can tell them apart.


async def _http_exception_handler(
    request: Request,
    exc: Exception,
) -> Response:
    """Render the documented HTML body for a 404 or 405 ``HTTPException``.

    Registered as the status handler for both 404 and 405 in
    :func:`_build_fallback_exception_handlers`. Dispatches on
    ``exc.status_code`` so the two branches share the surrounding
    plumbing (request access, Content-Type setting, escape rules) but
    each renders the body required by its own requirement:

    * **404** (Requirement 13.7) — the body must include the requested
      HTTP path verbatim. The path is read from
      :attr:`Request.url.path`; that returns the URL-decoded path the
      ASGI server delivered to Starlette, which is what an operator
      would type into a browser address bar. The path is HTML-escaped
      before interpolation so a path containing ``<`` or ``>`` cannot
      break out of the surrounding tags, but the visible glyphs remain
      intact so the rendered body shows the path as the operator
      requested it.
    * **405** (Requirement 13.8) — the body is fully static
      (:data:`_METHOD_NOT_ALLOWED_HTML`). The ``Allow`` header is set
      explicitly to the string ``"GET"`` (no spaces, no other methods)
      so the requirement's "header value exactly the string 'GET'"
      clause cannot regress through Starlette adding ``HEAD`` or other
      methods to the joined header value.

    Other status codes are not expected to flow through this handler
    (the Starlette default handlers cover them); the unreachable
    branch re-raises so a misconfiguration surfaces during tests
    rather than producing a silently-mangled response.
    """
    # ``HTTPException`` is the only type the integer status_handlers
    # ever receive at runtime; the narrower runtime guard below keeps
    # the code defensible if a caller registers this handler under a
    # class key by mistake.
    if not isinstance(exc, HTTPException):  # pragma: no cover - defensive
        raise exc

    if exc.status_code == _HTTP_STATUS_NOT_FOUND:
        return HTMLResponse(
            _render_not_found_html(request.url.path),
            status_code=_HTTP_STATUS_NOT_FOUND,
            media_type=HTML_CONTENT_TYPE,
        )
    if exc.status_code == _HTTP_STATUS_METHOD_NOT_ALLOWED:
        return HTMLResponse(
            _METHOD_NOT_ALLOWED_HTML,
            status_code=_HTTP_STATUS_METHOD_NOT_ALLOWED,
            headers={"Allow": "GET"},
            media_type=HTML_CONTENT_TYPE,
        )
    # Defensive fallback: re-raise so Starlette's default machinery
    # produces a sensible response. Reaching this branch indicates a
    # registration mistake, not a runtime condition the user can hit.
    raise exc  # pragma: no cover - defensive


async def _store_unavailable_handler(
    request: Request,
    exc: Exception,
) -> Response:
    """Render the documented 503 HTML body when the store is unavailable.

    Registered as the class-keyed exception handler for
    :class:`KnowledgeStoreUnavailableError`. Returns the static
    :data:`_STORE_UNAVAILABLE_HTML` body — the response is *not*
    derived from any cached or in-memory ``Project_Profile`` data, so
    Requirement 14.6's "never serve cached profile data" rule holds by
    construction.

    The body is distinct from :data:`_DEADLINE_EXCEEDED_HTML` so the
    two 503 paths (timeout vs unavailable store) remain
    distinguishable from the response body alone, as called out by the
    task 10.8 implementation guidance.
    """
    del request, exc  # The 503 body has no per-request inputs.
    return HTMLResponse(
        _STORE_UNAVAILABLE_HTML,
        status_code=_HTTP_STATUS_SERVICE_UNAVAILABLE,
        media_type=HTML_CONTENT_TYPE,
    )


def _build_fallback_exception_handlers() -> dict[Any, ExceptionHandler]:
    """Return the {status-code | exception-class -> handler} mapping.

    Two integer keys (``404`` and ``405``) route to
    :func:`_http_exception_handler` and one class key
    (:class:`KnowledgeStoreUnavailableError`) routes to
    :func:`_store_unavailable_handler`. Starlette's
    :class:`ExceptionMiddleware` separates integer keys (status
    handlers) from class keys (exception handlers) at registration
    time, so a single dict here suffices for both.

    Implements Requirements 13.5, 13.7, 13.8, 14.6.
    """
    handlers: dict[Any, ExceptionHandler] = {
        _HTTP_STATUS_NOT_FOUND: _http_exception_handler,
        _HTTP_STATUS_METHOD_NOT_ALLOWED: _http_exception_handler,
        KnowledgeStoreUnavailableError: _store_unavailable_handler,
    }
    return handlers


#: Filesystem path to the package-level static asset directory served
#: at ``/static/<filename>`` by the ``Visualization_Server``. Resolved
#: from the package's ``__file__`` so the directory is found whether
#: the project is run from a source checkout (``pip install -e``) or
#: from an installed wheel. The directory itself ships in the wheel
#: because Hatch's ``packages`` build target includes every file under
#: ``src/project_knowledge_mcp``, not just ``.py`` files. Operators
#: drop ``mermaid.min.js`` here to enable diagram rendering; see
#: ``static/README.md`` inside that directory for instructions.
_STATIC_DIR_PATH: Final[Path] = Path(__file__).resolve().parent / "static"


def _build_static_mount() -> Mount:
    """Return the ``Mount`` that serves ``/static/<filename>`` from the package.

    Uses Starlette's :class:`starlette.staticfiles.StaticFiles` with
    ``html=False`` (no implicit ``index.html`` fallback) and
    ``check_dir=False`` so the server still starts cleanly when the
    static directory has been emptied by a sysadmin. A request for a
    missing file returns ``404 Not Found`` and falls through to the
    existing 404 fallback handler.

    Path traversal is handled by Starlette: it rejects any path that
    resolves outside the configured directory with a ``404`` (e.g.
    ``GET /static/../../etc/passwd``).
    """
    return Mount(
        "/static",
        app=StaticFiles(directory=str(_STATIC_DIR_PATH), check_dir=False),
        name="static",
    )


def build_visualization_app(
    catalog: _CatalogReader,
    store: _ProfileReader,
) -> Starlette:
    """Build the Visualization_Server application with the diagram routes.

    Wires three diagram handlers — ``GET /``, ``GET /projects/{project_id}``,
    and ``GET /dependencies`` — onto the Starlette skeleton produced
    by :func:`build_app`, alongside the 404/405/503 fallbacks.

    The ``/conflicts`` route was retired by an operator-tuning
    decision: the conflict graph proved unreadable at scale (100+
    projects, hundreds of cross-edges) and the operator preferred to
    rely on the ``Conflict_Detector`` API directly rather than the
    interactive page. The detector library remains available for
    programmatic callers; only the HTTP surface has been removed.

    ``catalog`` and ``store`` are typed as the structural Protocols
    :class:`_CatalogReader` and :class:`_ProfileReader` so unit tests
    can pass duck-typed fakes and ``main.py`` can pass the live
    ``ProjectCatalog`` and ``KnowledgeStore`` instances.
    """
    routes: list[BaseRoute] = [
        _build_index_route(catalog, store),
        _build_per_project_route(catalog, store),
        _build_dependencies_route(catalog, store),
        _build_static_mount(),
    ]
    # Constrain every documented route to ``methods={"GET"}`` only.
    # Starlette's ``Route`` constructor automatically widens
    # ``methods=["GET"]`` to ``{"GET", "HEAD"}`` so HEAD requests would
    # otherwise be served by the GET handler with the body stripped.
    # Requirement 13.8 forbids that: every non-GET method (including
    # HEAD) on the three documented routes must produce HTTP 405 with
    # an ``Allow`` header whose value is exactly ``"GET"``. Removing
    # HEAD from each route's ``methods`` set makes Starlette raise
    # :class:`starlette.exceptions.HTTPException(status_code=405,
    # headers={"Allow": "GET"})` for every non-GET request, which the
    # 405 status handler registered below converts into the documented
    # HTML body while preserving the exact Allow header value.
    for route in routes:
        if isinstance(route, Route):
            route.methods = {"GET"}
    return build_app(
        routes=routes,
        exception_handlers=_build_fallback_exception_handlers(),
    )


# ---------------------------------------------------------------------------
# Loopback-only socket binding (Requirements 12.2, 12.6, 12.8)
# ---------------------------------------------------------------------------


def _bind_loopback_sockets(port: int) -> list[socket.socket]:
    """Bind one IPv4 and one IPv6 loopback socket to ``port``.

    Both sockets share the same TCP port. ``IPV6_V6ONLY`` is set on the
    IPv6 socket so that on dual-stack systems the kernel does not also
    map ``0.0.0.0:{port}`` requests onto the IPv6 socket; the result is
    that the server only ever accepts connections on ``127.0.0.1`` or
    ``::1`` (Requirement 12.2).

    Raises :class:`BindError` on any underlying :class:`OSError`. The
    original :class:`OSError` is preserved as ``__cause__`` so callers
    can distinguish ``EADDRINUSE`` from other failures via
    :func:`_format_startup_error`.

    Returns:
        A list ``[ipv4_socket, ipv6_socket]`` of bound, listening
        sockets ready to be passed to :meth:`uvicorn.Server.serve`.
    """
    sockets: list[socket.socket] = []
    try:
        ipv4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sockets.append(ipv4)
        ipv4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ipv4.bind((LOOPBACK_IPV4, port))
        ipv4.listen()

        ipv6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sockets.append(ipv6)
        ipv6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Without IPV6_V6ONLY a Linux kernel will route incoming IPv4
        # connections through the IPv6 socket, which would silently
        # widen the accepted-address set beyond the loopback pair.
        if hasattr(socket, "IPV6_V6ONLY"):
            ipv6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        ipv6.bind((LOOPBACK_IPV6, port))
        ipv6.listen()
    except OSError as exc:
        for s in sockets:
            with contextlib.suppress(OSError):
                s.close()
        # ``BindError`` carries a free-form reason; the caller decides
        # which of the two documented start-up error lines to emit by
        # inspecting ``__cause__.errno`` (see :func:`_format_startup_error`).
        if exc.errno == errno.EADDRINUSE:
            raise BindError(port, "address already in use") from exc
        raise BindError(port, str(exc)) from exc
    return sockets


def _format_startup_error(port: int, exc: BindError) -> str:
    """Return the documented start-up error line for ``exc``.

    Routes ``EADDRINUSE`` to the Requirement 12.6 wording and every other
    :class:`OSError` to the Requirement 12.8 wording. Decisions are made
    against the chained :class:`OSError` (``exc.__cause__``) when present
    so the choice depends on the OS-reported errno rather than on
    free-form text in :attr:`BindError.reason`.
    """
    underlying = exc.__cause__
    if isinstance(underlying, OSError) and underlying.errno == errno.EADDRINUSE:
        return PORT_IN_USE_TEMPLATE.format(port=port)
    if isinstance(underlying, OSError):
        return GENERIC_BIND_FAILURE_TEMPLATE.format(os_error=underlying)
    # No chained OSError: fall back to the BindError's own reason text.
    return GENERIC_BIND_FAILURE_TEMPLATE.format(os_error=exc.reason)


def bind_or_exit(
    port: int,
    *,
    stderr: TextIO | None = None,
) -> list[socket.socket]:
    """Bind both loopback sockets, or terminate the process on failure.

    On a successful bind, returns the two bound sockets. On any
    :class:`OSError`, writes the documented start-up error line to
    ``stderr`` and calls :func:`sys.exit` with a non-zero status.

    Implements Requirements 12.6 and 12.8.
    """
    err_stream = sys.stderr if stderr is None else stderr
    try:
        return _bind_loopback_sockets(port)
    except BindError as exc:
        print(_format_startup_error(port, exc), file=err_stream, flush=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Ready log line (Requirement 12.7)
# ---------------------------------------------------------------------------


def emit_ready_log(port: int, *, logger: logging.Logger | None = None) -> None:
    """Emit the documented ``"ready"`` log line at WARNING level.

    The wording is exactly ``"Visualization_Server ready at
    http://127.0.0.1:{port}"`` per Requirement 12.7 and is pinned by
    the task 10.10 unit test. The level is ``WARNING`` rather than the
    Python ``INFO`` default so the line surfaces under the default
    root-logger threshold of ``WARNING`` without requiring the
    operator to bump every third-party library's verbosity too. The
    line is operationally important (it's the signal that the
    Visualization_Server has bound its sockets and is ready to
    accept HTTP connections) so the WARNING level matches its
    practical role even though "ready" is not literally a warning.
    """
    log = _LOG if logger is None else logger
    log.warning(READY_LOG_TEMPLATE.format(port=port))


# ---------------------------------------------------------------------------
# Uvicorn server construction
# ---------------------------------------------------------------------------


def create_server(app: Starlette, port: int) -> uvicorn.Server:
    """Create a Uvicorn :class:`uvicorn.Server` for ``app``.

    The Uvicorn :class:`uvicorn.Config` is constructed with the loopback
    IPv4 host and the configured port even though the actual sockets are
    pre-bound and supplied to :meth:`uvicorn.Server.serve`; setting
    ``host`` and ``port`` on the config keeps Uvicorn's own diagnostics
    consistent with the addresses we actually serve on. Uvicorn's
    default log configuration is suppressed (``log_config=None``) so
    Uvicorn does not emit its own ``"Uvicorn running on..."`` line and
    duplicate the documented Requirement 12.7 log line emitted by
    :func:`emit_ready_log`.
    """
    config = uvicorn.Config(
        app,
        host=LOOPBACK_IPV4,
        port=port,
        log_config=None,
        access_log=False,
        # The visualization surface owns its own lifespan; we don't need
        # Uvicorn to send ASGI ``lifespan.startup`` / ``shutdown``
        # messages for the skeleton (Starlette tolerates either).
        lifespan="off",
    )
    return uvicorn.Server(config)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run(
    port: int,
    *,
    app: Starlette | None = None,
    stderr: TextIO | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Bind, log readiness, and serve until cancelled.

    This is the convenience entry point used by the in-module unit tests
    and by simple deployments. ``main.py`` (task 11.1) may instead call
    :func:`bind_or_exit`, :func:`emit_ready_log`, and
    :func:`create_server` directly so it can interleave the MCP stdio
    bind and scheduler start with the visualization start-up.

    Args:
        port: TCP port on which to bind both loopback sockets.
        app: Optional pre-built Starlette application. When ``None``, a
            fresh skeleton is built via :func:`build_app`.
        stderr: Optional stream for the documented start-up error line
            on bind failure (defaults to :data:`sys.stderr`).
        logger: Optional logger used to emit the ``"ready"`` log line
            (defaults to this module's logger).
    """
    application = build_app() if app is None else app
    sockets = bind_or_exit(port, stderr=stderr)
    emit_ready_log(port, logger=logger)
    server = create_server(application, port)
    asyncio.run(server.serve(sockets=sockets))


__all__ = [
    "GENERIC_BIND_FAILURE_TEMPLATE",
    "HANDLER_DEADLINE_SECONDS",
    "HTML_CONTENT_TYPE",
    "LOOPBACK_IPV4",
    "LOOPBACK_IPV6",
    "METHOD_NOT_ALLOWED_MESSAGE",
    "NOT_FOUND_MESSAGE_TEMPLATE",
    "PORT_IN_USE_TEMPLATE",
    "READY_LOG_TEMPLATE",
    "STORE_UNAVAILABLE_MESSAGE",
    "bind_or_exit",
    "build_app",
    "build_visualization_app",
    "create_server",
    "emit_ready_log",
    "run",
]

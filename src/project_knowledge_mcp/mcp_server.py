"""MCP transport layer (stdio), ``tools/list`` surface, and ``tools/call`` dispatch.

This module wires the Project Knowledge MCP Server's MCP surface
(Requirement 11) on top of the official Python MCP SDK
(:mod:`mcp.server.lowlevel`). It owns the ``Server`` lifecycle, binds
to ``stdin``/``stdout`` via :func:`mcp.server.stdio.stdio_server`, and
serves the design's ``initialize`` payload (Requirements 11.1, 11.2),
``tools/list`` payload (Requirement 11.3), and ``tools/call`` dispatch
with the design's exact error mapping (Requirements 11.4-11.7).

Task 9.4 still fills in:

* 9.4 - registration of each of the eight tools' handlers and
  wiring of those handlers to the injected collaborators.

The set of advertised tools is fixed by the design and lives at module
scope as :data:`BUILT_IN_TOOLS`. ``tools/list`` always returns those
eight tools regardless of whether 9.4 has wired their handlers yet:
the design treats the tool surface as part of the server's contract,
not as a side-effect of handler registration. Each ``Tool`` object
carries the ``inputSchema`` that the ``tools/call`` dispatcher uses
to validate arguments before invoking the registered handler.

To keep the surface area between this module and 9.4 small, this
module exposes one extension point:

* :meth:`MCPServer.register_tool` - 9.4 calls this once per tool. Each
  call records the per-tool handler that ``tools/call`` dispatches to
  after argument-schema validation. 9.4 does not touch the SDK
  directly; the dispatcher in :meth:`_register_default_handlers` looks
  up the handler in :attr:`_tool_handlers` after validating the
  request against :data:`BUILT_IN_TOOLS`.

The ``tools`` server-capability in the ``initialize`` response is
advertised because we register a ``list_tools`` handler with the SDK
at construction time. Per
:meth:`mcp.server.lowlevel.Server.get_capabilities`, the presence of
a ``ListToolsRequest`` handler in ``request_handlers`` is what causes
``capabilities.tools`` to be set on the ``initialize`` response, which
is exactly the ``{tools: {}}`` shape Requirement 11.2 mandates. The
SDK invokes that handler only in response to an explicit ``tools/list``
request from the client, so the server never emits an unsolicited
``tools/list`` response (Requirement 11.3).

The ``tools/call`` dispatcher is registered directly in the SDK's
``request_handlers`` mapping (rather than via the SDK's
``@Server.call_tool()`` decorator) for one specific reason:
Requirements 11.5 and 11.6 demand that ``tools/call`` errors for
unknown tools and for argument-schema violations surface as **MCP
JSON-RPC error responses** (codes ``MethodNotFound`` /
``InvalidParams``) rather than as ``isError: true`` tool results. The
SDK's ``call_tool`` decorator wraps the handler in a ``try/except
Exception`` block that converts every raised exception (including
:class:`mcp.shared.exceptions.McpError`) into an ``isError: true``
:class:`mcp.types.CallToolResult`, which would silently downgrade
those JSON-RPC errors into tool results. Registering the dispatcher
directly lets :class:`McpError` propagate to the SDK's
``_handle_request``, which catches it and maps ``err.error`` straight
onto a :class:`mcp.types.JSONRPCError` response - exactly the wire
shape Requirements 11.5 and 11.6 require.

Implements Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7.
"""

from __future__ import annotations

import asyncio
import json
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeAlias

import jsonschema
import mcp.types as mcp_types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

from .errors import IngestionInProgressError, ProjectNotInScopeError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from .conflict_detector import ConflictPair
    from .ingestion_coordinator import IngestionCoordinator
    from .knowledge_store import KnowledgeStore
    from .models import ConflictResult, ProjectProfile
    from .project_catalog import ProjectCatalog


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The MCP server name advertised in the ``initialize`` response. Fixed
#: by the design and surfaced verbatim to MCP clients (Requirement 11.2).
SERVER_NAME: Final[str] = "project-knowledge-mcp"

#: Fallback version string used when the package metadata cannot be
#: resolved (for example when running from a source tree that has not
#: been installed). Matches the ``version`` declared in
#: ``pyproject.toml`` so the two never disagree at startup.
FALLBACK_VERSION: Final[str] = "0.0.0"

#: Distribution name registered by ``pyproject.toml``. Used to look up
#: the version from installed package metadata when one is available.
_DISTRIBUTION_NAME: Final[str] = "project-knowledge-mcp"


# ---------------------------------------------------------------------------
# Tool registry constants
# ---------------------------------------------------------------------------

# The eight MCP tools advertised by ``tools/list`` are fixed by the
# design's "MCP Transport Layer" section (which sources Requirements
# 8.1, 8.2, and 10.1-10.6). Their names and argument shapes are part
# of the server's contract: ``tools/list`` always returns exactly
# this set, in this order, with these schemas. 9.4 registers the
# matching handlers; 9.3 dispatches against ``inputSchema`` to
# validate arguments before calling those handlers.

#: Tool name: returns all in-scope projects with their
#: ``gitlab_project_id`` and ``full_path`` (Requirement 10.6).
TOOL_NAME_LIST_PROJECTS: Final[str] = "list_projects"

#: Tool name: returns the purpose summary for a project identified by
#: its GitLab project ID (Requirement 10.1).
TOOL_NAME_GET_PROJECT_PURPOSE: Final[str] = "get_project_purpose"

#: Tool name: returns the ``Abstract_Inputs`` and ``Abstract_Outputs``
#: for a project identified by its GitLab project ID (Requirement 10.2).
TOOL_NAME_GET_PROJECT_IO: Final[str] = "get_project_io"

#: Tool name: returns the ``External_Service_Dependencies`` and
#: ``Database_Table_Dependencies`` for a project identified by its
#: GitLab project ID (Requirement 10.3).
TOOL_NAME_GET_PROJECT_DEPENDENCIES: Final[str] = "get_project_dependencies"

#: Tool name: returns the full ``Project_Profile`` for a project
#: identified by its GitLab project ID (Requirement 10.5).
TOOL_NAME_GET_PROJECT_PROFILE: Final[str] = "get_project_profile"

#: Tool name: returns the list of all ``Purpose_Conflict`` pairs
#: across the current snapshot (Requirement 10.4).
TOOL_NAME_LIST_PURPOSE_CONFLICTS: Final[str] = "list_purpose_conflicts"

#: Tool name: triggers an ``Ingestion_Job`` for all in-scope projects
#: (Requirement 8.1).
TOOL_NAME_REFRESH_ALL_PROJECTS: Final[str] = "refresh_all_projects"

#: Tool name: triggers an ``Ingestion_Job`` for a single project
#: identified by its GitLab project ID (Requirement 8.2).
TOOL_NAME_REFRESH_PROJECT: Final[str] = "refresh_project"


# ---------------------------------------------------------------------------
# Canonical error messages surfaced by tool handlers
# ---------------------------------------------------------------------------


def _not_in_scope_message(gitlab_project_id: int) -> str:
    """Return the canonical "project not in scope" message for tool results.

    Per Requirement 10.7 every project-id-typed MCP tool surfaces an
    out-of-scope GitLab project ID as a tool result with ``isError:
    true`` whose message names the offending ID. The wording is fixed
    by the design so the four read tools and ``refresh_project`` all
    emit identical text. The same wording also appears in the
    Visualization_Server's 404 path (Requirement 14.5) — keeping a
    single helper here avoids accidental drift between the two
    surfaces.
    """

    return f"project {gitlab_project_id} is not in scope"


def _not_yet_analyzed_message(gitlab_project_id: int) -> str:
    """Return the canonical "project not yet analyzed" message.

    Surfaced by every read tool when the requested project IS in the
    current ``Project_Catalog`` but no ``Project_Profile`` has been
    persisted yet — the in-scope-but-not-yet-analyzed state the
    Visualization_Server names in Requirement 14.3. The result is a
    *non-error* tool result (the project is in scope, the catalog is
    fresh, and the caller's request is well-formed); the message
    simply tells the caller that they must wait for an
    ``Ingestion_Job`` to commit a snapshot containing this project.
    """

    return (
        f"project {gitlab_project_id} has not yet been analyzed; "
        "run an Ingestion_Job to populate its profile"
    )


def _structured_result(payload: dict[str, Any]) -> mcp_types.CallToolResult:
    """Build a happy-path :class:`CallToolResult` from a JSON-compatible payload.

    Every read tool emits exactly one ``TextContent`` block carrying
    the JSON-serialized form of ``payload`` and the same ``payload``
    via ``structuredContent``. The text and structured surfaces stay
    aligned (text-mode clients see what structured-mode clients see)
    and the JSON encoding uses ``sort_keys=False`` plus a stable
    indent so the wire output is human-readable.

    ``payload`` must be JSON-serializable; the read-tool handlers
    feed it through ``model_dump(mode="json")`` first so datetimes
    and enums are already strings by the time they reach this
    helper.
    """

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=text)],
        structuredContent=payload,
    )


def _error_result(message: str) -> mcp_types.CallToolResult:
    """Build a failure :class:`CallToolResult` with ``isError: true``.

    Used by tool handlers that surface a domain-level rejection (out
    of scope, ingestion already running) as an MCP-level tool result
    rather than as a JSON-RPC error or as the dispatcher's generic
    "tool execution failed" wrapper. The canonical wording is
    supplied by the caller and embedded verbatim in a single
    ``TextContent`` block.
    """

    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=message)],
        isError=True,
    )


def _not_in_scope_result(gitlab_project_id: int) -> mcp_types.CallToolResult:
    """Build the canonical Requirement 10.7 out-of-scope tool result."""

    return _error_result(_not_in_scope_message(gitlab_project_id))


def _not_yet_analyzed_result(gitlab_project_id: int) -> mcp_types.CallToolResult:
    """Build the non-error "project not yet analyzed" tool result."""

    return mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(
                type="text",
                text=_not_yet_analyzed_message(gitlab_project_id),
            )
        ],
    )


def _no_args_input_schema() -> dict[str, Any]:
    """Return a fresh JSON Schema for tools that take no arguments.

    A new dict is returned on every call so the schemas attached to
    different :class:`mcp.types.Tool` instances do not alias one
    another. Aliased dicts would be safe in this module (we never
    mutate them), but freshly-built dicts are robust against future
    callers that might.

    Shape: ``{"type": "object", "properties": {}}``. Per the JSON
    Schema spec this accepts the empty object ``{}`` (and only the
    empty object once 9.3 enforces additional-property rejection at
    the dispatch layer); no-argument tools are invoked with an empty
    arguments mapping.
    """

    return {"type": "object", "properties": {}}


def _gitlab_project_id_input_schema() -> dict[str, Any]:
    """Return a fresh JSON Schema for the ``gitlab_project_id`` argument.

    Shared by every project-id-typed tool (``get_project_purpose``,
    ``get_project_io``, ``get_project_dependencies``,
    ``get_project_profile``, ``refresh_project``). The argument is
    typed as a JSON integer, which matches the GitLab API's project
    identifier shape and the type carried by
    :class:`project_knowledge_mcp.models.ProjectProfile`.

    A new dict is returned on every call (see
    :func:`_no_args_input_schema` for the rationale). The schema is
    intentionally minimal here; 9.3 layers the design's
    ``InvalidParams`` error mapping (Requirement 11.6) on top, and
    9.4's per-tool handlers may apply additional in-scope checks
    (Requirement 10.7).
    """

    return {
        "type": "object",
        "properties": {
            "gitlab_project_id": {
                "type": "integer",
                "description": (
                    "The numeric GitLab project ID of an in-scope project."
                ),
            },
        },
        "required": ["gitlab_project_id"],
    }


def _build_built_in_tools() -> tuple[mcp_types.Tool, ...]:
    """Build the eight built-in :class:`mcp.types.Tool` records.

    Defined as a function so each ``Tool`` instance owns its own
    ``inputSchema`` dict (the helpers above return fresh dicts on
    every call). The result is captured into the module-level
    :data:`BUILT_IN_TOOLS` constant once at import time.
    """

    return (
        mcp_types.Tool(
            name=TOOL_NAME_LIST_PROJECTS,
            description=(
                "Return all in-scope projects with their GitLab project "
                "ID and full path."
            ),
            inputSchema=_no_args_input_schema(),
        ),
        mcp_types.Tool(
            name=TOOL_NAME_GET_PROJECT_PURPOSE,
            description=(
                "Return the purpose summary for a project identified "
                "by its GitLab project ID."
            ),
            inputSchema=_gitlab_project_id_input_schema(),
        ),
        mcp_types.Tool(
            name=TOOL_NAME_GET_PROJECT_IO,
            description=(
                "Return the abstract inputs and abstract outputs for a "
                "project identified by its GitLab project ID."
            ),
            inputSchema=_gitlab_project_id_input_schema(),
        ),
        mcp_types.Tool(
            name=TOOL_NAME_GET_PROJECT_DEPENDENCIES,
            description=(
                "Return the external service dependencies and database "
                "table dependencies for a project identified by its "
                "GitLab project ID."
            ),
            inputSchema=_gitlab_project_id_input_schema(),
        ),
        mcp_types.Tool(
            name=TOOL_NAME_GET_PROJECT_PROFILE,
            description=(
                "Return the full Project_Profile for a project "
                "identified by its GitLab project ID."
            ),
            inputSchema=_gitlab_project_id_input_schema(),
        ),
        mcp_types.Tool(
            name=TOOL_NAME_LIST_PURPOSE_CONFLICTS,
            description=(
                "Return the list of all project pairs whose purpose "
                "summaries indicate a Purpose_Conflict."
            ),
            inputSchema=_no_args_input_schema(),
        ),
        mcp_types.Tool(
            name=TOOL_NAME_REFRESH_ALL_PROJECTS,
            description=(
                "Trigger an Ingestion_Job that refreshes every "
                "in-scope project."
            ),
            inputSchema=_no_args_input_schema(),
        ),
        mcp_types.Tool(
            name=TOOL_NAME_REFRESH_PROJECT,
            description=(
                "Trigger an Ingestion_Job that refreshes a single "
                "project identified by its GitLab project ID."
            ),
            inputSchema=_gitlab_project_id_input_schema(),
        ),
    )


#: The fixed set of eight tools the MCP_Server advertises in every
#: ``tools/list`` response. Order follows the design's tool table so
#: clients that render tools in declaration order see a stable list.
#:
#: Per Requirement 11.3, the contents of this tuple are the exact
#: tool set returned to every solicited ``tools/list`` request. The
#: tuple is module-level (and therefore built once at import time)
#: so identity-based assertions in tests remain stable across
#: multiple ``tools/list`` calls.
BUILT_IN_TOOLS: Final[tuple[mcp_types.Tool, ...]] = _build_built_in_tools()


#: Lookup map from tool name to the canonical :class:`mcp.types.Tool`
#: record. Built once at import time from :data:`BUILT_IN_TOOLS` so
#: ``tools/call`` dispatch can resolve a request's tool name to the
#: tool's ``inputSchema`` in O(1). Any tool name absent from this map
#: is treated as an unknown tool by the dispatcher (Requirement 11.5).
_TOOL_BY_NAME: Final[dict[str, mcp_types.Tool]] = {
    tool.name: tool for tool in BUILT_IN_TOOLS
}


# ---------------------------------------------------------------------------
# Collaborator protocols
# ---------------------------------------------------------------------------

# ``classify_pair`` and ``find_all_conflicts`` are injected as plain
# callables (not via a service object) because the production
# implementation in :mod:`project_knowledge_mcp.conflict_detector` is
# already a pure module-level function. Using ``Protocol``s keeps the
# coupling free of import cycles and lets unit tests pass purpose-built
# fakes without subclassing.


class ClassifyPairCallable(Protocol):
    """Callable signature matching :func:`conflict_detector.classify_pair`."""

    def __call__(
        self,
        profile_a: ProjectProfile,
        profile_b: ProjectProfile,
    ) -> ConflictResult: ...


class FindAllConflictsCallable(Protocol):
    """Callable signature matching :func:`conflict_detector.find_all_conflicts`."""

    def __call__(
        self,
        profiles: Sequence[ProjectProfile],
    ) -> list[ConflictPair]: ...


#: Per-tool handler signature used by :meth:`MCPServer.register_tool`.
#:
#: Tool handlers receive the validated ``arguments`` mapping (the SDK
#: has already validated it against the tool's ``inputSchema`` by the
#: time the handler is invoked, see 9.3) and return a fully-formed
#: :class:`mcp.types.CallToolResult`. Returning the ``CallToolResult``
#: directly (rather than raw content) gives 9.4's tool implementations
#: full control over the ``isError``/``content``/``structuredContent``
#: shape - in particular over Requirements 10.7 and 11.7's "tool
#: execution failed" results.
ToolHandler: TypeAlias = "Callable[[dict[str, Any]], Awaitable[mcp_types.CallToolResult]]"


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------


def _resolve_version() -> str:
    """Return the server version surfaced in the ``initialize`` response.

    Attempts to read the version from installed package metadata
    (``importlib.metadata``) first, so an installed wheel always
    advertises its real version. Falls back to :data:`FALLBACK_VERSION`
    when metadata is not available (for example when running from a
    source tree that has not been ``pip install``-ed).
    """

    try:
        return importlib_metadata.version(_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError:
        return FALLBACK_VERSION


# ---------------------------------------------------------------------------
# Argument-validation error mapping
# ---------------------------------------------------------------------------


def _failing_argument_name(
    exc: jsonschema.ValidationError, instance: dict[str, Any]
) -> str:
    """Identify the name of the argument that failed schema validation.

    Used by the ``tools/call`` dispatcher to populate the ``argument``
    field of the :class:`mcp.types.ErrorData.data` object that
    accompanies an ``InvalidParams`` JSON-RPC error response per
    Requirement 11.6 ("names the failing argument").

    The resolution rules below are derived from the shape of
    :class:`jsonschema.ValidationError`:

    * For per-field rule failures (``type``, ``minimum``, ``pattern``,
      ...) the failing field is the first segment of
      :attr:`~jsonschema.ValidationError.absolute_path`. For example,
      ``{"gitlab_project_id": "not-an-int"}`` failing the ``type``
      rule yields ``absolute_path == deque(['gitlab_project_id'])``.
    * For top-level ``required`` rule failures the path is empty (the
      error is reported on the parent object), but
      :attr:`~jsonschema.ValidationError.validator_value` carries the
      list of required field names. The first required name absent
      from ``instance`` is the one the rule rejected.
    * If neither approach identifies a specific argument (e.g. an
      ``additionalProperties`` failure on a tool whose schema does not
      allow extra fields, or any other root-level constraint the
      dispatcher does not need to specially recognize for the eight
      built-in tools), the empty string is returned. The dispatcher
      still surfaces a structured ``data`` object so clients see a
      machine-readable rule name.
    """

    if exc.absolute_path:
        # The path is a deque of path segments (strings for object
        # keys, integers for array indices). The first segment is the
        # top-level argument that failed.
        return str(exc.absolute_path[0])

    if exc.validator == "required":
        required = exc.validator_value
        if isinstance(required, list):
            for name in required:
                if isinstance(name, str) and name not in instance:
                    return name

    return ""


def _validation_rule_name(exc: jsonschema.ValidationError) -> str:
    """Identify the JSON Schema validator rule that rejected the input.

    Maps to the ``rule`` field of the ``InvalidParams`` ``data``
    object (Requirement 11.6, "the validation rule that failed").
    The rule name is taken from
    :attr:`~jsonschema.ValidationError.validator` (e.g. ``"type"``,
    ``"required"``, ``"minimum"``, ``"pattern"``). When the validator
    name is not available - which jsonschema's typing allows but the
    library never exhibits in practice for the schemas attached to
    :data:`BUILT_IN_TOOLS` - the empty string is returned so the
    ``data`` object still carries a string value for ``rule``.
    """

    return exc.validator if isinstance(exc.validator, str) else ""


# ---------------------------------------------------------------------------
# MCPServer
# ---------------------------------------------------------------------------


class MCPServer:
    """The Project Knowledge MCP Server's MCP transport layer.

    Wraps :class:`mcp.server.lowlevel.Server`, owns the stdio bind, and
    exposes a single :meth:`register_tool` extension point for tasks
    9.2-9.4 to populate. The class does not perform any business logic
    of its own; it is a thin transport adapter that:

    1. constructs an SDK ``Server`` with the configured ``name`` and
       ``version`` so ``initialize`` reports them verbatim
       (Requirement 11.2);
    2. registers a ``list_tools`` handler at construction time, which
       is what causes the SDK to advertise ``capabilities.tools`` in
       the ``initialize`` response (the ``{tools: {}}`` shape
       Requirement 11.2 mandates) - the handler returns the contents
       of :attr:`_tools` so 9.2 and 9.4 only need to populate the
       registry, not re-register with the SDK;
    3. registers a ``call_tool`` handler at construction time that
       dispatches to :attr:`_tool_handlers` - 9.3 will replace the
       placeholder body with the design's full error-mapping logic;
    4. owns the stdio transport lifecycle via :meth:`serve`.

    Collaborators are injected at construction so the class is unit
    testable (and so the wiring step in :mod:`main` controls the
    lifecycle of the underlying services). They are stored on the
    instance for later use by 9.4's tool handlers; this skeleton does
    not call them.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        catalog: ProjectCatalog,
        coordinator: IngestionCoordinator,
        classify_pair: ClassifyPairCallable,
        find_all_conflicts: FindAllConflictsCallable,
        *,
        name: str = SERVER_NAME,
        version: str | None = None,
    ) -> None:
        """Create the MCP transport adapter.

        Args:
            store: The shared ``Knowledge_Store`` used by every read
                tool (Requirement 14.1). 9.4's tool handlers look up
                profiles through this collaborator.
            catalog: The snapshot-scoped ``Project_Catalog``. 9.4's
                ``list_projects`` tool reads from this collaborator.
            coordinator: The single-flight ``Ingestion_Coordinator``.
                9.4's ``refresh_all_projects`` and ``refresh_project``
                tools dispatch to this collaborator.
            classify_pair: The pure pair-wise conflict classifier
                from :mod:`conflict_detector`. Stored for later use
                by 9.4's tools that may need ad-hoc classification.
            find_all_conflicts: The bulk conflict scanner from
                :mod:`conflict_detector`. 9.4's
                ``list_purpose_conflicts`` tool dispatches to this
                callable.
            name: The MCP server name. Defaults to
                :data:`SERVER_NAME`. Overridable for tests only.
            version: The MCP server version. ``None`` (the default)
                means "resolve from installed package metadata, then
                fall back to :data:`FALLBACK_VERSION`".
        """

        self._store = store
        self._catalog = catalog
        self._coordinator = coordinator
        self._classify_pair = classify_pair
        self._find_all_conflicts = find_all_conflicts

        self._name: Final[str] = name
        self._version: Final[str] = version if version is not None else _resolve_version()

        # Tool registry. Populated by ``register_tool`` (9.4).
        # ``_tools`` holds the schema/metadata that ``tools/list``
        # advertises and that the SDK uses to validate ``tools/call``
        # arguments. ``_tool_handlers`` holds the dispatch target for
        # each registered tool name. The two dicts are kept in lock
        # step by ``register_tool``.
        self._tools: dict[str, mcp_types.Tool] = {}
        self._tool_handlers: dict[str, ToolHandler] = {}

        # The underlying SDK server. ``Server`` is generic over
        # lifespan and request types but we use neither here, so the
        # ``Any``/``Any`` parameterization is intentional.
        self._server: Server[Any, Any] = Server(
            name=self._name,
            version=self._version,
        )

        # Wire the placeholder list/call handlers so the ``initialize``
        # response advertises ``capabilities.tools = {}`` (Requirement
        # 11.2). Tasks 9.2 and 9.3 will replace the bodies; the SDK
        # registration shape stays identical.
        self._register_default_handlers()

        # Wire the eight built-in tools' handlers (task 9.4). After
        # this call returns, ``tools/call`` for any name in
        # :data:`BUILT_IN_TOOLS` dispatches to a handler bound to the
        # injected collaborators (Requirements 8.1, 8.2, 10.1-10.7).
        # The wiring runs unconditionally at construction time so the
        # transport surface is fully operational the moment the
        # process entry point calls :meth:`serve`; collaborators are
        # captured by closure references on the handler methods and
        # are never re-resolved at dispatch time.
        self.register_built_in_tools()

    # ------------------------------------------------------------------
    # Public API used by 9.4 and by the process entry point
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The MCP server name surfaced in the ``initialize`` response."""

        return self._name

    @property
    def version(self) -> str:
        """The MCP server version surfaced in the ``initialize`` response."""

        return self._version

    def register_tool(self, tool: mcp_types.Tool, handler: ToolHandler) -> None:
        """Register one MCP tool's handler for ``tools/call`` dispatch.

        Called once per tool by the wiring step in 9.4. After this
        returns, ``tools/call`` for ``tool.name`` dispatches to
        ``handler`` after the dispatcher has validated the call's
        ``arguments`` against the canonical ``inputSchema`` from
        :data:`BUILT_IN_TOOLS`. The dispatcher uses
        :data:`BUILT_IN_TOOLS` (not the ``tool`` argument) as the
        source of truth for the schema, so 9.4's call site cannot
        widen the contract by passing a more permissive schema.

        Re-registering an existing tool name is rejected so that the
        wiring step cannot accidentally clobber a previously-wired
        tool: per the design every one of the eight tools has a
        distinct name and a single handler.

        Args:
            tool: The tool metadata. Currently kept on the instance
                for symmetry with :data:`BUILT_IN_TOOLS` and so future
                tasks can introspect the registered surface; the
                dispatcher does not consult this dict for schema
                resolution.
            handler: The async dispatch target. Receives the validated
                arguments mapping and returns a fully-formed
                :class:`mcp.types.CallToolResult`. Any exception the
                handler raises is converted by the dispatcher into a
                tool result with ``isError: true`` (Requirement 11.7).

        Raises:
            ValueError: when ``tool.name`` is already registered.
        """

        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        self._tool_handlers[tool.name] = handler

    async def serve(self) -> None:
        """Bind to stdio and run the MCP server loop.

        This is the entry point for the process wiring step
        (:mod:`main`). It opens the SDK's stdio transport (Requirement
        11.1), constructs the ``initialize`` payload (Requirement
        11.2), and runs the server until the client disconnects.

        Returns when the read stream is closed by the client (typical
        clean shutdown) or when the surrounding task group cancels the
        loop. Exceptions raised by the SDK propagate to the caller so
        the process entry point can translate them into the configured
        shutdown sequence.
        """

        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    # ------------------------------------------------------------------
    # Built-in tool registration (task 9.4)
    # ------------------------------------------------------------------

    def register_built_in_tools(self) -> None:
        """Register handlers for all eight built-in MCP tools.

        Wires each tool in :data:`BUILT_IN_TOOLS` to a method on this
        instance. The handlers close over the collaborators captured
        in :meth:`__init__` (``self._catalog``, ``self._store``,
        ``self._coordinator``, ``self._find_all_conflicts``); the
        registration is performed by name so the dispatcher's
        ``inputSchema`` source of truth (the canonical
        :data:`BUILT_IN_TOOLS` tuple) remains the only place a tool's
        argument shape is defined.

        Implements Requirements 8.1, 8.2, 10.1, 10.2, 10.3, 10.4,
        10.5, 10.6, 10.7. Targets Property 17 (project-id-typed MCP
        tools reject out-of-scope IDs).

        Raises:
            ValueError: if any of the eight tool names is already
                registered (e.g. when this method is called more than
                once on the same instance). The constructor calls
                this method exactly once; external callers should not
                invoke it again.
        """

        registrations: tuple[tuple[str, ToolHandler], ...] = (
            (TOOL_NAME_LIST_PROJECTS, self._handle_list_projects),
            (TOOL_NAME_GET_PROJECT_PURPOSE, self._handle_get_project_purpose),
            (TOOL_NAME_GET_PROJECT_IO, self._handle_get_project_io),
            (TOOL_NAME_GET_PROJECT_DEPENDENCIES, self._handle_get_project_dependencies),
            (TOOL_NAME_GET_PROJECT_PROFILE, self._handle_get_project_profile),
            (TOOL_NAME_LIST_PURPOSE_CONFLICTS, self._handle_list_purpose_conflicts),
            (TOOL_NAME_REFRESH_ALL_PROJECTS, self._handle_refresh_all_projects),
            (TOOL_NAME_REFRESH_PROJECT, self._handle_refresh_project),
        )
        for tool_name, handler in registrations:
            self.register_tool(_TOOL_BY_NAME[tool_name], handler)

    # ------------------------------------------------------------------
    # Tool handlers (task 9.4)
    # ------------------------------------------------------------------
    #
    # Each handler is an async method taking the SDK-validated
    # ``arguments`` mapping and returning a fully-formed
    # :class:`mcp.types.CallToolResult`. The dispatcher in
    # :meth:`_dispatch_tool_call` has already validated arguments
    # against the tool's ``inputSchema`` by the time the handler is
    # invoked, so handlers may access required arguments without
    # re-validation.
    #
    # Result shape conventions:
    #
    # * Every handler emits exactly one ``TextContent`` block whose
    #   ``text`` is the JSON-serialized form of the same payload also
    #   surfaced via ``structuredContent``. This keeps text-mode
    #   clients (which read ``content``) and structured-mode clients
    #   (which read ``structuredContent``) in lockstep.
    # * Read tools that take a ``gitlab_project_id`` short-circuit
    #   with ``isError: true`` and the canonical
    #   ``"project {id} is not in scope"`` message before consulting
    #   the ``Knowledge_Store`` (Requirement 10.7 / Property 17).
    # * Refresh tools surface ``IngestionInProgressError`` and (for
    #   ``refresh_project``) ``ProjectNotInScopeError`` as tool
    #   results with ``isError: true`` rather than letting them
    #   propagate to the dispatcher's "tool execution failed: ..."
    #   wrapper, so the wire shape matches the design's MCP error
    #   mapping table verbatim.

    async def _handle_list_projects(
        self,
        arguments: dict[str, Any],  # noqa: ARG002 - matches ToolHandler signature
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call list_projects``.

        Returns every in-scope project from the current snapshot,
        ordered by ``gitlab_project_id`` ascending (the order the
        ``Project_Catalog`` already returns). The empty list is
        returned as ``{"projects": []}`` when no ``Ingestion_Job``
        has committed a snapshot yet — the same empty-state the
        Visualization_Server surfaces (Requirement 14.4).

        Implements Requirement 10.6.
        """

        projects = self._catalog.list_in_scope()
        payload: dict[str, Any] = {
            "projects": [project.model_dump(mode="json") for project in projects],
        }
        return _structured_result(payload)

    async def _handle_get_project_purpose(
        self, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call get_project_purpose``.

        Returns the ``purpose_summary`` and ``purpose_summary_reason``
        for the requested project. Returns the canonical
        out-of-scope error (Requirement 10.7) when the project is not
        in the current ``Project_Catalog``, and a non-error
        "not yet analyzed" result (matching the design's empty state
        for Requirement 14.3) when the project is in scope but the
        current snapshot has no profile for it.

        Implements Requirements 10.1 and 10.7.
        """

        gitlab_project_id = int(arguments["gitlab_project_id"])
        if not self._catalog.is_in_scope(gitlab_project_id):
            return _not_in_scope_result(gitlab_project_id)

        profile = self._store.get_profile(gitlab_project_id)
        if profile is None:
            return _not_yet_analyzed_result(gitlab_project_id)

        payload: dict[str, Any] = {
            "gitlab_project_id": profile.gitlab_project_id,
            "purpose_summary": profile.purpose_summary,
            "purpose_summary_reason": profile.purpose_summary_reason,
        }
        return _structured_result(payload)

    async def _handle_get_project_io(
        self, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call get_project_io``.

        Returns the ``Abstract_Inputs`` and ``Abstract_Outputs`` for
        the requested project. Both lists are always present in the
        result (possibly empty per Requirements 4.5 and 4.6).

        Implements Requirements 10.2 and 10.7.
        """

        gitlab_project_id = int(arguments["gitlab_project_id"])
        if not self._catalog.is_in_scope(gitlab_project_id):
            return _not_in_scope_result(gitlab_project_id)

        profile = self._store.get_profile(gitlab_project_id)
        if profile is None:
            return _not_yet_analyzed_result(gitlab_project_id)

        payload: dict[str, Any] = {
            "gitlab_project_id": profile.gitlab_project_id,
            "abstract_inputs": [item.model_dump(mode="json") for item in profile.abstract_inputs],
            "abstract_outputs": [
                item.model_dump(mode="json") for item in profile.abstract_outputs
            ],
        }
        return _structured_result(payload)

    async def _handle_get_project_dependencies(
        self, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call get_project_dependencies``.

        Returns the ``External_Service_Dependencies`` and the
        ``Database_Table_Dependencies`` for the requested project.
        Both lists are always present in the result (possibly empty
        per Requirements 5.4 and 6.4).

        Implements Requirements 10.3 and 10.7.
        """

        gitlab_project_id = int(arguments["gitlab_project_id"])
        if not self._catalog.is_in_scope(gitlab_project_id):
            return _not_in_scope_result(gitlab_project_id)

        profile = self._store.get_profile(gitlab_project_id)
        if profile is None:
            return _not_yet_analyzed_result(gitlab_project_id)

        payload: dict[str, Any] = {
            "gitlab_project_id": profile.gitlab_project_id,
            "external_service_dependencies": [
                dep.model_dump(mode="json") for dep in profile.external_service_dependencies
            ],
            "database_table_dependencies": [
                dep.model_dump(mode="json") for dep in profile.database_table_dependencies
            ],
        }
        return _structured_result(payload)

    async def _handle_get_project_profile(
        self, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call get_project_profile``.

        Returns the full ``Project_Profile`` for the requested
        project, serialized through Pydantic's JSON-compatible
        ``model_dump(mode="json")`` so datetime and enum fields are
        rendered as the same strings the on-the-wire JSON-RPC layer
        emits.

        Implements Requirements 10.5 and 10.7.
        """

        gitlab_project_id = int(arguments["gitlab_project_id"])
        if not self._catalog.is_in_scope(gitlab_project_id):
            return _not_in_scope_result(gitlab_project_id)

        profile = self._store.get_profile(gitlab_project_id)
        if profile is None:
            return _not_yet_analyzed_result(gitlab_project_id)

        payload: dict[str, Any] = profile.model_dump(mode="json")
        return _structured_result(payload)

    async def _handle_list_purpose_conflicts(
        self,
        arguments: dict[str, Any],  # noqa: ARG002 - matches ToolHandler signature
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call list_purpose_conflicts``.

        Reads every profile in the current snapshot and runs
        ``Conflict_Detector.find_all_conflicts`` over the result. The
        ``conflicts`` list mirrors the ``ConflictPair`` records the
        detector produces (canonical ascending project-ID ordering;
        one entry per unordered pair).

        Implements Requirement 10.4.
        """

        profiles = self._store.list_profiles()
        conflict_pairs = self._find_all_conflicts(profiles)
        payload: dict[str, Any] = {
            "conflicts": [
                {
                    "project_id_a": pair.project_id_a,
                    "project_id_b": pair.project_id_b,
                    "justification": pair.justification,
                }
                for pair in conflict_pairs
            ],
        }
        return _structured_result(payload)

    async def _handle_refresh_all_projects(
        self,
        arguments: dict[str, Any],  # noqa: ARG002 - matches ToolHandler signature
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call refresh_all_projects``.

        Drives ``Ingestion_Coordinator.start_full_refresh`` to
        completion in a worker thread so the synchronous coordinator
        does not block the stdio event loop. ``IngestionInProgressError``
        is surfaced as a tool result with ``isError: true`` whose
        message is the canonical phrase from
        :class:`IngestionInProgressError` (Requirement 8.6); other
        exceptions propagate to the dispatcher and become
        ``"tool execution failed: ..."`` results (Requirement 11.7).

        Implements Requirement 8.1 and Requirement 8.6.
        """

        try:
            await asyncio.to_thread(self._coordinator.start_full_refresh)
        except IngestionInProgressError as exc:
            return _error_result(exc.message)

        payload: dict[str, Any] = {"status": "completed", "trigger": "full"}
        return _structured_result(payload)

    async def _handle_refresh_project(
        self, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        """Handle ``tools/call refresh_project``.

        Drives ``Ingestion_Coordinator.start_single_project_refresh``
        in a worker thread. ``ProjectNotInScopeError`` is surfaced
        with the canonical "project {id} is not in scope" wording
        (Requirement 10.7) so out-of-scope IDs return the same
        message regardless of whether the rejection came from the
        catalog check or from a stale parent snapshot detected by
        the coordinator. ``IngestionInProgressError`` is surfaced
        with the canonical phrase from the design's error mapping
        (Requirement 8.6); other exceptions propagate to the
        dispatcher (Requirement 11.7).

        Implements Requirements 8.2, 8.6, and 10.7.
        """

        gitlab_project_id = int(arguments["gitlab_project_id"])
        try:
            await asyncio.to_thread(
                self._coordinator.start_single_project_refresh,
                gitlab_project_id,
            )
        except ProjectNotInScopeError as exc:
            # Use the offending id from the exception so a coordinator
            # that learned about a stale parent ID still returns a
            # message that names the id the caller actually asked
            # about.
            return _not_in_scope_result(exc.gitlab_project_id)
        except IngestionInProgressError as exc:
            return _error_result(exc.message)

        payload: dict[str, Any] = {
            "status": "completed",
            "trigger": "single_project",
            "gitlab_project_id": gitlab_project_id,
        }
        return _structured_result(payload)

    # ------------------------------------------------------------------
    # Internal handler wiring
    # ------------------------------------------------------------------

    def _register_default_handlers(self) -> None:
        """Register the ``list_tools`` and ``call_tool`` handlers.

        * ``list_tools`` returns :data:`BUILT_IN_TOOLS` verbatim. The
          eight built-in tools are part of the server's contract, so
          ``tools/list`` advertises them whether or not 9.4 has wired
          their handlers yet (Requirement 11.3). The SDK invokes this
          handler only in response to an explicit ``tools/list``
          request from the client, so the server never emits an
          unsolicited ``tools/list`` response.

          Registering this handler also flips on the ``tools`` server
          capability that the ``initialize`` response advertises
          (Requirement 11.2), via :meth:`Server.get_capabilities`.

        * ``call_tool`` is the dispatcher that satisfies the design's
          ``tools/call`` error mapping (Requirements 11.4-11.7):

          - **unknown tool name** raises
            :class:`mcp.shared.exceptions.McpError` with code
            :data:`mcp.types.METHOD_NOT_FOUND` and message
            ``"tool '{name}' is unknown"`` (Requirement 11.5);

          - **argument-schema violation** raises
            :class:`McpError` with code
            :data:`mcp.types.INVALID_PARAMS`, a message that names the
            failing argument and rule, and a structured
            :attr:`~mcp.types.ErrorData.data` object containing
            ``{"argument": ..., "rule": ...}`` (Requirement 11.6);

          - **handler runtime / external-dependency failure** is
            caught and returned as a tool result with
            ``isError: true`` and message
            ``"tool execution failed: {reason}"`` (Requirement 11.7);

          - **happy path** returns the registered handler's
            :class:`mcp.types.CallToolResult` verbatim (Requirement
            11.4).

          The dispatcher is registered directly in
          :attr:`Server.request_handlers` rather than via the SDK's
          ``@Server.call_tool()`` decorator. The decorator wraps every
          handler in a ``try/except Exception`` block that converts
          :class:`McpError` (and every other exception) into an
          ``isError: true`` :class:`mcp.types.CallToolResult`, which
          would silently downgrade the JSON-RPC errors that
          Requirements 11.5 and 11.6 demand. By registering directly,
          the dispatcher's :class:`McpError` propagates to the SDK's
          ``_handle_request`` (which catches :class:`McpError` and
          maps ``err.error`` onto a :class:`mcp.types.JSONRPCError`
          response) so unknown-tool and invalid-params errors arrive
          on the wire as proper JSON-RPC error responses.

          Tool-handler-runtime exceptions are caught inside the
          dispatcher (rather than allowed to propagate) precisely so
          they do **not** become JSON-RPC errors: Requirement 11.7
          mandates a tool result with ``isError: true``, which is what
          this branch returns.

        Registration uses the SDK's :meth:`Server.list_tools`
        decorator so the SDK can wire the list handler into
        ``request_handlers`` and flip on the corresponding capability
        bit (:meth:`Server.get_capabilities`); the call-tool
        dispatcher is registered manually for the reasons above.
        """

        # Registration uses the SDK's decorator API; the decorator
        # factories on ``Server`` are not fully typed in the published
        # SDK, so the calls and decorator applications are narrowly
        # ignored. The wired handlers themselves remain fully typed.
        @self._server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def _list_tools() -> list[mcp_types.Tool]:
            # The SDK invokes this handler only when the client sends
            # an explicit ``tools/list`` request (one invocation per
            # request, Requirement 11.3). The body is unconditional
            # by design: every solicited ``tools/list`` response
            # carries exactly the eight built-in tools defined by
            # Requirements 8.1, 8.2, and 10.1-10.6, so the contents
            # do not depend on whether 9.4 has registered handlers
            # yet. ``list(BUILT_IN_TOOLS)`` returns a fresh list each
            # time so the SDK is free to mutate the returned list
            # while serializing without affecting the canonical
            # tuple.
            return list(BUILT_IN_TOOLS)

        # Register the ``tools/call`` dispatcher directly in the
        # SDK's request-handler map. We bypass the ``@call_tool()``
        # decorator on purpose: the decorator's wrapper converts
        # every raised exception (including ``McpError``) into an
        # ``isError: true`` ``CallToolResult``, which would silently
        # downgrade the JSON-RPC ``MethodNotFound`` /
        # ``InvalidParams`` errors that Requirements 11.5 and 11.6
        # demand. By registering directly we let ``McpError``
        # propagate to ``_handle_request`` (which catches it and
        # maps ``err.error`` onto a real ``JSONRPCError`` response),
        # while still catching tool-handler-runtime exceptions
        # ourselves and turning them into the ``isError: true`` tool
        # results Requirement 11.7 mandates.
        self._server.request_handlers[mcp_types.CallToolRequest] = (
            self._dispatch_tool_call
        )

    async def _dispatch_tool_call(
        self, req: mcp_types.CallToolRequest
    ) -> mcp_types.ServerResult:
        """Dispatch a ``tools/call`` request per the design's error mapping.

        Implements Requirements 11.4, 11.5, 11.6, 11.7 (Property 19).

        Steps:

        1. Resolve the tool name in :data:`_TOOL_BY_NAME`. An unknown
           name raises :class:`mcp.shared.exceptions.McpError` with
           code :data:`mcp.types.METHOD_NOT_FOUND`, which the SDK's
           ``_handle_request`` turns into a JSON-RPC error response
           whose ``message`` is ``"tool '{name}' is unknown"``
           (Requirement 11.5).
        2. Validate ``arguments`` against the tool's
           ``inputSchema`` using :mod:`jsonschema`. A failure raises
           :class:`McpError` with code
           :data:`mcp.types.INVALID_PARAMS`, a message naming the
           failing argument and rule, and a structured ``data``
           object ``{"argument": <name>, "rule": <validator>}``
           (Requirement 11.6).
        3. Look up the registered handler in
           :attr:`_tool_handlers`. If no handler is registered yet
           (the wiring step in 9.4 has not been applied) the
           dispatcher returns an ``isError: true`` tool result whose
           message follows the Requirement 11.7 wording. This branch
           never fires in production once 9.4 has wired all eight
           handlers; it exists so the surface stays predictable
           during incremental wiring and during partial test
           builds.
        4. Invoke the handler. Any exception raised by the handler
           is caught and turned into a tool result with
           ``isError: true`` and message
           ``"tool execution failed: {reason}"`` where ``reason`` is
           ``str(exc)`` (Requirement 11.7). This is the path
           travelled by ``KnowledgeStoreUnavailableError``,
           ``GitLabAuthError``, and any other
           :class:`project_knowledge_mcp.errors.ProjectKnowledgeError`
           subclass that 9.4's tool handlers may surface.
        5. On success, return the handler's
           :class:`mcp.types.CallToolResult` verbatim wrapped in a
           :class:`mcp.types.ServerResult` (Requirement 11.4).

        The dispatcher matches the SDK's expected handler signature
        for :data:`mcp.types.CallToolRequest` in
        :attr:`Server.request_handlers`: an awaitable that takes the
        request and returns a :class:`mcp.types.ServerResult`.
        """

        name = req.params.name
        # ``arguments`` is ``Optional[dict]`` on the wire; treat a
        # missing or null payload as the empty argument map so a
        # no-args tool call validates cleanly against an empty-object
        # schema.
        arguments: dict[str, Any] = req.params.arguments or {}

        # 1. Resolve the tool name. Unknown name -> JSON-RPC
        #    MethodNotFound (Requirement 11.5).
        tool = _TOOL_BY_NAME.get(name)
        if tool is None:
            raise McpError(
                mcp_types.ErrorData(
                    code=mcp_types.METHOD_NOT_FOUND,
                    message=f"tool '{name}' is unknown",
                )
            )

        # 2. Validate arguments against the tool's input schema.
        #    Failure -> JSON-RPC InvalidParams whose ``data`` names
        #    the failing argument and the rule that rejected it
        #    (Requirement 11.6).
        try:
            jsonschema.validate(instance=arguments, schema=tool.inputSchema)
        except jsonschema.ValidationError as exc:
            argument_name = _failing_argument_name(exc, arguments)
            rule = _validation_rule_name(exc)
            raise McpError(
                mcp_types.ErrorData(
                    code=mcp_types.INVALID_PARAMS,
                    message=(
                        f"argument '{argument_name}' for tool '{name}' "
                        f"failed validation rule '{rule}'"
                    ),
                    data={"argument": argument_name, "rule": rule},
                )
            ) from exc

        # 3. Look up the registered handler. Missing handler is an
        #    intra-process wiring error, not a protocol-level error;
        #    surface it as a tool result with ``isError: true`` so
        #    the message wording stays in the Requirement 11.7
        #    family rather than escalating to a JSON-RPC error.
        handler = self._tool_handlers.get(name)
        if handler is None:
            return mcp_types.ServerResult(
                mcp_types.CallToolResult(
                    content=[
                        mcp_types.TextContent(
                            type="text",
                            text=(
                                "tool execution failed: handler not "
                                "registered"
                            ),
                        )
                    ],
                    isError=True,
                )
            )

        # 4. Dispatch the handler. Catch every exception so a
        #    handler-internal failure (KnowledgeStoreUnavailableError,
        #    GitLabAuthError, ...) surfaces as a tool result rather
        #    than as a JSON-RPC error, per Requirement 11.7.
        try:
            result = await handler(arguments)
        except Exception as exc:
            # Requirement 11.7: any handler-internal failure (broad
            # ``Exception`` catch-all by design) becomes a tool result
            # with ``isError: true``, never a JSON-RPC error.
            return mcp_types.ServerResult(
                mcp_types.CallToolResult(
                    content=[
                        mcp_types.TextContent(
                            type="text",
                            text=f"tool execution failed: {exc}",
                        )
                    ],
                    isError=True,
                )
            )

        # 5. Happy path: forward the handler's CallToolResult.
        return mcp_types.ServerResult(result)

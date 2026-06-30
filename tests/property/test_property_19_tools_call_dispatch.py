# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 19: For all tools/call requests, the response SHALL satisfy: if the tool name is in the defined set and arguments validate, the response is a tool result produced by the tool's handler, if the tool name is not in the defined set, the response is an MCP error response indicating the tool is unknown, if arguments fail input-schema validation, the response is an MCP error response that names the failing argument and the validation rule that failed, if the handler raises a runtime or external-dependency failure, the response is a tool result with isError: true whose message names the failure reason.
"""Property test for ``tools/call`` dispatch correctness.

**Validates Requirements 11.4, 11.5, 11.6, 11.7** (Property 19 in the
design).

For every randomly generated ``tools/call`` request, the dispatcher in
:meth:`project_knowledge_mcp.mcp_server.MCPServer._dispatch_tool_call`
SHALL satisfy exactly one of four invariants depending on the request:

1. **Happy path** (Requirement 11.4) -- valid tool name + arguments
   that pass the tool's ``inputSchema`` validation: the dispatcher
   returns the registered handler's :class:`mcp.types.CallToolResult`
   verbatim, wrapped in a :class:`mcp.types.ServerResult`.
2. **Unknown tool name** (Requirement 11.5) -- the dispatcher raises
   :class:`mcp.shared.exceptions.McpError` with code
   :data:`mcp.types.METHOD_NOT_FOUND` and a message that names the
   offending tool name as unknown. (The SDK's ``_handle_request``
   maps this onto a JSON-RPC error response on the wire; the
   dispatcher itself just raises.)
3. **Invalid arguments** (Requirement 11.6) -- the dispatcher raises
   :class:`McpError` with code :data:`mcp.types.INVALID_PARAMS`,
   carries the failing argument name and the validation rule that
   rejected it on its structured ``data`` payload (and in the message
   text), and the message names the requested tool.
4. **Handler runtime / external-dependency failure** (Requirement
   11.7) -- the dispatcher catches the exception raised by the
   handler and returns a :class:`mcp.types.CallToolResult` with
   ``isError: true`` whose ``TextContent`` text begins with the
   canonical "tool execution failed:" prefix and contains
   ``str(exc)`` so the failure reason is named verbatim. The
   exceptions injected here are the design's two production-shaped
   external-dependency failures:
   :class:`project_knowledge_mcp.errors.KnowledgeStoreUnavailableError`
   and :class:`project_knowledge_mcp.errors.GitLabAuthError`.

The dispatcher under test is exercised directly (rather than over
the stdio transport) so the property exercises exactly the wire-shape
mapping logic without taking on a subprocess dependency. The MCP
SDK's :meth:`mcp.server.lowlevel.Server.run` invokes the same
:meth:`_dispatch_tool_call` method with the same
:class:`mcp.types.CallToolRequest` shape, so an invariant that holds
here also holds for live MCP sessions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from mcp.shared.exceptions import McpError

from project_knowledge_mcp.errors import (
    GitLabAuthError,
    KnowledgeStoreUnavailableError,
)
from project_knowledge_mcp.mcp_server import (
    BUILT_IN_TOOLS,
    TOOL_NAME_GET_PROJECT_DEPENDENCIES,
    TOOL_NAME_GET_PROJECT_IO,
    TOOL_NAME_GET_PROJECT_PROFILE,
    TOOL_NAME_GET_PROJECT_PURPOSE,
    TOOL_NAME_LIST_PROJECTS,
    TOOL_NAME_LIST_PURPOSE_CONFLICTS,
    TOOL_NAME_REFRESH_ALL_PROJECTS,
    TOOL_NAME_REFRESH_PROJECT,
    MCPServer,
    ToolHandler,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_knowledge_mcp.conflict_detector import ConflictPair
    from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator
    from project_knowledge_mcp.knowledge_store import KnowledgeStore
    from project_knowledge_mcp.models import ConflictResult, ProjectProfile
    from project_knowledge_mcp.project_catalog import ProjectCatalog


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Static partition of the eight built-in tools by argument shape
# ---------------------------------------------------------------------------

#: Tools whose ``inputSchema`` requires a ``gitlab_project_id`` integer.
#: Used by the strategies below to generate both valid and invalid
#: argument payloads that exercise Requirements 11.4 and 11.6.
PROJECT_ID_TOOLS: tuple[str, ...] = (
    TOOL_NAME_GET_PROJECT_PURPOSE,
    TOOL_NAME_GET_PROJECT_IO,
    TOOL_NAME_GET_PROJECT_DEPENDENCIES,
    TOOL_NAME_GET_PROJECT_PROFILE,
    TOOL_NAME_REFRESH_PROJECT,
)

#: Tools whose ``inputSchema`` accepts an empty argument object.
NO_ARGS_TOOLS: tuple[str, ...] = (
    TOOL_NAME_LIST_PROJECTS,
    TOOL_NAME_LIST_PURPOSE_CONFLICTS,
    TOOL_NAME_REFRESH_ALL_PROJECTS,
)

#: Lookup of every tool name advertised by ``BUILT_IN_TOOLS``. Used by
#: the unknown-tool-name strategy to filter out names that *are* in
#: the defined set.
KNOWN_TOOL_NAMES: frozenset[str] = frozenset(tool.name for tool in BUILT_IN_TOOLS)


# ---------------------------------------------------------------------------
# Synthetic handlers
# ---------------------------------------------------------------------------

#: Marker text returned by the happy-path handler. Carrying a unique,
#: human-readable string lets the assertion ``"the dispatcher returned
#: the handler's CallToolResult verbatim"`` (Requirement 11.4) avoid
#: any false positives from other code paths that might construct a
#: CallToolResult of their own.
HAPPY_PATH_TEXT: str = "happy-path-sentinel-result"


def _happy_handler() -> ToolHandler:
    """Return a handler that returns a known sentinel ``CallToolResult``.

    Used to exercise Requirement 11.4: when arguments validate, the
    dispatcher SHALL return the handler's ``CallToolResult`` verbatim.
    The sentinel text and the structured echo of the validated
    arguments make the "verbatim" check exact -- the test asserts the
    returned result is the same object the handler produced (or, at
    least, carries the sentinel and the echoed arguments).
    """

    async def handler(arguments: dict[str, Any]) -> mcp_types.CallToolResult:
        # Echo the validated arguments through ``structuredContent`` so
        # the test can also confirm the dispatcher passed them through
        # without mutation.
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=HAPPY_PATH_TEXT)],
            structuredContent={"received": arguments},
            isError=False,
        )

    return handler


def _failing_handler(exc: BaseException) -> ToolHandler:
    """Return a handler that raises ``exc`` when invoked.

    Used to exercise Requirement 11.7: when the handler raises a
    runtime / external-dependency failure, the dispatcher SHALL
    return a tool result with ``isError: true`` whose message names
    the failure reason (i.e. ``str(exc)``).
    """

    async def handler(arguments: dict[str, Any]) -> mcp_types.CallToolResult:
        # ``arguments`` is intentionally unused: the failing
        # handler raises before inspecting the payload.
        del arguments
        raise exc

    return handler


# ---------------------------------------------------------------------------
# Stand-in collaborators
# ---------------------------------------------------------------------------

# The dispatcher under test never consults the injected store /
# catalog / coordinator / conflict-detector callables -- it only
# resolves the tool name, validates arguments, and dispatches into
# whatever handler is registered in ``_tool_handlers``. The tests
# below replace those handlers with synthetic ones (see
# :func:`_install_handler`), so the real collaborators can be
# stand-ins. Failing the assertion if any of them is invoked makes
# any accidental coupling loud rather than silent.


def _unused_classify_pair(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
) -> ConflictResult:  # pragma: no cover - guarded by assertion
    raise AssertionError(
        "classify_pair must not be called by the tools/call dispatcher"
    )


def _unused_find_all_conflicts(
    profiles: Sequence[ProjectProfile],
) -> list[ConflictPair]:  # pragma: no cover - guarded by assertion
    raise AssertionError(
        "find_all_conflicts must not be called by the tools/call dispatcher"
    )


def _build_server() -> MCPServer:
    """Construct an :class:`MCPServer` with stand-in collaborators.

    The constructor wires the eight production tool handlers via
    :meth:`MCPServer.register_built_in_tools`; tests then overwrite
    the entries in :attr:`MCPServer._tool_handlers` for whatever
    tool name they need to exercise. The store / catalog /
    coordinator stand-ins are typed via :func:`cast` because the
    dispatcher under test never calls them.
    """

    return MCPServer(
        store=cast("KnowledgeStore", object()),
        catalog=cast("ProjectCatalog", object()),
        coordinator=cast("IngestionCoordinator", object()),
        classify_pair=_unused_classify_pair,
        find_all_conflicts=_unused_find_all_conflicts,
        version="0.0.0-test",
    )


def _install_handler(
    server: MCPServer, tool_name: str, handler: ToolHandler
) -> None:
    """Replace the registered handler for ``tool_name`` on ``server``.

    The dispatcher resolves handlers through
    :attr:`MCPServer._tool_handlers`. Overwriting the entry is the
    smallest possible change that lets the test inject either a
    happy-path or a failing handler without re-running the full
    construction wiring.
    """

    server._tool_handlers[tool_name] = handler


# ---------------------------------------------------------------------------
# The Case ADT and its strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ValidCase:
    """A ``tools/call`` request that should hit the handler's happy path."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class _UnknownToolCase:
    """A ``tools/call`` request whose tool name is not in the defined set."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class _InvalidArgsCase:
    """A ``tools/call`` request whose arguments fail schema validation."""

    tool_name: str
    arguments: dict[str, Any]
    expected_argument: str
    expected_rule: str


@dataclass(frozen=True)
class _HandlerFailureCase:
    """A ``tools/call`` request whose handler raises an injected error."""

    tool_name: str
    arguments: dict[str, Any]
    # The exception raised inside the handler. We carry its full
    # ``str()`` form on the case so the assertion that the
    # dispatcher's response message "names the failure reason" can
    # compare against the exact wording the design's error type
    # produces (Requirements 11.7).
    error: BaseException


_Case = _ValidCase | _UnknownToolCase | _InvalidArgsCase | _HandlerFailureCase


# ----- Argument strategies -------------------------------------------------

#: GitLab project IDs are 32-bit signed integers in practice; using
#: this range keeps Hypothesis-generated examples readable and well
#: within the ``int`` validator's accepted domain.
_GITLAB_PROJECT_ID = st.integers(min_value=-(2**31), max_value=2**31 - 1)


def _valid_args_for(tool_name: str) -> st.SearchStrategy[dict[str, Any]]:
    """Strategy producing valid ``arguments`` for ``tool_name``.

    Project-id-typed tools require ``{"gitlab_project_id": <int>}``;
    no-args tools accept any dict (the design's no-args input schema
    is ``{"type": "object", "properties": {}}`` which, per JSON Schema
    defaults, accepts arbitrary additional properties). We pick the
    empty dict for no-args tools so the assertion that the dispatcher
    threaded ``arguments`` through to the handler stays precise.
    """

    if tool_name in PROJECT_ID_TOOLS:
        return _GITLAB_PROJECT_ID.map(
            lambda value: {"gitlab_project_id": value}
        )
    return st.just({})


# Non-integer JSON-shaped values used to trigger ``type`` rule
# violations on the ``gitlab_project_id`` field. ``True``/``False``
# are explicitly listed because :mod:`jsonschema` rejects them as
# integers (they are JSON booleans), which lets the test cover the
# "boolean is not an integer" branch as well.
#
# Floats are constrained to have a non-zero fractional part because
# :mod:`jsonschema` (following the JSON Schema spec) accepts
# whole-number floats such as ``0.0`` and ``-3.0`` as ``integer``
# values: a "number with a zero fractional part" is mathematically
# an integer regardless of its Python ``type``. Filtering those out
# keeps every generated value a genuine type-rule violation.
_NON_INTEGER_FLOAT = st.floats(
    allow_nan=False,
    allow_infinity=False,
).filter(lambda f: not f.is_integer())

_NON_INTEGER_VALUES: st.SearchStrategy[Any] = st.one_of(
    st.text(max_size=8),
    _NON_INTEGER_FLOAT,
    st.booleans(),
    st.none(),
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(max_size=4), st.integers(), max_size=3),
)


# ----- Top-level case strategies ------------------------------------------


def _valid_case_strategy() -> st.SearchStrategy[_ValidCase]:
    """Strategy producing a happy-path ``tools/call`` request."""

    return st.sampled_from(BUILT_IN_TOOLS).flatmap(
        lambda tool: _valid_args_for(tool.name).map(
            lambda args: _ValidCase(tool_name=tool.name, arguments=args)
        )
    )


# Unknown-tool-name alphabet: alphanumeric plus underscore mirrors
# the shape of the production tool names. Keeping the alphabet
# narrow makes shrunken counterexamples easy to read; the
# ``filter`` excludes the eight known names so the case is always
# genuinely unknown.
_UNKNOWN_TOOL_NAME = (
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
        min_size=1,
        max_size=24,
    )
    .filter(lambda s: s not in KNOWN_TOOL_NAMES)
)


def _unknown_tool_case_strategy() -> st.SearchStrategy[_UnknownToolCase]:
    """Strategy producing a ``tools/call`` request with an unknown tool name."""

    # Argument shape doesn't matter for unknown-tool dispatch -- the
    # dispatcher rejects on name resolution before it consults the
    # tool's schema. We still vary it across {empty, project-id,
    # arbitrary} so the property exercises the "name resolution
    # happens before argument validation" ordering.
    arbitrary_args: st.SearchStrategy[dict[str, Any]] = st.one_of(
        st.just({}),
        _GITLAB_PROJECT_ID.map(lambda v: {"gitlab_project_id": v}),
        st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.integers(),
            max_size=3,
        ),
    )
    return st.builds(
        _UnknownToolCase,
        tool_name=_UNKNOWN_TOOL_NAME,
        arguments=arbitrary_args,
    )


def _missing_required_case(tool_name: str) -> _InvalidArgsCase:
    """Build an invalid-args case missing the required ``gitlab_project_id``."""

    return _InvalidArgsCase(
        tool_name=tool_name,
        arguments={},
        # Per :mod:`jsonschema` the ``required`` rule reports the
        # offending field name on the parent object; the dispatcher's
        # ``_failing_argument_name`` resolves this to the first
        # required-but-missing key, which is ``gitlab_project_id``
        # for every project-id-typed tool.
        expected_argument="gitlab_project_id",
        expected_rule="required",
    )


def _wrong_type_case(tool_name: str, bad_value: Any) -> _InvalidArgsCase:
    """Build an invalid-args case whose ``gitlab_project_id`` has the wrong type."""

    return _InvalidArgsCase(
        tool_name=tool_name,
        arguments={"gitlab_project_id": bad_value},
        expected_argument="gitlab_project_id",
        expected_rule="type",
    )


def _invalid_args_case_strategy() -> st.SearchStrategy[_InvalidArgsCase]:
    """Strategy producing a ``tools/call`` request that violates the schema.

    Only project-id-typed tools have a non-trivial schema (no-args
    tools accept any dict by JSON-Schema defaults), so the strategy
    samples a project-id tool and then either drops the required
    field or replaces it with a non-integer value.
    """

    project_id_tool = st.sampled_from(PROJECT_ID_TOOLS)
    missing_required = project_id_tool.map(_missing_required_case)
    wrong_type = st.builds(
        _wrong_type_case,
        tool_name=project_id_tool,
        bad_value=_NON_INTEGER_VALUES,
    )
    return st.one_of(missing_required, wrong_type)


def _injected_error_strategy() -> st.SearchStrategy[BaseException]:
    """Strategy producing a runtime / external-dependency failure to inject.

    The two error classes here are the design's production-shaped
    external-dependency failures: ``Knowledge_Store`` unavailable
    (storage layer cannot be queried) and ``GitLab`` auth failure
    (HTTP 401 / 403 from the upstream). Their ``str()`` forms carry
    the canonical wording that Requirement 11.7's tool-result
    message must name verbatim.
    """

    # ``KnowledgeStoreUnavailableError`` accepts an optional reason
    # string; we vary across both shapes so the resulting ``str(exc)``
    # exercises both code paths (with and without the colon-prefixed
    # reason).
    kss_with_reason = st.text(min_size=1, max_size=24).map(
        lambda reason: KnowledgeStoreUnavailableError(reason=reason)
    )
    kss_no_reason = st.just(KnowledgeStoreUnavailableError())
    # ``GitLabAuthError`` carries the HTTP status code; the only two
    # statuses the requirements call out are 401 and 403, but the
    # property holds for any int and we let Hypothesis explore.
    gla = st.integers(min_value=400, max_value=599).map(GitLabAuthError)
    return st.one_of(kss_with_reason, kss_no_reason, gla)


def _handler_failure_case_strategy() -> st.SearchStrategy[_HandlerFailureCase]:
    """Strategy producing a happy-shape request whose handler raises."""

    # Pair a tool name with a valid argument payload (so the
    # dispatcher gets past schema validation) and an injected error
    # the handler will raise.
    return st.sampled_from(BUILT_IN_TOOLS).flatmap(
        lambda tool: st.tuples(
            _valid_args_for(tool.name),
            _injected_error_strategy(),
        ).map(
            lambda pair: _HandlerFailureCase(
                tool_name=tool.name,
                arguments=pair[0],
                error=pair[1],
            )
        )
    )


def _case_strategy() -> st.SearchStrategy[_Case]:
    """Strategy producing one of the four ``tools/call`` request shapes."""

    return st.one_of(
        _valid_case_strategy(),
        _unknown_tool_case_strategy(),
        _invalid_args_case_strategy(),
        _handler_failure_case_strategy(),
    )


# ---------------------------------------------------------------------------
# Dispatch driver
# ---------------------------------------------------------------------------


def _dispatch(
    server: MCPServer, tool_name: str, arguments: dict[str, Any]
) -> mcp_types.ServerResult:
    """Drive ``MCPServer._dispatch_tool_call`` to completion synchronously.

    The dispatcher is async; Hypothesis property tests run in a
    plain sync function so we hand-roll an event loop per example.
    The dispatcher does not retain any cross-loop state (it never
    schedules tasks beyond the awaited handler), so creating a
    fresh loop per example is safe.
    """

    request = mcp_types.CallToolRequest(
        method="tools/call",
        params=mcp_types.CallToolRequestParams(
            name=tool_name,
            arguments=arguments,
        ),
    )
    return asyncio.run(server._dispatch_tool_call(request))


# ---------------------------------------------------------------------------
# Per-case assertion helpers
# ---------------------------------------------------------------------------


def _text_of(result: mcp_types.CallToolResult) -> str:
    """Return the text of the single ``TextContent`` block in ``result``."""

    assert len(result.content) == 1, (
        f"expected a single content block, got {len(result.content)}: {result.content!r}"
    )
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent), (
        f"expected TextContent, got {type(block).__name__}: {block!r}"
    )
    return block.text


def _assert_valid_case(server: MCPServer, case: _ValidCase) -> None:
    """Requirement 11.4: the response is the handler's tool result."""

    _install_handler(server, case.tool_name, _happy_handler())
    server_result = _dispatch(server, case.tool_name, case.arguments)
    inner = server_result.root
    assert isinstance(inner, mcp_types.CallToolResult), (
        f"expected CallToolResult, got {type(inner).__name__}"
    )
    # The handler returns ``isError=False`` and the sentinel text;
    # the dispatcher SHALL forward both verbatim.
    assert inner.isError is False, (
        f"expected isError=False on happy-path result, got {inner.isError!r}"
    )
    assert _text_of(inner) == HAPPY_PATH_TEXT, (
        f"expected text {HAPPY_PATH_TEXT!r}, got {_text_of(inner)!r}"
    )
    # The dispatcher MUST also pass the validated arguments through
    # to the handler unchanged. The handler echoes them via
    # ``structuredContent``; comparing equality (rather than
    # identity) keeps the property robust against any future
    # canonicalization the SDK might introduce.
    assert inner.structuredContent == {"received": case.arguments}, (
        f"handler did not receive the validated arguments verbatim: "
        f"expected {case.arguments!r}, got {inner.structuredContent!r}"
    )


def _assert_unknown_tool_case(server: MCPServer, case: _UnknownToolCase) -> None:
    """Requirement 11.5: an MCP error response indicating the tool is unknown."""

    with pytest.raises(McpError) as excinfo:
        _dispatch(server, case.tool_name, case.arguments)

    err = excinfo.value.error
    assert err.code == mcp_types.METHOD_NOT_FOUND, (
        f"expected METHOD_NOT_FOUND ({mcp_types.METHOD_NOT_FOUND}), "
        f"got {err.code}"
    )
    # The error message must indicate the tool is unknown and name
    # the offending tool. Exact wording is the design's
    # ``"tool '{name}' is unknown"`` template.
    assert case.tool_name in err.message, (
        f"unknown-tool error message must name the offending tool "
        f"{case.tool_name!r}, got {err.message!r}"
    )
    assert "unknown" in err.message.lower(), (
        f"unknown-tool error message must indicate unknownness, got "
        f"{err.message!r}"
    )


def _assert_invalid_args_case(server: MCPServer, case: _InvalidArgsCase) -> None:
    """Requirement 11.6: an MCP error naming the failing argument and rule."""

    # Replace the registered handler with one that fails loudly if
    # invoked. The dispatcher MUST short-circuit on schema
    # validation before reaching the handler -- if it doesn't, the
    # AssertionError surfaces directly in the test failure.
    def _must_not_dispatch(arguments: dict[str, Any]) -> Any:  # pragma: no cover
        raise AssertionError(
            "dispatcher invoked the handler despite invalid arguments: "
            f"tool={case.tool_name!r} args={arguments!r}"
        )

    async def _failing_validator_handler(
        arguments: dict[str, Any],
    ) -> mcp_types.CallToolResult:  # pragma: no cover - guarded by assertion
        _must_not_dispatch(arguments)
        # Unreachable: the assertion above raises.
        return mcp_types.CallToolResult(content=[])

    _install_handler(server, case.tool_name, _failing_validator_handler)

    with pytest.raises(McpError) as excinfo:
        _dispatch(server, case.tool_name, case.arguments)

    err = excinfo.value.error
    assert err.code == mcp_types.INVALID_PARAMS, (
        f"expected INVALID_PARAMS ({mcp_types.INVALID_PARAMS}), got {err.code}"
    )
    # The structured ``data`` payload SHALL carry both the failing
    # argument's name and the rule that rejected it (Requirement
    # 11.6, "names the failing argument and the validation rule that
    # failed").
    assert isinstance(err.data, dict), (
        f"InvalidParams error must carry a dict ``data`` payload, got "
        f"{type(err.data).__name__}: {err.data!r}"
    )
    data: dict[str, Any] = err.data
    assert data.get("argument") == case.expected_argument, (
        f"expected data.argument == {case.expected_argument!r}, got "
        f"{data.get('argument')!r}"
    )
    assert data.get("rule") == case.expected_rule, (
        f"expected data.rule == {case.expected_rule!r}, got "
        f"{data.get('rule')!r}"
    )
    # The message must mention the requesting tool, the failing
    # argument, and the failing rule so a text-mode client can
    # surface the failure without parsing the structured ``data``.
    assert case.tool_name in err.message, (
        f"InvalidParams message must name the tool {case.tool_name!r}, "
        f"got {err.message!r}"
    )
    assert case.expected_argument in err.message, (
        f"InvalidParams message must name the failing argument "
        f"{case.expected_argument!r}, got {err.message!r}"
    )
    assert case.expected_rule in err.message, (
        f"InvalidParams message must name the failing rule "
        f"{case.expected_rule!r}, got {err.message!r}"
    )


def _assert_handler_failure_case(
    server: MCPServer, case: _HandlerFailureCase
) -> None:
    """Requirement 11.7: tool result with isError=true naming the failure."""

    _install_handler(server, case.tool_name, _failing_handler(case.error))
    server_result = _dispatch(server, case.tool_name, case.arguments)
    inner = server_result.root
    assert isinstance(inner, mcp_types.CallToolResult), (
        f"expected CallToolResult, got {type(inner).__name__}"
    )
    assert inner.isError is True, (
        "handler-runtime failure must surface as a tool result with "
        f"isError=true, got isError={inner.isError!r}"
    )
    text = _text_of(inner)
    # Canonical "tool execution failed: {reason}" wrapping per the
    # dispatcher's Requirement 11.7 implementation.
    assert text.startswith("tool execution failed:"), (
        f"handler-failure tool result must start with the canonical "
        f"prefix, got {text!r}"
    )
    # Requirement 11.7 ("names the failure reason"): ``str(exc)`` --
    # the canonical wording each error class produces -- must appear
    # verbatim in the surfaced message text.
    failure_reason = str(case.error)
    assert failure_reason in text, (
        f"handler-failure tool result must name the failure reason "
        f"{failure_reason!r}, got {text!r}"
    )


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(case=_case_strategy())
@settings(max_examples=100)
def test_tools_call_dispatch_satisfies_property_19(case: _Case) -> None:
    """Property 19: the four-way ``tools/call`` invariant holds.

    Builds a fresh :class:`MCPServer` per example, installs a
    synthetic handler that matches the case kind (happy-path,
    failing, or never-invoked), drives the dispatcher with the
    generated request, and asserts the case-specific invariant.

    The four branches are mutually exclusive (a single ``Case``
    instance lands in exactly one branch by construction), and
    together they cover the four clauses of Property 19's
    SHALL-statement. Hypothesis explores 100 examples per run with
    the case strategy stratified across the four kinds, so each
    requirement (11.4, 11.5, 11.6, 11.7) is exercised on
    multiple examples per run on average.
    """

    server = _build_server()

    if isinstance(case, _ValidCase):
        _assert_valid_case(server, case)
    elif isinstance(case, _UnknownToolCase):
        _assert_unknown_tool_case(server, case)
    elif isinstance(case, _InvalidArgsCase):
        _assert_invalid_args_case(server, case)
    elif isinstance(case, _HandlerFailureCase):
        _assert_handler_failure_case(server, case)
    else:  # pragma: no cover - exhaustive over the union above
        raise AssertionError(f"unexpected case type: {type(case).__name__}")

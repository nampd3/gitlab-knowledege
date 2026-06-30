# ruff: noqa: E501
# Feature: go-analyzer-support, Property 5: HTTP detection is exactly the non-bootstrap registration set, with method/path correctly split.
"""Property test for the Go HTTP route-registration recognizer.

**Validates Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7** (Property 5
in the design, task 7.2 in ``tasks.md``).

The HTTP route recognizer is the first of five recognizers composed by
``extract_go_io`` (task 7.11). This property test exercises its
package-internal entry point :func:`_extract_http_routes` directly so
the per-recognizer contract is pinned independently of the eventual
composition step.

The properties below capture the design's documented contract:

1. **One AbstractInput and one AbstractOutput per recognized
   registration** (Requirements 3.1, 3.2 + Property 5 statement).
   Every ``HandleFunc`` / ``Handle`` call whose receiver chain is
   either exactly ``("http",)`` or a single identifier in a file that
   imports ``net/http`` produces exactly one
   ``AbstractInput(category=http_request)`` and one
   ``AbstractOutput(category=http_response)``.

2. **Method/path split honors Go 1.22 grammar** (Requirement 3.3).
   String-literal patterns of the form ``"<METHOD> <path>"`` (with a
   single space separator, METHOD drawn from
   ``GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS``) are split into method
   and path. Bare paths and any other literal shape are recorded with
   method ``"ANY"`` and the whole literal as the path.

3. **Non-literal patterns use the placeholder description**
   (Requirement 3.4). When the first positional argument is not a
   ``StringLitArg`` — ``DottedArg`` for ``cfg.Routes.Healthz``,
   ``IdentArg`` for ``healthzPath``, ``UnknownArg`` for opaque
   expressions, ``CallArg`` for nested calls, ``NumberLitArg`` for
   numeric literals, or a missing argument — the recognizer emits one
   input/output pair carrying a ``<dynamic at <file>:<line> on <recv>>``
   placeholder rather than the literal pattern.

4. **Bootstrap calls produce no emissions** (Requirement 3.5). The
   five bootstrap method names ``ListenAndServe``, ``ListenAndServeTLS``,
   ``Serve``, ``Shutdown``, ``Close`` are silently dropped wherever
   they appear — on ``http.<method>(...)`` and on any
   ``<id>.<method>(...)`` receiver — with no log, no error, and no
   entry. The recognizer's ``with`` and ``without`` outputs are
   identical when bootstrap events are stripped from the input.

5. **Source_Location is attached to every emission** (Requirement 3.6).
   The recognizer encodes the ``MethodCallEvent``'s ``file_path`` and
   1-indexed ``line`` into the description as a ``at <file>:<line>``
   suffix, matching the scheduler and ActiveMQ recognizers'
   convention.

6. **Coalescing follows the existing dedup contract** (Requirement 3.7).
   Entries are deduplicated by ``(category, description)``; since the
   description carries the call site's location, two
   ``MethodCallEvent``s with the same ``(file_path, line)`` coalesce
   into one emitted pair.

7. **fx and viper exclusions are inert at the dispatch boundary**
   (Requirements 12.1, 12.3, 13.1, 13.2). Adding arbitrary
   ``fx.<method>`` and ``viper.<method>`` calls — including those
   that re-use the recognizer's method-name vocabulary
   (``HandleFunc``, ``Handle``) — must not change the recognizer's
   output.

8. **Deterministic, path-sorted iteration** (Requirement 11.4). The
   recognizer is a pure function of its input mapping. Reversing the
   mapping's insertion order must not change the output lists.

The recognizer's :func:`_extract_http_routes` is package-internal; the
test imports it through the module path to mirror the convention used
by ``test_property_06_go_scheduler_detection.py`` for
:func:`_extract_schedulers` and
``test_property_07_go_activemq_detection.py`` for
:func:`_extract_activemq_io`.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
)
from project_knowledge_mcp.project_analyzer.go._events import (
    CallArg,
    DottedArg,
    IdentArg,
    ImportEvent,
    MethodCallEvent,
    NumberLitArg,
    StringLitArg,
    UnknownArg,
)
from project_knowledge_mcp.project_analyzer.go.go_io import _extract_http_routes

if TYPE_CHECKING:
    from project_knowledge_mcp.project_analyzer.go._events import ArgRef, GoEvent


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets and shared constants
# ---------------------------------------------------------------------------


#: Identifier alphabet for receiver names, file basenames, and dotted-arg
#: segments. Constrained to characters Go accepts in identifiers so the
#: generated fixtures look like real source.
_IDENT_CHARS: Final[str] = string.ascii_letters + string.digits + "_"


#: Path alphabet for HTTP route patterns. ``/`` and ``-`` are included
#: because they are the canonical separators in ``/api/v1/health`` and
#: ``/soap/repayment-service`` shapes observed in the four sample
#: repositories. Curly braces are included so the Go 1.22 wildcard
#: syntax (``{id}``) is exercised. Whitespace is excluded so the path
#: cannot accidentally synthesize a ``"<METHOD> <path>"`` literal that
#: the recognizer would split differently from what the generator
#: predicted.
_PATH_CHARS: Final[str] = string.ascii_letters + string.digits + "/_-{}"


#: HTTP verbs recognized by the Go 1.22 method-prefixed pattern grammar
#: (Requirement 3.3). The exact set is mirrored from the recognizer's
#: ``_HTTP_METHOD_PREFIX_RE`` so the test's predictions stay in lockstep
#: with the implementation.
_HTTP_METHODS: Final[tuple[str, ...]] = (
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "HEAD",
    "OPTIONS",
)


#: The five HTTP server bootstrap method names the recognizer silently
#: suppresses (Requirement 3.5). Spelt out here so the test exercises
#: every one of them rather than sampling.
_BOOTSTRAP_METHODS: Final[tuple[str, ...]] = (
    "ListenAndServe",
    "ListenAndServeTLS",
    "Serve",
    "Shutdown",
    "Close",
)


#: The two route-registration method names recognized on the
#: ``net/http`` family (Requirement 3.2).
_ROUTE_METHODS: Final[tuple[str, ...]] = ("HandleFunc", "Handle")


#: Categories the recognizer always emits.
_HTTP_REQUEST: Final[AbstractInputCategory] = AbstractInputCategory.HTTP_REQUEST
_HTTP_RESPONSE: Final[AbstractOutputCategory] = AbstractOutputCategory.HTTP_RESPONSE


#: Canonical description fragments — anchors the assertions verify on
#: every emission so a regression that drops the protocol prefix or
#: location suffix is caught even when the full-list equality assertion
#: also fails.
_HTTP_REQUEST_PREFIX: Final[str] = "HTTP "
_HTTP_RESPONSE_REQUEST_PLACEHOLDER: Final[str] = "HTTP request "
_HTTP_RESPONSE_PLACEHOLDER: Final[str] = "HTTP response "
_LOCATION_FRAGMENT: Final[str] = " at "
_DYNAMIC_TOKEN: Final[str] = "<dynamic at "


#: Import path the recognizer consults to enable the "any single
#: identifier may be a mux" relaxation (Requirement 3.2 carve-out).
_NET_HTTP_IMPORT_PATH: Final[str] = "net/http"


#: A small set of fx / viper method names sufficient to demonstrate
#: that the recognizer's dispatch-boundary exclusion holds. Method
#: names overlap with the recognized route vocabulary so a regression
#: that forgets to honor the receiver-chain exclusion would
#: immediately surface: the noise would otherwise satisfy the
#: route-method gate.
_FX_METHODS: Final[tuple[str, ...]] = (
    "Provide",
    "Invoke",
    "Module",
    "New",
    "HandleFunc",
    "Handle",
)
_VIPER_METHODS: Final[tuple[str, ...]] = (
    "New",
    "GetString",
    "BindEnv",
    "HandleFunc",
    "Handle",
)


# ---------------------------------------------------------------------------
# Helpers for receiver identifiers
# ---------------------------------------------------------------------------


# A mux-style receiver is any single identifier whose value is *not*
# the literal ``"http"``, ``"fx"``, or ``"viper"``. Excluding ``"http"``
# keeps the mux-style branch (which depends on the file importing
# ``net/http``) separate from the package-level branch (which is
# always recognized). Excluding ``"fx"`` and ``"viper"`` keeps the
# mux-style branch separate from the dispatch-boundary exclusion
# branch: a single-identifier chain whose head is ``fx`` or ``viper``
# is rejected by :func:`_is_excluded_receiver` before the recognizer
# inspects the method name, so allowing the generator to produce
# these identifiers would conflate the route-receiver gate with the
# fx / viper exclusion. The alphabet is the same as the general
# identifier alphabet so the generator can still produce realistic
# receiver names like ``mux``, ``r``, ``server``.
_RESERVED_RECEIVER_IDENTS: Final[frozenset[str]] = frozenset(
    {"http", "fx", "viper"},
)
_mux_ident = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8).filter(
    lambda s: s not in _RESERVED_RECEIVER_IDENTS,
)


_path_text = st.text(alphabet=_PATH_CHARS, min_size=1, max_size=20)


_dotted_segments = st.lists(
    st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=6),
    min_size=2,
    max_size=4,
).map(tuple)


# ---------------------------------------------------------------------------
# Pattern generation: bare paths, method-prefixed literals, non-literals
# ---------------------------------------------------------------------------


@st.composite
def _bare_path_literal(draw: st.DrawFn) -> str:
    """Generate a bare-path string literal that never matches the method prefix.

    The recognizer's method-prefix regex requires the literal to begin
    with a verb from :data:`_HTTP_METHODS` followed by a single space.
    The alphabet :data:`_PATH_CHARS` already excludes spaces, so any
    draw from it cannot accidentally synthesize a method-prefixed
    pattern. Prefixing every draw with ``"/"`` matches the canonical
    bare-path shape observed in the sample repositories (``/healthz``,
    ``/debug/pprof/``, ``/``).
    """

    body = draw(_path_text)
    return f"/{body}"


@st.composite
def _method_prefixed_literal(draw: st.DrawFn) -> tuple[str, str, str]:
    """Generate a Go 1.22 method-prefixed pattern plus its expected split.

    Returns ``(literal, expected_method, expected_path)``. The literal
    is constructed by interpolation rather than drawn from a free
    alphabet so the generator's prediction is guaranteed to match the
    recognizer's regex behavior. The path is at least one character
    long because the recognizer's regex requires ``.+`` after the
    separator space.
    """

    method = draw(st.sampled_from(_HTTP_METHODS))
    # The path body cannot be empty (the regex requires `.+`) and
    # must not start with a space; ``_path_text`` already guarantees
    # the latter and ``min_size=1`` guarantees the former.
    path = "/" + draw(_path_text)
    literal = f"{method} {path}"
    return literal, method, path


@st.composite
def _string_literal_pattern(
    draw: st.DrawFn,
) -> tuple[StringLitArg, str, str]:
    """Generate a string-literal pattern argument and its expected ``(method, path)``.

    Stratifies over the two literal shapes the recognizer's
    ``_parse_route_pattern`` distinguishes:

    * Bare path → ``method="ANY"`` and the literal is the entire path.
    * Method-prefixed → split into the chosen verb and the trailing
      path.

    Returns ``(arg, expected_method, expected_path)``.
    """

    if draw(st.booleans()):
        literal, method, path = draw(_method_prefixed_literal())
        return StringLitArg(value=literal), method, path
    literal = draw(_bare_path_literal())
    return StringLitArg(value=literal), "ANY", literal


@st.composite
def _non_literal_pattern(draw: st.DrawFn) -> "ArgRef | None":
    """Generate a first-argument shape that is *not* a ``StringLitArg``.

    Stratifies over every ``ArgRef`` variant other than
    :class:`StringLitArg` plus the "no argument" branch. Every variant
    feeds the recognizer's placeholder-description path (Requirement
    3.4): the recognizer never inspects the variant's contents beyond
    "is this a string literal?".
    """

    kind = draw(
        st.sampled_from(
            ("dotted", "ident", "unknown", "number", "call", "missing"),
        ),
    )
    if kind == "dotted":
        return DottedArg(parts=draw(_dotted_segments))
    if kind == "ident":
        return IdentArg(name=draw(st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8)))
    if kind == "unknown":
        return UnknownArg(text=draw(st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=10)))
    if kind == "number":
        return NumberLitArg(text=draw(st.text(alphabet=string.digits, min_size=1, max_size=4)))
    if kind == "call":
        # The recognizer never recurses into nested calls for the
        # route-pattern argument; the call's shape is uninteresting.
        nested = MethodCallEvent(
            receiver_chain=("cfg",),
            method_name="Path",
            args=(),
            file_path="dummy.go",
            line=1,
        )
        return CallArg(call=nested)
    # "missing": the call has no positional arguments at all.
    return None


# ---------------------------------------------------------------------------
# Receiver-shape kinds
# ---------------------------------------------------------------------------


# The receiver-shape kinds the recognizer's
# :func:`_resolve_route_receiver` distinguishes plus the rejection
# shapes that exercise its fall-through branches.
_PKG_HTTP_SHAPE: Final[str] = "pkg_http"        # ("http",) — always recognized
_MUX_WITH_IMPORT_SHAPE: Final[str] = "mux_imp"  # single ident, file imports net/http
_MUX_NO_IMPORT_SHAPE: Final[str] = "mux_noimp"  # single ident, file lacks net/http
_MULTI_SEGMENT_SHAPE: Final[str] = "multi_seg"  # ("a", "b", ...) — rejected
_EMPTY_CHAIN_SHAPE: Final[str] = "empty"        # () — rejected


_RECEIVER_SHAPES: Final[tuple[str, ...]] = (
    _PKG_HTTP_SHAPE,
    _MUX_WITH_IMPORT_SHAPE,
    _MUX_NO_IMPORT_SHAPE,
    _MULTI_SEGMENT_SHAPE,
    _EMPTY_CHAIN_SHAPE,
)


# ---------------------------------------------------------------------------
# Per-call generators
# ---------------------------------------------------------------------------


def _expected_recognized(
    *,
    chain: tuple[str, ...],
    file_imports_net_http: bool,
) -> str | None:
    """Mirror the recognizer's :func:`_resolve_route_receiver` decision.

    Returns the receiver label used in the description suffix when the
    chain satisfies the route-receiver gate, or ``None`` when it does
    not. The test uses this to predict the recognizer's emission.
    """

    if chain == ("http",):
        return "http"
    if len(chain) == 1 and file_imports_net_http:
        return chain[0]
    return None


@st.composite
def _route_call(
    draw: st.DrawFn,
    *,
    file_path: str,
    file_imports_net_http: bool,
) -> tuple[
    MethodCallEvent,
    str | None,
    tuple[str, str] | None,
    bool,
]:
    """Generate one ``HandleFunc`` / ``Handle`` call event and its expectation.

    Returns ``(event, receiver_label, method_path_split, is_literal)``:

    * ``receiver_label`` is the label the recognizer must use in the
      description, or ``None`` when the call shape rejects the
      receiver gate and produces no emission.
    * ``method_path_split`` is ``(http_method, http_path)`` when the
      first argument is a string literal that the recognizer's
      pattern grammar accepts; ``None`` for any non-literal shape.
    * ``is_literal`` is ``True`` when ``method_path_split`` is not
      ``None``; surfaced as a separate field so the placeholder
      branch's expectation is easy to compute downstream.
    """

    shape = draw(st.sampled_from(_RECEIVER_SHAPES))
    method = draw(st.sampled_from(_ROUTE_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))

    if shape == _PKG_HTTP_SHAPE:
        chain: tuple[str, ...] = ("http",)
    elif shape in (_MUX_WITH_IMPORT_SHAPE, _MUX_NO_IMPORT_SHAPE):
        # Mux-style: any single identifier that is not "http". The
        # net/http import requirement is enforced at the file level,
        # so this generator only encodes the receiver chain itself.
        chain = (draw(_mux_ident),)
    elif shape == _MULTI_SEGMENT_SHAPE:
        # The recognizer's receiver-shape gate rejects any chain
        # longer than 1 segment except the exact ``("http",)`` form,
        # which is already covered by the package-level branch. Two-
        # segment chains like ``("r", "Group")`` are the natural
        # source of these rejections (chi-style router).
        chain = (
            draw(_mux_ident),
            draw(_mux_ident),
        )
    else:  # _EMPTY_CHAIN_SHAPE
        chain = ()

    # Argument-shape stratification: literal vs non-literal vs missing.
    arg_kind = draw(st.sampled_from(("literal", "non_literal")))
    args: tuple[ArgRef, ...]
    method_path_split: tuple[str, str] | None
    is_literal: bool

    if arg_kind == "literal":
        lit_arg, http_method, http_path = draw(_string_literal_pattern())
        args = (lit_arg,)
        method_path_split = (http_method, http_path)
        is_literal = True
    else:
        non_lit = draw(_non_literal_pattern())
        args = () if non_lit is None else (non_lit,)
        method_path_split = None
        is_literal = False

    event = MethodCallEvent(
        receiver_chain=chain,
        method_name=method,
        args=args,
        file_path=file_path,
        line=line,
    )

    receiver_label = _expected_recognized(
        chain=chain,
        file_imports_net_http=file_imports_net_http,
    )
    return event, receiver_label, method_path_split, is_literal


@st.composite
def _bootstrap_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a bootstrap-method call event (Requirement 3.5).

    The recognizer must drop these silently regardless of receiver
    shape and regardless of whether the file imports ``net/http``. The
    generator covers both the package-level (``http.<method>``) and
    the single-identifier (``<srv>.<method>``) receiver shapes, which
    together encode the design's ``http`` package and ``*http.Server``
    cases.
    """

    method = draw(st.sampled_from(_BOOTSTRAP_METHODS))
    # Mix the two canonical receiver shapes. The recognizer rejects
    # both because bootstrap method names are excluded from the
    # route-method gate; the rejection path is identical for both.
    if draw(st.booleans()):
        chain: tuple[str, ...] = ("http",)
    else:
        chain = (draw(_mux_ident),)
    line = draw(st.integers(min_value=1, max_value=999))

    # A bootstrap call's argument shape never affects the recognizer's
    # decision. A single string-literal address argument is the most
    # adversarial choice — if the recognizer ever forgot the method-name
    # gate and fell through to ``_format_descriptions`` with this
    # argument, the description would emit a route literal we did not
    # predict, immediately breaking the equality assertion.
    return MethodCallEvent(
        receiver_chain=chain,
        method_name=method,
        args=(StringLitArg(value=":8080"),),
        file_path=file_path,
        line=line,
    )


@st.composite
def _unrelated_method_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a method call whose name is not in the recognized set.

    Covers the broad "method name does not match" rejection branch.
    Method names overlap with other Go recognizers' vocabularies
    (``AddFunc``, ``Subscribe``, ``SendMessage``, ``NewTicker``) so a
    regression that accidentally widened the route-method set would
    immediately fire here.
    """

    chain_choice = draw(st.sampled_from(("http", "mux", "multi", "empty")))
    if chain_choice == "http":
        chain: tuple[str, ...] = ("http",)
    elif chain_choice == "mux":
        chain = (draw(_mux_ident),)
    elif chain_choice == "multi":
        chain = (draw(_mux_ident), draw(_mux_ident))
    else:
        chain = ()
    method = draw(
        st.sampled_from(
            ("AddFunc", "AddJob", "Subscribe", "SendMessage", "NewTicker", "Get", "Post"),
        ),
    )
    line = draw(st.integers(min_value=1, max_value=999))
    return MethodCallEvent(
        receiver_chain=chain,
        method_name=method,
        # A string-literal first argument that looks route-shaped, so a
        # regression that forgot the method-name gate would emit an
        # entry the predictor does not expect.
        args=(StringLitArg(value="/regression-canary"),),
        file_path=file_path,
        line=line,
    )


@st.composite
def _fx_or_viper_noise_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate an fx / viper noise event excluded at the dispatch boundary.

    The recognizer must treat these as inert (Requirements 12.1, 12.3,
    13.1, 13.2). The noise's method-name selection deliberately
    overlaps with the recognized route vocabulary (``HandleFunc``,
    ``Handle``) and the argument is a fully valid string-literal route
    pattern, so a regression that forgets the receiver-chain exclusion
    would emit an entry the expected-output computation does not
    predict — instantly breaking the full-list equality assertion.
    """

    receiver = draw(st.sampled_from(("fx", "viper")))
    if receiver == "fx":
        method = draw(st.sampled_from(_FX_METHODS))
    else:
        method = draw(st.sampled_from(_VIPER_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))
    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name=method,
        args=(StringLitArg(value="/regression-canary"),),
        file_path=file_path,
        line=line,
    )


# ---------------------------------------------------------------------------
# Per-file fixture
# ---------------------------------------------------------------------------


@st.composite
def _file_events(
    draw: st.DrawFn,
    *,
    path: str,
) -> tuple[
    list["GoEvent"],
    bool,
    list[tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]],
]:
    """Generate one file's event list plus the per-recognizer expectations.

    Returns ``(events, file_imports_net_http, route_calls)``:

    * ``events`` is the order-preserving event list assigned to the
      file's key in the ``events_by_file`` mapping. Always carries
      noise — fx / viper events, bootstrap calls, unrelated-method
      calls — so the exclusion and rejection branches are exercised
      on every example.
    * ``file_imports_net_http`` records whether the file's event list
      contains an :class:`ImportEvent` for ``net/http``. The
      expected-output computation needs this to predict the
      mux-style receiver branch.
    * ``route_calls`` is the per-call metadata for ``HandleFunc`` /
      ``Handle`` events: ``(event, receiver_label_or_None,
      method_path_split_or_None, is_literal)`` per generated call.
      Each entry's expected emission is derived from this tuple in
      :func:`_expected_inputs_and_outputs`.
    """

    file_imports_net_http = draw(st.booleans())
    events: list[GoEvent] = []

    if file_imports_net_http:
        events.append(
            ImportEvent(
                path=_NET_HTTP_IMPORT_PATH,
                alias=None,
                file_path=path,
                line=1,
            ),
        )

    # Always at least one fx / viper noise event so the dispatch-
    # boundary exclusion property is exercised on every example.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    # Bootstrap noise — must emit nothing (Requirement 3.5).
    bootstrap = draw(
        st.lists(_bootstrap_call(file_path=path), min_size=0, max_size=3),
    )
    events.extend(bootstrap)

    # Unrelated-method noise — must emit nothing.
    unrelated = draw(
        st.lists(_unrelated_method_event(file_path=path), min_size=0, max_size=2),
    )
    events.extend(unrelated)

    # Recognized + rejected route calls. Each call's expected metadata
    # is captured so the assertion phase can compute the predicted
    # emission per call shape.
    route_calls = draw(
        st.lists(
            _route_call(
                file_path=path,
                file_imports_net_http=file_imports_net_http,
            ),
            min_size=0,
            max_size=4,
        ),
    )
    for event, _, _, _ in route_calls:
        events.append(event)

    # Trailing noise to guard against position-sensitivity regressions.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    return events, file_imports_net_http, route_calls


@st.composite
def _events_by_file(
    draw: st.DrawFn,
) -> tuple[
    dict[str, list["GoEvent"]],
    dict[
        str,
        tuple[
            bool,
            list[tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]],
        ],
    ],
]:
    """Generate a multi-file events mapping plus per-file expectation metadata.

    Returns ``(events_by_file, per_file_metadata)``. The
    ``per_file_metadata`` dict mirrors ``events_by_file`` keys and
    carries the two-tuple ``(file_imports_net_http, route_calls)``
    needed to compute the expected recognizer output.
    """

    names = draw(
        st.lists(
            st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=6),
            min_size=1,
            max_size=4,
            unique=True,
        ),
    )

    events_by_file: dict[str, list[GoEvent]] = {}
    metadata: dict[
        str,
        tuple[
            bool,
            list[tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]],
        ],
    ] = {}

    for name in names:
        path = f"{name}.go"
        events, imports, route_calls = draw(_file_events(path=path))
        events_by_file[path] = events
        metadata[path] = (imports, route_calls)

    return events_by_file, metadata


# ---------------------------------------------------------------------------
# Expected-output computation
# ---------------------------------------------------------------------------


def _expected_recognized_descriptions(
    *,
    event: MethodCallEvent,
    receiver_label: str,
    method_path_split: tuple[str, str] | None,
) -> tuple[str, str]:
    """Compute the ``(input_description, output_description)`` for a recognized call.

    Mirrors :func:`go_io._format_descriptions` exactly: the recognized-
    literal branch produces an HTTP-method-and-path prefix; the
    placeholder branch produces the ``<dynamic at ...>`` form. Both
    end with the ``via <recv>.<method>() at <file>:<line>`` suffix
    that Requirement 3.6 demands.
    """

    suffix = (
        f"via {receiver_label}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )
    if method_path_split is not None:
        http_method, http_path = method_path_split
        input_desc = f"HTTP {http_method} {http_path} {suffix}"
        output_desc = f"HTTP {http_method} response for {http_path} {suffix}"
    else:
        placeholder = f"<dynamic at {event.file_path}:{event.line} on {receiver_label}>"
        input_desc = f"HTTP request {placeholder}"
        output_desc = f"HTTP response {placeholder}"
    return input_desc, output_desc


def _expected_inputs_and_outputs(
    metadata: dict[
        str,
        tuple[
            bool,
            list[tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]],
        ],
    ],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Compute the deduplicated expected ``AbstractInput`` / ``AbstractOutput`` lists.

    Iteration order matches the recognizer's: paths in sorted order
    and, within each file, the order in which the route calls were
    appended to the events list. The dedup contract is
    ``(category, description)`` — identical to the recognizer's
    ``seen_inputs`` / ``seen_outputs`` sets.
    """

    inputs: list[AbstractInput] = []
    outputs: list[AbstractOutput] = []
    seen_in: set[tuple[AbstractInputCategory, str]] = set()
    seen_out: set[tuple[AbstractOutputCategory, str]] = set()

    for path in sorted(metadata):
        _imports, route_calls = metadata[path]
        for event, receiver_label, method_path_split, _is_literal in route_calls:
            if receiver_label is None:
                # Receiver-shape gate rejected the call: no emission.
                continue
            input_desc, output_desc = _expected_recognized_descriptions(
                event=event,
                receiver_label=receiver_label,
                method_path_split=method_path_split,
            )
            in_key = (_HTTP_REQUEST, input_desc)
            if in_key not in seen_in:
                seen_in.add(in_key)
                inputs.append(
                    AbstractInput(category=_HTTP_REQUEST, description=input_desc),
                )
            out_key = (_HTTP_RESPONSE, output_desc)
            if out_key not in seen_out:
                seen_out.add(out_key)
                outputs.append(
                    AbstractOutput(category=_HTTP_RESPONSE, description=output_desc),
                )

    return inputs, outputs


def _stripped_of_fx_viper(
    events_by_file: dict[str, list["GoEvent"]],
) -> dict[str, list["GoEvent"]]:
    """Return ``events_by_file`` with every fx / viper ``MethodCallEvent`` removed.

    The Requirement 12 / 13 exclusion property compares the
    recognizer's output before and after the removal; equality of the
    two outputs implies the recognizer never inspected the excluded
    receivers.
    """

    stripped: dict[str, list[GoEvent]] = {}
    for path, events in events_by_file.items():
        kept: list[GoEvent] = []
        for event in events:
            if (
                isinstance(event, MethodCallEvent)
                and event.receiver_chain
                and event.receiver_chain[0] in ("fx", "viper")
            ):
                continue
            kept.append(event)
        stripped[path] = kept
    return stripped


def _stripped_of_bootstrap(
    events_by_file: dict[str, list["GoEvent"]],
) -> dict[str, list["GoEvent"]]:
    """Return ``events_by_file`` with every bootstrap ``MethodCallEvent`` removed.

    Requirement 3.5 states that bootstrap calls produce no emissions;
    comparing the recognizer's output before and after the removal
    pins that contract directly.
    """

    bootstrap = set(_BOOTSTRAP_METHODS)
    stripped: dict[str, list[GoEvent]] = {}
    for path, events in events_by_file.items():
        kept: list[GoEvent] = []
        for event in events:
            if (
                isinstance(event, MethodCallEvent)
                and event.method_name in bootstrap
            ):
                continue
            kept.append(event)
        stripped[path] = kept
    return stripped


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_http_routes_matches_expected_output(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool,
                list[
                    tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]
                ],
            ],
        ],
    ],
) -> None:
    """Property 5 main invariant: one input/output per recognized registration.

    Validates Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7 jointly.
    The expected output is computed by mirroring the recognizer's
    decision tree on the per-file generator metadata, then asserting
    full-list equality. Per-emission invariants validated by
    construction:

    * Exactly one ``AbstractInput(category=http_request)`` and one
      ``AbstractOutput(category=http_response)`` per recognized
      registration (Requirement 3.1).
    * Method/path split honors the Go 1.22 grammar — bare paths get
      method ``"ANY"``; method-prefixed literals split on the
      single-space separator (Requirement 3.3).
    * Non-literal first arguments emit the placeholder description
      naming the file and receiver (Requirement 3.4).
    * Bootstrap calls produce no emission and unrelated method names
      produce no emission (Requirement 3.5 + recognizer's
      method-name gate).
    * The Source_Location suffix ``at <file>:<line>`` appears on
      every emission (Requirement 3.6).
    * Description-level dedup coalesces repeated registrations at
      identical call sites (Requirement 3.7).
    """

    events_by_file, metadata = case

    actual_inputs, actual_outputs = _extract_http_routes(events_by_file)
    expected_inputs, expected_outputs = _expected_inputs_and_outputs(metadata)

    assert actual_inputs == expected_inputs, (
        "HTTP recognizer inputs diverged from expected:\n"
        f"  actual:   {actual_inputs!r}\n"
        f"  expected: {expected_inputs!r}"
    )
    assert actual_outputs == expected_outputs, (
        "HTTP recognizer outputs diverged from expected:\n"
        f"  actual:   {actual_outputs!r}\n"
        f"  expected: {expected_outputs!r}"
    )

    # Universal invariants on the actual output, independent of the
    # expected-output computation. These would catch a regression that
    # made the recognizer's output structurally wrong even when the
    # full-list equality assertion also passed (e.g. by a coincidental
    # two-bug cancellation in the expected-output mirror).
    for entry in actual_inputs:
        assert entry.category is _HTTP_REQUEST, (
            f"input entry {entry!r} carries category {entry.category!r}; "
            f"the HTTP recognizer must always emit http_request "
            f"(Requirement 3.1)"
        )
        assert entry.description.startswith(_HTTP_REQUEST_PREFIX), (
            f"input entry {entry!r} does not begin with the canonical "
            f"'HTTP ' protocol prefix; Requirement 3.1 mandates a "
            f"protocol-tagged description"
        )
        assert _LOCATION_FRAGMENT in entry.description, (
            f"input entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 3.6 mandates a Source_Location on "
            f"every emission"
        )
    for entry in actual_outputs:
        assert entry.category is _HTTP_RESPONSE, (
            f"output entry {entry!r} carries category {entry.category!r}; "
            f"the HTTP recognizer must always emit http_response "
            f"(Requirement 3.1)"
        )
        assert entry.description.startswith(_HTTP_REQUEST_PREFIX), (
            f"output entry {entry!r} does not begin with the canonical "
            f"'HTTP ' protocol prefix; the description must be"
            f" protocol-tagged so consumers can pair it with the input"
        )
        assert _LOCATION_FRAGMENT in entry.description, (
            f"output entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 3.6 mandates a Source_Location on "
            f"every emission"
        )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_http_routes_preserves_literal_path_verbatim(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool,
                list[
                    tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]
                ],
            ],
        ],
    ],
) -> None:
    """Verbatim-preservation invariant for string-literal route patterns.

    Validates Requirement 3.3. For every recognized call whose first
    argument is a ``StringLitArg`` and whose receiver passes the
    route-receiver gate, the recognizer must place the parsed
    ``(method, path)`` split into the description verbatim — bare
    paths keep their entire literal as the path with method
    ``"ANY"``; method-prefixed literals split on the single-space
    separator into the chosen verb and the trailing path.
    """

    events_by_file, metadata = case
    actual_inputs, _ = _extract_http_routes(events_by_file)
    descriptions = [entry.description for entry in actual_inputs]

    for path in sorted(metadata):
        _imports, route_calls = metadata[path]
        for event, receiver_label, method_path_split, _is_literal in route_calls:
            if receiver_label is None or method_path_split is None:
                continue
            http_method, http_path = method_path_split
            location_suffix = f"at {event.file_path}:{event.line}"
            # The recognizer's method/path split must appear verbatim
            # in at least one input description carrying the call's
            # Source_Location suffix.
            matched = any(
                f"HTTP {http_method} {http_path}" in desc
                and location_suffix in desc
                for desc in descriptions
            )
            assert matched, (
                f"string-literal route ({http_method}, {http_path}) at "
                f"{event.file_path}:{event.line} not preserved verbatim "
                f"in any output description; descriptions: "
                f"{descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_http_routes_emits_placeholder_for_non_literal(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool,
                list[
                    tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]
                ],
            ],
        ],
    ],
) -> None:
    """Placeholder-description invariant for non-literal route patterns.

    Validates Requirement 3.4. When the first argument is not a
    string literal, the recognizer must record a
    ``<dynamic at <file>:<line> on <recv>>`` placeholder rather than
    a literal path. The test asserts the placeholder substring is
    present in *some* emitted description for every recognized
    non-literal call.
    """

    events_by_file, metadata = case
    actual_inputs, actual_outputs = _extract_http_routes(events_by_file)
    descriptions = [entry.description for entry in actual_inputs] + [
        entry.description for entry in actual_outputs
    ]

    for path in sorted(metadata):
        _imports, route_calls = metadata[path]
        for event, receiver_label, method_path_split, _is_literal in route_calls:
            if receiver_label is None:
                continue
            if method_path_split is not None:
                continue
            placeholder = (
                f"<dynamic at {event.file_path}:{event.line} on {receiver_label}>"
            )
            matched = any(placeholder in desc for desc in descriptions)
            assert matched, (
                f"non-literal route at {event.file_path}:{event.line} on "
                f"{receiver_label} did not produce the documented "
                f"placeholder {placeholder!r} in any emitted "
                f"description; descriptions: {descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_http_routes_ignores_bootstrap_calls(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool,
                list[
                    tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]
                ],
            ],
        ],
    ],
) -> None:
    """Bootstrap-suppression invariant.

    Validates Requirement 3.5. Running the recognizer over the full
    event stream and over the stream with every bootstrap call
    (``ListenAndServe``, ``ListenAndServeTLS``, ``Serve``, ``Shutdown``,
    ``Close``) removed must produce identical output. Equality of the
    two outputs implies the recognizer never reached an emission for
    any bootstrap call, regardless of receiver shape.
    """

    events_by_file, _ = case
    with_bootstrap = _extract_http_routes(events_by_file)
    without_bootstrap = _extract_http_routes(_stripped_of_bootstrap(events_by_file))
    assert with_bootstrap == without_bootstrap, (
        "HTTP recognizer output changed after removing bootstrap "
        "MethodCallEvents; Requirement 3.5 mandates silent suppression"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_http_routes_ignores_fx_and_viper_receivers(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool,
                list[
                    tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]
                ],
            ],
        ],
    ],
) -> None:
    """fx / viper dispatch-boundary exclusion invariant.

    Validates Requirements 12.1, 12.3, 13.1, 13.2. Running the
    recognizer over the full event stream and over the stream with
    every fx / viper ``MethodCallEvent`` removed must produce
    identical output. Equality of the two outputs implies the
    recognizer consulted no fx / viper receiver for any of its
    decisions.
    """

    events_by_file, _ = case
    with_noise = _extract_http_routes(events_by_file)
    without_noise = _extract_http_routes(_stripped_of_fx_viper(events_by_file))
    assert with_noise == without_noise, (
        "HTTP recognizer output changed after removing fx/viper "
        "MethodCallEvents; the dispatch-boundary exclusion must be "
        "inert (Requirements 12.1, 12.3, 13.1, 13.2)"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_http_routes_is_iteration_order_independent(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool,
                list[
                    tuple[MethodCallEvent, str | None, tuple[str, str] | None, bool]
                ],
            ],
        ],
    ],
) -> None:
    """Determinism invariant: output depends only on input contents.

    Validates Requirement 11.4 (cross-cutting on every Go recognizer).
    The recognizer sorts paths internally, so two mappings with the
    same keys and values but different insertion orders must produce
    identical output lists.
    """

    events_by_file, _ = case
    reversed_mapping = dict(reversed(list(events_by_file.items())))
    assert _extract_http_routes(events_by_file) == _extract_http_routes(reversed_mapping), (
        "HTTP recognizer output depends on insertion order; "
        "Requirement 11.4 mandates deterministic, path-sorted iteration"
    )

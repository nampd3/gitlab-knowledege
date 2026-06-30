# ruff: noqa: E501
# Feature: go-analyzer-support, Property 9: CLI entry-point detection follows the documented `cmd/`/root rules and excludes fx wiring.
"""Property test for the Go CLI entry-point recognizer.

**Validates Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6** (Property 9 in
the design, task 7.10 in ``tasks.md``).

The CLI entry-point recognizer is the fifth and final recognizer
composed by ``extract_go_io`` (task 7.11). This property test
exercises its package-internal entry point :func:`_extract_cli_entry_points`
directly so the per-recognizer contract is pinned independently of
the eventual composition step.

The properties below capture the design's documented contract:

1. **One ``AbstractInput(cli_argument)`` per recognized ``func main()``
   site**, with the binary name derived from the file path or
   ``go.mod`` per Requirements 7.1, 7.2, 7.3:

   * ``cmd/<name>/main.go`` → binary name is ``<name>`` from the
     intermediate directory segment (Requirement 7.1). Always
     eligible, regardless of whether a ``go.mod`` exists.
   * ``cmd/main.go`` → binary name is the last segment of the
     ``go.mod`` module path after stripping the
     ``<host>/<org>/`` prefix (Requirement 7.2). Requires a
     well-formed ``go.mod``; produces no input when ``go.mod`` is
     absent or its ``module`` line cannot be parsed.
   * ``main.go`` at the repository root → same module-path-derived
     binary name as ``cmd/main.go`` (Requirement 7.3). Same
     ``go.mod`` precondition.

   Any other path (``cmd/foo/bar/main.go``, ``internal/main.go``,
   ``pkg/x.go``) is ineligible — the recognizer must contribute zero
   binary inputs even when the file declares ``func main()``.

2. **One ``AbstractInput(cli_argument)`` per recognized
   ``flag.<method>`` call**, with the flag name extracted from the
   correct positional slot per Requirement 7.4:

   * Non-``Var`` registrations (``String``, ``Int``, ``Bool``,
     ``Float64``, ``Duration``, ``NewFlagSet``): flag name is the
     first positional argument when string-literal.
   * ``Var`` registrations (``StringVar``, ``IntVar``, ``BoolVar``,
     ``Float64Var``, ``DurationVar``): flag name is the second
     positional argument when string-literal (the first is the
     pointer receiver).
   * ``Parse``: no flag-name argument; a generic "flag parsing
     invoked" description is recorded.
   * Non-literal flag-name expressions (identifiers, dotted paths,
     calls, etc.) or absent arguments are recorded as
     ``<dynamic>``.

3. **fx and viper exclusions are inert at the dispatch boundary**
   (Requirement 7.5 + Requirements 12.1, 12.3, 13.1, 13.2). Adding
   arbitrary ``fx.<method>`` and ``viper.<method>`` calls to a file
   must not change the recognizer's output, even when the noise's
   method name overlaps with the recognized ``flag.<method>``
   vocabulary.

4. **fx.Lifecycle ``.Append`` exclusion is inert** (Requirement 7.5
   second sentence). Adding arbitrary ``<id>.Append(...)`` calls to a
   file that also imports any ``go.uber.org/fx`` submodule must not
   change the recognizer's output. (``Append`` is not in the recognized
   ``flag.<method>`` set, so this exclusion is observationally
   defensive — it pins the design contract at the recognizer boundary
   for any future cross-recognizer composition.)

5. **Source_Location is attached to every emission** (Requirement
   7.6). The recognizer encodes the originating event's ``file_path``
   and 1-indexed ``line`` into the description as an
   ``at <file>:<line>`` suffix, matching the HTTP, scheduler,
   ActiveMQ, and file-I/O recognizers' convention. Because the typed
   event dataclasses use ``int`` line numbers, the Requirement 7.6
   rejection branch (``line is None`` → drop) is not reachable from
   the public-internal entry; this test asserts the positive side of
   the contract (every emission carries a location suffix).

6. **Deterministic, path-sorted iteration** (Requirement 11.4). The
   recognizer is a pure function of its ``(repository_contents,
   events_by_file)`` inputs. Reversing the mapping's insertion order
   must not change the output list.

The recognizer's :func:`_extract_cli_entry_points` is package-internal; the
test imports it through the module path to mirror the convention used
by every other Go I/O recognizer property test in this directory.
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
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.go._events import (
    CallArg,
    DottedArg,
    FuncDeclEvent,
    IdentArg,
    ImportEvent,
    MethodCallEvent,
    NumberLitArg,
    StringLitArg,
    StructLitArg,
    StructLitEvent,
    UnknownArg,
)
from project_knowledge_mcp.project_analyzer.go.go_io import _extract_cli_entry_points

if TYPE_CHECKING:
    from project_knowledge_mcp.project_analyzer.go._events import ArgRef, GoEvent


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets and shared constants
# ---------------------------------------------------------------------------


#: Identifier alphabet. Constrained to characters Go accepts in
#: identifiers so the generated names look like real source. Uppercase
#: ``M`` is excluded so a synthetic identifier can never collide with
#: the literal token ``main`` used as a sentinel in the
#: ``FuncDeclEvent.name`` check.
_IDENT_CHARS: Final[str] = string.ascii_lowercase + string.digits + "_"


#: Module-path segment alphabet. Mirrors the alphabet used by
#: :mod:`test_property_04_go_purpose_priority` so the two tests'
#: generated ``go.mod`` fixtures stay in close lock-step. Excludes
#: ``/`` (the path separator), spaces (which would split the
#: ``module`` line), and ``"`` (which has no role in module paths).
_SEGMENT_CHARS: Final[str] = string.ascii_letters + string.digits + ".-_"


#: Flag-name literal alphabet. Constrained to characters that survive
#: the description's f-string interpolation unmodified: letters,
#: digits, hyphen, and underscore. Spaces, newlines, and tabs are
#: excluded so a generated flag name cannot break the
#: ``at <file>:<line>`` suffix's line orientation or split the
#: ``flag <name> via flag.<method>()`` prefix on whitespace.
_FLAG_NAME_CHARS: Final[str] = string.ascii_letters + string.digits + "-_"


#: Canonical category enum value every CLI recognizer emission carries.
_CLI_ARGUMENT: Final[AbstractInputCategory] = AbstractInputCategory.CLI_ARGUMENT


#: Canonical description tokens — anchors the per-emission assertions
#: verify so a regression that drops a fragment is caught even when
#: the full-list equality assertion also fails.
_BINARY_PREFIX: Final[str] = "binary "
_BINARY_VIA_FRAGMENT: Final[str] = "via func main() at "
_FLAG_PARSE_DESC_FRAGMENT: Final[str] = "flag parsing invoked via flag.Parse() at "
_FLAG_DESC_PREFIX: Final[str] = "flag "
_LOCATION_FRAGMENT: Final[str] = " at "
_DYNAMIC_FLAG_NAME: Final[str] = "<dynamic>"


#: Receiver chain used by every recognized ``flag.<method>`` call.
_FLAG_CHAIN: Final[tuple[str, ...]] = ("flag",)


#: Flag methods that take the flag name at index 0. Mirrors
#: :data:`go_io._FLAG_NAME_ARG_INDEX` for non-``Var`` entries.
_FLAG_NON_VAR_METHODS: Final[tuple[str, ...]] = (
    "String",
    "Int",
    "Bool",
    "Float64",
    "Duration",
    "NewFlagSet",
)


#: Flag methods that take the flag name at index 1 (after the pointer
#: receiver). Mirrors :data:`go_io._FLAG_NAME_ARG_INDEX` for ``Var``
#: entries.
_FLAG_VAR_METHODS: Final[tuple[str, ...]] = (
    "StringVar",
    "IntVar",
    "BoolVar",
    "Float64Var",
    "DurationVar",
)


#: Flag method with no flag-name argument; produces a generic
#: "flag parsing invoked" description. Mirrors
#: :data:`go_io._FLAG_PARSE_METHOD`.
_FLAG_PARSE_METHOD: Final[str] = "Parse"


#: Full set of recognized ``flag.<method>`` names.
_ALL_FLAG_METHODS: Final[tuple[str, ...]] = (
    *_FLAG_NON_VAR_METHODS,
    *_FLAG_VAR_METHODS,
    _FLAG_PARSE_METHOD,
)


#: Method names on the ``flag`` package that are *not* recognized.
#: Used to populate the "unrelated method on the flag receiver" branch
#: so a regression that accidentally widened the method-name set
#: would surface immediately.
_FLAG_UNRELATED_METHODS: Final[tuple[str, ...]] = (
    "Var",
    "PrintDefaults",
    "Args",
    "Arg",
    "NArg",
    "Set",
    "Lookup",
)


#: A small set of fx / viper method names sufficient to demonstrate
#: that the recognizer's dispatch-boundary exclusion holds. The
#: method-name selection overlaps with the recognized flag vocabulary
#: (``String``, ``Int``, ``Bool``, ``Parse``, etc.) so a regression
#: that forgets the receiver-chain exclusion would surface
#: immediately: the noise event would otherwise match the
#: recognizer's method-name gate.
_FX_METHODS: Final[tuple[str, ...]] = (
    "Provide",
    "Invoke",
    "Module",
    "New",
    "Annotate",
    "WithLogger",
    "String",
    "Int",
    "Bool",
    "Parse",
    "StringVar",
)
_VIPER_METHODS: Final[tuple[str, ...]] = (
    "New",
    "GetString",
    "BindEnv",
    "SetConfigName",
    "AddConfigPath",
    "ReadInConfig",
    "String",
    "Parse",
)


#: ``fx`` import path used to gate the ``.Append`` exclusion on the
#: noise side. Mirrors :data:`go_io._FX_IMPORT_PREFIX`.
_FX_IMPORT_PATH: Final[str] = "go.uber.org/fx"


#: Number of slash-separated segments at which the recognizer strips
#: a ``<host>/<org>/`` prefix from the module path. Mirrors
#: :data:`go_io._MODULE_PATH_PREFIXED_SEGMENTS` rather than importing
#: it so a silent implementation change in the stripping threshold
#: would surface here.
_MODULE_PATH_PREFIXED_SEGMENTS: Final[int] = 3


# ---------------------------------------------------------------------------
# Primitive strategies
# ---------------------------------------------------------------------------


_ident = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8)
_module_segment = st.text(alphabet=_SEGMENT_CHARS, min_size=1, max_size=12)
_flag_name_literal = st.text(alphabet=_FLAG_NAME_CHARS, min_size=1, max_size=12)
_number_literal_text = st.text(alphabet=string.digits, min_size=1, max_size=4)


# ---------------------------------------------------------------------------
# go.mod text + expected binary-name derivation
# ---------------------------------------------------------------------------


@st.composite
def _gomod_case(draw: st.DrawFn) -> tuple[str | None, str | None]:
    """Generate ``(gomod_text, expected_module_binary_name)``.

    Five branches stratify the recognizer's :func:`_module_binary_name`
    decision tree:

    * ``absent`` — no ``go.mod`` in the repository. Module-derived
      binary name is ``None``; the ``cmd/main.go`` and root ``main.go``
      branches contribute no input.
    * ``malformed`` — ``go.mod`` exists but contains no parseable
      ``module`` line. ``parse_go_mod`` returns ``None``; expected
      module binary name is ``None``.
    * ``bare`` — one segment (``"repayment_service"``,
      ``"cat-service"``, etc.). No stripping; expected binary name
      is the segment itself.
    * ``two_segments`` — two segments (``"acme/x"``). Below the
      stripping threshold; expected binary name is the second
      segment.
    * ``prefixed_3`` — three segments (``"github.com/acme/svc"``).
      The first two segments are stripped; expected binary name is
      the third.
    * ``prefixed_4`` — four segments
      (``"github.com/foo/bar/baz"``). The first two segments are
      stripped; expected binary name is the last segment of what
      remains (``"baz"``).
    """

    kind = draw(
        st.sampled_from(
            ("absent", "malformed", "bare", "two_segments", "prefixed_3", "prefixed_4"),
        ),
    )
    if kind == "absent":
        return None, None
    if kind == "malformed":
        # ``go.mod`` exists but every line either lacks the ``module``
        # keyword or is otherwise unparseable. A single comment line
        # is sufficient.
        body = draw(st.text(alphabet=string.ascii_lowercase, min_size=0, max_size=20))
        return f"// {body}\n", None
    if kind == "bare":
        seg = draw(_module_segment)
        return f"module {seg}\n", seg
    if kind == "two_segments":
        seg1 = draw(_module_segment)
        seg2 = draw(_module_segment)
        return f"module {seg1}/{seg2}\n", seg2
    if kind == "prefixed_3":
        host = draw(_module_segment)
        org = draw(_module_segment)
        svc = draw(_module_segment)
        return f"module {host}/{org}/{svc}\n", svc
    # ``prefixed_4``
    host = draw(_module_segment)
    org = draw(_module_segment)
    mid = draw(_module_segment)
    last = draw(_module_segment)
    return f"module {host}/{org}/{mid}/{last}\n", last


# ---------------------------------------------------------------------------
# Per-event generators — recognized flag.<method> calls
# ---------------------------------------------------------------------------


@st.composite
def _flag_name_argument(draw: st.DrawFn) -> tuple["ArgRef | None", str]:
    """Generate a flag-name argument plus its expected description fragment.

    Returns ``(value_or_none, expected_name)``:

    * ``value_or_none`` is the ``ArgRef`` to attach at the flag-name
      slot, or ``None`` to omit the argument (truncated call list).
    * ``expected_name`` is the substring the recognizer must place in
      the description — the verbatim literal for ``StringLitArg``,
      :data:`_DYNAMIC_FLAG_NAME` for every other variant (field
      access, identifier, nested call, opaque text, numeric literal,
      struct literal, absent argument).
    """

    kind = draw(
        st.sampled_from(
            (
                "string",
                "dotted",
                "ident",
                "call",
                "unknown",
                "number",
                "struct",
                "absent",
            ),
        ),
    )
    if kind == "string":
        value = draw(_flag_name_literal)
        return StringLitArg(value=value), value
    if kind == "dotted":
        parts = tuple(
            draw(st.lists(_ident, min_size=2, max_size=4)),
        )
        return DottedArg(parts=parts), _DYNAMIC_FLAG_NAME
    if kind == "ident":
        return IdentArg(name=draw(_ident)), _DYNAMIC_FLAG_NAME
    if kind == "call":
        nested = MethodCallEvent(
            receiver_chain=(draw(_ident),),
            method_name=draw(_ident),
            args=(),
            file_path="dummy.go",
            line=1,
        )
        return CallArg(call=nested), _DYNAMIC_FLAG_NAME
    if kind == "unknown":
        return UnknownArg(text=draw(_ident)), _DYNAMIC_FLAG_NAME
    if kind == "number":
        return NumberLitArg(text=draw(_number_literal_text)), _DYNAMIC_FLAG_NAME
    if kind == "struct":
        struct_event = StructLitEvent(
            type_name="X",
            package_alias=None,
            fields=(),
            is_pointer=False,
            file_path="dummy.go",
            line=1,
        )
        return StructLitArg(event=struct_event), _DYNAMIC_FLAG_NAME
    # ``absent`` — the flag-name slot is past the end of the args.
    return None, _DYNAMIC_FLAG_NAME


def _filler_arg(name: str) -> "ArgRef":
    """Return a generic non-name filler argument.

    Stands in for the pointer receiver (``&v``) of ``Var`` calls and
    for the default-value / usage arguments of the non-``Var``
    registrations. The recognizer never inspects these slots, so a
    bare :class:`IdentArg` is sufficient.
    """

    return IdentArg(name=name)


@st.composite
def _flag_call_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str]:
    """Generate one recognized ``flag.<method>(...)`` call event.

    Returns ``(event, expected_description)``. The generator
    stratifies over the recognizer's full decision tree:

    * Non-``Var`` registration → flag name at index 0.
    * ``Var`` registration → flag name at index 1 (the index-0
      pointer-receiver slot is filled with an :class:`IdentArg`).
    * ``Parse`` → no flag-name argument; generic description.
    """

    method = draw(st.sampled_from(_ALL_FLAG_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))

    if method == _FLAG_PARSE_METHOD:
        # ``flag.Parse()`` is conventionally zero-arity; the
        # recognizer ignores any actual arguments. Generating a small
        # arbitrary args list exercises that the recognizer's
        # ``Parse`` branch does not inspect them.
        n_args = draw(st.integers(min_value=0, max_value=2))
        args = tuple(_filler_arg(f"a{i}") for i in range(n_args))
        event = MethodCallEvent(
            receiver_chain=_FLAG_CHAIN,
            method_name=method,
            args=args,
            file_path=file_path,
            line=line,
        )
        expected_desc = (
            f"{_FLAG_PARSE_DESC_FRAGMENT}{file_path}:{line}"
        )
        return event, expected_desc

    flag_value, expected_name = draw(_flag_name_argument())
    is_var = method in _FLAG_VAR_METHODS
    if is_var:
        # The pointer-receiver slot (index 0) is always present in
        # Go's calling convention; only the index-1 flag-name slot
        # may be absent from the generated argument list.
        if flag_value is None:
            args = (_filler_arg("ptr"),)
        else:
            args = (_filler_arg("ptr"), flag_value)
    else:
        if flag_value is None:
            args = ()
        else:
            args = (flag_value,)
    # Optionally append trailing filler arguments (default value,
    # usage text) so the recognizer's "inspect by index" gate is
    # exercised against longer-than-minimum arg lists.
    n_trailing = draw(st.integers(min_value=0, max_value=2))
    args = args + tuple(_filler_arg(f"t{i}") for i in range(n_trailing))

    event = MethodCallEvent(
        receiver_chain=_FLAG_CHAIN,
        method_name=method,
        args=args,
        file_path=file_path,
        line=line,
    )
    expected_desc = (
        f"{_FLAG_DESC_PREFIX}{expected_name} via flag.{method}() "
        f"at {file_path}:{line}"
    )
    return event, expected_desc


# ---------------------------------------------------------------------------
# Per-event generators — noise events
# ---------------------------------------------------------------------------


@st.composite
def _flag_unrelated_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a ``flag.<unrecognized>`` call.

    A real-world Go source can call ``flag.PrintDefaults()`` or
    ``flag.Args()`` alongside the recognized registrations. The
    recognizer must reject these without emission via the
    method-name gate (Requirement 7.4 enumerates a closed set of
    eleven names).
    """

    method = draw(st.sampled_from(_FLAG_UNRELATED_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))
    # A string-literal first argument that looks flag-name-shaped, so
    # a regression that forgot the method-name gate would emit an
    # entry the expected-output computation does not predict.
    return MethodCallEvent(
        receiver_chain=_FLAG_CHAIN,
        method_name=method,
        args=(StringLitArg(value="regression-canary"),),
        file_path=file_path,
        line=line,
    )


@st.composite
def _non_flag_receiver_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a recognized flag-method name on a non-``flag`` receiver.

    Covers the "receiver chain does not equal ``('flag',)``" branch
    of :func:`go_io._try_flag_description`. The method name is drawn
    from the recognized set so a regression that widened the
    receiver-chain gate (for example, by matching any single-id
    receiver whose method is in :data:`_ALL_FLAG_METHODS`) would
    surface immediately.
    """

    receiver = draw(_ident.filter(lambda n: n not in ("flag", "fx", "viper")))
    method = draw(st.sampled_from(_ALL_FLAG_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))
    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name=method,
        args=(StringLitArg(value="regression-canary"),),
        file_path=file_path,
        line=line,
    )


@st.composite
def _fx_or_viper_noise_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a noise event excluded by the fx / viper dispatch-boundary skip.

    The recognizer must treat these as inert (Requirement 7.5 +
    Requirements 12.1, 12.3, 13.1, 13.2). The noise's method-name
    selection deliberately overlaps with the recognized flag
    vocabulary (``String``, ``Int``, ``Bool``, ``Parse``, etc.) so a
    regression that forgets the receiver-chain exclusion would
    surface immediately: the noise event would otherwise match the
    recognizer's method-name gate. The argument list is shaped like
    a valid registration so a regression that fell through would
    also satisfy the flag-name-argument extraction logic.
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
        args=(
            StringLitArg(value="regression-canary"),
            _filler_arg("default"),
        ),
        file_path=file_path,
        line=line,
    )


@st.composite
def _fx_lifecycle_append_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate an ``<id>.Append(...)`` call standing in for ``fx.Lifecycle.Append``.

    The recognizer's :func:`_is_excluded_fx_lifecycle_append`
    skip-gate matches any method call with:

    * method name exactly ``Append``;
    * receiver chain of length 1 (a single identifier);
    * the surrounding file containing any ``go.uber.org/fx`` import.

    Generating such events alongside an fx ``ImportEvent`` in the
    same file exercises the gate. The noise's argument shape is
    irrelevant to the recognizer because ``Append`` is not in the
    recognized ``flag.<method>`` set — the exclusion is defensive
    rather than load-bearing — but a structurally-valid struct
    literal is supplied so the noise reads like real source.
    """

    receiver = draw(_ident.filter(lambda n: n not in ("fx", "viper", "flag")))
    line = draw(st.integers(min_value=1, max_value=999))
    hook = StructLitEvent(
        type_name="Hook",
        package_alias="fx",
        fields=(
            ("OnStart", IdentArg(name="onStart")),
            ("OnStop", IdentArg(name="onStop")),
        ),
        is_pointer=False,
        file_path=file_path,
        line=line,
    )
    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name="Append",
        args=(StructLitArg(event=hook),),
        file_path=file_path,
        line=line,
    )


# ---------------------------------------------------------------------------
# Path-category model
# ---------------------------------------------------------------------------


# Path categories the recognizer must distinguish (Requirements 7.1,
# 7.2, 7.3). Names are exposed as constants so the per-emission
# assertions can match against them by name rather than by inferring
# eligibility from the path string.
_PATH_CMD_NAMED: Final[str] = "cmd_named"  # cmd/<name>/main.go
_PATH_CMD_MAIN: Final[str] = "cmd_main"  # cmd/main.go
_PATH_ROOT_MAIN: Final[str] = "root_main"  # main.go at repo root
_PATH_INELIGIBLE: Final[str] = "ineligible"  # any other path


#: Hand-picked ineligible path templates. Each one is a real shape
#: that exists in the four sample repositories (under ``internal/``,
#: ``pkg/``, or deeper than ``cmd/<name>/``); none of them is allowed
#: to contribute a binary input even when ``func main()`` is declared.
_INELIGIBLE_TEMPLATES: Final[tuple[str, ...]] = (
    "internal/main.go",
    "pkg/main.go",
    "cmd/sub/extra/main.go",  # deeper than ``cmd/<name>/main.go``
    "cmd/foo/bar/main.go",  # depth-2 under ``cmd/``
    "internal/server/server.go",
    "pkg/util/helpers.go",
    "main_test.go",  # root, but not ``main.go``
)


# ---------------------------------------------------------------------------
# Per-file fixture
# ---------------------------------------------------------------------------


@st.composite
def _file_record(
    draw: st.DrawFn,
    *,
    path: str,
    category: str,
    module_binary_name: str | None,
) -> tuple[list["GoEvent"], list[AbstractInput]]:
    """Generate one file's event list plus the per-file expected emissions.

    Returns ``(events, expected_inputs)``:

    * ``events`` is the order-preserving event list assigned to the
      file's key in ``events_by_file``.
    * ``expected_inputs`` is the per-file ``AbstractInput`` records
      the recognizer must produce for this file. The aggregate
      expected list is the per-file lists concatenated in
      path-sorted order (path sorting is handled by the caller).

    The generator interleaves four event populations:

    1. An optional ``func main()`` :class:`FuncDeclEvent`. Whether
       one is emitted is independently chosen from the path
       category, so the test exercises:

       * eligible path + has main → emits a binary input;
       * eligible path + no main → emits no binary input;
       * ineligible path + has main → emits no binary input
         (the path gate rejects);
       * ineligible path + no main → emits no binary input.

    2. Recognized ``flag.<method>`` calls — between zero and three
       per file — drawn from
       :func:`_flag_call_event`. Each contributes one expected
       input.

    3. ``flag.<unrelated>``, ``<other>.<flag-method>``, fx, viper,
       and ``<id>.Append`` noise — each in the inert-noise contract
       and contributing zero expected inputs.

    4. Optionally an ``ImportEvent`` for ``go.uber.org/fx`` so the
       ``.Append`` exclusion's file-level gate fires on some
       examples and not on others.
    """

    events: list[GoEvent] = []
    expected: list[AbstractInput] = []

    # File-level fx import. Independently chosen from category so
    # both fx-importing and non-fx-importing files of each category
    # are exercised.
    file_imports_fx = draw(st.booleans())
    if file_imports_fx:
        events.append(
            ImportEvent(
                path=_FX_IMPORT_PATH,
                alias=None,
                file_path=path,
                line=draw(st.integers(min_value=1, max_value=999)),
            ),
        )

    # Optional ``func main()`` declaration.
    has_main = draw(st.booleans())
    main_line: int | None = None
    if has_main:
        main_line = draw(st.integers(min_value=1, max_value=999))
        events.append(
            FuncDeclEvent(
                name="main",
                receiver_type=None,
                file_path=path,
                line=main_line,
                body_token_range=(0, 0),
            ),
        )
    # Optionally a *non-main* free-function declaration to confirm
    # the recognizer's name gate; never contributes a binary input.
    if draw(st.booleans()):
        events.append(
            FuncDeclEvent(
                name=draw(_ident.filter(lambda n: n != "main")),
                receiver_type=None,
                file_path=path,
                line=draw(st.integers(min_value=1, max_value=999)),
                body_token_range=(0, 0),
            ),
        )
    # Optionally a ``func main()`` with a receiver (a method, not a
    # free function); also never contributes a binary input.
    if draw(st.booleans()):
        events.append(
            FuncDeclEvent(
                name="main",
                receiver_type=draw(_ident),
                file_path=path,
                line=draw(st.integers(min_value=1, max_value=999)),
                body_token_range=(0, 0),
            ),
        )

    # Compute the expected binary input for the recognized
    # ``func main()`` declaration, mirroring the recognizer's
    # :func:`_try_main_binary_description` decision tree.
    if has_main and main_line is not None:
        binary_name = _expected_binary_name(
            path=path,
            category=category,
            module_binary_name=module_binary_name,
        )
        if binary_name is not None:
            expected.append(
                AbstractInput(
                    category=_CLI_ARGUMENT,
                    description=(
                        f"{_BINARY_PREFIX}{binary_name} "
                        f"{_BINARY_VIA_FRAGMENT}{path}:{main_line}"
                    ),
                ),
            )

    # Recognized flag.<method> calls.
    flag_events = draw(
        st.lists(_flag_call_event(file_path=path), min_size=0, max_size=3),
    )
    for event, desc in flag_events:
        events.append(event)
        expected.append(
            AbstractInput(category=_CLI_ARGUMENT, description=desc),
        )

    # Noise — all inert.
    unrelated_flag_methods = draw(
        st.lists(_flag_unrelated_event(file_path=path), min_size=0, max_size=2),
    )
    events.extend(unrelated_flag_methods)
    non_flag_receivers = draw(
        st.lists(_non_flag_receiver_event(file_path=path), min_size=0, max_size=2),
    )
    events.extend(non_flag_receivers)
    fx_viper_noise = draw(
        st.lists(_fx_or_viper_noise_event(file_path=path), min_size=1, max_size=3),
    )
    events.extend(fx_viper_noise)
    if file_imports_fx:
        # The ``.Append`` exclusion only fires when the file imports
        # fx. Generate ``.Append`` noise only on fx-importing files
        # so the recognizer's file-level gate is exercised on the
        # positive side; non-fx-importing files exercise the
        # gate's negative side by having no ``.Append`` events that
        # the gate must classify.
        append_noise = draw(
            st.lists(
                _fx_lifecycle_append_event(file_path=path),
                min_size=0,
                max_size=2,
            ),
        )
        events.extend(append_noise)

    return events, expected


def _expected_binary_name(
    *,
    path: str,
    category: str,
    module_binary_name: str | None,
) -> str | None:
    """Mirror :func:`go_io._binary_name_for_path` on the test side.

    Returns the binary name implied by ``(path, category)`` or
    ``None`` when no binary input is expected for this file.

    The ``category`` argument is the canonical record of the path's
    classification computed at generation time; using it rather than
    re-parsing the path here keeps the test's predictor in lock-step
    with the generator's intent. The recognizer reaches the same
    classification through path inspection alone.
    """

    if category == _PATH_CMD_NAMED:
        parts = path.split("/")
        # ``cmd/<name>/main.go`` has exactly three segments; the
        # generator guarantees a non-empty middle segment.
        return parts[1] if len(parts) == 3 and parts[1] else None
    if category in (_PATH_CMD_MAIN, _PATH_ROOT_MAIN):
        return module_binary_name
    # ``_PATH_INELIGIBLE``: no binary input.
    return None


# ---------------------------------------------------------------------------
# Multi-file fixture
# ---------------------------------------------------------------------------


@st.composite
def _events_by_file(
    draw: st.DrawFn,
) -> tuple[
    RepositoryContents,
    dict[str, list["GoEvent"]],
    list[AbstractInput],
]:
    """Generate the full ``(repository_contents, events_by_file, expected_inputs)`` case.

    Returns:
        * ``repository_contents`` — a :class:`RepositoryContents`
          carrying the generated ``go.mod`` text (if any) under the
          ``go.mod`` key. Other files are intentionally not added to
          ``files`` because the recognizer only consults
          ``repository_contents`` for ``go.mod``.
        * ``events_by_file`` — the per-file event mapping the
          recognizer iterates over.
        * ``expected_inputs`` — the deduplicated list of
          ``AbstractInput`` records the recognizer must produce, in
          the order the recognizer's path-sorted iteration would
          produce them.

    The path layout is generated as follows:

    * Between zero and two distinct ``cmd/<name>/main.go`` paths
      with unique ``<name>`` segments.
    * Optionally one ``cmd/main.go`` path.
    * Optionally one ``main.go`` path at the repository root.
    * Between zero and two distinct ineligible paths drawn from
      :data:`_INELIGIBLE_TEMPLATES`.

    Every chosen path is then independently populated with events
    via :func:`_file_record`, and the per-file expected lists are
    concatenated in path-sorted order to produce the aggregate
    expected list.
    """

    gomod_text, module_binary_name = draw(_gomod_case())

    # ``cmd/<name>/main.go`` paths with unique names.
    cmd_named_names = draw(
        st.lists(
            st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=6),
            min_size=0,
            max_size=2,
            unique=True,
        ),
    )
    cmd_named_paths = [f"cmd/{name}/main.go" for name in cmd_named_names]

    include_cmd_main = draw(st.booleans())
    include_root_main = draw(st.booleans())
    ineligible_paths = draw(
        st.lists(
            st.sampled_from(_INELIGIBLE_TEMPLATES),
            min_size=0,
            max_size=2,
            unique=True,
        ),
    )

    files_by_category: list[tuple[str, str]] = []
    for p in cmd_named_paths:
        files_by_category.append((p, _PATH_CMD_NAMED))
    if include_cmd_main:
        files_by_category.append(("cmd/main.go", _PATH_CMD_MAIN))
    if include_root_main:
        files_by_category.append(("main.go", _PATH_ROOT_MAIN))
    for p in ineligible_paths:
        files_by_category.append((p, _PATH_INELIGIBLE))

    # Always generate at least one file so the recognizer is
    # exercised with a non-empty input; otherwise both the actual
    # output and the expected output are trivially equal at the
    # empty list, which would let regressions pass.
    if not files_by_category:
        files_by_category.append((draw(st.sampled_from(_INELIGIBLE_TEMPLATES)), _PATH_INELIGIBLE))

    events_by_file: dict[str, list[GoEvent]] = {}
    per_file_expected: dict[str, list[AbstractInput]] = {}
    for path, category in files_by_category:
        events, expected = draw(
            _file_record(
                path=path,
                category=category,
                module_binary_name=module_binary_name,
            ),
        )
        events_by_file[path] = events
        per_file_expected[path] = expected

    # Aggregate the per-file expected lists in path-sorted order to
    # match the recognizer's iteration.
    expected_inputs: list[AbstractInput] = []
    seen: set[tuple[AbstractInputCategory, str]] = set()
    for path in sorted(per_file_expected):
        for entry in per_file_expected[path]:
            key = (entry.category, entry.description)
            if key in seen:
                continue
            seen.add(key)
            expected_inputs.append(entry)

    # Build the RepositoryContents. ``go.mod`` is the only file the
    # recognizer reads from the snapshot; the per-file event mapping
    # is delivered through ``events_by_file`` directly.
    files: dict[str, str] = {}
    if gomod_text is not None:
        files["go.mod"] = gomod_text
    rc = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeefcafef00d",
        files=files,
    )

    return rc, events_by_file, expected_inputs


# ---------------------------------------------------------------------------
# Helpers used by the inert-noise properties
# ---------------------------------------------------------------------------


def _stripped_of_fx_viper(
    events_by_file: dict[str, list["GoEvent"]],
) -> dict[str, list["GoEvent"]]:
    """Return ``events_by_file`` with every ``fx`` / ``viper`` method call removed.

    Equality of the recognizer's output before and after this
    stripping implies the recognizer consulted no ``fx`` / ``viper``
    receiver for any of its decisions (Requirements 7.5, 12.1, 12.3,
    13.1, 13.2).
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


def _stripped_of_fx_appends(
    events_by_file: dict[str, list["GoEvent"]],
) -> dict[str, list["GoEvent"]]:
    """Return ``events_by_file`` with every ``<id>.Append(...)`` call removed.

    Combined with :func:`_stripped_of_fx_viper`, equality of the
    recognizer's output before and after this stripping implies the
    recognizer never relies on the presence or absence of
    fx.Lifecycle ``.Append`` registrations for any of its decisions
    (Requirement 7.5 second sentence).
    """

    stripped: dict[str, list[GoEvent]] = {}
    for path, events in events_by_file.items():
        kept: list[GoEvent] = []
        for event in events:
            if (
                isinstance(event, MethodCallEvent)
                and event.method_name == "Append"
                and len(event.receiver_chain) == 1
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
def test_extract_cli_entry_points_matches_expected_output(
    case: tuple[
        RepositoryContents,
        dict[str, list["GoEvent"]],
        list[AbstractInput],
    ],
) -> None:
    """Property 9 main invariant: CLI detection follows the documented mapping.

    Validates Requirements 7.1, 7.2, 7.3, 7.4 jointly. The expected
    output is computed by mirroring the recognizer's decision tree
    on the per-file generator metadata (path category, ``func main()``
    presence, recognized flag-call list), then asserting full-list
    equality. Per-emission invariants validated by construction and
    by the universal-suffix check below:

    * One ``AbstractInput(cli_argument)`` per recognized
      ``func main()`` site on an eligible path
      (Requirements 7.1, 7.2, 7.3); zero per ineligible path or per
      eligible path without ``func main()``.
    * One ``AbstractInput(cli_argument)`` per recognized
      ``flag.<method>`` call with the correct flag-name extraction
      (Requirement 7.4).
    * Every emission carries an ``at <file>:<line>`` Source_Location
      suffix (Requirement 7.6); the typed dataclass guarantees no
      emission with ``line is None`` can be produced from a
      well-formed event stream.
    """

    rc, events_by_file, expected_inputs = case

    actual_inputs = _extract_cli_entry_points(rc, events_by_file)

    assert actual_inputs == expected_inputs, (
        "CLI recognizer inputs diverged from expected:\n"
        f"  actual:   {actual_inputs!r}\n"
        f"  expected: {expected_inputs!r}"
    )

    # Universal invariants on the actual output, independent of the
    # expected-output computation. These would catch a regression
    # that made the recognizer's output structurally wrong even when
    # the full-list equality assertion also passed (e.g. by a
    # coincidental two-bug cancellation in the expected-output
    # mirror).
    for entry in actual_inputs:
        assert entry.category is _CLI_ARGUMENT, (
            f"CLI recognizer emitted entry {entry!r} with category "
            f"{entry.category!r}; Requirement 7.1 / 7.2 / 7.3 / 7.4 "
            f"mandate ``cli_argument`` for every CLI input"
        )
        assert _LOCATION_FRAGMENT in entry.description, (
            f"CLI recognizer emitted entry {entry!r} without an "
            f"'at <file>:<line>' suffix; Requirement 7.6 mandates a "
            f"Source_Location on every emission"
        )
        # The description must start with one of the two canonical
        # prefixes the recognizer assigns: ``"binary "`` for the
        # ``func main()`` branch (Requirements 7.1, 7.2, 7.3) or
        # ``"flag "`` for the ``flag.<method>`` branch
        # (Requirement 7.4).
        assert (
            entry.description.startswith(_BINARY_PREFIX)
            or entry.description.startswith(_FLAG_DESC_PREFIX)
        ), (
            f"CLI recognizer emitted entry {entry!r} without a "
            f"recognized prefix; the description must begin with "
            f"'binary ' or 'flag '"
        )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_preserves_string_literal_flag_name_verbatim(
    case: tuple[
        RepositoryContents,
        dict[str, list["GoEvent"]],
        list[AbstractInput],
    ],
) -> None:
    """Verbatim-preservation invariant for string-literal flag-name arguments.

    Validates Requirement 7.4 ("includes the flag name when the call
    is one of the named registration functions and the flag-name
    argument is a string literal"). For every recognized
    ``flag.<method>`` call whose flag-name slot is a
    :class:`StringLitArg`, the recognizer must place the literal
    contents into the description verbatim — with no stripping,
    normalization, or truncation — paired with the call's
    Source_Location suffix.
    """

    rc, events_by_file, _ = case
    actual_inputs = _extract_cli_entry_points(rc, events_by_file)
    descriptions = [entry.description for entry in actual_inputs]

    for path in sorted(events_by_file):
        for event in events_by_file[path]:
            if not isinstance(event, MethodCallEvent):
                continue
            if event.receiver_chain != _FLAG_CHAIN:
                continue
            # Only the named registration variants carry a flag-name
            # argument; ``Parse`` and unrecognized methods are out
            # of scope for this property.
            if event.method_name in _FLAG_NON_VAR_METHODS:
                name_idx = 0
            elif event.method_name in _FLAG_VAR_METHODS:
                name_idx = 1
            else:
                continue
            if name_idx >= len(event.args):
                continue
            arg = event.args[name_idx]
            if not isinstance(arg, StringLitArg):
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            literal = arg.value
            matched = any(
                literal in desc and location_suffix in desc
                for desc in descriptions
            )
            assert matched, (
                f"string-literal flag name {literal!r} at "
                f"{event.file_path}:{event.line} (method "
                f"flag.{event.method_name}) was not preserved "
                f"verbatim in any emitted description; descriptions: "
                f"{descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_ignores_fx_and_viper_receivers(
    case: tuple[
        RepositoryContents,
        dict[str, list["GoEvent"]],
        list[AbstractInput],
    ],
) -> None:
    """fx / viper dispatch-boundary exclusion invariant.

    Validates Requirement 7.5 first sentence + Requirements 12.1,
    12.3, 13.1, 13.2. Running the recognizer over the full event
    stream and over the stream with every ``fx.<method>`` and
    ``viper.<method>`` call removed must produce identical output.
    Equality of the two outputs implies the recognizer consulted no
    fx / viper receiver for any of its decisions, including the
    (deliberately adversarial) noise events whose method names
    overlap with the recognized ``flag.<method>`` vocabulary.
    """

    rc, events_by_file, _ = case
    with_noise = _extract_cli_entry_points(rc, events_by_file)
    without_noise = _extract_cli_entry_points(rc, _stripped_of_fx_viper(events_by_file))
    assert with_noise == without_noise, (
        "CLI recognizer output changed after removing fx/viper "
        "MethodCallEvents; the dispatch-boundary exclusion must be "
        "inert (Requirements 7.5, 12.1, 12.3, 13.1, 13.2)"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_ignores_fx_lifecycle_append(
    case: tuple[
        RepositoryContents,
        dict[str, list["GoEvent"]],
        list[AbstractInput],
    ],
) -> None:
    """fx.Lifecycle ``.Append`` exclusion invariant.

    Validates Requirement 7.5 second sentence. Running the
    recognizer over the full event stream and over the stream with
    every single-id-receiver ``.Append(...)`` call removed must
    produce identical output. Equality of the two outputs implies
    the recognizer never emits a CLI input from an
    ``fx.Lifecycle.Append`` hook registration.

    Note: ``Append`` is not in the recognized ``flag.<method>`` set,
    so the recognizer would naturally drop these calls at the
    method-name gate even without the explicit exclusion. The
    property is pinned regardless so a future cross-recognizer
    composition step cannot accidentally surface fx wiring as a CLI
    input.
    """

    rc, events_by_file, _ = case
    with_noise = _extract_cli_entry_points(rc, events_by_file)
    without_noise = _extract_cli_entry_points(
        rc, _stripped_of_fx_appends(events_by_file),
    )
    assert with_noise == without_noise, (
        "CLI recognizer output changed after removing "
        "single-id-receiver .Append(...) calls; the fx.Lifecycle "
        "exclusion must be inert (Requirement 7.5)"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_is_iteration_order_independent(
    case: tuple[
        RepositoryContents,
        dict[str, list["GoEvent"]],
        list[AbstractInput],
    ],
) -> None:
    """Determinism invariant: output depends only on input contents.

    Validates Requirement 11.4 (cross-cutting on every Go
    recognizer). The recognizer sorts paths internally, so two
    mappings with the same keys and values but different insertion
    orders must produce identical output lists.
    """

    rc, events_by_file, _ = case
    reversed_mapping = dict(reversed(list(events_by_file.items())))
    assert _extract_cli_entry_points(rc, events_by_file) == _extract_cli_entry_points(
        rc, reversed_mapping,
    ), (
        "CLI recognizer output depends on insertion order; "
        "Requirement 11.4 mandates deterministic, path-sorted iteration"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_ineligible_paths_emit_no_binary(
    case: tuple[
        RepositoryContents,
        dict[str, list["GoEvent"]],
        list[AbstractInput],
    ],
) -> None:
    """Path-eligibility invariant for the ``func main()`` branch.

    Validates Requirements 7.1, 7.2, 7.3 negative side: only the
    three documented path shapes are eligible to contribute a
    binary input. Files at any other path — even when they declare
    ``func main()`` — must not contribute a binary description
    naming that file as the entry point. The check is per-file: for
    every ineligible path in the mapping, the recognizer's output
    must contain no description whose ``at <file>:<line>`` suffix
    references that path *and* whose prefix is ``"binary "``.
    """

    rc, events_by_file, _ = case
    actual_inputs = _extract_cli_entry_points(rc, events_by_file)
    binary_descriptions = [
        entry.description
        for entry in actual_inputs
        if entry.description.startswith(_BINARY_PREFIX)
    ]

    for path in events_by_file:
        # Classify the path as eligible or ineligible by mirroring
        # the recognizer's :func:`_binary_name_for_path` shape gate.
        parts = path.split("/")
        is_cmd_named = (
            len(parts) == 3 and parts[0] == "cmd" and parts[2] == "main.go"
        )
        is_eligible = is_cmd_named or path == "cmd/main.go" or path == "main.go"
        if is_eligible:
            continue
        location_suffix = f"at {path}:"
        for desc in binary_descriptions:
            assert location_suffix not in desc, (
                f"ineligible path {path!r} contributed a binary "
                f"description {desc!r}; Requirements 7.1 / 7.2 / 7.3 "
                f"restrict binary inputs to cmd/<name>/main.go, "
                f"cmd/main.go, and main.go at the repository root"
            )

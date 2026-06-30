# ruff: noqa: E501
# Feature: go-analyzer-support, Property 9: CLI entry-point detection follows the documented cmd/`/root rules and excludes fx wiring.
"""Property test for the Go CLI entry-point recognizer.

**Validates Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6** (Property 9 in
the design, task 7.10 in ``tasks.md``).

The CLI entry-point recognizer is the fifth of five recognizers
composed by ``extract_go_io`` (task 7.11). This property test
exercises its package-internal entry point
:func:`_extract_cli_entry_points` directly so the per-recognizer
contract is pinned independently of the eventual composition step.

The properties below capture the design's documented contract:

1. **``cmd/<name>/main.go`` emits one input naming the binary as
   ``<name>``** (Requirement 7.1). The recognizer reads the binary
   name from the path's middle segment, *not* from ``go.mod``, so the
   input is emitted even when no ``go.mod`` is present.

2. **``cmd/main.go`` and root ``main.go`` emit one input naming the
   binary using the last segment of the stripped ``go.mod`` module
   path** (Requirements 7.2, 7.3). When no ``go.mod`` is present, the
   recognizer emits nothing for these paths because the binary name
   cannot be derived. The stripping rule discards any leading
   ``<host>/<org>/`` prefix when the module path has three or more
   slash-separated segments; bare module names pass through unchanged.

3. **Each recognized ``flag.<X>(...)`` call emits one input**
   (Requirement 7.4). String-literal flag-name arguments appear
   verbatim in the description; every other argument shape — including
   identifier references, dotted paths, missing arguments, and so on —
   is recorded with the ``<dynamic>`` placeholder. ``flag.Parse()`` is
   recognized as a special "flag parsing invoked" emission with no
   flag name.

4. **fx wiring contributes zero inputs** (Requirement 7.5, jointly
   with Requirements 12.1 and 12.3). Any ``MethodCallEvent`` whose
   receiver chain begins with ``fx`` is dropped at the dispatch
   boundary. Single-identifier ``.Append`` calls in a file that
   imports any ``go.uber.org/fx`` submodule are likewise dropped (the
   conservative ``fx.Lifecycle`` approximation documented in design
   §"go.go_io" section 7).

5. **viper configuration reads contribute zero inputs** (Requirements
   13.1 and 13.2). Any ``MethodCallEvent`` whose receiver chain begins
   with ``viper`` is dropped at the dispatch boundary.

6. **Source_Location rejection** (Requirement 7.6). The recognizer
   never emits an input for an event whose ``line`` cannot be
   determined. The typed event dataclasses use ``int`` line numbers
   and never produce ``None`` in practice, but the guard is encoded
   explicitly so the requirement's contract is visible at the
   recognizer boundary; this test exercises it by constructing a
   degenerate event whose ``line`` slot has been overwritten with
   ``None``.

7. **Deterministic, path-sorted iteration** (Requirement 11.4). The
   recognizer is a pure function of its input mapping. Reversing the
   mapping's insertion order must not change the output list.

The recognizer's :func:`_extract_cli_entry_points` is
package-internal; the test imports it through the module path to
mirror the convention used by ``test_property_05_go_http_detection.py``
and ``test_property_06_go_scheduler_detection.py`` for their
respective package-internal entry points.
"""

from __future__ import annotations

import string
from dataclasses import replace
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
    DottedArg,
    FuncDeclEvent,
    IdentArg,
    ImportEvent,
    MethodCallEvent,
    NumberLitArg,
    StringLitArg,
    UnknownArg,
)
from project_knowledge_mcp.project_analyzer.go.go_io import (
    _extract_cli_entry_points,
)

if TYPE_CHECKING:
    from project_knowledge_mcp.project_analyzer.go._events import ArgRef, GoEvent


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets and shared constants
# ---------------------------------------------------------------------------


#: Identifier alphabet for binary names, receiver names, and flag names.
#: Constrained to characters Go accepts in identifiers plus the hyphen
#: so a generated ``cmd/<name>/main.go`` mid-segment can match the
#: kebab-case binaries observed in the sample repositories
#: (``cat-service``, ``payment-service``). The empty-string draw is
#: filtered out at the strategy level because the recognizer rejects
#: empty ``<name>`` segments.
_IDENT_CHARS: Final[str] = string.ascii_letters + string.digits + "_-"


#: Single-segment alphabet for module-path components. Excludes ``/``
#: so a generated module-path segment cannot accidentally split into
#: two segments. The dot is included so realistic host segments like
#: ``github.com`` can be generated.
_SEGMENT_CHARS: Final[str] = string.ascii_letters + string.digits + ".-_"


#: Flag-name alphabet. The recognizer records the literal verbatim, so
#: any non-empty string suffices; constraining to identifier characters
#: keeps fixtures readable and avoids accidental collision with the
#: ``<dynamic>`` placeholder token.
_FLAG_NAME_CHARS: Final[str] = string.ascii_letters + string.digits + "_-"


#: Canonical category emitted by the recognizer for every input it
#: produces.
_CLI_ARGUMENT: Final[AbstractInputCategory] = (
    AbstractInputCategory.CLI_ARGUMENT
)


#: Placeholder substituted for non-literal flag-name arguments
#: (Requirement 7.4 second sentence).
_DYNAMIC_FLAG_NAME: Final[str] = "<dynamic>"


#: Library tag included in every flag-call description (Requirement
#: 7.4 first clause).
_FLAG_LIBRARY_TOKEN: Final[str] = "flag"


#: Method name reserved for the no-flag-name special case.
_FLAG_PARSE_METHOD: Final[str] = "Parse"


#: Named ``flag.<method>`` registrations whose flag name is the first
#: positional argument. Mirrors the recognizer's
#: ``_FLAG_NAME_ARG_INDEX`` for indices equal to ``0``.
_FLAG_NAME_AT_ARG_0: Final[tuple[str, ...]] = (
    "String",
    "Int",
    "Bool",
    "Float64",
    "Duration",
    "NewFlagSet",
)


#: Named ``flag.<method>`` registrations whose flag name is the second
#: positional argument (after the pointer-receiver argument). Mirrors
#: the recognizer's ``_FLAG_NAME_ARG_INDEX`` for indices equal to ``1``.
_FLAG_NAME_AT_ARG_1: Final[tuple[str, ...]] = (
    "StringVar",
    "IntVar",
    "BoolVar",
    "Float64Var",
    "DurationVar",
)


#: Full set of recognized ``flag.<method>`` names. Drawn from for
#: noise events that should never emit because they appear under a
#: non-``flag`` receiver chain.
_ALL_FLAG_METHODS: Final[tuple[str, ...]] = (
    *_FLAG_NAME_AT_ARG_0,
    *_FLAG_NAME_AT_ARG_1,
    _FLAG_PARSE_METHOD,
)


#: A small set of fx / viper method names sufficient to demonstrate
#: that the recognizer's dispatch-boundary exclusion holds. Includes
#: method names that overlap the recognized ``flag.<method>``
#: vocabulary so a regression that forgets to honor the receiver-chain
#: exclusion would surface immediately.
_FX_METHODS: Final[tuple[str, ...]] = (
    "Provide",
    "Invoke",
    "Module",
    "New",
    "Annotate",
    "WithLogger",
    "String",  # overlaps flag.String — must be excluded regardless
    "Parse",   # overlaps flag.Parse — must be excluded regardless
)
_VIPER_METHODS: Final[tuple[str, ...]] = (
    "New",
    "GetString",
    "BindEnv",
    "SetConfigName",
    "AddConfigPath",
    "String",  # overlaps flag.String — must be excluded regardless
)


#: Import-path prefix that flips a file into fx-aware mode for the
#: ``.Append`` exclusion (Requirements 12.1, 12.3, design §"go.go_io"
#: section 7).
_FX_IMPORT_PREFIX: Final[str] = "go.uber.org/fx"


#: The canonical function name a CLI entry-point file declares.
_MAIN_FUNCTION_NAME: Final[str] = "main"


#: File-path shape tags used by the per-file generator. Each shape
#: exercises one branch of :func:`_binary_name_for_path`:
_PATH_SHAPE_CMD_NAMED: Final[str] = "cmd_named"      # cmd/<name>/main.go
_PATH_SHAPE_CMD_MAIN: Final[str] = "cmd_main"        # cmd/main.go
_PATH_SHAPE_ROOT_MAIN: Final[str] = "root_main"      # main.go
_PATH_SHAPE_TOO_DEEP: Final[str] = "too_deep"        # cmd/<a>/<b>/main.go
_PATH_SHAPE_INTERNAL: Final[str] = "internal"        # internal/<x>.go
_PATH_SHAPES: Final[tuple[str, ...]] = (
    _PATH_SHAPE_CMD_NAMED,
    _PATH_SHAPE_CMD_MAIN,
    _PATH_SHAPE_ROOT_MAIN,
    _PATH_SHAPE_TOO_DEEP,
    _PATH_SHAPE_INTERNAL,
)


# ---------------------------------------------------------------------------
# Module-path helpers
# ---------------------------------------------------------------------------


def _expected_module_binary_name(gomod_text: str | None) -> str | None:
    """Mirror the recognizer's ``_module_binary_name`` derivation.

    Reads ``go.mod`` content (when present), extracts the ``module``
    line via the same line-oriented regex shape :func:`parse_go_mod`
    uses, applies the ``<host>/<org>/`` prefix-stripping rule for
    paths with three or more slash-separated segments, and returns
    the last remaining segment.

    Returns ``None`` when ``gomod_text`` is ``None``, when the file
    contains no well-formed ``module`` line, or when stripping
    produces an empty result.
    """

    if gomod_text is None:
        return None

    # Mirror parse_go_mod's first-match behavior.
    for line in gomod_text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("module "):
            continue
        # Pull out the module path after the ``module`` keyword, before
        # any trailing comment.
        body = stripped[len("module "):].strip()
        # Drop the trailing ``//`` comment when present.
        comment_idx = body.find("//")
        if comment_idx != -1:
            body = body[:comment_idx].strip()
        if not body:
            return None
        module_path = body.split()[0]
        parts = module_path.split("/")
        if len(parts) >= 3:
            parts = parts[2:]
        if not parts:
            return None
        last = parts[-1]
        return last if last else None
    return None


# ---------------------------------------------------------------------------
# Per-event strategies
# ---------------------------------------------------------------------------


_ident = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8).filter(
    # Reject leading hyphens — those would produce paths the OS treats
    # as flags, and they never appear in any sample repository.
    lambda s: not s.startswith("-"),
)


_segment = st.text(alphabet=_SEGMENT_CHARS, min_size=1, max_size=10)


_flag_name_literal = st.text(
    alphabet=_FLAG_NAME_CHARS,
    min_size=1,
    max_size=12,
)


@st.composite
def _gomod_text(draw: st.DrawFn) -> str | None:
    """Generate a ``go.mod`` body, or ``None`` to omit the manifest.

    Three branches sampled at equal frequency:

    * ``None`` — no ``go.mod`` in the repository. The
      ``cmd/main.go`` and root ``main.go`` branches contribute no
      binary input under this case (Requirements 7.2, 7.3 both
      require a manifest).
    * Bare module name (one segment, no host prefix) — survives
      stripping unchanged. Mirrors the four sample repositories'
      module paths (``repayment_service``, ``cat-service``,
      ``aps_los_vtiger``, ``fec_pool_service``).
    * Three-or-more-segment module path — exercises the
      ``<host>/<org>/`` stripping rule. The last segment alone
      becomes the binary name.
    """

    shape = draw(st.sampled_from(("absent", "bare", "prefixed")))
    if shape == "absent":
        return None
    if shape == "bare":
        bare = draw(_ident)
        return f"module {bare}\n"
    # Three-or-more-segment path. The total segment count is drawn
    # from {3, 4} so both the canonical ``<host>/<org>/<name>``
    # shape and a deeper ``<host>/<org>/<group>/<name>`` shape are
    # exercised.
    n_segments = draw(st.integers(min_value=3, max_value=4))
    segments = draw(
        st.lists(_segment, min_size=n_segments, max_size=n_segments),
    )
    return f"module {'/'.join(segments)}\n"


@st.composite
def _file_path_and_funcdecl(
    draw: st.DrawFn,
) -> tuple[str, str, FuncDeclEvent | None]:
    """Generate a file path, its shape tag, and an optional ``func main()``.

    Returns ``(path, shape, funcdecl)`` where ``funcdecl`` is a
    :class:`FuncDeclEvent` for ``func main()`` at a random 1-indexed
    line, or ``None`` when the file does not declare ``main``.

    Path-shape branches:

    * ``cmd/<name>/main.go`` — emits a binary input when
      ``funcdecl`` is present (Requirement 7.1).
    * ``cmd/main.go`` — emits a binary input when ``funcdecl`` is
      present AND ``go.mod`` is present (Requirement 7.2).
    * ``main.go`` (root) — emits a binary input when ``funcdecl`` is
      present AND ``go.mod`` is present (Requirement 7.3).
    * ``cmd/<a>/<b>/main.go`` — too deep; never emits a binary input.
    * ``internal/<x>.go`` — wrong filename; never emits a binary input.

    The path uniqueness across files in a multi-file repository is
    enforced by the caller; this helper generates one shape at a
    time.
    """

    shape = draw(st.sampled_from(_PATH_SHAPES))
    if shape == _PATH_SHAPE_CMD_NAMED:
        name = draw(_ident)
        path = f"cmd/{name}/main.go"
    elif shape == _PATH_SHAPE_CMD_MAIN:
        path = "cmd/main.go"
    elif shape == _PATH_SHAPE_ROOT_MAIN:
        path = "main.go"
    elif shape == _PATH_SHAPE_TOO_DEEP:
        a = draw(_ident)
        b = draw(_ident)
        path = f"cmd/{a}/{b}/main.go"
    else:  # _PATH_SHAPE_INTERNAL
        name = draw(_ident)
        path = f"internal/{name}.go"

    # Whether the file declares ``func main()``. Sampled at moderate
    # frequency so both the "main present" and "main absent" branches
    # are exercised across the generator's examples.
    has_main = draw(st.booleans())
    if has_main:
        line = draw(st.integers(min_value=1, max_value=999))
        funcdecl: FuncDeclEvent | None = FuncDeclEvent(
            name=_MAIN_FUNCTION_NAME,
            receiver_type=None,
            file_path=path,
            line=line,
            body_token_range=(0, 0),
        )
    else:
        funcdecl = None

    return path, shape, funcdecl


@st.composite
def _flag_call_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str | None]:
    """Generate one ``flag.<method>(...)`` call event.

    Returns ``(event, expected_flag_name)`` where ``expected_flag_name``
    is:

    * the literal string when the flag-name argument is a
      :class:`StringLitArg`,
    * the :data:`_DYNAMIC_FLAG_NAME` placeholder for every other
      argument shape (including absent arguments and the wrong-index
      ``StringLitArg`` shape used by ``Var`` registrations),
    * ``None`` for ``flag.Parse()`` (no flag name to record).

    The expected-output computation uses ``expected_flag_name`` to
    predict the recognizer's emission without re-deriving the
    f-string template.
    """

    method = draw(st.sampled_from(_ALL_FLAG_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))

    if method == _FLAG_PARSE_METHOD:
        # flag.Parse() takes no positional arguments. Generate the
        # canonical zero-arity shape.
        args: tuple[ArgRef, ...] = ()
        return (
            MethodCallEvent(
                receiver_chain=("flag",),
                method_name=method,
                args=args,
                file_path=file_path,
                line=line,
            ),
            None,
        )

    is_var = method in _FLAG_NAME_AT_ARG_1
    name_arg_kind = draw(
        st.sampled_from(("string", "ident", "dotted", "number", "missing", "unknown")),
    )

    expected_flag_name: str | None
    if name_arg_kind == "string":
        flag_name = draw(_flag_name_literal)
        if is_var:
            # StringVar(&v, "name", "default", "usage") — flag name is
            # at index 1, so a pointer-like first argument plus the
            # literal at index 1 is the canonical shape.
            args = (
                IdentArg(name=draw(_ident)),
                StringLitArg(value=flag_name),
            )
        else:
            # String("name", "default", "usage") — flag name is at
            # index 0.
            args = (StringLitArg(value=flag_name),)
        expected_flag_name = flag_name
    elif name_arg_kind == "ident":
        if is_var:
            args = (IdentArg(name=draw(_ident)), IdentArg(name=draw(_ident)))
        else:
            args = (IdentArg(name=draw(_ident)),)
        expected_flag_name = _DYNAMIC_FLAG_NAME
    elif name_arg_kind == "dotted":
        dotted = draw(
            st.lists(_ident, min_size=2, max_size=3),
        )
        if is_var:
            args = (
                IdentArg(name=draw(_ident)),
                DottedArg(parts=tuple(dotted)),
            )
        else:
            args = (DottedArg(parts=tuple(dotted)),)
        expected_flag_name = _DYNAMIC_FLAG_NAME
    elif name_arg_kind == "number":
        if is_var:
            args = (
                IdentArg(name=draw(_ident)),
                NumberLitArg(text=draw(st.text(alphabet=string.digits, min_size=1, max_size=3))),
            )
        else:
            args = (NumberLitArg(text=draw(st.text(alphabet=string.digits, min_size=1, max_size=3))),)
        expected_flag_name = _DYNAMIC_FLAG_NAME
    elif name_arg_kind == "unknown":
        if is_var:
            args = (
                IdentArg(name=draw(_ident)),
                UnknownArg(text=draw(_flag_name_literal)),
            )
        else:
            args = (UnknownArg(text=draw(_flag_name_literal)),)
        expected_flag_name = _DYNAMIC_FLAG_NAME
    else:  # "missing"
        # Zero or (for Var) one positional argument — the flag-name
        # slot is absent from the source. The recognizer must record
        # ``<dynamic>`` rather than crash.
        if is_var:
            args = (IdentArg(name=draw(_ident)),)
        else:
            args = ()
        expected_flag_name = _DYNAMIC_FLAG_NAME

    event = MethodCallEvent(
        receiver_chain=("flag",),
        method_name=method,
        args=args,
        file_path=file_path,
        line=line,
    )
    return event, expected_flag_name


@st.composite
def _fx_or_viper_noise_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a noise event excluded at the dispatch boundary.

    The recognizer must treat any ``MethodCallEvent`` whose receiver
    chain begins with ``fx`` or ``viper`` as inert (Requirements
    12.1, 12.3, 13.1, 13.2). The method-name selection deliberately
    overlaps with the recognized ``flag.<method>`` vocabulary
    (``String``, ``Parse``) so a regression that forgets to honor
    the receiver-chain exclusion would surface immediately: the
    noise event would otherwise match the recognizer's method-name
    gate and emit a spurious input.
    """

    receiver = draw(st.sampled_from(("fx", "viper")))
    method = (
        draw(st.sampled_from(_FX_METHODS))
        if receiver == "fx"
        else draw(st.sampled_from(_VIPER_METHODS))
    )
    line = draw(st.integers(min_value=1, max_value=999))
    # Provide a string-literal argument that, if it were ever surfaced
    # in the output, would be observable; the property's equality
    # assertion implicitly verifies the string never appears.
    arg_text = draw(_flag_name_literal)
    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name=method,
        args=(StringLitArg(value=arg_text),),
        file_path=file_path,
        line=line,
    )


@st.composite
def _lc_append_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a ``<id>.Append(...)`` call on a likely ``fx.Lifecycle``.

    The recognizer drops these calls when the file imports any
    ``go.uber.org/fx`` submodule (Requirement 7.5 second sentence:
    "any method call on ``fx.Lifecycle``"). The generator always
    produces a single-identifier receiver chain so the file-level
    fx-import flag is the load-bearing discriminator.

    ``Append`` is not in the flag method set, so the recognizer
    would not emit an input from this call shape regardless of the
    exclusion. The exclusion is encoded for defense-in-depth and
    surfaced through the structural-invariants test below.
    """

    receiver = draw(_ident)
    line = draw(st.integers(min_value=1, max_value=999))
    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name="Append",
        args=(IdentArg(name=draw(_ident)),),
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
    shape: str,
    funcdecl: FuncDeclEvent | None,
) -> tuple[
    list["GoEvent"],
    bool,
    list[tuple[MethodCallEvent, str | None]],
]:
    """Generate one file's event list plus per-recognizer expectations.

    Returns ``(events, file_imports_fx, flag_calls)``:

    * ``events`` is the order-preserving event list assigned to the
      file's key in the ``events_by_file`` mapping. Always carries
      one or two noise events plus the optional ``func main()``
      declaration so the exclusion property is exercised on every
      example.
    * ``file_imports_fx`` records whether the file's event list
      contains an :class:`ImportEvent` whose path begins with
      ``go.uber.org/fx``. The recognizer's ``.Append`` exclusion is
      gated on this flag.
    * ``flag_calls`` is the per-call record list needed to predict
      the recognizer's emissions. Event order matches insertion
      into ``events``.
    """

    events: list[GoEvent] = []
    file_imports_fx = draw(st.booleans())

    if file_imports_fx:
        # The exact subpath does not matter; any path beginning with
        # ``go.uber.org/fx`` flips the recognizer into fx-aware mode.
        events.append(
            ImportEvent(
                path=_FX_IMPORT_PREFIX,
                alias=None,
                file_path=path,
                line=1,
            ),
        )

    # The ``func main()`` declaration, when generated, is placed
    # early in the event list so it interacts with any flag calls
    # that follow. The recognizer's per-event walk is order-
    # independent for emission (it scans for the main decl
    # separately from the per-event loop), but matching realistic
    # source order keeps fixtures readable.
    if funcdecl is not None:
        events.append(funcdecl)

    # One leading noise event so the exclusion is exercised even
    # for files with zero recognized calls.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    # Per-file flag calls.
    flag_calls = draw(
        st.lists(
            _flag_call_event(file_path=path),
            min_size=0,
            max_size=3,
        ),
    )
    for ev, _ in flag_calls:
        events.append(ev)

    # One or two single-identifier ``.Append`` calls. These should
    # contribute zero emissions regardless of the file's fx-import
    # flag — ``Append`` is not in the recognized flag method set, so
    # the recognizer's response is "no input" in both cases. The
    # generator includes them so a regression that accidentally
    # added ``Append`` to the flag method set would surface
    # immediately (the expected output would still be the same).
    events.extend(
        draw(
            st.lists(
                _lc_append_event(file_path=path),
                min_size=0,
                max_size=2,
            ),
        ),
    )

    # Trailing noise.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    return events, file_imports_fx, flag_calls


@st.composite
def _events_by_file(
    draw: st.DrawFn,
) -> tuple[
    dict[str, list["GoEvent"]],
    str | None,
    dict[
        str,
        tuple[
            str,
            FuncDeclEvent | None,
            bool,
            list[tuple[MethodCallEvent, str | None]],
        ],
    ],
]:
    """Generate a multi-file events mapping plus per-file metadata.

    Returns ``(events_by_file, gomod_text, per_file_metadata)``. The
    ``per_file_metadata`` dict mirrors ``events_by_file`` keys and
    carries the four-tuple needed to compute the expected output:
    ``(shape, funcdecl, file_imports_fx, flag_calls)``.

    Path uniqueness across files is enforced by the generator: only
    one of each fixed-path shape (``cmd/main.go``, ``main.go``) is
    produced per repository, and ``cmd/<name>/main.go`` /
    ``cmd/<a>/<b>/main.go`` / ``internal/<x>.go`` files use generated
    segments drawn from a unique-set strategy.
    """

    gomod_text = draw(_gomod_text())

    # Independently generate each fixed-path branch. Each branch
    # contributes at most one file to the mapping.
    cmd_main_present = draw(st.booleans())
    root_main_present = draw(st.booleans())

    # Variable-path branches.
    cmd_named_names = draw(
        st.lists(_ident, min_size=0, max_size=3, unique=True),
    )
    too_deep_pairs = draw(
        st.lists(
            st.tuples(_ident, _ident),
            min_size=0,
            max_size=2,
            unique_by=lambda p: f"{p[0]}/{p[1]}",
        ),
    )
    internal_names = draw(
        st.lists(_ident, min_size=0, max_size=2, unique=True),
    )

    paths_and_shapes: list[tuple[str, str]] = []
    if cmd_main_present:
        paths_and_shapes.append(("cmd/main.go", _PATH_SHAPE_CMD_MAIN))
    if root_main_present:
        paths_and_shapes.append(("main.go", _PATH_SHAPE_ROOT_MAIN))
    for name in cmd_named_names:
        paths_and_shapes.append((f"cmd/{name}/main.go", _PATH_SHAPE_CMD_NAMED))
    for a, b in too_deep_pairs:
        paths_and_shapes.append((f"cmd/{a}/{b}/main.go", _PATH_SHAPE_TOO_DEEP))
    for name in internal_names:
        paths_and_shapes.append((f"internal/{name}.go", _PATH_SHAPE_INTERNAL))

    # Ensure at least one file exists so the recognizer is exercised
    # against a non-empty mapping.
    if not paths_and_shapes:
        paths_and_shapes.append(("internal/empty.go", _PATH_SHAPE_INTERNAL))

    events_by_file: dict[str, list[GoEvent]] = {}
    metadata: dict[
        str,
        tuple[
            str,
            FuncDeclEvent | None,
            bool,
            list[tuple[MethodCallEvent, str | None]],
        ],
    ] = {}

    for path, shape in paths_and_shapes:
        # Whether this file declares ``func main()`` is independent
        # of its path shape so the generator exercises both
        # "eligible path with main decl" and "eligible path without
        # main decl" branches (the latter contributes no binary
        # input even when the path matches an eligible shape).
        has_main = draw(st.booleans())
        if has_main:
            line = draw(st.integers(min_value=1, max_value=999))
            funcdecl: FuncDeclEvent | None = FuncDeclEvent(
                name=_MAIN_FUNCTION_NAME,
                receiver_type=None,
                file_path=path,
                line=line,
                body_token_range=(0, 0),
            )
        else:
            funcdecl = None

        events, file_imports_fx, flag_calls = draw(
            _file_events(path=path, shape=shape, funcdecl=funcdecl),
        )
        events_by_file[path] = events
        metadata[path] = (shape, funcdecl, file_imports_fx, flag_calls)

    return events_by_file, gomod_text, metadata


# ---------------------------------------------------------------------------
# Expected-output computation
# ---------------------------------------------------------------------------


def _expected_binary_description(
    *,
    path: str,
    shape: str,
    funcdecl: FuncDeclEvent | None,
    module_binary_name: str | None,
) -> str | None:
    """Compute the binary description for one eligible file, or ``None``.

    Mirrors :func:`_extract_cli_entry_points`'s binary branch:

    * Returns ``None`` when the file does not declare ``func main()``,
      when the path shape is ineligible, or when the eligible shape
      requires ``go.mod`` and the module-derived binary name is
      ``None``.
    * Otherwise returns the canonical ``"binary <name> via func
      main() at <file>:<line>"`` description.
    """

    if funcdecl is None:
        return None

    if shape == _PATH_SHAPE_CMD_NAMED:
        # cmd/<name>/main.go — binary name is the middle segment.
        parts = path.split("/")
        binary_name = parts[1]
        if not binary_name:
            return None
    elif shape in (_PATH_SHAPE_CMD_MAIN, _PATH_SHAPE_ROOT_MAIN):
        # cmd/main.go or main.go — binary name comes from go.mod.
        if module_binary_name is None:
            return None
        binary_name = module_binary_name
    else:
        # Too-deep or internal — never eligible.
        return None

    return (
        f"binary {binary_name} "
        f"via func main() at {funcdecl.file_path}:{funcdecl.line}"
    )


def _expected_flag_description(
    event: MethodCallEvent,
    expected_flag_name: str | None,
) -> str:
    """Compute the description for one recognized ``flag.<method>(...)`` call.

    Mirrors :func:`_try_flag_description`:

    * ``flag.Parse()`` → ``"flag parsing invoked via flag.Parse() at
      <file>:<line>"``.
    * Named registration with literal name →
      ``"flag <name> via flag.<method>() at <file>:<line>"``.
    * Named registration with non-literal or missing name →
      ``"flag <dynamic> via flag.<method>() at <file>:<line>"``.
    """

    if event.method_name == _FLAG_PARSE_METHOD:
        return (
            f"{_FLAG_LIBRARY_TOKEN} parsing invoked "
            f"via {_FLAG_LIBRARY_TOKEN}.{_FLAG_PARSE_METHOD}() "
            f"at {event.file_path}:{event.line}"
        )
    name_token = (
        expected_flag_name
        if expected_flag_name is not None
        else _DYNAMIC_FLAG_NAME
    )
    return (
        f"{_FLAG_LIBRARY_TOKEN} {name_token} "
        f"via {_FLAG_LIBRARY_TOKEN}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_inputs(
    metadata: dict[
        str,
        tuple[
            str,
            FuncDeclEvent | None,
            bool,
            list[tuple[MethodCallEvent, str | None]],
        ],
    ],
    module_binary_name: str | None,
) -> list[AbstractInput]:
    """Compute the deduplicated expected ``AbstractInput`` list.

    Iteration order matches the recognizer's: paths in sorted order,
    and within each file the binary input (when emitted) precedes the
    file's flag inputs in the order they appear in the file's event
    list.
    """

    expected: list[AbstractInput] = []
    seen: set[tuple[AbstractInputCategory, str]] = set()

    for path in sorted(metadata):
        shape, funcdecl, _file_imports_fx, flag_calls = metadata[path]

        binary_desc = _expected_binary_description(
            path=path,
            shape=shape,
            funcdecl=funcdecl,
            module_binary_name=module_binary_name,
        )
        if binary_desc is not None:
            key = (_CLI_ARGUMENT, binary_desc)
            if key not in seen:
                seen.add(key)
                expected.append(
                    AbstractInput(
                        category=_CLI_ARGUMENT,
                        description=binary_desc,
                    ),
                )

        for event, expected_flag_name in flag_calls:
            desc = _expected_flag_description(event, expected_flag_name)
            key = (_CLI_ARGUMENT, desc)
            if key in seen:
                continue
            seen.add(key)
            expected.append(
                AbstractInput(
                    category=_CLI_ARGUMENT,
                    description=desc,
                ),
            )

    return expected


def _build_repo(gomod_text: str | None) -> RepositoryContents:
    """Build the minimal :class:`RepositoryContents` consumed by the recognizer.

    Mirrors the helper used by ``test_property_04_go_purpose_priority``
    so the two CLI-and-purpose suites share the same fixture shape.
    The recognizer reads only ``go.mod`` from
    ``RepositoryContents``; all other state is carried by
    ``events_by_file``.
    """

    files: dict[str, str] = {}
    if gomod_text is not None:
        files["go.mod"] = gomod_text
    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeefcafef00d",
        files=files,
    )


def _stripped_of_fx_viper(
    events_by_file: dict[str, list["GoEvent"]],
) -> dict[str, list["GoEvent"]]:
    """Return ``events_by_file`` with every fx / viper ``MethodCallEvent`` removed.

    The Requirements 12.1, 12.3, 13.1, 13.2 exclusion property
    compares the recognizer's output before and after the removal;
    equality of the two outputs implies the recognizer never
    inspected the excluded receivers.
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


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_matches_expected_output(
    case: tuple[
        dict[str, list["GoEvent"]],
        str | None,
        dict[
            str,
            tuple[
                str,
                FuncDeclEvent | None,
                bool,
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """Property 9 main invariant: the documented `cmd/`/root and flag rules.

    Validates Requirements 7.1, 7.2, 7.3, 7.4, 7.5 jointly. The
    expected output is computed by mirroring the recognizer's
    decision tree on the per-file generator metadata, then asserting
    full-list equality. Per-emission invariants validated by
    construction:

    * Exactly one ``AbstractInput`` per ``cmd/<name>/main.go`` with
      ``func main()`` (Requirement 7.1).
    * Exactly one ``AbstractInput`` per ``cmd/main.go`` with
      ``func main()`` and ``go.mod`` present (Requirement 7.2).
    * Exactly one ``AbstractInput`` per root ``main.go`` with
      ``func main()`` and ``go.mod`` present (Requirement 7.3).
    * Exactly one ``AbstractInput`` per recognized ``flag.<X>(...)``
      call (Requirement 7.4).
    * Zero ``AbstractInput`` from any ``fx.<method>(...)`` or
      ``viper.<method>(...)`` call (Requirements 7.5, 12.1, 12.3,
      13.1, 13.2 — covered jointly by the noise events the file
      generator always emits).
    * The Source_Location suffix ``at <file>:<line>`` appears on
      every emission per Requirement 7.6.
    * The category is always ``cli_argument`` per the design's
      §"go.go_io" section 7.
    """

    events_by_file, gomod_text, metadata = case

    rc = _build_repo(gomod_text)
    module_binary_name = _expected_module_binary_name(gomod_text)

    actual = _extract_cli_entry_points(rc, events_by_file)
    expected = _expected_inputs(metadata, module_binary_name)

    assert actual == expected, (
        "CLI entry-point recognizer output diverged from expected:\n"
        f"  actual:   {actual!r}\n"
        f"  expected: {expected!r}\n"
        f"  gomod_text: {gomod_text!r}\n"
        f"  module_binary_name: {module_binary_name!r}"
    )

    # Universal invariants on the actual output, independent of the
    # expected-output computation. These would catch a regression
    # that made the recognizer's output structurally wrong even when
    # the full-list equality passed (e.g. by a coincidental two-bug
    # cancellation in the expected-output mirror).
    for entry in actual:
        assert entry.category is _CLI_ARGUMENT, (
            f"entry {entry!r} carries category {entry.category!r}; "
            f"the CLI recognizer must always emit cli_argument"
        )
        # Every description carries the Source_Location suffix
        # ``at <file>:<line>`` per Requirement 7.6.
        assert " at " in entry.description, (
            f"entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 7.6 mandates a Source_Location on "
            f"every emission"
        )
        # Every description names either a binary or a flag.
        assert (
            entry.description.startswith("binary ")
            or entry.description.startswith(f"{_FLAG_LIBRARY_TOKEN} ")
        ), (
            f"entry {entry!r} does not start with the canonical "
            f"'binary' or 'flag' prefix; the CLI recognizer's two "
            f"branches are the only sources of emissions"
        )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_ignores_fx_and_viper_receivers(
    case: tuple[
        dict[str, list["GoEvent"]],
        str | None,
        dict[
            str,
            tuple[
                str,
                FuncDeclEvent | None,
                bool,
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """fx / viper dispatch-boundary exclusion invariant.

    Validates Requirements 7.5, 12.1, 12.3, 13.1, 13.2. Running the
    recognizer over the full event stream and over the stream with
    every fx / viper ``MethodCallEvent`` removed must produce
    identical output. Equality of the two outputs implies the
    recognizer consulted no fx / viper receiver for any of its
    decisions — neither for the binary-entry-point branch (which
    walks ``FuncDeclEvent``s and is therefore inert against
    ``MethodCallEvent`` perturbations by construction) nor for the
    flag-call branch (which the exclusion is designed to gate).
    """

    events_by_file, gomod_text, _metadata = case
    rc = _build_repo(gomod_text)

    with_noise = _extract_cli_entry_points(rc, events_by_file)
    without_noise = _extract_cli_entry_points(
        rc,
        _stripped_of_fx_viper(events_by_file),
    )
    assert with_noise == without_noise, (
        "CLI recognizer output changed after removing fx/viper "
        "MethodCallEvents; the dispatch-boundary exclusion must be "
        "inert (Requirements 7.5, 12.1, 12.3, 13.1, 13.2)"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_ignores_fx_lifecycle_append(
    case: tuple[
        dict[str, list["GoEvent"]],
        str | None,
        dict[
            str,
            tuple[
                str,
                FuncDeclEvent | None,
                bool,
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """``.Append`` on an fx.Lifecycle-typed identifier emits no input.

    Validates Requirement 7.5 second sentence ("any method call on
    ``fx.Lifecycle``"). The file generator interleaves
    single-identifier ``.Append`` calls into every file's event
    list; stripping them from a file that imports
    ``go.uber.org/fx`` must not change the recognizer's output. The
    invariant is structural rather than observational because
    ``Append`` is not in the recognized flag method set — a
    regression that accidentally added it would surface as a
    diverged expected-output computation in the main equality
    property above. This property pins the dispatch-boundary
    exclusion explicitly so the design's defense-in-depth contract
    is visible at the test boundary.
    """

    events_by_file, gomod_text, _metadata = case
    rc = _build_repo(gomod_text)

    def _strip_lc_append(
        ev_map: dict[str, list["GoEvent"]],
    ) -> dict[str, list["GoEvent"]]:
        out: dict[str, list[GoEvent]] = {}
        for path, events in ev_map.items():
            kept: list[GoEvent] = []
            for event in events:
                if (
                    isinstance(event, MethodCallEvent)
                    and event.method_name == "Append"
                    and len(event.receiver_chain) == 1
                ):
                    continue
                kept.append(event)
            out[path] = kept
        return out

    with_appends = _extract_cli_entry_points(rc, events_by_file)
    without_appends = _extract_cli_entry_points(
        rc,
        _strip_lc_append(events_by_file),
    )
    assert with_appends == without_appends, (
        "CLI recognizer output changed after removing single-"
        "identifier .Append() calls; Requirement 7.5's "
        "fx.Lifecycle.Append exclusion must be inert"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_cli_entry_points_is_iteration_order_independent(
    case: tuple[
        dict[str, list["GoEvent"]],
        str | None,
        dict[
            str,
            tuple[
                str,
                FuncDeclEvent | None,
                bool,
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """Determinism invariant: output depends only on input contents.

    Validates Requirement 11.4 (cross-referenced by the design's
    "path-sorted iteration" rule for every Go recognizer). The
    recognizer sorts paths internally, so two mappings with the
    same keys and values but different insertion orders must
    produce identical output lists.
    """

    events_by_file, gomod_text, _metadata = case
    rc = _build_repo(gomod_text)

    reversed_mapping = dict(reversed(list(events_by_file.items())))
    assert _extract_cli_entry_points(
        rc, events_by_file,
    ) == _extract_cli_entry_points(rc, reversed_mapping), (
        "CLI recognizer output depends on insertion order; "
        "Requirement 11.4 mandates deterministic, path-sorted "
        "iteration"
    )


def test_extract_cli_entry_points_rejects_funcdecl_without_line() -> None:
    """Source_Location rejection: no input when the main decl's line is ``None``.

    Validates Requirement 7.6 second sentence: "IF the
    Project_Analyzer cannot determine a Source_Location (both file
    path and 1-indexed line number) for a recognized CLI
    construction, THEN THE Project_Analyzer SHALL NOT emit an
    Abstract_Input for that construction."

    The typed ``FuncDeclEvent`` dataclass uses an ``int`` line
    number and never produces ``None`` in practice; the test
    constructs a degenerate event by overwriting the ``line`` slot
    via :func:`dataclasses.replace` so the recognizer's explicit
    ``line is None`` guard is exercised end-to-end. The same path
    shape with a well-formed ``line`` would emit one binary input
    (the comparison fixture below confirms that), so the rejection
    is the load-bearing observation.
    """

    # Sanity baseline: a well-formed FuncDeclEvent in a cmd/<name>/main.go
    # file emits one binary input.
    well_formed = FuncDeclEvent(
        name=_MAIN_FUNCTION_NAME,
        receiver_type=None,
        file_path="cmd/svc/main.go",
        line=10,
        body_token_range=(0, 0),
    )
    rc = _build_repo(gomod_text=None)
    baseline = _extract_cli_entry_points(
        rc,
        {"cmd/svc/main.go": [well_formed]},
    )
    assert len(baseline) == 1, (
        f"baseline fixture should emit one binary input; got "
        f"{baseline!r}"
    )
    assert baseline[0].description.startswith("binary svc "), (
        f"baseline binary description should name 'svc'; got "
        f"{baseline[0].description!r}"
    )

    # Rejection case: overwriting ``line`` with ``None`` exercises
    # the recognizer's explicit guard.
    degenerate = replace(well_formed, line=None)  # type: ignore[arg-type]
    rejected = _extract_cli_entry_points(
        rc,
        {"cmd/svc/main.go": [degenerate]},
    )
    assert rejected == [], (
        "CLI recognizer must reject a FuncDeclEvent with line=None "
        "(Requirement 7.6); got "
        f"{rejected!r}"
    )

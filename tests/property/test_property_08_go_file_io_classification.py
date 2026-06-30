# ruff: noqa: E501
# Feature: go-analyzer-support, Property 8: File I/O classification follows the documented O_* flag mapping, including the undecidable-flag case.
"""Property test for the Go file-I/O recognizer.

**Validates Requirements 6.1, 6.2, 6.3, 6.4** (Property 8 in the
design, task 7.8 in ``tasks.md``).

The file-I/O recognizer is the fourth of five recognizers composed by
``extract_go_io`` (task 7.11). This property test exercises its
package-internal entry point :func:`_extract_file_io` directly so the
per-recognizer contract is pinned independently of the eventual
composition step.

The properties below capture the design's documented contract:

1. **One AbstractInput per recognized read call** (Requirement 6.1).
   Every ``os.Open(<path>)``, ``os.ReadFile(<path>)``, and
   ``ioutil.ReadFile(<path>)`` produces exactly one
   ``AbstractInput(category=file_read)``.

2. **One AbstractOutput per recognized write call** (Requirement 6.2).
   Every ``os.Create(<path>)``, ``os.WriteFile(<path>, ...)``, and
   ``ioutil.WriteFile(<path>, ...)`` produces exactly one
   ``AbstractOutput(category=file_written)``.

3. **`os.OpenFile` flag classification follows the documented mapping**
   (Requirement 6.3). With the canonical ``(path, flag, mode)`` arity:

   * A statically determinable single-atom flag equal to
     ``os.O_RDONLY`` → emit one ``AbstractInput(file_read)``.
   * A statically determinable single-atom flag in
     ``{os.O_WRONLY, os.O_RDWR, os.O_APPEND, os.O_CREATE, os.O_TRUNC}``
     → emit one ``AbstractOutput(file_written)``.
   * An undecidable flag expression — non-canonical arity, a
     non-``os.`` atom, an unrecognized ``os.O_*`` atom, or any
     argument shape other than ``DottedArg(("os", "O_<NAME>"))`` —
     → emit **both** one ``AbstractInput(file_read)`` and one
     ``AbstractOutput(file_written)``.

4. **Path preservation** (Requirement 6.4 + task-level path clause).
   A ``StringLitArg`` path argument's contents appear verbatim in the
   description; every other argument shape (``DottedArg``, ``IdentArg``,
   ``CallArg``, ``UnknownArg``, ``NumberLitArg``, or a missing first
   argument) is recorded as ``<dynamic>``.

5. **Source_Location is attached to every emission** (Requirement
   6.4). The recognizer encodes the ``MethodCallEvent``'s
   ``file_path`` and 1-indexed ``line`` into the description as an
   ``at <file>:<line>`` suffix, matching the HTTP, scheduler, and
   ActiveMQ recognizers' convention.

6. **fx and viper exclusions are inert at the dispatch boundary**
   (Requirements 12.1, 12.3, 13.1, 13.2). Adding arbitrary
   ``fx.<method>`` and ``viper.<method>`` calls to a file — even
   when the method name and arguments would otherwise satisfy the
   recognizer's gates — must not change the recognizer's output.

7. **Deterministic, path-sorted iteration** (Requirement 11.4). The
   recognizer is a pure function of its input mapping. Reversing the
   mapping's insertion order must not change the output lists.

The recognizer's :func:`_extract_file_io` is package-internal; the
test imports it through the module path to mirror the convention used
by ``test_property_05_go_http_detection.py`` for
:func:`_extract_http_routes`,
``test_property_06_go_scheduler_detection.py`` for
:func:`_extract_schedulers`, and
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
    MethodCallEvent,
    NumberLitArg,
    StringLitArg,
    StructLitArg,
    StructLitEvent,
    UnknownArg,
)
from project_knowledge_mcp.project_analyzer.go.go_io import _extract_file_io

if TYPE_CHECKING:
    from project_knowledge_mcp.project_analyzer.go._events import ArgRef, GoEvent


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets and shared constants
# ---------------------------------------------------------------------------


#: Path-literal alphabet. Restricted to characters that survive the
#: description's f-string interpolation unmodified — letters, digits,
#: and a few path separators commonly observed in Go file paths
#: (``/etc/config``, ``./data.json``, ``relative/file``). ``\n``,
#: ``\r``, and ``\t`` are excluded so a generated path cannot break the
#: ``at <file>:<line>`` suffix's line orientation. Spaces are excluded
#: so the verbatim-preservation property can use a strict substring
#: match without ambiguity with the description's own internal spaces.
_PATH_CHARS: Final[str] = string.ascii_letters + string.digits + "._/-"


#: Identifier alphabet for receiver names, file basenames, and dotted-
#: arg segments. Constrained to characters Go accepts in identifiers so
#: the fixtures look like real source.
_IDENT_CHARS: Final[str] = string.ascii_letters + string.digits + "_"


#: Numeric-literal alphabet for filler argument shapes (the ``mode``
#: parameter of ``os.OpenFile`` and stand-in numeric paths).
_NUMBER_CHARS: Final[str] = string.digits


#: Canonical category enum values the recognizer always emits.
_FILE_READ: Final[AbstractInputCategory] = AbstractInputCategory.FILE_READ
_FILE_WRITTEN: Final[AbstractOutputCategory] = AbstractOutputCategory.FILE_WRITTEN


#: Canonical description fragments — anchors the assertions verify on
#: every emission so a regression that drops the action token or the
#: location suffix is caught even when the full-list equality
#: assertion also fails.
_FILE_READ_PREFIX: Final[str] = "file read "
_FILE_WRITTEN_PREFIX: Final[str] = "file written "
_LOCATION_FRAGMENT: Final[str] = " at "
_DYNAMIC_PATH: Final[str] = "<dynamic>"


#: Receiver chains the recognizer matches verbatim.
_OS_CHAIN: Final[tuple[str, ...]] = ("os",)
_IOUTIL_CHAIN: Final[tuple[str, ...]] = ("ioutil",)


#: Method names that unconditionally classify the call as a read or as
#: a write (Requirements 6.1, 6.2). Sets mirror the recognizer's own
#: constants verbatim — keeping the two in lockstep is part of what
#: the property test pins.
_OS_READ_METHODS: Final[tuple[str, ...]] = ("Open", "ReadFile")
_OS_WRITE_METHODS: Final[tuple[str, ...]] = ("Create", "WriteFile")
_IOUTIL_READ_METHODS: Final[tuple[str, ...]] = ("ReadFile",)
_IOUTIL_WRITE_METHODS: Final[tuple[str, ...]] = ("WriteFile",)


#: The ``OpenFile`` method name — its read/write classification depends
#: on the flag-bitmask argument (Requirement 6.3).
_OPENFILE_METHOD: Final[str] = "OpenFile"


#: Flag atom name that classifies an ``os.OpenFile`` call as a
#: read-only access (Requirement 6.1 third bullet).
_O_RDONLY: Final[str] = "O_RDONLY"


#: Flag atom names that classify an ``os.OpenFile`` call as a write
#: access (Requirement 6.2 third bullet).
_O_WRITE_ATOMS: Final[tuple[str, ...]] = (
    "O_WRONLY",
    "O_RDWR",
    "O_APPEND",
    "O_CREATE",
    "O_TRUNC",
)


#: A small set of fx / viper method names sufficient to demonstrate
#: that the recognizer's dispatch-boundary exclusion holds. Method
#: names overlap with the recognized vocabulary (``Open``, ``ReadFile``,
#: ``Create``, ``WriteFile``, ``OpenFile``) so a regression that
#: forgets to honor the receiver-chain exclusion would surface
#: immediately: the noise event would otherwise match the recognizer's
#: method-name gate.
_FX_METHODS: Final[tuple[str, ...]] = (
    "Provide",
    "Invoke",
    "Module",
    "New",
    "Open",
    "ReadFile",
    "Create",
    "WriteFile",
    "OpenFile",
)
_VIPER_METHODS: Final[tuple[str, ...]] = (
    "New",
    "GetString",
    "BindEnv",
    "Open",
    "ReadFile",
    "Create",
    "WriteFile",
    "OpenFile",
)


# ---------------------------------------------------------------------------
# Argument generators
# ---------------------------------------------------------------------------


_ident = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8)
_path_literal = st.text(alphabet=_PATH_CHARS, min_size=1, max_size=20)
_number_literal = st.text(alphabet=_NUMBER_CHARS, min_size=1, max_size=4)
_unknown_text = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=10)
_dotted_segments = st.lists(_ident, min_size=2, max_size=4).map(tuple)


@st.composite
def _path_argument(draw: st.DrawFn) -> tuple["ArgRef | None", str]:
    """Generate a path-argument value plus the expected description fragment.

    Returns ``(value_or_none, expected_path)``:

    * ``value_or_none`` is the ``ArgRef`` to attach as the call's first
      positional argument, or ``None`` to omit the argument entirely
      (testing the "missing first argument" branch of Requirement 6.4 /
      the recognizer's :func:`_extract_file_path_argument` fallback).
    * ``expected_path`` is the substring the recognizer must include in
      the description — the verbatim literal for ``StringLitArg``, the
      canonical ``<dynamic>`` placeholder for every other variant
      (field access, identifier, nested call, opaque text, numeric
      literal, struct literal, absent argument).
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
        value = draw(_path_literal)
        return StringLitArg(value=value), value
    if kind == "dotted":
        return DottedArg(parts=draw(_dotted_segments)), _DYNAMIC_PATH
    if kind == "ident":
        return IdentArg(name=draw(_ident)), _DYNAMIC_PATH
    if kind == "call":
        # A nested method call standing in for ``filepath.Join(...)`` or
        # similar. The recognizer never inspects the inner call's
        # identity for the path argument; the inner call's shape is
        # uninteresting.
        nested = MethodCallEvent(
            receiver_chain=(draw(_ident),),
            method_name=draw(_ident),
            args=(),
            file_path="dummy.go",
            line=1,
        )
        return CallArg(call=nested), _DYNAMIC_PATH
    if kind == "unknown":
        return UnknownArg(text=draw(_unknown_text)), _DYNAMIC_PATH
    if kind == "number":
        # A numeric path is observationally absurd but the recognizer's
        # classification still applies: any non-string literal is
        # recorded as ``<dynamic>``. Generating this branch guards
        # against a regression that only forwarded ``StringLitArg`` vs
        # ``DottedArg`` without considering other arg variants.
        return NumberLitArg(text=draw(_number_literal)), _DYNAMIC_PATH
    if kind == "struct":
        # A struct literal stands in for an unusual but recognized
        # ``ArgRef`` variant the recognizer might encounter. The
        # recognizer's path-argument helper inspects only the variant
        # tag (``isinstance``-based dispatch), so this case is
        # observationally equivalent to the other non-literal branches.
        struct_event = StructLitEvent(
            type_name="X",
            package_alias=None,
            fields=(),
            is_pointer=False,
            file_path="dummy.go",
            line=1,
        )
        return StructLitArg(event=struct_event), _DYNAMIC_PATH
    # ``absent``: the call has no positional arguments at all.
    return None, _DYNAMIC_PATH


@st.composite
def _openfile_flag_argument(
    draw: st.DrawFn,
) -> tuple["ArgRef", str]:
    """Generate a flag argument for ``os.OpenFile`` plus its expected verdict.

    Returns ``(value, verdict)`` where verdict is one of
    ``{"read", "write", "both"}``.

    The branches mirror the recognizer's :func:`_classify_openfile_flag`
    decision tree exactly:

    * ``os.O_RDONLY`` (``DottedArg(("os", "O_RDONLY"))``) → ``"read"``.
    * ``os.O_<WRITE_ATOM>`` for any of the five write atoms → ``"write"``.
    * ``os.O_<UNRECOGNIZED>`` (e.g. ``O_EXCL``, ``O_SYNC``,
      ``O_NONBLOCK``) → ``"both"``. Requirement 6.3 enumerates only six
      atoms and is silent on the rest, so the recognizer defers to
      "undecidable" — this branch pins that conservative outcome.
    * A ``DottedArg`` whose package alias is not ``os`` (e.g.
      ``flags.RDONLY``) → ``"both"``. Aliased ``os`` imports are not
      tracked by the parser, so any non-``os.`` prefix routes to the
      undecidable branch.
    * A ``DottedArg`` with more than two segments
      (``os.x.O_RDONLY``) → ``"both"``. The recognizer's shape check
      rejects anything that does not match the canonical
      ``("os", "O_<NAME>")`` length-two form.
    * Any non-``DottedArg`` shape — ``IdentArg`` (a Go-level variable
      holding a precomputed bitmask), ``NumberLitArg`` (a raw integer
      flag), ``CallArg`` (a function call), ``StringLitArg`` (an
      observationally absurd but structurally possible shape),
      ``UnknownArg`` (the parser's fallback for unparseable
      expressions, which is where binary-OR expressions land) →
      ``"both"``.
    """

    kind = draw(
        st.sampled_from(
            (
                "rdonly",
                "write_atom",
                "unrecognized_os_atom",
                "wrong_package",
                "three_segments",
                "ident",
                "number",
                "call",
                "string",
                "unknown",
            ),
        ),
    )
    if kind == "rdonly":
        return DottedArg(parts=(_OS_CHAIN[0], _O_RDONLY)), "read"
    if kind == "write_atom":
        atom = draw(st.sampled_from(_O_WRITE_ATOMS))
        return DottedArg(parts=(_OS_CHAIN[0], atom)), "write"
    if kind == "unrecognized_os_atom":
        # Atom names that are spelled ``O_*`` and live on the ``os``
        # package but are *not* in the closed read/write set. The
        # recognizer must treat these as undecidable so a closed-set
        # regression that accidentally widened the read/write atoms
        # would surface.
        atom = draw(
            st.sampled_from(("O_EXCL", "O_SYNC", "O_NONBLOCK", "O_DIRECT")),
        )
        return DottedArg(parts=(_OS_CHAIN[0], atom)), "both"
    if kind == "wrong_package":
        # A DottedArg whose package alias is not ``os``. The recognizer
        # cannot statically determine the atom's semantics without
        # resolving the aliased import, so it defers to undecidable.
        alias = draw(_ident.filter(lambda n: n != _OS_CHAIN[0]))
        atom = draw(st.sampled_from((_O_RDONLY, *_O_WRITE_ATOMS)))
        return DottedArg(parts=(alias, atom)), "both"
    if kind == "three_segments":
        # A three-segment dotted chain. The recognizer's shape check
        # rejects anything but ``("os", "O_<NAME>")``.
        segments = (
            _OS_CHAIN[0],
            draw(_ident),
            _O_RDONLY,
        )
        return DottedArg(parts=segments), "both"
    if kind == "ident":
        return IdentArg(name=draw(_ident)), "both"
    if kind == "number":
        return NumberLitArg(text=draw(_number_literal)), "both"
    if kind == "call":
        nested = MethodCallEvent(
            receiver_chain=(draw(_ident),),
            method_name=draw(_ident),
            args=(),
            file_path="dummy.go",
            line=1,
        )
        return CallArg(call=nested), "both"
    if kind == "string":
        return StringLitArg(value=draw(_path_literal)), "both"
    # ``unknown``: covers the parser's fallback for binary-OR flag
    # expressions like ``os.O_RDWR | os.O_CREATE``, which the design
    # documents as routing to the "both" branch.
    return UnknownArg(text=draw(_unknown_text)), "both"


# ---------------------------------------------------------------------------
# Per-call generators
# ---------------------------------------------------------------------------


# The ``_OpenFileShape`` enum tags the canonical and adversarial
# ``os.OpenFile`` argument-list shapes the recognizer must handle.
_CANONICAL_OPENFILE_ARITY: Final[int] = 3
_OPENFILE_CANONICAL_SHAPE: Final[str] = "canonical"  # (path, flag, mode)
_OPENFILE_EXTRA_ARGS_SHAPE: Final[str] = "extra_args"  # arity > 3 (binary-OR split)
_OPENFILE_TWO_ARGS_SHAPE: Final[str] = "two_args"  # arity = 2 (missing mode)
_OPENFILE_ONE_ARG_SHAPE: Final[str] = "one_arg"  # arity = 1 (missing flag + mode)


@st.composite
def _file_io_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate one recognized file-I/O call event and its expectation.

    Returns ``(event, verdict, expected_path)``:

    * ``verdict`` is one of ``{"read", "write", "both"}`` — what the
      recognizer must classify the call as.
    * ``expected_path`` is the substring the recognizer must include
      in the description — the verbatim literal for ``StringLitArg``
      paths, ``<dynamic>`` for every other shape.

    The generator stratifies over the recognizer's full decision space:

    1. ``os.Open`` / ``os.ReadFile`` → unconditional read.
    2. ``os.Create`` / ``os.WriteFile`` → unconditional write.
    3. ``ioutil.ReadFile`` → unconditional read.
    4. ``ioutil.WriteFile`` → unconditional write.
    5. ``os.OpenFile`` with canonical (path, flag, mode) arity →
       classification depends on the flag argument's shape.
    6. ``os.OpenFile`` with non-canonical arity →
       always undecidable ("both").
    """

    kind = draw(
        st.sampled_from(
            (
                "os_read",
                "os_write",
                "ioutil_read",
                "ioutil_write",
                "openfile_canonical",
                "openfile_extra",
                "openfile_two",
                "openfile_one",
            ),
        ),
    )

    line = draw(st.integers(min_value=1, max_value=999))
    path_value, expected_path = draw(_path_argument())

    if kind == "os_read":
        method = draw(st.sampled_from(_OS_READ_METHODS))
        args: tuple[ArgRef, ...] = () if path_value is None else (path_value,)
        return (
            MethodCallEvent(
                receiver_chain=_OS_CHAIN,
                method_name=method,
                args=args,
                file_path=file_path,
                line=line,
            ),
            "read",
            expected_path,
        )
    if kind == "os_write":
        method = draw(st.sampled_from(_OS_WRITE_METHODS))
        args = () if path_value is None else (path_value,)
        return (
            MethodCallEvent(
                receiver_chain=_OS_CHAIN,
                method_name=method,
                args=args,
                file_path=file_path,
                line=line,
            ),
            "write",
            expected_path,
        )
    if kind == "ioutil_read":
        method = draw(st.sampled_from(_IOUTIL_READ_METHODS))
        args = () if path_value is None else (path_value,)
        return (
            MethodCallEvent(
                receiver_chain=_IOUTIL_CHAIN,
                method_name=method,
                args=args,
                file_path=file_path,
                line=line,
            ),
            "read",
            expected_path,
        )
    if kind == "ioutil_write":
        method = draw(st.sampled_from(_IOUTIL_WRITE_METHODS))
        args = () if path_value is None else (path_value,)
        return (
            MethodCallEvent(
                receiver_chain=_IOUTIL_CHAIN,
                method_name=method,
                args=args,
                file_path=file_path,
                line=line,
            ),
            "write",
            expected_path,
        )
    if kind == "openfile_canonical":
        # Canonical arity: path + flag + mode. The flag argument drives
        # the verdict; the path argument is independently classified.
        # A missing path argument cannot reach this branch because the
        # canonical arity requires three positional args; substitute a
        # ``<dynamic>``-classified ``UnknownArg`` so the arity stays at
        # three while the path remains non-literal.
        flag_arg, verdict = draw(_openfile_flag_argument())
        mode_arg = NumberLitArg(text=draw(_number_literal))
        if path_value is None:
            actual_path_arg: ArgRef = UnknownArg(text="path_var")
        else:
            actual_path_arg = path_value
        args = (actual_path_arg, flag_arg, mode_arg)
        return (
            MethodCallEvent(
                receiver_chain=_OS_CHAIN,
                method_name=_OPENFILE_METHOD,
                args=args,
                file_path=file_path,
                line=line,
            ),
            verdict,
            expected_path,
        )
    if kind == "openfile_extra":
        # Non-canonical arity > 3, which is the parser's natural shape
        # for binary-OR flag expressions split across argument slots
        # (e.g. ``os.O_RDWR | os.O_CREATE`` becomes a ``DottedArg``
        # followed by an ``UnknownArg``). The recognizer routes any
        # non-canonical arity to the "both" branch regardless of what
        # the individual arguments look like.
        if path_value is None:
            actual_path_arg = UnknownArg(text="path_var")
        else:
            actual_path_arg = path_value
        # Inject a valid-looking flag fragment and a couple of trailing
        # filler arguments so the recognizer's shape check cannot
        # accidentally accept the call by ignoring the extra slots.
        extra_args = (
            DottedArg(parts=(_OS_CHAIN[0], _O_RDONLY)),
            UnknownArg(text="| os . O_CREATE"),
            NumberLitArg(text=draw(_number_literal)),
        )
        args = (actual_path_arg, *extra_args)
        return (
            MethodCallEvent(
                receiver_chain=_OS_CHAIN,
                method_name=_OPENFILE_METHOD,
                args=args,
                file_path=file_path,
                line=line,
            ),
            "both",
            expected_path,
        )
    if kind == "openfile_two":
        # Arity 2 — flag present but mode missing. Still non-canonical
        # arity so the recognizer routes to "both".
        if path_value is None:
            actual_path_arg = UnknownArg(text="path_var")
        else:
            actual_path_arg = path_value
        flag_arg, _ = draw(_openfile_flag_argument())
        args = (actual_path_arg, flag_arg)
        return (
            MethodCallEvent(
                receiver_chain=_OS_CHAIN,
                method_name=_OPENFILE_METHOD,
                args=args,
                file_path=file_path,
                line=line,
            ),
            "both",
            expected_path,
        )
    # ``openfile_one``: only the path argument is present (malformed
    # Go, but a defensible recognizer guards against it).
    if path_value is None:
        # No arguments at all. Both path and flag are missing.
        args = ()
        expected_path = _DYNAMIC_PATH
    else:
        args = (path_value,)
    return (
        MethodCallEvent(
            receiver_chain=_OS_CHAIN,
            method_name=_OPENFILE_METHOD,
            args=args,
            file_path=file_path,
            line=line,
        ),
        "both",
        expected_path,
    )


@st.composite
def _unrelated_method_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a method call with a non-file-I/O method name.

    Covers the broad "method name does not match" rejection branch.
    Concrete method-name choices overlap with other Go recognizers'
    vocabularies (``HandleFunc`` / ``AddFunc`` / ``Subscribe``) so a
    regression that accidentally widened the file-I/O method set
    would immediately fire here. Receiver chains span ``os``,
    ``ioutil``, and unrelated packages so the rejection covers all
    three branches in :func:`_classify_file_io_event`.
    """

    chain_choice = draw(st.sampled_from(("os", "ioutil", "other")))
    if chain_choice == "os":
        chain: tuple[str, ...] = _OS_CHAIN
    elif chain_choice == "ioutil":
        chain = _IOUTIL_CHAIN
    else:
        chain = (draw(_ident),)
    method = draw(
        st.sampled_from(
            (
                "HandleFunc",
                "Handle",
                "AddFunc",
                "AddJob",
                "Subscribe",
                "SendMessage",
                "Getenv",
                "Mkdir",
                "Stat",
            ),
        ),
    )
    line = draw(st.integers(min_value=1, max_value=999))
    return MethodCallEvent(
        receiver_chain=chain,
        method_name=method,
        # A string-literal first argument that looks path-shaped, so a
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
    """Generate a noise event excluded by the fx / viper dispatch-boundary skip.

    The recognizer must treat these as inert (Requirements 12.1, 12.3,
    13.1, 13.2). The noise's method-name selection deliberately
    overlaps with the recognized file-I/O vocabulary (``Open``,
    ``ReadFile``, ``Create``, ``WriteFile``, ``OpenFile``) so a
    regression that forgets the receiver-chain exclusion would
    surface immediately: the noise event would otherwise match the
    recognizer's method-name gate. The argument list is shaped like a
    valid ``OpenFile`` call so that a regression that fell through
    would also satisfy the arity and flag-shape checks.
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
            StringLitArg(value="/regression-canary"),
            DottedArg(parts=(_OS_CHAIN[0], _O_RDONLY)),
            NumberLitArg(text="0644"),
        ),
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
    list[tuple[MethodCallEvent, str, str]],
]:
    """Generate one file's event list plus the per-recognizer expectations.

    Returns ``(events, file_io_calls)``:

    * ``events`` is the order-preserving event list assigned to the
      file's key in the ``events_by_file`` mapping. Always carries
      one or two fx / viper noise events and at least one unrelated-
      method event so the exclusion and rejection branches are
      exercised on every example.
    * ``file_io_calls`` is the per-call metadata for recognized
      file-I/O events: ``(event, verdict, expected_path)`` per
      generated call. Each entry's expected emission is derived from
      this tuple in :func:`_expected_inputs_and_outputs`.
    """

    events: list[GoEvent] = []

    # Leading noise: dispatch-boundary fx / viper skip exercised on
    # every example.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    # Unrelated-method noise — must emit nothing.
    unrelated = draw(
        st.lists(_unrelated_method_event(file_path=path), min_size=0, max_size=2),
    )
    events.extend(unrelated)

    # Recognized file-I/O calls. Each call's expected metadata is
    # captured so the assertion phase can compute the predicted
    # emission per call shape.
    file_io_calls = draw(
        st.lists(_file_io_call(file_path=path), min_size=0, max_size=4),
    )
    for event, _, _ in file_io_calls:
        events.append(event)

    # Trailing noise to guard against position-sensitivity regressions.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    return events, file_io_calls


@st.composite
def _events_by_file(
    draw: st.DrawFn,
) -> tuple[
    dict[str, list["GoEvent"]],
    dict[str, list[tuple[MethodCallEvent, str, str]]],
]:
    """Generate a multi-file events mapping plus per-file expectation metadata.

    Returns ``(events_by_file, per_file_metadata)``. The
    ``per_file_metadata`` dict mirrors ``events_by_file`` keys and
    carries the recognized file-I/O calls needed to compute the
    expected recognizer output.
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
    metadata: dict[str, list[tuple[MethodCallEvent, str, str]]] = {}

    for name in names:
        file_path = f"{name}.go"
        events, file_io_calls = draw(_file_events(path=file_path))
        events_by_file[file_path] = events
        metadata[file_path] = file_io_calls

    return events_by_file, metadata


# ---------------------------------------------------------------------------
# Expected-output computation
# ---------------------------------------------------------------------------


def _receiver_label(event: MethodCallEvent) -> str:
    """Return the dotted-name receiver label used in description suffixes.

    Mirrors :func:`go_io._format_file_read_description` /
    :func:`go_io._format_file_written_description`: the receiver chain
    is joined with ``.``, and the literal token ``<unqualified>`` is
    substituted for the empty-chain case. Unqualified file-I/O calls
    do not occur in any sample repository; the fallback exists for
    structural completeness.
    """

    if not event.receiver_chain:
        return "<unqualified>"
    return ".".join(event.receiver_chain)


def _expected_read_description(
    event: MethodCallEvent, path_value: str,
) -> str:
    """Return the description a recognized file-read call must produce.

    Mirrors :func:`go_io._format_file_read_description` verbatim:
    ``file read <path> via <recv>.<method>() at <file>:<line>``.
    """

    return (
        f"{_FILE_READ_PREFIX}{path_value} "
        f"via {_receiver_label(event)}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_written_description(
    event: MethodCallEvent, path_value: str,
) -> str:
    """Return the description a recognized file-write call must produce.

    Mirrors :func:`go_io._format_file_written_description` verbatim:
    ``file written <path> via <recv>.<method>() at <file>:<line>``.
    """

    return (
        f"{_FILE_WRITTEN_PREFIX}{path_value} "
        f"via {_receiver_label(event)}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_inputs_and_outputs(
    metadata: dict[str, list[tuple[MethodCallEvent, str, str]]],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Compute the deduplicated expected ``AbstractInput`` / ``AbstractOutput`` lists.

    Iteration order matches the recognizer's: paths in sorted order,
    and within each file the order in which the file-I/O calls were
    appended to the events list. The dedup contract is
    ``(category, description)`` — identical to the recognizer's
    ``seen_inputs`` / ``seen_outputs`` sets.

    The ``"both"`` verdict — produced by Requirement 6.3's undecidable
    branch — emits one input *and* one output for the same call site;
    the two emissions share the call's Source_Location suffix but
    carry distinct ``file read`` / ``file written`` action prefixes,
    which keeps the per-category dedup keys distinct from each other
    and from any neighboring single-verdict emission.
    """

    inputs: list[AbstractInput] = []
    outputs: list[AbstractOutput] = []
    seen_in: set[tuple[AbstractInputCategory, str]] = set()
    seen_out: set[tuple[AbstractOutputCategory, str]] = set()

    for path in sorted(metadata):
        for event, verdict, expected_path in metadata[path]:
            read_desc = _expected_read_description(event, expected_path)
            write_desc = _expected_written_description(event, expected_path)
            if verdict in ("read", "both"):
                in_key = (_FILE_READ, read_desc)
                if in_key not in seen_in:
                    seen_in.add(in_key)
                    inputs.append(
                        AbstractInput(category=_FILE_READ, description=read_desc),
                    )
            if verdict in ("write", "both"):
                out_key = (_FILE_WRITTEN, write_desc)
                if out_key not in seen_out:
                    seen_out.add(out_key)
                    outputs.append(
                        AbstractOutput(category=_FILE_WRITTEN, description=write_desc),
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


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_file_io_matches_expected_output(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[str, list[tuple[MethodCallEvent, str, str]]],
    ],
) -> None:
    """Property 8 main invariant: classification follows the documented mapping.

    Validates Requirements 6.1, 6.2, 6.3, 6.4 jointly. The expected
    output is computed by mirroring the recognizer's decision tree on
    the per-file generator metadata, then asserting full-list
    equality. Per-emission invariants validated by construction:

    * Exactly one ``AbstractInput(category=file_read)`` per recognized
      ``os.Open`` / ``os.ReadFile`` / ``ioutil.ReadFile`` and per
      ``os.OpenFile(p, os.O_RDONLY, m)`` (Requirement 6.1).
    * Exactly one ``AbstractOutput(category=file_written)`` per
      recognized ``os.Create`` / ``os.WriteFile`` / ``ioutil.WriteFile``
      and per ``os.OpenFile(p, os.O_<WRITE>, m)`` (Requirement 6.2).
    * One ``AbstractInput`` and one ``AbstractOutput`` per recognized
      ``os.OpenFile`` whose flag expression is not statically
      determinable (Requirement 6.3 undecidable branch).
    * String-literal paths appear verbatim in the description
      (Requirement 6.4 + task-level "record the literal path argument
      when string-literal" clause); non-literal expressions and
      absent path arguments are recorded as ``<dynamic>``.
    * The Source_Location suffix ``at <file>:<line>`` appears on
      every emission (Requirement 6.4).
    """

    events_by_file, metadata = case

    actual_inputs, actual_outputs = _extract_file_io(events_by_file)
    expected_inputs, expected_outputs = _expected_inputs_and_outputs(metadata)

    assert actual_inputs == expected_inputs, (
        "file-I/O recognizer inputs diverged from expected:\n"
        f"  actual:   {actual_inputs!r}\n"
        f"  expected: {expected_inputs!r}"
    )
    assert actual_outputs == expected_outputs, (
        "file-I/O recognizer outputs diverged from expected:\n"
        f"  actual:   {actual_outputs!r}\n"
        f"  expected: {expected_outputs!r}"
    )

    # Universal invariants on the actual output, independent of the
    # expected-output computation. These would catch a regression that
    # made the recognizer's output structurally wrong even when the
    # full-list equality also passed (e.g. by a coincidental two-bug
    # cancellation in the expected-output mirror).
    for entry in actual_inputs:
        assert entry.category is _FILE_READ, (
            f"input entry {entry!r} carries category {entry.category!r}; "
            f"the file-I/O recognizer must always emit file_read for "
            f"the read branch (Requirement 6.1)"
        )
        assert entry.description.startswith(_FILE_READ_PREFIX), (
            f"input entry {entry!r} does not begin with the canonical "
            f"'file read ' action prefix; Requirement 6.1 mandates a "
            f"category-tagged description"
        )
        assert _LOCATION_FRAGMENT in entry.description, (
            f"input entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 6.4 mandates a Source_Location on "
            f"every emission"
        )
    for entry in actual_outputs:
        assert entry.category is _FILE_WRITTEN, (
            f"output entry {entry!r} carries category {entry.category!r}; "
            f"the file-I/O recognizer must always emit file_written for "
            f"the write branch (Requirement 6.2)"
        )
        assert entry.description.startswith(_FILE_WRITTEN_PREFIX), (
            f"output entry {entry!r} does not begin with the canonical "
            f"'file written ' action prefix; Requirement 6.2 mandates "
            f"a category-tagged description"
        )
        assert _LOCATION_FRAGMENT in entry.description, (
            f"output entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 6.4 mandates a Source_Location on "
            f"every emission"
        )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_file_io_preserves_string_literal_path_verbatim(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[str, list[tuple[MethodCallEvent, str, str]]],
    ],
) -> None:
    """Verbatim-preservation invariant for string-literal path arguments.

    Validates Requirement 6.4 + task-level "record the literal path
    argument when string-literal" clause. For every recognized call
    whose first argument is a ``StringLitArg``, the recognizer must
    place the literal contents into the description verbatim — with
    no stripping, normalization, or truncation — paired with the
    call's Source_Location suffix.
    """

    events_by_file, metadata = case
    actual_inputs, actual_outputs = _extract_file_io(events_by_file)
    descriptions = [entry.description for entry in actual_inputs] + [
        entry.description for entry in actual_outputs
    ]

    for path in sorted(metadata):
        for event, _verdict, expected_path in metadata[path]:
            if expected_path == _DYNAMIC_PATH:
                # Non-literal path: the dynamic-placeholder branch is
                # covered by the main invariant test. Skip here.
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            matched = any(
                expected_path in desc and location_suffix in desc
                for desc in descriptions
            )
            assert matched, (
                f"string-literal path {expected_path!r} at "
                f"{event.file_path}:{event.line} was not preserved "
                f"verbatim in any emitted description; descriptions: "
                f"{descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_file_io_undecidable_openfile_emits_both(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[str, list[tuple[MethodCallEvent, str, str]]],
    ],
) -> None:
    """Undecidable-flag invariant for ``os.OpenFile`` (Requirement 6.3).

    For every recognized ``os.OpenFile`` call whose flag expression
    cannot be statically determined — non-canonical arity, a non-``os``
    package prefix, a multi-segment or non-recognized atom, or any
    argument shape other than ``DottedArg(("os", "O_<NAME>"))`` — the
    recognizer must emit *both* one ``AbstractInput(file_read)`` and
    one ``AbstractOutput(file_written)`` sharing the call's
    Source_Location suffix.
    """

    events_by_file, metadata = case
    actual_inputs, actual_outputs = _extract_file_io(events_by_file)
    input_descriptions = [entry.description for entry in actual_inputs]
    output_descriptions = [entry.description for entry in actual_outputs]

    for path in sorted(metadata):
        for event, verdict, _expected_path in metadata[path]:
            if verdict != "both":
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            # The same call site must contribute both a file_read and
            # a file_written description carrying its location suffix.
            in_matched = any(location_suffix in desc for desc in input_descriptions)
            out_matched = any(location_suffix in desc for desc in output_descriptions)
            assert in_matched, (
                f"undecidable-flag os.OpenFile call at "
                f"{event.file_path}:{event.line} did not contribute a "
                f"file_read input description; Requirement 6.3 mandates "
                f"both emissions for the undecidable case; input "
                f"descriptions: {input_descriptions!r}"
            )
            assert out_matched, (
                f"undecidable-flag os.OpenFile call at "
                f"{event.file_path}:{event.line} did not contribute a "
                f"file_written output description; Requirement 6.3 "
                f"mandates both emissions for the undecidable case; "
                f"output descriptions: {output_descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_file_io_ignores_fx_and_viper_receivers(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[str, list[tuple[MethodCallEvent, str, str]]],
    ],
) -> None:
    """fx / viper dispatch-boundary exclusion invariant.

    Validates Requirements 12.1, 12.3, 13.1, 13.2. Running the
    recognizer over the full event stream and over the stream with
    every fx / viper ``MethodCallEvent`` removed must produce
    identical output. Equality of the two outputs implies the
    recognizer consulted no fx / viper receiver for any of its
    decisions — including the (deliberately adversarial) noise events
    that carry a *valid*-looking ``OpenFile``-shaped argument list.
    """

    events_by_file, _ = case
    with_noise = _extract_file_io(events_by_file)
    without_noise = _extract_file_io(_stripped_of_fx_viper(events_by_file))
    assert with_noise == without_noise, (
        "file-I/O recognizer output changed after removing fx/viper "
        "MethodCallEvents; the dispatch-boundary exclusion must be "
        "inert (Requirements 12.1, 12.3, 13.1, 13.2)"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_file_io_is_iteration_order_independent(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[str, list[tuple[MethodCallEvent, str, str]]],
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
    assert _extract_file_io(events_by_file) == _extract_file_io(reversed_mapping), (
        "file-I/O recognizer output depends on insertion order; "
        "Requirement 11.4 mandates deterministic, path-sorted iteration"
    )

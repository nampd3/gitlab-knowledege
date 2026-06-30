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

1. **Unconditional read/write methods emit exactly one entry per call**
   (Requirements 6.1, 6.2). Every recognized ``os.Open``,
   ``os.ReadFile``, and ``ioutil.ReadFile`` call produces exactly one
   ``AbstractInput(file_read)``. Every recognized ``os.Create``,
   ``os.WriteFile``, and ``ioutil.WriteFile`` call produces exactly
   one ``AbstractOutput(file_written)``.

2. **``os.OpenFile`` flag-bitmask classification follows the
   documented mapping** (Requirement 6.3). A canonical
   ``(path, flag, mode)`` call with:

   * a single ``os.O_RDONLY`` flag atom emits one
     ``AbstractInput(file_read)`` and nothing else;
   * a single ``os.O_WRONLY``, ``os.O_RDWR``, ``os.O_APPEND``,
     ``os.O_CREATE``, or ``os.O_TRUNC`` flag atom emits one
     ``AbstractOutput(file_written)`` and nothing else;
   * a flag expression that is not statically determinable (a
     multi-atom OR expression that the parser splits across argument
     slots, an identifier reference, a function call, or any other
     non-``DottedArg(("os", "O_<NAME>"))`` shape) emits **both** an
     ``AbstractInput(file_read)`` and an ``AbstractOutput(file_written)``.

3. **Path argument recording follows the verbatim/dynamic rule**
   (Requirement 6.4 and the task-level "record the literal path
   argument when string-literal, otherwise ``<dynamic>``" clause).
   String-literal paths appear verbatim in the description; every
   other argument shape (or an absent first argument) is recorded as
   ``<dynamic>``.

4. **Source_Location is attached to every emission** (Requirement
   6.4). The recognizer encodes the ``MethodCallEvent``'s
   ``file_path`` and 1-indexed ``line`` into the description as an
   ``at <file>:<line>`` suffix, matching the HTTP, scheduler, and
   ActiveMQ recognizers' convention.

5. **fx and viper exclusions are inert at the dispatch boundary**
   (Requirements 12.1, 12.3, 13.1, 13.2). Adding arbitrary
   ``fx.<method>`` and ``viper.<method>`` calls — even ones whose
   method names overlap the recognized file-I/O vocabulary — must not
   change the recognizer's output.

6. **Deterministic, path-sorted iteration** (Requirement 11.4). The
   recognizer is a pure function of its input mapping. Reversing the
   mapping's insertion order must not change the output lists.

The recognizer's :func:`_extract_file_io` is package-internal; the
test imports it through the module path to mirror the convention used
by ``test_property_06_go_scheduler_detection.py`` and
``test_property_07_go_activemq_detection.py`` for their respective
package-internal entry points.
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


# Path-literal alphabet. Restricted to characters that survive the
# description's f-string interpolation unmodified — letters, digits, and
# the path separators commonly observed in Go source. Newlines, carriage
# returns, and tabs are excluded so a generated path cannot break the
# ``at <file>:<line>`` suffix's line orientation. Spaces are excluded so
# the verbatim-preservation property can rely on a strict substring
# match.
_PATH_CHARS: Final[str] = string.ascii_letters + string.digits + "._/-"


# Identifier alphabet for receiver names, file basenames, and dotted-arg
# segments. Constrained to characters Go accepts in identifiers so the
# fixtures look like real source.
_IDENT_CHARS: Final[str] = string.ascii_letters + string.digits + "_"


# Numeric-literal alphabet for the ``mode`` argument on ``os.OpenFile``
# and for stand-in path arguments in the dynamic-path branch. Go file
# modes are typically of the form ``0644``; a small digit alphabet keeps
# the fixtures readable while still letting the recognizer's branches
# be exercised.
_NUMBER_CHARS: Final[str] = string.digits


# Unknown-argument text alphabet — identifiers and a few punctuators
# only, so that an ``UnknownArg`` payload remains visually distinct
# from a path literal.
_UNKNOWN_CHARS: Final[str] = string.ascii_letters + string.digits + "_|."


# Canonical category enum values the recognizer always emits.
_FILE_READ: Final[AbstractInputCategory] = AbstractInputCategory.FILE_READ
_FILE_WRITTEN: Final[AbstractOutputCategory] = AbstractOutputCategory.FILE_WRITTEN


# Canonical description fragments the recognizer always emits.
# Anchored by the implementation's :func:`_format_file_read_description`
# and :func:`_format_file_written_description` templates.
_READ_PREFIX: Final[str] = "file read "
_WRITTEN_PREFIX: Final[str] = "file written "
_DYNAMIC_TOKEN: Final[str] = "<dynamic>"


# Recognized read- and write-only method names per
# :data:`_OS_READ_METHODS`, :data:`_OS_WRITE_METHODS`,
# :data:`_IOUTIL_READ_METHODS`, and :data:`_IOUTIL_WRITE_METHODS` in
# the recognizer module.
_OS_READ_METHODS: Final[tuple[str, ...]] = ("Open", "ReadFile")
_OS_WRITE_METHODS: Final[tuple[str, ...]] = ("Create", "WriteFile")
_IOUTIL_READ_METHODS: Final[tuple[str, ...]] = ("ReadFile",)
_IOUTIL_WRITE_METHODS: Final[tuple[str, ...]] = ("WriteFile",)
_OS_OPENFILE_METHOD: Final[str] = "OpenFile"


# Flag-atom partitioning per Requirement 6.3.
_READ_ATOM: Final[str] = "O_RDONLY"
_WRITE_ATOMS: Final[tuple[str, ...]] = (
    "O_WRONLY",
    "O_RDWR",
    "O_APPEND",
    "O_CREATE",
    "O_TRUNC",
)
# Atoms that are documented as undecidable: they appear neither in the
# read-only single-atom set nor in the write-atom set, so the
# recognizer treats them as "not statically determinable" and routes
# the call to the "emit both" branch.
_UNDECIDABLE_ATOMS: Final[tuple[str, ...]] = ("O_EXCL", "O_SYNC", "O_NONBLOCK")


# A small set of fx / viper method names sufficient to demonstrate that
# the recognizer's dispatch-boundary exclusion holds. The Requirement
# 12 / 13 carve-out applies to *any* method on these receivers, so
# sampling a handful is enough to make the property's invariant
# observable: the output is unchanged when these calls are present or
# absent. Method names that overlap with the recognized vocabulary
# (``Open``, ``ReadFile``, ``Create``, ``WriteFile``, ``OpenFile``) are
# included so a regression that forgot to honor the receiver-chain
# exclusion would surface immediately.
_FX_METHODS: Final[tuple[str, ...]] = (
    "Provide",
    "Invoke",
    "Module",
    "Open",
    "ReadFile",
    "Create",
    "WriteFile",
    "OpenFile",
)
_VIPER_METHODS: Final[tuple[str, ...]] = (
    "New",
    "GetString",
    "Open",
    "ReadFile",
    "WriteFile",
    "OpenFile",
)


# ---------------------------------------------------------------------------
# Per-argument strategies
# ---------------------------------------------------------------------------


_ident = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8)
_path_literal = st.text(alphabet=_PATH_CHARS, min_size=1, max_size=20)
_dotted_segments = st.lists(_ident, min_size=2, max_size=4).map(tuple)
_number_literal = st.text(alphabet=_NUMBER_CHARS, min_size=1, max_size=4)
_unknown_text = st.text(alphabet=_UNKNOWN_CHARS, min_size=1, max_size=10)


@st.composite
def _path_argument(draw: st.DrawFn) -> tuple["ArgRef", str]:
    """Generate a path-argument value and its expected description fragment.

    Returns ``(value, expected_fragment)``:

    * ``value`` is the ``ArgRef`` placed at the path slot.
    * ``expected_fragment`` is the substring the recognizer must place
      in the description — the verbatim literal for ``StringLitArg``,
      and the canonical ``<dynamic>`` placeholder for every other
      argument shape (identifier, dotted-name reference, nested call,
      numeric literal, opaque text). Covers Requirement 6.4 and the
      task-level "record the literal path argument when string-literal,
      otherwise ``<dynamic>``" clause.
    """

    kind = draw(
        st.sampled_from(
            ("string", "ident", "dotted", "call", "number", "unknown"),
        ),
    )
    if kind == "string":
        value = draw(_path_literal)
        return StringLitArg(value=value), value
    if kind == "ident":
        return IdentArg(name=draw(_ident)), _DYNAMIC_TOKEN
    if kind == "dotted":
        return DottedArg(parts=draw(_dotted_segments)), _DYNAMIC_TOKEN
    if kind == "call":
        # A nested method call (e.g. ``cfg.Path()``). The recognizer
        # never inspects the inner call's identity for the path slot,
        # so any benign inner call works as a stand-in.
        nested = MethodCallEvent(
            receiver_chain=(draw(_ident),),
            method_name=draw(_ident),
            args=(),
            file_path="dummy.go",
            line=1,
        )
        return CallArg(call=nested), _DYNAMIC_TOKEN
    if kind == "number":
        return NumberLitArg(text=draw(_number_literal)), _DYNAMIC_TOKEN
    return UnknownArg(text=draw(_unknown_text)), _DYNAMIC_TOKEN


# Verdict tokens used to drive expected-output computation. They mirror
# the recognizer's ``_FileIOVerdict`` strings without depending on the
# private alias.
_READ_VERDICT: Final[str] = "read"
_WRITE_VERDICT: Final[str] = "write"
_BOTH_VERDICT: Final[str] = "both"


@st.composite
def _flag_argument(draw: st.DrawFn) -> tuple["ArgRef", str]:
    """Generate an ``os.OpenFile`` flag argument and its verdict token.

    Returns ``(value, verdict)`` where ``verdict`` is one of
    ``"read"``, ``"write"``, or ``"both"``. The five branches cover
    every shape Requirement 6.3 distinguishes:

    * ``DottedArg(("os", "O_RDONLY"))`` → ``"read"``.
    * ``DottedArg(("os", <write-atom>))`` → ``"write"``.
    * ``DottedArg(("os", <undecidable-atom>))`` → ``"both"`` (the
      recognizer's spec is silent on atoms outside the six named in
      Requirements 6.1/6.2, so the conservative "emit both" branch
      applies).
    * ``DottedArg`` with the wrong package alias, the wrong segment
      count, or a non-``DottedArg`` shape (``IdentArg``,
      ``NumberLitArg``, ``CallArg``, ``StringLitArg``,
      ``UnknownArg``) → ``"both"``.
    """

    kind = draw(
        st.sampled_from(
            (
                "read_only",
                "write_atom",
                "undecidable_atom",
                "wrong_alias",
                "long_dotted",
                "ident",
                "number",
                "call",
                "string",
                "unknown",
            ),
        ),
    )
    if kind == "read_only":
        return DottedArg(parts=("os", _READ_ATOM)), _READ_VERDICT
    if kind == "write_atom":
        atom = draw(st.sampled_from(_WRITE_ATOMS))
        return DottedArg(parts=("os", atom)), _WRITE_VERDICT
    if kind == "undecidable_atom":
        atom = draw(st.sampled_from(_UNDECIDABLE_ATOMS))
        return DottedArg(parts=("os", atom)), _BOTH_VERDICT
    if kind == "wrong_alias":
        alias = draw(st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=6))
        # Exclude exact ``"os"`` so the verdict stays ``both``; the
        # alias gate keys off the exact ``"os"`` string in the
        # recognizer.
        if alias == "os":
            alias = "syscall"
        return DottedArg(parts=(alias, _READ_ATOM)), _BOTH_VERDICT
    if kind == "long_dotted":
        # A three-segment dotted path (``foo.os.O_RDONLY``) cannot
        # encode a recognized flag, so the recognizer routes it to
        # the "both" branch via the segment-count gate.
        return DottedArg(parts=("foo", "os", _READ_ATOM)), _BOTH_VERDICT
    if kind == "ident":
        return IdentArg(name=draw(_ident)), _BOTH_VERDICT
    if kind == "number":
        return NumberLitArg(text=draw(_number_literal)), _BOTH_VERDICT
    if kind == "call":
        nested = MethodCallEvent(
            receiver_chain=(draw(_ident),),
            method_name=draw(_ident),
            args=(),
            file_path="dummy.go",
            line=1,
        )
        return CallArg(call=nested), _BOTH_VERDICT
    if kind == "string":
        return StringLitArg(value=draw(_path_literal)), _BOTH_VERDICT
    return UnknownArg(text=draw(_unknown_text)), _BOTH_VERDICT


# ---------------------------------------------------------------------------
# Per-call strategies
# ---------------------------------------------------------------------------


@st.composite
def _os_read_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate an ``os.<read>`` call.

    Returns ``(event, expected_path_fragment, verdict)``. The verdict is
    always ``"read"``: ``os.Open`` and ``os.ReadFile`` unconditionally
    emit one ``AbstractInput(file_read)`` per call (Requirement 6.1).
    """

    method = draw(st.sampled_from(_OS_READ_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))
    path_arg, expected_fragment = draw(_path_argument())
    return (
        MethodCallEvent(
            receiver_chain=("os",),
            method_name=method,
            args=(path_arg,),
            file_path=file_path,
            line=line,
        ),
        expected_fragment,
        _READ_VERDICT,
    )


@st.composite
def _os_write_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate an ``os.<write>`` call.

    Returns ``(event, expected_path_fragment, verdict)``. The verdict
    is always ``"write"``: ``os.Create`` and ``os.WriteFile``
    unconditionally emit one ``AbstractOutput(file_written)`` per call
    (Requirement 6.2). The mandatory data argument on ``WriteFile`` is
    represented by a stand-in identifier; the recognizer never
    inspects positions past the path slot for these methods.
    """

    method = draw(st.sampled_from(_OS_WRITE_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))
    path_arg, expected_fragment = draw(_path_argument())
    args: tuple[ArgRef, ...]
    if method == "WriteFile":
        args = (path_arg, IdentArg(name="data"), NumberLitArg(text="0644"))
    else:
        args = (path_arg,)
    return (
        MethodCallEvent(
            receiver_chain=("os",),
            method_name=method,
            args=args,
            file_path=file_path,
            line=line,
        ),
        expected_fragment,
        _WRITE_VERDICT,
    )


@st.composite
def _ioutil_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate an ``ioutil.{ReadFile,WriteFile}`` call.

    Returns ``(event, expected_path_fragment, verdict)``. ``ReadFile``
    is unconditionally a read; ``WriteFile`` is unconditionally a
    write (Requirements 6.1, 6.2 glossary entry
    "Go_File_IO_Function").
    """

    method = draw(st.sampled_from(_IOUTIL_READ_METHODS + _IOUTIL_WRITE_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))
    path_arg, expected_fragment = draw(_path_argument())
    if method in _IOUTIL_WRITE_METHODS:
        args = (path_arg, IdentArg(name="data"), NumberLitArg(text="0644"))
        verdict = _WRITE_VERDICT
    else:
        args = (path_arg,)
        verdict = _READ_VERDICT
    return (
        MethodCallEvent(
            receiver_chain=("ioutil",),
            method_name=method,
            args=args,
            file_path=file_path,
            line=line,
        ),
        expected_fragment,
        verdict,
    )


@st.composite
def _openfile_canonical_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate a canonical ``os.OpenFile(path, flag, mode)`` call.

    Returns ``(event, expected_path_fragment, verdict)``. The argument
    count is exactly three so the recognizer's arity gate passes and
    the verdict reflects the flag's classification (``"read"``,
    ``"write"``, or ``"both"`` per Requirement 6.3).
    """

    line = draw(st.integers(min_value=1, max_value=999))
    path_arg, expected_fragment = draw(_path_argument())
    flag_arg, verdict = draw(_flag_argument())
    mode_arg = NumberLitArg(text=draw(_number_literal))
    return (
        MethodCallEvent(
            receiver_chain=("os",),
            method_name=_OS_OPENFILE_METHOD,
            args=(path_arg, flag_arg, mode_arg),
            file_path=file_path,
            line=line,
        ),
        expected_fragment,
        verdict,
    )


@st.composite
def _openfile_split_or_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate an ``os.OpenFile`` with a multi-atom OR flag expression.

    The parser splits ``os.O_RDWR | os.O_CREATE`` across argument slots
    so the recognized call carries four or more positional arguments:
    a path, a leading ``DottedArg`` flag atom, one or more
    ``UnknownArg`` continuations carrying the ``"| os . O_CREATE"``
    text, and a trailing ``NumberLitArg`` mode. The arity-gate
    requirement of exactly three positional arguments fails, so the
    recognizer routes the call to the ``"both"`` branch documented in
    Requirement 6.3 third bullet.

    Returns ``(event, expected_path_fragment, "both")``.
    """

    line = draw(st.integers(min_value=1, max_value=999))
    path_arg, expected_fragment = draw(_path_argument())
    leading_flag = DottedArg(parts=("os", draw(st.sampled_from(_WRITE_ATOMS))))
    continuation = UnknownArg(text=draw(_unknown_text))
    mode_arg = NumberLitArg(text=draw(_number_literal))
    args = (path_arg, leading_flag, continuation, mode_arg)
    return (
        MethodCallEvent(
            receiver_chain=("os",),
            method_name=_OS_OPENFILE_METHOD,
            args=args,
            file_path=file_path,
            line=line,
        ),
        expected_fragment,
        _BOTH_VERDICT,
    )


@st.composite
def _openfile_wrong_arity_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate an ``os.OpenFile`` call with a non-canonical argument count.

    Either fewer than three positional arguments (malformed Go that
    would not compile but is a defensible recognizer-input shape) or
    more than three (a parser-split OR expression). The recognizer's
    arity gate routes both cases to the ``"both"`` branch.

    Returns ``(event, expected_path_fragment, "both")``.
    """

    line = draw(st.integers(min_value=1, max_value=999))
    path_arg, expected_fragment = draw(_path_argument())
    shape = draw(st.sampled_from(("zero", "one", "two", "four", "five")))
    if shape == "zero":
        args: tuple[ArgRef, ...] = ()
        # When the path argument is missing, the recognizer's
        # path-extraction fallback substitutes ``<dynamic>``.
        expected_fragment = _DYNAMIC_TOKEN
    elif shape == "one":
        args = (path_arg,)
    elif shape == "two":
        flag_arg, _ = draw(_flag_argument())
        args = (path_arg, flag_arg)
    elif shape == "four":
        flag_arg, _ = draw(_flag_argument())
        args = (
            path_arg,
            flag_arg,
            NumberLitArg(text=draw(_number_literal)),
            IdentArg(name="extra"),
        )
    else:  # "five"
        flag_arg, _ = draw(_flag_argument())
        args = (
            path_arg,
            flag_arg,
            NumberLitArg(text=draw(_number_literal)),
            IdentArg(name="extra"),
            UnknownArg(text=draw(_unknown_text)),
        )
    return (
        MethodCallEvent(
            receiver_chain=("os",),
            method_name=_OS_OPENFILE_METHOD,
            args=args,
            file_path=file_path,
            line=line,
        ),
        expected_fragment,
        _BOTH_VERDICT,
    )


@st.composite
def _file_io_call(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str, str]:
    """Generate one recognized file-I/O call.

    Returns ``(event, expected_path_fragment, verdict)``. The verdict
    drives the expected-output computation:

    * ``"read"`` → one ``AbstractInput(file_read)``.
    * ``"write"`` → one ``AbstractOutput(file_written)``.
    * ``"both"`` → one of each.
    """

    kind = draw(
        st.sampled_from(
            (
                "os_read",
                "os_write",
                "ioutil",
                "openfile_canonical",
                "openfile_split",
                "openfile_arity",
            ),
        ),
    )
    if kind == "os_read":
        return draw(_os_read_call(file_path=file_path))
    if kind == "os_write":
        return draw(_os_write_call(file_path=file_path))
    if kind == "ioutil":
        return draw(_ioutil_call(file_path=file_path))
    if kind == "openfile_canonical":
        return draw(_openfile_canonical_call(file_path=file_path))
    if kind == "openfile_split":
        return draw(_openfile_split_or_call(file_path=file_path))
    return draw(_openfile_wrong_arity_call(file_path=file_path))


# ---------------------------------------------------------------------------
# Noise strategies (must not affect output)
# ---------------------------------------------------------------------------


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
    regression that forgot to honor the receiver-chain exclusion
    would surface immediately: the noise event would otherwise match
    the recognizer's method-name gate.

    To make the noise as adversarial as possible, the call carries
    a canonical ``(path, flag, mode)`` argument shape with a path
    literal and a single ``os.O_RDONLY`` flag atom. If the recognizer
    ever inspected fx / viper receivers, this would emit a
    ``file_read`` entry that the expected-output computation does not
    predict, breaking the full-list equality assertion.
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
            StringLitArg(value=draw(_path_literal)),
            DottedArg(parts=("os", _READ_ATOM)),
            NumberLitArg(text=draw(_number_literal)),
        ),
        file_path=file_path,
        line=line,
    )


@st.composite
def _unrelated_call_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a method call with a non-file-I/O receiver or method.

    Covers the broad "receiver / method does not match" rejection
    branch. Concrete method-name choices overlap with other Go
    recognizers' vocabularies (``HandleFunc``, ``AddFunc``,
    ``Subscribe``, ``SendMessage``) so a regression that accidentally
    widened the recognizer's gates would immediately fire here.
    """

    kind = draw(st.sampled_from(("unrelated_recv", "unrelated_method")))
    line = draw(st.integers(min_value=1, max_value=999))
    if kind == "unrelated_recv":
        receiver = draw(_ident)
        # Use a method name from the recognized vocabulary to make
        # the noise adversarial against a regression that forgot to
        # gate on the receiver.
        method = draw(
            st.sampled_from(
                _OS_READ_METHODS
                + _OS_WRITE_METHODS
                + (_OS_OPENFILE_METHOD,)
                + ("ReadFile", "WriteFile"),
            ),
        )
        return MethodCallEvent(
            receiver_chain=(receiver,),
            method_name=method,
            args=(StringLitArg(value=draw(_path_literal)),),
            file_path=file_path,
            line=line,
        )
    # unrelated_method: ``os.`` or ``ioutil.`` receiver but a method
    # name outside the recognized set. A regression that widened the
    # method-name set would surface here.
    receiver_chain = draw(st.sampled_from((("os",), ("ioutil",))))
    method = draw(
        st.sampled_from(("HandleFunc", "AddFunc", "Subscribe", "SendMessage", "Getenv")),
    )
    return MethodCallEvent(
        receiver_chain=receiver_chain,
        method_name=method,
        args=(StringLitArg(value=draw(_path_literal)),),
        file_path=file_path,
        line=line,
    )


@st.composite
def _struct_lit_noise_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> StructLitEvent:
    """Generate a ``StructLitEvent`` to demonstrate it is ignored.

    The recognizer only inspects ``MethodCallEvent`` instances; struct
    literals must produce no emission regardless of their type name
    or field contents.
    """

    line = draw(st.integers(min_value=1, max_value=999))
    return StructLitEvent(
        type_name=draw(_ident),
        package_alias=draw(st.one_of(st.none(), _ident)),
        fields=(("Path", StringLitArg(value=draw(_path_literal))),),
        is_pointer=draw(st.booleans()),
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
) -> tuple[list["GoEvent"], list[tuple[MethodCallEvent, str, str]]]:
    """Generate one file's event list plus the per-call expectations.

    Returns ``(events, file_io_calls)``:

    * ``events`` is the order-preserving event list assigned to the
      file's key in the ``events_by_file`` mapping. Always carries
      one or more fx / viper noise events plus optional unrelated
      method calls and struct-literal noise so the exclusion
      properties below are exercised on every example.
    * ``file_io_calls`` is the per-call record the expected-output
      computation needs to predict the recognizer's output:
      ``(event, expected_path_fragment, verdict)`` triples. Their
      event order matches insertion into ``events``, which lets the
      test assert per-emission shape per file.
    """

    events: list[GoEvent] = []

    # A noise prefix exercising the dispatch-boundary fx / viper skip
    # on every example.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    # Optional unrelated calls — must emit nothing.
    unrelated = draw(
        st.lists(_unrelated_call_event(file_path=path), min_size=0, max_size=2),
    )
    events.extend(unrelated)

    # Optional struct-literal noise — must emit nothing.
    if draw(st.booleans()):
        events.append(draw(_struct_lit_noise_event(file_path=path)))

    # The recognized file-I/O calls themselves.
    file_io_calls = draw(
        st.lists(_file_io_call(file_path=path), min_size=0, max_size=4),
    )
    for event, _, _ in file_io_calls:
        events.append(event)

    # A trailing noise event in case the noise's position relative to
    # the recognized events ever matters (it must not).
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
    carries the list of recognized file-I/O calls needed to compute
    the expected recognizer output.
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
        path = f"{name}.go"
        events, file_io_calls = draw(_file_events(path=path))
        events_by_file[path] = events
        metadata[path] = file_io_calls

    return events_by_file, metadata


# ---------------------------------------------------------------------------
# Expected-output computation
# ---------------------------------------------------------------------------


def _receiver_label(event: MethodCallEvent) -> str:
    """Return the dotted-name receiver label used in description suffixes."""

    if not event.receiver_chain:
        return "<unqualified>"
    return ".".join(event.receiver_chain)


def _expected_read_description(event: MethodCallEvent, path_fragment: str) -> str:
    """Return the description a recognized read call must produce.

    Mirrors :func:`_format_file_read_description` in the recognizer:
    ``"file read <path> via <recv>.<method>() at <file>:<line>"``.
    """

    return (
        f"{_READ_PREFIX}{path_fragment} "
        f"via {_receiver_label(event)}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_written_description(event: MethodCallEvent, path_fragment: str) -> str:
    """Return the description a recognized write call must produce.

    Mirrors :func:`_format_file_written_description` in the
    recognizer: ``"file written <path> via <recv>.<method>() at
    <file>:<line>"``.
    """

    return (
        f"{_WRITTEN_PREFIX}{path_fragment} "
        f"via {_receiver_label(event)}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_inputs_and_outputs(
    metadata: dict[str, list[tuple[MethodCallEvent, str, str]]],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Compute the deduplicated expected I/O lists.

    Iteration order matches the recognizer's: paths in sorted order,
    and within each file the order in which calls were appended to the
    events list. Per-verdict emission rules per Requirements 6.1, 6.2,
    6.3:

    * ``"read"``  → one ``AbstractInput(file_read)``.
    * ``"write"`` → one ``AbstractOutput(file_written)``.
    * ``"both"``  → one of each.
    """

    inputs: list[AbstractInput] = []
    outputs: list[AbstractOutput] = []
    seen_in: set[tuple[AbstractInputCategory, str]] = set()
    seen_out: set[tuple[AbstractOutputCategory, str]] = set()

    for path in sorted(metadata):
        for event, path_fragment, verdict in metadata[path]:
            if verdict in (_READ_VERDICT, _BOTH_VERDICT):
                desc = _expected_read_description(event, path_fragment)
                key_in = (_FILE_READ, desc)
                if key_in not in seen_in:
                    seen_in.add(key_in)
                    inputs.append(
                        AbstractInput(category=_FILE_READ, description=desc),
                    )
            if verdict in (_WRITE_VERDICT, _BOTH_VERDICT):
                desc = _expected_written_description(event, path_fragment)
                key_out = (_FILE_WRITTEN, desc)
                if key_out not in seen_out:
                    seen_out.add(key_out)
                    outputs.append(
                        AbstractOutput(
                            category=_FILE_WRITTEN, description=desc,
                        ),
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
    the per-call generator metadata, then asserting full-list
    equality. Per-emission invariants validated by construction:

    * Exactly one ``AbstractInput(file_read)`` per recognized
      ``os.Open``, ``os.ReadFile``, ``ioutil.ReadFile`` call
      (Requirement 6.1).
    * Exactly one ``AbstractOutput(file_written)`` per recognized
      ``os.Create``, ``os.WriteFile``, ``ioutil.WriteFile`` call
      (Requirement 6.2).
    * Exactly one ``AbstractInput(file_read)`` per ``os.OpenFile``
      whose flag bitmask statically resolves to ``O_RDONLY``
      (Requirement 6.3 first bullet).
    * Exactly one ``AbstractOutput(file_written)`` per ``os.OpenFile``
      whose flag bitmask statically contains a write atom
      (Requirement 6.3 second bullet).
    * Both an ``AbstractInput(file_read)`` and an
      ``AbstractOutput(file_written)`` per ``os.OpenFile`` whose flag
      bitmask is not statically determinable (Requirement 6.3 third
      bullet).
    * The Source_Location suffix ``at <file>:<line>`` appears on
      every emission (Requirement 6.4).
    """

    events_by_file, metadata = case

    actual_inputs, actual_outputs = _extract_file_io(events_by_file)
    expected_inputs, expected_outputs = _expected_inputs_and_outputs(metadata)

    assert actual_inputs == expected_inputs, (
        "file-io recognizer inputs diverged from expected:\n"
        f"  actual:   {actual_inputs!r}\n"
        f"  expected: {expected_inputs!r}"
    )
    assert actual_outputs == expected_outputs, (
        "file-io recognizer outputs diverged from expected:\n"
        f"  actual:   {actual_outputs!r}\n"
        f"  expected: {expected_outputs!r}"
    )

    # Universal invariants on the actual output, independent of the
    # expected-output computation. These catch a regression that
    # made the recognizer's output structurally wrong even when the
    # full-list equality passed (e.g. by a coincidental two-bug
    # cancellation in the expected-output mirror).
    for entry in actual_inputs:
        assert entry.category is _FILE_READ, (
            f"input entry {entry!r} carries category {entry.category!r}; "
            f"the file-io recognizer must always emit file_read for "
            f"input entries (Requirement 6.1)"
        )
        assert entry.description.startswith(_READ_PREFIX), (
            f"input entry {entry!r} does not start with "
            f"{_READ_PREFIX!r}; the recognizer's description template "
            f"must declare the action verb at the front"
        )
        assert " at " in entry.description, (
            f"input entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 6.4 mandates a Source_Location on "
            f"every emission"
        )
    for entry in actual_outputs:
        assert entry.category is _FILE_WRITTEN, (
            f"output entry {entry!r} carries category {entry.category!r}; "
            f"the file-io recognizer must always emit file_written for "
            f"output entries (Requirement 6.2)"
        )
        assert entry.description.startswith(_WRITTEN_PREFIX), (
            f"output entry {entry!r} does not start with "
            f"{_WRITTEN_PREFIX!r}; the recognizer's description "
            f"template must declare the action verb at the front"
        )
        assert " at " in entry.description, (
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
    """Property 8 verbatim-path-preservation invariant.

    Validates Requirement 6.4 and the task-level "record the literal
    path argument when string-literal, otherwise ``<dynamic>``"
    clause. A string-literal path argument must appear in the
    description exactly as the source declared. The test asserts the
    literal substring is present in *some* output description for
    every recognized file-I/O call whose path argument is a
    ``StringLitArg``.
    """

    events_by_file, metadata = case
    actual_inputs, actual_outputs = _extract_file_io(events_by_file)
    descriptions = [entry.description for entry in actual_inputs]
    descriptions.extend(entry.description for entry in actual_outputs)

    for path in sorted(metadata):
        for event, path_fragment, _verdict in metadata[path]:
            if path_fragment == _DYNAMIC_TOKEN:
                # Non-literal path; the verbatim invariant does not
                # apply. The dynamic-token shape is exercised by the
                # main invariant test through expected-output equality.
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            matched = any(
                path_fragment in desc and location_suffix in desc
                for desc in descriptions
            )
            assert matched, (
                f"string-literal path {path_fragment!r} for "
                f"{event.file_path}:{event.line} not preserved verbatim "
                f"in any description; descriptions: {descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_file_io_openfile_undecidable_emits_both(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[str, list[tuple[MethodCallEvent, str, str]]],
    ],
) -> None:
    """Property 8 undecidable-flag invariant.

    Validates Requirement 6.3 third bullet directly. For every
    ``os.OpenFile`` call whose generator-side verdict is ``"both"``,
    the recognizer must emit one ``AbstractInput(file_read)`` *and*
    one ``AbstractOutput(file_written)`` at the call's Source_Location.
    The assertion checks that the call's ``at <file>:<line>`` suffix
    appears in both an input and an output description.
    """

    events_by_file, metadata = case
    actual_inputs, actual_outputs = _extract_file_io(events_by_file)
    input_descriptions = [entry.description for entry in actual_inputs]
    output_descriptions = [entry.description for entry in actual_outputs]

    for path in sorted(metadata):
        for event, _path_fragment, verdict in metadata[path]:
            if verdict != _BOTH_VERDICT:
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            has_input = any(location_suffix in desc for desc in input_descriptions)
            has_output = any(
                location_suffix in desc for desc in output_descriptions
            )
            assert has_input and has_output, (
                f"os.OpenFile undecidable-flag call at "
                f"{event.file_path}:{event.line} did not emit both an "
                f"input and an output; Requirement 6.3 third bullet "
                f"mandates both. "
                f"input_descriptions={input_descriptions!r}, "
                f"output_descriptions={output_descriptions!r}"
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
    that carry a canonical ``(path, flag, mode)`` argument shape with
    a recognized read-only flag.
    """

    events_by_file, _ = case
    with_noise = _extract_file_io(events_by_file)
    without_noise = _extract_file_io(_stripped_of_fx_viper(events_by_file))
    assert with_noise == without_noise, (
        "file-io recognizer output changed after removing fx/viper "
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

    Validates Requirement 11.4. The recognizer sorts paths internally,
    so two mappings with the same keys and values but different
    insertion orders must produce identical output lists.
    """

    events_by_file, _ = case
    reversed_mapping = dict(reversed(list(events_by_file.items())))
    assert _extract_file_io(events_by_file) == _extract_file_io(
        reversed_mapping,
    ), (
        "file-io recognizer output depends on insertion order; "
        "Requirement 11.4 mandates deterministic, path-sorted iteration"
    )

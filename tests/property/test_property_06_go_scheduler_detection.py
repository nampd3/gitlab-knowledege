# ruff: noqa: E501
# Feature: go-analyzer-support, Property 6: Scheduler detection emits one input per recognized registration, preserving schedule literals verbatim.
"""Property test for the Go scheduler-registration recognizer.

**Validates Requirements 4.1, 4.2, 4.3, 4.4, 4.5** (Property 6 in the
design, task 7.4 in ``tasks.md``).

The scheduler recognizer is the third of five recognizers composed by
``extract_go_io`` (task 7.11). This property test exercises its
package-internal entry point :func:`_extract_schedulers` directly so
the per-recognizer contract is pinned independently of the eventual
composition step.

The properties below capture the design's documented contract:

1. **One AbstractInput per recognized registration**
   (Requirement 4.1, 4.2 + Property 6 statement). Every event the
   recognizer accepts — a ``time.NewTicker`` / ``time.AfterFunc``
   call, a cron-recognized ``<v>.AddFunc`` / ``<v>.AddJob`` call,
   *or* a malformed cron-style call — produces exactly one entry
   with ``category=scheduled_event``.

2. **Schedule literals are recorded verbatim**
   (Requirement 4.3 first sentence). A ``StringLitArg`` schedule
   argument's contents appear unmodified in the description, even
   when the string is a six-field cron expression that a five-field
   parser would reject. The placeholder ``<dynamic>`` appears
   instead when the argument is not a string literal
   (Requirement 4.3 second sentence).

3. **Library marker reflects the call shape** (Requirement 4.2).
   Cron-style registrations carry the ``cron`` token; time-package
   registrations carry the ``time`` token. The
   ``(cron, seconds-precision)`` vs ``(cron, minute-precision)``
   marker mirrors the ``cron.WithSeconds()`` option's presence on the
   file's ``cron.New(...)`` call.

4. **Source_Location is attached to every emission**
   (Requirement 4.4). The recognizer encodes the
   ``MethodCallEvent``'s ``file_path`` and 1-indexed ``line`` into
   the description as a ``at <file>:<line>`` suffix, matching the
   HTTP and ActiveMQ recognizers' convention.

5. **Malformed shapes still emit one entry** (Requirement 4.5). When
   the method silhouette matches ``AddFunc`` / ``AddJob`` but the
   receiver is not cron-recognized — either because the file lacks a
   ``cron.New(...)`` call or because the receiver chain has the
   wrong shape — the recognizer still emits one entry with the
   ``<dynamic>`` schedule and the canonical "malformed or
   unsupported scheduler shape" suffix.

6. **fx and viper exclusions are inert at the dispatch boundary**
   (Requirements 12.1, 12.3, 13.1, 13.2). Adding arbitrary
   ``fx.<method>`` and ``viper.<method>`` calls to a file must not
   change the recognizer's output.

7. **Deterministic, path-sorted iteration** (Requirement 11.4). The
   recognizer is a pure function of its input mapping. Shuffling the
   mapping's insertion order must not change the output list.

The recognizer's :func:`_extract_schedulers` is package-internal; the
test imports it through the module path to mirror the convention used
by ``test_property_04_go_purpose_priority.py`` for
:func:`collect_go_candidates`.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import AbstractInput, AbstractInputCategory
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
from project_knowledge_mcp.project_analyzer.go.go_io import _extract_schedulers

if TYPE_CHECKING:
    from project_knowledge_mcp.project_analyzer.go._events import ArgRef, GoEvent


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets and shared constants
# ---------------------------------------------------------------------------

# Schedule-literal alphabet. Restricted to characters that survive the
# description's f-string interpolation unmodified — letters, digits, a
# few cron-syntax punctuators, and a space. ``\n``, ``\r``, and ``\t``
# are excluded so a generated schedule cannot break the
# ``at <file>:<line>`` suffix's line orientation. ``@`` is included so
# ``@every 5s`` and ``@hourly`` shapes can be generated; ``*`` is
# included so five- and six-field cron expressions can be generated.
_SCHEDULE_CHARS: Final[str] = string.ascii_letters + string.digits + "* /,-?@ "


# Identifier alphabet for receiver names, file basenames, and dotted-arg
# segments. Constrained to characters Go accepts in identifiers so the
# fixtures look like real source.
_IDENT_CHARS: Final[str] = string.ascii_letters + string.digits + "_"


# Numeric-literal alphabet for the time.NewTicker(<duration>) shape. Go
# duration literals are typically of the form ``5 * time.Second``; the
# recognizer only ever sees the parsed first-argument shape, so a bare
# ``NumberLitArg`` standing in for ``5`` is a valid stand-in for the
# entire expression.
_NUMBER_CHARS: Final[str] = string.digits


# The canonical category emitted by the scheduler recognizer.
_SCHEDULED_EVENT: Final[AbstractInputCategory] = AbstractInputCategory.SCHEDULED_EVENT


# Canonical markers that must appear in descriptions per design.
_CRON_LIBRARY_TOKEN: Final[str] = "cron"
_TIME_LIBRARY_TOKEN: Final[str] = "time"
_SECONDS_PRECISION_TOKEN: Final[str] = "seconds-precision"
_MINUTE_PRECISION_TOKEN: Final[str] = "minute-precision"
_DYNAMIC_TOKEN: Final[str] = "<dynamic>"
_MALFORMED_TOKEN: Final[str] = "malformed or unsupported scheduler shape"


# A small set of fx / viper method names sufficient to demonstrate that
# the recognizer's dispatch-boundary exclusion holds. The Requirement
# 12 / 13 carve-out applies to *any* method on these receivers, so
# sampling a handful is enough to make the property's invariant
# observable: the output is unchanged when these calls are present or
# absent.
_FX_METHODS: Final[tuple[str, ...]] = (
    "Provide",
    "Invoke",
    "Module",
    "New",
    "Annotate",
)
_VIPER_METHODS: Final[tuple[str, ...]] = (
    "New",
    "GetString",
    "BindEnv",
    "SetConfigName",
    "AddConfigPath",
)


# ---------------------------------------------------------------------------
# Per-event strategies
# ---------------------------------------------------------------------------


_ident = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8)


@st.composite
def _schedule_literal(draw: st.DrawFn) -> str:
    """Generate a non-empty schedule string.

    The recognizer accepts any string-literal verbatim, so the body of
    the generated literal does not need to be a valid cron expression.
    Substituting any all-whitespace draw with a sentinel non-blank
    value avoids wasting shrinking budget on a literal that the
    description's strict-substring assertion would still accept (a
    non-empty string after f-string interpolation is still detectable
    in the output).
    """

    text = draw(st.text(alphabet=_SCHEDULE_CHARS, min_size=1, max_size=30))
    if not text.strip():
        # Reduce shrinking pressure by collapsing all-whitespace draws
        # to a canonical non-blank schedule. Verbatim recording is the
        # property under test; whether the source contained leading
        # whitespace is irrelevant to the property.
        text = "* * * * *"
    return text


_receiver_ident = _ident
"""Identifier-typed receivers like ``c`` in ``c.AddFunc(...)``."""


_dotted_segments = st.lists(_ident, min_size=2, max_size=4).map(tuple)


_unknown_text = st.text(alphabet=_SCHEDULE_CHARS, min_size=1, max_size=10)


def _make_cron_new_event(
    *,
    with_seconds: bool,
    file_path: str,
    line: int,
) -> MethodCallEvent:
    """Build a ``cron.New(...)`` event.

    ``with_seconds`` controls whether the call's argument list carries a
    nested ``cron.WithSeconds()`` ``CallArg``, which is the exact
    structural signal the recognizer matches on. The same identifier
    cannot appear with both shapes in a single file, so generators
    that want to exercise both branches do so across distinct files.
    """

    args: tuple[ArgRef, ...]
    if with_seconds:
        with_seconds_call = MethodCallEvent(
            receiver_chain=("cron",),
            method_name="WithSeconds",
            args=(),
            file_path=file_path,
            line=line,
        )
        args = (CallArg(call=with_seconds_call),)
    else:
        args = ()

    return MethodCallEvent(
        receiver_chain=("cron",),
        method_name="New",
        args=args,
        file_path=file_path,
        line=line,
    )


@st.composite
def _cron_addfunc_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str | None, bool]:
    """Generate a ``<v>.AddFunc(...)`` / ``<v>.AddJob(...)`` event.

    Returns ``(event, literal_schedule, malformed_due_to_arg)``:

    * ``literal_schedule`` is the verbatim schedule string when the
      first argument is a ``StringLitArg``; ``None`` for every other
      argument shape (in which case the description must carry
      ``<dynamic>``).
    * ``malformed_due_to_arg`` is ``True`` when the schedule argument
      is of an unrecognized shape (``NumberLitArg``, ``StructLitArg``,
      ``UnknownArg``, or missing), independently of whether the file
      contains a ``cron.New(...)`` call. The caller pairs this with
      the file-level cron-awareness flag to compute the expected
      malformed verdict.
    """

    method = draw(st.sampled_from(("AddFunc", "AddJob")))
    recv = draw(_receiver_ident)
    line = draw(st.integers(min_value=1, max_value=999))

    arg_kind = draw(
        st.sampled_from(
            (
                "string",
                "dotted",
                "ident",
                "number",
                "struct",
                "unknown",
                "missing",
            ),
        ),
    )

    literal_schedule: str | None = None
    malformed_due_to_arg = False
    args: tuple[ArgRef, ...]

    if arg_kind == "string":
        schedule = draw(_schedule_literal())
        literal_schedule = schedule
        args = (StringLitArg(value=schedule),)
    elif arg_kind == "dotted":
        args = (DottedArg(parts=draw(_dotted_segments)),)
    elif arg_kind == "ident":
        args = (IdentArg(name=draw(_ident)),)
    elif arg_kind == "number":
        args = (NumberLitArg(text=draw(st.text(alphabet=_NUMBER_CHARS, min_size=1, max_size=4))),)
        malformed_due_to_arg = True
    elif arg_kind == "struct":
        struct = StructLitEvent(
            type_name="X",
            package_alias=None,
            fields=(),
            is_pointer=False,
            file_path=file_path,
            line=line,
        )
        args = (StructLitArg(event=struct),)
        malformed_due_to_arg = True
    elif arg_kind == "unknown":
        args = (UnknownArg(text=draw(_unknown_text)),)
        malformed_due_to_arg = True
    else:  # "missing"
        args = ()
        malformed_due_to_arg = True

    # The recognized cron pattern requires the receiver chain to be a
    # single identifier (the alias to which ``cron.New(...)`` was
    # assigned). Multi-segment receivers (``foo.bar.AddFunc(...)``)
    # never match the cron shape; they always fall into the malformed
    # branch, which is the design's "receiver was not recognized as a
    # ``*cron.Cron`` value" case.
    event = MethodCallEvent(
        receiver_chain=(recv,),
        method_name=method,
        args=args,
        file_path=file_path,
        line=line,
    )
    return event, literal_schedule, malformed_due_to_arg


@st.composite
def _time_scheduler_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> tuple[MethodCallEvent, str | None]:
    """Generate a ``time.NewTicker(...)`` / ``time.AfterFunc(...)`` event.

    Returns ``(event, literal_schedule)`` where ``literal_schedule`` is
    the verbatim schedule string for ``StringLitArg`` arguments and
    ``None`` for every other argument shape. The recognizer never
    flags the time-package shapes as malformed (Requirement 4.5
    applies only to ``AddFunc`` / ``AddJob``), so no second flag is
    needed.
    """

    method = draw(st.sampled_from(("NewTicker", "AfterFunc")))
    line = draw(st.integers(min_value=1, max_value=999))

    arg_kind = draw(st.sampled_from(("string", "dotted", "number", "missing")))
    literal_schedule: str | None = None
    args: tuple[ArgRef, ...]

    if arg_kind == "string":
        schedule = draw(_schedule_literal())
        literal_schedule = schedule
        args = (StringLitArg(value=schedule),)
    elif arg_kind == "dotted":
        args = (DottedArg(parts=draw(_dotted_segments)),)
    elif arg_kind == "number":
        # ``NumberLitArg`` is the canonical stand-in for a Go duration
        # literal at the recognizer boundary; the recognizer records
        # the literal's raw text in the description.
        text = draw(st.text(alphabet=_NUMBER_CHARS, min_size=1, max_size=4))
        literal_schedule = text
        args = (NumberLitArg(text=text),)
    else:  # missing
        args = ()

    event = MethodCallEvent(
        receiver_chain=("time",),
        method_name=method,
        args=args,
        file_path=file_path,
        line=line,
    )
    return event, literal_schedule


@st.composite
def _fx_or_viper_noise_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a noise event excluded by the fx / viper dispatch-boundary skip.

    The recognizer must treat these as inert (Requirements 12.1, 12.3,
    13.1, 13.2). The noise's method-name selection deliberately
    overlaps with the cron / time vocabulary (``New``, ``AddFunc``,
    ``NewTicker``) so a regression that forgets to honor the
    receiver-chain exclusion would surface immediately: the noise
    event would otherwise match the recognizer's method-name gate.
    """

    receiver = draw(st.sampled_from(("fx", "viper")))
    if receiver == "fx":
        method = draw(st.sampled_from(_FX_METHODS + ("AddFunc", "AddJob", "NewTicker")))
    else:
        method = draw(st.sampled_from(_VIPER_METHODS + ("AddFunc", "NewTicker")))

    line = draw(st.integers(min_value=1, max_value=999))
    # Provide a string-literal first argument that, if it were ever
    # used, would land in the description; the property's assertion
    # that the output is unchanged after stripping these events
    # implicitly verifies the string is *not* in the output.
    arg_text = draw(st.text(alphabet=_SCHEDULE_CHARS, min_size=1, max_size=10))
    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name=method,
        args=(StringLitArg(value=arg_text),),
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
    bool | None,
    list[tuple[MethodCallEvent, str | None, bool]],
    list[tuple[MethodCallEvent, str | None]],
]:
    """Generate one file's event list plus the expected-output metadata.

    Returns a tuple ``(events, cron_seconds_precision, cron_calls,
    time_calls)``:

    * ``events`` is the order-preserving event list assigned to the
      file's key in the ``events_by_file`` mapping. Always carries
      one or two fx / viper noise events so the exclusion property
      below is exercised on every example.
    * ``cron_seconds_precision`` is ``True`` when the file contains
      at least one ``cron.New(cron.WithSeconds())`` call, ``False``
      when the file contains a plain ``cron.New(...)`` call, and
      ``None`` when the file has no ``cron.New(...)``. This is the
      file-level marker the recognizer derives in
      ``_detect_cron_seconds_precision``.
    * ``cron_calls`` and ``time_calls`` are the per-call records the
      expected-output computation needs to predict the recognizer's
      output. Their event order matches insertion into ``events``,
      which lets the test assert per-emission shape per file.
    """

    # File-level cron-awareness: three branches so the property
    # exercises file-without-cron (malformed), file-with-cron, and
    # file-with-cron-and-seconds independently.
    cron_kind = draw(st.sampled_from(("no_cron", "cron_minute", "cron_seconds")))
    if cron_kind == "no_cron":
        seconds_precision: bool | None = None
    elif cron_kind == "cron_minute":
        seconds_precision = False
    else:
        seconds_precision = True

    events: list[GoEvent] = []

    # Place the ``cron.New(...)`` call early so the per-file pre-scan
    # in ``_detect_cron_seconds_precision`` can find it before any
    # ``AddFunc`` / ``AddJob`` is walked. (The recognizer scans the
    # whole event list before walking, so the order is purely
    # cosmetic for correctness, but matching a realistic source order
    # keeps fixtures readable.)
    if seconds_precision is not None:
        events.append(
            _make_cron_new_event(
                with_seconds=seconds_precision,
                file_path=path,
                line=1,
            ),
        )

    # Sprinkle one fx / viper noise event so the exclusion property is
    # exercised on every generated file.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    # Per-file cron calls: between zero and three to exercise both the
    # "no recognized cron calls" and "multiple recognized cron calls"
    # branches. Each call's expected metadata is captured for the
    # assertion phase.
    cron_calls = draw(
        st.lists(_cron_addfunc_event(file_path=path), min_size=0, max_size=3),
    )
    for ev, _, _ in cron_calls:
        events.append(ev)

    # Per-file time calls.
    time_calls = draw(
        st.lists(_time_scheduler_event(file_path=path), min_size=0, max_size=2),
    )
    for ev, _ in time_calls:
        events.append(ev)

    # One trailing noise event in case the noise's position relative to
    # the recognized events ever matters (it must not).
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    return events, seconds_precision, cron_calls, time_calls


@st.composite
def _events_by_file(
    draw: st.DrawFn,
) -> tuple[
    dict[str, list["GoEvent"]],
    dict[str, tuple[
        bool | None,
        list[tuple[MethodCallEvent, str | None, bool]],
        list[tuple[MethodCallEvent, str | None]],
    ]],
]:
    """Generate a multi-file events mapping plus per-file expectation metadata.

    Returns ``(events_by_file, per_file_metadata)``. The
    ``per_file_metadata`` dict mirrors ``events_by_file`` keys and
    carries the four-tuple needed to compute the expected recognizer
    output: ``(cron_seconds_precision, cron_calls, time_calls)``.
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
            bool | None,
            list[tuple[MethodCallEvent, str | None, bool]],
            list[tuple[MethodCallEvent, str | None]],
        ],
    ] = {}

    for name in names:
        path = f"{name}.go"
        events, seconds_precision, cron_calls, time_calls = draw(
            _file_events(path=path),
        )
        events_by_file[path] = events
        metadata[path] = (seconds_precision, cron_calls, time_calls)

    return events_by_file, metadata


# ---------------------------------------------------------------------------
# Expected-output computation
# ---------------------------------------------------------------------------


def _expected_description_for_cron(
    event: MethodCallEvent,
    literal_schedule: str | None,
    malformed_due_to_arg: bool,
    file_seconds_precision: bool | None,
) -> str:
    """Compute the description a cron AddFunc/AddJob call must produce.

    Mirrors the recognizer's branching logic without re-deriving the
    f-string templates: the property assertions verify a strict subset
    of the description (markers + verbatim schedule + location) rather
    than full-string equality, so the test stays robust to wording
    refinements while still pinning the load-bearing tokens.
    """

    recv = ".".join(event.receiver_chain) if event.receiver_chain else "<unqualified>"
    location = f"at {event.file_path}:{event.line}"

    # Receiver not recognized (no cron.New in file or chain shape wrong),
    # or schedule shape unsupported → malformed branch.
    cron_recognized = file_seconds_precision is not None and len(event.receiver_chain) == 1
    if not cron_recognized or malformed_due_to_arg:
        return (
            f"scheduled {_DYNAMIC_TOKEN} "
            f"({_MALFORMED_TOKEN}: "
            f"{event.method_name}({recv}) {location})"
        )

    # Recognized cron call: literal verbatim or <dynamic>.
    schedule_token = literal_schedule if literal_schedule is not None else _DYNAMIC_TOKEN
    precision = (
        _SECONDS_PRECISION_TOKEN if file_seconds_precision else _MINUTE_PRECISION_TOKEN
    )
    return (
        f"scheduled ({_CRON_LIBRARY_TOKEN}, {precision}) {schedule_token} "
        f"via {recv}.{event.method_name}() {location}"
    )


def _expected_description_for_time(
    event: MethodCallEvent,
    literal_schedule: str | None,
) -> str:
    """Compute the description a ``time`` scheduler call must produce."""

    schedule_token = literal_schedule if literal_schedule is not None else _DYNAMIC_TOKEN
    return (
        f"scheduled ({_TIME_LIBRARY_TOKEN}, {event.method_name}) {schedule_token} "
        f"via time.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_inputs(
    metadata: dict[
        str,
        tuple[
            bool | None,
            list[tuple[MethodCallEvent, str | None, bool]],
            list[tuple[MethodCallEvent, str | None]],
        ],
    ],
) -> list[AbstractInput]:
    """Compute the deduplicated expected ``AbstractInput`` list.

    Iteration order matches the recognizer's: paths in sorted order,
    and within each file the order in which cron and time events were
    appended to the events list (cron calls first, time calls second,
    per the file-fixture's construction).
    """

    expected: list[AbstractInput] = []
    seen: set[tuple[AbstractInputCategory, str]] = set()

    for path in sorted(metadata):
        seconds_precision, cron_calls, time_calls = metadata[path]
        for event, literal_schedule, malformed in cron_calls:
            desc = _expected_description_for_cron(
                event=event,
                literal_schedule=literal_schedule,
                malformed_due_to_arg=malformed,
                file_seconds_precision=seconds_precision,
            )
            key = (_SCHEDULED_EVENT, desc)
            if key in seen:
                continue
            seen.add(key)
            expected.append(
                AbstractInput(category=_SCHEDULED_EVENT, description=desc),
            )
        for event, literal_schedule in time_calls:
            desc = _expected_description_for_time(
                event=event,
                literal_schedule=literal_schedule,
            )
            key = (_SCHEDULED_EVENT, desc)
            if key in seen:
                continue
            seen.add(key)
            expected.append(
                AbstractInput(category=_SCHEDULED_EVENT, description=desc),
            )

    return expected


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
def test_extract_schedulers_matches_expected_output(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool | None,
                list[tuple[MethodCallEvent, str | None, bool]],
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """Property 6 main invariant: one input per recognized registration.

    Validates Requirements 4.1, 4.2, 4.3, 4.4, 4.5 jointly. The
    expected output is computed by mirroring the recognizer's decision
    tree on the per-file generator metadata, then asserting
    full-list equality. Per-emission invariants validated by
    construction:

    * Exactly one ``AbstractInput`` per recognized cron AddFunc / AddJob
      and per time NewTicker / AfterFunc.
    * Cron-style malformed shapes still emit one input (Requirement 4.5).
    * String-literal schedules appear verbatim in the description
      (Requirement 4.3); non-literal schedules appear as ``<dynamic>``.
    * The library marker (``cron`` or ``time``) and the precision
      marker (``seconds-precision`` / ``minute-precision``) appear in
      the description per Requirement 4.2.
    * The Source_Location suffix ``at <file>:<line>`` appears on every
      emission per Requirement 4.4.
    * The category is always ``scheduled_event`` per Requirement 4.1.
    """

    events_by_file, metadata = case

    actual = _extract_schedulers(events_by_file)
    expected = _expected_inputs(metadata)

    assert actual == expected, (
        "scheduler recognizer output diverged from expected:\n"
        f"  actual:   {actual!r}\n"
        f"  expected: {expected!r}"
    )

    # Universal invariants on the actual output, independent of the
    # expected-output computation. These would catch a regression that
    # made the recognizer's output structurally wrong even when the
    # full-list equality passed (e.g. by a coincidental two-bug
    # cancellation in the expected-output mirror).
    for entry in actual:
        assert entry.category is _SCHEDULED_EVENT, (
            f"entry {entry!r} carries category {entry.category!r}; "
            f"the scheduler recognizer must always emit scheduled_event"
        )
        # Every description carries the Source_Location suffix
        # ``at <file>:<line>`` per Requirement 4.4.
        assert " at " in entry.description, (
            f"entry {entry!r} is missing the 'at <file>:<line>' suffix; "
            f"Requirement 4.4 mandates a Source_Location on every "
            f"emission"
        )
        # Every description carries a library marker — either cron or
        # time — per Requirement 4.2. Mutually exclusive: a single
        # entry cannot be both, since the library token follows the
        # ``scheduled (`` prefix in both shapes.
        has_cron = f"({_CRON_LIBRARY_TOKEN}," in entry.description
        has_time = f"({_TIME_LIBRARY_TOKEN}," in entry.description
        has_malformed = _MALFORMED_TOKEN in entry.description
        # A malformed cron entry skips the ``(cron, ...)`` prefix and
        # instead carries the canonical malformed marker; that's the
        # third valid shape.
        assert has_cron or has_time or has_malformed, (
            f"entry {entry!r} does not name a recognized library "
            f"marker (cron, time) or carry the malformed marker; "
            f"Requirement 4.2 mandates one of these"
        )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_schedulers_preserves_string_literal_schedule_verbatim(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool | None,
                list[tuple[MethodCallEvent, str | None, bool]],
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """Property 6 verbatim-preservation invariant.

    Validates Requirement 4.3 first sentence: a literal six-field cron
    expression that a five-field parser would reject must still appear
    in the description exactly as the source declared. The test
    asserts the literal substring is present in *some* output
    description for every recognized cron / time call whose first
    argument is a ``StringLitArg``.
    """

    events_by_file, metadata = case
    actual = _extract_schedulers(events_by_file)
    descriptions = [entry.description for entry in actual]

    for path in sorted(metadata):
        seconds_precision, cron_calls, time_calls = metadata[path]
        cron_recognized = seconds_precision is not None
        for event, literal_schedule, malformed in cron_calls:
            if literal_schedule is None:
                continue
            # Only recognized, non-malformed cron calls carry the
            # literal verbatim. The malformed branch substitutes the
            # ``<dynamic>`` token regardless of the original argument
            # (per design's canonical malformed description shape).
            if not cron_recognized or malformed:
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            matched = any(
                literal_schedule in desc and location_suffix in desc
                for desc in descriptions
            )
            assert matched, (
                f"string-literal cron schedule {literal_schedule!r} for "
                f"{event.file_path}:{event.line} not preserved verbatim "
                f"in any output description; descriptions: "
                f"{descriptions!r}"
            )
        for event, literal_schedule in time_calls:
            if literal_schedule is None:
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            matched = any(
                literal_schedule in desc and location_suffix in desc
                for desc in descriptions
            )
            assert matched, (
                f"string-literal time schedule {literal_schedule!r} for "
                f"{event.file_path}:{event.line} not preserved verbatim "
                f"in any output description; descriptions: "
                f"{descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_schedulers_ignores_fx_and_viper_receivers(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool | None,
                list[tuple[MethodCallEvent, str | None, bool]],
                list[tuple[MethodCallEvent, str | None]],
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
    with_noise = _extract_schedulers(events_by_file)
    without_noise = _extract_schedulers(_stripped_of_fx_viper(events_by_file))
    assert with_noise == without_noise, (
        "scheduler recognizer output changed after removing fx/viper "
        "MethodCallEvents; the dispatch-boundary exclusion must be "
        "inert (Requirements 12.1, 12.3, 13.1, 13.2)"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_schedulers_is_iteration_order_independent(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                bool | None,
                list[tuple[MethodCallEvent, str | None, bool]],
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """Determinism invariant: output depends only on input contents.

    Validates Requirement 11.4. The recognizer sorts paths internally,
    so two mappings with the same keys and values but different
    insertion orders must produce identical output lists.
    """

    events_by_file, _ = case
    reversed_mapping = dict(reversed(list(events_by_file.items())))
    assert _extract_schedulers(events_by_file) == _extract_schedulers(reversed_mapping), (
        "scheduler recognizer output depends on insertion order; "
        "Requirement 11.4 mandates deterministic, path-sorted iteration"
    )

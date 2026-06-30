# ruff: noqa: E501
# Feature: go-analyzer-support, Property 7: ActiveMQ consumer/publisher emit exactly one I/O entry per call site, preserving literal destinations.
"""Property test for the Go ActiveMQ-call recognizer.

**Validates Requirements 5.1, 5.2, 5.3, 5.4, 5.5** (Property 7 in the
design, task 7.6 in ``tasks.md``).

The ActiveMQ recognizer is the second of five recognizers composed by
``extract_go_io`` (task 7.11). This property test exercises its
package-internal entry point :func:`_extract_activemq_io` directly so
the per-recognizer contract is pinned independently of the eventual
composition step.

The properties below capture the design's documented contract:

1. **One AbstractInput per matched Subscribe call site**
   (Requirement 5.1 + Property 7 statement). Every ``Subscribe`` call
   whose third positional argument is a
   ``StructLitArg(domain.SubscriberConfig{...})`` produces exactly one
   ``AbstractInput`` with ``category=message_consumed``. Calls whose
   third argument fails the struct-literal type/alias gate produce no
   emission and are not recorded.

2. **One AbstractOutput per matched SendMessage call site**
   (Requirement 5.2). Every ``SendMessage`` call whose fourth
   positional argument is a ``StructLitArg(domain.Message{...})``
   produces exactly one ``AbstractOutput`` with
   ``category=message_published``. Same fall-through behavior for
   non-matching shapes.

3. **String-literal ``Destination`` values are recorded verbatim**
   (Requirement 5.4 first sentence). The literal token recorded in
   the source is the exact substring in the description, with no
   redaction, normalization, or truncation. Every other expression
   shape — field access (``DottedArg``), bare identifier
   (``IdentArg``), nested call (``CallArg``), opaque text
   (``UnknownArg``), or an absent ``Destination`` field — is recorded
   as ``<dynamic>`` (Requirement 5.4 second sentence).

4. **Source_Location is attached to every emission** (Requirement
   5.5). The recognizer encodes the ``MethodCallEvent``'s ``file_path``
   and 1-indexed ``line`` into the description as a
   ``at <file>:<line>`` suffix, matching the HTTP and scheduler
   recognizers' convention.

5. **``activemq.NewClient(&activemq.JmsConfig{...})`` produces no I/O**
   (Requirement 5.3 third bullet). The connection-setup call is
   acknowledged by the external-services detector, not the I/O
   extractor. Adding arbitrary ``activemq.NewClient`` calls to the
   event stream must not change the recognizer's output.

6. **fx and viper exclusions are inert at the dispatch boundary**
   (Requirements 12.1, 12.3, 13.1, 13.2). Adding arbitrary
   ``fx.<method>`` and ``viper.<method>`` calls to a file must not
   change the recognizer's output.

7. **Deterministic, path-sorted iteration** (Requirement 11.4). The
   recognizer is a pure function of its input mapping. Shuffling the
   mapping's insertion order must not change the output lists.

The recognizer's :func:`_extract_activemq_io` is package-internal; the
test imports it through the module path to mirror the convention used
by ``test_property_06_go_scheduler_detection.py`` for
:func:`_extract_schedulers`.
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
from project_knowledge_mcp.project_analyzer.go.go_io import _extract_activemq_io

if TYPE_CHECKING:
    from project_knowledge_mcp.project_analyzer.go._events import ArgRef, GoEvent


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets and shared constants
# ---------------------------------------------------------------------------


# Destination-literal alphabet. Restricted to characters that survive the
# description's f-string interpolation unmodified — letters, digits, and
# a few separator characters commonly observed in ActiveMQ queue names
# (``REP.SERVICE.PAYMENT.ERR`` is the canonical sample in
# ``repayment_service``). ``\n``, ``\r``, and ``\t`` are excluded so a
# generated destination cannot break the ``at <file>:<line>`` suffix's
# line orientation. Spaces are excluded so the verbatim-preservation
# property can use a strict substring match.
_DESTINATION_CHARS: Final[str] = string.ascii_letters + string.digits + "._/-"


# Identifier alphabet for receiver names, file basenames, and dotted-arg
# segments. Constrained to characters Go accepts in identifiers so the
# fixtures look like real source.
_IDENT_CHARS: Final[str] = string.ascii_letters + string.digits + "_"


# Numeric-literal alphabet for filler argument shapes. The recognizer
# only inspects the index-2 (Subscribe) or index-3 (SendMessage)
# argument's struct-literal-ness, so the surrounding filler arguments'
# concrete shapes are arbitrary; a small numeric alphabet keeps the
# fixtures readable.
_NUMBER_CHARS: Final[str] = string.digits


# Canonical category enum values the recognizer always emits.
_MESSAGE_CONSUMED: Final[AbstractInputCategory] = AbstractInputCategory.MESSAGE_CONSUMED
_MESSAGE_PUBLISHED: Final[AbstractOutputCategory] = AbstractOutputCategory.MESSAGE_PUBLISHED


# Canonical description fragments — anchors the assertions verify on
# every emission so a regression that drops the library marker or the
# action token is caught even when the full-string equality also fails.
_ACTIVEMQ_LIBRARY_TOKEN: Final[str] = "activemq"
_CONSUMED_ACTION: Final[str] = "consumed from"
_PUBLISHED_ACTION: Final[str] = "published to"
_DYNAMIC_TOKEN: Final[str] = "<dynamic>"


# The two method names the recognizer matches. Recognition is by method
# name alone; the receiver type is not resolved. ``NewClient`` is
# explicitly *not* in this set so adding ``NewClient`` events to the
# stream cannot change the output.
_SUBSCRIBE_METHOD: Final[str] = "Subscribe"
_SEND_MESSAGE_METHOD: Final[str] = "SendMessage"


# The struct-literal type names + package alias that the recognizer
# requires at the inspected positional indices. The four sample
# repositories always import ``esb-go-libs/activemq/domain`` under the
# alias ``domain``; the recognizer's package-alias check excludes any
# other aliased or unqualified ``Message`` / ``SubscriberConfig`` type.
_SUBSCRIBER_CONFIG_TYPE: Final[str] = "SubscriberConfig"
_MESSAGE_TYPE: Final[str] = "Message"
_DOMAIN_ALIAS: Final[str] = "domain"


# A small set of fx / viper method names sufficient to demonstrate that
# the recognizer's dispatch-boundary exclusion holds. The Requirement
# 12 / 13 carve-out applies to *any* method on these receivers, so
# sampling a handful is enough to make the property's invariant
# observable: the output is unchanged when these calls are present or
# absent. Method names that overlap with the recognized vocabulary
# (``Subscribe``, ``SendMessage``) are included so a regression that
# forgets to honor the receiver-chain exclusion would surface
# immediately.
_FX_METHODS: Final[tuple[str, ...]] = (
    "Provide",
    "Invoke",
    "Module",
    "New",
    "Subscribe",
    "SendMessage",
)
_VIPER_METHODS: Final[tuple[str, ...]] = (
    "New",
    "GetString",
    "BindEnv",
    "Subscribe",
    "SendMessage",
)


# ---------------------------------------------------------------------------
# Per-event strategies
# ---------------------------------------------------------------------------


_ident = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=8)
_destination_literal = st.text(
    alphabet=_DESTINATION_CHARS, min_size=1, max_size=20,
)
_dotted_segments = st.lists(_ident, min_size=2, max_size=4).map(tuple)
_number_literal = st.text(alphabet=_NUMBER_CHARS, min_size=1, max_size=4)
_unknown_text = st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=10)


def _filler_arg(name: str) -> "ArgRef":
    """Return a generic positional-filler argument standing in for ``ctx`` / ``handler`` etc.

    The recognizer never inspects positions outside the documented
    index-2 (Subscribe) or index-3 (SendMessage) slot, so the filler
    shape is irrelevant to the recognizer's decisions. A bare
    :class:`IdentArg` keeps the fixtures readable and matches the
    natural Go source shape.
    """

    return IdentArg(name=name)


@st.composite
def _destination_field_value(draw: st.DrawFn) -> tuple["ArgRef | None", str]:
    """Generate a ``Destination``-field value and its expected description fragment.

    Returns ``(value_or_none, expected_destination)``:

    * ``value_or_none`` is the ``ArgRef`` to attach as the
      ``Destination`` field's value, or ``None`` to omit the field
      entirely (testing the "absent ``Destination``" branch of
      Requirement 5.4 / the recognizer's
      :func:`_extract_destination_field` fallback).
    * ``expected_destination`` is the substring the recognizer must
      include in the description — the verbatim literal for
      ``StringLitArg``, the canonical ``<dynamic>`` placeholder for
      every other variant (field access, identifier, nested call,
      opaque text, absent field).
    """

    kind = draw(
        st.sampled_from(
            ("string", "dotted", "ident", "call", "unknown", "number", "absent"),
        ),
    )
    if kind == "string":
        value = draw(_destination_literal)
        return StringLitArg(value=value), value
    if kind == "dotted":
        return DottedArg(parts=draw(_dotted_segments)), _DYNAMIC_TOKEN
    if kind == "ident":
        return IdentArg(name=draw(_ident)), _DYNAMIC_TOKEN
    if kind == "call":
        # A nested method call on an arbitrary receiver. The recognizer
        # never inspects the inner call's identity for this field, so
        # the inner call's shape is uninteresting; we use a benign
        # ``cfg.Get("x")`` stand-in.
        nested = MethodCallEvent(
            receiver_chain=(draw(_ident),),
            method_name=draw(_ident),
            args=(),
            file_path="dummy.go",
            line=1,
        )
        return CallArg(call=nested), _DYNAMIC_TOKEN
    if kind == "unknown":
        return UnknownArg(text=draw(_unknown_text)), _DYNAMIC_TOKEN
    if kind == "number":
        # Numeric destination is observationally absurd but the
        # recognizer's classification still applies: any non-string
        # literal is recorded as ``<dynamic>``. Generating this branch
        # guards against a regression that only forwarded ``StringLitArg``
        # vs ``DottedArg`` without considering other arg variants.
        return NumberLitArg(text=draw(_number_literal)), _DYNAMIC_TOKEN
    # absent: the ``Destination`` field is missing from the composite
    # literal entirely. The recognizer's fallback recovers the
    # ``<dynamic>`` placeholder.
    return None, _DYNAMIC_TOKEN


@st.composite
def _struct_lit(
    draw: st.DrawFn,
    *,
    type_name: str,
    package_alias: str | None,
    file_path: str,
    line: int,
) -> tuple[StructLitEvent, str]:
    """Build a composite literal carrying a ``Destination`` field.

    Returns ``(event, expected_destination)``. The literal's
    ``type_name`` and ``package_alias`` are passed in by the caller so
    the same builder can produce matching (``SubscriberConfig`` /
    ``Message`` under the ``domain`` alias) and non-matching (any
    other type or alias) shapes from a single strategy.

    ``is_pointer`` is randomized because the recognizer's contract
    treats both pointer and value composite literals the same way — the
    type-name and alias gates do not consult ``is_pointer``.
    """

    field_value, expected_destination = draw(_destination_field_value())
    fields: tuple[tuple[str, "ArgRef"], ...]
    if field_value is None:
        # Optionally include an unrelated field so the struct literal
        # is non-empty but still has no ``Destination`` — covers the
        # "destination absent among other fields" branch.
        if draw(st.booleans()):
            fields = (
                (
                    "OtherField",
                    IdentArg(name=draw(_ident)),
                ),
            )
        else:
            fields = ()
    else:
        fields = (("Destination", field_value),)

    return (
        StructLitEvent(
            type_name=type_name,
            package_alias=package_alias,
            fields=fields,
            is_pointer=draw(st.booleans()),
            file_path=file_path,
            line=line,
        ),
        expected_destination,
    )


# The shape kinds a Subscribe / SendMessage call can take. Each is named
# so the assertion-side error messages stay legible.
_VALID_SHAPE: Final[str] = "valid"
_WRONG_TYPE_SHAPE: Final[str] = "wrong_type"
_WRONG_ALIAS_SHAPE: Final[str] = "wrong_alias"
_UNQUALIFIED_SHAPE: Final[str] = "unqualified"
_NON_STRUCT_SHAPE: Final[str] = "non_struct"
_MISSING_ARG_SHAPE: Final[str] = "missing_arg"


_CALL_SHAPES: Final[tuple[str, ...]] = (
    _VALID_SHAPE,
    _WRONG_TYPE_SHAPE,
    _WRONG_ALIAS_SHAPE,
    _UNQUALIFIED_SHAPE,
    _NON_STRUCT_SHAPE,
    _MISSING_ARG_SHAPE,
)


@st.composite
def _activemq_call(
    draw: st.DrawFn,
    *,
    method: str,
    matched_type_name: str,
    inspected_index: int,
    total_args: int,
    file_path: str,
) -> tuple[MethodCallEvent, str | None]:
    """Generate a Subscribe or SendMessage event and its expected destination.

    Returns ``(event, expected_destination_or_None)``:

    * ``expected_destination_or_None`` is the substring the recognizer
      must place in the description when the call matches the gate,
      or ``None`` when the call must produce no emission.

    The non-matching branches exhaustively exercise the recognizer's
    rejection paths documented in :func:`_struct_lit_arg_of_type`:

    * ``wrong_type``: composite literal carries a different
      ``type_name`` (so the type-name gate fails).
    * ``wrong_alias``: composite literal carries an alias other than
      ``domain`` (the package-alias gate fails).
    * ``unqualified``: composite literal has ``package_alias=None``
      (bare ``Message{...}`` form — fails the alias gate).
    * ``non_struct``: the inspected positional argument is not a
      ``StructLitArg`` at all (e.g. an identifier or a nested call).
    * ``missing_arg``: the inspected positional index is past the end
      of the argument tuple (truncated argument list).
    """

    recv = draw(_ident)
    line = draw(st.integers(min_value=1, max_value=999))
    shape = draw(st.sampled_from(_CALL_SHAPES))

    args: list[ArgRef] = [_filler_arg(f"a{i}") for i in range(total_args)]

    expected: str | None
    if shape == _VALID_SHAPE:
        struct_event, dest = draw(
            _struct_lit(
                type_name=matched_type_name,
                package_alias=_DOMAIN_ALIAS,
                file_path=file_path,
                line=line,
            ),
        )
        args[inspected_index] = StructLitArg(event=struct_event)
        expected = dest
    elif shape == _WRONG_TYPE_SHAPE:
        # Any type name other than the matched one; appending a suffix
        # to the matched name is the simplest way to guarantee
        # divergence even if the matched name happens to be a substring
        # of the alphabet.
        bad_type = matched_type_name + "_x"
        struct_event, _ = draw(
            _struct_lit(
                type_name=bad_type,
                package_alias=_DOMAIN_ALIAS,
                file_path=file_path,
                line=line,
            ),
        )
        args[inspected_index] = StructLitArg(event=struct_event)
        expected = None
    elif shape == _WRONG_ALIAS_SHAPE:
        # Any alias other than ``domain``. ``activemq`` is the natural
        # confusable here — pointer-typed ``&activemq.SubscriberConfig{...}``
        # is a shape that exists in the codebase but is *not* the
        # ``domain``-aliased form the recognizer matches on.
        struct_event, _ = draw(
            _struct_lit(
                type_name=matched_type_name,
                package_alias="activemq",
                file_path=file_path,
                line=line,
            ),
        )
        args[inspected_index] = StructLitArg(event=struct_event)
        expected = None
    elif shape == _UNQUALIFIED_SHAPE:
        # Bare ``Message{...}`` / ``SubscriberConfig{...}`` with no
        # package alias. Requirement 5.3 requires the ``domain`` alias
        # explicitly, so this shape produces no emission.
        struct_event, _ = draw(
            _struct_lit(
                type_name=matched_type_name,
                package_alias=None,
                file_path=file_path,
                line=line,
            ),
        )
        args[inspected_index] = StructLitArg(event=struct_event)
        expected = None
    elif shape == _NON_STRUCT_SHAPE:
        # Any argument shape other than ``StructLitArg``: an identifier
        # holding a pre-built config, a dotted field reference, or
        # whatever else might appear in real source. A regression that
        # accepted ``IdentArg`` as a stand-in for the struct literal
        # would surface here.
        nonstruct_kind = draw(
            st.sampled_from(
                ("ident", "dotted", "unknown", "string"),
            ),
        )
        if nonstruct_kind == "ident":
            args[inspected_index] = IdentArg(name=draw(_ident))
        elif nonstruct_kind == "dotted":
            args[inspected_index] = DottedArg(parts=draw(_dotted_segments))
        elif nonstruct_kind == "unknown":
            args[inspected_index] = UnknownArg(text=draw(_unknown_text))
        else:
            args[inspected_index] = StringLitArg(value=draw(_destination_literal))
        expected = None
    else:  # _MISSING_ARG_SHAPE
        # Truncate the argument list so the inspected index is past the
        # end. The recognizer's bounds check rejects the call without
        # an emission.
        args = args[:inspected_index]
        expected = None

    event = MethodCallEvent(
        receiver_chain=(recv,),
        method_name=method,
        args=tuple(args),
        file_path=file_path,
        line=line,
    )
    return event, expected


def _subscribe_event(
    *,
    file_path: str,
) -> st.SearchStrategy[tuple[MethodCallEvent, str | None]]:
    """Strategy producing a ``<recv>.Subscribe(...)`` call event.

    Index 2 is the inspected slot per Requirement 5.3 first bullet;
    the call is generated with four positional arguments so the
    canonical ``Subscribe(ctx, handler, cfg, retryOpts)`` shape is
    produced, and the ``missing_arg`` branch can truncate to three
    arguments to land the inspected index past the end.
    """

    return _activemq_call(
        method=_SUBSCRIBE_METHOD,
        matched_type_name=_SUBSCRIBER_CONFIG_TYPE,
        inspected_index=2,
        total_args=4,
        file_path=file_path,
    )


def _send_message_event(
    *,
    file_path: str,
) -> st.SearchStrategy[tuple[MethodCallEvent, str | None]]:
    """Strategy producing a ``<recv>.SendMessage(...)`` call event.

    Index 3 is the inspected slot per Requirement 5.3 second bullet;
    the call is generated with four positional arguments matching the
    canonical ``SendMessage(ctx, transID, correlationID, msg)`` shape.
    """

    return _activemq_call(
        method=_SEND_MESSAGE_METHOD,
        matched_type_name=_MESSAGE_TYPE,
        inspected_index=3,
        total_args=4,
        file_path=file_path,
    )


@st.composite
def _newclient_event(draw: st.DrawFn, *, file_path: str) -> MethodCallEvent:
    """Generate a ``activemq.NewClient(&activemq.JmsConfig{...})`` event.

    Per Requirement 5.3 third bullet, the I/O extractor must emit
    nothing for this shape; the external-services detector consumes
    it instead. Sprinkling the event into the per-file event list
    exercises the requirement.

    The argument shape is the canonical pointer-to-composite-literal
    of ``activemq.JmsConfig``; the recognizer never reaches the
    argument at all because the method-name gate excludes ``NewClient``.
    """

    line = draw(st.integers(min_value=1, max_value=999))
    jms_config = StructLitEvent(
        type_name="JmsConfig",
        package_alias="activemq",
        fields=(
            ("BrokerUrl", StringLitArg(value=draw(_destination_literal))),
        ),
        is_pointer=True,
        file_path=file_path,
        line=line,
    )
    return MethodCallEvent(
        receiver_chain=("activemq",),
        method_name="NewClient",
        args=(StructLitArg(event=jms_config),),
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
    overlaps with the recognized vocabulary (``Subscribe``,
    ``SendMessage``) so a regression that forgets the receiver-chain
    exclusion would surface immediately: the noise event would
    otherwise match the recognizer's method-name gate.

    To make the noise as adversarial as possible, the inspected
    positional argument is a *valid* ``domain.SubscriberConfig`` /
    ``domain.Message`` struct literal. If the recognizer were to
    inspect fx / viper receivers, the noise would emit an entry the
    expected-output computation does not predict, breaking the
    full-list equality assertion.
    """

    receiver = draw(st.sampled_from(("fx", "viper")))
    if receiver == "fx":
        method = draw(st.sampled_from(_FX_METHODS))
    else:
        method = draw(st.sampled_from(_VIPER_METHODS))
    line = draw(st.integers(min_value=1, max_value=999))

    # Build a fully valid SubscriberConfig as one of the args so a
    # regression that skipped the receiver-chain check would emit an
    # entry from this noise event.
    struct_event = StructLitEvent(
        type_name=_SUBSCRIBER_CONFIG_TYPE,
        package_alias=_DOMAIN_ALIAS,
        fields=(("Destination", StringLitArg(value=draw(_destination_literal))),),
        is_pointer=False,
        file_path=file_path,
        line=line,
    )

    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name=method,
        args=(
            _filler_arg("ctx"),
            _filler_arg("handler"),
            StructLitArg(event=struct_event),
            _filler_arg("opts"),
        ),
        file_path=file_path,
        line=line,
    )


@st.composite
def _unrelated_method_event(
    draw: st.DrawFn,
    *,
    file_path: str,
) -> MethodCallEvent:
    """Generate a method call with a non-ActiveMQ method name.

    Covers the broad "method name does not match" rejection branch.
    Concrete method-name choices overlap with other Go recognizers'
    vocabularies (``HandleFunc`` / ``AddFunc`` / ``NewTicker``) so a
    regression that accidentally widened the method-name set would
    immediately fire here.
    """

    receiver = draw(_ident)
    method = draw(
        st.sampled_from(
            ("HandleFunc", "Handle", "AddFunc", "AddJob", "NewTicker", "Get", "Post"),
        ),
    )
    line = draw(st.integers(min_value=1, max_value=999))
    return MethodCallEvent(
        receiver_chain=(receiver,),
        method_name=method,
        args=(StringLitArg(value=draw(_destination_literal)),),
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
    list[tuple[MethodCallEvent, str | None]],
    list[tuple[MethodCallEvent, str | None]],
]:
    """Generate one file's event list plus the per-recognizer expectations.

    Returns ``(events, subscribe_calls, send_calls)``:

    * ``events`` is the order-preserving event list assigned to the
      file's key in the ``events_by_file`` mapping. Always carries at
      least one noise event (``NewClient`` or fx / viper) so the
      exclusion property is exercised on every example.
    * ``subscribe_calls`` and ``send_calls`` are the per-call records
      the expected-output computation needs to predict the
      recognizer's output. Their event order matches insertion into
      ``events``, which lets the test assert per-emission shape per
      file.
    """

    events: list[GoEvent] = []

    # A noise prefix exercising the dispatch-boundary fx / viper skip
    # on every example.
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    # NewClient connection-setup noise — must emit nothing
    # (Requirement 5.3 third bullet).
    if draw(st.booleans()):
        events.append(draw(_newclient_event(file_path=path)))

    # Unrelated method calls — must emit nothing.
    unrelated = draw(
        st.lists(_unrelated_method_event(file_path=path), min_size=0, max_size=2),
    )
    events.extend(unrelated)

    subscribe_calls = draw(
        st.lists(_subscribe_event(file_path=path), min_size=0, max_size=3),
    )
    for ev, _ in subscribe_calls:
        events.append(ev)

    send_calls = draw(
        st.lists(_send_message_event(file_path=path), min_size=0, max_size=3),
    )
    for ev, _ in send_calls:
        events.append(ev)

    # A trailing noise event in case the noise's position relative to
    # the recognized events ever matters (it must not).
    events.append(draw(_fx_or_viper_noise_event(file_path=path)))

    return events, subscribe_calls, send_calls


@st.composite
def _events_by_file(
    draw: st.DrawFn,
) -> tuple[
    dict[str, list["GoEvent"]],
    dict[
        str,
        tuple[
            list[tuple[MethodCallEvent, str | None]],
            list[tuple[MethodCallEvent, str | None]],
        ],
    ],
]:
    """Generate a multi-file events mapping plus per-file expectation metadata.

    Returns ``(events_by_file, per_file_metadata)``. The
    ``per_file_metadata`` dict mirrors ``events_by_file`` keys and
    carries the two lists needed to compute the expected recognizer
    output: ``(subscribe_calls, send_calls)``.
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
            list[tuple[MethodCallEvent, str | None]],
            list[tuple[MethodCallEvent, str | None]],
        ],
    ] = {}

    for name in names:
        path = f"{name}.go"
        events, subscribe_calls, send_calls = draw(_file_events(path=path))
        events_by_file[path] = events
        metadata[path] = (subscribe_calls, send_calls)

    return events_by_file, metadata


# ---------------------------------------------------------------------------
# Expected-output computation
# ---------------------------------------------------------------------------


def _receiver_label(event: MethodCallEvent) -> str:
    """Return the dotted-name receiver label used in description suffixes."""

    if not event.receiver_chain:
        return "<unqualified>"
    return ".".join(event.receiver_chain)


def _expected_consumer_description(
    event: MethodCallEvent, destination: str,
) -> str:
    """Return the description a recognized Subscribe call must produce.

    Mirrors :func:`_format_activemq_description` for the ``Subscribe``
    branch verbatim: ``activemq message consumed from <destination>
    via <recv>.Subscribe() at <file>:<line>``.
    """

    return (
        f"{_ACTIVEMQ_LIBRARY_TOKEN} message {_CONSUMED_ACTION} {destination} "
        f"via {_receiver_label(event)}.{_SUBSCRIBE_METHOD}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_publisher_description(
    event: MethodCallEvent, destination: str,
) -> str:
    """Return the description a recognized SendMessage call must produce.

    Mirrors :func:`_format_activemq_description` for the ``SendMessage``
    branch verbatim: ``activemq message published to <destination> via
    <recv>.SendMessage() at <file>:<line>``.
    """

    return (
        f"{_ACTIVEMQ_LIBRARY_TOKEN} message {_PUBLISHED_ACTION} {destination} "
        f"via {_receiver_label(event)}.{_SEND_MESSAGE_METHOD}() "
        f"at {event.file_path}:{event.line}"
    )


def _expected_inputs_and_outputs(
    metadata: dict[
        str,
        tuple[
            list[tuple[MethodCallEvent, str | None]],
            list[tuple[MethodCallEvent, str | None]],
        ],
    ],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Compute the deduplicated expected ``AbstractInput`` / ``AbstractOutput`` lists.

    Iteration order matches the recognizer's: paths in sorted order,
    and within each file the order in which subscribe and send events
    were appended to the events list (subscribe calls first, send
    calls second, per the file-fixture's construction).
    """

    inputs: list[AbstractInput] = []
    outputs: list[AbstractOutput] = []
    seen_in: set[tuple[AbstractInputCategory, str]] = set()
    seen_out: set[tuple[AbstractOutputCategory, str]] = set()

    for path in sorted(metadata):
        subscribe_calls, send_calls = metadata[path]
        for event, expected_dest in subscribe_calls:
            if expected_dest is None:
                continue
            desc = _expected_consumer_description(event, expected_dest)
            key_in = (_MESSAGE_CONSUMED, desc)
            if key_in in seen_in:
                continue
            seen_in.add(key_in)
            inputs.append(
                AbstractInput(category=_MESSAGE_CONSUMED, description=desc),
            )
        for event, expected_dest in send_calls:
            if expected_dest is None:
                continue
            desc = _expected_publisher_description(event, expected_dest)
            key_out = (_MESSAGE_PUBLISHED, desc)
            if key_out in seen_out:
                continue
            seen_out.add(key_out)
            outputs.append(
                AbstractOutput(category=_MESSAGE_PUBLISHED, description=desc),
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


def _stripped_of_newclient(
    events_by_file: dict[str, list["GoEvent"]],
) -> dict[str, list["GoEvent"]]:
    """Return ``events_by_file`` with every ``activemq.NewClient`` call removed.

    Requirement 5.3 third bullet states the I/O extractor must produce
    no entry for the broker-connection setup; comparing the
    recognizer's output before and after the removal pins that
    contract directly.
    """

    stripped: dict[str, list[GoEvent]] = {}
    for path, events in events_by_file.items():
        kept: list[GoEvent] = []
        for event in events:
            if (
                isinstance(event, MethodCallEvent)
                and event.receiver_chain == ("activemq",)
                and event.method_name == "NewClient"
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
def test_extract_activemq_io_matches_expected_output(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                list[tuple[MethodCallEvent, str | None]],
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """Property 7 main invariant: one I/O entry per matched call site.

    Validates Requirements 5.1, 5.2, 5.3, 5.4, 5.5 jointly. The
    expected output is computed by mirroring the recognizer's
    decision tree on the per-file generator metadata, then asserting
    full-list equality. Per-emission invariants validated by
    construction:

    * Exactly one ``AbstractInput(category=message_consumed)`` per
      recognized ``Subscribe(ctx, handler, domain.SubscriberConfig{...},
      retryOpts)`` call (Requirement 5.1).
    * Exactly one ``AbstractOutput(category=message_published)`` per
      recognized ``SendMessage(ctx, transID, correlationID,
      domain.Message{...})`` call (Requirement 5.2).
    * No emission for any other shape — wrong type, wrong alias,
      unqualified alias, non-struct positional argument, truncated
      argument list, ``NewClient``, ``HandleFunc``, ``AddFunc``, or
      any fx / viper call (Requirement 5.3 + carve-outs).
    * String-literal ``Destination`` values appear verbatim in the
      description (Requirement 5.4 first sentence); non-literal
      expressions and absent fields are recorded as ``<dynamic>``
      (Requirement 5.4 second sentence).
    * The Source_Location suffix ``at <file>:<line>`` appears on
      every emission (Requirement 5.5).
    """

    events_by_file, metadata = case

    actual_inputs, actual_outputs = _extract_activemq_io(events_by_file)
    expected_inputs, expected_outputs = _expected_inputs_and_outputs(metadata)

    assert actual_inputs == expected_inputs, (
        "activemq recognizer inputs diverged from expected:\n"
        f"  actual:   {actual_inputs!r}\n"
        f"  expected: {expected_inputs!r}"
    )
    assert actual_outputs == expected_outputs, (
        "activemq recognizer outputs diverged from expected:\n"
        f"  actual:   {actual_outputs!r}\n"
        f"  expected: {expected_outputs!r}"
    )

    # Universal invariants on the actual output, independent of the
    # expected-output computation. These would catch a regression that
    # made the recognizer's output structurally wrong even when the
    # full-list equality passed (e.g. by a coincidental two-bug
    # cancellation in the expected-output mirror).
    for entry in actual_inputs:
        assert entry.category is _MESSAGE_CONSUMED, (
            f"input entry {entry!r} carries category {entry.category!r}; "
            f"the activemq recognizer must always emit message_consumed "
            f"for the consumer branch (Requirement 5.1)"
        )
        # Library marker, action token, and Source_Location must all
        # appear on every emission.
        assert entry.description.startswith(f"{_ACTIVEMQ_LIBRARY_TOKEN} message "), (
            f"input entry {entry!r} does not name the activemq library "
            f"as the first token; Requirement 5.1 mandates the marker"
        )
        assert _CONSUMED_ACTION in entry.description, (
            f"input entry {entry!r} is missing the '{_CONSUMED_ACTION}' "
            f"action token; the consumer branch must declare its direction"
        )
        assert " at " in entry.description, (
            f"input entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 5.5 mandates a Source_Location on "
            f"every emission"
        )
    for entry in actual_outputs:
        assert entry.category is _MESSAGE_PUBLISHED, (
            f"output entry {entry!r} carries category {entry.category!r}; "
            f"the activemq recognizer must always emit message_published "
            f"for the publisher branch (Requirement 5.2)"
        )
        assert entry.description.startswith(f"{_ACTIVEMQ_LIBRARY_TOKEN} message "), (
            f"output entry {entry!r} does not name the activemq library "
            f"as the first token; Requirement 5.2 mandates the marker"
        )
        assert _PUBLISHED_ACTION in entry.description, (
            f"output entry {entry!r} is missing the '{_PUBLISHED_ACTION}' "
            f"action token; the publisher branch must declare its direction"
        )
        assert " at " in entry.description, (
            f"output entry {entry!r} is missing the 'at <file>:<line>' "
            f"suffix; Requirement 5.5 mandates a Source_Location on "
            f"every emission"
        )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_activemq_io_preserves_string_literal_destination_verbatim(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                list[tuple[MethodCallEvent, str | None]],
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """Property 7 verbatim-preservation invariant.

    Validates Requirement 5.4 first sentence: a string-literal
    ``Destination`` field value must appear in the description exactly
    as the source declared, with no redaction, normalization, or
    truncation. The test asserts the literal substring is present in
    *some* output description for every recognized Subscribe /
    SendMessage call whose ``Destination`` field is a
    ``StringLitArg``.
    """

    events_by_file, metadata = case
    actual_inputs, actual_outputs = _extract_activemq_io(events_by_file)
    input_descriptions = [entry.description for entry in actual_inputs]
    output_descriptions = [entry.description for entry in actual_outputs]

    for path in sorted(metadata):
        subscribe_calls, send_calls = metadata[path]
        for event, expected_dest in subscribe_calls:
            # Only the recognized + string-literal case carries a
            # verbatim destination. Non-matching shapes (expected_dest
            # is None) produce no emission, and dynamic-destination
            # matches (expected_dest is ``<dynamic>``) are covered by
            # the main invariant test.
            if expected_dest is None or expected_dest == _DYNAMIC_TOKEN:
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            matched = any(
                expected_dest in desc and location_suffix in desc
                for desc in input_descriptions
            )
            assert matched, (
                f"string-literal Subscribe destination "
                f"{expected_dest!r} for {event.file_path}:{event.line} "
                f"not preserved verbatim in any input description; "
                f"descriptions: {input_descriptions!r}"
            )
        for event, expected_dest in send_calls:
            if expected_dest is None or expected_dest == _DYNAMIC_TOKEN:
                continue
            location_suffix = f"at {event.file_path}:{event.line}"
            matched = any(
                expected_dest in desc and location_suffix in desc
                for desc in output_descriptions
            )
            assert matched, (
                f"string-literal SendMessage destination "
                f"{expected_dest!r} for {event.file_path}:{event.line} "
                f"not preserved verbatim in any output description; "
                f"descriptions: {output_descriptions!r}"
            )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_activemq_io_ignores_fx_and_viper_receivers(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                list[tuple[MethodCallEvent, str | None]],
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
    decisions — including the (deliberately adversarial) noise events
    that carry a *valid* ``domain.SubscriberConfig`` literal in their
    argument list.
    """

    events_by_file, _ = case
    with_noise = _extract_activemq_io(events_by_file)
    without_noise = _extract_activemq_io(_stripped_of_fx_viper(events_by_file))
    assert with_noise == without_noise, (
        "activemq recognizer output changed after removing fx/viper "
        "MethodCallEvents; the dispatch-boundary exclusion must be "
        "inert (Requirements 12.1, 12.3, 13.1, 13.2)"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_activemq_io_emits_nothing_for_newclient(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                list[tuple[MethodCallEvent, str | None]],
                list[tuple[MethodCallEvent, str | None]],
            ],
        ],
    ],
) -> None:
    """NewClient connection-setup neutrality invariant.

    Validates Requirement 5.3 third bullet: the
    ``activemq.NewClient(&activemq.JmsConfig{...})`` call must
    contribute zero I/O entries. Removing every NewClient call from
    the event stream must leave the recognizer's output unchanged.
    """

    events_by_file, _ = case
    with_newclient = _extract_activemq_io(events_by_file)
    without_newclient = _extract_activemq_io(_stripped_of_newclient(events_by_file))
    assert with_newclient == without_newclient, (
        "activemq recognizer output changed after removing "
        "activemq.NewClient calls; Requirement 5.3 third bullet "
        "mandates that NewClient produces zero I/O entries"
    )


@given(_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_extract_activemq_io_is_iteration_order_independent(
    case: tuple[
        dict[str, list["GoEvent"]],
        dict[
            str,
            tuple[
                list[tuple[MethodCallEvent, str | None]],
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
    assert _extract_activemq_io(events_by_file) == _extract_activemq_io(
        reversed_mapping,
    ), (
        "activemq recognizer output depends on insertion order; "
        "Requirement 11.4 mandates deterministic, path-sorted iteration"
    )

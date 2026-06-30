"""Go I/O extractor (tasks 7.1, 7.5: HTTP and ActiveMQ recognizers).

This module is the Go-side counterpart to ``project_analyzer.io_extractor``.
It consumes the per-file ``GoEvent`` mapping produced by
``project_analyzer.go.go_parser.parse_repo`` and emits ``AbstractInput`` /
``AbstractOutput`` records following the rules in design §"go.go_io".

The recognizers implemented here so far cover:

* §1 — HTTP route registration (Requirement 3.1, 3.2, 3.3, 3.4, 3.6, 3.7).
* §2 — HTTP bootstrap suppression (Requirement 3.5).
* §5 — ActiveMQ consumer / publisher recognition (Requirement 5.1, 5.2,
  5.3, 5.4, 5.5). The ``activemq.NewClient(&activemq.JmsConfig{...})``
  connection-setup call is acknowledged here only by *not* producing an
  I/O entry; the broker URL is consumed by the external-services detector
  (task 8.2) instead.

Together with the fx exclusion (Requirement 12.1, 12.3) and viper exclusion
(Requirement 13.1, 13.2) dispatch-boundary skips that apply to every Go
recognizer, these sections are implemented by two module-level helpers
:func:`_extract_http_routes` and :func:`_extract_activemq_io`. The
remaining recognizers (scheduler, file I/O, CLI) and the public composer
:func:`extract_go_io` arrive in tasks 7.3, 7.7, 7.9, and 7.11.

Each helper is intentionally narrow: it returns just the inputs and
outputs it produced, without per-file skip messages. ``parse_repo``
already turns file-level skip cases into ``SkipFileEvent``s that yield an
empty event list for that path, so the file naturally contributes zero
detections here. The aggregator wires per-file skip strings into
``degraded_sections`` through the public composer in task 7.11.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
)
from project_knowledge_mcp.project_analyzer.go._events import (
    CallArg,
    DottedArg,
    FuncDeclEvent,
    IdentArg,
    ImportEvent,
    MethodCallEvent,
    NumberLitArg,
    SkipFileEvent,
    StringLitArg,
    StructLitArg,
    StructLitEvent,
)
from project_knowledge_mcp.project_analyzer.go.go_parser import parse_go_mod

if TYPE_CHECKING:
    from collections.abc import Mapping

    from project_knowledge_mcp.models import RepositoryContents
    from project_knowledge_mcp.project_analyzer.go._events import ArgRef, GoEvent

__all__: list[str] = ["extract_go_io"]


# --- Recognizer constants ----------------------------------------------------


#: Method names that mark a Go ``net/http`` route registration. The
#: receiver-chain check applied alongside this set restricts matches to
#: the ``http`` package itself or to a single identifier in a file that
#: imports ``net/http`` (Requirement 3.2).
_HTTP_ROUTE_METHODS: Final[frozenset[str]] = frozenset({"HandleFunc", "Handle"})


#: Method names that mark an HTTP server bootstrap call. These are
#: silently suppressed wherever they appear on the ``http`` package or a
#: ``*http.Server`` receiver — they govern server lifecycle, not
#: endpoint declaration (Requirement 3.5). Recognizing these names
#: explicitly is purely documentary in Task 7.1 because the recognizer
#: only emits inputs/outputs from :data:`_HTTP_ROUTE_METHODS`; listing
#: them here records the design's intent and gives task 7.11's composer
#: a single source of truth when it routes events between recognizers.
_HTTP_BOOTSTRAP_METHODS: Final[frozenset[str]] = frozenset(
    {"ListenAndServe", "ListenAndServeTLS", "Serve", "Shutdown", "Close"},
)


#: Import path that triggers the "any single identifier may be a mux"
#: relaxation for HTTP route recognition. Files that do not import
#: ``net/http`` only match the exact ``http.HandleFunc`` / ``http.Handle``
#: receiver-chain shape; files that do may also match
#: ``<id>.HandleFunc`` / ``<id>.Handle`` (Requirement 3.2 carve-out).
_NET_HTTP_IMPORT_PATH: Final[str] = "net/http"


#: Receiver-chain prefix that marks a uber/fx dependency-injection call.
#: Calls whose receiver chain begins with this identifier are skipped at
#: the dispatch boundary regardless of method name (Requirement 12.1,
#: 12.3, design §"go.go_io" "fx exclusion").
_FX_RECEIVER_PREFIX: Final[str] = "fx"


#: Receiver-chain prefix that marks a viper configuration call. Calls
#: whose receiver chain begins with this identifier are skipped at the
#: dispatch boundary regardless of method name (Requirement 13.1, 13.2,
#: design §"go.go_io" "viper exclusion"). Identifiers typed ``*viper.Viper``
#: through ``v := viper.New()`` are not yet tracked in Task 7.1; that
#: alias map arrives with Task 7.3 alongside the ``cron.New`` tracker
#: and is sufficient for the purpose of Task 7.1 because route
#: registrations on a viper-typed receiver do not occur in any sample
#: repository.
_VIPER_RECEIVER_PREFIX: Final[str] = "viper"


#: Pattern recognizing the Go 1.22+ method-prefixed ``http.ServeMux``
#: route literal (Requirement 3.3): ``"<METHOD> <path>"`` with a single
#: space separator and a method drawn from the closed HTTP-verb set.
#: Patterns that do not match are recorded with method ``"ANY"`` and
#: the entire literal as the path.
_HTTP_METHOD_PREFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) (.+)$",
)


# --- Public-internal entry point --------------------------------------------


def _extract_http_routes(
    events_by_file: Mapping[str, list[GoEvent]],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Recognize Go HTTP route registrations across the per-file event mapping.

    Implements design §"go.go_io" sections 1 and 2 in isolation. Task
    7.11 will compose this helper with the four other Go I/O
    recognizers (scheduler, ActiveMQ, file I/O, CLI) into the public
    ``extract_go_io`` entry point; the helper is package-internal until
    then.

    Behavior:

    1. Iterate ``events_by_file`` in path-sorted order so the produced
       lists are a deterministic function of the input mapping
       (Requirement 11.4).
    2. Per file, scan ``ImportEvent`` records for the ``net/http`` path
       so the "any single identifier may be a mux" relaxation
       (Requirement 3.2) can be applied to ``MethodCallEvent`` records
       in the same file.
    3. Walk the file's ``MethodCallEvent`` records once, applying:

       * fx exclusion (Requirement 12.1, 12.3): skip when the receiver
         chain begins with ``fx``.
       * viper exclusion (Requirement 13.1, 13.2): skip when the
         receiver chain begins with ``viper``.
       * route-method gate: only ``HandleFunc`` and ``Handle`` are
         eligible to emit. Bootstrap method names are not in this set,
         so they are dropped silently here (Requirement 3.5). This is
         the central guarantee that Task 7.1 satisfies for §2: every
         path that could reach an emission has already passed the
         method-name gate.
       * receiver-shape gate: the chain must be exactly ``("http",)``
         or a single identifier in a file importing ``net/http``.
       * argument-shape gate: only the first positional argument is
         inspected. A ``StringLitArg`` value is parsed against the
         method-prefixed pattern; any other argument shape (including
         ``DottedArg``, ``CallArg``, ``IdentArg``, ``UnknownArg``)
         falls through to the placeholder description (Requirement 3.4).

    4. Each surviving registration emits exactly one
       ``AbstractInput(category=http_request)`` and one
       ``AbstractOutput(category=http_response)``. Both records carry a
       Source_Location-equivalent suffix in their description naming
       the file path and the 1-indexed line of the
       ``MethodCallEvent`` (Requirement 3.6); description-level
       deduplication keeps repeated registrations of the same call
       site from doubling the lists (Requirement 3.7).

    Args:
        events_by_file: Mapping from repository-relative file path to
            the list of ``GoEvent`` records produced for that file by
            ``parse_repo``. Files whose only event is a
            ``SkipFileEvent`` (build constraint, cgo, tokenization
            failure) naturally contribute zero method-call events and
            therefore zero registrations, satisfying Requirement 10.4
            by construction.

    Returns:
        A 2-tuple ``(inputs, outputs)`` where:

        * ``inputs`` is a list of ``AbstractInput`` records with
          ``category=http_request``, deduplicated by
          ``(category, description)``;
        * ``outputs`` is a list of ``AbstractOutput`` records with
          ``category=http_response``, deduplicated likewise.

        Both lists may be empty.
    """

    inputs: list[AbstractInput] = []
    outputs: list[AbstractOutput] = []
    seen_inputs: set[tuple[AbstractInputCategory, str]] = set()
    seen_outputs: set[tuple[AbstractOutputCategory, str]] = set()

    for path in sorted(events_by_file):
        events = events_by_file[path]
        imports_net_http = _file_imports_net_http(events)

        for event in events:
            if not isinstance(event, MethodCallEvent):
                continue
            registration = _try_route_registration(
                event,
                file_imports_net_http=imports_net_http,
            )
            if registration is None:
                continue
            input_desc, output_desc = registration

            in_key = (AbstractInputCategory.HTTP_REQUEST, input_desc)
            if in_key not in seen_inputs:
                seen_inputs.add(in_key)
                inputs.append(
                    AbstractInput(
                        category=AbstractInputCategory.HTTP_REQUEST,
                        description=input_desc,
                    ),
                )

            out_key = (AbstractOutputCategory.HTTP_RESPONSE, output_desc)
            if out_key not in seen_outputs:
                seen_outputs.add(out_key)
                outputs.append(
                    AbstractOutput(
                        category=AbstractOutputCategory.HTTP_RESPONSE,
                        description=output_desc,
                    ),
                )

    return inputs, outputs


# --- Per-event recognizer ----------------------------------------------------


def _try_route_registration(
    event: MethodCallEvent,
    *,
    file_imports_net_http: bool,
) -> tuple[str, str] | None:
    """Return ``(input_description, output_description)`` for a registration.

    Applies the dispatch-boundary skips (fx, viper) and the recognizer
    gates (route method, receiver shape, argument shape) in order.
    Returns ``None`` whenever any gate rejects the event so the caller
    advances to the next event without emitting.

    Splitting this out of :func:`_extract_http_routes` keeps the per-file
    loop cleanly focused on accumulation while exposing the per-event
    decision as a small, independently reasoned function.
    """

    chain = event.receiver_chain
    method = event.method_name

    if _is_excluded_receiver(chain):
        return None

    # Bootstrap suppression (Requirement 3.5): the bootstrap methods
    # never reach an emission because they are not in the route-method
    # set. The explicit early-return here makes the intent visible at
    # the recognizer boundary and matches the design's "silently drop"
    # phrasing — no log, no error, no input/output.
    if method in _HTTP_BOOTSTRAP_METHODS:
        return None

    if method not in _HTTP_ROUTE_METHODS:
        return None

    receiver_label = _resolve_route_receiver(
        chain,
        file_imports_net_http=file_imports_net_http,
    )
    if receiver_label is None:
        return None

    return _format_descriptions(event, method, receiver_label)


def _is_excluded_receiver(chain: tuple[str, ...]) -> bool:
    """Return ``True`` for receivers excluded at the dispatch boundary.

    The fx and viper exclusions apply uniformly to every Go I/O
    recognizer. Centralizing them here keeps the route-registration
    path from emitting accidental detections from DI wiring or
    configuration reads (Requirements 12.1, 12.3, 13.1, 13.2).

    An empty chain (an unqualified ``foo()`` call) is never excluded —
    an unqualified call cannot satisfy the receiver-shape gate further
    down the recognizer pipeline, so the question of whether to skip
    it is moot.
    """

    if not chain:
        return False
    head = chain[0]
    return head in (_FX_RECEIVER_PREFIX, _VIPER_RECEIVER_PREFIX)


def _resolve_route_receiver(
    chain: tuple[str, ...],
    *,
    file_imports_net_http: bool,
) -> str | None:
    """Return the receiver label to use in descriptions, or ``None``.

    The two receiver shapes recognized as HTTP route registrations
    (Requirement 3.2):

    * Exact ``("http",)`` chain — the package-level ``http.HandleFunc``
      / ``http.Handle`` form on the default mux. Always recognized,
      regardless of imports, because ``net/http`` is the only package
      in the Go ecosystem that exports these names at the package
      level.
    * Single-identifier chain in a file importing ``net/http`` — the
      ``mux.HandleFunc`` / ``mux.Handle`` form on a value produced by
      ``http.NewServeMux()``. The recognition is conservative per the
      requirement: any identifier qualifies in a ``net/http``-importing
      file, without resolving its declaration site.

    Any longer or empty chain (`foo()`, `r.Group(...).HandleFunc(...)`)
    falls through to ``None`` so the call is ignored.
    """

    if chain == ("http",):
        return "http"
    if len(chain) == 1 and file_imports_net_http:
        return chain[0]
    return None


def _format_descriptions(
    event: MethodCallEvent,
    method: str,
    receiver_label: str,
) -> tuple[str, str]:
    """Return ``(input_description, output_description)`` for one registration.

    The input description carries an ``HTTP <METHOD> <PATH>`` prefix
    when the first positional argument is a string literal that we can
    decompose; otherwise it carries the design's placeholder form
    ``HTTP request <dynamic at <file>:<line> on <recv>>``. The output
    description follows the same shape, prefixed with ``HTTP response``
    so cross-cutting downstream consumers (diagram renderer, profile
    JSON) can pair them visually with the corresponding input.

    Both descriptions end with a Source_Location-equivalent suffix
    ``at <file>:<line>`` so Requirement 3.6 (1-indexed line number on
    every emitted entry) is encoded in the description string. The
    parent spec's ``AbstractInput`` and ``AbstractOutput`` models do
    not carry a ``source_locations`` field; encoding the location into
    the description is the analyzer-wide convention used by
    ``io_extractor`` for every other language.
    """

    file_path = event.file_path
    line = event.line

    first_arg = event.args[0] if event.args else None
    if isinstance(first_arg, StringLitArg):
        http_method, http_path = _parse_route_pattern(first_arg.value)
        suffix = (
            f"via {receiver_label}.{method}() at {file_path}:{line}"
        )
        input_desc = f"HTTP {http_method} {http_path} {suffix}"
        output_desc = (
            f"HTTP {http_method} response for {http_path} {suffix}"
        )
    else:
        # Requirement 3.4: non-literal pattern produces a placeholder
        # description naming the file path and the receiver identifier.
        # The placeholder shape is documented on
        # ``RouteRegistration.path`` in ``_events.py``; we follow it
        # exactly so a future Task-7.11 caller that materializes
        # ``RouteRegistration`` records can reuse the same string
        # without divergence.
        placeholder = (
            f"<dynamic at {file_path}:{line} on {receiver_label}>"
        )
        input_desc = f"HTTP request {placeholder}"
        output_desc = f"HTTP response {placeholder}"

    return input_desc, output_desc


def _parse_route_pattern(literal: str) -> tuple[str, str]:
    """Split a string-literal route pattern into ``(method, path)``.

    Requirement 3.3 grammar:

    * ``"<METHOD> <path>"`` (single space separator, METHOD drawn from
      the closed HTTP-verb set) → ``(METHOD, path)``.
    * Anything else → ``("ANY", literal)``. The bare-path forms
      observed in every sample repository (``"/healthz"``,
      ``"/debug/pprof/"``, ``"/"``, ``"/soap/repayment-service"``)
      take this branch.

    The parser does not strip whitespace, normalize the path, or
    validate the path's syntax; the literal is recorded verbatim, so
    the produced description faithfully reflects the source.
    """

    match = _HTTP_METHOD_PREFIX_RE.match(literal)
    if match is None:
        return "ANY", literal
    return match.group(1), match.group(2)


# --- File-scope helpers ------------------------------------------------------


def _file_imports_net_http(events: list[GoEvent]) -> bool:
    """Return ``True`` when ``events`` contains a ``net/http`` import.

    The recognizer iterates the per-file event list once for imports
    before the per-event walker runs, rather than re-scanning on every
    method call. The cost is one extra pass over the events; the
    benefit is a constant-time lookup during the hot per-event loop
    and clearer separation between file-level state and per-event
    decisions.
    """

    return any(
        isinstance(e, ImportEvent) and e.path == _NET_HTTP_IMPORT_PATH
        for e in events
    )


# ===========================================================================
# Scheduler registration recognizer (task 7.3)
# ===========================================================================
#
# Implements design §"go.go_io" sections 3 and 4 (Requirements 4.1, 4.2, 4.3,
# 4.4, 4.5). Three call shapes are recognized:
#
# 1. ``<v>.AddFunc(<schedule>, h)`` / ``<v>.AddJob(<schedule>, h)`` where
#    ``<v>`` is an identifier introduced by ``<v> := cron.New(...)`` in
#    the same file. Each match emits one
#    ``AbstractInput(category=scheduled_event)`` with the ``cron``
#    library marker and the ``seconds-precision`` / ``minute-precision``
#    suffix derived from the presence or absence of a
#    ``cron.WithSeconds()`` argument on the ``cron.New(...)`` call.
# 2. ``time.NewTicker(<d>)`` and ``time.AfterFunc(<d>, h)`` — standard
#    library tickers and one-shot timers used at package or function
#    scope. Each match emits one
#    ``AbstractInput(category=scheduled_event)`` with the ``time``
#    library marker.
# 3. Any ``MethodCallEvent`` whose method is ``AddFunc`` or ``AddJob``
#    but whose receiver was not recognized as a ``*cron.Cron`` value,
#    **or** whose schedule argument is of an unsupported shape (a
#    numeric literal, a struct literal, an opaque expression), still
#    emits one ``AbstractInput`` with a ``<dynamic>`` schedule and a
#    "malformed or unsupported" suffix (Requirement 4.5). The analyzer
#    never silently drops a recognizable scheduler-call shape solely
#    because its arguments are malformed.
#
# Receiver tracking. The recognizer determines ``cron`` awareness per
# file by scanning the event list for any ``MethodCallEvent`` whose
# receiver chain is exactly ``("cron",)`` and whose method name is
# ``New``. The conservative shortcut — "any single-identifier receiver
# in a file that contains a ``cron.New(...)`` call is treated as
# ``*cron.Cron``" — mirrors the HTTP recognizer's "any single
# identifier in a ``net/http``-importing file is a mux" relaxation.
# Per-call seconds-precision tracking is honored: if **any**
# ``cron.New(...)`` call in the file carries a ``cron.WithSeconds()``
# argument, the file is treated as seconds-precision for every
# subsequent ``AddFunc``/``AddJob`` match. The four sample repositories
# never declare more than one ``*cron.Cron`` per file, so the conflation
# is observationally invisible.
#
# Schedule-argument classification (Requirement 4.3 + 4.5):
#
# * ``StringLitArg`` → literal value recorded verbatim (six-field
#   strings that a five-field parser would reject are kept exactly as
#   the source declares).
# * ``DottedArg`` / ``IdentArg`` → schedule recorded as ``<dynamic>``;
#   the call shape is otherwise valid (the dotted-config-field case
#   ``cfg.JobCfg.CronSchedule`` observed in ``cat-service`` lives here).
# * Every other argument shape → schedule recorded as ``<dynamic>``
#   *and* the registration is flagged ``malformed`` so the description
#   carries the "malformed or unsupported" suffix.
# * Missing schedule argument (zero-arity call) → same malformed branch.
#
# fx and viper exclusions (Requirements 12.1, 12.3, 13.1, 13.2) are
# applied uniformly through :func:`_is_excluded_receiver`, identical to
# the HTTP and ActiveMQ recognizers.


#: Method names that mark a cron-style scheduled-task registration on a
#: ``*cron.Cron`` receiver (Requirement 4.2 second bullet). Recognition
#: is by method name combined with the per-file ``cron.New(...)``
#: awareness scan; the receiver's concrete type is not resolved.
_CRON_REGISTER_METHODS: Final[frozenset[str]] = frozenset({"AddFunc", "AddJob"})


#: Receiver chain for the ``github.com/robfig/cron/v3`` constructor.
#: Matched verbatim on ``MethodCallEvent.receiver_chain``; a non-aliased
#: ``cron.New(...)`` call is the only shape observed in the four sample
#: repositories.
_CRON_PACKAGE_CHAIN: Final[tuple[str, ...]] = ("cron",)


#: Constructor method name on the ``cron`` package.
_CRON_NEW_METHOD: Final[str] = "New"


#: Option call recognized inside a ``cron.New(...)`` argument list that
#: flips a ``*cron.Cron`` into six-field (seconds-precision) mode
#: (Requirement 4.2 first bullet). Recognition is structural: the
#: option must appear as a nested ``MethodCallEvent`` (carried as a
#: :class:`CallArg`) whose receiver chain is ``("cron",)`` and whose
#: method name is ``WithSeconds``. Bare ``WithSeconds()`` calls without
#: the ``cron.`` qualifier are not matched, mirroring the design's
#: package-qualified discriminator.
_CRON_WITH_SECONDS_METHOD: Final[str] = "WithSeconds"


#: Receiver chain for standard library tickers and one-shot timers.
#: Matched verbatim on ``MethodCallEvent.receiver_chain``.
_TIME_PACKAGE_CHAIN: Final[tuple[str, ...]] = ("time",)


#: Method names on the ``time`` package that register a scheduled-task
#: equivalent. Both shapes are reported with the ``time`` library
#: marker; neither is malformed because there is no equivalent of
#: ``cron``'s receiver-recognition step (Requirement 4.2 third bullet).
_TIME_SCHEDULER_METHODS: Final[frozenset[str]] = frozenset(
    {"NewTicker", "AfterFunc"},
)


#: Placeholder recorded in the description when the schedule argument
#: is not a string literal (Requirement 4.3 second sentence). Shared
#: with the ActiveMQ recognizer's ``<dynamic>`` token so downstream
#: consumers see a single, stable marker across recognizers.
_DYNAMIC_SCHEDULE: Final[str] = "<dynamic>"


#: Library tag included in every cron-style scheduler description so a
#: downstream consumer can tell at a glance which scheduling library
#: registered the task.
_CRON_LIBRARY_NAME: Final[str] = "cron"


#: Library tag included in every time-package scheduler description.
_TIME_LIBRARY_NAME: Final[str] = "time"


# --- Package-internal entry point -------------------------------------------


def _extract_schedulers(
    events_by_file: Mapping[str, list[GoEvent]],
) -> list[AbstractInput]:
    """Recognize Go scheduler registrations across the per-file event mapping.

    Implements design §"go.go_io" sections 3 and 4 in isolation. Task
    7.11 will compose this helper with the four other Go I/O
    recognizers (HTTP, ActiveMQ, file I/O, CLI) into the public
    ``extract_go_io`` entry point; the helper is package-internal
    until then.

    Behavior:

    1. Iterate ``events_by_file`` in path-sorted order so the produced
       list is a deterministic function of the input mapping
       (Requirement 11.4).
    2. Per file, pre-scan for ``cron.New(...)`` calls to determine
       whether the file is cron-aware and whether the file is in
       seconds-precision mode.
    3. Walk every ``MethodCallEvent`` once, applying:

       * fx exclusion (Requirement 12.1, 12.3): skip when the receiver
         chain begins with ``fx``.
       * viper exclusion (Requirement 13.1, 13.2): skip when the
         receiver chain begins with ``viper``.
       * method-name and shape gates documented at the top of the
         section.

    4. Each surviving registration emits exactly one
       ``AbstractInput(category=scheduled_event)`` with a description
       that names the library, the schedule (literal or
       ``<dynamic>``), and the Source_Location of the call. Records
       are deduplicated by ``(category, description)`` so repeated
       registrations at identical call sites coalesce.

    Args:
        events_by_file: Mapping from repository-relative file path to
            the list of ``GoEvent`` records produced for that file by
            ``parse_repo``. Files whose only event is a
            ``SkipFileEvent`` (build constraint, cgo, tokenization
            failure) naturally contribute zero ``MethodCallEvent``s
            and therefore zero emissions, satisfying Requirement 10.4
            by construction.

    Returns:
        A list of ``AbstractInput`` records with
        ``category=scheduled_event``, deduplicated by
        ``(category, description)``. May be empty.
    """

    inputs: list[AbstractInput] = []
    seen: set[tuple[AbstractInputCategory, str]] = set()

    for path in sorted(events_by_file):
        events = events_by_file[path]
        cron_seconds_precision = _detect_cron_seconds_precision(events)

        for event in events:
            if not isinstance(event, MethodCallEvent):
                continue
            if _is_excluded_receiver(event.receiver_chain):
                continue

            description = _try_scheduler_description(
                event,
                cron_seconds_precision=cron_seconds_precision,
            )
            if description is None:
                continue

            key = (AbstractInputCategory.SCHEDULED_EVENT, description)
            if key in seen:
                continue
            seen.add(key)
            inputs.append(
                AbstractInput(
                    category=AbstractInputCategory.SCHEDULED_EVENT,
                    description=description,
                ),
            )

    return inputs


# --- Per-file pre-scan ------------------------------------------------------


def _detect_cron_seconds_precision(events: list[GoEvent]) -> bool | None:
    """Return the file's cron-precision marker, or ``None`` for non-cron files.

    Three return values:

    * ``True``  — the file contains at least one ``cron.New(...)`` call
      whose argument list carries a ``cron.WithSeconds()`` option. Every
      subsequent ``AddFunc`` / ``AddJob`` match in the file is recorded
      with the ``seconds-precision`` marker (Requirement 4.2 first
      bullet).
    * ``False`` — the file contains at least one ``cron.New(...)``
      call, but none of them carries ``cron.WithSeconds()``. Matches
      are recorded with the ``minute-precision`` marker.
    * ``None``  — the file contains no recognized ``cron.New(...)``
      call. ``AddFunc`` / ``AddJob`` matches in such a file are
      treated as malformed by the per-event recognizer
      (Requirement 4.5): the receiver was not recognized as a
      ``*cron.Cron`` value.

    The conflation across multiple ``cron.New(...)`` calls within the
    same file is intentional: the four sample repositories never
    declare more than one ``*cron.Cron`` per file, and the conservative
    "any seconds-precision constructor wins" rule keeps the recognizer
    free of per-identifier alias tracking. If a future fixture
    introduces two ``*cron.Cron`` values with different precisions in
    one file, this helper's contract narrows to the file-level marker
    while the per-call-site precision marker is recorded on the
    constructor's own ``MethodCallEvent``.
    """

    has_cron_new = False
    has_with_seconds = False

    for event in events:
        if not isinstance(event, MethodCallEvent):
            continue
        if event.receiver_chain != _CRON_PACKAGE_CHAIN:
            continue
        if event.method_name != _CRON_NEW_METHOD:
            continue
        has_cron_new = True
        if _cron_new_uses_seconds(event):
            has_with_seconds = True

    if not has_cron_new:
        return None
    return has_with_seconds


def _cron_new_uses_seconds(event: MethodCallEvent) -> bool:
    """Return ``True`` when ``cron.New(...)``'s args carry ``cron.WithSeconds()``.

    The recognizer walks the call's positional arguments looking for a
    :class:`CallArg` whose wrapped :class:`MethodCallEvent` has receiver
    chain ``("cron",)`` and method name ``WithSeconds``. Other option
    shapes (``cron.WithLocation(...)``, ``cron.WithLogger(...)``, etc.)
    are ignored. Non-``CallArg`` arguments — identifiers, dotted
    paths, string literals — never match because the design's
    discriminator is the literal call ``cron.WithSeconds()``.
    """

    for arg in event.args:
        if not isinstance(arg, CallArg):
            continue
        inner = arg.call
        if inner.receiver_chain != _CRON_PACKAGE_CHAIN:
            continue
        if inner.method_name == _CRON_WITH_SECONDS_METHOD:
            return True
    return False


# --- Per-event recognizer ---------------------------------------------------


def _try_scheduler_description(
    event: MethodCallEvent,
    *,
    cron_seconds_precision: bool | None,
) -> str | None:
    """Return the scheduler description, or ``None`` when no shape matches.

    Routes the event to one of three branches:

    * Time-package scheduler (``time.NewTicker`` / ``time.AfterFunc``):
      returns the description, never malformed.
    * Cron-package registration on a single-identifier receiver
      (``<v>.AddFunc`` / ``<v>.AddJob`` with the file containing a
      ``cron.New(...)`` call): returns the description, marking it
      malformed only when the schedule argument shape is unsupported.
    * AddFunc/AddJob call shape that does not fit the cron pattern —
      either because the file lacks a ``cron.New(...)`` call or
      because the receiver chain has the wrong shape: returns a
      malformed description (Requirement 4.5).

    Every other event shape returns ``None`` so the caller advances to
    the next event without emitting.
    """

    chain = event.receiver_chain
    method = event.method_name

    # Time-package scheduler shapes (Requirement 4.2 third bullet).
    if chain == _TIME_PACKAGE_CHAIN and method in _TIME_SCHEDULER_METHODS:
        schedule = _format_schedule_argument(event.args)
        return _format_time_scheduler_description(event, schedule)

    # Cron-package registration shapes (Requirement 4.2 second bullet
    # plus Requirement 4.5 malformed fallback).
    if method in _CRON_REGISTER_METHODS:
        cron_recognized = (
            cron_seconds_precision is not None and len(chain) == 1
        )
        if not cron_recognized:
            return _format_cron_malformed_description(event)
        schedule, malformed = _classify_cron_schedule_argument(event.args)
        if malformed:
            return _format_cron_malformed_description(event)
        return _format_cron_recognized_description(
            event=event,
            schedule=schedule,
            seconds_precision=bool(cron_seconds_precision),
        )

    return None


def _classify_cron_schedule_argument(
    args: tuple[ArgRef, ...],
) -> tuple[str, bool]:
    """Return ``(schedule, malformed)`` for the cron schedule argument.

    Requirement 4.3 mandates that string-literal schedules are recorded
    verbatim (including six-field strings that a five-field parser
    would reject). Requirement 4.3's second sentence and Requirement
    4.5 jointly mandate that ``<dynamic>`` is recorded for non-literal
    expressions, with the malformed flag set only when the argument
    shape is structurally unrecognized.

    The classification:

    * ``StringLitArg``: the literal's contents are returned verbatim;
      ``malformed=False``.
    * ``DottedArg`` (e.g. ``cfg.JobCfg.CronSchedule``) or
      ``IdentArg`` (e.g. ``schedule``): ``<dynamic>`` is returned;
      ``malformed=False``. These are the "known dynamic-config field"
      shapes the design carves out.
    * Anything else — ``NumberLitArg``, ``StructLitArg``, ``CallArg``,
      ``UnknownArg``, an absent argument: ``<dynamic>`` is returned;
      ``malformed=True`` so the description carries the "malformed or
      unsupported" suffix.
    """

    if not args:
        return _DYNAMIC_SCHEDULE, True
    first = args[0]
    if isinstance(first, StringLitArg):
        return first.value, False
    if isinstance(first, (DottedArg, IdentArg)):
        return _DYNAMIC_SCHEDULE, False
    return _DYNAMIC_SCHEDULE, True


def _format_schedule_argument(args: tuple[ArgRef, ...]) -> str:
    """Return the time-package scheduler's first-argument description fragment.

    Mirrors the cron classification's "literal verbatim, otherwise
    ``<dynamic>``" contract but without the malformed flag because
    Requirement 4.5's malformed fallback applies only to the
    ``AddFunc`` / ``AddJob`` cron shapes. The time-package shapes
    accept arbitrary duration expressions; their schedule is recorded
    as ``<dynamic>`` whenever it is not a string literal so the
    description is uniform across recognizers.
    """

    if not args:
        return _DYNAMIC_SCHEDULE
    first = args[0]
    if isinstance(first, StringLitArg):
        return first.value
    if isinstance(first, NumberLitArg):
        return first.text
    return _DYNAMIC_SCHEDULE


# --- Description formatters -------------------------------------------------


def _format_cron_recognized_description(
    *,
    event: MethodCallEvent,
    schedule: str,
    seconds_precision: bool,
) -> str:
    """Return the description for a recognized cron-style registration.

    Description shape (parallels the HTTP and ActiveMQ recognizers'
    ``via <recv>.<method>() at <file>:<line>`` suffix so a downstream
    consumer can pair the location across all I/O entries):

    ``"scheduled (cron, <precision>) <schedule> via <recv>.<method>() at <file>:<line>"``

    Where ``<precision>`` is ``"seconds-precision"`` when the file's
    ``cron.New(...)`` carries ``cron.WithSeconds()`` and
    ``"minute-precision"`` otherwise (Requirement 4.2 first bullet).
    """

    precision = (
        "seconds-precision" if seconds_precision else "minute-precision"
    )
    receiver_label = _receiver_label(event)
    return (
        f"scheduled ({_CRON_LIBRARY_NAME}, {precision}) {schedule} "
        f"via {receiver_label}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _format_cron_malformed_description(event: MethodCallEvent) -> str:
    """Return the malformed-cron description (Requirement 4.5).

    Description shape pinned by the design:

    ``"scheduled <dynamic> (malformed or unsupported scheduler shape: <method>(<receiver>) at <file>:<line>)"``

    The shape is identical regardless of whether the receiver was
    unrecognized or the schedule argument was unsupported — both cases
    are observationally "we recognized the method-name silhouette but
    could not fully classify the call". Pinning a single description
    shape keeps downstream string-matching predictable.
    """

    receiver_label = _receiver_label(event)
    return (
        f"scheduled {_DYNAMIC_SCHEDULE} "
        f"(malformed or unsupported scheduler shape: "
        f"{event.method_name}({receiver_label}) "
        f"at {event.file_path}:{event.line})"
    )


def _format_time_scheduler_description(
    event: MethodCallEvent,
    schedule: str,
) -> str:
    """Return the description for a ``time.NewTicker`` / ``time.AfterFunc``.

    Description shape parallels the cron-recognized shape but uses the
    ``time`` library marker and the bare method name in place of the
    precision suffix; the time package has no concept of cron-style
    precision so a uniform "scheduled (time, <method>)" prefix is the
    closest equivalent.
    """

    return (
        f"scheduled ({_TIME_LIBRARY_NAME}, {event.method_name}) {schedule} "
        f"via time.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _receiver_label(event: MethodCallEvent) -> str:
    """Return the dotted-name receiver label used in description suffixes.

    Mirrors the convention from the HTTP and ActiveMQ recognizers: the
    receiver chain is joined with ``.``, and the literal token
    ``<unqualified>`` is substituted for the empty-chain case so the
    description still names the call shape. Unqualified ``AddFunc(...)``
    / ``AddJob(...)`` calls do not occur in any sample repository; the
    fallback exists for structural completeness.
    """

    if not event.receiver_chain:
        return "<unqualified>"
    return ".".join(event.receiver_chain)


# ===========================================================================
# ActiveMQ consumer / publisher recognizer (task 7.5)
# ===========================================================================
#
# Implements design §"go.go_io" section 5 (Requirement 5.1, 5.2, 5.3, 5.4,
# 5.5). Two patterns are recognized:
#
# 1. ``<recv>.Subscribe(ctx, handler, <cfg>, retryOpts)`` whose third
#    positional argument is a ``StructLitEvent`` of type
#    ``domain.SubscriberConfig`` → one
#    ``AbstractInput(category=message_consumed)``.
# 2. ``<recv>.SendMessage(ctx, transID, correlationID, <msg>)`` whose
#    fourth positional argument is a ``StructLitEvent`` of type
#    ``domain.Message`` → one ``AbstractOutput(category=message_published)``.
#
# The third documented call shape — ``activemq.NewClient(&activemq.JmsConfig{
# BrokerUrl: <url>, ...})`` — is intentionally *not* matched here. It feeds
# the external-services detector (task 8.2) instead; the I/O extractor's
# contribution for that call is exactly zero entries, satisfying Requirement
# 5.3 third bullet by construction. The recognizer below only fires on the
# two method names ``Subscribe`` and ``SendMessage``, so ``NewClient``
# cannot accidentally reach an emission.
#
# Both shapes apply the same dispatch-boundary skips that gate the HTTP
# recognizer: fx and viper receivers are excluded uniformly via
# :func:`_is_excluded_receiver` (Requirements 12.1, 12.3, 13.1, 13.2).
#
# String-literal ``Destination`` field values are recorded verbatim;
# every other expression (field-access, function call, variable
# reference, or an absent field) is recorded as ``<dynamic>``
# (Requirement 5.4).


#: Method name for the in-house ActiveMQ wrapper's consumer entry point
#: (Requirement 5.3 first bullet). Recognition is by method name alone;
#: the receiver type is not resolved, because the in-house wrapper exposes
#: the method on a value (``*activemq.Receiver``) whose concrete shape
#: varies by call site and the design's discriminator is the
#: ``domain.SubscriberConfig`` positional argument rather than the
#: receiver.
_ACTIVEMQ_SUBSCRIBE_METHOD: Final[str] = "Subscribe"


#: Method name for the in-house ActiveMQ wrapper's publisher entry point
#: (Requirement 5.3 second bullet). Same recognition-by-method-name
#: rationale as :data:`_ACTIVEMQ_SUBSCRIBE_METHOD`.
_ACTIVEMQ_SEND_MESSAGE_METHOD: Final[str] = "SendMessage"


#: Composite-literal type name carrying the consumer's destination
#: configuration (Requirement 5.3 first bullet, glossary entry
#: "Go_ActiveMQ_Client").
_ACTIVEMQ_SUBSCRIBER_CONFIG_TYPE: Final[str] = "SubscriberConfig"


#: Composite-literal type name carrying the publisher's message payload
#: (Requirement 5.3 second bullet).
_ACTIVEMQ_MESSAGE_TYPE: Final[str] = "Message"


#: Package alias under which both ActiveMQ composite literals appear in
#: the four sample repositories. The wrapper's companion package
#: ``esb-go-libs/activemq/domain`` is imported under the local alias
#: ``domain`` by every observed call site; the recognizer requires this
#: alias so unrelated ``Message`` or ``SubscriberConfig`` types in other
#: packages do not match.
_ACTIVEMQ_DOMAIN_PACKAGE_ALIAS: Final[str] = "domain"


#: Field name carrying the destination string on both
#: ``domain.SubscriberConfig`` and ``domain.Message``. The two types are
#: distinct but their destination field is identically named, so a single
#: constant suffices.
_ACTIVEMQ_DESTINATION_FIELD: Final[str] = "Destination"


#: Positional index of the ``SubscriberConfig`` argument in the call
#: ``Subscribe(ctx, handler, cfg, retryOpts)`` — i.e. the "third
#: positional argument" the design and Requirement 5.3 first bullet
#: refer to (zero-based index 2 after ``ctx`` at 0 and ``handler`` at 1).
_SUBSCRIBE_CONFIG_ARG_INDEX: Final[int] = 2


#: Positional index of the ``Message`` argument in the call
#: ``SendMessage(ctx, transID, correlationID, msg)`` — i.e. the "fourth
#: positional argument" of Requirement 5.3 second bullet (zero-based
#: index 3 after ``ctx``, ``transID``, ``correlationID``).
_SEND_MESSAGE_MSG_ARG_INDEX: Final[int] = 3


#: Placeholder recorded in the description when the ``Destination`` field
#: value is not a string literal (Requirement 5.4). The same token is
#: used across every Go recognizer that surfaces a not-statically-
#: determinable value, so downstream consumers see a single, stable
#: marker.
_DYNAMIC_DESTINATION: Final[str] = "<dynamic>"


#: Library tag included verbatim in every emitted description so the
#: literal token ``activemq`` appears, satisfying Requirement 5.1 / 5.2's
#: "names the library as ``activemq``" clause.
_ACTIVEMQ_LIBRARY_NAME: Final[str] = "activemq"


# --- Package-internal entry point -------------------------------------------


def _extract_activemq_io(
    events_by_file: Mapping[str, list[GoEvent]],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Recognize Go ActiveMQ consumer / publisher calls across all files.

    Implements design §"go.go_io" section 5 in isolation. Task 7.11 will
    compose this helper with the other Go I/O recognizers (HTTP,
    scheduler, file I/O, CLI) into the public ``extract_go_io`` entry
    point; the helper is package-internal until then.

    Behavior:

    1. Iterate ``events_by_file`` in path-sorted order so the produced
       lists are a deterministic function of the input mapping
       (Requirement 11.4).
    2. Per file, walk every ``MethodCallEvent`` once, applying:

       * fx exclusion (Requirement 12.1, 12.3): skip when the receiver
         chain begins with ``fx``.
       * viper exclusion (Requirement 13.1, 13.2): skip when the
         receiver chain begins with ``viper``.
       * method-name gate: only ``Subscribe`` and ``SendMessage`` are
         eligible to emit. ``NewClient`` (the broker-connection setup
         from Requirement 5.3 third bullet) is therefore silently
         dropped here — by construction, never by an explicit
         suppression. The external-services detector consumes that
         call separately.
       * argument-shape gate: the recognizer inspects exactly one
         positional argument — index 2 for ``Subscribe``, index 3 for
         ``SendMessage`` — and requires it to be a ``StructLitArg``
         whose wrapped event is the expected
         ``domain.<TypeName>`` composite literal. Any other
         shape (a non-existent argument, a non-struct-literal, a
         struct literal of a different type, or a struct literal under
         a different package alias) drops the call without an
         emission.

    3. Each surviving call emits exactly one entry — an ``AbstractInput``
       for ``Subscribe``, an ``AbstractOutput`` for ``SendMessage`` —
       deduplicated by ``(category, description)`` so repeated
       registrations of the same call site coalesce (the same
       dedup contract the HTTP recognizer uses).

    Args:
        events_by_file: Mapping from repository-relative file path to
            the list of ``GoEvent`` records produced for that file by
            ``parse_repo``. Files whose only event is a
            ``SkipFileEvent`` (build constraint, cgo, tokenization
            failure) naturally contribute zero ``MethodCallEvent``s and
            therefore zero emissions, satisfying Requirement 10.4 by
            construction.

    Returns:
        A 2-tuple ``(inputs, outputs)`` where:

        * ``inputs`` is a list of ``AbstractInput`` records with
          ``category=message_consumed``, deduplicated by
          ``(category, description)``;
        * ``outputs`` is a list of ``AbstractOutput`` records with
          ``category=message_published``, deduplicated likewise.

        Both lists may be empty.
    """

    inputs: list[AbstractInput] = []
    outputs: list[AbstractOutput] = []
    seen_inputs: set[tuple[AbstractInputCategory, str]] = set()
    seen_outputs: set[tuple[AbstractOutputCategory, str]] = set()

    for path in sorted(events_by_file):
        events = events_by_file[path]
        for event in events:
            if not isinstance(event, MethodCallEvent):
                continue
            if _is_excluded_receiver(event.receiver_chain):
                continue

            consumer_desc = _try_activemq_consumer(event)
            if consumer_desc is not None:
                key = (AbstractInputCategory.MESSAGE_CONSUMED, consumer_desc)
                if key not in seen_inputs:
                    seen_inputs.add(key)
                    inputs.append(
                        AbstractInput(
                            category=AbstractInputCategory.MESSAGE_CONSUMED,
                            description=consumer_desc,
                        ),
                    )
                continue

            publisher_desc = _try_activemq_publisher(event)
            if publisher_desc is not None:
                key = (AbstractOutputCategory.MESSAGE_PUBLISHED, publisher_desc)
                if key not in seen_outputs:
                    seen_outputs.add(key)
                    outputs.append(
                        AbstractOutput(
                            category=AbstractOutputCategory.MESSAGE_PUBLISHED,
                            description=publisher_desc,
                        ),
                    )

    return inputs, outputs


# --- Per-event recognizers --------------------------------------------------


def _try_activemq_consumer(event: MethodCallEvent) -> str | None:
    """Return the consumer description, or ``None`` when the event does not match.

    Applies the method-name gate (``Subscribe``) and the argument-shape
    gate (third positional argument is a ``domain.SubscriberConfig``
    composite literal). Returns ``None`` whenever any gate rejects the
    event so the caller advances to the next event without emitting.
    """

    if event.method_name != _ACTIVEMQ_SUBSCRIBE_METHOD:
        return None

    config_event = _struct_lit_arg_of_type(
        event.args,
        index=_SUBSCRIBE_CONFIG_ARG_INDEX,
        expected_type_name=_ACTIVEMQ_SUBSCRIBER_CONFIG_TYPE,
        expected_package_alias=_ACTIVEMQ_DOMAIN_PACKAGE_ALIAS,
    )
    if config_event is None:
        return None

    destination = _extract_destination_field(config_event)
    return _format_activemq_description(
        event=event,
        method=_ACTIVEMQ_SUBSCRIBE_METHOD,
        action="consumed from",
        destination=destination,
    )


def _try_activemq_publisher(event: MethodCallEvent) -> str | None:
    """Return the publisher description, or ``None`` when the event does not match.

    Applies the method-name gate (``SendMessage``) and the argument-shape
    gate (fourth positional argument is a ``domain.Message`` composite
    literal). Returns ``None`` whenever any gate rejects the event.
    """

    if event.method_name != _ACTIVEMQ_SEND_MESSAGE_METHOD:
        return None

    message_event = _struct_lit_arg_of_type(
        event.args,
        index=_SEND_MESSAGE_MSG_ARG_INDEX,
        expected_type_name=_ACTIVEMQ_MESSAGE_TYPE,
        expected_package_alias=_ACTIVEMQ_DOMAIN_PACKAGE_ALIAS,
    )
    if message_event is None:
        return None

    destination = _extract_destination_field(message_event)
    return _format_activemq_description(
        event=event,
        method=_ACTIVEMQ_SEND_MESSAGE_METHOD,
        action="published to",
        destination=destination,
    )


# --- Argument-shape and field-extraction helpers ----------------------------


def _struct_lit_arg_of_type(
    args: tuple[ArgRef, ...],
    *,
    index: int,
    expected_type_name: str,
    expected_package_alias: str,
) -> StructLitEvent | None:
    """Return the wrapped ``StructLitEvent`` when ``args[index]`` matches.

    The match requires *all* of:

    * the positional argument at ``index`` exists,
    * it is a ``StructLitArg`` (i.e. the call literally embeds a
      composite literal at that position, not a variable holding one),
    * the wrapped event's ``type_name`` equals ``expected_type_name``,
    * and the wrapped event's ``package_alias`` equals
      ``expected_package_alias``.

    The first three conditions are uncontroversial. The package-alias
    check is the load-bearing one: it constrains the match to the
    in-house ``esb-go-libs/activemq/domain`` package (imported as
    ``domain`` in every observed call site) and excludes unrelated
    ``Message`` or ``SubscriberConfig`` types in other packages that
    might exist in a larger Go codebase. Bare-type composite literals
    (``Message{...}`` without a package qualifier) and pointer-typed
    composite literals (``&activemq.Message{...}``) both fail the
    package-alias check — the design's discriminator is explicitly the
    ``domain``-aliased form.
    """

    if index >= len(args):
        return None
    arg = args[index]
    if not isinstance(arg, StructLitArg):
        return None
    struct_event = arg.event
    if struct_event.type_name != expected_type_name:
        return None
    if struct_event.package_alias != expected_package_alias:
        return None
    return struct_event


def _extract_destination_field(struct_event: StructLitEvent) -> str:
    """Return the ``Destination`` field's value as a description fragment.

    Three cases per Requirement 5.4:

    * String-literal value (``"REP.SERVICE.PAYMENT.ERR"``) → the
      literal's unquoted contents are returned verbatim. The recognizer
      does not strip, normalize, or validate the literal; whatever the
      source declares is what the description carries.
    * Any non-string-literal value (field access like
      ``u.cfg.JMSConfig.PaymentErrQueue.Destination``, a function call,
      a variable reference, or any other expression shape the
      recognizer surfaces as ``IdentArg`` / ``DottedArg`` / ``CallArg``
      / ``UnknownArg`` / ``NumberLitArg``) → the placeholder
      :data:`_DYNAMIC_DESTINATION`.
    * The ``Destination`` field is absent from the composite literal →
      same placeholder. This is the conservative choice for an
      unexpected literal shape: emitting an entry with an unknown
      destination is preferable to silently dropping the call site,
      because the Subscribe / SendMessage method-name match is itself
      strong evidence of an ActiveMQ I/O boundary.
    """

    for field_name, value in struct_event.fields:
        if field_name != _ACTIVEMQ_DESTINATION_FIELD:
            continue
        if isinstance(value, StringLitArg):
            return value.value
        return _DYNAMIC_DESTINATION
    return _DYNAMIC_DESTINATION


def _format_activemq_description(
    *,
    event: MethodCallEvent,
    method: str,
    action: str,
    destination: str,
) -> str:
    """Return the description string emitted for one ActiveMQ I/O entry.

    Description shape (mirrors the HTTP recognizer's "via <recv>.<method>()
    at <file>:<line>" suffix so cross-section consumers see a uniform
    Source_Location encoding):

    ``"activemq message <action> <destination> via <recv>.<method>() at <file>:<line>"``

    Where:

    * ``<action>`` is ``"consumed from"`` for ``Subscribe`` and
      ``"published to"`` for ``SendMessage``.
    * ``<destination>`` is the literal destination string or
      :data:`_DYNAMIC_DESTINATION`.
    * ``<recv>`` is the dotted-name receiver chain of the
      ``MethodCallEvent``, joined with ``.``. For the (unlikely)
      unqualified call form ``Subscribe(...)`` / ``SendMessage(...)``
      the literal token ``<unqualified>`` is used so the description
      still names the call shape. The four sample repositories never
      exhibit this form for ActiveMQ calls; the fallback exists for
      structural completeness.
    * ``<file>`` / ``<line>`` are the ``MethodCallEvent``'s
      ``file_path`` and 1-indexed ``line``, satisfying Requirement 5.5.

    The literal token ``activemq`` always appears at the start of the
    description so Requirement 5.1 / 5.2's "names the library as
    ``activemq``" clause holds for every emitted entry, regardless of
    receiver shape or destination determinability.
    """

    receiver_label = (
        ".".join(event.receiver_chain) if event.receiver_chain else "<unqualified>"
    )
    return (
        f"{_ACTIVEMQ_LIBRARY_NAME} message {action} {destination} "
        f"via {receiver_label}.{method}() at {event.file_path}:{event.line}"
    )


# ===========================================================================
# File I/O recognizer (task 7.7)
# ===========================================================================
#
# Implements design §"go.go_io" section 6 (Requirement 6.1, 6.2, 6.3, 6.4).
# Six call shapes are recognized:
#
# 1. ``os.Open(<path>)`` and ``os.ReadFile(<path>)`` →
#    ``AbstractInput(category=file_read)``.
# 2. ``ioutil.ReadFile(<path>)`` → ``AbstractInput(category=file_read)``.
# 3. ``os.Create(<path>)`` and ``os.WriteFile(<path>, ...)`` →
#    ``AbstractOutput(category=file_written)``.
# 4. ``ioutil.WriteFile(<path>, ...)`` →
#    ``AbstractOutput(category=file_written)``.
# 5. ``os.OpenFile(<path>, <flag-expr>, <mode>)`` → classified by inspecting
#    ``<flag-expr>`` for ``os.O_*`` atoms (Requirement 6.3):
#
#    * single ``os.O_RDONLY`` atom → ``AbstractInput(file_read)`` only;
#    * single ``os.O_WRONLY``, ``os.O_RDWR``, ``os.O_APPEND``,
#      ``os.O_CREATE``, or ``os.O_TRUNC`` atom →
#      ``AbstractOutput(file_written)`` only;
#    * any other shape (multi-atom OR expression, identifier reference,
#      function call, unknown atom) → emit *both* an
#      ``AbstractInput(file_read)`` and an ``AbstractOutput(file_written)``.
#
# Path extraction is uniform across all six shapes: the first positional
# argument's string-literal value is recorded verbatim, otherwise the
# placeholder ``<dynamic>`` is recorded (Requirement 6.4 plus the
# task-level "record the literal path argument when string-literal"
# clause).
#
# The Go parser's ``_parse_arg`` does not emit a structured representation
# of binary-OR expressions: ``os.O_RDWR | os.O_CREATE`` is split across
# two argument slots — a ``DottedArg(parts=("os", "O_RDWR"))`` followed by
# an ``UnknownArg(text="| os . O_CREATE")``. The recognizer treats any
# ``os.OpenFile`` call whose positional-argument count differs from the
# canonical ``(path, flag, mode)`` triple as having a not-statically-
# determinable flag expression, which routes the call to the
# "emit both" branch documented above. This is the load-bearing rule
# that satisfies Requirement 6.3's undecidable case without needing
# the parser to model bitwise-OR atoms.
#
# fx and viper exclusions (Requirements 12.1, 12.3, 13.1, 13.2) are
# applied uniformly through :func:`_is_excluded_receiver`, identical to
# the HTTP, scheduler, and ActiveMQ recognizers.


#: Receiver chain for calls into the Go standard library ``os`` package.
#: Matched verbatim on ``MethodCallEvent.receiver_chain``; the recognizer
#: never resolves aliased imports (``import myos "os"``) because the four
#: sample repositories use only the canonical ``os`` name. Aliased
#: imports would still surface as the alias here and would fall through
#: to the "no match" branch — a conservative outcome that errs on the
#: side of under-reporting rather than mis-attributing the call.
_OS_PACKAGE_CHAIN: Final[tuple[str, ...]] = ("os",)


#: Receiver chain for calls into the legacy Go standard library
#: ``io/ioutil`` package. The Go standard library has deprecated this
#: package in favor of ``os`` and ``io``, but every observed legacy
#: codebase still imports it and the four sample repositories pin both
#: shapes through the in-house wrapper. Matched verbatim on
#: ``MethodCallEvent.receiver_chain``.
_IOUTIL_PACKAGE_CHAIN: Final[tuple[str, ...]] = ("ioutil",)


#: Method names on the ``os`` package that unconditionally read a file
#: (Requirement 6.1). ``Open`` returns a read-only ``*os.File``;
#: ``ReadFile`` returns the file's contents as a byte slice. Both
#: contribute exactly one ``AbstractInput(file_read)``.
_OS_READ_METHODS: Final[frozenset[str]] = frozenset({"Open", "ReadFile"})


#: Method names on the ``os`` package that unconditionally write a file
#: (Requirement 6.2). ``Create`` returns a writable ``*os.File`` and
#: truncates the file if it already exists; ``WriteFile`` writes a byte
#: slice to a named file. Both contribute exactly one
#: ``AbstractOutput(file_written)``.
_OS_WRITE_METHODS: Final[frozenset[str]] = frozenset({"Create", "WriteFile"})


#: Method names on the ``ioutil`` package that unconditionally read a
#: file. ``ioutil.ReadFile`` is the historical equivalent of
#: ``os.ReadFile`` and behaves identically for I/O-classification
#: purposes (Requirement 6.1, glossary entry "Go_File_IO_Function").
_IOUTIL_READ_METHODS: Final[frozenset[str]] = frozenset({"ReadFile"})


#: Method names on the ``ioutil`` package that unconditionally write a
#: file. ``ioutil.WriteFile`` is the historical equivalent of
#: ``os.WriteFile`` (Requirement 6.2).
_IOUTIL_WRITE_METHODS: Final[frozenset[str]] = frozenset({"WriteFile"})


#: Method name on the ``os`` package whose read/write classification
#: depends on the flag bitmask passed as the second positional
#: argument. Handled by the dedicated :func:`_classify_openfile_flag`
#: helper (Requirement 6.3).
_OS_OPENFILE_METHOD: Final[str] = "OpenFile"


#: Canonical positional-argument count for ``os.OpenFile`` —
#: ``(path, flag, mode)``. The recognizer routes any call whose
#: positional-argument count differs from this value to the
#: "not statically determinable" branch, because the only way the
#: count can grow is if the parser split a binary-OR flag expression
#: across multiple argument slots (e.g. ``os.O_RDWR | os.O_CREATE``).
_OS_OPENFILE_ARITY: Final[int] = 3


#: Positional index of the path argument on every recognized file-I/O
#: shape — always the first positional argument.
_FILE_PATH_ARG_INDEX: Final[int] = 0


#: Positional index of the flag argument on ``os.OpenFile`` —
#: ``(path=0, flag=1, mode=2)``.
_OS_OPENFILE_FLAG_ARG_INDEX: Final[int] = 1


#: Atom name on ``os`` that marks read-only access. A single
#: ``DottedArg(parts=("os", "O_RDONLY"))`` as the flag expression
#: classifies the call as a read; any other shape (including a
#: multi-atom expression that *includes* ``O_RDONLY``) does not.
_OS_OPENFILE_READ_ATOM: Final[str] = "O_RDONLY"


#: Atom names on ``os`` that mark write access (Requirement 6.2). A
#: single ``DottedArg(parts=("os", <atom>))`` whose atom is in this
#: set classifies the call as a write. The closed-set semantics match
#: the Go standard library: any bitmask containing one of these flags
#: opens the file for writing, regardless of whether ``O_RDONLY`` is
#: also OR'd in (though Go's ``open(2)`` semantics make
#: ``O_RDONLY | O_APPEND`` ill-defined in practice; treating any
#: write atom as authoritative is consistent with the requirement).
_OS_OPENFILE_WRITE_ATOMS: Final[frozenset[str]] = frozenset(
    {"O_WRONLY", "O_RDWR", "O_APPEND", "O_CREATE", "O_TRUNC"},
)


#: Placeholder recorded when the path argument is not a string literal
#: (Requirement 6.4 and the task-level "otherwise ``<dynamic>``"
#: clause). Shared with the ActiveMQ and scheduler recognizers'
#: ``<dynamic>`` token so downstream consumers see a single, stable
#: marker across recognizers.
_DYNAMIC_PATH: Final[str] = "<dynamic>"


#: Classification verdicts returned by :func:`_classify_openfile_flag`.
#: ``"read"``  — the flag expression resolves to a read-only access.
#: ``"write"`` — the flag expression resolves to a write access.
#: ``"both"``  — the flag expression is not statically determinable;
#:               the recognizer emits both an input and an output
#:               (Requirement 6.3).
_FileIOVerdict = str  # one of "read", "write", "both"


#: Expected number of segments in a fully-qualified ``os.O_<NAME>``
#: flag atom: a package alias segment (``"os"``) followed by the atom
#: name segment (``"O_RDONLY"``, ``"O_RDWR"``, ...). ``DottedArg``
#: instances whose ``parts`` length differs from this value cannot
#: encode a recognized flag and route to the "undecidable" branch.
_OS_FLAG_DOTTED_ARG_LEN: Final[int] = 2


# --- Package-internal entry point -------------------------------------------


def _extract_file_io(
    events_by_file: Mapping[str, list[GoEvent]],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Recognize Go file I/O calls across the per-file event mapping.

    Implements design §"go.go_io" section 6 in isolation. Task 7.11
    will compose this helper with the four other Go I/O recognizers
    (HTTP, scheduler, ActiveMQ, CLI) into the public ``extract_go_io``
    entry point; the helper is package-internal until then.

    Behavior:

    1. Iterate ``events_by_file`` in path-sorted order so the produced
       lists are a deterministic function of the input mapping
       (Requirement 11.4).
    2. Per file, walk every ``MethodCallEvent`` once, applying:

       * fx exclusion (Requirement 12.1, 12.3): skip when the receiver
         chain begins with ``fx``.
       * viper exclusion (Requirement 13.1, 13.2): skip when the
         receiver chain begins with ``viper``.
       * receiver-chain and method-name gates: only ``os`` and
         ``ioutil`` receivers, with a method name in the read- or
         write-method set or equal to ``OpenFile``, are eligible to
         emit.

    3. Each recognized read-only call emits exactly one
       ``AbstractInput(category=file_read)``. Each recognized
       write-only call emits exactly one
       ``AbstractOutput(category=file_written)``. Each recognized
       ``os.OpenFile`` call whose flag expression is not statically
       determinable emits *both* — one ``AbstractInput`` and one
       ``AbstractOutput`` — preserving Requirement 6.3 verbatim.
    4. Records are deduplicated by ``(category, description)`` so
       repeated calls at the same site coalesce.

    Args:
        events_by_file: Mapping from repository-relative file path to
            the list of ``GoEvent`` records produced for that file by
            ``parse_repo``. Files whose only event is a
            ``SkipFileEvent`` (build constraint, cgo, tokenization
            failure) naturally contribute zero ``MethodCallEvent``s
            and therefore zero emissions, satisfying Requirement 10.4
            by construction.

    Returns:
        A 2-tuple ``(inputs, outputs)`` where:

        * ``inputs`` is a list of ``AbstractInput`` records with
          ``category=file_read``, deduplicated by
          ``(category, description)``;
        * ``outputs`` is a list of ``AbstractOutput`` records with
          ``category=file_written``, deduplicated likewise.

        Both lists may be empty.
    """

    inputs: list[AbstractInput] = []
    outputs: list[AbstractOutput] = []
    seen_inputs: set[tuple[AbstractInputCategory, str]] = set()
    seen_outputs: set[tuple[AbstractOutputCategory, str]] = set()

    for path in sorted(events_by_file):
        events = events_by_file[path]
        for event in events:
            if not isinstance(event, MethodCallEvent):
                continue
            if _is_excluded_receiver(event.receiver_chain):
                continue

            verdict = _classify_file_io_event(event)
            if verdict is None:
                continue

            file_path_value = _extract_file_path_argument(event.args)
            input_desc = _format_file_read_description(event, file_path_value)
            output_desc = _format_file_written_description(event, file_path_value)

            if verdict in ("read", "both"):
                key = (AbstractInputCategory.FILE_READ, input_desc)
                if key not in seen_inputs:
                    seen_inputs.add(key)
                    inputs.append(
                        AbstractInput(
                            category=AbstractInputCategory.FILE_READ,
                            description=input_desc,
                        ),
                    )
            if verdict in ("write", "both"):
                key2 = (AbstractOutputCategory.FILE_WRITTEN, output_desc)
                if key2 not in seen_outputs:
                    seen_outputs.add(key2)
                    outputs.append(
                        AbstractOutput(
                            category=AbstractOutputCategory.FILE_WRITTEN,
                            description=output_desc,
                        ),
                    )

    return inputs, outputs


# --- Per-event classifier ---------------------------------------------------


def _classify_file_io_event(event: MethodCallEvent) -> _FileIOVerdict | None:  # noqa: PLR0911 - dispatch table over recognized shapes
    """Return the verdict for ``event`` or ``None`` when no shape matches.

    The verdict drives the per-event emission decision in
    :func:`_extract_file_io`:

    * ``"read"``  — emit exactly one ``AbstractInput(file_read)``.
    * ``"write"`` — emit exactly one ``AbstractOutput(file_written)``.
    * ``"both"``  — emit one of each (the ``os.OpenFile`` undecidable
                  branch; Requirement 6.3).
    * ``None``    — no recognized file-I/O shape; the caller advances
                  to the next event without emitting.

    Routing order:

    1. ``os`` receiver chain → either a read-method, a write-method,
       or the ``OpenFile`` flag-bitmask classifier.
    2. ``ioutil`` receiver chain → either a read-method or a
       write-method (no flag-bitmask shape exists in this package).
    3. Anything else → ``None``.
    """

    chain = event.receiver_chain
    method = event.method_name

    if chain == _OS_PACKAGE_CHAIN:
        if method in _OS_READ_METHODS:
            return "read"
        if method in _OS_WRITE_METHODS:
            return "write"
        if method == _OS_OPENFILE_METHOD:
            return _classify_openfile_flag(event.args)
        return None

    if chain == _IOUTIL_PACKAGE_CHAIN:
        if method in _IOUTIL_READ_METHODS:
            return "read"
        if method in _IOUTIL_WRITE_METHODS:
            return "write"
        return None

    return None


def _classify_openfile_flag(args: tuple[ArgRef, ...]) -> _FileIOVerdict:  # noqa: PLR0911 - dispatch table over O_* flag atoms
    """Return the read/write verdict for an ``os.OpenFile`` flag expression.

    Requirement 6.3 carves the classification into three buckets:

    * the flag expression resolves to a single ``os.O_RDONLY`` atom →
      ``"read"``;
    * the flag expression resolves to a single atom in
      :data:`_OS_OPENFILE_WRITE_ATOMS` → ``"write"``;
    * the flag expression cannot be statically determined → ``"both"``.

    The Go parser's ``_parse_arg`` returns each ``os.O_<NAME>`` token
    as a :class:`DottedArg` of parts ``("os", "O_<NAME>")``. Binary-OR
    expressions split across argument slots — ``os.O_RDWR | os.O_CREATE``
    surfaces as ``DottedArg(("os", "O_RDWR"))`` followed by
    ``UnknownArg(text="| os . O_CREATE")`` — so any ``OpenFile`` call
    whose positional-argument count differs from the canonical
    ``(path, flag, mode)`` triple is routed to the "both" branch.
    This is a conservative over-approximation: the multi-atom case
    contributes one extra entry per category, matching Requirement
    6.3's "emit both" prescription rather than risking a missed
    detection.

    Other failure modes that route to ``"both"``:

    * the call has fewer than two positional arguments (malformed Go
      that wouldn't compile, but a defensible recognizer guards
      against it);
    * the flag argument is an :class:`IdentArg` (e.g. ``flag``), a
      :class:`NumberLitArg`, a :class:`CallArg`, a :class:`StructLitArg`,
      a :class:`StringLitArg`, or an :class:`UnknownArg`;
    * the flag argument is a :class:`DottedArg` but its parts do not
      match the ``("os", "O_<NAME>")`` shape (e.g. a 3-segment chain,
      or a non-``os`` package);
    * the recognized atom is neither ``O_RDONLY`` nor one of the five
      write atoms (e.g. ``O_EXCL``, ``O_SYNC``, ``O_NONBLOCK``); the
      requirement enumerates only six atoms and is silent on the rest,
      so the recognizer defers to "undecidable".
    """

    if len(args) != _OS_OPENFILE_ARITY:
        return "both"

    flag = args[_OS_OPENFILE_FLAG_ARG_INDEX]
    if not isinstance(flag, DottedArg):
        return "both"
    if len(flag.parts) != _OS_FLAG_DOTTED_ARG_LEN:
        return "both"
    package_alias, atom = flag.parts
    if package_alias != _OS_PACKAGE_CHAIN[0]:
        return "both"

    if atom == _OS_OPENFILE_READ_ATOM:
        return "read"
    if atom in _OS_OPENFILE_WRITE_ATOMS:
        return "write"
    return "both"


# --- Argument extraction and description formatting -------------------------


def _extract_file_path_argument(args: tuple[ArgRef, ...]) -> str:
    """Return the path argument's description fragment.

    Two cases per Requirement 6.4 and the task-level "record the
    literal path argument when string-literal, otherwise ``<dynamic>``"
    clause:

    * the first positional argument is a :class:`StringLitArg` → its
      unquoted contents are returned verbatim. The recognizer does not
      strip, normalize, or validate the literal; whatever the source
      declares is what the description carries.
    * any other shape (the argument is absent, or it is an
      :class:`IdentArg` / :class:`DottedArg` / :class:`NumberLitArg` /
      :class:`CallArg` / :class:`StructLitArg` / :class:`UnknownArg`)
      → the placeholder :data:`_DYNAMIC_PATH`.

    Shared by the read, write, and OpenFile branches so the path
    fragment is identical regardless of how the call's verdict was
    reached.
    """

    if not args:
        return _DYNAMIC_PATH
    first = args[_FILE_PATH_ARG_INDEX]
    if isinstance(first, StringLitArg):
        return first.value
    return _DYNAMIC_PATH


def _format_file_read_description(
    event: MethodCallEvent,
    file_path_value: str,
) -> str:
    """Return the description string emitted for one file-read entry.

    Description shape (mirrors the HTTP, scheduler, and ActiveMQ
    recognizers' ``via <recv>.<method>() at <file>:<line>`` suffix so
    cross-section consumers see a uniform Source_Location encoding):

    ``"file read <path> via <recv>.<method>() at <file>:<line>"``

    Where:

    * ``<path>`` is the literal path string or :data:`_DYNAMIC_PATH`.
    * ``<recv>`` is the dotted-name receiver chain of the
      ``MethodCallEvent``, joined with ``.``. The receiver is always
      ``os`` or ``ioutil`` for emitted entries; the fallback
      ``<unqualified>`` exists for structural completeness.
    * ``<file>`` / ``<line>`` are the ``MethodCallEvent``'s
      ``file_path`` and 1-indexed ``line``, satisfying Requirement 6.4.
    """

    receiver_label = (
        ".".join(event.receiver_chain) if event.receiver_chain else "<unqualified>"
    )
    return (
        f"file read {file_path_value} "
        f"via {receiver_label}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _format_file_written_description(
    event: MethodCallEvent,
    file_path_value: str,
) -> str:
    """Return the description string emitted for one file-write entry.

    Description shape parallels :func:`_format_file_read_description`
    with the leading verb swapped from ``file read`` to
    ``file written`` so the two categories are visually distinct in
    downstream tooling while sharing the ``via <recv>.<method>() at
    <file>:<line>`` suffix and Source_Location encoding.
    """

    receiver_label = (
        ".".join(event.receiver_chain) if event.receiver_chain else "<unqualified>"
    )
    return (
        f"file written {file_path_value} "
        f"via {receiver_label}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


# ===========================================================================
# CLI entry-point recognizer (task 7.9)
# ===========================================================================
#
# Implements design §"go.go_io" section 7 (Requirements 7.1, 7.2, 7.3, 7.4,
# 7.5, 7.6, plus the fx exclusion clauses of Requirements 12.1 and 12.3).
# Two broad call shapes are recognized, both emitting
# ``AbstractInput(category=cli_argument)``:
#
# 1. ``func main()`` declarations in files that name a CLI binary
#    entry point. Three eligible file-path shapes drive the binary name:
#
#    * ``cmd/<name>/main.go`` — one directory deep under ``cmd/``. The
#      binary name is ``<name>`` derived from the intermediate directory
#      segment (Requirement 7.1).
#    * ``cmd/main.go`` — the binary name is the last segment of the
#      ``go.mod`` module path after stripping the ``<host>/<org>/``
#      prefix (Requirement 7.2). Requires a well-formed ``go.mod`` at
#      the repository root.
#    * ``main.go`` at the repository root — the binary name is the same
#      module-path-derived value used for ``cmd/main.go``
#      (Requirement 7.3). Requires a well-formed ``go.mod`` at the
#      repository root.
#
#    The recognizer reads ``go.mod`` once per ``_extract_cli_entry_points``
#    invocation through :func:`parse_go_mod`, mirroring the stripping
#    rule used by :func:`go_purpose._gomod_module_path` but reducing to
#    just the *last* segment of the stripped path. The four sample
#    repositories' bare module names (``repayment_service``,
#    ``cat-service``, ``aps_los_vtiger``, ``fec_pool_service``) survive
#    the stripping unchanged; a hypothetical
#    ``github.com/acme/payment-service`` reduces to ``payment-service``.
#
# 2. Calls into the ``flag`` package on the Go standard library
#    (Requirement 7.4). Eleven method names are recognized, falling
#    into three argument-shape buckets:
#
#    * Non-``Var`` registration variants
#      (``String``, ``Int``, ``Bool``, ``Float64``, ``Duration``,
#      ``NewFlagSet``): the flag name is the **first** positional
#      argument when string-literal.
#    * ``Var`` registration variants
#      (``StringVar``, ``IntVar``, ``BoolVar``, ``Float64Var``,
#      ``DurationVar``): the flag name is the **second** positional
#      argument when string-literal — the first argument is the pointer
#      receiver (``&v``) into which the parsed flag value is stored.
#    * ``Parse``: no flag-name argument; the call is recorded with a
#      generic description noting that flag parsing was invoked. This
#      is the catch-all case Requirement 7.4 carves out for the
#      "names the flag name when the call is one of the named
#      registration functions and the flag-name argument is a string
#      literal" rule — ``flag.Parse`` is recognized but is not a named
#      registration function and therefore has no flag name to record.
#
#    Non-literal flag-name arguments (identifiers, dotted paths, calls,
#    etc.) are recorded with the placeholder :data:`_DYNAMIC_FLAG_NAME`
#    so the call site is still surfaced to the downstream consumer.
#
# fx exclusion (Requirements 12.1, 12.3). Two exclusions apply at the
# dispatch boundary, both no-ops for the canonical ``flag.<method>``
# shapes (whose receiver chain is exactly ``("flag",)``) but pinned
# here to satisfy the design's defensive contract:
#
# * Any ``MethodCallEvent`` whose receiver chain begins with ``fx`` is
#   skipped via :func:`_is_excluded_receiver` (shared with the four
#   other Go I/O recognizers).
# * Any ``MethodCallEvent`` with method name ``Append`` whose receiver
#   is a single-identifier chain in a file that imports
#   ``go.uber.org/fx`` (any submodule) is skipped. This is a
#   conservative approximation of the design's "receiver type can be
#   resolved to ``fx.Lifecycle``" rule — without per-function
#   parameter-type tracking, treating any single-identifier
#   ``.Append`` receiver in an fx-importing file as a ``fx.Lifecycle``
#   is sound for the four sample repositories and over-rejects only
#   for hypothetical non-``fx.Lifecycle`` ``.Append`` calls that
#   coincidentally appear in the same file. Such calls would not
#   contribute a CLI input under any of the recognizers anyway, so the
#   over-rejection is observationally invisible.
#
# Source_Location rejection (Requirement 7.6). Every emitted
# ``AbstractInput`` carries a Source_Location-equivalent suffix
# ``at <file>:<line>`` derived from the originating event's
# ``file_path`` and 1-indexed ``line``. The recognizer rejects (does
# not emit) any candidate whose ``line`` cannot be determined — the
# typed event dataclasses use ``int`` line numbers and never produce
# ``None`` in practice, but the guard is encoded explicitly so the
# requirement's contract is visible at the recognizer boundary.


#: Receiver chain for calls into the Go standard library ``flag``
#: package. Matched verbatim on ``MethodCallEvent.receiver_chain``.
#: Aliased imports (``import f "flag"``) would surface the alias as
#: the chain head and fall through to the "no match" branch; the four
#: sample repositories use only the canonical ``flag`` name, so the
#: under-reporting is observationally invisible.
_FLAG_PACKAGE_CHAIN: Final[tuple[str, ...]] = ("flag",)


#: Positional argument index that carries the flag name for each
#: recognized ``flag.<method>`` registration shape. The mapping
#: encodes the Go ``flag`` package's calling conventions:
#:
#: * non-``Var`` registrations and ``NewFlagSet`` accept the name as
#:   the first positional argument;
#: * ``Var`` registrations accept the name as the second positional
#:   argument (the first is the pointer receiver into which the parsed
#:   value is stored).
#:
#: ``Parse`` is intentionally absent from this map; it is handled as a
#: zero-arity special case in :func:`_try_flag_description` because it
#: takes no flag-name argument at all.
_FLAG_NAME_ARG_INDEX: Final[Mapping[str, int]] = {
    "String": 0,
    "StringVar": 1,
    "Int": 0,
    "IntVar": 1,
    "Bool": 0,
    "BoolVar": 1,
    "Float64": 0,
    "Float64Var": 1,
    "Duration": 0,
    "DurationVar": 1,
    "NewFlagSet": 0,
}


#: Method name on the ``flag`` package whose call shape is recognized
#: but does not itself register a named flag (Requirement 7.4).
#: Emitted with a generic "flag parsing invoked" description.
_FLAG_PARSE_METHOD: Final[str] = "Parse"


#: Full set of recognized ``flag.<method>`` names. Computed once from
#: :data:`_FLAG_NAME_ARG_INDEX` plus the ``Parse`` special case so the
#: recognizer's method-name gate stays in lock-step with the
#: argument-index map without manual duplication.
_FLAG_METHODS: Final[frozenset[str]] = frozenset(
    {*_FLAG_NAME_ARG_INDEX, _FLAG_PARSE_METHOD},
)


#: Placeholder recorded when the flag-name argument is not a string
#: literal (Requirement 7.4 second sentence: "includes the flag name
#: when ... the flag-name argument is a string literal"). Shared with
#: the other Go recognizers' ``<dynamic>`` token so downstream
#: consumers see a single, stable marker across recognizers.
_DYNAMIC_FLAG_NAME: Final[str] = "<dynamic>"


#: Library tag included in every emitted flag-call description so the
#: literal token ``flag`` appears, satisfying Requirement 7.4's
#: "names the framework as ``flag``" clause.
_FLAG_LIBRARY_NAME: Final[str] = "flag"


#: Repository-relative path of the Go module manifest at the
#: repository root. Read via :meth:`RepositoryContents.read_text` to
#: derive the binary name for ``cmd/main.go`` and root-level
#: ``main.go`` entry points.
_GO_MOD_PATH: Final[str] = "go.mod"


#: Repository-relative path of the canonical ``cmd/main.go`` entry
#: point. Matched verbatim against the file path; the binary name is
#: sourced from ``go.mod`` rather than the directory tree
#: (Requirement 7.2).
_CMD_MAIN_PATH: Final[str] = "cmd/main.go"


#: Repository-relative path of a root-level ``main.go`` entry point.
#: Matched verbatim against the file path; the binary name is sourced
#: from ``go.mod`` rather than the file path (Requirement 7.3).
_ROOT_MAIN_PATH: Final[str] = "main.go"


#: Number of path segments in a ``cmd/<name>/main.go`` shape — exactly
#: three: ``cmd``, ``<name>``, ``main.go``. Paths with fewer or more
#: segments are not eligible for the directory-derived binary name
#: rule (Requirement 7.1 "one directory deep under ``cmd/``").
_CMD_NAMED_PATH_SEGMENTS: Final[int] = 3


#: Number of slash-separated segments in a module path that triggers
#: ``<host>/<org>/`` prefix stripping. Mirrors the constant of the
#: same purpose in :mod:`go_purpose` so the binary-name derivation
#: stays in sync with the purpose-summary derivation (Requirement 7.2
#: refers to "the last segment of the module path" which the design
#: pins to the same stripping rule).
_MODULE_PATH_PREFIXED_SEGMENTS: Final[int] = 3


#: Function name that identifies a Go program's entry point. The Go
#: runtime invokes exactly this free function (``func main()`` with no
#: receiver) when launching a binary.
_MAIN_FUNCTION_NAME: Final[str] = "main"


#: Method name on a ``fx.Lifecycle``-typed identifier that registers
#: start/stop hooks (Requirements 12.1, 12.3). Skipped at the dispatch
#: boundary alongside the receiver-chain ``fx.*`` exclusion to satisfy
#: the design's fx neutrality contract.
_FX_LIFECYCLE_APPEND_METHOD: Final[str] = "Append"


#: Import-path prefix for any module of the Uber ``fx`` dependency
#: injection framework. A file is treated as fx-importing when any of
#: its ``ImportEvent`` paths begin with this prefix; the check is used
#: only to gate the ``.Append`` exclusion in
#: :func:`_is_excluded_fx_lifecycle_append`.
_FX_IMPORT_PREFIX: Final[str] = "go.uber.org/fx"


# --- Package-internal entry point -------------------------------------------


def _extract_cli_entry_points(
    repository_contents: RepositoryContents,
    events_by_file: Mapping[str, list[GoEvent]],
) -> list[AbstractInput]:
    """Recognize Go CLI entry-point inputs across the per-file event mapping.

    Implements design §"go.go_io" section 7 in isolation. Task 7.11
    will compose this helper with the four other Go I/O recognizers
    (HTTP, scheduler, ActiveMQ, file I/O) into the public
    ``extract_go_io`` entry point; the helper is package-internal
    until then.

    Behavior:

    1. Read ``go.mod`` once at the top of the call through
       :func:`parse_go_mod` to derive the module-path-based binary
       name for ``cmd/main.go`` and root-level ``main.go`` entry
       points (Requirements 7.2, 7.3). When ``go.mod`` is absent or
       contains no well-formed ``module`` line, the
       module-path-derived binary name is ``None`` and the
       ``cmd/main.go`` / ``main.go`` branches contribute no entry.
    2. Iterate ``events_by_file`` in path-sorted order so the produced
       list is a deterministic function of the input mapping
       (Requirement 11.4).
    3. Per file, pre-scan for any ``go.uber.org/fx`` import so the
       ``.Append`` exclusion (Requirements 12.1, 12.3) can be gated
       on file-level fx awareness.
    4. Emit one ``AbstractInput(category=cli_argument)`` for each
       eligible ``func main()`` declaration whose path matches one of
       the three eligible shapes (Requirements 7.1, 7.2, 7.3).
    5. Walk every ``MethodCallEvent`` once, applying:

       * the fx-package exclusion (Requirements 12.1, 12.3) via
         :func:`_is_excluded_receiver`;
       * the fx.Lifecycle ``.Append`` exclusion via
         :func:`_is_excluded_fx_lifecycle_append`;
       * the receiver-chain gate (``("flag",)``) and method-name gate
         (:data:`_FLAG_METHODS`).

       Each surviving call emits one
       ``AbstractInput(category=cli_argument)`` whose description
       includes the literal flag name when the relevant positional
       argument is a string literal (Requirement 7.4) or
       :data:`_DYNAMIC_FLAG_NAME` otherwise.
    6. Reject (do not emit) any candidate whose ``line`` is ``None``
       (Requirement 7.6). The typed event dataclasses use ``int``
       line numbers and never produce ``None`` in practice, but the
       guard is encoded explicitly so the requirement's contract is
       visible at the recognizer boundary.
    7. Records are deduplicated by ``(category, description)`` so
       repeated registrations at identical call sites coalesce — the
       same dedup contract every other Go I/O recognizer uses.

    Args:
        repository_contents: The Repository_Contents snapshot for the
            project under analysis. Used to read ``go.mod`` for the
            module-path-derived binary name.
        events_by_file: Mapping from repository-relative file path to
            the list of ``GoEvent`` records produced for that file by
            ``parse_repo``. Files whose only event is a
            ``SkipFileEvent`` (build constraint, cgo, tokenization
            failure) naturally contribute zero ``FuncDeclEvent`` and
            ``MethodCallEvent`` records and therefore zero emissions,
            satisfying Requirement 10.4 by construction.

    Returns:
        A list of ``AbstractInput`` records with
        ``category=cli_argument``, deduplicated by
        ``(category, description)``. May be empty.
    """

    inputs: list[AbstractInput] = []
    seen: set[tuple[AbstractInputCategory, str]] = set()

    module_binary_name = _module_binary_name(repository_contents)

    for path in sorted(events_by_file):
        events = events_by_file[path]
        file_imports_fx = _file_imports_fx(events)

        binary_description = _try_main_binary_description(
            path=path,
            events=events,
            module_binary_name=module_binary_name,
        )
        if binary_description is not None:
            _record_cli_input(binary_description, inputs, seen)

        for event in events:
            if not isinstance(event, MethodCallEvent):
                continue
            if _is_excluded_receiver(event.receiver_chain):
                continue
            if _is_excluded_fx_lifecycle_append(
                event,
                file_imports_fx=file_imports_fx,
            ):
                continue

            flag_description = _try_flag_description(event)
            if flag_description is None:
                continue
            _record_cli_input(flag_description, inputs, seen)

    return inputs


def _record_cli_input(
    description: str,
    inputs: list[AbstractInput],
    seen: set[tuple[AbstractInputCategory, str]],
) -> None:
    """Append a ``cli_argument`` input, deduplicating by description.

    Shared by the ``func main()`` branch and the ``flag.<method>``
    branch so the dedup contract is identical across both: a
    description that has already been emitted in the current call is
    silently dropped on subsequent emissions. The dedup is per
    ``(category, description)``, mirroring the convention used by
    the HTTP, scheduler, ActiveMQ, and file-I/O recognizers.
    """

    key = (AbstractInputCategory.CLI_ARGUMENT, description)
    if key in seen:
        return
    seen.add(key)
    inputs.append(
        AbstractInput(
            category=AbstractInputCategory.CLI_ARGUMENT,
            description=description,
        ),
    )


# --- go.mod binary-name derivation ------------------------------------------


def _module_binary_name(rc: RepositoryContents) -> str | None:
    """Return the last segment of the stripped ``go.mod`` module path.

    Reads ``go.mod`` once via :meth:`RepositoryContents.read_text` and
    parses it with :func:`parse_go_mod`. Returns ``None`` when:

    * ``go.mod`` is absent from the repository,
    * ``go.mod`` is present but contains no well-formed ``module``
      line,
    * the parsed module path is empty after stripping.

    The stripping rule mirrors :func:`go_purpose._gomod_module_path`
    but reduces the result to the *last* slash-separated segment per
    Requirement 7.2's "last segment of the module path declared in the
    Go_Module_Manifest" clause. Worked examples:

    * ``"github.com/acme/payment-service"`` → strip
      ``github.com/acme/`` → ``"payment-service"``;
    * ``"repayment_service"`` (bare) → no stripping → ``"repayment_service"``;
    * ``"github.com/foo/bar/baz"`` → strip ``github.com/foo/`` →
      ``"bar/baz"`` → last segment → ``"baz"``.
    """

    text = rc.read_text(_GO_MOD_PATH)
    if text is None:
        return None
    event = parse_go_mod(text)
    if event is None:
        return None

    parts = event.module_path.split("/")
    if len(parts) >= _MODULE_PATH_PREFIXED_SEGMENTS:
        parts = parts[2:]
    if not parts:
        return None
    last = parts[-1]
    return last if last else None


# --- ``func main()`` binary entry-point recognizer --------------------------


def _try_main_binary_description(
    *,
    path: str,
    events: list[GoEvent],
    module_binary_name: str | None,
) -> str | None:
    """Return the binary description for an eligible ``func main()`` site.

    Returns ``None`` when any of the following holds:

    * ``path`` is not one of the three eligible shapes
      (``cmd/<name>/main.go``, ``cmd/main.go``, or ``main.go``);
    * ``events`` contains no ``FuncDeclEvent`` named ``main`` with a
      ``None`` receiver type (i.e. the file does not declare a
      ``func main()`` free function);
    * the binary name cannot be derived — either because the
      ``cmd/<name>/main.go`` segment is empty (defensive guard;
      should not occur for a well-formed Go layout) or because the
      ``cmd/main.go`` / ``main.go`` branch needs a ``go.mod`` and
      ``module_binary_name`` is ``None``;
    * the recognized ``FuncDeclEvent`` carries no line number
      (Requirement 7.6 Source_Location rejection rule — the typed
      dataclass never produces this, but the guard is explicit so the
      contract is visible at the boundary).

    On a successful match, the returned description follows the
    convention used by every other Go I/O recognizer's emission:

    ``"binary <name> via func main() at <file>:<line>"``

    The literal token ``binary`` opens the description so downstream
    tooling can pair every CLI entry-point input with its file
    location at a glance. The ``via func main() at ...`` suffix
    encodes the Source_Location into the description string, matching
    the analyzer-wide convention (the ``AbstractInput`` model does
    not carry a ``source_locations`` field).
    """

    binary_name = _binary_name_for_path(
        path=path,
        module_binary_name=module_binary_name,
    )
    if binary_name is None:
        return None

    main_decl = _find_main_func_decl(events)
    if main_decl is None:
        return None
    if main_decl.line is None:  # type: ignore[unreachable]
        return None

    return (
        f"binary {binary_name} "
        f"via func main() at {main_decl.file_path}:{main_decl.line}"
    )


def _binary_name_for_path(
    *,
    path: str,
    module_binary_name: str | None,
) -> str | None:
    """Return the binary name implied by ``path``, or ``None`` if ineligible.

    Three eligible shapes (Requirements 7.1, 7.2, 7.3):

    1. ``cmd/<name>/main.go`` — exactly three segments where the
       first is ``cmd`` and the third is ``main.go``. The binary
       name is the middle segment (Requirement 7.1). Worked example:
       ``cmd/api/main.go`` → ``"api"``.
    2. ``cmd/main.go`` — exactly the literal path. The binary name is
       :paramref:`module_binary_name`; returns ``None`` when no
       ``go.mod`` is present (Requirement 7.2).
    3. ``main.go`` at the repository root. The binary name is
       :paramref:`module_binary_name`; returns ``None`` when no
       ``go.mod`` is present (Requirement 7.3).

    Any other path shape (``cmd/foo/bar/main.go``,
    ``internal/main.go``, ``pkg/something.go``, etc.) is ineligible
    and returns ``None``.
    """

    parts = path.split("/")
    if (
        len(parts) == _CMD_NAMED_PATH_SEGMENTS
        and parts[0] == "cmd"
        and parts[2] == "main.go"
    ):
        name = parts[1]
        return name if name else None
    if path == _CMD_MAIN_PATH:
        return module_binary_name
    if path == _ROOT_MAIN_PATH:
        return module_binary_name
    return None


def _find_main_func_decl(events: list[GoEvent]) -> FuncDeclEvent | None:
    """Return the ``func main()`` free-function declaration, if any.

    A "free function" has a ``None`` receiver type; methods with
    receiver ``func (r *T) main()`` do not match. The recognizer
    returns the first matching event in source order so any
    accidental redeclaration (which would be a compile error in Go)
    produces a single, deterministic emission rather than two
    duplicates. Returns ``None`` when no matching event exists.
    """

    for event in events:
        if not isinstance(event, FuncDeclEvent):
            continue
        if event.name != _MAIN_FUNCTION_NAME:
            continue
        if event.receiver_type is not None:
            continue
        return event
    return None


# --- ``flag.<method>`` registration recognizer ------------------------------


def _try_flag_description(event: MethodCallEvent) -> str | None:
    """Return the flag-call description, or ``None`` when no shape matches.

    Applies the receiver-chain gate (``("flag",)``) and the
    method-name gate (:data:`_FLAG_METHODS`) before formatting the
    description. Three formatting branches:

    * ``flag.Parse()`` (no flag-name argument) →
      ``"flag parsing invoked via flag.Parse() at <file>:<line>"``.
    * Named registration whose flag-name argument is a string literal
      → ``"flag <name> via flag.<method>() at <file>:<line>"``.
    * Named registration whose flag-name argument is missing or any
      non-literal shape → ``"flag <dynamic> via flag.<method>() at
      <file>:<line>"``.

    The branch selection is driven by :data:`_FLAG_NAME_ARG_INDEX`
    (which maps each named registration variant to its flag-name
    argument index) and the special-case check for ``Parse``.

    Source_Location rejection (Requirement 7.6) is applied at the
    top: an event without a line number contributes no input. The
    typed event dataclass never produces ``None`` in practice, but
    the guard is explicit so the contract is visible at the
    recognizer boundary.
    """

    if event.receiver_chain != _FLAG_PACKAGE_CHAIN:
        return None
    if event.method_name not in _FLAG_METHODS:
        return None
    if event.line is None:  # type: ignore[unreachable]
        return None

    if event.method_name == _FLAG_PARSE_METHOD:
        return (
            f"{_FLAG_LIBRARY_NAME} parsing invoked "
            f"via {_FLAG_LIBRARY_NAME}.{_FLAG_PARSE_METHOD}() "
            f"at {event.file_path}:{event.line}"
        )

    name_idx = _FLAG_NAME_ARG_INDEX[event.method_name]
    flag_name = _extract_flag_name_argument(event.args, name_idx)
    return (
        f"{_FLAG_LIBRARY_NAME} {flag_name} "
        f"via {_FLAG_LIBRARY_NAME}.{event.method_name}() "
        f"at {event.file_path}:{event.line}"
    )


def _extract_flag_name_argument(
    args: tuple[ArgRef, ...],
    name_idx: int,
) -> str:
    """Return the flag name fragment for a ``flag.<method>`` registration.

    Two cases per Requirement 7.4 second sentence ("includes the flag
    name when the call is one of the named registration functions and
    the flag-name argument is a string literal"):

    * the positional argument at ``name_idx`` exists and is a
      :class:`StringLitArg` → the literal's unquoted contents are
      returned verbatim. The recognizer does not strip, normalize, or
      validate the literal; whatever the source declares is what the
      description carries.
    * the argument is absent (the call has fewer positional arguments
      than expected) or has any other shape (:class:`IdentArg`,
      :class:`DottedArg`, :class:`NumberLitArg`, :class:`CallArg`,
      :class:`StructLitArg`, :class:`UnknownArg`) → the placeholder
      :data:`_DYNAMIC_FLAG_NAME`.
    """

    if name_idx >= len(args):
        return _DYNAMIC_FLAG_NAME
    arg = args[name_idx]
    if isinstance(arg, StringLitArg):
        return arg.value
    return _DYNAMIC_FLAG_NAME


# --- fx.Lifecycle ``.Append`` exclusion -------------------------------------


def _file_imports_fx(events: list[GoEvent]) -> bool:
    """Return ``True`` when ``events`` contains any ``go.uber.org/fx`` import.

    Used solely to gate the ``.Append`` exclusion in
    :func:`_is_excluded_fx_lifecycle_append`. The check is across all
    fx submodules (``go.uber.org/fx``, ``go.uber.org/fx/fxevent``,
    ``go.uber.org/fx/fxtest``, etc.) because the design treats every
    fx-rooted import as evidence that the file is in the fx wiring
    domain.

    The recognizer iterates the per-file event list once for imports
    rather than re-scanning on every method call, mirroring the
    :func:`_file_imports_net_http` helper used by the HTTP
    recognizer.
    """

    return any(
        isinstance(e, ImportEvent) and e.path.startswith(_FX_IMPORT_PREFIX)
        for e in events
    )


def _is_excluded_fx_lifecycle_append(
    event: MethodCallEvent,
    *,
    file_imports_fx: bool,
) -> bool:
    """Return ``True`` for ``.Append`` calls on a likely ``fx.Lifecycle``.

    The design's exact rule is "every ``Append`` call on an
    ``fx.Lifecycle``-typed identifier". Without per-function
    parameter-type tracking, the recognizer applies the conservative
    approximation documented at the top of this section:

    * the method name must be ``Append``;
    * the receiver chain must be a single identifier
      (multi-segment chains like ``fx.Lifecycle.Append`` are already
      caught by :func:`_is_excluded_receiver` via the ``fx`` head; a
      single-identifier chain is the form ``lc.Append(...)`` produced
      by Go's syntax for method calls on a parameter binding);
    * the file must import any submodule of ``go.uber.org/fx``.

    Under these three conditions the call is treated as an
    ``fx.Lifecycle.Append`` registration and skipped. The
    over-approximation cost is bounded: a non-``fx.Lifecycle``
    ``.Append`` call (e.g. on a custom collection type) in the same
    file is also skipped, but no canonical Go I/O recognizer would
    have emitted an input for that call anyway — ``Append`` is not in
    the HTTP, scheduler, ActiveMQ, file-I/O, or flag method sets.

    Returning ``False`` for the call leaves the rest of the recognizer
    pipeline to decide whether the event matches any other shape; in
    practice it never does for ``Append``, so the falling-through
    return value is observationally indistinguishable from a skip for
    the CLI recognizer. The exclusion is encoded here regardless so
    the design's contract is visible at the recognizer boundary and
    so a future cross-recognizer composition step does not
    accidentally surface fx wiring as a CLI input.
    """

    if event.method_name != _FX_LIFECYCLE_APPEND_METHOD:
        return False
    if not file_imports_fx:
        return False
    return len(event.receiver_chain) == 1


# ===========================================================================
# Public entry point — extract_go_io composer (task 7.11)
# ===========================================================================
#
# Implements design §"go.go_io" composer contract. Composes the five
# package-internal recognizers — :func:`_extract_http_routes`,
# :func:`_extract_schedulers`, :func:`_extract_activemq_io`,
# :func:`_extract_file_io`, and :func:`_extract_cli_entry_points` — into a
# single function returning ``(inputs, outputs, file_skip_messages)``.
#
# The composer's three responsibilities:
#
# 1. Run each recognizer once and concatenate its emissions. Each
#    recognizer already iterates ``events_by_file`` in path-sorted order
#    (Requirement 11.4) and dedups its own emissions by
#    ``(category, description)``; the composer adds a final
#    cross-recognizer dedup pass so a description that two different
#    recognizers might both emit (e.g. an HTTP and a file-I/O
#    recognizer agreeing on the same Source_Location-encoded string)
#    is recorded only once in the merged output. In practice no two
#    recognizers share a category, so this pass is defensive — but
#    the dedup contract documented at the task level (Requirement 3.7
#    cross-references the existing ``io_extractor._Accumulator``
#    rule) is encoded explicitly so a future recognizer addition
#    cannot accidentally break the invariant.
# 2. Convert every ``SkipFileEvent`` in the per-file event mapping
#    into a single file-skip-message string of the form
#    ``"skipped <path> (<reason>)"`` so the aggregator's ``_safe_io``
#    wrapper (task 11.2) can prefix each one with the
#    ``"abstract_io: "`` section name and append it to
#    ``degraded_sections``. The composer iterates the mapping in
#    path-sorted order so the produced skip-message list is a
#    deterministic function of the input mapping
#    (Requirement 11.4) and ordered by path for ease of human
#    inspection in the aggregator's ``degraded_sections`` output.
# 3. Preserve the per-recognizer concatenation order in the merged
#    inputs and outputs lists so the function's output is a stable,
#    deterministic projection of the input event mapping. The five
#    recognizers contribute non-overlapping categories
#    (``http_request`` / ``http_response`` / ``scheduled_event`` /
#    ``message_consumed`` / ``message_published`` / ``file_read`` /
#    ``file_written`` / ``cli_argument``) so the concatenation
#    yields a list grouped by recognizer then by path, which is the
#    canonical order downstream consumers see.
#
# The function returns three lists rather than a tuple of (inputs,
# outputs) the way the parent-spec ``io_extractor.extract_io`` does;
# the third element — the per-file skip messages — surfaces the
# parser-level skip events that ``parse_repo`` materialized but the
# four list-shaped sections of the profile cannot themselves carry.
# Task 11.2 wires the third element into ``degraded_sections``.


# ===========================================================================
# Operator-tuning filters (post-recognizer noise suppression)
# ===========================================================================
#
# The recognizers above are spec-compliant: every emission satisfies a
# documented Requirement and the property tests pin those contracts
# verbatim. Real operator deployments however surface two categories
# of low-signal detections that the spec correctly identifies but the
# operator does not want in the rendered profile:
#
# 1. ``file_read`` (and the symmetric ``file_written``) emissions
#    whose call site is the canonical config loader at
#    ``config/config.go``. Every microservice in the operator's group
#    loads its configuration through ``os.ReadFile`` / ``ioutil.ReadFile``
#    here at startup; the detection is correct but conveys nothing
#    that distinguishes one service from another. Filtering by call
#    site (the Go file where the read happens, not the path argument
#    the call is reading) keeps the noise out of the integration view.
#
# 2. ``cli_argument`` binary entry-point emissions for the
#    ``cmd/main.go`` shape. The binary name on this path is derived
#    from ``go.mod`` (Requirement 7.2) and adds no information beyond
#    what the project manifest already surfaces; meanwhile the
#    ``cmd/<name>/main.go`` shape stays because the directory segment
#    distinguishes binaries in a multi-binary repository, and the
#    ``flag.<method>`` call detections stay because they convey the
#    actual flag-name surface the binary accepts.
#
# The filter runs after the recognizers return, before the
# cross-recognizer dedup, so the per-recognizer property tests
# (test_property_08_go_file_io, test_property_09_go_cli_detection)
# continue to assert the unfiltered, spec-compliant behavior. Only
# the composer's public output reflects the operator-tuning prune.


#: Repository-relative call-site paths whose ``file_read`` and
#: ``file_written`` emissions are pruned. The canonical config loader
#: at ``config/config.go`` is the only entry in the operator's data;
#: the set form leaves room for additional well-known startup-time
#: I/O sites without widening the regex check.
_OPERATOR_TUNED_FILE_IO_EXCLUDED_PATHS: Final[frozenset[str]] = frozenset(
    {
        "config/config.go",
    },
)


#: Repository-relative call-site paths whose ``cli_argument`` binary
#: entry-point emissions are pruned. The ``cmd/main.go`` shape derives
#: its binary name from ``go.mod`` and adds no integration-relevant
#: signal. The directory-shaped ``cmd/<name>/main.go`` and the
#: root-level ``main.go`` are intentionally *not* listed — the former
#: discriminates between binaries in multi-binary repos, and pruning
#: the latter is out of scope of the operator's request.
_OPERATOR_TUNED_CLI_BINARY_EXCLUDED_PATHS: Final[frozenset[str]] = frozenset(
    {
        "cmd/main.go",
    },
)


#: Descriptions emitted by the file-I/O and CLI binary-entry-point
#: branches end with a ``" at <path>:<line>"`` Source_Location suffix.
#: The regex below extracts ``<path>`` so the filter can compare it
#: against the operator-excluded sets above. The line-number group is
#: matched (so a description without it is rejected and the entry is
#: conservatively kept) but not captured.
_LOCATION_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"\s+at\s+(?P<path>\S+):\d+$",
)


#: Description prefix every binary-entry-point ``cli_argument``
#: emission carries (see :func:`_try_main_binary_description`). The
#: ``flag.*`` branch uses a different prefix and is therefore not
#: matched by the operator-tuning filter.
_CLI_BINARY_DESCRIPTION_PREFIX: Final[str] = "binary "


def _operator_tuned_location_path(description: str) -> str | None:
    """Return the Source_Location ``<path>`` from a recognizer description.

    The recognizers' description format pins a ``" at <path>:<line>"``
    suffix at the end of every emission. This helper extracts ``<path>``
    so the operator-tuning filter can match it against the excluded
    sets. Returns ``None`` when the description does not end with a
    recognizable suffix; the filter conservatively keeps any entry it
    cannot confidently attribute to a call site.
    """
    match = _LOCATION_SUFFIX_RE.search(description)
    if match is None:
        return None
    return match.group("path")


def _is_operator_tuned_file_io_input(entry: AbstractInput) -> bool:
    """True when an ``AbstractInput`` is a ``file_read`` at an excluded call site."""
    if entry.category is not AbstractInputCategory.FILE_READ:
        return False
    path = _operator_tuned_location_path(entry.description)
    return path in _OPERATOR_TUNED_FILE_IO_EXCLUDED_PATHS


def _is_operator_tuned_file_io_output(entry: AbstractOutput) -> bool:
    """True when an ``AbstractOutput`` is a ``file_written`` at an excluded call site.

    Symmetric with :func:`_is_operator_tuned_file_io_input` so an
    ``os.OpenFile`` call at the excluded path that resolves to the
    undecidable "both" verdict (Requirement 6.3) is pruned from
    both the input and the output list together.
    """
    if entry.category is not AbstractOutputCategory.FILE_WRITTEN:
        return False
    path = _operator_tuned_location_path(entry.description)
    return path in _OPERATOR_TUNED_FILE_IO_EXCLUDED_PATHS


def _is_operator_tuned_cli_input(entry: AbstractInput) -> bool:
    """True when an ``AbstractInput`` is a binary-entry-point at an excluded call site.

    The check guards on the ``"binary "`` description prefix so the
    ``flag.<method>`` branch (which uses a different prefix and
    encodes real CLI surface) is left intact even when the file lives
    at one of the excluded paths.
    """
    if entry.category is not AbstractInputCategory.CLI_ARGUMENT:
        return False
    if not entry.description.startswith(_CLI_BINARY_DESCRIPTION_PREFIX):
        return False
    path = _operator_tuned_location_path(entry.description)
    return path in _OPERATOR_TUNED_CLI_BINARY_EXCLUDED_PATHS


def extract_go_io(
    repository_contents: RepositoryContents,
    events_by_file: Mapping[str, list[GoEvent]],
) -> tuple[list[AbstractInput], list[AbstractOutput], list[str]]:
    """Extract Go-language abstract I/O from the per-file event mapping.

    Composes the five package-internal Go I/O recognizers (HTTP,
    scheduler, ActiveMQ, file I/O, CLI) into a single entry point,
    matching the design §"go.go_io" contract.

    Args:
        repository_contents: The Repository_Contents snapshot for the
            project under analysis. Used by the CLI recognizer to read
            ``go.mod`` for module-path-derived binary names
            (Requirements 7.2, 7.3); the other four recognizers
            consume only the per-file event mapping and ignore this
            argument.
        events_by_file: Mapping from repository-relative file path to
            the list of ``GoEvent`` records produced for that file by
            :func:`project_analyzer.go.go_parser.parse_repo`. Files
            whose only event is a :class:`SkipFileEvent` (build
            constraint, cgo, tokenization failure) contribute zero
            detections but surface as one entry in the third return
            value (Requirement 10.4).

    Returns:
        A 3-tuple ``(inputs, outputs, file_skip_messages)`` where:

        * ``inputs`` is a deduplicated list of :class:`AbstractInput`
          records (categories ``http_request``, ``scheduled_event``,
          ``message_consumed``, ``file_read``, ``cli_argument``),
          grouped by recognizer then by path-sorted file order. May
          be empty.
        * ``outputs`` is a deduplicated list of
          :class:`AbstractOutput` records (categories
          ``http_response``, ``message_published``, ``file_written``),
          grouped by recognizer then by path-sorted file order. May
          be empty.
        * ``file_skip_messages`` is a path-sorted list of skip
          messages of the form ``"skipped <path> (<reason>)"`` —
          one per :class:`SkipFileEvent` in the input mapping
          (Requirement 10.4 verbatim reason text preserved). May be
          empty.

        Dedup contract: every emitted record has a unique
        ``(category, description)`` pair across all five recognizers
        (Requirement 3.7 cross-references the
        :class:`io_extractor._Accumulator` rule). The composer never
        raises; per-file failures are surfaced through the
        ``file_skip_messages`` list, and the aggregator's outer
        ``try/except Exception`` (task 11.2) catches any programming
        bug in the recognizers themselves.
    """

    # --- Step 1: run each recognizer once ---------------------------------
    #
    # Each recognizer iterates ``events_by_file`` in path-sorted order
    # itself (Requirement 11.4) and dedups its own emissions; the
    # composer just concatenates the results. The recognizer order
    # below — HTTP, scheduler, ActiveMQ, file I/O, CLI — matches the
    # order documented at design §"go.go_io" sections 1 through 7 and
    # the task list ordering at tasks.md §7.1 through §7.9. Changing
    # this order would reorder the merged lists; consumers that
    # depend on stable list order across releases should be aware.
    http_inputs, http_outputs = _extract_http_routes(events_by_file)
    scheduler_inputs = _extract_schedulers(events_by_file)
    activemq_inputs, activemq_outputs = _extract_activemq_io(events_by_file)
    file_read_inputs, file_write_outputs = _extract_file_io(events_by_file)
    cli_inputs = _extract_cli_entry_points(repository_contents, events_by_file)

    # --- Step 1.5: operator-tuning noise suppression ----------------------
    #
    # Each recognizer's output is spec-compliant on its own. The
    # composer's public callers however want two well-known noise
    # categories pruned before they reach the rendered profile:
    #
    # * ``file_read`` / ``file_written`` whose call site is the
    #   canonical config loader (``config/config.go``). Every
    #   microservice loads config here; the detection is correct but
    #   adds no service-distinguishing signal.
    # * ``cli_argument`` binary entry-point at ``cmd/main.go``. The
    #   binary name on this path is the ``go.mod`` module name,
    #   already surfaced through other channels; the
    #   ``cmd/<name>/main.go`` and ``flag.<method>`` detections stay.
    #
    # The prune happens here (after the per-recognizer dedup, before
    # the cross-recognizer dedup) so the per-recognizer property
    # tests continue to assert the unfiltered spec contract; only the
    # composer's public output reflects the operator-tuning decision.
    file_read_inputs = [
        entry
        for entry in file_read_inputs
        if not _is_operator_tuned_file_io_input(entry)
    ]
    file_write_outputs = [
        entry
        for entry in file_write_outputs
        if not _is_operator_tuned_file_io_output(entry)
    ]
    cli_inputs = [
        entry
        for entry in cli_inputs
        if not _is_operator_tuned_cli_input(entry)
    ]

    # --- Step 2: cross-recognizer dedup -----------------------------------
    #
    # Each recognizer's contract guarantees per-recognizer dedup. The
    # composer adds a cross-recognizer pass so any (category,
    # description) pair shared between two recognizers — which the
    # current recognizer set never produces, but a future addition
    # could — surfaces only once. The pass preserves first-seen
    # order, matching ``io_extractor._Accumulator``'s rule.
    inputs = _dedup_inputs(
        http_inputs,
        scheduler_inputs,
        activemq_inputs,
        file_read_inputs,
        cli_inputs,
    )
    outputs = _dedup_outputs(
        http_outputs,
        activemq_outputs,
        file_write_outputs,
    )

    # --- Step 3: SkipFileEvent → file_skip_messages -----------------------
    #
    # The third return value carries one entry per ``SkipFileEvent``
    # in the input mapping. The aggregator's ``_safe_io`` wrapper
    # (task 11.2) prefixes each entry with ``"abstract_io: "`` and
    # appends it to ``degraded_sections``, producing the structured
    # strings the design pins at §"Aggregator Integration".
    file_skip_messages = _collect_file_skip_messages(events_by_file)

    return inputs, outputs, file_skip_messages


def _dedup_inputs(
    *sources: list[AbstractInput],
) -> list[AbstractInput]:
    """Concatenate per-recognizer input lists with cross-recognizer dedup.

    Each source list is already deduplicated by ``(category,
    description)`` per its own recognizer's contract. This helper
    runs one extra pass over the concatenation so a hypothetical
    overlap between recognizers (e.g. an HTTP and a file-I/O
    recognizer agreeing on the same description string) is recorded
    only once. The current recognizer set never produces such an
    overlap because each recognizer emits a distinct
    :class:`AbstractInputCategory` value, but encoding the dedup
    here keeps Requirement 3.7's contract explicit at the composition
    boundary and protects against future recognizer additions.

    Order is preserved: an entry appears in the returned list at the
    position of its first occurrence across the input sources, in the
    sources' concatenation order. This matches the
    :class:`io_extractor._Accumulator` rule used by the parent-spec
    extractor for non-Go languages (Requirement 3.7 cross-reference).
    """

    seen: set[tuple[AbstractInputCategory, str]] = set()
    merged: list[AbstractInput] = []
    for source in sources:
        for entry in source:
            key = (entry.category, entry.description)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    return merged


def _dedup_outputs(
    *sources: list[AbstractOutput],
) -> list[AbstractOutput]:
    """Concatenate per-recognizer output lists with cross-recognizer dedup.

    Mirrors :func:`_dedup_inputs` for the
    :class:`AbstractOutput` half of the composer's return value. The
    helper is split from the input variant because Python's typing
    system cannot express "either ``AbstractInput`` or
    ``AbstractOutput`` but not a mix" in a single function signature;
    two narrowly-typed helpers keep mypy happy without runtime
    overhead.
    """

    seen: set[tuple[AbstractOutputCategory, str]] = set()
    merged: list[AbstractOutput] = []
    for source in sources:
        for entry in source:
            key = (entry.category, entry.description)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
    return merged


def _collect_file_skip_messages(
    events_by_file: Mapping[str, list[GoEvent]],
) -> list[str]:
    """Return one ``"skipped <path> (<reason>)"`` message per ``SkipFileEvent``.

    Iterates the event mapping in path-sorted order so the produced
    list is a deterministic function of the input mapping
    (Requirement 11.4). The reason string is recorded verbatim,
    preserving the canonical strings ``parse_repo`` emits per
    Requirement 10.4:

    * ``"tokenization failed: <detail>"``
    * ``"build constraint requires toolchain"``
    * ``"cgo directive requires toolchain"``

    A file may carry at most one ``SkipFileEvent`` in practice (the
    recognizer emits the skip as the file's sole event), but the
    helper handles multiple skip events per file by emitting one
    message per event, in source order. This is defensive — the
    parser's contract guarantees a single ``SkipFileEvent`` per
    skipped file — but encoding it explicitly costs nothing and
    keeps the composer's behavior independent of the parser's
    internal invariant.

    The ``"skipped"`` prefix and the ``"<path> (<reason>)"`` body
    match the format mandated by task 7.11; the aggregator's
    ``_safe_io`` wrapper (task 11.2) prefixes the result with
    ``"abstract_io: "`` to produce the full structured
    ``degraded_sections`` entry pinned by design §"Aggregator
    Integration".
    """

    messages: list[str] = []
    for path in sorted(events_by_file):
        for event in events_by_file[path]:
            if isinstance(event, SkipFileEvent):
                messages.append(f"skipped {path} ({event.reason})")
    return messages

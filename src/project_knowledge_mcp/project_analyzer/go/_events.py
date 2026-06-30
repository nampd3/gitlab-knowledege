"""Internal event and token types for the Go sub-analyzers.

These types are intermediate representations passed between sibling modules
inside the ``project_analyzer.go`` sub-package and never cross the public
interface of the parent spec. Public model types (``AbstractInput``,
``AbstractOutput``, ``ExternalServiceDependency``, ``DatabaseTableDependency``,
``SourceLocation``) live in ``project_knowledge_mcp.models``.

This module currently defines the tokenizer-level types (``GoTokenKind`` and
``GoToken``). Recognizer event types (``ImportEvent``, ``FuncDeclEvent``,
``MethodCallEvent``, ``StructLitEvent``, ``PackageDocCommentEvent``,
``ModFileModuleEvent``, ``BuildConstraintEvent``, ``CgoDirectiveEvent``,
``SkipFileEvent``) and the ``ArgRef`` tagged union are appended in task 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class GoTokenKind(StrEnum):
    """Kind tag for a single Go source token.

    Mirrors the enum specified in design §4. Comments and newlines are
    emitted as first-class tokens (rather than stripped) so the recognizer
    can reconstruct package doc comments, detect build constraints, and
    handle Go's automatic semicolon insertion at line boundaries.
    """

    PACKAGE_KEYWORD = "package"
    IMPORT_KEYWORD = "import"
    FUNC_KEYWORD = "func"
    TYPE_KEYWORD = "type"
    STRUCT_KEYWORD = "struct"
    INTERFACE_KEYWORD = "interface"
    IDENTIFIER = "identifier"
    STRING_LITERAL = "string"
    RAW_STRING_LITERAL = "raw_string"
    NUMBER_LITERAL = "number"
    LBRACE = "lbrace"  # {
    RBRACE = "rbrace"  # }
    LPAREN = "lparen"  # (
    RPAREN = "rparen"  # )
    LBRACKET = "lbracket"  # [
    RBRACKET = "rbracket"  # ]
    COMMA = "comma"
    DOT = "dot"
    COLON = "colon"
    SEMICOLON = "semicolon"
    STAR = "star"  # *
    AMPERSAND = "ampersand"  # &
    ASSIGN = "assign"  # = or :=
    LINE_COMMENT = "line_comment"
    BLOCK_COMMENT = "block_comment"
    STRUCT_TAG = "struct_tag"  # backtick string immediately following a struct field decl
    BUILD_CONSTRAINT_COMMENT = "build_constraint"  # //go:build ... or // +build ...
    CGO_PRAGMA_COMMENT = "cgo_pragma"  # // #cgo ...
    NEWLINE = "newline"
    WHITESPACE = "whitespace"
    OTHER_OPERATOR = "operator"  # any other operator atom; opaque to the recognizer


@dataclass(frozen=True, slots=True)
class GoToken:
    """A single token produced by ``tokenize_go_source``.

    Attributes:
        kind: The token's classification.
        text: The exact source text the token spans.
        line: 1-indexed line number where the token starts.
        column: 1-indexed rune offset on the line where the token starts.
    """

    kind: GoTokenKind
    text: str
    line: int
    column: int


# ---------------------------------------------------------------------------
# ArgRef tagged union — atomic argument references
# ---------------------------------------------------------------------------
#
# ``MethodCallEvent.args`` and ``StructLitEvent.fields`` carry argument
# references rather than raw token slices, so sub-analyzers can match on
# argument shape without re-tokenizing. Every variant is a frozen, slotted
# dataclass so equality, hashing, and pattern-matching work uniformly.


@dataclass(frozen=True, slots=True)
class StringLitArg:
    """A string literal argument (regular ``"..."`` or raw ``\u0060...\u0060``).

    ``value`` is the unquoted contents of the literal.
    """

    value: str


@dataclass(frozen=True, slots=True)
class NumberLitArg:
    """A numeric literal argument; the raw source text is preserved."""

    text: str


@dataclass(frozen=True, slots=True)
class IdentArg:
    """A bare identifier argument (e.g. ``handler`` in ``mux.HandleFunc(p, handler)``)."""

    name: str


@dataclass(frozen=True, slots=True)
class DottedArg:
    """A dotted identifier path argument.

    For example ``cfg.JobCfg.CronSchedule`` becomes
    ``DottedArg(parts=("cfg", "JobCfg", "CronSchedule"))``.
    """

    parts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnknownArg:
    """Fallback for argument expressions the recognizer does not classify.

    Carries the raw source slice so downstream tooling can still inspect it
    without re-tokenizing. The recognizer prefers a more specific variant
    whenever possible.
    """

    text: str


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImportEvent:
    """An ``import "<path>"`` declaration.

    Attributes:
        path: The import path as a string-literal value (without quotes).
        alias: Local alias when present (e.g. ``import f "fmt"`` -> ``"f"``);
            ``None`` for unaliased imports.
        file_path: The Go source file the import was declared in.
        line: 1-indexed line of the import line within ``file_path``.
    """

    path: str
    alias: str | None
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class FuncDeclEvent:
    """A top-level function or method declaration.

    Attributes:
        name: Function or method name.
        receiver_type: Receiver type for methods (e.g. ``"*http.Server"``);
            ``None`` for free functions.
        file_path: Go source file containing the declaration.
        line: 1-indexed line of the ``func`` keyword.
        body_token_range: ``(start_index, end_index)`` half-open slice into
            the file's token stream covering the function body, used by
            sub-analyzers that need to walk nested calls.
    """

    name: str
    receiver_type: str | None
    file_path: str
    line: int
    body_token_range: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MethodCallEvent:
    """A ``<receiver>.<method>(<args>)`` call.

    ``receiver_chain`` is the dotted-name path leading to the method
    (e.g. ``("mux",)``, ``("c",)``, or ``("http",)`` for the package-level
    ``http.ListenAndServe`` form). An empty tuple denotes an unqualified
    call such as ``foo(...)``.

    ``args`` is a tuple of positional argument references in source order.
    """

    receiver_chain: tuple[str, ...]
    method_name: str
    args: tuple["ArgRef", ...]
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class StructLitEvent:
    """A composite literal with named-field syntax.

    Recognized forms: ``T{...}``, ``&T{...}``, ``*T{...}``, ``pkg.T{...}``.
    Positional struct literals are not used by the recognized patterns and
    are not emitted by the recognizer.

    Attributes:
        type_name: The struct type's name (e.g. ``"JmsConfig"``,
            ``"PoolServiceRequest"``).
        package_alias: The package alias prefix when present
            (e.g. ``"activemq"``); ``None`` for unqualified types.
        fields: Tuple of ``(field_name, value)`` pairs in source order;
            named fields only.
        is_pointer: ``True`` for ``&T{...}`` or ``*T{...}``; ``False`` for
            ``T{...}``.
        file_path: Go source file containing the literal.
        line: 1-indexed line of the literal's opening token.
    """

    type_name: str
    package_alias: str | None
    fields: tuple[tuple[str, "ArgRef"], ...]
    is_pointer: bool
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class StructLitArg:
    """An argument that is itself a composite literal."""

    event: StructLitEvent


@dataclass(frozen=True, slots=True)
class CallArg:
    """An argument that is itself a method call (e.g. ``cron.WithSeconds()``)."""

    call: MethodCallEvent


# Tagged union of all argument-reference variants. Sub-analyzers dispatch on
# variant identity (``isinstance`` or ``match``) to extract typed payloads.
ArgRef = (
    StringLitArg
    | NumberLitArg
    | IdentArg
    | DottedArg
    | StructLitArg
    | CallArg
    | UnknownArg
)


@dataclass(frozen=True, slots=True)
class PackageDocCommentEvent:
    """Package doc comment block.

    The contiguous comment block immediately preceding the ``package <name>``
    declaration with no blank line between. ``text`` is the joined comment
    body with leading ``//``, ``/*``, and trailing ``*/`` markers stripped
    and internal whitespace collapsed.
    """

    text: str
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class ModFileModuleEvent:
    """The ``module <module-path>`` declaration in a ``go.mod`` file.

    Emitted only by the ``go.mod`` recognizer entry point, never by
    ``recognize_constructs``.

    Attributes:
        module_path: The verbatim module path (e.g.
            ``"github.com/acme/payment-service"``).
        leading_comment: Body of a ``//``-comment immediately preceding the
            ``module`` line with no blank-line gap, stripped of ``//`` and
            surrounding whitespace; ``None`` when absent.
        trailing_comment: Body of a same-line trailing ``//``-comment,
            stripped of ``//`` and surrounding whitespace; ``None`` when
            absent.
        file_path: Always ``"go.mod"``.
        line: 1-indexed line of the ``module`` line.
    """

    module_path: str
    leading_comment: str | None
    trailing_comment: str | None
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class BuildConstraintEvent:
    """A ``//go:build`` or ``// +build`` constraint line.

    ``expression`` is the verbatim constraint expression. Trivial cases
    (empty constraints, the ``!ignore`` whitelisted form, and empty
    ``// +build`` lines) are emitted with ``expression=""`` and ignored by
    sub-analyzers; non-trivial expressions cause the recognizer to emit a
    ``SkipFileEvent`` instead.
    """

    expression: str
    kind: Literal["go_build", "plus_build"]
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class CgoDirectiveEvent:
    """An ``import "C"`` declaration (with or without surrounding ``// #cgo``
    pragmas or ``/* ... */`` C blocks). Causes the recognizer to emit a
    ``SkipFileEvent`` for the file."""

    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class SkipFileEvent:
    """Emitted as the **only** event for a file that should not contribute
    detections.

    ``reason`` is one of the canonical strings:

    - ``"tokenization failed: <detail>"``
    - ``"build constraint requires toolchain"``
    - ``"cgo directive requires toolchain"``

    ``line`` is ``1`` when the reason is whole-file (build constraint, cgo)
    and the offending construct's line number when the reason is local
    (tokenization failure).
    """

    reason: str
    file_path: str
    line: int


# Tagged union of every event type emitted by ``recognize_constructs`` and
# ``parse_go_mod``. Sub-analyzers iterate over ``list[GoEvent]`` and dispatch
# on variant identity.
GoEvent = (
    ImportEvent
    | FuncDeclEvent
    | MethodCallEvent
    | StructLitEvent
    | PackageDocCommentEvent
    | ModFileModuleEvent
    | BuildConstraintEvent
    | CgoDirectiveEvent
    | SkipFileEvent
)


# ---------------------------------------------------------------------------
# Internal helper records
# ---------------------------------------------------------------------------
#
# These records are the seam between the Go recognizer's neutral event stream
# and the per-sub-analyzer translation into existing public model types
# (``AbstractInput``, ``AbstractOutput``, ``ExternalServiceDependency``,
# ``DatabaseTableDependency``). They never cross the public interface of the
# parent spec.


@dataclass(frozen=True, slots=True)
class GoPurposeCandidates:
    """Purpose-summary candidates contributed by the Go layer.

    The aggregator interleaves these into the parent spec's existing
    candidate order at the documented priority positions; this record does
    not itself decide which candidate wins.
    """

    gomod_comment: str | None
    gomod_module_path: str | None
    package_doc_comment: str | None


@dataclass(frozen=True, slots=True)
class RouteRegistration:
    """An HTTP route registration recognized inside ``go_io``.

    ``path`` carries the route path verbatim when the first call argument is
    a string literal, or a placeholder of the form
    ``"<dynamic at <file>:<line> on <recv>>"`` otherwise.
    """

    method: str
    path: str
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class SchedulerRegistration:
    """A scheduler registration recognized inside ``go_io``.

    ``schedule`` is the literal text of the schedule expression, or
    ``"<dynamic>"`` for non-literal expressions. ``seconds_precision`` is
    ``True`` for ``cron.New(cron.WithSeconds())``, ``False`` for plain
    ``cron.New()``, and ``None`` for ``time.NewTicker`` /
    ``time.AfterFunc``. ``malformed`` is ``True`` when the call shape was
    recognized but its arguments could not be classified (Requirement 4.5).
    """

    library: Literal["cron", "time"]
    schedule: str
    seconds_precision: bool | None
    malformed: bool
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class ActiveMQCall:
    """An ActiveMQ call recognized inside ``go_io`` and
    ``go_external_services``.

    ``destination`` is populated for ``"consume"`` and ``"publish"``
    directions when the destination is a string literal, ``"<dynamic>"``
    when it is not, and ``None`` for ``"connect"``. ``broker_url`` is
    populated only when ``direction == "connect"``.
    """

    direction: Literal["consume", "publish", "connect"]
    destination: str | None
    broker_url: str | None
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class PoolServiceCall:
    """A pool-service database call recognized inside ``go_db_tables`` and
    ``go_external_services``.

    ``via`` indicates which of the three recognized call shapes matched:

    - ``"pb_executequery"``: ``pb.PoolAPIClient.ExecuteQuery`` on a
      ``pb.PoolExecuteQueryRequest``.
    - ``"dbadapter_poolexecutequery"``: ``IPoolAPIAdapter.PoolExecuteQuery``
      on a ``model.PoolServiceRequest``.
    - ``"wrapper_forward"``: an in-house wrapper struct literal carrying a
      ``QueryString`` field that is passed positionally into one of
      ``PoolExecuteQuery``, ``Execute``, or ``ExecuteRaw``.

    ``sql_text`` is the extracted SQL statement when the ``QueryString``
    field's value can be resolved to a literal or a ``fmt.Sprintf`` format
    string, or ``None`` when extraction was skipped.
    """

    via: Literal["pb_executequery", "dbadapter_poolexecutequery", "wrapper_forward"]
    sql_text: str | None
    file_path: str
    line: int

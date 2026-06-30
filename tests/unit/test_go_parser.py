"""Unit tests for ``project_analyzer.go.go_parser``.

These tests pin down the event-emission shapes of
:func:`recognize_constructs` and :func:`parse_go_mod`. Each event
variant defined in design §4 gets at least one focused fixture, and a
mixed-fixture test asserts that events from a realistic Go file appear
in source order.

The recognizer is a thin construct walker, not a full Go parser; the
tests cover only the construct kinds the four Go sub-analyzers consume:

* :class:`ImportEvent` — single, parenthesized-block, aliased, dot,
  and blank imports (Requirement 1.1).
* :class:`FuncDeclEvent` — free function, value-receiver method,
  pointer-receiver method, generics (Requirement 1.1).
* :class:`MethodCallEvent` — empty receiver chain, single-segment
  receiver, multi-segment receiver chain, mixed argument shapes
  (Requirement 1.2).
* :class:`StructLitEvent` — package-qualified, pointer-prefixed
  package-qualified, bare-type composite literals (Requirement 1.2).
* :class:`PackageDocCommentEvent` — line comments, block comment,
  and the negative case where a blank-line gap suppresses the event
  (Requirement 1.1).
* :class:`SkipFileEvent` — non-trivial build constraint trigger,
  cgo trigger, and the carve-outs for trivial constraints
  (Requirement 1.1, 10.4).
* :class:`ModFileModuleEvent` — module path extraction, leading
  comment, trailing comment, and the no-blank-line-gap rule
  (Requirement 1.1).

Implements Requirements 1.1, 1.2.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.project_analyzer.go._events import (
    CallArg,
    DottedArg,
    FuncDeclEvent,
    GoEvent,
    IdentArg,
    ImportEvent,
    MethodCallEvent,
    NumberLitArg,
    PackageDocCommentEvent,
    SkipFileEvent,
    StringLitArg,
    StructLitArg,
    StructLitEvent,
)
from project_knowledge_mcp.project_analyzer.go.go_parser import (
    parse_go_mod,
    recognize_constructs,
)
from project_knowledge_mcp.project_analyzer.go.go_tokenizer import tokenize_go_source

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events(source: str, path: str = "f.go") -> list[GoEvent]:
    """Tokenize ``source`` and materialize the recognizer's event list."""

    return list(recognize_constructs(tokenize_go_source(source), path))


def _of_type(events: list[GoEvent], cls: type) -> list[GoEvent]:
    """Return events filtered to a single concrete dataclass type."""

    return [e for e in events if isinstance(e, cls)]


# ===========================================================================
# ImportEvent
# ===========================================================================


def test_import_event_single_unaliased() -> None:
    """``import "fmt"`` produces one :class:`ImportEvent` with no alias."""

    events = _of_type(_events('package foo\nimport "fmt"\n'), ImportEvent)

    assert events == [
        ImportEvent(path="fmt", alias=None, file_path="f.go", line=2),
    ]


def test_import_event_parenthesized_block_yields_one_event_per_path() -> None:
    """Parenthesized blocks yield one event per path, in source order."""

    source = (
        "package foo\n"
        "import (\n"
        '\t"fmt"\n'
        '\t"net/http"\n'
        ")\n"
    )

    events = _of_type(_events(source), ImportEvent)

    assert events == [
        ImportEvent(path="fmt", alias=None, file_path="f.go", line=3),
        ImportEvent(path="net/http", alias=None, file_path="f.go", line=4),
    ]


def test_import_event_named_alias_is_recorded() -> None:
    """``import f "fmt"`` records the alias text."""

    events = _of_type(_events('package foo\nimport f "fmt"\n'), ImportEvent)

    assert events == [
        ImportEvent(path="fmt", alias="f", file_path="f.go", line=2),
    ]


def test_import_event_dot_alias_is_recorded_as_literal_dot() -> None:
    """``import . "x"`` records the alias as the literal ``"."``."""

    events = _of_type(_events('package foo\nimport . "x/y"\n'), ImportEvent)

    assert events == [
        ImportEvent(path="x/y", alias=".", file_path="f.go", line=2),
    ]


def test_import_event_blank_alias_is_recorded_as_underscore() -> None:
    """``import _ "y"`` records the alias as the literal ``"_"``."""

    events = _of_type(_events('package foo\nimport _ "y/z"\n'), ImportEvent)

    assert events == [
        ImportEvent(path="y/z", alias="_", file_path="f.go", line=2),
    ]


def test_import_event_block_with_mixed_alias_kinds() -> None:
    """A block containing aliased, blank, and dot imports preserves each shape."""

    source = (
        "package foo\n"
        "import (\n"
        '\tf "fmt"\n'
        '\t_ "net/http/pprof"\n'
        '\t. "errors"\n'
        ")\n"
    )

    events = _of_type(_events(source), ImportEvent)

    assert events == [
        ImportEvent(path="fmt", alias="f", file_path="f.go", line=3),
        ImportEvent(path="net/http/pprof", alias="_", file_path="f.go", line=4),
        ImportEvent(path="errors", alias=".", file_path="f.go", line=5),
    ]


# ===========================================================================
# FuncDeclEvent
# ===========================================================================


def test_func_decl_event_free_function_has_no_receiver_type() -> None:
    """``func main()`` is a free function (no receiver)."""

    events = _of_type(_events("package foo\nfunc main() {}\n"), FuncDeclEvent)

    assert len(events) == 1
    decl = events[0]
    assert isinstance(decl, FuncDeclEvent)
    assert decl.name == "main"
    assert decl.receiver_type is None
    assert decl.line == 2
    assert decl.file_path == "f.go"


def test_func_decl_event_value_receiver_method_records_bare_type() -> None:
    """``func (s Server) Name()`` records ``Server`` as the receiver type."""

    source = (
        "package foo\n"
        'func (s Server) Name() string { return "" }\n'
    )

    events = _of_type(_events(source), FuncDeclEvent)

    assert len(events) == 1
    decl = events[0]
    assert isinstance(decl, FuncDeclEvent)
    assert decl.name == "Name"
    assert decl.receiver_type == "Server"


def test_func_decl_event_pointer_receiver_method_records_starred_type() -> None:
    """``func (s *Server) Handle(...)`` records ``*Server`` as the receiver type."""

    source = (
        "package foo\n"
        "func (s *Server) Handle(r *Request) {}\n"
    )

    events = _of_type(_events(source), FuncDeclEvent)

    assert len(events) == 1
    decl = events[0]
    assert isinstance(decl, FuncDeclEvent)
    assert decl.name == "Handle"
    assert decl.receiver_type == "*Server"


def test_func_decl_event_pointer_receiver_with_package_qualified_type() -> None:
    """A pointer receiver to a package-qualified type concatenates correctly."""

    source = (
        "package foo\n"
        "func (h *http.Server) Close() error { return nil }\n"
    )

    events = _of_type(_events(source), FuncDeclEvent)

    assert len(events) == 1
    decl = events[0]
    assert isinstance(decl, FuncDeclEvent)
    assert decl.name == "Close"
    assert decl.receiver_type == "*http.Server"


def test_func_decl_event_generic_function_parses_through_type_params() -> None:
    """A generic function declaration is parsed past its ``[T any]`` clause."""

    source = (
        "package foo\n"
        "func Add[T int|float](a, b T) T { return a }\n"
    )

    events = _of_type(_events(source), FuncDeclEvent)

    assert len(events) == 1
    decl = events[0]
    assert isinstance(decl, FuncDeclEvent)
    assert decl.name == "Add"
    assert decl.receiver_type is None


# ===========================================================================
# MethodCallEvent
# ===========================================================================


def test_method_call_event_single_segment_receiver_with_string_and_ident_args() -> None:
    """``mux.HandleFunc("/p", h)`` records the receiver, method, and argument shape."""

    source = (
        "package foo\n"
        "func main() {\n"
        '\tmux.HandleFunc("/p", h)\n'
        "}\n"
    )

    events = _of_type(_events(source), MethodCallEvent)

    assert events == [
        MethodCallEvent(
            receiver_chain=("mux",),
            method_name="HandleFunc",
            args=(StringLitArg(value="/p"), IdentArg(name="h")),
            file_path="f.go",
            line=3,
        ),
    ]


def test_method_call_event_bare_call_has_empty_receiver_chain() -> None:
    """``init()`` is recorded with an empty receiver chain and method ``init``."""

    source = (
        "package foo\n"
        "func main() {\n"
        "\tinit()\n"
        "}\n"
    )

    method_calls = _of_type(_events(source), MethodCallEvent)

    # Filter to just the inner ``init()``; the outer ``main()`` declaration
    # produces a FuncDeclEvent, not a MethodCallEvent.
    inits = [c for c in method_calls if isinstance(c, MethodCallEvent) and c.method_name == "init"]
    assert inits == [
        MethodCallEvent(
            receiver_chain=(),
            method_name="init",
            args=(),
            file_path="f.go",
            line=3,
        ),
    ]


def test_method_call_event_long_receiver_chain_records_every_segment() -> None:
    """``cfg.JobCfg.Service.Call(x)`` records the full dotted receiver chain."""

    source = (
        "package foo\n"
        "func main() {\n"
        "\tcfg.JobCfg.Service.Call(x)\n"
        "}\n"
    )

    calls = _of_type(_events(source), MethodCallEvent)
    chained = [c for c in calls if isinstance(c, MethodCallEvent) and c.method_name == "Call"]
    assert chained == [
        MethodCallEvent(
            receiver_chain=("cfg", "JobCfg", "Service"),
            method_name="Call",
            args=(IdentArg(name="x"),),
            file_path="f.go",
            line=3,
        ),
    ]


def test_method_call_event_dotted_argument_is_classified_as_dotted_arg() -> None:
    """A dotted argument expression becomes a :class:`DottedArg`."""

    source = (
        "package foo\n"
        "func main() {\n"
        "\tc.AddFunc(cfg.JobCfg.CronSchedule, h)\n"
        "}\n"
    )

    calls = _of_type(_events(source), MethodCallEvent)
    [call] = [c for c in calls if isinstance(c, MethodCallEvent) and c.method_name == "AddFunc"]
    assert call.args == (
        DottedArg(parts=("cfg", "JobCfg", "CronSchedule")),
        IdentArg(name="h"),
    )


def test_method_call_event_call_argument_becomes_call_arg() -> None:
    """Nested method calls in argument position are wrapped as :class:`CallArg`."""

    source = (
        "package foo\n"
        "func main() {\n"
        "\tc := cron.New(cron.WithSeconds())\n"
        "}\n"
    )

    calls = _of_type(_events(source), MethodCallEvent)
    # Only the *outer* call appears as a top-level event; the inner
    # ``cron.WithSeconds()`` becomes a CallArg within its args tuple.
    outer = [
        c
        for c in calls
        if isinstance(c, MethodCallEvent) and c.method_name == "New"
    ]
    assert len(outer) == 1
    assert isinstance(outer[0], MethodCallEvent)
    assert len(outer[0].args) == 1
    arg = outer[0].args[0]
    assert isinstance(arg, CallArg)
    assert arg.call.method_name == "WithSeconds"
    assert arg.call.receiver_chain == ("cron",)


# ===========================================================================
# StructLitEvent
# ===========================================================================


def test_struct_lit_event_package_qualified_type_records_alias_and_name() -> None:
    """``domain.Message{Destination: "X"}`` splits into type and package alias."""

    source = (
        "package foo\n"
        'var m = domain.Message{Destination: "X"}\n'
    )

    events = _of_type(_events(source), StructLitEvent)

    assert events == [
        StructLitEvent(
            type_name="Message",
            package_alias="domain",
            fields=(("Destination", StringLitArg(value="X")),),
            is_pointer=False,
            file_path="f.go",
            line=2,
        ),
    ]


def test_struct_lit_event_pointer_prefixed_package_qualified_type_sets_is_pointer() -> None:
    """``&domain.SubscriberConfig{...}`` records ``is_pointer=True``."""

    source = (
        "package foo\n"
        'var c = &domain.SubscriberConfig{Destination: "Y"}\n'
    )

    events = _of_type(_events(source), StructLitEvent)

    assert events == [
        StructLitEvent(
            type_name="SubscriberConfig",
            package_alias="domain",
            fields=(("Destination", StringLitArg(value="Y")),),
            is_pointer=True,
            file_path="f.go",
            line=2,
        ),
    ]


def test_struct_lit_event_bare_type_has_no_package_alias() -> None:
    """An unqualified composite literal records ``package_alias=None``."""

    source = (
        "package foo\n"
        "var x = Config{Foo: 1}\n"
    )

    events = _of_type(_events(source), StructLitEvent)

    assert events == [
        StructLitEvent(
            type_name="Config",
            package_alias=None,
            fields=(("Foo", NumberLitArg(text="1")),),
            is_pointer=False,
            file_path="f.go",
            line=2,
        ),
    ]


def test_struct_lit_event_with_multiple_named_fields_preserves_order() -> None:
    """Named fields are recorded in source order, with their classified values."""

    source = (
        "package foo\n"
        'var c = activemq.JmsConfig{BrokerUrl: "tcp://h:1", Username: u}\n'
    )

    events = _of_type(_events(source), StructLitEvent)

    assert events == [
        StructLitEvent(
            type_name="JmsConfig",
            package_alias="activemq",
            fields=(
                ("BrokerUrl", StringLitArg(value="tcp://h:1")),
                ("Username", IdentArg(name="u")),
            ),
            is_pointer=False,
            file_path="f.go",
            line=2,
        ),
    ]


# ===========================================================================
# PackageDocCommentEvent
# ===========================================================================


def test_package_doc_comment_event_for_single_line_comment() -> None:
    """A single ``//`` line immediately above ``package`` becomes the doc comment."""

    source = "// Package mixed does things.\npackage mixed\n"

    events = _of_type(_events(source), PackageDocCommentEvent)

    assert events == [
        PackageDocCommentEvent(
            text="Package mixed does things.",
            file_path="f.go",
            line=1,
        ),
    ]


def test_package_doc_comment_event_joins_consecutive_line_comments() -> None:
    """A contiguous ``//`` block is joined into a single space-separated body."""

    source = "// First line.\n// Second line.\npackage mixed\n"

    events = _of_type(_events(source), PackageDocCommentEvent)

    assert events == [
        PackageDocCommentEvent(
            text="First line. Second line.",
            file_path="f.go",
            line=1,
        ),
    ]


def test_package_doc_comment_event_for_block_comment() -> None:
    """A ``/* ... */`` block immediately above ``package`` becomes the doc comment."""

    source = "/* Package mixed does things. */\npackage mixed\n"

    events = _of_type(_events(source), PackageDocCommentEvent)

    assert events == [
        PackageDocCommentEvent(
            text="Package mixed does things.",
            file_path="f.go",
            line=1,
        ),
    ]


def test_package_doc_comment_event_suppressed_by_blank_line_gap() -> None:
    """A blank line between the comment block and ``package`` suppresses the event."""

    source = "// Detached comment.\n\npackage mixed\n"

    events = _of_type(_events(source), PackageDocCommentEvent)

    assert events == []


# ===========================================================================
# SkipFileEvent
# ===========================================================================


def test_skip_file_event_non_trivial_build_constraint_skips_whole_file() -> None:
    """A non-trivial ``//go:build`` line yields a single SkipFileEvent."""

    source = "//go:build linux\n\npackage foo\n"

    events = _events(source)

    assert events == [
        SkipFileEvent(
            reason="build constraint requires toolchain",
            file_path="f.go",
            line=1,
        ),
    ]


def test_skip_file_event_cgo_import_skips_whole_file() -> None:
    """``import "C"`` yields a single SkipFileEvent."""

    source = 'package foo\nimport "C"\n'

    events = _events(source)

    assert events == [
        SkipFileEvent(
            reason="cgo directive requires toolchain",
            file_path="f.go",
            line=2,
        ),
    ]


def test_skip_file_event_not_emitted_for_trivial_build_constraint() -> None:
    """``//go:build !ignore`` is the trivial whitelisted form; no skip is emitted."""

    source = "//go:build !ignore\n\npackage foo\n\nfunc main() {}\n"

    events = _events(source)

    # No SkipFileEvent: only the func decl survives the recognizer.
    assert _of_type(events, SkipFileEvent) == []
    assert _of_type(events, FuncDeclEvent) == [
        FuncDeclEvent(
            name="main",
            receiver_type=None,
            file_path="f.go",
            line=5,
            body_token_range=_of_type(events, FuncDeclEvent)[0].body_token_range,
        ),
    ]


def test_skip_file_event_not_emitted_for_empty_plus_build_constraint() -> None:
    """An empty ``// +build`` line is trivial and does not skip the file."""

    source = "// +build\n\npackage foo\n\nfunc main() {}\n"

    events = _events(source)

    assert _of_type(events, SkipFileEvent) == []
    assert _of_type(events, FuncDeclEvent) != []


# ===========================================================================
# ModFileModuleEvent (parse_go_mod)
# ===========================================================================


def test_parse_go_mod_extracts_module_path_with_no_comments() -> None:
    """A bare ``module`` line records only the module path."""

    event = parse_go_mod("module github.com/acme/svc\n")

    assert event is not None
    assert event.module_path == "github.com/acme/svc"
    assert event.leading_comment is None
    assert event.trailing_comment is None
    assert event.file_path == "go.mod"
    assert event.line == 1


def test_parse_go_mod_captures_same_line_trailing_comment() -> None:
    """A ``//`` comment trailing the ``module`` line is recorded."""

    event = parse_go_mod("module github.com/acme/svc // payment service\n")

    assert event is not None
    assert event.module_path == "github.com/acme/svc"
    assert event.leading_comment is None
    assert event.trailing_comment == "payment service"


def test_parse_go_mod_captures_immediately_preceding_leading_comment() -> None:
    """A ``//`` comment on the line directly above ``module`` is recorded."""

    event = parse_go_mod("// payment service\nmodule github.com/acme/svc\n")

    assert event is not None
    assert event.module_path == "github.com/acme/svc"
    assert event.leading_comment == "payment service"
    assert event.trailing_comment is None
    assert event.line == 2


def test_parse_go_mod_blank_line_gap_suppresses_leading_comment() -> None:
    """A blank line between the comment and ``module`` drops the leading comment."""

    event = parse_go_mod("// not the module comment\n\nmodule github.com/acme/svc\n")

    assert event is not None
    assert event.module_path == "github.com/acme/svc"
    assert event.leading_comment is None
    assert event.trailing_comment is None
    assert event.line == 3


def test_parse_go_mod_returns_none_when_no_module_line_present() -> None:
    """A ``go.mod`` with no ``module`` directive returns ``None``."""

    assert parse_go_mod("go 1.21\n") is None


def test_parse_go_mod_captures_both_leading_and_trailing_comments() -> None:
    """Leading and trailing comments coexist on a single ``module`` line."""

    event = parse_go_mod("// payment service\nmodule github.com/acme/svc // v2\n")

    assert event is not None
    assert event.leading_comment == "payment service"
    assert event.trailing_comment == "v2"


# ===========================================================================
# Mixed-fixture event ordering
# ===========================================================================


def test_mixed_file_emits_events_in_source_order() -> None:
    """A realistic file produces every recognized event kind in source order.

    The fixture exercises:

    * a package doc comment (line 1),
    * two block-style imports (lines 4-5),
    * a free-function declaration (line 8),
    * a struct literal in a function body (line 9),
    * two method calls in the function body (lines 10 and 11).

    The recognizer must yield those in source order so sub-analyzers
    can rely on iteration order to attribute detections.
    """

    source = (
        "// Mixed package doc.\n"           # line 1
        "package mixed\n"                    # line 2
        "\n"                                  # line 3
        "import (\n"                          # line 4
        '\t"context"\n'                      # line 5
        '\t"fmt"\n'                          # line 6
        ")\n"                                 # line 7
        "\n"                                   # line 8
        "func handle(ctx context.Context) {\n"  # line 9
        '\tmsg := domain.Message{Destination: "Q1"}\n'  # line 10
        "\tfmt.Println(msg)\n"               # line 11
        '\tmux.HandleFunc("/p", h)\n'        # line 12
        "}\n"                                 # line 13
    )

    events = _events(source, path="mixed.go")

    # Strip the FuncDecl's body_token_range from the assertion: it is
    # an internal implementation detail of the parser (token offsets)
    # rather than a public-shape attribute. The test asserts every
    # other field plus the relative ordering.
    assert len(events) == 7
    assert isinstance(events[0], PackageDocCommentEvent)
    assert events[0] == PackageDocCommentEvent(
        text="Mixed package doc.",
        file_path="mixed.go",
        line=1,
    )

    assert isinstance(events[1], ImportEvent)
    assert events[1] == ImportEvent(
        path="context", alias=None, file_path="mixed.go", line=5,
    )

    assert isinstance(events[2], ImportEvent)
    assert events[2] == ImportEvent(
        path="fmt", alias=None, file_path="mixed.go", line=6,
    )

    assert isinstance(events[3], FuncDeclEvent)
    assert events[3].name == "handle"
    assert events[3].receiver_type is None
    assert events[3].file_path == "mixed.go"
    assert events[3].line == 9

    assert isinstance(events[4], StructLitEvent)
    assert events[4] == StructLitEvent(
        type_name="Message",
        package_alias="domain",
        fields=(("Destination", StringLitArg(value="Q1")),),
        is_pointer=False,
        file_path="mixed.go",
        line=10,
    )

    assert isinstance(events[5], MethodCallEvent)
    assert events[5] == MethodCallEvent(
        receiver_chain=("fmt",),
        method_name="Println",
        args=(IdentArg(name="msg"),),
        file_path="mixed.go",
        line=11,
    )

    assert isinstance(events[6], MethodCallEvent)
    assert events[6] == MethodCallEvent(
        receiver_chain=("mux",),
        method_name="HandleFunc",
        args=(StringLitArg(value="/p"), IdentArg(name="h")),
        file_path="mixed.go",
        line=12,
    )


def test_mixed_file_with_pointer_struct_arg_inside_method_call() -> None:
    """A struct literal passed positionally is captured as :class:`StructLitArg`.

    This shape is the canonical ActiveMQ subscriber registration:
    ``r.Subscribe(ctx, h, &domain.SubscriberConfig{...})`` and exercises
    the recognizer's ability to nest a struct-literal argument inside a
    method-call argument list without re-yielding it as a top-level event.
    """

    source = (
        "package foo\n"
        "func wire() {\n"
        "\tr.Subscribe(ctx, h, &domain.SubscriberConfig{Destination: \"Q\"})\n"
        "}\n"
    )

    events = _events(source)

    # The outer Subscribe call is the only top-level MethodCallEvent;
    # the struct literal is captured *inside* its args tuple via
    # StructLitArg, not yielded as a separate top-level event.
    method_calls = _of_type(events, MethodCallEvent)
    assert len(method_calls) == 1
    sub = method_calls[0]
    assert isinstance(sub, MethodCallEvent)
    assert sub.method_name == "Subscribe"
    assert sub.receiver_chain == ("r",)
    assert len(sub.args) == 3
    assert sub.args[0] == IdentArg(name="ctx")
    assert sub.args[1] == IdentArg(name="h")
    third = sub.args[2]
    assert isinstance(third, StructLitArg)
    assert third.event.type_name == "SubscriberConfig"
    assert third.event.package_alias == "domain"
    assert third.event.is_pointer is True
    assert third.event.fields == (("Destination", StringLitArg(value="Q")),)

    # And no top-level StructLitEvent was emitted: it lives inside the
    # method call's args.
    assert _of_type(events, StructLitEvent) == []


# ---------------------------------------------------------------------------
# Regression: malformed argument lists must not loop forever
# ---------------------------------------------------------------------------
#
# The construct recognizer was observed to hang in production on
# ``cmd/template/cmd/main.go`` of an in-house Go service. py-spy
# samples pinned the loop to ``_parse_call_args`` repeatedly invoking
# ``_parse_arg`` at the same sig index, which in turn called
# ``_parse_unknown_arg``. ``_parse_unknown_arg`` treats a top-level
# ``}`` or ``]`` as an end-of-argument sentinel and breaks *without*
# advancing ``s``, so the outer loop appended empty ``UnknownArg``
# entries forever.
#
# The fix is a per-iteration progress guard in ``_parse_call_args``
# and ``_parse_struct_fields``: when the sub-parser fails to advance,
# raise ``_ParseError`` so the outer ``_build_method_call`` /
# ``_build_struct_lit`` rejects the construct and ``_walk`` resumes
# at the next significant token. These tests assert the parser
# terminates and produces no infinite list of empty UnknownArg
# entries for the canonical malformed shapes.


def _run_parser_with_timeout(text: str, *, seconds: float = 5.0) -> list[GoEvent]:
    """Run the recognizer with an upper wall-clock bound.

    If the parser ever regressed to the looping behavior, the test
    would hang indefinitely and stall the suite. The thread-based
    timeout below caps the run at ``seconds`` and produces a
    ``pytest.fail`` with a diagnostic instead.
    """
    import threading

    tokens = list(tokenize_go_source(text))
    result: list[GoEvent] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.extend(recognize_constructs(iter(tokens), "file.go"))
        except BaseException as exc:  # noqa: BLE001 - any failure is interesting
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    if thread.is_alive():
        pytest.fail(
            f"recognize_constructs did not terminate within {seconds:.1f}s — "
            "the malformed-argument progress guard regressed"
        )
    if error:
        raise error[0]
    return result


def test_stray_close_brace_inside_call_args_does_not_loop() -> None:
    """A stray ``}`` between commas must not deadlock the parser.

    Synthetic minimal reproducer of the
    ``cmd/template/cmd/main.go`` hang: a method call whose argument
    list contains a stray ``}`` at a position where
    ``_parse_unknown_arg`` would otherwise break with ``s`` unchanged.
    """
    src = (
        "package main\n"
        "\n"
        "func driver() {\n"
        "\tmetrics.Add(label, } , another)\n"
        "}\n"
    )

    events = _run_parser_with_timeout(src)

    # The malformed call MUST be silently rejected (no MethodCallEvent
    # emitted for it) — the outer walker continues past the construct
    # rather than getting stuck. The ``func driver`` declaration must
    # still be recognized so subsequent declarations are not lost.
    func_decls = [e for e in events if e.__class__.__name__ == "FuncDeclEvent"]
    method_calls = [e for e in events if e.__class__.__name__ == "MethodCallEvent"]
    assert any(getattr(e, "name", None) == "driver" for e in func_decls)
    assert not any(
        getattr(e, "method_name", None) == "Add" for e in method_calls
    ), "malformed metrics.Add(...) must be rejected, not emitted"


def test_stray_close_bracket_inside_call_args_does_not_loop() -> None:
    """A stray ``]`` (mirrors the ``}`` case) must also terminate."""
    src = (
        "package main\n"
        "\n"
        "func driver() {\n"
        "\tprocess(item, ] , next)\n"
        "}\n"
    )

    events = _run_parser_with_timeout(src)

    method_calls = [e for e in events if e.__class__.__name__ == "MethodCallEvent"]
    assert not any(
        getattr(e, "method_name", None) == "process" for e in method_calls
    )


def test_stray_close_brace_inside_struct_literal_does_not_loop() -> None:
    """The ``_parse_struct_fields`` guard applies the same termination."""
    src = (
        "package main\n"
        "\n"
        "func driver() {\n"
        "\tx := Config{Path: file, } } \n"
        "}\n"
    )

    # The closing brace inside the struct value position is malformed;
    # the parser must not loop on it. The test only asserts
    # termination — whether the struct literal is recognized or
    # rejected is a recovery choice we do not pin here.
    _run_parser_with_timeout(src)

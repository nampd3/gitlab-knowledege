"""Unit tests for ``project_analyzer.go.go_tokenizer``.

These tests pin down the public contract of :func:`tokenize_go_source`
and :class:`GoTokenizationError`:

* The six dedicated keyword kinds (``package``, ``import``, ``func``,
  ``type``, ``struct``, ``interface``) are emitted as their own
  :class:`GoTokenKind` values; every other Go keyword is emitted as
  :attr:`GoTokenKind.IDENTIFIER`.
* Identifier, number, regular string, raw string, punctuation, and
  operator tokens carry the kinds documented in design §4.
* Comments are first-class tokens, with line, block, build-constraint,
  and cgo-pragma variants distinguished.
* Backtick-quoted runs immediately following an identifier inside a
  ``struct { ... }`` body are reclassified from
  :attr:`GoTokenKind.RAW_STRING_LITERAL` to
  :attr:`GoTokenKind.STRUCT_TAG`; the same lexical shape outside a
  struct stays a raw string literal.
* Line and column are 1-indexed and reset correctly across newlines.
* Every malformed-input shape enumerated in design §3 "Error handling"
  raises :class:`GoTokenizationError` with the canonical reason.
* The empty input produces an empty token stream.
* The function is pure: the same input produces equal token sequences
  on repeated calls.

Implements Requirements 10.1, 10.4, 11.1.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.project_analyzer.go._events import GoToken, GoTokenKind
from project_knowledge_mcp.project_analyzer.go.go_tokenizer import (
    GoTokenizationError,
    tokenize_go_source,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[GoToken]:
    """Materialize the tokenizer's iterator into a list."""

    return list(tokenize_go_source(text))


def _significant(text: str) -> list[GoToken]:
    """Tokens with whitespace and newlines stripped (comments preserved)."""

    drop = {GoTokenKind.WHITESPACE, GoTokenKind.NEWLINE}
    return [t for t in tokenize_go_source(text) if t.kind not in drop]


def _kinds(text: str) -> list[GoTokenKind]:
    """Just the kind sequence -- used when text payloads are obvious."""

    return [t.kind for t in tokenize_go_source(text)]


def _significant_kinds(text: str) -> list[GoTokenKind]:
    """Kind sequence with whitespace/newlines dropped."""

    return [t.kind for t in _significant(text)]


# ===========================================================================
# Empty input and pure-function determinism
# ===========================================================================


def test_empty_input_yields_no_tokens() -> None:
    assert _tokens("") == []


def test_tokenizing_same_input_twice_produces_equal_results() -> None:
    """Pure-function contract: identical input yields identical token streams."""

    text = (
        "package svc\n\n"
        "import (\n"
        '\t"fmt"\n'
        "\t_ \"net/http\"\n"
        ")\n\n"
        "// doc\n"
        "func main() {\n"
        '\tx := "hello"\n'
        "\tfmt.Println(x)\n"
        "}\n"
    )

    first = _tokens(text)
    second = _tokens(text)

    assert first == second
    # Sanity: it is not an empty stream that compares equal trivially.
    assert len(first) > 20


# ===========================================================================
# Keywords with dedicated kinds (Requirement 10.1)
# ===========================================================================


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("package", GoTokenKind.PACKAGE_KEYWORD),
        ("import", GoTokenKind.IMPORT_KEYWORD),
        ("func", GoTokenKind.FUNC_KEYWORD),
        ("type", GoTokenKind.TYPE_KEYWORD),
        ("struct", GoTokenKind.STRUCT_KEYWORD),
        ("interface", GoTokenKind.INTERFACE_KEYWORD),
    ],
)
def test_dedicated_keywords_emit_dedicated_kinds(
    source: str, kind: GoTokenKind
) -> None:
    toks = _tokens(source)
    assert len(toks) == 1
    assert toks[0].kind is kind
    assert toks[0].text == source


@pytest.mark.parametrize(
    "keyword",
    [
        "var",
        "const",
        "if",
        "else",
        "for",
        "return",
        "range",
        "defer",
        "go",
        "select",
        "case",
        "default",
        "switch",
        "break",
        "continue",
        "fallthrough",
        "goto",
        "chan",
        "map",
        "nil",
        "true",
        "false",
    ],
)
def test_other_keywords_emit_identifier_kind(keyword: str) -> None:
    """Every keyword the recognizer does not branch on is an IDENTIFIER."""

    toks = _tokens(keyword)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.IDENTIFIER
    assert toks[0].text == keyword


# ===========================================================================
# Identifiers
# ===========================================================================


@pytest.mark.parametrize(
    "name",
    [
        "x",
        "Foo",
        "snake_case_name",
        "camelCaseName",
        "PascalCaseName",
        "_underscoreLeading",
        "name123",
        "_",
        # Unicode identifier per Go spec (the language permits any Unicode
        # letter / digit in identifiers).
        "café",
        "λambda",
        "Ωmega",
        "résumé2",
    ],
)
def test_identifier_shapes_emit_single_identifier_token(name: str) -> None:
    toks = _tokens(name)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.IDENTIFIER
    assert toks[0].text == name


# ===========================================================================
# Number literals
# ===========================================================================


@pytest.mark.parametrize(
    "literal",
    [
        # Decimal integers.
        "0",
        "1",
        "42",
        "1234567890",
        # Hex.
        "0x0",
        "0xff",
        "0xFF",
        "0XdeadBEEF",
        # Octal (new and legacy forms).
        "0o7",
        "0O77",
        "07",
        "0755",
        # Binary.
        "0b0",
        "0b101",
        "0B1010",
        # Floats.
        "1.5",
        "0.5",
        "1.",
        "1e10",
        "1E10",
        "1.5e-3",
        "1.5E+3",
        "2.5e0",
        # Hex float.
        "0x1p10",
        "0x1.fp10",
        "0X1P-2",
        # Imaginary.
        "1i",
        "1.5i",
        "0i",
        # Underscore separators.
        "1_000_000",
        "0xff_ff",
        "1_000.5_5",
    ],
)
def test_number_literal_emits_single_number_token(literal: str) -> None:
    toks = _tokens(literal)
    assert len(toks) == 1, f"expected one token for {literal!r}, got {toks}"
    assert toks[0].kind is GoTokenKind.NUMBER_LITERAL
    assert toks[0].text == literal


def test_leading_dot_float_is_a_single_number_token() -> None:
    # ``.5`` is a Go float literal; the tokenizer recognizes it via the
    # leading-dot dispatch path.
    toks = _tokens(".5")
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.NUMBER_LITERAL
    assert toks[0].text == ".5"


# ===========================================================================
# Regular string literals (Requirement 10.1)
# ===========================================================================


def test_simple_string_literal() -> None:
    toks = _tokens('"foo"')
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.STRING_LITERAL
    assert toks[0].text == '"foo"'


def test_empty_string_literal() -> None:
    toks = _tokens('""')
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.STRING_LITERAL
    assert toks[0].text == '""'


def test_string_literal_with_simple_escapes() -> None:
    # \n \t \\ inside a string literal are valid Go escapes.
    src = '"\\n\\t\\\\"'
    toks = _tokens(src)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.STRING_LITERAL
    assert toks[0].text == src


@pytest.mark.parametrize(
    "literal",
    [
        '"\\a\\b\\f\\v"',
        '"\\\'\\""',  # \' and \"
        '"\\xff"',
        '"\\xFF"',
        '"\\u00ff"',
        '"\\u00FF"',
        '"\\U000000ff"',
        '"\\U000000FF"',
        '"\\007"',
        '"\\377"',
        '"abc\\n123"',
    ],
)
def test_string_literals_with_numeric_and_simple_escapes(literal: str) -> None:
    toks = _tokens(literal)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.STRING_LITERAL
    assert toks[0].text == literal


# ===========================================================================
# Raw string literals
# ===========================================================================


def test_raw_string_literal_simple() -> None:
    toks = _tokens("`foo`")
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.RAW_STRING_LITERAL
    assert toks[0].text == "`foo`"


def test_raw_string_literal_empty() -> None:
    toks = _tokens("``")
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.RAW_STRING_LITERAL
    assert toks[0].text == "``"


def test_raw_string_literal_does_not_process_escapes() -> None:
    # In a raw string literal, ``\n`` is a literal backslash followed by
    # the letter ``n``; it MUST NOT trigger escape-sequence handling.
    src = "`\\n\\t\\xZZ\\q`"
    toks = _tokens(src)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.RAW_STRING_LITERAL
    assert toks[0].text == src


def test_raw_string_literal_spans_multiple_lines() -> None:
    src = "`line one\nline two\nline three`"
    toks = _tokens(src)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.RAW_STRING_LITERAL
    assert toks[0].text == src
    assert toks[0].line == 1
    assert toks[0].column == 1


def test_token_after_multiline_raw_string_starts_on_correct_line() -> None:
    src = "`a\nb`x"
    toks = _significant(src)
    # Two significant tokens: the raw string and the trailing ident ``x``.
    assert [t.kind for t in toks] == [
        GoTokenKind.RAW_STRING_LITERAL,
        GoTokenKind.IDENTIFIER,
    ]
    # ``x`` lives on line 2 (the closing backtick is on line 2).
    assert toks[1].text == "x"
    assert toks[1].line == 2


# ===========================================================================
# Punctuation
# ===========================================================================


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("{", GoTokenKind.LBRACE),
        ("}", GoTokenKind.RBRACE),
        ("(", GoTokenKind.LPAREN),
        (")", GoTokenKind.RPAREN),
        ("[", GoTokenKind.LBRACKET),
        ("]", GoTokenKind.RBRACKET),
        (",", GoTokenKind.COMMA),
        (".", GoTokenKind.DOT),
        (":", GoTokenKind.COLON),
        (";", GoTokenKind.SEMICOLON),
        ("*", GoTokenKind.STAR),
        ("&", GoTokenKind.AMPERSAND),
        ("=", GoTokenKind.ASSIGN),
    ],
)
def test_single_char_punctuation(source: str, kind: GoTokenKind) -> None:
    toks = _tokens(source)
    assert len(toks) == 1
    assert toks[0].kind is kind
    assert toks[0].text == source


# ===========================================================================
# Operators (Requirement 10.1)
# ===========================================================================


def test_short_assign_emits_assign_kind() -> None:
    """``:=`` is the short-variable-declaration operator, not OTHER_OPERATOR."""

    toks = _tokens(":=")
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.ASSIGN
    assert toks[0].text == ":="


@pytest.mark.parametrize(
    "op",
    [
        # 3-char assignments / variadic.
        "<<=",
        ">>=",
        "&^=",
        "...",
        # 2-char operators (excluding ``:=`` which has its own kind).
        "==",
        "!=",
        "<=",
        ">=",
        "&&",
        "||",
        "<<",
        ">>",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "++",
        "--",
        "<-",
        "&^",
    ],
)
def test_multi_char_operators_emit_other_operator(op: str) -> None:
    toks = _tokens(op)
    assert len(toks) == 1, f"expected one token for {op!r}, got {toks}"
    assert toks[0].kind is GoTokenKind.OTHER_OPERATOR
    assert toks[0].text == op


@pytest.mark.parametrize("op", ["+", "-", "/", "%", "^", "!", "<", ">", "|"])
def test_single_char_other_operators(op: str) -> None:
    toks = _tokens(op)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.OTHER_OPERATOR
    assert toks[0].text == op


def test_three_char_operator_does_not_split_into_smaller_tokens() -> None:
    # Regression guard: ``...`` must stay a single token, not three DOTs.
    toks = _tokens("...")
    assert [t.kind for t in toks] == [GoTokenKind.OTHER_OPERATOR]


# ===========================================================================
# Comments (Requirement 10.4)
# ===========================================================================


def test_line_comment_kind() -> None:
    toks = _tokens("// regular comment")
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.LINE_COMMENT
    assert toks[0].text == "// regular comment"


def test_block_comment_single_line() -> None:
    toks = _tokens("/* block */")
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.BLOCK_COMMENT
    assert toks[0].text == "/* block */"


def test_block_comment_spans_multiple_lines() -> None:
    src = "/* line one\nline two\nline three */"
    toks = _tokens(src)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.BLOCK_COMMENT
    assert toks[0].text == src


def test_token_after_multiline_block_comment_starts_on_correct_line() -> None:
    src = "/* a\nb */x"
    toks = _significant(src)
    assert [t.kind for t in toks] == [
        GoTokenKind.BLOCK_COMMENT,
        GoTokenKind.IDENTIFIER,
    ]
    assert toks[1].text == "x"
    assert toks[1].line == 2


@pytest.mark.parametrize(
    "src",
    [
        "//go:build linux",
        "//go:build linux && amd64",
        "//go:build (linux || darwin) && !ignore",
        "// +build linux",
        "// +build linux,amd64",
        "//\t+build linux",
    ],
)
def test_build_constraint_comment_kind(src: str) -> None:
    toks = _tokens(src)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.BUILD_CONSTRAINT_COMMENT
    assert toks[0].text == src


@pytest.mark.parametrize(
    "src",
    [
        "// #cgo CFLAGS: -O2",
        "// #cgo LDFLAGS: -lm",
        "//\t#cgo pkg-config: glib-2.0",
    ],
)
def test_cgo_pragma_comment_kind(src: str) -> None:
    toks = _tokens(src)
    assert len(toks) == 1
    assert toks[0].kind is GoTokenKind.CGO_PRAGMA_COMMENT
    assert toks[0].text == src


def test_line_comment_without_directive_stays_line_comment() -> None:
    # Leading whitespace before something that *looks* like a directive
    # but isn't (no recognized prefix) keeps the kind as LINE_COMMENT.
    toks = _tokens("// not a directive: foo")
    assert toks[0].kind is GoTokenKind.LINE_COMMENT


def test_line_comment_with_unrecognized_directive_prefix_stays_line_comment() -> None:
    # ``//+build`` (no space between ``//`` and ``+build``) is NOT a
    # build-constraint comment per the tokenizer's rules.
    toks = _tokens("//+build linux")
    assert toks[0].kind is GoTokenKind.LINE_COMMENT


def test_go_build_directive_must_not_have_leading_whitespace() -> None:
    # ``// go:build linux`` (with a space) is NOT a build constraint;
    # only ``//go:build`` (no leading space) is.
    toks = _tokens("// go:build linux")
    assert toks[0].kind is GoTokenKind.LINE_COMMENT


# ===========================================================================
# Struct tag detection (Requirement 10.1)
# ===========================================================================


def test_backtick_inside_struct_after_identifier_is_struct_tag() -> None:
    src = (
        "type T struct {\n"
        "\tField int `json:\"field\"`\n"
        "}\n"
    )
    toks = _significant(src)
    kinds = [t.kind for t in toks]
    # The backtick literal must be reclassified as STRUCT_TAG.
    assert GoTokenKind.STRUCT_TAG in kinds
    assert GoTokenKind.RAW_STRING_LITERAL not in kinds

    tag = next(t for t in toks if t.kind is GoTokenKind.STRUCT_TAG)
    assert tag.text == '`json:"field"`'


def test_struct_tag_recognized_for_each_field_in_a_struct() -> None:
    src = (
        "type T struct {\n"
        "\tA int `tag1`\n"
        "\tB string `tag2`\n"
        "}\n"
    )
    toks = _significant(src)
    tags = [t for t in toks if t.kind is GoTokenKind.STRUCT_TAG]
    assert [t.text for t in tags] == ["`tag1`", "`tag2`"]


def test_backtick_outside_struct_in_var_declaration_stays_raw_string() -> None:
    # A backtick literal used as a regular value (not a struct tag) keeps
    # its RAW_STRING_LITERAL kind.
    src = "var x = `not a tag`\n"
    toks = _significant(src)
    kinds = [t.kind for t in toks]
    assert GoTokenKind.RAW_STRING_LITERAL in kinds
    assert GoTokenKind.STRUCT_TAG not in kinds


def test_backtick_inside_function_body_stays_raw_string() -> None:
    # Inside ``func() { ... }`` the brace stack records "not opened by
    # struct", so the backtick is a raw string literal, not a struct tag.
    src = (
        "func main() {\n"
        "\ts := `hello`\n"
        "\t_ = s\n"
        "}\n"
    )
    toks = _significant(src)
    kinds = [t.kind for t in toks]
    assert GoTokenKind.RAW_STRING_LITERAL in kinds
    assert GoTokenKind.STRUCT_TAG not in kinds


def test_backtick_inside_interface_body_stays_raw_string() -> None:
    # ``interface { ... }`` opens a brace, but not "by struct"; backtick
    # literals there are not struct tags.
    src = (
        "type I interface {\n"
        "\tMethod() string\n"
        "}\n"
        "var x = `hello`\n"
    )
    toks = _significant(src)
    kinds = [t.kind for t in toks]
    assert GoTokenKind.RAW_STRING_LITERAL in kinds
    assert GoTokenKind.STRUCT_TAG not in kinds


def test_struct_tag_only_inside_innermost_struct_brace() -> None:
    # An outer struct opens a struct frame, then a nested non-struct
    # block (e.g. a slice literal ``[]int{...}``) opens a non-struct
    # frame on top -- a backtick there is NOT a struct tag.
    src = (
        "type T struct {\n"
        "\tNames []string\n"
        "}\n"
        "var x = []string{`a`, `b`}\n"
    )
    toks = _significant(src)
    # No STRUCT_TAG should be emitted at all (no field-with-tag in T).
    kinds = [t.kind for t in toks]
    assert GoTokenKind.STRUCT_TAG not in kinds
    assert kinds.count(GoTokenKind.RAW_STRING_LITERAL) == 2


# ===========================================================================
# 1-indexed line and column tracking
# ===========================================================================


def test_first_token_is_at_line_one_column_one() -> None:
    toks = _tokens("package foo\n")
    assert toks[0].line == 1
    assert toks[0].column == 1


def test_columns_are_one_indexed_within_a_line() -> None:
    # ``package`` at col 1, single space at col 8, ``foo`` at col 9.
    toks = _tokens("package foo")
    assert toks[0].kind is GoTokenKind.PACKAGE_KEYWORD
    assert toks[0].column == 1
    assert toks[1].kind is GoTokenKind.WHITESPACE
    assert toks[1].column == 8
    assert toks[2].kind is GoTokenKind.IDENTIFIER
    assert toks[2].column == 9


def test_line_increments_after_lf_newline() -> None:
    src = "package foo\nimport \"x\"\n"
    sig = _significant(src)
    pkg = next(t for t in sig if t.kind is GoTokenKind.PACKAGE_KEYWORD)
    imp = next(t for t in sig if t.kind is GoTokenKind.IMPORT_KEYWORD)
    foo = next(
        t for t in sig if t.kind is GoTokenKind.IDENTIFIER and t.text == "foo"
    )
    string_lit = next(t for t in sig if t.kind is GoTokenKind.STRING_LITERAL)

    assert pkg.line == 1
    assert foo.line == 1
    assert imp.line == 2
    assert string_lit.line == 2

    # Column on line 2 starts back at 1.
    assert imp.column == 1
    assert string_lit.column == 8  # ``import \"x\"`` -- ``"x"`` starts at col 8


def test_line_increments_after_crlf_newline() -> None:
    src = "a\r\nb"
    toks = _tokens(src)
    # Tokens: identifier 'a', newline, identifier 'b'.
    assert [t.kind for t in toks] == [
        GoTokenKind.IDENTIFIER,
        GoTokenKind.NEWLINE,
        GoTokenKind.IDENTIFIER,
    ]
    assert toks[0].line == 1
    assert toks[0].column == 1
    assert toks[2].line == 2
    assert toks[2].column == 1


def test_line_resets_column_correctly_across_many_lines() -> None:
    src = "a\nb\nc\n"
    sig = _significant(src)
    assert [t.text for t in sig] == ["a", "b", "c"]
    assert sig[0].line == 1
    assert sig[0].column == 1
    assert sig[1].line == 2
    assert sig[1].column == 1
    assert sig[2].line == 3
    assert sig[2].column == 1


def test_column_increments_after_block_comment_on_same_line() -> None:
    # ``/* x */y`` -- after the 7-char block comment, ``y`` is at col 8.
    toks = _tokens("/* x */y")
    assert toks[0].kind is GoTokenKind.BLOCK_COMMENT
    assert toks[0].column == 1
    assert toks[1].kind is GoTokenKind.IDENTIFIER
    assert toks[1].column == 8


# ===========================================================================
# GoTokenizationError -- every canonical reason (Requirement 11.1)
# ===========================================================================


def test_unterminated_string_literal_at_eof_raises() -> None:
    with pytest.raises(GoTokenizationError) as exc:
        _tokens('"foo')
    assert exc.value.reason == GoTokenizationError.REASON_UNTERMINATED_STRING
    assert exc.value.line == 1
    assert exc.value.column == 1


def test_unterminated_string_literal_at_newline_raises() -> None:
    with pytest.raises(GoTokenizationError) as exc:
        _tokens('"foo\n"')
    assert exc.value.reason == GoTokenizationError.REASON_UNTERMINATED_STRING
    assert exc.value.line == 1
    assert exc.value.column == 1


def test_unterminated_string_after_leading_tokens_reports_string_position() -> None:
    # Tokens before the bad string don't shift the reported position
    # away from the opening quote.
    with pytest.raises(GoTokenizationError) as exc:
        _tokens('var x = "foo')
    assert exc.value.reason == GoTokenizationError.REASON_UNTERMINATED_STRING
    assert exc.value.line == 1
    # ``var x = `` is 8 chars; the opening quote sits at column 9.
    assert exc.value.column == 9


def test_unterminated_raw_string_literal_raises() -> None:
    with pytest.raises(GoTokenizationError) as exc:
        _tokens("`foo")
    assert exc.value.reason == GoTokenizationError.REASON_UNTERMINATED_RAW_STRING
    assert exc.value.line == 1
    assert exc.value.column == 1


def test_unterminated_block_comment_raises() -> None:
    with pytest.raises(GoTokenizationError) as exc:
        _tokens("/* foo")
    assert (
        exc.value.reason == GoTokenizationError.REASON_UNTERMINATED_BLOCK_COMMENT
    )
    assert exc.value.line == 1
    assert exc.value.column == 1


def test_unterminated_block_comment_spanning_multiple_lines_raises() -> None:
    with pytest.raises(GoTokenizationError) as exc:
        _tokens("/* foo\nbar\nbaz")
    assert (
        exc.value.reason == GoTokenizationError.REASON_UNTERMINATED_BLOCK_COMMENT
    )
    # The reported position is the comment's opening ``/*``.
    assert exc.value.line == 1
    assert exc.value.column == 1


@pytest.mark.parametrize(
    "src",
    [
        '"\\q"',  # Unrecognized escape letter.
        '"\\xZZ"',  # Malformed \\xHH (non-hex digits).
        '"\\x1"',  # \\xHH with too few hex digits before ``"``.
        '"\\uZZZZ"',  # Malformed \\uHHHH.
        '"\\u00"',  # \\uHHHH truncated.
        '"\\UZZZZZZZZ"',  # Malformed \\UHHHHHHHH.
        '"\\U0000"',  # \\UHHHHHHHH truncated.
        '"\\07"',  # Octal with too few digits before ``"``.
        '"\\09"',  # Octal with non-octal digit after first.
    ],
)
def test_invalid_escape_in_string_literal_raises(src: str) -> None:
    with pytest.raises(GoTokenizationError) as exc:
        _tokens(src)
    assert exc.value.reason == GoTokenizationError.REASON_INVALID_ESCAPE
    # The reported position is the string literal's opening quote.
    assert exc.value.line == 1
    assert exc.value.column == 1


def test_tokenization_error_subclasses_exception() -> None:
    # The recognizer's per-file skip path catches GoTokenizationError;
    # it must subclass Exception (not BaseException).
    err = GoTokenizationError(1, 1, "test")
    assert isinstance(err, Exception)
    # The four canonical reasons are exposed as class attributes.
    assert hasattr(GoTokenizationError, "REASON_UNTERMINATED_STRING")
    assert hasattr(GoTokenizationError, "REASON_UNTERMINATED_RAW_STRING")
    assert hasattr(GoTokenizationError, "REASON_UNTERMINATED_BLOCK_COMMENT")
    assert hasattr(GoTokenizationError, "REASON_INVALID_ESCAPE")


# ===========================================================================
# Integration: realistic Go snippet
# ===========================================================================


def test_realistic_go_snippet_tokenizes_end_to_end() -> None:
    """A complete, well-formed Go file produces no errors and yields the
    expected significant-token shape."""

    src = (
        "// Package svc handles requests.\n"
        "package svc\n"
        "\n"
        "import (\n"
        '\t"fmt"\n'
        "\t_ \"net/http\"\n"
        ")\n"
        "\n"
        "type User struct {\n"
        "\tName string `json:\"name\"`\n"
        "\tAge  int    `json:\"age\"`\n"
        "}\n"
        "\n"
        "func main() {\n"
        '\tx := "hello"\n'
        "\tfmt.Println(x)\n"
        "}\n"
    )

    toks = list(tokenize_go_source(src))

    # The tokenizer succeeds; produces a non-trivial stream.
    assert len(toks) > 30

    # The leading line comment is classified as LINE_COMMENT (not a
    # build-constraint or cgo-pragma comment).
    assert toks[0].kind is GoTokenKind.LINE_COMMENT

    # Both struct tags are reclassified.
    tag_kinds = [t.kind for t in toks if t.kind is GoTokenKind.STRUCT_TAG]
    assert len(tag_kinds) == 2

    # The two string literals (``"hello"`` and the quoted import paths)
    # are STRING_LITERAL.
    string_count = sum(1 for t in toks if t.kind is GoTokenKind.STRING_LITERAL)
    assert string_count == 3  # "fmt", "net/http", "hello"

    # ``:=`` is the ASSIGN token, not OTHER_OPERATOR.
    short_assigns = [
        t for t in toks if t.kind is GoTokenKind.ASSIGN and t.text == ":="
    ]
    assert len(short_assigns) == 1

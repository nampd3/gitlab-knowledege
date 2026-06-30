"""In-process tokenizer for Go source files.

Exposes :func:`tokenize_go_source`, a pure function that consumes a Go
source file as a ``str`` and yields :class:`GoToken` instances in source
order. The tokenizer never invokes a Go toolchain, has no global state,
and produces identical output for identical input (Requirements 10.1,
10.2, 11.4).

Design references:

* The full token-kind enumeration lives in design §4. Comments,
  newlines, and whitespace are emitted as first-class tokens (rather
  than stripped) so the construct recognizer can reconstruct package
  doc comments, detect build constraints, and handle Go's automatic
  semicolon insertion at line boundaries.
* Six Go keywords carry dedicated kinds because the recognizer
  branches on them: ``package``, ``import``, ``func``, ``type``,
  ``struct``, ``interface``. Every other Go keyword (``var``, ``const``,
  ``if``, ``for``, ``return``, ``range``, ``defer``, ``go``, ``select``,
  ``case``, ``default``, ``else``, ``switch``, ``break``, ``continue``,
  ``fallthrough``, ``goto``, ``chan``, ``map``, ``nil``, ``true``,
  ``false``) is emitted with kind :attr:`GoTokenKind.IDENTIFIER` because
  the recognizer does not need to distinguish them from ordinary
  identifiers.
* Backtick-quoted runs immediately following a struct field declaration
  are reclassified from :attr:`GoTokenKind.RAW_STRING_LITERAL` to
  :attr:`GoTokenKind.STRUCT_TAG`. The heuristic is: when a backtick
  literal appears inside a balanced brace pair opened by ``struct {``
  *and* the previous non-trivia token is an identifier (i.e. the
  field's type name), the literal is a struct tag.
* Build-constraint comments (``//go:build`` and ``// +build``) and cgo
  pragma comments (``// #cgo``) are recognized at tokenizer level so
  the recognizer's leading-comment scan can detect whole-file skips
  without re-classifying line comments (Requirement 10.4).

Error handling. The tokenizer raises :class:`GoTokenizationError` on:

* unterminated string literal,
* unterminated raw string literal,
* unterminated block comment,
* invalid escape sequence inside a string literal.

The reason string carried on each :class:`GoTokenizationError` is drawn
from the canonical enumeration in design §3 "Error handling" and is
exposed as a class attribute (``GoTokenizationError.REASON_*``) so the
recognizer's per-file skip path (task 4.2) can pattern-match on the
reason without string-literal duplication.

Implements Requirements 10.1, 10.2, 10.4, 11.1, 11.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.project_analyzer.go._events import GoToken, GoTokenKind

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "GoTokenizationError",
    "tokenize_go_source",
]


class GoTokenizationError(Exception):
    """Raised when Go source cannot be tokenized to completion.

    The tokenizer is contract-bound to consume any Go source file end to
    end and yield tokens, except for the four malformed-input shapes
    enumerated in design §3 "Error handling". Each malformed shape is
    reported through this exception with a canonical, stable ``reason``
    string so the recognizer's per-file skip path
    (``SkipFileEvent("tokenization failed: <reason>", line)``) can be
    constructed without string duplication.

    The four canonical reasons are exposed as class attributes:

    * :attr:`REASON_UNTERMINATED_STRING` — a regular ``"..."`` literal
      that runs into an unescaped newline or end of file.
    * :attr:`REASON_UNTERMINATED_RAW_STRING` — a backtick-delimited raw
      literal that runs to end of file without a closing backtick.
    * :attr:`REASON_UNTERMINATED_BLOCK_COMMENT` — a ``/* ... */`` block
      comment that runs to end of file without ``*/``.
    * :attr:`REASON_INVALID_ESCAPE` — a backslash escape inside a regular
      string literal that is not one of Go's permitted forms (``\\a``,
      ``\\b``, ``\\f``, ``\\n``, ``\\r``, ``\\t``, ``\\v``, ``\\\\``,
      ``\\'``, ``\\"``, ``\\xHH``, ``\\uHHHH``, ``\\UHHHHHHHH``, or a
      three-digit ``\\NNN`` octal in the range ``0-7``).

    Instances are catchable at the recognizer boundary
    (``go_parser.parse_repo``); they never propagate to sub-analyzers.
    A single bad file becomes a single :class:`SkipFileEvent`, leaving
    the rest of the repository unaffected (Requirement 11.1).

    Attributes:
        line: 1-indexed line number where the offending construct began.
            For string and raw-string literals this is the line of the
            opening quote; for block comments the line of the opening
            ``/*``; for invalid escapes the line of the enclosing
            string literal's opening quote (so the recognizer can record
            a single source location for the entire malformed token).
        column: 1-indexed rune column where the offending construct
            began, paired with ``line``.
        reason: One of the canonical strings above.
    """

    #: Reason for an unterminated regular string literal (``"foo``).
    REASON_UNTERMINATED_STRING: Final[str] = "unterminated string literal"

    #: Reason for an unterminated raw string literal (`` `foo ``).
    REASON_UNTERMINATED_RAW_STRING: Final[str] = "unterminated raw string literal"

    #: Reason for an unterminated block comment (``/* foo``).
    REASON_UNTERMINATED_BLOCK_COMMENT: Final[str] = "unterminated block comment"

    #: Reason for an invalid backslash escape inside a string literal.
    REASON_INVALID_ESCAPE: Final[str] = "invalid escape sequence"

    def __init__(self, line: int, column: int, reason: str) -> None:
        super().__init__(f"go tokenization failed at {line}:{column}: {reason}")
        self.line = line
        self.column = column
        self.reason = reason


# --- Internal lookup tables ---------------------------------------------------

#: Source-text spellings of the six keywords that carry dedicated kinds.
#: Every other Go keyword is emitted as :attr:`GoTokenKind.IDENTIFIER`
#: because the recognizer does not branch on it.
_KEYWORD_KINDS: Final[dict[str, GoTokenKind]] = {
    "package": GoTokenKind.PACKAGE_KEYWORD,
    "import": GoTokenKind.IMPORT_KEYWORD,
    "func": GoTokenKind.FUNC_KEYWORD,
    "type": GoTokenKind.TYPE_KEYWORD,
    "struct": GoTokenKind.STRUCT_KEYWORD,
    "interface": GoTokenKind.INTERFACE_KEYWORD,
}

#: Single-character punctuation/operator atoms with dedicated kinds. Any
#: character not in this map (and not part of a multi-character operator
#: handled separately) is emitted as :attr:`GoTokenKind.OTHER_OPERATOR`.
_SINGLE_CHAR_KIND: Final[dict[str, GoTokenKind]] = {
    "{": GoTokenKind.LBRACE,
    "}": GoTokenKind.RBRACE,
    "(": GoTokenKind.LPAREN,
    ")": GoTokenKind.RPAREN,
    "[": GoTokenKind.LBRACKET,
    "]": GoTokenKind.RBRACKET,
    ",": GoTokenKind.COMMA,
    ".": GoTokenKind.DOT,
    ":": GoTokenKind.COLON,
    ";": GoTokenKind.SEMICOLON,
    "*": GoTokenKind.STAR,
    "&": GoTokenKind.AMPERSAND,
    "=": GoTokenKind.ASSIGN,
}

#: Three-character operator atoms; checked before any 2- or 1-char dispatch
#: so e.g. ``...`` is not split into three ``.`` tokens.
_THREE_CHAR_OPS: Final[frozenset[str]] = frozenset({"<<=", ">>=", "&^=", "..."})

#: Two-character operator atoms (excluding ``:=``, which has its own kind).
_TWO_CHAR_OPS: Final[frozenset[str]] = frozenset(
    {
        "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
        "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
        "++", "--", "<-", "&^",
    }
)

#: Single backslash-escape characters Go's spec accepts inside regular
#: string literals (excluding the numeric escapes ``\xHH``, ``\uHHHH``,
#: ``\UHHHHHHHH``, and ``\NNN`` octal which are handled separately).
_SIMPLE_ESCAPES: Final[frozenset[str]] = frozenset("abfnrtv\\'\"")

#: Token kinds that do not count as "previous significant token" for
#: struct-tag detection or for the brace stack's "was opened by struct"
#: bookkeeping.
_TRIVIA_KINDS: Final[frozenset[GoTokenKind]] = frozenset(
    {
        GoTokenKind.WHITESPACE,
        GoTokenKind.NEWLINE,
        GoTokenKind.LINE_COMMENT,
        GoTokenKind.BLOCK_COMMENT,
        GoTokenKind.BUILD_CONSTRAINT_COMMENT,
        GoTokenKind.CGO_PRAGMA_COMMENT,
    }
)


# --- Public entry point -------------------------------------------------------


def tokenize_go_source(text: str) -> Iterator[GoToken]:
    """Yield :class:`GoToken` instances for ``text`` in source order.

    Args:
        text: A Go source file's contents, already UTF-8 decoded by the
            caller.

    Yields:
        :class:`GoToken` records carrying ``kind``, source ``text``,
        1-indexed ``line``, and 1-indexed rune ``column``.

    Raises:
        GoTokenizationError: When ``text`` contains an unterminated
            string literal, raw string literal, or block comment, or
            an invalid escape sequence inside a string literal.

    Example:
        >>> [t.kind.value for t in tokenize_go_source("package foo\\n")]
        ['package', 'whitespace', 'identifier', 'newline']
    """

    return _GoTokenizer(text).tokenize()


# --- Internal implementation --------------------------------------------------


class _GoTokenizer:
    """Stateful scanner backing :func:`tokenize_go_source`.

    The scanner is encapsulated in an object purely for ergonomic
    bookkeeping (the position, line, column, brace stack, and last
    significant token). The public entry point still satisfies "pure
    function with no side effects" because every instance is local to
    a single call and never mutates module-level state.
    """

    __slots__ = (
        "_col",
        "_line",
        "_n",
        "_pos",
        "_prev_significant",
        "_struct_brace_stack",
        "_text",
    )

    def __init__(self, text: str) -> None:
        self._text: str = text
        self._n: int = len(text)
        self._pos: int = 0
        self._line: int = 1
        self._col: int = 1
        # One bool per currently open brace; True iff that brace was
        # opened immediately after the ``struct`` keyword (modulo
        # trivia). Used to gate STRUCT_TAG reclassification.
        self._struct_brace_stack: list[bool] = []
        # The most recently emitted non-trivia token, or None when no
        # such token has been produced yet. Used both for STRUCT_TAG
        # detection and for "was the brace preceded by struct?" tracking.
        self._prev_significant: GoToken | None = None

    def tokenize(self) -> Iterator[GoToken]:
        """Yield every token in ``self._text`` in source order."""

        while self._pos < self._n:
            tok = self._scan_one()
            self._maintain_brace_stack(tok)
            yield tok
            if tok.kind not in _TRIVIA_KINDS:
                self._prev_significant = tok

    # --- Brace stack ---------------------------------------------------

    def _maintain_brace_stack(self, tok: GoToken) -> None:
        """Push/pop the struct-frame stack as braces enter or leave scope."""

        if tok.kind is GoTokenKind.LBRACE:
            opened_by_struct = (
                self._prev_significant is not None
                and self._prev_significant.kind is GoTokenKind.STRUCT_KEYWORD
            )
            self._struct_brace_stack.append(opened_by_struct)
        elif tok.kind is GoTokenKind.RBRACE and self._struct_brace_stack:
            self._struct_brace_stack.pop()

    def _in_struct_frame(self) -> bool:
        """Return True when the innermost open brace was opened by ``struct``."""

        return bool(self._struct_brace_stack) and self._struct_brace_stack[-1]

    # --- Top-level dispatch -------------------------------------------

    def _scan_one(self) -> GoToken:  # noqa: PLR0911 - dispatch table over token kinds
        """Scan and return exactly one token starting at ``self._pos``."""

        c = self._text[self._pos]

        if c == "\n":
            return self._scan_newline_lf()
        if c == "\r":
            if self._pos + 1 < self._n and self._text[self._pos + 1] == "\n":
                return self._scan_newline_crlf()
            # Bare CR is treated as whitespace (uncommon in modern Go
            # source but defensible).
            return self._scan_whitespace()
        if c in " \t":
            return self._scan_whitespace()
        if c == "/" and self._pos + 1 < self._n:
            nxt = self._text[self._pos + 1]
            if nxt == "/":
                return self._scan_line_comment()
            if nxt == "*":
                return self._scan_block_comment()
        if c == '"':
            return self._scan_string_literal()
        if c == "`":
            return self._scan_raw_string_literal()
        if c.isdigit():
            return self._scan_number()
        if c == "." and self._pos + 1 < self._n and self._text[self._pos + 1].isdigit():
            # A leading-dot float such as ``.5``.
            return self._scan_number()
        if c == "_" or c.isalpha():
            return self._scan_identifier_or_keyword()
        return self._scan_operator()

    # --- Newlines and whitespace --------------------------------------

    def _scan_newline_lf(self) -> GoToken:
        line = self._line
        col = self._col
        self._pos += 1
        self._line += 1
        self._col = 1
        return GoToken(GoTokenKind.NEWLINE, "\n", line, col)

    def _scan_newline_crlf(self) -> GoToken:
        line = self._line
        col = self._col
        self._pos += 2
        self._line += 1
        self._col = 1
        return GoToken(GoTokenKind.NEWLINE, "\r\n", line, col)

    def _scan_whitespace(self) -> GoToken:
        line = self._line
        col = self._col
        start = self._pos
        while self._pos < self._n:
            c = self._text[self._pos]
            if c in " \t":
                self._pos += 1
                self._col += 1
            elif c == "\r":
                # Bare CR, not part of a CRLF (the CRLF case is handled
                # by the top-level dispatch and never enters this loop
                # via a CR followed by an LF, since we'd have returned
                # before invoking _scan_whitespace).
                if self._pos + 1 < self._n and self._text[self._pos + 1] == "\n":
                    break
                self._pos += 1
                self._col += 1
            else:
                break
        return GoToken(GoTokenKind.WHITESPACE, self._text[start : self._pos], line, col)

    # --- Comments ------------------------------------------------------

    def _scan_line_comment(self) -> GoToken:
        line = self._line
        col = self._col
        start = self._pos
        # Consume the leading "//".
        self._pos += 2
        self._col += 2
        while self._pos < self._n and self._text[self._pos] != "\n":
            self._pos += 1
            self._col += 1
        text = self._text[start : self._pos]
        kind = _classify_line_comment(text)
        return GoToken(kind, text, line, col)

    def _scan_block_comment(self) -> GoToken:
        line = self._line
        col = self._col
        start = self._pos
        # Consume the leading "/*".
        self._pos += 2
        self._col += 2
        while self._pos < self._n:
            c = self._text[self._pos]
            if c == "*" and self._pos + 1 < self._n and self._text[self._pos + 1] == "/":
                self._pos += 2
                self._col += 2
                return GoToken(
                    GoTokenKind.BLOCK_COMMENT,
                    self._text[start : self._pos],
                    line,
                    col,
                )
            if c == "\n":
                self._pos += 1
                self._line += 1
                self._col = 1
            else:
                self._pos += 1
                self._col += 1
        raise GoTokenizationError(
            line, col, GoTokenizationError.REASON_UNTERMINATED_BLOCK_COMMENT
        )

    # --- String literals ----------------------------------------------

    def _scan_string_literal(self) -> GoToken:
        line = self._line
        col = self._col
        start = self._pos
        self._pos += 1  # skip opening "
        self._col += 1
        while self._pos < self._n:
            c = self._text[self._pos]
            if c == '"':
                self._pos += 1
                self._col += 1
                return GoToken(
                    GoTokenKind.STRING_LITERAL,
                    self._text[start : self._pos],
                    line,
                    col,
                )
            if c == "\n":
                # Regular string literals must not span newlines.
                raise GoTokenizationError(
                    line, col, GoTokenizationError.REASON_UNTERMINATED_STRING
                )
            if c == "\\":
                self._consume_escape_sequence(line, col)
            else:
                self._pos += 1
                self._col += 1
        raise GoTokenizationError(
            line, col, GoTokenizationError.REASON_UNTERMINATED_STRING
        )

    def _consume_escape_sequence(self, str_line: int, str_col: int) -> None:
        """Consume a backslash-escape sequence inside a regular string.

        Called with ``self._pos`` pointing at the leading backslash.
        Advances past the entire escape on success; raises
        :class:`GoTokenizationError` for unrecognized escapes (the
        ``str_line``/``str_col`` are reported as the failing position
        because the recognizer reports the string's start, matching
        design §3 "Error handling").
        """

        # Skip the backslash.
        self._pos += 1
        self._col += 1
        if self._pos >= self._n:
            # Backslash at EOF -- the surrounding string is unterminated.
            raise GoTokenizationError(
                str_line, str_col, GoTokenizationError.REASON_UNTERMINATED_STRING
            )

        c = self._text[self._pos]
        if c in _SIMPLE_ESCAPES:
            self._pos += 1
            self._col += 1
            return
        if c == "x":
            self._pos += 1
            self._col += 1
            self._consume_hex_digits(2, str_line, str_col)
            return
        if c == "u":
            self._pos += 1
            self._col += 1
            self._consume_hex_digits(4, str_line, str_col)
            return
        if c == "U":
            self._pos += 1
            self._col += 1
            self._consume_hex_digits(8, str_line, str_col)
            return
        if "0" <= c <= "7":
            # Three-digit octal escape (Go requires exactly three digits).
            for _ in range(3):
                if self._pos >= self._n or not ("0" <= self._text[self._pos] <= "7"):
                    raise GoTokenizationError(
                        str_line, str_col, GoTokenizationError.REASON_INVALID_ESCAPE
                    )
                self._pos += 1
                self._col += 1
            return
        raise GoTokenizationError(
            str_line, str_col, GoTokenizationError.REASON_INVALID_ESCAPE
        )

    def _consume_hex_digits(self, count: int, str_line: int, str_col: int) -> None:
        """Consume exactly ``count`` hex digits or raise."""

        for _ in range(count):
            if self._pos >= self._n or not _is_hex_digit(self._text[self._pos]):
                raise GoTokenizationError(
                    str_line, str_col, GoTokenizationError.REASON_INVALID_ESCAPE
                )
            self._pos += 1
            self._col += 1

    def _scan_raw_string_literal(self) -> GoToken:
        line = self._line
        col = self._col
        start = self._pos
        self._pos += 1  # skip opening backtick
        self._col += 1
        while self._pos < self._n:
            c = self._text[self._pos]
            if c == "`":
                self._pos += 1
                self._col += 1
                kind = self._classify_backtick_literal()
                return GoToken(kind, self._text[start : self._pos], line, col)
            if c == "\n":
                self._pos += 1
                self._line += 1
                self._col = 1
            else:
                self._pos += 1
                self._col += 1
        raise GoTokenizationError(
            line, col, GoTokenizationError.REASON_UNTERMINATED_RAW_STRING
        )

    def _classify_backtick_literal(self) -> GoTokenKind:
        """Return STRUCT_TAG when context warrants, else RAW_STRING_LITERAL.

        The heuristic (per design §"go.go_tokenizer" and the task
        statement): emit :attr:`GoTokenKind.STRUCT_TAG` when the
        innermost open brace was opened by ``struct`` *and* the previous
        non-trivia token is an identifier (i.e. the field's type name
        in a Go struct field declaration). Otherwise emit
        :attr:`GoTokenKind.RAW_STRING_LITERAL`.
        """

        if not self._in_struct_frame():
            return GoTokenKind.RAW_STRING_LITERAL
        if self._prev_significant is None:
            return GoTokenKind.RAW_STRING_LITERAL
        if self._prev_significant.kind is GoTokenKind.IDENTIFIER:
            return GoTokenKind.STRUCT_TAG
        return GoTokenKind.RAW_STRING_LITERAL

    # --- Numbers -------------------------------------------------------

    def _scan_number(self) -> GoToken:
        line = self._line
        col = self._col
        start = self._pos

        is_hex = False
        # Optional radix prefix: 0x, 0X, 0o, 0O, 0b, 0B.
        if (
            self._text[self._pos] == "0"
            and self._pos + 1 < self._n
            and self._text[self._pos + 1] in "xXoObB"
        ):
            is_hex = self._text[self._pos + 1] in "xX"
            self._pos += 2
            self._col += 2

        # Integer / mantissa-leading digits.
        self._consume_number_digits(is_hex)

        # Optional fractional part.
        if self._pos < self._n and self._text[self._pos] == ".":
            # Be conservative around the variadic operator ``...``: only
            # consume the dot when followed by another digit (or by a
            # non-dot character, in which case Go would treat ``1.`` as
            # a float literal). When followed by another dot we leave
            # the dot for the operator scanner.
            nxt_pos = self._pos + 1
            if nxt_pos < self._n:
                nxt = self._text[nxt_pos]
                if nxt.isdigit() or (is_hex and _is_hex_digit(nxt)):
                    self._pos += 1
                    self._col += 1
                    self._consume_number_digits(is_hex)
                elif nxt == ".":
                    # Likely the start of ``..`` or ``...``; leave it
                    # for the operator scanner.
                    pass
                else:
                    # Trailing dot float (``1.``).
                    self._pos += 1
                    self._col += 1
            else:
                # Trailing dot at end of file.
                self._pos += 1
                self._col += 1

        # Optional exponent: ``e``/``E`` for decimals, ``p``/``P`` for
        # hex floats.
        if self._pos < self._n:
            exp_char = self._text[self._pos]
            if (not is_hex and exp_char in "eE") or (is_hex and exp_char in "pP"):
                self._pos += 1
                self._col += 1
                if self._pos < self._n and self._text[self._pos] in "+-":
                    self._pos += 1
                    self._col += 1
                while self._pos < self._n and (
                    self._text[self._pos].isdigit() or self._text[self._pos] == "_"
                ):
                    self._pos += 1
                    self._col += 1

        # Optional imaginary suffix.
        if self._pos < self._n and self._text[self._pos] == "i":
            self._pos += 1
            self._col += 1

        return GoToken(
            GoTokenKind.NUMBER_LITERAL,
            self._text[start : self._pos],
            line,
            col,
        )

    def _consume_number_digits(self, is_hex: bool) -> None:
        """Consume a run of digits (and underscores; hex digits when applicable)."""

        while self._pos < self._n:
            c = self._text[self._pos]
            if c.isdigit() or c == "_" or (is_hex and _is_hex_digit(c)):
                self._pos += 1
                self._col += 1
            else:
                break

    # --- Identifiers ---------------------------------------------------

    def _scan_identifier_or_keyword(self) -> GoToken:
        line = self._line
        col = self._col
        start = self._pos
        while self._pos < self._n:
            c = self._text[self._pos]
            if c == "_" or c.isalpha() or c.isdigit():
                self._pos += 1
                self._col += 1
            else:
                break
        text = self._text[start : self._pos]
        kind = _KEYWORD_KINDS.get(text, GoTokenKind.IDENTIFIER)
        return GoToken(kind, text, line, col)

    # --- Operators -----------------------------------------------------

    def _scan_operator(self) -> GoToken:
        line = self._line
        col = self._col

        # 3-char operators first (so e.g. ``...`` is one token).
        if self._pos + 3 <= self._n:
            three = self._text[self._pos : self._pos + 3]
            if three in _THREE_CHAR_OPS:
                self._pos += 3
                self._col += 3
                return GoToken(GoTokenKind.OTHER_OPERATOR, three, line, col)

        # 2-char operators next; ``:=`` has its own kind, the rest are
        # opaque to the recognizer.
        if self._pos + 2 <= self._n:
            two = self._text[self._pos : self._pos + 2]
            if two == ":=":
                self._pos += 2
                self._col += 2
                return GoToken(GoTokenKind.ASSIGN, two, line, col)
            if two in _TWO_CHAR_OPS:
                self._pos += 2
                self._col += 2
                return GoToken(GoTokenKind.OTHER_OPERATOR, two, line, col)

        # Single-character dispatch.
        c = self._text[self._pos]
        self._pos += 1
        self._col += 1
        kind = _SINGLE_CHAR_KIND.get(c, GoTokenKind.OTHER_OPERATOR)
        return GoToken(kind, c, line, col)


# --- Module-level helpers -----------------------------------------------------


def _is_hex_digit(c: str) -> bool:
    """Return True iff ``c`` is a single ASCII hex digit."""

    return c.isdigit() or ("a" <= c <= "f") or ("A" <= c <= "F")


def _classify_line_comment(text: str) -> GoTokenKind:
    """Classify a line-comment token's kind from its source text.

    ``text`` includes the leading ``//`` and excludes the trailing
    newline. Recognized variants:

    * ``//go:build <expr>`` (no whitespace between ``//`` and
      ``go:build``) → :attr:`GoTokenKind.BUILD_CONSTRAINT_COMMENT`.
    * ``// +build <expr>`` (at least one space or tab between ``//``
      and ``+build``) → :attr:`GoTokenKind.BUILD_CONSTRAINT_COMMENT`.
    * ``// #cgo <expr>`` (at least one space or tab between ``//``
      and ``#cgo``) → :attr:`GoTokenKind.CGO_PRAGMA_COMMENT`.
    * Anything else → :attr:`GoTokenKind.LINE_COMMENT`.
    """

    body = text[2:]  # strip leading "//"

    # //go:build ... (no leading whitespace before "go:build")
    if body.startswith("go:build") and (
        len(body) == len("go:build") or body[len("go:build")] in (" ", "\t")
    ):
        return GoTokenKind.BUILD_CONSTRAINT_COMMENT

    # The remaining variants require at least one whitespace character
    # between "//" and the directive token.
    if not body or body[0] not in (" ", "\t"):
        return GoTokenKind.LINE_COMMENT

    stripped = body.lstrip(" \t")
    if stripped.startswith("+build") and (
        len(stripped) == len("+build") or stripped[len("+build")] in (" ", "\t")
    ):
        return GoTokenKind.BUILD_CONSTRAINT_COMMENT
    if stripped.startswith("#cgo") and (
        len(stripped) == len("#cgo") or stripped[len("#cgo")] in (" ", "\t")
    ):
        return GoTokenKind.CGO_PRAGMA_COMMENT
    return GoTokenKind.LINE_COMMENT

"""Construct recognizer for Go source files.

Consumes a stream of :class:`GoToken` instances produced by
:func:`tokenize_go_source` and yields :class:`GoEvent` instances in
source order.

Public API:

* :func:`recognize_constructs` — recognizer for a single Go source file.
* :func:`parse_go_mod` — recognizer for ``go.mod`` module declarations.
* :func:`parse_repo` — repo-level entry point that filters
  ``repository_contents`` through :func:`is_go_source_file`,
  tokenizes and recognizes each surviving file in sorted order, and
  exposes ``go.mod`` events under the ``"go.mod"`` key. The four Go
  sub-analyzers consume its result.

The recognizer is intentionally shallow: it does **not** implement a
full Go grammar. It recognizes only the construct kinds the four Go
sub-analyzers need to dispatch on (Requirements 1.1, 1.2):

* ``import "<path>"`` declarations, single-line and parenthesized blocks
  (:class:`ImportEvent`).
* The package doc-comment block immediately preceding the
  ``package <name>`` line, with no blank-line gap
  (:class:`PackageDocCommentEvent`).
* Function and method declarations (:class:`FuncDeclEvent`), with
  ``receiver_type`` extracted from method declarations.
* Method calls with a dotted receiver chain and positional arguments
  (:class:`MethodCallEvent`).
* Composite literals with named-field syntax in the four shapes
  ``T{...}``, ``&T{...}``, ``*T{...}``, and ``pkg.T{...}``
  (:class:`StructLitEvent`).

Two whole-file early-exits are detected before the construct walker
runs (Requirement 10.4):

* A non-trivial ``//go:build`` or ``// +build`` line in the leading
  comment block causes a single
  :class:`SkipFileEvent` ("build constraint requires toolchain") to be
  yielded as the file's only event.
* An unaliased ``import "C"`` declaration causes a single
  :class:`SkipFileEvent` ("cgo directive requires toolchain") to be
  yielded as the file's only event.

Trivial build-constraint expressions (empty, ``!ignore``, empty
``// +build``) are silently passed through; the recognizer does not
emit a :class:`BuildConstraintEvent` for them, matching the
sub-analyzers' contract that they are ignored.

Each top-level construct (import, type, function declaration) is
parsed inside its own ``try``/``except`` boundary. A construct that
fails to parse yields a :class:`SkipFileEvent` whose ``reason`` begins
``parse error in ...`` and the recognizer resumes at the next
package-level boundary (Requirement 11.1) — the rest of the file's
constructs continue to be recognized.

Implements Requirements 1.1, 1.2, 1.3, 10.4, 11.1, 11.4.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.project_analyzer.go._events import (
    CallArg,
    DottedArg,
    FuncDeclEvent,
    GoTokenKind,
    IdentArg,
    ImportEvent,
    MethodCallEvent,
    ModFileModuleEvent,
    NumberLitArg,
    PackageDocCommentEvent,
    SkipFileEvent,
    StringLitArg,
    StructLitArg,
    StructLitEvent,
    UnknownArg,
)
from project_knowledge_mcp.project_analyzer.go.go_filter import is_go_source_file
from project_knowledge_mcp.project_analyzer.go.go_tokenizer import (
    GoTokenizationError,
    tokenize_go_source,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from project_knowledge_mcp.models import RepositoryContents
    from project_knowledge_mcp.project_analyzer.go._events import (
        ArgRef,
        GoEvent,
        GoToken,
    )

__all__ = ["parse_go_mod", "parse_repo", "recognize_constructs"]


# --- Internal constants -------------------------------------------------------


class _ParseError(Exception):
    """Raised when a top-level construct cannot be parsed.

    Caught at the top-level dispatch boundary in :meth:`_Recognizer._walk`
    so a single bad construct yields one :class:`SkipFileEvent` and
    parsing resumes at the next package-level boundary, leaving the rest
    of the file's constructs intact (Requirement 11.1).
    """


#: Token kinds the construct recognizer treats as trivia. Building a
#: ``sig_indices`` array that excludes these lets the walker reason in
#: terms of significant tokens only; positions in the original token
#: stream are recovered through that index map when an event needs to
#: cite a line.
_TRIVIA_KINDS: Final[frozenset[GoTokenKind]] = frozenset(
    {
        GoTokenKind.WHITESPACE,
        GoTokenKind.NEWLINE,
        GoTokenKind.LINE_COMMENT,
        GoTokenKind.BLOCK_COMMENT,
        GoTokenKind.BUILD_CONSTRAINT_COMMENT,
        GoTokenKind.CGO_PRAGMA_COMMENT,
        GoTokenKind.STRUCT_TAG,
    }
)


#: Top-level keyword token kinds at which the recognizer dispatches to a
#: dedicated parser. ``var`` and ``const`` are not in this set because
#: the tokenizer emits them as identifiers (only six Go keywords carry
#: dedicated kinds); they are detected by text in
#: :meth:`_Recognizer._is_top_level_boundary`.
_TOP_LEVEL_KEYWORD_KINDS: Final[frozenset[GoTokenKind]] = frozenset(
    {
        GoTokenKind.PACKAGE_KEYWORD,
        GoTokenKind.IMPORT_KEYWORD,
        GoTokenKind.FUNC_KEYWORD,
        GoTokenKind.TYPE_KEYWORD,
    }
)


#: Identifier texts that begin a top-level declaration even though the
#: tokenizer emits them as :attr:`GoTokenKind.IDENTIFIER`.
_TOP_LEVEL_KEYWORD_IDENTS: Final[frozenset[str]] = frozenset({"var", "const"})


#: Number of dotted segments that map onto a ``pkg.Type`` composite
#: literal. A 1-segment chain is an unqualified ``Type`` literal; longer
#: chains are not valid type expressions in Go's grammar and produce a
#: no-match in :meth:`_Recognizer._split_chain_for_struct`.
_PKG_QUALIFIED_CHAIN_LEN: Final[int] = 2


#: Minimum length of a quoted string literal (``""`` or ```` `` ````).
#: Used by :func:`_string_literal_value` and
#: :func:`_raw_string_literal_value` to decide whether the surrounding
#: quotes can be stripped without going out of bounds.
_QUOTED_LITERAL_MIN_LEN: Final[int] = 2


#: Build-constraint expressions that are treated as trivial and do not
#: cause the file to be skipped. The empty string covers both the empty
#: ``//go:build`` line and the empty ``// +build`` line; ``!ignore`` is
#: the canonical "this file is intentionally excluded" form which the
#: design carve-out preserves as trivial because it does not require a
#: Go toolchain to evaluate.
_TRIVIAL_BUILD_CONSTRAINTS: Final[frozenset[str]] = frozenset({"", "!ignore"})


#: Pattern matching a ``module <module-path>`` line in a ``go.mod`` file.
#: Allows arbitrary leading whitespace, captures the module path as
#: group 1, and captures any trailing ``//``-comment body as group 2
#: (without the leading ``//``). Used by :func:`parse_go_mod`.
_RE_GO_MOD_MODULE: Final[re.Pattern[str]] = re.compile(
    r"^\s*module\s+(\S+)\s*(?://(.*))?$",
)


#: Repository-relative path of the Go module manifest. :func:`parse_repo`
#: surfaces ``go.mod`` events under this exact key when the manifest is
#: present at the repository root.
_GO_MOD_PATH: Final[str] = "go.mod"


# --- Public entry point -------------------------------------------------------


def recognize_constructs(
    tokens: Iterator[GoToken],
    path: str,
) -> Iterator[GoEvent]:
    """Yield :class:`GoEvent` instances for ``tokens`` in source order.

    Args:
        tokens: Iterator of :class:`GoToken` produced by
            :func:`tokenize_go_source` for a single Go source file.
        path: Repository-relative path of the file the tokens came
            from. Recorded on every emitted event's ``file_path``.

    Yields:
        :class:`ImportEvent`, :class:`FuncDeclEvent`,
        :class:`MethodCallEvent`, :class:`StructLitEvent`,
        :class:`PackageDocCommentEvent`, and :class:`SkipFileEvent`
        records, in source order. Whole-file skip cases (non-trivial
        build constraints, cgo) yield a single :class:`SkipFileEvent`
        and stop.

    Examples:
        >>> from project_knowledge_mcp.project_analyzer.go.go_tokenizer \\
        ...     import tokenize_go_source
        >>> source = 'package foo\\nimport "bar"\\n'
        >>> events = list(recognize_constructs(tokenize_go_source(source), "f.go"))
        >>> [type(e).__name__ for e in events]
        ['ImportEvent']
    """

    return _Recognizer(list(tokens), path).run()


def parse_go_mod(text: str) -> ModFileModuleEvent | None:
    """Recognize the ``module <module-path>`` declaration in a ``go.mod`` file.

    The Go module manifest has its own much simpler grammar than ``.go``
    source, so this recognizer uses a line-oriented scan rather than the
    Go tokenizer. Only the ``module`` line and its associated comments
    are extracted; ``require``, ``replace``, ``exclude``, ``retract``,
    and ``go <version>`` directives are ignored (Requirement 2.3 only
    needs the module path; Requirement 2.2 only needs the comment body).

    A ``//``-comment on the line immediately preceding the ``module``
    line — with no blank-line gap between — is captured as
    ``leading_comment``. A same-line trailing ``//``-comment is captured
    as ``trailing_comment``. Both bodies are stripped of the leading
    ``//`` and surrounding whitespace; an empty body is reported as
    ``None`` so downstream candidate-selection (Requirement 2.2 "comment
    body is non-empty") need not double-check.

    Args:
        text: Full text of the ``go.mod`` file. May use any line
            ending; :py:meth:`str.splitlines` handles ``\\n``,
            ``\\r\\n``, and ``\\r`` uniformly.

    Returns:
        A :class:`ModFileModuleEvent` populated from the first matching
        ``module`` line, or ``None`` when the file contains no
        well-formed ``module`` declaration.

    Examples:
        >>> ev = parse_go_mod("module github.com/acme/svc // payment service\\n")
        >>> ev.module_path
        'github.com/acme/svc'
        >>> ev.trailing_comment
        'payment service'
        >>> ev.leading_comment is None
        True
    """

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        match = _RE_GO_MOD_MODULE.match(line)
        if match is None:
            continue
        module_path = match.group(1)
        trailing_raw = match.group(2)
        trailing_comment = _normalize_mod_comment(trailing_raw)
        leading_comment = _leading_mod_comment(lines, idx)
        return ModFileModuleEvent(
            module_path=module_path,
            leading_comment=leading_comment,
            trailing_comment=trailing_comment,
            file_path="go.mod",
            line=idx + 1,
        )
    return None


def _normalize_mod_comment(body: str | None) -> str | None:
    """Trim a ``go.mod`` comment body, returning ``None`` when empty.

    Per Requirement 2.2, only non-empty bodies (after stripping the
    leading ``//`` and surrounding whitespace) qualify as purpose
    candidates. Returning ``None`` for empty / whitespace-only bodies
    matches the dataclass contract (``leading_comment``/``trailing_comment``
    are ``None`` when absent).
    """

    if body is None:
        return None
    stripped = body.strip()
    return stripped if stripped else None


def _leading_mod_comment(lines: list[str], idx: int) -> str | None:
    """Extract a ``//``-comment from the line immediately preceding ``idx``.

    A blank or non-comment preceding line yields ``None``. Only the
    single line directly before the ``module`` line is considered, so
    a multi-line comment block followed by a blank line and then the
    module line produces no leading comment — this matches the design's
    "no blank-line gap" rule (Requirement 2.2).
    """

    if idx == 0:
        return None
    prev = lines[idx - 1].strip()
    if not prev.startswith("//"):
        return None
    return _normalize_mod_comment(prev[2:])


# --- Public entry point: repo-level ------------------------------------------


def parse_repo(
    repository_contents: RepositoryContents,
) -> Mapping[str, list[GoEvent]]:
    """Tokenize and recognize every Go source file in the repository.

    This is the public entry point the four Go sub-analyzers consume.
    The aggregator calls it once per ``analyze()`` invocation and shares
    the resulting mapping across the sub-analyzers, so each ``.go`` file
    is tokenized and recognized exactly once per analysis run.

    Behavior, in order:

    1. Iterate the repository's file paths in lexicographic order.
       Sorted iteration makes the produced mapping order a pure function
       of the input (Requirement 11.4).
    2. Filter through :func:`is_go_source_file`, which rejects
       ``vendor/`` paths and non-``.go`` paths (Requirement 1.3).
    3. For each surviving path, tokenize the file with
       :func:`tokenize_go_source` and run the result through
       :func:`recognize_constructs`.
    4. A :class:`GoTokenizationError` raised at any point during
       tokenization is caught locally and replaced with a single
       :class:`SkipFileEvent` of reason
       ``"tokenization failed: <detail>"``. The rest of the repository
       continues to be parsed, so a malformed file never poisons the
       analysis (Requirement 11.1).
    5. When a ``go.mod`` exists at the repository root, the file is
       parsed by :func:`parse_go_mod` and its result is exposed under
       the path key ``"go.mod"``. When the manifest contains no
       well-formed ``module`` line, the value is the empty list (the
       manifest exists but contributes no events).

    Files for which :func:`recognize_constructs` itself yields a
    :class:`SkipFileEvent` (build constraints, ``import "C"``) appear
    in the mapping with that single event as their value, mirroring the
    tokenization-error case so the four downstream sub-analyzers can
    treat all three skip reasons uniformly (Requirement 10.4).

    Args:
        repository_contents: The Repository_Contents bundle for the
            project under analysis.

    Returns:
        A mapping from repository-relative path to the list of
        :class:`GoEvent` instances produced for that path. Files that
        :func:`is_go_source_file` rejects do not appear in the mapping.
        The ``"go.mod"`` key is present iff the manifest is present at
        the repository root.

    Examples:
        >>> from project_knowledge_mcp.models import RepositoryContents
        >>> rc = RepositoryContents(
        ...     gitlab_project_id=1,
        ...     commit_sha="abc",
        ...     files={
        ...         "main.go": "package main\\n",
        ...         "go.mod": "module github.com/acme/svc\\n",
        ...         "vendor/lib/a.go": "package lib\\n",
        ...     },
        ... )
        >>> events = parse_repo(rc)
        >>> sorted(events.keys())
        ['go.mod', 'main.go']
    """

    events_by_file: dict[str, list[GoEvent]] = {}

    for path in sorted(repository_contents.files):
        if not is_go_source_file(path):
            continue
        text = repository_contents.files[path]
        try:
            tokens = list(tokenize_go_source(text))
        except GoTokenizationError as err:
            events_by_file[path] = [
                SkipFileEvent(
                    reason=f"tokenization failed: {err.reason}",
                    file_path=path,
                    line=err.line,
                ),
            ]
            continue
        events_by_file[path] = list(recognize_constructs(iter(tokens), path))

    gomod_text = repository_contents.files.get(_GO_MOD_PATH)
    if gomod_text is not None:
        mod_event = parse_go_mod(gomod_text)
        events_by_file[_GO_MOD_PATH] = [mod_event] if mod_event is not None else []

    return events_by_file


# --- Internal recognizer ------------------------------------------------------


class _Recognizer:
    """Stateful walker backing :func:`recognize_constructs`.

    Holds the materialized token list, a precomputed mapping from
    significant-token index to original-token index, and the
    repository-relative path. The ``run`` generator orchestrates the
    three pre-passes (build-constraint scan, cgo scan, package
    doc-comment scan) and then drives the construct walker.
    """

    __slots__ = ("_n", "_path", "_sig", "_tokens")

    def __init__(self, tokens: list[GoToken], path: str) -> None:
        self._tokens: list[GoToken] = tokens
        self._n: int = len(tokens)
        self._path: str = path
        # Index map from significant-position to original token index.
        # Significant tokens are everything not in ``_TRIVIA_KINDS``.
        self._sig: list[int] = [
            i for i, tok in enumerate(tokens) if tok.kind not in _TRIVIA_KINDS
        ]

    # --- Orchestration -----------------------------------------------

    def run(self) -> Iterator[GoEvent]:
        """Drive the recognizer pipeline.

        Order of operations:

        1. Locate the first ``package`` keyword (if any). Comments before
           it form the candidate doc-comment block and the only place
           build constraints are honored.
        2. Scan leading comments for non-trivial build constraints. On
           hit, yield one :class:`SkipFileEvent` and stop.
        3. Scan all imports for the unaliased ``import "C"`` directive.
           On hit, yield one :class:`SkipFileEvent` and stop.
        4. If a package keyword exists, emit the
           :class:`PackageDocCommentEvent` (when a contiguous
           comment block precedes it with no blank-line gap).
        5. Walk all significant tokens, dispatching on top-level
           keywords (``import``, ``func``, ``type``) and recognizing
           method calls and struct literals at any depth.
        """

        pkg_t_idx = self._find_first_package_token()

        skip = self._scan_for_skip_file(pkg_t_idx)
        if skip is not None:
            yield skip
            return

        if pkg_t_idx >= 0:
            doc = self._build_package_doc_comment(pkg_t_idx)
            if doc is not None:
                yield doc

        yield from self._walk()


    def _find_first_package_token(self) -> int:
        """Return the original-token index of the first ``package`` keyword.

        Returns ``-1`` when the file contains no ``package`` keyword,
        which can happen for fragments and during error recovery.
        """

        for i, tok in enumerate(self._tokens):
            if tok.kind is GoTokenKind.PACKAGE_KEYWORD:
                return i
        return -1

    def _scan_for_skip_file(self, pkg_t_idx: int) -> SkipFileEvent | None:
        """Return a whole-file :class:`SkipFileEvent` when one applies.

        Order of checks matches Requirement 10.4: build constraints are
        honored before cgo because a build-constraint-excluded file is
        excluded regardless of its imports.
        """

        scan_end = pkg_t_idx if pkg_t_idx >= 0 else self._n
        skip = self._check_build_constraints(scan_end)
        if skip is not None:
            return skip
        return self._check_cgo()

    # --- Build-constraint scan ---------------------------------------

    def _check_build_constraints(self, scan_end: int) -> SkipFileEvent | None:
        """Scan tokens [0, scan_end) for a non-trivial build constraint.

        The Go spec requires build-constraint comments to appear before
        the ``package`` line; restricting the scan to the leading
        section avoids treating a stray comment elsewhere in the file
        as a build constraint.
        """

        for i in range(min(scan_end, self._n)):
            tok = self._tokens[i]
            if tok.kind is not GoTokenKind.BUILD_CONSTRAINT_COMMENT:
                continue
            expr = _extract_build_constraint_expression(tok.text)
            if expr is None:
                continue
            stripped = expr.strip()
            if stripped not in _TRIVIAL_BUILD_CONSTRAINTS:
                return SkipFileEvent(
                    reason="build constraint requires toolchain",
                    file_path=self._path,
                    line=tok.line,
                )
        return None


    # --- cgo scan -----------------------------------------------------

    def _check_cgo(self) -> SkipFileEvent | None:
        """Return a :class:`SkipFileEvent` when an unaliased ``import "C"`` exists.

        Both single-line ``import "C"`` and parenthesized block
        ``import ( ... "C" ... )`` forms are detected. Aliased imports
        (``import _ "C"``, ``import C2 "C"``) do not satisfy the
        unaliased shape and do not trigger the skip; the design carve-out
        is intentional because aliased imports of ``"C"`` are still
        cgo-bound, but the four sample repositories never use that form
        and treating them as cgo would over-skip.
        """

        i = 0
        n_sig = len(self._sig)
        while i < n_sig:
            tok = self._tokens[self._sig[i]]
            if tok.kind is not GoTokenKind.IMPORT_KEYWORD:
                i += 1
                continue
            skip = self._cgo_in_import_at(i)
            if skip is not None:
                return skip
            i = self._end_of_import_decl(i)
        return None

    def _cgo_in_import_at(self, s: int) -> SkipFileEvent | None:
        """Inspect the import declaration starting at sig index ``s``.

        ``s`` indexes the ``import`` keyword itself; the inspection
        looks at the next significant token to decide between the
        single-line and block forms. Only an unaliased ``"C"`` literal
        produces a hit.
        """

        n_sig = len(self._sig)
        if s + 1 >= n_sig:
            return None
        nxt = self._tokens[self._sig[s + 1]]

        if nxt.kind is GoTokenKind.STRING_LITERAL:
            return self._cgo_check_literal(nxt)

        if nxt.kind is GoTokenKind.LPAREN:
            return self._cgo_check_block(s + 1)

        return None


    def _cgo_check_literal(self, tok: GoToken) -> SkipFileEvent | None:
        """Return a cgo skip event when ``tok`` is the unaliased ``"C"`` literal."""

        if _string_literal_value(tok.text) == "C":
            return SkipFileEvent(
                reason="cgo directive requires toolchain",
                file_path=self._path,
                line=tok.line,
            )
        return None

    def _cgo_check_block(self, lparen_s: int) -> SkipFileEvent | None:
        """Walk a parenthesized import block and check each entry for cgo.

        ``lparen_s`` is the sig index of the opening ``(``. The block is
        scanned entry by entry; aliased entries (``alias "path"``,
        ``. "path"``, ``_ "path"``) only count as cgo when the literal
        appears alone — the shared rule with :meth:`_cgo_in_import_at`.
        """

        end_s = self._find_matching_paren(lparen_s)
        if end_s < 0:
            return None
        i = lparen_s + 1
        while i < end_s:
            tok = self._tokens[self._sig[i]]
            if (
                tok.kind is GoTokenKind.STRING_LITERAL
                and not self._import_entry_is_aliased(lparen_s + 1, i, end_s)
            ):
                # Standalone string literal in a block import; check
                # whether it is the unaliased ``"C"`` directive.
                skip = self._cgo_check_literal(tok)
                if skip is not None:
                    return skip
            i += 1
        return None

    def _import_entry_is_aliased(self, block_start: int, lit_s: int, _block_end: int) -> bool:
        """Return ``True`` when the string literal at ``lit_s`` is aliased.

        Looks at the previous significant token within the block: if it
        is the opening ``(``, a comma-equivalent (impossible here since
        the tokenizer drops only newlines as the entry separator), or
        another string literal end, the literal stands alone.

        For the purposes of cgo detection we treat any preceding
        identifier, ``.``, or ``_`` (also tokenized as IDENTIFIER) as an
        alias. Block-style imports use newline separation rather than
        commas, so this lookback is the cleanest way to distinguish
        ``"C"`` from e.g. ``alias "C"``.
        """

        if lit_s <= block_start:
            return False
        prev = self._tokens[self._sig[lit_s - 1]]
        if prev.kind is GoTokenKind.IDENTIFIER:
            return True
        return prev.kind is GoTokenKind.DOT


    def _end_of_import_decl(self, s: int) -> int:
        """Return sig index just past the import declaration starting at ``s``.

        Used by the cgo pre-pass to advance past one import declaration
        before looking for the next ``import`` keyword. ``s`` indexes
        the ``import`` keyword itself.
        """

        n_sig = len(self._sig)
        if s + 1 >= n_sig:
            return n_sig
        nxt = self._tokens[self._sig[s + 1]]
        if nxt.kind is GoTokenKind.LPAREN:
            end = self._find_matching_paren(s + 1)
            if end < 0:
                return n_sig
            return end + 1
        # Single-line form: alias? then string. Two or three sig tokens
        # past the ``import`` keyword.
        if nxt.kind is GoTokenKind.STRING_LITERAL:
            return s + 2
        # Aliased single-line: skip alias + string.
        if s + 2 < n_sig:
            return s + 3
        return n_sig

    # --- Package doc-comment scan ------------------------------------

    def _build_package_doc_comment(self, pkg_t_idx: int) -> PackageDocCommentEvent | None:
        """Return a :class:`PackageDocCommentEvent` for the doc-comment block.

        The block is the contiguous run of ``//``-line and ``/*...*/``
        block comments immediately preceding the ``package`` line, with
        no blank-line gap (Requirement 2.4 / design §"go.go_purpose"). A
        blank line is two consecutive ``NEWLINE`` tokens with only
        ``WHITESPACE`` between them.

        Returns ``None`` when no comment block is adjacent to the
        package line, or when the block has only empty comment bodies.
        """

        comment_tokens = self._collect_leading_doc_comments(pkg_t_idx)
        if not comment_tokens:
            return None
        body_lines = _strip_comment_bodies(comment_tokens)
        if not body_lines:
            return None
        return PackageDocCommentEvent(
            text=" ".join(body_lines),
            file_path=self._path,
            line=comment_tokens[0].line,
        )


    def _collect_leading_doc_comments(self, pkg_t_idx: int) -> list[GoToken]:
        """Walk backward from the package keyword collecting doc-comment tokens.

        Returns the comment tokens in source order. Stops when:

        * a blank line is encountered (two consecutive newlines with
          only whitespace between them),
        * a non-comment, non-trivia token is encountered,
        * the start of the file is reached.
        """

        comments: list[GoToken] = []
        i = pkg_t_idx - 1

        # Skip the trailing whitespace and at most one newline that
        # separates the package keyword from the preceding line.
        i, gap_too_wide = self._skip_one_newline_block(i)
        if gap_too_wide:
            return comments

        while i >= 0:
            kind = self._tokens[i].kind
            if kind in (GoTokenKind.LINE_COMMENT, GoTokenKind.BLOCK_COMMENT):
                comments.append(self._tokens[i])
                i -= 1
                i, gap_too_wide = self._skip_one_newline_block(i)
                if gap_too_wide:
                    break
                continue
            break

        comments.reverse()
        return comments

    def _skip_one_newline_block(self, i: int) -> tuple[int, bool]:
        """Skip whitespace and at most one newline going backward.

        Returns ``(new_index, blank_line_seen)``. A blank line is two
        newlines with only whitespace between them. The walker stops
        collecting doc comments as soon as a blank line appears, since
        that is the exact rule the design uses for "no blank line
        separating the comment block from the package line".
        """

        saw_newline = False
        while i >= 0:
            kind = self._tokens[i].kind
            if kind is GoTokenKind.WHITESPACE:
                i -= 1
                continue
            if kind is GoTokenKind.NEWLINE:
                if saw_newline:
                    return i, True
                saw_newline = True
                i -= 1
                continue
            return i, False
        return i, False


    # --- Top-level walker --------------------------------------------

    def _walk(self) -> Iterator[GoEvent]:
        """Walk significant tokens dispatching top-level constructs.

        Maintains an explicit depth counter (paren + brace + bracket).
        At depth zero, dispatches on the dedicated top-level keyword
        kinds (``package``, ``import``, ``func``, ``type``) and on the
        identifier-text top-level keywords (``var``, ``const``).

        At any depth, attempts to recognize a method call or composite
        literal at the current sig position. Recognition is balanced —
        the matched construct's internal parens/braces/brackets stay
        accounted for — so depth tracking remains consistent across
        match advances.
        """

        sig = self._sig
        n_sig = len(sig)
        s = 0
        depth = _DepthTracker()

        while s < n_sig:
            tok = self._tokens[sig[s]]
            kind = tok.kind

            if depth.is_zero() and self._is_top_level_keyword(tok):
                next_s, events = self._dispatch_top_level(s)
                yield from events
                s = next_s
                continue

            match_events, next_s = self._try_match_call_or_struct(s)
            if match_events is not None:
                yield from match_events
                s = next_s
                continue

            depth.update(kind)
            s += 1

    @staticmethod
    def _is_top_level_keyword(tok: GoToken) -> bool:
        """Return ``True`` when ``tok`` opens a top-level declaration."""

        if tok.kind in _TOP_LEVEL_KEYWORD_KINDS:
            return True
        return tok.kind is GoTokenKind.IDENTIFIER and tok.text in _TOP_LEVEL_KEYWORD_IDENTS


    def _dispatch_top_level(self, s: int) -> tuple[int, list[GoEvent]]:
        """Dispatch the top-level construct at sig index ``s``.

        Each construct is parsed inside a ``try``/``except _ParseError``
        boundary; on failure a :class:`SkipFileEvent` is emitted and
        parsing resumes at the next package-level boundary
        (Requirement 11.1). The package keyword itself never raises —
        it's just consumed and the walker continues.
        """

        tok = self._tokens[self._sig[s]]
        kind = tok.kind

        if kind is GoTokenKind.PACKAGE_KEYWORD:
            # ``package <name>`` — consume the keyword and the following
            # identifier; emit no event (the package doc-comment was
            # already emitted by ``run``).
            return min(s + 2, len(self._sig)), []

        if kind is GoTokenKind.IMPORT_KEYWORD:
            return self._dispatch_with_recovery(s, "import", self._parse_import)

        if kind is GoTokenKind.FUNC_KEYWORD:
            return self._dispatch_with_recovery(s, "func decl", self._parse_func_decl_header)

        if kind is GoTokenKind.TYPE_KEYWORD:
            return self._dispatch_with_recovery(s, "type decl", self._parse_type_skip)

        # ``var`` / ``const``: tracked so recovery uses them as
        # boundaries, but their initializers are walked by the main
        # loop's per-token call/struct matcher (no dedicated parser
        # needed because the constructs we care about — calls and
        # struct literals — can appear inside any expression position).
        return s + 1, []

    def _dispatch_with_recovery(
        self,
        s: int,
        construct_label: str,
        parser: _TopLevelParser,
    ) -> tuple[int, list[GoEvent]]:
        """Run ``parser`` with parse-error recovery.

        On :class:`_ParseError`, emits one :class:`SkipFileEvent` whose
        ``reason`` records the failing construct label and the
        exception's message, and advances ``s`` to the next package-
        level boundary so the next construct can be tried.
        """

        tok = self._tokens[self._sig[s]]
        try:
            events, next_s = parser(s)
        except _ParseError as err:
            skip = SkipFileEvent(
                reason=f"parse error in {construct_label}: {err}",
                file_path=self._path,
                line=tok.line,
            )
            return self._next_top_level_boundary(s + 1), [skip]
        return next_s, list(events)


    def _next_top_level_boundary(self, s: int) -> int:
        """Return the next sig index that begins a top-level declaration.

        Used by parse-error recovery: when a top-level construct is
        malformed, the walker advances to the next plausible boundary
        rather than abandoning the rest of the file. Boundaries are
        ``package``, ``import``, ``func``, ``type`` (dedicated kinds)
        and ``var`` / ``const`` (identifier text).
        """

        n_sig = len(self._sig)
        depth = _DepthTracker()
        while s < n_sig:
            tok = self._tokens[self._sig[s]]
            if depth.is_zero() and self._is_top_level_keyword(tok):
                return s
            depth.update(tok.kind)
            s += 1
        return n_sig

    # --- Import declaration parser -----------------------------------

    def _parse_import(self, s: int) -> tuple[list[GoEvent], int]:
        """Parse ``import "<path>"`` or ``import ( ... )``.

        ``s`` is the sig index of the ``import`` keyword. Returns a
        list of :class:`ImportEvent` (one per imported path) and the
        sig index immediately after the declaration.
        """

        n_sig = len(self._sig)
        s += 1  # skip the ``import`` keyword
        if s >= n_sig:
            return [], s

        nxt = self._tokens[self._sig[s]]
        if nxt.kind is GoTokenKind.LPAREN:
            return self._parse_import_block(s)
        return self._parse_import_single(s)

    def _parse_import_single(self, s: int) -> tuple[list[GoEvent], int]:
        """Parse a single-line import: ``[alias] "path"``.

        ``s`` indexes the first token after ``import``. Trailing
        whitespace and newlines are not part of the sig stream.
        """

        alias, s = self._parse_optional_import_alias(s)
        events, s = self._parse_import_path(s, alias)
        return events, s


    def _parse_import_block(self, lparen_s: int) -> tuple[list[GoEvent], int]:
        """Parse ``import ( ... )`` block, emitting one event per entry.

        ``lparen_s`` is the sig index of ``(``. Each entry has the
        same ``[alias] "path"`` shape as a single-line import; the
        block separator in source is a newline, which is stripped from
        the sig stream so entries appear back-to-back here.
        """

        end_s = self._find_matching_paren(lparen_s)
        if end_s < 0:
            raise _ParseError("unmatched ( in import block")

        events: list[GoEvent] = []
        i = lparen_s + 1
        while i < end_s:
            alias, after_alias = self._parse_optional_import_alias(i)
            if after_alias >= end_s:
                break
            entry_events, i = self._parse_import_path(after_alias, alias)
            events.extend(entry_events)
        return events, end_s + 1

    def _parse_optional_import_alias(self, s: int) -> tuple[str | None, int]:
        """Recognize an optional alias prefix on an import entry.

        Recognized forms:

        * ``IDENT "path"`` → alias is the identifier text
        * ``. "path"`` → alias is ``"."`` (dot import)
        * ``"path"`` → no alias

        Returns ``(alias_or_None, sig_index_of_string_literal_or_first_token_after_alias)``.
        Does not consume the path string itself; that is the caller's
        responsibility.
        """

        n_sig = len(self._sig)
        if s >= n_sig:
            return None, s
        tok = self._tokens[self._sig[s]]
        if tok.kind is GoTokenKind.IDENTIFIER:
            return tok.text, s + 1
        if tok.kind is GoTokenKind.DOT:
            return ".", s + 1
        return None, s


    def _parse_import_path(
        self,
        s: int,
        alias: str | None,
    ) -> tuple[list[GoEvent], int]:
        """Consume the path string literal of an import entry.

        Returns ``(events, next_sig_index)``. When the token at ``s`` is
        not a string literal, an :class:`_ImportPathSkip` is taken: the
        caller advances by one sig index to avoid an infinite loop on
        malformed import blocks. (The recovery is contained at the
        block-parse boundary so the rest of the file still parses.)
        """

        n_sig = len(self._sig)
        if s >= n_sig:
            return [], s
        tok = self._tokens[self._sig[s]]
        if tok.kind is not GoTokenKind.STRING_LITERAL:
            return [], s + 1
        path = _string_literal_value(tok.text)
        event = ImportEvent(
            path=path,
            alias=alias,
            file_path=self._path,
            line=tok.line,
        )
        return [event], s + 1

    # --- Type declaration "skip" parser ------------------------------

    def _parse_type_skip(self, s: int) -> tuple[list[GoEvent], int]:
        """Skip a ``type`` declaration without emitting any event.

        Type declarations contain ``struct { ... }`` and
        ``interface { ... }`` bodies whose contents look superficially
        like struct literals (identifier-followed-by-LBRACE prefix).
        Skipping the entire declaration prevents the call/struct
        matcher from misinterpreting field declarations as struct
        literals or method calls.

        Recognizes the parenthesized block form ``type ( ... )`` and
        the single declaration form ``type Name ...``. The single form
        is bounded by the next top-level keyword or end-of-file; this
        is conservative but safe because top-level declarations are
        always separated by newlines (which the tokenizer already
        treats as terminators for the parser's purposes).
        """

        n_sig = len(self._sig)
        s += 1  # skip ``type`` keyword
        if s >= n_sig:
            return [], s

        if self._tokens[self._sig[s]].kind is GoTokenKind.LPAREN:
            end_s = self._find_matching_paren(s)
            if end_s < 0:
                raise _ParseError("unmatched ( in type block")
            return [], end_s + 1

        return [], self._skip_single_type_decl(s)


    def _skip_single_type_decl(self, s: int) -> int:
        """Skip a single ``type Name ...`` declaration.

        Walks balanced parens, brackets, and braces. When the first
        token of the body is the dedicated ``struct`` or ``interface``
        keyword, the matching brace is found explicitly so the entire
        body is skipped in one step. Otherwise the walker advances
        token by token until the next top-level boundary.
        """

        n_sig = len(self._sig)
        # Skip the optional name + optional ``=`` for type aliases.
        if s < n_sig and self._tokens[self._sig[s]].kind is GoTokenKind.IDENTIFIER:
            s += 1
        if s < n_sig and self._tokens[self._sig[s]].kind is GoTokenKind.ASSIGN:
            s += 1
        # Optional generic type parameters [T any].
        if s < n_sig and self._tokens[self._sig[s]].kind is GoTokenKind.LBRACKET:
            end = self._find_matching_bracket(s)
            if end < 0:
                raise _ParseError("unmatched [ in type params")
            s = end + 1

        if s >= n_sig:
            return s

        first = self._tokens[self._sig[s]]
        if first.kind in (GoTokenKind.STRUCT_KEYWORD, GoTokenKind.INTERFACE_KEYWORD):
            s += 1
            if s < n_sig and self._tokens[self._sig[s]].kind is GoTokenKind.LBRACE:
                end = self._find_matching_brace(s)
                if end < 0:
                    raise _ParseError("unmatched { in struct/interface body")
                return end + 1
            return s

        # Walk forward until the next top-level boundary, treating
        # nested parens/brackets/braces as opaque.
        return self._walk_to_boundary(s)


    def _walk_to_boundary(self, s: int) -> int:
        """Walk forward to the next top-level boundary, ignoring nested groups.

        Stops at the first sig position where:

        * depth is zero, AND
        * the token is a top-level keyword (dedicated kind or
          ``var``/``const`` identifier).
        """

        n_sig = len(self._sig)
        depth = _DepthTracker()
        while s < n_sig:
            tok = self._tokens[self._sig[s]]
            if depth.is_zero() and self._is_top_level_keyword(tok):
                return s
            depth.update(tok.kind)
            s += 1
        return n_sig

    # --- Function declaration parser ---------------------------------

    def _parse_func_decl_header(self, s: int) -> tuple[list[GoEvent], int]:
        """Parse ``func [(recv)] Name [generic-params] (params) [returns]``.

        Emits a single :class:`FuncDeclEvent`. The body is **not**
        traversed here; the main walker takes over after the opening
        ``{`` and recognizes calls and struct literals naturally,
        which keeps the depth tracker monotonic across the body.

        Returns ``(events, next_sig_index)`` where ``next_sig_index``
        points at the body-opening ``{`` (so the main walker descends
        into the body) or past the declaration when there is no body.
        """

        sig = self._sig
        n_sig = len(sig)
        func_tok = self._tokens[sig[s]]
        func_line = func_tok.line
        s += 1  # consume ``func``

        receiver_type, s = self._parse_optional_receiver(s)
        name, s = self._parse_func_name(s)
        s = self._skip_optional_generic_params(s)
        s = self._skip_func_params(s)
        s = self._skip_optional_return_type(s)

        if s >= n_sig or self._tokens[sig[s]].kind is not GoTokenKind.LBRACE:
            # No body: forward declaration / interface method at top
            # level, or declaration terminated by another top-level
            # keyword. Emit a body-less event with an empty range.
            event = FuncDeclEvent(
                name=name,
                receiver_type=receiver_type,
                file_path=self._path,
                line=func_line,
                body_token_range=(0, 0),
            )
            return [event], s

        body_open_t_idx = sig[s]
        body_close_s = self._find_matching_brace(s)
        if body_close_s < 0:
            body_token_range = (body_open_t_idx, self._n)
        else:
            body_token_range = (body_open_t_idx, sig[body_close_s] + 1)

        event = FuncDeclEvent(
            name=name,
            receiver_type=receiver_type,
            file_path=self._path,
            line=func_line,
            body_token_range=body_token_range,
        )
        return [event], s


    def _parse_optional_receiver(self, s: int) -> tuple[str | None, int]:
        """Parse an optional ``(recv)`` block, returning the receiver type.

        Recognized receiver shapes:

        * ``(Foo)`` → ``"Foo"``
        * ``(*Foo)`` → ``"*Foo"``
        * ``(r Foo)`` → ``"Foo"`` (receiver name discarded)
        * ``(r *Foo)`` → ``"*Foo"``
        * ``(r *pkg.Foo)`` → ``"*pkg.Foo"``

        Returns ``(receiver_type_or_None, next_sig_index)``.
        """

        n_sig = len(self._sig)
        if s >= n_sig or self._tokens[self._sig[s]].kind is not GoTokenKind.LPAREN:
            return None, s
        end = self._find_matching_paren(s)
        if end < 0:
            raise _ParseError("unmatched ( in receiver")
        receiver_type = self._extract_receiver_type(s + 1, end)
        return receiver_type, end + 1

    def _extract_receiver_type(self, start_s: int, end_s: int) -> str | None:
        """Concatenate the receiver type tokens between ``[start_s, end_s)``.

        If the first token is an identifier followed by another
        identifier or ``*``, that first token is the receiver name and
        is dropped. The remaining tokens are concatenated without
        whitespace, which handles ``*pkg.Foo`` correctly because
        ``.`` is its own token kind.
        """

        if end_s <= start_s:
            return None
        sig = self._sig
        first = self._tokens[sig[start_s]]
        if first.kind is GoTokenKind.IDENTIFIER and start_s + 1 < end_s:
            second = self._tokens[sig[start_s + 1]]
            if second.kind in (GoTokenKind.IDENTIFIER, GoTokenKind.STAR):
                start_s += 1
        parts = [self._tokens[sig[i]].text for i in range(start_s, end_s)]
        return "".join(parts)


    def _parse_func_name(self, s: int) -> tuple[str, int]:
        """Consume the function name identifier."""

        n_sig = len(self._sig)
        if s >= n_sig or self._tokens[self._sig[s]].kind is not GoTokenKind.IDENTIFIER:
            raise _ParseError("expected function name")
        return self._tokens[self._sig[s]].text, s + 1

    def _skip_optional_generic_params(self, s: int) -> int:
        """Skip a ``[T any, U comparable]`` generic-parameter clause."""

        n_sig = len(self._sig)
        if s >= n_sig or self._tokens[self._sig[s]].kind is not GoTokenKind.LBRACKET:
            return s
        end = self._find_matching_bracket(s)
        if end < 0:
            raise _ParseError("unmatched [ in type params")
        return end + 1

    def _skip_func_params(self, s: int) -> int:
        """Skip the ``(params)`` clause of a function declaration."""

        n_sig = len(self._sig)
        if s >= n_sig or self._tokens[self._sig[s]].kind is not GoTokenKind.LPAREN:
            raise _ParseError("expected ( for params")
        end = self._find_matching_paren(s)
        if end < 0:
            raise _ParseError("unmatched ( in params")
        return end + 1

    def _skip_optional_return_type(self, s: int) -> int:
        """Walk forward past the return-type clause to the body-opening ``{``.

        The return-type clause ends at the body-opening ``{``, the next
        top-level boundary (in the no-body case), or end-of-file.
        Nested parens and brackets are skipped as opaque groups.
        """

        n_sig = len(self._sig)
        while s < n_sig:
            tok = self._tokens[self._sig[s]]
            if tok.kind is GoTokenKind.LBRACE:
                return s
            if self._is_top_level_keyword(tok):
                return s
            if tok.kind is GoTokenKind.LPAREN:
                end = self._find_matching_paren(s)
                if end < 0:
                    raise _ParseError("unmatched ( in return type")
                s = end + 1
                continue
            if tok.kind is GoTokenKind.LBRACKET:
                end = self._find_matching_bracket(s)
                if end < 0:
                    raise _ParseError("unmatched [ in return type")
                s = end + 1
                continue
            s += 1
        return s


    # --- Method-call / struct-literal recognizer ----------------------

    def _try_match_call_or_struct(
        self,
        s: int,
    ) -> tuple[list[GoEvent] | None, int]:
        """Attempt to recognize a call or composite literal at sig[s].

        On match, returns ``(events, next_sig_index)`` where ``events``
        is a single-element list and ``next_sig_index`` is just past
        the matched construct (after the closing ``)`` for calls or
        ``}`` for literals).

        On no-match, returns ``(None, s)`` and the caller advances by
        one sig token.

        The matcher is shared between calls and struct literals
        because they share the prefix grammar
        ``[& or *] IDENT [. IDENT]*``. After consuming the prefix,
        whether the construct is a call (``(...)`` follows) or a
        composite literal (``{IDENT: ...}`` follows) is decided by
        the next token.
        """

        prefix = self._parse_call_struct_prefix(s)
        if prefix is None:
            return None, s
        is_pointer, chain, line, after_chain_s = prefix

        if after_chain_s >= len(self._sig):
            return None, s

        nxt_kind = self._tokens[self._sig[after_chain_s]].kind
        if nxt_kind is GoTokenKind.LPAREN:
            return self._build_method_call(is_pointer, chain, line, after_chain_s, s)
        if nxt_kind is GoTokenKind.LBRACE:
            return self._build_struct_lit(is_pointer, chain, line, after_chain_s, s)
        return None, s


    def _parse_call_struct_prefix(
        self,
        s: int,
    ) -> tuple[bool, list[str], int, int] | None:
        """Parse the dotted-identifier prefix shared by calls and struct lits.

        Returns ``(is_pointer, chain, line_of_first_token, sig_index_after_chain)``
        or ``None`` when the position does not start a recognizable
        prefix.

        ``is_pointer`` is true when an ``&`` or ``*`` prefix was
        consumed. The construct's reported source line is the line of
        the optional prefix (or the first identifier when no prefix is
        present), matching the design's "first significant token of
        the construct" rule.
        """

        sig = self._sig
        n_sig = len(sig)
        if s >= n_sig:
            return None

        is_pointer = False
        line = self._tokens[sig[s]].line
        first_kind = self._tokens[sig[s]].kind
        if first_kind in (GoTokenKind.AMPERSAND, GoTokenKind.STAR):
            is_pointer = True
            s += 1
            if s >= n_sig:
                return None

        if self._tokens[sig[s]].kind is not GoTokenKind.IDENTIFIER:
            return None

        chain = [self._tokens[sig[s]].text]
        if not is_pointer:
            line = self._tokens[sig[s]].line
        s += 1

        # Collect ``.IDENT`` repetitions.
        while s + 1 < n_sig:
            if (
                self._tokens[sig[s]].kind is GoTokenKind.DOT
                and self._tokens[sig[s + 1]].kind is GoTokenKind.IDENTIFIER
            ):
                chain.append(self._tokens[sig[s + 1]].text)
                s += 2
            else:
                break

        return is_pointer, chain, line, s


    def _build_method_call(
        self,
        is_pointer: bool,
        chain: list[str],
        line: int,
        lparen_s: int,
        match_start: int,
    ) -> tuple[list[GoEvent] | None, int]:
        """Construct a :class:`MethodCallEvent` from a recognized prefix + ``(``.

        ``&Foo()`` and ``*Foo()`` are not valid Go expressions, so a
        pointer-prefixed call is rejected (returns no-match). The
        method name is the last segment of the dotted chain; the
        receiver chain is everything before it (the empty tuple for
        unqualified ``foo()`` calls).
        """

        if is_pointer:
            return None, match_start
        try:
            args, end_s = self._parse_call_args(lparen_s)
        except _ParseError:
            return None, match_start

        method_name = chain[-1]
        receiver_chain = tuple(chain[:-1])
        event = MethodCallEvent(
            receiver_chain=receiver_chain,
            method_name=method_name,
            args=args,
            file_path=self._path,
            line=line,
        )
        return [event], end_s

    def _build_struct_lit(
        self,
        is_pointer: bool,
        chain: list[str],
        line: int,
        lbrace_s: int,
        match_start: int,
    ) -> tuple[list[GoEvent] | None, int]:
        """Construct a :class:`StructLitEvent` from prefix + ``{``.

        Only the named-field shape ``{IDENT: <value>, ...}`` is
        recognized; positional struct literals and slice/map literals
        are deliberately not matched here. The ``{IDENT: ...}``
        lookahead also rules out type-declaration bodies and
        block statements which never have a leading ``IDENT:``.
        """

        if not self._brace_starts_named_fields(lbrace_s):
            return None, match_start
        type_name, package_alias = self._split_chain_for_struct(chain)
        if type_name is None:
            return None, match_start
        try:
            fields, end_s = self._parse_struct_fields(lbrace_s)
        except _ParseError:
            return None, match_start
        event = StructLitEvent(
            type_name=type_name,
            package_alias=package_alias,
            fields=fields,
            is_pointer=is_pointer,
            file_path=self._path,
            line=line,
        )
        return [event], end_s


    def _brace_starts_named_fields(self, lbrace_s: int) -> bool:
        """Return ``True`` when the ``{`` at ``lbrace_s`` begins a named-field
        composite literal.

        The signature shape required: the first non-trivia token after
        ``{`` is an identifier followed by ``:``. Empty composite
        literals (``T{}``) deliberately do not satisfy this — without
        a named field we cannot distinguish them from empty slice
        literals like ``[]T{}`` and the design's recognized shapes
        always carry at least one named field.
        """

        sig = self._sig
        n_sig = len(sig)
        s = lbrace_s + 1
        if s + 1 >= n_sig:
            return False
        first = self._tokens[sig[s]]
        second = self._tokens[sig[s + 1]]
        return first.kind is GoTokenKind.IDENTIFIER and second.kind is GoTokenKind.COLON

    @staticmethod
    def _split_chain_for_struct(chain: list[str]) -> tuple[str | None, str | None]:
        """Map a dotted chain onto ``(type_name, package_alias)``.

        Recognized lengths:

        * 1 segment: ``(chain[0], None)`` — unqualified type.
        * 2 segments: ``(chain[1], chain[0])`` — ``pkg.Type``.

        Longer chains are not recognized; ``a.b.c{...}`` would be a
        struct field reference at runtime and not a type expression in
        Go's grammar, so returning ``(None, None)`` produces a
        no-match.
        """

        if len(chain) == 1:
            return chain[0], None
        if len(chain) == _PKG_QUALIFIED_CHAIN_LEN:
            return chain[1], chain[0]
        return None, None


    # --- Argument and field parsers ----------------------------------

    def _parse_call_args(self, lparen_s: int) -> tuple[tuple[ArgRef, ...], int]:
        """Parse ``( arg, arg, ... )`` starting at the ``(``.

        Returns ``(args_tuple, sig_index_after_RPAREN)``. Trailing
        commas are tolerated. Each argument is parsed by
        :meth:`_parse_arg`, which is recursive — nested calls and
        struct literals are captured as :class:`CallArg` /
        :class:`StructLitArg` rather than being yielded as separate
        top-level events.
        """

        sig = self._sig
        n_sig = len(sig)
        s = lparen_s + 1
        args: list[ArgRef] = []

        while s < n_sig:
            tok = self._tokens[sig[s]]
            if tok.kind is GoTokenKind.RPAREN:
                return tuple(args), s + 1
            if tok.kind is GoTokenKind.COMMA:
                # Tolerate a leading or extra comma — the source
                # representation may have stripped a trailing newline
                # before the closing paren.
                s += 1
                continue
            before = s
            arg, s = self._parse_arg(s)
            args.append(arg)
            # Progress guard: ``_parse_arg`` is contractually required
            # to advance ``s`` by at least one significant token. A
            # malformed input (e.g. a stray ``}`` or ``]`` inside the
            # argument list, which ``_parse_unknown_arg`` would treat
            # as an end-of-arg sentinel and return ``s`` unchanged)
            # could otherwise loop forever appending empty
            # ``UnknownArg`` entries. Bail out as a parse error so the
            # caller's no-match path (``_build_method_call`` /
            # ``_build_struct_lit``) rejects the construct and
            # ``_walk`` resumes at the next token.
            if s <= before:
                raise _ParseError(
                    "argument parser did not advance; argument list "
                    "is malformed"
                )
            if s < n_sig and self._tokens[sig[s]].kind is GoTokenKind.COMMA:
                s += 1
        raise _ParseError("unterminated argument list")

    def _parse_struct_fields(
        self,
        lbrace_s: int,
    ) -> tuple[tuple[tuple[str, ArgRef], ...], int]:
        """Parse ``{ name: value, name: value, ... }``.

        Returns ``(fields_tuple, sig_index_after_RBRACE)``. The first
        token inside the brace is required to be an identifier
        followed by a colon — :meth:`_brace_starts_named_fields` already
        verified this. Subsequent fields are separated by commas with
        an optional trailing comma.
        """

        sig = self._sig
        n_sig = len(sig)
        s = lbrace_s + 1
        fields: list[tuple[str, ArgRef]] = []

        while s < n_sig:
            tok = self._tokens[sig[s]]
            if tok.kind is GoTokenKind.RBRACE:
                return tuple(fields), s + 1
            if tok.kind is GoTokenKind.COMMA:
                s += 1
                continue
            before = s
            field, s = self._parse_struct_field(s)
            fields.append(field)
            # Same progress guard as ``_parse_call_args`` —
            # ``_parse_struct_field`` ultimately also delegates to
            # ``_parse_arg`` for the value, which can fail to advance
            # on a stray closer at the field-value position.
            if s <= before:
                raise _ParseError(
                    "struct field parser did not advance; struct "
                    "literal is malformed"
                )
            if s < n_sig and self._tokens[sig[s]].kind is GoTokenKind.COMMA:
                s += 1
        raise _ParseError("unterminated struct literal")


    def _parse_struct_field(self, s: int) -> tuple[tuple[str, ArgRef], int]:
        """Parse one ``name: value`` pair inside a struct literal."""

        sig = self._sig
        n_sig = len(sig)
        if s >= n_sig or self._tokens[sig[s]].kind is not GoTokenKind.IDENTIFIER:
            raise _ParseError("expected struct field name")
        name = self._tokens[sig[s]].text
        s += 1
        if s >= n_sig or self._tokens[sig[s]].kind is not GoTokenKind.COLON:
            raise _ParseError("expected colon after struct field name")
        s += 1
        value, s = self._parse_arg(s)
        return (name, value), s

    def _parse_arg(self, s: int) -> tuple[ArgRef, int]:
        """Parse a single argument or struct-field-value expression.

        Recognizes (in order):

        * regular and raw string literals → :class:`StringLitArg`;
        * numeric literals → :class:`NumberLitArg`;
        * ``&`` / ``*`` prefixed expressions that match a struct
          literal → :class:`StructLitArg` (only struct literals; a
          pointer-prefixed call is not a valid Go expression);
        * identifier-prefixed calls → :class:`CallArg`;
        * identifier-prefixed struct literals → :class:`StructLitArg`;
        * bare identifiers and dotted identifier paths →
          :class:`IdentArg` / :class:`DottedArg`;
        * any other expression shape →
          :class:`UnknownArg` (text recorded as a fallback).
        """

        sig = self._sig
        n_sig = len(sig)
        if s >= n_sig:
            raise _ParseError("expected argument")

        tok = self._tokens[sig[s]]
        kind = tok.kind

        if kind is GoTokenKind.STRING_LITERAL:
            return StringLitArg(value=_string_literal_value(tok.text)), s + 1
        if kind is GoTokenKind.RAW_STRING_LITERAL:
            return StringLitArg(value=_raw_string_literal_value(tok.text)), s + 1
        if kind is GoTokenKind.NUMBER_LITERAL:
            return NumberLitArg(text=tok.text), s + 1

        if kind in (GoTokenKind.AMPERSAND, GoTokenKind.STAR, GoTokenKind.IDENTIFIER):
            arg = self._try_parse_compound_arg(s)
            if arg is not None:
                return arg

        return self._parse_unknown_arg(s)


    def _try_parse_compound_arg(self, s: int) -> tuple[ArgRef, int] | None:
        """Try the call / struct-literal / dotted-identifier shapes.

        Returns ``None`` when the position does not start any of these
        shapes (so the caller can fall back to :meth:`_parse_unknown_arg`).
        """

        events, after = self._try_match_call_or_struct(s)
        if events is not None:
            event = events[0]
            if isinstance(event, MethodCallEvent):
                return CallArg(call=event), after
            if isinstance(event, StructLitEvent):
                return StructLitArg(event=event), after

        # No call or struct literal matched. Try a bare identifier or
        # dotted path. Pointer-prefixed bare identifiers (``*x``,
        # ``&x``) without a following ``(`` or ``{`` fall through to
        # the unknown-arg path because the recognizer doesn't model
        # arbitrary pointer expressions.
        sig = self._sig
        if self._tokens[sig[s]].kind is not GoTokenKind.IDENTIFIER:
            return None
        return self._parse_dotted_ident(s)

    def _parse_dotted_ident(self, s: int) -> tuple[ArgRef, int]:
        """Parse ``IDENT [. IDENT]*`` as a bare or dotted identifier argument."""

        sig = self._sig
        n_sig = len(sig)
        chain = [self._tokens[sig[s]].text]
        s += 1
        while s + 1 < n_sig and (
            self._tokens[sig[s]].kind is GoTokenKind.DOT
            and self._tokens[sig[s + 1]].kind is GoTokenKind.IDENTIFIER
        ):
            chain.append(self._tokens[sig[s + 1]].text)
            s += 2
        if len(chain) == 1:
            return IdentArg(name=chain[0]), s
        return DottedArg(parts=tuple(chain)), s


    def _parse_unknown_arg(self, s: int) -> tuple[ArgRef, int]:
        """Consume an opaque expression up to the next argument boundary.

        Stops at a ``,`` or any closing bracket at the current depth.
        Internal parens, brackets, and braces are tracked so that
        commas inside them do not terminate the expression. The raw
        source text is recorded so sub-analyzers can still inspect the
        slice if they need to (e.g. for malformed-input attribution).
        """

        sig = self._sig
        n_sig = len(sig)
        parts: list[str] = []
        depth = _DepthTracker()

        while s < n_sig:
            tok = self._tokens[sig[s]]
            kind = tok.kind
            if depth.is_zero():
                if kind is GoTokenKind.COMMA:
                    break
                if kind in (
                    GoTokenKind.RPAREN,
                    GoTokenKind.RBRACE,
                    GoTokenKind.RBRACKET,
                ):
                    break
            depth.update(kind)
            parts.append(tok.text)
            s += 1
        return UnknownArg(text=" ".join(parts)), s


    # --- Bracket-balance helpers -------------------------------------

    def _find_matching_paren(self, s: int) -> int:
        """Return the sig index of the ``)`` that balances ``(`` at ``s``.

        Returns ``-1`` when the parens are unbalanced. Nested ``(``/``)``
        are counted; ``{}`` and ``[]`` are treated as opaque (Go's
        grammar guarantees they balance separately within any
        well-formed paren group).
        """

        return self._find_matching(s, GoTokenKind.LPAREN, GoTokenKind.RPAREN)

    def _find_matching_brace(self, s: int) -> int:
        """Return the sig index of the ``}`` that balances ``{`` at ``s``."""

        return self._find_matching(s, GoTokenKind.LBRACE, GoTokenKind.RBRACE)

    def _find_matching_bracket(self, s: int) -> int:
        """Return the sig index of the ``]`` that balances ``[`` at ``s``."""

        return self._find_matching(s, GoTokenKind.LBRACKET, GoTokenKind.RBRACKET)

    def _find_matching(
        self,
        s: int,
        open_kind: GoTokenKind,
        close_kind: GoTokenKind,
    ) -> int:
        """Return the sig index of the matching closing bracket.

        Generic implementation backing the three thin wrappers above.
        """

        sig = self._sig
        n_sig = len(sig)
        if s >= n_sig or self._tokens[sig[s]].kind is not open_kind:
            return -1
        depth = 1
        i = s + 1
        while i < n_sig:
            kind = self._tokens[sig[i]].kind
            if kind is open_kind:
                depth += 1
            elif kind is close_kind:
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return -1


# --- Module-level helpers -----------------------------------------------------


class _DepthTracker:
    """Tracks combined depth across parens, braces, and brackets.

    The construct walker uses this to know whether the current
    significant token is at the file's top level (no open groups) or
    nested inside an expression. The tracker also clamps to zero on
    unbalanced closes so a malformed file never produces a negative
    depth that would mislabel later top-level constructs as nested.
    """

    __slots__ = ("_brace", "_bracket", "_paren")

    def __init__(self) -> None:
        self._paren: int = 0
        self._brace: int = 0
        self._bracket: int = 0

    def is_zero(self) -> bool:
        """Return ``True`` when no group is currently open."""

        return self._paren == 0 and self._brace == 0 and self._bracket == 0

    def update(self, kind: GoTokenKind) -> None:
        """Adjust depth in response to a single token kind."""

        if kind is GoTokenKind.LPAREN:
            self._paren += 1
        elif kind is GoTokenKind.RPAREN:
            self._paren = max(0, self._paren - 1)
        elif kind is GoTokenKind.LBRACE:
            self._brace += 1
        elif kind is GoTokenKind.RBRACE:
            self._brace = max(0, self._brace - 1)
        elif kind is GoTokenKind.LBRACKET:
            self._bracket += 1
        elif kind is GoTokenKind.RBRACKET:
            self._bracket = max(0, self._bracket - 1)


def _string_literal_value(text: str) -> str:
    """Return the unquoted contents of a regular Go string literal.

    The raw source text — including escape sequences — is preserved.
    For example, ``"foo\\n"`` becomes ``foo\\n`` (with a literal
    backslash-n), not ``foo<newline>``. Sub-analyzers compare against
    literal substrings (route paths, message destinations, schedule
    expressions) where escape sequences never appear in practice; the
    decision keeps the recognizer trivially deterministic.
    """

    if len(text) < _QUOTED_LITERAL_MIN_LEN:
        return text
    return text[1:-1]


def _raw_string_literal_value(text: str) -> str:
    """Return the contents of a raw string literal, stripping backticks.

    Raw string literals do not have escape sequences, so the contents
    are the exact source between the backticks.
    """

    if len(text) < _QUOTED_LITERAL_MIN_LEN:
        return text
    return text[1:-1]


def _extract_build_constraint_expression(text: str) -> str | None:
    """Return the expression portion of a build-constraint comment.

    Recognized comment shapes (handled in priority order):

    * ``//go:build <expr>`` — no whitespace between ``//`` and
      ``go:build``.
    * ``// +build <expr>`` — at least one whitespace character between
      ``//`` and ``+build``.

    Returns ``None`` when the comment text is some other comment kind.
    The tokenizer already classifies build-constraint comments at
    tokenization time, so this helper is only called on
    :attr:`GoTokenKind.BUILD_CONSTRAINT_COMMENT` tokens; the ``None``
    branch exists purely as a defensive check.
    """

    body = text[2:]  # strip leading ``//``
    if body.startswith("go:build"):
        rest = body[len("go:build") :]
        if not rest or rest[0] in (" ", "\t"):
            return rest
    stripped = body.lstrip(" \t")
    if stripped.startswith("+build"):
        rest = stripped[len("+build") :]
        if not rest or rest[0] in (" ", "\t"):
            return rest
    return None


def _strip_comment_bodies(comment_tokens: list[GoToken]) -> list[str]:
    """Strip comment markers and whitespace, returning non-empty bodies.

    For each comment token:

    * Line comments lose their leading ``//`` and surrounding whitespace.
    * Block comments lose their ``/*`` opener and ``*/`` closer; their
      internal whitespace is collapsed to single spaces so multi-line
      block comments yield a single normalized body string.

    Empty bodies are dropped so a doc-comment block that contains only
    a single horizontal-rule comment (``//``) does not produce an
    empty :class:`PackageDocCommentEvent.text`.
    """

    bodies: list[str] = []
    for tok in comment_tokens:
        if tok.kind is GoTokenKind.LINE_COMMENT:
            body = tok.text[2:].strip()
        elif tok.kind is GoTokenKind.BLOCK_COMMENT:
            inner = tok.text[2:-2] if len(tok.text) >= 4 else ""  # noqa: PLR2004
            body = " ".join(inner.split())
        else:
            continue
        if body:
            bodies.append(body)
    return bodies


# --- Type aliases -------------------------------------------------------------

# Top-level construct parsers all share the same shape: take a sig
# index, return ``(events, next_sig_index)``. Defining the alias
# inline keeps the dispatch table readable without forcing every
# parser to subclass a Protocol.
if TYPE_CHECKING:
    from collections.abc import Callable

    _TopLevelParser = Callable[[int], tuple[list[GoEvent], int]]

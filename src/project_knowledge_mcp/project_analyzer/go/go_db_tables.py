"""Go database table dependency detector — path-scoped grep heuristics.

The previous parser-based implementation walked the full Go event
stream and recognized three composite-literal shapes
(``pb.PoolExecuteQueryRequest``, ``model.PoolServiceRequest``, and
an in-house wrapper struct flowing into ``PoolExecuteQuery`` /
``Execute`` / ``ExecuteRaw``). That approach had two problems
against real operator data:

1. The Go tokenizer entered an infinite loop on a few malformed Go
   files in production repositories.
2. The composite-literal recognizer produced weak signal relative to
   the cost of parsing: most ``QueryString`` values in the operator's
   repos are bound to identifiers populated from constants, so the
   detector emitted nothing for the great majority of repository
   files.

This module replaces that pipeline with a path-scoped grep
heuristic. Every ESB microservice repo in scope keeps its SQL inside
``internal/repository/<name>_repo.go`` (or ``<name>_repos.go``). For
each such file the detector:

1. Extracts every Go string literal — double-quoted strings and
   backtick raw strings — from the file body.
2. Runs SQL-keyword regexes on the **string-literal contents only**
   (never on bare Go source), so Go identifiers in dot-chains like
   ``db.Pool.QueryContext`` or struct field references like
   ``user.Cache`` cannot be misread as table names.
3. Recognizes Oracle stored-procedure calls of the form
   ``BEGIN <SCHEMA>.<PROC>(...); END;`` or ``CALL <SCHEMA>.<PROC>(...)``
   alongside ordinary table references. Procedures share the
   ``DatabaseTableDependency`` shape — the schema-qualified name
   (e.g. ``APP_ESB_MICROSVC.SP_CAT_CAPTURE_REPAYMENTS``) makes them
   distinguishable from plain tables.
4. Applies an admission filter: identifiers must match
   ``[A-Z_][A-Z0-9_]*`` (Oracle's uppercase convention, used by the
   operator's data) and must not be in :data:`_STOP_WORDS_LOWER`.
   Schema-qualified names (``A.B``) require both parts to satisfy
   the same shape, and the length floor for unqualified names is
   3 characters.

Public contract (unchanged):

* ``detect_go_database_tables(repository_contents, events_by_file)``
  returns ``(detections, file_skip_messages)``. The
  ``events_by_file`` parameter is ignored (the aggregator passes it
  positionally to every Go scanner and we keep the parameter for
  signature compatibility).
* Source locations attach a ``(path, line=None)`` entry per
  detection — line is ``None`` because grep does not track line
  numbers through the extractor's output, and ``Source_Location.line``
  is documented as optional.
* The second element of the returned tuple is always ``[]``; the
  grep path has no per-file skip events to surface.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.project_analyzer.db_tables import (
    _aggregate,
    extract_table_references_preserving_schema,
)
from project_knowledge_mcp.project_analyzer.go.go_filter import is_go_source_file
from project_knowledge_mcp.models import (
    DatabaseAccessMode,
    SourceLocation,
)

if TYPE_CHECKING:
    from project_knowledge_mcp.models import (
        DatabaseTableDependency,
        RepositoryContents,
    )


__all__ = ["detect_go_database_tables"]


# ---------------------------------------------------------------------------
# Path scoping
# ---------------------------------------------------------------------------

#: Directory prefix that holds repository-pattern files in the
#: operator's microservice layout. Exactly one path segment is allowed
#: between this prefix and the filename — a file at
#: ``internal/repository/sub/foo_repo.go`` does NOT qualify.
_REPO_DIR_PREFIX: Final[str] = "internal/repository/"

#: Filename suffixes that identify a repository-pattern file in the
#: operator's layout. The plural form ``_repos.go`` is accepted in
#: addition to the singular ``_repo.go`` because a handful of the
#: in-scope repos use the plural naming.
_REPO_FILENAME_SUFFIXES: Final[tuple[str, ...]] = ("_repo.go", "_repos.go")


def _is_repository_pattern_file(path: str) -> bool:
    """Return ``True`` for ``internal/repository/<name>_repo[s].go``.

    The check enforces:

    * The path begins with ``internal/repository/``.
    * Exactly one segment sits between that prefix and the filename
      (no nested subdirectories under ``internal/repository/``).
    * The filename ends with ``_repo.go`` or ``_repos.go``.
    * The path passes :func:`is_go_source_file` — defensive against a
      vendored repo file at the same path shape; vendored paths are
      already excluded upstream by the aggregator, but the local
      double-check makes the module self-contained.

    Test-suffix files such as ``*_repo_test.go`` are intentionally
    excluded because their SQL is fixture material that should not
    appear in production dependency lists.
    """

    if not is_go_source_file(path):
        return False
    if not path.startswith(_REPO_DIR_PREFIX):
        return False
    remainder = path[len(_REPO_DIR_PREFIX):]
    if "/" in remainder:
        # A nested ``internal/repository/<sub>/<name>.go`` path does
        # not qualify under the operator's confirmed layout.
        return False
    if not any(remainder.endswith(suffix) for suffix in _REPO_FILENAME_SUFFIXES):
        return False
    # ``foo_repo_test.go`` ends with ``_test.go`` which already does
    # not match either suffix, so no separate check is needed.
    return True


# ---------------------------------------------------------------------------
# Go string-literal extraction
# ---------------------------------------------------------------------------
#
# Go has two string-literal kinds:
#
# * Double-quoted: ``"hello\n"`` — supports backslash escape sequences,
#   does not span lines (a newline inside without an escape is a
#   syntax error).
# * Backtick raw: ```hello`` — no escape processing, spans lines.
#
# The regex below captures both. It is intentionally permissive:
#
# * Allows an unterminated double-quoted string at EOF (the
#   ``(?:[^"\\\n]|\\.)*`` body never escapes a final newline, so a
#   stray ``"`` followed by EOF without closing simply produces no
#   match for that span — neighbouring literals are unaffected).
# * Backtick literals are matched greedily but bounded by the next
#   ``\``` so multiple backtick literals on one line stay separated.
#
# The regex deliberately does not exclude ``//``-comment context.
# Comments containing string literals are extremely rare in
# repository-pattern files (the operator's confirmed scope), and a
# comment like ``// "SELECT * FROM users"`` is a documented SQL
# pattern that the operator typically wants picked up anyway.

_RE_GO_STRING_LITERAL: Final[re.Pattern[str]] = re.compile(
    r'"(?:[^"\\\n]|\\.)*"'      # double-quoted with escapes
    r"|"
    r"`[^`]*`",                  # backtick raw (may span lines)
    re.DOTALL,
)


def _iter_go_string_literal_contents(file_text: str) -> list[str]:
    """Yield the inner content of every Go string literal in ``file_text``.

    Delimiters (``"`` or `` ` ``) are stripped. Escape sequences inside
    double-quoted strings are NOT processed — the SQL extractor only
    looks for ASCII keywords and identifier characters, so escapes
    like ``\\n``, ``\\t``, or ``\\"`` are irrelevant to the downstream
    regex matches. Leaving escapes intact also keeps the source-text
    offsets aligned with the original file, which simplifies any
    future feature that wants to report a line number.
    """
    return [
        match.group(0)[1:-1]
        for match in _RE_GO_STRING_LITERAL.finditer(file_text)
    ]


# ---------------------------------------------------------------------------
# Literal-level blacklist (operator-tuning)
# ---------------------------------------------------------------------------
#
# Some Go files under ``internal/repository/`` carry string literals
# that are neither SQL nor stored-procedure call text but happen to
# include keywords the SQL extractor would otherwise scan. The most
# common offender in the operator's repos is the GitLab default
# auto-close-issues description ("Automatically close issues from
# merge requests") which the GitLab Go SDK keeps as a literal
# constant; the trailing ``from merge`` substring trips the FROM and
# MERGE patterns in the SQL extractor.
#
# Rather than refine the per-keyword regexes to ignore this specific
# phrase, the detector skips any Go string literal whose content
# contains a blacklisted phrase verbatim. The blacklist is a small
# set of operator-confirmed strings; adding to it does not require
# re-tuning the SQL regexes.
_BLACKLISTED_LITERAL_PHRASES: Final[tuple[str, ...]] = (
    "Automatically close issues from merge requests",
)


def _is_blacklisted_literal(literal_content: str) -> bool:
    """Return ``True`` when a string literal carries a blacklisted phrase.

    The membership test is a substring match (not a full-string
    equality) so the phrase survives surrounding context — e.g. when
    the GitLab SDK uses the phrase as a prefix to a longer template
    string.
    """
    return any(
        phrase in literal_content
        for phrase in _BLACKLISTED_LITERAL_PHRASES
    )


# ---------------------------------------------------------------------------
# Go fmt schema-placeholder normalization
# ---------------------------------------------------------------------------
#
# Operator repos heavily use ``fmt.Sprintf`` to inject the runtime
# schema name into SQL templates:
#
#     fmt.Sprintf("SELECT ... FROM %v.IIB_PAYMENT_CONFIRMED ...", schema)
#     fmt.Sprintf("BEGIN %v.GETEARLYLOANS_REP(:1); END;", schema)
#     fmt.Sprintf("INSERT INTO %v.IIB_PAYMENT_CONFIRMED ...", schema)
#
# The Go grep detector cannot resolve what ``%v`` evaluates to at
# runtime, but the table or procedure name *after* the dot is
# statically present in the source. Rewriting ``%v.NAME`` to ``NAME``
# (and the same for ``%s.``, ``%d.``, positional ``%[1]v.``) lets the
# downstream regexes match unqualified ``NAME`` without expanding the
# regex set itself to accept dynamic-schema prefixes. The trailing
# dot is required so a bare value placeholder like ``WHERE id = %v``
# is left untouched (stripping it would merge adjacent tokens and
# create spurious matches).

_RE_FMT_SCHEMA_PREFIX: Final[re.Pattern[str]] = re.compile(
    # ``%v.``, ``%s.``, ``%d.``, ``%+v.``, ``%[1]v.``, ``%-2s.``, etc.
    # Conservative: only strip when the placeholder is immediately
    # followed by a dot AND the dot is immediately followed by an
    # identifier-start character. This keeps the substitution from
    # collapsing tokens like ``%v.\n`` or ``%v. (foo)`` that are not
    # schema prefixes.
    r"%[-+#0 ]?\[?\d*\]?[a-zA-Z]+\.(?=[A-Za-z_])",
)


def _strip_fmt_schema_prefixes(literal_content: str) -> str:
    """Rewrite ``%v.NAME`` (and friends) to ``NAME`` inside a SQL literal.

    See the module-level commentary on
    :data:`_RE_FMT_SCHEMA_PREFIX` for the full motivation: the
    operator's repos build SQL strings via ``fmt.Sprintf`` and inject
    the schema name with a ``%v`` placeholder, but the table /
    procedure name after the dot is statically known and the
    detector should pick it up.
    """
    return _RE_FMT_SCHEMA_PREFIX.sub("", literal_content)


# ---------------------------------------------------------------------------
# Stored-procedure call detection
# ---------------------------------------------------------------------------
#
# Oracle PL/SQL anonymous blocks invoke stored procedures with:
#
#     BEGIN <SCHEMA>.<PROC>(:p1, :p2, ...); END;
#
# Some repos also use the shorter ``CALL`` form (Postgres, SQL Server,
# Oracle 18c+):
#
#     CALL <SCHEMA>.<PROC>(:p1, :p2, ...);
#
# The regex below recognizes both. The schema prefix is optional; an
# unqualified ``PROC`` is also accepted because the admission filter
# (``_is_acceptable_identifier``) enforces the uppercase convention
# the operator's repos use for procedures.

_RE_BEGIN_PROC_CALL: Final[re.Pattern[str]] = re.compile(
    r"\bBEGIN\s+(?P<proc>[A-Z_][A-Z0-9_]*(?:\.[A-Z_][A-Z0-9_]*)?)\s*\(",
    re.IGNORECASE,
)

_RE_CALL_PROC_CALL: Final[re.Pattern[str]] = re.compile(
    r"\bCALL\s+(?P<proc>[A-Z_][A-Z0-9_]*(?:\.[A-Z_][A-Z0-9_]*)?)\s*\(",
    re.IGNORECASE,
)


def _iter_stored_procedure_calls(
    literal_content: str,
) -> list[str]:
    """Return procedure names called inside ``literal_content``.

    Recognizes ``BEGIN <proc>(...);`` and ``CALL <proc>(...);`` shapes.
    The returned names preserve case verbatim from the source so
    schema-qualified names like ``APP_ESB_MICROSVC.SP_CAT_CAPTURE_REPAYMENTS``
    round-trip exactly.
    """
    names: list[str] = []
    for pattern in (_RE_BEGIN_PROC_CALL, _RE_CALL_PROC_CALL):
        for match in pattern.finditer(literal_content):
            names.append(match.group("proc"))
    return names


# ---------------------------------------------------------------------------
# Identifier admission filter
# ---------------------------------------------------------------------------
#
# Even within string literals, the SQL-keyword regex matches plenty of
# false positives — single-character names (`a`, `b` from
# ``SELECT a.x FROM b a JOIN ...``), Go identifiers in interpolated
# strings (``SELECT * FROM `+tableName+` WHERE id = ?``), Oracle
# pseudo-tables (``DUAL``), and SQL keywords that show up by accident.
# The stop-word list and the uppercase-convention check below cut
# those out so the operator's table list is dominated by real
# ``APP_DOMINO.IIB_POS_SUBSCRIBER_LOG``-style identifiers.

#: Identifiers (case-insensitive) that are never valid table or
#: procedure names. Three categories:
#:
#: * SQL keywords that the regex shouldn't have matched (defence in
#:   depth — most are excluded by the keyword-prefixed regex anyway).
#: * Oracle pseudo-tables (``DUAL``) and PL/SQL flow keywords.
#: * Common Go identifiers observed as false positives in operator
#:   data: ``DB``, ``SP``, ``CACHE``, ``POOL``, ``FX``, ``REDIS``,
#:   ``AUTHORIZE``, ``RULEIDS``, ``SYSOBJ``, ``REQUESTORIDENTIFIER``.
_STOP_WORDS_LOWER: Final[frozenset[str]] = frozenset(
    {
        # SQL keywords (defence in depth)
        "all",
        "and",
        "any",
        "as",
        "asc",
        "between",
        "by",
        "case",
        "desc",
        "distinct",
        "else",
        "end",
        "exists",
        "false",
        "from",
        "full",
        "group",
        "having",
        "if",
        "in",
        "inner",
        "into",
        "is",
        "join",
        "left",
        "like",
        "merge",
        "not",
        "null",
        "on",
        "or",
        "order",
        "outer",
        "right",
        "select",
        "set",
        "table",
        "then",
        "true",
        "union",
        "update",
        "using",
        "values",
        "when",
        "where",
        # Oracle / PL-SQL keywords and pseudo-tables
        "begin",
        "cursor",
        "declare",
        "dual",
        "exception",
        "loop",
        "package",
        "procedure",
        "raise",
        "rowtype",
        "type",
        # Common Go identifiers observed as false positives in
        # operator data — adjust if a real table has the same name.
        "authorize",
        "cache",
        "db",
        "fx",
        "pool",
        "redis",
        "requestoridentifier",
        "ruleids",
        "sp",
        "sysobj",
    }
)

#: Matches the operator's table/procedure naming convention:
#: uppercase letters, digits, and underscores, starting with a letter
#: or underscore. ``A``, ``A_B``, ``ABC123``, ``T1`` all match;
#: ``a``, ``camelCase``, ``snake_lower`` do not.
_RE_UPPERCASE_IDENT: Final[re.Pattern[str]] = re.compile(r"^[A-Z_][A-Z0-9_]*$")

#: Case-insensitive identifier shape used by the stored-procedure
#: admission filter. SQL is case-insensitive, and some operator
#: teams declare procedures in lower- or mixed-case
#: (e.g. ``mulcastrans.sp_outstanding_amt_chk``). The ``BEGIN
#: <name>(...); END;`` and ``CALL <name>(...)`` call shapes are
#: specific enough that a relaxed admission does not meaningfully
#: increase the false-positive surface: ordinary ``SELECT … FROM``
#: matches still go through the stricter uppercase rule below.
_RE_PROCEDURE_IDENT: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$",
)

#: Minimum length for an unqualified identifier. Schema-qualified
#: names (``A.B``) skip this check.
_MIN_UNQUALIFIED_LENGTH: Final[int] = 3


def _is_acceptable_identifier(name: str) -> bool:
    """Return ``True`` when ``name`` looks like a real table or procedure.

    Admission rules:

    * Non-empty.
    * Not in the case-insensitive stop-word list (SQL keywords, Oracle
      pseudo-tables, common Go identifiers observed as false positives).
    * Every dot-separated part matches the uppercase convention
      ``[A-Z_][A-Z0-9_]*`` — the operator's repos uniformly use Oracle
      UPPERCASE table names, so lowercase or camelCase candidates are
      effectively guaranteed to be Go identifiers, struct fields, or
      column references rather than tables.
    * Unqualified names must be at least :data:`_MIN_UNQUALIFIED_LENGTH`
      characters long — ``a``, ``t``, ``b1`` are almost always SQL
      aliases, not table names. Schema-qualified names (``A.B``)
      bypass this length floor because two-letter schema or table
      names are not unusual in older Oracle databases.
    """
    if not name:
        return False
    parts = name.split(".")
    bare = parts[-1]
    if bare.lower() in _STOP_WORDS_LOWER:
        return False
    # Reject the schema name too if it lies in the stop-word list,
    # so a contrived ``SELECT.FROM`` literal cannot survive.
    for part in parts:
        if part.lower() in _STOP_WORDS_LOWER:
            return False
    # Length floor for unqualified names only.
    if len(parts) == 1 and len(bare) < _MIN_UNQUALIFIED_LENGTH:
        return False
    # Uppercase convention enforced on every dot-separated part.
    for part in parts:
        if not _RE_UPPERCASE_IDENT.match(part):
            return False
    return True


def _is_acceptable_procedure_name(name: str) -> bool:
    """Return ``True`` when ``name`` looks like a real stored-procedure call.

    Mirrors :func:`_is_acceptable_identifier` but relaxes the case
    convention so lower-case and mixed-case procedure names such as
    ``mulcastrans.sp_outstanding_amt_chk`` are accepted. The relaxed
    admission is sound because the call-site grammar (``BEGIN
    <name>(...); END;`` or ``CALL <name>(...)``) is much more
    specific than the SQL ``FROM <name>`` shape: a Go identifier in a
    string literal almost never lands inside one of these PL/SQL
    framings.

    Admission rules (delta vs. :func:`_is_acceptable_identifier`):

    * Stop-word and length-floor checks unchanged — ``sp`` and ``db``
      remain rejected, and an unqualified one- or two-character name
      is treated as a SQL alias rather than a procedure.
    * Each dot-separated part must match the case-insensitive shape
      :data:`_RE_PROCEDURE_IDENT` (``[A-Za-z_][A-Za-z0-9_]*``) so
      identifiers in any case round-trip with their source casing
      preserved verbatim.
    """
    if not name:
        return False
    parts = name.split(".")
    bare = parts[-1]
    if bare.lower() in _STOP_WORDS_LOWER:
        return False
    for part in parts:
        if part.lower() in _STOP_WORDS_LOWER:
            return False
    if len(parts) == 1 and len(bare) < _MIN_UNQUALIFIED_LENGTH:
        return False
    for part in parts:
        if not _RE_PROCEDURE_IDENT.match(part):
            return False
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_go_database_tables(
    repository_contents: RepositoryContents,
    events_by_file: object,
) -> tuple[list[DatabaseTableDependency], list[str]]:
    """Detect database table and procedure dependencies via path-scoped grep.

    Iterates ``repository_contents.files`` in path-sorted order so the
    output is a deterministic function of the input. For every file
    matching :func:`_is_repository_pattern_file`:

    1. Extract every Go string literal (double-quoted or backtick raw)
       from the file body.
    2. For each literal, run the schema-preserving SQL extractor
       (``extract_table_references_preserving_schema``) on the literal
       content alone, *not* on the surrounding Go source — this is
       what eliminates false positives from Go identifiers in
       dot-chains like ``db.Pool.QueryContext`` or struct fields like
       ``user.Cache``.
    3. Recognize Oracle ``BEGIN <SCHEMA>.<PROC>(...); END;`` and
       ``CALL <SCHEMA>.<PROC>(...)`` stored-procedure calls in the
       same literal scan. Procedures are emitted as
       :class:`DatabaseTableDependency` entries with
       ``access_mode=DatabaseAccessMode.UNKNOWN`` because the call
       site alone does not reveal whether the procedure body reads or
       writes the underlying tables.
    4. Apply :func:`_is_acceptable_identifier` to filter out SQL
       keywords, Oracle pseudo-tables (``DUAL``), single-character
       SQL aliases, common Go identifiers (``DB``, ``cache``, ``pool``,
       …), and any name that does not match the operator's
       ``[A-Z_][A-Z0-9_]*`` uppercase convention.

    Coalescing across files is delegated to the existing
    :func:`~project_knowledge_mcp.project_analyzer.db_tables._aggregate`,
    which unions source locations by ``(path, line)`` and applies the
    Requirement 9.6 read+write→read_write rule plus the Requirement
    9.8 UNKNOWN-as-lowest-priority rule.

    Args:
        repository_contents: The :class:`RepositoryContents` snapshot
            for the project. Iterated for repository-pattern files.
        events_by_file: Unused; the aggregator passes the Go event
            map positionally. Typed as :class:`object` because this
            module no longer participates in the Go parser pipeline.

    Returns:
        ``(detections, file_skip_messages)``. The skip list is always
        empty under the grep contract — no parser is invoked, so
        there is no per-file tokenizer failure to record.
    """

    # ``events_by_file`` is part of the established public signature
    # but no longer consulted (see module docstring).
    _ = events_by_file

    raw_detections: list[tuple[str, DatabaseAccessMode, SourceLocation]] = []

    for path in sorted(repository_contents.files):
        if not _is_repository_pattern_file(path):
            continue
        content = repository_contents.read_text(path)
        if content is None:
            continue
        loc = SourceLocation(path=path, line=None)

        for literal_content in _iter_go_string_literal_contents(content):
            # Operator-tuning blacklist: skip string literals that
            # carry a phrase known to produce false positives (e.g.
            # GitLab SDK boilerplate that includes a substring the
            # SQL extractor would otherwise pick up).
            if _is_blacklisted_literal(literal_content):
                continue

            # Normalize ``fmt.Sprintf`` schema placeholders such as
            # ``%v.IIB_PAYMENT_CONFIRMED`` to ``IIB_PAYMENT_CONFIRMED``
            # so the downstream SQL regex and stored-procedure regex
            # can match the statically-known name (see
            # ``_strip_fmt_schema_prefixes`` for the full rationale).
            normalized = _strip_fmt_schema_prefixes(literal_content)

            # Stored-procedure calls. Emit as UNKNOWN access mode
            # because the call shape does not reveal the SP body.
            # Procedure names use a case-relaxed admission filter so
            # lower- and mixed-case names like
            # ``mulcastrans.sp_outstanding_amt_chk`` are accepted —
            # the ``BEGIN ... END;`` / ``CALL ...`` grammar is
            # specific enough that the false-positive surface stays
            # tight.
            for proc_name in _iter_stored_procedure_calls(normalized):
                if not _is_acceptable_procedure_name(proc_name):
                    continue
                raw_detections.append((proc_name, DatabaseAccessMode.UNKNOWN, loc))

            # Ordinary table references inside the literal.
            for table_name, access_mode in extract_table_references_preserving_schema(
                normalized
            ):
                if not _is_acceptable_identifier(table_name):
                    continue
                raw_detections.append((table_name, access_mode, loc))

    detections = _aggregate(raw_detections)

    # The grep path has no SkipFileEvents to surface; the second
    # element is always empty.
    return detections, []

"""Database table dependency detector for ``Project_Analyzer``.

Detects database table references in a project's repository contents and
returns one :class:`DatabaseTableDependency` per distinct table name. The
detector covers four sources of evidence (Requirement 6.1):

1. **Raw SQL** matched by case-insensitive regular expressions:

   * ``SELECT ... FROM <table>``      → :data:`DatabaseAccessMode.READ`
   * ``... JOIN <table>``             → :data:`DatabaseAccessMode.READ`
   * ``INSERT INTO <table>``          → :data:`DatabaseAccessMode.WRITE`
   * ``UPDATE <table> SET ...``       → :data:`DatabaseAccessMode.WRITE`
   * ``DELETE FROM <table>``          → :data:`DatabaseAccessMode.WRITE`
   * ``CREATE TABLE <table>``         → :data:`DatabaseAccessMode.WRITE`
   * ``MERGE INTO <table>``           → :data:`DatabaseAccessMode.WRITE`

2. **ORM model declarations**, treated as writes because the model owns
   the table (the design's documented conservative approach):

   * SQLAlchemy: ``__tablename__ = "..."``
   * Django: ``class Meta: ... db_table = "..."``
   * ActiveRecord: ``self.table_name = "..."``

3. **Alembic migration ops**: ``op.create_table``, ``op.drop_table``,
   ``op.alter_table``, and ``op.rename_table`` (which yields write entries
   for both the old and the new name).

4. **ORM query methods** referencing a model class discovered in (2):

   * Read: ``<Class>.query``, ``session.query(<Class>)``,
     ``select(<Class>)``, and ``<Class>.objects.<read_method>``
     (Django QuerySet read methods like ``all``, ``filter``, ``get``).
   * Write: ``<Class>.objects.create``/``update``/``delete``/
     ``bulk_create``/``bulk_update``/``get_or_create``/``update_or_create``,
     and ``session.add(<Class>(...))``.

All detection sites for a given ``table_name`` are coalesced into a single
:class:`DatabaseTableDependency` (Requirement 6.3). When both
:data:`DatabaseAccessMode.READ` and :data:`DatabaseAccessMode.WRITE` are
observed for the same table, the entry's ``access_mode`` is set to
:data:`DatabaseAccessMode.READ_WRITE`. Source locations are deduplicated
while preserving discovery order. The returned list is empty (not ``None``)
when nothing was detected (Requirement 6.4).

The detector is a pure function of the ``RepositoryContents`` snapshot:
identical inputs always yield identical outputs, so two analyses of the
same commit produce the same dependency list.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.models import (
    DatabaseAccessMode,
    DatabaseTableDependency,
    SourceLocation,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from project_knowledge_mcp.models import RepositoryContents


# ---------------------------------------------------------------------------
# Identifier matching
# ---------------------------------------------------------------------------

# A SQL table identifier as it appears in source code. Supports:
#   * Backtick-quoted (``MySQL``):           ```users```
#   * Double-quoted (``ANSI``/``Postgres``):  ``"users"`` and ``"public"."users"``
#   * Bracketed (``SQL Server``):             ``[users]`` and ``[dbo].[users]``
#   * Single-quoted (some Alembic ops):       ``'users'``
#   * Bare alphanumeric identifier with an optional schema prefix.
#
# Schema-qualified forms are matched as a single token so that the table
# name is recovered cleanly by :func:`_clean_identifier` (which keeps only
# the last segment).
_TABLE_TOKEN: Final[str] = (
    r"(?P<table>"
    r"`[^`]+`(?:\.`[^`]+`)?"
    r'|"[^"]+"(?:\."[^"]+")?'
    r"|\[[^\]]+\](?:\.\[[^\]]+\])?"
    r"|'[^']+'(?:\.'[^']+')?"
    r"|[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)?"
    r")"
)

# Same-shape identifier used by Alembic op patterns where the table name is
# always a quoted Python string literal.
_QUOTED_PY_STRING: Final[str] = r'(?P<table>"[^"]+"|\'[^\']+\')'

# Raw-SQL patterns. Case-insensitive so that ``select … from``, ``Select …
# From``, and ``SELECT … FROM`` all match. ``re.IGNORECASE`` does not affect
# the captured table token, so identifiers are returned with their source
# casing intact and are normalized in :func:`_aggregate`.
_RE_FROM = re.compile(rf"\bFROM\s+{_TABLE_TOKEN}", re.IGNORECASE)
_RE_JOIN = re.compile(rf"\bJOIN\s+{_TABLE_TOKEN}", re.IGNORECASE)
_RE_INSERT = re.compile(rf"\bINSERT\s+INTO\s+{_TABLE_TOKEN}", re.IGNORECASE)
# ``UPDATE`` requires a trailing ``SET`` so that we do not match the
# SQLAlchemy ``update(<expression>)`` builder or English prose.
_RE_UPDATE = re.compile(rf"\bUPDATE\s+{_TABLE_TOKEN}\s+SET\b", re.IGNORECASE)
_RE_DELETE = re.compile(rf"\bDELETE\s+FROM\s+{_TABLE_TOKEN}", re.IGNORECASE)
_RE_CREATE_TABLE = re.compile(
    rf"\bCREATE\s+(?:TEMPORARY\s+|TEMP\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{_TABLE_TOKEN}",
    re.IGNORECASE,
)
# ``MERGE INTO <table>`` (Oracle/SQL Server upsert) is a write. Added for
# Go analyzer support Requirement 9.5, where the ``fec_pool_service``
# adapter forwards arbitrary SQL — including MERGE statements — into the
# raw-SQL extractor. The same regex shape is reused by the
# schema-preserving helper below.
_RE_MERGE = re.compile(rf"\bMERGE\s+INTO\s+{_TABLE_TOKEN}", re.IGNORECASE)

# Alembic migration ops. Quoted-string-only because Alembic's documented
# call shape always passes the table name as a literal string.
_RE_OP_CREATE_TABLE = re.compile(rf"\bop\.create_table\s*\(\s*{_QUOTED_PY_STRING}")
_RE_OP_DROP_TABLE = re.compile(rf"\bop\.drop_table\s*\(\s*{_QUOTED_PY_STRING}")
_RE_OP_ALTER_TABLE = re.compile(rf"\bop\.alter_table\s*\(\s*{_QUOTED_PY_STRING}")
_RE_OP_RENAME_TABLE = re.compile(
    r"\bop\.rename_table\s*\(\s*"
    r"(?P<old>\"[^\"]+\"|'[^']+')\s*,\s*"
    r"(?P<new>\"[^\"]+\"|'[^']+')"
)

# ORM model declarations.
_RE_SQLA_TABLENAME = re.compile(
    r'\b__tablename__\s*=\s*(?P<q>["\'])(?P<table>[^"\']+)(?P=q)'
)
_RE_DJANGO_DBTABLE = re.compile(
    r'\bdb_table\s*=\s*(?P<q>["\'])(?P<table>[^"\']+)(?P=q)'
)
_RE_RUBY_TABLENAME = re.compile(
    r'\bself\.table_name\s*=\s*(?P<q>["\'])(?P<table>[^"\']+)(?P=q)'
)

# Python ``class <Name>:`` definitions (used for owning-class lookup in
# Pass A so that ORM query patterns in Pass D can be tied back to the
# right table). Captures the leading indent so we can prefer the
# *outermost* class over the inner ``class Meta`` for Django's
# ``db_table``.
_RE_PY_CLASS_DEF = re.compile(
    r"^(?P<indent>[ \t]*)class\s+(?P<name>\w+)\b",
    re.MULTILINE,
)

# Python ``from <module> import ...`` lines. Filtered out before raw-SQL
# scanning so that ``from app import x`` is not interpreted as the SQL
# clause ``FROM app``. The case-insensitive raw-SQL regex would otherwise
# fire on every Python import statement in the repository.
_RE_PY_IMPORT_LINE = re.compile(r"^\s*from\s+[\w.]+\s+import\b")

# Django QuerySet method classifier. Methods not listed in either set are
# ignored (rather than guessed) so the detector errs on the side of fewer
# false positives.
_DJANGO_READ_METHODS: Final[frozenset[str]] = frozenset(
    {
        "aggregate",
        "all",
        "annotate",
        "count",
        "distinct",
        "earliest",
        "exclude",
        "exists",
        "filter",
        "first",
        "get",
        "in_bulk",
        "last",
        "latest",
        "none",
        "order_by",
        "prefetch_related",
        "raw",
        "select_related",
        "values",
        "values_list",
    }
)
_DJANGO_WRITE_METHODS: Final[frozenset[str]] = frozenset(
    {
        "bulk_create",
        "bulk_update",
        "create",
        "delete",
        "get_or_create",
        "update",
        "update_or_create",
    }
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _line_for_pos(content: str, pos: int) -> int:
    """Return the 1-indexed line number containing ``pos`` in ``content``."""

    return content.count("\n", 0, pos) + 1


def _clean_identifier(raw: str) -> str:
    """Strip quote/bracket characters and any schema prefix from a table token.

    Schema-qualified names (``schema.table``, ``"schema"."table"``,
    ``[dbo].[users]``, …) are reduced to the unqualified table name. Quote
    and bracket characters are removed wherever they appear so that the
    returned string is always a bare identifier suitable for use as the
    ``table_name`` of a :class:`DatabaseTableDependency`.
    """

    cleaned = raw.translate(str.maketrans("", "", "\"`[]'"))
    if "." in cleaned:
        cleaned = cleaned.rsplit(".", 1)[-1]
    return cleaned.strip()


def _clean_identifier_preserving_schema(raw: str) -> str:
    """Strip quote/bracket characters but preserve any ``<schema>.<table>`` form.

    Unlike :func:`_clean_identifier`, the schema prefix is not dropped: a
    raw match such as ``"public"."users"`` is normalized to
    ``public.users``, ``[dbo].[users]`` to ``dbo.users``, and a bare
    ``schema.table`` is returned unchanged. This is the form Requirement
    9.5 mandates for SQL extracted from Go ``QueryString`` literals,
    where downstream consumers must see the schema-qualified table name
    exactly as it appears in the source SQL.
    """

    cleaned = raw.translate(str.maketrans("", "", "\"`[]'"))
    return cleaned.strip()


def _is_python_import_line(content: str, pos: int) -> bool:
    """True when the line containing ``pos`` is a Python ``from … import`` line."""

    line_start = content.rfind("\n", 0, pos) + 1
    line_end = content.find("\n", pos)
    if line_end == -1:
        line_end = len(content)
    return bool(_RE_PY_IMPORT_LINE.match(content[line_start:line_end]))


# ---------------------------------------------------------------------------
# Pass A: ORM model declarations and class → table mapping
# ---------------------------------------------------------------------------


def _detect_orm_models(
    content: str,
) -> Iterator[tuple[str, str | None, int]]:
    """Yield ``(table_name, owning_class_name_or_None, line)`` for each ORM model.

    The owning class name is resolved when possible so that Pass D can tie
    ORM query patterns (e.g. ``User.query``) back to a specific table.

    * SQLAlchemy ``__tablename__`` is declared inside the model class, so
      the *immediately preceding* class definition is the owner.
    * Django ``db_table`` is declared inside ``class Meta``, which is
      itself nested inside the model class. The owner is the *outermost*
      class reachable from the ``db_table`` position, identified by the
      smallest indent seen while walking the preceding class definitions
      backward.
    * ActiveRecord declarations are reported with ``owner=None`` because
      the detector does not parse Ruby class hierarchies.
    """

    classes = [
        (m.start(), len(m.group("indent")), m.group("name"))
        for m in _RE_PY_CLASS_DEF.finditer(content)
    ]

    def _outer_class_for_pos(pos: int) -> str | None:
        outer_name: str | None = None
        min_indent: int | None = None
        for start, indent, name in reversed(classes):
            if start >= pos:
                continue
            if min_indent is None or indent < min_indent:
                min_indent = indent
                outer_name = name
                if indent == 0:
                    # No further class can be more outer than indent 0.
                    break
        return outer_name

    def _innermost_class_for_pos(pos: int) -> str | None:
        for start, _indent, name in reversed(classes):
            if start < pos:
                return name
        return None

    for m in _RE_SQLA_TABLENAME.finditer(content):
        yield (
            _clean_identifier(m.group("table")),
            _innermost_class_for_pos(m.start()),
            _line_for_pos(content, m.start()),
        )

    for m in _RE_DJANGO_DBTABLE.finditer(content):
        yield (
            _clean_identifier(m.group("table")),
            _outer_class_for_pos(m.start()),
            _line_for_pos(content, m.start()),
        )

    for m in _RE_RUBY_TABLENAME.finditer(content):
        yield (
            _clean_identifier(m.group("table")),
            None,
            _line_for_pos(content, m.start()),
        )


# ---------------------------------------------------------------------------
# Pass B: raw SQL
# ---------------------------------------------------------------------------


def _detect_raw_sql(
    content: str,
) -> Iterator[tuple[str, DatabaseAccessMode, int]]:
    """Yield ``(table_name, access_mode, line)`` for each raw-SQL match.

    Write patterns (``INSERT``, ``UPDATE``, ``DELETE``, ``CREATE TABLE``)
    are matched first and their character spans are recorded. The read
    patterns (``FROM``, ``JOIN``) then skip any match whose start falls
    inside a recorded write span, so that ``DELETE FROM users`` produces a
    single ``WRITE`` entry rather than spuriously also producing a stray
    ``READ`` entry from the trailing ``FROM users``.

    Lines that look like Python ``from <module> import …`` statements are
    excluded so that the case-insensitive ``\\bFROM`` regex does not fire
    on every Python import in the repository.
    """

    consumed: list[tuple[int, int]] = []
    write_patterns = (
        _RE_INSERT,
        _RE_UPDATE,
        _RE_DELETE,
        _RE_CREATE_TABLE,
        _RE_MERGE,
    )
    for pattern in write_patterns:
        for m in pattern.finditer(content):
            if _is_python_import_line(content, m.start()):
                continue
            consumed.append((m.start(), m.end()))
            yield (
                _clean_identifier(m.group("table")),
                DatabaseAccessMode.WRITE,
                _line_for_pos(content, m.start()),
            )

    for pattern in (_RE_FROM, _RE_JOIN):
        for m in pattern.finditer(content):
            if _is_python_import_line(content, m.start()):
                continue
            if any(s <= m.start() < e for s, e in consumed):
                continue
            yield (
                _clean_identifier(m.group("table")),
                DatabaseAccessMode.READ,
                _line_for_pos(content, m.start()),
            )


# ---------------------------------------------------------------------------
# Schema-preserving public helper (Requirement 9.5)
# ---------------------------------------------------------------------------


def extract_table_references_preserving_schema(
    sql_text: str,
) -> list[tuple[str, DatabaseAccessMode]]:
    """Extract ``(table_name, access_mode)`` pairs from a raw SQL string.

    Mirrors the regex set used by :func:`_detect_raw_sql` — ``SELECT …
    FROM`` and ``… JOIN`` for :data:`DatabaseAccessMode.READ`; ``INSERT
    INTO``, ``UPDATE … SET``, ``DELETE FROM``, ``CREATE TABLE``, and
    ``MERGE INTO`` for :data:`DatabaseAccessMode.WRITE` — but returns the
    raw match group with quote and bracket characters stripped while
    **preserving** any ``<schema>.<table>`` prefix exactly as it appears
    in the source SQL (Go analyzer support Requirement 9.5).

    The Python-import-line filter applied by :func:`_detect_raw_sql` is
    intentionally not applied here: this helper is invoked on SQL strings
    extracted from a Go ``QueryString`` field, not on entire source files,
    so there are no Python ``from … import`` statements to disambiguate.

    Returns an empty list when no table reference is matched. Order of
    the returned pairs follows the same pass order as
    :func:`_detect_raw_sql`: writes first (``INSERT``, ``UPDATE``,
    ``DELETE``, ``CREATE TABLE``, ``MERGE``), then reads (``FROM``,
    ``JOIN``), with read matches whose character spans fall inside a
    write match's span suppressed (so ``DELETE FROM users`` yields a
    single ``WRITE`` entry rather than ``WRITE`` + ``READ``).
    """

    results: list[tuple[str, DatabaseAccessMode]] = []
    consumed: list[tuple[int, int]] = []

    write_patterns = (
        _RE_INSERT,
        _RE_UPDATE,
        _RE_DELETE,
        _RE_CREATE_TABLE,
        _RE_MERGE,
    )
    for pattern in write_patterns:
        for m in pattern.finditer(sql_text):
            consumed.append((m.start(), m.end()))
            results.append(
                (
                    _clean_identifier_preserving_schema(m.group("table")),
                    DatabaseAccessMode.WRITE,
                )
            )

    for pattern in (_RE_FROM, _RE_JOIN):
        for m in pattern.finditer(sql_text):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            results.append(
                (
                    _clean_identifier_preserving_schema(m.group("table")),
                    DatabaseAccessMode.READ,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Pass C: Alembic migration ops
# ---------------------------------------------------------------------------


def _detect_migrations(
    content: str,
) -> Iterator[tuple[str, DatabaseAccessMode, int]]:
    """Yield write entries for Alembic ``op.create_table``/``drop_table``/etc."""

    for pattern in (_RE_OP_CREATE_TABLE, _RE_OP_DROP_TABLE, _RE_OP_ALTER_TABLE):
        for m in pattern.finditer(content):
            yield (
                _clean_identifier(m.group("table")),
                DatabaseAccessMode.WRITE,
                _line_for_pos(content, m.start()),
            )

    for m in _RE_OP_RENAME_TABLE.finditer(content):
        line = _line_for_pos(content, m.start())
        yield _clean_identifier(m.group("old")), DatabaseAccessMode.WRITE, line
        yield _clean_identifier(m.group("new")), DatabaseAccessMode.WRITE, line


# ---------------------------------------------------------------------------
# Pass D: ORM query methods referencing a known model class
# ---------------------------------------------------------------------------


def _detect_orm_queries(
    content: str,
    class_to_table: Mapping[str, str],
) -> Iterator[tuple[str, DatabaseAccessMode, int]]:
    """Yield ORM query detections for every model class discovered in Pass A."""

    if not class_to_table:
        return

    for class_name, table in class_to_table.items():
        escaped = re.escape(class_name)

        # SQLAlchemy / Flask-SQLAlchemy: ``<Class>.query`` (read).
        for m in re.finditer(rf"\b{escaped}\.query\b", content):
            yield table, DatabaseAccessMode.READ, _line_for_pos(content, m.start())

        # SQLAlchemy: ``session.query(<Class>)`` and ``select(<Class>)`` (read).
        for m in re.finditer(rf"\bsession\.query\(\s*{escaped}\b", content):
            yield table, DatabaseAccessMode.READ, _line_for_pos(content, m.start())
        for m in re.finditer(rf"\bselect\(\s*{escaped}\b", content):
            yield table, DatabaseAccessMode.READ, _line_for_pos(content, m.start())

        # SQLAlchemy: ``session.add(<Class>(…))`` (write). Requires the
        # constructor open-paren so that ``session.add(some_user)`` (where
        # the variable name is unrelated to the class) is not falsely
        # attributed to this class.
        for m in re.finditer(rf"\bsession\.add\(\s*{escaped}\s*\(", content):
            yield table, DatabaseAccessMode.WRITE, _line_for_pos(content, m.start())

        # Django ORM: ``<Class>.objects.<method>``. Methods classified as
        # read or write are emitted with the matching access mode; unknown
        # methods are ignored so we do not fabricate detections.
        for m in re.finditer(rf"\b{escaped}\.objects\.(\w+)", content):
            method = m.group(1)
            if method in _DJANGO_WRITE_METHODS:
                yield table, DatabaseAccessMode.WRITE, _line_for_pos(content, m.start())
            elif method in _DJANGO_READ_METHODS:
                yield table, DatabaseAccessMode.READ, _line_for_pos(content, m.start())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    detections: list[tuple[str, DatabaseAccessMode, SourceLocation]],
) -> list[DatabaseTableDependency]:
    """Aggregate raw detections into one :class:`DatabaseTableDependency` per table.

    All access-mode observations for a table are unioned. When both
    :data:`DatabaseAccessMode.READ` and :data:`DatabaseAccessMode.WRITE`
    appear for the same table — or if any single observation already
    carries :data:`DatabaseAccessMode.READ_WRITE` — the entry's
    ``access_mode`` is set to :data:`DatabaseAccessMode.READ_WRITE`
    (Requirement 6.3).

    :data:`DatabaseAccessMode.UNKNOWN` is treated as the lowest-priority
    observation (Go analyzer support Requirement 9.8): if any of
    :data:`READ`, :data:`WRITE`, or :data:`READ_WRITE` is also observed
    for the same table, the more specific mode wins; the entry's
    ``access_mode`` is set to :data:`UNKNOWN` only when ``UNKNOWN`` is
    the sole observation for that table.

    Source locations are deduplicated by the ``(path, line)`` key while
    preserving discovery order (the order in which detection passes ran
    across files sorted by path), so the resulting list is
    deterministic. Entries are returned sorted by ``table_name``.
    """

    modes: dict[str, set[DatabaseAccessMode]] = defaultdict(set)
    locations: dict[str, list[SourceLocation]] = defaultdict(list)
    seen_keys: dict[str, set[tuple[str, int | None]]] = defaultdict(set)

    for table_name, mode, loc in detections:
        if not table_name:
            # A degenerate empty identifier (after stripping quotes) cannot
            # be persisted in a DatabaseTableDependency, which requires a
            # non-empty ``table_name``. Drop it rather than fail the
            # whole analysis.
            continue
        modes[table_name].add(mode)
        key = (loc.path, loc.line)
        if key not in seen_keys[table_name]:
            seen_keys[table_name].add(key)
            locations[table_name].append(loc)

    result: list[DatabaseTableDependency] = []
    for table_name in sorted(modes):
        observed = modes[table_name]
        has_read = DatabaseAccessMode.READ in observed
        has_write = DatabaseAccessMode.WRITE in observed
        has_read_write = DatabaseAccessMode.READ_WRITE in observed
        has_unknown = DatabaseAccessMode.UNKNOWN in observed
        if has_read_write or (has_read and has_write):
            access_mode = DatabaseAccessMode.READ_WRITE
        elif has_write:
            # WRITE takes precedence over a co-observed UNKNOWN
            # (Requirement 9.8).
            access_mode = DatabaseAccessMode.WRITE
        elif has_read:
            # READ takes precedence over a co-observed UNKNOWN
            # (Requirement 9.8).
            access_mode = DatabaseAccessMode.READ
        elif has_unknown:
            # UNKNOWN is recorded only when it is the sole observation
            # for this table (Requirement 9.8).
            access_mode = DatabaseAccessMode.UNKNOWN
        else:
            # ``modes[table_name]`` is non-empty by construction (entries
            # are only inserted from the raw-detection loop above), so
            # this branch is unreachable. Default to READ for safety to
            # preserve the historical fallback behavior.
            access_mode = DatabaseAccessMode.READ
        result.append(
            DatabaseTableDependency(
                table_name=table_name,
                access_mode=access_mode,
                source_locations=locations[table_name],
            )
        )
    return result


# ---------------------------------------------------------------------------
# File-extension filter (skip documentation files)
# ---------------------------------------------------------------------------
#
# The raw-SQL extractor scans every text file in
# ``repository_contents.files`` for the case-insensitive
# ``\bFROM\s+<token>`` / ``\bINSERT\s+INTO\s+<token>`` / etc.
# regex set. Documentation files written in natural language
# routinely contain those keywords in prose ("the service reads
# FROM the orders table", "we INSERT INTO the audit log"), which
# the regex set will misread as table references whose
# ``<token>`` is whatever English word follows the keyword. The
# operator's GitLab instance also seeds every new project with a
# README template that contains the phrase "Automatically close
# issues from merge requests", surfacing ``merge`` as a spurious
# table name for any project that has not yet committed any code.
#
# Skipping documentation extensions at the source closes both
# classes of false positive without affecting the four detection
# passes (raw SQL, ORM models, Alembic migrations, ORM queries):
# none of the ORM / migration patterns can be expressed in
# Markdown or reStructuredText anyway.


#: File-extension suffixes treated as documentation rather than
#: source code. The check is case-insensitive in
#: :func:`_is_documentation_file`.
_DOCUMENTATION_FILE_SUFFIXES: Final[tuple[str, ...]] = (
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".adoc",
)


def _is_documentation_file(path: str) -> bool:
    """Return ``True`` when ``path`` looks like a documentation file.

    Suffix-based and case-insensitive so ``README.md``,
    ``readme.MD``, ``docs/INSTALL.txt``, and ``Readme.markdown``
    all match. Files without any of the recognized suffixes (Go
    source, Python source, JSON manifests, ...) are not
    documentation and contribute to the SQL scan as before.
    """
    lowered = path.lower()
    return any(
        lowered.endswith(suffix)
        for suffix in _DOCUMENTATION_FILE_SUFFIXES
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_database_tables(
    repository_contents: RepositoryContents,
) -> list[DatabaseTableDependency]:
    """Detect database table dependencies across an entire repository snapshot.

    Implements Requirements 6.1, 6.2, 6.3, and 6.4. The returned list
    contains at most one entry per ``table_name``; mixed read/write access
    on the same table is coalesced to
    :data:`DatabaseAccessMode.READ_WRITE`; and an empty list is returned
    (rather than ``None``) when no detections are found.

    The function is deterministic: files are processed in path-sorted
    order, and detections within a file are emitted in pass order
    (model declarations, then raw SQL, then migration ops, then ORM
    query patterns).
    """

    detections: list[tuple[str, DatabaseAccessMode, SourceLocation]] = []
    class_to_table: dict[str, str] = {}

    files_sorted = sorted(
        (path, content)
        for path, content in repository_contents.files.items()
        if not _is_documentation_file(path)
    )

    # Pass A: ORM model declarations and class→table map.
    for path, content in files_sorted:
        for table_name, owner, line in _detect_orm_models(content):
            detections.append(
                (
                    table_name,
                    DatabaseAccessMode.WRITE,
                    SourceLocation(path=path, line=line),
                )
            )
            if owner is not None and table_name:
                # Last writer wins on collisions, but in practice each
                # class name maps to at most one table_name in well-formed
                # source code.
                class_to_table[owner] = table_name

    # Pass B + Pass C: raw SQL and Alembic migration ops.
    for path, content in files_sorted:
        for table_name, mode, line in _detect_raw_sql(content):
            detections.append(
                (table_name, mode, SourceLocation(path=path, line=line))
            )
        for table_name, mode, line in _detect_migrations(content):
            detections.append(
                (table_name, mode, SourceLocation(path=path, line=line))
            )

    # Pass D: ORM query patterns (depends on class_to_table from Pass A).
    for path, content in files_sorted:
        for table_name, mode, line in _detect_orm_queries(content, class_to_table):
            detections.append(
                (table_name, mode, SourceLocation(path=path, line=line))
            )

    return _aggregate(detections)


__all__ = [
    "detect_database_tables",
    "extract_table_references_preserving_schema",
]

# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 8: For all repositories where a single table is accessed from multiple source locations with both read and write access modes, the produced database_table_dependencies entry for that table SHALL have access_mode == "read_write".
"""Property test for mixed-mode access yielding ``read_write``.

**Validates Requirement 6.3** (Property 8 in the design).

For every synthetic repository in which a fixed set of distinct table
names is each touched by *at least one* read site (``SELECT ... FROM
<table>``) AND *at least one* write site (``INSERT INTO <table>``,
``UPDATE <table> SET ...``, or ``DELETE FROM <table>``), the
``DatabaseTableDependency`` produced by ``detect_database_tables`` for
each such table must have ``access_mode == DatabaseAccessMode.READ_WRITE``.

The strategy generates 1-3 distinct lowercase table names. For each
table it places the read sites and the write sites in separate files
under non-skipped paths (``src/...``, ``lib/...``) so that every
detection site is independently visible to the analyzer. Hypothesis is
configured with ``@settings(max_examples=100)`` per the design's
property-test convention.

The property is unaffected by Go-analyzer-support Requirement 9.8's
addition of :data:`DatabaseAccessMode.UNKNOWN` to the enum: ``UNKNOWN``
is treated as the lowest-priority observation by ``db_tables._aggregate``
(see :func:`_aggregate` and the parent design's "UNKNOWN coalescing"
section), so even if a co-observed ``UNKNOWN`` site were generated for
the same table, the read+write→read_write rule would still take
precedence. The strategy below does not synthesize ``UNKNOWN`` sites
because the language-agnostic regex extractor (``_detect_raw_sql``) only
ever emits ``READ`` or ``WRITE`` from the recognized SQL keywords; the
``UNKNOWN`` observation path is exercised end-to-end by the unit tests
in ``tests/unit/test_db_tables_detector.py`` (the ``test_aggregate_unknown_*``
suite) and by the Go detector's property test (Property 11).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import DatabaseAccessMode, RepositoryContents
from project_knowledge_mcp.project_analyzer.db_tables import detect_database_tables

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Reserved SQL keywords (and a couple of sub-clauses) that we exclude from
# generated table names so that no synthesized statement is degenerate
# under the detector's regex grammar (e.g. a table literally named
# ``"set"`` would still parse correctly, but excluding these keeps the
# generated SQL human-readable when Hypothesis prints a counterexample).
_SQL_KEYWORDS: frozenset[str] = frozenset(
    {
        "alter",
        "and",
        "as",
        "create",
        "delete",
        "drop",
        "exists",
        "from",
        "if",
        "insert",
        "into",
        "join",
        "key",
        "not",
        "null",
        "on",
        "or",
        "primary",
        "rename",
        "select",
        "set",
        "table",
        "update",
        "values",
        "where",
    }
)

# A generated identifier: lowercase ``a-z`` only, length 3-10, never a
# reserved SQL keyword. Lowercase keeps the canonical table_name stable
# under the detector's case-insensitive matching.
_table_name = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=3,
    max_size=10,
).filter(lambda s: s not in _SQL_KEYWORDS)

# Read- and write-statement templates. ``{t}`` is substituted with the
# generated table name. The detector's documented mappings:
#   * ``SELECT ... FROM <t>`` and ``... JOIN <t>``  -> READ
#   * ``INSERT INTO <t>``, ``UPDATE <t> SET ...``,
#     ``DELETE FROM <t>``                            -> WRITE
_READ_TEMPLATES: tuple[str, ...] = (
    "SELECT * FROM {t};",
    "SELECT id FROM {t} WHERE id > 0;",
    "SELECT a.id FROM other_tbl a JOIN {t} b ON a.id = b.id;",
)
_WRITE_TEMPLATES: tuple[str, ...] = (
    "INSERT INTO {t} (id) VALUES (1);",
    "UPDATE {t} SET status = 'x' WHERE id = 1;",
    "DELETE FROM {t} WHERE id = 1;",
)


@st.composite
def _mixed_mode_repositories(
    draw: st.DrawFn,
) -> tuple[RepositoryContents, frozenset[str]]:
    """Build a repository where every chosen table has read + write sites.

    Returns the ``RepositoryContents`` and the set of table names for
    which both at least one read site and at least one write site were
    placed in separate files. The asserted property holds for exactly
    this set.
    """

    table_names: list[str] = draw(
        st.lists(_table_name, min_size=1, max_size=3, unique=True)
    )

    files: dict[str, str] = {}
    mixed_mode: set[str] = set()

    for idx, table in enumerate(table_names):
        n_reads = draw(st.integers(min_value=1, max_value=3))
        n_writes = draw(st.integers(min_value=1, max_value=3))

        for r in range(n_reads):
            template = draw(st.sampled_from(_READ_TEMPLATES))
            # Distinct file path per (table, role, index) so the analyzer
            # sees independent source locations for every site.
            path = f"src/read_{idx}_{r}_{table}.sql"
            files[path] = template.format(t=table)

        for w in range(n_writes):
            template = draw(st.sampled_from(_WRITE_TEMPLATES))
            path = f"lib/write_{idx}_{w}_{table}.sql"
            files[path] = template.format(t=table)

        # By construction each table gets >= 1 read site and >= 1 write
        # site, so it must end up classified as read_write.
        mixed_mode.add(table)

    repository = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="cafef00d",
        files=files,
    )
    return repository, frozenset(mixed_mode)


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(case=_mixed_mode_repositories())
@settings(max_examples=100)
def test_mixed_read_and_write_access_yields_read_write(
    case: tuple[RepositoryContents, frozenset[str]],
) -> None:
    """Property 8: tables with both read and write sites get ``read_write``."""

    repository, expected_mixed = case
    deps = detect_database_tables(repository)

    by_name = {d.table_name: d for d in deps}

    # Requirement 6.3 (and the ``ProjectProfile`` invariant): at most one
    # entry per table_name. The detector returns a list, so check via the
    # cardinality of the by-name mapping vs. the list length.
    assert len(by_name) == len(deps), (
        "detector returned duplicate table_name entries: "
        f"{[d.table_name for d in deps]}"
    )

    for table in expected_mixed:
        assert table in by_name, (
            f"table {table!r} was injected with both read and write sites "
            f"but no DatabaseTableDependency was produced; got: "
            f"{sorted(by_name)}"
        )
        dep = by_name[table]
        assert dep.access_mode == DatabaseAccessMode.READ_WRITE, (
            f"table {table!r} had both read and write sites but the "
            f"produced access_mode is {dep.access_mode!r} (expected "
            f"{DatabaseAccessMode.READ_WRITE!r})"
        )

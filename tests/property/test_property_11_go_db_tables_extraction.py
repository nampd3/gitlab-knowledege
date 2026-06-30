# ruff: noqa: E501
# Feature: go-analyzer-support, Property 11 (rephrased): DB table extraction
# processes only ``internal/repository/*_repo[s].go`` files, preserves
# ``<schema>.<table>``, classifies access mode, and coalesces by name.
"""Property test for the Go database-table detector.

The detector is now a path-scoped grep heuristic. Files matching
``internal/repository/<name>_repo.go`` (or ``_repos.go``) are handed
to :func:`db_tables.extract_table_references_preserving_schema`; raw
results are aggregated by :func:`db_tables._aggregate`.

This property exercises the contract from the
``RepositoryContents`` boundary forward:

* Only files matching the strict
  ``internal/repository/<name>_repo[s].go`` glob contribute.
* Schema-qualified table names round-trip verbatim.
* ``READ`` + ``WRITE`` observations on the same table coalesce
  to ``READ_WRITE`` regardless of file/observation order.
* ``UNKNOWN`` is the lowest-priority observation: any more-specific
  mode wins when co-observed.
* Each emitted entry's source locations are ``(path, line=None)``
  pairs pointing at the contributing repository files.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Final, Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import (
    DatabaseAccessMode,
    DatabaseTableDependency,
    RepositoryContents,
    SourceLocation,
)
from project_knowledge_mcp.project_analyzer.go.go_db_tables import (
    _STOP_WORDS_LOWER,
    detect_go_database_tables,
)


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets and recognition constants
# ---------------------------------------------------------------------------

_IDENT_CHARS: Final[str] = string.ascii_uppercase

#: Tokens that must not appear as generated table names because they
#: would collide with the SQL templates' fixed keywords.
_SQL_KEYWORDS: Final[frozenset[str]] = frozenset(
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
        "lock",
        "merge",
        "not",
        "null",
        "on",
        "or",
        "primary",
        "rename",
        "select",
        "set",
        "table",
        "temporary",
        "truncate",
        "update",
        "values",
        "where",
    }
)

_READ_TEMPLATES: Final[tuple[str, ...]] = (
    "SELECT id FROM {t} WHERE id = :1",
    "SELECT a.id, a.name FROM {t} a WHERE a.status = :1",
    "SELECT x.id FROM other_tbl x JOIN {t} y ON x.id = y.id",
)
_WRITE_TEMPLATES: Final[tuple[str, ...]] = (
    "INSERT INTO {t} (id, status) VALUES (:1, :2)",
    "UPDATE {t} SET status = :1 WHERE id = :2",
    "DELETE FROM {t} WHERE id = :1",
    "MERGE INTO {t} USING dual ON (id = :1) WHEN MATCHED THEN UPDATE SET status = :2",
)


_TargetMode = Literal["read", "write", "read_write"]
_TARGET_MODES: Final[tuple[_TargetMode, ...]] = ("read", "write", "read_write")


# ---------------------------------------------------------------------------
# Observation record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Observation:
    """One planned SQL observation against one repository file.

    Each observation owns a distinct ``file_path`` so the detector's
    per-file iteration produces a deterministic source-location set.
    """

    table_name: str
    observed_mode: DatabaseAccessMode
    sql_text: str
    file_path: str


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


_bare_table_name = st.text(
    alphabet=_IDENT_CHARS, min_size=3, max_size=8,
).filter(lambda s: s.lower() not in _SQL_KEYWORDS and s.lower() not in _STOP_WORDS_LOWER)


@st.composite
def _table_name(draw: st.DrawFn) -> str:
    base = draw(_bare_table_name)
    qualify = draw(st.booleans())
    if not qualify:
        return base
    schema = draw(_bare_table_name)
    if schema == base:
        # Disambiguate with a digit so the suffix still satisfies the
        # uppercase-with-digits admission rule. Appending a lowercase
        # ``s`` would produce ``AAAs.AAA`` which the new
        # :func:`_is_acceptable_identifier` rejects on the schema part.
        schema = f"{schema}1"
    return f"{schema}.{base}"


def _sql_for_mode(
    mode: DatabaseAccessMode,
    table_name: str,
    template_index: int,
) -> str:
    if mode is DatabaseAccessMode.READ:
        template = _READ_TEMPLATES[template_index % len(_READ_TEMPLATES)]
    elif mode is DatabaseAccessMode.WRITE:
        template = _WRITE_TEMPLATES[template_index % len(_WRITE_TEMPLATES)]
    else:
        msg = f"_sql_for_mode does not synthesize SQL for {mode!r}"
        raise ValueError(msg)
    return template.format(t=table_name)


@st.composite
def _observations_for_table(
    draw: st.DrawFn,
    table_name: str,
    target_mode: _TargetMode,
) -> list[tuple[DatabaseAccessMode, str]]:
    """Plan the per-table observations leading to ``target_mode``."""

    plan_modes: list[DatabaseAccessMode] = []
    if target_mode == "read":
        n_reads = draw(st.integers(min_value=1, max_value=2))
        plan_modes.extend([DatabaseAccessMode.READ] * n_reads)
    elif target_mode == "write":
        n_writes = draw(st.integers(min_value=1, max_value=2))
        plan_modes.extend([DatabaseAccessMode.WRITE] * n_writes)
    else:  # "read_write"
        n_reads = draw(st.integers(min_value=1, max_value=2))
        n_writes = draw(st.integers(min_value=1, max_value=2))
        plan_modes.extend([DatabaseAccessMode.READ] * n_reads)
        plan_modes.extend([DatabaseAccessMode.WRITE] * n_writes)

    observations: list[tuple[DatabaseAccessMode, str]] = []
    for i, mode in enumerate(plan_modes):
        sql_text = _sql_for_mode(mode, table_name, template_index=i)
        observations.append((mode, sql_text))
    return observations


@st.composite
def _observation_plan(
    draw: st.DrawFn,
) -> tuple[list[_Observation], dict[str, DatabaseAccessMode]]:
    """Generate the full per-fixture observation plan."""

    tables: list[str] = draw(
        st.lists(_table_name(), min_size=1, max_size=4, unique=True),
    )

    expected_modes: dict[str, DatabaseAccessMode] = {}
    flat_plan: list[tuple[str, DatabaseAccessMode, str]] = []

    for table_name in tables:
        target_mode = draw(st.sampled_from(_TARGET_MODES))
        for mode, sql_text in draw(_observations_for_table(table_name, target_mode)):
            flat_plan.append((table_name, mode, sql_text))

        observed = {m for _t, m, _s in flat_plan if _t == table_name}
        if DatabaseAccessMode.READ in observed and DatabaseAccessMode.WRITE in observed:
            expected_modes[table_name] = DatabaseAccessMode.READ_WRITE
        elif DatabaseAccessMode.WRITE in observed:
            expected_modes[table_name] = DatabaseAccessMode.WRITE
        else:
            expected_modes[table_name] = DatabaseAccessMode.READ

    permuted_indices = draw(st.permutations(range(len(flat_plan))))
    observations: list[_Observation] = []
    for i, idx in enumerate(permuted_indices):
        table_name, mode, sql_text = flat_plan[idx]
        observations.append(
            _Observation(
                table_name=table_name,
                observed_mode=mode,
                sql_text=sql_text,
                file_path=f"internal/repository/obs_{i:03d}_repo.go",
            ),
        )

    return observations, expected_modes


# ---------------------------------------------------------------------------
# Fixture construction: RepositoryContents from observation plan
# ---------------------------------------------------------------------------


def _build_repo_contents(observations: list[_Observation]) -> RepositoryContents:
    files: dict[str, str] = {}
    for obs in observations:
        files[obs.file_path] = (
            "package repository\n"
            f'const q = "{obs.sql_text}"\n'
        )
    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )


def _expected_dependencies(
    observations: list[_Observation],
    expected_modes: dict[str, DatabaseAccessMode],
) -> list[DatabaseTableDependency]:
    locations_by_table: dict[str, list[SourceLocation]] = {}
    for obs in sorted(observations, key=lambda o: o.file_path):
        locations_by_table.setdefault(obs.table_name, []).append(
            SourceLocation(path=obs.file_path, line=None),
        )

    return [
        DatabaseTableDependency(
            table_name=table_name,
            access_mode=expected_modes[table_name],
            source_locations=locations_by_table[table_name],
        )
        for table_name in sorted(locations_by_table)
    ]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(_observation_plan())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_detect_matches_predicted_aggregation(
    case: tuple[list[_Observation], dict[str, DatabaseAccessMode]],
) -> None:
    """Detector output equals the predicted aggregation.

    Validates:
      * Path-scoped iteration (only repository files contribute).
      * Schema-preserving table names.
      * Read+write coalescing into ``READ_WRITE`` order-independently.
    """

    observations, expected_modes = case
    repo = _build_repo_contents(observations)

    actual, skips = detect_go_database_tables(repo, {})
    expected = _expected_dependencies(observations, expected_modes)

    assert actual == expected
    assert skips == []


@given(_observation_plan())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_non_repo_pattern_files_never_contribute(
    case: tuple[list[_Observation], dict[str, DatabaseAccessMode]],
) -> None:
    """Adding non-matching files with SQL must not change the output."""

    observations, expected_modes = case
    repo = _build_repo_contents(observations)
    expected_actual, _ = detect_go_database_tables(repo, {})

    # Build a polluted repo with extra files at non-matching paths that
    # contain SQL the detector would otherwise notice.
    polluted_files = dict(repo.files)
    polluted_files["cmd/main.go"] = 'package main\nconst _ = "SELECT id FROM rogue_a"\n'
    polluted_files["scripts/migrate.sql"] = "INSERT INTO rogue_b VALUES (1);\n"
    polluted_files["internal/repository/sub/foo_repo.go"] = (
        'package repository\nconst _ = "INSERT INTO rogue_c VALUES (:1)"\n'
    )
    polluted_files["internal/repository/foo_repo_test.go"] = (
        'package repository\nconst _ = "INSERT INTO rogue_d VALUES (:1)"\n'
    )
    polluted = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=polluted_files,
    )
    polluted_actual, _ = detect_go_database_tables(polluted, {})

    assert polluted_actual == expected_actual


@given(_observation_plan())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_vendored_repo_files_never_contribute(
    case: tuple[list[_Observation], dict[str, DatabaseAccessMode]],
) -> None:
    """A vendored ``.go`` at the same path shape must not leak in."""

    observations, _ = case
    repo = _build_repo_contents(observations)
    baseline, _ = detect_go_database_tables(repo, {})

    polluted_files = dict(repo.files)
    polluted_files[
        "vendor/example.com/lib/internal/repository/foo_repo.go"
    ] = 'package lib\nconst _ = "SELECT id FROM vendored_table"\n'
    polluted = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=polluted_files,
    )
    polluted_actual, _ = detect_go_database_tables(polluted, {})

    assert polluted_actual == baseline

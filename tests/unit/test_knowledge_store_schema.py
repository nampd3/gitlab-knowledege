"""Smoke tests for the ``Knowledge_Store`` schema bootstrap (task 3.1).

These tests verify that ``KnowledgeStore.open()`` brings the SQLite
database up in WAL mode with the four tables required by the design's
*Conceptual Schema* (``snapshots``, ``project_profiles``,
``ingestion_skips``, ``current_snapshot``) and that ``close()`` is safe
to call multiple times. Writer / reader interface tests live alongside
tasks 3.2 and 3.3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from project_knowledge_mcp.knowledge_store import KnowledgeStore

if TYPE_CHECKING:
    from pathlib import Path


# Tables the design's Conceptual Schema requires (Requirement 7).
EXPECTED_TABLES = {
    "snapshots",
    "project_profiles",
    "ingestion_skips",
    "current_snapshot",
}


@pytest.mark.unit
def test_open_creates_required_tables(tmp_path: Path) -> None:
    """``open()`` creates exactly the four tables the design names."""

    db_path = tmp_path / "knowledge.db"
    store = KnowledgeStore.open(db_path)
    try:
        rows = store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        table_names = {row[0] for row in rows}
        # Every required table is present. Additional sqlite-internal
        # tables (e.g. ``sqlite_sequence``) may exist; we only require
        # that ours are a subset.
        assert EXPECTED_TABLES.issubset(table_names)
    finally:
        store.close()


@pytest.mark.unit
def test_open_enables_wal_mode(tmp_path: Path) -> None:
    """The connection comes back in WAL journal mode (per task 3.1)."""

    store = KnowledgeStore.open(tmp_path / "wal.db")
    try:
        row = store.connection.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert str(row[0]).lower() == "wal"
    finally:
        store.close()


@pytest.mark.unit
def test_open_enables_foreign_keys(tmp_path: Path) -> None:
    """Foreign-key enforcement is on so the FK declarations actually fire."""

    store = KnowledgeStore.open(tmp_path / "fk.db")
    try:
        row = store.connection.execute("PRAGMA foreign_keys").fetchone()
        assert row is not None
        assert int(row[0]) == 1
    finally:
        store.close()


@pytest.mark.unit
def test_current_snapshot_seed_row_exists(tmp_path: Path) -> None:
    """``current_snapshot`` has its singleton row seeded with NULL pointer.

    Readers introduced in task 3.3 will treat ``snapshot_id IS NULL`` as
    "no committed snapshot yet", so the seed row must exist on a fresh
    store.
    """

    store = KnowledgeStore.open(tmp_path / "seed.db")
    try:
        rows = store.connection.execute(
            "SELECT id, snapshot_id FROM current_snapshot"
        ).fetchall()
        assert rows == [(1, None)]
    finally:
        store.close()


@pytest.mark.unit
def test_project_profiles_columns_match_design(tmp_path: Path) -> None:
    """``project_profiles`` exposes the column set named in the design."""

    store = KnowledgeStore.open(tmp_path / "cols.db")
    try:
        info = store.connection.execute(
            "PRAGMA table_info(project_profiles)"
        ).fetchall()
        # PRAGMA table_info returns (cid, name, type, notnull, dflt, pk).
        column_names = {row[1] for row in info}
        assert column_names == {
            "snapshot_id",
            "gitlab_project_id",
            "full_path",
            "analysis_branch",
            "analysis_branch_sha",
            "produced_at",
            "profile_json",
        }
    finally:
        store.close()


@pytest.mark.unit
def test_snapshots_columns_match_design(tmp_path: Path) -> None:
    """``snapshots`` exposes the column set named in the design."""

    store = KnowledgeStore.open(tmp_path / "snap.db")
    try:
        info = store.connection.execute("PRAGMA table_info(snapshots)").fetchall()
        column_names = {row[1] for row in info}
        assert column_names == {
            "snapshot_id",
            "started_at",
            "completed_at",
            "status",
            "trigger",
            "parent_snapshot_id",
        }
    finally:
        store.close()


@pytest.mark.unit
def test_ingestion_skips_columns_match_design(tmp_path: Path) -> None:
    """``ingestion_skips`` exposes the column set named in the design."""

    store = KnowledgeStore.open(tmp_path / "skips.db")
    try:
        info = store.connection.execute(
            "PRAGMA table_info(ingestion_skips)"
        ).fetchall()
        column_names = {row[1] for row in info}
        assert column_names == {
            "snapshot_id",
            "gitlab_project_id",
            "reason",
            "detail",
        }
    finally:
        store.close()


@pytest.mark.unit
def test_open_is_idempotent_against_existing_database(tmp_path: Path) -> None:
    """Re-opening an existing database preserves data and does not error.

    The schema statements all use ``IF NOT EXISTS`` so calling
    ``open()`` against a database that was already initialized must be
    safe. We seed a snapshot row, reopen, and verify it survives — this
    is the foundation for the persistence-across-restart behavior that
    task 3.5 will property-test.
    """

    db_path = tmp_path / "reopen.db"

    first = KnowledgeStore.open(db_path)
    try:
        # The connection is in autocommit mode (``isolation_level=None``),
        # so this insert is durably written without an explicit COMMIT.
        first.connection.execute(
            "INSERT INTO snapshots "
            "(snapshot_id, started_at, status, trigger) "
            "VALUES (?, ?, ?, ?)",
            (1, "2024-01-01T00:00:00", "completed", "full"),
        )
    finally:
        first.close()

    second = KnowledgeStore.open(db_path)
    try:
        rows = second.connection.execute(
            "SELECT snapshot_id, status, trigger FROM snapshots"
        ).fetchall()
        assert rows == [(1, "completed", "full")]
    finally:
        second.close()


@pytest.mark.unit
def test_close_is_idempotent(tmp_path: Path) -> None:
    """Calling ``close()`` twice is a no-op on the second call."""

    store = KnowledgeStore.open(tmp_path / "close.db")
    store.close()
    # Second close must not raise.
    store.close()


@pytest.mark.unit
def test_connection_after_close_raises(tmp_path: Path) -> None:
    """Accessing ``.connection`` after ``close()`` raises a clear error."""

    store = KnowledgeStore.open(tmp_path / "after-close.db")
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = store.connection


@pytest.mark.unit
def test_context_manager_closes_on_exit(tmp_path: Path) -> None:
    """The ``with`` statement form closes the store on exit."""

    db_path = tmp_path / "ctx.db"
    with KnowledgeStore.open(db_path) as store:
        assert store.connection is not None
    with pytest.raises(RuntimeError, match="closed"):
        _ = store.connection


@pytest.mark.unit
def test_open_creates_parent_directory(tmp_path: Path) -> None:
    """``open()`` creates missing parent directories under the db path."""

    nested = tmp_path / "a" / "b" / "c" / "knowledge.db"
    store = KnowledgeStore.open(nested)
    try:
        assert nested.exists()
    finally:
        store.close()

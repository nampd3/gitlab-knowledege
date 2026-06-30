"""Unit tests for the ``Knowledge_Store`` reader interface (task 3.3).

These tests cover the documented happy-path and empty-state behavior of
``get_current_snapshot_id``, ``get_profile``, ``list_profiles``, and
``get_snapshot_metadata``, plus the documented translation of an
underlying SQLite failure into ``KnowledgeStoreUnavailableError``.
The full property-based coverage (round-trip, persistence across
restart, snapshot isolation) lives in tasks 3.4 / 3.5 / 3.6.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from project_knowledge_mcp.errors import KnowledgeStoreUnavailableError
from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    ProjectProfile,
    SnapshotTrigger,
)

if TYPE_CHECKING:
    from pathlib import Path


PRODUCED_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
COMMIT_SHA = "deadbeef" * 5  # 40-char SHA-1 lookalike


def _make_profile(
    *,
    project_id: int,
    purpose: str = "Process orders for the storefront",
    full_path: str = "group/project",
    branch: str = "uat",
    commit_sha: str = COMMIT_SHA,
) -> ProjectProfile:
    """Build a minimal valid ``ProjectProfile`` for round-trip tests."""

    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=full_path,
        analysis_branch=branch,
        analysis_branch_commit_sha=commit_sha,
        produced_at=PRODUCED_AT,
        purpose_summary=purpose,
        abstract_inputs=[
            AbstractInput(
                category=AbstractInputCategory.HTTP_REQUEST,
                description="POST /orders",
            ),
        ],
    )


@pytest.mark.unit
def test_get_current_snapshot_id_is_none_on_fresh_store(tmp_path: Path) -> None:
    """Before any snapshot is committed, the pointer is ``None``."""

    store = KnowledgeStore.open(tmp_path / "fresh.db")
    try:
        assert store.get_current_snapshot_id() is None
    finally:
        store.close()


@pytest.mark.unit
def test_list_profiles_is_empty_on_fresh_store(tmp_path: Path) -> None:
    """An empty store yields an empty list, never ``None``."""

    store = KnowledgeStore.open(tmp_path / "empty.db")
    try:
        assert store.list_profiles() == []
    finally:
        store.close()


@pytest.mark.unit
def test_get_profile_is_none_on_fresh_store(tmp_path: Path) -> None:
    """Looking up a project before any snapshot exists returns ``None``."""

    store = KnowledgeStore.open(tmp_path / "no-snap.db")
    try:
        assert store.get_profile(42) is None
    finally:
        store.close()


@pytest.mark.unit
def test_get_snapshot_metadata_is_none_on_fresh_store(tmp_path: Path) -> None:
    """Snapshot metadata is ``None`` before any commit."""

    store = KnowledgeStore.open(tmp_path / "no-meta.db")
    try:
        assert store.get_snapshot_metadata() is None
    finally:
        store.close()


@pytest.mark.unit
def test_reads_only_observe_committed_snapshots(tmp_path: Path) -> None:
    """An in-progress snapshot's writes are invisible until ``commit_snapshot``."""

    store = KnowledgeStore.open(tmp_path / "in-progress.db")
    try:
        snap = store.begin_snapshot(SnapshotTrigger.FULL)
        store.write_profile(
            snap,
            _make_profile(project_id=7),
            produced_at=PRODUCED_AT,
            commit_sha=COMMIT_SHA,
        )
        # Pointer has not been swapped yet.
        assert store.get_current_snapshot_id() is None
        assert store.get_profile(7) is None
        assert store.list_profiles() == []
        assert store.get_snapshot_metadata() is None

        store.commit_snapshot(snap)

        # Now the snapshot is visible.
        assert store.get_current_snapshot_id() == snap
        profile = store.get_profile(7)
        assert profile is not None
        assert profile.gitlab_project_id == 7
        assert profile.purpose_summary == "Process orders for the storefront"
    finally:
        store.close()


@pytest.mark.unit
def test_list_profiles_orders_by_project_id(tmp_path: Path) -> None:
    """``list_profiles`` returns profiles sorted by ``gitlab_project_id`` ascending."""

    store = KnowledgeStore.open(tmp_path / "list.db")
    try:
        snap = store.begin_snapshot(SnapshotTrigger.FULL)
        for project_id in (3, 1, 2):
            store.write_profile(
                snap,
                _make_profile(project_id=project_id),
                produced_at=PRODUCED_AT,
                commit_sha=COMMIT_SHA,
            )
        store.commit_snapshot(snap)

        ids = [p.gitlab_project_id for p in store.list_profiles()]
        assert ids == [1, 2, 3]
    finally:
        store.close()


@pytest.mark.unit
def test_aborted_snapshot_does_not_become_visible(tmp_path: Path) -> None:
    """An aborted snapshot leaves the previous pointer intact."""

    store = KnowledgeStore.open(tmp_path / "abort.db")
    try:
        # First snapshot commits.
        first = store.begin_snapshot(SnapshotTrigger.FULL)
        store.write_profile(
            first,
            _make_profile(project_id=10, purpose="Original profile"),
            produced_at=PRODUCED_AT,
            commit_sha=COMMIT_SHA,
        )
        store.commit_snapshot(first)

        # Second snapshot writes then aborts.
        second = store.begin_snapshot(
            SnapshotTrigger.SINGLE_PROJECT, parent_snapshot_id=first
        )
        store.write_profile(
            second,
            _make_profile(project_id=10, purpose="Replacement should not be visible"),
            produced_at=PRODUCED_AT,
            commit_sha=COMMIT_SHA,
        )
        store.abort_snapshot(second)

        # Reads still see the first snapshot.
        assert store.get_current_snapshot_id() == first
        profile = store.get_profile(10)
        assert profile is not None
        assert profile.purpose_summary == "Original profile"
    finally:
        store.close()


@pytest.mark.unit
def test_get_snapshot_metadata_returns_per_project_commit_sha(tmp_path: Path) -> None:
    """Metadata exposes ``trigger`` and the per-project commit SHA map."""

    store = KnowledgeStore.open(tmp_path / "meta.db")
    try:
        snap = store.begin_snapshot(SnapshotTrigger.FULL)
        store.write_profile(
            snap,
            _make_profile(project_id=5),
            produced_at=PRODUCED_AT,
            commit_sha="cafef00d" * 5,
        )
        store.write_profile(
            snap,
            _make_profile(project_id=6, full_path="group/other"),
            produced_at=PRODUCED_AT,
            commit_sha="abadcafe" * 5,
        )
        store.commit_snapshot(snap)

        meta = store.get_snapshot_metadata()
        assert meta is not None
        assert meta.snapshot_id == snap
        assert meta.trigger is SnapshotTrigger.FULL
        assert meta.completed_at is not None
        assert meta.commit_sha_by_project == {
            5: "cafef00d" * 5,
            6: "abadcafe" * 5,
        }
    finally:
        store.close()


@pytest.mark.unit
def test_storage_failure_is_translated_to_unavailable_error(tmp_path: Path) -> None:
    """A SQLite-level failure surfaces as ``KnowledgeStoreUnavailableError``.

    We force the failure by closing the underlying connection out from
    under the reader (a stand-in for "disk unreachable"). The reader
    catches the resulting ``sqlite3.ProgrammingError`` (a
    ``sqlite3.Error`` subclass) and re-raises it as the documented
    public error.
    """

    store = KnowledgeStore.open(tmp_path / "broken.db")
    try:
        # Close the underlying handle while leaving the public ``store``
        # object's internal reference to it intact, so subsequent reads
        # hit ``sqlite3.ProgrammingError("Cannot operate on a closed
        # database")``.
        assert store._connection is not None
        store._connection.close()

        with pytest.raises(KnowledgeStoreUnavailableError):
            store.get_current_snapshot_id()
    finally:
        # The connection is already closed; ``KnowledgeStore.close`` is
        # idempotent and this also nulls out the internal reference.
        store.close()

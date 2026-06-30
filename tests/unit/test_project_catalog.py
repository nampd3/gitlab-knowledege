"""Unit tests for ``project_catalog.ProjectCatalog`` (task 4.2).

These tests cover the documented reader semantics of the snapshot-scoped
``Project_Catalog``:

* ``is_in_scope`` returns ``True`` for an enumerated project once the
  snapshot that populated it has been committed, and ``False`` otherwise.
* ``list_in_scope`` orders entries by ``gitlab_project_id`` ascending so
  the Visualization_Server can render the index page directly without
  re-sorting (Requirement 13.1, Requirement 14.3).
* Both readers treat ``current_snapshot.snapshot_id is None`` (a fresh
  store, no ``Ingestion_Job`` has ever committed) as a clean empty
  state — no exception, no implicit "everything is in scope" — matching
  the rest of the ``Knowledge_Store`` reader contract.
* In-progress catalog writes are invisible until ``commit_snapshot``
  swaps the pointer (snapshot-isolation, Property 11).
* A ``single_project`` snapshot inherits the parent's catalog so a
  refresh of one project does not erase the in-scope status of the
  others (this is the catalog-copy added to ``begin_snapshot`` in
  task 4.1).

Implements Requirements 14.3, 14.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.models import EnumeratedProject, SnapshotTrigger
from project_knowledge_mcp.project_catalog import InScopeProject, ProjectCatalog

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enumerated(
    gitlab_project_id: int,
    full_path: str,
    *,
    branch: str = "uat",
    commit_sha: str = "deadbeef" * 5,
) -> EnumeratedProject:
    """Build an ``EnumeratedProject`` whose Analysis_Branch exists.

    The catalog only consumes ``gitlab_project_id`` and ``full_path``,
    but ``EnumeratedProject`` requires the branch fields to be
    consistent (``branch_missing`` ↔ ``commit_sha is None``), so we
    populate them with a stable placeholder.
    """

    return EnumeratedProject(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch_name=branch,
        analysis_branch_commit_sha=commit_sha,
        branch_missing=False,
    )


def _populate_and_commit(
    store: KnowledgeStore,
    catalog: ProjectCatalog,
    projects: list[EnumeratedProject],
) -> int:
    """Run a minimal full-refresh-shaped catalog write and commit.

    Mirrors what the ``Ingestion_Coordinator`` does at the start of a
    full ``Ingestion_Job``: ``begin_snapshot("full")`` →
    ``populate_in_scope`` → ``commit_snapshot``. The intermediate
    profile-write step is intentionally omitted; the catalog is
    independent of ``project_profiles``.
    """

    snapshot_id = store.begin_snapshot(SnapshotTrigger.FULL)
    catalog.populate_in_scope(snapshot_id, projects)
    store.commit_snapshot(snapshot_id)
    return snapshot_id


# ---------------------------------------------------------------------------
# is_in_scope
# ---------------------------------------------------------------------------


def test_is_in_scope_true_for_enumerated_project(tmp_path: Path) -> None:
    """A project written to the catalog and committed is in scope.

    Implements Requirement 14.5 (the catalog is the authoritative
    answer to "is this gitlab_project_id in scope?").
    """

    store = KnowledgeStore.open(tmp_path / "catalog.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_and_commit(
            store,
            catalog,
            [_enumerated(101, "group/alpha")],
        )

        assert catalog.is_in_scope(101) is True
    finally:
        store.close()


def test_is_in_scope_false_for_unknown_project(tmp_path: Path) -> None:
    """A ``gitlab_project_id`` not in the enumeration is out of scope.

    Implements Requirement 14.5.
    """

    store = KnowledgeStore.open(tmp_path / "catalog.db")
    try:
        catalog = ProjectCatalog(store)
        _populate_and_commit(
            store,
            catalog,
            [_enumerated(101, "group/alpha")],
        )

        # 999 was never enumerated, so the catalog must report it
        # out-of-scope (the Visualization_Server uses this exact answer
        # to choose between "not yet analyzed" and "404 out of scope").
        assert catalog.is_in_scope(999) is False
    finally:
        store.close()


def test_is_in_scope_false_when_current_snapshot_is_null(tmp_path: Path) -> None:
    """Fresh store: every id is treated as out-of-scope.

    Before the first ``Ingestion_Job`` commits, ``current_snapshot.snapshot_id``
    is ``NULL``. Per the docstring, ``is_in_scope`` returns ``False``
    so callers do not need to special-case startup state.
    Implements Requirement 14.3.
    """

    store = KnowledgeStore.open(tmp_path / "fresh.db")
    try:
        catalog = ProjectCatalog(store)

        # Sanity-check the precondition: the store really is in the
        # "no committed snapshot" state.
        assert store.get_current_snapshot_id() is None

        assert catalog.is_in_scope(101) is False
    finally:
        store.close()


# ---------------------------------------------------------------------------
# list_in_scope
# ---------------------------------------------------------------------------


def test_list_in_scope_orders_by_gitlab_project_id_ascending(tmp_path: Path) -> None:
    """``list_in_scope`` returns rows sorted by ``gitlab_project_id``.

    The Visualization_Server's index page (Requirement 13.1) renders
    the result directly without re-sorting, so ordering must be
    enforced at the catalog layer.

    Implements Requirement 14.3.
    """

    store = KnowledgeStore.open(tmp_path / "ordering.db")
    try:
        catalog = ProjectCatalog(store)
        # Insert deliberately out-of-order so a plain insertion-order
        # implementation would fail this test.
        _populate_and_commit(
            store,
            catalog,
            [
                _enumerated(303, "group/charlie"),
                _enumerated(101, "group/alpha"),
                _enumerated(202, "group/bravo"),
            ],
        )

        result = catalog.list_in_scope()

        assert result == [
            InScopeProject(gitlab_project_id=101, full_path="group/alpha"),
            InScopeProject(gitlab_project_id=202, full_path="group/bravo"),
            InScopeProject(gitlab_project_id=303, full_path="group/charlie"),
        ]
        # Cross-check the sortedness invariant explicitly so a
        # regression that returns the rows in a different order but
        # still happens to match the equality above (e.g. duplicates)
        # would be caught.
        ids = [p.gitlab_project_id for p in result]
        assert ids == sorted(ids)
    finally:
        store.close()


def test_list_in_scope_empty_when_current_snapshot_is_null(tmp_path: Path) -> None:
    """Fresh store: ``list_in_scope`` returns ``[]``, never ``None``.

    The empty-state response matches the rest of the ``Knowledge_Store``
    reader contract (e.g. ``list_profiles`` on a fresh store).
    Implements Requirement 14.3.
    """

    store = KnowledgeStore.open(tmp_path / "fresh.db")
    try:
        catalog = ProjectCatalog(store)

        assert store.get_current_snapshot_id() is None
        assert catalog.list_in_scope() == []
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Snapshot isolation
# ---------------------------------------------------------------------------


def test_in_progress_catalog_is_invisible_until_commit(tmp_path: Path) -> None:
    """A populated-but-uncommitted catalog must not be visible to readers.

    Property 11 (snapshot isolation) requires that
    ``current_snapshot.snapshot_id`` only advances on
    ``commit_snapshot``; an in-flight ``Ingestion_Job``'s catalog rows
    must therefore be invisible to ``is_in_scope`` /
    ``list_in_scope`` until commit.
    """

    store = KnowledgeStore.open(tmp_path / "isolation.db")
    try:
        catalog = ProjectCatalog(store)

        snapshot_id = store.begin_snapshot(SnapshotTrigger.FULL)
        catalog.populate_in_scope(
            snapshot_id,
            [_enumerated(101, "group/alpha")],
        )

        # Pointer hasn't moved yet — readers see the previous (empty)
        # state.
        assert store.get_current_snapshot_id() is None
        assert catalog.is_in_scope(101) is False
        assert catalog.list_in_scope() == []

        # After commit, the same calls reveal the populated catalog.
        store.commit_snapshot(snapshot_id)
        assert catalog.is_in_scope(101) is True
        assert catalog.list_in_scope() == [
            InScopeProject(gitlab_project_id=101, full_path="group/alpha"),
        ]
    finally:
        store.close()


def test_single_project_snapshot_inherits_parent_catalog(tmp_path: Path) -> None:
    """A ``single_project`` snapshot copies the parent's catalog rows.

    The ``Ingestion_Coordinator`` does not re-run enumeration on a
    single-project refresh; ``begin_snapshot`` (task 4.1) copies the
    parent's catalog into the new snapshot so every project that was
    in scope before the refresh remains in scope after it.
    """

    store = KnowledgeStore.open(tmp_path / "inherit.db")
    try:
        catalog = ProjectCatalog(store)

        # Establish an initial committed snapshot with three projects.
        parent_id = _populate_and_commit(
            store,
            catalog,
            [
                _enumerated(101, "group/alpha"),
                _enumerated(202, "group/bravo"),
                _enumerated(303, "group/charlie"),
            ],
        )

        # Begin a single-project refresh against that parent. The
        # coordinator would normally re-write the profile for the
        # refreshed project here; for the catalog test we only need the
        # snapshot to exist and the catalog rows to have been copied.
        child_id = store.begin_snapshot(
            SnapshotTrigger.SINGLE_PROJECT,
            parent_snapshot_id=parent_id,
        )
        store.commit_snapshot(child_id)

        # Every project from the parent enumeration is still in scope,
        # and the listing is still ordered ascending.
        assert catalog.is_in_scope(101) is True
        assert catalog.is_in_scope(202) is True
        assert catalog.is_in_scope(303) is True
        assert catalog.list_in_scope() == [
            InScopeProject(gitlab_project_id=101, full_path="group/alpha"),
            InScopeProject(gitlab_project_id=202, full_path="group/bravo"),
            InScopeProject(gitlab_project_id=303, full_path="group/charlie"),
        ]
        # And the child snapshot is the one readers see.
        assert store.get_current_snapshot_id() == child_id
    finally:
        store.close()

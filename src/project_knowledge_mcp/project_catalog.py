"""Project_Catalog: snapshot-scoped list of in-scope projects.

The ``Project_Catalog`` is the authoritative answer to the question
"is this ``gitlab_project_id`` in scope?". It is populated at the
*start* of each ``Ingestion_Job`` from the result of
``GitLab_Connector.enumerate_projects()`` — *before* any analysis
runs — so the Visualization_Server can distinguish "in-scope but not
yet analyzed" from "out of scope" (Requirements 14.3 and 14.5).

The catalog is **snapshot-scoped**: rows are tagged with the
in-progress ``snapshot_id`` and only become visible to readers when
``KnowledgeStore.commit_snapshot`` swaps the ``current_snapshot``
pointer. This matches the snapshot-isolation semantics of
``project_profiles`` (Property 11) and is why
:meth:`ProjectCatalog.list_in_scope` and
:meth:`ProjectCatalog.is_in_scope` consult
``current_snapshot.snapshot_id`` via
:meth:`KnowledgeStore.get_current_snapshot_id` rather than reading
across all snapshots.

Single-project refreshes do not re-run enumeration; instead,
:meth:`KnowledgeStore.begin_snapshot` copies the catalog rows from the
parent snapshot so the in-progress single-project snapshot continues
to describe the same set of in-scope projects.

Implements Requirements 14.3 and 14.5.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from .errors import KnowledgeStoreUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .knowledge_store import KnowledgeStore
    from .models import EnumeratedProject


class InScopeProject(BaseModel):
    """A single entry in the in-scope project list returned by ``list_in_scope``.

    Carries only the identity fields the surfaces actually need: the
    ``gitlab_project_id`` (used as the link key on the index page and
    as the argument to MCP tools) and the GitLab ``full_path`` (used as
    the display label). The full ``Project_Profile`` is fetched
    separately when the user navigates to a per-project page.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    gitlab_project_id: int
    full_path: str = Field(..., min_length=1)


class ProjectCatalog:
    """Read/write helper for the snapshot-scoped ``project_catalog`` table.

    The catalog shares its connection with a :class:`KnowledgeStore`;
    constructing a ``ProjectCatalog`` is cheap and stateless. Writers
    (the ``Ingestion_Coordinator``) call :meth:`populate_in_scope`
    once per job, before any per-project analysis runs. Readers (the
    MCP ``list_projects`` tool, the Visualization_Server's index page
    and 404 path) call :meth:`list_in_scope` and :meth:`is_in_scope`
    at request time and never observe rows from non-committed
    snapshots.

    Underlying ``sqlite3.Error`` / ``OSError`` failures during reads
    are translated into :class:`KnowledgeStoreUnavailableError` to
    match the contract of the ``Knowledge_Store`` reader interface
    (Requirement 14.6); writes are not wrapped because they are only
    invoked by the coordinator during an in-progress
    ``Ingestion_Job`` and the coordinator has its own
    abort-on-failure handling (Requirements 2.3, 2.4).
    """

    def __init__(self, store: KnowledgeStore) -> None:
        # The catalog never owns the connection lifecycle — it borrows
        # the live handle from the store. Using a stored reference to
        # ``store`` rather than to ``store.connection`` lets us pick up
        # any future reopen and ensures we always check the current
        # snapshot pointer through the store's own accessor.
        self._store = store

    # -- Writer interface --------------------------------------------------

    def populate_in_scope(
        self,
        snapshot_id: int,
        enumerated_projects: Iterable[EnumeratedProject],
    ) -> None:
        """Write catalog rows for ``snapshot_id`` from ``enumerated_projects``.

        Called by the ``Ingestion_Coordinator`` immediately after
        ``GitLab_Connector.enumerate_projects()`` returns and *before*
        any ``write_profile`` calls so that the catalog reflects the
        full enumeration result even for projects that have not yet
        been analyzed (or that were skipped because their
        ``Analysis_Branch`` is missing — Requirement 15.5). The
        ``snapshot_id`` must reference an ``in_progress`` snapshot;
        attempting to populate a committed or aborted snapshot raises
        ``ValueError`` so the coordinator notices a stale id
        immediately rather than silently writing rows no reader will
        ever see.

        The write is performed inside a single explicit transaction so
        either every catalog row for this enumeration is persisted or
        none is, even if the iterator raises partway through. Existing
        catalog rows for the same ``(snapshot_id, gitlab_project_id)``
        — which can occur when ``begin_snapshot`` copied rows from a
        parent ``single_project`` snapshot — are overwritten via
        ``ON CONFLICT DO UPDATE`` so a refreshed enumeration is the
        source of truth for the snapshot.
        """

        # Fail-fast on stale snapshot ids using the same rule the
        # ``KnowledgeStore`` writers apply; this is the single source
        # of truth for "snapshot is alive".
        self._store._require_in_progress(snapshot_id)

        connection = self._store.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            for project in enumerated_projects:
                connection.execute(
                    """
                    INSERT INTO project_catalog (
                        snapshot_id, gitlab_project_id, full_path
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(snapshot_id, gitlab_project_id) DO UPDATE SET
                        full_path = excluded.full_path
                    """,
                    (
                        snapshot_id,
                        project.gitlab_project_id,
                        project.full_path,
                    ),
                )
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    # -- Reader interface --------------------------------------------------

    def list_in_scope(self) -> list[InScopeProject]:
        """Return every in-scope project from the current snapshot.

        Ordered by ``gitlab_project_id`` ascending so the index page
        from Requirement 13.1 ("lists every in-scope Project ordered
        by GitLab project ID ascending") can render the result
        directly without re-sorting. Returns an empty list — never
        ``None`` — when no snapshot has been committed yet, matching
        the empty-state behavior the surfaces already implement
        against ``KnowledgeStore.list_profiles``.
        """

        snapshot_id = self._store.get_current_snapshot_id()
        if snapshot_id is None:
            return []

        try:
            rows = self._store.connection.execute(
                "SELECT gitlab_project_id, full_path FROM project_catalog "
                "WHERE snapshot_id = ? ORDER BY gitlab_project_id ASC",
                (snapshot_id,),
            ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            raise KnowledgeStoreUnavailableError(str(exc)) from exc

        return [
            InScopeProject(gitlab_project_id=int(project_id), full_path=full_path)
            for project_id, full_path in rows
        ]

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        """Return ``True`` iff ``gitlab_project_id`` is in the current snapshot.

        Returns ``False`` when no snapshot has been committed yet so
        callers do not need to special-case the "before first
        Ingestion_Job" startup state — every id is treated as
        out-of-scope until enumeration has produced at least one
        catalog row and the snapshot has been committed.
        """

        snapshot_id = self._store.get_current_snapshot_id()
        if snapshot_id is None:
            return False

        try:
            row = self._store.connection.execute(
                "SELECT 1 FROM project_catalog "
                "WHERE snapshot_id = ? AND gitlab_project_id = ? LIMIT 1",
                (snapshot_id, gitlab_project_id),
            ).fetchone()
        except (sqlite3.Error, OSError) as exc:
            raise KnowledgeStoreUnavailableError(str(exc)) from exc

        return row is not None


__all__ = ["InScopeProject", "ProjectCatalog"]

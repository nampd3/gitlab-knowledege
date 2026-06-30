"""Snapshot-based ``Knowledge_Store`` backed by SQLite.

This module owns the persistence layer described in the design's
*Knowledge_Store* section. The store is snapshot-oriented:

* every ``Ingestion_Job`` writes profiles tagged with a fresh
  ``snapshot_id``;
* readers consult the ``current_snapshot`` pointer, which is updated
  atomically only when an ingestion completes;
* this gives readers a stable view that is never partially-written
  (Requirements 8.4 / 14.2 / Property 11).

Task 3.1 established the schema and the lifecycle (``open`` / ``close``).
Task 3.2 adds the writer interface used by the
``Ingestion_Coordinator``: :meth:`KnowledgeStore.begin_snapshot`,
:meth:`KnowledgeStore.write_profile`,
:meth:`KnowledgeStore.record_skip`,
:meth:`KnowledgeStore.commit_snapshot`, and
:meth:`KnowledgeStore.abort_snapshot`.

Task 3.3 adds the reader interface used by the MCP query tools and the
``Visualization_Server``: :meth:`KnowledgeStore.get_current_snapshot_id`,
:meth:`KnowledgeStore.get_profile`,
:meth:`KnowledgeStore.list_profiles`, and
:meth:`KnowledgeStore.get_snapshot_metadata`. Every reader consults the
``current_snapshot.snapshot_id`` pointer and never observes rows from
non-committed snapshots (Properties 10 and 11). On underlying storage
failure the readers raise :class:`KnowledgeStoreUnavailableError` —
there is no in-memory fallback (Requirement 14.6).

The schema below matches the design's *Conceptual Schema* exactly, using
the four tables ``snapshots``, ``project_profiles``, ``ingestion_skips``,
and ``current_snapshot``. ``current_snapshot`` is implemented as a
single-row table (constrained by ``CHECK (id = 1)``) holding the
``snapshot_id`` value readers see; on a fresh store the row exists with
``snapshot_id = NULL`` so reads do not need to special-case "no row yet".

Implements Requirements 7.1, 7.3, 7.4, 8.4, 8.5, 14.2.
Targets Properties 9 and 11.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from .errors import KnowledgeStoreUnavailableError
from .models import ProjectProfile, SnapshotMetadata, SnapshotTrigger

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Each statement is executed independently inside ``open()`` so that
# ``CREATE TABLE IF NOT EXISTS`` is safe to call against a pre-existing
# store (Requirement 7.2: persistence across restart).
_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    # The snapshots table records one row per Ingestion_Job. ``status``
    # transitions in_progress -> completed | failed. ``parent_snapshot_id``
    # is set for single-project refreshes that copy from a prior snapshot.
    """
    CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_id        INTEGER PRIMARY KEY,
        started_at         TIMESTAMP NOT NULL,
        completed_at       TIMESTAMP NULL,
        status             TEXT NOT NULL
                              CHECK (status IN ('in_progress', 'completed', 'failed')),
        trigger            TEXT NOT NULL
                              CHECK (trigger IN ('full', 'single_project',
                                                 'scheduled', 'startup_load')),
        parent_snapshot_id INTEGER NULL REFERENCES snapshots(snapshot_id)
    )
    """,
    # The project_profiles table holds one row per (snapshot, project).
    # ``profile_json`` is the full Project_Profile payload so the reader
    # can rehydrate it without joining sub-tables. ``analysis_branch_sha``
    # is NULL when the project did not have the configured Analysis_Branch
    # (the row is never written in that case under task 3.2's writer; the
    # column is nullable to match the design exactly).
    """
    CREATE TABLE IF NOT EXISTS project_profiles (
        snapshot_id              INTEGER NOT NULL
                                    REFERENCES snapshots(snapshot_id),
        gitlab_project_id        INTEGER NOT NULL,
        full_path                TEXT NOT NULL,
        analysis_branch          TEXT NOT NULL,
        analysis_branch_sha      TEXT NULL,
        produced_at              TIMESTAMP NOT NULL,
        profile_json             TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, gitlab_project_id)
    )
    """,
    # The ingestion_skips table records projects that the Ingestion_Job
    # skipped (e.g. ``reason = 'analysis_branch_missing'``, Requirement
    # 15.5). Multiple skips per (snapshot, project) are allowed in
    # principle so we do not declare a composite primary key here.
    """
    CREATE TABLE IF NOT EXISTS ingestion_skips (
        snapshot_id        INTEGER NOT NULL
                              REFERENCES snapshots(snapshot_id),
        gitlab_project_id  INTEGER NOT NULL,
        reason             TEXT NOT NULL,
        detail             TEXT NULL
    )
    """,
    # The current_snapshot table is the atomic-pointer-swap target. It
    # always contains exactly one row (id = 1); ``snapshot_id`` is NULL
    # until the first successful Ingestion_Job calls ``commit_snapshot``.
    """
    CREATE TABLE IF NOT EXISTS current_snapshot (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        snapshot_id INTEGER NULL REFERENCES snapshots(snapshot_id)
    )
    """,
    # The project_catalog table records the set of in-scope projects
    # produced by the GitLab_Connector enumeration step at the start of
    # each Ingestion_Job. It is snapshot-scoped (keyed by
    # ``(snapshot_id, gitlab_project_id)``) so that an in-flight job's
    # enumeration does not become visible until ``commit_snapshot``
    # swaps the pointer (Property 11). Catalog rows are written *before*
    # any analysis runs so the Visualization_Server can distinguish
    # "in-scope but not yet analyzed" from "out of scope" (Requirements
    # 14.3, 14.5).
    """
    CREATE TABLE IF NOT EXISTS project_catalog (
        snapshot_id        INTEGER NOT NULL
                              REFERENCES snapshots(snapshot_id),
        gitlab_project_id  INTEGER NOT NULL,
        full_path          TEXT NOT NULL,
        PRIMARY KEY (snapshot_id, gitlab_project_id)
    )
    """,
    # Helpful indexes for the reader interface (task 3.3) and the skip
    # reporting that task 3.2 will exercise.
    "CREATE INDEX IF NOT EXISTS ix_project_profiles_project "
    "ON project_profiles (gitlab_project_id)",
    "CREATE INDEX IF NOT EXISTS ix_ingestion_skips_snapshot "
    "ON ingestion_skips (snapshot_id)",
    # Lookup helper for ``ProjectCatalog.is_in_scope`` and
    # ``list_in_scope`` (task 4.1). The composite primary key already
    # supports range scans on ``snapshot_id``; this secondary index
    # speeds up cross-snapshot lookups by ``gitlab_project_id``.
    "CREATE INDEX IF NOT EXISTS ix_project_catalog_project "
    "ON project_catalog (gitlab_project_id)",
)


# ``current_snapshot`` is a single-row sentinel; this statement seeds the
# row idempotently after the table is created.
_SEED_CURRENT_SNAPSHOT: Final[str] = (
    "INSERT INTO current_snapshot (id, snapshot_id) VALUES (1, NULL) "
    "ON CONFLICT(id) DO NOTHING"
)


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


class _ReadGuard:
    """Context manager that maps storage failures to ``KnowledgeStoreUnavailableError``.

    Used internally by :class:`KnowledgeStore`'s reader interface so
    every read site has identical translation behavior. A successful
    block exits with no transformation; a :class:`sqlite3.Error` or
    :class:`OSError` is re-raised as a
    :class:`KnowledgeStoreUnavailableError` carrying the underlying
    message. Any other exception passes through unchanged so logic
    errors (e.g. attempting to read after ``close``) remain
    distinguishable from real storage failures.
    """

    def __enter__(self) -> _ReadGuard:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        if exc is None:
            return
        if isinstance(exc, sqlite3.Error | OSError):
            raise KnowledgeStoreUnavailableError(str(exc)) from exc
        # Anything else propagates unchanged (returning ``None`` from
        # ``__exit__`` does not suppress exceptions).
        return


class KnowledgeStore:
    """Lifecycle wrapper around the SQLite-backed Knowledge_Store.

    Construct via :meth:`open` (the canonical entry point) so the schema
    is created and WAL mode is enabled before any caller obtains a
    handle. The writer and reader methods are added by subsequent tasks
    (3.2 and 3.3); task 3.1 only establishes the schema and lifecycle.

    A ``KnowledgeStore`` instance owns a single :class:`sqlite3.Connection`.
    Writers and readers share that connection; the snapshot model means
    they never coordinate via locks at this layer (the in-memory
    ``Ingestion_Coordinator`` enforces single-flight separately, per
    Requirement 8.6).
    """

    #: Path to the underlying SQLite database. Stored for diagnostics
    #: and to support reopen-after-close behavior tested in task 3.5.
    path: Path

    #: The SQLite connection handle. ``None`` when the store is closed.
    _connection: sqlite3.Connection | None

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        # Direct construction is supported but ``open`` is the canonical
        # factory because it both opens the connection and bootstraps
        # the schema. ``__init__`` is kept minimal so tests can pass a
        # pre-built connection (e.g. an in-memory database) when needed.
        self.path = path
        self._connection = connection

    # -- Lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str) -> KnowledgeStore:
        """Open (or create) the SQLite database at ``path`` and bootstrap the schema.

        The database file's parent directory is created if missing so
        that callers can pass a fresh path under a temp directory. WAL
        journal mode is enabled (``PRAGMA journal_mode=WAL``) so the
        write-ahead log file accumulates under the configured path and
        readers do not block writers. Foreign key enforcement is enabled
        explicitly because SQLite leaves it off by default.

        On any failure during schema bootstrap the partially-opened
        connection is closed before the exception propagates so we do
        not leak a sqlite handle to the caller.
        """

        # Accept ``Path | str`` per the task description; everything
        # downstream works with ``Path``.
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            db_path,
            # The Ingestion_Coordinator enforces single-writer semantics
            # at the application layer, but the visualization and MCP
            # surfaces both read concurrently, so disable thread-checks
            # at the sqlite3 layer. Concurrent reads are safe under WAL.
            check_same_thread=False,
            # ``detect_types`` is intentionally left at the default;
            # task 3.2 will store timestamps as ISO-8601 strings rather
            # than rely on sqlite3 type adapters (deprecated in 3.12).
            isolation_level=None,  # autocommit; explicit transactions only
        )

        try:
            cls._configure_pragmas(connection)
            cls._bootstrap_schema(connection)
        except Exception:
            connection.close()
            raise

        return cls(path=db_path, connection=connection)

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Calling ``close()`` twice is safe; the second call is a no-op so
        callers can use the store inside ``try``/``finally`` without
        defensive ``is None`` checks. Once closed, the writer/reader
        methods (added in tasks 3.2 and 3.3) will refuse to operate.
        """

        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None

    # -- Context manager helpers ------------------------------------------

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    # -- Internal helpers --------------------------------------------------

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the live SQLite connection.

        Raises ``RuntimeError`` if the store has been closed. The writer
        and reader implementations in tasks 3.2 / 3.3 use this accessor
        to ensure they never silently operate on a stale handle.
        """

        if self._connection is None:
            raise RuntimeError("KnowledgeStore is closed")
        return self._connection

    # -- Writer interface (task 3.2) --------------------------------------
    #
    # The methods below are the *only* mutators of ``snapshots``,
    # ``project_profiles``, ``ingestion_skips``, and ``current_snapshot``.
    # The ``Ingestion_Coordinator`` is the sole authorized caller in
    # production (single-flight per Requirement 8.6); however, these
    # methods make no concurrency assumptions of their own. Each
    # mutation is either a single autocommit statement or a single
    # explicit transaction; readers consulting the ``current_snapshot``
    # pointer never observe partial writes (Properties 9 and 11).

    def begin_snapshot(
        self,
        trigger: SnapshotTrigger | str,
        parent_snapshot_id: int | None = None,
    ) -> int:
        """Insert a new ``in_progress`` snapshot row and return its id.

        The trigger value is normalized to its enum string form and
        validated against :class:`~project_knowledge_mcp.models.SnapshotTrigger`.
        For the ``single_project`` trigger, all profile rows *and* all
        ``project_catalog`` rows from ``parent_snapshot_id`` are copied
        into the new snapshot inside the same transaction so that the
        per-project re-write that the coordinator performs next sees a
        consistent baseline (and so that ``ProjectCatalog.is_in_scope``
        continues to return ``True`` for every project the previous
        enumeration found, including the one being refreshed).

        ``current_snapshot.snapshot_id`` is *not* updated by this
        method — that swap happens only in :meth:`commit_snapshot`,
        which is what makes Properties 9 and 11 hold.
        """

        trigger_value = self._coerce_trigger(trigger)

        if trigger_value == SnapshotTrigger.SINGLE_PROJECT.value:
            if parent_snapshot_id is None:
                raise ValueError(
                    "single_project trigger requires parent_snapshot_id"
                )
        elif parent_snapshot_id is not None:
            # The design only assigns ``parent_snapshot_id`` semantics to
            # ``single_project`` refreshes; refuse to silently accept a
            # parent on other triggers so callers do not accidentally
            # rely on a copy that will never happen.
            raise ValueError(
                f"parent_snapshot_id is only valid for trigger "
                f"{SnapshotTrigger.SINGLE_PROJECT.value!r}, got {trigger_value!r}"
            )

        started_at = self._now_iso()

        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                "INSERT INTO snapshots (started_at, status, trigger, parent_snapshot_id) "
                "VALUES (?, 'in_progress', ?, ?)",
                (started_at, trigger_value, parent_snapshot_id),
            )
            new_snapshot_id = cursor.lastrowid
            if new_snapshot_id is None:  # pragma: no cover - sqlite3 invariant
                raise RuntimeError("INSERT did not return a lastrowid")

            if trigger_value == SnapshotTrigger.SINGLE_PROJECT.value:
                # Copy every profile row from the parent into the new
                # snapshot. The coordinator will then call
                # ``write_profile`` for the single project being
                # refreshed; the ``ON CONFLICT`` clause in that method
                # will overwrite the copied row for that one project.
                connection.execute(
                    """
                    INSERT INTO project_profiles (
                        snapshot_id, gitlab_project_id, full_path,
                        analysis_branch, analysis_branch_sha,
                        produced_at, profile_json
                    )
                    SELECT
                        ?, gitlab_project_id, full_path,
                        analysis_branch, analysis_branch_sha,
                        produced_at, profile_json
                    FROM project_profiles
                    WHERE snapshot_id = ?
                    """,
                    (new_snapshot_id, parent_snapshot_id),
                )
                # Also copy the parent's project_catalog rows so the
                # in-progress single-project snapshot reflects the same
                # set of in-scope projects as its parent. The
                # coordinator does not re-run enumeration on a
                # single-project refresh; without this copy
                # ``ProjectCatalog.is_in_scope`` would erroneously
                # report every other project out-of-scope after
                # commit.
                connection.execute(
                    """
                    INSERT INTO project_catalog (
                        snapshot_id, gitlab_project_id, full_path
                    )
                    SELECT
                        ?, gitlab_project_id, full_path
                    FROM project_catalog
                    WHERE snapshot_id = ?
                    """,
                    (new_snapshot_id, parent_snapshot_id),
                )
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

        return int(new_snapshot_id)

    def write_profile(
        self,
        snapshot_id: int,
        profile: ProjectProfile,
        produced_at: datetime,
        commit_sha: str,
    ) -> None:
        """Persist (or overwrite) ``profile`` under ``snapshot_id``.

        Writes are last-write-wins per ``(snapshot_id, gitlab_project_id)``
        because the table's primary key is exactly that pair and we use
        an ``ON CONFLICT DO UPDATE`` clause; this is what Property 9
        verifies.

        ``produced_at`` and ``commit_sha`` are persisted in their own
        columns so :class:`SnapshotMetadata` queries (added in task 3.3)
        can return them without rehydrating ``profile_json``. The full
        ``ProjectProfile`` is also stored as JSON in ``profile_json``
        so the reader can return a faithful round-tripped value.

        Raises ``ValueError`` when the snapshot does not exist or has
        already been committed/aborted; this guards against the
        coordinator accidentally tagging writes with a stale id.
        """

        if not commit_sha:
            raise ValueError("commit_sha must be a non-empty string")

        self._require_in_progress(snapshot_id)

        profile_json = profile.model_dump_json()
        produced_at_iso = self._datetime_to_iso(produced_at)

        # Single statement, autocommitted by the connection.
        self.connection.execute(
            """
            INSERT INTO project_profiles (
                snapshot_id, gitlab_project_id, full_path, analysis_branch,
                analysis_branch_sha, produced_at, profile_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id, gitlab_project_id) DO UPDATE SET
                full_path = excluded.full_path,
                analysis_branch = excluded.analysis_branch,
                analysis_branch_sha = excluded.analysis_branch_sha,
                produced_at = excluded.produced_at,
                profile_json = excluded.profile_json
            """,
            (
                snapshot_id,
                profile.gitlab_project_id,
                profile.full_path,
                profile.analysis_branch,
                commit_sha,
                produced_at_iso,
                profile_json,
            ),
        )

    def record_skip(
        self,
        snapshot_id: int,
        gitlab_project_id: int,
        reason: str,
        detail: str | None,
    ) -> None:
        """Record a ``Skip`` row under ``snapshot_id``.

        Used by the Ingestion_Coordinator when a project is skipped —
        canonically when the configured ``Analysis_Branch`` is missing
        on the project (Requirement 15.5). Multiple skips per
        ``(snapshot_id, gitlab_project_id)`` are permitted by the schema
        because the coordinator may produce more than one diagnostic per
        project in future jobs; the writer does not deduplicate.
        """

        if not reason:
            raise ValueError("reason must be a non-empty string")

        self._require_in_progress(snapshot_id)

        self.connection.execute(
            "INSERT INTO ingestion_skips (snapshot_id, gitlab_project_id, reason, detail) "
            "VALUES (?, ?, ?, ?)",
            (snapshot_id, gitlab_project_id, reason, detail),
        )

    def abort_snapshot(self, snapshot_id: int) -> None:
        """Mark ``snapshot_id`` as ``failed`` without touching the pointer.

        Readers continue to see the previously-current snapshot (the
        atomic-pointer-swap that ``commit_snapshot`` performs is
        deliberately *not* performed here, which is the failure case of
        Property 11). The snapshot's profile/skip rows are retained for
        post-mortem inspection; they are simply unreachable to readers
        because the pointer never moved.
        """

        self._require_in_progress(snapshot_id)

        # Single statement is sufficient: only ``snapshots`` is touched
        # and ``current_snapshot.snapshot_id`` is intentionally left
        # alone so readers continue to see the previously-current
        # snapshot.
        self.connection.execute(
            "UPDATE snapshots SET status = 'failed', completed_at = ? "
            "WHERE snapshot_id = ?",
            (self._now_iso(), snapshot_id),
        )

    def commit_snapshot(self, snapshot_id: int) -> None:
        """Atomically promote ``snapshot_id`` to current.

        This is the *atomic-pointer-swap* that Properties 9 and 11
        depend on: within a single SQLite transaction, the snapshot's
        status is flipped to ``"completed"`` and only then is
        ``current_snapshot.snapshot_id`` updated. Readers either see the
        previous snapshot (and its profiles) or this snapshot (and its
        profiles) — never an intermediate state.

        The two-statement order matters: the status flip must happen
        first so that the (transactional) view a reader sees inside
        this transaction is always consistent: any reader joining
        ``current_snapshot`` against ``snapshots.status = 'completed'``
        sees a coherent picture once we COMMIT.
        """

        self._require_in_progress(snapshot_id)

        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            # Flip the snapshot row's status first. We re-check
            # ``status = 'in_progress'`` inside the UPDATE so a racing
            # second commit attempt cannot succeed.
            cursor = connection.execute(
                "UPDATE snapshots SET status = 'completed', completed_at = ? "
                "WHERE snapshot_id = ? AND status = 'in_progress'",
                (self._now_iso(), snapshot_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"failed to flip snapshot {snapshot_id} to 'completed'; "
                    "concurrent commit or aborted snapshot"
                )

            # Only after the status flip do we move the pointer. This
            # ordering is what gives Properties 9 and 11 their
            # all-or-nothing semantics.
            connection.execute(
                "UPDATE current_snapshot SET snapshot_id = ? WHERE id = 1",
                (snapshot_id,),
            )
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    # -- Writer-side helpers ----------------------------------------------

    @staticmethod
    def _coerce_trigger(trigger: SnapshotTrigger | str) -> str:
        """Normalize ``trigger`` to its closed-set string value.

        Accepts either a :class:`SnapshotTrigger` member or its string
        form so callers (the coordinator, tests) can use whichever is
        convenient. Raises ``ValueError`` on any unknown value so
        misspelled triggers surface immediately rather than living
        silently in the schema.
        """

        if isinstance(trigger, SnapshotTrigger):
            return trigger.value
        try:
            return SnapshotTrigger(trigger).value
        except ValueError as exc:
            raise ValueError(f"unknown snapshot trigger: {trigger!r}") from exc

    @staticmethod
    def _now_iso() -> str:
        """Return the current UTC timestamp in ISO-8601 form.

        SQLite stores timestamps as text under our schema; using a
        single helper guarantees every write uses the same
        timezone-aware representation, which keeps the round-trip
        through :meth:`datetime.fromisoformat` lossless.
        """

        return datetime.now(UTC).isoformat()

    @staticmethod
    def _datetime_to_iso(value: datetime) -> str:
        """Serialize ``value`` to ISO-8601, normalizing naive inputs to UTC.

        Naive datetimes are interpreted as UTC; this matches the
        analyzer's own use of :func:`datetime.utcnow`-replacement
        helpers and avoids storing ambiguous timestamps.
        """

        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()

    def _require_in_progress(self, snapshot_id: int) -> None:
        """Assert that ``snapshot_id`` exists and is currently ``in_progress``.

        Used by every writer that mutates a specific snapshot's data.
        Looking up the status before the actual write means a stale
        ``snapshot_id`` (e.g. one that was already committed or
        aborted) never produces silent writes that no reader will ever
        see; the caller gets a precise error instead.
        """

        row = self.connection.execute(
            "SELECT status FROM snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"snapshot_id {snapshot_id} does not exist")
        status = row[0]
        if status != "in_progress":
            raise ValueError(
                f"snapshot {snapshot_id} has status {status!r}; "
                "expected 'in_progress'"
            )

    # -- Reader interface (task 3.3) --------------------------------------
    #
    # These four methods are the only read-path the MCP query tools and
    # the Visualization_Server use against the Knowledge_Store. Every
    # method consults ``current_snapshot.snapshot_id`` (the atomic
    # pointer that :meth:`commit_snapshot` swaps) and never reads from
    # ``in_progress`` or ``failed`` snapshots — that is what gives
    # Property 11 (snapshot isolation) and Property 10 (persistence
    # across restart) their guarantees.
    #
    # On any underlying storage failure, the readers translate the
    # SQLite/IO error into :class:`KnowledgeStoreUnavailableError` so
    # the MCP layer can surface a tool-execution failure (Requirement
    # 11.7) and the Visualization_Server can emit HTTP 503
    # (Requirement 14.6). There is no in-memory fallback: per the
    # design, cached profile data is never used as a substitute for a
    # failed read.

    def get_current_snapshot_id(self) -> int | None:
        """Return the snapshot id readers currently see, or ``None``.

        ``None`` indicates that no ``Ingestion_Job`` has ever
        successfully committed a snapshot — which is the documented
        "no project knowledge available yet" condition both surfaces
        treat as an empty state (Requirement 14.4 for the
        Visualization_Server, Requirement 8.5 for the MCP tools).
        """

        with self._read_guard():
            row = self.connection.execute(
                "SELECT snapshot_id FROM current_snapshot WHERE id = 1"
            ).fetchone()

        if row is None:
            # The seed row is inserted at ``open()`` time, so we should
            # always find it. If we somehow do not (a corrupted
            # database), report unavailability rather than fabricate a
            # ``None``.
            raise KnowledgeStoreUnavailableError(
                "current_snapshot row is missing"
            )

        snapshot_id = row[0]
        if snapshot_id is None:
            return None
        return int(snapshot_id)

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        """Return the profile for ``gitlab_project_id`` from the current snapshot.

        Returns ``None`` when the project has no profile in the current
        snapshot (either because the snapshot does not yet exist or
        because the project was never analyzed in it). The MCP layer
        relies on ``None`` to detect "in-scope but not yet analyzed"
        and translate it into the documented empty-state response
        (Requirement 14.3); an out-of-scope id is rejected separately
        by ``Project_Catalog.is_in_scope``.

        Snapshot isolation is provided by the ``snapshot_id`` filter on
        the ``project_profiles`` query: the writer only swaps the
        ``current_snapshot`` pointer at commit time, and rows for prior
        snapshots are never deleted, so the SELECT below always sees a
        complete profile set for whichever snapshot is current at the
        moment the SELECT runs (Property 11).
        """

        snapshot_id = self.get_current_snapshot_id()
        if snapshot_id is None:
            return None

        with self._read_guard():
            row = self.connection.execute(
                "SELECT profile_json FROM project_profiles "
                "WHERE snapshot_id = ? AND gitlab_project_id = ?",
                (snapshot_id, gitlab_project_id),
            ).fetchone()

        if row is None:
            return None
        return self._rehydrate_profile(row[0])

    def list_profiles(self) -> list[ProjectProfile]:
        """Return every profile in the current snapshot, ordered by project id.

        An empty list means either no committed snapshot exists yet or
        the current snapshot recorded no profiles. Both surfaces treat
        the empty case identically — the Visualization_Server renders
        the "no project knowledge available" empty state per
        Requirement 14.4 — so the reader does not distinguish them
        here.

        The ordering is stable on ``gitlab_project_id`` so callers
        producing UI elements (e.g. the index page from Requirement
        13.1) can rely on a deterministic sequence without having to
        re-sort.
        """

        snapshot_id = self.get_current_snapshot_id()
        if snapshot_id is None:
            return []

        with self._read_guard():
            rows = self.connection.execute(
                "SELECT profile_json FROM project_profiles "
                "WHERE snapshot_id = ? ORDER BY gitlab_project_id ASC",
                (snapshot_id,),
            ).fetchall()

        return [self._rehydrate_profile(row[0]) for row in rows]

    def get_snapshot_metadata(self) -> SnapshotMetadata | None:
        """Return summary metadata for the current snapshot, or ``None``.

        The returned :class:`SnapshotMetadata` carries the snapshot's
        ``started_at``, ``completed_at``, and ``trigger`` together with
        the per-project ``analysis_branch_commit_sha`` map per the
        design's reader interface contract. ``None`` is returned when
        no snapshot has ever been committed — the same empty-state
        signal that :meth:`get_current_snapshot_id` returns ``None``
        for.

        Both queries (the ``snapshots`` row lookup and the per-project
        SHA aggregation) are filtered by the same ``snapshot_id`` we
        read from ``current_snapshot``, so the result is internally
        consistent under Property 11's snapshot-isolation rule. We
        deliberately do *not* re-read the pointer between the two
        queries — that would risk straddling a commit and producing a
        metadata row keyed by snapshot N alongside SHAs from snapshot
        N+1.
        """

        snapshot_id = self.get_current_snapshot_id()
        if snapshot_id is None:
            return None

        with self._read_guard():
            snapshot_row = self.connection.execute(
                "SELECT started_at, completed_at, trigger FROM snapshots "
                "WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()

            sha_rows = self.connection.execute(
                "SELECT gitlab_project_id, analysis_branch_sha FROM project_profiles "
                "WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchall()

        if snapshot_row is None:
            # The pointer references a non-existent snapshot row — the
            # database is in an inconsistent state.
            raise KnowledgeStoreUnavailableError(
                f"current_snapshot points to missing snapshot {snapshot_id}"
            )

        started_at_iso, completed_at_iso, trigger_value = snapshot_row

        return SnapshotMetadata(
            snapshot_id=snapshot_id,
            started_at=self._parse_iso(started_at_iso),
            completed_at=(
                self._parse_iso(completed_at_iso)
                if completed_at_iso is not None
                else None
            ),
            trigger=SnapshotTrigger(trigger_value),
            commit_sha_by_project={
                int(project_id): sha for project_id, sha in sha_rows
            },
        )

    # -- Reader-side helpers ----------------------------------------------

    @staticmethod
    def _parse_iso(value: str) -> datetime:
        """Parse an ISO-8601 timestamp written by the writer interface.

        The writer always emits a timezone-aware UTC string (see
        :meth:`_now_iso` and :meth:`_datetime_to_iso`); parsing
        therefore produces an aware ``datetime``. Older sqlite3
        adapter-written rows that lack a timezone designator are still
        accepted and assumed to be UTC, matching the writer's
        normalization rule.
        """

        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    @staticmethod
    def _rehydrate_profile(profile_json: str) -> ProjectProfile:
        """Rebuild a :class:`ProjectProfile` from its persisted JSON form.

        Pydantic's ``model_validate_json`` re-runs every field validator
        and the model-level invariants (uniqueness of service /
        table names, purpose-summary length, etc.) so a corrupted JSON
        payload surfaces as a validation error here rather than as a
        silent malformed value reaching the MCP / visualization
        surfaces.

        Any failure is re-raised as
        :class:`KnowledgeStoreUnavailableError` because, from the
        reader's perspective, an unrebuildable row is indistinguishable
        from an unreadable one — both mean "the persisted knowledge
        cannot be served right now" (Requirement 14.6).
        """

        try:
            return ProjectProfile.model_validate_json(profile_json)
        except Exception as exc:  # pragma: no cover - corrupted store
            raise KnowledgeStoreUnavailableError(
                f"failed to rehydrate Project_Profile: {exc}"
            ) from exc

    def _read_guard(self) -> _ReadGuard:
        """Translate sqlite/IO failures during a read into the public error.

        Returned as a context manager so call sites read naturally:

            with self._read_guard():
                rows = self.connection.execute(...).fetchall()

        Any :class:`sqlite3.Error` or :class:`OSError` raised inside
        the block is converted to a
        :class:`KnowledgeStoreUnavailableError` carrying the underlying
        reason. Every other exception (e.g. ``RuntimeError`` from
        ``KnowledgeStore is closed``) propagates unchanged so
        programmer errors remain distinguishable from storage
        unavailability.
        """

        return _ReadGuard()

    # -- Internal helpers --------------------------------------------------

    @staticmethod
    def _configure_pragmas(connection: sqlite3.Connection) -> None:
        """Apply the PRAGMA settings the design relies on.

        * ``journal_mode = WAL`` — required by the task description; also
          a prerequisite for non-blocking concurrent reads while the
          Ingestion_Coordinator is writing.
        * ``foreign_keys = ON`` — SQLite ships with foreign keys disabled
          by default; we want the FOREIGN KEY constraints declared in
          the schema to actually fire.
        * ``synchronous = NORMAL`` — safe in WAL mode and noticeably
          faster than the default ``FULL``; durability under crash is
          preserved up to the most recent commit, which is sufficient
          for a snapshot-based store where a partial snapshot is simply
          discarded.
        """

        # ``journal_mode`` is a query-style PRAGMA: it returns the new
        # mode. We assert the result so a misconfigured environment
        # surfaces immediately rather than silently falling back to
        # rollback-journal mode.
        cursor = connection.execute("PRAGMA journal_mode = WAL")
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        mode = (row[0] if row else "").lower()
        if mode != "wal":
            raise RuntimeError(
                f"failed to enable SQLite WAL journal mode (got {mode!r})"
            )

        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = NORMAL")

    @staticmethod
    def _bootstrap_schema(connection: sqlite3.Connection) -> None:
        """Create every table / index declared in :data:`_SCHEMA_STATEMENTS`.

        Wrapped in a single explicit transaction so a half-applied
        schema is impossible. Idempotent because every statement uses
        ``IF NOT EXISTS``.
        """

        connection.execute("BEGIN")
        try:
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(_SEED_CURRENT_SNAPSHOT)
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")


__all__ = ["KnowledgeStore"]

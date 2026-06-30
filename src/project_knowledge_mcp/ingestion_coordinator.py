"""Ingestion_Coordinator: single-flight orchestration of Ingestion_Jobs.

This module implements the state machine described in the design's
"Ingestion_Coordinator" section. An ``IngestionCoordinator`` enforces
Requirement 8.6 (single-flight): at most one ``Ingestion_Job`` is in the
``running`` state at any moment. Every concurrent start attempt issued
while the coordinator is running is rejected with
``IngestionInProgressError("Ingestion_Job is already in progress")`` and
leaves the coordinator state and the ``Knowledge_Store`` unchanged. The
state transition is performed atomically under a ``threading.Lock`` so
that two threads (or two coroutines marshalled onto the same event loop)
racing on ``try_start`` cannot both succeed.

The state machine has exactly two states, mirroring the design diagram::

    idle --start--> running --complete-or-abort--> idle

While ``running``, the coordinator retains the in-progress
``snapshot_id``, the ``trigger`` (``full | single_project | scheduled``),
and the ``started_at`` timestamp. ``current_state()`` returns this
information without mutation so callers (the MCP layer, the visualization
server's diagnostic endpoint, the scheduler's tick logic) can observe
whether a job is in flight.

Task 8.1 established only the state machine and the CAS guard. Task 8.2
layers the full-refresh procedure on top: :meth:`IngestionCoordinator.start_full_refresh`
opens a fresh snapshot via the ``Knowledge_Store``, acquires the running
slot, drives ``GitLab_Connector.enumerate_projects`` and the
``Project_Analyzer`` over the enumerated set, populates the
``Project_Catalog`` for the in-progress snapshot, then atomically commits
the snapshot (Requirements 8.1, 8.4, 8.5, 14.2). On
``GitLabAuthError`` / ``GitLabGroupNotFoundError`` the snapshot is
aborted and the running slot is released through a single try/finally
that surfaces whichever step fails (Requirements 2.3, 2.4). Task 8.3
layers :meth:`IngestionCoordinator.start_single_project_refresh` on
top: it opens a fresh ``"single_project"`` snapshot whose parent is
the current snapshot (so ``Knowledge_Store.begin_snapshot`` copies
the parent's profile rows AND ``Project_Catalog`` rows into the new
snapshot), re-analyzes only the requested project, and commits —
surfacing :class:`ProjectNotInScopeError` when the project is not in
the parent catalog (Requirements 8.2, 10.7). Task 8.4 splices the
explicit ``record_skip`` call for missing-``Analysis_Branch`` projects
into the per-project loop (Requirement 15.5 / Property 30).

Targets Property 12.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from .errors import (
    GitLabAuthError,
    GitLabGroupNotFoundError,
    IngestionInProgressError,
    ProjectNotInScopeError,
)
from .models import ANALYSIS_BRANCH_MISSING_REASON, SnapshotTrigger

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from .gitlab_connector import GitLabConnector
    from .knowledge_store import KnowledgeStore
    from .models import EnumeratedProject, ProjectProfile, RepositoryContents
    from .project_catalog import ProjectCatalog


#: Module-level logger used to emit progress lines during a full
#: ``Ingestion_Job``. Each project is logged at ``WARNING`` level so
#: the operator can watch the refresh progress at Python's default
#: root-logger threshold without bumping every third-party library's
#: verbosity too. The progress lines are not warnings in the strict
#: sense — they are operational signals (a snapshot started, project
#: N of M was analyzed, the body completed) that the operator wants
#: visible at the same baseline as a real warning. The logger's name
#: is unchanged so ``caplog`` filters and downstream handlers can
#: continue to target ``project_knowledge_mcp.ingestion_coordinator``.
_LOGGER = logging.getLogger(__name__)


class AnalyzeCallable(Protocol):
    """Callable signature matching ``project_analyzer.analyze``.

    Declared as a ``Protocol`` so the coordinator can be wired with the
    real ``project_analyzer.analyze`` function in production and with
    purpose-built fakes in unit tests without either side having to
    import a shared concrete symbol. The kwargs match the public
    signature in :mod:`project_knowledge_mcp.project_analyzer` exactly.
    """

    def __call__(
        self,
        project_id: int,
        full_path: str,
        analysis_branch: str,
        commit_sha: str,
        repo_description: str | None,
        repository_contents: RepositoryContents,
    ) -> ProjectProfile: ...


class CoordinatorState(StrEnum):
    """The two states the ``IngestionCoordinator`` can be in.

    Mirrors the design's state diagram. ``RUNNING`` is held by exactly
    one ``Ingestion_Job`` at a time; all other times the coordinator is
    ``IDLE``.
    """

    IDLE = "idle"
    RUNNING = "running"


@dataclass(frozen=True)
class CoordinatorStatus:
    """A read-only snapshot of the coordinator's state at one moment.

    When ``state == CoordinatorState.IDLE`` every other field is ``None``.
    When ``state == CoordinatorState.RUNNING`` ``snapshot_id``, ``trigger``,
    and ``started_at`` describe the job currently in flight (Requirement
    8.6). Returned by ``IngestionCoordinator.current_state()``; callers
    must not retain references and assume liveness because the underlying
    coordinator state can transition at any time.
    """

    state: CoordinatorState
    snapshot_id: int | None = None
    trigger: SnapshotTrigger | None = None
    started_at: datetime | None = None


class JobHandle:
    """A handle on the single ``Ingestion_Job`` currently running.

    Returned by ``IngestionCoordinator.try_start``. The caller MUST end
    the job exactly once by calling ``complete()`` or ``abort()`` (or by
    using the handle as a context manager). After end-of-job the
    coordinator transitions back to ``idle`` and the next ``try_start``
    will succeed.

    For the state-machine skeleton implemented in task 8.1, ``complete``
    and ``abort`` differ only by intent: both release the running slot
    so the next start succeeds. They become semantically distinct in
    task 8.2, when the full job procedure binds them to
    ``Knowledge_Store.commit_snapshot`` and ``Knowledge_Store.abort_snapshot``
    respectively. Keeping the two methods distinct from the start lets
    higher-level code express its intent without ambiguity even before
    the storage layer is wired in.
    """

    def __init__(
        self,
        coordinator: IngestionCoordinator,
        snapshot_id: int,
        trigger: SnapshotTrigger,
        started_at: datetime,
    ) -> None:
        self._coordinator = coordinator
        self._snapshot_id = snapshot_id
        self._trigger = trigger
        self._started_at = started_at
        self._ended = False

    @property
    def snapshot_id(self) -> int:
        """The ``snapshot_id`` this job is writing to."""

        return self._snapshot_id

    @property
    def trigger(self) -> SnapshotTrigger:
        """What caused this job to start (``full``, ``single_project``, ...)."""

        return self._trigger

    @property
    def started_at(self) -> datetime:
        """Wall-clock time at which the coordinator transitioned to ``running``."""

        return self._started_at

    @property
    def is_active(self) -> bool:
        """``True`` until ``complete()`` or ``abort()`` has been called."""

        return not self._ended

    def complete(self) -> None:
        """Mark the job as completed and release the running slot.

        Idempotent: calling ``complete`` (or ``abort``) more than once on
        the same handle is a no-op so cleanup paths can call it
        unconditionally without guarding for double-release.
        """

        self._end()

    def abort(self) -> None:
        """Mark the job as aborted and release the running slot.

        Idempotent for the same reason as ``complete``.
        """

        self._end()

    def _end(self) -> None:
        if self._ended:
            return
        self._ended = True
        # Hand back to the coordinator so it can release the slot under
        # its lock. The coordinator only releases the slot when *this*
        # handle is still the running one, defending against the case
        # where some confused caller resurrects an old handle.
        self._coordinator._release(self)

    def __enter__(self) -> JobHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # On clean exit, mark the job complete; on exception, abort. In
        # either case the slot is released. Returning ``None`` (i.e. not
        # suppressing the exception) preserves the caller's control flow.
        if exc is None:
            self.complete()
        else:
            self.abort()


class IngestionCoordinator:
    """Process-wide single-flight gate for ``Ingestion_Job``s.

    The coordinator owns a ``threading.Lock`` that serializes the
    ``idle → running`` and ``running → idle`` transitions. The lock is
    held for at most a few instructions: long-running work (analysis,
    GitLab calls, SQLite writes) happens *outside* the lock, with the
    ``running`` flag and the in-flight ``snapshot_id`` providing the
    coordination signal that other would-be starters observe.

    Implements Requirement 8.6 and targets Property 12 (at most one
    ``Ingestion_Job`` runs at a time, rejected starts leave state and
    store unchanged, idle starts always succeed).
    """

    def __init__(
        self,
        *,
        knowledge_store: KnowledgeStore | None = None,
        project_catalog: ProjectCatalog | None = None,
        gitlab_connector: GitLabConnector | None = None,
        analyze: AnalyzeCallable | None = None,
    ) -> None:
        # The lock guards every read/write of ``self._running`` so that
        # the CAS in ``try_start`` and the release in ``_release`` cannot
        # observe a torn state.
        self._lock = threading.Lock()
        self._running: JobHandle | None = None
        # Collaborators are injected so unit tests can substitute fakes
        # without monkey-patching module-level state. Each is optional
        # at construction time so the state-machine-only callers (the
        # task 8.1 tests, the scheduler's idle-check) can keep using
        # ``IngestionCoordinator()`` without supplying production
        # dependencies; the procedure-level methods (e.g.
        # :meth:`start_full_refresh`) require them and raise
        # ``RuntimeError`` when they are missing rather than silently
        # no-op'ing on a half-wired coordinator.
        self._knowledge_store = knowledge_store
        self._project_catalog = project_catalog
        self._gitlab_connector = gitlab_connector
        self._analyze = analyze

    def try_start(
        self,
        *,
        trigger: SnapshotTrigger,
        snapshot_id: int,
        started_at: datetime | None = None,
    ) -> JobHandle:
        """Atomically transition ``idle → running`` and return a ``JobHandle``.

        The CAS is performed under ``self._lock``: if the coordinator is
        already ``running``, this method raises ``IngestionInProgressError``
        with the canonical message and the coordinator state is
        unchanged (Requirement 8.6, Property 12). On success, a new
        ``JobHandle`` is constructed with the supplied ``snapshot_id``
        and ``trigger`` and is returned to the caller. The handle's
        ``started_at`` defaults to ``datetime.now(UTC)`` when not
        supplied; supplying an explicit value is useful for tests that
        want to use a virtual clock.

        Args:
            trigger: What is causing this job to start. Must be one of
                the closed-set ``SnapshotTrigger`` values.
            snapshot_id: The fresh ``snapshot_id`` returned by
                ``Knowledge_Store.begin_snapshot`` for this job.
            started_at: Wall-clock time at which the job began. Defaults
                to ``datetime.now(UTC)``.

        Returns:
            The ``JobHandle`` for the newly running job. The caller must
            call ``complete()`` or ``abort()`` (or use the handle as a
            context manager) exactly once.

        Raises:
            IngestionInProgressError: another ``Ingestion_Job`` is
                already in the ``running`` state. The coordinator state
                is left unchanged.
        """

        ts = started_at if started_at is not None else datetime.now(UTC)
        with self._lock:
            if self._running is not None:
                # Per Requirement 8.6 / Property 12, a rejected start
                # MUST leave the coordinator state unchanged. We do not
                # touch ``self._running`` and we do not release the
                # lock-protected ``snapshot_id``/``trigger`` retention
                # of the in-flight job.
                raise IngestionInProgressError()
            handle = JobHandle(
                coordinator=self,
                snapshot_id=snapshot_id,
                trigger=trigger,
                started_at=ts,
            )
            self._running = handle
            return handle

    def current_state(self) -> CoordinatorStatus:
        """Return a read-only snapshot of the coordinator's current state.

        When idle, the returned ``CoordinatorStatus`` has ``state ==
        CoordinatorState.IDLE`` and the other fields are ``None``. When
        running, it carries the in-flight ``snapshot_id``, ``trigger``,
        and ``started_at`` (Requirement 8.6).

        The returned object is a frozen dataclass; the underlying
        coordinator state can transition immediately after this call
        returns, so callers should not treat the result as live.
        """

        with self._lock:
            running = self._running
            if running is None:
                return CoordinatorStatus(state=CoordinatorState.IDLE)
            return CoordinatorStatus(
                state=CoordinatorState.RUNNING,
                snapshot_id=running.snapshot_id,
                trigger=running.trigger,
                started_at=running.started_at,
            )

    def is_idle(self) -> bool:
        """Convenience predicate for ``current_state().state == IDLE``."""

        with self._lock:
            return self._running is None

    def _release(self, handle: JobHandle) -> None:
        """Release the running slot when a ``JobHandle`` ends.

        Called from ``JobHandle.complete`` and ``JobHandle.abort``. The
        lock guarantees that the release is atomic with respect to any
        concurrent ``try_start``; the identity check ensures that an
        out-of-order release from a stale handle cannot accidentally
        clear a fresh job's state.
        """

        with self._lock:
            if self._running is handle:
                self._running = None

    # -- Full-refresh procedure (task 8.2) --------------------------------

    def start_full_refresh(self) -> None:
        """Run a full-refresh ``Ingestion_Job`` end-to-end.

        Drives the procedure described in the design's
        "Ingestion_Coordinator → Job procedure (full refresh)" section
        and implements Requirements 2.3, 2.4, 8.1, 8.4, 8.5, 14.2:

        1. Open a fresh ``"full"`` snapshot via
           ``Knowledge_Store.begin_snapshot`` so the writer interface
           has a snapshot id to tag every subsequent write with.
        2. Acquire the running slot through :meth:`try_start` using
           that snapshot id. If another ``Ingestion_Job`` is already
           running the just-opened snapshot row is rolled back via
           ``Knowledge_Store.abort_snapshot`` before the
           :class:`IngestionInProgressError` is re-raised — this keeps
           the store in a clean state for the next attempt
           (Requirement 8.6).
        3. Enumerate every descendant project of the configured
           GitLab group via ``GitLab_Connector.enumerate_projects()``
           (Requirements 2.1, 2.5).
        4. Populate the snapshot's ``Project_Catalog`` *before* any
           per-project analysis runs so the visualization surface can
           distinguish "in-scope but not yet analyzed" from "out of
           scope" (Requirements 14.3, 14.5).
        5. For each enumerated project that has a non-``None``
           ``analysis_branch_commit_sha``, fetch the repository
           contents pinned to that commit (Requirement 15.3), invoke
           the injected ``Project_Analyzer.analyze`` to produce a
           :class:`ProjectProfile`, and persist it via
           ``Knowledge_Store.write_profile``. The aggregator never
           raises (it records degraded sections instead) so a buggy
           sub-analyzer cannot derail the loop. Projects whose
           ``Analysis_Branch`` is missing are recorded as a ``Skip``
           row carrying the canonical
           ``ANALYSIS_BRANCH_MISSING_REASON`` reason and a detail
           naming both the configured branch and the project id, and
           the loop continues with the remaining projects
           (Requirement 15.5 / Property 30).
        6. Commit the snapshot via
           ``Knowledge_Store.commit_snapshot`` — the single moment
           readers transition to the new view (Properties 9 and 11) —
           and release the running slot.

        On ``GitLabAuthError`` or ``GitLabGroupNotFoundError`` raised
        anywhere inside the job (enumeration, branch lookups, file
        fetches), the snapshot is aborted and the running slot is
        released through a single ``try``/``finally`` block; the
        original exception is re-raised so the caller surfaces the
        underlying failure (Requirements 2.3, 2.4). The
        ``handle.abort()`` call lives in the ``finally`` arm so the
        coordinator returns to ``idle`` even when
        ``Knowledge_Store.abort_snapshot`` itself raises (in which case
        the store-level failure surfaces, matching the design's
        "surfaces whichever fails" rule).

        Raises:
            RuntimeError: When the coordinator was constructed without
                the collaborators required to run a job.
            IngestionInProgressError: When another ``Ingestion_Job`` is
                already in flight.
            GitLabAuthError: When any GitLab request returns HTTP
                401/403; the snapshot is aborted before this is
                re-raised.
            GitLabGroupNotFoundError: When the configured group returns
                HTTP 404; the snapshot is aborted before this is
                re-raised.
        """

        knowledge_store, project_catalog, gitlab_connector, analyze = (
            self._require_collaborators()
        )

        # Step 1: open the snapshot row first so ``try_start`` has a
        # real snapshot id to bind to the running slot. If we cannot
        # acquire the running slot we roll the row back below.
        snapshot_id = knowledge_store.begin_snapshot(SnapshotTrigger.FULL)

        try:
            handle = self.try_start(
                trigger=SnapshotTrigger.FULL,
                snapshot_id=snapshot_id,
            )
        except IngestionInProgressError:
            # Single-flight rejection: the snapshot row we just opened
            # has no owner, so abort it before re-raising. The
            # coordinator state is unchanged (Property 12).
            knowledge_store.abort_snapshot(snapshot_id)
            raise

        try:
            # Step 3: enumerate. We materialize the iterator into a
            # list so the catalog populate step (which iterates) and
            # the per-project analysis loop both see the same set of
            # projects, even though ``enumerate_projects`` is itself a
            # generator.
            enumerated: list[EnumeratedProject] = list(
                gitlab_connector.enumerate_projects()
            )

            # Step 4: catalog rows must be visible to the snapshot's
            # readers before any profile rows so "in-scope but not yet
            # analyzed" is well-defined (Requirements 14.3, 14.5).
            project_catalog.populate_in_scope(snapshot_id, enumerated)

            # Step 5: per-project analysis loop.
            self._analyze_enumerated_projects(
                snapshot_id=snapshot_id,
                enumerated=enumerated,
                gitlab_connector=gitlab_connector,
                knowledge_store=knowledge_store,
                analyze=analyze,
            )

            # Step 6: atomic-pointer-swap. After this returns, readers
            # see the new snapshot; before it returns, they continue
            # to see the previously-current snapshot (Property 11).
            knowledge_store.commit_snapshot(snapshot_id)
        except (GitLabAuthError, GitLabGroupNotFoundError):
            # Requirement 2.3 / 2.4: abort the job and release the
            # running slot, surfacing whichever step fails. The
            # ``finally`` ensures the slot is released even if
            # ``abort_snapshot`` raises; if it does, the store-level
            # exception surfaces (chained from the original auth /
            # group-not-found failure for diagnostics) and the bare
            # ``raise`` below is bypassed.
            try:
                knowledge_store.abort_snapshot(snapshot_id)
            finally:
                handle.abort()
            raise
        else:
            # Clean exit: release the running slot now that the
            # snapshot is committed. ``handle.complete()`` is
            # idempotent so it is safe to call here even though the
            # coordinator already considers the slot logically free
            # the moment ``commit_snapshot`` returned.
            handle.complete()

    # -- Single-project refresh procedure (task 8.3) ----------------------

    def start_single_project_refresh(self, gitlab_project_id: int) -> None:
        """Run a single-project ``Ingestion_Job`` end-to-end.

        Drives the procedure described in the design's
        "Ingestion_Coordinator → Job procedure (single project)" section
        and implements Requirements 8.2 and 10.7:

        1. Read the current snapshot id. The catalog used to decide
           whether ``gitlab_project_id`` is in scope lives inside that
           snapshot, so a single-project refresh fundamentally cannot
           run before any full refresh has committed. When
           ``Knowledge_Store.get_current_snapshot_id`` returns
           ``None`` the request is rejected with
           :class:`ProjectNotInScopeError` (Requirement 10.7) — the
           same outcome the caller observes when the project genuinely
           is not in scope.
        2. Consult the parent ``Project_Catalog`` via
           ``ProjectCatalog.is_in_scope``. If the project is not
           present in the parent snapshot's catalog, raise
           :class:`ProjectNotInScopeError` (Requirement 10.7). No new
           snapshot is opened in this branch, so there is nothing to
           abort.
        3. Open a fresh ``"single_project"`` snapshot via
           ``Knowledge_Store.begin_snapshot`` with the current snapshot
           as its parent. ``begin_snapshot`` copies every parent
           ``project_profiles`` row AND every parent
           ``project_catalog`` row into the new snapshot inside the
           same transaction, so the in-progress snapshot starts out
           describing exactly the same in-scope set and the same
           profiles as the current view; the only difference after
           commit will be the one project we re-analyze below.
        4. Acquire the running slot via :meth:`try_start`. If another
           ``Ingestion_Job`` is already running, the just-opened
           snapshot row is rolled back via
           ``Knowledge_Store.abort_snapshot`` before the
           :class:`IngestionInProgressError` is re-raised — same
           single-flight contract as :meth:`start_full_refresh`
           (Requirement 8.6).
        5. Re-enumerate via ``GitLab_Connector.enumerate_projects()``
           to obtain a fresh :class:`EnumeratedProject` for
           ``gitlab_project_id`` — including, crucially, its current
           ``analysis_branch_commit_sha``. If the id is no longer
           returned by enumeration (the project was in the parent
           catalog but has since disappeared from GitLab) the
           snapshot is aborted, the running slot is released, and
           :class:`ProjectNotInScopeError` is raised so the caller
           sees the same "not in scope" outcome regardless of whether
           the staleness was detected from the local catalog or from
           a fresh GitLab enumeration.
        6. Run the same per-project analysis loop used by the full
           refresh, but on a list of exactly one project. When the
           project's ``analysis_branch_commit_sha`` is ``None`` the
           loop calls ``Knowledge_Store.record_skip`` and continues
           without overwriting the parent profile copied by
           ``begin_snapshot`` (Requirement 15.5 / Property 30) — the
           single project's previously-known profile is preserved
           untouched so readers continue to see it after commit.
           Otherwise the loop fetches the repository contents pinned
           to that commit, invokes the injected ``Project_Analyzer``,
           and overwrites the copied parent row via
           ``Knowledge_Store.write_profile``.
        7. Commit the snapshot via
           ``Knowledge_Store.commit_snapshot`` — the atomic-pointer
           swap that makes the new snapshot the current view
           (Properties 9 and 11) — and release the running slot.

        On ``GitLabAuthError`` or ``GitLabGroupNotFoundError`` raised
        anywhere inside the job (enumeration, branch lookups, file
        fetches), the snapshot is aborted and the running slot is
        released through the same ``try``/``finally`` pattern as
        :meth:`start_full_refresh` (Requirements 2.3, 2.4); the
        original exception is re-raised so the caller surfaces the
        underlying failure.

        Args:
            gitlab_project_id: The GitLab project ID to refresh. Must
                be present in the current snapshot's
                ``Project_Catalog``; if it is not, the request is
                rejected without opening a new snapshot.

        Raises:
            RuntimeError: When the coordinator was constructed without
                the collaborators required to run a job.
            ProjectNotInScopeError: When no current snapshot exists,
                when ``gitlab_project_id`` is not in the parent
                snapshot's catalog, or when a fresh GitLab
                enumeration no longer returns it.
            IngestionInProgressError: When another ``Ingestion_Job``
                is already in flight.
            GitLabAuthError: When any GitLab request returns HTTP
                401/403; the snapshot is aborted before this is
                re-raised.
            GitLabGroupNotFoundError: When the configured group
                returns HTTP 404; the snapshot is aborted before this
                is re-raised.
        """

        knowledge_store, project_catalog, gitlab_connector, analyze = (
            self._require_collaborators()
        )

        # Step 1 + 2: scope check against the parent snapshot. We do
        # both checks *before* opening a new snapshot row so the
        # "not in scope" reject path leaves the store completely
        # untouched (no snapshot row to abort, no running slot to
        # release).
        current_snapshot_id = knowledge_store.get_current_snapshot_id()
        if current_snapshot_id is None:
            # No prior full refresh has committed, so by construction
            # nothing is in scope yet. Surfacing this as
            # ``ProjectNotInScopeError`` (rather than a separate
            # "not yet initialized" error) keeps the MCP-level
            # contract simple: the caller sees one failure mode for
            # "this id cannot be refreshed right now".
            raise ProjectNotInScopeError(gitlab_project_id)
        if not project_catalog.is_in_scope(gitlab_project_id):
            raise ProjectNotInScopeError(gitlab_project_id)

        # Step 3: open the new snapshot. ``begin_snapshot`` copies the
        # parent's profile rows and catalog rows into the new
        # snapshot inside its own transaction; we do not need to copy
        # anything ourselves.
        snapshot_id = knowledge_store.begin_snapshot(
            SnapshotTrigger.SINGLE_PROJECT,
            parent_snapshot_id=current_snapshot_id,
        )

        # Step 4: acquire the running slot. Mirrors the same
        # rejection-rolls-back-the-snapshot rule as
        # :meth:`start_full_refresh` so a denied start leaves no
        # orphaned ``in_progress`` snapshot row behind.
        try:
            handle = self.try_start(
                trigger=SnapshotTrigger.SINGLE_PROJECT,
                snapshot_id=snapshot_id,
            )
        except IngestionInProgressError:
            knowledge_store.abort_snapshot(snapshot_id)
            raise

        try:
            # Step 5: re-enumerate to get a fresh
            # :class:`EnumeratedProject` for ``gitlab_project_id``.
            # We materialize the iterator so the search loop and the
            # GitLab error semantics are both expressed in plain
            # Python rather than tangled into a generator chain.
            enumerated: list[EnumeratedProject] = list(
                gitlab_connector.enumerate_projects()
            )
            matching = next(
                (
                    project
                    for project in enumerated
                    if project.gitlab_project_id == gitlab_project_id
                ),
                None,
            )
            if matching is None:
                # The project was in the parent catalog at the
                # ``is_in_scope`` check above but is no longer
                # returned by enumeration — it was deleted, moved out
                # of the configured group, or lost visibility to the
                # configured token between the two reads. Treat this
                # as the same "not in scope" outcome the caller would
                # have observed if the staleness had been visible to
                # the parent catalog directly. The snapshot row and
                # the running slot are released through the same
                # try/finally idiom used for ``GitLab*`` errors so
                # ``abort_snapshot`` failures still let
                # ``handle.abort()`` run.
                try:
                    knowledge_store.abort_snapshot(snapshot_id)
                finally:
                    handle.abort()
                raise ProjectNotInScopeError(gitlab_project_id)

            # Step 6: re-analyze just this one project. Reusing
            # ``_analyze_enumerated_projects`` keeps the
            # missing-Analysis_Branch skip path (Requirement 15.5 /
            # Property 30) identical between the two job kinds — the
            # single-project loop body is exactly one iteration of
            # the full-refresh loop body.
            self._analyze_enumerated_projects(
                snapshot_id=snapshot_id,
                enumerated=[matching],
                gitlab_connector=gitlab_connector,
                knowledge_store=knowledge_store,
                analyze=analyze,
            )

            # Step 7: atomic-pointer-swap to publish the new
            # snapshot. Profiles for every other project remain the
            # parent's copies (preserved by ``begin_snapshot``); only
            # the one re-analyzed project — or, in the missing-branch
            # path, only the new ``Skip`` row — distinguishes the new
            # current view from the previous one.
            knowledge_store.commit_snapshot(snapshot_id)
        except (GitLabAuthError, GitLabGroupNotFoundError):
            # Same abort-and-release pattern as
            # :meth:`start_full_refresh`: the inner try/finally lets
            # ``handle.abort()`` run even when ``abort_snapshot``
            # itself raises (in which case the store-level failure
            # surfaces and the bare ``raise`` below is bypassed).
            try:
                knowledge_store.abort_snapshot(snapshot_id)
            finally:
                handle.abort()
            raise
        else:
            handle.complete()

    # -- Internal helpers for the refresh procedures ----------------------

    def _require_collaborators(
        self,
    ) -> tuple[KnowledgeStore, ProjectCatalog, GitLabConnector, AnalyzeCallable]:
        """Return the four injected collaborators, or raise if any is missing.

        Centralized so both :meth:`start_full_refresh` and
        :meth:`start_single_project_refresh` get the same precise error
        message when the coordinator was constructed without the
        production wiring it needs.
        """

        if (
            self._knowledge_store is None
            or self._project_catalog is None
            or self._gitlab_connector is None
            or self._analyze is None
        ):
            raise RuntimeError(
                "IngestionCoordinator was constructed without the "
                "knowledge_store / project_catalog / gitlab_connector / "
                "analyze collaborators required to run an Ingestion_Job"
            )
        return (
            self._knowledge_store,
            self._project_catalog,
            self._gitlab_connector,
            self._analyze,
        )

    @staticmethod
    def _analyze_enumerated_projects(
        *,
        snapshot_id: int,
        enumerated: Iterable[EnumeratedProject],
        gitlab_connector: GitLabConnector,
        knowledge_store: KnowledgeStore,
        analyze: AnalyzeCallable,
    ) -> None:
        """Run analyze + write_profile for every project that has a commit SHA.

        Extracted so the loop body is a single, narrow unit of work
        and so task 8.4 can splice in the explicit ``record_skip``
        path for ``analysis_branch_commit_sha is None`` without
        touching :meth:`start_full_refresh` itself.

        Projects whose ``Analysis_Branch`` is missing on the
        repository (``analysis_branch_commit_sha is None``) are
        recorded as a ``Skip`` row carrying the canonical
        ``ANALYSIS_BRANCH_MISSING_REASON`` reason and a ``detail``
        naming both the configured branch and the project id
        (Requirement 15.5 / Property 30), then the loop continues
        with the remaining projects so a single missing branch never
        derails the rest of the refresh.
        """

        projects = list(enumerated)
        total = len(projects)
        profiles_written = 0
        skipped = 0

        _LOGGER.warning(
            "Refresh started: %d projects to analyze in snapshot %d.",
            total,
            snapshot_id,
        )

        for index, project in enumerate(projects, start=1):
            commit_sha = project.analysis_branch_commit_sha
            if commit_sha is None:
                # Requirement 15.5 / Property 30: record a Skip row
                # naming both the configured Analysis_Branch and the
                # project id, then carry on with the remaining
                # projects. The reason is the module-level constant
                # so any reader correlating skip rows to the source
                # of the skip can match on it without re-typing the
                # literal.
                analysis_branch = project.analysis_branch_name
                gitlab_project_id = project.gitlab_project_id
                knowledge_store.record_skip(
                    snapshot_id,
                    gitlab_project_id,
                    reason=ANALYSIS_BRANCH_MISSING_REASON,
                    detail=(
                        f"branch '{analysis_branch}' missing on project "
                        f"{gitlab_project_id}"
                    ),
                )
                skipped += 1
                _LOGGER.warning(
                    "[%d/%d] skipped %s (branch %r missing).",
                    index,
                    total,
                    project.full_path,
                    analysis_branch,
                )
                continue

            _LOGGER.warning(
                "[%d/%d] fetching %s @ %s",
                index,
                total,
                project.full_path,
                commit_sha[:8] if len(commit_sha) >= 8 else commit_sha,
            )

            repository_contents = gitlab_connector.fetch_repository_contents(
                project.gitlab_project_id,
                commit_sha,
            )
            profile = analyze(
                project_id=project.gitlab_project_id,
                full_path=project.full_path,
                analysis_branch=project.analysis_branch_name,
                commit_sha=commit_sha,
                repo_description=project.repository_description,
                repository_contents=repository_contents,
            )
            knowledge_store.write_profile(
                snapshot_id=snapshot_id,
                profile=profile,
                produced_at=profile.produced_at,
                commit_sha=commit_sha,
            )
            profiles_written += 1
            _LOGGER.warning(
                "[%d/%d] analyzed %s (%d analyzable files).",
                index,
                total,
                project.full_path,
                len(repository_contents.files),
            )

        _LOGGER.warning(
            "Refresh body complete in snapshot %d: %d profiles, %d skipped.",
            snapshot_id,
            profiles_written,
            skipped,
        )


__all__ = [
    "AnalyzeCallable",
    "CoordinatorState",
    "CoordinatorStatus",
    "IngestionCoordinator",
    "JobHandle",
]

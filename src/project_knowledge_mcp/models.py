"""Core data models for the Project Knowledge MCP Server.

This module defines the pure-data structures that the rest of the system
operates on: ``ProjectProfile`` and its sub-objects, the ingestion-side
metadata records (``Snapshot``, ``Skip``, ``EnumeratedProject``,
``RepositoryContents``), and the ``ConflictResult`` produced by the
conflict detector.

All models are implemented with Pydantic v2 ``BaseModel``. Validators
enforce the closed-set enums and the structural invariants listed in
the design's Data Models section, including:

* ``len(purpose_summary) <= 1000`` (Requirement 3.4).
* ``purpose_summary_reason`` is non-null when ``purpose_summary == "unknown"``
  (Requirement 3.3).
* ``external_service_dependencies`` contains at most one entry per
  ``name`` (Requirement 5.3).
* ``database_table_dependencies`` contains at most one entry per
  ``table_name`` (Requirement 6.3).
* Every ``ExternalServiceDependency`` and every ``DatabaseTableDependency``
  has a non-empty ``source_locations`` list (Requirements 5.2, 6.2).

The closed-set enums encode:

* ``AbstractInputCategory`` -- Requirement 4.3.
* ``AbstractOutputCategory`` -- Requirement 4.4.
* ``ExternalServiceKind`` -- Requirement 5.2.
* ``DatabaseAccessMode`` -- Requirement 6.2.

Together with Property 6 (well-formed ``Project_Profile`` sections) these
invariants make every successfully-constructed ``ProjectProfile`` valid by
construction; later analyzer code does not need to re-verify them.
"""

from __future__ import annotations

# ``datetime`` and ``Mapping`` are referenced from Pydantic field annotations
# and must be resolvable at runtime by ``get_type_hints``; they cannot be
# moved into a ``TYPE_CHECKING`` block.
from collections.abc import Mapping  # noqa: TC003
from datetime import datetime  # noqa: TC003
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Maximum length of ``ProjectProfile.purpose_summary``. Requirement 3.4.
PURPOSE_SUMMARY_MAX_LEN: int = 1000

#: Canonical "unknown" value used when no purpose summary can be derived.
#: Requirement 3.3.
UNKNOWN_PURPOSE_SUMMARY: str = "unknown"

#: Canonical reason recorded with ``UNKNOWN_PURPOSE_SUMMARY`` when source
#: material is insufficient. Requirement 3.3.
INSUFFICIENT_SOURCE_MATERIAL_REASON: str = "insufficient source material"

#: Canonical reason recorded on a ``Skip`` entry when a project lacks the
#: configured ``Analysis_Branch``. Requirement 15.5.
ANALYSIS_BRANCH_MISSING_REASON: str = "analysis_branch_missing"


# ---------------------------------------------------------------------------
# Closed-set enums
# ---------------------------------------------------------------------------


class AbstractInputCategory(StrEnum):
    """Closed set of categories for ``AbstractInput.category`` (Requirement 4.3)."""

    HTTP_REQUEST = "http_request"
    SCHEDULED_EVENT = "scheduled_event"
    MESSAGE_CONSUMED = "message_consumed"
    FILE_READ = "file_read"
    CLI_ARGUMENT = "cli_argument"
    OTHER = "other"


class AbstractOutputCategory(StrEnum):
    """Closed set of categories for ``AbstractOutput.category`` (Requirement 4.4)."""

    HTTP_RESPONSE = "http_response"
    MESSAGE_PUBLISHED = "message_published"
    FILE_WRITTEN = "file_written"
    DATABASE_WRITE = "database_write"
    EXTERNAL_CALL = "external_call"
    OTHER = "other"


class ExternalServiceKind(StrEnum):
    """Closed set of ``ExternalServiceDependency.kind`` values (Requirement 5.2)."""

    HTTP_API = "http_api"
    MESSAGE_BROKER = "message_broker"
    OBJECT_STORE = "object_store"
    CACHE = "cache"
    AUTH_PROVIDER = "auth_provider"
    OTHER = "other"


class DatabaseAccessMode(StrEnum):
    """Closed set of ``DatabaseTableDependency.access_mode`` values (Requirement 6.2).

    The ``UNKNOWN`` value records the case where the analyzer identified a
    table reference but could not classify the SQL keyword as a read, write,
    or combined access (Go analyzer support, Requirement 9.8). It is the
    lowest-priority observation: when the same table is also seen with
    ``READ``, ``WRITE``, or ``READ_WRITE`` access in the same repository,
    the more specific mode wins; ``UNKNOWN`` is recorded only when it is
    the sole observation for that table.
    """

    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    UNKNOWN = "unknown"


class SnapshotStatus(StrEnum):
    """Status of a single ``Snapshot`` row in the ``Knowledge_Store``."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SnapshotTrigger(StrEnum):
    """What caused a ``Snapshot`` to begin."""

    FULL = "full"
    SINGLE_PROJECT = "single_project"
    SCHEDULED = "scheduled"
    STARTUP_LOAD = "startup_load"


class ConflictKind(StrEnum):
    """Result kind returned by ``Conflict_Detector.classify_pair``."""

    CONFLICT = "conflict"
    NO_CONFLICT = "no_conflict"
    INDETERMINATE = "indeterminate"


# ---------------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------------


class _FrozenModel(BaseModel):
    """Common Pydantic config: frozen, strict about extras, no arbitrary coercion."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=False,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------------
# Leaf records
# ---------------------------------------------------------------------------


class SourceLocation(_FrozenModel):
    """A file path within a repository, optionally tagged with a line number."""

    path: str = Field(..., min_length=1, description="Repository-relative file path.")
    line: int | None = Field(
        default=None,
        description="1-indexed line number, or None when not known.",
    )

    @field_validator("line")
    @classmethod
    def _line_positive_when_set(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("line must be >= 1 when set")
        return v


class AbstractInput(_FrozenModel):
    """A generalized description of data or events a Project consumes."""

    category: AbstractInputCategory
    description: str = Field(..., min_length=1)


class AbstractOutput(_FrozenModel):
    """A generalized description of data or events a Project produces."""

    category: AbstractOutputCategory
    description: str = Field(..., min_length=1)


class ExternalServiceDependency(_FrozenModel):
    """A named external service that the Project calls at runtime.

    ``source_locations`` is non-empty (Requirement 5.2). The
    ``ProjectProfile`` enforces uniqueness of ``name`` across its list of
    ``external_service_dependencies`` (Requirement 5.3).
    """

    name: str = Field(..., min_length=1)
    kind: ExternalServiceKind
    source_locations: list[SourceLocation]

    @field_validator("source_locations")
    @classmethod
    def _source_locations_non_empty(cls, v: list[SourceLocation]) -> list[SourceLocation]:
        if not v:
            raise ValueError("source_locations must be non-empty")
        return v


class DatabaseTableDependency(_FrozenModel):
    """A named database table the Project reads from or writes to.

    ``source_locations`` is non-empty (Requirement 6.2). The
    ``ProjectProfile`` enforces uniqueness of ``table_name`` across its
    list of ``database_table_dependencies`` (Requirement 6.3).
    """

    table_name: str = Field(..., min_length=1)
    access_mode: DatabaseAccessMode
    source_locations: list[SourceLocation]

    @field_validator("source_locations")
    @classmethod
    def _source_locations_non_empty(cls, v: list[SourceLocation]) -> list[SourceLocation]:
        if not v:
            raise ValueError("source_locations must be non-empty")
        return v


# ---------------------------------------------------------------------------
# Project_Profile (the core record)
# ---------------------------------------------------------------------------


class ProjectProfile(_FrozenModel):
    """Structured knowledge for a single Project.

    See the design's Data Models section for the field-by-field schema and
    the invariants enforced here. ``ProjectProfile`` is the value persisted
    by the ``Knowledge_Store`` and returned by the MCP query tools.
    """

    gitlab_project_id: int
    full_path: str = Field(..., min_length=1)
    analysis_branch: str = Field(..., min_length=1)
    analysis_branch_commit_sha: str = Field(..., min_length=1)
    produced_at: datetime
    purpose_summary: str = Field(..., max_length=PURPOSE_SUMMARY_MAX_LEN)
    purpose_summary_reason: str | None = None
    abstract_inputs: list[AbstractInput] = Field(default_factory=list)
    abstract_outputs: list[AbstractOutput] = Field(default_factory=list)
    external_service_dependencies: list[ExternalServiceDependency] = Field(default_factory=list)
    database_table_dependencies: list[DatabaseTableDependency] = Field(default_factory=list)
    degraded_sections: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_invariants(self) -> ProjectProfile:
        # Requirement 3.3: a purpose_summary of "unknown" must carry a
        # non-null, non-empty reason (the canonical reason is
        # INSUFFICIENT_SOURCE_MATERIAL_REASON, but other diagnostic reasons
        # are permitted).
        if self.purpose_summary == UNKNOWN_PURPOSE_SUMMARY and (
            self.purpose_summary_reason is None or self.purpose_summary_reason == ""
        ):
            raise ValueError(
                "purpose_summary_reason must be non-null and non-empty when "
                f"purpose_summary == {UNKNOWN_PURPOSE_SUMMARY!r}"
            )

        # Requirement 5.3: at most one entry per service name.
        service_names = [d.name for d in self.external_service_dependencies]
        if len(service_names) != len(set(service_names)):
            duplicates = sorted({n for n in service_names if service_names.count(n) > 1})
            raise ValueError(
                "external_service_dependencies must contain at most one entry "
                f"per name; duplicates: {duplicates}"
            )

        # Requirement 6.3: at most one entry per table_name. Mixed-mode
        # access on the same table must be coalesced into a single
        # read_write entry by the analyzer before construction.
        table_names = [d.table_name for d in self.database_table_dependencies]
        if len(table_names) != len(set(table_names)):
            duplicates = sorted({n for n in table_names if table_names.count(n) > 1})
            raise ValueError(
                "database_table_dependencies must contain at most one entry "
                f"per table_name; duplicates: {duplicates}"
            )

        return self


# ---------------------------------------------------------------------------
# Ingestion metadata
# ---------------------------------------------------------------------------


class Snapshot(_FrozenModel):
    """Metadata for one ingestion snapshot (a single Ingestion_Job run).

    A snapshot is the unit of atomic visibility for the ``Knowledge_Store``:
    profiles tagged with this ``snapshot_id`` only become visible to readers
    when ``commit_snapshot`` updates ``current_snapshot.snapshot_id``.
    """

    snapshot_id: int
    started_at: datetime
    completed_at: datetime | None = None
    status: SnapshotStatus
    trigger: SnapshotTrigger
    parent_snapshot_id: int | None = None

    @model_validator(mode="after")
    def _check_status_completed_at(self) -> Snapshot:
        # When the snapshot is in_progress, completed_at must be unset;
        # when completed or failed, completed_at should be set.
        if self.status is SnapshotStatus.IN_PROGRESS and self.completed_at is not None:
            raise ValueError("completed_at must be None when status is in_progress")
        return self


class Skip(_FrozenModel):
    """A record that ingestion skipped a project under a given snapshot.

    The canonical ``reason`` for a missing ``Analysis_Branch`` is
    ``ANALYSIS_BRANCH_MISSING_REASON`` (Requirement 15.5).
    """

    snapshot_id: int
    gitlab_project_id: int
    reason: str = Field(..., min_length=1)
    detail: str | None = None


class SnapshotMetadata(_FrozenModel):
    """The summary record returned by ``Knowledge_Store.get_snapshot_metadata``.

    Per the design's ``Knowledge_Store`` reader interface, callers receive
    the snapshot-level fields ``started_at``, ``completed_at``, and
    ``trigger`` together with the per-project ``analysis_branch_commit_sha``
    that each ``Project_Profile`` in the current snapshot was derived
    from (Requirement 7.4). The MCP query tools and the
    ``Visualization_Server`` use this record to surface "knowledge as of
    {timestamp}, derived from {commit}" information without re-reading
    full ``Project_Profile`` payloads.

    ``commit_sha_by_project`` is a read-only mapping from
    ``gitlab_project_id`` to the recorded commit SHA on the
    ``Analysis_Branch``. The value is ``None`` only when the underlying
    storage explicitly recorded a null SHA (the design's schema permits
    this on rows that were copied from a parent snapshot before the SHA
    was filled in); successfully analyzed projects always carry a
    non-empty SHA.
    """

    snapshot_id: int
    started_at: datetime
    completed_at: datetime | None = None
    trigger: SnapshotTrigger
    commit_sha_by_project: Mapping[int, str | None] = Field(default_factory=dict)


class EnumeratedProject(_FrozenModel):
    """One project yielded by ``GitLab_Connector.enumerate_projects``.

    Carries the GitLab identity (``gitlab_project_id``, ``full_path``) and
    the configured ``Analysis_Branch`` metadata (Requirements 2.2, 15.4):
    ``analysis_branch_name`` is the configured branch, and
    ``analysis_branch_commit_sha`` is the most recent commit SHA on that
    branch when it exists. When the branch does not exist on this project,
    ``analysis_branch_commit_sha`` is ``None`` and ``branch_missing`` is
    ``True`` so the analyzer skips and the coordinator can record a
    ``Skip`` entry per Requirement 15.5.
    """

    gitlab_project_id: int
    full_path: str = Field(..., min_length=1)
    analysis_branch_name: str = Field(..., min_length=1)
    analysis_branch_commit_sha: str | None = None
    branch_missing: bool = False
    repository_description: str | None = None

    @field_validator("analysis_branch_commit_sha")
    @classmethod
    def _commit_sha_non_empty_when_set(cls, v: str | None) -> str | None:
        if v is not None and v == "":
            raise ValueError("analysis_branch_commit_sha must be non-empty when set")
        return v

    @model_validator(mode="after")
    def _check_branch_missing_consistency(self) -> EnumeratedProject:
        # branch_missing ↔ analysis_branch_commit_sha is None.
        sha_present = self.analysis_branch_commit_sha is not None
        if self.branch_missing and sha_present:
            raise ValueError("branch_missing is True but analysis_branch_commit_sha is set")
        if not self.branch_missing and not sha_present:
            raise ValueError("branch_missing is False but analysis_branch_commit_sha is None")
        return self


class RepositoryContents(_FrozenModel):
    """An in-memory snapshot of a project's repository at a given commit.

    Returned by ``GitLab_Connector.fetch_repository_contents`` and consumed
    by ``Project_Analyzer``. ``files`` maps repository-relative paths to
    text content; binary blobs are not represented here because the
    analyzer only inspects text artifacts (READMEs, manifests, source
    code).
    """

    gitlab_project_id: int
    commit_sha: str = Field(..., min_length=1)
    files: Mapping[str, str] = Field(default_factory=dict)

    @property
    def file_paths(self) -> list[str]:
        """Sorted list of file paths present in this snapshot."""

        return sorted(self.files.keys())

    def read_text(self, path: str) -> str | None:
        """Return the text contents of ``path``, or ``None`` if absent."""

        return self.files.get(path)


# ---------------------------------------------------------------------------
# Conflict detection result
# ---------------------------------------------------------------------------


class ConflictResult(_FrozenModel):
    """The classification produced by ``Conflict_Detector.classify_pair``.

    ``justification`` is always a non-empty string describing the basis
    for the decision; for ``CONFLICT`` results it must reference either
    substantially-the-same primary responsibility or contradictory
    ownership of the same responsibility (Requirement 9.3); for
    ``INDETERMINATE`` results it must state that the purpose summary is
    unknown for the named project(s) (Requirement 9.4).
    """

    kind: ConflictKind
    justification: str = Field(..., min_length=1)


__all__ = [
    "ANALYSIS_BRANCH_MISSING_REASON",
    "INSUFFICIENT_SOURCE_MATERIAL_REASON",
    "PURPOSE_SUMMARY_MAX_LEN",
    "UNKNOWN_PURPOSE_SUMMARY",
    "AbstractInput",
    "AbstractInputCategory",
    "AbstractOutput",
    "AbstractOutputCategory",
    "ConflictKind",
    "ConflictResult",
    "DatabaseAccessMode",
    "DatabaseTableDependency",
    "EnumeratedProject",
    "ExternalServiceDependency",
    "ExternalServiceKind",
    "ProjectProfile",
    "RepositoryContents",
    "Skip",
    "Snapshot",
    "SnapshotMetadata",
    "SnapshotStatus",
    "SnapshotTrigger",
    "SourceLocation",
]

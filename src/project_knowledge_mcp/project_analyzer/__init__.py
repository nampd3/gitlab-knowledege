"""Project_Analyzer aggregator.

This module exposes :func:`analyze`, the single public entry point of
the ``Project_Analyzer`` component. It wires together the four
sub-analyzers (:mod:`purpose`, :mod:`io_extractor`,
:mod:`external_services`, :mod:`db_tables`) and assembles a complete
:class:`~project_knowledge_mcp.models.ProjectProfile`.

The aggregator is the resilience boundary for analysis. Per the
design's ``Project_Analyzer`` section, "the analyzer never throws; if
any sub-analyzer fails it records a ``degraded`` flag on the produced
``Project_Profile`` and continues with default empty values for the
failing section." That is exactly what :func:`analyze` does:

* Each sub-analyzer is invoked inside its own ``try/except`` block.
* When a sub-analyzer raises any :class:`Exception` the failure is
  logged, the canonical section name is appended to
  :attr:`ProjectProfile.degraded_sections`, and that section is
  populated with safe empty defaults so the rest of the pipeline
  (``Knowledge_Store.write_profile``, the MCP query tools, the
  visualization renderer) can keep making progress.
* As a final defensive layer, if assembling the
  :class:`ProjectProfile` itself raises (for example because a
  sub-analyzer returned a structurally-invalid record that the
  Pydantic model rejects), the aggregator falls back to a
  fully-degraded profile rather than propagating the exception.

Section names recorded in ``degraded_sections`` are stable identifiers
exposed for downstream consumers and visualization: see the
``_*_SECTION`` constants below.

Implements Requirements 3.1, 4.1, 4.2, 5.1, 6.1.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.models import (
    INSUFFICIENT_SOURCE_MATERIAL_REASON,
    UNKNOWN_PURPOSE_SUMMARY,
    AbstractInput,
    AbstractOutput,
    DatabaseTableDependency,
    ExternalServiceDependency,
    ProjectProfile,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.db_tables import (
    detect_database_tables,
)
from project_knowledge_mcp.project_analyzer.external_services import (
    detect_external_services,
)
from project_knowledge_mcp.project_analyzer.go.go_db_tables import (
    detect_go_database_tables,
)
from project_knowledge_mcp.project_analyzer.go.go_external_services import (
    detect_go_external_services,
)
from project_knowledge_mcp.project_analyzer.go._events import SkipFileEvent
from project_knowledge_mcp.project_analyzer.go.go_filter import (
    has_go_artefacts,
    is_go_source_file,
)
from project_knowledge_mcp.project_analyzer.go.go_io import extract_go_io
from project_knowledge_mcp.project_analyzer.go.go_parser import parse_repo
from project_knowledge_mcp.project_analyzer.go.go_purpose import (
    collect_go_candidates,
    is_eligible_main_path,
)
from project_knowledge_mcp.project_analyzer.io_extractor import (
    _Accumulator,
    extract_io,
)
from project_knowledge_mcp.project_analyzer.purpose import (
    PurposeCandidate,
    _normalize_description,
    _truncate,
    collect_purpose_candidates,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from project_knowledge_mcp.models import (
        DatabaseAccessMode,
        SourceLocation,
    )
    from project_knowledge_mcp.project_analyzer.go._events import GoEvent


__all__ = [
    "ABSTRACT_IO_SECTION",
    "DATABASE_TABLES_SECTION",
    "EXTERNAL_SERVICES_SECTION",
    "PURPOSE_SECTION",
    "analyze",
]


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section name constants (recorded in ProjectProfile.degraded_sections)
# ---------------------------------------------------------------------------

#: Recorded when the purpose summarizer fails. The profile's
#: ``purpose_summary`` is set to :data:`UNKNOWN_PURPOSE_SUMMARY` and
#: ``purpose_summary_reason`` to :data:`INSUFFICIENT_SOURCE_MATERIAL_REASON`.
PURPOSE_SECTION: Final[str] = "purpose"

#: Recorded when the I/O extractor fails. Both ``abstract_inputs`` and
#: ``abstract_outputs`` are emitted as empty lists.
ABSTRACT_IO_SECTION: Final[str] = "abstract_io"

#: Recorded when the external service detector fails.
EXTERNAL_SERVICES_SECTION: Final[str] = "external_services"

#: Recorded when the database table detector fails.
DATABASE_TABLES_SECTION: Final[str] = "database_tables"


# ---------------------------------------------------------------------------
# Go purpose-candidate priority slots
# ---------------------------------------------------------------------------
#
# Slots 1, 2, 5, and 6 are owned by :mod:`purpose` and are referenced
# only inside that module. Slots 3, 4, and 7 are documented in design
# §3 ``go.go_purpose`` as the interleaving positions for the three Go
# candidates produced by :func:`collect_go_candidates`. They are
# defined here, in the aggregator, because the aggregator is the
# component responsible for interleaving Go and non-Go candidates
# under Requirement 2.6; :mod:`purpose` itself only knows the non-Go
# slot numbers it reserves room for.

#: Priority of the ``gomod_comment`` candidate (slot 3, between GitLab
#: description and root manifest description).
_PRIORITY_GOMOD_COMMENT: Final[int] = 3

#: Priority of the ``gomod_module_path`` candidate (slot 4, between
#: ``gomod_comment`` and root manifest description).
_PRIORITY_GOMOD_MODULE_PATH: Final[int] = 4

#: Priority of the ``package_doc_comment`` candidate (slot 7, lowest;
#: only consulted after every other candidate has been ruled out).
_PRIORITY_PACKAGE_DOC_COMMENT: Final[int] = 7

#: Priority of the GitLab repository description candidate (slot 2).
#: Mirrors :data:`purpose._PRIORITY_GITLAB_DESCRIPTION` so the
#: aggregator can append the GitLab candidate alongside the Go ones
#: without depending on the private constant in :mod:`purpose`.
_PRIORITY_GITLAB_DESCRIPTION: Final[int] = 2


# ---------------------------------------------------------------------------
# Skip-message surfacing
# ---------------------------------------------------------------------------
#
# Each :class:`SkipFileEvent` produced by ``parse_repo`` represents a
# file the Go layer cannot analyze: a non-trivial build constraint, a
# cgo directive, or a tokenization error (Requirement 10.4 reasons,
# plus the ``"tokenization failed: <detail>"`` shape from
# Requirement 11.1). Each of the four ``_safe_*`` helpers surfaces
# these skips through ``degraded_sections`` as structured strings of
# the form ``"<section>: skipped <path> (<reason>)"``. Downstream
# consumers reading ``degraded_sections`` see a heterogeneous
# ``list[str]`` (plain section names appended on sub-analyzer-level
# failures, plus structured strings appended per skipped file), but
# the field's ``list[str]`` shape is preserved (design §7 "Skip
# surfacing"; Property 13 "structured ``degraded_sections`` entries").


def _file_level_skip_entries(
    events_by_file: Mapping[str, list[GoEvent]],
    section: str,
    *,
    path_filter: Callable[[str], bool] | None = None,
) -> list[str]:
    """Return one ``"<section>: skipped <path> (<reason>)"`` per ``SkipFileEvent``.

    Iterates ``events_by_file`` in path-sorted order so the returned
    list is a deterministic function of the input mapping
    (Requirement 11.4). The reason string is recorded verbatim,
    preserving the canonical strings ``parse_repo`` emits per
    Requirement 10.4 and Requirement 11.1:

    * ``"build constraint requires toolchain"``
    * ``"cgo directive requires toolchain"``
    * ``"tokenization failed: <detail>"``

    Args:
        events_by_file: The per-file event map returned by
            ``parse_repo``.
        section: The aggregator's canonical section name for the
            caller (``"purpose"``, ``"abstract_io"``,
            ``"external_services"``, or ``"database_tables"``).
        path_filter: Optional predicate restricting which paths
            contribute. The purpose section uses
            :func:`go_purpose.is_eligible_main_path` so only files at
            the repository root, at ``cmd/main.go``, or at
            ``cmd/<name>/main.go`` contribute — those are the only
            paths the purpose summarizer would have inspected, so
            "section that would otherwise have run on that file"
            (design Property 13) is computed exactly. The other
            three sections pass ``None`` because every Go file is in
            scope for them.
    """

    entries: list[str] = []
    for path in sorted(events_by_file):
        if path_filter is not None and not path_filter(path):
            continue
        for event in events_by_file[path]:
            if isinstance(event, SkipFileEvent):
                entries.append(f"{section}: skipped {path} ({event.reason})")
    return entries


# ---------------------------------------------------------------------------
# Language-agnostic detector vendor-aware view
# ---------------------------------------------------------------------------
#
# The four Go sub-analyzers filter vendor paths via
# :func:`is_go_source_file` before reaching the tokenizer (Requirement 1.3
# and design §3 "Vendor-Directory Exclusion"), so a vendored copy of a
# third-party Go source cannot contribute a Go-specific detection. The
# parent-spec language-agnostic detectors (``detect_database_tables``,
# ``detect_external_services``, ``extract_io``) have no equivalent
# vendor filter: they walk every text file in
# :class:`RepositoryContents` and apply regex-based passes that match
# SQL literals, URL literals, and SDK constructor patterns. A vendored
# Go source's embedded string literals (e.g. ``"SELECT id FROM USERS"``
# in a third-party library's test data) would therefore leak through
# the language-agnostic path even though the Go layer correctly skips
# the file.
#
# :func:`_view_without_vendored_go_files` removes those vendored Go
# source files from the snapshot before handing it to the
# language-agnostic detectors, so the vendor-exclusion contract holds
# end-to-end (Property 1: "vendor directories never contribute
# detections"). Only ``.go`` files under a ``vendor`` directory segment
# are removed; non-Go files in a vendor tree are preserved because
# Requirement 1.3 is specifically scoped to Go sub-analyzers and Go
# vendoring. The Go-specific detectors continue to receive the
# unfiltered snapshot — their parser is the single point that already
# applies :func:`is_go_source_file`, and they need the unfiltered view
# to read non-Go artefacts like ``go.mod`` for module-path-derived
# binary names.


def _is_vendored_go_path(path: str) -> bool:
    """Return ``True`` when ``path`` is a Go source file under a ``vendor`` tree.

    A path qualifies when it ends with the case-sensitive suffix
    ``.go`` *and* :func:`is_go_source_file` rejects it. The combination
    isolates the "vendored Go source" case: a non-Go file under
    ``vendor/`` has a ``.go``-less suffix and is preserved; a
    non-vendored ``.go`` file is accepted by :func:`is_go_source_file`
    and is also preserved.
    """

    return path.endswith(".go") and not is_go_source_file(path)


def _view_without_vendored_go_files(
    repository_contents: RepositoryContents,
) -> RepositoryContents:
    """Return a snapshot copy with vendored Go source files removed.

    When the input contains no vendored Go source file the function
    returns the input unchanged, so the common path allocates nothing.
    Otherwise a fresh :class:`RepositoryContents` is constructed with
    the same identity fields (``gitlab_project_id``, ``commit_sha``)
    and a ``files`` mapping that omits every vendored ``.go`` path.
    The original snapshot is never mutated; :class:`RepositoryContents`
    is a frozen model.
    """

    files = repository_contents.files
    filtered = {
        path: content
        for path, content in files.items()
        if not _is_vendored_go_path(path)
    }
    if len(filtered) == len(files):
        return repository_contents
    return RepositoryContents(
        gitlab_project_id=repository_contents.gitlab_project_id,
        commit_sha=repository_contents.commit_sha,
        files=filtered,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze(
    project_id: int,
    full_path: str,
    analysis_branch: str,
    commit_sha: str,
    repo_description: str | None,
    repository_contents: RepositoryContents,
) -> ProjectProfile:
    """Run all four sub-analyzers and assemble a :class:`ProjectProfile`.

    Args:
        project_id: GitLab project identifier (becomes
            ``ProjectProfile.gitlab_project_id``).
        full_path: GitLab full path of the project (e.g.
            ``"group/subgroup/project"``).
        analysis_branch: The configured ``Analysis_Branch`` name; copied
            verbatim into the produced profile (Requirement 15.4).
        commit_sha: Commit SHA of ``analysis_branch`` at enumeration
            time. Recorded as
            ``ProjectProfile.analysis_branch_commit_sha``.
        repo_description: Optional GitLab repository description, passed
            in from enumeration. Used by the purpose summarizer.
        repository_contents: An in-memory snapshot of the project's
            files at ``commit_sha``. Consumed by every sub-analyzer.

    Returns:
        A fully-populated :class:`ProjectProfile`. ``degraded_sections``
        is empty when every sub-analyzer succeeded; otherwise it lists
        the section names whose analysis raised, and those sections are
        populated with safe empty defaults.

    The function never raises. Any exception from a sub-analyzer is
    contained, recorded as a degraded section, and replaced with
    defaults so the surrounding ``Ingestion_Job`` can continue with the
    next project.
    """
    degraded: list[str] = []

    # Parse every Go source file in the repository exactly once per
    # analyze() invocation and share the resulting event map across the
    # four _safe_* helpers. When the repository contains no Go artefacts
    # at all, the empty mapping makes every Go branch downstream a
    # no-op, so the produced Project_Profile is byte-identical to the
    # pre-feature output (Requirement 11.5 holds by construction).
    events_by_file: Mapping[str, list[GoEvent]] = (
        parse_repo(repository_contents) if has_go_artefacts(repository_contents) else {}
    )

    # The four parent-spec detectors have no vendor filter of their
    # own (Requirement 1.3 is specifically scoped to the Go sub-
    # analyzers, but its rationale — "vendored copies of third-party
    # libraries must not contribute false detections" — applies to
    # every detector that scans a vendored Go source's string
    # literals). Build a vendor-aware view of the snapshot once and
    # hand it to each language-agnostic detector below. The Go
    # detectors continue to receive the original ``repository_contents``
    # because they need it to read non-Go artefacts (``go.mod`` for
    # module-path-derived binary names) and because the Go parser
    # already applies :func:`is_go_source_file` to skip vendored
    # files. When the repository contains no vendored Go file the
    # view is the input snapshot unchanged.
    non_go_repository_contents = _view_without_vendored_go_files(repository_contents)

    purpose_summary, purpose_reason = _safe_purpose(
        repository_contents,
        non_go_repository_contents,
        repo_description,
        project_id,
        full_path,
        degraded,
        events_by_file,
    )
    abstract_inputs, abstract_outputs = _safe_io(
        repository_contents,
        non_go_repository_contents,
        project_id,
        full_path,
        degraded,
        events_by_file,
    )
    external_services = _safe_external_services(
        repository_contents,
        non_go_repository_contents,
        project_id,
        full_path,
        degraded,
        events_by_file,
    )
    database_tables = _safe_database_tables(
        repository_contents,
        non_go_repository_contents,
        project_id,
        full_path,
        degraded,
        events_by_file,
    )

    return _build_profile(
        project_id=project_id,
        full_path=full_path,
        analysis_branch=analysis_branch,
        commit_sha=commit_sha,
        purpose_summary=purpose_summary,
        purpose_reason=purpose_reason,
        abstract_inputs=abstract_inputs,
        abstract_outputs=abstract_outputs,
        external_services=external_services,
        database_tables=database_tables,
        degraded=degraded,
    )


# ---------------------------------------------------------------------------
# Per-sub-analyzer wrappers
# ---------------------------------------------------------------------------


def _safe_purpose(
    repository_contents: RepositoryContents,
    non_go_repository_contents: RepositoryContents,
    repo_description: str | None,
    project_id: int,
    full_path: str,
    degraded: list[str],
    events_by_file: Mapping[str, list[GoEvent]],
) -> tuple[str, str | None]:
    """Invoke the purpose summarizer; on failure return the unknown pair.

    Collects the language-agnostic candidates first
    (:func:`collect_purpose_candidates` — README at slot 1, manifest at
    slot 5, top-level Python/JS docstring at slot 6), appends the
    GitLab repository description at slot 2 when present, then layers
    the three Go-specific candidates on top when the repository
    contains any Go artefact (``has_go_artefacts`` is ``True``). The
    Go candidates are interleaved at the documented priority positions
    from design §3 ``go.go_purpose`` (Requirement 2.6):

    The language-agnostic ``collect_purpose_candidates`` call runs
    against ``non_go_repository_contents`` so vendored ``.go`` files
    cannot contribute to README/manifest/docstring candidate
    extraction; the Go candidate collection runs against the
    unfiltered ``repository_contents`` because the Go parser already
    applies :func:`is_go_source_file` and needs the full snapshot to
    read ``go.mod`` (Requirement 1.3 and Property 1).

    1. README files (slot 1)
    2. GitLab repository description (slot 2)
    3. ``gomod_comment`` (slot 3, Go)
    4. ``gomod_module_path`` (slot 4, Go)
    5. Root-level package manifest description (slot 5)
    6. Top-level Python/JS docstring (slot 6)
    7. ``package_doc_comment`` (slot 7, Go)

    The first non-empty candidate wins. Length cap and whitespace
    normalization (Requirement 2.5) are applied via the parent-spec
    :func:`purpose._truncate` and :func:`purpose._normalize_description`
    so Go and non-Go candidates pass through identical sizing rules.
    Go candidates returned by :func:`collect_go_candidates` are
    already normalized and truncated, but ``_truncate`` is reapplied
    to the winning candidate uniformly to mirror the
    :func:`summarize_purpose` contract.

    When every slot is empty the function returns the canonical
    unknown pair ``(UNKNOWN_PURPOSE_SUMMARY,
    INSUFFICIENT_SOURCE_MATERIAL_REASON)`` (Requirement 2.5 unknown
    fallback). For repositories without any Go artefact the Go branch
    is a no-op and the result is byte-identical to the parent-spec
    :func:`summarize_purpose` output (Requirement 11.5).

    Per-file Go-parser skip reasons for files the purpose summarizer
    would have inspected (repository-root ``.go`` files,
    ``cmd/main.go``, or ``cmd/<name>/main.go``) are appended to
    ``degraded`` as structured ``"purpose: skipped <path>
    (<reason>)"`` strings (task 11.6, design Property 13). The
    path filter is :func:`go_purpose.is_eligible_main_path`, the same
    predicate :func:`collect_go_candidates` uses to choose
    package-doc-comment sources.

    The outer ``try/except Exception`` preserves the parent-spec
    contract: any uncaught failure records the canonical
    ``"purpose"`` section name in ``degraded`` and returns the safe
    fallback pair (Requirement 11.2).
    """
    try:
        candidates = collect_purpose_candidates(non_go_repository_contents)

        gitlab = _normalize_description(repo_description)
        if gitlab is not None:
            candidates.append(
                PurposeCandidate(
                    source="gitlab_description",
                    text=gitlab,
                    priority=_PRIORITY_GITLAB_DESCRIPTION,
                )
            )

        if has_go_artefacts(repository_contents):
            go_candidates = collect_go_candidates(
                repository_contents, events_by_file
            )
            if go_candidates.gomod_comment is not None:
                candidates.append(
                    PurposeCandidate(
                        source="gomod_comment",
                        text=go_candidates.gomod_comment,
                        priority=_PRIORITY_GOMOD_COMMENT,
                    )
                )
            if go_candidates.gomod_module_path is not None:
                candidates.append(
                    PurposeCandidate(
                        source="gomod_module_path",
                        text=go_candidates.gomod_module_path,
                        priority=_PRIORITY_GOMOD_MODULE_PATH,
                    )
                )
            if go_candidates.package_doc_comment is not None:
                candidates.append(
                    PurposeCandidate(
                        source="package_doc_comment",
                        text=go_candidates.package_doc_comment,
                        priority=_PRIORITY_PACKAGE_DOC_COMMENT,
                    )
                )

        # Stable sort by priority preserves the documented interleaving
        # order. ``list.sort`` is stable in CPython, so candidates that
        # share a priority slot (which the design does not currently
        # define) would fall back to insertion order; in practice every
        # slot above is owned by a single producer.
        candidates.sort(key=lambda candidate: candidate.priority)

        # Surface per-file skip reasons for files the purpose
        # summarizer would have inspected. Only files at the
        # repository root, at ``cmd/main.go``, or at
        # ``cmd/<name>/main.go`` are eligible package-doc-comment
        # sources (the ``go.mod`` candidate is not subject to
        # per-file skipping), so the path filter restricts the
        # structured ``"purpose: skipped <path> (<reason>)"`` entries
        # to that set. This is the same eligibility predicate
        # :func:`go_purpose.collect_go_candidates` uses when
        # iterating ``events_by_file`` for the package-doc-comment
        # candidate (design Property 13: "each section that would
        # otherwise have run on that file").
        degraded.extend(
            _file_level_skip_entries(
                events_by_file,
                PURPOSE_SECTION,
                path_filter=is_eligible_main_path,
            )
        )

        if candidates:
            return _truncate(candidates[0].text), None

        return UNKNOWN_PURPOSE_SUMMARY, INSUFFICIENT_SOURCE_MATERIAL_REASON
    except Exception:
        _logger.exception(
            "Purpose summarizer failed for project %d (%s)", project_id, full_path
        )
        degraded.append(PURPOSE_SECTION)
        return UNKNOWN_PURPOSE_SUMMARY, INSUFFICIENT_SOURCE_MATERIAL_REASON


def _safe_io(
    repository_contents: RepositoryContents,
    non_go_repository_contents: RepositoryContents,
    project_id: int,
    full_path: str,
    degraded: list[str],
    events_by_file: Mapping[str, list[GoEvent]],
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Invoke the I/O extractor; on failure return ``([], [])``.

    Runs the language-agnostic ``extract_io`` first against
    ``non_go_repository_contents`` so vendored ``.go`` files cannot
    contribute to URL/SDK/regex-based detections, then layers the Go
    I/O recognizers on top when the repository contains any Go
    artefact (``has_go_artefacts`` is ``True``). The Go branch reads
    ``repository_contents`` directly because the Go parser already
    applies :func:`is_go_source_file` and the Go scanners may need to
    read ``go.mod`` for module-path-derived binary names. The two
    detection sets are concatenated through
    :class:`io_extractor._Accumulator` so the cross-language coalescing
    rule (dedup by ``(category, description)``) is applied uniformly,
    matching the parent-spec dedup contract.

    Per-file skip reasons from the Go parser (build constraint, cgo,
    tokenization failure) are appended to ``degraded`` as structured
    ``"abstract_io: skipped <path> (<reason>)"`` strings so downstream
    consumers reading ``degraded_sections`` can attribute each skip to
    the section it came from. The structured entries are computed by
    :func:`_file_level_skip_entries` directly from ``events_by_file``,
    not by consuming :func:`extract_go_io`'s third return value;
    that keeps every ``_safe_*`` helper's surfacing path symmetric
    (task 11.6).

    The outer ``try/except Exception`` preserves the parent-spec
    contract: any uncaught failure in either the language-agnostic or
    the Go branch records the canonical ``"abstract_io"`` section name
    in ``degraded`` and returns the safe empty default ``([], [])``.
    """
    try:
        inputs, outputs = extract_io(non_go_repository_contents)
        if has_go_artefacts(repository_contents):
            go_inputs, go_outputs, _ = extract_go_io(
                repository_contents, events_by_file
            )
            # Route the concatenated detections through the existing
            # dedup function so a hypothetical (category, description)
            # overlap between a Python and a Go detection — e.g. a CLI
            # input whose description happens to collide — is recorded
            # only once. The accumulator preserves first-seen order, so
            # the parent-spec detections appear before Go-only ones.
            acc = _Accumulator()
            for entry in inputs:
                acc.add_input(entry.category, entry.description)
            for entry in go_inputs:
                acc.add_input(entry.category, entry.description)
            for entry in outputs:
                acc.add_output(entry.category, entry.description)
            for entry in go_outputs:
                acc.add_output(entry.category, entry.description)
            inputs = acc.inputs
            outputs = acc.outputs
            # Surface per-file Go-parser skip reasons under the
            # canonical ``"abstract_io"`` section name. The helper
            # iterates ``events_by_file`` itself (rather than
            # consuming :func:`extract_go_io`'s third return value)
            # so every ``_safe_*`` helper computes its own structured
            # ``degraded_sections`` entries from the shared parser
            # output (task 11.6). ``extract_go_io``'s
            # ``file_skip_messages`` return remains part of its
            # contract for callers who consume it directly, but the
            # aggregator's surfacing path is independent of it.
            degraded.extend(
                _file_level_skip_entries(events_by_file, ABSTRACT_IO_SECTION)
            )
        return inputs, outputs
    except Exception:
        _logger.exception(
            "I/O extractor failed for project %d (%s)", project_id, full_path
        )
        degraded.append(ABSTRACT_IO_SECTION)
        return [], []


def _safe_external_services(
    repository_contents: RepositoryContents,
    non_go_repository_contents: RepositoryContents,
    project_id: int,
    full_path: str,
    degraded: list[str],
    events_by_file: Mapping[str, list[GoEvent]],
) -> list[ExternalServiceDependency]:
    """Invoke the external service detector; on failure return ``[]``.

    Branches on ``has_go_artefacts(repository_contents)``:

    * **Go project** — calls ``detect_go_external_services`` only.
      The parent-spec language-agnostic ``detect_external_services``
      walks every text file in the repo and matches URL literals +
      SDK constructor patterns; for an ESB-style Go service that
      produces hundreds of false positives from ``.proto`` files,
      ``*.pb.go`` generated code, swagger UI JavaScript, vendored
      documentation, etc. The Go grep detector is the focused source
      of truth: it processes only ``config/config.go``,
      ``internal/adapter/*_adapter.go``, and ``internal/helper.go``.
      Skipping the language-agnostic pass entirely is what produces
      a clean External_Service_Dependency list for the operator's
      microservices.

    * **Non-Go project** — calls ``detect_external_services``
      against ``non_go_repository_contents`` (the vendor-Go-stripped
      view). This is the unchanged behaviour from before the
      Go-grep simplification: URL literals and SDK constructor
      patterns across every analysable text file.

    Per-file Go-parser skip reasons (build constraint, cgo,
    tokenization failure) are still appended to ``degraded`` as
    structured ``"external_services: skipped <path> (<reason>)"``
    strings, computed by :func:`_file_level_skip_entries` directly
    from ``events_by_file``. The Go detector itself returns an empty
    ``file_skip_messages`` list by design — the aggregator is the
    single point that converts :class:`SkipFileEvent` into structured
    ``degraded_sections`` entries (task 11.6).

    The outer ``try/except Exception`` preserves the parent-spec
    contract: any uncaught failure in either branch records the
    canonical ``"external_services"`` section name in ``degraded``
    and returns the safe empty default ``[]``.
    """
    try:
        if has_go_artefacts(repository_contents):
            # Go project — trust the Go grep detector alone. The
            # parent-spec language-agnostic ``detect_external_services``
            # walks every text file in the repo and matches URL
            # literals + SDK constructor patterns; for an ESB-style Go
            # service that means hundreds of false positives from
            # ``.proto`` files, ``*.pb.go`` generated code, swagger UI
            # JavaScript, vendored documentation, etc. The Go grep
            # detector is the focused source of truth: it greps
            # ``config/config.go``, ``internal/adapter/*_adapter.go``,
            # and ``internal/helper.go`` only. Skipping the
            # language-agnostic pass entirely is what produces a clean
            # External_Service_Dependency list for the operator's
            # microservices.
            go_services, _ = detect_go_external_services(
                repository_contents, events_by_file
            )
            services: list[ExternalServiceDependency] = go_services
            # Surface per-file Go-parser skip reasons under the
            # canonical ``"external_services"`` section name. The
            # detector itself returns an empty ``file_skip_messages``
            # list by design (see ``detect_go_external_services``
            # docstring); the aggregator computes the structured
            # ``degraded_sections`` entries directly from
            # ``events_by_file`` (task 11.6).
            degraded.extend(
                _file_level_skip_entries(events_by_file, EXTERNAL_SERVICES_SECTION)
            )
        else:
            services = detect_external_services(non_go_repository_contents)
        return services
    except Exception:
        _logger.exception(
            "External service detector failed for project %d (%s)",
            project_id,
            full_path,
        )
        degraded.append(EXTERNAL_SERVICES_SECTION)
        return []


def _safe_database_tables(
    repository_contents: RepositoryContents,
    non_go_repository_contents: RepositoryContents,
    project_id: int,
    full_path: str,
    degraded: list[str],
    events_by_file: Mapping[str, list[GoEvent]],
) -> list[DatabaseTableDependency]:
    """Invoke the database table detector; on failure return ``[]``.

    Branches on ``has_go_artefacts(repository_contents)``:

    * **Go project** — calls ``detect_go_database_tables`` only.
      The parent-spec language-agnostic ``detect_database_tables``
      runs a SQL-keyword regex over every text file in the repo,
      which on an ESB-style microservice produces a large set of
      false positives: words like ``viper``, ``dir``, ``queue``,
      ``the`` get caught by ``SELECT ... FROM <word>`` patterns when
      they appear in READMEs, generated ``*.pb.go`` files, dto
      comments, or config-loader source. The Go grep detector is
      the focused source of truth: it processes only
      ``internal/repository/*_repo.go`` / ``*_repos.go`` files and
      preserves ``<schema>.<table>`` exactly (Requirement 9.5).

    * **Non-Go project** — calls ``detect_database_tables``
      against ``non_go_repository_contents`` (the vendor-Go-stripped
      view). This is the unchanged behaviour from before the
      Go-grep simplification: SQL-keyword regex passes against
      every analysable text file.

    Per-file Go-parser skip reasons (build constraint, cgo,
    tokenization failure) are still appended to ``degraded`` as
    structured ``"database_tables: skipped <path> (<reason>)"``
    strings, computed by :func:`_file_level_skip_entries` directly
    from ``events_by_file``. The Go detector itself returns an empty
    ``file_skip_messages`` list by design — the aggregator is the
    single point that converts :class:`SkipFileEvent` into structured
    ``degraded_sections`` entries (task 11.6).

    The outer ``try/except Exception`` preserves the parent-spec
    contract: any uncaught failure in either branch records the
    canonical ``"database_tables"`` section name in ``degraded``
    and returns the safe empty default ``[]``.
    """
    try:
        if has_go_artefacts(repository_contents):
            # Go project — trust the Go grep detector alone. The
            # parent-spec language-agnostic ``detect_database_tables``
            # runs a SQL-keyword regex over every text file in the
            # repo, which on an ESB-style microservice produces a
            # large set of false positives: words like ``viper``,
            # ``dir``, ``queue``, ``the`` get caught by the
            # ``SELECT ... FROM <word>`` pattern when they appear in
            # READMEs, generated ``*.pb.go`` files, dto comments, or
            # config-loader source. The Go grep detector is the
            # focused source of truth: it processes only
            # ``internal/repository/*_repo.go`` /
            # ``*_repos.go`` files. Skipping the language-agnostic
            # pass entirely is what produces a clean
            # Database_Table_Dependency list for the operator's
            # microservices.
            go_tables, _ = detect_go_database_tables(
                repository_contents, events_by_file
            )
            tables: list[DatabaseTableDependency] = go_tables
            # Surface per-file Go-parser skip reasons under the
            # canonical ``"database_tables"`` section name. The
            # detector itself returns an empty ``file_skip_messages``
            # list by design (see ``detect_go_database_tables``
            # docstring); the aggregator computes the structured
            # ``degraded_sections`` entries directly from
            # ``events_by_file`` (task 11.6).
            degraded.extend(
                _file_level_skip_entries(events_by_file, DATABASE_TABLES_SECTION)
            )
        else:
            tables = detect_database_tables(non_go_repository_contents)
        return tables
    except Exception:
        _logger.exception(
            "Database table detector failed for project %d (%s)",
            project_id,
            full_path,
        )
        degraded.append(DATABASE_TABLES_SECTION)
        return []


# ---------------------------------------------------------------------------
# Profile assembly with model-validation fallback
# ---------------------------------------------------------------------------


def _build_profile(
    *,
    project_id: int,
    full_path: str,
    analysis_branch: str,
    commit_sha: str,
    purpose_summary: str,
    purpose_reason: str | None,
    abstract_inputs: list[AbstractInput],
    abstract_outputs: list[AbstractOutput],
    external_services: list[ExternalServiceDependency],
    database_tables: list[DatabaseTableDependency],
    degraded: list[str],
) -> ProjectProfile:
    """Construct a :class:`ProjectProfile`, falling back to safe defaults.

    The :class:`ProjectProfile` model enforces a small set of structural
    invariants (Requirements 3.3, 5.3, 6.3). If a sub-analyzer produced
    a value that trips one of them — for example by yielding two entries
    with the same service name despite the design's deduplication
    contract — Pydantic raises ``ValidationError`` from the constructor.
    Per the aggregator's "never raises" rule, we recover by marking
    every analysis section as degraded and emitting a profile with safe
    empty defaults. This keeps the surrounding ``Ingestion_Job``
    advancing rather than failing the whole batch on one project.
    """
    if purpose_summary == UNKNOWN_PURPOSE_SUMMARY and not purpose_reason:
        # Defensive: Requirement 3.3 forbids the unknown summary without
        # a non-empty reason. A buggy summarizer that returned ("unknown",
        # None) would otherwise trip the model validator.
        purpose_reason = INSUFFICIENT_SOURCE_MATERIAL_REASON

    try:
        return ProjectProfile(
            gitlab_project_id=project_id,
            full_path=full_path,
            analysis_branch=analysis_branch,
            analysis_branch_commit_sha=commit_sha,
            produced_at=datetime.now(UTC),
            purpose_summary=purpose_summary,
            purpose_summary_reason=purpose_reason,
            abstract_inputs=abstract_inputs,
            abstract_outputs=abstract_outputs,
            external_service_dependencies=external_services,
            database_table_dependencies=database_tables,
            degraded_sections=degraded,
        )
    except Exception:
        _logger.exception(
            "ProjectProfile assembly failed for project %d (%s); "
            "falling back to fully-degraded profile",
            project_id,
            full_path,
        )
        for section in (
            PURPOSE_SECTION,
            ABSTRACT_IO_SECTION,
            EXTERNAL_SERVICES_SECTION,
            DATABASE_TABLES_SECTION,
        ):
            if section not in degraded:
                degraded.append(section)
        return ProjectProfile(
            gitlab_project_id=project_id,
            full_path=full_path,
            analysis_branch=analysis_branch,
            analysis_branch_commit_sha=commit_sha,
            produced_at=datetime.now(UTC),
            purpose_summary=UNKNOWN_PURPOSE_SUMMARY,
            purpose_summary_reason=INSUFFICIENT_SOURCE_MATERIAL_REASON,
            abstract_inputs=[],
            abstract_outputs=[],
            external_service_dependencies=[],
            database_table_dependencies=[],
            degraded_sections=degraded,
        )

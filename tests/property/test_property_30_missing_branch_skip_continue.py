# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 30: For all Ingestion_Jobs where the configured Analysis_Branch does not exist on a subset S of in-scope projects, the job SHALL: not produce a Project_Profile for any project in S, record a Skip entry for every project in S with reason == "analysis_branch_missing" and a detail that names both the configured Analysis_Branch value and the project's gitlab_project_id, continue to attempt analysis for every other in-scope project not in S.
"""Property test for the missing-Analysis_Branch skip-and-continue path.

**Validates Requirement 15.5** (Property 30 in the design).

For every randomly generated set of in-scope projects in which an
arbitrary subset ``S`` lacks the configured ``Analysis_Branch``, a single
``IngestionCoordinator.start_full_refresh()`` run SHALL:

* leave **no** ``Project_Profile`` in the committed snapshot for any
  project in ``S`` (``Knowledge_Store.get_profile(id) is None``);
* record a ``Skip`` row in ``ingestion_skips`` for every project in
  ``S`` carrying ``reason == "analysis_branch_missing"`` and a
  ``detail`` string that names *both* the configured
  ``Analysis_Branch`` value *and* the project's ``gitlab_project_id``;
* still produce a ``Project_Profile`` for every other in-scope project
  (the loop "continues" past the missing-branch ones), with no spurious
  ``Skip`` rows for those.

The test exercises ``IngestionCoordinator`` end-to-end against a real
``KnowledgeStore`` and ``ProjectCatalog`` opened on a per-example
``tempfile.TemporaryDirectory``, with a fake ``GitLabConnector`` that
emits the synthesized ``EnumeratedProject`` records and a fake
``analyze`` callable that produces deterministic ``ProjectProfile``\\s.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator
from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.models import (
    ANALYSIS_BRANCH_MISSING_REASON,
    EnumeratedProject,
    ProjectProfile,
    RepositoryContents,
)
from project_knowledge_mcp.project_catalog import ProjectCatalog

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator, Sequence

pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Fixed test-fixture values
# ---------------------------------------------------------------------------

ANALYSIS_BRANCH = "uat"
PRODUCED_AT = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Strategy: a list of (project_id, branch_missing) pairs
# ---------------------------------------------------------------------------


@st.composite
def _project_specs(draw: st.DrawFn) -> list[tuple[int, bool]]:
    """Generate 1-8 in-scope projects with independent ``branch_missing`` flags.

    The ``gitlab_project_id`` values are drawn as a unique-by-element
    list so every generated project has a distinct identifier (matching
    the GitLab invariant that project IDs are unique within an
    instance). Each project then independently flips a coin to decide
    whether the configured ``Analysis_Branch`` is present on it; this
    lets the test cover the four interesting partitions:

    * every project has the branch (no skips, every project analyzed);
    * no project has the branch (every project skipped, no analysis);
    * a proper subset is missing (the mixed case Property 30 names
      explicitly).
    """

    project_ids = draw(
        st.lists(
            st.integers(min_value=1, max_value=10_000_000),
            min_size=1,
            max_size=8,
            unique=True,
        )
    )
    return [(pid, draw(st.booleans())) for pid in project_ids]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeGitLabConnector:
    """Minimal ``GitLabConnector`` stand-in driven by a generated project list.

    ``enumerate_projects`` yields one :class:`EnumeratedProject` per
    spec. Projects whose ``branch_missing`` flag is ``True`` carry
    ``analysis_branch_commit_sha = None`` (the canonical "missing
    branch" signal the coordinator inspects in
    :meth:`IngestionCoordinator._analyze_enumerated_projects`); the
    others carry a synthetic 40-hex SHA derived from their
    ``gitlab_project_id``.

    ``fetch_repository_contents`` returns a tiny
    :class:`RepositoryContents` for any commit it is asked about. It
    records every call so the test can assert that the coordinator
    *never* invokes it for a missing-branch project (a coordinator
    that did would double-skip and could mask the contract).
    """

    def __init__(self, specs: Sequence[tuple[int, bool]]) -> None:
        self._specs = list(specs)
        self.fetched_project_ids: list[int] = []

    @staticmethod
    def _commit_sha_for(project_id: int) -> str:
        # 40 hex characters so the SHA looks like a real one to the
        # rest of the system; the project id is encoded into the
        # suffix so SHAs differ per project.
        return f"{project_id:040x}"

    def enumerate_projects(self) -> Iterator[EnumeratedProject]:
        for project_id, branch_missing in self._specs:
            if branch_missing:
                yield EnumeratedProject(
                    gitlab_project_id=project_id,
                    full_path=f"group/p{project_id}",
                    analysis_branch_name=ANALYSIS_BRANCH,
                    analysis_branch_commit_sha=None,
                    branch_missing=True,
                    repository_description=None,
                )
            else:
                yield EnumeratedProject(
                    gitlab_project_id=project_id,
                    full_path=f"group/p{project_id}",
                    analysis_branch_name=ANALYSIS_BRANCH,
                    analysis_branch_commit_sha=self._commit_sha_for(project_id),
                    branch_missing=False,
                    repository_description=None,
                )

    def fetch_repository_contents(
        self,
        project_id: int,
        commit_sha: str,
    ) -> RepositoryContents:
        # Recorded so the test can assert the coordinator never reaches
        # this code for a missing-branch project — only present-branch
        # projects should cause a fetch.
        self.fetched_project_ids.append(project_id)
        return RepositoryContents(
            gitlab_project_id=project_id,
            commit_sha=commit_sha,
            files={},
        )


def _fake_analyze(
    *,
    project_id: int,
    full_path: str,
    analysis_branch: str,
    commit_sha: str,
    repo_description: str | None,
    repository_contents: RepositoryContents,
) -> ProjectProfile:
    """Deterministic ``analyze`` stand-in producing a minimal valid profile."""

    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=full_path,
        analysis_branch=analysis_branch,
        analysis_branch_commit_sha=commit_sha,
        produced_at=PRODUCED_AT,
        purpose_summary=f"profile for {project_id}",
    )


# ---------------------------------------------------------------------------
# Direct ``ingestion_skips`` query helper
# ---------------------------------------------------------------------------


def _read_skip_rows(
    store: KnowledgeStore,
    snapshot_id: int,
) -> list[tuple[int, str, str | None]]:
    """Return ``(gitlab_project_id, reason, detail)`` for every skip row.

    The reader interface on ``KnowledgeStore`` does not expose skip
    rows directly (they are observable via ``Knowledge_Store``'s
    schema, not via ``get_profile`` / ``list_profiles``), so we query
    them through the live connection. ``OperationalError`` /
    ``IntegrityError`` here would indicate a schema regression and is
    deliberately allowed to surface as a test failure rather than be
    swallowed.
    """

    cursor: sqlite3.Cursor = store.connection.execute(
        "SELECT gitlab_project_id, reason, detail FROM ingestion_skips "
        "WHERE snapshot_id = ? ORDER BY gitlab_project_id ASC, rowid ASC",
        (snapshot_id,),
    )
    return [(int(pid), reason, detail) for pid, reason, detail in cursor.fetchall()]


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(specs=_project_specs())
@settings(max_examples=100)
def test_missing_analysis_branch_projects_are_skipped_and_others_continue(
    specs: list[tuple[int, bool]],
) -> None:
    """Property 30: skip every project in ``S``, analyze every project not in ``S``."""

    missing_ids = {pid for pid, missing in specs if missing}
    present_ids = {pid for pid, missing in specs if not missing}

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "knowledge.db"
        store = KnowledgeStore.open(db_path)
        try:
            catalog = ProjectCatalog(store)
            connector = _FakeGitLabConnector(specs)
            coordinator = IngestionCoordinator(
                knowledge_store=store,
                project_catalog=catalog,
                gitlab_connector=connector,
                analyze=_fake_analyze,
            )

            # End-to-end run of the full-refresh job. The fake
            # connector cannot raise ``GitLabAuthError`` /
            # ``GitLabGroupNotFoundError`` so we expect a clean
            # commit; any other exception would itself be a
            # property violation.
            coordinator.start_full_refresh()

            # The committed snapshot is what readers see; we use the
            # public reader API for profile checks, then fall back to
            # the connection for the ``ingestion_skips`` table the
            # reader API deliberately does not expose.
            current_snapshot_id = store.get_current_snapshot_id()
            assert current_snapshot_id is not None, (
                "start_full_refresh should commit a snapshot when no "
                "GitLab error occurs"
            )

            # --- Property 30, clause 1: no profile for any project in S ---
            for pid in missing_ids:
                assert store.get_profile(pid) is None, (
                    f"project {pid} has no Analysis_Branch but a "
                    f"Project_Profile was produced anyway"
                )

            # --- Property 30, clause 2: skip rows for every project in S ---
            skip_rows = _read_skip_rows(store, current_snapshot_id)

            # Exactly one skip per missing-branch project. Anything
            # else (zero rows, multiple rows, or a row for a
            # present-branch project) violates Property 30.
            skip_ids = [pid for pid, _reason, _detail in skip_rows]
            assert sorted(skip_ids) == sorted(missing_ids), (
                f"expected one skip row per missing-branch project "
                f"{sorted(missing_ids)}, got rows for {sorted(skip_ids)}"
            )

            for pid, reason, detail in skip_rows:
                assert reason == ANALYSIS_BRANCH_MISSING_REASON, (
                    f"skip row for project {pid} has reason {reason!r}; "
                    f"expected {ANALYSIS_BRANCH_MISSING_REASON!r}"
                )
                assert detail is not None, (
                    f"skip row for project {pid} has no detail; the "
                    f"detail must name both the Analysis_Branch and "
                    f"the project id"
                )
                assert ANALYSIS_BRANCH in detail, (
                    f"skip detail {detail!r} does not name the "
                    f"configured Analysis_Branch {ANALYSIS_BRANCH!r}"
                )
                assert str(pid) in detail, (
                    f"skip detail {detail!r} does not name the "
                    f"project id {pid}"
                )

            # --- Property 30, clause 3: every other project IS analyzed ---
            for pid in present_ids:
                profile = store.get_profile(pid)
                assert profile is not None, (
                    f"project {pid} has the Analysis_Branch but no "
                    f"Project_Profile was produced — the loop did not "
                    f"continue past a missing-branch project"
                )
                # Sanity: the produced profile is the one we drove
                # through ``_fake_analyze`` for this id, not a stray
                # cross-wired payload.
                assert profile.gitlab_project_id == pid
                assert profile.analysis_branch == ANALYSIS_BRANCH
                assert profile.purpose_summary == f"profile for {pid}"

            # And no spurious skip rows for present-branch projects
            # (covered transitively by the equality check on
            # ``skip_ids`` above, but spelled out here so a regression
            # against the "continue" half of Property 30 surfaces
            # with a precise message).
            for pid in present_ids:
                assert pid not in skip_ids, (
                    f"project {pid} has the Analysis_Branch but a "
                    f"Skip row was recorded for it"
                )

            # --- Defensive: the connector was never asked to fetch
            # contents for a missing-branch project (it would have
            # had to make up a SHA, which is exactly the situation
            # Property 30 is preventing). ---
            assert set(connector.fetched_project_ids) == present_ids, (
                f"connector fetched contents for {connector.fetched_project_ids!r}; "
                f"expected exactly the present-branch projects {sorted(present_ids)}"
            )
        finally:
            store.close()

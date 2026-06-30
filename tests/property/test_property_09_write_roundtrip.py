# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 9: For all sequences of write_profile operations within a single snapshot, for every gitlab_project_id that received at least one write, get_profile(gitlab_project_id) SHALL return the value of the most recent write (after the snapshot is committed and made current), and the persisted record SHALL include produced_at and analysis_branch_commit_sha.
"""Property 9: write round-trip and last-write-wins within a snapshot.

**Validates Requirements 7.1, 7.3, 7.4** (Property 9 in the design).

For every Hypothesis-generated sequence of ``write_profile`` calls on a
freshly-opened ``KnowledgeStore``, this test:

1. opens a brand-new SQLite-backed store under a temp directory;
2. begins a single ``FULL`` snapshot;
3. writes a sequence of profiles drawn over a small pool of
   ``gitlab_project_id`` values so collisions are exercised on every
   run (last-write-wins);
4. commits the snapshot (the atomic-pointer-swap that Property 9
   depends on);
5. for every project id that received at least one write, asserts that
   ``get_profile`` returns *exactly* the most recent write -- including
   the ``produced_at`` and ``analysis_branch_commit_sha`` recorded by
   that write.

Each write uses a distinct ``produced_at``, ``analysis_branch_commit_sha``,
and ``purpose_summary`` so the most-recent write is identifiable in the
read-back profile.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.models import ProjectProfile, SnapshotTrigger

# ---------------------------------------------------------------------------
# Test fixture data
# ---------------------------------------------------------------------------

# A small pool of gitlab_project_id values keeps the probability of
# collision (and therefore last-write-wins exercise) high. With
# max_examples=100 and a write-list size in [1, 30], every example
# generates many overwrites of the same id.
_PROJECT_ID_POOL: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Base timestamp used to derive a distinct ``produced_at`` per write.
_BASE_PRODUCED_AT: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

#: ``Analysis_Branch`` value reused on every profile.
_ANALYSIS_BRANCH: str = "uat"


def _make_profile(
    *,
    project_id: int,
    full_path: str,
    analysis_branch: str,
    commit_sha: str,
    produced_at: datetime,
    purpose_summary: str,
) -> ProjectProfile:
    """Build a minimal valid ``ProjectProfile`` for the round-trip test.

    The persisted record is checked back via ``get_profile``; only the
    fields required by Property 9's assertion (``gitlab_project_id``,
    ``produced_at``, ``analysis_branch_commit_sha``, plus a
    distinguishing ``purpose_summary``) are exercised, so the optional
    list sections are left empty.
    """

    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=full_path,
        analysis_branch=analysis_branch,
        analysis_branch_commit_sha=commit_sha,
        produced_at=produced_at,
        purpose_summary=purpose_summary,
    )


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    write_sequence=st.lists(
        st.sampled_from(_PROJECT_ID_POOL),
        min_size=1,
        max_size=30,
    ),
)
@settings(max_examples=100)
def test_write_round_trip_and_last_write_wins(write_sequence: list[int]) -> None:
    """Property 9: get_profile returns the most-recent write per project id."""

    # Per-example, in-memory record of the *expected* state: the most
    # recent ``(produced_at, commit_sha, purpose_summary)`` written for
    # each project id. This is the oracle the read-back is checked
    # against.
    expected: dict[int, tuple[datetime, str, str]] = {}

    # A fresh database file per Hypothesis example so no example can
    # see another's writes (Property 9 is scoped to a single snapshot
    # on a fresh store; cross-restart persistence is the subject of
    # Property 10).
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test.db"
        store = KnowledgeStore.open(db_path)
        try:
            snapshot_id = store.begin_snapshot(SnapshotTrigger.FULL)

            for step, project_id in enumerate(write_sequence):
                # Distinct ``produced_at`` per write so the most-recent
                # write is identifiable in the persisted record.
                produced_at = _BASE_PRODUCED_AT + timedelta(seconds=step)
                # Distinct, schema-valid commit SHA per write (40 hex
                # chars, the SHA-1 lookalike used elsewhere in the
                # suite). The step index is encoded into the suffix so
                # every write has a unique value.
                commit_sha = f"{step:040x}"
                # Distinct summary so the read-back can also assert on
                # the payload itself, not just the metadata columns.
                purpose_summary = (
                    f"step {step}: write to project {project_id}"
                )

                profile = _make_profile(
                    project_id=project_id,
                    full_path=f"group/project-{project_id}",
                    analysis_branch=_ANALYSIS_BRANCH,
                    commit_sha=commit_sha,
                    produced_at=produced_at,
                    purpose_summary=purpose_summary,
                )

                store.write_profile(
                    snapshot_id,
                    profile,
                    produced_at=produced_at,
                    commit_sha=commit_sha,
                )

                # Last-write-wins oracle: this overwrites any prior
                # entry for the same project id.
                expected[project_id] = (produced_at, commit_sha, purpose_summary)

            # Atomic-pointer-swap: only after this returns is the
            # snapshot visible to readers.
            store.commit_snapshot(snapshot_id)

            # Every project id that received at least one write must be
            # readable, with the value of the most recent write.
            for project_id, (
                expected_produced_at,
                expected_commit_sha,
                expected_summary,
            ) in expected.items():
                profile = store.get_profile(project_id)

                assert profile is not None, (
                    f"get_profile({project_id}) returned None after commit; "
                    f"expected the most recent write"
                )
                assert profile.gitlab_project_id == project_id
                # The persisted record includes produced_at and the
                # analysis-branch commit SHA per Requirement 7.4.
                assert profile.produced_at == expected_produced_at, (
                    f"produced_at mismatch for project {project_id}: "
                    f"got {profile.produced_at}, expected {expected_produced_at}"
                )
                assert profile.analysis_branch_commit_sha == expected_commit_sha, (
                    f"analysis_branch_commit_sha mismatch for project "
                    f"{project_id}: got {profile.analysis_branch_commit_sha}, "
                    f"expected {expected_commit_sha}"
                )
                # The persisted payload itself is the most recent
                # write's payload (Requirement 7.3: replacement
                # semantics).
                assert profile.purpose_summary == expected_summary, (
                    f"purpose_summary mismatch for project {project_id}: "
                    f"got {profile.purpose_summary!r}, expected {expected_summary!r}"
                )
                assert profile.analysis_branch == _ANALYSIS_BRANCH
        finally:
            store.close()

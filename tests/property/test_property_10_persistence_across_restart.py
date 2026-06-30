# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 10: For all sequences write* → commit → close → reopen → read*, the values returned by reads after reopen SHALL equal the values written before the close, drawn from the last successfully committed snapshot.
"""Property test for persistence across restart.

**Validates Requirement 7.2** (Property 10 in the design).

For every randomly-generated trace of ``begin_snapshot → write_profile* →
commit_or_abort`` operations followed by ``close`` and ``reopen``, the
reads issued after reopen must return the values written before the
close, drawn from the **last successfully committed** snapshot. Aborted
snapshots leave the previous current snapshot in place; if no snapshot
was ever committed the store re-opens with no current snapshot.

The trace deliberately exercises:

* zero or more snapshots, each ending with ``commit_snapshot`` or
  ``abort_snapshot``;
* possibly-duplicate ``gitlab_project_id`` writes within a single
  snapshot (last-write-wins per Property 9, which Property 10 inherits);
* the full lifecycle ``open → writes/commits/aborts → close → reopen →
  reads`` against the same SQLite-backed file path.

Each example uses a fresh ``tempfile.TemporaryDirectory`` so the
``Knowledge_Store`` opens a brand-new SQLite database and the *same*
file path is later reopened by a second :meth:`KnowledgeStore.open`
call — that is what gives the test its persistence-across-restart
semantics.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
    ProjectProfile,
    SnapshotTrigger,
)

# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

# A fixed timezone-aware ``produced_at`` is used for every generated
# profile so the JSON round-trip through ``profile_json`` (Pydantic
# ``model_dump_json`` -> ``model_validate_json``) is value-stable. The
# point of Property 10 is that *reads return what was written*, not
# that an arbitrary timestamp survives the round-trip — so removing
# this one degree of freedom keeps the strategy small and the
# assertions sharp.
_FIXED_PRODUCED_AT: datetime = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

# Small project-id pool so traces routinely overwrite the same project
# inside a single snapshot (covering the last-write-wins clause that
# Property 10 inherits from Property 9).
_PROJECT_ID_POOL = st.integers(min_value=1, max_value=8)

# 40-char hex strings, the canonical SHA-1 form GitLab returns.
_COMMIT_SHA = st.text(alphabet="0123456789abcdef", min_size=40, max_size=40)

# Repository-relative paths kept to a small alphabet so JSON round-trip
# is unambiguously identity-preserving.
_FULL_PATH = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz/-_",
    min_size=1,
    max_size=20,
)

# Plain ASCII text used for descriptions. The purpose summary
# explicitly excludes the literal string ``"unknown"`` because that
# value forces the model to also carry a non-null
# ``purpose_summary_reason`` (Requirement 3.3) — we do not want the
# strategy to accidentally trip that invariant when it has nothing to
# do with the persistence property under test.
_DESCRIPTION = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz ",
    min_size=1,
    max_size=30,
)
_PURPOSE_SUMMARY = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz ",
    min_size=1,
    max_size=120,
).filter(lambda s: s.strip() != "unknown")


def _abstract_input() -> st.SearchStrategy[AbstractInput]:
    return st.builds(
        AbstractInput,
        category=st.sampled_from(list(AbstractInputCategory)),
        description=_DESCRIPTION,
    )


def _abstract_output() -> st.SearchStrategy[AbstractOutput]:
    return st.builds(
        AbstractOutput,
        category=st.sampled_from(list(AbstractOutputCategory)),
        description=_DESCRIPTION,
    )


@st.composite
def _profile_for(draw: st.DrawFn, project_id: int) -> ProjectProfile:
    """Build a ``ProjectProfile`` whose round-trip through JSON is identity.

    The dependency lists (external services, database tables) are left
    empty here. Each of those carries its own non-empty
    ``source_locations`` invariant whose generation would noticeably
    enlarge the strategy; Property 10 is concerned with ``write →
    commit → close → reopen → read`` equality, not with exhaustively
    populating every Project_Profile section. The shape of those
    sections is exhaustively covered by Property 6.
    """

    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=draw(_FULL_PATH),
        analysis_branch="uat",
        analysis_branch_commit_sha=draw(_COMMIT_SHA),
        produced_at=_FIXED_PRODUCED_AT,
        purpose_summary=draw(_PURPOSE_SUMMARY),
        abstract_inputs=draw(st.lists(_abstract_input(), max_size=3)),
        abstract_outputs=draw(st.lists(_abstract_output(), max_size=3)),
    )


# ---------------------------------------------------------------------------
# Trace generation
# ---------------------------------------------------------------------------


@st.composite
def _snapshot_trace(draw: st.DrawFn) -> dict[str, object]:
    """Generate one snapshot's worth of operations.

    A snapshot is described by the list of ``write_profile`` arguments
    it performs and the single terminal action — ``"commit"`` or
    ``"abort"``. The replay loop below uses these descriptors to drive
    the ``KnowledgeStore`` writer interface.
    """

    n_writes = draw(st.integers(min_value=0, max_value=4))
    writes: list[ProjectProfile] = []
    for _ in range(n_writes):
        pid = draw(_PROJECT_ID_POOL)
        writes.append(draw(_profile_for(pid)))
    action = draw(st.sampled_from(["commit", "abort"]))
    return {"writes": writes, "action": action}


_TRACES = st.lists(_snapshot_trace(), min_size=0, max_size=4)


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(trace=_TRACES)
@settings(max_examples=100)
def test_reads_after_reopen_return_last_committed_snapshot(
    trace: list[dict[str, object]],
) -> None:
    """Property 10: reopen sees the last successfully committed snapshot.

    The test runs in three phases:

    1. **Replay** — open a fresh store at a temp path, run the
       generated sequence of ``begin_snapshot``,
       ``write_profile`` (any number of times, possibly overwriting
       the same ``gitlab_project_id``), then either ``commit_snapshot``
       or ``abort_snapshot``. While replaying we keep an in-memory
       ``last_committed_*`` model of what the next reopen *should*
       return: it advances on commit and is left untouched on abort.
    2. **Close** — close the store. The temp directory is preserved
       so the SQLite file (with WAL) survives.
    3. **Reopen and read** — open a *new* :class:`KnowledgeStore`
       handle against the same path and assert that
       :meth:`get_current_snapshot_id`, :meth:`get_profile` for every
       written project id, and :meth:`list_profiles` all return the
       values from the in-memory model.
    """

    with tempfile.TemporaryDirectory() as tdir:
        db_path = Path(tdir) / "knowledge.db"

        # ``last_committed_snapshot_id`` is ``None`` until the first
        # successful commit; it never goes back to ``None`` afterward
        # (an abort does *not* clear the previously-current snapshot,
        # which is exactly the semantics Property 10 verifies).
        last_committed_snapshot_id: int | None = None
        last_committed_profiles: dict[int, ProjectProfile] = {}

        # ----- Phase 1 + 2: replay then close -----
        store = KnowledgeStore.open(db_path)
        try:
            for snap_op in trace:
                writes: list[ProjectProfile] = snap_op["writes"]  # type: ignore[assignment]
                action: str = snap_op["action"]  # type: ignore[assignment]

                snap_id = store.begin_snapshot(SnapshotTrigger.FULL)
                # Track this snapshot's eventual contents in the order
                # of writes so that last-write-wins per
                # ``(snapshot_id, gitlab_project_id)`` is reflected.
                snap_writes: dict[int, ProjectProfile] = {}
                for profile in writes:
                    store.write_profile(
                        snap_id,
                        profile,
                        produced_at=_FIXED_PRODUCED_AT,
                        commit_sha=profile.analysis_branch_commit_sha,
                    )
                    snap_writes[profile.gitlab_project_id] = profile

                if action == "commit":
                    store.commit_snapshot(snap_id)
                    last_committed_snapshot_id = snap_id
                    last_committed_profiles = snap_writes
                else:
                    store.abort_snapshot(snap_id)
                    # Aborted snapshots intentionally do *not* update
                    # ``last_committed_*``: Property 10 says the
                    # post-reopen reads come from the last
                    # *successfully committed* snapshot.
        finally:
            store.close()

        # ----- Phase 3: reopen and verify -----
        reopened = KnowledgeStore.open(db_path)
        try:
            # The current pointer survives ``close``/``open`` and
            # equals the last commit's snapshot id (or ``None`` if no
            # commit ever happened).
            assert (
                reopened.get_current_snapshot_id() == last_committed_snapshot_id
            )

            # Every written project id resolves to the exact profile
            # value that was written. ``ProjectProfile`` is a frozen
            # Pydantic model, so equality is structural.
            for project_id, expected in last_committed_profiles.items():
                got = reopened.get_profile(project_id)
                assert got is not None, (
                    f"project {project_id} missing after reopen; "
                    f"expected {expected!r}"
                )
                assert got == expected, (
                    f"profile for project {project_id} differs after reopen: "
                    f"got {got!r}, expected {expected!r}"
                )

            # ``list_profiles`` returns exactly the last committed
            # snapshot's profiles, ordered by ``gitlab_project_id``
            # ascending (this ordering is part of the reader's
            # documented contract).
            expected_list = [
                last_committed_profiles[pid]
                for pid in sorted(last_committed_profiles)
            ]
            assert reopened.list_profiles() == expected_list

            # And nothing leaks in from aborted or earlier snapshots.
            seen_ids = {p.gitlab_project_id for p in reopened.list_profiles()}
            assert seen_ids == set(last_committed_profiles)
        finally:
            reopened.close()

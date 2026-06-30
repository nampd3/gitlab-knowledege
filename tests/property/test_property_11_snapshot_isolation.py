# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 11: For all sequences of (begin_snapshot, partial_write*, commit_or_abort) operations and any read issued at any point in the sequence, the read result SHALL equal the result of reading from the snapshot that was current immediately before the most recent unfinished begin_snapshot (if any) or the most recent committed snapshot otherwise; readers SHALL never observe partial writes from an in-progress Ingestion_Job, and the Visualization_Server's diagram inputs SHALL be read from the Knowledge_Store at the moment the HTTP request is handled, with no in-memory caching.
"""Property test for snapshot isolation across all reads.

**Validates Requirements 8.4, 8.5, 14.1, 14.2** (Property 11 in the design).

For every interleaved trace of ``begin_snapshot``, ``write_profile``,
``commit_snapshot``/``abort_snapshot`` and reads
(``get_profile``/``list_profiles``/``get_current_snapshot_id``), the
read results must always equal the contents of the most recently
committed snapshot — never a partial view of an in-progress
``Ingestion_Job``. This is the atomic-pointer-swap guarantee
``Knowledge_Store.commit_snapshot`` offers and that
``Visualization_Server`` and the MCP query tools rely on every time
they handle a request.

The test maintains a parallel **expected-current-snapshot** model:

* ``begin_snapshot`` opens an in-progress snapshot but does **not**
  change the expected-current state.
* ``write_profile(in_progress_id, ...)`` buffers the write into the
  in-progress snapshot's pending writes; expected-current is
  **unchanged**.
* ``commit_snapshot(in_progress_id)`` flushes the in-progress pending
  writes into expected-current (replacing prior visible state for
  ``FULL`` triggers, which is what the writer implements).
* ``abort_snapshot(in_progress_id)`` discards the in-progress writes;
  expected-current is **unchanged**.

Reads are issued at random points throughout the trace **and** an
exhaustive read sweep is performed after every operation, so reads are
exercised at every state-machine position the property covers (before
any commit, during an in-progress snapshot, after abort, after commit).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.knowledge_store import KnowledgeStore
from project_knowledge_mcp.models import (
    ProjectProfile,
    SnapshotTrigger,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Test fixture data
# ---------------------------------------------------------------------------

# A fixed timestamp keeps the search space focused on the snapshot-
# isolation behavior rather than wandering through datetime parsing.
_PRODUCED_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

# A 40-char SHA-1 lookalike satisfies the non-empty
# ``analysis_branch_commit_sha`` invariant on ``ProjectProfile``.
_COMMIT_SHA = "deadbeef" * 5

# A small fixed pool of project ids keeps writes likely to overlap, so
# the trace exercises last-write-wins inside an in-progress snapshot
# *and* across-snapshot replacement on commit.
_PROJECT_IDS: tuple[int, ...] = (1, 2, 3, 4)

# An additional project id that is **never written** is included in the
# read-sweep so reads-of-absent-projects are part of every assertion.
_ABSENT_PROJECT_ID: int = 99


# Operation kinds used in the generated trace. Symbolic strings (rather
# than enum) keep Hypothesis shrinking output readable.
_BEGIN = "begin"
_WRITE = "write"
_COMMIT = "commit"
_ABORT = "abort"
_READ_GET = "read_get"
_READ_LIST = "read_list"
_READ_ID = "read_id"


def _make_profile(project_id: int, marker: str) -> ProjectProfile:
    """Build a minimal valid ``ProjectProfile`` distinguishable by ``marker``.

    ``marker`` is embedded in the ``purpose_summary`` so two writes for
    the same project id but different markers produce non-equal
    profiles. This is what lets the test detect leaked partial writes:
    if a reader were to ever observe an in-progress snapshot, the
    marker on the read profile would differ from the marker on the
    last *committed* write.
    """

    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=f"group/p{project_id}",
        analysis_branch="uat",
        analysis_branch_commit_sha=_COMMIT_SHA,
        produced_at=_PRODUCED_AT,
        purpose_summary=f"profile-{project_id}-{marker}",
    )


# ---------------------------------------------------------------------------
# Operation strategy
# ---------------------------------------------------------------------------


@st.composite
def _operations(draw: st.DrawFn) -> tuple[str, ...]:
    """Draw a single operation token.

    The operation kinds are weighted so the trace covers every leg of
    the state machine without being dominated by reads:

    * ``begin``/``commit``/``abort`` drive the snapshot lifecycle;
    * ``write`` produces partial-write candidates inside an in-progress
      snapshot;
    * ``read_*`` operations probe the reader interface.

    The interpreter resolves operations against the live state, so an
    operation that is invalid in the current state (e.g. ``commit``
    without an open snapshot) is simply skipped — this lets Hypothesis
    explore many interleavings cheaply.
    """

    kind = draw(
        st.sampled_from(
            [
                _BEGIN,
                _WRITE,
                _WRITE,  # weight writes higher to populate snapshots
                _COMMIT,
                _ABORT,
                _READ_GET,
                _READ_LIST,
                _READ_ID,
            ]
        )
    )

    if kind == _WRITE:
        project_id = draw(st.sampled_from(_PROJECT_IDS))
        # The marker is bounded so generated profiles stay small and
        # ``ProjectProfile`` validation (purpose_summary <= 1000 chars)
        # never trips.
        marker = draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll", "Lu", "Nd"),
                ),
                min_size=1,
                max_size=8,
            )
        )
        return (_WRITE, project_id, marker)

    if kind == _READ_GET:
        # Include the never-written id so reads of absent projects are
        # exercised; the expected value in that case is always ``None``.
        project_id = draw(st.sampled_from((*_PROJECT_IDS, _ABSENT_PROJECT_ID)))
        return (_READ_GET, project_id)

    return (kind,)


# ---------------------------------------------------------------------------
# Reader-sweep helper
# ---------------------------------------------------------------------------


def _assert_reads_match_expected(
    store: KnowledgeStore,
    *,
    expected_profiles: dict[int, ProjectProfile],
    expected_snapshot_id: int | None,
) -> None:
    """Assert every reader returns the most recently committed snapshot.

    ``expected_profiles`` is the parallel model's "current visible
    state" — the writes from the most recent committed snapshot, with
    no contribution from any in-progress or aborted snapshot. The
    sweep covers ``get_current_snapshot_id``, ``list_profiles`` and
    ``get_profile`` for every project id (including the never-written
    ``_ABSENT_PROJECT_ID``) so partial-write leakage anywhere in the
    reader interface would surface as a counterexample.
    """

    # Pointer agreement: the reader interface must report exactly the
    # snapshot id that was last committed (or ``None`` before any
    # commit).
    assert store.get_current_snapshot_id() == expected_snapshot_id

    # ``list_profiles`` is documented to return profiles ordered by
    # ``gitlab_project_id`` ascending, which is also the comparison we
    # apply to the expected mapping.
    actual_list = store.list_profiles()
    expected_list = [
        expected_profiles[pid] for pid in sorted(expected_profiles)
    ]
    assert actual_list == expected_list

    # Per-project ``get_profile`` reads. Including ``_ABSENT_PROJECT_ID``
    # makes the test fail if a reader were ever to fabricate a row.
    for project_id in (*_PROJECT_IDS, _ABSENT_PROJECT_ID):
        actual = store.get_profile(project_id)
        expected = expected_profiles.get(project_id)
        assert actual == expected, (
            f"reader leaked an unexpected profile for project_id={project_id}: "
            f"expected {expected!r}, got {actual!r}"
        )


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(trace=st.lists(_operations(), min_size=1, max_size=40))
@settings(max_examples=100)
def test_snapshot_isolation_across_all_reads(  # noqa: PLR0912
    tmp_path_factory: pytest.TempPathFactory,
    trace: list[tuple[str, ...]],
) -> None:
    """Property 11: reads always reflect the most recently committed snapshot."""

    db_path: Path = tmp_path_factory.mktemp("snapshot_isolation") / "store.db"
    store = KnowledgeStore.open(db_path)
    try:
        # Parallel expected-state model. ``expected_profiles`` is the
        # set of profiles that should be visible to readers right now;
        # ``expected_snapshot_id`` is the snapshot id readers should
        # observe via ``get_current_snapshot_id``. Both are advanced
        # only on a successful ``commit_snapshot``.
        expected_profiles: dict[int, ProjectProfile] = {}
        expected_snapshot_id: int | None = None

        # In-progress snapshot model. ``in_progress_id`` is the SQLite
        # snapshot id of the currently-open Ingestion_Job (``None``
        # when no job is open). ``in_progress_writes`` accumulates the
        # writes that ``commit_snapshot`` will atomically promote, or
        # that ``abort_snapshot`` will discard.
        in_progress_id: int | None = None
        in_progress_writes: dict[int, ProjectProfile] = {}

        # Initial sweep: a freshly-opened store has no committed
        # snapshot, so every reader must report the empty state.
        _assert_reads_match_expected(
            store,
            expected_profiles=expected_profiles,
            expected_snapshot_id=expected_snapshot_id,
        )

        for op in trace:
            kind = op[0]

            if kind == _BEGIN:
                # Only honor the begin if no snapshot is currently
                # in-progress. The design's Ingestion_Coordinator
                # enforces single-flight at the application layer, so
                # the test mirrors that invariant rather than asking
                # the store to defend against double-begin.
                if in_progress_id is None:
                    in_progress_id = store.begin_snapshot(SnapshotTrigger.FULL)
                    in_progress_writes = {}

            elif kind == _WRITE:
                if in_progress_id is not None:
                    project_id, marker = op[1], op[2]
                    profile = _make_profile(project_id, marker)
                    store.write_profile(
                        in_progress_id,
                        profile,
                        produced_at=_PRODUCED_AT,
                        commit_sha=_COMMIT_SHA,
                    )
                    # Last-write-wins inside the in-progress snapshot;
                    # this matches the writer's
                    # ``ON CONFLICT DO UPDATE`` clause.
                    in_progress_writes[project_id] = profile

            elif kind == _COMMIT:
                if in_progress_id is not None:
                    store.commit_snapshot(in_progress_id)
                    # Atomic-pointer-swap moment: expected-current now
                    # reflects exactly the writes accumulated under the
                    # just-committed snapshot. ``FULL`` triggers do not
                    # inherit prior snapshot rows, so the new committed
                    # state is the in-progress writes alone — matching
                    # the writer's behavior in
                    # ``KnowledgeStore.commit_snapshot``.
                    expected_profiles = dict(in_progress_writes)
                    expected_snapshot_id = in_progress_id
                    in_progress_id = None
                    in_progress_writes = {}

            elif kind == _ABORT:
                if in_progress_id is not None:
                    store.abort_snapshot(in_progress_id)
                    # Aborted writes never become visible. The
                    # expected-current state is unchanged: readers
                    # continue to see whatever the previously committed
                    # snapshot held.
                    in_progress_id = None
                    in_progress_writes = {}

            elif kind == _READ_GET:
                project_id = op[1]
                actual = store.get_profile(project_id)
                expected = expected_profiles.get(project_id)
                assert actual == expected, (
                    f"in-trace read leaked partial-write for "
                    f"project_id={project_id}: expected {expected!r}, "
                    f"got {actual!r}"
                )

            elif kind == _READ_LIST:
                actual_list = store.list_profiles()
                expected_list = [
                    expected_profiles[pid] for pid in sorted(expected_profiles)
                ]
                assert actual_list == expected_list

            elif kind == _READ_ID:
                assert store.get_current_snapshot_id() == expected_snapshot_id

            # After every operation — including writes that *should
            # not* be visible — re-run the full reader sweep. This is
            # the strongest form of the snapshot-isolation assertion:
            # at every point in the trace, every reader endpoint
            # returns the most recently committed snapshot's contents
            # and never observes the in-progress snapshot's writes.
            _assert_reads_match_expected(
                store,
                expected_profiles=expected_profiles,
                expected_snapshot_id=expected_snapshot_id,
            )
    finally:
        store.close()

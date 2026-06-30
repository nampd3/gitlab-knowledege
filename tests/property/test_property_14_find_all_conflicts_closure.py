# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 14: For all sets of Project_Profiles, Conflict_Detector.find_all_conflicts(profiles) SHALL return exactly the set of unordered pairs {a, b} (with a != b) such that classify_pair(a, b).kind == "conflict", with each unordered pair represented at most once.
"""Property test for ``find_all_conflicts`` symmetric closure.

**Validates Requirement 9.2** (Property 14 in the design).

For every list of ``ProjectProfile`` records, ``find_all_conflicts`` must
return exactly the set of unordered project-id pairs ``{a, b}`` (with
``a != b``) for which ``classify_pair(a, b).kind == ConflictKind.CONFLICT``,
with each unordered pair represented at most once.

The property is verified by cross-checking ``find_all_conflicts`` against a
straightforward brute-force enumeration of ``classify_pair`` over all
unordered pairs of distinct profiles (the ground truth). The strategy
generates 0-6 ``ProjectProfile`` records per example with distinct
``gitlab_project_id``s and purpose summaries drawn from a curated palette
that spans every classification branch of ``classify_pair``:

* identical / paraphrased "same primary responsibility" summaries that
  trigger the substantial-overlap conflict branch;
* matched-responsibility ownership/disclaimer pairs that trigger the
  contradictory-ownership conflict branch;
* clearly different responsibilities that produce ``no_conflict``;
* the canonical ``"unknown"`` summary that forces ``indeterminate``.

This palette guarantees that a non-trivial fraction of generated examples
contain at least one true conflict, so the test exercises both the
"include" path (every brute-force conflict appears in the result) and the
"exclude" path (no spurious pair is included).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.conflict_detector import (
    ConflictPair,
    classify_pair,
    find_all_conflicts,
)
from project_knowledge_mcp.models import (
    INSUFFICIENT_SOURCE_MATERIAL_REASON,
    UNKNOWN_PURPOSE_SUMMARY,
    ConflictKind,
    ProjectProfile,
)

# ---------------------------------------------------------------------------
# Fixed metadata shared across generated profiles
# ---------------------------------------------------------------------------

# A single fixed timestamp and commit SHA keep generated profiles cheap to
# build; ``classify_pair`` only inspects ``purpose_summary`` (and
# ``full_path`` for the indeterminate justification), so these fields are
# constants without affecting coverage of Property 14.
_PRODUCED_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
_COMMIT_SHA = "deadbeef" * 5  # 40-char SHA-1 lookalike
_ANALYSIS_BRANCH = "uat"

# ---------------------------------------------------------------------------
# Curated purpose-summary palette
# ---------------------------------------------------------------------------

# Each entry is a (summary, reason) pair. ``reason`` is non-None only for
# the canonical "unknown" entry, which is required by ``ProjectProfile``'s
# model validator (Requirement 3.3). The palette deliberately includes
# multiple summaries per responsibility/stance combination so that random
# pairings exercise:
#
#   * identical canonical key + high token overlap -> CONFLICT (overlap)
#   * identical canonical key + opposite ownership -> CONFLICT (contradiction)
#   * different canonical keys                     -> NO_CONFLICT
#   * either summary == "unknown"                  -> INDETERMINATE
_PURPOSE_PALETTE: tuple[tuple[str, str | None], ...] = (
    # --- "user authentication" responsibility, owner stance ---------------
    ("Owns the user authentication flow for the platform.", None),
    ("Manages user authentication and session lifecycle for the platform.", None),
    # --- "user authentication" responsibility, delegator stance -----------
    ("Is not responsible for user authentication; delegates to the SSO provider.", None),
    ("Delegates user authentication to the upstream identity service.", None),
    # --- "billing pipeline" responsibility, owner stance ------------------
    ("Owns the billing pipeline and invoicing workflow.", None),
    ("Manages the billing pipeline and customer invoicing workflow.", None),
    # --- "billing pipeline" responsibility, delegator stance --------------
    ("Does not own the billing pipeline; forwards events to the billing service.", None),
    # --- "order processing" responsibility, owner stance ------------------
    ("Owns the order processing pipeline for the storefront.", None),
    ("Handles the order processing pipeline for the storefront end to end.", None),
    # --- distinct responsibilities (no overlap) ---------------------------
    ("Renders frontend dashboards for internal operators.", None),
    ("Aggregates analytics events into the warehouse for reporting.", None),
    ("Provides a CLI for migrating legacy database schemas.", None),
    ("Serves static documentation pages for the developer portal.", None),
    # --- canonical "unknown" ---------------------------------------------
    (UNKNOWN_PURPOSE_SUMMARY, INSUFFICIENT_SOURCE_MATERIAL_REASON),
)


# ---------------------------------------------------------------------------
# Profile and profile-list strategies
# ---------------------------------------------------------------------------


@st.composite
def _project_profile(draw: st.DrawFn, *, project_id: int) -> ProjectProfile:
    """Build a single ``ProjectProfile`` with the given ``gitlab_project_id``.

    The ``full_path`` is derived from the project id so distinct ids yield
    distinct human-readable paths (which feed the indeterminate justification
    when applicable). The ``purpose_summary`` is drawn from the curated
    palette so the generated profile actually exercises one of the
    classification branches.
    """

    summary, reason = draw(st.sampled_from(_PURPOSE_PALETTE))
    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=f"group/project-{project_id}",
        analysis_branch=_ANALYSIS_BRANCH,
        analysis_branch_commit_sha=_COMMIT_SHA,
        produced_at=_PRODUCED_AT,
        purpose_summary=summary,
        purpose_summary_reason=reason,
    )


@st.composite
def _profile_list(draw: st.DrawFn) -> list[ProjectProfile]:
    """Build a list of 0-6 ``ProjectProfile``s with distinct ids.

    Sampling distinct ids first and then attaching one summary per id keeps
    the generator cheap and guarantees the brute-force ground truth has a
    well-defined unordered-pair key set (one per ``(min(id_a, id_b),
    max(id_a, id_b))``).
    """

    project_ids = draw(
        st.lists(
            st.integers(min_value=1, max_value=1_000_000),
            min_size=0,
            max_size=6,
            unique=True,
        )
    )
    return [draw(_project_profile(project_id=pid)) for pid in project_ids]


# ---------------------------------------------------------------------------
# Brute-force ground truth
# ---------------------------------------------------------------------------


def _ground_truth_pairs(
    profiles: list[ProjectProfile],
) -> set[tuple[int, int]]:
    """Compute the set of unordered conflict pairs by exhaustive enumeration.

    For every pair of distinct positions ``(i, j)`` with ``i < j``, invoke
    ``classify_pair(profiles[i], profiles[j])`` and, if the result is a
    ``CONFLICT``, record the unordered key ``(min(id_i, id_j), max(...))``.
    Same-id pairs are skipped to mirror ``find_all_conflicts``'s documented
    handling of malformed inputs (a project cannot conflict with itself).
    """

    pairs: set[tuple[int, int]] = set()
    n = len(profiles)
    for i in range(n):
        for j in range(i + 1, n):
            id_i = profiles[i].gitlab_project_id
            id_j = profiles[j].gitlab_project_id
            if id_i == id_j:
                continue
            first, second = (
                (profiles[i], profiles[j]) if id_i < id_j else (profiles[j], profiles[i])
            )
            if classify_pair(first, second).kind is ConflictKind.CONFLICT:
                pairs.add((first.gitlab_project_id, second.gitlab_project_id))
    return pairs


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(profiles=_profile_list())
@settings(max_examples=100)
def test_find_all_conflicts_equals_brute_force_symmetric_closure(
    profiles: list[ProjectProfile],
) -> None:
    """Property 14: ``find_all_conflicts`` is the symmetric closure of ``classify_pair``."""

    result: list[ConflictPair] = find_all_conflicts(profiles)

    # Each result entry is a well-formed ``ConflictPair`` with strictly
    # ascending ids; this is the canonical representation of the
    # unordered pair {a, b}.
    for pair in result:
        assert pair.project_id_a < pair.project_id_b, (
            f"ConflictPair must store ids in ascending order, got {pair}"
        )

    result_keys = [(p.project_id_a, p.project_id_b) for p in result]

    # (1) Each unordered pair is represented at most once.
    assert len(result_keys) == len(set(result_keys)), (
        f"find_all_conflicts returned duplicate unordered pairs: {result_keys}"
    )

    # (2) The set of unordered pairs from ``find_all_conflicts`` equals the
    # brute-force ground truth derived from ``classify_pair``.
    assert set(result_keys) == _ground_truth_pairs(profiles), (
        "find_all_conflicts disagrees with brute-force classify_pair "
        f"enumeration; result={set(result_keys)!r}, "
        f"expected={_ground_truth_pairs(profiles)!r}"
    )

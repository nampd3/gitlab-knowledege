# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 13: For all pairs of Project_Profiles, Conflict_Detector.classify_pair SHALL return a result whose kind is one of {conflict, no_conflict, indeterminate} and whose justification is a non-empty string referencing the purpose summaries that led to the classification.
"""Property test for the shape of ``classify_pair`` results.

**Validates Requirement 9.1** (Property 13 in the design).

For every pair of ``ProjectProfile`` records (both summaries known, one
unknown, or both unknown), ``Conflict_Detector.classify_pair`` must
produce a ``ConflictResult`` whose:

1. ``kind`` is one of ``CONFLICT``, ``NO_CONFLICT``, ``INDETERMINATE``;
2. ``justification`` is a non-empty string;
3. ``justification`` references the purpose summaries (or, in the
   ``INDETERMINATE`` branch, the GitLab paths of the projects whose
   summary is ``"unknown"``) that drove the classification.

The third clause is the essential "shape" guarantee from Property 13:
the justification must let a downstream operator tie the decision back
to the source material. The implementation fulfils this by embedding a
bounded excerpt of each known purpose summary (or, for ``INDETERMINATE``,
each unknown project's ``full_path``) in the justification string.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.conflict_detector import classify_pair
from project_knowledge_mcp.models import (
    INSUFFICIENT_SOURCE_MATERIAL_REASON,
    UNKNOWN_PURPOSE_SUMMARY,
    ConflictKind,
    ProjectProfile,
)

# ---------------------------------------------------------------------------
# Test fixture data
# ---------------------------------------------------------------------------

# A fixed timestamp keeps generated profiles trivially valid without
# expanding the search space along an axis Property 13 does not care about.
_PRODUCED_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

# A 40-char SHA-1 lookalike satisfies the non-empty
# analysis_branch_commit_sha invariant on ``ProjectProfile``.
_COMMIT_SHA = "deadbeef" * 5

# Allowed kinds — the closed set Property 13 requires ``classify_pair`` to
# inhabit.
_ALLOWED_KINDS: frozenset[ConflictKind] = frozenset(
    {ConflictKind.CONFLICT, ConflictKind.NO_CONFLICT, ConflictKind.INDETERMINATE}
)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Lowercase alphabetic words. Confining the alphabet to ASCII letters
# keeps generated summaries short, well-formed, and free of internal
# whitespace anomalies, so that the implementation's whitespace-collapsing
# excerpt of a summary equals the summary itself for the lengths we
# generate.
_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8)

# A "known" purpose summary is a single-space-joined sequence of words,
# capped well below the 80-character justification excerpt limit so the
# embedded excerpt is verbatim equal to the original summary.
_KNOWN_PURPOSE_SUMMARY = st.lists(_WORD, min_size=1, max_size=6).map(" ".join)

# A "GitLab path" is a single ``group/project`` segment of lowercase
# letters and digits. Two profiles in the same generated pair are allowed
# to share a path; ``classify_pair`` does not require distinct paths.
_FULL_PATH = st.from_regex(
    r"[a-z][a-z0-9]{1,12}/[a-z][a-z0-9]{1,12}",
    fullmatch=True,
)


@st.composite
def _project_profile(draw: st.DrawFn) -> ProjectProfile:
    """Generate a minimal valid ``ProjectProfile`` for Property 13.

    Each draw chooses, with ~50% probability, between a known purpose
    summary (a short space-joined word sequence) and the canonical
    ``"unknown"`` value paired with the documented insufficient-source
    reason. The other fields are populated with fixed valid values so
    every drawn profile satisfies the model's invariants by construction.
    """
    project_id = draw(st.integers(min_value=1, max_value=10_000_000))
    full_path = draw(_FULL_PATH)

    if draw(st.booleans()):
        purpose_summary = UNKNOWN_PURPOSE_SUMMARY
        purpose_summary_reason: str | None = INSUFFICIENT_SOURCE_MATERIAL_REASON
    else:
        purpose_summary = draw(_KNOWN_PURPOSE_SUMMARY)
        purpose_summary_reason = None

    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha=_COMMIT_SHA,
        produced_at=_PRODUCED_AT,
        purpose_summary=purpose_summary,
        purpose_summary_reason=purpose_summary_reason,
    )


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(profile_a=_project_profile(), profile_b=_project_profile())
@settings(max_examples=100)
def test_classify_pair_result_shape(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
) -> None:
    """Property 13: classify_pair returns a well-shaped result."""
    result = classify_pair(profile_a, profile_b)

    # (1) kind ∈ {conflict, no_conflict, indeterminate}.
    assert result.kind in _ALLOWED_KINDS, (
        f"unexpected kind {result.kind!r}; expected one of {_ALLOWED_KINDS}"
    )

    # (2) justification is a non-empty string.
    assert isinstance(result.justification, str)
    assert result.justification != ""

    # (3) justification references the purpose summaries that drove the
    # classification. The implementation distinguishes two cases:
    a_unknown = profile_a.purpose_summary == UNKNOWN_PURPOSE_SUMMARY
    b_unknown = profile_b.purpose_summary == UNKNOWN_PURPOSE_SUMMARY

    if a_unknown or b_unknown:
        # INDETERMINATE branch: the result must be INDETERMINATE, the
        # justification must explain *why* (the word "unknown"), and it
        # must name every project whose purpose summary is unknown by
        # its full_path so the operator can locate the source.
        assert result.kind is ConflictKind.INDETERMINATE
        assert "unknown" in result.justification
        if a_unknown:
            assert profile_a.full_path in result.justification, (
                f"justification does not name unknown project "
                f"{profile_a.full_path!r}: {result.justification!r}"
            )
        if b_unknown:
            assert profile_b.full_path in result.justification, (
                f"justification does not name unknown project "
                f"{profile_b.full_path!r}: {result.justification!r}"
            )
    else:
        # Both summaries are known. The justification must embed both
        # purpose summaries (the excerpt mechanism leaves short summaries
        # like the ones we generate verbatim) so the basis for the
        # CONFLICT / NO_CONFLICT decision is traceable.
        assert result.kind in {ConflictKind.CONFLICT, ConflictKind.NO_CONFLICT}
        assert profile_a.purpose_summary in result.justification, (
            f"justification does not reference profile_a's purpose summary "
            f"{profile_a.purpose_summary!r}: {result.justification!r}"
        )
        assert profile_b.purpose_summary in result.justification, (
            f"justification does not reference profile_b's purpose summary "
            f"{profile_b.purpose_summary!r}: {result.justification!r}"
        )

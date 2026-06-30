# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 16: For all pairs of Project_Profiles where at least one purpose summary equals "unknown", classify_pair(a, b) SHALL return a result with kind == "indeterminate" and a justification stating that the purpose summary is unknown for the named project(s).
"""Property test that ``"unknown"`` purpose summaries force ``INDETERMINATE``.

**Validates Requirement 9.4** (Property 16 in the design).

For every pair of ``ProjectProfile`` records in which *at least one*
``purpose_summary`` equals the canonical ``UNKNOWN_PURPOSE_SUMMARY``,
``Conflict_Detector.classify_pair`` must:

1. return a ``ConflictResult`` with ``kind == ConflictKind.INDETERMINATE``;
2. produce a justification that mentions the word ``"unknown"`` (so a
   downstream operator can tell *why* the pair was indeterminate); and
3. name every project whose ``purpose_summary`` is unknown by its
   ``full_path`` (so the operator can locate the source profile that
   needs better source material).

This complements Property 15 (which forbids ``CONFLICT`` outcomes when
either summary is unknown) by asserting the positive consequence: any
unknown summary forces ``INDETERMINATE``.
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
# expanding the search space along an axis Property 16 does not care about.
_PRODUCED_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

# A 40-char SHA-1 lookalike satisfies the non-empty
# analysis_branch_commit_sha invariant on ``ProjectProfile``.
_COMMIT_SHA = "deadbeef" * 5


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Lowercase alphabetic words. Confining the alphabet to ASCII letters
# keeps generated summaries short, well-formed, and free of internal
# whitespace anomalies.
_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8)

# A "known" purpose summary is a single-space-joined sequence of words,
# capped well below the 1000-character ``PURPOSE_SUMMARY_MAX_LEN`` limit.
# By construction the joined string never equals the canonical
# ``UNKNOWN_PURPOSE_SUMMARY`` (each token has length >= 2).
_KNOWN_PURPOSE_SUMMARY = st.lists(_WORD, min_size=1, max_size=6).map(" ".join)

# A "GitLab path" is a single ``group/project`` segment of lowercase
# letters and digits. Two profiles in the same generated pair are allowed
# to share a path; ``classify_pair`` does not require distinct paths and
# Requirement 9.4 says nothing about path uniqueness.
_FULL_PATH = st.from_regex(
    r"[a-z][a-z0-9]{1,12}/[a-z][a-z0-9]{1,12}",
    fullmatch=True,
)


def _build_profile(
    *,
    project_id: int,
    full_path: str,
    purpose_summary: str,
    purpose_summary_reason: str | None,
) -> ProjectProfile:
    """Construct a minimal valid ``ProjectProfile`` from the drawn values."""
    return ProjectProfile(
        gitlab_project_id=project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha=_COMMIT_SHA,
        produced_at=_PRODUCED_AT,
        purpose_summary=purpose_summary,
        purpose_summary_reason=purpose_summary_reason,
    )


@st.composite
def _unknown_pair(draw: st.DrawFn) -> tuple[ProjectProfile, ProjectProfile]:
    """Generate a pair of profiles where AT LEAST ONE summary is ``"unknown"``.

    Three sub-cases are sampled with roughly equal probability:

    * only ``profile_a`` is unknown (``profile_b`` carries a known summary);
    * only ``profile_b`` is unknown (``profile_a`` carries a known summary);
    * both profiles are unknown.

    In every case the unknown profile carries the documented
    ``INSUFFICIENT_SOURCE_MATERIAL_REASON`` so the model's invariant
    (Requirement 3.3) is satisfied by construction.
    """
    which = draw(st.sampled_from(("a_only", "b_only", "both")))

    a_unknown = which in ("a_only", "both")
    b_unknown = which in ("b_only", "both")

    profile_a = _build_profile(
        project_id=draw(st.integers(min_value=1, max_value=10_000_000)),
        full_path=draw(_FULL_PATH),
        purpose_summary=(
            UNKNOWN_PURPOSE_SUMMARY if a_unknown else draw(_KNOWN_PURPOSE_SUMMARY)
        ),
        purpose_summary_reason=(
            INSUFFICIENT_SOURCE_MATERIAL_REASON if a_unknown else None
        ),
    )
    profile_b = _build_profile(
        project_id=draw(st.integers(min_value=1, max_value=10_000_000)),
        full_path=draw(_FULL_PATH),
        purpose_summary=(
            UNKNOWN_PURPOSE_SUMMARY if b_unknown else draw(_KNOWN_PURPOSE_SUMMARY)
        ),
        purpose_summary_reason=(
            INSUFFICIENT_SOURCE_MATERIAL_REASON if b_unknown else None
        ),
    )
    return profile_a, profile_b


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(pair=_unknown_pair())
@settings(max_examples=100)
def test_unknown_purpose_forces_indeterminate(
    pair: tuple[ProjectProfile, ProjectProfile],
) -> None:
    """Property 16: an unknown purpose summary forces an INDETERMINATE result."""
    profile_a, profile_b = pair

    # Sanity-check the generator: at least one summary must be unknown,
    # otherwise the property's antecedent isn't met and the test would be
    # vacuously true. This guards against accidental generator drift.
    a_unknown = profile_a.purpose_summary == UNKNOWN_PURPOSE_SUMMARY
    b_unknown = profile_b.purpose_summary == UNKNOWN_PURPOSE_SUMMARY
    assert a_unknown or b_unknown

    result = classify_pair(profile_a, profile_b)

    # (1) kind must be INDETERMINATE.
    assert result.kind is ConflictKind.INDETERMINATE, (
        f"expected INDETERMINATE, got {result.kind!r}; "
        f"justification={result.justification!r}"
    )

    # (2) The justification must mention the word "unknown" so the
    # operator can see *why* the pair is indeterminate.
    assert "unknown" in result.justification, (
        f"justification does not mention 'unknown': {result.justification!r}"
    )

    # (3) The justification must name every project whose summary is
    # unknown by its full_path so the operator can locate the source
    # profile (Requirement 9.4: "named project(s)").
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

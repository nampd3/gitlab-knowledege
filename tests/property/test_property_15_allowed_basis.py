# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 15: For all pairs of Project_Profiles where classify_pair(a, b).kind == "conflict", neither a.purpose_summary nor b.purpose_summary equals "unknown", and the documented justification SHALL describe either substantially the same primary responsibility or contradictory ownership of the same responsibility; the classifier SHALL never return conflict on any other basis.
"""Property test for the allowed-basis classification rule.

**Validates Requirement 9.3** (Property 15 in the design).

For every pair of ``ProjectProfile`` records, whenever
``Conflict_Detector.classify_pair(a, b).kind == ConflictKind.CONFLICT``:

1. neither ``a.purpose_summary`` nor ``b.purpose_summary`` equals the
   canonical ``UNKNOWN_PURPOSE_SUMMARY`` (``"unknown"``); and
2. the ``justification`` describes one of the two -- and only two --
   bases on which a conflict may be reported: either *substantially the
   same primary responsibility* (the overlap branch) or *contradictory
   ownership of the same responsibility* (the contradiction branch).

The implementation embeds one of two canonical sub-phrases verbatim in
its justification, depending on which branch fired:

* overlap branch -> ``"substantially the same primary responsibility"``
* contradiction branch -> ``"contradictory ownership"``

This test asserts that at least one of those canonical sub-phrases is
present in the justification of every ``CONFLICT`` result, which is the
operational form of "the documented justification describes either of
the two allowed bases" given the implementation's public phrasing.

The strategy generates pairs that mix:

* paraphrased descriptions of the same canonical responsibility (likely
  to fire the overlap branch);
* owner / delegator descriptions of the same canonical responsibility
  (likely to fire the contradiction branch);
* descriptions of clearly different responsibilities (should never
  produce a ``CONFLICT``);
* one or both summaries set to the canonical ``"unknown"`` value (must
  never produce a ``CONFLICT``, exercising clause (1) above).
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
# Fixed metadata shared across generated profiles
# ---------------------------------------------------------------------------

# A single fixed timestamp and SHA keep generated profiles cheap; the
# classifier only inspects ``purpose_summary`` (and ``full_path`` for the
# indeterminate justification), so these fields are constants.
_PRODUCED_AT = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
_COMMIT_SHA = "deadbeef" * 5  # 40-char SHA-1 lookalike
_ANALYSIS_BRANCH = "uat"

# ---------------------------------------------------------------------------
# Canonical phrases that the implementation is documented to use in the
# justification of each allowed conflict basis.
# ---------------------------------------------------------------------------

# Overlap branch ("substantially the same primary responsibility").
_OVERLAP_PHRASE = "substantially the same primary responsibility"

# Contradiction branch ("contradictory ownership of the same responsibility").
# The implementation embeds the canonical responsibility key between
# ``ownership of`` and the closing quote, so we match on the leading
# fixed sub-phrase ``contradictory ownership``.
_CONTRADICTION_PHRASE = "contradictory ownership"

# ---------------------------------------------------------------------------
# Curated purpose-summary palette
# ---------------------------------------------------------------------------

# Each entry is a (summary, reason) pair. ``reason`` is non-None only for
# the canonical "unknown" entry, which is required by ``ProjectProfile``'s
# model validator (Requirement 3.3). The palette includes paraphrased
# descriptions of the same responsibility (overlap candidates), owner /
# delegator descriptions of the same responsibility (contradiction
# candidates), unrelated responsibilities (no-conflict candidates), and
# the canonical "unknown" value so a non-trivial fraction of generated
# pairs lands in each classification branch.
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
# Profile strategy
# ---------------------------------------------------------------------------


@st.composite
def _project_profile(draw: st.DrawFn) -> ProjectProfile:
    """Generate a single ``ProjectProfile`` drawn from the curated palette.

    The two profiles in a generated pair are produced independently, so
    every combination -- both summaries known, exactly one unknown, both
    unknown, paraphrases of the same responsibility, owner vs. delegator
    on the same responsibility, and unrelated responsibilities -- arises
    with non-negligible probability across the 100 examples Hypothesis
    runs.
    """

    project_id = draw(st.integers(min_value=1, max_value=1_000_000))
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


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(profile_a=_project_profile(), profile_b=_project_profile())
@settings(max_examples=100)
def test_conflict_only_on_allowed_basis(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
) -> None:
    """Property 15: ``CONFLICT`` is only reported on the two allowed bases."""

    result = classify_pair(profile_a, profile_b)

    # Property 15 only constrains the CONFLICT branch; INDETERMINATE and
    # NO_CONFLICT results are out of scope here.
    if result.kind is not ConflictKind.CONFLICT:
        return

    # (1) Neither summary is the canonical "unknown" value. An "unknown"
    # summary forces INDETERMINATE per Requirement 9.4, so a CONFLICT
    # result with an "unknown" summary on either side would violate
    # Property 15.
    assert profile_a.purpose_summary != UNKNOWN_PURPOSE_SUMMARY, (
        "classify_pair returned CONFLICT but profile_a.purpose_summary is "
        f"{UNKNOWN_PURPOSE_SUMMARY!r}; an unknown summary must force "
        "INDETERMINATE per Requirement 9.4."
    )
    assert profile_b.purpose_summary != UNKNOWN_PURPOSE_SUMMARY, (
        "classify_pair returned CONFLICT but profile_b.purpose_summary is "
        f"{UNKNOWN_PURPOSE_SUMMARY!r}; an unknown summary must force "
        "INDETERMINATE per Requirement 9.4."
    )

    # (2) The justification fits the allowed schema: it describes either
    # "substantially the same primary responsibility" (overlap branch) or
    # "contradictory ownership of the same responsibility" (contradiction
    # branch). At least one of the two canonical sub-phrases the
    # implementation is documented to embed must appear verbatim.
    justification = result.justification
    has_overlap = _OVERLAP_PHRASE in justification
    has_contradiction = _CONTRADICTION_PHRASE in justification
    assert has_overlap or has_contradiction, (
        "classify_pair returned CONFLICT but its justification does not "
        "describe either of the two allowed bases (substantially the same "
        "primary responsibility, or contradictory ownership of the same "
        f"responsibility): {justification!r}"
    )

"""Conflict_Detector: classifies pairs of Project_Profiles for Purpose_Conflict.

This module implements the deterministic heuristic specified in the design's
``Conflict_Detector`` section. ``classify_pair`` compares the
``purpose_summary`` fields of two ``ProjectProfile`` records and returns a
``ConflictResult`` whose ``kind`` is one of:

* ``ConflictKind.INDETERMINATE`` -- when either purpose summary equals
  ``UNKNOWN_PURPOSE_SUMMARY``. The justification names the project (or both
  projects) whose summary is unknown (Requirement 9.4).
* ``ConflictKind.CONFLICT`` -- when the two summaries describe substantially
  the same primary responsibility (token overlap above
  ``SUBSTANTIAL_OVERLAP_THRESHOLD`` *and* identical canonical responsibility
  keys), or assert contradictory ownership of the same responsibility
  (identical canonical responsibility keys *and* mutually exclusive
  ownership stances). These are the only two bases on which a conflict may
  be reported (Requirement 9.3).
* ``ConflictKind.NO_CONFLICT`` -- otherwise.

In every case the ``justification`` string is non-empty and references the
purpose summaries that drove the classification (Requirement 9.1). The
heuristic is deterministic: identical inputs always yield identical results
across runs and across processes.

``find_all_conflicts`` computes the symmetric closure of ``classify_pair``
across a set of profiles: for every unordered pair ``{a, b}`` (``a != b``)
where ``classify_pair(a, b).kind == "conflict"``, it returns one
``ConflictPair`` record carrying the two project IDs in canonical
ascending order and the justification produced by the classifier. Each
unordered pair appears at most once in the result (Requirement 9.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from project_knowledge_mcp.models import (
    UNKNOWN_PURPOSE_SUMMARY,
    ConflictKind,
    ConflictResult,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_knowledge_mcp.models import ProjectProfile


# ---------------------------------------------------------------------------
# Heuristic configuration
# ---------------------------------------------------------------------------

#: Minimum Jaccard similarity of cleaned token sets that qualifies as
#: "substantial overlap" of primary responsibility (paired with identical
#: canonical responsibility keys). Set high enough that incidental
#: vocabulary overlap cannot trigger a conflict on its own, low enough to
#: tolerate paraphrasing of the same responsibility.
SUBSTANTIAL_OVERLAP_THRESHOLD: float = 0.6

#: Maximum length of the purpose-summary excerpt embedded in a
#: ``ConflictResult.justification``. Keeps justifications bounded even when
#: source summaries approach the ``PURPOSE_SUMMARY_MAX_LEN`` ceiling.
JUSTIFICATION_EXCERPT_MAX_LEN: int = 80

#: Minimum token length for which a trailing ``s`` is stripped during the
#: simple plural-to-singular normalization in ``_tokenize``. Keeps short
#: tokens such as ``"is"`` and ``"as"`` from being mangled.
_SINGULARIZE_MIN_LEN: int = 4

#: Common English function words removed during tokenization so that the
#: canonical responsibility key is drawn from content tokens. Includes
#: modal verbs (``"must"``, ``"can"``, ...) and pronouns (``"it"``,
#: ``"they"``, ...) so that an ownership-marker phrase like
#: ``"must not"`` does not bleed into the canonical key while still being
#: visible to ``_ownership_stance`` (which inspects the original summary).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at",
        "be", "been", "being", "but", "by",
        "can", "could",
        "did", "do", "does",
        "for", "from",
        "had", "has", "have", "he", "her", "hers", "him", "his", "how",
        "i", "in", "into", "is", "it", "its",
        "may", "me", "might", "mine", "must", "my",
        "no", "nor", "not",
        "of", "on", "or", "ought", "our", "ours", "out", "over",
        "shall", "she", "should", "so",
        "than", "that", "the", "their", "theirs", "them", "they",
        "this", "those", "to", "too",
        "under", "up", "us",
        "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "whose", "why", "will", "with", "within", "without",
        "would",
        "you", "your", "yours",
    }
)

#: Phrases that mark a summary as asserting ownership of its responsibility.
#: Matched as case-insensitive substrings of the whitespace-normalized
#: summary by ``_ownership_stance``. Order is irrelevant for correctness;
#: the list is kept sorted for readability.
_POSITIVE_OWNERSHIP_PHRASES: tuple[str, ...] = (
    "controls",
    "handles",
    "is responsible for",
    "is the owner of",
    "is the source of truth for",
    "is the system of record for",
    "manages",
    "owner of",
    "owns",
)

#: Phrases that mark a summary as disclaiming ownership of its
#: responsibility (delegating it elsewhere or explicitly forbidding the
#: work). Matched the same way as ``_POSITIVE_OWNERSHIP_PHRASES``. When a
#: summary matches phrases from both lists, the negative stance dominates
#: so that "delegates X but is responsible for Y" is recognized as a
#: delegation overall.
_NEGATIVE_OWNERSHIP_PHRASES: tuple[str, ...] = (
    "delegates",
    "does not handle",
    "does not manage",
    "does not own",
    "is not responsible for",
    "is not the owner of",
    "must not",
)

#: Multi-word ownership-marker phrases stripped from a summary before
#: tokenization for canonical-key extraction, so that markers like
#: ``"is responsible for"`` do not become part of the canonical bigram.
#: Single-word markers (``"owns"``, ``"manages"``, ...) are absorbed by
#: the same normalization via the per-word logic in ``_clean_for_key``.
_OWNERSHIP_MARKER_PHRASES: tuple[str, ...] = tuple(
    sorted(
        {
            *_POSITIVE_OWNERSHIP_PHRASES,
            *_NEGATIVE_OWNERSHIP_PHRASES,
        },
        key=len,
        reverse=True,  # strip longest phrases first
    )
)

#: Single ownership-marker tokens whose presence would bias the canonical
#: key. Removed *after* the multi-word phrases above have been stripped.
_OWNERSHIP_MARKER_TOKENS: frozenset[str] = frozenset(
    {
        "controls",
        "delegates",
        "handles",
        "manages",
        "owner",
        "owns",
        "responsible",
    }
)

#: Pre-compiled regex used by ``_tokenize`` to extract lowercase
#: alphanumeric runs as tokens.
_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Tokenization, cleaning, and canonical-responsibility extraction
# ---------------------------------------------------------------------------


def _normalize(summary: str) -> str:
    """Return ``summary`` lowercased with whitespace runs collapsed to single spaces."""

    return " ".join(summary.lower().split())


def _strip_marker_phrases(normalized: str) -> str:
    """Remove ownership-marker phrases from an already-normalized summary.

    Multi-word phrases are removed first (longest-first, see
    ``_OWNERSHIP_MARKER_PHRASES``); single-word marker tokens are filtered
    later inside ``_tokenize_clean``. The result is intended only for
    canonical-key extraction, never for stance detection.
    """

    cleaned = normalized
    for phrase in _OWNERSHIP_MARKER_PHRASES:
        if " " in phrase:
            cleaned = cleaned.replace(phrase, " ")
    return cleaned


def _tokenize(summary: str) -> list[str]:
    """Tokenize ``summary`` into a deterministic ordered list of content tokens.

    Tokens are lowercase alphanumeric runs from ``summary``, with stopwords
    removed. Tokens of length at least ``_SINGULARIZE_MIN_LEN`` that end in
    ``"s"`` are simple-singularized by stripping the trailing ``s`` so that
    ``"users"`` and ``"user"`` collapse to the same key. Source order is
    preserved.
    """

    tokens: list[str] = []
    for match in _TOKEN_RE.findall(summary.lower()):
        if match in _STOPWORDS:
            continue
        token = match[:-1] if len(match) >= _SINGULARIZE_MIN_LEN and match.endswith("s") else match
        tokens.append(token)
    return tokens


def _tokenize_for_key(summary: str) -> list[str]:
    """Tokenize ``summary`` for canonical-key and Jaccard purposes.

    Identical to ``_tokenize`` except that ownership-marker phrases (e.g.
    ``"is responsible for"``) and ownership-marker tokens (e.g. ``"owns"``)
    are removed first so the canonical responsibility phrase is drawn from
    the *subject* of the responsibility, not from the verb that asserts it.
    Stance detection works against the original summary and is unaffected.
    """

    cleaned = _strip_marker_phrases(_normalize(summary))
    return [t for t in _tokenize(cleaned) if t not in _OWNERSHIP_MARKER_TOKENS]


def _canonical_responsibility_key(tokens: list[str]) -> tuple[str, ...]:
    """Return the canonical responsibility key for a list of cleaned tokens.

    The key is the first non-stopword bigram (or the only unigram, or the
    empty tuple when no content tokens remain). It gates both the
    substantial-overlap branch and the contradictory-ownership branch of
    the conflict heuristic.
    """

    if not tokens:
        return ()
    if len(tokens) == 1:
        return (tokens[0],)
    return (tokens[0], tokens[1])


def _jaccard(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity of two token sets in ``[0.0, 1.0]``.

    Two empty sets are defined to have similarity ``0.0`` so that empty or
    pure-stopword summaries cannot trigger the substantial-overlap branch.
    """

    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _ownership_stance(summary: str) -> str | None:
    """Classify the ownership stance of a non-unknown summary.

    Returns ``"owner"`` if any positive-ownership phrase is present in the
    normalized summary, ``"delegator"`` if any negative-ownership phrase is
    present (negative phrases dominate when both kinds appear so that
    "delegates X but is responsible for Y" is treated as a delegation), or
    ``None`` when the summary makes no ownership claim either way.
    """

    normalized = _normalize(summary)
    if any(phrase in normalized for phrase in _NEGATIVE_OWNERSHIP_PHRASES):
        return "delegator"
    if any(phrase in normalized for phrase in _POSITIVE_OWNERSHIP_PHRASES):
        return "owner"
    return None


# ---------------------------------------------------------------------------
# Justification helpers
# ---------------------------------------------------------------------------


def _excerpt(summary: str) -> str:
    """Return a bounded single-line excerpt of ``summary`` for justifications.

    The excerpt is the leading ``JUSTIFICATION_EXCERPT_MAX_LEN`` characters
    of the whitespace-collapsed summary, suffixed with ``"…"`` when the
    summary was truncated. The excerpt is safe to embed in MCP tool result
    messages and HTML bodies.
    """

    flat = " ".join(summary.split())
    if len(flat) <= JUSTIFICATION_EXCERPT_MAX_LEN:
        return flat
    return flat[: JUSTIFICATION_EXCERPT_MAX_LEN - 1].rstrip() + "…"


def _format_responsibility_phrase(key: tuple[str, ...]) -> str:
    """Render a canonical responsibility key as a human-readable phrase."""

    if not key:
        return "(no content tokens)"
    return " ".join(key)


def _indeterminate_justification(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
    *,
    a_unknown: bool,
    b_unknown: bool,
) -> str:
    """Build the ``INDETERMINATE`` justification naming the unknown project(s).

    Requirement 9.4 mandates that the justification state that the purpose
    summary is unknown for the named project(s). Each project is named by
    its ``full_path`` (the human-readable GitLab path); both names appear
    when both summaries are unknown.
    """

    if a_unknown and b_unknown:
        return (
            f"purpose summary is unknown for projects "
            f"'{profile_a.full_path}' and '{profile_b.full_path}'"
        )
    unknown_profile = profile_a if a_unknown else profile_b
    return f"purpose summary is unknown for project '{unknown_profile.full_path}'"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify_pair(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
) -> ConflictResult:
    """Classify a pair of Project_Profiles for ``Purpose_Conflict``.

    Implements Requirements 9.1, 9.3, and 9.4. The classification is
    deterministic and depends only on the ``purpose_summary`` fields of the
    two profiles (and, for the indeterminate justification, each profile's
    ``full_path``). Identical inputs always yield identical results.

    Args:
        profile_a: The first project's profile.
        profile_b: The second project's profile.

    Returns:
        A ``ConflictResult`` whose ``kind`` is one of ``CONFLICT``,
        ``NO_CONFLICT``, or ``INDETERMINATE`` and whose ``justification``
        is a non-empty string referencing the purpose summaries that drove
        the classification (Requirement 9.1).
    """

    summary_a = profile_a.purpose_summary
    summary_b = profile_b.purpose_summary
    a_unknown = summary_a == UNKNOWN_PURPOSE_SUMMARY
    b_unknown = summary_b == UNKNOWN_PURPOSE_SUMMARY

    # Requirement 9.4: any "unknown" summary forces the indeterminate result.
    if a_unknown or b_unknown:
        return ConflictResult(
            kind=ConflictKind.INDETERMINATE,
            justification=_indeterminate_justification(
                profile_a,
                profile_b,
                a_unknown=a_unknown,
                b_unknown=b_unknown,
            ),
        )

    tokens_a = _tokenize_for_key(summary_a)
    tokens_b = _tokenize_for_key(summary_b)
    key_a = _canonical_responsibility_key(tokens_a)
    key_b = _canonical_responsibility_key(tokens_b)
    keys_match = key_a == key_b and key_a != ()

    # Requirement 9.3 (a): substantial overlap of primary responsibility.
    similarity = _jaccard(set(tokens_a), set(tokens_b))
    if keys_match and similarity >= SUBSTANTIAL_OVERLAP_THRESHOLD:
        return ConflictResult(
            kind=ConflictKind.CONFLICT,
            justification=(
                f"both purpose summaries describe substantially the same "
                f"primary responsibility '{_format_responsibility_phrase(key_a)}': "
                f'"{_excerpt(summary_a)}" vs "{_excerpt(summary_b)}"'
            ),
        )

    # Requirement 9.3 (b): contradictory ownership of the same responsibility.
    if keys_match:
        stance_a = _ownership_stance(summary_a)
        stance_b = _ownership_stance(summary_b)
        if {stance_a, stance_b} == {"owner", "delegator"}:
            return ConflictResult(
                kind=ConflictKind.CONFLICT,
                justification=(
                    f"purpose summaries assert contradictory ownership of "
                    f"'{_format_responsibility_phrase(key_a)}': "
                    f'"{_excerpt(summary_a)}" vs "{_excerpt(summary_b)}"'
                ),
            )

    # Requirement 9.3: no other basis may produce a CONFLICT result.
    return ConflictResult(
        kind=ConflictKind.NO_CONFLICT,
        justification=(
            f"purpose summaries describe different primary responsibilities "
            f"('{_format_responsibility_phrase(key_a)}' vs "
            f"'{_format_responsibility_phrase(key_b)}'): "
            f'"{_excerpt(summary_a)}" vs "{_excerpt(summary_b)}"'
        ),
    )


__all__ = [
    "JUSTIFICATION_EXCERPT_MAX_LEN",
    "SUBSTANTIAL_OVERLAP_THRESHOLD",
    "ConflictPair",
    "classify_pair",
    "find_all_conflicts",
]


# ---------------------------------------------------------------------------
# Symmetric-closure entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConflictPair:
    """One Purpose_Conflict edge produced by ``find_all_conflicts``.

    ``project_id_a`` is always strictly less than ``project_id_b`` so that
    an unordered pair has a single canonical representation in the result
    list. The two ids are the GitLab project ids drawn from the
    corresponding ``ProjectProfile.gitlab_project_id`` fields.
    ``justification`` is the same string that ``classify_pair`` produced
    for the pair (with the lower-id profile passed as ``profile_a``),
    preserving Requirement 9.1's guarantee that every reported conflict
    carries a non-empty justification referencing the two purpose
    summaries.
    """

    project_id_a: int
    project_id_b: int
    justification: str


def find_all_conflicts(
    profiles: Sequence[ProjectProfile],
) -> list[ConflictPair]:
    """Return every unordered pair ``{a, b}`` (``a != b``) classified as a conflict.

    Implements Requirement 9.2 (Property 14): the result is exactly the set
    of unordered pairs ``{a, b}`` such that
    ``classify_pair(a, b).kind == ConflictKind.CONFLICT``, with each
    unordered pair represented at most once.

    The output is deterministic. Within each ``ConflictPair`` the two IDs
    are placed in ascending order (``project_id_a < project_id_b``); the
    list itself is sorted by ``(project_id_a, project_id_b)``.
    ``classify_pair`` is always invoked with the lower-id profile as
    ``profile_a`` so the embedded justification ordering is also stable.

    If ``profiles`` happens to contain two entries with the same
    ``gitlab_project_id`` (a malformed input — the catalog guarantees one
    profile per project per snapshot), those degenerate pairings are
    skipped: a project cannot conflict with itself, and we never invoke
    ``classify_pair`` on two distinct profiles that share an ID. Pairs are
    de-duplicated by the unordered ID pair, so a repeated profile cannot
    produce duplicate edges.

    Args:
        profiles: The set of ``ProjectProfile`` records to compare. Order
            of ``profiles`` does not affect the returned set; it only
            affects whether the same unordered pair is offered to
            ``classify_pair`` more than once (in which case we short-circuit
            on the first occurrence).

    Returns:
        A list of ``ConflictPair`` records, one per unordered pair of
        distinct project IDs whose profiles classify as a conflict, sorted
        by ``(project_id_a, project_id_b)``.
    """

    pairs_by_key: dict[tuple[int, int], ConflictPair] = {}
    n = len(profiles)
    for i in range(n):
        profile_i = profiles[i]
        for j in range(i + 1, n):
            profile_j = profiles[j]
            id_i = profile_i.gitlab_project_id
            id_j = profile_j.gitlab_project_id

            # A profile cannot conflict with itself; same-id duplicates in
            # the input are treated as the same project and skipped.
            if id_i == id_j:
                continue

            # Canonical ordering by gitlab_project_id makes the unordered
            # pair representation unique and stabilizes which profile is
            # passed as profile_a to classify_pair.
            if id_i < id_j:
                first, second = profile_i, profile_j
            else:
                first, second = profile_j, profile_i

            key = (first.gitlab_project_id, second.gitlab_project_id)
            if key in pairs_by_key:
                # Already classified this unordered pair from an earlier
                # iteration (possible only when profiles contains repeated
                # IDs); do not re-invoke classify_pair.
                continue

            result = classify_pair(first, second)
            if result.kind is ConflictKind.CONFLICT:
                pairs_by_key[key] = ConflictPair(
                    project_id_a=key[0],
                    project_id_b=key[1],
                    justification=result.justification,
                )

    return [pairs_by_key[key] for key in sorted(pairs_by_key)]

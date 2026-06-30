"""Integration golden test: ``analyze()`` reproduces the curated Project_Profile.

For each of the four sample Go repositories captured under
``tests/integration/golden/go/<repo-name>/``, this test:

1. Loads the committed ``repository_contents.json`` snapshot.
2. Runs :func:`project_knowledge_mcp.project_analyzer.analyze` with the
   canonical placeholder arguments documented in
   :mod:`tests.integration.golden.go._curate`.
3. Normalizes the ``produced_at`` timestamp to the fixed value the
   curate script bakes into every committed golden.
4. Asserts the resulting JSON document is deep-equal to the curated
   ``expected_profile.json`` for that repository.

The curate script (``_curate.py``) is the sole producer of the golden
files; this test is the sole consumer. Together they form a closed loop:
running ``python -m tests.integration.golden.go._curate`` regenerates
the goldens, and running this test verifies that ``analyze()`` produces
those same goldens on every subsequent invocation. A diff between the
analyzer's current output and a committed golden is the review surface
for any deliberate analyzer behavior change.

Implements task 12.4 of the Go analyzer support spec; exercises
Requirements 1.1, 8.2, 8.3, 8.5, 9.2, 9.3, 9.4, 9.5, and 9.6 against
real-world inputs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer import analyze

pytestmark = pytest.mark.integration


#: Fixed ``produced_at`` value baked into every curated golden. Must match
#: :data:`tests.integration.golden.go._curate.FIXED_PRODUCED_AT` exactly;
#: the curate script stamps this value, this test stamps it again so the
#: two sides agree byte-for-byte.
FIXED_PRODUCED_AT: datetime = datetime(2024, 1, 1, tzinfo=UTC)

#: The four sample repositories whose goldens this test compares against.
#: Matches the directory layout under ``tests/integration/golden/go/`` and
#: the tuple in :data:`tests.integration.golden.go._curate.SAMPLE_REPOS`.
SAMPLE_REPOS: tuple[str, ...] = (
    "fec_pool_service",
    "repayment_service",
    "cat-service",
    "aps_los_vtiger",
)


def _golden_dir() -> Path:
    """Return the directory holding the per-repo golden subdirectories."""
    return Path(__file__).resolve().parent / "golden" / "go"


def _load_snapshot(repo_name: str) -> RepositoryContents:
    """Load the ``repository_contents.json`` snapshot for ``repo_name``."""
    payload_path = _golden_dir() / repo_name / "repository_contents.json"
    with payload_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return RepositoryContents.model_validate(payload)


def _load_expected_profile(repo_name: str) -> dict[str, object]:
    """Load the curated ``expected_profile.json`` for ``repo_name`` as a dict.

    We compare against the parsed JSON dict (rather than re-validating
    into a :class:`ProjectProfile`) so the assertion catches any drift
    in the on-disk JSON shape — field ordering aside — including spurious
    or missing keys at any depth.
    """
    payload_path = _golden_dir() / repo_name / "expected_profile.json"
    with payload_path.open("r", encoding="utf-8") as fh:
        payload: dict[str, object] = json.load(fh)
    return payload


@pytest.mark.parametrize("repo_name", SAMPLE_REPOS)
def test_analyze_matches_curated_golden(repo_name: str) -> None:
    """``analyze()`` reproduces the curated Project_Profile for each sample repo.

    Mirrors the placeholder argument convention from
    :func:`tests.integration.golden.go._curate.curate_profile`:

    * ``project_id``       = ``1``
    * ``full_path``        = ``f"samples/{repo_name}"``
    * ``analysis_branch``  = ``"main"``
    * ``commit_sha``       = the snapshot's recorded ``commit_sha``
    * ``repo_description`` = ``None``
    * ``produced_at``      = forced to :data:`FIXED_PRODUCED_AT` via
      :meth:`ProjectProfile.model_copy` after the analyzer returns.

    The two sides are compared as parsed JSON documents (``dict``s of
    primitives) so the assertion is a structural deep-equal rather than
    a string compare on serialized JSON, which keeps the diff readable
    on failure without being sensitive to incidental formatting choices
    in either producer.
    """
    repo_contents = _load_snapshot(repo_name)

    profile = analyze(
        project_id=1,
        full_path=f"samples/{repo_name}",
        analysis_branch="main",
        commit_sha=repo_contents.commit_sha,
        repo_description=None,
        repository_contents=repo_contents,
    )
    # The aggregator stamps ``produced_at = datetime.now(UTC)`` on every
    # profile. Swap in the fixed timestamp the curate script uses so the
    # two JSON documents are directly comparable. ``model_copy`` re-runs
    # validation because ``ProjectProfile`` has ``validate_assignment=True``.
    normalized = profile.model_copy(update={"produced_at": FIXED_PRODUCED_AT})

    # Serialize via Pydantic's JSON mode and round-trip through ``json``
    # so the resulting dict mirrors exactly what Pydantic wrote to the
    # golden file on the producer side. This keeps datetime / enum /
    # nested-model serialization consistent between the two sides.
    produced = json.loads(normalized.model_dump_json())
    expected = _load_expected_profile(repo_name)

    assert produced == expected, (
        f"analyze() output for {repo_name!r} does not match the curated golden. "
        f"Regenerate with `python -m tests.integration.golden.go._curate` "
        f"if the change is intentional, then review the diff."
    )

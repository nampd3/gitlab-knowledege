"""Regenerate the curated ``Project_Profile`` golden JSONs for the four sample repos.

This helper loads each ``repository_contents.json`` snapshot produced by
:mod:`tests.integration.golden.go._snapshot`, runs :func:`analyze` against
it with the canonical stable placeholder arguments documented in spec
task 12.3, normalizes ``produced_at`` to a fixed timestamp, and writes
the result to ``tests/integration/golden/go/<repo-name>/expected_profile.json``.

The script is the only producer of the ``expected_profile.json`` goldens;
the integration test in task 12.4 is the consumer. Re-running this script
is the supported way to refresh the goldens after a deliberate analyzer
behavior change — the diff against the previously committed file is the
review surface.

The placeholder argument convention (so the integration test mirrors it):

* ``project_id``       = ``1``
* ``full_path``        = ``f"samples/{repo_name}"``
* ``analysis_branch``  = ``"main"``
* ``commit_sha``       = the snapshot's recorded ``commit_sha`` (the
  same fixed SHA the snapshot was captured against; reproducing the
  snapshot reproduces this value)
* ``repo_description`` = ``None``
* ``produced_at``      = forced to ``2024-01-01T00:00:00Z`` after the
  analyzer returns, replacing the ``datetime.now(UTC)`` value the
  aggregator stamps on the profile

Invoke as a module from the repository root::

    python -m tests.integration.golden.go._curate

The four sample repository names are derived from the directory layout
under ``tests/integration/golden/go/``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from project_knowledge_mcp.models import ProjectProfile, RepositoryContents
from project_knowledge_mcp.project_analyzer import analyze

#: The fixed ``produced_at`` timestamp recorded on every curated golden.
FIXED_PRODUCED_AT: datetime = datetime(2024, 1, 1, tzinfo=UTC)

#: The four sample repositories whose goldens this script regenerates.
#: The names match the directory layout under
#: ``tests/integration/golden/go/`` (which is also the directory layout
#: produced by :mod:`_snapshot` when invoked with the default ``--name``).
SAMPLE_REPOS: tuple[str, ...] = (
    "fec_pool_service",
    "repayment_service",
    "cat-service",
    "aps_los_vtiger",
)


def _golden_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_snapshot(repo_name: str) -> RepositoryContents:
    """Load the ``repository_contents.json`` snapshot for ``repo_name``."""
    payload_path = _golden_dir() / repo_name / "repository_contents.json"
    with payload_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return RepositoryContents.model_validate(payload)


def curate_profile(repo_name: str) -> ProjectProfile:
    """Run :func:`analyze` against the snapshot for ``repo_name`` with the
    canonical placeholder arguments and force ``produced_at`` to the fixed
    timestamp. Returns the resulting :class:`ProjectProfile`.

    The aggregator stamps ``produced_at = datetime.now(UTC)`` on every
    profile (see ``project_analyzer.__init__._build_profile``), so we
    take the analyzer output, immutably swap in :data:`FIXED_PRODUCED_AT`
    via :meth:`ProjectProfile.model_copy`, and return the normalized
    profile. Pydantic re-validates on copy because the model has
    ``validate_assignment=True``, so a bad fixed timestamp would fail
    fast here rather than in the consumer.
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
    return profile.model_copy(update={"produced_at": FIXED_PRODUCED_AT})


def write_golden(repo_name: str, profile: ProjectProfile) -> Path:
    """Write the curated golden ``expected_profile.json`` for ``repo_name``.

    Pretty-printed with 2-space indentation for diff-ability. A trailing
    newline keeps the file POSIX-friendly and matches the convention
    :mod:`_snapshot` uses for its sibling ``repository_contents.json``.
    """
    target = _golden_dir() / repo_name / "expected_profile.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = profile.model_dump_json(indent=2)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    repos = list(argv) if argv else list(SAMPLE_REPOS)
    for repo in repos:
        profile = curate_profile(repo)
        target = write_golden(repo, profile)
        # Summary diagnostics: counts make it easy to eyeball changes
        # between regenerations without diffing the full JSON payload.
        print(
            f"wrote {target.relative_to(_golden_dir().parent.parent.parent)} "
            f"(inputs={len(profile.abstract_inputs)}, "
            f"outputs={len(profile.abstract_outputs)}, "
            f"external_services={len(profile.external_service_dependencies)}, "
            f"database_tables={len(profile.database_table_dependencies)}, "
            f"degraded={len(profile.degraded_sections)})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))

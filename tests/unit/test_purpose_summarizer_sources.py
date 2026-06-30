"""Unit tests for purpose-summary derivation per source kind.

Each test isolates one of the four sources permitted by Requirement 3.2
and asserts that the source alone is sufficient to derive a non-default
summary, and that the derived summary actually carries identifying
content from the chosen source.

Implements Requirement 3.2.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import (
    UNKNOWN_PURPOSE_SUMMARY,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.purpose import summarize_purpose

pytestmark = pytest.mark.unit


def _repo(files: dict[str, str]) -> RepositoryContents:
    """Build a minimal :class:`RepositoryContents` for the given file map."""

    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )


# ---------------------------------------------------------------------------
# Source 1: README only.
# ---------------------------------------------------------------------------


def test_readme_only_yields_summary() -> None:
    """A root-level README alone is sufficient to derive a summary."""
    marker = "uniqueReadmeMarkerXYZ"
    readme = (
        "# My Project\n"
        "\n"
        f"This service handles {marker} ingestion for the demo portfolio.\n"
    )
    repo = _repo({"README.md": readme})

    summary, reason = summarize_purpose(repo, repository_description=None)

    assert summary != UNKNOWN_PURPOSE_SUMMARY
    assert reason is None
    assert marker in summary


# ---------------------------------------------------------------------------
# Source 2: GitLab repository description only.
# ---------------------------------------------------------------------------


def test_gitlab_description_only_yields_summary() -> None:
    """A non-empty GitLab description alone is sufficient to derive a summary."""
    marker = "uniqueGitlabMarkerABC"
    description = f"Repo description: {marker} pipeline orchestrator."
    repo = _repo({})

    summary, reason = summarize_purpose(repo, repository_description=description)

    assert summary != UNKNOWN_PURPOSE_SUMMARY
    assert reason is None
    assert marker in summary


# ---------------------------------------------------------------------------
# Source 3: Manifest description only (pyproject.toml).
# ---------------------------------------------------------------------------


def test_pyproject_only_yields_summary() -> None:
    """A ``pyproject.toml`` description alone is sufficient to derive a summary."""
    marker = "uniquePyprojectMarkerDEF"
    pyproject = (
        '[project]\n'
        'name = "synthetic"\n'
        'version = "0.1.0"\n'
        f'description = "A {marker} library for demo purposes."\n'
    )
    repo = _repo({"pyproject.toml": pyproject})

    summary, reason = summarize_purpose(repo, repository_description=None)

    assert summary != UNKNOWN_PURPOSE_SUMMARY
    assert reason is None
    assert marker in summary


# ---------------------------------------------------------------------------
# Source 4: Top-level module docstring only.
# ---------------------------------------------------------------------------


def test_module_docstring_only_yields_summary() -> None:
    """A root-level Python module docstring alone is sufficient to derive a summary."""
    marker = "uniqueDocstringMarkerGHI"
    source = (
        f'"""Top-level module that performs {marker} computations."""\n'
        "\n"
        "def main() -> None:\n"
        "    pass\n"
    )
    repo = _repo({"app.py": source})

    summary, reason = summarize_purpose(repo, repository_description=None)

    assert summary != UNKNOWN_PURPOSE_SUMMARY
    assert reason is None
    assert marker in summary

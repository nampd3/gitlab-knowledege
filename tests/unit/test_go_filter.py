"""Unit tests for ``project_analyzer.go.go_filter``.

These tests pin down the two pure predicates that gate every Go-aware
code path in the analyzer:

* :func:`is_go_source_file` -- a per-path filter that rejects vendored
  copies of third-party libraries and accepts only files with a
  case-sensitive ``.go`` suffix (Requirement 1.3).
* :func:`has_go_artefacts` -- the repo-level guard the aggregator
  consults exactly once before doing any Go work; when it returns
  ``False`` the rest of the Go pipeline is a no-op so the produced
  Project_Profile remains byte-identical to the pre-feature output
  (Requirements 1.4, 11.5).

Implements Requirements 1.3, 1.4, 11.5.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer.go.go_filter import (
    has_go_artefacts,
    is_go_source_file,
)

pytestmark = pytest.mark.unit


def _repo(files: dict[str, str]) -> RepositoryContents:
    """Build a minimal :class:`RepositoryContents` for the given file map."""

    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )


# ---------------------------------------------------------------------------
# is_go_source_file -- vendor-segment rejection (Requirement 1.3)
# ---------------------------------------------------------------------------


def test_is_go_source_file_rejects_top_level_vendor_segment() -> None:
    assert is_go_source_file("vendor/foo.go") is False


def test_is_go_source_file_rejects_nested_vendor_segment() -> None:
    assert is_go_source_file("src/vendor/foo.go") is False


def test_is_go_source_file_rejects_deeply_nested_vendor_segment() -> None:
    assert is_go_source_file("vendor/github.com/pkg/errors/errors.go") is False


def test_is_go_source_file_rejects_vendor_segment_in_middle_of_path() -> None:
    assert is_go_source_file("internal/vendor/foo/bar.go") is False


# ---------------------------------------------------------------------------
# is_go_source_file -- the literal filename ``vendor.go`` is allowed at any
# depth: only directory segments named ``vendor`` cause exclusion.
# ---------------------------------------------------------------------------


def test_is_go_source_file_accepts_filename_vendor_go_at_root() -> None:
    assert is_go_source_file("vendor.go") is True


def test_is_go_source_file_accepts_filename_vendor_go_when_nested() -> None:
    assert is_go_source_file("pkg/vendor.go") is True


def test_is_go_source_file_accepts_filename_vendor_go_at_deeper_path() -> None:
    assert is_go_source_file("src/cmd/vendor.go") is True


# ---------------------------------------------------------------------------
# is_go_source_file -- case sensitivity of the ``.go`` suffix
# (Requirement 1.3 carve-out: only lowercase ``.go`` is a Go source file)
# ---------------------------------------------------------------------------


def test_is_go_source_file_rejects_uppercase_suffix() -> None:
    assert is_go_source_file("main.GO") is False


def test_is_go_source_file_rejects_titlecase_suffix() -> None:
    assert is_go_source_file("main.Go") is False


def test_is_go_source_file_rejects_mixed_case_suffix() -> None:
    assert is_go_source_file("pkg/foo.gO") is False


def test_is_go_source_file_accepts_lowercase_suffix() -> None:
    assert is_go_source_file("main.go") is True


# ---------------------------------------------------------------------------
# is_go_source_file -- suffix requirement: non-Go suffixes are rejected
# even when no vendor segment is present.
# ---------------------------------------------------------------------------


def test_is_go_source_file_rejects_non_go_suffix() -> None:
    assert is_go_source_file("README.txt") is False


def test_is_go_source_file_rejects_no_suffix() -> None:
    assert is_go_source_file("Makefile") is False


def test_is_go_source_file_rejects_go_mod_manifest() -> None:
    # ``go.mod`` is the module manifest, not a Go source file: it must not
    # be processed by the per-file Go scanners.
    assert is_go_source_file("go.mod") is False


def test_is_go_source_file_rejects_go_sum_manifest() -> None:
    assert is_go_source_file("go.sum") is False


def test_is_go_source_file_rejects_path_containing_go_substring() -> None:
    assert is_go_source_file("foo.golang") is False


# ---------------------------------------------------------------------------
# is_go_source_file -- Windows path separators are normalized to ``/``
# before vendor-segment inspection (Requirement 1.3).
# ---------------------------------------------------------------------------


def test_is_go_source_file_rejects_vendor_segment_with_backslash_separators() -> None:
    assert is_go_source_file("src\\vendor\\foo.go") is False


def test_is_go_source_file_rejects_vendor_with_mixed_separators() -> None:
    assert is_go_source_file("src\\vendor/foo.go") is False


def test_is_go_source_file_accepts_backslash_path_without_vendor_segment() -> None:
    assert is_go_source_file("src\\pkg\\foo.go") is True


# ---------------------------------------------------------------------------
# is_go_source_file -- segment names that merely *contain* "vendor" do
# not trigger exclusion: only segments equal to ``vendor`` (case-sensitive)
# are vendored.
# ---------------------------------------------------------------------------


def test_is_go_source_file_accepts_segment_starting_with_vendor() -> None:
    assert is_go_source_file("myvendor/foo.go") is True


def test_is_go_source_file_accepts_segment_ending_with_vendor() -> None:
    assert is_go_source_file("thevendor/foo.go") is True


def test_is_go_source_file_accepts_plural_vendors_segment() -> None:
    assert is_go_source_file("vendors/foo.go") is True


def test_is_go_source_file_accepts_uppercase_vendor_segment() -> None:
    # The vendor segment match is case-sensitive; ``Vendor`` is a perfectly
    # ordinary directory name in user code.
    assert is_go_source_file("Vendor/foo.go") is True


def test_is_go_source_file_accepts_underscore_vendor_segment() -> None:
    assert is_go_source_file("_vendor/foo.go") is True


# ---------------------------------------------------------------------------
# is_go_source_file -- ordinary repository-relative paths
# ---------------------------------------------------------------------------


def test_is_go_source_file_accepts_root_main_go() -> None:
    assert is_go_source_file("main.go") is True


def test_is_go_source_file_accepts_nested_package_file() -> None:
    assert is_go_source_file("internal/server/router.go") is True


def test_is_go_source_file_accepts_test_file() -> None:
    # ``foo_test.go`` is a Go source file by language convention; the
    # filter does not single it out.
    assert is_go_source_file("internal/server/router_test.go") is True


# ---------------------------------------------------------------------------
# has_go_artefacts -- root go.mod presence (Requirement 1.4)
# ---------------------------------------------------------------------------


def test_has_go_artefacts_true_when_go_mod_at_root() -> None:
    repo = _repo({"go.mod": "module example.com/foo\n"})
    assert has_go_artefacts(repo) is True


def test_has_go_artefacts_true_when_go_mod_at_root_with_other_files() -> None:
    repo = _repo(
        {
            "go.mod": "module example.com/foo\n",
            "README.md": "# example\n",
            "Makefile": "all:\n",
        }
    )
    assert has_go_artefacts(repo) is True


def test_has_go_artefacts_false_when_go_mod_only_in_subdirectory() -> None:
    # The guard requires ``go.mod`` at exactly the repo root. A nested
    # ``go.mod`` (e.g. inside a sub-module sample directory) without any
    # actual Go source is not enough.
    repo = _repo(
        {
            "samples/nested/go.mod": "module example.com/sample\n",
            "README.md": "# example\n",
        }
    )
    assert has_go_artefacts(repo) is False


# ---------------------------------------------------------------------------
# has_go_artefacts -- presence of at least one Go source file
# ---------------------------------------------------------------------------


def test_has_go_artefacts_true_when_single_go_source_file_present() -> None:
    repo = _repo({"main.go": "package main\n"})
    assert has_go_artefacts(repo) is True


def test_has_go_artefacts_true_when_nested_go_source_file_present() -> None:
    repo = _repo({"internal/server/router.go": "package server\n"})
    assert has_go_artefacts(repo) is True


def test_has_go_artefacts_true_when_both_go_mod_and_source_present() -> None:
    repo = _repo(
        {
            "go.mod": "module example.com/foo\n",
            "cmd/app/main.go": "package main\n",
        }
    )
    assert has_go_artefacts(repo) is True


# ---------------------------------------------------------------------------
# has_go_artefacts -- no-Go repositories return False (Requirements 1.4, 11.5)
# ---------------------------------------------------------------------------


def test_has_go_artefacts_false_for_empty_repository() -> None:
    repo = _repo({})
    assert has_go_artefacts(repo) is False


def test_has_go_artefacts_false_for_python_repository() -> None:
    repo = _repo(
        {
            "pyproject.toml": "[project]\nname = 'svc'\n",
            "src/svc/__init__.py": "",
            "src/svc/main.py": "def main(): ...\n",
            "README.md": "# svc\n",
        }
    )
    assert has_go_artefacts(repo) is False


def test_has_go_artefacts_false_for_javascript_repository() -> None:
    repo = _repo(
        {
            "package.json": '{"name": "svc"}\n',
            "src/index.js": "module.exports = {}\n",
        }
    )
    assert has_go_artefacts(repo) is False


def test_has_go_artefacts_false_for_files_named_like_go_but_wrong_suffix() -> None:
    # Files named with mixed-case ``.GO`` are rejected by ``is_go_source_file``
    # so they must not flip the guard either.
    repo = _repo({"main.GO": "package main\n"})
    assert has_go_artefacts(repo) is False


# ---------------------------------------------------------------------------
# has_go_artefacts -- vendored-only repositories return False
#
# A repo whose only ``.go`` files live under a ``vendor/`` directory has
# no first-party Go code; ``is_go_source_file`` rejects every path so
# the guard returns False (assuming no root ``go.mod``).
# ---------------------------------------------------------------------------


def test_has_go_artefacts_false_when_only_vendored_go_files_exist() -> None:
    repo = _repo(
        {
            "vendor/github.com/pkg/errors/errors.go": "package errors\n",
            "vendor/github.com/pkg/errors/stack.go": "package errors\n",
            "README.md": "# svc\n",
        }
    )
    assert has_go_artefacts(repo) is False


def test_has_go_artefacts_false_when_only_nested_vendored_go_files_exist() -> None:
    repo = _repo(
        {
            "internal/vendor/foo/bar.go": "package foo\n",
            "internal/vendor/foo/baz.go": "package foo\n",
        }
    )
    assert has_go_artefacts(repo) is False


def test_has_go_artefacts_true_when_vendored_files_coexist_with_first_party_source() -> None:
    repo = _repo(
        {
            "vendor/github.com/pkg/errors/errors.go": "package errors\n",
            "internal/server/router.go": "package server\n",
        }
    )
    assert has_go_artefacts(repo) is True


def test_has_go_artefacts_true_for_vendored_only_repo_with_root_go_mod() -> None:
    # Even when every ``.go`` file is vendored, a root ``go.mod`` still
    # marks the repository as Go.
    repo = _repo(
        {
            "go.mod": "module example.com/foo\n",
            "vendor/github.com/pkg/errors/errors.go": "package errors\n",
        }
    )
    assert has_go_artefacts(repo) is True

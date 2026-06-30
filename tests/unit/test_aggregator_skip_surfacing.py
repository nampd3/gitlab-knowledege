# ruff: noqa: E501
"""Unit tests for ``_file_level_skip_entries`` and the four ``_safe_*``
helpers' surfacing of per-file Go-parser skip reasons in
``Project_Profile.degraded_sections`` (task 11.6).

The aggregator records every :class:`SkipFileEvent` produced by
``parse_repo`` as a structured ``degraded_sections`` entry of the form::

    "<section>: skipped <path> (<reason>)"

One entry per skipped file per affected section. The four sections are
``"purpose"``, ``"abstract_io"``, ``"external_services"``, and
``"database_tables"``.

* For ``"abstract_io"``, ``"external_services"``, and
  ``"database_tables"`` every Go file is in scope, so every
  :class:`SkipFileEvent` in ``events_by_file`` contributes one entry to
  that section.
* For ``"purpose"`` only files at the repository root, at
  ``cmd/main.go``, or at ``cmd/<name>/main.go`` are in scope (those are
  the only paths the purpose summarizer inspects for package doc
  comments).

The tests below exercise the path-sorted ordering, the canonical reason
strings from Requirement 10.4 and Requirement 11.1, and the path filter
for the purpose section. They also confirm that the ``list[str]`` shape
of ``degraded_sections`` is preserved when the heterogeneous mix of
plain section names (recorded on sub-analyzer failure) and structured
strings (recorded per skipped file) appear in the same list.

Validates Requirements 10.4, 11.1, 11.2.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer import (
    ABSTRACT_IO_SECTION,
    DATABASE_TABLES_SECTION,
    EXTERNAL_SERVICES_SECTION,
    PURPOSE_SECTION,
    analyze,
)
from project_knowledge_mcp.project_analyzer import (
    _file_level_skip_entries,  # type: ignore[attr-defined]
)
from project_knowledge_mcp.project_analyzer.go._events import (
    ImportEvent,
    SkipFileEvent,
)
from project_knowledge_mcp.project_analyzer.go.go_purpose import (
    is_eligible_main_path,
)

pytestmark = pytest.mark.unit


_PROJECT_ID = 42
_FULL_PATH = "group/sub/project"
_BRANCH = "main"
_COMMIT = "deadbeef" * 5


# ---------------------------------------------------------------------------
# _file_level_skip_entries: shape, ordering, reason verbatim
# ---------------------------------------------------------------------------


def test_file_level_skip_entries_empty_when_no_skip_events() -> None:
    events = {
        "a.go": [ImportEvent(path="fmt", alias=None, file_path="a.go", line=1)],
    }
    assert _file_level_skip_entries(events, ABSTRACT_IO_SECTION) == []


def test_file_level_skip_entries_canonical_build_constraint_reason() -> None:
    events = {
        "constrained.go": [
            SkipFileEvent(
                reason="build constraint requires toolchain",
                file_path="constrained.go",
                line=1,
            ),
        ],
    }
    assert _file_level_skip_entries(events, ABSTRACT_IO_SECTION) == [
        "abstract_io: skipped constrained.go (build constraint requires toolchain)",
    ]


def test_file_level_skip_entries_canonical_cgo_reason() -> None:
    events = {
        "cgo_user.go": [
            SkipFileEvent(
                reason="cgo directive requires toolchain",
                file_path="cgo_user.go",
                line=3,
            ),
        ],
    }
    assert _file_level_skip_entries(events, EXTERNAL_SERVICES_SECTION) == [
        "external_services: skipped cgo_user.go (cgo directive requires toolchain)",
    ]


def test_file_level_skip_entries_tokenization_failure_reason_verbatim() -> None:
    events = {
        "bad.go": [
            SkipFileEvent(
                reason="tokenization failed: unterminated string literal",
                file_path="bad.go",
                line=4,
            ),
        ],
    }
    assert _file_level_skip_entries(events, DATABASE_TABLES_SECTION) == [
        "database_tables: skipped bad.go (tokenization failed: unterminated string literal)",
    ]


def test_file_level_skip_entries_sorted_by_path() -> None:
    events = {
        "zebra.go": [SkipFileEvent(reason="r1", file_path="zebra.go", line=1)],
        "alpha.go": [SkipFileEvent(reason="r2", file_path="alpha.go", line=1)],
        "mike.go": [SkipFileEvent(reason="r3", file_path="mike.go", line=1)],
    }
    assert _file_level_skip_entries(events, ABSTRACT_IO_SECTION) == [
        "abstract_io: skipped alpha.go (r2)",
        "abstract_io: skipped mike.go (r3)",
        "abstract_io: skipped zebra.go (r1)",
    ]


def test_file_level_skip_entries_path_filter_purpose_only_eligible_paths() -> None:
    # Files at the repo root and at cmd/main.go are eligible; a deeply
    # nested file is not. The purpose section's filter restricts the
    # structured entries to files the purpose summarizer would have
    # inspected (design Property 13).
    events = {
        "main.go": [SkipFileEvent(reason="build constraint requires toolchain", file_path="main.go", line=1)],
        "cmd/main.go": [SkipFileEvent(reason="cgo directive requires toolchain", file_path="cmd/main.go", line=1)],
        "cmd/server/main.go": [SkipFileEvent(reason="tokenization failed: x", file_path="cmd/server/main.go", line=1)],
        "internal/pkg/foo.go": [SkipFileEvent(reason="build constraint requires toolchain", file_path="internal/pkg/foo.go", line=1)],
    }
    assert _file_level_skip_entries(
        events, PURPOSE_SECTION, path_filter=is_eligible_main_path
    ) == [
        "purpose: skipped cmd/main.go (cgo directive requires toolchain)",
        "purpose: skipped cmd/server/main.go (tokenization failed: x)",
        "purpose: skipped main.go (build constraint requires toolchain)",
    ]


# ---------------------------------------------------------------------------
# End-to-end via analyze(): every affected section records one entry
# ---------------------------------------------------------------------------


def _rc(files: dict[str, str]) -> RepositoryContents:
    return RepositoryContents(
        gitlab_project_id=_PROJECT_ID,
        commit_sha=_COMMIT,
        files=files,
    )


def test_analyze_build_constrained_internal_file_surfaces_three_sections() -> None:
    """An internal build-constrained file appears in three sections (not purpose).

    A file at ``internal/pkg/foo.go`` is NOT eligible for purpose
    (only repo-root ``.go``, ``cmd/main.go``, and
    ``cmd/<name>/main.go`` are). It IS in scope for the other three
    sections (the I/O, external-service, and database-table detectors
    walk every file). The aggregator therefore records exactly three
    structured ``degraded_sections`` entries for the same file.
    """

    rc = _rc(
        {
            "go.mod": "module example.com/m\n",
            "main.go": "package main\nfunc main() {}\n",
            "internal/pkg/foo.go": "//go:build linux && amd64\n\npackage foo\n",
        }
    )

    profile = analyze(_PROJECT_ID, _FULL_PATH, _BRANCH, _COMMIT, None, rc)

    structured = [s for s in profile.degraded_sections if s.startswith(("abstract_io: ", "external_services: ", "database_tables: ", "purpose: "))]
    assert structured == [
        "abstract_io: skipped internal/pkg/foo.go (build constraint requires toolchain)",
        "external_services: skipped internal/pkg/foo.go (build constraint requires toolchain)",
        "database_tables: skipped internal/pkg/foo.go (build constraint requires toolchain)",
    ]


def test_analyze_cgo_at_cmd_main_surfaces_all_four_sections() -> None:
    """A cgo-bound ``cmd/main.go`` is in scope for all four sections.

    Purpose would have inspected it for a package doc comment; the
    other three would have walked its events. Each section records
    one structured entry with the canonical cgo reason from
    Requirement 10.4.
    """

    rc = _rc(
        {
            "go.mod": "module example.com/m\n",
            "cmd/main.go": 'package main\nimport "C"\nfunc main() {}\n',
        }
    )

    profile = analyze(_PROJECT_ID, _FULL_PATH, _BRANCH, _COMMIT, None, rc)

    structured = [s for s in profile.degraded_sections if ": skipped " in s]
    assert structured == [
        "purpose: skipped cmd/main.go (cgo directive requires toolchain)",
        "abstract_io: skipped cmd/main.go (cgo directive requires toolchain)",
        "external_services: skipped cmd/main.go (cgo directive requires toolchain)",
        "database_tables: skipped cmd/main.go (cgo directive requires toolchain)",
    ]


def test_analyze_tokenization_failure_at_cmd_named_main_surfaces_all_four() -> None:
    """A tokenization failure at ``cmd/<name>/main.go`` is in scope for all four sections."""

    rc = _rc(
        {
            "go.mod": "module example.com/m\n",
            "cmd/server/main.go": 'package main\nvar x = "never closed\n',
        }
    )

    profile = analyze(_PROJECT_ID, _FULL_PATH, _BRANCH, _COMMIT, None, rc)

    structured = [s for s in profile.degraded_sections if ": skipped " in s]
    # The tokenizer's detail string is captured verbatim; the test
    # asserts each section gets exactly one entry and the canonical
    # ``"tokenization failed: "`` prefix is preserved.
    assert len(structured) == 4
    sections = []
    for entry in structured:
        section, rest = entry.split(": ", 1)
        sections.append(section)
        assert rest.startswith("skipped cmd/server/main.go (tokenization failed: ")
        assert rest.endswith(")")
    assert sections == [
        "purpose",
        "abstract_io",
        "external_services",
        "database_tables",
    ]


def test_analyze_no_go_artefacts_records_no_structured_entries() -> None:
    """A no-Go repository never produces a structured skip entry (Requirement 11.5)."""

    rc = _rc({"README.md": "Hello\n", "src/main.py": 'print("hi")\n'})
    profile = analyze(_PROJECT_ID, _FULL_PATH, _BRANCH, _COMMIT, None, rc)
    assert profile.degraded_sections == []


def test_analyze_well_formed_go_repo_records_no_structured_entries() -> None:
    """A well-formed Go repository never produces structured skip entries."""

    rc = _rc(
        {
            "go.mod": "module example.com/m\n",
            "main.go": "package main\nfunc main() {}\n",
            "internal/util.go": 'package util\nimport "fmt"\n',
        }
    )
    profile = analyze(_PROJECT_ID, _FULL_PATH, _BRANCH, _COMMIT, None, rc)
    structured = [s for s in profile.degraded_sections if ": skipped " in s]
    assert structured == []


def test_analyze_skipped_file_does_not_destabilize_well_formed_neighbours() -> None:
    """Build-constrained file is contained; well-formed neighbours still analyze.

    The four list-shaped sections (``abstract_inputs``,
    ``abstract_outputs``, ``external_service_dependencies``,
    ``database_table_dependencies``) on the mixed repo are equal to
    those on a same-base repo without the bad file. The bad file's
    only effect is on ``degraded_sections`` (Requirement 11.1).
    """

    base_files = {
        "go.mod": "module example.com/m\n",
        "main.go": "package main\nfunc main() {}\n",
    }
    bad_file = {"internal/pkg/foo.go": "//go:build linux\n\npackage foo\n"}

    base_profile = analyze(
        _PROJECT_ID, _FULL_PATH, _BRANCH, _COMMIT, None, _rc(base_files)
    )
    mixed_profile = analyze(
        _PROJECT_ID,
        _FULL_PATH,
        _BRANCH,
        _COMMIT,
        None,
        _rc({**base_files, **bad_file}),
    )

    assert base_profile.abstract_inputs == mixed_profile.abstract_inputs
    assert base_profile.abstract_outputs == mixed_profile.abstract_outputs
    assert (
        base_profile.external_service_dependencies
        == mixed_profile.external_service_dependencies
    )
    assert (
        base_profile.database_table_dependencies
        == mixed_profile.database_table_dependencies
    )
    # The well-formed base produces zero structured entries; the
    # mixed repo adds three (one per non-purpose section, because the
    # bad file is not eligible for purpose).
    base_structured = [s for s in base_profile.degraded_sections if ": skipped " in s]
    mixed_structured = [s for s in mixed_profile.degraded_sections if ": skipped " in s]
    assert base_structured == []
    assert mixed_structured == [
        "abstract_io: skipped internal/pkg/foo.go (build constraint requires toolchain)",
        "external_services: skipped internal/pkg/foo.go (build constraint requires toolchain)",
        "database_tables: skipped internal/pkg/foo.go (build constraint requires toolchain)",
    ]


def test_analyze_degraded_sections_remains_list_of_str() -> None:
    """``degraded_sections`` is ``list[str]`` regardless of entry shape.

    Downstream consumers receive a heterogeneous list (plain section
    names recorded on sub-analyzer failure, plus structured
    ``"<section>: skipped <path> (<reason>)"`` strings), but the
    field's ``list[str]`` shape is preserved (design §7 "Skip
    surfacing").
    """

    rc = _rc(
        {
            "go.mod": "module example.com/m\n",
            "main.go": 'package main\nimport "C"\nfunc main() {}\n',
        }
    )

    profile = analyze(_PROJECT_ID, _FULL_PATH, _BRANCH, _COMMIT, None, rc)

    assert isinstance(profile.degraded_sections, list)
    for entry in profile.degraded_sections:
        assert isinstance(entry, str)
        assert entry != ""

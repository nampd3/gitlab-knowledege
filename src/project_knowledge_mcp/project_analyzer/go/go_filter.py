"""Vendor-directory exclusion and repo-level Go-presence guard.

This module exposes two pure predicates that the aggregator uses before
invoking any Go sub-analyzer:

* :func:`is_go_source_file` decides whether a single repository-relative
  path identifies a Go source file the analyzer should inspect. It
  rejects any path containing a directory segment equal to ``vendor``
  after normalizing ``\\`` to ``/``, and requires a case-sensitive
  ``.go`` suffix. Vendored copies of third-party libraries are observed
  in all four sample repositories and must not contribute false
  detections (Requirement 1.3).

* :func:`has_go_artefacts` decides whether a Repository_Contents bundle
  contains anything Go-related at all. It is the single guard the
  aggregator consults before doing any Go work; when it returns
  ``False`` the aggregator short-circuits and the Project_Profile is
  produced exactly as it would have been before this feature was added
  (Requirements 1.4, 11.5).

Both predicates are pure: they have no side effects, perform no I/O,
and depend only on their arguments.

Implements Requirements 1.3, 1.4, 11.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from project_knowledge_mcp.models import RepositoryContents


__all__ = [
    "has_go_artefacts",
    "is_go_source_file",
]


#: Case-sensitive suffix that identifies a Go source file. The Go
#: language specification mandates a lowercase ``.go`` extension; files
#: named ``foo.GO`` or ``foo.Go`` are not Go source files and must not
#: be inspected (Requirement 1.3 carve-out for case sensitivity).
_GO_SUFFIX: str = ".go"

#: Directory segment name (case-sensitive) that marks a vendored copy of
#: a third-party library. Any path containing this segment is excluded
#: from Go analysis (Requirement 1.3).
_VENDOR_SEGMENT: str = "vendor"

#: Repository-root path of the Go module manifest. Presence of this file
#: is one of the two signals that triggers Go-aware behavior in the
#: aggregator (Requirement 2.1).
_GO_MOD_PATH: str = "go.mod"


def is_go_source_file(path: str) -> bool:
    """Return ``True`` when ``path`` identifies a Go source file to scan.

    A path qualifies when both of the following hold:

    * It ends with the suffix ``.go`` (case-sensitive).
    * After normalizing ``\\`` to ``/``, none of its directory segments
      equals ``vendor`` (case-sensitive). The final path component is
      excluded from this check, so a file literally named ``vendor.go``
      at any depth still qualifies; only directory segments named
      ``vendor`` cause exclusion.

    The function is a pure predicate; it does not consult
    :class:`~project_knowledge_mcp.models.RepositoryContents` and
    performs no I/O.
    """

    if not path.endswith(_GO_SUFFIX):
        return False

    normalized = path.replace("\\", "/")

    # Inspect every directory segment, which is every component except the
    # final filename. A leading "/" produces an empty first segment that
    # cannot equal "vendor"; a trailing "/" cannot occur on a file path
    # that ends with ".go".
    segments = normalized.split("/")
    directory_segments = segments[:-1]
    return _VENDOR_SEGMENT not in directory_segments


def has_go_artefacts(repository_contents: RepositoryContents) -> bool:
    """Return ``True`` when ``repository_contents`` looks like a Go repo.

    The bundle qualifies when either:

    * at least one path in the bundle satisfies
      :func:`is_go_source_file`, or
    * the bundle contains a path equal to exactly ``"go.mod"`` (the
      repository-root Go module manifest).

    The aggregator calls this once per ``analyze()``; when it returns
    ``False`` no Go code path runs and the produced Project_Profile is
    byte-identical to the output of the pre-feature analyzer
    (Requirement 11.5).
    """

    files = repository_contents.files
    if _GO_MOD_PATH in files:
        return True
    return any(is_go_source_file(path) for path in files)

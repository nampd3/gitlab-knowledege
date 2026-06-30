"""Go-specific purpose-summary candidate collector.

Implements the Go layer of the purpose summarizer (Requirements 2.1,
2.2, 2.3, 2.4, 2.5, 2.7). Returns the three Go-specific candidates as
a structured record; the aggregator interleaves these into the parent
spec's existing candidate order at the documented priority positions
(Requirement 2.6).

The Go purpose helper sources only two kinds of input:

1. The repository-root ``go.mod``, parsed by
   :func:`go_parser.parse_go_mod`. The ``module <module-path>`` line
   yields up to two candidates:

   * ``gomod_comment`` — the body of a ``//`` comment immediately
     preceding the line (no blank-line gap) or, when no leading
     comment is present, the body of a same-line trailing
     ``//`` comment, after stripping ``//`` and surrounding whitespace
     (Requirement 2.2).
   * ``gomod_module_path`` — the module path with any leading
     ``<host>/<org>/`` prefix stripped: ``github.com/acme/payment-service``
     becomes ``payment-service``. Bare module names without ``/``
     separators pass through unchanged (Requirement 2.3).

2. The :class:`PackageDocCommentEvent` of any Go source file at the
   repository root, at the path ``cmd/main.go``, or at any path
   matching ``cmd/<name>/main.go``. The first non-empty doc comment
   in sorted-path order wins (Requirement 2.4).

Length cap and whitespace normalization (Requirement 2.5) are applied
via the existing parent-spec helpers
:func:`purpose._normalize_description` and :func:`purpose._truncate`,
so Go candidates and non-Go candidates pass through identical sizing
rules.

Requirement 2.7 (viper exclusion) is satisfied by construction: this
helper only inspects ``go.mod`` and the package doc-comment block, and
neither carries arguments passed to viper calls.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.project_analyzer.go._events import (
    GoPurposeCandidates,
    PackageDocCommentEvent,
)
from project_knowledge_mcp.project_analyzer.go.go_parser import parse_go_mod
from project_knowledge_mcp.project_analyzer.purpose import (
    _normalize_description,
    _truncate,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from project_knowledge_mcp.models import RepositoryContents
    from project_knowledge_mcp.project_analyzer.go._events import (
        GoEvent,
        ModFileModuleEvent,
    )


__all__ = ["collect_go_candidates", "is_eligible_main_path"]


#: Path of the Go module manifest at the repository root.
_GO_MOD_PATH: Final[str] = "go.mod"

#: Repository-root path of the canonical ``cmd/main.go`` entry point.
_CMD_MAIN_PATH: Final[str] = "cmd/main.go"

#: Pattern matching ``cmd/<name>/main.go`` (one directory deep under ``cmd``).
_RE_CMD_NAMED_MAIN: Final[re.Pattern[str]] = re.compile(r"^cmd/[^/]+/main\.go$")

#: Number of slash-separated segments in a module path that triggers
#: ``<host>/<org>/`` prefix stripping. ``github.com/acme/payment-service``
#: has exactly three segments and is the canonical worked example in
#: Requirement 2.3.
_MODULE_PATH_PREFIXED_SEGMENTS: Final[int] = 3


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect_go_candidates(
    repository_contents: RepositoryContents,
    events_by_file: Mapping[str, list[GoEvent]],
) -> GoPurposeCandidates:
    """Return the three Go-specific purpose-summary candidates.

    Args:
        repository_contents: The fetched repository snapshot. Used to
            read the repository-root ``go.mod`` file when present.
        events_by_file: Mapping from repository-relative Go-source path
            to the recognizer event stream for that file. Inspected for
            :class:`PackageDocCommentEvent` instances on files at the
            repository root, at ``cmd/main.go``, or at
            ``cmd/<name>/main.go``.

    Returns:
        A :class:`GoPurposeCandidates` record. Each field is either a
        non-empty, normalized-and-truncated candidate string or
        ``None`` when no candidate could be derived.

    The aggregator (Requirement 2.6) selects among these candidates
    and the existing non-Go candidates by priority; this helper does
    not itself decide which candidate wins.
    """

    gomod_comment, gomod_module_path = _gomod_candidates(repository_contents)
    package_doc_comment = _package_doc_candidate(events_by_file)

    return GoPurposeCandidates(
        gomod_comment=gomod_comment,
        gomod_module_path=gomod_module_path,
        package_doc_comment=package_doc_comment,
    )


# ---------------------------------------------------------------------------
# go.mod candidate extraction
# ---------------------------------------------------------------------------


def _gomod_candidates(
    rc: RepositoryContents,
) -> tuple[str | None, str | None]:
    """Return ``(gomod_comment, gomod_module_path)`` from the root ``go.mod``.

    Returns ``(None, None)`` when ``go.mod`` is absent or contains no
    well-formed ``module <module-path>`` declaration.
    """

    text = rc.read_text(_GO_MOD_PATH)
    if text is None:
        return None, None

    event = parse_go_mod(text)
    if event is None:
        return None, None

    return _gomod_comment(event), _gomod_module_path(event)


def _gomod_comment(event: ModFileModuleEvent) -> str | None:
    """Return the leading or trailing ``module``-line comment as a candidate.

    Per Requirement 2.2, when both leading and trailing comments are
    present the leading comment wins. When neither yields a non-empty
    body after normalization, returns ``None``.
    """

    raw = event.leading_comment or event.trailing_comment
    return _candidate(raw)


def _gomod_module_path(event: ModFileModuleEvent) -> str | None:
    """Return the module path with leading ``<host>/<org>/`` prefix stripped.

    A path with three or more slash-separated segments is treated as
    ``<host>/<org>/<rest>`` and reduced to ``<rest>``. Paths with fewer
    segments (bare module names like ``repayment_service`` or
    ``cat-service``) pass through unchanged (Requirement 2.3).
    """

    parts = event.module_path.split("/")
    if len(parts) >= _MODULE_PATH_PREFIXED_SEGMENTS:
        candidate = "/".join(parts[2:])
    else:
        candidate = event.module_path
    return _candidate(candidate)


# ---------------------------------------------------------------------------
# Package doc-comment candidate extraction
# ---------------------------------------------------------------------------


def _package_doc_candidate(
    events_by_file: Mapping[str, list[GoEvent]],
) -> str | None:
    """Return the first non-empty package-doc-comment from an eligible file.

    Eligible files are Go source files at the repository root, at
    ``cmd/main.go``, or at any path matching ``cmd/<name>/main.go``.
    Files are inspected in sorted-path order (Requirement 2.4).
    """

    for path in sorted(events_by_file):
        if not is_eligible_main_path(path):
            continue
        for event in events_by_file[path]:
            if not isinstance(event, PackageDocCommentEvent):
                continue
            candidate = _candidate(event.text)
            if candidate is not None:
                return candidate
    return None


def is_eligible_main_path(path: str) -> bool:
    """Return ``True`` when ``path`` is an eligible package-doc source.

    Eligible paths:

    * any ``.go`` file at the repository root (no ``/`` separator),
    * the literal ``cmd/main.go``,
    * any ``cmd/<name>/main.go`` with exactly one directory under
      ``cmd``.

    This predicate is the canonical filter for "files the purpose
    summarizer would have processed", and is reused by the aggregator
    when surfacing :class:`SkipFileEvent` reasons under the
    ``"purpose"`` section in ``degraded_sections`` (task 11.6).
    """

    if path.endswith(".go") and "/" not in path:
        return True
    if path == _CMD_MAIN_PATH:
        return True
    return bool(_RE_CMD_NAMED_MAIN.match(path))


# ---------------------------------------------------------------------------
# Shared normalization + truncation
# ---------------------------------------------------------------------------


def _candidate(raw: str | None) -> str | None:
    """Normalize ``raw`` and return ``None`` when the result is blank.

    Applies the parent-spec ``_normalize_description`` (collapse
    internal whitespace; strip; ``None`` when blank) followed by
    ``_truncate`` (cap to ``PURPOSE_SUMMARY_MAX_LEN`` at a word
    boundary when possible). Both helpers are imported from
    :mod:`project_knowledge_mcp.project_analyzer.purpose` so Go
    candidates pass through identical sizing rules to non-Go
    candidates (Requirement 2.5).
    """

    normalized = _normalize_description(raw)
    if normalized is None:
        return None
    return _truncate(normalized)

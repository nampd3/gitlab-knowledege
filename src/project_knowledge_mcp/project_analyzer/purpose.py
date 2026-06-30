"""Purpose summarizer for the Project_Analyzer.

This sub-analyzer derives a project's purpose summary from the sources
permitted by Requirement 3.2:

1. **README files** at the repository root (any path matching ``README*``,
   case-insensitive, with no directory separator). Headings, badge lines,
   and Setext-underline blocks are skipped; the first prose paragraph
   wins.
2. **GitLab repository description** passed in from enumeration. Internal
   whitespace is collapsed to single spaces.
3. **Package manifest description**, in priority order:

   * the ``description`` field of a root-level ``package.json``,
   * the ``[project] description`` (or ``[tool.poetry] description``) of
     a root-level ``pyproject.toml``,
   * the top-level ``<description>`` element of a root-level ``pom.xml``.

4. **Top-level module docstrings**:

   * the module docstring of any root-level ``*.py`` file,
   * the module docstring of any ``src/<pkg>/__init__.py`` file,
   * the leading ``/* ... */`` block comment of a root-level ``*.js`` /
     ``*.mjs`` / ``*.cjs`` / ``*.ts`` file.

Per Requirement 3.2 any single source is sufficient. The summarizer
inspects the sources in the priority order above and uses the first
source that yields non-empty content.

The produced summary is truncated to at most
:data:`~project_knowledge_mcp.models.PURPOSE_SUMMARY_MAX_LEN` characters
(Requirement 3.4); when the input exceeds the limit the cut is taken at
the last whitespace at or before the limit so words are not split, with
a hard cut as the fallback for inputs that contain no whitespace.

When no source yields content the function returns the canonical pair
``(UNKNOWN_PURPOSE_SUMMARY, INSUFFICIENT_SOURCE_MATERIAL_REASON)``
(Requirement 3.3).

The function never raises: malformed manifests (invalid JSON, invalid
TOML, invalid XML, or unparseable Python) are silently skipped so a
single broken file in a project does not poison the analyzer for the
rest of the portfolio.

Implements Requirements 3.1, 3.2, 3.3, 3.4.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from project_knowledge_mcp.models import (
    INSUFFICIENT_SOURCE_MATERIAL_REASON,
    PURPOSE_SUMMARY_MAX_LEN,
    UNKNOWN_PURPOSE_SUMMARY,
)

if TYPE_CHECKING:
    from project_knowledge_mcp.models import RepositoryContents


__all__ = [
    "PurposeCandidate",
    "collect_purpose_candidates",
    "summarize_purpose",
]


# ---------------------------------------------------------------------------
# Public candidate record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PurposeCandidate:
    """A single non-empty purpose-summary candidate from one source.

    Attributes:
        source: A short identifier of the source kind, e.g. ``"readme"``,
            ``"gitlab_description"``, ``"manifest"``, ``"module_docstring"``.
            Other layers (the Go sub-analyzer) contribute candidates with
            their own source identifiers.
        text: The whitespace-normalized text content of the candidate.
            Always non-empty; empty/missing sources are filtered out by
            the producer.
        priority: The candidate's position in the documented purpose
            summary priority order (lower wins). The numeric values
            follow the slots reserved in the design document so that
            language-specific producers can interleave their candidates
            at the correct position without renumbering.
    """

    source: str
    text: str
    priority: int


# ---------------------------------------------------------------------------
# Priority slots — match the design's documented purpose-summary order so
# language-specific producers (e.g. the Go sub-analyzer) can interleave
# candidates at slots 3, 4, and 7 without renumbering this module.
# ---------------------------------------------------------------------------

#: Priority of a root-level README file (slot 1).
_PRIORITY_README: int = 1

#: Priority of the GitLab repository description (slot 2).
_PRIORITY_GITLAB_DESCRIPTION: int = 2

#: Priority of a root-level package manifest description (slot 5).
_PRIORITY_MANIFEST: int = 5

#: Priority of a top-level module docstring or JS leading block comment (slot 6).
_PRIORITY_MODULE_DOCSTRING: int = 6


# ---------------------------------------------------------------------------
# Module-level constants and patterns
# ---------------------------------------------------------------------------

#: Case-insensitive prefix that identifies a README file (Requirement 3.2).
_README_PREFIX: str = "README"

#: Manifest filenames inspected for a description field, in priority order.
_MANIFEST_FILES: tuple[str, ...] = ("package.json", "pyproject.toml", "pom.xml")

#: Suffixes recognized as JS-family source files for leading block comments.
_JS_SUFFIXES: tuple[str, ...] = (".js", ".mjs", ".cjs", ".ts")

# A line that is "just" an ATX heading: ``# foo``, ``## bar``, etc.
_ATX_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")

# A Setext-style underline: a line consisting of only ``=`` or ``-``.
_SETEXT_UNDERLINE_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")

# A line whose entire content is a Markdown badge link of the form
# ``[![alt](src)](href)``. Multiple badges separated by whitespace on the
# same line are also accepted.
_BADGE_LINE_RE = re.compile(
    r"^\s*(?:\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)\s*)+$",
)

# Leading C-style block comment at file start (allowing leading whitespace).
_JS_BLOCK_COMMENT_RE = re.compile(r"^\s*/\*+(.*?)\*+/", re.DOTALL)

# A Setext heading paragraph has at least the title line plus the underline
# line; this is named to keep ``ruff`` happy about magic numbers.
_SETEXT_MIN_PARAGRAPH_LINES: int = 2

# A ``src/<pkg>/__init__.py`` path has exactly three segments.
_SRC_INIT_PATH_SEGMENTS: int = 3


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def summarize_purpose(
    repo_contents: RepositoryContents,
    repository_description: str | None,
) -> tuple[str, str | None]:
    """Derive a purpose summary for a project.

    Args:
        repo_contents: The fetched repository contents at the analysis
            commit. Used to read README files, package manifests, and
            top-level module docstrings.
        repository_description: The GitLab repository description, when
            available, as supplied by enumeration.

    Returns:
        A ``(summary, reason)`` tuple. When at least one source yields
        content, ``summary`` is that content (normalized and truncated to
        :data:`PURPOSE_SUMMARY_MAX_LEN` characters at a word boundary
        when possible) and ``reason`` is ``None``. When no source yields
        content, the tuple is
        ``(UNKNOWN_PURPOSE_SUMMARY, INSUFFICIENT_SOURCE_MATERIAL_REASON)``.

    The returned ``summary`` always satisfies
    ``len(summary) <= PURPOSE_SUMMARY_MAX_LEN`` (Requirement 3.4).
    """
    candidates = collect_purpose_candidates(repo_contents)

    gitlab = _normalize_description(repository_description)
    if gitlab:
        candidates.append(
            PurposeCandidate(
                source="gitlab_description",
                text=gitlab,
                priority=_PRIORITY_GITLAB_DESCRIPTION,
            )
        )

    # Stable sort by priority preserves the original relative order
    # of the existing four sources (README → GitLab → manifest →
    # docstring) and leaves room for Go-specific candidates at slots
    # 3, 4, and 7 to be interleaved by the aggregator.
    candidates.sort(key=lambda candidate: candidate.priority)

    if candidates:
        return _truncate(candidates[0].text), None

    return UNKNOWN_PURPOSE_SUMMARY, INSUFFICIENT_SOURCE_MATERIAL_REASON


def collect_purpose_candidates(
    repo_contents: RepositoryContents,
) -> list[PurposeCandidate]:
    """Collect non-empty purpose-summary candidates from repository contents.

    Returns the candidates that can be derived from
    :class:`RepositoryContents` alone, in their documented priority order:

    * priority 1 — first prose paragraph of any root-level ``README*``
      file (Requirement 3.2 first source).
    * priority 5 — ``description`` field of a recognized root-level
      package manifest (``package.json``, ``pyproject.toml``, or
      ``pom.xml``).
    * priority 6 — module docstring of a top-level Python file or
      ``src/<pkg>/__init__.py``, or the leading ``/* ... */`` block
      comment of a root-level JS-family file.

    The GitLab repository description (priority 2) is supplied
    separately by :func:`summarize_purpose` because it is passed in from
    project enumeration rather than read from ``repo_contents``.
    Language-specific candidates produced by sibling sub-analyzers (for
    example, the Go sub-analyzer's ``gomod`` candidates at priorities 3
    and 4 and the package-doc-comment candidate at priority 7) are
    interleaved by the aggregator.

    Only candidates whose extracted text is non-empty are returned.
    """
    candidates: list[PurposeCandidate] = []

    readme = _extract_readme(repo_contents)
    if readme:
        candidates.append(
            PurposeCandidate(
                source="readme",
                text=readme,
                priority=_PRIORITY_README,
            )
        )

    manifest = _extract_manifest_description(repo_contents)
    if manifest:
        candidates.append(
            PurposeCandidate(
                source="manifest",
                text=manifest,
                priority=_PRIORITY_MANIFEST,
            )
        )

    docstring = _extract_top_level_docstring(repo_contents)
    if docstring:
        candidates.append(
            PurposeCandidate(
                source="module_docstring",
                text=docstring,
                priority=_PRIORITY_MODULE_DOCSTRING,
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _normalize_description(value: str | None) -> str | None:
    """Collapse internal whitespace and strip; return ``None`` when blank."""
    if value is None:
        return None
    collapsed = " ".join(value.split())
    return collapsed or None


def _truncate(value: str) -> str:
    """Truncate ``value`` to at most ``PURPOSE_SUMMARY_MAX_LEN`` characters.

    When the cut falls inside a word, back up to the last whitespace
    character at or before the limit so words are not split, then strip
    trailing whitespace. When the prefix contains no whitespace at all,
    fall back to a hard cut at the limit.
    """
    if len(value) <= PURPOSE_SUMMARY_MAX_LEN:
        return value
    cut = value[:PURPOSE_SUMMARY_MAX_LEN]
    last_ws = max(cut.rfind(" "), cut.rfind("\t"), cut.rfind("\n"))
    if last_ws > 0:
        trimmed = cut[:last_ws].rstrip()
        if trimmed:
            return trimmed
    return cut


def _is_root_path(path: str) -> bool:
    """Return ``True`` when ``path`` is a repository-root file (no separators)."""
    return "/" not in path and "\\" not in path


# ---------------------------------------------------------------------------
# README extraction
# ---------------------------------------------------------------------------


def _extract_readme(rc: RepositoryContents) -> str | None:
    """Return the first prose paragraph from any root-level ``README*`` file.

    Files are considered in sorted-path order so the choice is
    deterministic. Matching is case-insensitive on the basename's leading
    ``README`` token.
    """
    for path in rc.file_paths:
        if not _is_root_path(path):
            continue
        if not path.upper().startswith(_README_PREFIX):
            continue
        text = rc.read_text(path)
        if text is None:
            continue
        prose = _extract_readme_prose(text)
        if prose:
            return prose
    return None


def _extract_readme_prose(content: str) -> str | None:
    """Return the first paragraph of prose, skipping headings and badges.

    A paragraph is a run of non-blank lines separated by one or more
    blank lines. Within each paragraph:

    * a paragraph whose final non-blank line is a Setext underline
      (``===`` or ``---``) is treated as a heading and skipped entirely,
    * lines that are ATX headings (``# foo``) are dropped,
    * lines whose entire content is a Markdown badge link are dropped.

    The first paragraph that has at least one surviving line wins; its
    surviving lines are joined with single spaces.
    """
    paragraphs = re.split(r"\n\s*\n", content)
    for paragraph in paragraphs:
        lines = paragraph.split("\n")
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            continue
        # Whole-paragraph Setext heading: skip.
        if len(non_empty) >= _SETEXT_MIN_PARAGRAPH_LINES and _SETEXT_UNDERLINE_RE.match(
            non_empty[-1]
        ):
            continue
        kept: list[str] = []
        for line in non_empty:
            if _ATX_HEADING_RE.match(line):
                continue
            if _SETEXT_UNDERLINE_RE.match(line):
                continue
            if _BADGE_LINE_RE.match(line):
                continue
            kept.append(line.strip())
        if kept:
            return " ".join(kept)
    return None


# ---------------------------------------------------------------------------
# Manifest extraction
# ---------------------------------------------------------------------------


def _extract_manifest_description(rc: RepositoryContents) -> str | None:
    """Return a description string from a recognized root-level manifest.

    Manifests are tried in :data:`_MANIFEST_FILES` order. The first one
    that yields non-empty content wins. Malformed manifests are silently
    skipped.
    """
    for path in _MANIFEST_FILES:
        text = rc.read_text(path)
        if text is None:
            continue
        if path == "package.json":
            result = _extract_package_json_description(text)
        elif path == "pyproject.toml":
            result = _extract_pyproject_description(text)
        else:
            result = _extract_pom_xml_description(text)
        if result:
            return result
    return None


def _extract_package_json_description(text: str) -> str | None:
    """Return ``description`` from a root-level ``package.json``."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    description = data.get("description")
    if not isinstance(description, str):
        return None
    return _normalize_description(description)


def _extract_pyproject_description(text: str) -> str | None:
    """Return ``[project].description`` (or ``[tool.poetry].description``)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None

    project_table = data.get("project")
    if isinstance(project_table, dict):
        description = project_table.get("description")
        if isinstance(description, str):
            normalized = _normalize_description(description)
            if normalized is not None:
                return normalized

    tool_table = data.get("tool")
    if isinstance(tool_table, dict):
        poetry_table = tool_table.get("poetry")
        if isinstance(poetry_table, dict):
            description = poetry_table.get("description")
            if isinstance(description, str):
                normalized = _normalize_description(description)
                if normalized is not None:
                    return normalized

    return None


def _extract_pom_xml_description(text: str) -> str | None:
    """Return the top-level ``<description>`` element of a root-level ``pom.xml``.

    Handles the conventional Maven namespace by stripping the namespace
    prefix from each child element's local name before comparison.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    for child in root:
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name == "description":
            return _normalize_description(child.text)
    return None


# ---------------------------------------------------------------------------
# Top-level module docstring extraction
# ---------------------------------------------------------------------------


def _extract_top_level_docstring(rc: RepositoryContents) -> str | None:
    """Return a module-level docstring from a top-level source file.

    "Top-level" means either a file at the repository root or, for
    Python, a ``src/<pkg>/__init__.py`` package init file (a common
    Python project layout). Files are considered in sorted-path order so
    the result is deterministic. Files that fail to parse are silently
    skipped.
    """
    for path in sorted(rc.file_paths):
        text = rc.read_text(path)
        if text is None:
            continue

        if _is_python_top_level(path):
            extracted = _extract_python_module_docstring(text)
            if extracted:
                return extracted
            continue

        if _is_root_path(path) and path.endswith(_JS_SUFFIXES):
            extracted = _extract_js_leading_block_comment(text)
            if extracted:
                return extracted

    return None


def _is_python_top_level(path: str) -> bool:
    """Return ``True`` for root ``*.py`` files or ``src/<pkg>/__init__.py``."""
    if _is_root_path(path) and path.endswith(".py"):
        return True
    parts = path.split("/")
    # Match exactly src/<pkg>/__init__.py: three segments, first is ``src``,
    # last is ``__init__.py``.
    return (
        len(parts) == _SRC_INIT_PATH_SEGMENTS
        and parts[0] == "src"
        and parts[2] == "__init__.py"
    )


def _extract_python_module_docstring(text: str) -> str | None:
    """Return the module docstring of ``text``, normalized."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        # ValueError covers null-byte content; SyntaxError covers any
        # other parse failure. Either way, treat as "no docstring".
        return None
    docstring = ast.get_docstring(tree)
    return _normalize_description(docstring)


def _extract_js_leading_block_comment(text: str) -> str | None:
    """Return the leading ``/* ... */`` block comment as a single line.

    Per-line ``*`` decoration is stripped. Empty lines are dropped. The
    surviving lines are joined with single spaces.
    """
    match = _JS_BLOCK_COMMENT_RE.match(text)
    if match is None:
        return None
    body = match.group(1)
    lines: list[str] = []
    for raw_line in body.split("\n"):
        cleaned = raw_line.strip()
        if cleaned.startswith("*"):
            cleaned = cleaned[1:].strip()
        if cleaned:
            lines.append(cleaned)
    if not lines:
        return None
    return " ".join(lines)

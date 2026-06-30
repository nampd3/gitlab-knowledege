# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 5: For all repositories that have no README, no GitLab repository description, and no analyzable source-code metadata containing content from which a purpose summary could be derived, the Project_Analyzer SHALL produce purpose_summary == "unknown" and purpose_summary_reason == "insufficient source material".
"""Property test for content-free repositories yielding the canonical "unknown" result.

**Validates Requirement 3.3** (Property 5 in the design).

For every repository that carries no source material from which a purpose
summary could be derived -- i.e. every potential source the
``Project_Analyzer`` is allowed to consult is either absent or empty --
the standalone summarizer
:func:`project_knowledge_mcp.project_analyzer.purpose.summarize_purpose`
must return the canonical pair
``(UNKNOWN_PURPOSE_SUMMARY, INSUFFICIENT_SOURCE_MATERIAL_REASON)``,
i.e. ``("unknown", "insufficient source material")``.

The four sources the summarizer consults (per Requirement 3.2) are:

1. **README files at the repository root** (any path matching ``README*``,
   case-insensitive, with no directory separator).
2. **The GitLab repository description**, passed in from enumeration.
3. **A package manifest description**: ``package.json``,
   ``pyproject.toml`` (``[project] description`` or
   ``[tool.poetry] description``), or ``pom.xml`` (``<description>``).
4. **A top-level module docstring**: any root-level ``*.py`` file or any
   ``src/<pkg>/__init__.py``, or the leading ``/* ... */`` block comment
   of a root-level ``*.{js,mjs,cjs,ts}`` file.

The repository generator below produces inputs in which every one of
these sources is independently rendered content-free (missing field,
empty/whitespace-only value, headings/badges only, malformed manifest,
or simply absent), and additionally sprinkles in *non-source* files
that the design intentionally excludes (subdirectory READMEs, nested
manifests, non-top-level Python files with docstrings) to confirm those
files do not bypass the rule. The test asserts the canonical pair on
every generated input.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import (
    INSUFFICIENT_SOURCE_MATERIAL_REASON,
    UNKNOWN_PURPOSE_SUMMARY,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.purpose import summarize_purpose

# ---------------------------------------------------------------------------
# Whitespace / blank strategies
# ---------------------------------------------------------------------------

# Strings consisting only of whitespace, including the empty string. The
# summarizer's whitespace collapser reduces every member of this set to
# the empty string, which the candidate filter then treats as "no
# content".
_WHITESPACE_ONLY = st.text(
    alphabet=st.sampled_from([" ", "\t", "\n", "\r"]),
    min_size=0,
    max_size=20,
)

# Whitespace strategy safe for embedding inside a TOML basic string
# (no LF or CR, which would make the value an invalid TOML literal). A
# malformed TOML file would still return ``None`` from the summarizer's
# extractor, but keeping these inline keeps the generated TOML files
# realistic and ensures we are exercising the "valid TOML, empty
# description" branch in addition to the malformed branch.
_WHITESPACE_ONLY_INLINE = st.text(
    alphabet=st.sampled_from([" ", "\t"]),
    min_size=0,
    max_size=20,
)

# A repository description value that the summarizer must treat as
# absent: either ``None`` or a whitespace-only string.
_NONE_OR_BLANK = st.one_of(st.none(), _WHITESPACE_ONLY)


# ---------------------------------------------------------------------------
# README content strategies (content-free)
# ---------------------------------------------------------------------------

# README bodies that the prose extractor strips down to nothing. Each of
# the following is content-free per the summarizer's rules:
# * empty / whitespace-only,
# * ATX-heading-only paragraphs,
# * Setext-heading paragraphs (``Title\n=====`` / ``Subtitle\n-----``),
# * Markdown badge-only paragraphs,
# * combinations of the above.
_CONTENT_FREE_README_BODIES = st.sampled_from(
    [
        "",
        "   ",
        "\n\n\n",
        "   \n   \n   \n",
        "# Title only\n",
        "## Sub heading\n\n### Deeper\n",
        "# Title\n## Sub\n### Deeper\n",
        "Title\n=====\n",
        "Subtitle\n--------\n",
        "Title\n=====\n\nSubtitle\n--------\n",
        "[![CI](https://x/y.svg)](https://x/y)\n",
        "[![A](https://x/a.svg)](https://x/a) [![B](https://x/b.svg)](https://x/b)\n",
        "# Heading\n\n[![Build](https://x.svg)](https://x)\n",
        "[![A](https://x.svg)](https://x)\n[![B](https://y.svg)](https://y)\n",
    ]
)

# Conventional README basenames at the repository root.
_README_BASENAMES = st.sampled_from(
    [
        "README",
        "README.md",
        "README.rst",
        "README.txt",
        "readme.md",
        "Readme.markdown",
        "README.adoc",
    ]
)


# ---------------------------------------------------------------------------
# Manifest strategies (content-free)
# ---------------------------------------------------------------------------


@st.composite
def _content_free_package_json(draw: st.DrawFn) -> str:  # noqa: PLR0911
    """A ``package.json`` whose ``description`` (if any) yields no content."""
    kind = draw(
        st.sampled_from(
            [
                "no_description_field",
                "empty_description",
                "whitespace_description",
                "null_description",
                "non_string_description",
                "non_object_root",
                "malformed_json",
            ]
        )
    )
    if kind == "no_description_field":
        return json.dumps({"name": "synthetic", "version": "0.0.1"})
    if kind == "empty_description":
        return json.dumps({"name": "x", "description": ""})
    if kind == "whitespace_description":
        return json.dumps({"name": "x", "description": draw(_WHITESPACE_ONLY)})
    if kind == "null_description":
        return json.dumps({"name": "x", "description": None})
    if kind == "non_string_description":
        return json.dumps({"name": "x", "description": 42})
    if kind == "non_object_root":
        return json.dumps([1, 2, 3])
    return "{not valid json"


@st.composite
def _content_free_pyproject_toml(draw: st.DrawFn) -> str:  # noqa: PLR0911
    """A ``pyproject.toml`` whose description (if any) yields no content."""
    kind = draw(
        st.sampled_from(
            [
                "completely_empty",
                "no_project_table",
                "project_no_description",
                "project_empty_description",
                "project_whitespace_description",
                "tool_poetry_no_description",
                "tool_poetry_empty_description",
                "tool_poetry_whitespace_description",
                "malformed_toml",
            ]
        )
    )
    if kind == "completely_empty":
        return ""
    if kind == "no_project_table":
        return '[build-system]\nrequires = ["hatchling"]\n'
    if kind == "project_no_description":
        return '[project]\nname = "x"\nversion = "0.1.0"\n'
    if kind == "project_empty_description":
        return '[project]\nname = "x"\ndescription = ""\n'
    if kind == "project_whitespace_description":
        ws = draw(_WHITESPACE_ONLY_INLINE)
        return f'[project]\nname = "x"\ndescription = "{ws}"\n'
    if kind == "tool_poetry_no_description":
        return '[tool.poetry]\nname = "x"\nversion = "0.1.0"\n'
    if kind == "tool_poetry_empty_description":
        return '[tool.poetry]\nname = "x"\ndescription = ""\n'
    if kind == "tool_poetry_whitespace_description":
        ws = draw(_WHITESPACE_ONLY_INLINE)
        return f'[tool.poetry]\nname = "x"\ndescription = "{ws}"\n'
    return "not a [valid toml file"


@st.composite
def _content_free_pom_xml(draw: st.DrawFn) -> str:
    """A ``pom.xml`` whose ``<description>`` (if any) yields no content."""
    kind = draw(
        st.sampled_from(
            [
                "no_description_element",
                "empty_description",
                "self_closing_description",
                "whitespace_description",
                "namespaced_no_description",
                "malformed_xml",
            ]
        )
    )
    if kind == "no_description_element":
        return '<?xml version="1.0" encoding="UTF-8"?><project><name>x</name></project>'
    if kind == "empty_description":
        return '<?xml version="1.0"?><project><description></description></project>'
    if kind == "self_closing_description":
        return '<?xml version="1.0"?><project><description/></project>'
    if kind == "whitespace_description":
        ws = draw(_WHITESPACE_ONLY)
        return (
            '<?xml version="1.0"?><project><description>'
            f"{ws}</description></project>"
        )
    if kind == "namespaced_no_description":
        return (
            '<?xml version="1.0"?>'
            '<project xmlns="http://maven.apache.org/POM/4.0.0">'
            "<name>x</name></project>"
        )
    return "<not-closed-tag>"


# ---------------------------------------------------------------------------
# Top-level Python source strategies (content-free)
# ---------------------------------------------------------------------------

# Python file contents whose modules have no docstring. Each of these
# parses successfully but ``ast.get_docstring`` returns ``None``: empty
# files, comment-only files, and files whose first statement is an
# ``Import``, ``ImportFrom``, ``Assign``, ``AnnAssign``, ``FunctionDef``,
# ``ClassDef``, or ``If`` (none of which is a string-literal expression).
_PYTHON_NO_DOCSTRING = st.sampled_from(
    [
        "",
        "\n\n",
        "# only a comment\n",
        "# header comment\n# another comment\n",
        "import os\n",
        "from typing import Final\n\nVERSION: Final = '0.1.0'\n",
        "def main() -> None:\n    pass\n",
        "class Foo:\n    pass\n",
        "x = 1\ny = 2\n",
        "if __name__ == '__main__':\n    pass\n",
    ]
)

# JS-family file contents that do not begin with a ``/* ... */`` block
# comment. ``//`` line comments do not count, and code-only files do
# not match the leading-block-comment regex either.
_JS_NO_BLOCK_COMMENT = st.sampled_from(
    [
        "",
        "\n\n",
        "// just a line comment\n",
        "export const x = 1;\n",
        "export default function foo() { return 42; }\n",
        "import { foo } from './foo';\nexport { foo };\n",
        "// header\n// banner\nexport const VERSION = '0.1.0';\n",
    ]
)


# ---------------------------------------------------------------------------
# Repository generator
# ---------------------------------------------------------------------------


@st.composite
def _content_free_repository(  # noqa: PLR0912
    draw: st.DrawFn,
) -> tuple[RepositoryContents, str | None]:
    """Build a ``(RepositoryContents, repo_description)`` pair guaranteed
    to contain no source material from which a purpose summary could be
    derived.

    Each potential source is independently included or omitted; when
    included its value is one the summarizer treats as content-free
    (missing field, empty/whitespace value, headings/badges only,
    malformed manifest). The generator additionally sprinkles in files
    at *non-source* paths -- subdirectory READMEs, nested manifests,
    non-top-level Python files carrying docstrings -- to confirm those
    files do not slip through the design's "only root READMEs and
    recognized root manifests count" rule. The completely-empty
    ``files`` map is reachable when every booleans coin lands on
    ``False``.
    """
    files: dict[str, str] = {}

    # Optional empty / heading-only / badge-only README at the repo root.
    if draw(st.booleans()):
        files[draw(_README_BASENAMES)] = draw(_CONTENT_FREE_README_BODIES)

    # Optional README in a non-root location with prose content; this
    # MUST NOT be picked up as a purpose source.
    if draw(st.booleans()):
        non_root_readme = draw(
            st.sampled_from(
                [
                    "docs/README.md",
                    "subdir/README",
                    "nested/dir/README.txt",
                    "packages/foo/README.md",
                ]
            )
        )
        files[non_root_readme] = (
            "# Nested README\n\n"
            "This README documents a subdirectory only and must not be "
            "treated as the project's purpose summary.\n"
        )

    # Optional content-free package.json at the repo root.
    if draw(st.booleans()):
        files["package.json"] = draw(_content_free_package_json())

    # Optional content-free pyproject.toml at the repo root.
    if draw(st.booleans()):
        files["pyproject.toml"] = draw(_content_free_pyproject_toml())

    # Optional content-free pom.xml at the repo root.
    if draw(st.booleans()):
        files["pom.xml"] = draw(_content_free_pom_xml())

    # Optional Python file at the repo root with no module docstring.
    if draw(st.booleans()):
        py_name = draw(
            st.sampled_from(["main.py", "app.py", "setup.py", "conftest.py"])
        )
        files[py_name] = draw(_PYTHON_NO_DOCSTRING)

    # Optional ``src/<pkg>/__init__.py`` with no module docstring (the
    # other top-level Python location the summarizer recognizes).
    if draw(st.booleans()):
        pkg_name = draw(st.sampled_from(["mypkg", "app", "core", "lib"]))
        files[f"src/{pkg_name}/__init__.py"] = draw(_PYTHON_NO_DOCSTRING)

    # Optional JS/TS file at the repo root with no leading block comment.
    if draw(st.booleans()):
        js_name = draw(
            st.sampled_from(["index.js", "index.mjs", "index.cjs", "index.ts"])
        )
        files[js_name] = draw(_JS_NO_BLOCK_COMMENT)

    # Optional Python file at a non-recognized path that DOES carry a
    # docstring; this MUST NOT contribute. ``src/foo/bar.py`` is a
    # 3-segment src-layout path but the basename is not ``__init__.py``
    # so it does not match the summarizer's top-level rule.
    if draw(st.booleans()):
        sub_py = draw(
            st.sampled_from(
                [
                    "src/foo/bar.py",
                    "lib/utils/helper.py",
                    "tests/conftest.py",
                    "scripts/build.py",
                    "src/foo/bar/__init__.py",  # 4 segments, not top-level
                ]
            )
        )
        files[sub_py] = (
            '"""Docstring at a non-top-level location; must not be used '
            'as a purpose source."""\n'
            "x = 1\n"
        )

    # Optional manifest at a non-root path carrying a real description;
    # again MUST NOT contribute (only the root copies are inspected).
    if draw(st.booleans()):
        nested_manifest = draw(
            st.sampled_from(
                [
                    "apps/foo/package.json",
                    "subprojects/bar/pyproject.toml",
                    "modules/baz/pom.xml",
                ]
            )
        )
        if nested_manifest.endswith("package.json"):
            files[nested_manifest] = json.dumps(
                {"name": "nested", "description": "real nested description"}
            )
        elif nested_manifest.endswith("pyproject.toml"):
            files[nested_manifest] = (
                '[project]\nname = "nested"\n'
                'description = "real nested description"\n'
            )
        else:
            files[nested_manifest] = (
                '<?xml version="1.0"?>'
                "<project><description>real nested description"
                "</description></project>"
            )

    # Optional unrelated files at unrecognized paths. None of these
    # paths are inspected by the summarizer.
    if draw(st.booleans()):
        files["data.bin"] = "\x00\x01\x02\x03binary blob"
    if draw(st.booleans()):
        files[".gitignore"] = "*.pyc\n.venv/\n"
    if draw(st.booleans()):
        # ``LICENSE`` does not start with ``README`` so it is not a
        # README candidate even though it sits at the repo root.
        files["LICENSE"] = "Copyright (c) 2024\nAll rights reserved.\n"

    repo_description = draw(_NONE_OR_BLANK)

    repo = RepositoryContents(
        gitlab_project_id=draw(st.integers(min_value=1, max_value=10_000)),
        commit_sha=draw(st.sampled_from(["deadbeef", "abc123", "0" * 40, "f" * 7])),
        files=files,
    )
    return repo, repo_description


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(case=_content_free_repository())
@settings(max_examples=100)
def test_content_free_repository_yields_canonical_unknown(
    case: tuple[RepositoryContents, str | None],
) -> None:
    """Property 5: content-free repos yield ``("unknown", "insufficient source material")``."""
    repo_contents, repo_description = case

    summary, reason = summarize_purpose(repo_contents, repo_description)

    assert summary == UNKNOWN_PURPOSE_SUMMARY, (
        f"expected purpose_summary == {UNKNOWN_PURPOSE_SUMMARY!r}, "
        f"got {summary!r}; files were "
        f"{sorted(repo_contents.files.keys())!r}, "
        f"repo_description={repo_description!r}"
    )
    assert reason == INSUFFICIENT_SOURCE_MATERIAL_REASON, (
        f"expected purpose_summary_reason == "
        f"{INSUFFICIENT_SOURCE_MATERIAL_REASON!r}, got {reason!r}; files "
        f"were {sorted(repo_contents.files.keys())!r}, "
        f"repo_description={repo_description!r}"
    )

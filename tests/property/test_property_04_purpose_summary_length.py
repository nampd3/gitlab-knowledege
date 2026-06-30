# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 4: For all repositories, the purpose_summary produced by the Project_Analyzer SHALL be at most 1000 characters long.
"""Property test for the purpose-summary length bound.

**Validates Requirement 3.4** (Property 4 in the design).

For every repository -- including pathological ones with READMEs of tens
of thousands of characters, package manifests carrying ``description``
fields far in excess of the 1000-character bound, GitLab repository
descriptions of arbitrary length, and combinations of all of the above
that try to make the summarizer concatenate multiple oversized inputs --
the ``Project_Analyzer``'s purpose summarizer must produce a summary
whose length never exceeds
:data:`~project_knowledge_mcp.models.PURPOSE_SUMMARY_MAX_LEN`
(1000 characters).

The test exercises the standalone summarizer
:func:`project_knowledge_mcp.project_analyzer.purpose.summarize_purpose`
because that is the function that produces ``ProjectProfile.purpose_summary``
and is therefore the unit responsible for honoring the bound. The
aggregator wraps this same function and feeds its output into the
``ProjectProfile`` constructor (which independently rejects strings
longer than the bound), so the contract is clean to verify here without
spinning up the full pipeline.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import (
    PURPOSE_SUMMARY_MAX_LEN,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.purpose import summarize_purpose

# ---------------------------------------------------------------------------
# Character alphabets used to build the synthetic file contents.
# ---------------------------------------------------------------------------

# Printable ASCII with no whitespace controls and none of the characters
# that would break JSON / TOML basic strings or XML element bodies. Used
# to fill manifest description fields and "single-paragraph" READMEs.
_PRINTABLE_PROSE = st.characters(
    min_codepoint=0x20,
    max_codepoint=0x7e,
    blacklist_characters='"\\<>&\r\n\t',
)

# Same as above but allowing line feeds, so a generated README can carry
# multiple paragraphs and exercise the prose-extraction logic in the
# summarizer.
_PRINTABLE_PROSE_OR_LF = st.characters(
    min_codepoint=0x20,
    max_codepoint=0x7e,
    blacklist_characters='"\\<>&\r\t',
)

# Hand-picked edge-case strings chosen to stress the truncation function
# directly: hard cuts (no whitespace at all), word-boundary cuts, and
# inputs that sit exactly on either side of the 1000-character limit.
_EDGE_CASE_LONG_STRINGS: list[str] = [
    "x" * 5000,
    "ab " * 2500,
    "longwordnowhitespace" * 250,  # 5000 chars, no spaces at all
    "a" * (PURPOSE_SUMMARY_MAX_LEN - 1),
    "a" * PURPOSE_SUMMARY_MAX_LEN,
    "a" * (PURPOSE_SUMMARY_MAX_LEN + 1),
    "word " + "x" * 5000,  # whitespace only near the start
    " " * 2000 + "tail",  # mostly leading whitespace
]


def _long_text(alphabet: st.SearchStrategy[str], max_size: int) -> st.SearchStrategy[str]:
    """Strategy that mixes random text with hand-picked oversize edge cases."""
    return st.one_of(
        st.text(alphabet=alphabet, min_size=0, max_size=max_size),
        st.sampled_from(_EDGE_CASE_LONG_STRINGS),
    )


# ---------------------------------------------------------------------------
# Adversarial repository generator.
# ---------------------------------------------------------------------------


@st.composite
def _adversarial_repository(
    draw: st.DrawFn,
) -> tuple[RepositoryContents, str | None]:
    """Build a ``(RepositoryContents, repo_description)`` pair.

    Each of the four candidate sources -- README, GitLab repository
    description, ``package.json`` description, ``pyproject.toml``
    description, ``pom.xml`` description -- is included with
    independent probability so the test explores the full Cartesian
    product of "which sources are present", including the empty case
    in which the summarizer falls back to the canonical
    ``("unknown", "insufficient source material")`` result. Sizes are
    drawn from a strategy that mixes random lengths with a fixed
    catalog of pathological strings, so the truncation logic is
    exercised on inputs that:

    * have no whitespace at all (forcing a hard cut at the limit),
    * have whitespace only near the start (forcing a word-boundary cut
      back to a tiny prefix),
    * sit exactly at, just under, and just over the 1000-character
      bound,
    * are tens of thousands of characters long.
    """
    files: dict[str, str] = {}

    # README (sometimes; varied basename, varied size, possible LF).
    if draw(st.booleans()):
        readme_name = draw(
            st.sampled_from(
                ["README", "README.md", "README.rst", "README.txt", "readme.md"]
            )
        )
        files[readme_name] = draw(_long_text(_PRINTABLE_PROSE_OR_LF, 20000))

    # package.json with long description. ``json.dumps`` handles
    # whatever escaping is needed for the generated character set.
    if draw(st.booleans()):
        files["package.json"] = json.dumps(
            {
                "name": "synthetic",
                "version": "1.0.0",
                "description": draw(_long_text(_PRINTABLE_PROSE, 8000)),
            }
        )

    # pyproject.toml with long description. The generator alphabet
    # excludes ``"`` and ``\`` so the value is safe inside a TOML basic
    # string without further escaping.
    if draw(st.booleans()):
        desc = draw(_long_text(_PRINTABLE_PROSE, 8000))
        files["pyproject.toml"] = (
            '[project]\nname = "synthetic"\nversion = "0.1.0"\n'
            f'description = "{desc}"\n'
        )

    # pom.xml with long description. The generator alphabet excludes
    # ``<``, ``>``, and ``&`` so the value is XML-safe as element text.
    if draw(st.booleans()):
        desc = draw(_long_text(_PRINTABLE_PROSE, 8000))
        files["pom.xml"] = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<project><description>{desc}</description></project>"
        )

    # Repository description: ``None``, an arbitrary text, or one of
    # the oversized edge-case strings.
    repo_description = draw(
        st.one_of(
            st.none(),
            _long_text(_PRINTABLE_PROSE_OR_LF, 8000),
        )
    )

    return (
        RepositoryContents(
            gitlab_project_id=1,
            commit_sha="deadbeefcafef00d",
            files=files,
        ),
        repo_description,
    )


# ---------------------------------------------------------------------------
# The property.
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(case=_adversarial_repository())
@settings(max_examples=100)
def test_purpose_summary_length_never_exceeds_bound(
    case: tuple[RepositoryContents, str | None],
) -> None:
    """Property 4: ``len(purpose_summary) <= PURPOSE_SUMMARY_MAX_LEN``."""
    repo_contents, repo_description = case

    summary, _reason = summarize_purpose(repo_contents, repo_description)

    assert len(summary) <= PURPOSE_SUMMARY_MAX_LEN, (
        f"purpose_summary length {len(summary)} exceeds "
        f"PURPOSE_SUMMARY_MAX_LEN={PURPOSE_SUMMARY_MAX_LEN}; "
        f"summary[:80]={summary[:80]!r}"
    )

# ruff: noqa: E501
# Feature: go-analyzer-support, Property 3: No-Go repositories produce the pre-feature output.
"""Property test for the no-Go regression guarantee.

**Validates Requirements 1.4, 11.5** (Property 3 in the design, task
11.9 in ``tasks.md``).

The Go-analyzer-support feature adds Go-aware behavior at the
aggregator boundary by layering Go-specific scanners on top of the four
existing language-agnostic sub-analyzers. To avoid regressing the four
languages already supported by the parent spec (Python, JavaScript /
TypeScript, Java, and the language-agnostic YAML / manifest readers),
the aggregator consults a single repo-level guard
(:func:`~project_knowledge_mcp.project_analyzer.go.go_filter.has_go_artefacts`)
before doing any Go work. When that guard returns ``False`` — i.e. the
``RepositoryContents`` contains neither a ``.go`` file (under
:func:`~project_knowledge_mcp.project_analyzer.go.go_filter.is_go_source_file`'s
vendor-aware filter) nor a repository-root ``go.mod`` — the aggregator
short-circuits every Go branch and the produced
:class:`~project_knowledge_mcp.models.ProjectProfile` is byte-identical
to the profile produced by the pre-feature analyzer.

The "pre-feature" reference is reconstructed for each random input by
invoking the four parent-spec sub-analyzers directly:

* :func:`~project_knowledge_mcp.project_analyzer.purpose.summarize_purpose`
  for ``purpose_summary`` / ``purpose_summary_reason``.
* :func:`~project_knowledge_mcp.project_analyzer.io_extractor.extract_io`
  for ``abstract_inputs`` / ``abstract_outputs``.
* :func:`~project_knowledge_mcp.project_analyzer.external_services.detect_external_services`
  for ``external_service_dependencies``.
* :func:`~project_knowledge_mcp.project_analyzer.db_tables.detect_database_tables`
  for ``database_table_dependencies``.

The aggregator's ``_safe_*`` helpers each wrap one of these calls in a
``try / except`` block that adds a canonical section name to
``degraded_sections`` on failure but otherwise returns the legacy
detector's output unchanged when ``has_go_artefacts`` is ``False`` (see
the cross-references in
:mod:`project_knowledge_mcp.project_analyzer.__init__`). So when the
legacy detectors succeed, the legacy and aggregator outputs are
list-equal element-for-element; when the legacy detectors raise, both
the aggregator and the reference computation observe the same failure
and both return the same safe empty default. The property therefore
holds with or without legacy-detector failures.

The properties below capture every list-equality clause of Requirement
11.5 plus the broader Requirement 1.4 contract:

* ``abstract_inputs`` list-equal to :func:`extract_io`'s first return
  value (Requirement 11.5).
* ``abstract_outputs`` list-equal to :func:`extract_io`'s second return
  value (Requirement 11.5).
* ``external_service_dependencies`` list-equal to
  :func:`detect_external_services`'s return value (Requirement 11.5).
* ``database_table_dependencies`` list-equal to
  :func:`detect_database_tables`'s return value (Requirement 11.5).
* ``purpose_summary`` / ``purpose_summary_reason`` pair-equal to
  :func:`summarize_purpose`'s return tuple (Requirement 1.4).
* ``degraded_sections`` contains *no* structured Go skip strings
  (anything prefixed with ``"abstract_io: skipped "``,
  ``"external_services: skipped "``, or
  ``"database_tables: skipped "``). Requirement 11.5 says the no-Go
  output must be unchanged from the pre-feature analyzer; the
  pre-feature analyzer cannot have emitted Go-specific skip messages,
  so the no-Go aggregator must not either.

The generator deliberately mixes file kinds that the parent-spec
detectors do react to (Python source with route decorators, JSON /
TOML manifests with description fields, README files) so the assertion
is non-trivially exercised against real detection paths. Vendored Go
files — ``vendor/<...>.go`` — are also included on a fraction of the
generated examples because
:func:`is_go_source_file` is contract-bound to exclude them and they
do **not** trigger ``has_go_artefacts``; the no-Go regression must
hold even when a repository physically contains ``.go`` files under a
``vendor`` directory.
"""

from __future__ import annotations

import json
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer import analyze
from project_knowledge_mcp.project_analyzer.db_tables import detect_database_tables
from project_knowledge_mcp.project_analyzer.external_services import (
    detect_external_services,
)
from project_knowledge_mcp.project_analyzer.go.go_filter import has_go_artefacts
from project_knowledge_mcp.project_analyzer.io_extractor import extract_io
from project_knowledge_mcp.project_analyzer.purpose import summarize_purpose

pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Generator alphabets
# ---------------------------------------------------------------------------

# Printable ASCII for prose content (READMEs, descriptions). Excludes
# the characters that would break the JSON / TOML / XML manifests we
# embed the prose into so a single alphabet works for every file kind.
_PROSE_CHARS = st.characters(
    min_codepoint=0x20,
    max_codepoint=0x7e,
    blacklist_characters='"\\<>&\r\n\t',
)

# Identifier alphabet for Python / Java symbols and YAML keys. Pinning
# this to lower-case ASCII keeps the generator's output trivially
# parseable by the parent-spec detectors and prevents accidental
# collisions with reserved words.
_IDENT_CHARS = string.ascii_lowercase + "_"


# Prefixes that the Go-aware ``_safe_*`` helpers append to
# ``degraded_sections`` when their Go branch emits a
# :class:`~project_knowledge_mcp.project_analyzer.go._events.SkipFileEvent`.
# A no-Go repository must produce *none* of these — both because
# ``has_go_artefacts`` short-circuits the Go branch entirely and
# because Requirement 11.5 demands byte-identical output to the
# pre-feature analyzer (which knew nothing about Go skip strings).
_GO_SKIP_PREFIXES: tuple[str, ...] = (
    "abstract_io: skipped ",
    "external_services: skipped ",
    "database_tables: skipped ",
)


# ---------------------------------------------------------------------------
# Per-file-kind strategies
# ---------------------------------------------------------------------------


def _readme_content() -> st.SearchStrategy[str]:
    """A short README body. Lines are joined with a literal "\\n"."""

    return st.text(alphabet=_PROSE_CHARS, min_size=0, max_size=200)


def _readme_path() -> st.SearchStrategy[str]:
    """One of the well-known README basenames the parent spec reads."""

    return st.sampled_from(
        ["README.md", "README.rst", "README.txt", "README", "readme.md"],
    )


@st.composite
def _package_json(draw: st.DrawFn) -> str:
    """A ``package.json`` carrying a name, version, and description.

    ``json.dumps`` handles escaping for the generated alphabet, so the
    output is always valid JSON parsable by
    :mod:`project_knowledge_mcp.project_analyzer.purpose`.
    """

    return json.dumps(
        {
            "name": draw(st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=20)),
            "version": "1.0.0",
            "description": draw(
                st.text(alphabet=_PROSE_CHARS, min_size=0, max_size=200),
            ),
        },
    )


@st.composite
def _pyproject_toml(draw: st.DrawFn) -> str:
    """A ``pyproject.toml`` carrying a ``[project]`` table with a description.

    The generated prose alphabet excludes ``"`` and ``\\`` so the
    description body is safe inside a TOML basic string without
    further escaping.
    """

    name = draw(st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=20))
    description = draw(st.text(alphabet=_PROSE_CHARS, min_size=0, max_size=200))
    return (
        "[project]\n"
        f'name = "{name}"\n'
        'version = "0.1.0"\n'
        f'description = "{description}"\n'
    )


@st.composite
def _pom_xml(draw: st.DrawFn) -> str:
    """A minimal ``pom.xml`` with a ``<description>`` element.

    The generated prose alphabet excludes ``<``, ``>``, and ``&`` so
    the description body is XML-safe as element text.
    """

    description = draw(st.text(alphabet=_PROSE_CHARS, min_size=0, max_size=200))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<project><description>{description}</description></project>"
    )


@st.composite
def _python_source(draw: st.DrawFn) -> str:
    """A small Python source file.

    The body is empty or carries a module docstring plus a couple of
    function declarations. Some examples carry a Flask-style route
    decorator so the parent-spec I/O extractor's Python AST path is
    exercised; the route literal is always a string literal so the
    decorator parses cleanly without needing free variables to be
    resolved.
    """

    docstring = draw(st.text(alphabet=_PROSE_CHARS, min_size=0, max_size=80))
    fn_name = draw(st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=10))
    route_path = "/" + draw(st.text(alphabet=_IDENT_CHARS, min_size=0, max_size=10))
    include_route = draw(st.booleans())
    parts: list[str] = []
    if docstring:
        parts.append(f'"""{docstring}"""')
    if include_route:
        parts.append("from flask import Flask")
        parts.append("app = Flask(__name__)")
        parts.append(f'@app.route("{route_path}")')
    parts.append(f"def {fn_name}():")
    parts.append("    return None")
    return "\n".join(parts) + "\n"


@st.composite
def _javascript_source(draw: st.DrawFn) -> str:
    """A small JS / TS source file optionally calling ``fetch`` to an HTTP URL.

    Exercises the parent-spec external-service detector's URL
    extraction path.
    """

    fn_name = draw(st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=10))
    host = draw(st.text(alphabet=_IDENT_CHARS, min_size=3, max_size=12))
    include_fetch = draw(st.booleans())
    body = (
        f'  return await fetch("https://{host}.example.com/");\n'
        if include_fetch
        else "  return null;\n"
    )
    return f"async function {fn_name}() {{\n{body}}}\n"


@st.composite
def _java_source(draw: st.DrawFn) -> str:
    """A small Java source file optionally carrying a Spring annotation.

    Exercises the parent-spec I/O extractor's Java regex path.
    """

    class_name = draw(st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=8))
    route_path = "/" + draw(st.text(alphabet=_IDENT_CHARS, min_size=0, max_size=10))
    include_route = draw(st.booleans())
    annotation = (
        f'    @GetMapping("{route_path}")\n    public String handler() {{ return ""; }}\n'
        if include_route
        else ""
    )
    return f"public class {class_name} {{\n{annotation}}}\n"


@st.composite
def _yaml_source(draw: st.DrawFn) -> str:
    """A small YAML file optionally carrying a ``cron:`` entry.

    Exercises the parent-spec I/O extractor's YAML cron regex path.
    """

    include_cron = draw(st.booleans())
    if include_cron:
        return 'schedule:\n  cron: "0 * * * *"\n'
    return "metadata:\n  key: value\n"


# ---------------------------------------------------------------------------
# Repository-level strategy
# ---------------------------------------------------------------------------


@st.composite
def _no_go_repository(draw: st.DrawFn) -> tuple[RepositoryContents, str | None]:
    """Build a ``(RepositoryContents, repo_description)`` pair with no Go.

    The generated bundle contains zero files satisfying
    :func:`is_go_source_file` and no ``go.mod`` at the repository root.
    Vendored ``.go`` files (under a ``vendor/`` directory) are included
    on a fraction of examples because ``is_go_source_file`` rejects
    them — they must not trigger the Go branch, so they belong in the
    "no-Go" generator. The repository description is drawn
    independently of the file contents.
    """

    files: dict[str, str] = {}

    # README (optional). A README is one of the strongest purpose-summary
    # signals so it makes the assertion on summary equality
    # non-trivial.
    if draw(st.booleans()):
        files[draw(_readme_path())] = draw(_readme_content())

    # Root-level package manifests (optional, mutually independent).
    # Each manifest exercises a different parent-spec purpose-summary
    # candidate and a different I/O extractor manifest-parsing path
    # (``package.json``'s ``bin`` field, ``pyproject.toml``'s
    # ``[project.scripts]`` table).
    if draw(st.booleans()):
        files["package.json"] = draw(_package_json())
    if draw(st.booleans()):
        files["pyproject.toml"] = draw(_pyproject_toml())
    if draw(st.booleans()):
        files["pom.xml"] = draw(_pom_xml())

    # A handful of Python / JS / Java / YAML source files at synthesized
    # paths. The number is small (0-3 per kind) so Hypothesis can
    # explore many combinations within the 100-example budget.
    for kind, path_prefix, strategy in (
        ("py", "src/", _python_source()),
        ("js", "web/", _javascript_source()),
        ("java", "app/src/main/java/", _java_source()),
        ("yaml", ".github/workflows/", _yaml_source()),
    ):
        count = draw(st.integers(min_value=0, max_value=2))
        for idx in range(count):
            basename = draw(
                st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=10),
            )
            suffix = {
                "py": ".py",
                "js": ".js",
                "java": ".java",
                "yaml": ".yml",
            }[kind]
            files[f"{path_prefix}{basename}_{idx}{suffix}"] = draw(strategy)

    # Optionally include vendored ``.go`` files. These are excluded by
    # ``is_go_source_file`` and therefore by ``has_go_artefacts``, so
    # they belong in the no-Go generator. Each carries a tiny
    # well-formed Go fragment so even a regression that accidentally
    # parsed them would surface as a recognizable detection, not as a
    # parser error.
    if draw(st.booleans()):
        vendor_count = draw(st.integers(min_value=1, max_value=2))
        for idx in range(vendor_count):
            basename = draw(
                st.text(alphabet=_IDENT_CHARS, min_size=1, max_size=10),
            )
            files[f"vendor/github.com/lib/{basename}_{idx}.go"] = (
                'package lib\nimport "fmt"\nfunc Helper() { fmt.Println("x") }\n'
            )

    # GitLab repository description — independent of file contents.
    repo_description = draw(
        st.one_of(
            st.none(),
            st.text(alphabet=_PROSE_CHARS, min_size=0, max_size=200),
        ),
    )

    repo = RepositoryContents(
        gitlab_project_id=draw(st.integers(min_value=1, max_value=1_000_000)),
        commit_sha=draw(
            st.text(
                alphabet=string.hexdigits.lower(),
                min_size=40,
                max_size=40,
            ),
        ),
        files=files,
    )
    return repo, repo_description


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(case=_no_go_repository())
@settings(max_examples=100)
def test_no_go_repositories_produce_pre_feature_output(
    case: tuple[RepositoryContents, str | None],
) -> None:
    """Property 3: no-Go repositories reproduce the pre-feature output.

    For every generated ``RepositoryContents`` that contains no Go
    source file and no ``go.mod`` at the repository root, the four
    profile lists produced by :func:`analyze` MUST equal the lists
    produced by invoking the four parent-spec sub-analyzers directly
    (Requirement 11.5), the produced purpose-summary pair MUST equal
    the parent-spec summarizer's output (Requirement 1.4), and the
    ``degraded_sections`` field MUST NOT carry any of the structured
    Go skip strings the Go-aware ``_safe_*`` helpers emit.
    """

    repository_contents, repo_description = case

    # Sanity precondition: the generator promises no Go artefacts. If
    # this assertion ever fires, the generator's filtering logic
    # regressed, not the analyzer.
    assert not has_go_artefacts(repository_contents), (
        "test generator produced a Go-bearing repository; the no-Go "
        "regression property exercises the has_go_artefacts == False "
        "branch only"
    )

    # The aggregator-driven profile. ``analyze`` never raises by
    # contract, so a single call covers every legacy detector path.
    profile = analyze(
        project_id=repository_contents.gitlab_project_id,
        full_path="group/project",
        analysis_branch="main",
        commit_sha=repository_contents.commit_sha,
        repo_description=repo_description,
        repository_contents=repository_contents,
    )

    # Recompute the pre-feature reference by invoking each legacy
    # sub-analyzer directly. Each call is wrapped in a try/except that
    # mirrors the aggregator's ``_safe_*`` resilience contract: when a
    # legacy detector raises, the aggregator records a canonical
    # section name and substitutes the safe empty default, so the
    # reference computation does the same. This keeps the property
    # well-defined even on inputs that happen to trip a legacy
    # detector bug.
    try:
        ref_summary, ref_reason = summarize_purpose(
            repository_contents, repo_description,
        )
    except Exception:  # pragma: no cover - defensive, mirrors _safe_purpose
        from project_knowledge_mcp.models import (
            INSUFFICIENT_SOURCE_MATERIAL_REASON,
            UNKNOWN_PURPOSE_SUMMARY,
        )
        ref_summary, ref_reason = (
            UNKNOWN_PURPOSE_SUMMARY,
            INSUFFICIENT_SOURCE_MATERIAL_REASON,
        )

    try:
        ref_inputs, ref_outputs = extract_io(repository_contents)
    except Exception:  # pragma: no cover - defensive, mirrors _safe_io
        ref_inputs, ref_outputs = [], []

    try:
        ref_services = detect_external_services(repository_contents)
    except Exception:  # pragma: no cover - defensive, mirrors _safe_external_services
        ref_services = []

    try:
        ref_tables = detect_database_tables(repository_contents)
    except Exception:  # pragma: no cover - defensive, mirrors _safe_database_tables
        ref_tables = []

    # --- Requirement 11.5: the four lists are equal under list equality ---
    assert profile.abstract_inputs == ref_inputs, (
        "abstract_inputs diverged from extract_io() output on a no-Go "
        f"repository; aggregator={profile.abstract_inputs!r}, "
        f"reference={ref_inputs!r}"
    )
    assert profile.abstract_outputs == ref_outputs, (
        "abstract_outputs diverged from extract_io() output on a no-Go "
        f"repository; aggregator={profile.abstract_outputs!r}, "
        f"reference={ref_outputs!r}"
    )
    assert profile.external_service_dependencies == ref_services, (
        "external_service_dependencies diverged from "
        "detect_external_services() output on a no-Go repository; "
        f"aggregator={profile.external_service_dependencies!r}, "
        f"reference={ref_services!r}"
    )
    assert profile.database_table_dependencies == ref_tables, (
        "database_table_dependencies diverged from "
        "detect_database_tables() output on a no-Go repository; "
        f"aggregator={profile.database_table_dependencies!r}, "
        f"reference={ref_tables!r}"
    )

    # --- Requirement 1.4: the purpose-summary pair is equal ---
    assert (profile.purpose_summary, profile.purpose_summary_reason) == (
        ref_summary,
        ref_reason,
    ), (
        "purpose_summary or purpose_summary_reason diverged from "
        "summarize_purpose() output on a no-Go repository; "
        f"aggregator=({profile.purpose_summary!r}, "
        f"{profile.purpose_summary_reason!r}), "
        f"reference=({ref_summary!r}, {ref_reason!r})"
    )

    # --- No Go skip strings can appear on a no-Go repository ---
    # The pre-feature analyzer knew nothing about Go skip strings, so
    # the no-Go output must not carry any of them. The aggregator's
    # ``has_go_artefacts`` short-circuit guarantees this by construction;
    # this assertion pins the guarantee against a future regression.
    for entry in profile.degraded_sections:
        for prefix in _GO_SKIP_PREFIXES:
            assert not entry.startswith(prefix), (
                f"degraded_sections carries a Go skip string {entry!r} "
                "on a no-Go repository; has_go_artefacts() should have "
                "short-circuited every Go branch"
            )

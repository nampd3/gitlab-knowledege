# ruff: noqa: E501
# Feature: go-analyzer-support, Property 13: Build constraints, cgo files, and files with tokenization errors are skipped at the parser boundary, recorded as a single SkipFileEvent, and do not affect the rest of the repository.
"""Property test for the build-constraint / cgo / tokenization-error skip path.

**Validates Requirements 10.4, 11.1, 11.2** (Property 13 in the design,
task 4.5 in ``tasks.md``).

For every randomly generated synthetic Go repository whose files mix
"well-formed" and "bad" sources, a single
:func:`project_knowledge_mcp.project_analyzer.go.go_parser.parse_repo`
call SHALL satisfy the following properties for *each* file:

* A file with a non-trivial ``//go:build <expr>`` (or legacy
  ``// +build <expr>``) constraint produces exactly one
  :class:`SkipFileEvent` whose ``reason`` is the canonical
  ``"build constraint requires toolchain"`` string from
  Requirement 10.4.
* A file containing an unaliased ``import "C"`` declaration produces
  exactly one :class:`SkipFileEvent` whose ``reason`` is the canonical
  ``"cgo directive requires toolchain"`` string from Requirement 10.4.
* A file with a tokenization error (unterminated string literal,
  unterminated raw string literal, unterminated block comment, or
  invalid escape sequence) produces exactly one :class:`SkipFileEvent`
  whose ``reason`` starts with ``"tokenization failed: "`` (per
  Requirement 11.1, which mandates that a single malformed file is
  contained at the parser boundary).
* A "skipped" file produces *no other* events besides its single
  :class:`SkipFileEvent` — no :class:`ImportEvent`, no
  :class:`FuncDeclEvent`, no :class:`MethodCallEvent`, no
  :class:`StructLitEvent`, no :class:`PackageDocCommentEvent`. The
  recognizer must short-circuit before any per-construct walking
  happens (Requirement 10.4).
* A well-formed file is unaffected by the presence of bad neighbours:
  it produces its normal events, with no :class:`SkipFileEvent` in its
  list and ``file_path`` set to its own path on every event
  (Requirement 11.1's "the remaining files SHALL still be analyzed").
* Every emitted event's ``file_path`` matches the key under which it
  appears in the result mapping — there is no cross-file leakage of
  events.

The test calls ``parse_repo`` directly (not ``analyze``) so it pins
down the exact failure-containment behaviour of the parser layer that
the four sub-analyzers depend on. Property 11.2 is exercised
transitively: a file that produces a :class:`SkipFileEvent` here is
exactly the input each ``_safe_*`` helper turns into a structured
``"<section>: skipped <path> (<reason>)"`` entry in
``degraded_sections``.
"""

from __future__ import annotations

from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer.go._events import (
    FuncDeclEvent,
    ImportEvent,
    MethodCallEvent,
    PackageDocCommentEvent,
    SkipFileEvent,
    StructLitEvent,
)
from project_knowledge_mcp.project_analyzer.go.go_parser import parse_repo

pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Per-kind file fragment templates
# ---------------------------------------------------------------------------
#
# The strategies below intentionally produce *small* Go source fragments
# rather than full programs. The recognizer's skip path is triggered by
# the presence (or absence) of a leading-comment build constraint, an
# ``import "C"`` declaration, or a tokenizer-fatal lexical shape; none
# of these depend on having a complete program. Keeping the fragments
# tiny also keeps Hypothesis' search space tractable so the
# ``max_examples=100`` budget covers the four kinds many times over.

#: Well-formed Go source fragments, each guaranteed to produce at least
#: one non-:class:`SkipFileEvent` event when fed through ``parse_repo``.
#: The mix exercises the three event kinds the recognizer can emit
#: from a fragment of this size: :class:`ImportEvent`,
#: :class:`FuncDeclEvent`, and :class:`PackageDocCommentEvent`.
_WELL_FORMED_FRAGMENTS: Final[tuple[str, ...]] = (
    'package foo\nimport "fmt"\n',
    'package foo\nimport "net/http"\n',
    "package main\nfunc main() {}\n",
    'package foo\nimport "fmt"\n\nfunc Greet() {}\n',
    "// pkg foo does foo things.\npackage foo\n",
    'package foo\nimport (\n\t"fmt"\n\t"strings"\n)\n',
)


#: Non-trivial build-constraint expressions that the recognizer must
#: treat as toolchain-only (Requirement 10.4 carve-out: the empty
#: expression and ``!ignore`` are trivial and explicitly excluded). The
#: set covers single tags, conjunctions, disjunctions, version tags,
#: and negations so a regression that handles only one shape surfaces.
_BUILD_CONSTRAINT_EXPRS: Final[tuple[str, ...]] = (
    "linux",
    "linux && amd64",
    "darwin || windows",
    "go1.21",
    "!cgo",
    "linux && (amd64 || arm64)",
)


#: Both go-build syntaxes the recognizer understands. The legacy
#: ``// +build`` form has the same toolchain-only semantics as the
#: modern ``//go:build`` form (Requirement 10.4 names them together).
_BUILD_CONSTRAINT_PREFIXES: Final[tuple[str, ...]] = (
    "//go:build ",
    "// +build ",
)


#: Source fragments that the tokenizer rejects with one of the four
#: canonical reasons enumerated on
#: :class:`project_knowledge_mcp.project_analyzer.go.go_tokenizer.GoTokenizationError`
#: (``REASON_UNTERMINATED_STRING``, ``REASON_UNTERMINATED_RAW_STRING``,
#: ``REASON_UNTERMINATED_BLOCK_COMMENT``, ``REASON_INVALID_ESCAPE``).
#: ``parse_repo`` is contract-bound to catch each one and convert it
#: into a single :class:`SkipFileEvent` with reason
#: ``"tokenization failed: <detail>"`` (Requirement 11.1).
_TOKENIZATION_ERROR_FRAGMENTS: Final[tuple[str, ...]] = (
    # Unterminated regular string literal: closing quote missing,
    # then EOF.
    'package foo\nvar x = "never closed',
    # Unterminated regular string literal: literal runs into a
    # newline, which Go forbids.
    'package foo\nvar x = "broken by newline\n',
    # Unterminated raw string literal: opening backtick, then EOF.
    "package foo\nvar x = `raw never closed",
    # Unterminated block comment: ``/*`` then EOF.
    "package foo\n/* block never closed",
    # Invalid escape sequence: ``\q`` is not a recognized Go escape.
    'package foo\nvar x = "bad\\q escape"\n',
)


# ---------------------------------------------------------------------------
# Per-file spec strategy
# ---------------------------------------------------------------------------


# Canonical reason strings the recognizer attaches to a SkipFileEvent.
# Pinning them as module-level constants makes the assertions below
# unambiguous and makes a regression in the reason wording surface as
# a single literal mismatch instead of a structural change.
_REASON_BUILD_CONSTRAINT: Final[str] = "build constraint requires toolchain"
_REASON_CGO: Final[str] = "cgo directive requires toolchain"
_TOKENIZATION_REASON_PREFIX: Final[str] = "tokenization failed: "


@st.composite
def _file_spec(draw: st.DrawFn) -> tuple[str, str, str | None]:
    """Generate ``(kind, source_text, expected_skip_reason)`` for one file.

    ``kind`` is one of ``"well_formed"``, ``"build_constraint"``,
    ``"cgo"``, ``"tokenization_error"``. ``expected_skip_reason`` is
    ``None`` for well-formed files, the canonical reason string for
    cgo and build-constraint files, and the prefix
    ``"tokenization failed: "`` for tokenization-error files (the
    suffix is the tokenizer's specific reason, which the test treats
    as a free variable so a tokenizer change does not break the test).
    """

    kind = draw(
        st.sampled_from(
            ("well_formed", "build_constraint", "cgo", "tokenization_error"),
        ),
    )

    if kind == "well_formed":
        return ("well_formed", draw(st.sampled_from(_WELL_FORMED_FRAGMENTS)), None)

    if kind == "build_constraint":
        prefix = draw(st.sampled_from(_BUILD_CONSTRAINT_PREFIXES))
        expr = draw(st.sampled_from(_BUILD_CONSTRAINT_EXPRS))
        # Blank line between the constraint and the package keyword
        # is required by the Go spec; the recognizer's leading-comment
        # scan honours it.
        source = f"{prefix}{expr}\n\npackage foo\n"
        return ("build_constraint", source, _REASON_BUILD_CONSTRAINT)

    if kind == "cgo":
        # An unaliased ``import "C"`` declaration; the recognizer must
        # detect it and short-circuit. The minimal form below is the
        # one the four sample repositories use and the only form
        # Requirement 10.4 names.
        source = 'package foo\nimport "C"\n'
        return ("cgo", source, _REASON_CGO)

    # kind == "tokenization_error"
    source = draw(st.sampled_from(_TOKENIZATION_ERROR_FRAGMENTS))
    return ("tokenization_error", source, _TOKENIZATION_REASON_PREFIX)


@st.composite
def _go_repo(
    draw: st.DrawFn,
) -> tuple[dict[str, str], list[tuple[str, str, str | None]]]:
    """Generate a small synthetic Go repository.

    Each generated file is placed at a unique non-vendor path so
    ``is_go_source_file`` admits it. The mix is drawn from
    :func:`_file_spec`, so a single example may contain any
    combination of the four kinds (including all-bad and all-good
    extremes).

    Returns:
        A ``(files, specs)`` pair. ``files`` is the
        ``RepositoryContents.files`` mapping; ``specs`` is the list
        of ``(path, kind, expected_skip_reason)`` triples in the same
        order as the file specs were drawn.
    """

    file_specs = draw(st.lists(_file_spec(), min_size=1, max_size=6))
    files: dict[str, str] = {}
    specs: list[tuple[str, str, str | None]] = []
    for idx, (kind, source, reason) in enumerate(file_specs):
        # Stable, distinct, non-vendor path with the ``.go`` suffix
        # ``is_go_source_file`` requires. The path is the only
        # cross-file shared resource; everything else about a file is
        # generated independently.
        path = f"f{idx}.go"
        files[path] = source
        specs.append((path, kind, reason))
    return files, specs


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(repo=_go_repo())
@settings(max_examples=100)
def test_parse_repo_isolates_skip_files_and_preserves_well_formed_files(
    repo: tuple[dict[str, str], list[tuple[str, str, str | None]]],
) -> None:
    """Property 13: parse_repo skips bad files in isolation.

    For every file in the generated repository, ``parse_repo`` MUST
    produce a result that satisfies the per-kind contract documented
    in Requirements 10.4 and 11.1. The well-formed files MUST NOT be
    affected by the bad neighbours.
    """

    files, specs = repo

    rc = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="0" * 40,
        files=files,
    )

    result = parse_repo(rc)

    # --- Mapping shape: every input path appears as a key ---
    # Property 1 of parse_repo (Requirement 1.3): files admitted by
    # ``is_go_source_file`` (every path generated here is) appear as a
    # key in the result. No spurious keys.
    expected_paths = {path for path, _, _ in specs}
    assert set(result.keys()) == expected_paths, (
        f"parse_repo result keys {sorted(result.keys())!r} do not match "
        f"the input file paths {sorted(expected_paths)!r}"
    )

    # --- Per-file event-shape contract ---
    for path, kind, expected_reason in specs:
        events = result[path]

        # Universal invariant: every event's ``file_path`` matches
        # the key under which it appears. This rules out cross-file
        # leakage at the parser layer (Requirement 11.1's "remaining
        # files SHALL still be analyzed" depends on per-file
        # isolation).
        for event in events:
            assert event.file_path == path, (
                f"event {event!r} appears under key {path!r} but "
                f"carries file_path {event.file_path!r}; parse_repo "
                f"must not leak events across files"
            )

        if kind == "well_formed":
            # Well-formed files MUST NOT produce a SkipFileEvent;
            # the bad neighbours' presence cannot poison them
            # (Requirement 11.1).
            skip_events = [e for e in events if isinstance(e, SkipFileEvent)]
            assert skip_events == [], (
                f"well-formed file {path!r} unexpectedly produced "
                f"{len(skip_events)} SkipFileEvent(s): {skip_events!r}; "
                f"a well-formed neighbour must be unaffected by skipped "
                f"files in the same repo"
            )
            # Sanity: the chosen well-formed templates each produce
            # at least one normal event, so the "well-formed produces
            # normal events" half of the property is observable.
            assert len(events) >= 1, (
                f"well-formed file {path!r} produced no events; the "
                f"chosen template should yield at least one ImportEvent, "
                f"FuncDeclEvent, or PackageDocCommentEvent"
            )
            # And the events that it produces must be of the recognized
            # construct types — never anything else, never a stray
            # SkipFileEvent.
            allowed_types = (
                ImportEvent,
                FuncDeclEvent,
                MethodCallEvent,
                StructLitEvent,
                PackageDocCommentEvent,
            )
            for event in events:
                assert isinstance(event, allowed_types), (
                    f"well-formed file {path!r} produced an unexpected "
                    f"event type {type(event).__name__}: {event!r}"
                )
            continue

        # --- "Bad" file kinds: exactly one SkipFileEvent, nothing else ---
        # Requirement 10.4 mandates that a build-constraint or cgo
        # file is skipped wholesale; Requirement 11.1 mandates that
        # a tokenization error becomes a single SkipFileEvent. In
        # every case the recognizer must short-circuit before any
        # per-construct event is emitted.
        assert len(events) == 1, (
            f"{kind!r} file {path!r} produced {len(events)} events; "
            f"expected exactly one SkipFileEvent (the file's only "
            f"event per design §3 'go.go_parser')"
        )
        skip = events[0]
        assert isinstance(skip, SkipFileEvent), (
            f"{kind!r} file {path!r} produced an event of type "
            f"{type(skip).__name__}; expected SkipFileEvent"
        )

        # The reason string is fixed per Requirement 10.4 for the
        # build-constraint and cgo cases. The tokenization-error
        # case carries a tokenizer-specific suffix; the test pins
        # the prefix only so a tokenizer-internal wording change
        # does not require updating this property.
        if kind == "tokenization_error":
            assert skip.reason.startswith(_TOKENIZATION_REASON_PREFIX), (
                f"tokenization_error file {path!r} produced "
                f"SkipFileEvent.reason={skip.reason!r}; expected a "
                f"reason starting with {_TOKENIZATION_REASON_PREFIX!r} "
                f"(Requirement 11.1)"
            )
            # The detail after the prefix must be non-empty so a
            # downstream consumer can distinguish failure modes.
            detail = skip.reason[len(_TOKENIZATION_REASON_PREFIX):]
            assert detail.strip() != "", (
                f"tokenization_error file {path!r} produced an empty "
                f"detail after {_TOKENIZATION_REASON_PREFIX!r}; the "
                f"tokenizer's canonical reason string must follow"
            )
        else:
            assert skip.reason == expected_reason, (
                f"{kind!r} file {path!r} produced "
                f"SkipFileEvent.reason={skip.reason!r}; expected "
                f"the canonical {expected_reason!r} (Requirement 10.4)"
            )

        # The line number must be a positive 1-indexed source line
        # (the recognizer pins the offending construct's start line).
        assert skip.line >= 1, (
            f"{kind!r} file {path!r} produced SkipFileEvent.line="
            f"{skip.line}; expected a 1-indexed line number"
        )

    # --- Cross-file isolation: well-formed events are identical to
    # what they would be in a single-file repo ---
    # This pins down "do not affect the rest of the repo" in the
    # strongest possible form: parse_repo on the multi-file repo
    # produces exactly the same event lists for the well-formed
    # files as parse_repo on a repo containing only that file.
    # Without this check, a regression that perturbed (e.g.) line
    # numbering or event order in well-formed files based on
    # neighbour content would slip through.
    for path, kind, _ in specs:
        if kind != "well_formed":
            continue
        single_rc = RepositoryContents(
            gitlab_project_id=1,
            commit_sha="0" * 40,
            files={path: files[path]},
        )
        single_events = parse_repo(single_rc)[path]
        assert result[path] == single_events, (
            f"well-formed file {path!r} produced different events when "
            f"surrounded by skipped files than when analyzed alone:\n"
            f"  multi-file: {result[path]!r}\n"
            f"  alone:      {single_events!r}\n"
            f"the parser must not let bad neighbours influence "
            f"well-formed files"
        )

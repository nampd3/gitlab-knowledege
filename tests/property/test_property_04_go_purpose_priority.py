# ruff: noqa: E501
# Feature: go-analyzer-support, Property 4: Purpose summary follows the documented priority order.
"""Property test for the Go purpose-summary candidate helper.

**Validates Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7** (Property 4
in the design).

The design's documented priority order interleaves seven candidate
slots — README, GitLab description, ``gomod_comment``,
``gomod_module_path``, root-level package manifest description, top-level
Python or JavaScript module docstring, ``package_doc_comment`` — at the
aggregator boundary (task 11.5). This test focuses on the Go-layer
contribution that feeds slots 3, 4, and 7:
:func:`collect_go_candidates`. It asserts the structural guarantees of
each Go candidate slot:

1. ``gomod_comment`` is the body of the leading ``//`` comment when
   present (Requirement 2.2), falling back to the trailing
   same-line ``//`` comment, and ``None`` when neither is present.
2. ``gomod_module_path`` strips a leading ``<host>/<org>/`` prefix from
   module paths with three or more slash-separated segments and passes
   bare module names (one or two segments) through unchanged
   (Requirement 2.3).
3. ``package_doc_comment`` is sourced from the first non-empty
   :class:`PackageDocCommentEvent` over **eligible** paths in
   sorted-path order (Requirement 2.4); files at non-eligible paths
   (``internal/...``, deeper than ``cmd/<name>/main.go``) never
   contribute.
4. The helper does not consult any :class:`MethodCallEvent`, so viper
   ``SetConfigName`` / ``AddConfigPath`` strings (Requirement 2.7) are
   inert by construction. The property is verified by adding and
   removing arbitrary ``MethodCallEvent`` instances to the events
   stream and asserting the candidate output is unchanged.

Length capping and whitespace normalization (Requirement 2.5) are
applied through the parent-spec ``_normalize_description`` and
``_truncate`` helpers — the same path the non-Go candidates take —
so this test computes the expected output by mirroring those helpers
in the assertion path.

The full priority-order interleaving across non-Go and Go candidates
(Requirement 2.6) lives at the aggregator (``_safe_purpose``) and is
covered by the aggregator-level property at task 11.5; the
sub-analyzer-level test exercised here proves the Go-layer slot
contents are correct so the aggregator's choice between them is built
on a sound foundation.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer.go._events import (
    MethodCallEvent,
    PackageDocCommentEvent,
    StringLitArg,
)
from project_knowledge_mcp.project_analyzer.go.go_purpose import collect_go_candidates
from project_knowledge_mcp.project_analyzer.purpose import (
    _normalize_description,
    _truncate,
)

if TYPE_CHECKING:
    from project_knowledge_mcp.project_analyzer.go._events import GoEvent


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Strategy alphabets
# ---------------------------------------------------------------------------

# Comment-body alphabet: lowercase letters, digits, and a few punctuation
# characters that survive ``_normalize_description``. ``\n``, ``\r``,
# ``\t``, and ``/`` are excluded so a generated body cannot prematurely
# terminate a ``go.mod`` line or split into a second ``//`` comment.
# Uppercase letters are excluded so a generated body can never collide
# with the uppercase viper sentinel strings used in the
# Requirement 2.7 property below.
_BODY_CHARS = string.ascii_lowercase + string.digits + " .,-:"

# Module-path segment alphabet. Go's module-path grammar admits letters,
# digits, dot, hyphen, and underscore inside a single segment; we draw
# from that set so the generated ``module <path>`` line round-trips
# through the line-oriented ``parse_go_mod`` regex unchanged.
_SEGMENT_CHARS = string.ascii_letters + string.digits + ".-_"

# Identifier-like alphabet for synthetic file basenames (root-level
# ``*.go`` files and ``cmd/<name>/main.go`` directory segments).
# Exact path eligibility is the property under test, so the generator
# never produces strings that contain a slash.
_NAME_CHARS = string.ascii_lowercase + string.digits + "_-"


# Number of segments at which Requirement 2.3's ``<host>/<org>/`` prefix
# stripping kicks in. Mirrors the implementation constant in
# :mod:`project_knowledge_mcp.project_analyzer.go.go_purpose` rather
# than importing it so the test would catch a silent change to the
# implementation's strip threshold.
_PREFIX_STRIPPING_SEGMENT_THRESHOLD = 3

# Sentinel strings injected into every generated ``MethodCallEvent``
# argument list. Both contain uppercase letters and an underscore,
# neither of which appear in ``_BODY_CHARS`` or ``_SEGMENT_CHARS``, so
# they cannot be produced by any other strategy in this file. The
# Requirement 2.7 test verifies these strings never reach the helper's
# output by comparing the output with vs. without ``MethodCallEvent``s
# present, which is a strictly stronger property than substring
# containment.
_VIPER_SENTINEL_NAME = "VIPER_SENTINEL_NAME"
_VIPER_SENTINEL_PATH = "/etc/VIPER_SENTINEL_PATH"


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


_segment = st.text(alphabet=_SEGMENT_CHARS, min_size=1, max_size=16)


@st.composite
def _module_path(draw: st.DrawFn) -> str:
    """Generate a module path with 1, 2, 3, or 4 slash-separated segments.

    The mix exercises Requirement 2.3's three observable cases:

    * 1 segment — the bare-name shape used by all four sample
      repositories (``cat-service``, ``fec_pool_service``, etc.); the
      helper passes these through unchanged.
    * 2 segments — between bare and the canonical 3-segment form;
      passes through unchanged because the threshold is ``>= 3``.
    * 3+ segments — the canonical ``<host>/<org>/<rest>`` shape; the
      helper strips the first two segments.
    """
    n = draw(st.integers(min_value=1, max_value=4))
    return "/".join(draw(_segment) for _ in range(n))


@st.composite
def _nonblank_body(draw: st.DrawFn) -> str:
    """A comment body whose ``strip`` is non-empty.

    A body whose ``strip`` is empty would produce ``leading_comment``
    or ``trailing_comment`` of ``None`` from ``parse_go_mod``, which
    collapses the test cases for "comment present but blank" and
    "comment absent" — by enforcing non-empty bodies we keep the four
    presence combinations distinct and exercise them all.
    """
    body = draw(st.text(alphabet=_BODY_CHARS, min_size=1, max_size=80))
    if not body.strip():
        # The character set permits all-space draws; substitute a
        # canonical non-blank value rather than rejecting (which would
        # waste shrinking budget).
        body = "x"
    return body


_optional_body = st.one_of(st.none(), _nonblank_body())


@st.composite
def _gomod_case(draw: st.DrawFn) -> tuple[str, str | None, str | None, str]:
    """Generate ``(module_path, leading, trailing, gomod_text)``.

    Both comment slots are independently present-or-absent so the
    Cartesian product of (no comment, leading only, trailing only,
    both) is explored. The synthesized ``go.mod`` text is exactly the
    shape ``parse_go_mod`` recognizes: an optional ``// <leading>``
    line, immediately followed by the ``module <path>`` line with an
    optional same-line ``// <trailing>`` comment.
    """
    module_path = draw(_module_path())
    leading = draw(_optional_body)
    trailing = draw(_optional_body)

    parts: list[str] = []
    if leading is not None:
        parts.append(f"// {leading}")
    module_line = f"module {module_path}"
    if trailing is not None:
        module_line = f"{module_line} // {trailing}"
    parts.append(module_line)
    text = "\n".join(parts) + "\n"
    return module_path, leading, trailing, text


# ---------- Path generators for PackageDocCommentEvent placement ----------

# Root-level Go file: any name without a slash, with the ``.go`` suffix.
_root_go_path = st.text(
    alphabet=_NAME_CHARS,
    min_size=1,
    max_size=8,
).map(lambda name: f"{name}.go")

# ``cmd/<name>/main.go``: one directory deep under ``cmd``. Names are
# constrained to characters that cannot contain a slash, so the
# generated path always has exactly three segments.
_cmd_named_path = st.text(
    alphabet=_NAME_CHARS,
    min_size=1,
    max_size=8,
).map(lambda name: f"cmd/{name}/main.go")

# A sampled set of paths that the helper MUST treat as ineligible.
# Includes:
#   * ``internal/`` paths at depth 1 and 2 — the canonical "non-cmd
#     subtree" shape
#   * ``cmd/foo/bar/main.go`` — depth 2 under ``cmd``, which the
#     helper rejects because the regex requires exactly one segment
#     between ``cmd/`` and ``/main.go``
#   * ``cmd/foo/baz.go`` — under ``cmd`` but not named ``main.go``
#   * arbitrary nested paths
_non_eligible_path = st.sampled_from(
    [
        "internal/foo.go",
        "internal/svc/handler.go",
        "cmd/foo/bar/main.go",
        "cmd/foo/baz.go",
        "pkg/util/util.go",
        "deeper/nested/file.go",
    ]
)

# An eligible path: root-level ``*.go``, exactly ``cmd/main.go``, or
# ``cmd/<name>/main.go``. ``st.just`` is wrapped in ``st.one_of`` so
# the constant ``cmd/main.go`` case carries equal weight with the
# parameterized shapes.
_eligible_path = st.one_of(
    st.just("cmd/main.go"),
    _root_go_path,
    _cmd_named_path,
)


@st.composite
def _file_event_pair(
    draw: st.DrawFn,
) -> tuple[str, list["GoEvent"]]:
    """Generate a ``(path, events)`` pair for a single file.

    Every generated file's event stream carries two synthetic viper
    ``MethodCallEvent`` instances whose argument lists carry the
    sentinel strings. The Requirement 2.7 property below verifies
    those events have no effect on the helper's output.
    """
    path = draw(st.one_of(_eligible_path, _non_eligible_path))
    has_doc = draw(st.booleans())
    doc_text = draw(_nonblank_body()) if has_doc else None

    events: list[GoEvent] = [
        MethodCallEvent(
            receiver_chain=("viper",),
            method_name="SetConfigName",
            args=(StringLitArg(value=_VIPER_SENTINEL_NAME),),
            file_path=path,
            line=1,
        ),
        MethodCallEvent(
            receiver_chain=("viper",),
            method_name="AddConfigPath",
            args=(StringLitArg(value=_VIPER_SENTINEL_PATH),),
            file_path=path,
            line=2,
        ),
    ]
    if doc_text is not None:
        events.append(
            PackageDocCommentEvent(
                text=doc_text,
                file_path=path,
                line=3,
            )
        )
    return path, events


@st.composite
def _events_by_file(draw: st.DrawFn) -> dict[str, list["GoEvent"]]:
    """Generate a ``Mapping[str, list[GoEvent]]`` for the helper.

    Up to six unique paths so the tests exercise both the
    "no eligible files" case (when all paths happen to be ineligible
    or no doc-comment events are included) and the "multiple eligible
    files compete on sorted-path order" case.
    """
    pairs = draw(
        st.lists(
            _file_event_pair(),
            min_size=0,
            max_size=6,
            unique_by=lambda pair: pair[0],
        )
    )
    return dict(pairs)


# ---------------------------------------------------------------------------
# Pure-Python expected-output computations
# ---------------------------------------------------------------------------


def _expected_normalized(value: str | None) -> str | None:
    """Mirror the helper's per-candidate normalize-and-truncate pipeline.

    Returns ``None`` for values that ``_normalize_description`` reduces
    to ``None`` (``None`` input or all-whitespace input). Otherwise
    runs the same ``_normalize_description`` followed by ``_truncate``
    pipeline the helper applies internally, so the test computes its
    expected value through the same code path the implementation uses.
    """
    if value is None:
        return None
    normalized = _normalize_description(value)
    if normalized is None:
        return None
    return _truncate(normalized)


def _expected_module_path(module_path: str) -> str | None:
    """Compute the expected ``gomod_module_path`` candidate.

    Mirrors the implementation: 3+ segments → drop the first two and
    join the rest; 1 or 2 segments → pass through unchanged. The
    result is then run through ``_expected_normalized`` so the test's
    expected value passes through the same normalize-and-truncate
    rules the helper applies.
    """
    parts = module_path.split("/")
    if len(parts) >= _PREFIX_STRIPPING_SEGMENT_THRESHOLD:
        candidate = "/".join(parts[2:])
    else:
        candidate = module_path
    return _expected_normalized(candidate)


def _is_eligible(path: str) -> bool:
    """Return ``True`` for the three eligible main-path shapes.

    Mirrors :func:`go_purpose._is_eligible_main_path`:

    * a root-level ``.go`` file (no slash anywhere in the path),
    * the literal ``cmd/main.go``,
    * any ``cmd/<name>/main.go`` with exactly one directory under
      ``cmd``.
    """
    if path.endswith(".go") and "/" not in path:
        return True
    if path == "cmd/main.go":
        return True
    parts = path.split("/")
    return len(parts) == _PREFIX_STRIPPING_SEGMENT_THRESHOLD and parts[0] == "cmd" and parts[2] == "main.go"


def _expected_doc_comment(
    events_by_file: dict[str, list["GoEvent"]],
) -> str | None:
    """Compute the expected ``package_doc_comment`` candidate.

    Walks paths in sorted order (matching the helper's iteration),
    inspects only :class:`PackageDocCommentEvent` instances, and
    returns the first one whose body is non-empty after normalization.
    Non-eligible paths are skipped so multi-file repositories whose
    eligible files all lack doc comments fall through to ``None``.
    """
    for path in sorted(events_by_file):
        if not _is_eligible(path):
            continue
        for ev in events_by_file[path]:
            if isinstance(ev, PackageDocCommentEvent):
                cand = _expected_normalized(ev.text)
                if cand is not None:
                    return cand
    return None


def _build_repo(gomod_text: str | None) -> RepositoryContents:
    """Build a minimal :class:`RepositoryContents` carrying a single ``go.mod``.

    The helper reads only ``go.mod`` from ``RepositoryContents`` (all
    other reads come from ``events_by_file``), so this fixture is the
    smallest input that exercises the ``go.mod``-derived candidates.
    """
    files: dict[str, str] = {}
    if gomod_text is not None:
        files["go.mod"] = gomod_text
    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeefcafef00d",
        files=files,
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(case=_gomod_case(), events=_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_gomod_comment_prefers_leading_then_trailing(
    case: tuple[str, str | None, str | None, str],
    events: dict[str, list["GoEvent"]],
) -> None:
    """Property 4.1: ``gomod_comment`` is leading-or-trailing, normalized.

    Validates Requirement 2.2: when a ``//`` comment immediately
    precedes the ``module`` line (no blank-line gap), that body wins;
    when only a trailing same-line ``//`` comment is present, that
    body wins; when neither is present, the candidate is ``None``. The
    expected value passes through the same normalize-and-truncate
    pipeline the helper applies internally.
    """
    _module_path_, leading, trailing, text = case
    rc = _build_repo(text)

    candidates = collect_go_candidates(rc, events)

    expected_raw = leading if leading is not None else trailing
    assert candidates.gomod_comment == _expected_normalized(expected_raw)


@given(case=_gomod_case(), events=_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_gomod_module_path_strips_host_org_prefix(
    case: tuple[str, str | None, str | None, str],
    events: dict[str, list["GoEvent"]],
) -> None:
    """Property 4.2: 3+-segment module paths drop ``<host>/<org>/``.

    Validates Requirement 2.3: ``github.com/acme/payment-service`` (3
    segments) becomes ``payment-service``; bare module names like
    ``cat-service`` (1 segment) and two-segment paths like
    ``acme/lib`` are passed through unchanged. The
    ``_expected_module_path`` helper computes the expected value
    through the same code path the implementation uses, so a silent
    change to the strip threshold or the normalization pipeline would
    fail this property.
    """
    module_path, _leading, _trailing, text = case
    rc = _build_repo(text)

    candidates = collect_go_candidates(rc, events)

    assert candidates.gomod_module_path == _expected_module_path(module_path)


@given(case=_gomod_case(), events=_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_package_doc_comment_uses_eligible_paths_in_sorted_order(
    case: tuple[str, str | None, str | None, str],
    events: dict[str, list["GoEvent"]],
) -> None:
    """Property 4.3: ``package_doc_comment`` is the first eligible doc.

    Validates Requirement 2.4: eligible files are root-level
    ``*.go``, ``cmd/main.go``, and ``cmd/<name>/main.go`` (one level
    deep). Non-eligible files (``internal/...``, deeper nesting) never
    contribute. The first non-empty
    :class:`PackageDocCommentEvent` over eligible files in
    sorted-path order wins; later eligible files are not consulted
    even when their doc comment is also non-empty, and ineligible
    files are skipped regardless of whether they carry a non-empty
    doc comment.
    """
    _module_path_, _leading, _trailing, text = case
    rc = _build_repo(text)

    candidates = collect_go_candidates(rc, events)

    assert candidates.package_doc_comment == _expected_doc_comment(events)


@given(case=_gomod_case(), events=_events_by_file())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_method_call_events_are_never_consulted(
    case: tuple[str, str | None, str | None, str],
    events: dict[str, list["GoEvent"]],
) -> None:
    """Property 4.4: viper string literals are never picked up.

    Validates Requirement 2.7: the helper inspects ``go.mod`` and
    :class:`PackageDocCommentEvent` instances only; every other event
    kind, including the ``MethodCallEvent`` shape that carries
    ``viper.SetConfigName(<name>)`` and
    ``viper.AddConfigPath(<path>)`` literals, must be inert.

    The property is checked structurally: running the helper with
    every ``MethodCallEvent`` stripped from ``events_by_file`` must
    produce the same :class:`GoPurposeCandidates` as running with the
    full event stream. Equality of the two outputs implies the helper
    consulted no ``MethodCallEvent`` (and therefore no viper string
    literal) for any of its three candidate slots.
    """
    _module_path_, _leading, _trailing, text = case
    rc = _build_repo(text)

    with_method_calls = collect_go_candidates(rc, events)

    stripped = {
        path: [ev for ev in evs if not isinstance(ev, MethodCallEvent)]
        for path, evs in events.items()
    }
    without_method_calls = collect_go_candidates(rc, stripped)

    assert with_method_calls == without_method_calls

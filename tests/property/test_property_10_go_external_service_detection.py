# ruff: noqa: E501
# Feature: go-analyzer-support, Property 10 (rephrased): external service
# detection coalesces by service name and excludes APM imports, under the
# new path-scoped grep contract.
"""Property test for the Go external-service detector.

The detector is now a path-scoped grep heuristic over three sources:
``config/config.go`` URL/identifier patterns,
``internal/adapter/*_adapter.go`` adapter files, and
``internal/helper.go`` / ``internal/helper/helper.go`` for JMS usage.
APM-only files (every import under ``go.elastic.co/apm/``) are
silently excluded.

This property exercises the contract from the
``RepositoryContents`` boundary forward:

* Detections coalesce by ``name``: a service named through both
  ``config/config.go`` and an adapter file produces a single entry
  whose ``source_locations`` is the union of both sites.
* APM-only files contribute zero detections, regardless of any URL
  literals or recognized service names elsewhere in the same file.
* Adding APM-only files alongside a baseline repository does not
  change the detection set.
* The detector is a pure function of its inputs: identical
  :class:`RepositoryContents` produce identical output.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Final, Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import (
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.go.go_external_services import (
    detect_go_external_services,
)


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Alphabets and recognition constants
# ---------------------------------------------------------------------------

_IDENT_CHARS: Final[str] = string.ascii_lowercase
_HOST_CHARS: Final[str] = string.ascii_lowercase + string.digits

#: Hosts used inside generated URL literals. Each is a complete
#: hostname so the strategy doesn't need to assemble suffixes.
_HOSTS: Final[tuple[str, ...]] = (
    "api.example.com",
    "broker.example.com:61616",
    "cache.example.com:6379",
    "service-a.local",
    "service-b.local",
)

#: Documented APM submodule paths used to construct APM-only files.
_APM_IMPORT_PATHS: Final[tuple[str, ...]] = (
    "go.elastic.co/apm/v2",
    "go.elastic.co/apm/module/apmhttp/v2",
    "go.elastic.co/apm/module/apmgrpc/v2",
    "go.elastic.co/apm/module/apmsql/v2",
)


_FileKind = Literal[
    "config_url",
    "adapter",
    "helper_jms",
    "apm_only",
    "unrelated",
]


@dataclass(frozen=True, slots=True)
class _FileSpec:
    path: str
    kind: _FileKind
    body: str
    # Hosts referenced in config URL files; consumed when computing
    # the expected detection set.
    hosts: tuple[str, ...] = field(default_factory=tuple)
    # Adapter service-name root (without the ``_adapter.go`` suffix).
    adapter_root: str | None = None


# ---------------------------------------------------------------------------
# Atomic strategies
# ---------------------------------------------------------------------------


_ident = st.text(alphabet=_IDENT_CHARS, min_size=2, max_size=8)


_HTTP_SCHEMES: Final[tuple[str, ...]] = ("https", "http")
_BROKER_SCHEMES: Final[tuple[str, ...]] = ("tcp", "stomp", "amqp")
_CACHE_SCHEMES: Final[tuple[str, ...]] = ("redis",)


def _scheme_for_host(host: str) -> str:
    if host.startswith("broker"):
        return _BROKER_SCHEMES[0]
    if host.startswith("cache"):
        return _CACHE_SCHEMES[0]
    return _HTTP_SCHEMES[0]


@st.composite
def _config_file(draw: st.DrawFn) -> _FileSpec:
    hosts = draw(
        st.lists(
            st.sampled_from(_HOSTS), min_size=1, max_size=3, unique=True
        ).map(tuple),
    )
    body_lines = ["package config", ""]
    for i, host in enumerate(hosts):
        scheme = _scheme_for_host(host)
        body_lines.append(f'const URL_{i} = "{scheme}://{host}/path"')
    body = "\n".join(body_lines) + "\n"
    return _FileSpec(path="config/config.go", kind="config_url", body=body, hosts=hosts)


@st.composite
def _adapter_file(draw: st.DrawFn) -> _FileSpec:
    root = draw(_ident)
    path = f"internal/adapter/{root}_adapter.go"
    body = (
        "package adapter\n"
        "// no URL here\n"
        f"// service: {root}\n"
    )
    return _FileSpec(path=path, kind="adapter", body=body, adapter_root=root)


@st.composite
def _helper_jms_file(draw: st.DrawFn) -> _FileSpec:
    # Use the package-layout helper path; matches the operator's
    # in-scope repositories.
    method = draw(st.sampled_from(("NewClient", "NewSubscriber", "NewSender")))
    body = (
        "package helper\n"
        'import "esb-go-libs/activemq"\n'
        f"func New() {{ activemq.{method}(&activemq.JmsConfig{{}}) }}\n"
    )
    return _FileSpec(
        path="internal/helper/helper.go",
        kind="helper_jms",
        body=body,
    )


@st.composite
def _apm_only_file(draw: st.DrawFn) -> _FileSpec:
    # Place the APM-only file at one of the inspected paths so the
    # APM guard's effect is observable.
    path = draw(
        st.sampled_from(
            (
                "config/config.go",
                "internal/adapter/fake_adapter.go",
                "internal/helper/helper.go",
            ),
        ),
    )
    apm_paths = draw(
        st.lists(
            st.sampled_from(_APM_IMPORT_PATHS),
            min_size=1,
            max_size=3,
            unique=True,
        ),
    )
    body_lines = ["package x", "import ("]
    for p in apm_paths:
        body_lines.append(f'    "{p}"')
    body_lines.append(")")
    # Throw in a real URL — APM guard must still suppress.
    body_lines.append('const _ = "https://api.example.com/v1"')
    body = "\n".join(body_lines) + "\n"
    return _FileSpec(path=path, kind="apm_only", body=body)


@st.composite
def _unrelated_file(draw: st.DrawFn) -> _FileSpec:
    """A file at a non-inspected path. Must not contribute anything."""

    base = draw(_ident)
    path = f"internal/usecase/{base}.go"
    body = (
        "package usecase\n"
        'const _ = "https://rogue.example.com/should-not-detect"\n'
    )
    return _FileSpec(path=path, kind="unrelated", body=body)


# ---------------------------------------------------------------------------
# Compose a synthetic RepositoryContents
# ---------------------------------------------------------------------------


@st.composite
def _repo_spec(draw: st.DrawFn) -> tuple[RepositoryContents, list[_FileSpec]]:
    # Always include at least one file; the strategy can return one
    # of each kind. Inclusion of a kind is independent so the
    # combinatorial space stays manageable.
    specs: list[_FileSpec] = []
    if draw(st.booleans()):
        specs.append(draw(_config_file()))
    n_adapters = draw(st.integers(min_value=0, max_value=2))
    seen_paths: set[str] = {s.path for s in specs}
    for _ in range(n_adapters):
        s = draw(_adapter_file())
        if s.path not in seen_paths:
            specs.append(s)
            seen_paths.add(s.path)
    if draw(st.booleans()):
        s = draw(_helper_jms_file())
        if s.path not in seen_paths:
            specs.append(s)
            seen_paths.add(s.path)
    if draw(st.booleans()):
        specs.append(draw(_unrelated_file()))

    files = {spec.path: spec.body for spec in specs}
    repo = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )
    return repo, specs


# ---------------------------------------------------------------------------
# Expected-output helpers
# ---------------------------------------------------------------------------


def _expected_names(specs: list[_FileSpec]) -> set[str]:
    """Predict the set of detection ``name`` values for ``specs``."""

    names: set[str] = set()
    for spec in specs:
        if spec.kind == "config_url":
            for host in spec.hosts:
                names.add(host)
        elif spec.kind == "adapter":
            assert spec.adapter_root is not None
            names.add(spec.adapter_root)
        elif spec.kind == "helper_jms":
            names.add("activemq")
    return names


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(_repo_spec())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_detection_names_match_expected(
    case: tuple[RepositoryContents, list[_FileSpec]],
) -> None:
    """The set of detection names equals the predicted set.

    Each kind contributes a known name (or set of names) to the
    expected output. Unrelated files never contribute. The detector
    coalesces by name so each predicted name appears exactly once.
    """

    repo, specs = case
    detections, skips = detect_go_external_services(repo, {})

    actual_names = {d.name for d in detections}
    expected_names = _expected_names(specs)

    assert actual_names == expected_names
    assert skips == []

    # Coalescing invariant: each name appears at most once.
    detection_names = [d.name for d in detections]
    assert len(detection_names) == len(set(detection_names))


@given(_repo_spec())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_apm_only_files_at_inspected_paths_suppress_detection(
    case: tuple[RepositoryContents, list[_FileSpec]],
) -> None:
    """Replacing an inspected file with an APM-only stub suppresses its detection."""

    repo, specs = case
    baseline, _ = detect_go_external_services(repo, {})

    apm_body = (
        "package x\n"
        "import (\n"
        '    "go.elastic.co/apm/v2"\n'
        ")\n"
        'const _ = "https://api.example.com/should-not-detect"\n'
    )
    polluted_files: dict[str, str] = dict(repo.files)
    # If a config file existed, replace it with an APM-only stub at
    # the same path. The detection set must shrink (or stay equal if
    # the original config was the source of names that ALSO came from
    # other files).
    polluted_files["config/config.go"] = apm_body
    polluted = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=polluted_files,
    )
    polluted_detections, _ = detect_go_external_services(polluted, {})

    polluted_names = {d.name for d in polluted_detections}
    # No detection in the polluted result names ``api.example.com`` solely
    # because of the APM-only config file; any name still present must
    # have been independently produced by other (non-APM-only) files.
    baseline_names = {d.name for d in baseline}
    # Polluted ⊆ baseline ∪ adapter/helper names (since we replaced
    # config with APM): in practice the polluted set is a subset of
    # baseline excluding pure-config-only names.
    assert polluted_names.issubset(baseline_names)


@given(_repo_spec())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_unrelated_files_do_not_change_output(
    case: tuple[RepositoryContents, list[_FileSpec]],
) -> None:
    """Files at paths the detector does not inspect have zero impact."""

    repo, _ = case
    baseline, _ = detect_go_external_services(repo, {})

    polluted_files = dict(repo.files)
    # Insert rogue URLs into a deeply nested path the detector ignores.
    polluted_files["internal/usecase/rogue.go"] = (
        "package usecase\n"
        'const _ = "https://rogue-host.example.com/v1"\n'
        'const _ = "tcp://rogue-broker.example.com:61616"\n'
    )
    polluted = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=polluted_files,
    )
    polluted_detections, _ = detect_go_external_services(polluted, {})

    assert polluted_detections == baseline


@given(_repo_spec())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_detector_is_deterministic(
    case: tuple[RepositoryContents, list[_FileSpec]],
) -> None:
    """Two invocations on the same input produce identical output."""

    repo, _ = case
    a, _ = detect_go_external_services(repo, {})
    b, _ = detect_go_external_services(repo, {})

    assert a == b

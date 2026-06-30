# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 7: For all repositories, the produced external_service_dependencies list SHALL contain at most one entry per service name, and the union of source_locations across that single entry SHALL equal the set of source locations from which the service was detected in the repository.
"""Property test for external service deduplication.

**Validates Requirement 5.3** (Property 7 in the design).

For every repository in which a small set of distinct external services is
referenced from one or more files, ``detect_external_services`` must:

1. produce **at most one** ``ExternalServiceDependency`` entry per service
   ``name``;
2. populate that entry's ``source_locations`` with the *union* of every
   site at which the service was detected, deduplicated by ``(path, line)``
   and equal (as a set of file paths) to the set of files in which the
   strategy injected the corresponding fingerprint.

Strategy outline
----------------

The strategy picks a non-empty subset of *fingerprints* — short literal
substrings whose detection by ``detect_external_services`` produces a
known, fixed canonical service ``name``. Each chosen fingerprint is
sprinkled across 1-5 distinct repository files; every file holds exactly
one occurrence of one fingerprint, so the ground-truth set of detection
sites for each service equals the set of file paths that carry its
fingerprint.

Each fingerprint below is chosen so that its mere presence triggers
exactly one named SDK pattern in the detector and produces a
deterministic canonical name (see
``src/project_knowledge_mcp/project_analyzer/external_services.py``):

* ``KafkaProducer()``           -> ``kafka``
* ``redis.Redis()``             -> ``redis``
* ``minio.Minio()``             -> ``minio``
* ``Auth0Client``               -> ``auth0``
* ``KeycloakOpenID``            -> ``keycloak``
* ``memcache.Client()``         -> ``memcached``
* ``pika.BlockingConnection``   -> ``rabbitmq``
* ``boto3.client("sqs")``       -> ``aws-sqs``
* ``boto3.client("sns")``       -> ``aws-sns``
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer.external_services import (
    detect_external_services,
)

# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

# (canonical_service_name, fingerprint_text)
#
# Each fingerprint is a literal source-code substring whose detection by
# ``detect_external_services`` produces a service entry with
# ``name == canonical_service_name``. The fingerprints are mutually
# distinct in their canonical names, and none of them contains a URL
# (``scheme://...``) or any token that would also trigger a different
# named SDK pattern, so injecting one fingerprint per file produces
# exactly one detection per file.
_FINGERPRINTS: tuple[tuple[str, str], ...] = (
    ("kafka", "KafkaProducer()"),
    ("redis", "redis.Redis()"),
    ("minio", "minio.Minio()"),
    ("auth0", "Auth0Client"),
    ("keycloak", "KeycloakOpenID"),
    ("memcached", "memcache.Client()"),
    ("rabbitmq", "pika.BlockingConnection"),
    ("aws-sqs", 'boto3.client("sqs")'),
    ("aws-sns", 'boto3.client("sns")'),
)


# ---------------------------------------------------------------------------
# Path strategy
# ---------------------------------------------------------------------------

# Top-level directories that are *not* on the detector's skip list
# (``node_modules/``, ``vendor/``, ``.venv/``, ``dist/``, ``build/`` ...).
_SAFE_SUBDIRS: tuple[str, ...] = (
    "src",
    "lib",
    "app",
    "core",
    "pkg",
    "internal",
    "modules",
    "services",
)

# A small alphabet that yields short, distinct lowercase identifiers
# without hitting any of the detector's filename or extension skip rules.
_NAME_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

_path_strategy = st.builds(
    lambda subdir, name, idx: f"{subdir}/{name}_{idx}.py",
    st.sampled_from(_SAFE_SUBDIRS),
    st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=6),
    st.integers(min_value=0, max_value=9999),
)


# ---------------------------------------------------------------------------
# Composite strategy
# ---------------------------------------------------------------------------


@st.composite
def _multi_file_repos(
    draw: st.DrawFn,
) -> tuple[RepositoryContents, dict[str, set[str]]]:
    """Build a (repository, expected) pair.

    ``expected`` maps each canonical service name injected by the
    strategy to the exact set of file paths in which that service's
    fingerprint was placed. Each file holds exactly one fingerprint,
    so the union of ``(path, line)`` detection sites for that service
    is, as a set of paths, exactly ``expected[name]``.
    """
    # Pick a non-empty subset of fingerprints, distinct by canonical name.
    chosen = draw(
        st.lists(
            st.sampled_from(_FINGERPRINTS),
            min_size=1,
            max_size=len(_FINGERPRINTS),
            unique=True,
        )
    )

    # For each chosen fingerprint, decide how many files carry it (1..5).
    counts = [draw(st.integers(min_value=1, max_value=5)) for _ in chosen]
    total = sum(counts)

    # Pre-draw exactly ``total`` distinct safe paths so each fingerprint
    # gets a disjoint slice of file paths and no file ends up holding
    # more than one fingerprint.
    paths = draw(
        st.lists(
            _path_strategy,
            min_size=total,
            max_size=total,
            unique=True,
        )
    )

    files: dict[str, str] = {}
    expected: dict[str, set[str]] = {}
    cursor = 0
    for (service_name, fingerprint), n in zip(chosen, counts, strict=True):
        service_paths = set(paths[cursor : cursor + n])
        cursor += n
        # Deterministic, fingerprint-only content. The leading comment
        # never contains '://' or any other named-SDK trigger, so the
        # only detection in the file is the fingerprint on line 2.
        for path in service_paths:
            files[path] = f"# uses {service_name}\n{fingerprint}\n"
        expected[service_name] = service_paths

    repo = RepositoryContents(
        gitlab_project_id=1,
        commit_sha="0" * 40,
        files=files,
    )
    return repo, expected


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(case=_multi_file_repos())
@settings(max_examples=100)
def test_external_service_deduplication(
    case: tuple[RepositoryContents, dict[str, set[str]]],
) -> None:
    """Property 7: one entry per service; source_locations is the union of detection sites."""
    repo, expected = case

    result = detect_external_services(repo)

    # (1) At most one entry per service name (Requirement 5.3 / Property 7).
    detected_names = [dep.name for dep in result]
    assert len(detected_names) == len(set(detected_names)), (
        f"duplicate service entries in result: {detected_names}"
    )

    # (2) The set of detected service names equals the set of services
    # that the strategy actually injected -- nothing missing, nothing
    # extra. The fingerprints are chosen so each one yields exactly one
    # canonical name, with no overlap or accidental cross-detections.
    detected_by_name = {dep.name: dep for dep in result}
    assert set(detected_by_name.keys()) == set(expected.keys()), (
        f"detected names {sorted(detected_by_name)} != injected "
        f"{sorted(expected)}"
    )

    # (3) For each detected service, the union of source_locations
    # equals the exact set of detection sites injected by the strategy:
    # one (path, line=2) per file, with no internal duplicates.
    for service_name, expected_paths in expected.items():
        dep = detected_by_name[service_name]

        # No duplicate (path, line) pairs within a single entry.
        keyed = [(loc.path, loc.line) for loc in dep.source_locations]
        assert len(keyed) == len(set(keyed)), (
            f"service {service_name!r} has duplicated source_locations: {keyed}"
        )

        # The set of source-location *paths* equals the set of files in
        # which the strategy injected the fingerprint.
        actual_paths = {loc.path for loc in dep.source_locations}
        assert actual_paths == expected_paths, (
            f"service {service_name!r} source paths {sorted(actual_paths)} "
            f"!= injected {sorted(expected_paths)}"
        )

        # Every source location lies inside the repository's file set.
        assert actual_paths.issubset(set(repo.file_paths))

"""Unit tests for ``project_analyzer.go.go_external_services``.

The detector is now a path-scoped grep heuristic over three sources:

1. ``config/config.go`` URL and identifier-named constants.
2. ``internal/adapter/*_adapter.go`` (strict single-segment) files.
3. ``internal/helper.go`` or ``internal/helper/helper.go`` JMS usage.

These tests build synthetic :class:`RepositoryContents` snapshots and
assert on the public detection contract.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import (
    ExternalServiceKind,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.go.go_external_services import (
    detect_go_external_services,
)

pytestmark = pytest.mark.unit


def _repo(files: dict[str, str]) -> RepositoryContents:
    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )


# ---------------------------------------------------------------------------
# Empty / no-match cases
# ---------------------------------------------------------------------------


def test_empty_repo_returns_empty_detections() -> None:
    detections, skips = detect_go_external_services(_repo({}), {})

    assert detections == []
    assert skips == []


def test_repository_without_any_match_paths_emits_nothing() -> None:
    files = {
        "cmd/main.go": "package main\n",
        "go.mod": "module example.com\n",
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert detections == []


# ---------------------------------------------------------------------------
# config/config.go URL scan
# ---------------------------------------------------------------------------


def test_config_url_emits_http_api_with_host_name() -> None:
    files = {
        "config/config.go": (
            'package config\n'
            '\n'
            'var defaultConfig = `\n'
            'api_url: "https://api.example.com/v1/users"\n'
            '`\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.name == "api.example.com"
    assert dep.kind is ExternalServiceKind.HTTP_API


def test_config_tcp_url_emits_message_broker() -> None:
    files = {
        "config/config.go": (
            'package config\n'
            'const BROKER = "tcp://broker.example.com:61616"\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.name == "broker.example.com:61616"
    assert dep.kind is ExternalServiceKind.MESSAGE_BROKER


def test_config_redis_url_emits_cache() -> None:
    files = {
        "config/config.go": (
            'package config\n'
            'const CACHE = "redis://cache.example.com:6379"\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    assert detections[0].kind is ExternalServiceKind.CACHE


def test_config_identifier_with_non_url_value_uses_identifier_root() -> None:
    """An identifier ending in ``_URL`` whose value is not a recognized
    URL falls back to the identifier's stem (lower-cased)."""

    files = {
        "config/config.go": (
            'package config\n'
            'const FEC_POOL_SERVICE_URL = "internal-only-binding"\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    assert detections[0].name == "fec_pool_service"


def test_config_multiple_urls_coalesce_by_host() -> None:
    files = {
        "config/config.go": (
            'package config\n'
            'const A = "https://api.example.com/foo"\n'
            'const B = "https://api.example.com/bar"\n'
            'const C = "https://other.example.com"\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    names = sorted(d.name for d in detections)
    assert names == ["api.example.com", "other.example.com"]


# ---------------------------------------------------------------------------
# Adapter file scan
# ---------------------------------------------------------------------------


def test_adapter_file_emits_service_name_from_filename() -> None:
    files = {
        "internal/adapter/payment_adapter.go": (
            "package adapter\n"
            "// no URLs here\n"
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.name == "payment"
    assert dep.kind is ExternalServiceKind.OTHER


def test_adapter_file_with_url_folds_host_into_name() -> None:
    files = {
        "internal/adapter/activemq_adapter.go": (
            "package adapter\n"
            'const url = "tcp://broker.example.com:61616"\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.name == "activemq (broker.example.com:61616)"
    assert dep.kind is ExternalServiceKind.MESSAGE_BROKER


def test_adapter_at_nested_path_does_not_match_strict_single_segment_rule() -> None:
    """A path with an extra segment between adapter/ and the filename
    does not qualify."""

    files = {
        "internal/adapter/sub/payment_adapter.go": "package adapter\n",
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert detections == []


def test_adapter_filename_without_underscore_suffix_is_ignored() -> None:
    """``adapter.go`` (no leading service name) does not qualify."""

    files = {
        "internal/adapter/adapter.go": "package adapter\n",
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert detections == []


# ---------------------------------------------------------------------------
# Helper JMS detection
# ---------------------------------------------------------------------------


def test_helper_with_new_client_and_activemq_emits_message_broker() -> None:
    files = {
        "internal/helper/helper.go": (
            "package helper\n"
            'import "esb-go-libs/activemq"\n'
            "func New() *activemq.Client {\n"
            "    return activemq.NewClient(&activemq.JmsConfig{})\n"
            "}\n"
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.name == "activemq"
    assert dep.kind is ExternalServiceKind.MESSAGE_BROKER


def test_helper_at_root_path_also_matches() -> None:
    """The detector accepts both ``internal/helper.go`` and the
    package layout ``internal/helper/helper.go``."""

    files = {
        "internal/helper.go": (
            "package helper\n"
            'import "esb-go-libs/stomp"\n'
            "func New() { stomp.NewSubscriber() }\n"
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    assert detections[0].name == "activemq"


def test_helper_without_jms_library_token_does_not_emit() -> None:
    """``NewClient`` alone is not enough — the file must also mention
    ``activemq`` or ``stomp``."""

    files = {
        "internal/helper/helper.go": (
            "package helper\n"
            "func NewClient() {}\n"
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert detections == []


def test_helper_jms_destinations_from_config_appear_as_source_locations() -> None:
    """Destination constants in ``config/config.go`` attach to the
    activemq entry's ``source_locations``."""

    files = {
        "config/config.go": (
            "package config\n"
            'const REPDestination = "REP.SERVICE.PAYMENT.ERR"\n'
            'const SuccessDestination = "REP.SERVICE.PAYMENT.SUCCESS"\n'
        ),
        "internal/helper/helper.go": (
            "package helper\n"
            'import "esb-go-libs/activemq"\n'
            "func New() { activemq.NewClient(&activemq.JmsConfig{}) }\n"
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    activemq_entries = [d for d in detections if d.name == "activemq"]
    assert len(activemq_entries) == 1
    sources = activemq_entries[0].source_locations
    paths = sorted({loc.path for loc in sources})
    assert paths == ["config/config.go", "internal/helper/helper.go"]
    # Config destinations point at specific lines.
    config_lines = sorted(
        loc.line for loc in sources if loc.path == "config/config.go" and loc.line is not None
    )
    assert len(config_lines) == 2


# ---------------------------------------------------------------------------
# APM exclusion guard
# ---------------------------------------------------------------------------


def test_apm_only_helper_file_is_skipped() -> None:
    files = {
        "internal/helper/helper.go": (
            "package helper\n"
            "import (\n"
            '    "go.elastic.co/apm/v2"\n'
            '    "go.elastic.co/apm/module/apmhttp/v2"\n'
            ")\n"
            "// no real JMS — APM only\n"
            "func NewClient() {}\n"
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert detections == []


def test_apm_only_adapter_file_is_skipped() -> None:
    files = {
        "internal/adapter/payment_adapter.go": (
            "package adapter\n"
            "import (\n"
            '    "go.elastic.co/apm/v2"\n'
            ")\n"
            'const url = "https://api.example.com"\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert detections == []


def test_mixed_apm_and_other_imports_do_not_trigger_guard() -> None:
    files = {
        "internal/adapter/payment_adapter.go": (
            "package adapter\n"
            "import (\n"
            '    "go.elastic.co/apm/v2"\n'
            '    "net/http"\n'
            ")\n"
            'const url = "https://api.example.com"\n'
        ),
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    assert len(detections) == 1
    assert "payment" in detections[0].name


# ---------------------------------------------------------------------------
# Coalescing across passes
# ---------------------------------------------------------------------------


def test_same_name_from_config_and_adapter_coalesces() -> None:
    files = {
        "config/config.go": (
            "package config\n"
            'const FEC_POOL_SERVICE_URL = "internal-only"\n'
        ),
        "internal/adapter/fec_pool_service_adapter.go": "package adapter\n",
    }

    detections, _ = detect_go_external_services(_repo(files), {})

    fec_entries = [d for d in detections if d.name == "fec_pool_service"]
    assert len(fec_entries) == 1
    # Two distinct source locations contributed.
    assert len(fec_entries[0].source_locations) == 2

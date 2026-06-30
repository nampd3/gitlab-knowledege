"""Unit tests for ``gitlab_connector._is_analyzable_path``.

The predicate decides whether the connector will spend an HTTPS
round-trip on a given file path. Two behaviors are pinned here:

1. **Admission rules.** READMEs at any depth, the documented manifest
   basenames, and source-file extensions are admitted; everything
   else (lockfiles, images, fonts, sibling dotfiles) is rejected.
2. **Vendor exclusion.** Any path whose directory portion contains a
   segment literally named ``vendor`` is rejected regardless of
   extension. This mirrors :func:`is_go_source_file`'s parser-side
   rule (Requirement 1.3) and prevents the connector from spending
   hours fetching vendored Go bundles that the analyzer would
   immediately discard.

The vendor case is the most operationally important: a real Go
service (e.g. ``repayment_service``) commits thousands of vendored
``.go`` files under ``vendor/``. Without this rule the per-project
fetch took multiple hours; with it, fetches drop to the hand-written
source.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.gitlab_connector import _is_analyzable_path

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Admission rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "README",
        "docs/README.rst",
        "go.mod",
        "pyproject.toml",
        "package.json",
        "main.go",
        "internal/server/router.go",
        "src/index.ts",
        "src/app.py",
        "schema.sql",
    ],
)
def test_admits_documented_paths(path: str) -> None:
    """Each documented family of analyzable paths is admitted."""
    assert _is_analyzable_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "",
        "go.sum",  # lockfile
        "image.png",
        "fonts/Inter.woff2",
        "noextension",
        ".gitignore",  # leading dot, no extension match
    ],
)
def test_rejects_non_analyzable_paths(path: str) -> None:
    """Lockfiles, binaries, and extensionless dotfiles are skipped."""
    assert _is_analyzable_path(path) is False


# ---------------------------------------------------------------------------
# Vendor exclusion (Requirement 1.3 mirror)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "vendor/foo.go",
        "vendor/github.com/pkg/errors/errors.go",
        "src/vendor/foo.go",
        "internal/vendor/foo/bar.go",
        "src\\vendor\\foo.go",
        "src\\vendor/foo.go",
        # Non-Go file types under vendor are also rejected — the
        # convention applies across languages (Composer/PHP, etc.).
        "vendor/readme.md",
        "vendor/package.json",
        "vendor/schema.sql",
    ],
)
def test_rejects_paths_under_vendor_segment(path: str) -> None:
    """Any directory segment literally named ``vendor`` excludes the path."""
    assert _is_analyzable_path(path) is False


@pytest.mark.parametrize(
    "path",
    [
        "vendor.go",  # filename, not segment
        "pkg/vendor.go",
        "src/cmd/vendor.go",
        "myvendor/foo.go",  # segment merely starts with "vendor"
        "thevendor/foo.go",
        "vendors/foo.go",  # plural
        "Vendor/foo.go",  # case-sensitive
        "_vendor/foo.go",  # leading underscore
    ],
)
def test_admits_paths_with_vendor_in_name_but_no_vendor_segment(path: str) -> None:
    """The vendor exclusion is a directory-segment rule, not a substring rule."""
    assert _is_analyzable_path(path) is True


# ---------------------------------------------------------------------------
# Go-specific stricter filter (_is_analyzable_path_for_go)
# ---------------------------------------------------------------------------

from project_knowledge_mcp.gitlab_connector import _is_analyzable_path_for_go


@pytest.mark.parametrize(
    "path",
    [
        "main.go",
        "cmd/main.go",
        "cmd/server/server.go",
        "internal/repository/order_repo.go",
        "internal/adapter/payment_adapter.go",
        "internal/helper/helper.go",
        "pb/service.pb.go",
        "pb/service.pb.gw.go",
        "go.mod",
        "README.md",
        "README",
        "docs/README.rst",
    ],
)
def test_go_filter_admits_files_the_go_analyzer_actually_reads(path: str) -> None:
    """Files the Go analyzer pipeline consumes pass the stricter filter."""
    assert _is_analyzable_path_for_go(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # Test files: never useful for production analysis.
        "internal/util/query_validator_test.go",
        "cmd/server/server_test.go",
        # Protobuf definitions: junk URL literals only.
        "proto/google/api/http.proto",
        "proto/service.proto",
        # Swagger UI / docs assets: junk URL literals only.
        "docs/swagger-ui.js",
        "docs/swagger-ui.css",
        "docs/index.html",
        "docs/pool_apis.swagger.json",
        # SOAP / sample XML fixtures: junk.
        "wsdl/RepaymentService.wsdl",
        "sampleInput/Pega_ApplicationUpdatedMessage.xml",
        # Migration scripts: not consulted by the new DB detector.
        "scripts/sp_cat_capture_repayments.sql",
        # Non-root go.mod files (vendored sub-modules).
        "vendor/example.com/lib/go.mod",
        "third_party/lib/go.mod",
        # Vendor / third_party directory segments.
        "vendor/example.com/lib/foo.go",
        "third_party/protoc/include/google/protobuf/api.proto",
        # YAML / TOML / JSON / Dockerfile: not consulted by Go analyzer.
        "config/dev.yaml",
        "pyproject.toml",
        "package.json",
        "Dockerfile",
    ],
)
def test_go_filter_rejects_files_the_go_analyzer_ignores(path: str) -> None:
    """The fetch filter drops files the Go analyzer would discard anyway."""
    assert _is_analyzable_path_for_go(path) is False


@pytest.mark.parametrize(
    "path",
    [
        "vendor.go",  # filename, not a segment
        "pkg/vendor.go",
        "third_party.go",  # filename, not a segment
        "myvendor/foo.go",
        "vendors/foo.go",
        "Vendor/foo.go",  # case-sensitive
        "ThirdParty/foo.go",  # case-sensitive
    ],
)
def test_go_filter_treats_vendor_and_third_party_as_directory_segments(
    path: str,
) -> None:
    """Only exact directory-segment matches trigger the vendor / third_party rejection."""
    assert _is_analyzable_path_for_go(path) is True

"""Unit tests for ``config.load_and_validate`` happy paths.

These tests pin down the *acceptance* side of the configuration loader: for
each set of required-and-optional inputs that the design's *Config Loader /
Validator* table marks as valid, ``load_and_validate`` must return a
populated :class:`Config` whose fields match the inputs (or the documented
defaults, where the optional input is absent).

One test is provided per required key set, per the task's explicit list:

* minimal valid config (only required keys; defaults applied)
* explicit ``analysis.branch``
* explicit ``visualization.port``
* explicit ``refresh.interval``
* all keys populated together

Each test uses an injected ``env`` mapping (rather than mutating
:data:`os.environ`) so the test is hermetic, and asserts on every field of
:class:`Config` so a regression in any field surfaces immediately.

Implements Requirements 1.1, 1.2, 1.3, 12.3, 12.4, 15.1, 15.2.
"""

from __future__ import annotations

import io
from datetime import timedelta

import pytest

from project_knowledge_mcp.config import (
    DEFAULT_ANALYSIS_BRANCH,
    DEFAULT_VISUALIZATION_PORT,
    Config,
    load_and_validate,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared minimal-valid environment
# ---------------------------------------------------------------------------

# The three required environment variables, populated with realistic values.
# Every happy-path test starts from this base and adds zero or more optional
# variables on top.
_REQUIRED_ENV: dict[str, str] = {
    "GITLAB_BASE_URL": "https://gitlab.example.com",
    "GITLAB_GROUP_PATH": "acme/platform",
    "GITLAB_ACCESS_TOKEN": "glpat-abc123",
}


def _load(env: dict[str, str]) -> Config:
    """Invoke :func:`load_and_validate` with a captured stderr stream.

    A captured stream is passed even on the happy path so that any spurious
    diagnostic output would be visible in the test failure rather than the
    real test runner stderr. On success no output is produced.
    """
    err_buffer = io.StringIO()
    config = load_and_validate(env, stderr=err_buffer)
    # Happy paths must not write any diagnostic line to stderr.
    assert err_buffer.getvalue() == ""
    return config


# ---------------------------------------------------------------------------
# Happy-path tests, one per required key set
# ---------------------------------------------------------------------------


def test_minimal_required_keys_apply_documented_defaults() -> None:
    """Only the three required keys are set; all optional keys default.

    Implements Requirements 1.1, 1.2, 1.3, 12.4, 15.2.
    """
    config = _load(dict(_REQUIRED_ENV))

    assert config == Config(
        gitlab_base_url="https://gitlab.example.com",
        gitlab_group_path="acme/platform",
        gitlab_access_token="glpat-abc123",
        gitlab_verify_ssl=True,
        analysis_branch=DEFAULT_ANALYSIS_BRANCH,
        refresh_interval=None,
        visualization_port=DEFAULT_VISUALIZATION_PORT,
    )
    # Spot-check the documented default values directly so a regression in
    # the constants is also caught here.
    assert config.analysis_branch == "uat"
    assert config.visualization_port == 7345
    assert config.gitlab_verify_ssl is True


def test_explicit_analysis_branch_overrides_default() -> None:
    """An explicit ``ANALYSIS_BRANCH`` is preserved verbatim.

    Implements Requirements 1.1, 1.2, 1.3, 15.1.
    """
    env = {**_REQUIRED_ENV, "ANALYSIS_BRANCH": "main"}

    config = _load(env)

    assert config.analysis_branch == "main"
    # Other optional fields still take their defaults.
    assert config.refresh_interval is None
    assert config.visualization_port == DEFAULT_VISUALIZATION_PORT
    # Required fields are unchanged.
    assert config.gitlab_base_url == "https://gitlab.example.com"
    assert config.gitlab_group_path == "acme/platform"
    assert config.gitlab_access_token == "glpat-abc123"


def test_explicit_visualization_port_overrides_default() -> None:
    """An explicit ``VISUALIZATION_PORT`` is parsed as an integer.

    Implements Requirements 1.1, 1.2, 1.3, 12.3.
    """
    env = {**_REQUIRED_ENV, "VISUALIZATION_PORT": "8080"}

    config = _load(env)

    assert config.visualization_port == 8080
    assert isinstance(config.visualization_port, int)
    # Other optional fields still take their defaults.
    assert config.analysis_branch == DEFAULT_ANALYSIS_BRANCH
    assert config.refresh_interval is None


def test_explicit_refresh_interval_parsed_as_duration() -> None:
    """``REFRESH_INTERVAL`` accepts the documented duration syntax.

    Implements Requirements 1.1, 1.2, 1.3.
    """
    env = {**_REQUIRED_ENV, "REFRESH_INTERVAL": "5m"}

    config = _load(env)

    assert config.refresh_interval == timedelta(minutes=5)
    # Other optional fields still take their defaults.
    assert config.analysis_branch == DEFAULT_ANALYSIS_BRANCH
    assert config.visualization_port == DEFAULT_VISUALIZATION_PORT


def test_all_keys_populated_together() -> None:
    """Every documented configuration key is provided.

    Implements Requirements 1.1, 1.2, 1.3, 12.3, 15.1.
    """
    env = {
        "GITLAB_BASE_URL": "https://gitlab.internal.acme.com",
        "GITLAB_GROUP_PATH": "acme/integration",
        "GITLAB_ACCESS_TOKEN": "glpat-xyz789",
        "GITLAB_VERIFY_SSL": "false",
        "ANALYSIS_BRANCH": "release",
        "REFRESH_INTERVAL": "1h",
        "VISUALIZATION_PORT": "9090",
    }

    config = _load(env)

    assert config == Config(
        gitlab_base_url="https://gitlab.internal.acme.com",
        gitlab_group_path="acme/integration",
        gitlab_access_token="glpat-xyz789",
        gitlab_verify_ssl=False,
        analysis_branch="release",
        refresh_interval=timedelta(hours=1),
        visualization_port=9090,
    )

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("Yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("  true  ", True),
        ("  false  ", False),
    ],
)
def test_gitlab_verify_ssl_accepts_documented_spellings(
    raw: str, expected: bool
) -> None:
    """Every documented truthy/falsy spelling is recognized.

    Implements ``gitlab.verify_ssl`` parsing — see the README env-vars table.
    """
    env = {**_REQUIRED_ENV, "GITLAB_VERIFY_SSL": raw}

    config = _load(env)

    assert config.gitlab_verify_ssl is expected


@pytest.mark.parametrize("raw", ["", "   "])
def test_gitlab_verify_ssl_unset_or_empty_defaults_to_true(raw: str) -> None:
    """Unset and whitespace-only values fall back to the documented default."""
    env = {**_REQUIRED_ENV, "GITLAB_VERIFY_SSL": raw}

    config = _load(env)

    assert config.gitlab_verify_ssl is True


def test_gitlab_verify_ssl_unknown_value_is_rejected_with_named_key() -> None:
    """A value outside the documented spellings produces a single diagnostic line."""
    env = {**_REQUIRED_ENV, "GITLAB_VERIFY_SSL": "maybe"}
    err_buffer = io.StringIO()

    with pytest.raises(SystemExit) as exit_info:
        load_and_validate(env, stderr=err_buffer)

    assert exit_info.value.code == 1
    diagnostic = err_buffer.getvalue().rstrip("\n")
    assert diagnostic == (
        "configuration error for 'gitlab.verify_ssl': "
        "must be one of true/false, 1/0, yes/no, on/off"
    )

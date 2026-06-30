# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 1: For all configurations in which any of gitlab.base_url, gitlab.group_path, gitlab.access_token, analysis.branch (when present), or visualization.port (when present) is missing, empty when not allowed, or otherwise outside its declared validation rule, the MCP_Server SHALL fail startup before either surface accepts traffic, emit an error message that names the offending configuration key, and terminate the process.
"""Property test for configuration validation at startup.

**Validates Requirements 1.4, 1.5, 12.5, 15.6** (Property 1 in the design).

For every configuration mapping in which exactly one of the five keys named
in Property 1 has been mutated into an invalid value, ``load_and_validate``
must:

1. raise ``SystemExit`` with a non-zero status (the ``MCP_Server`` is
   terminated before either surface accepts traffic);
2. write **exactly one** line to the captured stderr stream;
3. and that line must name the dotted configuration key whose value was
   mutated, so the operator can locate the offending key.

The ``refresh.interval`` key is intentionally not mutated here: Property 1
scopes the obligation to ``gitlab.base_url``, ``gitlab.group_path``,
``gitlab.access_token``, ``analysis.branch`` (when present), and
``visualization.port`` (when present).
"""

from __future__ import annotations

import io

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.config import (
    ENV_ANALYSIS_BRANCH,
    ENV_GITLAB_ACCESS_TOKEN,
    ENV_GITLAB_BASE_URL,
    ENV_GITLAB_GROUP_PATH,
    ENV_VISUALIZATION_PORT,
    KEY_ANALYSIS_BRANCH,
    KEY_GITLAB_ACCESS_TOKEN,
    KEY_GITLAB_BASE_URL,
    KEY_GITLAB_GROUP_PATH,
    KEY_VISUALIZATION_PORT,
    load_and_validate,
)

# ---------------------------------------------------------------------------
# Test fixture data
# ---------------------------------------------------------------------------

# The five configuration keys explicitly enumerated in Property 1, each
# paired with the environment variable that supplies its raw value.
_KEY_TO_ENV: dict[str, str] = {
    KEY_GITLAB_BASE_URL: ENV_GITLAB_BASE_URL,
    KEY_GITLAB_GROUP_PATH: ENV_GITLAB_GROUP_PATH,
    KEY_GITLAB_ACCESS_TOKEN: ENV_GITLAB_ACCESS_TOKEN,
    KEY_ANALYSIS_BRANCH: ENV_ANALYSIS_BRANCH,
    KEY_VISUALIZATION_PORT: ENV_VISUALIZATION_PORT,
}


def _valid_base_env() -> dict[str, str]:
    """Return a fresh, fully-valid configuration environment.

    Every test vector is produced by mutating exactly one field of this
    base mapping into an invalid value so any failure is unambiguously
    attributable to the mutated key.
    """
    return {
        ENV_GITLAB_BASE_URL: "https://gitlab.example.com",
        ENV_GITLAB_GROUP_PATH: "my-group/sub-group",
        ENV_GITLAB_ACCESS_TOKEN: "valid-token-value",
    }


# ---------------------------------------------------------------------------
# Per-key invalid-value strategies
# ---------------------------------------------------------------------------

# Strings that are empty or contain only whitespace; these violate the
# "non-empty" rule for required string fields and the analysis.branch
# "must not be empty when present" rule.
_EMPTY_OR_WHITESPACE = st.sampled_from(["", " ", "   ", "\t", "\n", " \t \n "])

# Values that violate the http(s)-URL-with-host rule for gitlab.base_url:
# wrong scheme, missing scheme, missing host, or non-URL strings.
_INVALID_URLS = st.sampled_from(
    [
        "not-a-url",
        "example.com",
        "ftp://example.com",
        "file:///etc/passwd",
        "https://",
        "http://",
        "https:///path-only-no-host",
        "://example.com",
        "javascript:alert(1)",
    ]
)

# Strings that violate the visualization.port rule: either non-integer or
# out of [1, 65535]. An empty string is intentionally excluded because the
# loader treats absent/empty as "use the default" (Requirement 12.4), so
# only non-empty invalid forms exercise Property 1's "outside validation
# rule" clause.
_INVALID_PORTS = st.one_of(
    st.sampled_from(
        [
            "abc",
            "1.5",
            "1e3",
            "+42",
            " 42",
            "42 ",
            "0x10",
            "seven",
            "--1",
        ]
    ),
    st.integers(min_value=65536, max_value=10_000_000).map(str),
    st.integers(min_value=-10_000_000, max_value=0).map(str),
)


@st.composite
def _mutated_envs(draw: st.DrawFn) -> tuple[str, dict[str, str]]:
    """Build a (target_key, env) pair where exactly one field is invalid.

    The base environment populates every required field with a valid
    value. The composite then optionally adds the two optional fields
    with valid values (so the "when present" clause of Property 1 is
    exercised in both presence states), picks one of the five keys at
    random, and mutates only that key into a value that violates its
    declared validation rule.
    """
    env = _valid_base_env()

    # Optionally include the optional keys with valid values, so the
    # property test exercises both the "absent → default" and the
    # "valid override" paths for analysis.branch and visualization.port.
    if draw(st.booleans()):
        env[ENV_ANALYSIS_BRANCH] = draw(
            st.sampled_from(["uat", "main", "develop", "release/2024-Q1"])
        )
    if draw(st.booleans()):
        env[ENV_VISUALIZATION_PORT] = draw(
            st.integers(min_value=1, max_value=65535).map(str)
        )

    target_key = draw(st.sampled_from(list(_KEY_TO_ENV.keys())))
    env_var = _KEY_TO_ENV[target_key]

    if target_key == KEY_GITLAB_BASE_URL:
        mutation = draw(st.sampled_from(["missing", "empty", "non_url"]))
        if mutation == "missing":
            env.pop(env_var, None)
        elif mutation == "empty":
            env[env_var] = draw(_EMPTY_OR_WHITESPACE)
        else:
            env[env_var] = draw(_INVALID_URLS)
    elif target_key in (KEY_GITLAB_GROUP_PATH, KEY_GITLAB_ACCESS_TOKEN):
        mutation = draw(st.sampled_from(["missing", "empty"]))
        if mutation == "missing":
            env.pop(env_var, None)
        else:
            env[env_var] = draw(_EMPTY_OR_WHITESPACE)
    elif target_key == KEY_ANALYSIS_BRANCH:
        # "when present": invalid only when the key IS in the environment.
        # Empty / whitespace is the documented failure mode (Req 15.6).
        env[env_var] = draw(_EMPTY_OR_WHITESPACE)
    else:  # KEY_VISUALIZATION_PORT
        # "when present" with a non-empty invalid value (an empty string
        # would default to 7345 by design, not fail validation).
        env[env_var] = draw(_INVALID_PORTS)

    return target_key, env


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(case=_mutated_envs())
@settings(max_examples=100)
def test_invalid_configuration_terminates_with_named_key(
    case: tuple[str, dict[str, str]],
) -> None:
    """Property 1: invalid config terminates startup with a key-named line."""
    target_key, env = case
    captured = io.StringIO()

    with pytest.raises(SystemExit) as exc_info:
        load_and_validate(env, stderr=captured)

    # (1) Process is terminated with a non-zero status — neither MCP nor
    # the Visualization_Server has had a chance to accept traffic yet.
    assert exc_info.value.code is not None
    assert exc_info.value.code != 0

    output = captured.getvalue()

    # (2) Exactly one line is written to stderr. Empty trailing lines from
    # the final newline produced by ``print`` are not counted.
    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 1, (
        f"expected exactly one stderr line, got {len(lines)}: {output!r}"
    )

    # (3) That line names the dotted configuration key whose value was
    # mutated, so the operator can locate the offending key directly.
    assert target_key in lines[0], (
        f"expected stderr to name '{target_key}', got: {lines[0]!r}"
    )

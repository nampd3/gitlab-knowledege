"""Configuration loader and validator.

This module is the single termination-on-startup-failure path that
Requirements 1.4, 1.5, 12.5, and 15.6 collapse into. ``load_and_validate``
reads configuration from environment variables, normalizes and validates
each value per the *Config Loader / Validator* table in the design, and
either returns a populated :class:`Config` or terminates the process with
a single stderr line that names the offending configuration key and the
specific reason the value was rejected.

Environment variable mapping
----------------------------

================================  ============================  ==========================
Environment variable              Configuration key             Source Requirement
================================  ============================  ==========================
``GITLAB_BASE_URL``               ``gitlab.base_url``           1.1, 1.4
``GITLAB_GROUP_PATH``             ``gitlab.group_path``         1.2, 1.4
``GITLAB_ACCESS_TOKEN``           ``gitlab.access_token``       1.3, 1.5
``ANALYSIS_BRANCH``               ``analysis.branch``           15.1, 15.2, 15.6
``REFRESH_INTERVAL``              ``refresh.interval``          8.3
``VISUALIZATION_PORT``            ``visualization.port``        12.3, 12.4, 12.5
================================  ============================  ==========================

For testability, :func:`load_and_validate` accepts an optional
``env`` mapping; when not supplied it defaults to :data:`os.environ`.

Validation rules
----------------

The validation rules are precisely those listed in the *Config Loader /
Validator* table in the design:

* ``gitlab.base_url``: required, non-empty, must parse as an
  ``http://`` or ``https://`` URL with a host.
* ``gitlab.group_path``: required, non-empty.
* ``gitlab.access_token``: required, non-empty.
* ``analysis.branch``: optional, defaults to ``"uat"``; an empty
  string is rejected (Requirement 15.6).
* ``refresh.interval``: optional; absent or empty means *no schedule*.
  When provided, must parse as a duration (bare integer = seconds, or a
  positive integer with one of the suffixes ``s``, ``m``, ``h``, ``d``)
  and must be at least 1 minute.
* ``visualization.port``: optional, defaults to ``7345``; must parse
  as an integer in ``[1, 65535]``.

Failure mode
------------

On any validation failure :func:`load_and_validate` writes one line to
stderr of the form::

    configuration error for '{key}': {reason}

and calls :func:`sys.exit` with a non-zero status code. Neither MCP nor
Visualization_Server is started. This shared failure-mode implementation
is the consolidation point referenced by Requirements 1.4, 1.5, 12.5,
and 15.6.

Implements Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 12.3, 12.4, 12.5,
15.1, 15.2, 15.6.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Final
from urllib.parse import urlsplit

from .errors import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TextIO


# ---------------------------------------------------------------------------
# Defaults and bounds
# ---------------------------------------------------------------------------

#: Default Analysis_Branch when ``ANALYSIS_BRANCH`` is unset (Requirement 15.2).
DEFAULT_ANALYSIS_BRANCH: Final[str] = "uat"

#: Default Visualization_Server TCP port when ``VISUALIZATION_PORT`` is unset
#: (Requirement 12.4).
DEFAULT_VISUALIZATION_PORT: Final[int] = 7345

#: Default value for ``gitlab.verify_ssl``. SSL/TLS certificate verification
#: is on by default; operators MUST explicitly opt out (e.g. for a GitLab
#: instance fronted by a self-signed certificate) by setting
#: ``GITLAB_VERIFY_SSL=false``.
DEFAULT_GITLAB_VERIFY_SSL: Final[bool] = True

#: Minimum permitted refresh interval. The design's *Config Loader / Validator*
#: table requires "duration >= 1 minute" for ``refresh.interval``.
MIN_REFRESH_INTERVAL: Final[timedelta] = timedelta(minutes=1)

#: Lowest permitted TCP port value for ``visualization.port`` (Requirement 12.5).
MIN_PORT: Final[int] = 1

#: Highest permitted TCP port value for ``visualization.port`` (Requirement 12.5).
MAX_PORT: Final[int] = 65535


# ---------------------------------------------------------------------------
# Environment variable names (the runtime input surface)
# ---------------------------------------------------------------------------

ENV_GITLAB_BASE_URL: Final[str] = "GITLAB_BASE_URL"
ENV_GITLAB_GROUP_PATH: Final[str] = "GITLAB_GROUP_PATH"
ENV_GITLAB_ACCESS_TOKEN: Final[str] = "GITLAB_ACCESS_TOKEN"
ENV_GITLAB_VERIFY_SSL: Final[str] = "GITLAB_VERIFY_SSL"
ENV_ANALYSIS_BRANCH: Final[str] = "ANALYSIS_BRANCH"
ENV_REFRESH_INTERVAL: Final[str] = "REFRESH_INTERVAL"
ENV_VISUALIZATION_PORT: Final[str] = "VISUALIZATION_PORT"


# ---------------------------------------------------------------------------
# Configuration key names (the dotted names the design uses for diagnostics)
# ---------------------------------------------------------------------------

KEY_GITLAB_BASE_URL: Final[str] = "gitlab.base_url"
KEY_GITLAB_GROUP_PATH: Final[str] = "gitlab.group_path"
KEY_GITLAB_ACCESS_TOKEN: Final[str] = "gitlab.access_token"
KEY_GITLAB_VERIFY_SSL: Final[str] = "gitlab.verify_ssl"
KEY_ANALYSIS_BRANCH: Final[str] = "analysis.branch"
KEY_REFRESH_INTERVAL: Final[str] = "refresh.interval"
KEY_VISUALIZATION_PORT: Final[str] = "visualization.port"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Config:
    """The validated configuration of one ``MCP_Server`` process.

    All fields are populated by :func:`load_and_validate`. The dataclass is
    immutable (``frozen=True``) so downstream components can rely on its
    values not being mutated after startup; this is the input every other
    component in the design's component diagram receives via the
    ``Config Loader / Validator -> ...`` edges.

    Attributes:
        gitlab_base_url: Validated ``gitlab.base_url`` (e.g.
            ``"https://gitlab.example.com"``). Always an ``http://`` or
            ``https://`` URL with a host (Requirements 1.1, 1.4).
        gitlab_group_path: Non-empty group path or top-level group ID that
            scopes which projects are ingested (Requirements 1.2, 1.4).
        gitlab_access_token: Non-empty GitLab access token used by the
            GitLab_Connector (Requirements 1.3, 1.5).
        gitlab_verify_ssl: Whether the GitLab_Connector validates TLS
            certificates on outbound HTTPS calls to ``gitlab_base_url``.
            Defaults to ``True``. Set ``GITLAB_VERIFY_SSL=false`` to allow
            connections to a GitLab instance fronted by a self-signed
            certificate; doing so disables MITM protection for those
            requests.
        analysis_branch: Branch name used for all analysis. Defaults to
            ``"uat"`` (Requirements 15.1, 15.2, 15.6).
        refresh_interval: Optional periodic-refresh interval. When ``None``
            (the configuration value was unset or empty) no scheduler tick
            is started. When set, it is at least 1 minute (Requirement 8.3).
        visualization_port: Validated TCP port for the Visualization_Server,
            in ``[1, 65535]``. Defaults to ``7345`` (Requirements 12.3,
            12.4, 12.5).
    """

    gitlab_base_url: str
    gitlab_group_path: str
    gitlab_access_token: str
    gitlab_verify_ssl: bool
    analysis_branch: str
    refresh_interval: timedelta | None
    visualization_port: int


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_and_validate(
    env: Mapping[str, str] | None = None,
    *,
    stderr: TextIO | None = None,
) -> Config:
    """Load and validate configuration from ``env``.

    ``env`` defaults to :data:`os.environ`. The optional ``stderr``
    parameter exists so tests and other callers can capture the diagnostic
    line without having to monkey-patch :data:`sys.stderr`.

    Returns:
        A populated :class:`Config` on success.

    Raises:
        SystemExit: When any value fails validation. Before raising, a
            single line is written to stderr naming the offending key and
            the specific reason the value was rejected. This is the shared
            failure-mode implementation referenced by Requirements 1.4,
            1.5, 12.5, and 15.6.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    err_stream: TextIO = sys.stderr if stderr is None else stderr

    try:
        return _build_config(source)
    except ConfigError as exc:
        # The structured ``ConfigError.message`` already has the form
        # ``configuration error for '{key}': {reason}`` and therefore
        # names both the offending key and the specific reason in one
        # line. Print exactly one line and terminate; no further
        # initialization happens, so neither MCP nor the Visualization_Server
        # accepts traffic. (Requirements 1.4, 1.5, 12.5, 15.6.)
        print(exc.message, file=err_stream, flush=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-field validators
# ---------------------------------------------------------------------------


def _build_config(env: Mapping[str, str]) -> Config:
    """Validate every field of ``env`` and return a populated :class:`Config`.

    Validation order matches the *Config Loader / Validator* table in the
    design. The first failure short-circuits with :class:`ConfigError`
    which :func:`load_and_validate` translates to a single stderr line and
    a non-zero exit status.
    """
    base_url = _validate_required_url(
        env, ENV_GITLAB_BASE_URL, KEY_GITLAB_BASE_URL
    )
    group_path = _validate_required_non_empty(
        env, ENV_GITLAB_GROUP_PATH, KEY_GITLAB_GROUP_PATH
    )
    access_token = _validate_required_non_empty(
        env, ENV_GITLAB_ACCESS_TOKEN, KEY_GITLAB_ACCESS_TOKEN
    )
    verify_ssl = _validate_gitlab_verify_ssl(env)
    analysis_branch = _validate_analysis_branch(env)
    refresh_interval = _validate_refresh_interval(env)
    visualization_port = _validate_visualization_port(env)

    return Config(
        gitlab_base_url=base_url,
        gitlab_group_path=group_path,
        gitlab_access_token=access_token,
        gitlab_verify_ssl=verify_ssl,
        analysis_branch=analysis_branch,
        refresh_interval=refresh_interval,
        visualization_port=visualization_port,
    )


def _validate_required_non_empty(
    env: Mapping[str, str], env_var: str, key: str
) -> str:
    """Return ``env[env_var]`` or raise :class:`ConfigError`.

    The variable must be present (Requirements 1.4, 1.5: "missing"
    failure) and contain at least one non-whitespace character ("empty"
    failure). The raw value is returned unmodified so downstream
    consumers (e.g. the GitLab access token) see exactly what the
    operator configured.
    """
    raw = env.get(env_var)
    if raw is None:
        raise ConfigError(key, "is required")
    if raw.strip() == "":
        raise ConfigError(key, "must not be empty")
    return raw


def _validate_required_url(
    env: Mapping[str, str], env_var: str, key: str
) -> str:
    """Validate a required configuration value as an ``http(s)`` URL with a host."""
    raw = _validate_required_non_empty(env, env_var, key)
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https"):
        raise ConfigError(key, "must be an http:// or https:// URL")
    if not parsed.netloc:
        raise ConfigError(key, "must include a host")
    return raw


#: Accepted truthy spellings for ``gitlab.verify_ssl``. Lowercased before
#: matching so the operator-facing value is forgiving (``True``, ``TRUE``,
#: ``true``, ``On``, ``YES`` all parse identically).
_TRUTHY_VERIFY_SSL: Final[frozenset[str]] = frozenset(
    {"true", "1", "yes", "on"}
)

#: Accepted falsy spellings for ``gitlab.verify_ssl``.
_FALSY_VERIFY_SSL: Final[frozenset[str]] = frozenset(
    {"false", "0", "no", "off"}
)


def _validate_gitlab_verify_ssl(env: Mapping[str, str]) -> bool:
    """Validate ``gitlab.verify_ssl``.

    Defaults to :data:`DEFAULT_GITLAB_VERIFY_SSL` (``True``) when the
    variable is unset or empty. Accepts case-insensitive truthy/falsy
    spellings (``true``/``false``, ``1``/``0``, ``yes``/``no``,
    ``on``/``off``). Any other value yields a :class:`ConfigError`
    naming the offending key.
    """
    raw = env.get(ENV_GITLAB_VERIFY_SSL)
    if raw is None or raw.strip() == "":
        return DEFAULT_GITLAB_VERIFY_SSL
    normalized = raw.strip().lower()
    if normalized in _TRUTHY_VERIFY_SSL:
        return True
    if normalized in _FALSY_VERIFY_SSL:
        return False
    raise ConfigError(
        KEY_GITLAB_VERIFY_SSL,
        "must be one of true/false, 1/0, yes/no, on/off",
    )


def _validate_analysis_branch(env: Mapping[str, str]) -> str:
    """Validate ``analysis.branch``.

    When the environment variable is unset, defaults to ``"uat"``
    (Requirement 15.2). When set to the empty string, fails per
    Requirement 15.6.
    """
    raw = env.get(ENV_ANALYSIS_BRANCH)
    if raw is None:
        return DEFAULT_ANALYSIS_BRANCH
    if raw.strip() == "":
        # Requirement 15.6: an empty Analysis_Branch is not allowed.
        raise ConfigError(KEY_ANALYSIS_BRANCH, "must not be empty")
    return raw


def _validate_visualization_port(env: Mapping[str, str]) -> int:
    """Validate ``visualization.port``.

    Defaults to :data:`DEFAULT_VISUALIZATION_PORT` when unset or empty
    (Requirement 12.4). Otherwise the value must parse as a base-10
    integer (Requirement 12.5: "an integer is required") and must lie in
    ``[1, 65535]`` (Requirement 12.5: "the allowed range").
    """
    raw = env.get(ENV_VISUALIZATION_PORT)
    if raw is None or raw.strip() == "":
        return DEFAULT_VISUALIZATION_PORT
    # ``int(raw, 10)`` would happily accept ``+1`` or surrounding
    # whitespace; tighten with an explicit pattern so anything that is
    # not a sign-free decimal integer is rejected as "not an integer".
    if not re.fullmatch(r"-?\d+", raw):
        raise ConfigError(KEY_VISUALIZATION_PORT, "must be an integer")
    port = int(raw)
    if port < MIN_PORT or port > MAX_PORT:
        raise ConfigError(
            KEY_VISUALIZATION_PORT,
            f"must be in the range {MIN_PORT} to {MAX_PORT}",
        )
    return port


def _validate_refresh_interval(env: Mapping[str, str]) -> timedelta | None:
    """Validate ``refresh.interval``.

    When unset or empty, returns ``None`` to indicate "no schedule" (no
    scheduler tick will be started). When set, the value must parse as a
    duration and be at least :data:`MIN_REFRESH_INTERVAL`.
    """
    raw = env.get(ENV_REFRESH_INTERVAL)
    if raw is None or raw.strip() == "":
        return None
    interval = _parse_duration(raw)
    if interval is None:
        raise ConfigError(
            KEY_REFRESH_INTERVAL,
            "must be a duration like '60s', '5m', '1h'",
        )
    if interval < MIN_REFRESH_INTERVAL:
        raise ConfigError(KEY_REFRESH_INTERVAL, "must be at least 1 minute")
    return interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Pattern matching the supported duration syntax: a non-negative integer
#: with an optional unit suffix. Whitespace around the value is tolerated
#: so values pulled from a ``.env`` file are robust to incidental spaces.
_DURATION_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*(?P<value>\d+)\s*(?P<unit>[smhd]?)\s*"
)

#: Multiplier (in seconds) for each supported duration unit. The empty
#: unit is treated as seconds, matching the convention of other Python
#: tooling that accepts a bare integer for "seconds".
_DURATION_UNIT_SECONDS: Final[Mapping[str, int]] = {
    "": 1,
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def _parse_duration(raw: str) -> timedelta | None:
    """Parse a duration string into a :class:`timedelta`.

    Accepts a non-negative integer with an optional unit suffix from
    ``{s, m, h, d}``. A bare integer is treated as seconds. Returns
    ``None`` when the input does not match the supported syntax so the
    caller can produce a key-aware :class:`ConfigError`.
    """
    match = _DURATION_RE.fullmatch(raw)
    if match is None:
        return None
    value = int(match.group("value"))
    unit = match.group("unit")
    return timedelta(seconds=value * _DURATION_UNIT_SECONDS[unit])


__all__ = [
    "DEFAULT_ANALYSIS_BRANCH",
    "DEFAULT_GITLAB_VERIFY_SSL",
    "DEFAULT_VISUALIZATION_PORT",
    "ENV_ANALYSIS_BRANCH",
    "ENV_GITLAB_ACCESS_TOKEN",
    "ENV_GITLAB_BASE_URL",
    "ENV_GITLAB_GROUP_PATH",
    "ENV_GITLAB_VERIFY_SSL",
    "ENV_REFRESH_INTERVAL",
    "ENV_VISUALIZATION_PORT",
    "KEY_ANALYSIS_BRANCH",
    "KEY_GITLAB_ACCESS_TOKEN",
    "KEY_GITLAB_BASE_URL",
    "KEY_GITLAB_GROUP_PATH",
    "KEY_GITLAB_VERIFY_SSL",
    "KEY_REFRESH_INTERVAL",
    "KEY_VISUALIZATION_PORT",
    "MAX_PORT",
    "MIN_PORT",
    "MIN_REFRESH_INTERVAL",
    "Config",
    "load_and_validate",
]

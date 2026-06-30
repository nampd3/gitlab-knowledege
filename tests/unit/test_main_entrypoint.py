"""Unit tests for the ``main`` entry point's start-up sequencing.

Task 11.1 wires the documented start-up sequence (config → store →
in-process collaborators → visualization bind → MCP construction →
scheduler) and consolidates every start-up failure onto the same
"single line to stderr; non-zero exit" path that
:func:`config.load_and_validate` (task 2.3) and
:func:`visualization_server.bind_or_exit` (task 10.1) already use.
These tests pin the two failure paths the entry point itself owns:

* An invalid configuration causes :func:`main` to terminate non-zero
  via :func:`load_and_validate` *before* either surface starts. The
  documented ``configuration error for '{key}': {reason}`` line is
  written to stderr and no socket bind, MCP construction, or
  scheduler start happens.
* A failure to open the ``Knowledge_Store`` causes :func:`main` to
  emit a ``"startup error: ..."`` line and exit non-zero before
  either surface accepts traffic, satisfying Requirement 7.2's "load
  previously persisted profiles before serving queries" implication
  that the store must be openable to serve any request.

Implements Requirements 1.4, 1.5, 7.2, 12.5, 15.6 (start-up
termination path).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from project_knowledge_mcp import main as main_module
from project_knowledge_mcp.main import (
    DEFAULT_KNOWLEDGE_STORE_PATH,
    ENV_KNOWLEDGE_STORE_PATH,
    _resolve_store_path,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_env(tmp_path: Path) -> dict[str, str]:
    """Return a fully populated, valid configuration environment.

    Every key the configuration validator requires is set, so
    :func:`load_and_validate` cannot reject the env. Tests that want
    to break the configuration mutate one key after calling this
    helper.
    """
    return {
        "GITLAB_BASE_URL": "https://gitlab.example.com",
        "GITLAB_GROUP_PATH": "platform",
        "GITLAB_ACCESS_TOKEN": "glpat-deadbeef",
        "ANALYSIS_BRANCH": "uat",
        "VISUALIZATION_PORT": "7345",
        ENV_KNOWLEDGE_STORE_PATH: str(tmp_path / "ks.db"),
    }


# ---------------------------------------------------------------------------
# _resolve_store_path
# ---------------------------------------------------------------------------


def test_resolve_store_path_defaults_when_env_unset() -> None:
    """``_resolve_store_path`` returns the documented default path.

    When :data:`ENV_KNOWLEDGE_STORE_PATH` is not present in ``env``
    the helper returns :data:`DEFAULT_KNOWLEDGE_STORE_PATH` so a
    fresh deployment has a deterministic location for the SQLite
    database without having to set any environment variables.
    """
    assert _resolve_store_path({}) == DEFAULT_KNOWLEDGE_STORE_PATH


def test_resolve_store_path_uses_env_override(tmp_path: Path) -> None:
    """An explicit ``KNOWLEDGE_STORE_PATH`` overrides the default."""
    target = tmp_path / "alt.db"
    env = {ENV_KNOWLEDGE_STORE_PATH: str(target)}
    assert _resolve_store_path(env) == target


def test_resolve_store_path_treats_blank_env_as_unset(tmp_path: Path) -> None:
    """Whitespace-only values fall back to the default path.

    Robust to incidental whitespace from operators copy-pasting an
    empty value into a ``.env`` file.
    """
    env = {ENV_KNOWLEDGE_STORE_PATH: "   "}
    assert _resolve_store_path(env) == DEFAULT_KNOWLEDGE_STORE_PATH


# ---------------------------------------------------------------------------
# main: configuration-error termination path
# ---------------------------------------------------------------------------


def test_main_exits_non_zero_on_missing_required_config(tmp_path: Path) -> None:
    """A missing required key causes ``main`` to exit non-zero.

    The validator emits the documented
    ``configuration error for '{key}': {reason}`` line on stderr and
    calls :func:`sys.exit` with a non-zero status. ``main`` MUST
    propagate that termination — it MUST NOT continue to open the
    ``Knowledge_Store``, bind the Visualization_Server's sockets, or
    construct the MCP transport when the configuration is invalid
    (Requirements 1.4, 1.5).
    """
    env = _valid_env(tmp_path)
    # Knock out a required key so validation fails on the first check.
    del env["GITLAB_BASE_URL"]
    captured = io.StringIO()

    with pytest.raises(SystemExit) as exit_info:
        main(env=env, stderr=captured)

    assert exit_info.value.code is not None
    assert exit_info.value.code != 0
    # The validator's diagnostic line names the offending key. We do
    # not pin the exact wording here — that is the job of the task
    # 2.3 tests — but we do require that the line names the missing
    # key so the operator can identify which value to fix.
    assert "gitlab.base_url" in captured.getvalue()


def test_main_does_not_open_store_when_config_is_invalid(
    tmp_path: Path,
) -> None:
    """Config-error path terminates *before* the store is opened.

    The design consolidates start-up termination on a single path
    (Requirements 1.4, 1.5, 12.5, 15.6) and explicitly requires that
    no surface accepts traffic when start-up fails. The
    ``Knowledge_Store`` is the first piece of state that gets a
    handle to a real OS resource (a SQLite file), so we use it as a
    proxy for "did the wiring continue past the configuration
    check?".
    """
    env = _valid_env(tmp_path)
    del env["GITLAB_ACCESS_TOKEN"]

    with (
        patch.object(main_module.KnowledgeStore, "open") as open_mock,
        pytest.raises(SystemExit),
    ):
        main(env=env, stderr=io.StringIO())

    open_mock.assert_not_called()


# ---------------------------------------------------------------------------
# main: Knowledge_Store-open termination path
# ---------------------------------------------------------------------------


def test_main_exits_non_zero_when_store_open_fails(tmp_path: Path) -> None:
    """A failure to open the store terminates the process non-zero.

    The documented ``"startup error: ..."`` line names the
    ``Knowledge_Store`` and the underlying failure so an operator can
    diagnose the cause without spelunking through stack traces. The
    exit must happen before either surface accepts traffic; we use
    the absence of any visualization-server bind as a proxy for that
    via :func:`patch.object` on
    :func:`project_knowledge_mcp.main.bind_or_exit`.
    """
    env = _valid_env(tmp_path)
    captured = io.StringIO()

    failure = OSError("disk is full")

    def _raising_open(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    with (
        patch.object(main_module.KnowledgeStore, "open", new=_raising_open),
        patch.object(main_module, "bind_or_exit") as bind_mock,
        pytest.raises(SystemExit) as exit_info,
    ):
        main(env=env, stderr=captured)

    assert exit_info.value.code is not None
    assert exit_info.value.code != 0
    # The documented line starts with ``"startup error: "`` (matching
    # the convention used by :mod:`visualization_server`) and names
    # both the path and the underlying failure reason.
    line = captured.getvalue()
    assert line.startswith("startup error: ")
    assert "Knowledge_Store" in line
    assert "disk is full" in line
    # The Visualization_Server must not have started accepting
    # traffic — its bind helper is never invoked when the store
    # cannot be opened.
    bind_mock.assert_not_called()

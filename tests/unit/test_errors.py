"""Unit tests for the structured error type hierarchy in ``errors.py``.

Each test confirms that the error class:

* exposes its constructor arguments as plain attributes,
* produces the documented deterministic message via ``str(err)`` and the
  equivalent ``err.message`` attribute, and
* is a subclass of the shared ``ProjectKnowledgeError`` base.

These tests exercise the data-carrier contract that both the MCP layer and
the Visualization_Server rely on when surfacing failures.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.errors import (
    BindError,
    ConfigError,
    GitLabAuthError,
    GitLabGroupNotFoundError,
    IngestionInProgressError,
    KnowledgeStoreUnavailableError,
    ProjectKnowledgeError,
    ProjectNotInScopeError,
)

pytestmark = pytest.mark.unit


def test_all_errors_subclass_project_knowledge_error() -> None:
    for cls in (
        KnowledgeStoreUnavailableError,
        GitLabAuthError,
        GitLabGroupNotFoundError,
        IngestionInProgressError,
        ProjectNotInScopeError,
        ConfigError,
        BindError,
    ):
        assert issubclass(cls, ProjectKnowledgeError)
        assert issubclass(cls, Exception)


def test_knowledge_store_unavailable_default_message() -> None:
    err = KnowledgeStoreUnavailableError()
    assert err.reason is None
    assert err.message == "Knowledge_Store unavailable"
    assert str(err) == "Knowledge_Store unavailable"


def test_knowledge_store_unavailable_with_reason() -> None:
    err = KnowledgeStoreUnavailableError("disk I/O error")
    assert err.reason == "disk I/O error"
    assert err.message == "Knowledge_Store unavailable: disk I/O error"
    assert str(err) == "Knowledge_Store unavailable: disk I/O error"


def test_gitlab_auth_error_carries_status_code() -> None:
    err = GitLabAuthError(401)
    assert err.status_code == 401
    assert err.message == "GitLab authentication failed (HTTP 401)"
    assert str(err) == "GitLab authentication failed (HTTP 401)"


def test_gitlab_auth_error_with_403() -> None:
    err = GitLabAuthError(403)
    assert err.status_code == 403
    assert err.message == "GitLab authentication failed (HTTP 403)"


def test_gitlab_group_not_found_carries_group_path() -> None:
    err = GitLabGroupNotFoundError("acme/platform")
    assert err.group_path == "acme/platform"
    assert err.message == "GitLab group 'acme/platform' not found"
    assert str(err) == "GitLab group 'acme/platform' not found"


def test_ingestion_in_progress_fixed_message() -> None:
    err = IngestionInProgressError()
    # Wording is fixed by Requirement 8.6 so MCP and scheduler agree.
    assert err.message == "Ingestion_Job is already in progress"
    assert str(err) == "Ingestion_Job is already in progress"
    assert IngestionInProgressError.MESSAGE == "Ingestion_Job is already in progress"


def test_project_not_in_scope_carries_project_id() -> None:
    err = ProjectNotInScopeError(42)
    assert err.gitlab_project_id == 42
    assert err.message == "Project 42 is not in scope"
    assert str(err) == "Project 42 is not in scope"


def test_config_error_carries_key_and_reason() -> None:
    err = ConfigError("gitlab.base_url", "is required")
    assert err.key == "gitlab.base_url"
    assert err.reason == "is required"
    assert err.message == "configuration error for 'gitlab.base_url': is required"
    assert str(err) == "configuration error for 'gitlab.base_url': is required"


def test_config_error_message_includes_key_for_each_documented_rule() -> None:
    # The structured payload is shared across Requirements 1.4, 1.5, 12.5, 15.6;
    # confirm a representative reason from each surfaces the offending key.
    cases = [
        ("gitlab.access_token", "is required"),
        ("analysis.branch", "must not be empty"),
        ("visualization.port", "must be an integer"),
        ("visualization.port", "must be in the range 1 to 65535"),
    ]
    for key, reason in cases:
        err = ConfigError(key, reason)
        assert err.key == key
        assert err.reason == reason
        assert key in err.message
        assert reason in err.message
        assert err.message == f"configuration error for '{key}': {reason}"


def test_bind_error_carries_port_and_reason() -> None:
    err = BindError(7345, "address already in use")
    assert err.port == 7345
    assert err.reason == "address already in use"
    assert err.message == "failed to bind to port 7345: address already in use"
    assert str(err) == "failed to bind to port 7345: address already in use"


def test_bind_error_with_os_error_reason() -> None:
    err = BindError(7345, "permission denied")
    assert err.port == 7345
    assert err.reason == "permission denied"
    assert err.message == "failed to bind to port 7345: permission denied"


def test_errors_can_be_caught_by_base_class() -> None:
    # Catching ``ProjectKnowledgeError`` must catch every concrete subclass.
    raisers: list[ProjectKnowledgeError] = [
        KnowledgeStoreUnavailableError(),
        GitLabAuthError(401),
        GitLabGroupNotFoundError("group/path"),
        IngestionInProgressError(),
        ProjectNotInScopeError(7),
        ConfigError("k", "r"),
        BindError(7345, "address already in use"),
    ]
    for raised in raisers:
        with pytest.raises(ProjectKnowledgeError) as excinfo:
            raise raised
        assert excinfo.value is raised

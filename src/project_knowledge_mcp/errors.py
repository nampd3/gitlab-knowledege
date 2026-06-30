"""Structured error type hierarchy used by both MCP and visualization surfaces.

Every error in this module exposes its meaningful fields as plain attributes
and produces a deterministic human-readable message via ``str(error)`` (and
the equivalent ``error.message`` attribute). Both the MCP transport layer
(``mcp_server``) and the ``visualization_server`` use these errors to render
their responses, so the message wording here is stable, self-contained, and
suitable for surfacing through MCP tool result messages, MCP error responses,
and HTML error pages alike.

Each message names the offending field (status code, group path, project id,
config key, port, ...) so callers can quote the message verbatim into MCP
``isError`` tool results and visualization HTML without re-formatting.

Implements Requirements 2.3, 2.4, 8.6, 10.7, 11.5, 11.6, 11.7, 12.6, 12.8, 14.6.
"""

from __future__ import annotations


class ProjectKnowledgeError(Exception):
    """Common base class for every error emitted by this package.

    Subclasses set the structured fields that constitute the error payload
    and pass a deterministic formatted ``message`` to ``__init__``. The
    message is stored both as the standard ``Exception`` argument (so
    ``str(err)`` works) and on a dedicated ``message`` attribute for callers
    that want to forward it without re-formatting.
    """

    message: str

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class KnowledgeStoreUnavailableError(ProjectKnowledgeError):
    """Knowledge_Store reads cannot be served from underlying storage.

    Raised by the reader interface when the SQLite-backed store cannot be
    queried (file locked, disk error, schema corruption, etc.). There is no
    in-memory fallback. The MCP layer translates this into a tool result
    with ``isError: true`` and a message of the form
    ``"tool execution failed: Knowledge_Store unavailable"``
    (Requirement 11.7); the Visualization_Server translates it into an
    HTTP 503 response with an HTML body stating that project knowledge is
    temporarily unavailable (Requirement 14.6).

    An optional ``reason`` may be supplied to add diagnostic detail; if
    omitted, the bare canonical phrase is used.
    """

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        if reason:
            super().__init__(f"Knowledge_Store unavailable: {reason}")
        else:
            super().__init__("Knowledge_Store unavailable")


class GitLabAuthError(ProjectKnowledgeError):
    """GitLab returned an authentication failure (HTTP 401 or 403).

    Carries the GitLab response ``status_code`` so the Ingestion_Coordinator
    can both abort the in-progress Ingestion_Job and report the auth failure
    together with the status code in a single try/finally block
    (Requirement 2.3).
    """

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"GitLab authentication failed (HTTP {status_code})")


class GitLabGroupNotFoundError(ProjectKnowledgeError):
    """The configured GitLab group returned HTTP 404.

    Carries the configured ``group_path`` so the surfaced message names the
    group that could not be resolved (Requirement 2.4).
    """

    def __init__(self, group_path: str) -> None:
        self.group_path = group_path
        super().__init__(f"GitLab group '{group_path}' not found")


class IngestionInProgressError(ProjectKnowledgeError):
    """A refresh was requested while another Ingestion_Job is already running.

    The wording of the message is fixed by Requirement 8.6 (and referenced
    by the design's Property 12 documentation) so that the MCP ``tools/call``
    rejection and the scheduler's log line emit identical text. The
    Ingestion_Coordinator state and the Knowledge_Store remain unchanged
    when this error is raised.
    """

    MESSAGE = "Ingestion_Job is already in progress"

    def __init__(self) -> None:
        super().__init__(self.MESSAGE)


class ProjectNotInScopeError(ProjectKnowledgeError):
    """A tool call references a GitLab project ID that is not in scope.

    Carries the offending ``gitlab_project_id`` so the MCP error message can
    name the specific project. The MCP layer surfaces this as a tool result
    with ``isError: true`` and a message of the form
    ``"Project {gitlab_project_id} is not in scope"`` (Requirement 10.7);
    the Visualization_Server uses the same wording in its 404 HTML body
    (Requirement 14.5).
    """

    def __init__(self, gitlab_project_id: int) -> None:
        self.gitlab_project_id = gitlab_project_id
        super().__init__(f"Project {gitlab_project_id} is not in scope")


class ConfigError(ProjectKnowledgeError):
    """The configuration loader rejected a value.

    Carries the configuration ``key`` whose value failed validation and a
    free-form ``reason`` describing the validation rule that failed (e.g.
    ``"is required"``, ``"must not be empty"``, ``"must be an integer"``,
    ``"must be in the range 1 to 65535"``). The structured payload is
    shared across Requirements 1.4, 1.5, 12.5, and 15.6 (single
    failure-mode implementation in ``config.load_and_validate``).
    """

    def __init__(self, key: str, reason: str) -> None:
        self.key = key
        self.reason = reason
        super().__init__(f"configuration error for '{key}': {reason}")


class BindError(ProjectKnowledgeError):
    """The Visualization_Server could not bind to the configured TCP port.

    Carries the offending ``port`` and a free-form ``reason`` describing the
    underlying failure (e.g. ``"address already in use"`` for
    Requirement 12.6 or an OS error string for Requirement 12.8).
    """

    def __init__(self, port: int, reason: str) -> None:
        self.port = port
        self.reason = reason
        super().__init__(f"failed to bind to port {port}: {reason}")


__all__ = [
    "BindError",
    "ConfigError",
    "GitLabAuthError",
    "GitLabGroupNotFoundError",
    "IngestionInProgressError",
    "KnowledgeStoreUnavailableError",
    "ProjectKnowledgeError",
    "ProjectNotInScopeError",
]

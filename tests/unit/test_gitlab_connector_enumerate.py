"""Unit tests for ``GitLabConnector.enumerate_projects`` (task 5.2).

These tests drive the connector against an :class:`httpx.MockTransport`
fake so the request flow -- pagination of the group projects endpoint
plus a per-project ``Analysis_Branch`` lookup -- can be inspected without
any network I/O. Tests cover:

* the happy path (one project, branch present),
* the branch-missing path (HTTP 404 on the branch endpoint sets
  ``branch_missing=True`` and ``analysis_branch_commit_sha=None``),
* HTTP 401/403 on the group listing -> :class:`GitLabAuthError`,
* HTTP 404 on the configured group -> :class:`GitLabGroupNotFoundError`,
* HTTP 401/403 on a per-project branch lookup -> :class:`GitLabAuthError`,
* multi-page enumeration via the GitLab ``Link`` header,
* the connector wires ``include_subgroups=true`` and uses the configured
  ``Analysis_Branch`` when looking up the commit SHA, and
* the empty-group case (no projects -> empty iterator).

Real GitLab always emits an RFC 5988 ``Link`` header on its paginated
list endpoints, even when the response is the last (or only) page. The
fakes in this module mirror that behavior by always attaching a ``Link``
header on projects-list responses; absence of a ``rel="next"`` entry is
the canonical end-of-pagination signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from project_knowledge_mcp.errors import GitLabAuthError, GitLabGroupNotFoundError
from project_knowledge_mcp.gitlab_connector import GitLabConnector

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit


# Fixed values used across every test: the connector's identity does not
# influence the behavior under test, but pinning these makes the assertions
# self-documenting.
BASE_URL = "https://gitlab.example.com"
ACCESS_TOKEN = "test-token"
GROUP_PATH = "acme/platform"
ANALYSIS_BRANCH = "uat"
QUOTED_GROUP = "acme%2Fplatform"
GROUP_PROJECTS_PATH = f"/api/v4/groups/{QUOTED_GROUP}/projects"

#: Canonical "this is the last (or only) page" ``Link`` header. The
#: ``rel="first"`` entry signals to the connector that GitLab speaks
#: Link-based pagination (so the ``?page=N`` fallback is skipped); the
#: absence of ``rel="next"`` terminates the iterator on this page.
LAST_PAGE_LINK = (
    f'<{BASE_URL}{GROUP_PROJECTS_PATH}?page=1&per_page=100>; rel="first"'
)


def _request_path(request: httpx.Request) -> str:
    """Return the request's URL path with percent-encoding preserved.

    ``httpx.URL.path`` decodes ``%2F`` back to ``/`` before returning, so
    matching against an expected encoded path requires the raw bytes form.
    """

    return request.url.raw_path.decode("ascii").split("?", 1)[0]


def _projects_response(
    items: list[dict[str, Any]],
    *,
    next_url: str | None = None,
) -> httpx.Response:
    """Build a GitLab-style projects-list response.

    Attaches a ``Link`` header in every response so the connector takes the
    Link-pagination branch and never falls back to ``?page=N`` probing.
    When ``next_url`` is set the header carries a ``rel="next"`` entry
    pointing at it; otherwise only ``rel="first"`` is included, signalling
    end-of-pagination.
    """

    if next_url is None:
        link = LAST_PAGE_LINK
    else:
        link = (
            f'<{next_url}>; rel="next", '
            f'<{BASE_URL}{GROUP_PROJECTS_PATH}?page=1&per_page=100>; rel="first"'
        )
    return httpx.Response(200, json=items, headers={"Link": link})


def _make_connector(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    group_path: str = GROUP_PATH,
    analysis_branch: str = ANALYSIS_BRANCH,
) -> GitLabConnector:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return GitLabConnector(
        base_url=BASE_URL,
        access_token=ACCESS_TOKEN,
        group_path=group_path,
        analysis_branch=analysis_branch,
        client=client,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_enumerate_projects_yields_branch_sha_when_branch_present() -> None:
    branch_path = "/api/v4/projects/1/repository/branches/uat"

    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == GROUP_PROJECTS_PATH:
            assert request.url.params.get("include_subgroups") == "true"
            assert request.url.params.get("order_by") == "id"
            assert request.url.params.get("sort") == "asc"
            assert request.headers["PRIVATE-TOKEN"] == ACCESS_TOKEN
            return _projects_response(
                [
                    {
                        "id": 1,
                        "path_with_namespace": "acme/platform/svc-a",
                        "description": "Service A",
                    }
                ]
            )
        if path == branch_path:
            assert request.headers["PRIVATE-TOKEN"] == ACCESS_TOKEN
            return httpx.Response(
                200,
                json={"name": "uat", "commit": {"id": "abc123def456"}},
            )
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler) as connector:
        projects = list(connector.enumerate_projects())

    assert len(projects) == 1
    project = projects[0]
    assert project.gitlab_project_id == 1
    assert project.full_path == "acme/platform/svc-a"
    assert project.analysis_branch_name == "uat"
    assert project.analysis_branch_commit_sha == "abc123def456"
    assert project.branch_missing is False
    assert project.repository_description == "Service A"


def test_enumerate_projects_empty_group_yields_no_projects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == GROUP_PROJECTS_PATH:
            return _projects_response([])
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler) as connector:
        projects = list(connector.enumerate_projects())

    assert projects == []


# ---------------------------------------------------------------------------
# Branch-missing handling (Requirement 15.5)
# ---------------------------------------------------------------------------


def test_enumerate_projects_branch_404_sets_branch_missing_flag() -> None:
    branch_path = "/api/v4/projects/7/repository/branches/uat"

    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == GROUP_PROJECTS_PATH:
            return _projects_response(
                [
                    {
                        "id": 7,
                        "path_with_namespace": "acme/platform/legacy",
                        "description": None,
                    }
                ]
            )
        if path == branch_path:
            return httpx.Response(404, json={"message": "404 Branch Not Found"})
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler) as connector:
        projects = list(connector.enumerate_projects())

    assert len(projects) == 1
    project = projects[0]
    assert project.gitlab_project_id == 7
    assert project.branch_missing is True
    assert project.analysis_branch_commit_sha is None
    assert project.analysis_branch_name == "uat"
    assert project.repository_description is None


# ---------------------------------------------------------------------------
# Authentication failure mapping (Requirement 2.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_enumerate_projects_auth_failure_on_group_listing(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status, json={"message": f"{status} Unauthorized"})

    with _make_connector(handler) as connector, pytest.raises(GitLabAuthError) as excinfo:
        list(connector.enumerate_projects())

    assert excinfo.value.status_code == status


@pytest.mark.parametrize("status", [401, 403])
def test_enumerate_projects_auth_failure_on_branch_lookup(status: int) -> None:
    branch_path = "/api/v4/projects/1/repository/branches/uat"

    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == GROUP_PROJECTS_PATH:
            return _projects_response(
                [
                    {
                        "id": 1,
                        "path_with_namespace": "acme/platform/svc-a",
                        "description": None,
                    }
                ]
            )
        if path == branch_path:
            return httpx.Response(status, json={"message": f"{status} Forbidden"})
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler) as connector, pytest.raises(GitLabAuthError) as excinfo:
        list(connector.enumerate_projects())

    assert excinfo.value.status_code == status


# ---------------------------------------------------------------------------
# Group-not-found mapping (Requirement 2.4)
# ---------------------------------------------------------------------------


def test_enumerate_projects_group_404_raises_group_not_found_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(404, json={"message": "404 Group Not Found"})

    with (
        _make_connector(handler) as connector,
        pytest.raises(GitLabGroupNotFoundError) as excinfo,
    ):
        list(connector.enumerate_projects())

    assert excinfo.value.group_path == GROUP_PATH


# ---------------------------------------------------------------------------
# Pagination across the group-projects endpoint (Requirement 2.5)
# ---------------------------------------------------------------------------


def test_enumerate_projects_paginates_via_link_header() -> None:
    page2_url = f"{BASE_URL}{GROUP_PROJECTS_PATH}?page=2&per_page=100"

    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == GROUP_PROJECTS_PATH:
            page = request.url.params.get("page")
            if page is None:
                return _projects_response(
                    [
                        {
                            "id": 1,
                            "path_with_namespace": "acme/platform/a",
                            "description": None,
                        }
                    ],
                    next_url=page2_url,
                )
            if page == "2":
                return _projects_response(
                    [
                        {
                            "id": 2,
                            "path_with_namespace": "acme/platform/b",
                            "description": None,
                        }
                    ]
                )
        if path.startswith("/api/v4/projects/") and path.endswith("/repository/branches/uat"):
            project_id = int(path.split("/")[-4])
            return httpx.Response(
                200,
                json={"name": "uat", "commit": {"id": f"sha-{project_id}"}},
            )
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler) as connector:
        projects = list(connector.enumerate_projects())

    assert [p.gitlab_project_id for p in projects] == [1, 2]
    assert projects[0].full_path == "acme/platform/a"
    assert projects[1].full_path == "acme/platform/b"
    assert projects[0].analysis_branch_commit_sha == "sha-1"
    assert projects[1].analysis_branch_commit_sha == "sha-2"


# ---------------------------------------------------------------------------
# Configuration wiring
# ---------------------------------------------------------------------------


def test_enumerate_projects_uses_configured_analysis_branch() -> None:
    """The branch-lookup URL contains the configured Analysis_Branch (Requirement 15.4)."""

    seen_branch_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == GROUP_PROJECTS_PATH:
            return _projects_response(
                [
                    {
                        "id": 42,
                        "path_with_namespace": "acme/platform/svc",
                        "description": None,
                    }
                ]
            )
        if path.startswith("/api/v4/projects/42/repository/branches/"):
            seen_branch_paths.append(path)
            return httpx.Response(
                200,
                json={"name": "release", "commit": {"id": "deadbeef"}},
            )
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler, analysis_branch="release") as connector:
        projects = list(connector.enumerate_projects())

    assert len(projects) == 1
    assert projects[0].analysis_branch_name == "release"
    assert projects[0].analysis_branch_commit_sha == "deadbeef"
    assert seen_branch_paths == ["/api/v4/projects/42/repository/branches/release"]


def test_enumerate_projects_requires_group_and_analysis_branch() -> None:
    """A connector built without group_path / analysis_branch cannot enumerate."""

    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - never invoked
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    connector = GitLabConnector(base_url=BASE_URL, access_token=ACCESS_TOKEN, client=client)

    with connector, pytest.raises(RuntimeError, match="group_path"):
        list(connector.enumerate_projects())


def test_enumerate_projects_url_encodes_nested_group_path() -> None:
    """Nested group paths are URL-encoded as a single ``%2F``-separated segment."""

    nested_group = "acme/platform/sub-tier"
    encoded = "acme%2Fplatform%2Fsub-tier"
    expected_path = f"/api/v4/groups/{encoded}/projects"
    seen_listing_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == expected_path:
            seen_listing_paths.append(path)
            return _projects_response([])
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler, group_path=nested_group) as connector:
        projects = list(connector.enumerate_projects())

    assert projects == []
    assert seen_listing_paths == [expected_path]


def test_enumerate_projects_passes_per_page_and_ordering_params() -> None:
    """The first page request carries ``per_page=100``, ``order_by=id``, ``sort=asc``."""

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = _request_path(request)
        if path == GROUP_PROJECTS_PATH:
            captured["params"] = dict(request.url.params)
            return _projects_response([])
        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    with _make_connector(handler) as connector:
        list(connector.enumerate_projects())

    assert captured["params"]["per_page"] == "100"
    assert captured["params"]["order_by"] == "id"
    assert captured["params"]["sort"] == "asc"
    assert captured["params"]["include_subgroups"] == "true"

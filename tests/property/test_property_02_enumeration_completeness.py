# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 2: For all group trees (any nesting depth, any number of subgroups, any per-page count from a paginated GitLab API), the GitLab_Connector.enumerate_projects() result SHALL equal exactly the set of projects that are descendants of the configured group, with no duplicates and no omissions.
"""Property test for enumeration covering every descendant project.

**Validates Requirements 2.1, 2.5** (Property 2 in the design).

For every randomly generated GitLab group tree -- any nesting depth, any
number of subgroups per node, any number of projects per node, and any
per-page count chosen by the GitLab API fake -- the connector's
``enumerate_projects()`` result SHALL equal the set of projects that are
descendants of the configured group, with no duplicates and no omissions.

The fake honors GitLab's contract for ``include_subgroups=true``: a single
``GET /api/v4/groups/{group}/projects?include_subgroups=true`` returns
every descendant project (across one or more pages joined by RFC 5988
``Link: rel="next"`` headers). The fake's per-page chunk size is generated
independently of the connector's requested ``per_page`` so the connector's
pagination machinery is exercised regardless of what the server chose for
the page size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.gitlab_connector import GitLabConnector

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Fixed test fixture values
# ---------------------------------------------------------------------------

BASE_URL = "https://gitlab.example.com"
ACCESS_TOKEN = "test-token"
ANALYSIS_BRANCH = "uat"
ROOT_GROUP_PATH = "root"


# ---------------------------------------------------------------------------
# Synthetic group-tree model
# ---------------------------------------------------------------------------

# Path-segment alphabet: GitLab path segments are made of unreserved URL
# characters; lowercase alphanum + hyphen is well within that set and keeps
# generated examples readable when Hypothesis prints a counterexample.
_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=8,
).filter(lambda s: not s.startswith("-") and not s.endswith("-"))


@dataclass(frozen=True)
class _GroupNode:
    """One group in the synthetic tree (basenames only -- paths are derived)."""

    name: str
    projects: tuple[str, ...]
    subgroups: tuple[_GroupNode, ...]


@dataclass(frozen=True)
class _FakeProject:
    """One descendant project, with the identity GitLab would return."""

    project_id: int
    full_path: str


def _node_strategy(max_depth: int) -> st.SearchStrategy[_GroupNode]:
    """Generate a group subtree of depth at most ``max_depth``.

    Project basenames within a single group are unique, so every project's
    ``full_path`` is unique among its siblings. Subgroup names within a
    parent are likewise unique (``unique_by=name``), so paths assembled by
    joining a parent path with a subgroup name never collide between
    siblings. Together these two rules guarantee that every descendant
    project ends up at a unique ``full_path``, which makes the property's
    set-equality assertion meaningful.
    """
    if max_depth == 0:
        return st.builds(
            _GroupNode,
            name=_NAME,
            projects=st.lists(_NAME, min_size=0, max_size=2, unique=True).map(tuple),
            subgroups=st.just(()),
        )
    return st.builds(
        _GroupNode,
        name=_NAME,
        projects=st.lists(_NAME, min_size=0, max_size=2, unique=True).map(tuple),
        subgroups=st.lists(
            _node_strategy(max_depth - 1),
            min_size=0,
            max_size=2,
            unique_by=lambda n: n.name,
        ).map(tuple),
    )


def _flatten_descendants(root: _GroupNode, root_path: str) -> list[_FakeProject]:
    """Return every descendant project of ``root``, with sequential IDs.

    Walks the tree depth-first. Project IDs are sequential starting at 1
    so they are unique by construction. Project paths are unique by the
    construction rules in :func:`_node_strategy`; the ``seen_paths`` guard
    is defensive against any future relaxation of those rules.
    """
    out: list[_FakeProject] = []
    seen_paths: set[str] = set()
    next_id = 1

    def visit(node: _GroupNode, path: str) -> None:
        nonlocal next_id
        for proj_name in node.projects:
            full_path = f"{path}/{proj_name}"
            if full_path in seen_paths:
                continue
            seen_paths.add(full_path)
            out.append(_FakeProject(project_id=next_id, full_path=full_path))
            next_id += 1
        for sub in node.subgroups:
            visit(sub, f"{path}/{sub.name}")

    visit(root, root_path)
    return out


# ---------------------------------------------------------------------------
# Fake GitLab REST API (``httpx.MockTransport`` handler factory)
# ---------------------------------------------------------------------------


def _build_handler(
    descendants: list[_FakeProject],
    *,
    page_size: int,
    group_path: str,
    branch: str,
) -> Callable[[httpx.Request], httpx.Response]:
    """Build an :class:`httpx.MockTransport` handler that scripts the GitLab API.

    The handler answers two endpoint families:

    * ``GET /api/v4/groups/{group}/projects`` -- returns ``descendants``
      paginated at ``page_size`` items per page, joined by RFC 5988
      ``Link: rel="next"`` headers. The first page is the canonical
      ``include_subgroups=true`` response that GitLab would return when
      asked to enumerate every descendant of the configured group.
    * ``GET /api/v4/projects/{id}/repository/branches/{branch}`` -- returns
      a synthetic commit SHA for ``branch``. The connector calls this for
      every project; the property under test does not constrain the SHA
      value, but a successful response is required so the connector does
      not abort enumeration.

    The handler returns HTTP 500 on any other path so unexpected calls
    produce a loud failure during shrinking.
    """
    quoted_group = quote(group_path, safe="")
    listing_path = f"/api/v4/groups/{quoted_group}/projects"

    def handler(request: httpx.Request) -> httpx.Response:
        # ``raw_path`` preserves percent-encoding; ``url.path`` would decode
        # ``%2F`` back to ``/`` and break the listing-path comparison.
        path = request.url.raw_path.decode("ascii").split("?", 1)[0]
        params = request.url.params

        if path == listing_path:
            page_str = params.get("page")
            page = int(page_str) if page_str is not None else 1
            start = (page - 1) * page_size
            end = start + page_size
            chunk = descendants[start:end]
            items: list[dict[str, Any]] = [
                {
                    "id": project.project_id,
                    "path_with_namespace": project.full_path,
                    "description": None,
                }
                for project in chunk
            ]
            link_parts = [
                f'<{BASE_URL}{listing_path}?page=1&per_page={page_size}>; rel="first"'
            ]
            if end < len(descendants):
                link_parts.append(
                    f"<{BASE_URL}{listing_path}"
                    f'?page={page + 1}&per_page={page_size}>; rel="next"'
                )
            return httpx.Response(
                200,
                json=items,
                headers={"Link": ", ".join(link_parts)},
            )

        # Per-project Analysis_Branch lookup. The exact branch segment
        # never matters here because the connector URL-encodes it and the
        # fake matches on the path prefix; the property does not constrain
        # what the SHA looks like.
        if path.startswith("/api/v4/projects/") and "/repository/branches/" in path:
            project_id_segment = path.split("/")[4]
            return httpx.Response(
                200,
                json={
                    "name": branch,
                    "commit": {"id": f"sha-{project_id_segment}"},
                },
            )

        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    return handler


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(
    tree=_node_strategy(max_depth=3),
    page_size=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=100)
def test_enumerate_projects_equals_set_of_descendants(
    tree: _GroupNode,
    page_size: int,
) -> None:
    """``enumerate_projects()`` returns exactly the descendant project set.

    The ground-truth set is computed directly from the generated tree.
    The connector talks to a scripted ``httpx.MockTransport`` fake whose
    page size is independent of the connector's requested ``per_page``, so
    the connector's pagination machinery is exercised across any per-page
    count the GitLab API might choose.
    """
    descendants = _flatten_descendants(tree, ROOT_GROUP_PATH)
    handler = _build_handler(
        descendants,
        page_size=page_size,
        group_path=ROOT_GROUP_PATH,
        branch=ANALYSIS_BRANCH,
    )
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    connector = GitLabConnector(
        base_url=BASE_URL,
        access_token=ACCESS_TOKEN,
        group_path=ROOT_GROUP_PATH,
        analysis_branch=ANALYSIS_BRANCH,
        client=client,
    )

    try:
        with connector:
            enumerated = list(connector.enumerate_projects())
    finally:
        client.close()

    expected = {(project.project_id, project.full_path) for project in descendants}
    actual = {
        (project.gitlab_project_id, project.full_path) for project in enumerated
    }

    # Set equality covers "no omissions" (every descendant appears) and
    # "no extras" (no non-descendants appear).
    assert actual == expected, (
        f"enumerated set differs from descendant set; "
        f"missing={expected - actual!r}, extra={actual - expected!r}"
    )
    # List length matches the unique-set count -> "no duplicates".
    assert len(enumerated) == len(expected), (
        f"enumerated list contains duplicates: "
        f"len={len(enumerated)} but unique projects={len(expected)}"
    )

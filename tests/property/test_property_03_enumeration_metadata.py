# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 3: For all enumerated projects produced by an Ingestion_Job, the EnumeratedProject record SHALL contain a non-null gitlab_project_id, full_path, analysis_branch_name equal to the configured Analysis_Branch, and (where the branch exists on the project) analysis_branch_commit_sha.
"""Property test for enumerated project metadata completeness.

**Validates Requirements 2.2, 15.4** (Property 3 in the design).

For every randomly generated GitLab group tree -- any nesting depth, any
number of subgroups per node, any number of projects per node, and an
arbitrary subset of projects on which the configured ``Analysis_Branch``
is *missing* -- every :class:`EnumeratedProject` produced by
``GitLab_Connector.enumerate_projects()`` SHALL carry:

* a non-null ``gitlab_project_id``;
* a non-empty ``full_path`` matching the synthetic tree's expected path
  for that GitLab project ID;
* ``analysis_branch_name`` equal to the configured ``Analysis_Branch``;
* ``analysis_branch_commit_sha`` equal to the SHA returned by the fake
  GitLab API when the branch exists on the project, or ``None`` (with
  ``branch_missing == True``) when the branch does not exist.

The fake GitLab API in this test wears two hats:

* ``GET /api/v4/groups/{group}/projects?include_subgroups=true`` returns
  every descendant project, paginated by RFC 5988 ``Link: rel="next"``
  headers at an independently-generated page size.
* ``GET /api/v4/projects/{id}/repository/branches/{branch}`` returns
  ``200`` with a synthetic commit SHA for projects whose
  ``branch_exists`` flag is ``True``, and ``404`` otherwise. The 404
  case is the canonical "branch missing" signal driving the
  ``branch_missing == True`` / ``analysis_branch_commit_sha is None``
  invariant in :class:`EnumeratedProject`.
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

# Path-segment alphabet: lowercase alphanum + hyphen is well within GitLab's
# unreserved URL-character set and keeps generated examples readable when
# Hypothesis prints a counterexample.
_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=8,
).filter(lambda s: not s.startswith("-") and not s.endswith("-"))


@dataclass(frozen=True)
class _GroupNode:
    """One group in the synthetic tree (basenames only -- paths are derived)."""

    name: str
    # Each project carries both its basename and a per-project flag deciding
    # whether the configured ``Analysis_Branch`` exists on it. The flag is
    # carried alongside the name so the same generated tree can drive both
    # the listing endpoint and the per-project branch endpoint without a
    # second random decision elsewhere.
    projects: tuple[tuple[str, bool], ...]
    subgroups: tuple[_GroupNode, ...]


@dataclass(frozen=True)
class _FakeProject:
    """One descendant project, with the identity GitLab would return."""

    project_id: int
    full_path: str
    branch_exists: bool


def _project_strategy() -> st.SearchStrategy[tuple[str, bool]]:
    """Generate a (basename, branch_exists) pair for one project."""
    return st.tuples(_NAME, st.booleans())


def _node_strategy(max_depth: int) -> st.SearchStrategy[_GroupNode]:
    """Generate a group subtree of depth at most ``max_depth``.

    Project basenames within a single group are unique (``unique_by`` on
    the basename component of each ``(name, branch_exists)`` pair), so
    every project's ``full_path`` is unique among its siblings. Subgroup
    names within a parent are likewise unique, so paths assembled by
    joining a parent path with a subgroup name never collide. Together
    these rules guarantee that every descendant project ends up at a
    unique ``full_path``, which makes the per-project metadata
    assertions meaningful.
    """
    if max_depth == 0:
        return st.builds(
            _GroupNode,
            name=_NAME,
            projects=st.lists(
                _project_strategy(), min_size=0, max_size=2, unique_by=lambda p: p[0]
            ).map(tuple),
            subgroups=st.just(()),
        )
    return st.builds(
        _GroupNode,
        name=_NAME,
        projects=st.lists(
            _project_strategy(), min_size=0, max_size=2, unique_by=lambda p: p[0]
        ).map(tuple),
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
        for proj_name, branch_exists in node.projects:
            full_path = f"{path}/{proj_name}"
            if full_path in seen_paths:
                continue
            seen_paths.add(full_path)
            out.append(
                _FakeProject(
                    project_id=next_id,
                    full_path=full_path,
                    branch_exists=branch_exists,
                )
            )
            next_id += 1
        for sub in node.subgroups:
            visit(sub, f"{path}/{sub.name}")

    visit(root, root_path)
    return out


# ---------------------------------------------------------------------------
# Fake GitLab REST API (``httpx.MockTransport`` handler factory)
# ---------------------------------------------------------------------------


def _expected_sha(project_id: int) -> str:
    """The SHA the fake returns for ``project_id`` when its branch exists."""
    return f"sha-for-project-{project_id}"


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
      ``Link: rel="next"`` headers.
    * ``GET /api/v4/projects/{id}/repository/branches/{branch}`` -- returns
      ``200`` with a deterministic synthetic commit SHA when the
      addressed project's ``branch_exists`` flag is ``True``, and ``404``
      otherwise. The 404 case is the canonical "branch missing" signal
      under test.

    The handler returns HTTP 500 on any other path so unexpected calls
    produce a loud failure during shrinking.
    """
    quoted_group = quote(group_path, safe="")
    listing_path = f"/api/v4/groups/{quoted_group}/projects"
    encoded_branch = quote(branch, safe="")
    by_id: dict[int, _FakeProject] = {p.project_id: p for p in descendants}

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

        # Per-project Analysis_Branch lookup. Match on the exact branch
        # segment so a request for some other branch (which would be a bug
        # in the connector) produces the unexpected-path 500.
        branch_prefix = "/api/v4/projects/"
        branch_suffix = f"/repository/branches/{encoded_branch}"
        if path.startswith(branch_prefix) and path.endswith(branch_suffix):
            middle = path[len(branch_prefix) : -len(branch_suffix)]
            try:
                project_id = int(middle)
            except ValueError:
                return httpx.Response(
                    500, json={"message": f"unparseable project id: {middle}"}
                )
            project = by_id.get(project_id)
            if project is None:
                return httpx.Response(
                    500, json={"message": f"unknown project id: {project_id}"}
                )
            if not project.branch_exists:
                # Requirement 15.5: 404 on the configured Analysis_Branch
                # is the canonical "branch missing" signal.
                return httpx.Response(404, json={"message": "404 Branch Not Found"})
            return httpx.Response(
                200,
                json={
                    "name": branch,
                    "commit": {"id": _expected_sha(project_id)},
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
def test_enumerated_project_metadata_is_complete(
    tree: _GroupNode,
    page_size: int,
) -> None:
    """Every produced ``EnumeratedProject`` carries the required metadata.

    For every descendant project the connector returns:

    * ``gitlab_project_id`` is a non-null ``int`` and matches the ID the
      fake assigned;
    * ``full_path`` is a non-empty string and equals the path the fake
      assigned to that ID;
    * ``analysis_branch_name`` equals the configured ``Analysis_Branch``;
    * when the branch exists on the project, ``analysis_branch_commit_sha``
      equals the SHA the fake returned and ``branch_missing`` is ``False``;
    * when the branch does not exist, ``analysis_branch_commit_sha`` is
      ``None`` and ``branch_missing`` is ``True``.
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

    expected_by_id: dict[int, _FakeProject] = {
        project.project_id: project for project in descendants
    }

    # The completeness/no-duplicates aspects of enumeration are covered by
    # Property 2; here we just need every produced record to address a
    # known descendant exactly once so the per-record assertions below
    # are well-defined.
    seen_ids: set[int] = set()

    for produced in enumerated:
        # gitlab_project_id is non-null and corresponds to a known descendant.
        assert isinstance(produced.gitlab_project_id, int), (
            f"gitlab_project_id must be int, got {type(produced.gitlab_project_id)!r}"
        )
        assert produced.gitlab_project_id in expected_by_id, (
            f"unknown gitlab_project_id={produced.gitlab_project_id!r} "
            f"(not in generated tree)"
        )
        assert produced.gitlab_project_id not in seen_ids, (
            f"duplicate gitlab_project_id={produced.gitlab_project_id!r}"
        )
        seen_ids.add(produced.gitlab_project_id)

        expected = expected_by_id[produced.gitlab_project_id]

        # full_path is non-empty and matches the synthetic tree.
        assert isinstance(produced.full_path, str) and produced.full_path != "", (
            f"full_path must be non-empty string, got {produced.full_path!r}"
        )
        assert produced.full_path == expected.full_path, (
            f"full_path mismatch for project {produced.gitlab_project_id}: "
            f"expected {expected.full_path!r}, got {produced.full_path!r}"
        )

        # analysis_branch_name equals the configured Analysis_Branch.
        assert produced.analysis_branch_name == ANALYSIS_BRANCH, (
            f"analysis_branch_name must equal configured "
            f"{ANALYSIS_BRANCH!r}, got {produced.analysis_branch_name!r}"
        )

        # SHA / branch_missing invariant -- two halves of Property 3.
        if expected.branch_exists:
            assert produced.branch_missing is False, (
                f"branch_missing must be False when branch exists on "
                f"project {produced.gitlab_project_id}"
            )
            assert produced.analysis_branch_commit_sha is not None, (
                f"analysis_branch_commit_sha must be non-None when branch "
                f"exists on project {produced.gitlab_project_id}"
            )
            assert (
                produced.analysis_branch_commit_sha
                == _expected_sha(produced.gitlab_project_id)
            ), (
                f"analysis_branch_commit_sha mismatch for project "
                f"{produced.gitlab_project_id}: expected "
                f"{_expected_sha(produced.gitlab_project_id)!r}, "
                f"got {produced.analysis_branch_commit_sha!r}"
            )
        else:
            assert produced.branch_missing is True, (
                f"branch_missing must be True when branch is absent on "
                f"project {produced.gitlab_project_id}"
            )
            assert produced.analysis_branch_commit_sha is None, (
                f"analysis_branch_commit_sha must be None when branch is "
                f"absent on project {produced.gitlab_project_id}, got "
                f"{produced.analysis_branch_commit_sha!r}"
            )

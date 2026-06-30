# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 29: For all Ingestion_Jobs and all in-scope projects, the GitLab_Connector SHALL fetch repository contents from the configured Analysis_Branch regardless of the project's GitLab default branch, and the resulting Project_Profile (when produced) SHALL record analysis_branch equal to the configured value and analysis_branch_commit_sha equal to the most recent commit SHA on that branch.
"""Property test for fetching the configured ``Analysis_Branch``.

**Validates Requirements 15.3, 15.4** (Property 29 in the design).

For every randomly generated set of in-scope projects -- each carrying a
``default_branch`` deliberately *different* from the configured
``Analysis_Branch`` -- the ``GitLab_Connector`` SHALL:

* record ``analysis_branch_name`` equal to the configured value on every
  produced :class:`EnumeratedProject` (Requirement 15.4);
* record ``analysis_branch_commit_sha`` equal to the SHA on the configured
  ``Analysis_Branch`` -- *never* the SHA on the GitLab default branch
  (Requirement 15.4);
* fetch repository contents (tree + every blob) at that
  ``Analysis_Branch`` SHA (Requirement 15.3), so the resulting
  :class:`RepositoryContents` is pinned to the configured branch's commit
  regardless of GitLab's per-project ``default_branch``.

The fake GitLab API in this test wears four hats:

* ``GET /api/v4/groups/{group}/projects?include_subgroups=true`` returns
  every generated project, with each project carrying a ``default_branch``
  drawn from a set disjoint from the configured ``Analysis_Branch``.
* ``GET /api/v4/projects/{id}/repository/branches/{Analysis_Branch}``
  returns the project's ``analysis_branch_sha``.
* ``GET /api/v4/projects/{id}/repository/branches/{default_branch}`` is
  treated as a contract violation: the connector must never query the
  per-project default branch, so the fake returns HTTP 500 (which would
  surface as a non-2xx and fail the test loudly).
* ``GET /api/v4/projects/{id}/repository/tree?ref=...`` and
  ``GET /api/v4/projects/{id}/repository/files/.../raw?ref=...`` insist
  that ``ref`` equals the project's ``analysis_branch_sha``; any other
  ``ref`` (notably the ``default_branch_sha``) is treated as a contract
  violation. The handler also records every observed ``ref`` so the test
  can additionally cross-check after enumeration that *only* the
  ``Analysis_Branch`` SHA was ever used.
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

#: Common GitLab default-branch names. Drawn from a set deliberately
#: disjoint from :data:`ANALYSIS_BRANCH` so every generated project
#: satisfies the property's "default branch != Analysis_Branch"
#: precondition by construction.
_DEFAULT_BRANCH_NAMES = ("main", "master", "develop", "production", "release")


# ---------------------------------------------------------------------------
# Synthetic project model
# ---------------------------------------------------------------------------

# Path-segment alphabet: lowercase alphanum + hyphen is well within GitLab's
# unreserved URL-character set and keeps generated examples readable when
# Hypothesis prints a counterexample.
_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=1,
    max_size=8,
).filter(lambda s: not s.startswith("-") and not s.endswith("-"))

_DEFAULT_BRANCH = st.sampled_from(_DEFAULT_BRANCH_NAMES)


@dataclass(frozen=True)
class _ProjectSpec:
    """Hypothesis-generated, ID-free description of one in-scope project."""

    name: str
    default_branch: str


_PROJECT_SPEC = st.builds(
    _ProjectSpec,
    name=_NAME,
    default_branch=_DEFAULT_BRANCH,
)


@dataclass(frozen=True)
class _FakeProject:
    """One descendant project with the identity GitLab would return.

    ``analysis_branch_sha`` is the SHA the fake returns for the configured
    ``Analysis_Branch``; ``default_branch_sha`` is the SHA that *would*
    correspond to the GitLab default branch. The two SHAs are
    deterministic per-project and always distinct so the assertion
    "the connector recorded the analysis-branch SHA, not the
    default-branch SHA" is meaningful.
    """

    project_id: int
    full_path: str
    default_branch: str
    analysis_branch_sha: str
    default_branch_sha: str


def _materialize_projects(specs: list[_ProjectSpec]) -> list[_FakeProject]:
    """Assign sequential GitLab project IDs and synthesize per-branch SHAs."""
    projects: list[_FakeProject] = []
    for index, spec in enumerate(specs, start=1):
        projects.append(
            _FakeProject(
                project_id=index,
                # The basename is unique across specs (``unique_by=name`` on
                # the generating list) so every full_path is unique.
                full_path=f"{ROOT_GROUP_PATH}/{spec.name}",
                default_branch=spec.default_branch,
                # Deterministic, distinct, plainly-identifiable SHAs make
                # mismatches easy to read in a Hypothesis counterexample.
                analysis_branch_sha=f"sha-analysis-{index}",
                default_branch_sha=f"sha-default-{index}",
            )
        )
    return projects


# ---------------------------------------------------------------------------
# Fake GitLab REST API (``httpx.MockTransport`` handler factory)
# ---------------------------------------------------------------------------

# A single analyzable file lives at the root of every project's tree. The
# basename matches :func:`_is_analyzable_path`'s "starts with README"
# admission rule, so the connector's ``fetch_repository_contents`` will
# actually request the file. The exact bytes do not matter to the
# property; we only check that the fetch happened at the expected ``ref``.
_README_PATH = "README.md"
_README_BODY = b"# Project README\n\nFetched from the configured Analysis_Branch.\n"

# Trivial Link header used to short-circuit ``_paginate``'s ``?page=N``
# fallback. Any non-empty Link header without a ``rel="next"`` entry tells
# the connector "this server speaks RFC 5988 pagination and there are no
# more pages", which is exactly the behavior we want for the single-page
# fakes below. The URL is a placeholder; the connector never follows it.
_SINGLE_PAGE_LINK = '<https://gitlab.example.com/_>; rel="first"'


@dataclass
class _Recorder:
    """Records every ``ref`` query parameter the connector sent to the fake.

    Populated by :func:`_build_handler`. After ``enumerate_projects`` and
    ``fetch_repository_contents`` have been driven for every project, the
    test asserts that *every* recorded ``ref`` equals the corresponding
    project's ``analysis_branch_sha`` -- i.e. the connector never
    addressed a different commit.
    """

    tree_refs: list[tuple[int, str | None]]
    file_refs: list[tuple[int, str | None]]


def _build_handler(
    projects: list[_FakeProject],
) -> tuple[Callable[[httpx.Request], httpx.Response], _Recorder]:
    """Build an :class:`httpx.MockTransport` handler scripting the GitLab API.

    The handler answers four endpoint families:

    * ``GET /api/v4/groups/{group}/projects`` -- returns every generated
      project on a single page (with a Link header so ``_paginate`` does
      not fall back to ``?page=N``). Each project record carries the
      generated ``default_branch`` so the connector observes a
      ``default_branch != Analysis_Branch`` payload, even though the
      connector intentionally ignores that field.
    * ``GET /api/v4/projects/{id}/repository/branches/{Analysis_Branch}``
      -- returns the project's ``analysis_branch_sha``.
    * ``GET /api/v4/projects/{id}/repository/branches/{default_branch}``
      -- returns HTTP 500 (treated as a contract violation: the connector
      must never query the per-project default branch).
    * ``GET /api/v4/projects/{id}/repository/tree`` and
      ``GET /api/v4/projects/{id}/repository/files/README.md/raw`` --
      both insist ``ref`` equals the project's ``analysis_branch_sha``;
      any other ``ref`` (notably the project's ``default_branch_sha``) is
      a contract violation and produces HTTP 500.

    Returns the handler and a :class:`_Recorder` that captures every
    observed ``ref`` for cross-checking after the connector runs.
    """
    quoted_group = quote(ROOT_GROUP_PATH, safe="")
    listing_path = f"/api/v4/groups/{quoted_group}/projects"
    encoded_analysis_branch = quote(ANALYSIS_BRANCH, safe="")
    encoded_readme = quote(_README_PATH, safe="")
    by_id: dict[int, _FakeProject] = {p.project_id: p for p in projects}

    recorder = _Recorder(tree_refs=[], file_refs=[])

    project_prefix = "/api/v4/projects/"
    branches_marker = "/repository/branches/"
    tree_suffix = "/repository/tree"
    file_suffix = f"/repository/files/{encoded_readme}/raw"

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: PLR0911, PLR0912
        # ``raw_path`` preserves percent-encoding; ``url.path`` would decode
        # ``%2F`` back to ``/`` and break the listing-path comparison.
        path = request.url.raw_path.decode("ascii").split("?", 1)[0]
        params = request.url.params

        # 1. Group listing. Returns every generated project on a single
        # page; the Link header (with rel="first" only, no rel="next")
        # short-circuits ``_paginate``'s ``?page=N`` fallback.
        if path == listing_path:
            items: list[dict[str, Any]] = [
                {
                    "id": project.project_id,
                    "path_with_namespace": project.full_path,
                    "description": None,
                    # Carried verbatim into the listing payload so the
                    # connector observes a default_branch != Analysis_Branch
                    # value. The connector deliberately ignores this field;
                    # this test asserts that ignorance.
                    "default_branch": project.default_branch,
                }
                for project in projects
            ]
            return httpx.Response(
                200,
                json=items,
                headers={"Link": _SINGLE_PAGE_LINK},
            )

        # 2. Per-project branch SHA lookup. Must match the configured
        # Analysis_Branch; any other branch (including the project's
        # default_branch) is a contract violation that fails loudly.
        if path.startswith(project_prefix) and branches_marker in path:
            # ``branches_marker`` keeps its leading "/" so partition on the
            # path *after* stripping ``project_prefix`` (which strips the
            # trailing "/" of "/api/v4/projects/") finds the boundary
            # cleanly. e.g. "1/repository/branches/uat".partition(
            # "/repository/branches/") -> ("1", "/repository/branches/",
            # "uat").
            id_str, _, branch_segment = path[len(project_prefix) :].partition(
                branches_marker
            )
            try:
                project_id = int(id_str)
            except ValueError:
                return httpx.Response(
                    500, json={"message": f"unparseable project id: {id_str!r}"}
                )
            project = by_id.get(project_id)
            if project is None:
                return httpx.Response(
                    500, json={"message": f"unknown project id: {project_id}"}
                )
            if branch_segment == encoded_analysis_branch:
                return httpx.Response(
                    200,
                    json={
                        "name": ANALYSIS_BRANCH,
                        "commit": {"id": project.analysis_branch_sha},
                    },
                )
            # Any branch other than the configured Analysis_Branch is a
            # bug in the connector under test. Surfacing as HTTP 500 makes
            # the test fail with a clear, shrinkable counterexample.
            return httpx.Response(
                500,
                json={
                    "message": (
                        f"connector queried branch {branch_segment!r} on "
                        f"project {project_id}; only the configured "
                        f"Analysis_Branch={ANALYSIS_BRANCH!r} is allowed"
                    )
                },
            )

        # 3. Tree endpoint. ``ref`` must equal the project's
        # ``analysis_branch_sha`` (Requirement 15.3); any other ref is a
        # contract violation.
        if path.startswith(project_prefix) and path.endswith(tree_suffix):
            id_str = path[len(project_prefix) : -len(tree_suffix)]
            try:
                project_id = int(id_str)
            except ValueError:
                return httpx.Response(
                    500, json={"message": f"unparseable project id: {id_str!r}"}
                )
            project = by_id.get(project_id)
            if project is None:
                return httpx.Response(
                    500, json={"message": f"unknown project id: {project_id}"}
                )
            ref = params.get("ref")
            recorder.tree_refs.append((project_id, ref))
            if ref != project.analysis_branch_sha:
                return httpx.Response(
                    500,
                    json={
                        "message": (
                            f"connector requested tree for project "
                            f"{project_id} with ref={ref!r}; expected "
                            f"analysis_branch_sha="
                            f"{project.analysis_branch_sha!r}"
                        )
                    },
                )
            entries: list[dict[str, Any]] = [
                {
                    "id": f"blob-{project_id}",
                    "name": _README_PATH,
                    "type": "blob",
                    "path": _README_PATH,
                    "mode": "100644",
                }
            ]
            return httpx.Response(
                200,
                json=entries,
                headers={"Link": _SINGLE_PAGE_LINK},
            )

        # 4. File content endpoint. ``ref`` must equal the project's
        # ``analysis_branch_sha`` (Requirement 15.3).
        if path.startswith(project_prefix) and path.endswith(file_suffix):
            id_str = path[len(project_prefix) : -len(file_suffix)]
            try:
                project_id = int(id_str)
            except ValueError:
                return httpx.Response(
                    500, json={"message": f"unparseable project id: {id_str!r}"}
                )
            project = by_id.get(project_id)
            if project is None:
                return httpx.Response(
                    500, json={"message": f"unknown project id: {project_id}"}
                )
            ref = params.get("ref")
            recorder.file_refs.append((project_id, ref))
            if ref != project.analysis_branch_sha:
                return httpx.Response(
                    500,
                    json={
                        "message": (
                            f"connector fetched file {_README_PATH!r} for "
                            f"project {project_id} with ref={ref!r}; "
                            f"expected analysis_branch_sha="
                            f"{project.analysis_branch_sha!r}"
                        )
                    },
                )
            return httpx.Response(200, content=_README_BODY)

        return httpx.Response(500, json={"message": f"unexpected path: {path}"})

    return handler, recorder


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(
    specs=st.lists(
        _PROJECT_SPEC,
        min_size=1,
        max_size=5,
        # Unique basenames -> unique full_paths -> the per-project
        # assertions below address each generated project exactly once.
        unique_by=lambda s: s.name,
    ),
)
@settings(max_examples=100)
def test_connector_fetches_from_configured_analysis_branch(
    specs: list[_ProjectSpec],
) -> None:
    """Connector reads from ``Analysis_Branch``, not from ``default_branch``.

    For every generated project (each carrying a ``default_branch`` drawn
    from a set disjoint from the configured ``Analysis_Branch``):

    * the produced :class:`EnumeratedProject` records
      ``analysis_branch_name`` equal to the configured value
      (Requirement 15.4);
    * its ``analysis_branch_commit_sha`` equals the SHA on the configured
      ``Analysis_Branch`` and is *not* the SHA that would correspond to
      the project's GitLab default branch (Requirement 15.4);
    * ``fetch_repository_contents`` issued at that SHA returns
      :class:`RepositoryContents` whose ``commit_sha`` matches the
      ``Analysis_Branch`` SHA, and the fake records that *every* tree and
      file fetch carried ``ref=analysis_branch_sha`` -- never the
      project's ``default_branch_sha`` (Requirement 15.3).

    Any deviation from these invariants is surfaced either through a
    failed assertion below or through one of the contract-violation
    HTTP 500 responses in :func:`_build_handler` (which the connector
    propagates as :class:`httpx.HTTPStatusError`).
    """
    projects = _materialize_projects(specs)

    # Sanity-check the precondition the property's "regardless of the
    # project's GitLab default branch" clause depends on: every generated
    # project's default_branch differs from the configured Analysis_Branch.
    for project in projects:
        assert project.default_branch != ANALYSIS_BRANCH, (
            f"test fixture regression: project {project.project_id} has "
            f"default_branch == Analysis_Branch ({ANALYSIS_BRANCH!r})"
        )

    handler, recorder = _build_handler(projects)
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    connector = GitLabConnector(
        base_url=BASE_URL,
        access_token=ACCESS_TOKEN,
        group_path=ROOT_GROUP_PATH,
        analysis_branch=ANALYSIS_BRANCH,
        client=client,
    )

    by_id: dict[int, _FakeProject] = {p.project_id: p for p in projects}

    try:
        with connector:
            enumerated = list(connector.enumerate_projects())

            # Sanity: every generated project showed up exactly once.
            # Property 2 already proves enumeration completeness; this
            # cheap shape check makes the per-project assertions below
            # well-defined.
            assert len(enumerated) == len(projects), (
                f"expected {len(projects)} enumerated projects, "
                f"got {len(enumerated)}"
            )

            for produced in enumerated:
                expected = by_id[produced.gitlab_project_id]

                # Requirement 15.4: analysis_branch_name equals the
                # configured value, regardless of default_branch.
                assert produced.analysis_branch_name == ANALYSIS_BRANCH, (
                    f"analysis_branch_name must equal configured "
                    f"{ANALYSIS_BRANCH!r}, got "
                    f"{produced.analysis_branch_name!r}"
                )

                # Requirement 15.4: SHA equals the SHA on the configured
                # Analysis_Branch (NOT the project's default branch).
                assert (
                    produced.analysis_branch_commit_sha
                    == expected.analysis_branch_sha
                ), (
                    f"analysis_branch_commit_sha for project "
                    f"{produced.gitlab_project_id}: expected "
                    f"{expected.analysis_branch_sha!r}, got "
                    f"{produced.analysis_branch_commit_sha!r}"
                )
                assert (
                    produced.analysis_branch_commit_sha
                    != expected.default_branch_sha
                ), (
                    f"analysis_branch_commit_sha for project "
                    f"{produced.gitlab_project_id} unexpectedly equals "
                    f"the default_branch_sha "
                    f"{expected.default_branch_sha!r}"
                )
                assert produced.branch_missing is False, (
                    f"branch_missing must be False when Analysis_Branch "
                    f"exists on project {produced.gitlab_project_id}"
                )

                # Requirement 15.3: contents are read from the configured
                # Analysis_Branch's commit, regardless of default_branch.
                # The fake's tree and file endpoints assert ref ==
                # analysis_branch_sha; any other ref produces HTTP 500.
                assert produced.analysis_branch_commit_sha is not None
                contents = connector.fetch_repository_contents(
                    produced.gitlab_project_id,
                    produced.analysis_branch_commit_sha,
                )
                assert contents.commit_sha == expected.analysis_branch_sha, (
                    f"RepositoryContents.commit_sha for project "
                    f"{produced.gitlab_project_id}: expected "
                    f"{expected.analysis_branch_sha!r}, got "
                    f"{contents.commit_sha!r}"
                )
                assert _README_PATH in contents.files, (
                    f"expected {_README_PATH!r} in fetched contents for "
                    f"project {produced.gitlab_project_id}, got "
                    f"{sorted(contents.files.keys())!r}"
                )
    finally:
        client.close()

    # Cross-check: every recorded ref (across both tree and file fetches,
    # for every project the connector touched) was the project's
    # analysis_branch_sha. This guards against any code path that might
    # silently fall back to the default branch SHA.
    assert recorder.tree_refs, (
        "expected the connector to have requested at least one tree; "
        "got none"
    )
    for project_id, ref in recorder.tree_refs:
        assert ref == by_id[project_id].analysis_branch_sha, (
            f"tree fetch for project {project_id} used ref={ref!r}; "
            f"expected {by_id[project_id].analysis_branch_sha!r}"
        )
    assert recorder.file_refs, (
        "expected the connector to have fetched at least one file; "
        "got none"
    )
    for project_id, ref in recorder.file_refs:
        assert ref == by_id[project_id].analysis_branch_sha, (
            f"file fetch for project {project_id} used ref={ref!r}; "
            f"expected {by_id[project_id].analysis_branch_sha!r}"
        )

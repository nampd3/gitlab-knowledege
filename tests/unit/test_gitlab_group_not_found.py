"""Unit tests for HTTP 404 group-not-found handling (task 5.8).

Requirement 2.4 says that when the configured GitLab group returns
HTTP 404 during enumeration, the ``GitLab_Connector`` must raise
:class:`GitLabGroupNotFoundError` carrying the configured group path,
and the ``Ingestion_Coordinator`` must abort the in-progress
``Ingestion_Job`` and surface the error to the caller. The design's
"Ingestion_Coordinator → Job procedure (full refresh)" section
specifies that the job ends through a single ``try``/``finally``: the
in-progress snapshot row is rolled back via
``Knowledge_Store.abort_snapshot`` and the running slot is released,
both before the original exception is re-raised. Critically,
``commit_snapshot`` must *not* run on this path -- the readers must
continue to see whatever snapshot was current before the failed job
started.

These tests verify that contract end-to-end against an
:class:`IngestionCoordinator` wired up with purpose-built fakes for
``KnowledgeStore``, ``ProjectCatalog``, and ``GitLabConnector``. The
fakes record every method call so the assertions can prove (a) that
``begin_snapshot`` ran first, (b) that ``abort_snapshot`` ran on the
same ``snapshot_id``, and (c) that ``commit_snapshot`` never ran.

A second test exercises :class:`GitLabGroupNotFoundError` directly to
pin the message format and the ``group_path`` attribute that downstream
MCP / visualization surfaces depend on (Requirement 2.4 +
:mod:`project_knowledge_mcp.errors`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from project_knowledge_mcp.errors import GitLabGroupNotFoundError
from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator
from project_knowledge_mcp.models import SnapshotTrigger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from project_knowledge_mcp.models import (
        EnumeratedProject,
        ProjectProfile,
        RepositoryContents,
    )


pytestmark = pytest.mark.unit


# Fixed group path used across the tests. Pinning the value to a
# realistic, slash-bearing path makes the assertions self-documenting
# and exercises the same shape that GitLab encodes as ``acme%2Fplatform``
# on the wire.
GROUP_PATH = "acme/platform"


# ---------------------------------------------------------------------------
# Minimal fakes for the three collaborators the coordinator needs.
# ---------------------------------------------------------------------------


@dataclass
class _FakeKnowledgeStore:
    """Records every coordinator -> store call for later assertion.

    The coordinator only invokes ``begin_snapshot``, ``abort_snapshot``,
    and ``commit_snapshot`` on this code path (the per-project loop
    below the failing ``enumerate_projects`` call never runs). Each
    method appends to ``calls`` so the test can assert call ordering;
    ``begin_snapshot`` returns a deterministic id so the test can
    correlate the subsequent ``abort_snapshot`` argument.
    """

    begin_snapshot_id: int = 17
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def begin_snapshot(
        self,
        trigger: SnapshotTrigger | str,
        parent_snapshot_id: int | None = None,
    ) -> int:
        self.calls.append(("begin_snapshot", (trigger, parent_snapshot_id)))
        return self.begin_snapshot_id

    def abort_snapshot(self, snapshot_id: int) -> None:
        self.calls.append(("abort_snapshot", (snapshot_id,)))

    def commit_snapshot(self, snapshot_id: int) -> None:  # pragma: no cover
        # Reaching this method on the 404 path would be a regression.
        # Recording the call (rather than raising) lets the assertion
        # below produce a clearer error message naming the unexpected
        # entry instead of a stack trace from inside the coordinator.
        self.calls.append(("commit_snapshot", (snapshot_id,)))

    def write_profile(  # pragma: no cover - never reached on this path
        self,
        snapshot_id: int,
        profile: ProjectProfile,
        produced_at: object,
        commit_sha: str,
    ) -> None:
        self.calls.append(("write_profile", (snapshot_id, profile.gitlab_project_id)))

    def record_skip(  # pragma: no cover - never reached on this path
        self,
        snapshot_id: int,
        gitlab_project_id: int,
        reason: str,
        detail: str | None,
    ) -> None:
        self.calls.append(
            ("record_skip", (snapshot_id, gitlab_project_id, reason, detail))
        )


@dataclass
class _FakeProjectCatalog:
    """Catalog stub: ``populate_in_scope`` should never run on this path.

    The coordinator only reaches ``populate_in_scope`` *after*
    ``enumerate_projects`` returns successfully, so this fake exists
    purely to satisfy the ``IngestionCoordinator`` constructor's typing
    contract. Recording any call lets the test prove the catalog stayed
    untouched on the 404 path.
    """

    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def populate_in_scope(  # pragma: no cover - never reached on this path
        self,
        snapshot_id: int,
        enumerated: object,
    ) -> None:
        self.calls.append(("populate_in_scope", (snapshot_id,)))


@dataclass
class _FakeGitLabConnector:
    """Raises :class:`GitLabGroupNotFoundError` from ``enumerate_projects``.

    The error is raised eagerly (not from inside the generator) because
    the coordinator materializes the iterator with ``list(...)`` and
    the design's contract is "raised on the first call against the
    configured group". Either an eager raise or a generator raise
    surfaces the same way to the coordinator; the eager form is simpler
    to read.
    """

    group_path: str = GROUP_PATH

    def enumerate_projects(self) -> Iterator[EnumeratedProject]:
        raise GitLabGroupNotFoundError(self.group_path)

    def fetch_repository_contents(  # pragma: no cover - never reached
        self,
        project_id: int,
        commit_sha: str,
    ) -> RepositoryContents:
        msg = "fetch_repository_contents must not be called on the 404 path"
        raise AssertionError(msg)


def _unreachable_analyze(  # pragma: no cover - never reached on this path
    *,
    project_id: int,
    full_path: str,
    analysis_branch: str,
    commit_sha: str,
    repo_description: str | None,
    repository_contents: RepositoryContents,
) -> ProjectProfile:
    msg = "Project_Analyzer.analyze must not be called on the 404 path"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_404_on_group_aborts_snapshot_and_surfaces_group_not_found() -> None:
    """End-to-end: HTTP 404 on the group -> abort + raise (Requirement 2.4)."""

    store = _FakeKnowledgeStore()
    catalog = _FakeProjectCatalog()
    connector = _FakeGitLabConnector(group_path=GROUP_PATH)

    coordinator = IngestionCoordinator(
        knowledge_store=store,
        project_catalog=catalog,
        gitlab_connector=connector,
        analyze=_unreachable_analyze,
    )

    with pytest.raises(GitLabGroupNotFoundError) as excinfo:
        coordinator.start_full_refresh()

    err = excinfo.value

    # The error carries the configured group path verbatim so callers
    # (MCP tool result message, visualization HTML body) can name the
    # group that could not be resolved.
    assert err.group_path == GROUP_PATH
    # The formatted message includes the group path so MCP clients
    # surfacing ``str(err)`` directly still see it.
    assert GROUP_PATH in str(err)
    assert err.message == f"GitLab group '{GROUP_PATH}' not found"

    # Call ordering: begin_snapshot must run first (so there is a
    # snapshot row to roll back), and abort_snapshot must run on the
    # same id; commit_snapshot must never run because the snapshot
    # never reached the "completed" state.
    method_names = [name for name, _ in store.calls]
    assert method_names == ["begin_snapshot", "abort_snapshot"], (
        f"unexpected store call sequence: {store.calls}"
    )
    assert "commit_snapshot" not in method_names
    assert "write_profile" not in method_names
    assert "record_skip" not in method_names

    # The aborted snapshot id must equal the id ``begin_snapshot``
    # returned -- proving the coordinator is tearing down the *same*
    # snapshot it just opened, not some unrelated one.
    begin_args = store.calls[0][1]
    abort_args = store.calls[1][1]
    assert begin_args == (SnapshotTrigger.FULL, None)
    assert abort_args == (store.begin_snapshot_id,)

    # The catalog populate step lives *after* ``enumerate_projects`` in
    # the procedure, so it must not have run on the 404 path.
    assert catalog.calls == []

    # Coordinator state must be idle again so the next request can run.
    assert coordinator.is_idle() is True


def test_group_not_found_message_format() -> None:
    """Direct construction: pin the public shape MCP/visualization rely on."""

    err = GitLabGroupNotFoundError("some/group/path")

    assert err.group_path == "some/group/path"
    # Canonical message format from
    # ``project_knowledge_mcp.errors.GitLabGroupNotFoundError`` --
    # mirrored on both ``str(err)`` and ``err.message`` (Requirement 2.4).
    assert err.message == "GitLab group 'some/group/path' not found"
    assert str(err) == "GitLab group 'some/group/path' not found"

"""Unit tests for HTTP 401/403 abort + report try/finally semantics.

These tests cover the coordinator-level error path described in the
``Ingestion_Coordinator`` section of the design:

* When the ``GitLab_Connector`` raises :class:`GitLabAuthError` (HTTP
  401 or 403) during enumeration, the in-progress snapshot is aborted
  through ``Knowledge_Store.abort_snapshot``, the running slot is
  released through ``handle.abort()`` (in a single ``finally`` arm so
  it runs even when the abort step itself raises), and the original
  auth failure surfaces to the caller carrying its ``status_code``.
* If either step inside that ``try``/``finally`` pair fails on its
  own, the design's "surfaces whichever fails" rule applies: the
  store-level failure surfaces (with the original auth error attached
  via ``__context__``), and the coordinator still returns to idle.

The tests deliberately do not exercise the SQLite-backed
:class:`KnowledgeStore` or the network-facing
:class:`GitLabConnector` -- both are replaced by purpose-built fakes
that record every call and let each test inject exactly one error at
a time. That keeps these tests focused on the coordinator's
abort-and-report contract (Requirement 2.3) and its idle-on-failure
guarantee, and lets a single failing assertion finger the coordinator
rather than a transitive collaborator.

Implements Requirement 2.3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from project_knowledge_mcp.errors import GitLabAuthError
from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from project_knowledge_mcp.models import (
        EnumeratedProject,
        ProjectProfile,
        RepositoryContents,
    )

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeKnowledgeStore:
    """Minimal ``KnowledgeStore`` stand-in that records every coordinator call.

    The fake captures the order of ``begin_snapshot`` /
    ``abort_snapshot`` / ``commit_snapshot`` calls so the tests can
    assert the exact sequence the coordinator went through. It also
    lets a test inject an exception that ``abort_snapshot`` raises the
    next time it is called -- this is how test 3 simulates a
    store-level failure inside the coordinator's try/finally.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._next_snapshot_id = 0
        self.abort_snapshot_error: BaseException | None = None

    def begin_snapshot(
        self,
        trigger: Any,
        parent_snapshot_id: int | None = None,
    ) -> int:
        self._next_snapshot_id += 1
        snapshot_id = self._next_snapshot_id
        self.calls.append(("begin_snapshot", snapshot_id, trigger, parent_snapshot_id))
        return snapshot_id

    def abort_snapshot(self, snapshot_id: int) -> None:
        self.calls.append(("abort_snapshot", snapshot_id))
        if self.abort_snapshot_error is not None:
            raise self.abort_snapshot_error

    def commit_snapshot(self, snapshot_id: int) -> None:
        self.calls.append(("commit_snapshot", snapshot_id))

    def write_profile(
        self,
        snapshot_id: int,
        profile: ProjectProfile,
        produced_at: Any,
        commit_sha: str,
    ) -> None:
        # Not exercised in these tests -- enumeration raises before
        # any per-project write would run -- but defined so the fake
        # satisfies the duck-typed contract the coordinator expects.
        self.calls.append(("write_profile", snapshot_id))

    def record_skip(
        self,
        snapshot_id: int,
        gitlab_project_id: int,
        reason: str,
        detail: str | None,
    ) -> None:
        self.calls.append(("record_skip", snapshot_id, gitlab_project_id))

    def get_current_snapshot_id(self) -> int | None:
        return None

    def call_names(self) -> list[str]:
        """Return only the method-name component of every recorded call."""

        return [call[0] for call in self.calls]


class FakeProjectCatalog:
    """``ProjectCatalog`` stand-in: records ``populate_in_scope`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def populate_in_scope(
        self,
        snapshot_id: int,
        enumerated: Any,
    ) -> None:
        self.calls.append(("populate_in_scope", snapshot_id))

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        return False


class FakeGitLabConnector:
    """``GitLabConnector`` stand-in that raises a configured error on enumerate."""

    def __init__(self, enumerate_error: BaseException) -> None:
        self._enumerate_error = enumerate_error

    def enumerate_projects(self) -> Iterator[EnumeratedProject]:
        # Raised directly so the coordinator's
        # ``list(gitlab_connector.enumerate_projects())`` call surfaces
        # the auth error from the very first byte of the GitLab call.
        raise self._enumerate_error

    def fetch_repository_contents(
        self,
        project_id: int,
        commit_sha: str,
    ) -> RepositoryContents:
        # Never reached when enumerate_projects raises, but defined
        # so the duck-typed contract is satisfied.
        raise AssertionError("fetch_repository_contents should not be called")


def _unreachable_analyze(**_kwargs: Any) -> ProjectProfile:
    """Sentinel ``analyze`` callable for tests where enumeration aborts first."""

    raise AssertionError("analyze should not be called when enumeration raises")


def _make_coordinator(
    *,
    enumerate_error: BaseException,
    abort_snapshot_error: BaseException | None = None,
) -> tuple[IngestionCoordinator, FakeKnowledgeStore, FakeProjectCatalog]:
    """Build a coordinator wired with the three fakes used by these tests."""

    store = FakeKnowledgeStore()
    store.abort_snapshot_error = abort_snapshot_error
    catalog = FakeProjectCatalog()
    connector = FakeGitLabConnector(enumerate_error=enumerate_error)
    coordinator = IngestionCoordinator(
        knowledge_store=store,  # type: ignore[arg-type]
        project_catalog=catalog,  # type: ignore[arg-type]
        gitlab_connector=connector,  # type: ignore[arg-type]
        analyze=_unreachable_analyze,
    )
    return coordinator, store, catalog


# ---------------------------------------------------------------------------
# Test 1: 401 during enumeration aborts and surfaces the auth error
# ---------------------------------------------------------------------------


def test_401_during_enumeration_aborts_snapshot_and_surfaces_auth_error() -> None:
    coordinator, store, _catalog = _make_coordinator(
        enumerate_error=GitLabAuthError(401),
    )

    with pytest.raises(GitLabAuthError) as excinfo:
        coordinator.start_full_refresh()

    # Requirement 2.3: the surfaced error carries the GitLab status code.
    assert excinfo.value.status_code == 401

    # The coordinator opened a snapshot, then aborted that exact
    # snapshot id -- and never committed it. ``begin_snapshot`` and
    # ``abort_snapshot`` are the only store calls expected on this path.
    assert store.call_names() == ["begin_snapshot", "abort_snapshot"]
    begin_call = store.calls[0]
    abort_call = store.calls[1]
    assert begin_call[0] == "begin_snapshot"
    assert abort_call == ("abort_snapshot", begin_call[1])
    assert "commit_snapshot" not in store.call_names()

    # The single try/finally released the running slot, so the next
    # ``try_start`` (which ``is_idle`` consults under the same lock)
    # will succeed.
    assert coordinator.is_idle() is True


# ---------------------------------------------------------------------------
# Test 2: 403 during enumeration aborts and surfaces the auth error
# ---------------------------------------------------------------------------


def test_403_during_enumeration_aborts_snapshot_and_surfaces_auth_error() -> None:
    coordinator, store, _catalog = _make_coordinator(
        enumerate_error=GitLabAuthError(403),
    )

    with pytest.raises(GitLabAuthError) as excinfo:
        coordinator.start_full_refresh()

    assert excinfo.value.status_code == 403

    assert store.call_names() == ["begin_snapshot", "abort_snapshot"]
    assert "commit_snapshot" not in store.call_names()
    assert coordinator.is_idle() is True


# ---------------------------------------------------------------------------
# Test 3: abort_snapshot failure surfaces the underlying store error
# ---------------------------------------------------------------------------


def test_abort_snapshot_failure_surfaces_underlying_store_error() -> None:
    """The design's "surfaces whichever fails" rule.

    When the coordinator catches :class:`GitLabAuthError` and calls
    ``Knowledge_Store.abort_snapshot`` inside the single try/finally,
    a failure of ``abort_snapshot`` itself replaces the auth error
    on its way out; the auth error is preserved as ``__context__`` so
    diagnostics can still recover both. ``handle.abort()`` runs in
    the ``finally`` arm regardless, so the coordinator still returns
    to idle.
    """

    auth_error = GitLabAuthError(401)
    store_error = RuntimeError("disk full")
    coordinator, store, _catalog = _make_coordinator(
        enumerate_error=auth_error,
        abort_snapshot_error=store_error,
    )

    with pytest.raises(RuntimeError) as excinfo:
        coordinator.start_full_refresh()

    # The store-level failure is what surfaces.
    assert excinfo.value is store_error

    # The original auth error is preserved on the chain so callers
    # diagnosing the failure can still recover both -- this is the
    # "single try/finally" rule from the design (the auth error
    # context is not lost).
    assert excinfo.value.__context__ is auth_error

    # Both store calls were attempted: the coordinator opened the
    # snapshot, then attempted to abort it (which raised). No commit.
    assert store.call_names() == ["begin_snapshot", "abort_snapshot"]
    assert "commit_snapshot" not in store.call_names()

    # ``handle.abort()`` ran inside the ``finally`` arm even though
    # ``abort_snapshot`` raised, so the running slot is released and
    # the coordinator is back to idle for the next attempt.
    assert coordinator.is_idle() is True

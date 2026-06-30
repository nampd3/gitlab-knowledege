"""Subprocess runner that hosts ``main.main`` with a slow GitLab fake.

Used by :mod:`tests.integration.test_shutdown_e2e` as the subprocess
target for the end-to-end shutdown ordering test. The script monkey-
patches three collaborator entry points that the production ``main.main``
wiring would otherwise drive against a real GitLab instance, then
delegates to ``project_knowledge_mcp.main.main`` so the production
start-up sequence, signal-handler installation, and Requirement 12.9
shutdown ordering are exercised verbatim:

* :meth:`GitLabConnector.enumerate_projects` is replaced with a
  generator that **blocks until SIGTERM arrives** and then yields no
  projects. The block is implemented as a polling loop on a
  ``threading.Event`` set by the script's own SIGTERM/SIGINT handler.
  The block holds the ``Ingestion_Coordinator`` in the ``running``
  state so the test can observe an ``in_progress`` snapshot row in the
  ``Knowledge_Store`` and then send SIGTERM mid-flight, exactly as
  Requirement 12.9 mandates.

* :meth:`GitLabConnector._fetch_branch_sha` and
  :meth:`GitLabConnector.fetch_repository_contents` are replaced with
  no-op stubs as a defensive measure — the slow ``enumerate_projects``
  blocks the worker before either is reached, but stubbing them keeps
  the runner robust to refactors that move the sleep to a different
  call site.

The filename is intentionally prefixed with an underscore so pytest's
default test discovery (``test_*.py`` / ``*_test.py``) does not pick
it up as a test module.

Why a polling loop instead of ``time.sleep`` or ``threading.Event.wait``
without a timeout: the SIGTERM handler installed by ``main`` (via
``loop.add_signal_handler``) overrides any prior ``signal.signal``
handler we install at import time, so we can't rely on a Python-level
signal handler firing inside the worker thread. Instead, we expose a
sentinel file path (passed via the ``SHUTDOWN_E2E_RELEASE_FILE``
environment variable). The integration test creates the file *after*
sending SIGTERM and verifying the in-flight snapshot has been marked
``failed`` by the production shutdown code; the worker's polling loop
notices the file and unblocks. This sequencing gives the test a
deterministic window in which the in-flight ``Ingestion_Job`` is
observably ``in_progress`` AND a deterministic way to release the
worker thread once the assertions about the shutdown ordering have
been made.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from project_knowledge_mcp import gitlab_connector as _gitlab_connector_module
from project_knowledge_mcp.main import main as _real_main
from project_knowledge_mcp.models import RepositoryContents

if TYPE_CHECKING:
    from collections.abc import Iterator

    from project_knowledge_mcp.models import EnumeratedProject


# Environment variable carrying the absolute path to the sentinel file
# the integration test creates to release the slow ``enumerate_projects``
# generator. The runner refuses to start when the variable is unset so
# a misconfigured test does not hang silently — surfacing the missing
# wiring is more useful than a 30-second timeout in CI.
ENV_RELEASE_FILE = "SHUTDOWN_E2E_RELEASE_FILE"

# Polling interval, in seconds, used by the slow ``enumerate_projects``
# generator while it waits for the release file to appear. Small enough
# that the worker thread reacts to the test's release within roughly
# one tick (so the subprocess exits promptly once shutdown assertions
# are made), large enough that the polling loop itself is cheap.
_POLL_INTERVAL_SECONDS = 0.05

# Hard upper bound on how long ``enumerate_projects`` will block waiting
# for the release file. The integration test should always create the
# file within a few seconds of sending SIGTERM, so 60s is purely
# defensive: if the test's assertions hang for any reason, the runner
# unblocks itself rather than wedging the CI worker indefinitely.
_MAX_BLOCK_SECONDS = 60.0


def _slow_enumerate_projects(self: object) -> Iterator[EnumeratedProject]:
    """Stand-in for :meth:`GitLabConnector.enumerate_projects`.

    Blocks until either the release file appears at the path named by
    :data:`ENV_RELEASE_FILE` or :data:`_MAX_BLOCK_SECONDS` elapses,
    then yields no projects. The empty yield is sufficient: the
    coordinator's full-refresh procedure handles the empty enumeration
    gracefully (zero per-project iterations, then ``commit_snapshot``).
    For this test the *commit* never happens because the production
    shutdown code marks the in-flight snapshot ``failed`` before the
    worker thread reaches that step.

    The instance argument ``self`` is unused (it would be the
    :class:`GitLabConnector`); the parameter exists only to match the
    bound-method signature so the monkey-patched method has the same
    arity the production code expects.
    """

    del self  # unused; kept for signature compatibility.

    release_path_str = os.environ.get(ENV_RELEASE_FILE)
    if release_path_str is None or release_path_str.strip() == "":
        # The integration test must always wire this. Fail loudly
        # rather than silently hang for ``_MAX_BLOCK_SECONDS``.
        raise RuntimeError(
            f"shutdown e2e runner requires the {ENV_RELEASE_FILE!r} "
            "environment variable to point at the test's release file"
        )
    release_path = Path(release_path_str)

    deadline = time.monotonic() + _MAX_BLOCK_SECONDS
    while time.monotonic() < deadline:
        if release_path.exists():
            break
        time.sleep(_POLL_INTERVAL_SECONDS)

    # Yield nothing. The coordinator's per-project loop is a no-op on
    # an empty list; if execution ever reaches the subsequent
    # ``commit_snapshot`` call (i.e. shutdown did NOT mark the snapshot
    # ``failed``) the test will detect that as an
    # ``unexpected_pass``-style failure during DB verification.
    if False:  # pragma: no cover - keeps the function a generator.
        yield  # type: ignore[unreachable]
    return


def _stub_fetch_branch_sha(self: object, project_id: int, branch: str) -> str | None:
    """No-op stand-in for :meth:`GitLabConnector._fetch_branch_sha`.

    The slow ``enumerate_projects`` blocks before this method is
    reached, but stubbing it defends against future refactors that
    might move the network-touching code path. Returns ``None`` so any
    accidental invocation routes the project through the
    ``branch_missing`` branch (a Skip row) rather than producing
    arbitrary fake SHAs.
    """

    del self, project_id, branch  # unused; kept for signature compatibility.
    return None


def _stub_fetch_repository_contents(
    self: object,
    project_id: int,
    commit_sha: str,
) -> RepositoryContents:
    """No-op stand-in for :meth:`GitLabConnector.fetch_repository_contents`.

    Unreachable in this test scenario because the slow
    ``enumerate_projects`` yields no projects, but stubbed for the
    same defensive-against-refactor reason as ``_stub_fetch_branch_sha``.
    """

    del self  # unused; kept for signature compatibility.
    return RepositoryContents(
        gitlab_project_id=project_id,
        commit_sha=commit_sha,
        files={},
    )


def _install_monkey_patches() -> None:
    """Replace the GitLab-touching entry points with their stand-ins."""

    # ``setattr`` on the class so every instance the production wiring
    # constructs picks up the patched methods. ``main`` constructs the
    # ``GitLabConnector`` after import, so patching here (before the
    # call to ``_real_main``) is sufficient. ``setattr`` is preferred
    # over plain attribute assignment because mypy treats direct
    # method assignment on a class as a type error; ``setattr`` is
    # the conventional escape hatch for runtime monkey-patching.
    setattr(  # noqa: B010 - intentional dynamic patching of class methods
        _gitlab_connector_module.GitLabConnector,
        "enumerate_projects",
        _slow_enumerate_projects,
    )
    setattr(  # noqa: B010
        _gitlab_connector_module.GitLabConnector,
        "_fetch_branch_sha",
        _stub_fetch_branch_sha,
    )
    setattr(  # noqa: B010
        _gitlab_connector_module.GitLabConnector,
        "fetch_repository_contents",
        _stub_fetch_repository_contents,
    )


def main() -> int:
    """Entry point for ``python <this file>``."""

    _install_monkey_patches()
    return _real_main()


if __name__ == "__main__":
    sys.exit(main())

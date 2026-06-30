"""No-toolchain assertion test for the Go analyzer.

The Go layer is pure Python: it parses ``.go`` source and ``go.mod`` in
process and never delegates to an external Go toolchain (``go build``,
``go run``, ``gofmt``, ``goimports``, ``go vet``, ``staticcheck``, or any
other binary). This test pins that invariant down structurally by
monkey-patching every process-launching surface in the standard library
and asserting that :func:`analyze` does not touch any of them while
processing the four sample-repo snapshots committed under
``tests/integration/golden/go/<repo-name>/``.

If a future change introduces a subprocess call anywhere on the Go
analysis code path -- even an "innocent" one such as a version probe --
the patched surface raises :class:`AssertionError` and the test reports
the exact function, args, and repo that triggered the call.

Implements task 12.5 of the Go analyzer support spec. Validates
Requirements 10.1 (pure-Python parsing) and 10.2 (no Go toolchain).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from project_knowledge_mcp.models import RepositoryContents
from project_knowledge_mcp.project_analyzer import analyze

pytestmark = pytest.mark.unit


#: The four sample repositories whose snapshots this test exercises.
#: Matches the directory layout under ``tests/integration/golden/go/``
#: and the tuple in :data:`tests.integration.golden.go._curate.SAMPLE_REPOS`.
SAMPLE_REPOS: tuple[str, ...] = (
    "fec_pool_service",
    "repayment_service",
    "cat-service",
    "aps_los_vtiger",
)


#: Every process-launching surface in the ``subprocess`` module the
#: analyzer might plausibly reach for. The list is exhaustive within
#: the standard library's public API; any new helper future stdlib
#: versions add will still ultimately go through one of these.
_SUBPROCESS_SURFACES: tuple[str, ...] = (
    "run",
    "Popen",
    "call",
    "check_call",
    "check_output",
    "getoutput",
    "getstatusoutput",
)


#: Every process-launching surface in the ``os`` module. Some are
#: POSIX-only (``os.spawnvp``, ``os.posix_spawnp``); guarded with
#: ``hasattr`` so the test stays cross-platform.
_OS_SURFACES: tuple[str, ...] = (
    "system",
    "popen",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "posix_spawn",
    "posix_spawnp",
    "fork",
    "forkpty",
)


def _golden_dir() -> Path:
    """Return the directory holding the per-repo snapshot subdirectories."""
    return Path(__file__).resolve().parent.parent / "integration" / "golden" / "go"


def _load_snapshot(repo_name: str) -> RepositoryContents:
    """Load the ``repository_contents.json`` snapshot for ``repo_name``.

    Uses the same loading convention as
    :func:`tests.integration.golden.go._curate._load_snapshot` and
    :func:`tests.integration.test_go_analyzer_golden._load_snapshot` so
    all three consumers of the snapshots agree on the on-disk shape.
    """
    payload_path = _golden_dir() / repo_name / "repository_contents.json"
    with payload_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return RepositoryContents.model_validate(payload)


def _install_process_guards(
    monkeypatch: pytest.MonkeyPatch,
    invocations: list[tuple[str, tuple[Any, ...], dict[str, Any]]],
) -> None:
    """Replace every process-launching surface with a tripwire.

    Each replacement records the (qualified-name, args, kwargs) tuple in
    ``invocations`` and immediately raises :class:`AssertionError` so
    the offending call site shows up in the pytest traceback. The
    ``invocations`` list is a belt-and-braces audit trail: even if a
    caller swallows the :class:`AssertionError` we still see the
    forbidden call in the assertion message at the end of the test.

    Surfaces that do not exist on the host platform (POSIX-only helpers
    on Windows, ``forkpty`` on some platforms) are silently skipped.
    """

    def _make_tripwire(qualname: str) -> Any:
        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            invocations.append((qualname, args, kwargs))
            raise AssertionError(
                f"forbidden process-launch surface {qualname!r} called "
                f"with args={args!r} kwargs={kwargs!r}; the Go analyzer "
                f"must not invoke any external toolchain (Requirements "
                f"10.1, 10.2)"
            )

        return _forbidden

    for name in _SUBPROCESS_SURFACES:
        if not hasattr(subprocess, name):
            continue
        monkeypatch.setattr(
            subprocess,
            name,
            _make_tripwire(f"subprocess.{name}"),
            raising=True,
        )

    for name in _OS_SURFACES:
        if not hasattr(os, name):
            continue
        monkeypatch.setattr(
            os,
            name,
            _make_tripwire(f"os.{name}"),
            raising=True,
        )


@pytest.mark.parametrize("repo_name", SAMPLE_REPOS)
def test_analyze_invokes_no_subprocess(
    repo_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``analyze()`` against a sample-repo snapshot launches no subprocess.

    The Go layer is pure Python (design §1, §2 architecture decision):
    the tokenizer and recognizer read ``.go`` source in process and the
    four Go sub-scanners consume the resulting event stream. No call in
    the Go pipeline -- nor in any sub-analyzer the aggregator routes
    through for a Go repository -- shells out to ``go build``,
    ``go run``, ``gofmt``, or any other external binary.

    The test monkey-patches every process-launching surface in
    :mod:`subprocess` and :mod:`os` with a tripwire that raises
    :class:`AssertionError` on call and records the attempted call.
    With the guards installed it runs the analyzer with the same
    placeholder arguments the curate script uses, then asserts:

    1. ``analyze()`` returned a profile (it did not raise; the
       aggregator's outer ``try/except Exception`` cannot swallow
       :class:`AssertionError` because we install the patches at the
       module level where every importer sees them).
    2. The ``invocations`` audit trail is empty.

    Together these two assertions are the structural proof that
    Requirements 10.1 and 10.2 hold for every sample repository.
    """
    invocations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    _install_process_guards(monkeypatch, invocations)

    repo_contents = _load_snapshot(repo_name)

    # The placeholder arguments mirror ``_curate.curate_profile`` so the
    # call path through ``analyze()`` is identical to the one the
    # integration golden test exercises. If a subprocess were launched
    # on that path the integration test would still pass (it does not
    # patch anything) but this test would fail loudly here.
    profile = analyze(
        project_id=1,
        full_path=f"samples/{repo_name}",
        analysis_branch="main",
        commit_sha=repo_contents.commit_sha,
        repo_description=None,
        repository_contents=repo_contents,
    )

    # First assert the audit trail is empty -- this gives the most
    # informative failure message because it names every forbidden
    # surface that was touched, not just the first one.
    assert not invocations, (
        f"analyze() against {repo_name!r} invoked forbidden process-launch "
        f"surfaces: {invocations!r}. The Go analyzer must be pure Python "
        f"(Requirements 10.1, 10.2)."
    )

    # And confirm the analyzer actually ran end-to-end. A profile that
    # silently degrades because an internal sub-analyzer caught our
    # tripwire ``AssertionError`` would still fail the audit-trail
    # check above, but pinning the profile shape down here makes the
    # invariant explicit: the test is asserting "no subprocess AND
    # analyzer ran", not "no subprocess OR analyzer ran".
    assert profile.gitlab_project_id == 1
    assert profile.full_path == f"samples/{repo_name}"

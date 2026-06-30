"""Subprocess runner that hosts ``MCPServer`` over stdio.

Used by :mod:`tests.integration.test_mcp_stdio_handshake` as the
subprocess target. The script constructs a real
:class:`project_knowledge_mcp.mcp_server.MCPServer` instance with stub
collaborators and drives :meth:`MCPServer.serve` over the inheriting
process' stdin/stdout. The ``initialize`` handshake (Requirement 11.1)
never invokes the injected collaborators, so plain ``object()``
instances cast to the expected types are sufficient and keep the test
focused on the wire transport rather than the per-tool dispatch path.

The filename is intentionally prefixed with an underscore so pytest's
default test discovery (``test_*.py`` / ``*_test.py``) does not pick
it up as a test module.

This runner exists because the production process entry point
(``project_knowledge_mcp.main``, task 11.1) is not yet wired and
``project_knowledge_mcp.mcp_server`` does not have a ``__main__`` block
of its own. Once task 11.1 lands, the integration test can switch to
spawning the real entry point; the assertions on the response payload
do not change.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from project_knowledge_mcp.mcp_server import MCPServer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from project_knowledge_mcp.conflict_detector import ConflictPair
    from project_knowledge_mcp.ingestion_coordinator import IngestionCoordinator
    from project_knowledge_mcp.knowledge_store import KnowledgeStore
    from project_knowledge_mcp.models import ConflictResult, ProjectProfile
    from project_knowledge_mcp.project_catalog import ProjectCatalog


def _stub_classify_pair(
    profile_a: ProjectProfile,
    profile_b: ProjectProfile,
) -> ConflictResult:
    """Stand-in for :func:`conflict_detector.classify_pair`.

    The stdio handshake under test never reaches a tool that would
    call this collaborator. Raising on invocation surfaces any future
    regression that accidentally drives a conflict-classification code
    path during ``initialize``.
    """

    raise AssertionError(
        "classify_pair must not be called during the stdio handshake"
    )


def _stub_find_all_conflicts(
    profiles: Sequence[ProjectProfile],
) -> list[ConflictPair]:
    """Stand-in for :func:`conflict_detector.find_all_conflicts`.

    Raises on invocation for the same reason as :func:`_stub_classify_pair`:
    the ``initialize`` handshake never dispatches to the conflict
    detector, so any call here indicates a regression in the surface
    under test.
    """

    raise AssertionError(
        "find_all_conflicts must not be called during the stdio handshake"
    )


async def _run_server() -> None:
    """Construct ``MCPServer`` with stub collaborators and serve over stdio."""

    server = MCPServer(
        store=cast("KnowledgeStore", object()),
        catalog=cast("ProjectCatalog", object()),
        coordinator=cast("IngestionCoordinator", object()),
        classify_pair=_stub_classify_pair,
        find_all_conflicts=_stub_find_all_conflicts,
    )
    await server.serve()


def main() -> None:
    """Entry point for ``python <this file>``."""

    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

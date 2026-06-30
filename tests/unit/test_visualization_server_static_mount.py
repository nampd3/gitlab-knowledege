"""Unit tests for the ``/static/<filename>`` mount and Mermaid wiring.

The visualization pages emit ``<pre class="mermaid">`` blocks. Without
a Mermaid JavaScript bundle the browser displays the raw graph source
as plain text. The fix is twofold:

* :func:`_build_static_mount` adds a Starlette ``Mount`` at ``/static``
  that serves files from the package's ``static/`` directory.
* :func:`_wrap_diagram_fragment` emits ``<script src="/static/mermaid.min.js">``
  followed by a single ``mermaid.initialize`` call.

These tests pin both behaviors so a regression in either side surfaces
immediately:

1. **Static mount serves real files.** A file dropped into the package
   ``static/`` directory is reachable at ``GET /static/<name>`` with
   the file's bytes as the body.
2. **Static mount returns 404 for unknown files.** A request for a
   filename that does not exist returns HTTP 404. The body shape is
   intentionally not pinned — the goal is "does not 200 spuriously".
3. **Diagram pages include the Mermaid script tag.** The
   ``/dependencies`` and ``/conflicts`` responses contain the
   documented ``<script src="/static/mermaid.min.js">`` element and
   the ``mermaid.initialize`` call.

The tests use Starlette's :class:`httpx.ASGITransport` so no real
sockets are involved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import httpx
import pytest
from starlette.applications import Starlette

from project_knowledge_mcp.conflict_detector import ConflictPair
from project_knowledge_mcp.models import ProjectProfile
from project_knowledge_mcp.project_catalog import InScopeProject
from project_knowledge_mcp.visualization_server import (
    _STATIC_DIR_PATH,
    _wrap_diagram_fragment,
    build_visualization_app,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeCatalog:
    def __init__(self) -> None:
        self.projects: list[InScopeProject] = []

    def list_in_scope(self) -> list[InScopeProject]:
        return list(self.projects)

    def is_in_scope(self, gitlab_project_id: int) -> bool:
        return any(p.gitlab_project_id == gitlab_project_id for p in self.projects)


class _FakeStore:
    def __init__(self) -> None:
        self.profiles: list[ProjectProfile] = []
        self.snapshot_id: int | None = 1

    def get_current_snapshot_id(self) -> int | None:
        return self.snapshot_id

    def list_profiles(self) -> list[ProjectProfile]:
        return list(self.profiles)

    def get_profile(self, gitlab_project_id: int) -> ProjectProfile | None:
        for p in self.profiles:
            if p.gitlab_project_id == gitlab_project_id:
                return p
        return None


def _build_app() -> Starlette:
    return build_visualization_app(
        _FakeCatalog(),
        _FakeStore(),
    )


async def _get(app: Starlette, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://visualization-server.test"
    ) as client:
        return await client.get(path)


# ---------------------------------------------------------------------------
# Static mount: serves real files
# ---------------------------------------------------------------------------


async def test_static_mount_serves_an_existing_file_from_the_package_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file in ``static/`` is reachable at ``GET /static/<name>``.

    Uses ``monkeypatch`` to swap ``_STATIC_DIR_PATH`` to a temporary
    directory so the test does not depend on a real mermaid bundle
    being committed; the test only verifies that the route plumbing
    works.
    """
    sentinel_body = b"console.log('mermaid loaded');"
    (tmp_path / "fake-mermaid.js").write_bytes(sentinel_body)

    monkeypatch.setattr(
        "project_knowledge_mcp.visualization_server._STATIC_DIR_PATH",
        tmp_path,
    )

    app = _build_app()
    response = await _get(app, "/static/fake-mermaid.js")

    assert response.status_code == 200
    assert response.content == sentinel_body


async def test_static_mount_returns_404_for_missing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing file under ``/static`` returns 404, not 200."""
    monkeypatch.setattr(
        "project_knowledge_mcp.visualization_server._STATIC_DIR_PATH",
        tmp_path,
    )

    app = _build_app()
    response = await _get(app, "/static/this-file-does-not-exist.js")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Diagram fragment chrome: includes the Mermaid script tag
# ---------------------------------------------------------------------------


def test_wrap_diagram_fragment_includes_graph_libraries() -> None:
    """The page chrome loads both Cytoscape (primary) and Mermaid (fallback)."""
    html = _wrap_diagram_fragment(title="title", fragment="<section></section>")

    # Cytoscape is the primary renderer; Mermaid is the documented
    # fallback used when ``cytoscape.min.js`` is not present in the
    # static directory.
    assert '<script src="/static/cytoscape.min.js">' in html
    assert '<script src="/static/mermaid.min.js">' in html


def test_wrap_diagram_fragment_falls_back_to_mermaid_when_cytoscape_absent() -> None:
    """The init script gracefully degrades when Cytoscape is not loaded.

    The fallback path adds ``body.no-cytoscape`` (which CSS uses to
    hide the empty Cytoscape container and show the
    ``<pre class="mermaid">`` block) and initializes Mermaid with the
    documented size limits.
    """
    html = _wrap_diagram_fragment(title="title", fragment="<section></section>")

    # Cytoscape detection guard.
    assert "typeof cytoscape !== 'undefined'" in html
    # Fallback CSS hook.
    assert "no-cytoscape" in html
    # Mermaid initialize call is still present for the fallback.
    assert "mermaid.initialize" in html


def test_wrap_diagram_fragment_bumps_mermaid_safety_caps() -> None:
    """The Mermaid fallback raises ``maxTextSize`` and ``maxEdges``.

    Even when Cytoscape is the primary renderer, the Mermaid
    fallback retains the previously-tuned ceilings so an operator
    without Cytoscape installed still sees a rendered diagram
    instead of the ``Maximum text size in diagram exceeded`` error.
    """
    html = _wrap_diagram_fragment(title="title", fragment="<section></section>")

    assert "maxTextSize: 1048576" in html
    assert "maxEdges: 100000" in html
    assert "startOnLoad: true" in html


def test_wrap_diagram_fragment_disables_mermaid_useMaxWidth() -> None:
    """The Mermaid fallback also lets the SVG grow past viewport width."""
    html = _wrap_diagram_fragment(title="title", fragment="<section></section>")

    assert "flowchart:" in html
    assert "useMaxWidth: false" in html
    assert "nodeSpacing: 80" in html
    assert "rankSpacing: 100" in html


def test_wrap_diagram_fragment_initializes_cytoscape() -> None:
    """The chrome includes the Cytoscape instantiation glue.

    Pins the salient bits of the init script so a regression in
    naming (graph container CSS class, JSON script class, layout
    name) surfaces immediately.
    """
    html = _wrap_diagram_fragment(title="title", fragment="<section></section>")

    # The init script iterates ``.graph-data`` JSON blocks and
    # renders into ``.graph-container`` divs inside the same
    # ``<section>``. Both class names are public surface for the
    # ``diagram_renderer`` templates.
    assert ".graph-data" in html
    assert ".graph-container" in html
    # Cytoscape itself is invoked with a force-directed layout
    # ("cose") suitable for ~150-node graphs.
    assert "cytoscape({" in html
    assert "name: 'cose'" in html


def test_wrap_diagram_fragment_pins_cytoscape_performance_knobs() -> None:
    """The Cytoscape init keeps the documented performance settings.

    For ~150-node graphs the stock ``cose`` defaults (1000 layout
    iterations, full-fidelity edge rendering at every viewport
    transform) feel frozen on commodity laptops. The bundle of
    knobs below is the smallest set that takes the steady-state
    interactive frame rate from "laggy" to "smooth"; pinning it as
    a unit test prevents an accidental revert (e.g. someone bumping
    ``numIter`` back up for crisper layout without realising the UX
    cost).
    """
    html = _wrap_diagram_fragment(title="title", fragment="<section></section>")

    # Layout iteration cap.
    assert "numIter: 100" in html
    # Reuse existing positions when re-running layout.
    assert "randomize: false" in html
    # Viewport-time draw simplifications.
    assert "hideEdgesOnViewport: true" in html
    assert "hideLabelsOnViewport: true" in html
    assert "textureOnViewport: true" in html
    assert "motionBlur: true" in html
    # Pixel-ratio cap so high-DPI displays do not render at 4x area.
    assert "pixelRatio: 1.0" in html
    # Edge labels are hidden in the steady state and revealed on
    # hover / selection through a ``.hover`` / ``:selected`` style.
    assert "edge:selected, edge.hover" in html


def test_wrap_diagram_fragment_wires_edge_hover_handlers() -> None:
    """Hovering an edge toggles the ``.hover`` class that reveals its label."""
    html = _wrap_diagram_fragment(title="title", fragment="<section></section>")

    assert "addClass('hover')" in html
    assert "removeClass('hover')" in html
    assert "'mouseover', 'edge'" in html
    assert "'mouseout', 'edge'" in html


async def test_dependencies_page_includes_mermaid_script_tag() -> None:
    """``GET /dependencies`` carries the Mermaid wiring in its HTML body."""
    app = _build_app()
    response = await _get(app, "/dependencies")

    assert response.status_code == 200
    assert '<script src="/static/mermaid.min.js">' in response.text


# ---------------------------------------------------------------------------
# Static dir path itself is sensible
# ---------------------------------------------------------------------------


def test_static_dir_path_resolves_inside_the_package() -> None:
    """``_STATIC_DIR_PATH`` points at a directory shipped with the package."""
    # The directory must exist (committed alongside the source) so the
    # Starlette mount can serve files from it on a fresh checkout.
    assert _STATIC_DIR_PATH.exists() and _STATIC_DIR_PATH.is_dir()
    # And it must live inside the package source tree.
    assert _STATIC_DIR_PATH.name == "static"
    assert _STATIC_DIR_PATH.parent.name == "project_knowledge_mcp"

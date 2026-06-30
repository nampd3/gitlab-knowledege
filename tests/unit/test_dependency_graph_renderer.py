"""Unit tests for the ``Dependency_Graph_Diagram`` renderer.

These tests pin five behaviours of
:func:`project_knowledge_mcp.diagram_renderer.render_dependency_graph`:

1. A populated catalog with profiles that share a database table
   renders a ``"shared table: {table_name}"`` edge (the wording fixed
   by Requirement 13.3), each in-scope project appears as a Mermaid
   node, no empty-state message leaks into the populated branch, and
   shared external services do *not* contribute edges — the
   shared-external-service edge kind was retired by an operator-tuning
   decision to keep the dependency graph readable at ESB scale.
2. A populated catalog with no shared dependencies between Projects
   still renders the project nodes and surfaces the
   :data:`NO_SHARED_DEPENDENCIES_MESSAGE` empty-state message
   (Requirement 13.3's "still rendering project nodes" clause).
3. An empty catalog renders both
   :data:`EMPTY_CATALOG_MESSAGE` ("No Projects are in scope") and
   :data:`NO_SHARED_DEPENDENCIES_MESSAGE`, and emits no Mermaid block.
4. Profiles for ``gitlab_project_id`` values not in the catalog are
   ignored so the rendered node-set always equals ``catalog`` and no
   spurious edges to out-of-scope projects appear (Property 23's
   "node set equals the in-scope ``Project_Catalog``" clause).
5. The renderer is pure: equal inputs produce byte-for-byte equal
   output, regardless of catalog row order.

The tests use plain string-search assertions (``in`` rather than strict
equality on the entire HTML document) so they pin the *content
contract* required by Requirement 13.3 and Property 23 without
coupling to incidental whitespace or class-name choices. Empty-state
assertions, in contrast, use the exact module constants so a regression
in the literal wording is caught at unit-test time.

Implements Requirement 13.3.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from project_knowledge_mcp.diagram_renderer import (
    EMPTY_CATALOG_MESSAGE,
    NO_SHARED_DEPENDENCIES_MESSAGE,
    SHARED_TABLE_EDGE_LABEL_TEMPLATE,
    UNKNOWN_ACCESS_MODE_LABEL_SUFFIX,
    render_dependency_graph,
)
from project_knowledge_mcp.models import (
    DatabaseAccessMode,
    DatabaseTableDependency,
    ExternalServiceDependency,
    ExternalServiceKind,
    ProjectProfile,
    SourceLocation,
)
from project_knowledge_mcp.project_catalog import InScopeProject

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Profile / catalog fixtures
# ---------------------------------------------------------------------------
#
# Two helpers build small catalogs and matching profiles for the tests.
# ``_profile`` lifts the noisy ``ProjectProfile`` constructor (every
# field is required because the model is frozen and strict) into a
# single call that fills in the identity / branch / commit fields with
# stable defaults so each test only specifies what it actually exercises.


def _profile(
    *,
    gitlab_project_id: int,
    full_path: str,
    services: list[ExternalServiceDependency] | None = None,
    tables: list[DatabaseTableDependency] | None = None,
) -> ProjectProfile:
    """Build a minimal :class:`ProjectProfile` for the renderer tests."""
    return ProjectProfile(
        gitlab_project_id=gitlab_project_id,
        full_path=full_path,
        analysis_branch="uat",
        analysis_branch_commit_sha="cafef00d",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary=f"Profile for {full_path}.",
        external_service_dependencies=services or [],
        database_table_dependencies=tables or [],
    )


def _service(name: str, *, kind: ExternalServiceKind = ExternalServiceKind.HTTP_API,
             path: str = "src/clients.py", line: int | None = 1
             ) -> ExternalServiceDependency:
    """Build a single-source-location :class:`ExternalServiceDependency`."""
    return ExternalServiceDependency(
        name=name,
        kind=kind,
        source_locations=[SourceLocation(path=path, line=line)],
    )


def _table(name: str, *, mode: DatabaseAccessMode = DatabaseAccessMode.READ,
           path: str = "src/db.py", line: int | None = 1
           ) -> DatabaseTableDependency:
    """Build a single-source-location :class:`DatabaseTableDependency`."""
    return DatabaseTableDependency(
        table_name=name,
        access_mode=mode,
        source_locations=[SourceLocation(path=path, line=line)],
    )


# ---------------------------------------------------------------------------
# Populated catalog with shared dependencies (Requirement 13.3)
# ---------------------------------------------------------------------------


def test_populated_catalog_renders_nodes_and_shared_dependency_edges() -> None:
    """Two projects sharing a database table render the table edge.

    Verifies the populated branch of Requirement 13.3 after the
    operator-tuning decision to drop shared-external-service edges
    from the dependency graph:

    * Each in-scope project appears as a Mermaid node whose label
      contains the GitLab id and the full path.
    * The shared database table produces an edge labeled exactly
      ``"shared table: orders"`` (the wording fixed by Requirement
      13.3).
    * A shared external service does NOT produce an edge — the
      ``shared external service:`` label substring must be absent
      from the rendered HTML even when the two profiles list the
      same external service name.
    * Neither empty-state message leaks into the populated branch.
    * The Mermaid block is emitted (``<pre class="mermaid">``).
    """
    auth = InScopeProject(gitlab_project_id=7, full_path="acme/auth")
    payments = InScopeProject(gitlab_project_id=42, full_path="acme/payments")
    profiles = [
        _profile(
            gitlab_project_id=7,
            full_path="acme/auth",
            services=[_service("payments-api")],
            tables=[_table("orders")],
        ),
        _profile(
            gitlab_project_id=42,
            full_path="acme/payments",
            services=[_service("payments-api")],
            tables=[_table("orders", mode=DatabaseAccessMode.WRITE)],
        ),
    ]

    html = render_dependency_graph([auth, payments], profiles)

    # Mermaid block header is emitted (the populated branch, not the
    # empty-catalog branch).
    assert '<pre class="mermaid">' in html
    assert "graph LR" in html

    # Each in-scope project produces a Mermaid node whose label
    # contains both the GitLab id and the full path.
    assert "acme/auth" in html
    assert "acme/payments" in html
    assert "P7" in html
    assert "P42" in html

    # Shared external services no longer contribute edges to the
    # dependency graph — the operator-tuning decision retired this
    # edge kind because every ESB-scale microservice tends to
    # share a small set of shared infrastructure endpoints and the
    # resulting fully-connected hairball drowned out the more
    # discriminating shared-database-table edges.
    assert "shared external service:" not in html

    # The shared database table produces exactly one edge with the
    # wording fixed by Requirement 13.3.
    expected_table_edge_label = SHARED_TABLE_EDGE_LABEL_TEMPLATE.format(
        table_name="orders"
    )
    assert expected_table_edge_label == "shared table: orders"
    assert expected_table_edge_label in html

    # No empty-state message leaks into the populated, has-edges branch.
    assert NO_SHARED_DEPENDENCIES_MESSAGE not in html
    assert EMPTY_CATALOG_MESSAGE not in html


# ---------------------------------------------------------------------------
# Populated catalog with no shared dependencies (Requirement 13.3)
# ---------------------------------------------------------------------------


def test_populated_catalog_with_no_shared_dependencies_renders_empty_state() -> None:
    """Disjoint dependencies -> nodes are still rendered, plus empty-state.

    Verifies Requirement 13.3's "no two in-scope Projects share any
    Database_Table_Dependency" branch combined with the
    operator-tuning rule that prunes isolated nodes:

    * The Mermaid block is still emitted (``<pre class="mermaid">``,
      ``graph LR``) so the page has a deterministic shape, but
    * No node lines appear because every in-scope project is
      isolated and isolated nodes are pruned.
    * No edge label appears in the rendered HTML.
    * The :data:`NO_SHARED_DEPENDENCIES_MESSAGE` empty-state appears
      verbatim (and exactly once).
    * The :data:`EMPTY_CATALOG_MESSAGE` empty-state does NOT appear,
      because the catalog is non-empty.
    """
    auth = InScopeProject(gitlab_project_id=7, full_path="acme/auth")
    payments = InScopeProject(gitlab_project_id=42, full_path="acme/payments")
    profiles = [
        _profile(
            gitlab_project_id=7,
            full_path="acme/auth",
            services=[_service("auth-only-service")],
            tables=[_table("auth_only_table")],
        ),
        _profile(
            gitlab_project_id=42,
            full_path="acme/payments",
            services=[_service("payments-only-service")],
            tables=[_table("payments_only_table")],
        ),
    ]

    html = render_dependency_graph([auth, payments], profiles)

    assert '<pre class="mermaid">' in html
    assert "graph LR" in html
    # Isolated nodes are pruned by the operator-tuning rule: neither
    # project has a shared-table edge to the other, so neither is
    # rendered.
    assert "acme/auth" not in html
    assert "acme/payments" not in html
    # No edge label literal appears (no "shared X:" substrings).
    assert "shared external service:" not in html
    assert "shared table:" not in html
    # Empty-state for no shared deps is present, exactly once.
    assert NO_SHARED_DEPENDENCIES_MESSAGE in html
    assert html.count(NO_SHARED_DEPENDENCIES_MESSAGE) == 1
    # The catalog is non-empty, so the "No Projects are in scope"
    # empty-state must not be rendered.
    assert EMPTY_CATALOG_MESSAGE not in html


# ---------------------------------------------------------------------------
# Empty catalog (Requirement 13.3)
# ---------------------------------------------------------------------------


def test_empty_catalog_renders_no_projects_and_no_shared_dependencies_messages() -> None:
    """Empty catalog -> both empty-state messages, no Mermaid block.

    Verifies Requirement 13.3's "no in-scope Projects" branch:

    * The :data:`EMPTY_CATALOG_MESSAGE` ("No Projects are in scope")
      empty-state appears.
    * The :data:`NO_SHARED_DEPENDENCIES_MESSAGE` empty-state also
      appears, because Requirement 13.3 says the message is included
      whenever no two Projects share a dependency, which is trivially
      true when there are no Projects.
    * No Mermaid block is emitted (no ``<pre class="mermaid">``,
      no ``graph LR``).
    """
    html = render_dependency_graph([], [])

    assert EMPTY_CATALOG_MESSAGE in html
    assert NO_SHARED_DEPENDENCIES_MESSAGE in html
    assert '<pre class="mermaid">' not in html
    assert "graph LR" not in html


# ---------------------------------------------------------------------------
# Out-of-scope profiles are ignored (Property 23 node-set clause)
# ---------------------------------------------------------------------------


def test_profiles_for_projects_outside_catalog_are_ignored() -> None:
    """Profiles whose ``gitlab_project_id`` is not in ``catalog`` are ignored.

    A profile for a project that is not in the catalog must not
    appear as a node and must not contribute any edges, even if
    it shares a dependency with an in-scope project. With a
    single-project catalog and no in-scope peer, the operator-
    tuning node prune leaves the diagram with no nodes at all.
    """
    in_scope = InScopeProject(gitlab_project_id=7, full_path="acme/auth")
    profiles = [
        _profile(
            gitlab_project_id=7,
            full_path="acme/auth",
            tables=[_table("orders")],
        ),
        # gitlab_project_id 99 is NOT in the catalog. It shares a
        # table name with the in-scope project, but no edge should
        # be drawn because the node 99 doesn't exist in the diagram.
        _profile(
            gitlab_project_id=99,
            full_path="not/in/scope",
            tables=[_table("orders")],
        ),
    ]

    html = render_dependency_graph([in_scope], profiles)

    # The out-of-scope project must not appear anywhere.
    assert "not/in/scope" not in html
    assert "P99" not in html
    # The in-scope project is isolated (no in-scope peer to share
    # tables with) so the node-prune drops it from the rendered
    # graph as well.
    assert "P7" not in html
    assert "acme/auth" not in html
    # No edge is drawn because there's only one in-scope project.
    assert "shared external service:" not in html
    assert "shared table:" not in html
    # Single-node catalog has no shared deps -> empty-state message.
    assert NO_SHARED_DEPENDENCIES_MESSAGE in html


# ---------------------------------------------------------------------------
# Determinism / purity
# ---------------------------------------------------------------------------


def test_renderer_is_pure_and_deterministic_under_input_reordering() -> None:
    """Equal inputs -> equal output, regardless of catalog ordering.

    The renderer is documented as pure (no I/O, no global state, no
    caching). Property 23 also asserts byte-for-byte equality between
    two calls with structurally-equal inputs. This test exercises
    both: a catalog supplied in two different orders produces
    byte-for-byte equal HTML, because the renderer sorts by
    ``gitlab_project_id`` ascending before emitting Mermaid.
    """
    auth = InScopeProject(gitlab_project_id=7, full_path="acme/auth")
    payments = InScopeProject(gitlab_project_id=42, full_path="acme/payments")
    profiles = [
        _profile(
            gitlab_project_id=7,
            full_path="acme/auth",
            tables=[_table("orders")],
        ),
        _profile(
            gitlab_project_id=42,
            full_path="acme/payments",
            tables=[_table("orders")],
        ),
    ]

    forward = render_dependency_graph([auth, payments], profiles)
    reverse = render_dependency_graph([payments, auth], list(reversed(profiles)))

    assert forward == reverse
    # The single shared-table edge label appears exactly twice:
    # once in the inline Mermaid source (the documented fallback
    # used when ``cytoscape.min.js`` is not present) and once in
    # the JSON payload Cytoscape consumes. Both must encode the
    # same edge set so the two renderers cannot disagree on what
    # the snapshot shows.
    expected_edge = SHARED_TABLE_EDGE_LABEL_TEMPLATE.format(table_name="orders")
    assert forward.count(expected_edge) == 2


# ---------------------------------------------------------------------------
# Unknown access mode rendering (Go-Analyzer-Support Requirement 9.8)
# ---------------------------------------------------------------------------


def test_shared_table_with_unknown_access_mode_appends_suffix() -> None:
    """Shared-table edges flag the unknown access mode in the edge label.

    Validates Go-Analyzer-Support Requirement 9.8: when two in-scope
    projects share a database table and at least one of the two
    profiles records the table with
    :data:`DatabaseAccessMode.UNKNOWN`, the edge label appends the
    documented :data:`UNKNOWN_ACCESS_MODE_LABEL_SUFFIX` so a human
    reading the diagram sees that the access mode was indeterminate
    on at least one side. A second shared table whose access mode is
    determinate on both sides keeps its plain label so the suffix is
    unambiguously the indeterminate-access marker.
    """
    auth = InScopeProject(gitlab_project_id=7, full_path="acme/auth")
    payments = InScopeProject(gitlab_project_id=42, full_path="acme/payments")
    profiles = [
        _profile(
            gitlab_project_id=7,
            full_path="acme/auth",
            tables=[
                _table("REPAYMENT_TXN", mode=DatabaseAccessMode.UNKNOWN),
                _table("orders", mode=DatabaseAccessMode.READ),
            ],
        ),
        _profile(
            gitlab_project_id=42,
            full_path="acme/payments",
            tables=[
                _table("REPAYMENT_TXN", mode=DatabaseAccessMode.READ),
                _table("orders", mode=DatabaseAccessMode.WRITE),
            ],
        ),
    ]

    html = render_dependency_graph([auth, payments], profiles)

    # The shared "REPAYMENT_TXN" table is recorded with UNKNOWN access
    # mode in one profile -> the edge label is suffixed with the
    # documented "[access: unknown]" marker.
    plain_repayment_label = SHARED_TABLE_EDGE_LABEL_TEMPLATE.format(
        table_name="REPAYMENT_TXN"
    )
    expected_repayment_label = (
        f"{plain_repayment_label}{UNKNOWN_ACCESS_MODE_LABEL_SUFFIX}"
    )
    assert expected_repayment_label in html

    # The shared "orders" table is determinate on both sides -> its
    # edge label keeps the plain wording (no suffix).
    plain_orders_label = SHARED_TABLE_EDGE_LABEL_TEMPLATE.format(
        table_name="orders"
    )
    assert plain_orders_label in html
    assert f"{plain_orders_label}{UNKNOWN_ACCESS_MODE_LABEL_SUFFIX}" not in html

    # Pin the literal so a regression in the suffix wording is caught
    # alongside the structural assertion.
    assert UNKNOWN_ACCESS_MODE_LABEL_SUFFIX == " [access: unknown]"

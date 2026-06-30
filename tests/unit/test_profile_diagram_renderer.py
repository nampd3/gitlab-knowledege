"""Unit tests for the ``Project_Profile_Diagram`` renderer.

These tests pin three behaviours of
:func:`project_knowledge_mcp.diagram_renderer.render_profile_diagram`:

1. A populated profile renders the four documented sections
   (``Abstract_Inputs``, ``Abstract_Outputs``, ``External_Service_Dependencies``,
   ``Database_Table_Dependencies``), grouped or labelled per Requirement 13.2,
   with each entry's user-visible content reaching the rendered HTML.
2. Empty sections render the section-specific empty-state message named
   for that section (Requirement 13.6's "for example, 'No Abstract Inputs
   detected'" wording, generalized to one message per section).
3. The renderer is pure: equal inputs produce byte-for-byte equal output.

The tests are deliberately string-search assertions (``in`` rather than
strict equality on the whole HTML document) so they pin the *content
contract* without coupling to incidental whitespace or class-name choices.
The empty-state assertions, in contrast, use the exact constant strings
exported by the renderer so a regression in the literal wording is caught
at unit-test time.

Implements Requirements 13.2 and 13.6.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from project_knowledge_mcp.diagram_renderer import (
    EMPTY_ABSTRACT_INPUTS_MESSAGE,
    EMPTY_ABSTRACT_OUTPUTS_MESSAGE,
    EMPTY_DATABASE_TABLES_MESSAGE,
    EMPTY_EXTERNAL_SERVICES_MESSAGE,
    UNKNOWN_ACCESS_MODE_LABEL_SUFFIX,
    render_profile_diagram,
)
from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
    DatabaseAccessMode,
    DatabaseTableDependency,
    ExternalServiceDependency,
    ExternalServiceKind,
    ProjectProfile,
    SourceLocation,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Profile fixtures
# ---------------------------------------------------------------------------
#
# Two fixtures: a richly-populated profile that exercises every section,
# and a deliberately empty profile that triggers all four empty-state
# messages. Both share the same identifying metadata so test failures
# point at the renderer rather than at fixture mismatches.


def _populated_profile() -> ProjectProfile:
    """Build a profile that exercises every diagram section.

    Includes:

    * One Abstract_Input each in two distinct categories so the renderer
      must produce two separate input groups.
    * Two Abstract_Outputs sharing a single category so the renderer
      must collapse them into one group with two ``<li>`` entries.
    * Two External_Service_Dependencies of different kinds, plus one
      with multiple source locations, exercising both the kind grouping
      and the per-entry source-locations list.
    * Two Database_Table_Dependencies in different access modes.
    """
    return ProjectProfile(
        gitlab_project_id=42,
        full_path="acme/widgets",
        analysis_branch="uat",
        analysis_branch_commit_sha="cafef00d",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary="Generates and bills widget orders for the demo portfolio.",
        abstract_inputs=[
            AbstractInput(
                category=AbstractInputCategory.HTTP_REQUEST,
                description="POST /orders body containing widget specifications",
            ),
            AbstractInput(
                category=AbstractInputCategory.SCHEDULED_EVENT,
                description="Nightly billing cron at 02:00 UTC",
            ),
        ],
        abstract_outputs=[
            AbstractOutput(
                category=AbstractOutputCategory.MESSAGE_PUBLISHED,
                description="OrderCreated message on the orders.created topic",
            ),
            AbstractOutput(
                category=AbstractOutputCategory.MESSAGE_PUBLISHED,
                description="OrderBilled message on the orders.billed topic",
            ),
        ],
        external_service_dependencies=[
            ExternalServiceDependency(
                name="payments-api",
                kind=ExternalServiceKind.HTTP_API,
                source_locations=[
                    SourceLocation(path="src/billing.py", line=12),
                    SourceLocation(path="src/billing.py", line=87),
                ],
            ),
            ExternalServiceDependency(
                name="orders-broker",
                kind=ExternalServiceKind.MESSAGE_BROKER,
                source_locations=[SourceLocation(path="src/messaging.py")],
            ),
        ],
        database_table_dependencies=[
            DatabaseTableDependency(
                table_name="orders",
                access_mode=DatabaseAccessMode.READ_WRITE,
                source_locations=[SourceLocation(path="src/db/orders.py", line=4)],
            ),
            DatabaseTableDependency(
                table_name="audit_log",
                access_mode=DatabaseAccessMode.WRITE,
                source_locations=[SourceLocation(path="src/db/audit.py", line=22)],
            ),
        ],
    )


def _empty_sections_profile() -> ProjectProfile:
    """Build a profile whose four content sections are all empty.

    The purpose summary is still required (it is non-optional on
    ``ProjectProfile``), but every section the renderer enumerates is
    deliberately empty so the rendered HTML must surface the
    section-specific empty-state messages.
    """
    return ProjectProfile(
        gitlab_project_id=7,
        full_path="acme/empty",
        analysis_branch="uat",
        analysis_branch_commit_sha="0000abcd",
        produced_at=datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC),
        purpose_summary="A project with no detected I/O or dependencies.",
    )


# ---------------------------------------------------------------------------
# Populated-profile content
# ---------------------------------------------------------------------------


def test_populated_profile_renders_all_documented_sections() -> None:
    """A fully-populated profile produces every section's content.

    Asserts the four sections each contain their expected category /
    kind / access-mode labels, that each entry's user-visible
    description (or service name / table name) appears in the output,
    and that the purpose summary is rendered verbatim. This pins the
    Requirement 13.2 content contract without coupling to the exact
    HTML structure.
    """
    profile = _populated_profile()

    html = render_profile_diagram(profile)

    # The purpose summary always renders verbatim.
    assert profile.purpose_summary in html

    # Section titles for all four content sections appear.
    assert "Abstract Inputs" in html
    assert "Abstract Outputs" in html
    assert "External Service Dependencies" in html
    assert "Database Table Dependencies" in html

    # Every Abstract_Input description is present, and the two distinct
    # categories produce two distinct group labels.
    for entry in profile.abstract_inputs:
        assert entry.description in html
    assert AbstractInputCategory.HTTP_REQUEST.value in html
    assert AbstractInputCategory.SCHEDULED_EVENT.value in html

    # Both Abstract_Output descriptions render even though they share a
    # single category (which also appears).
    for entry in profile.abstract_outputs:
        assert entry.description in html
    assert AbstractOutputCategory.MESSAGE_PUBLISHED.value in html

    # External services render their name and their kind label.
    for service in profile.external_service_dependencies:
        assert service.name in html
        assert service.kind.value in html
        for loc in service.source_locations:
            assert loc.path in html

    # Database tables render their table_name and access_mode label.
    for table in profile.database_table_dependencies:
        assert table.table_name in html
        assert table.access_mode.value in html
        for loc in table.source_locations:
            assert loc.path in html

    # No empty-state message leaks into a populated section.
    assert EMPTY_ABSTRACT_INPUTS_MESSAGE not in html
    assert EMPTY_ABSTRACT_OUTPUTS_MESSAGE not in html
    assert EMPTY_EXTERNAL_SERVICES_MESSAGE not in html
    assert EMPTY_DATABASE_TABLES_MESSAGE not in html


# ---------------------------------------------------------------------------
# Empty-section behaviour (Requirement 13.6)
# ---------------------------------------------------------------------------


def test_empty_sections_render_section_specific_empty_state_messages() -> None:
    """Each empty section renders its named empty-state message.

    Requirement 13.2 requires that "any of {Abstract_Inputs,
    Abstract_Outputs, External_Service_Dependencies,
    Database_Table_Dependencies} that is empty for the Project is
    rendered with a visible empty-state message naming that section
    (for example, 'No Abstract Inputs detected')". This test exercises
    the four-empty-section profile and asserts each named message
    appears literally in the rendered HTML.

    Implements Requirement 13.6.
    """
    profile = _empty_sections_profile()

    html = render_profile_diagram(profile)

    # The four documented messages all appear, each naming its section.
    assert EMPTY_ABSTRACT_INPUTS_MESSAGE in html
    assert EMPTY_ABSTRACT_OUTPUTS_MESSAGE in html
    assert EMPTY_EXTERNAL_SERVICES_MESSAGE in html
    assert EMPTY_DATABASE_TABLES_MESSAGE in html

    # Sanity: the worked example from Requirement 13.2 is the literal
    # value of one of the constants. Pin it here so a future rename of
    # the constant that happens to keep all imports working but break
    # the user-visible wording fails this test.
    assert EMPTY_ABSTRACT_INPUTS_MESSAGE == "No Abstract Inputs detected"

    # The purpose summary still renders for an otherwise-empty profile.
    assert profile.purpose_summary in html


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_renderer_is_pure() -> None:
    """Calling the renderer twice with equal inputs returns equal HTML.

    The renderer is documented as pure (no I/O, no global mutable
    state, no caching). That property is load-bearing for task 10.6's
    route handler, which calls the renderer per request without any
    coordination. The simplest evidence of purity is that two
    independent calls with structurally-equal inputs produce
    byte-for-byte equal output.
    """
    first = render_profile_diagram(_populated_profile())
    second = render_profile_diagram(_populated_profile())

    assert first == second


# ---------------------------------------------------------------------------
# Unknown access mode rendering (Go-Analyzer-Support Requirement 9.8)
# ---------------------------------------------------------------------------


def _profile_with_unknown_access_mode_table() -> ProjectProfile:
    """Build a profile carrying a Database_Table_Dependency with unknown access.

    This shape is produced by the Go analyzer when it identifies a table
    reference inside a ``QueryString`` whose surrounding SQL keyword does
    not match the recognized ``SELECT`` / ``INSERT`` / ``UPDATE`` /
    ``DELETE`` / ``MERGE`` / ``CREATE TABLE`` set (Go-Analyzer-Support
    Requirement 9.8). The renderer must surface the indeterminate access
    mode by appending the documented
    :data:`UNKNOWN_ACCESS_MODE_LABEL_SUFFIX` to the table label.
    """
    return ProjectProfile(
        gitlab_project_id=99,
        full_path="acme/repayments",
        analysis_branch="uat",
        analysis_branch_commit_sha="cafef00d",
        produced_at=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        purpose_summary="Repayment service with an indeterminate-access query.",
        database_table_dependencies=[
            DatabaseTableDependency(
                table_name="REPAYMENT_TXN",
                access_mode=DatabaseAccessMode.UNKNOWN,
                source_locations=[
                    SourceLocation(path="internal/usecase/repayment.go", line=42)
                ],
            ),
            DatabaseTableDependency(
                table_name="orders",
                access_mode=DatabaseAccessMode.READ,
                source_locations=[SourceLocation(path="internal/db/orders.go", line=7)],
            ),
        ],
    )


def test_unknown_access_mode_table_is_labeled_explicitly() -> None:
    """A table with ``UNKNOWN`` access mode is rendered with the explicit label.

    Validates Go-Analyzer-Support Requirement 9.8: when a
    :class:`DatabaseTableDependency` carries
    :data:`DatabaseAccessMode.UNKNOWN`, the rendered table entry
    appends the documented
    :data:`UNKNOWN_ACCESS_MODE_LABEL_SUFFIX` ("``[access: unknown]``")
    to the table name so a human reading the diagram sees that the
    access mode was indeterminate. Tables with the recognized
    ``READ`` / ``WRITE`` / ``READ_WRITE`` modes in the same profile
    keep their plain label so the suffix is unambiguously the
    indeterminate-access marker.
    """
    profile = _profile_with_unknown_access_mode_table()

    html = render_profile_diagram(profile)

    # The unknown-access-mode table renders with the suffixed label
    # "REPAYMENT_TXN [access: unknown]".
    expected_unknown_label = f"REPAYMENT_TXN{UNKNOWN_ACCESS_MODE_LABEL_SUFFIX}"
    assert expected_unknown_label in html
    # The suffix wording matches the example from the parent design
    # document so a regression in the literal is caught here.
    assert UNKNOWN_ACCESS_MODE_LABEL_SUFFIX == " [access: unknown]"

    # The READ-access table renders with no suffix so the marker is
    # unambiguous.
    assert "orders" in html
    assert f"orders{UNKNOWN_ACCESS_MODE_LABEL_SUFFIX}" not in html

    # The grouping section header still names the access mode so the
    # rendered diagram remains structurally consistent with the other
    # access-mode groups.
    assert DatabaseAccessMode.UNKNOWN.value in html

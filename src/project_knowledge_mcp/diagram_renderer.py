"""Diagram Renderer: pure functions that turn persisted Project_Profile data
into HTML fragments served by the Visualization_Server.

This module is intentionally pure: every rendering function is a referentially
transparent transformation from value objects (``ProjectProfile``,
``Project_Catalog`` listings, ``ConflictResult`` pairs) to a ``str`` of HTML.
There is no I/O, no global mutable state, and no caching. That property is
load-bearing for two parts of the design:

* The Visualization_Server route handlers (tasks 10.5-10.7) call these
  renderers directly while holding a freshly-read snapshot from the
  ``Knowledge_Store``. Property 11 forbids the visualization surface from
  caching ``Project_Profile`` data; keeping the renderers pure makes that
  property a property of the *call site* rather than something the renderer
  itself has to police.
* The property-based tests (tasks 10.13-10.16) drive the renderers directly
  with Hypothesis-generated profiles. A pure function can be exercised at
  full ``max_examples=100`` without spinning up an HTTP server.

This file ships task 10.2's ``Project_Profile_Diagram`` renderer
(``render_profile_diagram``). Tasks 10.3 and 10.4 will add the
``Dependency_Graph_Diagram`` and ``Conflict_Overview_Diagram`` renderers in
this same module.

Implements Requirements 13.2 and 13.6 (the *Project_Profile_Diagram* shape).
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING, Final

from jinja2 import Environment, StrictUndefined, select_autoescape

from .models import (
    AbstractInputCategory,
    AbstractOutputCategory,
    DatabaseAccessMode,
    ExternalServiceKind,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .models import (
        AbstractInput,
        AbstractOutput,
        DatabaseTableDependency,
        ExternalServiceDependency,
        ProjectProfile,
    )
    from .project_catalog import InScopeProject


# ---------------------------------------------------------------------------
# Section labels and empty-state messages (Requirements 13.2, 13.6)
# ---------------------------------------------------------------------------
#
# Section titles and empty-state messages are kept in module-level constants
# rather than in the template body so:
#
# * Property tests (task 10.14) and unit tests below can import the exact
#   strings instead of duplicating brittle literals.
# * The phrase "No <Section> detected" is fixed by Requirement 13.2's example
#   ("No Abstract Inputs detected"). Each section's empty-state message uses
#   the same shape so the user-visible behaviour is uniform.

#: Display title for the Abstract_Inputs section of the profile diagram.
ABSTRACT_INPUTS_SECTION_TITLE: Final[str] = "Abstract Inputs"

#: Display title for the Abstract_Outputs section of the profile diagram.
ABSTRACT_OUTPUTS_SECTION_TITLE: Final[str] = "Abstract Outputs"

#: Display title for the External_Service_Dependencies section.
EXTERNAL_SERVICES_SECTION_TITLE: Final[str] = "External Service Dependencies"

#: Display title for the Database_Table_Dependencies section.
DATABASE_TABLES_SECTION_TITLE: Final[str] = "Database Table Dependencies"

#: Empty-state message for an empty Abstract_Inputs section. The wording is
#: fixed by Requirement 13.2 and is reproduced verbatim by the unit tests so
#: a regression in the literal is caught at unit-test time.
EMPTY_ABSTRACT_INPUTS_MESSAGE: Final[str] = f"No {ABSTRACT_INPUTS_SECTION_TITLE} detected"

#: Empty-state message for an empty Abstract_Outputs section.
EMPTY_ABSTRACT_OUTPUTS_MESSAGE: Final[str] = f"No {ABSTRACT_OUTPUTS_SECTION_TITLE} detected"

#: Empty-state message for an empty External_Service_Dependencies section.
EMPTY_EXTERNAL_SERVICES_MESSAGE: Final[str] = f"No {EXTERNAL_SERVICES_SECTION_TITLE} detected"

#: Empty-state message for an empty Database_Table_Dependencies section.
EMPTY_DATABASE_TABLES_MESSAGE: Final[str] = f"No {DATABASE_TABLES_SECTION_TITLE} detected"

#: Sentinel substring appended to a database-table label whenever the
#: associated ``DatabaseAccessMode`` carries the ``"unknown"`` value
#: (Requirement 9.8 of the Go-Analyzer-Support spec / open-question 6
#: of the parent spec). The label is appended to per-entry table names
#: in the ``Project_Profile_Diagram`` and to ``shared table: …`` edge
#: labels in the ``Dependency_Graph_Diagram`` so a human reading the
#: rendered diagram sees that the access mode was indeterminate. The
#: literal wording matches the example in the parent design document
#: ("table: REPAYMENT_TXN [access: unknown]"). Comparison against the
#: enum is performed via the underlying string value (a
#: :class:`StrEnum`) so the recognition is robust to additional enum
#: members beyond the four currently defined.
UNKNOWN_ACCESS_MODE_VALUE: Final[str] = "unknown"

#: Suffix appended to a table label whose access mode is
#: :data:`UNKNOWN_ACCESS_MODE_VALUE`. Held as a module constant so the
#: unit and property tests for the two affected diagrams can import
#: the exact literal rather than duplicating it.
UNKNOWN_ACCESS_MODE_LABEL_SUFFIX: Final[str] = " [access: unknown]"


def _format_table_label(table_name: str, access_mode_value: str) -> str:
    """Return a table display label, suffixed when access mode is unknown.

    ``access_mode_value`` is the underlying string value of a
    :class:`DatabaseAccessMode` (a :class:`StrEnum`), passed in as a
    plain ``str`` so the comparison degrades gracefully if the enum
    grows additional members in the future. Returns ``table_name``
    unchanged for any access mode other than ``"unknown"``; otherwise
    returns ``f"{table_name}[access: unknown]"`` with the documented
    suffix from :data:`UNKNOWN_ACCESS_MODE_LABEL_SUFFIX`.
    """
    if access_mode_value == UNKNOWN_ACCESS_MODE_VALUE:
        return f"{table_name}{UNKNOWN_ACCESS_MODE_LABEL_SUFFIX}"
    return table_name


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------
#
# A single module-level ``Environment`` is shared across calls because:
#
# * Compiling templates once and reusing them is the documented Jinja2
#   pattern; rebuilding the environment per call would be both slower and
#   technically a side effect (template-cache pollution) that breaks the
#   "pure function" framing.
# * ``autoescape`` is enabled so that arbitrary user-controlled content
#   (purpose summaries, descriptions, file paths) cannot inject HTML or
#   <script> tags into the rendered diagram. The ingestion path does not
#   sanitize these strings; the renderer is the trust boundary.
# * ``StrictUndefined`` makes any reference to a missing template variable
#   raise instead of silently rendering an empty string. That keeps test
#   failures localized to the misspelled variable rather than producing
#   plausible-but-wrong HTML.
# * ``trim_blocks`` and ``lstrip_blocks`` keep the rendered HTML tidy
#   (no stray blank lines from ``{% if %}`` blocks) so that string-search
#   assertions in the tests are unambiguous.

_JINJA_ENV: Final[Environment] = Environment(
    autoescape=select_autoescape(default_for_string=True, default=True),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


# ---------------------------------------------------------------------------
# Project_Profile_Diagram template
# ---------------------------------------------------------------------------
#
# The template renders a self-contained ``<section>`` HTML fragment so that
# task 10.6's ``GET /projects/{id}`` route handler can drop the result inside
# the surrounding page chrome (header, navigation, etc.). Keeping the diagram
# as a fragment also means a future task can render two diagrams onto the
# same page without nested <html> elements.
#
# The four content sections (inputs, outputs, external services, db tables)
# render their entries grouped by category / kind / access_mode. The renderer
# does the grouping in Python before passing the data to the template, so the
# template only iterates pre-computed lists; this keeps the template free of
# Python-only constructs (``itertools.groupby``) and makes the rendering
# deterministic regardless of the input list order.

_PROFILE_DIAGRAM_TEMPLATE_SOURCE: Final[str] = """\
<section class="project-profile-diagram" data-project-id="{{ profile.gitlab_project_id }}">
  <header class="profile-header">
    <h1>Project Profile: {{ profile.full_path }}</h1>
    <dl class="profile-metadata">
      <dt>GitLab project ID</dt><dd>{{ profile.gitlab_project_id }}</dd>
      <dt>Analysis branch</dt><dd>{{ profile.analysis_branch }}</dd>
      <dt>Commit SHA</dt><dd>{{ profile.analysis_branch_commit_sha }}</dd>
      <dt>Produced at</dt><dd>{{ profile.produced_at.isoformat() }}</dd>
    </dl>
  </header>

  <section class="purpose-summary">
    <h2>Purpose</h2>
    <p class="purpose-summary-text">{{ profile.purpose_summary }}</p>
    {% if profile.purpose_summary_reason %}
    <p class="purpose-summary-reason">Reason: {{ profile.purpose_summary_reason }}</p>
    {% endif %}
  </section>

  <section class="abstract-inputs" data-section="abstract_inputs">
    <h2>{{ inputs_title }}</h2>
    {% if input_groups %}
    {% for group in input_groups %}
    <section class="abstract-input-group" data-category="{{ group.category }}">
      <h3>{{ group.category }}</h3>
      <ul>
        {% for item in group.entries %}
        <li>{{ item.description }}</li>
        {% endfor %}
      </ul>
    </section>
    {% endfor %}
    {% else %}
    <p class="empty-state" data-empty-section="abstract_inputs">{{ empty_inputs_message }}</p>
    {% endif %}
  </section>

  <section class="abstract-outputs" data-section="abstract_outputs">
    <h2>{{ outputs_title }}</h2>
    {% if output_groups %}
    {% for group in output_groups %}
    <section class="abstract-output-group" data-category="{{ group.category }}">
      <h3>{{ group.category }}</h3>
      <ul>
        {% for item in group.entries %}
        <li>{{ item.description }}</li>
        {% endfor %}
      </ul>
    </section>
    {% endfor %}
    {% else %}
    <p class="empty-state" data-empty-section="abstract_outputs">{{ empty_outputs_message }}</p>
    {% endif %}
  </section>

  <section class="external-service-dependencies" data-section="external_service_dependencies">
    <h2>{{ services_title }}</h2>
    {% if service_groups %}
    {% for group in service_groups %}
    <section class="external-service-group" data-kind="{{ group.kind }}">
      <h3>{{ group.kind }}</h3>
      <ul>
        {% for service in group.entries %}
        <li>
          <span class="service-name">{{ service.name }}</span>
          <ul class="source-locations">
            {% for loc in service.source_locations %}
            <li>{{ loc.path }}{% if loc.line is not none %}:{{ loc.line }}{% endif %}</li>
            {% endfor %}
          </ul>
        </li>
        {% endfor %}
      </ul>
    </section>
    {% endfor %}
    {% else %}
    <p class="empty-state" data-empty-section="external_service_dependencies">\
{{ empty_services_message }}</p>
    {% endif %}
  </section>

  <section class="database-table-dependencies" data-section="database_table_dependencies">
    <h2>{{ tables_title }}</h2>
    {% if table_groups %}
    {% for group in table_groups %}
    <section class="database-table-group" data-access-mode="{{ group.access_mode }}">
      <h3>{{ group.access_mode }}</h3>
      <ul>
        {% for table in group.entries %}
        <li>
          <span class="table-name">{{ table.display_label }}</span>
          <ul class="source-locations">
            {% for loc in table.source_locations %}
            <li>{{ loc.path }}{% if loc.line is not none %}:{{ loc.line }}{% endif %}</li>
            {% endfor %}
          </ul>
        </li>
        {% endfor %}
      </ul>
    </section>
    {% endfor %}
    {% else %}
    <p class="empty-state" data-empty-section="database_table_dependencies">\
{{ empty_tables_message }}</p>
    {% endif %}
  </section>
</section>
"""

#: Compiled Jinja2 template for the Project_Profile_Diagram. Compiled once at
#: import time so :func:`render_profile_diagram` is a pure call: no parsing,
#: no environment mutation, no template-cache lookup at call time.
_PROFILE_DIAGRAM_TEMPLATE = _JINJA_ENV.from_string(_PROFILE_DIAGRAM_TEMPLATE_SOURCE)


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------
#
# Grouping is performed in Python (not in the Jinja2 template) for three
# reasons:
#
# * Determinism. We sort group keys by their canonical declaration order in
#   the closed-set enums (``AbstractInputCategory``, ``ExternalServiceKind``,
#   etc.) so the rendered HTML is byte-for-byte stable across calls. A
#   template-side ``groupby`` would order groups by first appearance, which
#   leaks input ordering into the output and makes property tests flaky.
# * Type safety. The grouping code can be type-checked under ``mypy --strict``;
#   Jinja2 templates cannot.
# * Readability. The template body is reduced to "iterate, render entries"
#   loops with no branching.


def _group_inputs_by_category(
    inputs: Sequence[AbstractInput],
) -> list[dict[str, object]]:
    """Group ``Abstract_Input`` entries by category, in canonical enum order.

    Returns one ``{"category": str, "entries": list[AbstractInput]}`` dict per
    category that has at least one entry, in the declaration order of
    :class:`AbstractInputCategory`. Categories with zero entries are omitted
    so the template's per-group ``<h3>`` does not render an empty group.

    The dict key is named ``"entries"`` rather than ``"items"`` because
    Jinja2's attribute-lookup chain (``getattr`` before ``__getitem__``)
    would otherwise resolve ``group.items`` to the bound ``dict.items``
    method, breaking the template's ``{% for item in group.entries %}``
    loop.
    """
    by_category: dict[AbstractInputCategory, list[AbstractInput]] = {
        cat: [] for cat in AbstractInputCategory
    }
    for entry in inputs:
        by_category[entry.category].append(entry)
    return [
        {"category": cat.value, "entries": entries}
        for cat, entries in by_category.items()
        if entries
    ]


def _group_outputs_by_category(
    outputs: Sequence[AbstractOutput],
) -> list[dict[str, object]]:
    """Group ``Abstract_Output`` entries by category, in canonical enum order.

    See :func:`_group_inputs_by_category` for the rationale.
    """
    by_category: dict[AbstractOutputCategory, list[AbstractOutput]] = {
        cat: [] for cat in AbstractOutputCategory
    }
    for entry in outputs:
        by_category[entry.category].append(entry)
    return [
        {"category": cat.value, "entries": entries}
        for cat, entries in by_category.items()
        if entries
    ]


def _group_services_by_kind(
    services: Sequence[ExternalServiceDependency],
) -> list[dict[str, object]]:
    """Group external services by ``kind`` in canonical enum order.

    Within each group, services are sorted by ``name`` ascending so that the
    rendered HTML does not leak the analyzer's discovery order into the
    user-visible output.
    """
    by_kind: dict[ExternalServiceKind, list[ExternalServiceDependency]] = {
        kind: [] for kind in ExternalServiceKind
    }
    for entry in services:
        by_kind[entry.kind].append(entry)
    return [
        {"kind": kind.value, "entries": sorted(entries, key=lambda s: s.name)}
        for kind, entries in by_kind.items()
        if entries
    ]


def _group_tables_by_access_mode(
    tables: Sequence[DatabaseTableDependency],
) -> list[dict[str, object]]:
    """Group database tables by ``access_mode`` in canonical enum order.

    Within each group, tables are sorted by ``table_name`` ascending so that
    the rendered HTML is stable regardless of detection order. See the note
    on determinism above the grouping helpers.

    Each entry in the returned ``"entries"`` list is a small dict carrying
    the raw :class:`DatabaseTableDependency` fields the template needs
    plus a pre-computed ``display_label`` produced by
    :func:`_format_table_label`. The label appends the
    :data:`UNKNOWN_ACCESS_MODE_LABEL_SUFFIX` suffix when the entry's
    access mode is the (currently-pending) ``unknown`` value, so a human
    reader of the rendered ``Project_Profile_Diagram`` sees that the
    access mode was indeterminate (Requirement 9.8 of the
    Go-Analyzer-Support spec).
    """
    by_mode: dict[DatabaseAccessMode, list[DatabaseTableDependency]] = {
        mode: [] for mode in DatabaseAccessMode
    }
    for entry in tables:
        by_mode[entry.access_mode].append(entry)
    return [
        {
            "access_mode": mode.value,
            "entries": [
                {
                    "table_name": entry.table_name,
                    "display_label": _format_table_label(entry.table_name, mode.value),
                    "source_locations": entry.source_locations,
                }
                for entry in sorted(entries, key=lambda t: t.table_name)
            ],
        }
        for mode, entries in by_mode.items()
        if entries
    ]


# ---------------------------------------------------------------------------
# Public renderer (Requirements 13.2, 13.6)
# ---------------------------------------------------------------------------


def render_profile_diagram(profile: ProjectProfile) -> str:
    """Render a ``Project_Profile_Diagram`` HTML fragment for ``profile``.

    The rendered fragment contains, in this order:

    * The project's purpose summary (and, when present, the
      ``purpose_summary_reason`` recorded for an "unknown" summary).
    * An ``Abstract_Inputs`` section, grouped by category in canonical enum
      order. Empty inputs render the empty-state message
      ``"No Abstract Inputs detected"`` (Requirement 13.2's worked example).
    * An ``Abstract_Outputs`` section, grouped by category in canonical enum
      order. Empty outputs render ``"No Abstract Outputs detected"``.
    * An ``External_Service_Dependencies`` section, labeled by service kind.
      Empty dependencies render ``"No External Service Dependencies
      detected"``.
    * A ``Database_Table_Dependencies`` section, labeled by access mode.
      Empty dependencies render ``"No Database Table Dependencies detected"``.

    The function is pure: no I/O, no global mutable state, no caching of
    profile data. Calling it twice with equal ``profile`` values produces
    byte-for-byte equal output. That property is what task 10.6's route
    handler relies on for Requirement 14.1 ("read profiles at the time the
    HTTP request is handled, ... no in-memory caches").

    Implements Requirements 13.2 and 13.6.
    """
    return _PROFILE_DIAGRAM_TEMPLATE.render(
        profile=profile,
        inputs_title=ABSTRACT_INPUTS_SECTION_TITLE,
        outputs_title=ABSTRACT_OUTPUTS_SECTION_TITLE,
        services_title=EXTERNAL_SERVICES_SECTION_TITLE,
        tables_title=DATABASE_TABLES_SECTION_TITLE,
        empty_inputs_message=EMPTY_ABSTRACT_INPUTS_MESSAGE,
        empty_outputs_message=EMPTY_ABSTRACT_OUTPUTS_MESSAGE,
        empty_services_message=EMPTY_EXTERNAL_SERVICES_MESSAGE,
        empty_tables_message=EMPTY_DATABASE_TABLES_MESSAGE,
        input_groups=_group_inputs_by_category(profile.abstract_inputs),
        output_groups=_group_outputs_by_category(profile.abstract_outputs),
        service_groups=_group_services_by_kind(profile.external_service_dependencies),
        table_groups=_group_tables_by_access_mode(profile.database_table_dependencies),
    )


__all__ = [
    "ABSTRACT_INPUTS_SECTION_TITLE",
    "ABSTRACT_OUTPUTS_SECTION_TITLE",
    "DATABASE_TABLES_SECTION_TITLE",
    "EMPTY_ABSTRACT_INPUTS_MESSAGE",
    "EMPTY_ABSTRACT_OUTPUTS_MESSAGE",
    "EMPTY_CATALOG_MESSAGE",
    "EMPTY_DATABASE_TABLES_MESSAGE",
    "EMPTY_EXTERNAL_SERVICES_MESSAGE",
    "EXTERNAL_SERVICES_SECTION_TITLE",
    "NOT_IN_SCOPE_MESSAGE_TEMPLATE",
    "NOT_YET_ANALYZED_MESSAGE",
    "NO_SHARED_DEPENDENCIES_MESSAGE",
    "NO_SNAPSHOT_MESSAGE",
    "SHARED_TABLE_EDGE_LABEL_TEMPLATE",
    "UNKNOWN_ACCESS_MODE_LABEL_SUFFIX",
    "UNKNOWN_ACCESS_MODE_VALUE",
    "render_dependency_graph",
    "render_index_page",
    "render_profile_diagram",
    "render_project_not_in_scope_page",
    "render_project_not_yet_analyzed_page",
    "render_project_profile_page",
]


# ---------------------------------------------------------------------------
# Index page (task 10.5 — ``GET /``)
#
# The index renderer is grouped here with the other Visualization_Server
# renderers so all four routes go through the same Jinja2 ``Environment``
# (``_JINJA_ENV``) with consistent autoescape and undefined-variable
# semantics. Unlike the other renderers, the index page is not a "diagram"
# in the formal sense (no nodes / edges / sections) — it is the landing
# page that lists in-scope projects and links to the per-project,
# dependency, and conflict diagrams.
#
# Implements Requirements 13.1 and 14.4 / Property 21.

#: Visible message rendered when no ``Ingestion_Job`` has ever completed
#: (``Knowledge_Store.get_current_snapshot_id() is None``). Pinned by
#: Requirement 14.4 and Property 21.
NO_SNAPSHOT_MESSAGE: Final[str] = (
    "no project knowledge available; run an Ingestion_Job"
)

#: Visible message rendered when a snapshot has been committed but the
#: ``Project_Catalog`` is empty. Pinned by Requirement 13.1 and Property 21.
EMPTY_CATALOG_MESSAGE: Final[str] = "No Projects are in scope"


# The template renders three mutually exclusive bodies, in priority order:
#
#   1. ``has_snapshot is False``  -> Requirement 14.4: show
#      :data:`NO_SNAPSHOT_MESSAGE`, no per-project list, no diagram links.
#   2. ``has_snapshot is True`` and ``in_scope_projects`` is empty  ->
#      Requirement 13.1's empty-catalog branch: show
#      :data:`EMPTY_CATALOG_MESSAGE`, no per-project list.
#   3. otherwise -> Requirement 13.1's populated branch: render exactly one
#      ``<li>`` per in-scope project ordered by ``gitlab_project_id``
#      ascending, where each entry includes the project's id, full path, a
#      link to ``/projects/{id}`` (Project_Profile_Diagram), a link to
#      ``/dependencies`` (Dependency_Graph_Diagram), and a link to
#      ``/conflicts`` (Conflict_Overview_Diagram).
#
# The no-snapshot and empty-catalog messages are emitted inside ``<p>`` tags
# so a downstream test (or a property test) can find them with a simple
# substring search; the constants contain only ASCII letters, spaces, and a
# semicolon, none of which Jinja's HTML autoescape transforms. The project
# ``full_path`` values originate in GitLab API responses and are escaped by
# the environment's ``autoescape=True`` setting.

_INDEX_PAGE_TEMPLATE_SOURCE: Final[str] = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Project Knowledge — Index</title>
</head>
<body>
<h1>Project Knowledge</h1>
{% if not has_snapshot %}
<p class="empty-state empty-state--no-snapshot">{{ no_snapshot_message }}</p>
{% elif not in_scope_projects %}
<p class="empty-state empty-state--no-projects">{{ empty_catalog_message }}</p>
{% else %}
<ul class="project-list">
{% for project in in_scope_projects %}
<li class="project-list__item">
<span class="project-list__id">{{ project.gitlab_project_id }}</span>
<span class="project-list__path">{{ project.full_path }}</span>
<a class="project-list__profile-link" href="/projects/{{ project.gitlab_project_id }}">\
Project Profile</a>
<a class="project-list__dependency-link" href="/dependencies">Dependency Graph</a>
</li>
{% endfor %}
</ul>
{% endif %}
</body>
</html>
"""

#: Compiled Jinja2 template for the index page. Compiled once at import time
#: so :func:`render_index_page` is a pure call, matching the framing used by
#: :data:`_PROFILE_DIAGRAM_TEMPLATE`.
_INDEX_PAGE_TEMPLATE = _JINJA_ENV.from_string(_INDEX_PAGE_TEMPLATE_SOURCE)


def render_index_page(
    *,
    in_scope_projects: Sequence[InScopeProject],
    has_snapshot: bool,
) -> str:
    """Render the ``GET /`` index page body.

    Implements Requirements 13.1 and 14.4 / Property 21.

    The caller is expected to have already resolved the two state
    variables — whether a snapshot has been committed (``has_snapshot``)
    and what the in-scope project list contains (``in_scope_projects``)
    — by reading them from the ``Knowledge_Store`` and
    ``Project_Catalog`` *at request time* (Requirement 14.1). This
    function performs no I/O, no caching, and no sorting of its own;
    the route handler is responsible for sorting by ``gitlab_project_id``
    ascending before calling.

    Args:
        in_scope_projects: The in-scope project list, already sorted by
            ``gitlab_project_id`` ascending. Ignored when
            ``has_snapshot`` is ``False`` so the no-snapshot body never
            displays project entries (Requirement 14.4: the response
            must not include a ``Project_Profile_Diagram``,
            ``Dependency_Graph_Diagram``, or ``Conflict_Overview_Diagram``
            until an ``Ingestion_Job`` has completed).
        has_snapshot: ``True`` iff
            ``Knowledge_Store.get_current_snapshot_id()`` returned a
            non-``None`` value. ``False`` activates the
            :data:`NO_SNAPSHOT_MESSAGE` branch.

    Returns:
        The rendered HTML body as a Unicode string. The route handler
        wraps it in an :class:`HTMLResponse` with the documented
        ``Content-Type`` (Requirement 13.5).
    """
    return _INDEX_PAGE_TEMPLATE.render(
        in_scope_projects=in_scope_projects,
        has_snapshot=has_snapshot,
        no_snapshot_message=NO_SNAPSHOT_MESSAGE,
        empty_catalog_message=EMPTY_CATALOG_MESSAGE,
    )


# ---------------------------------------------------------------------------
# Per-project page (task 10.6 — ``GET /projects/{project_id}``)
#
# The per-project route has three branches and each renders a *complete*
# HTML document (not just a fragment): the route handler returns whatever
# this module produces verbatim as the response body. Keeping all three
# templates here — alongside ``render_profile_diagram`` and
# ``render_index_page`` — keeps every Visualization_Server view in one
# place under the same Jinja2 ``Environment`` (autoescape on,
# ``StrictUndefined``).
#
# The three branches are:
#
#   1. In-scope + profile persisted -> :func:`render_project_profile_page`
#      wraps the existing :func:`render_profile_diagram` fragment in a
#      full HTML page (Requirements 13.2, 13.6).
#   2. In-scope + no profile persisted ->
#      :func:`render_project_not_yet_analyzed_page` returns an HTTP 200
#      page carrying :data:`NOT_YET_ANALYZED_MESSAGE` and *no* diagram
#      content (Requirement 14.3).
#   3. Not in scope -> :func:`render_project_not_in_scope_page` returns
#      a 404-bound page that names the requested ``project_id`` so the
#      operator sees which id was rejected (Requirements 13.6, 14.5).
#
# Implements Requirements 13.2, 13.6, 14.3, 14.5.

#: Visible message rendered when the requested project is in scope but
#: the current snapshot has no ``Project_Profile`` for it. The wording
#: is fixed by task 10.6's acceptance text and is reproduced verbatim by
#: the unit tests so a regression in the literal is caught at test time.
NOT_YET_ANALYZED_MESSAGE: Final[str] = (
    "Project has not yet been analyzed; run an Ingestion_Job"
)

#: Format string for the "not in scope" message. ``{project_id}`` is
#: filled with the digit-only id parsed from the request path. The
#: design and Property 22 require the rendered body to *include* the
#: requested ``project_id`` so an operator can tell which id was
#: rejected.
NOT_IN_SCOPE_MESSAGE_TEMPLATE: Final[str] = "Project {project_id} is not in scope"


# The "profile present" page wraps the diagram fragment produced by
# :func:`render_profile_diagram` in a full HTML document. The diagram
# fragment is already fully escaped by the ``_JINJA_ENV`` autoescape
# pipeline (every ``{{ ... }}`` substitution within the diagram template
# is escaped at render time), so embedding it via ``|safe`` here does
# not bypass the trust boundary — it merely tells Jinja2 not to
# double-escape an already-rendered fragment.
_PROJECT_PROFILE_PAGE_TEMPLATE_SOURCE: Final[str] = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Project Knowledge — {{ profile.full_path }}</title>
</head>
<body>
<nav class="page-nav"><a class="page-nav__index-link" href="/">Index</a></nav>
<main>
{{ diagram_fragment|safe }}
</main>
</body>
</html>
"""

#: Compiled Jinja2 template for the "profile present" branch. Compiled
#: once at import time to keep :func:`render_project_profile_page` a
#: pure call.
_PROJECT_PROFILE_PAGE_TEMPLATE = _JINJA_ENV.from_string(
    _PROJECT_PROFILE_PAGE_TEMPLATE_SOURCE
)


# The "in-scope + not yet analyzed" page deliberately does not embed a
# Project_Profile_Diagram (Requirement 14.3 forbids it). The
# ``data-project-id`` attribute makes the affected id discoverable to
# tests without introducing user-visible numeric noise.
_NOT_YET_ANALYZED_PAGE_TEMPLATE_SOURCE: Final[str] = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Project Knowledge — Project {{ project_id }} not yet analyzed</title>
</head>
<body>
<nav class="page-nav"><a class="page-nav__index-link" href="/">Index</a></nav>
<main>
<p class="empty-state empty-state--not-yet-analyzed" data-project-id="{{ project_id }}">\
{{ message }}</p>
</main>
</body>
</html>
"""

#: Compiled Jinja2 template for the "in-scope + no profile" branch.
_NOT_YET_ANALYZED_PAGE_TEMPLATE = _JINJA_ENV.from_string(
    _NOT_YET_ANALYZED_PAGE_TEMPLATE_SOURCE
)


# The "not in scope" page is rendered with HTTP 404 by the route
# handler; the renderer itself only owns the body. The body must
# *include the value of {project_id} that was requested* per Requirement
# 13.6 / 14.5, so the message string interpolates the numeric id into
# both the visible text and a ``data-project-id`` attribute.
_NOT_IN_SCOPE_PAGE_TEMPLATE_SOURCE: Final[str] = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Project Knowledge — Project {{ project_id }} not in scope</title>
</head>
<body>
<nav class="page-nav"><a class="page-nav__index-link" href="/">Index</a></nav>
<main>
<p class="empty-state empty-state--not-in-scope" data-project-id="{{ project_id }}">\
{{ message }}</p>
</main>
</body>
</html>
"""

#: Compiled Jinja2 template for the "not in scope" branch.
_NOT_IN_SCOPE_PAGE_TEMPLATE = _JINJA_ENV.from_string(_NOT_IN_SCOPE_PAGE_TEMPLATE_SOURCE)


def render_project_profile_page(profile: ProjectProfile) -> str:
    """Render a complete HTML page for the in-scope + profile-present branch.

    Embeds :func:`render_profile_diagram`'s fragment inside a minimal
    page chrome (``<!doctype html>``, ``<head>``, ``<body>``,
    navigation back to the index). The wrapper is intentionally
    minimal — the diagram fragment is the real payload — so future
    visual styling can be added by a steering CSS file without changes
    here.

    Implements Requirements 13.2 and 13.6.
    """
    diagram_fragment = render_profile_diagram(profile)
    return _PROJECT_PROFILE_PAGE_TEMPLATE.render(
        profile=profile,
        diagram_fragment=diagram_fragment,
    )


def render_project_not_yet_analyzed_page(*, project_id: int) -> str:
    """Render the "Project has not yet been analyzed" page (HTTP 200).

    Used when ``Project_Catalog.is_in_scope(project_id)`` is ``True``
    but ``KnowledgeStore.get_profile(project_id)`` returns ``None`` —
    the project is in scope but the current snapshot has no profile
    for it (e.g. analysis was skipped, or the project was added to
    the catalog by an in-progress single-project refresh that copied
    the parent snapshot before the new analysis completed).

    Implements Requirement 14.3.
    """
    return _NOT_YET_ANALYZED_PAGE_TEMPLATE.render(
        project_id=project_id,
        message=NOT_YET_ANALYZED_MESSAGE,
    )


def render_project_not_in_scope_page(*, project_id: int) -> str:
    """Render the "Project not in scope" page body for an HTTP 404 response.

    The body contains both a human-readable message stating that the
    requested project is not in scope and the requested ``project_id``
    value embedded in the message text and in a ``data-project-id``
    attribute, so an operator (or a test) can discover which id was
    rejected without parsing the URL.

    Implements Requirements 13.6 and 14.5.
    """
    return _NOT_IN_SCOPE_PAGE_TEMPLATE.render(
        project_id=project_id,
        message=NOT_IN_SCOPE_MESSAGE_TEMPLATE.format(project_id=project_id),
    )


# ---------------------------------------------------------------------------
# Dependency_Graph_Diagram (task 10.3 — ``GET /dependencies``)
# ---------------------------------------------------------------------------
#
# The Dependency_Graph_Diagram renders one node per in-scope ``Project_Catalog``
# entry and one edge per shared database-table dependency between every
# unordered pair of in-scope Projects whose ``ProjectProfile`` lists the
# same ``Database_Table_Dependency.table_name``. The edge label is pinned
# by Requirement 13.3: ``"shared table: {table_name}"``.
#
# The shared-external-service edge kind was retired by an operator-tuning
# decision: in real ESB-scale snapshots, every microservice tends to
# talk to a small set of shared infrastructure services (e.g. a single
# auth API, a single Kafka cluster), so the resulting graph was a fully
# connected hairball where the external-service edges drowned out the
# more discriminating shared-database-table edges.
#
# The renderer is a pure function from ``(catalog, profiles)`` to an HTML
# fragment containing inline Mermaid. Keeping it pure is load-bearing for
# the property test in task 10.15, which drives the renderer with
# Hypothesis-generated ``ProjectCatalog`` and ``ProjectProfile`` values and
# asserts the resulting diagram's node-set / edge-set against a brute-force
# specification.
#
# Implements Requirement 13.3 (and is consumed by Property 23 / task 10.15).


# ---------------------------------------------------------------------------
# Empty-state messages and edge-label templates
# ---------------------------------------------------------------------------
#
# All user-visible literal strings live in module-level constants so the
# unit tests below and the property test in task 10.15 can import them
# rather than duplicating brittle inline literals. The empty-state wording
# is fixed by Requirement 13.3's "no shared dependencies were detected
# between Projects" clause; the edge-label template is fixed verbatim by
# the same requirement and Property 23.

#: Visible message rendered inside the ``Dependency_Graph_Diagram`` when no
#: two in-scope Projects share any ``Database_Table_Dependency``. Pinned by
#: Requirement 13.3 and Property 23 (Validates Requirements 13.3, 14.4).
NO_SHARED_DEPENDENCIES_MESSAGE: Final[str] = (
    "No shared dependencies detected between Projects"
)

#: Format string for an edge produced by a shared database-table
#: dependency. The ``{table_name}`` placeholder is the
#: ``Database_Table_Dependency.table_name`` shared between the two
#: projects connected by the edge. Pinned verbatim by Requirement 13.3.
SHARED_TABLE_EDGE_LABEL_TEMPLATE: Final[str] = "shared table: {table_name}"


# ---------------------------------------------------------------------------
# Mermaid-label escaping
# ---------------------------------------------------------------------------
#
# The Mermaid block content lives inside ``<pre class="mermaid">``. The
# browser parses the ``<pre>`` element and decodes any HTML entities in
# its text content before Mermaid sees the string; Mermaid then parses
# its own quoted-string syntax.
#
# Two transformations are required to embed an arbitrary ``str`` as a
# Mermaid quoted label safely, and the order matters:
#
#   1. Replace any literal ``"`` with the Mermaid HTML-entity escape
#      ``&quot;``. Mermaid interprets the entity inside a quoted label
#      and renders it as ``"``; embedding a raw ``"`` would close the
#      label early and break the diagram.
#   2. Apply HTML escaping (``&`` -> ``&amp;``, ``<`` -> ``&lt;``,
#      ``>`` -> ``&gt;``) so the resulting string can be safely placed
#      inside the surrounding HTML page. The browser decodes ``&amp;``
#      back to ``&`` before Mermaid runs, so the ``&quot;`` injected in
#      step 1 reaches Mermaid intact via ``&amp;quot;`` -> ``&quot;``.
#
# Mermaid does not assign special meaning to ``'`` inside quoted
# labels, so ``html.escape(..., quote=False)`` is used to leave single
# quotes alone. The function deliberately handles step 1 and step 2 in
# this order so the unit tests below can pass simple ASCII labels
# (e.g. ``"payments-api"``) and observe them verbatim in the rendered
# HTML, while still being safe against adversarial inputs from the
# property test in task 10.15.


def _escape_mermaid_label(text: str) -> str:
    """Escape ``text`` for inclusion as a Mermaid quoted-label payload.

    Returns a string suitable to be placed between the double quotes of
    a Mermaid node or edge label (e.g. ``id["{escaped}"]`` or
    ``A ---|"{escaped}"| B``) inside an HTML ``<pre class="mermaid">``
    block. See the comment block above this function for the rationale
    behind the transformation order.
    """
    return html.escape(text.replace('"', "&quot;"), quote=False)


# ---------------------------------------------------------------------------
# Mermaid source builder
# ---------------------------------------------------------------------------
#
# Building the Mermaid source as a single ``str`` in Python (rather than
# inside the Jinja2 template) keeps the per-line escaping rules explicit
# and lets the template body stay branch-free. The template just decides
# whether to render the empty-catalog message or the Mermaid block; the
# Python helper does the per-pair edge computation.
#
# Determinism is essential: the property test in task 10.15 asserts
# byte-for-byte equality of two renders that receive equal inputs, and
# the route handler in task 10.7 reads from ``Project_Catalog`` and
# ``Knowledge_Store`` at request time, so any non-determinism here would
# leak into HTTP responses. We achieve determinism by:
#
#   * Sorting the catalog by ``gitlab_project_id`` ascending before
#     iterating.
#   * Iterating unordered pairs (a, b) with ``a.gitlab_project_id <
#     b.gitlab_project_id`` so each pair is visited exactly once and in
#     a fixed order.
#   * Emitting external-service edges before table edges per pair, and
#     within each kind iterating the shared names in ASCII-sorted order.
#
# An in-scope project that has no persisted ``ProjectProfile`` (e.g. a
# project enumerated by the catalog but skipped by the analyzer because
# its ``Analysis_Branch`` was missing — Requirement 15.5) is rendered as
# a node with no edges. Profiles for projects that are not in the
# catalog are ignored so the rendered node-set always matches
# ``catalog`` exactly, as required by Property 23.


def _encode_graph_data_for_html(data: dict[str, object]) -> str:
    """Serialize ``data`` to JSON that is safe inside ``<script>``.

    The page chrome emits the JSON inside a ``<script type="application/json"
    class="graph-data">...</script>`` block so the browser's HTML
    parser still recognizes the literal ``</script>`` token as the
    end of the block — regardless of the ``type`` attribute. The
    standard mitigation is to escape every ``</`` as ``<\\/`` so the
    sequence cannot close the surrounding tag. The browser's JSON
    parser treats ``\\/`` and ``/`` as the same character, so the
    escape is transparent to Cytoscape on the consuming side.

    Also escapes the Unicode line / paragraph separators which JSON
    permits but the browser's JavaScript parser does not, even though
    Cytoscape itself only ever sees the already-parsed object — this
    keeps the script block well-formed against any pathological
    label content.
    """
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _build_dependency_graph_data(
    sorted_catalog: Sequence[InScopeProject],
    profile_by_id: Mapping[int, ProjectProfile],
) -> dict[str, object]:
    """Build a Cytoscape-shaped ``{nodes, edges}`` payload for the dependency graph.

    Mirrors :func:`_build_dependency_graph_mermaid_source` exactly so
    the two outputs always describe the same node set and edge set;
    the only difference is the wire format. Each node carries the
    documented ``"{gitlab_project_id} {full_path}"`` label and a
    ``P{id}`` element id; each edge carries the shared-table label
    produced by the same template the Mermaid builder uses
    (:data:`SHARED_TABLE_EDGE_LABEL_TEMPLATE`) and the
    ``[access: unknown]`` suffix when applicable.

    Isolated nodes — projects with no shared-table edge to any
    other in-scope project — are pruned from the emitted node set
    by an operator-tuning decision. The Conflict_Overview view was
    removed entirely for similar readability reasons; the
    dependency graph keeps the same principle of "only render
    nodes that carry signal".
    """
    edges: list[dict[str, object]] = []
    participating_ids: set[int] = set()
    sorted_ids = [project.gitlab_project_id for project in sorted_catalog]
    for index, a_id in enumerate(sorted_ids):
        profile_a = profile_by_id.get(a_id)
        if profile_a is None:
            continue
        tables_a: dict[str, str] = {
            dep.table_name: dep.access_mode.value
            for dep in profile_a.database_table_dependencies
        }
        for b_id in sorted_ids[index + 1 :]:
            profile_b = profile_by_id.get(b_id)
            if profile_b is None:
                continue
            tables_b: dict[str, str] = {
                dep.table_name: dep.access_mode.value
                for dep in profile_b.database_table_dependencies
            }
            for table_name in sorted(tables_a.keys() & tables_b.keys()):
                edge_label = SHARED_TABLE_EDGE_LABEL_TEMPLATE.format(
                    table_name=table_name
                )
                if (
                    tables_a[table_name] == UNKNOWN_ACCESS_MODE_VALUE
                    or tables_b[table_name] == UNKNOWN_ACCESS_MODE_VALUE
                ):
                    edge_label = f"{edge_label}{UNKNOWN_ACCESS_MODE_LABEL_SUFFIX}"
                edges.append(
                    {
                        "data": {
                            "source": f"P{a_id}",
                            "target": f"P{b_id}",
                            "label": edge_label,
                            "kind": "shared_table",
                        }
                    }
                )
                participating_ids.add(a_id)
                participating_ids.add(b_id)

    # Emit nodes only for projects that participate in at least one
    # shared-table edge. Projects with no relation to any other
    # in-scope project are pruned from the rendered graph because
    # they convey no signal in a "shared dependencies" view; the
    # operator's snapshots routinely have 100+ projects, the
    # majority of which are isolated, and rendering them all made
    # the diagram unreadable.
    nodes: list[dict[str, object]] = []
    for project in sorted_catalog:
        if project.gitlab_project_id not in participating_ids:
            continue
        node_label = f"{project.gitlab_project_id} {project.full_path}"
        nodes.append(
            {
                "data": {
                    "id": f"P{project.gitlab_project_id}",
                    "label": node_label,
                }
            }
        )

    return {"nodes": nodes, "edges": edges}


def _build_dependency_graph_mermaid_source(
    sorted_catalog: Sequence[InScopeProject],
    profile_by_id: Mapping[int, ProjectProfile],
) -> tuple[str, int]:
    """Build the inline Mermaid source for a populated catalog.

    Returns a ``(mermaid_source, edge_count)`` pair. ``mermaid_source``
    is a newline-separated Mermaid program whose first line is
    ``graph LR`` followed by one node line per *participating* catalog
    entry and one edge line per shared dependency. ``edge_count`` is
    the number of edge lines emitted; the caller uses it to decide
    whether to render the :data:`NO_SHARED_DEPENDENCIES_MESSAGE`
    empty-state alongside the diagram. Projects with no shared-table
    edge to any other in-scope project are pruned from the node
    set by an operator-tuning decision — see the matching commentary
    on :func:`_build_dependency_graph_data` for the full rationale.
    """
    edge_lines: list[str] = []
    participating_ids: set[int] = set()
    edge_count = 0
    sorted_ids = [project.gitlab_project_id for project in sorted_catalog]
    for index, a_id in enumerate(sorted_ids):
        profile_a = profile_by_id.get(a_id)
        if profile_a is None:
            # An in-scope project with no persisted profile cannot
            # contribute any edges; skip its entire row of the pair
            # iteration. This is the catalog-vs-profiles asymmetry
            # called out in Requirement 14.3 (in-scope but not yet
            # analyzed).
            continue
        # Map each table name to its access mode so the per-edge label
        # can record an "[access: unknown]" suffix when either side of
        # a shared dependency carries an indeterminate access mode
        # (Requirement 9.8 of the Go-Analyzer-Support spec). Building
        # the mapping eagerly keeps the inner loop simple and remains
        # O(tables) per profile, matching the existing complexity.
        tables_a: dict[str, str] = {
            dep.table_name: dep.access_mode.value
            for dep in profile_a.database_table_dependencies
        }
        for b_id in sorted_ids[index + 1 :]:
            profile_b = profile_by_id.get(b_id)
            if profile_b is None:
                continue
            tables_b: dict[str, str] = {
                dep.table_name: dep.access_mode.value
                for dep in profile_b.database_table_dependencies
            }
            for table_name in sorted(tables_a.keys() & tables_b.keys()):
                edge_label = SHARED_TABLE_EDGE_LABEL_TEMPLATE.format(
                    table_name=table_name
                )
                # If either project records this shared table with the
                # ``unknown`` access mode, append the documented suffix
                # so the rendered edge label makes the indeterminate
                # access mode explicit. The comparison is done on the
                # underlying string value (``DatabaseAccessMode`` is a
                # ``StrEnum``) so the recognition is robust to
                # additional enum members beyond the four currently
                # defined.
                if (
                    tables_a[table_name] == UNKNOWN_ACCESS_MODE_VALUE
                    or tables_b[table_name] == UNKNOWN_ACCESS_MODE_VALUE
                ):
                    edge_label = f"{edge_label}{UNKNOWN_ACCESS_MODE_LABEL_SUFFIX}"
                edge_lines.append(
                    f'P{a_id} ---|"{_escape_mermaid_label(edge_label)}"| P{b_id}'
                )
                edge_count += 1
                participating_ids.add(a_id)
                participating_ids.add(b_id)

    # Node lines come first in the Mermaid source, even though they
    # are computed after the edge pass: ``graph LR`` plus one node
    # line per *participating* project, followed by every edge line.
    # Projects pruned from the node set never appear in edge_lines
    # because the pruning is driven by the same edge-emission loop.
    lines: list[str] = ["graph LR"]
    for project in sorted_catalog:
        if project.gitlab_project_id not in participating_ids:
            continue
        node_label = f"{project.gitlab_project_id} {project.full_path}"
        lines.append(
            f'P{project.gitlab_project_id}["{_escape_mermaid_label(node_label)}"]'
        )
    lines.extend(edge_lines)

    return "\n".join(lines), edge_count


# ---------------------------------------------------------------------------
# Dependency_Graph_Diagram template
# ---------------------------------------------------------------------------
#
# The template renders three mutually-exclusive bodies, in the priority
# order documented by Requirement 13.3:
#
#   1. Empty catalog -> :data:`EMPTY_CATALOG_MESSAGE` ("No Projects are in
#      scope") plus :data:`NO_SHARED_DEPENDENCIES_MESSAGE`. No Mermaid
#      block is emitted because there are no nodes to render and no
#      shared dependencies to be possible.
#   2. Populated catalog with no shared dependencies -> the Mermaid block
#      with one node per in-scope project but no edges, plus
#      :data:`NO_SHARED_DEPENDENCIES_MESSAGE`. The user still sees the
#      project nodes; Requirement 13.3 explicitly preserves the nodes in
#      this branch.
#   3. Populated catalog with at least one shared dependency -> the
#      Mermaid block with nodes and edges. No empty-state message.
#
# The Mermaid source is rendered with the ``| safe`` filter because the
# Python helper above has already applied ``_escape_mermaid_label`` to
# every user-controlled label. Without ``| safe`` Jinja's autoescape
# would re-encode the literal ``"`` characters that delimit Mermaid
# labels, which the browser would decode back, so the substitution
# would be a no-op for correctness but would obscure the rendered HTML
# when string-searched in the unit tests below.

_DEPENDENCY_GRAPH_TEMPLATE_SOURCE: Final[str] = """\
<section class="dependency-graph-diagram">
<h1>Dependency Graph</h1>
{% if not in_scope_projects %}
<p class="empty-state empty-state--no-projects">{{ empty_catalog_message }}</p>
<p class="empty-state empty-state--no-shared-deps">\
{{ no_shared_dependencies_message }}</p>
{% else %}
<pre class="mermaid">
{{ mermaid_source | safe }}
</pre>
<script type="application/json" class="graph-data">{{ graph_data_json | safe }}</script>
<div class="graph-container"></div>
{% if not has_edges %}
<p class="empty-state empty-state--no-shared-deps">\
{{ no_shared_dependencies_message }}</p>
{% endif %}
{% endif %}
</section>
"""

#: Compiled Jinja2 template for the Dependency_Graph_Diagram. Compiled
#: once at import time so :func:`render_dependency_graph` is a pure call
#: that performs no parsing, no environment mutation, and no template-
#: cache lookup at call time. Matches the framing used by
#: :data:`_PROFILE_DIAGRAM_TEMPLATE` and :data:`_INDEX_PAGE_TEMPLATE`.
_DEPENDENCY_GRAPH_TEMPLATE = _JINJA_ENV.from_string(
    _DEPENDENCY_GRAPH_TEMPLATE_SOURCE
)


def render_dependency_graph(
    catalog: Sequence[InScopeProject],
    profiles: Sequence[ProjectProfile],
) -> str:
    """Render a ``Dependency_Graph_Diagram`` HTML fragment.

    Renders an HTML ``<section>`` containing inline Mermaid that draws
    one node per in-scope ``Project_Catalog`` entry and one edge per
    shared database table between every unordered pair of in-scope
    Projects:

    * One edge per shared ``Database_Table_Dependency.table_name``
      labeled ``"shared table: {table_name}"``.

    When ``catalog`` is empty, the rendered fragment contains the
    :data:`EMPTY_CATALOG_MESSAGE` empty-state message ("No Projects
    are in scope") together with :data:`NO_SHARED_DEPENDENCIES_MESSAGE`
    ("No shared dependencies detected between Projects"), and no
    Mermaid block. When ``catalog`` is non-empty but no two Projects
    share any dependency, the Mermaid block still renders the nodes,
    and :data:`NO_SHARED_DEPENDENCIES_MESSAGE` is appended below the
    diagram. When at least one shared dependency exists, the Mermaid
    block renders both nodes and edges with no empty-state message.

    The function is **pure**: no I/O, no global mutable state, no
    caching of profile data. Two calls with structurally-equal inputs
    return byte-for-byte equal output. That property is load-bearing
    for the route handler in task 10.7 (which calls this renderer per
    request after reading the catalog and profiles from the
    ``Knowledge_Store`` at request time, per Requirement 14.1) and
    for the property test in task 10.15 (which drives the renderer
    with Hypothesis-generated inputs and asserts the resulting node-
    set / edge-set against a brute-force specification).

    Args:
        catalog: The in-scope projects from
            :meth:`ProjectCatalog.list_in_scope`. Need not be sorted;
            this function sorts by ``gitlab_project_id`` ascending
            before rendering so the resulting Mermaid source is
            deterministic regardless of catalog row order.
        profiles: The persisted ``ProjectProfile`` records from
            :meth:`KnowledgeStore.list_profiles`. Profiles whose
            ``gitlab_project_id`` is not in ``catalog`` are silently
            ignored so the rendered node-set always equals exactly
            ``catalog``, matching Property 23.

    Returns:
        The rendered HTML fragment as a Unicode string. The route
        handler wraps it in an :class:`HTMLResponse` with the
        documented ``Content-Type`` (Requirement 13.5).

    Implements Requirement 13.3.
    """
    sorted_catalog: list[InScopeProject] = sorted(
        catalog, key=lambda project: project.gitlab_project_id
    )
    in_scope_ids: set[int] = {
        project.gitlab_project_id for project in sorted_catalog
    }
    profile_by_id: dict[int, ProjectProfile] = {
        profile.gitlab_project_id: profile
        for profile in profiles
        if profile.gitlab_project_id in in_scope_ids
    }

    if sorted_catalog:
        mermaid_source, edge_count = _build_dependency_graph_mermaid_source(
            sorted_catalog, profile_by_id
        )
        graph_data_json = _encode_graph_data_for_html(
            _build_dependency_graph_data(sorted_catalog, profile_by_id)
        )
    else:
        # Empty catalog -> the template short-circuits before
        # consulting ``mermaid_source`` or ``graph_data_json``, but
        # Jinja2 in StrictUndefined mode requires every referenced
        # variable to be defined. Pass empty strings so the rendering
        # remains a single code path.
        mermaid_source = ""
        graph_data_json = ""
        edge_count = 0

    return _DEPENDENCY_GRAPH_TEMPLATE.render(
        in_scope_projects=sorted_catalog,
        mermaid_source=mermaid_source,
        graph_data_json=graph_data_json,
        has_edges=edge_count > 0,
        empty_catalog_message=EMPTY_CATALOG_MESSAGE,
        no_shared_dependencies_message=NO_SHARED_DEPENDENCIES_MESSAGE,
    )

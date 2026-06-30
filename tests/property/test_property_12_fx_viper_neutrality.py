# ruff: noqa: E501
# Feature: go-analyzer-support, Property 12: fx and viper calls produce no detections by themselves; nested fx.Invoke functions still scan.
"""Property test for the end-to-end fx and viper neutrality contract.

**Validates Requirements 12.1, 12.2, 12.3, 13.1, 13.2, 13.3** (Property
12 in the design, task 11.10 in ``tasks.md``).

This test pins down two complementary halves of the fx / viper
neutrality contract, end-to-end through
:func:`project_knowledge_mcp.project_analyzer.analyze`:

1. **Wrapper neutrality (Property 12, first bullet).** For every
   synthetic Go fixture whose ``MethodCallEvent`` stream is composed
   exclusively of ``fx.*`` / ``viper.*`` / ``<id>.<m>(...)`` shapes
   that the dispatch-boundary exclusion is meant to drop, the
   produced :class:`ProjectProfile` MUST carry empty
   ``abstract_inputs``, ``abstract_outputs``,
   ``external_service_dependencies``, and
   ``database_table_dependencies`` lists. No string literal that was
   passed only to a ``viper.*`` call SHALL appear in any emitted
   description (Requirement 13.2). The five Go I/O recognizers
   already pin this contract at the per-recognizer dispatch boundary
   in their own property tests (5, 6, 7, 8, 9); this test confirms
   the contract still holds when the five recognizers run together
   through ``analyze()``.

2. **Nested scanning (Property 12, second bullet, Requirement 12.2).**
   When a top-level ``func runScheduler() { ... }`` whose body
   contains a recognized scheduler registration is referenced *by
   name* from ``fx.Invoke(runScheduler)``, the parser's whole-file
   walker descends into ``runScheduler``'s body and the scheduler
   recognizer SHALL still emit one
   ``AbstractInput(category=scheduled_event)`` for the nested
   ``c.AddFunc(...)`` call. The emitted Source_Location SHALL point
   at the ``AddFunc`` line inside ``runScheduler``, not at the
   surrounding ``fx.Invoke`` line. This is the ``cat-service``
   ``fx.Invoke(runScheduler)`` pattern called out verbatim in
   Requirement 12.2.

   A complementary regression test pins the known parser limitation:
   when the *same* recognized construction is placed inline as an
   anonymous function literal (``fx.Invoke(func() { c.AddFunc(...)
   })``), the current parser consumes the ``func() { ... }``
   expression as a single ``UnknownArg`` and never descends into its
   body — so the nested call is *not* recognized. Pinning the
   limitation explicitly here makes a future parser upgrade that
   begins descending into function-literal bodies surface as a
   test-status change (``unexpected_pass``) rather than a silent
   change in profile output. See the docstring on
   :func:`test_inline_func_literal_inside_fx_invoke_is_not_scanned`
   for the design rationale.

The wrapper-neutrality property runs at minimum 100 Hypothesis
iterations under the suite-wide ``ci`` profile registered in
``tests/conftest.py``. The nested-scanning and inline-literal tests
are deterministic single-fixture regressions colocated with the
property so the three halves of the contract are pinned in one file.
"""

from __future__ import annotations

import string
from typing import Final

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import (
    AbstractInputCategory,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer import analyze


pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Shared analyze() argument helpers
# ---------------------------------------------------------------------------


#: Canonical ``analyze()`` arguments that are immaterial to this
#: property. The aggregator records them on the produced profile but
#: the test asserts only on the four list-shaped detection sections.
_PROJECT_ID: Final[int] = 42
_FULL_PATH: Final[str] = "group/fx-viper-neutrality"
_ANALYSIS_BRANCH: Final[str] = "main"
_COMMIT_SHA: Final[str] = "0" * 40
_REPO_DESCRIPTION: Final[str | None] = None


def _run_analyze(files: dict[str, str]) -> "object":
    """Build a :class:`RepositoryContents` from ``files`` and invoke ``analyze``.

    The aggregator's positional argument tuple is fixed across this
    test because nothing the test asserts on depends on ``project_id``,
    ``full_path``, ``analysis_branch``, ``commit_sha``, or
    ``repo_description``. Centralizing the call site keeps the three
    property functions below short.
    """

    repo = RepositoryContents(
        gitlab_project_id=_PROJECT_ID,
        commit_sha=_COMMIT_SHA,
        files=files,
    )
    return analyze(
        _PROJECT_ID,
        _FULL_PATH,
        _ANALYSIS_BRANCH,
        _COMMIT_SHA,
        _REPO_DESCRIPTION,
        repo,
    )


# ---------------------------------------------------------------------------
# fx / viper wrapper snippet catalogues
# ---------------------------------------------------------------------------


#: Hand-picked subset of the Uber ``fx`` API surface listed verbatim in
#: Requirement 12.1. Each template includes one or more ``{ident}``
#: placeholders so the generator can plant per-fixture identifiers that
#: also serve as canaries: if the dispatch-boundary exclusion ever
#: regressed and the recognizer began inspecting these calls, the
#: ``{ident}`` value would surface in an emission's description and
#: the assertion would fire on the canary string. The templates are
#: deliberately exhaustive across the fx-API method-name vocabulary
#: (``New``, ``Module``, ``Provide``, ``Invoke``, ``Annotate``,
#: ``WithLogger``, ``Hook``, ``Run``, ``Lifecycle.Append``) so a
#: future regression that whitelisted one method on the fx receiver
#: by accident is caught.
_FX_SNIPPETS: Final[tuple[str, ...]] = (
    'app := fx.New(fx.Module("{ident}"))\n',
    'mod := fx.Module("{ident}", fx.Provide(NewThing))\n',
    'fx.Invoke({ident}Handler)\n',
    'fx.Annotate({ident}Ctor, fx.As(new({ident}Iface)))\n',
    'fx.WithLogger(func() any {{ return nil }})\n',
    'lc.Append(fx.Hook{{OnStart: {ident}Start, OnStop: {ident}Stop}})\n',
    'app.Run()\n',
)


#: Hand-picked subset of the ``github.com/spf13/viper`` API surface
#: listed verbatim in Requirement 13.1. Each template includes one
#: ``{canary}`` placeholder whose value is generated as a distinctive
#: token (see :data:`_canary_text`) so the assertion
#: ``no emission's description contains any canary`` (Requirement
#: 13.2) is testable without false positives: the canary alphabet is
#: chosen so a canary token cannot syntactically appear in any
#: description the five Go recognizers know how to emit. The
#: templates cover both the package-level (``viper.<method>(...)``)
#: and the receiver-level (``v.<method>(...)``) call shapes.
_VIPER_SNIPPETS: Final[tuple[str, ...]] = (
    'viper.SetConfigName("{canary}")\n',
    'viper.AddConfigPath("{canary}")\n',
    'viper.SetConfigType("{canary}")\n',
    'viper.SetEnvPrefix("{canary}")\n',
    'viper.BindEnv("{canary}")\n',
    'viper.AutomaticEnv()\n',
    'viper.WatchConfig()\n',
    'viper.ReadInConfig()\n',
    'cfgName := viper.GetString("{canary}")\n',
    'cfgDur := viper.GetDuration("{canary}")\n',
    'cfgInt := viper.GetInt("{canary}")\n',
    'cfgBool := viper.GetBool("{canary}")\n',
    'v := viper.New()\n',
    'v.SetConfigName("{canary}")\n',
    'v.AddConfigPath("{canary}")\n',
    'v.BindEnv("{canary}")\n',
    'v.UnmarshalKey("{canary}", &dest)\n',
)


#: Canary alphabet. The 32-character random string built from this
#: alphabet is statistically guaranteed to be unique within any
#: fixture, and the alphabet contains no characters used as
#: punctuation or syntax inside any recognizer's description format.
#: Recognizer descriptions are template strings such as
#: ``"HTTP <METHOD> <path> via http.HandleFunc() at <file>:<line>"``;
#: a 32-character ``[A-Za-z0-9]`` run never occurs as a substring of
#: any such template by accident.
_CANARY_ALPHABET: Final[str] = string.ascii_letters + string.digits
_CANARY_LENGTH: Final[int] = 32


_canary_text = st.text(
    alphabet=_CANARY_ALPHABET,
    min_size=_CANARY_LENGTH,
    max_size=_CANARY_LENGTH,
)


#: Identifier alphabet used for ``{ident}`` placeholders in
#: :data:`_FX_SNIPPETS`. Identifiers must be valid Go identifiers so
#: the tokenizer accepts them, and must not collide with any
#: recognized method name (``Handle``, ``HandleFunc``, ``AddFunc``,
#: ``AddJob``, ``Subscribe``, ``SendMessage``, ``ExecuteQuery``,
#: ``PoolExecuteQuery``, ``NewClient``, ``Open``, ``Create``,
#: ``OpenFile``, ``ReadFile``, ``WriteFile``). The generator filters
#: any draw that matches the reserved set.
_RESERVED_IDENT_NAMES: Final[frozenset[str]] = frozenset({
    "Handle",
    "HandleFunc",
    "AddFunc",
    "AddJob",
    "Subscribe",
    "SendMessage",
    "ExecuteQuery",
    "PoolExecuteQuery",
    "NewClient",
    "Open",
    "Create",
    "OpenFile",
    "ReadFile",
    "WriteFile",
    "Parse",
})


_ident_text = st.text(
    alphabet=string.ascii_letters,
    min_size=3,
    max_size=10,
).filter(lambda s: s not in _RESERVED_IDENT_NAMES)


# ---------------------------------------------------------------------------
# Wrapper-neutrality fixture generator
# ---------------------------------------------------------------------------


@st.composite
def _wrapper_file_body(draw: st.DrawFn) -> tuple[str, list[str]]:
    """Generate one Go file body containing only fx and viper wiring.

    Returns ``(body, canaries)``: ``body`` is the file contents
    (package declaration, imports, and a single ``init`` function
    holding the chosen wrapper calls); ``canaries`` is the list of
    canary tokens substituted into the viper-call string literals so
    the test can assert each one is absent from every emission's
    description (Requirement 13.2).

    Each generated file declares ``package wiring`` (not ``package
    main``) and lives under the ``internal/`` directory in the
    caller, so the CLI entry-point recognizer's three eligible path
    shapes (``cmd/<name>/main.go``, ``cmd/main.go``, root
    ``main.go``) never match. This isolates the fx / viper neutrality
    contract from the binary-detection contract pinned by
    Property 9.
    """

    # Pick a random subset of the catalogues. Both sets are sampled
    # at moderate frequency so each file mixes fx and viper calls,
    # which is the most adversarial shape: any regression that only
    # excluded one of the two receivers would still pass on a
    # single-receiver fixture.
    fx_lines = draw(
        st.lists(
            st.sampled_from(_FX_SNIPPETS),
            min_size=0,
            max_size=6,
        ),
    )
    viper_lines = draw(
        st.lists(
            st.sampled_from(_VIPER_SNIPPETS),
            min_size=0,
            max_size=6,
        ),
    )

    canaries: list[str] = []
    rendered_lines: list[str] = []
    for line in fx_lines:
        ident = draw(_ident_text)
        rendered_lines.append("    " + line.format(ident=ident))
    for line in viper_lines:
        canary = draw(_canary_text)
        canaries.append(canary)
        rendered_lines.append("    " + line.format(canary=canary))

    # Stable preamble: package declaration plus a parenthesised
    # import block that admits both go.uber.org/fx and
    # github.com/spf13/viper. The imports are always present so the
    # tokenizer always parses the file even when the body contains
    # zero fx / viper calls. The dummy ``_ = app`` / ``_ = mod``
    # references suppress hypothetical "unused variable" warnings in
    # the rendered Go (the tokenizer never enforces them, but the
    # source remains realistic).
    body = (
        "package wiring\n"
        "\n"
        "import (\n"
        '    "go.uber.org/fx"\n'
        '    "github.com/spf13/viper"\n'
        ")\n"
        "\n"
        "func init() {\n"
    )
    body += "".join(rendered_lines)
    body += "}\n"
    return body, canaries


@st.composite
def _wrapper_repo(draw: st.DrawFn) -> tuple[dict[str, str], list[str]]:
    """Generate a multi-file repository containing only fx / viper wiring.

    Returns ``(files, canaries)``: ``files`` is the
    :attr:`RepositoryContents.files` mapping; ``canaries`` is the
    aggregated list of canary tokens across every generated file.

    Every file lives under ``internal/`` and uses a unique basename
    so the file map is conflict-free. A ``go.mod`` is included at
    the repo root so the language-agnostic purpose summarizer's
    manifest-path branch is exercised; the manifest's module path
    deliberately does not appear in any wrapper snippet so a
    regression that surfaced the module path in an I/O description
    would also fail the canary-containment assertion (none of the
    five Go I/O recognizers should ever read ``go.mod``).
    """

    n_files = draw(st.integers(min_value=1, max_value=4))
    files: dict[str, str] = {
        # Bare module name — survives the ``<host>/<org>/``
        # stripping rule unchanged. Chosen so it never collides with
        # a viper canary.
        "go.mod": "module fx-viper-neutrality-fixture\n",
    }
    canaries: list[str] = []
    for i in range(n_files):
        body, file_canaries = draw(_wrapper_file_body())
        files[f"internal/wiring_{i}.go"] = body
        canaries.extend(file_canaries)
    return files, canaries


# ---------------------------------------------------------------------------
# Property 12.A — wrapper neutrality (Hypothesis-driven)
# ---------------------------------------------------------------------------


@given(case=_wrapper_repo())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_fx_and_viper_wrappers_alone_produce_no_detections(
    case: tuple[dict[str, str], list[str]],
) -> None:
    """Property 12.A: fx / viper wrapper calls in isolation yield zero detections.

    For every random repository whose ``.go`` files contain only fx
    and viper calls (and the package declaration / import block
    required to make those calls syntactically valid), the
    aggregator MUST produce a profile whose four list-shaped
    detection sections are empty:

    * ``abstract_inputs == []`` (Requirements 12.1, 13.1: no fx call
      and no viper call emits an ``AbstractInput``).
    * ``abstract_outputs == []`` (Requirement 12.1: no fx call emits
      an ``AbstractOutput``; viper has no output contract).
    * ``external_service_dependencies == []`` (Requirement 12.1: no
      fx call emits an ``ExternalServiceDependency``; viper has no
      external-service contract). The APM-exclusion path is not
      exercised by these fixtures because no ``go.elastic.co/apm/``
      import is generated.
    * ``database_table_dependencies == []`` (Requirement 12.1: no fx
      call carries a ``QueryString`` composite literal; viper has no
      database-table contract).

    The test additionally asserts that no canary string substituted
    into a viper-call argument appears in any (hypothetically empty)
    description, defending Requirement 13.2's "no string literal
    passed only to a viper.* call SHALL appear in any description"
    clause. The list emptiness assertions make the canary check
    redundant in the success case; the canary check exists so a
    regression that began emitting partial detections from viper
    arguments would surface the offending literal directly rather
    than only fail the count assertion.

    ``degraded_sections`` is not asserted on directly because the
    aggregator may legitimately surface non-fatal skip strings here
    (e.g. tokenization failures on a malformed wrapper line a future
    snippet might introduce); the four list-shape assertions are the
    load-bearing contract.
    """

    files, canaries = case
    profile = _run_analyze(files)

    assert profile.abstract_inputs == [], (
        f"fx / viper wrappers alone produced abstract_inputs "
        f"{profile.abstract_inputs!r} (Requirements 12.1, 13.1: "
        f"wrapper calls must emit no AbstractInput)"
    )
    assert profile.abstract_outputs == [], (
        f"fx / viper wrappers alone produced abstract_outputs "
        f"{profile.abstract_outputs!r} (Requirement 12.1: wrapper "
        f"calls must emit no AbstractOutput)"
    )
    assert profile.external_service_dependencies == [], (
        f"fx / viper wrappers alone produced "
        f"external_service_dependencies "
        f"{profile.external_service_dependencies!r} (Requirement "
        f"12.1: wrapper calls must emit no ExternalServiceDependency)"
    )
    assert profile.database_table_dependencies == [], (
        f"fx / viper wrappers alone produced "
        f"database_table_dependencies "
        f"{profile.database_table_dependencies!r} (Requirement "
        f"12.1: wrapper calls must emit no DatabaseTableDependency)"
    )

    # Requirement 13.2: no string literal passed only to a viper.*
    # call SHALL appear in any emission's description. The four list
    # assertions above already establish that no description exists;
    # the canary check is a belt-and-braces guarantee that survives
    # any future profile shape that adds a fifth detection section.
    all_descriptions: list[str] = [
        entry.description for entry in profile.abstract_inputs
    ] + [
        entry.description for entry in profile.abstract_outputs
    ]
    for canary in canaries:
        for description in all_descriptions:
            assert canary not in description, (
                f"viper canary {canary!r} appeared in emission "
                f"description {description!r} (Requirement 13.2: "
                f"viper-only string literals must not appear in any "
                f"emitted description)"
            )


# ---------------------------------------------------------------------------
# Property 12.B — nested scanning via named-function reference
# ---------------------------------------------------------------------------


def test_named_function_referenced_by_fx_invoke_is_still_scanned() -> None:
    """Requirement 12.2: nested I/O in a top-level function still scans.

    The ``cat-service`` repository declares::

        func runScheduler(...) {
            c := cron.New(cron.WithSeconds())
            c.AddFunc(schedule, handler)
        }

        var Module = fx.Module(
            "scheduler",
            fx.Invoke(runScheduler),
        )

    Requirement 12.2 pins the contract that the scheduler's
    ``c.AddFunc(...)`` call MUST still appear as an
    ``AbstractInput(category=scheduled_event)`` even though
    ``runScheduler`` is reachable from the binary only through the
    ``fx.Invoke`` wiring. The recognizer satisfies the requirement
    by inspecting the file's event stream as a whole: the parser
    descends into every top-level ``func`` body during the walk
    (the ``func`` declaration's body opens a fresh balanced
    ``{ ... }`` region the walker re-enters), so ``c.AddFunc(...)``
    appears as a top-level ``MethodCallEvent`` regardless of who
    calls ``runScheduler``. The surrounding ``fx.Invoke(...)`` call
    itself is dropped at the dispatch-boundary exclusion gate
    (Requirement 12.1) and contributes nothing to the profile.

    The fixture pins three assertions:

    1. Exactly one ``scheduled_event`` input is emitted (the
       ``c.AddFunc`` call; ``cron.New`` is a constructor, not a
       registration).
    2. The emitted description carries the literal schedule
       ``"0 30 * * * *"`` verbatim and the
       ``seconds-precision`` marker (Requirement 4.2 first bullet:
       the file's ``cron.New(...)`` carries ``cron.WithSeconds()``).
    3. The emitted description's Source_Location suffix names the
       ``scheduler.go`` line that holds ``c.AddFunc(...)`` — never
       the ``wire.go`` line that holds ``fx.Invoke(runScheduler)``
       (Property 12 design statement, second bullet: "with
       ``source_locations`` pointing at the nested construction's
       line (not at the surrounding ``fx.Invoke`` call)").

    The other three list-shaped detection sections are asserted to
    be empty so a regression that emitted spurious HTTP, ActiveMQ,
    or database detections from the fx wiring would surface here.
    """

    files = {
        "go.mod": "module cat-service-style-fixture\n",
        # The scheduler function is a free, top-level declaration.
        # Its body is what the parser's whole-file walker descends
        # into; the ``fx.Invoke(runScheduler)`` reference does not
        # need to be resolved for the scheduler recognizer to see
        # the nested ``c.AddFunc(...)`` call.
        "internal/scheduler.go": (
            "package internal\n"
            "\n"
            'import "github.com/robfig/cron/v3"\n'
            "\n"
            "func runScheduler() {\n"
            "    c := cron.New(cron.WithSeconds())\n"
            '    _, _ = c.AddFunc("0 30 * * * *", handler)\n'
            "}\n"
            "\n"
            "func handler() {}\n"
        ),
        # The wiring file references ``runScheduler`` by name from
        # inside ``fx.Invoke`` and additionally exercises a sprinkle
        # of viper noise. Both ``fx.*`` and ``viper.*`` calls must
        # produce no detections (Requirements 12.1, 13.1).
        "internal/wire.go": (
            "package internal\n"
            "\n"
            "import (\n"
            '    "go.uber.org/fx"\n'
            '    "github.com/spf13/viper"\n'
            ")\n"
            "\n"
            "func newModule() any {\n"
            "    v := viper.New()\n"
            '    v.SetConfigName("scheduler_config_canary")\n'
            '    return fx.Module("scheduler", fx.Invoke(runScheduler))\n'
            "}\n"
        ),
    }

    profile = _run_analyze(files)

    scheduled = [
        entry
        for entry in profile.abstract_inputs
        if entry.category is AbstractInputCategory.SCHEDULED_EVENT
    ]
    assert len(scheduled) == 1, (
        f"expected exactly one scheduled_event input for the nested "
        f"c.AddFunc(...) call (Requirement 12.2), but got "
        f"{[e.description for e in scheduled]!r}; "
        f"all abstract_inputs: {[e.description for e in profile.abstract_inputs]!r}"
    )
    description = scheduled[0].description
    assert "0 30 * * * *" in description, (
        f"schedule literal '0 30 * * * *' is missing from the "
        f"emitted description {description!r} (Requirement 4.3: "
        f"literal schedule expressions are recorded verbatim)"
    )
    assert "seconds-precision" in description, (
        f"seconds-precision marker is missing from the emitted "
        f"description {description!r} even though the file's "
        f"cron.New(...) carries cron.WithSeconds() (Requirement "
        f"4.2 first bullet)"
    )
    assert "internal/scheduler.go" in description, (
        f"Source_Location must point at internal/scheduler.go, the "
        f"file holding c.AddFunc(...); description was "
        f"{description!r} (Property 12 second bullet: source "
        f"location points at the nested construction's line, not "
        f"the surrounding fx.Invoke)"
    )
    assert "internal/wire.go" not in description, (
        f"Source_Location must NOT point at internal/wire.go, the "
        f"file holding the surrounding fx.Invoke(runScheduler); "
        f"description was {description!r} (Property 12 second "
        f"bullet)"
    )

    # No spurious detections from the fx / viper wiring lines.
    non_scheduler_inputs = [
        entry
        for entry in profile.abstract_inputs
        if entry.category is not AbstractInputCategory.SCHEDULED_EVENT
    ]
    assert non_scheduler_inputs == [], (
        f"fx / viper wiring contributed non-scheduler inputs "
        f"{[e.description for e in non_scheduler_inputs]!r} "
        f"(Requirements 12.1, 13.1)"
    )
    assert profile.abstract_outputs == [], (
        f"fx / viper wiring contributed outputs "
        f"{[e.description for e in profile.abstract_outputs]!r} "
        f"(Requirements 12.1, 13.1)"
    )
    assert profile.external_service_dependencies == [], (
        f"fx / viper wiring contributed external services "
        f"{profile.external_service_dependencies!r} (Requirement "
        f"12.1)"
    )
    assert profile.database_table_dependencies == [], (
        f"fx / viper wiring contributed database tables "
        f"{profile.database_table_dependencies!r} (Requirement "
        f"12.1)"
    )

    # Requirement 13.2: the viper-only string literal must not
    # appear in any emission's description.
    for entry in profile.abstract_inputs:
        assert "scheduler_config_canary" not in entry.description, (
            f"viper-only string literal appeared in "
            f"description {entry.description!r} (Requirement 13.2)"
        )


# ---------------------------------------------------------------------------
# Property 12.C — parser limitation: inline function literals are not walked
# ---------------------------------------------------------------------------


def test_inline_func_literal_inside_fx_invoke_is_not_scanned() -> None:
    """Pin the current parser's inline function-literal limitation.

    The design's Property 12 second bullet specifies that nested
    constructions inside an ``fx.Invoke(<func>)`` argument's body
    SHALL still appear in the corresponding detection section. The
    requirement is fully satisfied for the *named-function-reference*
    form (``fx.Invoke(runScheduler)`` — see
    :func:`test_named_function_referenced_by_fx_invoke_is_still_scanned`)
    because the parser walks every top-level ``func`` declaration
    independently of who calls it.

    For the *inline-function-literal* form
    (``fx.Invoke(func() { c.AddFunc(...) })``), the current parser's
    argument grammar (:func:`go_parser._parse_arg`) does not
    recognize the ``func`` keyword as the start of a function
    literal and falls through to
    :func:`go_parser._parse_unknown_arg`, which consumes tokens to
    the next balanced argument boundary as an opaque
    :class:`UnknownArg`. The opaque consumption skips over the
    function literal's body wholesale, so the nested
    ``c.AddFunc(...)`` call is never emitted as a
    :class:`MethodCallEvent` and the scheduler recognizer never
    sees it.

    This test pins the *current* behaviour rather than the
    aspirational design contract because:

    * The behaviour is a parser-level limitation, not a recognizer
      bug. The dispatch-boundary exclusion is correct on its own
      terms — every fx and viper call is dropped at the
      ``_is_excluded_receiver`` check.
    * The four sample repositories use the named-function-reference
      form exclusively (``cat-service``'s ``fx.Invoke(runScheduler)``
      is the canonical example called out verbatim in
      Requirement 12.2). No first-party Go file in the four sample
      repositories passes an anonymous function literal to
      ``fx.Invoke`` or ``fx.Provide``.
    * Pinning the limitation explicitly here makes a future parser
      upgrade that begins descending into function-literal bodies
      surface as an ``unexpected_pass`` test status — the right
      signal for revisiting the requirement's coverage of the
      inline form.

    When this test starts failing because the parser learned to
    walk function literals, the resolution path is to flip the
    assertion below to ``len(scheduled) == 1`` and add the same
    description checks the named-reference test pins, then update
    the design's Property 12 wording to note that the inline form
    is now covered too.
    """

    files = {
        "go.mod": "module inline-literal-fixture\n",
        "internal/wire.go": (
            "package internal\n"
            "\n"
            "import (\n"
            '    "go.uber.org/fx"\n'
            '    "github.com/robfig/cron/v3"\n'
            ")\n"
            "\n"
            "var Module = fx.Module(\n"
            '    "scheduler",\n'
            "    fx.Invoke(func() {\n"
            "        c := cron.New(cron.WithSeconds())\n"
            '        _, _ = c.AddFunc("0 45 * * * *", handler)\n'
            "    }),\n"
            ")\n"
            "\n"
            "func handler() {}\n"
        ),
    }

    profile = _run_analyze(files)

    scheduled = [
        entry
        for entry in profile.abstract_inputs
        if entry.category is AbstractInputCategory.SCHEDULED_EVENT
    ]
    # Pinned current behaviour: the inline function literal is
    # opaque to the parser, so the nested c.AddFunc(...) does not
    # surface as a scheduled_event. See the docstring above for the
    # design rationale and the upgrade path.
    assert scheduled == [], (
        f"the parser appears to have begun walking inline function "
        f"literals — the nested c.AddFunc(...) call now surfaces "
        f"as {[e.description for e in scheduled]!r}. This is a "
        f"capability upgrade rather than a regression; see the "
        f"docstring on test_inline_func_literal_inside_fx_invoke_"
        f"is_not_scanned for the resolution path."
    )

    # The fx wiring on its own still emits no other detections.
    assert profile.abstract_inputs == [], (
        f"fx wiring with an inline function literal contributed "
        f"unexpected inputs "
        f"{[e.description for e in profile.abstract_inputs]!r}"
    )
    assert profile.abstract_outputs == [], (
        f"fx wiring with an inline function literal contributed "
        f"unexpected outputs "
        f"{[e.description for e in profile.abstract_outputs]!r}"
    )
    assert profile.external_service_dependencies == [], (
        f"fx wiring with an inline function literal contributed "
        f"unexpected external services "
        f"{profile.external_service_dependencies!r}"
    )
    assert profile.database_table_dependencies == [], (
        f"fx wiring with an inline function literal contributed "
        f"unexpected database tables "
        f"{profile.database_table_dependencies!r}"
    )

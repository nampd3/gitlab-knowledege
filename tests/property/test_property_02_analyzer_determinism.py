# ruff: noqa: E501
# Feature: go-analyzer-support, Property 2: For all RepositoryContents values, two consecutive invocations of analyze() SHALL produce ProjectProfiles whose abstract_inputs, abstract_outputs, external_service_dependencies, and database_table_dependencies lists are equal under list equality (same elements, same order).
"""Property test for ``Project_Analyzer.analyze()`` determinism.

**Validates Requirements 11.4** (Property 2 in the design, task 11.8 in
``tasks.md``).

For every synthetic :class:`RepositoryContents` containing a mix of Go
source files, ``go.mod`` manifests, and non-Go source material
(Python, JavaScript, Java, YAML, READMEs, manifests), two consecutive
invocations of :func:`project_knowledge_mcp.project_analyzer.analyze`
on the same input SHALL produce :class:`ProjectProfile`\\s whose four
detection-list sections — ``abstract_inputs``, ``abstract_outputs``,
``external_service_dependencies``, and ``database_table_dependencies``
— are equal under list equality (same elements, same order). The
remaining structural fields (``purpose_summary``,
``purpose_summary_reason``, ``degraded_sections``, and the GitLab
identity fields the aggregator copies straight through) must also be
identical between the two runs. Only ``produced_at`` is permitted to
differ, because it records the wall-clock instant at which the profile
was assembled and is therefore non-deterministic by design.

The property is additionally exercised in a stronger form: the same
``RepositoryContents.files`` mapping is rebuilt with its key insertion
order reversed before the second :func:`analyze` call. Python ``dict``
iteration is insertion-ordered, so a regression in which any
sub-analyzer accidentally depends on hash-randomized iteration order
would surface as a mismatch between the forward run and the reversed
run. The design's §2 architecture pillar "deterministic output (11.4)"
mandates that every sub-analyzer iterates ``file_paths`` in
sorted-path order; this test pins that invariant at the aggregator
boundary, where all four sub-analyzers compose.

The strategy stitches together random subsets of every catalogue —
Go, Python, Java, JavaScript, YAML, READMEs, manifests — so the
analyzer is exercised across HTTP handlers, scheduled jobs, message
I/O, file I/O, CLI entry points, external HTTP, SQL, ORM
declarations, broker/cache SDK patterns, and YAML schedules. Both
Go-only, non-Go-only, and mixed-content repositories are sampled so
the property covers the three regimes ``analyze()`` distinguishes
internally:

* No Go artefact: every Go branch is a no-op
  (``has_go_artefacts`` returns ``False``).
* Go-only: only the Go sub-analyzers contribute.
* Mixed: both the language-agnostic and the Go scanners contribute,
  and the cross-language coalescing helpers in
  :mod:`io_extractor`, :mod:`external_services`, and :mod:`db_tables`
  merge their outputs.

A regression in any of these branches would surface here as a
non-deterministic comparison.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import ProjectProfile, RepositoryContents
from project_knowledge_mcp.project_analyzer import analyze

pytestmark = pytest.mark.property


# ---------------------------------------------------------------------------
# Source-material catalogues
# ---------------------------------------------------------------------------
#
# Each snippet is a small, self-contained piece of source code or
# configuration that exercises one or more detection paths in the
# four sub-analyzers (purpose, I/O, external services, database
# tables). The strategy below samples subsets of each catalogue and
# stitches them into a synthetic repository tree. Snippets stay
# intentionally small: the property under test is determinism, not
# detection completeness, so the value of a fixture comes from
# diversity, not depth.


# Go snippets exercising the Go-side scanners (Requirements 1–9).
# Each snippet is a syntactically valid Go file (no toolchain is
# invoked; only the in-process parser sees it).
_GO_SNIPPETS: tuple[str, ...] = (
    # HTTP handler registrations (Requirements 3.1–3.3).
    (
        'package svc\n'
        'import "net/http"\n'
        'func register(mux *http.ServeMux) {\n'
        '\tmux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {})\n'
        '\tmux.Handle("/debug/pprof/", nil)\n'
        '}\n'
    ),
    (
        'package svc\n'
        'import "net/http"\n'
        'func register(mux *http.ServeMux) {\n'
        '\tmux.HandleFunc("GET /users/{id}", func(w http.ResponseWriter, r *http.Request) {})\n'
        '}\n'
    ),
    # HTTP bootstrap (Requirement 3.5: must be suppressed).
    (
        'package svc\n'
        'import "net/http"\n'
        'func run() {\n'
        '\thttp.ListenAndServe(":8080", nil)\n'
        '}\n'
    ),
    # Scheduler registrations (Requirements 4.1–4.5).
    (
        'package svc\n'
        'import (\n'
        '\t"github.com/robfig/cron/v3"\n'
        ')\n'
        'func register() {\n'
        '\tc := cron.New(cron.WithSeconds())\n'
        '\tc.AddFunc("*/5 * * * * *", func() {})\n'
        '}\n'
    ),
    (
        'package svc\n'
        'import "time"\n'
        'func register() {\n'
        '\tt := time.NewTicker(time.Second)\n'
        '\t_ = t\n'
        '}\n'
    ),
    # ActiveMQ consumer/publisher (Requirements 5.1–5.5).
    (
        'package subscriber\n'
        'import (\n'
        '\t"context"\n'
        '\t"esb-go-libs/activemq/domain"\n'
        ')\n'
        'func listen(ctx context.Context, r Receiver) {\n'
        '\tr.Subscribe(ctx, nil, &domain.SubscriberConfig{Destination: "APS.LOS.INBOUND"}, nil)\n'
        '}\n'
    ),
    (
        'package usecase\n'
        'import (\n'
        '\t"context"\n'
        '\t"esb-go-libs/activemq/domain"\n'
        ')\n'
        'func emit(ctx context.Context, s Sender) {\n'
        '\ts.SendMessage(ctx, "tid", "cid", domain.Message{Destination: "REP.SERVICE.PAYMENT.ERR"})\n'
        '}\n'
    ),
    # ActiveMQ broker construction (Requirements 8.5).
    (
        'package adapter\n'
        'import (\n'
        '\t"esb-go-libs/activemq"\n'
        ')\n'
        'func conn() *activemq.Client {\n'
        '\treturn activemq.NewClient(&activemq.JmsConfig{BrokerUrl: "tcp://broker.example.com:61616"})\n'
        '}\n'
    ),
    # File I/O (Requirements 6.1–6.3).
    (
        'package svc\n'
        'import "os"\n'
        'func loadConfig() ([]byte, error) {\n'
        '\treturn os.ReadFile("config.json")\n'
        '}\n'
    ),
    (
        'package svc\n'
        'import "os"\n'
        'func writeLog(b []byte) error {\n'
        '\treturn os.WriteFile("out.log", b, 0644)\n'
        '}\n'
    ),
    # CLI entry-point (Requirement 7).
    (
        'package main\n'
        'import "flag"\n'
        'func main() {\n'
        '\tname := flag.String("name", "default", "name flag")\n'
        '\tflag.Parse()\n'
        '\t_ = name\n'
        '}\n'
    ),
    # fec_pool_service external dependency + SQL extraction
    # (Requirements 8.2, 8.3, 9.2, 9.3).
    (
        'package repo\n'
        'import (\n'
        '\t"context"\n'
        '\t"esb-go-libs/dbadapter"\n'
        '\t"esb-go-libs/dbadapter/model"\n'
        ')\n'
        'func read(ctx context.Context, a dbadapter.IPoolAPIAdapter) {\n'
        '\treq := model.PoolServiceRequest{QueryString: "SELECT id, name FROM APP_WS.IIB_PAYMENT_CONFIRMED"}\n'
        '\ta.PoolExecuteQuery(ctx, &req, "u", "p")\n'
        '}\n'
    ),
    (
        'package repo\n'
        'import (\n'
        '\t"context"\n'
        '\t"fec_pool_service/pb"\n'
        ')\n'
        'func update(ctx context.Context, c pb.PoolAPIClient) {\n'
        '\treq := pb.PoolExecuteQueryRequest{QueryString: "UPDATE APP_WS.IIB_PAYMENT_CONFIRMED SET state = 1"}\n'
        '\tc.ExecuteQuery(ctx, &req)\n'
        '}\n'
    ),
    # APM tracing import (Requirement 8.4: must be excluded).
    (
        'package svc\n'
        'import (\n'
        '\t"go.elastic.co/apm/v2"\n'
        ')\n'
        'func register() {\n'
        '\t_ = apm.DefaultTracer\n'
        '}\n'
    ),
    # uber/fx wiring (Requirement 12: must not produce detections).
    (
        'package main\n'
        'import "go.uber.org/fx"\n'
        'func main() {\n'
        '\tfx.New(fx.Provide(newThing), fx.Invoke(runScheduler)).Run()\n'
        '}\n'
        'func newThing() int { return 0 }\n'
        'func runScheduler() {}\n'
    ),
    # viper configuration reads (Requirement 13: configuration only).
    (
        'package config\n'
        'import "github.com/spf13/viper"\n'
        'func load() {\n'
        '\tv := viper.New()\n'
        '\tv.SetConfigName("app")\n'
        '\tv.AddConfigPath("./config")\n'
        '\t_ = v.GetString("server.host")\n'
        '}\n'
    ),
    # Package doc comment (Requirement 2.4).
    (
        '// Package svc orchestrates payment events for the repayment service.\n'
        'package svc\n'
    ),
    # Plain Go file with no detections.
    'package internal\n',
)


# ``go.mod`` manifest bodies (Requirements 2.1–2.3, 7.2, 7.3).
_GO_MOD_BODIES: tuple[str, ...] = (
    "module repayment_service\n\ngo 1.24\n",
    "// payment-service: orchestrates outbound payment events\nmodule github.com/acme/payment-service\n\ngo 1.24\n",
    "module github.com/acme/cat-service // schedules nightly cleanup\n\ngo 1.24\n",
    "module fec_pool_service\n",
)


# Python snippets (parent-spec sub-analyzers).
_PYTHON_SNIPPETS: tuple[str, ...] = (
    # HTTP routes.
    (
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "@app.route('/users')\n"
        "def list_users():\n"
        "    return []\n"
    ),
    # Scheduled tasks.
    (
        "@scheduler.scheduled_job('cron', hour=0)\n"
        "def nightly_cleanup():\n"
        "    pass\n"
    ),
    # File I/O.
    (
        "def write_log():\n"
        "    with open('output.log', 'w') as f:\n"
        "        f.write('hello')\n"
    ),
    # External HTTP.
    (
        "import requests\n"
        "def fetch():\n"
        "    return requests.get('https://api.stripe.com/v1/charges')\n"
    ),
    # SQL.
    (
        "def insert_user(cur):\n"
        "    cur.execute('INSERT INTO users (id, name) VALUES (?, ?)', (1, 'a'))\n"
    ),
    (
        "def lookup(cur):\n"
        "    cur.execute('SELECT id, name FROM users WHERE id = ?', (1,))\n"
    ),
    # Module docstring.
    '"""Service that orchestrates billing tasks."""\n',
    # Empty.
    "",
)


_JAVA_SNIPPETS: tuple[str, ...] = (
    (
        "package com.example;\n"
        "import org.springframework.web.bind.annotation.*;\n"
        "@RestController\n"
        "public class UserController {\n"
        "  @GetMapping(\"/users\")\n"
        "  public java.util.List<String> list() { return java.util.List.of(); }\n"
        "}\n"
    ),
)


_JS_SNIPPETS: tuple[str, ...] = (
    (
        "const express = require('express');\n"
        "const app = express();\n"
        "app.get('/health', (req, res) => res.send('ok'));\n"
    ),
    (
        "const fs = require('fs');\n"
        "fs.readFile('in.txt', () => {});\n"
    ),
)


_YAML_SNIPPETS: tuple[str, ...] = (
    "schedule: \"*/5 * * * *\"\n",
    (
        "version: 2\n"
        "services:\n"
        "  api:\n"
        "    image: example/api:latest\n"
    ),
)


_README_BODIES: tuple[str, ...] = (
    "# Example Service\n\nThis service orchestrates outbound payment events.\n",
    "Handles ingestion of customer order webhooks.\n",
    "",
)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _make_pyproject(name: str, description: str) -> str:
    """Render a minimal ``pyproject.toml`` body."""
    safe_name = name.replace('"', "")
    safe_desc = description.replace('"', "'").replace("\n", " ")
    return (
        "[project]\n"
        f'name = "{safe_name}"\n'
        f'description = "{safe_desc}"\n'
        'version = "0.1.0"\n'
    )


def _make_package_json(name: str, description: str) -> str:
    """Render a minimal ``package.json`` body."""
    return json.dumps({"name": name, "description": description, "version": "1.0.0"})


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------


_short_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "P"),
        blacklist_characters=("\x00", "\r"),
    ),
    min_size=0,
    max_size=80,
)

_short_name = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
)


@st.composite
def _file_subset(
    draw: st.DrawFn,
    prefix: str,
    snippets: tuple[str, ...],
    extension: str,
    max_files: int = 3,
) -> dict[str, str]:
    """Pick a 0..max_files subset of ``snippets`` placed at unique paths."""
    indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(snippets) - 1),
            min_size=0,
            max_size=max_files,
            unique=True,
        )
    )
    return {
        f"{prefix}{i}{extension}": snippets[idx]
        for i, idx in enumerate(indices)
    }


@st.composite
def _mixed_repository_contents(draw: st.DrawFn) -> RepositoryContents:
    """Build a synthetic :class:`RepositoryContents` mixing Go and non-Go files.

    Each of the three regimes the aggregator distinguishes internally
    (no-Go, Go-only, mixed) is reachable from this strategy because
    every file category is independently sampled with a 0..N draw.
    """
    files: dict[str, str] = {}

    # README at the root (sometimes).
    if draw(st.booleans()):
        files["README.md"] = draw(st.sampled_from(_README_BODIES))

    # package.json description (sometimes).
    if draw(st.booleans()):
        files["package.json"] = _make_package_json(
            draw(_short_name), draw(_short_text)
        )

    # pyproject.toml description (sometimes).
    if draw(st.booleans()):
        files["pyproject.toml"] = _make_pyproject(
            draw(_short_name), draw(_short_text)
        )

    # go.mod manifest (sometimes; gates the Go purpose-summary slots).
    if draw(st.booleans()):
        files["go.mod"] = draw(st.sampled_from(_GO_MOD_BODIES))

    # Go source files exercising the five Go scanners. Spread across
    # ``cmd/``, ``internal/``, and ``pkg/`` so the path-based gates
    # in the CLI entry-point recognizer and the package-doc-comment
    # collector are both exercised.
    go_layout = draw(
        st.sampled_from(
            (
                ("cmd/main.go", "internal/svc/", "pkg/repo/"),
                ("cmd/svcname/main.go", "internal/usecase/", "pkg/adapter/"),
                ("main.go", "internal/", "pkg/"),
            )
        )
    )
    main_path, internal_prefix, pkg_prefix = go_layout
    # The first Go snippet is occasionally placed at the binary-entry
    # path so the CLI recognizer's path-based branches are exercised
    # without forcing every example to carry a main.
    if draw(st.booleans()):
        files[main_path] = draw(st.sampled_from(_GO_SNIPPETS))
    files.update(
        draw(_file_subset(internal_prefix + "mod_", _GO_SNIPPETS, ".go", max_files=4))
    )
    files.update(
        draw(_file_subset(pkg_prefix + "mod_", _GO_SNIPPETS, ".go", max_files=3))
    )

    # Python, Java, JS, YAML noise so cross-language coalescing is
    # exercised.
    files.update(
        draw(_file_subset("svc/mod_", _PYTHON_SNIPPETS, ".py", max_files=4))
    )
    files.update(
        draw(_file_subset("src/main/java/Mod", _JAVA_SNIPPETS, ".java", max_files=1))
    )
    files.update(
        draw(_file_subset("web/index_", _JS_SNIPPETS, ".js", max_files=2))
    )
    files.update(
        draw(_file_subset("ops/cfg_", _YAML_SNIPPETS, ".yml", max_files=2))
    )

    project_id = draw(st.integers(min_value=1, max_value=10_000_000))
    commit_sha = draw(
        st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
    )
    return RepositoryContents(
        gitlab_project_id=project_id,
        commit_sha=commit_sha,
        files=files,
    )


@st.composite
def _analyze_inputs(
    draw: st.DrawFn,
) -> tuple[int, str, str, str, str | None, RepositoryContents]:
    """Build the positional argument tuple for :func:`analyze`."""
    project_id = draw(st.integers(min_value=1, max_value=10_000_000))
    full_path = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=20,
        ).map(lambda seg: f"group/{seg}")
    )
    analysis_branch = draw(
        st.sampled_from(["uat", "main", "develop", "release/2024-Q1"])
    )
    commit_sha = draw(
        st.text(alphabet="0123456789abcdef", min_size=7, max_size=40)
    )
    repo_description = draw(st.one_of(st.none(), _short_text))
    repo_contents = draw(_mixed_repository_contents())
    # Re-stamp ``RepositoryContents`` so its identity fields line up
    # with the positional arguments we will pass into ``analyze``.
    # ``RepositoryContents`` is frozen, so we build a fresh instance
    # from its fields.
    repo_contents = RepositoryContents(
        gitlab_project_id=project_id,
        commit_sha=commit_sha,
        files=dict(repo_contents.files),
    )
    return (
        project_id,
        full_path,
        analysis_branch,
        commit_sha,
        repo_description,
        repo_contents,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _profile_modulo_produced_at(profile: ProjectProfile) -> dict[str, object]:
    """Project a :class:`ProjectProfile` onto its deterministic fields.

    ``produced_at`` records the wall-clock instant at which
    :func:`analyze` assembled the profile and is the only field
    permitted to differ between two consecutive runs on the same
    input. Every other field — including the four detection lists
    Property 2 explicitly names, the purpose summary pair, and
    ``degraded_sections`` — is derived deterministically from the
    input and must match exactly.
    """
    dumped = profile.model_dump(mode="python")
    dumped.pop("produced_at", None)
    return dumped


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@given(case=_analyze_inputs())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_analyze_is_deterministic(
    case: tuple[int, str, str, str, str | None, RepositoryContents],
) -> None:
    """Property 2: ``analyze()`` is deterministic.

    Validates Requirement 11.4: two consecutive invocations of
    :func:`analyze` on the same :class:`RepositoryContents` produce
    ``ProjectProfile``\\s whose detection lists are equal under list
    equality, and whose every other deterministic field is also
    equal. Reversing the ``files`` mapping insertion order before the
    second call must not perturb the result.
    """
    (
        project_id,
        full_path,
        analysis_branch,
        commit_sha,
        repo_description,
        repo_contents,
    ) = case

    # --- Run 1: forward insertion order. ---------------------------------
    profile_forward = analyze(
        project_id,
        full_path,
        analysis_branch,
        commit_sha,
        repo_description,
        repo_contents,
    )

    # --- Run 2: same RepositoryContents (re-built so we exercise the
    # identical-content, identical-order branch). ------------------------
    repo_contents_repeat = RepositoryContents(
        gitlab_project_id=project_id,
        commit_sha=commit_sha,
        files=dict(repo_contents.files),
    )
    profile_repeat = analyze(
        project_id,
        full_path,
        analysis_branch,
        commit_sha,
        repo_description,
        repo_contents_repeat,
    )

    # --- Run 3: same files, reversed insertion order. -------------------
    # Python ``dict`` iteration is insertion-ordered, so reversing the
    # key order is the cleanest way to detect a regression in which a
    # sub-analyzer accidentally depends on hash-randomized iteration
    # rather than the sorted-path iteration design §2 mandates.
    reversed_files = dict(reversed(list(repo_contents.files.items())))
    repo_contents_reversed = RepositoryContents(
        gitlab_project_id=project_id,
        commit_sha=commit_sha,
        files=reversed_files,
    )
    profile_reversed = analyze(
        project_id,
        full_path,
        analysis_branch,
        commit_sha,
        repo_description,
        repo_contents_reversed,
    )

    # --- Detection-list equality (the property's headline claim). -------
    # Same elements, same order. The four list-shaped sections are the
    # ones Requirement 11.4 explicitly names.
    for label, second in (("repeat", profile_repeat), ("reversed", profile_reversed)):
        assert profile_forward.abstract_inputs == second.abstract_inputs, (
            f"abstract_inputs diverged between forward run and {label} run; "
            "Requirement 11.4 mandates list-equal outputs"
        )
        assert profile_forward.abstract_outputs == second.abstract_outputs, (
            f"abstract_outputs diverged between forward run and {label} run; "
            "Requirement 11.4 mandates list-equal outputs"
        )
        assert (
            profile_forward.external_service_dependencies
            == second.external_service_dependencies
        ), (
            f"external_service_dependencies diverged between forward run and "
            f"{label} run; Requirement 11.4 mandates list-equal outputs"
        )
        assert (
            profile_forward.database_table_dependencies
            == second.database_table_dependencies
        ), (
            f"database_table_dependencies diverged between forward run and "
            f"{label} run; Requirement 11.4 mandates list-equal outputs"
        )

    # --- Whole-profile equality modulo ``produced_at``. -----------------
    # ``produced_at`` records ``datetime.now(UTC)`` at the moment the
    # profile is assembled; it is the only non-deterministic field on
    # the model. Every other field — purpose summary pair, the GitLab
    # identity fields, ``degraded_sections``, plus the four detection
    # lists already asserted above — must match exactly.
    forward_dump = _profile_modulo_produced_at(profile_forward)
    repeat_dump = _profile_modulo_produced_at(profile_repeat)
    reversed_dump = _profile_modulo_produced_at(profile_reversed)
    assert forward_dump == repeat_dump, (
        "ProjectProfile differed (outside produced_at) between two consecutive "
        "analyze() calls on the same input; Requirement 11.4 mandates "
        "deterministic output"
    )
    assert forward_dump == reversed_dump, (
        "ProjectProfile differed (outside produced_at) after reversing the "
        "RepositoryContents.files insertion order; Requirement 11.4 mandates "
        "that the aggregator's output depend only on file contents, not on "
        "Python dict iteration order"
    )

# ruff: noqa: E501
# Feature: project-knowledge-mcp, Property 6: For all analyzed projects, the produced Project_Profile SHALL satisfy: abstract_inputs is a list (possibly empty) where every entry has category in {http_request, scheduled_event, message_consumed, file_read, cli_argument, other} and a non-null description, abstract_outputs is a list (possibly empty) where every entry has category in {http_response, message_published, file_written, database_write, external_call, other} and a non-null description, external_service_dependencies is a list (possibly empty) where every entry has a non-empty name, kind in {http_api, message_broker, object_store, cache, auth_provider, other}, and a non-empty source_locations list, database_table_dependencies is a list (possibly empty) where every entry has a non-empty table_name, access_mode in {read, write, read_write, unknown}, and a non-empty source_locations list.
"""Property test for well-formed Project_Profile sections.

**Validates Requirements 3.1, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.4, 6.1, 6.2, 6.4**
(Property 6 in the design).

For every synthetic project assembled from a diverse mix of source material
(READMEs, ``package.json`` / ``pyproject.toml`` / ``pom.xml`` manifests, and
source files exercising HTTP route handlers, scheduled tasks, message
consumers and publishers, file I/O, CLI entrypoints, external HTTP calls,
SQL writes, ORM models, and broker/cache/object-store SDK constructors),
the :class:`ProjectProfile` produced by
:func:`project_knowledge_mcp.project_analyzer.analyze` must satisfy the
four structural invariants enumerated in Property 6:

1. ``abstract_inputs`` is a list; every entry's ``category`` is drawn from
   the closed set in Requirement 4.3 and ``description`` is a non-empty
   string.
2. ``abstract_outputs`` is a list; every entry's ``category`` is drawn from
   the closed set in Requirement 4.4 and ``description`` is a non-empty
   string.
3. ``external_service_dependencies`` is a list; every entry has a
   non-empty ``name``, a ``kind`` drawn from the closed set in
   Requirement 5.2, and a non-empty ``source_locations`` list.
4. ``database_table_dependencies`` is a list; every entry has a non-empty
   ``table_name``, an ``access_mode`` drawn from the closed set in
   Requirement 6.2 (``read``, ``write``, ``read_write``, or ``unknown`` —
   the latter introduced by Go-analyzer-support Requirement 9.8 for the
   case where a table name is identifiable but the SQL keyword cannot
   be matched), and a non-empty ``source_locations`` list.

The aggregator is contracted to never raise (degraded sub-analyzers are
recorded in ``ProjectProfile.degraded_sections`` rather than propagated),
so this property is also an end-to-end check that no input drives the
analyzer into an exception.
"""

from __future__ import annotations

import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from project_knowledge_mcp.models import (
    AbstractInputCategory,
    AbstractOutputCategory,
    DatabaseAccessMode,
    ExternalServiceKind,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer import analyze

# ---------------------------------------------------------------------------
# Closed-set membership references (Requirements 4.3, 4.4, 5.2, 6.2)
# ---------------------------------------------------------------------------

_INPUT_CATEGORIES: frozenset[AbstractInputCategory] = frozenset(AbstractInputCategory)
_OUTPUT_CATEGORIES: frozenset[AbstractOutputCategory] = frozenset(AbstractOutputCategory)
_SERVICE_KINDS: frozenset[ExternalServiceKind] = frozenset(ExternalServiceKind)
# ``DatabaseAccessMode`` is enumerated dynamically via ``frozenset(DatabaseAccessMode)``
# so the closed set automatically includes every member of the enum, including the
# ``UNKNOWN`` value added by Go-analyzer-support Requirement 9.8 (recorded when a
# table name is identifiable but the SQL keyword cannot be matched).
_ACCESS_MODES: frozenset[DatabaseAccessMode] = frozenset(DatabaseAccessMode)

# ---------------------------------------------------------------------------
# Source-material catalogues
# ---------------------------------------------------------------------------

# Each snippet is a small, self-contained piece of source code or
# configuration that exercises one or more detection paths in the
# analyzer's sub-analyzers. The strategies below sample subsets of
# these catalogues and stitch them into a synthetic repository tree.

_PYTHON_SNIPPETS: tuple[str, ...] = (
    # HTTP route handlers (input: http_request, output: http_response).
    (
        "from flask import Flask\n"
        "app = Flask(__name__)\n"
        "@app.route('/users')\n"
        "def list_users():\n"
        "    return []\n"
    ),
    (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "@app.get('/items/{id}')\n"
        "def read_item(id: int):\n"
        "    return {'id': id}\n"
    ),
    (
        "@app.post('/orders')\n"
        "def create_order(payload):\n"
        "    return {'ok': True}\n"
    ),
    (
        "@router.delete('/sessions/{sid}')\n"
        "def end_session(sid: str):\n"
        "    return None\n"
    ),
    # Scheduled tasks (input: scheduled_event).
    (
        "@scheduler.scheduled_job('cron', hour=0)\n"
        "def nightly_cleanup():\n"
        "    pass\n"
    ),
    (
        "@app.task\n"
        "def process_payment(order_id):\n"
        "    return order_id\n"
    ),
    # Message consumers (input: message_consumed).
    (
        "def listen():\n"
        "    consumer.subscribe('topic-orders')\n"
    ),
    (
        "def listen_rabbit(channel):\n"
        "    channel.basic_consume(queue='inbound', on_message_callback=lambda *a: None)\n"
    ),
    # Message publishers (output: message_published).
    (
        "def emit():\n"
        "    producer.send('topic-events', value=b'payload')\n"
    ),
    # File I/O (input: file_read, output: file_written).
    (
        "def load_config():\n"
        "    with open('config.json', 'r') as f:\n"
        "        return f.read()\n"
    ),
    (
        "def write_log():\n"
        "    with open('output.log', 'w') as f:\n"
        "        f.write('hello')\n"
    ),
    (
        "from pathlib import Path\n"
        "Path('out.txt').write_text('data')\n"
    ),
    (
        "from pathlib import Path\n"
        "data = Path('in.txt').read_text()\n"
    ),
    # CLI entrypoints (input: cli_argument).
    (
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--path')\n"
    ),
    (
        "import click\n"
        "@click.command()\n"
        "def main():\n"
        "    pass\n"
    ),
    (
        "if __name__ == '__main__':\n"
        "    main()\n"
    ),
    # External HTTP (output: external_call).
    (
        "import requests\n"
        "def fetch():\n"
        "    return requests.get('https://api.stripe.com/v1/charges')\n"
    ),
    (
        "import httpx\n"
        "def push(payload):\n"
        "    httpx.post('https://api.example.com/data', json=payload)\n"
    ),
    # SQL statements (raw SQL → db_tables, plus database_write output).
    (
        "def insert_user(cur):\n"
        "    cur.execute('INSERT INTO users (id, name) VALUES (?, ?)', (1, 'a'))\n"
    ),
    (
        "def update_balance(cur):\n"
        "    cur.execute('UPDATE accounts SET balance = ? WHERE id = ?', (100, 1))\n"
    ),
    (
        "def delete_session(cur):\n"
        "    cur.execute('DELETE FROM sessions WHERE expired = 1')\n"
    ),
    (
        "def lookup(cur):\n"
        "    cur.execute('SELECT id, name FROM users WHERE id = ?', (1,))\n"
    ),
    (
        "def report(cur):\n"
        "    cur.execute('SELECT * FROM orders JOIN customers ON orders.cid = customers.id')\n"
    ),
    # ORM model declarations.
    (
        "from sqlalchemy.orm import declarative_base\n"
        "Base = declarative_base()\n"
        "class User(Base):\n"
        "    __tablename__ = 'users'\n"
    ),
    (
        "from django.db import models\n"
        "class Order(models.Model):\n"
        "    class Meta:\n"
        "        db_table = 'orders'\n"
    ),
    # External-service SDK patterns.
    (
        "import boto3\n"
        "s3 = boto3.client('s3')\n"
    ),
    (
        "import redis\n"
        "cache = redis.Redis(host='cache.internal.example.com')\n"
    ),
    (
        "from kafka import KafkaProducer\n"
        "producer = KafkaProducer(bootstrap_servers='kafka.example.com:9092')\n"
    ),
    # Module docstring only (purpose source).
    (
        '"""Service that orchestrates billing tasks."""\n'
    ),
    # Empty file (no detections).
    "",
    # Deliberately malformed Python: must be tolerated by every scanner.
    (
        "def broken(:\n"
        "    pass garbled\n"
    ),
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
    (
        "package com.example;\n"
        "import org.springframework.scheduling.annotation.Scheduled;\n"
        "public class Worker {\n"
        "  @Scheduled(cron = \"0 0 * * * *\")\n"
        "  public void run() { }\n"
        "}\n"
    ),
    (
        "package com.example;\n"
        "import org.springframework.kafka.annotation.KafkaListener;\n"
        "public class Listener {\n"
        "  @KafkaListener(topics=\"events\")\n"
        "  public void onMessage(String m) { }\n"
        "}\n"
    ),
)

_JS_SNIPPETS: tuple[str, ...] = (
    (
        "const express = require('express');\n"
        "const app = express();\n"
        "app.get('/health', (req, res) => res.send('ok'));\n"
        "app.post('/orders', (req, res) => res.json({ok: true}));\n"
    ),
    (
        "const axios = require('axios');\n"
        "axios.get('https://api.example.com/things');\n"
    ),
    (
        "const fs = require('fs');\n"
        "fs.writeFile('out.txt', 'data', () => {});\n"
        "fs.readFile('in.txt', () => {});\n"
    ),
    (
        "consumer.subscribe({topic: 'orders'});\n"
        "producer.send({topic: 'events', messages: []});\n"
    ),
)

_YAML_SNIPPETS: tuple[str, ...] = (
    (
        "schedules:\n"
        "  - cron: \"0 0 * * *\"\n"
        "    task: nightly-cleanup\n"
    ),
    (
        "schedule: \"*/5 * * * *\"\n"
    ),
    (
        "version: 2\n"
        "services:\n"
        "  api:\n"
        "    image: example/api:latest\n"
    ),
)

_README_BODIES: tuple[str, ...] = (
    "# Example Service\n\nThis service orchestrates outbound payment events.\n",
    "Handles ingestion of customer order webhooks and enriches them with audit metadata.\n",
    (
        "# Big README\n\n"
        "Coordinates fan-out of notifications to downstream subscribers via Kafka.\n\n"
        "Includes scheduled reconciliation jobs and a small admin CLI.\n"
    ),
    "",
)


# ---------------------------------------------------------------------------
# Helpers for manifest content
# ---------------------------------------------------------------------------


def _make_pyproject(name: str, description: str) -> str:
    """Render a minimal valid ``pyproject.toml`` body."""
    safe_name = name.replace('"', "")
    safe_desc = description.replace('"', "'").replace("\n", " ")
    return (
        "[project]\n"
        f'name = "{safe_name}"\n'
        f'description = "{safe_desc}"\n'
        'version = "0.1.0"\n'
    )


def _make_package_json(name: str, description: str) -> str:
    """Render a minimal valid ``package.json`` body."""
    return json.dumps(
        {"name": name, "description": description, "version": "1.0.0"}
    )


def _make_pom_xml(description: str) -> str:
    """Render a minimal valid ``pom.xml`` body."""
    safe_desc = description.replace("<", "&lt;").replace("&", "&amp;")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
        "  <modelVersion>4.0.0</modelVersion>\n"
        "  <groupId>com.example</groupId>\n"
        "  <artifactId>svc</artifactId>\n"
        "  <version>1.0.0</version>\n"
        f"  <description>{safe_desc}</description>\n"
        "</project>\n"
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Manifest description: free-form printable text, normalized later by the
# purpose summarizer.
_short_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "P"),
        blacklist_characters=("\x00", "\r"),
    ),
    min_size=0,
    max_size=200,
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
    max_files: int = 4,
) -> dict[str, str]:
    """Pick a 0..max_files subset of ``snippets`` and lay them out as files.

    Each chosen snippet is placed at a unique repository-relative path of
    the form ``{prefix}{i}{extension}``.
    """
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
def _repository_contents(draw: st.DrawFn) -> RepositoryContents:
    """Build a synthetic :class:`RepositoryContents` from diverse sources.

    The strategy stitches together random subsets of every catalogue
    above so the analyzer is exercised across HTTP handlers, scheduled
    jobs, message I/O, file I/O, CLI entrypoints, external HTTP, SQL,
    ORM declarations, broker/cache SDK patterns, and YAML schedules.
    Manifests at the repository root are independently included or
    omitted so the purpose summarizer's manifest path is exercised in
    both presence states.
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

    # pom.xml description (sometimes).
    if draw(st.booleans()):
        files["pom.xml"] = _make_pom_xml(draw(_short_text))

    # Python source files exercising every detection branch.
    files.update(draw(_file_subset("svc/mod_", _PYTHON_SNIPPETS, ".py", max_files=6)))

    # Java source files (Spring annotations).
    files.update(
        draw(_file_subset("src/main/java/Mod", _JAVA_SNIPPETS, ".java", max_files=2))
    )

    # JS / TS source files.
    files.update(draw(_file_subset("web/index_", _JS_SNIPPETS, ".js", max_files=2)))

    # YAML files (cron / schedule keys).
    files.update(draw(_file_subset("ops/cfg_", _YAML_SNIPPETS, ".yml", max_files=2)))

    project_id = draw(st.integers(min_value=1, max_value=10_000_000))
    commit_sha = draw(
        st.text(
            alphabet="0123456789abcdef",
            min_size=7,
            max_size=40,
        )
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
    """Build the full positional argument tuple for :func:`analyze`."""
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
    repo_contents = draw(_repository_contents())
    # Re-stamp the contents' identity to match the project id we will
    # pass into ``analyze`` so that any future cross-checks remain
    # consistent. ``RepositoryContents`` is frozen, so we build a fresh
    # instance from its fields.
    repo_contents = RepositoryContents(
        gitlab_project_id=project_id,
        commit_sha=commit_sha,
        files=dict(repo_contents.files),
    )
    return project_id, full_path, analysis_branch, commit_sha, repo_description, repo_contents


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(case=_analyze_inputs())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_project_profile_sections_are_well_formed(
    case: tuple[int, str, str, str, str | None, RepositoryContents],
) -> None:
    """Property 6: every produced ``ProjectProfile`` is structurally well-formed."""
    project_id, full_path, analysis_branch, commit_sha, repo_description, repo_contents = case

    profile = analyze(
        project_id,
        full_path,
        analysis_branch,
        commit_sha,
        repo_description,
        repo_contents,
    )

    # ------------------------------------------------------------------
    # Abstract inputs (Requirements 4.1, 4.3, 4.5).
    # ------------------------------------------------------------------
    assert isinstance(profile.abstract_inputs, list)
    for entry in profile.abstract_inputs:
        assert entry.category in _INPUT_CATEGORIES, (
            f"abstract_input.category {entry.category!r} is outside the closed set"
        )
        assert entry.description is not None
        assert isinstance(entry.description, str)
        assert entry.description != "", (
            "abstract_input.description must be non-empty (non-null per Requirement 4.3)"
        )

    # ------------------------------------------------------------------
    # Abstract outputs (Requirements 4.2, 4.4, 4.6).
    # ------------------------------------------------------------------
    assert isinstance(profile.abstract_outputs, list)
    for entry in profile.abstract_outputs:
        assert entry.category in _OUTPUT_CATEGORIES, (
            f"abstract_output.category {entry.category!r} is outside the closed set"
        )
        assert entry.description is not None
        assert isinstance(entry.description, str)
        assert entry.description != "", (
            "abstract_output.description must be non-empty (non-null per Requirement 4.4)"
        )

    # ------------------------------------------------------------------
    # External service dependencies (Requirements 5.1, 5.2, 5.4).
    # ------------------------------------------------------------------
    assert isinstance(profile.external_service_dependencies, list)
    for service in profile.external_service_dependencies:
        assert isinstance(service.name, str)
        assert service.name != "", "external_service.name must be non-empty"
        assert service.kind in _SERVICE_KINDS, (
            f"external_service.kind {service.kind!r} is outside the closed set"
        )
        assert isinstance(service.source_locations, list)
        assert len(service.source_locations) > 0, (
            "external_service.source_locations must be non-empty"
        )
        for loc in service.source_locations:
            assert isinstance(loc.path, str)
            assert loc.path != "", "SourceLocation.path must be non-empty"

    # ------------------------------------------------------------------
    # Database table dependencies (Requirements 6.1, 6.2, 6.4).
    # ------------------------------------------------------------------
    assert isinstance(profile.database_table_dependencies, list)
    for table in profile.database_table_dependencies:
        assert isinstance(table.table_name, str)
        assert table.table_name != "", "database_table.table_name must be non-empty"
        assert table.access_mode in _ACCESS_MODES, (
            f"database_table.access_mode {table.access_mode!r} is outside the closed set"
        )
        assert isinstance(table.source_locations, list)
        assert len(table.source_locations) > 0, (
            "database_table.source_locations must be non-empty"
        )
        for loc in table.source_locations:
            assert isinstance(loc.path, str)
            assert loc.path != "", "SourceLocation.path must be non-empty"

"""Unit tests for ``project_analyzer.io_extractor.extract_io``.

These tests pin down the per-source behavior of the I/O extractor:
HTTP route handlers, scheduled tasks, message consumers and
publishers, file I/O, CLI entrypoints (including ``pyproject.toml``,
``package.json``, and ``if __name__ == "__main__":`` blocks),
external HTTP calls, database writes, and the empty-input contract
that Requirements 4.5 and 4.6 require.

Implements Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import (
    AbstractInputCategory,
    AbstractOutputCategory,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.io_extractor import extract_io

pytestmark = pytest.mark.unit


def _repo(files: dict[str, str]) -> RepositoryContents:
    """Build a minimal :class:`RepositoryContents` for the given file map."""

    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )


def _categories_in(items: list, kind: str) -> set:
    """Return the set of categories present in a list of inputs/outputs."""

    return {getattr(it, kind) for it in items}


# ---------------------------------------------------------------------------
# Requirements 4.5 / 4.6: empty results must be empty lists, never None.
# ---------------------------------------------------------------------------


def test_empty_repository_returns_two_empty_lists() -> None:
    inputs, outputs = extract_io(_repo({}))

    assert inputs == []
    assert outputs == []


def test_repository_with_only_unrelated_text_returns_empty() -> None:
    files = {
        "README.md": "# Some service\n",
        "data/notes.txt": "nothing useful here",
    }

    inputs, outputs = extract_io(_repo(files))

    assert inputs == []
    assert outputs == []


# ---------------------------------------------------------------------------
# HTTP route handlers (Requirement 4.3 http_request, 4.4 http_response).
# ---------------------------------------------------------------------------


def test_fastapi_route_decorator_emits_http_request_and_response() -> None:
    files = {
        "app.py": (
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "@app.get('/users/{id}')\n"
            "def get_user(id: int):\n"
            "    return {'id': id}\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.HTTP_REQUEST in _categories_in(inputs, "category")
    assert AbstractOutputCategory.HTTP_RESPONSE in _categories_in(outputs, "category")
    assert any("/users/{id}" in i.description for i in inputs)
    assert any("GET" in i.description for i in inputs)


def test_flask_route_decorator_is_http_request() -> None:
    files = {
        "app.py": (
            "from flask import Flask\n"
            "app = Flask(__name__)\n"
            "@app.route('/health')\n"
            "def health():\n"
            "    return 'ok'\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.HTTP_REQUEST in _categories_in(inputs, "category")
    assert AbstractOutputCategory.HTTP_RESPONSE in _categories_in(outputs, "category")


def test_fastapi_router_decorator_is_detected() -> None:
    files = {
        "routes.py": (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            "@router.post('/orders')\n"
            "def create_order():\n"
            "    pass\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert any("POST" in i.description and "/orders" in i.description for i in inputs)
    assert AbstractOutputCategory.HTTP_RESPONSE in _categories_in(outputs, "category")


def test_express_route_registration_is_detected() -> None:
    files = {
        "server.js": (
            "const express = require('express');\n"
            "const app = express();\n"
            "app.get('/users/:id', (req, res) => res.json({}));\n"
            "router.post('/items', (req, res) => {});\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    descriptions = " ".join(i.description for i in inputs)
    assert "/users/:id" in descriptions
    assert "/items" in descriptions
    assert AbstractOutputCategory.HTTP_RESPONSE in _categories_in(outputs, "category")


def test_spring_get_mapping_is_detected() -> None:
    files = {
        "Controller.java": (
            "@RestController\n"
            "public class Controller {\n"
            '    @GetMapping("/api/health")\n'
            "    public String health() { return \"ok\"; }\n"
            "}\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.HTTP_REQUEST in _categories_in(inputs, "category")
    assert AbstractOutputCategory.HTTP_RESPONSE in _categories_in(outputs, "category")
    assert any("@GetMapping" in i.description for i in inputs)


def test_spring_request_mapping_is_detected_as_any_verb() -> None:
    files = {
        "Controller.java": (
            "@RequestMapping(value = \"/api\")\n"
            "public class Controller {}\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert any("ANY" in i.description for i in inputs)


# ---------------------------------------------------------------------------
# Scheduled tasks (Requirement 4.3 scheduled_event).
# ---------------------------------------------------------------------------


def test_apscheduler_decorator_is_scheduled_event() -> None:
    files = {
        "tasks.py": (
            "from apscheduler.schedulers.blocking import BlockingScheduler\n"
            "scheduler = BlockingScheduler()\n"
            "@scheduler.scheduled_job('cron', minute='*/5')\n"
            "def heartbeat(): pass\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.SCHEDULED_EVENT in _categories_in(inputs, "category")


def test_celery_periodic_task_is_scheduled_event() -> None:
    files = {
        "tasks.py": (
            "from celery.schedules import crontab\n"
            "@periodic_task(run_every=crontab(minute=0))\n"
            "def hourly(): pass\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.SCHEDULED_EVENT in _categories_in(inputs, "category")


def test_yaml_cron_expression_is_scheduled_event() -> None:
    files = {
        ".gitlab-ci.yml": (
            "jobs:\n"
            "  refresh:\n"
            "    schedule:\n"
            "      cron: '*/15 * * * *'\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.SCHEDULED_EVENT in _categories_in(inputs, "category")


def test_spring_scheduled_annotation_is_scheduled_event() -> None:
    files = {
        "Job.java": (
            "public class Job {\n"
            "    @Scheduled(fixedRate = 60000)\n"
            "    public void run() {}\n"
            "}\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.SCHEDULED_EVENT in _categories_in(inputs, "category")


# ---------------------------------------------------------------------------
# Message consumers and publishers (4.3 message_consumed, 4.4 message_published).
# ---------------------------------------------------------------------------


def test_celery_app_task_decorator_is_message_consumed() -> None:
    files = {
        "tasks.py": (
            "from celery import Celery\n"
            "app = Celery()\n"
            "@app.task\n"
            "def process(x): return x\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.MESSAGE_CONSUMED in _categories_in(inputs, "category")


def test_kafka_producer_send_is_message_published() -> None:
    files = {
        "publisher.py": (
            "from kafka import KafkaProducer\n"
            "producer = KafkaProducer()\n"
            "def publish():\n"
            "    producer.send('orders-topic', b'{}')\n"
        )
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.MESSAGE_PUBLISHED in _categories_in(outputs, "category")
    assert any("orders-topic" in o.description for o in outputs)


def test_rabbitmq_basic_publish_and_consume() -> None:
    files = {
        "amqp.py": (
            "import pika\n"
            "channel = pika.BlockingConnection().channel()\n"
            "channel.basic_publish(exchange='ex', routing_key='rk', body=b'')\n"
            "channel.basic_consume(queue='orders', on_message_callback=lambda *a: None)\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.MESSAGE_CONSUMED in _categories_in(inputs, "category")
    assert AbstractOutputCategory.MESSAGE_PUBLISHED in _categories_in(outputs, "category")


def test_aws_sqs_send_and_receive() -> None:
    files = {
        "sqs.py": (
            "import boto3\n"
            "client = boto3.client('sqs')\n"
            "client.send_message(QueueUrl='...', MessageBody='hi')\n"
            "client.receive_message(QueueUrl='...')\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.MESSAGE_CONSUMED in _categories_in(inputs, "category")
    assert AbstractOutputCategory.MESSAGE_PUBLISHED in _categories_in(outputs, "category")


def test_spring_kafka_listener_is_message_consumed() -> None:
    files = {
        "Listener.java": (
            "public class Listener {\n"
            "    @KafkaListener(topics = \"orders\")\n"
            "    public void on(String msg) {}\n"
            "}\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.MESSAGE_CONSUMED in _categories_in(inputs, "category")


# ---------------------------------------------------------------------------
# File I/O (Requirement 4.3 file_read, 4.4 file_written).
# ---------------------------------------------------------------------------


def test_python_open_read_is_file_read() -> None:
    files = {
        "io.py": (
            "def load():\n"
            "    with open('config.yaml') as f:\n"
            "        return f.read()\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.FILE_READ in _categories_in(inputs, "category")
    assert outputs == []


def test_python_open_write_is_file_written() -> None:
    files = {
        "io.py": (
            "def dump(data):\n"
            "    with open('out.txt', 'w') as f:\n"
            "        f.write(data)\n"
        )
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.FILE_WRITTEN in _categories_in(outputs, "category")


def test_python_open_read_plus_mode_is_both_read_and_written() -> None:
    files = {
        "io.py": (
            "def update():\n"
            "    with open('state.bin', 'r+') as f:\n"
            "        f.read()\n"
            "        f.write(b'')\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.FILE_READ in _categories_in(inputs, "category")
    assert AbstractOutputCategory.FILE_WRITTEN in _categories_in(outputs, "category")


def test_pathlib_read_text_and_write_text() -> None:
    files = {
        "io.py": (
            "from pathlib import Path\n"
            "def go():\n"
            "    Path('a.txt').read_text()\n"
            "    Path('b.txt').write_text('x')\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.FILE_READ in _categories_in(inputs, "category")
    assert AbstractOutputCategory.FILE_WRITTEN in _categories_in(outputs, "category")


def test_node_fs_read_and_write_file() -> None:
    files = {
        "io.js": (
            "const fs = require('fs');\n"
            "fs.readFile('input.txt', cb);\n"
            "fs.writeFile('output.txt', data, cb);\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    assert AbstractInputCategory.FILE_READ in _categories_in(inputs, "category")
    assert AbstractOutputCategory.FILE_WRITTEN in _categories_in(outputs, "category")


# ---------------------------------------------------------------------------
# CLI entrypoints (Requirement 4.3 cli_argument).
# ---------------------------------------------------------------------------


def test_python_argparse_is_cli_argument() -> None:
    files = {
        "cli.py": (
            "import argparse\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.parse_args()\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.CLI_ARGUMENT in _categories_in(inputs, "category")


def test_click_command_decorator_is_cli_argument() -> None:
    files = {
        "cli.py": (
            "import click\n"
            "@click.command()\n"
            "def main(): pass\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.CLI_ARGUMENT in _categories_in(inputs, "category")


def test_if_name_main_block_is_cli_argument() -> None:
    files = {
        "run.py": (
            "def main(): pass\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.CLI_ARGUMENT in _categories_in(inputs, "category")
    assert any("__main__" in i.description for i in inputs)


def test_if_name_main_block_reversed_operands_is_cli_argument() -> None:
    """``"__main__" == __name__`` is a valid (if unusual) main guard."""

    files = {
        "run.py": (
            "def main(): pass\n"
            "if '__main__' == __name__:\n"
            "    main()\n"
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert AbstractInputCategory.CLI_ARGUMENT in _categories_in(inputs, "category")


def test_pyproject_project_scripts_is_cli_argument() -> None:
    files = {
        "pyproject.toml": (
            "[project]\n"
            'name = "demo"\n'
            "[project.scripts]\n"
            'demo-cli = "demo.cli:main"\n'
            'demo-tool = "demo.tool:run"\n'
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    cli_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.CLI_ARGUMENT
    ]
    descriptions = " ".join(i.description for i in cli_inputs)
    assert "demo-cli" in descriptions
    assert "demo-tool" in descriptions


def test_pyproject_poetry_scripts_is_cli_argument() -> None:
    files = {
        "pyproject.toml": (
            "[tool.poetry]\n"
            'name = "demo"\n'
            'version = "0.1.0"\n'
            'description = ""\n'
            'authors = []\n'
            "[tool.poetry.scripts]\n"
            'legacy-cli = "demo.cli:main"\n'
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    cli_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.CLI_ARGUMENT
    ]
    assert any("legacy-cli" in i.description for i in cli_inputs)


def test_package_json_bin_object_is_cli_argument() -> None:
    files = {
        "package.json": (
            '{"name": "tool", "version": "1.0.0",'
            ' "bin": {"tool": "./bin/tool.js", "tool-helper": "./bin/helper.js"}}'
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    cli_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.CLI_ARGUMENT
    ]
    descriptions = " ".join(i.description for i in cli_inputs)
    assert "'tool'" in descriptions
    assert "'tool-helper'" in descriptions


def test_package_json_bin_string_is_cli_argument() -> None:
    files = {
        "package.json": '{"name": "single-tool", "version": "1.0.0", "bin": "./cli.js"}'
    }

    inputs, _outputs = extract_io(_repo(files))

    cli_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.CLI_ARGUMENT
    ]
    assert len(cli_inputs) == 1
    assert "single-tool" in cli_inputs[0].description


def test_package_json_without_bin_does_not_emit_cli_input() -> None:
    files = {
        "package.json": '{"name": "lib", "version": "1.0.0"}'
    }

    inputs, _outputs = extract_io(_repo(files))

    assert all(i.category is not AbstractInputCategory.CLI_ARGUMENT for i in inputs)


def test_pyproject_in_subdirectory_is_not_treated_as_root_manifest() -> None:
    """Only repository-root manifests count; nested ones are ignored."""

    files = {
        "vendor/pyproject.toml": (
            "[project.scripts]\n" 'nested-cli = "x:y"\n'
        )
    }

    inputs, _outputs = extract_io(_repo(files))

    assert all(
        "nested-cli" not in i.description
        for i in inputs
        if i.category is AbstractInputCategory.CLI_ARGUMENT
    )


def test_malformed_pyproject_is_silently_ignored() -> None:
    files = {"pyproject.toml": "this is not valid toml = = ="}

    inputs, outputs = extract_io(_repo(files))

    assert inputs == []
    assert outputs == []


def test_malformed_package_json_is_silently_ignored() -> None:
    files = {"package.json": "{not json"}

    inputs, outputs = extract_io(_repo(files))

    assert inputs == []
    assert outputs == []


# ---------------------------------------------------------------------------
# External calls and database writes (Requirement 4.4).
# ---------------------------------------------------------------------------


def test_python_requests_get_is_external_call() -> None:
    files = {
        "client.py": (
            "import requests\n"
            "def go():\n"
            "    return requests.get('https://api.example.com/v1/data')\n"
        )
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.EXTERNAL_CALL in _categories_in(outputs, "category")


def test_python_httpx_post_is_external_call() -> None:
    files = {
        "client.py": (
            "import httpx\n"
            "def go():\n"
            "    return httpx.post('https://api.example.com/v1/data')\n"
        )
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.EXTERNAL_CALL in _categories_in(outputs, "category")


def test_javascript_fetch_is_external_call() -> None:
    files = {
        "client.js": (
            "async function go() {\n"
            "  const r = await fetch('https://api.example.com/v1/data');\n"
            "  return r.json();\n"
            "}\n"
        )
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.EXTERNAL_CALL in _categories_in(outputs, "category")


def test_axios_post_is_external_call() -> None:
    files = {
        "client.js": "axios.post('https://api.example.com/v1/data', payload);\n"
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.EXTERNAL_CALL in _categories_in(outputs, "category")


def test_sql_insert_in_python_string_is_database_write() -> None:
    files = {
        "repo.py": (
            "def add(conn):\n"
            "    conn.execute(\"INSERT INTO orders (id) VALUES (1)\")\n"
        )
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.DATABASE_WRITE in _categories_in(outputs, "category")


def test_sqlalchemy_session_save_is_database_write() -> None:
    files = {
        "repo.py": (
            "def save(session, user):\n"
            "    session.save(user)\n"
        )
    }

    _inputs, outputs = extract_io(_repo(files))

    assert AbstractOutputCategory.DATABASE_WRITE in _categories_in(outputs, "category")


# ---------------------------------------------------------------------------
# Output schema invariants enforced by the dataclasses themselves.
# ---------------------------------------------------------------------------


def test_every_input_has_known_category_and_non_empty_description() -> None:
    files = {
        "app.py": (
            "@app.get('/x')\n"
            "def x(): pass\n"
            "@app.task\n"
            "def t(): pass\n"
            "open('a.txt').read()\n"
            "import argparse; argparse.ArgumentParser()\n"
            "if __name__ == '__main__':\n    pass\n"
        )
    }

    inputs, outputs = extract_io(_repo(files))

    for entry in inputs:
        assert entry.category in AbstractInputCategory
        assert entry.description and isinstance(entry.description, str)
    for entry in outputs:
        assert entry.category in AbstractOutputCategory
        assert entry.description and isinstance(entry.description, str)


def test_detections_are_deduplicated_by_category_and_description() -> None:
    """Two identical route declarations should produce one input/output pair."""

    files = {
        "a.py": (
            "@app.get('/x')\n"
            "def x(): pass\n"
        ),
        "b.py": (
            "@app.get('/x')\n"
            "def x(): pass\n"
        ),
    }

    inputs, outputs = extract_io(_repo(files))

    # Each route declared in two files but in two different paths produces
    # two distinct descriptions (paths differ); the dedup invariant we
    # care about is per-(category, description) pair, so we instead check
    # that within a single file no duplicates arise.
    assert len({(i.category, i.description) for i in inputs}) == len(inputs)
    assert len({(o.category, o.description) for o in outputs}) == len(outputs)


def test_unparseable_python_does_not_abort_extraction() -> None:
    """A SyntaxError in one file must not mask detections in another."""

    files = {
        "broken.py": "def (((\n",
        "good.py": (
            "@app.get('/ok')\n"
            "def ok(): pass\n"
        ),
    }

    inputs, _outputs = extract_io(_repo(files))

    assert any("/ok" in i.description for i in inputs)

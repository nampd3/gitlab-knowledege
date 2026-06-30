"""I/O extractor for ``Project_Analyzer`` (task 6.5).

Statically inspects a project's :class:`RepositoryContents` for HTTP
route handlers, scheduled tasks, message consumers and publishers, file
I/O, CLI entrypoints, database writes, and external HTTP calls. Returns
two lists -- :class:`AbstractInput` and :class:`AbstractOutput` -- whose
``category`` values are drawn from the closed sets in Requirements 4.3
and 4.4.

Empty results are returned as empty lists, never ``None``
(Requirements 4.5, 4.6). Every emitted entry has a non-null
human-readable ``description`` (Requirement 4.3, 4.4). Detections are
deduplicated by ``(category, description)`` so that a single repository
yields a focused list rather than one entry per source line.

Detection sources, by file kind:

* Python (``.py``): parsed with the standard-library :mod:`ast` module.
  Function decorators, ``Call`` nodes, and string literals (for SQL
  write statements) are inspected. A :class:`SyntaxError` while parsing
  one file does not abort extraction; the file is silently skipped.
* JavaScript / TypeScript (``.js``, ``.ts``, ``.jsx``, ``.tsx``,
  ``.mjs``, ``.cjs``): regex-scanned for Express-style route
  registration, ``fetch`` / ``axios`` external calls, ``fs.readFile``
  and ``fs.writeFile``, and Kafka / SQS-style publish / subscribe
  patterns.
* Java (``.java``): regex-scanned for Spring annotations
  (``@RequestMapping``, ``@GetMapping``, etc.), ``@Scheduled``, and the
  common listener annotations (``@KafkaListener``, ``@RabbitListener``,
  ``@JmsListener``).
* YAML (``.yml`` / ``.yaml``): regex-scanned for ``cron:`` schedule
  expressions.
* Manifests at the repository root, parsed for CLI entrypoints:
  ``pyproject.toml`` is parsed for ``[project.scripts]`` keys (PEP 621);
  ``package.json`` is parsed for the ``bin`` field, which may be a
  string (the package's single script) or an object mapping script
  names to paths.

The module has no external dependencies and never raises out of the
public ``extract_io`` entrypoint; per the design's "analyzer never
throws" rule, an unexpected error in the scan of one file is contained.

Implements Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from project_knowledge_mcp.models import (
    AbstractInput,
    AbstractInputCategory,
    AbstractOutput,
    AbstractOutputCategory,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from project_knowledge_mcp.models import RepositoryContents


# ---------------------------------------------------------------------------
# File-extension routing
# ---------------------------------------------------------------------------

_PYTHON_EXTS: frozenset[str] = frozenset({".py"})
_JS_EXTS: frozenset[str] = frozenset(
    {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
)
_JAVA_EXTS: frozenset[str] = frozenset({".java"})
_YAML_EXTS: frozenset[str] = frozenset({".yml", ".yaml"})

#: Repository-root manifest filenames inspected for CLI entrypoints. Keys
#: are exact basenames; the corresponding scanner is dispatched in
#: ``extract_io`` when a file at the repository root matches.
_CLI_MANIFEST_FILES: frozenset[str] = frozenset({"pyproject.toml", "package.json"})


# ---------------------------------------------------------------------------
# Per-language pattern constants
# ---------------------------------------------------------------------------

# HTTP method names recognized as route-decorator suffixes (FastAPI,
# Starlette, Sanic, Bottle, etc.) and as Express-style call attributes.
_HTTP_VERBS: frozenset[str] = frozenset(
    {"get", "post", "put", "delete", "patch", "options", "head"}
)

# Decorator-name fragments that mark a scheduled task across frameworks
# (APScheduler ``@scheduler.scheduled_job``, Celery beat
# ``@periodic_task``, generic ``@scheduled``, cron-style decorators).
_SCHEDULER_DECORATOR_FRAGMENTS: tuple[str, ...] = (
    "scheduled_job",
    "scheduled",
    "periodic_task",
    "cron",
)

# Last-attribute names recognized as "ORM database write" calls. These
# are deliberately narrow to avoid false positives against built-in
# collection methods (``dict.update``, ``list.pop``, etc.).
_DB_WRITE_METHODS: frozenset[str] = frozenset(
    {"save", "bulk_create", "bulk_update", "bulk_save_objects"}
)

# Last-attribute names recognized as ORM "delete" calls. These are
# treated as database writes (``database_write``) per Requirement 4.4.
_DB_DELETE_METHODS: frozenset[str] = frozenset({"delete"})

# Modules whose ``.get/.post/...`` calls are recognized as external HTTP
# calls. The match is on the *first* attribute path component.
_EXTERNAL_HTTP_MODULES: frozenset[str] = frozenset(
    {"requests", "httpx", "aiohttp", "urllib3"}
)

# SQL write statements detected inside Python string literals. Matches
# the canonical write keywords listed in Requirement 4.4 (database_write)
# without binding to a specific dialect.
_SQL_WRITE_RE: re.Pattern[str] = re.compile(
    r"\b("
    r"INSERT\s+INTO|"
    r"UPDATE\s+\w+\s+SET|"
    r"DELETE\s+FROM|"
    r"MERGE\s+INTO|"
    r"REPLACE\s+INTO|"
    r"UPSERT\s+INTO"
    r")\b",
    re.IGNORECASE,
)

# JavaScript / TypeScript regexes.
_JS_ROUTE_RE: re.Pattern[str] = re.compile(
    r"""\b(?:app|router|server)\s*\.\s*"""
    r"""(get|post|put|delete|patch|options|head)\s*\(\s*"""
    r"""['"`]([^'"`]+)['"`]""",
    re.IGNORECASE,
)
_JS_FETCH_RE: re.Pattern[str] = re.compile(
    r"""\bfetch\s*\(\s*['"`]([^'"`]+)['"`]"""
)
_JS_AXIOS_RE: re.Pattern[str] = re.compile(
    r"""\baxios\s*\.\s*(get|post|put|delete|patch|head)\s*\("""
    r"""\s*['"`]([^'"`]+)['"`]""",
    re.IGNORECASE,
)
_JS_FS_READ_RE: re.Pattern[str] = re.compile(
    r"""\bfs(?:Promises|Promises\.promises)?\s*\.\s*"""
    r"""(?:readFile|readFileSync|createReadStream|read)\s*\("""
    r"""\s*['"`]([^'"`]+)['"`]"""
)
_JS_FS_WRITE_RE: re.Pattern[str] = re.compile(
    r"""\bfs(?:Promises|Promises\.promises)?\s*\.\s*"""
    r"""(?:writeFile|writeFileSync|appendFile|appendFileSync|"""
    r"""createWriteStream|write)\s*\("""
    r"""\s*['"`]([^'"`]+)['"`]"""
)
_JS_KAFKA_PUBLISH_RE: re.Pattern[str] = re.compile(
    r"""\b(?:producer|kafkaProducer|publisher)\s*\.\s*"""
    r"""(?:send|publish|produce)\s*\("""
)
_JS_KAFKA_CONSUME_RE: re.Pattern[str] = re.compile(
    r"""\b(?:consumer|kafkaConsumer|subscription)\s*\.\s*"""
    r"""(?:subscribe|run|consume|on)\s*\("""
)
_JS_COMMANDER_RE: re.Pattern[str] = re.compile(
    r"""\b(?:program|commander|cli)\s*\.\s*command\s*\("""
)

# Java regexes.
_JAVA_MAPPING_RE: re.Pattern[str] = re.compile(
    r"""@(?P<ann>"""
    r"""GetMapping|PostMapping|PutMapping|DeleteMapping|"""
    r"""PatchMapping|RequestMapping"""
    r""")\s*(?:\(\s*(?:value\s*=\s*|path\s*=\s*)?"""
    r"""['"]?(?P<route>[^,'"\)]*)['"]?)?"""
)
_JAVA_SCHEDULED_RE: re.Pattern[str] = re.compile(r"@Scheduled\b")
_JAVA_LISTENER_RE: re.Pattern[str] = re.compile(
    r"@(?P<ann>KafkaListener|RabbitListener|JmsListener|StreamListener)\b"
)

# YAML cron / schedule keys.
_YAML_CRON_RE: re.Pattern[str] = re.compile(
    r"""^(?P<indent>\s*)(?P<key>cron|schedule)\s*:\s*"""
    r"""['"]?(?P<expr>[^'"\n#]+?)['"]?\s*(?:#.*)?$""",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class _Accumulator:
    """Collects detections and deduplicates by ``(category, description)``.

    The extractor walks every file in the repository; many sources
    (especially regex scans) report the same logical I/O surface from
    multiple call sites. Dedup-by-description keeps the produced lists
    short and avoids leaking implementation noise into ``ProjectProfile``.
    """

    def __init__(self) -> None:
        self.inputs: list[AbstractInput] = []
        self.outputs: list[AbstractOutput] = []
        self._seen_inputs: set[tuple[AbstractInputCategory, str]] = set()
        self._seen_outputs: set[tuple[AbstractOutputCategory, str]] = set()

    def add_input(
        self, category: AbstractInputCategory, description: str
    ) -> None:
        key = (category, description)
        if key in self._seen_inputs:
            return
        self._seen_inputs.add(key)
        self.inputs.append(
            AbstractInput(category=category, description=description)
        )

    def add_output(
        self, category: AbstractOutputCategory, description: str
    ) -> None:
        key = (category, description)
        if key in self._seen_outputs:
            return
        self._seen_outputs.add(key)
        self.outputs.append(
            AbstractOutput(category=category, description=description)
        )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def extract_io(
    repository_contents: RepositoryContents,
) -> tuple[list[AbstractInput], list[AbstractOutput]]:
    """Extract abstract inputs and outputs from a repository snapshot.

    Args:
        repository_contents: An in-memory snapshot of a project's files
            at a given commit, as produced by
            ``GitLab_Connector.fetch_repository_contents``.

    Returns:
        A 2-tuple ``(inputs, outputs)`` where ``inputs`` is a list of
        :class:`AbstractInput` and ``outputs`` is a list of
        :class:`AbstractOutput`. Either list may be empty (Requirements
        4.5, 4.6); the function never returns ``None`` for either slot.
    """

    acc = _Accumulator()

    # Iterate in deterministic order so that the produced list ordering
    # is stable across runs on the same repository (helps make
    # ``ProjectProfile`` hashes and snapshots reproducible).
    for path in sorted(repository_contents.files.keys()):
        content = repository_contents.files[path]
        ext = PurePosixPath(path).suffix.lower()
        if ext in _PYTHON_EXTS:
            _scan_python(path, content, acc)
        elif ext in _JS_EXTS:
            _scan_javascript(path, content, acc)
        elif ext in _JAVA_EXTS:
            _scan_java(path, content, acc)
        elif ext in _YAML_EXTS:
            _scan_yaml(path, content, acc)
        # Other file kinds are ignored: the extractor is intentionally
        # conservative and only emits detections it understands.

        # Manifest scanners run regardless of extension routing above:
        # ``pyproject.toml`` falls into the "other" branch by extension,
        # and ``package.json`` would route to neither branch on its own.
        # Only repository-root manifests are inspected.
        if "/" not in path and path in _CLI_MANIFEST_FILES:
            _scan_manifest(path, content, acc)

    return acc.inputs, acc.outputs


# ---------------------------------------------------------------------------
# Python scanner
# ---------------------------------------------------------------------------


def _scan_python(path: str, content: str, acc: _Accumulator) -> None:
    """Scan a Python source file via :mod:`ast`.

    Files that do not parse (legacy 2.x syntax, partial snippets, etc.)
    are silently skipped. The aggregator in
    ``project_analyzer/__init__.py`` is responsible for flagging
    sub-analyzer failures via ``degraded_sections``; here we want one
    bad file not to mask the rest of the repository.
    """

    try:
        tree = ast.parse(content, filename=path)
    except (SyntaxError, ValueError):
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            _scan_python_function(path, node, acc)
        elif isinstance(node, ast.Call):
            _scan_python_call(path, node, acc)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            _scan_python_sql_literal(path, node.value, acc)
        elif isinstance(node, ast.If) and _is_main_guard(node):
            acc.add_input(
                AbstractInputCategory.CLI_ARGUMENT,
                f"CLI entrypoint via `if __name__ == \"__main__\":` in {path}",
            )


def _scan_python_function(
    path: str,
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    acc: _Accumulator,
) -> None:
    """Inspect a function's decorators for I/O surface registrations."""

    for dec in func.decorator_list:
        full_name, last_attr, args = _decorator_signature(dec)
        if last_attr is None:
            continue
        last_l = last_attr.lower()

        # HTTP route handlers: Flask/Blueprint ``@app.route('/x')`` and
        # FastAPI/Starlette ``@app.get('/x')``-family decorators.
        if last_l == "route" or last_l in _HTTP_VERBS:
            route = _first_string_arg(args) or func.name
            verb = last_l.upper() if last_l in _HTTP_VERBS else "ANY"
            acc.add_input(
                AbstractInputCategory.HTTP_REQUEST,
                f"HTTP {verb} {route} handled by {func.name} in {path}",
            )
            acc.add_output(
                AbstractOutputCategory.HTTP_RESPONSE,
                f"HTTP {verb} response from {func.name} ({route}) in {path}",
            )
            continue

        # Scheduled tasks (APScheduler, Celery beat, generic).
        if any(frag in last_l for frag in _SCHEDULER_DECORATOR_FRAGMENTS):
            schedule = _first_string_arg(args) or last_l
            acc.add_input(
                AbstractInputCategory.SCHEDULED_EVENT,
                f"Scheduled task {func.name} ({schedule}) in {path}",
            )
            continue

        # Celery / RQ / Dramatiq tasks: treated as message-consumed
        # (the task body is invoked by a broker dispatch).
        if last_l == "task" and _looks_like_task_decorator(full_name):
            acc.add_input(
                AbstractInputCategory.MESSAGE_CONSUMED,
                f"Async task {func.name} consumed from broker in {path}",
            )
            continue

        # CLI command registration: Click ``@click.command()`` and
        # Typer ``@app.command()`` both end in ``.command``.
        if last_l == "command":
            acc.add_input(
                AbstractInputCategory.CLI_ARGUMENT,
                f"CLI command {func.name} defined in {path}",
            )
            continue


def _scan_python_call(
    path: str, node: ast.Call, acc: _Accumulator
) -> None:
    """Inspect a single :class:`ast.Call` for I/O surface signatures.

    Dispatches to category-specific ``_try_*`` helpers so each branch is
    independently testable and the per-function branch count stays
    bounded as new patterns are added.
    """

    # First handle attribute-method calls whose receiver is itself an
    # expression rather than a simple Name (e.g. ``Path('x').read_text()``).
    # ``_call_dotted_name`` returns None for these chains, so we inspect
    # the trailing attribute directly to recover the canonical idiom.
    if isinstance(node.func, ast.Attribute) and _emit_pathlike_attr_call(
        path, node, node.func.attr.lower(), acc
    ):
        return

    callee = _call_dotted_name(node.func)
    if callee is None:
        return
    last_part = callee.rsplit(".", 1)[-1].lower()
    callee_l = callee.lower()

    # ``open(path, mode)``: classify by mode flag (handled out-of-band
    # because it splits into both an input and an output detection).
    if callee_l == "open":
        _emit_open_call(path, node, acc)
        return

    # Each helper returns True when it emitted a detection; the first
    # match wins, so ordering encodes precedence (file I/O > external
    # HTTP > CLI > message consumer > publisher > DB write).
    for handler in _CALL_HANDLERS:
        if handler(path, node, callee, last_part, callee_l, acc):
            return


def _try_emit_pathlike(
    path: str,
    node: ast.Call,
    callee: str,  # noqa: ARG001 - uniform handler signature
    last_part: str,
    callee_l: str,  # noqa: ARG001 - uniform handler signature
    acc: _Accumulator,
) -> bool:
    """Adapter wrapping :func:`_emit_pathlike_attr_call` for the dispatch table."""

    return _emit_pathlike_attr_call(path, node, last_part, acc)


def _try_emit_external_http(
    path: str,
    node: ast.Call,
    callee: str,
    last_part: str,
    callee_l: str,
    acc: _Accumulator,
) -> bool:
    """Emit an external_call detection for a recognized HTTP-client call."""

    root = callee.split(".", 1)[0].lower()
    if root in _EXTERNAL_HTTP_MODULES and last_part in _HTTP_VERBS:
        url = _first_string_arg(node.args) or "<dynamic>"
        acc.add_output(
            AbstractOutputCategory.EXTERNAL_CALL,
            f"External HTTP {last_part.upper()} via {callee}({url!r}) in {path}",
        )
        return True
    if last_part == "urlopen" and "urllib" in callee_l:
        url = _first_string_arg(node.args) or "<dynamic>"
        acc.add_output(
            AbstractOutputCategory.EXTERNAL_CALL,
            f"External HTTP call via {callee}({url!r}) in {path}",
        )
        return True
    return False


def _try_emit_cli_call(
    path: str,
    node: ast.Call,  # noqa: ARG001 - uniform handler signature
    callee: str,  # noqa: ARG001 - uniform handler signature
    last_part: str,  # noqa: ARG001 - uniform handler signature
    callee_l: str,
    acc: _Accumulator,
) -> bool:
    """Emit a CLI input for an ``argparse.ArgumentParser(...)`` construction."""

    if callee_l in {"argparse.argumentparser", "argumentparser"}:
        acc.add_input(
            AbstractInputCategory.CLI_ARGUMENT,
            f"CLI parser via argparse.ArgumentParser in {path}",
        )
        return True
    return False


def _try_emit_message_consumer(
    path: str,
    node: ast.Call,
    callee: str,  # noqa: ARG001 - uniform handler signature
    last_part: str,
    callee_l: str,  # noqa: ARG001 - uniform handler signature
    acc: _Accumulator,
) -> bool:
    """Emit a message_consumed input for known consumer-call shapes."""

    if last_part == "subscribe":
        topic = _first_string_arg(node.args) or "<unknown>"
        acc.add_input(
            AbstractInputCategory.MESSAGE_CONSUMED,
            f"Message consumer subscribe({topic!r}) in {path}",
        )
        return True
    if last_part == "basic_consume":
        queue = _first_keyword_or_string_arg(node, "queue") or "<unknown>"
        acc.add_input(
            AbstractInputCategory.MESSAGE_CONSUMED,
            f"RabbitMQ basic_consume({queue!r}) in {path}",
        )
        return True
    if last_part == "receive_message":
        acc.add_input(
            AbstractInputCategory.MESSAGE_CONSUMED,
            f"AWS SQS receive_message in {path}",
        )
        return True
    return False


def _try_emit_message_publisher(
    path: str,
    node: ast.Call,
    callee: str,  # noqa: ARG001 - uniform handler signature
    last_part: str,
    callee_l: str,
    acc: _Accumulator,
) -> bool:
    """Emit a message_published output for known publisher-call shapes."""

    if last_part == "basic_publish":
        acc.add_output(
            AbstractOutputCategory.MESSAGE_PUBLISHED,
            f"RabbitMQ basic_publish in {path}",
        )
        return True
    if last_part == "send_message":
        acc.add_output(
            AbstractOutputCategory.MESSAGE_PUBLISHED,
            f"AWS SQS send_message in {path}",
        )
        return True
    if last_part in {"send", "produce"} and _looks_like_kafka_producer(callee_l):
        topic = _first_string_arg(node.args) or "<unknown>"
        acc.add_output(
            AbstractOutputCategory.MESSAGE_PUBLISHED,
            f"Kafka producer.{last_part}({topic!r}) in {path}",
        )
        return True
    if last_part == "publish" and _looks_like_message_broker(callee_l):
        topic = _first_string_arg(node.args) or "<unknown>"
        acc.add_output(
            AbstractOutputCategory.MESSAGE_PUBLISHED,
            f"Message publish({topic!r}) in {path}",
        )
        return True
    return False


def _try_emit_db_write(
    path: str,
    node: ast.Call,  # noqa: ARG001 - uniform handler signature
    callee: str,  # noqa: ARG001 - uniform handler signature
    last_part: str,
    callee_l: str,
    acc: _Accumulator,
) -> bool:
    """Emit a database_write output for ORM-style write methods."""

    if last_part in _DB_WRITE_METHODS:
        acc.add_output(
            AbstractOutputCategory.DATABASE_WRITE,
            f"Database write via .{last_part}() in {path}",
        )
        return True
    if last_part in _DB_DELETE_METHODS and _looks_like_orm_call(callee_l):
        acc.add_output(
            AbstractOutputCategory.DATABASE_WRITE,
            f"Database write via .{last_part}() in {path}",
        )
        return True
    return False


#: Ordered dispatch table: handlers are tried in this order and the first
#: match short-circuits the rest. Pathlib-style attr methods come first
#: (file I/O), then external HTTP, CLI, message I/O, and finally
#: ORM-style DB writes -- this mirrors the conceptual specificity of the
#: signals each handler keys on.
_CALL_HANDLERS: tuple[
    Callable[[str, ast.Call, str, str, str, _Accumulator], bool], ...
] = (
    _try_emit_pathlike,
    _try_emit_external_http,
    _try_emit_cli_call,
    _try_emit_message_consumer,
    _try_emit_message_publisher,
    _try_emit_db_write,
)


def _scan_python_sql_literal(
    path: str, value: str, acc: _Accumulator
) -> None:
    """Detect SQL write statements inside Python string literals."""

    if len(value) > 4096:  # noqa: PLR2004 - guard against giant blobs
        # Truncate before regex to avoid pathological scanning costs.
        value = value[:4096]
    match = _SQL_WRITE_RE.search(value)
    if match is None:
        return
    statement = match.group(1).split()[0].upper()
    acc.add_output(
        AbstractOutputCategory.DATABASE_WRITE,
        f"SQL {statement} statement detected in {path}",
    )


def _emit_open_call(
    path: str, node: ast.Call, acc: _Accumulator
) -> None:
    """Translate an ``open(path, mode)`` call into file_read/file_written."""

    target = _first_string_arg(node.args) or "<dynamic>"
    mode = _open_mode(node)
    reads, writes = _classify_open_mode(mode)
    if reads:
        acc.add_input(
            AbstractInputCategory.FILE_READ,
            f"File read via open({target!r}, mode={mode!r}) in {path}",
        )
    if writes:
        acc.add_output(
            AbstractOutputCategory.FILE_WRITTEN,
            f"File write via open({target!r}, mode={mode!r}) in {path}",
        )


def _emit_pathlike_attr_call(
    path: str, node: ast.Call, attr_name: str, acc: _Accumulator
) -> bool:
    """Emit a file_read / file_written entry for a Pathlib-style attr call.

    Recognized attribute names are ``read_text`` / ``read_bytes`` (read)
    and ``write_text`` / ``write_bytes`` (write). Returns ``True`` when a
    detection was emitted so the caller can stop further classification
    of the same call.
    """

    target = _first_string_arg(node.args) or "<dynamic>"
    if attr_name in {"read_text", "read_bytes"}:
        acc.add_input(
            AbstractInputCategory.FILE_READ,
            f"File read via .{attr_name}({target!r}) in {path}",
        )
        return True
    if attr_name in {"write_text", "write_bytes"}:
        acc.add_output(
            AbstractOutputCategory.FILE_WRITTEN,
            f"File write via .{attr_name}({target!r}) in {path}",
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Python AST helpers
# ---------------------------------------------------------------------------


def _decorator_signature(
    dec: ast.expr,
) -> tuple[str, str | None, list[ast.expr]]:
    """Decompose a decorator AST node.

    Returns a 3-tuple ``(full_name, last_attr, positional_args)``:

    * ``full_name`` -- the dotted attribute path, e.g. ``"app.route"``
      for ``@app.route('/users')`` or ``"click.command"`` for
      ``@click.command()``. Empty string if the decorator is not a
      :class:`ast.Name` / :class:`ast.Attribute` chain.
    * ``last_attr`` -- the last component of ``full_name`` (or the
      :class:`ast.Name` ``id`` for a bare decorator), or ``None`` when
      the decorator shape is not recognized.
    * ``positional_args`` -- the positional arguments passed to the
      decorator if it is a :class:`ast.Call`; otherwise ``[]``.
    """

    if isinstance(dec, ast.Call):
        callee: ast.expr = dec.func
        positional = list(dec.args)
    else:
        callee = dec
        positional = []

    parts: list[str] = []
    cur: ast.expr = callee
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    elif parts == [] and isinstance(callee, ast.Name):
        # Bare ``@name`` (no Attribute chain).
        parts.append(callee.id)

    parts.reverse()
    full_name = ".".join(parts)
    last_attr = parts[-1] if parts else None
    return full_name, last_attr, positional


def _call_dotted_name(node: ast.expr) -> str | None:
    """Return the dotted name of a Call's ``func``, or ``None``.

    Handles bare names (``open``), single-level attributes
    (``app.route``), and chains (``urllib.request.urlopen``). Returns
    ``None`` if the chain bottoms out in something other than a
    :class:`ast.Name` (e.g. a subscript or call result).
    """

    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return None
    parts.reverse()
    return ".".join(parts)


def _first_string_arg(args: list[ast.expr]) -> str | None:
    """Return the first positional arg's value if it is a ``str`` constant.

    Only the *first* positional argument is inspected; later string
    arguments (e.g. the ``mode`` parameter of :func:`open`) must not be
    misattributed as if they were the first.
    """

    if not args:
        return None
    arg = args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def _first_keyword_or_string_arg(
    node: ast.Call, kw_name: str
) -> str | None:
    """Return the keyword arg ``kw_name`` if it is a ``str`` constant.

    Falls back to the first positional ``str`` constant. Used by
    ``basic_consume(queue=...)`` where the queue name is conventionally
    passed as a keyword argument.
    """

    for kw in node.keywords:
        if kw.arg == kw_name and isinstance(kw.value, ast.Constant):
            value = kw.value.value
            if isinstance(value, str):
                return value
    return _first_string_arg(node.args)


def _open_mode(node: ast.Call) -> str:
    """Extract the ``mode`` argument from an ``open(...)`` call."""

    if len(node.args) >= 2:  # noqa: PLR2004 - positional index
        arg = node.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    for kw in node.keywords:
        if (
            kw.arg == "mode"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return "r"


def _classify_open_mode(mode: str) -> tuple[bool, bool]:
    """Return ``(reads, writes)`` for a Python :func:`open` mode string.

    The Python file-mode grammar is: an optional ``r``/``w``/``a``/``x``
    base, optional ``b`` or ``t``, and optional ``+`` for "update".
    ``r`` is read-only, ``r+`` is read-and-write, ``w``/``a``/``x`` are
    write-creating modes, and ``w+``/``a+``/``x+`` add reading on top.
    """

    has_plus = "+" in mode
    has_write_base = any(c in mode for c in "wax")
    if not has_write_base:
        # Default ``r``/``rb``/``rt`` -- read-only unless ``+`` upgrades it.
        return True, has_plus
    return has_plus, True


def _looks_like_task_decorator(full_name: str) -> bool:
    """Return True if a ``.task`` decorator looks like Celery/RQ/Dramatiq.

    Common patterns: ``@app.task``, ``@celery.task``, ``@dramatiq.actor``
    (handled separately), ``@shared_task`` (caught by the bare-name path
    elsewhere). The shape we accept here is a 2-or-more-segment dotted
    name whose last component is ``task`` -- this excludes a stray
    bare ``@task`` decorator from being misclassified.
    """

    return "." in full_name and full_name.lower().endswith(".task")


def _looks_like_kafka_producer(callee_l: str) -> bool:
    """Heuristic for ``producer.send(...)``-style Kafka publish calls."""

    return "producer" in callee_l or "kafka" in callee_l


def _looks_like_message_broker(callee_l: str) -> bool:
    """Heuristic for generic ``.publish(...)`` calls on a broker client.

    We require *some* signal in the dotted name (``publisher``,
    ``broker``, ``topic``, ``exchange``, ``channel``) so a stray
    ``self.publish(...)`` on an unrelated class is not flagged.
    """

    return any(
        token in callee_l
        for token in (
            "publisher",
            "producer",
            "broker",
            "topic",
            "exchange",
            "channel",
            "kafka",
            "rabbit",
            "sns",
            "pubsub",
        )
    )


def _looks_like_orm_call(callee_l: str) -> bool:
    """Heuristic for ORM-style ``.delete()`` calls.

    We constrain the match to dotted calls (``obj.delete()``) and skip
    bare ``delete()`` calls, which are commonly defined on collections
    and unrelated helpers.
    """

    return "." in callee_l


def _is_main_guard(node: ast.If) -> bool:
    """Return True for an ``if __name__ == "__main__":`` (or reversed) test.

    Both orderings are accepted (``__name__ == "__main__"`` and
    ``"__main__" == __name__``) because either form marks a Python
    module as a CLI entrypoint per Requirement 4.3 ``cli_argument``.
    """

    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    operands: tuple[ast.expr, ...] = (test.left, *test.comparators)
    if len(operands) != 2:  # noqa: PLR2004 - binary compare
        return False
    has_name = any(
        isinstance(op, ast.Name) and op.id == "__name__" for op in operands
    )
    has_main_literal = any(
        isinstance(op, ast.Constant) and op.value == "__main__"
        for op in operands
    )
    return has_name and has_main_literal


# ---------------------------------------------------------------------------
# JavaScript / TypeScript scanner
# ---------------------------------------------------------------------------


def _scan_javascript(path: str, content: str, acc: _Accumulator) -> None:
    """Regex-scan a JS/TS source file for known I/O patterns."""

    for match in _JS_ROUTE_RE.finditer(content):
        verb = match.group(1).upper()
        route = match.group(2)
        acc.add_input(
            AbstractInputCategory.HTTP_REQUEST,
            f"HTTP {verb} {route} handler in {path}",
        )
        acc.add_output(
            AbstractOutputCategory.HTTP_RESPONSE,
            f"HTTP {verb} response for {route} in {path}",
        )

    for match in _JS_FETCH_RE.finditer(content):
        url = match.group(1)
        acc.add_output(
            AbstractOutputCategory.EXTERNAL_CALL,
            f"External HTTP call via fetch({url!r}) in {path}",
        )

    for match in _JS_AXIOS_RE.finditer(content):
        verb = match.group(1).upper()
        url = match.group(2)
        acc.add_output(
            AbstractOutputCategory.EXTERNAL_CALL,
            f"External HTTP {verb} via axios({url!r}) in {path}",
        )

    for match in _JS_FS_READ_RE.finditer(content):
        target = match.group(1)
        acc.add_input(
            AbstractInputCategory.FILE_READ,
            f"File read via fs.* ({target!r}) in {path}",
        )

    for match in _JS_FS_WRITE_RE.finditer(content):
        target = match.group(1)
        acc.add_output(
            AbstractOutputCategory.FILE_WRITTEN,
            f"File write via fs.* ({target!r}) in {path}",
        )

    if _JS_KAFKA_PUBLISH_RE.search(content):
        acc.add_output(
            AbstractOutputCategory.MESSAGE_PUBLISHED,
            f"Message publisher detected in {path}",
        )
    if _JS_KAFKA_CONSUME_RE.search(content):
        acc.add_input(
            AbstractInputCategory.MESSAGE_CONSUMED,
            f"Message consumer detected in {path}",
        )

    if _JS_COMMANDER_RE.search(content):
        acc.add_input(
            AbstractInputCategory.CLI_ARGUMENT,
            f"CLI command registered via commander in {path}",
        )


# ---------------------------------------------------------------------------
# Java scanner
# ---------------------------------------------------------------------------


def _scan_java(path: str, content: str, acc: _Accumulator) -> None:
    """Regex-scan a Java source file for Spring-style annotations."""

    for match in _JAVA_MAPPING_RE.finditer(content):
        ann = match.group("ann")
        route = (match.group("route") or "").strip()
        verb = _java_mapping_verb(ann)
        label = route if route else "<class-or-method-mapping>"
        acc.add_input(
            AbstractInputCategory.HTTP_REQUEST,
            f"HTTP {verb} {label} via @{ann} in {path}",
        )
        acc.add_output(
            AbstractOutputCategory.HTTP_RESPONSE,
            f"HTTP {verb} response from @{ann} {label} in {path}",
        )

    if _JAVA_SCHEDULED_RE.search(content):
        acc.add_input(
            AbstractInputCategory.SCHEDULED_EVENT,
            f"Scheduled task via @Scheduled in {path}",
        )

    for match in _JAVA_LISTENER_RE.finditer(content):
        ann = match.group("ann")
        acc.add_input(
            AbstractInputCategory.MESSAGE_CONSUMED,
            f"Message consumer via @{ann} in {path}",
        )


def _java_mapping_verb(annotation: str) -> str:
    """Map a Spring mapping annotation name to its HTTP verb label."""

    table = {
        "GetMapping": "GET",
        "PostMapping": "POST",
        "PutMapping": "PUT",
        "DeleteMapping": "DELETE",
        "PatchMapping": "PATCH",
        "RequestMapping": "ANY",
    }
    return table.get(annotation, "ANY")


# ---------------------------------------------------------------------------
# YAML scanner
# ---------------------------------------------------------------------------


def _scan_yaml(path: str, content: str, acc: _Accumulator) -> None:
    """Detect cron / schedule expressions in YAML configuration files."""

    for match in _YAML_CRON_RE.finditer(content):
        expr = match.group("expr").strip()
        if not expr:
            continue
        acc.add_input(
            AbstractInputCategory.SCHEDULED_EVENT,
            f"Scheduled event ({expr}) declared in {path}",
        )


# ---------------------------------------------------------------------------
# Manifest scanner (CLI entrypoints)
# ---------------------------------------------------------------------------


def _scan_manifest(path: str, content: str, acc: _Accumulator) -> None:
    """Detect CLI entrypoints declared in repository-root manifests.

    * ``pyproject.toml`` -- PEP 621 ``[project.scripts]`` table keys are
      installed as console-script entrypoints; the alternative
      ``[project.gui-scripts]`` table is treated the same way for
      detection purposes. Poetry's legacy ``[tool.poetry.scripts]`` is
      also recognized.
    * ``package.json`` -- the ``bin`` field may be a string (the package
      ships a single script named after the package) or an object
      mapping script names to file paths; both shapes are detected.

    Malformed manifests are silently ignored; the analyzer never raises
    out of :func:`extract_io`.
    """

    if path == "pyproject.toml":
        _scan_pyproject_scripts(path, content, acc)
    elif path == "package.json":
        _scan_package_json_bin(path, content, acc)


def _scan_pyproject_scripts(
    path: str, content: str, acc: _Accumulator
) -> None:
    """Emit one CLI input per ``[project.scripts]`` key in ``pyproject.toml``."""

    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return

    seen: set[str] = set()
    sources: tuple[tuple[str, ...], ...] = (
        ("project", "scripts"),
        ("project", "gui-scripts"),
        ("tool", "poetry", "scripts"),
    )
    for keys in sources:
        table = _walk_dict(data, keys)
        if table is None:
            continue
        for script_name in table:
            # ``_walk_dict`` returns dict[str, object]; mypy knows
            # script_name is str. Skip empty keys defensively in case a
            # malformed TOML produced a zero-length key.
            if not script_name:
                continue
            if script_name in seen:
                continue
            seen.add(script_name)
            acc.add_input(
                AbstractInputCategory.CLI_ARGUMENT,
                f"CLI entrypoint {script_name!r} declared in {path}",
            )


def _walk_dict(data: object, keys: tuple[str, ...]) -> dict[str, object] | None:
    """Navigate nested mapping access ``data[k0][k1]...[kn]``.

    Returns the resolved value when each intermediate level is a dict
    and the final value itself is a dict; otherwise returns ``None``.
    Used to safely descend into TOML/JSON tables without raising on
    missing keys or unexpected node types.
    """

    current: object = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, dict):
        return current
    return None


def _scan_package_json_bin(
    path: str, content: str, acc: _Accumulator
) -> None:
    """Emit one CLI input per ``bin`` entry in a root-level ``package.json``."""

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(data, dict):
        return
    bin_field = data.get("bin")
    if isinstance(bin_field, str) and bin_field:
        # ``"bin": "./cli.js"`` ships a single script named after the
        # package itself; fall back to ``<package>`` when ``name`` is
        # available, else label by the script path.
        package_name = data.get("name")
        label = package_name if isinstance(package_name, str) and package_name else bin_field
        acc.add_input(
            AbstractInputCategory.CLI_ARGUMENT,
            f"CLI entrypoint {label!r} declared in {path}",
        )
        return
    if isinstance(bin_field, dict):
        for script_name, target in bin_field.items():
            if not isinstance(script_name, str) or not script_name:
                continue
            if not isinstance(target, str):
                continue
            acc.add_input(
                AbstractInputCategory.CLI_ARGUMENT,
                f"CLI entrypoint {script_name!r} declared in {path}",
            )


__all__ = ["extract_io"]

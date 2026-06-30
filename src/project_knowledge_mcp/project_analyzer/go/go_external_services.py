"""Go external service dependency detector — path-scoped grep heuristics.

The previous parser-based implementation walked the Go event stream
and recognized ``fec_pool_service`` and ``activemq`` call shapes via
:class:`MethodCallEvent`/:class:`StructLitEvent` matching. The
operator confirmed that their ESB microservices follow a consistent
layout, so the detector can produce stronger signal from simple grep
heuristics scoped to a small set of well-known paths.

Three grep sources (in path-sorted iteration order):

1. **``config/config.go``** at the repo root — URL/endpoint/broker
   patterns. URL literals matching the canonical scheme set
   (``http``, ``https``, ``tcp``, ``stomp``, ``amqp``, ``redis``,
   ``grpc``) and identifier-equals-string-literal lines whose
   identifier name ends in ``_URL``/``_ENDPOINT``/``_BROKER``/
   ``_HOST``/``_URI`` (case-insensitive) both contribute.

2. **``internal/adapter/*_adapter.go``** — strict single-segment path
   under ``internal/adapter/``, filename ending in ``_adapter.go``.
   Service name is the filename root (e.g. ``activemq_adapter.go`` →
   ``activemq``); URLs found inside the file body are folded into the
   emitted entry's ``name`` because the parent
   :class:`ExternalServiceDependency` model has no auxiliary-text
   field.

3. **``internal/helper.go``** (and the equivalent
   ``internal/helper/helper.go``) — JMS detection. When the file
   mentions any of ``NewSubscriber``/``NewSender``/``NewClient`` AND
   contains the substring ``activemq`` or ``stomp``, the broker is
   emitted as a single ``ExternalServiceDependency`` named
   ``"activemq"`` with kind ``message_broker``. Destination
   constants discovered in ``config/config.go`` (identifier names
   containing ``Destination``/``Queue``/``Topic``/``Subscriber``
   followed by a string literal) are attached as additional
   ``SourceLocation`` entries pointing at the config file lines
   where each destination was declared.

APM exclusion guard: any file whose every ``import "..."`` line falls
under the ``go.elastic.co/apm/`` prefix is silently ignored. APM is
observability infrastructure, not a service dependency. Files with
mixed import surfaces are not skipped.

Coalescing: detections are folded by ``name``. The first detection
naming a service determines its kind; later detections of the same
name union their source locations (deduped by ``(path, line)``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.models import (
    ExternalServiceDependency,
    ExternalServiceKind,
    SourceLocation,
)
from project_knowledge_mcp.project_analyzer.go.go_filter import is_go_source_file

if TYPE_CHECKING:
    from collections.abc import Iterable

    from project_knowledge_mcp.models import RepositoryContents


__all__ = ["detect_go_external_services"]


# ---------------------------------------------------------------------------
# Path scoping
# ---------------------------------------------------------------------------

#: The repository-root config file the URL/identifier scan runs against.
_CONFIG_PATH: Final[str] = "config/config.go"

#: Directory prefix under which adapter files live in the operator's
#: layout. The single-segment rule below is enforced explicitly.
_ADAPTER_DIR_PREFIX: Final[str] = "internal/adapter/"

#: Filename suffix that identifies an adapter file. Combined with the
#: single-segment requirement under ``internal/adapter/`` this rejects
#: nested ``internal/adapter/<sub>/<name>_adapter.go`` and unrelated
#: files like ``adapter.go`` itself (no underscore prefix).
_ADAPTER_FILENAME_SUFFIX: Final[str] = "_adapter.go"

#: Helper-source paths the JMS detector inspects. The operator's spec
#: names ``internal/helper.go``; the in-scope repos all keep this
#: code at ``internal/helper/helper.go``, so both are accepted. Each
#: path is checked independently and the JMS pass runs on whichever
#: exists.
_HELPER_PATHS: Final[tuple[str, ...]] = (
    "internal/helper.go",
    "internal/helper/helper.go",
)


# ---------------------------------------------------------------------------
# APM exclusion guard
# ---------------------------------------------------------------------------

#: Import-path prefix identifying the Elastic APM Go agent
#: (``go.elastic.co/apm/v2``, ``go.elastic.co/apm/module/...``).
_APM_IMPORT_PREFIX: Final[str] = "go.elastic.co/apm/"

#: Regex matching a single Go ``import "..."`` statement on one line.
#: The grep contract is intentionally line-oriented: we do not parse
#: grouped ``import ( ... )`` blocks. The four sample repositories
#: all use grouped imports with one path per line, so a per-line
#: regex captures every import without needing a balanced-paren
#: parser.
_IMPORT_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r'^\s*(?:\w+\s+)?"(?P<path>[^"]+)"\s*$',
)


def _file_only_imports_apm(content: str) -> bool:
    """Return ``True`` when every ``import "..."`` line is APM.

    Scans the file line by line, matching the per-line import regex
    only when inside an ``import (`` block or on a standalone
    ``import "..."`` line. The check fires only when at least one
    import was observed; files with no imports at all are not
    skipped by the guard (the guard's purpose is to suppress
    observability-only files, not empty files).
    """

    has_any_import = False
    in_block = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not in_block:
            if line.startswith("import ("):
                in_block = True
                continue
            if line.startswith("import "):
                # Single-line import statement: ``import "path"`` or
                # ``import alias "path"``.
                m = _IMPORT_LINE_RE.match(line[len("import "):])
                if m is None:
                    continue
                has_any_import = True
                if not m.group("path").startswith(_APM_IMPORT_PREFIX):
                    return False
            continue
        # in_block: read paths until the closing ")".
        if line.startswith(")"):
            in_block = False
            continue
        if not line or line.startswith("//"):
            continue
        m = _IMPORT_LINE_RE.match(line)
        if m is None:
            continue
        has_any_import = True
        if not m.group("path").startswith(_APM_IMPORT_PREFIX):
            return False
    return has_any_import


# ---------------------------------------------------------------------------
# URL / identifier scanning shared by passes 1 and 2
# ---------------------------------------------------------------------------

#: URL literal matcher: one of the canonical schemes the operator's
#: repos use to express external service endpoints. The host portion
#: stops at whitespace or any of the common string-literal terminators
#: (``"``, ``'``, backtick). Path segments after the host are
#: discarded by :func:`_url_host` below — we only care about the host
#: portion for service-name derivation.
_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?P<scheme>https?|tcp|stomp|amqp|redis|grpc)://(?P<rest>[^\s\"'`)]+)",
    re.IGNORECASE,
)

#: Identifier-equals-string-literal matcher for ``config/config.go``.
#: The identifier name must end in one of ``_URL``, ``_ENDPOINT``,
#: ``_BROKER``, ``_HOST``, or ``_URI`` (case-insensitive). The
#: capture distinguishes the identifier and the literal so the
#: detector can fall back to the identifier root when the literal
#: doesn't contain a recognized URL.
_IDENT_URL_RE: Final[re.Pattern[str]] = re.compile(
    r'(?P<ident>[A-Za-z_][\w]*?(?:_URL|_ENDPOINT|_BROKER|_HOST|_URI))'
    r'\s*[:=]+\s*"(?P<value>[^"\n]*)"',
    re.IGNORECASE,
)

#: Identifier-equals-string-literal matcher for destination constants
#: inside ``config/config.go`` (JMS pass). The identifier name must
#: contain one of ``Destination``, ``Queue``, ``Topic``, or
#: ``Subscriber`` as a substring (case-insensitive).
_DESTINATION_RE: Final[re.Pattern[str]] = re.compile(
    r'(?P<ident>[A-Za-z_][\w]*?(?:Destination|Queue|Topic|Subscriber)[\w]*)'
    r'\s*[:=]+\s*"(?P<value>[^"\n]+)"',
    re.IGNORECASE,
)


def _line_for_offset(text: str, offset: int) -> int:
    """Return the 1-indexed line number that contains ``offset`` in ``text``."""

    return text.count("\n", 0, offset) + 1


def _url_host(rest: str) -> str:
    """Return the host portion of a URL match's tail.

    ``rest`` is the substring after ``<scheme>://``. The host stops at
    the first ``/`` (path), ``?`` (query), or ``#`` (fragment); the
    optional ``user[:pass]@`` prefix is stripped.
    """

    cutoff_chars = "/?#"
    end = len(rest)
    for ch in cutoff_chars:
        idx = rest.find(ch)
        if idx != -1 and idx < end:
            end = idx
    host = rest[:end]
    at = host.rfind("@")
    if at != -1:
        host = host[at + 1:]
    return host


def _kind_for_scheme(scheme: str) -> ExternalServiceKind:
    """Map a URL scheme to a canonical :class:`ExternalServiceKind`.

    Per the operator's heuristic table:

    * ``tcp``, ``stomp``, ``amqp`` → ``MESSAGE_BROKER``
    * ``redis``                    → ``CACHE``
    * ``http``, ``https``          → ``HTTP_API``
    * everything else (including ``grpc``) → ``OTHER``
    """

    s = scheme.lower()
    if s in ("tcp", "stomp", "amqp"):
        return ExternalServiceKind.MESSAGE_BROKER
    if s == "redis":
        return ExternalServiceKind.CACHE
    if s in ("http", "https"):
        return ExternalServiceKind.HTTP_API
    return ExternalServiceKind.OTHER


def _identifier_root(ident: str) -> str:
    """Strip the trailing ``_URL``/``_ENDPOINT``/etc. suffix from ``ident``.

    Used when an identifier-named URL constant has no URL literal on
    the same line. The remaining stem (lower-cased) becomes the
    canonical service name.
    """

    suffixes = ("_URL", "_ENDPOINT", "_BROKER", "_HOST", "_URI")
    upper = ident.upper()
    for suffix in suffixes:
        if upper.endswith(suffix):
            return ident[: -len(suffix)].lower()
    return ident.lower()


# ---------------------------------------------------------------------------
# Pass 1: ``config/config.go`` URL/identifier scan
# ---------------------------------------------------------------------------


def _scan_config(
    content: str,
    path: str,
) -> list[ExternalServiceDependency]:
    """Yield raw detections from URL/identifier patterns in ``content``.

    Iteration order: URL matches first (in source order), then
    identifier-only matches (in source order) for identifiers that
    were not already captured by a URL match on the same line. Two
    passes keep the precedence simple: a line like
    ``foo_url = "https://api.example.com"`` is captured once by the
    URL pass; a line like ``FEC_POOL_SERVICE_URL = otherCfg.X``
    falls through to the identifier-only pass.
    """

    detections: list[ExternalServiceDependency] = []
    captured_offsets: set[int] = set()

    for match in _URL_RE.finditer(content):
        host = _url_host(match.group("rest"))
        if not host:
            continue
        line = _line_for_offset(content, match.start())
        kind = _kind_for_scheme(match.group("scheme"))
        detections.append(
            ExternalServiceDependency(
                name=host,
                kind=kind,
                source_locations=[SourceLocation(path=path, line=line)],
            )
        )
        captured_offsets.add(match.start())

    for match in _IDENT_URL_RE.finditer(content):
        value = match.group("value")
        url_in_value = _URL_RE.search(value)
        if url_in_value is not None:
            # Already captured by the URL pass above (the URL regex
            # matched the same value). Skip.
            continue
        line = _line_for_offset(content, match.start())
        # No URL literal on this line — use the identifier's stem.
        name = _identifier_root(match.group("ident"))
        if not name:
            continue
        # Without a URL the scheme/kind is unknown; the identifier
        # suffix is a soft hint, so we default to ``OTHER``.
        ident_upper = match.group("ident").upper()
        if "BROKER" in ident_upper:
            kind = ExternalServiceKind.MESSAGE_BROKER
        elif "URL" in ident_upper or "ENDPOINT" in ident_upper or "URI" in ident_upper:
            kind = ExternalServiceKind.HTTP_API
        else:
            kind = ExternalServiceKind.OTHER
        detections.append(
            ExternalServiceDependency(
                name=name,
                kind=kind,
                source_locations=[SourceLocation(path=path, line=line)],
            )
        )

    return detections


# ---------------------------------------------------------------------------
# Pass 2: ``internal/adapter/*_adapter.go``
# ---------------------------------------------------------------------------


def _is_adapter_file(path: str) -> bool:
    """Return ``True`` for ``internal/adapter/<name>_adapter.go``.

    Strict single-segment rule: a file at
    ``internal/adapter/sub/foo_adapter.go`` does not qualify.
    """

    if not is_go_source_file(path):
        return False
    if not path.startswith(_ADAPTER_DIR_PREFIX):
        return False
    remainder = path[len(_ADAPTER_DIR_PREFIX):]
    if "/" in remainder:
        return False
    return remainder.endswith(_ADAPTER_FILENAME_SUFFIX)


def _service_name_from_adapter_path(path: str) -> str:
    """Strip the ``internal/adapter/`` prefix and ``_adapter.go`` suffix."""

    filename = path[len(_ADAPTER_DIR_PREFIX):]
    return filename[: -len(_ADAPTER_FILENAME_SUFFIX)]


def _scan_adapter(
    content: str,
    path: str,
) -> list[ExternalServiceDependency]:
    """Emit one detection for an adapter file.

    Service name is the filename root (e.g. ``activemq_adapter.go`` →
    ``activemq``). When the file body contains URL literals, the
    unique hosts are folded into the emitted ``name`` because the
    parent ``ExternalServiceDependency`` model has no auxiliary-text
    field. The kind is inferred from the first URL's scheme, or
    ``OTHER`` when no URL was found.

    A single source location is emitted: the line of the first URL
    when present, otherwise ``None``. Multiple URL hosts collapse
    into a single detection because the operator's adapters
    represent one service even when they reference several hosts
    (load-balanced brokers, dev/prod URLs, etc.).
    """

    base_name = _service_name_from_adapter_path(path)
    if not base_name:
        return []

    urls: list[tuple[str, ExternalServiceKind, int]] = []
    seen_hosts: set[str] = set()
    for match in _URL_RE.finditer(content):
        host = _url_host(match.group("rest"))
        if not host or host in seen_hosts:
            continue
        seen_hosts.add(host)
        line = _line_for_offset(content, match.start())
        kind = _kind_for_scheme(match.group("scheme"))
        urls.append((host, kind, line))

    if urls:
        hosts_text = ", ".join(host for host, _kind, _line in urls)
        name = f"{base_name} ({hosts_text})"
        kind = urls[0][1]
        first_line: int | None = urls[0][2]
    else:
        name = base_name
        kind = ExternalServiceKind.OTHER
        first_line = None

    return [
        ExternalServiceDependency(
            name=name,
            kind=kind,
            source_locations=[SourceLocation(path=path, line=first_line)],
        )
    ]


# ---------------------------------------------------------------------------
# Pass 3: JMS detection in ``internal/helper.go`` /
# ``internal/helper/helper.go``
# ---------------------------------------------------------------------------

#: Method-name fragments that signal JMS usage in the helper file.
#: The recognizer requires only that the substring appear in the
#: file text — no token-precise match is needed because these names
#: are extremely specific to the in-house ActiveMQ/STOMP wrapper.
_JMS_METHOD_TOKENS: Final[tuple[str, ...]] = ("NewSubscriber", "NewSender", "NewClient")

#: Library-name fragments that confirm a helper file's JMS context.
#: Either ``activemq`` or ``stomp`` anywhere in the file is enough.
_JMS_LIBRARY_TOKENS: Final[tuple[str, ...]] = ("activemq", "stomp")


def _helper_has_jms(content: str) -> bool:
    """Return ``True`` when ``content`` shows JMS-style broker usage.

    Requires both a JMS method token and a JMS library token to be
    present in the raw file text. The check is intentionally
    permissive — substring matching, not token-precise — because
    the operator's repos use the canonical helper names verbatim.
    """

    if not any(token in content for token in _JMS_METHOD_TOKENS):
        return False
    return any(token in content for token in _JMS_LIBRARY_TOKENS)


def _scan_jms_destinations(
    config_content: str,
    config_path: str,
) -> list[SourceLocation]:
    """Return one :class:`SourceLocation` per destination constant.

    Each match in ``config/config.go`` whose identifier name contains
    ``Destination``/``Queue``/``Topic``/``Subscriber`` (followed by a
    string literal) is recorded as a separate :class:`SourceLocation`
    pointing at the line that declared it. The model does not
    support auxiliary text on a source location, so the destination
    value itself is not encoded — the caller can read it from the
    referenced line in the config file when the additional context
    is needed.
    """

    sources: list[SourceLocation] = []
    seen: set[int] = set()
    for match in _DESTINATION_RE.finditer(config_content):
        line = _line_for_offset(config_content, match.start())
        if line in seen:
            continue
        seen.add(line)
        sources.append(SourceLocation(path=config_path, line=line))
    return sources


def _scan_jms(
    helper_content: str,
    helper_path: str,
    config_content: str | None,
) -> list[ExternalServiceDependency]:
    """Emit one ``activemq`` broker detection if the helper uses JMS.

    The returned detection's ``source_locations`` is built from:

    * one ``SourceLocation(helper_path, None)`` for the JMS usage
      site itself (grep does not track which line carries the
      method call), and
    * one ``SourceLocation(config/config.go, <line>)`` per
      destination constant discovered in the project's config file.

    When the helper does not show JMS usage the function returns
    ``[]`` and no detection is emitted.
    """

    if not _helper_has_jms(helper_content):
        return []

    sources: list[SourceLocation] = [SourceLocation(path=helper_path, line=None)]
    if config_content is not None:
        sources.extend(_scan_jms_destinations(config_content, _CONFIG_PATH))

    return [
        ExternalServiceDependency(
            name="activemq",
            kind=ExternalServiceKind.MESSAGE_BROKER,
            source_locations=sources,
        )
    ]


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------


def _coalesce(
    detections: Iterable[ExternalServiceDependency],
) -> list[ExternalServiceDependency]:
    """Group by ``name``; union source locations; preserve discovery order.

    The first detection naming a service determines both its position
    in the output list and its ``kind``. Later detections of the
    same name contribute additional source locations only, deduped
    by ``(path, line)``.

    The function is a pure transform on its input.
    """

    by_name: dict[
        str,
        tuple[ExternalServiceKind, list[SourceLocation], set[tuple[str, int | None]]],
    ] = {}

    for detection in detections:
        existing = by_name.get(detection.name)
        if existing is None:
            kind = detection.kind
            locations: list[SourceLocation] = []
            seen: set[tuple[str, int | None]] = set()
            by_name[detection.name] = (kind, locations, seen)
        else:
            _kind, locations, seen = existing
        for location in detection.source_locations:
            key = (location.path, location.line)
            if key in seen:
                continue
            seen.add(key)
            locations.append(location)

    return [
        ExternalServiceDependency(
            name=name,
            kind=kind,
            source_locations=list(locations),
        )
        for name, (kind, locations, _seen) in by_name.items()
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_go_external_services(
    repository_contents: RepositoryContents,
    events_by_file: object,
) -> tuple[list[ExternalServiceDependency], list[str]]:
    """Detect external service dependencies via path-scoped grep.

    Args:
        repository_contents: The :class:`RepositoryContents` snapshot
            for the project. Iterated for files matching the three
            grep sources documented in the module docstring.
        events_by_file: Unused; the aggregator passes the Go event
            map positionally. Typed as :class:`object` because this
            module no longer participates in the Go parser pipeline.

    Returns:
        ``(detections, file_skip_messages)``. The skip list is always
        empty under the grep contract.
    """

    _ = events_by_file

    config_content = repository_contents.read_text(_CONFIG_PATH)
    if config_content is not None and _file_only_imports_apm(config_content):
        config_content = None

    detections: list[ExternalServiceDependency] = []

    # Pass 1: config/config.go URL/identifier scan.
    if config_content is not None:
        detections.extend(_scan_config(config_content, _CONFIG_PATH))

    # Pass 2: internal/adapter/*_adapter.go. Path-sorted iteration
    # keeps the produced order deterministic.
    for path in sorted(repository_contents.files):
        if not _is_adapter_file(path):
            continue
        content = repository_contents.read_text(path)
        if content is None:
            continue
        if _file_only_imports_apm(content):
            continue
        detections.extend(_scan_adapter(content, path))

    # Pass 3: internal/helper.go (or internal/helper/helper.go) JMS
    # detection. The first existing helper path that shows JMS usage
    # contributes a single ``activemq`` detection; later paths
    # coalesce by name through :func:`_coalesce` if both exist.
    for helper_path in _HELPER_PATHS:
        helper_content = repository_contents.read_text(helper_path)
        if helper_content is None:
            continue
        if _file_only_imports_apm(helper_content):
            continue
        detections.extend(
            _scan_jms(helper_content, helper_path, config_content),
        )

    return _coalesce(detections), []

"""External service detector.

This module implements the *External service detector* sub-analyzer
described in the design's ``Project_Analyzer`` section. Given a
:class:`RepositoryContents` snapshot, :func:`detect_external_services`
returns a list of :class:`ExternalServiceDependency` records that
together describe the external services the project calls at runtime
(Requirement 5.1).

Detection strategy
------------------

The detector is a deliberately language-agnostic, string-based scanner.
It walks every analyzable text file in the repository and applies two
families of heuristics to each file:

1. **URL extraction.** A regex finds occurrences of ``<scheme>://<host>``
   in source. The scheme determines the kind for connection-string
   schemes (``kafka``, ``amqp``, ``redis``, ``s3`` ...). For ``http`` and
   ``https``, the host is matched against well-known SaaS patterns
   (Auth0, Cognito, S3, Storage, Redis-cloud, Kafka-cloud, ...) before
   falling back to ``http_api``.
2. **SDK constructor patterns.** A small set of regexes detects calls
   into well-known client libraries that themselves represent a fixed
   external service even when no URL is present in source. Examples:
   ``boto3.client("s3")`` (object store), ``KafkaProducer(...)``
   (message broker), ``redis.Redis(...)`` (cache).

Each detection produces a :class:`SourceLocation` carrying the file path
and the 1-indexed line number at which the match was found.

Aggregation (Requirement 5.3)
-----------------------------

After scanning, all detections are coalesced by ``name``: the returned
list contains *at most one* :class:`ExternalServiceDependency` per name,
and that entry's ``source_locations`` is the union of every site at
which the service was detected. When the same name is reported with
multiple kinds (typically because both a URL and an SDK constructor
were detected), the most specific kind wins -- ``message_broker``,
``object_store``, ``cache``, and ``auth_provider`` outrank ``http_api``,
which outranks the catch-all ``other``.

Empty result (Requirement 5.4)
------------------------------

When no detection succeeds, :func:`detect_external_services` returns an
empty list, never ``None``.

Implements Requirements 5.1, 5.2, 5.3, 5.4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from project_knowledge_mcp.models import (
    ExternalServiceDependency,
    ExternalServiceKind,
    SourceLocation,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from project_knowledge_mcp.models import RepositoryContents


# ---------------------------------------------------------------------------
# Path filtering
# ---------------------------------------------------------------------------

#: Path segments for vendored or generated artifacts that are not part of
#: the project's runtime call surface. Files whose normalized path
#: contains any of these as a directory component are skipped to avoid
#: false positives from third-party dependencies and build outputs.
_SKIP_PATH_SEGMENTS: Final[tuple[str, ...]] = (
    "node_modules/",
    "vendor/",
    ".git/",
    ".venv/",
    "venv/",
    "env/",
    "dist/",
    "build/",
    "target/",
    "__pycache__/",
    ".tox/",
    ".cache/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".idea/",
    ".vscode/",
    "site-packages/",
    "bower_components/",
)

#: Filenames whose contents are not analyzable for runtime call surface.
#: Lockfiles list the dependency closure of the project but do not
#: themselves invoke any service at runtime.
_SKIP_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "composer.lock",
        "go.sum",
    }
)

#: File-extension suffixes for documentation, generated bundles, and
#: other artifacts that do not represent runtime behaviour. Note that
#: configuration files (``.yaml``, ``.json``, ``.toml``, ``.env``,
#: ``.ini``) are deliberately *not* skipped because they often carry
#: hard-coded broker URLs and bucket names that real code reads at
#: startup.
_SKIP_EXTENSIONS: Final[tuple[str, ...]] = (
    ".md",
    ".rst",
    ".adoc",
    ".txt",
    ".min.js",
    ".bundle.js",
    ".map",
    ".lock",
    ".log",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
)


def _should_skip_file(path: str) -> bool:
    """Return True for paths that should not be scanned by the detector."""
    norm = path.replace("\\", "/")
    # Match ``segment/`` either at the very start or anywhere as a full
    # path component by anchoring on the leading ``/``.
    bracketed = "/" + norm
    for segment in _SKIP_PATH_SEGMENTS:
        if ("/" + segment) in bracketed:
            return True
    filename = norm.rsplit("/", 1)[-1]
    if filename in _SKIP_FILENAMES:
        return True
    lower = filename.lower()
    for ext in _SKIP_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return lower.startswith(("license", "changelog", "authors", "contributors"))


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

#: Regex matching a URL with an explicit scheme. Designed to be tolerant
#: of how URLs appear inside source: surrounded by quotes, embedded in
#: f-strings, or followed by punctuation. The host stops at any
#: character that is not a typical hostname character; the optional
#: path stops at whitespace or a small set of source-code punctuation.
_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    (?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://
    (?:[^@\s/?#:"'<>`]+@)?               # optional user[:pass]@
    (?P<host>
        \[[0-9a-fA-F:]+\]                # bracketed IPv6 literal
        |
        [a-zA-Z0-9._\-]+                 # hostname or IPv4 literal
    )
    (?::(?P<port>\d+))?
    (?P<path>/[^\s'"`<>{}\[\]\\,]*)?
    """,
    re.VERBOSE,
)

#: Schemes that are never external runtime services and are therefore
#: ignored entirely. ``ftp``/``smtp``/``ldap`` are noisy in
#: configuration and rarely represent the kinds of services the design
#: enumerates; if a project genuinely uses one as an external service it
#: will typically also register an SDK pattern that we *do* detect.
_SCHEME_DENYLIST: Final[frozenset[str]] = frozenset(
    {
        "git",
        "ssh",
        "file",
        "data",
        "blob",
        "javascript",
        "mailto",
        "tel",
        "urn",
        "ws",
        "wss",
        "ftp",
        "ftps",
        "smtp",
        "smtps",
        "ldap",
        "ldaps",
        "chrome",
        "chrome-extension",
        "about",
        "view-source",
    }
)

#: Schemes mapped directly to a service kind. URLs whose scheme is in
#: this map skip the host-pattern classifier because the scheme alone
#: identifies the kind unambiguously.
_SCHEME_KIND: Final[dict[str, ExternalServiceKind]] = {
    "kafka": ExternalServiceKind.MESSAGE_BROKER,
    "amqp": ExternalServiceKind.MESSAGE_BROKER,
    "amqps": ExternalServiceKind.MESSAGE_BROKER,
    "mqtt": ExternalServiceKind.MESSAGE_BROKER,
    "mqtts": ExternalServiceKind.MESSAGE_BROKER,
    "nats": ExternalServiceKind.MESSAGE_BROKER,
    "stomp": ExternalServiceKind.MESSAGE_BROKER,
    "redis": ExternalServiceKind.CACHE,
    "rediss": ExternalServiceKind.CACHE,
    "memcache": ExternalServiceKind.CACHE,
    "memcached": ExternalServiceKind.CACHE,
    "s3": ExternalServiceKind.OBJECT_STORE,
    "gs": ExternalServiceKind.OBJECT_STORE,
    "minio": ExternalServiceKind.OBJECT_STORE,
}

#: Hostnames considered local / loopback and ignored. A URL whose host
#: is on this list does not represent an *external* service dependency.
_LOCAL_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "[::1]",
        "[::]",
    }
)

#: Pattern matching ``s3.<region>.amazonaws.com`` and
#: ``s3-<region>.amazonaws.com`` host forms.
_S3_HOST_RE: Final[re.Pattern[str]] = re.compile(
    r"^s3[.\-][a-z0-9\-]+\.amazonaws\.com$"
)


def _is_local_host(host: str) -> bool:
    """Return True for loopback / non-routable host literals."""
    h = host.lower().strip("[]")
    if h in _LOCAL_HOSTS:
        return True
    return h.endswith(".localhost") or h.endswith(".local")


def _classify_http_host(host: str) -> ExternalServiceKind:
    """Classify an ``http(s)`` URL's host into an :class:`ExternalServiceKind`.

    The classifier matches well-known SaaS hostname patterns first; on
    no match it falls back to ``HTTP_API`` so any hard-coded HTTP base
    URL is at least recorded under the most generic kind.
    """
    h = host.lower()
    # Auth providers (Requirement 5.2 ``auth_provider``).
    if (
        h in ("accounts.google.com", "login.microsoftonline.com")
        or h.endswith((".auth0.com", ".okta.com", ".oktapreview.com"))
        or "cognito-idp" in h
        or "keycloak" in h
        or h.startswith(("oauth.", "auth.", "login."))
    ):
        return ExternalServiceKind.AUTH_PROVIDER
    # Object stores (Requirement 5.2 ``object_store``).
    if (
        h == "s3.amazonaws.com"
        or h.endswith(".s3.amazonaws.com")
        or _S3_HOST_RE.match(h) is not None
        or h == "storage.googleapis.com"
        or h.endswith(".storage.googleapis.com")
        or h.endswith(".blob.core.windows.net")
        or "minio" in h
    ):
        return ExternalServiceKind.OBJECT_STORE
    # Message brokers (Requirement 5.2 ``message_broker``).
    if (
        "kafka" in h
        or "rabbitmq" in h
        or h.endswith(".servicebus.windows.net")
        or h.startswith("sqs.")
        or h.endswith(".sqs.amazonaws.com")
        or h.startswith("sns.")
        or h.endswith(".sns.amazonaws.com")
    ):
        return ExternalServiceKind.MESSAGE_BROKER
    # Caches (Requirement 5.2 ``cache``).
    if "redis" in h or "memcached" in h:
        return ExternalServiceKind.CACHE
    # Default: a hard-coded HTTP base URL is an HTTP API dependency.
    return ExternalServiceKind.HTTP_API


# ---------------------------------------------------------------------------
# SDK pattern detection
# ---------------------------------------------------------------------------

#: Mapping of boto3 service identifier → ``(kind, canonical name)``.
#: ``boto3.client("<svc>")`` and ``boto3.resource("<svc>")`` patterns
#: are detected separately from URLs because the SDK call alone is
#: enough evidence of a runtime dependency on the corresponding AWS
#: service.
_BOTO3_SERVICE_KIND: Final[dict[str, tuple[ExternalServiceKind, str]]] = {
    "s3": (ExternalServiceKind.OBJECT_STORE, "aws-s3"),
    "sqs": (ExternalServiceKind.MESSAGE_BROKER, "aws-sqs"),
    "sns": (ExternalServiceKind.MESSAGE_BROKER, "aws-sns"),
    "kinesis": (ExternalServiceKind.MESSAGE_BROKER, "aws-kinesis"),
    "kafka": (ExternalServiceKind.MESSAGE_BROKER, "aws-msk"),
    "cognito-idp": (ExternalServiceKind.AUTH_PROVIDER, "aws-cognito"),
    "elasticache": (ExternalServiceKind.CACHE, "aws-elasticache"),
    "secretsmanager": (ExternalServiceKind.OTHER, "aws-secrets-manager"),
    "ses": (ExternalServiceKind.OTHER, "aws-ses"),
}

_BOTO3_RE: Final[re.Pattern[str]] = re.compile(
    r"""boto3\s*\.\s*(?:client|resource)\s*\(\s*['"](?P<svc>[a-z0-9\-]+)['"]"""
)

#: Patterns whose mere presence in source signals an external service of
#: a known kind. The third element of each tuple is the canonical
#: ``name`` written into the resulting :class:`ExternalServiceDependency`
#: when a more specific name (such as a host extracted from a URL)
#: is unavailable.
_NAMED_SDK_PATTERNS: Final[
    tuple[tuple[re.Pattern[str], ExternalServiceKind, str], ...]
] = (
    (
        re.compile(r"\bKafkaProducer\s*\("),
        ExternalServiceKind.MESSAGE_BROKER,
        "kafka",
    ),
    (
        re.compile(r"\bKafkaConsumer\s*\("),
        ExternalServiceKind.MESSAGE_BROKER,
        "kafka",
    ),
    (
        re.compile(r"\baiokafka\."),
        ExternalServiceKind.MESSAGE_BROKER,
        "kafka",
    ),
    (
        re.compile(r"\bpika\.(?:Blocking)?Connection\b"),
        ExternalServiceKind.MESSAGE_BROKER,
        "rabbitmq",
    ),
    (
        re.compile(r"\baio_pika\."),
        ExternalServiceKind.MESSAGE_BROKER,
        "rabbitmq",
    ),
    (
        re.compile(r"\bredis\.Redis\s*\("),
        ExternalServiceKind.CACHE,
        "redis",
    ),
    (
        re.compile(r"\bredis\.from_url\s*\("),
        ExternalServiceKind.CACHE,
        "redis",
    ),
    (
        re.compile(r"\baioredis\."),
        ExternalServiceKind.CACHE,
        "redis",
    ),
    (
        re.compile(r"\bmemcache\.Client\s*\("),
        ExternalServiceKind.CACHE,
        "memcached",
    ),
    (
        re.compile(r"\bpymemcache\."),
        ExternalServiceKind.CACHE,
        "memcached",
    ),
    (
        re.compile(r"\bminio\.Minio\s*\("),
        ExternalServiceKind.OBJECT_STORE,
        "minio",
    ),
    (
        re.compile(r"\bgoogle\.cloud\.storage\b"),
        ExternalServiceKind.OBJECT_STORE,
        "google-cloud-storage",
    ),
    (
        re.compile(r"\bAuth0Client\b"),
        ExternalServiceKind.AUTH_PROVIDER,
        "auth0",
    ),
    (
        re.compile(r"\bKeycloakOpenID\b"),
        ExternalServiceKind.AUTH_PROVIDER,
        "keycloak",
    ),
    (
        re.compile(r"\baxios\.create\s*\("),
        ExternalServiceKind.HTTP_API,
        "axios-http",
    ),
)


# ---------------------------------------------------------------------------
# Internal detection record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Detection:
    """A single detection: name, kind, source location.

    The intermediate type used while scanning. Aggregation in
    :func:`_aggregate` collapses many ``_Detection`` records into one
    :class:`ExternalServiceDependency` per ``name``.
    """

    name: str
    kind: ExternalServiceKind
    source: SourceLocation


def _line_for_position(content: str, position: int) -> int:
    """Return the 1-indexed line number for a character offset into ``content``."""
    return content.count("\n", 0, position) + 1


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------


def _scan_urls(path: str, content: str) -> Iterator[_Detection]:
    """Yield URL-based detections from ``content``.

    Drops loopback hosts and any URL whose scheme is on
    :data:`_SCHEME_DENYLIST`. URLs whose host contains a clear template
    placeholder (``${...}`` or ``{{...}}``) are also dropped because
    they do not name a concrete external service.
    """
    for match in _URL_RE.finditer(content):
        scheme = match.group("scheme").lower()
        if scheme in _SCHEME_DENYLIST:
            continue
        host = match.group("host")
        if not host:
            continue
        if "${" in host or "{{" in host:
            continue
        if _is_local_host(host):
            continue
        # Strip square brackets from IPv6 host literals for the name.
        name = host.strip("[]").lower()

        if scheme in _SCHEME_KIND:
            kind = _SCHEME_KIND[scheme]
        elif scheme in ("http", "https"):
            kind = _classify_http_host(host)
        else:
            kind = ExternalServiceKind.OTHER

        yield _Detection(
            name=name,
            kind=kind,
            source=SourceLocation(
                path=path,
                line=_line_for_position(content, match.start()),
            ),
        )


def _scan_sdk_patterns(path: str, content: str) -> Iterator[_Detection]:
    """Yield SDK-based detections from ``content``."""
    for match in _BOTO3_RE.finditer(content):
        svc = match.group("svc").lower()
        if svc in _BOTO3_SERVICE_KIND:
            kind, name = _BOTO3_SERVICE_KIND[svc]
        else:
            kind, name = ExternalServiceKind.OTHER, f"aws-{svc}"
        yield _Detection(
            name=name,
            kind=kind,
            source=SourceLocation(
                path=path,
                line=_line_for_position(content, match.start()),
            ),
        )

    for pattern, kind, name in _NAMED_SDK_PATTERNS:
        for match in pattern.finditer(content):
            yield _Detection(
                name=name,
                kind=kind,
                source=SourceLocation(
                    path=path,
                    line=_line_for_position(content, match.start()),
                ),
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

#: Priority used when the same ``name`` is reported with multiple
#: kinds. Lower values *win*; ``OTHER`` is the catch-all and ranks
#: highest. This keeps the well-known specific kinds (cache, broker,
#: object store, auth provider) from being clobbered by the more
#: generic ``http_api`` when both a URL and an SDK pattern fire on the
#: same host.
_KIND_PRIORITY: Final[dict[ExternalServiceKind, int]] = {
    ExternalServiceKind.MESSAGE_BROKER: 0,
    ExternalServiceKind.OBJECT_STORE: 0,
    ExternalServiceKind.CACHE: 0,
    ExternalServiceKind.AUTH_PROVIDER: 0,
    ExternalServiceKind.HTTP_API: 1,
    ExternalServiceKind.OTHER: 2,
}


def _kind_rank(kind: ExternalServiceKind) -> int:
    """Return the priority rank for ``kind`` (lower wins)."""
    return _KIND_PRIORITY.get(kind, 99)


def _aggregate(
    detections: Iterable[_Detection],
) -> list[ExternalServiceDependency]:
    """Coalesce detections into one entry per ``name`` (Requirement 5.3).

    For each unique ``name``:

    * The chosen ``kind`` is the most specific kind reported across all
      detections for that name (see :data:`_KIND_PRIORITY`).
    * ``source_locations`` is the union of every detection site for that
      name, deduplicated by ``(path, line)`` and sorted by ``(path,
      line)`` for deterministic output.

    The returned list itself is sorted by ``name`` for deterministic
    output.
    """
    by_name: dict[str, list[_Detection]] = {}
    for det in detections:
        by_name.setdefault(det.name, []).append(det)

    result: list[ExternalServiceDependency] = []
    for name in sorted(by_name):
        group = by_name[name]
        # Pick the most specific kind reported across all detections
        # for this name.
        best_kind = min((d.kind for d in group), key=_kind_rank)

        # Union of source locations, deduplicated by (path, line),
        # ordered by (path, line) for stability.
        seen: set[tuple[str, int | None]] = set()
        sources: list[SourceLocation] = []
        for det in group:
            key = (det.source.path, det.source.line)
            if key in seen:
                continue
            seen.add(key)
            sources.append(det.source)
        sources.sort(key=lambda s: (s.path, s.line if s.line is not None else 0))

        result.append(
            ExternalServiceDependency(
                name=name,
                kind=best_kind,
                source_locations=sources,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_external_services(
    repository_contents: RepositoryContents,
) -> list[ExternalServiceDependency]:
    """Detect external service dependencies in a repository.

    Walks every analyzable text file in ``repository_contents``, applies
    URL and SDK-pattern scanners to each, and aggregates the resulting
    detections into a list of :class:`ExternalServiceDependency` records
    with at most one entry per service ``name`` (Requirement 5.3).

    Args:
        repository_contents: An in-memory snapshot of the project's
            files at a specific commit, as produced by
            ``GitLab_Connector.fetch_repository_contents``.

    Returns:
        A list of :class:`ExternalServiceDependency` records. Empty
        (never ``None``) when no detection succeeds (Requirement 5.4).
        Each entry's ``source_locations`` is the union of every site at
        which the service was detected; the list is sorted by ``name``
        and each entry's ``source_locations`` is sorted by
        ``(path, line)`` for deterministic output.
    """
    detections: list[_Detection] = []
    for path in repository_contents.file_paths:
        if _should_skip_file(path):
            continue
        content = repository_contents.read_text(path)
        if content is None:
            continue
        detections.extend(_scan_urls(path, content))
        detections.extend(_scan_sdk_patterns(path, content))
    return _aggregate(detections)


__all__ = ["detect_external_services"]

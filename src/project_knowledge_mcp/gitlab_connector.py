"""GitLab_Connector: low-level GitLab REST API client wrapper.

This module implements the network-facing primitives that the
``GitLab_Connector`` component uses to talk to a GitLab instance:

* a single authenticated ``httpx.Client`` configured from
  ``Config.gitlab_base_url`` and ``Config.gitlab_access_token``;
* a private ``_get`` helper that issues an authenticated ``GET`` against an
  API path resolved under ``{base_url}/api/v4/``; and
* a private ``_paginate`` helper that yields every item across all pages of
  a list endpoint, following GitLab's RFC 5988 ``Link: rel="next"`` header
  when present and falling back to ``?page=N`` traversal otherwise.

The higher-level operations -- ``enumerate_projects`` (Requirements 2.1,
2.2, 2.3, 2.4, 15.4, 15.5) and ``fetch_repository_contents``
(Requirement 15.3) -- are implemented in tasks 5.2 and 5.3 on top of these
primitives. ``fetch_repository_contents`` always uses the supplied
``commit_sha`` as the ``ref`` for both the tree listing and every
individual file fetch, so contents are always read from the configured
``Analysis_Branch`` regardless of GitLab's project-level default branch.

This module deliberately keeps the low-level ``_get`` helper free of
status-code interpretation -- ``_get`` returns the raw :class:`httpx.Response`
so each higher-level method can apply the design's error rules in one
place. ``fetch_repository_contents`` translates HTTP 401/403 into
:class:`~project_knowledge_mcp.errors.GitLabAuthError` to mirror the
"any call during enumeration" rule from the design's GitLab_Connector
section, and surfaces other non-2xx statuses via
:meth:`httpx.Response.raise_for_status`.

Implements Requirement 2.5 (paginated retrieval of all pages before
completing enumeration) and Requirement 15.3 (analysis-branch-pinned
repository fetches).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import quote

import httpx

from .errors import GitLabAuthError, GitLabGroupNotFoundError
from .models import EnumeratedProject, RepositoryContents

if TYPE_CHECKING:
    from types import TracebackType

    from .config import Config


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default ``per_page`` query parameter used by :meth:`GitLabConnector._paginate`
#: when the caller does not supply one. GitLab's REST API caps ``per_page`` at
#: 100 for list endpoints, so 100 is both the maximum and the natural default
#: for minimizing round-trips during enumeration.
DEFAULT_PER_PAGE: Final[int] = 100

#: GitLab REST API v4 path prefix appended to ``Config.gitlab_base_url`` by
#: :meth:`GitLabConnector._build_url`.
API_PREFIX: Final[str] = "/api/v4"

#: HTTP header GitLab accepts for personal access token authentication.
PRIVATE_TOKEN_HEADER: Final[str] = "PRIVATE-TOKEN"

#: HTTP status codes that mean "the credentials we sent are not accepted".
#: GitLab uses ``401`` for missing/invalid tokens and ``403`` for tokens
#: that authenticate successfully but lack the required scope. Both flow
#: through the connector as :class:`GitLabAuthError` (Requirement 2.3).
_HTTP_UNAUTHORIZED: Final[int] = 401
_HTTP_FORBIDDEN: Final[int] = 403

#: HTTP status code used for "the resource you asked about does not exist".
#: Translation depends on context: a 404 on the configured group is a fatal
#: configuration error (Requirement 2.4); a 404 on a per-project branch
#: lookup is a "branch missing" signal that triggers a Skip rather than an
#: abort (Requirement 15.5); a 404 on a per-file fetch (in
#: :meth:`GitLabConnector._fetch_blob_text`) is a benign race in which the
#: blob disappeared between the tree listing and the file fetch.
_HTTP_NOT_FOUND: Final[int] = 404

#: Default per-request timeout. GitLab's list endpoints can take several
#: seconds for large groups; 30s is comfortable without making the process
#: feel hung when GitLab is unreachable.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0

#: Maximum number of concurrent in-flight blob-fetch requests inside
#: :meth:`GitLabConnector.fetch_repository_contents`. Files within a
#: single project are fetched in parallel on a bounded thread pool —
#: every project ingestion does one tree listing followed by N blob
#: fetches, and the per-file latency on a self-signed-cert or
#: geographically distant GitLab is the dominant cost. Eight workers
#: yields a 5-8x speedup in practice without saturating GitLab's API
#: rate limit (default GitLab limit is 600 req/min, well above what
#: a single ingestion produces even at this concurrency).
DEFAULT_BLOB_FETCH_CONCURRENCY: Final[int] = 8

#: RFC 5988 ``Link`` header entry pattern: ``<url>; rel="name"``. The pattern
#: is intentionally narrow -- GitLab emits exactly this shape -- so we do not
#: need a full RFC parser.
_LINK_ENTRY_RE: Final[re.Pattern[str]] = re.compile(
    r'<(?P<url>[^>]+)>\s*;\s*rel="(?P<rel>[^"]+)"'
)


# ---------------------------------------------------------------------------
# Heuristic for "analyzable" file paths used by fetch_repository_contents
# ---------------------------------------------------------------------------

#: Maximum byte size of an individual file loaded into ``RepositoryContents``.
#: Files larger than this are skipped so the in-memory snapshot stays bounded
#: and so a single huge generated artifact in a project does not blow up the
#: ``Ingestion_Job``. 1 MiB is comfortably above any real README, manifest,
#: or hand-written source file.
MAX_FILE_SIZE_BYTES: Final[int] = 1 * 1024 * 1024

#: Hard cap on the number of files loaded for one project. Most real projects
#: are well below this, but the cap protects against pathological repositories
#: (e.g. a vendored dependency tree of 100k JavaScript files) that would
#: otherwise produce an unbounded fetch storm and an unbounded mapping.
MAX_ANALYZABLE_FILES: Final[int] = 5000

#: Bare basenames (no directory, exact match, case-sensitive) treated as
#: analyzable manifests / configuration files. The ``Project_Analyzer``'s
#: purpose summarizer and external-service detector key off these.
_ANALYZABLE_BASENAMES: Final[frozenset[str]] = frozenset({
    "Dockerfile",
    "Gemfile",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "go.mod",
    "package.json",
    "pom.xml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "Cargo.toml",
})

#: File extensions (lowercased, including the leading dot) treated as
#: analyzable source code. Source extensions cover the languages the
#: ``Project_Analyzer`` knows how to inspect statically.
_ANALYZABLE_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".py",
    ".rb",
    ".rs",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
})


def _is_analyzable_path(path: str) -> bool:
    """Return True when ``path`` is worth fetching for static analysis.

    The heuristic admits three families:

    * Any file whose basename starts with ``README`` (case-insensitive),
      at any depth, so the purpose summarizer can find ``README.md``,
      ``docs/README.rst``, etc.
    * Common manifest / build files listed in :data:`_ANALYZABLE_BASENAMES`
      (``package.json``, ``pyproject.toml``, ``pom.xml``, ``Dockerfile``,
      ...).
    * Source files whose extension is in :data:`_ANALYZABLE_EXTENSIONS`
      (``.py``, ``.js``, ``.ts``, ``.go``, ``.java``, ``.rb``, ``.kt``,
      ``.rs``, ``.sql``, ``.yaml``, ``.yml``, ``.toml``).

    Anything else (lockfiles, images, fonts, vendored bundles, ...) is
    skipped. This keeps the in-memory ``RepositoryContents`` text-only
    (its ``files`` mapping is ``Mapping[str, str]``) and bounds the
    per-project fetch cost.

    Files under any directory segment literally named ``vendor`` are
    *also* rejected here, regardless of extension. ``vendor/`` is the
    standard convention for bundled third-party code in Go (and in
    PHP/Composer, Ruby, and several other ecosystems); the Go parser
    already filters such paths via :func:`is_go_source_file`
    (Requirement 1.3), so fetching them only to discard them later is
    pure overhead. A file literally named ``vendor.go`` (or any file
    with the bare ``vendor`` basename) at any depth is unaffected —
    only directory segments named exactly ``vendor`` cause exclusion.
    """
    if not path:
        return False
    # Mirror :func:`is_go_source_file`'s segment-matching rule so the
    # fetch-side filter is consistent with the parser-side filter:
    # normalize Windows separators, split on ``/``, and reject when
    # any directory segment (i.e. anything but the basename) equals
    # ``vendor`` exactly. Case-sensitive so a directory legitimately
    # named ``Vendor`` (proper noun, brand) is preserved.
    normalized = path.replace("\\", "/")
    segments = normalized.split("/")
    if any(seg == "vendor" for seg in segments[:-1]):
        return False
    # ``rsplit`` to be defensive against backslashes (GitLab paths are
    # always forward-slash separated, but the cost of being explicit is
    # negligible).
    basename = path.rsplit("/", 1)[-1]
    if basename.lower().startswith("readme"):
        return True
    if basename in _ANALYZABLE_BASENAMES:
        return True
    dot = basename.rfind(".")
    if dot < 0:
        return False
    extension = basename[dot:].lower()
    return extension in _ANALYZABLE_EXTENSIONS


def _is_analyzable_path_for_go(path: str) -> bool:
    """Stricter admission rule for repositories that have a root ``go.mod``.

    The Go analyzer pipeline (purpose summarizer, I/O extractor,
    external-service detector, database-table detector) consults
    exactly three file categories:

    * Non-test, non-vendor, non-third_party ``.go`` source files.
    * The root ``go.mod`` for purpose summary and the
      ``has_go_artefacts`` guard.
    * ``README*`` (case-insensitive) at any depth for the purpose
      summarizer.

    Everything else — ``.proto`` definitions, ``*.pb.gw.go`` is
    actually ``.go`` so still admitted, ``*.swagger.json`` API specs,
    ``docs/*.js`` swagger UI bundles, ``*.wsdl`` SOAP descriptors,
    ``sampleInput/*.xml`` test fixtures, ``*.sql`` migration scripts,
    YAML config files, lockfiles, manifests of unrelated languages —
    is pure fetch overhead because the Go analyzer ignores it. For an
    ESB-scale catalog this cuts the per-project fetch by an order of
    magnitude (a typical Go microservice in the operator's group has
    ~80 generated/test/docs files vs. ~25 hand-written ``.go`` files).

    Excluded patterns:

    * Any path with a directory segment literally named ``vendor`` or
      ``third_party`` (case-sensitive). ``vendor/`` is Go's standard
      bundled-dependencies convention; ``third_party/`` is the common
      label for vendored Google-style protobuf bundles.
    * Files whose basename ends in ``_test.go`` — Go test files never
      execute in production and the URL/SQL literals they contain
      (test fixtures) are not real service or table dependencies.
    """
    if not path:
        return False

    # Reject vendor / third_party directory segments. Mirrors the
    # rule used by :func:`is_go_source_file` (Requirement 1.3) and
    # extends it with the third_party convention. Case-sensitive so
    # an operator's legitimately-named ``Vendor`` or ``ThirdParty``
    # directory is preserved.
    normalized = path.replace("\\", "/")
    segments = normalized.split("/")
    excluded_dirs = {"vendor", "third_party"}
    if any(seg in excluded_dirs for seg in segments[:-1]):
        return False

    basename = segments[-1]

    # ``README*`` at any depth — the purpose summarizer's primary
    # source.
    if basename.lower().startswith("readme"):
        return True

    # ``go.mod`` at the repository root. Nested ``go.mod`` files
    # belong to vendored sub-modules and are rejected.
    if path == "go.mod":
        return True

    # ``.go`` source files except ``*_test.go``. Generated ``.pb.go``
    # and ``.pb.gw.go`` files are kept because the gRPC-gateway
    # routes they contain are legitimate HTTP API endpoints the
    # I/O extractor needs.
    if basename.endswith(".go") and not basename.endswith("_test.go"):
        return True

    return False


# Type alias for a JSON object payload returned by GitLab. Defined as a
# module-level alias so the signatures of :meth:`_paginate` and helpers stay
# readable, and so the ``Any`` that JSON values fundamentally need is
# declared in one place rather than sprinkled through annotations.
JsonObject = Mapping[str, Any]


def _parse_link_next(link_header: str | None) -> str | None:
    """Return the URL of the ``rel="next"`` entry in a ``Link`` header.

    Args:
        link_header: The raw value of GitLab's ``Link`` response header, or
            ``None`` when the response did not include one.

    Returns:
        The absolute URL of the next page, or ``None`` when the header is
        absent or carries no ``rel="next"`` entry (i.e. the current page is
        the last page).
    """
    if not link_header:
        return None
    for match in _LINK_ENTRY_RE.finditer(link_header):
        if match.group("rel") == "next":
            return match.group("url")
    return None


# ---------------------------------------------------------------------------
# GitLabConnector
# ---------------------------------------------------------------------------


class GitLabConnector:
    """Authenticated wrapper around the GitLab REST API.

    The connector owns (or borrows) a single :class:`httpx.Client`. It
    attaches a ``PRIVATE-TOKEN`` header to every request and resolves
    relative API paths against ``{base_url}/api/v4/``. It does not interpret
    status codes; callers (the higher-level methods added in tasks 5.2
    and 5.3) are responsible for translating non-2xx responses into the
    domain errors defined in :mod:`project_knowledge_mcp.errors`.

    The connector can be constructed in two ways:

    * From a validated :class:`~project_knowledge_mcp.config.Config`, which
      is the production wiring path: ``GitLabConnector(config)``.
    * From explicit ``base_url`` and ``access_token`` keyword arguments,
      which is the testability path: ``GitLabConnector(base_url=..., access_token=...)``.

    An :class:`httpx.Client` may be supplied via the ``client`` keyword to
    let tests inject a transport-level fake (e.g. ``httpx.MockTransport``).
    When the connector creates its own client, it closes that client on
    :meth:`close` (and on context-manager exit). When the client was
    supplied externally, the connector leaves it open -- ownership stays
    with the caller.

    The instance is also a context manager so callers can write::

        with GitLabConnector(config) as gl:
            for project in gl._paginate("groups/foo/projects", {}):
                ...
    """

    _base_url: str
    _access_token: str
    _group_path: str | None
    _analysis_branch: str | None
    _verify_ssl: bool
    _client: httpx.Client
    _owns_client: bool

    def __init__(
        self,
        config: Config | None = None,
        *,
        base_url: str | None = None,
        access_token: str | None = None,
        group_path: str | None = None,
        analysis_branch: str | None = None,
        verify_ssl: bool | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Build a :class:`GitLabConnector`.

        Args:
            config: A validated :class:`Config`. When supplied, every other
                value-bearing keyword (``base_url``, ``access_token``,
                ``group_path``, ``analysis_branch``) is ignored.
            base_url: GitLab instance base URL. Required when ``config`` is
                ``None``.
            access_token: GitLab access token. Required when ``config`` is
                ``None``.
            group_path: GitLab group path the connector enumerates against.
                Required by :meth:`enumerate_projects`. May be omitted when
                only the lower-level helpers are used (e.g. a unit test that
                drives :meth:`_paginate` against an arbitrary list endpoint).
            analysis_branch: Configured ``Analysis_Branch`` name used by
                :meth:`enumerate_projects` to look up the per-project
                commit SHA (Requirement 15.4). Required by
                :meth:`enumerate_projects`; may be omitted otherwise.
            verify_ssl: Whether the owned :class:`httpx.Client` validates
                TLS certificates. Ignored when ``config`` is supplied
                (the connector reads ``config.gitlab_verify_ssl``
                instead) and when ``client`` is supplied (the caller's
                client already has its own TLS configuration). Defaults
                to ``True`` when neither ``config`` nor ``client`` is
                given. ``False`` disables certificate validation for
                every outbound request and is intended for self-signed
                GitLab instances; doing so removes MITM protection.
            client: Optional pre-built :class:`httpx.Client`. When supplied,
                the connector does *not* take ownership and will not close
                the client on :meth:`close`. When omitted, the connector
                creates and owns a client with a sensible default timeout.

        Raises:
            ValueError: If neither ``config`` nor both of ``base_url`` /
                ``access_token`` are supplied.
        """
        if config is not None:
            resolved_base_url = config.gitlab_base_url
            resolved_token = config.gitlab_access_token
            resolved_group_path: str | None = config.gitlab_group_path
            resolved_analysis_branch: str | None = config.analysis_branch
            resolved_verify_ssl = config.gitlab_verify_ssl
        else:
            if base_url is None or access_token is None:
                raise ValueError(
                    "GitLabConnector requires either a Config or both "
                    "'base_url' and 'access_token' keyword arguments"
                )
            resolved_base_url = base_url
            resolved_token = access_token
            resolved_group_path = group_path
            resolved_analysis_branch = analysis_branch
            resolved_verify_ssl = True if verify_ssl is None else verify_ssl

        # Strip a single trailing slash so :meth:`_build_url` can append
        # ``/api/v4/...`` without producing ``//api/v4/...``.
        self._base_url = resolved_base_url.rstrip("/")
        self._access_token = resolved_token
        self._group_path = resolved_group_path
        self._analysis_branch = resolved_analysis_branch
        self._verify_ssl = resolved_verify_ssl

        if client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
                verify=resolved_verify_ssl,
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources held by the connector.

        When the connector created its own :class:`httpx.Client`, that
        client is closed. When the client was supplied externally, ownership
        stays with the caller and this method is a no-op.
        """
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GitLabConnector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # URL composition and headers
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The configured GitLab instance base URL, with no trailing slash."""
        return self._base_url

    def _build_url(self, path: str) -> str:
        """Resolve a GitLab API path to an absolute URL.

        ``path`` may be supplied with or without a leading ``/`` and with or
        without a leading ``api/v4/`` segment; the canonical form is the
        bare API path (e.g. ``groups/foo/projects``). Any of the following
        produce the same URL::

            "groups/foo/projects"
            "/groups/foo/projects"
            "api/v4/groups/foo/projects"
            "/api/v4/groups/foo/projects"
        """
        normalized = path.lstrip("/")
        api_prefix_no_slash = API_PREFIX.lstrip("/") + "/"
        if normalized.startswith(api_prefix_no_slash):
            normalized = normalized[len(api_prefix_no_slash) :]
        return f"{self._base_url}{API_PREFIX}/{normalized}"

    def _auth_headers(self) -> dict[str, str]:
        """Return the headers attached to every outgoing request."""
        return {PRIVATE_TOKEN_HEADER: self._access_token}

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Mapping[str, Any]) -> httpx.Response:
        """Issue an authenticated ``GET`` against a GitLab API path.

        The path is resolved through :meth:`_build_url` so callers pass the
        bare API path (e.g. ``groups/{id}/projects``). The
        ``PRIVATE-TOKEN`` header is attached automatically.

        This helper intentionally does **not** raise on non-2xx responses;
        higher-level methods (added in tasks 5.2 and 5.3) inspect
        ``response.status_code`` and translate to the domain errors in
        :mod:`project_knowledge_mcp.errors`.

        Args:
            path: The GitLab API path. Leading ``/`` and a leading
                ``api/v4/`` segment are both tolerated.
            params: Query-string parameters. May be empty. ``Any`` values
                are accepted because GitLab list endpoints take a mix of
                strings, integers, and booleans.

        Returns:
            The raw :class:`httpx.Response`.
        """
        url = self._build_url(path)
        return self._client.get(
            url,
            params=dict(params),
            headers=self._auth_headers(),
        )

    def _get_absolute(self, url: str) -> httpx.Response:
        """Issue an authenticated ``GET`` against an already-resolved URL.

        Used by :meth:`_paginate` to follow the URL extracted from a
        ``Link: rel="next"`` header verbatim, preserving any pagination
        cursor encoded by GitLab.
        """
        return self._client.get(url, headers=self._auth_headers())

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def _paginate(
        self,
        path: str,
        params: Mapping[str, Any],
    ) -> Iterator[JsonObject]:
        """Yield every item across all pages of a GitLab list endpoint.

        The first page is fetched against ``path``. If the response carries
        a ``Link`` header with a ``rel="next"`` entry, subsequent pages are
        fetched by following those URLs verbatim until no ``rel="next"`` is
        present (the canonical GitLab pagination protocol). When the first
        response carries no ``Link`` header at all, the connector falls
        back to ``?page=N`` traversal: pages 2, 3, ... are requested with
        the same ``params`` until a page returns an empty list.

        ``per_page`` is honored when set in ``params``; otherwise it
        defaults to :data:`DEFAULT_PER_PAGE` (100). Either way the value is
        carried through to the fallback pagination loop so every page uses
        the same page size.

        Implements Requirement 2.5: every page is retrieved before the
        iterator is exhausted.

        Args:
            path: The GitLab API path of a list endpoint.
            params: Query-string parameters. ``per_page`` may be supplied
                to override the default.

        Yields:
            Each JSON object from the concatenated pages, in the order
            GitLab returned them.
        """
        merged: dict[str, Any] = dict(params)
        merged.setdefault("per_page", DEFAULT_PER_PAGE)

        # First page.
        response = self._get(path, merged)
        yield from _items_from_response(response)

        link_header = response.headers.get("Link")
        next_url = _parse_link_next(link_header)

        if link_header is not None:
            # GitLab is supplying RFC 5988 pagination -- follow ``next``
            # links until exhausted. Even when the first response has no
            # ``rel="next"`` (i.e. only one page), the presence of the
            # ``Link`` header tells us GitLab speaks Link-based pagination
            # and we must not fall back to ``?page=N`` probing.
            while next_url is not None:
                response = self._get_absolute(next_url)
                yield from _items_from_response(response)
                next_url = _parse_link_next(response.headers.get("Link"))
            return

        # No Link header at all: fall back to manual ``?page=N`` traversal,
        # stopping as soon as a page returns zero items. Page 1 was the
        # initial request above, so we start at page 2.
        page = 2
        while True:
            page_params = dict(merged)
            page_params["page"] = page
            response = self._get(path, page_params)
            page_items = _items_from_response(response)
            if not page_items:
                break
            yield from page_items
            page += 1

    # ------------------------------------------------------------------
    # Project enumeration
    # ------------------------------------------------------------------

    def enumerate_projects(self) -> Iterator[EnumeratedProject]:
        """Yield every descendant project of the configured group.

        Calls ``GET /api/v4/groups/:group/projects`` with
        ``include_subgroups=true`` so the result is the full set of
        repositories under the configured ``gitlab.group_path`` -- including
        every nested subgroup -- with no per-subgroup recursion at the
        client (Requirement 2.1). Pagination is driven by :meth:`_paginate`,
        which retrieves every page before the iterator is exhausted
        (Requirement 2.5).

        For each enumerated project the connector then issues
        ``GET /api/v4/projects/:id/repository/branches/:branch`` against
        the configured ``Analysis_Branch`` to obtain the most recent
        commit SHA on that branch (Requirements 2.2, 15.4):

        * **Branch present (HTTP 200):** the SHA is recorded as
          ``analysis_branch_commit_sha`` and ``branch_missing`` is ``False``.
        * **Branch absent (HTTP 404):** ``analysis_branch_commit_sha`` is
          ``None`` and ``branch_missing`` is ``True`` so the
          ``Ingestion_Coordinator`` can record an
          ``"analysis_branch_missing"`` :class:`Skip` entry naming both
          the configured ``Analysis_Branch`` value and the project's
          ``gitlab_project_id`` (Requirement 15.5).

        Authentication failures translate to domain errors:

        * ``HTTP 401`` or ``HTTP 403`` from any call (group listing or
          per-project branch lookup) raises :class:`GitLabAuthError` with
          the response status code (Requirement 2.3). The
          ``Ingestion_Coordinator`` is responsible for the abort-and-report
          try/finally semantics described in the design's error-handling
          table.
        * ``HTTP 404`` on the configured group itself raises
          :class:`GitLabGroupNotFoundError` carrying the configured group
          path (Requirement 2.4). A 404 on a per-project branch lookup is
          *not* an error -- it is the canonical "branch missing" signal.

        Yields:
            One :class:`EnumeratedProject` per descendant repository, in
            ascending GitLab project ID order (the GitLab API is queried
            with ``order_by=id&sort=asc`` so the stream is deterministic
            across pages).

        Raises:
            GitLabAuthError: When any GitLab call returns HTTP 401 or 403.
            GitLabGroupNotFoundError: When the configured group returns
                HTTP 404.
            RuntimeError: When the connector was constructed without
                ``group_path`` or ``analysis_branch`` (the lower-level
                helpers can be exercised without these, but enumeration
                cannot).
        """
        group_path = self._group_path
        analysis_branch = self._analysis_branch
        if group_path is None or analysis_branch is None:
            raise RuntimeError(
                "GitLabConnector.enumerate_projects requires both "
                "'group_path' and 'analysis_branch' to be configured"
            )

        for project_json in self._iter_projects_in_group(group_path):
            yield self._build_enumerated_project(project_json, analysis_branch)

    def _iter_projects_in_group(self, group_path: str) -> Iterator[JsonObject]:
        """Yield every descendant project of ``group_path``, paginated.

        Status-code translation is concentrated here: a 404 on the *first*
        request identifies the configured group as missing
        (Requirement 2.4); a 401/403 on any page is an auth failure that
        must abort the job (Requirement 2.3); any other non-2xx status
        is surfaced through :meth:`httpx.Response.raise_for_status` so it
        propagates as a generic HTTP error rather than being silently
        ignored.
        """
        # ``safe=""`` quotes ``/`` to ``%2F`` so a nested group path like
        # ``acme/platform`` becomes a single GitLab "URL-encoded ID" path
        # segment. httpx preserves percent-encoded characters in URL
        # paths, so the encoding survives transport intact.
        quoted_group = quote(group_path, safe="")
        list_path = f"groups/{quoted_group}/projects"
        # ``include_subgroups=true`` is what makes a single call return
        # every descendant repository (Requirement 2.1). ``order_by=id``
        # plus ``sort=asc`` gives a deterministic enumeration order across
        # pages, which makes Property 2 (set equality) easy to test.
        base_params: dict[str, Any] = {
            "include_subgroups": "true",
            "per_page": DEFAULT_PER_PAGE,
            "order_by": "id",
            "sort": "asc",
        }

        # First page: this is the only request that can identify the
        # configured group as missing.
        response = self._get(list_path, base_params)
        if response.status_code == _HTTP_NOT_FOUND:
            raise GitLabGroupNotFoundError(group_path)
        self._raise_for_auth(response)
        response.raise_for_status()
        yield from _items_from_response(response)

        link_header = response.headers.get("Link")
        next_url = _parse_link_next(link_header)

        if link_header is not None:
            # GitLab speaks RFC 5988 pagination -- follow ``rel="next"``
            # links until exhausted, even if the first page was the only
            # page (in which case ``next_url`` is already None).
            while next_url is not None:
                response = self._get_absolute(next_url)
                self._raise_for_auth(response)
                response.raise_for_status()
                yield from _items_from_response(response)
                next_url = _parse_link_next(response.headers.get("Link"))
            return

        # No ``Link`` header at all (e.g. a fake or older GitLab): fall
        # back to manual ``?page=N`` traversal, stopping as soon as a page
        # returns no items.
        page = 2
        while True:
            page_params = dict(base_params)
            page_params["page"] = page
            response = self._get(list_path, page_params)
            self._raise_for_auth(response)
            response.raise_for_status()
            page_items = _items_from_response(response)
            if not page_items:
                break
            yield from page_items
            page += 1

    def _build_enumerated_project(
        self,
        project_json: JsonObject,
        analysis_branch: str,
    ) -> EnumeratedProject:
        """Materialize an :class:`EnumeratedProject` from one list-endpoint item.

        Issues the per-project ``Analysis_Branch`` lookup so the resulting
        record carries the most recent commit SHA on that branch (when the
        branch exists) or the ``branch_missing`` signal (when it does not).
        """
        project_id = int(project_json["id"])
        full_path = str(project_json["path_with_namespace"])
        # GitLab returns ``description`` as either a non-empty string,
        # an empty string, or JSON ``null``. Coalesce empty/None into
        # ``None`` so the resulting EnumeratedProject distinguishes
        # "no description" from "empty string" cleanly.
        raw_description = project_json.get("description")
        description: str | None = (
            raw_description
            if isinstance(raw_description, str) and raw_description != ""
            else None
        )

        sha = self._fetch_branch_sha(project_id, analysis_branch)
        if sha is None:
            # Requirement 15.5: branch missing -> branch_missing=True and
            # commit SHA is null. The Ingestion_Coordinator will record a
            # Skip entry naming the analysis_branch and gitlab_project_id.
            return EnumeratedProject(
                gitlab_project_id=project_id,
                full_path=full_path,
                analysis_branch_name=analysis_branch,
                analysis_branch_commit_sha=None,
                branch_missing=True,
                repository_description=description,
            )
        return EnumeratedProject(
            gitlab_project_id=project_id,
            full_path=full_path,
            analysis_branch_name=analysis_branch,
            analysis_branch_commit_sha=sha,
            branch_missing=False,
            repository_description=description,
        )

    def _fetch_branch_sha(self, project_id: int, branch: str) -> str | None:
        """Return the most recent commit SHA on ``branch`` for ``project_id``.

        Returns ``None`` when the branch does not exist on the project
        (HTTP 404), which is the canonical "branch missing" signal that
        :meth:`_build_enumerated_project` translates into ``branch_missing``
        on the resulting :class:`EnumeratedProject` (Requirement 15.5).
        Raises :class:`GitLabAuthError` on HTTP 401/403 (Requirement 2.3)
        and surfaces other non-2xx statuses through
        :meth:`httpx.Response.raise_for_status`.
        """
        # ``safe=""`` quotes any reserved characters in the branch name
        # (notably ``/`` for branches like ``release/v1``); GitLab's branch
        # endpoint requires the branch segment to be URL-encoded.
        encoded_branch = quote(branch, safe="")
        path = f"projects/{project_id}/repository/branches/{encoded_branch}"
        response = self._get(path, {})
        if response.status_code == _HTTP_NOT_FOUND:
            return None
        self._raise_for_auth(response)
        response.raise_for_status()

        # GitLab's branch payload is ``{"name": ..., "commit": {"id": "<sha>", ...}, ...}``.
        payload = response.json()
        commit = payload["commit"]
        sha = commit["id"]
        return str(sha)

    def _raise_for_auth(self, response: httpx.Response) -> None:
        """Raise :class:`GitLabAuthError` when ``response`` is 401 or 403.

        Implements Requirement 2.3's "any call during enumeration"
        clause: every GitLab response observed by enumeration flows
        through this helper before its body is interpreted.
        """
        if response.status_code in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
            raise GitLabAuthError(response.status_code)

    # ------------------------------------------------------------------
    # Repository contents
    # ------------------------------------------------------------------

    def fetch_repository_contents(
        self,
        project_id: int,
        commit_sha: str,
    ) -> RepositoryContents:
        """Return the analyzable text files of a project at ``commit_sha``.

        Both the tree listing and every individual file fetch use
        ``ref=commit_sha`` (Requirement 15.3): the contents are always read
        from the configured ``Analysis_Branch``'s commit, regardless of
        GitLab's per-project ``default_branch``. Callers therefore get a
        snapshot pinned to that exact commit even if the branch advances
        between the tree fetch and the file fetches.

        Heuristic for which files to load (kept narrow to bound the in-memory
        snapshot, since :class:`~project_knowledge_mcp.models.RepositoryContents`
        holds a ``Mapping[str, str]`` rather than a streaming accessor):

        * Skip directories and submodules (``type != "blob"``).
        * Skip paths that are not "analyzable" per
          :func:`_is_analyzable_path` -- only READMEs at any depth, common
          manifests (``package.json``, ``pyproject.toml``, ``pom.xml``,
          ``Dockerfile``, ...), and source files whose extension is in
          :data:`_ANALYZABLE_EXTENSIONS` are loaded.
        * Skip files larger than :data:`MAX_FILE_SIZE_BYTES` (1 MiB).
        * Skip files that cannot be decoded as UTF-8 (binary content), so
          the resulting mapping is text-only.
        * Stop iterating once :data:`MAX_ANALYZABLE_FILES` files have been
          loaded; this is a defensive cap against pathological repositories
          (e.g. vendored dependency trees) and is well above any
          hand-written project's analyzable surface.

        Args:
            project_id: The numeric GitLab project ID. The connector uses
                the numeric form so the URL never has to encode a path-style
                project identifier.
            commit_sha: The commit SHA on the configured ``Analysis_Branch``
                from which to fetch the tree and every file. Recorded
                verbatim on the returned ``RepositoryContents``.

        Returns:
            A :class:`RepositoryContents` whose ``files`` mapping holds one
            entry per loaded analyzable file, keyed by repository-relative
            path.

        Raises:
            GitLabAuthError: If any tree-listing or file-fetch request
                returns HTTP 401 or 403, mirroring the design's
                "any call during enumeration" rule for fetch operations
                so the ``Ingestion_Coordinator`` can apply the same
                abort-and-report behavior.
            httpx.HTTPStatusError: For any other non-2xx status returned
                by the GitLab API.
        """
        # Step 1: collect every analyzable path from the recursive tree
        # listing. The tree fetch is sequential by design — GitLab's
        # pagination is stateful per ``Link`` header and the page count
        # is small relative to the blob count, so parallelizing the
        # tree fetch would not move the needle.
        analyzable_paths = self._collect_analyzable_paths(project_id, commit_sha)

        # Apply the ``MAX_ANALYZABLE_FILES`` cap after sorting so the
        # set of files that survives the cap is deterministic across
        # runs even when GitLab's tree-pagination order varies. Sorting
        # also matches what callers see in ``RepositoryContents.files``
        # (a ``Mapping`` whose iteration order follows insertion in the
        # source dict; here insertion is by sorted path).
        if len(analyzable_paths) > MAX_ANALYZABLE_FILES:
            analyzable_paths = analyzable_paths[:MAX_ANALYZABLE_FILES]

        # Step 2: fetch every analyzable blob concurrently on a bounded
        # thread pool. ``httpx.Client`` is thread-safe for concurrent
        # reads (see ``httpx`` docs: "A Client instance maintains a
        # connection pool and can be used safely from multiple threads
        # simultaneously"), so the only resource we coordinate is the
        # worker count.
        files = self._fetch_blobs_concurrent(
            project_id, analyzable_paths, commit_sha
        )

        return RepositoryContents(
            gitlab_project_id=project_id,
            commit_sha=commit_sha,
            files=files,
        )

    def _collect_analyzable_paths(
        self,
        project_id: int,
        commit_sha: str,
    ) -> list[str]:
        """Return every analyzable repository-relative path at ``commit_sha``.

        Iterates the paginated recursive tree listing once, materializes
        the blob paths, and then chooses the admission filter based on
        the project's shape:

        * **Go project** (root ``go.mod`` present): only ``.go`` source
          (excluding ``_test.go``, ``vendor/**``, ``third_party/**``),
          the root ``go.mod`` itself, and ``README*`` files are kept.
          This drastically reduces the fetch set for ESB microservices
          (a ~150-project group goes from minutes-per-project to
          seconds-per-project) because the analyzer's Go pipeline
          ignores everything else anyway.
        * **Non-Go project**: the parent-spec ``_is_analyzable_path``
          rule applies — READMEs, common manifest basenames, and
          source files in the documented extension set.

        The returned list is sorted so a later ``MAX_ANALYZABLE_FILES``
        cap keeps a deterministic set of files across runs.
        """
        tree_path = f"projects/{project_id}/repository/tree"
        # ``recursive=true`` makes the tree endpoint walk every subdirectory
        # in a single paginated stream, which is exactly what the analyzer
        # needs and avoids one round-trip per directory. ``per_page`` is left
        # to ``_paginate``'s default of 100, the GitLab maximum.
        tree_params: dict[str, Any] = {
            "ref": commit_sha,
            "recursive": "true",
        }

        # First pass: materialize every blob path in the tree so we
        # can sniff for the ``go.mod`` marker before deciding which
        # admission filter to apply.
        blob_paths: list[str] = []
        for entry in self._paginate(tree_path, tree_params):
            # GitLab tree entries carry a ``type`` of ``"blob"`` (file),
            # ``"tree"`` (subdirectory), or ``"commit"`` (submodule). Only
            # blobs have fetchable text content.
            if entry.get("type") != "blob":
                continue
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or raw_path == "":
                continue
            blob_paths.append(raw_path)

        # A root ``go.mod`` is the canonical marker for "this is a Go
        # project". Nested ``go.mod`` files inside vendored sub-modules
        # are irrelevant for the language sniff.
        is_go_project = "go.mod" in blob_paths
        admit = (
            _is_analyzable_path_for_go if is_go_project else _is_analyzable_path
        )

        paths = [p for p in blob_paths if admit(p)]
        paths.sort()
        return paths

    def _fetch_blobs_concurrent(
        self,
        project_id: int,
        paths: list[str],
        commit_sha: str,
    ) -> dict[str, str]:
        """Fetch every blob in ``paths`` concurrently and return a dict.

        Uses a :class:`concurrent.futures.ThreadPoolExecutor` bounded
        by :data:`DEFAULT_BLOB_FETCH_CONCURRENCY`. Files that
        :meth:`_fetch_blob_text` returns ``None`` for (oversized or
        non-UTF-8) are skipped silently, matching the original
        sequential implementation. Auth errors from individual fetches
        propagate as :class:`GitLabAuthError` so the
        ``Ingestion_Coordinator`` can abort the snapshot.
        """
        if not paths:
            return {}

        files: dict[str, str] = {}

        def fetch_one(path: str) -> tuple[str, str | None]:
            return path, self._fetch_blob_text(project_id, path, commit_sha)

        # The executor's ``map`` is used because:
        #   1. It preserves the input order in the output iterator, so
        #      the resulting ``files`` dict iterates in sorted-path
        #      order (matching the order callers observed when the
        #      fetch was sequential and the tree was already roughly
        #      sorted by GitLab).
        #   2. The first exception raised by any worker propagates the
        #      moment we iterate to that result. ``GitLabAuthError`` —
        #      the only fetch exception ``Ingestion_Coordinator`` cares
        #      about — therefore surfaces as soon as the corresponding
        #      worker finishes, exactly as in the sequential code.
        with ThreadPoolExecutor(
            max_workers=DEFAULT_BLOB_FETCH_CONCURRENCY,
            thread_name_prefix="gitlab-blob-fetch",
        ) as executor:
            for path, text in executor.map(fetch_one, paths):
                if text is not None:
                    files[path] = text

        return files

    def _fetch_blob_text(
        self,
        project_id: int,
        path: str,
        commit_sha: str,
    ) -> str | None:
        """Return the text content of a single file at ``commit_sha``.

        Uses the ``GET /projects/:id/repository/files/:path/raw?ref=:sha``
        endpoint with ``:path`` URL-encoded so embedded slashes survive
        intact (GitLab requires ``/`` -> ``%2F`` in the path segment of
        this endpoint). The ``ref`` query parameter pins the response to
        the same commit on the configured ``Analysis_Branch``
        (Requirement 15.3).

        Returns ``None`` -- so the caller can drop the file from the
        snapshot without aborting the whole repository fetch -- when:

        * the file disappeared between the tree listing and the fetch
          (HTTP 404);
        * the body is larger than :data:`MAX_FILE_SIZE_BYTES`; or
        * the body is not valid UTF-8 (i.e. it is binary).

        Raises :class:`GitLabAuthError` on HTTP 401/403 and surfaces other
        non-2xx statuses through :meth:`httpx.Response.raise_for_status`.
        """
        # ``safe=""`` quotes every reserved character including ``/``,
        # producing the ``%2F``-separated path segment GitLab's "files"
        # endpoint requires. httpx preserves percent-encoded characters
        # in the URL path, so the encoding survives transport intact.
        encoded = quote(path, safe="")
        file_path = f"projects/{project_id}/repository/files/{encoded}/raw"
        response = self._get(file_path, {"ref": commit_sha})

        if response.status_code in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
            raise GitLabAuthError(response.status_code)
        if response.status_code == _HTTP_NOT_FOUND:
            # File listed in the tree but not retrievable at this commit
            # (rare race / permissions edge case). Drop it from the
            # snapshot rather than failing the whole fetch.
            return None
        response.raise_for_status()

        body = response.content
        if len(body) > MAX_FILE_SIZE_BYTES:
            return None
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            # Binary content: ``RepositoryContents.files`` is text-only,
            # so skipping is the correct behavior.
            return None


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _items_from_response(response: httpx.Response) -> list[JsonObject]:
    """Return the list of JSON objects from a GitLab list-endpoint response.

    GitLab list endpoints return a JSON array of objects. Status-code
    handling is the responsibility of higher-level methods (tasks 5.2 and
    5.3); this helper only deserializes the body so :meth:`_paginate` can
    drive pagination. When the body is anything other than a JSON array,
    the helper returns an empty list, which makes :meth:`_paginate`'s
    ``?page=N`` fallback terminate cleanly.
    """
    try:
        payload = response.json()
    except ValueError:
        return []
    if not isinstance(payload, list):
        return []
    return cast("list[JsonObject]", payload)


__all__ = [
    "API_PREFIX",
    "DEFAULT_BLOB_FETCH_CONCURRENCY",
    "DEFAULT_PER_PAGE",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_ANALYZABLE_FILES",
    "MAX_FILE_SIZE_BYTES",
    "PRIVATE_TOKEN_HEADER",
    "GitLabConnector",
    "JsonObject",
]

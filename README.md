# Project Knowledge MCP Server

An MCP server that builds and serves structured knowledge about projects in a GitLab group.

It runs in a single OS process and exposes two surfaces:

- **MCP over stdio** — eight tools that an MCP client can call to query projects and trigger refreshes.
- **Read-only HTML visualization** on `http://127.0.0.1:{port}` — diagrams of project profiles, shared dependencies, and purpose conflicts.

Both surfaces read from the same `Knowledge_Store` (a single SQLite file with snapshot-based atomic-pointer-swap semantics) so readers always see a consistent view, even while a refresh is in flight.

> **What does the analysis?**
> Static inspection in Python (no LLM). For every project, four sub-analyzers run: **purpose** (READMEs, GitLab description, manifests, top-level docstrings), **I/O extractor** (HTTP routes, scheduled tasks, message consumers/publishers, file I/O, CLI entrypoints), **external services** (HTTP clients, brokers, object stores), and **database tables** (SQL/ORM references, read/write modes). Source code is fetched at the configured `Analysis_Branch` (default `uat`).

## Go analyzer support

Go repositories are analyzed in-process with the same depth as Python, JavaScript/TypeScript, and Java repositories. No Go toolchain is invoked at runtime; a hand-written Python tokenizer reads `.go` source directly. The Go layer activates automatically when a `RepositoryContents` snapshot contains at least one `.go` file or a repo-root `go.mod` — there is no configuration to enable.

### What gets detected

| Sub-analyzer | Go patterns recognized |
|---|---|
| **Purpose** | `module <path>` line in `go.mod` (and any adjacent `//`-comment), package doc comment on repo-root, `cmd/main.go`, or `cmd/<name>/main.go` files. Module-path host/org prefixes (`github.com/acme/`) are stripped so the binary name remains. |
| **I/O extractor** | `http.HandleFunc` / `http.Handle` / `mux.HandleFunc` / `mux.Handle` with Go 1.22 method-prefixed pattern support; `cron.New(...)` + `AddFunc`/`AddJob` (six- or five-field schedules), `time.NewTicker`, `time.AfterFunc`; `*activemq.Receiver.Subscribe(..., &domain.SubscriberConfig{Destination: ...}, ...)` and `*activemq.Sender.SendMessage(..., &domain.Message{Destination: ...})`; `os.Open` / `os.ReadFile` / `os.Create` / `os.WriteFile` / `os.OpenFile` (flag bitmask classified); `flag.*` registrations; `cmd/<name>/main.go` / `cmd/main.go` / root `main.go` binary names. |
| **External services** | `fec_pool_service/pb` clients (`ExecuteQuery`), `esb-go-libs/dbadapter` clients (`PoolExecuteQuery`), and ActiveMQ broker connections via `activemq.NewClient(&activemq.JmsConfig{BrokerUrl: ...})`. Elastic APM imports (`go.elastic.co/apm/...`) are filtered out. |
| **Database tables** | SQL extracted from `QueryString` fields on `model.PoolServiceRequest`, `pb.PoolExecuteQueryRequest`, and in-house wrapper structs whose value flows into `PoolExecuteQuery`/`Execute`/`ExecuteRaw`. Schema-qualified names (`<schema>.<table>`) are preserved. `MERGE INTO` is classified as a write. When a SQL keyword cannot be matched but a table name is identifiable, the access mode is recorded as `unknown`. |

### What gets excluded

- **Vendor directories.** Any `.go` file whose path contains a directory segment named `vendor` is skipped end-to-end. The Go scanners reject vendored paths at the parser boundary, and the language-agnostic detectors are also handed a vendor-aware view so vendored Go source's embedded string literals cannot leak into SQL, URL, or SDK-pattern detections.
- **Build-constraint-gated files.** A file with a non-trivial `//go:build` or `// +build` line (e.g. `linux && amd64`, `darwin`, `!cgo`) is wholesale skipped because honoring the constraint requires a Go toolchain. Skipped files are surfaced in `degraded_sections` as `"<section>: skipped <path> (build constraint requires toolchain)"`.
- **cgo files.** A file containing `import "C"` is skipped for the same reason and surfaced under the same `degraded_sections` shape with reason `cgo directive requires toolchain`.
- **Tokenization errors.** Unterminated strings, raw strings, block comments, and invalid escape sequences are contained per file: the affected file contributes zero detections and is surfaced with reason `tokenization failed: <detail>`.
- **DI and config noise.** `fx.New` / `fx.Provide` / `fx.Invoke` / `fx.Module` / `fx.Hook` / `*fx.Lifecycle` calls describe internal wiring and never produce HTTP, scheduler, ActiveMQ, file-I/O, CLI, or external-service detections by themselves. `viper` configuration reads (`*viper.Viper.GetString`, `SetConfigName`, `AddConfigPath`, etc.) never contribute to the purpose summary or any detection.

### Triggering Go analysis

No flag, no environment variable, no client opt-in. Run the server as documented above against a GitLab group containing Go projects — every refresh that fetches a snapshot with `.go` files or a `go.mod` automatically routes through the Go sub-analyzers. The result is observable through the same MCP tools and visualization routes as any other project:

- `get_project_purpose` returns a summary derived from `go.mod` / package doc when no README is present.
- `get_project_io` lists HTTP routes, cron schedules, ActiveMQ destinations, file I/O, and binary entry points found in the Go source.
- `get_project_dependencies` lists `fec_pool_service` (when reached via either `pb` or `dbadapter`), ActiveMQ brokers, and any URL-literal-derived service, plus the database tables extracted from in-process SQL.

When a Go file is skipped for any reason listed above, the affected section name appears in the profile's `degraded_sections` list with a structured `"<section>: skipped <path> (<reason>)"` entry so the operator can see exactly which files the analyzer could not reach.

## Requirements

- Python **3.11+**
- A GitLab access token with `read_api` and `read_repository` scopes for the target group.

## Install

From this repository:

```bash
pip install -e ".[dev]"
```

The install registers a console script:

```
project-knowledge-mcp = project_knowledge_mcp.main:main
```

## Configuration

Configuration is read from environment variables. On any missing or invalid value the server prints a single line to stderr that names the offending key and exits non-zero before either surface accepts traffic.

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GITLAB_BASE_URL` | yes | — | e.g. `https://gitlab.example.com`. Must be `http://` or `https://` with a host. |
| `GITLAB_GROUP_PATH` | yes | — | Top-level group path (e.g. `acme/platform`) or numeric group id. |
| `GITLAB_ACCESS_TOKEN` | yes | — | Token with `read_api` + `read_repository`. |
| `GITLAB_VERIFY_SSL` | no | `true` | Whether outbound HTTPS calls to `GITLAB_BASE_URL` validate the server's TLS certificate. Set to `false` (or `0`/`no`/`off`) for a GitLab instance with a self-signed certificate; doing so removes MITM protection for those requests. |
| `ANALYSIS_BRANCH` | no | `uat` | Branch fetched per project, regardless of GitLab default branch. |
| `REFRESH_INTERVAL` | no | _none_ | When set (e.g. `15m`, `1h`), schedules a periodic full refresh and fires one immediately at startup so the visualization populates without waiting a full interval. Minimum 1 minute. When unset, no scheduler runs and no automatic refresh fires — trigger ingestion via the `refresh_all_projects` MCP tool. |
| `VISUALIZATION_PORT` | no | `7345` | Loopback-only TCP port (`127.0.0.1` and `::1`). Range `1..65535`. |
| `KNOWLEDGE_STORE_PATH` | no | `data/knowledge_store.db` | SQLite database path. The parent directory is created if missing. |
| `LOG_LEVEL` | no | `WARNING` | Root logging level for stderr output. The application's operationally important lines (visualization "ready" banner, refresh-progress per-project lines, refresh-complete summary) are emitted at `WARNING` so they surface under this default without dragging in third-party libraries' per-request chatter. Set `LOG_LEVEL=INFO` to additionally see less essential application INFO lines, or `LOG_LEVEL=DEBUG` for full diagnostic verbosity (including httpx/httpcore/uvicorn). Unknown values fall back to `WARNING`. |

The store persists across restarts: on reopen, readers immediately serve the last successfully committed snapshot, and an in-flight job aborted at shutdown does not affect that pointer.

## Run

```bash
GITLAB_BASE_URL=https://gitlab.example.com \
GITLAB_GROUP_PATH=acme/platform \
GITLAB_ACCESS_TOKEN=glpat-... \
project-knowledge-mcp
```

On startup you will see a single log line:

```
Visualization_Server ready at http://127.0.0.1:7345
```

The MCP server simultaneously speaks JSON-RPC over **stdin/stdout**, so this command is normally launched by an MCP client rather than run interactively.

### Use from an MCP client

Configure your MCP-aware tool (Claude Desktop, an editor MCP integration, etc.) to spawn the server. Example config:

```json
{
  "mcpServers": {
    "project-knowledge": {
      "command": "project-knowledge-mcp",
      "env": {
        "GITLAB_BASE_URL": "https://gitlab.example.com",
        "GITLAB_GROUP_PATH": "acme/platform",
        "GITLAB_ACCESS_TOKEN": "glpat-...",
        "ANALYSIS_BRANCH": "uat",
        "REFRESH_INTERVAL": "1h"
      }
    }
  }
}
```

### Available MCP tools

Eight tools are registered. The first six are read-only; the last two trigger ingestion.

| Tool | Arguments | Purpose |
|---|---|---|
| `list_projects` | — | List in-scope projects from the current `Project_Catalog`. |
| `get_project_purpose` | `gitlab_project_id` | Purpose summary + reason. |
| `get_project_io` | `gitlab_project_id` | Abstract inputs and outputs. |
| `get_project_dependencies` | `gitlab_project_id` | External services + database tables. |
| `get_project_profile` | `gitlab_project_id` | The complete `Project_Profile`. |
| `list_purpose_conflicts` | — | Pairs of projects with overlapping or contradictory primary responsibilities. |
| `refresh_all_projects` | — | Start a full refresh of every in-scope project. |
| `refresh_project` | `gitlab_project_id` | Re-analyze a single project, copying every other profile from the parent snapshot. |

Refresh tools enforce a single-flight rule: while one ingestion job is running, additional refresh requests (from any source — tool call or scheduler) are rejected with the message `Ingestion_Job is already in progress`. The `Knowledge_Store` is unchanged.

Out-of-scope project ids return a tool result with `isError: true` and the message `project {id} is not in scope`.

### Visualization

Browse `http://127.0.0.1:7345/` (or your configured port). All routes serve `text/html; charset=utf-8`:

| Route | Shows |
|---|---|
| `/` | Index of in-scope projects, sorted by id. Empty catalog → `No Projects are in scope`. No completed ingestion yet → `no project knowledge available; run an Ingestion_Job`. |
| `/projects/{project_id}` | Single project: purpose summary; abstract inputs grouped by category; abstract outputs grouped by category; external services labeled by kind; database tables labeled by access mode. Each empty section shows a section-specific empty-state message. |
| `/dependencies` | Mermaid graph: nodes = in-scope projects, edges = shared database tables. |
| `/static/<filename>` | Static file mount serving the package's `static/` directory. Used by `/dependencies` to load `cytoscape.min.js` (primary) and `mermaid.min.js` (fallback). |

#### Enable graph rendering

The `/dependencies` page renders an interactive knowledge graph using [Cytoscape.js](https://js.cytoscape.org/). Drop the bundle at `src/project_knowledge_mcp/static/cytoscape.min.js`:

```bash
curl -L -o src/project_knowledge_mcp/static/cytoscape.min.js \
  https://cdn.jsdelivr.net/npm/cytoscape@3/dist/cytoscape.min.js
```

When Cytoscape is loaded the page emits an interactive node-and-edge graph with zoom, pan, drag, and hover tooltips. The same diagram is also emitted as a Mermaid `<pre class="mermaid">` block; when `cytoscape.min.js` is absent the page falls back to Mermaid by dropping a [Mermaid](https://mermaid.js.org/) bundle at `src/project_knowledge_mcp/static/mermaid.min.js`:

```bash
curl -L -o src/project_knowledge_mcp/static/mermaid.min.js \
  https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js
```

For air-gapped networks, fetch either file from any machine that can reach the CDN and copy it into the same path. See `src/project_knowledge_mcp/static/README.md` for details. When both files are missing the pages still load — graphs degrade to the raw `graph LR ...` text.

Behavior pinned by the spec:
- Reads happen at request time — no in-memory caching of profiles.
- Non-matching paths → 404 with the requested path echoed in the body.
- Non-`GET` methods → 405 with `Allow: GET`.
- `Knowledge_Store` failures → 503 with no profile-derived content.
- Every successful response begins within 5 seconds.

## Typical workflow

1. Start the server with `REFRESH_INTERVAL=1h` (or trigger refreshes manually).
2. After the first ingestion completes, the visualization populates and the MCP query tools start returning data.
3. Iterate: as code in your GitLab projects evolves on `Analysis_Branch`, each refresh produces a new snapshot. Readers atomically switch to the new snapshot only when the ingestion commits.

## Operations

- **Skipped projects.** When a project has no `Analysis_Branch`, the ingestion records a skip with reason `analysis_branch_missing` and continues with the rest. No `Project_Profile` is produced for skipped projects; they still appear in the catalog so you can tell "in scope but not yet analyzed" from "out of scope".
- **Auth failures.** A `401`/`403` from GitLab aborts the in-flight job and surfaces the status code; the previous snapshot remains current. A `404` for the configured group raises `GitLabGroupNotFoundError(group_path)`.
- **Shutdown.** SIGTERM/SIGINT triggers a documented shutdown ordering: stop accepting new HTTP connections → close MCP stdio → mark any in-flight snapshot `failed` (the `current_snapshot` pointer is unchanged) → flush and close the SQLite store → exit.
- **Port already in use.** Startup prints `startup error: visualization.port {port} is already in use` and exits non-zero.

## Tests

```bash
pytest -m unit         # ~3s
pytest -m property     # Hypothesis, max_examples=100, ~30s
pytest -m integration  # subprocess + HTTP + SQLite, ~15s
```

Or all of them:

```bash
pytest
```

## Project layout

```
src/project_knowledge_mcp/
  config.py              # env loader + validator (single termination path)
  errors.py              # error type hierarchy
  models.py              # Project_Profile, EnumeratedProject, ConflictResult, ...
  knowledge_store.py     # SQLite + snapshot pointer + WAL
  project_catalog.py     # snapshot-scoped in-scope set
  gitlab_connector.py    # paginated GitLab API client
  project_analyzer/      # purpose, io_extractor, external_services, db_tables
  conflict_detector.py   # classify_pair + find_all_conflicts
  ingestion_coordinator.py  # single-flight state machine
  scheduler.py           # periodic refresh
  mcp_server.py          # MCP over stdio + 8 tools
  visualization_server.py   # Starlette + Uvicorn (loopback only)
  diagram_renderer.py    # Jinja2 + inline Mermaid
  main.py                # wiring, startup, shutdown ordering
```

The full specification (requirements, design, and the 30 correctness properties the implementation is verified against) lives under `.kiro/specs/project-knowledge-mcp/`.

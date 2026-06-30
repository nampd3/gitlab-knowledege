# Implementation Plan: Go Analyzer Support

## Overview

This plan delivers Go-aware behavior in the existing `Project_Analyzer` by adding a new `project_analyzer/go/` sub-package (tokenizer, recognizer, four sub-scanners) and layering its detections into the four `_safe_*` aggregator helpers. The plan begins with the parent-spec edits required to land `DatabaseAccessMode.UNKNOWN` and the schema-preserving SQL helper, then builds the tokenizer/recognizer foundation, then implements each Go sub-analyzer with its property test, then wires everything into the aggregator, and finally locks behavior with golden tests against the four sample repositories.

Implementation language is Python 3.11+ throughout, matching the parent spec. No Go toolchain is invoked at runtime.

## Tasks

- [x] 1. Land parent-spec edits required for unknown access mode
  - [x] 1.1 Add `UNKNOWN = "unknown"` to `DatabaseAccessMode` enum
    - Edit `src/project_knowledge_mcp/models.py`
    - Update enum docstring to describe the unknown case
    - _Requirements: 9.8_

  - [x] 1.2 Update `db_tables.py`: UNKNOWN coalescing, schema-preserving extractor, MERGE regex
    - In `src/project_knowledge_mcp/project_analyzer/db_tables.py`, modify `_aggregate` so `UNKNOWN` is treated as the lowest-priority observation: if `READ`, `WRITE`, or `READ_WRITE` is also observed for the same table, that takes precedence; only when `UNKNOWN` is the sole observation does the entry's `access_mode` become `UNKNOWN`
    - Add helper `extract_table_references_preserving_schema(sql_text)` that runs the existing regex set but returns the raw match group, preserving `<schema>.<table>` form
    - Add `_RE_MERGE = re.compile(r"\bMERGE\s+INTO\s+<TABLE_TOKEN>", re.IGNORECASE)` and route `MERGE` keyword to `WRITE` in the existing extractor
    - _Requirements: 9.5, 9.6, 9.8_

  - [x] 1.3 Update `diagram_renderer.py` to label `unknown` access mode
    - Edit `src/project_knowledge_mcp/diagram_renderer.py` so `Dependency_Graph_Diagram` and `Project_Profile_Diagram` render `unknown` access mode explicitly (e.g. `[access: unknown]`)
    - _Requirements: 9.8_

  - [x] 1.4 Update existing parent-spec property tests for the new enum value
    - Update Hypothesis strategies that draw `DatabaseAccessMode` values in `tests/property/test_property_06_project_profile_shape.py`, `tests/property/test_property_08_db_tables_mixed_mode.py`, and any `tests/unit/test_db_tables_detector.py` strategies to include `UNKNOWN`
    - **Validates: Requirements 9.8**

- [x] 2. Implement Go file filter and repo-level guard
  - [x] 2.1 Create `is_go_source_file` and `has_go_artefacts` predicates
    - Create `src/project_knowledge_mcp/project_analyzer/go/__init__.py` (re-exports placeholder)
    - Create `src/project_knowledge_mcp/project_analyzer/go/go_filter.py` with `is_go_source_file(path: str) -> bool` (rejects any path containing a directory segment equal to `vendor` after normalizing `\\` to `/`, requires `.go` suffix) and `has_go_artefacts(repository_contents) -> bool` (true when at least one Go source file exists or `go.mod` is at repo root)
    - _Requirements: 1.3, 1.4, 11.5_

  - [x] 2.2 Write unit tests for filter and guard
    - Cover vendor-segment rejection, case-sensitivity of suffix, root vs nested `go.mod`, no-Go repos
    - _Requirements: 1.3, 1.4, 11.5_

- [x] 3. Implement Go tokenizer
  - [x] 3.1 Define `GoToken` dataclass and `GoTokenKind` StrEnum
    - Create `src/project_knowledge_mcp/project_analyzer/go/_events.py` with `GoTokenKind` (full enum from design §4) and `GoToken(frozen=True, slots=True)` carrying `kind`, `text`, `line` (1-indexed), `column` (1-indexed)
    - _Requirements: 10.1, 10.2_

  - [x] 3.2 Implement `tokenize_go_source(text)` in `go_tokenizer.py`
    - Create `src/project_knowledge_mcp/project_analyzer/go/go_tokenizer.py`
    - Emit all token kinds enumerated in design §4 (keywords, identifiers, string and raw string literals, number literals, punctuation, comments as first-class tokens, struct tags as backtick-quoted runs that immediately follow a struct field declaration, build-constraint comments for `//go:build` and `// +build`, cgo pragma comments for `// #cgo`, newlines, whitespace)
    - Track 1-indexed line and rune-column on every token
    - Pure function with no side effects
    - _Requirements: 10.1, 10.2, 10.4, 11.4_

  - [x] 3.3 Implement `GoTokenizationError` and tokenizer error paths
    - In `go_tokenizer.py`, raise `GoTokenizationError(line, column, reason)` for unterminated string literal, unterminated raw string literal, unterminated block comment, and invalid escape sequence inside string literals (per design §3 "Error handling" enumeration)
    - _Requirements: 11.1_

  - [x] 3.4 Write unit tests for tokenizer happy paths and error cases
    - Cover keywords, identifiers, regular and raw string literals, struct tag detection, comment kinds (line, block, build-constraint, cgo pragma), and every `GoTokenizationError` reason
    - _Requirements: 10.1, 10.4, 11.1_

- [x] 4. Implement Go event types and construct recognizer
  - [x] 4.1 Define event dataclasses and `ArgRef` tagged union in `_events.py`
    - Append to `src/project_knowledge_mcp/project_analyzer/go/_events.py`: `ImportEvent`, `FuncDeclEvent`, `MethodCallEvent`, `StructLitEvent`, `PackageDocCommentEvent`, `ModFileModuleEvent`, `BuildConstraintEvent`, `CgoDirectiveEvent`, `SkipFileEvent`, and the `ArgRef` union (`StringLitArg`, `NumberLitArg`, `IdentArg`, `DottedArg`, `StructLitArg`, `CallArg`, `UnknownArg`) per design §4
    - Also add internal helper records `GoPurposeCandidates`, `RouteRegistration`, `SchedulerRegistration`, `ActiveMQCall`, `PoolServiceCall`
    - _Requirements: 10.1_

  - [x] 4.2 Implement `recognize_constructs(tokens, path)` in `go_parser.py`
    - Create `src/project_knowledge_mcp/project_analyzer/go/go_parser.py`
    - Recognize: import declarations (single-line and parenthesized blocks), package declarations with their preceding doc-comment block (no blank line gap), function declarations (`FuncDeclEvent` with `receiver_type` for methods), method calls with dotted `receiver_chain` and positional `args`, composite literals (`T{...}`, `&T{...}`, `*T{...}`, `pkg.T{...}`) with named-field syntax
    - Detect non-trivial build constraints (`//go:build` or `// +build` whose expression is non-empty and not `!ignore`) and emit `SkipFileEvent("build constraint requires toolchain", line)` as the file's only event
    - Detect cgo (`import "C"`) and emit `SkipFileEvent("cgo directive requires toolchain", line)` as the file's only event
    - Wrap each top-level construct in try/except so a single bad construct yields a `SkipFileEvent` and the recognizer continues at the next package-level boundary
    - _Requirements: 1.1, 1.2, 10.4, 11.1_

  - [x] 4.3 Implement `go.mod` recognizer producing `ModFileModuleEvent`
    - In `go_parser.py`, expose `parse_go_mod(text) -> ModFileModuleEvent | None` that finds the `module <module-path>` line and captures any `//`-comment immediately preceding the line (no blank-line gap) as `leading_comment` and any same-line trailing `//`-comment as `trailing_comment`, both with `//` and surrounding whitespace stripped
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 4.4 Implement `parse_repo(repository_contents)` entry point
    - In `go_parser.py`, expose `parse_repo(repository_contents) -> Mapping[str, list[GoEvent]]`: filters file paths through `is_go_source_file`, iterates in sorted order, tokenizes and recognizes per file, catches `GoTokenizationError` and yields a single `SkipFileEvent("tokenization failed: <detail>", line)` for that file, includes `go.mod` events under the path key `"go.mod"` when present
    - _Requirements: 1.1, 1.3, 11.1, 11.4_

  - [x] 4.5 Write property test for build-constraint, cgo, and tokenization-error skip behavior
    - **Property 13: Build constraints and cgo files are skipped, recorded in `degraded_sections`, and do not affect the rest of the repo**
    - **Validates: Requirements 10.4, 11.1, 11.2**

  - [x] 4.6 Write unit tests for recognizer event-emission shapes
    - Cover one fixture per event type, plus mixed-fixture event ordering
    - _Requirements: 1.1, 1.2_

- [x] 5. Checkpoint - tokenizer and recognizer foundation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Go purpose summarizer and refactor `purpose.py`
  - [x] 6.1 Refactor `purpose.py` to expose `collect_purpose_candidates()` helper
    - Edit `src/project_knowledge_mcp/project_analyzer/purpose.py`: extract the existing per-source candidate logic into a public `collect_purpose_candidates(repository_contents) -> list[PurposeCandidate]` helper; have `summarize_purpose(...)` delegate to that helper plus the existing prose-selection logic
    - Behavior for non-Go inputs MUST remain byte-identical
    - _Requirements: 2.6_

  - [x] 6.2 Implement `collect_go_candidates` in `go/go_purpose.py`
    - Create `src/project_knowledge_mcp/project_analyzer/go/go_purpose.py` exposing `collect_go_candidates(repository_contents, events_by_file) -> GoPurposeCandidates` with three optional fields: `gomod_comment`, `gomod_module_path` (with leading `<host>/<org>/` prefix stripped; bare module names passed through unchanged), `package_doc_comment` (taken from the first non-empty `PackageDocCommentEvent` for files at repo root, `cmd/main.go`, or `cmd/<name>/main.go` in sorted-path order)
    - Apply existing `purpose._truncate` and `purpose._normalize_description` from the parent spec
    - viper string literals (Requirement 2.7) are never sourced — only `go.mod` and `PackageDocCommentEvent` are read
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7_

  - [x] 6.3 Write property test for purpose summary priority order
    - **Property 4: Purpose summary follows the documented priority order**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

- [x] 7. Implement Go I/O extractor
  - [x] 7.1 Implement HTTP route registration recognizer
    - Create `src/project_knowledge_mcp/project_analyzer/go/go_io.py`
    - Match `MethodCallEvent` with method `HandleFunc` or `Handle` whose receiver is `["http"]` or any single identifier in a file importing `net/http`
    - Parse the first positional argument with the Go 1.22 method-prefixed pattern `^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) (.+)$`; on no match record method `"ANY"` and the entire literal as path
    - For non-string-literal patterns produce a placeholder description naming file path and receiver identifier
    - Emit one `AbstractInput(category=http_request)` and one `AbstractOutput(category=http_response)` per registration with Source_Location carrying the call line
    - Suppress bootstrap calls (`ListenAndServe`, `ListenAndServeTLS`, `Serve`, `Shutdown`, `Close` on `http` package or `*http.Server` receiver) silently — no input/output, no log, no error
    - Skip any `MethodCallEvent` whose receiver chain begins with `fx` (per design "fx exclusion") or `viper` / a `*viper.Viper`-typed identifier (per design "viper exclusion")
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 12.1, 12.3, 13.1, 13.2_

  - [x] 7.2 Write property test for HTTP detection
    - **Property 5: HTTP detection is exactly the non-bootstrap registration set, with method/path correctly split**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**

  - [x] 7.3 Implement scheduler registration recognizer
    - In `go_io.py`, track an in-file map `id -> "*cron.Cron"` from `<v> := cron.New(...)` assignments, including whether the `cron.New(...)` argument list contains a `cron.WithSeconds()` call (six-field marker) or not (five-field marker)
    - Match `<v>.AddFunc(<schedule>, h)` and `<v>.AddJob(<schedule>, h)` against the map
    - Match `time.NewTicker(<d>)` and `time.AfterFunc(<d>, h)` from the standard library
    - Record literal schedule strings verbatim (including six-field strings a five-field parser would reject) and `<dynamic>` for non-literal expressions
    - For any `MethodCallEvent` whose method is `AddFunc` or `AddJob` but whose receiver was not recognized as `*cron.Cron`, or whose schedule argument is malformed, still emit one `AbstractInput(category=scheduled_event)` with a malformed/unsupported marker
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 7.4 Write property test for scheduler detection
    - **Property 6: Scheduler detection emits one input per recognized registration, preserving schedule literals verbatim**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

  - [x] 7.5 Implement ActiveMQ consumer/publisher recognizer
    - In `go_io.py`, match `Subscribe` calls whose third positional argument is a `StructLitEvent` of type `domain.SubscriberConfig` (with package alias `domain`) → `AbstractInput(category=message_consumed, library="activemq", destination=<literal-or-dynamic>)`
    - Match `SendMessage` calls whose fourth positional argument is a `StructLitEvent` of type `domain.Message` → `AbstractOutput(category=message_published, library="activemq", destination=<literal-or-dynamic>)`
    - String-literal `Destination` field values are recorded verbatim; any other expression is recorded as `<dynamic>`
    - Recognize but do **not** emit I/O for `activemq.NewClient(&activemq.JmsConfig{...})` calls (they feed the external services detector instead)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 7.6 Write property test for ActiveMQ detection
    - **Property 7: ActiveMQ consumer/publisher emit exactly one I/O entry per call site, preserving literal destinations**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

  - [x] 7.7 Implement file I/O recognizer
    - In `go_io.py`, match `os.Open` and `os.ReadFile`, `ioutil.ReadFile` → `AbstractInput(category=file_read)`
    - Match `os.Create`, `os.WriteFile`, `ioutil.WriteFile` → `AbstractOutput(category=file_written)`
    - For `os.OpenFile(<path>, <flag-expr>, <mode>)`, parse `<flag-expr>` for `os.O_*` atoms: `O_RDONLY` only → input; any of `O_WRONLY|O_RDWR|O_APPEND|O_CREATE|O_TRUNC` → output; expression not statically determinable → emit both an input and an output
    - Record the literal path argument when string-literal, otherwise `<dynamic>`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 7.8 Write property test for file I/O classification
    - **Property 8: File I/O classification follows the documented `O_*` flag mapping, including the undecidable-flag case**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

  - [x] 7.9 Implement CLI entry-point recognizer
    - In `go_io.py`, emit `AbstractInput(category=cli_argument, description="binary <name>")` for `cmd/<name>/main.go` files containing `func main()`
    - Emit `AbstractInput(category=cli_argument)` using the last segment of the `go.mod` module path (after stripping `<host>/<org>/`) for `cmd/main.go` and root-level `main.go` when `go.mod` is present
    - Match `flag.{String,StringVar,Int,IntVar,Bool,BoolVar,Float64,Float64Var,Duration,DurationVar,Parse,NewFlagSet}` calls and emit one `AbstractInput(category=cli_argument)` each, including the flag name when literal
    - Skip every `MethodCallEvent` whose receiver chain starts with `fx` and every `Append` call on an `fx.Lifecycle`-typed identifier
    - Reject (do not emit) any CLI input whose Source_Location cannot be determined (line is `None`)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 12.1, 12.3_

  - [x] 7.10 Write property test for CLI entry-point detection and fx exclusion
    - **Property 9: CLI entry-point detection follows the documented `cmd/`/root rules and excludes fx wiring**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**

  - [x] 7.11 Wire `extract_go_io(repository_contents, events_by_file)` entry point
    - Compose the five recognizers into a single function returning `(inputs, outputs, file_skip_messages)`
    - Iterate `events_by_file` in path-sorted order
    - Deduplicate by `(category, description)` per the existing `io_extractor._Accumulator` rule
    - Convert `SkipFileEvent` instances into `file_skip_messages` strings of the form `"skipped <path> (<reason>)"`
    - _Requirements: 1.1, 1.2, 3.7, 11.4_

- [x] 8. Implement Go external services detector
  - [x] 8.1 Implement fec_pool_service detection from `pb` and `dbadapter` imports
    - Create `src/project_knowledge_mcp/project_analyzer/go/go_external_services.py` exposing `detect_go_external_services(repository_contents, events_by_file) -> tuple[list[ExternalServiceDependency], list[str]]`
    - For each file with `ImportEvent(path="fec_pool_service/pb")`, scan `MethodCallEvent`s with method `ExecuteQuery`; track `<v> := pb.NewPoolAPIClient(...)` to resolve receiver type. When unresolvable, fall back to the import line itself as evidence
    - For each file with `ImportEvent(path="esb-go-libs/dbadapter")`, scan `MethodCallEvent`s with method `PoolExecuteQuery`
    - Emit `ExternalServiceDependency(name="fec_pool_service", kind=other, source_locations=[(path, line)])` per match
    - _Requirements: 8.1, 8.2, 8.3, 8.7_

  - [x] 8.2 Implement ActiveMQ broker detection from `activemq.NewClient`
    - In `go_external_services.py`, match `MethodCallEvent` with receiver chain `["activemq"]` and method `NewClient` whose first argument is a `StructLitEvent` of type `activemq.JmsConfig`
    - Extract `BrokerUrl` field; when string-literal, parse host portion (`tcp://host:port` → `host:port`); otherwise record `<dynamic>`
    - Emit `ExternalServiceDependency(name="activemq", kind=message_broker, source_locations=[(path, line)])` with the host appended to the source-location auxiliary text
    - _Requirements: 8.5, 8.7_

  - [x] 8.3 Implement APM exclusion guard
    - In `go_external_services.py`, skip any file whose only "interesting" imports are paths beginning with `go.elastic.co/apm/`
    - No External_Service_Dependency is ever emitted for APM-related imports or calls
    - _Requirements: 8.4_

  - [x] 8.4 Aggregate detection sites by service name
    - Within the Go detector's own output, coalesce multiple detections of the same `name` into a single entry whose `source_locations` is the union of all sites, deduplicated by `(path, line)`
    - _Requirements: 8.6_

  - [x] 8.5 Write property test for external service detection
    - **Property 10: External service detection coalesces fec_pool_service across import paths and excludes APM**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 11.3**

- [x] 9. Implement Go database tables detector
  - [x] 9.1 Implement composite-literal recognition for the three request types
    - Create `src/project_knowledge_mcp/project_analyzer/go/go_db_tables.py` exposing `detect_go_database_tables(repository_contents, events_by_file) -> tuple[list[DatabaseTableDependency], list[str]]`
    - Recognize `StructLitEvent` of type `model.PoolServiceRequest` (from `esb-go-libs/dbadapter/model`), `pb.PoolExecuteQueryRequest` (from `fec_pool_service/pb`), and any in-house wrapper struct literal whose `fields` list contains a `QueryString` entry and whose value flows positionally into a `MethodCallEvent` with method in `{PoolExecuteQuery, Execute, ExecuteRaw}`
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 9.2 Implement SQL extraction from `QueryString` field values
    - In `go_db_tables.py`, support: string literal verbatim; raw string literal with backticks stripped; `fmt.Sprintf(<format-literal>, ...)` calls where the format text is used as the SQL with `%v`/`%s`/`%d` left in place
    - Skip extraction (no detection emitted) for any other expression
    - _Requirements: 9.2, 9.3, 9.4_

  - [x] 9.3 Wire SQL extraction to schema-preserving regex extractor and emit detections
    - Call `db_tables.extract_table_references_preserving_schema(sql_text)` from task 1.2 to obtain `(table_name, access_mode)` pairs preserving `<schema>.<table>` exactly
    - Emit `DatabaseTableDependency(name=table_name, access_mode=..., source_locations=[(path, line)])` per match
    - When a table name is identifiable but the SQL keyword cannot be matched, emit with `access_mode=DatabaseAccessMode.UNKNOWN`
    - _Requirements: 9.5, 9.6, 9.7, 9.8_

  - [x] 9.4 Write property test for database table extraction
    - **Property 11: Database table extraction preserves schema, classifies access mode, and coalesces order-independently**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8**

- [x] 10. Checkpoint - all Go sub-analyzers individually pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Integrate Go scanners into the existing aggregator
  - [x] 11.1 Compute `events_by_file` once at top of `analyze()`
    - Edit `src/project_knowledge_mcp/project_analyzer/__init__.py`
    - When `has_go_artefacts(rc)` is true, call `parse_repo(rc)` once and pass the resulting `Mapping[str, list[GoEvent]]` to each `_safe_*` helper as a positional argument
    - When false, pass an empty mapping so every Go branch is a no-op (Requirement 11.5 by construction)
    - _Requirements: 1.1, 1.4, 11.4, 11.5_

  - [x] 11.2 Layer Go I/O detections into `_safe_io`
    - Inside the existing `try` block, after the language-agnostic `extract_io(rc)` call, run `extract_go_io(rc, events_by_file)` when `has_go_artefacts(rc)` is true
    - Concatenate inputs and outputs and route through the existing dedup function so cross-language coalescing is automatic
    - Append per-file skip messages from `extract_go_io` to `degraded_sections` as structured strings of the form `"abstract_io: skipped <path> (<reason>)"`
    - Preserve the existing outer `try/except Exception` and the `"abstract_io"` section name on uncaught failure
    - _Requirements: 1.1, 11.2, 11.3, 11.4_

  - [x] 11.3 Layer Go external service detections into `_safe_external_services`
    - Same pattern as 11.2 against `detect_go_external_services` and the `_aggregate` function in `external_services.py`
    - Structured skip-string section name is `"external_services"`
    - _Requirements: 8.1, 8.6, 11.2, 11.3_

  - [x] 11.4 Layer Go database table detections into `_safe_database_tables`
    - Same pattern as 11.2 against `detect_go_database_tables` and the `_aggregate` function in `db_tables.py`
    - Structured skip-string section name is `"database_tables"`
    - The aggregator's existing read+write→read_write coalescing handles cross-language merging automatically (verified by inspection)
    - _Requirements: 9.1, 9.6, 11.2, 11.3_

  - [x] 11.5 Wire Go purpose candidates into `_safe_purpose` at documented priority position
    - Call `collect_go_candidates(rc, events_by_file)` and interleave the three Go candidates into the existing `collect_purpose_candidates(rc)` list at the priority positions documented in design §3 `go.go_purpose`: README → GitLab description → `gomod_comment` → `gomod_module_path` → root manifest description → top-level Python/JS docstring → `package_doc_comment`
    - Apply existing length cap, whitespace normalization, and unknown fallback unchanged
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 11.2_

  - [x] 11.6 Surface SkipFileEvent reasons across all four sections in `degraded_sections`
    - Confirm that each `_safe_*` helper appends structured strings of the form `"<section>: skipped <path> (<reason>)"` for every file-level skip (tokenization error, build constraint, cgo) returned by its Go scanner
    - Confirm that downstream consumers reading `degraded_sections` see a heterogeneous `list[str]` (plain section names plus structured strings) without breaking the existing `list[str]` field shape
    - _Requirements: 10.4, 11.1, 11.2_

  - [x] 11.7 Write property test for vendor exclusion
    - **Property 1: Vendor directories never contribute detections**
    - **Validates: Requirements 1.3**

  - [x] 11.8 Write property test for analyzer determinism
    - **Property 2: `analyze()` is deterministic**
    - **Validates: Requirements 11.4**

  - [x] 11.9 Write property test for no-Go regression
    - **Property 3: No-Go repositories produce the pre-feature output**
    - **Validates: Requirements 1.4, 11.5**

  - [x] 11.10 Write property test for fx and viper neutrality
    - **Property 12: fx and viper calls produce no detections by themselves; nested fx.Invoke functions still scan**
    - **Validates: Requirements 12.1, 12.2, 12.3, 13.1, 13.2, 13.3**

- [x] 12. Build integration golden tests against the four sample repositories
  - [x] 12.1 Create snapshot script for sample repositories
    - Create `tests/integration/golden/go/_snapshot.py` that walks a target repository, excludes `vendor/` and other non-source directories, and serializes every text file into a `RepositoryContents` JSON payload at `tests/integration/golden/go/<repo-name>/repository_contents.json`
    - _Requirements: 1.1, 1.3_

  - [x] 12.2 Snapshot the four sample repositories
    - Run the script against `/root/fec_pool_service`, `/root/repayment_service`, `/root/cat-service`, `/root/aps_los_vtiger`
    - Commit the four `repository_contents.json` files
    - _Requirements: 8.2, 8.3, 8.5, 9.2, 9.3, 9.4, 9.5_

  - [x] 12.3 Curate expected `Project_Profile` golden JSONs
    - For each snapshot, run `analyze()` once, hand-curate the produced profile for correctness against the requirements, normalize `produced_at` to a fixed timestamp, and commit `tests/integration/golden/go/<repo-name>/expected_profile.json`
    - _Requirements: 8.2, 8.3, 8.5, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 12.4 Write integration test asserting golden equality
    - Create `tests/integration/test_go_analyzer_golden.py` parameterized over the four repositories
    - Load the snapshot, run `analyze()`, normalize `produced_at`, and assert JSON deep-equal against the curated golden profile
    - _Requirements: 1.1, 8.2, 8.3, 8.5, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 12.5 Write no-toolchain assertion test
    - Create `tests/unit/test_go_analyzer_no_toolchain.py` that monkey-patches `subprocess.run`, `subprocess.Popen`, `os.execvp`, and other process-launching surfaces, runs `analyze()` against each of the four sample-repo snapshots, and asserts no Go subprocess was invoked
    - _Requirements: 10.1, 10.2_

- [x] 13. Final checkpoint - ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP. Core implementation tasks are never marked optional.
- Each task references the specific requirement clauses it implements for traceability.
- Property tests are colocated with the implementation they validate (one property per sub-task) so failures surface at the earliest possible point.
- The aggregator integration is intentionally last so each Go sub-analyzer can be unit-tested and property-tested in isolation before being layered into `analyze()`.
- The four golden integration tests in section 12 exercise Requirements 8.2, 8.3, 8.5, 9.2, 9.3, 9.4, 9.5, and 9.6 against real-world inputs and lock the analyzer's behavior against future regressions.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3", "2.1", "3.1", "6.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "3.2", "4.1"] },
    { "id": 2, "tasks": ["1.4", "3.3", "4.2"] },
    { "id": 3, "tasks": ["3.4", "4.3"] },
    { "id": 4, "tasks": ["4.4", "4.6", "6.2"] },
    { "id": 5, "tasks": ["4.5", "6.3", "7.1", "8.1", "9.1"] },
    { "id": 6, "tasks": ["7.2", "7.3", "8.2", "9.2"] }, 
    { "id": 7, "tasks": ["7.4", "7.5", "8.3", "9.3"] },
    { "id": 8, "tasks": ["7.6", "7.7", "8.4", "9.4"] },
    { "id": 9, "tasks": ["7.8", "7.9", "8.5"] },
    { "id": 10, "tasks": ["7.10", "7.11"] },
    { "id": 11, "tasks": ["11.1"] },
    { "id": 12, "tasks": ["11.2"] },
    { "id": 13, "tasks": ["11.3"] },
    { "id": 14, "tasks": ["11.4"] },
    { "id": 15, "tasks": ["11.5"] },
    { "id": 16, "tasks": ["11.6"] },
    { "id": 17, "tasks": ["11.7", "11.8", "11.9", "11.10", "12.1"] },
    { "id": 18, "tasks": ["12.2"] },
    { "id": 19, "tasks": ["12.3"] },
    { "id": 20, "tasks": ["12.4", "12.5"] }
  ]
}
```

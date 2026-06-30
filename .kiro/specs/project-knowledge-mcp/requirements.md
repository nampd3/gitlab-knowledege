# Requirements Document

## Introduction

The Project Knowledge MCP Server is a Model Context Protocol (MCP) server that builds and exposes structured knowledge about a collection of software projects hosted on a GitLab instance or group. It ingests repositories from a configured GitLab source, analyzes each repository to derive a structured Project Profile (purpose, abstract inputs and outputs, external service dependencies, and database table dependencies), persists those profiles, and serves them to other agents through MCP tools and resources. Other agents use the server to answer questions such as "what is the purpose of project X?", "which projects have overlapping or conflicting purposes?", and "what does project X depend on?".

Alongside the MCP surface for agents, the server also exposes a human-facing visualization surface: a local HTTP server, bound to the loopback interface, that renders the same persisted Project Profiles as browsable diagrams so a human operator can read and understand each Project's purpose, inputs, outputs, dependencies, and any purpose conflicts between Projects.

## Glossary

- **MCP_Server**: The Model Context Protocol server process that exposes tools and resources to client agents over the MCP transport.
- **GitLab_Connector**: The component that authenticates to a GitLab instance and enumerates and fetches repositories from a configured group.
- **GitLab_Source**: The configured GitLab instance URL plus a group path (or top-level group ID) whose descendant projects are in scope.
- **Project**: A single GitLab repository that is in scope of the configured GitLab_Source.
- **Project_Analyzer**: The component that inspects a Project's contents and produces a Project_Profile.
- **Project_Profile**: A structured record for a Project containing: project identifier, purpose summary, abstract inputs, abstract outputs, external service dependencies, and database table dependencies.
- **Abstract_Input**: A generalized description of data or events a Project consumes (for example, "HTTP requests carrying customer orders" rather than a specific JSON schema).
- **Abstract_Output**: A generalized description of data or events a Project produces (for example, "publishes order-confirmed events to a message bus").
- **External_Service_Dependency**: A named service outside the Project's own code that the Project calls at runtime (for example, a payment gateway, an internal HTTP API, a message broker).
- **Database_Table_Dependency**: A named database table that the Project reads from or writes to, with the access mode recorded as read, write, or read-write.
- **Knowledge_Store**: The persistence layer where Project_Profiles and ingestion metadata are stored.
- **Conflict_Detector**: The component that compares Project_Profiles and identifies pairs of Projects whose purposes overlap or conflict.
- **Purpose_Conflict**: A relationship between two Projects whose purpose summaries describe substantially the same responsibility or describe responsibilities that contradict each other (for example, two Projects that both claim ownership of "user authentication").
- **MCP_Client**: An external agent that connects to the MCP_Server using the Model Context Protocol to query project knowledge.
- **Ingestion_Job**: A run of the GitLab_Connector and Project_Analyzer that refreshes Project_Profiles for all in-scope Projects.
- **GitLab_Access_Token**: A credential used by the GitLab_Connector to authenticate to the GitLab instance.
- **Analysis_Branch**: The Git branch name on each in-scope Project from which the GitLab_Connector reads repository contents and the most recent commit SHA used for analysis. The Analysis_Branch is configured globally per MCP_Server instance and applies to every in-scope Project.
- **Loopback_Interface**: The network interface reachable only from the local host (for example, IPv4 address 127.0.0.1 and IPv6 address ::1).
- **Visualization_Server**: A local HTTP server, started by the MCP_Server in the same process and bound to the Loopback_Interface, that serves human-readable HTML pages and diagrams derived from persisted Project_Profiles.
- **Project_Knowledge_Diagram**: Any human-readable HTML rendering produced by the Visualization_Server that visualizes data drawn from Project_Profiles. Project_Profile_Diagram, Dependency_Graph_Diagram, and Conflict_Overview_Diagram are all kinds of Project_Knowledge_Diagram.
- **Project_Profile_Diagram**: A Project_Knowledge_Diagram that renders a single Project_Profile, displaying the Project's purpose summary, Abstract_Inputs grouped by category, Abstract_Outputs grouped by category, External_Service_Dependencies with their service kind, and Database_Table_Dependencies with their access mode.
- **Dependency_Graph_Diagram**: A Project_Knowledge_Diagram that renders all in-scope Projects as nodes whose edges represent shared External_Service_Dependencies and shared Database_Table_Dependencies between Projects, so that potential cross-Project coupling is visible.
- **Conflict_Overview_Diagram**: A Project_Knowledge_Diagram that renders every Project pair classified as a Purpose_Conflict by the Conflict_Detector, with the justification string for each pair shown on the connecting edge.

## Requirements

### Requirement 1: Configure GitLab Source

**User Story:** As an operator, I want to configure the GitLab instance and group the server reads from, so that the server knows which Projects are in scope.

#### Acceptance Criteria

1. THE MCP_Server SHALL accept a configuration value for the GitLab instance base URL.
2. THE MCP_Server SHALL accept a configuration value for the GitLab group path or top-level group ID that scopes which Projects are ingested.
3. THE MCP_Server SHALL accept a GitLab_Access_Token via configuration for use by the GitLab_Connector.
4. IF the GitLab instance base URL or the group path is missing at startup, THEN THE MCP_Server SHALL fail startup, emit an error message naming the missing configuration value, and terminate the MCP_Server process without continuing to run.
5. IF the GitLab_Access_Token is missing at startup, THEN THE MCP_Server SHALL fail startup, emit an error message naming the missing configuration value, and terminate the MCP_Server process without continuing to run.

### Requirement 2: Enumerate Projects from GitLab

**User Story:** As an operator, I want the server to discover all repositories under the configured GitLab group, so that every in-scope Project is analyzed.

#### Acceptance Criteria

1. WHEN an Ingestion_Job runs, THE GitLab_Connector SHALL enumerate every repository that is a descendant of the configured GitLab group, including repositories in subgroups.
2. WHEN enumerating repositories, THE GitLab_Connector SHALL record for each Project the GitLab project ID, the full path, the Analysis_Branch name as configured per Requirement 15, and the most recent commit SHA on the Analysis_Branch.
3. IF the GitLab instance returns an authentication error during enumeration, THEN THE GitLab_Connector SHALL both abort the Ingestion_Job and report an authentication failure including the GitLab response status code, and IF either the abort of the Ingestion_Job or the reporting of the authentication failure does not complete successfully, THEN THE GitLab_Connector SHALL treat the error handling as failed and surface the underlying failure of the abort or the reporting step rather than silently completing only one of the two steps.
4. IF the configured GitLab group is not found, THEN THE GitLab_Connector SHALL abort the Ingestion_Job and report a group-not-found error including the configured group path.
5. WHERE the GitLab API paginates responses, THE GitLab_Connector SHALL retrieve all pages before completing enumeration.

### Requirement 3: Analyze Project Purpose

**User Story:** As an MCP_Client, I want each Project to have a purpose summary, so that I can answer "what is the purpose of this project?".

#### Acceptance Criteria

1. WHEN the Project_Analyzer processes a Project, THE Project_Analyzer SHALL produce a purpose summary string for that Project and store it in the Project_Profile.
2. THE Project_Analyzer SHALL derive the purpose summary from any one or more of the following sources, treating any single source as sufficient: the Project's README files, the GitLab repository description, and source code metadata such as package manifests and module docstrings.
3. IF the Project_Analyzer cannot derive a purpose summary because no README, repository description, or analyzable source-code metadata exists, OR because the available sources contain no content from which a purpose summary can be derived, THEN THE Project_Analyzer SHALL store a purpose summary value of "unknown" in the Project_Profile and record a reason of "insufficient source material".
4. THE purpose summary SHALL be at most 1000 characters long.

### Requirement 4: Analyze Abstract Inputs and Outputs

**User Story:** As an MCP_Client, I want each Project's inputs and outputs described in generalized form, so that I can reason about what the Project consumes and produces without reading the source code.

#### Acceptance Criteria

1. WHEN the Project_Analyzer processes a Project, THE Project_Analyzer SHALL produce a list of Abstract_Inputs and store the list in the Project_Profile.
2. WHEN the Project_Analyzer processes a Project, THE Project_Analyzer SHALL produce a list of Abstract_Outputs and store the list in the Project_Profile.
3. THE Project_Analyzer SHALL record for each Abstract_Input a category drawn from the set {http_request, scheduled_event, message_consumed, file_read, cli_argument, other} and a human-readable description.
4. THE Project_Analyzer SHALL record for each Abstract_Output a category drawn from the set {http_response, message_published, file_written, database_write, external_call, other} and a human-readable description.
5. IF the Project_Analyzer detects no Abstract_Inputs for a Project, THEN THE Project_Analyzer SHALL store an empty list of Abstract_Inputs in the Project_Profile.
6. IF the Project_Analyzer detects no Abstract_Outputs for a Project, THEN THE Project_Analyzer SHALL store an empty list of Abstract_Outputs in the Project_Profile.

### Requirement 5: Analyze External Service Dependencies

**User Story:** As an MCP_Client, I want to know which external services a Project calls, so that I can understand its runtime dependency graph.

#### Acceptance Criteria

1. WHEN the Project_Analyzer processes a Project, THE Project_Analyzer SHALL produce a list of External_Service_Dependencies and store the list in the Project_Profile.
2. THE Project_Analyzer SHALL record for each External_Service_Dependency a service name, a service kind drawn from the set {http_api, message_broker, object_store, cache, auth_provider, other}, and the source location in the repository where the dependency was detected.
3. WHERE the same external service is referenced from multiple source locations in a Project, THE Project_Analyzer SHALL produce a single External_Service_Dependency entry for that service and list all detected source locations on that entry.
4. IF the Project_Analyzer detects no External_Service_Dependencies for a Project, THEN THE Project_Analyzer SHALL store an empty list of External_Service_Dependencies in the Project_Profile.

### Requirement 6: Analyze Database Table Dependencies

**User Story:** As an MCP_Client, I want to know which database tables a Project reads or writes, so that I can identify Projects that share data.

#### Acceptance Criteria

1. WHEN the Project_Analyzer processes a Project, THE Project_Analyzer SHALL produce a list of Database_Table_Dependencies and store the list in the Project_Profile.
2. THE Project_Analyzer SHALL record for each Database_Table_Dependency the table name, an access mode drawn from the set {read, write, read_write}, and the source location in the repository where the dependency was detected.
3. WHERE the same table is accessed from multiple source locations in a Project with different access modes, THE Project_Analyzer SHALL record the access mode as read_write on the Database_Table_Dependency entry.
4. IF the Project_Analyzer detects no Database_Table_Dependencies for a Project, THEN THE Project_Analyzer SHALL store an empty list of Database_Table_Dependencies in the Project_Profile.

### Requirement 7: Persist Project Profiles

**User Story:** As an operator, I want Project_Profiles to be persisted across server restarts, so that the server can serve queries without re-running ingestion.

#### Acceptance Criteria

1. WHEN the Project_Analyzer completes a Project_Profile, THE Knowledge_Store SHALL persist the Project_Profile keyed by the GitLab project ID.
2. WHEN the MCP_Server starts and a previously persisted Project_Profile exists for a Project, THE MCP_Server SHALL load that Project_Profile from the Knowledge_Store before serving queries about that Project.
3. WHEN the Project_Analyzer produces a new Project_Profile for a Project that already has a persisted Project_Profile, THE Knowledge_Store SHALL replace the existing Project_Profile with the new one.
4. THE Knowledge_Store SHALL record for each persisted Project_Profile the timestamp at which the Project_Profile was produced and the commit SHA the Project_Profile was derived from (the commit SHA on the Analysis_Branch as defined in Requirement 15).

### Requirement 8: Refresh Project Knowledge

**User Story:** As an operator, I want to refresh project knowledge on demand and on a schedule, so that Project_Profiles stay current as repositories change.

#### Acceptance Criteria

1. THE MCP_Server SHALL expose an MCP tool that triggers an Ingestion_Job for all in-scope Projects.
2. THE MCP_Server SHALL expose an MCP tool that triggers an Ingestion_Job for a single Project identified by its GitLab project ID.
3. WHERE a refresh interval is configured, THE MCP_Server SHALL start a new Ingestion_Job for all in-scope Projects every time the configured refresh interval elapses.
4. WHILE an Ingestion_Job is in progress, THE MCP_Server SHALL serve queries against the Project_Profiles that were persisted before the in-progress Ingestion_Job began, regardless of whether those Project_Profiles are complete or were left in a degraded state by an earlier failed Ingestion_Job.
5. WHILE no Ingestion_Job is in progress, THE MCP_Server SHALL serve queries against the most recently persisted Project_Profiles in the Knowledge_Store.
6. IF an Ingestion_Job is requested while another Ingestion_Job is already running, THEN THE MCP_Server SHALL reject the new request and return a message stating that an Ingestion_Job is already in progress, AND THE MCP_Server SHALL NOT reject Ingestion_Job requests through this acceptance criterion when no Ingestion_Job is running.

### Requirement 9: Detect Purpose Conflicts Between Projects

**User Story:** As an MCP_Client, I want to ask whether two Projects have conflicting purposes, so that I can spot duplicated or contradictory ownership across the codebase.

#### Acceptance Criteria

1. WHEN the Conflict_Detector is invoked for a pair of Projects, THE Conflict_Detector SHALL return a result indicating whether the pair has a Purpose_Conflict and a justification string referencing the purpose summaries that led to the result.
2. WHEN the Conflict_Detector is invoked for the full set of Projects, THE Conflict_Detector SHALL return the list of all Project pairs that have a Purpose_Conflict.
3. THE Conflict_Detector SHALL classify a pair of Projects as having a Purpose_Conflict only when, in addition to both Projects having a non-"unknown" purpose summary, the two purpose summaries either describe substantially the same primary responsibility or assert contradictory ownership of the same responsibility, and THE Conflict_Detector SHALL NOT classify a pair as having a Purpose_Conflict on the basis of any criterion other than substantially the same primary responsibility or contradictory ownership of the same responsibility.
4. IF either Project in a requested pair has a purpose summary value of "unknown", THEN THE Conflict_Detector SHALL return a result of "indeterminate" for that pair and include a justification stating that the purpose summary is unknown.

### Requirement 10: Expose MCP Tools for Querying Project Knowledge

**User Story:** As an MCP_Client, I want MCP tools that answer the core questions about Projects, so that I can integrate project knowledge into my own workflows.

#### Acceptance Criteria

1. THE MCP_Server SHALL expose an MCP tool that returns the purpose summary for a Project identified by its GitLab project ID.
2. THE MCP_Server SHALL expose an MCP tool that returns the Abstract_Inputs and Abstract_Outputs for a Project identified by its GitLab project ID.
3. THE MCP_Server SHALL expose an MCP tool that returns the External_Service_Dependencies and Database_Table_Dependencies for a Project identified by its GitLab project ID.
4. THE MCP_Server SHALL expose an MCP tool that returns the list of all Project pairs that have a Purpose_Conflict.
5. THE MCP_Server SHALL expose an MCP tool that returns the full Project_Profile for a Project identified by its GitLab project ID.
6. THE MCP_Server SHALL expose an MCP tool that returns the list of all in-scope Projects with their GitLab project IDs and full paths.
7. IF an MCP tool is invoked with a GitLab project ID that does not match an in-scope Project, THEN THE MCP_Server SHALL return an error result with a message stating that the Project is not in scope.

### Requirement 11: Conform to MCP Protocol

**User Story:** As an MCP_Client, I want the server to speak the Model Context Protocol correctly, so that any compliant MCP client can connect to it.

#### Acceptance Criteria

1. THE MCP_Server SHALL implement the Model Context Protocol server role over the standard input and standard output transport.
2. WHEN an MCP_Client sends an initialize request, THE MCP_Server SHALL respond with the MCP_Server's name, version, and the list of supported MCP capabilities.
3. WHEN an MCP_Client sends a tools/list request, THE MCP_Server SHALL return the list of MCP tools defined by Requirements 8 and 10 with their input schemas, and THE MCP_Server SHALL send a tools/list response only in response to an explicit tools/list request from an MCP_Client and SHALL NOT send unsolicited tools/list responses.
4. WHEN an MCP_Client sends a tools/call request for a defined tool with valid arguments, THE MCP_Server SHALL execute the tool and return the result in the MCP tool result format.
5. IF an MCP_Client sends a tools/call request for an undefined tool name, THEN THE MCP_Server SHALL return an MCP error response indicating that the tool is unknown.
6. IF an MCP_Client sends a tools/call request with arguments that fail the tool's input schema validation, THEN THE MCP_Server SHALL return an MCP error response that names the failing argument and the validation rule that failed.
7. IF the execution of a defined MCP tool invoked via a tools/call request fails internally due to a runtime error or an unavailable external dependency, THEN THE MCP_Server SHALL return an MCP error response that indicates the tool execution failed and that names the failure reason.

### Requirement 12: Run Local Visualization Server

**User Story:** As a human operator, I want the server to expose a local web interface on my own machine, so that I can open it in a browser to read and understand the project knowledge the MCP_Server has built.

#### Acceptance Criteria

1. WHEN the MCP_Server starts, THE MCP_Server SHALL start the Visualization_Server in the same OS process as the MCP_Server, so that the Visualization_Server shares the Knowledge_Store handle and lifecycle with the MCP_Server.
2. THE Visualization_Server SHALL bind only to IPv4 address 127.0.0.1 and IPv6 address ::1 on the Loopback_Interface, and THE Visualization_Server SHALL NOT accept HTTP connections on any other network interface address.
3. THE MCP_Server SHALL accept a configuration value for the TCP port the Visualization_Server listens on, and the configured TCP port value SHALL be an integer.
4. WHERE the Visualization_Server TCP port is not configured, THE Visualization_Server SHALL listen on TCP port 7345.
5. IF the configured Visualization_Server TCP port value is not an integer, THEN THE MCP_Server SHALL fail startup, emit an error message naming the Visualization_Server port configuration value and stating that an integer is required, and terminate the MCP_Server process, and IF the configured Visualization_Server TCP port value is an integer outside the range 1 through 65535, THEN THE MCP_Server SHALL fail startup, emit an error message naming the Visualization_Server port configuration value and the allowed range, and terminate the MCP_Server process.
6. IF the configured Visualization_Server TCP port is already in use at startup, THEN THE MCP_Server SHALL fail startup, emit an error message naming the port and stating that the port is already in use, and terminate the MCP_Server process.
7. WHEN the Visualization_Server has started successfully, defined as the Visualization_Server having bound to the configured TCP port and being ready to accept HTTP connections, THE MCP_Server SHALL emit a log message stating the URL "http://127.0.0.1:{port}" at which the Visualization_Server is reachable.
8. IF the Visualization_Server fails to start for any reason other than a Visualization_Server TCP port value that is not an integer or is outside the range 1 through 65535 (Requirement 12 acceptance criterion 5) or a port-already-in-use condition (Requirement 12 acceptance criterion 6), THEN THE MCP_Server SHALL fail startup, emit an error message that names the underlying failure reason reported by the operating system or HTTP runtime, and terminate the MCP_Server process.
9. WHEN the MCP_Server shuts down, THE MCP_Server SHALL stop the Visualization_Server before the MCP_Server process exits, and THE Visualization_Server SHALL stop accepting new HTTP connections during shutdown.

### Requirement 13: Render Project Knowledge Diagrams

**User Story:** As a human operator, I want the local web interface to render diagrams of each Project's profile, the dependencies across Projects, and the purpose conflicts between Projects, so that I can understand the project knowledge without calling MCP tools.

#### Acceptance Criteria

1. WHEN the Visualization_Server receives an HTTP GET request for the path "/", THE Visualization_Server SHALL respond with HTTP status 200 and an HTML page that lists every in-scope Project ordered by GitLab project ID ascending, where each list entry includes the Project's GitLab project ID, the Project's GitLab full path, a link to that Project's Project_Profile_Diagram, a link to the Dependency_Graph_Diagram, and a link to the Conflict_Overview_Diagram, and where, if there are zero in-scope Projects, the response body includes a visible message stating "No Projects are in scope" and omits per-Project list entries.
2. WHEN the Visualization_Server receives an HTTP GET request for the path "/projects/{project_id}" where {project_id} is a sequence of one or more decimal digits matching the GitLab project ID of an in-scope Project, THE Visualization_Server SHALL respond with HTTP status 200 and an HTML page that renders the Project_Profile_Diagram for that Project showing the Project's purpose summary, Abstract_Inputs grouped by category with each input's human-readable description, Abstract_Outputs grouped by category with each output's human-readable description, External_Service_Dependencies labeled by service kind, and Database_Table_Dependencies labeled by access mode, where any of {Abstract_Inputs, Abstract_Outputs, External_Service_Dependencies, Database_Table_Dependencies} that is empty for the Project is rendered with a visible empty-state message naming that section (for example, "No Abstract Inputs detected").
3. WHEN the Visualization_Server receives an HTTP GET request for the path "/dependencies", THE Visualization_Server SHALL respond with HTTP status 200 and an HTML page that renders the Dependency_Graph_Diagram, in which each in-scope Project is a node, an edge labeled "shared external service: {service_name}" connects every pair of Projects that both list the same External_Service_Dependency, and an edge labeled "shared table: {table_name}" connects every pair of Projects that both list a Database_Table_Dependency on the same table name, and where, when no two in-scope Projects share any External_Service_Dependency or any Database_Table_Dependency, the rendered Dependency_Graph_Diagram displays the in-scope Project nodes (or, if there are no in-scope Projects, a visible empty-state message stating that no Projects are in scope) and includes a visible message stating that no shared dependencies were detected between Projects.
4. WHEN the Visualization_Server receives an HTTP GET request for the path "/conflicts", THE Visualization_Server SHALL respond with HTTP status 200 and an HTML page that renders the Conflict_Overview_Diagram, in which each in-scope Project is a node and an edge connects every pair of Projects classified by the Conflict_Detector as having a Purpose_Conflict, with each edge labeled by the Purpose_Conflict justification string from Requirement 9, and where, when the Conflict_Detector reports zero Purpose_Conflicts across the in-scope Projects, the rendered Conflict_Overview_Diagram displays the in-scope Project nodes (or, if there are no in-scope Projects, a visible empty-state message stating that no Projects are in scope) and includes a visible message stating that no purpose conflicts were detected.
5. THE Visualization_Server SHALL return every Project_Knowledge_Diagram as an HTTP response with content type "text/html; charset=utf-8".
6. IF the Visualization_Server receives an HTTP GET request for the path "/projects/{project_id}" where {project_id} is a sequence of one or more decimal digits but does not match the GitLab project ID of any in-scope Project, THEN THE Visualization_Server SHALL respond with HTTP status 404 and an HTML page stating that the requested Project is not in scope and including the value of {project_id} that was requested.
7. IF the Visualization_Server receives an HTTP GET request for a path that is not "/", "/projects/{project_id}" where {project_id} is a sequence of one or more decimal digits, "/dependencies", or "/conflicts", THEN THE Visualization_Server SHALL respond with HTTP status 404 and an HTML page that includes the requested HTTP path and that states the requested page does not exist.
8. IF the Visualization_Server receives an HTTP request whose method is not GET for any of the paths "/", "/projects/{project_id}", "/dependencies", or "/conflicts", THEN THE Visualization_Server SHALL respond with HTTP status 405 and an Allow header whose value is exactly the string "GET".
9. WHEN the Visualization_Server receives any HTTP GET request for the paths "/", "/projects/{project_id}", "/dependencies", or "/conflicts", THE Visualization_Server SHALL begin sending the HTTP response within 5 seconds of receiving the request, measured at the Visualization_Server's HTTP layer.

### Requirement 14: Source Diagrams from Persisted Project Profiles

**User Story:** As a human operator, I want the diagrams to reflect the same Project_Profiles that the MCP tools serve, so that what I see in the browser matches what the MCP tools return and so that I am not shown stale or made-up data.

#### Acceptance Criteria

1. WHEN the Visualization_Server renders any Project_Knowledge_Diagram, THE Visualization_Server SHALL derive the rendered content from the persisted Project_Profiles in the Knowledge_Store that the MCP tools defined in Requirement 10 serve, and THE Visualization_Server SHALL read Project_Profiles from the Knowledge_Store at the time the HTTP request is handled, and THE Visualization_Server SHALL NOT render Project_Knowledge_Diagrams from any source other than the Knowledge_Store, including in-memory caches independent of the Knowledge_Store and synthesized data.
2. WHILE an Ingestion_Job is in progress, THE Visualization_Server SHALL render Project_Knowledge_Diagrams from the Project_Profiles that were persisted before the in-progress Ingestion_Job began, consistent with Requirement 8 acceptance criterion 4, and THE Visualization_Server SHALL NOT render content drawn from Project_Profiles that the in-progress Ingestion_Job has only partially written to the Knowledge_Store.
3. IF the Visualization_Server receives an HTTP GET request for the path "/projects/{project_id}" for an in-scope Project for which no Project_Profile has been persisted in the Knowledge_Store, THEN THE Visualization_Server SHALL respond with HTTP status 200 and an HTML page that contains a visible human-readable message stating that the Project has not yet been analyzed and that an Ingestion_Job must be run, and THE Visualization_Server SHALL NOT include a Project_Profile_Diagram for that Project in the response HTML.
4. IF the Visualization_Server receives an HTTP GET request for any of the paths "/", "/dependencies", or "/conflicts" and no Ingestion_Job has ever completed for the configured GitLab_Source, THEN THE Visualization_Server SHALL respond with HTTP status 200 and an HTML page that contains a visible human-readable message stating that no project knowledge is available yet and that an Ingestion_Job must be run, and THE Visualization_Server SHALL NOT include a Project_Profile_Diagram, Dependency_Graph_Diagram, or Conflict_Overview_Diagram in the response HTML.
5. IF the Visualization_Server receives an HTTP GET request for the path "/projects/{project_id}" where {project_id} is a sequence of one or more decimal digits but does not match any in-scope Project, THEN THE Visualization_Server SHALL respond per Requirement 13 acceptance criterion 6 with HTTP status 404 and an HTML page stating that the requested Project is not in scope, to disambiguate "in-scope but not yet analyzed" from "out of scope".
6. IF the Visualization_Server cannot read Project_Profiles from the Knowledge_Store while handling an HTTP GET request for "/", "/projects/{project_id}", "/dependencies", or "/conflicts", THEN THE Visualization_Server SHALL respond with HTTP status 503 and an HTML page stating that project knowledge is temporarily unavailable, and THE Visualization_Server SHALL NOT serve previously cached or in-memory Project_Profile data in response.

### Requirement 15: Configure Analysis Branch

**User Story:** As an operator, I want to choose which branch the server reads from for each Project, so that analysis can target a release-staging branch (such as "uat") rather than the project's default branch.

#### Acceptance Criteria

1. THE MCP_Server SHALL accept a configuration value for the Analysis_Branch.
2. WHERE the Analysis_Branch is not configured, THE MCP_Server SHALL use the value "uat" as the Analysis_Branch.
3. WHEN an Ingestion_Job runs, THE GitLab_Connector SHALL fetch repository contents and the most recent commit SHA from the Analysis_Branch on each in-scope Project, regardless of the Project's GitLab default branch.
4. WHEN enumerating repositories per Requirement 2 acceptance criterion 2, THE GitLab_Connector SHALL record the Analysis_Branch name as the branch used for analysis (replacing or supplementing the previously recorded "default branch name") and SHALL record the most recent commit SHA on the Analysis_Branch as the commit SHA from which the Project_Profile is derived (consistent with Requirement 7 acceptance criterion 4).
5. IF a Project does not have a branch matching the Analysis_Branch, THEN THE GitLab_Connector SHALL skip producing a Project_Profile for that Project, SHALL record a reason of "analysis_branch_missing" naming the Analysis_Branch value and the Project's GitLab project ID, and SHALL continue the Ingestion_Job for the remaining in-scope Projects.
6. IF the Analysis_Branch configuration value is the empty string, THEN THE MCP_Server SHALL fail startup, emit an error message naming the Analysis_Branch configuration value and stating that an empty Analysis_Branch is not allowed, and terminate the MCP_Server process.

"""Go-specific sub-analyzers for the Project_Analyzer.

This sub-package extends the Project_Analyzer with Go-aware behavior, so
repositories whose source code is written in Go produce Project_Profiles
with the same depth and shape as Python, JavaScript/TypeScript, or Java
repositories.

The implementation language is Python; no Go toolchain is invoked at
runtime. The four Go scanners (purpose, I/O, external services, database
tables) consume an event stream produced by an in-process tokenizer and
construct recognizer.

This ``__init__`` is currently a placeholder for the public re-exports
that subsequent tasks will introduce. The vendor-exclusion predicate and
the repo-level guard are the only Go-related symbols available so far.
"""

from __future__ import annotations

from project_knowledge_mcp.project_analyzer.go.go_filter import (
    has_go_artefacts,
    is_go_source_file,
)

__all__ = [
    "has_go_artefacts",
    "is_go_source_file",
]

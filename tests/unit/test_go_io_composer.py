"""Unit tests for the ``extract_go_io`` composer entry point.

These tests pin the contract of :func:`extract_go_io` — the public
composer that wires the five package-internal Go I/O recognizers
(HTTP, scheduler, ActiveMQ, file I/O, CLI) into a single function
returning ``(inputs, outputs, file_skip_messages)``.

The composer's responsibilities (task 7.11):

1. Compose all five recognizers in a deterministic order.
2. Iterate ``events_by_file`` in path-sorted order so the output is a
   stable projection of the input mapping (Requirement 11.4).
3. Deduplicate by ``(category, description)`` across all recognizers
   per the ``io_extractor._Accumulator`` rule (Requirement 3.7).
4. Convert every :class:`SkipFileEvent` into a
   ``"skipped <path> (<reason>)"`` string for the third return value
   (Requirement 10.4 cross-references).

Implements Requirements 1.1, 1.2, 3.7, 11.4.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import (
    AbstractInputCategory,
    AbstractOutputCategory,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.go._events import (
    FuncDeclEvent,
    GoEvent,
    ImportEvent,
    MethodCallEvent,
    SkipFileEvent,
    StringLitArg,
)
from project_knowledge_mcp.project_analyzer.go.go_io import extract_go_io

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo(files: dict[str, str] | None = None) -> RepositoryContents:
    """Build a minimal :class:`RepositoryContents` for the given file map.

    The composer reads ``go.mod`` only through the CLI recognizer
    (Requirements 7.2, 7.3); a missing or empty ``go.mod`` is a
    no-op for ``cmd/<name>/main.go`` directory-derived binaries.
    """

    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files or {},
    )


def _import(path: str, *, file_path: str, line: int = 1) -> ImportEvent:
    return ImportEvent(path=path, alias=None, file_path=file_path, line=line)


def _call(
    *,
    receiver_chain: tuple[str, ...],
    method_name: str,
    args: tuple,
    file_path: str,
    line: int,
) -> MethodCallEvent:
    return MethodCallEvent(
        receiver_chain=receiver_chain,
        method_name=method_name,
        args=args,
        file_path=file_path,
        line=line,
    )


def _func_main(file_path: str, line: int = 1) -> FuncDeclEvent:
    return FuncDeclEvent(
        name="main",
        receiver_type=None,
        file_path=file_path,
        line=line,
        body_token_range=(0, 0),
    )


# ---------------------------------------------------------------------------
# Composer behavior
# ---------------------------------------------------------------------------


def test_extract_go_io_empty_mapping_returns_three_empty_lists() -> None:
    """An empty event mapping yields three empty lists, never raises."""

    inputs, outputs, skips = extract_go_io(_repo(), {})

    assert inputs == []
    assert outputs == []
    assert skips == []


def test_extract_go_io_routes_http_through_to_inputs_and_outputs() -> None:
    """A single ``http.HandleFunc`` registration surfaces one input and one output."""

    events: dict[str, list[GoEvent]] = {
        "server.go": [
            _import("net/http", file_path="server.go", line=1),
            _call(
                receiver_chain=("http",),
                method_name="HandleFunc",
                args=(StringLitArg(value="GET /healthz"),),
                file_path="server.go",
                line=10,
            ),
        ],
    }

    inputs, outputs, skips = extract_go_io(_repo(), events)

    assert [i.category for i in inputs] == [AbstractInputCategory.HTTP_REQUEST]
    assert [o.category for o in outputs] == [AbstractOutputCategory.HTTP_RESPONSE]
    assert "GET" in inputs[0].description and "/healthz" in inputs[0].description
    assert skips == []


def test_extract_go_io_composes_multiple_recognizers_into_one_call() -> None:
    """HTTP, file-I/O, and CLI recognizers all contribute to one composer call."""

    events: dict[str, list[GoEvent]] = {
        "cmd/api/main.go": [
            _func_main("cmd/api/main.go", line=5),
        ],
        "handler.go": [
            _import("net/http", file_path="handler.go", line=1),
            _call(
                receiver_chain=("http",),
                method_name="HandleFunc",
                args=(StringLitArg(value="POST /submit"),),
                file_path="handler.go",
                line=20,
            ),
        ],
        "writer.go": [
            _call(
                receiver_chain=("os",),
                method_name="WriteFile",
                args=(StringLitArg(value="/tmp/out.log"),),
                file_path="writer.go",
                line=7,
            ),
        ],
    }

    inputs, outputs, skips = extract_go_io(_repo(), events)

    in_cats = {i.category for i in inputs}
    out_cats = {o.category for o in outputs}
    assert AbstractInputCategory.HTTP_REQUEST in in_cats
    assert AbstractInputCategory.CLI_ARGUMENT in in_cats
    assert AbstractOutputCategory.HTTP_RESPONSE in out_cats
    assert AbstractOutputCategory.FILE_WRITTEN in out_cats
    assert skips == []


def test_extract_go_io_deduplicates_by_category_and_description() -> None:
    """Repeated registrations at identical call sites coalesce."""

    repeated_call = _call(
        receiver_chain=("http",),
        method_name="HandleFunc",
        args=(StringLitArg(value="GET /ping"),),
        file_path="server.go",
        line=15,
    )
    events: dict[str, list[GoEvent]] = {
        "server.go": [
            _import("net/http", file_path="server.go", line=1),
            repeated_call,
            repeated_call,
        ],
    }

    inputs, outputs, _ = extract_go_io(_repo(), events)

    # Even though the event list contains the same call twice, dedup by
    # (category, description) yields exactly one input and one output.
    assert len(inputs) == 1
    assert len(outputs) == 1


def test_extract_go_io_path_sorted_iteration_is_deterministic() -> None:
    """Reordering the input mapping does not change the produced lists."""

    events_a: dict[str, list[GoEvent]] = {
        "a_first.go": [
            _import("net/http", file_path="a_first.go", line=1),
            _call(
                receiver_chain=("http",),
                method_name="HandleFunc",
                args=(StringLitArg(value="GET /a"),),
                file_path="a_first.go",
                line=2,
            ),
        ],
        "z_last.go": [
            _import("net/http", file_path="z_last.go", line=1),
            _call(
                receiver_chain=("http",),
                method_name="HandleFunc",
                args=(StringLitArg(value="GET /z"),),
                file_path="z_last.go",
                line=2,
            ),
        ],
    }
    events_b: dict[str, list[GoEvent]] = {
        "z_last.go": events_a["z_last.go"],
        "a_first.go": events_a["a_first.go"],
    }

    inputs_a, outputs_a, _ = extract_go_io(_repo(), events_a)
    inputs_b, outputs_b, _ = extract_go_io(_repo(), events_b)

    assert [i.description for i in inputs_a] == [i.description for i in inputs_b]
    assert [o.description for o in outputs_a] == [o.description for o in outputs_b]
    # The first description references the alphabetically earlier file.
    assert "a_first.go" in inputs_a[0].description


def test_extract_go_io_skip_file_event_produces_skip_message() -> None:
    """A ``SkipFileEvent`` materializes as a ``"skipped <path> (<reason>)"`` string."""

    events: dict[str, list[GoEvent]] = {
        "constrained.go": [
            SkipFileEvent(
                reason="build constraint requires toolchain",
                file_path="constrained.go",
                line=1,
            ),
        ],
        "cgo_user.go": [
            SkipFileEvent(
                reason="cgo directive requires toolchain",
                file_path="cgo_user.go",
                line=3,
            ),
        ],
    }

    inputs, outputs, skips = extract_go_io(_repo(), events)

    assert inputs == []
    assert outputs == []
    assert skips == [
        "skipped cgo_user.go (cgo directive requires toolchain)",
        "skipped constrained.go (build constraint requires toolchain)",
    ]


def test_extract_go_io_skip_messages_sorted_by_path() -> None:
    """Skip messages are emitted in path-sorted order regardless of mapping order."""

    events: dict[str, list[GoEvent]] = {
        "zebra.go": [
            SkipFileEvent(reason="r1", file_path="zebra.go", line=1),
        ],
        "alpha.go": [
            SkipFileEvent(reason="r2", file_path="alpha.go", line=1),
        ],
    }

    _, _, skips = extract_go_io(_repo(), events)

    assert skips == [
        "skipped alpha.go (r2)",
        "skipped zebra.go (r1)",
    ]


def test_extract_go_io_skip_and_emission_coexist() -> None:
    """A skip in one file does not suppress emissions from another file."""

    events: dict[str, list[GoEvent]] = {
        "skipped.go": [
            SkipFileEvent(
                reason="tokenization failed: unterminated string",
                file_path="skipped.go",
                line=42,
            ),
        ],
        "working.go": [
            _import("net/http", file_path="working.go", line=1),
            _call(
                receiver_chain=("http",),
                method_name="HandleFunc",
                args=(StringLitArg(value="GET /ok"),),
                file_path="working.go",
                line=5,
            ),
        ],
    }

    inputs, outputs, skips = extract_go_io(_repo(), events)

    assert len(inputs) == 1
    assert len(outputs) == 1
    assert skips == [
        "skipped skipped.go (tokenization failed: unterminated string)",
    ]


def test_extract_go_io_returns_tuple_of_three_lists() -> None:
    """The composer's return shape is exactly a 3-tuple of lists."""

    result = extract_go_io(_repo(), {})

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(part, list) for part in result)


# ---------------------------------------------------------------------------
# Operator-tuning noise suppression
# ---------------------------------------------------------------------------
#
# The composer applies two post-recognizer prunes before merging the
# per-recognizer outputs (see ``go_io._OPERATOR_TUNED_*`` constants):
#
# * ``file_read`` / ``file_written`` whose call site is the canonical
#   config loader (``config/config.go``).
# * ``cli_argument`` binary entry-points at ``cmd/main.go`` (the
#   directory-shaped ``cmd/<name>/main.go`` is kept).
#
# These tests pin the prune behavior at the composer's public
# boundary. The per-recognizer property tests
# (``test_property_08_go_file_io``, ``test_property_09_go_cli_detection``)
# continue to assert the unfiltered spec contract because the prune
# runs *after* the recognizers return.


def test_file_read_at_config_config_go_is_pruned() -> None:
    """``os.ReadFile`` whose call site is ``config/config.go`` is dropped."""

    events: dict[str, list[GoEvent]] = {
        "config/config.go": [
            _call(
                receiver_chain=("os",),
                method_name="ReadFile",
                args=(StringLitArg(value="config.yml"),),
                file_path="config/config.go",
                line=224,
            ),
        ],
    }

    inputs, outputs, _ = extract_go_io(_repo(), events)

    file_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.FILE_READ
    ]
    assert file_inputs == [], (
        "operator-tuning filter should prune file_read at config/config.go; "
        f"got {[i.description for i in file_inputs]}"
    )
    assert outputs == []


def test_file_read_at_other_paths_is_kept() -> None:
    """A read outside the operator-excluded path set is still emitted."""

    events: dict[str, list[GoEvent]] = {
        "internal/storage/disk.go": [
            _call(
                receiver_chain=("os",),
                method_name="ReadFile",
                args=(StringLitArg(value="payload.json"),),
                file_path="internal/storage/disk.go",
                line=42,
            ),
        ],
    }

    inputs, _outputs, _ = extract_go_io(_repo(), events)

    file_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.FILE_READ
    ]
    assert len(file_inputs) == 1
    assert "internal/storage/disk.go" in file_inputs[0].description


def test_file_written_at_config_config_go_is_also_pruned() -> None:
    """``os.WriteFile`` whose call site is ``config/config.go`` is dropped.

    The filter is symmetric so an ``os.OpenFile`` at the excluded
    path that resolves to the undecidable "both" verdict is pruned
    from both the input and the output list together.
    """

    events: dict[str, list[GoEvent]] = {
        "config/config.go": [
            _call(
                receiver_chain=("os",),
                method_name="WriteFile",
                args=(StringLitArg(value="config.yml"),),
                file_path="config/config.go",
                line=300,
            ),
        ],
    }

    inputs, outputs, _ = extract_go_io(_repo(), events)

    assert inputs == []
    file_outputs = [
        o for o in outputs if o.category is AbstractOutputCategory.FILE_WRITTEN
    ]
    assert file_outputs == [], (
        "operator-tuning filter should prune file_written at config/config.go; "
        f"got {[o.description for o in file_outputs]}"
    )


def test_cli_binary_at_cmd_main_go_is_pruned() -> None:
    """``func main()`` at ``cmd/main.go`` produces no binary cli_argument.

    A ``go.mod`` is provided so the binary name would otherwise
    resolve (Requirement 7.2); the prune is therefore the decisive
    factor here, not a missing go.mod.
    """

    repo = _repo({"go.mod": "module disb-status-inquiry\n"})
    events: dict[str, list[GoEvent]] = {
        "cmd/main.go": [
            _func_main("cmd/main.go", line=87),
        ],
    }

    inputs, _outputs, _ = extract_go_io(repo, events)

    cli_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.CLI_ARGUMENT
    ]
    assert cli_inputs == [], (
        "operator-tuning filter should prune binary entry-point at "
        f"cmd/main.go; got {[i.description for i in cli_inputs]}"
    )


def test_cli_binary_at_cmd_named_main_go_is_kept() -> None:
    """``cmd/<name>/main.go`` is still emitted; only ``cmd/main.go`` is excluded.

    The directory segment in ``cmd/api/main.go`` distinguishes
    binaries in a multi-binary repository and is therefore retained.
    """

    events: dict[str, list[GoEvent]] = {
        "cmd/api/main.go": [
            _func_main("cmd/api/main.go", line=5),
        ],
    }

    inputs, _outputs, _ = extract_go_io(_repo(), events)

    cli_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.CLI_ARGUMENT
    ]
    assert len(cli_inputs) == 1
    assert "binary api" in cli_inputs[0].description
    assert "cmd/api/main.go" in cli_inputs[0].description


def test_flag_method_at_cmd_main_go_is_still_emitted() -> None:
    """The CLI prune only targets the binary entry-point; ``flag.*`` stays.

    A ``flag.String`` call at ``cmd/main.go`` encodes the actual CLI
    surface the binary accepts and is therefore retained even though
    the file path is in the binary-entry-point exclusion set.
    """

    repo = _repo({"go.mod": "module disb-status-inquiry\n"})
    events: dict[str, list[GoEvent]] = {
        "cmd/main.go": [
            _func_main("cmd/main.go", line=87),
            _call(
                receiver_chain=("flag",),
                method_name="String",
                args=(
                    StringLitArg(value="port"),
                    StringLitArg(value="8080"),
                    StringLitArg(value="HTTP listener port"),
                ),
                file_path="cmd/main.go",
                line=90,
            ),
        ],
    }

    inputs, _outputs, _ = extract_go_io(repo, events)

    cli_inputs = [
        i for i in inputs if i.category is AbstractInputCategory.CLI_ARGUMENT
    ]
    # Exactly the flag.String detection survives.
    assert len(cli_inputs) == 1
    assert "port" in cli_inputs[0].description
    assert "flag.String" in cli_inputs[0].description
    # The binary entry-point at the same file was pruned.
    assert not any(d.startswith("binary ") for d in (i.description for i in cli_inputs))

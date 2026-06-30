"""Snapshot a Go repository on disk into a ``RepositoryContents`` JSON payload.

This helper walks a target repository, excludes vendored dependencies, VCS
metadata, IDE/build caches, and known binary artefacts, decodes every
remaining file as UTF-8, and serializes the result as a
``RepositoryContents`` JSON document. The output is intended to feed the
Go analyzer integration golden tests (task 12 of the Go analyzer support
spec): committing the JSON snapshots keeps the goldens reproducible
without requiring the sample repositories to be present on every CI
runner.

The script is intentionally conservative about what counts as a text
source file: it filters out directories that are never source (``vendor/``,
``.git/``, ``node_modules/``, common build/cache directories), skips files
with extensions that are reliably binary, rejects files larger than
``DEFAULT_MAX_FILE_SIZE``, and treats any remaining file that contains a
NUL byte or fails UTF-8 decoding as binary. This is a heuristic, but it is
deterministic and produces snapshots that are byte-for-byte stable across
runs on the same input tree, which is what the golden tests need.

Invoke as a module:

    python -m tests.integration.golden.go._snapshot <repo-path>
    python -m tests.integration.golden.go._snapshot <repo-path> --output-dir <dir>
    python -m tests.integration.golden.go._snapshot <repo-path> --name <repo-name>

The default output directory is the directory containing this script
(``tests/integration/golden/go/``) and the snapshot is written to
``<output-dir>/<repo-name>/repository_contents.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path

from project_knowledge_mcp.models import RepositoryContents

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Directories that are never source code and must be skipped wherever they
#: appear in the repository tree (matched on the directory's basename, not
#: just at the repository root).
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        # Go vendored dependencies (Requirement 1.3).
        "vendor",
        # VCS metadata.
        ".git",
        ".hg",
        ".svn",
        # Language ecosystems.
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "target",  # Java/Rust build output
        "dist",
        "build",
        "out",
        # Tooling caches.
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".cache",
        ".tox",
        ".nox",
        ".gradle",
        ".idea",
        ".vscode",
        ".vs",
        # Coverage / artefact directories.
        "coverage",
        ".coverage",
        ".terraform",
    }
)

#: File extensions that are reliably binary. These are skipped without
#: attempting to read the file.
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Executables and libraries.
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".a",
        ".o",
        ".obj",
        ".lib",
        ".class",
        ".jar",
        ".war",
        ".ear",
        # Python byte code.
        ".pyc",
        ".pyo",
        ".pyd",
        # Archives.
        ".zip",
        ".tar",
        ".tgz",
        ".tbz2",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".xz",
        # Images / media.
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".bmp",
        ".tiff",
        ".webp",
        ".pdf",
        ".mp3",
        ".mp4",
        ".mov",
        ".avi",
        # OS / editor artefacts.
        ".ds_store",
        ".swp",
        ".swo",
        # Database / blob.
        ".db",
        ".sqlite",
        ".sqlite3",
    }
)

#: File extensions that are not binary but are runtime/build artefacts and
#: would make the snapshot drift over time. These are skipped even when the
#: bytes are valid UTF-8 text.
NON_SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".log",  # runtime log output (e.g. cat-service/cmd/failed_kafka_logs.log)
        ".pid",
    }
)

#: File basenames that are never source artefacts.
EXCLUDED_FILE_NAMES: frozenset[str] = frozenset({".DS_Store"})

#: Default upper bound on individual file size. Files larger than this are
#: skipped from the snapshot. 1 MiB is comfortably larger than any source
#: file in the four sample repositories.
DEFAULT_MAX_FILE_SIZE: int = 1 * 1024 * 1024

#: Number of bytes read from the head of a file when sniffing for binary
#: content.
_BINARY_SNIFF_BYTES: int = 8192


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_excluded_dir(name: str) -> bool:
    return name in EXCLUDED_DIR_NAMES


def _is_binary_extension(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTENSIONS


def _is_non_source_extension(path: Path) -> bool:
    return path.suffix.lower() in NON_SOURCE_EXTENSIONS


def _looks_like_text(data: bytes) -> bool:
    """Heuristic: a leading NUL byte (or any NUL in the sniffed prefix)
    indicates a binary file. We deliberately stay conservative here so that
    we never include garbage in a snapshot."""
    return b"\x00" not in data


def _read_text_or_none(path: Path, max_size: int) -> str | None:
    """Return the file contents decoded as UTF-8, or ``None`` if the file
    should be skipped (too large, binary, or undecodable)."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > max_size:
        return None

    try:
        with path.open("rb") as fh:
            head = fh.read(_BINARY_SNIFF_BYTES)
            if not _looks_like_text(head):
                return None
            rest = fh.read()
    except OSError:
        return None

    raw = head + rest
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _walk_repository(
    repo_root: Path,
    max_file_size: int,
) -> dict[str, str]:
    """Walk ``repo_root`` and return a ``{relative-path: text-content}`` map.

    Paths in the returned mapping are repository-relative and use forward
    slashes regardless of host OS, matching the convention the rest of the
    analyzer expects.
    """
    files: dict[str, str] = {}

    # We use Path.walk-style iteration manually so we can prune directories
    # in-place. Path.walk is available on 3.12+; doing this by hand keeps
    # the script compatible with the project's >=3.11 floor.
    def visit(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for entry in entries:
            try:
                is_dir = entry.is_dir()
                is_symlink = entry.is_symlink()
            except OSError:
                continue
            # Do not follow symlinks: they could escape the repo root or
            # introduce non-determinism (target changes over time).
            if is_symlink:
                continue
            if is_dir:
                if _is_excluded_dir(entry.name):
                    continue
                visit(entry)
                continue
            # Regular file.
            if entry.name in EXCLUDED_FILE_NAMES:
                continue
            if _is_binary_extension(entry):
                continue
            if _is_non_source_extension(entry):
                continue
            text = _read_text_or_none(entry, max_file_size)
            if text is None:
                continue
            rel = entry.relative_to(repo_root).as_posix()
            files[rel] = text

    visit(repo_root)
    return files


def _git_head_sha(repo_root: Path) -> str | None:
    """Return the commit SHA at ``HEAD`` of the repository, or ``None`` if
    the path is not a git working tree."""
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _stable_project_id(name: str) -> int:
    """Derive a deterministic positive integer from the repo name.

    The integration goldens have no real GitLab project to point at, but
    ``RepositoryContents`` requires an integer. We use CRC32 of the repo
    name so the id is reproducible across runs and visibly distinct between
    the four sample repositories.
    """
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def snapshot_repository(
    repo_root: Path,
    *,
    name: str | None = None,
    commit_sha: str | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> RepositoryContents:
    """Build a :class:`RepositoryContents` for the repository at ``repo_root``.

    Args:
        repo_root: Path to the repository working tree.
        name: Repository name used to derive a stable ``gitlab_project_id``.
            Defaults to the directory's basename.
        commit_sha: Commit SHA to record. When ``None``, the script reads
            ``git rev-parse HEAD`` if available and falls back to a fixed
            placeholder when the directory is not a git tree.
        max_file_size: Upper bound on individual file size; files above the
            threshold are skipped.
    """
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repository path is not a directory: {repo_root}")

    repo_name = name or repo_root.name
    files = _walk_repository(repo_root, max_file_size=max_file_size)
    sha = commit_sha or _git_head_sha(repo_root) or "0" * 40
    project_id = _stable_project_id(repo_name)

    return RepositoryContents(
        gitlab_project_id=project_id,
        commit_sha=sha,
        files=files,
    )


def write_snapshot(
    contents: RepositoryContents,
    output_dir: Path,
    *,
    repo_name: str,
) -> Path:
    """Write ``contents`` to ``<output_dir>/<repo_name>/repository_contents.json``.

    The directory is created if it does not exist. Returns the path the
    JSON document was written to.
    """
    target_dir = output_dir / repo_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "repository_contents.json"

    # Pydantic's model_dump emits ``files`` in insertion order; the walker
    # already iterates entries in sorted order, but we re-sort here so the
    # JSON payload is fully deterministic regardless of dict order on the
    # source side.
    payload = contents.model_dump(mode="json")
    payload["files"] = dict(sorted(payload["files"].items()))

    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.integration.golden.go._snapshot",
        description=(
            "Snapshot a Go repository on disk into a RepositoryContents JSON "
            "payload for the Go analyzer integration golden tests."
        ),
    )
    parser.add_argument(
        "repo_path",
        type=Path,
        help="Path to the repository working tree to snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write the snapshot into. Defaults to the directory "
            "containing this script (tests/integration/golden/go/)."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Repository name to use for the snapshot directory and as the "
            "stable seed for gitlab_project_id. Defaults to the basename of "
            "repo_path."
        ),
    )
    parser.add_argument(
        "--commit-sha",
        type=str,
        default=None,
        help=(
            "Commit SHA to record. Defaults to `git rev-parse HEAD` of the "
            "repo when available, otherwise 40 zeros."
        ),
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE,
        help=(
            "Skip files larger than this many bytes. Defaults to "
            f"{DEFAULT_MAX_FILE_SIZE} ({DEFAULT_MAX_FILE_SIZE // 1024} KiB)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:]) if argv is None else argv)

    repo_path: Path = args.repo_path.resolve()
    repo_name: str = args.name or repo_path.name
    output_dir: Path = (args.output_dir or Path(__file__).resolve().parent).resolve()

    contents = snapshot_repository(
        repo_path,
        name=repo_name,
        commit_sha=args.commit_sha,
        max_file_size=args.max_file_size,
    )
    target = write_snapshot(contents, output_dir, repo_name=repo_name)

    print(
        f"Wrote {len(contents.files)} files to {target} "
        f"(commit_sha={contents.commit_sha}, gitlab_project_id={contents.gitlab_project_id})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

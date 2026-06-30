"""Unit tests for ``project_analyzer.go.go_db_tables.detect_go_database_tables``.

The detector is now a path-scoped grep heuristic: files matching
``internal/repository/<name>_repo.go`` (or ``_repos.go``) are handed
to :func:`db_tables.extract_table_references_preserving_schema`, and
the produced ``(table_name, access_mode)`` pairs are aggregated
through the existing :func:`db_tables._aggregate` helper.

These tests build synthetic :class:`RepositoryContents` snapshots and
assert on the public detection contract:

* Only repository-pattern files contribute (path scoping).
* Schema-qualified table names round-trip verbatim
  (Requirement 9.5).
* Read+write observations on the same table coalesce to
  ``READ_WRITE`` (Requirement 9.6).
* ``UNKNOWN`` is the lowest-priority observation (Requirement 9.8).
* Vendored ``.go`` files at the same path shape are excluded.
* ``events_by_file`` is ignored; passing an empty mapping is fine.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import (
    DatabaseAccessMode,
    RepositoryContents,
    SourceLocation,
)
from project_knowledge_mcp.project_analyzer.go.go_db_tables import (
    detect_go_database_tables,
)

pytestmark = pytest.mark.unit


def _repo(files: dict[str, str]) -> RepositoryContents:
    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )


# ---------------------------------------------------------------------------
# Empty / no-match cases
# ---------------------------------------------------------------------------


def test_empty_repo_returns_empty_detections() -> None:
    detections, skips = detect_go_database_tables(_repo({}), {})

    assert detections == []
    assert skips == []


def test_repository_with_no_repo_pattern_files_emits_nothing() -> None:
    """Non-matching paths are silently ignored, even when they contain SQL."""

    files = {
        "internal/usecase/foo.go": (
            'package usecase\n'
            'const q = "SELECT id FROM users"\n'
        ),
        "scripts/migrate.sql": "INSERT INTO orders VALUES (1);",
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert detections == []


def test_nested_repository_subdir_does_not_match_strict_path() -> None:
    """``internal/repository/sub/foo_repo.go`` is one segment too deep."""

    files = {
        "internal/repository/sub/foo_repo.go": (
            'package repository\n'
            'const q = "SELECT id FROM users WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert detections == []


def test_plural_repos_suffix_also_matches() -> None:
    """The detector accepts ``_repos.go`` in addition to ``_repo.go``."""

    files = {
        "internal/repository/widget_repos.go": (
            'package repository\n'
            'const q = "SELECT id FROM WIDGETS WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert len(detections) == 1
    assert detections[0].table_name == "WIDGETS"


def test_test_file_with_repo_suffix_does_not_match() -> None:
    """``foo_repo_test.go`` ends with ``_test.go`` and is excluded."""

    files = {
        "internal/repository/foo_repo_test.go": (
            'package repository\n'
            'const q = "INSERT INTO sensitive VALUES (1)"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert detections == []


def test_vendored_repo_file_is_excluded() -> None:
    """A vendored ``.go`` at the same path shape is dropped."""

    files = {
        "vendor/example.com/lib/internal/repository/foo_repo.go": (
            'package lib\n'
            'const q = "SELECT id FROM lib_table"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert detections == []


# ---------------------------------------------------------------------------
# Schema-preserving table-name extraction
# ---------------------------------------------------------------------------


def test_select_preserves_schema_qualifier() -> None:
    files = {
        "internal/repository/payment_repo.go": (
            'package repository\n'
            'const q = "SELECT id FROM APP_WS.IIB_PAYMENT_CONFIRMED WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.table_name == "APP_WS.IIB_PAYMENT_CONFIRMED"
    assert dep.access_mode is DatabaseAccessMode.READ
    # Source location is ``(path, None)`` — grep does not track line.
    assert dep.source_locations == [
        SourceLocation(path="internal/repository/payment_repo.go", line=None),
    ]


def test_insert_emits_write_access_mode() -> None:
    files = {
        "internal/repository/order_repo.go": (
            'package repository\n'
            'const q = "INSERT INTO ORDERS (id) VALUES (:1)"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert len(detections) == 1
    assert detections[0].table_name == "ORDERS"
    assert detections[0].access_mode is DatabaseAccessMode.WRITE


# ---------------------------------------------------------------------------
# Read+write coalescing
# ---------------------------------------------------------------------------


def test_read_and_write_on_same_table_coalesce_to_read_write() -> None:
    files = {
        "internal/repository/orders_repo.go": (
            'package repository\n'
            'const q1 = "SELECT id FROM ORDERS WHERE id = :1"\n'
            'const q2 = "INSERT INTO ORDERS (id) VALUES (:1)"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.table_name == "ORDERS"
    assert dep.access_mode is DatabaseAccessMode.READ_WRITE


def test_read_and_write_across_two_files_coalesce_to_read_write() -> None:
    files = {
        "internal/repository/orders_read_repo.go": (
            'package repository\n'
            'const q = "SELECT id FROM ORDERS WHERE id = :1"\n'
        ),
        "internal/repository/orders_write_repo.go": (
            'package repository\n'
            'const q = "INSERT INTO ORDERS (id) VALUES (:1)"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.access_mode is DatabaseAccessMode.READ_WRITE
    # Each file contributes one source location.
    paths = sorted(loc.path for loc in dep.source_locations)
    assert paths == [
        "internal/repository/orders_read_repo.go",
        "internal/repository/orders_write_repo.go",
    ]


# ---------------------------------------------------------------------------
# UNKNOWN access mode fallback
# ---------------------------------------------------------------------------


def test_extractor_handles_merge_into_as_write() -> None:
    """``MERGE INTO`` is part of the recognized write set."""

    files = {
        "internal/repository/order_merge_repo.go": (
            'package repository\n'
            'const q = "MERGE INTO ORDERS USING dual ON (id = :1) WHEN MATCHED THEN UPDATE SET status = :2"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert len(detections) == 1
    assert detections[0].table_name == "ORDERS"
    assert detections[0].access_mode is DatabaseAccessMode.WRITE


# ---------------------------------------------------------------------------
# Ignored parameters
# ---------------------------------------------------------------------------


def test_events_by_file_is_ignored_even_when_nonempty() -> None:
    """A non-empty ``events_by_file`` mapping must not change output."""

    files = {
        "internal/repository/users_repo.go": (
            'package repository\n'
            'const q = "SELECT id FROM USERS"\n'
        ),
    }

    detections_a, _ = detect_go_database_tables(_repo(files), {})
    # The second positional is typed as a mapping; a non-empty dict
    # with arbitrary content must be safely ignored.
    detections_b, _ = detect_go_database_tables(
        _repo(files),
        {"some/other/path.go": ["not-an-event"]},  # type: ignore[list-item]
    )

    assert detections_a == detections_b


# ---------------------------------------------------------------------------
# Multiple tables in one file
# ---------------------------------------------------------------------------


def test_multiple_tables_in_one_file_each_appear_once() -> None:
    files = {
        "internal/repository/multi_repo.go": (
            'package repository\n'
            'const q1 = "SELECT id FROM USERS"\n'
            'const q2 = "INSERT INTO AUDIT (msg) VALUES (:1)"\n'
            'const q3 = "DELETE FROM SESSIONS WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    names = {(d.table_name, d.access_mode) for d in detections}
    assert names == {
        ("USERS", DatabaseAccessMode.READ),
        ("AUDIT", DatabaseAccessMode.WRITE),
        ("SESSIONS", DatabaseAccessMode.WRITE),
    }


# ---------------------------------------------------------------------------
# Stored-procedure detection
# ---------------------------------------------------------------------------


def test_oracle_begin_end_stored_procedure_emits_unknown_access_mode() -> None:
    """``BEGIN <SCHEMA>.<PROC>(...); END;`` is detected as UNKNOWN access."""

    files = {
        "internal/repository/cat_service_repo.go": (
            "package repository\n"
            'const q = "BEGIN APP_ESB_MICROSVC.SP_CAT_CAPTURE_REPAYMENTS'
            '(:P_JOB_ID, :P_BATCH_SIZE, :P_RELEASE_SEC, :P_STATUS, :P_CURSOR); END;"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert len(detections) == 1
    dep = detections[0]
    assert dep.table_name == "APP_ESB_MICROSVC.SP_CAT_CAPTURE_REPAYMENTS"
    # SP body is unknown from the call site alone.
    assert dep.access_mode is DatabaseAccessMode.UNKNOWN


def test_call_statement_stored_procedure_is_also_detected() -> None:
    """The ``CALL <SCHEMA>.<PROC>(...)`` form (Postgres / Oracle 18c+) works."""

    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            'const q = "CALL APP_WS.SP_UPDATE_STATUS(:1, :2);"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "APP_WS.SP_UPDATE_STATUS" for d in detections
    )


def test_unqualified_stored_procedure_is_also_accepted() -> None:
    """A bare ``BEGIN SP_FOO(); END;`` (no schema prefix) is detected."""

    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            'const q = "BEGIN SP_RECALCULATE(); END;"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(d.table_name == "SP_RECALCULATE" for d in detections)


# ---------------------------------------------------------------------------
# False-positive suppression
# ---------------------------------------------------------------------------


def test_go_identifiers_in_dot_chains_are_not_misread_as_tables() -> None:
    """``db.Pool.QueryContext(...)`` is bare Go code, not SQL.

    The extractor only scans **inside** Go string literals; identifiers
    in dot-chains around them must not contribute to detections.
    """
    files = {
        "internal/repository/order_repo.go": (
            "package repository\n"
            "\n"
            "func (r *Repo) Get(ctx context.Context, id int) error {\n"
            "    // ``cache`` is a Go field name, not a SQL table.\n"
            "    rows, err := r.cache.db.Pool.QueryContext(ctx, q)\n"
            "    return err\n"
            "}\n"
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    # The file has zero string literals containing SQL keywords, so
    # zero detections. The Go identifiers ``cache``, ``db``, ``Pool``
    # never appear in the output.
    assert detections == []


@pytest.mark.parametrize(
    "stop_word",
    [
        "DUAL",        # Oracle pseudo-table
        "TABLE",       # SQL keyword
        "SELECT",      # SQL keyword
        "CACHE",       # common Go identifier observed in operator data
        "POOL",        # common Go identifier
        "DB",          # too generic
        "SP",          # too generic
        "FX",          # common Go package name
        "REDIS",       # common Go package name
        "AUTHORIZE",   # function name observed as a false positive
    ],
)
def test_stop_word_table_names_are_rejected(stop_word: str) -> None:
    """The stop-word list suppresses well-known false positives."""

    files = {
        "internal/repository/order_repo.go": (
            "package repository\n"
            f'const q = "SELECT id FROM {stop_word} WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    # The stop word is never emitted; nothing else in the literal
    # matches a table token.
    assert all(d.table_name.upper() != stop_word for d in detections)


@pytest.mark.parametrize(
    "lowercase_name",
    ["users", "orders", "RequestorIdentifier", "SysObj", "ruleIDs"],
)
def test_lowercase_or_camel_case_table_names_are_rejected(
    lowercase_name: str,
) -> None:
    """Non-uppercase identifiers are rejected as likely Go names.

    The operator's Oracle data uses UPPERCASE table names exclusively,
    so any camelCase or all-lowercase candidate is almost certainly a
    Go identifier (a struct field, package name, column reference,
    or similar) caught by accident by the SQL-keyword regex.
    """
    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            f'const q = "SELECT id FROM {lowercase_name} WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert all(d.table_name != lowercase_name for d in detections)


def test_uppercase_table_with_underscore_and_digits_is_accepted() -> None:
    """``APP_DOMINO.IIB_POS_SUBSCRIBER_LOG`` round-trips verbatim."""

    files = {
        "internal/repository/aps_repo.go": (
            "package repository\n"
            'const q = "INSERT INTO APP_DOMINO.IIB_POS_SUBSCRIBER_LOG '
            '(id, msg) VALUES (:1, :2)"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "APP_DOMINO.IIB_POS_SUBSCRIBER_LOG"
        and d.access_mode is DatabaseAccessMode.WRITE
        for d in detections
    )


def test_unqualified_two_character_name_is_rejected() -> None:
    """Two-character unqualified names are likely SQL aliases."""

    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            'const q = "SELECT id FROM T1 WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    # ``T1`` is only two chars and unqualified — likely the alias
    # ``T1`` in a JOIN, not a table name.
    assert all(d.table_name != "T1" for d in detections)


def test_two_character_schema_qualified_name_is_accepted() -> None:
    """Schema-qualified short names bypass the length floor."""

    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            'const q = "SELECT id FROM SC.TB WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(d.table_name == "SC.TB" for d in detections)


# ---------------------------------------------------------------------------
# Backtick raw strings are also scanned
# ---------------------------------------------------------------------------


def test_backtick_raw_string_literals_are_also_scanned() -> None:
    """Multi-line backtick raw strings contribute detections."""

    files = {
        "internal/repository/orders_repo.go": (
            "package repository\n"
            "\n"
            "const q = `\n"
            "SELECT id\n"
            "FROM ORDERS\n"
            "WHERE id = :1\n"
            "`\n"
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "ORDERS" and d.access_mode is DatabaseAccessMode.READ
        for d in detections
    )


# ---------------------------------------------------------------------------
# fmt.Sprintf schema-placeholder normalization
# ---------------------------------------------------------------------------
#
# Operator repos build SQL strings via ``fmt.Sprintf`` and inject the
# runtime schema name with a ``%v`` (or ``%s``, ``%d``, positional
# ``%[1]v``) placeholder. The static analyzer cannot know what the
# placeholder evaluates to, but the table or procedure name *after*
# the dot is statically present in the source — the detector strips
# the placeholder prefix so the downstream SQL regex can match the
# unqualified name.


def test_fmt_v_schema_prefix_on_select_from_is_stripped() -> None:
    """``FROM %v.IIB_PAYMENT_CONFIRMED`` → ``IIB_PAYMENT_CONFIRMED`` (READ)."""

    files = {
        "internal/repository/repaymentservice_repo.go": (
            "package repository\n"
            'const q = fmt.Sprintf("SELECT SUM(PAIDAMOUNT) PAID_DEDUCTION '
            'FROM %v.IIB_PAYMENT_CONFIRMED WHERE CONTRACTNO = :1", schema)\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "IIB_PAYMENT_CONFIRMED"
        and d.access_mode is DatabaseAccessMode.READ
        for d in detections
    )


def test_fmt_v_schema_prefix_on_insert_is_stripped_and_emits_write() -> None:
    """``INSERT INTO %v.IIB_PAYMENT_CONFIRMED`` → WRITE on the bare name."""

    files = {
        "internal/repository/repaymentservice_repo.go": (
            "package repository\n"
            'const q = fmt.Sprintf("INSERT INTO %v.IIB_PAYMENT_CONFIRMED '
            '(TRANSID, AMOUNT) VALUES(:transid, :amt)", schema)\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "IIB_PAYMENT_CONFIRMED"
        and d.access_mode is DatabaseAccessMode.WRITE
        for d in detections
    )


def test_fmt_v_schema_prefix_on_begin_proc_is_stripped() -> None:
    """``BEGIN %v.GETEARLYLOANS_REP(...); END;`` → ``GETEARLYLOANS_REP`` (UNKNOWN)."""

    files = {
        "internal/repository/repaymentservice_repo.go": (
            "package repository\n"
            'const q = fmt.Sprintf("BEGIN %v.GETEARLYLOANS_REP(:p_contractno, '
            ':p_nid, :p_loan); END;", schema)\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "GETEARLYLOANS_REP"
        and d.access_mode is DatabaseAccessMode.UNKNOWN
        for d in detections
    )


@pytest.mark.parametrize(
    "placeholder",
    [
        "%v",       # default formatter
        "%s",       # string
        "%d",       # integer (rare for schema names but valid syntactically)
        "%[1]v",    # positional / indexed
        "%-10s",    # left-aligned width
        "%+v",      # plus-flag formatter
    ],
)
def test_various_fmt_placeholder_shapes_are_stripped(placeholder: str) -> None:
    """All common ``fmt`` verbs followed by ``.IDENT`` are stripped."""

    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            f'const q = fmt.Sprintf("SELECT id FROM {placeholder}.WIDGETS '
            'WHERE id = :1", schema)\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "WIDGETS"
        and d.access_mode is DatabaseAccessMode.READ
        for d in detections
    ), f"placeholder {placeholder!r} not stripped"


def test_bare_value_placeholder_without_dot_identifier_is_left_alone() -> None:
    """``WHERE id = %v`` (no trailing ``.IDENT``) must NOT be stripped.

    Stripping a bare value placeholder would merge adjacent tokens
    and create spurious matches. The regex anchors on a following
    ``.<identifier-start>`` precisely to avoid that.
    """
    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            'const q = fmt.Sprintf("SELECT id FROM WIDGETS WHERE id = %v", id)\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    # ``WIDGETS`` is still detected (it has no fmt prefix in the
    # first place), and nothing is mangled around the trailing ``%v``.
    assert any(d.table_name == "WIDGETS" for d in detections)
    # No mangled token from merging ``%v`` with whatever follows.
    assert all(
        "%" not in d.table_name and "v" != d.table_name.lower()
        for d in detections
    )


# ---------------------------------------------------------------------------
# Stored-procedure case-relaxed admission
# ---------------------------------------------------------------------------
#
# SQL is case-insensitive, and some operator teams declare stored
# procedures in lower- or mixed-case (e.g.
# ``mulcastrans.sp_outstanding_amt_chk``). The
# ``BEGIN <name>(...); END;`` / ``CALL <name>(...)`` call shapes are
# specific enough that a relaxed admission for procedure names does
# not meaningfully increase the false-positive surface: ordinary
# ``SELECT … FROM`` matches keep going through the stricter uppercase
# rule that protects table-name extraction.


def test_lowercase_schema_qualified_stored_procedure_is_accepted() -> None:
    """``BEGIN mulcastrans.sp_outstanding_amt_chk(...); END;`` is detected.

    The captured name preserves source casing exactly so downstream
    consumers see the proc as declared in the Go source.
    """

    files = {
        "internal/repository/loan_check_repo.go": (
            "package repository\n"
            'const q = "BEGIN mulcastrans.sp_outstanding_amt_chk'
            "(:sNationalID, :sAttribute02, :Out_OutstandingAmount, "
            ':Out_RequestedLoanAmount); END;"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    matching = [
        d
        for d in detections
        if d.table_name == "mulcastrans.sp_outstanding_amt_chk"
    ]
    assert len(matching) == 1
    assert matching[0].access_mode is DatabaseAccessMode.UNKNOWN


def test_mixed_case_stored_procedure_via_call_statement_is_accepted() -> None:
    """``CALL Mulcastrans.Sp_Update_Status(:1);`` survives admission.

    The relaxed procedure-name shape ``[A-Za-z_][A-Za-z0-9_]*`` admits
    mixed-case identifiers; the casing in the detection round-trips
    from the source verbatim.
    """

    files = {
        "internal/repository/loan_repo.go": (
            "package repository\n"
            'const q = "CALL Mulcastrans.Sp_Update_Status(:1);"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "Mulcastrans.Sp_Update_Status" for d in detections
    )


def test_unqualified_lowercase_stored_procedure_is_accepted() -> None:
    """Bare ``BEGIN sp_recalculate(); END;`` with a lowercase name is detected."""

    files = {
        "internal/repository/loan_repo.go": (
            "package repository\n"
            'const q = "BEGIN sp_recalculate(); END;"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "sp_recalculate"
        and d.access_mode is DatabaseAccessMode.UNKNOWN
        for d in detections
    )


def test_lowercase_stored_procedure_with_fmt_schema_prefix_is_detected() -> None:
    """``BEGIN %v.sp_outstanding_amt_chk(...); END;`` round-trips after fmt-stripping."""

    files = {
        "internal/repository/loan_repo.go": (
            "package repository\n"
            'const q = fmt.Sprintf("BEGIN %v.sp_outstanding_amt_chk'
            '(:p_nid, :p_attr); END;", schema)\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(
        d.table_name == "sp_outstanding_amt_chk"
        and d.access_mode is DatabaseAccessMode.UNKNOWN
        for d in detections
    )


def test_lowercase_table_in_select_from_is_still_rejected() -> None:
    """The case-relaxed admission applies to procedures only, not tables.

    A lowercase ``users`` after ``FROM`` is still treated as a Go
    identifier (struct field, variable, …) rather than a real table,
    because the ``SELECT … FROM`` shape attracts many more
    non-table matches than the PL/SQL call grammar does.
    """

    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            'const q = "SELECT id FROM users WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert all(d.table_name != "users" for d in detections)


def test_stored_procedure_named_after_a_stop_word_is_still_rejected() -> None:
    """The stop-word check still applies even with the relaxed shape.

    A literal ``BEGIN db(); END;`` (which is unlikely in production
    SQL but trivial to construct in test fixtures) would be admitted
    by the shape regex; the stop-word list keeps it out.
    """

    files = {
        "internal/repository/widget_repo.go": (
            "package repository\n"
            'const q = "BEGIN db(); END;"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert all(d.table_name.lower() != "db" for d in detections)


# ---------------------------------------------------------------------------
# Operator-tuning literal blacklist
# ---------------------------------------------------------------------------
#
# Some Go files under ``internal/repository/`` carry string literals
# that are neither SQL nor stored-procedure call text but happen to
# embed keywords the SQL extractor would otherwise scan. The
# detector skips any literal whose content contains a blacklisted
# phrase verbatim. The phrase set is small and grows on demand.


def test_blacklisted_phrase_literal_emits_no_detection() -> None:
    """A literal carrying the GitLab auto-close-issues phrase is skipped.

    The phrase ``"Automatically close issues from merge requests"``
    contains the substring ``from merge`` which the SQL extractor's
    FROM regex would otherwise capture as a candidate table name.
    The literal-level blacklist drops the whole literal before any
    regex scan, so no detection is emitted regardless of what comes
    after the phrase.
    """

    files = {
        "internal/repository/issue_repo.go": (
            "package repository\n"
            'const desc = "Automatically close issues from merge requests, '
            'branches, or commits in this project."\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert detections == []


def test_blacklist_does_not_affect_other_literals_in_same_file() -> None:
    """One blacklisted literal in a file doesn't suppress neighbouring SQL.

    The blacklist operates per-literal, not per-file, so a file
    can contain both a noise literal (skipped) and a real SQL
    literal (scanned). The real SQL literal must still produce a
    detection.
    """

    files = {
        "internal/repository/mixed_repo.go": (
            "package repository\n"
            'const noise = "Automatically close issues from merge requests"\n'
            'const real = "SELECT id FROM IIB_PAYMENT_CONFIRMED WHERE id = :1"\n'
        ),
    }

    detections, _ = detect_go_database_tables(_repo(files), {})

    assert any(d.table_name == "IIB_PAYMENT_CONFIRMED" for d in detections)
    # And no leakage from the blacklisted literal.
    for d in detections:
        assert "merge" not in d.table_name.lower()
        assert "request" not in d.table_name.lower()

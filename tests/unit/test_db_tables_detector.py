"""Unit tests for ``project_analyzer.db_tables.detect_database_tables``.

These tests pin down the *acceptance* side of the database-table detector:
each test exercises one of the four detection sources documented in the
module's docstring (raw SQL, ORM model declarations, Alembic migration
ops, ORM query patterns) and the aggregation invariant that mixed read +
write observations for a single table coalesce into a single entry whose
``access_mode`` is :data:`DatabaseAccessMode.READ_WRITE` (Requirement 6.3).

Implements Requirements 6.1, 6.2, 6.3, 6.4.
"""

from __future__ import annotations

import pytest

from project_knowledge_mcp.models import (
    DatabaseAccessMode,
    RepositoryContents,
)
from project_knowledge_mcp.project_analyzer.db_tables import detect_database_tables

pytestmark = pytest.mark.unit


def _repo(files: dict[str, str]) -> RepositoryContents:
    """Build a minimal :class:`RepositoryContents` for the given file map."""

    return RepositoryContents(
        gitlab_project_id=1,
        commit_sha="deadbeef",
        files=files,
    )


# ---------------------------------------------------------------------------
# Requirement 6.4: empty-input behavior
# ---------------------------------------------------------------------------


def test_empty_repository_returns_empty_list() -> None:
    """No files → empty list (not ``None``). Implements Requirement 6.4."""

    assert detect_database_tables(_repo({})) == []


def test_repository_without_database_references_returns_empty_list() -> None:
    """Files without any DB markers → empty list. Implements Requirement 6.4."""

    files = {
        "src/main.py": "def main():\n    return 'hello'\n",
        "README.md": "# A pure compute service\n",
    }
    assert detect_database_tables(_repo(files)) == []


# ---------------------------------------------------------------------------
# Raw SQL detection (Requirements 6.1, 6.2)
# ---------------------------------------------------------------------------


def test_raw_sql_select_from_is_read() -> None:
    files = {"q.sql": "SELECT id, name FROM users WHERE id = 1;"}

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "users"
    assert deps[0].access_mode == DatabaseAccessMode.READ
    assert deps[0].source_locations[0].path == "q.sql"


def test_raw_sql_insert_update_delete_create_are_writes() -> None:
    files = {
        "writes.sql": (
            "INSERT INTO orders (id) VALUES (1);\n"
            "UPDATE orders SET status = 'paid' WHERE id = 1;\n"
            "DELETE FROM orders WHERE id = 2;\n"
            "CREATE TABLE shipments (id INTEGER PRIMARY KEY);\n"
        )
    }

    deps = {d.table_name: d for d in detect_database_tables(_repo(files))}

    assert deps["orders"].access_mode == DatabaseAccessMode.WRITE
    assert deps["shipments"].access_mode == DatabaseAccessMode.WRITE


def test_delete_from_does_not_double_count_as_read() -> None:
    """``DELETE FROM users`` must yield one WRITE entry, not WRITE + READ."""

    files = {"q.sql": "DELETE FROM users WHERE id = 1;"}

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "users"
    assert deps[0].access_mode == DatabaseAccessMode.WRITE


def test_raw_sql_is_case_insensitive() -> None:
    files = {
        "lower.sql": "select * from accounts;",
        "mixed.sql": "Insert Into accounts (id) values (1);",
    }

    deps = {d.table_name: d for d in detect_database_tables(_repo(files))}

    assert deps["accounts"].access_mode == DatabaseAccessMode.READ_WRITE


def test_raw_sql_handles_quoted_and_schema_qualified_identifiers() -> None:
    files = {
        "q.sql": (
            'SELECT * FROM "public"."users";\n'
            "SELECT * FROM `mysql_users`;\n"
            "SELECT * FROM [dbo].[ms_users];\n"
            "SELECT * FROM auth.profiles;\n"
        )
    }

    names = {d.table_name for d in detect_database_tables(_repo(files))}

    assert names == {"users", "mysql_users", "ms_users", "profiles"}


def test_python_from_import_lines_are_not_treated_as_sql_from() -> None:
    """``from app import x`` must NOT register ``app`` as a read table."""

    files = {
        "src/app.py": (
            "from collections import defaultdict\n"
            "from typing import Final\n"
            "VALUE = 1\n"
        )
    }

    assert detect_database_tables(_repo(files)) == []


# ---------------------------------------------------------------------------
# ORM model declarations (Requirement 6.1)
# ---------------------------------------------------------------------------


def test_sqlalchemy_tablename_declaration_is_write() -> None:
    files = {
        "models.py": (
            "class User(Base):\n"
            '    __tablename__ = "users"\n'
            "    id = Column(Integer, primary_key=True)\n"
        )
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "users"
    assert deps[0].access_mode == DatabaseAccessMode.WRITE


def test_django_db_table_declaration_is_write() -> None:
    files = {
        "models.py": (
            "class Order(models.Model):\n"
            "    id = models.IntegerField(primary_key=True)\n"
            "    class Meta:\n"
            '        db_table = "shop_orders"\n'
        )
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "shop_orders"
    assert deps[0].access_mode == DatabaseAccessMode.WRITE


def test_activerecord_table_name_declaration_is_write() -> None:
    files = {
        "app/models/customer.rb": (
            "class Customer < ApplicationRecord\n"
            '  self.table_name = "legacy_customers"\n'
            "end\n"
        )
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "legacy_customers"
    assert deps[0].access_mode == DatabaseAccessMode.WRITE


# ---------------------------------------------------------------------------
# Alembic migration ops (Requirement 6.1)
# ---------------------------------------------------------------------------


def test_alembic_create_drop_alter_table_are_writes() -> None:
    files = {
        "migrations/001_init.py": (
            "def upgrade():\n"
            '    op.create_table("users", sa.Column("id", sa.Integer))\n'
            '    op.alter_table("users", sa.Column("name", sa.String))\n'
            '    op.drop_table("legacy_users")\n'
        )
    }

    deps = {d.table_name: d for d in detect_database_tables(_repo(files))}

    assert deps["users"].access_mode == DatabaseAccessMode.WRITE
    assert deps["legacy_users"].access_mode == DatabaseAccessMode.WRITE


def test_alembic_rename_table_yields_writes_for_both_names() -> None:
    files = {
        "migrations/002_rename.py": (
            'def upgrade():\n    op.rename_table("old_users", "new_users")\n'
        )
    }

    deps = {d.table_name: d for d in detect_database_tables(_repo(files))}

    assert deps["old_users"].access_mode == DatabaseAccessMode.WRITE
    assert deps["new_users"].access_mode == DatabaseAccessMode.WRITE


# ---------------------------------------------------------------------------
# ORM query methods (Requirement 6.1)
# ---------------------------------------------------------------------------


def test_sqlalchemy_query_method_is_read_when_combined_with_model_decl() -> None:
    files = {
        "models.py": (
            "class User(Base):\n"
            '    __tablename__ = "users"\n'
        ),
        "views.py": (
            "from .models import User\n"
            "def get_all():\n"
            "    return User.query.all()\n"
        ),
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    # Model declaration → write; query → read; aggregated → read_write.
    assert deps[0].table_name == "users"
    assert deps[0].access_mode == DatabaseAccessMode.READ_WRITE


def test_django_objects_create_is_write() -> None:
    files = {
        "models.py": (
            "class Order(models.Model):\n"
            "    id = models.IntegerField()\n"
            "    class Meta:\n"
            '        db_table = "orders"\n'
        ),
        "service.py": (
            "from .models import Order\n"
            "def make():\n"
            "    Order.objects.create(id=1)\n"
        ),
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "orders"
    # Model decl write + objects.create write → still WRITE (no read seen).
    assert deps[0].access_mode == DatabaseAccessMode.WRITE


def test_django_objects_filter_is_read_yielding_read_write_aggregate() -> None:
    files = {
        "models.py": (
            "class Order(models.Model):\n"
            "    class Meta:\n"
            '        db_table = "orders"\n'
        ),
        "service.py": (
            "from .models import Order\n"
            "def list_paid():\n"
            "    return Order.objects.filter(status='paid')\n"
        ),
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].access_mode == DatabaseAccessMode.READ_WRITE


def test_session_query_and_session_add_for_known_class() -> None:
    files = {
        "models.py": (
            "class Account(Base):\n"
            '    __tablename__ = "accounts"\n'
        ),
        "service.py": (
            "def create(session):\n"
            "    session.add(Account(id=1))\n"
            "def fetch(session):\n"
            "    return session.query(Account).all()\n"
        ),
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "accounts"
    assert deps[0].access_mode == DatabaseAccessMode.READ_WRITE


# ---------------------------------------------------------------------------
# Aggregation invariants (Requirements 6.2, 6.3)
# ---------------------------------------------------------------------------


def test_mixed_read_and_write_aggregates_to_read_write() -> None:
    """Property 8's central invariant: read + write → read_write."""

    files = {
        "read.sql": "SELECT * FROM events;",
        "write.sql": "INSERT INTO events (id) VALUES (1);",
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "events"
    assert deps[0].access_mode == DatabaseAccessMode.READ_WRITE
    # Both source locations are present and deduplicated.
    paths = sorted({loc.path for loc in deps[0].source_locations})
    assert paths == ["read.sql", "write.sql"]


def test_only_reads_aggregates_to_read_only() -> None:
    files = {
        "a.sql": "SELECT * FROM events;",
        "b.sql": "SELECT id FROM events WHERE id > 1;",
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].access_mode == DatabaseAccessMode.READ
    assert len(deps[0].source_locations) == 2


def test_only_writes_aggregates_to_write_only() -> None:
    files = {
        "a.sql": "INSERT INTO events (id) VALUES (1);",
        "b.sql": "UPDATE events SET status = 'x' WHERE id = 1;",
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].access_mode == DatabaseAccessMode.WRITE
    assert len(deps[0].source_locations) == 2


def test_at_most_one_entry_per_table_name() -> None:
    """Requirement 6.3: at most one entry per table_name even with many sites."""

    files = {
        f"f{i}.sql": "SELECT * FROM widgets;" for i in range(5)
    } | {
        f"w{i}.sql": "INSERT INTO widgets (id) VALUES (1);" for i in range(3)
    }

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "widgets"
    assert deps[0].access_mode == DatabaseAccessMode.READ_WRITE
    # Eight distinct source locations.
    assert len({(loc.path, loc.line) for loc in deps[0].source_locations}) == 8


def test_results_are_sorted_by_table_name_for_determinism() -> None:
    files = {
        "a.sql": "SELECT * FROM zeta; SELECT * FROM alpha; SELECT * FROM mu;",
    }

    deps = detect_database_tables(_repo(files))

    assert [d.table_name for d in deps] == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# Go-analyzer-support task 1.2: MERGE keyword routes to WRITE
# (Requirements 9.5)
# ---------------------------------------------------------------------------


def test_raw_sql_merge_into_is_write() -> None:
    """``MERGE INTO <table>`` is classified as WRITE (Requirement 9.5)."""

    files = {
        "upsert.sql": (
            "MERGE INTO accounts AS dst\n"
            "USING staging_accounts AS src ON dst.id = src.id\n"
            "WHEN MATCHED THEN UPDATE SET dst.balance = src.balance;\n"
        ),
    }

    deps = {d.table_name: d for d in detect_database_tables(_repo(files))}

    assert "accounts" in deps
    assert deps["accounts"].access_mode == DatabaseAccessMode.WRITE


def test_raw_sql_merge_is_case_insensitive() -> None:
    files = {"q.sql": "merge into Orders using src on src.id = Orders.id;"}

    deps = detect_database_tables(_repo(files))

    assert len(deps) == 1
    assert deps[0].table_name == "Orders"
    assert deps[0].access_mode == DatabaseAccessMode.WRITE


# ---------------------------------------------------------------------------
# Go-analyzer-support task 1.2: schema-preserving extractor
# (Requirements 9.5, 9.6)
# ---------------------------------------------------------------------------


def test_extract_table_references_preserving_schema_select() -> None:
    """Schema-qualified ``SELECT ... FROM`` preserves ``<schema>.<table>``."""

    from project_knowledge_mcp.project_analyzer.db_tables import (
        extract_table_references_preserving_schema,
    )

    pairs = extract_table_references_preserving_schema(
        "SELECT * FROM APP_WS.IIB_PAYMENT_CONFIRMED WHERE id = :1"
    )

    assert pairs == [("APP_WS.IIB_PAYMENT_CONFIRMED", DatabaseAccessMode.READ)]


def test_extract_table_references_preserving_schema_insert_update_delete_merge() -> None:
    """Each write keyword surfaces with its schema-qualified name intact."""

    from project_knowledge_mcp.project_analyzer.db_tables import (
        extract_table_references_preserving_schema,
    )

    sql = (
        "INSERT INTO APP_WS.IIB_PAYMENT (id) VALUES (:1);"
        "UPDATE APP_WS.IIB_PAYMENT SET status = :1 WHERE id = :2;"
        "DELETE FROM APP_WS.IIB_PAYMENT WHERE id = :1;"
        "MERGE INTO APP_WS.IIB_PAYMENT USING dual ON (1=1);"
    )

    pairs = extract_table_references_preserving_schema(sql)

    assert all(name == "APP_WS.IIB_PAYMENT" for name, _ in pairs)
    modes = {mode for _, mode in pairs}
    assert modes == {DatabaseAccessMode.WRITE}
    # Each of the four write patterns produced exactly one match (the
    # ``DELETE FROM`` write match suppresses any ``FROM`` read match for
    # the same span).
    assert len(pairs) == 4


def test_extract_table_references_preserving_schema_quoted_forms() -> None:
    """Quote/bracket/backtick forms are stripped while the dot is preserved."""

    from project_knowledge_mcp.project_analyzer.db_tables import (
        extract_table_references_preserving_schema,
    )

    cases = [
        ('SELECT * FROM "public"."users";', "public.users"),
        ("SELECT * FROM `db`.`orders`;", "db.orders"),
        ("SELECT * FROM [dbo].[ms_users];", "dbo.ms_users"),
        ("SELECT * FROM 'sch'.'tbl';", "sch.tbl"),
        ("SELECT * FROM auth.profiles;", "auth.profiles"),
        ("SELECT * FROM bare_table;", "bare_table"),
    ]
    for sql, expected_name in cases:
        pairs = extract_table_references_preserving_schema(sql)
        assert pairs == [(expected_name, DatabaseAccessMode.READ)], (
            f"sql={sql!r} produced {pairs!r}, expected {expected_name!r}"
        )


def test_extract_table_references_preserving_schema_no_match_returns_empty() -> None:
    """A SQL string that matches no keyword pattern yields an empty list."""

    from project_knowledge_mcp.project_analyzer.db_tables import (
        extract_table_references_preserving_schema,
    )

    assert extract_table_references_preserving_schema("BEGIN; COMMIT;") == []
    assert extract_table_references_preserving_schema("") == []


def test_extract_table_references_preserving_schema_does_not_double_count() -> None:
    """``DELETE FROM <t>`` yields one WRITE entry, not WRITE + READ."""

    from project_knowledge_mcp.project_analyzer.db_tables import (
        extract_table_references_preserving_schema,
    )

    pairs = extract_table_references_preserving_schema(
        "DELETE FROM APP_WS.LOG WHERE id = :1"
    )

    assert pairs == [("APP_WS.LOG", DatabaseAccessMode.WRITE)]


# ---------------------------------------------------------------------------
# Go-analyzer-support task 1.2: UNKNOWN coalescing
# (Requirement 9.8)
# ---------------------------------------------------------------------------


def test_aggregate_unknown_alone_is_unknown() -> None:
    """When UNKNOWN is the sole observation the entry is UNKNOWN."""

    from project_knowledge_mcp.models import SourceLocation
    from project_knowledge_mcp.project_analyzer.db_tables import _aggregate

    detections = [
        ("mystery_tbl", DatabaseAccessMode.UNKNOWN, SourceLocation(path="x.go", line=1)),
    ]

    deps = _aggregate(detections)

    assert len(deps) == 1
    assert deps[0].table_name == "mystery_tbl"
    assert deps[0].access_mode == DatabaseAccessMode.UNKNOWN


def test_aggregate_unknown_loses_to_read() -> None:
    """READ takes precedence over a co-observed UNKNOWN (Requirement 9.8)."""

    from project_knowledge_mcp.models import SourceLocation
    from project_knowledge_mcp.project_analyzer.db_tables import _aggregate

    detections = [
        ("tbl", DatabaseAccessMode.UNKNOWN, SourceLocation(path="a.go", line=1)),
        ("tbl", DatabaseAccessMode.READ, SourceLocation(path="b.go", line=2)),
    ]

    deps = _aggregate(detections)

    assert len(deps) == 1
    assert deps[0].access_mode == DatabaseAccessMode.READ


def test_aggregate_unknown_loses_to_write() -> None:
    """WRITE takes precedence over a co-observed UNKNOWN (Requirement 9.8)."""

    from project_knowledge_mcp.models import SourceLocation
    from project_knowledge_mcp.project_analyzer.db_tables import _aggregate

    detections = [
        ("tbl", DatabaseAccessMode.UNKNOWN, SourceLocation(path="a.go", line=1)),
        ("tbl", DatabaseAccessMode.WRITE, SourceLocation(path="b.go", line=2)),
    ]

    deps = _aggregate(detections)

    assert len(deps) == 1
    assert deps[0].access_mode == DatabaseAccessMode.WRITE


def test_aggregate_unknown_loses_to_read_write() -> None:
    """READ_WRITE takes precedence over a co-observed UNKNOWN (Requirement 9.8)."""

    from project_knowledge_mcp.models import SourceLocation
    from project_knowledge_mcp.project_analyzer.db_tables import _aggregate

    detections = [
        ("tbl", DatabaseAccessMode.READ_WRITE, SourceLocation(path="a.go", line=1)),
        ("tbl", DatabaseAccessMode.UNKNOWN, SourceLocation(path="b.go", line=2)),
    ]

    deps = _aggregate(detections)

    assert len(deps) == 1
    assert deps[0].access_mode == DatabaseAccessMode.READ_WRITE


def test_aggregate_unknown_with_read_and_write_yields_read_write() -> None:
    """READ + WRITE + UNKNOWN coalesces to READ_WRITE (Requirement 9.8)."""

    from project_knowledge_mcp.models import SourceLocation
    from project_knowledge_mcp.project_analyzer.db_tables import _aggregate

    detections = [
        ("tbl", DatabaseAccessMode.UNKNOWN, SourceLocation(path="a.go", line=1)),
        ("tbl", DatabaseAccessMode.READ, SourceLocation(path="b.go", line=2)),
        ("tbl", DatabaseAccessMode.WRITE, SourceLocation(path="c.go", line=3)),
    ]

    deps = _aggregate(detections)

    assert len(deps) == 1
    assert deps[0].access_mode == DatabaseAccessMode.READ_WRITE



# ---------------------------------------------------------------------------
# Documentation-file filter
# ---------------------------------------------------------------------------
#
# The raw-SQL extractor scans every text file in the snapshot for
# the ``\bFROM\s+<token>`` / ``\bINSERT\s+INTO\s+<token>`` /
# ``\bMERGE\s+INTO\s+<token>`` regex set. Files written in natural
# language routinely contain those keywords in prose ("the service
# reads FROM the orders table") and the GitLab default project
# template seeds every new empty repo with a README that contains
# the phrase "Automatically close issues from merge requests".
# Both cases used to produce spurious table names (``orders``,
# ``merge``) for projects with no real code; the documentation
# filter excludes them at the source.


def test_gitlab_default_readme_does_not_emit_spurious_table_names() -> None:
    """A README carrying the GitLab default template emits zero detections.

    Pins the operator-confirmed regression for project 1581
    (``disb-update-status``): the GitLab default README template
    contains the line ``[Automatically close issues from merge
    requests]`` which the case-insensitive ``\\bFROM\\s+<token>``
    regex would otherwise match against, surfacing ``merge`` as a
    spurious read-only table dependency. The documentation-file
    filter (``_is_documentation_file``) skips the README at the
    source so the regex never sees the boilerplate phrase.
    """
    files = {
        "README.md": (
            "# Disbursement update status service\n"
            "\n"
            "To make it easy for you to get started with GitLab, "
            "here's a list of recommended next steps.\n"
            "\n"
            "## Collaborate with your team\n"
            "\n"
            "- [ ] [Invite team members and collaborators]\n"
            "- [ ] [Create a new merge request]\n"
            "- [ ] [Automatically close issues from merge requests]\n"
        ),
    }

    assert detect_database_tables(_repo(files)) == []


def test_documentation_file_extensions_are_all_skipped() -> None:
    """``.md``, ``.markdown``, ``.rst``, ``.txt``, and ``.adoc`` are docs.

    Each suffix is exercised with a body containing a SQL keyword
    in prose so the test fails closed if the filter regresses.
    """
    files = {
        "README.md": "We read FROM users every minute.",
        "INSTALL.markdown": "Run INSERT INTO accounts (id) VALUES (1).",
        "docs/usage.rst": "DELETE FROM sessions when the user logs out.",
        "NOTES.txt": "UPDATE balance SET amount = 0 weekly.",
        "manual.adoc": "MERGE INTO totals USING staging ON ...",
    }

    assert detect_database_tables(_repo(files)) == []


def test_documentation_filter_is_case_insensitive_on_extension() -> None:
    """``README.MD`` and ``Readme.Markdown`` are also skipped."""
    files = {
        "README.MD": "FROM users where ...",
        "Readme.Markdown": "INSERT INTO orders ...",
    }

    assert detect_database_tables(_repo(files)) == []


def test_documentation_filter_does_not_affect_source_files() -> None:
    """A real SQL file alongside docs still produces detections.

    The filter is per-file: a ``.sql`` file scanned next to a
    blacklisted README still surfaces every table it references.
    """
    files = {
        "README.md": (
            "Loaded FROM the GitLab merge-request template.\n"
        ),
        "queries/orders.sql": "SELECT id FROM ORDERS WHERE status = 'open';",
    }

    deps = detect_database_tables(_repo(files))

    # The README contributed nothing; the .sql file contributed
    # exactly one read detection on ``ORDERS``.
    assert len(deps) == 1
    assert deps[0].table_name == "ORDERS"
    assert deps[0].access_mode is DatabaseAccessMode.READ

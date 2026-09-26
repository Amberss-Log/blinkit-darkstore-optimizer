"""Tests for the schema-rebuild safety guard.

The rebuild drops tables with generic names, so pointing it at the wrong
database could destroy someone else's data. These tests run against throwaway
SQLite files, so they need no database server -- but the guard itself is
backend-agnostic and protects a MySQL server in exactly the same way.

The test that matters most is the first one. It is not enough for the guard to
raise an error: it has to raise *before* anything is dropped. So the test
checks that the unrelated table still exists afterwards, with its rows intact.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from src.db import PROJECT_TABLES, ForeignTablesError, apply_schema


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'guard_test.db'}", future=True)
    yield eng
    eng.dispose()


def test_refuses_database_with_foreign_tables_and_drops_nothing(engine):
    # Someone else's table, plus one that shares a name with ours -- the
    # worst case, since that is exactly the table the rebuild would drop.
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT)"))
        conn.execute(text("INSERT INTO customers (name) VALUES ('Asha'), ('Ravi')"))
        conn.execute(text("CREATE TABLE localities (id INTEGER PRIMARY KEY, label TEXT)"))
        conn.execute(text("INSERT INTO localities (label) VALUES ('not ours')"))

    with pytest.raises(ForeignTablesError) as excinfo:
        apply_schema(engine)

    assert "customers" in str(excinfo.value)

    # Nothing was touched: both tables survive with their original rows,
    # including the one whose name collides with the project's.
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM customers")).scalar() == 2
        label = conn.execute(text("SELECT label FROM localities")).scalar()
        assert label == "not ours"


def test_error_message_never_contains_a_password(tmp_path):
    """The message names the database, but must not leak credentials.

    Uses a SQLite URL, so there is no real password to leak -- the check is
    that the message goes through the password-stripping helper rather than
    printing engine.url raw, which would expose it on MySQL.
    """
    eng = create_engine(f"sqlite:///{tmp_path / 'leak.db'}", future=True)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE payroll (id INTEGER PRIMARY KEY)"))

    with pytest.raises(ForeignTablesError) as excinfo:
        apply_schema(eng)

    message = str(excinfo.value)
    assert str(tmp_path) not in message          # no absolute path either
    assert "sqlite:///data/" in message
    eng.dispose()


def test_empty_database_is_built_normally(engine):
    apply_schema(engine)

    tables = {t for t in inspect(engine).get_table_names() if not t.startswith("sqlite_")}
    assert tables == PROJECT_TABLES


def test_rebuild_over_our_own_tables_is_allowed(engine):
    """Re-running generate_data.py must keep working.

    After the first build the database holds exactly the project's tables --
    plus SQLite's internal sqlite_sequence, created automatically by
    AUTOINCREMENT, which the guard has to recognise as harmless.
    """
    apply_schema(engine)

    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO localities (locality_name, latitude, longitude, "
            "population_density, avg_household_income_tier) "
            "VALUES ('Palasia', 22.72, 75.88, 25000, 'high')"
        ))

    # Confirm the internal table really exists, so this test is exercising the
    # case it claims to. Queried from sqlite_master directly: SQLAlchemy's
    # inspector already hides sqlite_* tables, so it would not show it.
    with engine.connect() as conn:
        internal = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        )).scalar()
    assert internal == "sqlite_sequence"

    apply_schema(engine)          # must not raise

    with engine.connect() as conn:
        # And it genuinely rebuilt: the old row is gone.
        assert conn.execute(text("SELECT COUNT(*) FROM localities")).scalar() == 0

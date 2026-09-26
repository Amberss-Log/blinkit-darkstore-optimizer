"""Database connection and dialect handling.

The project targets MySQL 8 but falls back to SQLite so it runs anywhere with
no server installed. Everything that differs between the two backends is
isolated in this module, so the rest of the code never has to care which one is
in use.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine, make_url

# Project root: this file is at <root>/src/db.py
ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"

# Read .env if present. Values already in the real environment win, so you can
# override on the command line without editing the file.
load_dotenv(ROOT / ".env", override=False)

SQLITE_PATH = DATA_DIR / "blinkit_sim.db"


def get_engine(echo: bool = False) -> Engine:
    """Return a SQLAlchemy engine for whichever backend is configured.

    Reads DATABASE_URL from the environment. If it is unset or blank, falls
    back to a local SQLite file. The URL itself is never logged — it contains
    a password.
    """
    url = (os.getenv("DATABASE_URL") or "").strip()

    if not url:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{SQLITE_PATH}", echo=echo, future=True)
        _enable_sqlite_foreign_keys(engine)
        return engine

    return create_engine(url, echo=echo, future=True, pool_pre_ping=True)


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    """Turn on foreign key enforcement for every SQLite connection.

    SQLite ships with foreign keys DISABLED for backwards compatibility. Without
    this, every FOREIGN KEY clause in the schema is silently decorative — you
    can insert demand rows pointing at localities that do not exist. The pragma
    is per-connection, so it has to be set on connect, not once at startup.
    """

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()


def is_mysql(engine: Engine) -> bool:
    return engine.dialect.name == "mysql"


def backend_name(engine: Engine) -> str:
    """Human-readable backend label, safe to print (no credentials)."""
    return "MySQL" if is_mysql(engine) else "SQLite"


def describe_connection(engine: Engine) -> str:
    """Connection summary that is safe to print in a public notebook.

    Two things are stripped. SQLAlchemy's render_as_string() hides the password
    by default, which handles the MySQL case. The SQLite case needs separate
    handling: the engine URL holds an absolute path, so printing it verbatim
    would publish the directory layout of whichever machine ran the notebook.
    Neither is a serious vulnerability, but neither belongs in a repository
    strangers will read, and committed notebook output is easy to forget about.
    """
    url = make_url(str(engine.url))
    if url.get_backend_name() == "sqlite":
        name = Path(url.database).name if url.database else ":memory:"
        return f"sqlite:///data/{name}"
    return url.render_as_string(hide_password=True)


# ---------------------------------------------------------------------------
# Dialect shims
# ---------------------------------------------------------------------------
# MySQL and SQLite agree on the parts that matter here — CTEs, window
# functions, CASE WHEN, joins are all identical. Date handling is the one real
# divergence, so it lives behind these helpers.


def dow_expr(engine: Engine, column: str) -> str:
    """SQL expression returning day of week as 1=Sunday .. 7=Saturday.

    MySQL has DAYOFWEEK() which already uses that convention. SQLite has
    strftime('%w') which returns 0=Sunday..6=Saturday as a *string*, so it
    needs a cast and a +1 to line up.
    """
    if is_mysql(engine):
        return f"DAYOFWEEK({column})"
    return f"(CAST(strftime('%w', {column}) AS INTEGER) + 1)"


def is_weekend_expr(engine: Engine, column: str) -> str:
    """SQL expression that is true for Saturday and Sunday.

    With the 1=Sunday..7=Saturday convention, the weekend is 1 and 7.
    """
    dow = dow_expr(engine, column)
    return f"{dow} IN (1, 7)"


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------


def schema_path(engine: Engine) -> Path:
    return SQL_DIR / ("schema_mysql.sql" if is_mysql(engine) else "schema_sqlite.sql")


def _split_statements(script: str) -> list[str]:
    """Split a .sql file into individual statements.

    Naive semicolon splitting, which is fine because our schema files contain
    no stored procedures, triggers, or semicolons inside string literals.
    Comment lines are stripped first so a `--` comment containing a semicolon
    cannot break the split.
    """
    no_comments = re.sub(r"^\s*--.*$", "", script, flags=re.MULTILINE)
    return [stmt.strip() for stmt in no_comments.split(";") if stmt.strip()]


# The only tables this project ever creates. The schema rebuild is allowed to
# touch these and nothing else.
PROJECT_TABLES = frozenset({"localities", "demand", "candidate_sites", "existing_stores"})


class ForeignTablesError(RuntimeError):
    """Raised when the target database holds tables this project did not create."""


def check_database_is_ours(engine: Engine) -> None:
    """Refuse to continue if the target database holds anything but our tables.

    The rebuild drops tables called `localities` and `demand` -- generic enough
    names that some unrelated application could easily have its own. On MySQL
    it also switches foreign key checks off first, removing the one safeguard
    that might otherwise have blocked the drop. So if DATABASE_URL ever points
    at the wrong database, through a typo or a URL copied from another project,
    the rebuild would silently destroy someone else's data.

    This check runs before anything is dropped. An empty database, or one
    holding only this project's four tables, passes -- so first runs and
    re-runs both work. Anything else stops the run with nothing touched.

    Engine-internal tables are ignored: SQLite creates `sqlite_sequence` on its
    own to track AUTOINCREMENT counters, and it is not user data.
    """
    existing = {
        name for name in inspect(engine).get_table_names()
        if not name.startswith("sqlite_")
    }
    foreign = sorted(existing - PROJECT_TABLES)
    if foreign:
        raise ForeignTablesError(
            f"Refusing to rebuild the schema: the target database "
            f"({describe_connection(engine)}) contains tables this project did "
            f"not create: {', '.join(foreign)}.\n"
            f"Rebuilding would drop any table named "
            f"{', '.join(sorted(PROJECT_TABLES))} in it. Point DATABASE_URL at a "
            f"dedicated, empty database -- for example one created with\n"
            f"    CREATE DATABASE blinkit_sim CHARACTER SET utf8mb4 "
            f"COLLATE utf8mb4_unicode_ci;"
        )


def apply_schema(engine: Engine) -> None:
    """Drop and recreate every table. Destructive by design.

    This is a synthetic dataset that is meant to be rebuilt from scratch with
    one command, so a clean slate each run is the correct behaviour. It would
    NOT be correct against a database holding real data -- which is why
    check_database_is_ours() runs first and refuses to proceed if the database
    holds anything the project did not put there.
    """
    check_database_is_ours(engine)

    script = schema_path(engine).read_text(encoding="utf-8")
    statements = _split_statements(script)

    with engine.begin() as conn:
        if is_mysql(engine):
            # Dropping parents before children trips foreign key checks. The
            # schema file already orders drops children-first, but this makes
            # the script robust if that order is ever edited.
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for statement in statements:
            conn.execute(text(statement))
        if is_mysql(engine):
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))


def ensure_database_exists(url_string: str) -> None:
    """Create the MySQL database if it is missing.

    Chicken-and-egg problem: DATABASE_URL names a database that may not exist
    yet, and you cannot connect to a database in order to create it. So we
    connect to the server with no database selected, issue CREATE DATABASE,
    and disconnect. No-op for SQLite, where the file is created on connect.
    """
    url = make_url(url_string)
    if url.get_backend_name() != "mysql" or not url.database:
        return

    server_url = url.set(database=None)
    server_engine = create_engine(server_url, future=True)
    try:
        with server_engine.begin() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{url.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
    finally:
        server_engine.dispose()


def load_query(name: str) -> str:
    """Read a .sql file from sql/queries/ and return it as a string.

    Queries live in files rather than inline strings so they can be opened in a
    SQL editor, and so the notebook can print the exact text it is about to run.
    """
    return (SQL_DIR / "queries" / name).read_text(encoding="utf-8")

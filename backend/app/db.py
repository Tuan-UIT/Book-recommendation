"""Small database adapter for PostgreSQL production and SQLite verification."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from flask import current_app, g


class DatabaseConnection:
    def __init__(self, raw: Any, *, sqlite: bool):
        self.raw = raw
        self.sqlite = sqlite

    def _sql(self, statement: str) -> str:
        return statement.replace("%s", "?") if self.sqlite else statement

    def execute(self, statement: str, parameters: Iterable[Any] = ()):
        return self.raw.execute(self._sql(statement), tuple(parameters))

    def executemany(self, statement: str, rows: Iterable[Iterable[Any]]):
        return self.raw.executemany(self._sql(statement), rows)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()


def connect(database_url: str) -> DatabaseConnection:
    if database_url.startswith("sqlite:///"):
        path = database_url.removeprefix("sqlite:///")
        raw = sqlite3.connect(path)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys = ON")
        return DatabaseConnection(raw, sqlite=True)
    if database_url.startswith(("postgresql://", "postgres://")):
        import psycopg
        from psycopg.rows import dict_row

        raw = psycopg.connect(database_url, row_factory=dict_row)
        return DatabaseConnection(raw, sqlite=False)
    raise ValueError("DATABASE_URL must use postgresql:// or sqlite:///")


def get_db() -> DatabaseConnection:
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE_URL"])
    return g.db


def close_db(_error: BaseException | None = None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    schema_name = "schema_sqlite.sql" if db.sqlite else "schema_postgresql.sql"
    schema = Path(__file__).with_name(schema_name).read_text(encoding="utf-8")
    if db.sqlite:
        db.raw.executescript(schema)
    else:
        db.execute(schema)
    db.commit()


def init_app(app) -> None:
    app.teardown_appcontext(close_db)

    @app.cli.command("init-db")
    def init_db_command() -> None:
        init_db()
        print("Initialized database schema.")

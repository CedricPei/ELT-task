"""Small warehouse interface; model SQL is validated before it reaches a driver."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp

NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,62}\Z")
FUNCTIONS = {
    "AND",
    "OR",
    "NOT",
    "COUNT",
    "SUM",
    "MAX",
    "MIN",
    "AVG",
    "COALESCE",
    "LOWER",
    "UPPER",
    "ABS",
    "ROUND",
    "NULLIF",
    "CAST",
    "TRY_CAST",
    "IF",
    "CASE",
    "LENGTH",
    "TRIM",
    "ROW_NUMBER",
    "EXISTS",
}


class ToolError(ValueError):
    """Invalid model input or SQL that is safe to return as tool feedback."""


def identifier(name: str) -> str:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ToolError("Expected a simple SQL identifier")
    return name


def checked_sql(sql: str) -> exp.Expression:
    if not isinstance(sql, str) or len(sql) > 12000:
        raise ToolError("SQL exceeds the 12000 character limit")
    try:
        statements = sqlglot.parse(sql, read="snowflake")
    except (sqlglot.errors.ParseError, sqlglot.errors.TokenError) as exc:
        raise ToolError("SQL could not be parsed") from exc
    if len(statements) != 1 or statements[0] is None:
        raise ToolError("Execute one SQL statement per action")
    tree = statements[0]
    if isinstance(tree, exp.Create):
        if tree.kind not in {"TABLE", "VIEW"} or not isinstance(tree.expression, exp.Query):
            raise ToolError("Only CREATE TABLE/VIEW AS SELECT is allowed")
        target = tree.this
        if not isinstance(target, exp.Table) or target.name.lower().startswith("raw_"):
            raise ToolError("Transformation cannot replace raw source tables")
        if tree.args.get("properties"):
            raise ToolError("Table properties are not allowed")
    elif not isinstance(tree, exp.Query):
        raise ToolError("Only SELECT and CREATE TABLE/VIEW AS SELECT are allowed")
    for node in tree.walk():
        if isinstance(node, exp.Table):
            if node.db or node.catalog or not isinstance(node.this, exp.Identifier):
                raise ToolError("Only unqualified tables in the rollout namespace are allowed")
            identifier(node.name)
        if isinstance(node, exp.Func) and node.sql_name() not in FUNCTIONS:
            raise ToolError(f"Function {node.sql_name()} is outside the supported SQL subset")
        if isinstance(node, (exp.Command, exp.Into, exp.DML, exp.DDL)) and node is not tree:
            raise ToolError("Nested commands are not allowed")
    return tree


class Warehouse(ABC):
    dialect: str

    @abstractmethod
    def execute(self, sql: str) -> pd.DataFrame: ...

    @abstractmethod
    def load(self, table: str, frame: pd.DataFrame) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def model_sql(self, sql: str) -> pd.DataFrame:
        tree = checked_sql(sql)
        try:
            return self.execute(tree.sql(dialect=self.dialect))
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", None)
            if code != sqlite3.SQLITE_ERROR:
                raise
            raise ToolError(str(exc)[:400]) from exc
        except Exception as exc:
            # Only compilation errors from validated model SQL become tool feedback.
            state = str(getattr(exc, "sqlstate", ""))
            if (type(exc).__name__ == "ProgrammingError" and state[:2] in {"42", "22", "0A"}
                    and state != "42501"):
                raise ToolError(str(getattr(exc, "msg", "SQL compilation failed"))[:400]) from exc
            raise


class SQLiteWarehouse(Warehouse):
    dialect = "sqlite"

    def __init__(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.set_progress_handler(self._progress, 10000)
        self.operations = 0

    def _progress(self) -> int:
        self.operations += 1
        return int(self.operations > 1000)

    def execute(self, sql: str) -> pd.DataFrame:
        self.operations = 0
        # SQLite lacks CREATE OR REPLACE TABLE; its isolated namespace is disposable.
        tree = sqlglot.parse_one(sql, read="sqlite")
        if isinstance(tree, exp.Create) and tree.args.get("replace"):
            name = identifier(tree.this.name)
            self.connection.execute(f'DROP {tree.kind} IF EXISTS "{name}"')
            tree.set("replace", False)
            sql = tree.sql(dialect="sqlite")
        cursor = self.connection.execute(sql)
        if cursor.description:
            return pd.DataFrame(cursor.fetchmany(2001), columns=[c[0] for c in cursor.description])
        return pd.DataFrame()

    def load(self, table: str, frame: pd.DataFrame) -> None:
        frame.to_sql(identifier(table), self.connection, if_exists="replace", index=False)

    def close(self) -> None:
        self.connection.close()


class SnowflakeWarehouse(Warehouse):
    dialect = "snowflake"

    def __init__(self, credential_path: Path, tag: str):
        import snowflake.connector

        config = json.loads(credential_path.read_text())
        config["session_parameters"].update(
            {
                "QUERY_TAG": f"elt-tinker/{tag}",
                "STATEMENT_TIMEOUT_IN_SECONDS": 60,
                "STATEMENT_QUEUED_TIMEOUT_IN_SECONDS": 15,
                "LOCK_TIMEOUT": 5,
            }
        )
        self.connection = snowflake.connector.connect(**config)
        self.namespace = "RUN_" + uuid.uuid4().hex.upper()
        self.query_ids: list[str] = []
        warehouse = self.execute("SHOW WAREHOUSES LIKE 'ELT_TINKER_WH'").iloc[0]
        monitor = self.execute("SHOW RESOURCE MONITORS LIKE 'ELT_TINKER_MONITOR'").iloc[0]
        quota = json.loads((credential_path.parents[1] / "config/project.json").read_text())[
            "warehouse_credit_quota"
        ]
        if (
            warehouse["resource_monitor"] != "ELT_TINKER_MONITOR"
            or warehouse["size"] != "X-Small"
            or not 0 < int(warehouse["auto_suspend"]) <= 60
            or not 0 < float(monitor["credit_quota"]) <= quota <= 3
            or monitor["frequency"] != "NEVER"
            or monitor["suspend_immediately_at"] != "100%"
            or float(monitor["remaining_credits"]) < 0.05
        ):
            self.connection.close()
            raise ValueError("Warehouse expense controls are missing or exhausted")
        self.execute(f"CREATE TRANSIENT SCHEMA {self.namespace} DATA_RETENTION_TIME_IN_DAYS = 0")
        self.execute(f"USE SCHEMA {self.namespace}")

    def execute(self, sql: str) -> pd.DataFrame:
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            if cursor.sfqid:
                self.query_ids.append(cursor.sfqid)
            if cursor.description:
                rows = cursor.fetchmany(2001)
                if len(rows) > 2000:
                    raise ToolError("Result exceeds the task's 2000 row limit")
                return pd.DataFrame(rows, columns=[c[0] for c in cursor.description])
            return pd.DataFrame()

    def load(self, table: str, frame: pd.DataFrame) -> None:
        table = identifier(table)
        columns = [identifier(c) for c in frame.columns]
        definitions = []
        rows = []
        numeric = [pd.api.types.is_integer_dtype(frame[c]) for c in columns]
        for column, integer in zip(columns, numeric, strict=True):
            definitions.append(f"{column} {'BIGINT' if integer else 'VARCHAR'}")
        for row in frame.itertuples(index=False, name=None):
            rows.append(
                tuple(
                    None if pd.isna(v) else int(v) if n else str(v)
                    for v, n in zip(row, numeric, strict=True)
                )
            )
        self.execute(f"CREATE OR REPLACE TRANSIENT TABLE {table} ({','.join(definitions)})")
        with self.connection.cursor() as cursor:
            cursor.executemany(
                f"INSERT INTO {table} VALUES ({','.join(['%s'] * len(columns))})", rows
            )
            if cursor.sfqid:
                self.query_ids.append(cursor.sfqid)

    def close(self) -> None:
        try:
            self.execute(f"DROP SCHEMA IF EXISTS {self.namespace} CASCADE")
        finally:
            self.connection.close()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import duckdb
import polars as pl

from .config import is_gcs_uri


def _quote_sql_literal(value: str) -> str:
    return str(value).replace("'", "''")


def _quote_identifier(value: str) -> str:
    escaped = str(value).replace('"', '""')
    return f'"{escaped}"'


def _is_pyarrow_missing(exc: ModuleNotFoundError) -> bool:
    return exc.name == "pyarrow" or "pyarrow" in str(exc)


def _polars_dtype_to_duckdb(dtype: pl.DataType) -> str:
    if dtype == pl.Boolean:
        return "BOOLEAN"
    if dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }:
        return "BIGINT"
    if dtype in {pl.Float32, pl.Float64}:
        return "DOUBLE"
    if dtype == pl.Date:
        return "DATE"
    if dtype in {pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")}:
        return "TIMESTAMP"
    return "TEXT"


@dataclass
class DuckDBGateway:
    """Open short-lived DuckDB sessions attached read-only to a local or GCS database."""

    duckdb_path: str
    key_id: str
    secret: str
    attached_alias: str = "mtg_db"

    def describe_source(self) -> str:
        return "remote" if is_gcs_uri(self.duckdb_path) else "mounted"

    def _configure_remote_gcs_access(self, connection) -> None:
        connection.execute("INSTALL httpfs;")
        connection.execute("LOAD httpfs;")
        connection.execute(
            f"""
            CREATE SECRET (
                TYPE gcs,
                KEY_ID '{_quote_sql_literal(self.key_id)}',
                SECRET '{_quote_sql_literal(self.secret)}'
            );
            """
        )

    def _open_connection(self):
        connection = duckdb.connect()
        if is_gcs_uri(self.duckdb_path):
            self._configure_remote_gcs_access(connection)
        connection.execute(
            f"ATTACH '{_quote_sql_literal(self.duckdb_path)}' AS {self.attached_alias} (READ_ONLY);"
        )
        return connection

    def _register_relation(self, connection, name: str, obj: Any) -> None:
        """Registers Polars objects as queryable relations with pyarrow fallback"""
        try:
            connection.register(name, obj)
            return
        except ModuleNotFoundError as exc:
            if not _is_pyarrow_missing(exc):
                raise

        if isinstance(obj, pl.LazyFrame):
            obj = obj.collect()

        if not isinstance(obj, pl.DataFrame):
            raise RuntimeError(
                f"Unable to register relation '{name}' without pyarrow for object type {type(obj)!r}."
            )

        self._register_polars_without_pyarrow(connection, name, obj)

    def _register_polars_without_pyarrow(self, connection, name: str, frame: pl.DataFrame) -> None:
        columns = [
            f"{_quote_identifier(column_name)} {_polars_dtype_to_duckdb(dtype)}"
            for column_name, dtype in frame.schema.items()
        ]
        connection.execute(
            f"CREATE TEMP TABLE {_quote_identifier(name)} ({', '.join(columns)});"
        )
        if frame.height == 0:
            return

        placeholders = ", ".join("?" for _ in frame.columns)
        rows = frame.iter_rows()
        connection.executemany(
            f"INSERT INTO {_quote_identifier(name)} VALUES ({placeholders})",
            rows,
        )

    def query_polars(
        self,
        sql: str,
        params: Iterable[Any] | None = None,
        register: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        connection = self._open_connection()
        try:
            for name, obj in (register or {}).items():
                self._register_relation(connection, name, obj)
            cursor = connection.execute(sql, list(params or []))
            try:
                return cursor.pl()
            except ModuleNotFoundError as exc:
                if not _is_pyarrow_missing(exc):
                    raise

                column_names = [column[0] for column in cursor.description or []]
                rows = cursor.fetchall()
                return pl.DataFrame(rows, schema=column_names, orient="row")
        finally:
            connection.close()

    def query_rows(
        self,
        sql: str,
        params: Iterable[Any] | None = None,
        register: dict[str, Any] | None = None,
    ) -> list[tuple[Any, ...]]:
        connection = self._open_connection()
        try:
            for name, obj in (register or {}).items():
                self._register_relation(connection, name, obj)
            return connection.execute(sql, list(params or [])).fetchall()
        finally:
            connection.close()

    def query_scalar(
        self,
        sql: str,
        params: Iterable[Any] | None = None,
        register: dict[str, Any] | None = None,
    ) -> Any:
        rows = self.query_rows(sql, params=params, register=register)
        return rows[0][0] if rows else None

    def glob_remote_files(self, remote_uri: str) -> list[str]:
        patterns = [
            f"{remote_uri.rstrip('/') }/*",
            f"{remote_uri.rstrip('/') }/**/*",
        ]
        files: list[str] = []
        for pattern in patterns:
            rows = self.query_rows(f"SELECT * FROM glob('{_quote_sql_literal(pattern)}')")
            files.extend(row[0] for row in rows)

        seen: dict[str, None] = {}
        return [path for path in files if not (path in seen or seen.setdefault(path, None))]

    def read_remote_blob(self, remote_uri: str) -> bytes:
        connection = self._open_connection()
        try:
            blob = connection.execute(
                f"SELECT content FROM read_blob('{_quote_sql_literal(remote_uri)}')"
            ).fetchone()[0]
            return bytes(blob)
        finally:
            connection.close()

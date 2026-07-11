"""Exact spillable canonical key indexing for dataset storage."""

from __future__ import annotations

import math
import shutil
import sqlite3
import struct
import tempfile
from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from types import TracebackType

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.storage._scratch_ownership import (
    owned_scratch_prefix,
    scratch_owner_is_stale,
)


class ExactPrimaryKeyIndex:
    """Index Arrow-canonical keys in isolated SQLite namespaces."""

    def __init__(
        self,
        *,
        primary_key: Sequence[str],
        timestamp_columns: Sequence[str],
        scratch_parent: Path,
        key_schema: pa.Schema | None = None,
        sqlite_cache_bytes: int = 8 * 1024,
    ) -> None:
        self._primary_key = tuple(str(column) for column in primary_key)
        self._timestamp_columns = frozenset(
            str(column) for column in timestamp_columns
        )
        self._scratch_parent = Path(scratch_parent)
        self._key_schema = key_schema
        if (
            isinstance(sqlite_cache_bytes, bool)
            or not isinstance(sqlite_cache_bytes, int)
            or sqlite_cache_bytes <= 0
        ):
            raise ValueError("invalid_exact_key_sqlite_cache_budget")
        self._sqlite_cache_kib = max(1, sqlite_cache_bytes // 1024)
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._connection: sqlite3.Connection | None = None

    @property
    def database_path(self) -> Path:
        if self._temporary_directory is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        return Path(self._temporary_directory.name) / "k.sqlite3"

    def __enter__(self) -> ExactPrimaryKeyIndex:
        if self._connection is not None:
            raise RuntimeError("exact_primary_key_index_already_open")
        self._scratch_parent.mkdir(parents=True, exist_ok=True)
        _cleanup_stale_key_scratch(self._scratch_parent)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=owned_scratch_prefix(".k-"),
            dir=self._scratch_parent,
        )
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.database_path)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute(f"PRAGMA cache_size=-{self._sqlite_cache_kib}")
            connection.execute(
                "CREATE TABLE exact_keys ("
                "namespace TEXT NOT NULL, "
                "payload BLOB NOT NULL, "
                "PRIMARY KEY(namespace, payload)"
                ") WITHOUT ROWID"
            )
            connection.commit()
        except Exception:
            if connection is not None:
                connection.close()
            self._temporary_directory.cleanup()
            self._temporary_directory = None
            raise
        assert connection is not None
        self._connection = connection
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        connection = self._connection
        temporary_directory = self._temporary_directory
        self._connection = None
        self._temporary_directory = None
        try:
            if connection is not None:
                connection.close()
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

    def add_frame(self, frame: pd.DataFrame, *, namespace: str = "dataset") -> int:
        missing = [column for column in self._primary_key if column not in frame]
        if missing:
            raise ValueError("missing_dataset_primary_key_column")
        keys = frame.loc[:, list(self._primary_key)].copy()
        for column in self._timestamp_columns:
            keys[column] = strict_timestamp_series(
                keys[column],
                error="invalid_dataset_timestamp_value",
            )
        table = pa.Table.from_pandas(keys, preserve_index=False)
        return self.add_table(table, namespace=namespace)

    def add_table(self, table: pa.Table, *, namespace: str = "dataset") -> int:
        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("invalid_exact_key_namespace")
        missing = [
            column
            for column in self._primary_key
            if column not in table.column_names
        ]
        if missing:
            raise ValueError("missing_dataset_primary_key_column")
        keys = table.select(self._primary_key)
        if self._key_schema is None:
            self._key_schema = keys.schema.remove_metadata()
        else:
            try:
                keys = keys.cast(self._key_schema, safe=True)
            except (
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError,
            ) as exc:
                raise ValueError("invalid_dataset_primary_key_value") from exc
        pandas_keys = keys.to_pandas(use_threads=False)
        if bool(pandas_keys.isna().any(axis=None)):
            raise ValueError("dataset_primary_key_null")
        local_duplicate_mask = pandas_keys.duplicated(list(self._primary_key))
        local_duplicate_count = int(local_duplicate_mask.sum())
        if local_duplicate_count:
            keep_mask = ~local_duplicate_mask
            keys = keys.filter(pa.array(keep_mask.to_numpy(dtype=bool)))
            pandas_keys = pandas_keys.loc[keep_mask].reset_index(drop=True)
        before = connection.total_changes
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO exact_keys(namespace, payload) VALUES (?, ?)",
                self._payload_rows(
                    keys,
                    pandas_keys,
                    namespace=namespace,
                ),
            )
        inserted = connection.total_changes - before
        return local_duplicate_count + int(len(keys) - inserted)

    def difference_counts(
        self,
        *,
        expected_namespace: str,
        actual_namespace: str,
    ) -> tuple[int, int]:
        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        missing = connection.execute(
            "SELECT COUNT(*) FROM exact_keys AS expected "
            "WHERE expected.namespace = ? AND NOT EXISTS ("
            "SELECT 1 FROM exact_keys AS actual "
            "WHERE actual.namespace = ? AND actual.payload = expected.payload"
            ")",
            (expected_namespace, actual_namespace),
        ).fetchone()
        extra = connection.execute(
            "SELECT COUNT(*) FROM exact_keys AS actual "
            "WHERE actual.namespace = ? AND NOT EXISTS ("
            "SELECT 1 FROM exact_keys AS expected "
            "WHERE expected.namespace = ? AND expected.payload = actual.payload"
            ")",
            (actual_namespace, expected_namespace),
        ).fetchone()
        assert missing is not None and extra is not None
        return int(missing[0]), int(extra[0])

    def _payload_rows(
        self,
        keys: pa.Table,
        pandas_keys: pd.DataFrame,
        *,
        namespace: str,
    ) -> Iterator[tuple[str, bytes]]:
        fields = tuple(keys.schema.field(column) for column in self._primary_key)
        for row in pandas_keys.itertuples(index=False, name=None):
            payload = b"".join(
                _canonical_arrow_scalar_payload(value, field.type)
                for value, field in zip(row, fields, strict=True)
            )
            yield namespace, payload


def _canonical_arrow_scalar_payload(value: object, data_type: pa.DataType) -> bytes:
    type_payload = str(data_type).encode("ascii")
    if pa.types.is_dictionary(data_type):
        value_payload = _canonical_arrow_scalar_payload(
            value,
            data_type.value_type,
        )
    elif pa.types.is_timestamp(data_type):
        value_payload = struct.pack(">q", int(pd.Timestamp(value).value))
    elif pa.types.is_boolean(data_type):
        value_payload = b"1" if bool(value) else b"0"
    elif pa.types.is_signed_integer(data_type) or pa.types.is_unsigned_integer(
        data_type
    ):
        value_payload = str(int(value)).encode("ascii")
    elif pa.types.is_floating(data_type):
        number = float(value)
        if math.isnan(number):
            raise ValueError("dataset_primary_key_null")
        if number == 0.0:
            number = 0.0
        value_payload = (
            struct.pack(">f", number)
            if pa.types.is_float32(data_type)
            else struct.pack(">d", number)
        )
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        value_payload = str(value).encode("utf-8")
    elif pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        value_payload = bytes(value)
    elif pa.types.is_decimal(data_type):
        decimal = Decimal(value)
        if decimal == 0:
            decimal = Decimal(0)
        value_payload = format(decimal, "f").encode("ascii")
    elif pa.types.is_date(data_type):
        value_payload = pd.Timestamp(value).date().isoformat().encode("ascii")
    else:
        raise ValueError("invalid_dataset_primary_key_value")
    return _framed(type_payload, value_payload)


def _framed(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">Q", len(tag))
        + tag
        + struct.pack(">Q", len(payload))
        + payload
    )


def _cleanup_stale_key_scratch(parent: Path) -> None:
    for scratch in parent.glob(".k-*"):
        if not scratch.is_dir() or not scratch_owner_is_stale(
            scratch,
            prefix=".k-",
        ):
            continue
        shutil.rmtree(scratch)

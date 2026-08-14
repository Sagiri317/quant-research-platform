"""Exact spillable canonical key indexing for dataset storage."""

from __future__ import annotations

import math
import os
import sqlite3
import struct
import sys
from array import array
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from uuid import uuid4

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import require_free_disk_headroom
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._scratch_ownership import (
    cleanup_stale_owned_scratch,
    create_owned_scratch,
    owned_scratch_name,
    remove_owned_scratch,
    require_owned_scratch,
)

_MAX_SQLITE_CACHE_BYTES = 8 * 1024 * 1024
_DISK_RESERVE_BATCH_BYTES = 1024**2
EXACT_KEY_SQLITE_PAGE_BYTES = 1024
_EXACT_KEY_SQLITE_STARTUP_RESERVE_BYTES = 8 * EXACT_KEY_SQLITE_PAGE_BYTES


@dataclass(frozen=True, slots=True)
class CompactPositionState(Sequence[int]):
    """Expose admitted positions while retaining an explicit fixed backing state."""

    _positions: array
    keep_count: int

    def __len__(self) -> int:
        return self.keep_count

    def __getitem__(self, index):  # noqa: ANN001, ANN204
        return self.view[index]

    @property
    def view(self) -> memoryview:
        return memoryview(self._positions)[: self.keep_count].toreadonly()

    @property
    def retained_memory_bytes(self) -> int:
        return len(self._positions) * self._positions.itemsize


class ExactPrimaryKeyIndex:
    """Index Arrow-canonical keys in isolated SQLite namespaces."""

    def __init__(
        self,
        *,
        primary_key: Sequence[str],
        timestamp_columns: Sequence[str],
        scratch_parent: Path,
        key_schema: pa.Schema | None = None,
        sqlite_cache_bytes: int = EXACT_KEY_SQLITE_PAGE_BYTES,
    ) -> None:
        self._primary_key = tuple(str(column) for column in primary_key)
        self._timestamp_columns = frozenset(str(column) for column in timestamp_columns)
        self._scratch_parent = Path(scratch_parent)
        self._key_schema = key_schema
        if (
            isinstance(sqlite_cache_bytes, bool)
            or not isinstance(sqlite_cache_bytes, int)
            or sqlite_cache_bytes < 1024
            or sqlite_cache_bytes > _MAX_SQLITE_CACHE_BYTES
        ):
            raise ValueError("invalid_exact_key_sqlite_cache_budget")
        self._sqlite_cache_kib = sqlite_cache_bytes // 1024
        self._temporary_directory: Path | None = None
        self._temporary_identity: os.stat_result | None = None
        self._connection: sqlite3.Connection | None = None

    @property
    def database_path(self) -> Path:
        if self._temporary_directory is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        return self._temporary_directory / "k.db"

    @property
    def primary_key(self) -> tuple[str, ...]:
        return self._primary_key

    @property
    def retained_memory_bytes(self) -> int:
        """Return the Storage-controlled SQLite page-cache reserve."""

        return self._sqlite_cache_kib * 1024

    def __enter__(self) -> ExactPrimaryKeyIndex:
        if self._connection is not None:
            raise RuntimeError("exact_primary_key_index_already_open")
        require_unaliased_path(
            self._scratch_parent, error="invalid_storage_scratch_path"
        )
        self._scratch_parent.mkdir(parents=True, exist_ok=True)
        require_unaliased_path(
            self._scratch_parent, error="invalid_storage_scratch_path"
        )
        require_free_disk_headroom(
            self._scratch_parent,
            reserve_bytes=_EXACT_KEY_SQLITE_STARTUP_RESERVE_BYTES,
        )
        parent_identity = self._scratch_parent.lstat()
        cleanup_stale_owned_scratch(
            self._scratch_parent,
            parent_identity=parent_identity,
            prefix=".k-",
            directory=True,
        )
        require_unaliased_path(
            self._scratch_parent, error="invalid_storage_scratch_path"
        )
        temporary_directory = self._scratch_parent / owned_scratch_name(
            ".k-", uuid4().hex[:8]
        )
        temporary_identity: os.stat_result | None = None
        connection: sqlite3.Connection | None = None
        try:
            temporary_identity = create_owned_scratch(temporary_directory)
            require_owned_scratch(self._scratch_parent, parent_identity)
            self._temporary_directory = temporary_directory
            self._temporary_identity = temporary_identity
            require_unaliased_path(
                temporary_directory,
                error="invalid_storage_scratch_path",
            )
            require_owned_scratch(temporary_directory, temporary_identity)
            connection = sqlite3.connect(self.database_path)
            connection.execute("PRAGMA page_size=1024")
            sqlite_page_limit = (
                _EXACT_KEY_SQLITE_STARTUP_RESERVE_BYTES // EXACT_KEY_SQLITE_PAGE_BYTES
            )
            connection.execute(f"PRAGMA main.max_page_count={sqlite_page_limit}")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(f"PRAGMA temp.max_page_count={sqlite_page_limit}")
            connection.execute(f"PRAGMA cache_size=-{self._sqlite_cache_kib}")
            connection.execute(
                "CREATE TABLE exact_keys ("
                "namespace TEXT NOT NULL, "
                "payload BLOB NOT NULL, "
                "value BLOB, "
                "PRIMARY KEY(namespace, payload)"
                ") WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE comparison_rows ("
                "position INTEGER PRIMARY KEY, "
                "namespace TEXT NOT NULL, "
                "payload BLOB NOT NULL, "
                "value BLOB NOT NULL"
                ")"
            )
            connection.execute(
                "CREATE INDEX comparison_rows_identity "
                "ON comparison_rows(namespace, payload)"
            )
            connection.commit()
        except BaseException:
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass
            if temporary_identity is not None:
                try:
                    remove_owned_scratch(temporary_directory, temporary_identity)
                except BaseException:
                    pass
            self._temporary_directory = None
            self._temporary_identity = None
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
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except BaseException:
            pass

    def close(self) -> None:
        connection = self._connection
        temporary_directory = self._temporary_directory
        temporary_identity = self._temporary_identity
        self._connection = None
        self._temporary_directory = None
        self._temporary_identity = None
        try:
            if connection is not None:
                connection.close()
        except BaseException:
            if temporary_directory is not None and temporary_identity is not None:
                try:
                    remove_owned_scratch(temporary_directory, temporary_identity)
                except BaseException:
                    pass
            raise
        else:
            if temporary_directory is not None and temporary_identity is not None:
                remove_owned_scratch(temporary_directory, temporary_identity)

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
            column for column in self._primary_key if column not in table.column_names
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
        if any(column.null_count for column in keys.columns):
            raise ValueError("dataset_primary_key_null")
        before = connection.total_changes
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO exact_keys(namespace, payload, value) "
                "VALUES (?, ?, ?)",
                self._gated_rows(self._payload_rows(keys, namespace=namespace)),
            )
        inserted = connection.total_changes - before
        return int(keys.num_rows - inserted)

    def add_identity_payloads(
        self,
        *,
        namespace: str,
        payloads: Iterable[tuple[bytes, bytes]],
    ) -> int:
        return self.add_identity_payload_rows(
            rows=(
                (namespace, bytes(identity), bytes(value))
                for identity, value in payloads
            )
        )

    def add_identity_payload_rows(
        self,
        *,
        rows: Iterable[tuple[str, bytes, bytes]],
    ) -> int:
        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        before = connection.total_changes

        def validated_rows() -> Iterator[tuple[str, bytes, bytes]]:
            for namespace, identity, value in rows:
                if not isinstance(namespace, str) or not namespace:
                    raise ValueError("invalid_exact_key_namespace")
                yield namespace, bytes(identity), bytes(value)

        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO exact_keys(namespace, payload, value) "
                "VALUES (?, ?, ?)",
                self._gated_rows(validated_rows()),
            )
        return connection.total_changes - before

    def add_float_value_table(
        self,
        table: pa.Table,
        *,
        namespace: str,
        value_column: str,
        payload_target_bytes: int,
    ) -> int:
        """Index one bounded Arrow key/value batch with exact float values."""

        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        self._validate_comparison_table_grammar(
            table,
            namespace=namespace,
            value_columns=(value_column,),
            payload_target_bytes=payload_target_bytes,
        )
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
        if any(column.null_count for column in keys.columns):
            raise ValueError("dataset_primary_key_null")
        values = table.column(value_column)

        def rows() -> Iterator[tuple[str, bytearray, bytes]]:
            for (row_namespace, payload, _), scalar in zip(
                self._payload_rows(keys, namespace=namespace),
                values,
                strict=True,
            ):
                value = scalar.as_py()
                if value is None or not math.isfinite(float(value)):
                    raise ValueError("invalid_exact_float_value")
                yield row_namespace, payload, struct.pack(">d", float(value))

        before = connection.total_changes
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO exact_keys(namespace, payload, value) "
                "VALUES (?, ?, ?)",
                self._gated_rows(rows()),
            )
        return int(table.num_rows - (connection.total_changes - before))

    def lookup_float_values(
        self,
        table: pa.Table,
        *,
        source_namespace: str,
        matched_namespace: str | None,
        payload_target_bytes: int,
    ) -> tuple[array, bytearray]:
        """Look up one bounded Arrow key batch without materializing the index."""

        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        if matched_namespace is not None and (
            not isinstance(matched_namespace, str) or not matched_namespace
        ):
            raise ValueError("invalid_exact_key_namespace")
        rows = self._prepared_comparison_rows(
            table,
            namespace=source_namespace,
            value_columns=(),
            payload_target_bytes=payload_target_bytes,
        )
        state_bytes = table.num_rows * (array("d").itemsize + 1)
        if state_bytes > payload_target_bytes:
            raise ValueError("exact_key_position_state_exceeds_payload_target")
        values = array("d", [math.nan]) * table.num_rows
        found = bytearray(table.num_rows)
        with connection:
            connection.executemany(
                "INSERT INTO comparison_rows(position, namespace, payload, value) "
                "VALUES (?, ?, ?, ?)",
                (
                    (position, namespace, identity, value)
                    for position, (namespace, identity, value) in enumerate(rows)
                ),
            )
            matches = connection.execute(
                "SELECT incoming.position, existing.value "
                "FROM comparison_rows AS incoming "
                "LEFT JOIN exact_keys AS existing "
                "ON existing.namespace = ? "
                "AND existing.payload = incoming.payload "
                "ORDER BY incoming.position",
                (source_namespace,),
            )
            for position, payload in matches:
                if payload is None:
                    continue
                encoded = bytes(payload)
                if len(encoded) != 8:
                    raise ValueError("invalid_exact_float_value")
                values[int(position)] = struct.unpack(">d", encoded)[0]
                found[int(position)] = 1
            if matched_namespace is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO exact_keys(namespace, payload, value) "
                    "SELECT ?, incoming.payload, existing.value "
                    "FROM comparison_rows AS incoming "
                    "JOIN exact_keys AS existing "
                    "ON existing.namespace = ? "
                    "AND existing.payload = incoming.payload",
                    (matched_namespace, source_namespace),
                )
            connection.execute("DELETE FROM comparison_rows")
        return values, found

    def add_comparison_table(
        self,
        table: pa.Table,
        *,
        namespace: str,
        value_columns: Sequence[str],
        payload_target_bytes: int,
    ) -> int:
        """Index persisted primary keys and selected row values exactly."""

        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        before = connection.total_changes
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO exact_keys(namespace, payload, value) "
                "VALUES (?, ?, ?)",
                self._prepared_comparison_rows(
                    table,
                    namespace=namespace,
                    value_columns=value_columns,
                    payload_target_bytes=payload_target_bytes,
                ),
            )
        inserted = connection.total_changes - before
        return int(table.num_rows - inserted)

    def classify_comparison_table(
        self,
        table: pa.Table,
        *,
        namespace: str,
        value_columns: Sequence[str],
        payload_target_bytes: int,
    ) -> tuple[CompactPositionState, int]:
        """Return new-row positions while rejecting conflicting exact keys."""

        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        selected = self._validate_comparison_table_grammar(
            table,
            namespace=namespace,
            value_columns=value_columns,
            payload_target_bytes=payload_target_bytes,
        )
        position_bytes = table.num_rows * array("Q").itemsize
        if position_bytes > payload_target_bytes:
            raise ValueError("exact_key_position_state_exceeds_payload_target")
        position_state = array("Q", [0]) * table.num_rows
        rows = self._prepared_comparison_rows(
            table,
            namespace=namespace,
            value_columns=value_columns,
            payload_target_bytes=payload_target_bytes,
            validated_columns=selected,
        )
        keep_count = 0
        with connection:
            connection.executemany(
                "INSERT INTO comparison_rows(position, namespace, payload, value) "
                "VALUES (?, ?, ?, ?)",
                (
                    (position, row_namespace, identity, value)
                    for position, (row_namespace, identity, value) in enumerate(rows)
                ),
            )
            incoming_conflict = connection.execute(
                "SELECT 1 FROM comparison_rows "
                "GROUP BY namespace, payload HAVING MIN(value) != MAX(value) "
                "LIMIT 1"
            ).fetchone()
            existing_conflict = connection.execute(
                "SELECT 1 FROM comparison_rows AS incoming "
                "JOIN exact_keys AS existing "
                "ON existing.namespace = incoming.namespace "
                "AND existing.payload = incoming.payload "
                "WHERE existing.value != incoming.value LIMIT 1"
            ).fetchone()
            if incoming_conflict is not None or existing_conflict is not None:
                raise ValueError("conflicting_exact_primary_key_payload")
            admitted = connection.execute(
                "SELECT incoming.position FROM comparison_rows AS incoming "
                "WHERE NOT EXISTS (SELECT 1 FROM comparison_rows AS earlier "
                "WHERE earlier.namespace = incoming.namespace "
                "AND earlier.payload = incoming.payload "
                "AND earlier.position < incoming.position) "
                "AND NOT EXISTS (SELECT 1 FROM exact_keys AS existing "
                "WHERE existing.namespace = incoming.namespace "
                "AND existing.payload = incoming.payload) "
                "ORDER BY incoming.position"
            )
            for (position,) in admitted:
                position_state[keep_count] = int(position)
                keep_count += 1
            reserve = connection.execute(
                "SELECT COALESCE(SUM(27 + length(CAST(namespace AS BLOB)) "
                "+ length(payload) + length(value)), 0), COUNT(*) FROM ("
                "SELECT incoming.namespace, incoming.payload, MIN(incoming.value) "
                "AS value FROM comparison_rows AS incoming "
                "LEFT JOIN exact_keys AS existing "
                "ON existing.namespace = incoming.namespace "
                "AND existing.payload = incoming.payload "
                "WHERE existing.payload IS NULL "
                "GROUP BY incoming.namespace, incoming.payload)"
            ).fetchone()
            assert reserve is not None
            logical_bytes = int(reserve[0])
            row_count = int(reserve[1])
            if logical_bytes > 0:
                # WITHOUT ROWID table, doubled for journal, with B-tree page slack.
                reserve_bytes = 2 * (
                    logical_bytes + 4 * EXACT_KEY_SQLITE_PAGE_BYTES * row_count
                )
                self._admit_physical_write(reserve_bytes)
            connection.execute(
                "INSERT OR IGNORE INTO exact_keys(namespace, payload, value) "
                "SELECT namespace, payload, MIN(value) FROM comparison_rows "
                "GROUP BY namespace, payload"
            )
            connection.execute("DELETE FROM comparison_rows")
        return (
            CompactPositionState(position_state, keep_count),
            table.num_rows - keep_count,
        )

    def _prepared_comparison_rows(
        self,
        table: pa.Table,
        *,
        namespace: str,
        value_columns: Sequence[str],
        payload_target_bytes: int,
        validated_columns: tuple[str, ...] | None = None,
    ) -> Iterator[tuple[str, bytes, bytes]]:
        selected = validated_columns
        if selected is None:
            selected = self._validate_comparison_table_grammar(
                table,
                namespace=namespace,
                value_columns=value_columns,
                payload_target_bytes=payload_target_bytes,
            )
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
        if any(column.null_count for column in keys.columns):
            raise ValueError("dataset_primary_key_null")
        return self._gated_rows(
            self._comparison_payload_rows(
                keys,
                table.select(selected),
                namespace=namespace,
                payload_target_bytes=payload_target_bytes,
            )
        )

    def _validate_comparison_table_grammar(
        self,
        table: pa.Table,
        *,
        namespace: str,
        value_columns: Sequence[str],
        payload_target_bytes: int,
    ) -> tuple[str, ...]:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("invalid_exact_key_namespace")
        if (
            isinstance(payload_target_bytes, bool)
            or not isinstance(payload_target_bytes, int)
            or payload_target_bytes <= 0
        ):
            raise ValueError("invalid_dataset_comparison_payload_target")
        selected = tuple(str(column) for column in value_columns)
        if len(selected) != len(set(selected)) or any(
            column in self._primary_key for column in selected
        ):
            raise ValueError("invalid_dataset_comparison_columns")
        if any(
            column not in table.column_names
            for column in (*self._primary_key, *selected)
        ):
            raise ValueError("missing_dataset_comparison_column")
        return selected

    def identity_payload_values(self, *, namespace: str) -> Iterator[bytes]:
        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("invalid_exact_key_namespace")
        rows = connection.execute(
            "SELECT value FROM exact_keys WHERE namespace = ? ORDER BY payload",
            (namespace,),
        )
        try:
            for row in rows:
                if row[0] is None:
                    raise ValueError("exact_key_identity_payload_missing")
                yield bytes(row[0])
        finally:
            if sys.exception() is None:
                rows.close()
            else:
                try:
                    rows.close()
                except BaseException:
                    pass

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

    def comparison_counts(
        self,
        *,
        expected_namespace: str,
        actual_namespace: str,
    ) -> tuple[int, int, int]:
        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        missing_count, extra_count = self.difference_counts(
            expected_namespace=expected_namespace,
            actual_namespace=actual_namespace,
        )
        changed = connection.execute(
            "SELECT COUNT(*) FROM exact_keys AS expected "
            "JOIN exact_keys AS actual ON actual.payload = expected.payload "
            "WHERE expected.namespace = ? AND actual.namespace = ? "
            "AND expected.value != actual.value",
            (expected_namespace, actual_namespace),
        ).fetchone()
        assert changed is not None
        return missing_count, extra_count, int(changed[0])

    def _payload_rows(
        self,
        keys: pa.Table,
        *,
        namespace: str,
    ) -> Iterator[tuple[str, bytearray, None]]:
        fields = tuple(keys.schema.field(column) for column in self._primary_key)
        columns = tuple(keys.column(column) for column in self._primary_key)
        for row_index in range(keys.num_rows):
            payload = _BoundedPayloadSink(None)
            for column, field in zip(columns, fields, strict=True):
                _append_arrow_scalar_payload(
                    payload,
                    column[row_index],
                    field.type,
                    normalize_float_zero=(
                        pa.types.is_float32(field.type)
                        or pa.types.is_float64(field.type)
                    ),
                )
            yield namespace, payload.buffer, None

    def _comparison_payload_rows(
        self,
        keys: pa.Table,
        values: pa.Table,
        *,
        namespace: str,
        payload_target_bytes: int,
    ) -> Iterator[tuple[str, bytearray, bytearray]]:
        key_fields = tuple(keys.schema.field(column) for column in self._primary_key)
        key_columns = tuple(keys.column(column) for column in self._primary_key)
        value_fields = tuple(
            values.schema.field(column) for column in values.column_names
        )
        value_arrays = tuple(values.column(column) for column in values.column_names)
        for row_index in range(keys.num_rows):
            key_payload = _BoundedPayloadSink(payload_target_bytes)
            for column, field in zip(key_columns, key_fields, strict=True):
                _append_arrow_scalar_payload(
                    key_payload,
                    column[row_index],
                    field.type,
                    normalize_float_zero=(
                        pa.types.is_float32(field.type)
                        or pa.types.is_float64(field.type)
                    ),
                )
            value_payload = _BoundedPayloadSink(key_payload.remaining_bytes)
            for column, field in zip(value_arrays, value_fields, strict=True):
                _append_arrow_scalar_payload(
                    value_payload,
                    column[row_index],
                    field.type,
                    allow_null=True,
                    allow_nonfinite=True,
                )
            yield namespace, key_payload.buffer, value_payload.buffer

    def _gated_rows(
        self,
        rows: Iterable[tuple[str, bytes | bytearray, bytes | bytearray | None]],
    ) -> Iterator[tuple[str, bytes | bytearray, bytes | bytearray | None]]:
        admitted_bytes = 0
        for namespace, payload, value in rows:
            key_bytes = len(namespace.encode("utf-8")) + len(payload)
            logical_row_bytes = 27 + key_bytes + (0 if value is None else len(value))
            # Table + secondary index, doubled for journal, with B-tree page slack.
            reserve_bytes = 2 * (
                logical_row_bytes + 27 + key_bytes + 4 * EXACT_KEY_SQLITE_PAGE_BYTES
            )
            if reserve_bytes > admitted_bytes:
                admitted_bytes = max(_DISK_RESERVE_BATCH_BYTES, reserve_bytes)
                self._admit_physical_write(admitted_bytes)
            admitted_bytes -= reserve_bytes
            yield namespace, payload, value

    def _admit_physical_write(self, reserve_bytes: int) -> None:
        schemas = ("main", "temp")
        require_free_disk_headroom(
            self._scratch_parent,
            reserve_bytes=reserve_bytes * len(schemas),
        )
        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_index_not_open")
        growth_pages = (
            reserve_bytes + EXACT_KEY_SQLITE_PAGE_BYTES - 1
        ) // EXACT_KEY_SQLITE_PAGE_BYTES
        for schema in schemas:
            current_pages = int(
                connection.execute(f"PRAGMA {schema}.page_count").fetchone()[0]
            )
            connection.execute(
                f"PRAGMA {schema}.max_page_count={current_pages + growth_pages}"
            )


class _BoundedPayloadSink:
    __slots__ = ("buffer", "_auxiliary_bytes", "_limit")

    def __init__(self, limit: int | None) -> None:
        self.buffer = bytearray()
        self._auxiliary_bytes = 0
        self._limit = limit

    @property
    def remaining_bytes(self) -> int | None:
        if self._limit is None:
            return None
        return self._limit - len(self.buffer) - self._auxiliary_bytes

    def append(self, payload: bytes | bytearray | memoryview | pa.Buffer) -> None:
        remaining = self.remaining_bytes
        if remaining is not None and len(payload) > remaining:
            raise ValueError("dataset_comparison_payload_exceeds_target_bytes")
        self.buffer.extend(payload)

    def reserve_auxiliary(self, size: int) -> None:
        remaining = self.remaining_bytes
        if remaining is not None and size > remaining:
            raise ValueError("dataset_comparison_payload_exceeds_target_bytes")
        self._auxiliary_bytes += size

    def release_auxiliary(self, size: int) -> None:
        self._auxiliary_bytes -= size


def _append_arrow_scalar_payload(
    sink: _BoundedPayloadSink,
    value: pa.Scalar,
    data_type: pa.DataType,
    *,
    allow_null: bool = False,
    allow_nonfinite: bool = False,
    normalize_float_zero: bool = False,
) -> None:
    type_payload = str(data_type).encode("utf-8")
    length_offset, payload_start = _start_frame(sink, type_payload)
    if not value.is_valid:
        if not allow_null:
            raise ValueError("dataset_primary_key_null")
        sink.append(b"0")
        _finish_frame(sink, length_offset, payload_start)
        return
    sink.append(b"1")
    if pa.types.is_dictionary(data_type):
        _append_arrow_scalar_payload(
            sink,
            value.value,
            data_type.value_type,
            allow_null=allow_null,
            allow_nonfinite=allow_nonfinite,
        )
    elif pa.types.is_timestamp(data_type):
        sink.append(struct.pack(">q", int(value.value)))
    elif pa.types.is_time(data_type) or pa.types.is_duration(data_type):
        sink.append(struct.pack(">q", int(value.value)))
    elif pa.types.is_boolean(data_type):
        sink.append(b"1" if bool(value.as_py()) else b"0")
    elif pa.types.is_signed_integer(data_type) or pa.types.is_unsigned_integer(
        data_type
    ):
        sink.append(str(int(value.as_py())).encode("ascii"))
    elif pa.types.is_floating(data_type):
        number = float(value.as_py())
        if math.isnan(number) and not allow_nonfinite:
            raise ValueError("dataset_primary_key_null")
        if not math.isfinite(number) and not allow_nonfinite:
            raise ValueError("invalid_dataset_primary_key_value")
        if number == 0.0 and normalize_float_zero:
            number = 0.0
        sink.append(
            struct.pack(">f", number)
            if pa.types.is_float32(data_type)
            else struct.pack(">d", number)
        )
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        _append_arrow_scalar_buffer(sink, value)
    elif (
        pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
    ):
        _append_arrow_scalar_buffer(sink, value)
    elif pa.types.is_decimal(data_type):
        decimal = Decimal(value.as_py())
        if not decimal.is_finite() and not allow_nonfinite:
            raise ValueError("invalid_dataset_primary_key_value")
        if decimal == 0:
            decimal = Decimal(0)
        sink.append(format(decimal, "f").encode("ascii"))
    elif pa.types.is_date(data_type):
        sink.append(pd.Timestamp(value.as_py()).date().isoformat().encode("ascii"))
    elif pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        for item in value.values:
            _append_arrow_scalar_payload(
                sink,
                item,
                data_type.value_type,
                allow_null=allow_null,
                allow_nonfinite=allow_nonfinite,
            )
    elif pa.types.is_fixed_size_list(data_type):
        for item in value.values:
            _append_arrow_scalar_payload(
                sink,
                item,
                data_type.value_type,
                allow_null=allow_null,
                allow_nonfinite=allow_nonfinite,
            )
    elif pa.types.is_struct(data_type):
        for field in data_type:
            field_name = field.name.encode("utf-8")
            field_length_offset, field_payload_start = _start_frame(
                sink,
                field_name,
            )
            _append_arrow_scalar_payload(
                sink,
                value[field.name],
                field.type,
                allow_null=allow_null,
                allow_nonfinite=allow_nonfinite,
            )
            _finish_frame(
                sink,
                field_length_offset,
                field_payload_start,
            )
    elif pa.types.is_map(data_type):
        entries = value.values
        entry_count = len(entries)
        array_bytes = (entry_count + 1 + entry_count + entry_count) * 8
        boundary_bytes = entry_count * 4 * 8
        sink.reserve_auxiliary(array_bytes + boundary_bytes)
        offsets = array("Q")
        key_starts = array("Q")
        key_ends = array("Q")
        item_starts = array("Q")
        item_ends = array("Q")
        order = array("Q", range(entry_count))
        # Four comparison-boundary arrays replace Python tuple metadata. Their
        # bytes participate in the same bounded sink before allocation.
        region_start = len(sink.buffer)
        offsets.append(region_start)
        for entry in entries:
            entry_length_offset, entry_payload_start = _start_frame(sink, b"entry")
            key_starts.append(len(sink.buffer))
            _append_arrow_scalar_payload(
                sink,
                entry["key"],
                data_type.key_type,
                allow_null=False,
                allow_nonfinite=allow_nonfinite,
            )
            key_ends.append(len(sink.buffer))
            item_starts.append(len(sink.buffer))
            _append_arrow_scalar_payload(
                sink,
                entry["value"],
                data_type.item_type,
                allow_null=allow_null,
                allow_nonfinite=allow_nonfinite,
            )
            item_ends.append(len(sink.buffer))
            _finish_frame(sink, entry_length_offset, entry_payload_start)
            offsets.append(len(sink.buffer))
        _heap_sort_map_order(
            sink.buffer,
            order,
            key_starts,
            key_ends,
            item_starts,
            item_ends,
        )
        destinations = array("Q", [0]) * entry_count
        destination = 0
        for entry_index in order:
            destinations[entry_index] = destination
            destination += offsets[entry_index + 1] - offsets[entry_index]
        sink.release_auxiliary(boundary_bytes)
        del key_starts, key_ends, item_starts, item_ends
        visited_bytes = (destination + 7) // 8
        sink.reserve_auxiliary(visited_bytes)
        visited = bytearray(visited_bytes)
        _permute_map_region_in_place(
            sink.buffer,
            region_start=region_start,
            offsets=offsets,
            destinations=destinations,
            visited=visited,
        )
        sink.release_auxiliary(visited_bytes + array_bytes)
    else:
        raise ValueError("invalid_dataset_primary_key_value")
    _finish_frame(sink, length_offset, payload_start)


def _start_frame(sink: _BoundedPayloadSink, tag: bytes) -> tuple[int, int]:
    sink.append(struct.pack(">Q", len(tag)))
    sink.append(tag)
    length_offset = len(sink.buffer)
    sink.append(b"\x00" * 8)
    return length_offset, len(sink.buffer)


def _finish_frame(
    sink: _BoundedPayloadSink,
    length_offset: int,
    payload_start: int,
) -> None:
    struct.pack_into(
        ">Q",
        sink.buffer,
        length_offset,
        len(sink.buffer) - payload_start,
    )


def _append_arrow_scalar_buffer(
    sink: _BoundedPayloadSink,
    value: pa.Scalar,
) -> None:
    buffer = value.as_buffer()
    remaining_bytes = sink.remaining_bytes
    if remaining_bytes is not None and buffer.size > remaining_bytes:
        raise ValueError("dataset_comparison_payload_exceeds_target_bytes")
    sink.append(buffer)


def _heap_sort_map_order(
    buffer: bytearray,
    order: array[int],
    key_starts: array[int],
    key_ends: array[int],
    item_starts: array[int],
    item_ends: array[int],
) -> None:
    def greater(left: int, right: int) -> bool:
        key_order = _compare_buffer_ranges(
            buffer,
            key_starts[left],
            key_ends[left],
            key_starts[right],
            key_ends[right],
        )
        if key_order:
            return key_order > 0
        return (
            _compare_buffer_ranges(
                buffer,
                item_starts[left],
                item_ends[left],
                item_starts[right],
                item_ends[right],
            )
            > 0
        )

    def sift(root: int, stop: int) -> None:
        while (child := root * 2 + 1) < stop:
            if child + 1 < stop and greater(order[child + 1], order[child]):
                child += 1
            if not greater(order[child], order[root]):
                return
            order[root], order[child] = order[child], order[root]
            root = child

    for root in range(len(order) // 2 - 1, -1, -1):
        sift(root, len(order))
    for stop in range(len(order) - 1, 0, -1):
        order[0], order[stop] = order[stop], order[0]
        sift(0, stop)


def _permute_map_region_in_place(
    buffer: bytearray,
    *,
    region_start: int,
    offsets: array[int],
    destinations: array[int],
    visited: bytearray,
) -> None:
    total = offsets[-1] - region_start

    def destination_for(source: int) -> int:
        entry = bisect_right(offsets, region_start + source) - 1
        return destinations[entry] + source - (offsets[entry] - region_start)

    for source in range(total):
        if visited[source >> 3] & (1 << (source & 7)):
            continue
        current = source
        held = buffer[region_start + current]
        while True:
            destination = destination_for(current)
            displaced = buffer[region_start + destination]
            buffer[region_start + destination] = held
            visited[current >> 3] |= 1 << (current & 7)
            current = destination
            held = displaced
            if current == source:
                break


def _compare_buffer_ranges(
    buffer: bytearray,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> int:
    shared_length = min(left_end - left_start, right_end - right_start)
    for offset in range(shared_length):
        left_value = buffer[left_start + offset]
        right_value = buffer[right_start + offset]
        if left_value != right_value:
            return -1 if left_value < right_value else 1
    left_length = left_end - left_start
    right_length = right_end - right_start
    return (left_length > right_length) - (left_length < right_length)

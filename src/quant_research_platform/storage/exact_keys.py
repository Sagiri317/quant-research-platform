"""Exact canonical key indexing for dataset storage."""

from __future__ import annotations

import math
import struct
from array import array
from collections.abc import Iterable, Iterator, Sequence
from decimal import Decimal

import pandas as pd
import pyarrow as pa


class ExactPrimaryKeyIndex:
    """One canonical-byte index for exact key, value, and namespace semantics."""

    def __init__(
        self, *, primary_key: Sequence[str], key_schema: pa.Schema | None = None
    ) -> None:
        self._primary_key = tuple(str(column) for column in primary_key)
        self._key_schema = key_schema
        self._namespaces: dict[str, dict[bytes, bytes | None]] = {}

    @property
    def primary_key(self) -> tuple[str, ...]:
        return self._primary_key

    def _namespace(self, namespace: str) -> dict[bytes, bytes | None]:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("invalid_exact_key_namespace")
        return self._namespaces.get(namespace, {})

    def _insert_rows(self, rows: Iterable[tuple[str, bytes, bytes | None]]) -> int:
        pending: dict[str, dict[bytes, bytes | None]] = {}
        inserted = 0
        for namespace, identity, value in rows:
            existing = self._namespace(namespace)
            key = bytes(identity)
            additions = pending.setdefault(namespace, {})
            if key not in existing and key not in additions:
                additions[key] = None if value is None else bytes(value)
                inserted += 1
        for namespace, additions in pending.items():
            self._namespaces.setdefault(namespace, {}).update(additions)
        return inserted

    def _key_table(self, table: pa.Table) -> pa.Table:
        if any(column not in table.column_names for column in self._primary_key):
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
        return keys

    def add_table(self, table: pa.Table, *, namespace: str = "dataset") -> int:
        self._namespace(namespace)
        keys = self._key_table(table)
        return table.num_rows - self._insert_rows(
            self._payload_rows(keys, namespace=namespace)
        )

    def add_identity_payloads(
        self, *, namespace: str, payloads: Iterable[tuple[bytes, bytes]]
    ) -> int:
        return self.add_identity_payload_rows(
            rows=((namespace, identity, value) for identity, value in payloads)
        )

    def add_identity_payload_rows(
        self, *, rows: Iterable[tuple[str, bytes, bytes]]
    ) -> int:
        return self._insert_rows(rows)

    def add_float_value_table(
        self, table: pa.Table, *, namespace: str, value_column: str
    ) -> int:
        self._validate_comparison_table_grammar(
            table, namespace=namespace, value_columns=(value_column,)
        )
        keys = self._key_table(table)

        def rows():
            for (_, key, _), scalar in zip(
                self._payload_rows(keys, namespace=namespace),
                table.column(value_column),
                strict=True,
            ):
                value = scalar.as_py()
                if value is None or not math.isfinite(float(value)):
                    raise ValueError("invalid_exact_float_value")
                yield (namespace, key, struct.pack(">d", float(value)))

        return table.num_rows - self._insert_rows(rows())

    def lookup_float_values(
        self, table: pa.Table, *, source_namespace: str, matched_namespace: str | None
    ):
        source = self._namespace(source_namespace)
        if matched_namespace is not None:
            self._namespace(matched_namespace)
        values, found = (
            array("d", [math.nan]) * table.num_rows,
            bytearray(table.num_rows),
        )
        matched = []
        rows = self._prepared_comparison_rows(
            table, namespace=source_namespace, value_columns=()
        )
        for position, (_, key, _) in enumerate(rows):
            payload = source.get(bytes(key))
            if payload is None:
                continue
            if len(payload) != 8:
                raise ValueError("invalid_exact_float_value")
            values[position], found[position] = (struct.unpack(">d", payload)[0], 1)
            if matched_namespace is not None:
                matched.append((matched_namespace, key, payload))
        if matched:
            self._insert_rows(matched)
        return (values, found)

    def add_comparison_table(
        self, table: pa.Table, *, namespace: str, value_columns: Sequence[str]
    ) -> int:
        rows = self._prepared_comparison_rows(
            table, namespace=namespace, value_columns=value_columns
        )
        return table.num_rows - self._insert_rows(rows)

    def classify_comparison_table(
        self, table: pa.Table, *, namespace: str, value_columns: Sequence[str]
    ):
        existing = self._namespace(namespace)
        pending: dict[bytes, bytes] = {}
        positions: list[int] = []
        for position, (_, key, value) in enumerate(
            self._prepared_comparison_rows(
                table, namespace=namespace, value_columns=value_columns
            )
        ):
            key, value = (bytes(key), bytes(value))
            if key in existing:
                if existing[key] != value:
                    raise ValueError("conflicting_exact_primary_key_payload")
            elif key in pending:
                if pending[key] != value:
                    raise ValueError("conflicting_exact_primary_key_payload")
            else:
                pending[key] = value
                positions.append(position)
        self._namespaces.setdefault(namespace, {}).update(pending)
        return (
            positions,
            table.num_rows - len(positions),
        )

    def _prepared_comparison_rows(
        self,
        table: pa.Table,
        *,
        namespace: str,
        value_columns: Sequence[str],
    ):
        selected = self._validate_comparison_table_grammar(
            table, namespace=namespace, value_columns=value_columns
        )
        keys = self._key_table(table)
        return self._comparison_payload_rows(
            keys, table.select(selected), namespace=namespace
        )

    def identity_payload_values(self, *, namespace: str) -> Iterator[bytes]:
        values = self._namespace(namespace)
        for key in sorted(values):
            value = values[key]
            if value is None:
                raise ValueError("exact_key_identity_payload_missing")
            yield value

    def identity_payload_count(self, *, namespace: str) -> int:
        return len(self._namespace(namespace))

    def identity_payload_value(self, *, namespace: str, payload: bytes) -> bytes | None:
        return self._namespace(namespace).get(bytes(payload))

    def difference_counts(self, *, expected_namespace: str, actual_namespace: str):
        expected, actual = (
            self._namespace(expected_namespace),
            self._namespace(actual_namespace),
        )
        return (
            sum(key not in actual for key in expected),
            sum(key not in expected for key in actual),
        )

    def comparison_counts(self, *, expected_namespace: str, actual_namespace: str):
        expected, actual = (
            self._namespace(expected_namespace),
            self._namespace(actual_namespace),
        )
        missing, extra = self.difference_counts(
            expected_namespace=expected_namespace, actual_namespace=actual_namespace
        )
        changed = sum(
            (value != actual[key] for key, value in expected.items() if key in actual)
        )
        return (missing, extra, changed)

    def _validate_comparison_table_grammar(
        self, table: pa.Table, *, namespace: str, value_columns: Sequence[str]
    ) -> tuple[str, ...]:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("invalid_exact_key_namespace")
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

    def _payload_rows(
        self, keys: pa.Table, *, namespace: str
    ) -> Iterator[tuple[str, bytearray, None]]:
        fields = tuple(keys.schema.field(column) for column in self._primary_key)
        columns = tuple(keys.column(column) for column in self._primary_key)
        for row_index in range(keys.num_rows):
            payload = bytearray()
            for column, field in zip(columns, fields, strict=True):
                _append_arrow_scalar_payload(
                    payload,
                    column[row_index],
                    field.type,
                    normalize_float_zero=pa.types.is_floating(field.type),
                )
            yield (namespace, payload, None)

    def _comparison_payload_rows(
        self, keys: pa.Table, values: pa.Table, *, namespace: str
    ) -> Iterator[tuple[str, bytearray, bytearray]]:
        key_fields = tuple(keys.schema.field(column) for column in self._primary_key)
        key_columns = tuple(keys.column(column) for column in self._primary_key)
        value_fields = tuple(
            values.schema.field(column) for column in values.column_names
        )
        value_arrays = tuple(values.column(column) for column in values.column_names)
        for row_index in range(keys.num_rows):
            key_payload = bytearray()
            for column, field in zip(key_columns, key_fields, strict=True):
                _append_arrow_scalar_payload(
                    key_payload,
                    column[row_index],
                    field.type,
                    normalize_float_zero=pa.types.is_floating(field.type),
                )
            value_payload = bytearray()
            for column, field in zip(value_arrays, value_fields, strict=True):
                _append_arrow_scalar_payload(
                    value_payload,
                    column[row_index],
                    field.type,
                    allow_null=True,
                    allow_nonfinite=True,
                )
            yield (namespace, key_payload, value_payload)


def _append_arrow_scalar_payload(
    sink: bytearray,
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
        sink.extend(b"0")
        _finish_frame(sink, length_offset, payload_start)
        return
    sink.extend(b"1")
    if pa.types.is_dictionary(data_type):
        _append_arrow_scalar_payload(
            sink,
            value.value,
            data_type.value_type,
            allow_null=allow_null,
            allow_nonfinite=allow_nonfinite,
        )
    elif pa.types.is_timestamp(data_type):
        sink.extend(struct.pack(">q", int(value.value)))
    elif pa.types.is_time(data_type) or pa.types.is_duration(data_type):
        sink.extend(struct.pack(">q", int(value.value)))
    elif pa.types.is_boolean(data_type):
        sink.extend(b"1" if bool(value.as_py()) else b"0")
    elif pa.types.is_signed_integer(data_type) or pa.types.is_unsigned_integer(
        data_type
    ):
        sink.extend(str(int(value.as_py())).encode("ascii"))
    elif pa.types.is_floating(data_type):
        number = float(value.as_py())
        if math.isnan(number) and (not allow_nonfinite):
            raise ValueError("dataset_primary_key_null")
        if not math.isfinite(number) and (not allow_nonfinite):
            raise ValueError("invalid_dataset_primary_key_value")
        if number == 0.0 and normalize_float_zero:
            number = 0.0
        sink.extend(
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
        if not decimal.is_finite() and (not allow_nonfinite):
            raise ValueError("invalid_dataset_primary_key_value")
        if decimal == 0:
            decimal = Decimal(0)
        sink.extend(format(decimal, "f").encode("ascii"))
    elif pa.types.is_date(data_type):
        sink.extend(pd.Timestamp(value.as_py()).date().isoformat().encode("ascii"))
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
            field_length_offset, field_payload_start = _start_frame(sink, field_name)
            _append_arrow_scalar_payload(
                sink,
                value[field.name],
                field.type,
                allow_null=allow_null,
                allow_nonfinite=allow_nonfinite,
            )
            _finish_frame(sink, field_length_offset, field_payload_start)
    elif pa.types.is_map(data_type):
        entries = []
        for entry in value.values:
            key, item = (bytearray(), bytearray())
            _append_arrow_scalar_payload(
                key, entry["key"], data_type.key_type, allow_nonfinite=allow_nonfinite
            )
            _append_arrow_scalar_payload(
                item,
                entry["value"],
                data_type.item_type,
                allow_null=allow_null,
                allow_nonfinite=allow_nonfinite,
            )
            entries.append((key, item))
        for key, item in sorted(entries):
            offset, start = _start_frame(sink, b"entry")
            sink.extend(key)
            sink.extend(item)
            _finish_frame(sink, offset, start)
    else:
        raise ValueError("invalid_dataset_primary_key_value")
    _finish_frame(sink, length_offset, payload_start)


def _start_frame(sink: bytearray, tag: bytes) -> tuple[int, int]:
    sink.extend(struct.pack(">Q", len(tag)))
    sink.extend(tag)
    length_offset = len(sink)
    sink.extend(b"\x00" * 8)
    return (length_offset, len(sink))


def _finish_frame(sink: bytearray, length_offset: int, payload_start: int) -> None:
    struct.pack_into(">Q", sink, length_offset, len(sink) - payload_start)


def _append_arrow_scalar_buffer(sink: bytearray, value: pa.Scalar) -> None:
    sink.extend(value.as_buffer())

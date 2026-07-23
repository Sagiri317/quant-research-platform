"""Strict internal JSON and typed canonical identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from decimal import MAX_EMAX, MIN_EMIN, Context, Decimal, Rounded
from typing import Any, NoReturn

import pandas as pd

_LENGTH_BYTES = 8
_MAX_INTEGER_DECIMAL_DIGITS = 4_096
_MAX_INTEGER_BITS = 13_607
_MAX_NESTING_DEPTH = 64
_JSON_INDENT = 2
_TEXT_CHUNK_CHARACTERS = 16 * 1024
_TUPLE_BASE_BYTES = ().__sizeof__()
_POINTER_BYTES = struct.calcsize("P")
_RETIRED_JSON_IDENTITY_KEYS = frozenset({"contract_version", "schema_version"})


def parse_qrp_json_object(raw: bytes) -> dict[str, Any]:
    """Decode one strict UTF-8 QRP-owned JSON object."""

    if not isinstance(raw, bytes):
        raise TypeError("qrp_json_bytes_required")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("qrp_json_invalid_utf8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object_from_pairs,
            parse_constant=_reject_json_constant,
            parse_int=_parse_json_integer,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("qrp_json_invalid") from exc
    except RecursionError as exc:
        raise ValueError("qrp_json_recursion_limit") from exc
    if not isinstance(value, dict):
        raise ValueError("qrp_json_top_level_object_required")
    return _json_native_object(value)


def render_qrp_json_object(value: Mapping[str, Any]) -> str:
    """Render one deterministic strict QRP-owned JSON object."""

    payload = _json_native_object(value)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=_JSON_INDENT,
        sort_keys=True,
        allow_nan=False,
    )


def bounded_rendered_qrp_json_object_size(
    value: Mapping[str, Any],
    *,
    lazy_arrays: Mapping[tuple[str | int, ...], Iterable[object]],
    max_bytes: int,
) -> int:
    """Count exact rendered UTF-8 bytes without materializing declared arrays."""

    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("qrp_json_max_bytes_invalid")
    _require_bounded_json_preflight(value, max_bytes=max_bytes)
    payload = _json_native_object(value)
    arrays = dict(lazy_arrays)
    for path in arrays:
        if type(path) is not tuple or not path:
            raise ValueError("qrp_json_lazy_array_path_invalid")
        current: object = payload
        for component in path:
            if type(current) is dict and type(component) is str:
                if component not in current:
                    raise ValueError("qrp_json_lazy_array_path_invalid")
                current = current[component]
            elif type(current) is list and type(component) is int:
                if component < 0 or component >= len(current):
                    raise ValueError("qrp_json_lazy_array_path_invalid")
                current = current[component]
            else:
                raise ValueError("qrp_json_lazy_array_path_invalid")
        if type(current) is not list:
            raise ValueError("qrp_json_lazy_array_path_invalid")

    lazy_values: list[_LazyJsonArray] = []
    try:
        for path, source in arrays.items():
            parent: object = payload
            for component in path[:-1]:
                parent = parent[component]  # type: ignore[index]
            lazy_value = _LazyJsonArray(
                source,
                depth=len(path) + 1,
                max_bytes=max_bytes,
            )
            lazy_values.append(lazy_value)
            parent[path[-1]] = lazy_value  # type: ignore[index]
    except BaseException:
        _close_lazy_json_arrays(lazy_values)
        raise

    encoder = json.JSONEncoder(
        ensure_ascii=False,
        indent=_JSON_INDENT,
        sort_keys=True,
        allow_nan=False,
    )
    total = 0
    try:
        for chunk in encoder.iterencode(payload):
            total += len(chunk.encode("utf-8"))
            if total > max_bytes:
                raise ValueError("qrp_json_render_exceeds_max_bytes")
        return total
    finally:
        _close_lazy_json_arrays(lazy_values)


def _close_lazy_json_arrays(lazy_values: Iterable[_LazyJsonArray]) -> None:
    primary = sys.exception()
    cleanup_failure: BaseException | None = None
    for lazy_value in lazy_values:
        try:
            lazy_value.close()
        except BaseException as exc:
            if primary is None and cleanup_failure is None:
                cleanup_failure = exc
    if cleanup_failure is not None:
        raise cleanup_failure


class _LazyJsonArray(list[Any]):
    """One-pass list protocol for the stdlib encoder's existing array codec."""

    def __init__(
        self,
        source: Iterable[object],
        *,
        depth: int,
        max_bytes: int,
    ) -> None:
        super().__init__()
        self._source = iter(source)
        self._depth = depth
        self._max_bytes = max_bytes
        self._first: object | None = None
        self._has_first: bool | None = None
        self._iterated = False

    def __len__(self) -> int:
        self._prime()
        return 1 if self._has_first else 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        if self._iterated:
            raise RuntimeError("qrp_json_lazy_array_reused")
        self._iterated = True
        self._prime()
        if not self._has_first:
            return
        yield self._native_item(self._first)
        for item in self._source:
            yield self._native_item(item)

    def _native_item(self, item: object) -> Any:
        _require_bounded_json_value_preflight(
            item,
            max_bytes=self._max_bytes,
            depth=self._depth,
        )
        return _json_native_value(item, active=set(), depth=self._depth)

    def _prime(self) -> None:
        if self._has_first is not None:
            return
        try:
            self._first = next(self._source)
        except StopIteration:
            self._has_first = False
        else:
            self._has_first = True

    def close(self) -> None:
        close = getattr(self._source, "close", None)
        if close is not None:
            close()


def canonical_json_hash(value: object) -> str:
    """Hash one typed, collision-framed canonical identity value."""

    return hashlib.sha256(canonical_identity_bytes(value)).hexdigest()


def canonical_identity_bytes(value: object) -> bytes:
    """Return the typed canonical identity encoding for ``value``."""

    try:
        writer = CanonicalIdentityWriter(max_bytes=None)
        writer.value(value)
        return writer.finish()
    except RecursionError as exc:
        raise ValueError("canonical_json_recursion_limit") from exc


class CanonicalIdentityWriter:
    """Internal bounded writer for conceptual canonical identity values."""

    __slots__ = (
        "_active",
        "_scheduling_heuristic_bytes",
        "_buffer",
        "_finished",
        "_manual_base_depth",
        "_manual_counts",
        "_max_bytes",
        "_root_count",
    )

    def __init__(self, max_bytes: int | None) -> None:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("canonical_identity_max_bytes_invalid")
        self._active: set[int] = set()
        self._scheduling_heuristic_bytes = 0
        self._buffer = bytearray()
        self._finished = False
        self._manual_base_depth = 0
        self._manual_counts: list[list[int]] = []
        self._max_bytes = max_bytes
        self._root_count = 0

    def value(self, value: object) -> None:
        self._claim_item()
        self._write_value(
            value,
            depth=self._manual_base_depth + len(self._manual_counts),
        )

    @contextmanager
    def tuple_items(self, item_count: int):  # type: ignore[no-untyped-def]
        if type(item_count) is not int or item_count < 0:
            raise ValueError("canonical_identity_item_count_invalid")
        if self._manual_base_depth + len(self._manual_counts) > _MAX_NESTING_DEPTH:
            raise ValueError("canonical_json_recursion_limit")
        self._claim_item()
        length_offset, payload_start = self._start_frame(b"u")
        self._append(_uint64(item_count))
        state = [item_count, 0]
        self._manual_counts.append(state)
        try:
            yield self
            if state[1] != state[0]:
                raise ValueError("canonical_identity_item_count_mismatch")
        finally:
            self._manual_counts.pop()
        self._finish_frame(length_offset, payload_start)

    @contextmanager
    def canonical_order(
        self,
        item_count: int,
        emit: Callable[[CanonicalIdentityWriter, Any], None],
        *,
        collision_error: str | None = None,
        mapping: dict[object, object] | None = None,
    ):  # type: ignore[no-untyped-def]
        """Yield compact indexes ordered by one conceptual emitted identity."""

        if type(item_count) is not int or item_count < 0:
            raise ValueError("canonical_identity_item_count_invalid")
        if mapping is not None and len(mapping) != item_count:
            raise ValueError("canonical_identity_item_count_mismatch")
        snapshot_pointer_scheduling_heuristic_bytes = (
            0 if mapping is None else _TUPLE_BASE_BYTES + item_count * _POINTER_BYTES
        )
        scheduling_heuristic_bytes = (
            snapshot_pointer_scheduling_heuristic_bytes
            + (item_count + 1 + item_count) * 8
        )
        self._reserve_scheduling_heuristic(scheduling_heuristic_bytes)
        keys: tuple[object, ...] | None = None
        offsets = array("Q")
        order = array("Q")
        checkpoint = len(self._buffer)
        saved_counts = self._manual_counts
        saved_base_depth = self._manual_base_depth
        saved_root_count = self._root_count
        try:
            if mapping is not None:
                keys = tuple(mapping)
                if len(keys) != item_count:
                    raise RuntimeError("canonical_identity_mapping_changed")
            self._manual_base_depth += len(saved_counts)
            self._manual_counts = []
            offsets.append(checkpoint)
            for index in range(item_count):
                self._root_count = 0
                emit(self, index if keys is None else keys[index])
                if self._root_count != 1:
                    raise ValueError("canonical_identity_order_key_required")
                offsets.append(len(self._buffer))
                order.append(index)
            _heap_sort_identity_ranges(self._buffer, offsets, order)
            if collision_error is not None and any(
                _compare_identity_ranges(
                    self._buffer,
                    offsets[order[index]],
                    offsets[order[index] + 1],
                    offsets[order[index - 1]],
                    offsets[order[index - 1] + 1],
                )
                == 0
                for index in range(1, item_count)
            ):
                raise ValueError(collision_error)
            del self._buffer[checkpoint:]
            self._manual_counts = saved_counts
            self._manual_base_depth = saved_base_depth
            self._root_count = saved_root_count
            yield order if keys is None else (keys, order)
        finally:
            if (
                len(self._buffer) > checkpoint
                and self._manual_counts is not saved_counts
            ):
                del self._buffer[checkpoint:]
            self._manual_counts = saved_counts
            self._manual_base_depth = saved_base_depth
            self._root_count = saved_root_count
            self._release_scheduling_heuristic(scheduling_heuristic_bytes)

    def finish(self) -> bytes:
        if self._finished:
            raise RuntimeError("canonical_identity_writer_finished")
        if self._manual_counts or self._root_count != 1:
            raise ValueError("canonical_identity_single_value_required")
        self._require_capacity(len(self._buffer))
        payload = bytes(self._buffer)
        self._buffer.clear()
        self._finished = True
        return payload

    def _claim_item(self) -> None:
        if self._finished:
            raise RuntimeError("canonical_identity_writer_finished")
        if self._manual_counts:
            state = self._manual_counts[-1]
            state[1] += 1
            if state[1] > state[0]:
                raise ValueError("canonical_identity_item_count_mismatch")
        else:
            self._root_count += 1
            if self._root_count > 1:
                raise ValueError("canonical_identity_single_value_required")

    def _write_value(self, value: object, *, depth: int) -> None:
        if depth > _MAX_NESTING_DEPTH:
            raise ValueError("canonical_json_recursion_limit")
        if value is None:
            self._write_frame(b"n", b"")
        elif value is False:
            self._write_frame(b"f", b"")
        elif value is True:
            self._write_frame(b"t", b"")
        elif type(value) is int:
            _validate_integer_size(value, error="canonical_json_integer_too_large")
            self._write_frame(b"i", str(value).encode("ascii"))
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("canonical_json_nonfinite_number")
            normalized = 0.0 if value == 0.0 else value
            self._write_frame(b"r", struct.pack(">d", normalized))
        elif type(value) is str:
            self._write_text_frame(b"s", value)
        elif type(value) is bytes:
            self._write_frame(b"b", value)
        elif value is pd.NaT:
            raise ValueError("canonical_json_invalid_timestamp")
        elif type(value) is Decimal:
            self._write_decimal_frame(value)
        elif type(value) is pd.Timestamp:
            self._write_frame(b"p", _timestamp_payload(value))
        elif type(value) is datetime:
            self._write_frame(b"z", _datetime_payload(value))
        elif type(value) is date:
            self._write_frame(b"a", _date_payload(value))
        elif type(value) is list:
            self._write_sequence(value, tag=b"l", depth=depth)
        elif type(value) is tuple:
            self._write_sequence(value, tag=b"u", depth=depth)
        elif type(value) is dict:
            self._write_mapping(value, depth=depth)
        else:
            raise ValueError("canonical_json_unsupported_value")

    def _write_sequence(
        self, value: list[object] | tuple[object, ...], *, tag: bytes, depth: int
    ) -> None:
        identity = self._activate(value)
        try:
            length_offset, payload_start = self._start_frame(tag)
            self._append(_uint64(len(value)))
            for item in value:
                self._write_value(item, depth=depth + 1)
            self._finish_frame(length_offset, payload_start)
        finally:
            self._active.remove(identity)

    def _write_mapping(self, value: dict[object, object], *, depth: int) -> None:
        identity = self._activate(value)
        try:
            for key in value:
                if type(key) is not str:
                    raise ValueError("canonical_json_non_string_key")

            def emit_key(target: CanonicalIdentityWriter, key: object) -> None:
                assert type(key) is str
                target._claim_item()
                target._append_utf8(key)

            with self.canonical_order(
                len(value),
                emit_key,
                mapping=value,
            ) as (keys, order):
                length_offset, payload_start = self._start_frame(b"m")
                self._append(_uint64(len(keys)))
                for index in order:
                    key = keys[index]
                    assert type(key) is str
                    self._write_text_frame(b"k", key)
                    self._write_value(value[key], depth=depth + 1)
                self._finish_frame(length_offset, payload_start)
        finally:
            self._active.remove(identity)

    def _activate(self, value: object) -> int:
        identity = id(value)
        if identity in self._active:
            raise ValueError("canonical_json_cycle")
        self._active.add(identity)
        return identity

    def _write_frame(self, tag: bytes, payload: bytes) -> None:
        length_offset, payload_start = self._start_frame(tag)
        self._append(payload)
        self._finish_frame(length_offset, payload_start)

    def _write_text_frame(self, tag: bytes, value: str) -> None:
        length_offset, payload_start = self._start_frame(tag)
        self._append_utf8(value)
        self._finish_frame(length_offset, payload_start)

    def _append_utf8(self, value: str) -> None:
        offset = 0
        while offset < len(value):
            chunk_characters = _TEXT_CHUNK_CHARACTERS
            if self._max_bytes is not None:
                remaining = (
                    self._max_bytes
                    - len(self._buffer)
                    - self._scheduling_heuristic_bytes
                )
                if remaining <= 0:
                    self._require_capacity(1)
                chunk_characters = min(
                    chunk_characters,
                    max(1, remaining // 4),
                )
            chunk = value[offset : offset + chunk_characters]
            self._append(_utf8_bytes(chunk))
            offset += len(chunk)

    def _write_decimal_frame(self, value: Decimal) -> None:
        if not value.is_finite():
            raise ValueError("canonical_json_nonfinite_number")
        if self._max_bytes is not None:
            adjusted_exponent = value.adjusted()
            if adjusted_exponent < MIN_EMIN or adjusted_exponent > MAX_EMAX:
                raise ValueError("canonical_identity_exceeds_max_bytes")
            admission = Context(
                prec=_MAX_INTEGER_DECIMAL_DIGITS,
                Emin=MIN_EMIN,
                Emax=MAX_EMAX,
                traps=[Rounded],
            )
            try:
                admission.plus(value)
            except Rounded as exc:
                raise ValueError("canonical_identity_exceeds_max_bytes") from exc
        decimal_tuple = value.as_tuple()
        is_zero = not any(decimal_tuple.digits)
        sign = 0 if is_zero else int(decimal_tuple.sign)
        exponent = str(decimal_tuple.exponent).encode("ascii")
        payload_bytes = 1 + _LENGTH_BYTES + len(decimal_tuple.digits)
        payload_bytes += 1 + _LENGTH_BYTES + len(exponent)
        self._require_capacity(1 + _LENGTH_BYTES + payload_bytes)
        length_offset, payload_start = self._start_frame(b"d")
        self._append(bytes((sign,)))
        self._append(_uint64(len(decimal_tuple.digits)))
        self._append(bytes(decimal_tuple.digits))
        self._write_frame(b"e", exponent)
        self._finish_frame(length_offset, payload_start)

    def _start_frame(self, tag: bytes) -> tuple[int, int]:
        if len(tag) != 1:
            raise AssertionError("canonical_json_tag_must_be_one_byte")
        self._append(tag)
        length_offset = len(self._buffer)
        self._append(b"\x00" * _LENGTH_BYTES)
        return length_offset, len(self._buffer)

    def _finish_frame(self, length_offset: int, payload_start: int) -> None:
        struct.pack_into(
            ">Q",
            self._buffer,
            length_offset,
            len(self._buffer) - payload_start,
        )

    def _append(self, payload: bytes) -> None:
        self._require_capacity(len(payload))
        self._buffer.extend(payload)

    def _require_capacity(self, additional_bytes: int) -> None:
        if self._max_bytes is not None and (
            len(self._buffer)
            + self._scheduling_heuristic_bytes
            + additional_bytes
            > self._max_bytes
        ):
            raise ValueError("canonical_identity_exceeds_max_bytes")

    def _reserve_scheduling_heuristic(self, size: int) -> None:
        self._require_capacity(size)
        self._scheduling_heuristic_bytes += size

    def _release_scheduling_heuristic(self, size: int) -> None:
        self._scheduling_heuristic_bytes -= size


def _compare_identity_ranges(
    buffer: bytearray, left_start: int, left_end: int, right_start: int, right_end: int
) -> int:
    left = memoryview(buffer)[left_start:left_end]
    right = memoryview(buffer)[right_start:right_end]
    for left_byte, right_byte in zip(left, right, strict=False):
        if left_byte != right_byte:
            return -1 if left_byte < right_byte else 1
    return (left_end - left_start > right_end - right_start) - (
        left_end - left_start < right_end - right_start
    )


def _heap_sort_identity_ranges(
    buffer: bytearray, offsets: array[int], order: array[int]
) -> None:
    def greater(left: int, right: int) -> bool:
        return (
            _compare_identity_ranges(
                buffer,
                offsets[left],
                offsets[left + 1],
                offsets[right],
                offsets[right + 1],
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


def _strict_object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("qrp_json_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> NoReturn:
    raise ValueError("qrp_json_nonfinite_number")


def _parse_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_INTEGER_DECIMAL_DIGITS:
        raise ValueError("qrp_json_integer_too_large")
    return int(value)


def _require_bounded_json_preflight(
    value: Mapping[str, Any],
    *,
    max_bytes: int,
) -> None:
    """Reject an over-limit payload before copying its native containers."""

    if type(value) is not dict:
        raise ValueError("qrp_json_top_level_object_required")
    _require_bounded_json_value_preflight(value, max_bytes=max_bytes, depth=0)


def _require_bounded_json_value_preflight(
    value: object,
    *,
    max_bytes: int,
    depth: int,
) -> None:
    """Reject one over-limit value before copying its native containers."""

    remaining = max_bytes
    active: set[int] = set()

    def claim(size: int) -> None:
        nonlocal remaining
        remaining -= size
        if remaining < 0:
            raise ValueError("qrp_json_render_exceeds_max_bytes")

    def claim_text(text: str) -> None:
        claim(2)
        for offset in range(0, len(text), _TEXT_CHUNK_CHARACTERS):
            rendered = json.dumps(
                text[offset : offset + _TEXT_CHUNK_CHARACTERS],
                ensure_ascii=False,
            )
            claim(len(_utf8_bytes(rendered)) - 2)

    def claim_container_whitespace(*, item_count: int, depth: int) -> None:
        claim(2)
        if item_count == 0:
            return
        claim(item_count - 1)
        claim(item_count + 1)
        claim(_JSON_INDENT * (item_count * (depth + 1) + depth))

    def visit(item: object, *, depth: int) -> None:
        if depth > _MAX_NESTING_DEPTH:
            raise ValueError("qrp_json_recursion_limit")
        if item is None:
            claim(4)
        elif type(item) is bool:
            claim(4 if item else 5)
        elif type(item) is int:
            _validate_integer_size(item, error="qrp_json_integer_too_large")
            claim(len(str(item)))
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("qrp_json_nonfinite_number")
            claim(len(repr(item)))
        elif type(item) is str:
            claim_text(item)
        elif type(item) is list:
            identity = id(item)
            if identity in active:
                raise ValueError("qrp_json_cycle")
            active.add(identity)
            try:
                claim_container_whitespace(item_count=len(item), depth=depth)
                for child in item:
                    visit(child, depth=depth + 1)
            finally:
                active.remove(identity)
        elif type(item) is dict:
            identity = id(item)
            if identity in active:
                raise ValueError("qrp_json_cycle")
            active.add(identity)
            try:
                claim_container_whitespace(item_count=len(item), depth=depth)
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("qrp_json_non_string_key")
                    if key in _RETIRED_JSON_IDENTITY_KEYS:
                        raise ValueError("qrp_json_retired_identity_key")
                    claim_text(key)
                    claim(2)
                    visit(child, depth=depth + 1)
            finally:
                active.remove(identity)
        else:
            raise ValueError("qrp_json_unsupported_value")

    visit(value, depth=depth)


def _json_native_object(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("qrp_json_top_level_object_required")
    return _json_native_mapping(value, active=set(), depth=0)


def _json_native_mapping(
    value: Mapping[object, object],
    *,
    active: set[int],
    depth: int,
) -> dict[str, Any]:
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError("qrp_json_recursion_limit")
    identity = id(value)
    if identity in active:
        raise ValueError("qrp_json_cycle")
    active.add(identity)
    result: dict[str, Any] = {}
    try:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("qrp_json_non_string_key")
            _utf8_bytes(key)
            if key in _RETIRED_JSON_IDENTITY_KEYS:
                raise ValueError("qrp_json_retired_identity_key")
            result[key] = _json_native_value(
                item,
                active=active,
                depth=depth + 1,
            )
        return result
    finally:
        active.remove(identity)


def _json_native_value(value: object, *, active: set[int], depth: int) -> Any:
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError("qrp_json_recursion_limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        _validate_integer_size(value, error="qrp_json_integer_too_large")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("qrp_json_nonfinite_number")
        return value
    if type(value) is str:
        _utf8_bytes(value)
        return value
    if type(value) is list:
        identity = id(value)
        if identity in active:
            raise ValueError("qrp_json_cycle")
        active.add(identity)
        try:
            return [
                _json_native_value(item, active=active, depth=depth + 1)
                for item in value
            ]
        finally:
            active.remove(identity)
    if type(value) is dict:
        return _json_native_mapping(value, active=active, depth=depth)
    raise ValueError("qrp_json_unsupported_value")


def _frame(tag: bytes, payload: bytes) -> bytes:
    if len(tag) != 1:
        raise AssertionError("canonical_json_tag_must_be_one_byte")
    return tag + _uint64(len(payload)) + payload


def _uint64(value: int) -> bytes:
    if value < 0 or value >= 1 << (_LENGTH_BYTES * 8):
        raise ValueError("canonical_json_value_too_large")
    return value.to_bytes(_LENGTH_BYTES, byteorder="big", signed=False)


def _validate_integer_size(value: int, *, error: str) -> None:
    if value.bit_length() > _MAX_INTEGER_BITS:
        raise ValueError(error)
    if len(str(abs(value))) > _MAX_INTEGER_DECIMAL_DIGITS:
        raise ValueError(error)


def _timestamp_payload(value: pd.Timestamp) -> bytes:
    if pd.isna(value):
        raise ValueError("canonical_json_invalid_timestamp")
    unit = value.unit
    integer_value = int(value.asm8.view("i8"))
    return b"".join(
        (
            _frame(b"q", unit.encode("ascii")),
            _frame(b"v", str(integer_value).encode("ascii")),
            _frame(b"o", _timezone_payload(value.tzinfo, value)),
        )
    )


def _datetime_payload(value: datetime) -> bytes:
    if pd.isna(value):
        raise ValueError("canonical_json_invalid_timestamp")
    components = (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        value.fold,
    )
    return b"".join(
        (
            _frame(
                b"c",
                b",".join(str(item).encode("ascii") for item in components),
            ),
            _frame(b"o", _timezone_payload(value.tzinfo, value)),
        )
    )


def _date_payload(value: date) -> bytes:
    return b",".join(
        str(item).encode("ascii") for item in (value.year, value.month, value.day)
    )


def _timezone_payload(tzinfo: object | None, value: datetime) -> bytes:
    if tzinfo is None:
        return b""
    timezone_type = type(tzinfo)
    offset = value.utcoffset()
    offset_microseconds = (
        None if offset is None else int(offset.total_seconds() * 1_000_000)
    )
    name = value.tzname()
    return b"".join(
        (
            _frame(b"j", _utf8_bytes(timezone_type.__module__)),
            _frame(b"h", _utf8_bytes(timezone_type.__qualname__)),
            _frame(b"x", _utf8_bytes(str(tzinfo))),
            _frame(
                b"w",
                (
                    b""
                    if offset_microseconds is None
                    else str(offset_microseconds).encode("ascii")
                ),
            ),
            _frame(b"y", b"" if name is None else _utf8_bytes(name)),
        )
    )


def _utf8_bytes(value: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("qrp_json_invalid_unicode") from exc


__all__ = [
    "bounded_rendered_qrp_json_object_size",
    "canonical_identity_bytes",
    "canonical_json_hash",
    "parse_qrp_json_object",
    "render_qrp_json_object",
]

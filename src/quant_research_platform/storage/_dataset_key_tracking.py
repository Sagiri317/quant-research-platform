"""Exact spillable primary-key tracking for dataset writes."""

from __future__ import annotations

import math
import sqlite3
import struct
import tempfile
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from uuid import UUID

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series


class ExactPrimaryKeyTracker:
    """Track complete canonical primary keys in a disk-backed unique index."""

    def __init__(
        self,
        *,
        primary_key: Sequence[str],
        timestamp_columns: Sequence[str],
        temp_parent: Path | None = None,
    ) -> None:
        self._primary_key = tuple(str(column) for column in primary_key)
        self._timestamp_columns = frozenset(
            str(column) for column in timestamp_columns
        )
        self._temp_parent = None if temp_parent is None else Path(temp_parent)
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._connection: sqlite3.Connection | None = None

    @property
    def database_path(self) -> Path:
        if self._temporary_directory is None:
            raise RuntimeError("exact_primary_key_tracker_not_open")
        return Path(self._temporary_directory.name) / "primary-keys.sqlite3"

    def __enter__(self) -> ExactPrimaryKeyTracker:
        if self._connection is not None:
            raise RuntimeError("exact_primary_key_tracker_already_open")
        if self._temp_parent is not None:
            self._temp_parent.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="qrp-exact-primary-keys-",
            dir=self._temp_parent,
        )
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute(
                "CREATE TABLE dataset_keys (payload BLOB PRIMARY KEY) WITHOUT ROWID"
            )
            connection.commit()
        except Exception:
            connection.close()
            self._temporary_directory.cleanup()
            self._temporary_directory = None
            raise
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

    def add_frame(self, frame: pd.DataFrame) -> int:
        connection = self._connection
        if connection is None:
            raise RuntimeError("exact_primary_key_tracker_not_open")
        missing = [column for column in self._primary_key if column not in frame]
        if missing:
            raise ValueError("missing_dataset_primary_key_column")
        keys = frame.loc[:, list(self._primary_key)].copy()
        for column in self._timestamp_columns:
            keys[column] = strict_timestamp_series(
                keys[column],
                error="invalid_dataset_timestamp_value",
            )
        before = connection.total_changes
        connection.executemany(
            "INSERT OR IGNORE INTO dataset_keys(payload) VALUES (?)",
            self._payload_rows(keys),
        )
        connection.commit()
        inserted = connection.total_changes - before
        return int(len(keys) - inserted)

    def _payload_rows(self, keys: pd.DataFrame) -> Iterator[tuple[bytes]]:
        timestamp_positions = {
            index
            for index, column in enumerate(self._primary_key)
            if column in self._timestamp_columns
        }
        for row in keys.itertuples(index=False, name=None):
            payload = b"".join(
                _canonical_scalar_payload(
                    value,
                    timestamp=index in timestamp_positions,
                )
                for index, value in enumerate(row)
            )
            yield (payload,)


def _canonical_scalar_payload(value: object, *, timestamp: bool) -> bytes:
    if timestamp:
        parsed = pd.Timestamp(value)
        if parsed.tzinfo is not None:
            parsed = parsed.tz_convert("UTC").tz_localize(None)
        return _framed(b"t", struct.pack(">q", int(parsed.value)))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return _framed(b"s", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _framed(b"y", value)
    if isinstance(value, bool):
        return _framed(b"b", b"1" if value else b"0")
    if isinstance(value, int):
        return _framed(b"i", str(value).encode("ascii"))
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("dataset_primary_key_null")
        if value == 0.0:
            value = 0.0
        return _framed(b"f", struct.pack(">d", value))
    if isinstance(value, Decimal):
        if value == 0:
            value = Decimal(0)
        return _framed(b"d", str(value.normalize()).encode("ascii"))
    if isinstance(value, UUID):
        return _framed(b"u", value.bytes)
    if isinstance(value, datetime):
        return _framed(b"m", value.isoformat().encode("ascii"))
    if isinstance(value, date):
        return _framed(b"a", value.isoformat().encode("ascii"))
    if isinstance(value, tuple):
        return _framed(
            b"q",
            b"".join(
                _canonical_scalar_payload(item, timestamp=False) for item in value
            ),
        )
    raise ValueError("invalid_dataset_primary_key_value")


def _framed(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">Q", len(payload)) + payload

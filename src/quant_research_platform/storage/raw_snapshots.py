"""Immutable raw CSV snapshots with a single metadata sidecar."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from quant_research_platform.contracts.json_codec import canonical_identity_bytes
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    filesystem_iterdir,
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_manifest, write_manifest

SCHEMA_ID = "qrp-raw-csv-snapshot"
_CELL_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))


def _decode_bytes(value: dict[str, object]) -> bytes:
    if set(value) != {"bytes_hex"} or type(value["bytes_hex"]) is not str:
        raise ValueError("raw_csv_value_type_invalid")
    return bytes.fromhex(value["bytes_hex"])


_CELL_DECODER = json.JSONDecoder(object_hook=_decode_bytes)


def encode_raw_cell(value: object) -> str:
    """Preserve SDK scalar types, including nonfinite source values and bytes."""
    active: set[int] = set()

    def native(item: object) -> object:
        if type(item) is bytes:
            return {"bytes_hex": item.hex()}
        if type(item) is list:
            identity = id(item)
            if identity in active:
                raise ValueError("raw_csv_cycle")
            active.add(identity)
            try:
                return [native(child) for child in item]
            finally:
                active.remove(identity)
        if item is None or type(item) in (str, int, float, bool):
            return item
        raise ValueError("raw_csv_value_type_invalid")

    return _CELL_ENCODER.encode(native(value))


def decode_raw_cell(text: str) -> object:
    value = _CELL_DECODER.decode(text)
    if encode_raw_cell(value) != text:
        raise ValueError("raw_csv_cell_not_canonical")
    return value


def _record(values: Iterable[str]) -> bytes:
    buffer = io.StringIO(newline="")

    csv.writer(buffer, lineterminator="\n").writerow(values)
    return buffer.getvalue().encode("utf-8")


def write_csv_snapshot(
    root: Path,
    *,
    columns: tuple[str, ...],
    rows: Iterable[tuple[object, ...]],
    metadata: dict[str, object],
    validate: Callable[[Path], object],
) -> None:
    """Publish caller-provided rows atomically."""
    if (
        not columns
        or any(
            type(value) is not str or not value or "\n" in value or ("\r" in value)
            for value in columns
        )
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("raw_csv_columns_invalid")

    def write(staging: Path) -> None:
        digest = hashlib.sha256()
        size = count = 0
        handle = open_filesystem_file(staging / "response.csv", "xb")
        with closing_iterator(handle):
            header = _record(columns)
            handle.write(header)
            digest.update(header)
            size += len(header)
            for row in rows:
                if len(row) != len(columns):
                    raise ValueError("raw_csv_row_width_invalid")
                record = _record(encode_raw_cell(value) for value in row)
                size += len(record)
                handle.write(record)
                digest.update(record)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        write_manifest(
            staging / "response.metadata.json",
            {
                "schema_id": SCHEMA_ID,
                "encoding": "utf-8-json-cells",
                "columns": list(columns),
                "row_count": count,
                "byte_count": size,
                "sha256": digest.hexdigest(),
                "metadata": metadata,
            },
        )
        validate(staging)

    atomic_write_directory(root, write)


def iter_csv_snapshot(
    root: Path, *, expected_manifest: dict[str, object]
) -> Iterator[tuple[object, ...]]:
    """Validate source bytes during one complete acceptance traversal."""
    root = require_unaliased_path(root, error="raw_csv_path_invalid")
    if set(path.name for path in filesystem_iterdir(root)) != {
        "response.csv",
        "response.metadata.json",
    }:
        raise ValueError("raw_csv_members_invalid")
    sidecar = root / "response.metadata.json"
    manifest = read_manifest(sidecar)
    if (
        set(manifest)
        != {
            "schema_id",
            "encoding",
            "columns",
            "row_count",
            "byte_count",
            "sha256",
            "metadata",
        }
        or manifest["schema_id"] != SCHEMA_ID
        or manifest["encoding"] != "utf-8-json-cells"
        or (
            canonical_identity_bytes(manifest)
            != canonical_identity_bytes(expected_manifest)
        )
        or (type(manifest["byte_count"]) is not int)
        or (manifest["byte_count"] < 0)
        or (type(manifest["row_count"]) is not int)
        or (manifest["row_count"] < 0)
    ):
        raise ValueError("raw_csv_metadata_invalid")
    path = root / "response.csv"
    handle = open_filesystem_file(path)
    digest = hashlib.sha256()
    count = 0
    with closing_iterator(handle):

        def records() -> Iterator[list[str]]:
            csv.field_size_limit(sys.maxsize)
            while line := handle.readline():
                if not line.endswith(b"\n"):
                    raise ValueError("raw_csv_record_invalid")
                digest.update(line)
                try:
                    values = next(csv.reader([line.decode("utf-8")], strict=True))
                except (csv.Error, UnicodeError) as exc:
                    raise ValueError("raw_csv_record_invalid") from exc
                if _record(values) != line:
                    raise ValueError("raw_csv_record_not_canonical")
                yield values

        stream = records()
        if next(stream, None) != manifest["columns"]:
            raise ValueError("raw_csv_columns_invalid")

        def decoded_rows():
            nonlocal count
            for row in stream:
                if len(row) != len(manifest["columns"]):
                    raise ValueError("raw_csv_row_width_invalid")
                value = tuple(decode_raw_cell(cell) for cell in row)
                count += 1
                yield value

        with closing_iterator(decoded_rows()) as rows:
            for row in rows:
                try:
                    yield row
                except GeneratorExit:
                    return
        if (
            count != manifest["row_count"]
            or digest.hexdigest() != manifest["sha256"]
            or handle.tell() != manifest["byte_count"]
        ):
            raise ValueError("raw_csv_content_invalid")

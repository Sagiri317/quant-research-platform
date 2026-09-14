"""Immutable, bounded raw CSV snapshots with a single metadata sidecar."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from quant_research_platform.runtime.resources import require_free_disk_headroom
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    dataset_file_identity_from_stat,
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_manifest, write_manifest

SCHEMA_ID = "qrp-raw-csv-snapshot"
MAX_RECORD_BYTES = 1024 * 1024
_MAX_CELL_CHARACTERS = 64 * 1024


def encode_raw_cell(value: object, *, _depth: int = 0) -> str:
    """JSON preserves SDK strings/lists/null/bool/int/float without CSV coercion."""
    if _depth > 16 or (
        type(value) in (str, bytes, list) and len(value) > _MAX_CELL_CHARACTERS
    ):
        raise ValueError("raw_csv_cell_bound_exceeded")
    if type(value) is bytes:
        return json.dumps({"bytes_hex": value.hex()}, separators=(",", ":"))
    if value is not None and type(value) not in (str, int, float, bool, list):
        raise ValueError("raw_csv_value_type_invalid")
    if type(value) is list:
        # SDK date lists must remain lists, including lists containing nulls.
        parts = []
        size = 2
        for item in value:
            part = encode_raw_cell(item, _depth=_depth + 1)
            size += len(part) + 1
            if size > _MAX_CELL_CHARACTERS:
                raise ValueError("raw_csv_cell_bound_exceeded")
            parts.append(part)
        return "[" + ",".join(parts) + "]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def decode_raw_cell(text: str) -> object:
    def hook(value: dict[str, object]) -> bytes:
        if set(value) != {"bytes_hex"} or type(value["bytes_hex"]) is not str:
            raise ValueError("raw_csv_value_type_invalid")
        return bytes.fromhex(value["bytes_hex"])

    value = json.loads(text, object_hook=hook)
    if encode_raw_cell(value) != text:
        raise ValueError("raw_csv_cell_not_canonical")
    return value


def _record(values: Iterable[str]) -> bytes:
    buffer = io.StringIO(newline="")

    def bounded_values() -> Iterator[str]:
        total = 0
        for value in values:
            total += len(value)
            if len(value) > _MAX_CELL_CHARACTERS or total > MAX_RECORD_BYTES:
                raise ValueError("raw_csv_cell_bound_exceeded")
            yield value

    csv.writer(buffer, lineterminator="\n").writerow(bounded_values())
    value = buffer.getvalue().encode("utf-8")
    if len(value) > MAX_RECORD_BYTES:
        raise ValueError("raw_csv_record_bound_exceeded")
    return value


def write_csv_snapshot(
    root: Path,
    *,
    columns: tuple[str, ...],
    rows: Iterable[tuple[object, ...]],
    metadata: dict[str, object],
    max_bytes: int,
    validate: Callable[[Path], object],
) -> None:
    """Caller owns admission and borrowed rows through atomic publication."""
    if (
        not columns
        or any(
            type(value) is not str or not value or "\n" in value or "\r" in value
            for value in columns
        )
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("raw_csv_columns_invalid")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("raw_csv_byte_bound_invalid")
    require_free_disk_headroom(root, reserve_bytes=max_bytes + MAX_RECORD_BYTES)

    def write(staging: Path) -> None:
        digest = hashlib.sha256()
        size = count = 0
        handle = open_filesystem_file(staging / "response.csv", "xb")
        with handle_preserving_primary(handle):
            header = _record(columns)
            if len(header) > max_bytes:
                raise ValueError("raw_csv_byte_bound_exceeded")
            handle.write(header)
            digest.update(header)
            size += len(header)
            for row in rows:
                if len(row) != len(columns):
                    raise ValueError("raw_csv_row_width_invalid")
                record = _record(encode_raw_cell(value) for value in row)
                size += len(record)
                if size > max_bytes:
                    raise ValueError("raw_csv_byte_bound_exceeded")
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
        # The source validator fully consumes the persisted table before publish.
        validate(staging)

    atomic_write_directory(root, write)


def iter_csv_snapshot(
    root: Path,
    *,
    expected_manifest: dict[str, object],
    max_bytes: int,
) -> Iterator[tuple[object, ...]]:
    """Consume fully before exposing results; borrowed rows verify at exhaustion."""
    root = require_unaliased_path(root, error="raw_csv_path_invalid")
    if set(path.name for path in root.iterdir()) != {
        "response.csv",
        "response.metadata.json",
    }:
        raise ValueError("raw_csv_members_invalid")
    sidecar = root / "response.metadata.json"
    sidecar_identity = dataset_file_identity(sidecar)
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
        or manifest != expected_manifest
        or type(manifest["row_count"]) is not int
        or manifest["row_count"] < 0
    ):
        raise ValueError("raw_csv_metadata_invalid")
    path = root / "response.csv"
    identity = dataset_file_identity(path)
    if identity.size != manifest["byte_count"] or identity.size > max_bytes:
        raise ValueError("raw_csv_byte_count_invalid")
    handle = open_filesystem_file(path)
    digest = hashlib.sha256()
    count = 0
    with handle_preserving_primary(handle):
        if dataset_file_identity_from_stat(os.fstat(handle.fileno())) != identity:
            raise ValueError("raw_csv_identity_changed")

        # JSON escapes embedded newlines; every CSV record is one physical line.
        def records() -> Iterator[list[str]]:
            while line := handle.readline(MAX_RECORD_BYTES + 1):
                if len(line) > MAX_RECORD_BYTES or not line.endswith(b"\n"):
                    raise ValueError("raw_csv_record_bound_exceeded")
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

        rows = decoded_rows()
        try:
            # Keep the inner iterator open on GeneratorExit so verification drains it.
            for row in rows:  # noqa: UP028
                yield row
        finally:
            primary = sys.exception()
            try:
                for _row in rows:
                    pass
                if (
                    count != manifest["row_count"]
                    or digest.hexdigest() != manifest["sha256"]
                    or dataset_file_identity_from_stat(os.fstat(handle.fileno()))
                    != identity
                    or dataset_file_identity(path) != identity
                    or dataset_file_identity(sidecar) != sidecar_identity
                ):
                    raise ValueError("raw_csv_content_invalid")
            except BaseException as error:
                if primary is None or isinstance(primary, GeneratorExit):
                    raise
                primary.add_note(
                    "Raw CSV consumed-content verification also failed: "
                    + type(error).__name__
                )

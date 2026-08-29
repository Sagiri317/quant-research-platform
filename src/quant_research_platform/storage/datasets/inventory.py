"""Spillable exact part inventory owned by Storage DatasetSession."""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa

from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    require_free_disk_headroom,
)
from quant_research_platform.storage._scratch_ownership import (
    owned_scratch_name,
    remove_owned_scratch,
)
from quant_research_platform.storage.datasets.paths import parse_year_partition_label
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    require_unaliased_path,
)
from quant_research_platform.storage.parquet import DatasetPartMetadata

_PREFIX = ".i-"
_PART_FILE_PATTERN = re.compile(r"^part-(\d+)\.parquet$")
DATASET_INVENTORY_RESERVE_BYTES = 1024
_DATASET_INVENTORY_SQLITE_STARTUP_RESERVE_BYTES = 32 * DATASET_INVENTORY_RESERVE_BYTES


class DatasetInventoryPaths:
    """Minimal ordered path view without slice or materialization semantics."""

    def __init__(
        self,
        inventory: DatasetSessionInventory,
        *,
        selected: bool,
        path_selection: int | None = None,
    ) -> None:
        self._inventory = inventory
        self._selected = selected
        self._path_selection = path_selection

    def __len__(self) -> int:
        return self._inventory.path_count(
            selected=self._selected,
            path_selection=self._path_selection,
        )

    def __iter__(self) -> Iterator[Path]:
        yield from self._inventory.iter_paths(
            selected=self._selected,
            path_selection=self._path_selection,
        )

    def __getitem__(self, index: int) -> Path:
        if isinstance(index, slice):
            raise TypeError("dataset_inventory_path_slicing_unsupported")
        if index < 0:
            index += len(self)
        return self._inventory.path_at(
            index,
            selected=self._selected,
            path_selection=self._path_selection,
        )


class DatasetSessionInventory:
    """Disk-backed physical discovery and ordered part metadata owner."""

    def __init__(
        self,
        *,
        data_root: Path,
        scratch_parent: Path,
        resource_policy: RuntimeResourcePolicy,
    ) -> None:
        self.data_root = Path(data_root)
        self._scratch_parent = Path(scratch_parent)
        self._temporary_directory: Path | None = None
        self._connection: sqlite3.Connection | None = None
        self._next_selection_id = 0
        self._lease = resource_policy.lease(
            "storage.dataset.inventory",
            _DATASET_INVENTORY_SQLITE_STARTUP_RESERVE_BYTES,
        )
        temporary_directory: Path | None = None
        claimed_temporary = False
        try:
            require_unaliased_path(
                self._scratch_parent, error="invalid_storage_scratch_path"
            )
            self._scratch_parent.mkdir(parents=True, exist_ok=True)
            require_unaliased_path(
                self._scratch_parent, error="invalid_storage_scratch_path"
            )
            require_free_disk_headroom(
                self._scratch_parent,
                reserve_bytes=_DATASET_INVENTORY_SQLITE_STARTUP_RESERVE_BYTES,
            )
            require_unaliased_path(
                self._scratch_parent, error="invalid_storage_scratch_path"
            )
            temporary_directory = self._scratch_parent / owned_scratch_name(
                _PREFIX, uuid4().hex[:8]
            )
            os.mkdir(temporary_directory)
            claimed_temporary = True
            self._temporary_directory = temporary_directory
            require_unaliased_path(
                temporary_directory,
                error="invalid_storage_scratch_path",
            )
            self._connection = sqlite3.connect(temporary_directory / "i.db")
            connection = self._connection
            connection.execute(f"PRAGMA page_size={DATASET_INVENTORY_RESERVE_BYTES}")
            sqlite_page_limit = (
                _DATASET_INVENTORY_SQLITE_STARTUP_RESERVE_BYTES
                // DATASET_INVENTORY_RESERVE_BYTES
            )
            connection.execute(f"PRAGMA main.max_page_count={sqlite_page_limit}")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute(
                f"PRAGMA cache_size=-{DATASET_INVENTORY_RESERVE_BYTES // 1024}"
            )
            connection.execute(
                "CREATE TABLE year_labels (year INTEGER PRIMARY KEY, "
                "label TEXT UNIQUE NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE parts (relative_path TEXT PRIMARY KEY, "
                "partition_kind INTEGER NOT NULL, partition_year INTEGER NOT NULL, "
                "part_index INTEGER NOT NULL, selected INTEGER NOT NULL, "
                "device TEXT NOT NULL, inode TEXT NOT NULL, size TEXT NOT NULL, "
                "modified_ns TEXT NOT NULL, schema BLOB, row_count INTEGER, "
                "file_size_bytes INTEGER, logical_dtypes BLOB, "
                "UNIQUE(partition_kind, partition_year, part_index), "
                "UNIQUE(device, inode))"
            )
            connection.execute(
                "CREATE TABLE path_selections ("
                "selection_id INTEGER NOT NULL, relative_path TEXT NOT NULL, "
                "PRIMARY KEY(selection_id, relative_path))"
            )
            self._lease.shrink_to(DATASET_INVENTORY_RESERVE_BYTES)
        except BaseException as primary:
            if claimed_temporary and self._temporary_directory is None:
                object.__setattr__(self, "_temporary_directory", temporary_directory)
            try:
                self.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    "dataset inventory construction cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            raise

    def __enter__(self) -> DatasetSessionInventory:
        return self

    def __exit__(
        self,
        _exception_type: object,
        primary: BaseException | None,
        _traceback: object,
    ) -> None:
        if primary is None:
            self.close()
            return
        try:
            self.close()
        except BaseException as cleanup_error:
            primary.add_note(
                "dataset inventory cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )

    def close(self) -> None:
        primary = sys.exception()
        first_error: BaseException | None = None
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except BaseException as connection_error:
                if primary is not None:
                    primary.add_note(
                        "dataset inventory connection close also failed: "
                        f"{type(connection_error).__name__}"
                    )
                else:
                    first_error = connection_error

        temporary = self._temporary_directory
        self._temporary_directory = None
        if temporary is not None:
            try:
                remove_owned_scratch(temporary)
            except BaseException as temporary_error:
                if primary is not None:
                    primary.add_note(
                        "dataset inventory scratch cleanup also failed: "
                        f"{type(temporary_error).__name__}"
                    )
                elif first_error is None:
                    first_error = temporary_error
                else:
                    first_error.add_note(
                        "dataset inventory scratch cleanup also failed: "
                        f"{type(temporary_error).__name__}"
                    )

        try:
            self._lease.close()
        except BaseException as lease_error:
            if primary is not None:
                primary.add_note(
                    "dataset inventory lease release also failed: "
                    f"{type(lease_error).__name__}"
                )
            elif first_error is None:
                first_error = lease_error
            else:
                first_error.add_note(
                    "dataset inventory lease release also failed: "
                    f"{type(lease_error).__name__}"
                )
        if first_error is not None:
            raise first_error

    @property
    def scratch_path(self) -> Path:
        if self._temporary_directory is None:
            raise RuntimeError("dataset_inventory_closed")
        return self._temporary_directory

    def discover(self, spec: Mapping[str, Any]) -> None:
        """Stream the physical tree into the sole exact inventory owner."""

        self._scan_filesystem(
            spec,
            on_year=self._register_year,
            on_path=self._register_candidate,
        )

    def verify_filesystem(self, spec: Mapping[str, Any]) -> None:
        """Stream-compare current paths and identities with frozen inventory."""

        observed_count = 0

        def verify_year(year: int, label: str) -> None:
            row = (
                self._require_connection()
                .execute("SELECT label FROM year_labels WHERE year = ?", (year,))
                .fetchone()
            )
            if row is None or str(row[0]) != label:
                raise ValueError("dataset_inventory_changed")

        def verify_path(path: Path, coordinates: tuple[int, int, int]) -> None:
            nonlocal observed_count
            observed_count += 1
            relative = path.relative_to(self.data_root).as_posix()
            row = (
                self._require_connection()
                .execute(
                    "SELECT partition_kind, partition_year, part_index, device, "
                    "inode, size, modified_ns FROM parts WHERE relative_path = ?",
                    (relative,),
                )
                .fetchone()
            )
            if row is None or tuple(int(value) for value in row[:3]) != coordinates:
                raise ValueError("dataset_inventory_changed")
            identity = dataset_file_identity(path)
            if tuple(str(value) for value in row[3:]) != _identity_values(identity):
                raise ValueError("dataset_file_identity_changed")

        self._scan_filesystem(spec, on_year=verify_year, on_path=verify_path)
        if observed_count != self.path_count(selected=False):
            raise ValueError("dataset_inventory_changed")

    def iter_candidate_paths(self) -> Iterator[Path]:
        cursor = self._require_connection().execute(
            "SELECT relative_path FROM parts "
            "ORDER BY partition_kind, partition_year, part_index"
        )
        try:
            for row in cursor:
                yield self.data_root / str(row[0])
        finally:
            _close_cursor_preserving_primary(cursor)

    def add(self, path: Path, metadata: DatasetPartMetadata, *, selected: bool) -> None:
        """Insert one writer-observed part using its exact numeric order key."""

        coordinates = self._validated_coordinates(Path(path), partitioned=None)
        partition_kind, partition_year, part_index = coordinates
        if partition_kind:
            self._register_year(partition_year, Path(path).parent.name)
        relative = Path(path).relative_to(self.data_root).as_posix()
        identity = metadata.identity
        values = (
            relative,
            partition_kind,
            partition_year,
            part_index,
            int(selected),
            *_identity_values(identity),
            metadata.schema.serialize().to_pybytes(),
            metadata.row_count,
            metadata.file_size_bytes,
            metadata.logical_dtype_payload,
        )
        self._gate_row_write(values)
        try:
            with self._require_connection():
                self._require_connection().execute(
                    "INSERT INTO parts(relative_path, partition_kind, "
                    "partition_year, part_index, selected, device, inode, size, "
                    "modified_ns, schema, row_count, file_size_bytes, "
                    "logical_dtypes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("invalid_dataset_part_files") from exc

    def observe(
        self,
        path: Path,
        metadata: DatasetPartMetadata,
        *,
        selected: bool,
    ) -> None:
        """Attach bounded Parquet metadata to one registered physical candidate."""

        relative = Path(path).relative_to(self.data_root).as_posix()
        identity = metadata.identity
        row = (
            self._require_connection()
            .execute(
                "SELECT partition_kind, partition_year, part_index, "
                "device, inode, size, modified_ns FROM parts WHERE relative_path = ?",
                (relative,),
            )
            .fetchone()
        )
        if row is None:
            raise ValueError("dataset_inventory_changed")
        if tuple(str(value) for value in row[3:]) != _identity_values(identity):
            raise ValueError("dataset_file_identity_changed")
        schema_payload = metadata.schema.serialize().to_pybytes()
        prospective_row = (
            relative,
            *row[:3],
            int(selected),
            *row[3:],
            schema_payload,
            metadata.row_count,
            metadata.file_size_bytes,
            metadata.logical_dtype_payload,
        )
        values = (
            int(selected),
            schema_payload,
            metadata.row_count,
            metadata.file_size_bytes,
            metadata.logical_dtype_payload,
            relative,
        )
        self._gate_row_write(prospective_row)
        with self._require_connection():
            self._require_connection().execute(
                "UPDATE parts SET selected = ?, schema = ?, row_count = ?, "
                "file_size_bytes = ?, logical_dtypes = ? "
                "WHERE relative_path = ?",
                values,
            )

    def new_path_selection(self) -> int:
        self._next_selection_id += 1
        return self._next_selection_id

    def select_path(self, selection_id: int, path: Path) -> None:
        if type(selection_id) is not int or selection_id <= 0:
            raise ValueError("invalid_dataset_path_selection")
        relative = Path(path).relative_to(self.data_root).as_posix()
        connection = self._require_connection()
        before = connection.total_changes
        self._gate_row_write((selection_id, relative))
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO path_selections(selection_id, relative_path) "
                "SELECT ?, relative_path FROM parts "
                "WHERE relative_path = ? AND selected = 1",
                (selection_id, relative),
            )
        if connection.total_changes == before:
            row = connection.execute(
                "SELECT 1 FROM path_selections "
                "WHERE selection_id = ? AND relative_path = ?",
                (selection_id, relative),
            ).fetchone()
            if row is None:
                raise ValueError("dataset_path_not_selected")

    def release_path_selection(self, selection_id: int) -> None:
        connection = self._require_connection()
        reserve = connection.execute(
            "SELECT COALESCE(SUM(18 + length(CAST(selection_id AS BLOB)) "
            "+ length(CAST(relative_path AS BLOB))), 0), COUNT(*) "
            "FROM path_selections WHERE selection_id = ?",
            (selection_id,),
        ).fetchone()
        assert reserve is not None
        self._gate_physical_write(
            logical_bytes=int(reserve[0]),
            row_count=int(reserve[1]),
        )
        with connection:
            connection.execute(
                "DELETE FROM path_selections WHERE selection_id = ?",
                (selection_id,),
            )

    def iter_entries(
        self,
        *,
        selected: bool = False,
        path_selection: int | None = None,
    ) -> Iterator[tuple[Path, DatasetPartMetadata]]:
        join, where, parameters = self._path_selection_query(
            selected=selected,
            path_selection=path_selection,
        )
        cursor = self._require_connection().execute(
            "SELECT parts.relative_path, device, inode, size, modified_ns, schema, "
            "row_count, file_size_bytes, logical_dtypes FROM parts"
            f"{join}{where} ORDER BY partition_kind, partition_year, part_index",
            parameters,
        )
        try:
            for row in cursor:
                relative = str(row[0])
                yield (
                    self.data_root / relative,
                    _metadata_from_row(row[1:]),
                )
        finally:
            _close_cursor_preserving_primary(cursor)

    def iter_paths(
        self,
        *,
        selected: bool,
        path_selection: int | None = None,
    ) -> Iterator[Path]:
        for path, _ in self.iter_entries(
            selected=selected,
            path_selection=path_selection,
        ):
            yield path

    def path_count(
        self,
        *,
        selected: bool,
        path_selection: int | None = None,
    ) -> int:
        if path_selection is not None:
            row = (
                self._require_connection()
                .execute(
                    "SELECT COUNT(*) FROM path_selections "
                    "JOIN parts ON parts.relative_path = path_selections.relative_path "
                    "WHERE path_selections.selection_id = ? AND parts.selected = 1",
                    (path_selection,),
                )
                .fetchone()
            )
            assert row is not None
            return int(row[0])
        join, where, parameters = self._path_selection_query(
            selected=selected,
            path_selection=path_selection,
        )
        row = (
            self._require_connection()
            .execute(f"SELECT COUNT(*) FROM parts{join}{where}", parameters)
            .fetchone()
        )
        assert row is not None
        return int(row[0])

    def path_at(
        self,
        index: int,
        *,
        selected: bool,
        path_selection: int | None = None,
    ) -> Path:
        if index < 0:
            raise IndexError(index)
        join, where, parameters = self._path_selection_query(
            selected=selected,
            path_selection=path_selection,
        )
        row = (
            self._require_connection()
            .execute(
                "SELECT parts.relative_path FROM parts"
                f"{join}{where} ORDER BY partition_kind, partition_year, part_index "
                "LIMIT 1 OFFSET ?",
                (*parameters, index),
            )
            .fetchone()
        )
        if row is None:
            raise IndexError(index)
        return self.data_root / str(row[0])

    @staticmethod
    def _path_selection_query(
        *,
        selected: bool,
        path_selection: int | None,
    ) -> tuple[str, str, tuple[int, ...]]:
        if path_selection is None:
            return "", " WHERE selected = 1" if selected else "", ()
        return (
            "",
            " WHERE selected = 1 AND EXISTS ("
            "SELECT 1 FROM path_selections "
            "WHERE path_selections.selection_id = ? "
            "AND path_selections.relative_path = parts.relative_path)",
            (path_selection,),
        )

    def metadata_for(self, path: Path) -> DatasetPartMetadata:
        relative = Path(path).relative_to(self.data_root).as_posix()
        row = (
            self._require_connection()
            .execute(
                "SELECT device, inode, size, modified_ns, schema, row_count, "
                "file_size_bytes, logical_dtypes "
                "FROM parts WHERE relative_path = ?",
                (relative,),
            )
            .fetchone()
        )
        if row is None:
            raise ValueError("dataset_part_not_in_session")
        return _metadata_from_row(row)

    def schema_payload_bytes_for(self, path: Path) -> int:
        relative = Path(path).relative_to(self.data_root).as_posix()
        row = (
            self._require_connection()
            .execute(
                "SELECT length(schema) FROM parts WHERE relative_path = ?",
                (relative,),
            )
            .fetchone()
        )
        if row is None or row[0] is None:
            raise ValueError("dataset_inventory_changed")
        return int(row[0])

    def _scan_filesystem(self, spec, *, on_year, on_path) -> None:  # noqa: ANN001
        partition_columns = tuple(
            str(column) for column in spec.get("partition_columns", ())
        )
        if partition_columns not in ((), ("year",)):
            raise ValueError("unsupported_partitioning")
        require_unaliased_path(self.data_root, error="invalid_dataset_part_files")
        unpartitioned = any(
            path.suffix.casefold() == ".parquet" for path in self.data_root.iterdir()
        )
        partitioned = partition_columns == ("year",) and not unpartitioned
        if partitioned:
            for path in self.data_root.glob("year=*"):
                require_unaliased_path(path, error="invalid_dataset_part_files")
                if not path.is_dir():
                    continue
                on_year(parse_year_partition_label(path.name), path.name)

        def fail_walk(error: OSError) -> None:
            raise ValueError("invalid_dataset_part_files") from error

        for parent, dirnames, filenames in self.data_root.walk(
            follow_symlinks=False,
            on_error=fail_walk,
        ):
            dirnames.sort()
            for name in sorted(filenames):
                path = parent / name
                if path.suffix.casefold() != ".parquet":
                    continue
                coordinates = self._validated_coordinates(
                    path,
                    partitioned=partitioned,
                )
                on_path(path, coordinates)

    def _validated_coordinates(
        self,
        path: Path,
        *,
        partitioned: bool | None,
    ) -> tuple[int, int, int]:
        candidate = Path(path)
        relative = candidate.relative_to(self.data_root)
        require_unaliased_path(candidate, error="invalid_dataset_part_files")
        dataset_file_identity(candidate)
        match = _PART_FILE_PATTERN.fullmatch(candidate.name)
        if match is None:
            raise ValueError("invalid_dataset_part_files")
        if partitioned is False and len(relative.parts) != 1:
            raise ValueError("invalid_dataset_part_files")
        if partitioned is True and len(relative.parts) != 2:
            raise ValueError("invalid_dataset_part_files")
        if len(relative.parts) == 1:
            coordinates = (0, 0, int(match.group(1)))
        elif len(relative.parts) == 2:
            coordinates = (
                1,
                parse_year_partition_label(candidate.parent.name),
                int(match.group(1)),
            )
        else:
            raise ValueError("invalid_dataset_part_files")
        return coordinates

    def _register_year(self, year: int, label: str) -> None:
        row = (
            self._require_connection()
            .execute("SELECT label FROM year_labels WHERE year = ?", (year,))
            .fetchone()
        )
        if row is not None:
            if str(row[0]) != label:
                raise ValueError("invalid_dataset_part_files")
            return
        values = (year, label)
        self._gate_row_write(values)
        try:
            with self._require_connection():
                self._require_connection().execute(
                    "INSERT INTO year_labels(year, label) VALUES (?, ?)", values
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("invalid_dataset_part_files") from exc

    def _register_candidate(
        self,
        path: Path,
        coordinates: tuple[int, int, int],
    ) -> None:
        relative = path.relative_to(self.data_root).as_posix()
        identity = dataset_file_identity(path)
        values = (
            relative,
            *coordinates,
            0,
            *_identity_values(identity),
        )
        self._gate_row_write(values)
        try:
            with self._require_connection():
                self._require_connection().execute(
                    "INSERT INTO parts(relative_path, partition_kind, "
                    "partition_year, part_index, selected, device, inode, size, "
                    "modified_ns) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("invalid_dataset_part_files") from exc

    def _gate_row_write(self, values: tuple[object, ...]) -> None:
        logical_bytes = 0
        for value in values:
            if isinstance(value, bytes):
                encoded_size = len(value)
            elif isinstance(value, str):
                encoded_size = len(value.encode("utf-8"))
            elif isinstance(value, int):
                encoded_size = len(str(value))
            elif value is None:
                encoded_size = 0
            else:
                raise TypeError("invalid_dataset_inventory_row_value")
            logical_bytes += 1 + 8 + encoded_size

        self._gate_physical_write(
            logical_bytes=logical_bytes,
            row_count=1,
        )

    def _gate_physical_write(self, *, logical_bytes: int, row_count: int) -> None:
        if row_count == 0:
            return
        # Worst-case parts row: table plus three indexes, doubled for journal.
        reserve_bytes = 2 * (
            4 * logical_bytes + 8 * DATASET_INVENTORY_RESERVE_BYTES * row_count
        )
        require_free_disk_headroom(
            self._scratch_parent,
            reserve_bytes=reserve_bytes,
        )
        connection = self._require_connection()
        current_pages = int(connection.execute("PRAGMA main.page_count").fetchone()[0])
        growth_pages = (
            reserve_bytes + DATASET_INVENTORY_RESERVE_BYTES - 1
        ) // DATASET_INVENTORY_RESERVE_BYTES
        connection.execute(f"PRAGMA main.max_page_count={current_pages + growth_pages}")

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("dataset_inventory_closed")
        return self._connection


def _close_cursor_preserving_primary(cursor: sqlite3.Cursor) -> None:
    primary = sys.exception()
    if primary is None:
        cursor.close()
        return
    try:
        cursor.close()
    except BaseException as close_error:
        primary.add_note(
            "dataset inventory cursor close also failed: "
            f"{type(close_error).__name__}"
        )


def _identity_values(identity: DatasetFileIdentity) -> tuple[str, str, str, str]:
    return (
        str(identity.device),
        str(identity.inode),
        str(identity.size),
        str(identity.modified_ns),
    )


def _metadata_from_row(
    row: tuple[object, ...],
) -> DatasetPartMetadata:
    if any(value is None for value in row[4:]):
        raise RuntimeError("dataset_inventory_metadata_missing")
    return DatasetPartMetadata(
        identity=DatasetFileIdentity(
            int(row[0]), int(row[1]), int(row[2]), int(row[3])
        ),
        schema=pa.ipc.read_schema(pa.BufferReader(bytes(row[4]))),
        row_count=int(row[5]),
        file_size_bytes=int(row[6]),
        logical_dtype_payload=bytes(row[7]),
    )

"""Declared dataset part paths and deterministic partition order."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets.paths import parse_year_partition_label
from quant_research_platform.storage.identity import (
    filesystem_scandir,
    native_filesystem_path,
)

_PART_FILE_PATTERN = re.compile("^part-(\\d+)\\.parquet$")


def _directory_entries(directory: Path) -> Iterator[tuple[Path, bool]]:
    """Map vanished directories before yielding entries to business callbacks."""
    try:
        entries = filesystem_scandir(directory)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise ValueError("invalid_dataset_part_files") from error
    with closing_iterator(entries):
        while True:
            try:
                entry = next(entries)
                path = directory / entry.name
                is_directory = entry.is_dir()
            except StopIteration:
                return
            except (FileNotFoundError, NotADirectoryError) as error:
                raise ValueError("invalid_dataset_part_files") from error
            value = (path, is_directory)
            yield value


class DatasetSessionInventory:
    """Ordered paths, with no file identity or metadata snapshots."""

    def __init__(self, *, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self._parts: dict[Path, tuple[int, int, int]] = {}

    def discover(self, spec: Mapping[str, Any], *, years=None) -> None:
        self._scan_filesystem(
            spec,
            on_path=self._register_candidate,
            years=years,
        )
        self.sort_parts()

    def _register_candidate(
        self, path: Path, coordinates: tuple[int, int, int]
    ) -> None:
        self._parts[path] = coordinates

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._parts)

    def sort_parts(self) -> None:
        self._parts = dict(sorted(self._parts.items(), key=lambda item: item[1]))

    def add(self, path: Path) -> None:
        path = Path(path)
        if not Path(native_filesystem_path(path)).is_file():
            raise ValueError("invalid_dataset_part_files")
        coordinates = self._validated_coordinates(path, partitioned=None)
        self._register_candidate(path, coordinates)

    def _scan_filesystem(self, spec, *, on_path, years=None) -> None:
        partition_columns = tuple(
            str(column) for column in spec.get("partition_columns", ())
        )
        if partition_columns not in ((), ("year",)):
            raise ValueError("unsupported_partitioning")
        with closing_iterator(_directory_entries(self.data_root)) as entries:
            roots = list(entries)
        partitioned = partition_columns == ("year",) and not any(
            path.suffix.casefold() == ".parquet" for path, _ in roots
        )
        if not partitioned:
            for path, is_directory in roots:
                if not is_directory and path.suffix.casefold() == ".parquet":
                    on_path(path, self._validated_coordinates(path, partitioned=False))
            return
        for directory, is_directory in roots:
            if not is_directory or not directory.name.casefold().startswith("year="):
                continue
            year = parse_year_partition_label(directory.name)
            if years is not None and year not in years:
                continue
            with closing_iterator(_directory_entries(directory)) as entries:
                for path, is_directory in entries:
                    if not is_directory and path.suffix.casefold() == ".parquet":
                        on_path(
                            path, self._validated_coordinates(path, partitioned=True)
                        )

    def _validated_coordinates(
        self, path: Path, *, partitioned: bool | None
    ) -> tuple[int, int, int]:
        candidate = Path(path)
        relative = candidate.relative_to(self.data_root)
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

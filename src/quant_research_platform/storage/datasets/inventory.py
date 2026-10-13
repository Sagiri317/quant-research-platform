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
    path_is_filesystem_alias,
    require_unaliased_path,
)

_PART_FILE_PATTERN = re.compile("^part-(\\d+)\\.parquet$")


def _directory_entries(directory: Path) -> Iterator[tuple[Path, bool, bool]]:
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
                is_directory = entry.is_dir(follow_symlinks=False)
                is_alias = is_directory and path_is_filesystem_alias(path)
            except StopIteration:
                return
            except (FileNotFoundError, NotADirectoryError) as error:
                raise ValueError("invalid_dataset_part_files") from error
            value = (path, is_directory, is_alias)
            try:
                yield value
            except GeneratorExit:
                return


class DatasetSessionInventory:
    """Ordered paths, with no file identity or metadata snapshots."""

    def __init__(self, *, data_root: Path) -> None:
        self.data_root = Path(data_root)
        self._parts: dict[Path, tuple[int, int, int]] = {}
        self._coordinates: set[tuple[int, int, int]] = set()
        self._years: dict[int, str] = {}

    def discover(self, spec: Mapping[str, Any]) -> None:
        self._scan_filesystem(
            spec, on_year=self._register_year, on_path=self._register_candidate
        )
        self.sort_parts()

    def _register_year(self, year: int, label: str) -> None:
        if year in self._years and self._years[year] != label:
            raise ValueError("invalid_dataset_part_files")
        self._years[year] = label

    def _register_candidate(
        self, path: Path, coordinates: tuple[int, int, int]
    ) -> None:
        if path in self._parts or coordinates in self._coordinates:
            raise ValueError("invalid_dataset_part_files")
        self._parts[path] = coordinates
        self._coordinates.add(coordinates)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._parts)

    def sort_parts(self) -> None:
        self._parts = dict(sorted(self._parts.items(), key=lambda item: item[1]))

    def add(self, path: Path) -> None:
        path = Path(path)
        coordinates = self._validated_coordinates(path, partitioned=None)
        self._register_candidate(path, coordinates)
        if coordinates[0]:
            self._register_year(coordinates[1], path.parent.name)

    def _scan_filesystem(self, spec, *, on_year, on_path) -> None:
        partition_columns = tuple(
            str(column) for column in spec.get("partition_columns", ())
        )
        if partition_columns not in ((), ("year",)):
            raise ValueError("unsupported_partitioning")
        require_unaliased_path(self.data_root, error="invalid_dataset_part_files")
        with closing_iterator(_directory_entries(self.data_root)) as entries:
            unpartitioned = any(
                (path.suffix.casefold() == ".parquet" for path, _, _ in entries)
            )
        partitioned = partition_columns == ("year",) and (not unpartitioned)
        if partitioned:
            with closing_iterator(_directory_entries(self.data_root)) as entries:
                for path, is_directory, _ in entries:
                    if not path.name.casefold().startswith("year="):
                        continue
                    require_unaliased_path(path, error="invalid_dataset_part_files")
                    if is_directory:
                        on_year(parse_year_partition_label(path.name), path.name)
        pending = [_directory_entries(self.data_root)]
        try:
            while pending:
                entries = pending[-1]
                try:
                    path, is_directory, is_alias = next(entries)
                except StopIteration:
                    pending.pop().close()
                    continue
                if is_directory:
                    if not is_alias:
                        pending.append(_directory_entries(path))
                elif path.suffix.casefold() == ".parquet":
                    coordinates = self._validated_coordinates(
                        path, partitioned=partitioned
                    )
                    on_path(path, coordinates)
        except BaseException as primary:
            while pending:
                try:
                    pending.pop().close()
                except BaseException as close_error:
                    primary.add_note(
                        "dataset inventory close also failed: "
                        f"{type(close_error).__name__}"
                    )
            raise

    def _validated_coordinates(
        self, path: Path, *, partitioned: bool | None
    ) -> tuple[int, int, int]:
        candidate = Path(path)
        relative = candidate.relative_to(self.data_root)
        require_unaliased_path(candidate, error="invalid_dataset_part_files")
        if not candidate.is_file():
            raise ValueError("invalid_dataset_part_files")
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

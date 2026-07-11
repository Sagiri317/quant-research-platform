"""Dataset artifact path, partition, and timestamp helpers."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.policy.share_safety import is_local_only_path

_PART_FILE_PATTERN = re.compile(r"^part-(\d+)\.parquet$")


def partition_timestamp_series(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.Series:
    if timestamp_column not in frame:
        raise ValueError("missing_partition_timestamp_column")
    timestamp = frame[timestamp_column]
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        timestamp = pd.to_datetime(timestamp, errors="raise")
    if timestamp.isna().any():
        raise ValueError("missing_partition_timestamp_column")
    return timestamp


def partition_frames(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    partition_columns: tuple[str, ...],
) -> Iterator[tuple[str, pd.DataFrame]]:
    if frame.empty:
        return
    if partition_columns == ():
        yield "", frame
        return
    if partition_columns != ("year",):
        raise ValueError("unsupported_partitioning")
    if timestamp_column is None:
        raise ValueError("missing_partition_timestamp_column")
    timestamp = partition_timestamp_series(frame, timestamp_column=timestamp_column)
    min_timestamp = timestamp.min()
    max_timestamp = timestamp.max()
    timezone = timestamp.dt.tz
    for year in range(min_timestamp.year, max_timestamp.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz=timezone)
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone)
        mask = (timestamp >= start) & (timestamp < end)
        if bool(mask.any()):
            yield f"year={int(year)}", frame.loc[mask]


def partition_part_path(root: Path, partition: str, part_index: int) -> Path:
    if partition:
        return root / partition / f"part-{part_index:03d}.parquet"
    return root / f"part-{part_index:03d}.parquet"


def validate_dataset_artifact_root(root: Path, spec: DataSpec) -> None:
    if not is_local_only_path(root):
        raise ValueError("outside_local_only_boundary")
    path_parts = _path_parts(root)
    if isinstance(spec, InterimDataSpec):
        if (
            _unique_matching_suffix(
                path_parts,
                lambda candidate: _is_interim_output_suffix(candidate, spec),
            )
            is None
        ):
            raise ValueError("invalid_interim_dataset_root")
        return
    if isinstance(spec, ProcessedDataSpec):
        if (
            _unique_matching_suffix(
                path_parts,
                lambda candidate: _is_processed_output_suffix(candidate, spec),
            )
            is None
        ):
            raise ValueError("invalid_processed_dataset_root")
        return
    if isinstance(spec, ResearchDatasetSpec):
        if (
            _unique_matching_suffix(
                path_parts,
                lambda candidate: _is_research_dataset_output_suffix(candidate, spec),
            )
            is None
        ):
            raise ValueError("invalid_research_dataset_root")
        return
    raise ValueError("unsupported_data_spec_type")


def require_artifact_root(root: Path) -> Path:
    candidate = Path(root)
    if candidate.name == "data":
        raise ValueError("dataset_root_must_be_artifact_root")
    return candidate


def _path_parts(path: Path) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(path.as_posix()).parts if part != "/")


def _unique_matching_suffix(
    parts: tuple[str, ...],
    predicate: Any,
) -> tuple[str, ...] | None:
    candidates = [parts[index:] for index, part in enumerate(parts) if part == "data"]
    matches = [candidate for candidate in candidates if predicate(candidate)]
    storage_like_invalid = [
        candidate
        for candidate in candidates
        if _starts_storage_layout(candidate) and not predicate(candidate)
    ]
    if len(matches) != 1 or storage_like_invalid:
        return None
    return matches[0]


def _starts_storage_layout(parts: tuple[str, ...]) -> bool:
    return (
        len(parts) >= 2
        and parts[0] == "data"
        and parts[1]
        in {
            "interim",
            "processed",
            "research",
        }
    )


def _is_interim_output_suffix(parts: tuple[str, ...], spec: InterimDataSpec) -> bool:
    return (
        len(parts) == 4
        and parts[:3] == ("data", "interim", str(spec.source_id))
        and _matches_artifact_segment(parts[3], spec.data_id)
    )


def _is_processed_output_suffix(
    parts: tuple[str, ...],
    spec: ProcessedDataSpec,
) -> bool:
    expected_prefix = (
        "data",
        "processed",
        str(spec.market_id),
        spec.kind.value,
    )
    return (
        len(parts) == len(expected_prefix) + 1
        and parts[: len(expected_prefix)] == expected_prefix
        and _matches_artifact_segment(
            parts[len(expected_prefix)],
            spec.data_id,
        )
    )


def _is_research_dataset_output_suffix(
    parts: tuple[str, ...],
    spec: ResearchDatasetSpec,
) -> bool:
    return (
        len(parts) == 4
        and parts[:3] == ("data", "research", str(spec.market_id))
        and _matches_artifact_segment(parts[3], spec.data_id)
    )


def _matches_artifact_segment(segment: str, expected: str) -> bool:
    return segment == expected


def candidate_data_files(
    data_root: Path,
    *,
    spec: Mapping[str, Any],
) -> list[Path]:
    root = Path(data_root)
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    if partition_columns == ():
        paths = _validated_part_paths(root)
        _require_only_expected_parquet_paths(root, paths)
        return paths
    if partition_columns != ("year",):
        raise ValueError("unsupported_partitioning")
    unpartitioned = _validated_part_paths(root)
    if unpartitioned:
        if any(
            int(pq.ParquetFile(path).metadata.num_rows) != 0
            for path in unpartitioned
        ):
            raise ValueError("invalid_dataset_part_files")
        _require_only_expected_parquet_paths(root, unpartitioned)
        return unpartitioned
    year_directories: list[tuple[int, Path]] = []
    observed_years: set[int] = set()
    for path in root.glob("year=*"):
        if not path.is_dir():
            continue
        if _path_or_relative_component_is_alias(root, path):
            raise ValueError("invalid_dataset_part_files")
        value = path.name.removeprefix("year=")
        if not value.isdigit():
            raise ValueError("invalid_dataset_part_files")
        year = int(value)
        if year in observed_years:
            raise ValueError("invalid_dataset_part_files")
        observed_years.add(year)
        year_directories.append((year, path))
    paths: list[Path] = []
    for _, directory in sorted(year_directories):
        paths.extend(_validated_part_paths(directory))
    _require_only_expected_parquet_paths(root, paths)
    return paths


def _validated_part_paths(directory: Path) -> list[Path]:
    root = Path(directory)
    if _path_is_filesystem_alias(root):
        raise ValueError("invalid_dataset_part_files")
    indexed: list[tuple[int, Path]] = []
    observed_indices: set[int] = set()
    for path in root.iterdir():
        if path.suffix.casefold() != ".parquet":
            continue
        match = _PART_FILE_PATTERN.fullmatch(path.name)
        if (
            match is None
            or not path.is_file()
            or _path_or_relative_component_is_alias(root, path)
        ):
            raise ValueError("invalid_dataset_part_files")
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("invalid_dataset_part_files")
        except OSError as exc:
            raise ValueError("invalid_dataset_part_files") from exc
        index = int(match.group(1))
        if index in observed_indices:
            raise ValueError("invalid_dataset_part_files")
        observed_indices.add(index)
        indexed.append((index, path))
    return [path for _, path in sorted(indexed, key=lambda item: item[0])]


def _require_only_expected_parquet_paths(
    data_root: Path,
    expected: list[Path],
) -> None:
    root = Path(data_root)
    resolved_root = root.resolve()
    if _path_is_filesystem_alias(root):
        raise ValueError("invalid_dataset_part_files")
    observed_paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".parquet"
    ]
    all_paths = [*expected, *observed_paths]
    try:
        if any(
            not path.resolve().is_relative_to(resolved_root)
            or _path_or_relative_component_is_alias(root, path)
            for path in all_paths
        ):
            raise ValueError("invalid_dataset_part_files")
    except OSError as exc:
        raise ValueError("invalid_dataset_part_files") from exc
    observed = {path.absolute() for path in observed_paths}
    allowed = {path.absolute() for path in expected}
    resolved_allowed = {path.resolve() for path in expected}
    physical_allowed = {_physical_file_identity(path) for path in expected}
    if (
        observed != allowed
        or len(resolved_allowed) != len(expected)
        or len(physical_allowed) != len(expected)
    ):
        raise ValueError("invalid_dataset_part_files")


def _path_or_relative_component_is_alias(root: Path, path: Path) -> bool:
    candidate = Path(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    current = Path(root)
    if _path_is_filesystem_alias(current):
        return True
    for part in relative.parts:
        current /= part
        if _path_is_filesystem_alias(current):
            return True
    return False


def _physical_file_identity(path: Path) -> tuple[int, int]:
    try:
        stat = Path(path).stat()
    except OSError as exc:
        raise ValueError("invalid_dataset_part_files") from exc
    if int(stat.st_ino) <= 0 or int(stat.st_nlink) != 1:
        raise ValueError("invalid_dataset_part_files")
    return int(stat.st_dev), int(stat.st_ino)


def _path_is_filesystem_alias(path: Path) -> bool:
    candidate = Path(path)
    return candidate.is_symlink() or (
        hasattr(candidate, "is_junction") and candidate.is_junction()
    )


def timestamp_primary_key_column_from_spec(spec: DataSpec) -> str | None:
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
    ]
    return timestamp_columns[0] if timestamp_columns else None


def timestamp_column_from_spec_payload(spec: Mapping[str, Any]) -> str | None:
    roles = spec.get("column_roles", {})
    if not isinstance(roles, Mapping):
        raise ValueError("invalid_column_roles")
    primary_key = tuple(str(column) for column in spec.get("primary_key", ()))
    timestamp_columns = [
        column
        for column in primary_key
        if str(roles.get(column)) == ColumnRole.TIMESTAMP.value
    ]
    if len(timestamp_columns) > 1:
        raise ValueError("invalid_timestamp_primary_key")
    return timestamp_columns[0] if timestamp_columns else None


def relative_partition_label(data_root: Path, path: Path) -> str:
    relative_parent = path.parent.relative_to(data_root)
    label = relative_parent.as_posix()
    return "unpartitioned" if label == "." else label

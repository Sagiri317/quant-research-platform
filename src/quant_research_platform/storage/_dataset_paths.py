"""Dataset artifact path, partition, and timestamp helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.policy.share_safety import is_local_only_path


def _partition_timestamp_series(
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


def _partition_frames(
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
    timestamp = _partition_timestamp_series(frame, timestamp_column=timestamp_column)
    min_timestamp = timestamp.min()
    max_timestamp = timestamp.max()
    timezone = timestamp.dt.tz
    for year in range(min_timestamp.year, max_timestamp.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz=timezone)
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone)
        mask = (timestamp >= start) & (timestamp < end)
        if bool(mask.any()):
            yield f"year={int(year)}", frame.loc[mask]


def _partition_part_path(root: Path, partition: str, part_index: int) -> Path:
    if partition:
        return root / partition / f"part-{part_index:03d}.parquet"
    return root / f"part-{part_index:03d}.parquet"


def _validate_dataset_artifact_root(root: Path, spec: DataSpec) -> None:
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


def _require_artifact_root(root: Path) -> Path:
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
    return segment == expected or segment.startswith(f".{expected}.tmp-")


def _candidate_data_files(
    data_root: Path,
    *,
    spec: Mapping[str, Any],
) -> list[Path]:
    root = Path(data_root)
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    unpartitioned = sorted(root.glob("part-*.parquet"))
    if unpartitioned:
        return unpartitioned
    if partition_columns == ():
        return []
    if partition_columns != ("year",):
        raise ValueError("unsupported_partitioning")
    available_years = {
        path.name.split("=", 1)[1]
        for path in root.glob("year=*")
        if path.is_dir() and list(path.glob("*.parquet"))
    }
    paths: list[Path] = []
    for year in sorted(available_years, key=lambda value: int(value)):
        paths.extend(sorted((root / f"year={year}").glob("*.parquet")))
    return paths


def _timestamp_primary_key_column_from_spec(spec: DataSpec) -> str | None:
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
    ]
    return timestamp_columns[0] if timestamp_columns else None


def _timestamp_column_from_spec_payload(spec: Mapping[str, Any]) -> str | None:
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


def _relative_partition_label(data_root: Path, path: Path) -> str:
    relative_parent = path.parent.relative_to(data_root)
    label = relative_parent.as_posix()
    return "unpartitioned" if label == "." else label

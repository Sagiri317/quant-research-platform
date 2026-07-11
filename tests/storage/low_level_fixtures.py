from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.storage._dataset_paths import relative_partition_label
from quant_research_platform.storage._dataset_schema import (
    canonical_schema_from_file,
    sha256_json,
)


def make_default_dataset_frame(
    *,
    asset_ids: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
    values: Sequence[object] | None = None,
    value_column: str = "value",
    market_id: str = "a_share",
) -> pd.DataFrame:
    return _make_dataset_frame(
        asset_ids=asset_ids,
        dates=dates,
        values=values,
        value_column=value_column,
        market_id=market_id,
        defaults=_DEFAULT_DATASET_FRAME,
    )


def make_time_window_dataset_frame(
    *,
    value_column: str = "value",
    market_id: str = "a_share",
) -> pd.DataFrame:
    return _make_dataset_frame(
        value_column=value_column,
        market_id=market_id,
        defaults=_TIME_WINDOW_DATASET_FRAME,
    )


def make_helper_edges_dataset_frame(
    *,
    value_column: str = "value",
    market_id: str = "a_share",
) -> pd.DataFrame:
    return _make_dataset_frame(
        value_column=value_column,
        market_id=market_id,
        defaults=_HELPER_EDGES_DATASET_FRAME,
    )


def make_correctness_dataset_frame(
    *,
    value_column: str = "value",
    market_id: str = "a_share",
) -> pd.DataFrame:
    return _make_dataset_frame(
        value_column=value_column,
        market_id=market_id,
        defaults=_CORRECTNESS_DATASET_FRAME,
    )


def _make_dataset_frame(
    *,
    asset_ids: Sequence[str] | None = None,
    dates: Sequence[str] | None = None,
    values: Sequence[object] | None = None,
    value_column: str = "value",
    market_id: str = "a_share",
    defaults: dict[str, Sequence[object]],
) -> pd.DataFrame:
    asset_ids = defaults["asset_ids"] if asset_ids is None else asset_ids
    dates = defaults["dates"] if dates is None else dates
    values = defaults["values"] if values is None else values
    return pd.DataFrame(
        {
            "market_id": [market_id] * len(asset_ids),
            "asset_id": list(asset_ids),
            "timestamp": pd.to_datetime(list(dates)),
            value_column: list(values),
        }
    )


_DEFAULT_DATASET_FRAME = {
    "asset_ids": ("asset_002", "asset_001", "asset_001"),
    "dates": ("2015-01-05", "2014-01-02", "2015-01-05"),
    "values": (2.0, 1.0, float("nan")),
}
_TIME_WINDOW_DATASET_FRAME = {
    "asset_ids": ("000001", "000001", "000001"),
    "dates": ("2014-01-02", "2015-01-05", "2016-01-04"),
    "values": (1.0, 2.0, 3.0),
}
_CORRECTNESS_DATASET_FRAME = {
    "asset_ids": ("asset_a", "asset_b", "asset_a", "asset_b"),
    "dates": ("2020-01-02", "2020-01-02", "2020-01-06", "2020-01-06"),
    "values": (1.0, 2.0, 3.0, 4.0),
}
_HELPER_EDGES_DATASET_FRAME = {
    "asset_ids": ("asset_a", "asset_b"),
    "dates": ("2024-01-02", "2024-01-03"),
    "values": (1.0, 2.0),
}


def make_processed_dataset_spec(
    data_id: str = "x",
    *,
    kind: ProcessedDataKind = ProcessedDataKind.FACTOR,
    value_column: str = "value",
    value_role: ColumnRole | None = None,
    market_id: str = "a_share",
    primary_key: tuple[str, ...] = platform_entity_time_key(),
    partition_columns: tuple[str, ...] = ("year",),
) -> ProcessedDataSpec:
    role = value_role or (
        ColumnRole.LABEL if kind == ProcessedDataKind.LABEL else ColumnRole.FACTOR
    )
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=MarketId(market_id),
        kind=kind,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            value_column: role,
        },
        primary_key=primary_key,
        partition_columns=partition_columns,
    )


def make_dataset_artifact_root(
    tmp_path: Path,
    data_id: str = "x",
    *,
    kind: ProcessedDataKind = ProcessedDataKind.FACTOR,
    market_id: str = "a_share",
) -> Path:
    return tmp_path / "data" / "processed" / market_id / kind.value / data_id


def write_low_level_parquet_artifact(
    artifact_root: Path,
    frame: pd.DataFrame,
    *,
    primary_key: Sequence[str],
    partition_columns: Sequence[str],
) -> None:
    """Construct low-level parquet fixtures for storage-adjacent tests."""

    spec = {
        "schema_id": "qrp-dataset-spec",
        "data_id": Path(artifact_root).name,
        "primary_key": list(primary_key),
        "partition_columns": list(partition_columns),
        "required_columns": [str(column) for column in frame.columns],
        "column_roles": _column_roles(frame.columns, primary_key),
    }
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    data_root = root / "data"
    paths: list[Path] = []
    if "timestamp" in frame and tuple(partition_columns) == ("year",):
        result = frame.copy()
        result["_fixture_year"] = pd.to_datetime(result["timestamp"]).dt.year
        for year, year_frame in result.groupby("_fixture_year"):
            year_root = data_root / f"year={int(year)}"
            year_root.mkdir(parents=True, exist_ok=True)
            path = year_root / "part-000.parquet"
            year_frame.drop(columns=["_fixture_year"]).to_parquet(path, index=False)
            paths.append(path)
    else:
        data_root.mkdir(parents=True, exist_ok=True)
        path = data_root / "part-000.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    _write_runtime_manifest(root, data_root, paths)


def _write_runtime_manifest(root: Path, data_root: Path, paths: list[Path]) -> None:
    schemas: list[tuple[tuple[str, str], ...]] = []
    row_count = 0
    partition_row_counts: dict[str, int] = {}
    for path in sorted(paths):
        parquet_file = pq.ParquetFile(path)
        schemas.append(canonical_schema_from_file(parquet_file))
        partition = relative_partition_label(data_root, path)
        rows = int(parquet_file.metadata.num_rows)
        row_count += rows
        partition_row_counts[partition] = partition_row_counts.get(partition, 0) + rows
    first_schema = schemas[0]
    if any(schema != first_schema for schema in schemas[1:]):
        raise ValueError("dataset_schema_mismatch")
    manifest = {
        "schema_id": "qrp-dataset-manifest",
        "columns": [name for name, _ in first_schema],
        "schema_hash": sha256_json(first_schema),
        "partition_row_counts": dict(sorted(partition_row_counts.items())),
        "file_count": len(paths),
        "row_count": row_count,
    }
    manifest_path = root / "_manifest" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _column_roles(
    columns: Sequence[object],
    primary_key: Sequence[str],
) -> dict[str, str]:
    primary_key_set = {str(column) for column in primary_key}
    roles = {}
    for column in columns:
        name = str(column)
        if name == "timestamp":
            roles[name] = "timestamp"
        elif name in primary_key_set:
            roles[name] = "key"
        else:
            roles[name] = "feature"
    return roles

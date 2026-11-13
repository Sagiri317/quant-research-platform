"""Numerical tables for an independently computed frozen-model explanation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.manifests import (
    read_qrp_json_file,
    write_qrp_json_file,
)
from quant_research_platform.storage.parquet import (
    iter_dataset_part_batches,
    parquet_table_writer,
)

_S = pa.string()
_F = pa.float64()
_I = pa.int64()
_T = pa.timestamp("ns")
_SCHEMAS = {
    "native_importance": pa.schema(
        [
            ("feature", _S),
            ("value", _F),
            ("signed_value", _F),
            ("kind", _S),
            ("unit", _S),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "predictions": pa.schema(
        [
            ("timestamp", _T),
            ("asset_id", _S),
            ("base_value", _F),
            ("explained_output", _F),
            ("prediction_score", _F),
            ("additivity_residual", _F),
            ("additivity_tolerance", _F),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "contributions": pa.schema(
        [
            ("timestamp", _T),
            ("asset_id", _S),
            ("feature", _S),
            ("input_value", _S),
            ("contribution", _F),
        ]
    ),
    "summary": pa.schema(
        [
            ("feature", _S),
            ("mean_signed_contribution", _F),
            ("mean_abs_contribution", _F),
            ("observation_count", _I),
            ("date_count", _I),
            ("sample_count", _I),
            ("sample_date_count", _I),
            ("absolute_share", _F),
            ("rank", _I),
            ("period", _S),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "permutation_repeats": pa.schema(
        [
            ("group", _S),
            ("repeat", _I),
            ("metric", _S),
            ("baseline", _F),
            ("perturbed", _F),
            ("delta", _F),
            ("observation_count", _I),
            ("period_count", _I),
            ("unit", _S),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "permutation_summary": pa.schema(
        [
            ("group", _S),
            ("metric", _S),
            ("baseline", _F),
            ("observation_count", _I),
            ("period_count", _I),
            ("unit", _S),
            ("status", _S),
            ("reason", _S),
            ("delta_mean", _F),
            ("delta_std", _F),
            ("repeats", _I),
        ]
    ),
}


def write_explanation(path: Path, *, metadata: dict, tables: dict) -> Path:
    """Publish only selected tables, with fail-if-exists atomic storage."""
    path = Path(path)

    def write(staging):
        for name, frame in tables.items():
            with open_filesystem_file(
                staging / (name + ".parquet"), mode="xb"
            ) as handle:
                with parquet_table_writer(
                    handle, compression="zstd", schema=_SCHEMAS[name]
                ) as writer:
                    writer(
                        pa.Table.from_pandas(
                            frame, schema=_SCHEMAS[name], preserve_index=False
                        )
                    )
        write_qrp_json_file(
            staging / "metadata.json", {**metadata, "members": list(tables)}
        )

    atomic_write_directory(path, write)
    return path


def read_explanation(
    path: Path,
    *,
    members: list[str] | tuple[str, ...] | None = None,
    observation: tuple[pd.Timestamp, str] | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Decode selected native tables, optionally selecting one attribution key."""
    path = Path(path)
    metadata = read_qrp_json_file(path / "metadata.json")[1]
    declared = metadata["members"]
    selected = declared if members is None else members
    if set(selected) - set(declared):
        raise ValueError("explanation_members_invalid")
    tables = {}
    for name in selected:
        filters = None
        if observation is not None and name in {"predictions", "contributions"}:
            timestamp, asset = observation
            filters = [("timestamp", "==", timestamp), ("asset_id", "==", asset)]
        tables[name] = pa.Table.from_batches(
            list(
                iter_dataset_part_batches(
                    path / (name + ".parquet"), parquet_filters=filters
                )
            ),
            schema=_SCHEMAS[name],
        ).to_pandas()
    return metadata, tables

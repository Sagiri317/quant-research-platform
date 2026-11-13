"""Numerical tables for an independently computed frozen-model explanation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.manifests import read_qrp_json_file
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
_KEYS = {
    "native_importance": ["feature"],
    "predictions": ["timestamp", "asset_id"],
    "contributions": ["timestamp", "asset_id", "feature"],
    "summary": ["period", "feature"],
    "permutation_repeats": ["group", "repeat", "metric"],
    "permutation_summary": ["group", "metric"],
}


def _validate(metadata, tables):
    if not isinstance(metadata, dict) or metadata.get("kind") != "MODEL_EXPLANATION":
        raise ValueError("explanation_metadata_invalid")
    for key in ("model", "model_id", "model_algorithm", "target_column", "label_unit"):
        if type(metadata.get(key)) is not str or not metadata[key].strip():
            raise ValueError("explanation_metadata_invalid")
    columns = metadata.get("input_columns")
    methods = metadata.get("methods")
    if (
        type(columns) is not list
        or not columns
        or any(type(value) is not str or not value for value in columns)
        or len(set(columns)) != len(columns)
        or not isinstance(metadata.get("input_units"), dict)
        or type(methods) is not list
        or not methods
        or any(type(value) is not str for value in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("explanation_metadata_invalid")
    attribution = set(methods) - {"native_importance", "permutation"}
    if attribution or "permutation" in methods:
        if any(
            type(metadata.get(key)) is not str or not metadata[key].strip()
            for key in ("output_space", "unit", "sample_role")
        ):
            raise ValueError("explanation_output_metadata_invalid")
    expected = set()
    if "native_importance" in methods:
        expected.add("native_importance")
    if "permutation" in methods:
        expected.update(("permutation_repeats", "permutation_summary"))
    if attribution:
        expected.update(("predictions", "contributions", "summary"))
    if not expected or set(tables) != expected:
        raise ValueError("explanation_members_invalid")
    result = {}
    arrow_tables = {}
    for name, frame in tables.items():
        schema = _SCHEMAS[name]
        if list(frame.columns) != schema.names or frame.empty:
            raise ValueError("explanation_table_schema_invalid")
        try:
            table = pa.Table.from_pandas(
                frame, schema=schema, preserve_index=False, safe=True
            )
        except (pa.ArrowException, ValueError, TypeError):
            raise ValueError("explanation_table_schema_invalid") from None
        arrow_tables[name] = table
        if (
            frame[_KEYS[name]].isna().any(axis=None)
            or frame.duplicated(_KEYS[name]).any()
        ):
            raise ValueError("explanation_table_keys_invalid")
        for field in schema:
            if name == "contributions" and field.name == "input_value":
                values = pd.to_numeric(frame.input_value.dropna(), errors="coerce")
                if not np.isfinite(values).all():
                    raise ValueError("explanation_input_value_invalid")
                continue
            if pa.types.is_string(field.type) and (
                frame[field.name].isna().any()
                or field.name != "reason"
                and frame[field.name].str.strip().eq("").any()
            ):
                raise ValueError("explanation_table_string_invalid")
            if pa.types.is_floating(field.type):
                if not np.isfinite(
                    frame[field.name].dropna().to_numpy(dtype=float)
                ).all():
                    raise ValueError("explanation_table_nonfinite")
            if pa.types.is_integer(field.type) and (
                (field.name != "rank" and frame[field.name].isna().any())
                or frame[field.name].lt(0).any()
            ):
                raise ValueError("explanation_table_count_invalid")
        if (
            "feature" in frame
            and not frame.feature.isin(metadata["input_columns"]).all()
        ):
            raise ValueError("explanation_feature_binding_invalid")
        if "status" in frame:
            defined = frame.status.eq("DEFINED")
            measurements = {
                "native_importance": ["value"],
                "predictions": [
                    "base_value",
                    "additivity_residual",
                    "additivity_tolerance",
                ],
                "summary": [
                    "mean_abs_contribution",
                    "mean_signed_contribution",
                    "rank",
                ],
                "permutation_repeats": ["baseline", "perturbed", "delta"],
                "permutation_summary": ["baseline", "delta_mean", "delta_std"],
            }[name]
            if (
                not frame.status.isin(
                    ("DEFINED", "NOT_APPLICABLE")
                    if name == "native_importance"
                    else ("DEFINED", "UNDEFINED")
                ).all()
                or frame.loc[defined, measurements].isna().any(axis=None)
                or frame.loc[~defined, measurements].notna().any(axis=None)
                or frame.loc[defined, "reason"].ne("").any()
                or frame.loc[~defined, "reason"].str.strip().eq("").any()
            ):
                raise ValueError("explanation_table_status_invalid")
        if name in {"predictions", "contributions", "summary"}:
            nullable = {
                "predictions": {
                    "base_value",
                    "additivity_residual",
                    "additivity_tolerance",
                },
                "contributions": {"input_value", "contribution"},
                "summary": {
                    "absolute_share",
                    "mean_abs_contribution",
                    "mean_signed_contribution",
                    "rank",
                },
            }.get(name, set())
            if frame.drop(columns=list(nullable)).isna().any(axis=None):
                raise ValueError("explanation_table_null_invalid")
        result[name] = frame
    if "predictions" in result:
        predictions = result["predictions"].set_index(["timestamp", "asset_id"])
        if (
            metadata["output_space"] == "model_output"
            and not predictions["explained_output"]
            .eq(predictions["prediction_score"])
            .all()
        ):
            raise ValueError("explanation_prediction_binding_invalid")
        contributions = result["contributions"]
        counts = contributions.groupby(["timestamp", "asset_id"]).size()
        if (
            not counts.index.equals(predictions.sort_index().index)
            or not counts.eq(len(metadata["input_columns"])).all()
        ):
            raise ValueError("explanation_row_binding_invalid")
        defined = predictions.status.eq("DEFINED")
        row_defined = defined.reindex(
            pd.MultiIndex.from_frame(contributions[["timestamp", "asset_id"]])
        ).to_numpy()
        if contributions.contribution.notna().ne(row_defined).any():
            raise ValueError("explanation_row_status_invalid")
        # The producer computes additivity and its status once. Publication
        # checks the saved result's shape and bounds, not a second attribution.
        if (
            predictions.loc[defined, "additivity_residual"]
            .abs()
            .gt(predictions.loc[defined, "additivity_tolerance"])
            .any()
        ):
            raise ValueError("explanation_additivity_failed")
    if "summary" in result:
        summary = result["summary"]
        if (
            summary.observation_count.gt(summary.sample_count).any()
            or summary.date_count.gt(summary.sample_date_count).any()
            or summary.date_count.gt(summary.observation_count).any()
            or summary.status.eq("DEFINED").ne(summary.observation_count.gt(0)).any()
            or summary.mean_abs_contribution.lt(0).any()
            or not summary.absolute_share.dropna().between(0, 1).all()
        ):
            raise ValueError("explanation_summary_coverage_invalid")
        if "predictions" in result and (
            not summary.loc[summary.period.eq("all"), "observation_count"]
            .eq(int(defined.sum()))
            .all()
            or not summary.loc[summary.period.eq("all"), "sample_count"]
            .eq(len(predictions))
            .all()
        ):
            raise ValueError("explanation_summary_coverage_invalid")
    return arrow_tables


def write_explanation(path: Path, *, metadata: dict, tables: dict) -> Path:
    """Publish only selected tables, with fail-if-exists atomic storage."""
    tables = _validate(metadata, tables)
    path = Path(path)

    def write(staging):
        for name, table in tables.items():
            with open_filesystem_file(
                staging / (name + ".parquet"), mode="xb"
            ) as handle:
                with parquet_table_writer(
                    handle, compression="zstd", schema=_SCHEMAS[name]
                ) as writer:
                    writer(table)
        atomic_write_json(
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
    declared = metadata.get("members")
    if (
        type(declared) is not list
        or any(type(name) is not str for name in declared)
        or len(declared) != len(set(declared))
        or set(declared) - set(_SCHEMAS)
    ):
        raise ValueError("explanation_members_invalid")
    selected = declared if members is None else members
    if (
        not isinstance(selected, (list, tuple))
        or not selected
        or any(type(name) is not str for name in selected)
        or len(selected) != len(set(selected))
        or set(selected) - set(declared)
    ):
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

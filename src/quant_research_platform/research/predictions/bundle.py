"""Prediction bundle helpers for private report outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import (
    reject_filter_feature_columns,
    required_filter_columns,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports.report_artifacts import (
    write_prediction_report_artifacts,
)
from quant_research_platform.research._memory import iter_time_sized_frames
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.runtime.resources import ChunkAxis
from quant_research_platform.storage.layout import StorageLayout


def build_prediction_bundle(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    return _build_prediction_bundle_from_frames(
        (frame,),
        spec,
        coefficients,
        feature_columns,
    )


def _build_prediction_bundle_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    reject_filter_feature_columns(feature_columns)
    required = set((*spec.primary_key, *feature_columns))
    _validate_oos_split(spec)
    work_columns = tuple(
        dict.fromkeys(
            (
                *spec.primary_key,
                *feature_columns,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    result, filter_metadata, rows_after_dropna = _build_prediction_frame(
        _iter_prediction_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ),
        spec=spec,
        coefficients=coefficients,
        feature_columns=feature_columns,
    )
    if result.empty:
        raise ValueError("empty_prediction_window")
    effective_test_end = _date_string(pd.to_datetime(result["timestamp"]).max())
    result.attrs["prediction_metadata"] = {
        "contract_version": spec.contract_version,
        "prediction_mode": spec.prediction_mode,
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": list(filter_metadata["filter_columns_used"]),
        "filter_definition": str(filter_metadata["filter_definition"]),
        "rows_input": int(filter_metadata["rows_input"]),
        "rows_after_filter_mode": int(filter_metadata["rows_after_filter_mode"]),
        "rows_after_feature_dropna": int(rows_after_dropna),
        "prediction_rows": int(len(result)),
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "requested_test_end": spec.test_end,
        "effective_test_end": effective_test_end,
        "is_oos": spec.is_oos,
        "is_final_oos_window": spec.is_final_oos_window,
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "prediction_column": spec.prediction_column,
    }
    return result


def _build_prediction_frame(
    frames: Iterable[pd.DataFrame],
    *,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, object], int]:
    parts: list[pd.DataFrame] = []
    rows_input = 0
    rows_after_filter = 0
    rows_after_dropna = 0
    filter_columns_used: list[str] = []
    filter_definition = ""
    usable_columns = tuple(dict.fromkeys((*spec.primary_key, *feature_columns)))
    for frame in frames:
        for chunk in iter_time_sized_frames(
            frame,
            timestamp_column="timestamp",
            axis=ChunkAxis.TIME,
        ):
            filtered, metadata = apply_filter_mode(
                chunk,
                mode=spec.sample_filter_mode,
                metadata_key="sample_filter_mode",
            )
            rows_input += int(metadata["rows_input"])
            rows_after_filter += int(metadata["rows_after_filter_mode"])
            filter_columns_used = list(metadata["filter_columns_used"])
            filter_definition = str(metadata["filter_definition"])
            usable = filtered.dropna(subset=list(feature_columns)).loc[
                :, list(usable_columns)
            ]
            rows_after_dropna += int(len(usable))
            test_mask = usable["timestamp"] >= pd.Timestamp(spec.test_start)
            if spec.test_end is not None:
                test_mask &= usable["timestamp"] <= pd.Timestamp(spec.test_end)
            test_frame = usable.loc[test_mask]
            if test_frame.empty:
                continue
            x = test_frame.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
            output = test_frame.loc[:, spec.primary_key].copy()
            output[spec.prediction_column] = (
                float(coefficients[0]) + x @ coefficients[1:]
            )
            parts.append(output)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    filter_metadata = {
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    return result, filter_metadata, rows_after_dropna


def _validate_oos_split(spec: PredictionSpec) -> None:
    if pd.Timestamp(spec.train_end) >= pd.Timestamp(spec.test_start):
        raise ValueError("prediction_train_end_must_precede_test_start")
    if spec.test_end is not None and pd.Timestamp(spec.test_end) < pd.Timestamp(
        spec.test_start
    ):
        raise ValueError("prediction_test_end_must_not_precede_test_start")


def _date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def _iter_prediction_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    required: set[str],
    work_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    for frame in frames:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing_prediction_columns: {', '.join(missing)}")
        selected = [column for column in work_columns if column in frame.columns]
        yield frame.loc[:, selected]


def write_prediction_bundle(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    *,
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
    return write_prediction_report_artifacts(
        frame,
        spec=spec,
        report_spec=report_spec,
        layout=layout,
    )


__all__ = ["build_prediction_bundle", "write_prediction_bundle"]

"""Prediction bundle helpers for private report outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import reject_filter_feature_columns
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports.report_artifacts import (
    write_prediction_report_artifacts,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.storage.layout import StorageLayout


def build_prediction_bundle(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    reject_filter_feature_columns(feature_columns)
    required = set((*spec.primary_key, *feature_columns))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_columns: {', '.join(missing)}")
    _validate_oos_split(spec)
    filtered, metadata = apply_filter_mode(
        frame,
        mode=spec.sample_filter_mode,
        metadata_key="sample_filter_mode",
    )
    usable = filtered.dropna(subset=list(feature_columns)).copy()
    metadata["rows_after_feature_dropna"] = int(len(usable))
    test_mask = usable["timestamp"] >= pd.Timestamp(spec.test_start)
    if spec.test_end is not None:
        test_mask &= usable["timestamp"] <= pd.Timestamp(spec.test_end)
    test_frame = usable.loc[test_mask].copy()
    if test_frame.empty:
        raise ValueError("prediction_oos_test_window_empty")
    design = np.column_stack(
        [
            np.ones(len(test_frame)),
            test_frame.loc[:, feature_columns].to_numpy(dtype=float),
        ]
    )
    result = test_frame.loc[:, spec.primary_key].copy()
    result[spec.prediction_column] = design @ coefficients
    result.attrs["metadata"] = metadata
    return result


def _validate_oos_split(spec: PredictionSpec) -> None:
    if pd.Timestamp(spec.train_end) >= pd.Timestamp(spec.test_start):
        raise ValueError("prediction_train_end_must_precede_test_start")
    if spec.test_end is not None and pd.Timestamp(spec.test_end) < pd.Timestamp(
        spec.test_start
    ):
        raise ValueError("prediction_test_end_must_not_precede_test_start")


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

"""Prediction bundle helpers for private report outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports.bundles import write_prediction_bundle_files
from quant_research_platform.storage.layout import StorageLayout


def build_prediction_bundle(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = set((*spec.primary_key, *feature_columns))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_columns: {', '.join(missing)}")
    _validate_oos_split(spec)
    test_frame = frame.loc[frame["timestamp"] >= pd.Timestamp(spec.test_start)].copy()
    design = np.column_stack(
        [
            np.ones(len(test_frame)),
            test_frame.loc[:, feature_columns].to_numpy(dtype=float),
        ]
    )
    result = test_frame.loc[:, spec.primary_key].copy()
    result[spec.prediction_column] = design @ coefficients
    return result


def _validate_oos_split(spec: PredictionSpec) -> None:
    if pd.Timestamp(spec.train_end) >= pd.Timestamp(spec.test_start):
        raise ValueError("prediction_train_end_must_precede_test_start")


def write_prediction_bundle(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    *,
    report_spec: ReportSpec,
    layout: StorageLayout,
    output_root: Path | None = None,
) -> Path:
    return write_prediction_bundle_files(
        frame,
        spec=spec,
        report_spec=report_spec,
        layout=layout,
        output_root=output_root,
    )


__all__ = ["build_prediction_bundle", "write_prediction_bundle"]

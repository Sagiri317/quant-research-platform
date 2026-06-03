"""Prediction bundle helpers for private report outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.reports.bundles import write_prediction_bundle_files


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
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame.loc[:, feature_columns].to_numpy(dtype=float),
        ]
    )
    result = frame.loc[:, spec.primary_key].copy()
    result[spec.prediction_column] = design @ coefficients
    return result


def write_prediction_bundle(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    output_root: Path,
) -> Path:
    return write_prediction_bundle_files(
        frame,
        spec=spec,
        output_root=output_root,
    )


__all__ = ["build_prediction_bundle", "write_prediction_bundle"]

"""Minimal deterministic non-DL baseline model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import ModelSpec


def fit_linear_baseline(frame: pd.DataFrame, spec: ModelSpec) -> np.ndarray:
    required = set(spec.feature_columns)
    required.add(spec.label_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_model_columns: {', '.join(missing)}")
    x = frame.loc[:, spec.feature_columns].to_numpy(dtype=float)
    y = frame[spec.label_column].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


__all__ = ["fit_linear_baseline"]

"""Minimal deterministic non-DL baseline model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.filtering import apply_filter_mode


@dataclass(frozen=True)
class ModelFitResult:
    coefficients: np.ndarray
    metadata: dict[str, object]


def fit_linear_baseline_result(frame: pd.DataFrame, spec: ModelSpec) -> ModelFitResult:
    required = set(spec.feature_columns)
    required.add(spec.label_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_model_columns: {', '.join(missing)}")
    filtered, metadata = apply_filter_mode(
        frame,
        mode=spec.sample_filter_mode,
        metadata_key="sample_filter_mode",
    )
    usable = filtered.dropna(subset=[*spec.feature_columns, spec.label_column]).copy()
    metadata["rows_after_feature_label_dropna"] = int(len(usable))
    if usable.empty:
        raise ValueError("model_training_sample_empty")
    x = usable.loc[:, spec.feature_columns].to_numpy(dtype=float)
    y = usable[spec.label_column].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return ModelFitResult(coefficients=beta, metadata=metadata)


__all__ = ["ModelFitResult", "fit_linear_baseline_result"]

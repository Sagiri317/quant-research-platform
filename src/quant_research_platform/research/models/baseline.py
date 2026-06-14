"""Minimal deterministic non-DL baseline model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research._memory import iter_time_sized_frames
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.runtime.resources import ChunkAxis


@dataclass(frozen=True)
class ModelFitResult:
    coefficients: np.ndarray
    metadata: dict[str, object]


def fit_linear_baseline_result(frame: pd.DataFrame, spec: ModelSpec) -> ModelFitResult:
    required = set(spec.feature_columns)
    required.add(spec.label_column)
    required.add("timestamp")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_model_columns: {', '.join(missing)}")
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                *spec.feature_columns,
                spec.label_column,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    work_columns = tuple(column for column in work_columns if column in frame.columns)
    fit_columns = tuple(
        column
        for column in ("timestamp", *spec.feature_columns, spec.label_column)
        if column in work_columns
    )
    beta, row_counts, filter_metadata = _fit_linear_coefficients(
        frame.loc[:, list(work_columns)],
        spec=spec,
        fit_columns=fit_columns,
    )
    if row_counts["rows_after_feature_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    metadata = {
        "contract_version": spec.contract_version,
        "model_family": spec.model_family,
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": list(filter_metadata["filter_columns_used"]),
        "filter_definition": str(filter_metadata["filter_definition"]),
        "rows_input": int(filter_metadata["rows_input"]),
        "rows_after_filter_mode": int(filter_metadata["rows_after_filter_mode"]),
        "rows_after_feature_label_dropna": int(
            row_counts["rows_after_feature_label_dropna"]
        ),
        "feature_columns": list(spec.feature_columns),
        "feature_count": len(spec.feature_columns),
        "label_column": spec.label_column,
        "coefficient_count": int(len(beta)),
    }
    return ModelFitResult(coefficients=beta, metadata=metadata)


def _fit_linear_coefficients(
    frame: pd.DataFrame,
    *,
    spec: ModelSpec,
    fit_columns: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, int], dict[str, object]]:
    feature_count = len(spec.feature_columns)
    chunk_count = 0
    row_count = 0
    rows_input = 0
    rows_after_filter = 0
    first_usable: pd.DataFrame | None = None
    filter_columns_used: list[str] = []
    filter_definition = ""
    r_factor: np.ndarray | None = None
    qt_y: np.ndarray | None = None

    for chunk in iter_time_sized_frames(
        frame,
        timestamp_column="timestamp",
        axis=ChunkAxis.TIME,
    ):
        chunk_count += 1
        filtered, metadata = apply_filter_mode(
            chunk,
            mode=spec.sample_filter_mode,
            metadata_key="sample_filter_mode",
        )
        rows_input += int(metadata["rows_input"])
        rows_after_filter += int(metadata["rows_after_filter_mode"])
        filter_columns_used = list(metadata["filter_columns_used"])
        filter_definition = str(metadata["filter_definition"])
        usable = filtered.dropna(
            subset=[*spec.feature_columns, spec.label_column]
        ).loc[:, list(fit_columns)]
        if first_usable is None:
            first_usable = usable
        elif not first_usable.empty:
            first_usable = pd.DataFrame()
        y = usable.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
        row_count += int(len(y))
        if len(y) == 0:
            continue
        chunk_design = _linear_design_matrix(usable, spec)
        if r_factor is None or qt_y is None:
            stacked_design = chunk_design
            stacked_y = y
        else:
            stacked_design = np.vstack([r_factor, chunk_design])
            stacked_y = np.concatenate([qt_y, y])
        q_factor, r_factor = np.linalg.qr(stacked_design, mode="reduced")
        qt_y = q_factor.T @ stacked_y

    row_counts = {
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
        "rows_after_feature_label_dropna": row_count,
    }
    filter_metadata = {
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    if row_count == 0:
        return np.empty(feature_count + 1, dtype="float64"), row_counts, filter_metadata
    if chunk_count == 1 and first_usable is not None:
        return (
            _fit_linear_coefficients_exact(first_usable, spec),
            row_counts,
            filter_metadata,
        )
    if r_factor is None or qt_y is None:
        return np.empty(feature_count + 1, dtype="float64"), row_counts, filter_metadata
    return (
        _fit_linear_coefficients_from_tsqr(r_factor, qt_y),
        row_counts,
        filter_metadata,
    )


def _fit_linear_coefficients_exact(
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    design = _linear_design_matrix(frame, spec)
    y = frame.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def _linear_design_matrix(
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    x = frame.loc[:, spec.feature_columns].to_numpy(dtype="float64", copy=False)
    design = np.empty((len(x), len(spec.feature_columns) + 1), dtype="float64")
    design[:, 0] = 1.0
    design[:, 1:] = x
    return design


def _fit_linear_coefficients_from_tsqr(
    r_factor: np.ndarray,
    qt_y: np.ndarray,
) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(r_factor, qt_y, rcond=None)
    return beta


__all__ = ["ModelFitResult", "fit_linear_baseline_result"]

"""Prediction artifact helpers for private report outputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import (
    reject_filter_input_columns,
    required_filter_columns,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports.report_artifacts import (
    _write_prediction_report_artifacts_from_chunks,
    write_prediction_report_artifacts,
)
from quant_research_platform.research._memory import iter_complete_time_sized_frames
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.runtime.resources import ChunkAxis
from quant_research_platform.storage.layout import StorageLayout


def _write_prediction_artifacts_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: PredictionSpec,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    *,
    prediction_model: object | None = None,
    model_metadata: dict[str, object],
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
    reject_filter_input_columns(input_columns)
    required = set((*spec.primary_key, *input_columns))
    _validate_oos_split(spec)
    work_columns = tuple(
        dict.fromkeys(
            (
                *spec.primary_key,
                *input_columns,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    build_state = _PredictionBuildState()
    output_frames = _iter_prediction_output_frames(
        _iter_prediction_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ),
        spec=spec,
        coefficients=coefficients,
        input_columns=input_columns,
        predictor=_prediction_callable(coefficients, input_columns, prediction_model),
        build_state=build_state,
    )
    return _write_prediction_report_artifacts_from_chunks(
        output_frames,
        spec=spec,
        report_spec=report_spec,
        layout=layout,
        model_metadata=_prediction_static_window_model_metadata(model_metadata, spec),
        prediction_metadata_factory=lambda: build_state.prediction_metadata(
            spec=spec,
            input_columns=input_columns,
        ),
    )


@dataclass
class _PredictionBuildState:
    rows_input: int = 0
    rows_after_filter: int = 0
    rows_after_dropna: int = 0
    prediction_rows: int = 0
    filter_columns_used: tuple[str, ...] = ()
    filter_definition: str = ""
    effective_test_end: str | None = None

    def record_filter_metadata(self, metadata: dict[str, object]) -> None:
        self.rows_input += int(metadata["rows_input"])
        self.rows_after_filter += int(metadata["rows_after_filter_mode"])
        self.filter_columns_used = tuple(metadata["filter_columns_used"])
        self.filter_definition = str(metadata["filter_definition"])

    def record_input_dropna_rows(self, row_count: int) -> None:
        self.rows_after_dropna += int(row_count)

    def record_prediction_output(self, output: pd.DataFrame) -> None:
        if output.empty:
            return
        timestamps = pd.to_datetime(output["timestamp"], errors="coerce")
        if bool(timestamps.isna().any()):
            raise ValueError("invalid_prediction_timestamp_values")
        current_end = _date_string(timestamps.max())
        if self.effective_test_end is None or pd.Timestamp(current_end) > pd.Timestamp(
            self.effective_test_end
        ):
            self.effective_test_end = current_end
        self.prediction_rows += int(len(output))

    def prediction_metadata(
        self,
        *,
        spec: PredictionSpec,
        input_columns: tuple[str, ...],
    ) -> dict[str, object]:
        if self.prediction_rows == 0 or self.effective_test_end is None:
            raise ValueError("empty_prediction_window")
        return {
            "contract_version": spec.contract_version,
            "prediction_mode": spec.prediction_mode,
            "sample_filter_mode": spec.sample_filter_mode,
            "filter_columns_used": list(self.filter_columns_used),
            "filter_definition": self.filter_definition,
            "rows_input": int(self.rows_input),
            "rows_after_filter_mode": int(self.rows_after_filter),
            "rows_after_input_dropna": int(self.rows_after_dropna),
            "prediction_rows": int(self.prediction_rows),
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "requested_test_end": spec.test_end,
            "effective_test_end": self.effective_test_end,
            "is_oos": spec.is_oos,
            "is_final_oos_window": spec.is_final_oos_window,
            "input_columns": list(input_columns),
            "input_count": len(input_columns),
            "prediction_column": spec.prediction_column,
        }


def _prediction_static_window_model_metadata(
    model_metadata: dict[str, object],
    spec: PredictionSpec,
) -> dict[str, object]:
    return {
        **model_metadata,
        "test_start": spec.test_start,
        "test_end": spec.test_end,
        "oos_start": spec.test_start,
    }


def _iter_prediction_output_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    predictor: Callable[[pd.DataFrame], np.ndarray],
    build_state: _PredictionBuildState,
) -> Iterable[pd.DataFrame]:
    usable_columns = tuple(dict.fromkeys((*spec.primary_key, *input_columns)))
    for chunk in iter_complete_time_sized_frames(
        iter(frames),
        timestamp_column="timestamp",
        axis=ChunkAxis.TIME,
    ):
        filtered, metadata = apply_filter_mode(
            chunk,
            mode=spec.sample_filter_mode,
            metadata_key="sample_filter_mode",
        )
        build_state.record_filter_metadata(metadata)
        usable = filtered.dropna(subset=list(input_columns)).loc[
            :, list(usable_columns)
        ]
        build_state.record_input_dropna_rows(len(usable))
        test_mask = usable["timestamp"] >= pd.Timestamp(spec.test_start)
        if spec.test_end is not None:
            test_mask &= usable["timestamp"] <= pd.Timestamp(spec.test_end)
        test_frame = usable.loc[test_mask]
        if test_frame.empty:
            continue
        output = test_frame.loc[:, list(spec.primary_key)].copy()
        output[spec.prediction_column] = predictor(test_frame)
        build_state.record_prediction_output(output)
        yield output


def _prediction_callable(
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    prediction_model: object | None,
) -> Callable[[pd.DataFrame], np.ndarray]:
    if prediction_model is not None:
        def predict_from_model(frame: pd.DataFrame) -> np.ndarray:
            predicted = prediction_model.predict(frame.loc[:, input_columns])
            return np.asarray(predicted, dtype="float64")

        return predict_from_model

    def predict_from_coefficients(frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, input_columns].to_numpy(dtype=float, copy=False)
        return np.asarray(float(coefficients[0]) + x @ coefficients[1:])

    return predict_from_coefficients


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


def validate_prediction_artifacts(frame: pd.DataFrame, spec: PredictionSpec) -> None:
    expected = tuple(dict.fromkeys((*spec.primary_key, spec.prediction_column)))
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(
            "missing_prediction_artifact_columns: " + ", ".join(missing)
        )
    extra = sorted(set(frame.columns).difference(expected))
    if extra:
        raise ValueError(
            "unregistered_prediction_artifact_columns: " + ", ".join(extra)
        )
    if frame.empty:
        raise ValueError("empty_prediction_window")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_prediction_timestamp_values")
    if bool((timestamps < pd.Timestamp(spec.test_start)).any()):
        raise ValueError("prediction_artifact_before_test_start")
    if spec.test_end is not None and bool(
        (timestamps > pd.Timestamp(spec.test_end)).any()
    ):
        raise ValueError("prediction_artifact_after_test_end")


def write_prediction_artifacts(
    frame: pd.DataFrame,
    spec: PredictionSpec,
    *,
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
    output = frame.copy()
    if "model_metadata" in output.attrs:
        output.attrs["model_metadata"] = _prediction_static_window_model_metadata(
            output.attrs["model_metadata"],
            spec,
        )
    if "prediction_metadata" in output.attrs:
        output.attrs["prediction_metadata"] = dict(output.attrs["prediction_metadata"])
    return write_prediction_report_artifacts(
        output,
        spec=spec,
        report_spec=report_spec,
        layout=layout,
    )


__all__ = [
    "validate_prediction_artifacts",
    "write_prediction_artifacts",
]

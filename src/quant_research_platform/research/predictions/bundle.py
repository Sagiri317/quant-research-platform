"""Prediction bundle helpers for private report outputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
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
    _write_prediction_report_artifacts_from_chunks,
    write_prediction_report_artifacts,
)
from quant_research_platform.research._memory import iter_complete_time_sized_frames
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
    build_state = _PredictionBuildState()
    result = _build_prediction_frame(
        _iter_prediction_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ),
        spec=spec,
        coefficients=coefficients,
        feature_columns=feature_columns,
        build_state=build_state,
    )
    if result.empty:
        raise ValueError("empty_prediction_window")
    result.attrs["prediction_metadata"] = build_state.prediction_metadata(
        spec=spec,
        feature_columns=feature_columns,
    )
    return result


def _write_prediction_bundle_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
    *,
    model_metadata: dict[str, object],
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
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
    build_state = _PredictionBuildState()
    output_frames = _iter_prediction_output_frames(
        _iter_prediction_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ),
        spec=spec,
        coefficients=coefficients,
        feature_columns=feature_columns,
        build_state=build_state,
    )
    return _write_prediction_report_artifacts_from_chunks(
        output_frames,
        spec=spec,
        report_spec=report_spec,
        layout=layout,
        model_metadata=model_metadata,
        prediction_metadata_factory=lambda: build_state.prediction_metadata(
            spec=spec,
            feature_columns=feature_columns,
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

    def record_feature_dropna_rows(self, row_count: int) -> None:
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
        feature_columns: tuple[str, ...],
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
            "rows_after_feature_dropna": int(self.rows_after_dropna),
            "prediction_rows": int(self.prediction_rows),
            "train_end": spec.train_end,
            "test_start": spec.test_start,
            "requested_test_end": spec.test_end,
            "effective_test_end": self.effective_test_end,
            "is_oos": spec.is_oos,
            "is_final_oos_window": spec.is_final_oos_window,
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "prediction_column": spec.prediction_column,
        }


def _build_prediction_frame(
    frames: Iterable[pd.DataFrame],
    *,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
    build_state: _PredictionBuildState,
) -> pd.DataFrame:
    parts = list(
        _iter_prediction_output_frames(
            frames,
            spec=spec,
            coefficients=coefficients,
            feature_columns=feature_columns,
            build_state=build_state,
        )
    )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _iter_prediction_output_frames(
    frames: Iterable[pd.DataFrame],
    *,
    spec: PredictionSpec,
    coefficients: np.ndarray,
    feature_columns: tuple[str, ...],
    build_state: _PredictionBuildState,
) -> Iterable[pd.DataFrame]:
    usable_columns = tuple(dict.fromkeys((*spec.primary_key, *feature_columns)))
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
        usable = filtered.dropna(subset=list(feature_columns)).loc[
            :, list(usable_columns)
        ]
        build_state.record_feature_dropna_rows(len(usable))
        test_mask = usable["timestamp"] >= pd.Timestamp(spec.test_start)
        if spec.test_end is not None:
            test_mask &= usable["timestamp"] <= pd.Timestamp(spec.test_end)
        test_frame = usable.loc[test_mask]
        if test_frame.empty:
            continue
        x = test_frame.loc[:, feature_columns].to_numpy(dtype=float, copy=False)
        output = test_frame.loc[:, spec.primary_key].copy()
        output[spec.prediction_column] = float(coefficients[0]) + x @ coefficients[1:]
        build_state.record_prediction_output(output)
        yield output


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

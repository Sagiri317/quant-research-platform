"""One persisted-row summary observer shared by writer and inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_schema import date_string


@dataclass
class DatasetSummaryObserver:
    primary_key: tuple[str, ...]
    timestamp_column: str | None
    column_roles: dict[str, str]
    exact_key_index: ExactPrimaryKeyIndex
    row_count: int = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    null_count_summary: dict[str, int] = field(default_factory=dict)
    non_finite_count_summary: dict[str, int] = field(default_factory=dict)
    factor_nan_profile: dict[str, int] = field(default_factory=dict)
    label_nan_profile: dict[str, int] = field(default_factory=dict)
    duplicate_primary_key_count: int = 0

    def __post_init__(self) -> None:
        self.null_count_summary = {column: 0 for column in self.primary_key}

    def observe_batch(
        self,
        batch: pa.RecordBatch,
        *,
        frame: pd.DataFrame,
    ) -> None:
        table = pa.Table.from_batches([batch])
        duplicates = self.exact_key_index.add_table(table)
        self._observe_frame(frame, duplicates=duplicates)

    def observe_frame(self, frame: pd.DataFrame) -> None:
        duplicates = self.exact_key_index.add_frame(frame)
        self._observe_frame(frame, duplicates=duplicates)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "date_min": date_string(self.date_min),
            "date_max": date_string(self.date_max),
            "null_count_summary": dict(sorted(self.null_count_summary.items())),
            "non_finite_count_summary": dict(
                sorted(self.non_finite_count_summary.items())
            ),
            "duplicate_primary_key_count": int(self.duplicate_primary_key_count),
            "factor_nan_profile": dict(sorted(self.factor_nan_profile.items())),
            "label_nan_profile": dict(sorted(self.label_nan_profile.items())),
        }

    def _observe_frame(self, frame: pd.DataFrame, *, duplicates: int) -> None:
        self.row_count += int(len(frame))
        self.duplicate_primary_key_count += int(duplicates)
        for column in self.primary_key:
            self.null_count_summary[column] += int(frame[column].isna().sum())
        self._observe_dates(frame)
        self._observe_numeric_profiles(frame)

    def _observe_dates(self, frame: pd.DataFrame) -> None:
        if self.timestamp_column is None or frame.empty:
            return
        values = strict_timestamp_series(
            frame[self.timestamp_column],
            error="invalid_dataset_timestamp_value",
            allow_null=True,
        ).dropna()
        if values.empty:
            return
        current_min = values.min()
        current_max = values.max()
        self.date_min = (
            current_min if self.date_min is None else min(self.date_min, current_min)
        )
        self.date_max = (
            current_max if self.date_max is None else max(self.date_max, current_max)
        )

    def _observe_numeric_profiles(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        for column, role in self.column_roles.items():
            if column in self.primary_key or column not in frame:
                continue
            series = frame[column]
            nan_count = int(series.isna().sum())
            if role == "factor" and nan_count:
                self.factor_nan_profile[column] = (
                    self.factor_nan_profile.get(column, 0) + nan_count
                )
            if role == "label" and nan_count:
                self.label_nan_profile[column] = (
                    self.label_nan_profile.get(column, 0) + nan_count
                )
            if not pd.api.types.is_float_dtype(series.dtype):
                continue
            values = series.dropna().to_numpy(copy=False)
            inf_count = int(np.isinf(values).sum())
            if inf_count:
                self.non_finite_count_summary[column] = (
                    self.non_finite_count_summary.get(column, 0) + inf_count
                )

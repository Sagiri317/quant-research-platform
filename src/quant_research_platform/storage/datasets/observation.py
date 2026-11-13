"""Summarize keys, dates and missing values during writing or explicit inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from quant_research_platform.storage.datasets.schema import date_string
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


@dataclass
class DatasetSummaryObserver:
    primary_key: tuple[str, ...]
    timestamp_column: str | None
    column_roles: dict[str, str]
    exact_key_index: ExactPrimaryKeyIndex
    row_count: int = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    non_finite_count_summary: dict[str, int] = field(default_factory=dict)
    factor_nan_profile: dict[str, int] = field(default_factory=dict)
    label_nan_profile: dict[str, int] = field(default_factory=dict)
    duplicate_primary_key_count: int = 0

    def observe_batch(
        self, batch: pa.RecordBatch | pa.Table, *, frame: pd.DataFrame
    ) -> None:
        table = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
        duplicates = self.exact_key_index.add_table(table)
        self._observe_frame(frame, batch=batch, duplicates=duplicates)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "date_min": date_string(self.date_min),
            "date_max": date_string(self.date_max),
            "factor_nan_profile": dict(sorted(self.factor_nan_profile.items())),
            "label_nan_profile": dict(sorted(self.label_nan_profile.items())),
        }

    def _observe_frame(
        self, frame: pd.DataFrame, *, batch: pa.RecordBatch | pa.Table, duplicates: int
    ) -> None:
        self.row_count += int(len(frame))
        self.duplicate_primary_key_count += int(duplicates)
        self._observe_dates(frame)
        self._observe_numeric_profiles(frame, batch=batch)

    def _observe_dates(self, frame: pd.DataFrame) -> None:
        if self.timestamp_column is None or frame.empty:
            return
        values = frame[self.timestamp_column]
        current_min = values.min()
        current_max = values.max()
        self.date_min = (
            current_min if self.date_min is None else min(self.date_min, current_min)
        )
        self.date_max = (
            current_max if self.date_max is None else max(self.date_max, current_max)
        )

    def _observe_numeric_profiles(
        self, frame: pd.DataFrame, *, batch: pa.RecordBatch | pa.Table
    ) -> None:
        if frame.empty:
            return
        for column, role in self.column_roles.items():
            if column in self.primary_key or column not in frame:
                continue
            nan_count = (
                int(frame[column].isna().sum()) if role in {"factor", "label"} else 0
            )
            if role == "factor" and nan_count:
                self.factor_nan_profile[column] = (
                    self.factor_nan_profile.get(column, 0) + nan_count
                )
            if role == "label" and nan_count:
                self.label_nan_profile[column] = (
                    self.label_nan_profile.get(column, 0) + nan_count
                )
            values = batch.column(column)
            if not pa.types.is_floating(values.type):
                continue
            inf_count = int(pc.sum(pc.is_inf(values)).as_py() or 0)
            if inf_count:
                self.non_finite_count_summary[column] = (
                    self.non_finite_count_summary.get(column, 0) + inf_count
                )

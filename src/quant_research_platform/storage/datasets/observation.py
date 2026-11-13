"""Summarize actual rows, dates and research missing values during writing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant_research_platform.storage.datasets.schema import date_string


@dataclass
class DatasetSummaryObserver:
    primary_key: tuple[str, ...]
    timestamp_column: str | None
    column_roles: dict[str, str]
    row_count: int = 0
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    factor_nan_profile: dict[str, int] = field(default_factory=dict)
    label_nan_profile: dict[str, int] = field(default_factory=dict)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "date_min": date_string(self.date_min),
            "date_max": date_string(self.date_max),
            "factor_nan_profile": dict(sorted(self.factor_nan_profile.items())),
            "label_nan_profile": dict(sorted(self.label_nan_profile.items())),
        }

    def observe_frame(self, frame: pd.DataFrame) -> None:
        self.row_count += int(len(frame))
        self._observe_dates(frame)
        self._observe_numeric_profiles(frame)

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

    def _observe_numeric_profiles(self, frame: pd.DataFrame) -> None:
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

"""Scientific experiment windows and observable-label eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import (
    _validate_non_empty_string,
    strict_finite_numeric_series,
    strict_timestamp_series,
)

DEFAULT_ROOT_SEED: Final[int] = 42


@dataclass(frozen=True, kw_only=True)
class ExperimentFold:
    fold_id: str
    train: TimeWindow
    validation: TimeWindow

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.fold_id, "fold_id_required")
        if not all(
            isinstance(window, TimeWindow) for window in (self.train, self.validation)
        ):
            raise TypeError("experiment_fold_window_required")
        if isinstance(self.train.start, datetime) != isinstance(
            self.validation.start, datetime
        ):
            raise ValueError("experiment_time_kind_mismatch")
        if self.train.end >= self.validation.start:
            raise ValueError("experiment_fold_windows_overlap")


def label_maturity_mask(
    frame: pd.DataFrame,
    *,
    fitting_cutoff: object,
    availability_column: str | None = "label_available_at",
    target_column: str | None = None,
) -> pd.Series:
    """Select finite labels observable by the cutoff, independently of eligibility."""
    required = {"timestamp"}
    if availability_column is not None:
        required.add(availability_column)
    if target_column is not None:
        required.add(target_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("legal_training_columns_missing: " + ", ".join(missing))
    cutoff = strict_timestamp_series(
        pd.Series([fitting_cutoff]), error="invalid_fitting_cutoff"
    ).iloc[0]
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_training_timestamp"
    )
    keep = timestamps.le(cutoff)
    if availability_column is not None:
        available = strict_timestamp_series(
            frame[availability_column],
            error="invalid_label_available_at",
            allow_null=True,
        )
        keep &= available.notna() & available.le(cutoff)
    if target_column is not None:
        values = strict_finite_numeric_series(
            frame[target_column],
            error=f"invalid_training_value: {target_column}",
            allow_null=True,
        )
        keep &= values.notna()
    return keep


def legal_training_mask(
    frame: pd.DataFrame,
    *,
    fitting_cutoff: object,
    availability_column: str | None = "label_available_at",
    feature_columns: tuple[str, ...] = (),
    target_column: str | None = None,
) -> pd.Series:
    """Validate raw rows and identify eligible finite inputs with mature labels."""
    missing = sorted({"eligible", *feature_columns}.difference(frame.columns))
    if missing:
        raise ValueError("legal_training_columns_missing: " + ", ".join(missing))
    eligible = frame["eligible"]
    if not pd.api.types.is_bool_dtype(eligible) or bool(eligible.isna().any()):
        raise ValueError("invalid_training_eligibility")
    keep = (
        label_maturity_mask(
            frame,
            fitting_cutoff=fitting_cutoff,
            availability_column=availability_column,
            target_column=target_column,
        )
        & eligible
    )
    for column in feature_columns:
        values = strict_finite_numeric_series(
            frame[column], error=f"invalid_training_value: {column}", allow_null=True
        )
        keep &= values.notna()
    return keep


def legal_training_rows(
    frame: pd.DataFrame,
    *,
    fitting_cutoff: object,
    availability_column: str | None = "label_available_at",
    feature_columns: tuple[str, ...] = (),
    target_column: str | None = None,
) -> pd.DataFrame:
    """Select eligible finite inputs with mature labels."""
    return frame.loc[
        legal_training_mask(
            frame,
            fitting_cutoff=fitting_cutoff,
            availability_column=availability_column,
            feature_columns=feature_columns,
            target_column=target_column,
        )
    ].copy()


def training_availability_columns(label_column: str) -> tuple[str, ...]:
    if label_column == "model_target":
        return ("model_target_available_at",)
    if label_column.startswith(
        (
            "forward_return_",
            "future_mid_return_",
            "future_mid_direction_",
            "maker_bid_sell_pressure_proxy_",
        )
    ):
        return (f"{label_column}_available_at",)
    return ("label_available_at",)

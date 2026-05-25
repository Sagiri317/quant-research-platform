"""定义 evaluation kernel 共享协议和轻量配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

DATAFRAME_KERNEL = "dataframe"
TENSOR_KERNEL = "tensor"
SUPPORTED_KERNELS = {DATAFRAME_KERNEL, TENSOR_KERNEL}

WEIGHT_METHODS = {
    "equal_weight_quantile",
    "rank_weighted_quantile",
    "zscore_weighted_quantile",
    "signal_abs_weighted_quantile",
    "inverse_vol_scaled_quantile",
}
WEIGHT_DIRECTIONS = {"long_high", "long_low"}
PORTFOLIO_MODES = {"long_short", "long_only"}
BLOCKED_ENTRY_POLICIES = {"skip", "hold_cash", "redistribute"}
BLOCKED_EXIT_POLICIES = {"carry_forward", "force_exit", "error"}


class PanelSpec(BaseModel):
    """定义 timestamp-by-asset research panel 的 column contract。"""

    model_config = ConfigDict(frozen=True)

    feature_columns: tuple[str, ...]
    label_column: str
    timestamp_column: str = "timestamp"
    asset_column: str = "asset"

    @field_validator("feature_columns")
    @classmethod
    def _validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """校验 feature columns 非空且不重复。"""

        if not value:
            raise ValueError("feature_columns must contain at least one column.")
        if any(not isinstance(column, str) or not column for column in value):
            raise ValueError("feature_columns must contain non-empty strings.")
        if len(set(value)) != len(value):
            raise ValueError("feature_columns must not contain duplicates.")
        return value

    @field_validator("label_column", "timestamp_column", "asset_column")
    @classmethod
    def _validate_named_column(cls, value: str) -> str:
        """校验关键 column 名称非空。"""

        if not isinstance(value, str) or not value:
            raise ValueError("column names must be non-empty strings.")
        return value

    def model_post_init(self, __context: Any) -> None:
        """校验研究 columns 不复用 timestamp、asset 或 label 角色。"""

        if self.label_column in self.feature_columns:
            raise ValueError("label_column must not also be a feature column.")
        reserved = {self.timestamp_column, self.asset_column}
        if self.label_column in reserved or any(
            feature in reserved for feature in self.feature_columns
        ):
            raise ValueError(
                "research columns must not reuse timestamp or asset columns."
            )


@dataclass(frozen=True)
class WeightConfig:
    """资产实验选择的 portfolio weighting policy。"""

    method: str = "equal_weight_quantile"
    direction: str = "long_high"
    portfolio_mode: str = "long_short"
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    cost_bps: float = 0.0
    risk_column: str | None = None

    def __post_init__(self) -> None:
        """校验 weighting method、direction、quantiles 和 risk column。"""

        if self.method not in WEIGHT_METHODS:
            raise ValueError(f"method must be one of {sorted(WEIGHT_METHODS)}.")
        if self.direction not in WEIGHT_DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(WEIGHT_DIRECTIONS)}.")
        if self.portfolio_mode not in PORTFOLIO_MODES:
            raise ValueError(
                f"portfolio_mode must be one of {sorted(PORTFOLIO_MODES)}."
            )
        _validate_quantile(self.top_quantile, "top_quantile")
        _validate_quantile(self.bottom_quantile, "bottom_quantile")
        if self.top_quantile + self.bottom_quantile > 1.0:
            raise ValueError(
                "top_quantile + bottom_quantile must be less than or equal to 1."
            )
        _validate_nonnegative(self.cost_bps, "cost_bps")
        if self.method == "inverse_vol_scaled_quantile" and not self.risk_column:
            raise ValueError("risk_column is required for inverse_vol_scaled_quantile.")
        if self.risk_column is not None and (
            not isinstance(self.risk_column, str) or not self.risk_column
        ):
            raise ValueError("risk_column must be a non-empty string.")


@dataclass(frozen=True)
class TradabilityConfig:
    """市场无关的 eligibility 和 entry/exit constraint metadata。"""

    eligibility_column: str | None = None
    can_enter_column: str | None = None
    can_exit_column: str | None = None
    blocked_entry_policy: str = "skip"
    blocked_exit_policy: str = "carry_forward"

    def __post_init__(self) -> None:
        """校验 eligibility、entry/exit masks 和 blocked policies。"""

        _validate_optional_column_name(self.eligibility_column, "eligibility_column")
        _validate_optional_column_name(self.can_enter_column, "can_enter_column")
        _validate_optional_column_name(self.can_exit_column, "can_exit_column")
        if self.blocked_entry_policy not in BLOCKED_ENTRY_POLICIES:
            raise ValueError(
                "blocked_entry_policy must be one of "
                f"{sorted(BLOCKED_ENTRY_POLICIES)}."
            )
        if self.blocked_exit_policy not in BLOCKED_EXIT_POLICIES:
            raise ValueError(
                "blocked_exit_policy must be one of "
                f"{sorted(BLOCKED_EXIT_POLICIES)}."
            )


@dataclass(frozen=True)
class CostConfig:
    """面向通用 cost evaluation 的详细 cost model metadata。"""

    buy_cost_bps: float = 0.0
    sell_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float | None = None
    impact_model: str | None = None

    def __post_init__(self) -> None:
        """校验非负 cost fields 和可选 impact model。"""

        _validate_nonnegative(self.buy_cost_bps, "buy_cost_bps")
        _validate_nonnegative(self.sell_cost_bps, "sell_cost_bps")
        _validate_nonnegative(self.slippage_bps, "slippage_bps")
        if self.spread_bps is not None:
            _validate_nonnegative(self.spread_bps, "spread_bps")
        if self.impact_model is not None and (
            not isinstance(self.impact_model, str) or not self.impact_model
        ):
            raise ValueError("impact_model must be a non-empty string.")


def validate_panel(frame: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    """校验并规范化 long-format research panel。"""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not isinstance(spec, PanelSpec):
        raise TypeError("spec must be a PanelSpec.")

    required_columns = [
        spec.timestamp_column,
        spec.asset_column,
        *spec.feature_columns,
        spec.label_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"panel is missing columns: {missing_columns}.")

    result = frame.copy()
    result[spec.timestamp_column] = pd.to_datetime(result[spec.timestamp_column])
    if result[spec.timestamp_column].isna().any():
        raise ValueError("timestamp column contains missing timestamps.")
    if result[spec.asset_column].isna().any():
        raise ValueError("asset column contains missing assets.")
    result[spec.asset_column] = result[spec.asset_column].astype(str)

    duplicated = result.duplicated([spec.timestamp_column, spec.asset_column])
    if duplicated.any():
        raise ValueError("panel rows must have unique timestamp-asset keys.")

    for column in (*spec.feature_columns, spec.label_column):
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"column {column!r} must be numeric.")

    return result.sort_values(
        [spec.timestamp_column, spec.asset_column],
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_quantile(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    if not 0 < float(value) <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1.")


def _validate_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    if float(value) < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _validate_optional_column_name(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string.")

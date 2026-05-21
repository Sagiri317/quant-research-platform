"""Public contracts for asset-agnostic alpha research evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.core.panel import PanelSpec
from quant_research_platform.core.profiling import ProfileConfig

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
NEUTRALIZATION_METHODS = {"none", "demean_by_group", "cross_sectional_ols"}
NEUTRALIZATION_FIT_SCOPES = {"per_timestamp"}
TIME_BREAKDOWNS = {"year", "quarter", "month"}
EVALUATION_ENGINES = {"dataframe", "tensor"}
DEFAULT_PNL_RETURN_COLUMN = "forward_return_1d"
TENSOR_BACKENDS = {"numpy"}
TENSOR_DEVICES = {"cpu"}
TENSOR_VALUE_DTYPES = {"float32", "float64"}
TENSOR_AGGREGATION_DTYPES = {"float64"}
MINIMUM_RUN_METADATA_FIELDS = (
    "project",
    "kind",
    "domain",
    "data_stage",
    "uses_external_data",
)


@dataclass(frozen=True)
class EvaluationSpec:
    """Batch-native evaluation-set input contract."""

    panel: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    metadata: dict[str, Any]
    timestamp_column: str = "timestamp"
    asset_column: str = "asset"
    n_groups: int = 5
    weight_configs: tuple[WeightConfig, ...] = field(
        default_factory=lambda: (WeightConfig(),)
    )
    execution_config: ExecutionConfig = field(
        default_factory=lambda: ExecutionConfig()
    )
    tradability_config: TradabilityConfig = field(
        default_factory=lambda: TradabilityConfig()
    )
    cost_configs: tuple[CostConfig, ...] = field(
        default_factory=lambda: (CostConfig(),)
    )
    neutralization_config: NeutralizationConfig = field(
        default_factory=lambda: NeutralizationConfig()
    )
    tensor_config: TensorBatchConfig = field(
        default_factory=lambda: TensorBatchConfig()
    )
    segment_columns: tuple[str, ...] = ()
    time_breakdowns: tuple[str, ...] = ()
    engine: str = "dataframe"
    pnl_return_column: str | None = None
    output_dir: str | Path = Path("reports/runs/demo")
    profile_config: ProfileConfig = field(default_factory=ProfileConfig)
    domain_notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the evaluation set, metadata, and shared configuration."""

        if not isinstance(self.panel, pd.DataFrame):
            raise TypeError("panel must be a pandas DataFrame.")
        _validate_column_names(self.feature_columns, "feature_columns")
        _validate_column_names(self.label_columns, "label_columns")
        if set(self.feature_columns).intersection(self.label_columns):
            raise ValueError("feature_columns and label_columns must not overlap.")
        _validate_optional_column_name(self.timestamp_column, "timestamp_column")
        _validate_optional_column_name(self.asset_column, "asset_column")
        reserved = {self.timestamp_column, self.asset_column}
        if reserved.intersection((*self.feature_columns, *self.label_columns)):
            raise ValueError(
                "feature_columns and label_columns must not reuse timestamp "
                "or asset columns."
            )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        validate_run_metadata(self.metadata)
        if not isinstance(self.domain_notes, dict):
            raise TypeError("domain_notes must be a dictionary.")
        if not isinstance(self.n_groups, int) or isinstance(self.n_groups, bool):
            raise TypeError("n_groups must be an integer.")
        if self.n_groups < 2:
            raise ValueError("n_groups must be at least 2.")
        if not self.weight_configs:
            raise ValueError("weight_configs must contain at least one config.")
        if any(not isinstance(config, WeightConfig) for config in self.weight_configs):
            raise TypeError("weight_configs must contain WeightConfig objects.")
        if not self.cost_configs:
            raise ValueError("cost_configs must contain at least one config.")
        if any(not isinstance(config, CostConfig) for config in self.cost_configs):
            raise TypeError("cost_configs must contain CostConfig objects.")
        if not isinstance(self.execution_config, ExecutionConfig):
            raise TypeError("execution_config must be an ExecutionConfig.")
        if not isinstance(self.tradability_config, TradabilityConfig):
            raise TypeError("tradability_config must be a TradabilityConfig.")
        if not isinstance(self.neutralization_config, NeutralizationConfig):
            raise TypeError("neutralization_config must be a NeutralizationConfig.")
        if not isinstance(self.tensor_config, TensorBatchConfig):
            raise TypeError("tensor_config must be a TensorBatchConfig.")
        _validate_column_names(self.segment_columns, "segment_columns")
        if any(
            not isinstance(value, str) or value not in TIME_BREAKDOWNS
            for value in self.time_breakdowns
        ):
            raise ValueError(
                f"time_breakdowns must contain only {sorted(TIME_BREAKDOWNS)}."
            )
        if len(set(self.time_breakdowns)) != len(self.time_breakdowns):
            raise ValueError("time_breakdowns must not contain duplicates.")
        if self.engine not in EVALUATION_ENGINES:
            raise ValueError(f"engine must be one of {sorted(EVALUATION_ENGINES)}.")
        if self.pnl_return_column is not None and (
            not isinstance(self.pnl_return_column, str) or not self.pnl_return_column
        ):
            raise ValueError("pnl_return_column must be a non-empty string.")
        if not isinstance(self.profile_config, ProfileConfig):
            raise TypeError("profile_config must be a ProfileConfig.")

    def panel_spec_for_label(self, label_column: str) -> PanelSpec:
        """Build the single-label PanelSpec used by metric engines."""

        if label_column not in self.label_columns:
            raise ValueError(f"unknown label_column {label_column!r}.")
        return PanelSpec(
            feature_columns=self.feature_columns,
            label_column=label_column,
            timestamp_column=self.timestamp_column,
            asset_column=self.asset_column,
        )

    def resolve_pnl_return_column(self, panel: pd.DataFrame) -> str:
        """Resolve the realized-return column used for portfolio PnL."""

        if self.pnl_return_column is not None:
            return self.pnl_return_column
        if DEFAULT_PNL_RETURN_COLUMN in panel.columns:
            return DEFAULT_PNL_RETURN_COLUMN
        raise ValueError(
            "PnL curve requires pnl_return_column, but none was provided and "
            f"default column {DEFAULT_PNL_RETURN_COLUMN!r} was not found."
        )


@dataclass(frozen=True)
class WeightConfig:
    """Portfolio weighting policy selected by an asset-specific lab."""

    method: str = "equal_weight_quantile"
    direction: str = "long_high"
    portfolio_mode: str = "long_short"
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    cost_bps: float = 0.0
    risk_column: str | None = None

    def __post_init__(self) -> None:
        """Validate weighting method, direction, quantiles, and risk column."""

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
class ExecutionConfig:
    """Market-agnostic execution timing and price selection metadata."""

    execution_lag: int = 0
    execution_price_column: str | None = None

    def __post_init__(self) -> None:
        """Validate execution lag and optional execution price column."""

        if not isinstance(self.execution_lag, int) or isinstance(
            self.execution_lag,
            bool,
        ):
            raise TypeError("execution_lag must be an integer.")
        if self.execution_lag < 0:
            raise ValueError("execution_lag must be nonnegative.")
        _validate_optional_column_name(
            self.execution_price_column,
            "execution_price_column",
        )


@dataclass(frozen=True)
class TradabilityConfig:
    """Market-agnostic eligibility and entry/exit constraint metadata."""

    eligibility_column: str | None = None
    can_enter_column: str | None = None
    can_exit_column: str | None = None
    blocked_entry_policy: str = "skip"
    blocked_exit_policy: str = "carry_forward"

    def __post_init__(self) -> None:
        """Validate eligibility, entry/exit masks, and blocked policies."""

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
    """Detailed cost model metadata for future generic cost evaluation."""

    buy_cost_bps: float = 0.0
    sell_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float | None = None
    impact_model: str | None = None

    def __post_init__(self) -> None:
        """Validate nonnegative cost fields and optional impact model."""

        _validate_nonnegative(self.buy_cost_bps, "buy_cost_bps")
        _validate_nonnegative(self.sell_cost_bps, "sell_cost_bps")
        _validate_nonnegative(self.slippage_bps, "slippage_bps")
        if self.spread_bps is not None:
            _validate_nonnegative(self.spread_bps, "spread_bps")
        if self.impact_model is not None and (
            not isinstance(self.impact_model, str) or not self.impact_model
        ):
            raise ValueError("impact_model must be a non-empty string.")


@dataclass(frozen=True)
class NeutralizationConfig:
    """Market-agnostic cross-sectional feature neutralization metadata."""

    method: str = "none"
    group_columns: tuple[str, ...] = ()
    exposure_columns: tuple[str, ...] = ()
    fit_scope: str = "per_timestamp"

    def __post_init__(self) -> None:
        """Validate neutralization method and group/exposure columns."""

        if self.method not in NEUTRALIZATION_METHODS:
            raise ValueError(
                f"method must be one of {sorted(NEUTRALIZATION_METHODS)}."
            )
        if self.fit_scope not in NEUTRALIZATION_FIT_SCOPES:
            raise ValueError(
                f"fit_scope must be one of {sorted(NEUTRALIZATION_FIT_SCOPES)}."
            )
        _validate_column_names(self.group_columns, "group_columns")
        _validate_column_names(self.exposure_columns, "exposure_columns")
        if self.method == "demean_by_group" and not self.group_columns:
            raise ValueError("group_columns are required for demean_by_group.")
        if self.method == "cross_sectional_ols" and not self.exposure_columns:
            raise ValueError("exposure_columns are required for cross_sectional_ols.")
        if self.method == "none" and (self.group_columns or self.exposure_columns):
            raise ValueError("neutralization columns require a non-none method.")


@dataclass(frozen=True)
class TensorBatchConfig:
    """Internal tensor engine execution and memory-budget metadata."""

    backend: str = "numpy"
    device: str = "cpu"
    value_dtype: str = "float64"
    aggregation_dtype: str = "float64"
    signal_chunk_size: int | None = None
    label_chunk_size: int | None = None
    max_memory_mb: float | None = None

    def __post_init__(self) -> None:
        """Validate tensor backend, dtypes, chunk sizes, and memory budget."""

        if self.backend not in TENSOR_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(TENSOR_BACKENDS)}.")
        if self.device not in TENSOR_DEVICES:
            raise ValueError(f"device must be one of {sorted(TENSOR_DEVICES)}.")
        if self.value_dtype not in TENSOR_VALUE_DTYPES:
            raise ValueError(
                f"value_dtype must be one of {sorted(TENSOR_VALUE_DTYPES)}."
            )
        if self.aggregation_dtype not in TENSOR_AGGREGATION_DTYPES:
            raise ValueError(
                "aggregation_dtype must be one of "
                f"{sorted(TENSOR_AGGREGATION_DTYPES)}."
            )
        _validate_optional_positive_int(self.signal_chunk_size, "signal_chunk_size")
        _validate_optional_positive_int(self.label_chunk_size, "label_chunk_size")
        if self.max_memory_mb is not None:
            if isinstance(self.max_memory_mb, bool) or not isinstance(
                self.max_memory_mb,
                (int, float),
            ):
                raise TypeError("max_memory_mb must be numeric.")
            if float(self.max_memory_mb) < 0:
                raise ValueError("max_memory_mb must be nonnegative.")


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


def _validate_optional_positive_int(value: int | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")


def _validate_optional_column_name(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string.")


def _validate_column_names(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates.")


def validate_run_metadata(
    metadata: dict[str, Any],
    *,
    source: str = "metadata",
) -> None:
    """Validate provenance fields required by artifact consumers."""

    missing = [
        field for field in MINIMUM_RUN_METADATA_FIELDS if field not in metadata
    ]
    if missing:
        raise ValueError(
            f"{source} is missing required fields: {', '.join(missing)}."
        )
    for metadata_field in ("project", "kind", "domain", "data_stage"):
        value = metadata[metadata_field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{source} {metadata_field} must be a non-empty string."
            )
    if not isinstance(metadata["uses_external_data"], bool):
        raise TypeError(f"{source} uses_external_data must be bool.")

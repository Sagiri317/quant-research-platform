"""Portfolio contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.data import (
    DataFrequency,
    data_frequency_slug,
    normalize_data_frequency,
)
from quant_research_platform.contracts.filtering import reject_filter_input_columns
from quant_research_platform.contracts.ids import MarketId, validate_market_id
from quant_research_platform.contracts.signals import (
    SignalSpec,
    _cross_signal_id_column_overlaps,
)
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    strict_finite_float,
)

WEIGHT_METHODS = {
    "equal_weight_quantile",
    "rank_weighted_quantile",
    "zscore_weighted_quantile",
    "signal_abs_weighted_quantile",
    "inverse_vol_scaled_quantile",
}
PORTFOLIO_MODES = {"long_short", "long_only"}
BLOCKED_ENTRY_POLICIES = {"hold_cash", "redistribute"}
BLOCKED_EXIT_POLICIES = {"carry_forward", "force_exit", "error"}
SUPPORTED_REBALANCE_FREQUENCIES = {data_frequency_slug(DataFrequency.DAILY)}
PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "row_count",
    "positive_amount_row_count",
    "excluded_non_positive_amount_count",
    "gross_weight",
    "covered_gross_weight",
    "weighted_amount_proxy",
    "weighted_float_market_cap_proxy",
)
PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "exposure_column",
    "exposure_value",
    "row_count",
    "gross_weight",
)
PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "exposure_column",
    "exposure_value",
    "asset_count",
    "gross_weight",
    "net_weight",
    "long_weight",
    "short_weight",
)


@dataclass(frozen=True)
class WeightConfig:
    """Portfolio weighting policy for normalized signal inputs."""

    method: str = "equal_weight_quantile"
    portfolio_mode: str = "long_short"
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    risk_column: str | None = None

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in WEIGHT_METHODS:
            raise ValueError(f"method must be one of {sorted(WEIGHT_METHODS)}.")
        if (
            type(self.portfolio_mode) is not str
            or self.portfolio_mode not in PORTFOLIO_MODES
        ):
            raise ValueError(
                f"portfolio_mode must be one of {sorted(PORTFOLIO_MODES)}."
            )
        _validate_quantile(self.top_quantile, "top_quantile")
        if self.portfolio_mode == "long_short":
            _validate_quantile(self.bottom_quantile, "bottom_quantile")
        elif isinstance(self.bottom_quantile, bool) or not isinstance(
            self.bottom_quantile,
            int | float,
        ):
            raise TypeError("bottom_quantile must be numeric.")
        if (
            self.portfolio_mode == "long_short"
            and self.top_quantile + self.bottom_quantile > 1.0
        ):
            raise ValueError(
                "top_quantile + bottom_quantile must be less than or equal to 1."
            )
        if self.method == "inverse_vol_scaled_quantile" and not self.risk_column:
            raise ValueError("risk_column is required for inverse_vol_scaled_quantile.")
        _validate_optional_column_name(self.risk_column, "risk_column")


@dataclass(frozen=True)
class TradabilityConfig:
    """Market-agnostic eligibility and entry/exit constraint metadata."""

    eligibility_column: str | None = None
    can_enter_column: str | None = None
    can_exit_column: str | None = None
    blocked_entry_policy: str = "hold_cash"
    blocked_exit_policy: str = "carry_forward"

    def __post_init__(self) -> None:
        _validate_optional_column_name(self.eligibility_column, "eligibility_column")
        _validate_optional_column_name(self.can_enter_column, "can_enter_column")
        _validate_optional_column_name(self.can_exit_column, "can_exit_column")
        if (
            type(self.blocked_entry_policy) is not str
            or self.blocked_entry_policy not in BLOCKED_ENTRY_POLICIES
        ):
            raise ValueError(
                f"blocked_entry_policy must be one of {sorted(BLOCKED_ENTRY_POLICIES)}."
            )
        if (
            type(self.blocked_exit_policy) is not str
            or self.blocked_exit_policy not in BLOCKED_EXIT_POLICIES
        ):
            raise ValueError(
                f"blocked_exit_policy must be one of {sorted(BLOCKED_EXIT_POLICIES)}."
            )


@dataclass(frozen=True)
class CostConfig:
    """Detailed compute cost model metadata."""

    buy_cost_bps: float = 0.0
    sell_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative(self.buy_cost_bps, "buy_cost_bps")
        _validate_nonnegative(self.sell_cost_bps, "sell_cost_bps")
        _validate_nonnegative(self.slippage_bps, "slippage_bps")
        if self.spread_bps is not None:
            _validate_nonnegative(self.spread_bps, "spread_bps")


@dataclass(frozen=True)
class PortfolioDiagnosticsConfig:
    """Portfolio diagnostics column configuration."""

    capacity_amount_column: str | None = None
    capacity_float_market_cap_column: str | None = None
    numeric_exposure_columns: tuple[str, ...] = ()
    categorical_exposure_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_optional_column_name(
            self.capacity_amount_column,
            "capacity_amount_column",
        )
        _validate_optional_column_name(
            self.capacity_float_market_cap_column,
            "capacity_float_market_cap_column",
        )
        _validate_column_tuple(
            self.numeric_exposure_columns,
            "numeric_exposure_columns",
            required=False,
        )
        _validate_column_tuple(
            self.categorical_exposure_columns,
            "categorical_exposure_columns",
            required=False,
        )
        if (
            self.capacity_float_market_cap_column is not None
            and self.capacity_amount_column is None
        ):
            raise ValueError("capacity_amount_column_required")
        overlap = sorted(
            set(self.numeric_exposure_columns).intersection(
                self.categorical_exposure_columns
            )
        )
        if overlap:
            raise ValueError(
                "diagnostic_exposure_columns_must_not_overlap: " + ", ".join(overlap)
            )


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio_id: str
    market_id: MarketId
    research_dataset_id: str
    signals: tuple[SignalSpec, ...]
    return_source_column: str
    rebalance_frequency: str
    execution_lag_steps: int
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    cost_config: CostConfig = field(default_factory=CostConfig)
    tradability_config: TradabilityConfig = field(default_factory=TradabilityConfig)
    diagnostics_config: PortfolioDiagnosticsConfig = field(
        default_factory=PortfolioDiagnosticsConfig
    )
    run_type: str = "portfolio"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if type(self.contract_version) is not str or self.contract_version != "0.1":
            raise ValueError("unsupported_portfolio_contract_version")
        if type(self.portfolio_id) is not str or not self.portfolio_id:
            raise ValueError("portfolio_id_required")
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        if type(self.research_dataset_id) is not str or not self.research_dataset_id:
            raise ValueError("research_dataset_id_required")
        if type(self.signals) is not tuple:
            raise ValueError("portfolio_signals_must_be_tuple")
        if any(type(signal) is not SignalSpec for signal in self.signals):
            raise ValueError("portfolio_signals_invalid")
        if not self.signals:
            raise ValueError("portfolio_signals_required")
        signal_ids = [signal.signal_id for signal in self.signals]
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError("portfolio_signal_ids_must_not_contain_duplicates")
        signal_columns = [signal.source_column for signal in self.signals]
        if len(set(signal_columns)) != len(signal_columns):
            raise ValueError("portfolio_signal_columns_must_not_contain_duplicates")
        overlaps = _cross_signal_id_column_overlaps(self.signals)
        if overlaps:
            raise ValueError(
                "portfolio_signal_id_column_overlap: " + ", ".join(overlaps)
            )
        if type(self.return_source_column) is not str or not self.return_source_column:
            raise ValueError("portfolio_return_source_column_required")
        object.__setattr__(
            self,
            "rebalance_frequency",
            normalize_rebalance_frequency(self.rebalance_frequency),
        )
        if not isinstance(self.execution_lag_steps, int) or isinstance(
            self.execution_lag_steps,
            bool,
        ):
            raise TypeError("execution_lag_steps must be an integer")
        if self.execution_lag_steps < 0:
            raise ValueError("execution_lag_steps_must_be_nonnegative")
        if type(self.run_type) is not str or not self.run_type:
            raise ValueError("portfolio_run_type_required")
        reject_filter_input_columns((*signal_ids, *signal_columns))
        invalid_labels = sorted(
            set((*signal_ids, *signal_columns)).intersection(
                {self.return_source_column}
            )
        )
        if invalid_labels:
            raise ValueError(
                "portfolio_signal_return_overlap: " + ", ".join(invalid_labels)
            )


def _validate_quantile(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    error = f"{name} must be greater than 0 and at most 1."
    numeric = strict_finite_float(value, error=error)
    if not 0 < numeric <= 1:
        raise ValueError(error)


def _validate_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    error = f"{name} must be nonnegative."
    numeric = strict_finite_float(value, error=error)
    if numeric < 0:
        raise ValueError(error)


def _validate_optional_column_name(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string.")


def normalize_rebalance_frequency(value: str | DataFrequency) -> str:
    if isinstance(value, str) and not value:
        raise ValueError("portfolio_rebalance_frequency_required")
    try:
        normalized = data_frequency_slug(normalize_data_frequency(value))
    except ValueError as exc:
        raise ValueError("unsupported_portfolio_rebalance_frequency") from exc
    if normalized not in SUPPORTED_REBALANCE_FREQUENCIES:
        raise ValueError("unsupported_portfolio_rebalance_frequency")
    return normalized

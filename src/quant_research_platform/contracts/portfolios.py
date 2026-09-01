"""Common portfolio contracts and market-policy bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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

WeightMethod = Literal[
    "equal_weight_quantile",
    "float_cap_weighted_quantile",
    "rank_weighted_quantile",
    "zscore_weighted_quantile",
    "signal_abs_weighted_quantile",
    "inverse_vol_scaled_quantile",
]
WEIGHT_METHODS = {
    "equal_weight_quantile",
    "float_cap_weighted_quantile",
    "rank_weighted_quantile",
    "zscore_weighted_quantile",
    "signal_abs_weighted_quantile",
    "inverse_vol_scaled_quantile",
}
PORTFOLIO_MODES = {"long_short", "long_only"}
BLOCKED_ENTRY_POLICIES = {"hold_cash", "redistribute"}
BLOCKED_EXIT_POLICIES = {"carry_forward", "force_exit", "error"}
REBALANCE_SCHEDULES = {
    "daily",
    "weekly-last-trading-day",
    "monthly-last-trading-day",
}
EXECUTION_RETURN_COLUMN = "forward_return_1d"
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

    method: WeightMethod = "equal_weight_quantile"
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
            if self.top_quantile + self.bottom_quantile > 1.0:
                raise ValueError(
                    "top_quantile + bottom_quantile must be less than or equal to 1."
                )
        elif isinstance(self.bottom_quantile, bool) or not isinstance(
            self.bottom_quantile, int | float
        ):
            raise TypeError("bottom_quantile must be numeric.")
        if self.method == "inverse_vol_scaled_quantile" and not self.risk_column:
            raise ValueError("risk_column is required for inverse_vol_scaled_quantile.")
        _validate_optional_column_name(self.risk_column, "risk_column")


@dataclass(frozen=True)
class TradabilityConfig:
    """Market-agnostic mask names; meaning is supplied by a market binding."""

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
    """Explicit common cost parameters; market fees remain a market binding."""

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_bps: float = 0.0
    adv_participation: float = 1.0
    adv_sensitivities: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

    def __post_init__(self) -> None:
        _validate_nonnegative(self.commission_bps, "commission_bps")
        _validate_nonnegative(self.slippage_bps, "slippage_bps")
        _validate_nonnegative(self.impact_bps, "impact_bps")
        participation = strict_finite_float(
            self.adv_participation,
            error="adv_participation_must_be_positive",
        )
        if participation <= 0.0 or participation > 1.0:
            raise ValueError("adv_participation_must_be_positive")
        if type(self.adv_sensitivities) is not tuple or not self.adv_sensitivities:
            raise ValueError("adv_sensitivities_invalid")
        for value in self.adv_sensitivities:
            numeric = strict_finite_float(value, error="adv_sensitivities_invalid")
            if numeric <= 0.0 or numeric > 1.0:
                raise ValueError("adv_sensitivities_invalid")


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
    rebalance_schedule: str
    execution_policy_id: str
    cost_policy_id: str
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
        if type(self.signals) is not tuple or not self.signals:
            raise ValueError("portfolio_signals_required")
        if any(type(signal) is not SignalSpec for signal in self.signals):
            raise ValueError("portfolio_signals_invalid")
        signal_ids = [signal.signal_id for signal in self.signals]
        signal_columns = [signal.source_column for signal in self.signals]
        if len(set(signal_ids)) != len(signal_ids):
            raise ValueError("portfolio_signal_ids_must_not_contain_duplicates")
        if len(set(signal_columns)) != len(signal_columns):
            raise ValueError("portfolio_signal_columns_must_not_contain_duplicates")
        overlaps = _cross_signal_id_column_overlaps(self.signals)
        if overlaps:
            raise ValueError(
                "portfolio_signal_id_column_overlap: " + ", ".join(overlaps)
            )
        if self.rebalance_schedule not in REBALANCE_SCHEDULES:
            raise ValueError("unsupported_portfolio_rebalance_schedule")
        _validate_non_empty_identifier(self.execution_policy_id, "execution_policy_id")
        _validate_non_empty_identifier(self.cost_policy_id, "cost_policy_id")
        if type(self.run_type) is not str or not self.run_type:
            raise ValueError("portfolio_run_type_required")
        reject_filter_input_columns((*signal_ids, *signal_columns))


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
    numeric = strict_finite_float(value, error=f"{name} must be nonnegative.")
    if numeric < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _validate_optional_column_name(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string.")


def _validate_non_empty_identifier(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name}_required")

"""Portfolio contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.filtering import reject_filter_input_columns
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.signals import SignalSpec

WEIGHT_METHODS = {
    "equal_weight_quantile",
    "rank_weighted_quantile",
    "zscore_weighted_quantile",
    "signal_abs_weighted_quantile",
    "inverse_vol_scaled_quantile",
}
PORTFOLIO_MODES = {"long_short", "long_only"}
BLOCKED_ENTRY_POLICIES = {"skip", "hold_cash", "redistribute"}
BLOCKED_EXIT_POLICIES = {"carry_forward", "force_exit", "error"}


@dataclass(frozen=True)
class WeightConfig:
    """Portfolio weighting policy for normalized signal inputs."""

    method: str = "equal_weight_quantile"
    portfolio_mode: str = "long_short"
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    risk_column: str | None = None

    def __post_init__(self) -> None:
        if self.method not in WEIGHT_METHODS:
            raise ValueError(f"method must be one of {sorted(WEIGHT_METHODS)}.")
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
        if self.method == "inverse_vol_scaled_quantile" and not self.risk_column:
            raise ValueError("risk_column is required for inverse_vol_scaled_quantile.")
        _validate_optional_column_name(self.risk_column, "risk_column")


@dataclass(frozen=True)
class TradabilityConfig:
    """Market-agnostic eligibility and entry/exit constraint metadata."""

    eligibility_column: str | None = None
    can_enter_column: str | None = None
    can_exit_column: str | None = None
    blocked_entry_policy: str = "skip"
    blocked_exit_policy: str = "carry_forward"

    def __post_init__(self) -> None:
        _validate_optional_column_name(self.eligibility_column, "eligibility_column")
        _validate_optional_column_name(self.can_enter_column, "can_enter_column")
        _validate_optional_column_name(self.can_exit_column, "can_exit_column")
        if self.blocked_entry_policy not in BLOCKED_ENTRY_POLICIES:
            raise ValueError(
                f"blocked_entry_policy must be one of {sorted(BLOCKED_ENTRY_POLICIES)}."
            )
        if self.blocked_exit_policy not in BLOCKED_EXIT_POLICIES:
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
    impact_model: str | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative(self.buy_cost_bps, "buy_cost_bps")
        _validate_nonnegative(self.sell_cost_bps, "sell_cost_bps")
        _validate_nonnegative(self.slippage_bps, "slippage_bps")
        if self.spread_bps is not None:
            _validate_nonnegative(self.spread_bps, "spread_bps")
        _validate_optional_column_name(self.impact_model, "impact_model")


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio_id: str
    market_id: MarketId
    research_dataset_id: str
    signals: tuple[SignalSpec, ...]
    return_source_column: str
    frequency: str
    execution_lag_steps: int
    weight_config: WeightConfig = field(default_factory=WeightConfig)
    cost_config: CostConfig = field(default_factory=CostConfig)
    tradability_config: TradabilityConfig = field(default_factory=TradabilityConfig)
    run_type: str = "portfolio"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_portfolio_contract_version")
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
        if (
            not isinstance(self.return_source_column, str)
            or not self.return_source_column
        ):
            raise ValueError("portfolio_return_source_column_required")
        if not isinstance(self.frequency, str) or not self.frequency:
            raise ValueError("portfolio_frequency_required")
        if not isinstance(self.execution_lag_steps, int) or isinstance(
            self.execution_lag_steps,
            bool,
        ):
            raise TypeError("execution_lag_steps must be an integer")
        if self.execution_lag_steps < 0:
            raise ValueError("execution_lag_steps_must_be_nonnegative")
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
    if not 0 < float(value) <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1.")


def _validate_nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric.")
    if float(value) < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _validate_optional_column_name(value: str | None, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a non-empty string.")


def _cross_signal_id_column_overlaps(signals: tuple[SignalSpec, ...]) -> list[str]:
    overlaps = {
        signal.signal_id
        for index, signal in enumerate(signals)
        for other_index, other in enumerate(signals)
        if index != other_index and signal.signal_id == other.source_column
    }
    return sorted(overlaps)


__all__ = [
    "BLOCKED_ENTRY_POLICIES",
    "BLOCKED_EXIT_POLICIES",
    "CostConfig",
    "PORTFOLIO_MODES",
    "PortfolioSpec",
    "TradabilityConfig",
    "WEIGHT_METHODS",
    "WeightConfig",
]

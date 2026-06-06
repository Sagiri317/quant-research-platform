"""Compute-local runtime configuration objects."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class WeightConfig:
    """Portfolio weighting policy for compute runtimes."""

    method: str = "equal_weight_quantile"
    direction: str = "long_high"
    portfolio_mode: str = "long_short"
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    cost_bps: float = 0.0
    risk_column: str | None = None

    def __post_init__(self) -> None:
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
        if self.impact_model is not None and (
            not isinstance(self.impact_model, str) or not self.impact_model
        ):
            raise ValueError("impact_model must be a non-empty string.")


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


__all__ = [
    "BLOCKED_ENTRY_POLICIES",
    "BLOCKED_EXIT_POLICIES",
    "CostConfig",
    "PORTFOLIO_MODES",
    "TradabilityConfig",
    "WEIGHT_DIRECTIONS",
    "WEIGHT_METHODS",
    "WeightConfig",
]

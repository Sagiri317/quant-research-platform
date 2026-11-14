"""Causal decision points shared by learned policies and native accounts."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyDecision:
    """Information available before a target is submitted for subsequent execution.

    Features are indexed by asset ID. Quantities and sellable balances are shares;
    marks are currency per share, while cash and NAV are account currency amounts.
    Prediction inventory targets use physical_sellable before order reservations,
    after settlement and equity locks; cash plus reserved_cash is actual cash.
    The account consumes a target Series or dynamic PolicyExecution request sent
    back to the yielding iterator.
    """

    timestamp: pd.Timestamp
    features: pd.DataFrame
    quantities: pd.Series
    sellable: pd.Series
    marks: pd.Series
    cash: float
    nav: float
    opening_quantities: pd.Series | None = None
    pending_quantities: pd.Series | None = None
    pending_targets: pd.Series | None = None
    estimated_trade_cost_bps: pd.Series | None = None
    estimate_trade_cost: Callable[[str, float], float] | None = None
    reserved_cash: float = 0.0
    receivable_value: float = 0.0
    estimate_trade_cost_gradient: (
        Callable[[str, float, str], tuple[float, float]] | None
    ) = None
    order_state: pd.DataFrame | None = None
    physical_sellable: pd.Series | None = None


@dataclass(frozen=True)
class PolicyExecution:
    """Execution destinations and per-asset wait/cancel/maker/taker instructions.

    Destinations are shares, not a replacement portfolio mandate. Waiting preserves
    live orders; cancellation releases their reservations only on acknowledgement.
    """

    targets: pd.Series
    modes: pd.Series
    strategy_targets: pd.Series


def policy_target(value, assets, *, maximum=None, weights=False, allow_missing=False):
    """Validate an asset-indexed nonnegative target in shares or portfolio weights.

    The target must cover exactly the supplied assets. Weight totals cannot exceed
    one; share targets are checked against the optional per-asset inventory limits."""
    if not isinstance(value, pd.Series) or not value.index.is_unique:
        raise ValueError("policy_target_series_required")
    if set(value.index) != set(assets):
        raise ValueError("policy_target_assets_mismatch")
    if not pd.api.types.is_numeric_dtype(value) or pd.api.types.is_bool_dtype(value):
        raise ValueError("policy_target_numeric_required")
    result = value.reindex(assets).astype("float64")
    if (
        np.isinf(result).any()
        or (not allow_missing and result.isna().any())
        or (result < 0).any()
    ):
        raise ValueError("policy_target_invalid")
    if weights and result.sum() > 1.0 + 1e-12:
        raise ValueError("policy_target_budget_exceeded")
    if maximum is not None and (result > pd.Series(maximum).reindex(assets)).any():
        raise ValueError("policy_target_inventory_exceeded")
    return result

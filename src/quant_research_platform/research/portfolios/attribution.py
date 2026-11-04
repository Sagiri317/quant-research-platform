"""Holding-period attribution from realized portfolio and factor returns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.research.risk import FactorReturnFit


@dataclass(frozen=True)
class HoldingFactorInputs:
    decision_at: pd.Timestamp
    entry_at: pd.Timestamp
    end_at: pd.Timestamp
    fitted: FactorReturnFit | None
    decision_close: pd.Series
    entry_open: pd.Series
    benchmark_weights: pd.DataFrame
    reason: str | None = None
    closed_assets: frozenset[str] = frozenset()


def holding_attribution(
    inputs: HoldingFactorInputs,
    *,
    after_notionals: pd.Series,
    nav_before: float,
    nav_post: float,
    benchmark_id: str | None,
    benchmark_return: float | None,
    holding_returns: pd.Series,
) -> tuple[dict[str, float | None], dict[str, str], pd.Series]:
    """Ex-post relative-factor decomposition with an explicit replication gap."""
    components = (
        "benchmark",
        "industry",
        "style",
        "stock_residual_difference",
        "benchmark_replication_gap",
        "residual_unexplained",
    )
    values = dict.fromkeys(components)
    reasons = {}
    values["benchmark"] = 0.0 if benchmark_id is None else benchmark_return
    if nav_before <= 0 or nav_post <= 0:
        reason = "nonpositive_start_nav" if nav_before <= 0 else "nonpositive_post_nav"
        reasons.update({name: reason for name in components if name != "benchmark"})
        return values, reasons, pd.Series(dtype="float64")
    fitted = inputs.fitted
    if fitted is None:
        reasons.update(
            {
                name: inputs.reason or "holding_factor_fit_unidentified"
                for name in components
                if name != "benchmark"
            }
        )
        return values, reasons, pd.Series(dtype="float64")
    held = after_notionals.loc[after_notionals.ne(0.0)]
    missing = held.index.difference(fitted.exposures.index)
    if not missing.empty:
        prices = pd.concat(
            (
                inputs.decision_close.reindex(missing),
                inputs.entry_open.reindex(missing),
            ),
            axis=1,
        )
        known_closed = (
            set(missing).issubset(inputs.closed_assets)
            and np.isfinite(prices).all().all()
            and prices.gt(0).all().all()
            and np.isfinite(holding_returns.reindex(missing)).all()
            and holding_returns.reindex(missing).gt(-1).all()
        )
        reason = (
            "holding_closed_market_domain_unidentified"
            if known_closed
            else "holding_exposures_incomplete"
        )
        reasons.update({name: reason for name in components if name != "benchmark"})
        return values, reasons, pd.Series(dtype="float64")
    if not np.allclose(
        holding_returns.reindex(held.index),
        fitted.asset_returns.reindex(held.index),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("holding_return_ledger_mismatch")
    weights = held.div(nav_before).reindex(fitted.exposures.index, fill_value=0.0)
    realized = fitted.exposures.mul(
        held.div(nav_post).reindex(fitted.exposures.index, fill_value=0.0), axis=0
    ).sum()
    reference = pd.Series(0.0, index=fitted.exposures.index)
    if benchmark_id is not None:
        snapshot = inputs.benchmark_weights.loc[
            inputs.benchmark_weights.benchmark_id.eq(benchmark_id)
        ].set_index("asset_id")["weight"]
        if snapshot.index.has_duplicates:
            raise ValueError("attribution_benchmark_keys_invalid")
        complete = (
            not snapshot.empty
            and np.isfinite(snapshot).all()
            and snapshot.gt(0).all()
            and np.isclose(snapshot.sum(), 1.0, rtol=0, atol=1e-8)
            and set(snapshot.index).issubset(fitted.exposures.index)
        )
        prices = pd.concat(
            [
                inputs.decision_close.reindex(snapshot.index),
                inputs.entry_open.reindex(snapshot.index),
            ],
            axis=1,
        )
        if (
            not complete
            or prices.isna().any().any()
            or not np.isfinite(prices).all().all()
            or not prices.gt(0).all().all()
        ):
            reasons.update(
                {
                    name: "attribution_reference_basket_incomplete"
                    for name in components
                    if name != "benchmark"
                }
            )
            return values, reasons, realized
        drifted = snapshot * prices.iloc[:, 1] / prices.iloc[:, 0]
        reference = drifted.div(drifted.sum()).reindex(
            fitted.exposures.index, fill_value=0.0
        )
    relative = weights - reference
    contributions = fitted.exposures.mul(relative, axis=0).sum() * fitted.factor_returns
    values["industry"] = float(
        contributions.loc[contributions.index.str.startswith("industry_")].sum()
    )
    values["style"] = float(
        contributions.loc[~contributions.index.str.startswith("industry_")].sum()
    )
    values["stock_residual_difference"] = float(relative.dot(fitted.residuals))
    if values["benchmark"] is None:
        reasons["benchmark"] = reasons["benchmark_replication_gap"] = reasons[
            "residual_unexplained"
        ] = "benchmark_total_return_open_missing"
    else:
        values["benchmark_replication_gap"] = float(
            reference.dot(fitted.asset_returns) - values["benchmark"]
        )
        values["residual_unexplained"] = (
            values["stock_residual_difference"] + values["benchmark_replication_gap"]
        )
    return values, reasons, realized

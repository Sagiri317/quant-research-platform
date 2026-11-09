"""Stock targets reuse daily construction and the native absolute-risk solver."""

from dataclasses import asdict

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.portfolios import PORTFOLIO_CASH_ASSET_ID
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.optimizer import (
    AbsoluteRiskConfig,
    optimize_portfolio,
)


def portfolio_policy_parameters(value):
    parameters = _strict_config(
        {} if value is None else value,
        set(),
        {"construction", "top_quantile", "risk_limits", "trade_cost_bps"},
    )
    construction = parameters.setdefault("construction", "quantile_equal_weight")
    if construction not in {
        "quantile_equal_weight",
        "quantile_float_cap",
        "absolute_risk",
    }:
        raise ValueError("portfolio_policy_construction_invalid")
    if construction == "absolute_risk":
        if "top_quantile" in parameters:
            raise ValueError("portfolio_top_quantile_unused")
        parameters["risk_limits"] = asdict(
            AbsoluteRiskConfig(**parameters.get("risk_limits", {}))
        )
        if "trade_cost_bps" in parameters:
            cost = strict_finite_float(
                parameters["trade_cost_bps"], error="portfolio_trade_cost_invalid"
            )
            if not 0 <= cost < 10000:
                raise ValueError("portfolio_trade_cost_invalid")
            parameters["trade_cost_bps"] = cost
    else:
        if "risk_limits" in parameters or "trade_cost_bps" in parameters:
            raise ValueError("portfolio_risk_parameters_unused")
        quantile = strict_finite_float(
            parameters.get("top_quantile", 0.2), error="portfolio_quantile_invalid"
        )
        if not 0 < quantile <= 1:
            raise ValueError("portfolio_quantile_invalid")
        parameters["top_quantile"] = quantile
    return parameters


def portfolio_targets(decision, maximum, parameters, *, context=None):
    """Return share targets and optional target-risk facts, never account fills.

    A context contains an asset-indexed frame and its as_of, with optional
    covariance and industry/style columns. Its producer owns publication timing,
    lifecycle alignment and complete historical risk sessions. Checking as_of
    here does not prove those facts. Missing forecasts retain existing orders.
    """
    index = decision.features.index
    score = decision.features.prediction_score
    marks = decision.marks.reindex(index)
    quantities = decision.quantities.reindex(index)
    holding = (quantities * marks).where(quantities.ne(0), 0.0)
    if not np.isfinite(holding).all():
        raise ValueError("portfolio_held_mark_missing")
    valid = np.isfinite(score) & np.isfinite(marks) & marks.gt(0)
    target = pd.Series(np.nan, index=index, dtype="float64")
    if not valid.any() or decision.nav <= 0:
        return target, None
    frame = pd.DataFrame(index=index)
    covariance, industries, styles = None, (), ()
    if context is not None:
        if not isinstance(context, dict) or not {"as_of", "assets"} <= context.keys():
            raise ValueError("portfolio_context_invalid")
        as_of = pd.Timestamp(context["as_of"])
        if pd.isna(as_of) or as_of > decision.timestamp:
            raise ValueError("portfolio_context_after_decision")
        frame = context["assets"]
        if not isinstance(frame, pd.DataFrame) or not frame.index.is_unique:
            raise ValueError("portfolio_context_assets_invalid")
        frame = frame.reindex(index)
        covariance = context.get("factor_covariance")
        industries = tuple(context.get("industry_columns", ()))
        styles = tuple(context.get("style_columns", ()))
    # Cash is already net of the account's live-order reservations. Taker
    # intents have no such reservation, so retain cash for missing-score buys.
    pending = decision.pending_quantities
    pending = pd.Series(0.0, index=index) if pending is None else pending.reindex(index)
    missing_buys = float((pending.clip(lower=0) * marks).where(~valid, 0).sum())
    if decision.estimate_trade_cost is not None:
        missing_buys += sum(
            decision.estimate_trade_cost(asset, float(pending[asset]))
            for asset in index[~valid & pending.gt(0)]
        )
    extra_reserved = max(0.0, missing_buys - decision.reserved_cash)
    cash = max(0.0, decision.cash - extra_reserved)
    available = float(holding.sum()) + cash
    construction = parameters["construction"]
    maximum = pd.Series(maximum).reindex(index)
    if construction != "absolute_risk":
        weights = build_simple_target_weights(
            score,
            eligible=valid,
            method="float_cap_weight"
            if construction == "quantile_float_cap"
            else "equal_weight",
            top_quantile=parameters["top_quantile"],
            float_market_cap=frame.get("float_market_cap"),
        ).reindex(index)
        fixed = holding.where(~valid, 0.0)
        floor = (
            (quantities - decision.sellable.reindex(index)).clip(lower=0) * marks
        ).where(valid, 0.0)
        budget = max(0.0, available - float(fixed.sum()))
        desired = weights * budget
        increment = (desired - floor).clip(lower=0)
        residual = max(0.0, budget - float(floor.sum()))
        if increment.sum() > residual:
            increment *= residual / float(increment.sum())
        target.loc[valid] = ((floor + increment) / marks).loc[valid]
        return target.clip(upper=maximum), None

    limits = AbsoluteRiskConfig(**parameters["risk_limits"])
    assets = pd.DataFrame(
        {
            "asset_id": index,
            "expected_alpha": score.where(valid),
            "current_weight": holding / decision.nav,
            "sellable_weight": (decision.sellable.reindex(index) * marks).where(
                quantities.ne(0), 0.0
            )
            / decision.nav,
            "maximum_weight": (maximum * marks / decision.nav).clip(upper=1),
            "is_cash": False,
        },
        index=index,
    )
    for name in ("specific_variance", *industries, *styles):
        assets[name] = frame.get(name, np.nan)
    if "trade_capacity_weight" in frame:
        assets["trade_capacity_weight"] = frame.trade_capacity_weight
    assets.loc[PORTFOLIO_CASH_ASSET_ID] = {
        "asset_id": PORTFOLIO_CASH_ASSET_ID,
        "expected_alpha": 0.0,
        "current_weight": cash / decision.nav,
        "sellable_weight": cash / decision.nav,
        "maximum_weight": 1.0,
        "is_cash": True,
        **dict.fromkeys(("specific_variance", *industries, *styles), 0.0),
    }
    assets["is_cash"] = assets.is_cash.astype(bool)
    # Unquoted unheld assets have no executable target or fee evaluation.
    assets.loc[~valid.reindex(assets.index, fill_value=True), "maximum_weight"] = (
        assets.current_weight
    )

    def expected_cost(buys, sells):
        total = 0.0
        gradients = []
        for side, notionals in (("buy", buys), ("sell", sells)):
            gradient = np.zeros(len(assets))
            for position, asset in enumerate(index):
                if not valid[asset]:
                    continue
                if "trade_cost_bps" in parameters:
                    gradient[position] = parameters["trade_cost_bps"] / 10000
                    total += notionals[position] * gradient[position]
                else:
                    if decision.estimate_trade_cost_gradient is None:
                        raise ValueError("portfolio_native_cost_gradient_required")
                    cost, gradient[position] = decision.estimate_trade_cost_gradient(
                        asset, float(notionals[position]), side
                    )
                    total += cost
            gradients.append(gradient)
        return total, *gradients

    result = optimize_portfolio(
        assets.reset_index(drop=True),
        config=limits,
        nav=float(decision.nav),
        cash_column="is_cash",
        expected_transaction_cost=expected_cost,
        factor_covariance=covariance,
        industry_columns=industries,
        style_columns=styles,
        # The solver excludes unspendable NAV without pretending pending
        # orders are stock holdings. This includes receivables and reserved cash.
        receivable_weight=float(max(0.0, 1.0 - available / decision.nav)),
    )
    target.loc[valid] = (result.weights.reindex(index) * decision.nav / marks).loc[
        valid
    ]
    return target, result

"""Transparent inventory targets; execution and realized fees belong to the account."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.validation import strict_finite_float


def inventory_policy_parameters(value):
    parameters = _strict_config(
        {} if value is None else value,
        set(),
        {"trade_cost_bps", "exit_cost_bps", "inventory_risk_bps"},
    )
    parameters.setdefault("inventory_risk_bps", 0.0)
    for key in parameters:
        parameters[key] = strict_finite_float(
            parameters[key], error="inventory_policy_parameter_invalid"
        )
        if parameters[key] < 0:
            raise ValueError("inventory_policy_parameter_invalid")
    return parameters


def prediction_parameters(value):
    parameters = _strict_config(
        value,
        {"paths"},
        {"horizon_seconds", "decision_interval_seconds", "quote_visibility_delay_ms"},
    )
    paths = parameters["paths"]
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path for path in paths)
    ):
        raise ValueError("inventory_prediction_paths_required")
    for key, default in (
        ("horizon_seconds", 30),
        ("decision_interval_seconds", 3),
        ("quote_visibility_delay_ms", 1),
    ):
        parameters[key] = strict_finite_float(
            parameters.get(key, default), error="inventory_prediction_clock_invalid"
        )
        if parameters[key] < 0 or (
            key != "quote_visibility_delay_ms" and not parameters[key]
        ):
            raise ValueError("inventory_prediction_clock_invalid")
    return parameters


def validate_inventory_prediction(metadata, *, parameters):
    """Require native numerical return and decision-clock facts, without labels."""
    facts = metadata["model_facts"]
    definition = facts.get("target_definition") or {}
    semantics = definition.get("label_semantics") or {}
    if (
        facts["model"]["model_algorithm"] == "deeplob_classifier"
        or facts["label_unit"] != "decimal_return"
        or definition.get("target_transform") != "raw"
        or definition.get("target_clip") != "none"
        or semantics.get("family") != "future_mid_return"
        or semantics.get("horizon_unit") != "milliseconds"
        or semantics.get("horizon") != parameters["horizon_seconds"] * 1000
        or semantics.get("horizon_origin") != "common_decision_time"
        or semantics.get("decision_interval_seconds")
        != parameters["decision_interval_seconds"]
        or semantics.get("quote_visibility_delay_ms")
        != parameters["quote_visibility_delay_ms"]
    ):
        raise ValueError("inventory_prediction_return_or_clock_mismatch")


def inventory_targets(decision, maximum, parameters):
    """Compare quadratic stationary points and inventory knots with actual-size costs.

    Native fee estimates or explicit bps assumptions are not ledger charges.
    Live orders inform candidates and available budgets, never actual inventory.
    The account remains responsible for the shared cash budget and actual fills.
    """
    targets = pd.Series(np.nan, index=decision.features.index, dtype="float64")
    risk = parameters["inventory_risk_bps"]
    for asset, score in decision.features.prediction_score.items():
        price = decision.marks[asset]
        limit = maximum[asset]
        if pd.isna(score) or not np.isfinite(price) or price <= 0 or limit == 0:
            continue
        cost = parameters.get(
            "trade_cost_bps", decision.estimated_trade_cost_bps[asset]
        )
        exit_cost = parameters.get(
            "exit_cost_bps", decision.estimated_trade_cost_bps[asset]
        )
        if not np.isfinite(cost) or not np.isfinite(exit_cost):
            continue
        actual = decision.quantities[asset]
        opening = decision.opening_quantities[asset]
        pending = decision.pending_quantities[asset]
        committed = actual + pending
        lower = max(0, actual - decision.sellable[asset] + min(pending, 0))
        upper = max(
            lower, min(limit, actual + max(pending, 0) + max(0, decision.cash) / price)
        )
        # Proportional-cost stationary points plus knots form a transparent baseline.
        candidates = np.clip(
            [
                committed,
                actual,
                opening,
                lower,
                upper,
                *(
                    opening
                    + limit * (score * 10000 - cost * turn - exit_cost * side) / risk
                    for turn in (-1, 1)
                    for side in (-1, 1)
                    if risk > 0
                ),
            ],
            lower,
            upper,
        )
        displacement = candidates - opening

        def costs(delta, name, *, price=price, asset=asset):
            if name in parameters:
                return np.abs(delta) * price * parameters[name] / 10000
            return np.array(
                [decision.estimate_trade_cost(asset, value) for value in delta]
            )

        utility = (
            (score * displacement - risk / 10000 * displacement**2 / (2 * limit))
            * price
            - costs(candidates - actual, "trade_cost_bps")
            - costs(-displacement, "exit_cost_bps")
        )
        targets[asset] = candidates[np.argmax(utility)]
    return targets

"""Connect policy decisions to existing daily and quote accounting engines."""

from contextlib import contextmanager
from functools import partial

from quant_research_platform.contracts.json_codec import (
    _strict_config,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
)
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    strict_finite_float,
)
from quant_research_platform.policy.lifecycle import closing_iterator


def _episode(value):
    """Validate one bounded task, asset universe, and native account configuration."""
    config = _strict_config(
        value,
        {
            "kind",
            "task",
            "asset_ids",
            "feature_columns",
            "window",
            "account",
            "risk_aversion",
            "unfinished_penalty_bps",
        },
        {"order", "prediction", "capital_scale"},
    )
    assets, columns = tuple(config["asset_ids"]), tuple(config["feature_columns"])
    _validate_column_tuple(assets, "policy_assets")
    _validate_column_tuple(columns, "policy_features")
    if PORTFOLIO_CASH_ASSET_ID in assets or config["kind"] not in {"hft", "daily"}:
        raise ValueError("policy_episode_kind_invalid")
    if config["task"] not in {"execution", "inventory", "portfolio"}:
        raise ValueError("policy_task_invalid")
    if config["kind"] == "daily" and config["task"] != "portfolio":
        raise ValueError("daily_policy_requires_portfolio_task")
    window = TimeWindow(**config["window"])
    if window.end is None:
        raise ValueError("policy_episode_end_required")
    for name in ("risk_aversion", "unfinished_penalty_bps"):
        config[name] = strict_finite_float(config[name], error="policy_penalty_invalid")
        if config[name] < 0:
            raise ValueError("policy_penalty_invalid")
    if config["task"] == "execution":
        order = _strict_config(config.get("order"), {"asset_id", "quantity"}, set())
        if (
            order["asset_id"] not in assets
            or type(order["quantity"]) is not int
            or not order["quantity"]
        ):
            raise ValueError("policy_mother_order_invalid")
        config["order"] = order
    elif "order" in config:
        raise ValueError("policy_order_unused")
    elif config["unfinished_penalty_bps"] != 0:
        raise ValueError("policy_unfinished_penalty_unused")
    if "prediction" in config:
        from quant_research_platform.research.policies.inventory_policy import (
            prediction_parameters,
        )

        if (
            config["kind"] != "hft"
            or config["task"] not in {"inventory", "portfolio"}
            or columns != ("prediction_score",)
        ):
            raise ValueError("inventory_prediction_episode_invalid")
        config["prediction"] = prediction_parameters(config["prediction"])
        scale = strict_finite_float(
            config.get("capital_scale", 1), error="inventory_capital_scale_invalid"
        )
        if scale <= 0:
            raise ValueError("inventory_capital_scale_invalid")
        config["capital_scale"] = scale
        account = {
            **config["account"],
            "parameters": dict(config["account"]["parameters"]),
        }
        parameters = account["parameters"]
        capital = (
            1_000_000.0 * len(assets)
            if config["task"] == "inventory"
            else 1_000_000_000.0
        ) * scale
        if parameters.setdefault("initial_capital", capital) != capital:
            raise ValueError("inventory_capital_scale_mismatch")
        if config["task"] == "inventory":
            if parameters.setdefault("restore_at", "14:50:00") not in {
                "14:50",
                "14:50:00",
            }:
                raise ValueError("inventory_restore_clock_must_be_1450")
        elif parameters.setdefault("restore_at", None) is not None:
            raise ValueError("stock_portfolio_restoration_unused")
        config["account"] = account
    elif "capital_scale" in config:
        raise ValueError("inventory_capital_scale_unused")
    return config


@contextmanager
def _stream(episode):
    """Yield a closable account iterator that accepts targets at PolicyDecision
    events.
    """
    window = TimeWindow(**episode["window"])
    account = episode["account"]
    columns = tuple(episode["feature_columns"])
    if episode["kind"] == "hft":
        from quant_research_platform.research.portfolios.high_frequency import (
            _parameters,
            inventory_batches,
            inventory_inputs,
        )

        account = _strict_config(
            account,
            {"panel", "fee_schedule", "daily_inputs", "parameters"},
            {"trades", "stock_status", "candidate_id"},
        )
        prediction = episode.get("prediction")
        params = _parameters(account["parameters"], policy=True)
        if set(params["maximum_inventory"]) != set(episode["asset_ids"]) or (
            params["restore_at"] is not None and prediction is None
        ):
            raise ValueError("policy_inventory_or_restoration_conflict")
        from quant_research_platform.research.policies.inventory_policy import (
            validate_inventory_prediction,
        )

        sessions, schedule, trade_session, max_quote_gap_seconds = inventory_inputs(
            account,
            window,
            decision_columns=() if prediction is not None else columns,
            prediction_paths=prediction["paths"] if prediction is not None else (),
            prediction_validator=partial(
                validate_inventory_prediction, parameters=prediction
            )
            if prediction is not None
            else None,
        )
        with (
            closing_iterator(sessions) as inputs,
            closing_iterator(
                inventory_batches(
                    inputs,
                    schedule,
                    params,
                    window,
                    decision_columns=columns,
                    prediction_decisions=prediction is not None,
                    prediction_task=episode["task"],
                    quote_visibility_delay_ms=prediction["quote_visibility_delay_ms"]
                    if prediction is not None
                    else 0,
                    execution_order=episode.get("order"),
                    trade_session=trade_session,
                    max_quote_gap_seconds=max_quote_gap_seconds,
                    stock_status=account.get("stock_status"),
                )
            ) as stream,
        ):
            yield stream
    else:
        from quant_research_platform.research.portfolios.daily import backtest_a_share

        account = _strict_config(
            account,
            {
                "inputs",
                "candidate",
                "assumptions",
                "input_window",
                "formal_dataset_window",
            },
            set(),
        )
        assumptions = dict(account["assumptions"])
        assumptions["adv_sensitivities"] = tuple(assumptions["adv_sensitivities"])
        candidate_values = dict(account["candidate"])
        if "risk_style_columns" in candidate_values:
            candidate_values["risk_style_columns"] = tuple(
                candidate_values["risk_style_columns"]
            )
        candidate = PortfolioCandidateSpec(**candidate_values)
        with closing_iterator(
            backtest_a_share(
                candidate=candidate,
                inputs=account["inputs"],
                output_window=window,
                input_window=TimeWindow(**account["input_window"]),
                formal_dataset_window=TimeWindow(**account["formal_dataset_window"]),
                prediction_lookup=None,
                assumptions=ExecutionAssumptions(**assumptions),
                decision_columns=columns,
            )
        ) as stream:
            yield stream

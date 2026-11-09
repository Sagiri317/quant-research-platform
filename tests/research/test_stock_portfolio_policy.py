"""Stock allocation hand checks and native PRED-to-account replay."""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    write_prediction_outputs,
)
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.policies import PolicyDecision
from quant_research_platform.research.policies.accounts import _episode
from quant_research_platform.research.policies.portfolio_policy import (
    portfolio_policy_parameters,
    portfolio_targets,
)
from quant_research_platform.research.policies.reinforcement import backtest_policy
from quant_research_platform.research.protocols.a_share_hft import label_catalog
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.high_frequency import (
    parameters,
    quotes,
    schedule,
    write_daily_inputs,
)
from tests.helpers.output_frames import make_price_volume_prediction_output_frame
from tests.helpers.portfolio_inputs import _interim_specs


def decision(scores, quantities, sellable, *, cash=1000, pending=None):
    index = pd.Index(["a", "b", "c"])
    return PolicyDecision(
        pd.Timestamp("2024-01-02T01:30:03"),
        pd.DataFrame({"prediction_score": scores}, index=index),
        pd.Series(quantities, index=index, dtype=float),
        pd.Series(sellable, index=index, dtype=float),
        pd.Series(10.0, index=index),
        float(cash),
        float(cash + 10 * sum(quantities)),
        pending_quantities=pd.Series(pending or [0, 0, 0], index=index, dtype=float),
        estimate_trade_cost_gradient=lambda asset, amount, side: (0.0, 0.0),
    )


def test_simple_ranking_freezes_missing_capital_and_t1_before_redistribution():
    event = decision([np.nan, 0.03, 0.01], [100, 0, 100], [100, 0, 0])
    target, result = portfolio_targets(
        event, dict.fromkeys("abc", 1000), portfolio_policy_parameters({})
    )
    # 1000 missing-score holdings + 1000 unsellable holdings + 1000 free cash.
    assert np.isnan(target.a)
    assert target.b == pytest.approx(100)
    assert target.c == pytest.approx(100)
    assert result is None
    pending = replace(
        event, pending_quantities=pd.Series([50, 0, 0], index=list("abc"))
    )
    target, _ = portfolio_targets(
        pending, dict.fromkeys("abc", 1000), portfolio_policy_parameters({})
    )
    assert target.b == pytest.approx(50)
    reserved = replace(pending, cash=500.0, reserved_cash=500.0)
    target, _ = portfolio_targets(
        reserved, dict.fromkeys("abc", 1000), portfolio_policy_parameters({})
    )
    assert target.b == pytest.approx(50)  # The same reservation is not deducted twice.
    assert np.isnan(event.features.prediction_score.a)


def test_simple_float_cap_uses_explicit_asof_context_and_all_missing_needs_none():
    event = decision([0.03, 0.02, 0.01], [0, 0, 0], [0, 0, 0], cash=1000)
    params = portfolio_policy_parameters(
        {"construction": "quantile_float_cap", "top_quantile": 0.7}
    )
    context = {
        "as_of": "2024-01-01",
        "assets": pd.DataFrame({"float_market_cap": [1, 3, 100]}, index=list("abc")),
    }
    target, _ = portfolio_targets(
        event, dict.fromkeys("abc", 1000), params, context=context
    )
    assert target.tolist() == pytest.approx([25, 75, 0])
    with pytest.raises(ValueError, match="target_float_market_cap_required"):
        portfolio_targets(event, dict.fromkeys("abc", 1000), params)
    with pytest.raises(ValueError, match="portfolio_context_after_decision"):
        portfolio_targets(
            event,
            dict.fromkeys("abc", 1000),
            params,
            context={**context, "as_of": "2024-01-03"},
        )
    missing = replace(event, features=event.features * np.nan)
    target, _ = portfolio_targets(missing, dict.fromkeys("abc", 1000), params)
    assert target.isna().all()


def test_absolute_adapter_keeps_share_cap_cash_and_t1_inside_solver():
    event = decision([np.nan, 0.03, -0.01], [100, 0, 100], [100, 0, 40])
    params = portfolio_policy_parameters({"construction": "absolute_risk"})
    target, result = portfolio_targets(event, {"a": 1000, "b": 120, "c": 1000}, params)
    assert np.isnan(target.a)
    assert target.b == pytest.approx(120, abs=1e-6)
    assert target.c == pytest.approx(60, abs=1e-6)
    assert result.predicted_tracking_error is None
    assert result.predicted_total_volatility is None
    assert result.weights["__cash__"] == pytest.approx(200 / 3000)


def test_absolute_risk_is_optional_annualized_constraint_not_alpha_penalty():
    event = decision([0.03, 0.02, 0.01], [0, 0, 0], [0, 0, 0], cash=1000)
    context = {
        "as_of": "2024-01-01",
        "assets": pd.DataFrame(
            {"style": [1.0, 2.0, 3.0], "specific_variance": [0.0] * 3},
            index=list("abc"),
        ),
        "style_columns": ("style",),
        "factor_covariance": pd.DataFrame([[0.04]], index=["style"], columns=["style"]),
    }
    params = portfolio_policy_parameters(
        {
            "construction": "absolute_risk",
            "risk_limits": {"total_volatility_limit": 0.1},
        }
    )
    target, result = portfolio_targets(
        event, dict.fromkeys("abc", 1000), params, context=context
    )
    assert target.tolist() == pytest.approx([50, 0, 0], abs=1e-5)
    assert result.predicted_total_volatility == pytest.approx(0.1)
    with pytest.raises(ValueError, match="optimizer_covariance_unavailable"):
        portfolio_targets(event, dict.fromkeys("abc", 1000), params)
    assert event.features.prediction_score.tolist() == [0.03, 0.02, 0.01]


def write_replay(tmp_path):
    panel = pd.concat(
        [
            quotes(
                day,
                assets=("a", "b"),
                clocks=("01:30:00", "01:30:03.020", "01:30:06.020", "06:50:00.020"),
            )
            for day in ("2024-01-02", "2024-01-03")
        ],
        ignore_index=True,
    ).assign(instrument_type="stock")
    daily = write_daily_inputs(tmp_path / "daily", panel)
    roles = dict.fromkeys(panel.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    write_dataset(
        tmp_path / "panel",
        panel,
        spec=ProcessedDataSpec(
            data_id="hft_panel",
            market_id="a_share",
            kind="panel",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("timestamp", "asset_id"),
            partition_columns=(),
        ),
    )
    fees = pd.DataFrame([asdict(rule) for rule in schedule().rules])
    fees["effective_date"] = pd.to_datetime(fees.effective_date)
    fees["end_date"] = pd.to_datetime(fees.end_date)
    fees["included_in"] = fees.included_in.astype("string")
    write_dataset(
        tmp_path / "fees", fees, spec=_interim_specs().get("a_share_fee_schedule")
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02T01:30:03"] * 2
                + ["2024-01-02T01:30:06"] * 2
                + ["2024-01-03T01:30:03"] * 2
            ),
            "asset_id": ["a", "b"] * 3,
            "prediction_score": [0.03, 0.01, 0.01, 0.03, 0.01, 0.03],
            "prediction_rank_score": [0.5, 0.0, 0.0, 0.5, 0.0, 0.5],
        }
    )
    metadata = make_price_volume_prediction_output_frame().attrs["prediction_metadata"]
    facts = metadata["model_facts"]
    facts["model"]["target_column"] = "future_mid_return_30s"
    facts["target_definition"] = {
        "source_label": "future_mid_return_30s",
        "target_transform": "raw",
        "target_clip": "none",
        "label_semantics": {
            **label_catalog().get("future_mid_return_30s").to_manifest(),
            "horizon_origin": "common_decision_time",
            "decision_interval_seconds": 3,
            "quote_visibility_delay_ms": 1,
        },
    }
    metadata.update(
        prediction_window={"start": "2024-01-02", "end": "2024-01-03"},
        prediction_rows=len(frame),
        effective_prediction_end=frame.timestamp.max().isoformat(),
    )
    pred = write_prediction_outputs(
        output_path=tmp_path / "pred", output_chunks=[frame], metadata=metadata
    )
    params = parameters()
    for key in ("buy_threshold", "sell_threshold", "initial_capital", "restore_at"):
        del params[key]
    params.update(
        initial_inventory={"a": 0, "b": 0}, maximum_inventory={"a": 1000, "b": 1000}
    )
    return {
        "strategy": "stock_portfolio",
        "output": str(tmp_path / "result"),
        "episode": {
            "kind": "hft",
            "task": "portfolio",
            "asset_ids": ["a", "b"],
            "feature_columns": ["prediction_score"],
            "risk_aversion": 0.0,
            "unfinished_penalty_bps": 0.0,
            "capital_scale": 0.00001,
            "window": {"start": "2024-01-02", "end": "2024-01-03"},
            "prediction": {"paths": [str(pred)]},
            "account": {
                "panel": str(tmp_path / "panel"),
                "fee_schedule": str(tmp_path / "fees"),
                "daily_inputs": daily,
                "parameters": params,
            },
        },
    }


@pytest.mark.parametrize("construction", ["quantile_equal_weight", "absolute_risk"])
def test_native_stock_pred_cross_day_t1_no_restore_and_finite_capacity(
    tmp_path, construction
):
    config = write_replay(tmp_path)
    config["portfolio_parameters"] = {"construction": construction}
    root = backtest_policy(config)
    ledger = read_portfolio_ledger_artifact(root / "ledger")
    batches = list(ledger.iter_chunks())
    trades = [t for batch in batches for t in batch.trades]
    assert ledger.capital.initial_capital == 10000
    assert trades[0].asset_id == "a" and trades[0].quantity == 500
    assert all(abs(t.quantity) <= 500 for t in trades)
    sells = [t for t in trades if t.quantity < 0]
    assert sells and all(
        pd.Timestamp(t.timestamp_ns).date() == pd.Timestamp("2024-01-03").date()
        for t in sells
    )
    assert not any(pd.Timestamp(t.timestamp_ns).hour == 6 for t in trades)
    assert all(
        v.value >= 0
        for batch in batches
        for v in batch.valuations
        if v.asset_id == "__cash__"
    )
    facts = ledger.metadata["stock_portfolio_policy"]
    assert facts["benchmark"] is None
    assert facts["forecast_horizon_seconds"] == 30
    assert facts["decision_interval_seconds"] == 3
    assert facts["holding_period"] == "continuous_cross_session_holdings"
    default = {**config["episode"]}
    default.pop("capital_scale")
    assert _episode(default)["account"]["parameters"]["initial_capital"] == 1e9
    assert _episode(default)["account"]["parameters"]["restore_at"] is None

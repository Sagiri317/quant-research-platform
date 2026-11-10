"""Independent filter arithmetic and execution through the shared native account."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.policies.execution_schedules import (
    execution_fraction,
)
from quant_research_platform.research.policies.reinforcement import backtest_policy
from quant_research_platform.research.portfolios.high_frequency import (
    _parameters,
    inventory_batches,
)
from quant_research_platform.research.signals.processing import (
    process_prediction_signals,
    signal_processing_parameters,
)
from tests.helpers.high_frequency import daily_view, parameters, quotes, schedule
from tests.research.test_policy_learning import hft_episode


def processed(scores, stage, *, segments=None):
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=tuple(f"01:30:{i:02}" for i in range(len(scores))),
    )
    if segments is not None:
        panel["segment_id"] = segments
    predictions = panel[["timestamp", "asset_id"]].assign(prediction_score=scores)
    return process_prediction_signals(
        predictions, panel, signal_processing_parameters([stage])
    )


def test_filters_follow_hand_recursions_and_reset_at_segments():
    ewma = {"method": "ewma", "halflife_seconds": 1}
    result = processed([0.0, 2.0, 4.0, 10.0], ewma, segments=[0, 0, 0, 1])
    np.testing.assert_allclose(result.prediction_score, [0, 1, 2.5, 10])
    np.testing.assert_allclose(
        processed([0.0, 2.0], ewma).prediction_score, result.prediction_score.iloc[:2]
    )
    kalman = {
        "method": "kalman",
        "process_variance_per_second": 1,
        "measurement_variance": 1,
        "initial_variance": 0,
    }
    np.testing.assert_allclose(
        processed([0.0, 2.0, 4.0], kalman).prediction_score, [0, 1, 2.8]
    )
    for method, expected in (("rolling_mean", [11 / 3]), ("rolling_median", [2])):
        np.testing.assert_allclose(
            processed(
                [1.0, 8.0, 2.0], {"method": method, "window": 3}
            ).prediction_score,
            expected,
        )
    np.testing.assert_allclose(
        processed(
            [-0.6, -0.4, 0.4, 0.6], {"method": "threshold", "threshold": 0.5}
        ).prediction_score,
        [-0.6, 0, 0, 0.6],
    )
    np.testing.assert_allclose(
        processed(
            [-0.6, 0.4, 0.6], {"method": "deadband", "threshold": 0.5}
        ).prediction_score,
        [-0.1, 0, 0.1],
    )
    np.testing.assert_allclose(
        processed(
            [0.6, 1.2, 0.6, 0.2, -0.6, -1.2, -0.6],
            {"method": "hysteresis", "enter": 1, "exit": 0.3},
        ).prediction_score,
        [0, 1.2, 0.6, 0, 0, -1.2, -0.6],
    )


def test_filter_history_precedes_execution_window_and_does_not_mutate_predictions():
    panel = quotes("2024-01-02", assets=("a",))
    predictions = panel[["timestamp", "asset_id"]].assign(
        prediction_score=[-1.0, 3.0, -3.0]
    )
    config = parameters(("a",))
    config["signal_processing"] = [{"method": "rolling_mean", "window": 2}]
    config = _parameters(config)
    batches = list(
        inventory_batches(
            [(daily_view(panel), panel, predictions)],
            schedule(),
            config,
            TimeWindow("2024-01-02T01:30:03", "2024-01-02T01:30:07"),
        )
    )
    trades = [trade for batch in batches for trade in batch.trades]
    assert len(trades) == 1 and trades[0].quantity == 200
    assert trades[0].timestamp_ns == panel.timestamp.iloc[2].value
    assert predictions.prediction_score.tolist() == [-1, 3, -3]


def test_almgren_chriss_closed_form_limits_and_causal_signal_direction():
    config = {"risk_aversion": 1.0, "price_volatility": 1.0, "temporary_impact": 1.0}
    assert execution_fraction(0.3, 2, config) == pytest.approx(
        1 - np.sinh(1.4) / np.sinh(2)
    )
    assert execution_fraction(0.3, 2, {**config, "risk_aversion": 0}) == pytest.approx(
        0.3
    )
    assert execution_fraction(0.5, 10000, config) == 1
    for fraction in (0, 1):
        assert execution_fraction(fraction, 2, config) == fraction
    adjusted = {**config, "signal_gain": 0.2}
    base = execution_fraction(0.3, 2, config)
    assert execution_fraction(0.3, 2, adjusted, signal=1.0, side=1) > base
    assert execution_fraction(0.3, 2, adjusted, signal=1.0, side=-1) < base


def test_ac_and_signal_schedule_replay_native_account(tmp_path):
    episode = hft_episode(tmp_path / "inputs")
    episode["account"]["parameters"].update(
        initial_capital=10000.0,
        initial_inventory={"a": 500},
        maximum_inventory={"a": 1000},
    )
    episode["order"]["quantity"] = -400
    params = {
        "as_of": "2024-01-01",
        "risk_aversion": 0.0,
        "price_volatility": 1.0,
        "temporary_impact": 1.0,
    }
    summaries = {}
    for strategy in ("twap", "vwap", "almgren_chriss", "signal_ac"):
        config = {
            "episode": episode,
            "strategy": strategy,
            "output": str(tmp_path / strategy),
        }
        if strategy == "vwap":
            config["volume_curve"] = {
                "as_of": params["as_of"],
                "fractions": [0.0, 0.5, 1.0],
                "cumulative_volume": [0.0, 0.5, 1.0],
            }
        if strategy in {"almgren_chriss", "signal_ac"}:
            config["execution_parameters"] = dict(params)
        if strategy == "signal_ac":
            config["execution_parameters"].update(signal_column="mid", signal_gain=0.1)
        output = backtest_policy(config)
        summaries[strategy] = parse_qrp_json_object(
            (output / "summary.json").read_bytes()
        )
    assert summaries["vwap"] == summaries["almgren_chriss"] == summaries["twap"]
    assert summaries["twap"]["completion_rate"] == 0.5
    assert summaries["signal_ac"]["completion_rate"] < 0.5
    with pytest.raises(ValueError, match="not_historical"):
        backtest_policy(
            {
                "episode": episode,
                "strategy": "almgren_chriss",
                "execution_parameters": {**params, "as_of": episode["window"]["start"]},
                "output": str(tmp_path / "invalid"),
            }
        )


@pytest.mark.parametrize("strategy", ["vwap", "almgren_chriss"])
@pytest.mark.parametrize(
    "as_of", ["NaT", None, "2024-01-02T01:30:00", "2024-01-01T00:00:00Z"]
)
def test_execution_schedules_reject_unproved_history_before_writing(
    tmp_path, strategy, as_of
):
    episode = hft_episode(tmp_path / "inputs")
    config = {
        "episode": episode,
        "strategy": strategy,
        "output": str(tmp_path / "invalid"),
    }
    if strategy == "vwap":
        config["volume_curve"] = {
            "as_of": as_of,
            "fractions": [0.0, 1.0],
            "cumulative_volume": [0.0, 1.0],
        }
    else:
        config["execution_parameters"] = {
            "as_of": as_of,
            "risk_aversion": 0.0,
            "price_volatility": 1.0,
            "temporary_impact": 1.0,
        }
    with pytest.raises(ValueError, match="not_historical"):
        backtest_policy(config)
    assert not (tmp_path / "invalid").exists()


def test_off_quote_signals_use_visible_context_without_crossing_session():
    panel = quotes(
        "2024-01-02",
        assets=("a",),
        clocks=("01:30:00", "01:30:06", "05:00:00", "05:00:06"),
    )
    panel["asset_id"] = panel.asset_id.astype(object)
    panel["eligible"] = [True, False, True, True]
    panel["segment_id"] = [0, 0, 1, 1]
    predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02T01:30:03",
                    "2024-01-02T01:30:06",
                    "2024-01-02T04:59:59",
                    "2024-01-02T05:00:03",
                ]
            ),
            "asset_id": "a",
            "prediction_score": [1.0, 2.0, 3.0, 4.0],
        }
    )
    stage = signal_processing_parameters([{"method": "threshold", "threshold": 0.5}])
    result = process_prediction_signals(
        predictions, panel, stage, quote_visibility_delay_ms=1
    )
    assert result.prediction_score.tolist() == [1.0, 2.0, 4.0]
    assert result.timestamp.tolist() == predictions.timestamp.iloc[[0, 1, 3]].tolist()
    assert process_prediction_signals(predictions.iloc[[1]], panel, stage).empty
    # Filtering leaves the off-quote clock for the existing execution mapper.
    from quant_research_platform.research.portfolios.high_frequency import _decisions

    mapped = _decisions(panel, result, 0)
    assert (
        mapped[(predictions.timestamp.iloc[0].value, "a")][0]
        == panel.timestamp.iloc[1].value
    )

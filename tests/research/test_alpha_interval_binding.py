"""Frozen alpha binds only actual holiday-aware return interval semantics."""

import json

import pandas as pd
import pytest

from quant_research_platform.research.portfolios.accounting import (
    alpha_calibration_intervals,
    alpha_interval_semantics,
    frozen_alpha_calibration,
)
from quant_research_platform.research.portfolios.production import calibrate
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.portfolio_inputs import _interim_specs


def _calibration_inputs(tmp_path, schedule="weekly-last-trading-day", *, offset_days=0):
    shift = pd.Timedelta(days=offset_days)
    dates = pd.bdate_range("2023-01-02", "2023-04-03") - shift
    # Friday and Monday closures give real short/long weeks, not fixed seven-day rules.
    sessions = dates.difference(pd.to_datetime(["2023-01-13", "2023-01-16"]) - shift)
    intervals = alpha_calibration_intervals(sessions, schedule)
    observations = pd.DataFrame(
        [
            {
                "timestamp": day,
                "prediction_rank_score": score,
                "holding_return": score * 0.1 + i * 0.0001,
            }
            for i, day in enumerate(intervals.timestamp)
            for score in (-1.0, 0.0, 1.0)
        ]
    )
    observations.to_parquet(tmp_path / "observations.parquet", index=False)
    intervals.to_parquet(tmp_path / "intervals.parquet", index=False)
    calendar = pd.DataFrame(
        {
            "timestamp": dates,
            "calendar_date": dates,
            "source_market_type": 1,
            "is_trading_day": dates.isin(sessions),
        }
    )
    write_dataset(
        tmp_path / "calendar", calendar, spec=_interim_specs().get("trading_calendar")
    )
    return (
        {
            "observations": str(tmp_path / "observations.parquet"),
            "scheduled_intervals": str(tmp_path / "intervals.parquet"),
            "calendar": str(tmp_path / "calendar"),
            "rebalance_schedule": schedule,
            "cutoff": str((pd.Timestamp("2023-03-20") - shift).date()),
            "selection_group": "global",
            "output": str(tmp_path / "calibration.json"),
        },
        intervals,
        observations,
    )


def test_holiday_weekly_calibration_reusable_by_independent_compatible_consumers(
    tmp_path,
):
    config, intervals, observations = _calibration_inputs(tmp_path)
    assert (intervals.available_at - intervals.entry_at).dt.days.nunique() > 1
    payload = json.loads(calibrate(config).read_text())
    assert payload["interval_semantics"] == alpha_interval_semantics(
        config["rebalance_schedule"]
    )
    for _benchmark in ("csi500", "csi1000"):
        frozen, _ = frozen_alpha_calibration(
            payload,
            cutoff=config["cutoff"],
            selection_group="global",
            rebalance_schedule="weekly-last-trading-day",
        )
        assert frozen.slope == pytest.approx(0.1)
    with pytest.raises(ValueError, match="test_alpha_calibration_binding_invalid"):
        frozen_alpha_calibration(
            payload,
            cutoff=config["cutoff"],
            selection_group="global",
            rebalance_schedule="daily",
        )
    late = intervals.loc[intervals.available_at.gt(config["cutoff"]), "timestamp"]
    observations.loc[observations.timestamp.isin(late), "holding_return"] = 10000.0
    observations.to_parquet(config["observations"], index=False)
    changed = json.loads(
        calibrate({**config, "output": str(tmp_path / "late.json")}).read_text()
    )
    assert changed["calibration"] == payload["calibration"]


def test_direct_weekly_claim_rejects_daily_or_shifted_interval_samples(tmp_path):
    config, intervals, _ = _calibration_inputs(tmp_path, "daily")
    with pytest.raises(
        ValueError, match="alpha_calibration_interval_semantics_mismatch"
    ):
        calibrate({**config, "rebalance_schedule": "weekly-last-trading-day"})
    assert not (tmp_path / "calibration.json").exists()
    intervals.loc[0, "entry_at"] += pd.Timedelta(hours=1)
    intervals.to_parquet(config["scheduled_intervals"], index=False)
    with pytest.raises(
        ValueError, match="alpha_calibration_interval_semantics_mismatch"
    ):
        calibrate(config)


@pytest.mark.parametrize("retain_undeclared_observations", [False, True])
def test_missing_declared_intervals_do_not_shrink_calendar_coverage(
    tmp_path, retain_undeclared_observations
):
    config, intervals, observations = _calibration_inputs(tmp_path)
    mature = intervals.loc[intervals.available_at.le(config["cutoff"])]
    supplied = mature.iloc[:6]
    supplied.to_parquet(config["scheduled_intervals"], index=False)
    if not retain_undeclared_observations:
        observations = observations.loc[observations.timestamp.isin(supplied.timestamp)]
        observations.to_parquet(config["observations"], index=False)
    result = json.loads(calibrate(config).read_text())["calibration"]
    assert len(mature) == 10
    assert result["scheduled_rebalance_dates"] == len(mature)
    assert result["valid_rebalance_dates"] == 6
    assert result["coverage"] == pytest.approx(0.6)
    assert result["undefined_reason"] == "alpha_calibration_insufficient_coverage"


def test_duplicate_declared_intervals_remain_invalid(tmp_path):
    config, intervals, _ = _calibration_inputs(tmp_path)
    pd.concat((intervals, intervals.iloc[:1]), ignore_index=True).to_parquet(
        config["scheduled_intervals"], index=False
    )
    with pytest.raises(ValueError, match="alpha_calibration_schedule_invalid"):
        calibrate(config)


def test_native_independent_consumers_share_schedule_but_reject_other_horizon(tmp_path):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
    )
    from quant_research_platform.research.portfolios.production import backtest
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.direct_model_test_support import fit_frame
    from tests.research.test_closed_market_portfolios import _native_inputs

    calibration_dir = tmp_path / "alpha"
    calibration_dir.mkdir()
    calibration_config, _, _ = _calibration_inputs(calibration_dir, offset_days=112)
    calibration = calibrate(calibration_config)
    inputs, _, _, panel = _native_inputs(tmp_path / "market", fault="normal_open")
    training = read_dataset(inputs["dataset"])
    training = training.loc[training.timestamp.le("2022-12-20")].copy()
    training["signal"] = training.asset_id.str[1:].astype(float)
    training["label"] = training.signal * 0.001
    training["label_available_at"] = training.timestamp
    fitted = fit_frame(
        tmp_path / "learning",
        training,
        ModelSpec(
            model_id="ridge",
            model_algorithm="ridge_regression",
            input_factor_columns=("signal",),
            target_column="label",
            random_seed=42,
        ),
        market_id="a_share",
    )
    window = {"start": "2022-12-30", "end": "2023-01-06"}
    prediction = predict(
        {
            "model": str(fitted.root),
            "inputs": str(inputs["dataset"]),
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    benchmarks = pd.concat(
        [
            panel[["market_id", "asset_id", "timestamp"]].assign(
                benchmark_id=name, weight=1 / 12
            )
            for name in ("csi500", "csi1000")
        ],
        ignore_index=True,
    )
    benchmark = tmp_path / "benchmarks"
    write_dataset(
        benchmark, benchmarks, spec=catalog.processed_catalog().get("benchmark_weights")
    )
    inputs["benchmark"] = benchmark
    inputs["risk_context"] = [inputs["dataset"]]
    config = {
        "candidate": {
            "candidate_id": "weekly_csi500",
            "selection_group": "global",
            "construction": "benchmark_enhanced",
            "prediction_selector": "ridge",
            "rebalance_schedule": "weekly-last-trading-day",
            "top_quantile": None,
            "benchmark_id": "csi500",
            "risk_style_columns": ["signal"],
        },
        "inputs": {
            key: [str(path) for path in value] if key == "risk_context" else str(value)
            for key, value in inputs.items()
        },
        "predictions": [{"path": str(prediction), "window": window}],
        "output": str(tmp_path / "ledger-csi500"),
        "output_window": window,
        "input_window": {"start": "2022-12-01", "end": "2023-01-06"},
        "formal_dataset_window": {"start": "2022-12-01", "end": "2023-01-06"},
        "prediction_window": window,
        "calibration": str(calibration),
        "assumptions": {
            "initial_nav": 100.0,
            "slippage_bps": 0.0,
            "impact_bps": 0.0,
            "adv_participation": 1.0,
            "adv_sensitivities": [1.0],
        },
    }
    for name in ("csi500", "csi1000"):
        consumer = {
            **config,
            "candidate": {
                **config["candidate"],
                "candidate_id": "weekly_" + name,
                "benchmark_id": name,
            },
            "output": str(tmp_path / ("ledger-" + name)),
        }
        ledger = read_portfolio_ledger_artifact(backtest(consumer))
        slopes = [
            row.number
            for batch in ledger.iter_chunks()
            for row in batch.diagnostics
            if row.name == "alpha_slope"
        ]
        assert slopes and slopes == pytest.approx([0.1] * len(slopes))
    with pytest.raises(ValueError, match="test_alpha_calibration_binding_invalid"):
        backtest(
            {
                **config,
                "candidate": {**config["candidate"], "rebalance_schedule": "daily"},
                "output": str(tmp_path / "daily"),
            }
        )
    assert not (tmp_path / "daily").exists()

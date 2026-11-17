"""Return calibration preserves causal session windows and native account use."""

import json
import pickle
from copy import deepcopy
from pathlib import Path

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
from quant_research_platform.research.policies.reinforcement import (
    backtest_policy,
    train_policy,
)
from quant_research_platform.research.predictions.amplitude import (
    apply_amplitude,
    calibrate_amplitude,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.research.test_dynamic_policy_execution import execution_episode, replay
from tests.research.test_etf_inventory_policy import forecast_metadata

LABEL = "future_mid_return_30s"


def write_labels(path, frame, *, market="a_share"):
    roles = dict.fromkeys(frame, "context")
    roles.update(timestamp="timestamp", asset_id="key")
    if "eligible" in frame:
        roles["eligible"] = "mask"
    roles[LABEL] = "label"
    spec = ProcessedDataSpec(
        data_id="calibration_labels",
        market_id=market,
        kind="panel",
        frequency="tick",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles=roles,
        column_units={LABEL: "decimal_return"},
        label_availability_columns={LABEL: "available_at"},
    )
    write_dataset(
        path,
        frame,
        spec=spec,
        metadata={
            "producer_diagnostics": {
                "label_semantics": {
                    LABEL: forecast_metadata()["model_facts"]["target_definition"][
                        "label_semantics"
                    ]
                }
            }
        },
    )
    return str(path)


def write_predictions(path, frame):
    metadata = forecast_metadata()
    metadata["prediction_window"] = {
        "start": frame.timestamp.min().date().isoformat(),
        "end": frame.timestamp.max().date().isoformat(),
    }
    write_prediction_outputs(
        output_path=path,
        output_chunks=[frame.assign(prediction_rank_score=0.0)],
        metadata=metadata,
    )
    return str(path), metadata


def observations(day, x, y, *, asset="a"):
    times = (
        pd.Timestamp(day)
        + pd.Timedelta(hours=1, minutes=30, milliseconds=1)
        + pd.to_timedelta(np.arange(len(x)) * 3, unit="s")
    )
    return pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": asset,
            "prediction_score": x,
            LABEL: y,
            "available_at": times + pd.Timedelta(seconds=30),
            "eligible": True,
        }
    )


def test_daily_ols_uses_fixed_session_axis_and_strictly_mature_pairs(tmp_path):
    bp = 1e-4
    first = observations(
        "2024-01-02", np.array([-1, 0, 1]) * bp, np.array([0, -1, 4]) * bp
    )
    constant = observations("2024-01-02", [0.1] * 3, [0.001, 0.002, 0.003], asset="b")
    delayed = observations("2024-01-04", [-bp, bp, 2 * bp], [0.01, -0.01, 0.9])
    delayed["available_at"] = [
        pd.Timestamp("2024-01-05T01:30"),
        pd.Timestamp("2024-01-05T01:30"),
        pd.NaT,
    ]
    current = observations("2024-01-05", [-bp, bp], [0.02, -0.02])
    frame = pd.concat([first, constant, delayed, current], ignore_index=True)
    predictions, metadata = write_predictions(
        tmp_path / "predictions", frame[["timestamp", "asset_id", "prediction_score"]]
    )
    config = {
        "inputs": write_labels(
            tmp_path / "labels", frame.drop(columns="prediction_score")
        ),
        "predictions": [predictions],
        "label": LABEL,
        "asset_ids": ["a", "b"],
        "sessions": [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ],
        "window": {"start": "2024-01-03", "end": "2024-01-08"},
        "lookback_sessions": 2,
        "output": str(tmp_path / "calibration"),
    }
    table = read_dataset(calibrate_amplitude(config)).set_index(
        ["timestamp", "asset_id"]
    )
    row = table.loc[(pd.Timestamp("2024-01-04"), "a")]
    # x=(-1,0,1), y=(0,-1,4) bp: intercept=1, slope=2, residual=(1,-2,1).
    assert row.intercept == pytest.approx(bp)
    assert row.slope == pytest.approx(2)
    assert row.prediction_error_scale == pytest.approx(np.sqrt(2) * bp)
    assert row.history_rows == 3 and row.history_sessions == 1
    assert (
        table.loc[(pd.Timestamp("2024-01-03"), "a"), "reason"]
        == "insufficient_sessions"
    )
    assert (
        table.loc[(pd.Timestamp("2024-01-04"), "b"), "reason"] == "constant_prediction"
    )
    # Jan 3 remains on the axis; Jan 2 cannot fill the empty Jan 3/4 window.
    assert table.loc[(pd.Timestamp("2024-01-05"), "a"), "history_rows"] == 0
    assert (
        table.loc[(pd.Timestamp("2024-01-05"), "a"), "reason"] == "insufficient_pairs"
    )
    assert table.loc[(pd.Timestamp("2024-01-08"), "a"), "history_rows"] == 4
    assert table.loc[(pd.Timestamp("2024-01-08"), "a"), "slope"] == pytest.approx(-150)
    undefined = table.loc[
        ~table.eligible, ["intercept", "slope", "prediction_error_scale"]
    ]
    assert undefined.isna().all().all()
    prefix = read_dataset(
        calibrate_amplitude(
            {
                **config,
                "window": {"start": "2024-01-04", "end": "2024-01-04"},
                "output": str(tmp_path / "prefix"),
            }
        )
    )
    pd.testing.assert_frame_equal(
        prefix.set_index(["timestamp", "asset_id"]),
        table.loc[[pd.Timestamp("2024-01-04")], :],
    )
    changed = frame.drop(columns="prediction_score").copy()
    changed.loc[changed.timestamp.dt.day.eq(5), LABEL] *= 2
    changed_config = {
        **config,
        "inputs": write_labels(tmp_path / "changed-labels", changed),
        "output": str(tmp_path / "changed"),
    }
    changed_table = read_dataset(calibrate_amplitude(changed_config)).set_index(
        ["timestamp", "asset_id"]
    )
    pd.testing.assert_frame_equal(
        changed_table.loc[:"2024-01-05"], table.loc[:"2024-01-05"]
    )
    assert (
        changed_table.loc[(pd.Timestamp("2024-01-08"), "a"), "slope"]
        != table.loc[(pd.Timestamp("2024-01-08"), "a"), "slope"]
    )
    # Applying the saved coefficients has no dependency on labels.
    Path(config["inputs"]).rename(tmp_path / "unavailable-labels")
    new = observations("2024-01-04", [2 * bp], [999])[
        ["timestamp", "asset_id", "prediction_score"]
    ]
    calibrated = apply_amplitude(new, [metadata], config["output"])
    assert calibrated.raw_prediction_score.iloc[0] == 2 * bp
    assert calibrated.prediction_score.iloc[0] == pytest.approx(5 * bp)
    missing = new.assign(timestamp=pd.Timestamp("2024-01-09T01:30"))
    with pytest.raises(ValueError, match="calibration_session_missing"):
        apply_amplitude(missing, [metadata], config["output"])
    before_open = new.assign(timestamp=pd.Timestamp("2024-01-04T01:29"))
    with pytest.raises(ValueError, match="not_available_at_decision"):
        apply_amplitude(before_open, [metadata], config["output"])
    mismatched = deepcopy(metadata)
    mismatched["model_facts"]["model"]["model_algorithm"] = "xgboost"
    with pytest.raises(ValueError, match="prediction_family_mismatch"):
        apply_amplitude(new, [mismatched], config["output"])


@pytest.mark.parametrize("eligible", [False, None])
def test_calibration_pairs_forecasts_independently_of_research_eligibility(
    tmp_path, eligible
):
    frame = observations("2024-01-02", [-0.001, 0.001], [-0.002, 0.002])
    predictions, _ = write_predictions(
        tmp_path / "predictions", frame[["timestamp", "asset_id", "prediction_score"]]
    )
    labels = frame.drop(columns="prediction_score")
    labels = (
        labels.drop(columns="eligible")
        if eligible is None
        else labels.assign(eligible=eligible)
    )
    result = calibrate_amplitude(
        {
            "inputs": write_labels(tmp_path / "labels", labels),
            "predictions": [predictions],
            "label": LABEL,
            "asset_ids": ["a"],
            "sessions": ["2024-01-02", "2024-01-03"],
            "window": {"start": "2024-01-03", "end": "2024-01-03"},
            "lookback_sessions": 1,
            "output": str(tmp_path / "calibration"),
        }
    )
    row = read_dataset(result).iloc[0]
    assert row.history_rows == 2 and row.eligible
    assert row.intercept == pytest.approx(0)
    assert row.slope == pytest.approx(2)


def test_calibration_rejects_conflicting_label_market_and_overlapping_forecasts(
    tmp_path,
):
    frame = observations("2024-01-02", [-0.001, 0.001], [-0.002, 0.002])
    predictions, _ = write_predictions(
        tmp_path / "predictions", frame[["timestamp", "asset_id", "prediction_score"]]
    )
    config = {
        "inputs": write_labels(
            tmp_path / "labels", frame.drop(columns="prediction_score"), market="crypto"
        ),
        "predictions": [predictions],
        "label": LABEL,
        "asset_ids": ["a"],
        "sessions": ["2024-01-02", "2024-01-03"],
        "window": {"start": "2024-01-03", "end": "2024-01-03"},
        "lookback_sessions": 1,
        "output": str(tmp_path / "result"),
    }
    with pytest.raises(ValueError, match="input_market_mismatch"):
        calibrate_amplitude(config)
    with pytest.raises(ValueError, match="prediction_keys_overlap"):
        calibrate_amplitude({**config, "predictions": [predictions, predictions]})
    assert not Path(config["output"]).exists()


def test_calibrated_direct_and_real_ppo_share_saved_signal_and_frozen_scaling(tmp_path):
    scores = np.array([-0.02, -0.01, 0.01, 0.02])
    episode = execution_episode(tmp_path / "account", scores=tuple(scores))
    frames = [
        observations(day, scores, 0.25 * scores + np.array([1, -1, -1, 1]) * 1e-4)
        for day in ("2023-12-29", "2024-01-02", "2024-01-03")
    ]
    history, _ = write_predictions(
        tmp_path / "history", frames[0][["timestamp", "asset_id", "prediction_score"]]
    )
    labels = write_labels(
        tmp_path / "labels",
        pd.concat(frames, ignore_index=True).drop(columns="prediction_score"),
    )
    config = {
        "inputs": labels,
        "predictions": [history, *episode["prediction"]["paths"]],
        "label": LABEL,
        "asset_ids": ["a"],
        "sessions": ["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"],
        "window": {"start": "2024-01-02", "end": "2024-01-04"},
        "lookback_sessions": 1,
        "output": str(tmp_path / "calibration"),
    }
    episode["prediction"]["calibration"] = str(calibrate_amplitude(config))
    Path(labels).rename(tmp_path / "offline-labels")
    decisions, batches, *_ = replay(episode, [[7]] * 4)
    np.testing.assert_allclose(
        [d.features.loc["a", "prediction_score"] for d in decisions],
        scores * 0.25,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [d.features.loc["a", "prediction_error_scale"] for d in decisions],
        1e-4,
        atol=1e-12,
    )
    direct = backtest_policy(
        {
            "episode": episode,
            "strategy": "immediate",
            "output": str(tmp_path / "direct"),
        }
    )
    native_batches = list(
        read_portfolio_ledger_artifact(direct / "ledger").iter_chunks()
    )
    assert [t for b in native_batches for t in b.trades] == [
        t for b in batches for t in b.trades
    ]
    diagnostics = [d for b in native_batches for d in b.diagnostics]
    assert any(d.name == "raw_prediction_score" for d in diagnostics)
    assert any(d.name == "prediction_error_scale" for d in diagnostics)
    validation = {
        **episode,
        "window": {"start": "2024-01-03T01:30:00", "end": "2024-01-03T01:30:12"},
    }
    model = train_policy(
        {
            "episodes": [episode],
            "validation_episodes": [validation],
            "fit_cutoff": "2024-01-03T01:30:12",
            "output": str(tmp_path / "policy"),
            "timesteps": 4,
            "seed": 1,
            "device": "cpu",
            "n_envs": 1,
            "threads": 1,
            "early_stopping": {"eval_freq": 4, "patience": 1, "min_evals": 1},
            "ppo": {
                "n_steps": 4,
                "batch_size": 4,
                "n_epochs": 1,
                "policy_kwargs": {"net_arch": [8]},
            },
        }
    )
    facts = json.loads((model / "policy.json").read_text())
    assert (
        facts["config"]["episodes"][0]["feature_columns"][-1]
        == "prediction_error_scale"
    )
    scaler = pickle.loads((model / "feature_scaler.pkl").read_bytes())
    assert scaler.medians[-1] == pytest.approx(1e-4)
    later = {
        **episode,
        "window": {"start": "2024-01-04T01:30:00", "end": "2024-01-04T01:30:12"},
    }
    request = {
        "episode": later,
        "strategy": "ppo",
        "policy": str(model),
        "output": str(tmp_path / "ppo"),
    }
    output = backtest_policy(request)
    assert list(read_portfolio_ledger_artifact(output / "ledger").iter_chunks())
    uncalibrated = deepcopy(later)
    del uncalibrated["prediction"]["calibration"]
    with pytest.raises(ValueError, match="spaces_mismatch"):
        backtest_policy(
            {**request, "episode": uncalibrated, "output": str(tmp_path / "wrong-mode")}
        )

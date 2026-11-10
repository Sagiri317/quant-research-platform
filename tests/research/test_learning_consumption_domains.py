"""Learners establish their actual asset or paired-key domain before admission."""

import pickle
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.policies.cost_aware import fit_cost_model
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.predictions.uncertainty import (
    apply_uncertainty,
    calibrate_uncertainty,
    evaluate_uncertainty,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.learning import learning_panel
from tests.research.direct_model_test_support import fit_frame


@pytest.mark.parametrize("selected_issue", [None, "duplicate", "immature", "missing"])
def test_cost_learning_consumes_its_declared_asset_universe(tmp_path, selected_issue):
    times = pd.date_range("2020-01-01", periods=3)
    frame = pd.DataFrame(
        {
            "timestamp": times.repeat(2),
            "asset_id": ["a", "b"] * 3,
            "eligible": True,
            "x": [1.0, 3.0] * 3,
            "label": [0.001, 0.002] * 3,
            "available_at": (times + pd.Timedelta(days=1)).repeat(2),
            "return_end_at": (times + pd.Timedelta(days=1)).repeat(2),
            "observation": 0,
        }
    )
    spec = DataSpec(
        data_id="cost_learning",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id", "observation"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "observation": "key",
            "eligible": "mask",
            "x": "factor",
            "label": "label",
            "available_at": "context",
            "return_end_at": "context",
        },
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "available_at"},
    )
    write_dataset(tmp_path / "baseline", frame, spec=spec)
    outside = frame.loc[frame.asset_id.eq("a")].assign(asset_id="outside", x=1000.0)
    extended = pd.concat(
        [frame, outside, outside.assign(observation=1)], ignore_index=True
    )
    if selected_issue == "duplicate":
        extended = pd.concat(
            [extended, frame.iloc[[0]].assign(observation=1)], ignore_index=True
        )
    elif selected_issue == "immature":
        extended.loc[
            extended.asset_id.eq("a") & extended.timestamp.eq(times[1]), "available_at"
        ] = times[-1] + pd.Timedelta(days=2)
    elif selected_issue == "missing":
        extended = extended.loc[
            ~(extended.asset_id.eq("a") & extended.timestamp.eq(times[1]))
        ]
    write_dataset(tmp_path / "extended", extended, spec=spec)
    config = {
        "dataset": str(tmp_path / "extended"),
        "feature_columns": ["x"],
        "asset_ids": ["a", "b"],
        "label": "label",
        "return_end_column": "return_end_at",
        "window": {"start": "2020-01-01", "end": "2020-01-03"},
        "cutoff": "2020-01-04",
        "cost_bps": 10.0,
        "risk_aversion": 1.0,
        "epochs": 1,
        "learning_rate": 0.01,
        "hidden_sizes": [2],
        "device": "cpu",
        "threads": 1,
        "seed": 1,
        "output": str(tmp_path / "extended-model"),
    }
    if selected_issue:
        error = {
            "duplicate": "research_panel_key_invalid",
            "immature": "cost_learning_complete_mature_panel_required",
            "missing": "cost_learning_fixed_universe_required",
        }[selected_issue]
        with pytest.raises(ValueError, match=error):
            fit_cost_model(config)
        return
    baseline = fit_cost_model(
        {
            **config,
            "dataset": str(tmp_path / "baseline"),
            "output": str(tmp_path / "baseline-model"),
        }
    )
    actual = fit_cost_model(config)
    expected_facts = parse_qrp_json_object((baseline / "model.json").read_bytes())
    actual_facts = parse_qrp_json_object((actual / "model.json").read_bytes())
    for key in ("periods", "train_mean_net_return", "train_mean_turnover"):
        assert actual_facts[key] == expected_facts[key]
    assert pickle.loads((actual / "feature_scaler.pkl").read_bytes()).medians == (2.0,)


@pytest.mark.parametrize("stage", ["calibration", "evaluation"])
@pytest.mark.parametrize("duplicate_pair", [False, True])
def test_uncertainty_requires_unique_labels_only_for_actual_prediction_keys(
    tmp_path, stage, duplicate_pair
):
    frame, spec = learning_panel(tmp_path / "base")
    case = fit_frame(
        tmp_path / "fit",
        frame.loc[frame.timestamp.lt("2020-01-04")],
        ModelSpec(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
        ),
        cutoff=pd.Timestamp("2020-01-04"),
    )
    labels = frame.copy()
    labels["label"] += labels.timestamp.dt.day.map(
        {5: 0.01, 6: 0.02, 7: 0.03, 8: 0.04}
    ).fillna(0.02)
    write_dataset(tmp_path / "labels", labels, spec=spec)
    future = frame.loc[~(frame.asset_id.eq("a") & frame.timestamp.dt.day.isin([6, 11]))]
    write_dataset(tmp_path / "prediction-input", future, spec=spec)
    windows = {
        "calibration": {"start": "2020-01-05", "end": "2020-01-08"},
        "evaluation": {"start": "2020-01-10", "end": "2020-01-12"},
    }
    predictions = {}
    for name, window in windows.items():
        predictions[name] = predict(
            {
                "inputs": str(tmp_path / "prediction-input"),
                "model": str(case.root),
                "output": str(tmp_path / f"pred-{name}"),
                "prediction_window": window,
                "asset_ids": ["a"],
            }
        )
    calibration_config = {
        "dataset": str(tmp_path / "labels"),
        "predictions": str(predictions["calibration"]),
        "window": windows["calibration"],
        "label": "label",
        "cutoff": "2020-01-09",
        "alpha": 0.25,
        "unit": "decimal_return",
        "block": "timestamp",
        "output": str(tmp_path / "calibration.json"),
    }
    calibration = calibrate_uncertainty(calibration_config)
    intervals = apply_uncertainty(
        {
            "calibration": str(calibration),
            "predictions": str(predictions["evaluation"]),
            "window": windows["evaluation"],
            "cost": 0.0,
            "output": str(tmp_path / "intervals"),
        }
    )
    evaluation_config = {
        "dataset": str(tmp_path / "labels"),
        "intervals": str(intervals),
        "window": windows["evaluation"],
        "label": "label",
        "cutoff": "2020-01-13",
        "output": str(tmp_path / "evaluation.json"),
    }
    evaluation = evaluate_uncertainty(evaluation_config)
    extra = labels.loc[
        labels.asset_id.eq("c")
        | (labels.asset_id.eq("a") & labels.timestamp.dt.day.isin([6, 11]))
    ]
    if duplicate_pair:
        extra = labels.loc[
            labels.asset_id.eq("a") & labels.timestamp.dt.day.isin([5, 10])
        ]
    extended = pd.concat(
        [labels.assign(observation=0), extra.assign(observation=1)], ignore_index=True
    )
    write_dataset(
        tmp_path / "extended-labels",
        extended,
        spec=replace(
            spec,
            primary_key=(*spec.primary_key, "observation"),
            column_roles={**spec.column_roles, "observation": "key"},
        ),
    )
    function, config, baseline = (
        (calibrate_uncertainty, calibration_config, calibration)
        if stage == "calibration"
        else (evaluate_uncertainty, evaluation_config, evaluation)
    )
    config = {
        **config,
        "dataset": str(tmp_path / "extended-labels"),
        "output": str(tmp_path / f"extended-{stage}.json"),
    }
    if duplicate_pair:
        with pytest.raises(ValueError, match="research_panel_key_invalid"):
            function(config)
    else:
        actual = parse_qrp_json_object(function(config).read_bytes())
        expected = parse_qrp_json_object(baseline.read_bytes())
        metrics = (
            ("radius", "calibration_rows", "calibration_blocks")
            if stage == "calibration"
            else ("rows", "coverage", "mean_width")
        )
        for key in metrics:
            assert actual[key] == expected[key]
        assert actual["calibration_rows" if stage == "calibration" else "rows"] == (
            3 if stage == "calibration" else 2
        )

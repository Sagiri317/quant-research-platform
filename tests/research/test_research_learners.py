"""Hand-derived search and calibration results through native persisted datasets."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.datasets.learning import research_frame
from quant_research_platform.research.factors.search import (
    apply_formula,
    formula_values,
    search_factors,
)
from quant_research_platform.research.predictions.uncertainty import (
    apply_uncertainty,
    calibrate_uncertainty,
    conformal_radius,
    evaluate_uncertainty,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.research.direct_model_test_support import fit_frame, predict_frame


def learning_panel(root):
    stamps = pd.date_range("2020-01-01", periods=12).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": ["a", "b", "c"] * 12,
            "x": [1.0, 2.0, 3.0] * 12,
            "eligible": True,
            "label": [0.1, 0.2, 0.3] * 12,
            "label_available_at": stamps + pd.Timedelta(hours=1),
        }
    )
    spec = DataSpec(
        data_id="learning",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            "eligible": "mask",
            "label": "label",
            "label_available_at": "context",
        },
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "label_available_at"},
    )
    write_dataset(root, frame, spec=spec)
    return frame, spec


def test_learning_panel_checks_its_axis_when_dataset_keys_are_wider(tmp_path):
    frame, spec = learning_panel(tmp_path / "base")
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame["observation"] = np.arange(len(frame))
    spec = replace(
        spec,
        primary_key=(*spec.primary_key, "observation"),
        column_roles={**spec.column_roles, "observation": "key"},
    )
    root = tmp_path / "wider"
    write_dataset(root, frame, spec=spec)
    with pytest.raises(ValueError, match="research_panel_key_invalid"):
        research_frame(root, {"start": "2020-01-01", "end": "2020-01-12"}, ["x"])


def test_formula_whitelist_exact_windows_and_segment_reset():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="s"),
            "asset_id": "a",
            "eligible": True,
            "x": [1.0, 3.0, 5.0, 7.0],
            "segment_id": [0, 0, 1, 1],
        }
    )
    assert formula_values(frame, "mean(x,2)", ("x",)).tolist()[1::2] == [2.0, 6.0]
    assert formula_values(frame, "delay(x,1)", ("x",)).iloc[[0, 2]].isna().all()
    assert formula_values(frame, "x/(x-x)", ("x",)).isna().all()
    for expression in ('__import__("os")', "x.shift(-1)", "delay(x,-1)"):
        with pytest.raises(ValueError, match="not_allowed|positive_integer"):
            formula_values(frame, expression, ("x",))


@pytest.mark.parametrize("method", ["random", "ppo"])
def test_formula_search_retains_undefined_and_applies_frozen_selection(
    tmp_path, method
):
    root = tmp_path / "input"
    learning_panel(root)
    config = {
        "dataset": str(root),
        "columns": ["x"],
        "baseline_columns": [],
        "label": "label",
        "train_window": {"start": "2020-01-01", "end": "2020-01-03"},
        "validation_window": {"start": "2020-01-05", "end": "2020-01-07"},
        "train_cutoff": "2020-01-04",
        "selection_cutoff": "2020-01-08",
        "method": method,
        "budget": 4,
        "max_depth": 2,
        "windows": [2],
        "seed": 3,
        "expressions": ["x", "-x", "x-x"],
        "output": str(tmp_path / "search.json"),
    }
    if method == "ppo":
        config.update(
            timesteps=8,
            device="cpu",
            ppo={
                "n_steps": 4,
                "batch_size": 4,
                "n_epochs": 1,
                "policy_kwargs": {"net_arch": [8]},
            },
        )
    saved = search_factors(config)
    result = parse_qrp_json_object(saved.read_bytes())
    assert result["selected_expression"] == "x"
    assert result["candidates"][0]["train_rank_ic"] == pytest.approx(1)
    assert result["candidates"][1]["validation_rank_ic"] == pytest.approx(-1)
    assert result["candidates"][2]["status"] == "UNDEFINED"
    output = apply_formula(
        {
            "search": str(saved),
            "dataset": str(root),
            "input_window": {"start": "2020-01-08", "end": "2020-01-12"},
            "output_window": {"start": "2020-01-09", "end": "2020-01-12"},
            "output": str(tmp_path / "factor"),
        }
    )
    assert read_dataset(output).discovered_factor.tolist() == [1.0, 2.0, 3.0] * 4


def test_conformal_radius_and_real_prediction_calibration_roundtrip(tmp_path):
    assert conformal_radius([0.01, 0.02, 0.03, 0.04], 0.25) == pytest.approx(0.04)
    with pytest.raises(ValueError, match="too_small"):
        conformal_radius([0.1], 0.1)
    frame, spec = learning_panel(tmp_path / "base")
    model = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=1,
    )
    case = fit_frame(
        tmp_path / "fitted",
        frame.loc[frame.timestamp.lt("2020-01-04")],
        model,
        cutoff=pd.Timestamp("2020-01-04"),
    )
    validation = frame.loc[frame.timestamp.between("2020-01-05", "2020-01-08")].copy()
    validation["label"] += np.repeat([0.01, 0.02, 0.03, 0.04], 3)
    write_dataset(tmp_path / "labels", validation, spec=spec)
    window = {"start": "2020-01-05", "end": "2020-01-08"}
    _, predictions, _ = predict_frame(
        tmp_path / "cal-pred", validation, case, window=window
    )
    calibration = calibrate_uncertainty(
        {
            "dataset": str(tmp_path / "labels"),
            "predictions": str(predictions),
            "window": window,
            "label": "label",
            "cutoff": "2020-01-09",
            "alpha": 0.25,
            "unit": "decimal_return",
            "block": "timestamp",
            "output": str(tmp_path / "calibration.json"),
        }
    )
    facts = parse_qrp_json_object(calibration.read_bytes())
    assert facts["radius"] == pytest.approx(0.04)
    assert facts["calibration_blocks"] == 4
    future = frame.loc[frame.timestamp.ge("2020-01-10")].copy()
    future["label"] += 0.02
    window = {"start": "2020-01-10", "end": "2020-01-12"}
    _, predictions, _ = predict_frame(
        tmp_path / "future-pred", future, case, window=window
    )
    intervals = apply_uncertainty(
        {
            "calibration": str(calibration),
            "predictions": str(predictions),
            "window": window,
            "cost": 0.01,
            "output": str(tmp_path / "intervals"),
        }
    )
    values = read_dataset(intervals)
    assert values.conservative_score.tolist() == pytest.approx([0.05, 0.15, 0.25] * 3)
    write_dataset(tmp_path / "future-labels", future, spec=spec)
    report = evaluate_uncertainty(
        {
            "intervals": str(intervals),
            "dataset": str(tmp_path / "future-labels"),
            "window": window,
            "label": "label",
            "cutoff": "2020-01-13",
            "output": str(tmp_path / "coverage.json"),
        }
    )
    assert parse_qrp_json_object(report.read_bytes())["coverage"] == 1
    changed = tmp_path / "changed-label-definition"
    write_dataset(
        changed,
        frame,
        spec=spec,
        metadata={
            "producer_diagnostics": {
                "label_semantics": {"label": {"horizon_sessions": 99}}
            }
        },
    )
    with pytest.raises(ValueError, match="uncertainty_prediction_target_mismatch"):
        calibrate_uncertainty(
            {
                **facts["config"],
                "dataset": str(changed),
                "output": str(tmp_path / "invalid-calibration.json"),
            }
        )
    with pytest.raises(ValueError, match="uncertainty_evaluation_target_mismatch"):
        evaluate_uncertainty(
            {
                "intervals": str(intervals),
                "dataset": str(changed),
                "window": window,
                "label": "label",
                "cutoff": "2020-01-13",
                "output": str(tmp_path / "invalid-coverage.json"),
            }
        )

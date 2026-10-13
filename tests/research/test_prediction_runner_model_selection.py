"""Real candidate-domain scoring and frozen selection, independent of run state."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import (
    read_selection,
    score,
    select,
    select_best_validation_trial,
)
from quant_research_platform.research.predictions.runner import predict
from tests.research.direct_model_test_support import fit_frame


def candidates(
    tmp_path, *, algorithms=("linear_regression", "ic_weighted_rank_ensemble")
):
    stamps = pd.date_range("2020-01-01", periods=6).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": ["a", "b", "c"] * 6,
            "x": [1.0, 2.0, 3.0] * 6,
            "label": [0.1, 0.2, 0.3] * 6,
            "eligible": True,
            "label_available_at": stamps,
        }
    )
    frame.loc[
        (frame.timestamp >= pd.Timestamp("2020-01-04")) & frame.asset_id.eq("a"), "x"
    ] = np.nan
    cases, paths = [], []
    window = {"start": "2020-01-04", "end": "2020-01-06"}
    for algorithm in algorithms:
        spec = ModelSpec(
            model_id=algorithm,
            model_algorithm=algorithm,
            input_factor_columns=("x",),
            target_column="label",
            random_seed=19,
        )
        case = fit_frame(
            tmp_path / algorithm,
            frame.loc[frame.timestamp <= pd.Timestamp("2020-01-03")],
            spec,
        )
        future = tmp_path / algorithm / "future.parquet"
        frame.to_parquet(future, index=False)
        config = {
            "model": str(case.root),
            "inputs": str(future),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / algorithm / "predictions"),
            "prediction_window": window,
            "sample_role": "validation",
        }
        paths.append(str(predict(config)))
        cases.append(case)
    labels = tmp_path / "labels.parquet"
    frame.to_parquet(labels, index=False)
    config = {
        "predictions": paths,
        "candidate_ids": list(algorithms),
        "labels": str(labels),
        "label_spec": cases[0].config["input_spec"],
        "validation_windows": [window],
        "scoring_window": window,
        "cutoff": "2020-01-06",
        "test_window": {"start": "2020-01-07", "end": "2020-01-09"},
        "selection_group": "quarterly",
        "validation_axis": "cross_sectional",
        "label_unit": "decimal_return",
        "output": str(tmp_path / "scores.json"),
    }
    return cases, config


def test_candidate_domains_survive_new_common_comparison(tmp_path):
    cases, config = candidates(tmp_path)
    frames = [read_predictions(path)[0] for path in config["predictions"]]
    assert [len(frame) for frame in frames] == [6, 9]
    before = [
        (Path(path) / "prediction.parquet").read_bytes()
        for path in config["predictions"]
    ]
    scored = json.loads(score(config).read_text())
    assert scored["common_rows"] == 6
    assert all(row["validation_rows"] == 6 for row in scored["metrics"])
    single = {
        **config,
        "predictions": config["predictions"][:1],
        "candidate_ids": config["candidate_ids"][:1],
        "output": str(tmp_path / "single.json"),
    }
    score(single)
    assert before == [
        (Path(path) / "prediction.parquet").read_bytes()
        for path in config["predictions"]
    ]
    assert "score_eligible" not in frames[0]


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "wrong_key", "inf", "nat_metadata"]
)
def test_scoring_rejects_broken_candidate_before_intersection(tmp_path, mutation):
    _cases, config = candidates(tmp_path)
    root = Path(config["predictions"][0])
    frame = pd.read_parquet(root / "prediction.parquet")
    if mutation == "missing":
        frame = frame.iloc[:-1]
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "wrong_key":
        frame.loc[0, "asset_id"] = "unknown"
    elif mutation == "inf":
        frame.loc[0, "prediction_score"] = np.inf
    else:
        path = root / "manifest.json"
        payload = json.loads(path.read_text())
        payload["metadata"]["effective_prediction_end"] = "NaT"
        path.write_text(json.dumps(payload))
    frame.to_parquet(root / "prediction.parquet", index=False)
    with pytest.raises(ValueError):
        score(config)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_candidate",
        "duplicate_candidate",
        "missing_fold",
        "global_partial",
        "test_role",
        "test_metric",
    ],
)
def test_selection_requires_full_declared_validation_science(tmp_path, mutation):
    _cases, config = candidates(tmp_path)
    if mutation == "missing_candidate":
        config["predictions"] = config["predictions"][:1]
    elif mutation == "duplicate_candidate":
        config["candidate_ids"] = [config["candidate_ids"][0]] * 2
    elif mutation == "missing_fold":
        config["predictions"][0] = []
    elif mutation == "global_partial":
        config.update(
            selection_group="global",
            scoring_window={"start": "2020-01-05", "end": "2020-01-06"},
        )
    elif mutation == "test_role":
        path = Path(config["predictions"][0]) / "manifest.json"
        value = json.loads(path.read_text())
        value["metadata"]["sample_role"] = "test"
        path.write_text(json.dumps(value))
    else:
        path = score(config)
        with pytest.raises(ValueError):
            select(
                {
                    "scores": str(path),
                    "output": str(tmp_path / "selection.json"),
                    "selection_metric": "test_rankic",
                    "selection_direction": "maximize",
                    "tie_break_order": config["candidate_ids"],
                }
            )
        return
    with pytest.raises(ValueError):
        score(config)


def test_future_labels_do_not_change_prior_causal_selection(tmp_path):
    _cases, config = candidates(tmp_path)
    config.update(
        cutoff="2020-01-05",
        validation_windows=[{"start": "2020-01-04", "end": "2020-01-06"}],
    )
    # A validation interval after the decision is never admissible.
    with pytest.raises(ValueError, match="fit_window_after_cutoff"):
        score(config)


@pytest.mark.parametrize(
    "values,expected", [([-0.4, None], "a"), ([None, None], "b"), ([1.0, 1.0], "b")]
)
def test_defined_negative_and_all_undefined_keep_declared_ties(values, expected):
    rows = pd.DataFrame({"candidate": ["a", "b"], "validation_rankic": values})
    assert (
        select_best_validation_trial(
            rows,
            trial_id_column="candidate",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=["validation_rankic"],
            test_metric_columns=[],
            tie_break_order=("b", "a"),
        )
        == expected
    )


@pytest.mark.parametrize("failed_family", ["linear_regression", "failed_family"])
def test_failed_attempt_stays_in_complete_candidate_score(tmp_path, failed_family):
    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    failed = {
        **cases[0].config,
        "output": str(tmp_path / "failed"),
        "family_id": failed_family,
        "model": {**cases[0].config["model"], "model_id": "failed"},
        "fit_windows": [{"start": "2019-01-01", "end": "2019-01-02"}],
        "fit_cutoff": "2019-01-02",
    }
    with pytest.raises(ValueError):
        fit(failed)
    config["selection_group"] = "global"
    config["candidate_ids"].append("failed")
    config["predictions"].append({"trial": str(tmp_path / "failed.trial")})
    result = json.loads(score(config).read_text())
    assert result["candidate_ids"] == ["linear_regression", "failed"]
    assert result["candidates"][1]["status"] == "failed"
    assert result["metrics"][1]["validation_rows"] == 0
    assert result["metrics"][1]["validation_rankic"] is None
    assert result["metrics"][0]["validation_rows"] == 6
    selection = select(
        {
            "scores": config["output"],
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["failed", "linear_regression"],
            "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "refit_cutoff": "2020-01-06",
        }
    )
    frozen = read_selection(selection)
    assert frozen["selected_model_id"] == "linear_regression"
    family = next(
        item for item in frozen["families"] if item["family_id"] == failed_family
    )
    assert family["failure_reasons"] == {"failed": "ValueError"}
    if failed_family == "failed_family":
        assert family["selected_config_id"] is None
        assert failed_family not in frozen["refit_windows"]
        from quant_research_platform.artifacts.statistics import (
            read_statistics_artifact,
        )
        from quant_research_platform.research.statistics_production import (
            model_statistics,
        )

        statistics = model_statistics(
            {
                "dataset": config["labels"],
                "input_spec": config["label_spec"],
                "output": str(tmp_path / "statistics"),
                "window": config["scoring_window"],
                "sample_role": "validation",
                "evaluation": {"primary_label": "label"},
                "statistics": {
                    "hac_lag": 0,
                    "bootstrap_block": 1,
                    "bootstrap_repetitions": 10,
                },
                "root_seed": 37,
                "selection_group": "global",
                "comparison_model_id": "linear_regression",
                "prediction_id": "actual-family-predictions",
                "periods": [
                    {
                        "period_id": "validation",
                        "window": config["scoring_window"],
                        "selection": str(selection),
                        "predictions": config["predictions"][:1],
                    }
                ],
            }
        )
        saved = read_statistics_artifact(statistics)
        with saved.table("model_losses.parquet") as losses:
            failed_losses = losses.loc[losses.family_id.eq("failed_family")]
            assert not failed_losses.empty
            assert failed_losses.status.eq("UNDEFINED").all()
            assert failed_losses.reason.eq("family_training_failed").all()
            assert (
                losses.loc[losses.family_id.eq("linear_regression"), "status"]
                .eq("DEFINED")
                .all()
            )
        with saved.table("model_trials.parquet") as trials:
            assert set(trials.family_id) == {"linear_regression", "failed_family"}
            assert set(trials.config_id) == {"linear_regression", "failed"}


def test_refit_freeze_rejects_undeclared_crypto_gap(tmp_path):
    from quant_research_platform.research.models.trial_selection import (
        _validate_refit_scope,
    )

    value = {
        "refit_cutoff": "2020-01-06",
        "cutoff": "2020-01-06",
        "test_window": {"start": "2020-01-07", "end": "2020-01-09"},
        "families": [{"family_id": "f", "selected_config_id": "m"}],
        "candidates": [
            {
                "model_id": "m",
                "fit_windows": [{"start": "2020-01-01", "end": "2020-01-02"}],
            }
        ],
        "validation_windows": [{"start": "2020-01-04", "end": "2020-01-06"}],
        "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
    }
    with pytest.raises(ValueError, match="outside_observed"):
        _validate_refit_scope(value)
    value["refit_windows"] = [
        {"start": "2020-01-01", "end": "2020-01-02"},
        {"start": "2020-01-04", "end": "2020-01-06"},
    ]
    assert len(_validate_refit_scope(value)[0]["f"]) == 2


def test_each_frozen_family_winner_can_refit_but_family_change_cannot(tmp_path):
    cases, config = candidates(tmp_path)
    scores = score(config)
    selected = select(
        {
            "scores": str(scores),
            "output": str(tmp_path / "selected.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": config["candidate_ids"],
            "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "refit_cutoff": "2020-01-06",
        }
    )
    frozen = read_selection(selected)
    assert len(frozen["families"]) == 2
    for index, case in enumerate(cases):
        refit = {
            **case.config,
            "inputs": str(
                tmp_path / case.facts["model"]["model_id"] / "future.parquet"
            ),
            "output": str(tmp_path / f"final-{index}"),
            "fit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "fit_cutoff": "2020-01-06",
            "selection": str(selected),
            "test_window": config["test_window"],
        }
        assert fit(refit).exists()
        with pytest.raises(ValueError, match="family_market_or_units"):
            fit(
                {
                    **refit,
                    "output": str(tmp_path / f"wrong-{index}"),
                    "family_id": "wrong",
                }
            )


def test_explicit_prediction_asset_scope_is_preserved_at_score(tmp_path):
    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    path = predict(
        {
            "model": str(case.root),
            "inputs": str(tmp_path / "linear_regression" / "future.parquet"),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / "scoped-pred"),
            "prediction_window": config["validation_windows"][0],
            "sample_role": "validation",
            "asset_ids": ["b"],
        }
    )
    frame, metadata = read_predictions(path)
    assert set(frame.asset_id) == {"b"} and metadata["asset_ids"] == ["b"]
    config.update(predictions=[str(path)])
    assert json.loads(score(config).read_text())["common_rows"] == 3


@pytest.mark.parametrize(
    "algorithm",
    ["linear_regression", "equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"],
)
def test_output_asset_projection_preserves_full_current_cross_section(
    tmp_path, algorithm
):
    cases, config = candidates(tmp_path, algorithms=(algorithm,))
    case = cases[0]
    full, _ = read_predictions(config["predictions"][0])
    path = predict(
        {
            "model": str(case.root),
            "inputs": str(tmp_path / algorithm / "future.parquet"),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / "scoped"),
            "prediction_window": config["validation_windows"][0],
            "sample_role": "validation",
            "asset_ids": ["c"],
        }
    )
    actual, _ = read_predictions(path)
    expected = full.loc[full.asset_id.eq("c")].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("explicit_scope", [True, False, "multi"])
def test_crypto_validation_axis_comes_from_training_universe(tmp_path, explicit_scope):
    stamps = pd.date_range("2020-01-01", periods=6)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": "btc",
            "eligible": True,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "label": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "label_available_at": stamps,
        }
    )
    spec = ModelSpec(
        model_id="single",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        random_seed=19,
    )
    case = fit_frame(tmp_path / "single", frame.iloc[:3], spec)
    if explicit_scope:
        case_config = {
            **case.config,
            "output": str(tmp_path / "explicit"),
            "asset_ids": ["btc", "eth"] if explicit_scope == "multi" else ["btc"],
        }
        model = fit(case_config)
    else:
        model = case.root
    inputs = tmp_path / "inputs.parquet"
    frame.to_parquet(inputs, index=False)
    window = {"start": "2020-01-04", "end": "2020-01-06"}
    predictions = predict(
        {
            "model": str(model),
            "inputs": str(inputs),
            "input_spec": case.config["input_spec"],
            "output": str(tmp_path / "pred"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    config = {
        "predictions": [str(predictions)],
        "candidate_ids": ["single"],
        "labels": str(inputs),
        "label_spec": case.config["input_spec"],
        "validation_windows": [window],
        "scoring_window": window,
        "cutoff": "2020-01-06",
        "test_window": {"start": "2020-01-07", "end": "2020-01-09"},
        "selection_group": "global",
        "validation_axis": "cross_sectional",
        "label_unit": "decimal_return",
        "output": str(tmp_path / "scores.json"),
    }
    if explicit_scope == "multi":
        assert (
            json.loads(score(config).read_text())["metrics"][0]["validation_ic"] is None
        )
        config.update(
            validation_axis="single_asset_time_series",
            output=str(tmp_path / "invalid-axis.json"),
        )
        with pytest.raises(ValueError, match="training_scope_validation_axis"):
            score(config)
        return
    with pytest.raises(ValueError, match="training_scope_validation_axis"):
        score(config)
    config["validation_axis"] = "single_asset_time_series"
    assert json.loads(score(config).read_text())["metrics"][0]["validation_ic"] == 1.0


def test_scoring_uses_saved_training_domain_without_reopening_training_data(tmp_path):
    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    assert case.result.metadata["training_asset_ids"] == ["a", "b", "c"]
    Path(case.config["inputs"]).unlink()
    assert json.loads(score(config).read_text())["common_rows"] == 6

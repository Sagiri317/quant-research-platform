"""Real candidate-domain scoring and frozen selection, independent of run state."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import (
    read_selection,
    score,
    select,
    select_best_validation_trial,
)
from quant_research_platform.research.predictions.runner import predict
from tests.research.direct_model_test_support import candidates, fit_frame


def test_candidate_domains_survive_new_common_comparison(tmp_path):
    cases, config = candidates(tmp_path)
    second = cases[1]
    scoped = predict(
        {
            "model": str(second.root),
            "inputs": str(
                tmp_path / second.facts["model"]["model_id"] / "future.parquet"
            ),
            "input_spec": second.config["input_spec"],
            "output": str(tmp_path / "scoped"),
            "prediction_window": config["scoring_window"],
            "sample_role": "validation",
            "asset_ids": ["c"],
        }
    )
    config["predictions"][1] = str(scoped)
    frames = [read_predictions(path)[0] for path in config["predictions"]]
    assert [len(frame) for frame in frames] == [6, 3]
    assert [set(frame.asset_id) for frame in frames] == [{"b", "c"}, {"c"}]
    before = [
        (Path(path) / "prediction.parquet").read_bytes()
        for path in config["predictions"]
    ]
    scored = json.loads(score(config).read_text())
    assert scored["common_rows"] == 3
    assert all(row["validation_rows"] == 3 for row in scored["metrics"])
    single = {
        **config,
        "predictions": config["predictions"][:1],
        "candidate_ids": config["candidate_ids"][:1],
        "output": str(tmp_path / "single.json"),
    }
    assert json.loads(score(single).read_text())["common_rows"] == 6
    assert before == [
        (Path(path) / "prediction.parquet").read_bytes()
        for path in config["predictions"]
    ]
    assert "score_eligible" not in frames[0]


@pytest.mark.parametrize(
    "mutation",
    ["missing", "truncated", "duplicate", "wrong_key", "inf", "nat_metadata"],
)
def test_scoring_reports_missing_keys_and_rejects_invalid_rows(tmp_path, mutation):
    _cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    root = Path(config["predictions"][0])
    frame = pd.read_parquet(root / "prediction.parquet")
    if mutation in {"missing", "truncated"}:
        frame = frame.iloc[:-1]
        if mutation == "missing":
            # A smaller valid forecast domain is scientific missingness, not
            # damage to an artifact that still declares the original row count.
            metadata = json.loads((root / "manifest.json").read_text())["metadata"]
            metadata.pop("prediction_rows")
            metadata.pop("effective_prediction_end")
            root = write_prediction_outputs(
                output_path=tmp_path / "partial-predictions",
                output_chunks=[frame],
                metadata=metadata,
            )
            config["predictions"] = [str(root)]
            saved, saved_metadata = read_predictions(root)
            assert len(saved) == saved_metadata["prediction_rows"] == 5
            assert pd.Timestamp(saved_metadata["effective_prediction_end"]) == (
                saved.timestamp.max()
            )
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
    if mutation != "missing":
        frame.to_parquet(root / "prediction.parquet", index=False)
    if mutation in {"missing", "wrong_key"}:
        result = json.loads(score(config).read_text())
        coverage = result["candidates"][0]["coverage"]
        assert coverage["paired_prediction_rows"] == 5
        assert coverage["mature_label_rows_without_prediction"] == 4
        assert coverage["prediction_rows_without_label"] == int(mutation == "wrong_key")
    elif mutation == "truncated":
        with pytest.raises(ValueError, match="^prediction_metadata_inconsistent$"):
            score(config)
    else:
        with pytest.raises(ValueError):
            score(config)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_candidate",
        "duplicate_candidate",
        "missing_fold",
        "global_partial",
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
    else:
        path = score(config)
        with pytest.raises(ValueError, match="selection_metric_direction_invalid"):
            select(
                {
                    "scores": str(path),
                    "output": str(tmp_path / "selection.json"),
                    "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
                    "refit_cutoff": "2020-01-06",
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
    labels = Path(config["labels"])
    frame = pd.read_parquet(labels)
    future = frame.timestamp.eq(pd.Timestamp("2020-01-06"))
    assert future.sum() == 3
    frame.loc[future, "label_available_at"] = pd.Timestamp("2020-01-10")
    frame.to_parquet(labels, index=False)

    def selected(name):
        scores = score({**config, "output": str(tmp_path / f"{name}-scores.json")})
        selected = select(
            {
                "scores": str(scores),
                "output": str(tmp_path / f"{name}-selection.json"),
                "selection_metric": "validation_rankic",
                "selection_direction": "maximize",
                "tie_break_order": config["candidate_ids"],
                "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
                "refit_cutoff": "2020-01-06",
            }
        )
        return json.loads(scores.read_text()), read_selection(selected)

    before, frozen_before = selected("before")
    frame.loc[future, "label"] = [-1e9, 0.0, 1e9]
    frame.to_parquet(labels, index=False)
    after, frozen_after = selected("after")
    assert before["common_rows"] == after["common_rows"] == 4
    assert before["metrics"] == after["metrics"]
    assert frozen_before["selected_model_id"] == frozen_after["selected_model_id"]
    assert frozen_before["selected_model_id"] is not None
    # The oracle must react to a change on the admitted mature domain.
    mature = frame.timestamp.eq(pd.Timestamp("2020-01-04"))
    frame.loc[mature, "label"] = [0.3, 0.2, 0.1]
    frame.to_parquet(labels, index=False)
    changed, _ = selected("mature-changed")
    assert changed["metrics"] != before["metrics"]


def test_validation_window_after_cutoff_is_rejected(tmp_path):
    _cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    with pytest.raises(ValueError, match="fit_window_after_cutoff"):
        score({**config, "cutoff": "2020-01-05"})


@pytest.mark.parametrize(
    "values,expected", [([-0.4, None], "a"), ([None, None], None), ([1.0, 1.0], "b")]
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


def test_refit_uses_explicit_current_domain_without_filling_disjoint_gaps():
    from quant_research_platform.research.models.trial_selection import (
        _validate_refit_scope,
    )

    value = {
        "refit_cutoff": "2020-01-06",
        "cutoff": "2020-01-06",
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
    assert _validate_refit_scope(value)[0]["f"] == value["refit_windows"]
    value["refit_windows"] = [
        {"start": "2020-01-01", "end": "2020-01-02"},
        {"start": "2020-01-04", "end": "2020-01-06"},
    ]
    assert _validate_refit_scope(value)[0]["f"] == value["refit_windows"]
    value["refit_windows"] = [{"start": "2020-01-01", "end": "2020-01-07"}]
    with pytest.raises(ValueError, match="fit_window_after_cutoff"):
        _validate_refit_scope(value)


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
        }
        assert fit(refit).exists()
        with pytest.raises(
            ValueError, match="model_scientific_facts_mismatch: family_id"
        ):
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
def test_crypto_validation_axis_follows_current_samples(tmp_path, explicit_scope):
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
    training = frame.iloc[:3]
    if explicit_scope == "multi":
        training = pd.concat(
            [training, training.assign(asset_id="eth")], ignore_index=True
        )
    case = fit_frame(tmp_path / "single", training, spec)
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
        "selection_group": "global",
        "validation_axis": "cross_sectional",
        "label_unit": "decimal_return",
        "output": str(tmp_path / "scores.json"),
    }
    assert json.loads(score(config).read_text())["metrics"][0]["validation_ic"] is None
    config.update(
        validation_axis="single_asset_time_series",
        output=str(tmp_path / "temporal-scores.json"),
    )
    assert json.loads(score(config).read_text())["metrics"][0][
        "validation_ic"
    ] == pytest.approx(1.0)


def test_temporal_score_still_requires_actual_single_asset_samples(tmp_path):
    _, config = candidates(tmp_path, algorithms=("linear_regression",))
    config["validation_axis"] = "single_asset_time_series"
    with pytest.raises(
        ValueError, match="single_asset_time_series_evaluation_requires_one_asset"
    ):
        score(config)


def test_scoring_uses_saved_training_domain_without_reopening_training_data(tmp_path):
    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    assert case.result.metadata["training_asset_ids"] == ["a", "b", "c"]
    Path(case.config["inputs"]).unlink()
    assert json.loads(score(config).read_text())["common_rows"] == 6


def test_score_retains_only_predictions_and_matches_full_join(tmp_path, monkeypatch):
    import weakref

    from quant_research_platform.artifacts.outputs import prediction as artifact
    from quant_research_platform.research.models import trial_selection

    _cases, config = candidates(tmp_path)
    labels = pd.read_parquet(config["labels"])
    labels.loc[
        labels.timestamp.eq("2020-01-04") & labels.asset_id.eq("b"), "eligible"
    ] = False
    labels.loc[
        labels.timestamp.eq("2020-01-05") & labels.asset_id.eq("c"),
        "label_available_at",
    ] = pd.Timestamp("2020-01-07")
    labels.loc[labels.timestamp.eq("2020-01-06") & labels.asset_id.eq("c"), "label"] = (
        0.4
    )
    labels.to_parquet(config["labels"], index=False)
    # Literal mature shared keys: neither production filtering nor metric reduction
    # participates in this independent date-equal loss oracle.
    expected_keys = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2020-01-04"), "c"),
            (pd.Timestamp("2020-01-05"), "b"),
            (pd.Timestamp("2020-01-06"), "b"),
            (pd.Timestamp("2020-01-06"), "c"),
        ],
        names=["timestamp", "asset_id"],
    )
    expected_losses = {}
    for path in config["predictions"]:
        predictions, metadata = read_predictions(path)
        selected = predictions.set_index(["timestamp", "asset_id"]).loc[expected_keys]
        calibration = (
            artifact.prediction_model_metadata(metadata)
            .get("rank_parameters", {})
            .get("label_calibration")
        )
        economic = selected.prediction_score
        if calibration is not None:
            economic = calibration[0] + calibration[1] * economic
        target = labels.set_index(["timestamp", "asset_id"]).loc[expected_keys, "label"]
        loss = (economic - target) ** 2
        expected_losses[metadata["model_id"]] = (
            loss.groupby(level="timestamp").mean().mean()
        )

    retained = []
    original_concat = pd.concat

    def observe_concat(objects, *args, **kwargs):
        result = original_concat(objects, *args, **kwargs)
        if "prediction_label_value" in result:
            assert list(result.columns) == [
                "prediction_score",
                "prediction_label_value",
            ]
            retained.append(weakref.ref(result))
        return result

    reads = []
    original_read = artifact.read_predictions

    def observe_read(path, **kwargs):
        reads.append(path)
        return original_read(path, **kwargs)

    original_evaluate = trial_selection.evaluate_model_trial_metrics_from_chunks
    scored_count = 0

    def observe_evaluate(chunks, **kwargs):
        nonlocal scored_count
        assert all(reference() is None for reference in retained[:scored_count])
        assert set(chunks[0]) == {
            "timestamp",
            "asset_id",
            "prediction_score",
            "prediction_label_value",
            "label",
        }
        scored_count += 1
        return original_evaluate(chunks, **kwargs)

    monkeypatch.setattr(pd, "concat", observe_concat)
    monkeypatch.setattr(artifact, "read_predictions", observe_read)
    monkeypatch.setattr(
        trial_selection, "evaluate_model_trial_metrics_from_chunks", observe_evaluate
    )
    scored = json.loads(score(config).read_text())
    for metric in scored["metrics"]:
        assert metric["validation_rows"] == 4
        assert metric["validation_valid_ic_count"] == 1
        assert metric["validation_valid_rankic_count"] == 1
        assert metric["validation_mse"] == pytest.approx(
            expected_losses[metric["model_id"]], rel=1e-12, abs=1e-16
        )
        assert metric["validation_ic"] == pytest.approx(1.0, abs=1e-14)
        assert metric["validation_rankic"] == pytest.approx(1.0, abs=1e-14)
        assert metric["validation_ic_reason"] is None
        assert metric["validation_rankic_reason"] is None
    linear = next(
        item for item in scored["metrics"] if item["model_id"] == "linear_regression"
    )
    assert linear["validation_mse"] == pytest.approx(0.01 / 2 / 3, rel=1e-12, abs=1e-16)
    assert linear["validation_mse"] != pytest.approx(0.01 / 4, rel=1e-12, abs=1e-16)
    assert scored["common_rows"] == 4
    assert len(reads) == len(config["predictions"])
    assert len(retained) == len(config["candidate_ids"])
    assert all(reference() is None for reference in retained)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
    select_best_validation_trial,
    validate_trial_metric_columns,
)


def _assert_metric_equivalence(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value, rel=1e-12, abs=1e-14)
        else:
            assert actual[key] == value


@pytest.mark.parametrize(
    "target", ["label", "prediction_score", "prediction_label_value"]
)
def test_native_score_selection_is_invariant_to_label_name(tmp_path, target):
    import json

    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.models.trial_selection import (
        read_selection,
        score,
        select,
    )
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import write_dataset

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5).repeat(3),
            "asset_id": list("abc") * 5,
            "x": [1.0, 2.0, 3.0] * 5,
            target: [0.1, 0.2, 0.3] * 3 + [0.3, 0.2, 0.1] * 2,
            "eligible": True,
        }
    )
    frame["mature"] = frame.timestamp
    # One prediction is valid but its label is not yet mature at scoring cutoff.
    frame.loc[12, "mature"] = pd.Timestamp("2020-01-06")
    spec = ProcessedDataSpec(
        data_id="input",
        market_id="crypto",
        kind="panel",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            target: "label",
            "eligible": "mask",
            "mature": "context",
        },
        column_units={"x": "dimensionless", target: "decimal_return"},
        label_availability_columns={target: "mature"},
    )
    inputs = tmp_path / "input"
    write_dataset(inputs, frame, spec=spec)
    window = {"start": "2020-01-04", "end": "2020-01-05"}
    predictions = []
    for model_id in ("wrong", "correct"):
        training = frame.copy()
        if model_id == "correct":
            training[target] = 0.4 - training[target]
        train_path = tmp_path / f"{model_id}-input"
        write_dataset(train_path, training, spec=spec)
        model = fit(
            {
                "inputs": str(train_path),
                "output": str(tmp_path / f"{model_id}-model"),
                "market_id": "crypto",
                "model": {
                    "model_id": model_id,
                    "model_algorithm": "linear_regression",
                    "input_factor_columns": ["x"],
                    "target_column": target,
                },
                "fit_windows": [{"start": "2020-01-01", "end": "2020-01-03"}],
                "fit_cutoff": "2020-01-03",
            }
        )
        predictions.append(
            str(
                predict(
                    {
                        "model": str(model),
                        "inputs": str(inputs),
                        "output": str(tmp_path / f"{model_id}-prediction"),
                        "prediction_window": window,
                    }
                )
            )
        )
    scored_path = score(
        {
            "predictions": predictions,
            "candidate_ids": ["wrong", "correct"],
            "labels": str(inputs),
            "validation_windows": [window],
            "scoring_window": window,
            "cutoff": "2020-01-05",
            "selection_group": "global",
            "validation_axis": "cross_sectional",
            "output": str(tmp_path / "score.json"),
        }
    )
    scored = json.loads(scored_path.read_text())
    assert scored["target_column"] == target
    assert scored["common_rows"] == 5
    for candidate, metrics, sign, mse in zip(
        scored["candidates"],
        scored["metrics"],
        [-1.0, 1.0],
        [7 / 300, 0.0],
        strict=True,
    ):
        assert candidate["model"]["target_column"] == target
        assert candidate["coverage"] == {
            "current_mature_label_rows": 5,
            "paired_prediction_rows": 5,
            "mature_label_rows_without_prediction": 0,
            "prediction_rows_without_label": 0,
        }
        assert metrics["validation_rows"] == 5
        assert metrics["validation_ic"] == pytest.approx(sign)
        assert metrics["validation_rankic"] == pytest.approx(sign)
        # Equal date weights: mean((.04 + 0 + .04) / 3, (0 + .04) / 2).
        assert metrics["validation_mse"] == pytest.approx(mse, abs=1e-16)
    for metric, direction in [
        ("validation_rankic", "maximize"),
        ("validation_mse", "minimize"),
    ]:
        selected = read_selection(
            select(
                {
                    "scores": str(scored_path),
                    "output": str(tmp_path / f"{metric}-selection.json"),
                    "selection_metric": metric,
                    "selection_direction": direction,
                    "tie_break_order": ["wrong", "correct"],
                    "refit_windows": [{"start": "2020-01-01", "end": "2020-01-05"}],
                    "refit_cutoff": "2020-01-05",
                }
            )
        )
        assert selected["selected_model_id"] == "correct"
        assert selected["target_column"] == target
        assert selected["metrics"] == scored["metrics"]
        assert selected["common_rows"] == 5


def test_model_selection_uses_validation_only_and_explicit_ties() -> None:
    validate_trial_metric_columns(
        selection_metric="validation_rankic",
        validation_metric_columns=("validation_rankic",),
        test_metric_columns=("test_rankic",),
    )
    trials = pd.DataFrame(
        {"trial_id": ["trial_b", "trial_a"], "validation_rankic": [0.7, 0.7]}
    )
    assert (
        select_best_validation_trial(
            trials,
            trial_id_column="trial_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=("trial_a", "trial_b"),
        )
        == "trial_a"
    )
    with pytest.raises(ValueError, match="selection_metric_must_be_validation_metric"):
        validate_trial_metric_columns(
            selection_metric="test_rankic",
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=("test_rankic",),
        )


def test_model_selection_metrics_are_cross_sectional_and_chunk_parity_stable() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "asset_id": ["a", "b", "c"] * 2,
            "label": [1.0, 2.0, 3.0, 101.0, 102.0, 103.0],
            "prediction": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        }
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,),
        model_id="current",
        label_column="label",
        prediction_column="prediction",
    )
    partitioned = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:2], frame.iloc[2:5], frame.iloc[5:]),
        model_id="current",
        label_column="label",
        prediction_column="prediction",
    )
    _assert_metric_equivalence(partitioned, whole)
    assert whole["validation_rows"] == 6
    for prefix in ("ic", "rankic"):
        assert whole[f"validation_std_{prefix}"] == pytest.approx(1.0)
        assert whole[f"validation_{prefix}ir"] == 0.0
        assert whole[f"validation_valid_{prefix}_count"] == 2
        assert whole[f"validation_positive_{prefix}_rate"] == 0.5
        assert whole[f"validation_{prefix}ir_reason"] is None
    assert whole["validation_ic"] == pytest.approx(0.0)
    assert whole["validation_rankic"] == pytest.approx(0.0)


def test_model_selection_rejects_nonfinite_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 2),
            "asset_id": ["a", "b"],
            "label": [1.0, 2.0],
            "prediction": [np.nan, 1.0],
        }
    )
    with pytest.raises(ValueError, match="invalid_trial_metric_values"):
        evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id="current",
            label_column="label",
            prediction_column="prediction",
        )


def test_time_series_trial_metrics_keep_axis_across_chunks_and_degeneracy():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6),
            "asset_id": "a",
            "label": [2.0, 4.0, 1.0, 6.0, 3.0, 5.0],
            "prediction": [5.0, 9.0, 3.0, 13.0, 7.0, 11.0],
        }
    )
    arguments = dict(
        model_id="model", label_column="label", prediction_column="prediction"
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,), **arguments, validation_axis="single_asset_time_series"
    )
    split = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:2], frame.iloc[:0], frame.iloc[2:5], frame.iloc[5:]),
        **arguments,
        validation_axis="single_asset_time_series",
    )
    _assert_metric_equivalence(split, whole)
    assert whole["validation_ic"] == 1.0
    assert whole["validation_icir"] is None
    assert whole["validation_icir_reason"] == "insufficient_observations"
    assert whole["validation_valid_ic_count"] == 1
    assert whole["validation_rankicir_reason"] == "not_applicable"
    assert whole["validation_rankic"] is None
    assert whole["validation_rankic_reason"] == "not_applicable"
    cross_sectional = evaluate_model_trial_metrics_from_chunks((frame,), **arguments)
    assert cross_sectional["validation_ic"] is None
    assert cross_sectional["validation_ic_reason"] == "no_valid_observations"
    single = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:1],), **arguments, validation_axis="single_asset_time_series"
    )
    assert single["validation_ic_reason"] == "no_valid_observations"
    assert single["validation_rankic_reason"] == "not_applicable"
    constant = evaluate_model_trial_metrics_from_chunks(
        (frame.assign(prediction=1.0),),
        **arguments,
        validation_axis="single_asset_time_series",
    )
    assert constant["validation_ic_reason"] == "zero_variance"


@pytest.mark.parametrize("residual", [0.0, 0.0002, 0.02, 2.0])
def test_model_selection_mse_matches_daily_squared_residuals(residual: float) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100),
            "asset_id": "asset",
            "label": 0.0,
            "prediction": [residual, *[0.0] * 99],
        }
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,), model_id="model", label_column="label", prediction_column="prediction"
    )
    split = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:1], frame.iloc[1:]),
        model_id="model",
        label_column="label",
        prediction_column="prediction",
    )
    _assert_metric_equivalence(whole, split)
    assert whole["validation_mse"] == pytest.approx(
        residual**2 / len(frame), rel=1e-12, abs=0.0
    )
    assert whole["validation_ic"] is None
    assert whole["validation_rankic"] is None


@pytest.mark.parametrize("economic", [np.nan, np.inf])
def test_model_selection_undefined_mse_preserves_correlations(economic) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3),
            "asset_id": ["a", "b", "c"],
            "label": [-0.02, 0.0, 0.02],
            "prediction": [0.02, 0.0, -0.02],
            "economic": [economic, 0.0, 0.02],
        }
    )
    arguments = dict(
        model_id="model",
        label_column="label",
        prediction_column="prediction",
        label_prediction_column="economic",
    )
    whole = evaluate_model_trial_metrics_from_chunks((frame,), **arguments)
    split = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:1], frame.iloc[1:]), **arguments
    )
    _assert_metric_equivalence(split, whole)
    assert whole["validation_rows"] == 3
    assert whole["validation_mse"] is None
    assert whole["validation_ic"] == pytest.approx(-1.0)
    assert whole["validation_rankic"] == pytest.approx(-1.0)
    from quant_research_platform.contracts.models import (
        _validate_model_selection_trial_record,
    )

    _validate_model_selection_trial_record(whole)
    for metric, maximize in (("validation_ic", True), ("validation_mse", False)):
        other = dict(whole, model_id="defined", validation_ic=0.0, validation_mse=1.0)
        assert (
            select_best_validation_trial(
                pd.DataFrame([whole, other]),
                trial_id_column="model_id",
                selection_metric=metric,
                maximize=maximize,
                validation_metric_columns=(metric,),
                test_metric_columns=(),
            )
            == "defined"
        )


@pytest.mark.parametrize("field", ["validation_ic", "validation_rankic"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_persisted_trial_correlations_obey_exact_bounds(field, sign):
    import math

    from quant_research_platform.contracts.models import (
        _validate_model_selection_trial_record,
    )

    metrics = {
        "validation_rows": 2,
        "validation_ic": 1.0,
        "validation_std_ic": 0.0,
        "validation_std_ic_reason": None,
        "validation_icir": None,
        "validation_positive_ic_rate": 1.0,
        "validation_valid_ic_count": 1,
        "validation_icir_reason": "insufficient_observations",
        "validation_std_rankic": 0.0,
        "validation_std_rankic_reason": None,
        "validation_rankicir": None,
        "validation_positive_rankic_rate": 1.0,
        "validation_valid_rankic_count": 1,
        "validation_rankicir_reason": "insufficient_observations",
        "validation_ic_reason": None,
        "validation_rankic": 1.0,
        "validation_rankic_reason": None,
        "validation_mse": 0.0,
        "model_id": "model",
    }
    metrics[field] = sign
    _validate_model_selection_trial_record(metrics)
    metrics[field] = sign * math.nextafter(1.0, math.inf)
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        _validate_model_selection_trial_record(metrics)


@pytest.mark.parametrize("axis", ["single_asset_time_series", "cross_sectional"])
def test_ordered_metrics_release_consumed_frames(monkeypatch, axis):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=8),
            "asset_id": "a",
            "label": [1.0, 5.0, -2.0, 6.0, 3.0, -4.0, 7.0, 8.0],
            "prediction": [2.0, 3.0, -4.0, 5.0, 1.0, -3.0, 9.0, 7.0],
        }
    )
    kwargs = dict(
        model_id="probe",
        label_column="label",
        prediction_column="prediction",
        validation_axis=axis,
    )
    expected = evaluate_model_trial_metrics_from_chunks((frame,), **kwargs)
    closed = []

    def chunks():
        try:
            for offset in range(0, len(frame), 2):
                yield frame.iloc[offset : offset + 2]
        finally:
            closed.append(True)

    import weakref

    from quant_research_platform.research.models import trial_selection

    references = []
    original = trial_selection._metric_chunk

    def observed(*args, **kwargs):
        assert sum(ref() is not None for ref in references) <= 2
        values = original(*args, **kwargs)
        references.append(weakref.ref(values))
        return values

    monkeypatch.setattr(trial_selection, "_metric_chunk", observed)
    actual = evaluate_model_trial_metrics_from_chunks(chunks(), **kwargs)
    _assert_metric_equivalence(actual, expected)
    assert closed == [True]
    assert all(ref() is None for ref in references)
    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        evaluate_model_trial_metrics_from_chunks(
            (frame.iloc[4:], frame.iloc[:4]), **kwargs
        )


def test_streaming_metric_failure_closes_source():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2),
            "asset_id": "a",
            "label": [1.0, 2.0],
            "prediction": [2.0, np.inf],
        }
    )
    closed = []

    def chunks():
        try:
            yield frame.iloc[:1]
            yield frame.iloc[1:]
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="invalid_trial_metric_values"):
        evaluate_model_trial_metrics_from_chunks(
            chunks(),
            model_id="probe",
            label_column="label",
            prediction_column="prediction",
            validation_axis="single_asset_time_series",
        )
    assert closed == [True]


@pytest.mark.parametrize(
    "metric,direction",
    [("validation_rankic", "maximize"), ("validation_mse", "minimize")],
)
def test_undefined_auxiliary_mse_survives_score_select_roundtrip(
    tmp_path, metric, direction
):
    import json
    from pathlib import Path

    from quant_research_platform.research.models.trial_selection import (
        read_selection,
        score,
        select,
    )
    from tests.research.direct_model_test_support import candidates

    _, config = candidates(tmp_path)
    labels = pd.read_parquet(config["labels"])
    labels["label"] *= 100
    config["label_spec"]["column_units"]["label"] = "percent"
    config["label_unit"] = "percent"
    labels.to_parquet(config["labels"], index=False)
    score(config)
    scored = json.loads(Path(config["output"]).read_text())
    assert all(row["validation_mse"] is None for row in scored["metrics"])
    assert [row["validation_rankic"] for row in scored["metrics"]] == pytest.approx(
        [1.0] * len(scored["metrics"])
    )
    frozen = read_selection(
        select(
            {
                "scores": config["output"],
                "output": str(tmp_path / "selection.json"),
                "selection_metric": metric,
                "selection_direction": direction,
                "tie_break_order": config["candidate_ids"],
                "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
                "refit_cutoff": "2020-01-06",
            }
        )
    )
    assert frozen["metrics"] == scored["metrics"]
    assert "test_window" not in frozen
    assert frozen["selected_model_id"] == (
        None if metric == "validation_mse" else config["candidate_ids"][0]
    )


def test_materialized_score_metrics_match_independent_unequal_date_oracle():
    rng = np.random.default_rng(710)
    sizes = [7, 3, 11, 4]
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(pd.date_range("2024-01-01", periods=4), sizes),
            "asset_id": [f"a{i}" for count in sizes for i in range(count)],
            "label": rng.normal(0, 0.02, sum(sizes)),
            "prediction": rng.normal(0, 0.03, sum(sizes)),
        }
    )
    frame["economic"] = 0.001 + 0.5 * frame.prediction
    arguments = dict(
        model_id="oracle",
        label_column="label",
        prediction_column="prediction",
        label_prediction_column="economic",
    )
    actual = evaluate_model_trial_metrics_from_chunks(
        [frame.iloc[:2], frame.iloc[2:9], frame.iloc[9:17], frame.iloc[17:]],
        **arguments,
    )
    daily = list(frame.groupby("timestamp", sort=True))
    expected_mse = np.mean(
        [np.mean((group.economic - group.label) ** 2) for _, group in daily]
    )
    assert actual["validation_rows"] == sum(sizes)
    assert actual["validation_mse"] == pytest.approx(expected_mse, rel=1e-12, abs=1e-16)
    assert actual["validation_mse"] != pytest.approx(
        np.mean((frame.economic - frame.label) ** 2), rel=1e-5
    )
    for prefix, method in [("ic", "pearson"), ("rankic", "spearman")]:
        correlations = np.array(
            [group.prediction.corr(group.label, method=method) for _, group in daily]
        )
        assert actual[f"validation_{prefix}"] == pytest.approx(
            correlations.mean(), abs=1e-14
        )
        assert actual[f"validation_std_{prefix}"] == pytest.approx(
            correlations.std(), abs=1e-14
        )
        assert actual[f"validation_{prefix}ir"] == pytest.approx(
            correlations.mean() / correlations.std(), abs=1e-14
        )
        assert actual[f"validation_valid_{prefix}_count"] == 4
        assert actual[f"validation_positive_{prefix}_rate"] == np.mean(correlations > 0)
        assert actual[f"validation_{prefix}_reason"] is None

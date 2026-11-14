from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.rolling import (
    quarterly_folds,
    run_quarterly,
    run_sessions,
    session_folds,
)
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def test_quarterly_windows_expand_and_first_quarter_supplies_initial_training():
    sessions = pd.bdate_range("2014-01-01", "2026-10-30")
    folds = quarterly_folds(
        "2014-01-01",
        {"start": "2018-01-01", "end": "2023-12-31"},
        sessions=sessions,
    )
    assert len(folds) == 24
    assert str(folds[0].train.end) == "2017-12-01"
    assert str(folds[-1].train.end) == "2023-09-01"
    assert all(str(fold.train.start) == "2014-01-01" for fold in folds)
    same = quarterly_folds(
        "2014-01-01",
        {"start": "2014-01-01", "end": "2026-10-30"},
        sessions=sessions,
    )
    assert len(same) == 51
    assert str(same[0].prediction.start) == "2014-04-01"
    assert str(same[-1].prediction.end) == "2026-10-30"
    for fold in same:
        validation = sessions[
            (sessions >= pd.Timestamp(fold.validation.start))
            & (sessions <= pd.Timestamp(fold.validation.end))
        ]
        assert len(validation) == 20
        assert fold.train.end < fold.validation.start < fold.prediction.start


def test_quarterly_runs_real_fit_prediction_and_evaluation(tmp_path):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["timestamp"] = frame.timestamp.map(
        dict(
            zip(
                pd.date_range("2020-01-01", periods=9),
                pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-02-01",
                        "2020-03-01",
                        "2020-04-01",
                        "2020-05-01",
                        "2020-06-01",
                        "2020-07-01",
                        "2020-08-01",
                        "2020-09-01",
                    ]
                ),
                strict=True,
            )
        )
    )
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(days=1)
    frame.to_parquet(inputs, index=False)
    results = run_quarterly(
        {
            "inputs": {"path": str(inputs), "spec": input_spec()},
            "output": str(tmp_path / "rolling"),
            "market_id": "a_share",
            "models": [fit_config(tmp_path, inputs)["model"]],
            "training_start": "2020-01-01",
            "sessions": frame.timestamp.drop_duplicates()
            .dt.strftime("%Y-%m-%d")
            .tolist(),
            "validation_sessions": 1,
            "rolling_window": {"start": "2020-04-01", "end": "2020-09-30"},
            "evaluation_cutoff": "2020-10-01",
            "label_unit": "decimal_return",
            "evaluation": {
                "evaluation_id": "quarterly",
                "market_id": "a_share",
                "research_dataset_id": "synthetic",
                "signals": [
                    {
                        "signal_id": "score",
                        "source_kind": "prediction",
                        "source_column": "score",
                    }
                ],
                "label_columns": ["label"],
                "evaluation_axis": "cross_sectional",
            },
        }
    )
    assert len(results) == 2
    counts = []
    for item in results:
        model, facts = load_model(item["model"])
        counts.append(model.metadata["rows_after_input_label_dropna"])
        predictions, _ = read_predictions(item["predictions"])
        assert len(predictions) == 9
        assert predictions.timestamp.min() > pd.Timestamp(facts["fit_cutoff"])
        assert (
            tmp_path
            / "rolling"
            / item["fold_id"]
            / "linear"
            / "evaluation"
            / "manifest.json"
        ).is_file()
    assert counts == [9, 18]


def test_session_windows_keep_declared_axis_and_partial_final_block():
    days = pd.bdate_range("2022-01-03", periods=125)
    folds = session_folds(days)
    assert len(folds) == 3
    assert folds[0].train.start == days[0].date()
    assert folds[0].train.end == days[59].date()
    assert folds[0].validation.start == days[60].date()
    assert folds[0].validation.end == days[79].date()
    assert folds[0].prediction.start == days[80].date()
    assert folds[1].train.end == days[79].date()
    assert folds[-1].prediction.start == days[120].date()
    assert folds[-1].prediction.end == days[-1].date()


def test_four_models_select_historical_mature_rows_then_refit_without_future_labels(
    tmp_path,
    monkeypatch,
):
    days = pd.bdate_range("2022-01-03", periods=110)
    frame = pd.DataFrame(
        [
            (
                day,
                asset,
                value,
                (0.02 if index < 60 else -0.02) * value + 0.005,
                True,
                day + pd.Timedelta(days=1),
            )
            for index, day in enumerate(days)
            for asset, value in zip(("a", "b", "c"), (-1.0, 0.0, 1.0), strict=True)
        ],
        columns=[
            "timestamp",
            "asset_id",
            "x",
            "label",
            "eligible",
            "label_available_at",
        ],
    )
    immature = frame.timestamp.eq(days[62]) & frame.asset_id.eq("a")
    frame.loc[immature, "label_available_at"] = days[85]
    frame["late"] = frame.x.where(frame.timestamp.ge(days[60]))
    frame["initial_constant"] = np.where(
        frame.timestamp.lt(days[60]), 0.2, frame.x * 0.1
    )
    path = tmp_path / "inputs.parquet"
    frame.to_parquet(path, index=False)
    shared = {
        "input_feature_columns": ["x", "late", "initial_constant"],
        "target_column": "label",
        "random_seed": 17,
    }
    declared = input_spec()
    declared["column_roles"].update(late="feature", initial_constant="feature")
    config = {
        "inputs": {"path": str(path), "spec": declared},
        "output": str(tmp_path / "rolling"),
        "market_id": "a_share",
        "models": [
            {
                **shared,
                "model_id": "lasso",
                "model_algorithm": "lasso_regression",
                "hyperparameters": {"max_iter": None, "tol": 1e-8},
            },
            {
                **shared,
                "model_id": "xgb",
                "model_algorithm": "xgboost_regressor",
                "hyperparameters": {
                    "n_estimators": None,
                    "early_stopping_rounds": 1,
                    "device": "cpu",
                    "n_jobs": 1,
                    "learning_rate": 0.3,
                },
            },
            *[
                {
                    **shared,
                    "model_id": name,
                    "model_algorithm": algorithm,
                    "hyperparameters": {
                        "epochs": None,
                        "early_stopping_rounds": 1,
                        "device": "cpu",
                        "hidden_size": 4,
                        "sequence_length": 2,
                        "batch_size": 256,
                        "learning_rate": 0.05,
                        **({"n_heads": 2} if name == "transformer" else {}),
                    },
                }
                for name, algorithm in (
                    ("lstm", "lstm_regressor"),
                    ("transformer", "transformer_regressor"),
                )
            ],
        ],
        "parameter_candidates": {
            "lasso": [{"alpha": 1e-6}, {"alpha": 0.1}],
            "xgb": [{"max_depth": 1}, {"max_depth": 2}],
            "lstm": [{}, {"sequence_length": 3}],
            "transformer": [{}, {"sequence_length": 3}],
        },
        "training_start": days[0].date().isoformat(),
        "sessions": days.strftime("%Y-%m-%d").tolist(),
        "rolling_window": {
            "start": days[0].date().isoformat(),
            "end": days[89].date().isoformat(),
        },
        "evaluation_cutoff": (days[-1] + pd.Timedelta(days=3)).date().isoformat(),
        "label_unit": "decimal_return",
        "evaluation": {
            "evaluation_id": "session-models",
            "market_id": "a_share",
            "research_dataset_id": "toy",
            "signals": [
                {
                    "signal_id": "score",
                    "source_kind": "prediction",
                    "source_column": "score",
                }
            ],
            "label_columns": ["label"],
            "evaluation_axis": "cross_sectional",
        },
    }
    import quant_research_platform.research.models.production as production

    native_fit = production._fit_prepared
    sequence_scores = {}

    def fit_and_check_sequence(*args, **kwargs):
        result = native_fit(*args, **kwargs)
        spec = args[1]
        if (
            spec.model_algorithm not in {"lstm_regressor", "transformer_regressor"}
            or kwargs.get("validation_cutoff") is None
        ):
            return result
        import torch

        from quant_research_platform.research.models.neural import _network

        predictor = result.prediction_model
        validation_rows = kwargs["validation"].row_counts
        validation = frame.loc[
            frame.timestamp.between(
                validation_rows["train_start"], validation_rows["train_end"]
            )
            & frame.label_available_at.le(pd.Timestamp(kwargs["validation_cutoff"]))
        ]
        # Build each history directly, including observations before validation.
        length = predictor.parameters["sequence_length"]
        arrays = []
        for row in validation.itertuples():
            history = frame.loc[
                frame.asset_id.eq(row.asset_id) & frame.timestamp.le(row.timestamp),
                ["x"],
            ].tail(length)
            arrays.append(predictor.scaler.transform(history).to_numpy(dtype="float32"))
        with torch.random.fork_rng(), torch.inference_mode():
            network = _network(spec.model_algorithm, 1, predictor.parameters).eval()
            network.load_state_dict(predictor.state)
            values = network(
                torch.as_tensor(np.stack(arrays)), torch.full((len(arrays),), length)
            )
        values = (
            values.numpy().astype(float) * predictor.target_scale
            + predictor.target_mean
        )
        errors = pd.Series(
            (values - validation.label.to_numpy()) ** 2, index=validation.index
        )
        sequence_scores.setdefault(spec.model_id, []).append(
            errors.groupby(validation.timestamp.dt.normalize()).mean().mean()
        )
        best = result.metadata["early_stopping"]["best_iteration"]
        assert result.metadata["learning_curve"]["epochs"][best][
            "validation_loss"
        ] == pytest.approx(
            errors.mean(),
            rel=2e-5,
            abs=1e-10,
        )
        return result

    monkeypatch.setattr(production, "_fit_prepared", fit_and_check_sequence)
    results = run_sessions(config)
    assert len(results) == 4
    frozen = []
    for item in results:
        model, facts = load_model(item["model"])
        selection = facts["training_selection"]
        assert selection["validation_rule"] == "latest_mature_signal_sessions"
        assert selection["inputs"] == {
            "retained": ["x"],
            "all_missing": ["late"],
            "constant": ["initial_constant"],
        }
        assert model.metadata["input_columns"] == ["x"]
        assert selection["common_rows"] == 59
        assert selection["common_sessions"] == 20
        assert model.metadata["rows_after_input_label_dropna"] == 239
        assert pd.Timestamp(selection["label_available_max"]) <= pd.Timestamp(
            selection["validation_cutoff"]
        )
        assert facts["fit_cutoff"] == selection["validation_cutoff"]
        parameters = facts["model"]["hyperparameters"]
        if facts["model"]["model_algorithm"] == "lasso_regression":
            assert parameters["alpha"] == 0.1
            assert parameters["max_iter"] is None
            # The high-alpha constant has MSE mean((0.02*x)^2), with one x=-1 omitted.
            assert selection["candidates"][1]["validation_daily_mse"] == pytest.approx(
                np.mean([0.0004 * 2 / 3] * 19 + [0.0004 / 2]), rel=1e-10, abs=1e-14
            )
        else:
            assert "early_stopping_rounds" not in parameters
            winner = min(
                selection["candidates"], key=lambda row: row["validation_daily_mse"]
            )
            budget = (
                "n_estimators"
                if facts["model"]["model_algorithm"] == "xgboost_regressor"
                else "epochs"
            )
            assert parameters[budget] == winner["early_stopping"]["best_iteration"] + 1
            if facts["model"]["model_id"] in sequence_scores:
                for candidate, expected in zip(
                    selection["candidates"],
                    sequence_scores[facts["model"]["model_id"]],
                    strict=True,
                ):
                    assert candidate["validation_daily_mse"] == pytest.approx(
                        expected, rel=2e-5, abs=1e-10
                    )
        predictions, _ = read_predictions(item["predictions"])
        assert len(predictions) == 30
        assert predictions.timestamp.min() > pd.Timestamp(facts["fit_cutoff"])
        frozen.append((selection, predictions))
    frame.loc[immature | frame.timestamp.ge(days[80]), "label"] = 0.08
    frame.loc[frame.timestamp.ge(days[60]), ["late", "initial_constant"]] *= 3
    frame.to_parquet(path, index=False)
    repeated = run_sessions(
        {**deepcopy(config), "output": str(tmp_path / "changed-future")}
    )
    for item, (selection, predictions) in zip(repeated, frozen, strict=True):
        _, facts = load_model(item["model"])
        assert facts["training_selection"] == selection
        pd.testing.assert_frame_equal(
            read_predictions(item["predictions"])[0], predictions
        )
    later = deepcopy(config)
    later["output"] = str(tmp_path / "expanding-inputs")
    later["models"] = later["models"][:1]
    later["parameter_candidates"] = {"lasso": [{"alpha": 0.1}]}
    later["rolling_window"]["end"] = days[-1].date().isoformat()
    expanded = run_sessions(later)
    assert len(expanded) == 2
    _, facts = load_model(expanded[1]["model"])
    assert facts["training_selection"]["inputs"] == {
        "retained": shared["input_feature_columns"],
        "all_missing": [],
        "constant": [],
    }


@pytest.mark.parametrize("eligibility", ["absent", "false"])
def test_unfiltered_native_fit_and_mature_rolling_ignore_eligibility(
    tmp_path, eligibility
):
    from quant_research_platform.contracts.models import _model_spec
    from quant_research_platform.research.models.production import fit, fit_model

    path = data(tmp_path)
    frame = pd.read_parquet(path)
    declared = input_spec()
    if eligibility == "absent":
        frame = frame.drop(columns="eligible")
        declared["column_roles"].pop("eligible")
    else:
        frame["eligible"] = False
    frame.to_parquet(path, index=False)
    native = fit_config(tmp_path, path)
    native["inputs"]["spec"] = declared
    native["model"]["sample_filter_mode"] = "unfiltered"
    candidate = fit_model(
        [frame],
        _model_spec(native["model"]),
        fit_windows=native["fit_windows"],
        fit_cutoff=native["fit_cutoff"],
    )
    assert candidate.metadata["rows_after_input_label_dropna"] == 9
    model, _ = load_model(fit(native))
    assert model.metadata["rows_after_input_label_dropna"] == 9

    days = pd.bdate_range("2022-01-03", periods=90)
    frame = pd.DataFrame(
        [
            (day, asset, value, 0.02 * value + 0.005, day + pd.offsets.BDay(20))
            for day in days
            for asset, value in zip(("a", "b", "c"), (-1.0, 0.0, 1.0), strict=True)
        ],
        columns=["timestamp", "asset_id", "x", "label", "label_available_at"],
    )
    if eligibility == "false":
        frame["eligible"] = False
    frame.to_parquet(path, index=False)
    config = {
        "inputs": {"path": str(path), "spec": declared},
        "output": str(tmp_path / "mature-rolling"),
        "market_id": "a_share",
        "models": [native["model"]],
        "training_start": days[0].date().isoformat(),
        "sessions": days.strftime("%Y-%m-%d").tolist(),
        "rolling_window": {
            "start": days[0].date().isoformat(),
            "end": days[-1].date().isoformat(),
        },
        "evaluation_cutoff": (days[-1] + pd.offsets.BDay(20)).date().isoformat(),
        "label_unit": "decimal_return",
        "evaluation": {
            "evaluation_id": "mature",
            "market_id": "a_share",
            "research_dataset_id": "synthetic",
            "signals": [
                {
                    "signal_id": "score",
                    "source_kind": "prediction",
                    "source_column": "score",
                }
            ],
            "label_columns": ["label"],
            "evaluation_axis": "cross_sectional",
            "filter_mode": "unfiltered",
        },
    }
    if eligibility == "false":
        config["models"].append(
            {
                **native["model"],
                "model_id": "gap_lstm",
                "model_algorithm": "lstm_regressor",
                "hyperparameters": {
                    "epochs": 1,
                    "device": "cpu",
                    "hidden_size": 4,
                    "sequence_length": 2,
                    "batch_size": 256,
                },
            }
        )
    results = run_sessions(config)
    assert len(results) == len(config["models"])
    model, facts = load_model(results[0]["model"])
    selection = facts["training_selection"]
    assert selection["validation_window"] == {
        "start": days[40].date().isoformat(),
        "end": days[59].date().isoformat(),
    }
    assert selection["fit_window"]["end"] == days[39].date().isoformat()
    assert selection["common_sessions"] == 20
    assert selection["common_rows"] == 60
    assert facts["fit_windows"][0]["end"] == days[79].date().isoformat()
    assert model.metadata["rows_input"] == 240
    assert model.metadata["rows_after_input_label_dropna"] == 180
    predictions, _ = read_predictions(results[0]["predictions"])
    assert len(predictions) == 30
    np.testing.assert_allclose(
        predictions.prediction_score,
        np.tile((-0.015, 0.005, 0.025), 10),
        rtol=1e-12,
        atol=1e-14,
    )
    if eligibility == "false":
        sequence, _ = load_model(results[1]["model"])
        assert set(sequence.prediction_model.history.timestamp) == {days[79]}
        assert sequence.metadata["rows_after_input_label_dropna"] == 180


def test_candidate_context_and_validation_changes_keep_fresh_fit_semantics(tmp_path):
    from dataclasses import replace

    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.models.fitting import prediction_callable
    from quant_research_platform.research.models.production import fit_model
    from quant_research_platform.research.rolling import RollingFold, _select
    from quant_research_platform.storage.datasets.reading import (
        _dataset_input_options,
        open_dataset_session,
    )

    path = data(tmp_path)
    frame = pd.read_parquet(path)
    frame["episode"] = np.where(frame.timestamp.dt.day.le(3), "train", "held")
    frame.loc[frame.timestamp.dt.day.isin([2, 5]) & frame.asset_id.eq("b"), "x"] = (
        np.nan
    )
    frame.to_parquet(path, index=False)
    declared = input_spec()
    declared["column_roles"]["episode"] = "context"
    session = open_dataset_session(
        **_dataset_input_options({"path": str(path), "spec": declared})
    )
    fold = RollingFold(
        "test",
        TimeWindow("2020-01-01", "2020-01-03"),
        TimeWindow("2020-01-04", "2020-01-06"),
        TimeWindow("2020-01-07", "2020-01-09"),
    )
    base = ModelSpec(
        model_id="lstm",
        model_algorithm="lstm_regressor",
        input_feature_columns=("x",),
        target_column="label",
        random_seed=17,
    )
    defaults = dict(
        device="cpu", epochs=1, hidden_size=3, sequence_length=3, batch_size=32
    )
    candidates = [
        (
            "lstm",
            replace(base, hyperparameters={**defaults, **parameters}),
        )
        for parameters in (
            {"sequence_group_columns": ["episode"], "learning_rate": 0.01},
            {"sequence_group_columns": [], "learning_rate": 0.02},
            {"sequence_group_columns": ["episode"], "learning_rate": 0.03},
            {"sequence_group_columns": ["episode"], "learning_rate": 0.035},
            {
                "sequence_group_columns": ["episode"],
                "learning_rate": 0.04,
                "epochs": 3,
                "early_stopping_rounds": 1,
            },
        )
    ]
    selected = _select({"market_id": "a_share"}, fold, session, candidates, None)
    scores = selected[0][2]["candidates"]
    held = frame.loc[frame.timestamp.between("2020-01-04", "2020-01-06")]
    scoring = frame.loc[frame.timestamp.between("2020-01-03", "2020-01-06")]
    cutoff = pd.Timestamp("2020-01-07") - pd.Timedelta(1, "ns")
    for (_, spec), score in zip(candidates, scores, strict=True):
        validation = (
            {
                "validation_frames": [scoring],
                "validation_windows": (fold.validation,),
                "validation_cutoff": cutoff,
            }
            if spec.hyperparameters.get("early_stopping_rounds") is not None
            else {}
        )
        fresh = fit_model(
            [frame],
            spec,
            fit_windows=(fold.train,),
            fit_cutoff=pd.Timestamp("2020-01-04") - pd.Timedelta(1, "ns"),
            **validation,
        )
        predictions = prediction_callable(spec.input_columns, fresh.prediction_model)(
            scoring
        )[-len(held) :]
        observed = held.x.notna().to_numpy()
        errors = pd.Series(
            (predictions[observed] - held.label.to_numpy()[observed]) ** 2,
            index=held.timestamp.loc[observed],
        )
        assert score["hyperparameters"] == dict(spec.hyperparameters)
        assert score["validation_daily_mse"] == pytest.approx(
            errors.groupby(level=0).mean().mean(), rel=1e-12, abs=1e-14
        )
        if validation:
            assert score["early_stopping"] == fresh.metadata["early_stopping"]
        else:
            assert "early_stopping" not in score
    assert selected[0][2]["common_rows"] == 8


def test_quarterly_skips_immature_initial_history_and_continues(tmp_path):
    path = data(tmp_path)
    days = pd.bdate_range("2014-01-01", "2014-09-30")
    frame = pd.DataFrame(
        [
            (day, asset, value, 0.02 * value, True, day + pd.offsets.BDay(40))
            for day in days
            for asset, value in zip(("a", "b", "c"), (-1.0, 0.0, 1.0), strict=True)
        ],
        columns=[
            "timestamp",
            "asset_id",
            "x",
            "label",
            "eligible",
            "label_available_at",
        ],
    )
    frame.to_parquet(path, index=False)
    results = run_quarterly(
        {
            "inputs": {"path": str(path), "spec": input_spec()},
            "output": str(tmp_path / "quarters"),
            "market_id": "a_share",
            "models": [fit_config(tmp_path, path)["model"]],
            "training_start": "2014-01-01",
            "sessions": days.strftime("%Y-%m-%d").tolist(),
            "rolling_window": {"start": "2014-01-01", "end": "2014-09-30"},
            "evaluation_cutoff": "2015-01-01",
            "label_unit": "decimal_return",
            "evaluation": {
                "evaluation_id": "quarters",
                "market_id": "a_share",
                "research_dataset_id": "synthetic",
                "signals": [
                    {
                        "signal_id": "score",
                        "source_kind": "prediction",
                        "source_column": "score",
                    }
                ],
                "label_columns": ["label"],
                "evaluation_axis": "cross_sectional",
            },
        }
    )
    assert results[0] == {"fold_id": "2014Q2", "reason": "insufficient_mature_history"}
    assert results[1]["fold_id"] == "2014Q3"
    model, facts = load_model(results[1]["model"])
    assert model.metadata["rows_after_input_label_dropna"] > 0
    assert facts["fit_windows"] == [{"start": "2014-01-01", "end": "2014-06-30"}]
    assert read_predictions(results[1]["predictions"])[
        0
    ].timestamp.min() == pd.Timestamp("2014-07-01")

"""Physical member order must not change time/cross-section research semantics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.correctness.test_direct_model_operations import fit_config, input_spec
from tests.research.evaluation_test_support import evaluation_config
from tests.storage.low_level_fixtures import make_processed_dataset_spec


def model_data(root, *, ordered):
    frame = pd.DataFrame(
        [
            (t, a, float(i + t.day), float(2 * (i + t.day) + 1), True, t)
            for t in pd.date_range("2020-01-01", periods=9)
            for i, a in enumerate("abc")
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
    declared = input_spec()
    declared["primary_key"] = tuple(declared["primary_key"])
    declared["partition_columns"] = tuple(declared["partition_columns"])
    spec = DataSpec(**declared)
    chunks = (
        [frame] if ordered else [g.iloc[::-1] for _, g in frame.groupby("asset_id")]
    )
    write_dataset(root, iter(chunks), spec=spec)
    return root


@pytest.mark.parametrize("algorithm", ["linear_regression", "lstm_regressor"])
def test_native_unordered_fit_predict_preserves_scores_and_cross_section_ranks(
    tmp_path, algorithm
):
    results = []
    for ordered in (True, False):
        root = tmp_path / str(ordered)
        inputs = model_data(root / "data", ordered=ordered)
        config = fit_config(root, inputs)
        config["inputs"].pop("spec")
        config["model"]["model_algorithm"] = algorithm
        if algorithm == "lstm_regressor":
            config["model"]["hyperparameters"] = {
                "device": "cpu",
                "epochs": 1,
                "hidden_size": 4,
                "sequence_length": 3,
            }
        trained = fit(config)
        prediction = predict(
            {
                "model": str(trained),
                "inputs": str(inputs),
                "output": str(root / "pred"),
                "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
            }
        )
        rows = (
            read_predictions(prediction)[0]
            .sort_values(["timestamp", "asset_id"])
            .reset_index(drop=True)
        )
        results.append(rows)
        assert len(rows) == 9
        if algorithm == "linear_regression":
            np.testing.assert_allclose(
                rows.prediction_score,
                [2 * (i + day) + 1 for day in (4, 5, 6) for i in range(3)],
                atol=1e-10,
            )
            np.testing.assert_allclose(
                rows.prediction_rank_score, [-1 / 3, 0, 1 / 3] * 3, atol=1e-15
            )
    pd.testing.assert_frame_equal(*results, check_exact=False, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("filter_mode", ["eligible", "unfiltered"])
def test_native_unordered_evaluation_keeps_full_cross_sections(tmp_path, filter_mode):
    root = model_data(tmp_path / "data", ordered=False)
    spec = EvaluationSpec(
        evaluation_id="unordered",
        market_id="a_share",
        research_dataset_id="synthetic",
        signals=(SignalSpec(signal_id="x", source_kind="feature", source_column="x"),),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        n_groups=2,
        filter_mode=filter_mode,
    )
    result = evaluate(
        evaluation_config(
            root, tmp_path / "eval", spec, start="2020-01-04", end="2020-01-06"
        )
    )
    _, tables = read_evaluation_outputs(result, tables=("metrics",))
    assert tables["metrics"].n_obs.tolist() == [3, 3, 3]
    np.testing.assert_allclose(tables["metrics"].ic, 1.0, atol=1e-14)


@pytest.mark.parametrize("sparse_base", [False, True])
def test_native_unordered_base_and_auxiliary_assemble(
    tmp_path, sparse_base, monkeypatch
):
    frame = pd.DataFrame(
        {
            "market_id": "custom",
            "asset_id": list("abc") * 4,
            "timestamp": np.repeat(
                pd.date_range("2024-01-01", periods=4).as_unit("ns"), 3
            ),
            "factor": np.arange(12, dtype=float),
        }
    )
    spec = make_processed_dataset_spec(
        "base", market_id="custom", value_column="factor"
    )
    aux = frame.rename(columns={"factor": "other"}).assign(
        other=lambda x: x.other + 100
    )
    base = frame.iloc[[0, 4, 8, 11]] if sparse_base else frame
    for name, values, declared in (
        ("base", base, spec),
        (
            "aux",
            aux,
            replace(
                spec,
                data_id="aux",
                column_roles={
                    **{k: spec.column_roles[k] for k in spec.primary_key},
                    "other": "factor",
                },
            ),
        ),
    ):
        write_dataset(
            tmp_path / name,
            (g.iloc[::-1] for _, g in values.groupby("asset_id")),
            spec=declared,
        )
    from quant_research_platform.research.datasets import assembler

    original = assembler._join_processed_inputs

    def checked_join(**kwargs):
        keys = ["market_id", "asset_id", "timestamp"]
        base_keys = set(kwargs["base_frame"][keys].itertuples(index=False, name=None))
        for auxiliary in kwargs["input_frames"].values():
            assert set(auxiliary[keys].itertuples(index=False, name=None)) <= base_keys
        return original(**kwargs)

    monkeypatch.setattr(assembler, "_join_processed_inputs", checked_join)
    root = assemble(
        {
            "market_id": "custom",
            "dataset_id": "joined",
            "inputs": {name: str(tmp_path / name) for name in ("base", "aux")},
            "base_input_data_id": "base",
            "factor_columns": ["factor", "other"],
            "label_columns": [],
            "feature_recipe": None,
            "time_window": {"start": "2024-01-02", "end": "2024-01-03"},
            "output_root": str(tmp_path / "joined"),
        }
    )
    actual = (
        read_dataset(root).sort_values(["timestamp", "asset_id"]).reset_index(drop=True)
    )
    expected = (
        base.loc[base.timestamp.between("2024-01-02", "2024-01-03")]
        .assign(other=lambda x: x.factor + 100)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_unordered_prediction_on_independently_fitted_native_model(tmp_path):
    train = model_data(tmp_path / "train-data", ordered=True)
    inputs = model_data(tmp_path / "unordered-data", ordered=False)
    config = fit_config(tmp_path, train)
    config["inputs"].pop("spec")
    trained = fit(config)
    result = predict(
        {
            "model": str(trained),
            "inputs": str(inputs),
            "output": str(tmp_path / "pred"),
            "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
        }
    )
    rows = read_predictions(result)[0].sort_values(["timestamp", "asset_id"])
    np.testing.assert_allclose(
        rows.prediction_score,
        [2 * (i + day) + 1 for day in (4, 5, 6) for i in range(3)],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        rows.prediction_rank_score, [-1 / 3, 0, 1 / 3] * 3, atol=1e-15
    )

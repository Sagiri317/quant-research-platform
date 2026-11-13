import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.rolling import quarterly_folds, run_quarterly
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def test_quarterly_windows_expand_and_first_quarter_supplies_initial_training():
    folds = quarterly_folds("2014-01-01", {"start": "2018-01-01", "end": "2023-12-31"})
    assert len(folds) == 24
    assert str(folds[0].train.end) == "2017-12-31"
    assert str(folds[-1].train.end) == "2023-09-30"
    assert all(str(fold.train.start) == "2014-01-01" for fold in folds)
    same = quarterly_folds("2014-01-01", {"start": "2014-01-01", "end": "2014-12-31"})
    assert len(same) == 3
    assert str(same[0].validation.start) == "2014-04-01"


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

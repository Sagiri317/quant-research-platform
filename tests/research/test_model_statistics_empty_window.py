"""Empty input windows fail clearly without changing sparse observation counts."""

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.research import statistics_production
from tests.research.test_prediction_runner_model_selection import candidates


def _config(tmp_path, window):
    _, scoring = candidates(tmp_path, algorithms=("linear_regression",))
    return {
        "dataset": scoring["labels"],
        "input_spec": scoring["label_spec"],
        "output": str(tmp_path / "statistics"),
        "window": window,
        "family_ids": scoring["candidate_ids"],
        "evaluation": {"primary_label": "label"},
        "analyses": ["prediction_hac"],
        "statistics": {"hac_lag": 0},
        "periods": [
            {
                "period_id": "explicit",
                "window": window,
                "predictions": scoring["predictions"],
            }
        ],
    }


@pytest.mark.parametrize("empty_stream", [False, True])
def test_model_statistics_rejects_window_without_observations(
    tmp_path, monkeypatch, empty_stream
):
    config = _config(tmp_path, {"start": "2020-01-07", "end": "2020-01-08"})
    if empty_stream:
        monkeypatch.setattr(
            statistics_production, "iter_dataset_chunks", lambda session: iter(())
        )
    with pytest.raises(ValueError, match="statistics_model_observations_empty"):
        statistics_production.model_statistics(config)
    assert not Path(config["output"]).exists()


def test_model_statistics_keeps_sparse_windows_and_skips_empty_frames(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, {"start": "2020-01-04", "end": "2020-01-08"})
    config["periods"][0]["window"] = {"start": "2020-01-04", "end": "2020-01-06"}
    config["periods"].append(
        {
            **config["periods"][0],
            "period_id": "empty",
            "window": {"start": "2020-01-07", "end": "2020-01-08"},
        }
    )
    original = statistics_production.iter_complete_session_frames

    def with_empty_frames(*args, **kwargs):
        for frame in original(*args, **kwargs):
            yield frame.iloc[:0]
            yield frame
            yield frame.iloc[:0]

    monkeypatch.setattr(
        statistics_production, "iter_complete_session_frames", with_empty_frames
    )
    saved = read_statistics_artifact(statistics_production.model_statistics(config))
    with saved.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics.input_date_count.tolist() == [3]
        assert diagnostics.observation_count.tolist() == [3]
        assert diagnostics["mean"].tolist() == pytest.approx([1.0])


def test_model_statistics_preserves_observed_dates_with_undefined_labels(tmp_path):
    config = _config(tmp_path, {"start": "2020-01-04", "end": "2020-01-06"})
    config["label_cutoff"] = "2020-01-03"
    saved = read_statistics_artifact(statistics_production.model_statistics(config))
    with saved.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics.input_date_count.tolist() == [3]
        assert diagnostics.observation_count.tolist() == [0]
        assert diagnostics.status.tolist() == ["UNDEFINED"]
        assert pd.isna(diagnostics["mean"]).all()

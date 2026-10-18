"""Empty input windows fail clearly without changing sparse observation counts."""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.research import statistics_production
from tests.research.direct_model_test_support import candidates


@pytest.fixture(scope="module")
def saved_candidates(tmp_path_factory):
    return candidates(tmp_path_factory.mktemp("empty-window-predictions"))[1]


def _config(
    tmp_path,
    window,
    saved_candidates,
    *,
    algorithms=("linear_regression",),
    constant=False,
):
    scoring = deepcopy(saved_candidates)
    scoring["predictions"] = [
        path
        for family, path in zip(
            scoring["candidate_ids"], scoring["predictions"], strict=True
        )
        if family in algorithms
    ]
    scoring["candidate_ids"] = list(algorithms)
    if constant:
        labels = pd.read_parquet(scoring["labels"])
        labels.loc[labels.timestamp.between("2020-01-04", "2020-01-06"), "label"] = 0.2
        consumed = labels.loc[
            labels.timestamp.between("2020-01-04", "2020-01-06")
            & labels.asset_id.isin(["b", "c"])
            & labels.eligible
            & labels.label_available_at.le(pd.Timestamp("2020-01-06"))
        ]
        assert consumed.groupby("timestamp").size().tolist() == [2, 2, 2]
        assert consumed.groupby("timestamp").label.nunique().tolist() == [1, 1, 1]
        scoring["labels"] = str(tmp_path / "constant-labels.parquet")
        labels.to_parquet(scoring["labels"], index=False)
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
    tmp_path, monkeypatch, empty_stream, saved_candidates
):
    config = _config(
        tmp_path, {"start": "2020-01-07", "end": "2020-01-08"}, saved_candidates
    )
    if empty_stream:
        monkeypatch.setattr(
            statistics_production, "iter_dataset_chunks", lambda session: iter(())
        )
    with pytest.raises(ValueError, match="statistics_model_observations_empty"):
        statistics_production.model_statistics(config)
    assert not Path(config["output"]).exists()


def test_model_statistics_keeps_sparse_windows_and_skips_empty_frames(
    tmp_path, monkeypatch, saved_candidates
):
    config = _config(
        tmp_path, {"start": "2020-01-04", "end": "2020-01-08"}, saved_candidates
    )
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


def test_model_statistics_preserves_observed_dates_with_undefined_labels(
    tmp_path, saved_candidates
):
    config = _config(
        tmp_path, {"start": "2020-01-04", "end": "2020-01-06"}, saved_candidates
    )
    config["label_cutoff"] = "2020-01-03"
    saved = read_statistics_artifact(statistics_production.model_statistics(config))
    with saved.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics.input_date_count.tolist() == [3]
        assert diagnostics.observation_count.tolist() == [0]
        assert diagnostics.status.tolist() == ["UNDEFINED"]
        assert pd.isna(diagnostics["mean"]).all()


def test_model_statistics_keeps_constant_labels_explicitly_undefined(
    tmp_path, saved_candidates
):
    config = _config(
        tmp_path,
        {"start": "2020-01-04", "end": "2020-01-06"},
        saved_candidates,
        constant=True,
    )
    saved = read_statistics_artifact(statistics_production.model_statistics(config))
    with saved.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics.input_date_count.tolist() == [3]
        assert diagnostics.observation_count.tolist() == [0]
        assert diagnostics.status.tolist() == ["UNDEFINED"]
        assert diagnostics.reason.tolist() == ["incomplete_session_clock"]
        assert pd.isna(diagnostics["mean"]).all()


@pytest.mark.parametrize(
    "analysis,policy",
    [
        ("prediction_hac", {"hac_lag": -1}),
        ("prediction_hac", {"hac_lag": True}),
        ("prediction_hac", {"hac_lag": 0.5}),
        ("prediction_bootstrap", {"bootstrap_block": 0}),
        ("prediction_bootstrap", {"bootstrap_block": True}),
        ("prediction_bootstrap", {"bootstrap_block": 1, "bootstrap_repetitions": 1}),
    ],
)
def test_invalid_requested_configuration_precedes_input_read(
    tmp_path, analysis, policy
):
    config = {
        "dataset": str(tmp_path / "missing-input"),
        "output": str(tmp_path / "statistics"),
        "window": {"start": "2020-01-04", "end": "2020-01-06"},
        "evaluation": {"primary_label": "label"},
        "analyses": [analysis],
        "statistics": policy,
        "periods": [],
    }
    with pytest.raises(ValueError, match="model_statistics_policy_invalid"):
        statistics_production.model_statistics(config)
    assert not Path(config["output"]).exists()


def test_prediction_only_skips_unrequested_comparison_and_domain_work(
    tmp_path, monkeypatch, saved_candidates
):
    config = _config(
        tmp_path, {"start": "2020-01-04", "end": "2020-01-06"}, saved_candidates
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("unrequested common rank, domain hash, or comparison")

    for name in (
        "_common_model_rank_ic",
        "canonical_json_hash",
        "_model_comparison",
        "_paired_group_rankic",
    ):
        monkeypatch.setattr(statistics_production, name, unexpected)
    original = pd.DataFrame.from_records

    def records(*args, **kwargs):
        assert "asset_coverage_id" not in kwargs.get("columns", ())
        return original(*args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "from_records", records)
    artifact = read_statistics_artifact(statistics_production.model_statistics(config))
    assert set(artifact._members) == {"prediction_diagnostics.parquet"}


def test_requested_comparison_keeps_common_rank_without_unrequested_loss_hashes(
    tmp_path, monkeypatch, saved_candidates
):
    config = _config(
        tmp_path,
        {"start": "2020-01-04", "end": "2020-01-06"},
        saved_candidates,
        algorithms=("linear_regression", "ic_weighted_rank_ensemble"),
    )
    config.update(
        analyses=["prediction_hac", "model_comparison"],
        comparison_model_id="linear_regression",
    )
    config["statistics"].update(bootstrap_block=1, bootstrap_repetitions=10)
    original = statistics_production._common_model_rank_ic
    calls = []

    def common(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def unexpected(*args, **kwargs):
        raise AssertionError("unrequested loss domain hash")

    monkeypatch.setattr(statistics_production, "_common_model_rank_ic", common)
    monkeypatch.setattr(statistics_production, "canonical_json_hash", unexpected)
    artifact = read_statistics_artifact(statistics_production.model_statistics(config))
    assert len(calls) == 3
    assert set(artifact._members) == {
        "prediction_diagnostics.parquet",
        "model_comparison.parquet",
    }
    with artifact.table("model_comparison.parquet") as saved:
        assert saved.observation_count.tolist() == [3]
        assert saved["mean"].tolist() == [0.0]
        assert saved.reason.tolist() == ["hac_inference_undefined"]
        assert saved.bootstrap_status.tolist() == ["DEFINED"]

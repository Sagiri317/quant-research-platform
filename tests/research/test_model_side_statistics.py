"""Public prediction statistics retain fixed candidate sides and legacy losses."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.statistics import model_statistics
from quant_research_platform.research.predictions.runner import predict
from tests.research.direct_model_test_support import fit_frame


@pytest.fixture(scope="module")
def prediction_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("model-side-statistics")
    assets = list("abcdefghijkl")
    scores = [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 200.0]
    train_dates = pd.date_range("2020-01-01", periods=2).repeat(len(assets))
    train = pd.DataFrame(
        {
            "timestamp": train_dates,
            "asset_id": assets * 2,
            "x": scores * 2,
            "label": np.tile(scores, 2) / 100,
            "eligible": True,
            "label_available_at": train_dates,
        }
    )
    fitted = fit_frame(
        root / "fit",
        train,
        ModelSpec(
            model_id="timestamp",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
            sample_filter_mode="unfiltered",
            random_seed=1,
        ),
    )
    dates = pd.date_range("2020-01-04", periods=4).repeat(len(assets))
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "asset_id": assets * 4,
            "x": scores * 4,
            "label": [99.0, 0.0, 3.0, 1.0, 2.0, 0.0, 0.0, 3.0, 1.0, 99.0, 500.0, 600.0]
            * 4,
            "eligible": True,
            "label_available_at": dates,
        }
    )
    frame.loc[frame.asset_id.eq("a"), "x"] = np.nan
    frame.loc[frame.asset_id.eq("k"), "eligible"] = False
    frame.loc[frame.asset_id.eq("j"), "label_available_at"] = pd.Timestamp("2020-01-20")
    source = root / "input.parquet"
    frame.to_parquet(source, index=False)
    window = {"start": "2020-01-04", "end": "2020-01-07"}
    prediction = predict(
        {
            "model": str(fitted.root),
            "inputs": {"path": str(source), "spec": fitted.config["inputs"]["spec"]},
            "asset_ids": list("abcdefghijk"),
            "prediction_window": window,
            "output": str(root / "prediction"),
        }
    )
    return {
        "dataset": {"path": str(source), "spec": fitted.config["inputs"]["spec"]},
        "window": window,
        "family_ids": ["timestamp"],
        "evaluation": {
            "primary_label": "label",
            "raw_return_label": "label",
            "head_fraction": 0.4,
        },
        "analyses": ["prediction_hac", "prediction_bootstrap", "model_losses"],
        "statistics": {
            "hac_lag": 0,
            "bootstrap_block": 1,
            "bootstrap_repetitions": 19,
        },
        "periods": [
            {"period_id": "period", "window": window, "predictions": [str(prediction)]}
        ],
    }


def _run(tmp_path, prediction_case, *, quantile=None):
    config = deepcopy(prediction_case)
    config["output"] = str(tmp_path / "statistics")
    if quantile is not None:
        config["evaluation"]["top_quantile"] = quantile
    return read_statistics_artifact(model_statistics(config))


def test_model_side_summaries_use_candidates_before_maturity_and_keep_head_ceil(
    tmp_path, prediction_case
):
    artifact = _run(tmp_path, prediction_case, quantile=0.4)
    assert artifact.metadata["evaluation"]["top_quantile"] == 0.4
    with artifact.table("prediction_diagnostics.parquet") as rows:
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row.prediction_id == "timestamp"
        assert row.input_date_count == 4
        # Nine eligible predictions: h/i/j are long, with immature j removed only
        # after selection. b/c/d are short. a has no PRED; k/l are outside the pool.
        assert row.long_ic_mean == pytest.approx(-1.0)
        assert row.long_rankic_mean == pytest.approx(-1.0)
        assert row.short_ic_mean == pytest.approx(1 / 7)
        assert row.short_rankic_mean == pytest.approx(0.5)
        assert row.long_ic_observation_count == row.short_ic_observation_count == 4
        assert row.long_ic_bootstrap_status == "DEFINED"
        assert row.long_ic_bootstrap_lower == pytest.approx(-1.0)
        assert row.long_ic_bootstrap_upper == pytest.approx(-1.0)
        # Independent average-rank vector for paired b..i: [2,7.5,4.5,6,2,2,7.5,4.5].
        expected_rankic = 3 / np.sqrt(42 * 39)
        assert row["mean"] == pytest.approx(expected_rankic)
    with artifact.table("model_losses.parquet") as rows:
        assert rows.observation_count.eq(8).all()
        assert rows.loss.tolist() == pytest.approx([-expected_rankic] * 4)
    with artifact.table("model_prediction_metrics.parquet") as rows:
        assert rows.prediction_count.eq(9).all()
        assert rows.target_count.eq(8).all()
        # Existing head-return semantics use ceil(9 * .4) = 4, not side IC's floor.
        assert rows.head_selected_count.eq(4).all()
        assert rows.head_evaluable_count.eq(3).all()
        assert rows.head_mean_raw_return.tolist() == pytest.approx([4 / 3] * 4)


def test_model_default_quantile_persists_undefined_sides_without_reselection(
    tmp_path, prediction_case
):
    artifact = _run(tmp_path, prediction_case)
    assert artifact.metadata["evaluation"]["top_quantile"] == 0.2
    with artifact.table("prediction_diagnostics.parquet") as rows:
        row = rows.iloc[0]
        assert row.observation_count == row.ic_observation_count == 4
        for metric in ("long_ic", "long_rankic", "short_ic", "short_rankic"):
            assert row[f"{metric}_observation_count"] == 0
            assert np.isnan(row[f"{metric}_mean"])
            assert row[f"{metric}_status"] == "UNDEFINED"
            assert row[f"{metric}_reason"] == "incomplete_session_clock"
            assert row[f"{metric}_bootstrap_status"] == "UNDEFINED"


def test_model_quantile_one_matches_full_correlations(tmp_path, prediction_case):
    artifact = _run(tmp_path, prediction_case, quantile=1)
    with artifact.table("prediction_diagnostics.parquet") as rows:
        row = rows.iloc[0]
        for side in ("long", "short"):
            assert row[f"{side}_ic_mean"] == pytest.approx(row.ic_mean)
            assert row[f"{side}_rankic_mean"] == pytest.approx(row["mean"])
            assert row[f"{side}_ic_observation_count"] == row.ic_observation_count
            assert row[f"{side}_rankic_observation_count"] == row.observation_count


def test_model_side_boundary_ties_are_included_before_label_pairing(
    tmp_path, prediction_case
):
    config = deepcopy(prediction_case)
    predictions, metadata = read_predictions(config["periods"][0]["predictions"][0])
    predictions.loc[predictions.asset_id.eq("d"), "prediction_score"] = predictions.loc[
        predictions.asset_id.eq("c"), "prediction_score"
    ].iloc[0]
    tied = write_prediction_outputs(
        output_path=tmp_path / "tied-predictions",
        output_chunks=[predictions],
        metadata={
            **metadata,
            "prediction_rows": None,
            "effective_prediction_end": None,
        },
    )
    config["periods"][0]["predictions"] = [str(tied)]
    artifact = _run(tmp_path, config, quantile=0.3)
    with artifact.table("prediction_diagnostics.parquet") as rows:
        row = rows.iloc[0]
        # floor(9 * .3) = 2 includes b/c/d because c and d tie at the boundary.
        assert row.short_ic_mean == pytest.approx(2 / np.sqrt(7))
        assert row.short_rankic_mean == pytest.approx(np.sqrt(3) / 2)
        assert row.short_ic_observation_count == 4
        assert row.long_ic_observation_count == 0


def test_model_side_hac_uses_each_metric_series_and_shared_daily_clock(
    tmp_path, prediction_case
):
    config = deepcopy(prediction_case)
    labels = pd.read_parquet(config["dataset"]["path"])
    positive = labels.timestamp.isin(pd.to_datetime(["2020-01-05", "2020-01-07"]))
    labels.loc[positive & labels.asset_id.eq("h"), "label"] = 0.0
    source = tmp_path / "varying-labels.parquet"
    labels.to_parquet(source, index=False)
    config["dataset"]["path"] = str(source)
    artifact = _run(tmp_path, config, quantile=0.4)
    with artifact.table("prediction_diagnostics.parquet") as rows:
        row = rows.iloc[0]
        # Long's four daily correlations are [-1, 1, -1, 1]. HAC(0) gives
        # variance(sum) / n**2 = 4 / 16, while short stays constant at 1/7.
        for metric in ("long_ic", "long_rankic"):
            assert row[f"{metric}_mean"] == pytest.approx(0, abs=1e-14)
            assert row[f"{metric}_status"] == "DEFINED"
            assert row[f"{metric}_standard_error"] == pytest.approx(0.5)
            assert row[f"{metric}_t_statistic"] == pytest.approx(0, abs=1e-14)
            assert row[f"{metric}_p_value"] == pytest.approx(1.0)
        assert row.short_ic_status == "UNDEFINED"
        assert row.short_ic_reason == "hac_inference_undefined"
        assert row.short_ic_bootstrap_status == "DEFINED"


@pytest.mark.parametrize("quantile", [0, -0.2, 1.1, True, "0.2"])
def test_model_quantile_validation_precedes_input_read(tmp_path, quantile):
    with pytest.raises(ValueError, match="top_quantile"):
        model_statistics(
            {
                "dataset": str(tmp_path / "missing"),
                "output": str(tmp_path / "output"),
                "window": {"start": "2020-01-01", "end": "2020-01-02"},
                "periods": [],
                "evaluation": {"primary_label": "label", "top_quantile": quantile},
                "analyses": ["prediction_hac"],
                "statistics": {"hac_lag": 0},
            }
        )

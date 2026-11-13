"""Daily evaluation weighting and saved-prediction common-domain regressions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import (
    read_evaluation_outputs,
    write_evaluation_outputs_from_chunks,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
    score,
)
from quant_research_platform.research.predictions.runner import predict
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def _spec(axis):
    return EvaluationSpec(
        evaluation_id="intraday",
        market_id="a_share",
        research_dataset_id="synthetic",
        signals=(
            SignalSpec(signal_id="score", source_kind="feature", source_column="x"),
        ),
        label_columns=("label",),
        evaluation_axis=axis,
        correlation_clock="trading_day",
        n_groups=2,
    )


def _saved_metrics(tmp_path, frame, spec):
    with build_evaluation_result_from_frames(
        [frame.iloc[:1], frame.iloc[1:5], frame.iloc[5:]],
        spec=spec,
        asset_column="asset_id",
    ) as result:
        output = write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / spec.evaluation_id,
        )
    return read_evaluation_outputs(output)


def test_cross_section_summary_weights_days_not_timestamp_counts(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02 09:30:00", "2024-01-02 09:30:01", "2024-01-03 09:30:00"]
            ).repeat(3),
            "asset_id": list("abc") * 3,
            "x": [1.0, 2.0, 3.0] * 3,
            "label": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "eligible": True,
        }
    )
    metadata, tables = _saved_metrics(tmp_path, frame, _spec("cross_sectional"))
    assert metadata["metadata"]["correlation_clock"] == "trading_day"
    np.testing.assert_allclose(tables["metrics"].ic, [1, 1, -1])
    summary = tables["ic_summary"].iloc[0]
    assert summary.mean_ic == pytest.approx(0)
    assert summary.mean_rankic == pytest.approx(0)
    assert summary.valid_ic_count == summary.valid_rankic_count == 2
    assert summary.std_ic == pytest.approx(1)
    # Legacy artifacts remain readable and retain their explicit observation weighting.
    old_spec = replace(
        _spec("cross_sectional"),
        evaluation_id="legacy",
        correlation_clock="observation",
    )
    old_metadata, old_tables = _saved_metrics(tmp_path, frame, old_spec)
    assert "correlation_clock" not in old_metadata["metadata"]
    assert old_tables["ic_summary"].iloc[0].mean_ic == pytest.approx(1 / 3)


def test_etf_day_rankic_roundtrip_matches_validation_weighting(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:30:01",
                    "2024-01-02 09:30:02",
                    "2024-01-03 09:30:00",
                    "2024-01-03 09:30:01",
                    "2024-01-03 09:30:02",
                ]
            ),
            "asset_id": "etf",
            "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "eligible": True,
        }
    )
    _, tables = _saved_metrics(tmp_path, frame, _spec("single_asset_time_series"))
    np.testing.assert_allclose(tables["metrics"].rankic, [1, -1])
    assert tables["metrics"].timestamp.tolist() == frame.timestamp.iloc[[2, 5]].tolist()
    metrics = evaluate_model_trial_metrics_from_chunks(
        [frame.iloc[:2], frame.iloc[2:]],
        model_id="etf",
        label_column="label",
        prediction_column="x",
        validation_axis="single_asset_time_series",
        correlation_clock="trading_day",
        sessions=["2024-01-02", "2024-01-03"],
    )
    summary = tables["ic_summary"].iloc[0]
    assert metrics["validation_rankic"] == pytest.approx(summary.mean_rankic)
    assert metrics["validation_valid_rankic_count"] == summary.valid_rankic_count == 2
    assert metrics["validation_mse"] == pytest.approx(4 / 3)


def test_saved_predictions_common_intersection_reranks_and_preserves_own(tmp_path):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    frame["z"] = frame.x
    frame.loc[frame.timestamp.eq("2020-01-04"), "label"] = [1.0, 3.0, 2.0]
    frame.loc[frame.timestamp.eq("2020-01-04") & frame.asset_id.eq("b"), "z"] = np.nan
    frame.to_parquet(inputs, index=False)
    declaration = input_spec()
    declaration["column_roles"]["z"] = "feature"
    predictions = {}
    window = {"start": "2020-01-04", "end": "2020-01-05"}
    for name, column in (("dense", "x"), ("sparse", "z")):
        config = fit_config(tmp_path, inputs, name)
        config["inputs"]["spec"] = declaration
        config["model"]["input_feature_columns"] = [column]
        model = fit(config)
        predictions[name] = str(
            predict(
                {
                    "model": str(model),
                    "inputs": {"path": str(inputs), "spec": declaration},
                    "prediction_window": window,
                    "output": str(tmp_path / (name + "-pred")),
                }
            )
        )
    config = {
        "inputs": {"path": str(inputs), "spec": declaration},
        "predictions": predictions,
        "window": window,
        "cutoff": "2020-01-06",
        "sessions": ["2020-01-04", "2020-01-05"],
        "evaluation": {
            "evaluation_id": "predictions",
            "market_id": "a_share",
            "research_dataset_id": "synthetic",
            "signals": [
                {"signal_id": name, "source_kind": "prediction", "source_column": name}
                for name in predictions
            ],
            "label_columns": ["label"],
            "correlation_clock": "trading_day",
            "n_groups": 2,
        },
        "output": str(tmp_path / "own"),
    }
    for scope in ("own", "common"):
        with pytest.raises(ValueError, match="evaluation_input_session_not_declared"):
            evaluate(
                {
                    **config,
                    "sessions": ["2020-01-04"],
                    "sample_scope": scope,
                    "output": str(tmp_path / ("incomplete-" + scope)),
                }
            )
    _, own = read_evaluation_outputs(evaluate(config))
    metadata, common = read_evaluation_outputs(
        evaluate(
            {**config, "sample_scope": "common", "output": str(tmp_path / "common")}
        ),
        tables=("metrics", "daily_coverage"),
    )
    dense_own = own["metrics"].loc[own["metrics"].signal.eq("dense")].iloc[0]
    dense_common = common["metrics"].loc[common["metrics"].signal.eq("dense")].iloc[0]
    assert dense_own.n_obs == 3 and dense_common.n_obs == 2
    assert dense_own.rankic == pytest.approx(0.5)
    assert dense_common.rankic == pytest.approx(1.0)
    assert metadata["metadata"]["sessions"] == config["sessions"]
    assert common["daily_coverage"].domain_rows.tolist() == [2, 3]
    # Public validation scoring uses the same current label domain and daily clock.
    import json

    score_config = {
        "predictions": list(predictions.values()),
        "candidate_ids": list(predictions),
        "labels": {"path": str(inputs), "spec": declaration},
        "label_unit": "decimal_return",
        "validation_windows": [window],
        "scoring_window": window,
        "cutoff": "2020-01-06",
        "selection_group": "quarterly",
        "validation_axis": "cross_sectional",
        "correlation_clock": "trading_day",
        "sessions": config["sessions"],
        "output": str(tmp_path / "scores.json"),
    }
    output = score(score_config)
    scored = json.loads(output.read_text())
    assert scored["common_rows"] == 5
    assert all(
        item["validation_rankic"] == pytest.approx(1.0) for item in scored["metrics"]
    )

    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )

    sparse, prediction_metadata = read_predictions(predictions["sparse"])
    sparse.loc[
        sparse.timestamp.eq("2020-01-04"), ["prediction_score", "prediction_rank_score"]
    ] = 0.0
    constant = write_prediction_outputs(
        output_path=tmp_path / "constant-peer",
        output_chunks=[sparse],
        metadata=prediction_metadata,
    )
    constant_scores = score(
        {
            **score_config,
            "predictions": [predictions["dense"], str(constant)],
            "output": str(tmp_path / "constant-scores.json"),
        }
    )
    paired = json.loads(constant_scores.read_text())["metrics"]
    assert all(
        item["validation_valid_ic_count"] == item["validation_valid_rankic_count"] == 1
        for item in paired
    )
    assert [item["validation_mse"] for item in paired] == pytest.approx([15.75, 5.75])


def test_multiple_etf_signals_keep_daily_artifact_time_order(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02 09:30:00",
                    "2024-01-02 09:30:01",
                    "2024-01-03 09:30:00",
                    "2024-01-03 09:30:01",
                ]
            ),
            "asset_id": "etf",
            "x": [1.0, 2.0, 2.0, 1.0],
            "label": [1.0, 2.0, 1.0, 2.0],
            "eligible": True,
        }
    )
    frame["other"] = frame.x
    spec = _spec("single_asset_time_series")
    spec = replace(
        spec,
        signals=(
            *spec.signals,
            SignalSpec(
                signal_id="reverse",
                source_kind="feature",
                source_column="other",
                direction="lower_is_better",
            ),
        ),
    )
    _, tables = _saved_metrics(tmp_path, frame, spec)
    assert tables["metrics"].timestamp.is_monotonic_increasing
    assert len(tables["metrics"]) == 4
    assert tables["ic_summary"].valid_rankic_count.eq(2).all()


def test_common_daily_summary_uses_shared_defined_correlation_blocks(tmp_path):
    from tests.research.evaluation_test_support import (
        evaluation_config,
        write_evaluation_input,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02 09:30:00", "2024-01-02 09:30:03", "2024-01-03 09:30:00"]
            ).repeat(3),
            "asset_id": list("abc") * 3,
            "x": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "peer": [0.0, 0.0, 0.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0] * 3,
            "eligible": True,
            "label_available_at": pd.Timestamp("2024-01-03 09:31:00"),
        }
    )
    inputs = write_evaluation_input(
        tmp_path / "blocks", frame, features=("x", "peer"), labels=("label",)
    )
    spec = replace(
        _spec("cross_sectional"),
        signals=(
            *_spec("cross_sectional").signals,
            SignalSpec(signal_id="peer", source_kind="feature", source_column="peer"),
        ),
    )
    config = evaluation_config(
        inputs,
        tmp_path / "own",
        spec,
        start="2024-01-02",
        end="2024-01-03",
        cutoff="2024-01-04",
    )
    config["sessions"] = ["2024-01-02", "2024-01-03"]
    _, own = read_evaluation_outputs(evaluate(config))
    _, common = read_evaluation_outputs(
        evaluate(
            {**config, "sample_scope": "common", "output": str(tmp_path / "paired")}
        )
    )
    assert own["ic_summary"].set_index("signal").loc[
        "score", "mean_ic"
    ] == pytest.approx(0.5)
    assert common["ic_summary"].mean_ic.tolist() == pytest.approx([0.0, 0.0])
    assert common["ic_summary"].mean_rankic.tolist() == pytest.approx([0.0, 0.0])
    # Original per-timestamp values/counts stay visible as diagnostics.
    assert common["metrics"].n_obs.eq(3).all()
    assert common["metrics"].loc[
        common["metrics"].signal.eq("score"), "ic"
    ].tolist() == pytest.approx([1.0, -1.0, 1.0])

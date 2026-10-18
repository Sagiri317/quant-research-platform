"""Synthetic counterexamples for approved prediction and inference semantics."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.cross_section import (
    compute_ic_and_grouped_label_values,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.research.statistics import time_series_inference_summary
from quant_research_platform.research.statistics_production import (
    _model_comparison,
    _prediction_quality,
    _prediction_summary,
)


def test_no_selected_family_retains_undefined_statistics(tmp_path):
    from types import SimpleNamespace

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.statistics_production import (
        build_a_share_statistics_tables,
    )
    from tests.helpers.research_specs import make_demo_research_dataset_spec
    from tests.research.test_direct_statistics import _factor_tables

    _, _, dates = _factor_tables(tmp_path)
    spec = make_demo_research_dataset_spec(
        data_id="direct_factor",
        factor_columns=("factor",),
        label_columns=("forward_return_20d",),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    window = TimeWindow(dates[0], dates[-1])
    tables = build_a_share_statistics_tables(
        dataset_path=tmp_path / "input",
        dataset_spec=spec,
        prediction_id="no_winner",
        model_entries={
            ("period", "family"): {"model_metadata": None, "model_id": None}
        },
        selections={
            "period": {
                "families": [{"family_id": "family"}],
                "selected_family_id": None,
            }
        },
        period_windows={"period": window},
        trial_rows=pd.DataFrame(),
        config={
            "sample_role": "exploratory",
            "analyses": ["prediction_hac", "model_losses"],
            "evaluation": {"primary_label": "forward_return_20d"},
            "statistics": {"hac_lag": 0},
        },
        time_window=window,
        prediction_lookup=SimpleNamespace(coverage_counts=lambda: (0, 0)),
        global_prediction_lookup=None,
    )
    assert len(tables["prediction_diagnostics"]) == 1
    assert tables["prediction_diagnostics"].prediction_id.tolist() == ["no_winner"]
    expected_losses = {(date, "family") for date in dates}
    assert len(tables["model_losses"]) == len(expected_losses)
    assert (
        set(
            tables["model_losses"][["timestamp", "family_id"]].itertuples(
                index=False, name=None
            )
        )
        == expected_losses
    )
    assert tables["prediction_diagnostics"].status.eq("UNDEFINED").all()
    assert tables["model_losses"].status.eq("UNDEFINED").all()
    assert tables["model_losses"].config_id.isna().all()
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )

    output = write_statistics_artifact(
        tmp_path / "undefined-statistics",
        tables=tables,
        metadata={
            "sample_role": "exploratory",
            "window": {"start": str(window.start), "end": str(window.end)},
        },
    )
    with read_statistics_artifact(output).table("model_losses.parquet") as saved:
        assert len(saved) == len(expected_losses)
        assert (
            set(saved[["timestamp", "family_id"]].itertuples(index=False, name=None))
            == expected_losses
        )
        assert saved.config_id.eq("").all()
        assert saved.status.eq("UNDEFINED").all()


def test_head_is_fixed_before_label_availability_and_includes_cutoff_ties():
    scores = pd.Series([10.0, 9.0, 9.0, 1.0, 0.0])
    labels = pd.Series([np.nan, 0.4, 0.2, 0.0, -0.2])
    result = _prediction_quality(scores, labels, labels, head_fraction=0.4)
    assert result["head_selected_count"] == 3
    assert result["head_evaluable_count"] == 2
    assert result["head_coverage"] == pytest.approx(2 / 3)
    assert result["head_mean_raw_return"] == pytest.approx(0.3)
    assert result["pool_mean_raw_return"] == pytest.approx(0.1)
    assert result["head_excess_raw_return"] == pytest.approx(0.2)
    unavailable = _prediction_quality(scores, labels, labels, head_fraction=0.2)
    assert unavailable["head_selected_count"] == 1
    assert unavailable["head_evaluable_count"] == 0
    assert np.isnan(unavailable["head_mean_raw_return"])


def test_oos_r2_zero_baseline_and_constant_target_are_separate_from_ic():
    result = _prediction_quality(
        pd.Series([0.0, 0.0]),
        pd.Series([2.0, 2.0]),
        pd.Series([0.1, 0.1]),
        head_fraction=0.2,
    )
    assert result["mse"] == 4.0
    assert result["baseline_mse"] == 4.0
    assert result["oos_r2"] == 0.0
    result = _prediction_quality(
        pd.Series([1.0, 1.0]),
        pd.Series([0.0, 0.0]),
        pd.Series([0.0, 0.0]),
        head_fraction=0.2,
    )
    assert result["mse"] == 1.0
    assert np.isnan(result["oos_r2"])
    assert result["oos_r2_reason"] == "zero_baseline_error"


def test_group_assignments_do_not_replace_missing_top_label():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": list("abcde"),
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label": [0.0, 0.1, 0.2, 0.3, np.nan],
        }
    )
    metrics, groups = compute_ic_and_grouped_label_values(
        frame,
        ComputePanelSpec(
            signal_columns=("score",), label_columns=("label",), asset_column="asset_id"
        ),
    )
    assert metrics.n_obs.iloc[0] == 4
    top = groups.loc[groups.group.eq(5)].iloc[0]
    assert top.n_obs == 0 and np.isnan(top.mean_label_value)
    assert groups.loc[groups.group.eq(4), "mean_label_value"].iloc[0] == 0.3


def test_missing_session_blocks_inference_but_keeps_descriptive_mean():
    values = [0.1, 0.3, np.nan, -0.2, 0.4]
    from quant_research_platform.research.statistics_production import (
        _PREDICTION_COLUMN,
    )

    summary = _prediction_summary(
        pd.DataFrame({_PREDICTION_COLUMN: values}),
        prediction_id="candidate",
        inference_blocker=None,
        max_lags=0,
        block_size=1,
        repetitions=20,
        seed=42,
    ).iloc[0]
    assert summary["mean"] == pytest.approx(0.15)
    assert summary.status == summary.bootstrap_status == "UNDEFINED"
    assert summary.reason == summary.bootstrap_reason == "incomplete_session_clock"
    factors = time_series_inference_summary(
        pd.DataFrame(
            {"timestamp": pd.bdate_range("2024-01-02", periods=5), "factor": values}
        ),
        identity_column="timestamp",
        max_lags=0,
        fdr_method="BY",
        fdr_alpha=0.05,
    )
    assert factors.reason.iloc[0] == "incomplete_session_clock"
    assert factors["mean"].iloc[0] == pytest.approx(0.15)
    comparison, tests, _ = _model_comparison(
        pd.DataFrame({"a": values, "b": [0.2] * 5}),
        family_ids=("a", "b"),
        benchmark_id="b",
        sample_role="test",
        inference_blocker=None,
        max_lags=0,
        block_size=1,
        repetitions=20,
        root_seed=42,
        dataset_id="synthetic",
    )
    assert comparison.reason.eq("incomplete_session_clock").all()
    assert tests.reason.eq("incomplete_session_clock").all()


@pytest.mark.parametrize("repetitions", [None, 37])
def test_model_statistics_resolves_bootstrap_default_and_persists_head_metrics(
    tmp_path, repetitions
):
    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.research.statistics_production import model_statistics
    from tests.research.direct_model_test_support import candidates

    _, scoring = candidates(tmp_path)
    policy = {"bootstrap_block": 1}
    if repetitions is not None:
        policy["bootstrap_repetitions"] = repetitions
    config = {
        "dataset": scoring["labels"],
        "input_spec": scoring["label_spec"],
        "output": str(tmp_path / "statistics"),
        "window": scoring["scoring_window"],
        "label_cutoff": "2020-01-10",
        "family_ids": scoring["candidate_ids"],
        "analyses": ["model_losses", "prediction_bootstrap"],
        "evaluation": {"primary_label": "label", "raw_return_label": "label"},
        "statistics": policy,
        "periods": [
            {
                "period_id": "explicit",
                "window": scoring["scoring_window"],
                "predictions": scoring["predictions"],
            }
        ],
    }
    artifact = read_statistics_artifact(model_statistics(config))
    assert artifact.metadata["statistics"]["bootstrap_repetitions"] == (
        2000 if repetitions is None else repetitions
    )
    assert artifact.metadata["prediction_metrics"]["head_fraction"] == 0.2
    with artifact.table("model_prediction_metrics.parquet") as frame:
        assert len(frame) == 6
        assert frame.head_selected_count.ge(frame.head_evaluable_count).all()
        assert frame.head_mean_raw_return.notna().any()
        assert frame.mse.notna().all()
    with artifact.table("prediction_diagnostics.parquet") as frame:
        assert frame.bootstrap_repetitions.eq(
            2000 if repetitions is None else repetitions
        ).all()


def test_missing_top_group_round_trips_without_substituted_spread(tmp_path):
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.evaluations import EvaluationSpec
    from quant_research_platform.research.evaluations.engine import (
        build_evaluation_result_from_frames,
    )
    from tests.helpers.output_frames import make_artifact_signal_spec

    spec = EvaluationSpec(
        evaluation_id="fixed_head",
        market_id="a_share",
        research_dataset_id="synthetic",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": list("abcde"),
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label": [0.0, 0.1, 0.2, 0.3, np.nan],
            "eligible": True,
        }
    )
    with build_evaluation_result_from_frames(
        (frame,), spec=spec, asset_column="asset_id"
    ) as result:
        root = write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / "output",
        )
    grouped = pd.read_parquet(root / "grouped_label_values.parquet")
    assert grouped.loc[grouped.group.eq(5), "n_obs"].iloc[0] == 0
    assert pd.read_parquet(root / "group_spread_summary.parquet").empty

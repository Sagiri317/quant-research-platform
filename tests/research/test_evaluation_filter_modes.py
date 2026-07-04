from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import evaluate_signal_frame
from tests.helpers.research_specs import (
    make_signal_spec,
)
from tests.research.filter_mode_fixtures import (
    make_filter_mode_frame,
    sort_filter_mode_evaluation_frame,
)


def test_evaluation_filter_modes_and_diagnostics_use_hand_calculated_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_corr(*args, **kwargs):
        raise AssertionError("evaluation should delegate correlation math to compute")

    def fail_qcut(*args, **kwargs):
        raise AssertionError("evaluation should delegate grouped label math to compute")

    monkeypatch.setattr(pd.Series, "corr", fail_corr)
    monkeypatch.setattr(pd, "qcut", fail_qcut)

    frame = make_filter_mode_frame()
    eligible_spec = EvaluationSpec(
        run_id="eval_eligible",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    tradable_spec = EvaluationSpec(
        run_id="eval_tradable",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
        filter_mode="tradable",
    )
    unfiltered_spec = EvaluationSpec(
        run_id="eval_unfiltered",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
        filter_mode="unfiltered",
    )

    eligible = evaluate_signal_frame(frame, eligible_spec, n_groups=2)
    tradable = evaluate_signal_frame(frame, tradable_spec, n_groups=2)
    unfiltered = evaluate_signal_frame(frame, unfiltered_spec, n_groups=2)

    assert not hasattr(eligible, "metadata")
    assert eligible.diagnostics.filter_mode == "eligible"
    assert eligible.diagnostics.filter_columns_used == ("eligible",)
    assert eligible.diagnostics.filter_definition == "eligible == True"
    assert eligible.diagnostics.rows_input == 6
    assert eligible.diagnostics.rows_after_filter_mode == 5
    assert eligible.diagnostics.rows_after_signal_label_dropna == 4
    assert eligible.diagnostics.metric_rows == 1
    assert eligible.diagnostics.grouped_label_value_rows == 2
    assert eligible.diagnostics.ic_summary_rows == 1
    assert eligible.diagnostics.group_spread_summary_rows == 1
    assert eligible.diagnostics.timestamp_count == 1
    assert eligible.diagnostics.min_timestamp == "2024-01-02"
    assert eligible.diagnostics.max_timestamp == "2024-01-02"
    assert eligible.diagnostics.n_groups == 2
    assert eligible.diagnostics.skipped_timestamp_count == 0
    assert tradable.diagnostics.filter_columns_used == (
        "eligible",
        "can_enter",
        "can_exit",
    )
    assert tradable.diagnostics.rows_after_filter_mode == 3
    assert tradable.diagnostics.rows_after_signal_label_dropna == 2
    assert unfiltered.diagnostics.filter_columns_used == ()
    assert unfiltered.diagnostics.rows_after_filter_mode == 6
    assert unfiltered.diagnostics.rows_after_signal_label_dropna == 5

    assert eligible.metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert eligible.metrics.loc[0, "rankic"] == pytest.approx(1.0)
    assert list(eligible.ic_summary.columns) == [
        "signal",
        "label",
        "mean_ic",
        "std_ic",
        "icir",
        "positive_ic_rate",
        "valid_ic_count",
        "mean_rankic",
        "std_rankic",
        "rankicir",
        "positive_rankic_rate",
        "valid_rankic_count",
    ]
    assert list(eligible.group_spread_summary.columns) == [
        "signal",
        "label",
        "top_group_mean_label_value",
        "bottom_group_mean_label_value",
        "top_bottom_label_value_spread_mean",
        "top_bottom_label_value_spread_hit_rate",
        "group_monotonicity_score",
    ]
    assert eligible.grouped_label_values["mean_label_value"].tolist() == pytest.approx(
        [0.015, 0.045]
    )
    assert tradable.metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert tradable.grouped_label_values["mean_label_value"].tolist() == pytest.approx(
        [0.01, 0.02]
    )
    assert unfiltered.metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert unfiltered.grouped_label_values[
        "mean_label_value"
    ].tolist() == pytest.approx([0.02, 0.045])


def test_evaluation_all_nan_signal_label_reports_empty_diagnostics() -> None:
    frame = make_filter_mode_frame()
    frame["factor"] = np.nan
    frame["label"] = np.nan
    spec = EvaluationSpec(
        run_id="eval_all_nan",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
        filter_mode="unfiltered",
    )

    result = evaluate_signal_frame(frame, spec, n_groups=2)

    assert result.metrics["n_obs"].tolist() == [0]
    assert result.metrics["ic"].isna().all()
    assert result.metrics["rankic"].isna().all()
    assert result.grouped_label_values.empty
    assert result.group_spread_summary.empty
    assert result.diagnostics.rows_input == 6
    assert result.diagnostics.rows_after_filter_mode == 6
    assert result.diagnostics.rows_after_signal_label_dropna == 0
    assert result.diagnostics.metric_rows == 1
    assert result.diagnostics.grouped_label_value_rows == 0
    assert result.diagnostics.group_spread_summary_rows == 0
    assert result.diagnostics.skipped_timestamp_count == 1


def test_evaluation_group_spread_is_deterministic_for_shuffled_assets() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 12,
            "asset_id": [f"asset_{idx:02d}" for idx in range(12)],
            "timestamp": [pd.Timestamp("2024-01-02")] * 6
            + [pd.Timestamp("2024-01-03")] * 6,
            "factor": [1.0, 4.0, 2.0, 6.0, 3.0, 5.0] * 2,
            "label": [0.01, 0.04, 0.02, 0.06, 0.03, 0.05] * 2,
            "eligible": [True] * 12,
        }
    )
    spec = EvaluationSpec(
        run_id="eval_shuffle_stable",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    baseline = evaluate_signal_frame(frame, spec, n_groups=3)
    shuffled = evaluate_signal_frame(
        frame.sample(frac=1.0, random_state=31).reset_index(drop=True),
        spec,
        n_groups=3,
    )

    pd.testing.assert_frame_equal(
        sort_filter_mode_evaluation_frame(
            baseline.metrics,
            ("timestamp", "signal", "label"),
        ),
        sort_filter_mode_evaluation_frame(
            shuffled.metrics,
            ("timestamp", "signal", "label"),
        ),
    )
    pd.testing.assert_frame_equal(
        sort_filter_mode_evaluation_frame(
            baseline.grouped_label_values,
            ("timestamp", "signal", "label", "group"),
        ),
        sort_filter_mode_evaluation_frame(
            shuffled.grouped_label_values,
            ("timestamp", "signal", "label", "group"),
        ),
    )
    pd.testing.assert_frame_equal(
        sort_filter_mode_evaluation_frame(
            baseline.group_spread_summary,
            ("signal", "label"),
        ),
        sort_filter_mode_evaluation_frame(
            shuffled.group_spread_summary,
            ("signal", "label"),
        ),
    )
    assert shuffled.diagnostics == baseline.diagnostics


def test_evaluation_normalizes_lower_rank_signal_before_compute_output() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_b", "asset_c", "asset_d"],
            "timestamp": [pd.Timestamp("2024-01-02")] * 4,
            "risk_raw": [4.0, 3.0, 2.0, 1.0],
            "label": [0.01, 0.02, 0.03, 0.04],
            "eligible": [True, True, True, True],
        }
    )
    spec = EvaluationSpec(
        run_id="eval_lower_rank_signal",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(
            SignalSpec(
                signal_id="risk_signal",
                source_column="risk_raw",
                source_kind="factor",
                direction="lower_is_better",
                transform="rank",
            ),
        ),
        label_columns=("label",),
    )

    result = evaluate_signal_frame(frame, spec, n_groups=2)

    assert result.metrics.loc[0, "signal"] == "risk_signal"
    assert result.metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert result.metrics.loc[0, "rankic"] == pytest.approx(1.0)
    assert set(result.grouped_label_values["signal"]) == {"risk_signal"}
    assert (
        result.group_spread_summary.loc[
            0,
            "top_bottom_label_value_spread_mean",
        ]
        > 0
    )


def test_evaluation_computes_multiple_signals_in_one_run() -> None:
    frame = make_filter_mode_frame().assign(
        reverse_factor=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    spec = EvaluationSpec(
        run_id="eval_multi_signal",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(
            SignalSpec(
                signal_id="factor_signal",
                source_column="factor",
                source_kind="factor",
            ),
            SignalSpec(
                signal_id="reverse_signal",
                source_column="reverse_factor",
                source_kind="factor",
                direction="lower_is_better",
            ),
        ),
        label_columns=("label",),
    )

    result = evaluate_signal_frame(frame, spec, n_groups=2)

    assert set(result.metrics["signal"]) == {"factor_signal", "reverse_signal"}
    assert set(result.ic_summary["signal"]) == {"factor_signal", "reverse_signal"}
    assert set(result.group_spread_summary["signal"]) == {
        "factor_signal",
        "reverse_signal",
    }
    assert result.diagnostics.rows_after_filter_mode == 5
    assert result.diagnostics.rows_after_signal_label_dropna == 8


def test_evaluation_computes_signal_by_label_cross_product() -> None:
    frame = make_filter_mode_frame().assign(
        reverse_factor=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        label_alt=[0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
    )
    spec = EvaluationSpec(
        run_id="eval_multi_signal_label",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(
            SignalSpec(
                signal_id="factor_signal",
                source_column="factor",
                source_kind="factor",
            ),
            SignalSpec(
                signal_id="reverse_signal",
                source_column="reverse_factor",
                source_kind="factor",
                direction="lower_is_better",
            ),
        ),
        label_columns=("label", "label_alt"),
    )

    result = evaluate_signal_frame(frame, spec, n_groups=2)

    assert set(result.metrics["signal"]) == {"factor_signal", "reverse_signal"}
    assert set(result.metrics["label"]) == {"label", "label_alt"}
    assert len(result.metrics) == 4
    assert result.diagnostics.rows_after_signal_label_dropna == 18


def test_evaluation_diagnostics_contract_validates_shape() -> None:
    diagnostics = EvaluationDiagnostics(
        filter_mode="eligible",
        filter_columns_used=("eligible",),
        filter_definition="eligible == True",
        rows_input=6,
        rows_after_filter_mode=5,
        rows_after_signal_label_dropna=4,
        metric_rows=1,
        grouped_label_value_rows=2,
        ic_summary_rows=1,
        group_spread_summary_rows=1,
        timestamp_count=1,
        min_timestamp="2024-01-02",
        max_timestamp="2024-01-02",
        n_groups=2,
        skipped_timestamp_count=0,
    )

    assert diagnostics.contract_version == "0.1"

    with pytest.raises(ValueError, match="unsupported_evaluation_contract_version"):
        EvaluationDiagnostics(
            filter_mode="eligible",
            filter_columns_used=(),
            filter_definition="none",
            rows_input=0,
            rows_after_filter_mode=0,
            rows_after_signal_label_dropna=0,
            metric_rows=0,
            grouped_label_value_rows=0,
            ic_summary_rows=0,
            group_spread_summary_rows=0,
            timestamp_count=0,
            min_timestamp=None,
            max_timestamp=None,
            n_groups=2,
            skipped_timestamp_count=0,
            contract_version="9.9",
        )
    with pytest.raises(ValueError, match="invalid_evaluation_diagnostics_count"):
        EvaluationDiagnostics(
            filter_mode="eligible",
            filter_columns_used=(),
            filter_definition="none",
            rows_input=-1,
            rows_after_filter_mode=0,
            rows_after_signal_label_dropna=0,
            metric_rows=0,
            grouped_label_value_rows=0,
            ic_summary_rows=0,
            group_spread_summary_rows=0,
            timestamp_count=0,
            min_timestamp=None,
            max_timestamp=None,
            n_groups=2,
            skipped_timestamp_count=0,
        )
    with pytest.raises(ValueError, match="evaluation_diagnostics_n_groups_too_small"):
        EvaluationDiagnostics(
            filter_mode="eligible",
            filter_columns_used=(),
            filter_definition="none",
            rows_input=0,
            rows_after_filter_mode=0,
            rows_after_signal_label_dropna=0,
            metric_rows=0,
            grouped_label_value_rows=0,
            ic_summary_rows=0,
            group_spread_summary_rows=0,
            timestamp_count=0,
            min_timestamp=None,
            max_timestamp=None,
            n_groups=1,
            skipped_timestamp_count=0,
        )


@pytest.mark.parametrize(
    ("spec", "frame", "match"),
    [
        (
            EvaluationSpec(
                run_id="missing_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                signals=(make_signal_spec("factor"),),
                label_columns=("label",),
                filter_mode="tradable",
            ),
            make_filter_mode_frame().drop(columns=["can_exit"]),
            "missing_filter_columns: can_exit",
        ),
        (
            EvaluationSpec(
                run_id="null_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                signals=(make_signal_spec("factor"),),
                label_columns=("label",),
            ),
            make_filter_mode_frame().assign(
                eligible=[True, None, True, True, True, True],
            ),
            "null_filter_column_values: eligible",
        ),
        (
            EvaluationSpec(
                run_id="bad_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                signals=(make_signal_spec("factor"),),
                label_columns=("label",),
            ),
            make_filter_mode_frame().assign(eligible=["yes"] * 6),
            "invalid_filter_column_dtype: eligible",
        ),
    ],
)
def test_evaluation_filter_columns_fail_closed(
    spec: EvaluationSpec,
    frame: pd.DataFrame,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_signal_frame(frame, spec)


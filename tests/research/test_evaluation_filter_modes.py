from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.evaluations.engine as evaluation_engine
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import (
    _EvaluationStreamingState,
    _iter_evaluation_output_chunks,
)
from quant_research_platform.research.signals.materialization import (
    normalize_signal_frame,
)
from quant_research_platform.runtime.resources import (
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from tests.helpers.research_specs import (
    make_signal_spec,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.filter_mode_fixtures import (
    make_filter_mode_frame,
    sort_filter_mode_evaluation_frame,
)


def test_managed_evaluation_accepts_true_peak_and_rejects_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "asset_id": ["a", "b"],
            "raw": [1.0, 2.0],
            "label": [-0.01, 0.01],
        }
    )
    spec = EvaluationSpec(
        evaluation_id="lease_peak",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_kind="factor",
                source_column="raw",
            ),
        ),
        label_columns=("label",),
        filter_mode="unfiltered",
        n_groups=2,
    )

    normalized = normalize_signal_frame(
        frame.loc[:, ["timestamp", "asset_id", "raw", "label"]],
        spec.signals,
        source_columns_by_kind={"factor": ("raw",)},
    )
    formal_bound = (
        2 * pandas_frame_payload_bytes(normalized)
        + evaluation_engine._EVALUATION_TIMESTAMP_FACT_BYTES
        + evaluation_engine._evaluation_kernel_bytes(
            normalized,
            compute_columns=("timestamp", "asset_id", "signal", "label"),
            signal_count=1,
            n_groups=2,
        )
    )
    policy = capture_test_policy(
        available_memory_bytes=1024**2,
        explicit_cap_bytes=formal_bound,
    )
    with runtime_resource_policy_scope(policy):
        outputs = list(
            _iter_evaluation_output_chunks(
                (frame,),
                spec=spec,
                signal_columns=("signal",),
                label_columns=("label",),
                n_groups=2,
                asset_column="asset_id",
                source_columns_by_kind={"factor": ("raw",)},
                resource_policy=policy,
            )
        )
    assert len(outputs) == 1
    assert policy.used_memory_bytes == 0

    called: list[bool] = []

    def reject_kernel(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("evaluation kernel allocated before admission")

    monkeypatch.setattr(
        evaluation_engine,
        "compute_ic_and_grouped_label_values",
        reject_kernel,
    )
    insufficient = capture_test_policy(
        available_memory_bytes=1024**2,
        explicit_cap_bytes=formal_bound - 1,
    )
    with runtime_resource_policy_scope(insufficient):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=spec,
                    signal_columns=("signal",),
                    label_columns=("label",),
                    n_groups=2,
                    asset_column="asset_id",
                    source_columns_by_kind={"factor": ("raw",)},
                    resource_policy=insufficient,
                )
            )
    assert called == []
    assert insufficient.used_memory_bytes == 0


def test_evaluation_filter_modes_and_diagnostics_use_hand_calculated_values() -> None:
    frame = make_filter_mode_frame()
    eligible_spec = EvaluationSpec(
        evaluation_id="eval_eligible",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    tradable_spec = EvaluationSpec(
        evaluation_id="eval_tradable",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
        filter_mode="tradable",
    )
    unfiltered_spec = EvaluationSpec(
        evaluation_id="eval_unfiltered",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
        filter_mode="unfiltered",
    )

    eligible_policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(eligible_policy):
        eligible_state = _EvaluationStreamingState(
            spec=eligible_spec,
            n_groups=2,
            resource_policy=eligible_policy,
        )
        try:
            eligible_output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=eligible_spec,
                    signal_columns=("factor",),
                    label_columns=eligible_spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=eligible_policy,
                    summary_state=eligible_state,
                )
            )
            eligible_output = [item.value for item in eligible_output]
            eligible_metrics, eligible_grouped, _ = eligible_output[0]
            eligible_diagnostics = eligible_state.diagnostics()
            eligible_ic_summary = eligible_state.ic_summary()
            eligible_group_spread_summary = eligible_state.group_spread_summary()
        finally:
            eligible_state.close()

    tradable_policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(tradable_policy):
        tradable_state = _EvaluationStreamingState(
            spec=tradable_spec,
            n_groups=2,
            resource_policy=tradable_policy,
        )
        try:
            tradable_output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=tradable_spec,
                    signal_columns=("factor",),
                    label_columns=tradable_spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=tradable_policy,
                    summary_state=tradable_state,
                )
            )
            tradable_output = [item.value for item in tradable_output]
            tradable_metrics, tradable_grouped, _ = tradable_output[0]
            tradable_diagnostics = tradable_state.diagnostics()
        finally:
            tradable_state.close()

    unfiltered_policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(unfiltered_policy):
        unfiltered_state = _EvaluationStreamingState(
            spec=unfiltered_spec,
            n_groups=2,
            resource_policy=unfiltered_policy,
        )
        try:
            unfiltered_output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=unfiltered_spec,
                    signal_columns=("factor",),
                    label_columns=unfiltered_spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=unfiltered_policy,
                    summary_state=unfiltered_state,
                )
            )
            unfiltered_output = [item.value for item in unfiltered_output]
            unfiltered_metrics, unfiltered_grouped, _ = unfiltered_output[0]
            unfiltered_diagnostics = unfiltered_state.diagnostics()
        finally:
            unfiltered_state.close()

    assert eligible_diagnostics.filter_mode == "eligible"
    assert eligible_diagnostics.filter_columns_used == ("eligible",)
    assert eligible_diagnostics.filter_definition == "eligible == True"
    assert eligible_diagnostics.rows_input == 6
    assert eligible_diagnostics.rows_after_filter_mode == 5
    assert eligible_diagnostics.rows_after_signal_label_dropna == 4
    assert eligible_diagnostics.metric_rows == 1
    assert eligible_diagnostics.grouped_label_value_rows == 2
    assert eligible_diagnostics.ic_summary_rows == 1
    assert eligible_diagnostics.group_spread_summary_rows == 1
    assert eligible_diagnostics.timestamp_count == 1
    assert eligible_diagnostics.min_timestamp == "2024-01-02"
    assert eligible_diagnostics.max_timestamp == "2024-01-02"
    assert eligible_diagnostics.n_groups == 2
    assert eligible_diagnostics.skipped_timestamp_count == 0
    assert tradable_diagnostics.filter_columns_used == (
        "eligible",
        "can_enter",
        "can_exit",
    )
    assert tradable_diagnostics.rows_after_filter_mode == 3
    assert tradable_diagnostics.rows_after_signal_label_dropna == 2
    assert unfiltered_diagnostics.filter_columns_used == ()
    assert unfiltered_diagnostics.rows_after_filter_mode == 6
    assert unfiltered_diagnostics.rows_after_signal_label_dropna == 5

    assert eligible_metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert eligible_metrics.loc[0, "rankic"] == pytest.approx(1.0)
    assert list(eligible_ic_summary.columns) == [
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
        "ic_reason",
        "icir_reason",
        "rankic_reason",
        "rankicir_reason",
    ]
    assert list(eligible_group_spread_summary.columns) == [
        "signal",
        "label",
        "top_group_mean_label_value",
        "bottom_group_mean_label_value",
        "top_bottom_label_value_spread_mean",
        "top_bottom_label_value_spread_hit_rate",
        "group_monotonicity_score",
        "group_monotonicity_score_reason",
    ]
    assert eligible_grouped["mean_label_value"].tolist() == pytest.approx(
        [0.015, 0.045]
    )
    assert tradable_metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert tradable_grouped["mean_label_value"].tolist() == pytest.approx([0.01, 0.02])
    assert unfiltered_metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert unfiltered_grouped["mean_label_value"].tolist() == pytest.approx(
        [0.02, 0.045]
    )


def test_evaluation_all_nan_signal_label_reports_empty_diagnostics() -> None:
    frame = make_filter_mode_frame()
    frame["factor"] = np.nan
    frame["label"] = np.nan
    spec = EvaluationSpec(
        evaluation_id="eval_all_nan",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
        filter_mode="unfiltered",
    )

    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        state = _EvaluationStreamingState(
            spec=spec,
            n_groups=2,
            resource_policy=policy,
        )
        try:
            output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=spec,
                    signal_columns=("factor",),
                    label_columns=spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=policy,
                    summary_state=state,
                )
            )
            output = [item.value for item in output]
            metrics, grouped, _ = output[0]
            spread = state.group_spread_summary()
            diagnostics = state.diagnostics()
        finally:
            state.close()

    assert metrics["n_obs"].tolist() == [0]
    assert metrics["ic"].isna().all()
    assert metrics["rankic"].isna().all()
    assert metrics["ic_reason"].tolist() == ["insufficient_observations"]
    assert metrics["rankic_reason"].tolist() == ["insufficient_observations"]
    assert grouped.empty
    assert spread.empty
    assert diagnostics.rows_input == 6
    assert diagnostics.rows_after_filter_mode == 6
    assert diagnostics.rows_after_signal_label_dropna == 0
    assert diagnostics.metric_rows == 1
    assert diagnostics.grouped_label_value_rows == 0
    assert diagnostics.group_spread_summary_rows == 0
    assert diagnostics.skipped_timestamp_count == 1


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
        evaluation_id="eval_shuffle_stable",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    baseline_policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(baseline_policy):
        baseline_state = _EvaluationStreamingState(
            spec=spec,
            n_groups=3,
            resource_policy=baseline_policy,
        )
        try:
            baseline_output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=spec,
                    signal_columns=("factor",),
                    label_columns=spec.label_columns,
                    n_groups=3,
                    asset_column="asset_id",
                    resource_policy=baseline_policy,
                    summary_state=baseline_state,
                )
            )
            baseline_output = [item.value for item in baseline_output]
            baseline_metrics, baseline_grouped, _ = baseline_output[0]
            baseline_spread = baseline_state.group_spread_summary()
            baseline_diagnostics = baseline_state.diagnostics()
        finally:
            baseline_state.close()

    shuffled_policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(shuffled_policy):
        shuffled_state = _EvaluationStreamingState(
            spec=spec,
            n_groups=3,
            resource_policy=shuffled_policy,
        )
        try:
            shuffled_output = list(
                _iter_evaluation_output_chunks(
                    (frame.sample(frac=1.0, random_state=31).reset_index(drop=True),),
                    spec=spec,
                    signal_columns=("factor",),
                    label_columns=spec.label_columns,
                    n_groups=3,
                    asset_column="asset_id",
                    resource_policy=shuffled_policy,
                    summary_state=shuffled_state,
                )
            )
            shuffled_output = [item.value for item in shuffled_output]
            shuffled_metrics, shuffled_grouped, _ = shuffled_output[0]
            shuffled_spread = shuffled_state.group_spread_summary()
            shuffled_diagnostics = shuffled_state.diagnostics()
        finally:
            shuffled_state.close()

    pd.testing.assert_frame_equal(
        sort_filter_mode_evaluation_frame(
            baseline_metrics,
            ("timestamp", "signal", "label"),
        ),
        sort_filter_mode_evaluation_frame(
            shuffled_metrics,
            ("timestamp", "signal", "label"),
        ),
    )
    pd.testing.assert_frame_equal(
        sort_filter_mode_evaluation_frame(
            baseline_grouped,
            ("timestamp", "signal", "label", "group"),
        ),
        sort_filter_mode_evaluation_frame(
            shuffled_grouped,
            ("timestamp", "signal", "label", "group"),
        ),
    )
    pd.testing.assert_frame_equal(
        sort_filter_mode_evaluation_frame(
            baseline_spread,
            ("signal", "label"),
        ),
        sort_filter_mode_evaluation_frame(
            shuffled_spread,
            ("signal", "label"),
        ),
    )
    assert shuffled_diagnostics == baseline_diagnostics


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
        evaluation_id="eval_lower_rank_signal",
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

    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        state = _EvaluationStreamingState(
            spec=spec,
            n_groups=2,
            resource_policy=policy,
        )
        try:
            output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=spec,
                    signal_columns=("risk_signal",),
                    label_columns=spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=policy,
                    summary_state=state,
                )
            )
            output = [item.value for item in output]
            metrics, grouped, _ = output[0]
            spread = state.group_spread_summary()
        finally:
            state.close()

    assert metrics.loc[0, "signal"] == "risk_signal"
    assert metrics.loc[0, "ic"] == pytest.approx(1.0)
    assert metrics.loc[0, "rankic"] == pytest.approx(1.0)
    assert set(grouped["signal"]) == {"risk_signal"}
    assert (
        spread.loc[
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
        evaluation_id="eval_multi_signal",
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

    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        state = _EvaluationStreamingState(
            spec=spec,
            n_groups=2,
            resource_policy=policy,
        )
        try:
            output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=spec,
                    signal_columns=("factor_signal", "reverse_signal"),
                    label_columns=spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=policy,
                    summary_state=state,
                )
            )
            output = [item.value for item in output]
            metrics, _, _ = output[0]
            ic_summary = state.ic_summary()
            group_spread_summary = state.group_spread_summary()
            diagnostics = state.diagnostics()
        finally:
            state.close()

    assert set(metrics["signal"]) == {"factor_signal", "reverse_signal"}
    assert set(ic_summary["signal"]) == {"factor_signal", "reverse_signal"}
    assert set(group_spread_summary["signal"]) == {
        "factor_signal",
        "reverse_signal",
    }
    assert diagnostics.rows_after_filter_mode == 5
    assert diagnostics.rows_after_signal_label_dropna == 8


def test_evaluation_computes_signal_by_label_cross_product() -> None:
    frame = make_filter_mode_frame().assign(
        reverse_factor=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        label_alt=[0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
    )
    spec = EvaluationSpec(
        evaluation_id="eval_multi_signal_label",
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

    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        state = _EvaluationStreamingState(
            spec=spec,
            n_groups=2,
            resource_policy=policy,
        )
        try:
            output = list(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=spec,
                    signal_columns=("factor_signal", "reverse_signal"),
                    label_columns=spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=policy,
                    summary_state=state,
                )
            )
            output = [item.value for item in output]
            metrics, _, _ = output[0]
            diagnostics = state.diagnostics()
        finally:
            state.close()

    assert set(metrics["signal"]) == {"factor_signal", "reverse_signal"}
    assert set(metrics["label"]) == {"label", "label_alt"}
    assert len(metrics) == 4
    assert diagnostics.rows_after_signal_label_dropna == 18


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


def test_evaluation_spec_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="evaluation_axis_invalid"):
        EvaluationSpec(
            evaluation_id="bad_axis",
            market_id=MarketId("crypto"),
            research_dataset_id="demo_filter",
            signals=(make_signal_spec("factor"),),
            label_columns=("label",),
            evaluation_axis="rows",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("spec", "frame", "match"),
    [
        (
            EvaluationSpec(
                evaluation_id="missing_filter",
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
                evaluation_id="null_filter",
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
                evaluation_id="bad_filter",
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
    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match=match):
            tuple(
                _iter_evaluation_output_chunks(
                    (frame,),
                    spec=spec,
                    signal_columns=tuple(signal.signal_id for signal in spec.signals),
                    label_columns=spec.label_columns,
                    n_groups=spec.n_groups,
                    asset_column="asset_id",
                    resource_policy=policy,
                )
            )

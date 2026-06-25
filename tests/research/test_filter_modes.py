from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import assemble_research_dataset
from quant_research_platform.research.evaluations.engine import evaluate_signal_frame
from quant_research_platform.research.models import (
    fit_model_result,
    predict_model_frame,
)
from quant_research_platform.research.models.supervised import (
    model_algorithm_ids,
    model_smoke_hyperparameters,
)
from quant_research_platform.research.predictions import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.research.predictions import (
    outputs as prediction_outputs,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
    prediction_frame_from_artifact_frames,
    write_prediction_result,
)


def _signal(column: str, *, source_kind: str = "factor") -> SignalSpec:
    return SignalSpec(
        signal_id=column,
        source_column=column,
        source_kind=source_kind,
    )


def _filter_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"] * 6,
            "asset_id": [f"asset_{idx}" for idx in range(6)],
            "timestamp": [pd.Timestamp("2024-01-02")] * 6,
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "label": [0.01, 0.02, 0.03, 0.04, 0.05, np.nan],
            "eligible": [True, True, False, True, True, True],
            "can_enter": [True, True, True, False, True, True],
            "can_exit": [True, True, True, True, False, True],
        }
    )


def _dataset_catalog() -> ResearchDatasetCatalog:
    catalog = ResearchDatasetCatalog()
    catalog.register(
        ResearchDatasetSpec(
            data_id="demo_filter",
            market_id=MarketId("a_share"),
            input_data_ids=("demo_factor", "demo_label", "demo_annotation"),
            base_input_data_id="demo_annotation",
            factor_columns=("factor",),
            label_columns=("label",),
            filter_columns=("eligible", "can_enter", "can_exit"),
            frequency=DataFrequency.DAILY,
            grain=DataGrain.RESEARCH_ROW,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "factor": ColumnRole.FACTOR,
                "label": ColumnRole.LABEL,
                "eligible": ColumnRole.MASK,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def _processed_catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_factor",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "factor": ColumnRole.FACTOR,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_label",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.LABEL,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "label": ColumnRole.LABEL,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    catalog.register(
        ProcessedDataSpec(
            data_id="demo_annotation",
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.ANNOTATION,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "eligible": ColumnRole.UNIVERSE,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
            primary_key=platform_entity_time_key(),
            partition_columns=("year",),
        )
    )
    return catalog


def test_evaluation_filter_modes_and_diagnostics_use_hand_calculated_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_corr(*args, **kwargs):
        raise AssertionError("evaluation should delegate correlation math to compute")

    def fail_qcut(*args, **kwargs):
        raise AssertionError("evaluation should delegate grouped label math to compute")

    monkeypatch.setattr(pd.Series, "corr", fail_corr)
    monkeypatch.setattr(pd, "qcut", fail_qcut)

    frame = _filter_frame()
    eligible_spec = EvaluationSpec(
        run_id="eval_eligible",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
    )
    tradable_spec = EvaluationSpec(
        run_id="eval_tradable",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
        filter_mode="tradable",
    )
    unfiltered_spec = EvaluationSpec(
        run_id="eval_unfiltered",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
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
    frame = _filter_frame().assign(
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
    frame = _filter_frame().assign(
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


def test_evaluation_forced_tiny_resource_chunks_match_default_and_trim_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.evaluations import engine as eval_engine

    frame = pd.concat(
        [
            _filter_frame().assign(
                timestamp=pd.Timestamp(value),
                unused_blob="x" * 1000,
            )
            for value in (
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            )
        ],
        ignore_index=True,
    )
    spec = EvaluationSpec(
        run_id="eval_memory_safe",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
    )
    expected = evaluate_signal_frame(frame, spec, n_groups=2)
    original = eval_engine.compute_ic_and_grouped_label_values
    seen_columns: list[tuple[str, ...]] = []

    def wrapped_compute(frame, spec, *, n_groups):
        seen_columns.append(tuple(frame.columns))
        assert tuple(frame.columns) == ("timestamp", "asset_id", "factor", "label")
        return original(frame, spec, n_groups=n_groups)

    monkeypatch.setattr(
        eval_engine,
        "compute_ic_and_grouped_label_values",
        wrapped_compute,
    )
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")

    result = evaluate_signal_frame(frame, spec, n_groups=2)

    assert len(seen_columns) > 1
    pd.testing.assert_frame_equal(
        result.metrics.sort_values(["timestamp", "signal", "label"]).reset_index(
            drop=True
        ),
        expected.metrics.sort_values(["timestamp", "signal", "label"]).reset_index(
            drop=True
        ),
    )
    pd.testing.assert_frame_equal(result.ic_summary, expected.ic_summary)
    pd.testing.assert_frame_equal(
        result.group_spread_summary,
        expected.group_spread_summary,
    )
    assert result.diagnostics == expected.diagnostics


def test_evaluation_merges_adjacent_chunks_that_split_one_timestamp() -> None:
    from quant_research_platform.research.evaluations import engine as eval_engine

    frame = _filter_frame()
    spec = EvaluationSpec(
        run_id="eval_split_timestamp",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
    )
    expected = evaluate_signal_frame(frame, spec, n_groups=2)

    result = eval_engine._evaluate_signal_frame_chunks(
        (frame.iloc[:3], frame.iloc[3:]),
        spec,
        n_groups=2,
    )

    pd.testing.assert_frame_equal(result.metrics, expected.metrics)
    pd.testing.assert_frame_equal(
        result.grouped_label_values,
        expected.grouped_label_values,
    )
    pd.testing.assert_frame_equal(result.ic_summary, expected.ic_summary)
    pd.testing.assert_frame_equal(
        result.group_spread_summary,
        expected.group_spread_summary,
    )
    assert result.diagnostics == expected.diagnostics


def test_model_fit_merges_adjacent_chunks_that_split_one_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import supervised as model_supervised

    frame = _filter_frame()
    spec = ModelSpec(
        model_id="linear_split_timestamp",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(frame),
    )
    expected = model_supervised._fit_model_result_from_frames((frame,), spec)
    original_filter = model_supervised.apply_filter_mode
    filtered_chunk_rows: list[int] = []

    def wrapped_filter(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        return original_filter(frame, mode=mode, metadata_key=metadata_key)

    monkeypatch.setattr(model_supervised, "apply_filter_mode", wrapped_filter)
    result = model_supervised._fit_model_result_from_frames(
        (frame.iloc[:3], frame.iloc[3:]),
        spec,
    )

    assert filtered_chunk_rows == [len(frame)]
    assert result.coefficients.tolist() == pytest.approx(expected.coefficients.tolist())
    result_metadata = dict(result.metadata)
    expected_metadata = dict(expected.metadata)
    assert result_metadata.pop("feature_importance") == pytest.approx(
        expected_metadata.pop("feature_importance")
    )
    assert result_metadata == expected_metadata


def test_prediction_merges_adjacent_chunks_that_split_one_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.research.predictions.outputs as artifacts_module

    frame = _filter_frame().drop(columns=["label"])
    spec = PredictionSpec(
        prediction_id="pred_split_timestamp",
        model_id="linear_split_timestamp",
        research_dataset_id="demo_filter",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    coefficients = np.asarray([0.0, 0.01])
    expected = prediction_frame_from_artifact_frames(
        (frame,),
        spec,
        coefficients,
        ("factor",),
    )
    original_filter = artifacts_module.apply_filter_mode
    filtered_chunk_rows: list[int] = []

    def wrapped_filter(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        return original_filter(frame, mode=mode, metadata_key=metadata_key)

    monkeypatch.setattr(artifacts_module, "apply_filter_mode", wrapped_filter)
    result = prediction_frame_from_artifact_frames(
        (frame.iloc[:3], frame.iloc[3:]),
        spec,
        coefficients,
        ("factor",),
    )

    assert filtered_chunk_rows == [len(frame)]
    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs["prediction_metadata"] == expected.attrs["prediction_metadata"]


def test_evaluation_rejects_non_monotonic_time_chunk_order() -> None:
    from quant_research_platform.research.evaluations import engine as eval_engine

    frame = _filter_frame()
    first = frame.assign(timestamp=pd.Timestamp("2024-01-02"))
    second = frame.assign(timestamp=pd.Timestamp("2024-01-03"))
    late_overlap = frame.assign(timestamp=pd.Timestamp("2024-01-02"))
    spec = EvaluationSpec(
        run_id="eval_non_monotonic_chunks",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
    )

    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        eval_engine._evaluate_signal_frame_chunks(
            (first, second, late_overlap),
            spec,
            n_groups=2,
        )


def test_evaluation_rejects_pending_non_monotonic_time_overlap() -> None:
    from quant_research_platform.research.evaluations import engine as eval_engine

    frame = _filter_frame()
    first = frame.assign(timestamp=pd.Timestamp("2024-01-03"))
    earlier = frame.assign(timestamp=pd.Timestamp("2024-01-02"))
    spec = EvaluationSpec(
        run_id="eval_pending_non_monotonic_chunks",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
    )

    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        eval_engine._evaluate_signal_frame_chunks(
            (first, earlier),
            spec,
            n_groups=2,
        )


def test_evaluation_rejects_timestamp_boundary_overlap_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.evaluations import engine as eval_engine

    frame = _filter_frame()
    spec = EvaluationSpec(
        run_id="eval_boundary_over_budget",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
    )
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")

    with pytest.raises(
        ValueError,
        match="time_chunk_boundary_overlap_exceeds_memory_budget",
    ):
        eval_engine._evaluate_signal_frame_chunks(
            (frame.iloc[:3], frame.iloc[3:]),
            spec,
            n_groups=2,
        )


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
                signals=(_signal("factor"),),
                label_columns=("label",),
                filter_mode="tradable",
            ),
            _filter_frame().drop(columns=["can_exit"]),
            "missing_filter_columns: can_exit",
        ),
        (
            EvaluationSpec(
                run_id="null_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                signals=(_signal("factor"),),
                label_columns=("label",),
            ),
            _filter_frame().assign(eligible=[True, None, True, True, True, True]),
            "null_filter_column_values: eligible",
        ),
        (
            EvaluationSpec(
                run_id="bad_filter",
                market_id=MarketId("a_share"),
                research_dataset_id="demo_filter",
                signals=(_signal("factor"),),
                label_columns=("label",),
            ),
            _filter_frame().assign(eligible=["yes"] * 6),
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


def test_evaluation_rejects_invalid_time_chunk_timestamps() -> None:
    spec = EvaluationSpec(
        run_id="bad_timestamp",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(_signal("factor"),),
        label_columns=("label",),
    )

    with pytest.raises(ValueError, match="invalid_time_chunk_timestamp_values"):
        evaluate_signal_frame(
            _filter_frame().assign(timestamp=[pd.NaT] * len(_filter_frame())),
            spec,
        )


def test_filter_columns_cannot_be_features() -> None:
    with pytest.raises(ValueError, match="filter_column_in_input_columns: eligible"):
        EvaluationSpec(
            run_id="bad_eval_features",
            market_id=MarketId("a_share"),
            research_dataset_id="demo_filter",
            signals=(_signal("eligible"),),
            label_columns=("label",),
        )
    with pytest.raises(ValueError, match="filter_column_in_input_columns: can_exit"):
        ModelSpec(
            model_id="bad_model_features",
            model_algorithm="linear_regression",
            input_factor_columns=("can_exit",),
            label_column="label",
        )


def test_model_sample_filter_modes_and_metadata_ignore_tradability_masks() -> None:
    frame = _filter_frame()
    eligible_spec = ModelSpec(
        model_id="linear_eligible",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(frame),
    )
    unfiltered_spec = ModelSpec(
        model_id="linear_unfiltered",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        max_train_rows=len(frame),
    )

    eligible = fit_model_result(frame, eligible_spec)
    unfiltered = fit_model_result(frame, unfiltered_spec)

    assert (
        eligible.metadata
        | {
            key: eligible.metadata[key]
            for key in (
                "dependency_status",
                "feature_importance",
                "hyperparameters",
                "max_train_rows",
                "purge_gap_steps",
                "random_seed",
            )
        }
        == eligible.metadata
    )
    assert {
        key: eligible.metadata[key]
        for key in (
            "contract_version",
            "model_algorithm",
            "sample_filter_mode",
            "filter_columns_used",
            "filter_definition",
            "rows_input",
            "rows_after_filter_mode",
            "rows_after_input_label_dropna",
            "train_start",
            "train_end",
            "input_factor_columns",
            "input_feature_columns",
            "input_columns",
            "input_count",
            "label_column",
            "coefficient_count",
        )
    } == {
        "contract_version": "0.1",
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 6,
        "rows_after_filter_mode": 5,
        "rows_after_input_label_dropna": 4,
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
        "input_factor_columns": ["factor"],
        "input_feature_columns": [],
        "input_columns": ["factor"],
        "input_count": 1,
        "label_column": "label",
        "coefficient_count": 2,
    }
    assert eligible.metadata["dependency_status"]["status"] == "available"
    assert eligible.metadata["hyperparameters"] == {}
    assert eligible.metadata["random_seed"] is None
    assert eligible.metadata["max_train_rows"] == len(frame)
    assert eligible.metadata["purge_gap_steps"] == 0
    assert eligible.metadata["feature_importance"] == pytest.approx({"factor": 0.01})
    assert unfiltered.metadata["rows_after_filter_mode"] == 6
    assert unfiltered.metadata["rows_after_input_label_dropna"] == 5
    assert eligible.coefficients.tolist() == pytest.approx([0.0, 0.01])
    assert unfiltered.coefficients.tolist() == pytest.approx([0.0, 0.01])

    all_tradability_blocked = frame.assign(can_enter=False, can_exit=False)
    still_trains = fit_model_result(all_tradability_blocked, eligible_spec)
    assert still_trains.metadata["rows_after_input_label_dropna"] == 4


@pytest.mark.parametrize(
    "model_algorithm",
    model_algorithm_ids(),
)
def test_required_model_algorithm_ids_fit_and_predict(model_algorithm: str) -> None:
    train = pd.DataFrame(
        {
            "market_id": ["a_share"] * 8,
            "asset_id": [f"asset_{idx}" for idx in range(8)],
            "timestamp": [pd.Timestamp("2024-01-02")] * 8,
            "factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "feature": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "label": [0.11, 0.19, 0.31, 0.41, 0.49, 0.61, 0.69, 0.81],
            "eligible": [True] * 8,
        }
    )
    test = train.assign(
        timestamp=pd.Timestamp("2024-01-03"),
        label=np.nan,
    )
    frame = pd.concat([train, test], ignore_index=True)
    spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        random_seed=7,
        hyperparameters=_model_hyperparameters(model_algorithm),
        max_train_rows=len(train),
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_pred",
        model_id=spec.model_id,
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )

    result = fit_model_result(frame.loc[frame["timestamp"].lt("2024-01-03")], spec)
    prediction = prediction_frame_from_artifact(
        frame,
        prediction_spec,
        result.coefficients,
        spec.input_columns,
        prediction_model=result.prediction_model,
    )

    assert result.metadata["model_algorithm"] == model_algorithm
    assert result.metadata["input_columns"] == ["factor", "feature"]
    assert result.metadata["rows_after_input_label_dropna"] == 8
    assert len(prediction) == 8
    assert prediction[PREDICTION_SCORE_COLUMN].notna().all()


@pytest.mark.parametrize("model_algorithm", ["linear_regression", "ridge_regression"])
def test_model_training_controls_apply_to_linear_and_estimators(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "factor": [1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0],
            "eligible": [True, True, True],
        }
    )
    max_rows_spec = ModelSpec(
        model_id=f"{model_algorithm}_max_rows",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=1,
    )
    purge_spec = ModelSpec(
        model_id=f"{model_algorithm}_purge",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        purge_gap_steps=1,
        max_train_rows=len(frame),
    )

    max_rows = fit_model_result(frame, max_rows_spec)
    purge = fit_model_result(frame, purge_spec)

    assert max_rows.metadata["rows_after_input_label_dropna"] == 1
    assert max_rows.metadata["train_start"] == "2024-01-03"
    assert max_rows.metadata["train_end"] == "2024-01-03"
    assert purge.metadata["rows_after_input_label_dropna"] == 2
    assert purge.metadata["train_start"] == "2024-01-01"
    assert purge.metadata["train_end"] == "2024-01-02"


def test_model_max_train_rows_trims_streaming_training_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import supervised as model_supervised

    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)],
                    "asset_id": [f"asset_{day:02d}"],
                    "factor": [float(day)],
                    "label": [float(day) * 2.0],
                    "eligible": [True],
                }
            )
            for day in range(40)
        ],
        ignore_index=True,
    )
    spec = ModelSpec(
        model_id="linear_streaming_max_rows",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=7,
        purge_gap_steps=2,
    )
    expected = model_supervised._apply_training_controls(
        frame.loc[:, ["timestamp", "factor", "label"]],
        spec,
    )
    original_trim = model_supervised._trim_training_buffer
    observed_buffer_inputs: list[int] = []

    def wrapped_trim(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
        observed_buffer_inputs.append(len(frame))
        return original_trim(frame, spec)

    monkeypatch.setattr(model_supervised, "_trim_training_buffer", wrapped_trim)

    training, row_counts, _metadata = model_supervised._collect_training_frame(
        (chunk for _, chunk in frame.groupby(frame.index // 5)),
        spec,
    )

    pd.testing.assert_frame_equal(training, expected)
    assert row_counts["rows_after_input_label_dropna"] == len(expected)
    assert max(observed_buffer_inputs) < len(frame)


def test_model_training_collection_rejects_unbounded_collection() -> None:
    from quant_research_platform.research.models import supervised as model_supervised

    frame = pd.concat(
        [
            _filter_frame().assign(timestamp=pd.Timestamp(value))
            for value in ("2024-01-02", "2024-01-03", "2024-01-04")
        ],
        ignore_index=True,
    )
    spec = ModelSpec(
        model_id="linear_uncapped_streaming",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=None,
    )

    with pytest.raises(ValueError, match="unbounded_model_training_collection"):
        model_supervised._collect_training_frame(
            (frame,),
            spec,
        )

    with pytest.raises(ValueError, match="unbounded_model_training_collection"):
        model_supervised._collect_training_frame(
            (chunk for _, chunk in frame.groupby("timestamp")),
            spec,
        )


@pytest.mark.parametrize(
    "model_algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"],
)
def test_rank_ensembles_predict_cross_sectional_ranks_per_timestamp(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "asset_id": ["asset_a", "asset_b", "asset_a", "asset_b"],
            "factor": [1.0, 2.0, 1.0, 2.0],
            "feature": [10.0, 20.0, 10.0, 20.0],
            "label": [0.0, 1.0, 0.0, 1.0],
            "eligible": [True, True, True, True],
        }
    )
    spec = ModelSpec(
        model_id=f"{model_algorithm}_rank_scope",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )

    result = fit_model_result(frame, spec)
    predicted = result.prediction_model.predict(frame)
    predicted_frame = predict_model_frame(frame, spec, result)

    assert predicted.tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert predicted_frame["prediction_score"].tolist() == pytest.approx(
        [0.5, 1.0, 0.5, 1.0]
    )


@pytest.mark.parametrize(
    "model_algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"],
)
def test_rank_ensemble_prediction_outputs_are_chunk_boundary_invariant(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "asset_id": ["asset_a", "asset_b", "asset_a", "asset_b"],
            "factor": [1.0, 2.0, 1.0, 2.0],
            "feature": [10.0, 20.0, 10.0, 20.0],
            "label": [0.0, 1.0, 0.0, 1.0],
            "eligible": [True, True, True, True],
        }
    )
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_chunk_boundary",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_chunk_boundary_prediction",
        model_id=model_spec.model_id,
        research_dataset_id="rank_boundary_dataset",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )
    result = fit_model_result(frame, model_spec)

    one_chunk = _prediction_output_scores(
        [frame],
        prediction_spec=prediction_spec,
        model_spec=model_spec,
        result=result,
    )
    split_chunks = _prediction_output_scores(
        [chunk for _, chunk in frame.groupby("timestamp", sort=True)],
        prediction_spec=prediction_spec,
        model_spec=model_spec,
        result=result,
    )

    assert one_chunk["prediction_score"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert split_chunks["prediction_score"].tolist() == pytest.approx(
        one_chunk["prediction_score"].tolist()
    )


def _prediction_output_scores(
    frames: list[pd.DataFrame],
    *,
    prediction_spec: PredictionSpec,
    model_spec: ModelSpec,
    result,
) -> pd.DataFrame:
    build_state = prediction_outputs._PredictionBuildState()
    output = list(
        prediction_outputs._iter_prediction_output_frames(
            iter(frames),
            spec=prediction_spec,
            coefficients=result.coefficients,
            input_columns=model_spec.input_columns,
            predictor=prediction_outputs._prediction_callable(
                result.coefficients,
                model_spec.input_columns,
                result.prediction_model,
            ),
            build_state=build_state,
        )
    )
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["timestamp", "asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )


def _model_hyperparameters(model_algorithm: str) -> dict[str, object]:
    return model_smoke_hyperparameters(model_algorithm)


def test_model_forced_tiny_resource_chunks_match_default_without_design_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import supervised as model_supervised

    frame = pd.concat(
        [
            _filter_frame().assign(
                timestamp=pd.Timestamp(value),
                unused_blob="x" * 1000,
            )
            for value in (
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            )
        ],
        ignore_index=True,
    )
    spec = ModelSpec(
        model_id="linear_memory_safe",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(frame),
    )
    expected = fit_model_result(frame, spec)
    original_filter = model_supervised.apply_filter_mode
    filtered_chunk_rows: list[int] = []

    def wrapped_filter(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        return original_filter(frame, mode=mode, metadata_key=metadata_key)

    def fail_column_stack(*args, **kwargs):
        raise AssertionError("model fit must not allocate a full intercept matrix")

    monkeypatch.setattr(model_supervised, "apply_filter_mode", wrapped_filter)
    monkeypatch.setattr(np, "column_stack", fail_column_stack)
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")

    result = fit_model_result(frame, spec)

    assert len(filtered_chunk_rows) > 1
    assert max(filtered_chunk_rows) < len(frame)
    assert result.coefficients.tolist() == pytest.approx(expected.coefficients.tolist())
    result_metadata = dict(result.metadata)
    expected_metadata = dict(expected.metadata)
    assert result_metadata.pop("feature_importance") == pytest.approx(
        expected_metadata.pop("feature_importance")
    )
    assert result_metadata == expected_metadata


def test_model_rank_deficient_fit_matches_direct_lstsq_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x1 = [
        0.1257302210933933,
        -0.1321048632913019,
        0.6404226504432821,
        0.10490011715303971,
        -0.535669373161111,
        0.36159505490948474,
        1.3040000451301372,
        0.9470809631292422,
    ]
    x2 = [
        0.12573021405604093,
        -0.1321048759455166,
        0.6404226442105374,
        0.1049001175662995,
        -0.5356693964114188,
        0.3615950527215681,
        1.3040000326710277,
        0.9470809558065686,
    ]
    y = [
        -0.5442589828573099,
        -0.31630015636915454,
        0.4116305363741328,
        1.0425133694426776,
        -0.12853466294403426,
        1.3664634705496859,
        -0.6651946734866135,
        0.3515100700930197,
    ]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-05",
                ]
            ),
            "asset_id": [f"asset_{idx}" for idx in range(len(x1))],
            "factor": x1,
            "factor_shadow": x2,
            "label": y,
            "eligible": [True] * len(x1),
        }
    )
    spec = ModelSpec(
        model_id="linear_rank_deficient",
        model_algorithm="linear_regression",
        input_factor_columns=("factor", "factor_shadow"),
        label_column="label",
        max_train_rows=len(frame),
    )
    usable = frame.loc[frame["eligible"]].dropna(
        subset=[*spec.input_columns, spec.label_column]
    )
    x = usable.loc[:, spec.input_columns].to_numpy(dtype="float64")
    y = usable.loc[:, spec.label_column].to_numpy(dtype="float64")
    design = np.empty((len(x), len(spec.input_columns) + 1), dtype="float64")
    design[:, 0] = 1.0
    design[:, 1:] = x
    expected, *_ = np.linalg.lstsq(design, y, rcond=None)

    result = fit_model_result(frame, spec)

    assert result.coefficients.tolist() == pytest.approx(expected.tolist())

    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")
    tiny_result = fit_model_result(frame, spec)

    assert tiny_result.coefficients.tolist() == pytest.approx(
        expected.tolist(),
        rel=1e-7,
    )


def test_model_rejects_tradable_sample_filter_mode() -> None:
    with pytest.raises(ValueError, match="unsupported_model_sample_filter_mode"):
        ModelSpec(
            model_id="linear_tradable",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
            sample_filter_mode="tradable",
        )


def test_model_rejects_unsupported_contract_version() -> None:
    with pytest.raises(ValueError, match="unsupported_model_contract_version"):
        ModelSpec(
            model_id="linear_bad_contract",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
            contract_version="9.9",
        )


def test_prediction_sample_filter_modes_metadata_and_label_free_generation(
    tmp_path: Path,
) -> None:
    frame = pd.concat(
        [
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        ],
        ignore_index=True,
    )
    spec = PredictionSpec(
        prediction_id="pred_eligible",
        model_id="linear_eligible",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )
    prediction = prediction_frame_from_artifact(
        frame.drop(columns=["label"]),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
    )

    assert "metadata" not in prediction.attrs
    assert prediction.attrs["prediction_metadata"] == {
        "contract_version": "0.1",
        "prediction_mode": "minimal_oos",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 12,
        "rows_after_filter_mode": 10,
        "rows_after_input_dropna": 10,
        "prediction_rows": 5,
        "train_end": "2024-01-02",
        "test_start": "2024-01-03",
        "requested_test_end": "2024-01-03",
        "effective_test_end": "2024-01-03",
        "is_oos": True,
        "is_final_oos_window": False,
        "input_columns": ["factor"],
        "input_count": 1,
        "prediction_column": PREDICTION_SCORE_COLUMN,
    }
    assert len(prediction) == 5
    assert set(prediction.columns) == {
        "asset_id",
        "timestamp",
        PREDICTION_SCORE_COLUMN,
    }

    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(spec)

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_eligible",
    )
    run_root = write_prediction_result(layout, result)
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_metadata"]["contract_version"] == "0.1"
    assert manifest["model_metadata"]["input_count"] == 1
    assert manifest["model_metadata"]["coefficient_count"] == 2
    assert manifest["model_metadata"]["sample_filter_mode"] == "eligible"
    assert manifest["model_metadata"]["rows_after_input_label_dropna"] == 4
    assert manifest["model_metadata"]["train_start"] == "2024-01-02"
    assert manifest["model_metadata"]["train_end"] == "2024-01-02"
    assert manifest["prediction_metadata"]["contract_version"] == "0.1"
    assert manifest["prediction_metadata"]["prediction_rows"] == 5
    assert manifest["prediction_metadata"]["sample_filter_mode"] == "eligible"
    assert manifest["prediction_metadata"]["rows_after_input_dropna"] == 10
    assert manifest["prediction_metadata"]["train_end"] == "2024-01-02"
    assert manifest["prediction_metadata"]["requested_test_end"] == "2024-01-03"
    assert manifest["prediction_metadata"]["effective_test_end"] == "2024-01-03"


def test_prediction_open_ended_oos_model_metadata_uses_effective_test_end(
    tmp_path: Path,
) -> None:
    frame = pd.concat(
        [
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-04")),
        ],
        ignore_index=True,
    )
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-04"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_open_ended",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_open_ended",
            model_id="linear_open_ended",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            test_end=None,
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_open_ended",
    )
    run_root = write_prediction_result(layout, result)
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["prediction_metadata"]["requested_test_end"] is None
    assert manifest["prediction_metadata"]["effective_test_end"] == "2024-01-04"
    assert manifest["model_metadata"]["oos_end"] == "2024-01-04"


def test_prediction_forced_tiny_resource_chunks_match_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.research.predictions.outputs as artifacts_module

    frame = pd.concat(
        [
            _filter_frame().assign(
                timestamp=pd.Timestamp(value),
                unused_blob="x" * 1000,
            )
            for value in (
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            )
        ],
        ignore_index=True,
    )
    spec = PredictionSpec(
        prediction_id="pred_memory_safe",
        model_id="linear_memory_safe",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-05",
    )
    expected = prediction_frame_from_artifact(
        frame.drop(columns=["label"]),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
    )
    original_filter = artifacts_module.apply_filter_mode
    filtered_chunk_rows: list[int] = []

    def wrapped_filter(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        return original_filter(frame, mode=mode, metadata_key=metadata_key)

    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")
    monkeypatch.setattr(artifacts_module, "apply_filter_mode", wrapped_filter)
    result = prediction_frame_from_artifact(
        frame.drop(columns=["label"]),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
    )

    assert len(filtered_chunk_rows) > 1
    assert max(filtered_chunk_rows) < len(frame)
    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs["prediction_metadata"] == expected.attrs["prediction_metadata"]


def test_prediction_handoff_records_window_override_metadata(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        ],
        ignore_index=True,
    )
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_override",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-01",
            test_start="2024-01-04",
            test_end="2024-01-04",
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_override",
        prediction_train_end="2024-01-02",
        prediction_test_start="2024-01-03",
        prediction_test_end="2024-01-03",
    )
    run_root = write_prediction_result(layout, result)
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["train_end"] == "2024-01-02"
    assert manifest["test_start"] == "2024-01-03"
    assert manifest["test_end"] == "2024-01-03"
    assert manifest["model_metadata"]["train_start"] == "2024-01-02"
    assert manifest["model_metadata"]["train_end"] == "2024-01-02"
    assert manifest["prediction_metadata"]["train_end"] == "2024-01-02"
    assert manifest["prediction_metadata"]["test_start"] == "2024-01-03"
    assert manifest["prediction_metadata"]["requested_test_end"] == "2024-01-03"
    assert manifest["prediction_metadata"]["effective_test_end"] == "2024-01-03"


def test_prediction_handoff_model_fit_does_not_read_test_window_labels(
    tmp_path: Path,
) -> None:
    train_rows = _filter_frame().assign(
        timestamp=pd.Timestamp("2024-01-02"),
        factor=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        label=[5.0, 8.0, 11.0, 14.0, 17.0, 20.0],
        eligible=True,
    )
    test_rows = _filter_frame().assign(
        timestamp=pd.Timestamp("2024-01-03"),
        factor=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        label=[-10.0, -20.0, -30.0, -40.0, -50.0, -60.0],
        eligible=True,
    )
    frame = pd.concat([train_rows, test_rows], ignore_index=True)
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-02", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_no_leakage",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            test_end="2024-01-03",
        )
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_no_leakage",
    )
    run_root = write_prediction_result(layout, result)
    prediction = pd.read_parquet(run_root / "prediction.parquet").sort_values(
        "asset_id",
        kind="mergesort",
    )

    assert prediction[PREDICTION_SCORE_COLUMN].tolist() == pytest.approx(
        [5.0, 8.0, 11.0, 14.0, 17.0, 20.0]
    )


def _write_processed_inputs(
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    frame: pd.DataFrame,
) -> None:
    columns_by_id = {
        "demo_factor": ["market_id", "asset_id", "timestamp", "factor"],
        "demo_label": ["market_id", "asset_id", "timestamp", "label"],
        "demo_annotation": [
            "market_id",
            "asset_id",
            "timestamp",
            "eligible",
            "can_enter",
            "can_exit",
        ],
    }
    for data_id, columns in columns_by_id.items():
        spec = processed_catalog.get(data_id)
        write_dataset(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            frame.loc[:, columns],
            spec=spec,
        )


def test_prediction_unfiltered_mode_and_tradable_rejection() -> None:
    spec = PredictionSpec(
        prediction_id="pred_unfiltered",
        model_id="linear_unfiltered",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        sample_filter_mode="unfiltered",
    )
    prediction = prediction_frame_from_artifact(
        _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
    )

    assert "metadata" not in prediction.attrs
    assert prediction.attrs["prediction_metadata"]["rows_after_filter_mode"] == 6
    assert prediction.attrs["prediction_metadata"]["rows_after_input_dropna"] == 6

    with pytest.raises(ValueError, match="unsupported_prediction_sample_filter_mode"):
        PredictionSpec(
            prediction_id="pred_tradable",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            sample_filter_mode="tradable",
        )


def test_prediction_outputs_fail_closed_errors_are_specific() -> None:
    with pytest.raises(
        ValueError,
        match="prediction_train_end_must_precede_test_start",
    ):
        prediction_frame_from_artifact(
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            PredictionSpec(
                prediction_id="pred_bad_order",
                model_id="linear_eligible",
                research_dataset_id="demo_filter",
                train_end="2024-01-03",
                test_start="2024-01-03",
            ),
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("factor",),
        )

    with pytest.raises(
        ValueError,
        match="prediction_test_end_must_not_precede_test_start",
    ):
        prediction_frame_from_artifact(
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            PredictionSpec(
                prediction_id="pred_bad_test_end",
                model_id="linear_eligible",
                research_dataset_id="demo_filter",
                train_end="2024-01-02",
                test_start="2024-01-04",
                test_end="2024-01-03",
            ),
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("factor",),
        )

    with pytest.raises(ValueError, match="empty_prediction_window"):
        prediction_frame_from_artifact(
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-02")),
            PredictionSpec(
                prediction_id="pred_empty_test",
                model_id="linear_eligible",
                research_dataset_id="demo_filter",
                train_end="2024-01-02",
                test_start="2024-01-03",
                test_end="2024-01-03",
            ),
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("factor",),
        )


def test_prediction_handoff_empty_training_window_fails_closed(
    tmp_path: Path,
) -> None:
    frame = _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03"))
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = _dataset_catalog()
    processed_catalog = _processed_catalog()
    _write_processed_inputs(layout, processed_catalog, frame)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo_filter",
        time_window=TimeWindow("2024-01-03", "2024-01-03"),
    )
    model_catalog = ModelCatalog()
    model_catalog.register(
        ModelSpec(
            model_id="linear_eligible",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
        )
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="pred_empty_train",
            model_id="linear_eligible",
            research_dataset_id="demo_filter",
            train_end="2024-01-02",
            test_start="2024-01-03",
            test_end="2024-01-03",
        )
    )

    with pytest.raises(ValueError, match="empty_training_window"):
        build_prediction_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="pred_empty_train",
        )


def test_prediction_outputs_rejects_filter_columns_as_direct_features() -> None:
    spec = PredictionSpec(
        prediction_id="pred_bad_feature",
        model_id="linear_unfiltered",
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        sample_filter_mode="unfiltered",
    )

    with pytest.raises(ValueError, match="filter_column_in_input_columns: eligible"):
        prediction_frame_from_artifact(
            _filter_frame().assign(timestamp=pd.Timestamp("2024-01-03")),
            spec,
            coefficients=np.asarray([0.0, 0.01]),
            input_columns=("eligible",),
        )

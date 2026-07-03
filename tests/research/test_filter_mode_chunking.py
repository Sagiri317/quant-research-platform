from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PredictionSpec,
)
from quant_research_platform.research.evaluations.engine import evaluate_signal_frame
from quant_research_platform.research.models import (
    fit_model_result,
)
from quant_research_platform.research.models.supervised import (
    model_smoke_hyperparameters,
)
from quant_research_platform.research.predictions import (
    outputs as prediction_outputs,
)
from tests.helpers.research_specs import (
    make_signal_spec,
)
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
    prediction_frame_from_artifact_frames,
)


def _sort_evaluation_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return frame.sort_values(list(columns)).reset_index(drop=True)


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
            predictor=prediction_outputs.prediction_callable(
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
        signals=(make_signal_spec("factor"),),
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
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    expected = evaluate_signal_frame(frame, spec, n_groups=2)

    result = eval_engine.evaluate_signal_frame_chunks(
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
    from quant_research_platform.research.models import fitting as model_fitting
    from quant_research_platform.research.models import supervised as model_supervised

    frame = _filter_frame()
    spec = ModelSpec(
        model_id="linear_split_timestamp",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(frame),
    )
    expected = model_fitting.fit_model_result_from_frames(
        (frame,),
        spec,
        estimator=model_supervised.build_estimator_from_model_spec(spec),
    )
    original_filter = model_fitting.apply_filter_mode
    filtered_chunk_rows: list[int] = []

    def wrapped_filter(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        return original_filter(frame, mode=mode, metadata_key=metadata_key)

    monkeypatch.setattr(model_fitting, "apply_filter_mode", wrapped_filter)
    result = model_fitting.fit_model_result_from_frames(
        (frame.iloc[:3], frame.iloc[3:]),
        spec,
        estimator=model_supervised.build_estimator_from_model_spec(spec),
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
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        eval_engine.evaluate_signal_frame_chunks(
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
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        eval_engine.evaluate_signal_frame_chunks(
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
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "0.0001")

    with pytest.raises(
        ValueError,
        match="time_chunk_boundary_overlap_exceeds_memory_budget",
    ):
        eval_engine.evaluate_signal_frame_chunks(
            (frame.iloc[:3], frame.iloc[3:]),
            spec,
            n_groups=2,
        )


def test_evaluation_rejects_invalid_time_chunk_timestamps() -> None:
    spec = EvaluationSpec(
        run_id="bad_timestamp",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    with pytest.raises(ValueError, match="invalid_time_chunk_timestamp_values"):
        evaluate_signal_frame(
            _filter_frame().assign(timestamp=[pd.NaT] * len(_filter_frame())),
            spec,
        )


def test_model_forced_tiny_resource_chunks_match_default_without_design_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import fitting as model_fitting

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
    original_filter = model_fitting.apply_filter_mode
    filtered_chunk_rows: list[int] = []

    def wrapped_filter(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        return original_filter(frame, mode=mode, metadata_key=metadata_key)

    def fail_column_stack(*args, **kwargs):
        raise AssertionError("model fit must not allocate a full intercept matrix")

    monkeypatch.setattr(model_fitting, "apply_filter_mode", wrapped_filter)
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



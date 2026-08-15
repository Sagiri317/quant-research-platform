from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PredictionSpec,
)
from quant_research_platform.research.evaluations.engine import (
    evaluate_signal_frame_chunks,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
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
)
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
    prediction_frame_from_artifact_frames,
)


def wrap_filter_for_chunking(original_filter, filtered_chunk_rows: list[int]):
    def record_filtered_chunk(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        return original_filter(frame, mode=mode, metadata_key=metadata_key)

    return record_filtered_chunk


def test_evaluation_forced_tiny_resource_chunks_match_default_and_trim_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.research._memory as memory_owner
    from quant_research_platform.research.evaluations import engine as eval_engine

    frame = pd.concat(
        [
            make_filter_mode_frame().assign(
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
    expected = evaluate_signal_frame_chunks((frame,), spec, n_groups=2)
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
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    original_frame_bytes = memory_owner._frame_bytes
    monkeypatch.setattr(
        memory_owner,
        "_frame_bytes",
        lambda chunk: 5_000 * original_frame_bytes(chunk),
    )
    result = evaluate_signal_frame_chunks(
        (frame,),
        spec,
        n_groups=2,
        resource_policy=policy,
    )

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

    frame = make_filter_mode_frame()
    spec = EvaluationSpec(
        run_id="eval_split_timestamp",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    expected = evaluate_signal_frame_chunks((frame,), spec, n_groups=2)

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
    import quant_research_platform.research.models.supervised as model_supervised

    frame = make_filter_mode_frame()
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

    monkeypatch.setattr(
        model_fitting,
        "apply_filter_mode",
        wrap_filter_for_chunking(original_filter, filtered_chunk_rows),
    )
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

    frame = make_filter_mode_frame().drop(columns=["label"])
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

    monkeypatch.setattr(
        artifacts_module,
        "apply_filter_mode",
        wrap_filter_for_chunking(original_filter, filtered_chunk_rows),
    )
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

    frame = make_filter_mode_frame()
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

    frame = make_filter_mode_frame()
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


def test_complete_time_chunks_close_upstream_on_error_and_consumer_close() -> None:
    from quant_research_platform.research._memory import (
        iter_complete_time_sized_frames,
    )

    frame = make_filter_mode_frame().iloc[:1]
    closed: list[str] = []

    def retained(name: str, timestamps: tuple[str, ...]):
        try:
            for timestamp in timestamps:
                yield frame.assign(timestamp=pd.Timestamp(timestamp))
        finally:
            closed.append(name)

    invalid = iter_complete_time_sized_frames(
        retained("error", ("2024-01-03", "2024-01-02")),
        timestamp_column="timestamp",
    )
    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        tuple(invalid)
    assert closed == ["error"]

    abandoned = iter_complete_time_sized_frames(
        retained("early", ("2024-01-01", "2024-01-02", "2024-01-03")),
        timestamp_column="timestamp",
    )
    next(abandoned)
    abandoned.close()
    assert closed == ["error", "early"]

    ordinary = iter(
        (
            frame.assign(timestamp=pd.Timestamp("2024-01-01")),
            frame.assign(timestamp=pd.Timestamp("2024-01-02")),
        )
    )
    assert (
        len(
            tuple(
                iter_complete_time_sized_frames(
                    ordinary,
                    timestamp_column="timestamp",
                )
            )
        )
        == 2
    )


def test_evaluation_rejects_timestamp_boundary_overlap_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.evaluations import engine as eval_engine

    frame = make_filter_mode_frame()
    spec = EvaluationSpec(
        run_id="eval_boundary_over_budget",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 + 1_000,
    )

    with pytest.raises(
        ValueError,
        match="research_time_boundary_payloads_exceed_task_memory_budget",
    ):
        eval_engine.evaluate_signal_frame_chunks(
            (frame.iloc[:3], frame.iloc[3:]),
            spec,
            n_groups=2,
            resource_policy=policy,
        )


def test_evaluation_output_accumulation_honors_shared_root_headroom() -> None:
    frame = make_filter_mode_frame()
    spec = EvaluationSpec(
        run_id="eval_shared_headroom",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    frame_bytes = pandas_frame_payload_bytes(frame)
    assert frame_bytes < policy.task_memory_budget_bytes

    with runtime_resource_policy_scope(policy):
        with policy.reserve(
            "outer_owner",
            policy.task_memory_budget_bytes - frame_bytes,
        ):
            with pytest.raises(
                ValueError,
                match="task_memory_budget",
            ):
                evaluate_signal_frame_chunks(
                    (frame,),
                    spec,
                    n_groups=2,
                    resource_policy=policy,
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
        evaluate_signal_frame_chunks(
            (
                make_filter_mode_frame().assign(
                    timestamp=[pd.NaT] * len(make_filter_mode_frame())
                ),
            ),
            spec,
        )


def test_model_forced_tiny_resource_chunks_match_default_without_design_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.research._memory as memory_owner

    frame = pd.concat(
        [
            make_filter_mode_frame().assign(
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
    expected = model_fitting.fit_model_result_from_frames(
        (frame,), spec, estimator=build_estimator_from_model_spec(spec)
    )
    original_filter = model_fitting.apply_filter_mode
    filtered_chunk_rows: list[int] = []

    def fail_column_stack(*args, **kwargs):
        raise AssertionError("model fit must not allocate a full intercept matrix")

    monkeypatch.setattr(
        model_fitting,
        "apply_filter_mode",
        wrap_filter_for_chunking(original_filter, filtered_chunk_rows),
    )
    monkeypatch.setattr(np, "column_stack", fail_column_stack)
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    original_frame_bytes = memory_owner._frame_bytes
    monkeypatch.setattr(
        memory_owner,
        "_frame_bytes",
        lambda chunk: 10_000 * original_frame_bytes(chunk),
    )
    with runtime_resource_policy_scope(policy):
        result = model_fitting.fit_model_result_from_frames(
            (frame,),
            spec,
            estimator=build_estimator_from_model_spec(spec),
            resource_policy=policy,
        )

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
    import quant_research_platform.research._memory as memory_owner
    import quant_research_platform.research.predictions.outputs as artifacts_module

    frame = pd.concat(
        [
            make_filter_mode_frame().assign(
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

    monkeypatch.setattr(
        artifacts_module,
        "apply_filter_mode",
        wrap_filter_for_chunking(original_filter, filtered_chunk_rows),
    )
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    original_frame_bytes = memory_owner._frame_bytes
    monkeypatch.setattr(
        memory_owner,
        "_frame_bytes",
        lambda chunk: 5_000 * original_frame_bytes(chunk),
    )
    result = prediction_frame_from_artifact(
        frame.drop(columns=["label"]),
        spec,
        coefficients=np.asarray([0.0, 0.01]),
        input_columns=("factor",),
        resource_policy=policy,
    )

    assert len(filtered_chunk_rows) > 1
    assert max(filtered_chunk_rows) < len(frame)
    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs["prediction_metadata"] == expected.attrs["prediction_metadata"]

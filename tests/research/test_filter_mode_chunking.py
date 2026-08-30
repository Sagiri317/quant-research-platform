from __future__ import annotations

from contextlib import contextmanager

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
from quant_research_platform.research.evaluations import engine as eval_engine
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
    @contextmanager
    def record_filtered_chunk(frame, *, mode, metadata_key):
        filtered_chunk_rows.append(len(frame))
        with original_filter(
            frame,
            mode=mode,
            metadata_key=metadata_key,
        ) as result:
            yield result

    return record_filtered_chunk


def test_evaluation_forced_tiny_resource_chunks_match_default_and_trim_columns(
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
    spec = EvaluationSpec(
        evaluation_id="eval_memory_safe",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
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

    def run(policy):
        with runtime_resource_policy_scope(policy):
            state = eval_engine._EvaluationStreamingState(
                spec=spec,
                n_groups=2,
                resource_policy=policy,
            )
            try:
                outputs = list(
                    eval_engine._iter_evaluation_output_chunks(
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
                outputs = [item.value for item in outputs]
                return (
                    outputs,
                    state.ic_summary(),
                    state.group_spread_summary(),
                    state.diagnostics(),
                )
            finally:
                state.close()

    baseline_policy = capture_test_policy(
        available_memory_bytes=128 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    (
        baseline_outputs,
        baseline_ic_summary,
        baseline_group_spread_summary,
        baseline_diagnostics,
    ) = run(baseline_policy)

    assert len(baseline_outputs) == 2
    assert baseline_policy.used_memory_bytes == 0

    original_frame_bytes = memory_owner.pandas_frame_payload_bytes
    monkeypatch.setattr(
        memory_owner,
        "pandas_frame_payload_bytes",
        lambda chunk: 5_000 * original_frame_bytes(chunk),
    )
    forced_policy = capture_test_policy(
        available_memory_bytes=128 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    (
        forced_outputs,
        forced_ic_summary,
        forced_group_spread_summary,
        forced_diagnostics,
    ) = run(forced_policy)

    assert len(forced_outputs) > len(baseline_outputs)
    assert forced_policy.used_memory_bytes == 0
    assert len(seen_columns) > 1

    for index, sort_columns in (
        (0, ("timestamp", "signal", "label")),
        (1, ("timestamp", "signal", "label", "group")),
    ):
        baseline_frame = (
            pd.concat(
                [output[index] for output in baseline_outputs],
                ignore_index=True,
            )
            .sort_values(list(sort_columns), kind="mergesort")
            .reset_index(drop=True)
        )
        forced_frame = (
            pd.concat(
                [output[index] for output in forced_outputs],
                ignore_index=True,
            )
            .sort_values(list(sort_columns), kind="mergesort")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            forced_frame,
            baseline_frame,
            check_exact=False,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )

    for forced_summary, baseline_summary in (
        (forced_ic_summary, baseline_ic_summary),
        (forced_group_spread_summary, baseline_group_spread_summary),
    ):
        pd.testing.assert_frame_equal(
            forced_summary,
            baseline_summary,
            check_exact=False,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    assert forced_diagnostics == baseline_diagnostics


def test_evaluation_merges_adjacent_chunks_that_split_one_timestamp() -> None:
    frame = make_filter_mode_frame()
    spec = EvaluationSpec(
        evaluation_id="eval_split_timestamp",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    with runtime_resource_policy_scope() as expected_policy:
        expected_outputs = list(
            eval_engine._iter_evaluation_output_chunks(
                (frame,),
                spec=spec,
                signal_columns=("factor",),
                label_columns=spec.label_columns,
                n_groups=2,
                asset_column="asset_id",
                resource_policy=expected_policy,
            )
        )
    expected_outputs = [item.value for item in expected_outputs]
    with runtime_resource_policy_scope() as result_policy:
        result_outputs = list(
            eval_engine._iter_evaluation_output_chunks(
                (frame.iloc[:3], frame.iloc[3:]),
                spec=spec,
                signal_columns=("factor",),
                label_columns=spec.label_columns,
                n_groups=2,
                asset_column="asset_id",
                resource_policy=result_policy,
            )
        )
    result_outputs = [item.value for item in result_outputs]

    assert len(result_outputs) == len(expected_outputs)
    result_metrics, result_grouped, result_metadata = result_outputs[0]
    expected_metrics, expected_grouped, expected_metadata = expected_outputs[0]
    pd.testing.assert_frame_equal(result_metrics, expected_metrics)
    pd.testing.assert_frame_equal(
        result_grouped,
        expected_grouped,
    )
    assert result_metadata == expected_metadata


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
    expected_estimator = model_supervised.build_estimator_from_model_spec(spec)
    result_estimator = model_supervised.build_estimator_from_model_spec(spec)
    original_filter = model_fitting._filter_mode_scope
    filtered_chunk_rows: list[int] = []

    with runtime_resource_policy_scope():
        expected_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=expected_estimator,
        )
        try:
            expected = expected_owned.value
            monkeypatch.setattr(
                model_fitting,
                "_filter_mode_scope",
                wrap_filter_for_chunking(original_filter, filtered_chunk_rows),
            )
            result_owned = model_fitting._fit_model_result_owned_from_frames(
                (frame.iloc[:3], frame.iloc[3:]),
                spec,
                estimator=result_estimator,
            )
            try:
                result = result_owned.value
                assert filtered_chunk_rows == [len(frame)]
                assert result.coefficients.tolist() == pytest.approx(
                    expected.coefficients.tolist()
                )
                result_metadata = dict(result.metadata)
                expected_metadata = dict(expected.metadata)
                assert result_metadata.pop("feature_importance") == pytest.approx(
                    expected_metadata.pop("feature_importance")
                )
                assert result_metadata == expected_metadata
            finally:
                result_owned.lease.close()
        finally:
            expected_owned.lease.close()


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
    original_filter = artifacts_module._filter_mode_scope
    filtered_chunk_rows: list[int] = []

    monkeypatch.setattr(
        artifacts_module,
        "_filter_mode_scope",
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
    frame = make_filter_mode_frame()
    first = frame.assign(timestamp=pd.Timestamp("2024-01-02"))
    second = frame.assign(timestamp=pd.Timestamp("2024-01-03"))
    late_overlap = frame.assign(timestamp=pd.Timestamp("2024-01-02"))
    spec = EvaluationSpec(
        evaluation_id="eval_non_monotonic_chunks",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
            tuple(
                eval_engine._iter_evaluation_output_chunks(
                    (first, second, late_overlap),
                    spec=spec,
                    signal_columns=("factor",),
                    label_columns=spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=policy,
                )
            )


def test_evaluation_rejects_pending_non_monotonic_time_overlap() -> None:
    frame = make_filter_mode_frame()
    first = frame.assign(timestamp=pd.Timestamp("2024-01-03"))
    earlier = frame.assign(timestamp=pd.Timestamp("2024-01-02"))
    spec = EvaluationSpec(
        evaluation_id="eval_pending_non_monotonic_chunks",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
            tuple(
                eval_engine._iter_evaluation_output_chunks(
                    (first, earlier),
                    spec=spec,
                    signal_columns=("factor",),
                    label_columns=spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=policy,
                )
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
    import quant_research_platform.research._memory as memory_owner

    frame = make_filter_mode_frame()
    first = frame.iloc[:3]
    second = frame.iloc[3:]
    spec = EvaluationSpec(
        evaluation_id="eval_boundary_over_budget",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )
    boundary_payload_bytes = pandas_frame_payload_bytes(first) + (
        pandas_frame_payload_bytes(second)
    )
    policy = capture_test_policy(
        available_memory_bytes=boundary_payload_bytes * 2,
        explicit_cap_bytes=boundary_payload_bytes,
    )

    def reject_concat(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("boundary concat ran before admission")

    monkeypatch.setattr(memory_owner.pd, "concat", reject_concat)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError,
            match="research_time_boundary_payloads_exceed_task_memory_budget",
        ):
            tuple(
                eval_engine._iter_evaluation_output_chunks(
                    (first, second),
                    spec=spec,
                    signal_columns=("factor",),
                    label_columns=spec.label_columns,
                    n_groups=2,
                    asset_column="asset_id",
                    resource_policy=policy,
                )
            )


def test_evaluation_streaming_output_honors_shared_root_headroom() -> None:
    frame = make_filter_mode_frame()
    spec = EvaluationSpec(
        evaluation_id="eval_shared_headroom",
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
    assert frame_bytes < policy.memory_budget_bytes

    with runtime_resource_policy_scope(policy):
        with policy.lease(
            "outer_owner",
            policy.memory_budget_bytes - frame_bytes,
        ):
            with pytest.raises(
                ValueError,
                match="task_memory_budget",
            ):
                tuple(
                    eval_engine._iter_evaluation_output_chunks(
                        (frame,),
                        spec=spec,
                        signal_columns=("factor",),
                        label_columns=spec.label_columns,
                        n_groups=2,
                        asset_column="asset_id",
                        resource_policy=policy,
                    )
                )


def test_evaluation_rejects_invalid_time_chunk_timestamps() -> None:
    spec = EvaluationSpec(
        evaluation_id="bad_timestamp",
        market_id=MarketId("a_share"),
        research_dataset_id="demo_filter",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
    )

    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="invalid_time_chunk_timestamp_values"):
            tuple(
                eval_engine._iter_evaluation_output_chunks(
                    (
                        make_filter_mode_frame().assign(
                            timestamp=[pd.NaT] * len(make_filter_mode_frame())
                        ),
                    ),
                    spec=spec,
                    signal_columns=("factor",),
                    label_columns=spec.label_columns,
                    n_groups=spec.n_groups,
                    asset_column="asset_id",
                    resource_policy=policy,
                )
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
    estimator = build_estimator_from_model_spec(spec)
    with runtime_resource_policy_scope():
        expected_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,), spec, estimator=estimator
        )
        try:
            expected = expected_owned.value
            expected_coefficients = expected.coefficients.copy()
            expected_metadata = dict(expected.metadata)
        finally:
            expected_owned.lease.close()
    original_filter = model_fitting._filter_mode_scope
    filtered_chunk_rows: list[int] = []

    def fail_column_stack(*_args, **_kwargs):
        raise AssertionError("model fit must not allocate a full intercept matrix")

    monkeypatch.setattr(
        model_fitting,
        "_filter_mode_scope",
        wrap_filter_for_chunking(original_filter, filtered_chunk_rows),
    )
    monkeypatch.setattr(np, "column_stack", fail_column_stack)
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    original_frame_bytes = memory_owner.pandas_frame_payload_bytes
    monkeypatch.setattr(
        memory_owner,
        "pandas_frame_payload_bytes",
        lambda chunk: 10_000 * original_frame_bytes(chunk),
    )
    estimator = build_estimator_from_model_spec(spec)
    with runtime_resource_policy_scope(policy):
        owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=estimator,
        )
        try:
            result = owned.value
            assert result.coefficients.tolist() == pytest.approx(
                expected_coefficients.tolist()
            )
            result_metadata = dict(result.metadata)
            assert result_metadata.pop("feature_importance") == pytest.approx(
                expected_metadata.pop("feature_importance")
            )
            assert result_metadata == expected_metadata
        finally:
            owned.lease.close()

    assert len(filtered_chunk_rows) > 1
    assert max(filtered_chunk_rows) < len(frame)


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
    original_filter = artifacts_module._filter_mode_scope
    filtered_chunk_rows: list[int] = []

    monkeypatch.setattr(
        artifacts_module,
        "_filter_mode_scope",
        wrap_filter_for_chunking(original_filter, filtered_chunk_rows),
    )
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    original_frame_bytes = memory_owner.pandas_frame_payload_bytes
    monkeypatch.setattr(
        memory_owner,
        "pandas_frame_payload_bytes",
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

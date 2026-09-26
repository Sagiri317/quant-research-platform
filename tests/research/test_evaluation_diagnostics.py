from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.evaluations import engine as evaluation_engine
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.output_frames import make_artifact_signal_spec
from tests.helpers.runtime_resources import capture_test_policy


@pytest.mark.parametrize("boundary", ["soft", "captured", "explicit"])
def test_complete_cross_section_admits_kernel_against_capacity(monkeypatch, boundary):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": [f"asset_{i}" for i in range(500)],
            "signal": np.arange(500, dtype=float),
            "label": np.arange(500, dtype=float) * 0.1,
        }
    )
    bound = evaluation_engine._evaluation_batch_peak_bytes(
        frame,
        compute_columns_by_label=(("timestamp", "asset_id", "signal", "label"),),
        signal_count=1,
        n_groups=2,
    )
    policy = capture_test_policy(
        available_memory_bytes=bound - 1 if boundary == "captured" else bound * 5 // 4,
        explicit_cap_bytes=bound - 1 if boundary == "explicit" else None,
    )
    calls = []
    kernel = evaluation_engine.compute_ic_and_grouped_label_values

    def observed_kernel(*args, **kwargs):
        calls.append(True)
        return kernel(*args, **kwargs)

    monkeypatch.setattr(
        evaluation_engine, "compute_ic_and_grouped_label_values", observed_kernel
    )
    with (
        runtime_resource_policy_scope(policy),
        closing_iterator(
            evaluation_engine._iter_evaluation_filtered_chunks(
                frame,
                {
                    "rows_input": len(frame),
                    "rows_after_filter_mode": len(frame),
                    "filter_columns_used": [],
                    "filter_definition": "all",
                },
                signal_columns=("signal",),
                label_columns=("label",),
                n_groups=2,
                asset_column="asset_id",
                resource_policy=policy,
            )
        ) as outputs,
    ):
        if boundary == "soft":
            assert bound > policy.memory_budget_bytes
            owned = next(outputs)
            assert owned.value[0].n_obs.tolist() == [500]
            assert owned.value[0].ic.iloc[0] == pytest.approx(1.0)
            owned.release(owner="research.evaluation.output")
            with pytest.raises(StopIteration):
                next(outputs)
            assert calls == [True]
        else:
            with pytest.raises(
                ValueError,
                match="research_evaluation_working_set_exceeds_task_memory_budget",
            ):
                next(outputs)
            assert calls == []
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("valid_pairs", (None, 0, 1))
def test_single_asset_insufficient_pairs_publish_with_rankic_not_applicable(
    tmp_path: Path, valid_pairs: int | None
) -> None:
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
    from quant_research_platform.storage.layout import StorageLayout

    spec = EvaluationSpec(
        evaluation_id="single_undefined",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    output = OutputSpec(
        output_id="single_undefined",
        output_definition_id="evaluation_outputs",
        run_id="test_run",
        market_id=spec.market_id,
        run_type="evaluation",
        scope=OutputScope.LOCAL_ONLY,
    )
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-02")],
            "asset_id": "asset",
            "feature": 1.0,
            "label": 2.0 if valid_pairs else np.nan,
            "eligible": True,
        }
    )
    if valid_pairs is None:
        frame = frame.iloc[:0]
    with runtime_resource_policy_scope() as policy:
        with evaluation_engine.build_evaluation_result_from_frames(
            (frame,),
            spec=spec,
            signal_columns=("signal",),
            label_columns=("label",),
            n_groups=2,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("feature",)},
            resource_policy=policy,
        ) as result:
            root = write_evaluation_outputs_from_chunks(
                result.output_chunks,
                summary_factory=result.summaries,
                spec=spec,
                output_spec=output,
                layout=StorageLayout(tmp_path),
                n_groups=2,
            )
        metrics = pd.read_parquet(root / "metrics.parquet")
        assert metrics.n_obs.tolist() == ([] if valid_pairs is None else [valid_pairs])
        assert (
            metrics.ic.isna().all()
            and metrics.ic_reason.eq("insufficient_observations").all()
        )
        assert (
            metrics.rankic.isna().all()
            and metrics.rankic_reason.eq("not_applicable").all()
        )
        assert policy.used_memory_bytes == 0


def test_nanosecond_evaluation_diagnostics_survive_actual_writer_roundtrip(
    tmp_path: Path,
) -> None:
    from quant_research_platform.artifacts.outputs.bundle_validation import (
        validate_output_bundle,
    )
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
    from quant_research_platform.storage.layout import StorageLayout

    times = pd.Timestamp("2020-01-01") + pd.to_timedelta([1, 2], unit="ns")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": "asset",
            "feature": [1.0, 2.0],
            "label": [2.0, 4.0],
            "eligible": True,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="nanosecond_eval",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    output = OutputSpec(
        output_id=spec.evaluation_id,
        output_definition_id="evaluation_outputs",
        run_id="test_run",
        market_id=spec.market_id,
        run_type="evaluation",
        scope=OutputScope.LOCAL_ONLY,
    )
    with runtime_resource_policy_scope() as policy:
        with evaluation_engine.build_evaluation_result_from_frames(
            (frame,),
            spec=spec,
            signal_columns=("signal",),
            label_columns=("label",),
            n_groups=2,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("feature",)},
            resource_policy=policy,
        ) as result:
            root = write_evaluation_outputs_from_chunks(
                result.output_chunks,
                summary_factory=result.summaries,
                spec=spec,
                output_spec=output,
                layout=StorageLayout(tmp_path),
                n_groups=2,
            )
        validate_output_bundle(root)
        diagnostics = json.loads((root / "manifest.json").read_text(encoding="utf-8"))[
            "metadata"
        ]["evaluation_diagnostics"]
        assert diagnostics["min_timestamp"] == times[0].isoformat()
        assert diagnostics["max_timestamp"] == times[-1].isoformat()
        assert policy.used_memory_bytes == 0


def test_evaluation_chunk_producer_does_not_terminal_concat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = EvaluationSpec(
        evaluation_id="eval_stream",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    monkeypatch.setattr(
        pd,
        "concat",
        lambda *_args, **_kwargs: pytest.fail(
            "single_label_managed_evaluation_must_not_concat"
        ),
    )
    policy = capture_test_policy(available_memory_bytes=256 * 1024)
    with runtime_resource_policy_scope(policy):
        chunks = evaluation_engine._iter_evaluation_output_chunks(
            (_evaluation_frame(),),
            spec=spec,
            signal_columns=("signal",),
            label_columns=("label",),
            n_groups=2,
            asset_column="asset_id",
            resource_policy=policy,
        )
        owned = next(chunks)
        metrics, grouped, _ = owned.value
        chunks.close()

    assert list(metrics.columns) == [
        "timestamp",
        "signal",
        "label",
        "ic",
        "rankic",
        "n_obs",
        "ic_reason",
        "rankic_reason",
    ]
    assert list(grouped.columns) == [
        "timestamp",
        "signal",
        "label",
        "group",
        "mean_label_value",
        "n_obs",
    ]


def test_single_asset_time_series_evaluation_is_exact_across_chunks() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="100ms"),
            "asset_id": ["btc_usdt"] * 4,
            "feature": [1.0, 2.0, 3.0, 4.0],
            "label": [2.0, 4.0, 6.0, 8.0],
            "eligible": [True] * 4,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="single_asset_time_series",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    outputs: list[pd.DataFrame] = []
    for frames in ((frame,), (frame.iloc[:2], frame.iloc[2:])):
        policy = capture_test_policy(available_memory_bytes=256 * 1024)
        with runtime_resource_policy_scope(policy):
            with evaluation_engine.build_evaluation_result_from_frames(
                frames,
                spec=spec,
                signal_columns=("signal",),
                label_columns=("label",),
                n_groups=2,
                asset_column="asset_id",
                source_columns_by_kind={"feature": ("feature",)},
                resource_policy=policy,
            ) as result:
                chunks = iter(result.output_chunks)
                owned = next(chunks)
                metrics, grouped, metadata = owned.value
                with pytest.raises(StopIteration):
                    next(chunks)
                summary = result.summaries()[1]
        assert grouped.empty
        assert metadata["timestamp_count"] == 4
        assert metrics.loc[0, "n_obs"] == 4
        assert metrics.loc[0, "ic"] == pytest.approx(1.0)
        assert pd.isna(metrics.loc[0, "rankic"])
        assert metrics.loc[0, "rankic_reason"] == "not_applicable"
        assert summary.loc[0, "valid_ic_count"] == 1
        assert summary.loc[0, "valid_rankic_count"] == 0
        outputs.append(metrics)

    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)


@pytest.mark.parametrize("case", ("ordinary", "missing", "extreme"))
def test_single_asset_time_series_values_are_chunk_exact(case: str) -> None:
    row_count = 10_007
    random = np.random.default_rng(0)
    if case == "extreme":
        feature = random.normal(1e9, 1e-3, size=row_count)
        label = feature * 0.17 + random.normal(-1e8, 2e-3, size=row_count)
    else:
        feature = random.normal(size=row_count)
        label = random.normal(size=row_count)
        if case == "missing":
            feature[::997] = np.nan
            label[::613] = np.nan
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=row_count,
                freq="100ms",
            ),
            "asset_id": ["btc_usdt"] * row_count,
            "feature": feature,
            "label": label,
            "eligible": [True] * row_count,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="single_asset_chunk_exact",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )

    def evaluate(frames) -> pd.DataFrame:  # noqa: ANN001
        policy = capture_test_policy(available_memory_bytes=512 * 1024)
        with runtime_resource_policy_scope(policy):
            with evaluation_engine.build_evaluation_result_from_frames(
                frames,
                spec=spec,
                signal_columns=("signal",),
                label_columns=("label",),
                n_groups=2,
                asset_column="asset_id",
                source_columns_by_kind={"feature": ("feature",)},
                resource_policy=policy,
            ) as result:
                chunks = iter(result.output_chunks)
                metrics = next(chunks).value[0].copy()
                with pytest.raises(StopIteration):
                    next(chunks)
        return metrics

    chunks_127 = evaluate(
        tuple(frame.iloc[start : start + 127] for start in range(0, row_count, 127))
    )
    chunks_251 = evaluate(
        tuple(frame.iloc[start : start + 251] for start in range(0, row_count, 251))
    )

    pd.testing.assert_frame_equal(chunks_127, chunks_251, check_exact=True)


@pytest.mark.parametrize("epsilon", (1e-7, 1e-8))
def test_streaming_ic_summary_matches_stable_canonical_values(
    epsilon: float,
) -> None:
    values = (1.0 - epsilon, 1.0, 1.0 - 2 * epsilon)
    metrics = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=3),
            "signal": ["signal"] * 3,
            "label": ["label"] * 3,
            "ic": values,
            "rankic": values,
            "n_obs": [2, 2, 2],
            "ic_reason": [None, None, None],
            "rankic_reason": [None, None, None],
        }
    )
    spec = EvaluationSpec(
        evaluation_id="stable_ic_summary",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
        n_groups=2,
    )
    try:
        state.consume(
            metrics,
            pd.DataFrame(),
            {
                "rows_input": 6,
                "rows_after_filter_mode": 6,
                "filter_columns_used": ["eligible"],
                "filter_definition": "eligible == True",
                "timestamp_count": 3,
                "min_timestamp": "2024-01-02",
                "max_timestamp": "2024-01-04",
                "skipped_timestamp_count": 0,
            },
        )
        streaming = state.ic_summary().iloc[0]
    finally:
        state.close()

    expected_mean = sum(values) / len(values)
    expected_std = (
        sum((value - expected_mean) ** 2 for value in values) / len(values)
    ) ** 0.5
    expected_ratio = expected_mean / expected_std
    for column in ("mean_ic", "std_ic", "mean_rankic", "std_rankic"):
        expected = {
            "mean_ic": expected_mean,
            "std_ic": expected_std,
            "mean_rankic": expected_mean,
            "std_rankic": expected_std,
        }[column]
        assert streaming[column] == pytest.approx(
            expected,
            rel=0.0,
            abs=1e-12,
        )
    for column in ("icir", "rankicir"):
        assert streaming[column] == pytest.approx(
            expected_ratio,
            rel=1e-8,
            abs=1e-12,
        )


def test_streaming_group_spread_reducer_avoids_chunk_materialization() -> None:
    grouped = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "signal": ["signal"] * 4,
            "label": ["label"] * 4,
            "group": [1, 2, 1, 2],
            "mean_label_value": [0.1, 0.3, 0.2, 0.1],
            "n_obs": [2, 2, 2, 2],
        }
    )
    spec = EvaluationSpec(
        evaluation_id="streaming_group_spread",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(
        spec=spec,
        n_groups=2,
    )
    try:
        state.consume(
            pd.DataFrame(),
            grouped,
            {
                "rows_input": 4,
                "rows_after_filter_mode": 4,
                "filter_columns_used": ["eligible"],
                "filter_definition": "eligible == True",
                "timestamp_count": 2,
                "min_timestamp": "2024-01-02",
                "max_timestamp": "2024-01-03",
                "skipped_timestamp_count": 0,
            },
        )
        actual = state.group_spread_summary()
    finally:
        state.close()

    assert len(actual) == 1
    row = actual.iloc[0]
    assert row["signal"] == "signal"
    assert row["label"] == "label"
    assert row["top_group_mean_label_value"] == pytest.approx(0.2)
    assert row["bottom_group_mean_label_value"] == pytest.approx(0.15)
    assert row["top_bottom_label_value_spread_mean"] == pytest.approx(0.05)
    assert row["top_bottom_label_value_spread_hit_rate"] == pytest.approx(0.5)
    assert row["group_monotonicity_score"] == pytest.approx(0.0)
    assert pd.isna(row["group_monotonicity_score_reason"])


def _evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset_id": ["asset_a", "asset_b"],
            "feature": [1.0, 2.0],
            "label": [0.1, 0.2],
            "eligible": [True, True],
        }
    )


def test_evaluation_writer_roundtrips_long_windows_path(tmp_path):
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
    from quant_research_platform.storage.identity import native_filesystem_path
    from quant_research_platform.storage.layout import StorageLayout

    layout = StorageLayout(tmp_path / ("long" * 25) / ("path" * 25))
    spec = EvaluationSpec(
        evaluation_id="eval_long",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
        n_groups=2,
    )
    output = OutputSpec(
        output_id=spec.evaluation_id,
        output_definition_id="evaluation_outputs",
        run_id="test_run",
        market_id=spec.market_id,
        run_type="evaluation",
        scope=OutputScope.LOCAL_ONLY,
    )
    with runtime_resource_policy_scope() as policy:
        with evaluation_engine.build_evaluation_result_from_frames(
            (_evaluation_frame(),),
            spec=spec,
            signal_columns=("signal",),
            label_columns=("label",),
            n_groups=2,
            asset_column="asset_id",
            source_columns_by_kind={"feature": ("feature",)},
            resource_policy=policy,
        ) as result:
            root = write_evaluation_outputs_from_chunks(
                result.output_chunks,
                summary_factory=result.summaries,
                spec=spec,
                output_spec=output,
                layout=layout,
                n_groups=2,
            )
        assert len(str(root / "grouped_label_values.parquet")) > 260
        with pytest.raises(ValueError, match="output_already_exists"):
            write_evaluation_outputs_from_chunks(
                (),
                summary_factory=lambda: pytest.fail("existing_output_was_rewritten"),
                spec=spec,
                output_spec=output,
                layout=layout,
                n_groups=2,
            )
        metrics = pd.read_parquet(native_filesystem_path(root / "metrics.parquet"))
        assert metrics.n_obs.tolist() == [2]
        assert metrics.ic.iloc[0] == pytest.approx(1.0)
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_evaluation_initialization_error_releases_operation_with_traceback_retained(
    tmp_path, monkeypatch, cleanup_fails
):
    from contextlib import ExitStack

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.research.evaluations import runner
    from quant_research_platform.runtime.resources import (
        current_runtime_resource_policy,
    )
    from quant_research_platform.storage.layout import StorageLayout

    if cleanup_fails:

        class ClosingFailure(ExitStack):
            def close(self):
                super().close()
                raise RuntimeError("secondary_close")

        monkeypatch.setattr(runner, "ExitStack", ClosingFailure)
    profile = research_profile()
    retained = None
    try:
        with pytest.raises(
            PermissionError, match="research_dataset_access_window_forbidden"
        ) as caught:
            runner.build_evaluation_result_from_research_dataset(
                layout=StorageLayout(tmp_path),
                dataset_catalog=profile.research_datasets,
                evaluation_catalog=profile.evaluations,
                evaluation_id=profile.evaluations.list()[0].evaluation_id,
                time_window=TimeWindow("2024-01-02", "2024-01-03"),
            )
        retained = caught.value
        assert retained.__traceback__ is not None
        assert current_runtime_resource_policy() is None
        if cleanup_fails:
            assert any(
                "evaluation operation cleanup also failed" in note
                for note in retained.__notes__
            )
        with runtime_resource_policy_scope() as policy:
            assert policy.used_memory_bytes == 0
    finally:
        if current_runtime_resource_policy() is not None and retained is not None:
            trace = retained.__traceback__
            while trace is not None:
                if (
                    trace.tb_frame.f_code.co_name
                    == "build_evaluation_result_from_research_dataset"
                ):
                    trace.tb_frame.f_locals["operation"].close()
                    break
                trace = trace.tb_next


@pytest.mark.parametrize(
    "values",
    [
        (1.0, float(np.nextafter(1.0, 0.0))),
        (1e-200, 2e-200, 3e-200),
        (0.0, float(np.nextafter(0.0, 1.0))),
        (0.0, 0.0, float(np.nextafter(0.0, 1.0))),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 3])
def test_ic_summary_nonzero_variance_survives_rounding_and_persistence(
    values, reverse, chunk_size, tmp_path
):
    from statistics import mean, pstdev

    values = values[::-1] if reverse else values
    metrics = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=len(values)),
            "signal": "signal",
            "label": "label",
            "ic": values,
            "rankic": values,
            "n_obs": 2,
            "ic_reason": None,
            "rankic_reason": None,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="stable_ic_summary",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    state = evaluation_engine._EvaluationStreamingState(spec=spec, n_groups=2)
    try:
        for start in range(0, len(metrics), chunk_size):
            state.consume(
                metrics.iloc[start : start + chunk_size],
                pd.DataFrame(),
                {
                    "rows_input": 2,
                    "rows_after_filter_mode": 2,
                    "filter_columns_used": ["eligible"],
                    "filter_definition": "eligible",
                    "timestamp_count": 1,
                    "min_timestamp": "2024-01-02",
                    "max_timestamp": "2024-01-04",
                    "skipped_timestamp_count": 0,
                },
            )
        summary = state.ic_summary()
    finally:
        state.close()
    path = tmp_path / "summary.parquet"
    summary.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(restored, summary, check_exact=True)
    row = restored.iloc[0]
    expected_std = pstdev(values)
    for prefix in ("ic", "rankic"):
        if expected_std == 0.0:
            assert len(set(values)) > 1
            assert pd.isna(row[f"std_{prefix}"])
            assert pd.isna(row[f"{prefix}ir"])
            assert row[f"{prefix}ir_reason"] == "nonfinite_result"
        else:
            assert row[f"std_{prefix}"] == pytest.approx(expected_std, rel=1e-15, abs=0)
            assert row[f"{prefix}ir"] == pytest.approx(
                mean(values) / expected_std, rel=1e-15
            )


def test_persisted_unfiltered_evaluation_keeps_ineligible_mature_labels(tmp_path):
    from quant_research_platform.artifacts.outputs.bundle_validation import (
        validate_output_bundle,
    )
    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
    from quant_research_platform.catalog.evaluations import EvaluationCatalog
    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.evaluations.runner import (
        build_evaluation_result_from_research_dataset,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.research_specs import make_demo_research_dataset_spec

    layout = StorageLayout(tmp_path)
    dataset = make_demo_research_dataset_spec(
        data_id="maturity_fixture",
        factor_columns=("signal",),
        label_columns=("label", "available_at"),
        filter_columns=("eligible",),
        context_columns=(),
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b", "c", "d"],
            "timestamp": pd.Timestamp("2023-01-02"),
            "signal": [1.0, 2.0, 3.0, 100.0],
            "label": [2.0, 4.0, 6.0, -100.0],
            "available_at": pd.to_datetime(["2023-01-02"] * 3 + ["2023-01-04"]),
            "eligible": [True, False, True, True],
        }
    )
    with runtime_resource_policy_scope() as policy:
        write_dataset(
            layout.research_dataset_root(dataset.market_id, dataset.data_id),
            frame,
            spec=dataset,
            resource_policy=policy,
            borrowed_input=True,
        )
        for mode, expected_rows in (("unfiltered", 3), ("eligible", 2)):
            spec = EvaluationSpec(
                evaluation_id=mode,
                market_id=dataset.market_id,
                research_dataset_id=dataset.data_id,
                signals=(
                    SignalSpec(
                        signal_id="signal", source_column="signal", source_kind="factor"
                    ),
                ),
                label_columns=("label",),
                label_availability_columns={"label": "available_at"},
                filter_mode=mode,
                n_groups=2,
            )
            with build_evaluation_result_from_research_dataset(
                layout=layout,
                dataset_catalog=ResearchDatasetCatalog((dataset,)),
                evaluation_catalog=EvaluationCatalog((spec,)),
                evaluation_id=mode,
                time_window=TimeWindow("2023-01-02", "2023-01-03"),
                resource_policy=policy,
            ) as result:
                output = write_evaluation_outputs_from_chunks(
                    result.output_chunks,
                    summary_factory=result.summaries,
                    spec=spec,
                    output_spec=OutputSpec(
                        output_id=mode,
                        output_definition_id="evaluation_outputs",
                        run_id="test_run",
                        market_id=dataset.market_id,
                        run_type="evaluation",
                        scope=OutputScope.LOCAL_ONLY,
                    ),
                    layout=layout,
                    n_groups=2,
                )
            validate_output_bundle(
                output, expected_output_id=mode, expected_run_id="test_run"
            )
            metrics = pd.read_parquet(output / "metrics.parquet")
            assert metrics.n_obs.tolist() == [expected_rows]
            assert metrics.ic.tolist() == pytest.approx([1.0])
            assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    ("signal_id", "reason"),
    [
        ('{"token":\n"SYNTHETIC_SECRET"}', "credential"),
        ('{"token"\n \n:\n"SYNTHETIC_SECRET"}', "credential"),
        ('{"_token":"SYNTHETIC_SECRET"}', "credential"),
        ("MARKET_ID=synthetic\nordinary\nASSET_ID=synthetic", "row_sample"),
        ("market_id\n=synthetic\nordinary\nasset_id=synthetic", "row_sample"),
        ("asset_id\n=synthetic\nordinary\ntimestamp\n=synthetic", "row_sample"),
    ],
)
def test_evaluation_writer_rejects_unsafe_signal_before_publication(
    tmp_path: Path, signal_id: str, reason: str
) -> None:
    from dataclasses import replace

    from quant_research_platform.artifacts.outputs.evaluation import (
        write_evaluation_outputs_from_chunks,
    )
    from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
    from quant_research_platform.storage.layout import StorageLayout

    spec = EvaluationSpec(
        evaluation_id="unsafe_signal",
        market_id=MarketId("crypto"),
        research_dataset_id="hft_research",
        signals=(replace(make_artifact_signal_spec(), signal_id=signal_id),),
        label_columns=("label",),
        evaluation_axis="single_asset_time_series",
        n_groups=2,
    )
    output = OutputSpec(
        output_id=spec.evaluation_id,
        output_definition_id="evaluation_outputs",
        run_id="test_run",
        market_id=spec.market_id,
        run_type="evaluation",
        scope=OutputScope.LOCAL_ONLY,
    )
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-02")],
            "asset_id": "asset",
            "feature": 1.0,
            "label": 2.0,
            "eligible": True,
        }
    )
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(
            ValueError, match=f"evaluation_summary_share_unsafe: {reason}"
        ):
            with evaluation_engine.build_evaluation_result_from_frames(
                (frame,),
                spec=spec,
                signal_columns=(signal_id,),
                label_columns=("label",),
                n_groups=2,
                asset_column="asset_id",
                source_columns_by_kind={"feature": ("feature",)},
                resource_policy=policy,
            ) as result:
                write_evaluation_outputs_from_chunks(
                    result.output_chunks,
                    summary_factory=result.summaries,
                    spec=spec,
                    output_spec=output,
                    layout=StorageLayout(tmp_path),
                    n_groups=2,
                )
        assert policy.used_memory_bytes == 0
    assert not list(tmp_path.rglob("summary.md"))
    assert not list(tmp_path.rglob("manifest.json"))

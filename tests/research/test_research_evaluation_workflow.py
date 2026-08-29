from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.research.evaluations.runner as evaluation_runner_module
import quant_research_platform.research.predictions.runner as prediction_runner_module
import quant_research_platform.workflows.local_verification as local_verification_module
from quant_research_platform.artifacts.outputs.evaluation import (
    write_evaluation_outputs_from_chunks,
)
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.evaluations.engine import (
    _EvaluationStreamingState,
    _iter_evaluation_output_chunks,
    _iter_evaluation_work_frames,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.local_verification import (
    _write_evaluation_outputs_from_research_dataset,
)
from tests.helpers.research_processed_inputs import (
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import (
    make_research_dataset_catalog,
    make_signal_spec,
)
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
    capture_test_policy,
)
from tests.research.filter_mode_fixtures import make_filter_mode_frame
from tests.research.workflow_test_support import (
    _output_spec,
    _research_rows_for_dates,
)


def test_evaluation_input_lease_spans_normalization_and_closes_on_stop() -> None:
    frame = make_filter_mode_frame().assign(normalized_factor=999.0)
    spec = EvaluationSpec(
        evaluation_id="eval_input_lease",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_signal_spec("factor", signal_id="normalized_factor"),),
        label_columns=("label",),
    )
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        chunks = _iter_evaluation_work_frames(
            (frame,),
            spec=spec,
            label_columns=spec.label_columns,
            asset_column="asset_id",
            resource_policy=policy,
        )
        normalized = next(chunks)
        assert "normalized_factor" in normalized.columns
        assert policy.used_memory_bytes > 0
        chunks.close()
    assert policy.used_memory_bytes == 0


def test_managed_workflow_admits_summary_before_research_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_catalog = make_research_dataset_catalog()
    spec = EvaluationSpec(
        evaluation_id="eval_summary_admission",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("mom_20d"),),
        label_columns=("forward_return_20d",),
    )
    evaluation_catalog = EvaluationCatalog((spec,))
    pulled: list[bool] = []
    artifact_calls: list[bool] = []

    def chunks(*_args, **_kwargs):
        pulled.append(True)
        yield pd.DataFrame()

    def artifact_writer(*_args, **_kwargs):
        artifact_calls.append(True)
        raise AssertionError("artifact_writer_after_summary_admission")

    monkeypatch.setattr(
        evaluation_runner_module,
        "iter_research_dataset_chunks",
        chunks,
    )
    monkeypatch.setattr(
        local_verification_module,
        "write_evaluation_outputs_from_chunks",
        artifact_writer,
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024,
        explicit_cap_bytes=32 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            _write_evaluation_outputs_from_research_dataset(
                layout=StorageLayout(project_root=tmp_path),
                dataset_catalog=dataset_catalog,
                evaluation_catalog=evaluation_catalog,
                evaluation_id=spec.evaluation_id,
                output_spec=_output_spec(
                    output_definition_id="evaluation_outputs",
                    output_id=spec.evaluation_id,
                    market_id=spec.market_id,
                    run_type=spec.run_type,
                ),
                resource_policy=policy,
            )

    assert pulled == []
    assert artifact_calls == []
    assert policy.used_memory_bytes == 0


def test_evaluation_consumer_failure_closes_dataset_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_catalog = make_research_dataset_catalog()
    spec = EvaluationSpec(
        evaluation_id="eval_consumer_failure",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("mom_20d"),),
        label_columns=("forward_return_20d",),
    )
    staging = tmp_path / "held-staging"
    staging.mkdir()
    held_path = staging / "reader.bin"
    held_path.write_bytes(b"reader")
    closed: list[bool] = []

    def dataset_chunks():
        handle = held_path.open("rb")
        try:
            yield _research_rows_for_dates(["2014-01-02", "2014-01-03"])
        finally:
            handle.close()
            closed.append(True)

    child = dataset_chunks()

    def fail_after_first(chunks, **_kwargs):
        with closing_iterator(iter(chunks)) as active:
            next(active)
            raise KeyboardInterrupt("consumer_failure")

    monkeypatch.setattr(
        evaluation_runner_module,
        "iter_research_dataset_chunks",
        lambda *_args, **_kwargs: child,
    )
    monkeypatch.setattr(
        local_verification_module,
        "write_evaluation_outputs_from_chunks",
        fail_after_first,
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024**2,
    )

    try:
        with pytest.raises(KeyboardInterrupt, match="consumer_failure"):
            _write_evaluation_outputs_from_research_dataset(
                layout=StorageLayout(project_root=tmp_path),
                dataset_catalog=dataset_catalog,
                evaluation_catalog=EvaluationCatalog((spec,)),
                evaluation_id=spec.evaluation_id,
                output_spec=local_verification_module._output_spec(
                    output_definition_id="evaluation_outputs",
                    output_id=spec.evaluation_id,
                    run_id="test_run",
                    market_id=spec.market_id,
                    run_type=spec.run_type,
                ),
                resource_policy=policy,
            )

        assert closed == [True]
        assert policy.used_memory_bytes == 0
        held_path.unlink()
        staging.rmdir()
        fresh_policy = capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024**2,
        )
        with runtime_resource_policy_scope(fresh_policy):
            assert current_runtime_resource_policy() is fresh_policy
    finally:
        child.close()
        if held_path.exists():
            held_path.unlink()
        if staging.exists():
            staging.rmdir()


def test_prediction_consumer_failure_closes_dataset_child() -> None:
    dataset_spec = make_research_dataset_catalog().get("demo")
    closed: list[bool] = []

    def dataset_chunks():
        try:
            yield _research_rows_for_dates(["2014-01-02"])
        finally:
            closed.append(True)

    child = dataset_chunks()
    chunks = prediction_runner_module._validated_prediction_chunks(
        child,
        dataset_spec=dataset_spec,
        input_columns=("mom_20d",),
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="consumer_failure"):
            with closing_iterator(iter(chunks)) as active:
                next(active)
                raise KeyboardInterrupt("consumer_failure")
        assert closed == [True]
    finally:
        child.close()


def test_workflow_delegates_evaluation_lifecycle_to_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_catalog = make_research_dataset_catalog()
    spec = EvaluationSpec(
        evaluation_id="eval_result_owner",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("mom_20d"),),
        label_columns=("forward_return_20d",),
    )
    events: list[str] = []

    class Result:
        output_chunks = object()

        def __init__(self, frames) -> None:  # noqa: ANN001
            self.frames = frames

        def __enter__(self):  # noqa: ANN204
            events.append("enter")
            return self

        def __exit__(self, *_args) -> None:
            events.append("exit")
            self.frames.close()

        def summaries(self):  # noqa: ANN201
            return ("owned-summary",)

    def build(frames, **_kwargs):  # noqa: ANN001, ANN202
        events.append("build")
        return Result(frames)

    def write(chunks, *, summary_factory, **_kwargs):  # noqa: ANN001, ANN202
        events.append("write")
        assert chunks is Result.output_chunks
        assert summary_factory() == ("owned-summary",)
        return tmp_path / "bundle"

    monkeypatch.setattr(
        evaluation_runner_module,
        "build_evaluation_result_from_frames",
        build,
    )
    monkeypatch.setattr(
        evaluation_runner_module,
        "iter_research_dataset_chunks",
        lambda *_args, **_kwargs: iter((_research_rows_for_dates(["2014-01-02"]),)),
    )
    monkeypatch.setattr(
        local_verification_module,
        "write_evaluation_outputs_from_chunks",
        write,
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    assert (
        _write_evaluation_outputs_from_research_dataset(
            layout=StorageLayout(project_root=tmp_path),
            dataset_catalog=dataset_catalog,
            evaluation_catalog=EvaluationCatalog((spec,)),
            evaluation_id=spec.evaluation_id,
            output_spec=local_verification_module._output_spec(
                output_definition_id="evaluation_outputs",
                output_id=spec.evaluation_id,
                run_id="test_run",
                market_id=spec.market_id,
                run_type=spec.run_type,
            ),
            resource_policy=policy,
        )
        == tmp_path / "bundle"
    )
    assert events == ["build", "enter", "write", "exit"]


def test_managed_evaluation_runner_persists_without_full_result(
    tmp_path: Path,
) -> None:
    frame = _research_rows_for_dates(["2014-01-02", "2014-01-03", "2014-01-06"])
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    assembly_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-06"),
        resource_policy=assembly_policy,
    )
    spec = EvaluationSpec(
        evaluation_id="eval_managed",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("mom_20d"),),
        label_columns=("forward_return_20d",),
    )
    evaluation_catalog = EvaluationCatalog((spec,))
    evaluation_policy = capture_test_policy(
        available_memory_bytes=32 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    with runtime_resource_policy_scope(evaluation_policy):
        output_bundle = _write_evaluation_outputs_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            evaluation_catalog=evaluation_catalog,
            evaluation_id=spec.evaluation_id,
            output_spec=_output_spec(
                output_definition_id="evaluation_outputs",
                output_id=spec.evaluation_id,
                market_id=spec.market_id,
                run_type=spec.run_type,
            ),
            time_window=TimeWindow("2014-01-02", "2014-01-06"),
            resource_policy=evaluation_policy,
        )

    assert evaluation_policy.used_memory_bytes == 0
    metrics = pd.read_parquet(output_bundle / "metrics.parquet")
    grouped = pd.read_parquet(output_bundle / "grouped_label_values.parquet")
    assert len(metrics) == 3
    assert len(grouped) == 6
    assert len(pd.read_parquet(output_bundle / "ic_summary.parquet")) == 1
    assert len(pd.read_parquet(output_bundle / "group_spread_summary.parquet")) == 1


@pytest.mark.parametrize(
    ("filter_mode", "case"),
    (
        ("eligible", "normal"),
        ("tradable", "normal"),
        ("unfiltered", "ties"),
        ("unfiltered", "all_nan"),
    ),
)
def test_managed_evaluation_parity_covers_filters_ties_and_all_nan(
    tmp_path: Path,
    filter_mode: str,
    case: str,
) -> None:
    frame = make_filter_mode_frame()
    if case == "ties":
        frame["factor"] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
        frame["label"] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    elif case == "all_nan":
        frame["factor"] = float("nan")
        frame["label"] = float("nan")
    spec = EvaluationSpec(
        evaluation_id=f"eval_managed_{filter_mode}_{case}",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_signal_spec("factor"),),
        label_columns=("label",),
        filter_mode=filter_mode,
    )
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )
    output_spec = _output_spec(
        output_definition_id="evaluation_outputs",
        output_id=spec.evaluation_id,
        market_id=spec.market_id,
        run_type=spec.run_type,
    )
    with runtime_resource_policy_scope(policy):
        state = _EvaluationStreamingState(
            spec=spec,
            n_groups=2,
            resource_policy=policy,
        )
        try:
            chunks = _iter_evaluation_output_chunks(
                (frame,),
                spec=spec,
                signal_columns=("factor",),
                label_columns=spec.label_columns,
                n_groups=2,
                asset_column="asset_id",
                resource_policy=policy,
                summary_state=state,
            )
            output_bundle = write_evaluation_outputs_from_chunks(
                chunks,
                summary_factory=lambda: (
                    state.diagnostics(),
                    state.ic_summary(),
                    state.group_spread_summary(),
                    state.coverage_summary(),
                ),
                spec=spec,
                output_spec=output_spec,
                layout=StorageLayout(project_root=tmp_path),
                n_groups=2,
            )
        finally:
            state.close()

    metrics = pd.read_parquet(output_bundle / "metrics.parquet")
    grouped = pd.read_parquet(output_bundle / "grouped_label_values.parquet")
    spread = pd.read_parquet(output_bundle / "group_spread_summary.parquet")
    if case == "all_nan":
        assert metrics["n_obs"].tolist() == [0]
        assert grouped.empty
        assert spread.empty
        expected_dropna = 0
        expected_grouped_rows = 0
    else:
        assert metrics["ic"].notna().any()
        assert len(grouped) == 2
        assert len(spread) == 1
        expected_dropna = {"eligible": 4, "tradable": 2}.get(filter_mode, 6)
        expected_grouped_rows = 2
    manifest = json.loads((output_bundle / "manifest.json").read_text())
    diagnostics = manifest["metadata"]["evaluation_diagnostics"]
    assert diagnostics["filter_mode"] == filter_mode
    assert diagnostics["rows_input"] == 6
    assert diagnostics["rows_after_signal_label_dropna"] == expected_dropna
    assert diagnostics["grouped_label_value_rows"] == expected_grouped_rows
    assert policy.used_memory_bytes == 0


def test_managed_evaluation_two_label_order_and_shared_skip_parity(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")] * 4
            + [pd.Timestamp("2024-01-03")] * 4
            + [pd.Timestamp("2024-01-04")] * 4,
            "asset_id": [f"asset_{index}" for index in range(4)] * 3,
            "factor": [1.0, 2.0, 3.0, 4.0] * 3,
            "label_a": [0.1, 0.2, float("nan"), float("nan")] * 2
            + [0.1, float("nan"), float("nan"), float("nan")],
            "label_b": [0.2, 0.4, float("nan"), float("nan")] * 2
            + [0.2, float("nan"), float("nan"), float("nan")],
            "eligible": [True] * 12,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="eval_managed_two_labels",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_signal_spec("factor"),),
        label_columns=("label_a", "label_b"),
        n_groups=2,
    )

    def persist(
        frames: tuple[pd.DataFrame, ...],
        output_id: str,
    ) -> tuple[Path, RuntimeResourcePolicy]:
        policy = capture_test_policy(
            available_memory_bytes=32 * 1024**2,
            explicit_cap_bytes=16 * 1024**2,
        )
        output_spec = _output_spec(
            output_definition_id="evaluation_outputs",
            output_id=output_id,
            market_id=spec.market_id,
            run_type=spec.run_type,
        )
        with runtime_resource_policy_scope(policy):
            state = _EvaluationStreamingState(
                spec=spec,
                n_groups=2,
                resource_policy=policy,
            )
            try:
                output_bundle = write_evaluation_outputs_from_chunks(
                    _iter_evaluation_output_chunks(
                        frames,
                        spec=spec,
                        signal_columns=("factor",),
                        label_columns=spec.label_columns,
                        n_groups=2,
                        asset_column="asset_id",
                        resource_policy=policy,
                        summary_state=state,
                    ),
                    summary_factory=lambda: (
                        state.diagnostics(),
                        state.ic_summary(),
                        state.group_spread_summary(),
                        state.coverage_summary(),
                    ),
                    spec=spec,
                    output_spec=output_spec,
                    layout=StorageLayout(project_root=tmp_path),
                    n_groups=2,
                )
            finally:
                state.close()
        return output_bundle, policy

    output_bundle, policy = persist((frame,), "eval_managed_two_labels_full")
    split_bundle, split_policy = persist(
        tuple(frame.iloc[index : index + 1].copy() for index in range(len(frame))),
        "eval_managed_two_labels_split",
    )

    persisted_metrics = pd.read_parquet(output_bundle / "metrics.parquet")
    assert persisted_metrics["label"].tolist() == [
        "label_a",
        "label_b",
        "label_a",
        "label_b",
        "label_a",
        "label_b",
    ]
    manifest = json.loads((output_bundle / "manifest.json").read_text())
    diagnostics = manifest["metadata"]["evaluation_diagnostics"]
    assert diagnostics["filter_mode"] == "eligible"
    assert diagnostics["rows_input"] == 12
    assert diagnostics["rows_after_signal_label_dropna"] == 10
    assert diagnostics["skipped_timestamp_count"] == 1
    split_manifest = json.loads((split_bundle / "manifest.json").read_text())
    assert split_manifest["metadata"]["evaluation_diagnostics"] == diagnostics
    for name in (
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
    ):
        full_frame = pd.read_parquet(output_bundle / name)
        split_frame = pd.read_parquet(split_bundle / name)
        pd.testing.assert_frame_equal(
            full_frame,
            split_frame,
            check_exact=False,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    assert policy.used_memory_bytes == 0
    assert split_policy.used_memory_bytes == 0

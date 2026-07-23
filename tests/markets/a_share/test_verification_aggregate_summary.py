from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage._dataset_reading as dataset_reading
from quant_research_platform.artifacts._evaluation_output_validation import (
    EVALUATION_ARTIFACTS,
)
from quant_research_platform.artifacts._output_manifest import (
    evaluation_output_metadata,
    output_manifest_payload,
    validate_output_manifest_business_envelope,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    panels,
)
from quant_research_platform.markets.implementations.a_share.verification import (
    aggregates,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy

RESOURCE_POLICY = RuntimeResourcePolicy.from_environment(
    available_memory_bytes=1024**3,
    environ={},
)

_LIFECYCLE_PRIMARY_TYPES = (
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
    asyncio.CancelledError,
)


def _summary(frame: pd.DataFrame, scratch_parent) -> aggregates.AggregateSummary:
    return aggregates._summarize_frames(
        (frame,),
        scratch_parent=scratch_parent,
        resource_policy=RESOURCE_POLICY,
    )


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_b", "asset_c"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "forward_return_10d": [0.01, 0.02, None],
        }
    )


class CountingDatasetReader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.calls = 0
        self._frame = frame

    def __call__(self, *_args: object, **_kwargs: object) -> Iterator[pd.DataFrame]:
        self.calls += 1
        return (self._frame.copy() for _ in range(1))


def _write_valid_output_bundle(root) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    filter_mode = "unfiltered"
    spec = EvaluationSpec(
        run_id="synthetic",
        market_id=MarketId("a_share"),
        research_dataset_id="research_dataset",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_kind="factor",
                source_column="value",
                direction="higher_is_better",
                transform="identity",
            ),
        ),
        label_columns=("label",),
        filter_mode=filter_mode,
        n_groups=5,
    )
    diagnostics = EvaluationDiagnostics(
        filter_mode=filter_mode,
        filter_columns_used=required_filter_columns(filter_mode),
        filter_definition=filter_definition(filter_mode),
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
        n_groups=5,
        skipped_timestamp_count=0,
    )
    payload = output_manifest_payload(
        output_spec=OutputSpec(
            output_id=root.name,
            output_definition_id="evaluation_outputs",
            run_id=spec.run_id,
            market_id=spec.market_id,
            run_type=spec.run_type,
            scope=OutputScope.LOCAL_ONLY,
        ),
        metadata=evaluation_output_metadata(
            spec=spec,
            evaluation_diagnostics=diagnostics,
        ),
        files=EVALUATION_ARTIFACTS,
    )
    for name in EVALUATION_ARTIFACTS:
        path = root / name
        if path.suffix == ".parquet":
            pq.write_table(pa.table({"value": [1.0]}), path)
        else:
            path.write_text("summary", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_aggregates_summary_is_row_order_and_file_split_insensitive(tmp_path) -> None:
    frame = _base_frame()
    split_summary = aggregates._summarize_frames(
        (
            frame.iloc[[2]].copy(),
            frame.iloc[[0, 1]].copy(),
        ),
        scratch_parent=tmp_path,
        resource_policy=RESOURCE_POLICY,
    )
    shuffled_summary = _summary(frame.sample(frac=1.0, random_state=9), tmp_path)

    assert split_summary.aggregate_checksum == shuffled_summary.aggregate_checksum
    assert split_summary.key_value_paired_summary_hash == (
        shuffled_summary.key_value_paired_summary_hash
    )


def test_aggregates_output_frames_are_lazy(
    monkeypatch,
    tmp_path,
) -> None:
    report_root = tmp_path / "report"
    _write_valid_output_bundle(report_root)
    read_order: list[str] = []

    observed_targets: list[int] = []

    observed_identities: list[object] = []

    def fake_iter_declared_parquet_member_batches(  # noqa: ANN001
        path,
        *,
        target_chunk_bytes,
        operation_headroom_bytes,
        expected_identity,
        _previous_output_bytes,
        batch_observer=None,
    ):
        observed_targets.append(target_chunk_bytes)
        assert operation_headroom_bytes == (
            policy.task_memory_budget_bytes - aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
        )
        observed_identities.append(expected_identity)
        assert _previous_output_bytes == [0]
        read_order.append(path.name)
        frame = _base_frame().iloc[:1].copy()
        if batch_observer is not None:
            batch_observer(pa.RecordBatch.from_pandas(frame), frame)
        yield frame

    def reject_full_file_read(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("output bundle bypassed bounded physical reader")

    monkeypatch.setattr(
        aggregates,
        "iter_declared_parquet_member_batches",
        fake_iter_declared_parquet_member_batches,
        raising=False,
    )
    monkeypatch.setattr(aggregates.pd, "read_parquet", reject_full_file_read)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=64 * 1024,
        environ={},
    )
    with aggregates.runtime_resource_policy_scope(policy):
        frames = aggregates._iter_output_frames(
            (report_root,),
            target_chunk_bytes=policy.payload_target_bytes,
            operation_headroom_bytes=(
                policy.task_memory_budget_bytes - aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
            ),
        )

        assert read_order == []
        next(frames)
        assert read_order == ["metrics.parquet"]
        next(frames)
        assert read_order == ["metrics.parquet", "grouped_label_values.parquet"]
    assert observed_targets == [policy.payload_target_bytes] * 2
    assert all(identity is not None for identity in observed_identities)


@pytest.mark.parametrize(
    ("headroom_delta", "should_fail"),
    ((0, False), (-1, True)),
)
def test_output_frames_propagate_exact_converter_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    headroom_delta: int,
    should_fail: bool,
) -> None:
    report_root = tmp_path / "report"
    _write_valid_output_bundle(report_root)
    batch = pa.record_batch(
        [pa.array(["x" * 4_096])],
        names=["value"],
    )
    expected = batch.to_pandas(use_threads=False)
    target_chunk_bytes = aggregates.pandas_frame_payload_bytes(expected)
    exact_headroom = (
        batch.get_total_buffer_size() + aggregates.pandas_frame_payload_bytes(expected)
    )
    first_member = next(
        name for name in EVALUATION_ARTIFACTS if name.endswith(".parquet")
    )

    def parts(path, **_kwargs):  # noqa: ANN001, ANN003
        return iter((batch,)) if Path(path).name == first_member else iter(())

    monkeypatch.setattr(dataset_reading, "iter_dataset_part_batches", parts)
    frames = aggregates._iter_output_frames(
        (report_root,),
        target_chunk_bytes=target_chunk_bytes,
        operation_headroom_bytes=exact_headroom + headroom_delta,
    )

    if should_fail:
        with pytest.raises(
            ValueError,
            match="dataset_working_set_exceeds_task_memory_budget",
        ):
            tuple(frames)
        return

    output = tuple(frames)
    assert len(output) == 1
    pd.testing.assert_frame_equal(output[0], expected)


def test_output_bundle_summary_deducts_existing_reducer_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "report"
    frame = _base_frame()
    expected = _summary(frame, tmp_path)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
    )
    observed_read_headroom: list[int] = []
    observed_sqlite_cache: list[int] = []
    original_index = aggregates.ExactPrimaryKeyIndex

    def frames(  # noqa: ANN001
        _roots,
        *,
        target_chunk_bytes,
        operation_headroom_bytes,
        _retained_payload_bytes,
    ):
        assert target_chunk_bytes == policy.payload_target_bytes
        observed_read_headroom.append(operation_headroom_bytes)
        _retained_payload_bytes[0] = aggregates.pandas_frame_payload_bytes(frame)
        yield frame

    class RecordingIndex(original_index):
        def __init__(self, **kwargs):  # noqa: ANN003
            observed_sqlite_cache.append(kwargs["sqlite_cache_bytes"])
            super().__init__(**kwargs)

    monkeypatch.setattr(aggregates, "_iter_output_frames", frames)
    monkeypatch.setattr(aggregates, "ExactPrimaryKeyIndex", RecordingIndex)

    observed = aggregates._summarize_output_bundles(
        (root,),
        resource_policy=policy,
    )

    assert observed == expected
    assert observed_read_headroom == [
        policy.task_memory_budget_bytes - aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
    ]
    assert observed_sqlite_cache == [aggregates.EXACT_KEY_SQLITE_PAGE_BYTES]


def test_summary_releases_scale_auxiliaries_before_requesting_next_frame(
    tmp_path: Path,
) -> None:
    frame = _base_frame()
    retained_auxiliaries: list[set[str]] = []

    def frames():  # noqa: ANN202
        yield frame
        caller = inspect.currentframe().f_back
        assert caller is not None
        retained_auxiliaries.append(
            {name for name in ("key_frame", "dates") if name in caller.f_locals}
        )
        yield frame.iloc[:0].copy()

    summary = aggregates._summarize_frames(
        frames(),
        scratch_parent=tmp_path,
        resource_policy=RESOURCE_POLICY,
    )

    assert summary.row_count == len(frame)
    assert retained_auxiliaries == [set()]


def test_output_summary_bounds_all_reducer_blocks_with_live_parent_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row_count = 100_000
    batch = pa.record_batch(
        [
            pa.array(pd.date_range("2020-01-01", periods=row_count, freq="min")),
            pa.array(range(row_count), type=pa.int64()),
            pa.array(range(row_count), type=pa.int64()),
        ],
        names=["timestamp", "value", "other"],
    )
    frame = batch.to_pandas(use_threads=False)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * 1024**2,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "6",
            "QRP_PAYLOAD_TARGET_MB": "6",
        },
    )
    physical_and_frame_bytes = (
        batch.get_total_buffer_size()
        + aggregates.pandas_frame_payload_bytes(frame)
        + aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
    )
    assert physical_and_frame_bytes < policy.task_memory_budget_bytes
    assert (
        physical_and_frame_bytes + 3 * row_count * 8 > policy.task_memory_budget_bytes
    )
    observed_rows: dict[str, list[int]] = {
        "content": [],
        "paired": [],
        "key_index": [],
        "date": [],
        "numeric": [],
    }
    original_series_blocks = aggregates._series_hash_blocks
    original_key_blocks = aggregates._key_hash_blocks
    original_numeric = aggregates._finite_numeric_min_max
    original_index = aggregates.ExactPrimaryKeyIndex
    original_to_datetime = aggregates.pd.to_datetime

    def series_blocks(series, **kwargs):  # noqa: ANN001, ANN003
        observed_rows["content"].append(len(series))
        yield from original_series_blocks(series, **kwargs)

    def key_blocks(current, key_columns, **kwargs):  # noqa: ANN001, ANN003
        observed_rows["paired"].append(len(current))
        yield from original_key_blocks(current, key_columns, **kwargs)

    def numeric_min_max(series, **kwargs):  # noqa: ANN001, ANN003
        observed_rows["numeric"].append(len(series))
        return original_numeric(series, **kwargs)

    def to_datetime(value, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if isinstance(value, pd.Series):
            observed_rows["date"].append(len(value))
        return original_to_datetime(value, *args, **kwargs)

    class RecordingIndex(original_index):
        def add_frame(self, current, *, namespace="dataset"):  # noqa: ANN001
            observed_rows["key_index"].append(len(current))
            return super().add_frame(current, namespace=namespace)

    monkeypatch.setattr(aggregates, "_series_hash_blocks", series_blocks)
    monkeypatch.setattr(aggregates, "_key_hash_blocks", key_blocks)
    monkeypatch.setattr(aggregates, "_finite_numeric_min_max", numeric_min_max)
    monkeypatch.setattr(aggregates, "ExactPrimaryKeyIndex", RecordingIndex)
    monkeypatch.setattr(aggregates.pd, "to_datetime", to_datetime)

    started = perf_counter()
    summary = aggregates._summarize_frames(
        (frame,),
        scratch_parent=tmp_path,
        resource_policy=policy,
        _retained_payload_bytes=[batch.get_total_buffer_size()],
    )
    elapsed = perf_counter() - started

    assert summary.row_count == row_count
    assert all(values and max(values) < row_count for values in observed_rows.values())
    assert max(observed_rows["key_index"]) > 1
    assert len(observed_rows["key_index"]) < 50
    assert elapsed < 15.0


def test_all_summary_callers_inherit_one_fixed_policy_and_retained_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _base_frame()
    expected = _summary(frame, tmp_path)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_PAYLOAD_TARGET_MB": "4"},
    )
    observed: list[tuple[RuntimeResourcePolicy, int]] = []

    def generated(
        _root,
        *,
        resource_policy,
        _retained_payload_bytes,
    ):  # noqa: ANN001
        assert resource_policy is policy
        _retained_payload_bytes[0] = 101
        yield frame

    def output(
        _roots,
        *,
        target_chunk_bytes,
        operation_headroom_bytes,
        _retained_payload_bytes,
    ):  # noqa: ANN001
        assert target_chunk_bytes == policy.payload_target_bytes
        assert operation_headroom_bytes < policy.task_memory_budget_bytes
        _retained_payload_bytes[0] = 202
        yield frame

    def dataset(
        _root,
        *,
        resource_policy,
        _retained_payload_bytes,
        **_kwargs,
    ):  # noqa: ANN001
        assert resource_policy is policy
        _retained_payload_bytes[0] = 303
        yield frame

    def summarize(
        frames,
        *,
        resource_policy,
        _retained_payload_bytes,
        **_kwargs,
    ):  # noqa: ANN001
        tuple(frames)
        observed.append((resource_policy, _retained_payload_bytes[0]))
        return expected

    monkeypatch.setattr(
        aggregates,
        "_iter_generated_full_market_panel_chunks",
        generated,
    )
    monkeypatch.setattr(aggregates, "_iter_output_frames", output)
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks_with_policy",
        dataset,
    )
    monkeypatch.setattr(aggregates, "_summarize_frames", summarize)

    assert (
        aggregates._summarize_generated_full_market_panel(
            tmp_path,
            resource_policy=policy,
        )
        == expected
    )
    assert (
        aggregates._summarize_output_bundles(
            (tmp_path / "output",),
            resource_policy=policy,
        )
        == expected
    )
    assert (
        aggregates._summarize_dataset_root(
            tmp_path / "dataset",
            resource_policy=policy,
        )
        == expected
    )
    assert observed == [(policy, 101), (policy, 202), (policy, 303)]


def test_summary_low_reducer_headroom_fails_and_cleans_key_index(
    tmp_path: Path,
) -> None:
    frame = _base_frame().iloc[:1].copy()
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=8 * 1024**2,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "4"},
    )
    retained_payload_bytes = [
        policy.task_memory_budget_bytes
        - aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
        - aggregates.pandas_frame_payload_bytes(frame)
        - 1
    ]

    with pytest.raises(
        ValueError,
        match="aggregate_working_set_exceeds_task_memory_budget",
    ):
        aggregates._summarize_frames(
            (frame,),
            scratch_parent=tmp_path,
            resource_policy=policy,
            _retained_payload_bytes=retained_payload_bytes,
        )

    assert not list(tmp_path.glob(".k-*"))


def test_tight_summary_blocks_preserve_parity_order_duplicate_and_minmax(
    tmp_path: Path,
) -> None:
    frame = pd.concat([_base_frame()] * 10_000, ignore_index=True)
    expected = _summary(frame, tmp_path)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "8"},
    )
    reducer_headroom_bytes = 256 * 1024
    retained_payload_bytes = [
        policy.task_memory_budget_bytes
        - aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
        - aggregates.pandas_frame_payload_bytes(frame)
        - reducer_headroom_bytes
    ]

    observed = aggregates._summarize_frames(
        (frame,),
        scratch_parent=tmp_path,
        resource_policy=policy,
        _retained_payload_bytes=retained_payload_bytes,
    )

    assert observed == expected


def test_variable_width_candidate_retries_before_summary_or_index_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row_count = 1_000
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * row_count,
            "asset_id": ["narrow"]
            + [f"asset-{index:04d}-" + "x" * 4_096 for index in range(1, row_count)],
            "timestamp": pd.date_range("2024-01-01", periods=row_count, freq="min"),
            "value": list(range(row_count)),
        }
    )
    expected = _summary(frame, tmp_path)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024**2,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "8"},
    )
    reducer_headroom_bytes = 2_165_395
    retained_payload_bytes = [
        policy.task_memory_budget_bytes
        - aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
        - aggregates.pandas_frame_payload_bytes(frame)
        - reducer_headroom_bytes
    ]
    assert retained_payload_bytes[0] > 0
    index_rows: list[int] = []
    original_index = aggregates.ExactPrimaryKeyIndex
    original_paired_summary = aggregates._update_key_value_paired_summary
    latest_candidate: list[tuple[int, int]] = []

    def paired_summary(summary, current, *, key_columns):  # noqa: ANN001
        auxiliary_bytes = original_paired_summary(
            summary,
            current,
            key_columns=key_columns,
        )
        latest_candidate[:] = [
            (
                aggregates.pandas_frame_payload_bytes(current),
                auxiliary_bytes,
            )
        ]
        return auxiliary_bytes

    class RecordingIndex(original_index):
        def add_frame(self, current, *, namespace="dataset"):  # noqa: ANN001
            key_bytes = aggregates.pandas_frame_payload_bytes(current)
            assert key_bytes <= reducer_headroom_bytes
            assert latest_candidate
            candidate_bytes, auxiliary_bytes = latest_candidate[0]
            assert candidate_bytes + auxiliary_bytes <= reducer_headroom_bytes
            index_rows.append(len(current))
            return super().add_frame(current, namespace=namespace)

    monkeypatch.setattr(
        aggregates,
        "_update_key_value_paired_summary",
        paired_summary,
    )
    monkeypatch.setattr(aggregates, "ExactPrimaryKeyIndex", RecordingIndex)

    observed = aggregates._summarize_frames(
        (frame,),
        scratch_parent=tmp_path,
        resource_policy=policy,
        _retained_payload_bytes=retained_payload_bytes,
    )

    assert observed == expected
    assert max(index_rows) > 1
    assert len(index_rows) < 50


def test_single_oversized_candidate_fails_before_summary_or_index_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["x" * (512 * 1024)],
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "value": [1.0],
        }
    )
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=8 * 1024**2,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "4"},
    )
    reducer_headroom_bytes = 256 * 1024
    retained_payload_bytes = [
        policy.task_memory_budget_bytes
        - aggregates.EXACT_KEY_SQLITE_PAGE_BYTES
        - aggregates.pandas_frame_payload_bytes(frame)
        - reducer_headroom_bytes
    ]
    mutations: list[str] = []

    def content(*_args, **_kwargs):  # noqa: ANN002, ANN003
        mutations.append("content")
        return 0

    class RecordingIndex(aggregates.ExactPrimaryKeyIndex):
        def add_frame(self, current, *, namespace="dataset"):  # noqa: ANN001
            mutations.append("index")
            return super().add_frame(current, namespace=namespace)

    monkeypatch.setattr(aggregates, "_update_content_summary", content)
    monkeypatch.setattr(aggregates, "ExactPrimaryKeyIndex", RecordingIndex)

    with pytest.raises(
        ValueError,
        match="aggregate_working_set_exceeds_task_memory_budget",
    ):
        aggregates._summarize_frames(
            (frame,),
            scratch_parent=tmp_path,
            resource_policy=policy,
            _retained_payload_bytes=retained_payload_bytes,
        )

    assert mutations == []


def test_dataset_summary_retained_cell_records_arrow_parent_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _base_frame()
    batch = pa.RecordBatch.from_pandas(frame, preserve_index=False)
    retained_payload_bytes = [0]
    closed = False

    class Session:
        resource_policy = RESOURCE_POLICY

        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: object,
        ) -> None:
            if exc_type is None:
                self.close()
                return
            try:
                self.close()
            except BaseException:
                pass

        def close(self) -> None:
            nonlocal closed
            closed = True

    def chunks(_session, *, batch_observer, **_kwargs):  # noqa: ANN001, ANN003
        batch_observer(batch, frame)
        yield frame

    monkeypatch.setattr(
        aggregates.materialization_io,
        "open_dataset_session",
        lambda *_args, **_kwargs: Session(),
    )
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks",
        chunks,
    )

    observed = tuple(
        aggregates.materialization_io.iter_dataset_chunks_with_policy(
            Path("dataset"),
            resource_policy=RESOURCE_POLICY,
            _retained_payload_bytes=retained_payload_bytes,
        )
    )

    assert len(observed) == 1
    assert observed[0] is frame
    assert retained_payload_bytes == [batch.get_total_buffer_size()]
    assert closed


def test_generated_summary_next_window_read_deducts_previous_window_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _base_frame().assign(value=[1.0, 2.0, 3.0])
    sessions = tuple(pd.to_datetime(["2014-01-02", "2014-01-03", "2014-01-06"]))
    reads = [frame.copy() for _ in range(6)]
    reads[4] = frame.iloc[:0].copy()
    retained_at_read: list[int] = []
    bounded_frames: list[tuple[pd.DataFrame, ...]] = []
    compute_count = 0

    def read(*_args, retained_payload_bytes=0, **_kwargs):  # noqa: ANN002, ANN003
        if len(retained_at_read) == 5:
            generator_locals = inspect.getgeneratorlocals(generated)
            assert "market_bars" not in generator_locals
            assert generator_locals["previous_yielded_frame"] is first
            assert generator_locals["previous_window_payload_bytes"] == source_bytes
        retained_at_read.append(retained_payload_bytes)
        return reads[len(retained_at_read) - 1]

    class Planner:
        @classmethod
        def from_sessions(cls, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return cls()

        def windows(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
            return tuple(
                SimpleNamespace(read=None, core_sessions=(session,))
                for session in sessions
            )

    def compute(current: pd.DataFrame, **_kwargs) -> pd.DataFrame:  # noqa: ANN003
        nonlocal compute_count
        compute_count += 1
        if compute_count == 2:
            generator_locals = inspect.getgeneratorlocals(generated)
            assert "panel" not in generator_locals
            assert "output" not in generator_locals
            assert "selected" not in generator_locals
            assert generator_locals["market_bars"] is current
            assert generator_locals["previous_yielded_frame"] is first
        return current.copy()

    monkeypatch.setattr(
        aggregates.materialization_io,
        "read_materialized_dataset",
        read,
    )
    monkeypatch.setattr(
        aggregates,
        "_trading_sessions_from_calendar",
        lambda _frame: sessions,
    )
    monkeypatch.setattr(
        aggregates,
        "_estimate_generated_full_panel_memory_bytes",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(aggregates, "SessionWindowPlanner", Planner)
    monkeypatch.setattr(
        aggregates,
        "inspect_dataset",
        lambda _root: SimpleNamespace(columns=tuple(frame.columns)),
    )
    monkeypatch.setattr(
        processed_catalog,
        "processed_catalog",
        lambda: SimpleNamespace(
            get=lambda _data_id: SimpleNamespace(required_columns=tuple(frame.columns))
        ),
    )
    monkeypatch.setattr(
        panels,
        "compute_market_panel",
        compute,
    )
    monkeypatch.setattr(
        aggregates.materialization_planning,
        "filter_output_sessions",
        lambda output, _sessions: output.copy(),
    )
    monkeypatch.setattr(
        aggregates.materialization_planning,
        "require_known_materialization_payload_bound",
        lambda *, frames, **_kwargs: bounded_frames.append(tuple(frames)),
    )
    retained_payload_bytes = [0]
    generated = aggregates._iter_generated_full_market_panel_session_chunks(
        tmp_path,
        resource_policy=RESOURCE_POLICY,
        _retained_payload_bytes=retained_payload_bytes,
    )

    first = next(generated)
    source_bytes = aggregates.pandas_frame_payload_bytes(frame)
    assert retained_payload_bytes == [6 * source_bytes]
    second = next(generated)

    assert retained_at_read[3] == 3 * source_bytes
    assert retained_at_read[4] == 4 * source_bytes
    assert retained_at_read[5] == 4 * source_bytes
    assert any(candidate is first for candidate in bounded_frames[-1])
    assert all(
        candidate is not first
        for candidate in inspect.getgeneratorlocals(generated).values()
    )
    assert first.equals(second)
    generated.close()


@pytest.mark.parametrize("caller", ("generated", "output", "dataset"))
@pytest.mark.parametrize("failure", (None, ValueError, KeyboardInterrupt))
def test_summary_callers_close_upstream_on_success_error_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caller: str,
    failure: type[BaseException] | None,
) -> None:
    closed = False

    def frames():  # noqa: ANN202
        nonlocal closed
        try:
            yield _base_frame()
        finally:
            closed = True

    retained = frames()
    if caller == "generated":
        monkeypatch.setattr(
            aggregates,
            "_iter_generated_full_market_panel_chunks",
            lambda *_args, **_kwargs: retained,
        )
        invoke = lambda: aggregates._summarize_generated_full_market_panel(  # noqa: E731
            tmp_path,
            resource_policy=RESOURCE_POLICY,
        )
    elif caller == "output":
        monkeypatch.setattr(
            aggregates,
            "_iter_output_frames",
            lambda *_args, **_kwargs: retained,
        )
        invoke = lambda: aggregates._summarize_output_bundles(  # noqa: E731
            (tmp_path / "output",),
            resource_policy=RESOURCE_POLICY,
        )
    else:
        monkeypatch.setattr(
            aggregates.materialization_io,
            "iter_dataset_chunks_with_policy",
            lambda *_args, **_kwargs: retained,
        )
        invoke = lambda: aggregates._summarize_dataset_root(  # noqa: E731
            tmp_path / "dataset",
            resource_policy=RESOURCE_POLICY,
        )

    if failure is None:
        invoke()
    else:

        def fail(*_args, **_kwargs):  # noqa: ANN002, ANN003
            raise failure("consumer_stopped")

        monkeypatch.setattr(aggregates, "_update_content_summary", fail)

        with pytest.raises(failure, match="consumer_stopped"):
            invoke()

    assert closed


@pytest.mark.parametrize("primary_type", _LIFECYCLE_PRIMARY_TYPES)
def test_output_frames_preserve_traversal_primary_over_frozen_verify(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("traversal_primary")
    verify_calls = 0
    frozen = SimpleNamespace(parquet_members=((Path("member.parquet"), object()),))

    def fail_traversal(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise primary
        yield

    def fail_verify(_frozen: object) -> None:
        nonlocal verify_calls
        verify_calls += 1
        raise OSError("verify_secondary")

    monkeypatch.setattr(aggregates, "_freeze_output_bundle", lambda _root: frozen)
    monkeypatch.setattr(
        aggregates,
        "iter_declared_parquet_member_batches",
        fail_traversal,
    )
    monkeypatch.setattr(aggregates, "_verify_frozen_output_bundle", fail_verify)

    with pytest.raises(primary_type) as raised:
        list(
            aggregates._iter_output_frames(
                (Path("output"),),
                target_chunk_bytes=1024,
                operation_headroom_bytes=2048,
            )
        )

    assert raised.value is primary
    assert verify_calls == 1


def test_output_frames_propagate_frozen_verify_failure_without_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_failure = OSError("verify_only")
    verify_calls = 0
    frozen = SimpleNamespace(parquet_members=((Path("member.parquet"), object()),))

    def fail_verify(_frozen: object) -> None:
        nonlocal verify_calls
        verify_calls += 1
        raise verify_failure

    monkeypatch.setattr(aggregates, "_freeze_output_bundle", lambda _root: frozen)
    monkeypatch.setattr(
        aggregates,
        "iter_declared_parquet_member_batches",
        lambda *_args, **_kwargs: iter(()),
    )
    monkeypatch.setattr(aggregates, "_verify_frozen_output_bundle", fail_verify)

    with pytest.raises(OSError) as raised:
        list(
            aggregates._iter_output_frames(
                (Path("output"),),
                target_chunk_bytes=1024,
                operation_headroom_bytes=2048,
            )
        )

    assert raised.value is verify_failure
    assert verify_calls == 1


def test_output_frames_consumer_close_preserves_abandonment_over_frozen_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_calls = 0
    frozen = SimpleNamespace(parquet_members=((Path("member.parquet"), object()),))

    def batches(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        yield _base_frame()
        pytest.fail("consumer close did not abandon traversal")

    def fail_verify(_frozen: object) -> None:
        nonlocal verify_calls
        verify_calls += 1
        raise OSError("verify_secondary")

    monkeypatch.setattr(aggregates, "_freeze_output_bundle", lambda _root: frozen)
    monkeypatch.setattr(
        aggregates,
        "iter_declared_parquet_member_batches",
        batches,
    )
    monkeypatch.setattr(aggregates, "_verify_frozen_output_bundle", fail_verify)

    frames = aggregates._iter_output_frames(
        (Path("output"),),
        target_chunk_bytes=1024,
        operation_headroom_bytes=2048,
    )
    next(frames)
    frames.close()

    assert verify_calls == 1


@pytest.mark.parametrize("primary_type", _LIFECYCLE_PRIMARY_TYPES)
def test_summarize_frames_preserves_body_primary_over_input_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("summary_primary")
    close_calls = 0

    class Frames:
        def __init__(self) -> None:
            self._frames = iter((_base_frame(),))

        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._frames)

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            raise OSError("close_secondary")

    def fail_summary(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise primary

    monkeypatch.setattr(aggregates, "_update_content_summary", fail_summary)

    with pytest.raises(primary_type) as raised:
        aggregates._summarize_frames(
            Frames(),
            scratch_parent=tmp_path,
            resource_policy=RESOURCE_POLICY,
        )

    assert raised.value is primary
    assert close_calls == 1


def test_summarize_frames_propagates_input_close_failure_without_primary(
    tmp_path: Path,
) -> None:
    close_failure = OSError("close_only")
    close_calls = 0

    class Frames:
        def __init__(self) -> None:
            self._frames = iter((_base_frame(),))

        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._frames)

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            raise close_failure

    with pytest.raises(OSError) as raised:
        aggregates._summarize_frames(
            Frames(),
            scratch_parent=tmp_path,
            resource_policy=RESOURCE_POLICY,
        )

    assert raised.value is close_failure
    assert close_calls == 1


def test_aggregates_validate_all_members_before_selecting_parquet(tmp_path) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    frozen = aggregates._freeze_output_bundle(root)

    assert frozen.parquet_members == tuple(
        (
            root / name,
            aggregates.dataset_file_identity(root / name),
        )
        for name in EVALUATION_ARTIFACTS
        if name.endswith(".parquet")
    )


def test_aggregates_output_frames_reject_manifest_stray_parquet(tmp_path) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    table = pa.table({"value": [1.0]})
    pq.write_table(table, root / "stray.parquet")

    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


def test_aggregates_output_frames_reject_external_hardlink(tmp_path) -> None:
    external = tmp_path / "external.parquet"
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    pq.write_table(pa.table({"value": [1.0]}), external)
    (root / "metrics.parquet").unlink()
    os.link(external, root / "metrics.parquet")

    with pytest.raises(
        ValueError,
        match="output_member_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


@pytest.mark.parametrize(
    ("member_state", "expected_error"),
    (
        ("missing", "output_member_set_mismatch"),
        ("directory", "output_member_physical_identity_invalid"),
    ),
)
def test_aggregates_output_frames_validate_declared_non_parquet_members(
    tmp_path,
    member_state: str,
    expected_error: str,
) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    (root / "summary.md").unlink()
    if member_state == "directory":
        (root / "summary.md").mkdir()

    with pytest.raises(
        ValueError,
        match=expected_error,
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


def test_aggregates_output_frames_reject_declared_member_physical_duplicate(
    tmp_path,
) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    (root / "grouped_label_values.parquet").unlink()
    os.link(root / "metrics.parquet", root / "grouped_label_values.parquet")

    with pytest.raises(
        ValueError,
        match="output_member_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


def test_aggregates_output_frames_reject_normalized_manifest_alias(tmp_path) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    payload["files"] = ["metrics.parquet", "./metrics.parquet"]
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="output_file_path_invalid"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("unknown_field", "output_manifest_fields_invalid"),
        ("wrong_kind_output_id", "output_manifest_output_id_invalid"),
        ("wrong_kind_run_id", "output_manifest_run_id_invalid"),
        ("equal_ids", "output_manifest_output_id_invalid"),
        ("unknown_definition", "output_manifest_definition_id_invalid"),
        ("deep_nested_schema_id", "output_manifest_metadata_invalid"),
        ("backslash_file", "output_file_path_invalid"),
        ("casefold_duplicate_file", "output_files_invalid"),
    ),
)
def test_aggregates_explicit_output_rejects_noncurrent_manifest_envelope(
    tmp_path,
    mutation: str,
    expected_error: str,
) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    if mutation == "unknown_field":
        payload["unknown"] = True
    elif mutation == "wrong_kind_output_id":
        payload["output_id"] = 7
    elif mutation == "wrong_kind_run_id":
        payload["run_id"] = 7
    elif mutation == "equal_ids":
        payload["output_id"] = "evaluation_outputs"
    elif mutation == "unknown_definition":
        payload["output_definition_id"] = "unknown_outputs"
    elif mutation == "deep_nested_schema_id":
        payload["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    elif mutation == "backslash_file":
        payload["files"] = ["nested\\metrics.parquet"]
    elif mutation == "casefold_duplicate_file":
        payload["files"] = ["metrics.parquet", "METRICS.PARQUET"]

    with pytest.raises(ValueError, match=expected_error):
        validate_output_manifest_business_envelope(payload)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


@pytest.mark.parametrize("extra_kind", ("file", "nested"))
def test_aggregates_output_frames_reject_undeclared_non_parquet_members(
    tmp_path,
    extra_kind: str,
) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    if extra_kind == "file":
        (root / "extra.md").write_text("summary", encoding="utf-8")
    else:
        nested = root / "nested"
        nested.mkdir()
        (nested / "summary.md").write_text("summary", encoding="utf-8")
    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


def test_aggregates_output_frames_reject_manifest_symlink(tmp_path) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    (root / "manifest.json").unlink()
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    try:
        (root / "manifest.json").symlink_to(external_manifest)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(
        ValueError,
        match="output_manifest_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


def test_aggregates_output_frames_reject_manifest_external_hardlink(tmp_path) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    (root / "manifest.json").unlink()
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    try:
        os.link(external_manifest, root / "manifest.json")
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    with pytest.raises(
        ValueError,
        match="output_manifest_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
                operation_headroom_bytes=2 * 1024**2,
            )
        )


def test_aggregates_summary_detects_single_value_change(tmp_path) -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    candidate.loc[0, "forward_return_10d"] = 0.03

    assert _summary(baseline, tmp_path).aggregate_checksum != (
        _summary(candidate, tmp_path).aggregate_checksum
    )


def test_aggregates_summary_detects_value_swap_between_keys(tmp_path) -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    candidate.loc[[0, 1], "forward_return_10d"] = candidate.loc[
        [1, 0],
        "forward_return_10d",
    ].to_numpy()

    left = _summary(baseline, tmp_path)
    right = _summary(candidate, tmp_path)

    assert left.content_summary == right.content_summary
    assert left.key_value_paired_summary_hash != right.key_value_paired_summary_hash
    assert left.aggregate_checksum != right.aggregate_checksum


def test_aggregates_summary_canonicalizes_numeric_signed_zero(tmp_path) -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    baseline.loc[0, "forward_return_10d"] = 0.0
    candidate.loc[0, "forward_return_10d"] = -0.0

    assert _summary(baseline, tmp_path).key_value_paired_summary_hash == (
        _summary(candidate, tmp_path).key_value_paired_summary_hash
    )


def test_aggregates_summary_detects_contract_shape_changes(tmp_path) -> None:
    baseline = _base_frame()
    extra_column = baseline.assign(extra_factor=[1.0, 2.0, 3.0])
    dtype_changed = baseline.assign(
        forward_return_10d=baseline["forward_return_10d"].astype("string")
    )
    null_changed = baseline.copy()
    null_changed.loc[1, "forward_return_10d"] = None
    duplicate_key = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)

    baseline_summary = _summary(baseline, tmp_path)

    assert (
        baseline_summary.column_set_hash
        != _summary(
            extra_column,
            tmp_path,
        ).column_set_hash
    )
    assert baseline_summary.dtype_summary_hash != (
        _summary(dtype_changed, tmp_path).dtype_summary_hash
    )
    assert baseline_summary.null_summary_hash != (
        _summary(null_changed, tmp_path).null_summary_hash
    )
    assert _summary(duplicate_key, tmp_path).duplicate_key_count > 0


def test_aggregates_summary_is_dynamic_hash_block_insensitive(
    monkeypatch,
    tmp_path,
) -> None:
    frame = pd.concat([_base_frame()] * 2000, ignore_index=True)
    baseline = _summary(frame, tmp_path)
    monkeypatch.setattr(aggregates, "HASH_BLOCK_ROWS", 7)

    assert _summary(frame, tmp_path).aggregate_checksum == baseline.aggregate_checksum


def test_aggregates_summary_ignores_dataframe_index_labels(tmp_path) -> None:
    frame = _base_frame()
    duplicate_index = frame.copy()
    duplicate_index.index = [0, 0, 0]

    assert _summary(duplicate_index, tmp_path).aggregate_checksum == (
        _summary(frame.reset_index(drop=True), tmp_path).aggregate_checksum
    )


def test_aggregate_hash_and_minmax_blocks_are_resource_contract_independent() -> None:
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            aggregates._series_hash_blocks,
            aggregates._key_hash_blocks,
            aggregates._finite_numeric_min_max,
        )
    )

    assert "payload_target" not in source
    assert "active_runtime_resource_policy" not in source
    assert "memory_usage" not in source


def test_aggregates_numeric_min_max_ignores_all_null_datetimes() -> None:
    series = pd.Series(pd.to_datetime([None, None]))

    assert aggregates._finite_numeric_min_max(series) is None


def test_aggregates_dataset_summary_cache_reuses_same_root(
    monkeypatch,
    tmp_path,
) -> None:
    reader = CountingDatasetReader(_base_frame())
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks_with_policy",
        reader,
    )
    cache = {}
    root = tmp_path / "dataset"

    first = aggregates._summarize_dataset_root(
        root,
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    second = aggregates._summarize_dataset_root(
        root,
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )

    assert first.aggregate_checksum == second.aggregate_checksum
    assert reader.calls == 1


def test_aggregates_dataset_summary_cache_keeps_year_filters_distinct(
    monkeypatch,
    tmp_path,
) -> None:
    reader = CountingDatasetReader(_base_frame())
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks_with_policy",
        reader,
    )
    cache = {}
    root = tmp_path / "dataset"

    aggregates._summarize_dataset_root(
        root,
        years=(2014,),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    aggregates._summarize_dataset_root(
        root,
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    aggregates._summarize_dataset_root(
        root,
        years=(2014,),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )

    assert reader.calls == 2


def test_aggregates_dataset_summary_cache_normalizes_year_order(
    monkeypatch,
    tmp_path,
) -> None:
    reader = CountingDatasetReader(_base_frame())
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks_with_policy",
        reader,
    )
    cache = {}
    root = tmp_path / "dataset"

    aggregates._summarize_dataset_root(
        root,
        years=(2015, 2014),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    aggregates._summarize_dataset_root(
        root,
        years=(2014, 2015),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )

    assert reader.calls == 1

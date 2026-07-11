from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.markets.implementations.a_share.verification import (
    aggregates,
)


def _summary(frame: pd.DataFrame, scratch_parent) -> aggregates.AggregateSummary:
    return aggregates._summarize_frames(
        (frame,),
        scratch_parent=scratch_parent,
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
        return iter((self._frame.copy(),))


def test_aggregates_summary_is_row_order_and_file_split_insensitive(tmp_path) -> None:
    frame = _base_frame()
    split_summary = aggregates._summarize_frames(
        (
            frame.iloc[[2]].copy(),
            frame.iloc[[0, 1]].copy(),
        ),
        scratch_parent=tmp_path,
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
    report_root.mkdir()
    (report_root / "first.parquet").write_bytes(b"")
    (report_root / "second.parquet").write_bytes(b"")
    (report_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "qrp-output-manifest",
                "output_id": "synthetic",
                "output_definition_id": "evaluation_outputs",
                "run_id": "synthetic",
                "metadata": {},
                "files": ["second.parquet", "first.parquet"],
            }
        ),
        encoding="utf-8",
    )
    read_order: list[str] = []

    observed_targets: list[int] = []

    def fake_iter_parquet_file_batches(  # noqa: ANN001
        path,
        *,
        target_chunk_bytes,
    ):
        observed_targets.append(target_chunk_bytes)
        read_order.append(path.name)
        yield _base_frame().iloc[:1].copy()

    def reject_full_file_read(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("output bundle bypassed bounded physical reader")

    monkeypatch.setattr(
        aggregates,
        "iter_parquet_file_batches",
        fake_iter_parquet_file_batches,
        raising=False,
    )
    monkeypatch.setattr(aggregates.pd, "read_parquet", reject_full_file_read)
    policy = aggregates.RuntimeResourcePolicy(
        available_memory_bytes=1024**3,
        chunk_target_bytes=64 * 1024,
        memory_safety_factor=1.0,
    )
    with aggregates.runtime_resource_policy_scope(policy):
        frames = aggregates._iter_output_frames(
            (report_root,),
            target_chunk_bytes=64 * 1024,
        )

        assert read_order == []
        next(frames)
        assert read_order == ["second.parquet"]
        next(frames)
        assert read_order == ["second.parquet", "first.parquet"]
    assert observed_targets == [policy.effective_chunk_bytes] * 2


def test_aggregates_output_frames_reject_manifest_stray_parquet(tmp_path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    table = pa.table({"value": [1.0]})
    pq.write_table(table, root / "listed.parquet")
    pq.write_table(table, root / "stray.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "qrp-output-manifest",
                "output_id": "synthetic",
                "output_definition_id": "evaluation_outputs",
                "run_id": "synthetic",
                "metadata": {},
                "files": ["listed.parquet"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid_aggregate_output_files"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


def test_aggregates_output_frames_reject_external_hardlink(tmp_path) -> None:
    external = tmp_path / "external.parquet"
    root = tmp_path / "output"
    root.mkdir()
    pq.write_table(pa.table({"value": [1.0]}), external)
    os.link(external, root / "listed.parquet")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "qrp-output-manifest",
                "output_id": "synthetic",
                "output_definition_id": "evaluation_outputs",
                "run_id": "synthetic",
                "metadata": {},
                "files": ["listed.parquet"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid_aggregate_output_files"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
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

    assert baseline_summary.column_set_hash != _summary(
        extra_column,
        tmp_path,
    ).column_set_hash
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
    monkeypatch.setattr(aggregates, "HASH_BLOCK_MAX_ROWS", 7)

    assert _summary(frame, tmp_path).aggregate_checksum == baseline.aggregate_checksum


def test_aggregates_summary_ignores_dataframe_index_labels(tmp_path) -> None:
    frame = _base_frame()
    duplicate_index = frame.copy()
    duplicate_index.index = [0, 0, 0]

    assert _summary(duplicate_index, tmp_path).aggregate_checksum == (
        _summary(frame.reset_index(drop=True), tmp_path).aggregate_checksum
    )


def test_aggregates_dynamic_hash_block_uses_runtime_policy(
    monkeypatch,
) -> None:
    class SmallPolicy:
        effective_chunk_bytes = 64 * 1024

    class LargePolicy:
        effective_chunk_bytes = 512 * 1024 * 1024

    monkeypatch.setattr(
        aggregates.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(lambda: SmallPolicy()),
    )
    small_rows = aggregates._hash_block_rows(value_width=1)
    monkeypatch.setattr(
        aggregates.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(lambda: LargePolicy()),
    )
    large_rows = aggregates._hash_block_rows(value_width=1)

    assert small_rows < large_rows
    assert aggregates.HASH_BLOCK_MIN_ROWS <= small_rows
    assert large_rows <= aggregates.HASH_BLOCK_MAX_ROWS


def test_aggregates_dynamic_hash_block_uses_conservative_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        aggregates.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no_memory_probe"))),
    )

    assert aggregates._hash_block_rows(value_width=3) >= (
        aggregates.HASH_BLOCK_MIN_ROWS
    )


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

    first = aggregates._summarize_dataset_root(root, summary_cache=cache)
    second = aggregates._summarize_dataset_root(root, summary_cache=cache)

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

    aggregates._summarize_dataset_root(root, years=(2014,), summary_cache=cache)
    aggregates._summarize_dataset_root(root, summary_cache=cache)
    aggregates._summarize_dataset_root(root, years=(2014,), summary_cache=cache)

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
    )
    aggregates._summarize_dataset_root(
        root,
        years=(2014, 2015),
        summary_cache=cache,
    )

    assert reader.calls == 1

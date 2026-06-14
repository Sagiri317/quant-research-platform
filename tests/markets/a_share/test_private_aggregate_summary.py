from __future__ import annotations

import pandas as pd

from quant_research_platform.markets.implementations.a_share.verification import (
    private_aggregate,
)


def _summary(frame: pd.DataFrame) -> private_aggregate.AggregateSummary:
    return private_aggregate._summarize_frames((frame,))


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_b", "asset_c"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "forward_return_10d": [0.01, 0.02, None],
        }
    )


def test_private_aggregate_summary_is_row_order_and_file_split_insensitive() -> None:
    frame = _base_frame()
    split_summary = private_aggregate._summarize_frames(
        (
            frame.iloc[[2]].copy(),
            frame.iloc[[0, 1]].copy(),
        )
    )
    shuffled_summary = _summary(frame.sample(frac=1.0, random_state=9))

    assert split_summary.aggregate_checksum == shuffled_summary.aggregate_checksum
    assert split_summary.key_value_paired_summary_hash == (
        shuffled_summary.key_value_paired_summary_hash
    )


def test_private_aggregate_summary_detects_single_value_change() -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    candidate.loc[0, "forward_return_10d"] = 0.03

    assert _summary(baseline).aggregate_checksum != (
        _summary(candidate).aggregate_checksum
    )


def test_private_aggregate_summary_detects_value_swap_between_keys() -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    candidate.loc[[0, 1], "forward_return_10d"] = candidate.loc[
        [1, 0],
        "forward_return_10d",
    ].to_numpy()

    left = _summary(baseline)
    right = _summary(candidate)

    assert left.content_summary == right.content_summary
    assert left.key_value_paired_summary_hash != right.key_value_paired_summary_hash
    assert left.aggregate_checksum != right.aggregate_checksum


def test_private_aggregate_summary_canonicalizes_numeric_signed_zero() -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    baseline.loc[0, "forward_return_10d"] = 0.0
    candidate.loc[0, "forward_return_10d"] = -0.0

    assert _summary(baseline).key_value_paired_summary_hash == (
        _summary(candidate).key_value_paired_summary_hash
    )


def test_private_aggregate_summary_detects_contract_shape_changes() -> None:
    baseline = _base_frame()
    extra_column = baseline.assign(extra_factor=[1.0, 2.0, 3.0])
    dtype_changed = baseline.assign(
        forward_return_10d=baseline["forward_return_10d"].astype("string")
    )
    null_changed = baseline.copy()
    null_changed.loc[1, "forward_return_10d"] = None
    duplicate_key = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)

    baseline_summary = _summary(baseline)

    assert baseline_summary.column_set_hash != _summary(extra_column).column_set_hash
    assert baseline_summary.dtype_summary_hash != (
        _summary(dtype_changed).dtype_summary_hash
    )
    assert baseline_summary.null_summary_hash != (
        _summary(null_changed).null_summary_hash
    )
    assert _summary(duplicate_key).duplicate_key_count > 0


def test_private_aggregate_summary_is_dynamic_hash_block_insensitive(
    monkeypatch,
) -> None:
    frame = pd.concat([_base_frame()] * 2000, ignore_index=True)
    baseline = _summary(frame)
    monkeypatch.setattr(private_aggregate, "HASH_BLOCK_MAX_ROWS", 7)

    assert _summary(frame).aggregate_checksum == baseline.aggregate_checksum


def test_private_aggregate_summary_ignores_dataframe_index_labels() -> None:
    frame = _base_frame()
    duplicate_index = frame.copy()
    duplicate_index.index = [0, 0, 0]

    assert _summary(duplicate_index).aggregate_checksum == (
        _summary(frame.reset_index(drop=True)).aggregate_checksum
    )


def test_private_aggregate_dynamic_hash_block_uses_runtime_policy(
    monkeypatch,
) -> None:
    class SmallPolicy:
        effective_chunk_bytes = 64 * 1024

    class LargePolicy:
        effective_chunk_bytes = 512 * 1024 * 1024

    monkeypatch.setattr(
        private_aggregate.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(lambda: SmallPolicy()),
    )
    small_rows = private_aggregate._hash_block_rows(value_width=1)
    monkeypatch.setattr(
        private_aggregate.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(lambda: LargePolicy()),
    )
    large_rows = private_aggregate._hash_block_rows(value_width=1)

    assert small_rows < large_rows
    assert private_aggregate.HASH_BLOCK_MIN_ROWS <= small_rows
    assert large_rows <= private_aggregate.HASH_BLOCK_MAX_ROWS


def test_private_aggregate_dynamic_hash_block_uses_conservative_fallback(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        private_aggregate.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no_memory_probe"))),
    )

    assert private_aggregate._hash_block_rows(value_width=3) >= (
        private_aggregate.HASH_BLOCK_MIN_ROWS
    )


def test_private_aggregate_numeric_min_max_ignores_all_null_datetimes() -> None:
    series = pd.Series(pd.to_datetime([None, None]))

    assert private_aggregate._finite_numeric_min_max(series) is None

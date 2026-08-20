from __future__ import annotations

import json
import weakref
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.runtime.resources import (
    MemoryLease,
    _OwnedValue,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance import interim, raw_exports
from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.runtime_resources import capture_test_policy


def _bound_first_frames(frames):  # noqa: ANN001, ANN202
    for frame in frames:
        lease = yield 256 * 1024
        yield _OwnedValue(frame, lease)


def _collect_bound_first(data, policy):  # noqa: ANN001, ANN202
    frames = []
    bound = next(data)
    while True:
        lease = policy.lease("test.binance.consumer", bound)
        while True:
            try:
                owned = data.send(lease)
            except StopIteration:
                lease.close()
                return frames
            if not isinstance(owned, _OwnedValue):
                raise AssertionError("invalid_bound_first_output")
            break
        frames.append(owned.value)
        try:
            bound = data.send(None)
        except StopIteration:
            lease.close()
            return frames
        lease.close()


def test_binance_interim_materializer_writes_source_datasets(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    _append_jsonl(
        raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
        _kline_envelope(0),
    )
    _append_jsonl(
        raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl",
        _trade_envelope(0),
    )
    _append_jsonl(
        raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl",
        _depth_envelope(0),
    )

    result = materialize_interim_data(layout=StorageLayout(project_root=tmp_path))

    assert sorted(result.artifact_roots) == [
        "aggregate_trades_tick",
        "depth_levels_tick",
        "market_bars_1m",
    ]
    market_bars = read_dataset(result.artifact_roots["market_bars_1m"])
    trades = read_dataset(result.artifact_roots["aggregate_trades_tick"])
    depth = read_dataset(result.artifact_roots["depth_levels_tick"])

    assert list(market_bars.columns) == [
        "source_asset_id",
        "timestamp",
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "source_market_type",
        "interval",
    ]
    assert market_bars.loc[0, "source_asset_id"] == "BTCUSDT"
    assert trades.loc[0, "source_asset_id"] == "BTCUSDT"
    assert trades.loc[0, "signed_quantity"] == 2.0
    assert set(depth["side"]) == {"bid", "ask"}
    assert depth["timestamp"].map(type).eq(pd.Timestamp).all()
    assert depth["timestamp"].nunique() == 1


def test_binance_file_discovery_occurs_inside_one_root_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def discover(_root: Path):
        observed.append(current_runtime_resource_policy())
        return {}

    monkeypatch.setattr(interim, "dataset_files_by_id", discover)
    result = materialize_interim_data(layout=StorageLayout(project_root=tmp_path))

    assert len(observed) == 1 and observed[0] is not None
    assert not result.artifact_roots


def test_binance_interim_iterator_rejects_competing_root_at_creation() -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            RuntimeError,
            match="runtime_resource_policy_competing_root",
        ):
            interim.iter_interim_frames(
                data_id="market_bars_1m",
                input_files=(),
            )


def test_binance_interim_iterator_keeps_explicit_root_during_delayed_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    observed = []

    def batches(*_args: object, resource_policy, **_kwargs: object):  # noqa: ANN001
        observed.append((resource_policy, current_runtime_resource_policy()))
        return iter(())

    monkeypatch.setattr(interim, "_iter_raw_envelope_batches", batches)
    iterator = interim.iter_interim_frames(
        data_id="market_bars_1m",
        input_files=(tmp_path / "unused.jsonl",),
        resource_policy=policy,
    )

    assert current_runtime_resource_policy() is None
    assert len(tuple(iterator)) == 1
    assert observed == [(policy, policy)]
    assert current_runtime_resource_policy() is None


def test_binance_interim_materializer_writes_multiple_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    for offset in range(5):
        _append_jsonl(
            raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
            _kline_envelope(offset),
        )
    chunk_lengths: list[int] = []

    def record_write_dataset(
        dataset_root,
        data,
        *,
        spec,
        instance_metadata=None,
        resource_policy,
    ):
        chunks = _collect_bound_first(data, resource_policy)
        chunk_lengths.extend(len(chunk) for chunk in chunks)
        from quant_research_platform.storage._dataset_writing import write_dataset

        return write_dataset(
            dataset_root,
            _bound_first_frames(chunks),
            spec=spec,
            instance_metadata=instance_metadata,
            resource_policy=resource_policy,
        )

    monkeypatch.setattr(
        "quant_research_platform.sources.binance.interim.write_dataset",
        record_write_dataset,
    )

    result = materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=("market_bars_1m",),
        row_batch_size=2,
    )
    market_bars = read_dataset(result.artifact_roots["market_bars_1m"])

    assert chunk_lengths == [2, 2, 1]
    assert len(market_bars) == 5


def test_binance_private_owner_state_is_bounded_after_many_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    for offset in range(3):
        _append_jsonl(first, _kline_envelope(offset))
    for offset in range(3, 5):
        _append_jsonl(second, _kline_envelope(offset))
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    closed: list[MemoryLease] = []
    original_close = MemoryLease.close

    def observed_close(lease: MemoryLease) -> None:
        closed.append(lease)
        original_close(lease)

    monkeypatch.setattr(MemoryLease, "close", observed_close)
    with runtime_resource_policy_scope(policy):
        frames = _collect_bound_first(
            interim._iter_owned_interim_frames(
                data_id="market_bars_1m",
                input_files=(first, second),
                row_batch_size=1,
                start_date=None,
                end_date=None,
                resource_policy=policy,
            ),
            policy,
        )
        assert len(frames) == 5
        assert policy.used_memory_bytes == 0
    consumer_ids = [
        id(lease) for lease in closed if lease.owner == "test.binance.consumer"
    ]
    assert len(consumer_ids) >= 5
    assert len(consumer_ids) == len(set(consumer_ids))


def test_binance_filtered_first_file_reuses_admitted_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    calls = 0

    def raw_batches(path, **_kwargs):  # noqa: ANN001, ANN003
        offset = 0 if Path(path).name == "first.jsonl" else 1
        return _bound_first_frames(((_kline_envelope(offset),),))

    monkeypatch.setattr(interim, "_iter_raw_envelope_batches", raw_batches)
    real_filter = interim._filter_time_window

    def filter_first(batch, **kwargs):  # noqa: ANN001, ANN003
        nonlocal calls
        calls += 1
        if calls == 1:
            return batch.iloc[:0].copy()
        return real_filter(batch, **kwargs)

    monkeypatch.setattr(interim, "_filter_time_window", filter_first)
    with runtime_resource_policy_scope(policy):
        output = _collect_bound_first(
            interim._iter_owned_interim_frames(
                data_id="market_bars_1m",
                input_files=(first, second),
                row_batch_size=1,
                start_date=None,
                end_date=None,
                resource_policy=policy,
            ),
            policy,
        )

    assert len(output) == 1
    assert calls == 2
    assert policy.used_memory_bytes == 0


def test_binance_raw_iterator_closes_without_closing_consumer_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    closed: list[bool] = []

    def raw_batches(*_args, **_kwargs):  # noqa: ANN002, ANN003
        try:
            lease = yield 256 * 1024
            yield _OwnedValue((_kline_envelope(0),), lease)
        finally:
            closed.append(True)

    monkeypatch.setattr(interim, "_iter_raw_envelope_batches", raw_batches)
    iterator = interim._iter_owned_interim_frames(
        data_id="market_bars_1m",
        input_files=(tmp_path / "unused.jsonl",),
        row_batch_size=1,
        start_date=None,
        end_date=None,
        resource_policy=policy,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        lease = policy.lease("test.binance.consumer", bound)
        owned = iterator.send(lease)
        assert isinstance(owned, _OwnedValue)
        assert lease.active
        with pytest.raises(KeyboardInterrupt, match="primary"):
            iterator.throw(KeyboardInterrupt("primary"))
        assert closed == [True]
        assert lease.active
        lease.close()

    assert policy.used_memory_bytes == 0


def test_binance_raw_bound_pull_admits_scan_before_record_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.jsonl"
    _append_jsonl(path, _kline_envelope(0))
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    pulls: list[bool] = []
    real_next_length = raw_exports._next_record_length

    def observe_next_length(handle, buffer):  # noqa: ANN001
        pulls.append(True)
        return real_next_length(handle, buffer)

    monkeypatch.setattr(raw_exports, "_next_record_length", observe_next_length)
    stream = raw_exports._iter_raw_envelope_batches(
        path,
        batch_size=1,
        resource_policy=policy,
        validate_depth_payload=False,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(stream)
        assert pulls == []
        assert policy.reserved_memory_bytes == 0
        lease = policy.lease("test.binance.raw", bound)
        try:
            owned = stream.send(lease)
            assert isinstance(owned, _OwnedValue)
            assert pulls
        finally:
            stream.close()
            lease.close()

    assert policy.used_memory_bytes == 0


def test_binance_raw_output_suspension_has_no_scan_record_duplicate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.jsonl"
    _append_jsonl(path, _kline_envelope(0))
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    stream = raw_exports._iter_raw_envelope_batches(
        path,
        batch_size=1,
        resource_policy=policy,
        validate_depth_payload=False,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(stream)
        lease = policy.lease("test.binance.raw", bound)
        try:
            owned = stream.send(lease)
            assert isinstance(owned, _OwnedValue)
            assert policy.reserved_memory_bytes == lease.reserved_bytes
        finally:
            stream.close()
            lease.close()

    assert policy.used_memory_bytes == 0


def test_binance_raw_oversize_first_record_fails_before_read_or_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw.jsonl"
    _append_jsonl(path, _kline_envelope(0))
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    declared: list[int] = []
    monkeypatch.setattr(raw_exports, "_next_record_length", lambda *_args: 1)
    monkeypatch.setattr(
        raw_exports,
        "_json_record_working_set_bytes",
        lambda _record_bytes: declared[0] + 1,
    )
    monkeypatch.setattr(
        raw_exports,
        "_loads_json",
        lambda *_args, **_kwargs: pytest.fail("decode reached before bound failure"),
    )
    stream = raw_exports._iter_raw_envelope_batches(
        path,
        batch_size=1,
        resource_policy=policy,
        validate_depth_payload=False,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(stream)
        lease = policy.lease("test.binance.raw", bound)
        declared.append(lease.reserved_bytes)
        try:
            with pytest.raises(
                RuntimeError,
                match="binance_record_envelope_bound_invalid",
            ):
                stream.send(lease)
        finally:
            stream.close()
            lease.close()

    assert policy.used_memory_bytes == 0


def test_binance_whitespace_only_fails_after_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "whitespace.jsonl"
    path.write_bytes(b"\n  \n")
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    stream = interim._iter_raw_envelope_batches(
        path,
        batch_size=1,
        resource_policy=policy,
        validate_depth_payload=False,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(stream)
        lease = policy.lease("test.binance.whitespace", bound)
        try:
            with pytest.raises(ValueError, match="empty_binance_raw_file"):
                stream.send(lease)
        finally:
            stream.close()
            lease.close()

    assert policy.used_memory_bytes == 0


def test_binance_blank_line_and_valid_row_share_one_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blank-first.jsonl"
    path.write_bytes(
        b"\n" + (json.dumps(_kline_envelope(0)) + "\n").encode("utf-8")
    )
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    stream = interim._iter_raw_envelope_batches(
        path,
        batch_size=1,
        resource_policy=policy,
        validate_depth_payload=False,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(stream)
        lease = policy.lease("test.binance.blank", bound)
        try:
            owned = stream.send(lease)
            assert isinstance(owned, _OwnedValue)
            assert owned.lease is lease
        finally:
            stream.close()
            lease.close()

    assert policy.used_memory_bytes == 0


def test_binance_interim_clears_previous_output_before_next_bound(
    tmp_path: Path,
) -> None:
    path = tmp_path / "two-rows.jsonl"
    _append_jsonl(path, _kline_envelope(0))
    _append_jsonl(path, _kline_envelope(1))
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    iterator = interim._iter_owned_interim_frames(
        data_id="market_bars_1m",
        input_files=(path,),
        row_batch_size=1,
        start_date=None,
        end_date=None,
        resource_policy=policy,
    )
    with runtime_resource_policy_scope(policy):
        bound = next(iterator)
        lease = policy.lease("test.binance.interim", bound)
        try:
            owned = iterator.send(lease)
            assert isinstance(owned, _OwnedValue)
            output_ref = weakref.ref(owned.value)
            del owned
            next_bound = iterator.send(None)
            assert type(next_bound) is int and next_bound > 0
            assert output_ref() is None
        finally:
            iterator.close()
            lease.close()

    assert policy.used_memory_bytes == 0


def test_binance_depth_interim_parses_and_validates_order_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "depth.jsonl"
    _append_jsonl(path, _depth_envelope(0))
    raw_validations = 0
    side_parses = 0
    order_validations = 0
    raw_validate = raw_exports._validate_depth_payload
    parse_side = interim.depth_side_levels
    validate_order = interim._validate_normalized_order

    def count_raw(payload):  # noqa: ANN001, ANN202
        nonlocal raw_validations
        raw_validations += 1
        return raw_validate(payload)

    def count_side(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal side_parses
        side_parses += 1
        return parse_side(*args, **kwargs)

    def count_order(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal order_validations
        order_validations += 1
        return validate_order(*args, **kwargs)

    monkeypatch.setattr(raw_exports, "_validate_depth_payload", count_raw)
    monkeypatch.setattr(interim, "depth_side_levels", count_side)
    monkeypatch.setattr(interim, "_validate_normalized_order", count_order)

    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    chunks = tuple(
        interim.iter_interim_frames(
            data_id="depth_levels_tick",
            input_files=(path,),
            row_batch_size=1,
            resource_policy=policy,
        )
    )

    assert chunks
    assert raw_validations == 0
    assert side_parses == 2
    assert order_validations == 1


@pytest.mark.parametrize(
    ("data_id", "path_parts", "envelope_kind"),
    [
        (
            "aggregate_trades_tick",
            ("spot", "agg_trades", "BTCUSDT", "2026-07-06.jsonl"),
            "trade",
        ),
        (
            "depth_levels_tick",
            ("spot", "depth", "BTCUSDT", "2026-07-06.jsonl"),
            "depth",
        ),
    ],
)
@pytest.mark.parametrize("row_batch_size", [1, 2])
def test_binance_interim_rejects_global_order_reversal_independent_of_batch_size(
    tmp_path: Path,
    data_id: str,
    path_parts: tuple[str, ...],
    envelope_kind: str,
    row_batch_size: int,
) -> None:
    path = tmp_path.joinpath(*path_parts)
    envelope_factory = _trade_envelope if envelope_kind == "trade" else _depth_envelope
    _append_jsonl(path, envelope_factory(1))
    _append_jsonl(path, envelope_factory(0))

    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    with pytest.raises(ValueError, match="out_of_order_binance_interim_dependency"):
        tuple(
            interim.iter_interim_frames(
                data_id=data_id,
                input_files=(path,),
                row_batch_size=row_batch_size,
                resource_policy=policy,
            )
        )


def test_binance_date_only_interim_end_includes_full_day(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    for offset in range(2):
        _append_jsonl(
            raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
            _kline_envelope(offset),
        )
        _append_jsonl(
            raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl",
            _trade_envelope(offset),
        )
        _append_jsonl(
            raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl",
            _depth_envelope(offset),
        )

    result = materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        start_date="2026-07-06",
        end_date="2026-07-06",
        row_batch_size=1,
    )

    assert len(read_dataset(result.artifact_roots["market_bars_1m"])) == 2
    assert len(read_dataset(result.artifact_roots["aggregate_trades_tick"])) == 2
    assert len(read_dataset(result.artifact_roots["depth_levels_tick"])) == 6


def test_binance_datetime_interim_end_filters_precisely(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    for offset in range(3):
        _append_jsonl(
            raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
            _kline_envelope(offset),
        )
        _append_jsonl(
            raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl",
            _trade_envelope(offset),
        )
        _append_jsonl(
            raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl",
            _depth_envelope(offset),
        )

    result = materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        start_date="2026-07-06T00:01:00",
        end_date="2026-07-06T00:01:02",
        row_batch_size=1,
    )

    assert read_dataset(result.artifact_roots["market_bars_1m"])[
        "timestamp"
    ].tolist() == [pd.Timestamp("2026-07-06 00:01:00")]
    assert read_dataset(result.artifact_roots["aggregate_trades_tick"])[
        "timestamp"
    ].tolist() == [pd.Timestamp("2026-07-06 00:01:01")]
    assert (
        read_dataset(result.artifact_roots["depth_levels_tick"])["timestamp"].nunique()
        == 1
    )


def test_binance_interim_materializer_writes_empty_schema_after_filtering(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    _append_jsonl(
        raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl",
        _trade_envelope(0),
    )

    result = materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        start_date="2026-07-07",
        end_date="2026-07-07",
        data_ids=("aggregate_trades_tick",),
    )

    trades = read_dataset(result.artifact_roots["aggregate_trades_tick"])
    assert trades.empty
    assert (
        tuple(trades.columns)
        == interim.INTERIM_COLUMNS_BY_DATA_ID["aggregate_trades_tick"]
    )


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _kline_envelope(offset: int) -> dict[str, object]:
    open_time = 1783296000000 + offset * 60_000
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "klines",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": [
            open_time,
            str(100 + offset),
            str(101 + offset),
            str(99 + offset),
            str(100.5 + offset),
            "10.0",
            open_time + 59_999,
            "1005.0",
            7,
            "4.0",
            "402.0",
            "0",
        ],
    }


def _trade_envelope(offset: int) -> dict[str, object]:
    open_time = 1783296000000 + offset * 60_000
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "agg_trades",
        "symbol": "BTCUSDT",
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": {
            "a": offset + 1,
            "p": str(100.1 + offset),
            "q": "2.0",
            "f": offset * 2,
            "l": offset * 2 + 1,
            "T": open_time + 1_000,
            "m": False,
            "M": True,
        },
    }


def _depth_envelope(offset: int) -> dict[str, object]:
    open_time = 1783296000000 + offset * 60_000
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "depth",
        "symbol": "BTCUSDT",
        "captured_at": pd.Timestamp(open_time + 2_000, unit="ms", tz="UTC").isoformat(),
        "payload": {
            "lastUpdateId": offset + 1,
            "bids": [[str(100 + offset), "1.5"], [str(99.9 + offset), "2.0"]],
            "asks": [[str(100.2 + offset), "1.0"]],
        },
    }

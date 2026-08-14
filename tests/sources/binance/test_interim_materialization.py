from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.runtime.resources import current_runtime_resource_policy
from quant_research_platform.sources.binance import interim, raw_exports
from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout


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
        chunks = tuple(data)
        chunk_lengths.extend(len(chunk) for chunk in chunks)
        from quant_research_platform.storage._dataset_writing import write_dataset

        return write_dataset(
            dataset_root,
            chunks,
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

    chunks = tuple(
        interim.iter_interim_frames(
            data_id="depth_levels_tick",
            input_files=(path,),
            row_batch_size=1,
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

    with pytest.raises(ValueError, match="out_of_order_binance_interim_dependency"):
        tuple(
            interim.iter_interim_frames(
                data_id=data_id,
                input_files=(path,),
                row_batch_size=row_batch_size,
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

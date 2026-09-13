"""Bounded HFT market-fact producers over exact recorder/PANEL inputs."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace

import pandas as pd

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import is_intraday_time_window
from quant_research_platform.markets.implementations.crypto.hft import (
    HFT_GRID_MS,
    iter_factor_frames,
    iter_label_frames,
    iter_panel_frames,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    MARKET_ID,
    canonical_crypto_asset_id,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.protocols.crypto_hft import HFT_LABEL_HORIZONS_MS
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.sources.binance import ids as binance_ids
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import dataset_identity_payload

_PANEL_ROW_BUFFER_BYTES = 4 * 1024
_FRAME_FIXED_BYTES = 1024 * 1024
_EPOCH = dt.datetime(1970, 1, 1)


@contextmanager
def produce_hft_processed_data(
    spec,
    *,
    layout,
    time_window,
    processed_catalog,
    source_catalog,
    resource_policy,
    open_run,
    reconstruct_depth,
    read_trades,
    make_stats,
    raw_root=None,
    expected_source_identity=None,
):
    """Keep readers alive until the shared writer's prepublication validator."""
    if not is_intraday_time_window(time_window):
        raise ValueError("hft_materialization_requires_intraday_window")
    metadata = {}
    with ExitStack() as readers:
        if spec.data_id == HFT_PANEL_DATA_ID:
            run = readers.enter_context(open_run(raw_root, resource_policy))
            if canonical_json_hash(run.source_identity) != expected_source_identity:
                raise ValueError("hft_recorder_source_identity_changed")
            source_info = run.hft_source_info()
            start_ms, end_ms = (
                _milliseconds(time_window.start, ceil=True),
                _milliseconds(time_window.end),
            )
            stats = {symbol: make_stats() for symbol in binance_ids.RESEARCH_SYMBOLS}
            target = resource_policy.operation_headroom_bytes // 2
            if target <= _FRAME_FIXED_BYTES + _PANEL_ROW_BUFFER_BYTES:
                raise MemoryError("runtime_memory_budget_exceeded")
            batch_rows = (target - _FRAME_FIXED_BYTES) // _PANEL_ROW_BUFFER_BYTES
            bound = _FRAME_FIXED_BYTES + batch_rows * _PANEL_ROW_BUFFER_BYTES
            symbol_rows = max(1, batch_rows // len(binance_ids.RESEARCH_SYMBOLS))

            def symbol_frames(symbol):
                active = stats[symbol]
                return iter_panel_frames(
                    reconstruct_depth(
                        run,
                        binance_ids.FUTURES_MARKET_TYPE,
                        symbol=symbol,
                        stats=active,
                        end_ns=end_ms * 1_000_000,
                    ),
                    read_trades(
                        run,
                        binance_ids.FUTURES_MARKET_TYPE,
                        symbol=symbol,
                        end_ms=end_ms,
                    ),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    book_coverage=active.coverage_segments,
                    trade_coverage=run.observation_coverage(
                        market_type=binance_ids.FUTURES_MARKET_TYPE,
                        data_kind="agg_trades",
                        symbol=symbol,
                    ).covering,
                    market_id=str(MARKET_ID),
                    asset_id=canonical_crypto_asset_id(symbol),
                    batch_rows=symbol_rows,
                )

            def frames():
                with closing_iterator(
                    _common_symbol_panel_frames(
                        {
                            symbol: symbol_frames(symbol)
                            for symbol in binance_ids.RESEARCH_SYMBOLS
                        },
                        columns=spec.required_columns,
                        batch_rows=batch_rows,
                    )
                ) as active:
                    yield from _event_frame_producer(
                        active, columns=spec.required_columns, producer_bound=bound
                    )
                metadata["producer_diagnostics"] = {
                    "observation_grid": "common_btc_eth_observed_timestamps",
                    **source_info,
                    "reconstruction": {
                        symbol: asdict(value) for symbol, value in stats.items()
                    },
                }

            metadata["source_artifact_identity"] = dict(run.source_identity)
            data = frames()

            def verify(_receipt):
                readers.close()
        elif spec.data_id in {HFT_FACTOR_DATA_ID, HFT_LABEL_DATA_ID}:
            panel_spec = processed_catalog.get(HFT_PANEL_DATA_ID)
            root = layout.target_root(
                HFT_PANEL_DATA_ID,
                layout.processed_data_root(
                    panel_spec.market_id, panel_spec.kind, panel_spec.data_id
                ),
            )
            session = readers.enter_context(
                open_dataset_session(
                    root,
                    columns=None,
                    time_window=time_window,
                    resource_policy=resource_policy,
                )
            )
            metadata["input_dataset_identities"] = {
                HFT_PANEL_DATA_ID: dataset_identity_payload(
                    session.spec, session.manifest
                )
            }
            transform = (
                iter_factor_frames
                if spec.data_id == HFT_FACTOR_DATA_ID
                else lambda frames: iter_label_frames(
                    frames, horizons_ms=HFT_LABEL_HORIZONS_MS
                )
            )
            data = _panel_frame_producer(
                session,
                transform=transform,
                columns=spec.required_columns,
                resource_policy=resource_policy,
            )

            def verify(_receipt):
                session.verify()
                readers.close()
        else:
            raise ValueError("hft_processed_target_invalid")
        with closing_iterator(data):
            yield data, metadata, verify


def _common_symbol_panel_frames(
    frames_by_symbol: dict[str, Iterable[pd.DataFrame]],
    *,
    columns: tuple[str, ...],
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    """Merge sorted symbol rows by timestamp, independent of frame boundaries."""

    if set(frames_by_symbol) != set(binance_ids.RESEARCH_SYMBOLS) or batch_rows < len(
        binance_ids.RESEARCH_SYMBOLS
    ):
        raise ValueError("hft_common_panel_inputs_invalid")
    timestamp_position = columns.index("timestamp")
    segment_position = columns.index("segment_id")

    def rows(frames: Iterable[pd.DataFrame]) -> Iterator[tuple[object, ...]]:
        previous = None
        with closing_iterator(iter(frames)) as active:
            for frame in active:
                if tuple(frame.columns) != columns:
                    raise ValueError("hft_panel_frame_columns_invalid")
                for row in frame.itertuples(index=False, name=None):
                    timestamp = row[timestamp_position]
                    if previous is not None and timestamp <= previous:
                        raise ValueError("hft_symbol_panel_timestamp_invalid")
                    previous = timestamp
                    yield row

    streams = [
        rows(frames_by_symbol[symbol]) for symbol in binance_ids.RESEARCH_SYMBOLS
    ]
    records: list[tuple[object, ...]] = []
    previous_common_time = None
    common_start_ns = None
    with ExitStack() as live:
        active = [live.enter_context(closing_iterator(stream)) for stream in streams]
        empty = object()
        current = [next(stream, empty) for stream in active]
        if any(row is empty for row in current):
            for stream in active:
                for _row in stream:
                    pass
            return
        ended = False
        while not ended:
            target = max(row[timestamp_position] for row in current)
            for index, stream in enumerate(active):
                while current[index][timestamp_position] < target:
                    try:
                        current[index] = next(stream)
                    except StopIteration:
                        ended = True
                        break
                if ended:
                    break
            if ended:
                break
            if any(row[timestamp_position] != target for row in current):
                continue
            if records and len(records) + len(current) > batch_rows:
                yield pd.DataFrame.from_records(records, columns=columns).sort_values(
                    ["timestamp", "asset_id"], kind="mergesort", ignore_index=True
                )
                records.clear()
            if (
                previous_common_time is None
                or target - previous_common_time
                != pd.Timedelta(milliseconds=HFT_GRID_MS)
            ):
                common_start_ns = pd.Timestamp(target).value
            previous_common_time = target
            records.extend(
                (
                    *row[:segment_position],
                    f"{row[segment_position]}:common@{common_start_ns}",
                    *row[segment_position + 1 :],
                )
                for row in current
            )
            for index, stream in enumerate(active):
                try:
                    current[index] = next(stream)
                except StopIteration:
                    ended = True
            if len(records) >= batch_rows:
                yield pd.DataFrame.from_records(records, columns=columns).sort_values(
                    ["timestamp", "asset_id"], kind="mergesort", ignore_index=True
                )
                records.clear()
        if ended:
            for stream in active:
                for _row in stream:
                    pass
        if records:
            yield pd.DataFrame.from_records(records, columns=columns).sort_values(
                ["timestamp", "asset_id"], kind="mergesort", ignore_index=True
            )


def _milliseconds(value: dt.datetime, *, ceil: bool = False) -> int:
    delta = value - _EPOCH
    if delta < dt.timedelta(0):
        raise ValueError("hft_time_window_before_epoch")
    unit = dt.timedelta(milliseconds=1)
    return -(-delta // unit) if ceil else delta // unit


def _event_frame_producer(
    frames: Iterable[pd.DataFrame],
    *,
    columns: tuple[str, ...],
    producer_bound: int,
) -> Iterator[object]:
    active_frames = iter(frames)
    with closing_iterator(active_frames):
        while True:
            lease = yield producer_bound
            try:
                frame = next(active_frames)
            except StopIteration:
                return
            if tuple(frame.columns) != columns:
                raise ValueError("hft_panel_frame_columns_invalid")
            actual_bound = pandas_frame_payload_bytes(frame) + 64 * 1024
            if actual_bound > lease.reserved_bytes:
                raise ValueError("hft_dataframe_producer_bound_invalid")
            lease.shrink_to(actual_bound)
            yield OwnedValue(frame, lease)
            del frame


def _panel_frame_producer(
    session: DatasetSession,
    *,
    transform: Callable[[Iterable[pd.DataFrame]], Iterable[pd.DataFrame]],
    columns: tuple[str, ...],
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[object]:
    target_chunk_bytes = max(1, resource_policy.operation_headroom_bytes // 8)
    producer_bound = resource_policy.operation_headroom_bytes // 2
    if producer_bound <= _FRAME_FIXED_BYTES:
        raise MemoryError("runtime_memory_budget_exceeded")

    def frames() -> Iterator[pd.DataFrame]:
        for symbol in binance_ids.RESEARCH_SYMBOLS:
            asset_filter = ("asset_id", "==", canonical_crypto_asset_id(symbol))
            asset_session = replace(
                session,
                parquet_filters=(*(session.parquet_filters or ()), asset_filter),
                owns_inventory=False,
                owns_lease=False,
            )
            chunks = iter_dataset_chunks(
                asset_session, target_chunk_bytes=target_chunk_bytes
            )
            with (
                closing_iterator(chunks),
                closing_iterator(iter(transform(chunks))) as transformed,
            ):
                for frame in transformed:
                    yield frame.loc[:, list(columns)]

    yield from _event_frame_producer(
        frames(), columns=columns, producer_bound=producer_bound
    )

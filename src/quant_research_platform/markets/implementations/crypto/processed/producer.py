"""Bounded HFT market-fact producers over exact recorder/PANEL inputs."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from itertools import chain

import pandas as pd

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import is_datetime_window
from quant_research_platform.markets.implementations.crypto.hft import (
    HFT_GRID_MS,
    iter_cross_asset_factor_frames,
    iter_derivative_factor_frames,
    iter_external_context_frames,
    iter_factor_frames,
    iter_label_frames,
    iter_panel_frames,
    iter_spot_perp_factor_frames,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_INFORMATION_STATE_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_SPOT_PANEL_DATA_ID,
    MARKET_ID,
    canonical_crypto_asset_id,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.protocols.crypto_hft import HFT_FACTOR_COLUMNS
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

# OKX USDT contract face values effective March 2020; Bybit linear sizes are base.
EXTERNAL_INSTRUMENTS = {
    "bybit": {
        "exchange": "bybit",
        "symbols": {"btc_usdt": "BTCUSDT", "eth_usdt": "ETHUSDT"},
        "base_units_per_amount": {"btc_usdt": 1.0, "eth_usdt": 1.0},
    },
    "okx": {
        "exchange": "okex-swap",
        "symbols": {"btc_usdt": "BTC-USDT-SWAP", "eth_usdt": "ETH-USDT-SWAP"},
        "base_units_per_amount": {"btc_usdt": 0.01, "eth_usdt": 0.1},
    },
}


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
    market_type=binance_ids.FUTURES_MARKET_TYPE,
    symbols=binance_ids.RESEARCH_SYMBOLS,
    panel_id=HFT_PANEL_DATA_ID,
):
    """Keep readers alive until the shared writer's prepublication validator."""
    if not is_datetime_window(time_window):
        raise ValueError("hft_materialization_requires_intraday_window")
    metadata = {}
    with ExitStack() as readers:
        if spec.data_id == panel_id:
            run = readers.enter_context(open_run(raw_root, resource_policy))
            if canonical_json_hash(run.source_identity) != expected_source_identity:
                raise ValueError("hft_recorder_source_identity_changed")
            source_info = run.hft_source_info()
            start_ms, end_ms = (
                _milliseconds(time_window.start, ceil=True),
                _milliseconds(time_window.end),
            )
            stats = {symbol: make_stats() for symbol in symbols}
            target = max(
                _FRAME_FIXED_BYTES + _PANEL_ROW_BUFFER_BYTES,
                resource_policy.planning_headroom_bytes // 2,
            )
            batch_rows = (target - _FRAME_FIXED_BYTES) // _PANEL_ROW_BUFFER_BYTES
            bound = _FRAME_FIXED_BYTES + batch_rows * _PANEL_ROW_BUFFER_BYTES
            symbol_rows = max(1, batch_rows // len(symbols))

            def symbol_frames(symbol):
                active = stats[symbol]
                return iter_panel_frames(
                    reconstruct_depth(
                        run,
                        market_type,
                        symbol=symbol,
                        stats=active,
                        output_levels=10,
                        end_ns=end_ms * 1_000_000,
                    ),
                    read_trades(
                        run,
                        market_type,
                        symbol=symbol,
                        end_ms=end_ms,
                    ),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    book_coverage=active.coverage_segments,
                    trade_coverage=run.observation_coverage(
                        market_type=market_type,
                        data_kind="agg_trades",
                        symbol=symbol,
                    ).covering,
                    market_id=str(MARKET_ID),
                    asset_id=canonical_crypto_asset_id(symbol),
                    batch_rows=symbol_rows,
                )

            synchronize = market_type == binance_ids.FUTURES_MARKET_TYPE and tuple(
                symbols
            ) == tuple(binance_ids.RESEARCH_SYMBOLS)

            def independent_frames():
                for symbol in symbols:
                    with closing_iterator(symbol_frames(symbol)) as active:
                        yield from active

            def frames():
                source = (
                    _common_symbol_panel_frames(
                        {symbol: symbol_frames(symbol) for symbol in symbols},
                        columns=spec.required_columns,
                        batch_rows=batch_rows,
                    )
                    if synchronize
                    else independent_frames()
                )
                with closing_iterator(source) as active:
                    yield from _event_frame_producer(
                        active, columns=spec.required_columns, producer_bound=bound
                    )
                metadata["producer_diagnostics"] = {
                    "observation_grid": "common_btc_eth_observed_timestamps"
                    if synchronize
                    else "independent_asset_observed_timestamps",
                    **source_info,
                    "reconstruction": {
                        symbol: asdict(value) for symbol, value in stats.items()
                    },
                }

            metadata["source_artifact_identity"] = dict(run.source_identity)
            data = frames()

            def verify(_receipt):
                readers.close()
        elif spec.data_id in {
            HFT_FACTOR_DATA_ID,
            HFT_LABEL_DATA_ID,
            HFT_INFORMATION_STATE_DATA_ID,
        }:
            input_id = spec.processed_input_data_ids[0]
            panel_spec = processed_catalog.get(input_id)
            root = layout.target_root(
                input_id,
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
                input_id: dataset_identity_payload(session.spec, session.manifest)
            }
            transform = (
                (
                    lambda frames: iter_factor_frames(
                        frames,
                        columns=tuple(
                            name
                            for name in spec.required_columns
                            if name in HFT_FACTOR_COLUMNS
                        ),
                    )
                )
                if spec.data_id != HFT_LABEL_DATA_ID
                else lambda frames: iter_label_frames(
                    frames,
                    horizons_ms=tuple(
                        sorted(
                            {
                                item["horizon"]
                                for item in json.loads(spec.parameters_json)["labels"]
                            }
                        )
                    ),
                )
            )
            data = _panel_frame_producer(
                session,
                transform=transform,
                columns=spec.required_columns,
                resource_policy=resource_policy,
                asset_ids=tuple(json.loads(spec.parameters_json)["asset_ids"]),
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
    dtypes = {}
    timestamp_position = columns.index("timestamp")
    segment_position = columns.index("segment_id")

    def rows(frames: Iterable[pd.DataFrame]) -> Iterator[tuple[object, ...]]:
        previous = None
        with closing_iterator(iter(frames)) as active:
            for frame in active:
                if tuple(frame.columns) != columns:
                    raise ValueError("hft_panel_frame_columns_invalid")
                dtypes.update(frame.dtypes)
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
                yield (
                    pd.DataFrame.from_records(records, columns=columns)
                    .astype(dtypes)
                    .sort_values(
                        ["timestamp", "asset_id"], kind="mergesort", ignore_index=True
                    )
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
                yield (
                    pd.DataFrame.from_records(records, columns=columns)
                    .astype(dtypes)
                    .sort_values(
                        ["timestamp", "asset_id"], kind="mergesort", ignore_index=True
                    )
                )
                records.clear()
        if ended:
            for stream in active:
                for _row in stream:
                    pass
        if records:
            yield (
                pd.DataFrame.from_records(records, columns=columns)
                .astype(dtypes)
                .sort_values(
                    ["timestamp", "asset_id"], kind="mergesort", ignore_index=True
                )
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
    asset_ids: tuple[str, ...],
) -> Iterator[object]:
    target_chunk_bytes = max(1, resource_policy.planning_headroom_bytes // 8)
    producer_bound = max(
        _FRAME_FIXED_BYTES + _PANEL_ROW_BUFFER_BYTES,
        resource_policy.planning_headroom_bytes // 2,
    )

    def frames() -> Iterator[pd.DataFrame]:
        for asset in asset_ids:
            asset_filter = ("asset_id", "==", asset)
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


@contextmanager
def produce_derivative_processed_data(
    spec,
    *,
    layout,
    time_window,
    processed_catalog,
    resource_policy,
    open_archive,
    expected_source_identity,
    source_catalog,
    raw_root=None,
):
    """Align auxiliary states upstream of the unchanged exact-key assembler."""
    max_age_ms = json.loads(spec.parameters_json)["max_age_ms"]
    panel_spec = processed_catalog.get(HFT_PANEL_DATA_ID)
    root = layout.target_root(
        HFT_PANEL_DATA_ID,
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
    )
    with ExitStack() as readers:
        archive = readers.enter_context(open_archive(raw_root, resource_policy))
        session = readers.enter_context(
            open_dataset_session(
                root,
                columns=None,
                time_window=time_window,
                resource_policy=resource_policy,
            )
        )
        if canonical_json_hash(archive.source_identity) != expected_source_identity:
            raise ValueError("hft_derivative_source_identity_changed")
        metadata = {
            "source_artifact_identity": dict(archive.source_identity),
            "input_dataset_identities": {
                HFT_PANEL_DATA_ID: dataset_identity_payload(
                    session.spec, session.manifest
                ),
            },
            "producer_diagnostics": {
                "alignment": "observed_at_lte_decision_time",
                "max_age_ms": max_age_ms,
                "open_interest_unit": "base",
                "force_order_quantity": "observed_cumulative_filled_base_z",
                "force_order_coverage": "observed_proxy_not_complete_liquidation_tape",
            },
        }

        def transform(frames):
            first = next(frames, None)
            if first is None:
                return
            symbol = {
                canonical_crypto_asset_id(value): value
                for value in binance_ids.RESEARCH_SYMBOLS
            }[first["asset_id"].iloc[0]]
            yield from iter_derivative_factor_frames(
                chain((first,), frames),
                observations=archive.iter_state_observations(symbol=symbol),
                max_age_ms=max_age_ms,
            )

        data = _panel_frame_producer(
            session,
            transform=transform,
            columns=spec.required_columns,
            resource_policy=resource_policy,
            asset_ids=tuple(
                canonical_crypto_asset_id(symbol)
                for symbol in binance_ids.RESEARCH_SYMBOLS
            ),
        )

        def verify(_receipt):
            session.verify()
            readers.close()

        with closing_iterator(data):
            yield data, metadata, verify


@contextmanager
def produce_spot_perp_context(
    spec,
    *,
    layout,
    time_window,
    processed_catalog,
    source_catalog,
    resource_policy,
):
    """Publish information rows on the execution panel's exact primary keys."""
    max_age_ms = json.loads(spec.parameters_json)["max_age_ms"]
    with ExitStack() as readers:
        sessions = {}
        for data_id in (HFT_PANEL_DATA_ID, HFT_SPOT_PANEL_DATA_ID):
            input_spec = processed_catalog.get(data_id)
            root = layout.target_root(
                data_id,
                layout.processed_data_root(
                    input_spec.market_id, input_spec.kind, data_id
                ),
            )
            sessions[data_id] = readers.enter_context(
                open_dataset_session(
                    root,
                    columns=None,
                    time_window=time_window,
                    resource_policy=resource_policy,
                )
            )
        metadata = {
            "input_dataset_identities": {
                key: dataset_identity_payload(session.spec, session.manifest)
                for key, session in sessions.items()
            },
            "producer_diagnostics": {
                "alignment": "spot_grid_available_by_perpetual_decision",
                "max_age_ms": max_age_ms,
                "lag_ms": 100,
                "execution_market": "binance_futures_btc_eth_only",
                "clock_interpretation": (
                    "vendor_observation_order_not_colocated_latency"
                ),
            },
        }
        target_bytes = max(1, resource_policy.planning_headroom_bytes // 32)
        producer_bound = max(
            _FRAME_FIXED_BYTES + _PANEL_ROW_BUFFER_BYTES,
            resource_policy.planning_headroom_bytes // 2,
        )
        batch_rows = (producer_bound - _FRAME_FIXED_BYTES) // _PANEL_ROW_BUFFER_BYTES

        def frames():
            for symbol in binance_ids.RESEARCH_SYMBOLS:
                asset = canonical_crypto_asset_id(symbol)
                asset_sessions = {
                    key: replace(
                        session,
                        parquet_filters=(
                            *(session.parquet_filters or ()),
                            ("asset_id", "==", asset),
                        ),
                        owns_inventory=False,
                        owns_lease=False,
                    )
                    for key, session in sessions.items()
                }
                with (
                    closing_iterator(
                        iter_dataset_chunks(
                            asset_sessions[HFT_PANEL_DATA_ID],
                            target_chunk_bytes=target_bytes,
                        )
                    ) as perpetual,
                    closing_iterator(
                        iter_dataset_chunks(
                            asset_sessions[HFT_SPOT_PANEL_DATA_ID],
                            target_chunk_bytes=target_bytes,
                        )
                    ) as spot,
                    closing_iterator(
                        iter_spot_perp_factor_frames(
                            perpetual,
                            spot_frames=spot,
                            max_age_ms=max_age_ms,
                            batch_rows=batch_rows,
                        )
                    ) as aligned,
                ):
                    for frame in aligned:
                        frame["market_id"] = str(MARKET_ID)
                        frame["asset_id"] = asset
                        yield frame.loc[:, list(spec.required_columns)]

        data = _event_frame_producer(
            frames(),
            columns=spec.required_columns,
            producer_bound=producer_bound,
        )

        def verify(_receipt):
            for session in sessions.values():
                session.verify()
            readers.close()

        with closing_iterator(data):
            yield data, metadata, verify


@contextmanager
def produce_information_panel(
    spec,
    *,
    layout,
    time_window,
    processed_catalog,
    source_catalog,
    resource_policy,
):
    """Union disjoint BTC/ETH and SOL rows, preserving native source segments."""
    with ExitStack() as readers:
        sessions = []
        identities = {}
        for data_id in spec.processed_input_data_ids:
            source_spec = processed_catalog.get(data_id)
            root = layout.target_root(
                data_id,
                layout.processed_data_root(
                    source_spec.market_id, source_spec.kind, data_id
                ),
            )
            session = readers.enter_context(
                open_dataset_session(
                    root,
                    columns=spec.required_columns,
                    time_window=time_window,
                    resource_policy=resource_policy,
                )
            )
            sessions.append(session)
            identities[data_id] = dataset_identity_payload(
                session.spec, session.manifest
            )
        bound = max(
            _FRAME_FIXED_BYTES + _PANEL_ROW_BUFFER_BYTES,
            resource_policy.planning_headroom_bytes // 2,
        )

        def frames():
            for session in sessions:
                with closing_iterator(
                    iter_dataset_chunks(session, target_chunk_bytes=max(1, bound // 4))
                ) as chunks:
                    yield from chunks

        data = _event_frame_producer(
            frames(), columns=spec.required_columns, producer_bound=bound
        )

        def verify(_receipt):
            for session in sessions:
                session.verify()
            readers.close()

        with closing_iterator(data):
            yield data, {"input_dataset_identities": identities}, verify


@contextmanager
def produce_cross_asset_context(
    spec,
    *,
    layout,
    time_window,
    processed_catalog,
    source_catalog,
    resource_policy,
):
    """Read persisted shared-kernel information without creating an execution venue."""
    parameters = json.loads(spec.parameters_json)
    target_id, information_id = spec.processed_input_data_ids
    with ExitStack() as readers:
        sessions = {}
        for data_id in spec.processed_input_data_ids:
            data_spec = processed_catalog.get(data_id)
            root = layout.target_root(
                data_id,
                layout.processed_data_root(
                    data_spec.market_id, data_spec.kind, data_id
                ),
            )
            sessions[data_id] = readers.enter_context(
                open_dataset_session(
                    root,
                    columns=None,
                    time_window=time_window,
                    resource_policy=resource_policy,
                )
            )
        metadata = {
            "input_dataset_identities": {
                key: dataset_identity_payload(session.spec, session.manifest)
                for key, session in sessions.items()
            },
            "producer_diagnostics": {
                "information_assets": parameters["information_assets"],
                "target_assets": parameters["target_assets"],
                "max_age_ms": parameters["max_age_ms"],
                "units": "mature_simple_return_and_dimensionless_order_flow",
                "execution_assets": ["btc_usdt", "eth_usdt"],
            },
        }
        bound = max(
            _FRAME_FIXED_BYTES + _PANEL_ROW_BUFFER_BYTES,
            resource_policy.planning_headroom_bytes // 2,
        )
        batch_rows = (bound - _FRAME_FIXED_BYTES) // _PANEL_ROW_BUFFER_BYTES
        chunk_bytes = max(1, resource_policy.planning_headroom_bytes // 64)

        def asset_chunks(session, asset):
            view = replace(
                session,
                parquet_filters=(
                    *(session.parquet_filters or ()),
                    ("asset_id", "==", asset),
                ),
                owns_inventory=False,
                owns_lease=False,
            )
            yield from iter_dataset_chunks(view, target_chunk_bytes=chunk_bytes)

        def frames():
            for asset in parameters["target_assets"]:
                with closing_iterator(
                    iter_cross_asset_factor_frames(
                        asset_chunks(sessions[target_id], asset),
                        information_frames={
                            key: asset_chunks(sessions[information_id], key)
                            for key in parameters["information_assets"]
                        },
                        max_age_ms=parameters["max_age_ms"],
                        batch_rows=batch_rows,
                    )
                ) as aligned:
                    for frame in aligned:
                        frame["market_id"] = str(MARKET_ID)
                        frame["asset_id"] = asset
                        yield frame.loc[:, list(spec.required_columns)]

        data = _event_frame_producer(
            frames(), columns=spec.required_columns, producer_bound=bound
        )

        def verify(_receipt):
            for session in sessions.values():
                session.verify()
            readers.close()

        with closing_iterator(data):
            yield data, metadata, verify


@contextmanager
def produce_normalized_context(
    spec,
    *,
    layout,
    time_window,
    processed_catalog,
    source_catalog,
    resource_policy,
    raw_root,
    expected_source_identity,
    read_records,
    source_identity,
    archives,
):
    """Publish source-verified external information on Binance BTC/ETH keys."""
    parameters = json.loads(spec.parameters_json)
    if source_identity(raw_root, spec.data_id) != expected_source_identity:
        raise ValueError("hft_external_source_identity_changed")
    panel_spec = processed_catalog.get(HFT_PANEL_DATA_ID)
    root = layout.target_root(
        HFT_PANEL_DATA_ID,
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
    )
    with ExitStack() as readers:
        session = readers.enter_context(
            open_dataset_session(
                root,
                columns=None,
                time_window=time_window,
                resource_policy=resource_policy,
            )
        )
        metadata = {
            "source_artifact_identity": {
                "normalized_archives_sha256": expected_source_identity
            },
            "input_dataset_identities": {
                HFT_PANEL_DATA_ID: dataset_identity_payload(
                    session.spec, session.manifest
                ),
            },
            "producer_diagnostics": {
                "alignment": "local_timestamp_lte_decision_time",
                "coverage": "observed_only",
                "units": {
                    "quote_currency": "USDT",
                    "base_units_per_amount": parameters["base_units_per_amount"],
                },
                "max_age_ms": parameters["max_age_ms"],
                "execution_assets": ["btc_usdt", "eth_usdt"],
            },
        }

        def transform(frames):
            first = next(frames, None)
            if first is None:
                return
            asset = first["asset_id"].iloc[0]
            symbol = parameters["symbols"][asset]
            paths = archives[symbol]
            yield from iter_external_context_frames(
                chain((first,), frames),
                quotes=read_records(
                    raw_root / paths["quotes"], resource_policy=resource_policy
                ),
                trades=read_records(
                    raw_root / paths["trades"], resource_policy=resource_policy
                ),
                tickers=read_records(
                    raw_root / paths["derivative_ticker"],
                    resource_policy=resource_policy,
                ),
                venue=parameters["venue"],
                base_units_per_amount=parameters["base_units_per_amount"][asset],
                max_age_ms=parameters["max_age_ms"],
            )

        data = _panel_frame_producer(
            session,
            transform=transform,
            columns=spec.required_columns,
            resource_policy=resource_policy,
            asset_ids=("btc_usdt", "eth_usdt"),
        )

        def verify(_receipt):
            session.verify()
            readers.close()
            if source_identity(raw_root, spec.data_id) != expected_source_identity:
                raise ValueError("hft_external_source_identity_changed")

        with closing_iterator(data):
            yield data, metadata, verify

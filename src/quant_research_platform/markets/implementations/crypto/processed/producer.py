"""Bounded HFT market-fact producers over exact recorder/PANEL inputs."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from functools import partial
from itertools import chain

import pandas as pd

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.time import TimeWindow, is_datetime_window
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
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_DERIVATIVE_FACTOR_COLUMNS,
    HFT_FACTOR_COLUMNS,
    cross_asset_inputs,
    derivative_channels,
    external_channels,
    factor_catalog,
    label_catalog,
    spot_dependencies,
)
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.sources.binance import ids as binance_ids
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)

_EPOCH = dt.datetime(1970, 1, 1)
EXTERNAL_INSTRUMENTS = {
    "bybit": {
        "exchange": "bybit",
        "symbols": {"btc_usdt": "BTCUSDT", "eth_usdt": "ETHUSDT"},
        "base_units_per_amount": {"btc_usdt": 1.0, "eth_usdt": 1.0},
    },
    "okx": {
        "exchange": "okex-swap",
        "symbols": {"btc_usdt": "BTC-USDT-SWAP", "eth_usdt": "ETH-USDT-SWAP"},
        "base_units_per_amount": {
            "btc_usdt": {
                "before": 0.0001,
                "after": 0.01,
                "transition_start": "2020-03-20T08:00:00",
                "transition_end": "2020-03-20T08:30:00",
            },
            "eth_usdt": {
                "before": 0.001,
                "after": 0.1,
                "transition_start": "2020-03-19T08:00:00",
                "transition_end": "2020-03-19T08:30:00",
            },
        },
    },
}


@contextmanager
def produce_hft_processed_data(
    spec,
    *,
    inputs,
    time_window,
    processed_catalog,
    source_catalog,
    open_run,
    reconstruct_depth,
    read_trades,
    make_stats,
    raw_root=None,
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
            selected_assets = json.loads(spec.parameters_json).get("asset_ids")
            if selected_assets is not None:
                supported = {
                    canonical_crypto_asset_id(symbol): symbol for symbol in symbols
                }
                if not set(selected_assets) <= supported.keys():
                    raise ValueError("data_operation_asset_ids_invalid")
                symbols = tuple(supported[asset] for asset in selected_assets)
            # Keep all declared initialization history; discard only future archives.
            run = readers.enter_context(
                open_run(raw_root, time_window=TimeWindow(_EPOCH, time_window.end))
            )
            source_info = run.hft_source_info()
            start_ms, end_ms = (
                _milliseconds(time_window.start, ceil=True),
                _milliseconds(time_window.end),
            )
            market_metadata, unavailable = {}, {}
            for symbol in symbols:
                try:
                    market_metadata[symbol] = run.market_metadata(market_type, symbol)
                except PortfolioTargetBlocked as error:
                    if str(error) not in {
                        "tardis_price_observations_missing",
                        "depth_snapshot_missing",
                    }:
                        raise
                    unavailable[symbol] = str(error)
            if not market_metadata:
                raise PortfolioTargetBlocked("hft_source_assets_unavailable")
            stats = {symbol: make_stats() for symbol in symbols}
            batch_rows = DEFAULT_BATCH_ROWS
            symbol_rows = max(1, batch_rows // len(symbols))

            def symbol_frames(symbol):
                if symbol in unavailable:
                    return iter(())
                active = stats[symbol]
                return iter_panel_frames(
                    reconstruct_depth(
                        run,
                        market_type,
                        symbol=symbol,
                        stats=active,
                        required_levels=1,
                        output_levels=10,
                        end_ns=end_ms * 1000000,
                    ),
                    read_trades(
                        run, market_type, symbol=symbol, end_ns=end_ms * 1000000
                    ),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    book_coverage=active.coverage_segments,
                    trade_coverage=run.observation_coverage(
                        market_type=market_type, data_kind="agg_trades", symbol=symbol
                    ).covering,
                    market_id=str(MARKET_ID),
                    asset_id=canonical_crypto_asset_id(symbol),
                    batch_rows=symbol_rows,
                )

            def frames():
                source = _union_panel_frames(
                    (symbol_frames(symbol) for symbol in symbols),
                    columns=spec.required_columns,
                    batch_rows=batch_rows,
                )
                with closing_iterator(iter(source)) as active:
                    for frame in active:
                        try:
                            yield frame
                        except GeneratorExit:
                            return
                metadata["producer_diagnostics"] = {
                    "observation_grid": "independent_asset_observed_timestamps",
                    **source_info,
                    "requested_assets": list(symbols),
                    "available_assets": list(market_metadata),
                    "unavailable_assets": unavailable,
                    "reconstruction": {
                        symbol: asdict(value) for symbol, value in stats.items()
                    },
                }

            metadata["source_evidence"] = {
                **run.source_identity,
                "market_metadata": market_metadata,
            }
            data = frames()

        elif spec.data_id in {
            HFT_FACTOR_DATA_ID,
            HFT_LABEL_DATA_ID,
            HFT_INFORMATION_STATE_DATA_ID,
        }:
            input_id = spec.processed_input_data_ids[0]
            root = inputs[input_id]
            parameters = json.loads(spec.parameters_json)
            if spec.data_id == HFT_LABEL_DATA_ID:
                definitions = tuple(
                    item
                    for item in label_catalog().list()
                    if item.output_column in spec.required_columns
                )
                horizons = tuple(sorted({item.horizon for item in definitions}))
                transform = partial(
                    iter_label_frames,
                    horizons_ms=horizons,
                    columns=tuple(item.output_column for item in definitions),
                )
                computation_window = _computation_window(
                    time_window, future_ms=max(horizons)
                )
            else:
                factor_columns = tuple(
                    name for name in spec.required_columns if name in HFT_FACTOR_COLUMNS
                )
                definitions = tuple(
                    factor_catalog().get(name) for name in factor_columns
                )
                transform = partial(iter_factor_frames, columns=factor_columns)
                history_ms = max(
                    (
                        (max(item.required_windows, default=1) - 1) * HFT_GRID_MS
                        for item in definitions
                    ),
                    default=0,
                )
                computation_window = _computation_window(
                    time_window, history_ms=history_ms
                )
            inputs = tuple(
                dict.fromkeys(
                    (
                        "market_id",
                        "asset_id",
                        "timestamp",
                        "segment_id",
                        *(
                            column
                            for item in definitions
                            for column in item.input_columns
                            if column != "timestamp_ms"
                        ),
                    )
                )
            )
            session = _selected_context_session(root, inputs).filtered_view(
                computation_window
            )
            metadata["inputs"] = {input_id: str(session.root)}
            data = _panel_frame_producer(
                session,
                transform=transform,
                columns=spec.required_columns,
                asset_ids=tuple(parameters["asset_ids"]),
            )

        else:
            raise ValueError("hft_processed_target_invalid")
        data = _output_frames(data, time_window)
        with closing_iterator(data):
            yield (data, metadata)


def _computation_window(time_window, *, history_ms=0, future_ms=0):
    """Read only the finite mathematical warmup and maturity interval."""
    return TimeWindow(
        max(_EPOCH, time_window.start - dt.timedelta(milliseconds=history_ms)),
        time_window.end + dt.timedelta(milliseconds=future_ms),
    )


def _output_frames(frames, time_window):
    """Project after computation without resetting state at the output boundary."""
    with closing_iterator(iter(frames)) as active:
        for frame in active:
            selected = frame.loc[
                frame["timestamp"].between(time_window.start, time_window.end)
            ]
            if not selected.empty:
                yield selected


def _milliseconds(value: dt.datetime, *, ceil: bool = False) -> int:
    delta = value - _EPOCH
    if delta < dt.timedelta(0):
        raise ValueError("hft_time_window_before_epoch")
    unit = dt.timedelta(milliseconds=1)
    return -(-delta // unit) if ceil else delta // unit


def _panel_frame_producer(
    session: DatasetSession,
    *,
    transform: Callable[[Iterable[pd.DataFrame]], Iterable[pd.DataFrame]],
    columns: tuple[str, ...],
    asset_ids: tuple[str, ...],
) -> Iterator[object]:

    def frames() -> Iterator[pd.DataFrame]:
        for asset in asset_ids:
            asset_filter = ("asset_id", "==", asset)
            asset_session = replace(
                session,
                parquet_filters=(*(session.parquet_filters or ()), asset_filter),
            )
            chunks = iter_dataset_chunks(asset_session, batch_rows=DEFAULT_BATCH_ROWS)
            with closing_iterator(chunks):
                with closing_iterator(iter(transform(chunks))) as transformed:
                    for frame in transformed:
                        _yield_value = frame.loc[:, list(columns)]
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return

    with closing_iterator(frames()) as active:
        yield from active


@contextmanager
def produce_derivative_processed_data(
    spec,
    *,
    inputs,
    time_window,
    processed_catalog,
    open_archive,
    source_catalog,
    raw_root=None,
):
    """Align auxiliary states upstream of the unchanged exact-key assembler."""
    parameters = json.loads(spec.parameters_json)
    channels = derivative_channels(spec.required_columns)
    max_age_ms = parameters["max_age_ms"]
    if not set(channels) <= set(max_age_ms):
        raise ValueError("data_operation_max_age_ms_required")
    selected = tuple(
        name for name in spec.required_columns if name in HFT_DERIVATIVE_FACTOR_COLUMNS
    )
    root = inputs[HFT_PANEL_DATA_ID]
    with ExitStack() as readers:
        # State alignment can consume a prefix older than the output warmup.
        archive = readers.enter_context(
            open_archive(raw_root, time_window=TimeWindow(_EPOCH, time_window.end))
        )
        required = tuple(
            dict.fromkeys(
                (
                    "market_id",
                    "asset_id",
                    "timestamp",
                    "segment_id",
                    *(
                        column
                        for name in selected
                        for column in factor_catalog().get(name).input_columns
                    ),
                )
            )
        )
        history = max(
            (
                {
                    "open_interest_change_30s": 30_000,
                    "open_interest_change_1m": 60_000,
                    "open_interest_change_5m": 300_000,
                    "observed_force_order_imbalance_5s": 5000,
                    "observed_force_order_imbalance_30s": 30_000,
                }.get(name, 0)
                for name in selected
            ),
            default=0,
        )
        session = _selected_context_session(root, required).filtered_view(
            _computation_window(
                time_window, history_ms=history + parameters.get("input_delay_ms", 0)
            )
        )
        metadata = {
            "source_evidence": dict(archive.source_identity),
            "inputs": {HFT_PANEL_DATA_ID: str(session.root)},
            "producer_diagnostics": {
                "alignment": "observed_at_lte_decision_time",
                "max_age_ms": max_age_ms,
                "open_interest_unit": "base",
                "force_order_quantity": "observed_cumulative_filled_base_z",
                "force_order_coverage": "observed_proxy_not_complete_liquidation_tape",
            },
        }

        def transform(frames):
            first = next((frame for frame in frames if not frame.empty), None)
            if first is None:
                return
            symbol = {
                canonical_crypto_asset_id(value): value
                for value in binance_ids.RESEARCH_SYMBOLS
            }[first["asset_id"].iloc[0]]
            yield from iter_derivative_factor_frames(
                chain((first,), frames),
                observations=archive.iter_state_observations(
                    symbol=symbol, channels=channels
                ),
                columns=selected,
                max_age_ms=max_age_ms,
                input_delay_ms=json.loads(spec.parameters_json).get(
                    "input_delay_ms", 0
                ),
            )

        data = _panel_frame_producer(
            session,
            transform=transform,
            columns=spec.required_columns,
            asset_ids=tuple(
                canonical_crypto_asset_id(symbol)
                for symbol in binance_ids.RESEARCH_SYMBOLS
            ),
        )

        data = _output_frames(data, time_window)
        with closing_iterator(data):
            yield (data, metadata)


def _selected_context_session(root, columns):
    """Project the chosen definition while leaving absent values locally unknown."""
    options = _dataset_input_options(root)
    session = open_dataset_session(**{**options, "columns": None})
    keys = ("market_id", "asset_id", "timestamp")
    selected = options.get("columns")
    support = {*keys, "segment_id", "timestamp_ms", "observed_utc"}
    if selected is not None:
        # A supplied selector is configuration, not an observed missing value.
        session.projected_view(tuple(dict.fromkeys((*keys, *selected))))
        business = set(columns).intersection(session.arrow_schema.names) - support
        if not business <= set(selected):
            raise ValueError("data_operation_input_columns_missing")
    available = tuple(name for name in columns if name in session.arrow_schema.names)

    return session.projected_view(tuple(dict.fromkeys((*keys, *available))))


@contextmanager
def produce_spot_perp_context(
    spec, *, inputs, time_window, processed_catalog, source_catalog
):
    """Publish information rows on the execution panel's exact primary keys."""
    max_age_ms = json.loads(spec.parameters_json)["max_age_ms"]
    sessions = {}
    for data_id in (HFT_PANEL_DATA_ID, HFT_SPOT_PANEL_DATA_ID):
        root = inputs[data_id]
        dependencies, factors = spot_dependencies(
            spec.required_columns, perpetual=data_id == HFT_PANEL_DATA_ID
        )
        sessions[data_id] = _selected_context_session(
            root, (*dependencies, *(("segment_id",) if factors else ()))
        ).filtered_view(
            _computation_window(time_window, history_ms=500)
            if data_id == HFT_PANEL_DATA_ID
            else TimeWindow(_EPOCH, time_window.end)
        )
    metadata = {
        "inputs": {key: str(session.root) for key, session in sessions.items()},
        "producer_diagnostics": {
            "alignment": "spot_grid_available_by_perpetual_decision",
            "max_age_ms": max_age_ms,
            "lag_ms": 100,
            "execution_market": "binance_futures_btc_eth_only",
            "clock_interpretation": "vendor_observation_order_not_colocated_latency",
        },
    }
    batch_rows = DEFAULT_BATCH_ROWS

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
                )
                for key, session in sessions.items()
            }
            with closing_iterator(
                iter_dataset_chunks(
                    asset_sessions[HFT_PANEL_DATA_ID], batch_rows=DEFAULT_BATCH_ROWS
                )
            ) as perpetual:
                with closing_iterator(
                    iter_dataset_chunks(
                        asset_sessions[HFT_SPOT_PANEL_DATA_ID],
                        batch_rows=DEFAULT_BATCH_ROWS,
                    )
                ) as spot:
                    with closing_iterator(
                        iter_spot_perp_factor_frames(
                            perpetual,
                            spot_frames=spot,
                            columns=spec.required_columns,
                            max_age_ms=max_age_ms,
                            input_delay_ms=json.loads(spec.parameters_json).get(
                                "input_delay_ms", 0
                            ),
                            batch_rows=batch_rows,
                        )
                    ) as aligned:
                        for frame in aligned:
                            frame["market_id"] = str(MARKET_ID)
                            frame["asset_id"] = asset
                            _yield_value = frame.loc[:, list(spec.required_columns)]
                            try:
                                yield _yield_value
                            except GeneratorExit:
                                return

    data = frames()

    data = _output_frames(data, time_window)
    with closing_iterator(data):
        yield (data, metadata)


def _union_panel_frames(
    frame_streams: Iterable[Iterable[pd.DataFrame]],
    *,
    columns: tuple[str, ...],
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    """Merge sorted panel prefixes while retaining only bounded input batches."""
    columns = list(columns)

    def source_frames(frames):
        with closing_iterator(iter(frames)) as active:
            for frame in active:
                try:
                    yield frame
                except GeneratorExit:
                    return

    with ExitStack() as live:
        streams = [
            live.enter_context(closing_iterator(source_frames(frames)))
            for frames in frame_streams
        ]
        held = [None] * len(streams)
        exhausted = [False] * len(streams)
        empty = None
        output = []
        output_rows = 0
        emitted = False
        while True:
            for index, stream in enumerate(streams):
                if held[index] is None and not exhausted[index]:
                    for frame in stream:
                        if not frame.empty:
                            held[index] = frame.loc[:, columns]
                            break
                        empty = frame.loc[:, columns]
                    else:
                        exhausted[index] = True
            available = [
                (frame["timestamp"].iloc[-1], frame["asset_id"].iloc[-1], index)
                for index, frame in enumerate(held)
                if frame is not None
            ]
            if not available:
                break
            timestamp, asset, stream_rank = min(available)
            ready = []
            for index, frame in enumerate(held):
                if frame is None:
                    continue
                # Stream rank preserves heap-merge stability even when equal keys
                # continue into the next batch of an earlier input stream.
                same_time = frame["timestamp"] == timestamp
                before = (frame["timestamp"] < timestamp) | (
                    same_time & (frame["asset_id"] < asset)
                )
                if index <= stream_rank:
                    before |= same_time & (frame["asset_id"] == asset)
                count = int(before.sum())
                if count:
                    ready.append(frame.iloc[:count])
                    held[index] = frame.iloc[count:] if count < len(frame) else None
            merged = pd.concat(ready, ignore_index=True).sort_values(
                ["timestamp", "asset_id"], kind="stable", ignore_index=True
            )
            offset = 0
            while offset < len(merged):
                count = min(batch_rows - output_rows, len(merged) - offset)
                output.append(merged.iloc[offset : offset + count])
                output_rows += count
                offset += count
                if output_rows == batch_rows:
                    result = pd.concat(output, ignore_index=True)
                    output.clear()
                    output_rows = 0
                    emitted = True
                    try:
                        yield result
                    except GeneratorExit:
                        return
            del ready, merged
        if output or (not emitted and empty is not None):
            result = pd.concat(output, ignore_index=True) if output else empty
            try:
                yield result.reset_index(drop=True)
            except GeneratorExit:
                return


@contextmanager
def produce_information_panel(
    spec, *, inputs, time_window, processed_catalog, source_catalog
):
    """Union disjoint BTC/ETH and SOL rows, preserving native source segments."""
    sessions = []
    identities = {}
    for data_id in spec.processed_input_data_ids:
        root = inputs[data_id]
        options = _dataset_input_options(root)
        if options.get("columns") is not None and not set(
            spec.required_columns
        ).difference({"market_id", "asset_id", "timestamp", "segment_id"}) <= set(
            options["columns"]
        ):
            raise ValueError("data_operation_input_columns_missing")
        session = open_dataset_session(
            **{**options, "columns": spec.required_columns}
        ).filtered_view(time_window)
        sessions.append(session)
        identities[data_id] = str(session.root)
    batch_rows = DEFAULT_BATCH_ROWS
    frames = _union_panel_frames(
        (
            iter_dataset_chunks(session, batch_rows=DEFAULT_BATCH_ROWS)
            for session in sessions
        ),
        columns=spec.required_columns,
        batch_rows=batch_rows,
    )
    data = frames

    with closing_iterator(data):
        yield (data, {"inputs": identities})


@contextmanager
def produce_cross_asset_context(
    spec, *, inputs, time_window, processed_catalog, source_catalog
):
    """Read persisted shared-kernel information without creating an execution venue."""
    parameters = json.loads(spec.parameters_json)
    target_id, information_id = spec.processed_input_data_ids
    sessions = {}
    for data_id in spec.processed_input_data_ids:
        root = inputs[data_id]
        dependencies = (
            () if data_id == target_id else cross_asset_inputs(spec.required_columns)
        )
        sessions[data_id] = _selected_context_session(root, dependencies).filtered_view(
            time_window if data_id == target_id else TimeWindow(_EPOCH, time_window.end)
        )
        if (
            "segment_id" in dependencies
            and "segment_id" not in sessions[data_id].logical_columns
        ):
            raise ValueError("crypto_information_segment_required")
    metadata = {
        "inputs": {key: str(session.root) for key, session in sessions.items()},
        "producer_diagnostics": {
            "information_assets": parameters["information_assets"],
            "target_assets": parameters["target_assets"],
            "max_age_ms": parameters["max_age_ms"],
            "units": "mature_simple_return_and_dimensionless_order_flow",
            "execution_assets": ["btc_usdt", "eth_usdt"],
        },
    }
    batch_rows = DEFAULT_BATCH_ROWS

    def asset_chunks(session, asset):
        view = replace(
            session,
            parquet_filters=(
                *(session.parquet_filters or ()),
                ("asset_id", "==", asset),
            ),
        )
        yield from iter_dataset_chunks(view, batch_rows=DEFAULT_BATCH_ROWS)

    def frames():
        targets = _union_panel_frames(
            (
                asset_chunks(sessions[target_id], asset)
                for asset in parameters["target_assets"]
            ),
            columns=("market_id", "asset_id", "timestamp"),
            batch_rows=batch_rows,
        )
        with closing_iterator(
            iter_cross_asset_factor_frames(
                targets,
                information_frames={
                    key: asset_chunks(sessions[information_id], key)
                    for key in parameters["information_assets"]
                },
                max_age_ms=parameters["max_age_ms"],
                input_delay_ms=parameters.get("input_delay_ms", 0),
                batch_rows=batch_rows,
            )
        ) as aligned:
            for frame in aligned:
                try:
                    yield frame.loc[:, list(spec.required_columns)]
                except GeneratorExit:
                    return

    data = frames()

    data = _output_frames(data, time_window)
    with closing_iterator(data):
        yield (data, metadata)


@contextmanager
def produce_normalized_context(
    spec,
    *,
    inputs,
    time_window,
    processed_catalog,
    source_catalog,
    raw_root,
    read_records,
    source_schema_id,
    archives,
):
    """Publish source-verified external information on Binance BTC/ETH keys."""
    parameters = json.loads(spec.parameters_json)
    symbols = set(EXTERNAL_INSTRUMENTS[parameters["venue"]]["symbols"].values())
    if (
        not isinstance(archives, Mapping)
        or not set(archives) <= symbols
        or any(
            not isinstance(kinds, Mapping) or not set(kinds) <= {"quotes", "trades"}
            for kinds in archives.values()
        )
    ):
        raise ValueError("normalized_source_archives_invalid")
    root = inputs[HFT_PANEL_DATA_ID]
    computation_window = _computation_window(
        time_window,
        history_ms=500
        + (-(-parameters.get("input_delay_ms", 0) // HFT_GRID_MS)) * HFT_GRID_MS,
    )
    # Stale observed clocks remain meaningful after factor freshness expires.
    # Keep the source prefix, including prior-day observations and native units.
    source_window = TimeWindow(_EPOCH, time_window.end)
    selected_channels = external_channels(spec.required_columns)
    needs_mid = any(
        name.endswith(("_price_difference", "_relative_mid_dispersion"))
        for name in spec.required_columns
    )
    needs_history = any(
        name.removeprefix(parameters["venue"] + "_").startswith(
            ("return_", "observed_")
        )
        for name in spec.required_columns
    )
    session = _selected_context_session(
        root,
        (*(("segment_id",) if needs_history else ()), *(("mid",) if needs_mid else ())),
    ).filtered_view(computation_window)
    metadata = {
        "source_evidence": {
            "schema_id": source_schema_id,
            "artifact_id": spec.data_id,
        },
        "inputs": {HFT_PANEL_DATA_ID: str(session.root)},
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
        first = next((frame for frame in frames if not frame.empty), None)
        if first is None:
            return
        asset = first["asset_id"].iloc[0]
        symbol = parameters["symbols"][asset]
        yield from iter_external_context_frames(
            chain((first,), frames),
            columns=spec.required_columns,
            quotes=sources.get(symbol, {}).get("quotes", ()),
            trades=sources.get(symbol, {}).get("trades", ()),
            venue=parameters["venue"],
            base_units_per_amount=parameters["base_units_per_amount"][asset],
            max_age_ms=parameters["max_age_ms"],
            input_delay_ms=parameters.get("input_delay_ms", 0),
        )

    identities = {
        symbol: {kind: [] for kind in kinds if kind in selected_channels}
        for symbol, kinds in archives.items()
    }
    with closing_iterator(ExitStack()) as readers:
        sources = {
            symbol: {
                kind: readers.enter_context(
                    closing_iterator(
                        read_records(
                            raw_root,
                            archive_paths=paths,
                            time_window=source_window,
                            expected_scope=(
                                EXTERNAL_INSTRUMENTS[parameters["venue"]]["exchange"],
                                symbol,
                                kind,
                            ),
                            on_source=identities[symbol][kind].append,
                        )
                    )
                )
                for kind, paths in kinds.items()
                if kind in selected_channels
            }
            for symbol, kinds in archives.items()
        }

        def frames():
            data = _panel_frame_producer(
                session,
                transform=transform,
                columns=spec.required_columns,
                asset_ids=("btc_usdt", "eth_usdt"),
            )
            with closing_iterator(_output_frames(data, time_window)) as output:
                yield from output
            metadata["source_evidence"]["manifest_hash"] = canonical_json_hash(
                identities
            )

        with closing_iterator(frames()) as data:
            yield (data, metadata)

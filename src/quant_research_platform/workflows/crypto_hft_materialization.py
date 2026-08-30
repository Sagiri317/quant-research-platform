"""Bounded Binance HFT processed-data materialization."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Iterator

import pandas as pd

from quant_research_platform.contracts.time import (
    TimeWindow,
    is_intraday_time_window,
)
from quant_research_platform.markets.implementations.crypto.hft import (
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
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    HFT_LABEL_HORIZONS_MS,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_HFT_RECORDING_SCOPE,
    FUTURES_MARKET_TYPE,
)
from quant_research_platform.sources.binance.replay import (
    ReconstructionStats,
    RecorderRun,
    iter_agg_trades,
    iter_reconstructed_depth,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import dataset_identity_payload
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout

_PANEL_ROW_BUFFER_BYTES = 4 * 1024
_FRAME_FIXED_BYTES = 1024 * 1024
_EPOCH = dt.datetime(1970, 1, 1)


def materialize_hft_processed_data(
    *,
    layout: StorageLayout,
    run: RecorderRun,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> tuple[dict[str, DatasetInspection], ReconstructionStats]:
    """Write PANEL once, then derive FACTOR and LABEL from that persisted PANEL."""

    if not is_intraday_time_window(time_window):
        raise ValueError("hft_materialization_requires_intraday_window")
    if run.manifest["market_data_status"] == "FAIL":
        raise ValueError("hft_recorder_run_failed")
    if run.manifest["recording_scope"] != FUTURES_HFT_RECORDING_SCOPE:
        raise ValueError("hft_futures_run_missing")
    if FUTURES_MARKET_TYPE not in run.manifest["market_types"]:
        raise ValueError("hft_futures_run_missing")
    start_ms = _milliseconds(time_window.start)
    end_ms = _milliseconds(time_window.end)
    asset_id = canonical_crypto_asset_id(run.manifest["symbol"])
    catalog = hft_processed_catalog()
    stats = ReconstructionStats()
    inspections: dict[str, DatasetInspection] = {}
    with runtime_resource_policy_scope(resource_policy) as policy:
        panel_spec = catalog.get(HFT_PANEL_DATA_ID)
        panel_target = policy.operation_headroom_bytes // 2
        if panel_target <= _FRAME_FIXED_BYTES + _PANEL_ROW_BUFFER_BYTES:
            raise MemoryError("runtime_memory_budget_exceeded")
        panel_batch_rows = (
            panel_target - _FRAME_FIXED_BYTES
        ) // _PANEL_ROW_BUFFER_BYTES
        panel_producer_bound = (
            _FRAME_FIXED_BYTES + panel_batch_rows * _PANEL_ROW_BUFFER_BYTES
        )
        panel_frames = iter_panel_frames(
            iter_reconstructed_depth(
                run,
                FUTURES_MARKET_TYPE,
                symbol="BTCUSDT",
                stats=stats,
            ),
            iter_agg_trades(run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", end_ms=end_ms),
            start_ms=start_ms,
            end_ms=end_ms,
            book_coverage=stats.coverage_segments,
            trade_coverage=run.connection_coverage(
                market_type=FUTURES_MARKET_TYPE,
                data_kind="agg_trades",
            ),
            market_id=str(MARKET_ID),
            asset_id=asset_id,
            batch_rows=panel_batch_rows,
        )
        panel_root = layout.processed_data_root(
            panel_spec.market_id,
            panel_spec.kind,
            panel_spec.data_id,
        )
        inspections[HFT_PANEL_DATA_ID] = write_dataset(
            panel_root,
            _event_frame_producer(
                panel_frames,
                columns=panel_spec.required_columns,
                producer_bound=panel_producer_bound,
            ),
            spec=panel_spec,
            instance_metadata={
                "source_artifact_identity": run.source_identity,
            },
            resource_policy=policy,
        )
        with open_dataset_session(
            panel_root,
            columns=None,
            time_window=time_window,
            resource_policy=policy,
        ) as panel_session:

            def verify_panel(_receipt: object) -> None:
                panel_session.verify()

            panel_identity = dataset_identity_payload(
                panel_session.spec,
                panel_session.manifest,
            )
            input_metadata = {
                "input_dataset_identities": {
                    HFT_PANEL_DATA_ID: panel_identity,
                }
            }
            factor_spec = catalog.get(HFT_FACTOR_DATA_ID)
            inspections[HFT_FACTOR_DATA_ID] = write_dataset(
                layout.processed_data_root(
                    factor_spec.market_id,
                    factor_spec.kind,
                    factor_spec.data_id,
                ),
                _panel_frame_producer(
                    panel_session,
                    transform=iter_factor_frames,
                    columns=factor_spec.required_columns,
                    resource_policy=policy,
                ),
                spec=factor_spec,
                instance_metadata=input_metadata,
                staged_validator=verify_panel,
                resource_policy=policy,
            )
            label_spec = catalog.get(HFT_LABEL_DATA_ID)
            inspections[HFT_LABEL_DATA_ID] = write_dataset(
                layout.processed_data_root(
                    label_spec.market_id,
                    label_spec.kind,
                    label_spec.data_id,
                ),
                _panel_frame_producer(
                    panel_session,
                    transform=lambda frames: iter_label_frames(
                        frames,
                        horizons_ms=HFT_LABEL_HORIZONS_MS,
                    ),
                    columns=label_spec.required_columns,
                    resource_policy=policy,
                ),
                spec=label_spec,
                instance_metadata=input_metadata,
                staged_validator=verify_panel,
                resource_policy=policy,
            )
            panel_session.verify()
    return inspections, stats


def _milliseconds(value: dt.datetime) -> int:
    delta = value - _EPOCH
    milliseconds = delta // dt.timedelta(milliseconds=1)
    if milliseconds < 0:
        raise ValueError("hft_time_window_before_epoch")
    return milliseconds


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
    chunks = iter_dataset_chunks(session, target_chunk_bytes=target_chunk_bytes)
    frames = iter(transform(chunks))
    with closing_iterator(chunks), closing_iterator(frames):
        while True:
            producer_bound = resource_policy.operation_headroom_bytes // 2
            if producer_bound <= _FRAME_FIXED_BYTES:
                raise MemoryError("runtime_memory_budget_exceeded")
            lease = yield producer_bound
            try:
                frame = next(frames)
            except StopIteration:
                return
            output = frame.loc[:, list(columns)]
            actual_bound = pandas_frame_payload_bytes(output) + 64 * 1024
            if actual_bound > lease.reserved_bytes:
                raise ValueError("hft_dataframe_producer_bound_invalid")
            lease.shrink_to(actual_bound)
            yield OwnedValue(output, lease)
            del frame, output

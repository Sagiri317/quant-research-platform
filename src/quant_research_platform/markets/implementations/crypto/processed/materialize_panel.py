"""Crypto market panel materialization."""

from __future__ import annotations

from collections.abc import Hashable, Iterator

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.markets.implementations.crypto.ids import (
    canonical_crypto_asset_id,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_resources,
    materialization_validation,
    panel_input_reducers,
    panels,
    source_ids,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_reading import (
    build_dataset_equality_group_plan,
    build_dataset_read_context,
    dataset_context_for_equality_group,
    iter_dataset_chunks,
)
from quant_research_platform.storage._dataset_writing import write_dataset
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


def write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with runtime_resource_policy_scope(resource_policy):
        input_identities = materialization_validation.input_dataset_identities(
            materialization_validation.panel_input_inspections(
                layout=layout,
                source_id=source_id,
                spec=spec,
            )
        )
        chunks = materialization_resources.iter_disk_guarded_chunks(
            _iter_panel_chunks(
                spec,
                layout=layout,
                source_id=source_id,
                time_window=time_window,
                resource_policy=resource_policy,
            ),
            scratch_parent=root.parent,
        )

        def validate_staged_dataset(staged_root) -> None:
            staged_inspection = inspect_dataset(staged_root, time_window=time_window)
            materialization_validation.validate_panel_materialization(
                staged_inspection,
                layout=layout,
                source_id=source_id,
                spec=spec,
                time_window=time_window,
                expected_input_identities=input_identities,
                resource_policy=resource_policy,
                scratch_parent=staged_root,
            )

        write_dataset(
            root,
            chunks,
            spec=spec,
            instance_metadata={"input_dataset_identities": input_identities},
            staged_validator=validate_staged_dataset,
            resource_policy=resource_policy,
        )
        return inspect_dataset(root, time_window=time_window)


def _iter_panel_chunks(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    chunk_target_bytes = resource_policy.payload_target_bytes
    bar_root = layout.interim_data_root(
        source_id,
        source_ids.market_bars_source_data_id(spec),
    )
    trade_root = layout.interim_data_root(
        source_id,
        source_ids.aggregate_trades_source_data_id(spec),
    )
    depth_root = layout.interim_data_root(
        source_id,
        source_ids.depth_levels_source_data_id(spec),
    )
    bar_context = build_dataset_read_context(
        bar_root,
        columns=None,
        time_window=time_window,
    )
    group_plan = build_dataset_equality_group_plan(
        bar_context,
        columns=("source_asset_id",),
        target_chunk_bytes=chunk_target_bytes,
    )
    materialization_validation.validate_single_asset_part_grouping(
        group_plan,
        error="invalid_crypto_panel_source_physical_asset_grouping",
    )
    source_assets = _canonical_source_assets(group_plan.keys)
    order_state = materialization_validation.OrderedDependencyState()
    for source_asset_id in source_assets:
        trade_continuity_state = (
            panel_input_reducers.AggregateTradeContinuityState()
        )
        group_context = dataset_context_for_equality_group(
            bar_context,
            group_plan,
            (source_asset_id,),
        )
        for bars in iter_dataset_chunks(
            group_context,
            target_chunk_bytes=chunk_target_bytes,
        ):
            if bars.empty:
                continue
            _validate_matching_bar_keys(bars)
            prepared_bars = panels.prepare_panel_bars(bars)
            prepared_bars = (
                materialization_validation.validated_ordered_dependency_chunk(
                    prepared_bars,
                    state=order_state,
                    error="out_of_order_crypto_panel_dependency",
                )
            )
            materialization_resources.require_frame_within_budget(
                prepared_bars,
                target_bytes=resource_policy.payload_target_bytes,
                error="crypto_indivisible_reducer_unit_exceeds_payload_target",
            )
            trade_chunks = _iter_matching_trade_chunks(
                trade_root,
                prepared_bars,
                target_chunk_bytes=chunk_target_bytes,
            )
            depth_chunks = _iter_matching_depth_chunks(
                depth_root,
                prepared_bars,
                target_chunk_bytes=chunk_target_bytes,
            )
            output = panels.calculate_market_panel_from_chunks(
                prepared_bars,
                trade_chunks=trade_chunks,
                depth_chunks=depth_chunks,
                payload_target_bytes=resource_policy.payload_target_bytes,
                trade_continuity_state=trade_continuity_state,
                max_quote_age=panels.DEFAULT_MAX_QUOTE_AGE,
            ).loc[:, spec.required_columns]
            materialization_resources.require_frame_within_budget(
                output,
                target_bytes=resource_policy.payload_target_bytes,
                error="crypto_indivisible_reducer_unit_exceeds_payload_target",
            )
            yield output


def _canonical_source_assets(
    keys: tuple[tuple[object, ...], ...],
) -> tuple[str, ...]:
    source_by_canonical: dict[str, str] = {}
    for key in keys:
        if len(key) != 1 or not isinstance(key[0], str) or not key[0]:
            raise ValueError("invalid_crypto_matching_bar_source_asset_id")
        source_value = key[0]
        canonical = canonical_crypto_asset_id(source_value)
        existing = source_by_canonical.get(canonical)
        if existing is not None:
            if existing != source_value:
                raise ValueError("duplicate_crypto_canonical_asset_id")
            continue
        source_by_canonical[canonical] = source_value
    return tuple(
        source_by_canonical[canonical]
        for canonical in sorted(source_by_canonical)
    )


def _iter_matching_trade_chunks(
    dataset_root,
    bars: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    asset_ids, start, end, read_window = _matching_read_scope(
        bars,
        max_quote_age=pd.Timedelta(0),
    )
    yield from _iter_matching_chunks(
        dataset_root,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=read_window,
        max_quote_age=pd.Timedelta(0),
        source_asset_error="invalid_crypto_matching_trade_source_asset_id",
        timestamp_error="invalid_crypto_matching_trade_timestamp",
        target_chunk_bytes=target_chunk_bytes,
    )


def _iter_matching_depth_chunks(
    dataset_root,
    bars: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    max_quote_age = panels.DEFAULT_MAX_QUOTE_AGE
    asset_ids, start, end, read_window = _matching_read_scope(
        bars,
        max_quote_age=max_quote_age,
    )
    yield from _iter_matching_chunks(
        dataset_root,
        asset_ids=asset_ids,
        start=start,
        end=end,
        time_window=read_window,
        max_quote_age=max_quote_age,
        source_asset_error="invalid_crypto_matching_depth_source_asset_id",
        timestamp_error="invalid_crypto_matching_depth_timestamp",
        target_chunk_bytes=target_chunk_bytes,
    )


def _matching_read_scope(
    bars: pd.DataFrame,
    *,
    max_quote_age: pd.Timedelta,
) -> tuple[set[str], pd.Timestamp, pd.Timestamp, TimeWindow]:
    asset_ids = set(
        _strict_source_asset_ids(
            bars,
            error="invalid_crypto_matching_bar_source_asset_id",
        )
    )
    start = strict_timestamp_series(
        bars["timestamp"],
        error="invalid_crypto_matching_bar_timestamp",
    ).min()
    end = strict_timestamp_series(
        bars["close_time"],
        error="invalid_crypto_matching_bar_close_time",
    ).max()
    return (
        asset_ids,
        start,
        end,
        TimeWindow(
            (start - max_quote_age).to_pydatetime(),
            end.to_pydatetime(),
        ),
    )


def _validate_matching_bar_keys(bars: pd.DataFrame) -> None:
    _strict_source_asset_ids(
        bars,
        error="invalid_crypto_matching_bar_source_asset_id",
    )
    strict_timestamp_series(
        bars["timestamp"],
        error="invalid_crypto_matching_bar_timestamp",
    )
    strict_timestamp_series(
        bars["close_time"],
        error="invalid_crypto_matching_bar_close_time",
    )


def _iter_matching_chunks(
    dataset_root,
    *,
    asset_ids: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    time_window: TimeWindow,
    max_quote_age: pd.Timedelta,
    source_asset_error: str,
    timestamp_error: str,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    context = build_dataset_read_context(
        dataset_root,
        columns=None,
        time_window=time_window,
        row_filters=(
            (("source_asset_id", "==", next(iter(asset_ids))),)
            if len(asset_ids) == 1
            else ()
        ),
    )
    for chunk in iter_dataset_chunks(
        context,
        target_chunk_bytes=target_chunk_bytes,
    ):
        if chunk.empty:
            continue
        matching = chunk.loc[
            _matching_window_mask(
                chunk,
                asset_ids=asset_ids,
                start=start,
                end=end,
                max_quote_age=max_quote_age,
                source_asset_error=source_asset_error,
                timestamp_error=timestamp_error,
            )
        ].copy()
        if not matching.empty:
            yield matching


def _matching_window_mask(
    frame: pd.DataFrame,
    *,
    asset_ids: set[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_quote_age: pd.Timedelta,
    source_asset_error: str,
    timestamp_error: str,
) -> pd.Series:
    timestamps = strict_timestamp_series(frame["timestamp"], error=timestamp_error)
    source_ids = _strict_source_asset_ids(frame, error=source_asset_error)
    mask = source_ids.isin(asset_ids)
    return mask & timestamps.ge(start - max_quote_age) & timestamps.le(end)


def _strict_source_asset_ids(frame: pd.DataFrame, *, error: str) -> pd.Series:
    if "source_asset_id" not in frame.columns:
        raise ValueError(error)
    values = frame["source_asset_id"]
    for value in values:
        if (
            value is None
            or isinstance(value, (list, dict, set))
            or not isinstance(value, Hashable)
            or not isinstance(value, str)
            or value == ""
        ):
            raise ValueError(error)
    return values


__all__ = ["write_panel"]

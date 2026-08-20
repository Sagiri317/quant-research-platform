"""A-share processed panel materialization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import ExitStack

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io as materialization_io,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_planning as materialization_planning,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_validation as materialization_validation,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    panels as panels,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_inspection import (
    dataset_inspection_retained_memory_bytes,
    inspect_dataset,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
)
from quant_research_platform.storage.layout import StorageLayout


def write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    market_bars_data_id: str,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: pd.DatetimeIndex,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    retained_written_bytes = sum(
        dataset_inspection_retained_memory_bytes(inspection)
        for inspection in written.values()
    )
    market_bars_root = layout.interim_data_root(source_id, market_bars_data_id)
    market_bars_inspection = inspect_dataset(
        market_bars_root,
        resource_policy=policy,
        retained_payload_bytes=retained_written_bytes,
    )
    materialization_validation.require_source_input_columns_from_schema(
        market_bars_data_id,
        market_bars_inspection.columns,
        columns=("open", "high", "low", "close", "volume", "amount"),
    )
    del market_bars_inspection
    market_bars_row_buffer = (
        materialization_planning.registered_output_buffer_bytes(
            spec,
            row_count=1,
        )
    )
    source_join_row_buffer = market_bars_row_buffer
    output_row_buffer = market_bars_row_buffer
    merge_index_row_buffer = 3 * 8
    operation_row_bound = (
        market_bars_row_buffer
        + source_join_row_buffer
        + output_row_buffer
        + merge_index_row_buffer
    )

    def chunks() -> Iterable[object]:
        def materialize(
            asset_ids: tuple[object, ...],
            current_sessions: Sequence[pd.Timestamp],
            session_context: Sequence[pd.Timestamp],
            upstream_retained_bytes: int,
        ) -> pd.DataFrame:
            with ExitStack() as live:
                read_window = materialization_planning.session_batch_window(
                    session_context,
                    current_sessions,
                    lookahead_sessions=1,
                )
                source_inputs = materialization_io.read_source_inputs_for_spec(
                    spec,
                    layout=layout,
                    source_id=source_id,
                    market_bars_data_id=market_bars_data_id,
                    columns_by_data_id={
                        "trading_calendar": (
                            "calendar_date",
                            "source_market_type",
                            "is_trading_day",
                        )
                    },
                    asset_ids_by_data_id={
                        "stock_master": asset_ids,
                        "market_cap": asset_ids,
                    },
                    asset_columns_by_data_id={
                        "stock_master": "source_asset_id",
                        "market_cap": "source_asset_id",
                    },
                    retained_payload_bytes=upstream_retained_bytes,
                    resource_policy=policy,
                )
                materialization_validation.require_source_inputs_for_spec(
                    spec, source_inputs
                )
                source_bytes = sum(
                    pandas_frame_payload_bytes(frame)
                    for frame in source_inputs.values()
                )
                live.enter_context(
                    policy.reserve("a_share.panel.source_inputs", source_bytes)
                )
                stock_master = source_inputs["stock_master"]
                market_cap = source_inputs["market_cap"]
                trading_calendar = source_inputs["trading_calendar"]
                market_bars = materialization_io.read_materialized_dataset(
                    market_bars_root,
                    time_window=read_window,
                    resource_policy=policy,
                    retained_payload_bytes=upstream_retained_bytes,
                    asset_ids=asset_ids,
                    asset_column="source_asset_id",
                )
                materialization_validation.require_source_input_columns(
                    market_bars_data_id,
                    market_bars,
                    columns=(
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                    ),
                )
                market_bars_bytes = pandas_frame_payload_bytes(market_bars)
                live.enter_context(
                    policy.reserve("a_share.panel.market_bars", market_bars_bytes)
                )
                output_buffer = (
                    materialization_planning.registered_output_buffer_bytes(
                        spec,
                        row_count=len(market_bars),
                    )
                )
                merge_state = source_bytes + market_bars_bytes
                panel_bound = output_buffer + merge_state
                live.enter_context(
                    policy.reserve("a_share.panel.calculate", panel_bound)
                )
                panel = panels.compute_market_panel(
                    market_bars,
                    stock_master=stock_master,
                    market_cap=market_cap,
                    trading_calendar=trading_calendar,
                )
                if pandas_frame_payload_bytes(panel) > panel_bound:
                    raise materialization_io.MaterializationCapacityExceeded(
                        "a_share_known_payloads_exceed_task_memory_budget"
                    )
                selected = materialization_planning.filter_output_sessions(
                    panel,
                    current_sessions,
                    columns=spec.required_columns,
                    retained_frames=(*source_inputs.values(), market_bars),
                    retained_payload_bytes=upstream_retained_bytes,
                    resource_policy=policy,
                )
                return selected

        yield from materialization_planning.iter_active_materialization_batches(
            market_bars_root,
            asset_column="source_asset_id",
            time_window=time_window,
            sessions=sessions,
            resource_policy=policy,
            output_row_bound_bytes=operation_row_bound,
            retained_payload_bytes=retained_written_bytes,
            materialize=materialize,
        )

    return materialization_io.write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
        resource_policy=policy,
    )


__all__ = ["write_panel"]

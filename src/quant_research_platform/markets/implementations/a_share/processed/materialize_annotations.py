"""A-share processed annotation materialization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import ExitStack

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.fundamentals import (
    FINANCIAL_TABLE_CODE_BY_DATA_ID,
    financial_events_from_interim,
    materialize_pit_fundamentals,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
    MARKET_ID,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    annotations as annotations,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io as materialization_io,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_planning as materialization_planning,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_validation as materialization_validation,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout


def write_annotations(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId | None = None,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: pd.DatetimeIndex,
    written: dict[str, DatasetInspection],
    cleanup_stack: ExitStack | None = None,
) -> DatasetInspection:
    if spec.data_id == FUNDAMENTAL_PIT_PROCESSED_DATA_ID:
        if source_id is None:
            raise ValueError("fundamental_pit_source_id_required")
        return _write_fundamental_pit(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            cleanup_stack=cleanup_stack,
        )
    catalog = processed_catalog()
    contract = materialization_validation.annotation_input_contract(spec)
    materialization_validation.validate_processed_input_contract(
        spec,
        contract=contract,
        purpose="annotation",
    )
    panel_spec = catalog.get(contract.primary_input_data_id)
    panel_root = materialization_io.processed_root(layout, panel_spec)
    input_row_buffer = materialization_planning.registered_output_buffer_bytes(
        panel_spec,
        row_count=1,
    )
    output_row_buffer = materialization_planning.registered_output_buffer_bytes(
        spec,
        row_count=1,
    )
    annotation_row_vectors = len(spec.required_columns) * 8
    operation_row_bound = input_row_buffer + output_row_buffer + annotation_row_vectors

    def chunks() -> Iterable[object]:
        def materialize(
            asset_ids: tuple[object, ...],
            current_sessions: Sequence[pd.Timestamp],
            session_context: Sequence[pd.Timestamp],
        ) -> pd.DataFrame:
            with ExitStack() as live:
                read_window = materialization_planning.session_batch_window(
                    session_context,
                    current_sessions,
                    lookahead_sessions=1,
                )
                processed_inputs = materialization_io.read_processed_inputs_for_spec(
                    spec,
                    layout=layout,
                    processed_catalog=catalog,
                    time_window=read_window,
                    resource_policy=policy,
                    asset_ids=asset_ids,
                    cleanup_stack=live,
                )
                materialization_validation.validate_processed_input_contract(
                    spec,
                    contract=contract,
                    purpose="annotation",
                    processed_inputs=processed_inputs,
                )
                panel = processed_inputs[contract.primary_input_data_id]
                panel_bytes = pandas_frame_payload_bytes(panel)
                output_buffer = materialization_planning.registered_output_buffer_bytes(
                    spec,
                    row_count=len(panel),
                )
                annotation_bound = panel_bytes + output_buffer
                live.enter_context(
                    policy.lease(
                        "a_share.annotation.calculate",
                        annotation_bound,
                    )
                )
                outputs = annotations.derive_annotations(
                    panel,
                    data_ids=(spec.data_id,),
                )
                derived = outputs[spec.data_id]
                if pandas_frame_payload_bytes(derived) > annotation_bound:
                    raise materialization_io.MaterializationCapacityExceeded(
                        "a_share_known_payloads_exceed_task_memory_budget"
                    )
                output = materialization_planning.filter_output_sessions(
                    derived,
                    current_sessions,
                    columns=spec.required_columns,
                    resource_policy=policy,
                )
                return output

        yield from materialization_planning.iter_active_materialization_batches(
            panel_root,
            time_window=time_window,
            sessions=sessions,
            resource_policy=policy,
            output_row_bound_bytes=operation_row_bound,
            materialize=materialize,
        )

    return materialization_io.write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
        resource_policy=policy,
        cleanup_stack=cleanup_stack,
    )


def _write_fundamental_pit(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: pd.DatetimeIndex,
    written: dict[str, DatasetInspection],
    cleanup_stack: ExitStack | None,
) -> DatasetInspection:
    market_bars_data_id = spec.source_input_data_ids[0]
    market_bars_root = layout.interim_data_root(source_id, market_bars_data_id)
    output_row_bound = (
        materialization_planning.registered_output_buffer_bytes(spec, row_count=1)
        + len(spec.required_columns) * 8
    )

    def chunks() -> Iterable[object]:
        def materialize(
            asset_ids: tuple[object, ...],
            current_sessions: Sequence[pd.Timestamp],
            _session_context: Sequence[pd.Timestamp],
        ) -> pd.DataFrame:
            batch_window = TimeWindow(
                current_sessions[0].date(), current_sessions[-1].date()
            )
            with ExitStack() as live:
                asset_filters = {
                    data_id: asset_ids
                    for data_id in (
                        market_bars_data_id,
                        *FINANCIAL_TABLE_CODE_BY_DATA_ID,
                        "financial_disclosures",
                    )
                }
                inputs = materialization_io.read_source_inputs_for_spec(
                    spec,
                    layout=layout,
                    source_id=source_id,
                    columns_by_data_id={
                        market_bars_data_id: (
                            "source_asset_id",
                            "timestamp",
                            "total_market_cap",
                        ),
                        "trading_calendar": (
                            "timestamp",
                            "calendar_date",
                            "source_market_type",
                            "is_trading_day",
                        ),
                    },
                    time_windows_by_data_id={market_bars_data_id: batch_window},
                    asset_ids_by_data_id=asset_filters,
                    asset_columns_by_data_id={
                        data_id: "source_asset_id" for data_id in asset_filters
                    },
                    resource_policy=policy,
                    cleanup_stack=live,
                )
                input_bytes = sum(
                    pandas_frame_payload_bytes(frame) for frame in inputs.values()
                )
                output_bytes = output_row_bound * len(asset_ids) * len(current_sessions)
                live.enter_context(
                    policy.lease(
                        "a_share.fundamental_pit.calculate",
                        6 * input_bytes + output_bytes + 64 * 1024,
                    )
                )
                disclosures = inputs["financial_disclosures"].rename(
                    columns={"source_asset_id": "asset_id"}
                )
                events = financial_events_from_interim(
                    {
                        data_id: inputs[data_id]
                        for data_id in FINANCIAL_TABLE_CODE_BY_DATA_ID
                    },
                    disclosures.loc[
                        :,
                        (
                            "asset_id",
                            "table_id",
                            "report_period",
                            "first_disclosure_date",
                        ),
                    ],
                    trading_sessions=unified_trading_sessions(
                        inputs["trading_calendar"]
                    ),
                )
                observations = pd.MultiIndex.from_product(
                    (asset_ids, current_sessions), names=("asset_id", "timestamp")
                ).to_frame(index=False)
                pit = materialize_pit_fundamentals(observations, events)
                facts = pit.pivot_table(
                    index=("asset_id", "timestamp"),
                    columns="metric_id",
                    values="value",
                    aggfunc="first",
                ).reset_index()
                prior = pit.loc[
                    pit["metric_id"].eq("total_assets"),
                    ("asset_id", "timestamp", "previous_value"),
                ].rename(columns={"previous_value": "previous_total_assets"})
                market_cap = inputs[market_bars_data_id].rename(
                    columns={
                        "source_asset_id": "asset_id",
                        "total_market_cap": "market_cap",
                    }
                )
                output = (
                    observations.merge(
                        facts,
                        on=("asset_id", "timestamp"),
                        how="left",
                        validate="one_to_one",
                    )
                    .merge(
                        prior,
                        on=("asset_id", "timestamp"),
                        how="left",
                        validate="one_to_one",
                    )
                    .merge(
                        market_cap.loc[:, ("asset_id", "timestamp", "market_cap")],
                        on=("asset_id", "timestamp"),
                        how="left",
                        validate="one_to_one",
                    )
                )
                output.insert(0, "market_id", str(MARKET_ID))
                return output.loc[:, spec.required_columns].sort_values(
                    list(spec.primary_key), kind="mergesort"
                )

        yield from materialization_planning.iter_active_materialization_batches(
            market_bars_root,
            asset_column="source_asset_id",
            time_window=time_window,
            sessions=sessions,
            resource_policy=policy,
            output_row_bound_bytes=output_row_bound,
            materialize=materialize,
        )

    return materialization_io.write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
        resource_policy=policy,
        cleanup_stack=cleanup_stack,
    )

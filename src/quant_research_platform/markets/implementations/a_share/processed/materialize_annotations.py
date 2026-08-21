"""A-share processed annotation materialization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import ExitStack

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
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
from quant_research_platform.storage._dataset_inspection import (
    dataset_inspection_retained_memory_bytes,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout


def write_annotations(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: pd.DatetimeIndex,
    written: dict[str, DatasetInspection],
    cleanup_stack: ExitStack | None = None,
) -> DatasetInspection:
    retained_written_bytes = sum(
        dataset_inspection_retained_memory_bytes(inspection)
        for inspection in written.values()
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
    operation_row_bound = (
        input_row_buffer + output_row_buffer + annotation_row_vectors
    )

    def chunks() -> Iterable[object]:
        def materialize(
            asset_ids: tuple[object, ...],
            current_sessions: Sequence[pd.Timestamp],
            session_context: Sequence[pd.Timestamp],
            retained_payload_bytes: int,
        ) -> pd.DataFrame:
            with ExitStack() as live:
                read_window = materialization_planning.session_batch_window(
                    session_context,
                    current_sessions,
                    lookahead_sessions=1,
                )
                processed_inputs = (
                    materialization_io.read_processed_inputs_for_spec(
                        spec,
                        layout=layout,
                        processed_catalog=catalog,
                        time_window=read_window,
                        resource_policy=policy,
                        asset_ids=asset_ids,
                        retained_payload_bytes=retained_payload_bytes,
                    )
                )
                materialization_validation.validate_processed_input_contract(
                    spec,
                    contract=contract,
                    purpose="annotation",
                    processed_inputs=processed_inputs,
                )
                panel = processed_inputs[contract.primary_input_data_id]
                panel_bytes = pandas_frame_payload_bytes(panel)
                live.enter_context(
                    policy.reserve("a_share.annotation.input", panel_bytes)
                )
                output_buffer = (
                    materialization_planning.registered_output_buffer_bytes(
                        spec,
                        row_count=len(panel),
                    )
                )
                annotation_bound = panel_bytes + output_buffer
                live.enter_context(
                    policy.reserve(
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
                    retained_frames=(panel, *outputs.values()),
                    retained_payload_bytes=retained_payload_bytes,
                    resource_policy=policy,
                )
                return output

        yield from materialization_planning.iter_active_materialization_batches(
            panel_root,
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
        cleanup_stack=cleanup_stack,
    )


__all__ = ["write_annotations"]

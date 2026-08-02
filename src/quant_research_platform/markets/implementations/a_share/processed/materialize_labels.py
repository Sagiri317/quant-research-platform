"""A-share processed label materialization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
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


def select_label_definitions_for_spec(
    spec: ProcessedDataSpec,
    catalog: LabelCatalog,
) -> tuple[LabelDefinition, ...]:
    requested = set(spec.required_columns).difference(spec.primary_key)
    selected = tuple(
        definition
        for definition in catalog.list()
        if definition.output_column in requested
    )
    missing = sorted(
        requested.difference(definition.output_column for definition in selected)
    )
    if missing:
        raise ValueError("unknown_label_output_columns: " + ", ".join(missing))
    return selected


def write_labels(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: pd.DatetimeIndex,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    retained_written_bytes = sum(
        dataset_inspection_retained_memory_bytes(inspection)
        for inspection in written.values()
    )
    catalog = processed_catalog()
    label_catalog = label_definitions.label_catalog()
    definitions = select_label_definitions_for_spec(spec, label_catalog)
    contract = materialization_validation.label_input_contract(definitions)
    materialization_validation.validate_processed_input_contract(
        spec,
        contract=contract,
        purpose="label",
    )
    panel_spec = catalog.get(contract.primary_input_data_id)
    panel_root = materialization_io.processed_root(layout, panel_spec)
    lookahead_sessions = label_calculations.label_lookahead_bars(definitions)
    panel_input_columns = label_calculations.label_panel_input_columns(definitions)

    def chunks() -> Iterable[pd.DataFrame]:
        def materialize(
            asset_ids: tuple[object, ...],
            current_sessions: Sequence[pd.Timestamp],
            session_context: Sequence[pd.Timestamp],
            retained_payload_bytes: int,
            current_keys: pd.DataFrame,
        ) -> pd.DataFrame:
            del session_context
            read_window = label_calculations.label_read_window_for_panel(
                current_keys,
                definitions=definitions,
                read_window=TimeWindow(
                    current_sessions[0].date(), current_sessions[-1].date()
                ),
                core_sessions=current_sessions,
            )
            processed_inputs = materialization_io.read_processed_inputs_for_spec(
                spec,
                layout=layout,
                processed_catalog=catalog,
                time_window=read_window,
                columns_by_data_id={
                    contract.primary_input_data_id: panel_input_columns,
                },
                resource_policy=policy,
                asset_ids=asset_ids,
                retained_payload_bytes=(
                    retained_payload_bytes + pandas_frame_payload_bytes(current_keys)
                ),
            )
            materialization_validation.validate_processed_input_contract(
                spec,
                contract=contract,
                purpose="label",
                processed_inputs=processed_inputs,
            )
            panel = processed_inputs[contract.primary_input_data_id]
            materialization_planning.require_known_materialization_payload_bound(
                frames=(panel, current_keys),
                output=None,
                retained_payload_bytes=(
                    retained_payload_bytes
                    + materialization_planning.registered_output_buffer_bytes(
                        spec,
                        row_count=len(panel),
                    )
                ),
                resource_policy=policy,
            )
            labels = label_calculations.calculate_labels_for_panel(
                panel, definitions=definitions
            )
            materialization_planning.require_known_materialization_payload_bound(
                frames=(panel, labels, current_keys),
                output=None,
                retained_payload_bytes=(
                    retained_payload_bytes + pandas_frame_payload_bytes(labels)
                ),
                resource_policy=policy,
            )
            selected = materialization_planning.filter_output_sessions(
                labels,
                current_sessions,
                columns=spec.required_columns,
                retained_frames=(panel, current_keys),
                retained_payload_bytes=retained_payload_bytes,
                resource_policy=policy,
            )
            materialization_planning.require_known_materialization_payload_bound(
                frames=(panel, labels, current_keys),
                output=selected,
                retained_payload_bytes=retained_payload_bytes,
                resource_policy=policy,
            )
            return selected

        yield from materialization_planning.iter_active_materialization_batches(
            panel_root,
            time_window=time_window,
            sessions=sessions,
            resource_policy=policy,
            valid_bar_offsets=(0, lookahead_sessions),
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


__all__ = ["select_label_definitions_for_spec", "write_labels"]

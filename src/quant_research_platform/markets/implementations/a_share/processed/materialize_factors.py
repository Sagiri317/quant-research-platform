"""A-share processed factor materialization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
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


def select_factor_definitions_for_spec(
    spec: ProcessedDataSpec,
    catalog: FactorCatalog,
) -> tuple[FactorDefinition, ...]:
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
        raise ValueError("unknown_factor_output_columns: " + ", ".join(missing))
    return selected


def write_factors(
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
    factor_catalog = factor_definitions.factor_catalog()
    definitions = select_factor_definitions_for_spec(spec, factor_catalog)
    contract = materialization_validation.factor_input_contract(definitions)
    materialization_validation.validate_processed_input_contract(
        spec,
        contract=contract,
        purpose="factor",
    )
    panel_spec = catalog.get(contract.primary_input_data_id)
    panel_root = materialization_io.processed_root(layout, panel_spec)
    lookback_days = factor_calculations.factor_lookback_bars(definitions)
    panel_input_columns = factor_calculations.factor_panel_input_columns(definitions)

    def chunks() -> Iterable[pd.DataFrame]:
        def materialize(
            asset_ids: tuple[object, ...],
            current_sessions: Sequence[pd.Timestamp],
            session_context: Sequence[pd.Timestamp],
            retained_payload_bytes: int,
            current_keys: pd.DataFrame,
        ) -> pd.DataFrame:
            del session_context
            read_window = factor_calculations.factor_read_window_for_panel(
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
                purpose="factor",
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
            factors = factor_calculations.calculate_factors_for_panel(
                panel, definitions=definitions
            )
            materialization_planning.require_known_materialization_payload_bound(
                frames=(panel, factors, current_keys),
                output=None,
                retained_payload_bytes=(
                    retained_payload_bytes + pandas_frame_payload_bytes(factors)
                ),
                resource_policy=policy,
            )
            selected = materialization_planning.filter_output_sessions(
                factors,
                current_sessions,
                columns=spec.required_columns,
                retained_frames=(panel, current_keys),
                retained_payload_bytes=retained_payload_bytes,
                resource_policy=policy,
            )
            materialization_planning.require_known_materialization_payload_bound(
                frames=(panel, factors, current_keys),
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
            valid_bar_offsets=(lookback_days, 0),
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


__all__ = ["select_factor_definitions_for_spec", "write_factors"]

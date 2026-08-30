"""A-share processed factor materialization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import ExitStack

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
    cleanup_stack: ExitStack | None = None,
) -> DatasetInspection:
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
    input_row_buffer = materialization_planning.registered_output_buffer_bytes(
        panel_spec,
        row_count=1,
    )
    output_row_buffer = materialization_planning.registered_output_buffer_bytes(
        spec,
        row_count=1,
    )
    calculation_row_vectors = (len(definitions) + 3) * 8
    operation_row_bound = input_row_buffer + output_row_buffer + calculation_row_vectors

    def chunks() -> Iterable[object]:
        def materialize(
            asset_ids: tuple[object, ...],
            current_sessions: Sequence[pd.Timestamp],
            session_context: Sequence[pd.Timestamp],
            current_keys: pd.DataFrame,
        ) -> pd.DataFrame:
            del session_context
            with ExitStack() as live:
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
                    cleanup_stack=live,
                )
                materialization_validation.validate_processed_input_contract(
                    spec,
                    contract=contract,
                    purpose="factor",
                    processed_inputs=processed_inputs,
                )
                panel = processed_inputs[contract.primary_input_data_id]
                panel_bytes = pandas_frame_payload_bytes(panel)
                output_buffer = materialization_planning.registered_output_buffer_bytes(
                    spec,
                    row_count=len(panel),
                )
                calculation_vectors = len(panel) * (len(definitions) + 3) * 8
                factor_bound = panel_bytes + output_buffer + calculation_vectors
                live.enter_context(
                    policy.lease("a_share.factor.calculate", factor_bound)
                )
                factors = factor_calculations.calculate_factors_for_panel(
                    panel, definitions=definitions
                )
                if pandas_frame_payload_bytes(factors) > factor_bound:
                    raise materialization_io.MaterializationCapacityExceeded(
                        "a_share_known_payloads_exceed_task_memory_budget"
                    )
                selected = materialization_planning.filter_output_sessions(
                    factors,
                    current_sessions,
                    columns=spec.required_columns,
                    resource_policy=policy,
                )
                return selected

        yield from materialization_planning.iter_active_materialization_batches(
            panel_root,
            time_window=time_window,
            sessions=sessions,
            resource_policy=policy,
            valid_bar_offsets=(lookback_days, 0),
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

"""A-share processed factor materialization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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
    materialization_io,
    materialization_planning,
    materialization_validation,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import DatasetInspection, inspect_dataset
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
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    panel_key_coverages: dict[
        Path,
        dict[object, tuple[pd.Timestamp, ...]],
    ] | None = None,
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
    available = inspect_dataset(panel_root)
    lookback_days = factor_calculations.factor_lookback_bars(definitions)
    panel_input_columns = factor_calculations.factor_panel_input_columns(definitions)
    panel_key_coverage = materialization_planning.processed_panel_key_coverage(
        panel_root,
        cache=panel_key_coverages,
        resource_policy=policy,
    )
    estimated = materialization_planning.estimate_factor_bytes_per_session(
        sessions,
        time_window=time_window,
        panel_root=panel_root,
        inspection=available,
        lookback_days=lookback_days,
        panel_input_columns=panel_input_columns,
        required_columns=spec.required_columns,
        definitions=definitions,
        resource_policy=policy,
    )
    windows = materialization_planning.planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=available,
        estimated_bytes_per_session=estimated,
        warmup_sessions=lookback_days,
        reason="factor_warmup",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            read_window = factor_calculations.factor_read_window_for_panel(
                panel_key_coverage,
                definitions=definitions,
                read_window=window.read,
                core_sessions=window.core_sessions,
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
            )
            materialization_validation.validate_processed_input_contract(
                spec,
                contract=contract,
                purpose="factor",
                processed_inputs=processed_inputs,
            )
            panel = processed_inputs[contract.primary_input_data_id]
            factors = factor_calculations.calculate_factors_for_panel(
                panel,
                definitions=definitions,
            )
            output = materialization_planning.filter_output_sessions(
                factors,
                window.core_sessions,
            )
            selected = output.loc[:, list(spec.required_columns)].copy()
            materialization_planning.require_known_materialization_payload_bound(
                frames=(panel, factors, output),
                output=selected,
                resource_policy=policy,
            )
            yield selected

    return materialization_io.write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
        resource_policy=policy,
    )


__all__ = ["select_factor_definitions_for_spec", "write_factors"]

"""A-share processed label materialization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    panel_key_coverages: dict[
        Path,
        dict[object, tuple[pd.Timestamp, ...]],
    ] | None = None,
) -> DatasetInspection:
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
    available = inspect_dataset(panel_root)
    lookahead_sessions = label_calculations.label_lookahead_bars(definitions)
    panel_input_columns = label_calculations.label_panel_input_columns(definitions)
    panel_key_coverage = materialization_planning.processed_panel_key_coverage(
        panel_root,
        cache=panel_key_coverages,
        resource_policy=policy,
    )
    estimated = materialization_planning.estimate_label_bytes_per_session(
        sessions,
        time_window=time_window,
        panel_root=panel_root,
        inspection=available,
        lookahead_sessions=lookahead_sessions,
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
        lookahead_sessions=lookahead_sessions,
        reason="label_lookahead",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            read_window = label_calculations.label_read_window_for_panel(
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
                purpose="label",
                processed_inputs=processed_inputs,
            )
            panel = processed_inputs[contract.primary_input_data_id]
            labels = label_calculations.calculate_labels_for_panel(
                panel,
                definitions=definitions,
            )
            output = materialization_planning.filter_output_sessions(
                labels,
                window.core_sessions,
            )
            yield output.loc[:, list(spec.required_columns)].copy()

    return materialization_io.write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
        resource_policy=policy,
    )


__all__ = ["select_label_definitions_for_spec", "write_labels"]

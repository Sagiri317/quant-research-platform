"""A-share processed annotation materialization."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
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


def write_annotations(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    catalog = processed_catalog()
    contract = materialization_validation.annotation_input_contract(spec)
    materialization_validation.validate_processed_input_contract(
        spec,
        contract=contract,
        purpose="annotation",
    )
    panel_spec = catalog.get(contract.primary_input_data_id)
    panel_root = materialization_io.processed_root(layout, panel_spec)
    windows = materialization_planning.planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=inspect_dataset(panel_root),
        lookahead_sessions=1,
        reason="annotation_lookahead",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            processed_inputs = materialization_io.read_processed_inputs_for_spec(
                spec,
                layout=layout,
                processed_catalog=catalog,
                time_window=window.read,
                resource_policy=policy,
            )
            materialization_validation.validate_processed_input_contract(
                spec,
                contract=contract,
                purpose="annotation",
                processed_inputs=processed_inputs,
            )
            panel = processed_inputs[contract.primary_input_data_id]
            outputs = annotations.derive_annotations(panel, data_ids=(spec.data_id,))
            output = materialization_planning.filter_output_sessions(
                outputs[spec.data_id],
                window.core_sessions,
            )
            materialization_planning.require_materialization_working_set(
                frames=(panel, *outputs.values()),
                output=output,
                retained_state_bytes=0,
                resource_policy=policy,
            )
            yield output

    return materialization_io.write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
        resource_policy=policy,
    )


__all__ = ["write_annotations"]

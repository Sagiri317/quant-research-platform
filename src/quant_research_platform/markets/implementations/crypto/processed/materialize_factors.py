"""Crypto microstructure factor materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_dependencies,
    materialization_resources,
    materialization_validation,
)
from quant_research_platform.research.factors.microstructure import (
    calculate_microstructure_factors,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
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

MICROSTRUCTURE_ROLLING_WINDOW_MINUTES = 5
MICROSTRUCTURE_ROLLING_CARRY_BARS = MICROSTRUCTURE_ROLLING_WINDOW_MINUTES - 1
_CORE_ROW = "_qrp_core_row"


def write_factors(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    resource_policy.require_valid_chunk_budget()
    panel_root = layout.processed_data_root(
        MARKET_ID,
        ProcessedDataKind.PANEL,
        MARKET_PANEL_DATA_ID,
    )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    input_identities = materialization_validation.input_dataset_identities(
        materialization_validation.panel_dependency_inspections(layout=layout)
    )
    materialization_resources.require_materialization_disk_headroom(
        resource_policy,
        root.parent,
    )
    chunks = materialization_resources.iter_disk_guarded_chunks(
        _iter_factor_chunks(
            spec,
            panel_root=panel_root,
            time_window=time_window,
            resource_policy=resource_policy,
        ),
        policy=resource_policy,
        scratch_parent=root.parent,
    )
    def validate_staged_dataset(staged_root) -> None:
        staged_inspection = inspect_dataset(staged_root, time_window=time_window)
        materialization_validation.validate_panel_dependent_materialization(
            staged_inspection,
            layout=layout,
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
    )
    inspection = inspect_dataset(root, time_window=time_window)
    return inspection


def _iter_factor_chunks(
    spec: ProcessedDataSpec,
    *,
    panel_root,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    budget = materialization_resources.MaterializationBudget.from_policy(
        resource_policy
    )
    chunk_target_bytes = budget.physical_batch_bytes
    active_key: tuple[str, str] | None = None
    active_carry: pd.DataFrame | None = None
    order_state = materialization_validation.OrderedDependencyState()
    read_window = materialization_dependencies.factor_read_window(time_window)
    context = build_dataset_read_context(
        panel_root,
        columns=None,
        time_window=read_window,
    )
    group_plan = build_dataset_equality_group_plan(
        context,
        columns=("market_id", "asset_id"),
        target_chunk_bytes=chunk_target_bytes,
        target_state_bytes=budget.dependency_key_index_bytes,
    )
    materialization_validation.validate_single_asset_part_grouping(
        group_plan,
        error="invalid_crypto_panel_physical_asset_grouping",
    )
    for key in group_plan.keys:
        group_context = dataset_context_for_equality_group(
            context,
            group_plan,
            key,
        )
        for panel in iter_dataset_chunks(
            group_context,
            target_chunk_bytes=chunk_target_bytes,
        ):
            if panel.empty:
                continue
            panel = materialization_validation.validated_ordered_dependency_chunk(
                panel,
                state=order_state,
                error="out_of_order_crypto_factor_dependency",
            )
            panel = panel.copy()
            panel[_CORE_ROW] = materialization_validation.core_time_window_mask(
                panel,
                time_window,
            )
            for (segment_market_id, segment_asset_id), segment in panel.groupby(
                ["market_id", "asset_id"],
                sort=False,
                dropna=False,
            ):
                key = (str(segment_market_id), str(segment_asset_id))
                if active_key != key:
                    active_key = key
                    active_carry = None
                combined = _append_active_factor_carry(segment, active_carry)
                factors = calculate_microstructure_factors(
                    combined,
                    expected_interval=pd.Timedelta(minutes=1),
                    min_periods=MICROSTRUCTURE_ROLLING_WINDOW_MINUTES,
                    trade_flow_observed_column="trade_flow_observed",
                )
                output = factors.loc[factors[_CORE_ROW]].copy()
                active_carry = _next_factor_carry(combined)
                if (
                    materialization_resources.materialization_frame_bytes(
                        active_carry
                    )
                    > budget.carry_bytes
                ):
                    raise ValueError("crypto_factor_state_exceeds_resource_budget")
                if not output.empty:
                    result = output.loc[:, spec.required_columns]
                    materialization_resources.require_frame_within_budget(
                        result,
                        target_bytes=budget.materialized_output_bytes,
                        error="crypto_factor_output_exceeds_resource_budget",
                    )
                    yield result


def _append_active_factor_carry(
    segment: pd.DataFrame,
    active_carry: pd.DataFrame | None,
) -> pd.DataFrame:
    if active_carry is None or active_carry.empty:
        return segment
    return pd.concat(
        (active_carry.assign(**{_CORE_ROW: False}), segment),
        ignore_index=True,
    )


def _next_factor_carry(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.drop(columns=[_CORE_ROW])
        .tail(MICROSTRUCTURE_ROLLING_CARRY_BARS)
        .copy()
    )


__all__ = ["write_factors"]

"""Crypto forward-return label materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_HORIZONS,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_dependencies,
    materialization_resources,
    materialization_validation,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage._dataset_reading import (
    build_dataset_equality_group_plan,
    build_dataset_read_context,
    dataset_context_for_equality_group,
    filter_frame_by_time_window,
    iter_dataset_chunks,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

LABEL_LOOKAHEAD_BARS = max(CRYPTO_FORWARD_RETURN_HORIZONS) + 1
_CORE_ROW = "_qrp_core_row"


def write_labels(
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
        _iter_label_chunks(
            spec,
            panel_root=panel_root,
            time_window=time_window,
            resource_policy=resource_policy,
        ),
        policy=resource_policy,
        scratch_parent=root.parent,
    )
    write_dataset(
        root,
        chunks,
        spec=spec,
        instance_metadata={"input_dataset_identities": input_identities},
    )
    inspection = inspect_dataset(root, time_window=time_window)
    materialization_validation.validate_panel_dependent_materialization(
        inspection,
        layout=layout,
        spec=spec,
        time_window=time_window,
        expected_input_identities=input_identities,
        resource_policy=resource_policy,
    )
    return inspection


def _iter_label_chunks(
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
    buffer = pd.DataFrame()
    active_market_asset: tuple[str, str] | None = None
    order_state = materialization_validation.OrderedDependencyState()
    read_window = materialization_dependencies.label_read_window(time_window)
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
                error="out_of_order_crypto_label_dependency",
            )
            panel = panel.copy()
            panel[_CORE_ROW] = _core_mask(panel, time_window)
            for (segment_market_id, segment_asset_id), asset_rows in panel.groupby(
                ["market_id", "asset_id"],
                sort=False,
                dropna=False,
            ):
                market_asset = (str(segment_market_id), str(segment_asset_id))
                if active_market_asset is None:
                    active_market_asset = market_asset
                elif market_asset != active_market_asset:
                    finalized = _final_label_output(buffer, spec)
                    if not finalized.empty:
                        materialization_resources.require_frame_within_budget(
                            finalized,
                            target_bytes=budget.materialized_output_bytes,
                            error="crypto_label_output_exceeds_resource_budget",
                        )
                        yield finalized
                    buffer = pd.DataFrame()
                    active_market_asset = market_asset
                buffer = _ordered_panel(
                    asset_rows.copy()
                    if buffer.empty
                    else pd.concat((buffer, asset_rows), ignore_index=True)
                )
                emit_mask = _label_emit_mask(buffer) & buffer[_CORE_ROW].astype(
                    "boolean"
                )
                if emit_mask.any():
                    labeled = _label_frame(buffer)
                    output = labeled.loc[emit_mask].loc[:, spec.required_columns]
                    materialization_resources.require_frame_within_budget(
                        output,
                        target_bytes=budget.materialized_output_bytes,
                        error="crypto_label_output_exceeds_resource_budget",
                    )
                    yield output
                    buffer = buffer.loc[~emit_mask].copy()
                if _frame_bytes(buffer) > budget.carry_bytes:
                    raise ValueError("crypto_label_state_exceeds_resource_budget")
    if not buffer.empty:
        finalized = _final_label_output(buffer, spec)
        if not finalized.empty:
            materialization_resources.require_frame_within_budget(
                finalized,
                target_bytes=budget.materialized_output_bytes,
                error="crypto_label_output_exceeds_resource_budget",
            )
            yield finalized


def _frame_bytes(frame: pd.DataFrame) -> int:
    return int(frame.memory_usage(index=True, deep=True).sum())


def _final_label_output(
    buffer: pd.DataFrame,
    spec: ProcessedDataSpec,
) -> pd.DataFrame:
    if buffer.empty:
        return pd.DataFrame(columns=list(spec.required_columns))
    core_mask = buffer[_CORE_ROW].astype("boolean")
    if not core_mask.any():
        return pd.DataFrame(columns=list(spec.required_columns))
    return _label_frame(buffer).loc[core_mask].loc[:, spec.required_columns]


def _label_frame(panel: pd.DataFrame) -> pd.DataFrame:
    return add_forward_return_labels(
        panel,
        horizons=CRYPTO_FORWARD_RETURN_HORIZONS,
        frequency=DataFrequency.MINUTE,
        execution_price_column="close",
        expected_interval=pd.Timedelta(minutes=1),
        require_contiguous=True,
    )


def _label_emit_mask(panel: pd.DataFrame) -> pd.Series:
    positions = panel.groupby("asset_id", sort=False).cumcount()
    group_sizes = panel.groupby("asset_id", sort=False)["asset_id"].transform("size")
    return positions.lt(group_sizes - LABEL_LOOKAHEAD_BARS)


def _ordered_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)


def _core_mask(frame: pd.DataFrame, time_window: TimeWindow) -> pd.Series:
    spec = {
        "column_roles": {"timestamp": "timestamp"},
        "primary_key": ("timestamp",),
    }
    work = frame.loc[:, ["timestamp"]].copy()
    work["_qrp_original_position"] = range(len(frame))
    filtered = filter_frame_by_time_window(
        work,
        spec=spec,
        time_window=time_window,
    )
    positions = pd.Series(False, index=range(len(frame)))
    positions.iloc[filtered["_qrp_original_position"].to_numpy(dtype=int)] = True
    return pd.Series(positions.to_numpy(dtype=bool), index=frame.index)


__all__ = ["write_labels"]

"""Crypto forward-return label materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.data import (
    DataFrequency,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_HORIZONS,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_resources,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_validation as validation,
)
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_paths import require_artifact_root
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    closing_dataset_iterator,
    iter_dataset_chunks,
)
from quant_research_platform.storage._dataset_writing import (
    DatasetWriteChunk,
    DatasetWriteReceipt,
    write_dataset,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
)
from quant_research_platform.storage.layout import StorageLayout

LABEL_LOOKAHEAD_BARS = max(CRYPTO_FORWARD_RETURN_HORIZONS) + 1
_CORE_ROW = "_qrp_core_row"


def write_labels(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    panel_session: DatasetSession,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int,
) -> DatasetInspection:
    if panel_session.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    root = require_artifact_root(
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    )
    with runtime_resource_policy_scope(resource_policy):
        input_identities = validation.input_dataset_identities(
            {MARKET_PANEL_DATA_ID: panel_session}
        )
        chunks = (
            DatasetWriteChunk(
                frame,
                retained_payload_bytes + panel_session.retained_memory_bytes,
            )
            for frame in materialization_resources.iter_disk_guarded_chunks(
                _iter_label_chunks(
                    spec,
                    panel_session=panel_session,
                    time_window=time_window,
                    resource_policy=resource_policy,
                    retained_payload_bytes=retained_payload_bytes,
                ),
                scratch_parent=root.parent,
            )
        )

        def validate_staged_dataset(receipt: DatasetWriteReceipt) -> None:
            validation.validate_panel_dependent_materialization(
                receipt,
                panel_session=panel_session,
                spec=spec,
                time_window=time_window,
                expected_input_identities=input_identities,
                resource_policy=resource_policy,
                retained_payload_bytes=retained_payload_bytes,
            )
            panel_session.verify()

        return write_dataset(
            root,
            chunks,
            spec=spec,
            instance_metadata={"input_dataset_identities": input_identities},
            staged_validator=validate_staged_dataset,
            resource_policy=resource_policy,
            retained_payload_bytes=(
                retained_payload_bytes + panel_session.retained_memory_bytes
            ),
        )


def _iter_label_chunks(
    spec: ProcessedDataSpec,
    *,
    panel_session: DatasetSession,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int,
) -> Iterator[pd.DataFrame]:
    if panel_session.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        chunk_target_bytes = resource_policy.remaining_memory_bytes
        buffer = pd.DataFrame()
        active_market_asset: tuple[str, str] | None = None
        order_state = validation.OrderedDependencyState()
        with validation.crypto_asset_session_views(
            panel_session,
            columns=("market_id", "asset_id"),
            target_chunk_bytes=chunk_target_bytes,
            error="invalid_crypto_panel_physical_asset_grouping",
            operation_headroom_bytes=(
                resource_policy.task_memory_budget_bytes - retained_payload_bytes
            ),
        ) as asset_views:
            for (_market_id, _asset_id), asset_session in asset_views:
                with closing_dataset_iterator(
                    iter_dataset_chunks(
                        asset_session,
                        target_chunk_bytes=chunk_target_bytes,
                        operation_headroom_bytes=(
                            resource_policy.task_memory_budget_bytes
                            - retained_payload_bytes
                        ),
                    )
                ) as panels:
                    for panel in panels:
                        if panel.empty:
                            continue
                        panel = validation.validated_ordered_dependency_chunk(
                            panel,
                            state=order_state,
                            error="out_of_order_crypto_label_dependency",
                        )
                        panel = panel.copy()
                        panel[_CORE_ROW] = (
                            validation.core_time_window_mask(
                                panel,
                                time_window,
                            )
                        )
                        for (
                            segment_market_id,
                            segment_asset_id,
                        ), asset_rows in panel.groupby(
                            ["market_id", "asset_id"],
                            sort=False,
                            dropna=False,
                        ):
                            market_asset = (
                                str(segment_market_id),
                                str(segment_asset_id),
                            )
                            if active_market_asset is None:
                                active_market_asset = market_asset
                            elif market_asset != active_market_asset:
                                finalized = _final_label_output(buffer, spec)
                                if not finalized.empty:
                                    materialization_resources.require_frame_within_budget(
                                        finalized,
                                        target_bytes=resource_policy.remaining_memory_bytes,
                                        error=(
                                            "crypto_indivisible_reducer_unit_exceeds_payload_target"
                                        ),
                                    )
                                    yield finalized
                                buffer = pd.DataFrame()
                                active_market_asset = market_asset
                            buffer = (
                                asset_rows.copy().reset_index(drop=True)
                                if buffer.empty
                                else pd.concat((buffer, asset_rows), ignore_index=True)
                            )
                            positions = buffer.groupby(
                                "asset_id", sort=False
                            ).cumcount()
                            group_sizes = buffer.groupby(
                                "asset_id", sort=False
                            )["asset_id"].transform("size")
                            emit_mask = positions.lt(
                                group_sizes - LABEL_LOOKAHEAD_BARS
                            ) & buffer[_CORE_ROW].astype("boolean")
                            if emit_mask.any():
                                labeled = _label_frame(buffer)
                                output = labeled.loc[emit_mask].loc[
                                    :, spec.required_columns
                                ]
                                materialization_resources.require_frame_within_budget(
                                    output,
                                    target_bytes=resource_policy.remaining_memory_bytes,
                                    error=(
                                        "crypto_indivisible_reducer_unit_exceeds_payload_target"
                                    ),
                                )
                                yield output
                                buffer = buffer.loc[~emit_mask].copy()
                            if (
                                pandas_frame_payload_bytes(buffer)
                                > resource_policy.remaining_memory_bytes
                            ):
                                raise ValueError(
                                    "crypto_indivisible_reducer_unit_exceeds_payload_target"
                                )
            if not buffer.empty:
                finalized = _final_label_output(buffer, spec)
                if not finalized.empty:
                    materialization_resources.require_frame_within_budget(
                        finalized,
                        target_bytes=resource_policy.remaining_memory_bytes,
                        error="crypto_indivisible_reducer_unit_exceeds_payload_target",
                    )
                    yield finalized
        panel_session.verify()


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


__all__ = ["write_labels"]

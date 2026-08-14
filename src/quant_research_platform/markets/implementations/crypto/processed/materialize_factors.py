"""Crypto microstructure factor materialization."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_resources,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_validation as validation,
)
from quant_research_platform.research.factors.microstructure.calculations import (
    calculate_microstructure_factors,
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
    DatasetWriteReceipt,
    write_dataset,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
)
from quant_research_platform.storage.layout import StorageLayout

MICROSTRUCTURE_ROLLING_WINDOW_MINUTES = 5
MICROSTRUCTURE_ROLLING_CARRY_BARS = MICROSTRUCTURE_ROLLING_WINDOW_MINUTES - 1
_CORE_ROW = "_qrp_core_row"


def write_factors(
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
        chunks = materialization_resources.iter_write_chunks(
            _iter_factor_chunks(
                spec,
                panel_session=panel_session,
                time_window=time_window,
                resource_policy=resource_policy,
                retained_payload_bytes=retained_payload_bytes,
            ),
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


def _iter_factor_chunks(
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
        active_key: tuple[str, str] | None = None
        active_carry: pd.DataFrame | None = None
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
                            error="out_of_order_crypto_factor_dependency",
                        )
                        panel = panel.copy()
                        panel[_CORE_ROW] = validation.core_time_window_mask(
                            panel,
                            time_window,
                        )
                        for (
                            segment_market_id,
                            segment_asset_id,
                        ), segment in panel.groupby(
                            ["market_id", "asset_id"],
                            sort=False,
                            dropna=False,
                        ):
                            key = (str(segment_market_id), str(segment_asset_id))
                            if active_key != key:
                                active_key = key
                                active_carry = None
                            combined = _append_active_factor_carry(
                                segment,
                                active_carry,
                            )
                            factors = calculate_microstructure_factors(
                                combined,
                                expected_interval=pd.Timedelta(minutes=1),
                                min_periods=MICROSTRUCTURE_ROLLING_WINDOW_MINUTES,
                                trade_flow_observed_column="trade_flow_observed",
                            )
                            output = factors.loc[factors[_CORE_ROW]].copy()
                            active_carry = (
                                combined.drop(columns=[_CORE_ROW])
                                .tail(MICROSTRUCTURE_ROLLING_CARRY_BARS)
                                .copy()
                            )
                            if (
                                pandas_frame_payload_bytes(active_carry)
                                > resource_policy.remaining_memory_bytes
                            ):
                                raise ValueError(
                                    "crypto_indivisible_reducer_unit_exceeds_payload_target"
                                )
                            if not output.empty:
                                result = output.loc[:, spec.required_columns]
                                materialization_resources.require_frame_within_budget(
                                    result,
                                    target_bytes=resource_policy.remaining_memory_bytes,
                                    error=(
                                        "crypto_indivisible_reducer_unit_exceeds_payload_target"
                                    ),
                                )
                                yield result
        panel_session.verify()


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


__all__ = ["write_factors"]

"""Crypto forward-return label materialization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

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
    MemoryLease,
    RuntimeResourcePolicy,
    _OwnedValue,
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
        chunks = _iter_label_chunks(
            spec,
            panel_session=panel_session,
            time_window=time_window,
            resource_policy=resource_policy,
            retained_payload_bytes=retained_payload_bytes,
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
) -> Iterator[object]:
    if panel_session.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        available_bytes = max(1, resource_policy.remaining_memory_bytes)
        chunk_target_bytes = max(1, available_bytes // 8)
        producer_bound = max(1, available_bytes // 2)
        buffer = pd.DataFrame()
        buffer_lease: MemoryLease | None = None
        active_market_asset: tuple[str, str] | None = None
        order_state = validation.OrderedDependencyState()
        def close_buffer() -> None:
            nonlocal buffer, buffer_lease
            buffer = pd.DataFrame()
            if buffer_lease is not None:
                buffer_lease.close()
                buffer_lease = None

        def replace_buffer(build, lease):  # noqa: ANN001, ANN202
            nonlocal buffer, buffer_lease
            bound = min(
                max(1, lease.reserved_bytes),
                max(1, resource_policy.remaining_memory_bytes),
            )
            next_lease = resource_policy.lease("crypto.label.buffer", bound)
            try:
                next_buffer = build()
                build = None
                next_bytes = pandas_frame_payload_bytes(next_buffer)
                next_lease.shrink_to(
                    next_bytes + min(64 * 1024, max(1, bound // 2))
                )
            except BaseException:
                next_lease.close()
                raise
            old_buffer_lease = buffer_lease
            buffer = next_buffer
            buffer_lease = next_lease
            if old_buffer_lease is not None:
                old_buffer_lease.close()

        with ExitStack() as cleanup, validation.crypto_asset_session_views(
            panel_session,
            columns=("market_id", "asset_id"),
            target_chunk_bytes=chunk_target_bytes,
            error="invalid_crypto_panel_physical_asset_grouping",
            operation_headroom_bytes=(
                resource_policy.task_memory_budget_bytes - retained_payload_bytes
            ),
        ) as asset_views:
            cleanup.callback(close_buffer)
            lease: MemoryLease | None = None
            for (market_id, asset_id), asset_session in asset_views:
                market_asset = (str(market_id), str(asset_id))
                if active_market_asset is None:
                    active_market_asset = market_asset
                elif market_asset != active_market_asset:
                    finalized = _final_label_output(buffer, spec)
                    close_buffer()
                    active_market_asset = market_asset
                    if not finalized.empty:
                        materialization_resources.require_frame_within_budget(
                            finalized,
                            target_bytes=resource_policy.remaining_memory_bytes,
                            error=(
                                "crypto_indivisible_reducer_unit_exceeds_payload_target"
                            ),
                        )
                        actual_bound = (
                            pandas_frame_payload_bytes(finalized) + 128 * 1024
                        )
                        if actual_bound > lease.reserved_bytes:
                            raise ValueError("crypto_label_bound_invalid")
                        lease.shrink_to(actual_bound)
                        yield _OwnedValue(finalized, lease)
                        del finalized
                        lease = None
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
                    while True:
                        if lease is None:
                            lease = yield producer_bound
                        while True:
                            try:
                                panel = next(panels)
                            except StopIteration:
                                panel = None
                                break
                            if not panel.empty:
                                break
                            del panel
                        if panel is None:
                            break
                        if (
                            pandas_frame_payload_bytes(panel) + 128 * 1024
                            > lease.reserved_bytes
                        ):
                            raise ValueError("crypto_label_bound_invalid")
                        panel = validation.validated_ordered_dependency_chunk(
                            panel,
                            state=order_state,
                            error="out_of_order_crypto_label_dependency",
                        )
                        panel = panel.copy()
                        panel[_CORE_ROW] = validation.core_time_window_mask(
                            panel,
                            time_window,
                        )
                        replace_buffer(
                            lambda rows=panel, current=buffer: (
                                rows.copy().reset_index(drop=True)
                                if current.empty
                                else pd.concat((current, rows), ignore_index=True)
                            ),
                            lease,
                        )
                        positions = buffer.groupby("asset_id", sort=False).cumcount()
                        group_sizes = buffer.groupby("asset_id", sort=False)[
                            "asset_id"
                        ].transform("size")
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
                            producer_payload_bytes = pandas_frame_payload_bytes(panel)
                            actual_bound = (
                                pandas_frame_payload_bytes(output)
                                + producer_payload_bytes
                                + 128 * 1024
                            )
                            if actual_bound > lease.reserved_bytes:
                                raise ValueError("crypto_label_bound_invalid")
                            replace_buffer(
                                lambda mask=emit_mask, current=buffer: current.loc[
                                    ~mask
                                ].copy(),
                                lease,
                            )
                            del positions, group_sizes, emit_mask
                            del labeled
                            lease.shrink_to(actual_bound)
                            yield _OwnedValue(output, lease)
                            del output
                            lease = None
                        else:
                            del positions, group_sizes, emit_mask
                        if (
                            pandas_frame_payload_bytes(buffer)
                            > resource_policy.remaining_memory_bytes
                        ):
                            raise ValueError(
                                "crypto_indivisible_reducer_unit_exceeds_payload_target"
                            )
                        panel = None
            if not buffer.empty:
                finalized = _final_label_output(buffer, spec)
                if not finalized.empty:
                    materialization_resources.require_frame_within_budget(
                        finalized,
                        target_bytes=resource_policy.remaining_memory_bytes,
                        error="crypto_indivisible_reducer_unit_exceeds_payload_target",
                    )
                    actual_bound = (
                        pandas_frame_payload_bytes(finalized)
                        + 128 * 1024
                    )
                    if actual_bound > lease.reserved_bytes:
                        raise ValueError("crypto_label_bound_invalid")
                    lease.shrink_to(actual_bound)
                    close_buffer()
                    yield _OwnedValue(finalized, lease)
                    del finalized
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

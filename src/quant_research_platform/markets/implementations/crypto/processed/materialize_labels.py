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
    materialization_validation as validation,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
)
from quant_research_platform.storage.datasets.writing import (
    DatasetWriteReceipt,
    _write_dataset_owned,
)
from quant_research_platform.storage.identity import require_artifact_root
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
    cleanup_stack: ExitStack | None = None,
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
        )

        def validate_staged_dataset(receipt: DatasetWriteReceipt) -> None:
            validation.validate_panel_dependent_materialization(
                receipt,
                panel_session=panel_session,
                spec=spec,
                time_window=time_window,
                expected_input_identities=input_identities,
                resource_policy=resource_policy,
            )
            panel_session.verify()

        owned = _write_dataset_owned(
            root,
            chunks,
            spec=spec,
            instance_metadata={"input_dataset_identities": input_identities},
            staged_validator=validate_staged_dataset,
        )
        result = owned.value

        def release_result() -> None:
            nonlocal result
            result = None
            owned.release(owner="crypto.labels.write.result")

        if cleanup_stack is None:
            owned.release(owner="crypto.labels.write.result")
        else:
            cleanup_stack.callback(release_result)
        return result


def _iter_label_chunks(
    spec: ProcessedDataSpec,
    *,
    panel_session: DatasetSession,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[object]:
    if panel_session.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        available_bytes = max(1, resource_policy.planning_headroom_bytes)
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

        def next_producer_bound() -> int:
            retained_tail = (
                0 if buffer_lease is None else buffer_lease.reserved_bytes
            )
            return max(1, producer_bound - retained_tail)

        with ExitStack() as cleanup, validation.crypto_asset_session_views(
            panel_session,
            columns=("market_id", "asset_id"),
            target_chunk_bytes=chunk_target_bytes,
            error="invalid_crypto_panel_physical_asset_grouping",
        ) as asset_views:
            cleanup.callback(close_buffer)
            lease: MemoryLease | None = None
            for (market_id, asset_id), asset_session in asset_views:
                market_asset = (str(market_id), str(asset_id))
                if active_market_asset is None:
                    active_market_asset = market_asset
                elif market_asset != active_market_asset:
                    if lease is None:
                        lease = yield next_producer_bound()
                    finalized = _final_label_output(buffer, spec)
                    close_buffer()
                    active_market_asset = market_asset
                    if not finalized.empty:
                        actual_bound = (
                            pandas_frame_payload_bytes(finalized) + 128 * 1024
                        )
                        if actual_bound > lease.reserved_bytes:
                            raise ValueError("crypto_label_bound_invalid")
                        lease.shrink_to(actual_bound)
                        yield OwnedValue(finalized, lease)
                        del finalized
                        lease = None
                    else:
                        lease.close()
                        lease = None
                with closing_iterator(
                    iter_dataset_chunks(
                        asset_session,
                        target_chunk_bytes=chunk_target_bytes,
                    )
                ) as panels:
                    while True:
                        if lease is None:
                            lease = yield next_producer_bound()
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
                        combined = (
                            panel
                            if buffer.empty
                            else pd.concat((buffer, panel), ignore_index=True)
                        )
                        buffer = pd.DataFrame()
                        if buffer_lease is not None:
                            buffer_lease.close()
                            buffer_lease = None
                        panel = None
                        combined_bytes = pandas_frame_payload_bytes(combined)
                        if combined_bytes + 128 * 1024 > lease.reserved_bytes:
                            raise ValueError("crypto_label_bound_invalid")
                        emit_mask = pd.Series(False, index=combined.index)
                        if len(combined) > LABEL_LOOKAHEAD_BARS:
                            emit_mask.iloc[:-LABEL_LOOKAHEAD_BARS] = True
                        emit_mask &= combined[_CORE_ROW].astype("boolean")
                        has_output = bool(emit_mask.any())
                        if has_output:
                            labeled = _label_frame(combined)
                            output = labeled.loc[emit_mask].loc[
                                :, spec.required_columns
                            ]
                            actual_bound = (
                                pandas_frame_payload_bytes(output)
                                + 128 * 1024
                            )
                            if actual_bound > lease.reserved_bytes:
                                raise ValueError("crypto_label_bound_invalid")
                            del labeled
                        tail_view = combined.tail(LABEL_LOOKAHEAD_BARS)
                        tail = tail_view.copy()
                        tail_bytes = pandas_frame_payload_bytes(tail)
                        if has_output:
                            if actual_bound + tail_bytes > lease.reserved_bytes:
                                raise ValueError("crypto_label_bound_invalid")
                            tail_bound = tail_bytes + min(
                                64 * 1024,
                                lease.reserved_bytes - actual_bound - tail_bytes,
                            )
                        del combined, tail_view, emit_mask
                        if has_output:
                            lease.shrink_to(actual_bound + tail_bound)
                            buffer_lease = resource_policy.lease(
                                "crypto.label.buffer",
                                tail_bound,
                            )
                            buffer = tail
                            del tail
                            lease.shrink_to(actual_bound)
                            yield OwnedValue(output, lease)
                            del output
                            lease = None
                        else:
                            buffer = tail
                            del tail
                            buffer_lease = None
            if not buffer.empty:
                if lease is None:
                    lease = yield next_producer_bound()
                finalized = _final_label_output(buffer, spec)
                if not finalized.empty:
                    actual_bound = (
                        pandas_frame_payload_bytes(finalized)
                        + 128 * 1024
                    )
                    if actual_bound > lease.reserved_bytes:
                        raise ValueError("crypto_label_bound_invalid")
                    lease.shrink_to(actual_bound)
                    close_buffer()
                    yield OwnedValue(finalized, lease)
                    del finalized
                    lease = None
                else:
                    lease.close()
                    lease = None
            elif lease is not None:
                lease.close()
                lease = None
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

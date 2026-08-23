"""Crypto microstructure factor materialization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack

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
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.factors.microstructure.calculations import (
    calculate_microstructure_factors,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _OwnedValue,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_inspection import (
    _release_owned_value,
)
from quant_research_platform.storage._dataset_paths import require_artifact_root
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    iter_dataset_chunks,
)
from quant_research_platform.storage._dataset_writing import (
    DatasetWriteReceipt,
    _write_dataset_owned,
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
        chunks = _iter_factor_chunks(
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
            _release_owned_value(
                owned,
                owner="crypto.factors.write.result",
            )

        if cleanup_stack is None:
            _release_owned_value(owned, owner="crypto.factors.write.result")
        else:
            cleanup_stack.callback(release_result)
        return result


def _iter_factor_chunks(
    spec: ProcessedDataSpec,
    *,
    panel_session: DatasetSession,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[object]:
    if panel_session.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        available_bytes = max(1, resource_policy.available_memory_bytes)
        chunk_target_bytes = max(1, available_bytes // 8)
        producer_bound = max(1, available_bytes // 2)
        active_key: tuple[str, str] | None = None
        active_carry: pd.DataFrame | None = None
        carry_lease: MemoryLease | None = None
        order_state = validation.OrderedDependencyState()
        def close_carry() -> None:
            nonlocal active_carry, carry_lease
            active_carry = None
            if carry_lease is not None:
                carry_lease.close()
                carry_lease = None

        with ExitStack() as cleanup, validation.crypto_asset_session_views(
            panel_session,
            columns=("market_id", "asset_id"),
            target_chunk_bytes=chunk_target_bytes,
            error="invalid_crypto_panel_physical_asset_grouping",
        ) as asset_views:
            cleanup.callback(close_carry)
            lease: MemoryLease | None = None
            for (market_id, asset_id), asset_session in asset_views:
                key = (str(market_id), str(asset_id))
                if active_key != key:
                    active_key = key
                    close_carry()
                with closing_iterator(
                    iter_dataset_chunks(
                        asset_session,
                        target_chunk_bytes=chunk_target_bytes,
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
                            raise ValueError("crypto_factor_bound_invalid")
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
                        combined = _append_active_factor_carry(
                            panel,
                            active_carry,
                        )
                        if (
                            pandas_frame_payload_bytes(combined) + 128 * 1024
                            > lease.reserved_bytes
                        ):
                            raise ValueError("crypto_factor_bound_invalid")
                        factors = calculate_microstructure_factors(
                            combined,
                            expected_interval=pd.Timedelta(minutes=1),
                            min_periods=MICROSTRUCTURE_ROLLING_WINDOW_MINUTES,
                            trade_flow_observed_column="trade_flow_observed",
                        )
                        output = factors.loc[factors[_CORE_ROW]].copy()
                        carry_bound = min(
                            max(1, lease.reserved_bytes),
                            max(1, resource_policy.available_memory_bytes),
                        )
                        next_carry_lease = resource_policy.lease(
                            "crypto.factor.carry",
                            carry_bound,
                        )
                        try:
                            next_carry = (
                                combined.drop(columns=[_CORE_ROW])
                                .tail(MICROSTRUCTURE_ROLLING_CARRY_BARS)
                                .copy()
                            )
                            next_carry_bytes = pandas_frame_payload_bytes(next_carry)
                            next_carry_lease.shrink_to(
                                next_carry_bytes
                                + min(64 * 1024, max(1, carry_bound // 2))
                            )
                        except BaseException as primary:
                            try:
                                next_carry_lease.close()
                            except BaseException as cleanup_error:
                                primary.add_note(
                                    "crypto factor carry lease close also failed: "
                                    f"{type(cleanup_error).__name__}"
                                )
                            raise
                        old_carry_lease = carry_lease
                        active_carry = next_carry
                        carry_lease = next_carry_lease
                        next_carry = None
                        next_carry_lease = None
                        if old_carry_lease is not None:
                            old_carry_lease.close()
                        if (
                            pandas_frame_payload_bytes(active_carry)
                            > resource_policy.available_memory_bytes
                        ):
                            raise ValueError(
                                "crypto_indivisible_reducer_unit_exceeds_payload_target"
                            )
                        if not output.empty:
                            result = output.loc[:, spec.required_columns]
                            materialization_resources.require_frame_within_budget(
                                result,
                                target_bytes=resource_policy.available_memory_bytes,
                                error=(
                                    "crypto_indivisible_reducer_unit_exceeds_payload_target"
                                ),
                            )
                            result_bytes = pandas_frame_payload_bytes(result)
                            producer_payload_bytes = pandas_frame_payload_bytes(panel)
                            actual_bound = (
                                result_bytes
                                + producer_payload_bytes
                                + 128 * 1024
                            )
                            if actual_bound > lease.reserved_bytes:
                                raise ValueError("crypto_factor_bound_invalid")
                            del factors, output, combined
                            lease.shrink_to(actual_bound)
                            yield _OwnedValue(result, lease)
                            del result
                            lease = None
                        else:
                            del factors, output, combined
                        panel = None
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

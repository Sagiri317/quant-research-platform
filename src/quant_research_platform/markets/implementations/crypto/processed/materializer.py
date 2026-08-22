"""Crypto processed dataset materialization."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    MARKET_PANEL_DATA_ID,
    MICROSTRUCTURE_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    catalog as processed_catalog_module,
)
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_dependencies,
    materialization_validation,
    materialize_factors,
    materialize_labels,
    materialize_panel,
    source_ids,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_inspection import (
    _inspect_dataset_session_owned,
    _release_owned_value,
)
from quant_research_platform.storage._dataset_paths import require_artifact_root
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    open_dataset_session,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout

processed_catalog = processed_catalog_module.processed_catalog


def materialize_processed_data(
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    data_ids: tuple[str, ...] | None = None,
    kinds: frozenset[ProcessedDataKind] | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, DatasetInspection]:
    """Materialize registered crypto processed datasets from local interim data."""

    with runtime_resource_policy_scope(resource_policy) as policy:
        return _materialize_processed_data_with_policy(
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            data_ids=data_ids,
            kinds=kinds,
            resource_policy=policy,
        )


def _materialize_processed_data_with_policy(
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    data_ids: tuple[str, ...] | None,
    kinds: frozenset[ProcessedDataKind] | None,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, DatasetInspection]:
    policy = resource_policy
    if (data_ids is None) == (kinds is None):
        raise ValueError("data_ids_xor_kinds_required")
    catalog = processed_catalog()
    if data_ids is not None:
        data_ids = _strict_data_ids(data_ids)
        requested = tuple(catalog.get(data_id) for data_id in data_ids)
    else:
        assert kinds is not None
        kinds = _strict_kinds(kinds)
        requested = tuple(spec for spec in catalog.list() if spec.kind in kinds)
    if not requested:
        raise ValueError("no_processed_targets")
    for spec in requested:
        if _processed_root(layout, spec).exists():
            raise ValueError("dataset_exists")
    requested_by_id = {spec.data_id: spec for spec in requested}
    specs = _include_dependencies(requested_by_id)
    with ExitStack() as session_stack:
        return _materialize_requested_specs(
            specs=specs,
            requested_by_id=requested_by_id,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            resource_policy=policy,
            session_stack=session_stack,
        )


def _materialize_requested_specs(
    *,
    specs: Mapping[str, ProcessedDataSpec],
    requested_by_id: Mapping[str, ProcessedDataSpec],
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    session_stack: ExitStack,
) -> dict[str, DatasetInspection]:
    written: dict[str, DatasetInspection] = {}
    panel_session: DatasetSession | None = None
    needs_panel_session = any(
        data_id in specs
        for data_id in (
            MICROSTRUCTURE_PROCESSED_DATA_ID,
            FORWARD_RETURNS_PROCESSED_DATA_ID,
        )
    )
    for data_id in (
        MARKET_PANEL_DATA_ID,
        MICROSTRUCTURE_PROCESSED_DATA_ID,
        FORWARD_RETURNS_PROCESSED_DATA_ID,
    ):
        spec = specs.get(data_id)
        if spec is None:
            continue
        target_time_window = _target_time_window(
            spec,
            specs=specs,
            requested_by_id=requested_by_id,
            core_time_window=time_window,
        )
        if (
            spec.data_id not in requested_by_id
            and _processed_root(layout, spec).exists()
        ):
            inspection, panel_session = _inspect_existing_dependency(
                layout,
                spec,
                source_id=source_id,
                time_window=target_time_window,
                resource_policy=resource_policy,
                session_stack=session_stack,
            )
            written[data_id] = inspection
            continue
        written[data_id] = _materialize_one(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=target_time_window,
            resource_policy=resource_policy,
            panel_session=panel_session,
            cleanup_stack=session_stack,
        )
        if spec.kind is ProcessedDataKind.PANEL and needs_panel_session:
            panel_session = session_stack.enter_context(
                open_dataset_session(
                    _processed_root(layout, spec),
                    columns=None,
                    time_window=target_time_window,
                    resource_policy=resource_policy,
                ),
            )
    return written


def _processed_root(layout: StorageLayout, spec: ProcessedDataSpec):
    return require_artifact_root(
        layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    )


def _inspect_existing_dependency(
    layout: StorageLayout,
    spec: ProcessedDataSpec,
    *,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    session_stack: ExitStack,
) -> tuple[DatasetInspection, DatasetSession]:
    bar_data_id = source_ids.market_bars_source_data_id(spec)
    trade_data_id = source_ids.aggregate_trades_source_data_id(spec)
    depth_data_id = source_ids.depth_levels_source_data_id(spec)
    with ExitStack() as input_stack:
        input_sessions: dict[str, DatasetSession] = {}
        for data_id in (bar_data_id, trade_data_id, depth_data_id):
            input_sessions[data_id] = input_stack.enter_context(
                open_dataset_session(
                    layout.interim_data_root(source_id, data_id),
                    columns=None,
                    time_window=time_window,
                    resource_policy=resource_policy,
                ),
            )
        expected_input_identities = materialization_validation.input_dataset_identities(
            input_sessions
        )
        panel_session = session_stack.enter_context(
            open_dataset_session(
                _processed_root(layout, spec),
                columns=None,
                time_window=time_window,
                resource_policy=resource_policy,
            ),
        )
        owned = _inspect_dataset_session_owned(
            panel_session,
            resource_policy=resource_policy,
        )
        inspection = owned.value

        def release_inspection() -> None:
            nonlocal inspection
            inspection = None
            _release_owned_value(
                owned,
                owner="crypto.materializer.inspection.result",
            )

        session_stack.callback(release_inspection)
        materialization_validation.validate_existing_panel_materialization(
            panel_session,
            bar_session=input_sessions[bar_data_id],
            spec=spec,
            expected_input_identities=expected_input_identities,
            resource_policy=resource_policy,
        )
        panel_session.verify()
        for session in input_sessions.values():
            session.verify()
    return inspection, panel_session


def _include_dependencies(
    requested_by_id: Mapping[str, ProcessedDataSpec],
) -> dict[str, ProcessedDataSpec]:
    catalog = processed_catalog()
    expanded = dict(requested_by_id)
    if any(
        data_id in expanded
        for data_id in (
            MICROSTRUCTURE_PROCESSED_DATA_ID,
            FORWARD_RETURNS_PROCESSED_DATA_ID,
        )
    ):
        expanded[MARKET_PANEL_DATA_ID] = catalog.get(MARKET_PANEL_DATA_ID)
    return expanded


def _materialize_one(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    panel_session: DatasetSession | None,
    cleanup_stack: ExitStack,
) -> DatasetInspection:
    if spec.kind is ProcessedDataKind.PANEL:
        return materialize_panel.write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            resource_policy=resource_policy,
            cleanup_stack=cleanup_stack,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        if panel_session is None:
            raise ValueError("missing_crypto_panel_session")
        return materialize_factors.write_factors(
            spec,
            layout=layout,
            panel_session=panel_session,
            time_window=time_window,
            resource_policy=resource_policy,
            cleanup_stack=cleanup_stack,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        if panel_session is None:
            raise ValueError("missing_crypto_panel_session")
        return materialize_labels.write_labels(
            spec,
            layout=layout,
            panel_session=panel_session,
            time_window=time_window,
            resource_policy=resource_policy,
            cleanup_stack=cleanup_stack,
        )
    raise ValueError(f"unsupported_crypto_processed_kind: {spec.kind}")


def _target_time_window(
    spec: ProcessedDataSpec,
    *,
    specs: Mapping[str, ProcessedDataSpec],
    requested_by_id: Mapping[str, ProcessedDataSpec],
    core_time_window: TimeWindow,
) -> TimeWindow:
    if spec.data_id != MARKET_PANEL_DATA_ID:
        return core_time_window
    requested_factor = MICROSTRUCTURE_PROCESSED_DATA_ID in requested_by_id
    requested_label = FORWARD_RETURNS_PROCESSED_DATA_ID in requested_by_id
    if requested_factor and requested_label:
        return materialization_dependencies.panel_dependency_window(core_time_window)
    if requested_factor:
        return materialization_dependencies.factor_read_window(core_time_window)
    if requested_label:
        return materialization_dependencies.label_read_window(core_time_window)
    return core_time_window


def _strict_data_ids(data_ids: object) -> tuple[str, ...]:
    if isinstance(data_ids, str) or not isinstance(data_ids, tuple | list):
        raise ValueError("invalid_crypto_processed_data_ids")
    values = tuple(data_ids)
    if any(not isinstance(data_id, str) for data_id in values):
        raise ValueError("invalid_crypto_processed_data_ids")
    return values


def _strict_kinds(kinds: object) -> frozenset[ProcessedDataKind]:
    if not isinstance(kinds, frozenset) or any(
        not isinstance(kind, ProcessedDataKind) for kind in kinds
    ):
        raise ValueError("invalid_crypto_processed_kinds")
    return kinds


__all__ = ["materialize_processed_data", "processed_catalog"]

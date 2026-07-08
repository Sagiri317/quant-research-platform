"""Crypto processed dataset materialization."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
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
    materialize_factors,
    materialize_labels,
    materialize_panel,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    read_dataset,
)
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

    policy = resource_policy or active_runtime_resource_policy()
    policy.require_valid_chunk_budget()
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
    written: dict[str, DatasetInspection] = {}
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
            written[data_id] = _inspect_existing_dependency(
                layout,
                spec,
                time_window=target_time_window,
            )
            continue
        written[data_id] = _materialize_one(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=target_time_window,
            resource_policy=policy,
        )
    return written


def _processed_root(layout: StorageLayout, spec: ProcessedDataSpec):
    return layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)


def _inspect_existing_dependency(
    layout: StorageLayout,
    spec: ProcessedDataSpec,
    *,
    time_window: TimeWindow,
) -> DatasetInspection:
    inspection = inspect_dataset(_processed_root(layout, spec), time_window=time_window)
    if not _inspection_covers_window(layout, spec, inspection, time_window):
        raise ValueError("insufficient_crypto_dependency_coverage")
    return inspection


def _inspection_covers_window(
    layout: StorageLayout,
    spec: ProcessedDataSpec,
    inspection: DatasetInspection,
    time_window: TimeWindow,
) -> bool:
    if inspection.row_count <= 0:
        return False
    timestamp_columns = [
        column
        for column in spec.primary_key
        if str(column) == "timestamp"
    ]
    if len(timestamp_columns) != 1:
        raise ValueError("time_window_not_supported")
    timestamps = pd.to_datetime(
        read_dataset(
            _processed_root(layout, spec),
            columns=tuple(timestamp_columns),
            time_window=time_window,
        )[timestamp_columns[0]],
        errors="coerce",
    )
    if timestamps.isna().any() or timestamps.empty:
        return False
    start_bound, end_bound, end_inclusive = time_window_timestamp_interval(time_window)
    required_start = pd.Timestamp(start_bound)
    required_end = pd.Timestamp(end_bound)
    if not end_inclusive:
        required_end = required_end - pd.Timedelta(minutes=1)
    return timestamps.min() <= required_start and timestamps.max() >= required_end


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
) -> DatasetInspection:
    if spec.kind is ProcessedDataKind.PANEL:
        return materialize_panel.write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            resource_policy=resource_policy,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        return materialize_factors.write_factors(
            spec,
            layout=layout,
            time_window=time_window,
            resource_policy=resource_policy,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        return materialize_labels.write_labels(
            spec,
            layout=layout,
            time_window=time_window,
            resource_policy=resource_policy,
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

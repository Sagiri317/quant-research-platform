"""Crypto processed dataset materialization."""

from __future__ import annotations

from collections.abc import Mapping

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
    materialize_factors,
    materialize_labels,
    materialize_panel,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
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

    del resource_policy
    if (data_ids is None) == (kinds is None):
        raise ValueError("data_ids_xor_kinds_required")
    catalog = processed_catalog()
    if data_ids is not None:
        requested = tuple(catalog.get(data_id) for data_id in data_ids)
    else:
        assert kinds is not None
        requested = tuple(spec for spec in catalog.list() if spec.kind in kinds)
    if not requested:
        raise ValueError("no_processed_targets")
    specs = _include_dependencies({spec.data_id: spec for spec in requested})
    written: dict[str, DatasetInspection] = {}
    for data_id in (
        MARKET_PANEL_DATA_ID,
        MICROSTRUCTURE_PROCESSED_DATA_ID,
        FORWARD_RETURNS_PROCESSED_DATA_ID,
    ):
        spec = specs.get(data_id)
        if spec is None:
            continue
        written[data_id] = _materialize_one(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
        )
    return written


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
) -> DatasetInspection:
    if spec.kind is ProcessedDataKind.PANEL:
        return materialize_panel.write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        return materialize_factors.write_factors(
            spec,
            layout=layout,
            time_window=time_window,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        return materialize_labels.write_labels(
            spec,
            layout=layout,
            time_window=time_window,
        )
    raise ValueError(f"unsupported_crypto_processed_kind: {spec.kind}")


__all__ = ["materialize_processed_data", "processed_catalog"]

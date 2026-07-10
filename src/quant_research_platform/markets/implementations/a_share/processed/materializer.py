"""A-share processed dataset materialization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as processed_catalog_module,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_dependencies,
    materialization_io,
    materialization_planning,
    materialization_validation,
    materialize_annotations,
    materialize_factors,
    materialize_labels,
    materialize_panel,
    source_ids,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    require_free_disk_headroom,
)
from quant_research_platform.storage.datasets import DatasetInspection, inspect_dataset
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
    """Materialize registered A-share processed datasets."""

    policy = resource_policy or active_runtime_resource_policy()
    policy.require_valid_chunk_budget()
    if (data_ids is None) == (kinds is None):
        raise ValueError("data_ids_xor_kinds_required")
    catalog = processed_catalog()
    if data_ids is not None:
        targets = tuple(catalog.get(data_id) for data_id in data_ids)
    else:
        assert kinds is not None
        targets = tuple(spec for spec in catalog.list() if spec.kind in kinds)
    if not targets:
        raise ValueError("no_processed_targets")
    require_free_disk_headroom(
        policy,
        layout.project_root,
        reserve_bytes=policy.effective_chunk_bytes * 2,
    )
    materialization_validation.validate_processed_dependency_graph(catalog, targets)
    materialization_validation.validate_active_processed_profile_frequencies(
        catalog,
        targets,
    )
    sessions = materialization_planning.available_trading_sessions(layout, source_id)
    materialization_planning.require_target_sessions(sessions, time_window)
    dependency_windows = (
        materialization_dependencies.planned_processed_dependency_windows(
            targets,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            sessions=sessions,
        )
    )
    written: dict[str, DatasetInspection] = {}
    panel_key_coverages: dict[Path, dict[object, tuple[pd.Timestamp, ...]]] = {}
    for spec in targets:
        target_time_window = materialization_dependencies.merge_time_windows(
            time_window,
            dependency_windows.get(spec.data_id, time_window),
        )
        _materialize_one(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=target_time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            requested=True,
            dependency_windows=dependency_windows,
            panel_key_coverages=panel_key_coverages,
        )
    return written


def _materialize_one(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    requested: bool,
    dependency_windows: dict[str, TimeWindow],
    panel_key_coverages: dict[Path, dict[object, tuple[pd.Timestamp, ...]]],
) -> DatasetInspection:
    root = materialization_io.processed_root(layout, spec)
    if spec.data_id in written:
        return written[spec.data_id]
    if root.exists():
        if requested:
            raise ValueError("dataset_exists")
        materialization_planning.require_dataset_coverage(
            root,
            time_window,
            sessions=sessions,
        )
        return inspect_dataset(root, time_window=time_window)
    catalog = processed_catalog()
    materialization_dependencies.ensure_processed_dependencies(
        spec,
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        policy=policy,
        sessions=sessions,
        written=written,
        processed_catalog=catalog,
        dependency_windows=dependency_windows,
        panel_key_coverages=panel_key_coverages,
        materialize_one=_materialize_one,
    )
    if spec.kind is ProcessedDataKind.PANEL:
        return materialize_panel.write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            market_bars_data_id=source_ids.market_bars_source_data_id(spec),
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    if spec.kind is ProcessedDataKind.ANNOTATION:
        return materialize_annotations.write_annotations(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        return materialize_factors.write_factors(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            panel_key_coverages=panel_key_coverages,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        return materialize_labels.write_labels(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            panel_key_coverages=panel_key_coverages,
        )
    raise ValueError(f"unsupported_processed_kind: {spec.kind}")


__all__ = ["materialize_processed_data"]

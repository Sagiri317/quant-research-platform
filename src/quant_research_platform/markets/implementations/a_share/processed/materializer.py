"""A-share processed dataset materialization."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
    panels,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout


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

    del resource_policy
    if (data_ids is None) == (kinds is None):
        raise ValueError("data_ids_xor_kinds_required")
    catalog = a_share_processed_catalog()
    if data_ids is not None:
        targets = tuple(catalog.get(data_id) for data_id in data_ids)
    else:
        assert kinds is not None
        targets = tuple(spec for spec in catalog.list() if spec.kind in kinds)
    if not targets:
        raise ValueError("no_processed_targets")
    written: dict[str, DatasetInspection] = {}
    for spec in targets:
        _materialize_one(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            written=written,
            requested=True,
        )
    return written


def _materialize_one(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
    requested: bool,
) -> DatasetInspection:
    root = _processed_root(layout, spec)
    if spec.data_id in written:
        return written[spec.data_id]
    if root.exists():
        if requested:
            raise ValueError("dataset_exists")
        _require_dataset_coverage(root, time_window)
        return inspect_dataset(root, time_window=time_window)
    if spec.kind is ProcessedDataKind.PANEL:
        return _write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            written=written,
        )
    panel_spec = _catalog_get("daily_panel")
    panel_root = _processed_root(layout, panel_spec)
    if not panel_root.exists():
        _materialize_one(
            panel_spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            written=written,
            requested=False,
        )
    elif panel_spec.data_id not in written:
        _require_dataset_coverage(panel_root, time_window)
    if spec.kind is ProcessedDataKind.ANNOTATION:
        return _write_annotation(
            spec,
            layout=layout,
            time_window=time_window,
            written=written,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        return _write_factors(
            spec,
            layout=layout,
            time_window=time_window,
            written=written,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        return _write_labels(
            spec,
            layout=layout,
            time_window=time_window,
            written=written,
        )
    raise ValueError(f"unsupported_processed_kind: {spec.kind}")


def _write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    daily_market = read_dataset(
        layout.interim_data_root(source_id, "daily_market"),
        time_window=time_window,
    )
    stock_master = read_dataset(layout.interim_data_root(source_id, "stock_master"))
    market_cap = read_dataset(layout.interim_data_root(source_id, "market_cap"))
    trading_calendar = read_dataset(
        layout.interim_data_root(source_id, "trading_calendar")
    )
    panel = panels.compute_daily_panel(
        daily_market,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
    )
    panel = _filter_output_window(panel, time_window)
    output = panel.loc[:, list(spec.required_columns)].copy()
    return _write_and_inspect(
        spec,
        output,
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_annotation(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    panel = read_dataset(
        _processed_root(layout, _catalog_get("daily_panel")),
        time_window=time_window,
    )
    outputs = annotations.derive_annotations(panel, data_ids=(spec.data_id,))
    return _write_and_inspect(
        spec,
        outputs[spec.data_id],
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_factors(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    panel_root = _processed_root(layout, _catalog_get("daily_panel"))
    available = inspect_dataset(panel_root)
    available_window = _available_window(available)
    lookback_days = factor_calculations.price_volume_lookback_days()
    read_window = TimeWindow(
        max(
            available_window.start,
            time_window.start - timedelta(days=lookback_days * 3),
        ),
        min(available_window.end, time_window.end),
    )
    panel = read_dataset(panel_root, time_window=read_window)
    factors = factor_calculations.calculate_price_volume_factors_for_panel(
        panel,
    )
    output = _filter_output_window(factors, time_window)
    output = output.loc[:, list(spec.required_columns)].copy()
    return _write_and_inspect(
        spec,
        output,
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_labels(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    panel_root = _processed_root(layout, _catalog_get("daily_panel"))
    available = inspect_dataset(panel_root)
    available_window = _available_window(available)
    read_window = TimeWindow(
        max(available_window.start, time_window.start),
        available_window.end,
    )
    panel = read_dataset(panel_root, time_window=read_window)
    labels = label_calculations.calculate_forward_return_labels_for_panel(
        panel,
    )
    output = _filter_output_window(labels, time_window)
    output = output.loc[:, list(spec.required_columns)].copy()
    return _write_and_inspect(
        spec,
        output,
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_and_inspect(
    spec: ProcessedDataSpec,
    frame: pd.DataFrame,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    root = _processed_root(layout, spec)
    if root.exists():
        raise ValueError("dataset_exists")
    output = frame.sort_values(list(spec.primary_key)).reset_index(drop=True)
    write_dataset(root, output, spec=spec)
    inspection = inspect_dataset(root, time_window=time_window)
    written[spec.data_id] = inspection
    return inspection


def _processed_root(layout: StorageLayout, spec: ProcessedDataSpec) -> Path:
    return layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)


def _catalog_get(data_id: str) -> ProcessedDataSpec:
    return a_share_processed_catalog().get(data_id)


def _filter_output_window(frame: pd.DataFrame, time_window: TimeWindow) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    dates = output["timestamp"].dt.date
    mask = (dates >= time_window.start) & (dates <= time_window.end)
    return output.loc[mask.fillna(False)].copy()


def _require_dataset_coverage(root: Path, time_window: TimeWindow) -> None:
    available = _available_window(inspect_dataset(root))
    if available.start > time_window.start or available.end < time_window.end:
        raise ValueError("insufficient_dataset_coverage")


def _available_window(inspection: DatasetInspection) -> TimeWindow:
    if inspection.date_min is None or inspection.date_max is None:
        raise ValueError("time_window_not_supported")
    return TimeWindow(inspection.date_min, inspection.date_max)


__all__ = ["materialize_processed_data"]

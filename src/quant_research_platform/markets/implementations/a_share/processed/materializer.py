"""A-share processed dataset materialization."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.data import (
    ProcessedDataKind,
    ProcessedDataSpec,
    market_bars_data_id,
)
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.calendars import (
    SessionWindow,
    SessionWindowPlanner,
    session_dates,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    market_panel_data_id,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
    panels,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as processed_catalog_module,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

SIZING_SAMPLE_SESSION_CAP = 20


processed_catalog = processed_catalog_module.processed_catalog


@dataclass(frozen=True)
class _ProcessedInputContract:
    required_input_data_ids: tuple[str, ...]
    primary_input_data_id: str


_MARKET_PANEL_INPUT_CONTRACT = _ProcessedInputContract(
    required_input_data_ids=(market_panel_data_id(),),
    primary_input_data_id=market_panel_data_id(),
)
_FACTOR_FAMILY_INPUT_CONTRACTS = {
    "price_volume": _MARKET_PANEL_INPUT_CONTRACT,
}
_LABEL_FAMILY_INPUT_CONTRACTS = {
    "forward_return": _MARKET_PANEL_INPUT_CONTRACT,
}


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
    _validate_processed_dependency_graph(catalog, targets)
    sessions = _available_trading_sessions(layout, source_id)
    _require_target_sessions(sessions, time_window)
    dependency_windows = _planned_processed_dependency_windows(
        targets,
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        sessions=sessions,
    )
    written: dict[str, DatasetInspection] = {}
    panel_key_coverages: dict[Path, dict[object, tuple[pd.Timestamp, ...]]] = {}
    for spec in targets:
        target_time_window = _merge_time_windows(
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
    root = _processed_root(layout, spec)
    if spec.data_id in written:
        return written[spec.data_id]
    if root.exists():
        if requested:
            raise ValueError("dataset_exists")
        _require_dataset_coverage(root, time_window, sessions=sessions)
        return inspect_dataset(root, time_window=time_window)
    catalog = processed_catalog()
    _ensure_processed_dependencies(
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
    )
    if spec.kind is ProcessedDataKind.PANEL:
        return _write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    if spec.kind is ProcessedDataKind.ANNOTATION:
        return _write_annotation(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        return _write_factors(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            panel_key_coverages=panel_key_coverages,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        return _write_labels(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            panel_key_coverages=panel_key_coverages,
        )
    raise ValueError(f"unsupported_processed_kind: {spec.kind}")


def _write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    source_inputs = _read_source_inputs_for_spec(
        spec,
        layout=layout,
        source_id=source_id,
    )
    _require_source_inputs_for_spec(spec, source_inputs)
    stock_master = source_inputs["stock_master"]
    market_cap = source_inputs["market_cap"]
    trading_calendar = source_inputs["trading_calendar"]
    market_bars_data_id = _market_bars_source_data_id(spec)
    market_bars_root = layout.interim_data_root(source_id, market_bars_data_id)
    market_bars_inspection = inspect_dataset(market_bars_root)
    _require_source_input_columns_from_schema(
        market_bars_data_id,
        market_bars_inspection.columns,
        columns=("open", "high", "low", "close", "volume", "amount"),
    )
    estimated = _estimate_panel_bytes_per_session(
        sessions,
        time_window=time_window,
        market_bars_root=market_bars_root,
        inspection=market_bars_inspection,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
        required_columns=spec.required_columns,
    )
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=market_bars_inspection,
        estimated_bytes_per_session=estimated,
        lookahead_sessions=1,
        reason="panel_next_session",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            market_bars = read_dataset(market_bars_root, time_window=window.read)
            _require_source_input_columns(
                market_bars_data_id,
                market_bars,
                columns=("open", "high", "low", "close", "volume", "amount"),
            )
            panel = panels.compute_market_panel(
                market_bars,
                stock_master=stock_master,
                market_cap=market_cap,
                trading_calendar=trading_calendar,
            )
            output = _filter_output_sessions(panel, window.core_sessions)
            yield output.loc[:, list(spec.required_columns)].copy()

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_annotation(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    catalog = processed_catalog()
    contract = _annotation_input_contract(spec)
    _validate_processed_input_contract(
        spec,
        contract=contract,
        purpose="annotation",
    )
    panel_spec = catalog.get(contract.primary_input_data_id)
    panel_root = _processed_root(layout, panel_spec)
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=inspect_dataset(panel_root),
        lookahead_sessions=1,
        reason="annotation_lookahead",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            processed_inputs = _read_processed_inputs_for_spec(
                spec,
                layout=layout,
                processed_catalog=catalog,
                time_window=window.read,
            )
            _validate_processed_input_contract(
                spec,
                contract=contract,
                purpose="annotation",
                processed_inputs=processed_inputs,
            )
            panel = processed_inputs[contract.primary_input_data_id]
            outputs = annotations.derive_annotations(panel, data_ids=(spec.data_id,))
            yield _filter_output_sessions(outputs[spec.data_id], window.core_sessions)

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_factors(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    panel_key_coverages: dict[
        Path,
        dict[object, tuple[pd.Timestamp, ...]],
    ] | None = None,
) -> DatasetInspection:
    catalog = processed_catalog()
    factor_catalog = factor_definitions.factor_catalog()
    definitions = _select_factor_definitions_for_spec(spec, factor_catalog)
    contract = _factor_input_contract(definitions)
    _validate_processed_input_contract(
        spec,
        contract=contract,
        purpose="factor",
    )
    panel_spec = catalog.get(contract.primary_input_data_id)
    panel_root = _processed_root(layout, panel_spec)
    available = inspect_dataset(panel_root)
    lookback_days = factor_calculations.factor_lookback_bars(definitions)
    panel_input_columns = factor_calculations.factor_panel_input_columns(definitions)
    panel_key_coverage = _processed_panel_key_coverage(
        panel_root,
        cache=panel_key_coverages,
    )
    estimated = _estimate_factor_bytes_per_session(
        sessions,
        time_window=time_window,
        panel_root=panel_root,
        inspection=available,
        lookback_days=lookback_days,
        panel_input_columns=panel_input_columns,
        required_columns=spec.required_columns,
        definitions=definitions,
    )
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=available,
        estimated_bytes_per_session=estimated,
        warmup_sessions=lookback_days,
        reason="factor_warmup",
    )
    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            read_window = factor_calculations.factor_read_window_for_panel(
                panel_key_coverage,
                definitions=definitions,
                read_window=window.read,
                core_sessions=window.core_sessions,
            )
            processed_inputs = _read_processed_inputs_for_spec(
                spec,
                layout=layout,
                processed_catalog=catalog,
                time_window=read_window,
                columns_by_data_id={
                    contract.primary_input_data_id: panel_input_columns,
                },
            )
            _validate_processed_input_contract(
                spec,
                contract=contract,
                purpose="factor",
                processed_inputs=processed_inputs,
            )
            panel = processed_inputs[contract.primary_input_data_id]
            factors = factor_calculations.calculate_factors_for_panel(
                panel,
                definitions=definitions,
            )
            output = _filter_output_sessions(factors, window.core_sessions)
            yield output.loc[:, list(spec.required_columns)].copy()

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_labels(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    panel_key_coverages: dict[
        Path,
        dict[object, tuple[pd.Timestamp, ...]],
    ] | None = None,
) -> DatasetInspection:
    catalog = processed_catalog()
    label_catalog = label_definitions.label_catalog()
    definitions = _select_label_definitions_for_spec(spec, label_catalog)
    contract = _label_input_contract(definitions)
    _validate_processed_input_contract(
        spec,
        contract=contract,
        purpose="label",
    )
    panel_spec = catalog.get(contract.primary_input_data_id)
    panel_root = _processed_root(layout, panel_spec)
    available = inspect_dataset(panel_root)
    lookahead_sessions = label_calculations.label_lookahead_bars(definitions)
    panel_input_columns = label_calculations.label_panel_input_columns(definitions)
    panel_key_coverage = _processed_panel_key_coverage(
        panel_root,
        cache=panel_key_coverages,
    )
    estimated = _estimate_label_bytes_per_session(
        sessions,
        time_window=time_window,
        panel_root=panel_root,
        inspection=available,
        lookahead_sessions=lookahead_sessions,
        panel_input_columns=panel_input_columns,
        required_columns=spec.required_columns,
        definitions=definitions,
    )
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=available,
        estimated_bytes_per_session=estimated,
        lookahead_sessions=lookahead_sessions,
        reason="label_lookahead",
    )
    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            read_window = label_calculations.label_read_window_for_panel(
                panel_key_coverage,
                definitions=definitions,
                read_window=window.read,
                core_sessions=window.core_sessions,
            )
            processed_inputs = _read_processed_inputs_for_spec(
                spec,
                layout=layout,
                processed_catalog=catalog,
                time_window=read_window,
                columns_by_data_id={
                    contract.primary_input_data_id: panel_input_columns,
                },
            )
            _validate_processed_input_contract(
                spec,
                contract=contract,
                purpose="label",
                processed_inputs=processed_inputs,
            )
            panel = processed_inputs[contract.primary_input_data_id]
            labels = label_calculations.calculate_labels_for_panel(
                panel,
                definitions=definitions,
            )
            output = _filter_output_sessions(labels, window.core_sessions)
            yield output.loc[:, list(spec.required_columns)].copy()

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_and_inspect(
    spec: ProcessedDataSpec,
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    root = _processed_root(layout, spec)
    if root.exists():
        raise ValueError("dataset_exists")
    output = _iter_sorted_output_chunks(frame, spec)
    write_dataset(root, output, spec=spec)
    inspection = inspect_dataset(root, time_window=time_window)
    written[spec.data_id] = inspection
    return inspection


def _processed_root(layout: StorageLayout, spec: ProcessedDataSpec) -> Path:
    return layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)


def _filter_output_sessions(
    frame: pd.DataFrame,
    sessions: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    session_dates_set = {session.date() for session in sessions}
    dates = output["timestamp"].dt.date
    return output.loc[dates.isin(session_dates_set).fillna(False)].copy()


def _require_dataset_coverage(
    root: Path,
    time_window: TimeWindow,
    *,
    sessions: tuple[pd.Timestamp, ...],
) -> None:
    target = _target_sessions(sessions, time_window)
    coverage_window = TimeWindow(target[0].date(), target[-1].date())
    available_sessions = _dataset_session_dates(root, coverage_window)
    missing_sessions = [
        session.date()
        for session in target
        if session.date() not in available_sessions
    ]
    if missing_sessions:
        raise ValueError(
            "insufficient_dataset_coverage: "
            f"missing_sessions={len(missing_sessions)} "
            f"first_missing_session={missing_sessions[0]}"
        )


def _dataset_session_dates(
    root: Path,
    time_window: TimeWindow,
) -> set[object]:
    inspection = inspect_dataset(root)
    if "timestamp" not in inspection.columns:
        raise ValueError("time_window_not_supported")
    chunks = read_dataset(
        root,
        columns=["timestamp"],
        time_window=time_window,
        mode=DatasetReadMode.CHUNKED,
    )
    dates: set[object] = set()
    for chunk in chunks:
        timestamp = pd.to_datetime(chunk["timestamp"], errors="coerce")
        dates.update(timestamp.dropna().dt.date)
    return dates


def _processed_panel_key_coverage(
    root: Path,
    *,
    cache: dict[Path, dict[object, tuple[pd.Timestamp, ...]]] | None = None,
) -> dict[object, tuple[pd.Timestamp, ...]]:
    cache_key = root.resolve()
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    inspection = inspect_dataset(root)
    missing = {"asset_id", "timestamp"}.difference(inspection.columns)
    if missing:
        raise ValueError(
            "missing_processed_panel_key_columns: " + ", ".join(sorted(missing))
        )
    chunks = read_dataset(
        root,
        columns=["asset_id", "timestamp"],
        mode=DatasetReadMode.CHUNKED,
    )
    key_chunks = (chunks,) if isinstance(chunks, pd.DataFrame) else chunks
    coverage: dict[object, set[pd.Timestamp]] = {}
    for chunk in key_chunks:
        if chunk.empty:
            continue
        keys = chunk.loc[:, ["asset_id", "timestamp"]].copy()
        keys["timestamp"] = pd.to_datetime(keys["timestamp"], errors="coerce")
        keys = keys.dropna(subset=["asset_id", "timestamp"])
        for asset_id, asset_rows in keys.groupby("asset_id", sort=False):
            asset_dates = coverage.setdefault(asset_id, set())
            asset_dates.update(asset_rows["timestamp"].dt.normalize())
    result = {
        asset_id: tuple(sorted(asset_dates))
        for asset_id, asset_dates in coverage.items()
    }
    if cache is not None:
        cache[cache_key] = result
    return result


def _iter_sorted_output_chunks(
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    spec: ProcessedDataSpec,
) -> Iterable[pd.DataFrame]:
    frames = [frame] if isinstance(frame, pd.DataFrame) else frame
    for chunk in frames:
        output = chunk.loc[:, list(spec.required_columns)].copy()
        if output.empty:
            continue
        yield output.sort_values(list(spec.primary_key)).reset_index(drop=True)


def _available_trading_sessions(
    layout: StorageLayout,
    source_id: SourceId,
) -> tuple[pd.Timestamp, ...]:
    calendar = read_dataset(layout.interim_data_root(source_id, "trading_calendar"))
    source_market_type = pd.to_numeric(
        calendar["source_market_type"],
        errors="coerce",
    )
    mask = (
        calendar["is_trading_day"].astype(bool)
        & source_market_type.isin(CURRENT_MARKET_TYPES)
    )
    return session_dates(calendar.loc[mask, "timestamp"])


def _target_sessions(
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
) -> tuple[pd.Timestamp, ...]:
    target = tuple(
        session
        for session in sessions
        if time_window.start <= session.date() <= time_window.end
    )
    if not target:
        raise ValueError("no_trading_sessions_in_time_window")
    return target


def _require_target_sessions(
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
) -> None:
    _ = _target_sessions(sessions, time_window)


def _planned_session_windows(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    inspection: DatasetInspection,
    reason: str,
    warmup_sessions: int = 0,
    lookahead_sessions: int = 0,
    estimated_bytes_per_session: int | None = None,
) -> tuple[SessionWindow, ...]:
    target_sessions = _target_sessions(sessions, time_window)
    estimated = (
        _estimated_bytes_per_session(inspection, len(target_sessions))
        if estimated_bytes_per_session is None
        else estimated_bytes_per_session
    )
    resource_plan = ResourceChunkPlanner(policy).plan_sessions(
        axis=ChunkAxis.TIME,
        session_count=len(target_sessions),
        estimated_bytes_per_session=estimated,
    )
    return SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=resource_plan.target_core_session_count,
        warmup_sessions=warmup_sessions,
        lookahead_sessions=lookahead_sessions,
    ).windows(time_window, reason=reason)


def _estimated_bytes_per_session(
    inspection: DatasetInspection,
    session_count: int,
) -> int:
    size = int(inspection.manifest.get("file_size_bytes", 0) or 0)
    if size <= 0:
        size = max(1, int(inspection.row_count)) * 128
    return max(1, ceil(size / max(1, session_count)))


def _estimate_panel_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    market_bars_root: Path,
    inspection: DatasetInspection,
    stock_master: pd.DataFrame,
    market_cap: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        lookahead_sessions=1,
        reason="panel_sizing",
    )
    market_bars = read_dataset(market_bars_root, time_window=sample.read)
    panel = panels.compute_market_panel(
        market_bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
    )
    output = _filter_output_sessions(panel, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(market_bars, panel, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def _estimate_factor_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    panel_root: Path,
    inspection: DatasetInspection,
    lookback_days: int,
    panel_input_columns: tuple[str, ...],
    required_columns: tuple[str, ...],
    definitions: Sequence[FactorDefinition],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        warmup_sessions=lookback_days,
        reason="factor_sizing",
    )
    panel = read_dataset(
        panel_root,
        time_window=sample.read,
        columns=panel_input_columns,
    )
    factors = factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=definitions,
    )
    output = _filter_output_sessions(factors, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(panel, factors, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def _estimate_label_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    panel_root: Path,
    inspection: DatasetInspection,
    lookahead_sessions: int,
    panel_input_columns: tuple[str, ...],
    required_columns: tuple[str, ...],
    definitions: Sequence[LabelDefinition],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        lookahead_sessions=lookahead_sessions,
        reason="label_sizing",
    )
    panel = read_dataset(
        panel_root,
        time_window=sample.read,
        columns=panel_input_columns,
    )
    labels = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=definitions,
    )
    output = _filter_output_sessions(labels, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(panel, labels, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def _sample_window(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    reason: str,
    warmup_sessions: int = 0,
    lookahead_sessions: int = 0,
) -> SessionWindow:
    target_sessions = _target_sessions(sessions, time_window)
    sample_session_count = min(len(target_sessions), SIZING_SAMPLE_SESSION_CAP)
    return SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=sample_session_count,
        warmup_sessions=warmup_sessions,
        lookahead_sessions=lookahead_sessions,
    ).windows(time_window, reason=reason)[0]


def _measured_peak_bytes_per_session(
    *,
    frames: tuple[pd.DataFrame, ...],
    row_count: int,
    sessions: tuple[pd.Timestamp, ...],
    inspection: DatasetInspection,
    target_session_count: int,
) -> int:
    if row_count <= 0:
        return 1
    sample_bytes = sum(
        int(frame.memory_usage(index=True, deep=True).sum()) for frame in frames
    )
    sample_rows_per_session = max(1, row_count) / max(1, len(sessions))
    target_rows_per_session = max(
        sample_rows_per_session,
        int(inspection.row_count) / max(1, target_session_count),
    )
    bytes_per_row = sample_bytes / max(1, row_count)
    return max(1, ceil(bytes_per_row * target_rows_per_session))


def _select_factor_definitions_for_spec(
    spec: ProcessedDataSpec,
    catalog: FactorCatalog,
) -> tuple[FactorDefinition, ...]:
    requested = set(spec.required_columns).difference(spec.primary_key)
    selected = tuple(
        definition
        for definition in catalog.list()
        if definition.output_column in requested
    )
    missing = sorted(
        requested.difference(definition.output_column for definition in selected)
    )
    if missing:
        raise ValueError("unknown_factor_output_columns: " + ", ".join(missing))
    return selected


def _select_label_definitions_for_spec(
    spec: ProcessedDataSpec,
    catalog: LabelCatalog,
) -> tuple[LabelDefinition, ...]:
    requested = set(spec.required_columns).difference(spec.primary_key)
    selected = tuple(
        definition
        for definition in catalog.list()
        if definition.output_column in requested
    )
    missing = sorted(
        requested.difference(definition.output_column for definition in selected)
    )
    if missing:
        raise ValueError("unknown_label_output_columns: " + ", ".join(missing))
    return selected


def _annotation_input_contract(spec: ProcessedDataSpec) -> _ProcessedInputContract:
    if spec.kind is not ProcessedDataKind.ANNOTATION:
        raise ValueError(f"invalid_annotation_spec_kind: {spec.data_id}")
    return _MARKET_PANEL_INPUT_CONTRACT


def _factor_input_contract(
    definitions: Sequence[FactorDefinition],
) -> _ProcessedInputContract:
    families = tuple(dict.fromkeys(definition.family for definition in definitions))
    if not families:
        return _MARKET_PANEL_INPUT_CONTRACT
    if len(families) > 1:
        raise ValueError("mixed_factor_families: " + ", ".join(families))
    try:
        return _FACTOR_FAMILY_INPUT_CONTRACTS[families[0]]
    except KeyError as exc:
        raise ValueError(f"unknown_factor_family: {families[0]}") from exc


def _label_input_contract(
    definitions: Sequence[LabelDefinition],
) -> _ProcessedInputContract:
    families = tuple(dict.fromkeys(definition.family for definition in definitions))
    if not families:
        return _MARKET_PANEL_INPUT_CONTRACT
    if len(families) > 1:
        raise ValueError("mixed_label_families: " + ", ".join(families))
    try:
        return _LABEL_FAMILY_INPUT_CONTRACTS[families[0]]
    except KeyError as exc:
        raise ValueError(f"unknown_label_family: {families[0]}") from exc


def _validate_processed_input_contract(
    spec: ProcessedDataSpec,
    *,
    contract: _ProcessedInputContract,
    purpose: str,
    processed_inputs: dict[str, pd.DataFrame] | None = None,
) -> None:
    declared = tuple(spec.processed_input_data_ids)
    required = tuple(contract.required_input_data_ids)
    available = tuple(processed_inputs) if processed_inputs is not None else declared
    missing = [data_id for data_id in required if data_id not in available]
    if missing:
        raise ValueError(
            f"missing_processed_input_for_{purpose}: {spec.data_id}: "
            + ", ".join(missing)
        )
    unexpected = [data_id for data_id in available if data_id not in required]
    if unexpected:
        raise ValueError(
            f"unexpected_processed_input_for_{purpose}: {spec.data_id}: "
            + ", ".join(unexpected)
        )
    if contract.primary_input_data_id not in required:
        raise ValueError(
            f"invalid_primary_processed_input_for_{purpose}: {spec.data_id}: "
            f"{contract.primary_input_data_id}"
        )


def _read_source_inputs_for_spec(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    market_bars_data_id = (
        _market_bars_source_data_id(spec)
        if spec.kind is ProcessedDataKind.PANEL
        else None
    )
    for data_id in spec.source_input_data_ids:
        if spec.kind is ProcessedDataKind.PANEL and data_id == market_bars_data_id:
            continue
        root = layout.interim_data_root(source_id, data_id)
        inputs[data_id] = read_dataset(root)
    return inputs


def _market_bars_source_data_id(spec: ProcessedDataSpec) -> str:
    expected = market_bars_data_id()
    candidates = tuple(
        data_id
        for data_id in spec.source_input_data_ids
        if data_id == expected
    )
    if len(candidates) != 1:
        raise ValueError("panel_market_bars_source_input_not_unique")
    return candidates[0]


def _require_source_inputs_for_spec(
    spec: ProcessedDataSpec,
    source_inputs: dict[str, pd.DataFrame],
) -> None:
    if spec.kind is not ProcessedDataKind.PANEL:
        return
    missing = [
        data_id
        for data_id in ("stock_master", "market_cap", "trading_calendar")
        if data_id not in source_inputs
    ]
    if missing:
        raise ValueError("missing_source_input_for_panel: " + ", ".join(missing))


def _read_processed_inputs_for_spec(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    processed_catalog: ProcessedDataCatalog,
    time_window: TimeWindow | None = None,
    columns_by_data_id: dict[str, Sequence[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    for data_id in spec.processed_input_data_ids:
        dependency = processed_catalog.get(data_id)
        inputs[data_id] = read_dataset(
            _processed_root(layout, dependency),
            time_window=time_window,
            columns=(
                None
                if columns_by_data_id is None
                else columns_by_data_id.get(data_id)
            ),
        )
    return inputs


def _ensure_processed_dependencies(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    processed_catalog: ProcessedDataCatalog,
    dependency_windows: dict[str, TimeWindow],
    panel_key_coverages: dict[Path, dict[object, tuple[pd.Timestamp, ...]]],
) -> None:
    dependency_window = _processed_dependency_time_window(
        spec,
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        sessions=sessions,
    )
    for data_id in spec.processed_input_data_ids:
        dependency = processed_catalog.get(data_id)
        required_window = _merge_time_windows(
            dependency_window,
            dependency_windows.get(data_id, dependency_window),
        )
        dependency_root = _processed_root(layout, dependency)
        if dependency_root.exists():
            if data_id in written:
                continue
            _require_dataset_coverage(
                dependency_root,
                required_window,
                sessions=sessions,
            )
            continue
        _materialize_one(
            dependency,
            layout=layout,
            source_id=source_id,
            time_window=required_window,
            policy=policy,
            sessions=sessions,
            written=written,
            requested=False,
            dependency_windows=dependency_windows,
            panel_key_coverages=panel_key_coverages,
        )


def _planned_processed_dependency_windows(
    targets: tuple[ProcessedDataSpec, ...],
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    sessions: tuple[pd.Timestamp, ...],
) -> dict[str, TimeWindow]:
    windows: dict[str, TimeWindow] = {}
    for spec in targets:
        if not spec.processed_input_data_ids:
            continue
        dependency_window = _processed_dependency_time_window(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            sessions=sessions,
        )
        for data_id in spec.processed_input_data_ids:
            windows[data_id] = _merge_time_windows(
                windows.get(data_id, dependency_window),
                dependency_window,
            )
    return windows


def _merge_time_windows(left: TimeWindow, right: TimeWindow) -> TimeWindow:
    return TimeWindow(min(left.start, right.start), max(left.end, right.end))


def _processed_dependency_time_window(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    sessions: tuple[pd.Timestamp, ...],
) -> TimeWindow:
    if not spec.processed_input_data_ids:
        return time_window
    if spec.kind is ProcessedDataKind.FACTOR:
        definitions = _select_factor_definitions_for_spec(
            spec,
            factor_definitions.factor_catalog(),
        )
        _validate_processed_input_contract(
            spec,
            contract=_factor_input_contract(definitions),
            purpose="factor",
        )
        lookback = factor_calculations.factor_lookback_bars(definitions)
        return _session_offset_time_window(
            sessions=sessions,
            time_window=time_window,
            lookback_sessions=lookback,
            lookahead_sessions=0,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        definitions = _select_label_definitions_for_spec(
            spec,
            label_definitions.label_catalog(),
        )
        _validate_processed_input_contract(
            spec,
            contract=_label_input_contract(definitions),
            purpose="label",
        )
        lookahead = label_calculations.label_lookahead_bars(definitions)
        return _session_offset_time_window(
            sessions=sessions,
            time_window=time_window,
            lookback_sessions=0,
            lookahead_sessions=lookahead,
        )
    return time_window


def _session_offset_time_window(
    *,
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
    lookback_sessions: int,
    lookahead_sessions: int,
) -> TimeWindow:
    if not sessions or (lookback_sessions <= 0 and lookahead_sessions <= 0):
        return time_window
    normalized_sessions = tuple(
        pd.Timestamp(session).normalize() for session in sessions
    )
    start = pd.Timestamp(time_window.start).normalize()
    end = pd.Timestamp(time_window.end).normalize()
    core_positions = [
        position
        for position, session in enumerate(normalized_sessions)
        if start <= session <= end
    ]
    if not core_positions:
        return time_window
    start_position = max(0, core_positions[0] - max(0, int(lookback_sessions)))
    end_position = min(
        len(normalized_sessions) - 1,
        core_positions[-1] + max(0, int(lookahead_sessions)),
    )
    return TimeWindow(
        normalized_sessions[start_position].date(),
        normalized_sessions[end_position].date(),
    )


def _validate_processed_dependency_graph(
    processed_catalog: ProcessedDataCatalog,
    targets: tuple[ProcessedDataSpec, ...],
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(spec: ProcessedDataSpec) -> None:
        if spec.data_id in visited:
            return
        if spec.data_id in visiting:
            raise ValueError(f"processed_dependency_cycle: {spec.data_id}")
        visiting.add(spec.data_id)
        for dependency_id in spec.processed_input_data_ids:
            try:
                dependency = processed_catalog.get(dependency_id)
            except KeyError as exc:
                raise ValueError(
                    f"unknown_processed_dependency: {dependency_id}"
                ) from exc
            visit(dependency)
        visiting.remove(spec.data_id)
        visited.add(spec.data_id)

    for target in targets:
        visit(target)


def _require_source_input_columns(
    data_id: str,
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> None:
    _require_source_input_columns_from_schema(data_id, frame.columns, columns=columns)


def _require_source_input_columns_from_schema(
    data_id: str,
    available_columns: Iterable[str],
    *,
    columns: tuple[str, ...],
) -> None:
    available = set(available_columns)
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(
            f"missing_source_input_columns: {data_id}: " + ", ".join(missing)
        )


__all__ = ["materialize_processed_data"]

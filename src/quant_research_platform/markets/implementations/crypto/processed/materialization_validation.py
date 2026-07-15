"""Crypto processed materialization lifecycle validation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import (
    strict_hashable_key_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
    canonical_crypto_asset_id,
)
from quant_research_platform.markets.implementations.crypto.processed import source_ids
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
)
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_reading import (
    DatasetEqualityGroupPlan,
    build_dataset_equality_group_plan,
    build_dataset_read_context,
    dataset_primary_key_schema,
    filter_frame_by_time_window,
    iter_dataset_chunks,
)
from quant_research_platform.storage._dataset_schema import dataset_identity_payload
from quant_research_platform.storage._dataset_types import DatasetInspection
from quant_research_platform.storage.datasets import inspect_dataset
from quant_research_platform.storage.layout import StorageLayout


@dataclass
class OrderedDependencyState:
    """Bounded ordering state shared across dependency chunks."""

    last_key: tuple[str, str, pd.Timestamp] | None = None


def validate_single_asset_part_grouping(
    plan: DatasetEqualityGroupPlan,
    *,
    error: str,
) -> None:
    key_by_path: dict[Path, tuple[object, ...]] = {}
    for key, paths in plan.paths_by_key.items():
        for path in paths:
            existing = key_by_path.setdefault(path, key)
            if existing != key:
                raise ValueError(error)


def core_time_window_mask(
    frame: pd.DataFrame,
    time_window: TimeWindow,
) -> pd.Series:
    spec = {
        "column_roles": {"timestamp": "timestamp"},
        "primary_key": ("timestamp",),
    }
    work = frame.loc[:, ["timestamp"]].copy()
    work["_qrp_original_position"] = range(len(frame))
    filtered = filter_frame_by_time_window(
        work,
        spec=spec,
        time_window=time_window,
    )
    positions = pd.Series(False, index=range(len(frame)))
    positions.iloc[filtered["_qrp_original_position"].to_numpy(dtype=int)] = True
    return pd.Series(positions.to_numpy(dtype=bool), index=frame.index)


def panel_input_inspections(
    *,
    layout: StorageLayout,
    source_id: SourceId,
    spec: ProcessedDataSpec,
) -> dict[str, DatasetInspection]:
    data_ids = (
        source_ids.market_bars_source_data_id(spec),
        source_ids.aggregate_trades_source_data_id(spec),
        source_ids.depth_levels_source_data_id(spec),
    )
    return {
        data_id: inspect_dataset(layout.interim_data_root(source_id, data_id))
        for data_id in data_ids
    }


def panel_dependency_inspections(
    *,
    layout: StorageLayout,
) -> dict[str, DatasetInspection]:
    panel_root = layout.processed_data_root(
        MARKET_ID,
        ProcessedDataKind.PANEL,
        MARKET_PANEL_DATA_ID,
    )
    return {MARKET_PANEL_DATA_ID: inspect_dataset(panel_root)}


def input_dataset_identities(
    inspections: Mapping[str, DatasetInspection],
) -> dict[str, dict[str, str]]:
    if not inspections:
        raise ValueError("missing_crypto_materialization_inputs")
    identities: dict[str, dict[str, str]] = {}
    for data_id, inspection in inspections.items():
        if not isinstance(data_id, str) or inspection.spec.get("data_id") != data_id:
            raise ValueError("invalid_crypto_materialization_input_identity")
        identities[data_id] = dataset_identity_payload(
            inspection.spec,
            inspection.manifest,
        )
    return dict(sorted(identities.items()))


def validate_materialization_input_identities(
    inspection: DatasetInspection,
    *,
    expected: Mapping[str, Mapping[str, str]],
    error: str,
) -> None:
    if inspection.manifest.get("input_dataset_identities") != dict(expected):
        raise ValueError(error)


def validate_panel_materialization(
    inspection: DatasetInspection,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    spec: ProcessedDataSpec,
    time_window: TimeWindow,
    expected_input_identities: Mapping[str, Mapping[str, str]],
    resource_policy: RuntimeResourcePolicy,
    scratch_parent: Path | None = None,
) -> None:
    validate_materialization_input_identities(
        inspection,
        expected=expected_input_identities,
        error="crypto_panel_input_dataset_identity_mismatch",
    )
    bar_root = layout.interim_data_root(
        source_id,
        source_ids.market_bars_source_data_id(spec),
    )
    chunk_target_bytes = resource_policy.payload_target_bytes
    bar_context = build_dataset_read_context(
        bar_root,
        columns=("source_asset_id", "timestamp"),
        time_window=time_window,
    )
    actual_context = build_dataset_read_context(
        inspection.root,
        columns=spec.primary_key,
        time_window=time_window,
    )
    actual_group_plan = build_dataset_equality_group_plan(
        actual_context,
        columns=("market_id", "asset_id"),
        target_chunk_bytes=chunk_target_bytes,
    )
    validate_single_asset_part_grouping(
        actual_group_plan,
        error="invalid_crypto_panel_physical_asset_grouping",
    )
    validate_exact_dependency_key_coverage(
        _iter_panel_primary_key_chunks(
            iter_dataset_chunks(
                bar_context,
                target_chunk_bytes=chunk_target_bytes,
            ),
            primary_key=spec.primary_key,
        ),
        iter_dataset_chunks(
            actual_context,
            target_chunk_bytes=chunk_target_bytes,
        ),
        primary_key=spec.primary_key,
        error="invalid_crypto_panel_key_coverage",
        resource_policy=resource_policy,
        scratch_parent=(
            inspection.root.parent if scratch_parent is None else scratch_parent
        ),
        key_schema=dataset_primary_key_schema(actual_context, spec.primary_key),
    )


def validate_panel_dependent_materialization(
    inspection: DatasetInspection,
    *,
    layout: StorageLayout,
    spec: ProcessedDataSpec,
    time_window: TimeWindow,
    expected_input_identities: Mapping[str, Mapping[str, str]],
    resource_policy: RuntimeResourcePolicy,
    scratch_parent: Path | None = None,
) -> None:
    validate_materialization_input_identities(
        inspection,
        expected=expected_input_identities,
        error="crypto_processed_input_dataset_identity_mismatch",
    )
    panel_root = layout.processed_data_root(
        MARKET_ID,
        ProcessedDataKind.PANEL,
        MARKET_PANEL_DATA_ID,
    )
    chunk_target_bytes = resource_policy.payload_target_bytes
    expected_context = build_dataset_read_context(
        panel_root,
        columns=spec.primary_key,
        time_window=time_window,
    )
    actual_context = build_dataset_read_context(
        inspection.root,
        columns=spec.primary_key,
        time_window=time_window,
    )
    validate_exact_dependency_key_coverage(
        iter_dataset_chunks(
            expected_context,
            target_chunk_bytes=chunk_target_bytes,
        ),
        iter_dataset_chunks(
            actual_context,
            target_chunk_bytes=chunk_target_bytes,
        ),
        primary_key=spec.primary_key,
        error="invalid_crypto_processed_key_coverage",
        resource_policy=resource_policy,
        scratch_parent=(
            inspection.root.parent if scratch_parent is None else scratch_parent
        ),
        key_schema=dataset_primary_key_schema(actual_context, spec.primary_key),
    )


def validate_exact_dependency_key_coverage(
    expected: Iterable[pd.DataFrame],
    actual: Iterable[pd.DataFrame],
    *,
    primary_key: tuple[str, ...],
    error: str,
    resource_policy: RuntimeResourcePolicy,
    scratch_parent: Path,
    key_schema: pa.Schema | None = None,
) -> None:
    if not primary_key:
        raise ValueError(error)
    chunk_target_bytes = resource_policy.payload_target_bytes
    timestamp_columns = tuple(
        column for column in primary_key if column == "timestamp"
    )
    with ExactPrimaryKeyIndex(
        primary_key=primary_key,
        timestamp_columns=timestamp_columns,
        scratch_parent=scratch_parent,
        key_schema=key_schema,
    ) as index:
        for namespace, frames in (("expected", expected), ("actual", actual)):
            for frame in frames:
                if not isinstance(frame, pd.DataFrame):
                    raise ValueError(error)
                keys = _normalized_primary_keys(frame, primary_key, error=error)
                key_bytes = pandas_frame_payload_bytes(keys)
                if key_bytes > chunk_target_bytes:
                    raise ValueError(error)
                require_free_disk_headroom(
                    scratch_parent,
                    reserve_bytes=key_bytes,
                )
                try:
                    duplicates = index.add_frame(keys, namespace=namespace)
                except ValueError as exc:
                    raise ValueError(error) from exc
                if duplicates:
                    raise ValueError(error)
        missing_count, extra_count = index.difference_counts(
            expected_namespace="expected",
            actual_namespace="actual",
        )
        if missing_count or extra_count:
            raise ValueError(error)


def validated_ordered_dependency_chunk(
    frame: pd.DataFrame,
    *,
    state: OrderedDependencyState,
    error: str,
) -> pd.DataFrame:
    required = ("market_id", "asset_id", "timestamp")
    if any(column not in frame for column in required):
        raise ValueError(error)
    ordered = frame.copy()
    for column in ("market_id", "asset_id"):
        if not ordered[column].map(
            lambda value: isinstance(value, str) and bool(value)
        ).all():
            raise ValueError(error)
    ordered["timestamp"] = strict_timestamp_series(
        ordered["timestamp"],
        error=error,
    )
    previous_key = state.last_key
    for row in ordered.loc[:, list(required)].itertuples(index=False, name=None):
        key = (str(row[0]), str(row[1]), pd.Timestamp(row[2]))
        if previous_key is not None and key <= previous_key:
            raise ValueError(error)
        previous_key = key
    state.last_key = previous_key
    return ordered.reset_index(drop=True)


def _normalized_primary_keys(
    frame: pd.DataFrame,
    primary_key: tuple[str, ...],
    *,
    error: str,
) -> pd.DataFrame:
    if any(column not in frame for column in primary_key):
        raise ValueError(error)
    keys = frame.loc[:, list(primary_key)].copy()
    for column in primary_key:
        if column == "timestamp":
            keys[column] = strict_timestamp_series(keys[column], error=error)
        else:
            keys[column] = strict_hashable_key_series(keys[column], error=error)
    return keys


def _panel_primary_keys(
    bars: pd.DataFrame,
    *,
    primary_key: tuple[str, ...],
) -> pd.DataFrame:
    required = {"market_id", "asset_id", "timestamp"}
    if set(primary_key) != required or any(
        column not in bars for column in ("source_asset_id", "timestamp")
    ):
        raise ValueError("invalid_crypto_panel_primary_key_contract")
    expected = pd.DataFrame(index=bars.index)
    expected["market_id"] = str(MARKET_ID)
    try:
        expected["asset_id"] = bars["source_asset_id"].map(
            canonical_crypto_asset_id
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_crypto_panel_source_asset_id") from exc
    expected["timestamp"] = strict_timestamp_series(
        bars["timestamp"],
        error="invalid_crypto_panel_dependency_timestamp",
    )
    return expected.loc[:, list(primary_key)]


def _iter_panel_primary_key_chunks(
    bars: Iterable[pd.DataFrame],
    *,
    primary_key: tuple[str, ...],
) -> Iterator[pd.DataFrame]:
    for chunk in bars:
        if chunk.empty:
            continue
        yield _panel_primary_keys(chunk, primary_key=primary_key)


__all__ = [
    "input_dataset_identities",
    "core_time_window_mask",
    "panel_dependency_inspections",
    "panel_input_inspections",
    "validate_exact_dependency_key_coverage",
    "validate_materialization_input_identities",
    "validate_panel_dependent_materialization",
    "validate_panel_materialization",
    "validated_ordered_dependency_chunk",
    "validate_single_asset_part_grouping",
]

"""Crypto processed materialization lifecycle validation."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

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
from quant_research_platform.storage._dataset_schema import dataset_identity_payload
from quant_research_platform.storage._dataset_types import DatasetInspection
from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
from quant_research_platform.storage.layout import StorageLayout


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
    bars = _materialized_frame(
        read_dataset(
            bar_root,
            columns=("source_asset_id", "timestamp"),
            time_window=time_window,
        )
    )
    expected = _panel_primary_keys(bars, primary_key=spec.primary_key)
    actual = _materialized_frame(
        read_dataset(
            inspection.root,
            columns=spec.primary_key,
            time_window=time_window,
        )
    )
    validate_exact_dependency_key_coverage(
        expected,
        actual,
        primary_key=spec.primary_key,
        error="invalid_crypto_panel_key_coverage",
    )


def validate_panel_dependent_materialization(
    inspection: DatasetInspection,
    *,
    layout: StorageLayout,
    spec: ProcessedDataSpec,
    time_window: TimeWindow,
    expected_input_identities: Mapping[str, Mapping[str, str]],
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
    expected = _materialized_frame(
        read_dataset(
            panel_root,
            columns=spec.primary_key,
            time_window=time_window,
        )
    )
    actual = _materialized_frame(
        read_dataset(
            inspection.root,
            columns=spec.primary_key,
            time_window=time_window,
        )
    )
    validate_exact_dependency_key_coverage(
        expected,
        actual,
        primary_key=spec.primary_key,
        error="invalid_crypto_processed_key_coverage",
    )


def validate_exact_dependency_key_coverage(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    primary_key: tuple[str, ...],
    error: str,
) -> None:
    if not primary_key:
        raise ValueError(error)
    expected_keys = _normalized_primary_keys(expected, primary_key, error=error)
    actual_keys = _normalized_primary_keys(actual, primary_key, error=error)
    if (
        bool(expected_keys.duplicated(list(primary_key)).any())
        or bool(actual_keys.duplicated(list(primary_key)).any())
        or set(expected_keys.itertuples(index=False, name=None))
        != set(actual_keys.itertuples(index=False, name=None))
    ):
        raise ValueError(error)


def validated_ordered_dependency_chunk(
    frame: pd.DataFrame,
    *,
    watermarks: dict[tuple[str, str], pd.Timestamp],
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
    ordered = ordered.sort_values(list(required), kind="mergesort").reset_index(
        drop=True
    )
    next_watermarks = dict(watermarks)
    for (market_id, asset_id), rows in ordered.groupby(
        ["market_id", "asset_id"],
        sort=False,
        dropna=False,
    ):
        timestamps = rows["timestamp"]
        if bool(timestamps.duplicated().any()):
            raise ValueError(error)
        key = (str(market_id), str(asset_id))
        previous = next_watermarks.get(key)
        earliest = pd.Timestamp(timestamps.iloc[0])
        if previous is not None and earliest <= previous:
            raise ValueError(error)
        next_watermarks[key] = pd.Timestamp(timestamps.iloc[-1])
    watermarks.clear()
    watermarks.update(next_watermarks)
    return ordered


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


def _materialized_frame(value: object) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError("materialized_dataset_read_required")
    return value


__all__ = [
    "input_dataset_identities",
    "panel_dependency_inspections",
    "panel_input_inspections",
    "validate_exact_dependency_key_coverage",
    "validate_materialization_input_identities",
    "validate_panel_dependent_materialization",
    "validate_panel_materialization",
    "validated_ordered_dependency_chunk",
]

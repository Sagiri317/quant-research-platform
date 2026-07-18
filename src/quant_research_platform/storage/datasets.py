"""Dataset artifact IO, inspection, fingerprints, and comparisons."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage._dataset_comparison import (
    compare_datasets as _compare_datasets,
)
from quant_research_platform.storage._dataset_fingerprinting import (
    fingerprint_dataset as _fingerprint_dataset,
)
from quant_research_platform.storage._dataset_inspection import (
    inspect_dataset as _inspect_dataset,
)
from quant_research_platform.storage._dataset_reading import (
    read_dataset as _read_dataset,
)
from quant_research_platform.storage._dataset_types import (
    DatasetComparison,
    DatasetFingerprint,
    DatasetInspection,
    DatasetReadMode,
)
from quant_research_platform.storage._dataset_writing import (
    write_dataset as _write_dataset,
)


def write_dataset(
    dataset_root: Path,
    data: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    spec: DataSpec,
    instance_metadata: Mapping[str, Any] | None = None,
) -> None:
    _write_dataset(
        dataset_root,
        data,
        spec=spec,
        instance_metadata=instance_metadata,
        resource_policy=None,
    )


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    return _read_dataset(
        dataset_root,
        columns=columns,
        time_window=time_window,
        mode=mode,
        resource_policy=None,
    )


def inspect_dataset(
    dataset_root: Path,
    *,
    time_window: TimeWindow | None = None,
) -> DatasetInspection:
    return _inspect_dataset(
        dataset_root,
        time_window=time_window,
        resource_policy=None,
    )


def fingerprint_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetFingerprint:
    return _fingerprint_dataset(
        dataset_root,
        columns=columns,
        time_window=time_window,
        resource_policy=None,
    )


def compare_datasets(
    left_root: Path,
    right_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetComparison:
    return _compare_datasets(
        left_root,
        right_root,
        columns=columns,
        time_window=time_window,
        resource_policy=None,
    )


__all__ = [
    "DatasetReadMode",
    "DatasetInspection",
    "DatasetFingerprint",
    "DatasetComparison",
    "write_dataset",
    "read_dataset",
    "inspect_dataset",
    "fingerprint_dataset",
    "compare_datasets",
]

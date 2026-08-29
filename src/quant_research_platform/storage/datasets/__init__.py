"""Dataset artifact IO, inspection, fingerprints, and comparisons."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import current_runtime_resource_policy
from quant_research_platform.storage.datasets.types import (
    DatasetComparison,
    DatasetFingerprint,
    DatasetInspection,
    DatasetReadMode,
)


def _reject_unrelated_active_root() -> None:
    if current_runtime_resource_policy() is not None:
        raise RuntimeError("runtime_resource_policy_competing_root")


def write_dataset(
    dataset_root: Path,
    data: pd.DataFrame | list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    *,
    spec: DataSpec,
    instance_metadata: Mapping[str, Any] | None = None,
) -> None:
    if not (
        type(data) is pd.DataFrame
        or (
            type(data) in (list, tuple)
            and all(type(frame) is pd.DataFrame for frame in data)
        )
    ):
        raise TypeError("invalid_dataset_write_input")
    from quant_research_platform.storage.datasets.writing import (
        write_dataset as _write_dataset,
    )

    _reject_unrelated_active_root()
    _write_dataset(
        dataset_root,
        data,
        spec=spec,
        instance_metadata=instance_metadata,
        resource_policy=None,
        borrowed_input=True,
    )


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    from quant_research_platform.storage.datasets.reading import (
        read_dataset as _read_dataset,
    )

    _reject_unrelated_active_root()
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
    from quant_research_platform.storage.datasets.inspection import (
        inspect_dataset as _inspect_dataset,
    )

    _reject_unrelated_active_root()
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
    from quant_research_platform.storage.datasets.fingerprinting import (
        fingerprint_dataset as _fingerprint_dataset,
    )

    _reject_unrelated_active_root()
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
    from quant_research_platform.storage.datasets.comparison import (
        compare_datasets as _compare_datasets,
    )

    _reject_unrelated_active_root()
    return _compare_datasets(
        left_root,
        right_root,
        columns=columns,
        time_window=time_window,
        resource_policy=None,
    )

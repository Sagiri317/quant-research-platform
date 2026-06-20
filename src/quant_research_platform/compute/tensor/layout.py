"""Tensor layout contracts for cross-sectional factor evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TensorLayout:
    """Axis labels for time x asset x signal arrays."""

    timestamps: tuple[pd.Timestamp, ...]
    assets: tuple[str, ...]
    signals: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class PanelTensor:
    """Dense tensor view of a long-format research panel."""

    layout: TensorLayout
    signal_values: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class TensorBatchLayout:
    """Axis labels for time x asset x signal x label tensor batches."""

    timestamps: tuple[pd.Timestamp, ...]
    assets: tuple[str, ...]
    signals: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TensorBatch:
    """Dense tensor batch view for long-format research panels."""

    layout: TensorBatchLayout
    signal_values: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray

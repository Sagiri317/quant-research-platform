"""Internal tensor layout contracts for cross-sectional factor evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TensorLayout:
    """Axis labels for arrays shaped as time x asset x feature."""

    timestamps: tuple[pd.Timestamp, ...]
    assets: tuple[str, ...]
    features: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class PanelTensor:
    """Dense internal tensor view of a long-format research panel."""

    layout: TensorLayout
    feature_values: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class TensorBatchLayout:
    """Axis labels for the main time x asset x signal x label tensor contract."""

    timestamps: tuple[pd.Timestamp, ...]
    assets: tuple[str, ...]
    signals: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TensorBatch:
    """Dense internal tensor batch view of a long-format research panel.

    `signal_values` is shaped time x asset x signal.
    `label_values` is shaped time x asset x label.
    `valid_mask` is shaped time x asset x signal x label.
    """

    layout: TensorBatchLayout
    signal_values: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray

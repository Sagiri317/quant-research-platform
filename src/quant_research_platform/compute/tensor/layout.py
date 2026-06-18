"""    cross-sectional factor evaluation     ?tensor layout contracts ?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TensorLayout:
    """    time x asset x signal arrays  ?axis labels ?"""

    timestamps: tuple[pd.Timestamp, ...]
    assets: tuple[str, ...]
    signals: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class PanelTensor:
    """    long-format research panel     ?dense tensor view ?"""

    layout: TensorLayout
    signal_values: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class TensorBatchLayout:
    """ ?time x asset x signal x label tensor contract  ?axis labels ?"""

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

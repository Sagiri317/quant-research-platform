"""定义 cross-sectional factor evaluation 的内部 tensor layout contracts。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TensorLayout:
    """定义 time x asset x feature arrays 的 axis labels。"""

    timestamps: tuple[pd.Timestamp, ...]
    assets: tuple[str, ...]
    features: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class PanelTensor:
    """表示 long-format research panel 的内部 dense tensor view。"""

    layout: TensorLayout
    feature_values: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class TensorBatchLayout:
    """主 time x asset x signal x label tensor contract 的 axis labels。"""

    timestamps: tuple[pd.Timestamp, ...]
    assets: tuple[str, ...]
    signals: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TensorBatch:
    """表示 long-format research panel 的内部 dense tensor batch view。

    `signal_values` 的形状是 time x asset x signal。
    `label_values` 的形状是 time x asset x label。
    `valid_mask` 的形状是 time x asset x signal x label。
    """

    layout: TensorBatchLayout
    signal_values: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray

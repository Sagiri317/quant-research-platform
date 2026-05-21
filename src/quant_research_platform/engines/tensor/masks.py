"""Mask helpers for internal tensor evaluation."""

from __future__ import annotations

import numpy as np

from quant_research_platform.engines.tensor.layout import PanelTensor


def valid_observation_mask(tensor: PanelTensor) -> np.ndarray:
    """Return finite feature-label observation mask."""

    return np.asarray(tensor.valid_mask, dtype=bool)

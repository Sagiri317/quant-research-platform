"""       tensor evaluation  ?mask helpers ?"""

from __future__ import annotations

import numpy as np

from quant_research_platform.compute.tensor.layout import PanelTensor


def valid_observation_mask(tensor: PanelTensor) -> np.ndarray:
    """       signal-label observation mask ?"""

    return np.asarray(tensor.valid_mask, dtype=bool)

"""提供内部 tensor evaluation 的 mask helpers。"""

from __future__ import annotations

import numpy as np

from quant_research_platform.kernels.tensor.layout import PanelTensor


def valid_observation_mask(tensor: PanelTensor) -> np.ndarray:
    """返回有限 feature-label observation mask。"""

    return np.asarray(tensor.valid_mask, dtype=bool)

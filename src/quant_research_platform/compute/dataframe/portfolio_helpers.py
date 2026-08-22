"""DataFrame portfolio weighting and tradability helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import WeightConfig


def risk_values(
    timestamp_frame: pd.DataFrame,
    weight_config: WeightConfig,
) -> np.ndarray | None:
    if weight_config.method != "inverse_vol_scaled_quantile":
        return None
    return timestamp_frame[str(weight_config.risk_column)].to_numpy(
        dtype="float64",
        copy=False,
    )

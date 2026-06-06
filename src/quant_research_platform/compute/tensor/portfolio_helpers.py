"""提供 Tensor portfolio weighting 和 tradability helpers。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.config import WeightConfig


def _directed_signal_values(values: np.ndarray, direction: str) -> np.ndarray:
    if direction == "long_high":
        return values
    if direction == "long_low":
        return -values
    raise ValueError("direction must be 'long_high' or 'long_low'.")


def _leg_weight_values(
    *,
    asset_indices: np.ndarray,
    feature_values: np.ndarray,
    risk_values: np.ndarray | None,
    feature: str,
    weight_config: WeightConfig,
    sign: float,
) -> tuple[np.ndarray, str]:
    if len(asset_indices) == 0:
        return np.array([], dtype="float64"), ""
    fallback_reason = ""
    leg_values = feature_values[asset_indices]
    if weight_config.method == "equal_weight_quantile":
        return np.full(
            len(asset_indices),
            sign / len(asset_indices),
            dtype="float64",
        ), fallback_reason
    elif weight_config.method == "rank_weighted_quantile":
        raw = np.arange(1, len(asset_indices) + 1, dtype="float64")
    elif weight_config.method == "zscore_weighted_quantile":
        std = leg_values.std(ddof=0)
        raw = (
            np.abs((leg_values - leg_values.mean()) / std)
            if std > 0
            else leg_values * 0
        )
    elif weight_config.method == "signal_abs_weighted_quantile":
        raw = np.abs(leg_values)
    elif weight_config.method == "inverse_vol_scaled_quantile":
        if risk_values is None:
            raise ValueError(f"risk_column is required for feature {feature!r}.")
        risk = risk_values[asset_indices]
        if (risk <= 0).any():
            raise ValueError("risk_column values must be positive.")
        raw = np.abs(leg_values) / risk
    else:
        raise ValueError(f"unsupported weight method {weight_config.method!r}.")
    if not np.isfinite(raw).all() or float(raw.sum()) <= 0:
        raw = np.ones(len(asset_indices), dtype="float64")
        fallback_reason = "equal_weight_fallback"
    return sign * raw / raw.sum(), fallback_reason



def _filter_panel_group(
    *,
    panel: pd.DataFrame,
    segment_columns: tuple[str, ...],
    group_values: dict[str, object],
) -> pd.DataFrame:
    if not segment_columns:
        return panel
    mask = pd.Series(True, index=panel.index)
    for column in segment_columns:
        mask &= panel[column] == group_values[column]
    return panel.loc[mask]

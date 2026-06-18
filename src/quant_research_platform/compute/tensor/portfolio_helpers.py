"""    Tensor portfolio weighting  ?tradability helpers ?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import WeightConfig


def _leg_weight_values(
    *,
    asset_indices: np.ndarray,
    signal_values: np.ndarray,
    risk_values: np.ndarray | None,
    signal: str,
    weight_config: WeightConfig,
    sign: float,
) -> tuple[np.ndarray, str]:
    if len(asset_indices) == 0:
        return np.array([], dtype="float64"), ""
    fallback_reason = ""
    leg_values = signal_values[asset_indices]
    if weight_config.method == "equal_weight_quantile":
        return np.full(
            len(asset_indices),
            sign / len(asset_indices),
            dtype="float64",
        ), fallback_reason
    elif weight_config.method == "rank_weighted_quantile":
        raw = (
            np.arange(len(asset_indices), 0, -1, dtype="float64")
            if sign < 0
            else np.arange(1, len(asset_indices) + 1, dtype="float64")
        )
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
            raise ValueError(f"risk_column is required for signal {signal!r}.")
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

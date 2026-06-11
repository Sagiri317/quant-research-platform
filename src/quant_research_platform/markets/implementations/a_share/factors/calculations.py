"""A-share-specific factor calculations."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.research.factors.price_volume import (
    calculate_price_volume_factors,
    largest_price_volume_lookback_days,
)

from .definitions import a_share_price_volume_factor_columns


def calculate_price_volume_factors_for_panel(panel_frame: pd.DataFrame) -> pd.DataFrame:
    return calculate_price_volume_factors(
        panel_frame,
        factor_columns=a_share_price_volume_factor_columns(),
    )


def price_volume_lookback_days() -> int:
    return largest_price_volume_lookback_days(a_share_price_volume_factor_columns())


__all__ = [
    "calculate_price_volume_factors_for_panel",
    "price_volume_lookback_days",
]

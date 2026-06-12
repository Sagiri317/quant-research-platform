"""A-share-specific factor calculations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.research.factors.price_volume import (
    PRICE_VOLUME_FACTOR_FAMILY,
    calculate_price_volume_factors,
)


def calculate_factors_for_panel(
    panel_frame: pd.DataFrame,
    *,
    definitions: Sequence[FactorDefinition],
) -> pd.DataFrame:
    requested = tuple(definitions)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return panel_frame.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    if families != (PRICE_VOLUME_FACTOR_FAMILY,):
        unsupported = next(
            family for family in families if family != PRICE_VOLUME_FACTOR_FAMILY
        )
        raise ValueError(f"unsupported_a_share_factor_family: {unsupported}")
    return calculate_price_volume_factors(
        panel_frame,
        factor_columns=tuple(definition.output_column for definition in requested),
    )


__all__ = ["calculate_factors_for_panel"]

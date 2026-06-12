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
    _require_factor_input_columns(panel_frame, requested)
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


def _require_factor_input_columns(
    panel_frame: pd.DataFrame,
    definitions: Sequence[FactorDefinition],
) -> None:
    for definition in definitions:
        missing = [
            column
            for column in definition.input_columns
            if column not in panel_frame.columns
        ]
        if missing:
            raise ValueError(
                "missing_a_share_factor_input_columns: "
                f"{definition.family}: {', '.join(missing)}"
            )


__all__ = ["calculate_factors_for_panel"]

"""A-share-specific label calculations."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)


def calculate_labels_for_panel(
    panel_frame: pd.DataFrame,
    *,
    definitions: Sequence[LabelDefinition],
) -> pd.DataFrame:
    requested = tuple(definitions)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return panel_frame.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    if families != ("forward_return",):
        unsupported = next(family for family in families if family != "forward_return")
        raise ValueError(f"unsupported_a_share_label_family: {unsupported}")
    return add_forward_return_labels(
        panel_frame,
        horizons=tuple(definition.horizon for definition in requested),
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
    )


__all__ = ["calculate_labels_for_panel"]

"""A-share-specific label calculations."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)

from .definitions import a_share_forward_return_horizons


def calculate_forward_return_labels_for_panel(
    panel_frame: pd.DataFrame,
) -> pd.DataFrame:
    return add_forward_return_labels(
        panel_frame,
        horizons=a_share_forward_return_horizons(),
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
    )


__all__ = ["calculate_forward_return_labels_for_panel"]

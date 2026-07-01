"""A-share processed-data diagnostics configuration."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

_MASK_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "eligible",
    "can_enter",
    "can_exit",
    "is_st",
    "is_pt",
    "limit_status",
)


def schema_inventory_important_columns(
    *,
    label_horizons: Iterable[int] = (1, 5, 10, 20, 60, 120),
    label_frequency: DataFrequency = DataFrequency.DAILY,
) -> tuple[str, ...]:
    """Return A-share columns worth counting in private schema inventory."""

    label_columns = tuple(
        forward_return_label_column(horizon, label_frequency)
        for horizon in label_horizons
    )
    return (*_MASK_DIAGNOSTIC_COLUMNS, *label_columns)


__all__ = ["schema_inventory_important_columns"]

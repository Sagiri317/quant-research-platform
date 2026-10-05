"""Column roles used by portable data contracts."""

from __future__ import annotations

from enum import StrEnum


class ColumnRole(StrEnum):
    KEY = "key"
    TIMESTAMP = "timestamp"
    SOURCE = "source"
    FACTOR = "factor"
    LABEL = "label"
    MASK = "mask"
    UNIVERSE = "universe"
    CONTEXT = "context"
    FEATURE = "feature"
    PREDICTION = "prediction"
    PORTFOLIO = "portfolio"


PLATFORM_ASSET_TIME_KEY = ("market_id", "asset_id", "timestamp")

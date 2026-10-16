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


def _validate_asset_time_key(primary_key, column_roles) -> None:
    """Require the real axis consumed by asset-time research operations."""
    if not {"asset_id", "timestamp"}.issubset(primary_key):
        raise ValueError("invalid_research_dataset_primary_key")
    if (
        column_roles.get("asset_id") != ColumnRole.KEY
        or column_roles.get("timestamp") != ColumnRole.TIMESTAMP
    ):
        raise ValueError("invalid_research_dataset_key_roles")

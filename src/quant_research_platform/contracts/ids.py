"""Stable platform identifiers."""

from __future__ import annotations

from typing import NewType

MarketId = NewType("MarketId", str)
AssetId = NewType("AssetId", str)
SourceId = NewType("SourceId", str)
SourceAssetId = NewType("SourceAssetId", str)

__all__ = ["AssetId", "MarketId", "SourceAssetId", "SourceId"]

"""Stable platform identifiers."""

from __future__ import annotations

from typing import NewType

MarketId = NewType("MarketId", str)
SourceId = NewType("SourceId", str)

__all__ = ["MarketId", "SourceId"]

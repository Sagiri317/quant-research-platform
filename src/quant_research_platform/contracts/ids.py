"""Stable platform identifiers."""

from __future__ import annotations

from typing import NewType

MarketId = NewType("MarketId", str)
SourceId = NewType("SourceId", str)


def validate_market_id(value: object) -> MarketId:
    if type(value) is not str or not value:
        raise ValueError("invalid_market_id")
    return MarketId(value)


def validate_source_id(value: object) -> SourceId:
    if type(value) is not str or not value:
        raise ValueError("invalid_source_id")
    return SourceId(value)

"""Privacy level contract shared by reports, evidence, and runtime summaries."""

from __future__ import annotations

from enum import StrEnum


class PrivacyLevel(StrEnum):
    PUBLIC_SAFE_AGGREGATE = "public_safe_aggregate"
    PRIVATE_ROW_LEVEL = "private_row_level"
    PRIVATE_SUMMARY = "private_summary"


__all__ = ["PrivacyLevel"]

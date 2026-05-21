"""API-facing artifact names and small static templates."""

from __future__ import annotations

FACTOR_CSV_TEMPLATE = "\n".join(
    [
        "timestamp,asset,factor,forward_return,forward_return_1d",
        "2026-01-01,asset_001,0.42,0.015,0.006",
        "2026-01-01,asset_002,-0.13,-0.004,-0.002",
        "2026-01-02,asset_001,0.37,0.008,0.003",
        "2026-01-02,asset_002,-0.09,0.002,0.001",
        "",
    ]
)

"""Portfolio contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioSpec:
    portfolio_id: str
    prediction_id: str
    weight_column: str = "weight"
    contract_version: str = "0.1"


__all__ = ["PortfolioSpec"]

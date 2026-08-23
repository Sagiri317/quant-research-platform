"""Portfolio catalog."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.portfolios import PortfolioSpec


class PortfolioCatalog(ImmutableCatalog[PortfolioSpec]):
    def __init__(self, items: Iterable[PortfolioSpec] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for spec in values:
            if spec.portfolio_id in seen:
                raise ValueError(f"duplicate_portfolio_spec: {spec.portfolio_id}")
            seen.add(spec.portfolio_id)
        super().__init__(sorted(values, key=lambda spec: spec.portfolio_id))

    def get(self, portfolio_id: str) -> PortfolioSpec:
        try:
            return next(
                spec for spec in self._items if spec.portfolio_id == portfolio_id
            )
        except StopIteration as exc:
            raise KeyError(f"unknown_portfolio_spec: {portfolio_id}") from exc


__all__ = ["PortfolioCatalog"]

"""Portfolio catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.portfolios import PortfolioSpec


@dataclass
class PortfolioCatalog:
    _items: dict[str, PortfolioSpec] = field(default_factory=dict)

    def register(self, spec: PortfolioSpec) -> None:
        if spec.portfolio_id in self._items:
            raise ValueError(f"duplicate_portfolio_spec: {spec.portfolio_id}")
        self._items[spec.portfolio_id] = spec

    def get(self, portfolio_id: str) -> PortfolioSpec:
        try:
            return self._items[portfolio_id]
        except KeyError as exc:
            raise KeyError(f"unknown_portfolio_spec: {portfolio_id}") from exc

    def list(self) -> tuple[PortfolioSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["PortfolioCatalog"]

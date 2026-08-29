"""DataFrame portfolio turnover and cost algorithms."""

from __future__ import annotations


def position_change_turnover(
    *,
    previous: dict[str, float],
    current: dict[str, float],
) -> tuple[float, float]:
    buy_turnover = 0.0
    sell_turnover = 0.0
    for asset in previous.keys() | current.keys():
        change = current.get(asset, 0.0) - previous.get(asset, 0.0)
        if change > 0:
            buy_turnover += change
        elif change < 0:
            sell_turnover -= change
    return buy_turnover, sell_turnover

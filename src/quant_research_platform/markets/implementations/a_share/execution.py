"""A-share-specific next-open tradability and historical fee bindings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeeSchedule:
    effective_date: date
    buy_commission_bps: float
    sell_commission_bps: float
    stamp_duty_bps: float
    transfer_fee_bps: float

    def __post_init__(self) -> None:
        if not isinstance(self.effective_date, date):
            raise ValueError("a_share_fee_schedule_invalid")
        values = (
            self.buy_commission_bps,
            self.sell_commission_bps,
            self.stamp_duty_bps,
            self.transfer_fee_bps,
        )
        if any(
            type(value) is not float or not np.isfinite(value) or value < 0.0
            for value in values
        ):
            raise ValueError("a_share_fee_schedule_invalid")


def derive_execution_mask(panel: pd.DataFrame) -> pd.DataFrame:
    """Return only the two future execution facts used by portfolio consumers."""

    required = {
        "market_id",
        "asset_id",
        "timestamp",
        "is_trading",
        "scenario_universe",
        "limit_status",
        "open",
        "high",
        "low",
        "close",
    }
    if not required.issubset(panel.columns):
        raise ValueError("execution_mask_columns_invalid")
    result = panel.sort_values(["asset_id", "timestamp"], kind="mergesort").copy()
    status = pd.to_numeric(result["limit_status"], errors="raise")
    if bool((status.dropna().isin((-1, 0, 1)) == False).any()):  # noqa: E712
        raise ValueError("execution_limit_status_invalid")
    prices = result.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="raise"
    )
    equal_price = (
        prices["open"].eq(prices["high"])
        & prices["open"].eq(prices["low"])
        & prices["open"].eq(prices["close"])
    )
    result["one_word_limit_up"] = status.eq(1) & equal_price
    result["one_word_limit_down"] = status.eq(-1) & equal_price
    grouped = result.groupby("asset_id", sort=False)
    next_trading = grouped["is_trading"].shift(-1).fillna(False).astype(bool)
    next_up = grouped["one_word_limit_up"].shift(-1).fillna(False).astype(bool)
    next_down = grouped["one_word_limit_down"].shift(-1).fillna(False).astype(bool)
    output = result.loc[:, ("market_id", "asset_id", "timestamp")].copy()
    output["can_enter"] = next_trading & ~next_up
    output["can_exit"] = next_trading & ~next_down
    return output.sort_values(
        ["market_id", "asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def fee_schedule_for_date(
    schedules: tuple[FeeSchedule, ...], execution_date: date
) -> FeeSchedule:
    if len({schedule.effective_date for schedule in schedules}) != len(schedules):
        raise ValueError("a_share_fee_schedule_date_duplicate")
    active = [
        schedule for schedule in schedules if schedule.effective_date <= execution_date
    ]
    if not active:
        raise ValueError("a_share_fee_schedule_missing")
    return max(active, key=lambda schedule: schedule.effective_date)

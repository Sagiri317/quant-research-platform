"""提供 A-share research panels 的 price-volume factor 和 label helpers。"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PRICE_VOLUME_WINDOWS: tuple[int, ...] = (20, 60, 120)

FIRST_PRICE_VOLUME_FACTOR_FAMILIES: tuple[str, ...] = (
    "momentum",
    "reversal",
    "volatility",
    "amount_liquidity",
    "turnover",
    "amihud",
    "amplitude",
)


def expected_price_volume_factor_columns(
    *, windows: tuple[int, ...] = DEFAULT_PRICE_VOLUME_WINDOWS
) -> tuple[str, ...]:
    """返回第一批 price-volume factor column contract。"""

    columns: list[str] = []
    for window in windows:
        columns.extend(
            [
                f"mom_{window}d",
                f"rev_{window}d",
                f"vol_{window}d",
                f"avg_amount_{window}d",
                f"turnover_{window}d",
                f"amihud_{window}d",
                f"amplitude_{window}d",
            ]
        )
    return tuple(columns)


def calculate_price_volume_factors(
    frame: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_PRICE_VOLUME_WINDOWS,
    asset_column: str = "stock_code",
    date_column: str = "trade_date",
    price_column: str = "adj_close",
    return_column: str = "return_1d",
    amount_column: str = "amount",
    turnover_column: str = "turnover",
    high_column: str = "high",
    low_column: str = "low",
    prev_close_column: str = "prev_close",
) -> pd.DataFrame:
    """只使用当前和历史 rows 添加 rolling price-volume factors。

    Signals are computed after sorting each stock by trading date. Rolling
    windows include the current row but never future rows; the forward-looking
    part of the table is kept exclusively in label columns.
    """

    result = frame.sort_values([asset_column, date_column]).copy()
    grouped = result.groupby(asset_column, sort=False)

    for window in windows:
        shifted_price = grouped[price_column].shift(window)
        trailing_return = result[price_column] / shifted_price - 1.0
        result[f"mom_{window}d"] = trailing_return
        result[f"rev_{window}d"] = -trailing_return
        rolling_window = window
        result[f"vol_{window}d"] = grouped[return_column].transform(
            lambda series, w=rolling_window: series.rolling(w, min_periods=w).std()
        )
        result[f"avg_amount_{window}d"] = grouped[amount_column].transform(
            lambda series, w=rolling_window: series.rolling(w, min_periods=w).mean()
        )
        result[f"turnover_{window}d"] = grouped[turnover_column].transform(
            lambda series, w=rolling_window: series.rolling(w, min_periods=w).mean()
        )
        amount = pd.to_numeric(result[amount_column], errors="coerce")
        denominator = amount.where(amount != 0.0)
        returns = pd.to_numeric(result[return_column], errors="coerce")
        illiquidity = (returns.abs() / denominator).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        result[f"amihud_{window}d"] = illiquidity.groupby(
            result[asset_column], sort=False
        ).transform(
            lambda series, w=rolling_window: series.rolling(w, min_periods=w).mean()
        )
        amplitude = (result[high_column] - result[low_column]) / result[
            prev_close_column
        ]
        result[f"amplitude_{window}d"] = amplitude.groupby(
            result[asset_column], sort=False
        ).transform(
            lambda series, w=rolling_window: series.rolling(w, min_periods=w).mean()
        )

    return result


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 60, 120),
    asset_column: str = "stock_code",
    date_column: str = "trade_date",
    execution_price_column: str = "adj_open",
) -> pd.DataFrame:
    """按 asset 和 trading-date 顺序添加 forward execution-price labels。

    The entry price is the next execution open. A horizon-N label exits at the
    open after N holding days, matching the evaluation convention where today's
    signal can only trade on the next available execution price.
    """

    result = frame.sort_values([asset_column, date_column]).copy()
    grouped = result.groupby(asset_column, sort=False)

    entry_price = grouped[execution_price_column].shift(-1)
    for horizon in horizons:
        exit_price = grouped[execution_price_column].shift(-(horizon + 1))
        result[f"forward_return_{horizon}d"] = exit_price / entry_price - 1.0

    return result

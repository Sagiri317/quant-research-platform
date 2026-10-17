"""Explicit zero-balance cash leg for synthetic optimizer stock universes."""

import numpy as np
import pandas as pd


def with_cash_asset(stocks):
    assert "is_cash" not in stocks
    cash = {
        column: False if pd.api.types.is_bool_dtype(stocks[column]) else 0.0
        for column in stocks
    }
    cash.update(asset_id="__cash__", adv=np.inf, is_cash=True)
    return pd.concat(
        [stocks.assign(is_cash=False), pd.DataFrame([cash])], ignore_index=True
    )

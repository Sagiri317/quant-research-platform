from __future__ import annotations

import pandas as pd


def make_prediction_frame(*, filter_flip: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    if filter_flip:
        return pd.DataFrame(
            {
                "market_id": ["a_share"] * len(dates),
                "asset_id": ["asset_a"] * len(dates),
                "timestamp": dates,
                "label_available_at": dates,
                "factor_a": [0.0, 1.0, 2.0, 100.0, 101.0, 0.0, 7.0, 8.0],
                "factor_b": [0.0, 1.0, 2.0, 0.0, 0.0, 10.0, 7.0, 8.0],
                "eligible": [True, True, True, False, False, True, True, True],
                "label": [0.0, 1.0, 2.0, 100.0, 101.0, 10.0, 7.0, 8.0],
            }
        )

    factor_a = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0]
    factor_b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    label = [1.0 + 2.0 * value for value in factor_b]
    return pd.DataFrame(
        {
            "market_id": ["a_share"] * len(dates),
            "asset_id": ["asset_a"] * len(dates),
            "timestamp": dates,
            "label_available_at": dates,
            "factor_a": factor_a,
            "factor_b": factor_b,
            "eligible": [True] * len(dates),
            "label": label,
        }
    )

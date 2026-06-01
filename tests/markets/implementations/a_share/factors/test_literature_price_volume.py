from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
    LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID,
    calculate_literature_price_volume_factors,
    expected_literature_price_volume_factor_columns,
    select_literature_price_volume_features,
)


def _toy_panel() -> pd.DataFrame:
    rows = []
    returns = [0.10, -0.05, 0.02, -0.01, 0.04]
    closes = [10.0, 9.5, 9.7, 9.6, 10.0]
    for offset, (ret, close) in enumerate(zip(returns, closes, strict=True)):
        rows.append(
            {
                "stock_code": "000001",
                "trade_date": pd.Timestamp("2024-01-01")
                + pd.offsets.BDay(offset),
                "adj_close": close,
                "adj_open": close - 0.1,
                "return_1d": ret,
                "amount": [100.0, 0.0, 300.0, 400.0, 500.0][offset],
                "turnover": [0.10, 0.20, 0.40, 0.20, 0.10][offset],
                "high": close + 1.0,
                "low": close - 1.0,
                "prev_close": close - ret,
                "float_market_cap": 1_000.0 + offset,
                "total_market_cap": 2_000.0 + offset,
                "forward_return_20d": ret / 10.0,
                "eligible": True,
            }
        )
    return pd.DataFrame(rows)


def test_literature_pack_declares_expected_contract_columns() -> None:
    assert LITERATURE_PRICE_VOLUME_REBUILD_PACK_ID == (
        "a_share_literature_price_volume_rebuild_v1"
    )
    assert LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID == (
        "a_share_literature_price_volume_preset_v1"
    )

    columns = expected_literature_price_volume_factor_columns()

    for expected in (
        "rev_1d",
        "mom_252d",
        "downside_vol_20d",
        "upside_vol_60d",
        "max_return_20d",
        "drawdown_60d",
        "skew_20d",
        "kurt_60d",
        "turnover_chg_20d",
        "turnover_vol_60d",
        "zero_volume_ratio_20d",
        "intraday_range_60d",
        "log_float_market_cap",
        "log_total_market_cap",
    ):
        assert expected in columns


def test_literature_factor_formulas_use_trailing_data_only() -> None:
    result = calculate_literature_price_volume_factors(_toy_panel(), windows=(2,))
    last = result[result["stock_code"] == "000001"].reset_index(drop=True).iloc[-1]

    assert last["rev_1d"] == -0.04
    assert last["rev_2d"] == -(10.0 / 9.7 - 1.0)
    assert last["mom_2d"] == 10.0 / 9.7 - 1.0
    assert last["downside_vol_2d"] == pytest.approx(abs(-0.01) / np.sqrt(2.0))
    assert last["upside_vol_2d"] == pytest.approx(0.04 / np.sqrt(2.0))
    assert last["max_return_2d"] == 0.04
    assert last["min_return_2d"] == -0.01
    assert last["turnover_chg_2d"] == 0.10 / 0.40 - 1.0
    assert last["turnover_vol_2d"] > 0
    assert last["zero_volume_ratio_2d"] == 0.0
    assert last["intraday_range_2d"] > 0
    assert last["log_float_market_cap"] == np.log(1_004.0)
    assert last["log_total_market_cap"] == np.log(2_004.0)


def test_feature_preset_excludes_labels_filters_and_raw_identifiers() -> None:
    frame = calculate_literature_price_volume_factors(_toy_panel(), windows=(2,))
    features, warnings = select_literature_price_volume_features(frame)

    assert "mom_2d" in features
    assert "rev_2d" in features
    assert "forward_return_20d" not in features
    assert "trade_date" not in features
    assert "stock_code" not in features
    assert "eligible" not in features
    assert warnings

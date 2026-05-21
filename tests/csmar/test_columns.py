from __future__ import annotations

import pandas as pd

from quant_research_platform.markets.a_share.csmar.columns import (
    apply_interim_column_rules,
    load_or_create_column_overrides,
)


def test_column_rules_keep_default_and_manual_optional_columns() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "trade_date": ["2024-01-02"],
            "market_type": [4],
            "trading_status_code": [1],
            "filling_flag": [0],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
            "prev_close": [9.9],
            "limit_status": [0],
            "return_1d": [0.01],
            "adj_close": [10.2],
            "adj_open": [10.0],
            "float_market_cap": [1.0],
            "total_market_cap": [2.0],
            "is_limit_up": [False],
            "is_limit_down": [False],
            "is_trading": [True],
            "is_st": [False],
            "is_pt": [False],
            "is_a_share_scope": [True],
            "eligible_base": [True],
            "turnover": [pd.NA],
            "return_with_dividend": [0.01],
            "return_without_dividend": [0.02],
        }
    )

    pruned, report = apply_interim_column_rules(
        frame,
        "daily_market",
        overrides={"daily_market": {"keep": ["return_without_dividend"], "drop": []}},
    )

    assert "return_1d" in pruned.columns
    assert "return_without_dividend" in pruned.columns
    assert "return_with_dividend" not in pruned.columns
    assert report["auto_dropped_columns"]["return_with_dividend"]


def test_load_or_create_column_overrides_writes_documented_template(tmp_path) -> None:
    path = tmp_path / "private_column_overrides.json"

    payload = load_or_create_column_overrides(path)

    assert path.exists()
    assert payload["daily_market"]["keep"] == []
    assert "return_without_dividend" in payload["daily_market"][
        "_available_optional_columns"
    ]

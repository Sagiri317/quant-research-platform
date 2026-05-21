from __future__ import annotations

import pandas as pd

from quant_research_platform.markets.a_share.datasets import build_panel as panels
from quant_research_platform.markets.a_share.datasets.build_panel import (
    build_research_panel,
)
from quant_research_platform.markets.a_share.factors.build_factors import (
    calculate_panel_factors,
)
from quant_research_platform.markets.a_share.validation.research_panel import (
    check_research_panel,
)


def test_build_research_panel_adds_lifecycle_turnover_and_execution_flags() -> None:
    daily = pd.DataFrame(
        {
            "stock_code": ["000001"] * 4,
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "market_type": [4, 4, 4, 4],
            "open": [10.0, 10.2, 10.4, 10.6],
            "high": [10.5, 10.6, 10.8, 11.0],
            "low": [9.8, 10.0, 10.1, 10.3],
            "close": [10.2, 10.4, 10.6, 10.8],
            "prev_close": [9.9, 10.2, 10.4, 10.6],
            "adj_open": [10.0, 10.2, 10.4, 10.6],
            "adj_close": [10.2, 10.4, 10.6, 10.8],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "amount": [10200.0, 11440.0, 12720.0, 14040.0],
            "return_1d": [0.01, 0.02, 0.019, 0.018],
            "limit_status": [0, 1, -1, 0],
            "is_trading": [True, True, True, False],
            "is_st": [False, False, False, False],
            "is_pt": [False, False, False, False],
            "is_limit_up": [False, True, False, False],
            "is_limit_down": [False, False, True, False],
            "eligible_base": [True, True, True, False],
        }
    )
    stock_master = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "stock_name": ["骞冲畨閾惰"],
            "listing_date": pd.to_datetime(["2023-01-01"]),
            "delisting_date": [pd.NaT],
        }
    )
    market_cap = pd.DataFrame(
        {
            "stock_code": ["000001"],
            "share_change_date": pd.to_datetime(["2023-12-31"]),
            "total_shares": [2000.0],
            "a_float_shares": [100_000.0],
        }
    )

    panel = build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=market_cap,
        min_listing_trading_days=2,
    )

    assert panel.loc[0, "listing_age_trading_days"] == 1
    assert panel.loc[1, "eligible_after_listing_age"]
    assert panel.loc[0, "turnover"] == 0.01
    assert not panel.loc[0, "can_enter"]
    assert panel.loc[1, "can_enter"]
    assert not panel.loc[1, "can_exit"]
    assert panel.loc[2, "next_day_is_trading"] is False


def test_research_panel_quality_flags_blocking_issues() -> None:
    panel = pd.DataFrame(
        {
            "stock_code": ["000001", "000001"],
            "trade_date": ["2024-01-02", "2024-01-02"],
            "open": [10.0, 10.0],
            "high": [10.5, 10.5],
            "low": [9.8, 9.8],
            "close": [10.2, 10.2],
            "volume": [1000.0, 1000.0],
            "amount": [10200.0, 10200.0],
            "turnover": [0.01, 0.01],
            "eligible": [True, True],
            "can_enter": [True, True],
            "can_exit": [True, True],
        }
    )

    report = check_research_panel(panel, expected_year=2024)

    issue_counts = {issue.check: issue.count for issue in report.issues}
    assert issue_counts["duplicate_primary_key"] == 2


def test_calculate_panel_factors_adds_expected_columns() -> None:
    rows = []
    for idx in range(8):
        rows.append(
            {
                "stock_code": "000001",
                "trade_date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(idx),
                "adj_open": 10.0 + idx,
                "adj_close": 10.2 + idx,
                "high": 10.5 + idx,
                "low": 9.8 + idx,
                "prev_close": 9.9 + idx,
                "amount": 10000.0 + idx,
                "turnover": 0.01 + idx * 0.001,
                "return_1d": 0.01,
            }
        )
    panel = pd.DataFrame(rows)

    factored = calculate_panel_factors(panel, windows=(3,), horizons=(1, 3))

    assert "mom_3d" in factored.columns
    assert "amihud_3d" in factored.columns
    assert "forward_return_1d" in factored.columns
    assert factored["mom_3d"].notna().sum() == 5
    assert factored["forward_return_1d"].notna().sum() == 6


def test_build_panel_full_from_interim_writes_one_full_output(tmp_path) -> None:
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    (interim_root / "stock_master").mkdir(parents=True)
    (interim_root / "market_cap").mkdir(parents=True)
    (interim_root / "daily_market_by_year").mkdir(parents=True)
    pd.DataFrame(
        {
            "stock_code": ["000001"],
            "stock_name": ["平安银行"],
            "listing_date": ["2023-01-01"],
            "delisting_date": [""],
        }
    ).to_csv(
        interim_root / "stock_master" / "stock_master.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "stock_code": ["000001"],
            "share_change_date": ["2023-12-31"],
            "total_shares": [2000.0],
            "a_float_shares": [100_000.0],
        }
    ).to_csv(
        interim_root / "market_cap" / "market_cap_to_20251231.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_row = {
        "stock_code": "000001",
        "market_type": 4,
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "prev_close": 9.9,
        "adj_open": 10.0,
        "adj_close": 10.2,
        "volume": 1000.0,
        "amount": 10200.0,
        "return_1d": 0.01,
        "limit_status": 0,
        "is_trading": True,
        "is_st": False,
        "is_pt": False,
        "is_limit_up": False,
        "is_limit_down": False,
        "eligible_base": True,
    }
    pd.DataFrame(
        {
            **{key: [value] for key, value in base_row.items()},
            "trade_date": ["2024-01-02"],
        }
    ).to_csv(
        interim_root / "daily_market_by_year" / "daily_market_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            **{key: [value] for key, value in base_row.items()},
            "trade_date": ["2025-01-02"],
        }
    ).to_csv(
        interim_root / "daily_market_by_year" / "daily_market_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_path, rows = panels.build_panel_full_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2024,
        end_year=2025,
        end_date="2025-12-31",
    )

    assert output_path == (
        processed_root / "panel_full" / "research_panel_2024_2025.parquet"
    )
    assert rows == 2
    loaded = pd.read_parquet(output_path, engine="pyarrow")
    assert loaded["trade_date"].dt.year.tolist() == [2024, 2025]


def test_build_panel_full_from_interim_keeps_cross_year_state(tmp_path) -> None:
    interim_root = tmp_path / "interim"
    processed_root = tmp_path / "processed"
    (interim_root / "stock_master").mkdir(parents=True)
    (interim_root / "market_cap").mkdir(parents=True)
    (interim_root / "daily_market_by_year").mkdir(parents=True)
    pd.DataFrame(
        {
            "stock_code": ["000001"],
            "stock_name": ["平安银行"],
            "listing_date": ["2023-01-01"],
            "delisting_date": [""],
        }
    ).to_csv(
        interim_root / "stock_master" / "stock_master.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "stock_code": ["000001"],
            "share_change_date": ["2023-12-31"],
            "total_shares": [2000.0],
            "a_float_shares": [100_000.0],
        }
    ).to_csv(
        interim_root / "market_cap" / "market_cap_to_20251231.csv",
        index=False,
        encoding="utf-8-sig",
    )
    base_row = {
        "stock_code": "000001",
        "market_type": 4,
        "open": 10.0,
        "high": 10.5,
        "low": 9.8,
        "close": 10.2,
        "prev_close": 9.9,
        "adj_open": 10.0,
        "adj_close": 10.2,
        "volume": 1000.0,
        "amount": 10200.0,
        "return_1d": 0.01,
        "limit_status": 0,
        "is_trading": True,
        "is_st": False,
        "is_pt": False,
        "is_limit_up": False,
        "is_limit_down": False,
        "eligible_base": True,
    }
    pd.DataFrame(
        {
            **{key: [value, value] for key, value in base_row.items()},
            "trade_date": ["2024-12-30", "2024-12-31"],
        }
    ).to_csv(
        interim_root / "daily_market_by_year" / "daily_market_2024.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            **{key: [value, value] for key, value in base_row.items()},
            "trade_date": ["2025-01-02", "2025-01-03"],
        }
    ).to_csv(
        interim_root / "daily_market_by_year" / "daily_market_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )

    output_path, _rows = panels.build_panel_full_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=2024,
        end_year=2025,
        end_date="2025-12-31",
        min_listing_trading_days=2,
    )

    loaded = pd.read_parquet(output_path, engine="pyarrow")
    year_end = loaded[loaded["trade_date"].eq(pd.Timestamp("2024-12-31"))].iloc[0]
    year_start = loaded[loaded["trade_date"].eq(pd.Timestamp("2025-01-02"))].iloc[0]
    assert year_end["next_day_is_trading"]
    assert year_end["can_enter"]
    assert year_start["listing_age_trading_days"] == 3
    assert year_start["eligible_after_listing_age"]

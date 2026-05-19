from __future__ import annotations

import pandas as pd
from validation.common import has_blocking_issues
from validation.internal_tables import check_benchmark_index_market, check_daily_market


def test_daily_market_quality_flags_blocking_issues() -> None:
    frame = pd.DataFrame(
        {
            "stock_code": ["000001", "000001", "000002"],
            "trade_date": ["2024-01-02", "2024-01-02", "2025-01-02"],
            "open": [10.0, 10.0, 20.0],
            "high": [10.5, 9.0, 20.5],
            "low": [9.8, 9.8, 19.5],
            "close": [10.2, 10.2, 20.2],
            "volume": [1000.0, 1000.0, -1.0],
            "amount": [10200.0, 10200.0, 40400.0],
            "float_market_cap": [1_000_000.0, 1_000_000.0, 2_000_000.0],
            "total_market_cap": [2_000_000.0, 2_000_000.0, 4_000_000.0],
            "is_a_share_scope": [True, False, True],
            "is_trading": [True, True, True],
            "is_st": [False, False, True],
            "is_pt": [False, False, False],
            "is_limit_up": [False, False, True],
            "is_limit_down": [False, False, False],
            "filling_flag": [0, 0, 0],
            "limit_status": [0, 0, 1],
        }
    )

    report = check_daily_market(frame, expected_year=2024)

    issue_counts = {issue.check: issue.count for issue in report.issues}
    assert issue_counts["duplicate_primary_key"] == 2
    assert issue_counts["date_outside_expected_year"] == 1
    assert issue_counts["ohlc_violation"] == 1
    assert issue_counts["negative_liquidity_or_market_cap"] == 1
    assert issue_counts["outside_a_share_scope"] == 1
    assert has_blocking_issues(report.issues)


def test_benchmark_index_market_quality_accepts_clean_frame() -> None:
    frame = pd.DataFrame(
        {
            "index_code": ["000300", "000300"],
            "trade_date": ["2024-01-02", "2024-01-03"],
            "open": [3000.0, 3010.0],
            "high": [3020.0, 3030.0],
            "low": [2990.0, 3000.0],
            "close": [3010.0, 3020.0],
            "volume_10k_shares": [100.0, 110.0],
            "amount_10k_cny": [200.0, 220.0],
            "return_1d": [0.01, 0.0033],
        }
    )

    report = check_benchmark_index_market(frame, expected_year=2024)

    assert report.row_count == 2
    assert report.issues == []
    assert not has_blocking_issues(report.issues)


def test_benchmark_index_quality_does_not_block_on_flagged_ohlc() -> None:
    frame = pd.DataFrame(
        {
            "index_code": ["000300", "000905"],
            "trade_date": ["2024-01-02", "2024-01-02"],
            "open": [3000.0, 5000.0],
            "high": [3020.0, 4990.0],
            "low": [2990.0, 4980.0],
            "close": [3010.0, 5005.0],
            "volume_10k_shares": [100.0, 110.0],
            "amount_10k_cny": [200.0, 220.0],
            "return_1d": [0.01, 0.0033],
            "ohlc_valid": [True, False],
        }
    )

    report = check_benchmark_index_market(frame, expected_year=2024)

    issue_counts = {issue.check: issue.count for issue in report.issues}
    assert issue_counts["flagged_ohlc_invalid"] == 1
    assert not has_blocking_issues(report.issues)

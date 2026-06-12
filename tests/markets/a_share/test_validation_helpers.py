from __future__ import annotations

import pandas as pd

from quant_research_platform.markets.implementations.a_share.validation import (
    internal_tables,
    quality_report,
)


def test_quality_report_helpers_cover_common_issue_paths() -> None:
    issues = [
        quality_report.QualityIssue(
            dataset="daily_market",
            check="duplicate",
            severity="blocking",
            count=1,
            message="duplicate",
        ),
        quality_report.QualityIssue(
            dataset="daily_market",
            check="warning",
            severity="warning",
            count=99,
            message="warning",
        ),
    ]
    assert quality_report.has_blocking_issues(issues) is True
    assert quality_report.date_bounds(pd.to_datetime(pd.Series([pd.NaT])))[0] is None

    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "open": [10.0, 10.0],
            "high": [9.0, 11.0],
            "low": [9.5, 9.5],
            "close": [10.5, 10.5],
            "volume": [1.0, -1.0],
            "flag": [True, False],
        }
    )

    duplicate = quality_report.duplicate_issue(
        frame,
        "daily_market",
        ["market_id", "asset_id", "timestamp"],
    )
    missing = quality_report.duplicate_issue(frame, "daily_market", ["missing"])
    year = quality_report.expected_year_issue(
        frame["timestamp"],
        "daily_market",
        2025,
    )
    ohlc = quality_report.ohlc_issue(frame, "daily_market")
    negative = quality_report.negative_value_issue(
        frame,
        "daily_market",
        ["volume"],
        "negative_liquidity",
    )

    assert duplicate is not None and duplicate.check == "duplicate_primary_key"
    assert missing is not None and missing.check == "missing_primary_key_columns"
    assert year is not None and year.check == "date_outside_expected_year"
    assert ohlc is not None and ohlc.check == "ohlc_violation"
    assert negative is not None and negative.check == "negative_liquidity"
    assert quality_report.value_counts(frame, "flag") == {"True": 1, "False": 1}
    assert quality_report.value_counts(frame, "missing") == {}


def test_internal_daily_market_validation_excludes_annotation_fields() -> None:
    forbidden = {
        "eligible_base",
        "eligible",
        "can_enter",
        "can_exit",
        "next_day_is_trading",
        "next_day_limit_status",
    }
    assert forbidden.isdisjoint(internal_tables.DAILY_MARKET_REQUIRED_COLUMNS)

    frame = _daily_market_validation_frame().assign(
        is_a_share_scope=[True, False],
        volume=[1.0, -1.0],
    )
    report = internal_tables.check_daily_market(
        frame,
        expected_year=2024,
        file="synthetic.parquet",
    )
    checks = {issue.check for issue in report.issues}

    assert report.dataset == "daily_market"
    assert report.min_date == "2024-01-02"
    assert report.max_date == "2024-01-03"
    assert "outside_a_share_scope" in checks
    assert "negative_liquidity_or_market_cap" in checks


def test_internal_benchmark_validation_uses_flag_or_ohlc_checks() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["bench"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "return_1d": [0.01],
            "volume_10k_shares": [-1.0],
            "amount_10k_cny": [1.0],
            "ohlc_valid": [False],
        }
    )

    report = internal_tables.check_benchmark_index_market(
        frame,
        expected_year=2025,
    )
    checks = {issue.check for issue in report.issues}

    assert "flagged_ohlc_invalid" in checks
    assert "date_outside_expected_year" in checks
    assert "negative_liquidity" in checks


def _daily_market_validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.5, 10.5],
            "volume": [1.0, 1.0],
            "amount": [100.0, 100.0],
            "return_1d": [0.01, 0.01],
            "adj_close": [10.5, 10.5],
            "adj_open": [10.0, 10.0],
            "float_market_cap": [1000.0, 1000.0],
            "total_market_cap": [2000.0, 2000.0],
            "is_a_share_scope": [True, True],
            "is_trading": [True, False],
            "is_st": [False, False],
            "is_pt": [False, False],
            "limit_status": [0, 1],
            "filling_flag": [0, 1],
        }
    )

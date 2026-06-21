"""Validation checks for source-normalized A-share interim tables."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.data import DataFrequency, data_frequency_slug

from .quality_report import (
    QualityIssue,
    QualityReport,
    date_bounds,
    duplicate_issue,
    expected_year_issue,
    missing_columns_issue,
    negative_value_issue,
    ohlc_issue,
    to_datetime,
    value_counts,
)

MARKET_BARS_DATA_ID = f"market_bars_{data_frequency_slug(DataFrequency.DAILY)}"
MARKET_BARS_REQUIRED_COLUMNS = (
    "market_id",
    "asset_id",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "period_return",
    "adj_close",
    "adj_open",
    "float_market_cap",
    "total_market_cap",
    "is_a_share_scope",
)

BENCHMARK_INDEX_REQUIRED_COLUMNS = (
    "market_id",
    "asset_id",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "period_return",
)


def check_market_bars(
    frame: pd.DataFrame, *, expected_year: int | None = None, file: str = ""
) -> QualityReport:
    """Validate one normalized market-bars interim frame."""

    result = frame.copy()
    result["timestamp"] = to_datetime(result["timestamp"])
    issues = [
        issue
        for issue in (
            missing_columns_issue(
                result,
                MARKET_BARS_DATA_ID,
                MARKET_BARS_REQUIRED_COLUMNS,
            ),
            duplicate_issue(
                result,
                MARKET_BARS_DATA_ID,
                ["market_id", "asset_id", "timestamp"],
            ),
            expected_year_issue(
                result["timestamp"],
                MARKET_BARS_DATA_ID,
                expected_year,
            ),
            ohlc_issue(result, MARKET_BARS_DATA_ID),
            negative_value_issue(
                result,
                MARKET_BARS_DATA_ID,
                ["volume", "amount", "float_market_cap", "total_market_cap"],
                "negative_liquidity_or_market_cap",
            ),
        )
        if issue is not None
    ]
    if "is_a_share_scope" in result.columns:
        outside_scope = int((~result["is_a_share_scope"].astype(bool)).sum())
        if outside_scope:
            issues.append(
                QualityIssue(
                    dataset=MARKET_BARS_DATA_ID,
                    check="outside_a_share_scope",
                    severity="blocking",
                    count=outside_scope,
                    message="Rows are outside the first-version A-share scope.",
                )
            )

    min_date, max_date = date_bounds(result["timestamp"])
    return QualityReport(
        dataset=MARKET_BARS_DATA_ID,
        file=file,
        row_count=len(result),
        column_count=len(result.columns),
        expected_year=expected_year,
        min_date=min_date,
        max_date=max_date,
        issues=issues,
        stats={
            "is_trading_counts": value_counts(result, "is_trading"),
            "is_st_counts": value_counts(result, "is_st"),
            "is_pt_counts": value_counts(result, "is_pt"),
            "limit_status_counts": value_counts(result, "limit_status"),
            "filling_flag_counts": value_counts(result, "filling_flag"),
        },
    )


def check_benchmark_index_market(
    frame: pd.DataFrame, *, expected_year: int | None = None, file: str = ""
) -> QualityReport:
    """检查单个标准化 benchmark index interim frame。"""

    result = frame.copy()
    result["timestamp"] = to_datetime(result["timestamp"])
    ohlc_validation_issue = None
    if "ohlc_valid" in result.columns:
        invalid_count = int((~result["ohlc_valid"].astype(bool)).sum())
        if invalid_count:
            ohlc_validation_issue = QualityIssue(
                dataset="benchmark_index_market",
                check="flagged_ohlc_invalid",
                severity="warning",
                count=invalid_count,
                message="Rows have ohlc_valid=False and should not feed OHLC factors.",
            )
    else:
        ohlc_validation_issue = ohlc_issue(result, "benchmark_index_market")

    issues = [
        issue
        for issue in (
            missing_columns_issue(
                result,
                "benchmark_index_market",
                BENCHMARK_INDEX_REQUIRED_COLUMNS,
            ),
            duplicate_issue(
                result,
                "benchmark_index_market",
                ["market_id", "asset_id", "timestamp"],
            ),
            expected_year_issue(
                result["timestamp"], "benchmark_index_market", expected_year
            ),
            ohlc_validation_issue,
            negative_value_issue(
                result,
                "benchmark_index_market",
                ["volume_10k_shares", "amount_10k_cny"],
                "negative_liquidity",
            ),
        )
        if issue is not None
    ]
    min_date, max_date = date_bounds(result["timestamp"])
    return QualityReport(
        dataset="benchmark_index_market",
        file=file,
        row_count=len(result),
        column_count=len(result.columns),
        expected_year=expected_year,
        min_date=min_date,
        max_date=max_date,
        issues=issues,
        stats={},
    )

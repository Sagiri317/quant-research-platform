"""提供 processed research panel tables 的 validation checks。"""

from __future__ import annotations

import pandas as pd

from ..catalog import (
    a_share_price_volume_panel_input_columns,
)
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

PANEL_REQUIRED_COLUMNS = (
    "market_id",
    "asset_id",
    "timestamp",
    "eligible",
    "can_enter",
    "can_exit",
    "listing_date",
    "turnover",
    *a_share_price_volume_panel_input_columns(),
)


def check_research_panel(
    frame: pd.DataFrame, *, expected_year: int | None = None, file: str = ""
) -> QualityReport:
    """检查单个 yearly research panel。"""

    result = frame.copy()
    result["timestamp"] = to_datetime(result["timestamp"])
    issues = [
        issue
        for issue in (
            missing_columns_issue(result, "research_panel", PANEL_REQUIRED_COLUMNS),
            duplicate_issue(
                result,
                "research_panel",
                ["market_id", "asset_id", "timestamp"],
            ),
            expected_year_issue(result["timestamp"], "research_panel", expected_year),
            ohlc_issue(result, "research_panel"),
            negative_value_issue(
                result,
                "research_panel",
                ["volume", "amount", "turnover"],
                "negative_liquidity",
            ),
        )
        if issue is not None
    ]
    if "listing_date" in result.columns:
        missing_listing = int(result["listing_date"].isna().sum())
        if missing_listing:
            issues.append(
                QualityIssue(
                    dataset="research_panel",
                    check="missing_listing_date",
                    severity="blocking",
                    count=missing_listing,
                    message="Rows cannot be matched to stock_master listing_date.",
                )
            )
    min_date, max_date = date_bounds(result["timestamp"])
    return QualityReport(
        dataset="research_panel",
        file=file,
        row_count=len(result),
        column_count=len(result.columns),
        expected_year=expected_year,
        min_date=min_date,
        max_date=max_date,
        issues=issues,
        stats={
            "eligible_counts": value_counts(result, "eligible"),
            "can_enter_counts": value_counts(result, "can_enter"),
            "can_exit_counts": value_counts(result, "can_exit"),
        },
    )

"""提供 slim factor tables 的 validation checks。"""

from __future__ import annotations

import pandas as pd

from ..csmar.columns import (
    FACTOR_COLUMN_PREFIXES,
    SLIM_FACTOR_BASE_COLUMNS,
)
from .quality_report import (
    QualityIssue,
    QualityReport,
    date_bounds,
    duplicate_issue,
    expected_year_issue,
    missing_columns_issue,
    to_datetime,
    value_counts,
)

LABEL_PREFIX = "forward_return_"
FORBIDDEN_SLIM_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_close",
    "stock_name",
    "listing_date",
    "csrc_2012_industry_name",
)


def factor_columns(frame: pd.DataFrame) -> list[str]:
    """返回排除 forward labels 的 factor columns。"""

    return [
        column
        for column in frame.columns
        if column.startswith(FACTOR_COLUMN_PREFIXES)
        and not column.startswith(LABEL_PREFIX)
    ]


def label_columns(frame: pd.DataFrame) -> list[str]:
    """返回 forward-return label columns。"""

    return [column for column in frame.columns if column.startswith(LABEL_PREFIX)]


def check_factor_table(
    frame: pd.DataFrame, *, expected_year: int | None = None, file: str = ""
) -> QualityReport:
    """检查单个 slim factor table。

    expected_year is optional because current production artifacts are
    full-sample Parquet files. It remains available for narrow diagnostics or
    historical one-year fixtures.
    """

    result = frame.copy()
    result["trade_date"] = to_datetime(result["trade_date"])
    factors = factor_columns(result)
    labels = label_columns(result)
    issues = [
        issue
        for issue in (
            missing_columns_issue(result, "factor_table", SLIM_FACTOR_BASE_COLUMNS),
            duplicate_issue(result, "factor_table", ["stock_code", "trade_date"]),
            expected_year_issue(result["trade_date"], "factor_table", expected_year),
        )
        if issue is not None
    ]
    if not factors:
        issues.append(
            QualityIssue(
                dataset="factor_table",
                check="missing_factor_columns",
                severity="blocking",
                count=1,
                message="No factor columns were found.",
            )
        )
    if not labels:
        issues.append(
            QualityIssue(
                dataset="factor_table",
                check="missing_forward_label_columns",
                severity="blocking",
                count=1,
                message="No forward return label columns were found.",
            )
        )
    forbidden_present = [
        column for column in FORBIDDEN_SLIM_COLUMNS if column in result
    ]
    if forbidden_present:
        issues.append(
            QualityIssue(
                dataset="factor_table",
                check="panel_only_columns_in_slim_factor",
                severity="blocking",
                count=len(forbidden_present),
                message=(
                    "Slim factor table contains panel-only columns: "
                    f"{forbidden_present}."
                ),
            )
        )
    min_date, max_date = date_bounds(result["trade_date"])
    rows = len(result)
    non_null_rates = {
        column: float(result[column].notna().mean()) if rows else 0.0
        for column in [*factors, *labels]
    }
    return QualityReport(
        dataset="factor_table",
        file=file,
        row_count=rows,
        column_count=len(result.columns),
        expected_year=expected_year,
        min_date=min_date,
        max_date=max_date,
        issues=issues,
        stats={
            "factor_count": len(factors),
            "label_count": len(labels),
            "eligible_counts": value_counts(result, "eligible"),
            "can_enter_counts": value_counts(result, "can_enter"),
            "can_exit_counts": value_counts(result, "can_exit"),
            "non_null_rates": non_null_rates,
        },
    )

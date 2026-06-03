"""Validation checks for split A-share factor processed data."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_factor_catalog,
    a_share_label_catalog,
    a_share_processed_catalog,
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

FORBIDDEN_FACTOR_TABLE_COLUMNS = (
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
    """Return registered factor columns present in a split factor table."""

    registered = {
        definition.output_column for definition in a_share_factor_catalog().list()
    }
    return [column for column in frame.columns if column in registered]


def label_columns(frame: pd.DataFrame) -> list[str]:
    """Return registered label columns present in a split factor table."""

    registered = {
        definition.output_column for definition in a_share_label_catalog().list()
    }
    return [column for column in frame.columns if column in registered]


def check_factor_table(
    frame: pd.DataFrame, *, expected_year: int | None = None, file: str = ""
) -> QualityReport:
    """Validate a split price-volume factor table against catalog/spec columns."""

    spec = a_share_processed_catalog().get("price_volume_56")
    result = frame.copy()
    result["timestamp"] = to_datetime(result["timestamp"])
    factors = factor_columns(result)
    labels = label_columns(result)
    issues = [
        issue
        for issue in (
            missing_columns_issue(result, "factor_table", _factor_base_columns()),
            duplicate_issue(result, "factor_table", list(spec.primary_key)),
            expected_year_issue(result["timestamp"], "factor_table", expected_year),
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
                message="No registered factor columns were found.",
            )
        )
    if labels:
        issues.append(
            QualityIssue(
                dataset="factor_table",
                check="label_columns_in_factor_table",
                severity="blocking",
                count=len(labels),
                message="Factor table contains split label columns.",
            )
        )
    forbidden_present = [
        column for column in FORBIDDEN_FACTOR_TABLE_COLUMNS if column in result
    ]
    if forbidden_present:
        issues.append(
            QualityIssue(
                dataset="factor_table",
                check="panel_only_columns_in_factor_table",
                severity="blocking",
                count=len(forbidden_present),
                message=(
                    "Factor table contains panel-only columns: "
                    f"{forbidden_present}."
                ),
            )
        )
    min_date, max_date = date_bounds(result["timestamp"])
    rows = len(result)
    non_null_rates = {
        column: float(result[column].notna().mean()) if rows else 0.0
        for column in factors
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


def _factor_base_columns() -> tuple[str, ...]:
    spec = a_share_processed_catalog().get("price_volume_56")
    return tuple(
        column
        for column in spec.required_columns
        if spec.column_roles[column] is not ColumnRole.FACTOR
    )

"""Validation checks for split A-share factor and label processed data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
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
)

FORBIDDEN_FACTOR_TABLE_COLUMNS = (
    "stock_code",
    "trade_date",
    "index_code",
    "source_id",
    "source_asset_id",
    "session_date",
    "available_at",
    "eligible",
    "can_enter",
    "can_exit",
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
        definition.output_column
        for definition in factor_definitions.a_share_factor_catalog().list()
    }
    return [column for column in frame.columns if column in registered]


def label_columns(frame: pd.DataFrame) -> list[str]:
    """Return registered label columns present in a split factor table."""

    registered = {
        definition.output_column
        for definition in label_definitions.a_share_label_catalog().list()
    }
    return [column for column in frame.columns if column in registered]


def check_factor_table(
    frame: pd.DataFrame, *, expected_year: int | None = None, file: str = ""
) -> QualityReport:
    """Validate a split price-volume factor table against catalog/spec columns."""

    spec = a_share_processed_catalog().get(PRICE_VOLUME_PROCESSED_DATA_ID)
    result = frame.copy()
    timestamps = _timestamp_series(result)
    if "timestamp" in result:
        result["timestamp"] = timestamps
    factors = factor_columns(result)
    labels = label_columns(result)
    issues = [
        issue
        for issue in (
            missing_columns_issue(result, "factor_table", spec.required_columns),
            duplicate_issue(result, "factor_table", list(spec.primary_key)),
            expected_year_issue(timestamps, "factor_table", expected_year),
            _unexpected_columns_issue(
                result,
                dataset="factor_table",
                expected_columns=spec.required_columns,
                check="unexpected_columns_in_factor_table",
            ),
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
    non_finite_count = _non_finite_numeric_count(result, factors)
    if non_finite_count:
        issues.append(
            QualityIssue(
                dataset="factor_table",
                check="non_finite_factor_values",
                severity="blocking",
                count=non_finite_count,
                message="Factor table contains non-finite numeric factor values.",
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
    min_date, max_date = date_bounds(timestamps)
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
            "non_null_rates": non_null_rates,
        },
    )


def check_label_table(
    frame: pd.DataFrame, *, expected_year: int | None = None, file: str = ""
) -> QualityReport:
    """Validate a split forward-return label table against catalog/spec columns."""

    spec = a_share_processed_catalog().get(FORWARD_RETURNS_PROCESSED_DATA_ID)
    result = frame.copy()
    timestamps = _timestamp_series(result)
    if "timestamp" in result:
        result["timestamp"] = timestamps
    labels = label_columns(result)
    factors = factor_columns(result)
    present_labels = [column for column in labels if column in spec.required_columns]
    issues = [
        issue
        for issue in (
            missing_columns_issue(result, "label_table", spec.required_columns),
            duplicate_issue(result, "label_table", list(spec.primary_key)),
            expected_year_issue(timestamps, "label_table", expected_year),
            _unexpected_columns_issue(
                result,
                dataset="label_table",
                expected_columns=spec.required_columns,
                check="unexpected_columns_in_label_table",
            ),
        )
        if issue is not None
    ]
    if factors:
        issues.append(
            QualityIssue(
                dataset="label_table",
                check="factor_columns_in_label_table",
                severity="blocking",
                count=len(factors),
                message="Label table contains split factor columns.",
            )
        )
    non_finite_count = _non_finite_numeric_count(result, present_labels)
    if non_finite_count:
        issues.append(
            QualityIssue(
                dataset="label_table",
                check="non_finite_label_values",
                severity="blocking",
                count=non_finite_count,
                message="Label table contains non-finite numeric label values.",
            )
        )
    min_date, max_date = date_bounds(timestamps)
    rows = len(result)
    non_null_rates = {
        column: float(result[column].notna().mean()) if rows else 0.0
        for column in present_labels
    }
    return QualityReport(
        dataset="label_table",
        file=file,
        row_count=rows,
        column_count=len(result.columns),
        expected_year=expected_year,
        min_date=min_date,
        max_date=max_date,
        issues=issues,
        stats={
            "label_count": len(present_labels),
            "factor_count": len(factors),
            "non_null_rates": non_null_rates,
        },
    )


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return to_datetime(frame["timestamp"])


def _unexpected_columns_issue(
    frame: pd.DataFrame,
    *,
    dataset: str,
    expected_columns: tuple[str, ...],
    check: str,
) -> QualityIssue | None:
    unexpected = sorted(set(frame.columns) - set(expected_columns))
    if not unexpected:
        return None
    return QualityIssue(
        dataset=dataset,
        check=check,
        severity="blocking",
        count=len(unexpected),
        message=f"Unexpected columns: {unexpected}.",
    )


def _non_finite_numeric_count(frame: pd.DataFrame, columns: list[str]) -> int:
    count = 0
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        count += int(np.isinf(numeric).sum())
    return count

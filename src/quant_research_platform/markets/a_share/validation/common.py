"""Common validation report structures and checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class QualityIssue:
    """One validation issue."""

    dataset: str
    check: str
    severity: str
    count: int
    message: str


@dataclass(frozen=True)
class QualityReport:
    """Validation report for one loaded table."""

    dataset: str
    file: str
    row_count: int
    column_count: int
    expected_year: int | None
    min_date: str | None
    max_date: str | None
    issues: list[QualityIssue]
    stats: dict[str, object]


def has_blocking_issues(issues: list[QualityIssue]) -> bool:
    """Return whether any issue should stop the pipeline."""

    return any(issue.severity == "blocking" and issue.count > 0 for issue in issues)


def to_datetime(series: pd.Series) -> pd.Series:
    """Parse a Series as datetimes with invalid values set to NaT."""

    return pd.to_datetime(series, errors="coerce")


def date_bounds(dates: pd.Series) -> tuple[str | None, str | None]:
    """Return min/max date strings for a datetime Series."""

    valid = dates.dropna()
    if valid.empty:
        return None, None
    return valid.min().strftime("%Y-%m-%d"), valid.max().strftime("%Y-%m-%d")


def duplicate_issue(
    frame: pd.DataFrame, dataset: str, primary_key: list[str]
) -> QualityIssue | None:
    """Return a blocking issue for duplicated primary keys."""

    missing = [column for column in primary_key if column not in frame.columns]
    if missing:
        return QualityIssue(
            dataset=dataset,
            check="missing_primary_key_columns",
            severity="blocking",
            count=len(missing),
            message=f"Missing primary key columns: {missing}.",
        )
    duplicated = frame.duplicated(primary_key, keep=False)
    count = int(duplicated.sum())
    if count == 0:
        return None
    return QualityIssue(
        dataset=dataset,
        check="duplicate_primary_key",
        severity="blocking",
        count=count,
        message=f"Duplicated rows on primary key {primary_key}.",
    )


def expected_year_issue(
    dates: pd.Series, dataset: str, expected_year: int | None
) -> QualityIssue | None:
    """Return a blocking issue for rows outside the expected year."""

    if expected_year is None:
        return None
    count = int((dates.dt.year != expected_year).sum())
    if count == 0:
        return None
    return QualityIssue(
        dataset=dataset,
        check="date_outside_expected_year",
        severity="blocking",
        count=count,
        message=f"Rows fall outside expected year {expected_year}.",
    )


def missing_columns_issue(
    frame: pd.DataFrame, dataset: str, columns: tuple[str, ...]
) -> QualityIssue | None:
    """Return a blocking issue when required columns are absent."""

    missing = [column for column in columns if column not in frame.columns]
    if not missing:
        return None
    return QualityIssue(
        dataset=dataset,
        check="missing_required_columns",
        severity="blocking",
        count=len(missing),
        message=f"Missing required columns: {missing}.",
    )


def ohlc_issue(frame: pd.DataFrame, dataset: str) -> QualityIssue | None:
    """Return a blocking issue for invalid OHLC relationships."""

    required = ["open", "high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return None
    price_frame = frame[required].apply(pd.to_numeric, errors="coerce")
    invalid = (
        price_frame["high"] < price_frame[["open", "close"]].max(axis=1)
    ) | (
        price_frame["low"] > price_frame[["open", "close"]].min(axis=1)
    ) | (
        price_frame["high"] < price_frame["low"]
    )
    count = int(invalid.sum())
    if count == 0:
        return None
    return QualityIssue(
        dataset=dataset,
        check="ohlc_violation",
        severity="blocking",
        count=count,
        message="Rows violate high/low/open/close consistency.",
    )


def negative_value_issue(
    frame: pd.DataFrame, dataset: str, columns: list[str], check: str
) -> QualityIssue | None:
    """Return a blocking issue for negative values in selected columns."""

    available = [column for column in columns if column in frame.columns]
    if not available:
        return None
    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    count = int((numeric < 0).any(axis=1).sum())
    if count == 0:
        return None
    return QualityIssue(
        dataset=dataset,
        check=check,
        severity="blocking",
        count=count,
        message=f"Rows contain negative values in {available}.",
    )


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return value counts as a JSON-friendly dict."""

    if column not in frame.columns:
        return {}
    counts = frame[column].value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}

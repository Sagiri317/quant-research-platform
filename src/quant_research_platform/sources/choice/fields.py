"""Canonical Choice disclosure fields."""

from quant_research_platform.contracts.columns import ColumnRole

FIRST_DISCLOSURE_COLUMNS = (
    "asset_id",
    "report_period",
    "first_disclosure_date",
)


def field_roles() -> dict[str, ColumnRole]:
    return {
        "asset_id": ColumnRole.KEY,
        "report_period": ColumnRole.TIMESTAMP,
        "first_disclosure_date": ColumnRole.TIMESTAMP,
    }

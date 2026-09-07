"""Official Choice general-enterprise fields and canonical interim names."""

from quant_research_platform.contracts.columns import ColumnRole

FIRST_DISCLOSURE_COLUMNS = (
    "asset_id",
    "report_period",
    "first_disclosure_date",
)

FINANCIAL_FIELDS = {
    "financial_balance": {
        "TBALANCESTATEMENT_74": "total_assets",
        "TBALANCESTATEMENT_128": "total_liabilities",
        "TBALANCESTATEMENT_140": "parent_equity",
        "TBALANCESTATEMENT_9": "cash",
        "TBALANCESTATEMENT_75": "short_term_borrowings",
        "TBALANCESTATEMENT_88": "current_portion_noncurrent_liabilities",
        "TBALANCESTATEMENT_94": "long_term_borrowings",
        "TBALANCESTATEMENT_95": "bonds_payable",
    },
    "financial_income": {
        "TINCOMESTATEMENT_9": "operating_revenue",
        "TINCOMESTATEMENT_10": "operating_cost",
        "TINCOMESTATEMENT_60": "consolidated_net_income",
        "TINCOMESTATEMENT_61": "parent_net_income",
    },
    "financial_cashflow": {
        "TCASHFLOWSTATEMENT_39": "operating_cash_flow",
        "TCASHFLOWSTATEMENT_49": "capex_cash_paid",
    },
}
FINANCIAL_BASE_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "report_period",
    "report_type",
    "is_corrected",
    "correction_disclosure_date",
    "version_identity",
)


def field_roles() -> dict[str, ColumnRole]:
    return {
        "asset_id": ColumnRole.KEY,
        "report_period": ColumnRole.TIMESTAMP,
        "first_disclosure_date": ColumnRole.KEY,
    }

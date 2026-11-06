from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.sources.csmar import field_mappings


def _adjustment_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    required = (
        "FwardFactor",
        "BwardFactor",
        "CumulateFwardFactor",
        "CumulateBwardFactor",
    )
    all_columns = (*columns, *required)
    return pd.DataFrame([["1"] * len(all_columns)], columns=all_columns)


def test_csmar_source_accepts_provider_excel_like_stock_code_forms() -> None:
    values = pd.Series([1, 1.0, "1", "000001", "000001.0"])

    result = field_mappings.normalize_provider_stock_code_series(values)

    assert result.tolist() == ["000001"] * 5


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.NA,
        float("nan"),
        "",
        "nan",
        "None",
        True,
        -1,
        1.5,
        [],
        {},
        set(),
        1000000,
    ],
)
def test_normalize_provider_stock_code_series_rejects_bad_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_csmar_stock_code"):
        field_mappings.normalize_provider_stock_code_series(pd.Series([value]))


def test_map_columns_rejects_primary_alias_collision() -> None:
    frame = _adjustment_frame(("Symbol", "Stkcd", "TradingDate"))

    with pytest.raises(ValueError, match="ambiguous_csmar_column_mapping"):
        field_mappings.map_columns(frame, "adjustment")


def test_map_columns_rejects_duplicate_provider_headers() -> None:
    frame = _adjustment_frame(("Symbol", "Symbol", "TradingDate"))

    with pytest.raises(ValueError, match="duplicate_csmar_provider_column"):
        field_mappings.map_columns(frame, "adjustment")


def test_adjustment_mapping_facts_keep_aliases_out_of_primary_mapping() -> None:
    assert "Stkcd" not in field_mappings.COLUMN_MAPPINGS["adjustment"]
    assert "Trddt" not in field_mappings.COLUMN_MAPPINGS["adjustment"]


@pytest.mark.parametrize(
    ("stock_column", "date_column"),
    (("Symbol", "TradingDate"), ("Stkcd", "Trddt")),
)
def test_adjustment_primary_only_and_alias_only_map_to_unique_normalized_columns(
    stock_column: str,
    date_column: str,
) -> None:
    result = field_mappings.map_columns(
        _adjustment_frame((stock_column, date_column)),
        "adjustment",
    )

    assert result.columns.is_unique
    assert "stock_code" in result
    assert "trade_date" in result


def test_mapping_collision_precedes_missing_required_column_error() -> None:
    frame = pd.DataFrame([["1", "1"]], columns=["Symbol", "Stkcd"])

    with pytest.raises(ValueError, match="ambiguous_csmar_column_mapping"):
        field_mappings.map_columns(frame, "adjustment")


def test_all_registered_mapping_facts_are_self_consistent() -> None:
    assert set(field_mappings.COLUMN_ALIASES) <= set(field_mappings.COLUMN_MAPPINGS)
    for dataset, mappings in field_mappings.COLUMN_MAPPINGS.items():
        assert len(set(mappings.values())) == len(mappings)
        aliases = field_mappings.COLUMN_ALIASES.get(dataset, {})
        assert set(aliases) <= set(mappings)
        provider_aliases = [name for values in aliases.values() for name in values]
        assert len(set(provider_aliases)) == len(provider_aliases)
        assert not set(provider_aliases) & set(mappings)
        frame = pd.DataFrame(
            columns=tuple(
                dict.fromkeys(
                    (
                        *mappings,
                        *field_mappings.RAW_TABLE_CONTRACTS[dataset].required_columns,
                    )
                )
            )
        )
        result = field_mappings.map_columns(frame, dataset)
        assert result.columns.is_unique


@pytest.mark.parametrize(
    "facts",
    (
        field_mappings.COLUMN_MAPPINGS,
        field_mappings.COLUMN_ALIASES,
    ),
)
def test_registered_mapping_facts_are_deeply_immutable(facts: object) -> None:
    with pytest.raises(TypeError):
        facts["adjustment"]["mutation"] = "forbidden"  # type: ignore[index]


def test_all_required_provider_contracts_and_column_order_are_preserved():
    expected = [
        (
            "stock_master",
            "TRDNEW_Co",
            "Stkcd Stknme Listdt DelistedDate Markettype Nnindcd Nnindnme Statco "
            "Statdt",
        ),
        (
            "market_bars_1d",
            "TRDNEW_Dalyr",
            "Stkcd Trddt Markettype Trdsta Filling Opnprc Hiprc Loprc Clsprc "
            "Dnshrtrd Dnvaltrd Ahshrtrd_D Ahvaltrd_D Dsmvosd Dsmvtl3 Dretwd "
            "Adjprcwd PreClosePrice LimitStatus",
        ),
        ("market_cap", "TRDNEW_Capchg", "Stkcd Shrchgdt Nshrttl Nshra"),
        (
            "adjustment",
            "TRD_AdjustFactor",
            "Symbol TradingDate FwardFactor BwardFactor CumulateFwardFactor "
            "CumulateBwardFactor",
        ),
        (
            "index_market",
            "IDX_Idxtrd",
            "Indexcd Idxtrd01 Idxtrd02 Idxtrd03 Idxtrd04 Idxtrd05 Idxtrd06 "
            "Idxtrd07 Idxtrd08 Idxtrd09",
        ),
        ("index_info", "IDX_Idxinfo", "Indexcd Idxinfo01 Idxinfo11 Idxinfo02"),
        (
            "index_membership",
            "IDX_Chgsmp",
            "Indexcd Chgsmp01 Chgsmp02 Chgsmp03 Chgsmp04 Chgsmp05 Chgsmp06 Chgsmp07",
        ),
        ("index_weights", "IDX_Smprat", "Indexcd Enddt Stkcd Constdnme Weight"),
        (
            "financial_balance",
            "FS_Combas",
            "Stkcd ShortName IfCorrect DeclareDate Accper Typrep A001000000 "
            "A002000000 A003100000 A001101000 A002101000 A002125000 A002201000 "
            "A002203000",
        ),
        (
            "financial_income",
            "FS_Comins",
            "Stkcd ShortName IfCorrect DeclareDate Accper Typrep B001101000 "
            "B001201000 B002000000 B002000101",
        ),
        (
            "financial_cashflow",
            "FS_Comscfd",
            "Stkcd ShortName IfCorrect DeclareDate Accper Typrep C001000000 C002006000",
        ),
        (
            "industry_classification",
            "STK_INDUSTRYCLASS",
            "Symbol IndustryClassificationID ImplementDate IndustryCode IndustryName",
        ),
        (
            "industry_classification_annual",
            "STK_IndustryClassAnl",
            "Symbol EndDate IndustryCode1 IndustryName1",
        ),
        (
            "industry_classification_sets",
            "STK_INDCLASSIFYSETS",
            "INDUSTRYCODE INDCLASSIFYSYSTEMCODE INDUSTRYNAME P_INDUSTRYCODE "
            "INDUSTRYID RANK",
        ),
        (
            "industry_classification_versions",
            "STK_INDCLASSIFYVERSION",
            "IndClassifySystemCode IndClassifySystem ImplementDate AbandonDate "
            "InstitutionName",
        ),
        (
            "listing_status_changes",
            "STK_ITEMCHANGE",
            "Symbol DeclareDate ChangeDate ChangedItem ValueBefore ValueAfter",
        ),
        (
            "stock_info_changes",
            "STK_LISTEDCOINFOCHG",
            "Symbol AnnouncementDate ImplementDate ChangedItem Value_Before "
            "Value_After",
        ),
        (
            "stock_info_annual",
            "STK_LISTEDCOINFOANL",
            "Symbol EndDate IndustryCode IndustryName LISTINGDATE LISTINGSTATE",
        ),
        ("trading_no_limit", "TRD_NoLimit", "TradingDate Symbol Reason"),
        ("special_treatment_changes", "BDT_Trdchg", "Symbol Chgtype Annoudt Execudt"),
        (
            "corporate_actions",
            "TRDNEW_Cptl",
            "Stkcd Disttyp Exdistdt Markettype Annodt Paydt Amount Roprc",
        ),
        ("equity_changes", "FN_FN046", "Stkcd Accper Typrep DataSources SubjectCode"),
        (
            "analyst_forecast",
            "AF_Forecast",
            "Stkcd ReportID AnanmID DeclareDate Fenddt InstitutionID Feps Fpe",
        ),
        ("trading_calendar", "TRDNEW_Cale", "Markettype Clddt Daywk State"),
    ]
    assert tuple(field_mappings.RAW_TABLE_CONTRACTS) == tuple(
        name for name, _, _ in expected
    )
    for name, table_code, columns in expected:
        contract = field_mappings.RAW_TABLE_CONTRACTS[name]
        assert contract.table_code == table_code
        assert contract.required_columns == tuple(columns.split())

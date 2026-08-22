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
    assert field_mappings.normalize_stock_code(values).tolist() == result.tolist()


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


def test_mapping_facts_reject_duplicate_canonical_targets_before_frame_checks() -> None:
    with pytest.raises(ValueError, match="invalid_csmar_column_mapping_facts"):
        field_mappings._validate_mapping_facts(
            {"adjustment": {"Symbol": "stock_code", "TradingDate": "stock_code"}},
            {},
        )


@pytest.mark.parametrize(
    "aliases",
    (
        {"MissingPrimary": ("ProviderAlias",)},
        {"Symbol": ("ProviderAlias",), "TradingDate": ("ProviderAlias",)},
        {"Symbol": ("TradingDate",)},
    ),
)
def test_mapping_facts_reject_invalid_alias_ownership_before_frame_checks(
    aliases: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(ValueError, match="invalid_csmar_column_mapping_facts"):
        field_mappings._validate_mapping_facts(
            {"adjustment": {"Symbol": "stock_code", "TradingDate": "trade_date"}},
            {"adjustment": aliases},
        )


def test_all_registered_mapping_facts_are_self_consistent() -> None:
    for dataset, mappings in field_mappings.COLUMN_MAPPINGS.items():
        frame = pd.DataFrame(columns=tuple(mappings))
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

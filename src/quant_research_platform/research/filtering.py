"""Research sample filtering implementation."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
)


def apply_filter_mode(
    frame: pd.DataFrame,
    *,
    mode: str,
    metadata_key: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = required_filter_columns(mode)
    if mode == "unfiltered":
        filtered = frame
    else:
        keep = _validated_bool_mask(frame, columns[0])
        for column in columns[1:]:
            mask = _validated_bool_mask(frame, column)
            keep &= mask

        filtered = frame.loc[keep]

    return filtered, {
        metadata_key: mode,
        "filter_columns_used": list(columns),
        "filter_definition": filter_definition(mode),
        "rows_input": int(len(frame)),
        "rows_after_filter_mode": int(len(filtered)),
    }


def _validated_bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"missing_filter_columns: {column}")
    series = frame[column]
    if bool(series.isna().any()):
        raise ValueError(f"null_filter_column_values: {column}")
    if is_bool_dtype(series):
        return series.eq(True)
    if is_integer_dtype(series):
        invalid = ~series.isin((0, 1))
        if not bool(invalid.any()):
            return series.eq(1)
    raise ValueError(f"invalid_filter_column_dtype: {column}")


A_SHARE_ELIGIBILITY_FACTS = (
    "is_a_share_scope",
    "is_trading",
    "is_st",
    "is_pt",
    "has_252_trading_sessions",
    "close",
    "adj_close",
    "volume",
    "amount",
    "turnover",
)
_A_SHARE_FUSION_ELIGIBILITY_FACTS = (
    "is_nonfinancial_industry",
    "has_legal_fundamental_pit",
)


def a_share_eligibility(frame: pd.DataFrame, recipe: str) -> pd.Series:
    if recipe not in {"a_share_all", "a_share_nonfinancial", "a_share_fundamental"}:
        raise ValueError("invalid_a_share_eligibility_recipe")
    bool_facts = {
        column: strict_bool_series(
            frame[column],
            error=f"invalid_a_share_eligibility_fact: {column}",
            allow_null=True,
        )
        for column in (
            "is_a_share_scope",
            "is_trading",
            "is_st",
            "is_pt",
            "has_252_trading_sessions",
        )
    }
    numeric_facts = {
        column: strict_finite_numeric_series(
            frame[column],
            error=f"invalid_a_share_eligibility_fact: {column}",
            allow_null=True,
        )
        for column in (
            "close",
            "adj_close",
            "volume",
            "amount",
            "turnover",
        )
    }
    eligible = (
        bool_facts["is_a_share_scope"]
        & bool_facts["is_trading"]
        & ~bool_facts["is_st"]
        & ~bool_facts["is_pt"]
        & bool_facts["has_252_trading_sessions"]
        & numeric_facts["close"].gt(0.0)
        & numeric_facts["adj_close"].gt(0.0)
        & numeric_facts["volume"].gt(0.0)
        & numeric_facts["amount"].gt(0.0)
        & numeric_facts["turnover"].gt(0.0)
    ).fillna(False)
    if recipe in {"a_share_nonfinancial", "a_share_fundamental"}:
        required = (
            ("is_nonfinancial_industry",)
            if recipe == "a_share_nonfinancial"
            else _A_SHARE_FUSION_ELIGIBILITY_FACTS
        )
        for column in required:
            if column not in frame:
                raise ValueError(f"a_share_eligibility_facts_missing: {column}")
            bool_facts[column] = strict_bool_series(
                frame[column],
                error=f"invalid_a_share_eligibility_fact: {column}",
                allow_null=True,
            ).fillna(False)
            eligible &= bool_facts[column]
        for column in (
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
        ):
            if column not in frame:
                raise ValueError(f"a_share_eligibility_facts_missing: {column}")
            values = frame[column].astype("string").str.strip()
            eligible &= values.notna() & values.ne("")
    return eligible.astype(bool)

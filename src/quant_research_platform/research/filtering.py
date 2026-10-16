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


_A_SHARE_FUSION_ELIGIBILITY_FACTS = (
    "is_nonfinancial_industry",
    "has_legal_fundamental_pit",
)


def a_share_eligibility_columns(
    *,
    minimum_listing_sessions: int = 252,
    exclude_st: bool = True,
    exclude_pt: bool = True,
) -> tuple[str, ...]:
    """Select only the facts used by the requested research scope."""
    if (
        type(minimum_listing_sessions) is not int
        or minimum_listing_sessions < 0
        or type(exclude_st) is not bool
        or type(exclude_pt) is not bool
    ):
        raise ValueError("invalid_a_share_eligibility_parameters")
    columns = ["is_a_share_scope"]
    if exclude_st:
        columns.append("is_st")
    if exclude_pt:
        columns.append("is_pt")
    if minimum_listing_sessions:
        columns.append(
            "has_252_trading_sessions"
            if minimum_listing_sessions == 252
            else "listing_age_trading_days"
        )
    return tuple(columns)


A_SHARE_ELIGIBILITY_FACTS = a_share_eligibility_columns()


def a_share_eligibility(
    frame: pd.DataFrame,
    recipe: str,
    *,
    minimum_listing_sessions: int = 252,
    exclude_st: bool = True,
    exclude_pt: bool = True,
) -> pd.Series:
    """Derive research membership without imposing factor or execution validity.

    Unknown relevant facts are retained in the input and never affirmed as eligible.
    """
    if recipe not in {"a_share_all", "a_share_nonfinancial", "a_share_fundamental"}:
        raise ValueError("invalid_a_share_eligibility_recipe")
    columns = a_share_eligibility_columns(
        minimum_listing_sessions=minimum_listing_sessions,
        exclude_st=exclude_st,
        exclude_pt=exclude_pt,
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError("a_share_eligibility_facts_missing: " + ", ".join(missing))
    eligible = pd.Series(True, index=frame.index, dtype="boolean")
    for column in columns:
        if column == "listing_age_trading_days":
            values = strict_finite_numeric_series(
                frame[column],
                error=f"invalid_a_share_eligibility_fact: {column}",
                allow_null=True,
            )
            if ((values.dropna() < 0) | (values.dropna() % 1 != 0)).any():
                raise ValueError(f"invalid_a_share_eligibility_fact: {column}")
            eligible &= values.ge(minimum_listing_sessions)
        else:
            fact = strict_bool_series(
                frame[column],
                error=f"invalid_a_share_eligibility_fact: {column}",
                allow_null=True,
            )
            eligible &= ~fact if column in {"is_st", "is_pt"} else fact
    if recipe in {"a_share_nonfinancial", "a_share_fundamental"}:
        required = (
            ("is_nonfinancial_industry",)
            if recipe == "a_share_nonfinancial"
            else _A_SHARE_FUSION_ELIGIBILITY_FACTS
        )
        for column in required:
            if column not in frame:
                raise ValueError(f"a_share_eligibility_facts_missing: {column}")
            eligible &= strict_bool_series(
                frame[column],
                error=f"invalid_a_share_eligibility_fact: {column}",
                allow_null=True,
            )
        for column in (
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
        ):
            if column not in frame:
                raise ValueError(f"a_share_eligibility_facts_missing: {column}")
            values = frame[column].astype("string").str.strip()
            eligible &= values.notna() & values.ne("")
    return eligible.fillna(False).astype(bool)

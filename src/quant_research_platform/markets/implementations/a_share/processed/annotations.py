"""Pure A-share annotation derivation from the processed market panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.validation import strict_integer_series
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.validation import (
    strict_values,
)


def derive_annotations(
    panel_frame: pd.DataFrame,
    *,
    data_ids: tuple[str, ...] | None = None,
    min_listing_trading_days: int = 120,
) -> dict[str, pd.DataFrame]:
    """Derive registered A-share annotation frames from the market panel."""

    catalog = processed_catalog()
    requested = set(data_ids) if data_ids is not None else None
    if requested is not None:
        annotation_ids = {
            spec.data_id
            for spec in catalog.list()
            if spec.kind is ProcessedDataKind.ANNOTATION
            and spec.processed_input_data_ids
        }
        missing_ids = sorted(requested.difference(annotation_ids))
        if missing_ids:
            raise ValueError("unknown_annotation_data_id: " + ", ".join(missing_ids))
    annotation_frame = _with_annotation_masks(
        panel_frame,
        min_listing_trading_days=min_listing_trading_days,
    )
    outputs: dict[str, pd.DataFrame] = {}
    for spec in catalog.list():
        if spec.kind is not ProcessedDataKind.ANNOTATION:
            continue
        if not spec.processed_input_data_ids:
            continue
        if requested is not None and spec.data_id not in requested:
            continue
        missing = [
            column for column in spec.required_columns if column not in annotation_frame
        ]
        if missing:
            raise ValueError(
                f"missing_annotation_input_columns: {spec.data_id}: "
                + ", ".join(missing)
            )
        outputs[spec.data_id] = (
            annotation_frame.loc[:, list(spec.required_columns)]
            .copy()
            .sort_values(list(spec.primary_key))
            .reset_index(drop=True)
        )
    if requested is not None and set(outputs) != requested:
        missing_ids = sorted(requested.difference(outputs))
        raise ValueError("unknown_annotation_data_id: " + ", ".join(missing_ids))
    return outputs


def _with_annotation_masks(
    panel_frame: pd.DataFrame,
    *,
    min_listing_trading_days: int,
) -> pd.DataFrame:
    result = panel_frame.sort_values(["market_id", "asset_id", "timestamp"]).copy()
    required = (
        "is_a_share_scope",
        "is_trading",
        "is_st",
        "is_pt",
        "listing_age_trading_days",
        "amount",
        "volume",
        "turnover",
    )
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError("missing_annotation_input_columns: " + ", ".join(missing))
    listing_age = strict_integer_series(
        result["listing_age_trading_days"],
        error="invalid_annotation_numeric",
    )
    amount = strict_values.strict_numeric_value_series(
        result["amount"],
        error="invalid_annotation_numeric",
    )
    volume = strict_values.strict_numeric_value_series(
        result["volume"],
        error="invalid_annotation_numeric",
    )
    turnover = strict_values.strict_numeric_value_series(
        result["turnover"],
        error="invalid_annotation_numeric",
        allow_null=True,
    )
    is_scope = strict_values.strict_trading_bool_series(
        result["is_a_share_scope"],
        error="invalid_annotation_bool",
    )
    is_trading = strict_values.strict_trading_bool_series(
        result["is_trading"],
        error="invalid_annotation_bool",
    )
    is_st = strict_values.strict_trading_bool_series(
        result["is_st"],
        error="invalid_annotation_bool",
    )
    is_pt = strict_values.strict_trading_bool_series(
        result["is_pt"],
        error="invalid_annotation_bool",
    )
    result["eligible_after_listing_age"] = (
        listing_age >= min_listing_trading_days
    ).astype(bool)
    result["eligible_liquidity"] = (
        amount.gt(0) & volume.gt(0) & turnover.fillna(-1).ge(0)
    ).astype(bool)
    result["eligible_base"] = (is_scope & is_trading & ~is_st & ~is_pt).astype(bool)
    result["eligible"] = (
        result["eligible_base"]
        & result["eligible_after_listing_age"]
        & result["eligible_liquidity"]
    ).astype(bool)
    result = _add_one_word_limit_masks(result)
    grouped = result.groupby("asset_id", sort=False)
    result["next_bar_is_trading"] = _shifted_bool_mask(grouped["is_trading"])
    result["next_bar_limit_status"] = grouped["limit_status"].shift(-1)
    next_one_word_limit_up = _shifted_bool_mask(grouped["one_word_limit_up"])
    next_one_word_limit_down = _shifted_bool_mask(grouped["one_word_limit_down"])
    result["can_enter"] = (
        result["eligible"]
        & strict_values.strict_trading_bool_series(
            result["next_bar_is_trading"],
            error="invalid_annotation_bool",
        )
        & ~next_one_word_limit_up
    ).astype(bool)
    result["can_exit"] = (
        strict_values.strict_trading_bool_series(
            result["next_bar_is_trading"],
            error="invalid_annotation_bool",
        )
        & ~next_one_word_limit_down
    ).astype(bool)
    return result


def _add_one_word_limit_masks(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    explicit = {"one_word_limit_up", "one_word_limit_down"}
    if explicit.issubset(result.columns):
        result["one_word_limit_up"] = _bool_like_column(result, "one_word_limit_up")
        result["one_word_limit_down"] = _bool_like_column(
            result,
            "one_word_limit_down",
        )
        return result
    if explicit.intersection(result.columns):
        missing = sorted(explicit.difference(result.columns))
        raise ValueError(
            "missing_one_word_limit_explicit_columns: " + ", ".join(missing)
        )
    required = ("open", "high", "low", "close", "limit_status")
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(
            "missing_one_word_limit_inference_columns: " + ", ".join(missing)
        )
    prices = pd.DataFrame(
        {
            column: strict_values.strict_numeric_value_series(
                result[column],
                error="invalid_one_word_limit_inference_values",
            )
            for column in ("open", "high", "low", "close")
        },
        index=result.index,
    )
    status = strict_integer_series(
        result["limit_status"],
        error="invalid_one_word_limit_status",
        allow_null=True,
    )
    invalid_non_null = status.isna() & result["limit_status"].notna()
    invalid_value = status.notna() & ~status.isin([-1, 0, 1])
    if invalid_non_null.any() or invalid_value.any():
        raise ValueError("invalid_one_word_limit_status")
    # Null limit_status is unknown/no signal, not an explicit neutral status.
    status = status.fillna(0)
    equal_prices = (
        np.isclose(prices["open"], prices["high"], rtol=1e-7, atol=1e-8)
        & np.isclose(prices["open"], prices["low"], rtol=1e-7, atol=1e-8)
        & np.isclose(prices["open"], prices["close"], rtol=1e-7, atol=1e-8)
    )
    result["one_word_limit_up"] = status.eq(1) & equal_prices
    result["one_word_limit_down"] = status.eq(-1) & equal_prices
    return result


def _shifted_bool_mask(series: pd.core.groupby.SeriesGroupBy) -> pd.Series:
    shifted = series.shift(-1)
    parsed = strict_values.strict_trading_bool_series(
        shifted.where(shifted.notna(), False),
        error="invalid_annotation_bool",
    )
    return parsed.astype(bool)


def _bool_like_column(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if bool(series.isna().any()):
        raise ValueError(f"null_one_word_limit_column: {column}")
    try:
        return strict_values.strict_trading_bool_series(
            series,
            error=f"invalid_one_word_limit_column: {column}",
        )
    except ValueError:
        numeric = strict_integer_series(
            series,
            error=f"invalid_one_word_limit_column: {column}",
        )
    if not set(numeric.unique()).issubset({0, 1}):
        raise ValueError(f"invalid_one_word_limit_column: {column}")
    return numeric.astype(bool)

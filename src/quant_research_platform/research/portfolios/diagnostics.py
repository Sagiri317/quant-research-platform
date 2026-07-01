"""Research-only portfolio diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def summarize_capacity_proxy(
    weights: pd.DataFrame,
    liquidity: pd.DataFrame,
    *,
    asset_column: str = "asset",
    timestamp_column: str = "timestamp",
    weight_column: str = "weight",
    amount_column: str = "amount",
    float_market_cap_column: str | None = None,
) -> dict[str, float]:
    """Summarize research-only capacity proxy diagnostics."""

    _require_columns(
        weights,
        (asset_column, timestamp_column, weight_column),
        "weights",
    )
    liquidity_columns = [asset_column, timestamp_column, amount_column]
    if float_market_cap_column is not None:
        liquidity_columns.append(float_market_cap_column)
    _require_columns(liquidity, tuple(liquidity_columns), "liquidity")
    _reject_duplicate_keys(liquidity, (asset_column, timestamp_column), "liquidity")

    merged = weights.merge(
        liquidity,
        on=[asset_column, timestamp_column],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if bool(merged["_merge"].ne("both").any()):
        raise ValueError("missing_liquidity_for_weight")

    weight_values = pd.to_numeric(merged[weight_column], errors="coerce")
    amount_values = pd.to_numeric(merged[amount_column], errors="coerce")
    if bool(weight_values.isna().any()) or bool(amount_values.isna().any()):
        raise ValueError("invalid_capacity_proxy_values")

    positive_amount = amount_values.gt(0)
    valid_weights = weight_values[positive_amount]
    valid_amounts = amount_values[positive_amount]
    result = {
        "row_count": float(len(merged)),
        "positive_amount_row_count": float(positive_amount.sum()),
        "excluded_non_positive_amount_count": float((~positive_amount).sum()),
        "gross_weight": float(weight_values.abs().sum()),
        "covered_gross_weight": float(valid_weights.abs().sum()),
        "weighted_amount_proxy": float((valid_weights.abs() * valid_amounts).sum()),
    }
    if float_market_cap_column is not None:
        float_market_cap = pd.to_numeric(
            merged[float_market_cap_column],
            errors="coerce",
        )
        if bool(float_market_cap.isna().any()):
            raise ValueError("invalid_capacity_proxy_values")
        result["weighted_float_market_cap_proxy"] = float(
            (valid_weights.abs() * float_market_cap[positive_amount]).sum()
        )
    return result


def summarize_portfolio_exposures(
    weights: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    asset_column: str = "asset",
    timestamp_column: str = "timestamp",
    weight_column: str = "weight",
    exposure_columns: Sequence[str],
) -> pd.DataFrame:
    """Return timestamp-level weighted exposure diagnostics."""

    if not exposure_columns:
        raise ValueError("exposure_columns_required")
    _require_columns(
        weights,
        (asset_column, timestamp_column, weight_column),
        "weights",
    )
    _require_columns(
        exposures,
        (asset_column, timestamp_column, *tuple(exposure_columns)),
        "exposures",
    )
    _reject_duplicate_keys(exposures, (asset_column, timestamp_column), "exposures")

    merged = weights.merge(
        exposures,
        on=[asset_column, timestamp_column],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if bool(merged["_merge"].ne("both").any()):
        raise ValueError("missing_asset_exposure")

    merged[weight_column] = pd.to_numeric(merged[weight_column], errors="coerce")
    if bool(merged[weight_column].isna().any()):
        raise ValueError("invalid_portfolio_exposure_values")
    records: list[dict[str, object]] = []
    for timestamp, group in merged.groupby(timestamp_column, sort=True, dropna=False):
        record: dict[str, object] = {
            timestamp_column: timestamp,
            "row_count": float(len(group)),
            "gross_weight": float(group[weight_column].abs().sum()),
        }
        for column in exposure_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            if bool(values.isna().any()):
                raise ValueError("invalid_portfolio_exposure_values")
            record[column] = float((group[weight_column] * values).sum())
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=(
            timestamp_column,
            "row_count",
            "gross_weight",
            *tuple(exposure_columns),
        ),
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_{name}_columns: {', '.join(missing)}")


def _reject_duplicate_keys(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    name: str,
) -> None:
    if bool(frame.duplicated(list(keys)).any()):
        raise ValueError(f"duplicate_{name}_keys")


__all__ = ["summarize_capacity_proxy", "summarize_portfolio_exposures"]

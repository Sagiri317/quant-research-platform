"""Production portfolio diagnostics builders."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS,
    PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS,
    PortfolioDiagnosticsConfig,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_hashable_key_series,
    strict_string_series,
    strict_timestamp_series,
)


def build_capacity_diagnostics(
    positions: pd.DataFrame,
    context: pd.DataFrame,
    config: PortfolioDiagnosticsConfig,
    *,
    asset_column: str = "asset",
    context_asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    weight_column: str = "position",
) -> pd.DataFrame:
    """Build capacity proxy diagnostics for portfolio positions."""

    if config.capacity_amount_column is None:
        return pd.DataFrame(columns=list(PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS))
    summary = _summarize_capacity_proxy(
        positions,
        context,
        asset_column=asset_column,
        context_asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        signal_column="signal",
        weight_column=weight_column,
        amount_column=config.capacity_amount_column,
        float_market_cap_column=config.capacity_float_market_cap_column,
    )
    return summary.loc[:, list(PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS)]


def build_numeric_exposure_diagnostics(
    positions: pd.DataFrame,
    context: pd.DataFrame,
    config: PortfolioDiagnosticsConfig,
    *,
    asset_column: str = "asset",
    context_asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    weight_column: str = "position",
) -> pd.DataFrame:
    """Build weighted numeric exposure diagnostics."""

    if not config.numeric_exposure_columns:
        return pd.DataFrame(columns=list(PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS))
    summary = _summarize_numeric_exposures(
        positions,
        context,
        exposure_columns=config.numeric_exposure_columns,
        asset_column=asset_column,
        context_asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        signal_column="signal",
        weight_column=weight_column,
    )
    return summary.loc[:, list(PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS)]


def build_categorical_exposure_diagnostics(
    positions: pd.DataFrame,
    context: pd.DataFrame,
    config: PortfolioDiagnosticsConfig,
    *,
    asset_column: str = "asset",
    context_asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    weight_column: str = "position",
) -> pd.DataFrame:
    """Build categorical exposure diagnostics."""

    if not config.categorical_exposure_columns:
        return pd.DataFrame(
            columns=list(PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS)
        )
    summary = _summarize_categorical_exposures(
        positions,
        context,
        exposure_columns=config.categorical_exposure_columns,
        asset_column=asset_column,
        context_asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        signal_column="signal",
        weight_column=weight_column,
    )
    return summary.loc[:, list(PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS)]


def _summarize_capacity_proxy(
    weights: pd.DataFrame,
    liquidity: pd.DataFrame,
    *,
    asset_column: str,
    context_asset_column: str,
    timestamp_column: str,
    signal_column: str,
    weight_column: str,
    amount_column: str,
    float_market_cap_column: str | None,
) -> pd.DataFrame:
    _require_columns(
        weights,
        (asset_column, timestamp_column, signal_column, weight_column),
        "weights",
    )
    liquidity_columns = [context_asset_column, timestamp_column, amount_column]
    if float_market_cap_column is not None:
        liquidity_columns.append(float_market_cap_column)
    _require_columns(liquidity, tuple(liquidity_columns), "liquidity")
    merged = _merge_position_context(
        weights,
        liquidity,
        asset_column=asset_column,
        context_asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        context_columns=tuple(liquidity_columns),
        missing_error="missing_liquidity_for_weight",
        signal_column=signal_column,
        context_name="liquidity",
    )
    _validate_finite_numeric_column(
        merged,
        weight_column,
        error="invalid_capacity_proxy_values",
    )
    _validate_finite_numeric_column(
        merged,
        amount_column,
        error="invalid_capacity_proxy_values",
    )
    if float_market_cap_column is not None:
        _validate_finite_numeric_column(
            merged,
            float_market_cap_column,
            error="invalid_capacity_proxy_values",
        )
    else:
        merged["weighted_float_market_cap_proxy"] = 0.0
    records: list[dict[str, object]] = []
    for (timestamp, signal), group in merged.groupby(
        [timestamp_column, signal_column],
        sort=True,
        dropna=False,
    ):
        weight_values = group[weight_column]
        amount_values = group[amount_column]
        positive_amount = amount_values.gt(0)
        valid_weights = weight_values[positive_amount]
        record = {
            "timestamp": timestamp,
            "signal": signal,
            "row_count": int(len(group)),
            "positive_amount_row_count": int(positive_amount.sum()),
            "excluded_non_positive_amount_count": int((~positive_amount).sum()),
            "gross_weight": float(weight_values.abs().sum()),
            "covered_gross_weight": float(valid_weights.abs().sum()),
            "weighted_amount_proxy": float(
                (valid_weights.abs() * amount_values[positive_amount]).sum()
            ),
            "weighted_float_market_cap_proxy": 0.0,
        }
        if float_market_cap_column is not None:
            record["weighted_float_market_cap_proxy"] = float(
                (
                    valid_weights.abs()
                    * group.loc[positive_amount, float_market_cap_column]
                ).sum()
            )
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=PORTFOLIO_CAPACITY_DIAGNOSTIC_COLUMNS,
    )


def _summarize_numeric_exposures(
    weights: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    exposure_columns: Sequence[str],
    asset_column: str,
    context_asset_column: str,
    timestamp_column: str,
    signal_column: str,
    weight_column: str,
) -> pd.DataFrame:
    if not exposure_columns:
        raise ValueError("exposure_columns_required")
    merged = _merged_exposure_frame(
        weights,
        exposures,
        exposure_columns=exposure_columns,
        asset_column=asset_column,
        context_asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        signal_column=signal_column,
        weight_column=weight_column,
    )
    for column in exposure_columns:
        _validate_finite_numeric_column(
            merged,
            column,
            error="invalid_portfolio_exposure_values",
        )
    records: list[dict[str, object]] = []
    for (timestamp, signal), group in merged.groupby(
        [timestamp_column, signal_column],
        sort=True,
        dropna=False,
    ):
        weights_abs = group[weight_column].abs()
        gross_weight = float(weights_abs.sum())
        for column in exposure_columns:
            values = group[column]
            records.append(
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "exposure_column": column,
                    "exposure_value": float((group[weight_column] * values).sum()),
                    "row_count": int(len(group)),
                    "gross_weight": gross_weight,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=PORTFOLIO_NUMERIC_EXPOSURE_DIAGNOSTIC_COLUMNS,
    )


def _summarize_categorical_exposures(
    weights: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    exposure_columns: Sequence[str],
    asset_column: str,
    context_asset_column: str,
    timestamp_column: str,
    signal_column: str,
    weight_column: str,
) -> pd.DataFrame:
    if not exposure_columns:
        raise ValueError("exposure_columns_required")
    merged = _merged_exposure_frame(
        weights,
        exposures,
        exposure_columns=exposure_columns,
        asset_column=asset_column,
        context_asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        signal_column=signal_column,
        weight_column=weight_column,
    )
    for column in exposure_columns:
        merged[column] = strict_hashable_key_series(
            merged[column],
            error=f"categorical exposure column {column!r} contains invalid values",
            allow_null=True,
        )
    records: list[dict[str, object]] = []
    for column in exposure_columns:
        for (timestamp, signal, value), group in merged.groupby(
            [timestamp_column, signal_column, column],
            sort=True,
            dropna=False,
        ):
            weights = group[weight_column]
            records.append(
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "exposure_column": column,
                    "exposure_value": value,
                    "asset_count": int(len(group)),
                    "gross_weight": float(weights.abs().sum()),
                    "net_weight": float(weights.sum()),
                    "long_weight": float(weights[weights > 0].sum()),
                    "short_weight": float(weights[weights < 0].sum()),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=PORTFOLIO_CATEGORICAL_EXPOSURE_DIAGNOSTIC_COLUMNS,
    )


def _merged_exposure_frame(
    weights: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    exposure_columns: Sequence[str],
    asset_column: str,
    context_asset_column: str,
    timestamp_column: str,
    signal_column: str,
    weight_column: str,
) -> pd.DataFrame:
    _require_columns(
        weights,
        (asset_column, timestamp_column, signal_column, weight_column),
        "weights",
    )
    _require_columns(
        exposures,
        (context_asset_column, timestamp_column, *tuple(exposure_columns)),
        "exposures",
    )
    merged = _merge_position_context(
        weights,
        exposures,
        asset_column=asset_column,
        context_asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        context_columns=(
            context_asset_column,
            timestamp_column,
            *tuple(exposure_columns),
        ),
        missing_error="missing_asset_exposure",
        signal_column=signal_column,
        context_name="exposures",
    )
    _validate_finite_numeric_column(
        merged,
        weight_column,
        error="invalid_portfolio_exposure_values",
    )
    return merged


def _merge_position_context(
    weights: pd.DataFrame,
    context: pd.DataFrame,
    *,
    asset_column: str,
    context_asset_column: str,
    timestamp_column: str,
    context_columns: tuple[str, ...],
    missing_error: str,
    signal_column: str | None,
    context_name: str,
) -> pd.DataFrame:
    normalized_weights = _normalize_identity_frame(
        weights,
        asset_column=asset_column,
        timestamp_column=timestamp_column,
        signal_column=signal_column,
        name="weights",
    )
    normalized_context = _normalize_identity_frame(
        context,
        asset_column=context_asset_column,
        timestamp_column=timestamp_column,
        signal_column=None,
        name=context_name,
    )
    _reject_duplicate_keys(
        normalized_context,
        (context_asset_column, timestamp_column),
        context_name,
    )
    renamed_context = normalized_context.loc[:, list(context_columns)].rename(
        columns={context_asset_column: asset_column}
    )
    merged = normalized_weights.merge(
        renamed_context,
        on=[asset_column, timestamp_column],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if bool(merged["_merge"].ne("both").any()):
        raise ValueError(missing_error)
    return merged.drop(columns=["_merge"])


def _normalize_identity_frame(
    frame: pd.DataFrame,
    *,
    asset_column: str,
    timestamp_column: str,
    signal_column: str | None,
    name: str,
) -> pd.DataFrame:
    result = frame.copy()
    result[timestamp_column] = strict_timestamp_series(
        result[timestamp_column], error=f"invalid_{name}_identity_keys"
    )
    result[asset_column] = strict_string_series(
        result[asset_column],
        error=f"invalid_{name}_identity_keys",
        strip=False,
    )
    if signal_column is not None:
        result[signal_column] = strict_string_series(
            result[signal_column],
            error=f"invalid_{name}_identity_keys",
            strip=False,
        )
    return result


def _validate_finite_numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    error: str,
) -> None:
    if not pd.api.types.is_numeric_dtype(frame[column]):
        raise ValueError(error)
    frame[column] = strict_finite_numeric_series(
        frame[column],
        error=error,
    ).astype("float64")


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

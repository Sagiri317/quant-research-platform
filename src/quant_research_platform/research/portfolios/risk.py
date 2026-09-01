"""Common factor-risk estimation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_finite_numeric_series


@dataclass(frozen=True)
class FactorRiskEstimate:
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    factor_half_life: float
    specific_half_life: float
    diagonal_shrinkage: float
    annualization_factor: float


def estimate_factor_risk(
    history: pd.DataFrame,
    *,
    exposure_columns: tuple[str, ...],
    weight_column: str,
    factor_half_life: float = 126.0,
    specific_half_life: float = 63.0,
    diagonal_shrinkage: float = 0.20,
    annualization_factor: float = 252.0,
    factor_window: int = 504,
    specific_window: int = 252,
    minimum_factor_dates: int = 252,
    minimum_specific_dates: int = 126,
    minimum_cross_section: int = 100,
) -> FactorRiskEstimate:
    """Estimate daily factor returns and EWMA/shrunk factor and specific risk."""

    required = {
        "timestamp",
        "asset_id",
        "asset_return",
        weight_column,
        *exposure_columns,
    }
    if not isinstance(history, pd.DataFrame) or not required.issubset(history.columns):
        raise ValueError("factor_risk_columns_invalid")
    if not exposure_columns or len(set(exposure_columns)) != len(exposure_columns):
        raise ValueError("factor_risk_exposure_columns_invalid")
    _positive_float(factor_half_life, "factor_half_life_invalid")
    _positive_float(specific_half_life, "specific_half_life_invalid")
    _fraction(diagonal_shrinkage, "diagonal_shrinkage_invalid")
    _positive_float(annualization_factor, "annualization_factor_invalid")
    for value, name in (
        (factor_window, "factor_window_invalid"),
        (specific_window, "specific_window_invalid"),
        (minimum_factor_dates, "minimum_factor_dates_invalid"),
        (minimum_specific_dates, "minimum_specific_dates_invalid"),
        (minimum_cross_section, "minimum_cross_section_invalid"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(name)
    frame = history.loc[:, list(required)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["asset_id"] = frame["asset_id"].astype("string")
    if frame.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("factor_risk_primary_key_duplicate")
    for column in ("asset_return", weight_column, *exposure_columns):
        frame[column] = strict_finite_numeric_series(
            frame[column], error="factor_risk_numeric_invalid"
        ).astype("float64")
    if frame[weight_column].le(0.0).any():
        raise ValueError("factor_risk_weight_invalid")

    factor_rows: list[dict[str, object]] = []
    residual_rows: list[dict[str, object]] = []
    for timestamp, cross in frame.groupby("timestamp", sort=True):
        p = len(exposure_columns)
        if len(cross) < max(minimum_cross_section, 5 * p):
            continue
        design = cross.loc[:, exposure_columns].to_numpy(dtype="float64")
        if np.linalg.matrix_rank(design) != p:
            continue
        statistical_weight = cross[weight_column].to_numpy(dtype="float64")
        root_weight = np.sqrt(statistical_weight)
        weighted_design = design * root_weight[:, None]
        coefficients = np.linalg.lstsq(
            weighted_design,
            cross["asset_return"].to_numpy(dtype="float64") * root_weight,
            rcond=None,
        )[0]
        residuals = (
            cross["asset_return"].to_numpy(dtype="float64") - design @ coefficients
        )
        factor_rows.append(
            {
                "timestamp": timestamp,
                **dict(zip(exposure_columns, coefficients, strict=True)),
            }
        )
        residual_rows.extend(
            {"timestamp": timestamp, "asset_id": asset, "residual": value}
            for asset, value in zip(cross["asset_id"], residuals, strict=True)
        )
    factor_returns = pd.DataFrame.from_records(
        factor_rows, columns=("timestamp", *exposure_columns)
    )
    if len(factor_returns) < minimum_factor_dates:
        raise ValueError("factor_risk_factor_history_insufficient")
    factor_returns = factor_returns.sort_values("timestamp", kind="mergesort").tail(
        factor_window
    )
    factor_values = factor_returns.loc[:, exposure_columns].to_numpy(dtype="float64")
    factor_covariance = _ewma_covariance(
        factor_values,
        half_life=factor_half_life,
        diagonal_shrinkage=diagonal_shrinkage,
        annualization_factor=annualization_factor,
        columns=exposure_columns,
    )
    residual_frame = pd.DataFrame.from_records(residual_rows)
    specific: dict[str, float] = {}
    for asset_id, values in residual_frame.groupby("asset_id", sort=True):
        values = values.sort_values("timestamp", kind="mergesort").tail(specific_window)
        if len(values) < minimum_specific_dates:
            continue
        weights = _ewma_weights(len(values), specific_half_life)
        specific[str(asset_id)] = float(
            np.dot(weights, values["residual"].to_numpy(dtype="float64") ** 2)
            * annualization_factor
        )
    specific_variance = pd.Series(specific, dtype="float64", name="specific_variance")
    return FactorRiskEstimate(
        factor_returns=factor_returns.reset_index(drop=True),
        factor_covariance=factor_covariance,
        specific_variance=specific_variance,
        factor_half_life=factor_half_life,
        specific_half_life=specific_half_life,
        diagonal_shrinkage=diagonal_shrinkage,
        annualization_factor=annualization_factor,
    )


def _ewma_covariance(
    values: np.ndarray,
    *,
    half_life: float,
    diagonal_shrinkage: float,
    annualization_factor: float,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    weights = _ewma_weights(len(values), half_life)
    centered = values - np.average(values, axis=0, weights=weights)
    covariance = (centered * weights[:, None]).T @ centered
    covariance = (1.0 - diagonal_shrinkage) * covariance + diagonal_shrinkage * np.diag(
        np.diag(covariance)
    )
    covariance = (covariance + covariance.T) / 2.0 * annualization_factor
    if np.linalg.eigvalsh(covariance).min(initial=0.0) < -1e-12:
        raise ValueError("factor_risk_covariance_not_psd")
    return pd.DataFrame(covariance, index=columns, columns=columns)


def _ewma_weights(count: int, half_life: float) -> np.ndarray:
    decay = 0.5 ** (1.0 / half_life)
    values = decay ** np.arange(count - 1, -1, -1, dtype="float64")
    return values / values.sum()


def _positive_float(value: float, error: str) -> None:
    if type(value) is not float or not np.isfinite(value) or value <= 0.0:
        raise ValueError(error)


def _fraction(value: float, error: str) -> None:
    if type(value) is not float or not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(error)

"""Common factor-risk estimation primitives."""

from __future__ import annotations

from array import array
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


@dataclass(frozen=True)
class MarketModelRiskStyles:
    beta: pd.Series
    residual_volatility: pd.Series


class MarketModelRiskState:
    """Causal rolling market model used by the six formal risk styles."""

    def __init__(self, *, window: int = 252, residual_window: int = 60):
        self.window = window
        self.residual_window = residual_window
        self._returns: dict[str, array] = {}
        self._market_returns: dict[str, array] = {}
        self._residuals: dict[str, array] = {}
        self._sums: dict[str, list[float]] = {}
        self._previous_weights: dict[str, float] = {}

    def observe(self, panel: pd.DataFrame) -> MarketModelRiskStyles:
        required = {
            "asset_id",
            "period_return",
            "float_market_cap",
        }
        if not isinstance(panel, pd.DataFrame) or not required.issubset(panel.columns):
            raise ValueError("market_model_risk_columns_invalid")
        if "eligible" in panel.columns:
            if (
                not pd.api.types.is_bool_dtype(panel["eligible"])
                or panel["eligible"].isna().any()
            ):
                raise ValueError("market_model_risk_eligibility_invalid")
            panel = panel.loc[panel["eligible"]]
        frame = panel.loc[:, list(required)].copy()
        frame["asset_id"] = frame["asset_id"].astype("string")
        for column in ("period_return", "float_market_cap"):
            frame[column] = strict_finite_numeric_series(
                frame[column],
                error="market_model_risk_numeric_invalid",
            ).astype("float64")
        if (
            frame["asset_id"].duplicated().any()
            or frame["float_market_cap"].le(0.0).any()
        ):
            raise ValueError("market_model_risk_key_invalid")
        frame = frame.sort_values("asset_id", kind="mergesort").reset_index(drop=True)

        weighted = [
            (
                str(row.asset_id),
                float(row.period_return),
                self._previous_weights.get(str(row.asset_id)),
                float(row.float_market_cap),
            )
            for row in frame.itertuples(index=False)
        ]
        market_denominator = sum(
            weight
            for _asset, _return_value, weight, _current_weight in weighted
            if weight is not None
        )
        market_return = (
            sum(
                weight * return_value
                for _asset, return_value, weight, _current_weight in weighted
                if weight is not None
            )
            / market_denominator
            if market_denominator > 0.0
            else np.nan
        )

        betas: dict[str, float] = {}
        volatilities: dict[str, float] = {}
        for asset, return_value, _previous_weight, current_weight in weighted:
            returns = self._returns.setdefault(asset, array("d"))
            markets = self._market_returns.setdefault(asset, array("d"))
            residuals = self._residuals.setdefault(asset, array("d"))
            totals = self._sums.setdefault(asset, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if np.isfinite(market_return):
                returns.append(return_value)
                markets.append(market_return)
                totals[0] += 1.0
                totals[1] += return_value
                totals[2] += market_return
                totals[3] += return_value * return_value
                totals[4] += return_value * market_return
                totals[5] += market_return * market_return
                if len(returns) > self.window:
                    old_return = float(returns.pop(0))
                    old_market = float(markets.pop(0))
                    totals[0] -= 1.0
                    totals[1] -= old_return
                    totals[2] -= old_market
                    totals[3] -= old_return * old_return
                    totals[4] -= old_return * old_market
                    totals[5] -= old_market * old_market
            beta = np.nan
            residual = np.nan
            if totals[0] == self.window:
                market_variance = totals[5] - totals[2] * totals[2] / totals[0]
                if market_variance > 0.0:
                    covariance = totals[4] - totals[1] * totals[2] / totals[0]
                    beta = covariance / market_variance
                    residual = return_value - beta * market_return
                    residuals.append(residual * residual)
                    if len(residuals) > self.residual_window:
                        residuals.pop(0)
            betas[asset] = float(beta)
            volatilities[asset] = (
                float(np.sqrt(np.mean(residuals)))
                if len(residuals) == self.residual_window
                else np.nan
            )
            self._previous_weights[asset] = current_weight

        return MarketModelRiskStyles(
            beta=pd.Series(betas, dtype="float64", name="beta_252d"),
            residual_volatility=pd.Series(
                volatilities,
                dtype="float64",
                name="residual_volatility_60d",
            ),
        )


def preprocess_risk_styles(
    frame: pd.DataFrame,
    *,
    style_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Apply the formal per-date 3MAD winsorization and population zscore."""

    required = {"timestamp", *style_columns}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("risk_style_preprocessing_columns_invalid")
    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="raise")
    for column in style_columns:
        values = pd.to_numeric(result[column], errors="raise").astype("float64")
        grouped = values.groupby(timestamps, sort=False)
        median = grouped.transform("median")
        mad = grouped.transform(
            lambda values: (
                float(np.median(np.abs(values.dropna() - values.median())))
                if len(values.dropna())
                else np.nan
            )
        )
        bound = 3.0 * 1.4826 * mad
        clipped = values.clip(lower=median - bound, upper=median + bound)
        means = clipped.groupby(timestamps, sort=False).transform("mean")
        scales = clipped.groupby(timestamps, sort=False).transform(
            lambda values: (
                float(np.sqrt(np.mean((values.dropna() - values.mean()) ** 2)))
                if len(values.dropna())
                else np.nan
            )
        )
        result[column] = ((clipped - means) / scales).where(scales.ne(0.0), 0.0)
    return result


class FactorRiskState:
    """Incremental, exposure-lagged factor-risk estimator."""

    def __init__(
        self,
        *,
        style_columns: tuple[str, ...],
        weight_column: str = "float_market_cap",
        minimum_cross_section: int = 100,
        minimum_factor_dates: int = 252,
        minimum_specific_dates: int = 126,
    ):
        self.style_columns = style_columns
        self.weight_column = weight_column
        self.minimum_cross_section = minimum_cross_section
        self.minimum_factor_dates = minimum_factor_dates
        self.minimum_specific_dates = minimum_specific_dates
        self._industry_codes: set[str] = set()
        self._factor_rows: list[dict[str, object]] = []
        self._residuals: dict[str, array] = {}

    def observe_return_session(self, cross: pd.DataFrame) -> None:
        required = {
            "timestamp",
            "asset_id",
            "asset_return",
            "industry_level1_code",
            self.weight_column,
            *self.style_columns,
        }
        if not isinstance(cross, pd.DataFrame) or not required.issubset(cross.columns):
            raise ValueError("factor_risk_columns_invalid")
        frame = cross.loc[:, list(required)].copy()
        if "eligible" in cross.columns:
            if (
                not pd.api.types.is_bool_dtype(cross["eligible"])
                or cross["eligible"].isna().any()
            ):
                raise ValueError("factor_risk_eligibility_invalid")
            frame = frame.loc[cross["eligible"]]
        frame["asset_id"] = frame["asset_id"].astype("string")
        for column in ("asset_return", self.weight_column, *self.style_columns):
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(
                "float64"
            )
        frame = frame.dropna().loc[lambda item: item[self.weight_column].gt(0.0)]
        if frame.empty:
            return
        frame = preprocess_risk_styles(frame, style_columns=self.style_columns)
        current_industries = sorted(set(frame["industry_level1_code"].astype(str)))
        self._industry_codes.update(current_industries)
        industry_columns = tuple(f"industry_{code}" for code in current_industries)
        exposure_columns = (*industry_columns, *self.style_columns)
        design = np.column_stack(
            (
                frame["industry_level1_code"].astype(str).eq(current_industries[0])
                if len(current_industries) == 1
                else pd.get_dummies(
                    frame["industry_level1_code"].astype(str),
                    dtype="float64",
                ).reindex(columns=sorted(current_industries), fill_value=0.0),
                frame.loc[:, list(self.style_columns)].to_numpy(dtype="float64"),
            )
        )
        if design.shape[1] != len(exposure_columns):
            raise ValueError("factor_risk_exposure_columns_invalid")
        p = design.shape[1]
        if len(frame) < max(self.minimum_cross_section, 5 * p):
            return
        if np.linalg.matrix_rank(design) != p:
            return
        weights = frame[self.weight_column].to_numpy(dtype="float64")
        root_weights = np.sqrt(weights)
        returns = frame["asset_return"].to_numpy(dtype="float64")
        coefficients = np.linalg.lstsq(
            design * root_weights[:, None],
            returns * root_weights,
            rcond=None,
        )[0]
        residuals = returns - design @ coefficients
        timestamp = pd.Timestamp(frame["timestamp"].iloc[0])
        row: dict[str, object] = {"timestamp": timestamp}
        row.update(
            zip(
                exposure_columns,
                coefficients,
                strict=True,
            )
        )
        self._factor_rows.append(row)
        for asset, residual in zip(
            frame["asset_id"].astype(str),
            residuals,
            strict=True,
        ):
            self._residuals.setdefault(asset, array("d")).append(float(residual**2))
            if len(self._residuals[asset]) > 252:
                self._residuals[asset].pop(0)
        if len(self._factor_rows) > 504:
            self._factor_rows.pop(0)

    def current_exposures(self, cross: pd.DataFrame) -> pd.DataFrame:
        required = {"asset_id", "industry_level1_code", *self.style_columns}
        if not isinstance(cross, pd.DataFrame) or not required.issubset(cross.columns):
            raise ValueError("factor_risk_exposure_columns_invalid")
        frame = cross.loc[:, list(required)].copy()
        frame["asset_id"] = frame["asset_id"].astype("string")
        codes = frame["industry_level1_code"].astype(str)
        self._industry_codes.update(set(codes))
        industries = pd.get_dummies(codes, dtype="float64").reindex(
            columns=sorted(self._industry_codes),
            fill_value=0.0,
        )
        industries.columns = tuple(f"industry_{code}" for code in industries.columns)
        result = pd.concat(
            (
                frame.loc[:, ("asset_id", "industry_level1_code")].reset_index(
                    drop=True
                ),
                industries.reset_index(drop=True),
                frame.loc[:, list(self.style_columns)].reset_index(drop=True),
            ),
            axis=1,
        )
        return result

    def estimate(self) -> FactorRiskEstimate:
        if len(self._factor_rows) < self.minimum_factor_dates:
            raise ValueError("factor_risk_factor_history_insufficient")
        industry_columns = tuple(
            f"industry_{code}" for code in sorted(self._industry_codes)
        )
        columns = (*industry_columns, *self.style_columns)
        factor_returns = pd.DataFrame(self._factor_rows).reindex(
            columns=["timestamp", *columns],
            fill_value=0.0,
        )
        covariance = _ewma_covariance(
            factor_returns.loc[:, list(columns)].to_numpy(dtype="float64"),
            half_life=126.0,
            diagonal_shrinkage=0.20,
            annualization_factor=252.0,
            columns=columns,
        )
        specific: dict[str, float] = {}
        for asset, values in self._residuals.items():
            if len(values) >= self.minimum_specific_dates:
                recent = values[-252:]
                weights = _ewma_weights(len(recent), 63.0)
                specific[asset] = float(np.dot(weights, recent) * 252.0)
        return FactorRiskEstimate(
            factor_returns=factor_returns.reset_index(drop=True),
            factor_covariance=covariance,
            specific_variance=pd.Series(
                specific,
                dtype="float64",
                name="specific_variance",
            ),
            factor_half_life=126.0,
            specific_half_life=63.0,
            diagonal_shrinkage=0.20,
            annualization_factor=252.0,
        )


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

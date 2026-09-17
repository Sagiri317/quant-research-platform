"""Common factor-risk estimation primitives."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import frexp, ldexp

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_real_numeric_series,
)
from quant_research_platform.research.features.transforms import _mad_winsorize
from quant_research_platform.research.models.scaler import ModelMatrixScaler


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
        frame = panel.loc[
            :, list(required) + (["eligible"] if "eligible" in panel else [])
        ].copy()
        frame["asset_id"] = frame["asset_id"].astype("string")
        for column in ("period_return", "float_market_cap"):
            frame[column] = strict_finite_numeric_series(
                frame[column],
                error="market_model_risk_numeric_invalid",
                allow_null=True,
            ).astype("float64")
        eligible = frame.loc[frame.eligible] if "eligible" in frame else frame
        if eligible[["period_return", "float_market_cap"]].isna().any().any():
            raise ValueError("market_model_risk_numeric_invalid")
        if (
            frame["asset_id"].duplicated().any()
            or frame["float_market_cap"].le(0.0).any()
        ):
            raise ValueError("market_model_risk_key_invalid")
        frame = frame.sort_values("asset_id", kind="mergesort").reset_index(drop=True)
        for asset in self._returns.keys() - set(frame.asset_id):
            del self._returns[asset]
            del self._market_returns[asset]
            del self._residuals[asset]

        weighted = [
            (
                str(row.asset_id),
                float(row.period_return),
                self._previous_weights.get(str(row.asset_id)),
                float(row.float_market_cap),
                bool(row.eligible) if "eligible" in frame else True,
            )
            for row in frame.itertuples(index=False)
        ]
        market_scale = max(
            (
                weight
                for _asset, _return, weight, _current, eligible in weighted
                if eligible and weight is not None
            ),
            default=1.0,
        )
        market_exponent = frexp(market_scale)[1]
        market_denominator = sum(
            ldexp(weight, -market_exponent)
            for _asset, _return_value, weight, _current_weight, eligible in weighted
            if eligible and weight is not None
        )
        market_return = (
            sum(
                ldexp(weight, -market_exponent) * return_value
                for _asset, return_value, weight, _current_weight, eligible in weighted
                if eligible and weight is not None
            )
            / market_denominator
            if market_denominator > 0.0
            else np.nan
        )

        betas: dict[str, float] = {}
        volatilities: dict[str, float] = {}
        for (
            asset,
            return_value,
            _previous_weight,
            _current_weight,
            _eligible,
        ) in weighted:
            returns = self._returns.setdefault(asset, array("d"))
            markets = self._market_returns.setdefault(asset, array("d"))
            residuals = self._residuals.setdefault(asset, array("d"))
            if np.isfinite(market_return) and np.isfinite(return_value):
                returns.append(return_value)
                markets.append(market_return)
                if len(returns) > self.window:
                    returns.pop(0)
                    markets.pop(0)
            else:
                del returns[:]
                del markets[:]
            beta = np.nan
            residual = np.nan
            if len(returns) == self.window:
                market_values = np.asarray(markets)
                return_values = np.asarray(returns)
                market_differences = market_values - market_values[0]
                return_differences = return_values - return_values[0]
                market_centered = market_differences - market_differences.mean()
                return_centered = return_differences - return_differences.mean()
                scale = float(np.max(np.abs(market_centered)))
                if scale > 0.0:
                    market_centered /= scale
                    beta = float(
                        np.dot(return_centered, market_centered)
                        / np.dot(market_centered, market_centered)
                        / scale
                    )
                    residual = return_centered[-1] - beta * market_centered[-1] * scale
            residuals.append(residual)
            if len(residuals) > self.residual_window:
                residuals.pop(0)
            betas[asset] = float(beta)
            volatilities[asset] = (
                float(np.std(residuals, ddof=0))
                if len(residuals) == self.residual_window
                else np.nan
            )
        self._previous_weights = {
            asset: current_weight
            for asset, _return, _previous_weight, current_weight, _eligible in weighted
            if np.isfinite(current_weight)
        }

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
    eligible = pd.Series(True, index=result.index)
    if "eligible" in result:
        if (
            not pd.api.types.is_bool_dtype(result.eligible)
            or result.eligible.isna().any()
        ):
            raise ValueError("factor_risk_eligibility_invalid")
        eligible = result.eligible
    for column in style_columns:
        values = strict_real_numeric_series(
            result[column], error="risk_style_numeric_invalid"
        )
        grouped = values.where(eligible).groupby(timestamps, sort=False)
        clipped = _mad_winsorize(values, grouped)
        result[column] = np.nan
        for positions in timestamps.groupby(timestamps, sort=False).indices.values():
            current = clipped.iloc[positions]
            fitting = current.loc[eligible.iloc[positions]].dropna()
            if fitting.empty:
                continue
            if fitting.nunique() == 1:
                result.iloc[positions, result.columns.get_loc(column)] = 0.0
                continue
            scaler = ModelMatrixScaler.fit(fitting.to_frame(name=column))
            scaled = pd.Series(np.nan, index=current.index, dtype="float64")
            observed = current.notna()
            scaled.loc[observed] = scaler.transform(
                current.loc[observed].to_frame(name=column)
            )[column].to_numpy()
            result.iloc[positions, result.columns.get_loc(column)] = scaled.to_numpy()
    return result


@dataclass(frozen=True)
class FactorReturnFit:
    exposures: pd.DataFrame
    factor_returns: pd.Series
    residuals: pd.Series
    asset_returns: pd.Series


@dataclass(frozen=True)
class HoldingFactorInputs:
    decision_at: pd.Timestamp
    entry_at: pd.Timestamp
    end_at: pd.Timestamp
    fitted: FactorReturnFit | None
    decision_close: pd.Series
    entry_open: pd.Series
    benchmark_weights: pd.DataFrame
    reason: str | None = None


def holding_attribution(
    inputs: HoldingFactorInputs,
    *,
    after_notionals: pd.Series,
    nav_before: float,
    nav_post: float,
    benchmark_id: str | None,
    benchmark_return: float | None,
    holding_returns: pd.Series,
) -> tuple[dict[str, float | None], dict[str, str], pd.Series]:
    """Ex-post relative-factor decomposition with an explicit replication gap."""
    components = (
        "benchmark",
        "industry",
        "style",
        "stock_residual_difference",
        "benchmark_replication_gap",
        "residual_unexplained",
    )
    values = dict.fromkeys(components)
    reasons = {}
    values["benchmark"] = 0.0 if benchmark_id is None else benchmark_return
    fitted = inputs.fitted
    if fitted is None:
        reasons.update(
            {
                name: inputs.reason or "holding_factor_fit_unidentified"
                for name in components
                if name != "benchmark"
            }
        )
        return values, reasons, pd.Series(dtype="float64")
    held = after_notionals.loc[after_notionals.ne(0.0)]
    if not set(held.index).issubset(fitted.exposures.index):
        reasons.update(
            {
                name: "holding_exposures_incomplete"
                for name in components
                if name != "benchmark"
            }
        )
        return values, reasons, pd.Series(dtype="float64")
    if not np.allclose(
        holding_returns.reindex(held.index),
        fitted.asset_returns.reindex(held.index),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("holding_return_ledger_mismatch")
    weights = held.div(nav_before).reindex(fitted.exposures.index, fill_value=0.0)
    realized = fitted.exposures.mul(
        held.div(nav_post).reindex(fitted.exposures.index, fill_value=0.0), axis=0
    ).sum()
    reference = pd.Series(0.0, index=fitted.exposures.index)
    if benchmark_id is not None:
        snapshot = inputs.benchmark_weights.loc[
            inputs.benchmark_weights.benchmark_id.eq(benchmark_id)
        ].set_index("asset_id")["weight"]
        if snapshot.index.has_duplicates:
            raise ValueError("attribution_benchmark_keys_invalid")
        complete = (
            not snapshot.empty
            and np.isfinite(snapshot).all()
            and snapshot.gt(0).all()
            and np.isclose(snapshot.sum(), 1.0, rtol=0, atol=1e-8)
            and set(snapshot.index).issubset(fitted.exposures.index)
        )
        prices = pd.concat(
            [
                inputs.decision_close.reindex(snapshot.index),
                inputs.entry_open.reindex(snapshot.index),
            ],
            axis=1,
        )
        if (
            not complete
            or prices.isna().any().any()
            or not np.isfinite(prices).all().all()
            or not prices.gt(0).all().all()
        ):
            reasons.update(
                {
                    name: "attribution_reference_basket_incomplete"
                    for name in components
                    if name != "benchmark"
                }
            )
            return values, reasons, realized
        drifted = snapshot * prices.iloc[:, 1] / prices.iloc[:, 0]
        reference = drifted.div(drifted.sum()).reindex(
            fitted.exposures.index, fill_value=0.0
        )
    relative = weights - reference
    contributions = fitted.exposures.mul(relative, axis=0).sum() * fitted.factor_returns
    values["industry"] = float(
        contributions.loc[contributions.index.str.startswith("industry_")].sum()
    )
    values["style"] = float(
        contributions.loc[~contributions.index.str.startswith("industry_")].sum()
    )
    values["stock_residual_difference"] = float(relative.dot(fitted.residuals))
    if values["benchmark"] is None:
        reasons["benchmark"] = reasons["benchmark_replication_gap"] = reasons[
            "residual_unexplained"
        ] = "benchmark_total_return_open_missing"
    else:
        values["benchmark_replication_gap"] = float(
            reference.dot(fitted.asset_returns) - values["benchmark"]
        )
        values["residual_unexplained"] = (
            values["stock_residual_difference"] + values["benchmark_replication_gap"]
        )
        expected_gross = float(
            held.mul(holding_returns.reindex(held.index)).sum() / nav_before
        )
        actual_gross = sum(
            values[name]
            for name in ("benchmark", "industry", "style", "residual_unexplained")
        )
        if not np.isclose(expected_gross, actual_gross, rtol=1e-11, atol=1e-12):
            raise ValueError("holding_attribution_reconciliation_invalid")
    return values, reasons, realized


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
        self._residual_sessions: dict[str, array] = {}
        self._session_count = 0

    def prepare_exposures(self, cross: pd.DataFrame) -> pd.DataFrame:
        required = {
            "timestamp",
            "asset_id",
            "industry_level1_code",
            self.weight_column,
            *self.style_columns,
        }
        if not isinstance(cross, pd.DataFrame) or not required.issubset(cross.columns):
            raise ValueError("factor_risk_columns_invalid")
        frame = cross.loc[
            :,
            list(required)
            + (["eligible"] if "eligible" in cross else [])
            + (["asset_return"] if "asset_return" in cross else []),
        ].copy()
        if "eligible" in frame:
            if (
                not pd.api.types.is_bool_dtype(frame.eligible)
                or frame.eligible.isna().any()
            ):
                raise ValueError("factor_risk_eligibility_invalid")
            frame = frame.loc[frame.eligible].copy()
        frame["asset_id"] = frame.asset_id.astype("string")
        for column in (self.weight_column, *self.style_columns):
            frame[column] = strict_real_numeric_series(
                frame[column], error="factor_risk_numeric_invalid"
            )
        return (
            frame.dropna(subset=list(required))
            .loc[
                lambda item: (
                    item[self.weight_column].gt(0)
                    & item.industry_level1_code.astype(str).ne("")
                )
            ]
            .copy()
        )

    def fit_return_session(
        self, cross: pd.DataFrame, *, require_complete: bool = False
    ) -> FactorReturnFit | None:
        if "asset_return" not in cross:
            raise ValueError("factor_risk_columns_invalid")
        frame = self.prepare_exposures(cross)
        frame["asset_return"] = strict_real_numeric_series(
            frame.asset_return, error="factor_risk_numeric_invalid"
        )
        if require_complete and (
            frame.asset_return.isna().any() or not np.isfinite(frame.asset_return).all()
        ):
            raise ValueError("holding_returns_incomplete")
        if frame.empty:
            return None
        if frame.asset_id.duplicated().any() or frame.timestamp.nunique() != 1:
            raise ValueError("factor_return_cross_section_keys_invalid")
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
        weights = np.sqrt(frame[self.weight_column].to_numpy(dtype="float64"))
        root_weights = np.sqrt(weights)
        returns = frame["asset_return"].to_numpy(dtype="float64")
        coefficients = np.linalg.lstsq(
            design * root_weights[:, None],
            returns * root_weights,
            rcond=None,
        )[0]
        residuals = returns - design @ coefficients
        assets = pd.Index(frame["asset_id"].astype(str), name="asset_id")
        return FactorReturnFit(
            pd.DataFrame(design, index=assets, columns=exposure_columns),
            pd.Series(coefficients, index=exposure_columns),
            pd.Series(residuals, index=assets),
            pd.Series(returns, index=assets),
        )

    def observe_return_session(self, cross: pd.DataFrame) -> None:
        fitted = self.fit_return_session(cross)
        self._session_count += 1
        while (
            self._factor_rows
            and self._factor_rows[0]["session"] <= self._session_count - 504
        ):
            self._factor_rows.pop(0)
        for asset in tuple(self._residuals):
            sessions = self._residual_sessions[asset]
            values = self._residuals[asset]
            while sessions and sessions[0] <= self._session_count - 252:
                sessions.pop(0)
                values.pop(0)
            if not sessions:
                del self._residual_sessions[asset]
                del self._residuals[asset]
        if fitted is None:
            return
        frame = fitted.exposures.reset_index()
        exposure_columns = tuple(fitted.factor_returns.index)
        coefficients = fitted.factor_returns.to_numpy()
        residuals = fitted.residuals.to_numpy()
        timestamp = pd.Timestamp(cross["timestamp"].iloc[0])
        row: dict[str, object] = {
            "timestamp": timestamp,
            "session": self._session_count,
        }
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
            self._residual_sessions.setdefault(asset, array("Q")).append(
                self._session_count
            )

    def current_exposures(self, cross: pd.DataFrame) -> pd.DataFrame:
        required = {"asset_id", "industry_level1_code", *self.style_columns}
        if not isinstance(cross, pd.DataFrame) or not required.issubset(cross.columns):
            raise ValueError("factor_risk_exposure_columns_invalid")
        frame = cross.loc[:, list(required)].copy()
        frame["asset_id"] = frame["asset_id"].astype("string")
        codes = frame["industry_level1_code"].astype(str)
        self._industry_codes = set(codes)
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
        factor_returns = (
            pd.DataFrame(self._factor_rows)
            .reindex(columns=["session", "timestamp", *columns])
            .dropna()
        )
        if len(factor_returns) < self.minimum_factor_dates:
            raise ValueError("factor_risk_factor_history_insufficient")
        factor_ages = self._session_count - factor_returns.pop("session").to_numpy()
        covariance = _ewma_covariance(
            factor_returns.loc[:, list(columns)].to_numpy(dtype="float64"),
            ages=factor_ages,
            half_life=126.0,
            diagonal_shrinkage=0.20,
            annualization_factor=252.0,
            columns=columns,
        )
        specific: dict[str, float] = {}
        for asset, values in self._residuals.items():
            if len(values) >= self.minimum_specific_dates:
                ages = self._session_count - np.asarray(self._residual_sessions[asset])
                weights = _ewma_weights(ages, 63.0)
                specific[asset] = float(np.dot(weights, values) * 252.0)
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


def _ewma_covariance(
    values: np.ndarray,
    *,
    ages: np.ndarray,
    half_life: float,
    diagonal_shrinkage: float,
    annualization_factor: float,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    weights = _ewma_weights(ages, half_life)
    centered = values - np.average(values, axis=0, weights=weights)
    covariance = (centered * weights[:, None]).T @ centered
    covariance = (1.0 - diagonal_shrinkage) * covariance + diagonal_shrinkage * np.diag(
        np.diag(covariance)
    )
    covariance = (covariance + covariance.T) / 2.0 * annualization_factor
    if np.linalg.eigvalsh(covariance).min(initial=0.0) < -1e-12:
        raise ValueError("factor_risk_covariance_not_psd")
    return pd.DataFrame(covariance, index=columns, columns=columns)


def _ewma_weights(ages: np.ndarray, half_life: float) -> np.ndarray:
    decay = 0.5 ** (1.0 / half_life)
    values = decay**ages
    return values / values.sum()

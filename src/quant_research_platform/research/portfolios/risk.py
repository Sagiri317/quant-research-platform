"""Common factor-risk estimation primitives."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from decimal import Decimal, localcontext

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_real_numeric_series,
)
from quant_research_platform.research.features.transforms import (
    _industry_group_codes,
    _mad_winsorize,
)
from quant_research_platform.research.models.scaler import ModelMatrixScaler


@dataclass(frozen=True)
class FactorRiskConfig:
    factor_window: int = 504
    specific_window: int = 252
    factor_half_life: float = 126.0
    specific_half_life: float = 63.0
    minimum_factor_dates: int = 252
    minimum_specific_dates: int = 126
    minimum_cross_section: int = 100
    diagonal_shrinkage: float = 0.20
    annualization: float = 252.0
    weight_column: str = "float_market_cap"
    weighting: str = "sqrt_cap"

    def __post_init__(self):
        for name in (
            "factor_window",
            "specific_window",
            "minimum_factor_dates",
            "minimum_specific_dates",
            "minimum_cross_section",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("factor_risk_config_invalid")
        if (
            self.minimum_factor_dates > self.factor_window
            or self.minimum_specific_dates > self.specific_window
            or type(self.diagonal_shrinkage) not in {int, float}
            or not np.isfinite(self.diagonal_shrinkage)
            or not 0 <= self.diagonal_shrinkage <= 1
            or self.weighting != "sqrt_cap"
            or type(self.weight_column) is not str
            or not self.weight_column
        ):
            raise ValueError("factor_risk_config_invalid")
        for value in (
            self.factor_half_life,
            self.specific_half_life,
            self.annualization,
        ):
            if type(value) not in {float, int} or not np.isfinite(value) or value <= 0:
                raise ValueError("factor_risk_config_invalid")


@dataclass(frozen=True)
class FactorRiskEstimate:
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    factor_half_life: float
    specific_half_life: float
    diagonal_shrinkage: float
    annualization_factor: float


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
        config: FactorRiskConfig | None = None,
    ):
        self.style_columns = style_columns
        self.config = FactorRiskConfig() if config is None else config
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
            "industry_standard_id",
            self.config.weight_column,
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
        for column in (self.config.weight_column, *self.style_columns):
            frame[column] = strict_real_numeric_series(
                frame[column], error="factor_risk_numeric_invalid"
            )
        return (
            frame.dropna(subset=list(required))
            .loc[
                lambda item: (
                    item[self.config.weight_column].gt(0)
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
        frame = frame.dropna().loc[lambda item: item[self.config.weight_column].gt(0.0)]
        if frame.empty:
            return
        frame = preprocess_risk_styles(frame, style_columns=self.style_columns)
        group_codes = _industry_group_codes(frame)
        current_industries = sorted(set(group_codes))
        self._industry_codes.update(current_industries)
        industry_columns = tuple(f"industry_{code}" for code in current_industries)
        exposure_columns = (*industry_columns, *self.style_columns)
        design = np.column_stack(
            (
                group_codes.eq(current_industries[0])
                if len(current_industries) == 1
                else pd.get_dummies(
                    group_codes,
                    dtype="float64",
                ).reindex(columns=sorted(current_industries), fill_value=0.0),
                frame.loc[:, list(self.style_columns)].to_numpy(dtype="float64"),
            )
        )
        if design.shape[1] != len(exposure_columns):
            raise ValueError("factor_risk_exposure_columns_invalid")
        p = design.shape[1]
        if len(frame) < max(self.config.minimum_cross_section, 5 * p):
            return
        if np.linalg.matrix_rank(design) != p:
            return
        weights = np.sqrt(frame[self.config.weight_column].to_numpy(dtype="float64"))
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
            and self._factor_rows[0]["session"]
            <= self._session_count - self.config.factor_window
        ):
            self._factor_rows.pop(0)
        for asset in tuple(self._residuals):
            sessions = self._residual_sessions[asset]
            values = self._residuals[asset]
            while (
                sessions
                and sessions[0] <= self._session_count - self.config.specific_window
            ):
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
            self._residuals.setdefault(asset, array("d")).append(float(residual))
            self._residual_sessions.setdefault(asset, array("Q")).append(
                self._session_count
            )

    def current_exposures(self, cross: pd.DataFrame) -> pd.DataFrame:
        required = {
            "asset_id",
            "industry_standard_id",
            "industry_level1_code",
            *self.style_columns,
        }
        if not isinstance(cross, pd.DataFrame) or not required.issubset(cross.columns):
            raise ValueError("factor_risk_exposure_columns_invalid")
        frame = cross.loc[:, list(required)].copy()
        frame["asset_id"] = frame["asset_id"].astype("string")
        codes = _industry_group_codes(frame)
        frame = frame.loc[codes.notna()].copy()
        codes = codes.loc[codes.notna()]
        self._industry_codes = set(codes)
        industries = pd.get_dummies(codes, dtype="float64").reindex(
            columns=sorted(self._industry_codes),
            fill_value=0.0,
        )
        industries.columns = tuple(f"industry_{code}" for code in industries.columns)
        result = pd.concat(
            (
                frame.loc[
                    :, ("asset_id", "industry_standard_id", "industry_level1_code")
                ].reset_index(drop=True),
                industries.reset_index(drop=True),
                frame.loc[:, list(self.style_columns)].reset_index(drop=True),
            ),
            axis=1,
        )
        return result

    def estimate(self) -> FactorRiskEstimate:
        if len(self._factor_rows) < self.config.minimum_factor_dates:
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
        if len(factor_returns) < self.config.minimum_factor_dates:
            raise ValueError("factor_risk_factor_history_insufficient")
        factor_ages = self._session_count - factor_returns.pop("session").to_numpy()
        covariance = _ewma_covariance(
            factor_returns.loc[:, list(columns)].to_numpy(dtype="float64"),
            ages=factor_ages,
            half_life=self.config.factor_half_life,
            diagonal_shrinkage=self.config.diagonal_shrinkage,
            annualization_factor=self.config.annualization,
            columns=columns,
        )
        specific: dict[str, float] = {}
        for asset, values in self._residuals.items():
            if len(values) >= self.config.minimum_specific_dates:
                ages = self._session_count - np.asarray(self._residual_sessions[asset])
                weights = _ewma_weights(ages, self.config.specific_half_life)
                if weights.dtype == object or any(
                    abs(value) > np.sqrt(np.finfo(np.float64).max) for value in values
                ):
                    with localcontext() as context:
                        context.prec = 80
                        variance = sum(
                            (
                                weight
                                if isinstance(weight, Decimal)
                                else Decimal.from_float(float(weight))
                            )
                            * Decimal.from_float(value) ** 2
                            for weight, value in zip(weights, values, strict=True)
                        ) * Decimal.from_float(float(self.config.annualization))
                else:
                    variance = (
                        np.dot(weights, np.square(values)) * self.config.annualization
                    )
                specific[asset] = float(variance)
                if not np.isfinite(specific[asset]):
                    raise ValueError("factor_risk_specific_variance_non_finite")
        return FactorRiskEstimate(
            factor_returns=factor_returns.reset_index(drop=True),
            factor_covariance=covariance,
            specific_variance=pd.Series(
                specific,
                dtype="float64",
                name="specific_variance",
            ),
            factor_half_life=self.config.factor_half_life,
            specific_half_life=self.config.specific_half_life,
            diagonal_shrinkage=self.config.diagonal_shrinkage,
            annualization_factor=self.config.annualization,
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
    with localcontext() as context:
        context.prec = 80
        if weights.dtype == object or np.any(
            np.abs(values) > np.sqrt(np.finfo(np.float64).max)
        ):
            weights = np.array(
                [
                    weight
                    if isinstance(weight, Decimal)
                    else Decimal.from_float(float(weight))
                    for weight in weights
                ],
                dtype=object,
            )
            values = np.array(
                [[Decimal.from_float(float(value)) for value in row] for row in values],
                dtype=object,
            )
            diagonal_shrinkage = Decimal.from_float(float(diagonal_shrinkage))
            annualization_factor = Decimal.from_float(float(annualization_factor))
        centered = values - np.average(values, axis=0, weights=weights)
        covariance = (centered * weights[:, None]).T @ centered
        covariance = (
            1 - diagonal_shrinkage
        ) * covariance + diagonal_shrinkage * np.diag(np.diag(covariance))
        covariance = np.asarray(
            (covariance + covariance.T) / 2 * annualization_factor, dtype="float64"
        )
    if not np.isfinite(covariance).all():
        raise ValueError("factor_risk_covariance_non_finite")
    if np.linalg.eigvalsh(covariance).min(initial=0.0) < -1e-12:
        raise ValueError("factor_risk_covariance_not_psd")
    return pd.DataFrame(covariance, index=columns, columns=columns)


def _ewma_weights(ages: np.ndarray, half_life: float) -> np.ndarray:
    relative_ages = ages - ages.min()
    with np.errstate(over="ignore"):
        exponents = relative_ages / half_life
    if np.any(exponents > -np.finfo(np.float64).minexp):
        # Keep tail mass until it has multiplied the returns or residual squares.
        with localcontext() as context:
            context.prec = 80
            half = Decimal.from_float(float(half_life))
            values = np.array(
                [Decimal(2) ** (-Decimal(int(age)) / half) for age in relative_ages],
                dtype=object,
            )
            return values / sum(values)
    values = np.exp2(-exponents)
    return values / values.sum()

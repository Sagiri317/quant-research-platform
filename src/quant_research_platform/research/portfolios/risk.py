"""Common factor-risk estimation primitives."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_real_numeric_series,
    strict_timestamp_series,
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
    minimum_factor_dates: int | None = None
    minimum_specific_dates: int | None = None
    min_fresh_fraction: float = 0.75
    minimum_cross_section: int = 100
    diagonal_shrinkage: float = 0.20
    annualization: float = 252.0
    weight_column: str = "total_market_cap"
    weighting: str = "sqrt_cap"

    def __post_init__(self):
        if (
            type(self.min_fresh_fraction) not in {int, float}
            or not np.isfinite(self.min_fresh_fraction)
            or not 0 < self.min_fresh_fraction <= 1
        ):
            raise ValueError("factor_risk_config_invalid")
        for name in ("factor_window", "specific_window"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("factor_risk_config_invalid")
        for name, window in (
            ("minimum_factor_dates", self.factor_window),
            ("minimum_specific_dates", self.specific_window),
        ):
            if getattr(self, name) is None:
                object.__setattr__(
                    self, name, max(2, ceil(self.min_fresh_fraction * window))
                )
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
            if len(fitting) < 2:
                continue
            if fitting.nunique() == 1:
                result.iloc[positions, result.columns.get_loc(column)] = np.where(
                    current.notna(), 0.0, np.nan
                )
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
    closed_assets: frozenset[str] = frozenset()


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
    if nav_before <= 0 or nav_post <= 0:
        reason = "nonpositive_start_nav" if nav_before <= 0 else "nonpositive_post_nav"
        reasons.update({name: reason for name in components if name != "benchmark"})
        return values, reasons, pd.Series(dtype="float64")
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
    missing = held.index.difference(fitted.exposures.index)
    if not missing.empty:
        prices = pd.concat(
            (
                inputs.decision_close.reindex(missing),
                inputs.entry_open.reindex(missing),
            ),
            axis=1,
        )
        known_closed = (
            set(missing).issubset(inputs.closed_assets)
            and np.isfinite(prices).all().all()
            and prices.gt(0).all().all()
            and np.isfinite(holding_returns.reindex(missing)).all()
            and holding_returns.reindex(missing).gt(-1).all()
        )
        reason = (
            "holding_closed_market_domain_unidentified"
            if known_closed
            else "holding_exposures_incomplete"
        )
        reasons.update({name: reason for name in components if name != "benchmark"})
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
        self._listings: dict[str, pd.Timestamp] = {}
        self._residuals: dict[str, array] = {}
        self._residual_sessions: dict[str, array] = {}
        self._residual_fresh: dict[str, array] = {}
        self._session_count = 0
        self._specific_session_counts: dict[str, int] = {}

    def prepare_exposures(
        self, cross: pd.DataFrame, *, include_ineligible: bool = False
    ) -> pd.DataFrame:
        """Select the risk domain without discarding per-style observations."""
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
            if not include_ineligible:
                frame = frame.loc[frame.eligible].copy()
        frame["asset_id"] = frame.asset_id.astype("string")
        for column in (self.config.weight_column, *self.style_columns):
            frame[column] = strict_real_numeric_series(
                frame[column], error="factor_risk_numeric_invalid"
            )
        frame = frame.dropna(
            subset=list(
                required - set(self.style_columns) - {self.config.weight_column}
            )
        )
        weighted = frame[self.config.weight_column].gt(0)
        if include_ineligible and "eligible" in frame:
            # Projection uses fitted coefficients, not the holding's fit weight.
            weighted |= ~frame.eligible
        return frame.loc[
            weighted & frame.industry_level1_code.astype(str).ne("")
        ].copy()

    def fit_return_session(
        self,
        cross: pd.DataFrame,
        *,
        require_complete: bool = False,
        project_ineligible: bool = False,
    ) -> FactorReturnFit | None:
        fitted, industries = self._fit_return_session(
            cross,
            require_complete=require_complete,
            project_ineligible=project_ineligible,
        )
        self._industry_codes.update(industries)
        return fitted

    def _fit_return_session(
        self,
        cross: pd.DataFrame,
        *,
        require_complete: bool = False,
        project_ineligible: bool = False,
    ) -> tuple[FactorReturnFit | None, tuple[str, ...]]:
        if "asset_return" not in cross:
            raise ValueError("factor_risk_columns_invalid")
        frame = self.prepare_exposures(cross, include_ineligible=project_ineligible)
        frame["asset_return"] = strict_real_numeric_series(
            frame.asset_return, error="factor_risk_numeric_invalid"
        )
        if frame.empty:
            return None, ()
        if frame.asset_id.duplicated().any() or frame.timestamp.nunique() != 1:
            raise ValueError("factor_return_cross_section_keys_invalid")
        # Joint regression availability must not define each style's fit domain.
        frame = preprocess_risk_styles(frame, style_columns=self.style_columns)
        frame = frame.dropna(subset=list(self.style_columns))
        if require_complete and (
            frame.asset_return.isna().any() or not np.isfinite(frame.asset_return).all()
        ):
            raise ValueError("holding_returns_incomplete")
        frame = frame.dropna(subset=["asset_return"])
        projection = frame.iloc[:0]
        if project_ineligible and "eligible" in frame:
            projection = frame.loc[~frame.eligible]
            frame = frame.loc[frame.eligible]
        if frame.empty:
            return None, ()
        group_codes = _industry_group_codes(frame)
        current_industries = sorted(set(group_codes))
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
            return None, tuple(current_industries)
        returns = frame["asset_return"].to_numpy(dtype="float64")
        fitted = _weighted_fit(
            design,
            returns,
            np.sqrt(frame[self.config.weight_column].to_numpy(dtype="float64")),
        )
        if fitted is None:
            return None, tuple(current_industries)
        coefficients, residuals = fitted
        if not np.isfinite(coefficients).all() or not np.isfinite(residuals).all():
            raise ValueError("factor_risk_fit_non_finite")
        assets = pd.Index(frame["asset_id"].astype(str), name="asset_id")
        if not projection.empty:
            projection_codes = _industry_group_codes(projection)
            identified = projection_codes.isin(current_industries)
            projection = projection.loc[identified]
            projection_codes = projection_codes.loc[identified]
            projected_design = np.column_stack(
                (
                    pd.get_dummies(projection_codes, dtype="float64").reindex(
                        columns=current_industries, fill_value=0.0
                    ),
                    projection.loc[:, list(self.style_columns)].to_numpy(
                        dtype="float64"
                    ),
                )
            )
            projected_returns = projection.asset_return.to_numpy(dtype="float64")
            # Project with the eligible fit; these rows never enter risk history.
            projected_residuals = projected_returns - projected_design @ coefficients
            if not np.isfinite(projected_residuals).all():
                raise ValueError("factor_risk_fit_non_finite")
            assets = assets.append(
                pd.Index(projection.asset_id.astype(str), name="asset_id")
            )
            design = np.concatenate((design, projected_design))
            residuals = np.concatenate((residuals, projected_residuals))
            returns = np.concatenate((returns, projected_returns))
        return FactorReturnFit(
            pd.DataFrame(design, index=assets, columns=exposure_columns),
            pd.Series(coefficients, index=exposure_columns),
            pd.Series(residuals, index=assets),
            pd.Series(returns, index=assets),
        ), tuple(current_industries)

    def observe_return_session(
        self,
        cross: pd.DataFrame,
        *,
        closed_assets=frozenset(),
    ) -> None:
        fresh = (
            strict_bool_series(
                cross["is_trading"],
                error="factor_risk_trading_invalid",
                allow_null=True,
            ).fillna(False)
            if "is_trading" in cross
            else cross["asset_return"].notna()
        )
        fresh_by_asset = pd.Series(
            fresh.to_numpy(dtype=bool), index=cross.asset_id.astype(str)
        )
        listings = strict_timestamp_series(
            cross["listing_date"],
            error="factor_risk_listing_invalid",
            allow_null=True,
            errors="coerce",
        )
        for asset, listing in zip(cross.asset_id.astype(str), listings, strict=True):
            if pd.isna(listing) or (
                asset in self._listings and self._listings[asset] != listing
            ):
                self._residuals.pop(asset, None)
                self._residual_sessions.pop(asset, None)
                self._residual_fresh.pop(asset, None)
                self._specific_session_counts.pop(asset, None)
            if pd.isna(listing):
                self._listings.pop(asset, None)
                continue
            self._listings[asset] = listing
        for asset in self._listings:
            if asset not in closed_assets:
                self._specific_session_counts[asset] = (
                    self._specific_session_counts.get(asset, 0) + 1
                )
        fitted, industries = self._fit_return_session(cross.loc[listings.notna()])
        fitted = None if fitted is None else (fitted.factor_returns, fitted.residuals)
        self._industry_codes.update(industries)
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
                and sessions[0]
                <= self._specific_session_counts[asset] - self.config.specific_window
            ):
                sessions.pop(0)
                values.pop(0)
                self._residual_fresh[asset].pop(0)
            if not sessions:
                del self._residual_sessions[asset]
                del self._residuals[asset]
                del self._residual_fresh[asset]
        if fitted is None:
            return
        factor_returns, residual_values = fitted
        exposure_columns = tuple(factor_returns.index)
        coefficients = factor_returns.to_numpy()
        residuals = residual_values.to_numpy()
        timestamp = pd.Timestamp(cross["timestamp"].iloc[0])
        row: dict[str, object] = {
            "timestamp": timestamp,
            "session": self._session_count,
            "fresh": int(
                fresh_by_asset.reindex(residual_values.index).fillna(False).sum()
            )
            >= max(self.config.minimum_cross_section, 5 * len(exposure_columns)),
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
            residual_values.index.astype(str),
            residuals,
            strict=True,
        ):
            self._residuals.setdefault(asset, array("d")).append(float(residual))
            self._residual_fresh.setdefault(asset, array("b")).append(
                bool(fresh_by_asset.loc[asset])
            )
            self._residual_sessions.setdefault(asset, array("Q")).append(
                self._specific_session_counts[asset]
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
        if (
            self._session_count < self.config.factor_window
            or len(self._factor_rows) < self.config.minimum_factor_dates
        ):
            raise ValueError("factor_risk_factor_history_insufficient")
        industry_columns = tuple(
            f"industry_{code}" for code in sorted(self._industry_codes)
        )
        columns = (*industry_columns, *self.style_columns)
        factor_returns = (
            pd.DataFrame(self._factor_rows)
            .reindex(columns=["session", "timestamp", "fresh", *columns])
            .dropna()
        )
        if (
            len(factor_returns) < self.config.minimum_factor_dates
            or factor_returns["fresh"].sum() < self.config.minimum_factor_dates
        ):
            raise ValueError("factor_risk_factor_history_insufficient")
        factor_returns.pop("fresh")
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
            if (
                self._specific_session_counts[asset] >= self.config.specific_window
                and sum(self._residual_fresh[asset])
                >= self.config.minimum_specific_dates
            ):
                values = np.asarray(values)
                ages = self._specific_session_counts[asset] - np.asarray(
                    self._residual_sessions[asset]
                )
                weights = _ewma_weights(ages, self.config.specific_half_life)
                specific[asset] = float(
                    np.dot(weights, np.square(values)) * self.config.annualization
                )
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


def _validated_covariance(covariance: np.ndarray, *, error_prefix: str) -> np.ndarray:
    """Validate and symmetrize once, in the covariance's actual units."""
    if not np.isfinite(covariance).all():
        raise ValueError(f"{error_prefix}_invalid")
    tolerance = (
        8
        * len(covariance)
        * np.finfo(float).eps
        * np.linalg.norm(covariance, ord=np.inf)
    )
    if np.max(np.abs(covariance - covariance.T)) > tolerance:
        raise ValueError(f"{error_prefix}_invalid")
    covariance = (covariance + covariance.T) / 2
    if np.linalg.eigvalsh(covariance)[0] < -tolerance:
        raise ValueError(f"{error_prefix}_not_psd")
    return covariance


def _weighted_fit(
    design: np.ndarray, returns: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Solve the sqrt(cap)-weighted objective without normal equations."""
    root_weights = np.sqrt(weights / weights.max())
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * root_weights[:, None], returns * root_weights, rcond=None
    )
    if rank != design.shape[1]:
        return None
    residuals = returns - design @ coefficients
    # An exactly explained ordinary return should not create specific risk.
    tolerance = 1e-12 * float(np.max(np.abs(returns)))
    residuals[np.abs(residuals) <= tolerance] = 0.0
    return coefficients, residuals


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
    # Anchor before centering so a constant column retains exact zero variance.
    centered = values - values[0]
    centered -= np.average(centered, axis=0, weights=weights)
    weighted = centered * np.sqrt(weights[:, None])
    if diagonal_shrinkage == 1:
        covariance = np.diag(np.sum(weighted * weighted, axis=0))
    else:
        covariance = weighted.T @ weighted
        covariance = (
            1 - diagonal_shrinkage
        ) * covariance + diagonal_shrinkage * np.diag(np.diag(covariance))
    covariance *= annualization_factor
    if not np.isfinite(covariance).all():
        raise ValueError("factor_risk_covariance_non_finite")
    covariance = _validated_covariance(
        covariance, error_prefix="factor_risk_covariance"
    )
    return pd.DataFrame(covariance, index=columns, columns=columns)


def _ewma_weights(ages: np.ndarray, half_life: float) -> np.ndarray:
    ages = np.asarray(ages, dtype="float64")
    values = np.exp2(-(ages - ages.min()) / half_life)
    return values / values.sum()

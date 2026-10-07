"""Causal rolling market-model numerical state shared by factors and risk."""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _CorrelationMoments
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
)


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
        self._moments: dict[str, _CorrelationMoments] = {}
        self._market_returns: dict[str, array] = {}
        self._residuals: dict[str, array] = {}
        self._previous_weights: dict[str, float] = {}
        self._asset_markets: dict[str, int] = {}
        self._pending_market_gross: dict[str, Fraction | None] = {}

    def observe(
        self,
        panel: pd.DataFrame,
        *,
        open_market_types: frozenset[int] | None = None,
        asset_market_types: Mapping[str, int] | None = None,
    ) -> MarketModelRiskStyles:
        required = {
            "asset_id",
            "period_return",
            "float_market_cap",
        }
        if open_market_types is not None:
            required.add("market_type")
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
        if open_market_types is not None:
            frame["market_type"] = strict_integer_series(
                frame["market_type"],
                error="market_model_risk_market_type_invalid",
                allow_null=True,
            )
        for column in ("period_return", "float_market_cap"):
            frame[column] = strict_finite_numeric_series(
                frame[column],
                error="market_model_risk_numeric_invalid",
                allow_null=True,
            ).astype("float64")
        if (
            frame["asset_id"].duplicated().any()
            or frame["float_market_cap"].le(0.0).any()
        ):
            raise ValueError("market_model_risk_key_invalid")
        if open_market_types is not None:
            known = frame["market_type"].notna()
            frame["eligible"] = (
                frame["eligible"] if "eligible" in frame else True
            ) & known
            frame.loc[~known, ["period_return", "float_market_cap"]] = np.nan
        eligible = frame.loc[frame.eligible] if "eligible" in frame else frame
        if eligible[["period_return", "float_market_cap"]].isna().any().any():
            raise ValueError("market_model_risk_numeric_invalid")
        frame = frame.sort_values("asset_id", kind="mergesort").reset_index(drop=True)
        if open_market_types is not None:
            known = frame.loc[frame.market_type.notna()]
            self._asset_markets.update(
                zip(known.asset_id, known.market_type, strict=True)
            )
            if asset_market_types is not None:
                self._asset_markets.update(
                    (asset, asset_market_types[asset])
                    for asset in self._returns
                    if asset in asset_market_types
                )
        absent = self._returns.keys() - set(frame.asset_id)
        closed_assets = {
            asset
            for asset in absent
            if open_market_types is not None
            and asset in self._asset_markets
            and self._asset_markets[asset] not in open_market_types
        }
        for asset in absent - closed_assets:
            del self._returns[asset]
            del self._moments[asset]
            del self._market_returns[asset]
            del self._residuals[asset]
            self._previous_weights.pop(asset, None)
            self._asset_markets.pop(asset, None)
            self._pending_market_gross.pop(asset, None)

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
        market_weight = Fraction()
        market_total = Fraction()
        for _, value, weight, _, eligible in weighted:
            if eligible and weight is not None:
                exact_weight = Fraction(weight)
                market_weight += exact_weight
                market_total += exact_weight * Fraction(value)
        market_return = float(market_total / market_weight) if market_weight else np.nan
        for asset in closed_assets:
            gross = self._pending_market_gross.get(asset, Fraction(1))
            self._pending_market_gross[asset] = (
                gross * (1 + Fraction.from_float(float(market_return)))
                if gross is not None and np.isfinite(market_return)
                else None
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
            if asset not in self._moments:
                self._moments[asset] = _CorrelationMoments()
            moments = self._moments[asset]
            returns = self._returns.setdefault(asset, array("d"))
            markets = self._market_returns.setdefault(asset, array("d"))
            residuals = self._residuals.setdefault(asset, array("d"))
            interval_market_return = market_return
            if asset in self._pending_market_gross:
                gross = self._pending_market_gross.pop(asset)
                interval_market_return = (
                    float(gross * (1 + Fraction.from_float(float(market_return))) - 1)
                    if gross is not None and np.isfinite(market_return)
                    else np.nan
                )
            if np.isfinite(interval_market_return) and np.isfinite(return_value):
                moments.update(float(interval_market_return), return_value)
                returns.append(return_value)
                markets.append(interval_market_return)
                if len(returns) > self.window:
                    moments.remove(markets.pop(0), returns.pop(0))
            else:
                del returns[:]
                del markets[:]
                self._moments[asset] = _CorrelationMoments()
            beta = np.nan
            residual = np.nan
            if len(returns) == self.window:
                market_mean, return_mean, market_square, _, product = moments.averages()
                variance = market_square - market_mean * market_mean
                if variance > 0:
                    slope = (product - market_mean * return_mean) / variance
                    beta = float(slope)
                    residual = float(
                        Fraction(return_value)
                        - return_mean
                        - slope
                        * (Fraction(float(interval_market_return)) - market_mean)
                    )
            residuals.append(residual)
            if len(residuals) > self.residual_window:
                residuals.pop(0)
            betas[asset] = float(beta)
            volatility = np.nan
            if len(residuals) == self.residual_window:
                values = np.asarray(residuals)
                residual_scale = float(np.max(np.abs(values)))
                volatility = (
                    0.0
                    if residual_scale == 0.0
                    else float(np.std(values / residual_scale, ddof=0) * residual_scale)
                )
            volatilities[asset] = volatility
        self._previous_weights = {
            **{
                asset: self._previous_weights[asset]
                for asset in closed_assets
                if asset in self._previous_weights
            },
            **{
                asset: current_weight
                for (
                    asset,
                    _return,
                    _previous_weight,
                    current_weight,
                    _eligible,
                ) in weighted
                if np.isfinite(current_weight)
            },
        }

        return MarketModelRiskStyles(
            beta=pd.Series(betas, dtype="float64", name="beta_252d"),
            residual_volatility=pd.Series(
                volatilities,
                dtype="float64",
                name="residual_volatility_60d",
            ),
        )

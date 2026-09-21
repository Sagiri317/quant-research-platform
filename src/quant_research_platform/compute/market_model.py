"""Causal rolling market-model numerical state shared by factors and risk."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import frexp, ldexp

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_finite_numeric_series


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

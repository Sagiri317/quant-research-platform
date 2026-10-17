"""Causal rolling market-model numerical state shared by factors and risk."""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import ceil

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _CorrelationMoments
from quant_research_platform.contracts.validation import (
    finite_exact_float,
    strict_finite_numeric_series,
    strict_integer_series,
    strict_timestamp_series,
)


@dataclass(frozen=True)
class MarketModelRiskStyles:
    beta: pd.Series
    residual_volatility: pd.Series


class MarketModelRiskState:
    """Causal rolling market model used by the six formal risk styles."""

    def __init__(
        self,
        *,
        window: int = 252,
        residual_window: int = 60,
        min_fresh_fraction: float = 0.75,
    ):
        if (
            isinstance(min_fresh_fraction, (bool, np.bool_))
            or not np.isfinite(min_fresh_fraction)
            or not 0 < min_fresh_fraction <= 1
        ):
            raise ValueError("market_model_min_fresh_fraction_invalid")
        self.min_fresh_fraction = min_fresh_fraction
        self.window = window
        self.residual_window = residual_window
        self._listings: dict[str, pd.Timestamp | tuple[pd.Timestamp, int]] = {}
        self._returns: dict[str, array] = {}
        self._moments: dict[str, _CorrelationMoments] = {}
        self._market_returns: dict[str, array] = {}
        self._residuals: dict[str, array] = {}
        self._previous_weights: dict[str, float] = {}
        self._asset_markets: dict[str, int] = {}
        self._pending_market_gross: dict[str, Fraction | None] = {}
        self._previous_market_complete = True

    def observe(
        self,
        panel: pd.DataFrame,
        *,
        open_market_types: frozenset[int] | None = None,
        asset_market_types: Mapping[str, int] | None = None,
        market_input_complete: bool = True,
    ) -> MarketModelRiskStyles:
        if type(market_input_complete) is not bool:
            raise ValueError("market_model_input_completeness_invalid")
        required = {
            "asset_id",
            "listing_date",
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
        optional = [
            name
            for name in (
                "eligible",
                "_calendar_segment",
                "filling_flag",
                "is_trading",
                "is_a_share_scope",
                "is_st",
                "is_pt",
                "has_252_trading_sessions",
            )
            if name in panel
        ]
        frame = panel.loc[:, list(required) + optional].copy()
        # A bad cap is unavailable only to consumers of that asset's weight.
        raw_caps = frame["float_market_cap"]
        caps = pd.to_numeric(raw_caps, errors="coerce")
        if not pd.api.types.is_numeric_dtype(raw_caps) or pd.api.types.is_bool_dtype(
            raw_caps
        ):
            caps = caps.mask(
                raw_caps.map(lambda value: isinstance(value, (bool, np.bool_)))
            )
        frame["float_market_cap"] = caps.where(np.isfinite(caps) & caps.gt(0))
        unknown_quote = pd.Series(False, index=frame.index)
        for name in ("period_return", "float_market_cap"):
            if name in frame:
                unknown_quote |= frame[name].isna()
        if "filling_flag" in frame:
            unknown_quote |= ~frame.filling_flag.isin((0, 1))
        elif "is_trading" in frame:
            unknown_quote |= frame.is_trading.isna()
        known_excluded = pd.Series(False, index=frame.index)
        for name, excluded_value in (
            ("is_a_share_scope", False),
            ("is_st", True),
            ("is_pt", True),
            ("has_252_trading_sessions", False),
        ):
            if name in frame:
                unknown_quote |= frame[name].isna()
                known_excluded |= frame[name].eq(excluded_value).fillna(False)
        frame["unknown_quote"] = unknown_quote & ~known_excluded
        frame["fresh"] = (
            frame.filling_flag.eq(0).fillna(False)
            if "filling_flag" in frame
            else frame.is_trading.eq(True).fillna(False)
            if "is_trading" in frame
            else True
        )
        frame["suspended"] = (
            frame.filling_flag.eq(1).fillna(False)
            if "filling_flag" in frame
            else frame.is_trading.eq(False).fillna(False)
            if "is_trading" in frame
            else False
        )
        frame["asset_id"] = frame["asset_id"].astype("string")
        frame["listing_date"] = strict_timestamp_series(
            frame["listing_date"],
            error="market_model_risk_listing_invalid",
            allow_null=True,
        )
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
        if frame["asset_id"].duplicated().any():
            raise ValueError("market_model_risk_key_invalid")
        if open_market_types is not None:
            known = frame["market_type"].notna()
            frame["eligible"] = (
                frame["eligible"] if "eligible" in frame else True
            ) & known
            frame.loc[~known, ["period_return", "float_market_cap"]] = np.nan
            frame.loc[~known, "unknown_quote"] = True
        frame = frame.sort_values("asset_id", kind="mergesort").reset_index(drop=True)
        if open_market_types is not None and asset_market_types is not None:
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
        listings = dict(zip(frame.asset_id, frame.listing_date, strict=True))
        if "_calendar_segment" in frame:
            listings = {
                row.asset_id: (row.listing_date, segment)
                for row, segment in zip(
                    frame.itertuples(index=False),
                    frame["_calendar_segment"],
                    strict=True,
                )
            }
        changed = {
            asset
            for asset, listing in listings.items()
            if asset in self._listings and self._listings[asset] != listing
        }
        # Resetting regression history must not erase evidence that an unknown
        # incumbent was required in the lagged market proxy.
        required_previous = {}
        for row in frame.itertuples(index=False):
            prior = self._listings.get(row.asset_id)
            prior_listing = prior[0] if isinstance(prior, tuple) else prior
            known_new_listing = (
                prior_listing is not None
                and pd.notna(row.listing_date)
                and pd.notna(prior_listing)
                and row.listing_date != prior_listing
            )
            if (
                row.unknown_quote
                and not known_new_listing
                and row.asset_id in self._previous_weights
            ):
                required_previous[row.asset_id] = self._previous_weights[row.asset_id]
        for asset in (absent - closed_assets) | changed:
            self._listings.pop(asset, None)
            del self._returns[asset]
            del self._moments[asset]
            del self._market_returns[asset]
            del self._residuals[asset]
            self._previous_weights.pop(asset, None)
            self._asset_markets.pop(asset, None)
            self._pending_market_gross.pop(asset, None)
        self._listings.update(listings)
        if open_market_types is not None:
            known = frame.loc[frame.market_type.notna()]
            self._asset_markets.update(
                zip(known.asset_id, known.market_type, strict=True)
            )

        weighted = [
            (
                str(row.asset_id),
                float(row.period_return),
                self._previous_weights.get(
                    str(row.asset_id), required_previous.get(str(row.asset_id))
                ),
                float(row.float_market_cap),
                bool(row.eligible) if "eligible" in frame else True,
                bool(row.unknown_quote),
                bool(row.fresh),
                bool(row.suspended),
            )
            for row in frame.itertuples(index=False)
        ]
        market_weight = Fraction()
        market_total = Fraction()
        for _, value, weight, _, eligible, unknown_quote, _, _ in weighted:
            if weight is not None and unknown_quote:
                market_weight = Fraction()
                break
            if eligible and weight is not None:
                if np.isnan(value):
                    # A required member makes the whole session unknown.
                    market_weight = Fraction()
                    break
                exact_weight = Fraction(weight)
                market_weight += exact_weight
                market_total += exact_weight * Fraction(value)
        # A selector gap also leaves the next session's lagged weights unknown.
        market_return = (
            float(market_total / market_weight)
            if market_weight
            and market_input_complete
            and self._previous_market_complete
            else np.nan
        )
        self._previous_market_complete = market_input_complete and not any(
            unknown
            and not np.isfinite(cap)
            and (eligible or previous_weight is not None)
            for _, _, previous_weight, cap, eligible, unknown, _, _ in weighted
        )
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
            _unknown_quote,
            fresh,
            suspended,
        ) in weighted:
            if asset not in self._moments:
                self._moments[asset] = _CorrelationMoments()
            moments = self._moments[asset]
            returns = self._returns.setdefault(asset, array("d"))
            markets = self._market_returns.setdefault(asset, array("d"))
            residuals = self._residuals.setdefault(asset, array("d"))
            interval_market_return = market_return
            if suspended:
                gross = self._pending_market_gross.get(asset, Fraction(1))
                self._pending_market_gross[asset] = (
                    gross * (1 + Fraction.from_float(float(market_return)))
                    if gross is not None and np.isfinite(market_return)
                    else None
                )
                interval_market_return = np.nan
            elif asset in self._pending_market_gross:
                gross = self._pending_market_gross.pop(asset)
                converted = (
                    finite_exact_float(
                        gross * (1 + Fraction.from_float(float(market_return))) - 1
                    )
                    if gross is not None and np.isfinite(market_return)
                    else None
                )
                interval_market_return = np.nan if converted is None else converted
            paired = (
                fresh
                and np.isfinite(interval_market_return)
                and np.isfinite(return_value)
            )
            returns.append(return_value if paired else np.nan)
            markets.append(interval_market_return if paired else np.nan)
            if paired:
                moments.update(float(interval_market_return), return_value)
            if len(returns) > self.window:
                old_market, old_return = markets.pop(0), returns.pop(0)
                if np.isfinite(old_market) and np.isfinite(old_return):
                    moments.remove(old_market, old_return)
            beta = np.nan
            residual = np.nan
            if len(returns) == self.window and moments.count >= max(
                2, ceil(self.min_fresh_fraction * self.window)
            ):
                market_mean, return_mean, market_square, _, product = moments.averages()
                variance = market_square - market_mean * market_mean
                if variance > 0:
                    slope = (product - market_mean * return_mean) / variance
                    converted = finite_exact_float(slope)
                    if converted is not None:
                        beta = converted
                    if converted is not None and paired:
                        converted = finite_exact_float(
                            Fraction(return_value)
                            - return_mean
                            - slope
                            * (Fraction(float(interval_market_return)) - market_mean)
                        )
                        if converted is not None:
                            residual = converted
            residuals.append(residual)
            if len(residuals) > self.residual_window:
                residuals.pop(0)
            betas[asset] = float(beta)
            volatility = np.nan
            if len(residuals) == self.residual_window:
                values = np.asarray(residuals)
                values = values[np.isfinite(values)]
                if len(values) < ceil(self.min_fresh_fraction * self.residual_window):
                    volatilities[asset] = np.nan
                    continue
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
                # An unknown incumbent's last weight is retained only as a
                # membership obligation, never consumed for a valid proxy.
                asset: current_weight
                if np.isfinite(current_weight)
                else previous_weight
                for (
                    asset,
                    _return,
                    previous_weight,
                    current_weight,
                    _eligible,
                    unknown_quote,
                    _fresh,
                    _suspended,
                ) in weighted
                if np.isfinite(current_weight)
                or (unknown_quote and previous_weight is not None)
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

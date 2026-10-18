"""Causal rolling market-model numerical state shared by factors and risk."""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
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
        self._market_returns: dict[str, array] = {}
        self._residuals: dict[str, array] = {}
        self._previous_weights: dict[str, float] = {}
        self._asset_markets: dict[str, int] = {}
        self._pending_market_returns: dict[str, float] = {}
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
        unknown_quote = frame["period_return"].isna()
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
        # Today's proxy consumes lagged caps; today's absent cap only makes
        # the next session's weight unknown, unless membership is excluded.
        frame["unknown_weight"] = frame.float_market_cap.isna() & ~known_excluded
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
            frame.loc[~known, ["unknown_quote", "unknown_weight"]] = True
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
                (row.unknown_quote or row.unknown_weight)
                and not known_new_listing
                and row.asset_id in self._previous_weights
            ):
                required_previous[row.asset_id] = self._previous_weights[row.asset_id]
        for asset in (absent - closed_assets) | changed:
            self._listings.pop(asset, None)
            del self._returns[asset]
            del self._market_returns[asset]
            del self._residuals[asset]
            self._previous_weights.pop(asset, None)
            self._asset_markets.pop(asset, None)
            self._pending_market_returns.pop(asset, None)
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
                # A weight carried across an unknown lifecycle boundary is
                # only a membership obligation, not a usable lagged weight.
                bool(row.unknown_quote)
                or (
                    row.asset_id in required_previous
                    and row.asset_id not in self._previous_weights
                ),
                bool(row.unknown_weight),
                bool(row.fresh),
                bool(row.suspended),
            )
            for row in frame.itertuples(index=False)
        ]
        # A selector gap also leaves the next session's lagged weights unknown.
        market_return = np.nan
        members = [
            (value, weight)
            for _, value, weight, _, eligible, _, _, _, _ in weighted
            if eligible and weight is not None
        ]
        if (
            members
            and market_input_complete
            and self._previous_market_complete
            and not any(
                weight is not None and unknown_quote
                for _, _, weight, _, _, unknown_quote, _, _, _ in weighted
            )
        ):
            market_values, weights = np.asarray(members).T
            weights = weights / weights.max()
            anchor = market_values[weights.argmax()]
            market_return = float(
                anchor + np.dot(weights, market_values - anchor) / weights.sum()
            )
        self._previous_market_complete = market_input_complete and not any(
            unknown_weight and (eligible or previous_weight is not None)
            for _, _, previous_weight, _, eligible, _, unknown_weight, _, _ in weighted
        )
        for asset in closed_assets:
            pending = self._pending_market_returns.get(asset, 0.0)
            self._pending_market_returns[asset] = (
                pending + (1 + pending) * market_return
            )

        assets = frame.asset_id.to_list()
        betas = np.full(len(assets), np.nan)
        residual_values = np.full(len(assets), np.nan)
        mature = []
        for position, (
            asset,
            return_value,
            _previous_weight,
            _current_weight,
            _eligible,
            _unknown_quote,
            _unknown_weight,
            fresh,
            suspended,
        ) in enumerate(weighted):
            returns = self._returns.setdefault(asset, array("d"))
            markets = self._market_returns.setdefault(asset, array("d"))
            interval_market_return = market_return
            if suspended:
                pending = self._pending_market_returns.get(asset, 0.0)
                self._pending_market_returns[asset] = (
                    pending + (1 + pending) * market_return
                )
                interval_market_return = np.nan
            elif asset in self._pending_market_returns:
                pending = self._pending_market_returns.pop(asset)
                interval_market_return = pending + (1 + pending) * market_return
            paired = (
                fresh
                and np.isfinite(interval_market_return)
                and np.isfinite(return_value)
            )
            returns.append(return_value if paired else np.nan)
            markets.append(interval_market_return if paired else np.nan)
            if len(returns) > self.window:
                markets.pop(0)
                returns.pop(0)
            if len(returns) == self.window:
                mature.append(position)

        if mature:
            markets = np.asarray([self._market_returns[assets[i]] for i in mature])
            returns = np.asarray([self._returns[assets[i]] for i in mature])
            paired = np.isfinite(markets) & np.isfinite(returns)
            counts = paired.sum(axis=1)
            enough = counts >= max(2, ceil(self.min_fresh_fraction * self.window))
            # Center observed differences, avoiding raw second-moment subtraction.
            centered = []
            for values in (markets, returns):
                anchors = values[np.arange(len(values)), paired.argmax(axis=1)]
                differences = np.where(paired, values - anchors[:, None], 0.0)
                means = np.divide(
                    differences.sum(axis=1),
                    counts,
                    out=np.zeros(len(values)),
                    where=counts > 0,
                )
                centered.append(np.where(paired, differences - means[:, None], 0.0))
            x, y = centered
            variance = np.sum(x * x, axis=1)
            slopes = np.divide(
                np.sum(x * y, axis=1),
                variance,
                out=np.full(len(mature), np.nan),
                where=enough & (variance > 0),
            )
            slopes[~np.isfinite(slopes)] = np.nan
            betas[mature] = slopes
            residual_values[mature] = np.where(
                paired[:, -1], y[:, -1] - slopes * x[:, -1], np.nan
            )
        residual_values[~np.isfinite(residual_values)] = np.nan

        mature = []
        for position, asset in enumerate(assets):
            residuals = self._residuals.setdefault(asset, array("d"))
            residuals.append(residual_values[position])
            if len(residuals) > self.residual_window:
                residuals.pop(0)
            if len(residuals) == self.residual_window:
                mature.append(position)
        volatilities = np.full(len(assets), np.nan)
        if mature:
            values = np.asarray([self._residuals[assets[i]] for i in mature])
            finite = np.isfinite(values)
            counts = finite.sum(axis=1)
            valid = counts >= ceil(self.min_fresh_fraction * self.residual_window)
            values, finite, counts = values[valid], finite[valid], counts[valid]
            means = np.sum(np.where(finite, values, 0.0), axis=1) / counts
            centered = np.where(finite, values - means[:, None], 0.0)
            volatilities[np.asarray(mature)[valid]] = np.sqrt(
                np.sum(centered * centered, axis=1) / counts
            )
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
                    unknown_weight,
                    _fresh,
                    _suspended,
                ) in weighted
                if np.isfinite(current_weight)
                or ((unknown_quote or unknown_weight) and previous_weight is not None)
            },
        }

        return MarketModelRiskStyles(
            beta=pd.Series(betas, index=assets, dtype="float64", name="beta_252d"),
            residual_volatility=pd.Series(
                volatilities,
                index=assets,
                dtype="float64",
                name="residual_volatility_60d",
            ),
        )

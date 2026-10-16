"""Formal simple portfolio target construction."""

from __future__ import annotations

from fractions import Fraction
from math import floor
from typing import Literal

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.validation import (
    strict_finite_float,
    strict_real_numeric_series,
)
from quant_research_platform.research.features.transforms import _mad_winsorize
from quant_research_platform.research.portfolios.optimizer import (
    _project_neutral_target,
)

TargetMethod = Literal["equal_weight", "float_cap_weight"]
PortfolioMode = Literal["long_only", "long_short"]


def build_simple_target_weights(
    signal: pd.Series,
    *,
    eligible: pd.Series,
    method: TargetMethod = "equal_weight",
    portfolio_mode: PortfolioMode = "long_only",
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    float_market_cap: pd.Series | None = None,
    neutrality: Literal["none", "industry", "size", "industry_size"] = "none",
    industry: pd.Series | None = None,
) -> pd.Series:
    """Build one target from current eligible signal and capitalization facts."""

    assets = _canonical_assets(signal, eligible)
    values = _numeric_series(signal, assets, "target_signal_invalid")
    eligibility = _bool_series(eligible, assets, "target_eligibility_invalid")
    if type(method) is not str or method not in {"equal_weight", "float_cap_weight"}:
        raise ValueError("target_method_invalid")
    if type(portfolio_mode) is not str or portfolio_mode not in {
        "long_only",
        "long_short",
    }:
        raise ValueError("target_portfolio_mode_invalid")
    top_quantile = _validate_quantile(top_quantile, "target_top_quantile_invalid")
    bottom_quantile = _validate_quantile(
        bottom_quantile,
        "target_bottom_quantile_invalid",
    )
    if portfolio_mode == "long_short" and top_quantile + bottom_quantile > 1.0:
        raise ValueError("target_quantiles_invalid")
    context = None
    if neutrality != "none":
        if portfolio_mode != "long_only":
            raise ValueError("target_neutrality_requires_long_only")
        context = _neutrality_context(
            eligibility, float_market_cap, industry, neutrality
        )
        eligibility = eligibility & eligibility.index.isin(context.index)
    valid = eligibility & values.notna() & np.isfinite(values)
    if context is None and bool((eligibility & ~valid).any()):
        raise ValueError("target_signal_invalid")
    if int(valid.sum()) == 0:
        raise ValueError("target_selection_unavailable")
    valid_values = values.loc[valid]
    top = _select_boundary(valid_values, top_quantile, ascending=False)
    bottom = (
        _select_boundary(valid_values, bottom_quantile, ascending=True)
        if portfolio_mode == "long_short"
        else pd.Index([], dtype="string")
    )
    if portfolio_mode == "long_short" and top.intersection(bottom).size:
        raise ValueError("target_long_short_boundary_overlap")
    capitals = _selected_capitals(
        float_market_cap,
        assets,
        selected=top.union(bottom),
        method=method,
    )
    result = pd.Series(0.0, index=assets, dtype="float64", name="target_weight")
    leg_gross = 0.5 if portfolio_mode == "long_short" else 1.0
    _assign_leg(
        result,
        top,
        sign=1.0,
        gross=leg_gross,
        method=method,
        capitals=capitals,
    )
    if portfolio_mode == "long_short":
        _assign_leg(
            result,
            bottom,
            sign=-1.0,
            gross=leg_gross,
            method=method,
            capitals=capitals,
        )
    if (
        not np.isfinite(result.to_numpy()).all()
        or not np.isclose(result.sum(), 0.0 if portfolio_mode == "long_short" else 1.0)
        or not np.isclose(result.abs().sum(), 1.0)
    ):
        raise ValueError("target_weights_invalid")
    return _project_neutral_target(result, context) if context is not None else result


def _canonical_assets(signal: pd.Series, eligible: pd.Series) -> pd.Index:
    if (
        not isinstance(signal, pd.Series)
        or not isinstance(eligible, pd.Series)
        or not signal.index.is_unique
        or not eligible.index.is_unique
        or set(signal.index) != set(eligible.index)
        or not len(signal.index)
        or any(type(asset) is not str or not asset for asset in signal.index)
    ):
        raise ValueError("target_asset_index_invalid")
    return pd.Index(sorted(signal.index), dtype="string")


def _numeric_series(values: pd.Series, assets: pd.Index, error: str) -> pd.Series:
    return strict_real_numeric_series(values.reindex(assets), error=error)


def _bool_series(values: pd.Series, assets: pd.Index, error: str) -> pd.Series:
    if not pd.api.types.is_bool_dtype(values):
        raise ValueError(error)
    result = values.reindex(assets)
    if result.isna().any():
        raise ValueError(error)
    return result.astype(bool)


def _select_boundary(
    values: pd.Series,
    quantile: float,
    *,
    ascending: bool,
) -> pd.Index:
    count = max(1, floor(len(values) * quantile))
    ranks = values.rank(method="average", ascending=ascending)
    boundary_rank = float(
        ranks.loc[values.sort_values(ascending=ascending).index[count - 1]]
    )
    return pd.Index(values.index[ranks <= boundary_rank], dtype="string")


def _selected_capitals(
    values: pd.Series | None,
    assets: pd.Index,
    *,
    selected: pd.Index,
    method: TargetMethod,
) -> pd.Series | None:
    if method == "equal_weight":
        return None
    if values is None:
        raise ValueError("target_float_market_cap_required")
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError("target_float_market_cap_invalid")
    capitals = strict_real_numeric_series(
        values.reindex(assets), error="target_float_market_cap_invalid"
    )
    selected_values = capitals.reindex(selected)
    if (
        selected_values.isna().any()
        or not np.isfinite(selected_values.to_numpy(dtype="float64")).all()
        or selected_values.le(0.0).any()
    ):
        raise ValueError("target_float_market_cap_invalid")
    return capitals


def _assign_leg(
    result: pd.Series,
    selected: pd.Index,
    *,
    sign: float,
    gross: float,
    method: TargetMethod,
    capitals: pd.Series | None,
) -> None:
    if selected.empty:
        return
    if method == "equal_weight":
        weights = pd.Series(
            gross / len(selected),
            index=selected,
            dtype="float64",
        )
    else:
        assert capitals is not None
        selected_capitals = [
            Fraction(float(value)) for value in capitals.reindex(selected)
        ]
        scale = Fraction(gross) / sum(selected_capitals)
        weights = pd.Series(
            [float(scale * value) for value in selected_capitals],
            index=selected,
            dtype="float64",
        )
    result.loc[selected] = sign * weights


def _validate_quantile(value: float, error: str) -> float:
    numeric = strict_finite_float(value, error=error)
    if not 0.0 < numeric <= 1.0:
        raise ValueError(error)
    return numeric


def _neutrality_context(eligible, float_market_cap, industry, neutrality):
    """Use the full lawful benchmark domain, including assets without scores."""
    if type(neutrality) is not str or neutrality not in {
        "industry",
        "size",
        "industry_size",
    }:
        raise ValueError("target_neutrality_invalid")
    assets = eligible.index
    if (
        not isinstance(float_market_cap, pd.Series)
        or not float_market_cap.index.is_unique
    ):
        raise ValueError("target_float_market_cap_required")
    capitals = _numeric_series(
        float_market_cap, assets, "target_float_market_cap_invalid"
    )
    domain = eligible & capitals.gt(0) & np.isfinite(capitals)
    if neutrality in {"industry", "industry_size"}:
        if not isinstance(industry, pd.Series) or not industry.index.is_unique:
            raise ValueError("target_industry_required")
        industry = industry.reindex(assets)
        if any(type(value) is not str or not value for value in industry.dropna()):
            raise ValueError("target_industry_invalid")
        domain &= industry.notna()
    capitals = capitals.loc[domain]
    if capitals.empty:
        raise PortfolioTargetBlocked("target_neutrality_benchmark_empty")
    exact = [Fraction(float(value)) for value in capitals]
    total = sum(exact)
    context = pd.DataFrame(
        {"benchmark_weight": [float(value / total) for value in exact]},
        index=capitals.index,
    )
    if neutrality in {"industry", "industry_size"}:
        codes = industry.reindex(context.index)
        for code in sorted(codes.unique()):
            context[f"industry:{code}"] = codes.eq(code).astype(float)
    if neutrality in {"size", "industry_size"}:
        size = np.log(capitals)
        context["size"] = _mad_winsorize(size, {0: np.arange(len(size))})
    return context

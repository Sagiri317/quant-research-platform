"""Formal simple portfolio target construction."""

from __future__ import annotations

from math import floor
from typing import Literal

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.validation import (
    _real_numeric_series,
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
    benchmark_weights: pd.Series | None = None,
    total_market_cap: pd.Series | None = None,
    size_cap_column: Literal[
        "total_market_cap", "float_market_cap"
    ] = "total_market_cap",
    industry_neutrality_limit: float = 0.02,
    size_neutrality_limit: float = 0.2,
) -> pd.Series:
    """Build one target from current eligible signal and capitalization facts."""

    assets = _canonical_assets(signal, eligible)
    values = _real_numeric_series(signal.reindex(assets), error="target_signal_invalid")
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
    if type(size_cap_column) is not str or size_cap_column not in {
        "total_market_cap",
        "float_market_cap",
    }:
        raise ValueError("target_size_cap_column_invalid")
    context = None
    if neutrality != "none":
        if portfolio_mode != "long_only":
            raise ValueError("target_neutrality_requires_long_only")
        context = _neutrality_context(
            eligibility,
            float_market_cap,
            industry,
            neutrality,
            benchmark_weights=benchmark_weights,
            size_market_cap=(
                total_market_cap
                if size_cap_column == "total_market_cap"
                else float_market_cap
            ),
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
    return (
        _project_neutral_target(
            result,
            context,
            industry_limit=industry_neutrality_limit,
            size_limit=size_neutrality_limit,
        )
        if context is not None
        else result
    )


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
    try:
        capitals = strict_real_numeric_series(
            values.reindex(selected), error="target_float_market_cap_invalid"
        )
    except ValueError as exc:
        raise PortfolioTargetBlocked("target_float_market_cap_missing") from exc
    capitals = capitals.where(np.isfinite(capitals) & capitals.gt(0))
    if capitals.reindex(selected).isna().any():
        raise PortfolioTargetBlocked("target_float_market_cap_missing")
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
        selected_capitals = capitals.reindex(selected)
        weights = gross * selected_capitals / selected_capitals.sum()
    result.loc[selected] = sign * weights


def _validate_quantile(value: float, error: str) -> float:
    numeric = strict_finite_float(value, error=error)
    if not 0.0 < numeric <= 1.0:
        raise ValueError(error)
    return numeric


def _neutrality_context(
    eligible,
    float_market_cap,
    industry,
    neutrality,
    *,
    size_market_cap=None,
    benchmark_weights=None,
):
    """Use the full lawful benchmark domain, including assets without scores."""
    if type(neutrality) is not str or neutrality not in {
        "industry",
        "size",
        "industry_size",
    }:
        raise ValueError("target_neutrality_invalid")
    assets = eligible.index
    if benchmark_weights is None:
        if (
            not isinstance(float_market_cap, pd.Series)
            or not float_market_cap.index.is_unique
        ):
            raise ValueError("target_float_market_cap_required")
        capitals = _numeric_series(
            float_market_cap, assets, "target_float_market_cap_invalid"
        )
        domain = eligible & capitals.gt(0) & np.isfinite(capitals)
        weights = capitals.loc[domain]
        weights = weights / weights.sum()
    else:
        if (
            not isinstance(benchmark_weights, pd.Series)
            or not benchmark_weights.index.is_unique
        ):
            raise ValueError("portfolio_benchmark_weights_invalid")
        if benchmark_weights.empty:
            raise PortfolioTargetBlocked("target_neutrality_benchmark_empty")
        weights = _numeric_series(
            benchmark_weights,
            benchmark_weights.index,
            "portfolio_benchmark_weights_invalid",
        )
        if not (np.isfinite(weights) & weights.gt(0)).all() or not np.isclose(
            weights.sum(), 1.0, rtol=0.0, atol=1e-8
        ):
            raise ValueError("portfolio_benchmark_weights_invalid")
        assets = eligible.index[eligible].union(weights.index)
        domain = pd.Series(True, index=assets)
    weights = weights.reindex(assets, fill_value=0.0)
    if neutrality in {"industry", "industry_size"}:
        if not isinstance(industry, pd.Series) or not industry.index.is_unique:
            raise ValueError("target_industry_required")
        industry = industry.reindex(assets)
        if any(type(value) is not str or not value for value in industry.dropna()):
            raise ValueError("target_industry_invalid")
        domain &= industry.notna()
    if not domain.any():
        raise PortfolioTargetBlocked("target_neutrality_benchmark_empty")
    context = pd.DataFrame({"benchmark_weight": weights.loc[domain]})
    if benchmark_weights is not None:
        context.attrs["eligible"] = eligible.reindex(context.index, fill_value=False)
    if neutrality in {"industry", "industry_size"}:
        context.attrs["unknown_industry_weight"] = float(weights.loc[~domain].sum())
        codes = industry.reindex(context.index)
        for code in sorted(codes.unique()):
            context[f"industry:{code}"] = codes.eq(code).astype(float)
    if neutrality in {"size", "industry_size"}:
        if size_market_cap is None:
            raise PortfolioTargetBlocked("target_neutrality_size_missing")
        if (
            not isinstance(size_market_cap, pd.Series)
            or not size_market_cap.index.is_unique
        ):
            raise ValueError("target_size_market_cap_invalid")
        size_capitals = _numeric_series(
            size_market_cap, context.index, "target_size_market_cap_invalid"
        )
        if not (size_capitals.gt(0) & np.isfinite(size_capitals)).all():
            raise PortfolioTargetBlocked("target_neutrality_size_missing")
        size = np.log(size_capitals)
        context["size"] = _stable_zscore(
            _mad_winsorize(size, {0: np.arange(len(size))})
        )
        if context["size"].isna().any():
            raise PortfolioTargetBlocked("target_neutrality_size_undefined")
    return context

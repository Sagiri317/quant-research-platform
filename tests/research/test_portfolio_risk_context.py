"""Native PIT history, independent WLS moments and dependency-local policy inputs."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.risk import FactorRiskConfig, FactorRiskModel
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.stock_pool import (
    load_stock_pool,
    stock_pool_rows,
)
from quant_research_platform.research.portfolios.risk_context import (
    portfolio_risk_context,
)
from tests.helpers.research_processed_inputs import assembly_source
from tests.research.test_hft_stock_pool import fact


def sources(tmp_path, *, decision_day="2024-01-05", unknown_clock=False, future=0.0):
    day = pd.Timestamp(decision_day)
    at = day + pd.Timedelta(hours=1, minutes=30)
    dates = pd.bdate_range(day - pd.Timedelta(days=15), day + pd.Timedelta(days=3))
    assets = ["a", "b", *(f"a{i}" for i in range(2, 20))]
    style = np.linspace(-2, 2, len(assets))
    cap = 100.0 + np.arange(len(assets))
    listing = pd.Timestamp("2000-01-01")
    model = FactorRiskModel(
        ("style",),
        FactorRiskConfig(
            factor_window=3,
            specific_window=3,
            minimum_factor_dates=2,
            minimum_specific_dates=2,
            minimum_cross_section=10,
            factor_half_life=2,
            specific_half_life=2,
            diagonal_shrinkage=0,
        ),
    )
    stocks = pd.DataFrame(
        {"source_asset_id": assets, "timestamp": listing, "listing_date": listing}
    )
    status = pd.DataFrame(
        {
            "asset_id": assets,
            "timestamp": listing,
            "listing_date": listing,
            "published": listing,
            "market_type": 1,
            "is_st": False,
            "is_pt": False,
            "is_delisting": False,
        }
    )
    natural = pd.date_range(dates.min() - pd.Timedelta(days=20), dates.max())
    calendar = pd.DataFrame(
        {
            "calendar_date": natural,
            "source_market_type": 1,
            "is_trading_day": natural.day_of_week < 5,
        }
    )
    pool = {
        "stock_master": fact(
            tmp_path / "stocks", stocks, ("source_asset_id", "timestamp")
        ),
        "status": fact(tmp_path / "status", status, ("asset_id", "timestamp")),
        "available_at_column": "published",
        "trading_calendar": fact(
            tmp_path / "calendar",
            calendar,
            ("source_market_type", "calendar_date"),
            "calendar_date",
        ),
    }
    risk, responses = [], []
    for i, date in enumerate(dates):
        cross = pd.DataFrame(
            {
                "market_id": "a_share",
                "asset_id": assets,
                "timestamp": date,
                "listing_date": listing,
                "published": date + pd.Timedelta(hours=7, minutes=5),
                "industry_standard_id": "test",
                "industry_level1_code": "A",
                "total_market_cap": cap,
                "float_market_cap": cap / 2,
                "style": style,
            }
        )
        if date >= day:
            cross["style"] += future * np.cos(np.arange(len(assets)))
        risk.append(cross)
        if i:
            response = cross[
                ["market_id", "asset_id", "timestamp", "listing_date"]
            ].copy()
            response["return_start_at"] = dates[i - 1] + pd.Timedelta(hours=7)
            response["return_available_at"] = date + pd.Timedelta(hours=7, minutes=5)
            response["is_trading"] = True
            response["period_return"] = (
                0.001 * i
                + 0.002 * np.cos(i) * style
                + np.sin(np.arange(len(assets)) + i) * 0.001
            )
            if date == dates[dates < day][-1]:
                response.loc[0, "return_available_at"] = at + pd.Timedelta(seconds=6)
                response.loc[0, "period_return"] = 0.08
            if date >= day:
                response["period_return"] += future
            responses.append(response)
    risk = pd.concat(risk, ignore_index=True)
    returns = pd.concat(responses, ignore_index=True)
    if unknown_clock:
        returns = returns.drop(columns="return_available_at")
    risk_root, _ = assembly_source(tmp_path, risk, data_id="risk")
    return_root, _ = assembly_source(tmp_path, returns, data_id="returns")
    config = {
        "inputs": {"risk": str(risk_root), "returns": str(return_root)},
        "risk_model": model.to_dict(),
        "stock_pool": pool,
        "risk_exposure_input": {"data_id": "risk", "available_at_column": "published"},
        "daily_return_input": {"data_id": "returns"},
    }
    return config, at, returns, style, cap


def test_native_provider_uses_full_domain_freezes_origins_and_replays_only_late_returns(
    tmp_path, monkeypatch
):
    from quant_research_platform.research.risk import FactorRiskState

    config, at, returns, style, cap = sources(tmp_path)
    observed = []
    original = FactorRiskState.observe_return_session

    def measured(self, cross, **kwargs):
        observed.append(pd.Timestamp(cross.timestamp.iloc[0]) if len(cross) else None)
        return original(self, cross, **kwargs)

    monkeypatch.setattr(FactorRiskState, "observe_return_session", measured)
    provider = portfolio_risk_context(
        config,
        window=TimeWindow(at, at + pd.Timedelta(days=1)),
        construction="absolute_risk",
    )
    # A one-name account cannot turn the risk regression into a one-name fit.
    decision = SimpleNamespace(timestamp=at, features=pd.DataFrame(index=["a"]))
    first = provider(decision)
    assert len(first["assets"]) == 20
    assert len(observed) == 3
    unchanged = provider(SimpleNamespace(timestamp=at + pd.Timedelta(seconds=3)))
    assert len(observed) == 3
    assert unchanged["factor_covariance"] is first["factor_covariance"]
    actual = provider(SimpleNamespace(timestamp=at + pd.Timedelta(seconds=6)))
    assert len(observed) == 6
    assert observed[:3] == observed[3:]
    assert not actual["factor_covariance"].equals(first["factor_covariance"])
    # Independent WLS and population EWMA formulas, on the complete origin domain.
    design = np.column_stack(
        (np.ones(len(style)), (style - style.mean()) / style.std())
    )
    weights = cap**0.25
    coefficients, residuals = [], []
    days = sorted(
        returns.loc[returns.timestamp < at.normalize(), "timestamp"].unique()
    )[-3:]
    for day in days:
        y = returns.loc[returns.timestamp.eq(day), "period_return"].to_numpy()
        beta = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
        coefficients.append(beta)
        residuals.append(y - design @ beta)
    beta = np.asarray(coefficients)
    decay = 2.0 ** (-np.arange(2, -1, -1) / 2)
    decay /= decay.sum()
    centered = beta - np.average(beta, axis=0, weights=decay)
    covariance = (centered * decay[:, None]).T @ centered * 252
    np.testing.assert_allclose(
        actual["factor_covariance"], covariance, rtol=1e-11, atol=1e-14
    )
    np.testing.assert_allclose(
        actual["assets"].specific_variance,
        pd.Series(
            np.asarray(residuals).T ** 2 @ decay * 252,
            index=returns.loc[returns.timestamp.eq(days[0]), "asset_id"],
        ).reindex(actual["assets"].index),
        rtol=1e-11,
        atol=1e-14,
    )
    # The following day appends one session; it does not refit prior sessions.
    provider(SimpleNamespace(timestamp=at + pd.Timedelta(days=1)))
    assert len(observed) == 7


def test_missing_return_session_keeps_real_calendar_ages_and_new_life_has_no_old_risk(
    tmp_path,
):
    from quant_research_platform.storage.datasets import read_dataset

    config, at, returns, style, cap = sources(tmp_path)
    days = sorted(
        returns.loc[returns.timestamp < at.normalize(), "timestamp"].unique()
    )[-3:]
    returns = returns.loc[returns.timestamp.ne(days[1])]
    root, _ = assembly_source(tmp_path, returns, data_id="missing_day")
    config["inputs"]["returns"] = str(root)
    actual = portfolio_risk_context(
        config,
        window=TimeWindow(at, at + pd.Timedelta(seconds=6)),
        construction="absolute_risk",
    )(SimpleNamespace(timestamp=at + pd.Timedelta(seconds=6)))
    design = np.column_stack(
        (np.ones(len(style)), (style - style.mean()) / style.std())
    )
    coefficients = []
    for day in (days[0], days[2]):
        y = returns.loc[returns.timestamp.eq(day), "period_return"].to_numpy()
        coefficients.append(
            np.linalg.lstsq(design * cap[:, None] ** 0.25, y * cap**0.25, rcond=None)[0]
        )
    values = np.asarray(coefficients)
    weights = np.array([1.0, 2.0]) / 3
    centered = values - np.average(values, axis=0, weights=weights)
    np.testing.assert_allclose(
        actual["factor_covariance"],
        (centered * weights[:, None]).T @ centered * 252,
        rtol=1e-11,
        atol=1e-14,
    )
    stocks = read_dataset(config["stock_pool"]["stock_master"])
    stocks["delisting_date"] = pd.NaT
    stocks.loc[stocks.source_asset_id.eq("a"), "delisting_date"] = (
        at.normalize() - pd.Timedelta(days=1)
    )
    new = stocks.loc[stocks.source_asset_id.eq("a")].assign(
        timestamp=at.normalize(), listing_date=at.normalize(), delisting_date=pd.NaT
    )
    config["stock_pool"]["stock_master"] = fact(
        tmp_path / "relisted",
        pd.concat((stocks, new)),
        ("source_asset_id", "timestamp"),
    )
    changed = portfolio_risk_context(
        config,
        window=TimeWindow(at, at + pd.Timedelta(seconds=6)),
        construction="absolute_risk",
    )(SimpleNamespace(timestamp=at + pd.Timedelta(seconds=6)))
    assert pd.isna(changed["assets"].loc["a", "specific_variance"])
    assert changed["assets"].drop(index="a").specific_variance.notna().all()


@pytest.mark.parametrize(
    "endpoint", ["missing_row", "unmatched_life", "missing_initial_row"]
)
def test_origin_normalization_does_not_depend_on_future_endpoint_rows(
    tmp_path, endpoint
):
    config, at, returns, style, cap = sources(tmp_path)
    days = sorted(
        returns.loc[returns.timestamp < at.normalize(), "timestamp"].unique()
    )[-3:]
    affected = returns.timestamp.isin(days) & returns.asset_id.eq("a")
    if endpoint == "missing_initial_row":
        affected &= returns.timestamp.eq(days[0])
    if endpoint != "unmatched_life":
        responses = returns.loc[~affected].copy()
    else:
        responses = returns.copy()
        responses.loc[affected, "listing_date"] = responses.loc[affected, "timestamp"]
    root, _ = assembly_source(tmp_path, responses, data_id="endpoint_changed")
    config["inputs"]["returns"] = str(root)
    actual = portfolio_risk_context(
        config,
        window=TimeWindow(at, at + pd.Timedelta(seconds=6)),
        construction="absolute_risk",
    )(SimpleNamespace(timestamp=at + pd.Timedelta(seconds=6)))
    # The origin has all 20 stocks; only the later response is unavailable.
    # Normalize on 20, then solve WLS on the 19 observed same-life returns.
    design = np.column_stack(
        (np.ones(len(style)), (style - style.mean()) / style.std())
    )
    coefficients, first_stock_residuals = [], []
    for day in days:
        observed = np.ones(len(style), dtype=bool)
        observed[0] = endpoint == "missing_initial_row" and day != days[0]
        y = returns.loc[returns.timestamp.eq(day), "period_return"].to_numpy()[observed]
        weights = cap[observed] ** 0.25
        matrix = design[observed]
        beta = np.linalg.lstsq(matrix * weights[:, None], y * weights, rcond=None)[0]
        coefficients.append(beta)
        if observed[0]:
            first_stock_residuals.append(y[0] - design[0] @ beta)
    values = np.asarray(coefficients)
    decay = 2.0 ** (-np.arange(2, -1, -1) / 2)
    decay /= decay.sum()
    centered = values - np.average(values, axis=0, weights=decay)
    np.testing.assert_allclose(
        actual["factor_covariance"],
        (centered * decay[:, None]).T @ centered * 252,
        rtol=1e-11,
        atol=1e-14,
    )
    if endpoint == "missing_initial_row":
        weights = np.array([2**-0.5, 1.0])
        weights /= weights.sum()
        np.testing.assert_allclose(
            actual["assets"].loc["a", "specific_variance"],
            weights @ np.square(first_stock_residuals) * 252,
            rtol=1e-11,
            atol=1e-14,
        )


def test_future_mutation_and_missing_clocks_do_not_change_independent_context(tmp_path):
    config, at, *_ = sources(tmp_path / "original")
    changed, _, *_ = sources(tmp_path / "changed", future=100.0)
    window = TimeWindow(at, at + pd.Timedelta(seconds=3))
    first = portfolio_risk_context(config, window=window, construction="absolute_risk")(
        SimpleNamespace(timestamp=at)
    )
    other = portfolio_risk_context(
        changed, window=window, construction="absolute_risk"
    )(SimpleNamespace(timestamp=at))
    pd.testing.assert_frame_equal(first["assets"], other["assets"])
    pd.testing.assert_frame_equal(
        first["factor_covariance"], other["factor_covariance"]
    )
    unknown, _, *_ = sources(tmp_path / "unknown", unknown_clock=True)
    result = portfolio_risk_context(
        unknown, window=window, construction="absolute_risk"
    )(SimpleNamespace(timestamp=at))
    assert result["factor_covariance"] is None
    assert result["assets"].specific_variance.isna().all()
    pd.testing.assert_series_equal(
        result["assets"].float_market_cap, first["assets"].float_market_cap
    )
    # A cap-only construction never opens the unrelated return input.
    unknown["inputs"]["returns"] = "/nonexistent/unneeded/returns"
    cap = portfolio_risk_context(
        unknown, window=window, construction="quantile_float_cap"
    )(SimpleNamespace(timestamp=at))
    pd.testing.assert_series_equal(
        cap["assets"].float_market_cap, first["assets"].float_market_cap
    )


def test_risk_history_qualification_omits_only_continuous_session_gate(tmp_path):
    config, at, *_ = sources(tmp_path)
    pool = load_stock_pool(config["stock_pool"])
    close = at.normalize() + pd.Timedelta(hours=7)
    default = stock_pool_rows(pd.DatetimeIndex([close]), **pool)
    history = stock_pool_rows(
        pd.DatetimeIndex([close]), require_continuous=False, **pool
    )
    assert not default.eligible.any()
    assert history.eligible.all()
    weekend = close + pd.Timedelta(days=1)
    assert not stock_pool_rows(
        pd.DatetimeIndex([weekend]), require_continuous=False, **pool
    ).eligible.any()


def test_native_policy_builds_context_lazily_and_persists_risk_definition(tmp_path):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.policies.reinforcement import backtest_policy
    from tests.research.test_stock_portfolio_policy import write_replay

    config = write_replay(tmp_path / "equal")
    config["portfolio_context"] = {
        "unused": "must not open or validate on equal-weight path"
    }
    backtest_policy(config)
    config = write_replay(tmp_path / "risk")
    provider, _, *_ = sources(tmp_path / "context", decision_day="2024-01-02")
    config["portfolio_parameters"] = {
        "construction": "absolute_risk",
        "risk_limits": {"total_volatility_limit": 0.5},
    }
    config["portfolio_context"] = provider
    root = backtest_policy(config)
    ledger = read_portfolio_ledger_artifact(root / "ledger")
    assert (
        ledger.metadata["stock_portfolio_policy"]["context_source"]
        == "native_point_in_time_daily_history"
    )
    assert (
        ledger.metadata["policy"]["portfolio_context"]["risk_model"]
        == provider["risk_model"]
    )
    assert list(ledger.iter_chunks())

    # Undefined risk targets must not stop the unrelated account/holdings path.
    config = write_replay(tmp_path / "unknown_policy")
    provider, _, *_ = sources(
        tmp_path / "unknown_context", decision_day="2024-01-02", unknown_clock=True
    )
    config["portfolio_parameters"] = {
        "construction": "absolute_risk",
        "risk_limits": {"total_volatility_limit": 0.5},
    }
    config["portfolio_context"] = provider
    config["episode"]["account"]["parameters"]["initial_inventory"]["a"] = 100
    ledger = read_portfolio_ledger_artifact(backtest_policy(config) / "ledger")
    batches = list(ledger.iter_chunks())
    assert not [trade for batch in batches for trade in batch.trades]
    assert any(
        value.asset_id == "a" and value.quantity == 100
        for batch in batches
        for value in batch.valuations
    )

"""Research scope and nominal/fresh risk windows are independent contracts."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.filtering import (
    a_share_eligibility,
    a_share_eligibility_columns,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
)


def test_research_scope_does_not_require_quotes_or_trading():
    frame = pd.DataFrame(
        {
            "is_a_share_scope": pd.array([True, True, True, None], dtype="boolean"),
            "is_st": [False, False, True, False],
            "is_pt": False,
            "has_252_trading_sessions": True,
            "is_trading": [False, True, True, True],
            "close": [np.nan, -1, np.nan, np.nan],
        }
    )
    assert a_share_eligibility(frame, "a_share_all").tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert pd.isna(frame.is_a_share_scope.iloc[-1])
    assert a_share_eligibility(frame, "a_share_all", exclude_st=False).tolist() == [
        True,
        True,
        True,
        False,
    ]


def test_custom_scope_only_requires_its_selected_facts():
    frame = pd.DataFrame(
        {"is_a_share_scope": True, "listing_age_trading_days": [19, 20, None]}
    )
    kwargs = dict(exclude_st=False, exclude_pt=False, minimum_listing_sessions=20)
    assert a_share_eligibility_columns(**kwargs) == (
        "is_a_share_scope",
        "listing_age_trading_days",
    )
    assert a_share_eligibility(frame, "a_share_all", **kwargs).tolist() == [
        False,
        True,
        False,
    ]
    assert a_share_eligibility(
        frame[["is_a_share_scope"]],
        "a_share_all",
        exclude_st=False,
        exclude_pt=False,
        minimum_listing_sessions=0,
    ).all()


def _cross(day, *, paused=(), missing=()):
    assets = [f"a{i}" for i in range(8)]
    returns = np.arange(8, dtype=float) / 100
    returns[list(paused)] = 0
    returns[list(missing)] = np.nan
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=day),
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": assets,
            "industry_standard_id": "S",
            "industry_level1_code": "I",
            "total_market_cap": 100.0,
            "asset_return": returns,
            "eligible": True,
            "is_trading": [i not in paused for i in range(8)],
        }
    )


def _state():
    return FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            factor_window=4,
            specific_window=4,
            minimum_cross_section=5,
            annualization=1,
        ),
    )


def test_risk_defaults_are_ceiled_three_quarters_of_nominal_windows():
    default = FactorRiskConfig()
    assert (default.minimum_factor_dates, default.minimum_specific_dates) == (378, 189)
    odd = FactorRiskConfig(factor_window=7, specific_window=5)
    assert (odd.minimum_factor_dates, odd.minimum_specific_dates) == (6, 4)


def test_risk_waits_for_full_span_even_with_enough_fresh_observations():
    state = _state()
    for day in range(3):
        state.observe_return_session(_cross(day))
    with pytest.raises(ValueError, match="factor_history_insufficient"):
        state.estimate()
    state.observe_return_session(_cross(3))
    assert len(state.estimate().specific_variance) == 8


def test_pause_zero_is_numeric_but_not_fresh_unknown_is_not_zero():
    state = _state()
    for day in range(4):
        state.observe_return_session(
            _cross(
                day, paused=(0,) if day >= 2 else (), missing=(1,) if day >= 2 else ()
            )
        )
    result = state.estimate()
    assert "a0" not in result.specific_variance
    assert "a1" not in result.specific_variance
    assert len(state._residuals["a0"]) == 4
    assert len(state._residuals["a1"]) == 2
    state.observe_return_session(_cross(4))
    assert "a0" not in state.estimate().specific_variance
    state.observe_return_session(_cross(5))
    assert "a0" not in state.estimate().specific_variance
    state.observe_return_session(_cross(6))
    assert "a0" in state.estimate().specific_variance
    assert "a1" in state.estimate().specific_variance
    unknown = _cross(7)
    unknown.loc[unknown.asset_id.eq("a0"), ["listing_date", "eligible"]] = [
        pd.NaT,
        False,
    ]
    state.observe_return_session(unknown)
    assert "a0" not in state._listings and "a0" not in state._residuals
    assert "a1" in state.estimate().specific_variance
    state.observe_return_session(_cross(8))
    assert len(state._residuals["a0"]) == 1
    assert "a0" not in state.estimate().specific_variance


def test_whole_unknown_session_occupies_window_without_a_zero_fit():
    state = _state()
    for day in range(3):
        state.observe_return_session(_cross(day))
    state.observe_return_session(_cross(3, missing=tuple(range(8))))
    result = state.estimate()
    assert len(result.factor_returns) == 3
    assert len(result.specific_variance) == 8
    state.observe_return_session(_cross(4, missing=tuple(range(8))))
    with pytest.raises(ValueError, match="factor_history_insufficient"):
        state.estimate()


def test_assembly_persists_custom_research_scope_without_market_activity(tmp_path):
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
    from tests.research.test_direct_assembly_config import _config, _frame, _source

    frame = _frame(3).assign(
        is_a_share_scope=True, listing_age_trading_days=[19, 20, 21]
    )
    root, _ = _source(tmp_path, frame)
    output = assemble(
        _config(
            tmp_path,
            root,
            eligibility_recipe="a_share_all",
            minimum_listing_sessions=20,
            exclude_st=False,
            exclude_pt=False,
            filter_columns=["eligible"],
        )
    )
    assert read_dataset(output).eligible.tolist() == [False, True, True]
    facts = inspect_dataset(output).spec
    assert facts["minimum_listing_sessions"] == 20
    assert facts["exclude_st"] is False
    assert facts["exclude_pt"] is False

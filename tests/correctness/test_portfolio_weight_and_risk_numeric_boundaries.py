from collections import deque
from fractions import Fraction
from math import isqrt

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_DECISION_REASON_SIMPLE,
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    prepare_trade_cost,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    producer as market_sessions,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.diagnostics import (
    iter_holding_factor_inputs,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
    holding_attribution,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.portfolio_ledger import (
    ledger_frame,
    write_market_evidence,
    write_test_sessions,
)
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.test_portfolio_ledger import _real_fee_schedule
from tests.workflows.test_portfolio_workflow_integration import _processed_specs


@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_cross_scale_capital_weights_round_only_final_leg_ratio(mode):
    assets = (
        ["large", "small"]
        if mode == "long_only"
        else ["long_large", "long_small", "short_large", "short_small"]
    )
    capitals = pd.Series(
        [1e308, 3e-16] if mode == "long_only" else [1e308, 6e-16, 1e308, 6e-16],
        index=assets,
    )
    signal = pd.Series(
        [1.0, 1.0] if mode == "long_only" else [1.0, 1.0, -1.0, -1.0], index=assets
    )
    expected = {}
    for selected, gross in (
        ((assets, Fraction(1)),)
        if mode == "long_only"
        else ((assets[:2], Fraction(1, 2)), (assets[2:], Fraction(-1, 2)))
    ):
        total = sum(Fraction(float(capitals[asset])) for asset in selected)
        expected.update(
            {
                asset: float(gross * Fraction(float(capitals[asset])) / total)
                for asset in selected
            }
        )
    for order in (assets, assets[::-1]):
        actual = build_simple_target_weights(
            signal.reindex(order),
            eligible=pd.Series(True, index=order),
            method="float_cap_weight",
            portfolio_mode=mode,
            top_quantile=1.0 if mode == "long_only" else 0.5,
            bottom_quantile=0.5,
            float_market_cap=capitals.reindex(order),
        )
        assert actual.dtype == "float64"
        assert actual.index.tolist() == sorted(assets)
        assert actual.to_dict() == expected
        assert all(
            abs(value) > 0 for asset, value in actual.items() if "small" in asset
        )


@pytest.mark.parametrize("capitals", [[1.0, 2.0, 3.0], [1e308, 1e308, 3e-16]])
def test_capital_sum_overflow_and_unrepresentable_final_weight_match_fraction(capitals):
    assets = [str(i) for i in range(len(capitals))]
    actual = build_simple_target_weights(
        pd.Series(1.0, index=assets),
        eligible=pd.Series(True, index=assets),
        method="float_cap_weight",
        top_quantile=1.0,
        float_market_cap=pd.Series(capitals, index=assets),
    )
    exact = [Fraction(value) for value in capitals]
    assert actual.tolist() == [float(value / sum(exact)) for value in exact]


def _cross(count=100):
    return pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": [f"synthetic_{i:03d}" for i in range(count)],
            "industry_standard_id": "fixture",
            "industry_level1_code": ["large"] * (count // 2) + ["small"] * (count // 2),
            "float_market_cap": 1.0,
            "asset_return": [1e308] * (count // 2) + [1.0] * (count // 2),
        }
    )


@pytest.mark.parametrize(
    "order", [list(range(10)), list(range(9, -1, -1)), [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]
)
def test_two_industry_wls_keeps_small_constant_response_and_zero_residuals(order):
    fitted = FactorRiskState(
        style_columns=(), config=FactorRiskConfig(minimum_cross_section=10)
    ).fit_return_session(_cross(10).iloc[order])
    assert fitted is not None
    assert fitted.factor_returns.to_dict() == {
        "industry_fixture:large": 1e308,
        "industry_fixture:small": 1.0,
    }
    assert fitted.residuals.eq(0.0).all()


def _full_wls_fraction_oracle(exposures, cross):
    aligned = cross.set_index("asset_id").reindex(exposures.index)
    weights = [Fraction(isqrt(int(cap))) for cap in aligned.float_market_cap]
    assert all(
        weight**2 == Fraction(float(cap))
        for weight, cap in zip(weights, aligned.float_market_cap, strict=True)
    )
    x = [[Fraction(float(value)) for value in row] for row in exposures.to_numpy()]
    y = [Fraction(float(value)) for value in aligned.asset_return]
    dimension = exposures.shape[1]
    system = [
        [
            sum(w * row[left] * row[right] for w, row in zip(weights, x, strict=True))
            for right in range(dimension)
        ]
        + [
            sum(
                w * row[left] * value
                for w, row, value in zip(weights, x, y, strict=True)
            )
        ]
        for left in range(dimension)
    ]
    for column in range(dimension):
        pivot = system[column][column]
        system[column] = [value / pivot for value in system[column]]
        for row in range(dimension):
            if row != column:
                ratio = system[row][column]
                system[row] = [
                    value - ratio * reference
                    for value, reference in zip(
                        system[row], system[column], strict=True
                    )
                ]
    coefficients = [row[-1] for row in system]
    residuals = [
        float(
            value
            - sum(
                term * coefficient
                for term, coefficient in zip(row, coefficients, strict=True)
            )
        )
        for row, value in zip(x, y, strict=True)
    ]
    return list(map(float, coefficients)), residuals


@pytest.mark.parametrize("magnitude", [1e308, 0.01])
@pytest.mark.parametrize("style_count", [1, 3])
def test_weighted_style_and_residuals_match_independent_full_normal_system(
    magnitude, style_count
):
    frame = _cross()
    frame["float_market_cap"] = np.tile([1.0, 4.0, 16.0, 64.0, 256.0], 20)
    frame["style"] = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 20)
    styles = ("style",) + tuple(f"style_{i}" for i in range(1, style_count))
    rng = np.random.default_rng(61003)
    for column in styles[1:]:
        frame[column] = np.tile(rng.normal(size=50), 2)
    frame["asset_return"] = [magnitude] * 50 + list(
        np.tile(
            [1.0, 2.0, 3.0, 4.0, 5.0]
            if magnitude > 1
            else [-0.02, -0.01, 0.0, 0.01, 0.02],
            10,
        )
    )
    original = frame.copy(deep=True)
    results = []
    for order in (
        np.arange(len(frame)),
        np.arange(len(frame))[::-1],
        np.r_[np.arange(0, len(frame), 2), np.arange(1, len(frame), 2)],
    ):
        current = frame.iloc[order]
        fitted = FactorRiskState(style_columns=styles).fit_return_session(current)
        assert fitted is not None
        coefficients, residuals = _full_wls_fraction_oracle(fitted.exposures, current)
        np.testing.assert_array_equal(fitted.factor_returns, coefficients)
        np.testing.assert_array_equal(fitted.residuals, residuals)
        assert fitted.factor_returns["style"] != 0
        results.append((fitted.factor_returns, fitted.residuals.sort_index()))
    for coefficients, residuals in results[1:]:
        pd.testing.assert_series_equal(coefficients, results[0][0], check_exact=True)
        pd.testing.assert_series_equal(residuals, results[0][1], check_exact=True)
    pd.testing.assert_frame_equal(frame, original, check_exact=True)


def test_risk_fit_rejects_insufficient_samples_and_rank_deficient_designs():
    assert (
        FactorRiskState(
            style_columns=(), config=FactorRiskConfig(minimum_cross_section=10)
        ).fit_return_session(_cross(10).iloc[:9])
        is None
    )
    frame = _cross().assign(style=np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 20))
    frame["duplicate"] = frame["style"]
    assert (
        FactorRiskState(style_columns=("style", "duplicate")).fit_return_session(frame)
        is None
    )


def _panel(assets, dates):
    return pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "market_id": "a_share",
            "asset_id": assets * len(dates),
            "timestamp": dates.repeat(len(assets)),
            "adj_open": 1.0,
            "adj_close": 1.0,
            "amount": 1000.0,
            "market_type": 1,
            "period_return": 0.0,
            "float_market_cap": 1.0,
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": (
                ["large"] * (len(assets) // 2)
                + ["small"] * (len(assets) - len(assets) // 2)
            )
            * len(dates),
            "is_nonfinancial_industry": True,
            "is_a_share_scope": True,
            "is_trading": True,
            "is_st": False,
            "is_pt": False,
            "listing_age_trading_days": 300,
            "has_252_trading_sessions": True,
            "close": 1.0,
            "volume": 1000.0,
            "turnover": 0.01,
        }
    )


def test_native_panel_feeds_risk_state_and_holding_attribution(tmp_path):
    dates = pd.bdate_range("2023-01-02", periods=3)
    assets = [f"synthetic_{i:03d}" for i in range(100)]
    panel = _panel(assets, dates)
    large = panel.asset_id.isin(assets[:50])
    for day, big, small in zip(
        dates, [1e-308, 1.0, 1e308], [1.0, 1.0, 2.0], strict=True
    ):
        current = panel.timestamp.eq(day)
        panel.loc[current & large, ["adj_open", "adj_close", "close"]] = big
        panel.loc[current & ~large, ["adj_open", "adj_close", "close"]] = small
    panel.loc[large & panel.timestamp.ne(dates[0]), "period_return"] = 1e308
    panel.loc[~large & panel.timestamp.eq(dates[2]), "period_return"] = 1.0
    spec = _processed_specs().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root("a_share", spec.kind, spec.data_id)
    write_dataset(root, panel, spec=spec)
    saved = read_dataset(root)
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            minimum_factor_dates=2,
            minimum_specific_dates=2,
            factor_half_life=1.0,
            diagonal_shrinkage=0.0,
            annualization=1.0,
        ),
    )
    previous = None
    for day in dates:
        current = saved.loc[saved.timestamp.eq(day)]
        if previous is not None:
            state.observe_return_session(
                production._factor_return_cross(exposures=previous, panel=current)
            )
        previous = production._enhanced_risk_cross(
            panel=current, risk_frames=[], style_columns=()
        )
    estimate = state.estimate()
    assert estimate.factor_returns["industry_CSRC2012:large"].tolist() == [1e308, 1e308]
    assert estimate.factor_returns["industry_CSRC2012:small"].tolist() == [0.0, 1.0]
    assert estimate.factor_covariance.loc[
        "industry_CSRC2012:small", "industry_CSRC2012:small"
    ] == float(Fraction(2, 9))
    assert estimate.factor_covariance.loc["industry_CSRC2012:large"].eq(0.0).all()
    assert estimate.specific_variance.eq(0.0).all()
    calendar = pd.DataFrame(
        {"calendar_date": dates, "source_market_type": 1, "is_trading_day": True}
    )
    stream = iter_holding_factor_inputs(
        iter_portfolio_sessions=market_sessions.iter_portfolio_sessions,
        layout=layout.with_target_roots(
            {"market_panel_1d": root, **write_market_evidence(layout, panel)}
        ),
        calendar=calendar,
        time_window=TimeWindow(dates[0], dates[-1]),
        unified_trading_sessions=unified_trading_sessions,
        style_columns=(),
        risk_context_target_ids=(),
        benchmark_required=False,
    )
    with closing_iterator(stream):
        inputs = next(stream)
        assert inputs.reason is None and inputs.fitted is not None
        assert inputs.fitted.factor_returns["industry_CSRC2012:small"] == 1.0
        held = pd.Series({assets[-1]: 1.0})
        values, reasons, _exposures = holding_attribution(
            inputs,
            after_notionals=held,
            nav_before=1.0,
            nav_post=1.0,
            benchmark_id=None,
            benchmark_return=None,
            holding_returns=inputs.fitted.asset_returns,
        )
        assert not reasons
        assert values["industry"] == 1.0
        assert values["stock_residual_difference"] == 0.0


def test_native_simple_production_and_ledger_keep_representable_positive_target(
    tmp_path,
):
    assets = ["large", "small"]
    dates = pd.bdate_range("2023-01-02", periods=3)
    layout = StorageLayout(tmp_path)
    specs = _processed_specs()
    panel_root = layout.processed_data_root(
        "a_share", specs.get("market_panel_1d").kind, "market_panel_1d"
    )
    write_dataset(panel_root, _panel(assets, dates), spec=specs.get("market_panel_1d"))
    panel = read_dataset(panel_root)
    dataset = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": assets,
            "timestamp": dates[0],
            "eligible": True,
            "signal": 1.0,
            "label": 0.0,
            "float_market_cap": [1e308, 3e-16],
        }
    )
    dataset_spec = make_demo_research_dataset_spec(
        data_id="extreme_capitalization",
        factor_columns=("signal",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )
    dataset_root = layout.research_dataset_root("a_share", dataset_spec.data_id)
    write_dataset(dataset_root, dataset, spec=dataset_spec)
    dataset = read_dataset(dataset_root)
    execution = dataset[["market_id", "asset_id", "timestamp"]].assign(
        can_enter=True, can_exit=True
    )
    execution_root = layout.processed_data_root(
        "a_share", specs.get("execution_mask").kind, "execution_mask"
    )
    write_dataset(execution_root, execution, spec=specs.get("execution_mask"))
    schedule = _real_fee_schedule(tmp_path / "fees")
    candidate = PortfolioCandidateSpec(
        candidate_id="extreme_capitalization",
        selection_group="global",
        construction="top20_float_cap",
        prediction_target_id="prediction",
        prediction_selector="champion",
        rebalance_schedule="daily",
        top_quantile=1.0,
        benchmark_id=None,
        benchmark_weights_target_id=None,
        research_dataset_id="dataset",
        market_panel_target_id="panel",
        execution_mask_target_id="execution",
        trading_calendar_target_id="calendar",
        fee_schedule_target_id="fee",
    )
    alpha = dataset.set_index("asset_id").signal
    account, returns, adv, optimized, delta, _snapshot = production._build_session(
        candidate=candidate,
        decision_timestamp=dates[0],
        next_timestamp=dates[1],
        panel=panel.loc[panel.timestamp.eq(dates[0])],
        next_panel=panel.loc[panel.timestamp.eq(dates[1])],
        endpoint_panel=panel.loc[panel.timestamp.eq(dates[2])],
        closed_assets=frozenset(),
        dataset=dataset,
        execution=read_dataset(execution_root).assign(execution_at=dates[1]),
        holdings=pd.Series(dtype="float64"),
        cash=1.0,
        nav=1.0,
        initialize_at_execution=True,
        amount_history={
            (asset, panel.listing_date.iloc[0]): deque([1000.0] * 20, maxlen=20)
            for asset in assets
        },
        alpha=alpha,
        enhanced_decision=None,
        target_override=None,
        decision_reason=PORTFOLIO_DECISION_REASON_SIMPLE,
        assumptions=ExecutionAssumptions(1.0, 0.0, 0.0, 1.0, (1.0,)),
        fee_schedule=schedule,
        fee_identity=schedule.content_identity,
        prepare_trade_cost=prepare_trade_cost,
        market_name_for_type=lambda _market: "shanghai",
    )
    assert optimized is None
    expected = float(Fraction(3e-16) / (Fraction(1e308) + Fraction(3e-16)))
    assert expected > 0 and delta.loc["small"] == expected
    frame = ledger_frame(
        account=account,
        decision_timestamp=dates[0],
        returns=returns,
        alpha_score=alpha,
        adv20=adv,
        enhanced_decision=None,
        benchmark_weights=None,
        optimization_result=None,
        decision_reason=PORTFOLIO_DECISION_REASON_SIMPLE,
        decision_delta=delta,
    )
    envelope = build_business_manifest_envelope(
        logical_artifact_id="extreme_capitalization",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2023-01-02",
            "end": "2023-01-04",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={
            "portfolio_capital": {
                "candidate_id": "extreme_capitalization",
                "initial_capital": 1.0,
                "valuation_currency": "CNY",
                "valuation_method": "adjusted_notional",
            }
        },
    )
    ledger_root = write_test_sessions(
        tmp_path / "ledger", envelope=envelope, ledger_chunks=iter((frame,))
    )
    artifact = read_portfolio_ledger_artifact(
        ledger_root, expected_identity=ledger_root.name
    )
    (batch,) = tuple(artifact.iter_chunks())
    diagnostics = {
        row.name: row.number for row in batch.diagnostics if row.asset_id == "small"
    }
    assert diagnostics["target_notional"] == expected
    assert diagnostics["decision_delta_weight"] == expected
    assert any(
        row.asset_id == "small" and row.phase == "after" and row.value > 0
        for row in batch.valuations
    )
    assert any(row.asset_id == "small" and row.notional > 0 for row in batch.trades)


def test_native_market_model_relisting_resets_beta_and_residual_warmup(tmp_path):
    from quant_research_platform.research.factors.engine import _market_model_frames

    dates = pd.bdate_range("2020-01-02", periods=911)
    assets = [f"a{i:02d}" for i in range(12)]
    panel = _panel(assets, dates)
    panel["listing_date"] = dates[0]
    day = np.repeat(np.arange(len(dates)), len(assets))
    stock = np.tile(np.arange(len(assets)), len(dates))
    panel["period_return"] = 0.01 * np.sin(day / 7) + 0.004 * np.cos(day / 3 + stock)
    new = panel.asset_id.eq(assets[0]) & panel.timestamp.ge(dates[600])
    panel.loc[new, "listing_date"] = dates[600]
    panel.loc[new, "listing_age_trading_days"] = day[new] - 599
    panel.loc[new, "has_252_trading_sessions"] = day[new] >= 851
    spec = _processed_specs().get("market_panel_1d")
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, panel, spec=spec)
    saved = read_dataset(root)
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
    )

    factor_spec = catalog.processed_catalog().get("market_model_factors")
    factor_root = StorageLayout(tmp_path).processed_data_root(
        factor_spec.market_id, factor_spec.kind, factor_spec.data_id
    )
    write_dataset(
        factor_root,
        _market_model_frames([saved], trading_calendar=None, market_history=None),
        spec=factor_spec,
    )
    actual = read_dataset(factor_root)
    # Native session boundaries must not change rolling lifecycle history.
    split = pd.concat(
        _market_model_frames(
            [
                saved.iloc[: 600 * len(assets)],
                saved.iloc[600 * len(assets) : 851 * len(assets)],
                saved.iloc[851 * len(assets) :],
            ],
            trading_calendar=None,
            market_history=None,
        ),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(actual, split, check_dtype=False, check_exact=True)
    control = saved.copy()
    new = control.asset_id.eq(assets[0]) & control.timestamp.ge(dates[600])
    control.loc[new, "asset_id"] = "new_life"
    expected = pd.concat(
        _market_model_frames([control], trading_calendar=None, market_history=None)
    )
    actual = actual.loc[actual.asset_id.eq(assets[0]) & actual.timestamp.ge(dates[600])]
    expected = expected.loc[expected.asset_id.eq("new_life")]
    columns = ["beta_252d", "residual_volatility_60d"]
    np.testing.assert_array_equal(actual[columns], expected[columns])
    assert actual.beta_252d.iloc[:251].isna().all()
    assert np.isfinite(actual.beta_252d.iloc[251])
    assert actual.residual_volatility_60d.iloc[:310].isna().all()
    assert np.isfinite(actual.residual_volatility_60d.iloc[310])


def test_lagged_risk_preserves_prior_eligibility_and_resets_new_life():
    dates = pd.bdate_range("2023-01-02", periods=4)
    assets = [f"a{i:02d}" for i in range(12)]
    panel = _panel(assets, dates[:1])
    panel["industry_level1_code"] = "one"
    previous = production._enhanced_risk_cross(
        panel=panel, risk_frames=[], style_columns=()
    )
    state = FactorRiskState(
        style_columns=(),
        config=FactorRiskConfig(
            minimum_cross_section=10,
            minimum_factor_dates=2,
            minimum_specific_dates=2,
        ),
    )
    for day in dates[:3]:
        state.observe_return_session(previous.assign(timestamp=day))
    assert assets[0] in state.estimate().specific_variance
    current = panel.assign(timestamp=dates[-1])
    current.loc[current.asset_id.eq(assets[0]), "listing_date"] = dates[-1]
    current.loc[current.asset_id.eq(assets[0]), "period_return"] = 0.5
    current.loc[current.asset_id.eq(assets[1]), "period_return"] = 0.11
    current.loc[current.asset_id.isin(assets[:2]), "has_252_trading_sessions"] = False
    cross = production._factor_return_cross(exposures=previous, panel=current)
    assert cross.eligible.all()
    assert pd.isna(cross.set_index("asset_id").loc[assets[0], "asset_return"])
    fitted = state.fit_return_session(cross)
    assert fitted.factor_returns.iloc[0] == pytest.approx(0.01)
    state.observe_return_session(cross)
    assert assets[0] not in state.estimate().specific_variance
    assert assets[1] in state.estimate().specific_variance

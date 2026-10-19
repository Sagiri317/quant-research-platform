"""Native enhanced decisions preserve receivables and local unknown domains."""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.accounting import (
    alpha_calibration_intervals,
)
from quant_research_platform.research.portfolios.optimizer import (
    _stock_leg_active_exposure,
)
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.optimizer import with_cash_asset
from tests.helpers.portfolio_inputs import _interim_specs, _processed_specs
from tests.helpers.portfolio_inputs import closed_market_inputs as _native_inputs
from tests.helpers.portfolio_inputs import decision_inputs as _decision_inputs
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.direct_model_test_support import fit_frame


def _data_spec():
    return make_demo_research_dataset_spec(
        data_id="fixture",
        factor_columns=("signal",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )


@pytest.fixture(scope="module")
def enhanced_config(tmp_path_factory):
    root = tmp_path_factory.mktemp("enhanced-native")
    inputs, _, calendar, panel = _native_inputs(root / "market", fault="normal_open")
    panel[["open", "close", "adj_open", "adj_close"]] = 100.0
    panel.loc[panel.timestamp.ge("2023-01-04"), ["open", "close"]] = 99.0
    panel["period_return"] = 0.0
    write_dataset(root / "panel", panel, spec=_processed_specs().get("market_panel_1d"))
    inputs["panel"] = root / "panel"
    data = read_dataset(inputs["dataset"])
    data["signal"] = data.asset_id.str[1:].astype(float)
    write_dataset(root / "data", data, spec=_data_spec())
    inputs["dataset"] = root / "data"
    fees = read_dataset(inputs["fees"])
    fees["rate"] = 0.0
    write_dataset(
        root / "fees", fees, spec=_interim_specs().get("a_share_fee_schedule")
    )
    inputs["fees"] = root / "fees"
    original_ca = read_dataset(inputs["corporate_actions"])
    ca = pd.DataFrame(
        [
            dict(
                source_asset_id=asset,
                timestamp=pd.Timestamp("2023-01-04"),
                kind="CA",
                market_type=1 if asset.startswith("a") else 4,
                announcement_date=pd.Timestamp("2022-12-20"),
                settlement_date=pd.Timestamp("2023-01-06"),
                settlement_date_unresolved=False,
                amount=1.0,
            )
            for asset in sorted(panel.asset_id.unique())
        ]
    ).reindex(columns=original_ca.columns)
    write_dataset(root / "ca", ca, spec=_interim_specs().get("corporate_actions"))
    inputs["corporate_actions"] = root / "ca"
    training = data.loc[data.timestamp.le("2022-12-20")].copy()
    training["label"] = training.signal * 0.001
    training["label_available_at"] = training.timestamp
    model = fit_frame(
        root / "learning",
        training,
        ModelSpec(
            model_id="ridge",
            model_algorithm="ridge_regression",
            input_factor_columns=("signal",),
            target_column="label",
            random_seed=42,
        ),
        market_id="a_share",
    )
    window = {"start": "2023-01-02", "end": "2023-01-06"}
    prediction = predict(
        dict(
            model=str(model.root),
            inputs=str(inputs["dataset"]),
            output=str(root / "predictions"),
            prediction_window=window,
            sample_role="validation",
        )
    )
    benchmark = panel[["market_id", "asset_id", "timestamp"]].assign(
        benchmark_id="csi500", weight=1 / panel.asset_id.nunique()
    )
    write_dataset(
        root / "benchmark", benchmark, spec=_processed_specs().get("benchmark_weights")
    )
    intervals = alpha_calibration_intervals(unified_trading_sessions(calendar), "daily")
    intervals.to_parquet(root / "intervals.parquet", index=False)
    pd.DataFrame(
        [
            dict(
                timestamp=day,
                prediction_rank_score=x,
                holding_return=0.001 + 0.01 * x + 1e-4 * np.sin(i + x),
            )
            for i, day in enumerate(intervals.timestamp)
            for x in [-0.4, -0.2, 0.0, 0.2, 0.4]
        ]
    ).to_parquet(root / "observations.parquet", index=False)
    calibration = production.calibrate(
        dict(
            observations=str(root / "observations.parquet"),
            scheduled_intervals=str(root / "intervals.parquet"),
            rebalance_schedule="daily",
            calendar=str(inputs["calendar"]),
            cutoff="2022-12-29",
            selection_group="global",
            output=str(root / "calibration.json"),
        )
    )
    config = dict(
        candidate=dict(
            candidate_id="enhanced",
            selection_group="global",
            construction="benchmark_enhanced",
            prediction_selector="ridge",
            rebalance_schedule="daily",
            top_quantile=None,
            benchmark_id="csi500",
            risk_style_columns=["signal"],
        ),
        inputs={
            **{
                key: str(value)
                for key, value in inputs.items()
                if key != "risk_context"
            },
            "benchmark": str(root / "benchmark"),
            "risk_context": [str(inputs["dataset"])],
        },
        predictions=[dict(path=str(prediction), window=window)],
        output=str(root / "unused-ledger"),
        output_window=window,
        input_window={"start": "2022-12-01", "end": "2023-01-06"},
        formal_dataset_window={"start": "2022-12-01", "end": "2023-01-06"},
        prediction_window=window,
        assumptions=dict(
            initial_nav=100.0,
            slippage_bps=0.0,
            impact_bps=0.0,
            adv_participation=1.0,
            adv_sensitivities=[1.0],
        ),
        risk_config=dict(
            factor_window=3,
            specific_window=3,
            minimum_factor_dates=2,
            minimum_specific_dates=2,
            minimum_cross_section=10,
        ),
        calibration=str(calibration),
    )
    return config, model.root


def _facts(batch):
    return {
        row.name: row.number if row.number is not None else row.text
        for row in batch.diagnostics
        if row.asset_id == "__cash__"
    }


@pytest.mark.parametrize("payment_date", ["2023-01-04", "2023-01-06"])
@pytest.mark.parametrize("override", [{}, {"style_active_limit": 0.25}])
def test_native_receivable_stays_in_nav_and_never_funds_buys(
    tmp_path, enhanced_config, payment_date, override
):
    config = deepcopy(enhanced_config[0])
    ca = read_dataset(config["inputs"]["corporate_actions"])
    ca["settlement_date"] = pd.Timestamp(payment_date)
    write_dataset(tmp_path / "ca", ca, spec=_interim_specs().get("corporate_actions"))
    config["inputs"]["corporate_actions"] = str(tmp_path / "ca")
    config["output"] = str(tmp_path / "ledger")
    if override:
        config["optimizer_config"] = override
    artifact = read_portfolio_ledger_artifact(production.backtest(config))
    batches = list(artifact.iter_chunks())
    assert len(batches) == 3
    assert artifact.metadata["optimizer_config"] == asdict(
        production.OptimizerConfig(**override)
    )
    assert (
        artifact.metadata["initial_turnover_policy"]
        == "first_actual_buy_exempt_fees_and_capacity_apply"
    )
    for batch in batches:
        for phase in ("before", "after", "end"):
            assert sum(
                row.value for row in batch.valuations if row.phase == phase
            ) == pytest.approx(100.0)
        cash_before = sum(
            row.value
            for row in batch.valuations
            if row.phase == "before" and row.asset_id == "__cash__"
        )
        entry_at = next(
            row.timestamp_ns for row in batch.valuations if row.phase == "after"
        )
        paid = sum(
            flow.amount
            for flow in batch.cashflows
            if flow.kind == "dividend" and flow.timestamp_ns == entry_at
        )
        buys = sum(trade.notional for trade in batch.trades if trade.quantity > 0)
        sells = -sum(trade.notional for trade in batch.trades if trade.quantity < 0)
        assert buys <= cash_before + paid + sells + 1e-8
        cash_after = sum(
            row.value
            for row in batch.valuations
            if row.phase == "after" and row.asset_id == "__cash__"
        )
        assert cash_after == pytest.approx(cash_before + paid + sells - buys, abs=1e-8)
        assert all(
            row.value >= -1e-8 for row in batch.valuations if row.asset_id == "__cash__"
        )
    after = [row for row in batches[1].valuations if row.phase == "after"]
    pending = sum(row.receivable for row in after)
    if payment_date == "2023-01-06":
        assert pending == pytest.approx(0.94)
        assert sum(row.value - row.receivable for row in after) == pytest.approx(99.06)
        assert _facts(batches[-1])["decision_status"] == "AVAILABLE"
        assert _facts(batches[-1])["realized_constraints_status"] == "SATISFIED"
    else:
        assert pending == 0.0
        assert sum(
            flow.amount
            for batch in batches
            for flow in batch.cashflows
            if flow.kind == "dividend"
        ) == pytest.approx(0.94)


@pytest.mark.parametrize("missing_context", [False, True])
@pytest.mark.parametrize("independent_exit", ["none", "eligibility", "membership"])
def test_native_missing_predictions_hold_or_exit_without_new_target_context(
    tmp_path, enhanced_config, missing_context, independent_exit
):
    config, model = deepcopy(enhanced_config)
    data = read_dataset(config["inputs"]["dataset"])
    data.loc[data.timestamp.eq("2023-01-04"), "signal"] = np.nan
    if independent_exit == "eligibility":
        data.loc[
            data.timestamp.ge("2023-01-04") & data.asset_id.eq("a0"), "eligible"
        ] = False
    write_dataset(tmp_path / "data", data, spec=_data_spec())
    config["inputs"]["dataset"] = str(tmp_path / "data")
    config["inputs"]["risk_context"] = [str(tmp_path / "data")]
    prediction = predict(
        dict(
            model=str(model),
            inputs=str(tmp_path / "data"),
            output=str(tmp_path / "predictions"),
            prediction_window=config["prediction_window"],
            sample_role="validation",
        )
    )
    config["predictions"][0]["path"] = str(prediction)
    if missing_context:
        panel = read_dataset(config["inputs"]["panel"])
        panel.loc[
            panel.timestamp.ge("2023-01-04"),
            ["industry_standard_id", "industry_level1_code", "total_market_cap"],
        ] = pd.NA
        write_dataset(
            tmp_path / "panel", panel, spec=_processed_specs().get("market_panel_1d")
        )
        config["inputs"]["panel"] = str(tmp_path / "panel")
    if independent_exit == "membership":
        benchmark = read_dataset(config["inputs"]["benchmark"])
        changed = benchmark.timestamp.ge("2023-01-04")
        benchmark = benchmark.loc[~(changed & benchmark.asset_id.eq("a0"))].copy()
        benchmark.loc[benchmark.timestamp.ge("2023-01-04"), "weight"] = 1 / 11
        write_dataset(
            tmp_path / "benchmark",
            benchmark,
            spec=_processed_specs().get("benchmark_weights"),
        )
        config["inputs"]["benchmark"] = str(tmp_path / "benchmark")
    config["output"] = str(tmp_path / "ledger")
    batches = list(
        read_portfolio_ledger_artifact(production.backtest(config)).iter_chunks()
    )
    last = batches[-1]
    assert _facts(last)["decision_skip_reason"] == "all_predictions_missing"
    if independent_exit != "none":
        assert last.trades and all(
            trade.asset_id == "a0" and trade.quantity < 0 for trade in last.trades
        )
    else:
        assert not last.trades
    if missing_context:
        assert _facts(last)["realized_exposure_status"] == "UNDEFINED"
        assert (
            _facts(last)["realized_exposure_reason"]
            == "portfolio_industry_context_missing"
        )


def test_native_new_target_with_missing_industry_skips_and_keeps_account(
    enhanced_config, tmp_path
):
    config = deepcopy(enhanced_config[0])
    panel = read_dataset(config["inputs"]["panel"])
    panel.loc[
        panel.timestamp.eq("2023-01-04"),
        ["industry_standard_id", "industry_level1_code"],
    ] = pd.NA
    write_dataset(
        tmp_path / "panel", panel, spec=_processed_specs().get("market_panel_1d")
    )
    config["inputs"]["panel"] = str(tmp_path / "panel")
    config["output"] = str(tmp_path / "ledger")
    artifact = read_portfolio_ledger_artifact(production.backtest(config))
    batches = list(artifact.iter_chunks())
    assert len(batches) == 3
    assert not batches[-1].trades
    assert (
        _facts(batches[-1])["decision_skip_reason"]
        == "portfolio_industry_context_missing"
    )
    assert _facts(batches[-1])["realized_exposure_status"] == "UNDEFINED"
    assert sum(
        row.value for row in batches[-1].valuations if row.phase == "after"
    ) == pytest.approx(100.0)


def test_cash_missing_stock_exposure_keeps_te_and_persists_local_undefined(tmp_path):
    args = _decision_inputs(holdings=(0.5, 0.4))
    args["cash"] = 10.0
    args["optimizer_config"] = production.OptimizerConfig(active_name_limit=0.2)
    enhanced = args["enhanced_decision"]
    args["enhanced_decision"] = replace(
        enhanced, expected_alpha=enhanced.expected_alpha * 0.0
    )
    enhanced.exposures.loc[enhanced.exposures.asset_id.eq("a"), "style"] = np.nan
    account, adv, optimized, delta, ledger = production._build_session(**args)
    assert optimized is None
    assert ledger.executed_delta_notional.eq(0.0).all()
    assert account.post_trade_cash == 10.0
    assert (
        ledger.attrs["decision_availability"]["decision_skip_reason"]
        == "optimizer_stock_exposure_unavailable"
    )
    batch = production._ledger_batch(
        account=account,
        asset_ledger=ledger,
        sequence=0,
        end_timestamp=args["endpoint_panel"].timestamp.iloc[0],
        decision_timestamp=args["decision_timestamp"],
        held_units=args["held_units"],
        execution_units=args["held_units"],
        previous_values={},
        rights_adjustments=(),
        prediction_rank_score=args["alpha"],
        adv20=adv,
        enhanced_decision=args["enhanced_decision"],
        benchmark_weights=enhanced.benchmark_weights,
        optimization_result=optimized,
        decision_reason=args["decision_reason"],
        decision_delta=delta,
        next_panel=args["next_panel"],
        endpoint_panel=args["endpoint_panel"],
        endpoint_events=pd.DataFrame(),
        corporate_events=pd.DataFrame(),
        endpoint_corporate_events=pd.DataFrame(),
        entitlements=production.EquityEntitlements(),
        execution_lives=args["next_panel"].set_index("asset_id").listing_date,
    )
    artifact = read_portfolio_ledger_artifact(
        write_portfolio_ledger_artifact(
            tmp_path / "ledger",
            capital=PortfolioCapital("timing", 100.0, "CNY", "cash_equity"),
            metadata={"window": {"start": "2023-01-03", "end": "2023-01-05"}},
            produce=lambda emit: emit(batch),
        )
    )
    facts = _facts(list(artifact.iter_chunks())[0])
    assert facts["realized_constraints_status"] == "UNDEFINED"
    assert facts["realized_exposure_status"] == "UNDEFINED"
    assert facts["realized_tracking_error_status"] == "DEFINED"
    assert facts["realized_tracking_error"] == pytest.approx(np.sqrt(0.0002))
    assert facts["realized_style_active_exposure_status"] == "UNDEFINED"
    assert "realized_style_active_exposure" not in facts


@pytest.mark.parametrize("cash_weight", [0.0, 0.5])
def test_exact_zero_stock_coefficients_do_not_consume_missing_exposures(cash_weight):
    weights = np.array([(1.0 - cash_weight) / 2] * 2 + [cash_weight])
    exposure = np.array([[np.nan], [1.0], [0.0]])
    actual = _stock_leg_active_exposure(
        weights, np.array([0.5, 0.5, 0.0]), exposure, np.array([False, False, True])
    )
    assert actual.tolist() == [0.0]


@pytest.mark.parametrize("neutrality", ["industry", "size", "industry_size"])
def test_missing_prediction_independent_exit_needs_no_neutrality_context(neutrality):
    args = _decision_inputs()
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight",
        benchmark_id=None,
        top_quantile=0.5,
        neutrality=neutrality,
    )
    args["enhanced_decision"] = None
    args["alpha"][:] = np.nan
    args["dataset"].loc[args["dataset"].asset_id.eq("b"), "eligible"] = False
    args["dataset"][["float_market_cap", "total_market_cap"]] = np.nan
    args["panel"]["industry_standard_id"] = pd.NA
    args["panel"]["industry_level1_code"] = pd.NA
    account, _, _, _, ledger = production._build_session(**args)
    assert ledger.executed_delta_notional.tolist() == [0.0, -50.0]
    assert account.post_trade_cash == 50.0
    assert (
        ledger.attrs["decision_availability"]["decision_skip_reason"]
        == "all_predictions_missing"
    )


@pytest.mark.parametrize("tilt", [0.0, 1e-10])
def test_proportional_stock_leg_ignores_only_float_normalization_roundoff(tilt):
    weights = np.array([0.3 + tilt, 0.6 - tilt, 0.1])
    reference = np.array([1 / 3, 2 / 3, 0.0])
    exposure = np.array([[np.nan], [1.0], [0.0]])
    actual = _stock_leg_active_exposure(
        weights, reference, exposure, np.array([False, False, True])
    )
    if tilt:
        assert np.isnan(actual[0])
    else:
        assert actual.tolist() == pytest.approx([0.0], abs=1e-15)


@pytest.mark.parametrize("receivable", [0.0, 5.0, 10.0])
def test_enhanced_budget_reserves_receivable_without_renormalizing_nav(receivable):
    args = _decision_inputs(holdings=(0.45, 0.45))
    args.update(
        cash=10.0 - receivable, decision_receivable=receivable, receivable=receivable
    )
    args["optimizer_config"] = production.OptimizerConfig(active_name_limit=0.2)
    account, _, optimized, _, ledger = production._build_session(**args)
    assert optimized is not None
    assert optimized.weights.sum() == pytest.approx(1.0 - receivable / 100.0)
    assert account.nav_after == pytest.approx(100.0)
    assert account.post_trade_cash >= -1e-8
    assert (
        ledger.post_trade_notional.sum() + account.post_trade_cash + receivable
        == pytest.approx(100.0)
    )
    assert ledger.executed_delta_notional.sum() <= 10.0 - receivable + 1e-8


@pytest.mark.parametrize("receivable_weight", [-0.01, 1.01, np.nan, np.inf, True])
def test_optimizer_rejects_invalid_reserved_receivable_weight(receivable_weight):
    from quant_research_platform.research.portfolios.optimizer import optimize_portfolio

    assets = pd.DataFrame(
        {
            "asset_id": ["a"],
            "current_weight": [1.0],
            "benchmark_weight": [1.0],
            "expected_alpha": [0.0],
            "specific_variance": [0.0],
            "adv": [1000.0],
            "style": [1.0],
        }
    )
    with pytest.raises(ValueError, match="optimizer_receivable_weight_invalid"):
        optimize_portfolio(
            with_cash_asset(assets),
            cash_column="is_cash",
            factor_covariance=pd.DataFrame([[0.0]], index=["style"], columns=["style"]),
            style_columns=("style",),
            expected_transaction_cost=lambda buy, sell: (
                0.0,
                np.zeros_like(buy),
                np.zeros_like(sell),
            ),
            nav=100.0,
            assumptions=_decision_inputs()["assumptions"],
            receivable_weight=receivable_weight,
        )


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("industry", "target_neutrality_benchmark_empty"),
        ("size", "target_neutrality_size_missing"),
        ("single_size", "target_neutrality_size_undefined"),
    ],
)
def test_missing_new_target_context_does_not_cancel_independent_exit(kind, reason):
    args = _decision_inputs()
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight",
        benchmark_id=None,
        top_quantile=0.5,
        neutrality="industry" if kind == "industry" else "size",
    )
    args["enhanced_decision"] = None
    args["dataset"].loc[args["dataset"].asset_id.eq("b"), "eligible"] = False
    if kind == "industry":
        args["panel"]["industry_standard_id"] = pd.NA
        args["panel"]["industry_level1_code"] = pd.NA
    elif kind == "size":
        args["dataset"]["total_market_cap"] = np.nan
    account, _, optimized, _, ledger = production._build_session(**args)
    assert optimized is None
    assert ledger.executed_delta_notional.tolist() == [0.0, -50.0]
    assert account.post_trade_cash == 50.0
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] == reason


@pytest.mark.parametrize(
    "construction,neutrality",
    [
        ("benchmark_enhanced", "none"),
        ("quantile_equal_weight", "none"),
        ("quantile_float_cap", "none"),
        ("quantile_equal_weight_half_step", "none"),
        ("quantile_equal_weight", "size"),
    ],
)
def test_every_construction_reserves_receivable_in_executable_target(
    construction, neutrality
):
    args = _decision_inputs(holdings=(0.45, 0.45))
    args.update(cash=0.0, decision_receivable=10.0, receivable=10.0)
    if construction == "benchmark_enhanced":
        args["enhanced_decision"] = replace(args["enhanced_decision"], risk=None)
    else:
        args["candidate"] = replace(
            args["candidate"],
            construction=construction,
            top_quantile=1.0,
            benchmark_id=None,
            risk_style_columns=(),
            neutrality=neutrality,
        )
        args["enhanced_decision"] = None
    account, _, optimized, _, ledger = production._build_session(**args)
    assert optimized is None
    assert (ledger.target_notional / account.nav_before).sum() == pytest.approx(0.9)
    assert (ledger.target_notional / account.nav_before).tolist() == pytest.approx(
        [0.45, 0.45]
    )
    assert ledger.target_notional.sum() == pytest.approx(90.0)
    assert ledger.executed_delta_notional.tolist() == pytest.approx(
        [0.0, 0.0], abs=1e-8
    )
    assert account.nav_after == pytest.approx(100.0)
    assert account.post_trade_cash == pytest.approx(0.0)
    if construction == "benchmark_enhanced":
        assert (
            ledger.attrs["realized_constraints"]["realized_tracking_error_status"]
            == "UNDEFINED"
        )


def test_partial_missing_target_keeps_held_weight_and_reserves_receivable():
    args = _decision_inputs(holdings=(0.45, 0.45))
    args.update(
        cash=0.0, decision_receivable=10.0, receivable=10.0, enhanced_decision=None
    )
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight",
        top_quantile=1.0,
        benchmark_id=None,
        risk_style_columns=(),
    )
    args["alpha"].loc["a"] = np.nan
    account, _, _, _, ledger = production._build_session(**args)
    assert (ledger.target_notional / account.nav_before).tolist() == pytest.approx(
        [0.45, 0.45]
    )
    assert ledger.executed_delta_notional.tolist() == pytest.approx(
        [0.0, 0.0], abs=1e-8
    )
    assert account.post_trade_cash == pytest.approx(0.0)


@pytest.mark.parametrize("receivable", [0.0, 10.0])
@pytest.mark.parametrize("partial", [False, True])
def test_half_step_budgets_new_leg_before_mixing_and_keeps_missing_holdings(
    receivable, partial
):
    args = _decision_inputs(holdings=(0.4, 0.4))
    args.update(
        cash=20.0 - receivable,
        decision_receivable=receivable,
        receivable=receivable,
        enhanced_decision=None,
    )
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight_half_step",
        top_quantile=1.0,
        benchmark_id=None,
        risk_style_columns=(),
    )
    if partial:
        args["alpha"].loc["a"] = np.nan
    account, _, _, _, ledger = production._build_session(**args)
    expected_buy = (20.0 - receivable) / 2.0
    assert ledger.target_notional.sum() == pytest.approx(80.0 + expected_buy)
    assert ledger.executed_delta_notional.sum() == pytest.approx(expected_buy)
    assert account.post_trade_cash == pytest.approx(expected_buy)
    assert account.nav_after == pytest.approx(100.0)
    if partial:
        assert ledger.set_index("asset_id").loc[
            "a", "target_notional"
        ] == pytest.approx(40.0)
        assert ledger.set_index("asset_id").loc["a", "executed_delta_notional"] == 0.0


def test_half_step_neutrality_preserves_budgeted_stock_gross():
    args = _decision_inputs(holdings=(0.4, 0.4))
    args.update(
        cash=10.0, decision_receivable=10.0, receivable=10.0, enhanced_decision=None
    )
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_equal_weight_half_step",
        top_quantile=1.0,
        benchmark_id=None,
        risk_style_columns=(),
        neutrality="size",
    )
    account, _, _, _, ledger = production._build_session(**args)
    assert ledger.raw_target_weight.sum() == pytest.approx(0.85)
    assert ledger.target_weight.sum() == pytest.approx(0.85)
    assert ledger.target_notional.sum() == pytest.approx(85.0)
    assert ledger.executed_delta_notional.sum() == pytest.approx(5.0)
    assert account.post_trade_cash == pytest.approx(5.0)
    assert ledger.attrs["neutrality_exposures"]["target_exposure_status"] == "DEFINED"


@pytest.mark.parametrize(
    "construction", ["quantile_float_cap", "quantile_equal_weight"]
)
def test_missing_float_cap_blocks_only_dependent_target_and_keeps_exit(construction):
    args = _decision_inputs()
    args["enhanced_decision"] = None
    args["candidate"] = replace(
        args["candidate"],
        construction=construction,
        benchmark_id=None,
        top_quantile=1.0,
        risk_style_columns=(),
    )
    args["dataset"].loc[args["dataset"].asset_id.eq("a"), "float_market_cap"] = np.nan
    args["dataset"].loc[args["dataset"].asset_id.eq("b"), "eligible"] = False
    account, _, _, _, ledger = production._build_session(**args)
    if construction == "quantile_float_cap":
        assert (
            ledger.attrs["decision_availability"]["decision_skip_reason"]
            == "target_float_market_cap_missing"
        )
        assert ledger.executed_delta_notional.tolist() == [0.0, -50.0]
        assert account.post_trade_cash == 50.0
    else:
        assert ledger.attrs["decision_availability"]["decision_status"] == "AVAILABLE"
        assert ledger.executed_delta_notional.tolist() == [50.0, -50.0]
        assert account.post_trade_cash == 0.0


@pytest.mark.parametrize("invalid", [-1.0, 0.0, np.inf])
def test_invalid_float_cap_blocks_only_dependent_target(invalid):
    args = _decision_inputs()
    args["enhanced_decision"] = None
    args["candidate"] = replace(
        args["candidate"],
        construction="quantile_float_cap",
        benchmark_id=None,
        top_quantile=1.0,
        risk_style_columns=(),
    )
    args["dataset"].loc[args["dataset"].asset_id.eq("a"), "float_market_cap"] = invalid
    args["dataset"].loc[args["dataset"].asset_id.eq("b"), "eligible"] = False
    account, _, _, _, ledger = production._build_session(**args)
    assert ledger.executed_delta_notional.tolist() == [0.0, -50.0]
    assert account.post_trade_cash == 50.0


@pytest.mark.parametrize(
    "holdings,neutrality,construction",
    [
        ((0.0, 0.0), "industry", "quantile_equal_weight_half_step"),
        ((0.5, 0.5), "industry_size", "quantile_equal_weight_half_step"),
        ((0.0, 0.0), "size", "quantile_float_cap"),
        ((0.5, 0.5), "industry", "quantile_float_cap"),
    ],
)
def test_empty_lawful_stock_domain_preserves_cash_and_proven_exits(
    holdings, neutrality, construction
):
    args = _decision_inputs(holdings=holdings)
    args.update(cash=100.0 - sum(holdings) * 100, enhanced_decision=None)
    args["candidate"] = replace(
        args["candidate"],
        construction=construction,
        benchmark_id=None,
        top_quantile=1.0,
        risk_style_columns=(),
        neutrality=neutrality,
    )
    args["dataset"]["eligible"] = False
    args["dataset"][["total_market_cap", "float_market_cap"]] = np.nan
    account, _, optimized, _, ledger = production._build_session(**args)
    assert optimized is None
    assert ledger.attrs["decision_availability"]["decision_status"] == "AVAILABLE"
    assert ledger.target_notional.eq(0.0).all()
    assert ledger.post_trade_notional.eq(0.0).all()
    assert ledger.executed_delta_notional.tolist() == pytest.approx(
        [-weight * 100 for weight in holdings]
    )
    assert account.post_trade_cash == pytest.approx(100.0)
    for kind in ("target", "realized"):
        assert (
            ledger.attrs["neutrality_exposures"][f"{kind}_exposure_status"]
            == "UNDEFINED"
        )


def test_zero_stock_optimizer_target_keeps_undefined_stock_exposure_local():
    args = _decision_inputs()
    args["dataset"]["eligible"] = False
    args["optimizer_config"] = production.OptimizerConfig(
        active_name_limit=1.0, tracking_error_limit=1.0, target_turnover_limit=1.0
    )
    account, _, optimized, _, ledger = production._build_session(**args)
    assert optimized is not None
    assert ledger.attrs["decision_availability"]["decision_status"] == "AVAILABLE"
    assert ledger.post_trade_notional.eq(0.0).all()
    assert account.post_trade_cash == pytest.approx(100.0)
    diagnostics = ledger.attrs["realized_constraints"]
    assert diagnostics["realized_exposure_status"] == "UNDEFINED"
    assert diagnostics["realized_exposure_reason"] == "nonpositive_stock_value"
    assert diagnostics["realized_tracking_error_status"] == "DEFINED"
    assert diagnostics["realized_tracking_error"] == pytest.approx(np.sqrt(0.015))


def _simple_native_config(enhanced_config, tmp_path, construction):
    config = deepcopy(enhanced_config[0])
    config["candidate"] = dict(
        candidate_id="simple",
        selection_group="global",
        construction=construction,
        prediction_selector="ridge",
        rebalance_schedule="daily",
        top_quantile=1.0,
        benchmark_id=None,
    )
    for name in ("risk_config", "optimizer_config", "calibration"):
        config.pop(name, None)
    for name in ("benchmark", "risk_context"):
        config["inputs"].pop(name, None)
    config["output"] = str(tmp_path / "ledger")
    return config


@pytest.mark.parametrize("payment_date", ["2023-01-04", "2023-01-06"])
def test_native_half_step_cash_and_receivable_payment_control(
    enhanced_config, tmp_path, payment_date
):
    config = _simple_native_config(
        enhanced_config, tmp_path, "quantile_equal_weight_half_step"
    )
    ca = read_dataset(config["inputs"]["corporate_actions"])
    ca["settlement_date"] = pd.Timestamp(payment_date)
    write_dataset(tmp_path / "ca", ca, spec=_interim_specs().get("corporate_actions"))
    config["inputs"]["corporate_actions"] = str(tmp_path / "ca")
    batches = list(
        read_portfolio_ledger_artifact(production.backtest(config)).iter_chunks()
    )
    last = batches[-1]
    after = [row for row in last.valuations if row.phase == "after"]
    pending = sum(row.receivable for row in after)
    assert pending == pytest.approx(0.5 if payment_date == "2023-01-06" else 0.0)
    # Before the final decision stocks are 75, with cash 25 less unpaid rights.
    # The newly budgeted target is 100 less rights; half-step buys only half cash.
    stock_target = (100.0 - pending + 75.0) / 2.0
    assert sum(
        row.value - row.receivable for row in after if row.asset_id != "__cash__"
    ) == pytest.approx(stock_target)
    assert sum(row.value for row in after) == pytest.approx(100.0)
    assert sum(
        row.value for row in after if row.asset_id == "__cash__"
    ) == pytest.approx((25.0 - pending) / 2.0)
    assert sum(trade.notional for trade in last.trades) == pytest.approx(
        (25.0 - pending) / 2.0
    )
    assert _facts(last)["decision_status"] == "AVAILABLE"


def test_native_missing_selected_float_cap_preserves_account(enhanced_config, tmp_path):
    config = _simple_native_config(enhanced_config, tmp_path, "quantile_float_cap")
    data = read_dataset(config["inputs"]["dataset"])
    data.loc[
        data.timestamp.eq("2023-01-04") & data.asset_id.eq("a0"), "float_market_cap"
    ] = np.nan
    write_dataset(tmp_path / "data", data, spec=_data_spec())
    config["inputs"]["dataset"] = str(tmp_path / "data")
    batches = list(
        read_portfolio_ledger_artifact(production.backtest(config)).iter_chunks()
    )
    last = batches[-1]
    assert not last.trades
    assert _facts(last)["decision_skip_reason"] == "target_float_market_cap_missing"
    assert sum(
        row.value for row in last.valuations if row.phase == "after"
    ) == pytest.approx(100.0)


@pytest.mark.parametrize("missing_industry", [False, True])
def test_native_receivable_only_asset_has_no_stock_context_dependency(
    tmp_path, enhanced_config, missing_industry
):
    config, model = deepcopy(enhanced_config)
    data = read_dataset(config["inputs"]["dataset"])
    data.loc[data.timestamp.ge("2023-01-03") & data.asset_id.eq("a0"), "eligible"] = (
        False
    )
    write_dataset(tmp_path / "data", data, spec=_data_spec())
    config["inputs"]["dataset"] = str(tmp_path / "data")
    config["inputs"]["risk_context"] = [str(tmp_path / "data")]
    config["predictions"][0]["path"] = str(
        predict(
            dict(
                model=str(model),
                inputs=str(tmp_path / "data"),
                output=str(tmp_path / "predictions"),
                prediction_window=config["prediction_window"],
                sample_role="validation",
            )
        )
    )
    benchmark = read_dataset(config["inputs"]["benchmark"])
    benchmark = benchmark.loc[
        ~(benchmark.timestamp.ge("2023-01-03") & benchmark.asset_id.eq("a0"))
    ].copy()
    benchmark.loc[benchmark.timestamp.ge("2023-01-03"), "weight"] = 1 / 11
    write_dataset(
        tmp_path / "benchmark",
        benchmark,
        spec=_processed_specs().get("benchmark_weights"),
    )
    config["inputs"]["benchmark"] = str(tmp_path / "benchmark")
    panel = read_dataset(config["inputs"]["panel"])
    # A lawful market move offsets the ex-dividend adjustment, allowing an
    # exact zero-share exit instead of a tiny genuine residual stock position.
    panel[["open", "close", "adj_open", "adj_close"]] = 100.0
    if missing_industry:
        panel.loc[
            panel.timestamp.ge("2023-01-04") & panel.asset_id.eq("a0"),
            ["industry_standard_id", "industry_level1_code"],
        ] = pd.NA
    write_dataset(
        tmp_path / "panel", panel, spec=_processed_specs().get("market_panel_1d")
    )
    config["inputs"]["panel"] = str(tmp_path / "panel")
    config["output"] = str(tmp_path / "ledger")
    batches = list(
        read_portfolio_ledger_artifact(production.backtest(config)).iter_chunks()
    )
    last = batches[-1]
    before = next(
        row for row in last.valuations if row.phase == "before" and row.asset_id == "a0"
    )
    after = next(
        row for row in last.valuations if row.phase == "after" and row.asset_id == "a0"
    )
    assert before.quantity == after.quantity == 0
    assert before.receivable > 0.0
    assert after.receivable == before.receivable
    assert _facts(last)["decision_status"] == "AVAILABLE"
    assert _facts(last)["realized_constraints_status"] == "SATISFIED"
    assert last.trades and all(trade.asset_id != "a0" for trade in last.trades)


def test_optimizer_stock_domain_excludes_zero_current_and_reference_names():
    args = _decision_inputs(benchmark=(1.0, 0.0), holdings=(0.9, 0.0))
    enhanced = args["enhanced_decision"]
    enhanced.exposures.loc[enhanced.exposures.asset_id.eq("b"), "style"] = np.nan
    assets = production._optimizer_assets(
        benchmark_weights=enhanced.benchmark_weights,
        current_weights=args["holdings"] / 100.0,
        cash=0.0,
        nav=100.0,
        receivable=10.0,
        expected_alpha=enhanced.expected_alpha,
        exposures=enhanced.exposures,
        specific_variance=enhanced.risk.specific_variance,
        adv=pd.Series(1000.0, index=["a", "b"]),
        eligible=args["dataset"].set_index("asset_id").eligible,
    )
    assert assets.asset_id.tolist() == ["a", "__cash__"]
    assert assets.current_weight.tolist() == [0.9, 0.0]

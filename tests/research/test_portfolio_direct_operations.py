"""Native portfolio publication and local missing-input semantics."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts import portfolio_ledger as artifacts
from quant_research_platform.contracts.portfolios import (
    ExecutedTrade,
    ExecutionAssumptions,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.research.portfolios.optimizer import (
    OptimizerConfig,
    optimize_portfolio,
)
from quant_research_platform.research.portfolios.production import (
    _ResearchValuationMarks,
)


def _view(day, opening, closing, *, life="2020-01-01", unproved=False):
    frame = pd.DataFrame(
        {
            "asset_id": ["a"],
            "listing_date": [pd.Timestamp(life)],
            "timestamp": [pd.Timestamp(day)],
            "market_type": pd.Series([1], dtype="Int64"),
            "adj_open": [opening],
            "adj_close": [closing],
            "is_trading": pd.Series([True], dtype="boolean"),
        }
    )
    return SimpleNamespace(
        panel=frame,
        valuation=frame.copy(),
        timestamp=pd.Timestamp(day),
        rights_events=pd.DataFrame(),
        closed_assets=frozenset(),
        unproved_lifecycle_assets=frozenset({"a"}) if unproved else frozenset(),
        confirmed_delisted_lifecycles=frozenset(),
    )


def test_ordinary_missing_quote_uses_past_anchor_and_recovers():
    marks = _ResearchValuationMarks()
    marks.project(_view("2023-01-01", 10.0, 11.0))
    missing = _view("2023-01-02", np.nan, 12.0)
    original = missing.panel.copy(deep=True)
    result = marks.project(missing)
    assert result.adj_open.iloc[0] == 11.0
    assert result.adj_close.iloc[0] == 12.0
    assert result.research_open_mark_kind.iloc[0] == "ordinary_missing_quote"
    assert result.research_open_mark_at.iloc[0] == pd.Timestamp("2023-01-01")
    pd.testing.assert_frame_equal(missing.panel, original)
    recovered = marks.project(_view("2023-01-03", 13.0, 14.0))
    assert recovered.adj_open.iloc[0] == 13.0
    assert recovered.research_open_mark_at.isna().all()


@pytest.mark.parametrize("kind", ["no_anchor", "new_life", "unproved"])
def test_no_trusted_same_life_anchor_never_uses_future_close(kind):
    marks = _ResearchValuationMarks()
    if kind != "no_anchor":
        marks.project(_view("2023-01-01", 10.0, 11.0))
    result = marks.project(
        _view(
            "2023-01-02",
            np.nan,
            12.0,
            life="2023-01-02" if kind == "new_life" else "2020-01-01",
            unproved=kind == "unproved",
        )
    )
    assert result.adj_open.isna().all()


def _account():
    start, end = (pd.Timestamp(day).value for day in ("2023-01-01", "2023-01-02"))
    values = []
    for phase, at, shares, cash in (
        ("before", start, 0, 100.0),
        ("after", start, 5, 49.0),
        ("end", end, 5, 49.0),
    ):
        values.extend(
            (
                PortfolioValuation(
                    at,
                    "a",
                    phase,
                    Fraction(shares),
                    shares * 10.0,
                    Fraction(10),
                    lifecycle_id="2020-01-01",
                ),
                PortfolioValuation(at, "__cash__", phase, cash, cash),
            )
        )
    return PortfolioLedgerBatch(
        0,
        trades=(ExecutedTrade("fill", "order", start, "a", Fraction(5), 50.0),),
        cashflows=(
            PortfolioCashFlow("principal", start, "a", "fill", "principal", -50.0),
            PortfolioCashFlow("fee", start, "a", "fill", "commission_cost", -1.0),
        ),
        valuations=tuple(values),
        diagnostics=(
            PortfolioDiagnostic("a", None, "target_notional", number=50.0),
            PortfolioDiagnostic("a", None, "adv20", number=1000.0),
            PortfolioDiagnostic("__cash__", None, "target_notional", number=50.0),
            PortfolioDiagnostic(
                "__cash__", None, "decision_reason", text="simple_rule"
            ),
        ),
        rights_adjustments=(),
    )


def test_native_publication_readback_amounts_and_lazy_consumption(
    tmp_path, monkeypatch
):
    capital = PortfolioCapital("candidate", 100.0, "CNY", "adjusted_notional")
    batch = _account()
    metadata = {"window": {"start": "2023-01-01", "end": "2023-01-02"}}
    root = artifacts.write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata=metadata,
        produce=lambda emit: emit(batch),
    )
    calls = []
    original = artifacts._iter_batches

    def observed(*args, **kwargs):
        calls.append(args[0])
        yield from original(*args, **kwargs)

    monkeypatch.setattr(artifacts, "_iter_batches", observed)
    artifact = artifacts.read_portfolio_ledger_artifact(root)
    assert calls == []
    assert list(artifact.iter_chunks()) == [batch]
    assert len(calls) == 1
    summary = artifacts.summarize_portfolio_ledger(capital, artifact.iter_chunks())
    assert summary["fees"] == 1.0
    assert summary["net_pnl"] == -1.0
    assert summary["nav"] == 99.0
    from quant_research_platform.storage.errors import OutputAlreadyExistsError

    with pytest.raises(OutputAlreadyExistsError):
        artifacts.write_portfolio_ledger_artifact(
            root, capital=capital, metadata=metadata, produce=lambda emit: emit(batch)
        )


@pytest.mark.parametrize("limit,passes", [(0.03, True), (0.02, False)])
def test_frozen_adv_retains_actual_factor_and_specific_risk(limit, passes, monkeypatch):
    import quant_research_platform.research.portfolios.optimizer as optimizer

    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "expected_alpha": [0.0, 0.0],
            "current_weight": [0.6, 0.4],
            "benchmark_weight": [0.5, 0.5],
            "specific_variance": [0.01, 0.0],
            "adv": [np.nan, 1e9],
            "style": [1.0, 0.0],
        }
    )
    captured = {}

    def solve(objective, x0, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(success=True, x=x0)

    monkeypatch.setattr(optimizer, "minimize", solve)
    kwargs = dict(
        factor_covariance=pd.DataFrame([[0.04]], index=["style"], columns=["style"]),
        style_columns=("style",),
        expected_transaction_cost=lambda weights: 0.0,
        nav=1000.0,
        assumptions=ExecutionAssumptions(1000.0, 0.0, 0.0, 1.0, (1.0,)),
        config=OptimizerConfig(active_name_limit=0.2, tracking_error_limit=limit),
    )
    if passes:
        result = optimize_portfolio(frame, **kwargs)
        assert result.factor_variance == pytest.approx(0.0004)
        assert result.specific_variance == pytest.approx(0.0001)
        assert result.predicted_tracking_error == pytest.approx(np.sqrt(0.0005))
    else:
        with pytest.raises(ValueError, match="optimizer_solution_invalid"):
            optimize_portfolio(frame, **kwargs)
    assert captured["bounds"][0] == (0.6, 0.6)


def test_frozen_quantity_is_not_rebalanced_to_old_weight_after_fees():
    from datetime import date

    from quant_research_platform.contracts.portfolios import TradeCostComponents
    from tests.helpers.portfolio_ledger import execute_continuous_target

    assets = pd.Index(["a", "b"])
    result = execute_continuous_target(
        pd.Series([0.4, 0.4], index=assets),
        pd.Series([0.4, 0.5], index=assets),
        nav=1000.0,
        average_daily_volume=pd.Series([np.nan, 10000.0], index=assets),
        can_enter=pd.Series([pd.NA, True], index=assets, dtype="boolean"),
        can_exit=pd.Series([pd.NA, True], index=assets, dtype="boolean"),
        execution_date=date(2023, 1, 1),
        cost_for_trade=lambda day, asset, side, notional, adv, assumptions: (
            TradeCostComponents(
                notional, notional * 0.01, 0.0, 0.0, 0.0, 0.0, "fees.csv"
            )
        ),
        assumptions=ExecutionAssumptions(1000.0, 0.0, 0.0, 1.0, (1.0,)),
        cost_source_identity="fees.csv",
        unquoted_assets=frozenset({"a"}),
    )
    ledger = result.asset_ledger.set_index("asset_id")
    assert ledger.loc["a", "executed_delta_notional"] == 0.0
    assert ledger.post_trade_notional.tolist() == [400.0, 500.0]
    assert result.post_trade_cash == 99.0
    assert result.nav_after == 999.0


@pytest.mark.parametrize("held", [False, True])
def test_unproved_rights_only_block_the_account_that_holds_them(held):
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
    from quant_research_platform.research.portfolios.production import (
        _apply_rights_events,
    )

    event = pd.DataFrame(
        [
            dict(
                asset_id="a",
                listing_date=pd.Timestamp("2020-01-01"),
                kind="cash_recovery",
                unit_basis_proved=False,
                entitlement_proved=True,
            )
        ]
    )
    units = pd.Series(
        {"a": Fraction(2)} if held else {"other": Fraction(3)}, dtype="object"
    )
    lives = pd.Series({asset: pd.Timestamp("2020-01-01") for asset in units.index})
    if held:
        with pytest.raises(
            PortfolioTargetBlocked, match="portfolio_held_rights_unproved"
        ):
            _apply_rights_events(event, units, lives, pd.Timestamp("2023-01-01"))
    else:
        result, _, cash, adjustments = _apply_rights_events(
            event, units, lives, pd.Timestamp("2023-01-01")
        )
        pd.testing.assert_series_equal(result, units)
        assert cash == 0 and adjustments == ()


def test_unproved_rights_quote_cannot_create_a_valuation_anchor():
    marks = _ResearchValuationMarks()
    marks.project(_view("2023-01-01", 10.0, 11.0))
    view = _view("2023-01-02", np.nan, np.nan)
    view.rights_events = pd.DataFrame(
        [
            dict(
                asset_id="a",
                listing_date=pd.Timestamp("2020-01-01"),
                kind="quote",
                unit_basis_proved=False,
                entitlement_proved=True,
            )
        ]
    )
    result = marks.project(view)
    assert result.adj_open.isna().all()
    assert result.adj_close.isna().all()


@pytest.mark.parametrize("consume", ["iterate", "amount_summary"])
def test_consumed_ledger_rejects_unsupported_research_valuation_policy(
    tmp_path, consume
):
    from dataclasses import replace

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.contracts.portfolios import RESEARCH_VALUATION_VERSION

    batch = _account()
    batch = replace(
        batch,
        diagnostics=(
            *batch.diagnostics,
            PortfolioDiagnostic(
                "__cash__",
                None,
                "research_valuation_version",
                text=RESEARCH_VALUATION_VERSION,
            ),
        ),
    )
    capital = PortfolioCapital("policy", 100.0, "CNY", "adjusted_notional")
    root = artifacts.write_portfolio_ledger_artifact(
        tmp_path / "policy",
        capital=capital,
        metadata={"window": {"start": "2023-01-01", "end": "2023-01-02"}},
        produce=lambda emit: emit(batch),
    )
    path = root / "diagnostics.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows:
        if row["name"] == "research_valuation_version":
            row["text"] = "unsupported_scientific_policy"
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(
        ValueError, match="portfolio_research_valuation_version_invalid"
    ):
        if consume == "iterate":
            list(artifacts.read_portfolio_ledger_artifact(root).iter_chunks())
        else:
            artifacts.read_portfolio_ledger_artifact(
                root,
                consume_batches=lambda artifact, batches: (
                    artifacts.summarize_portfolio_ledger(artifact.capital, batches)
                ),
            )


def test_explicit_native_fit_prediction_and_backtest_roundtrip(tmp_path):
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.portfolios.production import backtest
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import read_dataset
    from tests.research.direct_model_test_support import fit_frame
    from tests.research.test_closed_market_portfolios import _native_inputs

    inputs, _, _, _ = _native_inputs(tmp_path / "market", fault="normal_open")
    training = read_dataset(inputs["dataset"])
    training = training.loc[training.timestamp.le("2022-12-20")].copy()
    training["signal"] = training.asset_id.str[1:].astype(float)
    training["label"] = training.signal * 0.001
    training["label_available_at"] = training.timestamp
    fitted = fit_frame(
        tmp_path / "learning",
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
    predicted = predict(
        {
            "model": str(fitted.root),
            "inputs": str(inputs["dataset"]),
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    config = {
        "candidate": {
            "candidate_id": "native",
            "selection_group": "global",
            "construction": "top20_equal_weight",
            "prediction_selector": "ridge",
            "rebalance_schedule": "daily",
            "top_quantile": 0.5,
            "benchmark_id": None,
        },
        "inputs": {
            key: str(value) for key, value in inputs.items() if key != "risk_context"
        },
        "predictions": [
            {"path": str(predicted), "model": str(fitted.root), "window": window}
        ],
        "output": str(tmp_path / "ledger"),
        "output_window": window,
        "input_window": {"start": "2022-12-01", "end": "2023-01-06"},
        "formal_dataset_window": {"start": "2022-12-01", "end": "2023-01-06"},
        "prediction_window": window,
        "sample_role": "validation",
        "assumptions": {
            "initial_nav": 100.0,
            "slippage_bps": 0.0,
            "impact_bps": 0.0,
            "adv_participation": 1.0,
            "adv_sensitivities": [1.0],
        },
    }
    root = backtest(config)
    ledger = artifacts.read_portfolio_ledger_artifact(root)
    batches = list(ledger.iter_chunks())
    assert batches and any(batch.trades for batch in batches)
    summary = artifacts.summarize_portfolio_ledger(ledger.capital, iter(batches))
    assert np.isfinite(summary["nav"])
    assert summary["fees"] > 0
    assert ledger.metadata["initial_holding_convention"] == "cash"


def test_adv_warmup_counts_twenty_own_market_observations_across_union_holiday(
    tmp_path, monkeypatch
):
    from quant_research_platform.contracts.time import TimeWindow
    from tests.research.test_portfolio_production import _build_producer, _inputs

    frames, fees, window = _inputs()
    frames["panel"].loc[frames["panel"].asset_id.eq("b"), "market_type"] = 4
    other_calendar = frames["calendar"].assign(source_market_type=4)
    other_calendar.loc[
        other_calendar.calendar_date.eq("2022-12-20"), "is_trading_day"
    ] = False
    frames["calendar"] = pd.concat(
        (frames["calendar"], other_calendar), ignore_index=True
    )
    for name in ("panel", "dataset", "execution"):
        frame = frames[name]
        frames[name] = frame.loc[
            ~(frame.asset_id.eq("b") & frame.timestamp.eq("2022-12-20"))
        ].copy()
    output, _ = _build_producer(
        tmp_path / "inputs",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        input_window=TimeWindow("2022-12-01", "2023-01-10"),
        typed_batches=True,
    )
    batches = list(output)
    first = batches[0]
    assert any(trade.asset_id == "b" and trade.notional > 0 for trade in first.trades)
    assert any(
        row.asset_id == "b" and row.name == "adv20" and row.number == 1000.0
        for row in first.diagnostics
    )
    root = artifacts.write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("holiday", 100.0, "CNY", "adjusted_notional"),
        metadata={"window": {"start": str(window.start), "end": str(window.end)}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    artifact = artifacts.read_portfolio_ledger_artifact(root)
    assert (
        artifacts.summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())[
            "turnover"
        ]
        > 0
    )

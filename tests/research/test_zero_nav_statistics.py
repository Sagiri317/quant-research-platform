"""Amount-ledger recovery and undefined rate persistence after total loss."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.portfolios import (
    ExecutedTrade,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioRightsAdjustment,
    PortfolioValuation,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.attribution import (
    HoldingFactorInputs,
)
from quant_research_platform.research.portfolios.diagnostics import (
    _portfolio_statistics_tables,
)
from tests.helpers.statistics_artifacts import (
    portfolio_statistics_config,
    statistics_metadata,
)


@pytest.mark.parametrize("recovery", ["quote", "cash_retained", "cash_cancelled"])
@pytest.mark.parametrize("access", ["validation", "test"])
def test_zero_nav_rights_recovery_persists_amounts_and_undefined_returns(
    tmp_path, access, recovery
):
    dates = pd.bdate_range("2023-01-02", periods=6)
    window = {
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "access": access,
    }
    candidates, manifest = portfolio_statistics_config(sample_role=access)
    is_cash = recovery.startswith("cash")
    initial = 1000.0
    terminal = 500.0 if is_cash else 550.0
    marks = (10.0, 0.0, 0.0, 0.0, 0.0) if is_cash else (10.0, 0.0, 0.0, 5.0, 5.5)
    batches = []
    for sequence, (start, finish) in enumerate(zip(marks[:-1], marks[1:], strict=True)):
        at, end = dates[sequence + 1].value, dates[sequence + 2].value
        valuations = []
        diagnostics = []
        for phase in ("before", "after", "end"):
            quantity = 0.0 if phase == "before" and sequence == 0 else 100.0
            if recovery == "cash_cancelled" and (
                sequence > 2 or (sequence == 2 and phase != "before")
            ):
                quantity = 0.0
            price = finish if phase == "end" else start
            stamp = end if phase == "end" else at
            valuations.append(
                PortfolioValuation(
                    stamp,
                    "asset",
                    phase,
                    float(quantity),
                    float(float(quantity) * float(price)),
                    price,
                    lifecycle_id="2000-01-01",
                )
            )
            cash = initial if phase == "before" and sequence == 0 else 0.0
            if is_cash and (sequence > 2 or (sequence == 2 and phase != "before")):
                cash = 500.0
            valuations.append(PortfolioValuation(stamp, "__cash__", phase, cash, cash))
        for asset, target in (
            ("asset", float(100 * start)),
            ("__cash__", 500.0 if is_cash and sequence >= 2 else 0.0),
        ):
            diagnostics.extend(
                (
                    PortfolioDiagnostic(asset, None, "target_notional", target),
                    PortfolioDiagnostic(asset, None, "decision_delta_weight", 0.0),
                )
            )
        diagnostics.extend(
            (
                PortfolioDiagnostic("asset", None, "adv20", 1000000.0),
                PortfolioDiagnostic(
                    "__cash__", None, "decision_reason", text="simple_rule"
                ),
            )
        )
        trades = (
            (ExecutedTrade("buy", "buy", at, "asset", float(100), initial),)
            if sequence == 0
            else ()
        )
        flows = (
            (
                PortfolioCashFlow(
                    "buy:principal", at, "asset", "buy", "principal", -initial
                ),
            )
            if sequence == 0
            else ()
        )
        adjustments = ()
        if is_cash and sequence == 2:
            flows = (
                PortfolioCashFlow(
                    "recovery:recovery", at, "asset", None, "recovery", 500.0
                ),
            )
            adjustments = (
                PortfolioRightsAdjustment(
                    "recovery",
                    at,
                    "asset",
                    "2000-01-01",
                    100.0,
                    0.0 if recovery == "cash_cancelled" else 100.0,
                    500.0,
                    None,
                    None,
                    0.0,
                    "d" * 64,
                ),
            )
        batches.append(
            PortfolioLedgerBatch(
                sequence,
                trades,
                flows,
                tuple(valuations),
                tuple(diagnostics),
                adjustments,
            )
        )

    ledgers = {}
    for candidate in candidates:
        capital = PortfolioCapital(
            **{
                "candidate_id": candidate.candidate_id,
                "initial_capital": initial,
                "valuation_currency": "CNY",
                "valuation_method": "cash_equity",
            }
        )
        metadata = {
            "window": {key: value for key, value in window.items() if key != "access"},
            "sample_role": access,
        }
        root = write_portfolio_ledger_artifact(
            tmp_path / candidate.candidate_id,
            capital=capital,
            metadata=metadata,
            produce=lambda emit, batches=batches: [emit(batch) for batch in batches],
        )
        ledger = read_portfolio_ledger_artifact(root)
        ledgers[candidate.candidate_id] = ledger
        summary = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
        assert summary["nav"] == terminal
        assert summary["net_pnl"] == terminal - initial
        assert summary["capital_return"] == pytest.approx(
            (terminal - initial) / initial
        )
        assert summary["capital_return_reason"] is None
        assert summary["fill_events"] == 1
        restored = list(ledger.iter_chunks())
        assert (
            next(
                row
                for row in restored[1].valuations
                if row.asset_id == "asset" and row.phase == "end"
            ).quantity
            == 100.0
        )

    if is_cash:
        assert (
            sum(
                flow.amount
                for batch in restored
                for flow in batch.cashflows
                if flow.kind == "recovery"
            )
            == 500.0
        )
        assert sum(len(batch.rights_adjustments) for batch in restored) == 1
        final_position = next(
            row
            for row in restored[-1].valuations
            if row.asset_id == "asset" and row.phase == "end"
        )
        assert final_position.quantity == (
            0.0 if recovery == "cash_cancelled" else 100.0
        )
    inputs = [
        HoldingFactorInputs(
            dates[i],
            dates[i + 1],
            dates[i + 2],
            None,
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            pd.DataFrame(),
            "holding_factor_fit_unidentified",
        )
        for i in range(4)
    ]
    benchmark_opens = pd.DataFrame(columns=("benchmark_id", "timestamp", "open"))
    tables = _portfolio_statistics_tables(
        ledgers=ledgers,
        candidates=candidates,
        config=manifest,
        time_window=TimeWindow(dates[0], dates[-1]),
        benchmark_opens=benchmark_opens,
        calendar=pd.DataFrame({"calendar_date": dates}),
        unified_trading_sessions=lambda frame: pd.DatetimeIndex(frame.calendar_date),
        holding_inputs={
            candidate.risk_style_columns: iter(inputs) for candidate in candidates
        },
    )
    daily = tables["portfolio_daily"]
    first = daily.loc[daily.candidate_id.eq(candidates[0].candidate_id)]
    assert first.nav_end.tolist() == [
        0.0,
        0.0,
        500.0,
        terminal,
    ]
    assert first.return_status.tolist() == [
        "DEFINED",
        "UNDEFINED",
        "UNDEFINED",
        "DEFINED",
    ]
    assert first.net_return.iloc[0] == -1.0
    assert first.net_return.iloc[2:3].isna().all()
    assert first.return_reason.iloc[2] == "nonpositive_start_nav"
    assert first.net_return.iloc[3] == (0.0 if is_cash else 0.1)
    performance = tables["portfolio_performance"].set_index(
        ["candidate_id", "metric_id"]
    )
    candidate = candidates[0].candidate_id
    assert performance.loc[(candidate, "terminal_nav"), "value"] == terminal
    assert performance.loc[(candidate, "max_drawdown"), "value"] == -1.0
    assert performance.loc[(candidate, "sharpe"), "status"] == "UNDEFINED"
    assert performance.loc[(candidate, "cost"), "value"] == 0.0
    metadata = statistics_metadata(
        TimeWindow(window["start"], window["end"]),
        sample_role=access,
        candidate_ids=[c.candidate_id for c in candidates],
    )
    root = write_statistics_artifact(
        tmp_path / "statistics",
        metadata=metadata,
        tables=tables,
        table_group="portfolio",
    )
    artifact = read_statistics_artifact(root)
    with artifact.table("portfolio_daily.parquet") as persisted:
        assert len(persisted) == 4 * len(candidates)
        assert persisted.loc[persisted.nav_before.eq(0), "net_return"].isna().all()


def test_zero_value_retained_units_are_counted_in_valuation_diagnostics():
    from quant_research_platform.research.portfolios.daily import (
        RESEARCH_VALUATION_POLICY,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _research_valuation_observation,
    )

    diagnostics = [
        PortfolioDiagnostic(
            "__cash__",
            None,
            "research_valuation_policy",
            text=RESEARCH_VALUATION_POLICY,
        ),
        *(
            PortfolioDiagnostic(
                "__cash__", None, f"research_valuation_nav_{percent}pct", 0.0
            )
            for percent in (0, 50, 100)
        ),
    ]
    for phase in ("entry", "end"):
        diagnostics.extend(
            (
                PortfolioDiagnostic(
                    "asset", None, f"research_{phase}_mark_at", text="2023-01-01"
                ),
                PortfolioDiagnostic(
                    "asset", None, f"research_{phase}_mark_age_days", 1.0
                ),
            )
        )
    batch = PortfolioLedgerBatch(
        0,
        (),
        (),
        tuple(
            row
            for phase in ("before", "after", "end")
            for row in (
                PortfolioValuation(1, "asset", phase, 100.0, 0.0, 0.0),
                PortfolioValuation(1, "__cash__", phase, 0.0, 0.0),
            )
        ),
        tuple(diagnostics),
    )
    result = _research_valuation_observation(
        batch,
        {
            "entry_at": pd.Timestamp("2023-01-02"),
            "end_at": pd.Timestamp("2023-01-03"),
            "nav_end": 0.0,
        },
    )
    assert result["estimated_held_assets"] == 1
    assert result["estimated_value_end"] == 0.0
    assert pd.isna(result["estimated_weight_end"])

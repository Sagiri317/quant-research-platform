"""Amount-ledger recovery and undefined rate persistence after total loss."""

from dataclasses import replace
from fractions import Fraction

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
from quant_research_platform.research.portfolios.diagnostics import (
    _portfolio_statistics_tables,
)
from quant_research_platform.research.portfolios.risk import HoldingFactorInputs
from tests.helpers.statistics_artifacts import (
    portfolio_statistics_config,
    statistics_metadata,
)


@pytest.mark.parametrize(
    "recovery", ["quote", "cash_retained", "cash_cancelled", "cash_tiny_start"]
)
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
    tiny = recovery == "cash_tiny_start"
    initial = float.fromhex("0x0.0000000000001p-1022") if tiny else 1000.0
    terminal = 500.0 if is_cash else 550.0
    marks = (10.0, 0.0, 0.0, 0.0, 0.0) if is_cash else (10.0, 0.0, 0.0, 5.0, 5.5)
    if tiny:
        marks = (Fraction(initial) / 100,) * 5
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
                    Fraction(quantity),
                    float(Fraction(quantity) * Fraction(price)),
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
            (ExecutedTrade("buy", "buy", at, "asset", Fraction(100), initial),)
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
            produce=lambda emit: [emit(batch) for batch in batches],
        )
        ledger = read_portfolio_ledger_artifact(root)
        ledgers[candidate.candidate_id] = ledger
        summary = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
        assert summary["nav"] == terminal
        assert summary["net_pnl"] == terminal - initial
        assert summary["capital_return"] == (
            None if tiny else (terminal - initial) / initial
        )
        assert summary["capital_return_reason"] == (
            "nonfinite_result" if tiny else None
        )
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

    if recovery == "cash_retained":
        replay = replace(
            batches[2].rights_adjustments[0],
            timestamp_ns=dates[4].value,
            cash_amount=0.0,
        )
        repeated = [*batches[:3], replace(batches[3], rights_adjustments=(replay,))]
        with pytest.raises(
            ValueError, match="portfolio_ledger_duplicate_rights_adjustment"
        ):
            write_portfolio_ledger_artifact(
                tmp_path / "replayed",
                capital=capital,
                metadata=metadata,
                produce=lambda emit: [emit(batch) for batch in repeated],
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
    if tiny:
        benchmark_opens = pd.DataFrame(
            [
                {"benchmark_id": benchmark, "timestamp": date, "open": value}
                for benchmark in ("csi500", "csi1000")
                for date, value in zip(
                    dates[1:], (initial, initial, initial, 500.0, 500.0), strict=True
                )
            ]
        )
    if tiny:
        from tests.helpers.benchmark_references import reference_input

        manifest["analyses"].append("benchmark_comparison")
        manifest["execution_mode"] = "continuous"
        manifest["comparison_benchmarks"] = [
            reference_input(
                tmp_path / "tiny-reference.parquet",
                [initial, initial, initial, 500.0, 500.0],
                dates=dates[1:],
            )
        ]
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
        initial if tiny else 0.0,
        initial if tiny else 0.0,
        500.0,
        terminal,
    ]
    assert first.return_status.tolist() == [
        "DEFINED",
        "DEFINED" if tiny else "UNDEFINED",
        "UNDEFINED",
        "DEFINED",
    ]
    assert first.net_return.iloc[0] == (0.0 if tiny else -1.0)
    assert first.net_return.iloc[2:3].isna().all()
    assert first.return_reason.iloc[2] == (
        "nonfinite_result" if tiny else "nonpositive_start_nav"
    )
    assert first.net_return.iloc[3] == (0.0 if is_cash else 0.1)
    if tiny:
        compared = tables["portfolio_benchmark_intervals"]
        for _, enhanced in compared.groupby("candidate_id"):
            assert enhanced.benchmark_status.tolist() == [
                "DEFINED",
                "DEFINED",
                "UNDEFINED",
                "DEFINED",
            ]
            assert enhanced.benchmark_reason.iloc[2] == "nonfinite_result"
            assert enhanced.relative_wealth.iloc[:2].eq(1).all()
            assert enhanced.relative_wealth.iloc[2:].isna().all()
    performance = tables["portfolio_performance"].set_index(
        ["candidate_id", "metric_id"]
    )
    candidate = candidates[0].candidate_id
    assert performance.loc[(candidate, "terminal_nav"), "value"] == terminal
    assert performance.loc[(candidate, "max_drawdown"), "value"] == (
        0.0 if tiny else -1.0
    )
    assert performance.loc[(candidate, "sharpe"), "status"] == "UNDEFINED"
    assert performance.loc[(candidate, "cost"), "value"] == 0.0
    assert tables["portfolio_group_comparison"].status.eq("UNDEFINED").all()
    for group in ("global", "quarterly"):
        wealth = tables["portfolio_group_comparison"][f"{group}_terminal_wealth"]
        assert wealth.isna().all() if tiny else wealth.eq(terminal / initial).all()
        assert (
            tables["portfolio_group_comparison"][f"{group}_terminal_wealth_reason"]
            .eq("nonfinite_result" if tiny else "")
            .all()
        )
    metadata = statistics_metadata(
        TimeWindow(window["start"], window["end"]),
        sample_role=access,
        execution_mode="continuous",
        candidate_ids=[c.candidate_id for c in candidates],
        comparison_portfolio_ids=manifest["statistics"]["group_comparison"][
            "portfolio_ids"
        ],
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
    damaged = {**tables, "portfolio_daily": daily.copy()}
    damaged["portfolio_daily"].loc[
        daily.return_status.eq("UNDEFINED"), "net_return"
    ] = 0.0
    with pytest.raises(ValueError, match="statistics_return_status_invalid"):
        write_statistics_artifact(
            tmp_path / "invalid",
            metadata=metadata,
            tables=damaged,
            table_group="portfolio",
        )


def test_undefined_return_pair_preserves_other_pairs_and_fdr_family():
    import numpy as np

    from quant_research_platform.research.portfolios.diagnostics import (
        _group_portfolio_comparison,
    )

    _, manifest = portfolio_statistics_config(sample_role="test")
    pairs = manifest["statistics"]["group_comparison"]["portfolio_ids"]
    dates = pd.bdate_range("2023-01-02", periods=76)
    matrix = pd.DataFrame(
        np.random.default_rng(51).normal(0.0001, 0.01, (75, 2 * len(pairs))),
        columns=[
            f"{group}_{pair}" for pair in pairs for group in ("global", "quarterly")
        ],
        index=pd.MultiIndex.from_arrays(
            [dates[:-1], dates[1:]], names=["entry_at", "end_at"]
        ),
    )
    matrix.iloc[20, 0] = np.nan
    result = _group_portfolio_comparison(matrix, manifest).set_index("portfolio_id")
    assert len(result) == len(pairs)
    assert result.fdr_family_size.eq(len(pairs)).all()
    assert result.observation_count.eq(75).all()
    assert result.loc[pairs[0], "status"] == "UNDEFINED"
    assert result.loc[pairs[0], "reason"] == "nonpositive_start_nav"
    assert pd.isna(result.loc[pairs[0], "global_terminal_wealth"])
    assert result.loc[pairs[1:], "status"].eq("DEFINED").all()


def test_zero_value_retained_units_are_counted_in_valuation_diagnostics():
    from quant_research_platform.research.portfolios.diagnostics import (
        _research_valuation_observation,
    )
    from quant_research_platform.research.portfolios.production import (
        RESEARCH_VALUATION_POLICY,
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


@pytest.mark.parametrize(
    "value", [0.0, -0.0, 1.5, float.fromhex("0x0.0000000000001p-1022")]
)
def test_float_quantity_encoding_preserves_crypto_type_and_bits(value):
    from quant_research_platform.contracts.validation import (
        decode_exact_numeric,
        encode_exact_numeric,
    )

    restored = decode_exact_numeric(encode_exact_numeric(value))
    assert type(restored) is float
    assert restored.hex() == value.hex()


@pytest.mark.parametrize(
    "encoded",
    ["q:0x2/0x4", "q:0x0/0x2", "q:0x1/-0x2", "q:0x01/0x2", "f:1.0", "1.0", None],
)
def test_quantity_encoding_rejects_noncanonical_persistence(encoded):
    from quant_research_platform.contracts.validation import decode_exact_numeric

    with pytest.raises(ValueError, match="exact_numeric_encoding_invalid"):
        decode_exact_numeric(encoded)


@pytest.mark.parametrize("kind", ["quantity", "mark", "fractional"])
def test_exact_quantity_encoding_exceeds_float_without_losing_amount_ledger(
    tmp_path, kind
):
    from fractions import Fraction

    from quant_research_platform.contracts.validation import (
        decode_exact_numeric,
        encode_exact_numeric,
    )

    mark = (
        float.fromhex("0x0.0000000000001p-1022")
        if kind == "quantity"
        else Fraction(3, 100)
        if kind == "fractional"
        else Fraction(10**400)
    )
    quantity = Fraction(1, 1) / Fraction(mark)
    assert decode_exact_numeric(encode_exact_numeric(quantity)) == quantity
    valuations = tuple(
        row
        for phase in ("before", "after", "end")
        for row in (
            PortfolioValuation(
                1 if phase == "end" else 0,
                "asset",
                phase,
                Fraction() if phase == "before" else quantity,
                0.0 if phase == "before" else 1.0,
                mark,
            ),
            PortfolioValuation(
                1 if phase == "end" else 0,
                "__cash__",
                phase,
                1.0 if phase == "before" else 0.0,
                1.0 if phase == "before" else 0.0,
            ),
        )
    )
    batch = PortfolioLedgerBatch(
        0,
        (ExecutedTrade("buy", "buy", 0, "asset", quantity, 1.0),),
        (PortfolioCashFlow("principal", 0, "asset", "buy", "principal", -1.0),),
        valuations,
        (
            PortfolioDiagnostic(
                "__cash__", None, "decision_reason", text="simple_rule"
            ),
            PortfolioDiagnostic("asset", None, "target_notional", 1.0),
            PortfolioDiagnostic("__cash__", None, "target_notional", 0.0),
            PortfolioDiagnostic("asset", None, "adv20", 1.0),
        ),
    )
    capital = PortfolioCapital(
        **{
            "candidate_id": "exact",
            "initial_capital": 1.0,
            "valuation_currency": "CNY",
            "valuation_method": "cash_equity",
        }
    )
    metadata = {
        "window": {
            key: value
            for key, value in {
                "start": "1970-01-01",
                "end": "1970-01-02",
                "access": "validation",
            }.items()
            if key != "access"
        },
        "sample_role": "validation",
    }
    root = write_portfolio_ledger_artifact(
        tmp_path / "exact",
        capital=capital,
        metadata=metadata,
        produce=lambda emit: emit(batch),
    )
    artifact = read_portfolio_ledger_artifact(root)
    summary = summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())
    assert summary["nav"] == 1.0
    assert summary["net_pnl"] == 0.0
    if kind == "fractional":
        assert summary["fill_quantity"] == float(quantity)
        assert summary["fill_quantity_reason"] is None
    else:
        assert summary["fill_quantity"] is None
        assert summary["fill_quantity_reason"] == "nonfinite_result"
    restored = list(artifact.iter_chunks())[0]
    assert restored.trades[0].quantity == quantity
    assert isinstance(restored.trades[0].quantity, Fraction)

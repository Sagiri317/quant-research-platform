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
    assert tables["portfolio_group_comparison"].status.eq("UNDEFINED").all()
    for group in ("global", "quarterly"):
        wealth = tables["portfolio_group_comparison"][f"{group}_terminal_wealth"]
        assert wealth.to_numpy() == pytest.approx(terminal / initial)
        assert (
            tables["portfolio_group_comparison"][f"{group}_terminal_wealth_reason"]
            .eq("")
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
    with artifact.table("portfolio_group_comparison.parquet") as persisted:
        for group in ("global", "quarterly"):
            column = f"{group}_terminal_wealth"
            pd.testing.assert_series_equal(
                persisted[column],
                tables["portfolio_group_comparison"][column].astype("float64"),
                check_exact=True,
            )
            assert persisted[f"{column}_reason"].eq("").all()
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


@pytest.mark.parametrize("daily_complete", [False, True])
@pytest.mark.parametrize("empty_pair", [False, True])
def test_undefined_return_pair_preserves_other_pairs_and_fdr_family(
    tmp_path, daily_complete, empty_pair
):
    import numpy as np

    from quant_research_platform.research.portfolios.diagnostics import (
        _group_portfolio_comparison,
    )

    from math import erfc, sqrt

    pairs = ["affected", "ordinary"]
    manifest = {
        "sample_role": "test",
        "statistics": {
            "hac_lag": 0,
            "bootstrap_block": 1,
            "bootstrap_repetitions": 2,
            "group_comparison": {
                "portfolio_ids": pairs,
                "primary_portfolio_id": pairs[0],
                "fdr_method": "BY",
                "alpha": 0.05,
                "seed": 42,
                "confidence": 0.95,
            },
        },
    }
    dates = pd.bdate_range("2024-01-01", periods=3)
    matrix = pd.DataFrame(
        {
            "global_affected": [np.nan if empty_pair else 0.0, np.nan],
            "quarterly_affected": [0.2, 1000.0],
            "global_ordinary": [0.0, 0.0],
            "quarterly_ordinary": [0.01, 0.03],
        },
        index=pd.MultiIndex.from_arrays(
            [dates[:-1], dates[1:]], names=["entry_at", "end_at"]
        ),
    )
    result = _group_portfolio_comparison(
        matrix, manifest, daily_complete=daily_complete
    ).set_index("portfolio_id")
    assert len(result) == len(pairs)
    assert result.fdr_family_size.eq(len(pairs)).all()
    assert result.observation_count.eq(2).all()
    assert result.loc[pairs[0], "status"] == "UNDEFINED"
    reason = "nonpositive_start_nav" if daily_complete else "incomplete_daily_axis"
    assert result.loc[pairs[0], "reason"] == reason
    assert result.loc[pairs[0], "bootstrap_reason"] == reason
    assert (
        result.loc[pairs[1:], "status"]
        .eq("DEFINED" if daily_complete else "UNDEFINED")
        .all()
    )
    ordinary = result.loc[pairs[1]]
    assert ordinary.bootstrap_status == ("DEFINED" if daily_complete else "UNDEFINED")
    if daily_complete:
        standard_error = sqrt((0.01**2 + 0.01**2) / 4)
        p_value = erfc((0.02 / standard_error) / sqrt(2))
        assert ordinary.standard_error == pytest.approx(standard_error)
        assert ordinary.p_value == pytest.approx(p_value)
        # BY retains both planned hypotheses, including the undefined pair.
        assert ordinary.fdr_adjusted_p_value == pytest.approx(3 * p_value)
    means = ["global_mean", "quarterly_mean", "mean_difference"]
    if empty_pair:
        assert result.loc[pairs[0], means].isna().all()
    else:
        assert result.loc[pairs[0], means].to_numpy(dtype=float) == pytest.approx(
            [0.0, 0.2, 0.2]
        )
    # The inference helper has no endpoint ledger; wealth is independently supplied.
    for group in ("global", "quarterly"):
        result[f"{group}_terminal_wealth"] = 1.0
        result[f"{group}_terminal_wealth_reason"] = ""
    output = write_statistics_artifact(
        tmp_path / "comparison",
        metadata=statistics_metadata(
            TimeWindow(dates[0], dates[-1]),
            sample_role="test",
            comparison_portfolio_ids=pairs,
        ),
        tables={"portfolio_group_comparison": result.reset_index()},
        table_group="portfolio",
    )
    with read_statistics_artifact(output).table(
        "portfolio_group_comparison.parquet"
    ) as saved:
        pd.testing.assert_frame_equal(
            saved.set_index("portfolio_id")[means], result[means], check_exact=True
        )


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


def test_fractional_quantity_native_amount_ledger(tmp_path):
    from fractions import Fraction

    from quant_research_platform.contracts.validation import (
        decode_exact_numeric,
        encode_exact_numeric,
    )

    mark = Fraction(3, 100)
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
    assert summary["fill_quantity"] == float(quantity)
    assert summary["fill_quantity_reason"] is None
    restored = list(artifact.iter_chunks())[0]
    assert restored.trades[0].quantity == quantity
    assert isinstance(restored.trades[0].quantity, Fraction)


@pytest.mark.parametrize("with_calendar", [False, True])
def test_native_group_descriptions_share_finite_days(tmp_path, with_calendar):
    from dataclasses import asdict

    from quant_research_platform.contracts.portfolios import PortfolioCandidateSpec
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )
    from quant_research_platform.sources.csmar.interim import interim_catalog
    from quant_research_platform.storage.datasets import write_dataset

    dates = pd.bdate_range("2024-01-01", periods=3)
    window = {"start": str(dates[0].date()), "end": str(dates[-1].date())}
    paths = []
    # G loses everything; Q remains positive. Only the first day has a finite pair.
    for group, marks in (
        ("global", [10, 0, 0]),
        ("quarterly", [10, 11, 13.2]),
    ):
        candidate = PortfolioCandidateSpec(
            candidate_id=f"{group}_ordinary",
            selection_group=group,
            construction="quantile_equal_weight",
            prediction_selector="champion",
            rebalance_schedule="daily",
            top_quantile=1.0,
            benchmark_id=None,
        )
        batches = []
        for i, (start, finish) in enumerate(zip(marks[:-1], marks[1:], strict=True)):
            at, end = dates[i].value, dates[i + 1].value
            valuations = []
            for phase in ("before", "after", "end"):
                quantity = Fraction(0 if phase == "before" and i == 0 else 100)
                price = finish if phase == "end" else start
                stamp = end if phase == "end" else at
                cash = 1000.0 if phase == "before" and i == 0 else 0.0
                valuations.extend(
                    (
                        PortfolioValuation(
                            stamp,
                            "asset",
                            phase,
                            quantity,
                            float(quantity * Fraction(price)),
                            price,
                            lifecycle_id="2000-01-01",
                        ),
                        PortfolioValuation(stamp, "__cash__", phase, cash, cash),
                    )
                )
            diagnostics = (
                PortfolioDiagnostic("asset", None, "adv20", 1000000.0),
                PortfolioDiagnostic("asset", None, "target_notional", 100 * start),
                PortfolioDiagnostic("__cash__", None, "target_notional", 0.0),
                PortfolioDiagnostic("asset", None, "decision_delta_weight", 0.0),
                PortfolioDiagnostic("__cash__", None, "decision_delta_weight", 0.0),
                PortfolioDiagnostic(
                    "__cash__", None, "decision_reason", text="simple_rule"
                ),
            )
            batches.append(
                PortfolioLedgerBatch(
                    i,
                    (ExecutedTrade("buy", "buy", at, "asset", Fraction(100), 1000.0),)
                    if i == 0
                    else (),
                    (
                        PortfolioCashFlow(
                            "buy:principal", at, "asset", "buy", "principal", -1000.0
                        ),
                    )
                    if i == 0
                    else (),
                    tuple(valuations),
                    diagnostics,
                )
            )
        paths.append(
            str(
                write_portfolio_ledger_artifact(
                    tmp_path / group,
                    capital=PortfolioCapital(
                        candidate.candidate_id, 1000.0, "CNY", "cash_equity"
                    ),
                    metadata={
                        "window": window,
                        "sample_role": "exploratory",
                        "candidate": asdict(candidate),
                    },
                    produce=lambda emit: [emit(batch) for batch in batches],
                )
            )
        )
    inputs = {}
    if with_calendar:
        calendar = tmp_path / "calendar"
        write_dataset(
            calendar,
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "calendar_date": dates,
                    "source_market_type": 1,
                    "is_trading_day": True,
                    "source_day_of_week": dates.dayofweek,
                    "source_state": "O",
                }
            ),
            spec=interim_catalog().get("trading_calendar"),
        )
        inputs["calendar"] = str(calendar)
    output = portfolio_statistics(
        {
            "inputs": inputs,
            "ledgers": paths,
            "output": str(tmp_path / "statistics"),
            "window": window,
            "analyses": ["group_comparison"],
            "statistics": {
                "hac_lag": 0,
                "bootstrap_block": 1,
                "bootstrap_repetitions": 10,
                "group_comparison": {
                    "portfolio_ids": ["ordinary"],
                    "primary_portfolio_id": "ordinary",
                    "fdr_method": "BY",
                    "alpha": 0.05,
                    "seed": 42,
                    "confidence": 0.95,
                },
            },
        }
    )
    with read_statistics_artifact(output).table(
        "portfolio_group_comparison.parquet"
    ) as rows:
        row = rows.iloc[0]
        means = row[["global_mean", "quarterly_mean", "mean_difference"]]
        assert means.to_numpy(dtype=float) == pytest.approx([-1.0, 0.1, 1.1])
        assert row.observation_count == 2
        assert row.fdr_family_size == 1
        assert row.status == row.bootstrap_status == "UNDEFINED"
        assert (
            row.reason
            == row.bootstrap_reason
            == ("nonpositive_start_nav" if with_calendar else "incomplete_daily_axis")
        )
        assert pd.isna(row.p_value) and pd.isna(row.bootstrap_lower)
        assert row.global_terminal_wealth == 0.0
        assert row.quarterly_terminal_wealth == pytest.approx(1.32)

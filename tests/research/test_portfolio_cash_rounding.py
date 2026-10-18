"""Many-fill cash conservation and native return publication use money units."""

from dataclasses import asdict, replace
from fractions import Fraction
from math import ulp

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_DECISION_REASON_SIMPLE,
    ExecutedTrade,
    PortfolioCandidateSpec,
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.research.portfolios.accounting import (
    portfolio_interval_economics,
)
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from tests.research.test_portfolio_production import _build_producer, _inputs


def _publish(tmp_path, batches, *, capital, candidate, window):
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital(candidate.candidate_id, capital, "CNY", "cash_equity"),
        metadata={"candidate": asdict(candidate), "window": window},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    assert list(read_portfolio_ledger_artifact(root).iter_chunks()) == batches
    output = portfolio_statistics(
        {
            "ledgers": [str(root)],
            "output": str(tmp_path / "statistics"),
            "window": window,
            "sample_role": "validation",
            "analyses": ["performance"],
        }
    )
    with read_statistics_artifact(output).table("portfolio_daily.parquet") as daily:
        return daily.copy()


@pytest.mark.parametrize("count", (100, 1000))
@pytest.mark.parametrize(
    "construction", ("quantile_equal_weight", "quantile_equal_weight_half_step")
)
def test_many_fills_conserve_cash_and_publish_native_returns(
    tmp_path, monkeypatch, count, construction
):
    frames, fees, window = _inputs()
    for key in ("panel", "dataset", "execution"):
        first = frames[key].loc[frames[key].asset_id.eq("a")]
        frames[key] = pd.concat(
            [first.assign(asset_id=f"stock_{i:04}") for i in range(count)],
            ignore_index=True,
        )
    frames["panel"][["open", "close", "adj_open", "adj_close"]] = 1.0
    frames["panel"]["amount"] = 1e10
    candidate = PortfolioCandidateSpec(
        "many_fills", "global", construction, "champion", "daily", 1.0, None
    )

    class Predictions:
        def values_for_frame(self, frame, *, mark_matched):
            return np.ones(len(frame)), bytearray([1] * len(frame))

        def coverage_counts(self):
            return 0, 0

    chunks, _ = _build_producer(
        tmp_path / "inputs",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        prediction_lookup=Predictions(),
        initial_nav=10_000_001.0,
        typed_batches=True,
    )
    batches = list(chunks)
    first = batches[0]
    assert len(first.trades) == count
    # No price change: the independent economic PnL is exactly the fee outflow.
    for batch in batches:
        before_cash = next(
            row.value
            for row in batch.valuations
            if row.asset_id == "__cash__" and row.phase == "before"
        )
        cash = Fraction(before_cash) + sum(
            (Fraction(flow.amount) for flow in batch.cashflows), Fraction()
        )
        actual_cash = next(
            row.value
            for row in batch.valuations
            if row.asset_id == "__cash__" and row.phase == "end"
        )
        # Existing weights and NAV each round when reconstructing pretrade cash;
        # the cashflow reduction must not add an error growing with fill count.
        assert abs(Fraction(actual_cash) - cash) <= 2 * Fraction(ulp(10_000_001.0))
        fees_paid = -sum(
            (
                Fraction(flow.amount)
                for flow in batch.cashflows
                if flow.kind.endswith("_cost")
            ),
            Fraction(),
        )
        asset_reads = 0
        original_getattribute = ExecutedTrade.__getattribute__

        def count_trade_assets(trade, name, getattribute=original_getattribute):
            nonlocal asset_reads
            asset_reads += name == "asset_id"
            return getattribute(trade, name)

        with monkeypatch.context() as observed:
            observed.setattr(ExecutedTrade, "__getattribute__", count_trade_assets)
            economic = portfolio_interval_economics(batch)
        assert asset_reads <= 2 * len(batch.trades)
        assert economic["net_return"] == float(
            -fees_paid / Fraction(economic["nav_before"])
        )
    economics_calls = []

    def observe_economics(batch):
        economics_calls.append(batch.sequence)
        return portfolio_interval_economics(batch)

    monkeypatch.setattr(
        "quant_research_platform.research.portfolios.diagnostics.portfolio_interval_economics",
        observe_economics,
    )
    daily = _publish(
        tmp_path,
        batches,
        capital=10_000_001.0,
        candidate=candidate,
        window={"start": str(window.start), "end": str(window.end)},
    )
    assert economics_calls == [batch.sequence for batch in batches]
    assert daily.return_status.eq("DEFINED").all()
    # One purchase of a constant-price asset pool costs 0.04% of bought notional.
    expected = -0.0002 if construction.endswith("half_step") else -0.0004 / 1.0004
    assert daily.net_return.iloc[0] == pytest.approx(expected, rel=2e-13)


def _held_batch(sequence, start, end):
    entry = pd.Timestamp("2024-01-02") + pd.Timedelta(days=sequence)
    values = []
    for phase, mark in (("before", start), ("after", start), ("end", end)):
        timestamp = (entry + pd.Timedelta(days=int(phase == "end"))).value
        values.extend(
            (
                PortfolioValuation(
                    timestamp,
                    "stock",
                    phase,
                    Fraction(1),
                    mark,
                    mark,
                    lifecycle_id="2000-01-01",
                ),
                PortfolioValuation(timestamp, "__cash__", phase, 0.0, 0.0),
            )
        )
    facts = tuple(
        PortfolioDiagnostic(asset, None, name, number=value)
        for asset, target in (("stock", start), ("__cash__", 0.0))
        for name, value in (("target_notional", target), ("decision_delta_weight", 0.0))
    ) + (
        PortfolioDiagnostic(
            "__cash__", None, "decision_reason", text=PORTFOLIO_DECISION_REASON_SIMPLE
        ),
    )
    return PortfolioLedgerBatch(sequence, (), (), tuple(values), facts)


@pytest.mark.parametrize("kind", ("zero_nav", "unrepresentable_return", "tiny_pnl"))
def test_native_return_states_keep_their_economic_meaning(tmp_path, kind):
    candidate = PortfolioCandidateSpec(
        "return_states",
        "global",
        "quantile_equal_weight",
        "champion",
        "daily",
        1.0,
        None,
    )
    if kind == "zero_nav":
        capital = 1.0
        batches = [_held_batch(0, 1.0, 0.0), _held_batch(1, 0.0, 2.0)]
    elif kind == "unrepresentable_return":
        capital = np.nextafter(0.0, 1.0).item()
        batches = [_held_batch(0, capital, 1.0)]
    else:
        capital = 1.0
        # The raw exact mark changes below one NAV ULP; value rounds to one.
        batch = _held_batch(0, 1.0, 1.0)
        batches = [
            replace(
                batch,
                valuations=tuple(
                    replace(row, mark_price=Fraction(1) + Fraction(1, 10**20))
                    if row.asset_id == "stock" and row.phase == "end"
                    else row
                    for row in batch.valuations
                ),
            )
        ]
    daily = _publish(
        tmp_path,
        batches,
        capital=capital,
        candidate=candidate,
        window={"start": "2024-01-01", "end": "2024-01-05"},
    )
    last = daily.iloc[-1]
    if kind == "tiny_pnl":
        assert last.return_status == "DEFINED" and last.return_reason == ""
        assert last.net_return == 1e-20
        assert last.nav_before == last.nav_end == 1.0
    else:
        assert last.return_status == "UNDEFINED" and pd.isna(last.net_return)
        assert last.return_reason == (
            "nonpositive_start_nav" if kind == "zero_nav" else "nonfinite_result"
        )

    # The persisted reader must still reject a fabricated finite return, whether
    # the real result is tiny, zero-denominator, or outside the float range.
    path = tmp_path / "statistics" / "portfolio_daily.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[-1].update(net_return=1e-8, return_status="DEFINED", return_reason="")
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="statistics_return_status_invalid"):
        with read_statistics_artifact(tmp_path / "statistics").table(
            "portfolio_daily.parquet"
        ):
            pass


def test_overflowing_trial_buy_spend_still_finds_finite_cash_funded_size(
    tmp_path, monkeypatch
):
    frames, fees, window = _inputs()
    fees["rate"] = np.where(fees.fee_code.eq("BROKER_COMMISSION"), 1.0, 0.0)
    frames["panel"][["open", "close", "adj_open", "adj_close"]] = 1.0
    frames["panel"]["amount"] = 1e308
    candidate = PortfolioCandidateSpec(
        "large_finite_cash",
        "global",
        "quantile_equal_weight",
        "champion",
        "daily",
        0.5,
        None,
    )
    chunks, _ = _build_producer(
        tmp_path / "inputs",
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        initial_nav=1e308,
        typed_batches=True,
    )
    batches = list(chunks)
    assert len(batches[0].trades) == 1
    # Requested stock plus 100% fee exceeds float range, but half is affordable.
    assert batches[0].trades[0].notional == 5e307
    assert sum(flow.amount for flow in batches[0].cashflows) == -1e308
    daily = _publish(
        tmp_path,
        batches,
        capital=1e308,
        candidate=candidate,
        window={"start": str(window.start), "end": str(window.end)},
    )
    assert daily.net_return.tolist() == [-0.5, 0.0]
    assert daily.cash_post.eq(0.0).all()

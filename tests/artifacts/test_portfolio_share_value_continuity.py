"""Raw-share monetary boundaries have one native writer/reader owner."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts import portfolio_ledger as ledger
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.contracts.validation import encode_exact_numeric


def _batch(sequence, *, a=30.0, b=30.0, receivable=0.0):
    return PortfolioLedgerBatch(
        sequence,
        (),
        (),
        tuple(
            row
            for phase in ("before", "after", "end")
            for row in (
                PortfolioValuation(sequence + 1, "__cash__", phase, 40.0, 40.0),
                PortfolioValuation(
                    sequence + 1,
                    "a",
                    phase,
                    Fraction(3),
                    a + receivable,
                    Fraction(a) / 3,
                    lifecycle_id="life",
                    receivable=receivable,
                ),
                PortfolioValuation(
                    sequence + 1,
                    "b",
                    phase,
                    Fraction(3),
                    b,
                    Fraction(b) / 3,
                    lifecycle_id="life",
                ),
            )
        ),
        (
            PortfolioDiagnostic("__cash__", None, "target_notional", number=40.0),
            PortfolioDiagnostic("a", None, "target_notional", number=a),
            PortfolioDiagnostic("b", None, "target_notional", number=b),
            PortfolioDiagnostic(
                "__cash__",
                None,
                "decision_reason",
                text=PORTFOLIO_DECISION_REASON_SIMPLE,
            ),
        ),
    )


def _write(path, batches, initial=100.0):
    return ledger.write_portfolio_ledger_artifact(
        path,
        capital=PortfolioCapital("continuity", initial, "CNY", "cash_equity"),
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )


@pytest.mark.parametrize("shift_b", (False, True))
def test_native_writer_rejects_changed_adjacent_raw_share_wealth(tmp_path, shift_b):
    # Keeping total NAV fixed does not authorize moving wealth between stocks.
    changed = _batch(1, a=31.0, b=29.0 if shift_b else 30.0)
    with pytest.raises(
        ValueError, match="portfolio_ledger_cross_session_value_invalid"
    ):
        _write(tmp_path / "invalid", [_batch(0), changed])
    assert not (tmp_path / "invalid").exists()


def test_native_reader_rejects_changed_adjacent_raw_share_wealth(tmp_path):
    root = _write(tmp_path / "valid", [_batch(0), _batch(1)])
    path = root / "valuations.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    # Change every phase coherently, bypassing neither per-row value identities
    # nor per-batch accounting; only the adjacent economic boundary is broken.
    for row in rows:
        if row["sequence"] == 1 and row["asset_id"] == "a":
            row["value"] = 31.0
            row["mark_price"] = encode_exact_numeric(Fraction(31, 3))
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    path = root / "diagnostics.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    for row in rows:
        if row["sequence"] == 1 and row["asset_id"] == "a":
            row["number"] = 31.0
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(
        ValueError, match="portfolio_ledger_cross_session_value_invalid"
    ):
        list(ledger.read_portfolio_ledger_artifact(root).iter_chunks())


def test_native_boundary_accepts_monetary_roundoff_and_unpaid_receivable(tmp_path):
    second = _batch(1, a=np.nextafter(30.0, np.inf), receivable=7.0)
    root = _write(tmp_path / "roundoff", [_batch(0, receivable=7.0), second], 107.0)
    batches = list(ledger.read_portfolio_ledger_artifact(root).iter_chunks())
    assert len(batches) == 2
    assert sum(
        row.number for row in batches[-1].diagnostics if row.name == "target_notional"
    ) == pytest.approx(100.0)
    assert (
        sum(row.receivable for row in batches[-1].valuations if row.phase == "end")
        == 7.0
    )


def test_native_boundary_does_not_use_monetary_tolerance_for_exact_shares(tmp_path):
    changed = _batch(1)
    quantity = Fraction(3) + Fraction(1, 10**20)
    changed = replace(
        changed,
        valuations=tuple(
            replace(row, quantity=quantity, mark_price=Fraction(30) / quantity)
            if row.asset_id == "a"
            else row
            for row in changed.valuations
        ),
    )
    with pytest.raises(
        ValueError, match="portfolio_ledger_cross_session_state_invalid"
    ):
        _write(tmp_path / "exact", [_batch(0), changed])


def test_cash_reserve_contract_preserves_gross_stock_budget(tmp_path):
    batch = _batch(0, receivable=7.0)
    batch = replace(
        batch,
        diagnostics=(
            PortfolioDiagnostic("__cash__", None, "target_notional", number=0.0),
            PortfolioDiagnostic("a", None, "target_notional", number=200.0),
            PortfolioDiagnostic("b", None, "target_notional", number=30.0),
            PortfolioDiagnostic("a", None, "block_reason", text="cash_pro_rata"),
            PortfolioDiagnostic(
                "__cash__",
                None,
                "decision_reason",
                text=PORTFOLIO_DECISION_REASON_SIMPLE,
            ),
        ),
    )
    with pytest.raises(ValueError, match="portfolio_ledger_target_state_invalid"):
        _write(tmp_path / "oversized", [batch], 107.0)

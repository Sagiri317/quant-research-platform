"""Raw-share monetary boundaries have one native writer/reader owner."""

import numpy as np
import pytest

from quant_research_platform.artifacts import portfolio_ledger as ledger
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_DECISION_REASON_SIMPLE,
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)


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
                    float(3),
                    a + receivable,
                    float(a) / 3,
                    lifecycle_id="life",
                    receivable=receivable,
                ),
                PortfolioValuation(
                    sequence + 1,
                    "b",
                    phase,
                    float(3),
                    b,
                    float(b) / 3,
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

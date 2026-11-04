"""Publication validates cross-batch identities without replaying child ledgers."""

from dataclasses import fields, replace

import pytest

from quant_research_platform.artifacts import portfolio_ledger
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioLedgerBatch,
    PortfolioRightsAdjustment,
    PortfolioValuation,
)


def _batch(sequence):
    timestamp = sequence + 1
    return PortfolioLedgerBatch(
        sequence,
        (),
        (),
        tuple(
            row
            for phase in ("before", "after", "end")
            for row in (
                PortfolioValuation(timestamp, "__cash__", phase, 100.0, 100.0),
                PortfolioValuation(
                    timestamp, "a", phase, 1.0, 0.0, 0.0, lifecycle_id="life"
                ),
            )
        ),
        (),
        (
            PortfolioRightsAdjustment(
                "event",
                timestamp,
                "a",
                "life",
                1.0,
                1.0,
                0.0,
                None,
                None,
                0.0,
                "a" * 64,
            ),
        ),
    )


@pytest.mark.parametrize("invalid", [True, None, float("nan"), float("inf"), "1.0"])
def test_rights_adjustment_rejects_invalid_numeric_fields(invalid):
    batch = _batch(0)
    event = batch.rights_adjustments[0]
    for field in fields(event):
        if field.type != "float":
            continue
        with pytest.raises(ValueError, match="portfolio_ledger_numeric_invalid"):
            replace(
                batch,
                rights_adjustments=(replace(event, **{field.name: invalid}),),
            )


def test_cross_batch_nontrade_ids_rejected_before_publication(tmp_path, monkeypatch):
    def no_replay(*args, **kwargs):
        raise AssertionError("writer must validate its canonical stream")

    monkeypatch.setattr(portfolio_ledger, "_iter_batches", no_replay)
    with pytest.raises(
        ValueError, match="portfolio_ledger_duplicate_rights_adjustment"
    ):
        portfolio_ledger.write_portfolio_ledger_artifact(
            tmp_path / "ledger",
            capital=PortfolioCapital("candidate", 100.0, "CNY", "cash_equity"),
            metadata={"window": {"start": "1970-01-01", "end": "1970-01-01"}},
            produce=lambda emit: [emit(_batch(i)) for i in range(2)],
        )
    assert not (tmp_path / "ledger").exists()

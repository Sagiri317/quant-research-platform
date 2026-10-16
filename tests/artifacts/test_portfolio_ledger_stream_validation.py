"""Publication validates cross-batch identities without replaying child ledgers."""

import pandas as pd
import pytest

from quant_research_platform.artifacts import crypto_hft, portfolio_ledger
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioLedgerBatch,
    PortfolioRightsAdjustment,
    PortfolioValuation,
)


def _batch(sequence, *, repeated):
    timestamp = sequence + 1
    return PortfolioLedgerBatch(
        sequence,
        (),
        (PortfolioCashFlow("flow", timestamp, "a", None, "funding", 0.0),)
        if repeated == "cashflow"
        else (),
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
        )
        if repeated == "rights_adjustment"
        else (),
    )


@pytest.mark.parametrize("kind", ["cashflow", "rights_adjustment"])
def test_cross_batch_nontrade_ids_rejected_before_publication(
    tmp_path, monkeypatch, kind
):
    def no_replay(*args, **kwargs):
        raise AssertionError("writer must validate its canonical stream")

    monkeypatch.setattr(portfolio_ledger, "_iter_batches", no_replay)
    with pytest.raises(ValueError, match=f"portfolio_ledger_duplicate_{kind}"):
        portfolio_ledger.write_portfolio_ledger_artifact(
            tmp_path / "ledger",
            capital=PortfolioCapital("candidate", 100.0, "USDT", "linear_perpetual"),
            metadata={"window": {"start": "1970-01-01", "end": "1970-01-01"}},
            produce=lambda emit: [emit(_batch(i, repeated=kind)) for i in range(2)],
        )
    assert not (tmp_path / "ledger").exists()


def test_independent_episode_publication_does_not_read_back_children(
    tmp_path, monkeypatch
):
    def no_replay(*args, **kwargs):
        raise AssertionError("publication must not replay canonical child batches")

    with monkeypatch.context() as patch:
        patch.setattr(portfolio_ledger, "_iter_batches", no_replay)
        patch.setattr(crypto_hft, "_episode_rows", no_replay)

        def produce(emit):
            for index in range(2):
                start, stop = index * 2, index * 2 + 2
                emit(
                    start_ns=start,
                    stop_ns=stop,
                    termination="PLANNED_END" if index == 1 else "TRUNCATED",
                    stop_reasons=("planned_oos_end",)
                    if index == 1
                    else ("archive_end",),
                    produce=lambda emit, index=index: emit(
                        _batch(0, repeated=None)
                        if index == 0
                        else PortfolioLedgerBatch(
                            0,
                            (),
                            (),
                            tuple(
                                PortfolioValuation(3, "__cash__", phase, 100.0, 100.0)
                                for phase in ("before", "after", "end")
                            ),
                            (),
                        )
                    ),
                )

        root = crypto_hft.write_crypto_episode_artifact(
            tmp_path / "episodes",
            capital=PortfolioCapital("candidate", 100.0, "USDT", "linear_perpetual"),
            metadata={
                "window": {
                    "start": pd.Timestamp(0).isoformat(),
                    "end": pd.Timestamp(3).isoformat(),
                }
            },
            produce=produce,
        )
    artifact = crypto_hft.read_crypto_episode_artifact(root)
    assert len(list(artifact.iter_episodes())) == 2

"""Native Crypto ledgers retain independent and continuous economic semantics."""

import pandas as pd
import pytest

from quant_research_platform.artifacts import crypto_hft, portfolio_ledger
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
)
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics


@pytest.mark.parametrize("mode", ("continuous", "independent_segments"))
def test_crypto_native_writer_reader_economic_consumption(tmp_path, mode):
    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
    metadata = {
        "window": {
            "start": pd.Timestamp(0).isoformat(),
            "end": pd.Timestamp(30_000_000).isoformat(),
        },
        "execution_mode": mode,
    }

    def produce(emit):
        account = PortfolioEconomics(capital, "btc_usdt", emit)
        account.mark(0, 100.0)
        account.fill(
            timestamp_ns=10_000_000,
            order_id="one",
            quantity=1.0,
            price=100.0,
            mark_price=100.0,
            fee=1.0,
        )
        account.mark(
            20_000_000,
            110.0,
            diagnostics=(
                PortfolioDiagnostic("btc_usdt", None, "execution_summary", text="{}"),
            ),
        )

    summaries = []
    if mode == "continuous":
        root = portfolio_ledger.write_portfolio_ledger_artifact(
            tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
        )
        portfolio_ledger.read_portfolio_ledger_artifact(
            root,
            consume_batches=lambda artifact, batches: summaries.append(
                crypto_hft.summarize_crypto_ledger(artifact, batches)
            ),
        )
    else:
        root = crypto_hft.write_crypto_episode_artifact(
            tmp_path / "ledger",
            capital=capital,
            metadata=metadata,
            produce=lambda emit: emit(
                start_ns=0,
                stop_ns=30_000_001,
                termination="PLANNED_END",
                stop_reasons=("planned_oos_end",),
                produce=produce,
            ),
        )
        crypto_hft.read_crypto_episode_artifact(
            root,
            consume_episode=lambda row, artifact, batches: summaries.append(
                crypto_hft.summarize_crypto_ledger(artifact, batches)
            ),
        )
    assert len(summaries) == 1
    assert summaries[0]["economics"]["net_pnl"] == 9.0
    assert summaries[0]["economics"]["fees"] == 1.0
    assert root == tmp_path / "ledger"

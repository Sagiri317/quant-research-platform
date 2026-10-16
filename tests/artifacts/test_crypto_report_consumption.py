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


@pytest.mark.parametrize("damage", ["missing", "invalid"])
def test_basic_ledger_consumption_does_not_read_optional_final_calibration(
    tmp_path, damage
):
    from quant_research_platform.interfaces.queries import portfolio_curve
    from quant_research_platform.reports.plots import plot

    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")

    def produce(emit):
        account = PortfolioEconomics(capital, "btc_usdt", emit)
        account.mark(0, 100.0)
        account.mark(10_000_000, 110.0)

    root = portfolio_ledger.write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=produce,
        final_calibration=lambda: {
            "schema_id": "qrp-test-calibration",
            "status": "UNDEFINED",
        },
    )
    artifact = portfolio_ledger.read_portfolio_ledger_artifact(root)
    assert artifact.final_alpha_calibration["status"] == "UNDEFINED"
    calibration = root / "final_alpha_calibration.json"
    if damage == "missing":
        calibration.unlink()
    else:
        calibration.write_text("invalid json", encoding="utf-8")
    artifact = portfolio_ledger.read_portfolio_ledger_artifact(root)
    summary = portfolio_ledger.summarize_portfolio_ledger(
        artifact.capital, artifact.iter_chunks()
    )
    assert summary["net_pnl"] == 0.0
    assert portfolio_curve(root)["status"] == "DEFINED"
    assert (
        plot(
            {
                "kind": "curve",
                "inputs": str(root),
                "output": str(tmp_path / "curve.png"),
                "timezone": "UTC",
            }
        )
        .stat()
        .st_size
        > 1000
    )
    with pytest.raises(ValueError):
        _ = artifact.final_alpha_calibration
    (root / "valuations.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        portfolio_curve(root)

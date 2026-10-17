"""Public execution retains sub-millisecond facts and exact source cutoffs."""

import json
from dataclasses import asdict

import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.markets.implementations.crypto import hft, portfolio
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import iter_agg_trades
from quant_research_platform.sources.market import open_market_source
from tests.helpers.binance_hft import write_binance_hft_run, write_hft_funding_fixture
from tests.markets.crypto.test_independent_execution_source import _mutate, _panel


@pytest.mark.parametrize("stamp", [100_500_000, 100_900_000, 100_900_001])
@pytest.mark.parametrize("unknown", [False, True])
def test_public_maker_includes_exact_tail_trade_and_excludes_future(
    tmp_path, stamp, unknown
):
    raw = tmp_path / "source" / "raw"
    write_binance_hft_run(raw, count=3, maker_liquidity=True)
    write_hft_funding_fixture(raw)
    window = {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:00.100900"}
    panel = _panel(tmp_path / "panel", raw, window)

    def change(row, member):
        if (
            member["symbol"] == "BTCUSDT"
            and member["data_kind"] == "agg_trades"
            and row["payload"]["a"] == 1
        ):
            row.update(
                observed_time_ns=stamp,
                receive_time=pd.Timestamp(stamp, unit="ns").isoformat() + "Z",
            )
            row["payload"].update(p="99.5", m=True)
            if unknown:
                row["payload"].pop("nq")

    manifest = _mutate(raw, change)
    (raw / "run.json").write_text(json.dumps(manifest))
    inside = stamp <= 100_900_000
    with open_market_source(raw) as source:
        events = list(
            iter_agg_trades(
                source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", end_ns=100_900_000
            )
        )
        assert [event.observed_time_ns for event in events] == (
            [stamp] if inside else []
        )
    result = portfolio.backtest_maker(
        {
            "inputs": str(panel),
            "panel": str(panel),
            "source": str(raw),
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
            "prediction_window": window,
            "execution_mode": "continuous",
            "capital": asdict(
                PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(
                portfolio.MakerConfig(
                    order_quantity=1.0,
                    max_position=2.0,
                    maker_fee_bps=0.0,
                    inventory_target=2.0,
                    inventory_skew=1.0,
                    markout_horizon_ms=100,
                    latency_ms=0,
                )
            ),
            "models": [],
            "calibration": None,
            "output": str(tmp_path / "ledger"),
        }
    )
    ledger = read_portfolio_ledger_artifact(result)
    summary = summarize_crypto_ledger(ledger, ledger.iter_chunks())
    partial = inside and unknown
    assert summary["execution"]["account_path_status"] == (
        "PARTIAL" if partial else "COMPLETE"
    )
    assert summary["execution"]["filled_orders"] == int(inside and not unknown)
    if partial:
        assert summary["execution"]["account_stop_ns"] == stamp
        assert (
            summary["execution"]["account_stop_reason"]
            == "hft_trade_visible_quantity_unknown"
        )
        assert summary["economics"]["net_pnl"] is None


def test_return_only_factors_do_not_construct_unused_exact_arithmetic(monkeypatch):
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp_ms": [0, 100, 200],
            "segment_id": "one",
            "mid": [100.0, 101.0, 102.0],
        }
    )

    def unused(*_args, **_kwargs):
        pytest.fail("return-only factors must not construct unused Fractions")

    monkeypatch.setattr(hft, "Fraction", unused)
    result = pd.concat(
        hft.iter_factor_frames([frame], columns=("return_100ms",)), ignore_index=True
    )
    assert pd.isna(result.return_100ms.iloc[0])
    assert result.return_100ms.iloc[1:].tolist() == [0.01, 1 / 101]


def test_public_calibration_uses_exact_maximum_maturity_cutoff(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.sources.binance.replay import iter_reconstructed_depth
    from tests.markets.crypto.test_direct_portfolio import _independent_maker_case

    config, _backtest = _independent_maker_case(tmp_path, monkeypatch)
    raw = tmp_path / "source" / "raw"
    write_binance_hft_run(raw, count=30, maker_liquidity=True)
    cutoff = 2_700_900_000
    tail = cutoff - 400_000

    def change(row, member):
        if (
            member["symbol"] == "BTCUSDT"
            and member["data_kind"] == "agg_trades"
            and row["payload"]["a"] == 27
        ):
            row.update(
                observed_time_ns=tail,
                receive_time=pd.Timestamp(tail, unit="ns").isoformat() + "Z",
            )

    manifest = _mutate(raw, change)
    (raw / "run.json").write_text(json.dumps(manifest))
    config["source"] = str(raw)
    config["panel"] = str(
        _panel(
            tmp_path / "native-panel",
            raw,
            {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:03"},
        )
    )
    config["models"][0]["cutoff"] = pd.Timestamp(cutoff, unit="ns").isoformat()
    seen = []

    @contextmanager
    def open_source(root, *, archive_paths=None, time_window=None):
        assert pd.Timestamp(time_window.start).value == 0
        assert pd.Timestamp(time_window.end).value == cutoff
        with open_market_source(
            root, archive_paths=archive_paths, time_window=time_window
        ) as source:
            yield source

    def trades(source, market_type, *, symbol, end_ns):
        assert end_ns == cutoff
        for event in iter_agg_trades(source, market_type, symbol=symbol, end_ns=end_ns):
            seen.append(event.observed_time_ns)
            yield event

    monkeypatch.setattr(portfolio, "open_market_source", open_source)
    monkeypatch.setattr(portfolio, "iter_reconstructed_depth", iter_reconstructed_depth)
    monkeypatch.setattr(portfolio, "iter_agg_trades", trades)
    output = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.tolist() == [2, 2]
    assert tail in seen
    assert max(seen) == tail

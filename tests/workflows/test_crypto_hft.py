"""Crypto scientific regressions through explicit datasets and native ledgers.

The former research DAG fixture is retired. This file keeps its uncovered local
science; the detailed old-to-new assertion map lives with migration evidence.
"""

import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.markets.implementations.crypto.execution import (
    simulate_maker,
    simulate_taker,
)
from quant_research_platform.research.protocols.crypto_hft import research_chronology
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.helpers.binance_hft import KNOWN_NO_FUNDING, _quote
from tests.markets.crypto.calibration_test_support import calibration_from_paths
from tests.markets.crypto.test_direct_portfolio import _market, _window
from tests.markets.crypto.test_hft_execution import (
    _maker_config,
    _state,
    _taker_config,
    _trade,
)


def _inputs(root, stamps, *, assets=None, x=None, eligible=None):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(stamps, unit="ms"),
            "asset_id": assets or ["btc_usdt"] * len(stamps),
            "x": x or [1.0] * len(stamps),
            "eligible": eligible or [True] * len(stamps),
            "segment_id": "segment",
            "book_segment_id": "s1",
            "trade_observation_id": "trade-connection-1",
        }
    )
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", x="feature", eligible="mask")
    write_dataset(
        root,
        frame,
        spec=DataSpec(
            data_id="crypto_regression",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
        ),
        metadata={
            "source_evidence": {
                "market_metadata": {
                    symbol: {"price_unit": "0.1", "snapshot_limit": 10}
                    for symbol in ("BTCUSDT", "ETHUSDT")
                }
            },
            "producer_diagnostics": {
                "reconstruction": {
                    symbol: {"coverage_segments": {"s1": [0, 10_000_000_000]}}
                    for symbol in ("BTCUSDT", "ETHUSDT")
                }
            },
        },
    )
    return frame


def test_hft_prediction_alignment_consumes_saved_keys_and_requires_context(
    tmp_path, monkeypatch
):
    root = tmp_path / "inputs"
    frame = _inputs(
        root,
        [100, 200, 300],
        x=[1.0, float("nan"), 2.0],
        eligible=[True, True, False],
    )
    model = {
        "model_id": "price",
        "model_algorithm": "linear_regression",
        "input_feature_columns": ["x"],
        "target_column": "future_mid_return_500ms",
        "sample_filter_mode": "eligible",
    }
    metadata = {
        "training_asset_ids": ["btc_usdt"],
        "model_algorithm": "linear_regression",
        "input_columns": ["x"],
        "sample_filter_mode": "eligible",
    }
    prediction = frame.loc[[0], ["timestamp", "asset_id"]].assign(prediction_score=0.5)

    def rows(predictions):
        return list(
            portfolio._prediction_rows(
                inputs=root,
                evidence=(metadata, {"model": model}, predictions),
                window=TimeWindow(**_window(100, 300)),
                fold_id="fold",
                asset_id="btc_usdt",
            )
        )

    assert rows(prediction.iloc[:0]) == []
    assert [row["prediction_score"] for row in rows(prediction)] == [0.5]
    # Current feature NaNs and eligibility do not re-admit saved forecasts;
    # context rows without a saved forecast do not require a new prediction.
    for index in (1, 2):
        extra = frame.loc[[index], ["timestamp", "asset_id"]].assign(
            prediction_score=0.7
        )
        result = rows(pd.concat((extra, prediction), ignore_index=True))
        assert [row["timestamp_ms"] for row in result] == [100, (index + 1) * 100]
        assert [row["prediction_score"] for row in result] == [0.5, 0.7]
        assert all(row["book_segment_id"] == "s1" for row in result)

    missing_context = prediction.assign(timestamp=pd.Timestamp(250, unit="ms"))
    with pytest.raises(ValueError, match="^misaligned_persisted_prediction_keys$"):
        rows(missing_context)

    chunks = portfolio.iter_dataset_chunks

    def duplicate_context(session):
        for chunk in chunks(session):
            yield chunk
            yield chunk.iloc[:1]

    monkeypatch.setattr(portfolio, "iter_dataset_chunks", duplicate_context)
    with pytest.raises(ValueError, match="^duplicate_prediction_input_keys$"):
        rows(prediction)


def test_crypto_chronology_preserves_multi_day_noncalendar_boundaries():
    chronology = {
        "folds": [
            {
                "fold_id": "fold",
                "train": {
                    "start": "2026-08-01T13:17:00",
                    "end": "2026-09-02T11:29:00",
                },
                "validation": {
                    "start": "2026-09-03T01:02:00",
                    "end": "2026-09-05T05:06:00",
                },
            }
        ],
        "test": {"start": "2026-09-09T11:12:00", "end": "2026-09-15T13:14:00"},
    }
    folds, test = research_chronology(chronology)
    assert folds[0].train == TimeWindow(**chronology["folds"][0]["train"])
    assert folds[0].validation == TimeWindow(**chronology["folds"][0]["validation"])
    assert test == TimeWindow(**chronology["test"])


@pytest.mark.parametrize("calibrated", (False, True))
def test_future_gap_keeps_native_maker_decision_prefix(tmp_path, calibrated):
    capital = PortfolioCapital("maker", 1000.0, "USDT", "linear_perpetual")
    model = calibration_from_paths(
        [([_state(0)], [_trade(50, price="99.5", quantity="3", maker_buyer=True)])],
        [0.0],
    )
    results = []
    for stop in (400, 1000):

        def produce(emit, stop=stop):
            simulate_maker(
                [_state(t) for t in (0, 100, 200, 350, 390, 500, 900)],
                [
                    _trade(t, price="99.5", quantity="3", maker_buyer=True)
                    for t in (50, 150, 380, 600)
                ],
                [
                    {
                        **_quote(t, book_segment_id="s1"),
                        "fold_id": "fold",
                        "price_prediction": 0.0,
                        "proxy_prediction": 1.0,
                    }
                    for t in (0, 100, 350)
                ],
                config=replace(_maker_config(calibrated=calibrated), latency_ms=100),
                observed_coverage={"s1": (0, stop * 1_000_000)},
                trade_coverage={"trade-connection-1": (0, stop * 1_000_000)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=KNOWN_NO_FUNDING,
                calibrations={"fold": model} if calibrated else None,
                decision_windows_ns=((0, 900_000_000),),
                observation_window_ns=(0, stop * 1_000_000),
                on_ledger=emit,
            )

        root = write_portfolio_ledger_artifact(
            tmp_path / str(stop),
            capital=capital,
            metadata={"window": _window(0, 1000)},
            produce=produce,
        )
        artifact = read_portfolio_ledger_artifact(root)
        batches = list(artifact.iter_chunks())
        decisions = [
            (max(v.timestamp_ns for v in batch.valuations), row)
            for batch in batches
            if max((v.timestamp_ns for v in batch.valuations), default=0) <= 350_000_000
            for row in batch.diagnostics
            if row.name.startswith("maker_")
        ]
        summaries = [
            json.loads(row.text)
            for batch in batches
            for row in batch.diagnostics
            if row.name == "execution_summary"
        ]
        assert len(summaries) == 1
        assert "cancellations do not advance queue" in summaries[0]["queue_assumption"]
        results.append(decisions)
    assert results[0] and results[0] == results[1]


def test_native_taker_latency_changes_fills_and_is_reported(tmp_path):
    capital = PortfolioCapital("taker", 1000.0, "USDT", "linear_perpetual")
    fills = []
    for latency in (0, 100, 200):

        def produce(emit, latency=latency):
            simulate_taker(
                [_state(t) for t in (0, 100, 200, 300)],
                [{"timestamp_ms": 0, "prediction": 0.1}],
                config=_taker_config(latency_ms=latency),
                prediction_column="prediction",
                prediction_horizon_ms=500,
                observed_coverage={"s1": (0, 301_000_000)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=KNOWN_NO_FUNDING,
                on_ledger=emit,
            )

        root = write_portfolio_ledger_artifact(
            tmp_path / str(latency),
            capital=capital,
            metadata={"window": _window(0, 300)},
            produce=produce,
        )
        artifact = read_portfolio_ledger_artifact(root)
        batches = list(artifact.iter_chunks())
        summary = summarize_crypto_ledger(artifact, iter(batches))
        assert summary["execution"]["latency_ms"] == latency
        assert summary["execution"]["market_event_count"] > 0
        fills.append(
            tuple(
                (r.timestamp_ns, r.quantity, r.notional)
                for b in batches
                for r in b.trades
            )
        )
    assert all(fills) and len(set(fills)) == 3


@pytest.mark.parametrize("exposed", [False, True])
def test_funding_unknown_is_asset_scoped_and_keeps_old_good(
    tmp_path, monkeypatch, exposed
):
    root = tmp_path / "inputs"
    _inputs(root, [8600, 8600, 9000, 9000], assets=["btc_usdt", "eth_usdt"] * 2)
    _market(monkeypatch)
    source_trades = portfolio.iter_agg_trades
    monkeypatch.setattr(
        portfolio,
        "iter_agg_trades",
        lambda *args, **kwargs: (
            iter(())
            if kwargs["symbol"] == "ETHUSDT" and not exposed
            else source_trades(*args, **kwargs)
        ),
    )

    def funding(_roots, *, symbol):
        return nullcontext(
            SimpleNamespace(
                coverage=((0, 10_000_000_000),) if symbol == "BTCUSDT" else (),
                events=(),
                forecasts=KNOWN_NO_FUNDING,
                settlement_qualification=(),
            )
        )

    monkeypatch.setattr(portfolio, "read_funding_observations", funding)
    config = {
        "inputs": str(root),
        "panel": str(root),
        "source": "source",
        "prediction_window": _window(8600, 9000),
        "execution_mode": "continuous",
        "capital": asdict(
            PortfolioCapital("maker", 1000.0, "USDT", "linear_perpetual")
        ),
        "execution": asdict(_maker_config()),
        "models": [],
        "calibration": None,
    }
    btc = portfolio.backtest_maker(
        {
            **config,
            "output": str(tmp_path / "btc"),
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
        }
    )
    saved = {p.relative_to(btc): p.read_bytes() for p in btc.rglob("*") if p.is_file()}
    btc_artifact = read_portfolio_ledger_artifact(btc)
    btc_summary = summarize_crypto_ledger(btc_artifact, btc_artifact.iter_chunks())
    assert btc_summary["execution"]["account_path_status"] == "COMPLETE"
    assert btc_summary["economics"]["fill_events"] == 1
    eth = portfolio.backtest_maker(
        {
            **config,
            "output": str(tmp_path / "eth"),
            "asset_id": "eth_usdt",
            "symbol": "ETHUSDT",
        }
    )
    artifact = read_portfolio_ledger_artifact(eth)
    batches = list(artifact.iter_chunks())
    summary = summarize_crypto_ledger(artifact, iter(batches))
    execution, economics = summary["execution"], summary["economics"]
    fills = [fill for batch in batches for fill in batch.trades]
    if exposed:
        assert [
            (fill.asset_id, fill.timestamp_ns, fill.quantity) for fill in fills
        ] == [("eth_usdt", 8_700_000_000, 1)]
        assert execution["account_path_status"] == "PARTIAL"
        assert (
            execution["account_stop_reason"]
            == "portfolio_required_funding_coverage_missing"
        )
        assert execution["account_stop_ns"] == 8_700_000_001
        assert execution["known_prefix_valuation_timestamp_ns"] == 8_700_000_000
        assert execution["censored_active_order_count"] > 0
        assert economics["observed_totals_scope"] == "known_prefix"
        assert economics["net_pnl"] is economics["nav"] is economics["funding"] is None
    else:
        # Certified empty trade flow leaves the account flat throughout the window.
        assert not fills
        assert execution["account_path_status"] == "COMPLETE"
        assert execution["account_stop_reason"] is None
        assert economics["net_pnl"] == economics["funding"] == 0.0
        assert economics["nav"] == 1000.0
    assert saved == {
        p.relative_to(btc): p.read_bytes() for p in btc.rglob("*") if p.is_file()
    }
    # A local unknown account path does not poison later explicit asset operations.
    assert portfolio.backtest_maker(
        {
            **config,
            "output": str(tmp_path / "btc-again"),
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
        }
    ).exists()


def test_crypto_consumption_respects_explicit_prediction_asset_domain(tmp_path):
    root = tmp_path / "two-assets"
    frame = _inputs(root, [100, 100], assets=["btc_usdt", "eth_usdt"])
    metadata = {
        "training_asset_ids": ["btc_usdt", "eth_usdt"],
        "model_algorithm": "linear_regression",
        "input_columns": ["x"],
        "sample_filter_mode": "eligible",
    }
    facts = {
        "model": {
            "model_id": "price",
            "model_algorithm": "linear_regression",
            "input_feature_columns": ["x"],
            "target_column": "future_mid_return_500ms",
        },
        "prediction_asset_ids": ["btc_usdt"],
    }
    predictions = frame.loc[
        frame.asset_id.eq("btc_usdt"), ["timestamp", "asset_id"]
    ].assign(prediction_score=0.5)

    def rows(asset):
        return list(
            portfolio._prediction_rows(
                inputs=root,
                evidence=(metadata, facts, predictions),
                window=TimeWindow(**_window(100, 100)),
                fold_id="fold",
                asset_id=asset,
            )
        )

    assert [row["asset_id"] for row in rows("btc_usdt")] == ["btc_usdt"]
    with pytest.raises(ValueError, match="asset_outside_scope"):
        rows("eth_usdt")

    # A real explicit forecast domain may include assets outside the training set.
    facts["prediction_asset_ids"] = ["btc_usdt", "eth_usdt"]
    metadata["training_asset_ids"] = ["btc_usdt"]
    predictions = frame.loc[:, ["timestamp", "asset_id"]].assign(prediction_score=0.5)
    assert [row["asset_id"] for row in rows("eth_usdt")] == ["eth_usdt"]

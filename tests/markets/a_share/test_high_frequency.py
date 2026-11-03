import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.markets.implementations.a_share.high_frequency import (
    build_panel,
    compute_factors,
    compute_labels,
    continuous_session,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.protocols.a_share_hft import label_catalog
from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.sources.venvoo import (
    STREAM_FILES,
    normalize,
    source_columns,
)
from quant_research_platform.storage.datasets.reading import read_dataset

INSTRUMENTS = [{"wind_code": "000001.SZ", "asset_id": "000001.SZ", "exchange": "SZSE"}]


def raw_quotes(times=(93000000, 93003000, 93006000, 93009000)):
    result = pd.DataFrame(
        {
            "wind_code": "000001.sz",
            "date": 20180214,
            "time": list(times),
            "price": np.arange(len(times)) * 100 + 100000.0,
            "volume": 100.0,
        }
    )
    for level in range(1, 11):
        result[f"bid_px{level}"] = result.price - level * 100
        result[f"ask_px{level}"] = result.price + level * 100
        result[f"bid_vol{level}"] = 300.0
        result[f"ask_vol{level}"] = 100.0
    return result


def raw_events(stream):
    frame = raw_quotes((93000000, 93002000, 93005000, 93010000))
    frame["volume"] = [100.0, 200.0, 300.0, 100.0]
    for column in source_columns(stream):
        if column not in frame:
            frame[column] = "0" if "code" in column or "type" in column else 1
    if stream == "hft_orders":
        frame["order_code"] = "B"
    else:
        frame["bs_flag"] = "B"
        frame.loc[1, "trade_code"] = "C"
    return frame.loc[:, source_columns(stream)]


def test_source_units_cancellation_and_snapshot_math():
    quotes = normalize(raw_quotes(), "hft_quotes", INSTRUMENTS)
    trades = normalize(raw_events("hft_trades"), "hft_trades", INSTRUMENTS)
    orders = normalize(raw_events("hft_orders"), "hft_orders", INSTRUMENTS)
    assert quotes.timestamp.iloc[0] == pd.Timestamp("2018-02-14 01:30:00")
    assert quotes.bid_px1.iloc[0] == 9.99
    assert trades.event_kind.tolist() == ["trade", "cancel", "trade", "trade"]
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    factors = compute_factors(panel)
    labels = compute_labels(panel, label_catalog().list())
    assert np.isnan(panel.observed_trade_volume.iloc[0])
    assert panel.observed_trade_volume.iloc[1] == 0
    assert panel.observed_cancel_volume.iloc[1] == 200
    assert panel.observed_trade_volume.iloc[2] == 300
    np.testing.assert_allclose(factors.imbalance_10, 0.5)
    np.testing.assert_allclose(factors.relative_spread.iloc[0], 0.02 / 10)
    np.testing.assert_allclose(factors.microprice_deviation.iloc[0], 0.005 / 10)
    np.testing.assert_allclose(labels.future_mid_return_3s.iloc[0], 10.01 / 10 - 1)
    assert labels.future_mid_return_3s_available_at.iloc[0] == quotes.timestamp.iloc[1]
    assert pd.isna(labels.future_mid_return_3s.iloc[-1])
    np.testing.assert_allclose(factors.book_ofi.iloc[1:], 400.0)
    prefix = trades.loc[trades.timestamp.le(quotes.timestamp.iloc[1])]
    historical = build_panel(quotes.iloc[:2], orders, prefix, max_gap_seconds=6)
    pd.testing.assert_series_equal(
        historical.observed_trade_volume,
        panel.observed_trade_volume.iloc[:2],
    )
    from quant_research_platform.research.protocols.a_share_hft import processed_specs

    units = next(
        item.column_units for item in processed_specs() if item.data_id == "hft_factors"
    )
    assert units["realized_volatility_60s"] == "dimensionless"
    assert units["observed_trade_volume_30s"] == "shares"


def test_sessions_gaps_and_missing_order_history_are_not_zero():
    times = (112957000, 130000000, 130003000, 130030000)
    quotes = normalize(raw_quotes(times), "hft_quotes", INSTRUMENTS)
    orders = normalize(raw_events("hft_orders").iloc[:0], "hft_orders", INSTRUMENTS)
    trades = normalize(raw_events("hft_trades"), "hft_trades", INSTRUMENTS)
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    labels = compute_labels(panel, label_catalog().list())
    assert panel.segment_id.nunique() == 3
    assert panel.observed_order_volume.isna().all()
    assert labels.future_mid_return_3s.isna().tolist() == [True, False, True, True]
    local = pd.Series(pd.to_datetime(["2018-08-17 06:58:00", "2018-08-20 06:58:00"]))
    assert continuous_session(local, pd.Series(["SSE", "SSE"])).tolist() == ["pm", ""]


def test_native_source_to_assembled_hft_dataset(tmp_path):
    raw = tmp_path / "raw" / "20180214"
    raw.mkdir(parents=True)
    for stream, name in STREAM_FILES.items():
        frame = raw_quotes() if stream == "hft_quotes" else raw_events(stream)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), raw / name)
    outputs = {stream: str(tmp_path / stream) for stream in STREAM_FILES}
    result = prepare_source(
        {
            "source_id": "venvoo",
            "raw_root": str(raw.parent),
            "outputs": outputs,
            "instruments": INSTRUMENTS,
            "revision": "a" * 40,
            "start_date": "2018-02-14",
            "end_date": "2018-02-14",
        }
    )
    assert set(result.artifact_roots) == set(STREAM_FILES)
    window = {"start": "2018-02-14", "end": "2018-02-14"}
    inputs = {}
    for name in ("hft_panel", "hft_factors", "hft_labels"):
        inputs[name] = str(
            prepare(
                {
                    "market_id": "a_share",
                    "data_id": name,
                    "time_window": window,
                    "inputs": outputs
                    if name == "hft_panel"
                    else {"hft_panel": inputs["hft_panel"]},
                    "output_root": str(tmp_path / name),
                }
            )
        )
    path = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "a_share_hft",
            "inputs": inputs,
            "time_window": window,
            "output_root": str(tmp_path / "research"),
            "factor_columns": ["relative_spread", "imbalance_1"],
            "label_columns": ["future_mid_return_3s"],
        }
    )
    result = read_dataset(path)
    assert len(result) == 4
    assert result.future_mid_return_3s.notna().sum() == 3
    assert result.eligible.all()

    from quant_research_platform.markets.implementations.a_share.hft_portfolio import (
        backtest,
    )
    from quant_research_platform.research.evaluations.runner import evaluate
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.sources.reference.interim import fee_schedule_spec
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.manifests import read_qrp_json_file

    model = fit(
        {
            "inputs": str(path),
            "output": str(tmp_path / "model"),
            "market_id": "a_share",
            "model": {
                "model_id": "ols",
                "model_algorithm": "linear_regression",
                "input_factor_columns": ["relative_spread"],
                "target_column": "future_mid_return_3s",
            },
            "fit_windows": [
                {"start": "2018-02-14T01:30:00", "end": "2018-02-14T01:30:03"}
            ],
            "fit_cutoff": "2018-02-14T01:30:03",
        }
    )
    forecast_window = {"start": "2018-02-14T01:30:06", "end": "2018-02-14T01:30:09"}
    predictions = predict(
        {
            "inputs": str(path),
            "model": str(model),
            "output": str(tmp_path / "predictions"),
            "prediction_window": forecast_window,
        }
    )
    evaluate(
        {
            "inputs": str(path),
            "output": str(tmp_path / "evaluation"),
            "window": forecast_window,
            "cutoff": forecast_window["end"],
            "predictions": {"score": str(predictions)},
            "evaluation": {
                "evaluation_id": "hft",
                "market_id": "a_share",
                "research_dataset_id": "a_share_hft",
                "signals": [
                    {
                        "signal_id": "score",
                        "source_kind": "prediction",
                        "source_column": "score",
                    }
                ],
                "label_columns": ["future_mid_return_3s"],
                "evaluation_axis": "single_asset_time_series",
            },
        }
    )
    fees = pd.DataFrame(
        [
            {
                "fee_code": code,
                "effective_date": pd.Timestamp("2018-01-01"),
                "end_date": pd.Timestamp("2018-12-31"),
                "market": "a_share",
                "side": side,
                "basis": "amount",
                "rate": rate,
                "included_in_ledger": True,
                "included_in": None,
            }
            for side in ("buy", "sell")
            for code, rate in (
                ("BROKER_COMMISSION", 0.0003),
                ("TRANSFER_FEE", 0.00002),
                *((("STAMP_DUTY", 0.001),) if side == "sell" else ()),
            )
        ]
    )
    fees["included_in"] = fees.included_in.astype("string")
    write_dataset(tmp_path / "fees", fees, spec=fee_schedule_spec())
    destination = backtest(
        {
            "panel": inputs["hft_panel"],
            "predictions": str(predictions),
            "fee_schedule": str(tmp_path / "fees"),
            "window": forecast_window,
            "output": str(tmp_path / "accounts"),
            "parameters": {
                "initial_capital": 10000.0,
                "initial_inventory": 100,
                "maximum_inventory": 200,
                "lot_size": 100,
                "buy_threshold": 0.0,
                "sell_threshold": -1.0,
                "latency_ms": 0,
                "participation_rate": 1.0,
                "minimum_commission": 5.0,
            },
        }
    )
    _, summary = read_qrp_json_file(destination / "summary.json")
    assert len(summary["accounts"]) == 1
    assert summary["accounts"][0]["fill_events"] == 1


@pytest.mark.parametrize("bad_time", [246000000, -1, 93000000.5])
def test_invalid_vendor_clock_fails(bad_time):
    frame = raw_quotes()
    frame["time"] = bad_time
    with pytest.raises(ValueError, match="level2_time_invalid"):
        normalize(frame, "hft_quotes", INSTRUMENTS)


def test_inventory_t1_subsequent_quote_fees_and_native_ledger(tmp_path):
    from datetime import date

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import PortfolioCapital
    from quant_research_platform.markets.implementations.a_share.execution import (
        FeeRule,
        FeeSchedule,
    )
    from quant_research_platform.markets.implementations.a_share.hft_portfolio import (
        _parameters,
        inventory_batches,
    )

    quotes = normalize(raw_quotes(), "hft_quotes", INSTRUMENTS)
    empty_orders = normalize(
        raw_events("hft_orders").iloc[:0], "hft_orders", INSTRUMENTS
    )
    empty_trades = normalize(
        raw_events("hft_trades").iloc[:0], "hft_trades", INSTRUMENTS
    )
    panel = build_panel(quotes, empty_orders, empty_trades, max_gap_seconds=6)
    predictions = panel.loc[:, ["timestamp", "asset_id"]].copy()
    predictions["prediction_score"] = [1.0, -1.0, -1.0, 1.0]
    schedule = FeeSchedule(
        tuple(
            FeeRule(
                code,
                date(2018, 1, 1),
                date(2018, 12, 31),
                "a_share",
                side,
                "amount",
                rate,
                True,
            )
            for side in ("buy", "sell")
            for code, rate in (
                ("BROKER_COMMISSION", 0.0003),
                ("TRANSFER_FEE", 0.00002),
                *((("STAMP_DUTY", 0.001),) if side == "sell" else ()),
            )
        ),
        "test-fees",
    )
    params = _parameters(
        {
            "initial_capital": 10000.0,
            "initial_inventory": 100,
            "maximum_inventory": 200,
            "lot_size": 100,
            "buy_threshold": 0.5,
            "sell_threshold": -0.5,
            "latency_ms": 0,
            "participation_rate": 1.0,
            "minimum_commission": 5.0,
        }
    )
    capital = PortfolioCapital("inventory", 10000.0, "CNY", "cash_equity")

    def produce(emit):
        for batch in inventory_batches(panel, predictions, schedule, params):
            emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "2018-02-14", "end": "2018-02-14"}},
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    batches = list(artifact.iter_chunks())
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [100, -100]
    assert [trade.timestamp_ns for trade in trades] == panel.timestamp.iloc[1:3].map(
        lambda t: t.value
    ).tolist()
    np.testing.assert_allclose([trade.notional for trade in trades], [1002.0, -1001.0])
    assert batches[-1].valuations[-2].quantity == 100
    costs = 10 + 1001 * 0.001 + (1002 + 1001) * 0.00002
    summary = summarize_portfolio_ledger(capital, iter(batches))
    assert summary["fees"] == pytest.approx(costs)
    # Initial shares gain 3 CNY; the round trip loses 1 CNY before fees.
    assert summary["net_pnl"] == pytest.approx(2.0 - costs)


@pytest.mark.parametrize("initial_capital,expected_trades", [(4.0, 0), (6.0, 1)])
def test_sale_requires_cash_to_cover_minimum_commission(
    tmp_path, initial_capital, expected_trades
):
    from datetime import date

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import PortfolioCapital
    from quant_research_platform.markets.implementations.a_share.execution import (
        FeeRule,
        FeeSchedule,
    )
    from quant_research_platform.markets.implementations.a_share.hft_portfolio import (
        _parameters,
        inventory_batches,
    )

    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2018-02-14T01:30:00", "2018-02-14T01:30:03"]),
            "asset_id": "stock",
            "session_date": pd.Timestamp("2018-02-14"),
            "exchange": "SZSE",
            "eligible": True,
            "segment_id": 0,
            "mid": 0.04,
            "bid_px1": 0.03,
            "ask_px1": 0.05,
            "bid_vol1": 1000,
            "ask_vol1": 1000,
        }
    )
    predictions = quotes.iloc[:1][["timestamp", "asset_id"]].assign(
        prediction_score=-1.0
    )
    schedule = FeeSchedule(
        tuple(
            FeeRule(
                code,
                date(2018, 1, 1),
                date(2018, 12, 31),
                "a_share",
                side,
                "amount",
                0.0,
                True,
            )
            for side in ("buy", "sell")
            for code in ("BROKER_COMMISSION", "TRANSFER_FEE", "STAMP_DUTY")
            if code != "STAMP_DUTY" or side == "sell"
        ),
        "minimum-commission",
    )
    params = _parameters(
        dict(
            initial_capital=initial_capital,
            initial_inventory=100,
            maximum_inventory=100,
            lot_size=100,
            buy_threshold=0.5,
            sell_threshold=-0.5,
            latency_ms=0,
            participation_rate=1.0,
            minimum_commission=5.0,
        )
    )

    def produce(emit):
        for batch in inventory_batches(quotes, predictions, schedule, params):
            emit(batch)

    path = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("inventory", initial_capital, "CNY", "cash_equity"),
        metadata={"window": {"start": "2018-02-14", "end": "2018-02-14"}},
        produce=produce,
    )
    batches = list(read_portfolio_ledger_artifact(path).iter_chunks())
    assert sum(len(batch.trades) for batch in batches) == expected_trades
    ending = {row.asset_id: row for row in batches[-1].valuations if row.phase == "end"}
    assert ending["__cash__"].value == 0.0
    assert ending["stock"].quantity == 100 * (1 - expected_trades)
    # Selling yields 3 CNY and costs 5: only the account with 2 CNY cash can execute.
    assert sum(row.value for row in ending.values()) == (
        4.0 if not expected_trades else 0.0
    )

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    continuous_session,
)
from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
    build_panel,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.factors.microstructure import compute_factors
from quant_research_platform.research.labels.quote_returns import compute_labels
from quant_research_platform.research.protocols.a_share_hft import label_catalog
from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.sources.venvoo import (
    STREAM_FILES,
    normalize,
    source_columns,
)
from quant_research_platform.storage.datasets.reading import read_dataset
from tests.helpers.high_frequency import daily_view, write_daily_inputs

INSTRUMENTS = [{"wind_code": "000001.SZ", "asset_id": "000001.SZ", "exchange": "SZSE"}]


def test_quote_only_preparation_preserves_book_factors_and_labels(tmp_path):
    from quant_research_platform.sources.venvoo import interim_spec
    from quant_research_platform.storage.datasets import write_dataset

    source = tmp_path / "quotes"
    quotes = normalize(
        raw_quotes(tuple(93000000 + 3000 * i for i in range(13))),
        "hft_quotes",
        INSTRUMENTS,
    )
    write_dataset(source, quotes, spec=interim_spec("hft_quotes"))
    panel = prepare(
        {
            "market_id": "a_share",
            "data_id": "hft_panel",
            "inputs": {"hft_quotes": str(source)},
            "time_window": {"start": "2018-02-14", "end": "2018-02-14"},
            "output_root": str(tmp_path / "panel"),
        }
    )
    observed = read_dataset(panel)
    assert observed.mid.notna().all()
    assert observed.observed_trade_volume.isna().all()
    assert observed.observed_order_volume.isna().all()
    for data_id, selection in (
        ("hft_factors", {"factor_columns": ["relative_spread", "book_ofi"]}),
        ("hft_labels", {"label_columns": ["future_mid_return_3s"]}),
        ("hft_event_factors", {"factor_columns": ["trade_interval_seconds"]}),
    ):
        output = prepare(
            {
                "market_id": "a_share",
                "data_id": data_id,
                "inputs": {"hft_panel": str(panel)},
                "time_window": {"start": "2018-02-14", "end": "2018-02-14"},
                "output_root": str(tmp_path / data_id),
                **selection,
            }
        )
        actual = read_dataset(output)
        if data_id == "hft_factors":
            assert actual.relative_spread.notna().all()
            assert actual.book_ofi.iloc[1:].notna().all()
        elif data_id == "hft_labels":
            assert actual.future_mid_return_3s.iloc[0] == pytest.approx(
                observed.mid.iloc[1] / observed.mid.iloc[0] - 1
            )
        else:
            assert actual.trade_interval_seconds.isna().all()


def test_source_mixed_second_and_millisecond_clocks():
    raw = raw_quotes((93000, 93003123, 150001))
    counts = {"HHMMSS": 0, "HHMMSSmmm": 0}
    normalized = normalize(raw, "hft_quotes", INSTRUMENTS, clock_counts=counts)
    expected = normalize(
        raw.assign(time=[93000000, 93003123, 150001000]), "hft_quotes", INSTRUMENTS
    )
    pd.testing.assert_frame_equal(normalized, expected)
    assert counts == {"HHMMSS": 2, "HHMMSSmmm": 1}
    assert normalized.timestamp.dtype == np.dtype("datetime64[ns]")
    assert normalized.session_date.dtype == np.dtype("datetime64[ns]")
    assert normalized.timestamp.iloc[1] == pd.Timestamp("2018-02-14T01:30:03.123")
    with pytest.raises(ValueError, match="level2_time_invalid"):
        normalize(raw.assign(time=[93060, 93003123, 150001]), "hft_quotes", INSTRUMENTS)


@pytest.mark.parametrize("stream", ["hft_orders", "hft_trades"])
def test_source_preserves_nullable_unsigned_event_identifiers(tmp_path, stream):
    identifiers = [name for name in source_columns(stream) if name.endswith("_id")]
    expected = [7, 8, 9, 10, 2**53 + 1, None, 2**64 - 1, 0]
    for day, values in (("20180214", expected[:4]), ("20180215", expected[4:])):
        raw = tmp_path / "raw" / day
        raw.mkdir(parents=True)
        frame = raw_events(stream).assign(date=int(day))
        for name in identifiers:
            frame[name] = pd.Series(values, dtype="UInt64")
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
                None
            ),
            raw / STREAM_FILES[stream],
        )
    output = tmp_path / "interim"
    prepare_source(
        {
            "source_id": "venvoo",
            "raw_root": str(tmp_path / "raw"),
            "outputs": {stream: str(output)},
            "instruments": INSTRUMENTS,
            "revision": "a" * 40,
            "start_date": "2018-02-14",
            "end_date": "2018-02-15",
        }
    )
    result = read_dataset(output).sort_values("timestamp")
    for name in identifiers:
        pd.testing.assert_series_equal(
            result[name].reset_index(drop=True),
            pd.Series(expected, dtype="UInt64", name=name),
        )


def test_hft_prepare_rejects_conflicting_declared_market(tmp_path):
    from dataclasses import replace

    from quant_research_platform.research.protocols.a_share_hft import processed_specs
    from quant_research_platform.storage.datasets import write_dataset

    panel = build_panel(
        normalize(raw_quotes(), "hft_quotes", INSTRUMENTS),
        normalize(raw_events("hft_orders"), "hft_orders", INSTRUMENTS),
        normalize(raw_events("hft_trades"), "hft_trades", INSTRUMENTS),
        max_gap_seconds=6,
    )
    spec = next(item for item in processed_specs() if item.data_id == "hft_panel")
    root = tmp_path / "wrong-market"
    panel["market_id"] = "crypto"
    write_dataset(root, panel, spec=replace(spec, market_id="crypto"))
    with pytest.raises(ValueError, match="input_market_mismatch"):
        prepare(
            {
                "market_id": "a_share",
                "data_id": "hft_factors",
                "inputs": {"hft_panel": str(root)},
                "factor_columns": ["book_ofi"],
                "time_window": {"start": "2018-02-14", "end": "2018-02-14"},
                "output_root": str(tmp_path / "factors"),
            }
        )


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
        frame.loc[1, "bs_flag"] = " "
        frame.loc[1, "ask_order_id"] = 0
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
    assert panel.observed_signed_cancel_volume.iloc[1] == 200
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


def test_shenzhen_cancellation_side_requires_one_known_order_index():
    frame = pd.concat([raw_events("hft_trades").iloc[[1]]] * 7, ignore_index=True)
    frame["bid_order_id"] = pd.Series([7, 0, 7, 0, None, 2**64 - 1, 0], dtype="UInt64")
    frame["ask_order_id"] = pd.Series([0, 9, 9, 0, 0, 0, 2**64 - 1], dtype="UInt64")
    result = normalize(frame, "hft_trades", INSTRUMENTS)
    assert result.side.iloc[:2].tolist() == ["buy", "sell"]
    assert result.side.iloc[2:].isna().all()


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


def test_etf_sessions_include_historical_shanghai_fund_close():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2023-11-06 06:59:57",
                "2023-11-06 07:00:00",
                "2023-11-06 06:58:00",
                "2026-07-03 06:58:00",
                "2026-07-06 06:58:00",
            ]
        )
    )
    assert continuous_session(
        timestamps,
        pd.Series(["SSE", "SSE", "SZSE", "SSE", "SSE"]),
        pd.Series(["equity_etf"] * 5),
    ).tolist() == ["pm", "", "", "pm", ""]
    instruments = [
        {
            "wind_code": "518880.SZ",
            "asset_id": "518880",
            "exchange": "SSE",
            "instrument_type": "gold_etf",
        }
    ]
    raw = raw_quotes((145959000, 150000000)).assign(
        date=20231106, wind_code="518880.SZ"
    )
    panel = build_panel(
        normalize(raw, "hft_quotes", instruments),
        normalize(raw_events("hft_orders").iloc[:0], "hft_orders", instruments),
        normalize(raw_events("hft_trades").iloc[:0], "hft_trades", instruments),
        max_gap_seconds=6,
    )
    assert panel.timestamp.tolist() == [pd.Timestamp("2023-11-06 06:59:59")]
    assert panel.instrument_type.tolist() == ["gold_etf"]
    assert panel.exchange.tolist() == ["SSE"]


@pytest.mark.parametrize("instrument_type", ["stock", "equity_etf"])
def test_native_source_to_assembled_hft_dataset(tmp_path, monkeypatch, instrument_type):
    raw = tmp_path / "raw" / "20180214"
    raw.mkdir(parents=True)
    for stream, name in STREAM_FILES.items():
        frame = raw_quotes() if stream == "hft_quotes" else raw_events(stream)
        if stream == "hft_quotes":
            frame.loc[1, "bid_vol1"] = np.nan
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), raw / name)
    outputs = {stream: str(tmp_path / stream) for stream in STREAM_FILES}
    result = prepare_source(
        {
            "source_id": "venvoo",
            "raw_root": str(raw.parent),
            "outputs": outputs,
            "instruments": [dict(INSTRUMENTS[0], instrument_type=instrument_type)],
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
            "factor_columns": [
                "relative_spread",
                "imbalance_1",
                "observed_cancel_flow_imbalance",
            ],
            "label_columns": ["future_mid_return_3s"],
        }
    )
    result = read_dataset(path)
    assert len(result) == 4
    assert result.future_mid_return_3s.notna().sum() == 3
    assert result.eligible.all()
    assert result.relative_spread.iloc[1] == pytest.approx(0.02 / 10.01)
    assert pd.isna(result.imbalance_1.iloc[1])
    assert result.observed_cancel_flow_imbalance.iloc[1] == 1.0

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.evaluations.runner import evaluate
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.portfolios import high_frequency
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.sources.reference.interim import fee_schedule_spec
    from quant_research_platform.storage.datasets.writing import write_dataset

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
                "market": "a_share" if instrument_type == "stock" else "etf",
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
    panel = read_dataset(inputs["hft_panel"])
    history = panel.assign(
        timestamp=panel.timestamp - pd.Timedelta(days=1),
        session_date=panel.session_date - pd.Timedelta(days=1),
    )
    quote_reads = []
    materialize = high_frequency.materialize_dataset_session

    def record_quote_read(session):
        quote_reads.append(session.time_window)
        return materialize(session)

    monkeypatch.setattr(
        high_frequency, "materialize_dataset_session", record_quote_read
    )
    account_config = {
        "panel": inputs["hft_panel"],
        "predictions": [str(predictions)],
        "fee_schedule": str(tmp_path / "fees"),
        "daily_inputs": write_daily_inputs(
            tmp_path / "daily", pd.concat((history, panel), ignore_index=True)
        ),
        "window": forecast_window,
        "output": str(tmp_path / "accounts"),
        "parameters": {
            "initial_capital": 10000.0,
            "initial_inventory": {"000001.SZ": 100},
            "maximum_inventory": {"000001.SZ": 200},
            "restore_at": None,
            "buy_threshold": 0.0,
            "sell_threshold": -1.0,
            "latency_ms": 0,
            "participation_rate": 1.0,
            "minimum_commission": 5.0,
        },
    }
    destination = high_frequency.backtest(account_config)
    ledger = read_portfolio_ledger_artifact(destination)
    assert len(quote_reads) == 1
    assert (
        ledger.metadata["account_semantics"]
        == "shared_cash_continuous_inventory_by_instrument"
    )
    assert sum(len(batch.trades) for batch in ledger.iter_chunks()) == 1
    held = high_frequency.backtest(
        {
            **account_config,
            "predictions": [],
            "output": str(tmp_path / "holding"),
        }
    )
    held_batches = list(read_portfolio_ledger_artifact(held).iter_chunks())
    assert not any(batch.trades for batch in held_batches)
    expected = 10000.0 + 100 * (panel.mid.iloc[-1] - panel.mid.iloc[-2])
    assert sum(
        row.value for row in held_batches[-1].valuations if row.phase == "end"
    ) == pytest.approx(expected)


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
    from quant_research_platform.research.portfolios.high_frequency import (
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
            "initial_inventory": {"000001.SZ": 100},
            "maximum_inventory": {"000001.SZ": 200},
            "restore_at": None,
            "buy_threshold": 0.5,
            "sell_threshold": -0.5,
            "latency_ms": 0,
            "participation_rate": 1.0,
            "minimum_commission": 5.0,
        }
    )
    capital = PortfolioCapital("inventory", 10000.0, "CNY", "cash_equity")

    def produce(emit):
        for batch in inventory_batches(
            [(daily_view(panel), panel, predictions)],
            schedule,
            params,
            TimeWindow("2018-02-14", "2018-02-14"),
        ):
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
    from quant_research_platform.research.portfolios.high_frequency import (
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
            initial_inventory={"stock": 100},
            maximum_inventory={"stock": 100},
            restore_at=None,
            buy_threshold=0.5,
            sell_threshold=-0.5,
            latency_ms=0,
            participation_rate=1.0,
            minimum_commission=5.0,
        )
    )

    def produce(emit):
        for batch in inventory_batches(
            [(daily_view(quotes), quotes, predictions)],
            schedule,
            params,
            TimeWindow("2018-02-14", "2018-02-14"),
        ):
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


@pytest.mark.parametrize("missing", ["bid_vol1", "ask_vol1", "bid_px1"])
def test_missing_book_field_keeps_independent_price_and_event_history(missing):
    quotes = normalize(raw_quotes(), "hft_quotes", INSTRUMENTS)
    trades = normalize(raw_events("hft_trades"), "hft_trades", INSTRUMENTS)
    orders = normalize(raw_events("hft_orders"), "hft_orders", INSTRUMENTS)
    baseline = build_panel(quotes, orders, trades, max_gap_seconds=6)
    quotes.loc[1, missing] = np.nan
    panel = build_panel(quotes, orders, trades, max_gap_seconds=6)
    assert not panel.eligible.iloc[1]
    assert panel.segment_id.nunique() == 1
    pd.testing.assert_series_equal(
        panel.observed_trade_volume, baseline.observed_trade_volume
    )
    factors = compute_factors(panel)
    expected = compute_factors(baseline)
    pd.testing.assert_series_equal(
        factors.observed_order_flow_imbalance, expected.observed_order_flow_imbalance
    )
    labels = compute_labels(panel, label_catalog().list())
    if missing.endswith("vol1"):
        assert panel.mid.iloc[1] == baseline.mid.iloc[1]
        assert factors.spread_cny.iloc[1] == pytest.approx(0.02)
        assert labels.future_mid_return_3s.iloc[0] == pytest.approx(10.01 / 10 - 1)
        assert (
            labels.future_mid_return_3s_available_at.iloc[0] == panel.timestamp.iloc[1]
        )
        assert pd.isna(factors.imbalance_1.iloc[1])
    else:
        assert pd.isna(panel.mid.iloc[1])
        assert pd.isna(factors.spread_cny.iloc[1])
        assert pd.isna(factors.book_ofi.iloc[1])
        assert pd.isna(factors.book_ofi.iloc[2])
        # A price-only label selects the next valid quote in the observed segment.
        assert labels.future_mid_return_3s.iloc[0] == pytest.approx(10.02 / 10 - 1)
        assert (
            labels.future_mid_return_3s_available_at.iloc[0] == panel.timestamp.iloc[2]
        )
    from quant_research_platform.research.factors.event_microstructure import (
        compute_event_factors,
    )
    from quant_research_platform.research.factors.hawkes import compute_hawkes_factors

    options = {"columns": ("trade_signed_run",), "config": {}}
    pd.testing.assert_frame_equal(
        compute_event_factors(panel, orders, trades, **options),
        compute_event_factors(baseline, orders, trades, **options),
    )
    model = (
        (np.array([1.0, 1.0]), np.zeros((2, 2)), 1.0),
        panel.timestamp.iloc[0] - pd.Timedelta(seconds=1),
    )
    pd.testing.assert_frame_equal(
        compute_hawkes_factors(panel, trades, model),
        compute_hawkes_factors(baseline, trades, model),
    )

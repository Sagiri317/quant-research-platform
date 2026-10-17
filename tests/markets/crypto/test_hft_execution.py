from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    price_to_index,
)
from quant_research_platform.contracts.portfolios import (
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerConfig,
    TakerConfig,
    _Metrics,
    _PendingMarkouts,
    simulate_maker,
    simulate_taker,
)
from quant_research_platform.research.portfolios.diagnostics import (
    fixed_turnover_cost_sensitivity,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import KNOWN_NO_FUNDING, _coverage, _quote
from tests.helpers.portfolio_ledger import capture_fill_rows


def _native_taker_projection_case(tmp_path, monkeypatch, *, count=300):
    import json
    from contextlib import contextmanager
    from dataclasses import asdict
    from types import SimpleNamespace

    from quant_research_platform.contracts.portfolios import (
        FundingEvent,
        PortfolioTargetBlocked,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.crypto import portfolio
    from quant_research_platform.sources.binance.replay import (
        ReconstructionStats,
        iter_reconstructed_depth,
    )
    from quant_research_platform.sources.market import open_market_source
    from quant_research_platform.sources.tardis.raw import import_slices
    from tests.helpers.tardis import _slices

    start = pd.Timestamp("2026-09-01T00:00:00Z")
    end = start + pd.Timedelta(milliseconds=count * 10)
    lines = []
    for index in range(count + 1):
        stamp = start + pd.Timedelta(milliseconds=index * 10)
        payload = (
            {
                "lastUpdateId": 100,
                "bids": [[str(10000 - level), "1"] for level in range(1000)],
                "asks": [[str(10002 + level), "1"] for level in range(1000)],
            }
            if index == 0
            else {
                "e": "depthUpdate",
                "E": stamp.value // 1_000_000,
                "T": stamp.value // 1_000_000,
                "s": "BTCUSDT",
                "st": 1,
                "ps": "BTCUSDT",
                "U": 99 + index,
                "u": 100 + index,
                "pu": 99 + index,
                "b": [[str(10000 - level), "1"] for level in range(20)],
                "a": [[str(10002 + level), "1"] for level in range(20)],
            }
        )
        record = {
            "stream": "btcusdt@depthSnapshot" if index == 0 else "btcusdt@depth@0ms",
            "data": payload,
        }
        if index == 0:
            record["generated"] = True
        lines.append(
            (
                stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                + " "
                + json.dumps(record)
                + "\n"
            ).encode()
        )
    root = tmp_path / "archive"
    import_slices(
        root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", b"".join(lines)),
    )
    initial = ReconstructionStats()
    with open_market_source(root) as source:
        for _event in iter_reconstructed_depth(
            source, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=initial
        ):
            pass
    with open_market_source(root) as source:
        source_metadata = source.market_metadata(FUTURES_MARKET_TYPE, "BTCUSDT")
    coverage = initial.coverage_segments
    window = TimeWindow(
        (start + pd.Timedelta(milliseconds=100)).isoformat(), end.isoformat()
    )
    monkeypatch.setattr(
        portfolio,
        "read_dataset_manifest_evidence",
        lambda *_args: {
            "source_evidence": {"market_metadata": {"BTCUSDT": source_metadata}},
            "producer_diagnostics": {
                "reconstruction": {"BTCUSDT": {"coverage_segments": coverage}}
            },
        },
    )
    monkeypatch.setattr(
        portfolio,
        "_prediction_rows",
        lambda **_kwargs: iter(
            {
                "timestamp_ms": start.value // 1_000_000 + offset,
                "prediction_score": 0.1 if offset % 200 else -0.1,
            }
            for offset in range(100, count * 10 - 300, 100)
        ),
    )

    def execute(*, latency, horizon, mode, projection="native", fault=None):
        closed = []
        stats = ReconstructionStats()
        emitted_levels = []
        summaries, batches = [], []

        @contextmanager
        def open_run(_root, **_kwargs):
            with open_market_source(root) as source:
                try:
                    yield SimpleNamespace(
                        source_identity=source.source_identity,
                        market_metadata=source.market_metadata,
                        iter_market_records=source.iter_market_records,
                        _coverage_index=source._coverage_index,
                        funding_roots=lambda *_args, **_kwargs: (),
                        observation_coverage=lambda **_kwargs: SimpleNamespace(
                            intervals=lambda: iter(((start.value, end.value + 1),)),
                            covering=lambda *_args: ("synthetic-trade-coverage",),
                        ),
                    )
                finally:
                    closed.append("source")

        @contextmanager
        def funding(*_args, **_kwargs):
            cuts = (
                tuple(
                    FundingEvent(start.value + offset * 1_000_000, None, None)
                    for offset in (750, 1500, 2250)
                )
                if mode == "independent_segments"
                else ()
            )
            try:
                yield SimpleNamespace(
                    identity="f" * 64,
                    coverage=((start.value, end.value + 1),),
                    settlement_qualification={"source": "synthetic"},
                    events=cuts,
                    forecasts=(
                        FundingForecast(start.value, end.value + 1_000_000_000, ()),
                    ),
                )
            finally:
                closed.append("funding")

        def reconstruct(source, *, output_levels=None, market_type=None, **kwargs):
            prescan = output_levels == 0
            active_stats = kwargs.pop("stats", None) or stats
            if projection != "native" and not prescan:
                output_levels = None if projection == "full" else 1
            if fault is not None and not getattr(source, "_faulted", False):
                source._faulted = True
                records = source.iter_market_records

                def unordered(**kwargs):
                    for index, row in enumerate(records(**kwargs)):
                        if index == 20:
                            if fault == "time_order":
                                row = {
                                    **row,
                                    "observed_time_ns": start.value,
                                    "receive_time": start.isoformat(),
                                }
                            else:
                                row = {
                                    **row,
                                    "payload": {
                                        **row["payload"],
                                        "b": [
                                            [str(10000 - level), "0"]
                                            for level in range(1, 1000)
                                        ],
                                    },
                                }
                        yield row

                source.iter_market_records = unordered
            events = iter_reconstructed_depth(
                source,
                FUTURES_MARKET_TYPE,
                output_levels=output_levels,
                required_levels=20,
                stats=active_stats,
                **kwargs,
            )
            try:
                for event in events:
                    if isinstance(event, BookState) and not prescan:
                        emitted_levels.append((len(event.bids), len(event.asks)))
                    yield event
            finally:
                events.close()
                stats.__dict__.update(active_stats.__dict__)
                closed.append("prescan" if prescan else "books")

        def episode(**kwargs):
            result = kwargs.pop("produce")(batches.append)
            summaries.append((kwargs, result))

        config = {
            "output": str(tmp_path / "unused"),
            "inputs": "explicit-dataset",
            "panel": "explicit-panel",
            "source": str(root),
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
            "predictions": "explicit-predictions",
            "prediction_window": {
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            },
            "execution_mode": mode,
            "capital": {
                "candidate_id": "fixture",
                "initial_capital": 1e6,
                "valuation_currency": "USDT",
                "valuation_method": "linear_perpetual",
            },
            "execution": {
                "latency_ms": latency,
                "taker_fee_bps": 2.0,
                "markout_horizon_ms": 100,
                "order_quantity": 3.0,
                "position_size": 6.0,
            },
        }
        monkeypatch.setattr(portfolio, "open_market_source", open_run)
        monkeypatch.setattr(portfolio, "iter_reconstructed_depth", reconstruct)
        monkeypatch.setattr(portfolio, "read_funding_observations", funding)
        monkeypatch.setattr(
            portfolio,
            "_prediction_evidence",
            lambda *args, **kwargs: (
                {},
                {"model": {"target_column": f"future_mid_return_{horizon}ms"}},
                None,
            ),
        )
        monkeypatch.setattr(
            portfolio,
            "_write_backtest",
            lambda config, capital, produce: produce(
                on_ledger=batches.append, on_episode=episode
            ),
        )
        try:
            result = portfolio.backtest_taker(config)
        except (ValueError, PortfolioTargetBlocked) as error:
            result = (type(error).__name__, str(error))
        assert sorted(closed) == sorted(
            ["books", "funding", "source"]
            + (["prescan"] if mode == "independent_segments" else [])
        )
        return (result, summaries, batches, asdict(stats)), emitted_levels

    return execute


@pytest.mark.parametrize("mode", ("continuous", "independent_segments"))
@pytest.mark.parametrize(
    "latency,horizon", ((0, 100), (100, 100), (200, 100), (100, 500))
)
def test_taker_native_projection_preserves_complete_ledger(
    tmp_path, monkeypatch, latency, horizon, mode
):
    execute = _native_taker_projection_case(tmp_path, monkeypatch)
    full, full_levels = execute(
        latency=latency, horizon=horizon, mode=mode, projection="full"
    )
    projected, levels = execute(latency=latency, horizon=horizon, mode=mode)
    assert full == projected
    assert len(levels) == len(full_levels) == 300
    expected = 1 if latency >= horizon else 1000
    assert set(levels) == {(expected, expected)}
    summaries = (
        [row[1] for row in full[1]] if mode == "independent_segments" else [full[0]]
    )
    assert len(summaries) == (7 if mode == "independent_segments" else 1)
    if latency >= horizon:
        assert all(summary["created_orders"] == 0 for summary in summaries)
        assert all(
            summary["selection_status"] in {"UNDEFINED", "COMPLETE"}
            for summary in summaries
        )
        assert all(
            summary["selection_status"] == "UNDEFINED"
            for summary in summaries
            if summary["equity_rows"]
        )
    else:
        assert all(
            summary["created_orders"] > 0
            for summary in summaries
            if summary["equity_rows"]
        )
        assert any(abs(fill.quantity) > 1 for batch in full[2] for fill in batch.trades)


@pytest.mark.parametrize("fault", ("depth", "time_order"))
def test_taker_projection_keeps_native_reconstruction_failures(
    tmp_path, monkeypatch, fault
):
    execute = _native_taker_projection_case(tmp_path, monkeypatch)
    full, _levels = execute(
        latency=100, horizon=100, mode="continuous", projection="full", fault=fault
    )
    projected, _levels = execute(
        latency=100, horizon=100, mode="continuous", fault=fault
    )
    assert full == projected
    if fault == "depth":
        assert full[3]["unsupported_coverage"] == 1
        # The next continuous absolute update restores all required levels.
        # Preserve the real uncovered interval rather than halting forever.
        assert full[3]["emitted_states"] == 299
        assert full[3]["valid_segments"] == 2
        assert full[3]["invalid_segments"] == full[3]["recovered_segments"] == 1
        bounds = sorted(full[3]["coverage_segments"].values())
        start = pd.Timestamp("2026-09-01T00:00:00Z").value
        assert bounds == [
            (start + 10_000_000, start + 200_000_000),
            (start + 210_000_000, start + 3_000_000_001),
        ]
    else:
        assert full[0] == ("ValueError", "binance_replay_observed_time_non_monotonic")


@pytest.mark.usefixtures("coordinate_scale")
def test_zero_latency_maker_replacement_releases_old_reservation() -> None:
    from tests.markets.crypto.calibration_test_support import calibration_from_paths

    fills = []
    calibration = calibration_from_paths(
        [
            (
                [_state(0)],
                [
                    _trade(400, price="99.5", quantity="3", maker_buyer=True),
                    _trade(401, price="100.5", quantity="2", maker_buyer=False),
                ],
            )
        ],
        [0.0],
    )
    result = simulate_maker(
        [_state(stamp) for stamp in (0, 100, 200, 300, 400, 500, 600, 700)],
        [
            _trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)
            for stamp, quantity in ((100, "1"), (350, "2"), (650, "1"))
        ],
        [
            {
                **_quote(stamp, book_segment_id="s1"),
                "fold_id": "fold",
                "price_prediction": 0.0,
                "proxy_prediction": 1.0,
            }
            for stamp in (0, 200)
        ],
        config=replace(
            _maker_config(calibrated=True),
            max_position=1.0,
            maker_fee_bps=0.0,
            latency_ms=0,
        ),
        observed_coverage={"s1": (0, 1000000000)},
        trade_coverage={"trade-connection-1": (0, 1000000000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=(FundingForecast(0, 1000000000, ()),),
        calibrations={"fold": calibration},
        on_ledger=capture_fill_rows(fills.append),
    )
    assert result["created_orders"] == 4
    assert [fill["timestamp_ms"] for fill in fills] == [650]
    assert "inventory_target" not in result and "inventory_skew" not in result


def test_fixed_maker_inventory_controls_change_real_fill_quantity():
    quantities = []
    for target, skew in ((0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)):
        fills = []
        result = simulate_maker(
            [_state(0), _state(200)],
            [_trade(50, price="99.5", quantity="4", maker_buyer=True)],
            [_quote(0, book_segment_id="s1")],
            config=replace(
                _maker_config(), inventory_target=target, inventory_skew=skew
            ),
            observed_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
            on_ledger=capture_fill_rows(fills.append),
            capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
        )
        assert result["inventory_target"] == target
        assert result["inventory_skew"] == skew
        quantities.append(sum(fill["fill_qty"] for fill in fills))
    assert quantities == [1.0, 0.5, 1.0]


def _state(
    timestamp_ms: int,
    *,
    bid_index: int = 995,
    ask_index: int = 1005,
    bid_levels: tuple[tuple[int, float], ...] | None = None,
    ask_levels: tuple[tuple[int, float], ...] | None = None,
    segment_id: str = "s1",
) -> BookState:
    return BookState(
        source_sequence=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        update_id=timestamp_ms + 1,
        segment_id=segment_id,
        observation_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        bids=bid_levels or ((bid_index, 2.0), (bid_index - 5, 3.0)),
        asks=ask_levels or ((ask_index, 1.0), (ask_index + 5, 2.0)),
    )


def _boundary(timestamp_ms: int, *, segment_id: str = "s1") -> BookBoundary:
    return BookBoundary(
        source_sequence=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        segment_id=segment_id,
        observation_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        reason="sequence_gap",
    )


def _trade(
    timestamp_ms: int, *, price: str, quantity: str, maker_buyer: bool
) -> TradeEvent:
    quantity_value = float(quantity)
    return TradeEvent(
        source_sequence=timestamp_ms + 10,
        event_time_ms=timestamp_ms,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 10,
        observation_id="trade-connection-1",
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        price_index=price_to_index(price, "0.1", error="bad"),
        quantity=quantity_value,
        visible_quantity=quantity_value,
        buyer_is_maker=maker_buyer,
    )


def _taker_config(*, latency_ms: int = 0, order_quantity: float = 1.0) -> TakerConfig:
    return TakerConfig(
        order_quantity=order_quantity,
        position_size=2.0,
        latency_ms=latency_ms,
        taker_fee_bps=1.0,
        markout_horizon_ms=100,
    )


def _maker_config(*, calibrated=False) -> MakerConfig:
    return MakerConfig(
        latency_ms=0,
        order_quantity=1.0,
        max_position=2.0,
        maker_fee_bps=-1.0,
        inventory_target=None if calibrated else 0.0,
        inventory_skew=None if calibrated else 0.0,
        markout_horizon_ms=100,
    )


@pytest.mark.parametrize(
    "lineage_identity", (None, "book_segment_id", "trade_observation_id")
)
def test_common_panel_maker_quotes_bind_actual_source_identities(
    lineage_identity,
) -> None:
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _paired_validation,
    )

    frames = {
        symbol: pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(0),
                    "asset_id": asset,
                    "segment_id": "s1:trade-connection-1",
                    "book_segment_id": "s1",
                    "trade_observation_id": "trade-connection-1",
                    "prediction_score": 1.0,
                    "label": 0.0,
                    "label_available_at": pd.Timestamp(100000000),
                }
            ]
        )
        for symbol, asset in (("BTCUSDT", "btc_usdt"), ("ETHUSDT", "eth_usdt"))
    }
    panel = pd.concat(frames.values(), ignore_index=True)
    assert panel["segment_id"].tolist() == ["s1:trade-connection-1"] * 2
    if lineage_identity is not None:
        panel[lineage_identity] = "independent-input-identity"

    quotes = list(
        _paired_validation(
            [
                panel.loc[panel.asset_id.eq("btc_usdt")].itertuples(index=False)
                for _ in range(2)
            ],
            asset_id="btc_usdt",
        )
    )
    assert len(quotes) == 1
    assert "segment_id" not in quotes[0]
    fills = []
    result = simulate_maker(
        [_state(0), _state(200)],
        [_trade(50, price="99.4", quantity="0.1", maker_buyer=True)],
        quotes,
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 201)},
        trade_coverage={"trade-connection-1": _coverage(0, 201)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 1
    assert result["created_orders"] == 2


@pytest.mark.usefixtures("coordinate_scale")
def test_taker_uses_latest_actual_state_at_effective_time() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _state(5), _state(15, bid_index=1005, ask_index=1015)],
        [{"timestamp_ms": 0, "prediction": 0.02, "segment_id": "s1"}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": _coverage(0, 15)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert fills[0]["decision_time_ms"] == 0
    assert fills[0]["effective_time_ms"] == 10
    assert fills[0]["state_observed_time_ms"] == 5
    assert fills[0]["timestamp_ms"] == 10
    assert fills[0]["fill_price"] == 100.5
    assert diagnostics["latency_fidelity"] == "full_l2_observed_event"
    assert diagnostics["assumption"] == "historical_book_zero_market_impact"


def test_failing_fill_observer_is_not_replayed_during_cleanup() -> None:
    calls: list[int] = []

    def fail(fill):
        calls.append(fill["order_id"])
        raise KeyboardInterrupt("observer_primary")

    with pytest.raises(KeyboardInterrupt, match="observer_primary"):
        simulate_taker(
            [_state(0), _state(200)],
            [{"timestamp_ms": 0, "prediction": 0.02}],
            funding_forecasts=KNOWN_NO_FUNDING,
            prediction_horizon_ms=500,
            prediction_column="prediction",
            config=_taker_config(),
            observed_coverage={"s1": _coverage(0, 300)},
            on_ledger=capture_fill_rows(fail),
            capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
        )
    assert calls == [1]


@pytest.mark.usefixtures("coordinate_scale")
def test_taker_walks_known_l2_and_reports_partial_coverage_honestly() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _state(5, ask_levels=((1005, 0.4), (1010, 0.6)))],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(latency_ms=10, order_quantity=1.5),
        observed_coverage={"s1": _coverage(0, 11)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 1
    assert math.isclose(fills[0]["fill_qty"], 1.0)
    assert fills[0]["levels_consumed"] == 2
    assert math.isclose(fills[0]["fill_price"], 100.8)
    assert math.isclose(diagnostics["unfilled_quantity"], 0.5)
    assert diagnostics["zero_fill_orders"] == 0
    assert "unfilled_orders" not in diagnostics
    assert diagnostics["markout_undefined_fill_events"] == 1
    assert diagnostics["average_markout_bps"] is None


def test_taker_does_not_fill_after_coverage_or_explicit_boundary() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _boundary(50)],
        [{"timestamp_ms": 100, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 49)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert fills == []
    assert diagnostics["filled_orders"] == 0
    assert diagnostics["gross_pnl"] == 0.0


def test_taker_terminal_pnl_is_unknown_after_book_boundary() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _boundary(50)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 50)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 1
    assert diagnostics["gross_pnl"] is None
    assert diagnostics["net_pnl"] is None


def test_taker_markout_does_not_use_state_past_observed_coverage() -> None:
    diagnostics = simulate_taker(
        [_state(0), _state(200, bid_index=1005, ask_index=1015)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 100)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert diagnostics["average_markout_bps"] is None
    assert diagnostics["markout_undefined_fill_events"] == 1


@pytest.mark.parametrize(
    "at_target, tail", [(False, True), (True, True), (True, False)]
)
def test_taker_markout_uses_latest_state_as_of_target(at_target, tail) -> None:
    fills: list[dict[str, object]] = []
    stamp = 100 if at_target else 50
    books = [
        _state(0, bid_index=1000, ask_index=1005),
        _state(stamp, bid_index=995, ask_index=1005),
        replace(
            _state(stamp, bid_index=1005, ask_index=1015),
            source_sequence=stamp + 2,
            update_id=stamp + 2,
        ),
    ]
    if tail:
        books.append(_state(200, bid_index=1095, ask_index=1105))
    diagnostics = simulate_taker(
        books,
        [{"timestamp_ms": 0, "prediction": 0.1}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 300)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert diagnostics["markout_undefined_fill_events"] == 0
    assert fills[0]["markout_per_unit"] == pytest.approx(0.5)


@pytest.mark.usefixtures("coordinate_scale")
def test_taker_target_position_does_not_accumulate_repeated_signal() -> None:
    diagnostics = simulate_taker(
        [_state(0), _state(100), _state(200)],
        [
            {"timestamp_ms": 0, "prediction": 0.02},
            {"timestamp_ms": 100, "prediction": 0.02},
            {"timestamp_ms": 200, "prediction": 0.02},
        ],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=TakerConfig(2.0, 2.0, 0, 0.0, 100),
        observed_coverage={"s1": _coverage(0, 200)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert diagnostics["created_orders"] == 1
    assert diagnostics["max_abs_position"] == 2.0


@pytest.mark.usefixtures("coordinate_scale")
def test_maker_same_price_depletion_partial_fill_and_order_lifetime() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _state(200, bid_index=1000, ask_index=1010)],
        [
            _trade(50, price="99.5", quantity="2.5", maker_buyer=True),
            _trade(100, price="99.5", quantity="0.5", maker_buyer=True),
        ],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 201)},
        trade_coverage={"trade-connection-1": _coverage(0, 201)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        decision_windows_ns=((0, 200_000_000),),
        funding_events=(),
    )
    assert [fill["fill_qty"] for fill in fills] == [0.5, 0.5]
    assert diagnostics["partial_fill_events"] == 1
    assert diagnostics["filled_orders"] == 1
    assert diagnostics["average_filled_order_lifetime_ms"] == 100.0
    assert diagnostics["average_markout_bps"] is not None
    assert diagnostics["fees"] < 0
    assert diagnostics["spread_capture_pnl"] > 0
    assert diagnostics["inventory_revaluation_pnl"] is not None


@pytest.mark.usefixtures("coordinate_scale")
def test_maker_trade_through_fills_and_boundary_blocks_stale_trade() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _boundary(75)],
        [
            _trade(50, price="99.4", quantity="0.1", maker_buyer=True),
            _trade(100, price="99.0", quantity="1", maker_buyer=True),
        ],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 74)},
        trade_coverage={"trade-connection-1": _coverage(0, 101)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 1
    assert fills[0]["fill_reason"] == "trade_through"
    assert diagnostics["trade_through_fill_events"] == 1
    assert diagnostics["gross_pnl"] is None


def test_maker_terminal_pnl_is_unknown_after_uncovered_state() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _boundary(75), _state(100, segment_id="s2")],
        [_trade(50, price="99.4", quantity="1", maker_buyer=True)],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 75)},
        trade_coverage={"trade-connection-1": _coverage(0, 101)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 1
    assert diagnostics["gross_pnl"] is None
    assert diagnostics["net_pnl"] is None


def test_maker_terminal_pnl_preserves_flat_realized_cash_after_boundary() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _boundary(75)],
        [
            _trade(50, price="99.4", quantity="1", maker_buyer=True),
            _trade(60, price="100.6", quantity="1", maker_buyer=False),
        ],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 75)},
        trade_coverage={"trade-connection-1": _coverage(0, 101)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 2
    assert diagnostics["gross_pnl"] == pytest.approx(1.0)
    assert diagnostics["net_pnl"] == pytest.approx(1.02)


def test_cost_sensitivity_reprices_actual_turnover_and_net_pnl() -> None:
    result = fixed_turnover_cost_sensitivity(
        turnover=10000.0,
        net_pnl=5.0,
        base_fee_bps=1.0,
        fee_scenarios_bps=(1.0, 2.0),
        additional_spread_scenarios_bps=(0.0, 1.0),
    )
    assert [row["net_pnl"] for row in result] == pytest.approx([5.0, 4.0, 4.0, 3.0])
    assert {row["method"] for row in result} == {
        "constant_actual_turnover_additive_sensitivity"
    }


def test_taker_long_stream_returns_counts_without_dataframe_results() -> None:

    def decisions():
        for index in range(20000):
            yield {"timestamp_ms": index, "prediction": 0.0}

    diagnostics = simulate_taker(
        [_state(0), _state(19999)],
        decisions(),
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 19999)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert diagnostics["fill_events"] == 0
    assert diagnostics["equity_rows"] == 20000
    # This is a long-stream count/result-shape smoke test, not a memory bound.
    assert not any(isinstance(value, pd.DataFrame) for value in diagnostics.values())


def test_taker_freezes_orders_across_overlapping_latency_and_future_book_changes():
    from quant_research_platform.contracts.portfolios import FundingEvent

    selections = []
    fills_by_case = []
    for future_quantity in (2.0, 0.25):
        ledger = []
        fills = []
        result = simulate_taker(
            [_state(0), _state(25, ask_levels=((1100, future_quantity),)), _state(80)],
            [{"timestamp_ms": stamp, "prediction": 0.02} for stamp in (0, 10, 20)],
            funding_forecasts=KNOWN_NO_FUNDING,
            prediction_column="prediction",
            prediction_horizon_ms=100,
            config=_taker_config(latency_ms=30),
            observed_coverage={"s1": _coverage(0, 100)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=capture_fill_rows(fills.append, ledger.append),
        )
        quantities = [
            row.number
            for batch in ledger
            for row in batch.diagnostics
            if row.name == "decision_requested_quantity"
        ]
        selections.append(quantities)
        fills_by_case.append([row["fill_qty"] for row in fills])
        assert result["created_orders"] == 2 and result["max_abs_position"] <= 2.0
    assert selections == [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert fills_by_case == [[1.0, 1.0], [0.25, 0.25]]
    result = simulate_taker(
        [_state(0), _state(100)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_column="prediction",
        prediction_horizon_ms=100,
        config=_taker_config(latency_ms=50),
        observed_coverage={"s1": _coverage(0, 101)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(FundingEvent(25000000, None, None),),
    )
    assert result["selection_status"] == "COMPLETE"
    assert result["created_orders"] == 1 and result["funding"] == 0.0
    for rate in (0.001, -0.001):
        ledger = []
        result = simulate_taker(
            [_state(0), _state(50), _state(100)],
            [
                {"timestamp_ms": 0, "prediction": 0.02},
                {"timestamp_ms": 50, "prediction": 0.02},
            ],
            funding_forecasts=(FundingForecast(0, 1000000000, ((75000000, None),)),),
            prediction_column="prediction",
            prediction_horizon_ms=50,
            config=_taker_config(),
            observed_coverage={"s1": _coverage(0, 101)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(FundingEvent(75000000, rate, 100.0),),
            on_ledger=ledger.append,
        )
        assert (
            result["selection_status"] == "PARTIAL" and result["blocked_decisions"] == 1
        )
        assert result["funding"] == pytest.approx(-100 * rate)
        assert result["net_pnl"] == pytest.approx(-0.5 - 0.01005 - 100 * rate)
        assert any(
            row.text == "portfolio_expected_funding_missing"
            for batch in ledger
            for row in batch.diagnostics
        )


def test_taker_depth_breakpoint_optimum_zero_and_unavailable_decision():
    cases = [
        (
            [_state(0, ask_levels=((1005, 0.4), (1030, 10.0)))],
            0,
            0.02,
            0,
            0.4,
            "positive_expected_net_gain",
        ),
        ([_state(0)], 0, 0.0, 0, 0.0, "no_positive_expected_net_gain"),
        ([_state(5)], 0, 0.02, 10, 0.0, "decision_book_unavailable"),
        ([_state(0)], 0, 0.02, 100, 0.0, "forecast_expired_before_execution"),
    ]
    for books, stamp, prediction, latency, quantity, reason in cases:
        equity = []
        fills = []
        result = simulate_taker(
            books,
            [{"timestamp_ms": stamp, "prediction": prediction}],
            funding_forecasts=KNOWN_NO_FUNDING,
            prediction_column="prediction",
            prediction_horizon_ms=100,
            config=_taker_config(latency_ms=latency, order_quantity=2.0),
            observed_coverage={"s1": _coverage(0, 200)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=capture_fill_rows(fills.append, equity.append),
        )
        decisions = [
            {row.name: row for row in batch.diagnostics}
            for batch in equity
            if any(row.name == "decision_reason" for row in batch.diagnostics)
        ]
        if reason == "decision_book_unavailable":
            assert not decisions
        else:
            assert decisions[0]["decision_requested_quantity"].number == quantity
            assert decisions[0]["decision_reason"].text == reason
        assert result["selection_status"] == (
            "UNDEFINED"
            if reason
            in {"decision_book_unavailable", "forecast_expired_before_execution"}
            else "COMPLETE"
        )
        assert result["created_orders"] == int(quantity > 0)
        if quantity:
            assert fills[0]["requested_quantity"] == 0.4
            assert fills[0]["expected_net_gain"] == pytest.approx(
                0.4 * (102 - 100.5 - 0.01005)
            )


def test_taker_rejects_nonpositive_forecast_and_preserves_primary():
    for prediction in (-1.0, -2.0):
        with pytest.raises(ValueError, match="hft_prediction_price_invalid"):
            simulate_taker(
                [_state(0)],
                [{"timestamp_ms": 0, "prediction": prediction}],
                funding_forecasts=KNOWN_NO_FUNDING,
                prediction_column="prediction",
                prediction_horizon_ms=100,
                config=_taker_config(),
                observed_coverage={"s1": _coverage(0, 100)},
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=(),
            )
    primary = KeyboardInterrupt("decision_primary")
    closed = []
    calls = []

    def books():
        try:
            for stamp in (0, 100):
                yield _state(stamp)
        finally:
            closed.append(True)
            raise OSError("book_close_secondary")

    def emit(batch):
        calls.append(batch.sequence)
        if len(calls) == 2:
            raise primary

    with pytest.raises(KeyboardInterrupt) as raised:
        simulate_taker(
            books(),
            [{"timestamp_ms": stamp, "prediction": 0.02} for stamp in (0, 10)],
            funding_forecasts=KNOWN_NO_FUNDING,
            prediction_column="prediction",
            prediction_horizon_ms=100,
            config=_taker_config(latency_ms=30),
            observed_coverage={"s1": _coverage(0, 101)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=emit,
        )
    assert raised.value is primary and closed


def test_taker_terminal_mark_cannot_reuse_price_before_pending_effective_time():
    result = simulate_taker(
        [_state(0), _state(20)],
        [
            {"timestamp_ms": 0, "prediction": 0.02},
            {"timestamp_ms": 20, "prediction": 0.02},
        ],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_column="prediction",
        prediction_horizon_ms=100,
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": _coverage(0, 25)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["filled_orders"] == 1 and result["unfilled_quantity"] == 0.0
    assert result["censored_pending_command_count"] == 1
    assert result["censored_order_quantity"] == 1.0
    assert result["net_pnl"] is None and result["nav"] is None


@pytest.fixture(params=(1, 10, 100))
def coordinate_scale(request, monkeypatch):
    import sys
    from dataclasses import replace

    multiplier = request.param
    unit = {1: "0.1", 10: "0.01", 100: "0.001"}[multiplier]
    original_state, original_trade = (_state, _trade)

    def state(*args, **kwargs):
        value = original_state(*args, **kwargs)
        return replace(
            value,
            price_unit=unit,
            bids=tuple(((index * multiplier, qty) for index, qty in value.bids)),
            asks=tuple(((index * multiplier, qty) for index, qty in value.asks)),
        )

    def trade(*args, **kwargs):
        value = original_trade(*args, **kwargs)
        return replace(
            value, price_unit=unit, price_index=value.price_index * multiplier
        )

    monkeypatch.setattr(sys.modules[__name__], "_state", state)
    monkeypatch.setattr(sys.modules[__name__], "_trade", trade)


def test_observation_stop_does_not_rewrite_maker_decision_or_cancel_pending_quote():
    results = []
    for stop in (400000000, 1000000000):
        batches, fills = ([], [])
        result = simulate_maker(
            [_state(stamp) for stamp in (0, 350, 390, 500, 900)],
            [],
            [_quote(350, book_segment_id="s1")],
            config=replace(_maker_config(), latency_ms=100),
            observed_coverage={"s1": (0, 2000000001)},
            trade_coverage={"trade-connection-1": (0, 2000000001)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            decision_windows_ns=((0, 2000000000),),
            observation_window_ns=(0, stop),
            on_ledger=capture_fill_rows(fills.append, batches.append),
        )
        decisions = [
            batch
            for batch in batches
            if any(row.name.startswith("maker_") for row in batch.diagnostics)
        ]
        results.append((result, batches, decisions, fills))
    assert results[0][2] == results[1][2]
    assert results[0][0]["cancelled_orders"] == 0
    assert results[0][0]["censored_pending_command_count"] == 2
    assert results[0][0]["average_order_lifetime_ms"] is None
    assert not results[0][3]
    assert (
        max(row.timestamp_ns for batch in results[0][1] for row in batch.valuations)
        == 390000000
    )
    prefixes = [
        [
            batch
            for batch in rows
            if max(value.timestamp_ns for value in batch.valuations) <= 350000000
        ]
        for _result, rows, _equity, _fills in results
    ]
    assert len(prefixes[0]) >= 2
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize("kind", ["maker", "taker"])
def test_complete_observation_window_preserves_batches_without_fills(kind):
    produced = []
    for observed in (None, (0, 1000000001)):
        batches = []
        common = dict(
            observed_coverage={"s1": (0, 1000000001)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            decision_windows_ns=((0, 1000000000),),
            observation_window_ns=observed,
            on_ledger=batches.append,
        )
        books = [_state(stamp) for stamp in (0, 350, 500, 900, 1000)]
        if kind == "maker":
            simulate_maker(
                books,
                [],
                [_quote(350, book_segment_id="s1")],
                config=_maker_config(),
                trade_coverage={"trade-connection-1": (0, 1000000001)},
                **common,
            )
        else:
            simulate_taker(
                books,
                [],
                prediction_column="score",
                prediction_horizon_ms=500,
                config=TakerConfig(
                    order_quantity=1.0,
                    position_size=1.0,
                    latency_ms=100,
                    taker_fee_bps=0.0,
                    markout_horizon_ms=500,
                ),
                funding_forecasts=(),
                **common,
            )
        produced.append(batches)
    assert produced[0] == produced[1]


def test_pending_markout_fifo_preserves_maturity_order_with_pending_suffix():
    completed = []
    with _PendingMarkouts(
        _Metrics(), lambda fill: completed.append(fill["ordinal"])
    ) as pending:

        def add(index):
            pending.add(
                {
                    "ordinal": index,
                    "fill_qty": 1.0,
                    "fill_price": 100.0,
                    "side": "buy",
                    "segment_id": "s1",
                    "markout_target_time_ns": index,
                }
            )

        for index in range(10):
            add(index)
        for index in range(20):
            pending.advance(index, None, {})
            add(index + 10)
        assert completed == list(range(20))
        pending.expire_all()
        assert completed == list(range(30))


def test_fee_stress_keeps_representable_extreme_change():
    from fractions import Fraction

    row = fixed_turnover_cost_sensitivity(
        turnover=100.0,
        net_pnl=0.0,
        base_fee_bps=0.0,
        fee_scenarios_bps=(1e308,),
        additional_spread_scenarios_bps=(0.0,),
    )[0]
    expected = float(Fraction(100) * Fraction(1e308) / 10000)
    assert row["cost_delta"] == expected
    assert row["net_pnl"] == -expected


@pytest.mark.parametrize("close_price,realized", [(110.0, 10.0), (90.0, -10.0)])
def test_capture_fill_rows_keeps_delayed_identity_without_derived_fees(
    close_price, realized
):
    from quant_research_platform.markets.implementations.crypto.execution import (
        PortfolioEconomics,
    )

    fills, batches = [], []
    economics = PortfolioEconomics(
        PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        "btc_usdt",
        capture_fill_rows(fills.append, batches.append),
    )
    first = economics.fill(
        timestamp_ns=0,
        order_id="buy",
        quantity=1.0,
        price=100.0,
        mark_price=100.0,
        fee=1.0,
    )
    second = economics.fill(
        timestamp_ns=1,
        order_id="sell",
        quantity=-1.0,
        price=close_price,
        mark_price=close_price,
        fee=1.0,
    )
    economics.diagnostic({"fill_id": second, "markout": 2.0})
    economics.diagnostic({"fill_id": first, "markout": 1.0})
    assert [row["fill_id"] for row in fills] == [second, first]
    assert [row["markout"] for row in fills] == [2.0, 1.0]
    assert all("fee" not in row for row in fills)
    flows = [flow for batch in batches for flow in batch.cashflows]
    assert [flow.amount for flow in flows if flow.kind == "commission_cost"] == [
        -1.0,
        -1.0,
    ]
    assert [flow.amount for flow in flows if flow.kind == "realized_pnl"] == [
        0.0,
        realized,
    ]
    assert economics.mark(2, close_price)["net_pnl"] == realized - 2.0


def test_truncated_pending_cancels_survive_native_ledger_summary(tmp_path):

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
        write_portfolio_ledger_artifact,
    )

    capital = PortfolioCapital("pending-cancels", 1000.0, "USDT", "linear_perpetual")
    metadata = {
        "window": {
            "start": "1970-01-01",
            "end": "1970-01-02",
        }
    }
    results = []

    def produce(emit):
        result = simulate_maker(
            [_state(stamp) for stamp in (0, 100, 200, 240, 400)],
            [],
            [
                _quote(0, book_segment_id="s1"),
                _quote(200, book_segment_id="s1"),
            ],
            config=replace(_maker_config(), latency_ms=100),
            observed_coverage={"s1": (0, 1000000001)},
            trade_coverage={"trade-connection-1": (0, 1000000001)},
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
            decision_windows_ns=((0, 250000000),),
            observation_window_ns=(0, 250000000),
            on_ledger=emit,
        )
        results.append(result)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
    )
    saved = read_portfolio_ledger_artifact(root)
    summary = summarize_portfolio_ledger(saved.capital, saved.iter_chunks())
    assert results[0]["censored_pending_command_count"] == 2
    assert results[0]["cancelled_orders"] == results[0]["filled_orders"] == 0
    import json

    diagnostics = [
        json.loads(row.text)
        for batch in saved.iter_chunks()
        for row in batch.diagnostics
        if row.name == "execution_summary"
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0]["censored_pending_command_count"] == 2
    assert diagnostics[0]["cancelled_orders"] == diagnostics[0]["filled_orders"] == 0
    assert summary["net_pnl"] == 0.0
    assert not any(batch.trades or batch.cashflows for batch in saved.iter_chunks())


def _persist_execution(tmp_path, capital, produce):

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )

    metadata = {
        "window": {
            "start": pd.Timestamp(0).isoformat(),
            "end": pd.Timestamp(201_000_000).isoformat(),
        }
    }
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
    )
    return read_portfolio_ledger_artifact(root)


@pytest.mark.parametrize("quote_active", (False, True))
@pytest.mark.parametrize("visible", (None, 0.0, 3.0))
@pytest.mark.parametrize(
    "price,maker_buyer,relevant",
    (
        ("100.5", False, False),
        ("100.0", True, False),
        ("99.5", True, True),
        ("99.4", True, True),
    ),
)
def test_maker_unknown_visible_quantity_only_blocks_dependent_orders(
    tmp_path, quote_active, visible, price, maker_buyer, relevant
):
    from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger

    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
    results = []

    def produce(emit):
        results.append(
            simulate_maker(
                [_state(0), _state(200)],
                [
                    replace(
                        _trade(100, price=price, quantity="3", maker_buyer=maker_buyer),
                        visible_quantity=visible,
                    )
                ],
                [_quote(0, book_segment_id="s1")] if quote_active else [],
                # Only the buy side is quoted at initial inventory zero.
                config=replace(
                    _maker_config(), inventory_target=2.0, inventory_skew=1.0
                ),
                observed_coverage={"s1": _coverage(0, 201)},
                trade_coverage={"trade-connection-1": _coverage(0, 201)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                decision_windows_ns=((0, 200_000_000),),
                on_ledger=emit,
            )
        )

    artifact = _persist_execution(tmp_path, capital, produce)
    summary = summarize_crypto_ledger(
        artifact, artifact.iter_chunks(), windows_ns=((0, 200_000_000),)
    )
    expected_partial = quote_active and relevant and visible is None
    status = "PARTIAL" if expected_partial else "COMPLETE"
    assert results[0]["created_orders"] == int(quote_active)
    assert results[0]["account_path_status"] == status
    assert summary["execution"]["account_path_status"] == status
    if expected_partial:
        assert results[0]["account_stop_ns"] == 100_000_000
        assert results[0]["net_pnl"] is None
        assert summary["execution"]["account_stop_reason"] == (
            "hft_trade_visible_quantity_unknown"
        )
    else:
        expected_pnl = 0.50995 if quote_active and relevant and visible else 0.0
        assert results[0]["net_pnl"] == pytest.approx(expected_pnl)
        assert summary["economics"]["net_pnl"] == pytest.approx(expected_pnl)
        assert summary["execution"]["account_stop_reason"] is None


@pytest.mark.parametrize("quantity,unit", [(1.0, "0.001"), (10.0, "0.01")])
def test_taker_weighted_diagnostics_survive_unit_scaling(tmp_path, quantity, unit):
    import json
    from decimal import Decimal
    from fractions import Fraction

    from quant_research_platform.sources.binance.replay import (
        DepthReconstructor,
        ReconstructionStats,
    )

    config = TakerConfig(
        order_quantity=quantity,
        position_size=quantity,
        latency_ms=0,
        taker_fee_bps=0.0,
        markout_horizon_ms=100,
    )
    identity = {
        "market_type": FUTURES_MARKET_TYPE,
        "segment_id": "s1",
        "observation_id": "connection-s1",
    }
    tick = Decimal(unit)
    engine = DepthReconstructor(
        FUTURES_MARKET_TYPE,
        price_unit=unit,
        snapshot_limit=10,
        required_levels=10,
        connection_close_ns=201_000_000,
        stats=ReconstructionStats(),
        output_levels=10,
    )
    engine.bootstrap(
        {
            **identity,
            "payload": {
                "lastUpdateId": 100,
                "bids": [
                    [str((995 - index) * tick), str(quantity if index == 0 else 1)]
                    for index in range(10)
                ],
                "asks": [
                    [str((1005 + index) * tick), str(quantity if index == 0 else 1)]
                    for index in range(10)
                ],
            },
        }
    )
    books = []
    for sequence, timestamp in enumerate((0, 200), 1):
        books.extend(
            engine.feed(
                {
                    **identity,
                    "source_sequence": sequence,
                    "observed_time_ns": timestamp * 1_000_000,
                    "receive_time": pd.Timestamp(
                        timestamp, unit="ms", tz="UTC"
                    ).isoformat(),
                    "receive_monotonic_ns": sequence,
                    "payload": {
                        "U": 99 + sequence,
                        "u": 100 + sequence,
                        "pu": 99 + sequence,
                        "E": timestamp,
                        "T": timestamp,
                        "b": [],
                        "a": [],
                    },
                }
            )
        )
    assert len(books) == 2 and all(isinstance(book, BookState) for book in books)
    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
    results, fills = [], []

    def produce(emit):
        results.append(
            simulate_taker(
                books,
                [{"timestamp_ms": 0, "prediction": 0.1}],
                prediction_column="prediction",
                prediction_horizon_ms=500,
                config=config,
                observed_coverage={"s1": _coverage(0, 201)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=KNOWN_NO_FUNDING,
                on_ledger=capture_fill_rows(fills.append, emit),
            )
        )

    saved = _persist_execution(tmp_path, capital, produce)
    (fill,) = fills
    price, mid = (Fraction(float(index * tick)) for index in (1005, 1000))
    notional = Fraction(quantity) * price
    pnl = Fraction(quantity) * (mid - price)
    expected = {
        "average_slippage_bps": float((price - mid) / mid * 10000),
        "average_markout_bps": float((mid - price) / price * 10000),
    }
    assert fill["fill_qty"] == quantity
    assert fill["notional"] == float(notional)
    assert results[0]["gross_pnl"] == pytest.approx(float(pnl), rel=1e-12)
    persisted = [
        json.loads(row.text)
        for batch in saved.iter_chunks()
        for row in batch.diagnostics
        if row.name == "execution_summary"
    ]
    assert len(persisted) == 1
    for name, value in expected.items():
        assert math.isfinite(value)
        assert results[0][name] == pytest.approx(value, rel=1e-12)
        assert persisted[0][name] == results[0][name]


def test_taker_weighted_diagnostics_use_each_fill_quantity(tmp_path):
    import json
    from fractions import Fraction

    books = [
        _state(0, ask_levels=((1005, 3.0),)),
        _state(50, bid_index=1005, ask_levels=((1015, 3.0),)),
        _state(200, bid_index=1005, ask_levels=((1015, 3.0),)),
    ]
    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
    fills = []

    def produce(emit):
        simulate_taker(
            books,
            [{"timestamp_ms": stamp, "prediction": 0.1} for stamp in (0, 50)],
            prediction_column="prediction",
            prediction_horizon_ms=500,
            config=replace(
                _taker_config(order_quantity=2.0), position_size=3.0, taker_fee_bps=0.0
            ),
            observed_coverage={"s1": _coverage(0, 201)},
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
            funding_forecasts=KNOWN_NO_FUNDING,
            on_ledger=capture_fill_rows(fills.append, emit),
        )

    saved = _persist_execution(tmp_path, capital, produce)
    assert [fill["fill_qty"] for fill in fills] == [2.0, 1.0]
    summary = next(
        json.loads(row.text)
        for batch in saved.iter_chunks()
        for row in batch.diagnostics
        if row.name == "execution_summary"
    )
    for name, values in (
        ("average_slippage_bps", (Fraction(1, 200), Fraction(1, 202))),
        ("average_markout_bps", (Fraction(1, 201), Fraction(-1, 203))),
    ):
        expected = float((2 * values[0] + values[1]) / 3 * 10000)
        assert summary[name] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("last_value", (9.0, -9.0))
@pytest.mark.parametrize("first_value,quantity", ((90.0, 1000.0), (9e16, 1e16)))
def test_weighted_diagnostics_preserve_unequal_quantity_contributions(
    reverse, last_value, first_value, quantity
):
    from fractions import Fraction

    pairs = [(1.0, first_value), (quantity, last_value)]
    if reverse:
        pairs.reverse()
    metrics = _Metrics()
    for quantity, value in pairs:
        fill = {
            "fill_qty": quantity,
            "notional": 0.0,
            "slippage_bps": value,
            "markout_per_unit": 0.0,
            "markout_bps": value,
            "markout_pnl": 0.0,
            "adverse_selection_cost": 0.0,
        }
        metrics.observe_fill(fill)
        metrics.finalize_markout(fill)
    expected = float(
        sum(Fraction(q) * Fraction(value) for q, value in pairs)
        / sum(Fraction(q) for q, _ in pairs)
    )
    assert metrics.payload()["average_slippage_bps"] == pytest.approx(
        expected, rel=1e-12
    )
    assert metrics.payload()["average_markout_bps"] == pytest.approx(
        expected, rel=1e-12
    )


@pytest.mark.parametrize("values", [(1e308, -1e308), (1e308, 1e308)])
def test_weighted_diagnostic_update_avoids_intermediate_overflow(values):
    from fractions import Fraction

    metrics = _Metrics()
    for quantity, value in zip((1.0, 3.0), values, strict=True):
        fill = {
            "fill_qty": quantity,
            "notional": 0.0,
            "slippage_bps": value,
            "markout_per_unit": 0.0,
            "markout_bps": value,
            "markout_pnl": 0.0,
            "adverse_selection_cost": 0.0,
        }
        metrics.observe_fill(fill)
        metrics.finalize_markout(fill)
    expected = float((Fraction(values[0]) + 3 * Fraction(values[1])) / 4)
    assert metrics.payload()["average_slippage_bps"] == expected
    assert metrics.payload()["average_markout_bps"] == expected


@pytest.mark.parametrize("sign", (1, -1))
@pytest.mark.parametrize(
    "requested,depth",
    [
        (1e16, (1.0,)),
        (1e13, (0.001,)),
        (1e16, (1.0, 1e16)),
        (1e16, (3.0, 1e16)),
        (1.0, (0.4, 0.6)),
    ],
)
def test_taker_actual_depth_survives_large_frozen_order_and_roundtrip(
    tmp_path, sign, requested, depth
):
    from fractions import Fraction

    from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger

    tick = 1005 if sign > 0 else 995
    levels = tuple((tick + sign * 5 * i, q) for i, q in enumerate(depth))
    side = "ask_levels" if sign > 0 else "bid_levels"
    books = [
        _state(0, **{side: ((tick, requested),)}),
        _state(5, **{side: levels}),
        _state(200, **{side: levels}),
    ]
    capital = PortfolioCapital("fixture", 1e20, "USDT", "linear_perpetual")
    config = replace(
        _taker_config(latency_ms=10, order_quantity=requested),
        position_size=requested,
    )
    fills = []
    results = []

    def produce(emit):
        observer = capture_fill_rows(fills.append, emit)
        results.append(
            simulate_taker(
                books,
                [{"timestamp_ms": 0, "prediction": sign * 0.1}],
                prediction_column="prediction",
                prediction_horizon_ms=500,
                config=config,
                observed_coverage={"s1": _coverage(0, 201)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=KNOWN_NO_FUNDING,
                on_ledger=observer,
            )
        )

    artifact = _persist_execution(tmp_path, capital, produce)
    remainder = Fraction(requested)
    expected_takes = []
    for index, available in levels:
        take = min(remainder, Fraction(available))
        assert 0 <= take <= Fraction(available)
        expected_takes.append((take, Fraction(index, 10)))
        remainder -= take
    total = sum(q for q, _ in expected_takes)
    notional = sum(q * price for q, price in expected_takes)
    assert total <= Fraction(requested)
    (fill,) = fills
    assert fill["requested_quantity"] == requested
    assert fill["fill_qty"] == total
    assert fill["notional"] == float(notional)
    assert fill["fill_price"] == float(notional / total)
    assert (
        min(float(p) for _, p in expected_takes)
        <= fill["fill_price"]
        <= max(float(p) for _, p in expected_takes)
    )
    if len(levels) == 1:
        assert fill["fill_price"] == float(expected_takes[0][1])
    fee = float(notional * Fraction(config.taker_fee_bps) / 10000)
    assert results[0]["unfilled_quantity"] == float(remainder)
    batches = list(artifact.iter_chunks())
    trades = [trade for batch in batches for trade in batch.trades]
    assert len(trades) == 1
    assert trades[0].quantity == sign * total
    cash = [
        flow.amount
        for batch in batches
        for flow in batch.cashflows
        if flow.kind == "commission_cost"
    ]
    assert cash == [-fee]
    economics = summarize_crypto_ledger(artifact, artifact.iter_chunks())["economics"]
    # Use actual depth consideration, before the diagnostic VWAP is rounded.
    expected_pnl = float(sign * (total * Fraction(100) - notional) - Fraction(fee))
    assert economics["net_pnl"] == pytest.approx(expected_pnl, rel=2e-15, abs=1e-15)


@pytest.mark.parametrize("fee_bps", (-1.0, 1.0))
def test_maker_fee_rebate_roundtrip_keeps_representable_result(tmp_path, fee_bps):
    from fractions import Fraction

    from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger

    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
    fills = []

    def produce(emit):
        simulate_maker(
            [_state(0), _state(200)],
            [_trade(50, price="99.5", quantity="3", maker_buyer=True)],
            [_quote(0, book_segment_id="s1")],
            config=replace(_maker_config(), maker_fee_bps=fee_bps),
            observed_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
            capital=capital,
            asset_id="btc_usdt",
            decision_windows_ns=((0, 200_000_000),),
            funding_events=(),
            on_ledger=capture_fill_rows(fills.append, emit),
        )

    artifact = _persist_execution(tmp_path, capital, produce)
    (fill,) = fills
    expected_fee = float(Fraction(99.5) * Fraction(fee_bps) / 10000)
    batches = list(artifact.iter_chunks())
    assert [
        f.amount for b in batches for f in b.cashflows if f.kind == "commission_cost"
    ] == [-expected_fee]
    economics = summarize_crypto_ledger(artifact, artifact.iter_chunks())["economics"]
    assert economics["net_pnl"] == float(Fraction(0.5) - Fraction(expected_fee))


def test_execution_fee_rejects_truly_unrepresentable_result():
    from quant_research_platform.markets.implementations.crypto.execution import (
        _execution_fee,
    )

    with pytest.raises(ValueError, match="hft_execution_fee_unrepresentable"):
        _execution_fee(1e308, 100.0, 1e308)


@pytest.mark.parametrize("quantity,scale", [(1.0, 1), (0.001, 1000)])
@pytest.mark.parametrize("keep_existing", (False, True))
def test_calibrated_maker_rebate_keeps_complete_gain(quantity, scale, keep_existing):
    from fractions import Fraction

    from quant_research_platform.markets.implementations.crypto.execution import (
        _maker_join,
        _maker_new_order,
    )
    from tests.markets.crypto.calibration_test_support import calibration_from_paths

    state = _state(
        0,
        bid_levels=((995 * scale, quantity), (990 * scale, 3 * quantity)),
        ask_levels=((1005 * scale, quantity), (1010 * scale, 2 * quantity)),
    )
    trade = _trade(
        100, price=str(99.5 * scale), quantity=str(2 * quantity), maker_buyer=True
    )
    model = calibration_from_paths([([state], [trade])], [0.0])
    quote = _quote(0, book_segment_id="s1")
    order = _maker_new_order(2, "buy", quantity, state, quote, 0)
    active = (
        _maker_new_order(1, "buy", quantity, state, quote, 0) if keep_existing else None
    )
    if active is not None:
        assert (
            _maker_join(
                active,
                state,
                {"s1": _coverage(0, 2000)},
                {"trade-connection-1": _coverage(0, 2000)},
            )
            is None
        )
    keep, selected, gain = model.select(
        state=state,
        decision_ns=0,
        new_order=order,
        active_order=active,
        commands=(),
        lifetime_ns=500_000_000,
        prediction=0.0,
        fee_bps=-1.0,
        funding=[],
    )
    exact = Fraction(quantity) * (
        Fraction(100 * scale)
        - Fraction(99.5 * scale)
        - Fraction(99.5 * scale) * Fraction(-1.0) / 10000
    )
    assert keep is keep_existing
    assert selected == (0.0 if keep_existing else quantity)
    assert gain == pytest.approx(float(exact), rel=2e-15, abs=1e-15)


@pytest.mark.parametrize("quantity,scale", [(1.0, 1), (0.001, 1000)])
def test_calibrated_maker_rebate_survives_native_ledger(tmp_path, quantity, scale):
    from fractions import Fraction

    from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
    from tests.markets.crypto.calibration_test_support import calibration_from_paths

    state = _state(
        0,
        bid_levels=((995 * scale, quantity), (990 * scale, 3 * quantity)),
        ask_levels=((1005 * scale, quantity), (1010 * scale, 2 * quantity)),
    )
    trade = _trade(
        100, price=str(99.5 * scale), quantity=str(2 * quantity), maker_buyer=True
    )
    model = calibration_from_paths([([state], [trade])], [0.0])
    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")

    def produce(emit):
        simulate_maker(
            [state, replace(state, observed_time_ns=200_000_000, source_sequence=201)],
            [trade],
            [
                {
                    **_quote(0, book_segment_id="s1"),
                    "fold_id": "fold",
                    "price_prediction": 0.0,
                    "proxy_prediction": 1.0,
                }
            ],
            config=replace(
                _maker_config(calibrated=True),
                maker_fee_bps=-1.0,
                order_quantity=quantity,
                max_position=2 * quantity,
            ),
            observed_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
            funding_forecasts=KNOWN_NO_FUNDING,
            calibrations={"fold": model},
            on_ledger=emit,
        )

    artifact = _persist_execution(tmp_path, capital, produce)
    expected_fee = float(
        Fraction(quantity) * Fraction(99.5 * scale) * Fraction(-1.0) / 10000
    )
    batches = list(artifact.iter_chunks())
    assert [trade.quantity for batch in batches for trade in batch.trades] == [quantity]
    assert [
        f.amount for b in batches for f in b.cashflows if f.kind == "commission_cost"
    ] == [-expected_fee]
    economics = summarize_crypto_ledger(artifact, artifact.iter_chunks())["economics"]
    assert economics["net_pnl"] == pytest.approx(
        float(Fraction(quantity) * Fraction(0.5 * scale) - Fraction(expected_fee)),
        rel=2e-15,
        abs=1e-15,
    )


def test_maker_final_unrepresentable_fee_fails_at_execution_boundary():
    with pytest.raises(ValueError, match="hft_execution_fee_unrepresentable"):
        simulate_maker(
            [_state(0), _state(200)],
            [_trade(50, price="99.0", quantity="1", maker_buyer=True)],
            [_quote(0, book_segment_id="s1")],
            config=replace(
                _maker_config(),
                order_quantity=1e308,
                max_position=1e308,
                maker_fee_bps=1e308,
            ),
            observed_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
        )


@pytest.mark.parametrize(
    "fee,expected", [(-2.0, 6.0), (-1.0, 5.0), (0.0, 4.0), (1.0, 3.0)]
)
def test_cost_sensitivity_accepts_signed_maker_rebates(fee, expected):
    row = fixed_turnover_cost_sensitivity(
        turnover=10000.0,
        net_pnl=5.0,
        base_fee_bps=-1.0,
        fee_scenarios_bps=(fee,),
        additional_spread_scenarios_bps=(0.0,),
    )[0]
    assert row["net_pnl"] == pytest.approx(expected)


def test_cost_sensitivity_still_rejects_negative_additional_spread():
    with pytest.raises(ValueError, match="hft_cost_sensitivity_invalid"):
        fixed_turnover_cost_sensitivity(
            turnover=10000.0,
            net_pnl=5.0,
            base_fee_bps=-1.0,
            fee_scenarios_bps=(-2.0,),
            additional_spread_scenarios_bps=(-1.0,),
        )


@pytest.mark.parametrize("missing", ["book", "trade", "overlap"])
def test_maker_source_context_requires_unique_observable_coverage(missing):
    coverage = {"actual-trade": (0, 300_000_000)}
    if missing == "trade":
        coverage = {"actual-trade": (1, 300_000_000)}
    elif missing == "overlap":
        coverage["overlapping-trade"] = (0, 300_000_000)
    result = simulate_maker(
        [
            _state(stamp, segment_id="actual-book")
            for stamp in ((100, 200) if missing == "book" else (0, 200))
        ],
        [],
        [
            _quote(
                0,
                book_segment_id="independent-input",
                trade_observation_id="independent-input",
            )
        ],
        config=_maker_config(),
        observed_coverage={"actual-book": (0, 300_000_000)},
        trade_coverage=coverage,
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["created_orders"] == 0
    assert result["undefined_decisions"] == 1


@pytest.mark.parametrize("visible_quantity", [0.0, 3.0])
@pytest.mark.parametrize(
    "quote_active,maker_buyer", [(False, True), (True, False), (True, True)]
)
def test_maker_unknown_trade_price_blocks_only_dependent_order(
    quote_active, maker_buyer, visible_quantity
):
    result = simulate_maker(
        [_state(0), _state(200)],
        [
            replace(
                _trade(100, price="99.5", quantity="3", maker_buyer=maker_buyer),
                price_index=None,
                visible_quantity=visible_quantity,
            )
        ],
        [_quote(0, book_segment_id="s1")] if quote_active else [],
        config=replace(_maker_config(), inventory_target=2.0, inventory_skew=1.0),
        observed_coverage={"s1": _coverage(0, 201)},
        trade_coverage={"trade-connection-1": _coverage(0, 201)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        decision_windows_ns=((0, 200_000_000),),
        on_ledger=lambda batch: None,
    )
    if quote_active and maker_buyer and visible_quantity:
        assert result["account_path_status"] == "PARTIAL"
        assert result["account_stop_reason"] == "hft_trade_price_unknown"
        assert result["net_pnl"] is None
    else:
        assert result["account_path_status"] == "COMPLETE"
        assert result["net_pnl"] == 0.0


def test_known_zero_visible_trade_needs_no_price():
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.execution import (
        _maker_trade_fill,
    )

    trade = replace(
        _trade(100, price="99.5", quantity="3", maker_buyer=True),
        price_index=None,
        visible_quantity=0.0,
    )
    order = SimpleNamespace(trade_observation_id=trade.observation_id, side="buy")
    assert _maker_trade_fill(order, trade) == (0.0, "zero_visible_trade_quantity")

import math
from dataclasses import replace
from fractions import Fraction

import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.markets.implementations.crypto.execution import (
    TakerConfig,
    _position_headroom,
    simulate_maker,
    simulate_taker,
)
from tests.helpers.binance_hft import KNOWN_NO_FUNDING, _coverage, _quote
from tests.helpers.binance_hft import execution_state as _state
from tests.helpers.binance_hft import execution_trade as _trade
from tests.helpers.binance_hft import maker_config as _maker_config


@pytest.mark.parametrize("direction", (-1, 1))
@pytest.mark.parametrize("latency", (0, 300))
def test_taker_exact_headroom_never_exceeds_position_cap(tmp_path, direction, latency):
    capital = PortfolioCapital("exact_headroom", 1000.0, "USDT", "linear_perpetual")
    limit = 1e16 + 2
    results = []

    def produce(emit):
        books = [
            _state(
                stamp,
                **{
                    "ask_levels" if direction == 1 else "bid_levels": (
                        (1005 if direction == 1 else 995, quantity),
                    )
                },
            )
            for stamp, quantity in (
                (0, 1e16),
                (100, 1.0),
                (200, 2.0),
                *((((300, 1e16), (400, 1.0), (500, 2.0))) if latency else ()),
                (600, 2.0),
            )
        ]
        results.append(
            simulate_taker(
                books,
                [
                    {"timestamp_ms": stamp, "prediction": direction * 0.1}
                    for stamp in (0, 100, 200)
                ],
                prediction_column="prediction",
                prediction_horizon_ms=1000,
                config=TakerConfig(1e16, limit, latency, 0.0, 100),
                observed_coverage={"s1": _coverage(0, 701)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=KNOWN_NO_FUNDING,
                on_ledger=emit,
            )
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    fills = [
        trade.quantity for batch in artifact.iter_chunks() for trade in batch.trades
    ]
    assert fills == [direction * Fraction(1e16), direction, direction]
    assert abs(sum(fills)) == Fraction(limit)
    assert results[0]["max_abs_position"] == limit
    assert type(results[0]["max_abs_position"]) is float


@pytest.mark.parametrize("direction", (-1, 1))
def test_maker_partial_order_keeps_exact_remaining_capacity(tmp_path, direction):
    capital = PortfolioCapital("exact_maker", 1000.0, "USDT", "linear_perpetual")
    limit = 1e16
    maker_buyer = direction == 1

    def produce(emit):
        return simulate_maker(
            [_state(0), _state(300)],
            [
                _trade(
                    50,
                    price="99.5" if maker_buyer else "100.5",
                    quantity="3" if maker_buyer else "2",
                    maker_buyer=maker_buyer,
                ),
                _trade(
                    150,
                    price="99.4" if maker_buyer else "100.6",
                    quantity="1",
                    maker_buyer=maker_buyer,
                ),
            ],
            [_quote(0, book_segment_id="s1")],
            config=replace(
                _maker_config(),
                order_quantity=limit,
                max_position=limit,
                maker_fee_bps=0,
            ),
            observed_coverage={"s1": _coverage(0, 301)},
            trade_coverage={"trade-connection-1": _coverage(0, 301)},
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=emit,
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    fills = [
        trade.quantity for batch in artifact.iter_chunks() for trade in batch.trades
    ]
    assert fills == [direction, direction * (Fraction(limit) - 1)]
    assert abs(sum(fills)) == Fraction(limit)


@pytest.mark.parametrize("direction", (-1, 1))
def test_maker_requote_uses_exact_position_headroom(tmp_path, direction):
    capital = PortfolioCapital("maker_headroom", 1000.0, "USDT", "linear_perpetual")
    limit = 1e16 + 2
    maker_buyer = direction == 1

    def produce(emit):
        return simulate_maker(
            [_state(stamp) for stamp in (0, 100, 200, 300)],
            [
                _trade(
                    stamp,
                    price=("99.5" if maker_buyer else "100.5")
                    if stamp == 150
                    else ("99.4" if maker_buyer else "100.6"),
                    quantity=("3" if maker_buyer else "2") if stamp == 150 else "1",
                    maker_buyer=maker_buyer,
                )
                for stamp in (50, 150, 250)
            ],
            [_quote(stamp, book_segment_id="s1") for stamp in (0, 100, 200)],
            config=replace(
                _maker_config(),
                order_quantity=1e16,
                max_position=limit,
                maker_fee_bps=0,
            ),
            observed_coverage={"s1": _coverage(0, 301)},
            trade_coverage={"trade-connection-1": _coverage(0, 301)},
            capital=capital,
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=emit,
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(root)
    fills = [
        trade.quantity for batch in artifact.iter_chunks() for trade in batch.trades
    ]
    assert fills == [direction * Fraction(1e16), direction, direction]
    assert abs(sum(fills)) == Fraction(limit)


@pytest.mark.parametrize("direction", (-1, 1))
def test_maker_queue_depletion_preserves_small_observations(direction):
    maker_buyer = direction == 1
    level = 995 if maker_buyer else 1005
    side = "bid_levels" if maker_buyer else "ask_levels"
    batches = []
    simulate_maker(
        [_state(0, **{side: ((level, 1e16),)}), _state(300)],
        [
            _trade(
                stamp,
                price="99.5" if maker_buyer else "100.5",
                quantity=quantity,
                maker_buyer=maker_buyer,
            )
            for stamp, quantity in ((50, "1"), (150, "1e16"))
        ],
        [_quote(0, book_segment_id="s1")],
        config=replace(
            _maker_config(), order_quantity=5, max_position=5, maker_fee_bps=0
        ),
        observed_coverage={"s1": _coverage(0, 301)},
        trade_coverage={"trade-connection-1": _coverage(0, 301)},
        capital=PortfolioCapital("exact_queue", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        on_ledger=batches.append,
    )
    fills = [trade.quantity for batch in batches for trade in batch.trades]
    assert fills == [direction]


@pytest.mark.parametrize(
    "limit,position,reserved",
    (
        (1e16, Fraction(1), Fraction()),
        (1e16, Fraction(), Fraction(1)),
        (1e16 + 2, Fraction(10**16 + 1), Fraction()),
        (1e16 + 2, Fraction(10**16), Fraction(1)),
        (1e308, -Fraction(1e308), Fraction()),
    ),
)
def test_float_headroom_is_maximal_without_rounding_above_exact_capacity(
    limit, position, reserved
):
    capacity = Fraction(limit) - position - reserved
    actual = _position_headroom(limit, position, reserved)
    assert math.isfinite(actual)
    assert Fraction(actual) <= capacity
    next_value = math.nextafter(actual, math.inf)
    assert math.isinf(next_value) or Fraction(next_value) > capacity


@pytest.mark.parametrize(
    "target,skew", ((0.0, None), (None, 0.0), (0.0, 0.0), (1.0, 0.5))
)
def test_calibrated_maker_rejects_unconsumed_inventory_parameters(target, skew):
    with pytest.raises(
        ValueError, match="hft_maker_calibrated_inventory_parameters_invalid"
    ):
        simulate_maker(
            [],
            [],
            [],
            config=replace(
                _maker_config(), inventory_target=target, inventory_skew=skew
            ),
            observed_coverage={},
            trade_coverage={},
            capital=PortfolioCapital("calibrated", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            calibrations={},
        )


@pytest.mark.parametrize("direction", (-1, 1))
def test_taker_multilevel_notional_and_pnl_round_once_at_native_boundary(
    tmp_path, direction
):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
        write_portfolio_ledger_artifact,
    )

    side = "ask_levels" if direction == 1 else "bid_levels"
    first, second = (1005, 1010) if direction == 1 else (995, 990)
    batches = []
    capital = PortfolioCapital("exact_depth_money", 1000.0, "USDT", "linear_perpetual")
    simulate_taker(
        [
            _state(0, **{side: ((first, 1e16 + 2),)}),
            _state(5, **{side: ((first, 1e16), (second, 1.0))}),
            _state(200),
        ],
        [{"timestamp_ms": 0, "prediction": direction * 0.1}],
        prediction_column="prediction",
        prediction_horizon_ms=500,
        config=TakerConfig(1e16 + 2, 1e16 + 2, 10, 0.0, 100),
        observed_coverage={"s1": _coverage(0, 201)},
        capital=capital,
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=KNOWN_NO_FUNDING,
        on_ledger=batches.append,
    )
    expected_notional = Fraction(10**16) * Fraction(first, 10) + Fraction(second, 10)
    trades = [trade for batch in batches for trade in batch.trades]
    assert len(trades) == 1
    trade = trades[0]
    assert trade.quantity == direction * (10**16 + 1)
    final = next(
        row
        for row in batches[-1].valuations
        if row.asset_id == "btc_usdt" and row.phase == "end"
    )
    assert final.quantity == trade.quantity
    assert trade.notional == float(direction * expected_notional)
    expected_pnl = float(direction * (Fraction(10**16 + 1) * 100 - expected_notional))
    output = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    artifact = read_portfolio_ledger_artifact(output)
    assert (
        summarize_portfolio_ledger(capital, artifact.iter_chunks())["net_pnl"]
        == expected_pnl
    )

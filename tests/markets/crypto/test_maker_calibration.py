from dataclasses import replace

import numpy as np
import pytest

from quant_research_platform.contracts.portfolios import (
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    replay_maker_calibration,
    simulate_maker,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.binance_hft import _quote
from tests.helpers.runtime_resources import capture_test_policy
from tests.markets.crypto.test_hft_execution import _maker_config, _state, _trade


def test_many_known_funding_offsets_complete_under_one_small_query_budget():
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2, explicit_cap_bytes=64 * 1024
    )
    with runtime_resource_policy_scope(policy):
        model = MakerCalibration(
            np.array([1.0, 0.5]),
            np.array([-0.002, 0.003]),
            np.array([[0.0, 10_000_000.0, 1.0], [1.0, 20_000_000.0, 0.5]]),
            resource_policy=policy,
        )
        try:
            for index in range(2048):
                elapsed = (5_000_000, 15_000_000, 25_000_000)[index % 3] + index
                quantity, gain = model.select(
                    limit=1.0,
                    side="buy",
                    mid=100.0,
                    price=100.0,
                    prediction=0.0,
                    fee_bps=1.0,
                    lifetime_ns=500_000_000,
                    funding=[(elapsed, 0.02)],
                )
                assert quantity == 0.5
                assert gain == pytest.approx((0.02, 0.025, 0.03)[index % 3])
        finally:
            model.close()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("include_zero", (False, True))
def test_paired_fill_error_curve_has_interior_optimum_and_keeps_zero_samples(
    include_zero,
):
    quantities = np.array([1.0, 0.5, 0.0] if include_zero else [1.0, 0.5])
    errors = np.array([-0.002, 0.003, 10.0] if include_zero else [-0.002, 0.003])
    fills = np.array([[0.0, 10_000_000.0, 1.0], [1.0, 20_000_000.0, 0.5]])
    with runtime_resource_policy_scope() as policy:
        model = MakerCalibration(quantities, errors, fills, resource_policy=policy)
        try:
            quantity, gain = model.select(
                limit=1.0,
                side="buy",
                mid=100.0,
                price=100.0,
                prediction=0.0,
                fee_bps=1.0,
                lifetime_ns=500_000_000,
                funding=[],
            )
            assert quantity == 0.5
            assert gain == pytest.approx(0.04 / len(quantities))
            funded_quantity, _gain = model.select(
                limit=1.0,
                side="buy",
                mid=100.0,
                price=100.0,
                prediction=0.0,
                fee_bps=1.0,
                lifetime_ns=500_000_000,
                funding=[(15_000_000, 1.0)],
            )
            assert funded_quantity == 1.0
        finally:
            model.close()
        assert policy.used_memory_bytes == 0


def test_delayed_post_only_rejection_preserves_decision_quantity_and_zero_sample():
    config = replace(_maker_config(), latency_ms=10)
    coverage = {"s1": (0, 600_000_000)}
    trades_coverage = {"trade-connection-1": (0, 600_000_000)}
    trade = _trade(20, price="99.5", quantity="3", maker_buyer=True)
    row = {
        **_quote(0, book_segment_id="s1"),
        "price_prediction": 0.0,
        "price_label": 0.003,
        "proxy_prediction": 1.0,
        "label_available_at_ns": 500_000_000,
    }
    chosen = []
    for crossing in (False, True):
        books = [
            _state(0),
            _state(5, bid_tick=980, ask_tick=990) if crossing else _state(5),
            _state(500, bid_tick=998, ask_tick=1008),
        ]
        batches, observations = [], []
        with runtime_resource_policy_scope() as policy:
            result = simulate_maker(
                books,
                [trade],
                [_quote(0, book_segment_id="s1")],
                config=config,
                observed_coverage=coverage,
                trade_coverage=trades_coverage,
                resource_policy=policy,
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=(),
                decision_windows_ns=((0, 500_000_000),),
                on_ledger=batches.append,
            )
            replay_maker_calibration(
                books,
                [trade],
                [row],
                config=config,
                observed_coverage=coverage,
                trade_coverage=trades_coverage,
                cutoff_ns=500_000_000,
                resource_policy=policy,
                on_sample=lambda observations=observations, **sample: (
                    observations.append({**sample, "fills": tuple(sample["fills"])})
                ),
            )
            assert policy.used_memory_bytes == 0
        chosen.append(
            next(
                item.number
                for batch in batches
                for item in batch.diagnostics
                if item.name == "maker_buy_requested_quantity"
            )
        )
        buy = next(sample for sample in observations if sample["side"] == "buy")
        assert buy["filled"] == (0.0 if crossing else 1.0)
        assert result["post_only_rejections"] == int(crossing)
        if crossing:
            assert buy["terminal_reason"] == "post_only_crossing_rejected"
    assert chosen == [1.0, 1.0]


def test_reused_queue_cannot_renew_original_prediction_expiry():
    batches = []
    with runtime_resource_policy_scope() as policy:
        result = simulate_maker(
            [_state(0), _state(500), _state(600)],
            [
                _trade(499, price="99.5", quantity="1", maker_buyer=True),
                _trade(501, price="99.5", quantity="3", maker_buyer=True),
            ],
            [_quote(stamp, book_segment_id="s1") for stamp in (0, 100, 200, 300, 400)],
            config=_maker_config(),
            observed_coverage={"s1": (0, 600_000_000)},
            trade_coverage={"trade-connection-1": (0, 600_000_000)},
            resource_policy=policy,
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=batches.append,
        )
        assert result["created_orders"] == 2 and result["filled_orders"] == 0
        assert result["average_order_lifetime_ms"] == 500.0
        assert any(
            row.text == "keep_queue_without_renewal"
            for batch in batches
            for row in batch.diagnostics
        )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "pending, keep_wins", ((False, False), (True, False), (False, True))
)
def test_calibrated_maker_compares_old_expiry_with_real_cancel_replace(
    pending, keep_wins
):
    # New q=1 earns .5; the old remaining TTL excludes the 400ms empirical fill.
    decision_ms = 100 if pending else 200
    trade_schedule = (
        [(350, "3"), (450, "3")] if pending else [(100, "1"), (350, "2"), (650, "1")]
    )
    fills, batches = [], []
    with runtime_resource_policy_scope() as policy:
        model = MakerCalibration(
            np.array([1.0, 0.5] if keep_wins else [1.0]),
            np.array([-0.0021, -0.0071] if keep_wins else [0.0]),
            np.array(
                [[0.0, 100_000_000.0, 1.0], [1.0, 400_000_000.0, 0.5]]
                if keep_wins
                else [[0.0, 400_000_000.0, 1.0]]
            ),
            resource_policy=policy,
        )
        try:
            result = simulate_maker(
                [_state(stamp) for stamp in (0, 100, 200, 300, 400, 500, 600, 700)],
                [
                    _trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)
                    for stamp, quantity in trade_schedule
                ],
                [
                    {
                        **_quote(stamp, book_segment_id="s1"),
                        "fold_id": "fold",
                        "price_prediction": 0.0,
                        "proxy_prediction": 1.0,
                    }
                    for stamp in (0, decision_ms)
                ],
                config=replace(
                    _maker_config(), maker_fee_bps=0.0, latency_ms=300 if pending else 0
                ),
                observed_coverage={"s1": (0, 1_000_000_000)},
                trade_coverage={"trade-connection-1": (0, 1_000_000_000)},
                resource_policy=policy,
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=(FundingForecast(0, 1_000_000_000, ()),),
                calibrations={"fold": {"buy": model, "sell": model}},
                on_ledger=batches.append,
                on_fill=fills.append,
            )
        finally:
            model.close()
        assert result["created_orders"] == (3 if keep_wins else 4)
        assert [fill["timestamp_ms"] for fill in fills] == (
            [350] if keep_wins else [350, 450] if pending else [650]
        )
        assert all(fill["estimated_queue_ahead_at_join"] == 2.0 for fill in fills)
        gains = [
            row.number
            for batch in batches
            for row in batch.diagnostics
            if row.name == "maker_buy_expected_net_gain"
        ]
        assert gains == pytest.approx([0.0925, 0.145] if keep_wins else [0.5, 0.5])
        if keep_wins:
            assert fills[0]["order_id"] == 1
        assert policy.used_memory_bytes == 0

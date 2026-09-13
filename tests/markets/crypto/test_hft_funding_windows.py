from dataclasses import asdict, replace

import pytest

from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    simulate_taker,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.markets.crypto.test_hft_execution import _state, _taker_config


def test_taker_known_funding_and_continuous_gap_have_hand_calculated_economics(
    tmp_path,
):
    from quant_research_platform.artifacts.crypto_hft import (
        summarize_crypto_ledger_windows,
    )
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )

    decisions = [
        {"timestamp_ms": 0, "prediction": 0.02},
        {"timestamp_ms": 15, "prediction": 0.02},
        {"timestamp_ms": 60, "prediction": 0.0},
        {"timestamp_ms": 80, "prediction": -0.02},
    ]
    books = [
        _state(stamp, bid_index=995 + step * 10, ask_index=1005 + step * 10)
        for step, stamp in enumerate((0, 20, 30, 40, 60, 80, 100))
    ]
    # This future book must never supply an earlier boundary valuation.
    books.append(_state(1000, bid_index=9995, ask_index=10005))
    forecasts = (
        FundingForecast(
            0,
            1_000_000_000,
            ((40_000_000, -0.103), (60_000_000, 0.104), (80_000_000, -0.21)),
        ),
    )

    def run(events, known=forecasts):
        batches, fills = [], []
        with runtime_resource_policy_scope() as policy:
            result = simulate_taker(
                books,
                decisions,
                prediction_column="prediction",
                prediction_horizon_ms=50,
                config=_taker_config(latency_ms=10),
                observed_coverage={"s1": (0, 2_000_000_000)},
                resource_policy=policy,
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=events,
                funding_forecasts=known,
                decision_windows_ns=((0, 20_000_000), (60_000_000, 100_000_000)),
                on_ledger=batches.append,
                on_fill=lambda row: fills.append(dict(row)),
            )
            assert policy.used_memory_bytes == 0
        selected = []
        boundaries = []
        for batch in batches:
            diagnostics = {row.name: row for row in batch.diagnostics}
            if "decision_requested_quantity" in diagnostics:
                selected.append(diagnostics["decision_requested_quantity"].number)
                if selected[-1] == 0:
                    assert diagnostics["expected_funding_cashflow"].number == 0.0
            for marker in ("oos_window_start", "oos_window_end"):
                if marker in diagnostics:
                    boundaries.append(
                        sum(v.value for v in batch.valuations if v.phase == "end")
                    )
            assert all(v.timestamp_ns <= 100_000_000 for v in batch.valuations)
        return result, selected, boundaries, fills, batches

    events = (
        FundingEvent(40_000_000, 0.001, 103.0),
        FundingEvent(60_000_000, -0.001, 104.0),
        FundingEvent(80_000_000, 0.002, 105.0),
    )
    result, selected, boundaries, fills, batches = run(events)
    assert selected == [1.0, 1.0, 0.0, -1.0]
    assert [row["timestamp_ms"] for row in fills] == [10, 90]
    assert result["created_orders"] == 3 and result["cancelled_orders"] == 1
    assert result["funding"] == pytest.approx(-0.209)
    assert result["fees"] == pytest.approx(0.0205)
    assert result["net_pnl"] == pytest.approx(3.7705)
    assert boundaries == pytest.approx([1000, 1000.48995, 1003.49095, 1003.7705])
    assert boundaries[2] - boundaries[1] == pytest.approx(3.001)

    envelope = build_business_manifest_envelope(
        logical_artifact_id="fixture",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "1970-01-01T00:00:00",
            "end": "1970-01-01T00:00:00.100",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={
            "portfolio_capital": asdict(
                PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
            )
        },
    )
    with runtime_resource_policy_scope() as policy:
        root = write_portfolio_ledger_artifact(
            tmp_path / "ledger",
            envelope=envelope,
            produce=lambda emit: [emit(batch) for batch in batches],
        )
        artifact = read_portfolio_ledger_artifact(
            root, expected_identity=envelope.instance_identity
        )
        intervals = summarize_crypto_ledger_windows(
            artifact, windows_ns=((0, 20_000_000), (60_000_000, 100_000_000))
        )
        assert [row["kind"] for row in intervals] == ["oos", "gap", "oos"]
        assert [row["net_pnl"] for row in intervals] == pytest.approx(
            [0.48995, 3.001, 0.27955]
        )
        assert [row["funding"] for row in intervals] == pytest.approx([0, 0.001, -0.21])
        assert sum(row["fees"] for row in intervals) == pytest.approx(result["fees"])
        assert sum(row["net_pnl"] for row in intervals) == pytest.approx(
            result["net_pnl"]
        )
        with pytest.raises(ValueError, match="crypto_ledger_window_marker_invalid"):
            summarize_crypto_ledger_windows(
                artifact, windows_ns=((0, 21_000_000), (60_000_000, 100_000_000))
            )
        assert policy.used_memory_bytes == 0

    # Real settlements never size orders; new forecasts act only after availability.
    altered = tuple(
        replace(event, rate=-event.rate, mark_price=event.mark_price + 1)
        for event in events
    )
    changed, chosen, _, _, _ = run(altered)
    assert chosen == selected and changed["net_pnl"] != result["net_pnl"]
    for amount in (-10.0, 10.0):
        future = FundingForecast(61_000_000, 1_000_000_000, ((95_000_000, amount),))
        _, chosen, _, _, _ = run(events, (*forecasts, future))
        assert chosen[:3] == selected[:3]


def test_taker_missing_forecast_and_boundary_price_fail_without_future_fill():
    kwargs = dict(
        prediction_column="prediction",
        prediction_horizon_ms=50,
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": (0, 18_000_000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        decision_windows_ns=((0, 20_000_000),),
    )
    with runtime_resource_policy_scope() as policy:
        result = simulate_taker(
            [_state(0), _state(30)],
            [{"timestamp_ms": 0, "prediction": 0.02}],
            funding_forecasts=(),
            resource_policy=policy,
            **kwargs,
        )
        assert result["selection_status"] == "BLOCKED" and result["created_orders"] == 0
        with pytest.raises(
            PortfolioTargetBlocked, match="portfolio_window_boundary_valuation_missing"
        ):
            simulate_taker(
                [_state(0), _state(30)],
                [{"timestamp_ms": 0, "prediction": 0.02}],
                funding_forecasts=(FundingForecast(0, 1_000_000_000, ()),),
                resource_policy=policy,
                **kwargs,
            )
        assert policy.used_memory_bytes == 0

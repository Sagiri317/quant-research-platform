"""Local funding values stay unknown through source import and execution."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    FundingForecast,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    simulate_taker,
)
from quant_research_platform.sources.binance.funding import read_funding_observations
from quant_research_platform.sources.tardis.funding import import_funding_history
from quant_research_platform.sources.tardis.raw import import_slices
from quant_research_platform.sources.tardis.replay import open_tardis_archive
from tests.helpers.binance_hft import execution_state as _state
from tests.helpers.binance_hft import taker_config as _taker_config
from tests.helpers.tardis import _line, _slices

_START_MS = pd.Timestamp("2026-09-01T00:00:00Z").value // 1_000_000


def _mark(second, **changes):
    return {
        "e": "markPriceUpdate",
        "s": "BTCUSDT",
        "E": _START_MS + second * 1000 - 100,
        "p": "100",
        "r": "0.001",
        "T": _START_MS + 10000,
        **changes,
    }


def _import(root, observations, settlements=(), *, source_id="binance"):
    root.mkdir()
    native = b"".join(
        _line(second, "markPrice@1s", payload) for second, payload in observations
    )
    slices = _slices(root / "native", native)
    raw_before = [path.read_bytes() for path in slices]
    response = root / "official.json"
    if source_id == "xoomar":
        settlements = {
            "data": {
                "history": [
                    {
                        "exchange": "binance",
                        "baseAsset": "BTC",
                        "slug": "btc",
                        "fundingTime": pd.Timestamp(
                            row["fundingTime"], unit="ms", tz="UTC"
                        ).isoformat(),
                        **{
                            field: row[field]
                            for field in ("fundingRate", "markPrice")
                            if field in row
                        },
                    }
                    for row in (
                        {"fundingTime": _START_MS + 600000},
                        *reversed(settlements),
                        {"fundingTime": _START_MS - 1},
                    )
                ]
            }
        }
    original = json.dumps(settlements).encode()
    response.write_bytes(original)
    destination = root / "archive"
    import_slices(destination, date="2026-09-01", slices=slices)
    with open_tardis_archive(destination) as archive:
        import_funding_history(
            archive,
            symbol="BTCUSDT",
            source_id=source_id,
            response_path=response,
            http_status=200,
            start_ms=_START_MS,
            end_ms=_START_MS + 599999,
            request_limit=1000 if source_id == "binance" else 5000,
        )
        roots = archive.funding_roots(
            "BTCUSDT",
            time_window=TimeWindow("2026-09-01T00:00:00", "2026-09-01T00:09:59"),
        )
    assert [path.read_bytes() for path in slices] == raw_before
    assert response.read_bytes() == original
    assert (roots[0].parent / "settlements.json").read_bytes() == original
    return roots


def _simulate(source, *, flat=False, latency_ms=10, end_ms=4000):
    batches = []
    result = simulate_taker(
        [_state(_START_MS + stamp) for stamp in (1000, 2000, 3000, end_ms)],
        [
            {"timestamp_ms": _START_MS + stamp, "prediction": 0.0 if flat else score}
            for stamp, score in ((1000, 0.02), (2000, -0.02), (3000, -0.02))
        ],
        prediction_column="prediction",
        prediction_horizon_ms=2000,
        config=replace(_taker_config(latency_ms=latency_ms), taker_fee_bps=0.0),
        observed_coverage={
            "s1": (_START_MS * 1_000_000, (_START_MS + end_ms + 1000) * 1_000_000)
        },
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=source.events,
        funding_forecasts=source.forecasts,
        funding_coverage=source.coverage,
        decision_windows_ns=(
            ((_START_MS + 1000) * 1_000_000, (_START_MS + end_ms) * 1_000_000),
        ),
        on_ledger=batches.append,
    )
    selected = [
        diagnostic.number
        for batch in batches
        for diagnostic in batch.diagnostics
        if diagnostic.name == "decision_requested_quantity"
    ]
    trades = [trade for batch in batches for trade in batch.trades]
    return result, selected, trades


@pytest.mark.parametrize(
    "changes",
    (
        {"r": None},
        {"r": "NaN"},
        {"r": "Infinity"},
        {"r": "bad"},
        {"r": True},
        {"r": {}},
        {"p": None},
        {"p": "NaN"},
        {"p": "0"},
        {"p": "-1"},
        {"p": []},
        {"r": "1e308", "p": "1e308"},
    ),
)
def test_invalid_forecast_amount_survives_import_reader_and_local_decision(
    tmp_path, changes
):
    roots = _import(
        tmp_path / "case",
        ((1, _mark(1)), (2, _mark(2, T=_START_MS + 4000, **changes)), (3, _mark(3))),
    )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        forecasts = tuple(source.forecasts)
        assert len(forecasts) == 3
        assert forecasts[1].available_at_ns == (_START_MS + 2000) * 1_000_000
        assert forecasts[1].cashflows == (((_START_MS + 4000) * 1_000_000, None),)
        assert forecasts[0] == FundingForecast(
            (_START_MS + 1000) * 1_000_000,
            (_START_MS + 10000) * 1_000_000,
            (((_START_MS + 10000) * 1_000_000, -0.1),),
        )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        result, selected, trades = _simulate(source)
    assert selected == [1.0, 0.0, -1.0]
    assert result["blocked_decisions"] == 1
    assert result["selection_status"] == "PARTIAL"
    assert result["account_path_status"] == "COMPLETE"
    assert [trade.timestamp_ns for trade in trades] == [
        (_START_MS + stamp) * 1_000_000 for stamp in (1010, 3010)
    ]


@pytest.mark.parametrize(
    "horizon", (None, -1, "bad", True, _START_MS + 2000, _START_MS + 1999)
)
def test_invalid_or_expired_horizon_supersedes_old_forecast_without_future_coverage(
    tmp_path, horizon
):
    roots = _import(
        tmp_path / "case",
        ((1, _mark(1)), (2, _mark(2, T=horizon)), (3, _mark(3))),
    )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        forecasts = tuple(source.forecasts)
        assert len(forecasts) == 3
        assert forecasts[1] == FundingForecast(
            (_START_MS + 2000) * 1_000_000, (_START_MS + 2000) * 1_000_000, ()
        )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        result, selected, trades = _simulate(source)
    assert selected == [1.0, 0.0, -1.0]
    assert result["blocked_decisions"] == 1
    assert result["account_path_status"] == "COMPLETE"
    assert [trade.timestamp_ns for trade in trades] == [
        (_START_MS + stamp) * 1_000_000 for stamp in (1010, 3010)
    ]


@pytest.mark.parametrize("source_id", ("binance", "xoomar"))
@pytest.mark.parametrize(
    "changes",
    (
        {"fundingRate": None},
        {"fundingRate": "NaN"},
        {"markPrice": "0"},
        {"markPrice": None},
    ),
)
def test_unknown_settlement_preserves_flat_cash_and_stops_only_exposed_path(
    tmp_path, source_id, changes
):
    settlement = {
        "symbol": "BTCUSDT",
        "fundingTime": _START_MS + 1500,
        "fundingRate": "0.1",
        "markPrice": "100",
        **changes,
    }
    roots = _import(
        tmp_path / "case", ((1, _mark(1)),), (settlement,), source_id=source_id
    )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        assert len(source.events) == 1
        event = source.events[0]
        assert event.timestamp_ns == (_START_MS + 1500) * 1_000_000
        if "fundingRate" in changes:
            assert event.rate is None and event.mark_price == 100.0
        else:
            assert event.rate == 0.1 and event.mark_price is None
        flat, _, flat_trades = _simulate(source, flat=True)
    assert flat["account_path_status"] == "COMPLETE"
    assert flat["net_pnl"] == flat["funding"] == 0.0
    assert flat_trades == []
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        exposed, selected, trades = _simulate(source)
    assert exposed["account_path_status"] == "PARTIAL"
    assert exposed["account_stop_reason"] == "portfolio_required_funding_missing"
    assert exposed["net_pnl"] is None
    assert exposed["known_prefix_net_pnl"] is not None
    assert selected == [1.0]
    assert len(trades) == 1


def test_zero_length_forecast_contract_has_no_cashflows():
    assert FundingForecast(5, 5, ()).cashflows == ()
    with pytest.raises(ValueError, match="portfolio_funding_forecast_coverage_invalid"):
        FundingForecast(5, 4, ())
    with pytest.raises(ValueError, match="portfolio_funding_forecast_events_invalid"):
        FundingForecast(5, 5, ((5, None),))


@pytest.mark.parametrize("rate", ("0", "0.01", "-0.01", None))
def test_identifiable_missing_mark_settlement_keeps_only_mathematical_zero(
    tmp_path, rate
):
    roots = _import(
        tmp_path / "case",
        ((1, _mark(1)),),
        ({"symbol": "BTCUSDT", "fundingTime": _START_MS + 1500, "fundingRate": rate},),
    )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        event = source.events[0]
        assert event.rate == (None if rate is None else float(rate))
        assert event.mark_price is None
        exposed, _, trades = _simulate(source)
    if rate == "0":
        assert exposed["account_path_status"] == "COMPLETE"
        assert exposed["funding"] == 0.0
        assert exposed["net_pnl"] is not None
        assert len(trades) == 3
    else:
        assert exposed["account_path_status"] == "PARTIAL"
        assert exposed["account_stop_reason"] == "portfolio_required_funding_missing"
        assert exposed["net_pnl"] is None
        assert len(trades) == 1
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        flat, _, trades = _simulate(source, flat=True)
    assert flat["account_path_status"] == "COMPLETE"
    assert flat["net_pnl"] == flat["funding"] == 0.0
    assert trades == []


@pytest.mark.parametrize("field", ("r", "p", "T"))
def test_absent_forecast_field_is_local_missingness(tmp_path, field):
    payload = _mark(2, T=_START_MS + 4000)
    del payload[field]
    roots = _import(tmp_path / "case", ((1, _mark(1)), (2, payload), (3, _mark(3))))
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        result, selected, _ = _simulate(source)
    assert result["blocked_decisions"] == 1
    assert selected == [1.0, 0.0, -1.0]


@pytest.mark.parametrize(
    "changes,error",
    (
        ({"symbol": "ETHUSDT"}, "identity"),
        ({"fundingTime": None}, "time"),
        ({"fundingTime": True}, "time"),
        ({"fundingTime": -1}, "time"),
        ({"fundingTime": "invalid"}, "time"),
        ({"fundingTime": _START_MS - 1}, "order"),
    ),
)
def test_unknown_numeric_settlement_keeps_identity_and_time_strict(
    tmp_path, changes, error
):
    with pytest.raises(ValueError, match=f"binance_funding_settlement_{error}_invalid"):
        _import(
            tmp_path / "case",
            ((1, _mark(1)),),
            (
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": _START_MS + 1500,
                    "fundingRate": None,
                    "markPrice": None,
                    **changes,
                },
            ),
        )
    assert not (tmp_path / "case/archive/funding/BTCUSDT").exists()


@pytest.mark.parametrize("kind", ("duplicate", "reverse"))
def test_unknown_numeric_settlements_keep_strict_order(tmp_path, kind):
    times = (1500, 1500) if kind == "duplicate" else (2000, 1500)
    with pytest.raises(ValueError, match="binance_funding_settlement_order_invalid"):
        _import(
            tmp_path / "case",
            ((1, _mark(1)),),
            tuple(
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": _START_MS + time,
                    "fundingRate": None,
                    "markPrice": None,
                }
                for time in times
            ),
        )


@pytest.mark.parametrize(
    "raw,error",
    (
        (b"[", ValueError),
        (b"[NaN]", ValueError),
        (b"[null]", ValueError),
        (b"{}", PortfolioTargetBlocked),
    ),
)
def test_settlement_envelope_does_not_become_local_numeric_missingness(raw, error):
    from quant_research_platform.sources.binance.funding import _settlement_response

    with pytest.raises(error):
        _settlement_response(
            raw,
            source_id="binance",
            symbol="BTCUSDT",
            start_ms=_START_MS,
            end_ms=_START_MS + 599999,
            request_limit=1000,
        )


@pytest.mark.parametrize("conflict", (False, True))
def test_unknown_forecasts_keep_same_time_retransmission_identity(tmp_path, conflict):
    observations = (
        (1, _mark(1, r="bad")),
        (1, _mark(1, r="NaN" if conflict else "bad")),
    )
    roots = _import(tmp_path / "case", observations)
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        forecasts = tuple(source.forecasts)
        assert len(forecasts) == 1
        assert forecasts[0].cashflows == (((_START_MS + 10000) * 1_000_000, None),)


def test_unknown_settlement_cannot_be_rewritten_as_zero_in_projection(tmp_path):
    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.raw_snapshots import (
        iter_csv_snapshot,
        write_csv_snapshot,
    )

    (root,) = _import(
        tmp_path / "case",
        ((1, _mark(1)),),
        (
            {
                "symbol": "BTCUSDT",
                "fundingTime": _START_MS + 1500,
                "fundingRate": "bad",
                "markPrice": "100",
            },
        ),
    )
    manifest = read_manifest(root / "response.metadata.json")
    rows = [list(row) for row in iter_csv_snapshot(root, expected_manifest=manifest)]
    for row in rows:
        if row[0] == "settlement":
            assert row[3] is None
            row[3] = 0.0
    changed = tmp_path / "changed/observations"
    changed.parent.mkdir()
    original = (root.parent / "settlements.json").read_bytes()
    (changed.parent / "settlements.json").write_bytes(original)

    def validate(staging):
        checked = read_manifest(staging / "response.metadata.json")
        assert len(tuple(iter_csv_snapshot(staging, expected_manifest=checked))) == len(
            rows
        )

    write_csv_snapshot(
        changed,
        columns=tuple(manifest["columns"]),
        rows=rows,
        metadata=manifest["metadata"],
        validate=validate,
    )
    with pytest.raises(
        ValueError, match="binance_funding_settlement_projection_invalid"
    ):
        with read_funding_observations((changed,), symbol="BTCUSDT"):
            pytest.fail("unknown actual settlement must not become exact zero")
    assert (root.parent / "settlements.json").read_bytes() == original


def test_imported_unknown_horizon_does_not_undo_committed_latency(tmp_path):
    roots = _import(
        tmp_path / "case", ((1, _mark(1)), (2, _mark(2, T=None)), (3, _mark(3)))
    )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        result, selected, trades = _simulate(source, latency_ms=1500, end_ms=6000)
    assert selected == [1.0, 0.0, -1.0]
    assert result["account_path_status"] == "COMPLETE"
    assert result["blocked_decisions"] == 1
    assert [trade.timestamp_ns for trade in trades] == [
        (_START_MS + stamp) * 1_000_000 for stamp in (2500, 4500)
    ]


@pytest.mark.parametrize("rate", ("0", "0.01", "-0.01", None))
@pytest.mark.parametrize("actual_rate", ("0.5", "-0.5"))
def test_zero_rate_forecast_needs_no_mark_and_never_uses_actual_settlement(
    tmp_path, rate, actual_rate
):
    roots = _import(
        tmp_path / "case",
        (
            (1, _mark(1)),
            (2, _mark(2, T=_START_MS + 4000, r=rate, p=None)),
            (3, _mark(3)),
        ),
        (
            {
                "symbol": "BTCUSDT",
                "fundingTime": _START_MS + 4000,
                "fundingRate": actual_rate,
                "markPrice": "100",
            },
        ),
    )
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        forecasts = tuple(source.forecasts)
        assert forecasts[1].cashflows == (
            ((_START_MS + 4000) * 1_000_000, 0.0 if rate == "0" else None),
        )
        assert source.events[0].rate == float(actual_rate)
        assert source.events[0].mark_price == 100.0
    with read_funding_observations(roots, symbol="BTCUSDT") as source:
        result, selected, trades = _simulate(source)
    assert result["account_path_status"] == "COMPLETE"
    assert selected == ([1.0, -1.0, -1.0] if rate == "0" else [1.0, 0.0, -1.0])
    assert result["blocked_decisions"] == (0 if rate == "0" else 1)
    assert len(trades) == (3 if rate == "0" else 2)
    assert result["funding"] == (100.0 * float(actual_rate) if rate == "0" else 0.0)

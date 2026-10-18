"""Causal state alignment, full windows, and observed-only liquidation quantities."""

import copy

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_derivative_factor_frames,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_DERIVATIVE_CHANNELS,
)


def test_derivative_oracle_windows_age_chunks_and_causal_observations():
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-08-01", periods=3003, freq="100ms"),
            "segment_id": ["a"] * 3002 + ["b"],
            "mid": 100.0,
        }
    )
    origin = panel.timestamp.iloc[0].value
    events = [
        (
            origin,
            "premiumIndex",
            dict(
                markPrice="102",
                indexPrice="101",
                lastFundingRate="0.0001",
                nextFundingTime=origin // 1000000 + 3600000,
                time=origin // 1000000 - 10,
            ),
            0,
        ),
        (origin, "openInterest", dict(openInterest="10", time=origin // 1000000), 0),
        (
            origin + 4500000000,
            "forceOrder",
            {"E": origin // 1000000 + 4400, "o": {"S": "SELL", "z": "2", "q": "100"}},
            0,
        ),
        (
            origin + 4600000000,
            "forceOrder",
            {"E": origin // 1000000 + 4500, "o": {"S": "BUY", "z": "6", "q": "1"}},
            0,
        ),
        (
            origin + 30000000000,
            "openInterest",
            dict(openInterest="20", time=origin // 1000000 + 29900),
            0,
        ),
    ]
    ages = dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000)
    ages["premiumIndex"] = 5000

    def compute(frames, observations=events):
        return pd.concat(
            iter_derivative_factor_frames(
                frames, observations=iter(observations), max_age_ms=ages
            ),
            ignore_index=True,
        )

    whole = compute([panel])
    split = compute([panel.iloc[i : i + 17] for i in range(0, len(panel), 17)])
    pd.testing.assert_frame_equal(whole, split, check_exact=True)
    assert whole.mark_mid_basis.iloc[0] == pytest.approx(0.02)
    assert whole.mark_index_basis.iloc[0] == pytest.approx(102 / 101 - 1)
    assert whole.perp_index_basis.iloc[0] == pytest.approx(100 / 101 - 1)
    assert whole.funding_rate.iloc[0] == 0.0001
    assert whole.time_to_next_funding.iloc[50] == 3595
    assert whole.mark_mid_basis.iloc[51:].isna().all()
    assert whole.premiumIndex_age_ms.iloc[51] == 5100
    assert whole.premiumIndex_observed_at.iloc[0] == panel.timestamp.iloc[0]
    assert whole.premiumIndex_event_at.iloc[0] == panel.timestamp.iloc[
        0
    ] - pd.Timedelta(milliseconds=10)
    assert whole.log_open_interest.iloc[300] == pytest.approx(np.log(20))
    for column, first in (
        ("open_interest_change_30s", 300),
        ("open_interest_change_1m", 600),
        ("open_interest_change_5m", 3000),
    ):
        assert whole[column].iloc[:first].isna().all()
        assert whole[column].iloc[first] == 1
        assert pd.isna(whole[column].iloc[-1])
    assert whole.observed_force_order_imbalance_5s.iloc[:49].isna().all()
    assert whole.observed_force_order_imbalance_5s.iloc[49] == 0.5
    assert whole.time_since_observed_force_order.iloc[49] == 0.3
    assert pd.isna(whole.observed_force_order_imbalance_5s.iloc[100])
    changed = copy.deepcopy(events)
    changed[3][2]["o"]["z"] = "600"
    pd.testing.assert_frame_equal(
        whole.iloc[:46], compute([panel], changed).iloc[:46], check_exact=True
    )
    gap = panel.iloc[:700].copy()
    gap.loc[200:, "segment_id"] = "after_gap"
    reset = compute([gap])
    assert reset.open_interest_change_30s.iloc[200:500].isna().all()
    assert reset.open_interest_change_30s.iloc[500] == 1


@pytest.mark.parametrize(
    "mode", ["no_gap", "future_gap", "other_asset_reset", "same_asset_reset"]
)
def test_native_epoch_invalidates_all_assets_only_after_known_clock(tmp_path, mode):
    import json

    from quant_research_platform.markets.implementations.crypto.processed import catalog
    from quant_research_platform.sources.market import open_market_source
    from quant_research_platform.sources.tardis.raw import STATE_CHANNELS, import_slices
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.paths import StorageLayout
    from tests.helpers.tardis import _slices

    origin = pd.Timestamp("2026-09-01").value

    def line(milliseconds, symbol, channel, payload):
        stamp = (
            pd.Timestamp(origin + milliseconds * 1000000, tz="UTC")
            .isoformat()
            .replace("+00:00", "Z")
        )
        return (
            stamp
            + " "
            + json.dumps({"stream": symbol.lower() + "@" + channel, "data": payload})
            + "\n"
        ).encode()

    native = line(
        4500,
        "BTCUSDT",
        "forceOrder",
        {
            "e": "forceOrder",
            "E": origin // 1000000 + 4500,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",
                "q": "2",
                "l": "2",
                "z": "2",
                "T": origin // 1000000 + 4500,
            },
        },
    )
    if mode != "no_gap":
        native += b"\n"
    next_ms = 5100 if mode == "future_gap" else 4600
    symbol = "BTCUSDT" if mode == "same_asset_reset" else "ETHUSDT"
    native += line(
        next_ms,
        symbol,
        "openInterest",
        {"symbol": symbol, "time": origin // 1000000 + next_ms, "openInterest": "10"},
    )
    native += line(
        6000,
        "BTCUSDT",
        "openInterest",
        {"symbol": "BTCUSDT", "time": origin // 1000000 + 6000, "openInterest": "10"},
    )
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-09-01", periods=51, freq="100ms"),
            "segment_id": "independent_continuous_source",
            "mid": 100.0,
        }
    )
    root = tmp_path / "archive"
    import_slices(
        root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native, count=1),
        channels=STATE_CHANNELS,
    )
    with open_market_source(root) as archive:
        frames = iter_derivative_factor_frames(
            [panel],
            observations=archive.iter_state_observations(symbol="BTCUSDT"),
            max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000),
        )
        spec = catalog.hft_processed_catalog().get(
            "binance_futures_derivative_factors_100ms"
        )
        layout = StorageLayout(tmp_path)
        output = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        write_dataset(output, frames, spec=spec)
        result = read_dataset(output)
        value = result.observed_force_order_imbalance_5s.iloc[-1]
        if mode in ("no_gap", "future_gap"):
            assert value == -1
        else:
            assert pd.isna(value)


@pytest.mark.parametrize("count", [200, 400])
def test_derivative_catchup_retains_no_expired_force_orders(monkeypatch, count):
    from collections import deque

    import quant_research_platform.markets.implementations.crypto.hft as hft

    peak = 0

    class TrackedDeque(deque):
        def append(self, value):
            nonlocal peak
            super().append(value)
            peak = max(peak, len(self))

    monkeypatch.setattr(hft, "deque", TrackedDeque)
    origin = pd.Timestamp("2026-09-01").value
    panel = pd.DataFrame(
        {
            "market_id": ["crypto"],
            "asset_id": ["btc_usdt"],
            "timestamp": [pd.Timestamp(origin)],
            "segment_id": ["s"],
            "mid": [100.0],
        }
    )
    events = [
        (
            origin - 31000000000 - (count - 1 - i) * 1000000,
            "forceOrder",
            {
                "E": (origin - 31000000000 - (count - 1 - i) * 1000000) // 1000000,
                "o": {"S": "SELL", "z": "2", "q": "100"},
            },
            0,
        )
        for i in range(count)
    ]
    result = next(
        iter_derivative_factor_frames(
            [panel],
            observations=iter(events),
            max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000),
        )
    )
    assert peak == 0
    assert result.time_since_observed_force_order.iloc[0] == 31
    assert result.observed_force_order_imbalance_30s.isna().all()


def test_native_force_windows_preserve_observed_mass_resets_and_early_close(tmp_path):
    import json

    from quant_research_platform.markets.implementations.crypto.processed import catalog
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.sources.market import open_market_source
    from quant_research_platform.sources.tardis.raw import STATE_CHANNELS, import_slices
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.paths import StorageLayout
    from tests.helpers.tardis import _slices

    origin = pd.Timestamp("2026-09-01", tz="UTC").value
    native = []
    for milliseconds, side, quantity in (
        (50, "SELL", "0.1"),
        (200, "BUY", "0.2"),
        (4900, "SELL", "0.1"),
        (5100, "BUY", "0.3"),
        (8000, "SELL", "1.0"),
        (8200, "BUY", "1.0"),
        (33000, "SELL", "0.2"),
        (36000, "BUY", "0.4"),
        (36100, "SELL", "0.1"),
        (64000, "BUY", "1"),
        (65000, "SELL", "2"),
        (71000, "BUY", "3"),
    ):
        if milliseconds == 36000:
            native.append(b"\n")
        stamp = pd.Timestamp(origin + milliseconds * 10**6, tz="UTC")
        payload = {
            "e": "forceOrder",
            "E": origin // 10**6 + milliseconds,
            "o": {
                "s": "BTCUSDT",
                "S": side,
                "q": quantity,
                "l": quantity,
                "z": quantity,
                "T": origin // 10**6 + milliseconds,
            },
        }
        native.append(
            (
                stamp.isoformat().replace("+00:00", "Z")
                + " "
                + json.dumps({"stream": "btcusdt@forceOrder", "data": payload})
                + "\n"
            ).encode()
        )
    archive_root = tmp_path / "archive"
    import_slices(
        archive_root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", b"".join(native), count=1),
        channels=STATE_CHANNELS,
    )
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-09-01", periods=701, freq="100ms"),
            "segment_id": "first",
            "mid": 100.0,
        }
    )
    panel.loc[
        panel.timestamp.ge(pd.Timestamp(origin + 10000 * 10**6)), "segment_id"
    ] = "second"
    spec = catalog.hft_processed_catalog().get(
        "binance_futures_derivative_factors_100ms"
    )

    def calculate(selected, batch, name):
        destination = StorageLayout(tmp_path / name).processed_data_root(
            spec.market_id, spec.kind, spec.data_id
        )
        with open_market_source(archive_root) as archive:
            factors = iter_derivative_factor_frames(
                (selected.iloc[i : i + batch] for i in range(0, len(selected), batch)),
                observations=archive.iter_state_observations(symbol="BTCUSDT"),
                max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 10000),
            )
            write_dataset(destination, factors, spec=spec)
        return read_dataset(destination)

    whole = calculate(panel, len(panel), "whole")
    split = calculate(panel, 17, "split")
    pd.testing.assert_frame_equal(whole, split, check_exact=True)
    beginning = pd.Timestamp(origin + 10000 * 10**6)
    narrow = calculate(panel.loc[panel.timestamp.ge(beginning)], 31, "narrow")
    pd.testing.assert_frame_equal(
        whole.loc[whole.timestamp.ge(beginning)].reset_index(drop=True),
        narrow,
        check_exact=True,
    )
    by_time = whole.set_index("timestamp")
    row = by_time.loc[pd.Timestamp(origin + 8300 * 10**6)]
    # In (3.3s, 8.3s], SELL 0.1 + 1.0 and BUY 0.3 + 1.0 give 0.2 / 2.4.
    assert row.observed_force_order_imbalance_5s == pytest.approx(1 / 12, rel=1e-14)
    row = by_time.loc[beginning]
    assert row.forceOrder_observed_at == pd.Timestamp(origin + 8200 * 10**6)
    assert pd.isna(row.observed_force_order_imbalance_5s)
    row = by_time.loc[pd.Timestamp(origin + 69000 * 10**6)]
    assert row.observed_force_order_imbalance_5s == -1
    assert row.observed_force_order_imbalance_30s == pytest.approx(-1 / 3, rel=1e-14)
    assert row.time_since_observed_force_order == 4
    primary = RuntimeError("synthetic_consumer_stop")
    for fail in (False, True):
        try:
            with open_market_source(archive_root) as archive:
                observations = archive.iter_state_observations(symbol="BTCUSDT")
                with closing_iterator(
                    iter_derivative_factor_frames(
                        (panel.iloc[i : i + 17] for i in range(0, len(panel), 17)),
                        observations=observations,
                        max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 10000),
                    )
                ) as values:
                    assert len(next(values)) == 17
                    if fail:
                        raise primary
        except RuntimeError as error:
            assert fail and error is primary
        else:
            assert not fail
        assert observations.gi_frame is None


@pytest.mark.parametrize("input_delay_ms", (0, 50))
@pytest.mark.parametrize("boundary_ms", (10000, 10050))
@pytest.mark.parametrize("reset", (False, True))
def test_force_order_epoch_rewarms_from_observed_boundary(
    input_delay_ms, boundary_ms, reset
):
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-08-01", periods=405, freq="100ms"),
            "segment_id": "independent",
            "mid": 100.0,
        }
    )
    origin = panel.timestamp.iloc[0].value
    epoch = int(reset)
    events = [
        (origin, None, None, 0),
        (origin + boundary_ms * 10**6, None, None, epoch),
    ]
    for milliseconds in (boundary_ms + 1, boundary_ms + 4900, boundary_ms + 29900):
        events.append(
            (
                origin + milliseconds * 10**6,
                "forceOrder",
                {"E": origin // 10**6 + milliseconds, "o": {"S": "BUY", "z": "1"}},
                epoch,
            )
        )

    def calculate(batch):
        return pd.concat(
            iter_derivative_factor_frames(
                (panel.iloc[i : i + batch] for i in range(0, len(panel), batch)),
                observations=iter(events),
                input_delay_ms=input_delay_ms,
                max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000),
            ),
            ignore_index=True,
        )

    actual = calculate(len(panel))
    pd.testing.assert_frame_equal(actual, calculate(17), check_exact=True)
    information_ms = np.arange(len(panel)) * 100 - input_delay_ms
    for seconds in (5, 30):
        boundary = boundary_ms if reset else 0
        first = int(np.flatnonzero(information_ms >= boundary + seconds * 1000)[0])
        column = actual[f"observed_force_order_imbalance_{seconds}s"]
        if reset:
            assert column.iloc[first - 1] != column.iloc[first - 1]
            assert column.iloc[first] == 1.0
        else:
            after_order = max(
                first, int(np.flatnonzero(information_ms >= boundary_ms + 1)[0])
            )
            assert column.iloc[after_order] == 1.0


@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("field", ["mark", "mid"])
def test_derivative_relative_changes_keep_adjacent_prices_and_oi(steps, field):
    from fractions import Fraction

    price = 3.0
    for _ in range(steps):
        price = float(np.nextafter(price, np.inf))
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-08-01", periods=301, freq="100ms"),
            "segment_id": "a",
            "mid": price if field == "mid" else 3.0,
        }
    )
    origin = panel.timestamp.iloc[0].value
    events = [
        (
            origin,
            "premiumIndex",
            {
                "markPrice": str(price if field == "mark" else 3.0),
                "indexPrice": "3.0",
                "lastFundingRate": "0.0",
                "nextFundingTime": origin // 1000000 + 3600000,
                "time": origin // 1000000,
            },
            0,
        ),
        (origin, "openInterest", {"openInterest": "3.0", "time": origin // 1000000}, 0),
        (
            origin + 30000000000,
            "openInterest",
            {"openInterest": str(price), "time": origin // 1000000 + 30000},
            0,
        ),
    ]
    result = pd.concat(
        iter_derivative_factor_frames(
            (panel.iloc[:17], panel.iloc[17:]),
            observations=iter(events),
            max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 400000),
        ),
        ignore_index=True,
    )
    for name, numerator, denominator in (
        (
            "mark_mid_basis",
            price if field == "mark" else 3.0,
            price if field == "mid" else 3.0,
        ),
        ("mark_index_basis", price if field == "mark" else 3.0, 3.0),
        ("perp_index_basis", price if field == "mid" else 3.0, 3.0),
    ):
        assert result[name].iloc[0] == float(
            (Fraction(numerator) - Fraction(denominator)) / Fraction(denominator)
        )
    assert result.open_interest_change_30s.iloc[:300].isna().all()
    assert result.open_interest_change_30s.iloc[300] == float((Fraction(price) - 3) / 3)


@pytest.mark.parametrize("side", [None, "unknown"])
def test_native_unknown_force_order_side_is_local_flow_barrier(tmp_path, side):
    from quant_research_platform.sources.market import open_market_source
    from quant_research_platform.sources.tardis.raw import import_slices
    from tests.helpers.tardis import _line, _slices

    origin = pd.Timestamp("2026-09-01").value
    native = b""
    for second, value in ((0, "SELL"), (5, side), (6, "BUY")):
        stamp = origin // 1_000_000 + second * 1000
        native += _line(
            second,
            "forceOrder",
            {
                "e": "forceOrder",
                "E": stamp,
                "o": {
                    "s": "BTCUSDT",
                    "S": value,
                    "T": stamp,
                    "q": "2",
                    "l": "2",
                    "z": "2",
                },
            },
        )
    archive_root = tmp_path / "archive"
    import_slices(
        archive_root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", native, count=1),
        symbols=("btcusdt",),
        channels=("forceOrder",),
    )
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": pd.date_range("2026-09-01", periods=101, freq="100ms"),
            "segment_id": "book",
        }
    )
    with open_market_source(archive_root) as source:
        observations = list(source.iter_state_observations(symbol="BTCUSDT"))
    assert observations[1][2]["o"]["S"] is None
    result = pd.concat(
        iter_derivative_factor_frames(
            (panel.iloc[:53], panel.iloc[53:]),
            observations=observations,
            max_age_ms={"forceOrder": 10000},
            columns=(
                "observed_force_order_imbalance_5s",
                "time_since_observed_force_order",
            ),
        ),
        ignore_index=True,
    )
    assert result.time_since_observed_force_order.iloc[50] == 0
    assert result.forceOrder_age_ms.iloc[50] == 0
    assert result.observed_force_order_imbalance_5s.iloc[50:100].isna().all()
    assert result.observed_force_order_imbalance_5s.iloc[100] == 1


def test_native_constant_liquidation_ratio_has_no_standardized_signal(tmp_path):
    import json

    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.implementations.crypto.processed import catalog
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.models.scaler import ModelMatrixScaler
    from quant_research_platform.sources.market import open_market_source
    from quant_research_platform.sources.tardis.raw import import_slices
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.paths import StorageLayout
    from tests.helpers.tardis import _slices

    times = pd.date_range("2026-09-01", periods=401, freq="100ms")
    quantities = (0.01, 0.015, 0.03, 0.035, 0.07, 0.1, 0.15, 0.3, 0.5)
    native = []
    for i, stamp in enumerate(times):
        for side, multiplier in (("BUY", 2), ("SELL", 1)):
            quantity = str(multiplier * quantities[i % len(quantities)])
            event_ms = stamp.value // 10**6
            payload = {
                "e": "forceOrder",
                "E": event_ms,
                "o": {
                    "s": "BTCUSDT",
                    "S": side,
                    "T": event_ms,
                    "q": quantity,
                    "l": quantity,
                    "z": quantity,
                },
            }
            native.append(
                (
                    stamp.isoformat()
                    + "Z "
                    + json.dumps({"stream": "btcusdt@forceOrder", "data": payload})
                    + "\n"
                ).encode()
            )
    archive_root = tmp_path / "archive"
    import_slices(
        archive_root,
        date="2026-09-01",
        slices=_slices(tmp_path / "native", b"".join(native)),
        channels=("forceOrder",),
    )
    panel = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": times,
            "segment_id": "book",
            "mid": 60000.0,
        }
    )
    spec = catalog.hft_processed_catalog().get(
        "binance_futures_derivative_factors_100ms"
    )
    output = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    with open_market_source(archive_root) as source:
        frames = iter_derivative_factor_frames(
            (panel.iloc[:53], panel.iloc[53:]),
            observations=source.iter_state_observations(symbol="BTCUSDT"),
            max_age_ms={"forceOrder": 10000},
        )
        write_dataset(output, frames, spec=spec)
    result = read_dataset(output)
    panel_root = tmp_path / "panel"
    roles = dict.fromkeys(panel.columns, "context")
    roles.update(market_id="key", asset_id="key", timestamp="timestamp")
    write_dataset(
        panel_root,
        panel,
        spec=DataSpec(
            data_id="liquidation_input",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
        ),
    )
    prepared = read_dataset(
        prepare(
            {
                "market_id": "crypto",
                "data_id": spec.data_id,
                "inputs": {"binance_futures_l2_panel_100ms": str(panel_root)},
                "source": {"root": str(archive_root)},
                "parameters": {"max_age_ms": {"forceOrder": 10000}},
                "factor_columns": [
                    "observed_force_order_imbalance_5s",
                    "observed_force_order_imbalance_30s",
                ],
                "time_window": {
                    "start": times[0].isoformat(),
                    "end": times[-1].isoformat(),
                },
                "output_root": str(tmp_path / "prepared"),
            }
        )
    )
    pd.testing.assert_frame_equal(prepared, result[prepared.columns], check_exact=True)
    for seconds in (5, 30):
        column = f"observed_force_order_imbalance_{seconds}s"
        first = seconds * 10 - 1
        assert prepared[column].iloc[:first].isna().all()
        mature = prepared[[column]].iloc[first:]
        # Every pair has BUY = 2 * SELL, hence (BUY-SELL)/(BUY+SELL) = 1/3.
        assert mature.notna().all().all()
        assert mature.iloc[:, 0].nunique() == 1
        assert mature.iloc[0, 0] == pytest.approx(1 / 3, rel=1e-14)
        assert not ModelMatrixScaler.fit(mature).transform(mature).to_numpy().any()

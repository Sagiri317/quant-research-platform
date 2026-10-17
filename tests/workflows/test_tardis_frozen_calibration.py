"""Frozen per-asset maker coordinates survive an expanded daily source window."""

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    _maker_new_order,
)
from quant_research_platform.markets.implementations.crypto.portfolio import (
    _calibration_events,
    _calibration_samples,
    validate_maker_calibration,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.tardis.raw import import_slices
from tests.artifacts.test_crypto_maker_calibration import maker_calibration_tables
from tests.helpers.binance_hft import _quote, hft_research_plan
from tests.helpers.tardis import _slices
from tests.markets.crypto.test_hft_execution import _maker_config, _state


def _publish(root, date, *, fine):
    midnight = pd.Timestamp(date).value // 1_000_000
    native = []
    for second, channel in ((1, "depth@0ms"), (2, "depthSnapshot"), (3, "aggTrade")):
        for symbol, unit in (("BTCUSDT", "1"), ("ETHUSDT", "0.01")):
            unit = Decimal(unit)
            price = Decimal(100) * unit + (unit / 10 if fine else 0)
            if channel == "depthSnapshot":
                payload = {
                    "lastUpdateId": 100,
                    "bids": [[str(price - index * unit), "2"] for index in range(12)],
                    "asks": [
                        [str(price + (index + 1) * unit), "2"] for index in range(12)
                    ],
                }
            elif channel == "depth@0ms":
                payload = {
                    "e": "depthUpdate",
                    "E": midnight + 1000,
                    "T": midnight + 1000,
                    "s": symbol,
                    "st": 1,
                    "ps": symbol,
                    "U": 99,
                    "u": 101,
                    "pu": 98,
                    "b": [[str(price), "3"]],
                    "a": [],
                }
            else:
                payload = {
                    "e": "aggTrade",
                    "E": midnight + 3000,
                    "T": midnight + 3000,
                    "s": symbol,
                    "st": 1,
                    "a": 1,
                    "f": 1,
                    "l": 1,
                    "p": str(price),
                    "q": "3",
                    "nq": "3",
                    "m": True,
                }
            value = {"stream": symbol.lower() + "@" + channel, "data": payload}
            if channel == "depthSnapshot":
                value["generated"] = True
            native.append(
                f"{date}T00:00:{second:02d}.000000Z " + json.dumps(value) + "\n"
            )
    destination = root / "binance-futures" / date
    import_slices(
        destination,
        date=date,
        slices=_slices(root / ("native-" + date), "".join(native).encode(), count=1),
    )
    return destination.relative_to(root).as_posix()


@pytest.mark.parametrize(
    "asset,unit,current_unit", [("btc_usdt", "1", "0.1"), ("eth_usdt", "0.01", "0.001")]
)
def test_explicit_source_collection_and_persisted_maker_units(
    tmp_path, asset, unit, current_unit
):
    tables, metadata = maker_calibration_tables.__wrapped__()
    tables["crypto_maker_calibration"]["price_unit"] = unit
    shift = pd.Timedelta(pd.Timestamp("2026-08-01").value, unit="ns")
    for column in ("decision_at", "prediction_end", "label_available_at"):
        tables["crypto_maker_samples"][column] += shift
    tables["crypto_maker_events"]["event_at"] += shift
    plan = hft_research_plan()
    plan["folds"] = [plan["folds"][0]]
    plan["folds"][0]["train"] = {
        "start": "2026-07-01T00:00:00",
        "end": "2026-07-01T00:00:02",
    }
    plan["folds"][0]["validation"] = {
        "start": "2026-08-01T00:00:02.100000",
        "end": "2026-08-01T00:00:03.500000",
    }
    plan["test"] = {"start": "2026-09-01T00:00:08.600000", "end": "2026-09-01T00:00:09"}
    raw = tmp_path / "raw"
    raw.mkdir()
    symbol = {"btc_usdt": "BTCUSDT", "eth_usdt": "ETHUSDT"}[asset]
    archives = [
        _publish(raw, day, fine=index == 2)
        for index, day in enumerate(("2026-07-01", "2026-08-01", "2026-09-01"))
    ]
    from quant_research_platform.sources.market import open_market_source

    with open_market_source(raw, archive_paths=archives[:2]) as source:
        assert len(source.archives) == 2
        assert source.market_metadata(FUTURES_MARKET_TYPE, symbol)["price_unit"] == unit
    with open_market_source(raw, archive_paths=archives) as source:
        assert len(source.archives) == 3
        assert (
            source.market_metadata(FUTURES_MARKET_TYPE, symbol)["price_unit"]
            == current_unit
        )
    metadata = {
        **metadata,
        "chronology": {"folds": plan["folds"], "test": plan["test"]},
        "window": plan["folds"][0]["validation"],
        "asset_id": asset,
        "symbol": symbol,
    }

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    root = write_statistics_artifact(
        tmp_path / "calibration",
        metadata=metadata,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=validate_maker_calibration,
    )

    def fingerprints():
        return {
            path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    before = fingerprints()
    artifact = read_statistics_artifact(
        root,
        validate_tables=validate_maker_calibration,
    )
    with artifact.table("crypto_maker_calibration.parquet") as summary:
        frozen_unit = summary.price_unit.iloc[0]
    samples = partial(_calibration_samples, artifact, "fold_1")
    events = partial(_calibration_events, artifact, "fold_1")

    def fine_samples():
        for sample in samples():
            yield sample | {
                key: sample[key] * 10
                for key in ("initial_bid_index", "initial_ask_index")
            }

    def fine_events():
        for event in events():
            values = vars(event).copy()
            if event.kind == "book":
                values.update(
                    {
                        side: tuple(
                            (tick * 10, quantity) for tick, quantity in values[side]
                        )
                        for side in ("bids", "asks")
                    }
                )
            elif event.kind == "trade":
                values["price_index"] *= 10
            yield SimpleNamespace(**values)

    historical = MakerCalibration(samples, events, price_unit=frozen_unit)
    baseline = MakerCalibration(fine_samples, fine_events, price_unit=current_unit)
    state = replace(
        _state(0),
        price_unit=current_unit,
        bids=tuple((tick * 10, quantity) for tick, quantity in _state(0).bids),
        asks=tuple((tick * 10, quantity) for tick, quantity in _state(0).asks),
    )
    new = _maker_new_order(
        2,
        "buy",
        1.0,
        state,
        _quote(200, book_segment_id="s1"),
        _maker_config().latency_ms,
    )
    new.effective_at_ns = 300_000_000
    old = replace(
        new,
        order_id=1,
        effective_at_ns=0,
        price_index=new.price_index + 50,
        queue_ahead=0,
    )
    pending = (
        (
            250_000_000,
            "buy",
            replace(new, order_id=3, effective_at_ns=250_000_000),
        ),
    )
    responses = {}
    for commands in ((), pending):
        query = dict(
            state=state,
            decision_ns=200_000_000,
            new_order=new,
            active_order=old,
            commands=commands,
            lifetime_ns=500_000_000,
            prediction=0.02,
            fee_bps=1.0,
            funding=[],
        )
        expected = baseline.select(**query)
        actual = historical.select(**query)
        assert actual[:2] == expected[:2]
        assert actual[2] == pytest.approx(expected[2])
        for replacement in (None, 0.0, 1.0):
            outputs = []
            for model in (historical, baseline):
                sample_stream = model.samples("buy")
                try:
                    sample = next(sample_stream)
                    outputs.append(
                        list(
                            model._response(
                                sample,
                                state=state,
                                decision_ns=200_000_000,
                                new_order=new,
                                active_order=old,
                                commands=commands,
                                lifetime_ns=500_000_000,
                                replace_quantity=replacement,
                                history_scale=(
                                    model.price_unit / Fraction(new.price_unit)
                                ).numerator,
                                events=model.events(
                                    sample["event_start"],
                                    sample["event_end"],
                                ),
                            )
                        )
                    )
                finally:
                    sample_stream.close()
            assert outputs[0] == outputs[1]
            assert outputs[0]
            responses[bool(commands), replacement] = outputs[0]
    for replacement in (None, 0.0, 1.0):
        assert responses[False, replacement] != responses[True, replacement]
        assert any(
            50_000_000 < fill[0] < 100_000_000 for fill in responses[True, replacement]
        )
    assert fingerprints() == before


@pytest.mark.parametrize(
    "early_defined,last_defined", [(False, True), (True, True), (True, False)]
)
def test_final_maker_gate_uses_only_required_fold_but_validates_full_artifact(
    tmp_path, monkeypatch, early_defined, last_defined
):
    import quant_research_platform.markets.implementations.crypto.portfolio as portfolio
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    original_tables, metadata = maker_calibration_tables.__wrapped__()
    plan = hft_research_plan()
    plan["folds"][0] = metadata["chronology"]["folds"][0]
    metadata["chronology"] = {"folds": plan["folds"], "test": plan["test"]}
    metadata["window"] = {
        "start": plan["folds"][0]["validation"]["start"],
        "end": plan["folds"][-1]["validation"]["end"],
    }
    metadata["model_ids"]["fold_2"] = metadata["model_ids"]["fold_1"]
    pieces = {name: [] for name in original_tables}
    for fold_id, defined, shift in (
        ("fold_1", early_defined, pd.Timedelta(0)),
        ("fold_2", last_defined, pd.Timedelta(seconds=4.5)),
    ):
        for name, source in original_tables.items():
            frame = source.copy()
            frame["fold_id"] = fold_id
            if name == "crypto_maker_calibration":
                if not defined:
                    frame.loc[:, ["sample_count", "zero_fill_count"]] = 0
                    frame["status"] = "UNDEFINED"
                    frame["reason"] = "insufficient_calibration_samples"
            elif not defined:
                frame = frame.iloc[:0]
            elif name == "crypto_maker_samples":
                for column in ("decision_at", "prediction_end", "label_available_at"):
                    frame[column] += shift
            else:
                frame["event_at"] += shift
            pieces[name].append(frame)
    tables = {
        name: pd.concat(frames, ignore_index=True) for name, frames in pieces.items()
    }
    config = {
        "output": str(tmp_path / "backtest"),
        "inputs": "inputs",
        "panel": "panel",
        "source": str(tmp_path),
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "prediction_window": plan["test"],
        "execution_mode": "continuous",
        "capital": {
            "candidate_id": "maker",
            "initial_capital": 1000.0,
            "valuation_currency": "USDT",
            "valuation_method": "linear_perpetual",
        },
        "execution": {
            "order_quantity": 1.0,
            "max_position": 2.0,
            "maker_fee_bps": 0.0,
            "inventory_target": 0.0,
            "inventory_skew": 0.0,
            "markout_horizon_ms": 100,
            "latency_ms": 10,
        },
        "models": [
            {"predictions": value + "-predictions"}
            for value in metadata["model_ids"]["fold_2"]
        ],
        "calibration": str(tmp_path / "calibration"),
    }

    class RawReached(Exception):
        pass

    reached = []

    def open_run(*args, **kwargs):
        reached.append(True)
        raise RawReached()

    monkeypatch.setattr(
        portfolio,
        "read_dataset_manifest_evidence",
        lambda *args: {
            "producer_diagnostics": {
                "reconstruction": {"BTCUSDT": {"coverage_segments": {}}}
            },
        },
    )

    def produce(on_table):
        for name, frame in tables.items():
            on_table(name, frame)

    write_statistics_artifact(
        tmp_path / "calibration",
        metadata=metadata,
        tables=produce,
        table_group="crypto_maker_calibration",
        validate_tables=validate_maker_calibration,
    )
    monkeypatch.setattr(portfolio, "open_market_source", open_run)

    def prediction_evidence(predictions, *_args, **_kwargs):
        model = predictions.removesuffix("-predictions")
        index = metadata["model_ids"]["fold_2"].index(model)
        return (
            {},
            {
                "model": {
                    "model_id": model,
                    "target_column": (
                        "future_mid_return_500ms",
                        "maker_bid_sell_pressure_proxy_500ms",
                    )[index],
                }
            },
            None,
        )

    monkeypatch.setattr(portfolio, "_prediction_evidence", prediction_evidence)
    with pytest.raises(
        RawReached if last_defined else PortfolioTargetBlocked
    ) as caught:
        portfolio.backtest_maker(config)
    if last_defined:
        assert reached == [True]
    else:
        assert str(caught.value) == "maker_calibration_insufficient_samples"
        assert not reached


@pytest.mark.parametrize(
    "current_unit,accepted", (("1", False), ("0.3", False), ("0.01", True))
)
def test_frozen_maker_coordinates_require_an_exact_integer_rescale(
    current_unit, accepted
):
    from tests.markets.crypto.calibration_test_support import calibration_from_paths
    from tests.markets.crypto.test_hft_execution import _trade

    history = _state(0)
    model = calibration_from_paths(
        [([history], [_trade(100, price="99.5", quantity="3", maker_buyer=True)])],
        [0.0],
    )
    factor = 10 if accepted else 1
    current = replace(
        history,
        price_unit=current_unit,
        bids=tuple((tick * factor, qty) for tick, qty in history.bids),
        asks=tuple((tick * factor, qty) for tick, qty in history.asks),
    )
    order = _maker_new_order(1, "buy", 1.0, current, _quote(0, book_segment_id="s1"), 0)
    query = dict(
        state=current,
        decision_ns=0,
        new_order=order,
        active_order=None,
        commands=(),
        lifetime_ns=500000000,
        prediction=0.0,
        fee_bps=0.0,
        funding=[],
    )
    if accepted:
        assert model.select(**query)[1] == 1.0
    else:
        with pytest.raises(ValueError, match="calibration_price_unit_incompatible"):
            model.select(**query)

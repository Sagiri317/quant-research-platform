"""Frozen per-asset maker coordinates survive an expanded daily source window."""

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from functools import partial
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    _maker_new_order,
)
from quant_research_platform.markets.implementations.crypto.portfolio import (
    _calibration_events,
    _calibration_samples,
    validate_maker_calibration,
)
from quant_research_platform.profiles.research import (
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.protocols.crypto_hft import CryptoMakerCandidate
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.tardis.raw import import_slices
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.software_identity import (
    production_software_identity,
)
from quant_research_platform.workflows import research
from tests.artifacts.test_statistics_artifact import maker_calibration_tables
from tests.helpers.binance_hft import _quote, hft_research_plan
from tests.helpers.research_configuration import research_configuration
from tests.markets.crypto.test_hft_execution import _maker_config, _state
from tests.sources.tardis.test_raw import _slices


def _publish(root, date, policy, *, fine):
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
        byte_budget=1024**2,
        resource_policy=policy,
    )
    return destination.relative_to(root).as_posix()


@pytest.mark.parametrize(
    "asset,unit,current_unit", [("btc_usdt", "1", "0.1"), ("eth_usdt", "0.01", "0.001")]
)
def test_frozen_archive_identity_and_persisted_maker_units(
    tmp_path, asset, unit, current_unit
):
    tables, _ = maker_calibration_tables.__wrapped__()
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
    config = research_configuration(
        "crypto",
        "baseline",
        plan=plan,
        development=tuple(plan["folds"][0]["validation"].values()),
    )
    next(
        item
        for item in config["portfolios"]
        if item["candidate_id"] == "crypto_hft_maker"
    )["asset_id"] = asset
    raw = tmp_path / "raw"
    raw.mkdir()
    layout = StorageLayout(tmp_path / "project")
    symbol = {"btc_usdt": "BTCUSDT", "eth_usdt": "ETHUSDT"}[asset]
    with runtime_resource_policy_scope() as policy:
        config["source_archives"] = [
            _publish(raw, day, policy, fine=index == 2)
            for index, day in enumerate(("2026-07-01", "2026-08-01", "2026-09-01"))
        ]
        profiles = {
            access: bind_research_configuration(
                research_profile(config["profile_id"]),
                config,
                tardis_root=raw,
                access_window=access,
            )
            for access in ("development", "holdout")
        }
        protocols = {
            access: profile.protocol(config["scenario_id"], access_window=access)
            for access, profile in profiles.items()
        }
        candidate = next(
            item
            for item in protocols["development"].portfolio_candidates()
            if isinstance(item, CryptoMakerCandidate)
            and item.calibrated
            and item.asset_id == asset
        )
        target_id = candidate.calibration_target_id
        specs = {
            access: next(
                item
                for item in protocol.target_specs("research run")
                if item.target_id == target_id
            )
            for access, protocol in protocols.items()
        }
        identities = {}
        for access, profile in profiles.items():
            binding = profile.producer_bindings[target_id]
            identities[access] = binding.source_identity(raw, target_id)
            with binding.producer.keywords["open_run"](raw, policy) as source:
                assert len(source.archives) == 2
                assert (
                    source.market_metadata(FUTURES_MARKET_TYPE, symbol)["price_unit"]
                    == unit
                )
        assert identities["development"] == identities["holdout"]
        panel = profiles["holdout"].producer_bindings["binance_futures_l2_panel_100ms"]
        with panel.producer.keywords["open_run"](raw, policy) as source:
            assert len(source.archives) == 3
            assert (
                source.market_metadata(FUTURES_MARKET_TYPE, symbol)["price_unit"]
                == current_unit
            )
        envelope = research._target_envelope_for_spec(
            profile=profiles["development"],
            protocol=protocols["development"],
            target_spec=specs["development"],
            data_spec=None,
            research_window={
                **config["windows"]["development"],
                "access": "development",
            },
            upstream_identities={
                key: "1" * 64 for key in specs["development"].dependencies
            },
            software_identity=production_software_identity(),
            raw_content_identity=identities["development"],
        )

        def produce(on_table):
            for name, frame in tables.items():
                on_table(name, frame)

        root = write_statistics_artifact(
            research._target_logical_root(profiles["development"], layout, target_id),
            envelope=envelope,
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
        for access, profile in profiles.items():
            result = research._reuse_target(
                profile=profile,
                protocol=protocols[access],
                layout=layout,
                spec=specs[access],
                planned={
                    "requested_identity": envelope.instance_identity,
                    "matching_instance_identity": envelope.instance_identity,
                },
                policy=policy,
                raw_root=raw,
                time_window=TimeWindow(
                    envelope.research_window["start"], envelope.research_window["end"]
                ),
            )
            assert result.status.value == "REUSED"
        with policy.lease("tests.frozen_calibration", 0) as lease:
            artifact = read_statistics_artifact(
                root,
                expected_identity=envelope.instance_identity,
                validate_tables=validate_maker_calibration,
                lease=lease,
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
                                    (tick * 10, quantity)
                                    for tick, quantity in values[side]
                                )
                                for side in ("bids", "asks")
                            }
                        )
                    elif event.kind == "trade":
                        values["price_index"] *= 10
                    yield SimpleNamespace(**values)

            historical = MakerCalibration(
                samples, events, price_unit=frozen_unit, resource_policy=policy
            )
            baseline = MakerCalibration(
                fine_samples,
                fine_events,
                price_unit=current_unit,
                resource_policy=policy,
            )
            state = replace(
                _state(0),
                price_unit=current_unit,
                bids=tuple((tick * 10, quantity) for tick, quantity in _state(0).bids),
                asks=tuple((tick * 10, quantity) for tick, quantity in _state(0).asks),
            )
            new = _maker_new_order(
                2, "buy", 1.0, state, _quote(0, book_segment_id="s1"), _maker_config()
            )
            new.effective_at_ns = 100_000_000
            old = replace(
                new,
                order_id=1,
                effective_at_ns=0,
                price_index=new.price_index + 50,
                queue_ahead=0,
            )
            pending = (
                (
                    200_000_000,
                    "buy",
                    replace(new, order_id=3, effective_at_ns=200_000_000),
                ),
            )
            try:
                for commands in ((), pending):
                    query = dict(
                        state=state,
                        decision_ns=0,
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
                                            decision_ns=0,
                                            new_order=new,
                                            active_order=old,
                                            commands=commands,
                                            lifetime_ns=500_000_000,
                                            replace_quantity=replacement,
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
            finally:
                historical.close()
                baseline.close()
        assert fingerprints() == before
        assert policy.used_memory_bytes == 0

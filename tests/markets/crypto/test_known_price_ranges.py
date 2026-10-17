"""Public preparation and execution consume only causally proved book ranges."""

import json
from dataclasses import asdict
from decimal import Decimal

import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.markets.implementations.crypto.processed import producer
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.helpers.binance_hft import write_binance_hft_run, write_hft_funding_fixture
from tests.markets.crypto.test_independent_execution_source import (
    _ledger,
    _mutate,
    _panel,
)


def _source(root, delayed_bridge, levels):
    write_binance_hft_run(root, count=90, maker_liquidity=True)
    initial = [[str(Decimal("99.5") - Decimal("0.1") * i), "2"] for i in range(10)]
    outside = [[str(Decimal("98.4") - Decimal("0.1") * i), "2"] for i in range(levels)]

    def change(row, member):
        if member["symbol"] != "BTCUSDT":
            return
        payload = row["payload"]
        if member["data_kind"] == "depth_snapshot":
            payload["bids"] = initial
        elif member["data_kind"] == "depth":
            timestamp = payload["T"]
            payload["b"] = []
            payload["a"] = []
            if timestamp == 100:
                payload["b"] = outside + ([] if delayed_bridge else [["98.5", "0"]])
            elif timestamp == 200:
                payload["b"] = [[p, "0"] for p, _q in initial]
            elif timestamp == 300 and delayed_bridge:
                payload["b"] = [["98.5", "0"]]
        elif member["data_kind"] == "agg_trades":
            payload.update(p="98.4", m=True)

    manifest = _mutate(root, change)
    (root / "run.json").write_text(json.dumps(manifest))
    write_hft_funding_fixture(root)


@pytest.mark.parametrize("levels", [1, 10])
@pytest.mark.parametrize("delayed_bridge", [False, True])
def test_public_prepare_batch_invariance_and_maker_taker_use_proven_ranges(
    tmp_path, monkeypatch, delayed_bridge, levels
):
    raw = tmp_path / "source" / "raw"
    _source(raw, delayed_bridge, levels)
    full = {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:09"}
    frames = []
    for index, batch_rows in enumerate((2, 17, 10000)):
        monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
        panel = _panel(tmp_path / f"panel{index}", raw, full)
        frames.append(read_dataset(panel))
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], frame, check_exact=True)
    frame = frames[0]
    btc = frame.loc[frame.asset_id.eq("btc_usdt")]
    assert len(btc) == (89 if delayed_bridge else 91)
    after = btc.loc[btc.timestamp.ge(pd.Timestamp("1970-01-01T00:00:00.300"))]
    assert after.mid.eq((98.4 + 100.5) / 2).all()
    assert after.book_spread.eq(2.1).all()
    if delayed_bridge:
        # A recovered state at .300 cannot certify its preceding 100ms bin.
        assert not btc.timestamp.isin(
            pd.to_datetime(["1970-01-01T00:00:00.200", "1970-01-01T00:00:00.300"])
        ).any()
        assert after.book_segment_id.str.contains(":recovery:").all()
    else:
        assert btc.book_segment_id.nunique() == 1

    # Explicit synthetic labels isolate source support from predictive research.
    target = "future_mid_return_500ms"
    frame = frame.copy()
    frame["x"] = frame.timestamp.astype("int64") / 1e9
    frame[target] = -0.03
    frame[target + "_available_at"] = frame.timestamp + pd.Timedelta(milliseconds=500)
    roles = dict.fromkeys(frame, "context")
    roles.update(
        market_id="key",
        asset_id="key",
        timestamp="timestamp",
        eligible="mask",
        x="feature",
    )
    roles[target] = "label"
    inputs = tmp_path / "model-inputs"
    write_dataset(
        inputs,
        frame,
        spec=DataSpec(
            data_id="synthetic",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
        ),
    )
    model = fit(
        {
            "market_id": "crypto",
            "inputs": str(inputs),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "price",
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["x"],
                "target_column": target,
                "random_seed": 17,
            },
            "fit_windows": [{"start": full["start"], "end": "1970-01-01T00:00:02"}],
            "fit_cutoff": "1970-01-01T00:00:02.600",
            "label_unit": "decimal_return",
        }
    )
    window = {"start": "1970-01-01T00:00:08.600", "end": full["end"]}
    predictions = predict(
        {
            "model": str(model),
            "inputs": str(inputs),
            "output": str(tmp_path / "predictions"),
            "prediction_window": window,
            "sample_role": "test",
        }
    )
    common = {
        "inputs": str(inputs),
        "panel": str(panel),
        "source": str(raw),
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "prediction_window": window,
        "execution_mode": "continuous",
        "capital": asdict(
            PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual")
        ),
    }
    taker, taker_fills = _ledger(
        portfolio.backtest_taker(
            {
                **common,
                "output": str(tmp_path / "taker"),
                "predictions": str(predictions),
                "execution": asdict(portfolio.TakerConfig(5.0, 5.0, 0, 0.0, 100)),
            }
        )
    )
    assert taker["execution"]["account_path_status"] == "COMPLETE"
    assert taker["execution"]["filled_orders"] > 0
    assert taker_fills
    if levels == 1:
        assert all(abs(fill.quantity) <= 2.0 for fill in taker_fills)
        assert all(
            abs(float(fill.notional / fill.quantity)) == pytest.approx(98.4)
            for fill in taker_fills
        )
    maker, maker_fills = _ledger(
        portfolio.backtest_maker(
            {
                **common,
                "output": str(tmp_path / "maker"),
                "models": [],
                "calibration": None,
                "execution": asdict(
                    portfolio.MakerConfig(
                        order_quantity=1.0,
                        max_position=2.0,
                        maker_fee_bps=0.0,
                        inventory_target=2.0,
                        inventory_skew=1.0,
                        markout_horizon_ms=100,
                        latency_ms=0,
                    )
                ),
            }
        )
    )
    assert maker["execution"]["account_path_status"] == "COMPLETE"
    assert maker["execution"]["filled_orders"] > 0
    assert maker_fills

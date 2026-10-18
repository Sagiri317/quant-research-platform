"""Public native execution binds independent source UUIDs without changing economics."""

import copy
import datetime as dt
import gzip
import hashlib
import json
import shutil
from dataclasses import asdict, replace

import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.helpers.binance_hft import (
    _write_file,
    write_binance_hft_run,
    write_hft_funding_fixture,
)
from tests.research.test_crypto_execution_partial import _close_depth

WINDOW = {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:01"}


def _assert_same_source_economics(actual, expected, fills, expected_fills):
    # Recomputed monetary amounts/estimates are not copied source facts.
    numeric = {
        "net_pnl",
        "gross_pnl",
        "nav",
        "capital_return",
        "fees",
        "funding",
        "turnover",
        "average_slippage_bps",
        "spread_capture_pnl",
        "average_markout_per_unit",
        "average_markout_bps",
        "markout_pnl",
        "adverse_selection_cost",
        "average_order_lifetime_ms",
        "average_filled_order_lifetime_ms",
    }
    for section in ("economics", "execution"):
        left, right = actual[section], expected[section]
        assert left.keys() == right.keys()
        for key, value in left.items():
            if key in numeric and value is not None:
                assert right[key] is not None
                assert value == pytest.approx(right[key], rel=2e-13, abs=0)
            else:
                assert value == right[key]
    for left, right in zip(fills, expected_fills, strict=True):
        assert replace(left, notional=right.notional) == right
        assert left.notional == pytest.approx(right.notional, rel=2e-15, abs=0)


def _mutate(root, transform):
    manifest = json.loads((root / "run.json").read_text())
    for index, member in enumerate(manifest["files"]):
        with gzip.open(root / member["path"], "rt") as stream:
            rows = [json.loads(line) for line in stream]
        for row in rows:
            transform(row, member)
        manifest["files"][index] = _write_file(
            root,
            symbol=member["symbol"],
            data_kind=member["data_kind"],
            stream=member["stream"],
            rows=rows,
        )
    return manifest


def _independent_identity(root):
    def rename(row, _member=None):
        for key in ("connection_id", "segment_id"):
            if row.get(key) is not None:
                row[key] = "independent-source:" + row[key]

    manifest = _mutate(root, rename)
    manifest["run_id"] = "independent-source"
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(sidecar, "rt") as stream:
        intervals = [json.loads(line) for line in stream]
    for row in intervals:
        rename(row)
    payload = gzip.compress(
        "".join(json.dumps(row) + "\n" for row in intervals).encode(), mtime=0
    )
    sidecar.write_bytes(payload)
    manifest["capture"]["connection_intervals"].update(
        compressed_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    (root / "run.json").write_text(json.dumps(manifest))
    return manifest


def _economics(root):
    result = []
    for member in json.loads((root / "run.json").read_text())["files"]:
        with gzip.open(root / member["path"], "rt") as stream:
            result.append(
                (
                    member["symbol"],
                    member["data_kind"],
                    [
                        {
                            key: value
                            for key, value in json.loads(line).items()
                            if key not in ("connection_id", "segment_id")
                        }
                        for line in stream
                    ],
                )
            )
    return result


def _panel(root, source, window=WINDOW):
    return prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(source)},
            "time_window": window,
            "output_root": str(root),
        }
    )


def _maker(root, panel, source):
    output = portfolio.backtest_maker(
        {
            "output": str(root),
            "inputs": str(panel),
            "panel": str(panel),
            "source": str(source),
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
            "prediction_window": WINDOW,
            "execution_mode": "continuous",
            "capital": asdict(
                PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(portfolio.MakerConfig(1.0, 2.0, 1.0, 0.0, 0.0, 100, 0)),
            "models": [],
            "calibration": None,
        }
    )
    return _ledger(output)


def _ledger(output):
    artifact = read_portfolio_ledger_artifact(output)
    batches = list(artifact.iter_chunks())
    summary = summarize_crypto_ledger(artifact, iter(batches))
    return summary, [trade for batch in batches for trade in batch.trades]


@pytest.mark.parametrize(
    "change", ["identity", "closed", "price_unit", "opposite", "future"]
)
def test_public_maker_uses_independent_source_context_and_real_events(
    tmp_path, change, monkeypatch
):
    first, second = tmp_path / "first" / "raw", tmp_path / "second" / "raw"
    write_binance_hft_run(first, count=10, maker_liquidity=True)
    write_binance_hft_run(second, count=10, maker_liquidity=True)
    manifest = _independent_identity(second)
    write_hft_funding_fixture(second)
    assert _economics(first) == _economics(second)
    old_panel = _panel(tmp_path / "first-panel", first)
    new_panel = _panel(tmp_path / "second-panel", second)
    baseline, baseline_fills = _maker(tmp_path / "baseline", new_panel, second)
    assert baseline["execution"]["filled_orders"] == 6
    selected = second
    if change == "identity":
        read_evidence = portfolio.read_dataset_manifest_evidence

        def independent_panel_depth(path):
            evidence = copy.deepcopy(read_evidence(path))
            if path == old_panel:
                evidence["source_evidence"]["market_metadata"]["BTCUSDT"][
                    "snapshot_limit"
                ] = 100
            return evidence

        monkeypatch.setattr(
            portfolio, "read_dataset_manifest_evidence", independent_panel_depth
        )
    if change != "identity":
        selected = tmp_path / "changed" / "raw"
        shutil.copytree(second.parent, selected.parent)
        if change == "closed":
            _close_depth(selected, manifest)
        else:

            def modify(row, member):
                if (
                    change == "price_unit"
                    and member["symbol"] == "BTCUSDT"
                    and member["data_kind"] == "metadata"
                ):
                    for item in row["payload"]["filters"]:
                        if item["filterType"] == "PRICE_FILTER":
                            item["tickSize"] = "0.01"
                if member["data_kind"] == "agg_trades":
                    if change == "opposite":
                        row["payload"]["m"] = not row["payload"]["m"]
                    elif change == "future":
                        row["observed_time_ns"] += 2_000_000_000
                        stamp = dt.datetime.fromisoformat(
                            row["receive_time"].replace("Z", "+00:00")
                        ) + dt.timedelta(seconds=2)
                        row["receive_time"] = stamp.isoformat(
                            timespec="microseconds"
                        ).replace("+00:00", "Z")
                        for key in ("E", "T"):
                            row["payload"][key] += 2000

            manifest = _mutate(selected, modify)
            if change == "price_unit":
                manifest["market_metadata"]["BTCUSDT"]["usd_s_m_futures"][
                    "tick_size"
                ] = "0.01"
            (selected / "run.json").write_text(json.dumps(manifest))
    if change == "price_unit":
        with pytest.raises(
            ValueError, match="crypto_execution_source_coordinates_mismatch"
        ):
            _maker(tmp_path / "actual", old_panel, selected)
        return
    result, fills = _maker(tmp_path / "actual", old_panel, selected)
    if change == "identity":
        _assert_same_source_economics(result, baseline, fills, baseline_fills)
    elif change == "closed":
        assert result["execution"]["account_path_status"] == "PARTIAL"
        assert result["economics"]["net_pnl"] is None
        assert fills and max(trade.timestamp_ns for trade in fills) < 500_000_000
    else:
        assert result["execution"]["filled_orders"] == 0
        assert not fills


def test_public_taker_frozen_predictions_use_actual_source_identity(tmp_path):
    first, second = tmp_path / "first" / "raw", tmp_path / "second" / "raw"
    for source in (first, second):
        write_binance_hft_run(source, count=90, maker_liquidity=True)
        write_hft_funding_fixture(source)
    _independent_identity(second)
    assert _economics(first) == _economics(second)
    full = {"start": WINDOW["start"], "end": "1970-01-01T00:00:09"}
    panel = _panel(tmp_path / "panel", first, full)
    inputs = tmp_path / "model-inputs"
    frame = read_dataset(panel)
    target = "future_mid_return_500ms"
    frame["x"] = frame.mid
    # Explicit synthetic targets isolate source qualification from model estimation.
    frame[target] = 0.03
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
            "fit_windows": [{"start": WINDOW["start"], "end": "1970-01-01T00:00:02"}],
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
    results = []
    for index, source in enumerate((first, second)):
        output = portfolio.backtest_taker(
            {
                "output": str(tmp_path / f"ledger{index}"),
                "inputs": str(inputs),
                "panel": str(panel),
                "source": str(source),
                "asset_id": "btc_usdt",
                "symbol": "BTCUSDT",
                "prediction_window": window,
                "predictions": str(predictions),
                "execution_mode": "continuous",
                "capital": asdict(
                    PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual")
                ),
                "execution": asdict(portfolio.TakerConfig(1.0, 2.0, 0, 0.0, 100)),
            }
        )
        results.append(_ledger(output))
    assert results[0][0]["execution"]["filled_orders"] > 0
    _assert_same_source_economics(
        results[0][0], results[1][0], results[0][1], results[1][1]
    )


@pytest.mark.parametrize("levels", [1, 10])
def test_public_calibration_and_frozen_execution_bind_selected_native_source(
    tmp_path, monkeypatch, levels
):
    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from tests.markets.crypto.test_direct_portfolio import _independent_maker_case

    calibration, backtest = _independent_maker_case(tmp_path, monkeypatch)
    # Restore the real native reader after the existing fixture has prepared models.
    monkeypatch.undo()
    raw = tmp_path / "native" / "raw"
    write_binance_hft_run(raw, count=90, maker_liquidity=True)
    write_hft_funding_fixture(raw)
    _independent_identity(raw)
    if levels == 1:

        def remove_deep_levels(row, member):
            if member["data_kind"] == "depth" and row["payload"]["T"] == 100:
                offset = 0 if member["symbol"] == "BTCUSDT" else 100
                for side, best, sign in (
                    ("b", 99.5 + offset, -1),
                    ("a", 100.5 + offset, 1),
                ):
                    row["payload"][side].extend(
                        [f"{best + sign * index:.1f}", "0"] for index in range(1, 10)
                    )

        manifest = _mutate(raw, remove_deep_levels)
        (raw / "run.json").write_text(json.dumps(manifest))
    panel = _panel(
        tmp_path / "native-panel",
        raw,
        {"start": WINDOW["start"], "end": "1970-01-01T00:00:09"},
    )
    for config in (calibration, backtest):
        config.update(source=str(raw), panel=str(panel))
    output = portfolio.calibrate_maker(calibration)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_samples.parquet") as samples:
        assert len(samples) == 4
        assert samples.book_segment_id.str.contains("independent-source:").all()
        assert samples.trade_observation_id.str.contains("independent-source:").all()
    summary, _fills = _ledger(portfolio.backtest_maker(backtest))
    assert summary["execution"]["selection_status"] == "COMPLETE"
    # Independently specified validation folds may overlap; each reader owns its cursor.
    calibration["output"] = str(tmp_path / "overlapping-calibration")
    calibration["models"].append({**calibration["models"][0], "fold_id": "overlapping"})
    artifact = read_statistics_artifact(
        portfolio.calibrate_maker(calibration),
        validate_tables=portfolio.validate_maker_calibration,
    )
    with artifact.table("crypto_maker_calibration.parquet") as summary:
        assert summary.sample_count.tolist() == [2, 2, 2, 2]


@pytest.mark.parametrize(
    "trained_basis,current_basis,reason",
    [
        ("mid", "mid", None),
        ("mid", "mark", "prediction_target_definition_mismatch"),
        ("mid", None, "prediction_target_definition_missing"),
        (None, "mid", "prediction_target_definition_missing"),
        (None, None, None),
    ],
)
def test_native_calibration_residual_preserves_target_meaning(
    tmp_path, trained_basis, current_basis, reason
):
    from pathlib import Path

    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from tests.helpers.binance_hft import maker_config
    from tests.markets.crypto.test_direct_portfolio import _inputs, _window

    original = _inputs(tmp_path)
    frame = read_dataset(original)
    spec = portfolio.open_dataset_session(original).spec
    price = "future_mid_return_500ms"
    proxy = "maker_bid_sell_pressure_proxy_500ms"
    paths = {}
    for name, basis in (("training", trained_basis), ("current", current_basis)):
        semantics = {proxy: {"source": name}}
        if basis is not None:
            semantics[price] = {"price_basis": basis}
        data = frame.copy()
        if name == "current":
            data[price] = 0.02
        paths[name] = tmp_path / name
        write_dataset(
            paths[name],
            data,
            spec=DataSpec(
                data_id=name,
                frequency="tick",
                grain="asset_timestamp",
                primary_key=("timestamp", "asset_id"),
                partition_columns=("year",),
                column_roles=spec["column_roles"],
            ),
            metadata={"producer_diagnostics": {"label_semantics": semantics}},
        )
    observed, execution = _window(2100, 2200), _window(8600, 9000)
    item = {
        "fold_id": "observed",
        "inputs": str(paths["current"]),
        "window": observed,
        "cutoff": _window(0, 2700)["end"],
    }
    models = []
    for role, target, unit in (
        ("price", price, "decimal_return"),
        ("proxy", proxy, "quantity_ratio"),
    ):
        model = fit(
            {
                "market_id": "crypto",
                "inputs": str(paths["training"]),
                "output": str(tmp_path / (role + "-model")),
                "model": {
                    "model_id": role,
                    "model_algorithm": "linear_regression",
                    "input_feature_columns": ["x"],
                    "target_column": target,
                },
                "fit_windows": [_window(0, 1000)],
                "fit_cutoff": _window(0, 2000)["end"],
                "label_unit": unit,
            }
        )
        for purpose, window in (("calibration", observed), ("execution", execution)):
            predictions = predict(
                {
                    "model": str(model),
                    "inputs": str(paths["current"]),
                    "output": str(tmp_path / (role + "-" + purpose)),
                    "prediction_window": window,
                }
            )
            if purpose == "calibration":
                item[role + "_predictions"] = str(predictions)
            else:
                models.append({"predictions": str(predictions)})
    raw = tmp_path / "native" / "raw"
    write_binance_hft_run(raw, count=90, maker_liquidity=True)
    write_hft_funding_fixture(raw)
    panel = _panel(tmp_path / "panel", raw, _window(0, 9000))
    common = {
        "panel": str(panel),
        "source": str(raw),
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
    }
    # Unconsumed years do not become calibration metadata dependencies.
    unselected = paths["current"] / "data" / "year=1971"
    unselected.mkdir()
    (unselected / "part-00000.parquet").write_bytes(b"unselected broken partition")
    config = {
        **common,
        "output": str(tmp_path / "calibration"),
        "models": [item],
        "reference": {
            "order_quantity": 1.0,
            "latency_ms": 10,
            "min_calibration_samples": 1,
        },
    }
    if reason is not None:
        with pytest.raises(ValueError, match=reason):
            portfolio.calibrate_maker(config)
        assert not Path(config["output"]).exists()
        return
    artifact = read_statistics_artifact(
        portfolio.calibrate_maker(config),
        validate_tables=portfolio.validate_maker_calibration,
    )
    with artifact.table("crypto_maker_samples.parquet") as samples:
        assert len(samples) == 4
        assert samples.price_prediction.tolist() == pytest.approx([0.0] * 4)
        assert samples.observed_return.tolist() == [0.02] * 4
        assert samples.forecast_error.tolist() == pytest.approx([0.02] * 4)
    # Execution consumes signals and the frozen residual, not current label values
    # or definitions. Drop labels altogether after calibration.
    labels = [price, proxy, price + "_available_at", proxy + "_available_at"]
    context = tmp_path / "context"
    write_dataset(
        context,
        frame.drop(columns=labels),
        spec=DataSpec(
            data_id="context",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
            column_roles={
                k: v for k, v in spec["column_roles"].items() if k not in labels
            },
        ),
    )
    output = portfolio.backtest_maker(
        {
            **common,
            "output": str(tmp_path / "ledger"),
            "inputs": str(context),
            "models": models,
            "calibration": config["output"],
            "prediction_window": execution,
            "execution_mode": "continuous",
            "capital": asdict(
                PortfolioCapital("maker", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(maker_config(calibrated=True)),
        }
    )
    ledger = read_portfolio_ledger_artifact(output)
    gains = [
        diagnostic.number
        for batch in ledger.iter_chunks()
        for diagnostic in batch.diagnostics
        if diagnostic.name == "maker_buy_expected_net_gain"
    ]
    # One of the two historical buy paths fills. At 8.6s the new mid is
    # 100.2 and bid 99.7; the maker rebate is 1bp and residual return 2%.
    expected_gain = (100.2 - 99.7 + 99.7 * 0.0001 + 100.2 * 0.02) / 2
    assert gains[0] == pytest.approx(expected_gain, rel=1e-12)

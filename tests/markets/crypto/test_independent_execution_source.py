"""Public native execution binds independent source UUIDs without changing economics."""

import datetime as dt
import gzip
import hashlib
import json
import shutil
from dataclasses import asdict

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
def test_public_maker_uses_independent_source_context_and_real_events(tmp_path, change):
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
        assert result["execution"] == baseline["execution"]
        assert result["economics"] == baseline["economics"]
        assert fills == baseline_fills
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
    assert results[0][0]["economics"] == results[1][0]["economics"]
    assert results[0][0]["execution"] == results[1][0]["execution"]
    assert results[0][1] == results[1][1]


def test_public_calibration_and_frozen_execution_bind_selected_native_source(
    tmp_path, monkeypatch
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

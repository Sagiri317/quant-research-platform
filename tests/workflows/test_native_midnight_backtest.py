"""Native cross-midnight science reaches frozen public backtests without adapters."""

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto import ids, portfolio
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.models import fitting, production, trial_selection
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.sources.tardis.funding import import_funding_history
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.experiments import read_trial_ledger
from tests.helpers.binance_hft import hft_research_plan
from tests.helpers.tardis_midnight import write_tardis_midnight_case
from tests.markets.crypto.test_hft_execution import _maker_config, _taker_config


def _bytes(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _prepare_midnight(tmp_path, gap_kind):
    raw, origin = write_tardis_midnight_case(tmp_path, gap_kind)
    settlements = tmp_path / "settlements.json"
    settlements.write_text("[]", encoding="utf-8")
    start_ms = pd.Timestamp("2026-09-01T23:50:00Z").value // 1_000_000
    with open_market_source(
        raw, time_window=TimeWindow("2026-09-01", "2026-09-01")
    ) as source:
        for symbol in ("BTCUSDT", "ETHUSDT"):
            import_funding_history(
                source,
                symbol=symbol,
                source_id="binance",
                request_limit=1000,
                response_path=settlements,
                http_status=200,
                start_ms=start_ms,
                end_ms=start_ms + 1_200_000 - 1,
            )
    chronology = hft_research_plan()
    chronology = {key: chronology[key] for key in ("folds", "test")}
    for window in [
        *(fold[key] for fold in chronology["folds"] for key in ("train", "validation")),
        chronology["test"],
    ]:
        for key, value in window.items():
            window[key] = (
                pd.Timestamp(value) + pd.Timedelta(origin.value, unit="ns")
            ).isoformat()
    window = {
        "start": origin.tz_localize(None).isoformat(),
        "end": (origin.tz_localize(None) + pd.Timedelta(seconds=11)).isoformat(),
    }
    roots = {}
    for data_id in (
        ids.HFT_PANEL_DATA_ID,
        ids.HFT_FACTOR_DATA_ID,
        ids.HFT_LABEL_DATA_ID,
    ):
        config = {
            "market_id": "crypto",
            "data_id": data_id,
            "inputs": {}
            if not roots
            else {ids.HFT_PANEL_DATA_ID: roots[ids.HFT_PANEL_DATA_ID]},
            "time_window": window,
            "output_root": str(tmp_path / data_id),
        }
        if data_id == ids.HFT_PANEL_DATA_ID:
            config["source"] = {"root": str(raw)}
        roots[data_id] = str(prepare(config))
    inputs = assemble(
        {
            "market_id": "crypto",
            "dataset_id": ids.HFT_RESEARCH_DATASET_ID,
            "inputs": roots,
            "factor_columns": ["return_100ms", "relative_spread"],
            "context_columns": list(portfolio.HFT_EXECUTION_CONTEXT_COLUMNS),
            "time_window": window,
            "output_root": str(tmp_path / "research"),
        }
    )
    frame = read_dataset(inputs)
    assert frame.timestamp.dt.date.nunique() == 2
    assert set(frame.asset_id) == {"btc_usdt", "eth_usdt"}
    midnight = origin.tz_localize(None).ceil("D")
    across = frame.loc[frame.timestamp.eq(midnight)]
    assert len(across) == 2
    assert across.return_100ms.notna().all()
    assert across.future_mid_return_500ms.notna().all()
    return raw, Path(roots[ids.HFT_PANEL_DATA_ID]), inputs, chronology


def _freeze_models(root, inputs, chronology):
    validation = [
        dict(fold_id=fold["fold_id"], inputs=str(inputs))
        for fold in chronology["folds"]
    ]
    final = []
    validation_windows = [fold["validation"] for fold in chronology["folds"]]
    cutoff = validation_windows[-1]["end"]
    refit_windows = [chronology["folds"][-1]["train"], validation_windows[-1]]
    for role, target, unit in (
        ("price", "future_mid_return_500ms", "decimal_return"),
        ("proxy", "maker_bid_sell_pressure_proxy_500ms", "quantity_ratio"),
    ):
        config = {
            "market_id": "crypto",
            "inputs": str(inputs),
            "model": {
                "model_id": role,
                "model_algorithm": "linear_regression",
                "input_factor_columns": ["return_100ms", "relative_spread"],
                "target_column": target,
                "random_seed": 17,
            },
            "label_unit": unit,
        }
        predictions = []
        for fold, item in zip(chronology["folds"], validation, strict=True):
            name = role + "-" + fold["fold_id"]
            model = production.fit(
                {
                    **config,
                    "output": str(root / name),
                    "fit_windows": [fold["train"]],
                    "fit_cutoff": fold["train"]["end"],
                }
            )
            pred = predict(
                {
                    "model": str(model),
                    "inputs": str(inputs),
                    "output": str(root / (name + "-validation")),
                    "prediction_window": fold["validation"],
                    "sample_role": "validation",
                }
            )
            predictions.append(str(pred))
            item[role + "_predictions"] = str(pred)
        scores = trial_selection.score(
            {
                "predictions": [predictions],
                "labels": str(inputs),
                "candidate_ids": [role],
                "validation_windows": validation_windows,
                "scoring_window": {
                    "start": validation_windows[0]["start"],
                    "end": cutoff,
                },
                "cutoff": cutoff,
                "selection_group": "global",
                "validation_axis": "cross_sectional",
                "output": str(root / (role + "-scores.json")),
                "label_unit": unit,
            }
        )
        selection = trial_selection.select(
            {
                "scores": str(scores),
                "output": str(root / (role + "-selection.json")),
                "selection_metric": "validation_mse",
                "selection_direction": "minimize",
                "tie_break_order": [role],
                "refit_windows": refit_windows,
                "refit_cutoff": cutoff,
            }
        )
        model = production.fit(
            {
                **config,
                "output": str(root / (role + "-final")),
                "selection": str(selection),
                "fit_windows": refit_windows,
                "fit_cutoff": cutoff,
            }
        )
        result, facts = load_model(model)
        assert facts["selection"] == str(selection)
        assert facts["fit_windows"] == refit_windows
        assert "test_window" not in facts
        assert set(result.metadata["training_asset_ids"]) == {"btc_usdt", "eth_usdt"}
        final.append({"model": str(model)})
    trials = list(root.glob("*.trial"))
    assert len(trials) == 6
    for path in trials:
        assert [item["status"] for item in read_trial_ledger(path)] == [
            "attempted",
            "completed",
        ]
    return validation, final


@pytest.mark.complete_chain
@pytest.mark.parametrize("gap_kind", (None, "depth", "agg_trades"))
def test_native_midnight_models_reach_frozen_backtests(tmp_path, monkeypatch, gap_kind):
    raw, panel, inputs, chronology = _prepare_midnight(tmp_path, gap_kind)
    # A future archive may have different coordinates or damaged raw members;
    # it must not enter either calibration or OOS source qualification.
    from tests.workflows.test_tardis_frozen_calibration import _publish

    future = _publish(tmp_path, "2026-09-03", fine=True)
    for member in (tmp_path / future).rglob("*.gz"):
        member.write_bytes(b"unreadable future source")
    archive_paths = [raw.relative_to(tmp_path).as_posix(), future]
    learned = tmp_path / "learned"
    validation, models = _freeze_models(learned, inputs, chronology)
    calibrations = {}
    for asset, symbol in (("btc_usdt", "BTCUSDT"), ("eth_usdt", "ETHUSDT")):
        calibration = portfolio.calibrate_maker(
            {
                "output": str(learned / (asset + "-calibration")),
                "panel": str(panel),
                "source": str(tmp_path),
                "archive_paths": archive_paths,
                "asset_id": asset,
                "symbol": symbol,
                "chronology": chronology,
                "reference": {
                    "order_quantity": 1.0,
                    "latency_ms": 10,
                    "min_calibration_samples": 2,
                },
                "models": validation,
            }
        )
        saved = read_statistics_artifact(
            calibration, validate_tables=portfolio.validate_maker_calibration
        )
        with saved.table("crypto_maker_calibration.parquet") as frame:
            assert set(frame.fold_id) == {
                fold["fold_id"] for fold in chronology["folds"]
            }
            assert frame.status.eq("DEFINED").all()
            assert frame.sample_count.ge(2).all()
        calibrations[asset] = str(calibration)
    frozen = _bytes(learned)
    frame = read_dataset(inputs)
    expected_offsets = {
        None: [0, 100, 200, 300, 400],
        "depth": [0, 100],
        "agg_trades": [0, 100, 200, 300, 400],
    }[gap_kind]
    expected_times = {
        pd.Timestamp(chronology["test"]["start"]) + pd.Timedelta(offset, unit="ms")
        for offset in expected_offsets
    }

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "frozen prediction/backtest must not fit, score, select, or calibrate"
        )

    # Only forbidden learning calls are replaced; all sources, predictions,
    # execution, publication and persisted consumers remain the real path.
    monkeypatch.setattr(fitting, "_fit_model_result", forbidden)
    monkeypatch.setattr(production, "fit", forbidden)
    monkeypatch.setattr(trial_selection, "score", forbidden)
    monkeypatch.setattr(trial_selection, "select", forbidden)
    monkeypatch.setattr(portfolio, "calibrate_maker", forbidden)
    monkeypatch.setattr(portfolio, "_produce_maker_calibration", forbidden)
    for index, item in enumerate(models):
        item["predictions"] = str(
            predict(
                {
                    "model": item["model"],
                    "inputs": str(inputs),
                    "output": str(tmp_path / "predictions" / str(index)),
                    "prediction_window": chronology["test"],
                    "sample_role": "test",
                }
            )
        )
        rows, metadata = read_predictions(Path(item["predictions"]))
        assert not rows.empty
        assert set(rows.asset_id) == {"btc_usdt", "eth_usdt"}
        assert metadata["sample_role"] == "test"
        assert metadata["prediction_window"] == chronology["test"]
        for _asset, asset_rows in rows.groupby("asset_id"):
            assert set(asset_rows.timestamp) == expected_times
        # The 8.6-9 s holdout still predicts when its 500 ms target matures after
        # test end, or never matures because a real source gap interrupts it.
        target = metadata["model_facts"]["model"]["target_column"]
        available = target + "_available_at"
        matched = rows.merge(
            frame[["timestamp", "asset_id", available]], validate="one_to_one"
        )
        assert (
            matched[available].isna()
            | matched[available].gt(pd.Timestamp(chronology["test"]["end"]))
        ).all()
    prediction_bytes = _bytes(tmp_path / "predictions")
    assert _bytes(learned) == frozen
    results = {}
    for repeat in range(2):
        for asset, symbol in (("btc_usdt", "BTCUSDT"), ("eth_usdt", "ETHUSDT")):
            for style in ("taker", "maker_baseline", "maker_calibrated"):
                output = tmp_path / "ledgers" / f"{repeat}-{asset}-{style}"
                config = {
                    "output": str(output),
                    "inputs": str(inputs),
                    "panel": str(panel),
                    "source": str(tmp_path),
                    "archive_paths": archive_paths,
                    "asset_id": asset,
                    "symbol": symbol,
                    "chronology": chronology,
                    "prediction_window": chronology["test"],
                    "execution_mode": "continuous",
                    "capital": asdict(
                        PortfolioCapital(style, 1000.0, "USDT", "linear_perpetual")
                    ),
                }
                if style == "taker":
                    config.update(
                        execution=asdict(
                            replace(
                                _taker_config(latency_ms=10), markout_horizon_ms=500
                            )
                        ),
                        predictions=models[0]["predictions"],
                    )
                    backtest = portfolio.backtest_taker
                else:
                    calibrated = style == "maker_calibrated"
                    config.update(
                        execution=asdict(
                            replace(
                                _maker_config(calibrated=calibrated),
                                latency_ms=10,
                                markout_horizon_ms=500,
                            )
                        ),
                        models=[{"predictions": item["predictions"]} for item in models]
                        if calibrated
                        else [],
                        calibration=calibrations[asset] if calibrated else None,
                    )
                    backtest = portfolio.backtest_maker
                ledger = backtest(config)
                artifact = read_portfolio_ledger_artifact(ledger)
                batches = list(artifact.iter_chunks())
                summary = summarize_crypto_ledger(artifact, iter(batches))
                assert summary["execution"]["selection_status"] == "COMPLETE"
                assert summary["execution"]["market_event_count"] > 0
                assert summary["execution"]["no_eligible_decisions"] is False
                if style == "taker":
                    # These genuine forecasts do not clear the spread and fee.
                    assert summary["economics"]["fill_events"] == 0
                    assert summary["economics"]["nav"] == 1000.0
                elif gap_kind is not None:
                    assert summary["execution"]["account_path_status"] == "PARTIAL"
                    assert summary["economics"]["net_pnl"] is None
                    assert summary["execution"]["censored_active_order_count"] > 0
                    stop = summary["execution"]["account_stop_ns"]
                    assert all(
                        fill.timestamp_ns < stop
                        for batch in batches
                        for fill in batch.trades
                    )
                else:
                    assert summary["economics"]["fill_events"] == 2
                    assert summary["economics"]["gross_pnl"] == pytest.approx(0.8)
                    assert summary["economics"]["fees"] == pytest.approx(
                        -0.02002 if asset == "btc_usdt" else -0.04002
                    )
                    assert summary["execution"]["maker_prediction_used"] is (
                        style == "maker_calibrated"
                    )
                assert artifact.metadata["sample_role"] == "prediction"
                assert artifact.metadata["window"] == chronology["test"]
                assert batches and any(batch.valuations for batch in batches)
                if repeat == 0:
                    results[asset, style] = (batches, summary, _bytes(output))
                else:
                    old_batches, old_summary, original_bytes = results[asset, style]
                    assert batches == old_batches
                    assert summary == old_summary
                    assert (
                        _bytes(output.with_name(f"0-{asset}-{style}")) == original_bytes
                    )
        assert _bytes(learned) == frozen
        assert _bytes(tmp_path / "predictions") == prediction_bytes

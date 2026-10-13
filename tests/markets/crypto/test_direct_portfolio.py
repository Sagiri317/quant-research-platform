"""Explicit saved models, predictions, calibration and backtests retain science."""

from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.portfolios import (
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score, select
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets.writing import write_dataset
from tests.markets.crypto.test_hft_execution import _maker_config, _state, _trade


def _window(start, end):
    return {
        "start": pd.Timestamp(start, unit="ms").isoformat(),
        "end": pd.Timestamp(end, unit="ms").isoformat(),
    }


@pytest.fixture
def explicit_models(tmp_path):
    stamps = [*range(0, 1100, 100), 2100, 2200, 8600, 8800, 9000]
    times = pd.to_datetime(stamps, unit="ms")
    price, proxy = "future_mid_return_500ms", "maker_bid_sell_pressure_proxy_500ms"
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": "btc_usdt",
            "x": np.arange(len(times), dtype=float),
            price: 0.0,
            proxy: 1.0,
            "eligible": True,
            "book_segment_id": "s1",
            "trade_observation_id": "trade-connection-1",
            "segment_id": "segment",
        }
    )
    for target in (price, proxy):
        frame[target + "_available_at"] = times + pd.Timedelta(milliseconds=500)
    roles = {name: "context" for name in frame.columns}
    roles.update(timestamp="timestamp", asset_id="key", x="feature", eligible="mask")
    roles.update({price: "label", proxy: "label"})
    inputs = tmp_path / "inputs"
    write_dataset(
        inputs,
        frame,
        spec=DataSpec(
            data_id="crypto_fixture",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
        ),
        metadata={
            "source_evidence": {
                "market_metadata": {
                    "BTCUSDT": {"price_unit": "0.1", "snapshot_limit": 10}
                }
            },
            "producer_diagnostics": {
                "reconstruction": {
                    "BTCUSDT": {"coverage_segments": {"s1": [0, 10_000_000_000]}}
                }
            },
        },
    )
    chronology = {
        "folds": [
            {
                "fold_id": "first",
                "train": _window(0, 2000),
                "validation": _window(2100, 3500),
            }
        ],
        "test": _window(8600, 9000),
    }
    validation_models, final_models = [], []
    for model_id, target in (("price", price), ("proxy", proxy)):
        config = {
            "market_id": "crypto",
            "inputs": str(inputs),
            "output": str(tmp_path / model_id),
            "model": {
                "model_id": model_id,
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["x"],
                "target_column": target,
                "random_seed": 17,
            },
            "fit_windows": [chronology["folds"][0]["train"]],
            "fit_cutoff": _window(0, 2000)["end"],
            "label_unit": "decimal_return" if model_id == "price" else "quantity_ratio",
        }
        model = fit(config)
        pred_config = {
            "model": str(model),
            "inputs": str(inputs),
            "output": str(tmp_path / (model_id + "-validation")),
            "prediction_window": chronology["folds"][0]["validation"],
            "sample_role": "validation",
        }
        predictions = predict(pred_config)
        validation_models.append({"model": str(model), "predictions": str(predictions)})
        scores = score(
            {
                "predictions": [[str(predictions)]],
                "labels": str(inputs),
                "candidate_ids": [model_id],
                "validation_windows": [chronology["folds"][0]["validation"]],
                "scoring_window": chronology["folds"][0]["validation"],
                "cutoff": chronology["folds"][0]["validation"]["end"],
                "test_window": chronology["test"],
                "selection_group": "global",
                "validation_axis": "single_asset_time_series",
                "output": str(tmp_path / (model_id + "-scores.json")),
                "label_unit": config["label_unit"],
            }
        )
        selection = select(
            {
                "scores": str(scores),
                "output": str(tmp_path / (model_id + "-selection.json")),
                "selection_metric": "validation_mse",
                "selection_direction": "minimize",
                "tie_break_order": [model_id],
                "refit_windows": [
                    chronology["folds"][0]["train"],
                    chronology["folds"][0]["validation"],
                ],
                "refit_cutoff": chronology["folds"][0]["validation"]["end"],
            }
        )
        final = fit(
            {
                **config,
                "output": str(tmp_path / (model_id + "-final")),
                "selection": str(selection),
                "test_window": chronology["test"],
                "fit_cutoff": chronology["folds"][0]["validation"]["end"],
                "fit_windows": [
                    chronology["folds"][0]["train"],
                    chronology["folds"][0]["validation"],
                ],
            }
        )
        predictions = predict(
            {
                **pred_config,
                "model": str(final),
                "output": str(tmp_path / (model_id + "-test")),
                "prediction_window": chronology["test"],
                "sample_role": "test",
            }
        )
        final_models.append({"model": str(final), "predictions": str(predictions)})
    return inputs, chronology, validation_models, final_models


def _market(monkeypatch):
    class Coverage(dict):
        def covering(self, *_):
            return tuple(self)

        def intervals(self):
            return iter(self.values())

    source = SimpleNamespace(
        market_metadata=lambda *_: {"price_unit": "0.1", "snapshot_limit": 10},
        observation_coverage=lambda **_: Coverage(
            {"trade-connection-1": (0, 10_000_000_000)}
        ),
        funding_roots=lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        portfolio, "open_market_source", lambda *_args, **_kwargs: nullcontext(source)
    )
    monkeypatch.setattr(
        portfolio,
        "iter_reconstructed_depth",
        lambda *_args, **_kwargs: iter(
            _state(t) for t in (2100, 2150, 2650, 2700, 8600, 8800, 9000)
        ),
    )
    monkeypatch.setattr(
        portfolio,
        "iter_agg_trades",
        lambda *_args, **_kwargs: iter(
            _trade(t, price="99.5", quantity="3", maker_buyer=True)
            for t in (2180, 2400, 8700, 8900)
        ),
    )
    funding = SimpleNamespace(
        coverage=((0, 10_000_000_000),),
        events=(),
        forecasts=(FundingForecast(0, 11_000_000_000, ()),),
        settlement_qualification=(),
    )
    monkeypatch.setattr(
        portfolio,
        "read_funding_observations",
        lambda *_args, **_kwargs: nullcontext(funding),
    )


@pytest.mark.parametrize("minimum", (2, 100000))
def test_explicit_maker_calibration_and_frozen_backtest(
    tmp_path, monkeypatch, explicit_models, minimum
):
    inputs, chronology, validation, final = explicit_models
    _market(monkeypatch)
    config = {
        "output": str(tmp_path / "calibration"),
        "panel": str(inputs),
        "source": "explicit-source",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "chronology": chronology,
        "reference": {
            "order_quantity": 1.0,
            "latency_ms": 10,
            "min_calibration_samples": minimum,
        },
        "models": [
            {
                "fold_id": "first",
                "inputs": str(inputs),
                "price_model": validation[0]["model"],
                "price_predictions": validation[0]["predictions"],
                "proxy_model": validation[1]["model"],
                "proxy_predictions": validation[1]["predictions"],
            }
        ],
    }
    calibration = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        calibration, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.tolist() == [2, 2]
    # Consumption uses the already learned models, with independent exact-key checks.
    import quant_research_platform.research.models.fitting as fitting

    monkeypatch.setattr(
        fitting, "_fit_model_result", lambda *_a, **_k: pytest.fail("backtest fitted")
    )
    backtest = {
        "output": str(tmp_path / "ledger"),
        "inputs": str(inputs),
        "panel": str(inputs),
        "source": "explicit-source",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "chronology": chronology,
        "prediction_window": chronology["test"],
        "execution_mode": "continuous",
        "capital": asdict(
            PortfolioCapital("maker", 1000.0, "USDT", "linear_perpetual")
        ),
        "execution": asdict(_maker_config(calibrated=True)),
        "calibration": str(calibration),
        "models": final,
    }
    if minimum > 2:
        from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

        with pytest.raises(PortfolioTargetBlocked, match="insufficient_samples"):
            portfolio.backtest_maker(backtest)
        assert not (tmp_path / "ledger").exists()
        # An insufficient calibration qualifies only its dependent maker, not the
        # independent explicit baseline operation or its already learned inputs.
        backtest.update(calibration=None, models=[], execution=asdict(_maker_config()))
    ledger = portfolio.backtest_maker(backtest)
    saved = read_portfolio_ledger_artifact(ledger)
    summary = summarize_crypto_ledger(saved, saved.iter_chunks())
    assert summary["execution"]["selection_status"] == "COMPLETE"
    assert saved.metadata["execution_style"] == "maker"


@pytest.mark.parametrize("mutation", ("missing", "extra", "rank_units"))
def test_prediction_consumer_preserves_full_expected_domain(
    tmp_path, explicit_models, mutation
):
    inputs, chronology, _validation, final = explicit_models
    window = TimeWindow(**chronology["test"])
    metadata, facts, predictions = portfolio._model_evidence(
        final[0]["model"],
        final[0]["predictions"],
        window=window,
        sample_role="test",
        chronology=chronology,
        fold_id="first",
    )
    rows = list(
        portfolio._prediction_rows(
            inputs=inputs,
            evidence=(metadata, facts, predictions),
            window=window,
            fold_id="first",
            asset_id="btc_usdt",
        )
    )
    assert [row["timestamp_ms"] for row in rows] == [8600, 8800, 9000]
    if mutation == "missing":
        predictions = predictions.iloc[1:]
    elif mutation == "extra":
        extra = predictions.iloc[[0]].copy()
        extra.asset_id = "eth_usdt"
        predictions = pd.concat([predictions, extra], ignore_index=True)
    else:
        metadata["model_algorithm"] = "equal_weight_rank_ensemble"
    with pytest.raises(
        ValueError,
        match="missing_required_prediction_keys|misaligned_persisted_prediction_keys|score_units",
    ):
        list(
            portfolio._prediction_rows(
                inputs=inputs,
                evidence=(metadata, facts, predictions),
                window=window,
                fold_id="first",
                asset_id="btc_usdt",
            )
        )


@pytest.mark.parametrize(
    "operation",
    (portfolio.backtest_taker, portfolio.backtest_maker, portfolio.calibrate_maker),
)
def test_direct_operations_reject_unused_fields(operation):
    with pytest.raises(ValueError, match="config_fields"):
        operation({"unconsumed": True})


@pytest.mark.parametrize("damage", ("gap", "after_validation"))
def test_backtest_rejects_refit_outside_declared_learning_windows(
    explicit_models, monkeypatch, damage
):
    _inputs, chronology, _validation, final = explicit_models
    load = portfolio.load_model

    def changed(path):
        model, facts = load(path)
        if damage == "gap":
            facts["fit_windows"] = [_window(0, 3500)]
        else:
            facts["fit_cutoff"] = _window(0, 4000)["end"]
        return model, facts

    monkeypatch.setattr(portfolio, "load_model", changed)
    with pytest.raises(ValueError, match="freeze_invalid|outside_chronology"):
        portfolio._model_evidence(
            final[0]["model"],
            final[0]["predictions"],
            window=TimeWindow(**chronology["test"]),
            sample_role="test",
            chronology=chronology,
            fold_id="first",
        )


@pytest.mark.parametrize(
    "field,value", (("price_unit", "0.01"), ("snapshot_limit", 100))
)
def test_panel_and_replay_must_share_actual_coordinate_spec(field, value):
    recorded = {"price_unit": "0.1", "snapshot_limit": 10}
    source = SimpleNamespace(market_metadata=lambda *_: {**recorded, field: value})
    panel = {"source_evidence": {"market_metadata": {"BTCUSDT": recorded}}}
    with pytest.raises(ValueError, match="source_coordinates_mismatch"):
        portfolio._source_coordinates(source, panel, "BTCUSDT")


@pytest.mark.parametrize("value", (None, "NaT"))
def test_episode_publication_rejects_nat_metadata_before_production(tmp_path, value):
    from quant_research_platform.artifacts.crypto_hft import (
        write_crypto_episode_artifact,
    )

    with pytest.raises(ValueError, match="window_invalid"):
        write_crypto_episode_artifact(
            tmp_path / "invalid",
            capital=PortfolioCapital("episode", 1000.0, "USDT", "linear_perpetual"),
            metadata={"window": {"start": value, "end": "2024-01-01T00:00:00"}},
            produce=lambda _: pytest.fail("production started with NaT"),
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("mode", ("continuous", "independent_segments"))
def test_explicit_taker_uses_saved_predictions_without_training(
    tmp_path, monkeypatch, explicit_models, mode
):
    import quant_research_platform.research.models.fitting as fitting
    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
    )

    inputs, chronology, _validation, final = explicit_models
    _market(monkeypatch)
    monkeypatch.setattr(
        fitting, "_fit_model_result", lambda *_a, **_k: pytest.fail("backtest fitted")
    )
    output = portfolio.backtest_taker(
        {
            "output": str(tmp_path / "taker"),
            "inputs": str(inputs),
            "panel": str(inputs),
            "source": "explicit-source",
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
            "chronology": chronology,
            "prediction_window": chronology["test"],
            "execution_mode": mode,
            "capital": asdict(
                PortfolioCapital("taker", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(portfolio.TakerConfig(1.0, 2.0, 0, 0.0, 100)),
            "model": final[0]["model"],
            "predictions": final[0]["predictions"],
        }
    )
    values = []

    def consume(artifact, batches):
        values.append(summarize_crypto_ledger(artifact, batches)["economics"])

    if mode == "continuous":
        read_portfolio_ledger_artifact(output, consume_batches=consume)
    else:
        read_crypto_episode_artifact(
            output,
            consume_episode=lambda _index, artifact, batches: consume(
                artifact, batches
            ),
        )
    assert len(values) == 1
    assert values[0]["nav"] == 1000.0
    assert values[0]["fill_events"] == 0

"""Explicit saved models, predictions, calibration and backtests retain science."""

from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
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


def _inputs(tmp_path, *, ineligible_at=None):
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
    if ineligible_at is not None:
        frame.loc[
            frame.timestamp.eq(pd.Timestamp(ineligible_at, unit="ms")), "eligible"
        ] = False
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
    return inputs


@pytest.fixture
def explicit_models(tmp_path):
    inputs = _inputs(tmp_path)
    price, proxy = "future_mid_return_500ms", "maker_bid_sell_pressure_proxy_500ms"
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

    def books(*_args, stats=None, **_kwargs):
        if stats is not None:
            stats.coverage_segments["s1"] = (0, 10_000_000_000)
        yield from (_state(t) for t in (2100, 2150, 2650, 2700, 8600, 8800, 9000))

    monkeypatch.setattr(portfolio, "iter_reconstructed_depth", books)
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
@pytest.mark.parametrize("declare_chronology", (True, False))
def test_explicit_maker_calibration_and_frozen_backtest(
    tmp_path, monkeypatch, explicit_models, minimum, declare_chronology
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
                "price_predictions": validation[0]["predictions"],
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
        "models": [{"predictions": item["predictions"]} for item in final],
    }
    if not declare_chronology:
        backtest.pop("chronology")
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
    metadata, facts, predictions = portfolio._prediction_evidence(
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
    read = portfolio.read_predictions

    def changed(path, **kwargs):
        predictions, metadata = read(path, **kwargs)
        facts = metadata["model_facts"]
        if damage == "gap":
            facts["fit_windows"] = [_window(0, 3500)]
        else:
            facts["fit_cutoff"] = _window(0, 4000)["end"]
        return predictions, metadata

    monkeypatch.setattr(portfolio, "read_predictions", changed)
    with pytest.raises(ValueError, match="freeze_invalid|outside_chronology"):
        portfolio._prediction_evidence(
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
@pytest.mark.parametrize("declare_chronology", (True, False))
def test_explicit_taker_uses_saved_predictions_without_training(
    tmp_path, monkeypatch, explicit_models, mode, declare_chronology
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
            **({"chronology": chronology} if declare_chronology else {}),
            "prediction_window": chronology["test"],
            "execution_mode": mode,
            "capital": asdict(
                PortfolioCapital("taker", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(portfolio.TakerConfig(1.0, 2.0, 0, 0.0, 100)),
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


def test_prediction_facts_are_self_contained_after_model_files_move(explicit_models):
    inputs, chronology, _validation, final = explicit_models
    model_path = Path(final[0]["model"])
    model_path.rename(model_path.with_name(model_path.name + "-detached"))
    assert not model_path.exists()
    window = TimeWindow(**chronology["test"])
    evidence = portfolio._prediction_evidence(
        final[0]["predictions"],
        window=window,
    )
    rows = list(
        portfolio._prediction_rows(
            inputs={
                "path": str(inputs),
                "row_filters": [["asset_id", "in", ["btc_usdt"]]],
            },
            evidence=evidence,
            window=window,
            fold_id="first",
            asset_id="btc_usdt",
        )
    )
    assert [row["timestamp_ms"] for row in rows] == [8600, 8800, 9000]


def _independent_maker_case(tmp_path, monkeypatch, *, unfiltered=False):
    inputs = _inputs(tmp_path, ineligible_at=2200 if unfiltered else None)
    _market(monkeypatch)
    observed_window = _window(2100, 2200)
    execution_window = _window(8600, 9000)
    source = {"path": str(inputs), "row_filters": [["asset_id", "==", "btc_usdt"]]}
    calibration_models = {
        "fold_id": "observed",
        "inputs": source,
        "window": observed_window,
        "cutoff": _window(0, 2700)["end"],
    }
    execution_models = []
    for role, target, unit in (
        ("price", "future_mid_return_500ms", "decimal_return"),
        ("proxy", "maker_bid_sell_pressure_proxy_500ms", "quantity_ratio"),
    ):
        model = fit(
            {
                "market_id": "crypto",
                "inputs": source,
                "output": str(tmp_path / (role + "-model")),
                "model": {
                    "model_id": role,
                    "model_algorithm": "linear_regression",
                    "input_feature_columns": ["x"],
                    "target_column": target,
                    "random_seed": 17,
                    "sample_filter_mode": "unfiltered" if unfiltered else "eligible",
                },
                "fit_windows": [_window(0, 1000)],
                "fit_cutoff": _window(0, 2000)["end"],
                "label_unit": unit,
            }
        )
        for purpose, window in (
            ("calibration", observed_window),
            ("execution", execution_window),
        ):
            predictions = predict(
                {
                    "model": str(model),
                    "inputs": source,
                    "output": str(tmp_path / (role + "-" + purpose)),
                    "prediction_window": window,
                }
            )
            if purpose == "calibration":
                calibration_models[role + "_predictions"] = str(predictions)
            else:
                execution_models.append({"predictions": str(predictions)})
        # The downstream operations must consume self-contained predictions only.
        model.rename(model.with_name("unused-" + model.name))
    common = {
        "panel": str(inputs),
        "source": "explicit-source",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
    }
    calibration = {
        **common,
        "output": str(tmp_path / "calibration"),
        "reference": {
            "order_quantity": 1.0,
            "latency_ms": 10,
            "min_calibration_samples": 1,
        },
        "models": [calibration_models],
    }
    backtest = {
        **common,
        "output": str(tmp_path / "ledger"),
        "inputs": source,
        "prediction_window": execution_window,
        "execution_mode": "continuous",
        "capital": asdict(
            PortfolioCapital("maker", 1000.0, "USDT", "linear_perpetual")
        ),
        "execution": asdict(_maker_config(calibrated=True)),
        "calibration": calibration["output"],
        "models": execution_models,
    }
    return calibration, backtest


@pytest.fixture
def independent_maker(tmp_path, monkeypatch):
    return _independent_maker_case(tmp_path, monkeypatch)


@pytest.mark.parametrize("maker_buyer", (False, True))
@pytest.mark.parametrize("relevant", (False, True))
def test_native_calibration_preserves_unknown_trades_and_reference_relevance(
    monkeypatch, independent_maker, maker_buyer, relevant
):
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
    from quant_research_platform.markets.implementations.crypto.execution import (
        MakerCalibration,
        _maker_new_order,
    )
    from tests.helpers.binance_hft import _quote

    config, _backtest = independent_maker
    price = ("99.5" if maker_buyer else "100.5") if relevant else "100.0"
    trade = replace(
        _trade(2300, price=price, quantity="2", maker_buyer=maker_buyer),
        visible_quantity=None,
    )

    def books(*_args, stats=None, **_kwargs):
        if stats is not None:
            stats.coverage_segments["s1"] = (0, 10_000_000_000)
        yield _state(2100)
        # The second reference pair cannot be touched by the first pair's price.
        yield from (
            _state(stamp, bid_index=990, ask_index=1010) for stamp in (2150, 2650, 2700)
        )

    monkeypatch.setattr(portfolio, "iter_reconstructed_depth", books)
    monkeypatch.setattr(
        portfolio, "iter_agg_trades", lambda *_args, **_kwargs: iter((trade,))
    )
    output = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    count = 1 if relevant else 2
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.tolist() == [count, count]
        assert frame.excluded_decisions.tolist() == [int(relevant)] * 2
    with artifact.table("crypto_maker_events.parquet") as frame:
        trades = frame.loc[frame.event_kind.eq("trade")]
        assert len(trades) == 1 and trades.quantity.isna().all()
        event_frame = frame.copy()
    events = list(portfolio._calibration_events(artifact, "observed"))
    assert (
        next(event for event in events if event.kind == "trade").visible_quantity
        is None
    )
    with artifact.table("crypto_maker_samples.parquet") as frame:
        assert frame.filled_quantity.eq(0).all()
        assert frame.decision_at.min() == pd.Timestamp(
            2200 if relevant else 2100, unit="ms"
        )
    for kind, quantity in (("book", np.nan), ("trade", np.inf), ("trade", -1.0)):
        damaged = event_frame.copy()
        damaged.loc[damaged.event_kind.eq(kind), "quantity"] = quantity
        with monkeypatch.context() as patch:
            patch.setattr(
                portfolio,
                "_calibration_rows",
                lambda *_args, _frame=damaged, **_kwargs: iter(
                    _frame.itertuples(index=False)
                ),
            )
            with pytest.raises(ValueError, match="crypto_calibration_event_invalid"):
                list(
                    portfolio._calibration_headers(artifact, "observed", 2_700_000_000)
                )
    if relevant:
        return

    model = MakerCalibration(
        partial(portfolio._calibration_samples, artifact, "observed"),
        partial(portfolio._calibration_events, artifact, "observed"),
        price_unit="0.1",
    )
    side = "buy" if maker_buyer else "sell"
    sample = next(model.samples(side))
    state = _state(8600)
    order = _maker_new_order(1, side, 1.0, state, _quote(8600, book_segment_id="s1"), 0)

    def response(active_order, replace_quantity):
        return list(
            model._response(
                sample,
                state=state,
                decision_ns=8_600_000_000,
                new_order=order,
                active_order=active_order,
                commands=(),
                lifetime_ns=500_000_000,
                replace_quantity=replace_quantity,
                history_scale=1,
                events=model.events(sample["event_start"], sample["event_end"]),
            )
        )

    assert response(None, 1.0) == []
    assert response(order, None) == []
    dependent_keep = replace(order, price_index=1000)
    with pytest.raises(PortfolioTargetBlocked, match="visible_quantity_unknown"):
        response(dependent_keep, None)


@pytest.mark.parametrize("cutoff,count", ((2600, 1), (2700, 2)))
def test_independent_mature_calibration_reaches_maker_and_statistics(
    tmp_path, independent_maker, cutoff, count
):
    from quant_research_platform.research.portfolios.diagnostics import (
        crypto_statistics,
    )

    config, backtest = independent_maker
    config["models"][0]["cutoff"] = _window(0, cutoff)["end"]
    output = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    assert "chronology" not in artifact.metadata
    assert artifact.metadata["window"] == _window(2100, 2200)
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.tolist() == [count, count]
    with artifact.table("crypto_maker_samples.parquet") as frame:
        assert frame.decision_at.max() <= pd.Timestamp(_window(0, 2200)["end"])
        assert frame.label_available_at.max() == pd.Timestamp(_window(0, cutoff)["end"])
    ledger = portfolio.backtest_maker(backtest)
    saved = read_portfolio_ledger_artifact(ledger)
    assert saved.metadata["sample_role"] == "prediction"
    assert (
        summarize_crypto_ledger(saved, saved.iter_chunks())["execution"][
            "selection_status"
        ]
        == "COMPLETE"
    )
    statistics = crypto_statistics(
        {
            "ledger": str(ledger),
            "output": str(tmp_path / "statistics"),
            "analyses": ["performance", "execution"],
        }
    )
    assert read_statistics_artifact(statistics).metadata["sample_role"] == "prediction"


@pytest.mark.parametrize(
    "damage", ("window", "cutoff", "late_fit", "units", "asset", "label")
)
def test_independent_calibration_rejects_missing_or_wrong_science(
    monkeypatch, independent_maker, damage
):
    config, _backtest = independent_maker
    item = config["models"][0]
    read = portfolio.read_predictions
    if damage in {"window", "cutoff"}:
        item.pop(damage)
    elif damage == "label":
        session = portfolio.open_dataset_session(Path(item["inputs"]["path"]))
        item["inputs"]["columns"] = [
            name
            for name in session.logical_columns
            if name != "future_mid_return_500ms"
        ]
    else:

        def changed(path, **kwargs):
            rows, metadata = read(path, **kwargs)
            if damage == "late_fit":
                metadata["model_facts"]["fit_cutoff"] = item["window"]["start"]
            elif damage == "units":
                metadata["model_facts"]["label_unit"] = "percentage_return"
            else:
                metadata["asset_ids"] = ["eth_usdt"]
            return rows, metadata

        monkeypatch.setattr(portfolio, "read_predictions", changed)
    with pytest.raises(
        ValueError,
        match="window_cutoff_required|after_decision|label_unit_invalid|outside_scope|column",
    ):
        portfolio.calibrate_maker(config)
    assert not Path(config["output"]).exists()


def test_independent_calibration_cutoff_must_precede_execution(independent_maker):
    config, backtest = independent_maker
    config["models"][0]["cutoff"] = backtest["prediction_window"]["start"]
    portfolio.calibrate_maker(config)
    with pytest.raises(ValueError, match="consumer_relationship_invalid"):
        portfolio.backtest_maker(backtest)
    assert not Path(backtest["output"]).exists()


def test_calibration_samples_stay_inside_actual_window(independent_maker):
    config, _backtest = independent_maker
    output = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    metadata = artifact.metadata
    metadata["window"] = _window(2100, 2100)
    metadata["models"][0]["window"] = metadata["window"]
    with pytest.raises(ValueError, match="outcome_not_mature"):
        portfolio.validate_maker_calibration(artifact, metadata)


@pytest.mark.parametrize(
    "damage", ("outside_validation", "late_cutoff", "missing_cutoff", "earlier_cutoff")
)
def test_explicit_chronology_only_bounds_actual_calibration_window(damage):
    chronology = {
        "folds": [
            {
                "fold_id": "observed",
                "train": _window(0, 2000),
                "validation": _window(2100, 3500),
            }
        ],
        "test": _window(8600, 9000),
    }
    entry = {
        "fold_id": "observed",
        "window": _window(2100, 2200),
        "cutoff": _window(0, 2700)["end"],
    }
    if damage == "outside_validation":
        entry["window"] = _window(2000, 2200)
    elif damage == "late_cutoff":
        entry["cutoff"] = _window(0, 3600)["end"]
    elif damage == "earlier_cutoff":
        entry["cutoff"] = _window(0, 2100)["end"]
    else:
        entry.pop("cutoff")
    config = {"chronology": chronology, "models": [entry]}
    if damage == "missing_cutoff":
        assert portfolio._calibration_windows(config)["observed"] == (
            TimeWindow(**entry["window"]),
            pd.Timestamp(chronology["folds"][0]["validation"]["end"]),
        )
    else:
        with pytest.raises(ValueError, match="window_invalid"):
            portfolio._calibration_windows(config)


def test_unknown_calibration_maturity_is_excluded_and_streams_close(
    monkeypatch, independent_maker
):
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    config, backtest = independent_maker
    read = portfolio.iter_dataset_chunks
    closed = []

    def unknown_maturity(session):
        chunks = read(session)
        try:
            with portfolio.closing_iterator(chunks):
                for frame in chunks:
                    yield frame.assign(
                        **{
                            column: pd.NaT
                            for column in frame
                            if column.endswith("_available_at")
                        }
                    )
        finally:
            closed.append(True)

    monkeypatch.setattr(portfolio, "iter_dataset_chunks", unknown_maturity)
    output = portfolio.calibrate_maker(config)
    assert len(closed) == 2
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.tolist() == [0, 0]
        assert frame.status.eq("UNDEFINED").all()
    with pytest.raises(PortfolioTargetBlocked, match="insufficient_samples"):
        portfolio.backtest_maker(backtest)
    assert not Path(backtest["output"]).exists()


@pytest.mark.parametrize("damage", ("missing_price", "missing_proxy", "wrong_target"))
def test_calibrated_maker_requires_both_prediction_roles(independent_maker, damage):
    config, _ = independent_maker
    models = config["models"][0]
    if damage == "wrong_target":
        models["price_predictions"] = models["proxy_predictions"]
        error = "crypto_maker_prediction_target_invalid"
    else:
        del models[damage.removeprefix("missing_") + "_predictions"]
        error = "config_fields"
    with pytest.raises(ValueError, match=error):
        portfolio.calibrate_maker(config)


def test_unfiltered_predictions_keep_full_coverage_but_calibration_uses_eligible(
    tmp_path, monkeypatch
):
    config, _ = _independent_maker_case(tmp_path, monkeypatch, unfiltered=True)
    output = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.tolist() == [1, 1]


def test_calibration_closes_support_columns_and_ignores_descriptive_prediction_role(
    monkeypatch, independent_maker
):
    config, _ = independent_maker
    config["models"][0]["inputs"]["columns"] = [
        "x",
        "future_mid_return_500ms",
        "maker_bid_sell_pressure_proxy_500ms",
    ]
    original = portfolio.read_predictions

    def named(path, **kwargs):
        rows, metadata = original(path, **kwargs)
        metadata["sample_role"] = "test"
        return rows, metadata

    monkeypatch.setattr(portfolio, "read_predictions", named)
    output = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.tolist() == [2, 2]

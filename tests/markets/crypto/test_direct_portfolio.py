"""Explicit saved models, predictions, calibration and backtests retain science."""

from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from weakref import ref

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


def _inputs(tmp_path, *, ineligible_at=None, distinct_domains=False):
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
    if distinct_domains:
        frame["y"] = frame["x"]
        frame.loc[frame.timestamp.eq(pd.Timestamp(8600, unit="ms")), "x"] = np.nan
        frame.loc[frame.timestamp.eq(pd.Timestamp(9000, unit="ms")), "y"] = np.nan
    if ineligible_at is not None:
        frame.loc[
            frame.timestamp.eq(pd.Timestamp(ineligible_at, unit="ms")), "eligible"
        ] = False
    for target in (price, proxy):
        frame[target + "_available_at"] = times + pd.Timedelta(milliseconds=500)
    roles = {name: "context" for name in frame.columns}
    roles.update(timestamp="timestamp", asset_id="key", x="feature", eligible="mask")
    if distinct_domains:
        roles["y"] = "feature"
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


def _explicit_models(tmp_path, *, roles):
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
    targets = {"price": price, "proxy": proxy}
    for model_id in roles:
        target = targets[model_id]
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


@pytest.fixture
def explicit_models(tmp_path):
    return _explicit_models(tmp_path, roles=("price", "proxy"))


@pytest.fixture
def explicit_price_models(tmp_path):
    return _explicit_models(tmp_path, roles=("price",))


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
        "prediction_window": chronology["test"],
        "execution_mode": "continuous",
        "capital": asdict(
            PortfolioCapital("maker", 1000.0, "USDT", "linear_perpetual")
        ),
        "execution": asdict(_maker_config(calibrated=True)),
        "calibration": str(calibration),
        "models": [{"predictions": item["predictions"]} for item in final],
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


@pytest.mark.parametrize("mutation", ("missing", "extra", "rank_units", "duplicate"))
def test_prediction_consumer_preserves_full_expected_domain(
    tmp_path, monkeypatch, explicit_price_models, mutation
):
    inputs, chronology, _validation, final = explicit_price_models
    window = TimeWindow(**chronology["test"])
    metadata, facts, predictions = portfolio._prediction_evidence(
        final[0]["predictions"],
        window=window,
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
    if mutation in {"missing", "duplicate"}:
        chunks = portfolio.iter_dataset_chunks

        def without_context(session):
            with portfolio.closing_iterator(chunks(session)) as source:
                for frame in source:
                    selected = frame.timestamp.eq(pd.Timestamp(8600, unit="ms"))
                    if mutation == "missing":
                        yield frame.loc[~selected]
                    else:
                        # A wider source PK can repeat the consumed asset/time axis
                        # within one physical batch. Both matches must be rejected.
                        yield pd.concat([frame, frame.loc[selected]], ignore_index=True)

        monkeypatch.setattr(portfolio, "iter_dataset_chunks", without_context)
    elif mutation == "extra":
        extra = predictions.iloc[[0]].copy()
        extra.timestamp = pd.Timestamp(8700, unit="ms")
        predictions = pd.concat([predictions, extra], ignore_index=True)
    else:
        metadata["model_algorithm"] = "equal_weight_rank_ensemble"
    with pytest.raises(
        ValueError,
        match=(
            "duplicate_prediction_input_keys"
            if mutation == "duplicate"
            else (
                "missing_required_prediction_keys|"
                "misaligned_persisted_prediction_keys|score_units"
            )
        ),
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


@pytest.mark.parametrize("scope", ("continuous_history", "later_cutoff"))
def test_backtest_uses_actual_fitting_facts_without_model_registration(
    tmp_path, explicit_price_models, monkeypatch, scope
):
    inputs, chronology, _validation, final = explicit_price_models
    _, previous = portfolio.read_predictions(final[0]["predictions"])
    facts = previous["model_facts"]
    windows = (
        [_window(0, 3500)] if scope == "continuous_history" else facts["fit_windows"]
    )
    cutoff = _window(0, 4000)["end"]
    model = fit(
        {
            "market_id": "crypto",
            "inputs": str(inputs),
            "model": facts["model"],
            "label_unit": facts["label_unit"],
            "fit_windows": windows,
            "fit_cutoff": cutoff,
            "output": str(tmp_path / "later_fit"),
        }
    )
    predictions = predict(
        {
            "model": str(model),
            "inputs": str(inputs),
            "prediction_window": chronology["test"],
            "output": str(tmp_path / "later_prediction"),
        }
    )
    _market(monkeypatch)
    ledger = portfolio.backtest_taker(
        {
            "output": str(tmp_path / "later_taker"),
            "inputs": str(inputs),
            "panel": str(inputs),
            "source": "explicit-source",
            "asset_id": "btc_usdt",
            "symbol": "BTCUSDT",
            "prediction_window": chronology["test"],
            "execution_mode": "continuous",
            "capital": asdict(
                PortfolioCapital("taker", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(portfolio.TakerConfig(1.0, 2.0, 0, 0.0, 100)),
            "predictions": str(predictions),
        }
    )
    artifact = read_portfolio_ledger_artifact(ledger)
    summary = summarize_crypto_ledger(artifact, artifact.iter_chunks())
    assert summary["economics"]["nav"] == pytest.approx(1000.0)


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
@pytest.mark.parametrize("window_kind", ("test", "dev"))
def test_explicit_taker_uses_saved_predictions_without_training(
    tmp_path, monkeypatch, explicit_price_models, mode, window_kind
):
    import quant_research_platform.research.models.fitting as fitting
    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
    )

    inputs, chronology, validation, final = explicit_price_models
    window = (
        chronology["test"]
        if window_kind == "test"
        else chronology["folds"][0]["validation"]
    )
    models = final if window_kind == "test" else validation
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
            "prediction_window": window,
            "execution_mode": mode,
            "capital": asdict(
                PortfolioCapital("taker", 1000.0, "USDT", "linear_perpetual")
            ),
            "execution": asdict(portfolio.TakerConfig(1.0, 2.0, 0, 0.0, 100)),
            "predictions": models[0]["predictions"],
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


def test_prediction_facts_are_self_contained_after_model_files_move(
    explicit_price_models,
):
    inputs, chronology, _validation, final = explicit_price_models
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


def _independent_maker_case(
    tmp_path, monkeypatch, *, unfiltered=False, distinct_domains=False
):
    inputs = _inputs(
        tmp_path,
        ineligible_at=2200 if unfiltered else None,
        distinct_domains=distinct_domains,
    )
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
                    "input_feature_columns": [
                        "y" if distinct_domains and role == "proxy" else "x"
                    ],
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
                list(portfolio._calibration_events(artifact, "observed", 2_700_000_000))
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
def test_chronology_supplies_defaults_without_binding_explicit_calibration(damage):
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
    elif damage == "earlier_cutoff":
        with pytest.raises(ValueError, match="window_invalid"):
            portfolio._calibration_windows(config)
    else:
        assert portfolio._calibration_windows(config)["observed"] == (
            TimeWindow(**entry["window"]),
            pd.Timestamp(entry["cutoff"]),
        )


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


def test_calibration_loads_and_releases_one_fold_at_a_time(
    monkeypatch, independent_maker
):
    config, _ = independent_maker
    baseline = read_statistics_artifact(
        portfolio.calibrate_maker(config),
        validate_tables=portfolio.validate_maker_calibration,
    )
    config["output"] += "-multiple-folds"
    config["models"] = [
        {**config["models"][0], "fold_id": fold}
        for fold in ("first", "second", "third")
    ]
    read = portfolio.read_predictions
    replay = portfolio.replay_maker_calibration
    frames = []
    completed = 0

    def tracked_read(path, **kwargs):
        # No future fold is decoded before the current fold has finished, and
        # completed folds no longer own even their original prediction frames.
        assert len(frames) // 2 == completed
        assert all(frame() is None for frame in frames[: completed * 2])
        rows, metadata = read(path, **kwargs)
        frames.append(ref(rows))
        return rows, metadata

    def tracked_replay(*args, **kwargs):
        nonlocal completed
        assert len(frames) == (completed + 1) * 2
        result = replay(*args, **kwargs)
        assert all(frame() is None for frame in frames)
        completed += 1
        return result

    monkeypatch.setattr(portfolio, "read_predictions", tracked_read)
    monkeypatch.setattr(portfolio, "replay_maker_calibration", tracked_replay)
    output = portfolio.calibrate_maker(config)
    assert completed == 3
    assert len(frames) == 6
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    assert artifact.metadata["model_ids"] == dict.fromkeys(
        ("first", "second", "third"), ["price", "proxy"]
    )
    for table in (
        "crypto_maker_calibration.parquet",
        "crypto_maker_samples.parquet",
        "crypto_maker_events.parquet",
    ):
        with baseline.table(table) as expected, artifact.table(table) as actual:
            for fold in ("first", "second", "third"):
                pd.testing.assert_frame_equal(
                    actual.loc[actual.fold_id.eq(fold)]
                    .drop(columns="fold_id")
                    .reset_index(drop=True),
                    expected.drop(columns="fold_id"),
                    check_exact=True,
                )


def test_dev_maker_consumes_causal_calibration_with_independent_explicit_windows(
    independent_maker,
):
    config, backtest = independent_maker
    config["chronology"] = {
        "folds": [
            {
                "fold_id": "other",
                "train": _window(0, 1000),
                "validation": _window(6000, 7000),
            }
        ],
        "test": _window(12000, 13000),
    }
    calibration = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        calibration, validate_tables=portfolio.validate_maker_calibration
    )
    assert artifact.metadata["models"][0]["window"] == config["models"][0]["window"]
    assert artifact.metadata["models"][0]["cutoff"] == config["models"][0]["cutoff"]
    ledger = read_portfolio_ledger_artifact(portfolio.backtest_maker(backtest))
    assert ledger.metadata["sample_role"] == "prediction"
    assert ledger.metadata["window"] == backtest["prediction_window"]
    assert (
        summarize_crypto_ledger(ledger, ledger.iter_chunks())["execution"][
            "selection_status"
        ]
        == "COMPLETE"
    )


@pytest.mark.parametrize("style", ("taker", "maker"))
def test_backtest_rejects_unused_chronology_in_python_and_cli(tmp_path, style):
    import json

    from typer.testing import CliRunner

    from quant_research_platform.interfaces.cli.commands import app

    extra = {"predictions"} if style == "taker" else {"models", "calibration"}
    config = dict.fromkeys(portfolio._BACKTEST_KEYS | extra)
    config["chronology"] = {"unused": True}
    operation = getattr(portfolio, f"backtest_{style}")
    with pytest.raises(ValueError, match="config_fields"):
        operation(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(
        app, [f"backtest-crypto-{style}", "--config", str(path)]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "config_fields" in str(result.exception)


@pytest.mark.parametrize(
    "damage",
    (
        None,
        "missing_tail",
        "extra_tail",
        "missing_context_tail",
        "shared_context",
        "consumer_error",
    ),
)
def test_maker_intersects_independent_domains_and_validates_unused_tails(
    tmp_path, monkeypatch, damage
):
    config, backtest = _independent_maker_case(
        tmp_path, monkeypatch, distinct_domains=True
    )
    portfolio.calibrate_maker(config)
    if damage in {"missing_tail", "extra_tail"}:
        member = Path(backtest["models"][0]["predictions"]) / "prediction.parquet"
        rows = pd.read_parquet(member)
        if damage == "missing_tail":
            rows = rows.loc[rows.timestamp.ne(pd.Timestamp(9000, unit="ms"))]
        else:
            extra = rows.iloc[[-1]].assign(timestamp=pd.Timestamp(8900, unit="ms"))
            rows = pd.concat([rows, extra], ignore_index=True).sort_values("timestamp")
        rows.to_parquet(member, index=False, write_statistics=["timestamp"])
    if damage == "missing_context_tail":
        chunks = portfolio.iter_dataset_chunks

        def without_tail(session):
            with portfolio.closing_iterator(chunks(session)) as source:
                for frame in source:
                    yield frame.loc[frame.timestamp.ne(pd.Timestamp(9000, unit="ms"))]

        monkeypatch.setattr(portfolio, "iter_dataset_chunks", without_tail)
    closed = []
    prediction_rows = portfolio._prediction_rows

    def tracked_rows(**kwargs):
        role = kwargs["evidence"][1]["model"]["model_id"]
        try:
            with portfolio.closing_iterator(prediction_rows(**kwargs)) as rows:
                for row in rows:
                    if (
                        damage == "shared_context"
                        and role == "proxy"
                        and row["timestamp_ms"] == 8800
                    ):
                        row = {**row, "book_segment_id": "other-book"}
                    yield row
        finally:
            closed.append(role)

    monkeypatch.setattr(portfolio, "_prediction_rows", tracked_rows)
    observed = []
    execute = portfolio._execute_portfolio

    def capture(**kwargs):
        decisions = kwargs["decisions"]
        if damage == "consumer_error":
            with portfolio.closing_iterator(decisions):
                next(decisions)
                raise ValueError("synthetic_consumer_failure")

        def recording():
            with portfolio.closing_iterator(decisions):
                for row in decisions:
                    observed.append(row["timestamp_ms"])
                    yield row

        return execute(**{**kwargs, "decisions": recording()})

    monkeypatch.setattr(portfolio, "_execute_portfolio", capture)
    if damage:
        reason = {
            "missing_tail": "prediction_metadata_inconsistent",
            "extra_tail": "prediction_metadata_inconsistent",
            "missing_context_tail": "misaligned_persisted_prediction_keys",
            "shared_context": "maker_prediction_keys_misaligned",
            "consumer_error": "synthetic_consumer_failure",
        }[damage]
        with pytest.raises(ValueError, match=reason):
            portfolio.backtest_maker(backtest)
        assert not Path(backtest["output"]).exists()
    else:
        saved = read_portfolio_ledger_artifact(portfolio.backtest_maker(backtest))
        assert observed == [8800]
        assert (
            summarize_crypto_ledger(saved, saved.iter_chunks())["execution"][
                "selection_status"
            ]
            == "COMPLETE"
        )
    # Corrupt artifact cardinality fails before opening the context streams.
    assert sorted(closed) == (
        [] if damage in {"missing_tail", "extra_tail"} else ["price", "proxy"]
    )


@pytest.mark.parametrize("change", ("excluded", "removed", "missingness"))
def test_saved_predictions_do_not_requalify_historical_model_features(
    tmp_path, monkeypatch, change
):
    config, backtest = _independent_maker_case(tmp_path, monkeypatch)
    if change == "excluded":
        session = portfolio.open_dataset_session(Path(backtest["inputs"]["path"]))
        # Keep current labels, eligibility and execution context, omit historical x.
        backtest["inputs"]["columns"] = [
            name for name in session.logical_columns if name != "x"
        ]
    else:
        session = portfolio.open_dataset_session(Path(backtest["inputs"]["path"]))
        with portfolio.closing_iterator(
            portfolio.iter_dataset_chunks(session)
        ) as chunks:
            frame = pd.concat(chunks, ignore_index=True)
        roles = dict(session.spec["column_roles"])
        if change == "removed":
            frame = frame.drop(columns="x")
            roles.pop("x")
        else:
            frame["x"] = np.nan
        current = tmp_path / "current-inputs"
        write_dataset(
            current,
            frame,
            spec=DataSpec(
                data_id="current_context",
                frequency="tick",
                grain="asset_timestamp",
                column_roles=roles,
                primary_key=("timestamp", "asset_id"),
                partition_columns=("year",),
            ),
        )
        backtest["inputs"]["path"] = str(current)
    portfolio.calibrate_maker(config)
    saved = read_portfolio_ledger_artifact(portfolio.backtest_maker(backtest))
    assert (
        summarize_crypto_ledger(saved, saved.iter_chunks())["execution"][
            "selection_status"
        ]
        == "COMPLETE"
    )


def test_selected_asset_does_not_require_other_prediction_assets_context(
    tmp_path, monkeypatch
):
    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )

    config, backtest = _independent_maker_case(tmp_path, monkeypatch)
    references = [
        (config["models"][0], name)
        for name in ("price_predictions", "proxy_predictions")
    ]
    references += [(item, "predictions") for item in backtest["models"]]
    for index, (item, field) in enumerate(references):
        rows, metadata = portfolio.read_predictions(Path(item[field]))
        rows = pd.concat(
            [rows, rows.assign(asset_id="eth_usdt")], ignore_index=True
        ).sort_values(["timestamp", "asset_id"])
        metadata["prediction_rows"] = len(rows)
        if "asset_ids" in metadata:
            metadata["asset_ids"] = ["btc_usdt", "eth_usdt"]
        item[field] = str(
            write_prediction_outputs(
                output_path=tmp_path / f"multi-{index}",
                output_chunks=[rows],
                metadata=metadata,
            )
        )
    portfolio.calibrate_maker(config)
    saved = read_portfolio_ledger_artifact(portfolio.backtest_maker(backtest))
    assert (
        summarize_crypto_ledger(saved, saved.iter_chunks())["execution"][
            "selection_status"
        ]
        == "COMPLETE"
    )


@pytest.mark.parametrize("eligible", (False, True))
@pytest.mark.parametrize("distinct_domains", (False, True))
def test_saved_execution_scores_ignore_changed_current_sample_filter(
    tmp_path, monkeypatch, eligible, distinct_domains
):
    config, backtest = _independent_maker_case(
        tmp_path, monkeypatch, distinct_domains=distinct_domains
    )
    portfolio.calibrate_maker(config)
    session = portfolio.open_dataset_session(Path(backtest["inputs"]["path"]))
    with portfolio.closing_iterator(portfolio.iter_dataset_chunks(session)) as chunks:
        frame = pd.concat(chunks, ignore_index=True)
    frame["eligible"] = eligible
    frame["x"] = frame["x"].fillna(0.0)
    if "y" in frame:
        frame["y"] = frame["y"].fillna(0.0)
    current = tmp_path / "changed-admission"
    write_dataset(
        current,
        frame,
        spec=DataSpec(
            data_id="current_context",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=session.spec["column_roles"],
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
        ),
    )
    backtest["inputs"] = {**backtest["inputs"], "path": str(current)}
    observed = []
    rows = portfolio._prediction_rows

    def record(**kwargs):
        with portfolio.closing_iterator(rows(**kwargs)) as source:
            for row in source:
                observed.append(row["timestamp_ms"])
                yield row

    monkeypatch.setattr(portfolio, "_prediction_rows", record)
    saved = read_portfolio_ledger_artifact(portfolio.backtest_maker(backtest))
    assert sorted(observed) == (
        [8600, 8800, 8800, 9000]
        if distinct_domains
        else [8600, 8600, 8800, 8800, 9000, 9000]
    )
    assert (
        summarize_crypto_ledger(saved, saved.iter_chunks())["execution"][
            "selection_status"
        ]
        == "COMPLETE"
    )


@pytest.mark.parametrize("distinct_domains", (False, True))
@pytest.mark.parametrize("order", ("reverse", "interleaved"))
def test_native_unordered_context_and_predictions_emit_chronological_decisions(
    tmp_path, monkeypatch, distinct_domains, order
):
    config, backtest = _independent_maker_case(
        tmp_path, monkeypatch, distinct_domains=distinct_domains
    )
    session = portfolio.open_dataset_session(Path(backtest["inputs"]["path"]))
    with portfolio.closing_iterator(portfolio.iter_dataset_chunks(session)) as chunks:
        frame = pd.concat(chunks, ignore_index=True)
    frame = (
        frame.iloc[::-1]
        if order == "reverse"
        else pd.concat([frame.iloc[1::2], frame.iloc[::2]])
    )
    current = tmp_path / "unordered-context"
    write_dataset(
        current,
        (frame.iloc[[i]] for i in range(len(frame))),
        spec=DataSpec(
            data_id="unordered_context",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=session.spec["column_roles"],
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
        ),
    )
    backtest["inputs"]["path"] = str(current)
    paths = [
        config["models"][0][field]
        for field in ("price_predictions", "proxy_predictions")
    ]
    paths += [item["predictions"] for item in backtest["models"]]
    for path in paths:
        member = Path(path) / "prediction.parquet"
        rows = pd.read_parquet(member).iloc[::-1]
        rows.to_parquet(
            member, index=False, row_group_size=1, write_statistics=["timestamp"]
        )
    output = portfolio.calibrate_maker(config)
    with read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    ).table("crypto_maker_calibration.parquet") as calibration:
        assert calibration.sample_count.tolist() == [2, 2]
    execute = portfolio._execute_portfolio
    observed = []

    def capture(**kwargs):
        def recording():
            with portfolio.closing_iterator(kwargs["decisions"]) as rows:
                for row in rows:
                    observed.append(row["timestamp_ms"])
                    yield row

        return execute(**{**kwargs, "decisions": recording()})

    monkeypatch.setattr(portfolio, "_execute_portfolio", capture)
    saved = read_portfolio_ledger_artifact(portfolio.backtest_maker(backtest))
    assert observed == ([8800] if distinct_domains else [8600, 8800, 9000])
    assert (
        summarize_crypto_ledger(saved, saved.iter_chunks())["execution"][
            "selection_status"
        ]
        == "COMPLETE"
    )


@pytest.mark.parametrize("immature_first", (False, True))
def test_ordered_prediction_context_yields_incrementally_and_skips_immature_head(
    tmp_path, monkeypatch, immature_first
):
    config, backtest = _independent_maker_case(tmp_path, monkeypatch)
    session = portfolio.open_dataset_session(Path(backtest["inputs"]["path"]))
    with portfolio.closing_iterator(portfolio.iter_dataset_chunks(session)) as chunks:
        frame = pd.concat(chunks, ignore_index=True).sort_values("timestamp")
    if immature_first:
        frame.loc[
            frame.timestamp.eq(pd.Timestamp(2100, unit="ms")),
            "future_mid_return_500ms_available_at",
        ] = pd.Timestamp(2800, unit="ms")
    current = tmp_path / "ordered-context"
    write_dataset(
        current,
        (frame.iloc[[i]] for i in range(len(frame))),
        spec=DataSpec(
            data_id="ordered_context",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=session.spec["column_roles"],
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
        ),
    )
    item = config["models"][0]
    window = TimeWindow(
        **(item["window"] if immature_first else backtest["prediction_window"])
    )
    path = (
        item["price_predictions"]
        if immature_first
        else backtest["models"][0]["predictions"]
    )
    evidence = portfolio._prediction_evidence(path, window=window)
    chunks = portfolio.iter_dataset_chunks
    fetched = []
    closed = []

    def observe(session):
        try:
            with portfolio.closing_iterator(chunks(session)) as source:
                for frame in source:
                    fetched.append(frame.timestamp.tolist())
                    yield frame
        finally:
            closed.append(True)

    monkeypatch.setattr(portfolio, "iter_dataset_chunks", observe)
    with portfolio.closing_iterator(
        portfolio._prediction_rows(
            inputs=current,
            evidence=evidence,
            window=window,
            fold_id="f",
            asset_id="btc_usdt",
            sample_role="validation" if immature_first else "test",
            label_cutoff=pd.Timestamp(2700, unit="ms"),
        )
    ) as rows:
        first = next(rows)
        if immature_first:
            assert first.timestamp == pd.Timestamp(2200, unit="ms")
            assert len(fetched) == 2
            assert list(rows) == []
        else:
            assert first["timestamp_ms"] == 8600
            assert len(fetched) == 1
    assert closed == [True]


@pytest.mark.parametrize("visible_quantity", [0.0, 2.0])
def test_native_calibration_roundtrips_unknown_trade_price(
    monkeypatch, independent_maker, visible_quantity
):
    config, _backtest = independent_maker
    trade = replace(
        _trade(2150, price="99.5", quantity="2", maker_buyer=True),
        price_index=None,
        visible_quantity=visible_quantity,
    )
    monkeypatch.setattr(
        portfolio, "iter_agg_trades", lambda *_args, **_kwargs: iter((trade,))
    )
    output = portfolio.calibrate_maker(config)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_events.parquet") as frame:
        trades = frame.loc[frame.event_kind.eq("trade")]
        assert len(trades) == 1 and trades.price_index.isna().all()
        assert trades.quantity.tolist() == [visible_quantity]
        damaged = frame.copy()
        damaged.loc[damaged.event_kind.eq("book"), "price_index"] = pd.NA
    with monkeypatch.context() as patch:
        patch.setattr(
            portfolio,
            "_calibration_rows",
            lambda *a, **k: iter(damaged.itertuples(index=False)),
        )
        with pytest.raises(ValueError, match="crypto_calibration_event_invalid"):
            list(portfolio._calibration_events(artifact, "observed", 2_700_000_000))
    events = list(portfolio._calibration_events(artifact, "observed"))
    assert next(event for event in events if event.kind == "trade").price_index is None
    with artifact.table("crypto_maker_calibration.parquet") as frame:
        assert frame.sample_count.gt(0).all()
        assert (
            frame.excluded_decisions.gt(0).all()
            if visible_quantity
            else frame.excluded_decisions.eq(0).all()
        )


@pytest.mark.parametrize(
    ("market", "mode", "order"),
    (
        ("crypto", "continuous", "reverse"),
        (None, "independent_segments", "interleaved"),
        ("a_share", "continuous", "reverse"),
    ),
)
def test_fixed_maker_validates_market_and_orders_native_context(
    tmp_path, monkeypatch, market, mode, order
):
    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
    )
    from quant_research_platform.contracts.data import ProcessedDataSpec

    panel = _inputs(tmp_path)
    _market(monkeypatch)
    session = portfolio.open_dataset_session(panel)
    with portfolio.closing_iterator(portfolio.iter_dataset_chunks(session)) as chunks:
        frame = pd.concat(chunks, ignore_index=True)
    frame = (
        frame.iloc[::-1]
        if order == "reverse"
        else pd.concat([frame.iloc[1::2], frame.iloc[::2]])
    )
    fields = dict(
        data_id="fixed_context",
        frequency="tick",
        grain="asset_timestamp",
        column_roles=session.spec["column_roles"],
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
    )
    current = tmp_path / "fixed-context"
    write_dataset(
        current,
        (frame.iloc[[i]] for i in range(len(frame))),
        spec=DataSpec(**fields)
        if market is None
        else ProcessedDataSpec(**fields, market_id=market, kind="panel"),
    )
    config = {
        "output": str(tmp_path / "fixed-ledger"),
        "inputs": str(current),
        "panel": str(panel),
        "source": "explicit-source",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "prediction_window": _window(8600, 9000),
        "execution_mode": mode,
        "capital": asdict(
            PortfolioCapital("maker", 1000.0, "USDT", "linear_perpetual")
        ),
        "execution": asdict(_maker_config()),
        "calibration": None,
        "models": [],
    }
    if market == "a_share":
        with pytest.raises(ValueError, match="input_market_mismatch"):
            portfolio.backtest_maker(config)
        assert not Path(config["output"]).exists()
        return
    observed = []
    execute = portfolio._execute_portfolio

    def capture(**kwargs):
        def decisions():
            with portfolio.closing_iterator(kwargs["decisions"]) as rows:
                for row in rows:
                    observed.append(row["timestamp_ms"])
                    yield row

        return execute(**{**kwargs, "decisions": decisions()})

    monkeypatch.setattr(portfolio, "_execute_portfolio", capture)
    output = portfolio.backtest_maker(config)
    assert observed == [8600, 8800, 9000]
    summaries = []

    def consume(artifact, batches):
        summaries.append(summarize_crypto_ledger(artifact, batches))

    if mode == "continuous":
        read_portfolio_ledger_artifact(output, consume_batches=consume)
    else:
        read_crypto_episode_artifact(
            output,
            consume_episode=lambda _index, artifact, batches: consume(
                artifact, batches
            ),
        )
    assert len(summaries) == 1
    assert summaries[0]["execution"]["selection_status"] == "COMPLETE"


@pytest.mark.parametrize("completion", ("exhaust", "close", "duplicate"))
def test_fixed_maker_spool_releases_resources_and_checks_actual_axis(
    tmp_path, monkeypatch, completion
):
    panel = _inputs(tmp_path)
    session = portfolio.open_dataset_session(panel)
    with portfolio.closing_iterator(portfolio.iter_dataset_chunks(session)) as chunks:
        frame = pd.concat(chunks, ignore_index=True)
    fields = dict(
        data_id="axis_context",
        frequency="tick",
        grain="asset_timestamp",
        column_roles={**session.spec["column_roles"], "variant": "key"},
        primary_key=("timestamp", "asset_id", "variant"),
        partition_columns=("year",),
    )
    frame["variant"] = "a"
    if completion == "duplicate":
        duplicate = frame.loc[frame.timestamp.eq(pd.Timestamp(8600, unit="ms"))].copy()
        duplicate["variant"] = "b"
        # Even a duplicate with missing execution context must not disappear.
        duplicate["segment_id"] = ""
        frame = pd.concat([frame, duplicate], ignore_index=True)
    current = tmp_path / "axis-context"
    write_dataset(current, frame, spec=DataSpec(**fields))
    temporary = portfolio.TemporaryDirectory
    paths = []

    def record_temporary(**kwargs):
        directory = temporary(dir=tmp_path, **kwargs)
        paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(portfolio, "TemporaryDirectory", record_temporary)
    rows = portfolio._fixed_maker_rows(
        current, None, TimeWindow(**_window(8600, 9000)), "btc_usdt", "fixed"
    )
    with portfolio.closing_iterator(rows):
        if completion == "duplicate":
            with pytest.raises(ValueError, match="duplicate_prediction_input_keys"):
                list(rows)
        elif completion == "close":
            assert next(rows)["timestamp_ms"] == 8600
            assert paths[0].is_dir()
        else:
            assert [row["timestamp_ms"] for row in rows] == [8600, 8800, 9000]
    assert len(paths) == 1 and not paths[0].exists()


def test_calibration_validation_shares_event_decode_and_summary_read(
    independent_maker, monkeypatch
):
    from collections import Counter

    from quant_research_platform.artifacts.statistics import StatisticsArtifact

    config, _backtest = independent_maker
    output = portfolio.calibrate_maker(config)
    batches = StatisticsArtifact._batches
    traversals, decoded = Counter(), Counter()

    def counted(self, member, **kwargs):
        traversals[member] += 1
        with portfolio.closing_iterator(batches(self, member, **kwargs)) as frames:
            for frame in frames:
                decoded[member] += len(frame)
                yield frame

    monkeypatch.setattr(StatisticsArtifact, "_batches", counted)
    artifact = read_statistics_artifact(
        output, validate_tables=portfolio.validate_maker_calibration
    )
    with artifact.table("crypto_maker_events.parquet") as events:
        assert decoded["crypto_maker_events.parquet"] == 2 * len(events)
    # One validated native replay, then one necessary all-fold domain/count check.
    assert traversals["crypto_maker_events.parquet"] == 2
    assert traversals["crypto_maker_calibration.parquet"] == 0

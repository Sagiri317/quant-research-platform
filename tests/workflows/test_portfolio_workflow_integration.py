"""Scientific native consumer boundaries, independently of any run or stage state."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import (
    read_selection,
    score,
    select,
)
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.statistics_production import model_statistics
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.research_configuration import CORE6_FACTORS
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.helpers.statistics_artifacts import statistics_config


@pytest.fixture(scope="module")
def native_comparison(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-comparison")
    dates = pd.bdate_range("2018-08-01", "2019-02-28")
    factor = CORE6_FACTORS[0]
    frame = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "timestamp": day,
                "asset_id": f"a{asset}",
                "eligible": True,
                factor: float(asset) + np.sin(index + asset) * 0.01,
                "total_market_cap": float(asset + 1),
                "forward_return_20d": asset * 0.01 + np.cos(index + asset) * 0.001,
                "forward_return_20d_available_at": day + pd.offsets.BDay(21),
            }
            for index, day in enumerate(dates)
            for asset in range(6)
        ]
    )
    spec = make_demo_research_dataset_spec(
        data_id="pv_all",
        factor_columns=(factor,),
        label_columns=("forward_return_20d", "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
    )
    write_dataset(root / "dataset", frame, spec=spec)
    dataset = root / "dataset"
    windows = {
        "quarterly": {"start": "2018-11-01", "end": "2018-12-31"},
        "global": {"start": "2019-01-02", "end": "2019-02-28"},
    }
    models, predictions, selections = {}, {}, {}
    for algorithm in ("linear_regression", "equal_weight_rank_ensemble"):
        model = fit(
            {
                "market_id": "a_share",
                "inputs": str(dataset),
                "output": str(root / algorithm / "model"),
                "model": {
                    "model_id": algorithm,
                    "model_algorithm": algorithm,
                    "input_factor_columns": [factor],
                    "target_column": "forward_return_20d",
                    "random_seed": 42,
                },
                "fit_windows": [{"start": "2018-08-01", "end": "2018-10-31"}],
                "fit_cutoff": "2018-10-31",
                "label_unit": "decimal_return",
            }
        )
        models[algorithm] = model
        for group, window in windows.items():
            predictions[group, algorithm] = predict(
                {
                    "model": str(model),
                    "inputs": str(dataset),
                    "output": str(root / algorithm / group),
                    "prediction_window": window,
                    "sample_role": "validation",
                }
            )
    for group, window in windows.items():
        scores = score(
            {
                "predictions": [str(predictions[group, model]) for model in models],
                "labels": str(dataset),
                "candidate_ids": list(models),
                "scoring_window": window,
                "validation_windows": [window],
                "cutoff": window["end"],
                "test_window": {"start": "2020-01-01", "end": "2020-03-31"},
                "selection_group": group,
                "validation_axis": "cross_sectional",
                "label_unit": "decimal_return",
                "output": str(root / group / "scores.json"),
            }
        )
        selections[group] = select(
            {
                "scores": str(scores),
                "output": str(root / group / "selection.json"),
                "selection_metric": "validation_rankic",
                "selection_direction": "maximize",
                "tie_break_order": list(models),
                "refit_windows": [{"start": "2018-08-01", "end": "2018-10-31"}, window],
                "refit_cutoff": window["end"],
            }
        )
    return SimpleNamespace(
        root=root,
        dataset=dataset,
        factor=factor,
        models=models,
        predictions=predictions,
        selections=selections,
        windows=windows,
        frame=frame,
        spec=spec,
    )


def _statistics_request(case, output, *, group, end, global_statistics=None):
    window = {"start": "2019-01-02", "end": end}
    config = statistics_config(factors=(case.factor,), selection_group=group)
    config["evaluation"] = {"primary_label": config["evaluation"]["primary_label"]}
    config["statistics"] = {
        key: value
        for key, value in config["statistics"].items()
        if key in {"hac_lag", "bootstrap_block", "bootstrap_repetitions"}
        or key == "group_comparison"
        and group == "quarterly"
    }
    config["statistics"]["bootstrap_repetitions"] = 32
    config.update(
        dataset=str(case.dataset),
        output=str(output),
        window=window,
        prediction_id="explicit_predictions",
        comparison_model_id=next(iter(case.models)),
        periods=[
            {
                "period_id": "declared_period",
                "window": window,
                "selection": str(case.selections[group]),
                "predictions": [
                    str(case.predictions["global", model]) for model in case.models
                ],
            }
        ],
    )
    if global_statistics is not None:
        config["global_statistics"] = str(global_statistics)
    return config


def test_native_model_selection_chronology(native_comparison):
    case = native_comparison
    global_selection = read_selection(case.selections["global"])
    quarterly_selection = read_selection(case.selections["quarterly"])
    assert pd.Timestamp(quarterly_selection["cutoff"]) < pd.Timestamp("2019-01-02")
    assert global_selection["scoring_window"] == case.windows["global"]
    assert quarterly_selection["scoring_window"] == case.windows["quarterly"]
    for selected in (global_selection, quarterly_selection):
        assert {item["family_id"] for item in selected["families"]} == set(case.models)
        assert all(item["trial_metrics"] for item in selected["families"])
        assert all(
            pd.Timestamp(load_model(model)[1]["fit_cutoff"])
            < pd.Timestamp(selected["scoring_window"]["start"])
            for model in case.models.values()
        )


@pytest.mark.parametrize("end,zero", [("2019-01-15", True), ("2019-02-15", False)])
def test_native_statistics_short_and_zero_maturity_windows(
    native_comparison, tmp_path, end, zero
):
    case = native_comparison
    previous = None
    for group in ("global", "quarterly"):
        output = model_statistics(
            _statistics_request(
                case, tmp_path / group, group=group, end=end, global_statistics=previous
            )
        )
        artifact = read_statistics_artifact(output)
        with artifact.table("prediction_diagnostics.parquet") as diagnostics:
            assert diagnostics.status.eq("UNDEFINED").all()
            if zero:
                assert diagnostics.observation_count.eq(0).all()
            else:
                assert diagnostics.observation_count.between(1, 19).all()
            assert diagnostics.reason.eq(
                "selection_reuses_evaluation_window"
                if group == "global"
                else "hac_lag_exceeds_observations"
            ).all()
        with artifact.table("robustness.parquet") as robustness:
            assert robustness.status.eq("UNDEFINED").all()
            assert robustness.observation_count.lt(20).all()
            assert robustness.reason.eq(
                "selection_reuses_evaluation_window"
                if group == "global"
                else "bootstrap_block_exceeds_common_observations"
            ).all()
        if group == "quarterly":
            with artifact.table("model_group_comparison.parquet") as paired:
                assert set(paired.family_id) == set(case.models)
                if zero:
                    assert paired.observation_count.eq(0).all()
                    assert paired.status.eq("UNDEFINED").all()
                    assert paired.reason.eq("no_common_observations").all()
                    assert paired.bootstrap_reason.eq("no_common_observations").all()
                else:
                    assert paired.observation_count.between(1, 19).all()
        previous = output


def test_native_prediction_missing_family_is_rejected(native_comparison, tmp_path):
    config = _statistics_request(
        native_comparison, tmp_path / "statistics", group="global", end="2019-02-15"
    )
    config["periods"][0]["predictions"].pop()
    with pytest.raises(
        ValueError, match="statistics_prediction_family_coverage_missing"
    ):
        model_statistics(config)
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("failure", ["interrupt", "close_failure"])
def test_native_prediction_interruption_preserves_primary_and_publication(
    native_comparison, tmp_path, failure
):
    case = native_comparison
    frame, metadata = read_predictions(
        case.predictions["global", next(iter(case.models))]
    )
    primary = KeyboardInterrupt("prediction publication interrupted")
    closed = []

    class Interrupted:
        def __init__(self):
            self.sent = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.sent:
                raise primary
            self.sent = True
            return frame.iloc[:3].copy()

        def close(self):
            closed.append(True)
            if failure == "close_failure":
                raise RuntimeError("secondary close failure")

    output = tmp_path / "prediction"
    with pytest.raises(KeyboardInterrupt) as caught:
        write_prediction_outputs(
            output_path=output, output_chunks=Interrupted(), metadata=metadata
        )
    assert caught.value is primary
    assert closed == [True]
    assert not output.exists()
    assert not list(tmp_path.iterdir())


def test_native_adv_gap_preserves_portfolio_statistics(tmp_path):
    from dataclasses import asdict

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        portfolio_statistics,
    )
    from quant_research_platform.research.portfolios.production import backtest
    from quant_research_platform.research.portfolios.risk import FactorRiskConfig
    from tests.helpers.portfolio_inputs import (
        _dataset_spec,
        _frames,
        _interim_specs,
        _processed_specs,
    )
    from tests.helpers.statistics_artifacts import portfolio_statistics_config

    dates = pd.bdate_range("2018-12-01", "2019-02-15")
    frames = _frames(dates)
    panel = frames["market_panel_1d"]
    panel.loc[panel.asset_id.eq("b") & panel.timestamp.eq("2019-01-02"), "amount"] = (
        np.nan
    )
    schemas = {
        spec.data_id: spec
        for catalog in (_interim_specs(), _processed_specs())
        for spec in catalog.list()
    }
    schemas["pv_all"] = _dataset_spec()
    roots = {}
    for name, frame in frames.items():
        roots[name] = tmp_path / "inputs" / name
        write_dataset(roots[name], frame, spec=schemas[name])
    model = fit(
        {
            "market_id": "a_share",
            "inputs": str(roots["pv_all"]),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "linear",
                "model_algorithm": "linear_regression",
                "input_factor_columns": ["x"],
                "target_column": "label",
                "random_seed": 42,
            },
            "fit_windows": [{"start": "2018-12-01", "end": "2018-12-28"}],
            "fit_cutoff": "2018-12-28",
            "label_unit": "decimal_return",
        }
    )
    window = {"start": "2019-01-02", "end": "2019-02-15"}
    prediction = predict(
        {
            "model": str(model),
            "inputs": str(roots["pv_all"]),
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    inputs = {
        role: str(roots[name])
        for role, name in {
            "dataset": "pv_all",
            "panel": "market_panel_1d",
            "execution": "execution_mask",
            "calendar": "trading_calendar",
            "fees": "a_share_fee_schedule",
            "market_bars": "market_bars_1d",
            "stock_master": "stock_master",
        }.items()
    }
    assumptions = {
        "initial_nav": 100.0,
        "slippage_bps": 0.0,
        "impact_bps": 0.0,
        "adv_participation": 1.0,
        "adv_sensitivities": [1.0],
    }
    ledger_root = backtest(
        {
            "candidate": {
                "candidate_id": "global_fixed",
                "selection_group": "global",
                "construction": "top20_equal_weight",
                "prediction_selector": "linear",
                "top_quantile": 0.5,
                "rebalance_schedule": "daily",
                "benchmark_id": None,
            },
            "inputs": inputs,
            "predictions": [{"path": str(prediction), "window": window}],
            "output": str(tmp_path / "ledger"),
            "output_window": window,
            "input_window": {"start": "2018-12-01", "end": "2019-02-15"},
            "formal_dataset_window": {"start": "2018-12-01", "end": "2019-02-15"},
            "prediction_window": window,
            "sample_role": "validation",
            "assumptions": assumptions,
        }
    )
    ledger = read_portfolio_ledger_artifact(ledger_root)
    batches = list(ledger.iter_chunks())
    missing_dates = []
    for batch in batches:
        facts = {(row.asset_id, row.name): row for row in batch.diagnostics}
        reason = facts.get(("b", "block_reason"))
        if reason is not None and reason.text == "adv_unavailable":
            missing_dates.append(pd.Timestamp(facts["b", "decision_timestamp"].text))
            assert ("b", "adv20") not in facts
            assert facts["b", "adv20_observation_count"].number == 20
            assert facts["b", "adv20_missing_observations"].number == 1
            assert not any(row.asset_id == "b" for row in batch.trades)
            assert not any(row.asset_id == "b" for row in batch.cashflows)
    assert len(missing_dates) == 20
    assert max(missing_dates) < pd.Timestamp("2019-01-31")
    assert any(
        row.asset_id == "b"
        and pd.Timestamp(row.timestamp_ns) >= pd.Timestamp("2019-01-31")
        for batch in batches
        for row in batch.trades
    )
    _, default = portfolio_statistics_config()
    policy = {"cscv_blocks": default["statistics"]["cscv_blocks"]}
    inputs["risk_context"] = [
        str(roots[name]) for name in ("price_volume_facts", "market_model_factors")
    ]
    output = portfolio_statistics(
        {
            "ledgers": [str(ledger_root)],
            "inputs": inputs,
            "output": str(tmp_path / "statistics"),
            "window": window,
            "sample_role": "validation",
            "benchmark_total_return_indices": {},
            "risk": asdict(FactorRiskConfig()),
            "adv_sensitivities": assumptions["adv_sensitivities"],
            "statistics": policy,
            "analyses": [
                "performance",
                "benchmark",
                "attribution",
                "capacity",
                "sharpe_inference",
                "cscv",
            ],
        }
    )
    stats = read_statistics_artifact(output)
    with stats.table("portfolio_daily.parquet") as daily:
        assert len(daily) == len(batches)
    with stats.table("portfolio_capacity.parquet") as capacity:
        affected = capacity.loc[capacity.decision_timestamp.isin(missing_dates)]
        assert not affected.empty
        assert affected.status.eq("UNDEFINED").all() and affected.capacity.isna().all()
        assert affected.reason.eq("capacity_required_adv_missing").all()
        restored = capacity.loc[~capacity.decision_timestamp.isin(missing_dates)]
        assert not restored.empty and restored.status.eq("DEFINED").all()
    with stats.table("portfolio_capacity_summary.parquet") as capacity:
        assert set(capacity.period) == {"all", "2019"}
        assert capacity.status.eq("UNDEFINED").all() and capacity.value.isna().all()
    with stats.table("portfolio_performance.parquet") as metrics:
        selected = metrics.set_index("metric_id")
        assert selected.loc["adv_unavailable_intervals", "value"] == len(missing_dates)
        assert selected.loc["adv_unavailable_buy_orders", "value"] == len(missing_dates)
        assert selected.loc["adv_unavailable_buy_target_notional", "value"] > 0
        assert selected.loc["adv_unavailable_sell_orders", "value"] == 0
    assert summarize_portfolio_ledger(ledger.capital, iter(batches))["nav"] > 0


def test_future_data_does_not_change_prior_fit_prediction_or_selection(
    native_comparison, tmp_path
):
    """Perturb later data and rerun the actual earlier producers."""
    case = native_comparison
    changed = case.frame.copy()
    future = changed.timestamp.ge("2019-01-01")
    changed.loc[future, case.factor] *= -1000
    changed.loc[future, "forward_return_20d"] += 1000
    new_input = tmp_path / "new-input"
    write_dataset(new_input, changed, spec=case.spec)
    predictions = []
    original_selection = case.selections["quarterly"].read_bytes()
    for identifier, original in case.models.items():
        _, facts = load_model(original)
        new_model = fit(
            {
                "market_id": "a_share",
                "inputs": str(new_input),
                "output": str(tmp_path / identifier / "model"),
                "model": facts["model"],
                "fit_windows": facts["fit_windows"],
                "fit_cutoff": facts["fit_cutoff"],
                "label_unit": facts["label_unit"],
            }
        )
        predicted = predict(
            {
                "model": str(new_model),
                "inputs": str(new_input),
                "output": str(tmp_path / identifier / "prediction"),
                "prediction_window": case.windows["quarterly"],
                "sample_role": "validation",
            }
        )
        predictions.append(str(predicted))
        pd.testing.assert_frame_equal(
            read_predictions(predicted)[0],
            read_predictions(case.predictions["quarterly", identifier])[0],
            check_exact=True,
        )
    window = case.windows["quarterly"]
    scores = score(
        {
            "predictions": predictions,
            "labels": str(new_input),
            "candidate_ids": list(case.models),
            "validation_windows": [window],
            "scoring_window": window,
            "cutoff": window["end"],
            "test_window": {"start": "2020-01-01", "end": "2020-03-31"},
            "selection_group": "quarterly",
            "validation_axis": "cross_sectional",
            "label_unit": "decimal_return",
            "output": str(tmp_path / "scores.json"),
        }
    )
    frozen = read_selection(case.selections["quarterly"])
    selection = select(
        {
            "scores": str(scores),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": list(case.models),
            "refit_windows": frozen["refit_windows"],
            "refit_cutoff": frozen["refit_cutoff"],
        }
    )
    changed_selection = read_selection(selection)
    for field in (
        "selected_model_id",
        "selected_family_id",
        "families",
        "common_rows",
        "metrics",
    ):
        assert changed_selection[field] == frozen[field]
    assert case.selections["quarterly"].read_bytes() == original_selection


def test_explicit_saved_predictions_ignore_changed_future_labels(
    native_comparison, tmp_path, monkeypatch
):
    from quant_research_platform.research.models import fitting

    case = native_comparison
    changed = case.frame.copy()
    future = changed.timestamp.ge(case.windows["global"]["start"])
    changed.loc[future, "forward_return_20d"] *= -100
    assert (
        changed.loc[future, "forward_return_20d"]
        .ne(case.frame.loc[future, "forward_return_20d"])
        .any()
    )
    inputs = tmp_path / "changed-labels"
    write_dataset(inputs, changed, spec=case.spec)
    monkeypatch.setattr(
        fitting,
        "_fit_model_result",
        lambda *args, **kwargs: pytest.fail("prediction attempted fitting"),
    )
    for identifier, model in case.models.items():
        trial = Path(load_model(model)[1]["trial"])
        trial_before = {
            path.relative_to(trial): path.read_bytes()
            for path in trial.rglob("*")
            if path.is_file()
        }
        assert trial_before
        before = {
            path.relative_to(model): path.read_bytes()
            for path in model.rglob("*")
            if path.is_file()
        }
        output = predict(
            {
                "model": str(model),
                "inputs": str(inputs),
                "output": str(tmp_path / identifier),
                "prediction_window": case.windows["global"],
                "sample_role": "validation",
            }
        )
        pd.testing.assert_frame_equal(
            read_predictions(output)[0],
            read_predictions(case.predictions["global", identifier])[0],
            check_exact=True,
        )
        assert {
            path.relative_to(model): path.read_bytes()
            for path in model.rglob("*")
            if path.is_file()
        } == before
        assert {
            path.relative_to(trial): path.read_bytes()
            for path in trial.rglob("*")
            if path.is_file()
        } == trial_before

"""Explicit native market facts cannot contradict model or evaluation facts."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.contracts.data import DataSpec, ProcessedDataSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.models.explanations import explain
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.statistics_production import model_statistics
from quant_research_platform.storage.datasets.writing import write_dataset


def _input(path, market):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=5).repeat(3),
            "asset_id": list("abc") * 5,
            "x": [1.0, 2.0, 3.0] * 5,
            "label": [0.1, 0.2, 0.3] * 5,
            "eligible": True,
        }
    )
    frame["label_available_at"] = frame.timestamp
    values = dict(
        data_id="synthetic_market",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            "label": "label",
            "eligible": "mask",
            "label_available_at": "context",
        },
        column_units={"x": "dimensionless", "label": "decimal_return"},
        label_availability_columns={"label": "label_available_at"},
    )
    spec = (
        DataSpec(**values)
        if market is None
        else ProcessedDataSpec(**values, market_id=market, kind="panel")
    )
    write_dataset(path, frame, spec=spec)
    return str(path)


def _fit_request(source, output):
    return dict(
        inputs=source,
        output=str(output),
        market_id="crypto",
        model=dict(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=["x"],
            target_column="label",
            random_seed=1,
        ),
        fit_windows=[dict(start="2020-01-01", end="2020-01-03")],
        fit_cutoff="2020-01-03",
    )


@pytest.mark.parametrize("market", ["crypto", None, "a_share"])
@pytest.mark.parametrize(
    "operation", ["fit", "predict", "explain", "evaluate", "score", "statistics"]
)
def test_known_market_conflicts_fail_and_unknown_remains_usable(
    tmp_path, market, operation
):
    source = _input(tmp_path / "input", market)
    output = tmp_path / "output"
    window = dict(start="2020-01-04", end="2020-01-05")
    if operation == "fit":
        function, config = fit, _fit_request(source, output)
    elif operation == "evaluate":
        function, config = (
            evaluate,
            dict(
                inputs=source,
                output=str(output),
                evaluation=dict(
                    evaluation_id="market",
                    research_dataset_id="synthetic_market",
                    market_id="crypto",
                    signals=[
                        dict(signal_id="x", source_kind="factor", source_column="x")
                    ],
                    label_columns=["label"],
                    n_groups=2,
                ),
                window=window,
                cutoff="2020-01-05",
            ),
        )
    else:
        matching = _input(tmp_path / "matching", "crypto")
        model = fit(_fit_request(matching, tmp_path / "model"))
        prediction_config = dict(
            model=str(model),
            inputs=source,
            output=str(output),
            prediction_window=window,
        )
        if operation == "predict":
            function, config = predict, prediction_config
        elif operation == "explain":
            function, config = (
                explain,
                dict(
                    model=str(model),
                    inputs=source,
                    output=str(output),
                    methods=["affine"],
                    window=window,
                    sample_role="evaluation",
                    selection=dict(max_rows=6, max_dates=None, root_seed=1),
                    output_space="model_output",
                ),
            )
        else:
            predictions = predict(
                {
                    **prediction_config,
                    "inputs": matching,
                    "output": str(tmp_path / "predictions"),
                }
            )
            if operation == "score":
                function, config = (
                    score,
                    dict(
                        predictions=[str(predictions)],
                        candidate_ids=["linear"],
                        labels=source,
                        validation_windows=[window],
                        scoring_window=window,
                        cutoff="2020-01-05",
                        selection_group="global",
                        validation_axis="cross_sectional",
                        output=str(output),
                    ),
                )
            else:
                function, config = (
                    model_statistics,
                    dict(
                        dataset=source,
                        output=str(output),
                        window=window,
                        family_ids=["linear"],
                        comparison_model_id="linear",
                        root_seed=1,
                        evaluation=dict(primary_label="label"),
                        statistics=dict(
                            hac_lag=0, bootstrap_block=1, bootstrap_repetitions=10
                        ),
                        periods=[
                            dict(
                                period_id="p",
                                window=window,
                                predictions=[str(predictions)],
                            )
                        ],
                    ),
                )
    if market == "a_share":
        with pytest.raises(ValueError, match="input_market_mismatch"):
            function(config)
        assert not output.exists()
    else:
        assert function(config) == output


@pytest.mark.parametrize("evaluation_market", ["crypto", "a_share"])
def test_prediction_facts_bind_evaluation_when_input_market_is_unknown(
    tmp_path, evaluation_market
):
    source = _input(tmp_path / "input", None)
    model = fit(_fit_request(source, tmp_path / "model"))
    window = dict(start="2020-01-04", end="2020-01-05")
    predictions = predict(
        dict(
            model=str(model),
            inputs=source,
            output=str(tmp_path / "predictions"),
            prediction_window=window,
        )
    )
    output = tmp_path / "evaluation"
    request = dict(
        inputs=source,
        predictions={"prediction": str(predictions)},
        output=str(output),
        evaluation=dict(
            evaluation_id="market",
            research_dataset_id="synthetic_market",
            market_id=evaluation_market,
            signals=[
                dict(
                    signal_id="prediction",
                    source_kind="prediction",
                    source_column="prediction",
                )
            ],
            label_columns=["label"],
            n_groups=2,
        ),
        window=window,
        cutoff="2020-01-05",
    )
    if evaluation_market == "a_share":
        with pytest.raises(ValueError, match="input_market_mismatch"):
            evaluate(request)
        assert not output.exists()
    else:
        assert read_evaluation_outputs(evaluate(request))[1][
            "metrics"
        ].n_obs.tolist() == [
            3,
            3,
        ]


@pytest.mark.parametrize("peer_market", ["a_share", None, "crypto"])
def test_common_evaluation_reuses_market_boundary_for_comparison_input(
    tmp_path, peer_market
):
    source = _input(tmp_path / "input", "a_share")
    peer = _input(tmp_path / "peer", peer_market)
    output = tmp_path / "evaluation"
    request = dict(
        inputs=source,
        output=str(output),
        evaluation=dict(
            evaluation_id="market",
            research_dataset_id="synthetic_market",
            market_id="a_share",
            signals=[dict(signal_id="x", source_kind="factor", source_column="x")],
            label_columns=["label"],
            n_groups=2,
        ),
        sample_scope="common",
        comparison_inputs=[dict(inputs=peer, signal_columns=["x"])],
        window=dict(start="2020-01-04", end="2020-01-05"),
        cutoff="2020-01-05",
    )
    if peer_market == "crypto":
        with pytest.raises(ValueError, match="input_market_mismatch"):
            evaluate(request)
        assert not output.exists()
    else:
        assert read_evaluation_outputs(evaluate(request))[1][
            "metrics"
        ].n_obs.tolist() == [
            3,
            3,
        ]


@pytest.mark.parametrize("second_market", ["crypto", "a_share"])
def test_statistics_reconciles_prediction_markets_with_unknown_label_market(
    tmp_path, second_market
):
    source = _input(tmp_path / "input", None)
    window = dict(start="2020-01-04", end="2020-01-05")
    predictions = []
    for model_id, market in (("linear", "crypto"), ("other", second_market)):
        request = _fit_request(source, tmp_path / model_id)
        request["model"]["model_id"] = model_id
        request["market_id"] = market
        model = fit(request)
        predictions.append(
            str(
                predict(
                    dict(
                        model=str(model),
                        inputs=source,
                        output=str(tmp_path / f"prediction-{model_id}"),
                        prediction_window=window,
                    )
                )
            )
        )
    output = tmp_path / "statistics"
    request = dict(
        dataset=source,
        output=str(output),
        window=window,
        family_ids=["linear", "other"],
        comparison_model_id="linear",
        root_seed=1,
        evaluation=dict(primary_label="label"),
        statistics=dict(hac_lag=0, bootstrap_block=1, bootstrap_repetitions=10),
        periods=[dict(period_id="p", window=window, predictions=predictions)],
    )
    if second_market == "a_share":
        with pytest.raises(ValueError, match="input_market_mismatch"):
            model_statistics(request)
        assert not output.exists()
    else:
        assert model_statistics(request) == output

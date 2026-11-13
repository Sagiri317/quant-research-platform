"""Native DeepLOB regression/classification and continuous observed histories."""

import numpy as np
import pandas as pd
import pytest
import torch

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.compute.classification import (
    classification_metrics,
    direction_classes,
)
from quant_research_platform.contracts.models import (
    CLASSIFICATION_METRICS,
    DEEPLOB_INPUT_COLUMNS,
    PROBABILITY_COLUMNS,
    ModelSpec,
)
from quant_research_platform.markets.implementations.a_share.processed.high_frequency import (  # noqa: E501
    build_panel,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.protocols.a_share_hft import processed_specs
from quant_research_platform.sources.venvoo import normalize
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.markets.a_share.test_high_frequency import (
    INSTRUMENTS,
    raw_events,
    raw_quotes,
)


def book_dataset(root):
    panels = []
    for day in (20180214, 20180215, 20180216):
        quotes = normalize(
            raw_quotes(
                (93000000, 93003000, 93006000, 93009000, 93012000, 93030000, 93033000)
            ).assign(date=day),
            "hft_quotes",
            INSTRUMENTS,
        )
        orders = normalize(
            raw_events("hft_orders").assign(date=day), "hft_orders", INSTRUMENTS
        )
        trades = normalize(
            raw_events("hft_trades").assign(date=day), "hft_trades", INSTRUMENTS
        )
        panels.append(build_panel(quotes, orders, trades, max_gap_seconds=6))
    panel_path = root / "panel"
    write_dataset(
        panel_path,
        pd.concat(panels, ignore_index=True),
        spec=next(item for item in processed_specs() if item.data_id == "hft_panel"),
    )
    window = {"start": "2018-02-14", "end": "2018-02-16"}
    labels = prepare(
        {
            "market_id": "a_share",
            "data_id": "hft_labels",
            "inputs": {"hft_panel": str(panel_path)},
            "label_columns": ["future_mid_return_3s"],
            "time_window": window,
            "output_root": str(root / "labels"),
        }
    )
    return assemble(
        {
            "market_id": "a_share",
            "dataset_id": "book_test",
            "inputs": {"hft_panel": str(panel_path), "hft_labels": str(labels)},
            "base_input_data_id": "hft_panel",
            "feature_columns": list(DEEPLOB_INPUT_COLUMNS),
            "factor_columns": [],
            "label_columns": ["future_mid_return_3s"],
            "filter_columns": ["eligible"],
            "context_columns": ["session_date", "segment_id"],
            "feature_recipe": None,
            "time_window": window,
            "output_root": str(root / "assembled"),
        }
    )


@pytest.mark.parametrize("algorithm", ["deeplob_regressor", "deeplob_classifier"])
def test_native_deeplob_roundtrip_probabilities_and_segment_isolation(
    tmp_path, algorithm, monkeypatch
):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = book_dataset(tmp_path / "inputs")
        params = {
            "epochs": 1,
            "batch_size": 8,
            "sequence_length": 4,
            "hidden_size": 4,
            "conv_channels": 2,
            "inception_channels": 2,
            "device": "cpu",
        }
        if algorithm == "deeplob_classifier":
            params["class_threshold"] = 0.001
        fitted = fit(
            {
                "inputs": str(data),
                "market_id": "a_share",
                "model": {
                    "model_id": algorithm,
                    "model_algorithm": algorithm,
                    "input_feature_columns": list(DEEPLOB_INPUT_COLUMNS),
                    "target_column": "future_mid_return_3s",
                    "hyperparameters": params,
                },
                "fit_windows": [{"start": "2018-02-14", "end": "2018-02-14"}],
                "fit_cutoff": "2018-02-14T07:00:00",
                "output": str(tmp_path / "model"),
            }
        )
        result, _ = load_model(fitted)
        frame = read_dataset(data)
        frame = frame.loc[frame.timestamp.ge("2018-02-15")].reset_index(drop=True)
        expected = result.prediction_model.predict(frame)
        changed = frame.copy()
        changed.loc[
            changed.timestamp.lt("2018-02-16"), list(DEEPLOB_INPUT_COLUMNS)
        ] *= 2
        last_day = frame.timestamp.ge("2018-02-16")
        np.testing.assert_allclose(
            result.prediction_model.predict(changed)[last_day],
            expected[last_day],
            atol=1e-7,
            rtol=1e-6,
        )
        changed = frame.copy()
        first_segment = frame.segment_id.eq(frame.segment_id.iloc[0])
        changed.loc[first_segment, list(DEEPLOB_INPUT_COLUMNS)] *= 2
        np.testing.assert_allclose(
            result.prediction_model.predict(changed)[~first_segment],
            expected[~first_segment],
            atol=1e-7,
            rtol=1e-6,
        )
        from quant_research_platform.research.models.explanations import (
            _output,
            explain,
        )

        missing_book = frame.copy()
        missing_book.loc[0, "ask_vol10"] = np.nan
        assert np.isnan(_output(result, missing_book, "model_output")[0][0])
        explain(
            {
                "model": str(fitted),
                "inputs": str(data),
                "output": str(tmp_path / "explanation"),
                "methods": ["permutation"],
                "window": {
                    "start": "2018-02-15T00:00:00",
                    "end": "2018-02-16T08:00:00",
                },
                "sample_role": "synthetic",
                "output_space": "model_output",
                "selection": {"max_rows": 100, "max_dates": None},
                "permutation": {
                    "metrics": ["ic", "mse"],
                    "cutoff": "2018-02-16T08:00:00",
                    "axis": "temporal_blocks",
                    "episode_column": "segment_id",
                    "block_size": 1,
                    "step_seconds": 3,
                    "groups": {"book": list(DEEPLOB_INPUT_COLUMNS)},
                    "repeats": 1,
                },
            }
        )
        from quant_research_platform.research.predictions import runner

        consumed = []
        original_chunks = runner.iter_dataset_time_chunks

        def measured_chunks(session):
            for chunk in original_chunks(session):
                consumed.extend(chunk.timestamp.tolist())
                yield chunk

        monkeypatch.setattr(runner, "iter_dataset_time_chunks", measured_chunks)
        built_predictions = []
        predictor_type = type(result.prediction_model)
        original_prediction_function = predictor_type.prediction_function

        def measured_prediction_function(self, *, probabilities=False):
            built_predictions.append(probabilities)
            return original_prediction_function(self, probabilities=probabilities)

        monkeypatch.setattr(
            predictor_type, "prediction_function", measured_prediction_function
        )
        output = predict(
            {
                "model": str(fitted),
                "inputs": str(data),
                "prediction_window": {
                    "start": "2018-02-15T00:00:00",
                    "end": "2018-02-16T08:00:00",
                },
                "output": str(tmp_path / "predictions"),
            }
        )
        assert built_predictions == [algorithm == "deeplob_classifier"]
        predictions, metadata = read_predictions(output)
        projection = ["timestamp", "asset_id", "prediction_score"]
        selected, _ = read_predictions(
            output, columns=projection, include_available_probabilities=True
        )
        expected_columns = [
            *projection,
            *(PROBABILITY_COLUMNS if algorithm == "deeplob_classifier" else ()),
        ]
        pd.testing.assert_frame_equal(selected, predictions[expected_columns])
        for invalid_projection in ((), ("prediction_score", "prediction_score")):
            with pytest.raises(
                ValueError, match="prediction_projection_columns_invalid"
            ):
                read_predictions(
                    output,
                    columns=invalid_projection,
                    include_available_probabilities=True,
                )
        assert min(consumed) >= pd.Timestamp("2018-02-15")
        np.testing.assert_allclose(
            predictions.prediction_score, expected, atol=1e-7, rtol=1e-6
        )
        if algorithm == "deeplob_classifier":
            probabilities = predictions[list(PROBABILITY_COLUMNS)].to_numpy()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            np.testing.assert_allclose(
                predictions.prediction_score,
                probabilities[:, 2] - probabilities[:, 0],
                atol=1e-12,
            )
            assert metadata["label_unit"] == "decimal_return"
            from quant_research_platform.artifacts.outputs.evaluation import (
                read_evaluation_outputs,
            )
            from quant_research_platform.contracts.json_codec import (
                parse_qrp_json_object,
            )
            from quant_research_platform.research.evaluations.runner import evaluate
            from quant_research_platform.research.models.statistics import (
                _selection_trial_rows,
            )
            from quant_research_platform.research.models.trial_selection import (
                read_selection,
                score,
                select,
            )

            window = {"start": "2018-02-15", "end": "2018-02-16"}
            scores = score(
                {
                    "predictions": [str(output)],
                    "labels": str(data),
                    "candidate_ids": [algorithm],
                    "scoring_window": window,
                    "validation_windows": [window],
                    "cutoff": "2018-02-16T08:00:00",
                    "selection_group": "quarterly",
                    "validation_axis": "single_asset_time_series",
                    "sessions": ["2018-02-15", "2018-02-16"],
                    "output": str(tmp_path / "scores.json"),
                }
            )
            from quant_research_platform.contracts.data import ProcessedDataSpec
            from quant_research_platform.storage.datasets import inspect_dataset

            declaration = inspect_dataset(data).spec
            changed_labels = tmp_path / "changed_labels"
            write_dataset(
                changed_labels,
                read_dataset(data),
                spec=ProcessedDataSpec(
                    data_id="changed",
                    market_id="a_share",
                    kind="panel",
                    frequency="tick",
                    grain="asset_timestamp",
                    primary_key=tuple(declaration["primary_key"]),
                    partition_columns=(),
                    column_roles=declaration["column_roles"],
                    column_units=declaration["column_units"],
                    label_availability_columns=declaration[
                        "label_availability_columns"
                    ],
                ),
                metadata={
                    "producer_diagnostics": {
                        "label_semantics": {
                            "future_mid_return_3s": {"horizon_seconds": 99}
                        }
                    }
                },
            )
            with pytest.raises(
                ValueError, match="scoring_classification_target_mismatch"
            ):
                score(
                    {
                        "predictions": [str(output)],
                        "labels": str(changed_labels),
                        "candidate_ids": [algorithm],
                        "scoring_window": window,
                        "validation_windows": [window],
                        "cutoff": "2018-02-16T08:00:00",
                        "selection_group": "quarterly",
                        "validation_axis": "single_asset_time_series",
                        "sessions": ["2018-02-15", "2018-02-16"],
                        "output": str(tmp_path / "invalid_scores.json"),
                    }
                )
            metrics = parse_qrp_json_object(scores.read_bytes())["metrics"][0]
            assert (
                metrics["validation_log_loss"] > 0 and metrics["validation_mse"] is None
            )
            selection = select(
                {
                    "scores": str(scores),
                    "output": str(tmp_path / "selection.json"),
                    "selection_metric": "validation_log_loss",
                    "selection_direction": "minimize",
                    "tie_break_order": [algorithm],
                    "refit_windows": [{"start": "2018-02-14", "end": "2018-02-16"}],
                    "refit_cutoff": "2018-02-16T08:00:00",
                }
            )
            trial_rows = _selection_trial_rows(
                {"synthetic": read_selection(selection)}, "quarterly"
            ).set_index("metric_id")
            # Two days each have five observed three-second label intervals.
            assert metrics["validation_rows"] == 10
            assert (
                trial_rows.loc[
                    [f"validation_{name}" for name in CLASSIFICATION_METRICS],
                    "observation_count",
                ].tolist()
                == [10] * 4
            )
            evaluation_config = {
                "inputs": str(data),
                "predictions": {"direction": str(output)},
                "window": window,
                "cutoff": "2018-02-16T08:00:00",
                "evaluation": {
                    "evaluation_id": "classification",
                    "market_id": "a_share",
                    "research_dataset_id": "book_test",
                    "signals": [
                        {
                            "signal_id": "direction",
                            "source_column": "direction",
                            "source_kind": "prediction",
                        }
                    ],
                    "label_columns": ["future_mid_return_3s"],
                    "evaluation_axis": "single_asset_time_series",
                },
                "output": str(tmp_path / "evaluation"),
            }
            evaluation = evaluate(evaluation_config)
            metadata = parse_qrp_json_object(
                (evaluation / "manifest.json").read_bytes()
            )["metadata"]
            assert metadata["classification_metrics"]["direction"][
                "log_loss"
            ] == pytest.approx(metrics["validation_log_loss"])
            read_evaluation_outputs(evaluation)
            common = evaluate(
                {
                    **evaluation_config,
                    "sample_scope": "common",
                    "evaluation": {
                        **evaluation_config["evaluation"],
                        "evaluation_axis": "cross_sectional",
                    },
                    "output": str(tmp_path / "common-classification"),
                }
            )
            common_manifest, _ = read_evaluation_outputs(common)
            common_metrics = common_manifest["metadata"]["classification_metrics"][
                "direction"
            ]
            own_metrics = metadata["classification_metrics"]["direction"]
            assert common_metrics["rows"] == own_metrics["rows"]
            assert common_metrics["target_column"] == own_metrics["target_column"]
            for name in ("class_threshold", *CLASSIFICATION_METRICS):
                assert common_metrics[name] == pytest.approx(
                    own_metrics[name], rel=1e-12, abs=1e-15
                )
    finally:
        torch.set_num_threads(original_threads)


def test_class_targets_metrics_and_book_layout_are_explicit():
    np.testing.assert_array_equal(
        direction_classes([-0.02, -0.01, 0, 0.01, 0.02], 0.01), [0, 1, 1, 1, 2]
    )
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.6, 0.3], [0.1, 0.7, 0.2]])
    metrics = classification_metrics([-0.02, 0, 0.02], probabilities, 0.01)
    assert metrics["log_loss"] == pytest.approx(-np.log([0.8, 0.6, 0.2]).mean())
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["brier"] == pytest.approx((0.06 + 0.26 + 1.14) / 3)
    assert metrics["macro_f1"] == pytest.approx((1 + 2 / 3) / 3)
    with pytest.raises(ValueError, match="ordered_ten_level_book"):
        ModelSpec(
            model_id="bad",
            model_algorithm="deeplob_regressor",
            input_feature_columns=tuple(reversed(DEEPLOB_INPUT_COLUMNS)),
            target_column="return",
        )

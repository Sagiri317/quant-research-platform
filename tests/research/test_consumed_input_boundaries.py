"""Native producers consume selected columns and prediction asset domains."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.data import DataSpec, ProcessedDataSpec
from quant_research_platform.contracts.models import DEEPLOB_INPUT_COLUMNS
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.features.book_normalization import normalize_book
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.research.test_evaluation_diagnostics import (
    explicit_evaluation as _explicit_evaluation,
)


@pytest.fixture
def evaluation_inputs(tmp_path):
    return _explicit_evaluation.__wrapped__(tmp_path)


@pytest.mark.parametrize("duplicate_asset", ["asset_b", "asset_a"])
def test_prediction_asset_scope_precedes_narrow_key_check(
    tmp_path, evaluation_inputs, duplicate_asset
):
    frame, _, config = evaluation_inputs
    _, metadata = read_predictions(Path(config["predictions"]["pred"]))
    prediction = predict(
        {
            "inputs": config["inputs"],
            "output": str(tmp_path / "scoped"),
            "model": metadata["model"],
            "prediction_window": config["window"],
            "sample_role": "validation",
            "asset_ids": ["asset_a"],
        }
    )
    other = frame.assign(asset_id=duplicate_asset, source_row=1)
    both = pd.concat([frame.assign(source_row=0), other], ignore_index=True)
    if duplicate_asset == "asset_b":
        both = pd.concat([both, other.assign(source_row=2)], ignore_index=True)
    both = both.sort_values(["timestamp", "asset_id", "source_row"])
    declared = inspect_dataset(config["inputs"]).spec
    spec = DataSpec(
        data_id="observations",
        frequency=declared["frequency"],
        grain=declared["grain"],
        column_roles={**declared["column_roles"], "source_row": "key"},
        primary_key=("timestamp", "asset_id", "source_row"),
        partition_columns=(),
    )
    source = tmp_path / "observations"
    write_dataset(source, both, spec=spec)
    config = {
        **config,
        "inputs": str(source),
        "predictions": {"pred": str(prediction)},
        "evaluation": {
            **config["evaluation"],
            "signals": [config["evaluation"]["signals"][0]],
        },
    }
    if duplicate_asset == "asset_a":
        with pytest.raises(ValueError, match="evaluation_input_duplicate_keys"):
            evaluate(config)
    else:
        _, tables = read_evaluation_outputs(evaluate(config))
        assert tables["metrics"].set_index("label").n_obs.to_dict() == {
            "short": 2,
            "long": 3,
        }


@pytest.mark.parametrize("keep_label", [False, True])
def test_book_projection_keeps_only_consumed_roles_and_label_dependencies(
    tmp_path, keep_label
):
    columns = list(DEEPLOB_INPUT_COLUMNS)
    times = pd.to_datetime(
        [
            "2020-01-01T01:30:00",
            "2020-01-01T01:30:03",
            "2020-01-02T01:30:00",
            "2020-01-02T01:30:03",
        ]
    )
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": "asset",
            "label": [1.0, 2.0, 3.0, 4.0],
            "available_at": times + pd.Timedelta(seconds=3),
            "unused_diagnostic": "context",
            **{column: [1.0, 3.0, 5.0, 7.0] for column in columns},
        }
    )
    roles = {
        "timestamp": "timestamp",
        "asset_id": "key",
        "label": "label",
        "available_at": "context",
        "unused_diagnostic": "context",
        **dict.fromkeys(columns, "feature"),
    }
    spec = ProcessedDataSpec(
        data_id="book",
        market_id="a_share",
        kind="panel",
        frequency="tick",
        grain="asset_timestamp",
        column_roles=roles,
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_units={
            "label": "decimal_return",
            "unused_diagnostic": "text",
            **dict.fromkeys(columns, "raw_book"),
        },
        label_availability_columns={"label": "available_at"},
    )
    source = tmp_path / "book"
    write_dataset(source, frame, spec=spec)
    selected = ["timestamp", "asset_id", *columns, *(["label"] if keep_label else [])]
    root = normalize_book(
        {
            "inputs": {"path": str(source), "columns": selected},
            "output": str(tmp_path / "normalized"),
            "dataset_id": "normalized",
            "sessions": ["2020-01-01", "2020-01-02"],
            "lookback_sessions": 1,
        }
    )
    actual = read_dataset(root)
    expected_columns = {"timestamp", "asset_id", *selected} | (
        {"available_at"} if keep_label else set()
    )
    assert set(actual) == expected_columns
    info = inspect_dataset(root)
    assert set(info.spec["column_roles"]) == expected_columns
    assert set(info.spec["column_units"]) <= expected_columns
    assert info.spec["label_availability_columns"] == (
        {"label": "available_at"} if keep_label else {}
    )
    np.testing.assert_allclose(actual.loc[2:, columns[0]], [3.0, 5.0])
    if keep_label:
        pd.testing.assert_series_equal(actual.available_at, frame.available_at)


@pytest.mark.parametrize("ordinary", [False, True])
def test_scoped_prediction_domain_does_not_shrink_ordinary_factors(tmp_path, ordinary):
    from quant_research_platform.research.models.production import fit

    times = pd.date_range("2020-01-01", periods=3).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "asset_id": ["a", "b", "c"] * 3,
            "x": np.arange(1.0, 10.0),
            "label": np.arange(1.0, 10.0) * 2,
            "available_at": times,
            "eligible": True,
        }
    )
    spec = ProcessedDataSpec(
        data_id="cross_section",
        market_id="a_share",
        kind="panel",
        frequency="1d",
        grain="asset_timestamp",
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "feature",
            "label": "label",
            "available_at": "context",
            "eligible": "mask",
        },
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "available_at"},
    )
    clean = tmp_path / "clean"
    write_dataset(clean, frame, spec=spec)
    model = fit(
        {
            "inputs": str(clean),
            "output": str(tmp_path / "model"),
            "market_id": "a_share",
            "model": {
                "model_id": "linear",
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["x"],
                "target_column": "label",
            },
            "fit_windows": [{"start": "2020-01-01", "end": "2020-01-02"}],
            "fit_cutoff": "2020-01-02",
        }
    )
    window = {"start": "2020-01-03", "end": "2020-01-03"}
    prediction = predict(
        {
            "inputs": str(clean),
            "model": str(model),
            "output": str(tmp_path / "predictions"),
            "prediction_window": window,
            "sample_role": "validation",
            "asset_ids": ["a", "b"],
        }
    )
    expanded = pd.concat(
        [
            frame.assign(source_row=0),
            frame.loc[frame.asset_id.eq("c")].assign(source_row=1),
        ],
        ignore_index=True,
    )
    expanded = expanded.sort_values(["timestamp", "asset_id", "source_row"])
    source = tmp_path / "observations"
    write_dataset(
        source,
        expanded,
        spec=replace(
            spec,
            column_roles={**spec.column_roles, "source_row": "key"},
            primary_key=("timestamp", "asset_id", "source_row"),
        ),
    )
    config = {
        "inputs": str(source),
        "output": str(tmp_path / "evaluation"),
        "predictions": {"pred": str(prediction)},
        "window": window,
        "cutoff": "2020-01-03",
        "sample_scope": "common",
        "evaluation": {
            "evaluation_id": "scoped",
            "market_id": "a_share",
            "research_dataset_id": "cross_section",
            "signals": [
                {
                    "signal_id": "pred",
                    "source_kind": "prediction",
                    "source_column": "pred",
                },
                *(
                    [{"signal_id": "x", "source_kind": "feature", "source_column": "x"}]
                    if ordinary
                    else []
                ),
            ],
            "label_columns": ["label"],
            "n_groups": 2,
        },
    }
    if ordinary:
        with pytest.raises(ValueError, match="evaluation_input_duplicate_keys"):
            evaluate(config)
    else:
        _, tables = read_evaluation_outputs(evaluate(config))
        assert tables["metrics"].n_obs.tolist() == [2]

"""Open prediction keeps native persistence, causal samples, and frozen test bounds."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score, select
from quant_research_platform.research.predictions.runner import predict
from tests.research.direct_model_test_support import fit_frame
from tests.research.test_prediction_runner_model_selection import candidates


@pytest.mark.parametrize(
    "algorithm", ["linear_regression", "equal_weight_rank_ensemble"]
)
def test_native_open_prediction_is_causal_label_free_and_full_cross_section(
    tmp_path, monkeypatch, algorithm
):
    stamps = pd.date_range("2020-01-01", periods=6).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": ["a", "b", "c"] * 6,
            "x": [1.0, 2.0, 3.0] * 6,
            "label": [0.1, 0.2, 0.3] * 6,
            "eligible": True,
            "label_available_at": stamps,
        }
    )
    frame.loc[0, "label_available_at"] = pd.Timestamp("2030-01-01")
    frame.loc[0, "label"] = 1e6
    case = fit_frame(
        tmp_path / "fit",
        frame.iloc[:9],
        ModelSpec(
            model_id="open",
            model_algorithm=algorithm,
            input_factor_columns=("x",),
            target_column="label",
            random_seed=19,
        ),
    )
    assert case.result.metadata["rows_after_input_label_dropna"] == 8
    import quant_research_platform.research.models.fitting as fitting

    monkeypatch.setattr(
        fitting, "_fit_model_result", lambda *a, **kw: pytest.fail("predict trained")
    )
    future = frame.drop(columns=["label", "label_available_at"])
    future.loc[
        future.timestamp.eq(pd.Timestamp("2020-01-04")) & future.asset_id.eq("a"), "x"
    ] = np.nan
    # A later valid feature cannot backfill the missing feature at an earlier time.
    future.loc[
        future.timestamp.eq(pd.Timestamp("2020-01-06")) & future.asset_id.eq("c"),
        "eligible",
    ] = False
    source = tmp_path / "future.parquet"
    config = {
        "model": str(case.root),
        "inputs": str(source),
        "input_spec": case.config["input_spec"],
        "prediction_window": {"start": "2020-01-04", "end": None},
        "sample_role": "validation",
    }
    before = {path.name: path.read_bytes() for path in case.root.iterdir()}
    future.loc[future.timestamp <= pd.Timestamp("2020-01-05")].to_parquet(
        source, index=False
    )
    first, first_metadata = read_predictions(
        predict({**config, "output": str(tmp_path / "first")})
    )
    future.to_parquet(source, index=False)
    extended, metadata = read_predictions(
        predict({**config, "output": str(tmp_path / "extended")})
    )
    pd.testing.assert_frame_equal(
        first,
        extended.loc[extended.timestamp <= pd.Timestamp("2020-01-05")].reset_index(
            drop=True
        ),
    )
    assert metadata["prediction_window"] == config["prediction_window"]
    assert pd.Timestamp(first_metadata["effective_prediction_end"]) == pd.Timestamp(
        "2020-01-05"
    )
    assert pd.Timestamp(metadata["effective_prediction_end"]) == pd.Timestamp(
        "2020-01-06"
    )
    assert not extended.timestamp.lt(pd.Timestamp("2020-01-04")).any()
    if algorithm == "linear_regression":
        assert not (
            extended.timestamp.eq(pd.Timestamp("2020-01-04"))
            & extended.asset_id.eq("a")
        ).any()
        expected = extended.asset_id.map({"a": 0.1, "b": 0.2, "c": 0.3})
        np.testing.assert_allclose(extended.prediction_score, expected)
    scoped, scoped_metadata = read_predictions(
        predict({**config, "asset_ids": ["c"], "output": str(tmp_path / "scoped")})
    )
    pd.testing.assert_frame_equal(
        scoped, extended.loc[extended.asset_id.eq("c")].reset_index(drop=True)
    )
    assert pd.Timestamp(scoped_metadata["effective_prediction_end"]) == pd.Timestamp(
        "2020-01-05"
    )
    empty, empty_metadata = read_predictions(
        predict(
            {
                **config,
                "output": str(tmp_path / "empty"),
                "prediction_window": {"start": "2020-01-07", "end": None},
            }
        )
    )
    assert empty.empty and empty_metadata["effective_prediction_end"] is None
    assert (
        empty_metadata["prediction_rows"] == 0 and empty_metadata["status"] == "empty"
    )
    assert {path.name: path.read_bytes() for path in case.root.iterdir()} == before


def test_open_prediction_preserves_nanoseconds_in_filter_and_saved_extent(tmp_path):
    stamps = pd.to_datetime([f"2020-01-01T00:00:00.00000000{i}" for i in range(1, 6)])
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": "a",
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "label": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label_available_at": stamps,
            "eligible": True,
        }
    )
    case = fit_frame(
        tmp_path / "fit",
        frame.iloc[:2],
        ModelSpec(
            model_id="nanoseconds",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
            random_seed=19,
        ),
    )
    source = tmp_path / "future.parquet"
    frame.drop(columns=["label", "label_available_at"]).to_parquet(source, index=False)
    config = {
        "model": str(case.root),
        "inputs": str(source),
        "input_spec": case.config["input_spec"],
        "prediction_window": {"start": stamps[3].isoformat(), "end": None},
        "sample_role": "validation",
        "output": str(tmp_path / "prediction"),
    }
    output = predict(config)
    actual, metadata = read_predictions(output)
    assert actual.timestamp.tolist() == list(stamps[3:])
    assert metadata["prediction_window"] == config["prediction_window"]
    assert metadata["effective_prediction_end"] == stamps[-1].isoformat()
    manifest = output / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["metadata"]["effective_prediction_end"] = stamps[2].isoformat()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prediction_effective_end_outside_window"):
        read_predictions(output)


def test_open_prediction_cannot_bypass_frozen_test_scope(tmp_path):
    cases, config = candidates(tmp_path, algorithms=("linear_regression",))
    case = cases[0]
    scores = score(config)
    selection = select(
        {
            "scores": str(scores),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["linear_regression"],
            "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "refit_cutoff": "2020-01-06",
        }
    )
    final = fit(
        {
            **case.config,
            "inputs": config["labels"],
            "output": str(tmp_path / "final"),
            "fit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "fit_cutoff": "2020-01-06",
            "selection": str(selection),
            "test_window": config["test_window"],
        }
    )
    for role, error in (
        ("test", "prediction_window_outside_freeze"),
        ("validation", "prediction_window_outside_freeze"),
    ):
        with pytest.raises(ValueError, match=error):
            predict(
                {
                    "model": str(final),
                    "inputs": config["labels"],
                    "input_spec": case.config["input_spec"],
                    "output": str(tmp_path / role),
                    "prediction_window": {"start": "2020-01-07", "end": None},
                    "sample_role": role,
                }
            )
        assert not (tmp_path / role).exists()

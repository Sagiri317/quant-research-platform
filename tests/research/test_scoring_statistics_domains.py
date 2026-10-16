"""Persisted prediction coverage is independent of supervised sample eligibility."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    read_predictions,
    write_prediction_outputs,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.trial_selection import score, select
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.statistics_production import model_statistics
from quant_research_platform.storage.datasets import write_dataset
from tests.research.direct_model_test_support import fit_frame


def _source(root, frame, declared, *, bare):
    declared = {
        **declared,
        "column_roles": {
            name: role
            for name, role in declared["column_roles"].items()
            if name in frame
        },
    }
    if bare:
        root = root.with_suffix(".parquet")
        frame.to_parquet(root, index=False)
    else:
        write_dataset(
            root,
            frame,
            spec=DataSpec(
                **{
                    **declared,
                    "primary_key": tuple(declared["primary_key"]),
                    "partition_columns": tuple(declared["partition_columns"]),
                }
            ),
        )
    return root, declared


@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("partial_labels", [False, True])
def test_unfiltered_coverage_precedes_mature_eligible_score_and_statistics(
    tmp_path, bare, partial_labels
):
    stamps = pd.date_range("2020-01-01", periods=6).repeat(3)
    frame = pd.DataFrame(
        {
            "timestamp": stamps,
            "asset_id": list("abc") * 6,
            "x": [1.0, 2.0, 3.0] * 6,
            "label": [0.1, 0.2, 9.0] * 6,
            "eligible": [True, True, False] * 6,
            "label_available_at": stamps,
        }
    )
    case = fit_frame(
        tmp_path / "fit",
        frame.iloc[:9],
        ModelSpec(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
            sample_filter_mode="unfiltered",
            random_seed=1,
        ),
    )
    assert case.result.metadata["rows_after_input_label_dropna"] == 6
    if partial_labels:
        frame.loc[
            frame.timestamp.eq(pd.Timestamp("2020-01-05")) & frame.asset_id.eq("b"),
            "label",
        ] = np.nan
        frame.loc[
            frame.timestamp.eq(pd.Timestamp("2020-01-06")), "label_available_at"
        ] = pd.Timestamp("2020-01-10")
    window = {"start": "2020-01-04", "end": "2020-01-06"}
    source, declared = _source(
        tmp_path / "inputs", frame, case.config["input_spec"], bare=bare
    )
    prediction = predict(
        {
            "model": str(case.root),
            "inputs": str(source),
            **({"input_spec": declared} if bare else {}),
            "prediction_window": window,
            "output": str(tmp_path / "predictions"),
        }
    )
    rows, metadata = read_predictions(prediction)
    assert len(rows) == 9 and rows.asset_id.eq("c").sum() == 3
    shutil.rmtree(case.root)
    shutil.rmtree(case.root.with_name(case.root.name + ".trial"))
    labels, label_spec = _source(
        tmp_path / "labels", frame.drop(columns="x"), declared, bare=bare
    )
    scoring = {
        "predictions": [str(prediction)],
        "candidate_ids": ["linear"],
        "labels": str(labels),
        **({"label_spec": label_spec} if bare else {}),
        "validation_windows": [window],
        "scoring_window": window,
        "cutoff": "2020-01-06",
        "selection_group": "global",
        "validation_axis": "cross_sectional",
        "label_unit": "decimal_return",
        "output": str(tmp_path / "scores.json"),
    }
    scores = score(scoring)
    scored = json.loads(scores.read_text())
    expected_counts = [2, 1, 0] if partial_labels else [2, 2, 2]
    assert scored["common_rows"] == sum(expected_counts)
    assert scored["metrics"][0]["validation_rows"] == sum(expected_counts)
    assert scored["metrics"][0]["validation_mse"] < 1e-24
    statistics = {
        "dataset": str(labels),
        **({"input_spec": label_spec} if bare else {}),
        "output": str(tmp_path / "statistics"),
        "window": window,
        "family_ids": ["linear"],
        "comparison_model_id": "linear",
        "root_seed": 1,
        "evaluation": {"primary_label": "label"},
        "statistics": {"hac_lag": 0, "bootstrap_block": 1, "bootstrap_repetitions": 10},
        "periods": [
            {"period_id": "p", "window": window, "predictions": [str(prediction)]}
        ],
    }
    chosen = select(
        {
            "scores": str(scores),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["linear"],
            "refit_windows": [{"start": "2020-01-01", "end": "2020-01-06"}],
            "refit_cutoff": "2020-01-06",
        }
    )
    selected = {
        **statistics,
        "dataset": str(source),
        **({"input_spec": declared} if bare else {}),
        "output": str(tmp_path / "selected-statistics"),
        "selection_group": "global",
        "prediction_id": "selected",
        "periods": [{**statistics["periods"][0], "selection": str(chosen)}],
    }
    selected.pop("family_ids")
    for config in (statistics, selected):
        artifact = read_statistics_artifact(model_statistics(config))
        with artifact.table("model_losses.parquet") as losses:
            assert losses.observation_count.tolist() == expected_counts
            assert losses.loss.iloc[0] == pytest.approx(-1.0)
            if partial_labels:
                assert losses.loss.iloc[1:].isna().all()
                assert losses.status.iloc[1:].eq("UNDEFINED").all()
    missing_labels, _ = _source(
        tmp_path / "missing-labels",
        frame.loc[
            ~(frame.timestamp.eq(pd.Timestamp("2020-01-06")) & frame.asset_id.eq("c"))
        ].drop(columns="x"),
        declared,
        bare=bare,
    )
    with pytest.raises(ValueError, match="candidate_label_coverage_missing"):
        score(
            {
                **scoring,
                "labels": str(missing_labels),
                "output": str(tmp_path / "missing-scores.json"),
            }
        )
    with pytest.raises(ValueError, match="prediction_dataset_coverage_mismatch"):
        model_statistics(
            {
                **statistics,
                "dataset": str(missing_labels),
                "output": str(tmp_path / "missing-statistics"),
            }
        )
    incomplete = write_prediction_outputs(
        output_path=tmp_path / "incomplete-predictions",
        output_chunks=[rows.loc[~rows.asset_id.eq("c")]],
        metadata={
            **metadata,
            "prediction_rows": None,
            "effective_prediction_end": None,
        },
    )
    with pytest.raises(ValueError, match="candidate_prediction_coverage_missing"):
        score(
            {
                **scoring,
                "predictions": [str(incomplete)],
                "output": str(tmp_path / "incomplete-scores.json"),
            }
        )
    with pytest.raises(ValueError, match="missing_required_prediction_keys"):
        model_statistics(
            {
                **selected,
                "periods": [
                    {**selected["periods"][0], "predictions": [str(incomplete)]}
                ],
                "output": str(tmp_path / "incomplete-statistics"),
            }
        )

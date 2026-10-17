"""Disjoint native fits exclude supplied gap rows and unavailable labels."""

import json

import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import score, select
from quant_research_platform.research.predictions.runner import predict


def test_disjoint_refit_counts_and_predictions_ignore_gap_and_late_labels(tmp_path):
    start = pd.Timestamp("2020-01-01T00:00:00")
    frame = pd.DataFrame(
        [
            (
                start + pd.Timedelta(seconds=t),
                asset,
                float(t + i),
                float(2 * (t + i) + 1),
                not (t == 2 and i == 0),
                start + pd.Timedelta(seconds=t + 1),
            )
            for t in range(12)
            for i, asset in enumerate(("a", "b"))
        ],
        columns=[
            "timestamp",
            "asset_id",
            "x",
            "label",
            "eligible",
            "label_available_at",
        ],
    )

    def bound(second):
        return (start + pd.Timedelta(seconds=second)).isoformat()

    def window(first, last):
        return {"start": bound(first), "end": bound(last)}

    source = tmp_path / "source.parquet"
    frame.to_parquet(source, index=False)
    spec = {
        "data_id": "disjoint",
        "frequency": "tick",
        "grain": "asset_timestamp",
        "primary_key": ["timestamp", "asset_id"],
        "partition_columns": ["year"],
        "column_roles": {
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "feature",
            "label": "label",
            "eligible": "mask",
            "label_available_at": "context",
        },
    }
    config = {
        "market_id": "crypto",
        "inputs": str(source),
        "input_spec": spec,
        "model": {
            "model_id": "linear",
            "model_algorithm": "linear_regression",
            "input_feature_columns": ["x"],
            "target_column": "label",
            "random_seed": 17,
        },
        "label_unit": "decimal_return",
        "fit_windows": [window(1, 3)],
        "fit_cutoff": bound(3),
        "output": str(tmp_path / "train"),
    }
    train_path = fit(config)
    trained, _ = load_model(train_path)
    assert trained.metadata["rows_after_input_label_dropna"] == 3
    union = [window(1, 3), window(6, 8)]
    validation = predict(
        {
            "model": str(train_path),
            "inputs": str(source),
            "input_spec": spec,
            "output": str(tmp_path / "validation-pred"),
            "prediction_window": window(6, 8),
            "sample_role": "validation",
        }
    )
    scored = score(
        {
            "predictions": [str(validation)],
            "labels": str(source),
            "label_spec": spec,
            "candidate_ids": ["linear"],
            "scoring_window": window(6, 8),
            "validation_windows": [window(6, 8)],
            "cutoff": bound(8),
            "selection_group": "global",
            "validation_axis": "cross_sectional",
            "label_unit": "decimal_return",
            "output": str(tmp_path / "scores.json"),
        }
    )
    assert json.loads(scored.read_text())["common_rows"] == 4
    selection = select(
        {
            "scores": str(scored),
            "output": str(tmp_path / "selection.json"),
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "tie_break_order": ["linear"],
            "refit_windows": union,
            "refit_cutoff": bound(8),
        }
    )
    predictions = []
    for variant in ("original", "perturbed"):
        input_path = source
        if variant == "perturbed":
            changed = frame.copy()
            second = (changed.timestamp - start).dt.total_seconds()
            excluded = ~((second.between(1, 3)) | (second.between(6, 8)))
            excluded |= changed.label_available_at.gt(start + pd.Timedelta(seconds=8))
            excluded |= ~changed.eligible
            changed.loc[excluded, ["x", "label"]] = -1e12
            input_path = tmp_path / "perturbed.parquet"
            changed.to_parquet(input_path, index=False)
        model_path = fit(
            {
                **config,
                "inputs": str(input_path),
                "output": str(tmp_path / variant),
                "fit_windows": union,
                "fit_cutoff": bound(8),
                "selection": str(selection),
            }
        )
        fitted, facts = load_model(model_path)
        assert fitted.metadata["rows_after_input_label_dropna"] == 9
        assert facts["fit_windows"] == union
        published = predict(
            {
                "model": str(model_path),
                "inputs": str(source),
                "input_spec": spec,
                "output": str(tmp_path / (variant + "-pred")),
                "prediction_window": window(9, 11),
                "sample_role": "test",
            }
        )
        prediction = read_predictions(published)[0]
        assert len(prediction) == 6
        expected_keys = frame.loc[
            frame.timestamp.ge(start + pd.Timedelta(seconds=9)),
            ["timestamp", "asset_id"],
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            prediction.loc[:, ["timestamp", "asset_id"]].reset_index(drop=True),
            expected_keys,
            check_dtype=False,
        )
        predictions.append(prediction)
    pd.testing.assert_frame_equal(*predictions, check_exact=False, rtol=1e-12, atol=0.0)

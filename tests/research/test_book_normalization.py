"""Independent historical moments, calendar gaps, and native normalized books."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.models import DEEPLOB_INPUT_COLUMNS
from quant_research_platform.research.features.book_normalization import normalize_book
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)


def test_native_historical_book_normalization_and_deeplob(tmp_path):
    columns = list(DEEPLOB_INPUT_COLUMNS)
    rows = []
    for day in range(1, 5):
        for asset, shift in (("a", 0), ("b", 10)):
            for observation in range(2):
                timestamp = pd.Timestamp(2020, 1, day, 1, 30, observation * 3)
                rows.append(
                    {
                        "timestamp": timestamp,
                        "asset_id": asset,
                        "session_date": timestamp.normalize(),
                        "segment_id": 0,
                        "eligible": True,
                        "label": observation * 0.001,
                        "label_available_at": timestamp + pd.Timedelta(seconds=3),
                        **dict.fromkeys(
                            columns, 4 * (day - 1) + 1 + 2 * observation + shift
                        ),
                    }
                )
    frame = pd.DataFrame(rows).astype(dict.fromkeys(columns, "float64"))
    roles = {
        "timestamp": "timestamp",
        "asset_id": "key",
        "session_date": "context",
        "segment_id": "context",
        "eligible": "mask",
        "label": "label",
        "label_available_at": "context",
        **dict.fromkeys(columns, "feature"),
    }
    spec = ResearchDatasetSpec(
        data_id="book",
        market_id="a_share",
        frequency="tick",
        grain="asset_timestamp",
        column_roles=roles,
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_units={"label": "decimal_return", **dict.fromkeys(columns, "raw_book")},
        label_availability_columns={"label": "label_available_at"},
        input_data_ids=("raw",),
        base_input_data_id="raw",
        feature_columns=tuple(columns),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("session_date", "segment_id", "label_available_at"),
        target_transform="raw",
    )
    source = tmp_path / "raw"
    write_dataset(source, frame, spec=spec)
    config = {
        "inputs": str(source),
        "output": str(tmp_path / "normalized"),
        "dataset_id": "normalized_book",
        "lookback_sessions": 2,
        "sessions": [f"2020-01-0{day}" for day in range(1, 5)],
    }
    output = normalize_book(config)
    normalized = read_dataset(output)
    assert (
        normalized.loc[normalized.session_date.lt("2020-01-03"), columns]
        .isna()
        .all(axis=None)
    )
    # Previous observations [1,3,5,7] have mean 4 and population variance 5.
    for asset in ("a", "b"):
        observed = normalized.loc[
            normalized.asset_id.eq(asset) & normalized.session_date.eq("2020-01-03")
        ]
        np.testing.assert_allclose(observed[columns[0]], np.array([5, 7]) / np.sqrt(5))
    metadata = inspect_dataset(output)
    assert metadata.spec["column_units"][columns[0]] == "zscore"
    moments = metadata.manifest["producer_diagnostics"]["book_normalization"][
        "daily_parameters"
    ]
    assert moments[0]["history_rows"] == 4
    assert moments[0]["mean"][0] == 4
    model = fit(
        {
            "inputs": str(output),
            "output": str(tmp_path / "model"),
            "market_id": "a_share",
            "model": {
                "model_id": "normalized_deeplob",
                "model_algorithm": "deeplob_regressor",
                "input_feature_columns": columns,
                "target_column": "label",
                "training_scaling": False,
                "hyperparameters": {
                    "device": "cpu",
                    "epochs": 1,
                    "sequence_length": 2,
                    "hidden_size": 2,
                    "conv_channels": 2,
                    "inception_channels": 2,
                },
            },
            "fit_windows": [{"start": "2020-01-03", "end": "2020-01-03"}],
            "fit_cutoff": "2020-01-03T08:00:00",
        }
    )
    assert load_model(model)[0].metadata["scaler"]["scaling"] is False
    predictions, _ = read_predictions(
        predict(
            {
                "model": str(model),
                "inputs": str(output),
                "output": str(tmp_path / "predictions"),
                "prediction_window": {"start": "2020-01-04", "end": "2020-01-04"},
            }
        )
    )
    assert len(predictions) == 4
    assert np.isfinite(predictions.prediction_score).all()
    # A missing intervening declared session invalidates its subsequent lookback.
    missing_source = tmp_path / "missing_day"
    write_dataset(
        missing_source, frame.loc[frame.session_date.ne("2020-01-02")], spec=spec
    )
    missing = read_dataset(
        normalize_book(
            {
                **config,
                "inputs": str(missing_source),
                "output": str(tmp_path / "gapped"),
            }
        )
    )
    assert missing[columns].isna().all(axis=None)
    with pytest.raises(ValueError, match="normalization_sessions_invalid"):
        normalize_book(
            {
                **config,
                "output": str(tmp_path / "invalid"),
                "sessions": ["2020-01-02", "2020-01-01"],
            }
        )

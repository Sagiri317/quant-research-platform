# Structural serialization fixtures; metric values are not statistical oracles.
from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

PRICE_VOLUME_SIGNAL_COLUMN = "momentum_20d"
FORWARD_RETURN_LABEL_COLUMN = forward_return_label_column(20)


def make_price_volume_prediction_output_frame(
    *,
    with_metadata: bool = True,
    row_count: int = 1,
) -> pd.DataFrame:
    frame = _price_volume_prediction_frame(row_count=row_count)
    if with_metadata:
        frame.attrs["model_metadata"] = _price_volume_model_metadata()
        frame.attrs["prediction_metadata"] = {
            **_price_volume_prediction_metadata(),
            "prediction_rows": row_count,
        }
    return frame


def make_price_volume_prediction_spec() -> PredictionSpec:
    return PredictionSpec(
        prediction_id="pred-run",
        model_id="linear-demo",
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )


def make_prediction_report_spec() -> OutputSpec:
    return OutputSpec(
        output_id="pred-run",
        output_definition_id="prediction_outputs",
        run_id="test_run",
        market_id=MarketId("a_share"),
        run_type="prediction",
        scope=OutputScope.LOCAL_ONLY,
    )


def make_artifact_signal_spec() -> SignalSpec:
    return SignalSpec(
        signal_id="signal",
        source_column="feature",
        source_kind="feature",
    )


def _price_volume_prediction_frame(*, row_count: int) -> pd.DataFrame:
    assets = ["asset-a", "asset-b"][:row_count]
    scores = [0.25, 0.35][:row_count]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-01-03"] * row_count),
            "asset_id": assets,
            "prediction_score": scores,
        }
    )


def _price_volume_model_metadata() -> dict[str, object]:
    return {
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2014-01-02",
        "train_end": "2014-01-02",
        "test_start": "2014-01-03",
        "test_end": "2014-01-03",
        "oos_start": "2014-01-03",
        "oos_end": "2014-01-03",
        "input_factor_columns": [PRICE_VOLUME_SIGNAL_COLUMN],
        "input_feature_columns": [],
        "input_columns": [PRICE_VOLUME_SIGNAL_COLUMN],
        "input_count": 1,
        "label_column": FORWARD_RETURN_LABEL_COLUMN,
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": None,
        "max_train_rows": None,
        "training_weighting": {"method": "equal_timestamp", "half_life": None},
        "scaler": None,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
        "feature_importance": {PRICE_VOLUME_SIGNAL_COLUMN: 1.0},
    }


def _price_volume_prediction_metadata() -> dict[str, object]:
    return {
        "prediction_mode": "minimal_oos",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_dropna": 2,
        "prediction_rows": 1,
        "train_end": "2014-01-02",
        "test_start": "2014-01-03",
        "requested_test_end": "2014-01-03",
        "effective_test_end": "2014-01-03",
        "is_oos": True,
        "is_final_oos_window": False,
        "input_columns": [PRICE_VOLUME_SIGNAL_COLUMN],
        "input_count": 1,
        "prediction_column": "prediction_score",
    }

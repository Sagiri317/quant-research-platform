from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.factors.price_volume.columns import (
    price_volume_window_column,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

PRICE_VOLUME_SIGNAL_COLUMN = price_volume_window_column("mom", 20)
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


def make_artifact_prediction_output_frame(
    *,
    with_metadata: bool = True,
) -> pd.DataFrame:
    frame = _artifacts_prediction_frame()
    if with_metadata:
        frame.attrs["model_metadata"] = _artifacts_model_metadata()
        frame.attrs["prediction_metadata"] = _artifacts_prediction_metadata()
    return frame


def make_price_volume_signal_spec() -> SignalSpec:
    return SignalSpec(
        signal_id=PRICE_VOLUME_SIGNAL_COLUMN,
        source_column=PRICE_VOLUME_SIGNAL_COLUMN,
        source_kind="factor",
    )


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


def make_price_volume_evaluation_metrics_frame() -> pd.DataFrame:
    return _evaluation_metrics_frame_from_values(_price_volume_evaluation_values())


def make_artifact_evaluation_metrics_frame() -> pd.DataFrame:
    return _evaluation_metrics_frame_from_values(_artifacts_evaluation_values())


def _evaluation_metrics_frame_from_values(values: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [values["timestamp"]],
            "signal": [values["signal"]],
            "label": [values["label"]],
            "ic": [0.1],
            "rankic": [0.2],
            "n_obs": [2],
            "ic_reason": [None],
            "rankic_reason": [None],
        }
    )


def make_price_volume_evaluation_grouped_label_values_frame() -> pd.DataFrame:
    return _evaluation_grouped_label_values_frame_from_values(
        _price_volume_evaluation_values()
    )


def make_artifact_evaluation_grouped_label_values_frame() -> pd.DataFrame:
    return _evaluation_grouped_label_values_frame_from_values(
        _artifacts_evaluation_values()
    )


def _evaluation_grouped_label_values_frame_from_values(
    values: dict[str, object],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [values["timestamp"]],
            "signal": [values["signal"]],
            "label": [values["label"]],
            "group": [1],
            "mean_label_value": [values["mean_label_value"]],
            "n_obs": [2],
        }
    )


def make_price_volume_evaluation_ic_summary_frame() -> pd.DataFrame:
    return _evaluation_ic_summary_frame_from_values(_price_volume_evaluation_values())


def make_artifact_evaluation_ic_summary_frame() -> pd.DataFrame:
    return _evaluation_ic_summary_frame_from_values(_artifacts_evaluation_values())


def _evaluation_ic_summary_frame_from_values(values: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [values["signal"]],
            "label": [values["label"]],
            "mean_ic": [0.1],
            "std_ic": [0.0],
            "icir": [None],
            "positive_ic_rate": [1.0],
            "valid_ic_count": [1],
            "mean_rankic": [0.2],
            "std_rankic": [0.0],
            "rankicir": [None],
            "positive_rankic_rate": [1.0],
            "valid_rankic_count": [1],
            "ic_reason": [None],
            "icir_reason": ["insufficient_observations"],
            "rankic_reason": [None],
            "rankicir_reason": ["insufficient_observations"],
        }
    )


def make_price_volume_evaluation_group_spread_summary_frame() -> pd.DataFrame:
    return _evaluation_group_spread_summary_frame_from_values(
        _price_volume_evaluation_values()
    )


def make_artifact_evaluation_group_spread_summary_frame() -> pd.DataFrame:
    return _evaluation_group_spread_summary_frame_from_values(
        _artifacts_evaluation_values()
    )


def _evaluation_group_spread_summary_frame_from_values(
    values: dict[str, object],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [values["signal"]],
            "label": [values["label"]],
            "top_group_mean_label_value": [values["mean_label_value"]],
            "bottom_group_mean_label_value": [0.0],
            "top_bottom_label_value_spread_mean": [values["mean_label_value"]],
            "top_bottom_label_value_spread_hit_rate": [1.0],
            "group_monotonicity_score": [1.0],
            "group_monotonicity_score_reason": [None],
        }
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


def _artifacts_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "asset_id": ["asset_a"],
            "prediction_score": [0.1],
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
        "sample_weight_protocol": "equal_date_total_equal_sum_n_train",
        "scaler": None,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
        "coefficient_count": 2,
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


def _artifacts_model_metadata() -> dict[str, object]:
    return {
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2024-01-01",
        "train_end": "2024-01-01",
        "test_start": "2024-01-02",
        "test_end": None,
        "oos_start": "2024-01-02",
        "oos_end": "2024-01-02",
        "input_factor_columns": ["factor"],
        "input_feature_columns": ["feature"],
        "input_columns": ["factor", "feature"],
        "input_count": 2,
        "label_column": "label",
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": None,
        "max_train_rows": None,
        "sample_weight_protocol": "equal_date_total_equal_sum_n_train",
        "scaler": None,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
        "coefficient_count": 3,
        "feature_importance": {"factor": 0.1, "feature": 0.2},
    }


def _artifacts_prediction_metadata() -> dict[str, object]:
    return {
        "prediction_mode": "minimal_oos",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_dropna": 2,
        "prediction_rows": 1,
        "train_end": "2024-01-01",
        "test_start": "2024-01-02",
        "requested_test_end": None,
        "effective_test_end": "2024-01-02",
        "is_oos": True,
        "is_final_oos_window": True,
        "input_columns": ["factor", "feature"],
        "input_count": 2,
        "prediction_column": "prediction_score",
    }


def _price_volume_evaluation_values() -> dict[str, object]:
    return {
        "timestamp": pd.Timestamp("2014-01-03"),
        "signal": PRICE_VOLUME_SIGNAL_COLUMN,
        "label": FORWARD_RETURN_LABEL_COLUMN,
        "mean_label_value": 0.01,
    }


def _artifacts_evaluation_values() -> dict[str, object]:
    return {
        "timestamp": pd.Timestamp("2024-01-02"),
        "signal": "signal",
        "label": "label",
        "mean_label_value": 0.1,
    }

"""Output validation and destination checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.artifacts.layout import output_bundle_root
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.policy.share_safety import (
    sanitize_shared_object,
)
from quant_research_platform.storage.layout import StorageLayout

_EVALUATION_METRICS_COLUMNS = ("timestamp", "signal", "label", "ic", "rankic", "n_obs")
_EVALUATION_GROUPED_LABEL_VALUE_COLUMNS = (
    "timestamp",
    "signal",
    "label",
    "group",
    "mean_label_value",
    "n_obs",
)
_EVALUATION_IC_SUMMARY_COLUMNS = (
    "signal",
    "label",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_rate",
    "valid_ic_count",
    "mean_rankic",
    "std_rankic",
    "rankicir",
    "positive_rankic_rate",
    "valid_rankic_count",
)
_EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS = (
    "signal",
    "label",
    "top_group_mean_label_value",
    "bottom_group_mean_label_value",
    "top_bottom_label_value_spread_mean",
    "top_bottom_label_value_spread_hit_rate",
    "group_monotonicity_score",
)
_EVALUATION_ARTIFACTS = [
    "metrics.parquet",
    "grouped_label_values.parquet",
    "ic_summary.parquet",
    "group_spread_summary.parquet",
]
_PORTFOLIO_ARTIFACTS = {
    "positions": "positions.parquet",
    "turnover": "turnover.parquet",
    "cost_breakdown": "cost_breakdown.parquet",
    "portfolio_returns": "portfolio_returns.parquet",
    "cumulative_returns": "cumulative_returns.parquet",
    "performance_summary": "performance_summary.parquet",
    "exposure_summary": "exposure_summary.parquet",
    "concentration": "concentration.parquet",
    "period_breakdown": "period_breakdown.parquet",
    "diagnostics": "diagnostics.parquet",
}
_PORTFOLIO_POSITION_COLUMNS = ("timestamp", "asset", "signal", "position")
_PORTFOLIO_TURNOVER_COLUMNS = ("timestamp", "signal", "turnover")
_PORTFOLIO_COST_BREAKDOWN_COLUMNS = (
    "timestamp",
    "signal",
    "buy_turnover",
    "sell_turnover",
    "turnover",
    "buy_cost",
    "sell_cost",
    "slippage_cost",
    "spread_cost",
    "impact_cost",
    "total_cost",
    "cost_model",
)
_PORTFOLIO_RETURN_COLUMNS = (
    "timestamp",
    "signal",
    "return_source_column",
    "portfolio_mode",
    "long_return",
    "short_return",
    "gross_portfolio_return",
    "turnover",
    "cost",
    "net_portfolio_return",
    "long_count",
    "short_count",
)
_PORTFOLIO_CUMULATIVE_RETURN_COLUMNS = (
    "timestamp",
    "signal",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "cumulative_gross_return",
    "cumulative_net_return",
    "cumulative_cost",
    "high_water_mark",
    "drawdown",
)
_PORTFOLIO_PERFORMANCE_SUMMARY_COLUMNS = (
    "signal",
    "periods",
    "final_cumulative_gross_return",
    "final_cumulative_net_return",
    "mean_net_portfolio_return",
    "volatility_net_portfolio_return",
    "downside_volatility",
    "sharpe_like",
    "sortino_like",
    "calmar_like",
    "best_period_return",
    "worst_period_return",
    "skew",
    "kurtosis",
    "var_95",
    "cvar_95",
    "tail_ratio",
    "positive_return_rate",
    "negative_return_rate",
    "win_rate",
    "loss_rate",
    "profit_factor",
    "payoff_ratio",
    "average_win",
    "average_loss",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "max_drawdown",
    "drawdown_duration",
    "max_drawdown_start",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    "mean_turnover",
    "max_turnover",
    "total_cost",
    "mean_cost",
    "cost_to_gross_return",
    "cost_to_net_return",
)
_PORTFOLIO_EXPOSURE_SUMMARY_COLUMNS = (
    "signal",
    "mean_gross_exposure",
    "mean_net_exposure",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "mean_long_count",
    "mean_short_count",
    "mean_position_count",
)
_PORTFOLIO_CONCENTRATION_COLUMNS = (
    "signal",
    "periods",
    "mean_hhi",
    "max_hhi",
    "mean_top_weight",
    "max_top_weight",
    "mean_position_count",
)
_PORTFOLIO_PERIOD_BREAKDOWN_COLUMNS = (
    "signal",
    "year",
    "month",
    "periods",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "turnover",
    "hit_rate",
)
_PORTFOLIO_DIAGNOSTIC_COLUMNS = (
    "timestamp",
    "signal",
    "method",
    "universe_count",
    "eligible_count",
    "can_enter_count",
    "can_exit_count",
    "entry_blocked_count",
    "exit_blocked_count",
    "long_count",
    "short_count",
    "fallback_reason",
)
_MODEL_METADATA_BASE_KEYS = frozenset(
    {
        "contract_version",
        "model_algorithm",
        "sample_filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_label_dropna",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "oos_start",
        "oos_end",
        "input_factor_columns",
        "input_feature_columns",
        "input_columns",
        "input_count",
        "label_column",
        "hyperparameters",
        "random_seed",
        "max_train_rows",
        "purge_gap_steps",
        "dependency_status",
        "coefficient_count",
    }
)
_MODEL_METADATA_FEATURE_KEYS = frozenset(
    {
        "feature_importance",
        "feature_importance_status",
    }
)
_PREDICTION_METADATA_KEYS = frozenset(
    {
        "contract_version",
        "prediction_mode",
        "sample_filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_dropna",
        "prediction_rows",
        "train_end",
        "test_start",
        "requested_test_end",
        "effective_test_end",
        "is_oos",
        "is_final_oos_window",
        "input_columns",
        "input_count",
        "prediction_column",
    }
)



def _portfolio_required_columns() -> dict[str, tuple[str, ...]]:
    return {
        "positions": _PORTFOLIO_POSITION_COLUMNS,
        "turnover": _PORTFOLIO_TURNOVER_COLUMNS,
        "cost_breakdown": _PORTFOLIO_COST_BREAKDOWN_COLUMNS,
        "portfolio_returns": _PORTFOLIO_RETURN_COLUMNS,
        "cumulative_returns": _PORTFOLIO_CUMULATIVE_RETURN_COLUMNS,
        "performance_summary": _PORTFOLIO_PERFORMANCE_SUMMARY_COLUMNS,
        "exposure_summary": _PORTFOLIO_EXPOSURE_SUMMARY_COLUMNS,
        "concentration": _PORTFOLIO_CONCENTRATION_COLUMNS,
        "period_breakdown": _PORTFOLIO_PERIOD_BREAKDOWN_COLUMNS,
        "diagnostics": _PORTFOLIO_DIAGNOSTIC_COLUMNS,
    }


def _validate_portfolio_artifact_semantics(
    artifacts: Mapping[str, pd.DataFrame],
    spec: PortfolioSpec,
) -> None:
    for artifact_name, frame in artifacts.items():
        _validate_portfolio_signal_column(
            frame,
            artifact_name=artifact_name,
            spec=spec,
        )
    portfolio_returns = artifacts["portfolio_returns"]
    return_sources = {
        str(value)
        for value in (
            portfolio_returns["return_source_column"].dropna().unique().tolist()
        )
    }
    if return_sources and return_sources != {spec.return_source_column}:
        raise ValueError("portfolio_return_source_column_mismatch")


def _validate_portfolio_signal_column(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    spec: PortfolioSpec,
) -> None:
    signal_ids = {signal.signal_id for signal in spec.signals}
    if "signal" not in frame.columns:
        return
    artifact_signals = {
        str(value) for value in frame["signal"].dropna().unique().tolist()
    }
    if not artifact_signals.issubset(signal_ids):
        raise ValueError("portfolio_artifact_signal_mismatch: " + artifact_name)


def _required_frame_metadata(
    frame: pd.DataFrame,
    *,
    key: str,
    error_name: str,
) -> dict[str, object]:
    if key not in frame.attrs:
        raise ValueError(error_name)
    return _sanitize_required_metadata(frame.attrs[key], error_name=error_name)


def _sanitize_required_metadata(
    value: object,
    *,
    error_name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(error_name)
    sanitized = sanitize_shared_object(dict(value))
    if not isinstance(sanitized, dict) or not sanitized:
        raise ValueError(error_name)
    return sanitized


def _validate_model_metadata_shape(metadata: dict[str, object]) -> dict[str, object]:
    metadata_keys = set(metadata)
    feature_keys = metadata_keys.intersection(_MODEL_METADATA_FEATURE_KEYS)
    if (
        not _MODEL_METADATA_BASE_KEYS.issubset(metadata_keys)
        or len(feature_keys) != 1
        or metadata_keys.difference(_MODEL_METADATA_BASE_KEYS).difference(
            _MODEL_METADATA_FEATURE_KEYS
        )
    ):
        raise ValueError("invalid_model_metadata")
    if metadata["contract_version"] != "0.1":
        raise ValueError("invalid_model_metadata")
    _require_non_negative_metadata_int(metadata, "rows_input", "invalid_model_metadata")
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_model_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_input_label_dropna",
        "invalid_model_metadata",
    )
    input_factor_columns = _require_metadata_string_list(
        metadata,
        "input_factor_columns",
        "invalid_model_metadata",
        allow_empty=True,
    )
    input_feature_columns = _require_metadata_string_list(
        metadata,
        "input_feature_columns",
        "invalid_model_metadata",
        allow_empty=True,
    )
    input_columns = _require_metadata_string_list(
        metadata,
        "input_columns",
        "invalid_model_metadata",
    )
    input_count = _require_non_negative_metadata_int(
        metadata,
        "input_count",
        "invalid_model_metadata",
    )
    coefficient_count = _require_non_negative_metadata_int(
        metadata,
        "coefficient_count",
        "invalid_model_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "purge_gap_steps",
        "invalid_model_metadata",
    )
    if metadata["max_train_rows"] is not None:
        _require_non_negative_metadata_int(
            metadata,
            "max_train_rows",
            "invalid_model_metadata",
        )
    if metadata["random_seed"] is not None and not isinstance(
        metadata["random_seed"],
        int,
    ):
        raise ValueError("invalid_model_metadata")
    if not isinstance(metadata["hyperparameters"], dict):
        raise ValueError("invalid_model_metadata")
    if not isinstance(metadata["dependency_status"], dict):
        raise ValueError("invalid_model_metadata")
    if metadata["dependency_status"].get("status") != "available":
        raise ValueError("invalid_model_metadata")
    if "feature_importance" in metadata and not isinstance(
        metadata["feature_importance"],
        dict,
    ):
        raise ValueError("invalid_model_metadata")
    if "feature_importance_status" in metadata:
        _require_metadata_string(
            metadata,
            "feature_importance_status",
            "invalid_model_metadata",
        )
    if (
        input_columns != [*input_factor_columns, *input_feature_columns]
        or input_count != len(input_columns)
        or coefficient_count < 1
    ):
        raise ValueError("invalid_model_metadata")
    _require_metadata_string(metadata, "model_algorithm", "invalid_model_metadata")
    _require_metadata_string(metadata, "sample_filter_mode", "invalid_model_metadata")
    _require_metadata_string(metadata, "filter_definition", "invalid_model_metadata")
    _require_metadata_string(metadata, "label_column", "invalid_model_metadata")
    _require_metadata_string(metadata, "train_start", "invalid_model_metadata")
    _require_metadata_string(metadata, "train_end", "invalid_model_metadata")
    _require_optional_metadata_string(
        metadata,
        "test_start",
        "invalid_model_metadata",
    )
    _require_optional_metadata_string(metadata, "test_end", "invalid_model_metadata")
    _require_optional_metadata_string(metadata, "oos_start", "invalid_model_metadata")
    _require_optional_metadata_string(metadata, "oos_end", "invalid_model_metadata")
    _require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_model_metadata",
        allow_empty=True,
    )
    return metadata


def _validate_prediction_metadata_shape(
    metadata: dict[str, object],
) -> dict[str, object]:
    if set(metadata) != _PREDICTION_METADATA_KEYS:
        raise ValueError("invalid_prediction_metadata")
    if metadata["contract_version"] != "0.1":
        raise ValueError("invalid_prediction_metadata")
    _require_non_negative_metadata_int(
        metadata,
        "rows_input",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_input_dropna",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "prediction_rows",
        "invalid_prediction_metadata",
    )
    input_columns = _require_metadata_string_list(
        metadata,
        "input_columns",
        "invalid_prediction_metadata",
    )
    input_count = _require_non_negative_metadata_int(
        metadata,
        "input_count",
        "invalid_prediction_metadata",
    )
    if input_count != len(input_columns):
        raise ValueError("invalid_prediction_metadata")
    _require_metadata_string(
        metadata,
        "prediction_mode",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "sample_filter_mode",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "filter_definition",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(metadata, "test_start", "invalid_prediction_metadata")
    _require_metadata_string(metadata, "train_end", "invalid_prediction_metadata")
    if metadata["requested_test_end"] is not None:
        _require_metadata_string(
            metadata,
            "requested_test_end",
            "invalid_prediction_metadata",
        )
    _require_metadata_string(
        metadata,
        "effective_test_end",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "prediction_column",
        "invalid_prediction_metadata",
    )
    _require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_prediction_metadata",
        allow_empty=True,
    )
    if not isinstance(metadata["is_oos"], bool) or not isinstance(
        metadata["is_final_oos_window"],
        bool,
    ):
        raise ValueError("invalid_prediction_metadata")
    return metadata


def _validate_model_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
) -> None:
    input_count = int(metadata["input_count"])
    input_columns = metadata["input_columns"]
    if not isinstance(input_columns, list) or input_count != len(input_columns):
        raise ValueError("model_metadata_inconsistent")
    _validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_input_label_dropna"),
        "model_metadata_inconsistent",
    )
    _validate_filter_metadata(
        mode=str(metadata["sample_filter_mode"]),
        columns=metadata["filter_columns_used"],
        definition=str(metadata["filter_definition"]),
        error_name="model_metadata_inconsistent",
    )
    train_start = pd.Timestamp(metadata["train_start"])
    train_end = pd.Timestamp(metadata["train_end"])
    if train_start > train_end or train_end > pd.Timestamp(spec.train_end):
        raise ValueError("model_metadata_inconsistent")
    if (
        metadata["test_start"] != spec.test_start
        or metadata["test_end"] != spec.test_end
        or metadata["oos_start"] != spec.test_start
    ):
        raise ValueError("model_metadata_inconsistent")
    if int(metadata["coefficient_count"]) <= 0:
        raise ValueError("model_metadata_inconsistent")


def _validate_prediction_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
    frame: pd.DataFrame,
) -> None:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("prediction_metadata_inconsistent")
    effective_test_end = pd.Timestamp(timestamps.max()).date().isoformat()
    _validate_prediction_metadata_consistency_from_summary(
        metadata,
        spec=spec,
        row_count=len(frame),
        effective_test_end=effective_test_end,
    )


def _validate_prediction_metadata_consistency_from_summary(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
    row_count: int,
    effective_test_end: str,
) -> None:
    expected = {
        "prediction_rows": row_count,
        "prediction_column": spec.prediction_column,
        "prediction_mode": spec.prediction_mode,
        "sample_filter_mode": spec.sample_filter_mode,
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "requested_test_end": spec.test_end,
        "is_oos": spec.is_oos,
        "is_final_oos_window": spec.is_final_oos_window,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError("prediction_metadata_inconsistent")
    _validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_input_dropna"),
        "prediction_metadata_inconsistent",
    )
    if int(metadata["prediction_rows"]) > int(metadata["rows_after_input_dropna"]):
        raise ValueError("prediction_metadata_inconsistent")
    _validate_filter_metadata(
        mode=str(metadata["sample_filter_mode"]),
        columns=metadata["filter_columns_used"],
        definition=str(metadata["filter_definition"]),
        error_name="prediction_metadata_inconsistent",
    )
    if metadata["effective_test_end"] != effective_test_end:
        raise ValueError("prediction_metadata_inconsistent")
    requested_test_end = metadata["requested_test_end"]
    if requested_test_end is not None and pd.Timestamp(
        effective_test_end
    ) > pd.Timestamp(requested_test_end):
        raise ValueError("prediction_metadata_inconsistent")


def _validate_model_prediction_metadata_consistency(
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> None:
    if model_metadata["input_columns"] != prediction_metadata["input_columns"]:
        raise ValueError("prediction_metadata_inconsistent")
    if model_metadata["input_count"] != prediction_metadata["input_count"]:
        raise ValueError("prediction_metadata_inconsistent")
    if model_metadata["oos_end"] != prediction_metadata["effective_test_end"]:
        raise ValueError("prediction_metadata_inconsistent")


def _model_metadata_with_effective_prediction_end(
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        **model_metadata,
        "oos_end": prediction_metadata["effective_test_end"],
    }


def _validate_evaluation_diagnostics_consistency(
    *,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
) -> None:
    diagnostics_payload = asdict(evaluation_diagnostics)
    rows_input = int(diagnostics_payload["rows_input"])
    rows_after_filter_mode = int(diagnostics_payload["rows_after_filter_mode"])
    rows_after_signal_label_dropna = int(
        diagnostics_payload["rows_after_signal_label_dropna"]
    )
    max_signal_label_rows = (
        rows_after_filter_mode * len(spec.signals) * len(spec.label_columns)
    )
    if not (
        0 <= rows_after_filter_mode <= rows_input
        and 0 <= rows_after_signal_label_dropna <= max_signal_label_rows
    ):
        raise ValueError("evaluation_diagnostics_inconsistent")
    _validate_filter_metadata(
        mode=evaluation_diagnostics.filter_mode,
        columns=list(evaluation_diagnostics.filter_columns_used),
        definition=evaluation_diagnostics.filter_definition,
        error_name="evaluation_diagnostics_inconsistent",
    )
    timestamps = pd.to_datetime(metrics["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("evaluation_diagnostics_inconsistent")
    expected_timestamp_count = int(timestamps.nunique()) if not timestamps.empty else 0
    expected_min_timestamp = (
        pd.Timestamp(timestamps.min()).date().isoformat()
        if not timestamps.empty
        else None
    )
    expected_max_timestamp = (
        pd.Timestamp(timestamps.max()).date().isoformat()
        if not timestamps.empty
        else None
    )
    n_obs = pd.to_numeric(metrics["n_obs"], errors="coerce")
    if bool(n_obs.isna().any()):
        raise ValueError("evaluation_diagnostics_inconsistent")
    expected_skipped_timestamp_count = int(
        metrics.loc[n_obs.lt(2), "timestamp"].nunique()
    )
    expected_rows_after_signal_label_dropna = int(n_obs.sum())
    expected = {
        "filter_mode": spec.filter_mode,
        "filter_columns_used": tuple(required_filter_columns(spec.filter_mode)),
        "filter_definition": filter_definition(spec.filter_mode),
        "rows_after_signal_label_dropna": expected_rows_after_signal_label_dropna,
        "metric_rows": len(metrics),
        "grouped_label_value_rows": len(grouped_label_values),
        "ic_summary_rows": len(ic_summary),
        "group_spread_summary_rows": len(group_spread_summary),
        "timestamp_count": expected_timestamp_count,
        "min_timestamp": expected_min_timestamp,
        "max_timestamp": expected_max_timestamp,
        "skipped_timestamp_count": expected_skipped_timestamp_count,
    }
    for key, value in expected.items():
        if getattr(evaluation_diagnostics, key) != value:
            raise ValueError("evaluation_diagnostics_inconsistent")


def _validate_evaluation_artifact_structure(
    *,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    spec: EvaluationSpec,
    n_groups: int,
) -> None:
    _reject_duplicate_keys(metrics, ("timestamp", "signal", "label"), "metrics")
    _reject_duplicate_keys(
        grouped_label_values,
        ("timestamp", "signal", "label", "group"),
        "grouped_label_values",
    )
    _reject_duplicate_keys(ic_summary, ("signal", "label"), "ic_summary")
    _reject_duplicate_keys(
        group_spread_summary,
        ("signal", "label"),
        "group_spread_summary",
    )
    for frame in (metrics, grouped_label_values, ic_summary, group_spread_summary):
        _validate_signal_label_membership(frame, spec=spec)
    groups = pd.to_numeric(grouped_label_values["group"], errors="coerce")
    if bool(groups.isna().any()):
        raise ValueError("evaluation_group_out_of_range")
    if (
        bool((groups % 1 != 0).any())
        or bool(groups.lt(1).any())
        or bool(groups.gt(n_groups).any())
    ):
        raise ValueError("evaluation_group_out_of_range")


def _reject_duplicate_keys(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    artifact_name: str,
) -> None:
    if bool(frame.duplicated(list(keys)).any()):
        raise ValueError(f"duplicate_evaluation_{artifact_name}_keys")


def _validate_signal_label_membership(
    frame: pd.DataFrame,
    *,
    spec: EvaluationSpec,
) -> None:
    signal_ids = {signal.signal_id for signal in spec.signals}
    if not set(frame["signal"]).issubset(signal_ids) or not set(
        frame["label"]
    ).issubset(set(spec.label_columns)):
        raise ValueError("evaluation_artifact_signal_label_mismatch")


def _validate_metadata_row_count_chain(
    metadata: dict[str, object],
    keys: tuple[str, str, str],
    error_name: str,
) -> None:
    first, second, third = (int(metadata[key]) for key in keys)
    if not (0 <= third <= second <= first):
        raise ValueError(error_name)


def _validate_filter_metadata(
    *,
    mode: str,
    columns: object,
    definition: str,
    error_name: str,
) -> None:
    if list(columns) != list(required_filter_columns(mode)):
        raise ValueError(error_name)
    if definition != filter_definition(mode):
        raise ValueError(error_name)


def _require_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str:
    value = metadata[key]
    if not isinstance(value, str) or not value:
        raise ValueError(error_name)
    return value


def _require_optional_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str | None:
    value = metadata[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(error_name)
    return value


def _require_metadata_string_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = metadata[key]
    if not isinstance(value, list):
        raise ValueError(error_name)
    if (not allow_empty and not value) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(error_name)
    return value


def _require_non_negative_metadata_int(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> int:
    value = metadata[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(error_name)
    return value


def _validate_output_frame(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    error_name: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_{error_name}_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_{error_name}_columns: {', '.join(extra)}")


def _derive_output_bundle_root(
    *,
    output_spec: OutputSpec,
    layout: StorageLayout,
    expected_definition_id: str,
) -> Path:
    try:
        definition = output_catalog().get(output_spec.output_definition_id)
    except KeyError as exc:
        raise ValueError("output_spec_mismatch") from exc
    if definition.output_definition_id != expected_definition_id:
        raise ValueError("output_spec_mismatch")
    if definition.run_type != output_spec.run_type:
        raise ValueError("output_spec_mismatch")
    if definition.scope != output_spec.scope:
        raise ValueError("output_spec_mismatch")
    if output_spec.scope != OutputScope.LOCAL_ONLY:
        raise ValueError("output_scope_must_be_local_only")
    output_bundle = output_bundle_root(layout.project_root, output_spec.output_id)
    if any(
        f"data/{layer}" in output_bundle.as_posix()
        for layer in ("raw", "interim", "processed", "research")
    ):
        raise ValueError("output_bundle_must_not_be_dataset_root")
    return output_bundle

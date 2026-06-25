"""Local output writers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
    sanitize_shared_text,
)
from quant_research_platform.storage.atomic import (
    atomic_write_csv,
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
    reject_if_exists,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest

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


@dataclass(frozen=True)
class _PredictionOutputWriteSummary:
    row_count: int
    effective_test_end: str


def write_stage_profile_outputs(
    frame: pd.DataFrame,
    *,
    spec: OutputSpec,
    markdown_text: str,
    layout: StorageLayout,
) -> dict[str, Path]:
    """Write local stage-profile outputs."""

    destination = _derive_output_bundle_root(
        output_spec=spec,
        layout=layout,
        expected_definition_id="stage_profile_diagnostic",
    )

    def _write(temp_root: Path) -> None:
        safe_rows = sanitize_shared_object(frame.to_dict(orient="records"))
        safe_frame = pd.DataFrame.from_records(safe_rows)
        atomic_write_csv(safe_frame, temp_root / "stage_profile.csv", index=False)
        atomic_write_json(
            temp_root / "stage_profile.json",
            {
                "schema_id": "qrp-stage-profile",
                "rows": safe_rows,
            },
        )
        atomic_write_text(
            temp_root / "stage_profile.md",
            sanitize_shared_text(markdown_text),
            encoding="utf-8",
        )
        write_manifest(
            temp_root / "manifest.json",
            _stage_profile_output_manifest_payload(spec=spec),
        )

    atomic_write_directory(destination, _write)
    csv_path = destination / "stage_profile.csv"
    json_path = destination / "stage_profile.json"
    markdown_path = destination / "stage_profile.md"
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def write_prediction_outputs(
    frame: pd.DataFrame | None = None,
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    output_chunks: Iterable[pd.DataFrame] | None = None,
    model_metadata: Mapping[str, object] | None = None,
    prediction_metadata_factory: Callable[[], Mapping[str, object]] | None = None,
) -> Path:
    """Write local prediction outputs."""

    if output_chunks is not None:
        if model_metadata is None or prediction_metadata_factory is None:
            raise ValueError("prediction_chunk_metadata_required")
        return _write_prediction_output_chunks(
            output_chunks,
            spec=spec,
            output_spec=output_spec,
            layout=layout,
            model_metadata=model_metadata,
            prediction_metadata_factory=prediction_metadata_factory,
        )
    if frame is None:
        raise ValueError("prediction_output_frame_required")

    run_root = _derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="prediction_outputs",
    )
    required = (*spec.primary_key, spec.prediction_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_output_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_prediction_output_columns: {', '.join(extra)}")
    if bool(frame.duplicated(list(spec.primary_key)).any()):
        raise ValueError("duplicate_prediction_keys")
    model_metadata = _validate_model_metadata_shape(
        _required_frame_metadata(
            frame,
            key="model_metadata",
            error_name="missing_model_metadata",
        )
    )
    prediction_metadata = _validate_prediction_metadata_shape(
        _required_frame_metadata(
            frame,
            key="prediction_metadata",
            error_name="missing_prediction_metadata",
        )
    )
    _validate_model_metadata_consistency(model_metadata, spec=spec)
    _validate_prediction_metadata_consistency(
        prediction_metadata,
        spec=spec,
        frame=frame,
    )
    model_metadata = _model_metadata_with_effective_prediction_end(
        model_metadata,
        prediction_metadata,
    )
    _validate_model_metadata_consistency(model_metadata, spec=spec)
    _validate_model_prediction_metadata_consistency(
        model_metadata,
        prediction_metadata,
    )

    def _write(temp_root: Path) -> None:
        _write_parquet(
            frame.loc[:, list(required)],
            temp_root / "prediction.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            _output_manifest_payload(
                output_spec=output_spec,
                metadata=_prediction_output_metadata(
                    spec=spec,
                    model_metadata=model_metadata,
                    prediction_metadata=prediction_metadata,
                ),
                files=("prediction.parquet",),
            ),
        )

    atomic_write_directory(run_root, _write)
    return run_root


def _write_prediction_output_chunks(
    frames: Iterable[pd.DataFrame],
    *,
    spec: PredictionSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
    model_metadata: Mapping[str, object],
    prediction_metadata_factory: Callable[[], Mapping[str, object]],
) -> Path:
    run_root = _derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="prediction_outputs",
    )
    required = (*spec.primary_key, spec.prediction_column)
    model_metadata_payload = _validate_model_metadata_shape(
        _sanitize_required_metadata(
            model_metadata,
            error_name="missing_model_metadata",
        )
    )
    _validate_model_metadata_consistency(model_metadata_payload, spec=spec)

    def _write(temp_root: Path) -> None:
        write_summary = _write_parquet_chunks(
            frames,
            temp_root / "prediction.parquet",
            required=required,
            key_columns=tuple(spec.primary_key),
            compression="zstd",
        )
        prediction_metadata = _validate_prediction_metadata_shape(
            _sanitize_required_metadata(
                prediction_metadata_factory(),
                error_name="missing_prediction_metadata",
            )
        )
        _validate_prediction_metadata_consistency_from_summary(
            prediction_metadata,
            spec=spec,
            row_count=write_summary.row_count,
            effective_test_end=write_summary.effective_test_end,
        )
        effective_model_metadata = _model_metadata_with_effective_prediction_end(
            model_metadata_payload,
            prediction_metadata,
        )
        _validate_model_metadata_consistency(effective_model_metadata, spec=spec)
        _validate_model_prediction_metadata_consistency(
            effective_model_metadata,
            prediction_metadata,
        )
        write_manifest(
            temp_root / "manifest.json",
            _output_manifest_payload(
                output_spec=output_spec,
                metadata=_prediction_output_metadata(
                    spec=spec,
                    model_metadata=effective_model_metadata,
                    prediction_metadata=prediction_metadata,
                ),
                files=("prediction.parquet",),
            ),
        )

    atomic_write_directory(run_root, _write)
    return run_root


def write_evaluation_outputs(
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    *,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    """Write local evaluation outputs under local output root."""

    run_root = _derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="evaluation_outputs",
    )

    _validate_output_frame(
        metrics,
        required=_EVALUATION_METRICS_COLUMNS,
        error_name="evaluation_metrics",
    )
    _validate_output_frame(
        grouped_label_values,
        required=_EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
        error_name="evaluation_grouped_label_values",
    )
    _validate_output_frame(
        ic_summary,
        required=_EVALUATION_IC_SUMMARY_COLUMNS,
        error_name="evaluation_ic_summary",
    )
    _validate_output_frame(
        group_spread_summary,
        required=_EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
        error_name="evaluation_group_spread_summary",
    )
    _validate_evaluation_artifact_structure(
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        spec=spec,
        n_groups=evaluation_diagnostics.n_groups,
    )
    _validate_evaluation_diagnostics_consistency(
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        evaluation_diagnostics=evaluation_diagnostics,
        spec=spec,
    )

    def _write(temp_root: Path) -> None:
        _write_parquet(
            metrics,
            temp_root / "metrics.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _write_parquet(
            grouped_label_values,
            temp_root / "grouped_label_values.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _write_parquet(
            ic_summary,
            temp_root / "ic_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _write_parquet(
            group_spread_summary,
            temp_root / "group_spread_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            _output_manifest_payload(
                output_spec=output_spec,
                metadata=_evaluation_output_metadata(
                    spec=spec,
                    evaluation_diagnostics=evaluation_diagnostics,
                ),
                files=_EVALUATION_ARTIFACTS,
            ),
        )

    atomic_write_directory(run_root, _write)
    return run_root


def write_portfolio_outputs(
    *,
    output_chunks: Iterable[Mapping[str, pd.DataFrame]],
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]],
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    """Write local portfolio outputs."""

    return _write_portfolio_output_chunks(
        output_chunks,
        summary_factory=summary_factory,
        spec=spec,
        output_spec=output_spec,
        layout=layout,
    )


def _write_portfolio_output_chunks(
    output_chunks: Iterable[Mapping[str, pd.DataFrame]],
    *,
    summary_factory: Callable[[], Mapping[str, pd.DataFrame]],
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    """Write local portfolio row outputs from bounded chunks."""

    run_root = _derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="portfolio_outputs",
    )
    required_columns = _portfolio_required_columns()
    row_artifacts = (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
    )
    summary_names = tuple(
        name for name in _PORTFOLIO_ARTIFACTS if name not in row_artifacts
    )

    def _write(temp_root: Path) -> None:
        row_counts = _write_portfolio_row_artifact_chunks(
            output_chunks,
            temp_root=temp_root,
            row_artifacts=row_artifacts,
            required_columns=required_columns,
            spec=spec,
        )
        summary_outputs = summary_factory()
        for artifact_name in summary_names:
            _validate_output_frame(
                summary_outputs[artifact_name],
                required=required_columns[artifact_name],
                error_name=f"portfolio_{artifact_name}",
            )
            _validate_portfolio_signal_column(
                summary_outputs[artifact_name],
                artifact_name=artifact_name,
                spec=spec,
            )
            _write_parquet(
                summary_outputs[artifact_name],
                temp_root / _PORTFOLIO_ARTIFACTS[artifact_name],
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            row_counts[artifact_name] = int(len(summary_outputs[artifact_name]))
        write_manifest(
            temp_root / "manifest.json",
            _output_manifest_payload(
                output_spec=output_spec,
                metadata=_portfolio_output_metadata_from_row_counts(
                    spec=spec,
                    row_counts=row_counts,
                ),
                files=tuple(_PORTFOLIO_ARTIFACTS.values()),
            ),
        )

    atomic_write_directory(run_root, _write)
    return run_root


def _stage_profile_output_manifest_payload(
    *,
    spec: OutputSpec,
) -> dict[str, object]:
    return _output_manifest_payload(
        output_spec=spec,
        metadata={
            "market_id": str(spec.market_id),
            "run_type": spec.run_type,
            "scope": spec.scope,
        },
        files=("stage_profile.csv", "stage_profile.json", "stage_profile.md"),
    )


def _output_manifest_payload(
    *,
    output_spec: OutputSpec,
    metadata: Mapping[str, object],
    files: Iterable[str],
) -> dict[str, object]:
    return {
        "schema_id": "qrp-output-manifest",
        "output_id": output_spec.output_id,
        "output_definition_id": output_spec.output_definition_id,
        "run_id": output_spec.run_id,
        "metadata": _without_json_version_fields(dict(metadata)),
        "files": list(files),
    }


def _without_json_version_fields(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_json_version_fields(item)
            for key, item in value.items()
            if key not in {"schema_version", "contract_version"}
        }
    if isinstance(value, list | tuple):
        return [_without_json_version_fields(item) for item in value]
    return value


def _prediction_output_metadata(
    *,
    spec: PredictionSpec,
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "prediction_id": spec.prediction_id,
        "model_id": spec.model_id,
        "research_dataset_id": spec.research_dataset_id,
        "prediction_mode": spec.prediction_mode,
        "is_oos": spec.is_oos,
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "test_end": spec.test_end,
        "is_final_oos_window": spec.is_final_oos_window,
        "prediction_column": spec.prediction_column,
        "primary_key": list(spec.primary_key),
        "sample_filter_mode": spec.sample_filter_mode,
        "model_metadata": model_metadata,
        "prediction_metadata": prediction_metadata,
    }


def _evaluation_output_metadata(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "run_type": spec.run_type,
        "market_id": str(spec.market_id),
        "research_dataset_id": spec.research_dataset_id,
        "signals": [asdict(signal) for signal in spec.signals],
        "label_columns": list(spec.label_columns),
        "filter_mode": spec.filter_mode,
        "evaluation_diagnostics": sanitize_shared_object(
            asdict(evaluation_diagnostics)
        ),
    }


def _portfolio_output_metadata(
    *,
    spec: PortfolioSpec,
    artifacts: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    return {
        "portfolio_id": spec.portfolio_id,
        "market_id": str(spec.market_id),
        "research_dataset_id": spec.research_dataset_id,
        "run_type": spec.run_type,
        "signals": [asdict(signal) for signal in spec.signals],
        "return_source_column": spec.return_source_column,
        "rebalance_frequency": spec.rebalance_frequency,
        "execution_lag_steps": spec.execution_lag_steps,
        "weight_config": asdict(spec.weight_config),
        "tradability_config": asdict(spec.tradability_config),
        "cost_config": asdict(spec.cost_config),
        "row_counts": {
            artifact_name: int(len(frame)) for artifact_name, frame in artifacts.items()
        },
    }


def _portfolio_output_metadata_from_row_counts(
    *,
    spec: PortfolioSpec,
    row_counts: Mapping[str, int],
) -> dict[str, object]:
    payload = _portfolio_output_metadata(
        spec=spec,
        artifacts={
            artifact_name: pd.DataFrame(index=range(int(row_counts[artifact_name])))
            for artifact_name in _PORTFOLIO_ARTIFACTS
        },
    )
    payload["row_counts"] = {
        artifact_name: int(row_counts[artifact_name])
        for artifact_name in _PORTFOLIO_ARTIFACTS
    }
    return payload


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


def _write_portfolio_row_artifact_chunks(
    artifact_chunks: Iterable[Mapping[str, pd.DataFrame]],
    *,
    temp_root: Path,
    row_artifacts: tuple[str, ...],
    required_columns: Mapping[str, tuple[str, ...]],
    spec: PortfolioSpec,
) -> dict[str, int]:
    writers: dict[str, pq.ParquetWriter] = {}
    temp_paths = {
        artifact_name: temp_root / _PORTFOLIO_ARTIFACTS[artifact_name]
        for artifact_name in row_artifacts
    }
    row_counts = {artifact_name: 0 for artifact_name in _PORTFOLIO_ARTIFACTS}
    try:
        for artifact_chunk in artifact_chunks:
            for artifact_name in row_artifacts:
                frame = artifact_chunk[artifact_name]
                selected = _portfolio_output_chunk_frame(
                    frame,
                    artifact_name=artifact_name,
                    required=required_columns[artifact_name],
                    spec=spec,
                )
                if selected.empty:
                    continue
                table = pa.Table.from_pandas(selected, preserve_index=False)
                writer = writers.get(artifact_name)
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_paths[artifact_name],
                        table.schema,
                        compression="zstd",
                    )
                    writers[artifact_name] = writer
                writer.write_table(table)
                row_counts[artifact_name] += int(len(selected))
        for writer in writers.values():
            writer.close()
        writers.clear()
        for artifact_name in row_artifacts:
            if row_counts[artifact_name] == 0:
                _write_parquet(
                    pd.DataFrame(columns=list(required_columns[artifact_name])),
                    temp_paths[artifact_name],
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )
    except Exception:
        for writer in writers.values():
            writer.close()
        raise
    return row_counts


def _portfolio_output_chunk_frame(
    frame: pd.DataFrame,
    *,
    artifact_name: str,
    required: tuple[str, ...],
    spec: PortfolioSpec,
) -> pd.DataFrame:
    _validate_output_frame(
        frame,
        required=required,
        error_name=f"portfolio_{artifact_name}",
    )
    if "signal" in frame.columns:
        signal_ids = {signal.signal_id for signal in spec.signals}
        artifact_signals = {
            str(value) for value in frame["signal"].dropna().unique().tolist()
        }
        if not artifact_signals.issubset(signal_ids):
            raise ValueError("portfolio_artifact_signal_mismatch: " + artifact_name)
    if artifact_name == "portfolio_returns":
        return_sources = {
            str(value)
            for value in frame["return_source_column"].dropna().unique().tolist()
        }
        if return_sources and return_sources != {spec.return_source_column}:
            raise ValueError("portfolio_return_source_column_mismatch")
    return frame.loc[:, list(required)]


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
    run_root = output_bundle_root(layout.project_root, output_spec.output_id)
    if any(
        f"data/{layer}" in run_root.as_posix()
        for layer in ("raw", "interim", "processed", "research")
    ):
        raise ValueError("output_must_use_artifact_root")
    return run_root


def _write_parquet_chunks(
    frames: Iterable[pd.DataFrame],
    path: Path,
    *,
    required: tuple[str, ...],
    key_columns: tuple[str, ...],
    compression: str,
) -> _PredictionOutputWriteSummary:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".{final_path.name}.tmp-{uuid4().hex[:12]}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    seen_keys: set[tuple[object, ...]] = set()
    row_count = 0
    effective_test_end: str | None = None
    try:
        for frame in frames:
            selected = _prediction_output_chunk_frame(
                frame,
                required=required,
                key_columns=key_columns,
                seen_keys=seen_keys,
            )
            if selected.empty:
                continue
            timestamps = pd.to_datetime(selected["timestamp"], errors="coerce")
            if bool(timestamps.isna().any()):
                raise ValueError("prediction_metadata_inconsistent")
            current_end = pd.Timestamp(timestamps.max()).date().isoformat()
            if effective_test_end is None or pd.Timestamp(current_end) > pd.Timestamp(
                effective_test_end
            ):
                effective_test_end = current_end
            table = pa.Table.from_pandas(selected, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    table.schema,
                    compression=compression,
                )
            writer.write_table(table)
            row_count += len(selected)
        if writer is None or row_count == 0 or effective_test_end is None:
            raise ValueError("empty_prediction_window")
        writer.close()
        writer = None
        reject_if_exists(final_path)
        _rename_with_transient_permission_retry(temp_path, final_path)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()
        raise
    return _PredictionOutputWriteSummary(
        row_count=row_count,
        effective_test_end=effective_test_end,
    )


def _prediction_output_chunk_frame(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    key_columns: tuple[str, ...],
    seen_keys: set[tuple[object, ...]],
) -> pd.DataFrame:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_output_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_prediction_output_columns: {', '.join(extra)}")
    selected = frame.loc[:, list(required)]
    if bool(selected.duplicated(list(key_columns)).any()):
        raise ValueError("duplicate_prediction_keys")
    for key in selected.loc[:, list(key_columns)].itertuples(index=False, name=None):
        normalized_key = tuple(key)
        if normalized_key in seen_keys:
            raise ValueError("duplicate_prediction_keys")
        seen_keys.add(normalized_key)
    return selected


def _write_parquet(
    frame: pd.DataFrame,
    path: Path,
    **kwargs: object,
) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".{final_path.name}.tmp-{uuid4().hex[:12]}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(temp_path, **kwargs)
        reject_if_exists(final_path)
        _rename_with_transient_permission_retry(temp_path, final_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return final_path


def _rename_with_transient_permission_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 3,
    delay_seconds: float = 0.10,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)


__all__ = [
    "write_evaluation_outputs",
    "write_prediction_outputs",
    "write_portfolio_outputs",
    "write_stage_profile_outputs",
]

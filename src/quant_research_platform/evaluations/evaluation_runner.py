"""核心 factor 和 prediction signal evaluation runner。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.artifact_writer import (
    write_evaluation_set_artifacts,
)
from quant_research_platform.evaluations.diagnostics import (
    DiagnosticRunner,
)
from quant_research_platform.evaluations.diagnostics.inputs import DiagnosticInputBundle
from quant_research_platform.evaluations.evaluation_payload import EvaluationRunPayload
from quant_research_platform.evaluations.evaluation_result import EvaluationSetResult
from quant_research_platform.evaluations.evaluation_spec import EvaluationSpec
from quant_research_platform.evaluations.neutralization import apply_neutralization
from quant_research_platform.evaluations.panel_validation import (
    PanelSpec,
    validate_evaluation_set_return_source_column,
    validate_evaluation_spec_columns,
    validate_panel,
)
from quant_research_platform.evaluations.period_breakdowns import (
    PERIOD_BREAKDOWN_POLICY,
    build_time_breakdown_metrics,
)
from quant_research_platform.kernels.dataframe.cross_section import (
    evaluate_ic_and_grouped_returns_from_validated_panel,
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.kernels.dataframe.portfolio import (
    _simulate_weighted_long_short_returns_and_cost_from_validated_panel,
    calculate_cumulative_returns,
    summarize_exposure,
    summarize_performance,
)
from quant_research_platform.kernels.kernel_registry import get_kernel_descriptor
from quant_research_platform.kernels.tensor.conversion import (
    panel_to_tensor,
    panel_to_tensor_batch,
)
from quant_research_platform.kernels.tensor.cross_section import (
    evaluate_ic_and_grouped_returns_tensor,
)
from quant_research_platform.kernels.tensor.portfolio import (
    build_tensor_portfolio_artifacts,
    summarize_tensor_exposure_from_path,
)
from quant_research_platform.runtime.profiling import (
    StageProfiler,
    active_stage_profiler,
    measure_stage,
)

POSITION_ARTIFACT_POLICY = "nonzero_positions_with_flat_timestamp_marker"


def run_evaluation(spec: EvaluationSpec) -> EvaluationSetResult:
    """运行 batch-native evaluation set 并写入 evaluation-set artifacts。"""

    if not isinstance(spec, EvaluationSpec):
        raise TypeError("spec must be an EvaluationSpec.")
    if active_stage_profiler() is None and (
        spec.profile_config.enabled or spec.profile_config.write_artifacts
    ):
        profiler = StageProfiler(
            spec.profile_config,
            metadata=_evaluation_set_profile_metadata(spec),
        )
        with profiler:
            result = run_evaluation(spec)
        if spec.profile_config.write_artifacts:
            output_dir = spec.profile_config.output_dir or spec.output_dir
            profiler.write_artifacts(output_dir)
        return result
    payload = evaluate_to_payload(spec)
    artifacts = write_evaluation_set_artifacts(
        spec.output_dir,
        run_context=payload.run_context,
        summary=payload.summary,
        tables=payload.tables,
        report=payload.report,
    )
    return EvaluationSetResult(
        summary=payload.summary,
        config=payload.config,
        metrics={
            name: frame
            for name, frame in payload.tables.items()
            if name != "combinations"
        },
        artifacts=artifacts,
        combinations=payload.tables["combinations"],
        combinations_path=artifacts.metric_paths["combinations"],
    )


def evaluate_to_payload(spec: EvaluationSpec) -> EvaluationRunPayload:
    """计算 evaluation-set payload tables，但不写入 artifacts。"""

    if not isinstance(spec, EvaluationSpec):
        raise TypeError("spec must be an EvaluationSpec.")
    validate_evaluation_spec_columns(spec)
    return_source_column = validate_evaluation_set_return_source_column(spec)
    kernel = get_kernel_descriptor(spec.kernel)
    if kernel.uses_tensor_layout:
        measure_stage(
            "kernel",
            "tensor_conversion",
            lambda: _prepare_tensor_batch(spec),
        )
    validated_panels = _validate_panels_by_label(spec)
    all_tables: dict[str, list[pd.DataFrame]] = {
        "ic": [],
        "ic_summary": [],
        "grouped_returns": [],
        "group_spread_summary": [],
        "performance_summary": [],
        "exposure_summary": [],
        "positions": [],
        "turnover": [],
        "returns": [],
        "cumulative_returns": [],
        "cost_breakdown": [],
        "weight_diagnostics": [],
        "neutralization_diagnostics": [],
        "tradability_diagnostics": [],
    }
    combination_records: list[dict[str, object]] = []
    weight_config = spec.weight_configs[0]
    cost_config = spec.cost_configs[0]
    portfolio_mode = weight_config.portfolio_mode
    for label in spec.label_columns:
        runner = (
            _run_tensor_feature_set
            if kernel.uses_tensor_layout
            else _run_dataframe_feature_set
        )
        raw_metrics = runner(
            spec=spec,
            validated_panel=validated_panels[label],
            label=label,
            weight_config=weight_config,
            cost_config=cost_config,
            return_source_column=return_source_column,
        )
        dimensioned_metrics = {
            name: _add_feature_set_combination_columns(
                frame,
                label=label,
                kernel=spec.kernel,
                portfolio_mode=portfolio_mode,
            )
            for name, frame in raw_metrics.items()
        }
        for name, frame in dimensioned_metrics.items():
            all_tables.setdefault(name, []).append(frame)
        for feature in spec.feature_columns:
            combination_id = _combination_id(
                feature=feature,
                label=label,
                kernel=spec.kernel,
                portfolio_mode=portfolio_mode,
            )
            combination_records.append(
                _combination_record(
                    combination_id=combination_id,
                    feature=feature,
                    label=label,
                    kernel=spec.kernel,
                    portfolio_mode=portfolio_mode,
                )
            )

    raw_tables = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in all_tables.items()
    }
    combinations = pd.DataFrame.from_records(combination_records)
    raw_tables["combinations"] = combinations
    tables = _to_schema_07_tables(
        raw_tables,
        spec=spec,
        return_source_column=return_source_column,
    )
    combinations = tables["combinations"]
    summary = _build_evaluation_set_summary(
        spec=spec,
        return_source_column=return_source_column,
        tables=tables,
        combinations=combinations,
    )
    run_context = _build_evaluation_set_run_context(
        spec=spec,
        return_source_column=return_source_column,
        summary=summary,
    )
    return EvaluationRunPayload(
        summary=summary,
        config=run_context["config"],
        run_context=run_context,
        tables=tables,
        report=_render_evaluation_set_report(summary, tables),
    )


def _run_dataframe_feature_set(
    *,
    spec: EvaluationSpec,
    validated_panel: pd.DataFrame,
    label: str,
    weight_config,
    cost_config,
    return_source_column: str,
) -> dict[str, pd.DataFrame]:
    panel_spec = PanelSpec(
        feature_columns=spec.feature_columns,
        label_column=label,
        timestamp_column=spec.timestamp_column,
        asset_column=spec.asset_column,
    )
    panel, neutralization_diagnostics = measure_stage(
        "metric",
        "neutralization",
        lambda: apply_neutralization(
            validated_panel,
            panel_spec,
            spec.neutralization_config,
        ),
    )
    ic, grouped_returns = _evaluate_cross_section_metrics_for_spec(
        panel=panel,
        panel_spec=panel_spec,
        n_groups=spec.n_groups,
        segment_columns=spec.segment_columns,
    )
    positions, turnover, returns, diagnostics, cost_breakdown = (
        _simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            panel_spec,
            weight_config,
            segment_columns=spec.segment_columns,
            portfolio_features=spec.feature_columns,
            tradability_config=spec.tradability_config,
            cost_config=cost_config,
            return_source_column=return_source_column,
        )
    )
    returns = returns.copy()
    if "label" in returns.columns:
        returns["return_source_column"] = returns["label"]
        returns["label"] = label
    cumulative_returns = measure_stage(
        "metric",
        "cumulative_returns",
        lambda: calculate_cumulative_returns(
            returns,
            segment_columns=spec.segment_columns,
        ),
    )
    performance_summary = measure_stage(
        "metric",
        "performance_summary",
        lambda: summarize_performance(
            returns,
            cumulative_returns,
            segment_columns=spec.segment_columns,
        ),
    )
    ic_summary = measure_stage(
        "metric",
        "ic_summary",
        lambda: summarize_ic(ic, segment_columns=spec.segment_columns),
    )
    group_spread_summary = measure_stage(
        "metric",
        "group_spread_summary",
        lambda: summarize_group_spread(
            grouped_returns,
            segment_columns=spec.segment_columns,
        ),
    )
    exposure_summary = measure_stage(
        "metric",
        "exposure_summary",
        lambda: summarize_exposure(
            positions,
            returns,
            segment_columns=spec.segment_columns,
        ),
    )
    tradability_diagnostics = _build_tradability_diagnostics(diagnostics)
    metrics = {
        "ic": ic,
        "ic_summary": ic_summary,
        "grouped_returns": grouped_returns,
        "group_spread_summary": group_spread_summary,
        "performance_summary": performance_summary,
        "exposure_summary": exposure_summary,
        "positions": positions,
        "turnover": turnover,
        "returns": returns,
        "cumulative_returns": cumulative_returns,
        "cost_breakdown": cost_breakdown,
        "weight_diagnostics": diagnostics,
        "neutralization_diagnostics": neutralization_diagnostics,
        "tradability_diagnostics": tradability_diagnostics,
    }
    if spec.time_breakdowns:
        metrics.update(
            build_time_breakdown_metrics(
                metrics,
                time_breakdowns=spec.time_breakdowns,
                segment_columns=spec.segment_columns,
            )
        )
    return metrics


def _run_tensor_feature_set(
    *,
    spec: EvaluationSpec,
    validated_panel: pd.DataFrame,
    label: str,
    weight_config,
    cost_config,
    return_source_column: str,
) -> dict[str, pd.DataFrame]:
    panel_spec = PanelSpec(
        feature_columns=spec.feature_columns,
        label_column=label,
        timestamp_column=spec.timestamp_column,
        asset_column=spec.asset_column,
    )
    panel, neutralization_diagnostics = measure_stage(
        "metric",
        "neutralization",
        lambda: apply_neutralization(
            validated_panel,
            panel_spec,
            spec.neutralization_config,
        ),
    )
    ic, grouped_returns = _evaluate_tensor_cross_section_metrics_for_spec(
        panel=panel,
        panel_spec=panel_spec,
        n_groups=spec.n_groups,
        segment_columns=spec.segment_columns,
        value_dtype=spec.tensor_config.value_dtype,
    )
    portfolio_artifacts = measure_stage(
        "metric",
        "tensor_portfolio_artifacts",
        lambda: build_tensor_portfolio_artifacts(
            panel,
            panel_spec,
            weight_config,
            cost_config,
            segment_columns=spec.segment_columns,
            portfolio_features=spec.feature_columns,
            tradability_config=spec.tradability_config,
            return_source_column=return_source_column,
            value_dtype=spec.tensor_config.value_dtype,
        ),
    )
    returns = portfolio_artifacts.returns.copy()
    if "label" in returns.columns:
        returns["return_source_column"] = returns["label"]
        returns["label"] = label
    cumulative_returns = measure_stage(
        "metric",
        "cumulative_returns",
        lambda: calculate_cumulative_returns(
            returns,
            segment_columns=spec.segment_columns,
        ),
    )
    performance_summary = measure_stage(
        "metric",
        "performance_summary",
        lambda: summarize_performance(
            returns,
            cumulative_returns,
            segment_columns=spec.segment_columns,
        ),
    )
    ic_summary = measure_stage(
        "metric",
        "ic_summary",
        lambda: summarize_ic(ic, segment_columns=spec.segment_columns),
    )
    group_spread_summary = measure_stage(
        "metric",
        "group_spread_summary",
        lambda: summarize_group_spread(
            grouped_returns,
            segment_columns=spec.segment_columns,
        ),
    )
    exposure_summary = measure_stage(
        "metric",
        "exposure_summary",
        lambda: summarize_tensor_exposure_from_path(
            portfolio_artifacts.exposure_path,
            returns,
            segment_columns=spec.segment_columns,
        ),
    )
    tradability_diagnostics = _build_tradability_diagnostics(
        portfolio_artifacts.diagnostics
    )
    metrics = {
        "ic": ic,
        "ic_summary": ic_summary,
        "grouped_returns": grouped_returns,
        "group_spread_summary": group_spread_summary,
        "performance_summary": performance_summary,
        "exposure_summary": exposure_summary,
        "positions": portfolio_artifacts.positions,
        "turnover": portfolio_artifacts.turnover,
        "returns": returns,
        "cumulative_returns": cumulative_returns,
        "cost_breakdown": portfolio_artifacts.cost_breakdown,
        "weight_diagnostics": portfolio_artifacts.diagnostics,
        "neutralization_diagnostics": neutralization_diagnostics,
        "tradability_diagnostics": tradability_diagnostics,
    }
    if spec.time_breakdowns:
        metrics.update(
            build_time_breakdown_metrics(
                metrics,
                time_breakdowns=spec.time_breakdowns,
                segment_columns=spec.segment_columns,
            )
        )
    return metrics


def _metrics_for_feature(
    metrics: dict[str, pd.DataFrame],
    feature: str,
) -> dict[str, pd.DataFrame]:
    return {
        name: (
            frame[frame["feature"] == feature] if "feature" in frame.columns else frame
        )
        for name, frame in metrics.items()
    }


def _build_tradability_diagnostics(weight_diagnostics: pd.DataFrame) -> pd.DataFrame:
    result = weight_diagnostics.copy()
    if result.empty:
        for column in (
            "eligible_rate",
            "can_enter_rate",
            "can_exit_rate",
            "entry_blocked_rate",
            "exit_blocked_rate",
        ):
            result[column] = pd.Series(dtype="float64")
        return result
    result["eligible_rate"] = _safe_rate(
        result["eligible_count"],
        result["universe_count"],
    )
    result["can_enter_rate"] = _safe_rate(
        result["can_enter_count"],
        result["eligible_count"],
    )
    result["can_exit_rate"] = _safe_rate(
        result["can_exit_count"],
        result["universe_count"],
    )
    result["entry_blocked_rate"] = _safe_rate(
        result["entry_blocked_count"],
        result["eligible_count"],
    )
    result["exit_blocked_rate"] = _safe_rate(
        result["exit_blocked_count"],
        result["universe_count"],
    )
    return result


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    bases = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    return values.divide(bases.where(bases.ne(0.0))).fillna(0.0)


def _evaluation_set_profile_metadata(spec: EvaluationSpec) -> dict[str, object]:
    return {
        "kernel": spec.kernel,
        "rows": int(len(spec.panel)),
        "features": len(spec.feature_columns),
        "labels": len(spec.label_columns),
        "combinations": (
            len(spec.feature_columns)
            * len(spec.label_columns)
            * len(spec.weight_configs)
            * len(spec.cost_configs)
        ),
        "run_id": str(spec.metadata.get("run_id", Path(spec.output_dir).name)),
    }


def _prepare_tensor_batch(spec: EvaluationSpec) -> None:
    panel_to_tensor_batch(
        spec.panel,
        timestamp_column=spec.timestamp_column,
        asset_column=spec.asset_column,
        signal_columns=spec.feature_columns,
        label_columns=spec.label_columns,
        value_dtype=spec.tensor_config.value_dtype,
    )


def _validate_panels_by_label(spec: EvaluationSpec) -> dict[str, pd.DataFrame]:
    return {
        label: measure_stage(
            "metric",
            "panel_validation",
            lambda label=label: validate_panel(
                spec.panel,
                PanelSpec(
                    feature_columns=spec.feature_columns,
                    label_column=label,
                    timestamp_column=spec.timestamp_column,
                    asset_column=spec.asset_column,
                ),
            ),
        )
        for label in spec.label_columns
    }


def _evaluate_cross_section_metrics_for_spec(
    *,
    panel: pd.DataFrame,
    panel_spec: PanelSpec,
    n_groups: int,
    segment_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not segment_columns:
        return measure_stage(
            "metric",
            "cross_section_metrics",
            lambda: evaluate_ic_and_grouped_returns_from_validated_panel(
                panel,
                panel_spec,
                n_groups=n_groups,
            ),
        )
    return measure_stage(
        "metric",
        "cross_section_metrics",
        lambda: _evaluate_cross_section_metrics_by_segment(
            panel=panel,
            spec=panel_spec,
            n_groups=n_groups,
            segment_columns=list(segment_columns),
        ),
    )


def _evaluate_tensor_cross_section_metrics_for_spec(
    *,
    panel: pd.DataFrame,
    panel_spec: PanelSpec,
    n_groups: int,
    segment_columns: tuple[str, ...],
    value_dtype: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not segment_columns:
        return measure_stage(
            "metric",
            "tensor_cross_section_metrics",
            lambda: evaluate_ic_and_grouped_returns_tensor(
                panel_to_tensor(panel, panel_spec, value_dtype=value_dtype),
                n_groups=n_groups,
            ),
        )
    return measure_stage(
        "metric",
        "tensor_cross_section_metrics",
        lambda: _evaluate_tensor_cross_section_metrics_by_segment(
            panel=panel,
            spec=panel_spec,
            n_groups=n_groups,
            segment_columns=list(segment_columns),
            value_dtype=value_dtype,
        ),
    )


def _add_feature_set_combination_columns(
    frame: pd.DataFrame,
    *,
    label: str,
    kernel: str,
    portfolio_mode: str,
) -> pd.DataFrame:
    result = frame.copy()
    if "feature" not in result.columns:
        result["feature"] = pd.Series(dtype="object", index=result.index)
    result["feature"] = result["feature"].astype(str)
    feature_to_combination_id = {
        feature: _combination_id(
            feature=feature,
            label=label,
            kernel=kernel,
            portfolio_mode=portfolio_mode,
        )
        for feature in result["feature"].drop_duplicates()
    }
    result["combination_id"] = result["feature"].map(feature_to_combination_id)
    result["label"] = label
    result["kernel"] = kernel
    result["portfolio_mode"] = portfolio_mode
    dimension_columns = [
        "combination_id",
        "feature",
        "label",
        "kernel",
        "portfolio_mode",
    ]
    remaining_columns = [
        column for column in result.columns if column not in dimension_columns
    ]
    return result[dimension_columns + remaining_columns]


def _combination_id(
    *,
    feature: str,
    label: str,
    kernel: str,
    portfolio_mode: str,
) -> str:
    return "__".join(_slug(value) for value in (feature, label, kernel, portfolio_mode))


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _combination_record(
    *,
    combination_id: str,
    feature: str,
    label: str,
    kernel: str,
    portfolio_mode: str,
) -> dict[str, object]:
    return {
        "combination_id": combination_id,
        "feature": feature,
        "label": label,
        "kernel": kernel,
        "portfolio_mode": portfolio_mode,
    }


def _to_schema_07_tables(
    tables: dict[str, pd.DataFrame],
    *,
    spec: EvaluationSpec,
    return_source_column: str,
) -> dict[str, pd.DataFrame]:
    combinations = tables["combinations"].copy()
    combinations["kernel"] = spec.kernel
    if "portfolio_mode" not in combinations.columns and spec.weight_configs:
        combinations["portfolio_mode"] = spec.weight_configs[0].portfolio_mode
    id_remap = {
        str(row["combination_id"]): _combination_id(
            feature=str(row["feature"]),
            label=str(row["label"]),
            kernel=str(row["kernel"]),
            portfolio_mode=str(row["portfolio_mode"]),
        )
        for _, row in combinations.iterrows()
    }
    combinations["combination_id"] = combinations["combination_id"].map(id_remap)
    schema_07: dict[str, pd.DataFrame] = {
        "combinations": combinations[
            ["combination_id", "feature", "label", "kernel", "portfolio_mode"]
        ].copy()
    }
    schema_07["ic"] = _remap_fact_table(tables["ic"], id_remap)
    schema_07["grouped_returns"] = _remap_fact_table(
        tables["grouped_returns"],
        id_remap,
    )
    returns = _remap_fact_table(
        tables["returns"],
        id_remap,
        keep_columns=("portfolio_mode",),
    )
    if "return_source_column" not in returns.columns:
        returns["return_source_column"] = return_source_column
    returns = returns.rename(columns={"net_portfolio_return": "net_return"})
    schema_07["returns"] = returns[
        [
            column
            for column in (
                "combination_id",
                "timestamp",
                "portfolio_mode",
                "long_return",
                "short_return",
                "gross_portfolio_return",
                "turnover",
                "cost",
                "net_return",
                "long_count",
                "short_count",
                "return_source_column",
                *spec.segment_columns,
            )
            if column in returns.columns
        ]
    ].copy()
    ic_summary = _remap_fact_table(tables["ic_summary"], id_remap)
    group_spread_summary = _remap_fact_table(
        tables["group_spread_summary"],
        id_remap,
    )
    performance_summary = _remap_fact_table(tables["performance_summary"], id_remap)
    schema_07["evaluation_metrics_summary"] = _evaluation_metrics_summary_table(
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        performance_summary=performance_summary,
    )
    schema_07["coverage_summary"] = _coverage_summary_table(spec)
    schema_07.update(
        DiagnosticRunner(spec.diagnostics_spec).run(
            DiagnosticInputBundle(
                base_tables={
                    **schema_07,
                    "positions": _schema_07_weights(tables["positions"], id_remap),
                    "weight_diagnostics": _remap_fact_table(
                        tables["weight_diagnostics"],
                        id_remap,
                    ),
                    "cost_breakdown": _remap_fact_table(
                        tables["cost_breakdown"],
                        id_remap,
                    ),
                },
                metadata={
                    "a_share_assumptions": _a_share_assumptions_from_metadata(spec),
                    "domain": spec.metadata.get("domain", "evaluation"),
                },
            )
        )
    )
    schema_07.update(_schema_07_breakdown_tables(tables, id_remap))
    return schema_07


def _remap_fact_table(
    frame: pd.DataFrame,
    id_remap: dict[str, str],
    keep_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    result = frame.copy()
    if "combination_id" in result.columns:
        result["combination_id"] = result["combination_id"].map(id_remap)
    retired = ["feature", "label", "kernel", "portfolio_mode"]
    keep = set(keep_columns)
    retired_columns = [column for column in retired if column in result.columns]
    retired_columns = [column for column in retired_columns if column not in keep]
    return result.drop(columns=retired_columns)


def _schema_07_weights(
    positions: pd.DataFrame,
    id_remap: dict[str, str],
) -> pd.DataFrame:
    weights = _remap_fact_table(positions, id_remap)
    weights = weights.rename(columns={"position": "target_weight"})
    required = ["combination_id", "timestamp", "asset", "target_weight"]
    return weights[[column for column in required if column in weights.columns]]


def _label_availability_table(spec: EvaluationSpec) -> pd.DataFrame:
    records = []
    for label in spec.label_columns:
        available = int(spec.panel[label].notna().sum())
        total = int(len(spec.panel))
        records.append(
            {
                "label": label,
                "available_rows": available,
                "missing_rows": total - available,
                "coverage_rate": available / total if total else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _factor_coverage_table(spec: EvaluationSpec) -> pd.DataFrame:
    records = []
    for feature in spec.feature_columns:
        available = int(spec.panel[feature].notna().sum())
        total = int(len(spec.panel))
        records.append(
            {
                "feature": feature,
                "available_rows": available,
                "missing_rows": total - available,
                "coverage_rate": available / total if total else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _tradability_coverage_table(spec: EvaluationSpec) -> pd.DataFrame:
    records = []
    for field in ("eligible", "can_enter", "can_exit"):
        if field in spec.panel.columns:
            available = int(spec.panel[field].notna().sum())
            missing = int(len(spec.panel) - available)
            coverage = available / len(spec.panel) if len(spec.panel) else 0.0
        else:
            available = int(len(spec.panel))
            missing = 0
            coverage = 1.0
        records.append(
            {
                "field": field,
                "available_rows": available,
                "missing_rows": missing,
                "coverage_rate": coverage,
            }
        )
    return pd.DataFrame.from_records(records)


def _evaluation_summary_table(
    *,
    combinations: pd.DataFrame,
    ic_summary: pd.DataFrame,
    performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    ic_compact = (
        ic_summary.groupby("combination_id", as_index=False)["mean_ic"].mean()
        if not ic_summary.empty
        else pd.DataFrame(columns=["combination_id", "mean_ic"])
    )
    performance_compact = (
        performance_summary.groupby("combination_id", as_index=False).agg(
            final_cumulative_net_return=("final_cumulative_net_return", "mean"),
            mean_turnover=("mean_turnover", "mean"),
            total_cost=("total_cost", "sum"),
        )
        if not performance_summary.empty
        else pd.DataFrame(
            columns=[
                "combination_id",
                "final_cumulative_net_return",
                "mean_turnover",
                "total_cost",
            ]
        )
    )
    metrics = combinations.merge(
        ic_compact,
        on="combination_id",
        how="left",
    )
    return metrics.merge(
        performance_compact,
        on="combination_id",
        how="left",
    )


def _evaluation_metrics_summary_table(
    *,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    metric_sources = [
        (
            "ic",
            ic_summary,
            (
                "mean_ic",
                "std_ic",
                "icir",
                "positive_ic_rate",
                "mean_rankic",
                "std_rankic",
                "rankicir",
                "positive_rankic_rate",
            ),
        ),
        (
            "group_spread",
            group_spread_summary,
            (
                "top_group_mean_return",
                "bottom_group_mean_return",
                "top_bottom_spread_mean",
                "top_bottom_spread_hit_rate",
                "group_monotonicity_score",
            ),
        ),
        (
            "portfolio",
            performance_summary,
            (
                "periods",
                "final_cumulative_gross_return",
                "final_cumulative_net_return",
                "mean_net_portfolio_return",
                "mean_turnover",
                "max_drawdown",
                "sharpe_like",
                "sortino_like",
                "total_cost",
                "win_rate",
            ),
        ),
    ]
    for family, frame, columns in metric_sources:
        for _, row in frame.iterrows():
            combination_id = str(row["combination_id"])
            for column in columns:
                if column not in frame.columns:
                    continue
                records.append(
                    {
                        "combination_id": combination_id,
                        "metric_family": family,
                        "metric_name": column,
                        "metric_value": row.get(column),
                        "unit": "ratio",
                        "n_observations": row.get("periods", row.get("n_obs", "")),
                        "diagnostic_status": "ok",
                        "status_reason": "",
                        "warning": "",
                    }
                )
    result = pd.DataFrame.from_records(
        records,
        columns=[
            "combination_id",
            "metric_family",
            "metric_name",
            "metric_value",
            "unit",
            "n_observations",
            "diagnostic_status",
            "status_reason",
            "warning",
        ],
    )
    if result.empty:
        return result
    grouped = result.groupby(
        ["combination_id", "metric_family", "metric_name"],
        as_index=False,
        dropna=False,
    ).agg(
        metric_value=("metric_value", "mean"),
        unit=("unit", "first"),
        n_observations=("n_observations", "sum"),
        diagnostic_status=("diagnostic_status", "first"),
        status_reason=("status_reason", "first"),
        warning=("warning", "first"),
    )
    return grouped[
        [
            "combination_id",
            "metric_family",
            "metric_name",
            "metric_value",
            "unit",
            "n_observations",
            "diagnostic_status",
            "status_reason",
            "warning",
        ]
    ]


def _coverage_summary_table(spec: EvaluationSpec) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in _label_availability_table(spec).to_dict(orient="records"):
        records.append(_coverage_record("label", str(row["label"]), row))
    for row in _factor_coverage_table(spec).to_dict(orient="records"):
        records.append(_coverage_record("factor", str(row["feature"]), row))
    for row in _tradability_coverage_table(spec).to_dict(orient="records"):
        records.append(_coverage_record("tradability", str(row["field"]), row))
    return pd.DataFrame.from_records(records)


def _coverage_record(
    input_family: str,
    input_name: str,
    row: dict[str, object],
) -> dict[str, object]:
    return {
        "input_family": input_family,
        "input_name": input_name,
        "available_rows": row["available_rows"],
        "missing_rows": row["missing_rows"],
        "coverage_rate": row["coverage_rate"],
        "diagnostic_status": "ok",
        "status_reason": "",
        "warning": "",
    }


def _schema_07_breakdown_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "metric_name" not in result.columns:
        value_columns = [
            column
            for column in result.columns
            if column
            not in {
                "combination_id",
                "period_type",
                "period",
                "period_start",
                "period_end",
            }
        ]
        records = []
        for _, row in result.iterrows():
            for column in value_columns:
                records.append(
                    {
                        "combination_id": row.get("combination_id"),
                        "period_type": row.get("period_type"),
                        "period": row.get("period"),
                        "period_start": row.get("period_start"),
                        "period_end": row.get("period_end"),
                        "metric_family": "evaluation",
                        "metric_name": column,
                        "metric_value": row.get(column),
                        "unit": "ratio",
                        "n_observations": row.get("periods", ""),
                        "diagnostic_status": "ok",
                        "status_reason": "",
                        "warning": "",
                    }
                )
        result = pd.DataFrame.from_records(records)
    return result


def _schema_07_breakdown_tables(
    tables: dict[str, pd.DataFrame],
    id_remap: dict[str, str],
) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for period in ("year", "quarter", "month"):
        ic = tables.get(f"ic_summary_by_{period}", pd.DataFrame())
        spread = tables.get(f"group_spread_summary_by_{period}", pd.DataFrame())
        performance = tables.get(f"performance_summary_by_{period}", pd.DataFrame())
        if ic.empty and spread.empty and performance.empty:
            continue
        if period == "year" and not ic.empty:
            result["ic_breakdown_by_year"] = _schema_07_yearly_ic_breakdown(
                tables["ic"],
                id_remap,
            )
        if period == "month" and not ic.empty:
            result["ic_breakdown_by_month"] = _schema_07_monthly_ic_breakdown(
                tables["ic"],
                id_remap,
            )
        result[f"evaluation_metrics_summary_by_{period}"] = _schema_07_period_metrics(
            period=period,
            ic_summary=_remap_fact_table(ic, id_remap),
            group_spread_summary=_remap_fact_table(spread, id_remap),
            performance_summary=_remap_fact_table(performance, id_remap),
        )
    return result


def _schema_07_yearly_ic_breakdown(
    ic: pd.DataFrame,
    id_remap: dict[str, str],
) -> pd.DataFrame:
    columns = [
        "combination_id",
        "period_type",
        "period",
        "period_start",
        "period_end",
        "mean_ic",
        "std_ic",
        "icir",
        "positive_ic_rate",
        "mean_rankic",
        "std_rankic",
        "rankicir",
        "positive_rankic_rate",
        "n_observations",
        "diagnostic_status",
        "status_reason",
        "warning",
    ]
    frame = _remap_fact_table(ic, id_remap)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.copy()
    timestamp = pd.to_datetime(frame["timestamp"])
    frame["period"] = timestamp.dt.year.astype(str)
    records = []
    for (combination_id, period), group in frame.groupby(
        ["combination_id", "period"],
        sort=True,
        dropna=False,
    ):
        ic_values = pd.to_numeric(group["ic"], errors="coerce").dropna()
        rankic_values = pd.to_numeric(group["rankic"], errors="coerce").dropna()
        mean_ic = float(ic_values.mean()) if len(ic_values) else 0.0
        std_ic = float(ic_values.std(ddof=0)) if len(ic_values) else 0.0
        mean_rankic = float(rankic_values.mean()) if len(rankic_values) else 0.0
        std_rankic = float(rankic_values.std(ddof=0)) if len(rankic_values) else 0.0
        records.append(
            {
                "combination_id": combination_id,
                "period_type": "year",
                "period": period,
                "period_start": period,
                "period_end": period,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": _safe_ratio(mean_ic, std_ic),
                "positive_ic_rate": (
                    float((ic_values > 0).mean()) if len(ic_values) else 0.0
                ),
                "mean_rankic": mean_rankic,
                "std_rankic": std_rankic,
                "rankicir": _safe_ratio(mean_rankic, std_rankic),
                "positive_rankic_rate": (
                    float((rankic_values > 0).mean()) if len(rankic_values) else 0.0
                ),
                "n_observations": int(len(group)),
                "diagnostic_status": "ok",
                "status_reason": "",
                "warning": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _schema_07_monthly_ic_breakdown(
    ic: pd.DataFrame,
    id_remap: dict[str, str],
) -> pd.DataFrame:
    columns = [
        "combination_id",
        "period_type",
        "period",
        "period_start",
        "period_end",
        "mean_ic",
        "std_ic",
        "icir",
        "positive_ic_rate",
        "mean_rankic",
        "std_rankic",
        "rankicir",
        "positive_rankic_rate",
        "n_observations",
        "diagnostic_status",
        "status_reason",
        "warning",
    ]
    frame = _remap_fact_table(ic, id_remap)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.copy()
    timestamp = pd.to_datetime(frame["timestamp"])
    frame["period"] = timestamp.dt.to_period("M").astype(str)
    records = []
    for (combination_id, period), group in frame.groupby(
        ["combination_id", "period"],
        sort=True,
        dropna=False,
    ):
        ic_values = pd.to_numeric(group["ic"], errors="coerce").dropna()
        rankic_values = pd.to_numeric(group["rankic"], errors="coerce").dropna()
        mean_ic = float(ic_values.mean()) if len(ic_values) else 0.0
        std_ic = float(ic_values.std(ddof=0)) if len(ic_values) else 0.0
        mean_rankic = float(rankic_values.mean()) if len(rankic_values) else 0.0
        std_rankic = float(rankic_values.std(ddof=0)) if len(rankic_values) else 0.0
        records.append(
            {
                "combination_id": combination_id,
                "period_type": "month",
                "period": period,
                "period_start": period,
                "period_end": period,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": _safe_ratio(mean_ic, std_ic),
                "positive_ic_rate": (
                    float((ic_values > 0).mean()) if len(ic_values) else 0.0
                ),
                "mean_rankic": mean_rankic,
                "std_rankic": std_rankic,
                "rankicir": _safe_ratio(mean_rankic, std_rankic),
                "positive_rankic_rate": (
                    float((rankic_values > 0).mean()) if len(rankic_values) else 0.0
                ),
                "n_observations": int(len(group)),
                "diagnostic_status": "ok",
                "status_reason": "",
                "warning": "",
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _schema_07_period_metrics(
    *,
    period: str,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    performance_summary: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for family, frame in (
        ("ic", ic_summary),
        ("group_spread", group_spread_summary),
        ("portfolio", performance_summary),
    ):
        if frame.empty:
            continue
        normalized = frame.copy()
        if "period" not in normalized.columns and period in normalized.columns:
            normalized["period"] = normalized[period].astype(str)
        if "period_start" not in normalized.columns:
            normalized["period_start"] = normalized["period"]
        if "period_end" not in normalized.columns:
            normalized["period_end"] = normalized["period"]
        converted = _schema_07_breakdown_table(
            normalized.drop(columns=[period], errors="ignore")
        )
        converted["metric_family"] = converted["metric_family"].where(
            converted["metric_family"].ne("evaluation"),
            family,
        )
        frames.append(converted)
    if not frames:
        return pd.DataFrame(
            columns=[
                "combination_id",
                "period_type",
                "period",
                "period_start",
                "period_end",
                "metric_family",
                "metric_name",
                "metric_value",
                "unit",
                "n_observations",
                "diagnostic_status",
                "status_reason",
                "warning",
            ]
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["period_type"] = period
    return combined[
        [
            "combination_id",
            "period_type",
            "period",
            "period_start",
            "period_end",
            "metric_family",
            "metric_name",
            "metric_value",
            "unit",
            "n_observations",
            "diagnostic_status",
            "status_reason",
            "warning",
        ]
    ]


def _actual_date_coverage(spec: EvaluationSpec) -> dict[str, object]:
    timestamps = pd.to_datetime(spec.panel[spec.timestamp_column], errors="coerce")
    timestamps = timestamps.dropna()
    if timestamps.empty:
        return {"start_date": None, "end_date": None, "n_timestamps": 0}
    return {
        "start_date": str(timestamps.min().date()),
        "end_date": str(timestamps.max().date()),
        "n_timestamps": int(timestamps.nunique()),
    }


def _a_share_assumptions_from_metadata(spec: EvaluationSpec) -> dict[str, object]:
    if spec.metadata.get("domain") != "a_share":
        return {"status": "not_applicable"}
    return {
        "status": "recorded",
        "limit_up_down": "handled by tradability inputs when supplied",
        "suspension": "handled by tradability inputs when supplied",
        "t_plus_one": True,
        "short_sale": "not simulated",
        "cost_model": "configured modeled cost only",
    }


def _combination_id(
    *,
    feature: str,
    label: str,
    kernel: str,
    portfolio_mode: str,
) -> str:
    return "__".join(_slug(value) for value in (feature, label, kernel, portfolio_mode))


def _build_evaluation_set_summary(
    *,
    spec: EvaluationSpec,
    return_source_column: str,
    tables: dict[str, pd.DataFrame],
    combinations: pd.DataFrame,
) -> dict[str, object]:
    run_id = str(spec.metadata.get("run_id", Path(spec.output_dir).name))
    return {
        "run_id": run_id,
        "project": spec.metadata["project"],
        "kind": spec.metadata["kind"],
        "data_stage": spec.metadata["data_stage"],
        "base_evaluation": {
            "timestamp_column": spec.timestamp_column,
            "asset_column": spec.asset_column,
            "feature_columns": list(spec.feature_columns),
            "label_columns": list(spec.label_columns),
            "return_source_column": return_source_column,
            "kernel": spec.kernel,
            "n_groups": int(spec.n_groups),
            "segment_columns": list(spec.segment_columns),
            "breakdown_enabled": bool(spec.time_breakdowns),
            "breakdown_columns": list(spec.time_breakdowns),
            "period_breakdown_policy": PERIOD_BREAKDOWN_POLICY,
            "domain": spec.metadata.get("domain", "evaluation"),
            "uses_external_data": bool(spec.metadata.get("uses_external_data", False)),
        },
        "diagnostics": {
            **spec.diagnostics_spec.to_public_dict(),
            "status": "enabled" if spec.diagnostics_spec.enabled else "disabled",
        },
        "n_rows_evaluated": int(len(spec.panel)),
        "n_combinations": int(len(combinations)),
        "tables": {name: int(len(frame)) for name, frame in tables.items()},
        "warnings": [],
        "boundary": spec.domain_notes.get(
            "boundary",
            "No real data, API connection, credentials, paper trading, live trading, "
            "or investment advice is provided by this evaluation layer.",
        ),
    }


def _build_evaluation_set_run_context(
    *,
    spec: EvaluationSpec,
    return_source_column: str,
    summary: dict[str, object],
) -> dict[str, object]:
    evaluation_spec = {
        "metadata": spec.metadata,
        "domain_notes": spec.domain_notes,
        "schema": {
            "timestamp_column": spec.timestamp_column,
            "asset_column": spec.asset_column,
            "feature_columns": list(spec.feature_columns),
            "label_columns": list(spec.label_columns),
            "return_source_column": return_source_column,
        },
        "n_groups": spec.n_groups,
        "segment_columns": list(spec.segment_columns),
        "time_breakdowns": list(spec.time_breakdowns),
        "kernel": spec.kernel,
        "weight_configs": [asdict(config) for config in spec.weight_configs],
        "execution_config": asdict(spec.execution_config),
        "tradability_config": asdict(spec.tradability_config),
        "cost_configs": [asdict(config) for config in spec.cost_configs],
        "neutralization_config": asdict(spec.neutralization_config),
        "time_semantics_config": asdict(spec.time_semantics_config),
        "tensor_config": asdict(spec.tensor_config),
    }
    return {
        "config": {
            "evaluation_spec": evaluation_spec,
            "diagnostics_spec": spec.diagnostics_spec.to_public_dict(),
            "a_share_assumptions": _a_share_assumptions_from_metadata(spec),
            "config_fingerprint": "unavailable",
        },
        "provenance": {
            "run_id": summary["run_id"],
            "input_panel_fingerprint": None,
            "input_data_version": "unavailable",
            "input_artifact_refs": [],
            "output_artifact_index": {},
            "diagnostic_policy_fingerprint": _diagnostic_policy_fingerprint(spec),
            "generated_at": "unavailable",
            "generator": "quant_research_platform",
            "code_version": "unavailable",
            "environment_summary": {},
            "staging_payload_fingerprint": None,
            "factor_definition_version": "unavailable",
            "random_seed": "not_applicable",
            "input_fingerprint": None,
            "actual_date_coverage": _actual_date_coverage(spec),
        },
    }


def _diagnostic_policy_fingerprint(spec: EvaluationSpec) -> str:
    if not spec.diagnostics_spec.enabled:
        return "disabled"
    return "n5_v1_descriptive_proxy"


def _render_evaluation_set_report(
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> str:
    base_evaluation = summary.get("base_evaluation")
    kernel = (
        base_evaluation.get("kernel") if isinstance(base_evaluation, dict) else ""
    )
    lines = [
        "# Evaluation Set Report",
        "",
        "This report summarizes one batch-native evaluation set.",
        "",
        "## Summary",
        "",
        f"- run_id: {summary['run_id']}",
        f"- n_combinations: {summary['n_combinations']}",
        f"- kernel: {kernel}",
        "",
        "## Tables",
        "",
    ]
    for name, frame in tables.items():
        lines.append(f"- {name}: {len(frame)} rows")
    lines.extend(["", "## Boundary", "", str(summary["boundary"]), ""])
    return "\n".join(lines)


def _evaluate_cross_section_metrics_by_segment(
    *,
    panel: pd.DataFrame,
    spec,
    n_groups: int,
    segment_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic_frames: list[pd.DataFrame] = []
    grouped_return_frames: list[pd.DataFrame] = []
    grouped = panel.groupby(segment_columns, sort=True, dropna=False)
    for segment_values, segment_frame in grouped:
        if not isinstance(segment_values, tuple):
            segment_values = (segment_values,)
        ic, grouped_returns = evaluate_ic_and_grouped_returns_from_validated_panel(
            segment_frame,
            spec,
            n_groups=n_groups,
        )
        for frame in (ic, grouped_returns):
            for offset, (column, value) in enumerate(
                zip(segment_columns, segment_values, strict=True)
            ):
                frame.insert(offset, column, value)
        ic_frames.append(ic)
        grouped_return_frames.append(grouped_returns)
    if ic_frames:
        return (
            pd.concat(ic_frames, ignore_index=True),
            pd.concat(grouped_return_frames, ignore_index=True),
        )

    ic, grouped_returns = evaluate_ic_and_grouped_returns_from_validated_panel(
        panel.iloc[0:0],
        spec,
        n_groups=n_groups,
    )
    for frame in (ic, grouped_returns):
        for offset, column in enumerate(segment_columns):
            frame.insert(offset, column, pd.Series(dtype="object"))
    return ic, grouped_returns


def _evaluate_tensor_cross_section_metrics_by_segment(
    *,
    panel: pd.DataFrame,
    spec,
    n_groups: int,
    segment_columns: list[str],
    value_dtype: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ic_frames: list[pd.DataFrame] = []
    grouped_return_frames: list[pd.DataFrame] = []
    grouped = panel.groupby(segment_columns, sort=True, dropna=False)
    for segment_values, segment_frame in grouped:
        if not isinstance(segment_values, tuple):
            segment_values = (segment_values,)
        ic, grouped_returns = evaluate_ic_and_grouped_returns_tensor(
            panel_to_tensor(segment_frame, spec, value_dtype=value_dtype),
            n_groups=n_groups,
        )
        for frame in (ic, grouped_returns):
            for offset, (column, value) in enumerate(
                zip(segment_columns, segment_values, strict=True)
            ):
                frame.insert(offset, column, value)
        ic_frames.append(ic)
        grouped_return_frames.append(grouped_returns)
    if ic_frames:
        return (
            pd.concat(ic_frames, ignore_index=True),
            pd.concat(grouped_return_frames, ignore_index=True),
        )

    ic, grouped_returns = evaluate_ic_and_grouped_returns_tensor(
        panel_to_tensor(panel.iloc[0:0], spec, value_dtype=value_dtype),
        n_groups=n_groups,
    )
    for frame in (ic, grouped_returns):
        for offset, column in enumerate(segment_columns):
            frame.insert(offset, column, pd.Series(dtype="object"))
    return ic, grouped_returns

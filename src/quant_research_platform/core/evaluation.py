"""Core factor and prediction signal evaluation runner."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.display import build_display_manifest
from quant_research_platform.artifacts.writer import (
    ARTIFACT_SCHEMA_VERSION,
    write_evaluation_set_artifacts,
)
from quant_research_platform.core.contracts import EvaluationSpec
from quant_research_platform.core.neutralization import apply_neutralization
from quant_research_platform.core.panel import PanelSpec, validate_panel
from quant_research_platform.core.period_breakdowns import (
    PERIOD_BREAKDOWN_POLICY,
    build_time_breakdown_metrics,
)
from quant_research_platform.core.profiling import (
    StageProfiler,
    active_stage_profiler,
    measure_stage,
)
from quant_research_platform.core.results import EvaluationSetResult
from quant_research_platform.engines.dataframe.cross_section import (
    evaluate_ic_and_grouped_returns_from_validated_panel,
    summarize_group_spread,
    summarize_ic,
)
from quant_research_platform.engines.dataframe.portfolio import (
    _simulate_weighted_long_short_returns_and_cost_from_validated_panel,
    calculate_cumulative_pnl,
    summarize_exposure,
    summarize_performance,
)
from quant_research_platform.engines.tensor.conversion import panel_to_tensor_batch

POSITION_ARTIFACT_POLICY = "nonzero_positions_with_flat_timestamp_marker"


def run_evaluation(spec: EvaluationSpec) -> EvaluationSetResult:
    """Run a batch-native evaluation set and write evaluation-set artifacts."""

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
    _validate_evaluation_spec_columns(spec)
    pnl_return_column = _validate_evaluation_set_pnl_return_column(spec)
    if spec.engine == "tensor":
        measure_stage(
            "engine",
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
        "pnl": [],
        "cost_breakdown": [],
        "weight_diagnostics": [],
        "neutralization_diagnostics": [],
    }
    combination_records: list[dict[str, object]] = []
    for label in spec.label_columns:
        for weight_index, weight_config in enumerate(spec.weight_configs):
            for cost_index, cost_config in enumerate(spec.cost_configs):
                weight_id = f"w{weight_index}"
                cost_id = f"c{cost_index}"
                raw_metrics = _run_dataframe_feature_set(
                    spec=spec,
                    validated_panel=validated_panels[label],
                    label=label,
                    weight_id=weight_id,
                    cost_id=cost_id,
                    weight_config=weight_config,
                    cost_config=cost_config,
                    pnl_return_column=pnl_return_column,
                )
                dimensioned_metrics = {
                    name: _add_feature_set_combination_columns(
                        frame,
                        label=label,
                        weight_id=weight_id,
                        cost_id=cost_id,
                    )
                    for name, frame in raw_metrics.items()
                }
                for name, frame in dimensioned_metrics.items():
                    all_tables.setdefault(name, []).append(frame)
                for feature in spec.feature_columns:
                    combination_id = _combination_id(
                        feature=feature,
                        label=label,
                        weight_id=weight_id,
                        cost_id=cost_id,
                    )
                    feature_metrics = _metrics_for_feature(raw_metrics, feature)
                    combination_records.append(
                        _combination_record(
                            combination_id=combination_id,
                            feature=feature,
                            label=label,
                            weight_id=weight_id,
                            cost_id=cost_id,
                            weight_config=weight_config,
                            cost_config=cost_config,
                            metrics=feature_metrics,
                        )
                    )

    tables = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in all_tables.items()
    }
    combinations = pd.DataFrame.from_records(combination_records)
    tables["combinations"] = combinations
    summary = _build_evaluation_set_summary(
        spec=spec,
        pnl_return_column=pnl_return_column,
        tables=tables,
        combinations=combinations,
    )
    config = _build_evaluation_set_config(
        spec=spec,
        pnl_return_column=pnl_return_column,
    )
    manifest = _build_evaluation_set_manifest(
        summary=summary,
        output_dir=spec.output_dir,
    )
    artifacts = write_evaluation_set_artifacts(
        spec.output_dir,
        manifest=manifest,
        config=config,
        summary=summary,
        tables=tables,
        report=_render_evaluation_set_report(summary, tables),
    )
    return EvaluationSetResult(
        summary=summary,
        metrics={
            name: frame for name, frame in tables.items() if name != "combinations"
        },
        artifacts=artifacts,
        combinations=combinations,
        combinations_path=artifacts.metric_paths["combinations"],
    )


def _run_dataframe_feature_set(
    *,
    spec: EvaluationSpec,
    validated_panel: pd.DataFrame,
    label: str,
    weight_id: str,
    cost_id: str,
    weight_config,
    cost_config,
    pnl_return_column: str,
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
            pnl_return_column=pnl_return_column,
        )
    )
    returns = returns.copy()
    if "label" in returns.columns:
        returns["pnl_return_column"] = returns["label"]
        returns["label"] = label
    pnl = measure_stage(
        "metric",
        "pnl",
        lambda: calculate_cumulative_pnl(
            returns,
            segment_columns=spec.segment_columns,
        ),
    )
    performance_summary = measure_stage(
        "metric",
        "performance_summary",
        lambda: summarize_performance(
            returns,
            pnl,
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
        "pnl": pnl,
        "cost_breakdown": cost_breakdown,
        "weight_diagnostics": diagnostics,
        "neutralization_diagnostics": neutralization_diagnostics,
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
            frame[frame["feature"] == feature]
            if "feature" in frame.columns
            else frame
        )
        for name, frame in metrics.items()
    }


def _evaluation_set_profile_metadata(spec: EvaluationSpec) -> dict[str, object]:
    return {
        "engine": spec.engine,
        "rows": int(len(spec.panel)),
        "features": len(spec.feature_columns),
        "labels": len(spec.label_columns),
        "combinations": (
            len(spec.feature_columns)
            * len(spec.label_columns)
            * len(spec.weight_configs)
            * len(spec.cost_configs)
        ),
        "evaluation_id": str(spec.metadata.get("run_id", Path(spec.output_dir).name)),
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


def _validate_evaluation_spec_columns(spec: EvaluationSpec) -> None:
    panel = spec.panel
    requested_columns = [
        spec.timestamp_column,
        spec.asset_column,
        *spec.feature_columns,
        *spec.label_columns,
        *spec.segment_columns,
        spec.pnl_return_column,
        spec.tradability_config.eligibility_column,
        spec.tradability_config.can_enter_column,
        spec.tradability_config.can_exit_column,
        spec.execution_config.execution_price_column,
        *spec.neutralization_config.group_columns,
        *spec.neutralization_config.exposure_columns,
        *(config.risk_column for config in spec.weight_configs),
    ]
    missing = sorted(
        {
            column
            for column in requested_columns
            if column is not None and column not in panel.columns
        }
    )
    if missing:
        raise ValueError(f"evaluation panel is missing columns: {missing}.")


def _validate_evaluation_set_pnl_return_column(spec: EvaluationSpec) -> str:
    pnl_return_column = spec.resolve_pnl_return_column(spec.panel)
    if pnl_return_column not in spec.panel.columns:
        raise ValueError(
            "PnL curve requires pnl_return_column, but column "
            f"{pnl_return_column!r} was not found."
        )
    if not pd.api.types.is_numeric_dtype(spec.panel[pnl_return_column]):
        raise TypeError(f"pnl_return_column {pnl_return_column!r} must be numeric.")
    return pnl_return_column


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


def _add_feature_set_combination_columns(
    frame: pd.DataFrame,
    *,
    label: str,
    weight_id: str,
    cost_id: str,
) -> pd.DataFrame:
    result = frame.copy()
    if "feature" not in result.columns:
        result["feature"] = pd.Series(dtype="object", index=result.index)
    result["feature"] = result["feature"].astype(str)
    feature_to_combination_id = {
        feature: _combination_id(
            feature=feature,
            label=label,
            weight_id=weight_id,
            cost_id=cost_id,
        )
        for feature in result["feature"].drop_duplicates()
    }
    result["combination_id"] = result["feature"].map(feature_to_combination_id)
    result["label"] = label
    result["weight_id"] = weight_id
    result["cost_id"] = cost_id
    dimension_columns = ["combination_id", "feature", "label", "weight_id", "cost_id"]
    remaining_columns = [
        column for column in result.columns if column not in dimension_columns
    ]
    return result[dimension_columns + remaining_columns]


def _combination_id(
    *,
    feature: str,
    label: str,
    weight_id: str,
    cost_id: str,
) -> str:
    return "__".join(
        _slug(value) for value in (feature, label, weight_id, cost_id)
    )


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _combination_record(
    *,
    combination_id: str,
    feature: str,
    label: str,
    weight_id: str,
    cost_id: str,
    weight_config,
    cost_config,
    metrics: dict[str, pd.DataFrame],
) -> dict[str, object]:
    performance = _first_row(metrics["performance_summary"])
    ic_summary = _first_row(metrics["ic_summary"])
    return {
        "combination_id": combination_id,
        "feature": feature,
        "label": label,
        "weight_id": weight_id,
        "cost_id": cost_id,
        "weight_method": weight_config.method,
        "direction": weight_config.direction,
        "portfolio_mode": weight_config.portfolio_mode,
        "cost_bps": _effective_cost_bps(weight_config, cost_config),
        "mean_ic": ic_summary.get("mean_ic", 0.0),
        "mean_rankic": ic_summary.get("mean_rankic", 0.0),
        "final_cumulative_net_return": performance.get(
            "final_cumulative_net_return",
            0.0,
        ),
        "sharpe_like": performance.get("sharpe_like", 0.0),
        "max_drawdown": performance.get("max_drawdown", 0.0),
    }


def _first_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _effective_cost_bps(weight_config, cost_config) -> float:
    return float(
        weight_config.cost_bps
        + cost_config.buy_cost_bps
        + cost_config.sell_cost_bps
        + cost_config.slippage_bps
        + (cost_config.spread_bps or 0.0)
    )


def _build_evaluation_set_summary(
    *,
    spec: EvaluationSpec,
    pnl_return_column: str,
    tables: dict[str, pd.DataFrame],
    combinations: pd.DataFrame,
) -> dict[str, object]:
    evaluation_id = str(spec.metadata.get("run_id", Path(spec.output_dir).name))
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "evaluation_set",
        "evaluation_id": evaluation_id,
        "project": spec.metadata["project"],
        "kind": spec.metadata["kind"],
        "domain": spec.metadata["domain"],
        "data_stage": spec.metadata["data_stage"],
        "uses_external_data": spec.metadata["uses_external_data"],
        "schema": {
            "timestamp_column": spec.timestamp_column,
            "asset_column": spec.asset_column,
            "feature_columns": list(spec.feature_columns),
            "label_columns": list(spec.label_columns),
            "pnl_return_column": pnl_return_column,
        },
        "n_rows_evaluated": int(len(spec.panel)),
        "n_combinations": int(len(combinations)),
        "n_groups": int(spec.n_groups),
        "segment_columns": list(spec.segment_columns),
        "breakdown_enabled": bool(spec.time_breakdowns),
        "breakdown_columns": list(spec.time_breakdowns),
        "period_breakdown_policy": PERIOD_BREAKDOWN_POLICY,
        "engine": spec.engine,
        "weight_configs": [asdict(config) for config in spec.weight_configs],
        "execution_config": asdict(spec.execution_config),
        "tradability_config": asdict(spec.tradability_config),
        "cost_configs": [asdict(config) for config in spec.cost_configs],
        "neutralization_config": asdict(spec.neutralization_config),
        "tensor_config": asdict(spec.tensor_config),
        "tables": {name: int(len(frame)) for name, frame in tables.items()},
        "position_artifact_policy": POSITION_ARTIFACT_POLICY,
        "boundary": spec.domain_notes.get(
            "boundary",
            "No real data, API connection, credentials, paper trading, live trading, "
            "or investment advice is provided by this core evaluation.",
        ),
    }


def _build_evaluation_set_config(
    *,
    spec: EvaluationSpec,
    pnl_return_column: str,
) -> dict[str, object]:
    return {
        "metadata": spec.metadata,
        "domain_notes": spec.domain_notes,
        "schema": {
            "timestamp_column": spec.timestamp_column,
            "asset_column": spec.asset_column,
            "feature_columns": list(spec.feature_columns),
            "label_columns": list(spec.label_columns),
            "pnl_return_column": pnl_return_column,
        },
        "n_groups": spec.n_groups,
        "segment_columns": list(spec.segment_columns),
        "time_breakdowns": list(spec.time_breakdowns),
        "engine": spec.engine,
        "weight_configs": [asdict(config) for config in spec.weight_configs],
        "execution_config": asdict(spec.execution_config),
        "tradability_config": asdict(spec.tradability_config),
        "cost_configs": [asdict(config) for config in spec.cost_configs],
        "neutralization_config": asdict(spec.neutralization_config),
        "tensor_config": asdict(spec.tensor_config),
    }


def _build_evaluation_set_manifest(
    *,
    summary: dict[str, object],
    output_dir: str | Path,
) -> dict[str, object]:
    base = Path(output_dir)
    artifacts = {
        "manifest": "manifest.json",
        "config": "config.json",
        "summary": "summary.json",
        "combinations": "combinations.csv",
        "metrics": {
            "ic": _relative(base, base / "metrics" / "ic.csv"),
            "ic_summary": _relative(base, base / "metrics" / "ic_summary.csv"),
            "grouped_returns": _relative(
                base,
                base / "metrics" / "grouped_returns.csv",
            ),
            "group_spread_summary": _relative(
                base,
                base / "metrics" / "group_spread_summary.csv",
            ),
            "performance_summary": _relative(
                base,
                base / "metrics" / "performance_summary.csv",
            ),
        },
        **_breakdown_artifact_sections(base, summary),
        "portfolio": {
            "positions": _relative(base, base / "portfolio" / "positions.csv"),
            "turnover": _relative(base, base / "portfolio" / "turnover.csv"),
            "returns": _relative(base, base / "portfolio" / "returns.csv"),
            "pnl": _relative(base, base / "portfolio" / "pnl.csv"),
            "cost_breakdown": _relative(
                base,
                base / "portfolio" / "cost_breakdown.csv",
            ),
            "exposure_summary": _relative(
                base,
                base / "portfolio" / "exposure_summary.csv",
            ),
        },
        "diagnostics": {
            "weight_diagnostics": _relative(
                base,
                base / "diagnostics" / "weight_diagnostics.csv",
            ),
            "neutralization_diagnostics": _relative(
                base,
                base / "diagnostics" / "neutralization_diagnostics.csv",
            ),
        },
        "report": "report.md",
    }
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "evaluation_set",
        "evaluation_id": summary["evaluation_id"],
        "project": summary["project"],
        "kind": summary["kind"],
        "domain": summary["domain"],
        "data_stage": summary["data_stage"],
        "uses_external_data": summary["uses_external_data"],
        "schema": summary["schema"],
        "n_combinations": summary["n_combinations"],
        "segment_columns": summary["segment_columns"],
        "breakdown_enabled": summary["breakdown_enabled"],
        "breakdown_columns": summary["breakdown_columns"],
        "period_breakdown_policy": summary["period_breakdown_policy"],
        "engine": summary["engine"],
        "position_artifact_policy": summary["position_artifact_policy"],
        "artifacts": artifacts,
        "display": build_display_manifest(
            artifacts,
            schema_version=ARTIFACT_SCHEMA_VERSION,
        ),
    }


def _render_evaluation_set_report(
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> str:
    lines = [
        "# Evaluation Set Report",
        "",
        "This report summarizes one batch-native evaluation set.",
        "",
        "## Summary",
        "",
        f"- evaluation_id: {summary['evaluation_id']}",
        f"- artifact_schema_version: {summary['artifact_schema_version']}",
        f"- n_combinations: {summary['n_combinations']}",
        f"- engine: {summary['engine']}",
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


def _breakdown_artifact_sections(
    base: Path,
    summary: dict[str, object],
) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    breakdown_columns = summary.get("breakdown_columns")
    if not isinstance(breakdown_columns, list):
        return sections
    for period in breakdown_columns:
        if not isinstance(period, str):
            continue
        directory = f"metrics_by_{period}"
        sections[directory] = {
            f"ic_summary_by_{period}": _relative(
                base,
                base / directory / f"ic_summary_by_{period}.csv",
            ),
            f"group_spread_summary_by_{period}": _relative(
                base,
                base / directory / f"group_spread_summary_by_{period}.csv",
            ),
            f"performance_summary_by_{period}": _relative(
                base,
                base / directory / f"performance_summary_by_{period}.csv",
            ),
        }
    return sections


def _relative(base: Path, path: Path) -> str:
    return path.relative_to(base).as_posix()

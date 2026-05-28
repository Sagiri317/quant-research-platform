"""Display manifest builder for schema 0.7 evaluation-set artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from quant_research_platform.artifacts.display_tables import (
    DISPLAY_PREVIEW_ROWS_DEFAULT,
    build_display_table_preview_descriptor,
)
from quant_research_platform.artifacts.schema_registry import (
    EVALUATION_SET_SCHEMA_VERSION,
    chart_specs,
    get_table_spec,
)

TABLE_LABELS = {
    "combinations": ("组合配置", "dimensions"),
    "ic": ("IC / RankIC", "metrics"),
    "grouped_returns": ("分组收益", "metrics"),
    "evaluation_metrics_summary": ("评价指标汇总", "metrics"),
    "returns": ("组合收益明细", "portfolio"),
    "coverage_summary": ("输入覆盖汇总", "input_quality"),
    "diagnostic_policies": ("诊断策略", "dimensions"),
    "diagnostic_summary": ("诊断总览", "diagnostics"),
    "diagnostic_metrics": ("诊断指标", "diagnostics"),
    "portfolio_assumptions": ("组合假设", "dimensions"),
    "weights": ("目标权重", "portfolio"),
    "tradability_diagnostics": ("可交易性诊断", "diagnostics"),
    "tradability_summary": ("可交易性汇总", "diagnostics"),
    "scenario_metrics": ("场景指标", "diagnostics"),
    "exposure_capacity_summary": ("暴露和容量代理", "diagnostics"),
    "portfolio_risk_summary": ("组合风险代理", "diagnostics"),
    "diagnostic_observations": ("诊断观察", "diagnostics"),
    "ic_breakdown_by_month": ("月度 IC / RankIC", "breakdowns"),
    "evaluation_metrics_summary_by_year": ("年度评价指标", "breakdowns"),
    "evaluation_metrics_summary_by_quarter": ("季度评价指标", "breakdowns"),
    "evaluation_metrics_summary_by_month": ("月度评价指标", "breakdowns"),
}


def build_display_manifest(
    artifacts: Mapping[str, object],
    *,
    schema_version: str,
    default_selection: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build schema 0.7 manifest.display from declared artifacts."""

    if schema_version != EVALUATION_SET_SCHEMA_VERSION:
        raise ValueError("display schema_version must be 0.7.")
    tables = _table_entries(artifacts)
    metadata_items = [
        {
            "kind": "metadata_file",
            "key": "manifest",
            "label": "Manifest manifest.json",
            "path": "manifest.json",
        },
        {
            "kind": "metadata_file",
            "key": "summary",
            "label": "Summary summary.json",
            "path": "summary.json",
        },
        {
            "kind": "metadata_file",
            "key": "run_context",
            "label": "Run context run_context.json",
            "path": "run_context.json",
        },
    ]
    if artifacts.get("report") == "report.md":
        metadata_items.append(
            {
                "kind": "metadata_file",
                "key": "report",
                "label": "研究报告 report.md",
                "path": "report.md",
            }
        )
    table_items = [
        {
            "kind": "table",
            "key": key,
            "label": f"{_table_label(key)} {_format_label_for_path(path)}",
            "path": path,
            "table": key,
        }
        for key, path in _download_table_paths(artifacts).items()
    ]
    all_item_refs = [
        {"kind": item["kind"], "key": item["key"]}
        for item in [*metadata_items, *table_items]
    ]
    table_item_refs = [{"kind": "table", "key": str(table["key"])} for table in tables]
    return {
        "schema_version": schema_version,
        "dimension_source": "dimensions/combinations.csv",
        "default_selection": dict(default_selection or {}),
        "tables": tables,
        "charts": [_chart_entry(chart) for chart in chart_specs()],
        "downloads": {
            "items": [*metadata_items, *table_items],
            "presets": [
                {"key": "all", "label": "全选", "items": all_item_refs},
                {
                    "key": "tables",
                    "label": "仅表格",
                    "items": table_item_refs,
                },
                {
                    "key": "repro",
                    "label": "复现包",
                    "items": [
                        {"kind": "metadata_file", "key": "manifest"},
                        {"kind": "metadata_file", "key": "summary"},
                        {"kind": "metadata_file", "key": "run_context"},
                        *table_item_refs,
                    ],
                },
            ],
        },
    }


def _table_entries(artifacts: Mapping[str, object]) -> list[dict[str, object]]:
    paths = {
        key: path
        for key, path in _table_paths(artifacts).items()
        if get_table_spec(key).display.eligible
    }
    entries = [_table_entry(key, path) for key, path in paths.items()]
    return sorted(
        entries,
        key=lambda entry: (
            entry["section_order"],
            entry["display_order"],
            entry["key"],
        ),
    )


def _table_entry(key: str, path: str) -> dict[str, object]:
    spec = get_table_spec(key)
    label, role = TABLE_LABELS.get(
        key, (key.replace("_", " ").title(), spec.display.section)
    )
    table_format = "parquet" if path.endswith(".parquet") else "csv"
    supports_combination_filter = spec.display.supports_combination_filter
    return {
        "key": key,
        "label": label,
        "group": role,
        "role": role,
        "section": spec.display.section,
        "section_order": spec.display.section_order,
        "display_order": spec.display.display_order,
        "path": path,
        "format": table_format,
        "download_ref": {"kind": "table", "key": key},
        "dimension_source": "dimensions/combinations.csv",
        "join_key": "combination_id" if supports_combination_filter else None,
        "preview_rows": DISPLAY_PREVIEW_ROWS_DEFAULT,
        "supports_combination_filter": supports_combination_filter,
        "conditional": spec.materialization != "required",
        "required": spec.materialization == "required",
        "empty_state": "No rows are available for this table.",
        "preview": build_display_table_preview_descriptor(
            table_key=key,
            supports_combination_filter=supports_combination_filter,
        ),
    }


def _table_paths(artifacts: Mapping[str, object]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for section in artifacts.values():
        if not isinstance(section, Mapping):
            continue
        for key, path in section.items():
            if isinstance(key, str) and isinstance(path, str):
                paths[key] = path
    return paths


def _table_format_label(table: Mapping[str, object]) -> str:
    return "Parquet" if table.get("format") == "parquet" else "CSV"


def _format_label_for_path(path: str) -> str:
    return "Parquet" if path.endswith(".parquet") else "CSV"


def _table_label(key: str) -> str:
    label, _role = TABLE_LABELS.get(
        key,
        (key.replace("_", " ").title(), get_table_spec(key).display.section),
    )
    return label


def _download_table_paths(artifacts: Mapping[str, object]) -> dict[str, str]:
    return {
        key: path
        for key, path in _table_paths(artifacts).items()
        if get_table_spec(key).display.download_eligible
    }


def _chart_entry(chart) -> dict[str, object]:
    return {
        "id": chart.key,
        "title": "累计组合净收益",
        "description": "由组合收益表按组合派生的累计净收益。",
        "section": "primary",
        "kind": chart.kind,
        "source_table_key": chart.source_table_key,
        "source_path": chart.source_path,
        "x": chart.x_column,
        "grouping_column": chart.grouping_column,
        "input_series": chart.input_series,
        "transform": chart.transform,
        "output_series": chart.output_series,
        "supports_combination_filter": chart.supports_combination_filter,
    }


__all__ = ["build_display_manifest"]

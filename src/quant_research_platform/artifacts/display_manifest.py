"""本地 evidence workbench 的 display manifest helper。"""

from __future__ import annotations

from collections.abc import Mapping

TABLE_LABELS = {
    "combinations": ("组合配置", "setup"),
    "ic": ("IC / RankIC", "metrics"),
    "ic_summary": ("IC / RankIC 汇总", "metrics"),
    "grouped_returns": ("分组收益", "metrics"),
    "group_spread_summary": ("分组 Spread 汇总", "metrics"),
    "performance_summary": ("Performance 汇总", "metrics"),
    "exposure_summary": ("敞口汇总", "portfolio"),
    "positions": ("持仓明细", "portfolio"),
    "turnover": ("Turnover 明细", "portfolio"),
    "returns": ("组合收益明细", "portfolio"),
    "pnl": ("PnL 曲线", "portfolio"),
    "cost_breakdown": ("成本拆分", "portfolio"),
    "weight_diagnostics": ("权重诊断", "diagnostics"),
    "neutralization_diagnostics": ("中性化诊断", "diagnostics"),
}

TABLE_ORDER = tuple(TABLE_LABELS)

CHARTS = [
    {
        "id": "pnl",
        "title": "累计组合 PnL",
        "description": "净值曲线和回撤",
        "section": "primary",
        "kind": "line",
        "table": "pnl",
        "x": "timestamp",
        "include_zero": True,
        "series": [
            {
                "key": "cumulative_pnl",
                "label": "cumulative_pnl",
                "color": "#2563eb",
                "format": "percent",
            },
            {
                "key": "drawdown",
                "label": "drawdown",
                "color": "#dc2626",
                "format": "percent",
            },
        ],
    },
    {
        "id": "ic_rankic",
        "title": "IC / RankIC",
        "description": "IC 和 RankIC 的时间序列",
        "section": "primary",
        "kind": "line",
        "table": "ic",
        "x": "timestamp",
        "series": [
            {"key": "ic", "label": "ic", "color": "#2563eb"},
            {"key": "rankic", "label": "rankic", "color": "#059669"},
        ],
    },
    {
        "id": "net_portfolio_return",
        "title": "单期净收益",
        "description": "每期组合净收益",
        "section": "diagnostic",
        "kind": "bar",
        "table": "returns",
        "x": "timestamp",
        "include_zero": True,
        "series": [
            {
                "key": "net_portfolio_return",
                "label": "net_portfolio_return",
                "color": "#2563eb",
            },
        ],
    },
    {
        "id": "turnover",
        "title": "换手率",
        "description": "每期换手强度",
        "section": "diagnostic",
        "kind": "bar",
        "table": "returns",
        "x": "timestamp",
        "include_zero": True,
        "series": [
            {"key": "turnover", "label": "turnover", "color": "#f59e0b"},
        ],
    },
    {
        "id": "cost",
        "title": "单期成本",
        "description": "每期成本拖累",
        "section": "diagnostic",
        "kind": "bar",
        "table": "returns",
        "x": "timestamp",
        "include_zero": True,
        "series": [
            {"key": "cost", "label": "cost", "color": "#dc2626"},
        ],
    },
]


def build_display_manifest(
    artifacts: Mapping[str, object],
    *,
    schema_version: str,
) -> dict[str, object]:
    """从 artifact path contract 构造前端 display contract。"""

    tables = _table_entries(artifacts)
    table_items = [str(table["download_item"]) for table in tables]
    metadata_items = [
        {"item": "report.md", "label": "研究报告 report.md", "kind": "report"},
        {"item": "comparison.csv", "label": "Run 对比 comparison.csv", "kind": "index"},
        {"item": "manifest.json", "label": "Manifest manifest.json", "kind": "json"},
        {"item": "config.json", "label": "Config config.json", "kind": "json"},
        {"item": "summary.json", "label": "Summary summary.json", "kind": "json"},
    ]
    return {
        "schema_version": schema_version,
        "tables": tables,
        "charts": [
            chart
            for chart in CHARTS
            if any(table["key"] == chart["table"] for table in tables)
        ],
        "downloads": {
            "items": [
                *metadata_items,
                *[
                    {
                        "item": str(table["download_item"]),
                        "label": f"{table['label']} CSV",
                        "kind": "table",
                        "table": table["key"],
                    }
                    for table in tables
                ],
            ],
            "presets": [
                {
                    "key": "all",
                    "label": "全选",
                    "items": [item["item"] for item in metadata_items] + table_items,
                },
                {"key": "report", "label": "仅报告", "items": ["report.md"]},
                {
                    "key": "tables",
                    "label": "仅表格",
                    "items": ["comparison.csv", *table_items],
                },
                {
                    "key": "repro",
                    "label": "复现包",
                    "items": [
                        "report.md",
                        "comparison.csv",
                        "manifest.json",
                        "config.json",
                        "summary.json",
                        *table_items,
                    ],
                },
            ],
        },
    }


def _table_entries(artifacts: Mapping[str, object]) -> list[dict[str, object]]:
    paths = _table_paths(artifacts)
    ordered_keys = [key for key in TABLE_ORDER if key in paths]
    extra_keys = sorted(key for key in paths if key not in TABLE_LABELS)
    return [_table_entry(key, paths[key]) for key in [*ordered_keys, *extra_keys]]


def _table_entry(key: str, path: str) -> dict[str, object]:
    label, group = TABLE_LABELS.get(key, (_humanize_key(key), _group_for_path(path)))
    return {
        "key": key,
        "label": label,
        "group": group,
        "path": path,
        "download_item": path,
        "supports_combination_filter": key != "combinations",
    }


def _table_paths(artifacts: Mapping[str, object]) -> dict[str, str]:
    paths: dict[str, str] = {}
    combinations = artifacts.get("combinations")
    if isinstance(combinations, str):
        paths["combinations"] = combinations
    for group in ("metrics", "portfolio", "diagnostics"):
        _extend_section_paths(paths, artifacts.get(group))
    for group, section in artifacts.items():
        if isinstance(group, str) and group.startswith("metrics_by_"):
            _extend_section_paths(paths, section)
    return paths


def _extend_section_paths(paths: dict[str, str], section: object) -> None:
    if not isinstance(section, Mapping):
        return
    for key, path in section.items():
        if isinstance(key, str) and isinstance(path, str):
            paths[key] = path


def _group_for_path(path: str) -> str:
    if path.startswith("portfolio/"):
        return "portfolio"
    if path.startswith("diagnostics/"):
        return "diagnostics"
    if path.startswith("metrics_by_"):
        return "breakdown"
    return "metrics"


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").title()


__all__ = ["build_display_manifest"]

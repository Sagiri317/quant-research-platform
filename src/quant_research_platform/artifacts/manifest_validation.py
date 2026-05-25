"""校验 artifact manifest 和 display 的 helper。"""

from __future__ import annotations

from pathlib import PurePosixPath

import pandas as pd

from quant_research_platform.artifacts.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    BREAKDOWN_PERIODS,
    DISPLAY_METADATA_DOWNLOAD_ITEMS,
    EVALUATION_SET_DIAGNOSTIC_TABLES,
    EVALUATION_SET_INPUT_QUALITY_TABLES,
    EVALUATION_SET_METRIC_TABLES,
    EVALUATION_SET_PORTFOLIO_TABLES,
    EVALUATION_SET_SUMMARY_TABLES,
    OPTIONAL_ARTIFACT_GROUPS,
)
from quant_research_platform.runtime.run_metadata import validate_run_metadata

DISPLAY_PREVIEW_ROWS_MAX = 1000

TRADABILITY_DIAGNOSTIC_COLUMNS = frozenset(
    {
        "timestamp",
        "method",
        "direction",
        "universe_count",
        "eligible_count",
        "can_enter_count",
        "can_exit_count",
        "entry_blocked_count",
        "exit_blocked_count",
        "eligible_rate",
        "can_enter_rate",
        "can_exit_rate",
        "entry_blocked_rate",
        "exit_blocked_rate",
        "fallback_reason",
    }
)


def evaluation_set_csv_paths(artifacts: dict[object, object]) -> dict[str, str]:
    """返回 evaluation-set manifest 声明并通过校验的 table paths."""

    _validate_known_artifact_entries(artifacts)
    paths: dict[str, str] = {}
    dimensions = artifacts.get("dimensions")
    combinations = (
        dimensions.get("combinations") if isinstance(dimensions, dict) else None
    )
    if not isinstance(combinations, str) or not combinations:
        raise ValueError("manifest artifacts must include dimensions.combinations.")
    validate_artifact_relative_path(
        combinations,
        source="artifacts.dimensions.combinations",
    )
    paths["combinations"] = combinations

    for group, names in {
        "metrics": EVALUATION_SET_METRIC_TABLES,
        "portfolio": EVALUATION_SET_PORTFOLIO_TABLES,
        "diagnostics": EVALUATION_SET_DIAGNOSTIC_TABLES,
        "input_quality": EVALUATION_SET_INPUT_QUALITY_TABLES,
        "summaries": EVALUATION_SET_SUMMARY_TABLES,
    }.items():
        section = artifacts.get(group)
        if not isinstance(section, dict):
            raise ValueError(f"manifest artifacts must include {group}.")
        for name in names:
            relative_path = section.get(name)
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"manifest artifacts must include {group}.{name}.")
            validate_artifact_relative_path(
                relative_path,
                source=f"artifacts.{group}.{name}",
            )
            paths[name] = relative_path
    for period in BREAKDOWN_PERIODS:
        group = f"breakdowns/{period}"
        section = artifacts.get(group)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"manifest artifacts {group} must be an object.")
        for name in (
            f"ic_summary_by_{period}",
            f"group_spread_summary_by_{period}",
            f"performance_summary_by_{period}",
        ):
            relative_path = section.get(name)
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"manifest artifacts must include {group}.{name}.")
            validate_artifact_relative_path(
                relative_path,
                source=f"artifacts.{group}.{name}",
            )
            paths[name] = relative_path
    return paths


def _validate_known_artifact_entries(artifacts: dict[object, object]) -> None:
    required_top_level = {
        "manifest",
        "config",
        "lineage",
        "summary",
        "dimensions",
        "metrics",
        "portfolio",
        "diagnostics",
        "input_quality",
        "summaries",
        "report",
    }
    allowed_top_level = {*required_top_level, *OPTIONAL_ARTIFACT_GROUPS}
    top_level_keys = {key for key in artifacts if isinstance(key, str)}
    missing_groups = sorted(required_top_level.difference(top_level_keys))
    if missing_groups:
        raise ValueError(
            "manifest artifacts missing required artifact groups: "
            f"{missing_groups}."
        )
    unknown_groups = sorted(top_level_keys.difference(allowed_top_level))
    if unknown_groups:
        raise ValueError(
            "manifest artifacts contain unknown artifact groups: "
            f"{unknown_groups}."
        )
    for key in ("manifest", "config", "lineage", "summary", "report"):
        relative_path = artifacts.get(key)
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"manifest artifacts must include {key}.")
        validate_artifact_relative_path(relative_path, source=f"artifacts.{key}")

    _require_no_unknown_section_keys(artifacts, "dimensions", {"combinations"})
    _require_no_unknown_section_keys(
        artifacts,
        "metrics",
        set(EVALUATION_SET_METRIC_TABLES),
    )
    _require_no_unknown_section_keys(
        artifacts,
        "portfolio",
        set(EVALUATION_SET_PORTFOLIO_TABLES),
    )
    _require_no_unknown_section_keys(
        artifacts,
        "diagnostics",
        set(EVALUATION_SET_DIAGNOSTIC_TABLES),
    )
    _require_no_unknown_section_keys(
        artifacts,
        "input_quality",
        set(EVALUATION_SET_INPUT_QUALITY_TABLES),
    )
    _require_no_unknown_section_keys(
        artifacts,
        "summaries",
        set(EVALUATION_SET_SUMMARY_TABLES),
    )
    for period in BREAKDOWN_PERIODS:
        group = f"breakdowns/{period}"
        if group not in artifacts:
            continue
        _require_no_unknown_section_keys(
            artifacts,
            group,
            {
                f"ic_summary_by_{period}",
                f"group_spread_summary_by_{period}",
                f"performance_summary_by_{period}",
            },
        )


def _require_no_unknown_section_keys(
    artifacts: dict[object, object],
    group: str,
    allowed_keys: set[str],
) -> None:
    section = artifacts.get(group)
    if not isinstance(section, dict):
        return
    keys = {key for key in section if isinstance(key, str)}
    unknown = sorted(keys.difference(allowed_keys))
    if unknown:
        raise ValueError(f"manifest contains unknown artifacts.{group}: {unknown}.")


def validate_evaluation_set_table_columns(tables: dict[str, pd.DataFrame]) -> None:
    """校验当前 evaluation-set table columns，并拒绝 retired fields。"""

    combination_columns = {
        "combination_id",
        "feature",
        "label",
        "kernel",
        "portfolio_mode",
    }
    fact_key = {"combination_id"}
    retired_columns = {
        "total_gross_portfolio_return",
        "total_net_portfolio_return",
        "weight_id",
        "cost_id",
        "position",
        "pnl_return_column",
    }
    for name, frame in tables.items():
        stale = sorted(retired_columns.intersection(frame.columns))
        if stale:
            raise ValueError(f"{name} contains retired columns: {stale}.")
        if name == "combinations":
            required = combination_columns
            extra_profiles = {
                column
                for column in frame.columns
                if column.endswith("_profile_id") or column.endswith("_policy_id")
            }
            if extra_profiles:
                raise ValueError(
                    "combinations contains first-stage run-level profile/policy "
                    f"columns: {sorted(extra_profiles)}."
                )
            if set(frame.columns) != combination_columns:
                extras = sorted(set(frame.columns).difference(combination_columns))
                raise ValueError(
                    "combinations must contain only first-stage dynamic dimensions; "
                    f"extra={extras}."
                )
        elif name == "performance_summary":
            required = {
                *fact_key,
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
            }
        elif name == "tradability_diagnostics":
            required = {*fact_key, *TRADABILITY_DIAGNOSTIC_COLUMNS}
        elif name == "ic":
            required = {*fact_key, "timestamp", "ic", "rankic", "n_obs"}
        elif name == "ic_summary":
            required = {
                *fact_key,
                "mean_ic",
                "std_ic",
                "icir",
                "positive_ic_rate",
                "mean_rankic",
                "std_rankic",
                "rankicir",
                "positive_rankic_rate",
            }
        elif name == "grouped_returns":
            required = {
                *fact_key,
                "timestamp",
                "group",
                "mean_forward_return",
                "n_obs",
            }
        elif name == "group_spread_summary":
            required = {
                *fact_key,
                "top_group_mean_return",
                "bottom_group_mean_return",
                "top_bottom_spread_mean",
                "top_bottom_spread_hit_rate",
                "group_monotonicity_score",
            }
        elif name == "returns":
            required = {
                *fact_key,
                "timestamp",
                "portfolio_mode",
                "long_return",
                "short_return",
                "gross_portfolio_return",
                "turnover",
                "cost",
                "net_portfolio_return",
                "long_count",
                "short_count",
                "return_source_column",
            }
        elif name == "cumulative_returns":
            required = {
                *fact_key,
                "timestamp",
                "gross_return",
                "net_return",
                "cost",
                "cumulative_gross_return",
                "cumulative_net_return",
                "cumulative_cost",
                "high_water_mark",
                "drawdown",
            }
        elif name == "weights":
            required = {*fact_key, "timestamp", "asset", "target_weight"}
        elif name == "label_availability":
            required = {"label", "available_rows", "missing_rows", "coverage_rate"}
        elif name == "factor_coverage":
            required = {"feature", "available_rows", "missing_rows", "coverage_rate"}
        elif name == "tradability_coverage":
            required = {"field", "available_rows", "missing_rows", "coverage_rate"}
        elif name == "evaluation_summary":
            required = {
                *combination_columns,
                "mean_ic",
                "final_cumulative_net_return",
                "mean_turnover",
                "total_cost",
            }
        else:
            required = set()
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}.")


def validate_artifact_schema_version(
    payload: dict[str, object],
    *,
    source: str,
) -> None:
    """要求使用当前 artifact schema version。"""

    version = payload.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{source} has unsupported artifact_schema_version {version!r}; "
            f"expected {ARTIFACT_SCHEMA_VERSION!r}."
        )


def validate_evaluation_set_payload(
    payload: dict[str, object],
    *,
    source: str,
) -> None:
    """要求使用当前 evaluation-set metadata。"""

    artifact_type = payload.get("artifact_type")
    if artifact_type != "evaluation_set":
        raise ValueError(
            f"{source} has unsupported artifact_type {artifact_type!r}; "
            "expected 'evaluation_set'."
        )
    validate_run_metadata(payload, source=source)


def validate_display_manifest(manifest: dict[str, object]) -> None:
    """按 artifact paths 和 display schema 校验 manifest.display。"""

    display = manifest.get("display")
    if not isinstance(display, dict):
        raise ValueError("manifest.json must contain a display object.")
    if display.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("manifest display schema_version must be supported.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    artifact_paths = evaluation_set_csv_paths(artifacts)

    tables = display.get("tables")
    if not isinstance(tables, list):
        raise ValueError("manifest display.tables must be a list.")
    table_paths: dict[str, str] = {}
    table_downloads: dict[str, str] = {}
    for table in tables:
        if not isinstance(table, dict):
            raise ValueError("manifest display.tables entries must be objects.")
        key = table.get("key")
        path = table.get("path")
        download_item = table.get("download_item")
        table_format = table.get("format")
        role = table.get("role")
        if not isinstance(key, str) or not key:
            raise ValueError("manifest display table key must be a non-empty string.")
        if key not in artifact_paths:
            raise ValueError(f"manifest display table {key!r} is not an artifact.")
        if path != artifact_paths[key]:
            raise ValueError(f"manifest display table {key!r} path is inconsistent.")
        if download_item != artifact_paths[key]:
            raise ValueError(
                f"manifest display table {key!r} download_item is inconsistent."
            )
        if table_format not in {"csv", "parquet", "json", "markdown"}:
            raise ValueError(f"manifest display table {key!r} format is unsupported.")
        if key == "weights" and table_format != "parquet":
            raise ValueError("manifest display weights table must be parquet.")
        if not isinstance(role, str) or not role:
            raise ValueError(f"manifest display table {key!r} role is required.")
        if key == "combinations" or key in EVALUATION_SET_INPUT_QUALITY_TABLES:
            if table.get("join_key") is not None:
                raise ValueError(
                    f"manifest display table {key!r} join_key must be absent."
                )
            if table.get("supports_combination_filter") is not False:
                raise ValueError(
                    f"manifest display table {key!r} must not support "
                    "combination filter."
                )
        else:
            if table.get("join_key") != "combination_id":
                raise ValueError(
                    f"manifest display table {key!r} join_key must be combination_id."
                )
            if table.get("supports_combination_filter") is not True:
                raise ValueError(
                    f"manifest display table {key!r} must support combination filter."
                )
        preview_rows = table.get("preview_rows")
        if (
            not isinstance(preview_rows, int)
            or preview_rows <= 0
            or preview_rows > DISPLAY_PREVIEW_ROWS_MAX
        ):
            raise ValueError(f"manifest display table {key!r} preview_rows invalid.")
        validate_display_path(path, source=f"display table {key}")
        table_paths[key] = path
        table_downloads[key] = download_item
    if table_paths != artifact_paths:
        missing = sorted(set(artifact_paths).difference(table_paths))
        extra = sorted(set(table_paths).difference(artifact_paths))
        raise ValueError(
            "manifest display.tables must cover artifact CSV tables; "
            f"missing={missing}, extra={extra}."
        )

    downloads = display.get("downloads")
    if not isinstance(downloads, dict) or not isinstance(downloads.get("items"), list):
        raise ValueError("manifest display.downloads.items must be a list.")
    download_items: set[str] = set()
    for option in downloads["items"]:
        if not isinstance(option, dict):
            raise ValueError(
                "manifest display.downloads.items entries must be objects."
            )
        item = option.get("item")
        if not isinstance(item, str) or not item:
            raise ValueError("manifest display download item must be non-empty.")
        validate_display_path(item, source=f"display download item {item}")
        table_key = option.get("table")
        if table_key is not None:
            if not isinstance(table_key, str) or table_key not in table_paths:
                raise ValueError(
                    f"manifest display download item {item!r} references "
                    "an unknown table."
                )
            if item != table_downloads[table_key]:
                raise ValueError(
                    f"manifest display download item {item!r} is inconsistent "
                    f"with table {table_key!r}."
                )
        elif item not in DISPLAY_METADATA_DOWNLOAD_ITEMS:
            raise ValueError(
                f"manifest display download item {item!r} is not a known "
                "metadata item or table artifact."
            )
        download_items.add(item)
    missing_downloads = sorted(set(table_downloads.values()).difference(download_items))
    if missing_downloads:
        raise ValueError(
            "manifest display.downloads.items must include every display table; "
            f"missing={missing_downloads}."
        )

    presets = downloads.get("presets")
    if presets is not None:
        if not isinstance(presets, list):
            raise ValueError("manifest display.downloads.presets must be a list.")
        for preset in presets:
            if not isinstance(preset, dict):
                raise ValueError(
                    "manifest display.downloads.presets entries must be objects."
                )
            preset_items = preset.get("items")
            if not isinstance(preset_items, list):
                raise ValueError(
                    "manifest display download preset items must be a list."
                )
            for item in preset_items:
                if not isinstance(item, str) or item not in download_items:
                    raise ValueError(
                        f"manifest display download preset references "
                        f"unknown item {item!r}."
                    )

    charts = display.get("charts")
    if not isinstance(charts, list):
        raise ValueError("manifest display.charts must be a list.")
    for chart in charts:
        if not isinstance(chart, dict):
            raise ValueError("manifest display.charts entries must be objects.")
        kind = chart.get("kind")
        if kind not in {"line", "bar"}:
            raise ValueError(
                f"manifest display chart {chart.get('id')!r} has unsupported kind."
            )
        table_key = chart.get("table")
        if not isinstance(table_key, str) or table_key not in table_paths:
            raise ValueError(
                f"manifest display chart {chart.get('id')!r} references "
                "an unknown table."
            )
        if not isinstance(chart.get("x"), str) or not chart["x"]:
            raise ValueError("manifest display chart x must be a non-empty string.")
        series = chart.get("series")
        if not isinstance(series, list) or not series:
            raise ValueError("manifest display chart series must be a non-empty list.")
        for entry in series:
            if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                raise ValueError(
                    "manifest display chart series entries must include key."
                )


def validate_display_chart_columns(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
    """校验 display chart 引用的 columns 存在于已加载 tables。"""

    display = manifest.get("display")
    if not isinstance(display, dict):
        return
    charts = display.get("charts")
    if not isinstance(charts, list):
        return
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        table_key = chart.get("table")
        if not isinstance(table_key, str):
            continue
        frame = tables.get(table_key)
        if frame is None:
            continue
        x_key = chart.get("x")
        if not isinstance(x_key, str) or x_key not in frame.columns:
            raise ValueError(
                f"manifest display chart {chart.get('id')!r} references "
                f"missing x column {x_key!r}."
            )
        series = chart.get("series")
        if not isinstance(series, list):
            continue
        for entry in series:
            if not isinstance(entry, dict):
                continue
            series_key = entry.get("key")
            if not isinstance(series_key, str) or series_key not in frame.columns:
                raise ValueError(
                    f"manifest display chart {chart.get('id')!r} references "
                    f"missing series column {series_key!r}."
                )


def display_chart_table_names(manifest: dict[str, object]) -> set[str]:
    """返回 display charts 引用的 table names。"""

    display = manifest.get("display")
    if not isinstance(display, dict):
        return set()
    charts = display.get("charts")
    if not isinstance(charts, list):
        return set()
    names: set[str] = set()
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        table_key = chart.get("table")
        if isinstance(table_key, str):
            names.add(table_key)
    return names


def validate_artifact_relative_path(path: str, *, source: str) -> None:
    validate_relative_posix_path(path, source=source)


def validate_display_path(path: str, *, source: str) -> None:
    validate_relative_posix_path(path, source=source)


def validate_relative_posix_path(path: str, *, source: str) -> None:
    """拒绝绝对路径、父级跳转、空路径或平台专属路径。"""

    parsed = PurePosixPath(path)
    if (
        "\\" in path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"manifest {source} 路径不安全 / unsafe path: {path!r}.")


__all__ = [
    "display_chart_table_names",
    "evaluation_set_csv_paths",
    "validate_artifact_relative_path",
    "validate_artifact_schema_version",
    "validate_display_chart_columns",
    "validate_display_manifest",
    "validate_display_path",
    "validate_evaluation_set_payload",
    "validate_evaluation_set_table_columns",
    "validate_relative_posix_path",
]

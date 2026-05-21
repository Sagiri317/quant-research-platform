"""File artifact writer for reproducible research runs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pandas as pd
from pydantic import BaseModel, ConfigDict

from quant_research_platform.core.contracts import validate_run_metadata
from quant_research_platform.core.profiling import measure_stage

ARTIFACT_SCHEMA_VERSION = "0.5"


class ArtifactOutputs(BaseModel):
    """Paths written by a research run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    output_dir: Path
    summary_path: Path
    report_path: Path
    metric_paths: dict[str, Path]
    manifest_path: Path | None = None
    config_path: Path | None = None


class LoadedEvaluationSet(BaseModel):
    """Validated evaluation-set artifact contents."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    output_dir: Path
    manifest: dict[str, object]
    config: dict[str, object]
    summary: dict[str, object]
    tables: dict[str, pd.DataFrame]
    report: str


def write_run_artifacts(
    output_dir: str | Path,
    summary: dict[str, object],
    metrics: dict[str, pd.DataFrame],
) -> ArtifactOutputs:
    """Write summary JSON, metric CSVs, and a compact Markdown report."""

    destination = Path(output_dir)
    _timed_artifact_operation(
        "directory_creation",
        lambda: destination.mkdir(parents=True, exist_ok=True),
    )

    summary_path = destination / "summary.json"
    _timed_artifact_operation(
        "json_writing",
        lambda: summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        ),
    )

    metric_paths: dict[str, Path] = {}
    for name, frame in metrics.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"metric {name!r} must be a pandas DataFrame.")
        path = destination / f"{name}.csv"
        _timed_artifact_operation(
            "csv_writing",
            lambda frame=frame, path=path: frame.to_csv(path, index=False),
        )
        metric_paths[name] = path

    report_path = destination / "report.md"
    _timed_artifact_operation(
        "report_writing",
        lambda: report_path.write_text(
            _render_report(summary, metrics),
            encoding="utf-8",
        ),
    )

    return ArtifactOutputs(
        output_dir=destination,
        summary_path=summary_path,
        report_path=report_path,
        metric_paths=metric_paths,
    )


def write_evaluation_set_artifacts(
    output_dir: str | Path,
    *,
    manifest: dict[str, object],
    config: dict[str, object],
    summary: dict[str, object],
    tables: dict[str, pd.DataFrame],
    report: str,
) -> ArtifactOutputs:
    """Write the batch-native evaluation-set artifact contract."""

    destination = Path(output_dir)
    metrics_dir = destination / "metrics"
    portfolio_dir = destination / "portfolio"
    diagnostics_dir = destination / "diagnostics"
    for directory in (destination, metrics_dir, portfolio_dir, diagnostics_dir):
        _timed_artifact_operation(
            "directory_creation",
            lambda directory=directory: directory.mkdir(parents=True, exist_ok=True),
        )

    manifest_path = destination / "manifest.json"
    config_path = destination / "config.json"
    summary_path = destination / "summary.json"
    report_path = destination / "report.md"

    for path, payload in (
        (manifest_path, manifest),
        (config_path, config),
        (summary_path, summary),
    ):
        _timed_artifact_operation(
            "json_writing",
            lambda path=path, payload=payload: path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            ),
        )

    metric_paths: dict[str, Path] = {}
    for name, frame in tables.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"table {name!r} must be a pandas DataFrame.")
        path = _evaluation_set_table_path(
            name=name,
            output_dir=destination,
            metrics_dir=metrics_dir,
            portfolio_dir=portfolio_dir,
            diagnostics_dir=diagnostics_dir,
        )
        _timed_artifact_operation(
            "directory_creation",
            lambda path=path: path.parent.mkdir(parents=True, exist_ok=True),
        )
        _timed_artifact_operation(
            "csv_writing",
            lambda frame=frame, path=path: frame.to_csv(path, index=False),
        )
        metric_paths[name] = path

    _timed_artifact_operation(
        "report_writing",
        lambda: report_path.write_text(report, encoding="utf-8"),
    )
    return ArtifactOutputs(
        output_dir=destination,
        summary_path=summary_path,
        report_path=report_path,
        metric_paths=metric_paths,
        manifest_path=manifest_path,
        config_path=config_path,
    )


def load_evaluation_set(
    output_dir: str | Path,
    *,
    table_names: set[str] | None = None,
) -> LoadedEvaluationSet:
    """Read and validate a batch-native evaluation-set artifact directory."""

    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    config_path = destination / "config.json"
    summary_path = destination / "summary.json"
    for path in (manifest_path, config_path, summary_path):
        _require_file(path, base=destination)

    manifest = _read_json(manifest_path)
    config = _read_json(config_path)
    summary = _read_json(summary_path)
    _validate_artifact_schema_version(manifest, source="manifest.json")
    _validate_artifact_schema_version(summary, source="summary.json")
    _validate_evaluation_set_payload(manifest, source="manifest.json")
    _validate_evaluation_set_payload(summary, source="summary.json")
    config_metadata = config.get("metadata")
    if not isinstance(config_metadata, dict):
        raise ValueError("config.json must contain a metadata object.")
    validate_run_metadata(config_metadata, source="config.json metadata")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    table_paths = _evaluation_set_csv_paths(artifacts)
    validate_display_manifest(manifest)
    if table_names is not None:
        missing_tables = sorted(table_names.difference(table_paths))
        if missing_tables:
            raise ValueError(f"manifest tables are missing: {missing_tables}.")
        table_paths = {
            name: relative_path
            for name, relative_path in table_paths.items()
            if name in table_names
        }
    tables = {
        name: pd.read_csv(_require_file(destination / relative_path, base=destination))
        for name, relative_path in table_paths.items()
    }
    _validate_evaluation_set_table_columns(tables)
    if _display_chart_table_names(manifest).issubset(tables):
        _validate_display_chart_columns(manifest, tables)

    report_relative = artifacts.get("report")
    if not isinstance(report_relative, str) or not report_relative:
        raise ValueError("manifest artifacts must include report.")
    _validate_artifact_relative_path(report_relative, source="artifacts.report")
    report_path = _require_file(destination / report_relative, base=destination)
    return LoadedEvaluationSet(
        output_dir=destination,
        manifest=manifest,
        config=config,
        summary=summary,
        tables=tables,
        report=report_path.read_text(encoding="utf-8"),
    )


def _evaluation_set_table_path(
    *,
    name: str,
    output_dir: Path,
    metrics_dir: Path,
    portfolio_dir: Path,
    diagnostics_dir: Path,
) -> Path:
    if name == "combinations":
        return output_dir / "combinations.csv"
    if name in {
        "ic",
        "ic_summary",
        "grouped_returns",
        "group_spread_summary",
        "performance_summary",
    }:
        return metrics_dir / f"{name}.csv"
    if name in {
        "positions",
        "turnover",
        "returns",
        "pnl",
        "cost_breakdown",
        "exposure_summary",
    }:
        return portfolio_dir / f"{name}.csv"
    if name in {"weight_diagnostics", "neutralization_diagnostics"}:
        return diagnostics_dir / f"{name}.csv"
    for period in ("year", "quarter", "month"):
        if name.endswith(f"_by_{period}"):
            return output_dir / f"metrics_by_{period}" / f"{name}.csv"
    return metrics_dir / f"{name}.csv"


def _timed_artifact_operation(category: str, operation: Callable[[], object]) -> object:
    return measure_stage("artifact", category, operation)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def _require_file(path: Path, *, base: Path) -> Path:
    base_path = base.resolve()
    candidate = path.resolve()
    if candidate != base_path and base_path not in candidate.parents:
        raise ValueError(f"artifact path is unsafe: {path}.")
    if not candidate.is_file():
        try:
            missing = candidate.relative_to(base_path).as_posix()
        except ValueError:
            missing = str(path)
        raise FileNotFoundError(missing)
    return candidate


def _evaluation_set_csv_paths(artifacts: dict[object, object]) -> dict[str, str]:
    paths: dict[str, str] = {}
    combinations = artifacts.get("combinations")
    if not isinstance(combinations, str) or not combinations:
        raise ValueError("manifest artifacts must include combinations.")
    _validate_artifact_relative_path(combinations, source="artifacts.combinations")
    paths["combinations"] = combinations

    for group, names in {
        "metrics": (
            "ic",
            "ic_summary",
            "grouped_returns",
            "group_spread_summary",
            "performance_summary",
        ),
        "portfolio": (
            "positions",
            "turnover",
            "returns",
            "pnl",
            "cost_breakdown",
            "exposure_summary",
        ),
        "diagnostics": (
            "weight_diagnostics",
            "neutralization_diagnostics",
        ),
    }.items():
        section = artifacts.get(group)
        if not isinstance(section, dict):
            raise ValueError(f"manifest artifacts must include {group}.")
        for name in names:
            relative_path = section.get(name)
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"manifest artifacts must include {group}.{name}.")
            _validate_artifact_relative_path(
                relative_path,
                source=f"artifacts.{group}.{name}",
            )
            paths[name] = relative_path
    for period in ("year", "quarter", "month"):
        group = f"metrics_by_{period}"
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
            _validate_artifact_relative_path(
                relative_path,
                source=f"artifacts.{group}.{name}",
            )
            paths[name] = relative_path
    return paths


def _validate_evaluation_set_table_columns(tables: dict[str, pd.DataFrame]) -> None:
    dimension_columns = {"combination_id", "feature", "label", "weight_id", "cost_id"}
    retired_columns = {
        "total_gross_portfolio_return",
        "total_net_portfolio_return",
    }
    for name, frame in tables.items():
        stale = sorted(retired_columns.intersection(frame.columns))
        if stale:
            raise ValueError(f"{name} contains retired columns: {stale}.")
        if name == "combinations":
            required = {
                *dimension_columns,
                "weight_method",
                "direction",
                "cost_bps",
            }
        elif name == "performance_summary":
            required = {
                *dimension_columns,
                "final_cumulative_gross_return",
                "final_cumulative_net_return",
            }
        elif name in {
            "ic",
            "ic_summary",
            "grouped_returns",
            "group_spread_summary",
            "positions",
            "turnover",
            "returns",
            "pnl",
            "cost_breakdown",
            "exposure_summary",
            "weight_diagnostics",
            "neutralization_diagnostics",
        }:
            required = dimension_columns
        else:
            required = set()
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}.")


def _validate_artifact_schema_version(
    payload: dict[str, object],
    *,
    source: str,
) -> None:
    version = payload.get("artifact_schema_version")
    if version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{source} has unsupported artifact_schema_version {version!r}; "
            f"expected {ARTIFACT_SCHEMA_VERSION!r}."
        )


def _validate_evaluation_set_payload(
    payload: dict[str, object],
    *,
    source: str,
) -> None:
    artifact_type = payload.get("artifact_type")
    if artifact_type != "evaluation_set":
        raise ValueError(
            f"{source} has unsupported artifact_type {artifact_type!r}; "
            "expected 'evaluation_set'."
        )
    validate_run_metadata(payload, source=source)


DISPLAY_METADATA_DOWNLOAD_ITEMS = {
    "report.md",
    "comparison.csv",
    "manifest.json",
    "config.json",
    "summary.json",
}


def validate_display_manifest(manifest: dict[str, object]) -> None:
    display = manifest.get("display")
    if not isinstance(display, dict):
        raise ValueError("manifest.json must contain a display object.")
    if display.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("manifest display schema_version must be supported.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest.json must contain an artifacts object.")
    artifact_paths = _evaluation_set_csv_paths(artifacts)

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
        _validate_display_path(path, source=f"display table {key}")
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
        _validate_display_path(item, source=f"display download item {item}")
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


def _validate_display_chart_columns(
    manifest: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> None:
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


def _display_chart_table_names(manifest: dict[str, object]) -> set[str]:
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


def _validate_artifact_relative_path(path: str, *, source: str) -> None:
    _validate_relative_posix_path(path, source=source)


def _validate_display_path(path: str, *, source: str) -> None:
    _validate_relative_posix_path(path, source=source)


def _validate_relative_posix_path(path: str, *, source: str) -> None:
    parsed = PurePosixPath(path)
    if (
        "\\" in path
        or parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError(f"manifest {source} 路径不安全 / unsafe path: {path!r}.")


def _render_report(
    summary: dict[str, object],
    metrics: dict[str, pd.DataFrame],
) -> str:
    lines = [
        "# Quant Research Core Synthetic Demo",
        "",
        "This report is generated from synthetic/toy data only.",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Metrics", ""])
    for name, frame in metrics.items():
        lines.append(f"- `{name}`: {len(frame)} rows")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "No real data, API connection, credentials, paper trading, or live trading "
            "is used in this run.",
            "",
        ]
    )
    return "\n".join(lines)

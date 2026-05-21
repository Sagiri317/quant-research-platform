"""Pipeline run snapshot writer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_research_platform.markets.a_share.pipeline.config import (
    PipelineConfig,
    ValidationIssue,
    validate_pipeline_config,
)


class PipelineSnapshotError(RuntimeError):
    """Raised when a pipeline snapshot cannot be written."""


@dataclass(frozen=True)
class PipelineSnapshot:
    run_dir: Path
    warnings: list[ValidationIssue]


def write_run_snapshot(
    config: PipelineConfig, *, overwrite: bool = False
) -> PipelineSnapshot:
    validation = validate_pipeline_config(config)
    if validation.has_blocking:
        codes = ", ".join(issue.code for issue in validation.blocking)
        raise PipelineSnapshotError(f"blocking config issues prevent snapshot: {codes}")

    run_dir = _resolve_run_dir(config.output_root, config.run_id, overwrite=overwrite)
    run_dir.mkdir(parents=True, exist_ok=overwrite)

    (run_dir / "run_config.yaml").write_text(
        _format_yaml(config.to_snapshot_dict()),
        encoding="utf-8",
    )
    (run_dir / "warnings.json").write_text(
        json.dumps([asdict(issue) for issue in validation.warnings], indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(
        _format_summary(config, validation.warnings),
        encoding="utf-8",
    )
    return PipelineSnapshot(run_dir=run_dir, warnings=validation.warnings)


def _resolve_run_dir(output_root: Path, run_id: str, *, overwrite: bool) -> Path:
    base = output_root / run_id
    if overwrite or not base.exists():
        return base

    for _ in range(100):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        candidate = output_root / f"{run_id}_{stamp}"
        if not candidate.exists():
            return candidate
    raise PipelineSnapshotError(f"could not create unique run directory for {run_id}")


def _format_yaml(values: dict[str, Any]) -> str:
    lines = _format_mapping(values, indent=0)
    return "\n".join(lines) + "\n"


def _format_mapping(values: dict[str, Any], *, indent: int) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in values.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_format_mapping(value, indent=indent + 2))
        elif isinstance(value, list):
            lines.extend(_format_list(key, value, indent=indent))
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines


def _format_list(key: str, values: list[Any], *, indent: int) -> list[str]:
    prefix = " " * indent
    if not values:
        return [f"{prefix}{key}: []"]

    lines = [f"{prefix}{key}:"]
    for value in values:
        if isinstance(value, dict):
            lines.append(f"{prefix}  -")
            lines.extend(_format_mapping(value, indent=indent + 4))
        else:
            lines.append(f"{prefix}  - {value}")
    return lines


def _format_summary(config: PipelineConfig, warnings: list[ValidationIssue]) -> str:
    lines = [
        f"# Pipeline Run {config.run_id}",
        "",
        f"- market: {config.market}",
        f"- date range: {config.date_start} to {config.date_end}",
        f"- factor_template_id: {config.factor_template_id}",
        f"- portfolio_mode: {config.portfolio_mode}",
        f"- warnings: {len(warnings)}",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {issue.code}: {issue.message}" for issue in warnings)
        lines.append("")
    return "\n".join(lines)

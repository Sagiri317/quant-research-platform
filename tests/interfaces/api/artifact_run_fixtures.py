from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.display_manifest import build_display_manifest
from tests.artifacts.artifact_fixtures import write_evaluation_set as _write_artifact


def write_evaluation_set(
    output_dir: Path,
    *,
    evaluation_id: str = "run_a",
    run_id: str | None = None,
    project: str = "quant-research-platform",
    kind: str = "factor",
    domain: str = "evaluation_synthetic",
    data_stage: str = "synthetic_only",
    uses_external_data: bool = False,
    ic_values: list[float] | None = None,
    rankic_values: list[float] | None = None,
    net_returns: list[float] | None = None,
    turnover: list[float] | None = None,
    cost: list[float] | None = None,
    n_rows_evaluated: int = 12,
    time_breakdowns: tuple[str, ...] = (),
) -> None:
    """Write the schema 0.7 artifact fixture used by API tests."""

    _write_artifact(
        output_dir,
        evaluation_id=evaluation_id,
        run_id=run_id,
        project=project,
        kind=kind,
        domain=domain,
        data_stage=data_stage,
        uses_external_data=uses_external_data,
        ic_values=ic_values,
        rankic_values=rankic_values,
        net_returns=net_returns,
        turnover=turnover,
        cost=cost,
        n_rows_evaluated=n_rows_evaluated,
    )
    if time_breakdowns:
        _add_breakdown_markers(output_dir, time_breakdowns)
    _add_weights_marker(output_dir)


def _refresh_display(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"] = build_display_manifest(
        manifest["artifacts"],
        schema_version=manifest["artifact_schema_version"],
        default_selection=manifest["display"]["default_selection"],
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _add_weights_marker(output_dir: Path) -> None:
    directory = output_dir / "portfolio"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "combination_id": ["factor__forward_return__dataframe__long_short"],
            "timestamp": ["2026-01-01"],
            "asset": ["asset_001"],
            "target_weight": [1.0],
        }
    ).to_parquet(directory / "weights.parquet", index=False)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].setdefault("portfolio", {})["weights"] = (
        "portfolio/weights.parquet"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_display(output_dir)


def _add_breakdown_markers(output_dir: Path, time_breakdowns: tuple[str, ...]) -> None:
    # The shared artifact fixture is intentionally minimal; API tests that need
    # breakdown routing only need the display paths and tiny valid CSV files.
    for breakdown in time_breakdowns:
        if breakdown != "year":
            continue
        directory = output_dir / "breakdowns" / "year"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "evaluation_metrics_summary.csv").write_text(
            "combination_id,period_type,period,period_start,period_end,"
            "metric_family,metric_name,metric_value,unit,n_observations,"
            "diagnostic_status,status_reason,warning\n"
            "factor__forward_return__dataframe__long_short,year,2026,"
            "2026-01-01,2026-12-31,portfolio,final_cumulative_net_return,"
            "0.029,ratio,1,ok,,\n",
            encoding="utf-8",
        )
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
        artifacts["breakdowns"] = {
            "evaluation_metrics_summary_by_year": (
                "breakdowns/year/evaluation_metrics_summary.csv"
            ),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _refresh_display(output_dir)

from __future__ import annotations

import json
from pathlib import PurePosixPath

import pandas as pd
import pytest

from quant_research_platform.api.run_index import load_run_index
from quant_research_platform.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    load_evaluation_set,
)
from tests.artifacts.helpers import write_evaluation_set


def test_load_evaluation_set_reads_manifest_summary_tables_and_report(tmp_path):
    write_evaluation_set(tmp_path)

    loaded = load_evaluation_set(tmp_path)

    assert loaded.output_dir == tmp_path
    assert loaded.manifest["evaluation_id"] == "run_a"
    assert loaded.manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert loaded.manifest["artifact_type"] == "evaluation_set"
    assert loaded.manifest["domain"] == "core_synthetic"
    assert loaded.manifest["data_stage"] == "synthetic_only"
    assert loaded.manifest["uses_external_data"] is False
    assert loaded.summary["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert loaded.summary["domain"] == "core_synthetic"
    assert loaded.summary["uses_external_data"] is False
    assert loaded.summary["n_rows_evaluated"] == 12
    assert set(loaded.tables) == {
        "combinations",
        "ic",
        "ic_summary",
        "grouped_returns",
        "group_spread_summary",
        "performance_summary",
        "exposure_summary",
        "positions",
        "turnover",
        "returns",
        "pnl",
        "cost_breakdown",
        "weight_diagnostics",
        "neutralization_diagnostics",
    }
    assert loaded.tables["ic"]["ic"].tolist() == [0.4]
    assert loaded.report == "# Report\n"


def test_manifest_artifact_paths_are_portable_relative_paths(tmp_path):
    write_evaluation_set(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = _collect_artifact_paths(manifest["artifacts"])

    assert artifact_paths
    for path in artifact_paths:
        parsed = PurePosixPath(path)
        assert not parsed.is_absolute()
        assert ":" not in path
        assert "\\" not in path


def test_load_evaluation_set_reports_missing_contract_file(tmp_path):
    write_evaluation_set(tmp_path)
    (tmp_path / "portfolio" / "returns.csv").unlink()

    with pytest.raises(FileNotFoundError, match="portfolio/returns.csv"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_reports_missing_manifest_section(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["portfolio"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="portfolio"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_reports_missing_required_csv_column(tmp_path):
    write_evaluation_set(tmp_path)
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "feature": ["factor"],
            "ic": [0.4],
            "rankic": [0.5],
            "n_obs": [5],
        }
    ).to_csv(tmp_path / "metrics" / "ic.csv", index=False)

    with pytest.raises(ValueError, match="ic is missing columns"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_unsupported_schema_version(tmp_path):
    write_evaluation_set(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifact_schema_version"] = "999.0"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported artifact_schema_version"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_legacy_0_4_artifacts(tmp_path):
    write_evaluation_set(tmp_path)
    for file_name in ("manifest.json", "summary.json"):
        path = tmp_path / file_name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifact_schema_version"] = "0.4"
        if file_name == "manifest.json":
            payload["display"]["schema_version"] = "0.4"
        path.write_text(json.dumps(payload), encoding="utf-8")
    performance_path = tmp_path / "metrics" / "performance_summary.csv"
    performance = pd.read_csv(performance_path).rename(
        columns={
            "final_cumulative_gross_return": "total_gross_portfolio_return",
            "final_cumulative_net_return": "total_net_portfolio_return",
        }
    )
    performance.to_csv(performance_path, index=False)

    with pytest.raises(ValueError, match="unsupported artifact_schema_version"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_stale_return_columns_under_current_schema(
    tmp_path,
):
    write_evaluation_set(tmp_path)
    performance_path = tmp_path / "metrics" / "performance_summary.csv"
    performance = pd.read_csv(performance_path).rename(
        columns={
            "final_cumulative_gross_return": "total_gross_portfolio_return",
            "final_cumulative_net_return": "total_net_portfolio_return",
        }
    )
    performance.to_csv(performance_path, index=False)

    with pytest.raises(ValueError, match="retired columns"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_requires_current_final_return_columns(tmp_path):
    write_evaluation_set(tmp_path)
    performance_path = tmp_path / "metrics" / "performance_summary.csv"
    performance = pd.read_csv(performance_path).drop(
        columns=["final_cumulative_net_return"]
    )
    performance.to_csv(performance_path, index=False)

    with pytest.raises(ValueError, match="performance_summary is missing columns"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_display_download_item_that_escapes_zip(
    tmp_path,
):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["items"].append(
        {
            "item": "../evil.csv",
            "label": "Unsafe",
            "kind": "table",
            "table": "ic",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="display"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_display_manifest_missing_artifact_table(
    tmp_path,
):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["tables"] = [
        table
        for table in manifest["display"]["tables"]
        if table.get("key") != "performance_summary"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="display.tables must cover"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_unknown_display_download_item(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["items"].append(
        {
            "item": "missing.csv",
            "label": "Missing",
            "kind": "csv",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="known metadata item or table artifact"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_display_chart_for_unknown_table(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"].append(
        {
            "id": "unknown",
            "title": "Unknown",
            "description": "Unknown chart",
            "kind": "line",
            "table": "missing_table",
            "x": "timestamp",
            "series": [{"key": "value", "color": "#111827"}],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="display chart"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_report_path_that_escapes_run_dir(tmp_path):
    write_evaluation_set(tmp_path)
    outside_report = tmp_path.parent / "outside_report.md"
    outside_report.write_text("# Outside\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["report"] = "../outside_report.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe"):
        load_evaluation_set(tmp_path)


def test_load_run_index_rejects_metric_path_that_escapes_run_dir(tmp_path):
    write_evaluation_set(tmp_path)
    outside_performance = tmp_path.parent / "outside_performance_summary.csv"
    outside_performance.write_bytes(
        (tmp_path / "metrics" / "performance_summary.csv").read_bytes()
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics"][
        "performance_summary"
    ] = "../outside_performance_summary.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe"):
        load_run_index("core", tmp_path)


def test_load_evaluation_set_rejects_display_chart_for_missing_columns(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"].append(
        {
            "id": "bad_columns",
            "title": "Bad columns",
            "description": "Bad chart",
            "kind": "line",
            "table": "ic",
            "x": "missing_timestamp",
            "series": [{"key": "missing_ic", "color": "#111827"}],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="display chart"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_display_chart_for_unsupported_kind(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"].append(
        {
            "id": "scatter",
            "title": "Scatter",
            "description": "Unsupported chart",
            "kind": "scatter",
            "table": "ic",
            "x": "timestamp",
            "series": [{"key": "ic", "color": "#111827"}],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="display chart"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_missing_minimum_run_metadata(tmp_path):
    write_evaluation_set(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("domain", None)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="summary.json is missing required fields"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_non_boolean_external_data_flag(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["uses_external_data"] = "false"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        TypeError,
        match="manifest.json uses_external_data must be bool",
    ):
        load_evaluation_set(tmp_path)


def _collect_artifact_paths(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        paths: list[str] = []
        for nested in value.values():
            paths.extend(_collect_artifact_paths(nested))
        return paths
    return []

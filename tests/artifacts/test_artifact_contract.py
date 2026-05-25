from __future__ import annotations

import json
from pathlib import PurePosixPath

import pandas as pd
import pytest

from quant_research_platform.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    load_evaluation_set,
)
from quant_research_platform.artifacts.artifact_comparison import (
    compare_evaluation_sets,
)
from quant_research_platform.artifacts.artifact_inventory import index_evaluation_sets
from quant_research_platform.artifacts.artifact_loader import (
    load_evaluation_set as load_evaluation_set_from_target_module,
)
from quant_research_platform.artifacts.artifact_schema import (
    EVALUATION_SET_ARTIFACT_TYPE,
    EVALUATION_SET_REQUIRED_TABLES,
)
from quant_research_platform.artifacts.artifact_type_registry import (
    get_artifact_type_contract,
)
from quant_research_platform.artifacts.artifact_writer import (
    write_evaluation_set_artifacts,
)
from quant_research_platform.artifacts.display_manifest import build_display_manifest
from quant_research_platform.artifacts.manifest_validation import (
    validate_display_manifest,
)
from quant_research_platform.interfaces.api.run_index import load_run_index
from tests.artifacts.artifact_fixtures import write_evaluation_set


def test_load_evaluation_set_reads_manifest_summary_tables_and_report(tmp_path):
    write_evaluation_set(tmp_path)

    loaded = load_evaluation_set(tmp_path)

    assert loaded.output_dir == tmp_path
    assert loaded.manifest["evaluation_id"] == "run_a"
    assert loaded.manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert loaded.manifest["artifact_type"] == "evaluation_set"
    assert loaded.manifest["domain"] == "evaluation_synthetic"
    assert loaded.manifest["data_stage"] == "synthetic_only"
    assert loaded.manifest["uses_external_data"] is False
    assert loaded.summary["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert loaded.summary["domain"] == "evaluation_synthetic"
    assert loaded.summary["uses_external_data"] is False
    assert loaded.summary["n_rows_evaluated"] == 12
    assert set(loaded.tables) == {
        "combinations",
        "ic",
        "ic_summary",
        "grouped_returns",
        "group_spread_summary",
        "performance_summary",
        "returns",
        "cumulative_returns",
        "weights",
        "tradability_diagnostics",
        "label_availability",
        "factor_coverage",
        "tradability_coverage",
        "evaluation_summary",
    }
    display_tables = {
        table["key"]: table
        for table in loaded.manifest["display"]["tables"]
    }
    assert display_tables["tradability_diagnostics"]["path"] == (
        "diagnostics/tradability_diagnostics.csv"
    )
    download_items = {
        item["item"]
        for item in loaded.manifest["display"]["downloads"]["items"]
    }
    assert "diagnostics/tradability_diagnostics.csv" in download_items
    assert "portfolio/weights.parquet" in download_items
    assert "lineage.json" in download_items
    download_labels = {
        item["item"]: item["label"]
        for item in loaded.manifest["display"]["downloads"]["items"]
    }
    assert download_labels["portfolio/weights.parquet"].endswith("Parquet")
    assert loaded.tables["ic"]["ic"].tolist() == [0.4]
    assert loaded.report == "# Report\n"


def test_artifact_split_modules_expose_current_evaluation_set_contract():
    contract = get_artifact_type_contract("evaluation_set")

    assert EVALUATION_SET_ARTIFACT_TYPE == "evaluation_set"
    assert "combinations" in EVALUATION_SET_REQUIRED_TABLES
    assert contract.name == "evaluation_set"
    assert contract.version == ARTIFACT_SCHEMA_VERSION
    assert contract.inputs == ("EvaluationRunPayload",)
    assert "manifest.json" in contract.outputs
    assert "metrics/ic.csv" in contract.downloads
    assert "portfolio/weights.parquet" in contract.downloads
    assert "lineage.json" in contract.outputs
    assert "ic" not in contract.downloads
    assert "display" in contract.capabilities
    assert set(contract.tables) == set(EVALUATION_SET_REQUIRED_TABLES)
    assert callable(load_evaluation_set_from_target_module)
    assert callable(compare_evaluation_sets)
    assert callable(index_evaluation_sets)
    assert callable(build_display_manifest)
    assert callable(validate_display_manifest)
    assert callable(write_evaluation_set_artifacts)


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


def test_display_manifest_rejects_preview_rows_above_api_limit(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["display"]["tables"][0]["preview_rows"] = 1001

    with pytest.raises(ValueError, match="preview_rows"):
        validate_display_manifest(manifest)


def test_display_manifest_rejects_filterable_combinations_table(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    combinations = next(
        table
        for table in manifest["display"]["tables"]
        if table["key"] == "combinations"
    )
    combinations["join_key"] = "combination_id"
    combinations["supports_combination_filter"] = True

    with pytest.raises(ValueError, match="combinations"):
        validate_display_manifest(manifest)


def test_run_index_rejects_display_chart_missing_columns(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"].append(
        {
            "id": "stale-chart",
            "title": "Stale chart",
            "description": "References a removed column",
            "kind": "line",
            "table": "returns",
            "x": "timestamp",
            "series": [{"key": "missing_return_column"}],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stale-chart"):
        load_run_index("platform", tmp_path)


def test_load_evaluation_set_rejects_legacy_run_without_tradability_table(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["diagnostics"]["tradability_diagnostics"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="diagnostics.tradability_diagnostics"):
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


def test_load_evaluation_set_reports_missing_tradability_diagnostic_column(tmp_path):
    write_evaluation_set(tmp_path)
    path = tmp_path / "diagnostics" / "tradability_diagnostics.csv"
    diagnostics = pd.read_csv(path).drop(columns=["eligible_rate"])
    diagnostics.to_csv(path, index=False)

    with pytest.raises(ValueError, match="tradability_diagnostics is missing columns"):
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


def test_load_evaluation_set_requires_index_performance_columns(tmp_path):
    write_evaluation_set(tmp_path)
    performance_path = tmp_path / "metrics" / "performance_summary.csv"
    performance = pd.read_csv(performance_path).drop(columns=["sharpe_like"])
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


def test_load_evaluation_set_rejects_unknown_artifact_entries(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["private_debug"] = "private/debug.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown artifact groups"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_unknown_artifact_table_entries(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["summaries"]["private_debug"] = (
        "summaries/private_debug.csv"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown artifacts.summaries"):
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


def test_writer_rejects_missing_required_table_before_writing(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_evaluation_set(source)
    loaded = load_evaluation_set(source)
    tables = dict(loaded.tables)
    tables.pop("cumulative_returns")

    with pytest.raises(ValueError, match="missing required evaluation-set tables"):
        write_evaluation_set_artifacts(
            target,
            manifest=loaded.manifest,
            config=loaded.config,
            summary=loaded.summary,
            tables=tables,
            report=loaded.report,
        )

    assert not (target / "manifest.json").exists()


def test_writer_rejects_display_chart_missing_columns_before_writing(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_evaluation_set(source)
    loaded = load_evaluation_set(source)
    manifest = json.loads(json.dumps(loaded.manifest))
    manifest["display"]["charts"].append(
        {
            "id": "bad-chart",
            "title": "Bad chart",
            "kind": "line",
            "table": "ic",
            "x": "timestamp",
            "series": [{"key": "missing_ic_column", "label": "Missing"}],
        }
    )

    with pytest.raises(ValueError, match="bad-chart"):
        write_evaluation_set_artifacts(
            target,
            manifest=manifest,
            config=loaded.config,
            summary=loaded.summary,
            tables=loaded.tables,
            report=loaded.report,
        )

    assert not (target / "manifest.json").exists()


def test_writer_rejects_display_table_missing_from_write_payload(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_evaluation_set(source)
    loaded = load_evaluation_set(source)
    tables = dict(loaded.tables)
    manifest = json.loads(json.dumps(loaded.manifest))
    manifest["artifacts"]["summaries"]["custom_summary"] = (
        "summaries/custom_summary.csv"
    )
    manifest["display"] = build_display_manifest(
        manifest["artifacts"],
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )
    with pytest.raises(ValueError, match="unknown artifacts.summaries"):
        write_evaluation_set_artifacts(
            target,
            manifest=manifest,
            config=loaded.config,
            summary=loaded.summary,
            tables=tables,
            report=loaded.report,
        )

    assert not (target / "manifest.json").exists()


def test_writer_rejects_tables_not_declared_by_manifest(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_evaluation_set(source)
    loaded = load_evaluation_set(source)
    tables = dict(loaded.tables)
    tables["private_debug"] = tables["ic"].copy()

    with pytest.raises(ValueError, match="not declared by manifest"):
        write_evaluation_set_artifacts(
            target,
            manifest=loaded.manifest,
            config=loaded.config,
            summary=loaded.summary,
            tables=tables,
            report=loaded.report,
        )

    assert not (target / "manifest.json").exists()


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
        load_run_index("platform", tmp_path)


def test_load_run_index_rejects_missing_required_artifact_file(tmp_path):
    write_evaluation_set(tmp_path)
    (tmp_path / "portfolio" / "weights.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="portfolio/weights.parquet"):
        load_run_index("platform", tmp_path)


def test_load_run_index_rejects_missing_required_columns(tmp_path):
    write_evaluation_set(tmp_path)
    performance_path = tmp_path / "metrics" / "performance_summary.csv"
    performance = pd.read_csv(performance_path).drop(
        columns=["mean_net_portfolio_return"]
    )
    performance.to_csv(performance_path, index=False)

    with pytest.raises(ValueError, match="performance_summary is missing columns"):
        load_run_index("platform", tmp_path)


def test_load_run_index_rejects_unknown_artifact_entries(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["private_debug"] = "private/debug.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown artifact groups"):
        load_run_index("platform", tmp_path)


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

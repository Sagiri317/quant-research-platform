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
from quant_research_platform.artifacts.display_tables import (
    display_download_items,
    display_table_preview_limits,
    resolve_display_table_artifact_path,
)
from quant_research_platform.artifacts.manifest_validation import (
    validate_display_manifest,
)
from quant_research_platform.artifacts.schema_registry import (
    RETIRED_SCHEMA_06_IDENTITIES,
    required_metadata_files,
    required_table_keys,
)
from quant_research_platform.interfaces.api.run_index import load_run_index
from tests.artifacts.artifact_fixtures import write_evaluation_set


def test_load_evaluation_set_reads_manifest_summary_tables_and_report(tmp_path):
    write_evaluation_set(tmp_path)

    loaded = load_evaluation_set(tmp_path)

    assert loaded.output_dir == tmp_path
    assert loaded.manifest["run_id"] == "run_a"
    assert "evaluation_id" not in loaded.manifest
    assert loaded.manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert loaded.manifest["artifact_type"] == "evaluation_set"
    assert loaded.manifest["base_evaluation"]["domain"] == "evaluation_synthetic"
    assert loaded.manifest["data_stage"] == "synthetic_only"
    assert loaded.manifest["base_evaluation"]["uses_external_data"] is False
    assert loaded.summary["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert loaded.summary["run_id"] == "run_a"
    assert loaded.summary["base_evaluation"]["domain"] == "evaluation_synthetic"
    assert loaded.summary["base_evaluation"]["uses_external_data"] is False
    assert loaded.summary["n_rows_evaluated"] == 12
    assert loaded.run_context["provenance"]["run_id"] == "run_a"
    assert set(loaded.tables) == set(required_table_keys())
    display_tables = {
        table["key"]: table for table in loaded.manifest["display"]["tables"]
    }
    assert display_tables["diagnostic_summary"]["path"] == (
        "diagnostics/diagnostic_summary.csv"
    )
    download_items = display_download_items(loaded.manifest)
    assert {
        "metadata_file:manifest",
        "metadata_file:summary",
        "metadata_file:run_context",
        "metadata_file:report",
        "table:diagnostic_summary",
        "table:evaluation_metrics_summary",
        "table:returns",
        "table:coverage_summary",
    }.issubset(download_items)
    assert "dynamic:comparison" not in download_items
    assert not any(item in RETIRED_SCHEMA_06_IDENTITIES for item in download_items)
    assert loaded.tables["ic"]["ic"].tolist() == [0.4]
    assert loaded.report == "# Report\n"


def test_artifact_split_modules_expose_current_evaluation_set_contract():
    contract = get_artifact_type_contract("evaluation_set")

    assert EVALUATION_SET_ARTIFACT_TYPE == "evaluation_set"
    assert "combinations" in EVALUATION_SET_REQUIRED_TABLES
    assert contract.name == "evaluation_set"
    assert contract.version == ARTIFACT_SCHEMA_VERSION
    assert contract.inputs == ("EvaluationRunPayload",)
    assert contract.outputs == required_metadata_files()
    assert "metrics/ic.csv" in contract.downloads
    assert "metrics/evaluation_metrics_summary.csv" in contract.downloads
    assert "diagnostics/diagnostic_summary.csv" in contract.downloads
    assert "lineage.json" not in contract.outputs
    assert "config.json" not in contract.outputs
    assert "portfolio/cumulative_returns.csv" not in contract.downloads
    assert "ic" not in contract.downloads
    assert "display" in contract.capabilities
    assert set(contract.tables) == set(EVALUATION_SET_REQUIRED_TABLES)
    assert set(contract.tables) == set(required_table_keys())
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

    with pytest.raises(ValueError, match="returns"):
        load_evaluation_set(tmp_path)


def test_display_manifest_rejects_preview_rows_above_api_limit(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["display"]["tables"][0]["preview_rows"] = 1001

    with pytest.raises(ValueError, match="preview_rows"):
        validate_display_manifest(manifest)


def test_display_manifest_rejects_missing_preview_descriptor(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["display"]["tables"][0].pop("preview")

    with pytest.raises(ValueError, match="preview"):
        validate_display_manifest(manifest)


def test_display_manifest_rejects_duplicate_table_keys(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    duplicate = dict(manifest["display"]["tables"][0])
    manifest["display"]["tables"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate or invalid table key"):
        validate_display_manifest(manifest)


def test_display_manifest_rejects_preview_descriptor_mismatch(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    table = next(
        table
        for table in manifest["display"]["tables"]
        if table["key"] == "diagnostic_summary"
    )
    table["preview"]["supports_combination_filter"] = False

    with pytest.raises(ValueError, match="preview"):
        validate_display_manifest(manifest)


def test_display_table_resolver_reads_preview_limits_and_artifact_path(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    path = resolve_display_table_artifact_path(
        manifest,
        tmp_path,
        "diagnostic_summary",
    )
    preview = display_table_preview_limits(
        manifest,
        "diagnostic_summary",
    )

    assert path == tmp_path / "diagnostics" / "diagnostic_summary.csv"
    assert preview.default_limit == 20
    assert preview.max_limit == 1000
    assert preview.supports_combination_filter is True
    assert preview.filter_parameter == "combination_id"


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
            "kind": "derived_cumulative_return",
            "source_table_key": "returns",
            "x": "timestamp",
            "input_series": "missing_return_column",
            "transform": "compound_cumulative_return",
            "output_series": "missing_cumulative_return",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stale-chart"):
        load_run_index("platform", tmp_path)


def test_run_index_rejects_display_chart_missing_source_table(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"][0]["source_table_key"] = "weights"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source table mismatch"):
        load_run_index("platform", tmp_path)


def test_run_index_rejects_display_chart_stale_source_path(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"][0]["source_path"] = (
        "portfolio/cumulative_returns.csv"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source_path mismatch"):
        load_run_index("platform", tmp_path)


def test_run_index_rejects_duplicate_schema_registry_unique_key(tmp_path):
    write_evaluation_set(tmp_path)
    diagnostics_path = tmp_path / "diagnostics" / "diagnostic_summary.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    pd.concat([diagnostics, diagnostics.iloc[[0]]], ignore_index=True).to_csv(
        diagnostics_path,
        index=False,
    )

    with pytest.raises(ValueError, match="diagnostic_summary duplicate unique key"):
        load_run_index("platform", tmp_path)


def test_display_manifest_requires_registry_charts(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["display"].pop("charts")

    with pytest.raises(ValueError, match="display.charts"):
        validate_display_manifest(manifest)


def test_display_manifest_rejects_legacy_chart_table_field(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"][0]["table"] = "portfolio/cumulative_returns"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must not include table"):
        load_run_index("platform", tmp_path)


def test_display_manifest_rejects_preset_unexposed_metadata_ref(tmp_path):
    write_evaluation_set(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["presets"][0]["items"].append(
        {"kind": "metadata_file", "key": "lineage"}
    )

    with pytest.raises(ValueError, match="unexposed item"):
        validate_display_manifest(manifest)


def test_load_evaluation_set_rejects_legacy_schema_06_table_identity(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics"]["performance_summary"] = (
        "metrics/performance_summary.csv"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="retired identity"):
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

    with pytest.raises(ValueError, match="ic missing columns"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_reports_missing_diagnostic_summary_column(tmp_path):
    write_evaluation_set(tmp_path)
    path = tmp_path / "diagnostics" / "diagnostic_summary.csv"
    diagnostics = pd.read_csv(path).drop(columns=["decision_usable"])
    diagnostics.to_csv(path, index=False)

    with pytest.raises(ValueError, match="diagnostic_summary missing columns"):
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

    with pytest.raises(ValueError, match="unsupported artifact_schema_version"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_stale_schema_06_return_columns(
    tmp_path,
):
    write_evaluation_set(tmp_path)
    returns_path = tmp_path / "portfolio" / "returns.csv"
    returns = pd.read_csv(returns_path).rename(
        columns={"net_return": "net_portfolio_return"}
    )
    returns.to_csv(returns_path, index=False)

    with pytest.raises(ValueError, match="returns missing columns"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_requires_current_evaluation_metric_columns(tmp_path):
    write_evaluation_set(tmp_path)
    metrics_path = tmp_path / "metrics" / "evaluation_metrics_summary.csv"
    metrics = pd.read_csv(metrics_path).drop(columns=["metric_value"])
    metrics.to_csv(metrics_path, index=False)

    with pytest.raises(ValueError, match="evaluation_metrics_summary missing columns"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_requires_current_coverage_columns(tmp_path):
    write_evaluation_set(tmp_path)
    coverage_path = tmp_path / "input_quality" / "coverage_summary.csv"
    coverage = pd.read_csv(coverage_path).drop(columns=["coverage_rate"])
    coverage.to_csv(coverage_path, index=False)

    with pytest.raises(ValueError, match="coverage_summary missing columns"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_display_download_item_that_escapes_zip(
    tmp_path,
):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["items"].append(
        {
            "kind": "table",
            "key": "../evil",
            "label": "Unsafe",
            "path": "../evil.csv",
            "table": "../evil",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="path mismatch"):
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
        if table.get("key") != "evaluation_metrics_summary"
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

    with pytest.raises(ValueError, match="unknown artifact group"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_missing_metadata_artifact_key(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["run_context"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must include run_context"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_unknown_artifact_table_entries(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics"]["private_debug"] = "metrics/private_debug.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not in schema registry"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_unknown_display_download_item(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["items"].append(
        {
            "kind": "table",
            "key": "missing",
            "label": "Missing",
            "path": "missing.csv",
            "table": "missing",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="path mismatch"):
        load_evaluation_set(tmp_path)


def test_writer_rejects_missing_required_table_before_writing(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    write_evaluation_set(source)
    loaded = load_evaluation_set(source)
    tables = dict(loaded.tables)
    tables.pop("diagnostic_summary")

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
            "kind": "derived_cumulative_return",
            "source_table_key": "ic",
            "x": "timestamp",
            "input_series": "missing_ic_column",
            "transform": "compound_cumulative_return",
            "output_series": "missing_cumulative_ic",
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
    manifest["display"]["tables"] = [
        table
        for table in manifest["display"]["tables"]
        if table.get("key") != "diagnostic_summary"
    ]
    with pytest.raises(ValueError, match="display.tables must cover"):
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
    tables["weights"] = pd.DataFrame(
        {
            "combination_id": ["factor__forward_return__dataframe__long_short"],
            "timestamp": ["2026-01-01"],
            "asset": ["asset_001"],
            "target_weight": [1.0],
        }
    )

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


def test_load_evaluation_set_rejects_summary_with_evaluation_id(tmp_path):
    write_evaluation_set(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evaluation_id"] = summary["run_id"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="must not serialize evaluation_id"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_report_path_that_escapes_run_dir(tmp_path):
    write_evaluation_set(tmp_path)
    outside_report = tmp_path.parent / "outside_report.md"
    outside_report.write_text("# Outside\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["report"] = "../outside_report.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="report.md"):
        load_evaluation_set(tmp_path)


def test_load_run_index_rejects_metric_path_that_escapes_run_dir(tmp_path):
    write_evaluation_set(tmp_path)
    outside_metrics = tmp_path.parent / "outside_evaluation_metrics_summary.csv"
    outside_metrics.write_bytes(
        (tmp_path / "metrics" / "evaluation_metrics_summary.csv").read_bytes()
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics"]["evaluation_metrics_summary"] = (
        "../outside_evaluation_metrics_summary.csv"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe"):
        load_run_index("platform", tmp_path)


def test_load_run_index_rejects_missing_required_artifact_file(tmp_path):
    write_evaluation_set(tmp_path)
    (tmp_path / "diagnostics" / "diagnostic_summary.csv").unlink()

    with pytest.raises(FileNotFoundError, match="diagnostics/diagnostic_summary.csv"):
        load_run_index("platform", tmp_path)


def test_load_run_index_rejects_missing_required_columns(tmp_path):
    write_evaluation_set(tmp_path)
    metrics_path = tmp_path / "metrics" / "evaluation_metrics_summary.csv"
    metrics = pd.read_csv(metrics_path).drop(columns=["metric_value"])
    metrics.to_csv(metrics_path, index=False)

    with pytest.raises(
        ValueError, match="evaluation_metrics_summary is missing columns"
    ):
        load_run_index("platform", tmp_path)


def test_load_run_index_rejects_unknown_artifact_entries(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["private_debug"] = "private/debug.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown artifact group"):
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
            "kind": "derived_cumulative_return",
            "source_table_key": "ic",
            "x": "missing_timestamp",
            "input_series": "missing_ic",
            "transform": "compound_cumulative_return",
            "output_series": "missing_cumulative_ic",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="display chart"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_manifest_with_evaluation_id(tmp_path):
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_id"] = manifest["run_id"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must not serialize evaluation_id"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_missing_minimum_run_metadata(tmp_path):
    write_evaluation_set(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("base_evaluation", None)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="summary.json is missing required fields"):
        load_evaluation_set(tmp_path)


def test_load_evaluation_set_rejects_run_context_id_mismatch(tmp_path):
    write_evaluation_set(tmp_path)
    run_context_path = tmp_path / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["provenance"]["run_id"] = "other"
    run_context_path.write_text(json.dumps(run_context), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id"):
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

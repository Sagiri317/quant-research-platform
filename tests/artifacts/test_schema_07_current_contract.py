from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.artifacts.artifact_loader import load_evaluation_set
from quant_research_platform.artifacts.artifact_schema import (
    EVALUATION_SET_REQUIRED_TABLES,
)
from quant_research_platform.artifacts.display_tables import display_download_items
from tests.artifacts.artifact_fixtures import write_evaluation_set


def test_schema_07_contract_names_required_tables_only() -> None:
    assert ARTIFACT_SCHEMA_VERSION == "0.7"
    assert set(EVALUATION_SET_REQUIRED_TABLES) == {
        "combinations",
        "ic",
        "grouped_returns",
        "evaluation_metrics_summary",
        "returns",
        "coverage_summary",
        "diagnostic_policies",
        "diagnostic_summary",
        "diagnostic_metrics",
    }
    assert "ic_summary" not in EVALUATION_SET_REQUIRED_TABLES
    assert "performance_summary" not in EVALUATION_SET_REQUIRED_TABLES
    assert "cumulative_returns" not in EVALUATION_SET_REQUIRED_TABLES
    assert "evaluation_summary" not in EVALUATION_SET_REQUIRED_TABLES
    assert "pnl" not in EVALUATION_SET_REQUIRED_TABLES
    assert "positions" not in EVALUATION_SET_REQUIRED_TABLES
    assert "turnover" not in EVALUATION_SET_REQUIRED_TABLES


def test_writer_persists_schema_07_artifact_with_run_context(tmp_path) -> None:
    output_dir = tmp_path / "schema_07_eval"

    write_evaluation_set(output_dir, run_id="schema_07_eval")

    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "run_context.json").is_file()
    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "lineage.json").exists()
    assert (output_dir / "dimensions" / "combinations.csv").is_file()
    assert (output_dir / "metrics" / "evaluation_metrics_summary.csv").is_file()
    assert (output_dir / "portfolio" / "returns.csv").is_file()
    assert (output_dir / "input_quality" / "coverage_summary.csv").is_file()
    assert (output_dir / "diagnostics" / "diagnostic_summary.csv").is_file()
    assert not (output_dir / "portfolio" / "weights.parquet").exists()
    assert not (output_dir / "portfolio" / "cumulative_returns.csv").exists()
    assert not (output_dir / "metrics" / "performance_summary.csv").exists()
    assert not (output_dir / "summaries" / "evaluation_summary.csv").exists()
    assert not (output_dir / "portfolio" / "pnl.csv").exists()
    assert not (output_dir / "portfolio" / "positions.csv").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    run_context = json.loads(
        (output_dir / "run_context.json").read_text(encoding="utf-8")
    )
    combinations = pd.read_csv(output_dir / "dimensions" / "combinations.csv")

    assert manifest["artifact_schema_version"] == "0.7"
    assert summary["artifact_schema_version"] == "0.7"
    assert manifest["run_id"] == "schema_07_eval"
    assert summary["run_id"] == "schema_07_eval"
    assert run_context["provenance"]["run_id"] == "schema_07_eval"
    assert "evaluation_id" not in manifest
    assert "evaluation_id" not in summary
    assert list(combinations.columns) == [
        "combination_id",
        "feature",
        "label",
        "kernel",
        "portfolio_mode",
    ]

    display = manifest["display"]
    assert display["schema_version"] == "0.7"
    assert display["dimension_source"] == "dimensions/combinations.csv"
    assert display["default_selection"] == {
        "feature": "factor",
        "label": "forward_return",
        "kernel": "dataframe",
        "portfolio_mode": "long_short",
    }
    display_tables = {table["key"]: table for table in display["tables"]}
    assert display_tables["diagnostic_summary"]["format"] == "csv"
    assert display_tables["diagnostic_summary"]["path"] == (
        "diagnostics/diagnostic_summary.csv"
    )
    assert display_tables["diagnostic_summary"]["join_key"] == "combination_id"
    assert display_tables["diagnostic_summary"]["supports_combination_filter"] is True
    assert display_tables["coverage_summary"]["join_key"] is None
    assert display_tables["coverage_summary"]["supports_combination_filter"] is False
    assert display_tables["coverage_summary"]["preview"] == {
        "kind": "table",
        "table_key": "coverage_summary",
        "default_limit": 20,
        "max_limit": 1000,
        "supports_offset": True,
        "supports_combination_filter": False,
        "filter_parameter": None,
    }
    assert {
        "metadata_file:manifest",
        "metadata_file:summary",
        "metadata_file:run_context",
        "metadata_file:report",
        "table:evaluation_metrics_summary",
        "table:returns",
        "table:coverage_summary",
        "table:diagnostic_summary",
    }.issubset(display_download_items(manifest))


@pytest.mark.parametrize(
    ("artifact_group", "artifact_key", "artifact_path"),
    [
        ("metrics", "performance_summary", "metrics/performance_summary.csv"),
        ("portfolio", "cumulative_returns", "portfolio/cumulative_returns.csv"),
        ("summaries", "evaluation_summary", "summaries/evaluation_summary.csv"),
        ("metadata", "config", "config.json"),
        ("metadata", "lineage", "lineage.json"),
    ],
)
def test_schema_07_rejects_retired_schema_06_manifest_identities(
    tmp_path,
    artifact_group: str,
    artifact_key: str,
    artifact_path: str,
) -> None:
    write_evaluation_set(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_group == "metadata":
        manifest["artifacts"][artifact_key] = artifact_path
    else:
        manifest["artifacts"].setdefault(artifact_group, {})[artifact_key] = (
            artifact_path
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema 0.7|must not include"):
        load_evaluation_set(tmp_path)

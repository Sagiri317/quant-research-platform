from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION
from quant_research_platform.artifacts.artifact_loader import load_evaluation_set
from quant_research_platform.artifacts.artifact_writer import (
    write_evaluation_set_artifacts,
)


def test_schema_07_writer_loader_use_run_context_and_required_tables(tmp_path) -> None:
    output_dir = tmp_path / "schema_07_run"

    write_evaluation_set_artifacts(
        output_dir,
        summary=_summary(),
        run_context=_run_context(),
        tables=_tables(),
        report=None,
    )

    assert ARTIFACT_SCHEMA_VERSION == "0.7"
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "run_context.json").is_file()
    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "lineage.json").exists()
    assert not (output_dir / "report.md").exists()
    assert not (output_dir / "portfolio" / "cumulative_returns.csv").exists()
    assert not (output_dir / "summaries" / "evaluation_summary.csv").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    display = manifest["display"]
    assert manifest["artifact_schema_version"] == "0.7"
    assert manifest["artifact_type"] == "evaluation_set"
    assert manifest["run_id"] == "schema_07_run"
    assert "evaluation_id" not in manifest
    assert display["schema_version"] == "0.7"
    assert "report.md" not in {item["key"] for item in display["downloads"]["items"]}
    assert {"kind": "metadata_file", "key": "run_context"} in [
        {"kind": item["kind"], "key": item["key"]}
        for item in display["downloads"]["items"]
    ]

    loaded = load_evaluation_set(output_dir)

    assert loaded.summary["run_id"] == "schema_07_run"
    assert loaded.report is None
    assert set(loaded.tables) == set(_tables())


def test_schema_07_loader_rejects_run_id_mismatch(tmp_path) -> None:
    output_dir = tmp_path / "schema_07_run"
    write_evaluation_set_artifacts(
        output_dir,
        summary=_summary(),
        run_context=_run_context(),
        tables=_tables(),
        report=None,
    )
    run_context_path = output_dir / "run_context.json"
    run_context = json.loads(run_context_path.read_text(encoding="utf-8"))
    run_context["provenance"]["run_id"] = "other"
    run_context_path.write_text(json.dumps(run_context), encoding="utf-8")

    with pytest.raises(ValueError, match="run_id"):
        load_evaluation_set(output_dir)


def _summary() -> dict[str, object]:
    return {
        "run_id": "schema_07_run",
        "project": "quant-research-platform",
        "kind": "factor",
        "data_stage": "synthetic_only",
        "n_rows_evaluated": 4,
        "n_combinations": 1,
        "base_evaluation": {"kernel": "dataframe"},
        "diagnostics": {"enabled": False, "status": "disabled"},
        "warnings": [],
    }


def _run_context() -> dict[str, object]:
    return {
        "config": {
            "evaluation_spec": {"kernel": "dataframe"},
            "diagnostics_spec": {"enabled": False},
            "a_share_assumptions": {"status": "not_applicable"},
            "config_fingerprint": "sha256:fixture",
        },
        "provenance": {
            "run_id": "schema_07_run",
            "input_panel_fingerprint": "sha256:fixture",
            "input_data_version": "fixture",
            "input_artifact_refs": [],
            "output_artifact_index": {},
            "diagnostic_policy_fingerprint": "disabled",
            "generated_at": "2026-01-01T00:00:00Z",
            "generator": "pytest",
            "code_version": "fixture",
            "environment_summary": {"python": "fixture"},
            "staging_payload_fingerprint": "sha256:fixture",
            "factor_definition_version": "fixture",
            "random_seed": "not_applicable",
        },
    }


def _tables() -> dict[str, pd.DataFrame]:
    combo = "factor__forward_return__dataframe__long_only"
    return {
        "combinations": pd.DataFrame(
            {
                "combination_id": [combo],
                "feature": ["factor"],
                "label": ["forward_return"],
                "kernel": ["dataframe"],
                "portfolio_mode": ["long_only"],
            }
        ),
        "ic": pd.DataFrame(
            {
                "combination_id": [combo],
                "timestamp": ["2026-01-01"],
                "ic": [0.1],
                "rankic": [0.2],
                "n_obs": [4],
            }
        ),
        "grouped_returns": pd.DataFrame(
            {
                "combination_id": [combo],
                "timestamp": ["2026-01-01"],
                "group": [1],
                "mean_forward_return": [0.01],
                "n_obs": [2],
            }
        ),
        "evaluation_metrics_summary": pd.DataFrame(
            {
                "combination_id": [combo],
                "metric_family": ["ic"],
                "metric_name": ["mean_ic"],
                "metric_value": [0.1],
                "unit": ["ratio"],
                "n_observations": [1],
                "diagnostic_status": ["ok"],
                "status_reason": [""],
                "warning": [""],
            }
        ),
        "returns": pd.DataFrame(
            {
                "combination_id": [combo],
                "timestamp": ["2026-01-01"],
                "portfolio_mode": ["long_only"],
                "long_return": [0.01],
                "short_return": [0.0],
                "gross_portfolio_return": [0.01],
                "turnover": [0.0],
                "cost": [0.0],
                "net_return": [0.01],
                "long_count": [2],
                "short_count": [0],
                "return_source_column": ["forward_return"],
            }
        ),
        "coverage_summary": pd.DataFrame(
            {
                "input_family": ["label"],
                "input_name": ["forward_return"],
                "available_rows": [4],
                "missing_rows": [0],
                "coverage_rate": [1.0],
                "diagnostic_status": ["ok"],
                "status_reason": [""],
                "warning": [""],
            }
        ),
        "diagnostic_policies": pd.DataFrame(
            {
                "policy_id": ["__diagnostics_disabled__"],
                "policy_family": ["diagnostics_contract"],
                "enabled": [False],
                "requested": [False],
                "materialized": [True],
                "policy_fingerprint": ["disabled"],
                "diagnostic_status": ["disabled"],
                "status_reason": ["diagnostics disabled by request"],
                "warning": [""],
            }
        ),
        "diagnostic_summary": pd.DataFrame(
            {
                "combination_id": [combo],
                "policy_id": ["__diagnostics_disabled__"],
                "policy_family": ["diagnostics_contract"],
                "decision_usable": [False],
                "materialized_views": [""],
                "missing_inputs": [""],
                "diagnostic_status": ["disabled"],
                "status_reason": ["diagnostics disabled by request"],
                "warning": [""],
            }
        ),
        "diagnostic_metrics": pd.DataFrame(
            columns=[
                "combination_id",
                "policy_id",
                "policy_family",
                "metric_name",
                "metric_value",
                "unit",
                "decision_usable",
                "required_inputs_met",
                "diagnostic_status",
                "status_reason",
                "warning",
            ]
        ),
    }

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_portfolio_artifacts,
    make_output_edge_portfolio_spec,
    make_output_edge_report_spec,
    write_output_edge_portfolio_chunks,
)


def test_portfolio_outputs_write_manifest_and_local_scope(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    output_bundle = write_output_edge_portfolio_chunks(
        artifacts,
        spec=spec,
        output_spec=make_output_edge_report_spec(
            "portfolio_outputs",
            spec.portfolio_id,
            "portfolio",
        ),
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert output_bundle.as_posix().endswith("outputs/portfolio_demo_report")
    assert manifest["schema_id"] == "qrp-output-manifest"
    assert manifest["output_id"] == "portfolio_demo_report"
    assert manifest["output_definition_id"] == "portfolio_outputs"
    assert manifest["run_id"] == "portfolio_demo"
    assert manifest["metadata"]["portfolio_id"] == "portfolio_demo"
    assert manifest["metadata"]["research_dataset_id"] == "research_demo"
    assert manifest["metadata"]["signals"] == [
        {
            "signal_id": "signal",
            "source_kind": "feature",
            "source_column": "feature",
            "direction": "higher_is_better",
            "transform": "identity",
        }
    ]
    assert manifest["metadata"]["row_counts"]["positions"] == 2
    assert manifest["metadata"]["row_counts"]["portfolio_returns"] == 1
    assert manifest["metadata"]["weight_config"]["method"] == "rank_weighted_quantile"
    assert manifest["files"] == [
        "positions.parquet",
        "turnover.parquet",
        "cost_breakdown.parquet",
        "portfolio_returns.parquet",
        "cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
        "period_breakdown.parquet",
        "diagnostics.parquet",
        "capacity_diagnostics.parquet",
        "numeric_exposure_diagnostics.parquet",
        "categorical_exposure_diagnostics.parquet",
    ]


def test_portfolio_outputs_write_row_artifacts_from_chunks(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    row_artifacts = (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
        "capacity_diagnostics",
        "numeric_exposure_diagnostics",
        "categorical_exposure_diagnostics",
    )
    chunks = (
        {
            artifact_name: artifacts[artifact_name].iloc[:1].copy()
            for artifact_name in row_artifacts
        },
        {
            artifact_name: artifacts[artifact_name].iloc[1:].copy()
            for artifact_name in row_artifacts
        },
    )
    output_bundle = portfolio_outputs._write_portfolio_output_chunks(
        chunks,
        summary_factory=lambda: {
            artifact_name: artifacts[artifact_name]
            for artifact_name in artifacts
            if artifact_name not in row_artifacts
        },
        spec=spec,
        output_spec=make_output_edge_report_spec(
            "portfolio_outputs",
            spec.portfolio_id,
            "portfolio",
        ),
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["row_counts"]["positions"] == 2
    assert manifest["metadata"]["row_counts"]["turnover"] == 1
    assert len(pd.read_parquet(output_bundle / "positions.parquet")) == 2
    assert len(pd.read_parquet(output_bundle / "portfolio_returns.parquet")) == 1


def test_portfolio_outputs_reject_bad_frames(tmp_path: Path) -> None:
    spec = make_output_edge_portfolio_spec()
    report_spec = make_output_edge_report_spec(
        "portfolio_outputs",
        spec.portfolio_id,
        "portfolio",
    )
    layout = StorageLayout(project_root=tmp_path)

    missing_positions = make_output_edge_portfolio_artifacts()
    missing_positions["positions"] = missing_positions["positions"].drop(
        columns=["asset"]
    )
    with pytest.raises(ValueError, match="missing_portfolio_positions_columns"):
        write_output_edge_portfolio_chunks(
            missing_positions,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    extra_returns = make_output_edge_portfolio_artifacts()
    extra_returns["portfolio_returns"] = extra_returns["portfolio_returns"].assign(
        extra=1
    )
    with pytest.raises(ValueError, match="unregistered_portfolio_portfolio_returns"):
        write_output_edge_portfolio_chunks(
            extra_returns,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    wrong_return_source = make_output_edge_portfolio_artifacts()
    wrong_return_source["portfolio_returns"] = wrong_return_source[
        "portfolio_returns"
    ].assign(return_source_column="other_return")
    with pytest.raises(ValueError, match="portfolio_return_source_column_mismatch"):
        write_output_edge_portfolio_chunks(
            wrong_return_source,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    wrong_signal = make_output_edge_portfolio_artifacts()
    wrong_signal["portfolio_returns"] = wrong_signal["portfolio_returns"].assign(
        signal="other_signal"
    )
    with pytest.raises(ValueError, match="portfolio_artifact_signal_mismatch"):
        write_output_edge_portfolio_chunks(
            wrong_signal,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

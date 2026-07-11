from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.artifacts._portfolio_output_validation as validation
import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
from quant_research_platform.artifacts._portfolio_output_validation import (
    validate_portfolio_artifact_membership_columns,
)
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


@pytest.mark.parametrize(
    "bad_signal",
    [1, ["x"], {"x": 1}, {"x"}, None, pd.NA, float("nan")],
)
def test_portfolio_outputs_reject_invalid_signal_membership_values(
    tmp_path: Path,
    bad_signal: object,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    artifacts["positions"]["signal"] = artifacts["positions"]["signal"].astype(object)
    artifacts["positions"].at[0, "signal"] = bad_signal

    with pytest.raises(ValueError, match="portfolio_artifact_signal_mismatch"):
        write_output_edge_portfolio_chunks(
            artifacts,
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "portfolio_outputs",
                spec.portfolio_id,
                "portfolio",
            ),
            layout=StorageLayout(project_root=tmp_path),
        )


@pytest.mark.parametrize("bad_return_source", [1, ["label"], {"label": 1}, None])
def test_portfolio_outputs_reject_invalid_return_source_membership_values(
    tmp_path: Path,
    bad_return_source: object,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    artifacts["portfolio_returns"]["return_source_column"] = artifacts[
        "portfolio_returns"
    ]["return_source_column"].astype(object)
    artifacts["portfolio_returns"].at[0, "return_source_column"] = bad_return_source

    with pytest.raises(ValueError, match="portfolio_return_source_column_mismatch"):
        write_output_edge_portfolio_chunks(
            artifacts,
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "portfolio_outputs",
                spec.portfolio_id,
                "portfolio",
            ),
            layout=StorageLayout(project_root=tmp_path),
        )


@pytest.mark.parametrize(
    "padded_signal",
    [" signal", "signal ", "\tsignal", "signal\n"],
)
def test_portfolio_outputs_reject_padded_signal_membership_values(
    tmp_path: Path,
    padded_signal: str,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    artifacts["positions"] = artifacts["positions"].assign(signal=padded_signal)

    with pytest.raises(ValueError, match="portfolio_artifact_signal_mismatch"):
        write_output_edge_portfolio_chunks(
            artifacts,
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "portfolio_outputs",
                spec.portfolio_id,
                "portfolio",
            ),
            layout=StorageLayout(project_root=tmp_path),
        )


@pytest.mark.parametrize("padded_return_source", [" label", "label "])
def test_portfolio_outputs_reject_padded_return_source_membership_values(
    tmp_path: Path,
    padded_return_source: str,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    artifacts["portfolio_returns"] = artifacts["portfolio_returns"].assign(
        return_source_column=padded_return_source
    )

    with pytest.raises(ValueError, match="portfolio_return_source_column_mismatch"):
        write_output_edge_portfolio_chunks(
            artifacts,
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "portfolio_outputs",
                spec.portfolio_id,
                "portfolio",
            ),
            layout=StorageLayout(project_root=tmp_path),
        )


def test_portfolio_validation_rejects_invalid_membership_values() -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    artifacts["portfolio_returns"] = artifacts["portfolio_returns"].assign(signal=1)

    with pytest.raises(ValueError, match="portfolio_artifact_signal_mismatch"):
        for artifact_name, frame in artifacts.items():
            validate_portfolio_artifact_membership_columns(
                frame,
                artifact_name=artifact_name,
                spec=spec,
            )


def test_portfolio_validation_has_no_test_only_bulk_parallel_path() -> None:
    assert not hasattr(validation, "validate_portfolio_artifact_semantics")

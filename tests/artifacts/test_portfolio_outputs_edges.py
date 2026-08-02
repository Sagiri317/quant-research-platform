from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.artifacts._portfolio_output_validation as validation
import quant_research_platform.artifacts._portfolio_outputs as portfolio_outputs
from quant_research_platform.artifacts._output_manifest import (
    portfolio_output_metadata_from_row_counts,
)
from quant_research_platform.artifacts._portfolio_output_validation import (
    validate_portfolio_artifact_membership_columns,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_portfolio_artifacts,
    make_output_edge_portfolio_spec,
    make_output_edge_report_spec,
    output_edge_portfolio_row_artifact_names,
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


@pytest.mark.parametrize(
    ("mapping_name", "mutation"),
    (
        ("rows", "missing"),
        ("rows", "extra"),
        ("rows", "value"),
        ("summary", "missing"),
        ("summary", "extra"),
        ("summary", "value"),
    ),
)
def test_portfolio_outputs_reject_exact_mapping_mutations_and_clean_scratch(
    tmp_path: Path,
    mapping_name: str,
    mutation: str,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    row_names = output_edge_portfolio_row_artifact_names()
    row_mapping = {name: artifacts[name] for name in row_names}
    summary_mapping = {
        name: artifacts[name] for name in artifacts if name not in row_names
    }
    target = row_mapping if mapping_name == "rows" else summary_mapping
    if mutation == "missing":
        target.pop(next(iter(target)))
    elif mutation == "extra":
        target["unexpected_artifact"] = pd.DataFrame()
    else:
        target[next(iter(target))] = object()

    final_path = tmp_path / "outputs" / "portfolio_demo_report"
    error = "portfolio_row" if mapping_name == "rows" else "portfolio_summary"
    with pytest.raises(ValueError, match=error):
        portfolio_outputs._write_portfolio_output_chunks(
            (row_mapping,),
            summary_factory=lambda: summary_mapping,
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "portfolio_outputs", spec.portfolio_id, "portfolio"
            ),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert not final_path.exists()


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("missing", None),
        ("extra", None),
        ("bool", True),
        ("string", "1"),
        ("float", 1.0),
        ("numpy", np.int64(1)),
        ("negative", -1),
    ),
)
def test_portfolio_row_counts_exact_mutation_matrix(
    mutation: str,
    value: object,
) -> None:
    row_counts = {name: 0 for name in validation.PORTFOLIO_ARTIFACTS}
    if mutation == "missing":
        row_counts.pop("positions")
    elif mutation == "extra":
        row_counts["unexpected"] = 0
    else:
        row_counts["positions"] = value

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        validation.validate_portfolio_row_counts(row_counts)


@pytest.mark.parametrize(
    "mutation",
    (
        "weight_method",
        "weight_mode",
        "weight_quantile_sum",
        "weight_inverse_risk",
        "tradability_entry_policy",
        "tradability_exit_policy",
        "cost_negative",
        "cost_nonfinite",
        "diagnostics_capacity_dependency",
        "diagnostics_exposure_overlap",
        "rebalance_frequency",
        "signal_source_kind",
        "signal_direction",
        "signal_transform",
        "signal_filter_column",
        "signal_return_overlap",
    ),
)
def test_portfolio_metadata_reuses_complete_business_owner_graph(
    mutation: str,
) -> None:
    spec = make_output_edge_portfolio_spec()
    metadata = portfolio_output_metadata_from_row_counts(
        spec=spec,
        row_counts={name: 0 for name in validation.PORTFOLIO_ARTIFACTS},
    )
    assert validation.validate_portfolio_output_metadata_shape(metadata) is metadata

    if mutation == "weight_method":
        metadata["weight_config"]["method"] = "unknown"
    elif mutation == "weight_mode":
        metadata["weight_config"]["portfolio_mode"] = "unknown"
    elif mutation == "weight_quantile_sum":
        metadata["weight_config"].update(
            {"top_quantile": 0.8, "bottom_quantile": 0.8}
        )
    elif mutation == "weight_inverse_risk":
        metadata["weight_config"].update(
            {"method": "inverse_vol_scaled_quantile", "risk_column": None}
        )
    elif mutation == "tradability_entry_policy":
        metadata["tradability_config"]["blocked_entry_policy"] = "unknown"
    elif mutation == "tradability_exit_policy":
        metadata["tradability_config"]["blocked_exit_policy"] = "unknown"
    elif mutation == "cost_negative":
        metadata["cost_config"]["buy_cost_bps"] = -1.0
    elif mutation == "cost_nonfinite":
        metadata["cost_config"]["spread_bps"] = float("inf")
    elif mutation == "diagnostics_capacity_dependency":
        metadata["diagnostics_config"]["capacity_float_market_cap_column"] = (
            "float_market_cap"
        )
    elif mutation == "diagnostics_exposure_overlap":
        metadata["diagnostics_config"]["numeric_exposure_columns"] = ["sector"]
        metadata["diagnostics_config"]["categorical_exposure_columns"] = ["sector"]
    elif mutation == "rebalance_frequency":
        metadata["rebalance_frequency"] = "weekly"
    elif mutation == "signal_source_kind":
        metadata["signals"][0]["source_kind"] = "unknown"
    elif mutation == "signal_direction":
        metadata["signals"][0]["direction"] = "unknown"
    elif mutation == "signal_transform":
        metadata["signals"][0]["transform"] = "unknown"
    elif mutation == "signal_filter_column":
        metadata["signals"][0]["source_column"] = "eligible"
    else:
        metadata["signals"][0]["source_column"] = metadata[
            "return_source_column"
        ]

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        validation.validate_portfolio_output_metadata_shape(metadata)

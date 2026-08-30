from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.artifacts.outputs.parquet as parquet_writers
import quant_research_platform.artifacts.outputs.portfolio as portfolio_outputs
import quant_research_platform.artifacts.outputs.portfolio_validation as validation
from quant_research_platform.artifacts.outputs import (
    write_portfolio_outputs as public_write_portfolio_outputs,
)
from quant_research_platform.artifacts.outputs.manifest import (
    portfolio_output_metadata_from_row_counts,
)
from quant_research_platform.artifacts.outputs.portfolio_validation import (
    validate_portfolio_artifact_membership_columns,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_portfolio_artifacts,
    make_output_edge_portfolio_spec,
    make_output_edge_report_spec,
    output_edge_portfolio_row_artifact_names,
    write_output_edge_portfolio_chunks,
)
from tests.helpers.runtime_resources import capture_test_policy


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
        "additive_cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
        "period_breakdown.parquet",
        "diagnostics.parquet",
        "capacity_diagnostics.parquet",
        "numeric_exposure_diagnostics.parquet",
        "categorical_exposure_diagnostics.parquet",
    ]


@pytest.mark.parametrize(
    "mutation",
    ("bad_string", "infinite", "missing_reason", "reason_with_value", "unknown_reason"),
)
def test_portfolio_statistic_reason_pair_rejects_tampering(mutation: str) -> None:
    spec = make_output_edge_portfolio_spec()
    frame = make_output_edge_portfolio_artifacts()["performance_summary"].copy()
    if mutation == "bad_string":
        frame.loc[0, "sharpe_like"] = "0.1"
    elif mutation == "infinite":
        frame.loc[0, "sharpe_like"] = float("inf")
    elif mutation == "missing_reason":
        frame.loc[0, "sharpe_like_reason"] = None
    elif mutation == "reason_with_value":
        frame.loc[0, "sharpe_like"] = 1.0
    else:
        frame.loc[0, "sharpe_like_reason"] = "made_up_reason"

    with pytest.raises(ValueError, match="invalid_portfolio_statistic_reason"):
        validate_portfolio_artifact_membership_columns(
            frame,
            artifact_name="performance_summary",
            spec=spec,
        )


def test_public_portfolio_writer_forwards_cumulative_factory(
    tmp_path: Path,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    row_artifacts = tuple(
        name
        for name in output_edge_portfolio_row_artifact_names()
        if name != "additive_cumulative_returns"
    )
    output_bundle = public_write_portfolio_outputs(
        output_chunks=({name: artifacts[name] for name in row_artifacts},),
        summary_factory=lambda: {
            name: artifacts[name]
            for name in artifacts
            if name not in (*row_artifacts, "additive_cumulative_returns")
        },
        additive_cumulative_returns_factory=lambda: iter(
            (artifacts["additive_cumulative_returns"],)
        ),
        spec=spec,
        output_spec=make_output_edge_report_spec(
            "portfolio_outputs", spec.portfolio_id, "portfolio"
        ),
        layout=StorageLayout(project_root=tmp_path),
    )

    persisted = pd.read_parquet(output_bundle / "additive_cumulative_returns.parquet")
    pd.testing.assert_frame_equal(
        persisted,
        artifacts["additive_cumulative_returns"],
        check_dtype=False,
    )


def test_portfolio_aggregate_rows_exceed_budget_without_full_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = make_output_edge_portfolio_spec()
    artifacts = make_output_edge_portfolio_artifacts()
    row_artifacts = tuple(
        name
        for name in output_edge_portfolio_row_artifact_names()
        if name != "additive_cumulative_returns"
    )
    monkeypatch.setattr(parquet_writers, "_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(portfolio_outputs, "_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES", 1)
    monkeypatch.setattr(
        portfolio_outputs,
        "_PORTFOLIO_SUMMARY_FIXED_WORKING_BYTES",
        1,
    )
    monkeypatch.setattr(
        parquet_writers,
        "_table_parquet_page_working_set_bytes",
        lambda _table, *, rows: 1,
    )
    monkeypatch.setattr(
        parquet_writers,
        "_parquet_row_group_footer_upper_bound",
        lambda _schema, *, rows: 0,
    )
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024,
    )

    def row_chunks():
        for _ in range(256):
            yield {name: artifacts[name] for name in row_artifacts}

    def cumulative_chunks():
        for _ in range(256):
            yield artifacts["additive_cumulative_returns"]

    with runtime_resource_policy_scope(policy):
        output_bundle = portfolio_outputs._write_portfolio_output_chunks(
            row_chunks(),
            summary_factory=lambda: {
                name: artifacts[name]
                for name in artifacts
                if name not in (*row_artifacts, "additive_cumulative_returns")
            },
            additive_cumulative_returns_factory=cumulative_chunks,
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "portfolio_outputs", spec.portfolio_id, "portfolio"
            ),
            layout=StorageLayout(project_root=tmp_path),
        )
        assert policy.used_memory_bytes == 0

    cumulative_path = output_bundle / "additive_cumulative_returns.parquet"
    persisted_cumulative = pd.read_parquet(cumulative_path)
    expected_cumulative = pd.concat(
        [artifacts["additive_cumulative_returns"]] * 256,
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(
        persisted_cumulative,
        expected_cumulative,
        check_dtype=False,
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text())
    row_counts = manifest["metadata"]["row_counts"]
    for artifact_name in validation.PORTFOLIO_ARTIFACTS:
        expected_count = len(artifacts[artifact_name])
        if (
            artifact_name in row_artifacts
            or artifact_name == "additive_cumulative_returns"
        ):
            expected_count *= 256
        assert row_counts[artifact_name] == expected_count
    persisted_bytes = sum(
        path.stat().st_size
        for path in output_bundle.iterdir()
        if path.suffix == ".parquet"
    )
    assert persisted_bytes > policy.memory_budget_bytes
    assert policy.used_memory_bytes == 0


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
            if artifact_name not in (*row_artifacts, "additive_cumulative_returns")
        },
        additive_cumulative_returns_factory=lambda: iter(
            (artifacts["additive_cumulative_returns"],)
        ),
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


def test_portfolio_row_writers_reject_fixed_workspace_before_first_pull(
    tmp_path: Path,
) -> None:
    row_artifacts = output_edge_portfolio_row_artifact_names()
    fixed_bytes = (
        len(row_artifacts) * parquet_writers._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=fixed_bytes,
    )
    pulls = 0

    class Input:
        def __iter__(self) -> Input:
            return self

        def __next__(self) -> object:
            nonlocal pulls
            pulls += 1
            raise AssertionError("portfolio_input_pulled_before_fixed_admission")

    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            portfolio_outputs._write_portfolio_row_artifact_chunks(
                Input(),  # type: ignore[arg-type]
                temp_root=tmp_path,
                row_artifacts=row_artifacts,
                required_columns={name: ("value",) for name in row_artifacts},
                spec=object(),  # type: ignore[arg-type]
            )

    assert pulls == 0


def test_portfolio_row_writers_disk_preflight_happens_before_first_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_artifacts = tuple(
        name
        for name in output_edge_portfolio_row_artifact_names()
        if name != "additive_cumulative_returns"
    )
    spec = make_output_edge_portfolio_spec()
    output_spec = make_output_edge_report_spec(
        "portfolio_outputs",
        spec.portfolio_id,
        "portfolio",
    )
    observed: dict[str, object] = {}
    pulls = 0

    class Input:
        def __iter__(self) -> Input:
            return self

        def __next__(self) -> object:
            nonlocal pulls
            pulls += 1
            raise AssertionError("portfolio_input_pulled_before_disk_preflight")

    def reject_disk(path: Path, *, reserve_bytes: int) -> None:
        observed["path"] = path
        observed["reserve_bytes"] = reserve_bytes
        raise OSError("portfolio_disk_preflight")

    monkeypatch.setattr(portfolio_outputs, "require_free_disk_headroom", reject_disk)

    with pytest.raises(OSError, match="portfolio_disk_preflight"):
        portfolio_outputs._write_portfolio_output_chunks(
            Input(),  # type: ignore[arg-type]
            summary_factory=lambda: {},
            additive_cumulative_returns_factory=lambda: iter(()),
            spec=spec,
            output_spec=output_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    assert pulls == 0
    assert (
        observed["reserve_bytes"]
        == (len(row_artifacts) + 1)
        * portfolio_outputs._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
    )
    assert not (tmp_path / "outputs" / output_spec.output_id).exists()
    assert not list((tmp_path / "outputs").glob(".a-*"))


def test_portfolio_row_writer_fixed_reservation_covers_close_without_double_charge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row_artifacts = output_edge_portfolio_row_artifact_names()
    fixed_one = parquet_writers._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
    fixed_bytes = len(row_artifacts) * fixed_one
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=fixed_bytes + 2 * 64 * 1024,
    )
    frame = pd.DataFrame({"value": [1.0]})
    pull_reservations: list[int] = []
    close_reservations: list[int] = []
    metadata_reservations: list[int] = []
    input_close_reservations: list[int] = []

    class Input:
        yielded = False

        def __iter__(self) -> Input:
            return self

        def __next__(self) -> dict[str, pd.DataFrame]:
            pull_reservations.append(policy.used_memory_bytes)
            if self.yielded:
                raise StopIteration
            self.yielded = True
            return {name: frame for name in row_artifacts}

        def close(self) -> None:
            input_close_reservations.append(policy.used_memory_bytes)

    class Writer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def write_table(self, _table: object, **_kwargs: object) -> None:
            pass

        def add_key_value_metadata(self, _metadata: object) -> None:
            metadata_reservations.append(policy.used_memory_bytes)

        def close(self) -> None:
            close_reservations.append(policy.used_memory_bytes)

    monkeypatch.setattr(
        portfolio_outputs,
        "validate_portfolio_artifact_mapping_keys",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        portfolio_outputs,
        "_portfolio_output_chunk_frame",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(portfolio_outputs.pq, "ParquetWriter", Writer)
    monkeypatch.setattr(
        parquet_writers,
        "_table_parquet_page_working_set_bytes",
        lambda _table, *, rows: fixed_one + 1,
    )
    monkeypatch.setattr(
        parquet_writers,
        "_parquet_row_group_footer_upper_bound",
        lambda _schema, *, rows: 0,
    )

    with runtime_resource_policy_scope(policy):
        portfolio_outputs._write_portfolio_row_artifact_chunks(
            Input(),
            temp_root=tmp_path,
            row_artifacts=row_artifacts,
            required_columns={name: ("value",) for name in row_artifacts},
            spec=object(),  # type: ignore[arg-type]
        )
        assert policy.used_memory_bytes == 0

    assert pull_reservations == [fixed_bytes, fixed_bytes + len(row_artifacts)]
    assert metadata_reservations == [
        fixed_bytes + remaining for remaining in range(len(row_artifacts), 0, -1)
    ]
    assert close_reservations == metadata_reservations
    assert input_close_reservations == [fixed_bytes + len(row_artifacts)]


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
    row_names = tuple(
        name
        for name in output_edge_portfolio_row_artifact_names()
        if name != "additive_cumulative_returns"
    )
    row_mapping = {name: artifacts[name] for name in row_names}
    summary_mapping = {
        name: artifacts[name]
        for name in artifacts
        if name not in (*row_names, "additive_cumulative_returns")
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
            additive_cumulative_returns_factory=lambda: iter(
                (artifacts["additive_cumulative_returns"],)
            ),
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
        metadata["weight_config"].update({"top_quantile": 0.8, "bottom_quantile": 0.8})
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
        metadata["signals"][0]["source_column"] = metadata["return_source_column"]

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        validation.validate_portfolio_output_metadata_shape(metadata)

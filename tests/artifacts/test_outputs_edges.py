from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts import outputs
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.portfolios import PortfolioSpec, WeightConfig
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids


@pytest.mark.parametrize(
    "output_id",
    (
        "",
        "../escape",
        "nested/output",
        "nested\\output",
        "C:\\absolute",
        "CON",
        "aux.txt",
        "free text",
        "with.dot",
    ),
)
def test_output_spec_rejects_filesystem_unsafe_output_id(output_id: str) -> None:
    with pytest.raises(ValueError, match="invalid_output_id"):
        OutputSpec(
            output_id=output_id,
            output_definition_id="prediction_outputs",
            run_id="prediction_run",
            market_id=MarketId("a_share"),
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        )


def test_output_spec_rejects_output_definition_id_as_output_id() -> None:
    with pytest.raises(ValueError, match="output_id_must_identify_instance"):
        OutputSpec(
            output_id="prediction_outputs",
            output_definition_id="prediction_outputs",
            run_id="prediction_run",
            market_id=MarketId("a_share"),
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        )


def test_report_frame_validation_rejects_missing_and_extra_columns() -> None:
    with pytest.raises(ValueError, match="missing_evaluation_metrics_columns: ic"):
        outputs._validate_output_frame(
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            required=("timestamp", "ic"),
            error_name="evaluation_metrics",
        )
    with pytest.raises(ValueError, match="unregistered_evaluation_metrics_columns"):
        outputs._validate_output_frame(
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-02")],
                    "ic": [0.1],
                    "extra": [1],
                }
            ),
            required=("timestamp", "ic"),
            error_name="evaluation_metrics",
        )


def test_prediction_output_validation_rejects_bad_frames(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)
    valid = _prediction_frame(with_metadata=True)

    with pytest.raises(ValueError, match="missing_prediction_output_columns"):
        outputs.write_prediction_outputs(
            valid.drop(columns=["prediction_score"]),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="unregistered_prediction_output_columns"):
        outputs.write_prediction_outputs(
            valid.assign(extra=1),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    with pytest.raises(ValueError, match="duplicate_prediction_keys"):
        outputs.write_prediction_outputs(
            pd.concat([valid, valid], ignore_index=True),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_require_standard_metadata_attrs(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="missing_model_metadata"):
        outputs.write_prediction_outputs(
            _prediction_frame(with_metadata=False),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    model_only = _prediction_frame(with_metadata=False)
    model_only.attrs["model_metadata"] = _model_metadata()
    with pytest.raises(ValueError, match="missing_prediction_metadata"):
        outputs.write_prediction_outputs(
            model_only,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    generic_metadata = _prediction_frame(with_metadata=False)
    generic_metadata.attrs["model_metadata"] = _model_metadata()
    generic_metadata.attrs["metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="missing_prediction_metadata"):
        outputs.write_prediction_outputs(
            generic_metadata,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    incomplete_model = _prediction_frame(with_metadata=True)
    incomplete_model.attrs["model_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            incomplete_model,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    incomplete_prediction = _prediction_frame(with_metadata=True)
    incomplete_prediction.attrs["prediction_metadata"] = {"contract_version": "0.1"}
    with pytest.raises(ValueError, match="invalid_prediction_metadata"):
        outputs.write_prediction_outputs(
            incomplete_prediction,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_inconsistent_metadata(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_rows = _prediction_frame(with_metadata=True)
    bad_rows.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "prediction_rows": 2,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_rows,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_window = _prediction_frame(with_metadata=True)
    bad_window.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "test_start": "2024-01-03",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_window,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_train_end = _prediction_frame(with_metadata=True)
    bad_train_end.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "train_end": "2024-01-02",
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_train_end,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model = _prediction_frame(with_metadata=True)
    bad_model.attrs["model_metadata"] = {
        **_model_metadata(),
        "coefficient_count": 0,
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        outputs.write_prediction_outputs(
            bad_model,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_feature_columns = _prediction_frame(with_metadata=True)
    bad_feature_columns.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "input_columns": ["factor", "other_input"],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_feature_columns,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_bad_row_count_chain_and_filter_contract(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_prediction_counts = _prediction_frame(with_metadata=True)
    bad_prediction_counts.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "rows_after_input_dropna": 0,
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_prediction_counts,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model_counts = _prediction_frame(with_metadata=True)
    bad_model_counts.attrs["model_metadata"] = {
        **_model_metadata(),
        "rows_after_filter_mode": 1,
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_counts,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_prediction_filter = _prediction_frame(with_metadata=True)
    bad_prediction_filter.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "filter_columns_used": [],
    }
    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_prediction_filter,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_model_filter = _prediction_frame(with_metadata=True)
    bad_model_filter.attrs["model_metadata"] = {
        **_model_metadata(),
        "filter_definition": "no eligible/can_enter/can_exit filter",
    }
    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_filter,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_prediction_outputs_reject_bad_effective_test_end(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_effective_end = _prediction_frame(with_metadata=True)
    bad_effective_end.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "effective_test_end": "2024-01-03",
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_effective_end,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_unparseable_prediction_timestamp(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_timestamp = _prediction_frame(with_metadata=True)
    bad_timestamp["timestamp"] = ["bad-date"]

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_timestamp,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_effective_test_end_after_requested_end(
    tmp_path: Path,
) -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    late_prediction = _prediction_frame(with_metadata=True).assign(
        timestamp=pd.to_datetime(["2024-01-03"])
    )
    late_prediction.attrs["model_metadata"] = {
        **_model_metadata(),
        "test_end": "2024-01-02",
        "oos_end": "2024-01-02",
    }
    late_prediction.attrs["prediction_metadata"] = {
        **_prediction_metadata(),
        "requested_test_end": "2024-01-02",
        "effective_test_end": "2024-01-03",
        "is_final_oos_window": False,
    }

    with pytest.raises(ValueError, match="prediction_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            late_prediction,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_prediction_outputs_reject_model_train_window_after_spec_train_end(
    tmp_path: Path,
) -> None:
    spec = _prediction_spec()
    report_spec = _report_spec(
        "prediction_outputs",
        spec.prediction_id,
        "prediction",
    )
    bad_model_window = _prediction_frame(with_metadata=True)
    bad_model_window.attrs["model_metadata"] = {
        **_model_metadata(),
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
    }

    with pytest.raises(ValueError, match="model_metadata_inconsistent"):
        outputs.write_prediction_outputs(
            bad_model_window,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_output_bundle_root_rejects_mismatched_specs(tmp_path: Path) -> None:
    layout = StorageLayout(project_root=tmp_path)

    with pytest.raises(ValueError, match="output_spec_mismatch"):
        outputs._derive_output_bundle_root(
            output_spec=_report_spec("unknown", "run_demo", "prediction"),
            layout=layout,
            expected_definition_id="prediction_outputs",
        )
    with pytest.raises(ValueError, match="output_spec_mismatch"):
        outputs._derive_output_bundle_root(
            output_spec=_report_spec(
                "evaluation_outputs",
                "run_demo",
                "evaluation",
            ),
            layout=layout,
            expected_definition_id="prediction_outputs",
        )


@pytest.mark.parametrize(
    "data_layer",
    ["raw", "interim", "processed", "research"],
)
def test_output_bundle_root_rejects_dataset_roots(
    tmp_path: Path,
    data_layer: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        outputs,
        "output_bundle_root",
        lambda _root, _output_id: tmp_path / "data" / data_layer / "report",
    )

    with pytest.raises(ValueError, match="output_must_use_artifact_root"):
        outputs._derive_output_bundle_root(
            output_spec=_report_spec(
                "prediction_outputs",
                "prediction_demo",
                "prediction",
            ),
            layout=StorageLayout(project_root=tmp_path),
            expected_definition_id="prediction_outputs",
        )


def test_evaluation_outputs_validation_rejects_bad_inputs(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")

    with pytest.raises(ValueError, match="missing_evaluation_metrics_columns"):
        outputs.write_evaluation_outputs(
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_ic_summary_columns"):
        outputs.write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=pd.DataFrame({"signal": ["signal"]}),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_group_spread_summary"):
        outputs.write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=pd.DataFrame({"signal": ["signal"]}),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_evaluation_outputs_rejects_inconsistent_diagnostics(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")
    layout = StorageLayout(project_root=tmp_path)

    bad_metric_rows = replace(_evaluation_diagnostics(), metric_rows=2)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_metric_rows,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_filter_mode = replace(_evaluation_diagnostics(), filter_mode="unfiltered")
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_filter_mode,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_count = replace(_evaluation_diagnostics(), timestamp_count=2)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_timestamp_count,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    skipped_metrics = _evaluation_metrics_frame().assign(n_obs=[1])
    bad_skipped = replace(_evaluation_diagnostics(), skipped_timestamp_count=0)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            skipped_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_skipped,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_metrics = _evaluation_metrics_frame().assign(timestamp=["bad-date"])
    bad_timestamp_diagnostics = replace(
        _evaluation_diagnostics(),
        timestamp_count=0,
        min_timestamp=None,
        max_timestamp=None,
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            bad_timestamp_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_timestamp_diagnostics,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_evaluation_outputs_rejects_bad_keys_features_groups_and_counts(
    tmp_path: Path,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")
    layout = StorageLayout(project_root=tmp_path)

    duplicate_metrics = pd.concat(
        [_evaluation_metrics_frame(), _evaluation_metrics_frame()],
        ignore_index=True,
    )
    duplicate_diagnostics = replace(
        _evaluation_diagnostics(),
        metric_rows=2,
        rows_after_signal_label_dropna=4,
    )
    with pytest.raises(ValueError, match="duplicate_evaluation_metrics_keys"):
        outputs.write_evaluation_outputs(
            duplicate_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=duplicate_diagnostics,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_feature_metrics = _evaluation_metrics_frame().assign(signal="other")
    with pytest.raises(ValueError, match="evaluation_artifact_signal_label_mismatch"):
        outputs.write_evaluation_outputs(
            bad_feature_metrics,
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_group = _evaluation_grouped_label_values_frame().assign(group=3)
    with pytest.raises(ValueError, match="evaluation_group_out_of_range"):
        outputs.write_evaluation_outputs(
            _evaluation_metrics_frame(),
            bad_group,
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_row_chain = replace(_evaluation_diagnostics(), rows_after_filter_mode=1)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            _evaluation_metrics_frame(),
            _evaluation_grouped_label_values_frame(),
            ic_summary=_evaluation_ic_summary_frame(),
            group_spread_summary=_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_row_chain,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_evaluation_outputs_writes_label_value_artifact_name(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = _report_spec("evaluation_outputs", spec.run_id, "evaluation")

    run_root = outputs.write_evaluation_outputs(
        _evaluation_metrics_frame(),
        _evaluation_grouped_label_values_frame(),
        ic_summary=_evaluation_ic_summary_frame(),
        group_spread_summary=_evaluation_group_spread_summary_frame(),
        evaluation_diagnostics=_evaluation_diagnostics(),
        spec=spec,
        output_spec=report_spec,
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    assert (run_root / "grouped_label_values.parquet").is_file()
    assert (run_root / "ic_summary.parquet").is_file()
    assert (run_root / "group_spread_summary.parquet").is_file()
    assert not (run_root / "grouped_returns.parquet").exists()
    assert manifest["schema_id"] == "qrp-output-manifest"
    assert manifest["files"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
    ]


def test_portfolio_outputs_write_manifest_and_local_scope(
    tmp_path: Path,
) -> None:
    spec = _portfolio_spec()
    artifacts = _portfolio_artifacts()
    run_root = _write_portfolio_chunks_for_test(
        artifacts,
        spec=spec,
        output_spec=_report_spec(
            "portfolio_outputs",
            spec.portfolio_id,
            "portfolio",
        ),
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    assert run_root.as_posix().endswith("outputs/portfolio_demo_report")
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
    ]


def test_portfolio_outputs_write_row_artifacts_from_chunks(
    tmp_path: Path,
) -> None:
    spec = _portfolio_spec()
    artifacts = _portfolio_artifacts()
    row_artifacts = (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
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
    run_root = outputs._write_portfolio_output_chunks(
        chunks,
        summary_factory=lambda: {
            artifact_name: artifacts[artifact_name]
            for artifact_name in artifacts
            if artifact_name not in row_artifacts
        },
        spec=spec,
        output_spec=_report_spec(
            "portfolio_outputs",
            spec.portfolio_id,
            "portfolio",
        ),
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["metadata"]["row_counts"]["positions"] == 2
    assert manifest["metadata"]["row_counts"]["turnover"] == 1
    assert len(pd.read_parquet(run_root / "positions.parquet")) == 2
    assert len(pd.read_parquet(run_root / "portfolio_returns.parquet")) == 1


def test_portfolio_outputs_reject_bad_frames(tmp_path: Path) -> None:
    spec = _portfolio_spec()
    report_spec = _report_spec(
        "portfolio_outputs",
        spec.portfolio_id,
        "portfolio",
    )
    layout = StorageLayout(project_root=tmp_path)

    missing_positions = _portfolio_artifacts()
    missing_positions["positions"] = missing_positions["positions"].drop(
        columns=["asset"]
    )
    with pytest.raises(ValueError, match="missing_portfolio_positions_columns"):
        _write_portfolio_chunks_for_test(
            missing_positions,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    extra_returns = _portfolio_artifacts()
    extra_returns["portfolio_returns"] = extra_returns["portfolio_returns"].assign(
        extra=1
    )
    with pytest.raises(ValueError, match="unregistered_portfolio_portfolio_returns"):
        _write_portfolio_chunks_for_test(
            extra_returns,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    wrong_return_source = _portfolio_artifacts()
    wrong_return_source["portfolio_returns"] = wrong_return_source[
        "portfolio_returns"
    ].assign(return_source_column="other_return")
    with pytest.raises(ValueError, match="portfolio_return_source_column_mismatch"):
        _write_portfolio_chunks_for_test(
            wrong_return_source,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    wrong_signal = _portfolio_artifacts()
    wrong_signal["portfolio_returns"] = wrong_signal["portfolio_returns"].assign(
        signal="other_signal"
    )
    with pytest.raises(ValueError, match="portfolio_artifact_signal_mismatch"):
        _write_portfolio_chunks_for_test(
            wrong_signal,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )


def test_portfolio_outputs_clean_temp_bundle_directory_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    spec = _portfolio_spec()
    layout = StorageLayout(project_root=tmp_path)
    report_spec = _report_spec(
        "portfolio_outputs",
        spec.portfolio_id,
        "portfolio",
    )
    run_parent = tmp_path / "outputs"

    def fail_write(frame, path, **kwargs):  # noqa: ANN001
        del frame, kwargs
        Path(path).write_text("partial", encoding="utf-8")
        raise RuntimeError("portfolio parquet failed")

    monkeypatch.setattr(outputs, "_write_parquet", fail_write)

    with pytest.raises(RuntimeError, match="portfolio parquet failed"):
        _write_portfolio_chunks_for_test(
            _portfolio_artifacts(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )
    assert not (run_parent / report_spec.output_id).exists()
    assert not list(run_parent.glob(".tmp-*"))


def test_report_parquet_writer_cleans_temp_on_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"value": [1]})

    def fail_to_parquet(self, path, **kwargs):  # noqa: ANN001
        Path(path).write_text("temporary", encoding="utf-8")
        raise RuntimeError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    with pytest.raises(RuntimeError, match="write failed"):
        outputs._write_parquet(
            frame,
            tmp_path / "report.parquet",
        )
    assert not list(tmp_path.glob("*.tmp-*"))


def _prediction_spec() -> PredictionSpec:
    return PredictionSpec(
        prediction_id="prediction_demo",
        model_id="model_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )


def _portfolio_spec() -> PortfolioSpec:
    return PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(_signal_spec(),),
        return_source_column="label",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        weight_config=WeightConfig(method="rank_weighted_quantile"),
    )


def _portfolio_artifacts() -> dict[str, pd.DataFrame]:
    timestamp = pd.Timestamp("2024-01-02")
    return {
        "positions": pd.DataFrame(
            {
                "timestamp": [timestamp, timestamp],
                "asset": ["asset_a", "asset_b"],
                "signal": ["signal", "signal"],
                "position": [1.0, -1.0],
            }
        ),
        "turnover": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "turnover": [2.0],
            }
        ),
        "cost_breakdown": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "buy_turnover": [1.0],
                "sell_turnover": [1.0],
                "turnover": [2.0],
                "buy_cost": [0.0],
                "sell_cost": [0.0],
                "slippage_cost": [0.0],
                "spread_cost": [0.0],
                "impact_cost": [0.0],
                "total_cost": [0.0],
                "cost_model": ["scalar_cost_bps"],
            }
        ),
        "portfolio_returns": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "return_source_column": ["label"],
                "portfolio_mode": ["long_short"],
                "long_return": [0.02],
                "short_return": [-0.01],
                "gross_portfolio_return": [0.03],
                "turnover": [2.0],
                "cost": [0.0],
                "net_portfolio_return": [0.03],
                "long_count": [1],
                "short_count": [1],
            }
        ),
        "cumulative_returns": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "gross_portfolio_return": [0.03],
                "net_portfolio_return": [0.03],
                "cost": [0.0],
                "cumulative_gross_return": [0.03],
                "cumulative_net_return": [0.03],
                "cumulative_cost": [0.0],
                "high_water_mark": [0.03],
                "drawdown": [0.0],
            }
        ),
        "performance_summary": pd.DataFrame(
            {
                "signal": ["signal"],
                "periods": [1],
                "final_cumulative_gross_return": [0.03],
                "final_cumulative_net_return": [0.03],
                "mean_net_portfolio_return": [0.03],
                "volatility_net_portfolio_return": [0.0],
                "downside_volatility": [0.0],
                "sharpe_like": [0.0],
                "sortino_like": [0.0],
                "calmar_like": [0.0],
                "best_period_return": [0.03],
                "worst_period_return": [0.03],
                "skew": [0.0],
                "kurtosis": [0.0],
                "var_95": [0.03],
                "cvar_95": [0.03],
                "tail_ratio": [0.0],
                "positive_return_rate": [1.0],
                "negative_return_rate": [0.0],
                "win_rate": [1.0],
                "loss_rate": [0.0],
                "profit_factor": [0.0],
                "payoff_ratio": [0.0],
                "average_win": [0.03],
                "average_loss": [0.0],
                "max_consecutive_wins": [1],
                "max_consecutive_losses": [0],
                "max_drawdown": [0.0],
                "drawdown_duration": [0],
                "max_drawdown_start": [None],
                "max_drawdown_valley": [None],
                "max_drawdown_recovery": [None],
                "mean_turnover": [2.0],
                "max_turnover": [2.0],
                "total_cost": [0.0],
                "mean_cost": [0.0],
                "cost_to_gross_return": [0.0],
                "cost_to_net_return": [0.0],
            }
        ),
        "exposure_summary": pd.DataFrame(
            {
                "signal": ["signal"],
                "mean_gross_exposure": [2.0],
                "mean_net_exposure": [0.0],
                "max_gross_exposure": [2.0],
                "max_abs_net_exposure": [0.0],
                "mean_long_count": [1.0],
                "mean_short_count": [1.0],
                "mean_position_count": [2.0],
            }
        ),
        "concentration": pd.DataFrame(
            {
                "signal": ["signal"],
                "periods": [1],
                "mean_hhi": [0.5],
                "max_hhi": [0.5],
                "mean_top_weight": [0.5],
                "max_top_weight": [0.5],
                "mean_position_count": [2.0],
            }
        ),
        "period_breakdown": pd.DataFrame(
            {
                "signal": ["signal"],
                "year": [2024],
                "month": [1],
                "periods": [1],
                "gross_portfolio_return": [0.03],
                "net_portfolio_return": [0.03],
                "cost": [0.0],
                "turnover": [2.0],
                "hit_rate": [1.0],
            }
        ),
        "diagnostics": pd.DataFrame(
            {
                "timestamp": [timestamp],
                "signal": ["signal"],
                "method": ["rank_weighted_quantile"],
                "universe_count": [2],
                "eligible_count": [2],
                "can_enter_count": [2],
                "can_exit_count": [2],
                "entry_blocked_count": [0],
                "exit_blocked_count": [0],
                "long_count": [1],
                "short_count": [1],
                "fallback_reason": [None],
            }
        ),
    }


def _write_portfolio_chunks_for_test(
    artifacts: dict[str, pd.DataFrame],
    *,
    spec: PortfolioSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    row_artifacts = _portfolio_row_artifact_names()
    return outputs._write_portfolio_output_chunks(
        ({artifact_name: artifacts[artifact_name] for artifact_name in row_artifacts},),
        summary_factory=lambda: {
            artifact_name: artifacts[artifact_name]
            for artifact_name in artifacts
            if artifact_name not in row_artifacts
        },
        spec=spec,
        output_spec=output_spec,
        layout=layout,
    )


def _portfolio_row_artifact_names() -> tuple[str, ...]:
    return (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
    )


def _prediction_frame(*, with_metadata: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02"]),
            "prediction_score": [0.1],
        }
    )
    if with_metadata:
        frame.attrs["model_metadata"] = _model_metadata()
        frame.attrs["prediction_metadata"] = _prediction_metadata()
    return frame


def _model_metadata() -> dict[str, object]:
    return {
        "contract_version": "0.1",
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2024-01-01",
        "train_end": "2024-01-01",
        "test_start": "2024-01-02",
        "test_end": None,
        "oos_start": "2024-01-02",
        "oos_end": "2024-01-02",
        "input_factor_columns": ["factor"],
        "input_feature_columns": ["feature"],
        "input_columns": ["factor", "feature"],
        "input_count": 2,
        "label_column": "label",
        "hyperparameters": {},
        "random_seed": None,
        "max_train_rows": None,
        "purge_gap_steps": 0,
        "dependency_status": {"status": "available", "package": "sklearn"},
        "coefficient_count": 3,
        "feature_importance": {"factor": 0.1, "feature": 0.2},
    }


def _prediction_metadata() -> dict[str, object]:
    return {
        "contract_version": "0.1",
        "prediction_mode": "minimal_oos",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_dropna": 2,
        "prediction_rows": 1,
        "train_end": "2024-01-01",
        "test_start": "2024-01-02",
        "requested_test_end": None,
        "effective_test_end": "2024-01-02",
        "is_oos": True,
        "is_final_oos_window": True,
        "input_columns": ["factor", "feature"],
        "input_count": 2,
        "prediction_column": "prediction_score",
    }


def _signal_spec() -> SignalSpec:
    return SignalSpec(
        signal_id="signal",
        source_column="feature",
        source_kind="feature",
    )


def _report_spec(
    output_definition_id: str,
    run_id: str,
    run_type: str,
) -> OutputSpec:
    return OutputSpec(
        output_id=f"{run_id}_report",
        output_definition_id=output_definition_id,
        run_id=run_id,
        market_id=MarketId("a_share"),
        run_type=run_type,
        scope=OutputScope.LOCAL_ONLY,
    )


def _evaluation_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "signal": ["signal"],
            "label": ["label"],
            "ic": [0.1],
            "rankic": [0.2],
            "n_obs": [2],
        }
    )


def _evaluation_grouped_label_values_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "signal": ["signal"],
            "label": ["label"],
            "group": [1],
            "mean_label_value": [0.1],
            "n_obs": [2],
        }
    )


def _evaluation_ic_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": ["signal"],
            "label": ["label"],
            "mean_ic": [0.1],
            "std_ic": [0.0],
            "icir": [0.0],
            "positive_ic_rate": [1.0],
            "valid_ic_count": [1],
            "mean_rankic": [0.2],
            "std_rankic": [0.0],
            "rankicir": [0.0],
            "positive_rankic_rate": [1.0],
            "valid_rankic_count": [1],
        }
    )


def _evaluation_group_spread_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": ["signal"],
            "label": ["label"],
            "top_group_mean_label_value": [0.1],
            "bottom_group_mean_label_value": [0.0],
            "top_bottom_label_value_spread_mean": [0.1],
            "top_bottom_label_value_spread_hit_rate": [1.0],
            "group_monotonicity_score": [1.0],
        }
    )


def _evaluation_diagnostics() -> EvaluationDiagnostics:
    return EvaluationDiagnostics(
        filter_mode="eligible",
        filter_columns_used=("eligible",),
        filter_definition="eligible == True",
        rows_input=2,
        rows_after_filter_mode=2,
        rows_after_signal_label_dropna=2,
        metric_rows=1,
        grouped_label_value_rows=1,
        ic_summary_rows=1,
        group_spread_summary_rows=1,
        timestamp_count=1,
        min_timestamp="2024-01-02",
        max_timestamp="2024-01-02",
        n_groups=2,
        skipped_timestamp_count=0,
    )

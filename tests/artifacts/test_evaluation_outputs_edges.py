from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts import outputs
from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_evaluation_diagnostics,
    make_output_edge_report_spec,
)
from tests.helpers.report_output_frames import (
    make_artifact_evaluation_group_spread_summary_frame,
    make_artifact_evaluation_grouped_label_values_frame,
    make_artifact_evaluation_ic_summary_frame,
    make_artifact_evaluation_metrics_frame,
    make_artifact_signal_spec,
)


def test_evaluation_outputs_validation_rejects_bad_inputs(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )

    with pytest.raises(ValueError, match="missing_evaluation_metrics_columns"):
        outputs.write_evaluation_outputs(
            pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-02")]}),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_ic_summary_columns"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=pd.DataFrame({"signal": ["signal"]}),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )

    with pytest.raises(ValueError, match="missing_evaluation_group_spread_summary"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=pd.DataFrame({"signal": ["signal"]}),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_evaluation_outputs_rejects_inconsistent_diagnostics(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )
    layout = StorageLayout(project_root=tmp_path)

    bad_metric_rows = replace(make_output_edge_evaluation_diagnostics(), metric_rows=2)
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_metric_rows,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_filter_mode = replace(
        make_output_edge_evaluation_diagnostics(),
        filter_mode="unfiltered",
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_filter_mode,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_count = replace(
        make_output_edge_evaluation_diagnostics(),
        timestamp_count=2,
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_timestamp_count,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    skipped_metrics = make_artifact_evaluation_metrics_frame().assign(
        n_obs=[1]
    )
    bad_skipped = replace(
        make_output_edge_evaluation_diagnostics(),
        skipped_timestamp_count=0,
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            skipped_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=bad_skipped,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_timestamp_metrics = make_artifact_evaluation_metrics_frame().assign(
        timestamp=["bad-date"]
    )
    bad_timestamp_diagnostics = replace(
        make_output_edge_evaluation_diagnostics(),
        timestamp_count=0,
        min_timestamp=None,
        max_timestamp=None,
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            bad_timestamp_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
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
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )
    layout = StorageLayout(project_root=tmp_path)

    duplicate_metrics = pd.concat(
        [
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_metrics_frame(),
        ],
        ignore_index=True,
    )
    duplicate_diagnostics = replace(
        make_output_edge_evaluation_diagnostics(),
        metric_rows=2,
        rows_after_signal_label_dropna=4,
    )
    with pytest.raises(ValueError, match="duplicate_evaluation_metrics_keys"):
        outputs.write_evaluation_outputs(
            duplicate_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=duplicate_diagnostics,
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_feature_metrics = make_artifact_evaluation_metrics_frame().assign(
        signal="other"
    )
    with pytest.raises(ValueError, match="evaluation_artifact_signal_label_mismatch"):
        outputs.write_evaluation_outputs(
            bad_feature_metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_group = make_artifact_evaluation_grouped_label_values_frame().assign(group=3)
    with pytest.raises(ValueError, match="evaluation_group_out_of_range"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            bad_group,
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=layout,
        )

    bad_row_chain = replace(
        make_output_edge_evaluation_diagnostics(),
        rows_after_filter_mode=1,
    )
    with pytest.raises(ValueError, match="evaluation_diagnostics_inconsistent"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
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
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )

    output_bundle = outputs.write_evaluation_outputs(
        make_artifact_evaluation_metrics_frame(),
        make_artifact_evaluation_grouped_label_values_frame(),
        ic_summary=make_artifact_evaluation_ic_summary_frame(),
        group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
        evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
        spec=spec,
        output_spec=report_spec,
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))

    assert (output_bundle / "grouped_label_values.parquet").is_file()
    assert (output_bundle / "ic_summary.parquet").is_file()
    assert (output_bundle / "group_spread_summary.parquet").is_file()
    assert not (output_bundle / "grouped_returns.parquet").exists()
    assert manifest["schema_id"] == "qrp-output-manifest"
    assert manifest["files"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
        "summary.md",
    ]


def test_evaluation_outputs_writes_share_safe_summary_md(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    report_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )

    output_bundle = outputs.write_evaluation_outputs(
        make_artifact_evaluation_metrics_frame(),
        make_artifact_evaluation_grouped_label_values_frame(),
        ic_summary=make_artifact_evaluation_ic_summary_frame(),
        group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
        evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
        spec=spec,
        output_spec=report_spec,
        layout=StorageLayout(project_root=tmp_path),
    )
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    summary = (output_bundle / "summary.md").read_text(encoding="utf-8")

    assert manifest["files"] == [
        "metrics.parquet",
        "grouped_label_values.parquet",
        "ic_summary.parquet",
        "group_spread_summary.parquet",
        "summary.md",
    ]
    assert "# Evaluation Summary" in summary
    assert "- run_id: `eval_demo`" in summary
    assert "- market_id: `a_share`" in summary
    assert "- research_dataset_id: `research_demo`" in summary
    assert "- rows_input: `2`" in summary
    assert "- rows_after_filter: `2`" in summary
    assert "- timestamp_count: `1`" in summary
    assert "- min_timestamp: `2024-01-02`" in summary
    assert "- max_timestamp: `2024-01-02`" in summary
    assert "| signal | label | mean_ic | mean_rankic |" in summary
    assert "| signal | label | top_bottom_label_value_spread_mean |" in summary
    assert "source_asset_id" not in summary
    assert "data/raw" not in summary

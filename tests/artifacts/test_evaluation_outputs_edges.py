from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts import (
    _evaluation_output_validation as evaluation_output_validation,
)
from quant_research_platform.artifacts import (
    _evaluation_outputs as evaluation_outputs_owner,
)
from quant_research_platform.artifacts import (
    outputs,
)
from quant_research_platform.artifacts._output_manifest import (
    evaluation_output_metadata,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.policy.share_safety import (
    MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS,
)
from quant_research_platform.research.evaluations.engine import (
    _EvaluationStreamingState,
    _iter_evaluation_output_chunks,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
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
from tests.helpers.runtime_resources import capture_test_policy


def _static_managed_summary_factory():
    return (
        make_output_edge_evaluation_diagnostics(),
        make_artifact_evaluation_ic_summary_frame(),
        make_artifact_evaluation_group_spread_summary_frame(),
        pd.DataFrame(
            {
                "signal": ["signal"],
                "label": ["label"],
                "mean_n_obs": [2.0],
                "min_n_obs": [2],
                "non_empty_period_count": [1],
            }
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "diagnostics_n_groups",
        "filter_mode",
        "filter_columns",
        "filter_definition",
        "signal_source_kind",
        "signal_direction",
        "signal_transform",
        "signal_filter_column",
        "signal_label_overlap",
    ),
)
def test_evaluation_metadata_reuses_complete_business_owner_graph(
    mutation: str,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    metadata = evaluation_output_metadata(
        spec=spec,
        evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
    )
    assert (
        evaluation_output_validation.validate_evaluation_output_metadata_shape(metadata)
        is metadata
    )
    metadata = deepcopy(metadata)
    diagnostics = metadata["evaluation_diagnostics"]
    if mutation == "diagnostics_n_groups":
        diagnostics["n_groups"] = 1
    elif mutation == "filter_mode":
        metadata["filter_mode"] = "unknown"
        diagnostics["filter_mode"] = "unknown"
    elif mutation == "filter_columns":
        diagnostics["filter_columns_used"] = []
    elif mutation == "filter_definition":
        diagnostics["filter_definition"] = "wrong"
    elif mutation == "signal_source_kind":
        metadata["signals"][0]["source_kind"] = "unknown"
    elif mutation == "signal_direction":
        metadata["signals"][0]["direction"] = "unknown"
    elif mutation == "signal_transform":
        metadata["signals"][0]["transform"] = "unknown"
    elif mutation == "signal_filter_column":
        metadata["signals"][0]["source_column"] = "eligible"
    else:
        metadata["signals"][0]["source_column"] = metadata["label_columns"][0]

    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        evaluation_output_validation.validate_evaluation_output_metadata_shape(metadata)


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
        n_obs=[1],
        ic=[None],
        rankic=[None],
        ic_reason=["insufficient_observations"],
        rankic_reason=["insufficient_observations"],
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


def test_evaluation_diagnostics_accept_equivalent_timezone_representations() -> None:
    metrics = pd.concat(
        [
            make_artifact_evaluation_metrics_frame().assign(
                timestamp=pd.Timestamp(
                    "2024-01-02T12:00:00",
                    tz=timezone(timedelta(hours=1)),
                ),
                n_obs=1,
            ),
            make_artifact_evaluation_metrics_frame().assign(
                timestamp=pd.Timestamp(
                    "2024-01-02T12:01:00",
                    tz=ZoneInfo("Europe/Paris"),
                ),
                n_obs=1,
            ),
        ],
        ignore_index=True,
    )
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    diagnostics = replace(
        make_output_edge_evaluation_diagnostics(),
        metric_rows=2,
        timestamp_count=2,
        skipped_timestamp_count=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluation_output_validation.validate_evaluation_diagnostics_consistency(
            metrics=metrics,
            grouped_label_values=make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=diagnostics,
            spec=spec,
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


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("bad_string", "invalid_evaluation_numeric_values"),
        ("infinite", "invalid_evaluation_numeric_values"),
        ("out_of_range", "invalid_evaluation_correlation_values"),
        ("missing_reason", "invalid_evaluation_statistic_reason"),
        ("reason_with_value", "invalid_evaluation_statistic_reason"),
        ("unknown_reason", "invalid_evaluation_statistic_reason"),
        ("insufficient_reason_with_value", "invalid_evaluation_statistic_reason"),
        (
            "insufficient_reason_with_enough_observations",
            "invalid_evaluation_statistic_reason",
        ),
        ("summary_mean_out_of_range", "invalid_evaluation_correlation_values"),
        ("valid_ic_count_negative", "invalid_evaluation_summary_values"),
        ("std_negative", "invalid_evaluation_summary_values"),
        ("positive_rate_out_of_range", "invalid_evaluation_summary_values"),
        ("spread_hit_rate_out_of_range", "invalid_evaluation_summary_values"),
        ("monotonicity_out_of_range", "invalid_evaluation_summary_values"),
    ),
)
def test_evaluation_artifact_numeric_and_reason_tampering(
    mutation: str,
    expected_error: str,
) -> None:
    metrics = make_artifact_evaluation_metrics_frame().copy()
    ic_summary = make_artifact_evaluation_ic_summary_frame().copy()
    group_spread_summary = make_artifact_evaluation_group_spread_summary_frame().copy()
    if mutation == "bad_string":
        metrics["ic"] = metrics["ic"].astype(object)
        metrics.loc[0, "ic"] = "0.1"
    elif mutation == "infinite":
        metrics.loc[0, "ic"] = float("inf")
    elif mutation == "out_of_range":
        metrics.loc[0, "ic"] = 1.1
    elif mutation == "missing_reason":
        metrics.loc[0, "ic"] = float("nan")
        metrics.loc[0, "ic_reason"] = None
    elif mutation == "reason_with_value":
        metrics.loc[0, "ic_reason"] = "zero_variance"
    elif mutation == "unknown_reason":
        metrics.loc[0, "ic"] = float("nan")
        metrics.loc[0, "ic_reason"] = "made_up_reason"
    elif mutation == "insufficient_reason_with_value":
        metrics.loc[0, "n_obs"] = 1
    elif mutation == "insufficient_reason_with_enough_observations":
        metrics.loc[0, "ic"] = float("nan")
        metrics.loc[0, "ic_reason"] = "insufficient_observations"
    elif mutation == "summary_mean_out_of_range":
        ic_summary.loc[0, "mean_ic"] = 1.1
    elif mutation == "valid_ic_count_negative":
        ic_summary.loc[0, "valid_ic_count"] = -1
    elif mutation == "std_negative":
        ic_summary.loc[0, "std_ic"] = -1.0
    elif mutation == "positive_rate_out_of_range":
        ic_summary.loc[0, "positive_ic_rate"] = 1.1
    elif mutation == "spread_hit_rate_out_of_range":
        group_spread_summary.loc[0, "top_bottom_label_value_spread_hit_rate"] = 1.1
    elif mutation == "monotonicity_out_of_range":
        group_spread_summary.loc[0, "group_monotonicity_score"] = -1.1
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    with pytest.raises(ValueError, match=expected_error):
        evaluation_output_validation.validate_evaluation_artifact_structure(
            metrics=metrics,
            grouped_label_values=make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=ic_summary,
            group_spread_summary=group_spread_summary,
            spec=spec,
            n_groups=2,
        )


@pytest.mark.parametrize("bad_value", [[], {}, set(), None, pd.NA, ""])
@pytest.mark.parametrize("column", ["signal", "label"])
def test_evaluation_outputs_rejects_invalid_signal_label_membership_values(
    tmp_path: Path,
    column: str,
    bad_value: object,
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
    metrics = make_artifact_evaluation_metrics_frame()
    metrics[column] = metrics[column].astype(object)
    metrics.iat[0, metrics.columns.get_loc(column)] = bad_value

    with pytest.raises(ValueError, match="evaluation_artifact_signal_label_mismatch"):
        outputs.write_evaluation_outputs(
            metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


@pytest.mark.parametrize(
    ("column", "padded_value"),
    [
        ("signal", "signal "),
        ("label", "label "),
        ("signal", "\tsignal"),
        ("label", "label\n"),
    ],
)
def test_evaluation_outputs_rejects_padded_signal_label_membership_values(
    tmp_path: Path,
    column: str,
    padded_value: str,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )
    metrics = make_artifact_evaluation_metrics_frame().assign(**{column: padded_value})

    with pytest.raises(ValueError, match="evaluation_artifact_signal_label_mismatch"):
        outputs.write_evaluation_outputs(
            metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "evaluation_outputs",
                spec.run_id,
                "evaluation",
            ),
            layout=StorageLayout(project_root=tmp_path),
        )


@pytest.mark.parametrize("n_obs", [2.5, True, -1, float("inf")])
def test_evaluation_outputs_rejects_invalid_metric_observation_counts(
    tmp_path: Path,
    n_obs: object,
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
    metrics = make_artifact_evaluation_metrics_frame().assign(n_obs=[n_obs])

    with pytest.raises(ValueError, match="invalid_evaluation_observation_counts"):
        outputs.write_evaluation_outputs(
            metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_evaluation_outputs_rejects_fractional_grouped_label_group(
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
    grouped = make_artifact_evaluation_grouped_label_values_frame().assign(group=1.5)

    with pytest.raises(ValueError, match="evaluation_group_out_of_range"):
        outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            grouped,
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
        )


def test_evaluation_outputs_rejects_non_finite_summary_numbers(
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
    metrics = make_artifact_evaluation_metrics_frame().assign(ic=[float("inf")])

    with pytest.raises(ValueError, match="invalid_evaluation_numeric_values"):
        outputs.write_evaluation_outputs(
            metrics,
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
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
    assert "## Signal/Label Coverage Summary" in summary
    assert (
        "| signal | label | mean_n_obs | min_n_obs | non_empty_period_count |"
        in summary
    )
    assert "| signal | label | 2 | 2 | 1 |" in summary
    assert "| signal | label | top_bottom_label_value_spread_mean |" in summary
    assert "source_asset_id" not in summary
    assert "data/raw" not in summary


@pytest.mark.parametrize(
    ("extra_characters", "should_pass"),
    [(0, True), (1, False)],
)
def test_evaluation_outputs_enforces_evidence_line_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_characters: int,
    should_pass: bool,
) -> None:
    monkeypatch.setattr(
        evaluation_outputs_owner,
        "_render_evaluation_summary",
        lambda **_kwargs: (
            "x" * (MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS + extra_characters)
        ),
    )
    spec = EvaluationSpec(
        run_id="eval_line_bound",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )

    def write_outputs() -> Path:
        return outputs.write_evaluation_outputs(
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            ic_summary=make_artifact_evaluation_ic_summary_frame(),
            group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
            spec=spec,
            output_spec=make_output_edge_report_spec(
                "evaluation_outputs",
                spec.run_id,
                "evaluation",
            ),
            layout=StorageLayout(project_root=tmp_path),
        )

    if should_pass:
        assert write_outputs().joinpath("summary.md").is_file()
    else:
        with pytest.raises(
            ValueError,
            match="evaluation_summary_share_unsafe: evidence_capacity_exceeded",
        ):
            write_outputs()


def test_evaluation_outputs_feeds_bounded_encoded_summary_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = "safe metric\n" * 100_000
    observed_sizes: list[int] = []
    real_scan = evaluation_outputs_owner.evidence_text_stream_unsafe_reason

    def scan(chunks) -> str | None:
        captured = []
        for chunk in chunks:
            observed_sizes.append(len(chunk))
            captured.append(bytes(chunk))
        assert b"".join(captured).decode("utf-8") == rendered
        return real_scan(captured)

    monkeypatch.setattr(
        evaluation_outputs_owner,
        "_render_evaluation_summary",
        lambda **_kwargs: rendered,
    )
    monkeypatch.setattr(
        evaluation_outputs_owner,
        "evidence_text_stream_unsafe_reason",
        scan,
    )
    spec = EvaluationSpec(
        run_id="eval_bounded_feed",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(make_artifact_signal_spec(),),
        label_columns=("label",),
    )

    output = outputs.write_evaluation_outputs(
        make_artifact_evaluation_metrics_frame(),
        make_artifact_evaluation_grouped_label_values_frame(),
        ic_summary=make_artifact_evaluation_ic_summary_frame(),
        group_spread_summary=make_artifact_evaluation_group_spread_summary_frame(),
        evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
        spec=spec,
        output_spec=make_output_edge_report_spec(
            "evaluation_outputs", spec.run_id, "evaluation"
        ),
        layout=StorageLayout(project_root=tmp_path),
    )

    assert output.joinpath("summary.md").is_file()
    assert max(observed_sizes) <= MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS


def test_summary_markdown_escapes_pipe_characters(tmp_path: Path) -> None:
    spec = EvaluationSpec(
        run_id="eval_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            replace(
                make_artifact_signal_spec(),
                signal_id="sig|nal",
                source_column="sig|nal",
            ),
        ),
        label_columns=("lab|el",),
    )
    report_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )
    metrics = make_artifact_evaluation_metrics_frame().assign(
        signal="sig|nal",
        label="lab|el",
    )
    grouped = make_artifact_evaluation_grouped_label_values_frame().assign(
        signal="sig|nal",
        label="lab|el",
    )
    ic_summary = make_artifact_evaluation_ic_summary_frame().assign(
        signal="sig|nal",
        label="lab|el",
    )
    spread = make_artifact_evaluation_group_spread_summary_frame().assign(
        signal="sig|nal",
        label="lab|el",
    )

    output_bundle = outputs.write_evaluation_outputs(
        metrics,
        grouped,
        ic_summary=ic_summary,
        group_spread_summary=spread,
        evaluation_diagnostics=make_output_edge_evaluation_diagnostics(),
        spec=spec,
        output_spec=report_spec,
        layout=StorageLayout(project_root=tmp_path),
    )
    summary = (output_bundle / "summary.md").read_text(encoding="utf-8")

    assert "sig\\|nal" in summary
    assert "lab\\|el" in summary
    assert "| sig|nal |" not in summary
    assert "| lab|el |" not in summary
    assert "source_asset_id" not in summary
    assert "data/raw" not in summary


def test_managed_evaluation_admits_fixed_writers_before_first_pull(
    tmp_path: Path,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_admission",
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
    pulled: list[bool] = []

    def chunks():
        pulled.append(True)
        yield (
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            {
                "rows_input": 2,
                "rows_after_filter_mode": 2,
                "filter_columns_used": ["eligible"],
                "filter_definition": "eligible == True",
                "timestamp_count": 1,
                "min_timestamp": "2024-01-02",
                "max_timestamp": "2024-01-02",
                "skipped_timestamp_count": 0,
            },
        )

    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=64 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            evaluation_outputs_owner._write_evaluation_outputs_from_chunks(
                chunks(),
                summary_factory=_static_managed_summary_factory,
                spec=spec,
                output_spec=report_spec,
                layout=StorageLayout(project_root=tmp_path),
                n_groups=2,
            )

    assert pulled == []
    assert policy.used_memory_bytes == 0


def test_managed_evaluation_disk_preflight_happens_before_first_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_disk_admission",
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
    pulled: list[bool] = []
    observed: dict[str, object] = {}

    def chunks():
        pulled.append(True)
        yield (
            make_artifact_evaluation_metrics_frame(),
            make_artifact_evaluation_grouped_label_values_frame(),
            {},
        )

    def reject_disk(path: Path, *, reserve_bytes: int) -> None:
        observed["path"] = path
        observed["reserve_bytes"] = reserve_bytes
        raise OSError("evaluation_disk_preflight")

    monkeypatch.setattr(
        evaluation_outputs_owner,
        "require_free_disk_headroom",
        reject_disk,
    )

    with pytest.raises(OSError, match="evaluation_disk_preflight"):
        evaluation_outputs_owner._write_evaluation_outputs_from_chunks(
            chunks(),
            summary_factory=_static_managed_summary_factory,
            spec=spec,
            output_spec=report_spec,
            layout=StorageLayout(project_root=tmp_path),
            n_groups=2,
        )

    assert pulled == []
    assert observed["reserve_bytes"] == (
        2 * evaluation_outputs_owner._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
    )
    assert not (tmp_path / "outputs" / report_spec.output_id).exists()
    assert not list((tmp_path / "outputs").glob(".a-*"))


def test_managed_evaluation_failure_closes_partial_writer_and_preserves_primary(
    tmp_path: Path,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_failure",
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
    primary = RuntimeError("managed_evaluation_primary")
    closed: list[bool] = []

    def chunks():
        try:
            yield (
                make_artifact_evaluation_metrics_frame(),
                make_artifact_evaluation_grouped_label_values_frame(),
                {
                    "rows_input": 2,
                    "rows_after_filter_mode": 2,
                    "filter_columns_used": ["eligible"],
                    "filter_definition": "eligible == True",
                    "timestamp_count": 1,
                    "min_timestamp": "2024-01-02",
                    "max_timestamp": "2024-01-02",
                    "skipped_timestamp_count": 0,
                },
            )
            raise primary
        finally:
            closed.append(True)

    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(RuntimeError) as raised:
            evaluation_outputs_owner._write_evaluation_outputs_from_chunks(
                chunks(),
                summary_factory=_static_managed_summary_factory,
                spec=spec,
                output_spec=report_spec,
                layout=StorageLayout(project_root=tmp_path),
                n_groups=2,
            )

    assert raised.value is primary
    assert closed == [True]
    assert policy.used_memory_bytes == 0
    assert not (tmp_path / "outputs" / report_spec.output_id).exists()


def test_managed_evaluation_empty_output_failure_closes_summary_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_empty_failure",
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
    primary = OSError("empty_output_write_failed")
    monkeypatch.setattr(
        evaluation_outputs_owner,
        "write_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(OSError) as raised:
            evaluation_outputs_owner._write_evaluation_outputs_from_chunks(
                iter(()),
                summary_factory=_static_managed_summary_factory,
                spec=spec,
                output_spec=report_spec,
                layout=StorageLayout(project_root=tmp_path),
                n_groups=2,
            )
    assert raised.value is primary
    assert policy.used_memory_bytes == 0


def test_managed_evaluation_pre_writer_failure_releases_table_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = EvaluationSpec(
        run_id="eval_pre_writer_failure",
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
    primary = OSError("pre_writer_failure")
    monkeypatch.setattr(
        evaluation_outputs_owner.pq,
        "ParquetWriter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    metrics = make_artifact_evaluation_metrics_frame()
    grouped = make_artifact_evaluation_grouped_label_values_frame()
    policy = capture_test_policy(available_memory_bytes=32 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(OSError) as raised:
            evaluation_outputs_owner._write_evaluation_outputs_from_chunks(
                iter(
                    (
                        (
                            metrics,
                            grouped,
                            {
                                "rows_input": 2,
                                "rows_after_filter_mode": 2,
                                "filter_columns_used": ["eligible"],
                                "filter_definition": "eligible == True",
                                "timestamp_count": 1,
                                "min_timestamp": "2024-01-02",
                                "max_timestamp": "2024-01-02",
                                "skipped_timestamp_count": 0,
                            },
                        ),
                    )
                ),
                summary_factory=_static_managed_summary_factory,
                spec=spec,
                output_spec=report_spec,
                layout=StorageLayout(project_root=tmp_path),
                n_groups=2,
            )
    assert raised.value is primary
    assert policy.used_memory_bytes == 0


def test_managed_evaluation_persists_aggregate_beyond_ram_budget(
    tmp_path: Path,
) -> None:
    signals = tuple(
        replace(
            make_artifact_signal_spec(),
            signal_id=f"signal_{index}",
            source_column=f"signal_{index}",
        )
        for index in range(100)
    )
    labels = ("label_a", "label_b")
    spec = EvaluationSpec(
        run_id="eval_spill",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=signals,
        label_columns=labels,
        n_groups=2,
    )
    report_spec = make_output_edge_report_spec(
        "evaluation_outputs",
        spec.run_id,
        "evaluation",
    )
    total_timestamps = 1_600
    timestamps: list[pd.Timestamp] = []
    asset_ids: list[str] = []
    eligible: list[bool] = []
    labels_by_column = {label: [] for label in labels}
    signals_by_column = {signal.source_column: [] for signal in signals}
    for offset in range(total_timestamps):
        timestamp = pd.Timestamp("2000-01-01") + pd.Timedelta(days=offset)
        for asset_index in range(2):
            timestamps.append(timestamp)
            asset_ids.append(f"asset_{asset_index}")
            eligible.append(True)
            labels_by_column["label_a"].append(offset * 0.001 + asset_index)
            labels_by_column["label_b"].append(offset * 0.002 - asset_index)
            for signal_index, signal in enumerate(signals):
                signals_by_column[signal.source_column].append(
                    (signal_index + 1) * 0.1 + offset * 0.001 + asset_index
                )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": asset_ids,
            "eligible": eligible,
            **labels_by_column,
            **signals_by_column,
        }
    )

    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        state = _EvaluationStreamingState(
            spec=spec,
            n_groups=2,
            resource_policy=policy,
        )
        try:
            chunks = _iter_evaluation_output_chunks(
                (frame,),
                spec=spec,
                signal_columns=tuple(signal.signal_id for signal in signals),
                label_columns=labels,
                n_groups=2,
                asset_column="asset_id",
                resource_policy=policy,
                summary_state=state,
            )
            output_bundle = (
                evaluation_outputs_owner._write_evaluation_outputs_from_chunks(
                    chunks,
                    summary_factory=lambda: (
                        state.diagnostics(),
                        state.ic_summary(),
                        state.group_spread_summary(),
                        state.coverage_summary(),
                    ),
                    spec=spec,
                    output_spec=report_spec,
                    layout=StorageLayout(project_root=tmp_path),
                    n_groups=2,
                )
            )
        finally:
            state.close()

    logical_persisted_bytes = sum(
        parquet_file.metadata.row_group(index).total_byte_size
        for path in (
            output_bundle / "metrics.parquet",
            output_bundle / "grouped_label_values.parquet",
        )
        for parquet_file in (pq.ParquetFile(path),)
        for index in range(parquet_file.metadata.num_row_groups)
    )
    assert logical_persisted_bytes > policy.memory_budget_bytes
    persisted_metrics = pd.read_parquet(output_bundle / "metrics.parquet")
    persisted_grouped = pd.read_parquet(
        output_bundle / "grouped_label_values.parquet"
    )
    assert len(persisted_metrics) == total_timestamps * len(signals) * len(labels)
    assert (
        len(persisted_grouped)
        == total_timestamps * len(signals) * len(labels) * 2
    )
    assert not persisted_metrics.duplicated(
        ["timestamp", "signal", "label"]
    ).any()
    assert not persisted_grouped.duplicated(
        ["timestamp", "signal", "label", "group"]
    ).any()
    first_metric = persisted_metrics.sort_values(
        ["timestamp", "signal", "label"]
    ).iloc[0]
    assert first_metric["signal"] == "signal_0"
    assert first_metric["label"] == "label_a"
    assert first_metric["ic"] == 1.0
    manifest = json.loads((output_bundle / "manifest.json").read_text())
    diagnostics = manifest["metadata"]["evaluation_diagnostics"]
    assert diagnostics["rows_input"] == total_timestamps * 2
    assert diagnostics["metric_rows"] == len(persisted_metrics)
    assert diagnostics["grouped_label_value_rows"] == len(persisted_grouped)
    assert diagnostics["timestamp_count"] == total_timestamps
    assert diagnostics["min_timestamp"] == "2000-01-01"
    assert diagnostics["max_timestamp"] == "2004-05-18"
    assert policy.used_memory_bytes == 0

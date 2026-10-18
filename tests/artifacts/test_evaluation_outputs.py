"""Numeric evaluation publication validates the complete primary-key domain."""

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import (
    read_evaluation_outputs,
    write_evaluation_outputs_from_chunks,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)


def _spec():
    return EvaluationSpec(
        evaluation_id="native_eval",
        market_id="a_share",
        research_dataset_id="explicit_input",
        signals=(
            SignalSpec(
                signal_id="feature_signal",
                source_kind="feature",
                source_column="feature",
            ),
        ),
        label_columns=("label",),
        n_groups=2,
    )


def _frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-02"] * 3),
            "asset_id": ["a", "b", "c"],
            "feature": [1.0, 2.0, 3.0],
            "label": [2.0, 3.0, 5.0],
            "eligible": True,
        }
    )


@pytest.mark.parametrize("duplicate", ["metrics", "grouped_label_values"])
def test_evaluation_rejects_cross_chunk_duplicate_keys_before_publication(
    tmp_path: Path, duplicate
):
    spec = _spec()
    with build_evaluation_result_from_frames(
        (_frame(),),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        original = list(result.output_chunks)
        assert len(original) == 1
        metrics, grouped, metadata = original[0]
        repeated = (
            metrics if duplicate == "metrics" else metrics.iloc[:0],
            grouped if duplicate == "grouped_label_values" else grouped.iloc[:0],
            metadata,
        )
        output = tmp_path / "evaluation"
        with pytest.raises(ValueError, match=f"duplicate_evaluation_{duplicate}_keys"):
            write_evaluation_outputs_from_chunks(
                (*original, repeated),
                summary_factory=result.summaries,
                spec=spec,
                output_path=output,
            )
    assert not output.exists()
    assert not list(tmp_path.glob(".a-*"))


def test_evaluation_publishes_numeric_members_and_reads_only_requested_summaries(
    tmp_path,
    monkeypatch,
):
    spec = _spec()
    output = tmp_path / "evaluation"
    with build_evaluation_result_from_frames(
        (_frame(),),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"feature": ("feature",)},
    ) as result:
        write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=output,
        )
    assert not (output / "summary.md").exists()
    manifest, tables = read_evaluation_outputs(output)
    assert len(tables["metrics"]) == 1
    assert len(tables["ic_summary"]) == 1
    assert manifest["metadata"]["evaluation_diagnostics"]["metric_rows"] == 1
    from quant_research_platform.artifacts.outputs import evaluation

    validate_structure = evaluation.validate_evaluation_artifact_structure
    validated_summary_rows = []

    def record_validation(**kwargs):
        validated_summary_rows.extend(kwargs["ic_summary"].index)
        return validate_structure(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            evaluation, "validate_evaluation_artifact_structure", record_validation
        )
        read_evaluation_outputs(output, tables=("ic_summary",))
    assert len(validated_summary_rows) == len(tables["ic_summary"])

    grouped = tables["grouped_label_values"].copy()
    grouped.loc[0, "n_obs"] += 1
    grouped.to_parquet(output / "grouped_label_values.parquet", index=False)
    read_evaluation_outputs(output, tables=("metrics",))
    read_evaluation_outputs(output, tables=("grouped_label_values",))
    with pytest.raises(
        ValueError, match="evaluation_metric_group_observations_mismatch"
    ):
        read_evaluation_outputs(output, tables=("metrics", "grouped_label_values"))
    tables["grouped_label_values"].to_parquet(
        output / "grouped_label_values.parquet", index=False
    )
    for scope in ("LOCAL_ONLY", "shared"):
        manifest["metadata"]["scope"] = scope
        (output / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
            read_evaluation_outputs(output, tables=("ic_summary",))
    del manifest["metadata"]["scope"]
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "metrics.parquet").write_bytes(b"unselected damaged member")
    _, summary = read_evaluation_outputs(
        output, tables=("ic_summary", "group_spread_summary")
    )
    assert summary["ic_summary"].mean_ic.iloc[0] > 0
    with pytest.raises(ValueError, match="dataset_parquet_footer_invalid"):
        read_evaluation_outputs(output, tables=("metrics",))
    (output / "metrics.parquet").unlink()
    _, selected = read_evaluation_outputs(output, tables=("ic_summary",))
    pd.testing.assert_frame_equal(selected["ic_summary"], tables["ic_summary"])
    with pytest.raises(ValueError, match="output_member_path_invalid"):
        read_evaluation_outputs(output)
    with pytest.raises(ValueError, match="output_member_path_invalid"):
        read_evaluation_outputs(output, tables=("metrics",))
    original = tables["ic_summary"].copy()
    invalid = original.copy()
    invalid["mean_ic"] = 2.0
    invalid.to_parquet(output / "ic_summary.parquet", index=False)
    with pytest.raises(ValueError):
        read_evaluation_outputs(output, tables=("ic_summary",))
    original.to_parquet(output / "ic_summary.parquet", index=False)
    manifest["metadata"]["evaluation_diagnostics"]["metric_rows"] = -1
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="output_manifest_metadata_invalid"):
        read_evaluation_outputs(output, tables=("ic_summary",))

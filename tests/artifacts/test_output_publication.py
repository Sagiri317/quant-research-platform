from __future__ import annotations

import json

import pandas as pd
import pytest

import quant_research_platform.artifacts.outputs.evaluation as evaluation_outputs
import quant_research_platform.artifacts.outputs.prediction as prediction_outputs
from quant_research_platform.artifacts.outputs import (
    write_evaluation_outputs_from_chunks,
    write_prediction_outputs,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.layout import StorageLayout
from tests.artifacts.output_edge_fixtures import (
    make_output_edge_evaluation_diagnostics,
)
from tests.helpers.output_frames import (
    make_prediction_report_spec,
    make_price_volume_evaluation_group_spread_summary_frame,
    make_price_volume_evaluation_grouped_label_values_frame,
    make_price_volume_evaluation_ic_summary_frame,
    make_price_volume_evaluation_metrics_frame,
    make_price_volume_prediction_output_frame,
    make_price_volume_prediction_spec,
    make_price_volume_signal_spec,
)


def _write_prediction_output(frame: pd.DataFrame, *, layout) -> object:
    return write_prediction_outputs(
        output_chunks=(frame,),
        spec=make_price_volume_prediction_spec(),
        output_spec=make_prediction_report_spec(),
        layout=layout,
        model_metadata=frame.attrs["model_metadata"],
        prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
    )


def _write_evaluation_output(
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    *,
    ic_summary=None,
    group_spread_summary=None,
    evaluation_diagnostics,
    spec: EvaluationSpec,
    output_spec,
    layout,
) -> object:
    return write_evaluation_outputs_from_chunks(
        (
            (
                metrics,
                grouped_label_values,
                {
                    "rows_input": evaluation_diagnostics.rows_input,
                    "rows_after_filter_mode": (
                        evaluation_diagnostics.rows_after_filter_mode
                    ),
                    "filter_columns_used": list(
                        evaluation_diagnostics.filter_columns_used
                    ),
                    "filter_definition": evaluation_diagnostics.filter_definition,
                },
            ),
        ),
        summary_factory=lambda: (
            evaluation_diagnostics,
            make_price_volume_evaluation_ic_summary_frame()
            if ic_summary is None
            else ic_summary,
            make_price_volume_evaluation_group_spread_summary_frame()
            if group_spread_summary is None
            else group_spread_summary,
            pd.DataFrame(
                columns=[
                    "signal",
                    "label",
                    "mean_n_obs",
                    "min_n_obs",
                    "non_empty_period_count",
                ]
            ),
        ),
        spec=spec,
        output_spec=output_spec,
        layout=layout,
        n_groups=evaluation_diagnostics.n_groups,
    )


def test_prediction_report_chunk_writer_matches_split_chunks(tmp_path) -> None:
    frame = make_price_volume_prediction_output_frame(row_count=2)
    materialized_root = write_prediction_outputs(
        output_chunks=(frame,),
        spec=make_price_volume_prediction_spec(),
        output_spec=make_prediction_report_spec(),
        layout=StorageLayout(project_root=tmp_path / "materialized"),
        model_metadata=frame.attrs["model_metadata"],
        prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
    )
    chunked_root = write_prediction_outputs(
        output_chunks=(frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=make_price_volume_prediction_spec(),
        output_spec=make_prediction_report_spec(),
        layout=StorageLayout(project_root=tmp_path / "chunked"),
        model_metadata=frame.attrs["model_metadata"],
        prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(materialized_root / "prediction.parquet"),
        pd.read_parquet(chunked_root / "prediction.parquet"),
    )
    assert json.loads((chunked_root / "manifest.json").read_text(encoding="utf-8")) == (
        json.loads((materialized_root / "manifest.json").read_text(encoding="utf-8"))
    )


def test_prediction_chunk_writer_rejects_cross_chunk_duplicates(tmp_path) -> None:
    frame = make_price_volume_prediction_output_frame()
    with pytest.raises(ValueError, match="duplicate_prediction_keys"):
        write_prediction_outputs(
            output_chunks=(frame.copy(), frame.copy()),
            spec=make_price_volume_prediction_spec(),
            output_spec=make_prediction_report_spec(),
            layout=StorageLayout(project_root=tmp_path),
            model_metadata=frame.attrs["model_metadata"],
            prediction_metadata_factory=lambda: frame.attrs["prediction_metadata"],
        )

    assert not (tmp_path / "outputs" / "pred-run").exists()


def test_prediction_outputs_reject_existing_output_bundle_without_writes(
    tmp_path,
):
    output_bundle = tmp_path / "outputs" / "pred-run"
    output_bundle.mkdir(parents=True)
    sentinel = output_bundle / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        _write_prediction_output(
            make_price_volume_prediction_output_frame(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (output_bundle / "prediction.parquet").exists()
    assert not (output_bundle / "manifest.json").exists()


def test_evaluation_outputs_reject_existing_output_bundle_without_writes(
    tmp_path,
):
    output_bundle = tmp_path / "outputs" / "eval-run"
    output_bundle.mkdir(parents=True)
    sentinel = output_bundle / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        _write_evaluation_output(
            make_price_volume_evaluation_metrics_frame(),
            make_price_volume_evaluation_grouped_label_values_frame(),
            ic_summary=make_price_volume_evaluation_ic_summary_frame(),
            group_spread_summary=make_price_volume_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(
                min_timestamp="2014-01-03T00:00:00",
                max_timestamp="2014-01-03T00:00:00",
            ),
            spec=_evaluation_spec(),
            output_spec=_evaluation_outputs_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (output_bundle / "metrics.parquet").exists()
    assert not (output_bundle / "manifest.json").exists()


def test_prediction_outputs_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_bundle = tmp_path / "outputs" / "pred-run"

    def fail_parquet_write(_frame, path, **_kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "partial.parquet").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("simulated_prediction_failure")

    monkeypatch.setattr(
        prediction_outputs,
        "write_parquet_chunks",
        fail_parquet_write,
    )

    with pytest.raises(RuntimeError, match="simulated_prediction_failure"):
        _write_prediction_output(
            make_price_volume_prediction_output_frame(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert not output_bundle.exists()
    assert list(output_bundle.parent.glob(".pred-run.tmp-*")) == []


def test_evaluation_outputs_clean_temp_root_after_mid_write_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    output_bundle = tmp_path / "outputs" / "eval-run"
    call_count = 0

    def fail_second_parquet_write(_frame, path, **_kwargs):
        nonlocal call_count
        call_count += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        if call_count == 2:
            (path.parent / "partial.parquet").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("simulated_evaluation_failure")
        path.write_text("first\n", encoding="utf-8")

    monkeypatch.setattr(
        evaluation_outputs,
        "write_parquet",
        fail_second_parquet_write,
    )

    with pytest.raises(RuntimeError, match="simulated_evaluation_failure"):
        _write_evaluation_output(
            make_price_volume_evaluation_metrics_frame(),
            make_price_volume_evaluation_grouped_label_values_frame(),
            ic_summary=make_price_volume_evaluation_ic_summary_frame(),
            group_spread_summary=make_price_volume_evaluation_group_spread_summary_frame(),
            evaluation_diagnostics=make_output_edge_evaluation_diagnostics(
                min_timestamp="2014-01-03T00:00:00",
                max_timestamp="2014-01-03T00:00:00",
            ),
            spec=_evaluation_spec(),
            output_spec=_evaluation_outputs_spec(),
            layout=StorageLayout(project_root=tmp_path),
        )

    assert not output_bundle.exists()
    assert list(output_bundle.parent.glob(".eval-run.tmp-*")) == []


def _evaluation_spec() -> EvaluationSpec:
    return EvaluationSpec(
        evaluation_id="eval-run",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_price_volume_signal_spec(),),
        label_columns=("forward_return_20d",),
    )


def _evaluation_outputs_spec() -> OutputSpec:
    return OutputSpec(
        output_id="eval-run",
        output_definition_id="evaluation_outputs",
        run_id="test_run",
        market_id=MarketId("a_share"),
        run_type="evaluation",
        scope=OutputScope.LOCAL_ONLY,
    )

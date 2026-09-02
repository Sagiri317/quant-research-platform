from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
import quant_research_platform.research.predictions.runner as prediction_runner_module
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_PRIMARY_KEY,
    PREDICTION_SCORE_COLUMN,
    RAW_PREDICTION_COLUMN,
    PredictionSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.predictions.outputs import prediction_callable
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.research.protocols.a_share import model_specs
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_processed_inputs import (
    make_demo_processed_catalog,
    write_demo_processed_inputs,
)
from tests.helpers.research_specs import (
    make_research_dataset_catalog,
)
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
)
from tests.research.prediction_output_helpers import (
    write_prediction_output_and_read,
)
from tests.research.workflow_test_support import (
    _research_rows,
    _research_rows_for_dates,
    _write_prediction_result,
)


def test_evaluation_prediction_and_report_boundary(tmp_path: Path) -> None:
    frame = _research_rows()
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("mom_20d",),
        label_column="forward_return_20d",
        max_train_rows=len(frame),
    )
    prediction_spec = PredictionSpec(
        prediction_id="pred_demo",
        model_id=model_spec.model_id,
        research_dataset_id="demo",
        train_end="2014-01-02",
        test_start="2014-01-03",
        test_end="2014-01-03",
    )
    estimator = build_estimator_from_model_spec(model_spec)

    with runtime_resource_policy_scope():
        model_result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame.loc[frame["timestamp"] <= pd.Timestamp(prediction_spec.train_end)],),
            model_spec,
            estimator=estimator,
            fitting_cutoff=prediction_spec.train_end,
        )
        try:
            model_result = model_result_owned.value
            prediction, output_bundle, _ = write_prediction_output_and_read(
                tmp_path,
                (frame,),
                prediction_spec,
                model_result.coefficients,
                model_spec.input_columns,
                model_metadata=model_result.metadata,
            )

            assert prediction_spec.prediction_column in prediction.columns
            assert prediction["timestamp"].min() >= pd.Timestamp("2014-01-03")
            assert prediction["timestamp"].max() <= pd.Timestamp("2014-01-03")
            assert output_bundle.as_posix().endswith("outputs/pred_demo")
        finally:
            model_result_owned.lease.close()


def test_a_share_prediction_requires_persisted_quarter_lineage() -> None:
    with pytest.raises(
        ValueError,
        match="a_share_persisted_prediction_requires_quarter_lineage",
    ):
        PredictionSpec(
            prediction_id="a_share_prediction",
            model_id="linear_a_share",
            research_dataset_id="demo",
            train_end="2014-01-02",
            test_start="2014-01-03",
            prediction_mode="a_share_oos",
            prediction_column=RAW_PREDICTION_COLUMN,
        )


def test_a_share_model_ladder_fits_and_predicts_each_retained_algorithm() -> None:
    specs = model_specs("pv_all")
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": [date for date in dates for _ in range(3)],
            "asset_id": [asset for _ in dates for asset in ("a", "b", "c")],
            "eligible": True,
            "model_target": [
                float(date_index + asset_index) / 10.0
                for date_index in range(1, len(dates) + 1)
                for asset_index in range(1, 4)
            ],
            "forward_return_20d_available_at": [
                date for date in dates for _ in range(3)
            ],
        }
    )
    for spec in specs:
        for column in spec.input_columns:
            frame[column] = [
                float((date_index + asset_index) % 7 + 1)
                for date_index in range(len(dates))
                for asset_index in range(3)
            ]
    preflight_model_specs(specs)

    with runtime_resource_policy_scope():
        for spec in specs:
            owned = model_fitting._fit_model_result_owned_from_frames(
                (frame,),
                spec,
                estimator=build_estimator_from_model_spec(spec),
                fitting_cutoff="2020-01-12",
            )
            try:
                result = owned.value
                columns = tuple(dict.fromkeys((*spec.input_columns, "timestamp")))
                prediction = prediction_callable(
                    result.coefficients,
                    spec.input_columns,
                    result.prediction_model,
                )(frame.loc[:, list(columns)])
                assert len(prediction) == len(frame)
                assert pd.Series(prediction).notna().all()
            finally:
                owned.lease.close()


@pytest.mark.parametrize(
    ("model_algorithm", "expected_package", "wrong_package"),
    (
        ("linear_regression", "internal", "sklearn"),
        ("ridge_regression", "sklearn", "internal"),
    ),
)
def test_prediction_builder_binds_backend_package_to_actual_model_owner(
    model_algorithm: str,
    expected_package: str,
    wrong_package: str,
) -> None:
    frame = _research_rows()
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("mom_20d",),
        label_column="forward_return_20d",
        max_train_rows=len(frame),
    )
    estimator = build_estimator_from_model_spec(model_spec)
    with runtime_resource_policy_scope():
        model_result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            model_spec,
            estimator=estimator,
            fitting_cutoff="2014-01-02",
        )
        try:
            model_result = model_result_owned.value
            prediction_spec = PredictionSpec(
                prediction_id=f"{model_algorithm}_prediction",
                model_id=model_spec.model_id,
                research_dataset_id="demo",
                train_end="2014-01-02",
                test_start="2014-01-03",
                test_end="2014-01-03",
            )
            result = prediction_runner_module.build_prediction_result_from_frames(
                (frame,),
                prediction_spec,
                model_result.coefficients,
                model_spec.input_columns,
                prediction_model=model_result.prediction_model,
                model_metadata=model_result.metadata,
            )

            assert (
                result.model_metadata["dependency_status"]["package"]
                == expected_package
            )
            wrong_metadata = {
                **model_result.metadata,
                "dependency_status": {
                    **model_result.metadata["dependency_status"],
                    "package": wrong_package,
                },
            }
            with pytest.raises(ValueError, match="invalid_model_metadata"):
                prediction_runner_module.build_prediction_result_from_frames(
                    (frame,),
                    prediction_spec,
                    model_result.coefficients,
                    model_spec.input_columns,
                    prediction_model=model_result.prediction_model,
                    model_metadata=wrong_metadata,
                )
        finally:
            model_result_owned.lease.close()


def test_prediction_handoff_reads_research_dataset_and_writes_report(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    model_catalog = ModelCatalog(
        (
            ModelSpec(
                model_id="linear_demo",
                model_algorithm="linear_regression",
                input_factor_columns=("mom_20d",),
                label_column="forward_return_20d",
            ),
        )
    )
    prediction_catalog = PredictionCatalog(
        (
            PredictionSpec(
                prediction_id="pred_demo",
                model_id="linear_demo",
                research_dataset_id="demo",
                train_end="2014-01-02",
                test_start="2014-01-03",
                test_end="2014-01-03",
            ),
        )
    )

    with build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    ) as result:
        output_bundle = _write_prediction_result(layout, result)

    assert output_bundle.as_posix().endswith("outputs/pred_demo")
    assert (output_bundle / "prediction.parquet").is_file()
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["is_oos"] is True
    assert manifest["metadata"]["prediction_mode"] == "minimal_oos"
    assert manifest["metadata"]["prediction_column"] == PREDICTION_SCORE_COLUMN
    assert manifest["metadata"]["primary_key"] == list(PREDICTION_PRIMARY_KEY)
    assert manifest["metadata"]["model_metadata"]["input_count"] == 1
    assert manifest["metadata"]["model_metadata"]["coefficient_count"] == 2
    assert manifest["metadata"]["model_metadata"]["train_start"] == "2014-01-02"
    assert manifest["metadata"]["model_metadata"]["train_end"] == "2014-01-02"
    assert manifest["metadata"]["prediction_metadata"]["prediction_rows"] == 2
    assert (
        manifest["metadata"]["prediction_metadata"]["requested_test_end"]
        == "2014-01-03"
    )
    assert (
        manifest["metadata"]["prediction_metadata"]["effective_test_end"]
        == "2014-01-03"
    )


def test_prediction_handoff_writes_prediction_report_without_materialized_bundle(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog)
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-03"),
    )
    model_catalog = ModelCatalog(
        (
            ModelSpec(
                model_id="linear_demo",
                model_algorithm="linear_regression",
                input_factor_columns=("mom_20d",),
                label_column="forward_return_20d",
            ),
        )
    )
    prediction_catalog = PredictionCatalog(
        (
            PredictionSpec(
                prediction_id="pred_demo",
                model_id="linear_demo",
                research_dataset_id="demo",
                train_end="2014-01-02",
                test_start="2014-01-03",
                test_end="2014-01-03",
            ),
        )
    )

    with build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="pred_demo",
    ) as result:
        output_bundle = _write_prediction_result(layout, result)

    assert (output_bundle / "prediction.parquet").is_file()


def test_prediction_handoff_streams_research_chunks_for_model_and_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _research_rows_for_dates(
        [
            "2014-01-02",
            "2014-01-03",
            "2014-01-06",
            "2014-01-07",
            "2014-01-08",
        ]
    )
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        filter_column_roles={"eligible": ColumnRole.UNIVERSE},
    )
    processed_catalog = make_demo_processed_catalog()
    write_demo_processed_inputs(layout, processed_catalog, frame=frame)
    assembly_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    assemble_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        processed_catalog=processed_catalog,
        dataset_id="demo",
        time_window=TimeWindow("2014-01-02", "2014-01-08"),
        resource_policy=assembly_policy,
    )
    model_catalog = ModelCatalog(
        (
            ModelSpec(
                model_id="linear_demo",
                model_algorithm="linear_regression",
                input_factor_columns=("mom_20d",),
                label_column="forward_return_20d",
            ),
        )
    )
    prediction_catalog = PredictionCatalog(
        (
            PredictionSpec(
                prediction_id="pred_demo",
                model_id="linear_demo",
                research_dataset_id="demo",
                train_end="2014-01-03",
                test_start="2014-01-06",
                test_end="2014-01-08",
            ),
        )
    )

    original_iter = prediction_runner_module.iter_research_dataset_chunks
    original_fit = prediction_runner_module.fitting._fit_model_result_owned_from_frames
    original_write = prediction_runner_module.build_prediction_result_from_frames
    read_chunk_rows: list[int] = []
    fit_chunk_rows: list[int] = []
    prediction_chunk_rows: list[int] = []
    read_columns_seen: list[tuple[str, ...]] = []

    def wrapped_iter(*args, **kwargs):
        read_columns_seen.append(tuple(kwargs["columns"]))
        for chunk in original_iter(*args, **kwargs):
            read_chunk_rows.append(len(chunk))
            yield chunk

    def wrapped_fit(frames, spec, *, estimator, fitting_cutoff):
        chunks = list(frames)
        fit_chunk_rows.extend(len(chunk) for chunk in chunks)
        return original_fit(
            iter(chunks),
            spec,
            estimator=estimator,
            fitting_cutoff=fitting_cutoff,
        )

    def wrapped_write(frames, spec, coefficients, input_columns, **kwargs):
        def counted_frames():
            for chunk in frames:
                prediction_chunk_rows.append(len(chunk))
                yield chunk

        return original_write(
            counted_frames(),
            spec,
            coefficients,
            input_columns,
            **kwargs,
        )

    monkeypatch.setattr(
        prediction_runner_module,
        "iter_research_dataset_chunks",
        wrapped_iter,
    )
    monkeypatch.setattr(
        prediction_runner_module.fitting,
        "_fit_model_result_owned_from_frames",
        wrapped_fit,
    )
    monkeypatch.setattr(
        prediction_runner_module,
        "build_prediction_result_from_frames",
        wrapped_write,
    )

    prediction_policy = capture_storage_test_policy(
        working_headroom_bytes=4 * 1024**2,
    )
    with runtime_resource_policy_scope(prediction_policy):
        with build_prediction_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="pred_demo",
            resource_policy=prediction_policy,
        ) as result:
            output_bundle = _write_prediction_result(layout, result)

    assert (output_bundle / "prediction.parquet").is_file()
    assert read_chunk_rows == [len(frame), len(frame)]
    assert fit_chunk_rows == [len(frame)]
    assert prediction_chunk_rows == [len(frame)]
    assert any("forward_return_20d" in columns for columns in read_columns_seen)
    assert any("forward_return_20d" not in columns for columns in read_columns_seen)

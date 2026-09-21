from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
import quant_research_platform.research.predictions.runner as prediction_runner_module
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    RAW_PREDICTION_COLUMN,
    PredictionSpec,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.predictions.outputs import prediction_callable
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from tests.helpers.research_configuration import configured_models
from tests.research.prediction_output_helpers import (
    write_prediction_output_and_read,
)
from tests.research.workflow_test_support import (
    _research_rows,
)


def test_evaluation_prediction_and_report_boundary(tmp_path: Path) -> None:
    frame = _research_rows().assign(
        instant_target=lambda value: value["momentum_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("momentum_20d",),
        label_column="instant_target",
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
                np.asarray(model_result.prediction_model.coefficients),
                model_spec.input_columns,
                prediction_model=model_result.prediction_model,
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
    specs = configured_models("pv_all")
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
                date + pd.offsets.BDay(20) for date in dates for _ in range(3)
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
                fitting_cutoff="2020-02-12",
            )
            try:
                result = owned.value
                columns = tuple(dict.fromkeys((*spec.input_columns, "timestamp")))
                prediction = prediction_callable(
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
    frame = _research_rows().assign(
        instant_target=lambda value: value["momentum_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("momentum_20d",),
        label_column="instant_target",
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
                    model_spec.input_columns,
                    prediction_model=model_result.prediction_model,
                    model_metadata=wrong_metadata,
                )
        finally:
            model_result_owned.lease.close()

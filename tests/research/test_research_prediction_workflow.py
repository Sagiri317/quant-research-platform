from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
from quant_research_platform.contracts.models import ModelSpec, _model_spec
from quant_research_platform.research.models.estimators import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.fitting import prediction_callable
from tests.helpers.research_configuration import research_configuration
from tests.helpers.research_processed_inputs import _demo_research_rows
from tests.research.direct_model_test_support import fit_frame, predict_frame


def test_native_prediction_preserves_raw_units_rank_and_family(tmp_path: Path) -> None:
    frame = _demo_research_rows().assign(
        instant_target=lambda value: value["reversal_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    model_spec = ModelSpec(
        model_id="linear_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("reversal_20d",),
        target_column="instant_target",
    )
    case = fit_frame(
        tmp_path / "fit",
        frame.loc[frame.timestamp.le(pd.Timestamp("2014-01-02"))],
        model_spec,
        market_id="a_share",
    )
    prediction, output, metadata = predict_frame(
        tmp_path / "future",
        frame,
        case,
        window={"start": "2014-01-03", "end": "2014-01-03"},
    )
    assert tuple(prediction.columns) == (
        "timestamp",
        "asset_id",
        "prediction_score",
        "prediction_rank_score",
    )
    assert prediction.timestamp.eq(pd.Timestamp("2014-01-03")).all()
    np.testing.assert_allclose(prediction.prediction_score, [0.02, 0.01])
    assert prediction.prediction_rank_score.tolist() == [0.25, -0.25]
    facts = metadata["model_facts"]
    assert facts["model"]["model_id"] == facts["family_id"] == model_spec.model_id
    assert output == tmp_path / "future" / "prediction"


def test_a_share_model_ladder_fits_and_predicts_each_retained_algorithm() -> None:
    specs = tuple(_model_spec(item) for item in research_configuration()["models"])
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
            "model_target_available_at": [
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
    for spec in specs:
        result = model_fitting._fit_model_result(
            model_fitting._collect_training_frame(
                (frame,),
                spec,
                fitting_cutoff="2020-02-12",
            ),
            spec,
            estimator=build_estimator_from_model_spec(spec),
        )
        columns = tuple(dict.fromkeys((*spec.input_columns, "timestamp", "asset_id")))
        prediction = prediction_callable(spec.input_columns, result.prediction_model)(
            frame.loc[:, list(columns)]
        )
        assert len(prediction) == len(frame)
        assert pd.Series(prediction).notna().all()


@pytest.mark.parametrize("start", ["2014-01-01", "2014-01-02"])
def test_native_prediction_must_follow_saved_fit_cutoff(tmp_path, start):
    frame = _demo_research_rows().assign(
        instant_target=lambda value: value["reversal_20d"] * 0.01,
        label_available_at=lambda value: value["timestamp"],
    )
    case = fit_frame(
        tmp_path / "fit",
        frame.loc[frame.timestamp.eq(pd.Timestamp("2014-01-02"))],
        ModelSpec(
            model_id="cutoff",
            model_algorithm="linear_regression",
            input_factor_columns=("reversal_20d",),
            target_column="instant_target",
        ),
        market_id="a_share",
    )
    with pytest.raises(ValueError, match="prediction_must_follow_fit_cutoff"):
        predict_frame(
            tmp_path / "future",
            frame,
            case,
            window={"start": start, "end": "2014-01-03"},
        )
    assert not (tmp_path / "future" / "prediction").exists()

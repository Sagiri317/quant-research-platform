from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.research.models import (
    fit_model_result,
    predict_model_frame,
)
from quant_research_platform.research.models.supervised import (
    model_algorithm_ids,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from tests.helpers.research_specs import (
    make_signal_spec,
)
from tests.research.filter_mode_fixtures import (
    make_filter_mode_frame,
    make_filter_mode_model_hyperparameters,
    make_filter_mode_prediction_output_scores,
)
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
)


def test_filter_columns_cannot_be_features() -> None:
    with pytest.raises(ValueError, match="filter_column_in_input_columns: eligible"):
        EvaluationSpec(
            run_id="bad_eval_features",
            market_id=MarketId("a_share"),
            research_dataset_id="demo_filter",
            signals=(make_signal_spec("eligible"),),
            label_columns=("label",),
        )
    with pytest.raises(ValueError, match="filter_column_in_input_columns: can_exit"):
        ModelSpec(
            model_id="bad_model_features",
            model_algorithm="linear_regression",
            input_factor_columns=("can_exit",),
            label_column="label",
        )


def test_model_sample_filter_modes_and_metadata_ignore_tradability_masks() -> None:
    frame = make_filter_mode_frame()
    eligible_spec = ModelSpec(
        model_id="linear_eligible",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(frame),
    )
    unfiltered_spec = ModelSpec(
        model_id="linear_unfiltered",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        max_train_rows=len(frame),
    )

    eligible = fit_model_result(frame, eligible_spec)
    unfiltered = fit_model_result(frame, unfiltered_spec)

    assert (
        eligible.metadata
        | {
            key: eligible.metadata[key]
            for key in (
                "dependency_status",
                "feature_importance",
                "hyperparameters",
                "max_train_rows",
                "purge_gap_steps",
                "random_seed",
            )
        }
        == eligible.metadata
    )
    assert {
        key: eligible.metadata[key]
        for key in (
            "model_algorithm",
            "sample_filter_mode",
            "filter_columns_used",
            "filter_definition",
            "rows_input",
            "rows_after_filter_mode",
            "rows_after_input_label_dropna",
            "train_start",
            "train_end",
            "input_factor_columns",
            "input_feature_columns",
            "input_columns",
            "input_count",
            "label_column",
            "coefficient_count",
        )
    } == {
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 6,
        "rows_after_filter_mode": 5,
        "rows_after_input_label_dropna": 4,
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
        "input_factor_columns": ["factor"],
        "input_feature_columns": [],
        "input_columns": ["factor"],
        "input_count": 1,
        "label_column": "label",
        "coefficient_count": 2,
    }
    assert eligible.metadata["dependency_status"]["status"] == "available"
    assert eligible.metadata["hyperparameters"] == {}
    assert eligible.metadata["random_seed"] is None
    assert eligible.metadata["max_train_rows"] == len(frame)
    assert eligible.metadata["purge_gap_steps"] == 0
    assert eligible.metadata["feature_importance"] == pytest.approx({"factor": 0.01})
    assert unfiltered.metadata["rows_after_filter_mode"] == 6
    assert unfiltered.metadata["rows_after_input_label_dropna"] == 5
    assert eligible.coefficients.tolist() == pytest.approx([0.0, 0.01])
    assert unfiltered.coefficients.tolist() == pytest.approx([0.0, 0.01])

    all_tradability_blocked = frame.assign(can_enter=False, can_exit=False)
    still_trains = fit_model_result(all_tradability_blocked, eligible_spec)
    assert still_trains.metadata["rows_after_input_label_dropna"] == 4


@pytest.mark.parametrize(
    ("metadata_name", "field", "value"),
    [
        ("filter_metadata", "rows_input", True),
        ("filter_metadata", "rows_input", 1.5),
        ("filter_metadata", "rows_input", -1),
        ("filter_metadata", "rows_after_filter_mode", True),
        ("filter_metadata", "rows_after_filter_mode", 1.5),
        ("row_counts", "rows_after_input_label_dropna", True),
        ("row_counts", "rows_after_input_label_dropna", 1.5),
        ("row_counts", "rows_after_input_label_dropna", -1),
    ],
)
def test_model_metadata_rejects_invalid_count_values(
    metadata_name: str,
    field: str,
    value: object,
) -> None:
    from quant_research_platform.research.models import fitting as model_fitting

    spec = ModelSpec(
        model_id="linear_bad_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
    )
    filter_metadata = {
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
    }
    row_counts = {
        "rows_after_input_label_dropna": 2,
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
    }
    target = filter_metadata if metadata_name == "filter_metadata" else row_counts
    target[field] = value

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        model_fitting.model_metadata(
            spec,
            filter_metadata=filter_metadata,
            row_counts=row_counts,
            coefficient_count=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_input", True),
        ("rows_input", 1.5),
        ("rows_input", -1),
        ("rows_after_filter_mode", True),
        ("rows_after_filter_mode", 1.5),
        ("rows_after_filter_mode", -1),
    ],
)
def test_model_streaming_fit_rejects_invalid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    from quant_research_platform.research.models import fitting as model_fitting

    metadata = _model_filter_metadata()
    metadata[field] = value

    def fake_apply_filter_mode(frame: pd.DataFrame, **_: object):
        return frame, metadata

    monkeypatch.setattr(model_fitting, "apply_filter_mode", fake_apply_filter_mode)
    spec = ModelSpec(
        model_id="linear_bad_streaming_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
    )

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        fit_model_result(make_filter_mode_frame(), spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_input", True),
        ("rows_input", 1.5),
        ("rows_input", -1),
        ("rows_after_filter_mode", True),
        ("rows_after_filter_mode", 1.5),
        ("rows_after_filter_mode", -1),
    ],
)
def test_model_training_collection_rejects_invalid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    from quant_research_platform.research.models import fitting as model_fitting

    metadata = _model_filter_metadata()
    metadata[field] = value

    def fake_apply_filter_mode(frame: pd.DataFrame, **_: object):
        return frame, metadata

    monkeypatch.setattr(model_fitting, "apply_filter_mode", fake_apply_filter_mode)
    spec = ModelSpec(
        model_id="linear_bad_collected_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(make_filter_mode_frame()),
    )

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        model_fitting.collect_training_frame((make_filter_mode_frame(),), spec)


def test_model_fit_accepts_valid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import fitting as model_fitting

    def fake_apply_filter_mode(frame: pd.DataFrame, **_: object):
        return frame, _model_filter_metadata()

    monkeypatch.setattr(model_fitting, "apply_filter_mode", fake_apply_filter_mode)
    spec = ModelSpec(
        model_id="linear_valid_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
    )

    result = fit_model_result(make_filter_mode_frame(), spec)

    assert result.metadata["rows_input"] == 2
    assert result.metadata["rows_after_filter_mode"] == 2


@pytest.mark.parametrize(
    "model_algorithm",
    model_algorithm_ids(),
)
def test_required_model_algorithm_ids_fit_and_predict(model_algorithm: str) -> None:
    train = pd.DataFrame(
        {
            "market_id": ["a_share"] * 8,
            "asset_id": [f"asset_{idx}" for idx in range(8)],
            "timestamp": [pd.Timestamp("2024-01-02")] * 8,
            "factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "feature": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "label": [0.11, 0.19, 0.31, 0.41, 0.49, 0.61, 0.69, 0.81],
            "eligible": [True] * 8,
        }
    )
    test = train.assign(
        timestamp=pd.Timestamp("2024-01-03"),
        label=np.nan,
    )
    frame = pd.concat([train, test], ignore_index=True)
    spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        random_seed=7,
        hyperparameters=make_filter_mode_model_hyperparameters(model_algorithm),
        max_train_rows=len(train),
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_pred",
        model_id=spec.model_id,
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )

    result = fit_model_result(frame.loc[frame["timestamp"].lt("2024-01-03")], spec)
    prediction = prediction_frame_from_artifact(
        frame,
        prediction_spec,
        result.coefficients,
        spec.input_columns,
        prediction_model=result.prediction_model,
    )

    assert result.metadata["model_algorithm"] == model_algorithm
    assert result.metadata["input_columns"] == ["factor", "feature"]
    assert result.metadata["rows_after_input_label_dropna"] == 8
    assert len(prediction) == 8
    assert prediction[PREDICTION_SCORE_COLUMN].notna().all()


@pytest.mark.parametrize("model_algorithm", ["linear_regression", "ridge_regression"])
def test_model_training_controls_apply_to_linear_and_estimators(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "factor": [1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0],
            "eligible": [True, True, True],
        }
    )
    max_rows_spec = ModelSpec(
        model_id=f"{model_algorithm}_max_rows",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=1,
    )
    purge_spec = ModelSpec(
        model_id=f"{model_algorithm}_purge",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        purge_gap_steps=1,
        max_train_rows=len(frame),
    )

    max_rows = fit_model_result(frame, max_rows_spec)
    purge = fit_model_result(frame, purge_spec)

    assert max_rows.metadata["rows_after_input_label_dropna"] == 1
    assert max_rows.metadata["train_start"] == "2024-01-03"
    assert max_rows.metadata["train_end"] == "2024-01-03"
    assert purge.metadata["rows_after_input_label_dropna"] == 2
    assert purge.metadata["train_start"] == "2024-01-01"
    assert purge.metadata["train_end"] == "2024-01-02"


def test_model_max_train_rows_trims_streaming_training_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import fitting as model_fitting
    from quant_research_platform.research.models import validation_split

    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)],
                    "asset_id": [f"asset_{day:02d}"],
                    "factor": [float(day)],
                    "label": [float(day) * 2.0],
                    "eligible": [True],
                }
            )
            for day in range(40)
        ],
        ignore_index=True,
    )
    spec = ModelSpec(
        model_id="linear_streaming_max_rows",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=7,
        purge_gap_steps=2,
    )
    expected = validation_split.apply_training_controls(
        frame.loc[:, ["timestamp", "factor", "label"]],
        spec,
    )
    original_trim = validation_split.trim_training_buffer
    observed_buffer_inputs: list[int] = []

    def wrapped_trim(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
        observed_buffer_inputs.append(len(frame))
        return original_trim(frame, spec)

    monkeypatch.setattr(validation_split, "trim_training_buffer", wrapped_trim)

    training, row_counts, _metadata = model_fitting.collect_training_frame(
        (chunk for _, chunk in frame.groupby(frame.index // 5)),
        spec,
    )

    pd.testing.assert_frame_equal(training, expected)
    assert row_counts["rows_after_input_label_dropna"] == len(expected)
    assert max(observed_buffer_inputs) < len(frame)


def test_model_training_collection_rejects_unbounded_collection() -> None:
    from quant_research_platform.research.models import fitting as model_fitting

    frame = pd.concat(
        [
            make_filter_mode_frame().assign(timestamp=pd.Timestamp(value))
            for value in ("2024-01-02", "2024-01-03", "2024-01-04")
        ],
        ignore_index=True,
    )
    spec = ModelSpec(
        model_id="linear_uncapped_streaming",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=None,
    )

    with pytest.raises(ValueError, match="unbounded_model_training_collection"):
        model_fitting.collect_training_frame(
            (frame,),
            spec,
        )

    with pytest.raises(ValueError, match="unbounded_model_training_collection"):
        model_fitting.collect_training_frame(
            (chunk for _, chunk in frame.groupby("timestamp")),
            spec,
        )


@pytest.mark.parametrize(
    "model_algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"],
)
def test_rank_ensembles_predict_cross_sectional_ranks_per_timestamp(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "asset_id": ["asset_a", "asset_b", "asset_a", "asset_b"],
            "factor": [1.0, 2.0, 1.0, 2.0],
            "feature": [10.0, 20.0, 10.0, 20.0],
            "label": [0.0, 1.0, 0.0, 1.0],
            "eligible": [True, True, True, True],
        }
    )
    spec = ModelSpec(
        model_id=f"{model_algorithm}_rank_scope",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )

    result = fit_model_result(frame, spec)
    predicted = result.prediction_model.predict(frame)
    predicted_frame = predict_model_frame(frame, spec, result)

    assert predicted.tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert predicted_frame["prediction_score"].tolist() == pytest.approx(
        [0.5, 1.0, 0.5, 1.0]
    )


@pytest.mark.parametrize(
    "model_algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"],
)
def test_rank_ensemble_prediction_outputs_are_chunk_boundary_invariant(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "asset_id": ["asset_a", "asset_b", "asset_a", "asset_b"],
            "factor": [1.0, 2.0, 1.0, 2.0],
            "feature": [10.0, 20.0, 10.0, 20.0],
            "label": [0.0, 1.0, 0.0, 1.0],
            "eligible": [True, True, True, True],
        }
    )
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_chunk_boundary",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_chunk_boundary_prediction",
        model_id=model_spec.model_id,
        research_dataset_id="rank_boundary_dataset",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )
    result = fit_model_result(frame, model_spec)

    one_chunk = make_filter_mode_prediction_output_scores(
        [frame],
        prediction_spec=prediction_spec,
        model_spec=model_spec,
        result=result,
    )
    split_chunks = make_filter_mode_prediction_output_scores(
        [chunk for _, chunk in frame.groupby("timestamp", sort=True)],
        prediction_spec=prediction_spec,
        model_spec=model_spec,
        result=result,
    )

    assert one_chunk["prediction_score"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert split_chunks["prediction_score"].tolist() == pytest.approx(
        one_chunk["prediction_score"].tolist()
    )


def test_model_rank_deficient_fit_matches_direct_lstsq_design() -> None:
    x1 = [
        0.1257302210933933,
        -0.1321048632913019,
        0.6404226504432821,
        0.10490011715303971,
        -0.535669373161111,
        0.36159505490948474,
        1.3040000451301372,
        0.9470809631292422,
    ]
    x2 = [
        0.12573021405604093,
        -0.1321048759455166,
        0.6404226442105374,
        0.1049001175662995,
        -0.5356693964114188,
        0.3615950527215681,
        1.3040000326710277,
        0.9470809558065686,
    ]
    y = [
        -0.5442589828573099,
        -0.31630015636915454,
        0.4116305363741328,
        1.0425133694426776,
        -0.12853466294403426,
        1.3664634705496859,
        -0.6651946734866135,
        0.3515100700930197,
    ]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-05",
                ]
            ),
            "asset_id": [f"asset_{idx}" for idx in range(len(x1))],
            "factor": x1,
            "factor_shadow": x2,
            "label": y,
            "eligible": [True] * len(x1),
        }
    )
    spec = ModelSpec(
        model_id="linear_rank_deficient",
        model_algorithm="linear_regression",
        input_factor_columns=("factor", "factor_shadow"),
        label_column="label",
        max_train_rows=len(frame),
    )
    usable = frame.loc[frame["eligible"]].dropna(
        subset=[*spec.input_columns, spec.label_column]
    )
    x = usable.loc[:, spec.input_columns].to_numpy(dtype="float64")
    y = usable.loc[:, spec.label_column].to_numpy(dtype="float64")
    design = np.empty((len(x), len(spec.input_columns) + 1), dtype="float64")
    design[:, 0] = 1.0
    design[:, 1:] = x
    expected, *_ = np.linalg.lstsq(design, y, rcond=None)

    result = fit_model_result(frame, spec)

    assert result.coefficients.tolist() == pytest.approx(expected.tolist())

    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=4_000,
        environ={},
    )
    with runtime_resource_policy_scope(policy):
        tiny_result = fit_model_result(frame, spec)

    assert tiny_result.coefficients.tolist() == pytest.approx(
        expected.tolist(),
        rel=1e-7,
    )


def test_model_rejects_tradable_sample_filter_mode() -> None:
    with pytest.raises(ValueError, match="unsupported_model_sample_filter_mode"):
        ModelSpec(
            model_id="linear_tradable",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
            sample_filter_mode="tradable",
        )


def test_model_rejects_unsupported_contract_version() -> None:
    with pytest.raises(ValueError, match="unsupported_model_contract_version"):
        ModelSpec(
            model_id="linear_bad_contract",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
            contract_version="9.9",
        )


def _model_filter_metadata() -> dict[str, object]:
    return {
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
    }

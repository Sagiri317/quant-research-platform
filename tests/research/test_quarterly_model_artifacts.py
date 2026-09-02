from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.bundle_validation import (
    read_prediction_output_bundle,
)
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.models import (
    ModelSpec,
    validate_model_selection_metadata,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.targets import (
    ProducerBinding,
    ProducerKind,
    TargetSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.fitting import (
    ModelFitResult,
    training_scope_chunks,
)
from quant_research_platform.research.models.persisted import (
    _validated_quarterly_payload,
    read_quarterly_model_artifact,
    write_quarterly_model_artifact,
)
from quant_research_platform.research.models.production import (
    QuarterlyModelResult,
    _a_share_trial_specs,
    _fit_selected_model_result,
)
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    select_best_validation_trial,
    select_prediction_model_spec,
)
from quant_research_platform.research.predictions import runner
from quant_research_platform.research.protocols import a_share
from quant_research_platform.research.protocols.a_share import resolve_trial_parameters
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research as research_workflow
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.prediction_output_helpers import write_prediction_result


class _ColumnPredictor:
    def __init__(self, column: str, offset: float = 0.0) -> None:
        self.column = column
        self.offset = offset

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.column].to_numpy(dtype="float64") + self.offset


class _UnpicklablePredictor(_ColumnPredictor):
    def __getstate__(self) -> object:
        raise RuntimeError("predictor_serialization_failed")


def _metadata(
    columns: tuple[str, ...],
    *,
    algorithm: str = "linear_regression",
    model_id: str = "champion",
    rank_parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "model_algorithm": algorithm,
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
        "rows_after_input_label_dropna": 2,
        "train_start": "2014-01-01",
        "train_end": "2018-12-31",
        "input_factor_columns": list(columns),
        "input_feature_columns": [],
        "input_columns": list(columns),
        "input_count": len(columns),
        "label_column": "label",
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": 11,
        "max_train_rows": None,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
        "coefficient_count": len(columns) + 1,
        "sample_weight_protocol": "equal_date_total_equal_sum_n_train",
        "scaler": None,
        "feature_importance": {column: 1.0 for column in columns},
        "fit_cutoff": "2018-12-31",
        "model_selection": _selection_metadata(
            family_id=algorithm,
            model_id=model_id,
            columns=columns,
        ),
    }
    if rank_parameters is not None:
        result["rank_parameters"] = rank_parameters
    return result


def _selection_metadata(
    *,
    family_id: str,
    model_id: str,
    columns: tuple[str, ...],
) -> dict[str, object]:
    trial = {
        "model_id": model_id,
        "validation_rows": 2,
        "validation_ic": 0.1,
        "validation_rankic": 0.2,
        "validation_ic_reason": None,
        "validation_rankic_reason": None,
        "validation_mse": 1.0,
        "resolved_hyperparameters": {},
        "random_seed": 11,
        "n_train": 2,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "test",
        },
    }
    return {
        "status": "selected",
        "selection_metric": "validation_rankic",
        "selection_direction": "maximize",
        "validation_start": "2018-01-01",
        "validation_end": "2018-12-31",
        "candidate_family_ids": [family_id],
        "tie_break_order": [family_id],
        "selected_family_id": family_id,
        "selected_model_id": model_id,
        "families": [
            {
                "family_id": family_id,
                "config_tie_break_order": [model_id],
                "selected_config_id": model_id,
                "trials": [trial],
                "validation_rows": 2,
                "selection_metric_value": 0.2,
                "selection_metric_reason": None,
            }
        ],
    }


def _fold(quarter: int) -> ExperimentFold:
    start = f"2019-{1 + (quarter - 1) * 3:02d}-01"
    end = (pd.Timestamp(start) + pd.offsets.QuarterEnd()).date().isoformat()
    return ExperimentFold(
        fold_id=f"2019-q{quarter}",
        train=TimeWindow("2014-01-01", "2017-12-31"),
        validation=TimeWindow("2018-01-01", "2018-12-31"),
        development_test=TimeWindow(start, end),
    )


def _envelope() -> object:
    return build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-06-30",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={},
    )


def _quarterly_result(
    quarter: int,
    *,
    model_id: str,
    columns: tuple[str, ...],
    predictor: object | None = None,
) -> QuarterlyModelResult:
    model = predictor or _ColumnPredictor(columns[0])
    algorithm = (
        "ic_weighted_rank_ensemble"
        if isinstance(model, ICWeightedRankEnsemble)
        else "equal_weight_rank_ensemble"
        if isinstance(model, RankEnsemblePredictor)
        else "linear_regression"
    )
    return QuarterlyModelResult(
        quarter=_fold(quarter),
        model_id=model_id,
        model_result=ModelFitResult(
            coefficients=np.zeros(len(columns) + 1),
            metadata=_metadata(columns, algorithm=algorithm, model_id=model_id),
            prediction_model=model,
        ),
    )


def _dataset_spec() -> ResearchDatasetSpec:
    return make_demo_research_dataset_spec(
        data_id="research_demo",
        input_data_ids=("processed_demo",),
        base_input_data_id="processed_demo",
        factor_columns=("x", "y"),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=(),
    )


def test_persisted_quarter_consumption_uses_active_columns_and_actual_model_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(
                1,
                model_id="champion_q1",
                columns=("x",),
                predictor=_ColumnPredictor("x", 1.0),
            ),
            _quarterly_result(
                2,
                model_id="champion_q2",
                columns=("y",),
                predictor=_ColumnPredictor("y", 2.0),
            ),
        ),
    )
    artifact = read_quarterly_model_artifact(root)
    spec = _dataset_spec()
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["a", "a"],
            "timestamp": pd.to_datetime(["2019-02-01", "2019-05-01"]),
            "x": [3.0, np.nan],
            "y": [np.nan, 4.0],
            "eligible": [True, True],
        }
    )
    monkeypatch.setattr(
        runner,
        "iter_research_dataset_chunks",
        lambda *_args, **_kwargs: iter((frame,)),
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction",
        model_id="model-artifact",
        research_dataset_id="research_demo",
        train_end=None,
        test_start="2019-01-01",
        test_end="2019-06-30",
        prediction_mode="a_share_oos",
        prediction_column="raw_prediction",
    )
    result = runner.build_prediction_result_from_persisted_model(
        layout=StorageLayout(project_root=tmp_path),
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        prediction_spec=prediction_spec,
        persisted_model=artifact,
        resource_policy=None,
    )
    try:
        frames = []
        for item in result.output_chunks:
            frames.append(item.value.copy() if isinstance(item, OwnedValue) else item)
        output = pd.concat(frames, ignore_index=True)
    finally:
        result.close()
    assert len(output) == 2
    assert output["model_id"].tolist() == ["champion_q1", "champion_q2"]
    assert output["raw_prediction"].tolist() == pytest.approx([4.0, 6.0])
    assert output["alpha_score"].tolist() == pytest.approx([0.0, 0.0])


def test_persisted_prediction_alpha_is_permutation_invariant_and_reader_uses_alpha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        envelope=_envelope(),
        quarterly_results=(_quarterly_result(1, model_id="champion", columns=("x",)),),
    )
    artifact = read_quarterly_model_artifact(root)
    spec = _dataset_spec()
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 3,
            "asset_id": ["a", "b", "c"],
            "timestamp": pd.to_datetime(["2019-02-01"] * 3),
            "x": [1.0, 2.0, 3.0],
            "y": [np.nan] * 3,
            "eligible": [True] * 3,
        }
    )
    monkeypatch.setattr(
        runner,
        "iter_research_dataset_chunks",
        lambda *_args, **_kwargs: iter((frame,)),
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction",
        model_id="model-artifact",
        research_dataset_id="research_demo",
        train_end=None,
        test_start="2019-01-01",
        test_end="2019-03-31",
        prediction_mode="a_share_oos",
        prediction_column="raw_prediction",
    )
    result = runner.build_prediction_result_from_persisted_model(
        layout=StorageLayout(project_root=tmp_path),
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        prediction_spec=prediction_spec,
        persisted_model=artifact,
        resource_policy=None,
    )
    bundle = write_prediction_result(StorageLayout(project_root=tmp_path), result)
    table = pd.read_parquet(bundle / "prediction.parquet").sort_values(
        "asset_id", kind="mergesort"
    )
    assert table["alpha_score"].tolist() == pytest.approx([-1.0 / 3.0, 0.0, 1.0 / 3.0])
    assert float(table["alpha_score"].mean()) == pytest.approx(0.0)

    with runtime_resource_policy_scope(None):
        with read_prediction_output_bundle(
            bundle,
            expected_output_id="prediction",
            expected_run_id="test_run",
        ) as (_manifest, lookup):
            assert lookup.prediction_column == "alpha_score"

    shuffled = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    monkeypatch.setattr(
        runner,
        "iter_research_dataset_chunks",
        lambda *_args, **_kwargs: iter((shuffled,)),
    )
    second = runner.build_prediction_result_from_persisted_model(
        layout=StorageLayout(project_root=tmp_path),
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        prediction_spec=prediction_spec,
        persisted_model=artifact,
        resource_policy=None,
    )
    try:
        second_table = pd.concat(
            [
                item.value if isinstance(item, OwnedValue) else item
                for item in second.output_chunks
            ],
            ignore_index=True,
        ).sort_values("asset_id", kind="mergesort")
    finally:
        second.close()
    pd.testing.assert_frame_equal(
        table.reset_index(drop=True),
        second_table.reset_index(drop=True),
        check_dtype=True,
    )


def test_persisted_quarter_request_is_clipped_to_requested_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        envelope=_envelope(),
        quarterly_results=(_quarterly_result(1, model_id="champion", columns=("x",)),),
    )
    artifact = read_quarterly_model_artifact(root)
    spec = _dataset_spec()
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "a", "a"],
            "timestamp": pd.to_datetime(["2019-01-02", "2019-02-15", "2019-03-30"]),
            "x": [1.0, 2.0, 3.0],
            "y": np.nan,
            "eligible": True,
        }
    )
    monkeypatch.setattr(
        runner,
        "iter_research_dataset_chunks",
        lambda *_args, **_kwargs: iter((frame,)),
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction",
        model_id="model-artifact",
        research_dataset_id="research_demo",
        train_end=None,
        test_start="2019-02-01",
        test_end="2019-03-15",
        prediction_mode="a_share_oos",
        prediction_column="raw_prediction",
    )
    result = runner.build_prediction_result_from_persisted_model(
        layout=StorageLayout(project_root=tmp_path),
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        prediction_spec=prediction_spec,
        persisted_model=artifact,
        resource_policy=None,
    )
    try:
        frames = []
        for item in result.output_chunks:
            frames.append(item.value.copy() if isinstance(item, OwnedValue) else item)
        output = pd.concat(frames, ignore_index=True)
    finally:
        result.close()
    assert output["timestamp"].min() >= pd.Timestamp("2019-02-01")
    assert output["timestamp"].max() <= pd.Timestamp("2019-03-15")
    assert len(output) == 1


def test_persisted_quarter_route_rejects_uncovered_first_quarter(
    tmp_path: Path,
) -> None:
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        envelope=_envelope(),
        quarterly_results=(_quarterly_result(2, model_id="champion", columns=("x",)),),
    )
    artifact = read_quarterly_model_artifact(root)
    prediction_spec = PredictionSpec(
        prediction_id="prediction",
        model_id="model-artifact",
        research_dataset_id="research_demo",
        train_end=None,
        test_start="2019-01-01",
        test_end="2019-06-30",
        prediction_mode="a_share_oos",
        prediction_column="raw_prediction",
    )
    with pytest.raises(
        ValueError,
        match="persisted_model_prediction_window_uncovered",
    ):
        runner.build_prediction_result_from_persisted_model(
            layout=StorageLayout(project_root=tmp_path),
            dataset_catalog=ResearchDatasetCatalog((_dataset_spec(),)),
            prediction_spec=prediction_spec,
            persisted_model=artifact,
            resource_policy=None,
        )


def test_quarter_id_must_match_prediction_window_and_repeated_champion_is_valid() -> (
    None
):
    metadata = _metadata(("x",))
    valid = {
        "quarter_id": "2019Q1",
        "model_id": "champion",
        "training_window": {"start": "2014-01-01", "end": "2017-12-31"},
        "validation_window": {"start": "2018-01-01", "end": "2018-12-31"},
        "prediction_window": {"start": "2019-01-01", "end": "2019-03-31"},
        "model_metadata": metadata,
    }
    assert _validated_quarterly_payload([valid])[0]["model_id"] == "champion"
    invalid = {
        **valid,
        "prediction_window": {"start": "2020-01-01", "end": "2020-03-31"},
    }
    with pytest.raises(ValueError, match="model_artifact_quarter_window_mismatch"):
        _validated_quarterly_payload([invalid])


def test_same_champion_and_ic_columns_are_canonicalized(tmp_path: Path) -> None:
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(1, model_id="same_champion", columns=("x",)),
            _quarterly_result(2, model_id="same_champion", columns=("x",)),
        ),
    )
    artifact = read_quarterly_model_artifact(root)
    assert [item["model_id"] for item in artifact.quarterly_models] == [
        "same_champion",
        "same_champion",
    ]

    ic_result = _quarterly_result(
        1,
        model_id="ic_champion",
        columns=("x", "y"),
        predictor=ICWeightedRankEnsemble(
            columns=("x",),
            weights=(1.0,),
            coefficient_count=1,
        ),
    )
    ic_root = write_quarterly_model_artifact(
        tmp_path / "ic-model",
        model_id="model-artifact",
        envelope=_envelope(),
        quarterly_results=(ic_result,),
    )
    ic_artifact = read_quarterly_model_artifact(ic_root)
    assert ic_artifact.quarterly_models[0]["model_metadata"]["rank_parameters"] == {
        "columns": ["x"],
        "weights": [1.0],
    }


def test_model_member_mutation_is_rejected_before_loading_predictor(
    tmp_path: Path,
) -> None:
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        envelope=_envelope(),
        quarterly_results=(_quarterly_result(1, model_id="champion", columns=("x",)),),
    )
    (root / "models" / "2019Q1.pkl").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_quarterly_model_artifact(root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "dependency_status",
            {"status": "available", "package": "sklearn", "version": "test"},
        ),
        ("rows_input", 1),
        ("filter_columns_used", []),
    ),
)
def test_quarterly_model_core_rejects_contract_drift(
    field: str,
    value: object,
) -> None:
    metadata = _metadata(("x",))
    metadata[field] = value
    payload = {
        "quarter_id": "2019Q1",
        "model_id": "champion",
        "training_window": {"start": "2014-01-01", "end": "2017-12-31"},
        "validation_window": {"start": "2018-01-01", "end": "2018-12-31"},
        "prediction_window": {"start": "2019-01-01", "end": "2019-03-31"},
        "model_metadata": metadata,
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        _validated_quarterly_payload([payload])


def test_quarterly_model_requires_two_level_selection() -> None:
    metadata = _metadata(("x",))
    metadata.pop("model_selection")
    payload = {
        "quarter_id": "2019Q1",
        "model_id": "champion",
        "training_window": {"start": "2014-01-01", "end": "2017-12-31"},
        "validation_window": {"start": "2018-01-01", "end": "2018-12-31"},
        "prediction_window": {"start": "2019-01-01", "end": "2019-03-31"},
        "model_metadata": metadata,
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        _validated_quarterly_payload([payload])


def test_a_share_model_folds_require_formal_history_start_year() -> None:
    protocol = a_share.protocol(scenario_id="pv_all", access_window="development")
    manifest = protocol.manifest()
    sessions = pd.date_range("2017-01-01", "2019-03-31", freq="B")
    with pytest.raises(ValueError, match="model_chronology_formal_history_missing"):
        a_share.model_folds(
            sessions,
            chronology=manifest["model_chronology"],
            formal_window=manifest["formal_window"],
            development_window=manifest["development_window"],
        )


def test_mixed_rank_and_supervised_trials_share_validation_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.catalog.models import ModelCatalog
    from quant_research_platform.research.models import trial_selection

    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 9,
            "timestamp": pd.to_datetime(
                ["2018-12-31"] * 3 + ["2019-01-01"] * 3 + ["2019-01-02"] * 3
            ),
            "asset_id": ["a", "b", "c"] * 3,
            "x": [1.0, 2.0, 3.0, 1.0, np.nan, 3.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 1.0, 2.0, np.nan, 2.0, 3.0, 4.0],
            "label": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0],
            "eligible": [True] * 9,
        }
    )
    catalog = ModelCatalog(
        (
            ModelSpec(
                model_id="linear",
                model_algorithm="linear_regression",
                input_factor_columns=("x",),
                label_column="label",
            ),
            ModelSpec(
                model_id="rank",
                model_algorithm="equal_weight_rank_ensemble",
                input_factor_columns=("x", "y"),
                label_column="label",
            ),
        )
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction",
        model_id="linear",
        research_dataset_id="research_demo",
        train_end="2018-12-31",
        test_start="2019-01-03",
        candidate_model_ids=("rank",),
        model_selection_validation_start="2019-01-01",
        model_selection_validation_end="2019-01-02",
    )
    keys: list[tuple[tuple[pd.Timestamp, str], ...]] = []
    original_evaluate = trial_selection.evaluate_model_trial_metrics_from_chunks

    def capture(chunks, **kwargs):  # noqa: ANN001, ANN003
        observed: list[tuple[pd.Timestamp, str]] = []

        def replay():
            for chunk in chunks:
                frame_value = chunk.value if isinstance(chunk, OwnedValue) else chunk
                observed.extend(
                    frame_value[["timestamp", "asset_id"]].itertuples(
                        index=False,
                        name=None,
                    )
                )
                yield chunk

        result = original_evaluate(replay(), **kwargs)
        keys.append(tuple(observed))
        return result

    monkeypatch.setattr(
        trial_selection,
        "evaluate_model_trial_metrics_from_chunks",
        capture,
    )
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        selected, _metadata_value = select_prediction_model_spec(
            prediction_spec=prediction_spec,
            dataset_spec=_dataset_spec(),
            model_catalog=catalog,
            chunks=lambda columns, *, input_columns: (frame.loc[:, list(columns)],),
        )
    assert selected.model_id in {"linear", "rank"}
    assert len(keys) == 2
    assert keys[0] == keys[1]
    assert len(keys[0]) == 5


def test_fitting_counts_raw_filter_and_label_rows_once() -> None:
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="model_target",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]
            ),
            "asset_id": ["a", "b", "c", "d"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "model_target": [1.0, 2.0, 3.0, 4.0],
            "eligible": [True, True, True, False],
            "forward_return_20d_available_at": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2019-01-03", "2018-01-04"]
            ),
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        result = fitting._fit_model_result_owned_from_frames(  # noqa: SLF001
            (frame,),
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2018-12-31",
        )
        model_result = result.value if isinstance(result, OwnedValue) else result
        try:
            assert model_result.metadata["rows_input"] == 4
            assert model_result.metadata["rows_after_filter_mode"] == 3
            assert model_result.metadata["rows_after_input_label_dropna"] == 2
        finally:
            if isinstance(result, OwnedValue) and result.lease.active:
                result.release(owner="test.model.fitting")


def test_training_asset_scope_owns_scoped_payload_until_consumer_release() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=8 * 1024**2,
        explicit_cap_bytes=2 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        scoped = training_scope_chunks((frame,), asset_ids=("b",))
        selected = next(scoped)
        assert selected["asset_id"].tolist() == ["b"]
        assert policy.used_memory_bytes > 0
        scoped.close()
        assert policy.used_memory_bytes == 0


def test_training_asset_scope_preserves_consumer_primary_on_early_close() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "value": [1.0, 2.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=8 * 1024**2,
        explicit_cap_bytes=2 * 1024**2,
    )
    primary = RuntimeError("consumer_primary")
    with runtime_resource_policy_scope(policy):
        scoped = training_scope_chunks((frame,), asset_ids=("b",))
        next(scoped)
        with pytest.raises(RuntimeError) as raised:
            scoped.throw(primary)
        assert raised.value is primary
        assert policy.used_memory_bytes == 0


def test_prediction_lineage_requires_exact_model_dependency_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_identity = "a" * 64
    profile = SimpleNamespace(
        producer_bindings={
            "prediction": ProducerBinding(ProducerKind.PREDICTION, lambda: None),
            "model": ProducerBinding(ProducerKind.MODEL, lambda: None),
        }
    )
    target = TargetSpec(
        target_id="prediction",
        dependencies=("model",),
        producer_id="prediction",
    )
    protocol = SimpleNamespace()
    envelope = {
        "logical_artifact_id": "prediction",
        "upstream_identities": {"model": model_identity},
    }
    manifest = {
        "metadata": {
            "model_metadata": {
                "model_artifact_id": "model",
                "model_artifact_identity": "b" * 64,
                "model_artifact_schema": "qrp-a-share-quarterly-model-artifact",
            }
        }
    }

    monkeypatch.setattr(
        research_workflow,
        "read_semantic_artifact",
        lambda *_args, **_kwargs: envelope,
    )
    monkeypatch.setattr(
        research_workflow,
        "validate_prediction_output_metadata_shape",
        lambda _metadata: None,
    )

    @contextmanager
    def bundle(*_args, **_kwargs):  # noqa: ANN002, ANN003
        yield manifest, object()

    monkeypatch.setattr(research_workflow, "read_prediction_output_bundle", bundle)
    with pytest.raises(ValueError, match="prediction_inner_artifact_invalid"):
        research_workflow._validate_semantic_target_root(  # noqa: SLF001
            profile,
            protocol,
            target,
            Path("prediction-root"),
            "c" * 64,
        )


def test_model_selection_rejects_protocol_family_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_target = "model-target"
    expected_model_id = "scenario_linear_regression"
    profile = SimpleNamespace(
        producer_bindings={
            model_target: ProducerBinding(ProducerKind.MODEL, lambda: None),
        },
        models=SimpleNamespace(
            get=lambda _model_id: SimpleNamespace(input_columns=()),
        ),
    )
    protocol = SimpleNamespace(
        scenario_id="scenario",
        manifest=lambda: {"selection_tie_break": ["linear_regression"]},
    )
    target = TargetSpec(model_target, producer_id=model_target)
    selection = {
        "candidate_family_ids": [expected_model_id],
        "tie_break_order": [expected_model_id],
        "selected_family_id": "different_family",
    }
    persisted = SimpleNamespace(
        quarterly_models=(
            {
                "model_metadata": {
                    "model_selection": selection,
                    "model_algorithm": "linear_regression",
                    "input_columns": [],
                }
            },
        )
    )
    monkeypatch.setattr(
        research_workflow,
        "read_semantic_artifact",
        lambda *_args, **_kwargs: {"logical_artifact_id": model_target},
    )
    monkeypatch.setattr(
        research_workflow,
        "read_quarterly_model_artifact",
        lambda *_args, **_kwargs: persisted,
    )
    with pytest.raises(ValueError, match="model_selection_family_mismatch"):
        research_workflow._validate_semantic_target_root(  # noqa: SLF001
            profile,
            protocol,
            target,
            Path("model-root"),
            "a" * 64,
        )


def test_quarterly_writer_releases_owned_member_on_primary_failure(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=8 * 1024 * 1024,
        explicit_cap_bytes=2 * 1024 * 1024,
    )
    quarter = _quarterly_result(
        1,
        model_id="champion",
        columns=("x",),
        predictor=_UnpicklablePredictor("x"),
    )

    def yielded() -> object:
        lease = policy.lease("research.test.quarter", 1024)
        owned = OwnedValue(quarter, lease)
        try:
            yield owned
        finally:
            if owned.lease.active:
                owned.release(owner="research.test.quarter")

    with runtime_resource_policy_scope(policy):
        with pytest.raises(RuntimeError, match="predictor_serialization_failed"):
            write_quarterly_model_artifact(
                tmp_path / "model",
                model_id="model-artifact",
                envelope=_envelope(),
                quarterly_results=yielded(),
            )
    assert policy.used_memory_bytes == 0


def test_rank_missing_contributes_neutral_without_renormalization() -> None:
    predictor = RankEnsemblePredictor(
        columns=("x", "y"),
        directions=(1.0, 1.0),
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2019-01-01"] * 2),
            "x": [1.0, 2.0],
            "y": [np.nan, 4.0],
        }
    )
    assert predictor.predict(frame).tolist() == pytest.approx([-0.125, 0.125])


def test_trial_selection_prefers_defined_negative_and_tie_breaks_undefined() -> None:
    partial = pd.DataFrame(
        {
            "trial_id": ["defined", "undefined"],
            "validation_rankic": [-0.4, np.nan],
        }
    )
    assert (
        select_best_validation_trial(
            partial,
            trial_id_column="trial_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=("defined", "undefined"),
        )
        == "defined"
    )
    all_undefined = partial.assign(validation_rankic=np.nan)
    assert (
        select_best_validation_trial(
            all_undefined,
            trial_id_column="trial_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=("undefined", "defined"),
        )
        == "undefined"
    )


def test_seed_changes_by_quarter_component_and_refit_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial_q1 = derive_seed(42, "a_share_model_trial", "ridge", "config", "2019Q1", 0)
    trial_q2 = derive_seed(42, "a_share_model_trial", "ridge", "config", "2019Q2", 0)
    refit_q1 = derive_seed(42, "a_share_model_refit", "ridge", "config", "2019Q1")
    assert trial_q1 == derive_seed(
        42, "a_share_model_trial", "ridge", "config", "2019Q1", 0
    )
    assert len({trial_q1, trial_q2, refit_q1}) == 3
    base = ModelSpec(
        model_id="ridge_family",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        random_seed=999,
    )
    q1_specs = _a_share_trial_specs(
        base,
        n_train=10,
        resolve_trial_parameters=resolve_trial_parameters,
        quarter_id="2019Q1",
        root_seed=42,
    )
    q2_specs = _a_share_trial_specs(
        base,
        n_train=10,
        resolve_trial_parameters=resolve_trial_parameters,
        quarter_id="2019Q2",
        root_seed=42,
    )
    assert q1_specs[0].random_seed != q2_specs[0].random_seed
    assert q1_specs[0].random_seed != base.random_seed

    captured: list[int] = []

    def fit_owned(_frames, spec, **_kwargs):  # noqa: ANN001, ANN003
        captured.append(spec.random_seed)
        return ModelFitResult(
            coefficients=np.array([0.0, 1.0]),
            metadata=_metadata(("x",)),
            prediction_model=_ColumnPredictor("x"),
        )

    monkeypatch.setattr(fitting, "_fit_model_result_owned_from_frames", fit_owned)
    _fit_selected_model_result(
        dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
        selected_spec=base,
        chunks=lambda *_args, **_kwargs: (),
        fitting_cutoff="2018-12-31",
        quarter_id="2019Q1",
        root_seed=42,
        selection_metadata={},
    )
    assert captured == [
        derive_seed(
            42, "a_share_model_refit", "ridge_regression", "ridge_family", "2019Q1"
        )
    ]


def test_a_share_evaluation_catalog_contains_all_formal_scenarios() -> None:
    assert {
        "pv_all_evaluation",
        "fusion_nf_core_evaluation",
        "fusion_nf_expectations_evaluation",
    } == {spec.evaluation_id for spec in research_profile().evaluations.list()}


def test_convergence_warning_is_a_model_fit_failure() -> None:
    from sklearn.exceptions import ConvergenceWarning

    class WarningEstimator:
        def fit(self, _matrix, _target, **_kwargs):
            warnings.warn("not converged", ConvergenceWarning, stacklevel=2)

    spec = ModelSpec(
        model_id="ridge_warning",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        hyperparameters={"alpha": 1.0},
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2018-01-01", "2018-01-02"]),
            "x": [1.0, 2.0],
            "label": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="model_fit_non_converged"):
        fitting._fit_predictor(
            spec,
            training=frame,
            y=np.array([1.0, 2.0]),
            estimator=WarningEstimator(),
            threads=1,
        )


def test_two_level_selection_mutations_are_rejected() -> None:
    trial = {
        "model_id": "ridge_config_0",
        "validation_rows": 2,
        "validation_ic": 0.1,
        "validation_rankic": 0.2,
        "validation_ic_reason": None,
        "validation_rankic_reason": None,
        "validation_mse": 1.0,
        "resolved_hyperparameters": {"alpha": 1.0},
        "random_seed": 7,
        "n_train": 20,
        "dependency_status": {
            "status": "available",
            "package": "sklearn",
            "version": "test",
        },
    }
    family = {
        "family_id": "ridge",
        "config_tie_break_order": ["ridge_config_0"],
        "selected_config_id": "ridge_config_0",
        "trials": [trial],
        "validation_rows": 2,
        "selection_metric_value": 0.2,
        "selection_metric_reason": None,
    }
    payload = {
        "status": "selected",
        "selection_metric": "validation_rankic",
        "selection_direction": "maximize",
        "validation_start": "2018-01-01",
        "validation_end": "2018-12-31",
        "candidate_family_ids": ["ridge"],
        "tie_break_order": ["ridge"],
        "selected_family_id": "ridge",
        "selected_model_id": "ridge_config_0",
        "families": [family],
    }
    validate_model_selection_metadata(payload)
    for mutation in (
        {**payload, "selected_model_id": "other"},
        {
            **payload,
            "families": [{**family, "selection_metric_value": 0.3}],
        },
        {
            **payload,
            "families": [
                {
                    **family,
                    "trials": [{**trial, "random_seed": -1}],
                }
            ],
        },
    ):
        with pytest.raises(ValueError, match="invalid_model_metadata"):
            validate_model_selection_metadata(mutation)

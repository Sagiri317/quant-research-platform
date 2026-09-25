from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
from quant_research_platform.contracts.models import (
    A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    ModelSpec,
    model_spec_payload,
    validate_model_selection_metadata,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.targets import (
    ProducerBinding,
    ProducerKind,
    TargetSpec,
)
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
    _fit_selected_model_result,
    _refit_model_spec,
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
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_iterdir,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_replace,
    filesystem_stat,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research as research_workflow
from tests.helpers.research_configuration import configured_models
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.prediction_output_helpers import write_prediction_result
from tests.research.workflow_test_support import (
    QUARTERLY_FIXTURE_PROTOCOL,
    _column_predictor,
    _fold,
    _metadata,
    _quarterly_result,
    _selection_metadata,
    _trial_reference,
    fixture_quarterly_results,
    fixture_refit_reference,
    fixture_validation_coverage,
)


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): filesystem_read_bytes(parent / name)
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


def _mock_prediction_dataset(monkeypatch, frame):
    import pyarrow as pa

    @dataclass
    class Session:
        logical_columns: tuple = ()
        physical_columns: tuple = ()
        owns_inventory: bool = False
        owns_lease: bool = False

        @property
        def arrow_schema(self):
            return pa.Schema.from_pandas(frame, preserve_index=False)

        @property
        def spec(self):
            return {"timestamp_column": "timestamp"}

        @contextmanager
        def filtered_view(self, _window):
            yield self

    @contextmanager
    def open_session(*_args, **_kwargs):
        yield Session()

    monkeypatch.setattr(runner, "open_dataset_session", open_session)
    monkeypatch.setattr(
        runner, "iter_dataset_chunks", lambda *_args, **_kwargs: iter((frame,))
    )


def test_full_scope_family_metadata_fits_quarter_shards_and_protocol_member_limit(
    tmp_path: Path,
):
    from quant_research_platform.contracts.models import (
        A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    )
    from quant_research_platform.research.models.production import QuarterlyModelResult
    from quant_research_platform.research.models.supervised import preflight_model_specs
    from quant_research_platform.storage.manifests import read_qrp_json

    protocol = a_share.protocol("fusion_nf_expectations")
    specs = tuple(
        replace(spec, label_column="label")
        for spec in configured_models(protocol.scenario_id, factor_scope="full")
    )
    protocol_payload = {
        **protocol.manifest(),
        "research_configuration": {
            "models": [
                {
                    "model_id": spec.model_id,
                    "hyperparameters": dict(spec.hyperparameters),
                }
                for spec in specs
            ]
        },
    }
    preflight_model_specs(specs)
    inputs = tuple(
        dict.fromkeys(column for spec in specs for column in spec.input_columns)
    )
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(192, len(inputs))), columns=inputs)
    frame["timestamp"] = pd.bdate_range("2018-01-02", periods=32).repeat(6)
    frame["asset_id"] = [f"asset_{i}" for _ in range(32) for i in range(6)]
    frame["market_id"] = "a_share"
    frame["eligible"] = True
    frame["label"] = frame[inputs[0]] * 0.3 + rng.normal(scale=0.01, size=len(frame))
    frame["label_available_at"] = frame["timestamp"]
    dataset = replace(
        a_share.research_dataset_catalog().get(protocol.scenario_id), data_id="fixture"
    )
    layout = StorageLayout(tmp_path)
    references = tuple(
        _trial_reference(
            layout,
            _fold(1),
            family_id=spec.model_id,
            model_id=spec.model_id,
            protocol=protocol_payload,
            model_spec=spec,
            root_seed=42,
            validation_specs=specs,
        )
        for spec in specs
    )
    selection = _selection_metadata(
        family_id=specs[0].model_id, model_id=specs[0].model_id, columns=inputs
    )
    selection["candidate_family_ids"] = selection["tie_break_order"] = [
        spec.model_id for spec in specs
    ]
    selection["common_validation_coverage"] = fixture_validation_coverage(
        _fold(1), specs
    )
    selection["families"] = [
        {
            "family_id": spec.model_id,
            "config_tie_break_order": [spec.model_id],
            "selected_config_id": spec.model_id,
            "trial_ledger_semantic_identity": reference["semantic_identity"],
        }
        for spec, reference in zip(specs, references, strict=True)
    ]
    envelope = build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-03-31",
            "access": "development",
        },
        upstream_identities={"fixture": "d" * 64},
        semantic_metadata={
            "target_contract": {
                "protocol": protocol_payload,
                "model_specs": [model_spec_payload(spec) for spec in specs],
            }
        },
    )

    def chunks(columns, **_kwargs):
        yield frame.loc[:, list(columns)]

    def fitted():
        for spec, reference in zip(specs, references, strict=True):
            result, lease = _fit_selected_model_result(
                training_counts={},
                dataset_spec=dataset,
                selected_spec=spec,
                candidate_parameters=dict(spec.hyperparameters),
                chunks=chunks,
                fitting_cutoff="2018-12-31",
                quarter_id="2019Q1",
                root_seed=42,
            )
            item = QuarterlyModelResult(
                _fold(1),
                spec.model_id,
                result,
                spec.model_id,
                selection,
                (
                    reference,
                    fixture_refit_reference(layout, reference, result.metadata),
                ),
            )
            owned = OwnedValue(item, lease)
            item = result = None
            try:
                yield owned
            finally:
                item = result = None
                if isinstance(owned, OwnedValue) and owned.lease.active:
                    owned.release(owner="research.model.artifact")
                owned = None

    with runtime_resource_policy_scope() as policy:
        root = write_quarterly_model_artifact(
            tmp_path / "full_models",
            model_id="model-artifact",
            envelope=envelope,
            quarterly_results=fixture_quarterly_results(fitted()),
            trial_layout=layout,
        )
        artifact = read_quarterly_model_artifact(root, trial_layout=layout)
        assert len(artifact.quarterly_models) == 8
        shard = root / "models" / "2019Q1" / "metadata.json"
        payload = read_qrp_json(
            shard,
            expected_schema_id=A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
            error="invalid_full_metadata",
        )
        assert len(payload["models"]) == 8
        assert filesystem_stat(shard).st_size < 1024**2
        assert any(
            item["model_metadata"]["scaler"] is not None for item in payload["models"]
        )
        from quant_research_platform.research.models import supervised
        from quant_research_platform.research.models.persisted import (
            read_quarterly_predictor,
        )

        for entry in artifact.quarterly_models:
            algorithm = entry["model_metadata"]["model_algorithm"]
            with read_quarterly_predictor(
                artifact, "2019Q1", entry["family_id"], resource_policy=policy
            ) as actual:
                if algorithm in {
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                }:
                    changed = (
                        replace(
                            actual,
                            directions=tuple(-value for value in actual.directions),
                        )
                        if algorithm == "equal_weight_rank_ensemble"
                        else replace(
                            actual, weights=tuple(-value for value in actual.weights)
                        )
                    )
                    with pytest.raises(
                        ValueError, match="model_state_metadata_mismatch"
                    ):
                        fitting.validate_model_state(changed, entry["model_metadata"])
                    changed = None
                elif algorithm == "random_forest_regressor":
                    with pytest.MonkeyPatch.context() as patch:
                        patch.setattr(
                            supervised,
                            "model_cpu_capacity",
                            lambda: pytest.fail(
                                "persisted model validation rediscovered host capacity"
                            ),
                        )
                        fitting.validate_model_state(actual, entry["model_metadata"])
                    assert (
                        actual.n_jobs
                        == entry["model_metadata"]["resolved_hyperparameters"]["n_jobs"]
                    )
        assert policy.used_memory_bytes == 0
    development = len(
        pd.period_range(a_share.DEVELOPMENT_START, a_share.DEVELOPMENT_END, freq="Q")
    )
    total = len(
        pd.period_range(a_share.DEVELOPMENT_START, a_share.HOLDOUT_END, freq="Q")
    )
    assert (development, total) == (20, 28)
    per_quarter_members = len(specs) + 2
    assert (1 + development * per_quarter_members, 1 + total * per_quarter_members) == (
        201,
        281,
    )


def _fixture_protocol(specs):
    return {
        "fixture": True,
        "research_configuration": {
            "models": [
                {
                    "model_id": spec.model_id,
                    "hyperparameters": dict(spec.hyperparameters),
                }
                for spec in specs
            ]
        },
    }


def _envelope(*, software_identity="c" * 64, specs=None) -> object:
    if specs is None:
        specs = tuple(
            ModelSpec(
                model_id=item["model_id"],
                model_algorithm=item["model_id"],
                input_factor_columns=("x",),
                label_column="label",
            )
            for item in QUARTERLY_FIXTURE_PROTOCOL["research_configuration"]["models"]
        )
    protocol = _fixture_protocol(specs)
    return build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity=software_identity,
        research_window={
            "start": "2019-01-01",
            "end": "2019-06-30",
            "access": "development",
        },
        upstream_identities={"fixture": "d" * 64},
        semantic_metadata={
            "target_contract": {
                "protocol": protocol,
                "model_specs": [model_spec_payload(spec) for spec in specs],
            }
        },
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
    specs = tuple(
        ModelSpec(
            model_id="linear_" + column,
            model_algorithm="linear_regression",
            input_factor_columns=(column,),
            label_column="label",
        )
        for column in ("x", "y")
    )
    protocol = _fixture_protocol(specs)

    def results():
        for quarter in (1, 2):
            members = [
                _quarterly_result(
                    quarter,
                    trial_layout=StorageLayout(tmp_path),
                    model_spec=spec,
                    protocol=protocol,
                    validation_specs=specs,
                    validation_rankic=0.3 if index == quarter - 1 else 0.1,
                    predictor=_column_predictor(spec.input_columns[0], float(quarter)),
                )
                for index, spec in enumerate(specs)
            ]
            selection = members[quarter - 1].selection_metadata
            selection.update(
                candidate_family_ids=[spec.model_id for spec in specs],
                tie_break_order=[spec.model_id for spec in specs],
                families=[
                    member.selection_metadata["families"][0] for member in members
                ],
            )
            for member in members:
                yield OwnedValue(
                    replace(member, selection_metadata=selection),
                    policy.lease(
                        "test.model",
                        fitting._linear_model_state_upper_bound_bytes(specs[0]),
                    ),
                )

    with runtime_resource_policy_scope() as policy:
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=StorageLayout(tmp_path),
            envelope=_envelope(specs=specs),
            quarterly_results=fixture_quarterly_results(results()),
        )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
    from quant_research_platform.research.statistics_production import _model_trials

    trials = _model_trials(artifact, StorageLayout(tmp_path))
    assert len(artifact.trial_ledgers) == 8
    assert len(trials) == 4
    assert set(trials.config_id) == {"linear_x", "linear_y"}
    assert trials.observation_count.tolist() == [2, 2, 2, 2]
    spec = _dataset_spec()
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 4,
            "asset_id": ["a", "b", "a", "b"],
            "timestamp": pd.to_datetime(["2019-02-01"] * 2 + ["2019-05-01"] * 2),
            "x": [3.0, 5.0, np.nan, 7.0],
            "y": [np.nan, 6.0, 4.0, 8.0],
            "eligible": [True] * 4,
        }
    )
    _mock_prediction_dataset(monkeypatch, frame)
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
    assert len(output) == 6
    assert output["model_id"].tolist() == [
        "linear_x",
        "linear_x",
        "linear_y",
        "linear_x",
        "linear_y",
        "linear_y",
    ]
    assert output["raw_prediction"].tolist() == pytest.approx(
        [4.0, 6.0, 7.0, 9.0, 6.0, 10.0]
    )
    assert output.iloc[[2, 3]]["alpha_score"].tolist() == pytest.approx([0.0, 0.0])
    active_only = output.loc[output.asset_id.eq("a")]
    assert active_only.model_id.tolist() == ["linear_x", "linear_y"]
    assert active_only.raw_prediction.tolist() == [4.0, 6.0]
    assert [
        selection["selected_family_id"]
        for selection in artifact.quarterly_selections.values()
    ] == ["linear_x", "linear_y"]
    assert policy.used_memory_bytes == 0


def test_persisted_prediction_alpha_is_permutation_invariant_and_reader_uses_alpha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with runtime_resource_policy_scope() as policy:
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=StorageLayout(tmp_path),
            envelope=_envelope(),
            quarterly_results=fixture_quarterly_results(
                OwnedValue(
                    item,
                    policy.lease(
                        "test.model",
                        fitting._linear_model_state_upper_bound_bytes(
                            ModelSpec(
                                model_id=item.model_id,
                                model_algorithm=item.model_result.metadata[
                                    "model_algorithm"
                                ],
                                input_factor_columns=tuple(
                                    item.model_result.metadata["input_columns"]
                                ),
                                label_column="label",
                            )
                        ),
                    ),
                )
                for item in (
                    _quarterly_result(
                        1,
                        trial_layout=StorageLayout(tmp_path),
                        model_spec=ModelSpec(
                            model_id="linear_regression",
                            model_algorithm="linear_regression",
                            input_factor_columns=("x",),
                            label_column="label",
                        ),
                    ),
                )
            ),
        )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
    spec = _dataset_spec()
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 3,
            "asset_id": ["000001", "000002", "000003"],
            "timestamp": pd.to_datetime(["2019-02-01"] * 3),
            "x": [1.0, 2.0, 3.0],
            "y": [np.nan] * 3,
            "eligible": [True] * 3,
        }
    )
    _mock_prediction_dataset(monkeypatch, frame)
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
    table = pd.read_parquet(
        native_filesystem_path(bundle / "prediction.parquet")
    ).sort_values("asset_id", kind="mergesort")
    assert table["alpha_score"].tolist() == pytest.approx([-1.0 / 3.0, 0.0, 1.0 / 3.0])
    assert float(table["alpha_score"].mean()) == pytest.approx(0.0)

    from quant_research_platform.research.predictions.alignment import (
        _iter_prediction_joined_chunks,
    )

    requested = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    with runtime_resource_policy_scope(None):
        for mutation in (None, "asset", "timestamp"):
            selected = requested.copy()
            if mutation == "asset":
                selected.loc[selected.asset_id.eq("000001"), "asset_id"] = "1"
            if mutation == "timestamp":
                selected.loc[0, "timestamp"] += pd.Timedelta(days=1)
            with read_prediction_output_bundle(
                bundle,
                expected_output_id="prediction",
                expected_run_id="test_run",
                family_selection="champion",
            ) as (_manifest, lookup):
                assert lookup.prediction_column == "alpha_score"
                joined = _iter_prediction_joined_chunks(
                    (selected.iloc[:1], selected.iloc[1:]),
                    prediction_lookup=lookup,
                    input_columns=("x",),
                    model_algorithm="linear_regression",
                    sample_filter_mode=prediction_spec.sample_filter_mode,
                    test_start=prediction_spec.test_start,
                    test_end=prediction_spec.test_end,
                )
                if mutation is None:
                    matched = pd.concat(joined, ignore_index=True)
                    assert matched.asset_id.tolist() == ["000003", "000001", "000002"]
                    assert matched.alpha_score.tolist() == pytest.approx(
                        [1.0 / 3.0, -1.0 / 3.0, 0.0]
                    )
                    assert lookup.coverage_counts() == (0, 0)
                else:
                    with pytest.raises(
                        ValueError, match="misaligned_persisted_prediction_keys"
                    ):
                        list(joined)

    shuffled = frame.iloc[[2, 0, 1]].reset_index(drop=True)
    _mock_prediction_dataset(monkeypatch, shuffled)
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
    assert policy.used_memory_bytes == 0


def test_persisted_quarter_request_is_clipped_to_requested_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with runtime_resource_policy_scope() as policy:
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=StorageLayout(tmp_path),
            envelope=_envelope(),
            quarterly_results=fixture_quarterly_results(
                OwnedValue(
                    item,
                    policy.lease(
                        "test.model",
                        fitting._linear_model_state_upper_bound_bytes(
                            ModelSpec(
                                model_id=item.model_id,
                                model_algorithm=item.model_result.metadata[
                                    "model_algorithm"
                                ],
                                input_factor_columns=tuple(
                                    item.model_result.metadata["input_columns"]
                                ),
                                label_column="label",
                            )
                        ),
                    ),
                )
                for item in (
                    _quarterly_result(
                        1,
                        trial_layout=StorageLayout(tmp_path),
                        model_spec=ModelSpec(
                            model_id="linear_regression",
                            model_algorithm="linear_regression",
                            input_factor_columns=("x",),
                            label_column="label",
                        ),
                    ),
                )
            ),
        )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
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
    _mock_prediction_dataset(monkeypatch, frame)
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
    assert policy.used_memory_bytes == 0


def test_persisted_quarter_route_rejects_uncovered_first_quarter(
    tmp_path: Path,
) -> None:
    with runtime_resource_policy_scope() as policy:
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=StorageLayout(tmp_path),
            envelope=_envelope(),
            quarterly_results=fixture_quarterly_results(
                OwnedValue(
                    item,
                    policy.lease(
                        "test.model",
                        fitting._linear_model_state_upper_bound_bytes(
                            ModelSpec(
                                model_id=item.model_id,
                                model_algorithm=item.model_result.metadata[
                                    "model_algorithm"
                                ],
                                input_factor_columns=tuple(
                                    item.model_result.metadata["input_columns"]
                                ),
                                label_column="label",
                            )
                        ),
                    ),
                )
                for item in (
                    _quarterly_result(
                        2,
                        trial_layout=StorageLayout(tmp_path),
                        model_spec=ModelSpec(
                            model_id="linear_regression",
                            model_algorithm="linear_regression",
                            input_factor_columns=("x",),
                            label_column="label",
                        ),
                    ),
                )
            ),
        )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
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
    assert policy.used_memory_bytes == 0


def test_quarter_id_must_match_prediction_window_and_repeated_champion_is_valid() -> (
    None
):
    metadata = _metadata(("x",))
    valid = {
        "quarter_id": "2019Q1",
        "family_id": "linear_regression",
        "model_id": "champion",
        "training_window": {"start": "2014-01-01", "end": "2017-12-31"},
        "validation_window": {"start": "2018-01-01", "end": "2018-12-31"},
        "prediction_window": {"start": "2019-01-01", "end": "2019-03-31"},
        "model_metadata": metadata,
    }
    selections = {
        "2019Q1": _selection_metadata(
            family_id="linear_regression", model_id="champion", columns=("x",)
        )
    }
    assert (
        _validated_quarterly_payload([valid], selections)[0]["model_id"] == "champion"
    )
    invalid = {
        **valid,
        "prediction_window": {"start": "2020-01-01", "end": "2020-03-31"},
    }
    with pytest.raises(ValueError, match="model_artifact_quarter_window_mismatch"):
        _validated_quarterly_payload([invalid], selections)


def test_same_champion_and_ic_columns_are_canonicalized(tmp_path: Path) -> None:
    with runtime_resource_policy_scope() as policy:
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=StorageLayout(tmp_path),
            envelope=_envelope(),
            quarterly_results=fixture_quarterly_results(
                OwnedValue(
                    item,
                    policy.lease(
                        "test.model",
                        fitting._linear_model_state_upper_bound_bytes(
                            ModelSpec(
                                model_id=item.model_id,
                                model_algorithm=item.model_result.metadata[
                                    "model_algorithm"
                                ],
                                input_factor_columns=tuple(
                                    item.model_result.metadata["input_columns"]
                                ),
                                label_column="label",
                            )
                        ),
                    ),
                )
                for item in (
                    _quarterly_result(
                        1,
                        trial_layout=StorageLayout(tmp_path),
                        model_spec=ModelSpec(
                            model_id="linear_regression",
                            model_algorithm="linear_regression",
                            input_factor_columns=("x",),
                            label_column="label",
                        ),
                    ),
                    _quarterly_result(
                        2,
                        trial_layout=StorageLayout(tmp_path),
                        model_spec=ModelSpec(
                            model_id="linear_regression",
                            model_algorithm="linear_regression",
                            input_factor_columns=("x",),
                            label_column="label",
                        ),
                    ),
                )
            ),
        )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
    assert [item["model_id"] for item in artifact.quarterly_models] == [
        "linear_regression",
        "linear_regression",
    ]

    ic_spec = ModelSpec(
        model_id="ic_weighted_rank_ensemble",
        model_algorithm="ic_weighted_rank_ensemble",
        input_factor_columns=("x", "y"),
        label_column="label",
    )
    ic_protocol = _fixture_protocol((ic_spec,))
    ic_result = _quarterly_result(
        1,
        trial_layout=StorageLayout(tmp_path),
        model_spec=ic_spec,
        protocol=ic_protocol,
        predictor=ICWeightedRankEnsemble(
            columns=("x",),
            weights=(1.0,),
        ),
    )
    with runtime_resource_policy_scope() as policy:
        ic_root = write_quarterly_model_artifact(
            tmp_path / "ic-model",
            model_id="model-artifact",
            trial_layout=StorageLayout(tmp_path),
            envelope=_envelope(specs=(ic_spec,)),
            quarterly_results=fixture_quarterly_results(
                OwnedValue(
                    item,
                    policy.lease(
                        "test.model",
                        fitting._linear_model_state_upper_bound_bytes(
                            ModelSpec(
                                model_id=item.model_id,
                                model_algorithm=item.model_result.metadata[
                                    "model_algorithm"
                                ],
                                input_factor_columns=tuple(
                                    item.model_result.metadata["input_columns"]
                                ),
                                label_column="label",
                            )
                        ),
                    ),
                )
                for item in (ic_result,)
            ),
        )
    ic_artifact = read_quarterly_model_artifact(
        ic_root, trial_layout=StorageLayout(tmp_path)
    )
    assert ic_artifact.quarterly_models[0]["model_metadata"]["rank_parameters"] == {
        "columns": ["x"],
        "weights": [1.0],
    }
    assert policy.used_memory_bytes == 0


def test_model_member_mutation_is_rejected_before_loading_predictor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact, layout = _small_common_coverage_case(tmp_path)
    from quant_research_platform.research.models import persisted

    root = artifact.instance_root
    path = root / "models" / "2019Q1" / "linear_x.pkl"
    replacement = path.with_suffix(".replacement")
    filesystem_write_bytes(replacement, filesystem_read_bytes(path))
    filesystem_replace(replacement, path)
    with monkeypatch.context() as patch:
        patch.setattr(
            persisted.pickle,
            "load",
            lambda *_args: pytest.fail("replaced physical member reached decoder"),
        )
        with pytest.raises(ValueError, match="model_artifact_model_invalid"):
            with persisted.read_quarterly_predictor(artifact, "2019Q1", "linear_x"):
                pytest.fail("replaced physical member reached consumer")
    filesystem_write_bytes(path, b"mutated")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))


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
        "family_id": "linear_regression",
        "model_id": "champion",
        "training_window": {"start": "2014-01-01", "end": "2017-12-31"},
        "validation_window": {"start": "2018-01-01", "end": "2018-12-31"},
        "prediction_window": {"start": "2019-01-01", "end": "2019-03-31"},
        "model_metadata": metadata,
    }
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        _validated_quarterly_payload(
            [payload],
            {
                "2019Q1": _selection_metadata(
                    family_id="linear_regression", model_id="champion", columns=("x",)
                )
            },
        )


def test_quarterly_model_requires_two_level_selection() -> None:
    metadata = _metadata(("x",))
    payload = {
        "quarter_id": "2019Q1",
        "model_id": "champion",
        "training_window": {"start": "2014-01-01", "end": "2017-12-31"},
        "validation_window": {"start": "2018-01-01", "end": "2018-12-31"},
        "prediction_window": {"start": "2019-01-01", "end": "2019-03-31"},
        "model_metadata": metadata,
    }
    with pytest.raises(ValueError, match="model_artifact_quarterly_invalid"):
        _validated_quarterly_payload([payload], {})


def test_a_share_model_folds_require_formal_history_start_year() -> None:
    protocol = a_share.protocol(scenario_id="pv_all", access_window="development")
    manifest = protocol.manifest()
    sessions = pd.date_range("2017-01-01", "2019-03-31", freq="B")
    with pytest.raises(ValueError, match="model_chronology_formal_history_missing"):
        a_share.model_folds(
            sessions,
            chronology=manifest["model_chronology"],
            formal_dataset_window=manifest["formal_dataset_window"],
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
            "label_available_at": pd.to_datetime(
                ["2018-12-31"] * 3 + ["2019-01-01"] * 3 + ["2019-01-02"] * 3
            ),
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
            chunks=lambda columns, *, input_columns, start, end: (
                frame.loc[:, list(columns)],
            ),
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
    protocol = SimpleNamespace(protocol_id="a_share_low_frequency")
    envelope = {
        "logical_artifact_id": "prediction",
        "upstream_identities": {"model": model_identity},
    }
    manifest = {
        "metadata": {
            "model_metadata": {
                "model_artifact_id": "model",
                "model_artifact_identity": "b" * 64,
                "model_artifact_schema": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
            }
        }
    }

    monkeypatch.setattr(
        research_workflow,
        "read_semantic_artifact_record",
        lambda *_args, **_kwargs: (envelope, {}),
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
        protocol_id="a_share_low_frequency",
        scenario_id="scenario",
        manifest=lambda: {"selection_tie_break": ["linear_regression"]},
        model_ids_for_target=lambda _target_id: (expected_model_id,),
    )
    target = TargetSpec(model_target, producer_id=model_target)
    selection = {
        "candidate_family_ids": [expected_model_id],
        "tie_break_order": [expected_model_id],
        "selected_family_id": "different_family",
    }
    persisted = SimpleNamespace(
        quarterly_selections={"2019Q1": selection},
        quarterly_models=(
            {
                "quarter_id": "2019Q1",
                "family_id": "different_family",
                "model_metadata": {
                    "model_algorithm": "linear_regression",
                    "input_columns": [],
                },
            },
        ),
    )
    monkeypatch.setattr(
        research_workflow,
        "read_semantic_artifact_record",
        lambda *_args, **_kwargs: ({"logical_artifact_id": model_target}, {}),
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import persisted

    def fail_dump(*_args, **_kwargs):
        raise RuntimeError("predictor_serialization_failed")

    monkeypatch.setattr(persisted.pickle, "dump", fail_dump)
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024 * 1024,
    )
    quarter = _quarterly_result(
        1,
        trial_layout=StorageLayout(tmp_path),
        model_spec=ModelSpec(
            model_id="linear_regression",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
    )

    def yielded() -> object:
        lease = policy.lease(
            "research.test.quarter",
            fitting._linear_model_state_upper_bound_bytes(
                ModelSpec(
                    model_id="champion",
                    model_algorithm="linear_regression",
                    input_factor_columns=("x",),
                    label_column="label",
                )
            ),
        )
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
                trial_layout=StorageLayout(tmp_path),
                envelope=_envelope(),
                quarterly_results=fixture_quarterly_results(yielded()),
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
    refits = [
        _refit_model_spec(
            base,
            n_train=10,
            candidate_parameters={"regularization_strength": 0.1},
            quarter_id=quarter,
            root_seed=42,
        )
        for quarter in ("2019Q1", "2019Q2")
    ]
    assert refits[0].random_seed != refits[1].random_seed
    assert refits[0].random_seed != base.random_seed
    assert dict(refits[0].hyperparameters) == {"alpha": 1.0}

    captured: list[int] = []

    def fit_owned(_frames, spec, **_kwargs):  # noqa: ANN001, ANN003
        captured.append(spec.random_seed)
        return ModelFitResult(
            metadata=_metadata(("x",)),
            prediction_model=_column_predictor("x"),
        )

    monkeypatch.setattr(fitting, "_fit_model_result_owned_from_frames", fit_owned)
    _fit_selected_model_result(
        training_counts={},
        dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
        selected_spec=base,
        candidate_parameters=dict(base.hyperparameters),
        chunks=lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2018-01-01"]),
                    "asset_id": ["a"],
                    "x": [1.0],
                    "label": [2.0],
                    "label_available_at": pd.to_datetime(["2018-01-01"]),
                    "eligible": [True],
                }
            ),
        ),
        fitting_cutoff="2018-12-31",
        quarter_id="2019Q1",
        root_seed=42,
    )
    assert captured == [
        derive_seed(
            42, "a_share_model_refit", "ridge_regression", "ridge_family", "2019Q1"
        )
    ]


def test_a_share_evaluation_catalog_contains_all_formal_scenarios() -> None:
    assert {
        "pv_all_evaluation",
        "pv_expanded_evaluation",
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
    family = {
        "family_id": "ridge",
        "config_tie_break_order": ["ridge_config_0"],
        "selected_config_id": "ridge_config_0",
        "trial_ledger_semantic_identity": "a" * 64,
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
        "common_validation_coverage": {
            "path": "models/2019Q1/validation-coverage.parquet",
            "semantic_identity": "b" * 64,
            "row_count": 2,
        },
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
                    "trial_ledger_semantic_identity": "invalid",
                }
            ],
        },
    ):
        with pytest.raises(ValueError, match="invalid_model_metadata"):
            validate_model_selection_metadata(mutation)


@pytest.mark.parametrize(
    "max_train_rows,train_rows,refit_rows,trial_alpha,refit_alpha",
    ((None, 3, 6, 30.0, 60.0), (2, 2, 2, 20.0, 20.0), (3, 3, 3, 30.0, 30.0)),
)
def test_selected_protocol_strength_is_resolved_on_real_refit_rows_and_persisted(
    tmp_path: Path,
    max_train_rows,
    train_rows,
    refit_rows,
    trial_alpha,
    refit_alpha,
) -> None:
    import json
    from dataclasses import replace

    from quant_research_platform.catalog.models import ModelCatalog
    from quant_research_platform.contracts.filtering import filter_definition
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.research.models import persisted
    from quant_research_platform.research.models.persisted import (
        read_quarterly_predictor,
    )
    from quant_research_platform.research.models.production import (
        QuarterlyModelResult,
        _select_model_family_winners,
    )
    from quant_research_platform.research.models.supervised import preflight_model_specs
    from quant_research_platform.research.models.trial_selection import (
        ModelSampleCoverage,
        model_validation_coverage_scope,
    )
    from quant_research_platform.storage.experiments import read_trial_ledger_reference
    from quant_research_platform.storage.semantic_artifacts import (
        write_semantic_artifact,
    )
    from tests.research.workflow_test_support import _fold

    layout = StorageLayout(tmp_path).with_target_roots(
        {"fixture": tmp_path / ("d" * 64)}
    )
    dataset = replace(_dataset_spec(), data_id="fixture")
    base = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        random_seed=42,
        max_train_rows=max_train_rows,
    )
    preflight_model_specs((base,))
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(day),
                "asset_id": asset,
                "market_id": "a_share",
                "x": float(i),
                "label": float(i * 2),
                "label_available_at": pd.Timestamp(day),
                "eligible": True,
            }
            for day in ("2017-12-29", "2018-12-28", "2019-01-02")
            for i, asset in enumerate(("a", "b", "c"), start=1)
        ]
    )

    def chunks(columns, **_kwargs):
        return (frame.loc[:, list(columns)],)

    prediction = PredictionSpec(
        prediction_id="selection",
        model_id="ridge",
        research_dataset_id="fixture",
        train_end="2017-12-31",
        test_start="2019-01-01",
        test_end="2019-03-31",
        model_selection_validation_start="2018-01-01",
        model_selection_validation_end="2018-12-31",
    )
    from tests.helpers.research_configuration import research_configuration

    declared_ridge = next(
        item
        for item in research_configuration()["models"]
        if item["model_algorithm"] == "ridge_regression"
    )
    configuration = {
        "models": [
            {
                "model_id": "ridge",
                "hyperparameters": dict(base.hyperparameters),
                "parameter_candidates": declared_ridge["parameter_candidates"],
            }
        ]
    }
    protocol = {"fixture": True, "research_configuration": configuration}
    validation_coverage = ModelSampleCoverage(
        model_validation_coverage_scope("d" * 64, _fold(1), (base,))
    )
    validation_keys = frame.loc[
        frame["timestamp"].eq(pd.Timestamp("2018-12-28")),
        ["timestamp", "asset_id"],
    ]
    validation_coverage.add(validation_keys)
    envelope = build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity=canonical_json_hash(
            {"model_artifact_schema": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID}
        ),
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-03-31",
            "access": "development",
        },
        upstream_identities={"fixture": "d" * 64},
        semantic_metadata={
            "target_contract": {
                "protocol": protocol,
                "model_artifact_schema": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                "model_specs": [model_spec_payload(base)],
            }
        },
    )
    with runtime_resource_policy_scope() as policy:
        winners, metadata, refs, indices = _select_model_family_winners(
            training_counts={},
            validation_coverage=validation_coverage,
            validation_input_columns=("x",),
            selection_spec=prediction,
            dataset_spec=dataset,
            model_catalog=ModelCatalog((base,)),
            chunks=chunks,
            model_ids=("ridge",),
            tie_break_order=("ridge",),
            quarter_id="2019Q1",
            root_seed=42,
            layout=layout,
            run_id="run_refit",
            model_target_id="model-artifact",
            research_protocol_identity=canonical_json_hash(protocol),
            quarter=_fold(1),
            holdout_window={"start": "2020-01-01", "end": "2020-12-31"},
            backend_versions={"sklearn": "test"},
            research_configuration=configuration,
        )
        metadata["common_validation_coverage"] = {
            "path": "models/2019Q1/validation-coverage.parquet",
            "semantic_identity": validation_coverage.semantic_identity,
            "row_count": validation_coverage.row_count,
        }
        selected, index = winners["ridge"], indices["ridge"]
        assert (
            index == 0 and selected.hyperparameters["alpha"] == trial_alpha
        )  # Retained training rows times the selected lambda 10.
        result, lease = _fit_selected_model_result(
            training_counts={},
            dataset_spec=dataset,
            selected_spec=selected,
            candidate_parameters=configuration["models"][0]["parameter_candidates"][
                index
            ],
            chunks=chunks,
            fitting_cutoff="2018-12-31",
            quarter_id="2019Q1",
            root_seed=42,
        )
        try:
            assert result.prediction_model.estimator.alpha == refit_alpha
            fit_keys = frame.loc[
                frame["timestamp"].le(pd.Timestamp("2018-12-31")),
                ["timestamp", "asset_id"],
            ]
            if max_train_rows is not None:
                fit_keys = fit_keys.tail(max_train_rows)
            assert len(fit_keys) == refit_rows
            refit_reference = fixture_refit_reference(
                layout,
                refs[0],
                result.metadata,
                candidate_parameters=configuration["models"][0]["parameter_candidates"][
                    index
                ],
                fit_keys=fit_keys,
            )
            quarterly = QuarterlyModelResult(
                _fold(1),
                selected.model_id,
                result,
                "ridge",
                metadata,
                (*refs, refit_reference),
            )
            root = write_quarterly_model_artifact(
                tmp_path / "model",
                model_id="model-artifact",
                envelope=envelope,
                quarterly_results=fixture_quarterly_results(
                    (OwnedValue(quarterly, lease),),
                    validation_keys=validation_keys,
                ),
                trial_layout=layout,
            )
        finally:
            if lease is not None and lease.active:
                lease.close()
        artifact = read_quarterly_model_artifact(root, trial_layout=layout)
        saved = artifact.quarterly_models[0]["model_metadata"]
        assert pd.Timestamp(saved["train_start"]) > pd.Timestamp(
            artifact.quarterly_models[0]["training_window"]["start"]
        )
        assert saved["rows_after_input_label_dropna"] == refit_rows
        assert saved["resolved_hyperparameters"]["alpha"] == refit_alpha
        plan, trials = read_trial_ledger_reference(layout, artifact.trial_ledgers[0])
        assert plan.candidate_ids[index] == selected.model_id
        assert (
            next(
                record for record in trials if record.candidate_id == selected.model_id
            ).normalized_parameters["hyperparameters"]["alpha"]
            == trial_alpha
        )
        assert all(record.metrics["n_train"] == train_rows for record in trials)
        with read_quarterly_predictor(
            artifact, "2019Q1", "ridge", resource_policy=policy
        ) as predictor:
            assert predictor.estimator.alpha == refit_alpha
            probe = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
            before = predictor.predict(probe).copy()
        original_bytes = _file_bytes(root)
        for mutation in (
            "none_start",
            "early_start",
            "nat_end",
            "cutoff",
            "algorithm",
            "inputs",
            "label",
            "filter",
            "seed",
            "refit_count",
            "refit_strength",
            "winner",
            "backend",
            "scaler",
        ):

            def changed(staging, mutation=mutation):
                for relative, content in original_bytes.items():
                    if relative == Path("manifest.json"):
                        continue
                    destination = staging / relative
                    filesystem_mkdir(destination.parent, parents=True, exist_ok=True)
                    filesystem_write_bytes(destination, content)
                metadata_path = staging / "models/2019Q1/metadata.json"
                payload = json.loads(
                    filesystem_read_text(metadata_path, encoding="utf-8")
                )
                entry = payload["models"][0]
                altered = entry["model_metadata"]
                if mutation == "none_start":
                    altered["train_start"] = None
                elif mutation == "early_start":
                    altered["train_start"] = (
                        pd.Timestamp(entry["training_window"]["start"])
                        - pd.Timedelta(days=1)
                    ).isoformat()
                elif mutation == "nat_end":
                    altered["train_end"] = "NaT"
                elif mutation == "cutoff":
                    altered["fit_cutoff"] = "NaT"
                elif mutation == "algorithm":
                    altered["model_algorithm"] = "lasso_regression"
                elif mutation == "inputs":
                    altered["input_factor_columns"] = altered["input_columns"] = [
                        "other"
                    ]
                    altered["scaler"]["columns"] = ["other"]
                    altered["feature_importance"] = {
                        "other": altered["feature_importance"]["x"]
                    }
                elif mutation == "label":
                    altered["label_column"] = "other_label"
                elif mutation == "filter":
                    altered["sample_filter_mode"] = "unfiltered"
                    altered["filter_columns_used"] = []
                    altered["filter_definition"] = filter_definition("unfiltered")
                elif mutation == "seed":
                    altered["random_seed"] += 1
                elif mutation == "refit_count":
                    for key in (
                        "rows_input",
                        "rows_after_filter_mode",
                        "rows_after_input_label_dropna",
                    ):
                        altered[key] += 1
                elif mutation == "refit_strength":
                    altered["hyperparameters"]["alpha"] *= 2
                    altered["resolved_hyperparameters"]["alpha"] *= 2
                elif mutation == "winner":
                    loser = next(
                        candidate
                        for candidate in plan.candidate_ids
                        if candidate != selected.model_id
                    )
                    entry["model_id"] = loser
                    payload["selection"]["selected_model_id"] = loser
                    payload["selection"]["families"][0]["selected_config_id"] = loser
                if mutation in {"backend", "scaler", "refit_strength"}:
                    relative = "models/2019Q1/ridge.pkl"
                    with persisted._read_model_member(
                        root, artifact.member_records, relative, resource_policy=policy
                    ) as owned:
                        model = owned.value
                        if mutation == "scaler":
                            model.scaler = replace(
                                model.scaler, mean_=(model.scaler.mean_[0] + 1.0,)
                            )
                        else:
                            model.estimator.set_params(alpha=model.estimator.alpha * 2)
                        persisted._write_model_member(
                            staging / relative,
                            model,
                            state_bytes=owned.lease.reserved_bytes,
                        )
                        model = None
                filesystem_write_text(
                    metadata_path, json.dumps(payload), encoding="utf-8"
                )

            altered_root = write_semantic_artifact(
                tmp_path / mutation,
                envelope,
                changed,
                operational_metadata={
                    "model_trial_ledgers": list(artifact.trial_ledgers)
                },
            )
            with pytest.raises(
                ValueError,
                match=(
                    "invalid_model_metadata|model_trial_|model_state_metadata_mismatch"
                    "|model_artifact_training_range_invalid"
                ),
            ):
                corrupted = read_quarterly_model_artifact(
                    altered_root, trial_layout=layout
                )
                with read_quarterly_predictor(
                    corrupted, "2019Q1", "ridge", resource_policy=policy
                ):
                    pass
            assert policy.used_memory_bytes == 0
        assert _file_bytes(root) == original_bytes
        with persisted._read_model_member(
            root,
            artifact.member_records,
            "models/2019Q1/ridge.pkl",
            resource_policy=policy,
        ) as owned:
            altered_metadata = {**saved, "label_column": "other_label"}
            altered = QuarterlyModelResult(
                _fold(1),
                selected.model_id,
                ModelFitResult(
                    altered_metadata,
                    owned.value,
                ),
                "ridge",
                metadata,
                quarterly.trial_ledgers,
            )
            owned.value = altered
            altered = None
            with pytest.raises(ValueError, match="model_trial_parameters_misaligned"):
                write_quarterly_model_artifact(
                    tmp_path / "writer_reject",
                    model_id="model-artifact",
                    envelope=envelope,
                    quarterly_results=fixture_quarterly_results(
                        (owned,), validation_keys=validation_keys
                    ),
                    trial_layout=layout,
                )
        assert policy.used_memory_bytes == 0
        frame.loc[frame.timestamp.ge("2019-01-01"), ["x", "label"]] *= 1000000.0
        later_winners, later_selection, later_refs, later_indices = (
            _select_model_family_winners(
                training_counts={},
                validation_coverage=validation_coverage,
                validation_input_columns=("x",),
                selection_spec=prediction,
                dataset_spec=dataset,
                model_catalog=ModelCatalog((base,)),
                chunks=chunks,
                model_ids=("ridge",),
                tie_break_order=("ridge",),
                quarter_id="2019Q1",
                root_seed=42,
                layout=layout,
                run_id="run_future_variant",
                model_target_id="model-artifact",
                research_protocol_identity=canonical_json_hash(protocol),
                quarter=_fold(1),
                holdout_window={"start": "2020-01-01", "end": "2020-12-31"},
                backend_versions={"sklearn": "test"},
                research_configuration=configuration,
            )
        )
        assert later_indices == indices
        assert later_selection == {
            key: value
            for key, value in metadata.items()
            if key != "common_validation_coverage"
        }
        assert later_refs != refs
        later, later_lease = _fit_selected_model_result(
            training_counts={},
            dataset_spec=dataset,
            selected_spec=later_winners["ridge"],
            candidate_parameters=configuration["models"][0]["parameter_candidates"][
                later_indices["ridge"]
            ],
            chunks=chunks,
            fitting_cutoff="2018-12-31",
            quarter_id="2019Q1",
            root_seed=42,
        )
        try:
            np.testing.assert_array_equal(before, later.prediction_model.predict(probe))
        finally:
            later = None
            if later_lease is not None:
                later_lease.close()
        assert policy.used_memory_bytes == 0


def test_model_publication_rejects_self_consistent_wrong_trial_provenance(
    tmp_path: Path,
) -> None:
    from copy import deepcopy
    from dataclasses import replace

    from tests.research.workflow_test_support import _fold, _trial_reference

    layout = StorageLayout(tmp_path)
    model_spec = ModelSpec(
        model_id="linear_regression",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    original = _quarterly_result(1, trial_layout=layout, model_spec=model_spec)
    for index, (fold, arguments, expected_error) in enumerate(
        (
            (
                _fold(1),
                {"dataset_id": "other", "data_identity": "e" * 64},
                "model_validation_coverage_mismatch",
            ),
            (
                _fold(1),
                {"target_id": "other_model"},
                "model_trial_artifact_binding_invalid",
            ),
            (
                _fold(1),
                {"protocol": {"different_protocol": True}},
                "model_trial_artifact_binding_invalid",
            ),
            (_fold(2), {}, "model_trial_references_missing"),
            (_fold(1), {"family_id": "other_family"}, "model_trial_references_missing"),
        )
    ):
        reference = _trial_reference(
            layout,
            fold,
            model_id=original.model_id,
            model_spec=model_spec,
            validation_specs=(model_spec,),
            **({"family_id": "linear_regression"} | arguments),
        )
        metadata = deepcopy(original.selection_metadata)
        metadata["families"][0]["trial_ledger_semantic_identity"] = reference[
            "semantic_identity"
        ]
        wrong = replace(
            original,
            selection_metadata=metadata,
            trial_ledgers=(reference, original.trial_ledgers[1]),
        )
        logical_root = tmp_path / f"wrong_{index}"
        with pytest.raises(ValueError, match=expected_error):
            with runtime_resource_policy_scope() as policy:
                write_quarterly_model_artifact(
                    logical_root,
                    model_id="model-artifact",
                    envelope=_envelope(),
                    quarterly_results=fixture_quarterly_results(
                        OwnedValue(
                            item,
                            policy.lease(
                                "test.model",
                                fitting._linear_model_state_upper_bound_bytes(
                                    ModelSpec(
                                        model_id=item.model_id,
                                        model_algorithm=item.model_result.metadata[
                                            "model_algorithm"
                                        ],
                                        input_factor_columns=tuple(
                                            item.model_result.metadata["input_columns"]
                                        ),
                                        label_column="label",
                                    )
                                ),
                            ),
                        )
                        for item in (wrong,)
                    ),
                    trial_layout=layout,
                )
        assert not filesystem_exists(
            logical_root / "instances" / _envelope().instance_identity
        )
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("max_train_rows", (None, 4))
def test_linear_workspace_cap_rejects_before_weights_and_solver(
    monkeypatch, max_train_rows
):
    from tests.helpers.runtime_resources import capture_test_policy

    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
        max_train_rows=max_train_rows,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2017-01-02", periods=8),
            "label_available_at": pd.bdate_range("2017-01-02", periods=8),
            "x": np.arange(8.0),
            "label": np.arange(8.0) * 2 + 1,
            "eligible": True,
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=1_000_000, explicit_cap_bytes=200_000
    )
    actual_collect = fitting._collect_training_frame_owned
    blockers = []

    def collect(*args, **kwargs):
        owned = actual_collect(*args, **kwargs)
        assert policy.used_memory_bytes == (
            owned.lease.reserved_bytes
            + fitting._linear_model_state_upper_bound_bytes(spec)
        )
        blockers.append(
            policy.lease("test.blocker", 200_000 - policy.used_memory_bytes - 1)
        )
        return owned

    def forbidden(*args, **kwargs):
        raise AssertionError("linear_allocation_before_admission")

    monkeypatch.setattr(fitting, "_collect_training_frame_owned", collect)
    monkeypatch.setattr(fitting, "equal_timestamp_sample_weights", forbidden)
    monkeypatch.setattr(fitting.np.linalg, "lstsq", forbidden)
    with runtime_resource_policy_scope(policy):
        try:
            with pytest.raises(
                ValueError,
                match="model_training_working_set_exceeds_task_memory_budget",
            ):
                fitting._fit_linear_model_result_owned_from_frames(
                    (frame,), spec, fitting_cutoff="2018-12-31"
                )
            assert policy.used_memory_bytes == blockers[0].reserved_bytes
        finally:
            for blocker in blockers:
                blocker.close()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("boundary", ("soft", "captured", "explicit"))
def test_estimator_admission_uses_capacity_not_soft_thread_planning(
    monkeypatch, boundary
):
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=12 * 1024**2 if boundary == "explicit" else None,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-01-01", periods=1000),
            "label_available_at": pd.date_range("2015-01-01", periods=1000),
            "x": np.arange(1000, dtype="float64"),
            "label": np.arange(1000, dtype="float64") / 10,
            "eligible": True,
        }
    )
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    estimator = build_estimator_from_model_spec(spec)
    original_collect = fitting._collect_training_frame_owned
    original_fit = fitting._fit_predictor
    blockers = []
    fits = []

    def collect(*args, **kwargs):
        owned = original_collect(*args, **kwargs)
        training = owned.value[0]
        required = (
            len(training) * 8
            + fitting._model_state_upper_bound_bytes(
                training, spec, estimator=estimator
            )
            + fitting._estimator_fit_working_set_bytes(training, spec, threads=1)
        )
        limit = (
            policy.memory_budget_bytes
            if boundary == "soft"
            else policy.captured_available_memory_bytes
            if boundary == "captured"
            else policy.explicit_memory_cap_bytes
        )
        blockers.append(
            policy.lease(
                "test.other_work",
                limit - policy.used_memory_bytes - required + required // 4,
            )
        )
        assert policy.used_memory_bytes + required > limit
        return owned

    def fit(*args, **kwargs):
        fits.append(kwargs["threads"])
        return original_fit(*args, **kwargs)

    with runtime_resource_policy_scope(policy):
        baseline = fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2018-12-31",
        )
        try:
            expected = baseline.value.prediction_model.predict(
                frame.loc[:, spec.input_columns]
            )
        finally:
            baseline.release(owner="test.baseline")
        monkeypatch.setattr(fitting, "_collect_training_frame_owned", collect)
        monkeypatch.setattr(fitting, "_fit_predictor", fit)
        try:
            if boundary == "soft":
                fitted = fitting._fit_model_result_owned_from_frames(
                    (frame,), spec, estimator=estimator, fitting_cutoff="2018-12-31"
                )
                try:
                    np.testing.assert_array_equal(
                        fitted.value.prediction_model.predict(
                            frame.loc[:, spec.input_columns]
                        ),
                        expected,
                    )
                finally:
                    fitted.release(owner="test.soft.fit")
                assert fits == [1]
            else:
                with pytest.raises(
                    ValueError,
                    match="model_training_working_set_exceeds_task_memory_budget",
                ):
                    fitting._fit_model_result_owned_from_frames(
                        (frame,), spec, estimator=estimator, fitting_cutoff="2018-12-31"
                    )
                assert fits == []
            assert policy.used_memory_bytes == blockers[0].reserved_bytes
        finally:
            for blocker in blockers:
                blocker.close()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("max_train_rows", (None, 400))
def test_model_disk_admission_precedes_pickle_and_keeps_full_linear_state(
    tmp_path, monkeypatch, max_train_rows
):
    from quant_research_platform.research.models import persisted
    from quant_research_platform.research.models.persisted import (
        read_quarterly_predictor,
    )
    from quant_research_platform.runtime import resources

    layout = StorageLayout(tmp_path)
    template = _quarterly_result(
        1,
        trial_layout=layout,
        model_spec=ModelSpec(
            model_id="linear_regression",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
    )
    spec = ModelSpec(
        model_id="linear_regression",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
        max_train_rows=max_train_rows,
        random_seed=template.model_result.metadata["random_seed"],
    )
    reference = _trial_reference(
        layout,
        _fold(1),
        family_id="linear_regression",
        model_id="linear_regression",
        model_spec=spec,
        protocol=_fixture_protocol((spec,)),
    )
    selection = {
        **template.selection_metadata,
        "families": [
            {
                **template.selection_metadata["families"][0],
                "trial_ledger_semantic_identity": reference["semantic_identity"],
            }
        ],
    }
    template = replace(
        template, selection_metadata=selection, trial_ledgers=(reference,)
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2017-01-02"] * 400 + ["2017-01-03"] * 400)
            + pd.to_timedelta(list(range(400)) * 2, unit="ns"),
            "label_available_at": pd.to_datetime(
                ["2017-01-02"] * 400 + ["2017-01-03"] * 400
            ),
            "x": np.arange(800.0),
            "label": np.arange(800.0) * 2.0 + 1.0,
            "eligible": True,
        }
    )
    logical = tmp_path / "model"
    with runtime_resource_policy_scope() as policy:
        actual_collect = fitting._collect_training_frame_owned
        actual_lstsq = fitting.np.linalg.lstsq
        training_bytes = []
        solver_calls = []

        def collect(*args, **kwargs):
            value = actual_collect(*args, **kwargs)
            training_bytes.append(value.lease.reserved_bytes)
            return value

        def solve(design, labels, **kwargs):
            assert (
                policy.used_memory_bytes
                >= training_bytes[-1] + design.nbytes + labels.nbytes
            )
            solver_calls.append(design.shape[0])
            return actual_lstsq(design, labels, **kwargs)

        monkeypatch.setattr(fitting, "_collect_training_frame_owned", collect)
        monkeypatch.setattr(fitting.np.linalg, "lstsq", solve)

        def members():
            owned = fitting._fit_linear_model_result_owned_from_frames(
                (frame,), spec, fitting_cutoff="2018-12-31"
            )
            try:
                assert (
                    owned.lease.reserved_bytes
                    == fitting._linear_model_state_upper_bound_bytes(spec)
                )
                result = replace(
                    template,
                    model_result=owned.value,
                    trial_ledgers=(
                        reference,
                        fixture_refit_reference(
                            layout, reference, owned.value.metadata
                        ),
                    ),
                )
                owned.value = result
                result = None
                yield owned
            finally:
                if owned.lease.active:
                    owned.release(owner="test.model")

        root = write_quarterly_model_artifact(
            logical,
            model_id="model-artifact",
            envelope=_envelope(specs=(spec,)),
            quarterly_results=fixture_quarterly_results(members()),
            trial_layout=layout,
        )
        assert solver_calls == [800 if max_train_rows is None else max_train_rows]
        artifact = read_quarterly_model_artifact(root, trial_layout=layout)
        with read_quarterly_predictor(
            artifact, "2019Q1", "linear_regression", resource_policy=policy
        ) as predictor:
            np.testing.assert_allclose(
                predictor.predict(pd.DataFrame({"x": [1.0, 2.0]})), [3.0, 5.0]
            )
        old_good = _file_bytes(root)
        previous = set(filesystem_iterdir(root.parent))
        dump = persisted.pickle.dump
        calls = []

        def traced_dump(*args, **kwargs):
            calls.append(True)
            return dump(*args, **kwargs)

        monkeypatch.setattr(persisted.pickle, "dump", traced_dump)
        paths = []

        def no_space(path):
            paths.append(Path(path))
            return SimpleNamespace(free=1)

        monkeypatch.setattr(resources.shutil, "disk_usage", no_space)
        with pytest.raises(RuntimeError, match="insufficient_free_disk"):
            write_quarterly_model_artifact(
                logical,
                model_id="model-artifact",
                envelope=_envelope(software_identity="e" * 64, specs=(spec,)),
                quarterly_results=fixture_quarterly_results(members()),
                trial_layout=layout,
            )
        assert calls == []
        assert paths and all(
            path.is_relative_to(native_filesystem_path(logical)) for path in paths
        )
        assert set(filesystem_iterdir(root.parent)) == previous
        assert _file_bytes(root) == old_good
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("cap", (None, 13))
@pytest.mark.parametrize("chunk_rows", (5, 17))
def test_training_retention_preserves_trailing_rows_and_same_day_order(cap, chunk_rows):
    from tests.helpers.runtime_resources import capture_test_policy

    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
        max_train_rows=cap,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2017-01-02", periods=16).repeat(2),
            "label_available_at": pd.Timestamp("2017-02-01"),
            "asset_id": ["b", "a"] * 16,
            "x": np.arange(32.0),
            "label": np.arange(32.0) * 2 + 1,
            "eligible": True,
        }
    )
    frame.loc[frame.index % 7 == 0, "label_available_at"] = pd.Timestamp("2018-01-01")
    expected = frame.loc[
        frame.index % 7 != 0, ["timestamp", "asset_id", "x", "label"]
    ].copy()
    expected["timestamp"] = expected["timestamp"].dt.as_unit("ns")
    if cap is not None:
        expected = expected.tail(cap)
    policy = capture_test_policy(available_memory_bytes=1024**2)
    with runtime_resource_policy_scope(policy):
        owned = fitting._collect_training_frame_owned(
            (
                frame.iloc[start : start + chunk_rows]
                for start in range(0, 32, chunk_rows)
            ),
            spec,
            fitting_cutoff="2017-12-31",
        )
        try:
            actual, _, _ = owned.value
            pd.testing.assert_frame_equal(
                actual, expected.reset_index(drop=True), check_exact=True
            )
            assert policy.used_memory_bytes == owned.lease.reserved_bytes
        finally:
            owned.lease.close()
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "mutation",
    ("part", "spec", "manifest", "unselected_year", "cancel", "exception", "staged"),
)
def test_model_selection_reuses_snapshot_and_rejects_between_scan_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.datasets.io import (
        _research_dataset_access_window,
    )
    from quant_research_platform.research.models import production
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from quant_research_platform.storage.datasets.writing import write_dataset

    dataset = make_demo_research_dataset_spec(
        data_id="snapshot",
        factor_columns=("x",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("label_available_at",),
    )
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 3,
            "asset_id": ["a"] * 3,
            "timestamp": pd.to_datetime(["2014-01-02", "2018-01-02", "2024-01-02"]),
            "label_available_at": pd.to_datetime(
                ["2014-02-03", "2018-02-02", "2024-02-02"]
            ),
            "x": [1.0, 2.0, 3.0],
            "label": [0.1, 0.2, 0.3],
            "eligible": True,
        }
    )
    frame = frame.astype(
        {"timestamp": "datetime64[ns]", "label_available_at": "datetime64[ns]"}
    )
    layout = StorageLayout(tmp_path)
    root = layout.research_dataset_root(dataset.market_id, dataset.data_id)
    with runtime_resource_policy_scope() as policy:
        write_dataset(
            root, frame, spec=dataset, resource_policy=policy, borrowed_input=True
        )
        with open_dataset_session(
            root,
            columns=None,
            time_window=_research_dataset_access_window(dataset.market_id, None),
            resource_policy=policy,
        ) as session:
            scratch = session.inventory.scratch_path
            baseline = policy.used_memory_bytes
            observed = []

            def exercise(**kwargs):
                assert list(kwargs["sessions"]) == list(frame["timestamp"][:2])

                def read():
                    return kwargs["chunks"](
                        tuple(frame.columns),
                        input_columns=("x",),
                        start=None,
                        end="2018-12-31",
                    )

                first = pd.concat(read(), ignore_index=True)
                pd.testing.assert_frame_equal(
                    first, frame.iloc[:2].reset_index(drop=True)
                )
                observed.append(True)
                if mutation == "staged":
                    item = _quarterly_result(
                        1,
                        trial_layout=layout,
                        model_spec=ModelSpec(
                            model_id="linear_regression",
                            model_algorithm="linear_regression",
                            input_factor_columns=("x",),
                            label_column="label",
                        ),
                    )
                    state_bytes = fitting._linear_model_state_upper_bound_bytes(
                        ModelSpec(
                            model_id="champion",
                            model_algorithm="linear_regression",
                            input_factor_columns=("x",),
                            label_column="label",
                        )
                    )

                    def members(change):
                        owned = OwnedValue(
                            item, policy.lease("test.model", state_bytes)
                        )
                        try:
                            yield owned
                            if change:
                                path = root / "spec.json"
                                filesystem_write_bytes(
                                    path, filesystem_read_bytes(path) + b" "
                                )
                        finally:
                            if owned.lease.active:
                                owned.release(owner="test.model")

                    logical = tmp_path / "models"
                    good = write_quarterly_model_artifact(
                        logical,
                        model_id="model-artifact",
                        envelope=_envelope(),
                        quarterly_results=fixture_quarterly_results(members(False)),
                        trial_layout=layout,
                        verify_source=session.verify,
                    )
                    old_good = _file_bytes(good)
                    previous = set(filesystem_iterdir(good.parent))
                    with pytest.raises(
                        ValueError, match="dataset_spec_identity_changed"
                    ):
                        write_quarterly_model_artifact(
                            logical,
                            model_id="model-artifact",
                            envelope=_envelope(software_identity="e" * 64),
                            quarterly_results=fixture_quarterly_results(members(True)),
                            trial_layout=layout,
                            verify_source=session.verify,
                        )
                    assert set(filesystem_iterdir(good.parent)) == previous
                    assert _file_bytes(good) == old_good
                    return ()
                if mutation in {"cancel", "exception"}:
                    iterator = iter(read())
                    try:
                        next(iterator)
                        if mutation == "exception":
                            raise RuntimeError("consumer_failure")
                    finally:
                        iterator.close()
                    return ()
                if mutation == "part":
                    part = session.read_paths[0]
                    filesystem_write_bytes(part, filesystem_read_bytes(part) + b" ")
                elif mutation in {"spec", "manifest"}:
                    path = (
                        root / "spec.json"
                        if mutation == "spec"
                        else root / "_manifest" / "manifest.json"
                    )
                    filesystem_write_bytes(path, filesystem_read_bytes(path) + b" ")
                else:
                    path = root / "data" / "year=2025" / "part-00000.parquet"
                    filesystem_mkdir(path.parent)
                    filesystem_write_bytes(
                        path, filesystem_read_bytes(session.all_paths[0])
                    )
                return tuple(read())

            monkeypatch.setattr(production, "_build_a_share_quarterly_models", exercise)

            def run():
                return production.build_model_selection_result(
                    layout=layout,
                    dataset_catalog=ResearchDatasetCatalog((dataset,)),
                    model_catalog=research_profile().models,
                    dataset_id=dataset.data_id,
                    model_ids=("linear",),
                    tie_break_order=("linear",),
                    time_window=TimeWindow("2019-01-01", "2019-03-31"),
                    resource_policy=policy,
                    protocol_manifest={
                        **a_share.protocol("pv_all").manifest(),
                        "research_configuration": {
                            "models": [{"model_id": "linear", "hyperparameters": {}}]
                        },
                    },
                    run_id="snapshot",
                    research_protocol_identity="a" * 64,
                    model_target_id="snapshot_model",
                    backend_versions={},
                    dataset_session=session,
                    on_validation_coverage=lambda *_args: pytest.fail(
                        "coverage must not run before snapshot validation"
                    ),
                )

            if mutation in {"cancel", "staged"}:
                assert run() == ()
            elif mutation == "exception":
                with pytest.raises(RuntimeError, match="^consumer_failure$"):
                    run()
            else:
                with pytest.raises(
                    ValueError,
                    match="dataset_(file_identity|spec_identity|manifest_identity|inventory)_changed",
                ):
                    run()
            assert observed == [True]
            assert policy.used_memory_bytes == baseline
        assert not filesystem_exists(scratch)
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "fraction,depth,expected_nodes", ((0.25, 6, 7), (0.0001, 2, 7), (1e-320, 3, 15))
)
def test_rf_state_bound_uses_declared_leaf_weight_without_losing_row_or_depth_limits(
    fraction, depth, expected_nodes
):
    frame = pd.DataFrame({"x": np.arange(128, dtype="float64")})
    spec = ModelSpec(
        model_id="rf",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("x",),
        label_column="label",
        hyperparameters={
            "n_estimators": 3,
            "max_depth": depth,
            "min_weight_fraction_leaf": fraction,
            "bootstrap": False,
        },
        random_seed=42,
    )
    estimator = build_estimator_from_model_spec(spec)
    with runtime_resource_policy_scope() as policy:
        with policy.lease("test.rf_state", 1024**2):
            estimator.fit(
                frame, np.sin(frame.x), sample_weight=np.linspace(1.0, 2.0, len(frame))
            )
            assert all(
                tree.tree_.node_count <= expected_nodes
                for tree in estimator.estimators_
            )
            assert fitting._model_state_upper_bound_bytes(
                frame, spec, estimator=estimator
            ) == 3 * (4096 + expected_nodes * 256)
        assert policy.used_memory_bytes == 0


def test_estimator_without_fit_cannot_be_returned_as_fitted():
    spec = ModelSpec(
        model_id="missing_fit",
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
    with pytest.raises(AttributeError, match="fit"):
        fitting._fit_predictor(
            spec,
            training=frame,
            y=frame.label.to_numpy(),
            estimator=object(),
            threads=1,
        )


@pytest.mark.parametrize("arrow_features", [False, True])
@pytest.mark.parametrize("missing_position", [0, 200, 501, 997, 999])
def test_training_admits_materialized_filter_index_without_changing_rows(
    missing_position,
    arrow_features,
):
    from tests.helpers.runtime_resources import capture_test_policy

    positions = np.arange(1000, dtype="float64")
    features = tuple(f"x{index}" for index in range(6))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="100ms"),
            "label_available_at": pd.Timestamp("2024-01-02"),
            "asset_id": ["asset_b", "asset_a"] * 500,
            "eligible": True,
            "label": positions * 2 + 3,
            **{name: positions + index for index, name in enumerate(features)},
        }
    )
    if arrow_features:
        frame = frame.astype({name: "float64[pyarrow]" for name in features})
    frame.loc[missing_position, "label"] = np.nan
    original = frame.copy(deep=True)
    expected_positions = [
        position for position in range(1000) if position != missing_position
    ]
    spec = ModelSpec(
        model_id="sparse_training",
        model_algorithm="linear_regression",
        input_factor_columns=features,
        label_column="label",
    )
    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    with runtime_resource_policy_scope(policy):
        owned = fitting._collect_training_frame_owned(
            [frame], spec, fitting_cutoff="2024-01-03"
        )
        try:
            actual, counts, _metadata = owned.value
            assert actual.x0.tolist() == expected_positions
            assert actual.label.tolist() == [
                position * 2 + 3 for position in expected_positions
            ]
            assert actual.timestamp.tolist() == [
                frame.timestamp.iloc[position] for position in expected_positions
            ]
            assert counts["rows_after_input_label_dropna"] == 999
            pd.testing.assert_frame_equal(frame, original, check_exact=True)
            assert policy.used_memory_bytes == owned.lease.reserved_bytes
            actual = None
        finally:
            owned.release(owner="test.sparse_training")
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("consumer", ("quarterly", "crypto", "staged"))
@pytest.mark.parametrize(
    "failure", ("open", "load", "close", "missing", "notdir", "corrupt")
)
def test_model_member_io_preserves_failure_class(
    tmp_path, monkeypatch, consumer, failure
):
    import pickle

    from quant_research_platform.research.models import persisted
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_record,
        write_semantic_artifact,
    )

    layout = StorageLayout(tmp_path)
    bound = fitting._linear_model_state_upper_bound_bytes(
        ModelSpec(
            model_id="linear",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        )
    )
    with runtime_resource_policy_scope() as policy:
        item = _quarterly_result(
            1,
            trial_layout=layout,
            model_spec=ModelSpec(
                model_id="linear_regression",
                model_algorithm="linear_regression",
                input_factor_columns=("x",),
                label_column="label",
            ),
        )
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=layout,
            envelope=_envelope(),
            quarterly_results=fixture_quarterly_results(
                (OwnedValue(item, policy.lease("test.model", bound)),)
            ),
        )
        item = None
        artifact = read_quarterly_model_artifact(root, trial_layout=layout)
        metadata = artifact.quarterly_models[0]["model_metadata"]
        if consumer == "crypto":

            def write_crypto_member(stage):
                directory = stage / "models/fold_1"
                filesystem_mkdir(directory, parents=True)
                with persisted.read_quarterly_predictor(
                    artifact, "2019Q1", "linear_regression", resource_policy=policy
                ) as model:
                    persisted._write_model_member(
                        directory / "linear.pkl", model, state_bytes=bound
                    )

            crypto_root = write_semantic_artifact(
                tmp_path / "crypto", _envelope(), write_crypto_member
            )
            identities = {}
            _, members = read_semantic_artifact_record(
                crypto_root, member_identities=identities
            )
            crypto_artifact = SimpleNamespace(
                instance_root=crypto_root,
                member_records=members,
                member_identities=identities,
            )
            entry = dict(fold_id="fold_1", model_id="linear", model_metadata=metadata)
        staged_layout = StorageLayout(tmp_path / "staged-trials")
        staged_item = (
            _quarterly_result(
                1,
                trial_layout=staged_layout,
                model_spec=ModelSpec(
                    model_id="linear_regression",
                    model_algorithm="linear_regression",
                    input_factor_columns=("x",),
                    label_column="label",
                ),
            )
            if consumer == "staged"
            else None
        )
        original_bytes = _file_bytes(root)
        primary = (
            FileNotFoundError("model_missing")
            if failure == "missing"
            else NotADirectoryError("model_not_directory")
            if failure == "notdir"
            else pickle.UnpicklingError("model_corrupt")
            if failure == "corrupt"
            else OSError(5, "model_system_io")
        )
        native_open = persisted.open_filesystem_file
        native_load = persisted.pickle.load
        handles = []

        class CloseFailure:
            def __init__(self, handle):
                self.handle = handle

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def close(self):
                self.handle.close()
                raise primary

        def open_member(path, mode="rb", **kwargs):
            if Path(path).suffix != ".pkl" or mode != "rb":
                return native_open(path, mode, **kwargs)
            if failure in {"open", "missing", "notdir"}:
                raise primary
            handle = native_open(path, mode, **kwargs)
            handles.append(handle)
            return CloseFailure(handle) if failure == "close" else handle

        def load_member(handle):
            if failure in {"load", "corrupt"}:
                raise primary
            return native_load(handle)

        monkeypatch.setattr(persisted, "open_filesystem_file", open_member)
        monkeypatch.setattr(persisted.pickle, "load", load_member)
        invalid = failure in {"missing", "notdir", "corrupt"}
        with pytest.raises(ValueError if invalid else OSError) as caught:
            if consumer == "staged":
                write_quarterly_model_artifact(
                    tmp_path / "staged",
                    model_id="model-artifact",
                    trial_layout=staged_layout,
                    envelope=_envelope(),
                    quarterly_results=fixture_quarterly_results(
                        (OwnedValue(staged_item, policy.lease("test.model", bound)),)
                    ),
                )
            else:
                context = (
                    persisted.read_crypto_model_member(
                        crypto_artifact, entry, resource_policy=policy
                    )
                    if consumer == "crypto"
                    else persisted.read_quarterly_predictor(
                        artifact, "2019Q1", "linear_regression", resource_policy=policy
                    )
                )
                with context:
                    pytest.fail("model I/O failure was swallowed")
        staged_item = None
        if invalid:
            assert str(caught.value) == "model_artifact_model_invalid"
            assert caught.value.__cause__ is primary
        else:
            assert caught.value is primary
        assert all(handle.closed for handle in handles)
        assert policy.used_memory_bytes == 0
        assert _file_bytes(root) == original_bytes
        if consumer == "staged":
            assert not filesystem_exists(
                tmp_path / "staged/instances" / _envelope().instance_identity
            )
            assert not tuple(filesystem_glob(tmp_path / "staged", "**/.a-*"))


@pytest.mark.parametrize(
    "failure_type", (FileNotFoundError, NotADirectoryError, OSError)
)
def test_quarterly_model_final_stat_preserves_system_error(
    tmp_path, monkeypatch, failure_type
):
    from quant_research_platform.research.models import persisted

    layout = StorageLayout(tmp_path)
    with runtime_resource_policy_scope() as policy:
        item = _quarterly_result(
            1,
            trial_layout=layout,
            model_spec=ModelSpec(
                model_id="linear_regression",
                model_algorithm="linear_regression",
                input_factor_columns=("x",),
                label_column="label",
            ),
        )
        bound = fitting._linear_model_state_upper_bound_bytes(
            ModelSpec(
                model_id="linear",
                model_algorithm="linear_regression",
                input_factor_columns=("x",),
                label_column="label",
            )
        )
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            trial_layout=layout,
            envelope=_envelope(),
            quarterly_results=fixture_quarterly_results(
                (OwnedValue(item, policy.lease("test.model", bound)),)
            ),
        )
        item = None
        primary = failure_type("model_stat_failure")
        native_stat = persisted.os.stat
        native_validate = persisted._validate_trial_references
        armed = False

        def validate_then_arm(*args, **kwargs):
            nonlocal armed
            result = native_validate(*args, **kwargs)
            armed = True
            return result

        def fail_stat(path, *args, **kwargs):
            if armed and not isinstance(path, int) and Path(path).suffix == ".pkl":
                raise primary
            return native_stat(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(persisted, "_validate_trial_references", validate_then_arm)
            patch.setattr(persisted.os, "stat", fail_stat)
            missing = failure_type in (FileNotFoundError, NotADirectoryError)
            with pytest.raises(ValueError if missing else OSError) as caught:
                read_quarterly_model_artifact(root, trial_layout=layout)
            assert armed
            if missing:
                assert str(caught.value) == "model_artifact_quarterly_invalid"
                assert caught.value.__cause__ is primary
            else:
                assert caught.value is primary
        assert policy.used_memory_bytes == 0
        assert (
            read_quarterly_model_artifact(root, trial_layout=layout).model_id
            == "model-artifact"
        )


def test_equal_rank_single_undefined_direction_names_required_input():
    from quant_research_platform.research.models.rank_ensemble import (
        EqualWeightRankEnsembleFactory,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"] * 3),
            "valid": [1.0, 2.0, 3.0],
            "constant": [1.0] * 3,
            "label": [1.0, 2.0, 3.0],
        }
    )
    estimator = EqualWeightRankEnsembleFactory(("valid", "constant"), "label")
    with pytest.raises(
        ValueError,
        match="ensemble_direction_undefined:constant:no_valid_training_rankic",
    ) as caught:
        estimator.fit(frame, frame.label.to_numpy())
    from quant_research_platform.workflows.research import _semantic_failure_code

    assert _semantic_failure_code(caught.value) == str(caught.value)
    frame["constant"] = [3.0, 2.0, 1.0]
    fitted = estimator.fit(frame, frame.label.to_numpy())
    assert fitted.columns == ("valid", "constant")
    assert fitted.directions == (1.0, -1.0)


def _small_common_coverage_case(tmp_path, *, fail_refit_scaler=False, monkeypatch=None):
    from functools import partial

    from quant_research_platform.catalog.models import ModelCatalog
    from quant_research_platform.contracts.features import FeatureTransformSpec
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.experiments import TrialRecorder
    from quant_research_platform.research.models.production import (
        _build_a_share_quarterly_models,
    )
    from quant_research_platform.storage.experiments import (
        read_model_trial_ledger_reference,
    )

    dates = pd.to_datetime(
        [
            "2014-01-02",
            "2016-01-04",
            "2017-12-29",
            "2018-01-02",
            "2018-06-01",
            "2018-12-31",
            "2019-01-02",
        ]
    )
    frame = pd.DataFrame(
        [
            {
                "timestamp": date,
                "market_id": "a_share",
                "asset_id": f"asset_{asset}",
                "eligible": True,
                "x": np.nan if asset == 0 else float(asset),
                "y": np.nan if asset == 1 else float(asset * 2),
                "raw": np.nan if asset == 7 else float(asset),
                "feature_f": np.nan
                if asset == 6
                else 0.0
                if asset == 7
                else float(asset),
                "feature_missing_f": 1 if asset >= 6 else 0,
                "feature_observed_f": 0 if asset >= 6 else 1,
                "f": float(asset),
                "label": asset * 0.03 + index * 0.01,
                "label_available_at": date,
            }
            for index, date in enumerate(dates)
            for asset in range(8)
        ]
    )
    specs = (
        ModelSpec(
            model_id="linear_x",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
            max_train_rows=12,
        ),
        ModelSpec(
            model_id="linear_y",
            model_algorithm="linear_regression",
            input_factor_columns=("y",),
            label_column="label",
            max_train_rows=12,
        ),
        ModelSpec(
            model_id="rank_raw",
            model_algorithm="equal_weight_rank_ensemble",
            input_factor_columns=("raw",),
            label_column="label",
            max_train_rows=12,
        ),
    )
    specs = (
        *specs,
        ModelSpec(
            model_id="rank_flag",
            model_algorithm="equal_weight_rank_ensemble",
            input_feature_columns=("feature_missing_f",),
            label_column="label",
            max_train_rows=12,
        ),
    )
    dataset = make_demo_research_dataset_spec(
        data_id="fixture",
        input_data_ids=("processed_demo",),
        base_input_data_id="processed_demo",
        factor_columns=("x", "y", "raw", "f"),
        feature_columns=("feature_f", "feature_missing_f", "feature_observed_f"),
        feature_transforms=(
            FeatureTransformSpec(
                transform_id="canonical",
                source_columns=("f",),
                transform="a_share_canonical",
                output_columns=("feature_f", "feature_missing_f", "feature_observed_f"),
            ),
        ),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=(),
    )
    configuration = {
        "models": [{"model_id": spec.model_id, "hyperparameters": {}} for spec in specs]
    }
    protocol = {"research_configuration": configuration}
    layout = StorageLayout(tmp_path).with_target_roots(
        {"fixture": tmp_path / ("d" * 64)}
    )
    envelope = build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity=canonical_json_hash(protocol),
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-02",
            "end": "2019-01-02",
            "access": "development",
        },
        upstream_identities={"fixture": "d" * 64},
        semantic_metadata={
            "target_contract": {
                "protocol": protocol,
                "model_specs": [model_spec_payload(spec) for spec in specs],
            }
        },
    )

    def chunks(columns, *, input_columns, start, end):
        selected = frame.loc[frame.timestamp.le(pd.Timestamp(end))]
        if start is not None:
            selected = selected.loc[selected.timestamp.ge(pd.Timestamp(start))]
        for start_index in range(0, len(selected), 5):
            yield selected.iloc[start_index : start_index + 5].loc[:, list(columns)]

    attempts = []
    if fail_refit_scaler:
        original_begin = TrialRecorder.begin
        original_fit = fitting.ModelMatrixScaler.fit

        def begin(recorder, *args, **kwargs):
            record = original_begin(recorder, *args, **kwargs)
            attempts.append((recorder, record))
            return record

        def fit(matrix, *args, **kwargs):
            if attempts and attempts[-1][0].plan.stage == "refit":
                raise (
                    KeyboardInterrupt if fail_refit_scaler == "cancel" else RuntimeError
                )("refit_scaler_oracle")
            return original_fit(matrix, *args, **kwargs)

        monkeypatch.setattr(TrialRecorder, "begin", begin)
        monkeypatch.setattr(fitting.ModelMatrixScaler, "fit", fit)
    results = partial(
        _build_a_share_quarterly_models,
        layout=layout,
        dataset_spec=dataset,
        model_catalog=ModelCatalog(specs),
        dataset_id="fixture",
        model_ids=tuple(spec.model_id for spec in specs),
        tie_break_order=tuple(spec.model_id for spec in specs),
        time_window=TimeWindow("2019-01-02", "2019-01-02"),
        chunks=chunks,
        sessions=pd.DatetimeIndex(dates),
        chronology={"validation_months": 12},
        formal_dataset_window={"start": "2014-01-01", "end": "2019-01-02"},
        development_window={"start": "2019-01-01", "end": "2019-01-02"},
        root_seed=42,
        run_id="run_small",
        research_protocol_identity=canonical_json_hash(protocol),
        model_target_id="model-artifact",
        backend_versions={"internal": "test"},
        holdout_window={"start": "2020-01-01", "end": "2020-12-31"},
        research_configuration=configuration,
    )
    with runtime_resource_policy_scope() as policy:
        if fail_refit_scaler:
            with pytest.raises(
                KeyboardInterrupt if fail_refit_scaler == "cancel" else RuntimeError,
                match="refit_scaler_oracle",
            ):
                write_quarterly_model_artifact(
                    tmp_path / "models",
                    model_id="model-artifact",
                    envelope=envelope,
                    quarterly_results=results,
                    trial_layout=layout,
                )
            recorder, record = attempts[-1]
            assert recorder.plan.stage == "refit"
            from quant_research_platform.storage.experiments import read_trial_ledger

            events = read_trial_ledger(recorder.root, recorder.plan)
            assert events[-1].status == (
                "abandoned" if fail_refit_scaler == "cancel" else "failed"
            )
            assert events[-1].metrics == {}
            assert events[-1].undefined_reasons == {
                "failure": "KeyboardInterrupt"
                if fail_refit_scaler == "cancel"
                else "RuntimeError"
            }
            assert not tuple(filesystem_glob(tmp_path / "models", ".*staging*"))
            assert policy.used_memory_bytes == 0
            return
        root = write_quarterly_model_artifact(
            tmp_path / "models",
            model_id="model-artifact",
            envelope=envelope,
            quarterly_results=results,
            trial_layout=layout,
        )
        saved = read_quarterly_model_artifact(
            root, expected_identity=envelope.instance_identity, trial_layout=layout
        )
        coverage = saved.quarterly_selections["2019Q1"]["common_validation_coverage"]
        keys = pd.read_parquet(native_filesystem_path(root / coverage["path"]))
        assert coverage["row_count"] == 15
        assert set(keys.asset_id) == {f"asset_{i}" for i in (2, 3, 4, 5, 7)}
        assert len(saved.trial_ledgers) == 8
        for reference in saved.trial_ledgers:
            plan, records = read_model_trial_ledger_reference(layout, reference)
            if plan.stage == "selection":
                assert records[0].common_coverage_id == coverage["semantic_identity"]
                assert records[0].metrics["validation_rows"] == 15
            else:
                assert dict(records[0].metrics) == {"n_train": 12}
                assert plan.selection_metric is None
                assert records[0].common_coverage_id != coverage["semantic_identity"]
        assert policy.used_memory_bytes == 0
        return saved, layout


def test_common_pk_and_refit_use_real_small_model_write_read_chain(tmp_path):
    _small_common_coverage_case(tmp_path)


def test_refit_scaler_failure_is_recorded_before_any_learning(tmp_path, monkeypatch):
    _small_common_coverage_case(
        tmp_path, fail_refit_scaler=True, monkeypatch=monkeypatch
    )


def test_common_pk_identity_distinguishes_equal_count_and_ignores_chunking():
    from quant_research_platform.research.models.trial_selection import (
        ModelSampleCoverage,
    )

    asset_major = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2019-01-01", "2019-01-02"] * 2),
            "asset_id": ["a", "a", "b", "b"],
        }
    )
    time_major = asset_major.iloc[[0, 2, 1, 3]]
    whole, pieces, changed, duplicate, invalid = (
        ModelSampleCoverage({"scope": "oracle"}) for _ in range(5)
    )
    whole.add(asset_major)
    pieces.add(time_major.iloc[:2])
    pieces.add(time_major.iloc[2:])
    changed.add(asset_major.assign(asset_id=["a", "a", "c", "c"]))
    assert whole.row_count == pieces.row_count == changed.row_count == 4
    assert whole.semantic_identity == pieces.semantic_identity
    assert whole.semantic_identity != changed.semantic_identity
    duplicate.add(asset_major)
    with pytest.raises(ValueError, match="model_coverage_key_order_invalid"):
        duplicate.add(asset_major.iloc[-1:])
    with pytest.raises(ValueError, match="model_coverage_key_invalid"):
        invalid.add(asset_major.assign(asset_id=["a", "a", None, "b"]))


def test_validation_coverage_readback_ignores_physical_key_interleaving(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.research.models.persisted import (
        _validate_validation_coverage,
    )
    from quant_research_platform.research.models.trial_selection import (
        ModelSampleCoverage,
    )

    scope = {
        "validation_start": "2019-01-01",
        "validation_end": "2019-01-02",
    }
    asset_major = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2019-01-01", "2019-01-02"] * 2),
            "asset_id": ["a", "a", "b", "b"],
        }
    )
    time_major = asset_major.iloc[[0, 2, 1, 3]]
    coverage = ModelSampleCoverage(scope)
    coverage.add(asset_major)
    metadata = {
        "path": "models/2019Q1/validation-coverage.parquet",
        "row_count": coverage.row_count,
        "semantic_identity": coverage.semantic_identity,
    }
    for name, physical_rows in (
        ("asset_major", asset_major),
        ("time_major", time_major),
    ):
        root = tmp_path / name
        path = root / metadata["path"]
        filesystem_mkdir(path.parent, parents=True)
        pq.write_table(
            pa.Table.from_pandas(physical_rows, preserve_index=False),
            native_filesystem_path(path),
        )
        _validate_validation_coverage(
            root, "2019Q1", metadata, scope, member_identities=None
        )


def test_refit_scaler_cancellation_is_recorded_before_any_learning(
    tmp_path, monkeypatch
):
    _small_common_coverage_case(
        tmp_path, fail_refit_scaler="cancel", monkeypatch=monkeypatch
    )


def test_reader_binds_formal_model_basis_and_actual_coverage_count(
    tmp_path, monkeypatch
):
    from copy import deepcopy

    from quant_research_platform.research.models import persisted

    saved, layout = _small_common_coverage_case(tmp_path)
    envelope = deepcopy(saved.envelope)
    # The common domain stays unchanged: the next family already requires y.
    # All published model metadata and trial parameters still agree on x alone.
    envelope["semantic_metadata"]["target_contract"]["model_specs"][0][
        "input_factor_columns"
    ] = ["x", "y"]
    with pytest.raises(ValueError, match="model_trial_parameters_misaligned"):
        persisted._validate_trial_references(
            saved.quarterly_models,
            saved.quarterly_selections,
            saved.trial_ledgers,
            layout,
            envelope,
            root=saved.instance_root,
            member_identities=saved.member_identities,
        )
    reduced = deepcopy(saved.quarterly_selections)
    for selection in reduced.values():
        selection["candidate_family_ids"].remove("rank_flag")
        selection["tie_break_order"].remove("rank_flag")
        selection["families"] = [
            item for item in selection["families"] if item["family_id"] != "rank_flag"
        ]
        selection["selected_family_id"] = "linear_x"
        selection["selected_model_id"] = "linear_x"
    references = tuple(
        ref
        for ref in saved.trial_ledgers
        if persisted.read_model_trial_ledger_reference(layout, ref)[
            0
        ].candidate_family_id
        != "rank_flag"
    )
    with pytest.raises(ValueError, match="model_trial_family_coverage_mismatch"):
        persisted._validate_trial_references(
            tuple(
                entry
                for entry in saved.quarterly_models
                if entry["family_id"] != "rank_flag"
            ),
            reduced,
            references,
            layout,
            saved.envelope,
            root=saved.instance_root,
            member_identities=saved.member_identities,
        )
    read = persisted.read_model_trial_ledger_reference

    def altered(layout, reference):
        plan, records = read(layout, reference)
        if plan.stage == "selection":
            records = tuple(
                replace(record, metrics={**record.metrics, "validation_rows": 16})
                for record in records
            )
        return plan, records

    monkeypatch.setattr(persisted, "read_model_trial_ledger_reference", altered)
    with pytest.raises(ValueError, match="model_trial_common_coverage_mismatch"):
        read_quarterly_model_artifact(saved.instance_root, trial_layout=layout)

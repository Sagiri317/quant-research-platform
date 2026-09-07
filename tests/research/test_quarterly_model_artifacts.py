from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import replace
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
    ModelSpec,
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
from tests.research.workflow_test_support import (
    _ColumnPredictor,
    _fold,
    _metadata,
    _quarterly_result,
    _selection_metadata,
    _trial_reference,
    _UnpicklablePredictor,
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
    from quant_research_platform.storage.semantic_artifacts import (
        MAX_SEMANTIC_ARTIFACT_MEMBERS,
    )

    protocol = a_share.protocol("fusion_nf_expectations", factor_scope="full")
    specs = tuple(
        replace(spec, label_column="label")
        for spec in a_share.model_specs(protocol.scenario_id, factor_scope="full")
    )
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
            protocol=protocol.manifest(),
        )
        for spec in specs
    )
    selection = _selection_metadata(
        family_id=specs[0].model_id, model_id=specs[0].model_id, columns=inputs
    )
    selection["candidate_family_ids"] = selection["tie_break_order"] = [
        spec.model_id for spec in specs
    ]
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
                "protocol": protocol.manifest(),
                "model_specs": [{"model_id": spec.model_id} for spec in specs],
            }
        },
    )

    def chunks(columns, **_kwargs):
        yield frame.loc[:, list(columns)]

    def fitted():
        for spec, reference in zip(specs, references, strict=True):
            result, lease = _fit_selected_model_result(
                dataset_spec=dataset,
                selected_spec=spec,
                chunks=chunks,
                fitting_cutoff="2018-12-31",
                quarter_id="2019Q1",
                root_seed=42,
                config_index=0,
            )
            item = QuarterlyModelResult(
                _fold(1), spec.model_id, result, spec.model_id, selection, (reference,)
            )
            owned = OwnedValue(item, lease) if lease is not None else item
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
            quarterly_results=fitted(),
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
        assert shard.stat().st_size < 1024**2
        assert any(
            item["model_metadata"]["scaler"] is not None for item in payload["models"]
        )
        assert policy.used_memory_bytes == 0
    development = len(
        pd.period_range(a_share.DEVELOPMENT_START, a_share.DEVELOPMENT_END, freq="Q")
    )
    total = len(
        pd.period_range(a_share.DEVELOPMENT_START, a_share.HOLDOUT_END, freq="Q")
    )
    assert (development, total) == (20, 28)
    assert (1 + development * 9, 1 + total * 9) == (181, 253)
    assert 1 + total * 9 <= MAX_SEMANTIC_ARTIFACT_MEMBERS == 256


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
        upstream_identities={"fixture": "d" * 64},
        semantic_metadata={
            "target_contract": {
                "protocol": {"fixture": True},
                "model_specs": [
                    {"model_id": name}
                    for name in (
                        "linear_regression",
                        "equal_weight_rank_ensemble",
                        "ic_weighted_rank_ensemble",
                    )
                ],
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
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        trial_layout=StorageLayout(tmp_path),
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(
                1,
                trial_layout=StorageLayout(tmp_path),
                model_id="champion_q1",
                columns=("x",),
                predictor=_ColumnPredictor("x", 1.0),
            ),
            _quarterly_result(
                2,
                trial_layout=StorageLayout(tmp_path),
                model_id="champion_q2",
                columns=("y",),
                predictor=_ColumnPredictor("y", 2.0),
            ),
        ),
    )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
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
        trial_layout=StorageLayout(tmp_path),
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(
                1,
                trial_layout=StorageLayout(tmp_path),
                model_id="champion",
                columns=("x",),
            ),
        ),
    )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
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
        trial_layout=StorageLayout(tmp_path),
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(
                1,
                trial_layout=StorageLayout(tmp_path),
                model_id="champion",
                columns=("x",),
            ),
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
        trial_layout=StorageLayout(tmp_path),
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(
                2,
                trial_layout=StorageLayout(tmp_path),
                model_id="champion",
                columns=("x",),
            ),
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
    root = write_quarterly_model_artifact(
        tmp_path / "model",
        model_id="model-artifact",
        trial_layout=StorageLayout(tmp_path),
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(
                1,
                trial_layout=StorageLayout(tmp_path),
                model_id="same_champion",
                columns=("x",),
            ),
            _quarterly_result(
                2,
                trial_layout=StorageLayout(tmp_path),
                model_id="same_champion",
                columns=("x",),
            ),
        ),
    )
    artifact = read_quarterly_model_artifact(root, trial_layout=StorageLayout(tmp_path))
    assert [item["model_id"] for item in artifact.quarterly_models] == [
        "same_champion",
        "same_champion",
    ]

    ic_result = _quarterly_result(
        1,
        trial_layout=StorageLayout(tmp_path),
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
        trial_layout=StorageLayout(tmp_path),
        envelope=_envelope(),
        quarterly_results=(ic_result,),
    )
    ic_artifact = read_quarterly_model_artifact(
        ic_root, trial_layout=StorageLayout(tmp_path)
    )
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
        trial_layout=StorageLayout(tmp_path),
        envelope=_envelope(),
        quarterly_results=(
            _quarterly_result(
                1,
                trial_layout=StorageLayout(tmp_path),
                model_id="champion",
                columns=("x",),
            ),
        ),
    )
    (root / "models" / "2019Q1" / "linear_regression.pkl").write_bytes(b"mutated")
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
        trial_layout=StorageLayout(tmp_path),
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
                trial_layout=StorageLayout(tmp_path),
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
        chunks=lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2018-01-01"]),
                    "asset_id": ["a"],
                    "x": [1.0],
                    "label": [2.0],
                    "eligible": [True],
                }
            ),
        ),
        fitting_cutoff="2018-12-31",
        quarter_id="2019Q1",
        root_seed=42,
        config_index=0,
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


def test_selected_protocol_strength_is_resolved_on_real_refit_rows_and_persisted(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from quant_research_platform.catalog.models import ModelCatalog
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.research.models.persisted import (
        read_quarterly_predictor,
    )
    from quant_research_platform.research.models.production import (
        QuarterlyModelResult,
        _select_model_family_winners,
    )
    from quant_research_platform.research.models.supervised import preflight_model_specs
    from quant_research_platform.storage.experiments import read_trial_ledger_reference
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
    protocol = {"fixture": True}
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
                "protocol": protocol,
                "model_specs": [{"model_id": "ridge"}],
            }
        },
    )
    with runtime_resource_policy_scope() as policy:
        winners, metadata, refs, indices = _select_model_family_winners(
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
        )
        selected, index = winners["ridge"], indices["ridge"]
        assert (
            index == 3 and selected.hyperparameters["alpha"] == 30.0
        )  # 3 * lambda 10.
        result, lease = _fit_selected_model_result(
            dataset_spec=dataset,
            selected_spec=selected,
            chunks=chunks,
            fitting_cutoff="2018-12-31",
            quarter_id="2019Q1",
            root_seed=42,
            config_index=index,
        )
        assert result.prediction_model.estimator.alpha == 60.0  # 6 * same lambda 10.
        quarterly = QuarterlyModelResult(
            _fold(1), selected.model_id, result, "ridge", metadata, refs
        )
        root = write_quarterly_model_artifact(
            tmp_path / "model",
            model_id="model-artifact",
            envelope=envelope,
            quarterly_results=(
                OwnedValue(quarterly, lease) if lease is not None else quarterly,
            ),
            trial_layout=layout,
        )
        artifact = read_quarterly_model_artifact(root, trial_layout=layout)
        saved = artifact.quarterly_models[0]["model_metadata"]
        assert saved["rows_after_input_label_dropna"] == 6
        assert saved["resolved_hyperparameters"]["alpha"] == 60.0
        plan, trials = read_trial_ledger_reference(layout, artifact.trial_ledgers[0])
        assert plan.candidate_ids[index] == selected.model_id
        assert (
            next(
                record for record in trials if record.candidate_id == selected.model_id
            ).normalized_parameters["hyperparameters"]["alpha"]
            == 30.0
        )
        assert all(record.validation_metrics["n_train"] == 3 for record in trials)
        with read_quarterly_predictor(
            artifact, "2019Q1", "ridge", resource_policy=policy
        ) as predictor:
            assert predictor.estimator.alpha == 60.0
            probe = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
            before = predictor.predict(probe).copy()
        frame.loc[frame.timestamp.ge("2019-01-01"), ["x", "label"]] *= 1000000.0
        later_winners, later_selection, later_refs, later_indices = (
            _select_model_family_winners(
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
            )
        )
        assert later_indices == indices
        assert later_selection == metadata
        assert later_refs != refs
        later, later_lease = _fit_selected_model_result(
            dataset_spec=dataset,
            selected_spec=later_winners["ridge"],
            chunks=chunks,
            fitting_cutoff="2018-12-31",
            quarter_id="2019Q1",
            root_seed=42,
            config_index=later_indices["ridge"],
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
    original = _quarterly_result(
        1, trial_layout=layout, model_id="champion", columns=("x",)
    )
    for index, (fold, arguments) in enumerate(
        (
            (_fold(1), {"dataset_id": "other", "data_identity": "e" * 64}),
            (_fold(1), {"target_id": "other_model"}),
            (_fold(1), {"protocol": {"different_protocol": True}}),
            (_fold(2), {}),
            (_fold(1), {"family_id": "other_family"}),
        )
    ):
        reference = _trial_reference(
            layout,
            fold,
            model_id="champion",
            **({"family_id": "linear_regression"} | arguments),
        )
        metadata = deepcopy(original.selection_metadata)
        metadata["families"][0]["trial_ledger_semantic_identity"] = reference[
            "semantic_identity"
        ]
        wrong = replace(
            original,
            selection_metadata=metadata,
            trial_ledgers=(reference,),
        )
        logical_root = tmp_path / f"wrong_{index}"
        with pytest.raises(ValueError, match="model_trial_"):
            write_quarterly_model_artifact(
                logical_root,
                model_id="model-artifact",
                envelope=_envelope(),
                quarterly_results=(wrong,),
                trial_layout=layout,
            )
        assert not (logical_root / "instances" / _envelope().instance_identity).exists()

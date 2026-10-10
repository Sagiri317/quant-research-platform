from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.models import fitting, persisted
from quant_research_platform.research.models.fitting import (
    ModelFitResult,
    training_scope_chunks,
)
from quant_research_platform.research.models.persisted import (
    read_quarterly_model_artifact,
    read_quarterly_predictor,
)
from quant_research_platform.research.models.production import (
    _fit_selected_model_result,
    _refit_model_spec,
)
from quant_research_platform.research.models.rank_ensemble import RankEnsemblePredictor
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.models.trial_selection import (
    select_best_validation_trial,
)
from quant_research_platform.research.predictions import runner
from quant_research_platform.research.protocols import a_share
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_walk,
    filesystem_write_bytes,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_record,
    write_semantic_artifact,
)
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.prediction_output_helpers import write_prediction_result
from tests.research.workflow_test_support import (
    _column_predictor,
    _metadata,
    shared_model_case,
    shared_model_protocol,
    shared_model_rows,
)


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): filesystem_read_bytes(parent / name)
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


@pytest.mark.parametrize("with_candidates", (False, True))
def test_bound_missing_hyperparameters_reach_native_model_artifact(
    tmp_path, with_candidates
):
    from quant_research_platform.profiles.research import bind_research_configuration
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    model = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "linear_regression"
    )
    model.pop("hyperparameters")
    model.pop("parameter_candidates", None)
    model.update(
        input_factor_columns=[config["evaluation"]["factors"][0]],
        input_feature_columns=[],
    )
    if with_candidates:
        model["parameter_candidates"] = [{}]
    config.update(
        models=[model],
        portfolios=[],
        primary_portfolio_id=None,
        comparison_model_id=model["model_id"],
    )
    bound = bind_research_configuration(research_profile(config["profile_id"]), config)
    normalized = bound.protocol(config["scenario_id"]).manifest()[
        "research_configuration"
    ]
    assert normalized["models"][0]["hyperparameters"] == {}
    specs = tuple(bound.models.list())
    protocol = shared_model_protocol(specs, configurations=normalized["models"])
    case = shared_model_case(tmp_path, specs=specs, protocol=protocol)
    selection = case.artifact.selection("global", None, access_window="development")
    assert selection["selected_family_id"] == model["model_id"]
    predictor = read_quarterly_predictor(case.artifact, "global", model["model_id"])
    metadata = next(
        entry["model_metadata"]
        for entry in case.artifact.final_models
        if entry["selection_group"] == "global"
    )
    fitting.validate_model_state(predictor, metadata)


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
            "model_target_available_at": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2019-01-03", "2018-01-04"]
            ),
        }
    )
    result = fitting._fit_model_result(
        fitting._collect_training_frame((frame,), spec, fitting_cutoff="2018-12-31"),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    model_result = result
    assert model_result.metadata["rows_input"] == 4
    assert model_result.metadata["rows_after_filter_mode"] == 3
    assert model_result.metadata["rows_after_input_label_dropna"] == 2


def test_training_asset_scope_preserves_consumer_primary_on_early_close() -> None:
    frame = pd.DataFrame({"asset_id": ["a", "b"], "value": [1.0, 2.0]})
    primary = RuntimeError("consumer_primary")
    scoped = training_scope_chunks((frame,), asset_ids=("b",))
    next(scoped)
    with pytest.raises(RuntimeError) as raised:
        scoped.throw(primary)
    assert raised.value is primary


def test_rank_missing_contributes_neutral_without_renormalization() -> None:
    predictor = RankEnsemblePredictor(columns=("x", "y"), directions=(1.0, 1.0))
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
        {"trial_id": ["defined", "undefined"], "validation_rankic": [-0.4, np.nan]}
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
    assert (trial_q1, trial_q2, refit_q1) == (2404282703, 28539577, 2825241916)
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

    def fit_model(_frames, spec, **_kwargs):
        captured.append(spec.random_seed)
        return ModelFitResult(
            metadata=_metadata(("x",)), prediction_model=_column_predictor("x")
        )

    monkeypatch.setattr(fitting, "_fit_model_result", fit_model)
    _fit_selected_model_result(
        prepared_training=fitting._collect_training_frame(
            [
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(["2018-01-01"]),
                        "asset_id": ["a"],
                        "x": [1.0],
                        "label": [2.0],
                        "eligible": [True],
                        "label_available_at": pd.to_datetime(["2018-01-01"]),
                    }
                )
            ],
            base,
            fitting_cutoff="2018-12-31",
        ),
        selected_spec=base,
        candidate_parameters=dict(base.hyperparameters),
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
        "pv_nf_expectations_evaluation",
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
            prepared_training=fitting._collect_training_frame(
                [frame.assign(eligible=True, label_available_at=frame.timestamp)],
                spec,
                fitting_cutoff="2018-01-02",
            ),
            estimator=WarningEstimator(),
            threads=1,
        )


@pytest.mark.parametrize("cap", (None, 13))
@pytest.mark.parametrize("chunk_rows", (5, 17))
def test_training_retention_preserves_trailing_rows_and_same_day_order(cap, chunk_rows):
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
    prepared = fitting._collect_training_frame(
        (frame.iloc[start : start + chunk_rows] for start in range(0, 32, chunk_rows)),
        spec,
        fitting_cutoff="2017-12-31",
    )
    pd.testing.assert_frame_equal(
        prepared.training, expected.reset_index(drop=True), check_exact=True
    )


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
            prepared_training=fitting._collect_training_frame(
                [frame.assign(eligible=True, label_available_at=frame.timestamp)],
                spec,
                fitting_cutoff="2018-01-02",
            ),
            estimator=object(),
            threads=1,
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


def _prediction_result(
    case, *, group="global", start="2019-01-01", end="2019-06-30", access="development"
):
    return runner.build_prediction_result_from_persisted_model(
        layout=case.layout,
        dataset_catalog=ResearchDatasetCatalog((case.dataset,)),
        prediction_spec=PredictionSpec(
            prediction_id=f"prediction_{group}",
            model_id="model-artifact",
            research_dataset_id=case.dataset.data_id,
            train_end=None,
            test_start=start,
            test_end=end,
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        ),
        persisted_model=case.artifact,
        selection_group=group,
        access_window=access,
    )


def _republish(case, logical, *, mutate, operational=None):
    if not operational:
        operational = {}
        read_semantic_artifact_record(case.root, operational_metadata=operational)

    def write(stage):
        for member in case.artifact.member_records:
            relative = member["path"]
            destination = stage / relative
            filesystem_mkdir(destination.parent, parents=True, exist_ok=True)
            payload = filesystem_read_bytes(case.root / relative)
            if relative.endswith(".json"):
                document = json.loads(payload)
                mutate(relative, document)
                payload = json.dumps(document).encode()
            filesystem_write_bytes(destination, payload)

    return write_semantic_artifact(
        logical,
        case.envelope,
        write,
        operational_metadata=operational,
    )


def test_champion_family_is_rejected_before_a_share_production_reads(
    tmp_path, monkeypatch
):
    from quant_research_platform.research.models import production

    def forbidden(*args, **kwargs):
        pytest.fail("reserved family reached model chronology")

    monkeypatch.setattr(production, "model_folds", forbidden)
    spec = ModelSpec(
        model_id="champion",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    with pytest.raises(ValueError, match="^a_share_model_id_reserved$"):
        shared_model_case(tmp_path, specs=(spec,))
    assert not tuple(filesystem_glob(tmp_path / "models", "**/model.json"))


def test_model_reader_rejects_champion_family_in_a_bound_contract(tmp_path):
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.json_codec import canonical_json_hash

    case = shared_model_case(tmp_path / "good", development_end="2019-03-31")
    old_good = _file_bytes(case.root)
    payload = case.envelope.payload()
    contract = payload["semantic_metadata"]["target_contract"]
    contract["model_specs"][0]["model_id"] = "champion"
    contract["protocol"]["research_configuration"]["models"][0]["model_id"] = "champion"
    case.envelope = build_business_manifest_envelope(
        **{
            key: value
            for key, value in payload.items()
            if key not in {"instance_identity", "status", "protocol_or_spec_identity"}
        },
        protocol_or_spec_identity=canonical_json_hash(contract["protocol"]),
    )
    root = _republish(
        case, tmp_path / "conflict", mutate=lambda _relative, _payload: None
    )
    with pytest.raises(ValueError, match="^a_share_model_id_reserved$"):
        read_quarterly_model_artifact(root, trial_layout=case.layout)
    assert _file_bytes(case.root) == old_good


def test_prediction_writer_and_reader_reject_champion_family_lineage(tmp_path):
    from quant_research_platform.artifacts.outputs import read_prediction_output_bundle

    case = shared_model_case(tmp_path, development_end="2019-03-31")
    bundle = write_prediction_result(
        case.layout, _prediction_result(case, end="2019-03-31")
    )
    old_good = _file_bytes(bundle)
    conflict = _prediction_result(case, end="2019-03-31")
    for model in conflict.model_metadata["quarterly_models"]:
        model["family_id"] = "champion"
    conflict.model_metadata["champion_by_quarter"] = {
        quarter: "champion"
        for quarter in conflict.model_metadata["champion_by_quarter"]
    }
    with pytest.raises(ValueError, match="^a_share_model_id_reserved$"):
        write_prediction_result(case.layout, conflict)
    assert _file_bytes(bundle) == old_good
    path = bundle / "manifest.json"
    manifest = json.loads(filesystem_read_bytes(path))
    manifest["metadata"]["model_metadata"] = conflict.model_metadata
    filesystem_write_bytes(path, json.dumps(manifest).encode())
    with pytest.raises(
        ValueError, match="^output_manifest_metadata_invalid$"
    ) as failure:
        read_prediction_output_bundle(bundle, expected_output_id="prediction_global")
    assert str(failure.value.__cause__) == "a_share_model_id_reserved"


@pytest.mark.parametrize("group", ["global", "quarterly"])
def test_champion_selector_and_named_families_read_their_real_predictions(
    tmp_path, group
):
    from quant_research_platform.artifacts.outputs import read_prediction_output_bundle

    specs = tuple(
        ModelSpec(
            model_id=name,
            model_algorithm="linear_regression",
            input_factor_columns=(column,),
            label_column="label",
        )
        for name, column in (("named_model", "x"), ("winner", "y"))
    )
    frame = shared_model_rows(specs, development_end="2019-03-31")
    frame["x"] = np.sin(np.arange(len(frame)))
    case = shared_model_case(
        tmp_path, specs=specs, frame=frame, development_end="2019-03-31"
    )
    assert (
        case.artifact.selection(group, "2019Q1", access_window="development")[
            "selected_family_id"
        ]
        == "winner"
    )
    bundle = write_prediction_result(
        case.layout, _prediction_result(case, group=group, end="2019-03-31")
    )
    query = frame.loc[frame.timestamp.ge("2019-01-01"), ["timestamp", "asset_id"]]
    scores = {}
    for selector in ("champion", "winner", "named_model"):
        _, lookup = read_prediction_output_bundle(
            bundle, expected_output_id=f"prediction_{group}", family_selection=selector
        )
        scores[selector], found = lookup.values_for_frame(query, mark_matched=True)
        assert all(found)
        assert lookup.row_count == len(query)
        assert lookup.coverage_counts() == (0, 0)
    np.testing.assert_array_equal(scores["champion"], scores["winner"])
    assert not np.array_equal(scores["named_model"], scores["winner"])


@pytest.mark.parametrize(
    "second_family",
    ["alpha__config_0", "renamed", "winner", "champion__config_0", None],
)
def test_validation_members_are_scoped_by_family(tmp_path, second_family):
    specs = [
        ModelSpec(
            model_id="alpha",
            model_algorithm="ridge_regression",
            input_factor_columns=("x",),
            label_column="label",
        )
    ]
    configurations = [
        {
            "model_id": "alpha",
            "hyperparameters": {},
            "parameter_candidates": [
                {"regularization_strength": 0.01},
                {"regularization_strength": 0.1},
            ],
        }
    ]
    if second_family is not None:
        specs.append(
            ModelSpec(
                model_id=second_family,
                model_algorithm="linear_regression",
                input_factor_columns=("y",),
                label_column="label",
            )
        )
        configurations.append({"model_id": second_family, "hyperparameters": {}})
    specs = tuple(specs)
    preflight_model_specs(specs)
    case = shared_model_case(
        tmp_path,
        specs=specs,
        protocol=shared_model_protocol(
            specs, development_end="2019-03-31", configurations=configurations
        ),
    )
    candidates = case.artifact.candidate_models
    assert len(candidates) == 5 * (2 + (second_family is not None))
    paths = [entry["validation_prediction"]["path"] for entry in candidates]
    assert len(set(paths)) == len(candidates)
    for entry in candidates:
        member = entry["validation_prediction"]
        assert (
            member["path"] == f"models/{entry['quarter_id']}/{entry['family_id']}/"
            f"{entry['model_id']}.parquet"
        )
        assert filesystem_exists(case.root / member["path"])
    for quarter in {entry["quarter_id"] for entry in candidates}:
        quarter_entries = [e for e in candidates if e["quarter_id"] == quarter]
        assert {
            e["model_id"] for e in quarter_entries if e["family_id"] == "alpha"
        } == {"alpha__config_0", "alpha__config_1"}
        if second_family is not None:
            assert [
                e["model_id"]
                for e in quarter_entries
                if e["family_id"] == second_family
            ] == [second_family]
    assert len(case.artifact.final_models) == 2 * len(specs)
    for group in ("global", "quarterly"):
        for spec in specs:
            predictor = read_quarterly_predictor(case.artifact, group, spec.model_id)
            predictions = predictor.predict(case.frame.loc[:, list(spec.input_columns)])
            assert len(predictions) == len(case.frame)
            assert np.isfinite(predictions).all()


@pytest.mark.parametrize("shared_inputs", (False, True))
def test_candidates_fit_once_and_both_groups_have_real_final_models(
    tmp_path, monkeypatch, shared_inputs
):
    specs = (
        ModelSpec(
            model_id="linear_x",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
        ModelSpec(
            model_id="ridge_y",
            model_algorithm="ridge_regression",
            input_factor_columns=("x",) if shared_inputs else ("y",),
            label_column="label",
        ),
    )
    configurations = [
        {"model_id": specs[0].model_id, "hyperparameters": {}},
        {
            "model_id": specs[1].model_id,
            "hyperparameters": {},
            "parameter_candidates": [
                {"regularization_strength": 0.01},
                {"regularization_strength": 0.1},
            ],
        },
    ]
    calls, prepared = [], []
    counts = {"weights": 0, "scaler": 0}
    original_weights = fitting.equal_timestamp_sample_weights
    original_scaler = fitting.ModelMatrixScaler.fit

    def weights(*args, **kwargs):
        counts["weights"] += 1
        return original_weights(*args, **kwargs)

    def scaler(*args, **kwargs):
        counts["scaler"] += 1
        return original_scaler(*args, **kwargs)

    monkeypatch.setattr(fitting, "equal_timestamp_sample_weights", weights)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "fit", scaler)
    original = fitting._fit_model_result

    def fit(training, spec, **kwargs):
        calls.append(spec.model_id)
        prepared.append(training)
        before = training.training.copy(deep=True)
        result = original(training, spec, **kwargs)
        pd.testing.assert_frame_equal(training.training, before, check_exact=True)
        return result

    monkeypatch.setattr(fitting, "_fit_model_result", fit)
    preflight_model_specs(specs)
    case = shared_model_case(
        tmp_path,
        specs=specs,
        protocol=shared_model_protocol(specs, configurations=configurations),
    )
    assert len(calls) == 6 * 3 + 2
    assert len({id(item) for item in prepared}) == 7 * (1 if shared_inputs else 2)
    assert counts == {
        "weights": 7 * (1 if shared_inputs else 2),
        "scaler": 7 * (1 if shared_inputs else 2),
    }
    assert len(case.artifact.candidate_models) == 18
    assert len(case.artifact.final_models) == 4
    assert set(case.artifact.selections["global"]) == {"final"}
    assert set(case.artifact.selections["quarterly"]) == {"2019Q1", "2019Q2", "final"}
    assert {entry["quarter_id"] for entry in case.artifact.candidate_models} == {
        "2018Q1",
        "2018Q2",
        "2018Q3",
        "2018Q4",
        "2019Q1",
        "2019Q2",
    }
    for family in specs:
        entries = [
            entry
            for entry in case.artifact.final_models
            if entry["family_id"] == family.model_id
        ]
        assert (
            entries[0]["fit_windows"]
            == entries[1]["fit_windows"]
            == [{"start": "2014-01-01", "end": "2019-06-30"}]
        )
        assert entries[0]["model_metadata"] == entries[1]["model_metadata"]
        assert entries[0]["model_member"] == entries[1]["model_member"]
        assert (
            entries[0]["refit_trial_semantic_identity"]
            == entries[1]["refit_trial_semantic_identity"]
        )
        g = read_quarterly_predictor(case.artifact, "global", family.model_id)
        q = read_quarterly_predictor(case.artifact, "quarterly", family.model_id)
        np.testing.assert_array_equal(
            g.predict(case.frame.loc[:, list(family.input_columns)]),
            q.predict(case.frame.loc[:, list(family.input_columns)]),
        )
    for reference in case.artifact.trial_ledgers:
        plan, records = read_model_trial_ledger_reference(case.layout, reference)
        if plan.stage == "refit":
            assert plan.folds == ()
            assert plan.selection_groups == ("global", "quarterly")
            assert plan.selection_metric is None
            assert set(records[0].metrics) == {"n_train"}


def test_candidate_own_prediction_rows_survive_shared_scoring_domain(tmp_path):
    specs = tuple(
        ModelSpec(
            model_id=f"linear_{column}",
            model_algorithm="linear_regression",
            input_factor_columns=(column,),
            label_column="label",
        )
        for column in ("x", "y")
    )
    frame = shared_model_rows(specs)
    frame.loc[frame.asset_id.eq("asset_0"), "y"] = np.nan
    case = shared_model_case(tmp_path, specs=specs, frame=frame)
    for family, count in (("linear_x", 8), ("linear_y", 7)):
        entry = next(
            e
            for e in case.artifact.candidate_models
            if e["quarter_id"] == "2019Q1" and e["family_id"] == family
        )
        saved = pd.concat(case.artifact.validation_chunks(entry), ignore_index=True)
        assert saved.groupby("timestamp").size().tolist() == [count, count]
        assert saved.groupby("timestamp").score_eligible.sum().tolist() == [7, 7]
    for group in ("global", "quarterly"):
        result = _prediction_result(case, group=group)
        bundle = write_prediction_result(case.layout, result)
        prediction = pd.read_parquet(bundle / "prediction.parquet")
        assert (
            prediction.groupby(["timestamp", "family_id"])
            .size()
            .unstack()["linear_x"]
            .tolist()
            == [8] * 4
        )
        assert (
            prediction.groupby(["timestamp", "family_id"])
            .size()
            .unstack()["linear_y"]
            .tolist()
            == [7] * 4
        )
    for reference in case.artifact.trial_ledgers:
        plan, records = read_model_trial_ledger_reference(case.layout, reference)
        if plan.stage == "selection":
            assert records[0].metrics["validation_rows"] == 7


def test_common_pk_readback_is_independent_of_chunking(tmp_path):
    first = shared_model_case(tmp_path / "single", chunk_size=1)
    second = shared_model_case(tmp_path / "whole", chunk_size=1024)
    for group in ("global", "quarterly"):
        for key, decision in first.artifact.selections[group].items():
            other = second.artifact.selections[group][key]
            assert (
                decision["common_validation_coverage"]
                == other["common_validation_coverage"]
            )
            assert decision["families"] == other["families"]


def test_future_validation_changes_do_not_change_prior_quarter_selection(tmp_path):
    specs = (
        ModelSpec(
            model_id="positive",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
        ModelSpec(
            model_id="negative",
            model_algorithm="linear_regression",
            input_factor_columns=("y",),
            label_column="label",
        ),
    )
    original = shared_model_rows(specs)
    original["y"] = np.where(
        original.timestamp.lt("2019-01-01"), original.x, -original.x
    )
    changed = original.copy()
    changed.loc[changed.timestamp.ge("2019-04-01"), "label"] *= -13.0
    before = shared_model_case(tmp_path / "before", specs=specs, frame=original)
    after = shared_model_case(tmp_path / "after", specs=specs, frame=changed)
    for quarter in ("2019Q1", "2019Q2"):
        assert (
            before.artifact.selections["quarterly"][quarter]
            == after.artifact.selections["quarterly"][quarter]
        )
    assert (
        before.artifact.selections["global"]["final"]
        != after.artifact.selections["global"]["final"]
    )


@pytest.mark.parametrize("group", ["global", "quarterly"])
def test_prediction_request_clips_saved_rows_without_retraining(
    tmp_path, monkeypatch, group
):
    case = shared_model_case(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("development_prediction_refit")

    monkeypatch.setattr(fitting, "_fit_model_result", forbidden)
    result = _prediction_result(case, group=group, start="2019-01-03", end="2019-04-02")
    bundle = write_prediction_result(case.layout, result)
    frame = pd.read_parquet(bundle / "prediction.parquet")
    assert frame.timestamp.min() >= pd.Timestamp("2019-01-03")
    assert frame.timestamp.max() <= pd.Timestamp("2019-04-02")
    assert frame.model_id.eq("linear_regression").all()


def test_prediction_rejects_uncovered_quarter(tmp_path):
    case = shared_model_case(tmp_path)
    with pytest.raises(PermissionError, match="a_share_prediction_window_invalid"):
        _prediction_result(case, start="2018-10-01")


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate_seed",
        "candidate_columns",
        "candidate_rows",
        "candidate_trial",
        "quarter_coordinate",
        "quarter_matrix",
        "selection_winner",
        "selection_cutoff",
        "selection_coverage",
        "final_group",
        "final_cutoff",
        "final_seed",
        "final_fit_window",
        "final_trial",
        "unknown_member",
    ],
)
def test_reader_rejects_self_consistent_publication_drift(tmp_path, mutation):
    case = shared_model_case(tmp_path / "good")
    old_good = _file_bytes(case.root)

    def change(relative, payload):
        if relative == "models/2019Q1/metadata.json":
            entry = payload["candidate_models"][0]
            if mutation == "candidate_seed":
                entry["model_metadata"]["random_seed"] += 1
            elif mutation == "candidate_columns":
                entry["model_metadata"]["input_columns"] = ["other"]
            elif mutation == "candidate_rows":
                entry["model_metadata"]["rows_after_input_label_dropna"] -= 1
            elif mutation == "candidate_trial":
                entry["trial_semantic_identity"] = "e" * 64
            elif mutation == "quarter_coordinate":
                entry["quarter_id"] = "2019Q2"
            elif mutation == "quarter_matrix":
                payload["candidate_models"] = []
        if relative == "selections/global/final.json":
            decision = payload["selection"]
            if mutation == "selection_winner":
                decision["selected_model_id"] = "other"
            elif mutation == "selection_score":
                decision["families"][0]["trial_metrics"][0]["validation_rankic"] = -0.75
            elif mutation == "selection_cutoff":
                decision["cutoff"] = "2024-01-01"
            elif mutation == "selection_coverage":
                decision["common_validation_coverage"]["row_count"] += 1
        if relative == "models/final/global/metadata.json":
            entry = payload["final_models"][0]
            if mutation == "final_group":
                entry["selection_group"] = "quarterly"
            elif mutation == "final_cutoff":
                entry["fit_cutoff"] = "2024-01-01"
            elif mutation == "final_seed":
                entry["model_metadata"]["random_seed"] += 1
            elif mutation == "final_fit_window":
                entry["fit_windows"][0]["start"] = "2015-01-01"
            elif mutation == "final_trial":
                entry["refit_trial_semantic_identity"] = "e" * 64
        if relative == "model.json" and mutation == "unknown_member":
            payload["extra"] = True

    root = _republish(case, tmp_path / "corrupt", mutate=change)
    with pytest.raises(ValueError):
        read_quarterly_model_artifact(root, trial_layout=case.layout)
    assert _file_bytes(case.root) == old_good


def test_model_member_mutation_is_rejected_before_loading_predictor(
    tmp_path, monkeypatch
):
    case = shared_model_case(tmp_path)
    path = case.root / case.artifact.final_models[0]["model_member"]
    filesystem_write_bytes(path, filesystem_read_bytes(path) + b" ")
    monkeypatch.setattr(
        persisted.pickle, "load", lambda *_: pytest.fail("load before identity")
    )
    with pytest.raises(ValueError, match="model_artifact_model_invalid"):
        read_quarterly_predictor(case.artifact, "global", "linear_regression")
        pytest.fail("mutated member accepted")


@pytest.mark.parametrize("consumer", ["staged", "persisted", "crypto"])
@pytest.mark.parametrize(
    "failure", ["open", "load", "close", "missing", "notdir", "corrupt"]
)
def test_model_member_io_preserves_failure_class(
    tmp_path, monkeypatch, consumer, failure
):
    case = shared_model_case(tmp_path / "good")
    original_bytes = _file_bytes(case.root)
    metadata = case.artifact.final_models[0]["model_metadata"]
    if consumer == "crypto":

        def write(stage):
            target = stage / "models/final/linear.pkl"
            filesystem_mkdir(target.parent, parents=True)
            filesystem_write_bytes(
                target,
                filesystem_read_bytes(
                    case.root / case.artifact.final_models[0]["model_member"]
                ),
            )

        crypto_root = write_semantic_artifact(tmp_path / "crypto", case.envelope, write)
        from quant_research_platform.storage.semantic_artifacts import (
            read_semantic_artifact_record,
        )

        identities = {}
        _, members = read_semantic_artifact_record(
            crypto_root, member_identities=identities
        )
        crypto = SimpleNamespace(
            instance_root=crypto_root,
            member_records=members,
            member_identities=identities,
        )
        entry = {"model_id": "linear", "model_metadata": metadata}
    primary = (
        FileNotFoundError("model_missing")
        if failure == "missing"
        else NotADirectoryError("model_not_directory")
        if failure == "notdir"
        else pickle.UnpicklingError("model_corrupt")
        if failure == "corrupt"
        else OSError(5, "model_system_io")
    )
    actual_open, actual_load = (persisted.open_filesystem_file, persisted.pickle.load)
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
            return actual_open(path, mode, **kwargs)
        if failure in {"open", "missing", "notdir"}:
            raise primary
        handle = actual_open(path, mode, **kwargs)
        handles.append(handle)
        return CloseFailure(handle) if failure == "close" else handle

    def load_member(handle):
        if failure in {"load", "corrupt"}:
            raise primary
        return actual_load(handle)

    monkeypatch.setattr(persisted, "open_filesystem_file", open_member)
    monkeypatch.setattr(persisted.pickle, "load", load_member)
    invalid = failure in {"missing", "notdir", "corrupt"}
    with pytest.raises(ValueError if invalid else OSError) as caught:
        if consumer == "staged":
            shared_model_case(tmp_path / "staged")
        else:
            context = (
                persisted.read_crypto_model_member(crypto, entry)
                if consumer == "crypto"
                else read_quarterly_predictor(
                    case.artifact, "global", "linear_regression"
                )
            )
            with context:
                pytest.fail("model I/O failure swallowed")
    assert caught.value.__cause__ is primary if invalid else caught.value is primary
    assert all(handle.closed for handle in handles)
    assert _file_bytes(case.root) == original_bytes
    assert not tuple(filesystem_glob(tmp_path / "staged", "**/.a-*"))


@pytest.mark.parametrize("cancellation", [False, True])
def test_refit_scaler_failure_is_recorded_before_learning(
    tmp_path, monkeypatch, cancellation
):
    from quant_research_platform.research.experiments import TrialRecorder
    from quant_research_platform.storage.experiments import read_trial_ledger

    attempts = []
    original_begin, original_fit = (TrialRecorder.begin, fitting.ModelMatrixScaler.fit)

    def begin(recorder, *args, **kwargs):
        record = original_begin(recorder, *args, **kwargs)
        attempts.append((recorder, record))
        return record

    primary = (
        KeyboardInterrupt("refit_scaler_oracle")
        if cancellation
        else RuntimeError("refit_scaler_oracle")
    )

    def fit(*args, **kwargs):
        if attempts and attempts[-1][0].plan.stage == "refit":
            raise primary
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(TrialRecorder, "begin", begin)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "fit", fit)
    with pytest.raises(type(primary)) as caught:
        shared_model_case(tmp_path)
    assert caught.value is primary
    recorder, _ = attempts[-1]
    _, events = read_trial_ledger(recorder.root, recorder.plan)
    assert events[-1].status == ("abandoned" if cancellation else "failed")
    assert events[-1].metrics == {}
    assert events[-1].undefined_reasons == {"failure": type(primary).__name__}
    assert not tuple(filesystem_glob(tmp_path / "models", "**/.a-*"))


def test_failed_source_verification_preserves_published_model_and_releases_state(
    tmp_path,
):
    good = shared_model_case(tmp_path / "good")
    before = _file_bytes(good.root)
    primary = RuntimeError("source_identity_changed")

    def verify():
        raise primary

    with pytest.raises(RuntimeError) as caught:
        shared_model_case(tmp_path / "failed", verify_source=verify)
    assert caught.value is primary
    assert _file_bytes(good.root) == before
    assert not tuple(filesystem_glob(tmp_path / "failed/models", "**/.a-*"))


@pytest.mark.parametrize("max_train_rows", [None, 12])
def test_final_refit_strength_and_state_follow_actual_rows(tmp_path, max_train_rows):
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        max_train_rows=max_train_rows,
    )
    configuration = [
        {
            "model_id": "ridge",
            "hyperparameters": {},
            "parameter_candidates": [{"regularization_strength": 0.1}],
        }
    ]
    preflight_model_specs((spec,))
    case = shared_model_case(
        tmp_path,
        specs=(spec,),
        protocol=shared_model_protocol((spec,), configurations=configuration),
    )
    for entry in case.artifact.final_models:
        count = entry["model_metadata"]["rows_after_input_label_dropna"]
        assert count == (len(case.frame) if max_train_rows is None else max_train_rows)
        assert entry["model_metadata"]["hyperparameters"]["alpha"] == pytest.approx(
            count * 0.1
        )
        predictor = read_quarterly_predictor(
            case.artifact, entry["selection_group"], "ridge"
        )
        assert np.isfinite(
            predictor.predict(case.frame.loc[:, list(spec.input_columns)])
        ).all()


def test_pickle_io_failure_preserves_old_good(tmp_path, monkeypatch):
    case = shared_model_case(tmp_path / "good")
    before = _file_bytes(case.root)
    calls = []

    def fail_dump(*_args, **_kwargs):
        calls.append(True)
        raise OSError("model_write_failed")

    monkeypatch.setattr(persisted.pickle, "dump", fail_dump)
    with pytest.raises(OSError, match="model_write_failed"):
        shared_model_case(tmp_path / "failed")
    assert calls == [True]
    assert _file_bytes(case.root) == before


@pytest.mark.parametrize(
    "failure_type", [FileNotFoundError, NotADirectoryError, OSError]
)
def test_final_model_post_load_stat_preserves_system_error(
    tmp_path, monkeypatch, failure_type
):
    case = shared_model_case(tmp_path)
    primary = failure_type("model_stat_failure")
    actual_load, actual_stat = (persisted.pickle.load, persisted.os.stat)
    armed = False

    def load(handle):
        nonlocal armed
        value = actual_load(handle)
        armed = True
        return value

    def stat(path, *args, **kwargs):
        if armed and (not isinstance(path, int)) and (Path(path).suffix == ".pkl"):
            raise primary
        return actual_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(persisted.pickle, "load", load)
        patch.setattr(persisted.os, "stat", stat)
        with pytest.raises(
            ValueError if failure_type is not OSError else OSError
        ) as caught:
            read_quarterly_predictor(case.artifact, "global", "linear_regression")
        assert armed
        if failure_type is OSError:
            assert caught.value is primary
        else:
            assert caught.value.__cause__ is primary


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
    write_dataset(root, frame, spec=dataset)
    session = open_dataset_session(
        root,
        columns=None,
        time_window=_research_dataset_access_window(dataset.market_id, None),
    )
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
        pd.testing.assert_frame_equal(first, frame.iloc[:2].reset_index(drop=True))
        observed.append(True)
        if mutation == "staged":
            good = shared_model_case(
                tmp_path / "good-model", verify_source=session.verify
            )
            old_good = _file_bytes(good.root)

            def mutate_after_fit(actual):

                def results(**callbacks):
                    yield from actual(**callbacks)
                    path = root / "spec.json"
                    filesystem_write_bytes(path, filesystem_read_bytes(path) + b" ")

                return results

            with pytest.raises(ValueError, match="dataset_spec_identity_changed"):
                shared_model_case(
                    tmp_path / "failed-model",
                    verify_source=session.verify,
                    model_results_wrapper=mutate_after_fit,
                )
            assert _file_bytes(good.root) == old_good
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
            filesystem_write_bytes(path, filesystem_read_bytes(session.all_paths[0]))
        return tuple(read())

    monkeypatch.setattr(production, "_build_a_share_models", exercise)

    def run():
        return production.build_model_selection_result(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((dataset,)),
            model_catalog=research_profile().models,
            dataset_id=dataset.data_id,
            model_ids=("linear",),
            time_window=TimeWindow("2019-01-01", "2019-03-31"),
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
            on_validation=lambda *_args: pytest.fail("unexpected_validation"),
            on_candidate=lambda *_args: pytest.fail("unexpected_candidate"),
            on_trial=lambda *_args: pytest.fail("unexpected_trial"),
            select_models=lambda *_args: pytest.fail("unexpected_selection"),
            on_observation=lambda *_args: pytest.fail("unexpected_observation"),
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


@pytest.mark.parametrize(
    "mutation", ["dataset", "target", "protocol", "fold", "family"]
)
def test_publication_rejects_valid_ledger_from_wrong_artifact_scope(tmp_path, mutation):
    from dataclasses import replace
    from uuid import uuid4

    from quant_research_platform.contracts.experiments import ExperimentFold
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.experiments import TrialRecorder

    bad_layout = StorageLayout(tmp_path / "bad")
    substituted = []

    def wrap(builder):

        def run(**callbacks):
            on_candidate, on_trial = (callbacks["on_candidate"], callbacks["on_trial"])
            metadata = {}

            def candidate(fold, family, spec, result):
                metadata.clear()
                metadata.update(result.metadata)
                on_candidate(fold, family, spec, result)

            def trial(fold, family, reference):
                if substituted:
                    on_trial(fold, family, reference)
                    return
                plan, records = read_model_trial_ledger_reference(bad_layout, reference)
                record = records[0]
                changes = {"experiment_id": "experiment_" + uuid4().hex}
                if mutation == "dataset":
                    changes.update(
                        research_dataset_id="other_dataset",
                        data_snapshot_id="e" * 64,
                        input_lineage=("e" * 64,),
                    )
                elif mutation == "target":
                    changes["target_id"] = "other_model"
                elif mutation == "protocol":
                    changes["research_protocol_identity"] = "e" * 64
                elif mutation == "family":
                    changes["candidate_family_id"] = "other_family"
                else:
                    changes["folds"] = (
                        ExperimentFold(
                            fold_id="2018-q2",
                            train=fold.train,
                            validation=TimeWindow(
                                pd.Timestamp("2018-04-02"), pd.Timestamp("2018-04-03")
                            ),
                        ),
                    )
                wrong = replace(plan, **changes)
                payload = record.payload()["normalized_parameters"]
                for key in ("input_factor_columns", "input_feature_columns"):
                    payload[key] = tuple(payload[key])
                actual_spec = ModelSpec(**payload)
                recorder = TrialRecorder(bad_layout, "run_wrong_scope", wrong)
                attempt = recorder.begin(
                    actual_spec, common_coverage_id=record.common_coverage_id
                )
                recorder.finish(
                    attempt, metrics=dict(record.metrics), model_metadata=metadata
                )
                invalid_reference = recorder.reference()
                restored, completed = read_model_trial_ledger_reference(
                    bad_layout, invalid_reference
                )
                assert restored == wrong
                assert completed[0].status == "completed"
                substituted.append(invalid_reference)
                on_trial(fold, family, invalid_reference)

            callbacks.update(on_candidate=candidate, on_trial=trial)
            yield from builder(**callbacks)

        return run

    good = shared_model_case(tmp_path / "good")
    old_good = _file_bytes(good.root)
    expected = (
        "model_trial_fold_binding_invalid"
        if mutation == "fold"
        else "model_trial_artifact_binding_invalid"
    )
    with pytest.raises(ValueError, match=expected):
        shared_model_case(tmp_path / "bad", model_results_wrapper=wrap)
    assert len(substituted) == 1
    assert _file_bytes(good.root) == old_good
    assert not tuple(filesystem_glob(tmp_path / "bad/models", "**/model.json"))


def test_full_scope_family_metadata_preserves_shards_and_native_model_state(
    tmp_path,
):
    from dataclasses import replace

    from quant_research_platform.research.features.transforms import (
        model_input_validity_columns,
    )
    from quant_research_platform.research.models import supervised
    from tests.helpers.research_configuration import research_configuration

    configuration = research_configuration(
        name="fusion_nf_expectations", small_models=True
    )
    specs = tuple(
        ModelSpec(
            model_id=item["model_id"],
            model_algorithm=item["model_algorithm"],
            input_factor_columns=tuple(item["input_factor_columns"]),
            input_feature_columns=tuple(item["input_feature_columns"]),
            label_column=item["label_column"],
            hyperparameters=item.get("hyperparameters", {}),
        )
        for item in configuration["models"]
    )
    preflight_model_specs(specs)
    # The first mature fold needs enough rows for fractional RF bootstrapping.
    frame = shared_model_rows(specs, development_end="2019-03-31", assets=12)
    for column in frame:
        if column.startswith("feature_missing_"):
            frame[column] = 0.0
    validity = tuple(
        dict.fromkeys(
            column
            for spec in specs
            for column in model_input_validity_columns(
                spec.input_columns, spec.model_algorithm
            )
            if column not in frame.columns
        )
    )
    for column in validity:
        frame[column] = True
    dataset = a_share.research_dataset_catalog().get("fusion_nf_expectations")
    case = shared_model_case(
        tmp_path,
        specs=specs,
        frame=frame,
        dataset=dataset,
        development_end="2019-03-31",
    )
    assert len(case.artifact.candidate_models) == 5 * len(specs)
    assert len(case.artifact.final_models) == 2 * len(specs)
    shard = json.loads(filesystem_read_bytes(case.root / "models/2019Q1/metadata.json"))
    assert len(shard["candidate_models"]) == len(specs)
    assert any(
        item["model_metadata"]["scaler"] is not None
        for item in shard["candidate_models"]
    )
    for entry in case.artifact.final_models:
        metadata = entry["model_metadata"]
        algorithm = metadata["model_algorithm"]
        actual = read_quarterly_predictor(
            case.artifact, entry["selection_group"], entry["family_id"]
        )
        if algorithm in {"equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"}:
            changed = (
                replace(actual, directions=tuple(-value for value in actual.directions))
                if algorithm == "equal_weight_rank_ensemble"
                else replace(actual, weights=tuple(-value for value in actual.weights))
            )
            with pytest.raises(ValueError, match="model_state_metadata_mismatch"):
                fitting.validate_model_state(changed, metadata)
        elif algorithm == "random_forest_regressor":
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    supervised,
                    "model_cpu_capacity",
                    lambda: pytest.fail(
                        "persisted model validation rediscovered host capacity"
                    ),
                )
                fitting.validate_model_state(actual, metadata)
            assert actual.n_jobs == metadata["resolved_hyperparameters"]["n_jobs"]


@pytest.mark.parametrize(
    "field,value",
    (
        (
            "dependency_status",
            {"status": "available", "package": "sklearn", "version": "test"},
        ),
        ("rows_input", 1),
        ("filter_columns_used", []),
    ),
)
def test_quarterly_model_core_rejects_contract_drift(tmp_path, field, value):
    case = shared_model_case(tmp_path / "good")
    before = _file_bytes(case.root)

    def mutate(relative, payload):
        if relative == "models/2019Q1/metadata.json":
            payload["candidate_models"][0]["model_metadata"][field] = value

    root = _republish(case, tmp_path / "bad", mutate=mutate)
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        read_quarterly_model_artifact(root, trial_layout=case.layout)
    assert _file_bytes(case.root) == before


@pytest.mark.parametrize("half_life", (None, 756))
def test_overridden_base_parameters_preserve_native_model_science(tmp_path, half_life):
    from copy import deepcopy

    from quant_research_platform.contracts.experiments import (
        trial_ledger_semantic_payload,
    )
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.profiles.research import bind_research_configuration
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    model = next(
        item
        for item in config["models"]
        if item["model_algorithm"] == "ridge_regression"
    )
    model.update(
        input_factor_columns=[config["evaluation"]["factors"][0]],
        input_feature_columns=[],
        hyperparameters={"alpha": 10.0},
        training_weighting={"method": "equal_timestamp", "half_life": half_life},
        parameter_candidates=[
            {"regularization_strength": 0.01},
            {"regularization_strength": 0.1},
        ],
    )
    config.update(
        models=[model],
        portfolios=[],
        primary_portfolio_id=None,
        comparison_model_id=model["model_id"],
    )
    changed = deepcopy(config)
    changed["models"][0]["hyperparameters"] = {
        "alpha": -1.0,
        "random_state": 999,
        "unused_native_option": True,
    }
    protocols, cases = [], []
    for index, submitted in enumerate((config, changed)):
        bound = bind_research_configuration(research_profile(), submitted)
        protocol = bound.protocol(submitted["scenario_id"])
        projected = protocol.target_manifest(
            protocol.model_selection_target_id, "MODEL"
        )
        protocols.append(protocol)
        specs = tuple(bound.models.list())
        assert dict(specs[0].hyperparameters) == {}
        case = shared_model_case(
            tmp_path / str(index),
            specs=specs,
            protocol=shared_model_protocol(
                specs, configurations=projected["research_configuration"]["models"]
            ),
        )
        for entry in case.artifact.final_models:
            metadata = entry["model_metadata"]
            assert metadata["training_weighting"]["half_life"] == half_life
            predictor = read_quarterly_predictor(
                case.artifact, entry["selection_group"], model["model_id"]
            )
            fitting.validate_model_state(predictor, metadata)
            assert metadata["hyperparameters"]["alpha"] == pytest.approx(
                metadata["rows_after_input_label_dropna"]
                * model["parameter_candidates"][0]["regularization_strength"]
            )
        cases.append(case)
    assert protocols[0].manifest() != protocols[1].manifest()
    assert (
        protocols[1].configuration()["models"][0]["hyperparameters"]
        == changed["models"][0]["hyperparameters"]
    )
    assert protocols[0].target_manifest(
        protocols[0].model_selection_target_id, "MODEL"
    ) == protocols[1].target_manifest(protocols[1].model_selection_target_id, "MODEL")
    assert cases[0].envelope.instance_identity == cases[1].envelope.instance_identity
    assert cases[0].artifact.member_records == cases[1].artifact.member_records
    assert cases[0].artifact.candidate_models == cases[1].artifact.candidate_models
    assert cases[0].artifact.final_models == cases[1].artifact.final_models
    ledgers = [
        [
            trial_ledger_semantic_payload(
                *read_model_trial_ledger_reference(case.layout, reference)
            )
            for reference in case.artifact.trial_ledgers
        ]
        for case in cases
    ]
    assert ledgers[0] == ledgers[1]
    changed["models"][0]["parameter_candidates"][0]["regularization_strength"] = 0.02
    actual_change = bind_research_configuration(research_profile(), changed).protocol(
        changed["scenario_id"]
    )
    assert canonical_json_hash(
        actual_change.target_manifest(actual_change.model_selection_target_id, "MODEL")
    ) != canonical_json_hash(
        protocols[0].target_manifest(protocols[0].model_selection_target_id, "MODEL")
    )


def test_model_reader_uses_published_science_but_publisher_rechecks(
    tmp_path, monkeypatch
):
    case = shared_model_case(tmp_path / "good")

    def change(relative, payload):
        if relative == "selections/global/final.json":
            payload["selection"]["families"][0]["trial_metrics"][0][
                "validation_rankic"
            ] = -0.75

    root = _republish(case, tmp_path / "rewritten", mutate=change)
    original_load = persisted.pickle.load
    original_score = persisted.evaluate_model_trial_metrics_from_chunks

    def forbidden(*args, **kwargs):
        pytest.fail("published metadata read recomputed science or loaded a predictor")

    monkeypatch.setattr(persisted.pickle, "load", forbidden)
    monkeypatch.setattr(
        persisted, "evaluate_model_trial_metrics_from_chunks", forbidden
    )
    artifact = read_quarterly_model_artifact(root, trial_layout=case.layout)
    assert (
        artifact.selections["global"]["final"]["families"][0]["trial_metrics"][0][
            "validation_rankic"
        ]
        == -0.75
    )
    monkeypatch.setattr(persisted.pickle, "load", original_load)
    monkeypatch.setattr(
        persisted, "evaluate_model_trial_metrics_from_chunks", original_score
    )
    with pytest.raises(ValueError, match="model_selection_decision_mismatch"):
        persisted._validate_a_share_model_payload(
            root,
            artifact.envelope,
            artifact.member_records,
            list(artifact.trial_ledgers),
            case.layout,
            verify_science=True,
            operational_metadata=artifact.operational_metadata,
            directory_identities=artifact.directory_identities,
        )
    predictor = read_quarterly_predictor(artifact, "global", "linear_regression")
    assert np.isfinite(predictor.predict(case.frame[["x"]])).all()


def test_distinct_final_choices_have_distinct_real_refits(tmp_path, monkeypatch):
    spec = ModelSpec(
        model_id="ridge_x",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    protocol = shared_model_protocol(
        (spec,),
        configurations=[
            {
                "model_id": spec.model_id,
                "hyperparameters": {},
                "parameter_candidates": [
                    {"regularization_strength": 0.01},
                    {"regularization_strength": 0.1},
                ],
            }
        ],
    )
    original_select = persisted._select_a_share_models

    def select(*args, **kwargs):
        result = original_select(*args, **kwargs)
        # Exercise refit identity independently of the statistical winner rule.
        if kwargs["decision_id"] == "final":
            family = result["families"][0]
            choice = family["config_tie_break_order"][kwargs["group"] == "quarterly"]
            family["selected_config_id"] = result["selected_model_id"] = choice
        return result

    calls = []
    original_fit = fitting._fit_model_result

    def fit(training, spec, **kwargs):
        calls.append(spec.model_id)
        return original_fit(training, spec, **kwargs)

    monkeypatch.setattr(persisted, "_select_a_share_models", select)
    monkeypatch.setattr(fitting, "_fit_model_result", fit)
    case = shared_model_case(tmp_path, specs=(spec,), protocol=protocol)
    assert len(calls) == 6 * 2 + 2
    left, right = case.artifact.final_models
    assert left["model_id"] != right["model_id"]
    assert left["model_member"] != right["model_member"]
    assert (
        left["refit_trial_semantic_identity"] != right["refit_trial_semantic_identity"]
    )
    groups = []
    for reference in case.artifact.trial_ledgers:
        plan, records = read_model_trial_ledger_reference(case.layout, reference)
        if plan.stage == "refit":
            groups.append(plan.selection_groups)
            assert len(records) == 1 and records[0].status == "completed"
    assert groups == [("global",), ("quarterly",)]


def test_shared_preprocessing_matches_independent_candidate_states(monkeypatch):
    specs = tuple(
        ModelSpec(
            model_id=f"ridge_{index}",
            model_algorithm="ridge_regression",
            input_factor_columns=("x",),
            label_column="label",
            hyperparameters={"alpha": alpha},
            random_seed=37,
        )
        for index, alpha in enumerate((0.01, 0.1, 1.0, 10.0))
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=12).repeat(3),
            "x": np.arange(36.0),
            "label": np.sin(np.arange(36.0)),
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-13"),
        }
    )
    expected = [
        fitting._fit_model_result(
            fitting._collect_training_frame([frame], spec, fitting_cutoff="2020-01-13"),
            spec,
            estimator=build_estimator_from_model_spec(spec),
        )
        for spec in specs
    ]
    counts = {"weights": 0, "scaler": 0, "transform": 0}
    originals = (
        fitting.equal_timestamp_sample_weights,
        fitting.ModelMatrixScaler.fit,
        fitting.ModelMatrixScaler.transform,
    )

    def weights(*args, **kwargs):
        counts["weights"] += 1
        return originals[0](*args, **kwargs)

    def scaler(*args, **kwargs):
        counts["scaler"] += 1
        return originals[1](*args, **kwargs)

    def transform(*args, **kwargs):
        counts["transform"] += 1
        return originals[2](*args, **kwargs)

    monkeypatch.setattr(fitting, "equal_timestamp_sample_weights", weights)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "fit", scaler)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "transform", transform)
    prepared = fitting._collect_training_frame(
        [frame], specs[0], fitting_cutoff="2020-01-13"
    )
    actual = [
        fitting._fit_model_result(
            prepared, spec, estimator=build_estimator_from_model_spec(spec)
        )
        for spec in specs
    ]
    assert counts == {"weights": 1, "scaler": 1, "transform": 1}
    for reference, result in zip(expected, actual, strict=True):
        assert result.metadata == reference.metadata
        assert pickle.dumps(result.prediction_model) == pickle.dumps(
            reference.prediction_model
        )
        restored = pickle.loads(pickle.dumps(result.prediction_model))
        fitting.validate_model_state(restored, result.metadata)
        np.testing.assert_array_equal(
            restored.predict(frame[["x"]]),
            reference.prediction_model.predict(frame[["x"]]),
        )


@pytest.mark.parametrize(
    "algorithm", ("ridge_regression", "equal_weight_rank_ensemble")
)
def test_estimator_mutation_cannot_change_shared_preprocessing(algorithm):
    spec = ModelSpec(
        model_id="isolated",
        model_algorithm=algorithm,
        input_factor_columns=("x",),
        label_column="label",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "x": [1.0, 2.0, 3.0],
            "label": [3.0, 1.0, 2.0],
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-02"),
        }
    )
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2020-01-02"
    )
    original = pickle.dumps(prepared.preprocessing)
    training = prepared.training.copy(deep=True)

    class MutatingEstimator:
        def fit(self, matrix, target, *, sample_weight):
            matrix.loc[:, "x"] = 999.0
            target[:] = 999.0
            sample_weight[:] = 999.0
            raise RuntimeError("mutated_estimator_inputs")

    with pytest.raises(RuntimeError, match="mutated_estimator_inputs"):
        fitting._fit_predictor(
            spec, prepared_training=prepared, estimator=MutatingEstimator(), threads=1
        )
    assert pickle.dumps(prepared.preprocessing) == original
    pd.testing.assert_frame_equal(prepared.training, training, check_exact=True)


def test_training_raw_validation_runs_once_with_duplicate_dataframe_index(monkeypatch):
    from quant_research_platform.research.models.production import _prepare_training

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b"],
            "x": [1.0, 2.0],
            "label": [3.0, 4.0],
            "eligible": [False, True],
            "label_available_at": pd.Timestamp("2020-01-02"),
        },
        index=[7, 7],
    )
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    calls = []
    original = fitting.legal_training_mask

    def validate(raw, **kwargs):
        calls.append(len(raw))
        return original(raw, **kwargs)

    monkeypatch.setattr(fitting, "legal_training_mask", validate)
    prepared = _prepare_training(
        SimpleNamespace(primary_key=("timestamp", "asset_id")),
        spec,
        chunks=lambda *args, **kwargs: iter([frame]),
        train_end="2020-01-02",
    )
    assert calls == [2]
    assert prepared.training.asset_id.tolist() == ["b"]
    assert prepared.row_counts["rows_input"] == 2
    assert prepared.row_counts["rows_after_filter_mode"] == 1


@pytest.mark.parametrize(
    "column,value",
    (
        ("x", np.inf),
        ("label", np.inf),
        ("eligible", None),
        ("label_available_at", "not-a-date"),
    ),
)
def test_filtered_raw_training_rows_still_reject_invalid_values(column, value):
    from quant_research_platform.research.models.production import _prepare_training

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b"],
            "x": [1.0, 2.0],
            "label": [3.0, 4.0],
            "eligible": [True, False],
            "label_available_at": pd.Timestamp("2020-01-02"),
        }
    )
    if column in {"eligible", "label_available_at"}:
        frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    with pytest.raises(ValueError):
        _prepare_training(
            SimpleNamespace(primary_key=("timestamp", "asset_id")),
            spec,
            chunks=lambda *args, **kwargs: iter([frame]),
            train_end="2020-01-02",
        )


def test_partial_quarter_native_model_binds_validation_and_maturity_to_request(
    tmp_path,
):
    specs = (
        ModelSpec(
            model_id="linear_regression",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
    )
    rows = shared_model_rows(specs)
    later = rows.loc[rows.timestamp.eq("2019-04-03")].copy()
    later["timestamp"] = pd.Timestamp("2019-04-04")
    availability = next(column for column in rows if "available_at" in column)
    later[availability] = pd.Timestamp("2019-04-05")
    rows = pd.concat([rows, later], ignore_index=True).sort_values(
        ["timestamp", "asset_id"]
    )
    changed = rows.copy()
    changed.loc[changed.timestamp.gt("2019-04-03"), "x"] *= -100
    changed.loc[changed[availability].gt("2019-04-03"), "label"] += 100
    before = shared_model_case(
        tmp_path / "before",
        specs=specs,
        frame=rows,
        requested_end="2019-04-03",
        chunk_size=5,
    )
    after = shared_model_case(
        tmp_path / "after",
        specs=specs,
        frame=changed,
        requested_end="2019-04-03",
        chunk_size=1024,
    )
    assert not before.artifact.final_models
    assert not before.artifact.selections["global"]
    for first, second in zip(
        before.artifact.candidate_models, after.artifact.candidate_models, strict=True
    ):
        assert first == second
        facts = pd.concat(before.artifact.validation_chunks(first), ignore_index=True)
        other = pd.concat(after.artifact.validation_chunks(second), ignore_index=True)
        pd.testing.assert_frame_equal(facts, other, check_exact=True)
        assert facts.timestamp.le("2019-04-03").all()
        assert facts.label_available_at.dropna().le("2019-04-03").all()
        assert pd.Timestamp(first["validation_window"]["end"]) <= pd.Timestamp(
            "2019-04-03"
        )
    last = before.artifact.candidate_models[-1]
    assert last["validation_window"]["end"] == "2019-04-03"
    last_rows = pd.concat(before.artifact.validation_chunks(last))
    assert last_rows.loc[last_rows.timestamp.eq("2019-04-03"), "label"].isna().all()
    for reference in before.artifact.trial_ledgers:
        plan, records = read_model_trial_ledger_reference(before.layout, reference)
        assert plan.stage == "selection"
        assert pd.Timestamp(plan.folds[0].validation.end) <= pd.Timestamp("2019-04-03")
        if plan.folds[0].fold_id == "2019-q2":
            assert records[0].metrics["validation_rows"] == 8

    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )

    full = shared_model_case(
        tmp_path / "full", specs=specs, frame=rows, requested_end="2019-04-04"
    )
    payload = full.envelope.payload()
    payload.pop("instance_identity")
    payload.pop("status")
    payload["research_window"]["end"] = "2019-04-03"
    narrowed = SimpleNamespace(
        **{**vars(full), "envelope": build_business_manifest_envelope(**payload)}
    )
    root = _republish(narrowed, tmp_path / "narrowed", mutate=lambda *_: None)
    with pytest.raises(ValueError, match="model_trial_fold_binding_invalid"):
        read_quarterly_model_artifact(root, trial_layout=full.layout)

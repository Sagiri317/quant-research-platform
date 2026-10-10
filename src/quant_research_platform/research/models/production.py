"""Production model selection and quarterly refit owner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from uuid import uuid4

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.contracts.experiments import (
    ExperimentFold,
    ExperimentPlan,
    label_maturity_mask,
    legal_training_rows,
    training_availability_columns,
)
from quant_research_platform.contracts.models import (
    _model_algorithm_facts,
    _validate_a_share_model_family_ids,
    model_spec_payload,
)
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import _research_dataset_access_window
from quant_research_platform.research.datasets.validation import (
    iter_validated_research_input_chunks,
)
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.fitting import (
    model_training_read_columns,
    training_scope_chunks,
)
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
    _label_calibration,
    rank_model_parameters,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    ModelSampleCoverage,
    crypto_validation_coverage_scope,
    fit_validation_candidates,
    model_validation_coverage_scope,
    model_validation_inputs,
)
from quant_research_platform.research.protocols.a_share import (
    model_folds,
    model_selection_windows,
    quarter_id_for_fold,
)
from quant_research_platform.research.randomness import ROOT_SEED, derive_seed
from quant_research_platform.runtime.execution_metrics import record_run_observations
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    resolve_columns_from_schema,
)
from quant_research_platform.storage.layout import StorageLayout


@dataclass(frozen=True)
class FinalModelResult:
    selection_groups: tuple[str, ...]
    family_id: str
    spec: object
    model_result: fitting.ModelFitResult
    fit_windows: tuple[TimeWindow, ...]
    fit_cutoff: datetime
    fit_coverage_id: str
    refit_trial_reference: dict[str, str]


def _model_input_chunks(dataset_session, *, dataset_spec, columns, time_window):
    columns = resolve_columns_from_schema(
        tuple(dataset_session.arrow_schema.names),
        spec=dataset_session.spec,
        columns=columns,
    )
    projection = replace(
        dataset_session,
        logical_columns=columns,
        physical_columns=columns,
    )
    window = _research_dataset_access_window(dataset_spec.market_id, time_window)
    view = projection.filtered_view(window)
    with closing_iterator(iter_dataset_chunks(view)) as source:
        for yielded_value in source:
            try:
                yield yielded_value
            except GeneratorExit:
                return


def build_model_selection_result(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    model_catalog: ModelCatalog,
    dataset_id: str,
    model_ids: tuple[str, ...],
    time_window: TimeWindow,
    protocol_manifest: Mapping[str, object],
    run_id: str,
    research_protocol_identity: str,
    model_target_id: str,
    backend_versions: Mapping[str, str],
    dataset_session: DatasetSession,
    on_validation,
    on_candidate,
    on_trial,
    select_models,
    on_observation,
) -> Iterable[FinalModelResult]:
    if not model_ids:
        raise ValueError("model_selection_model_ids_invalid")
    dataset_spec = dataset_catalog.get(dataset_id)
    if protocol_manifest.get("protocol_id") != "a_share_low_frequency":
        raise ValueError("a_share_protocol_required")
    chronology = protocol_manifest.get("model_chronology")
    formal_dataset_window = protocol_manifest.get("formal_dataset_window")
    development_window = protocol_manifest.get("development_window")
    if (
        type(chronology) is not dict
        or set(chronology) != {"scoring_lookback_months", "warmup_window"}
        or type(formal_dataset_window) is not dict
        or (type(development_window) is not dict)
    ):
        raise ValueError("model_chronology_missing")
    root_seed = protocol_manifest.get("root_seed")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("model_root_seed_missing")

    def chunks(
        read_columns: tuple[str, ...],
        *,
        input_columns: tuple[str, ...],
        start: object | None,
        end: object,
    ) -> Iterable[pd.DataFrame]:
        with closing_iterator(
            iter_validated_research_input_chunks(
                _model_input_chunks(
                    dataset_session,
                    dataset_spec=dataset_spec,
                    columns=read_columns,
                    time_window=TimeWindow(
                        pd.Timestamp(
                            formal_dataset_window["start"] if start is None else start
                        ),
                        pd.Timestamp(end),
                    ),
                ),
                dataset_spec=dataset_spec,
                input_columns=input_columns,
            )
        ) as delegated_chunks:
            for yielded_value in delegated_chunks:
                try:
                    yield yielded_value
                except GeneratorExit:
                    return

    return _build_a_share_models(
        layout=layout,
        dataset_spec=dataset_spec,
        model_catalog=model_catalog,
        dataset_id=dataset_id,
        model_ids=model_ids,
        time_window=time_window,
        chunks=chunks,
        sessions=_dataset_sessions(
            _model_input_chunks(
                dataset_session,
                dataset_spec=dataset_spec,
                columns=("timestamp",),
                time_window=TimeWindow(formal_dataset_window["start"], time_window.end),
            )
        ),
        chronology=chronology,
        formal_dataset_window=formal_dataset_window,
        development_window=development_window,
        root_seed=root_seed,
        run_id=run_id,
        research_protocol_identity=research_protocol_identity,
        model_target_id=model_target_id,
        backend_versions=backend_versions,
        holdout_window=protocol_manifest["holdout_window"],
        research_configuration=protocol_manifest["research_configuration"],
        on_validation=on_validation,
        on_candidate=on_candidate,
        on_trial=on_trial,
        select_models=select_models,
        on_observation=on_observation,
    )


def _build_a_share_models(
    *,
    layout,
    dataset_spec,
    model_catalog,
    dataset_id,
    model_ids,
    time_window,
    chunks,
    sessions,
    chronology,
    formal_dataset_window,
    development_window,
    root_seed,
    run_id,
    research_protocol_identity,
    model_target_id,
    backend_versions,
    holdout_window,
    research_configuration,
    on_validation,
    on_candidate,
    on_trial,
    select_models,
    on_observation,
):
    _validate_a_share_model_family_ids(model_ids)
    requested_end = pd.Timestamp(time_window.end)
    folds = model_folds(
        sessions[sessions <= requested_end],
        chronology=chronology,
        formal_dataset_window=formal_dataset_window,
        development_window={**development_window, "end": time_window.end},
    )
    if not folds:
        raise ValueError("a_share_walk_forward_window_empty")
    family_specs = tuple(model_catalog.get(model_id) for model_id in model_ids)
    common_inputs = model_validation_inputs(family_specs)
    label = family_specs[0].label_column
    availability = training_availability_columns(label)[0]
    dataset_identity = layout.target_root(
        dataset_spec.data_id,
        layout.research_dataset_root(dataset_spec.market_id, dataset_spec.data_id),
    ).name
    test = TimeWindow(
        *(pd.Timestamp(holdout_window[key]).to_pydatetime() for key in ("start", "end"))
    )
    configurations = {
        item["model_id"]: item for item in research_configuration["models"]
    }
    selection_windows = {
        (group, key): (cutoff, window)
        for group, key, cutoff, window in model_selection_windows(
            (
                (quarter_id_for_fold(fold), fold.train.end, fold.validation.start)
                for fold in folds
            ),
            chronology=chronology,
            development_window=development_window,
            complete=pd.Timestamp(time_window.end)
            >= pd.Timestamp(development_window["end"]),
        )
    }
    last_recorders = {}
    last_specs = {}
    prepared_domains = {}

    def prepare_domain(spec, cutoff):
        domain = (cutoff, *_training_domain(spec))
        if domain not in prepared_domains:
            prepared_domains[domain] = _prepare_training(
                dataset_spec,
                spec,
                chunks=chunks,
                train_end=cutoff,
            )
        return prepared_domains[domain]

    for fold in folds:
        prepared_domains.clear()
        quarter_id = quarter_id_for_fold(fold)
        if ("quarterly", quarter_id) in selection_windows:
            select_models(
                "quarterly", quarter_id, *selection_windows[("quarterly", quarter_id)]
            )
        coverage = ModelSampleCoverage(
            model_validation_coverage_scope(dataset_identity, fold, family_specs)
        )
        columns = tuple(
            dict.fromkeys(
                (
                    *dataset_spec.primary_key,
                    *common_inputs,
                    label,
                    availability,
                    "eligible",
                )
            )
        )
        with closing_iterator(
            chunks(
                columns,
                input_columns=common_inputs,
                start=fold.validation.start,
                end=fold.validation.end,
            )
        ) as frames:
            for frame in frames:
                legal = legal_training_rows(
                    frame,
                    fitting_cutoff=fold.validation.end,
                    availability_column=availability,
                    feature_columns=common_inputs,
                    target_column=label,
                )
                coverage.add(legal)

        if not coverage.row_count:
            raise PortfolioTargetBlocked("a_share_validation_scoring_coverage_missing")
        for family_id in model_ids:
            base = model_catalog.get(family_id)
            configuration = configurations[family_id]
            parameters = configuration.get(
                "parameter_candidates", [configuration["hyperparameters"]]
            )
            prepared_training = prepare_domain(base, fold.train.end)
            n_train = len(prepared_training.training)
            candidates = _selection_model_specs(
                base,
                parameters,
                n_train=n_train,
                root_seed=root_seed,
                quarter_id=quarter_id,
            )
            candidate_ids = tuple(candidate.model_id for candidate in candidates)
            plan = ExperimentPlan(
                stage="selection",
                experiment_id=f"experiment_{uuid4().hex}",
                market_id=dataset_spec.market_id,
                research_dataset_id=dataset_id,
                target_id=model_target_id,
                candidate_family_id=family_id,
                data_snapshot_id=dataset_identity,
                input_lineage=(dataset_identity,),
                research_protocol_identity=research_protocol_identity,
                folds=(fold,),
                test=test,
                selection_metric="validation_rankic",
                selection_direction="maximize",
                candidate_ids=candidate_ids,
                trial_budget=len(candidate_ids),
                random_seed=root_seed,
                tie_break_order=candidate_ids,
                backend_versions=backend_versions,
                thread_settings={"blas_threads": 1},
                candidate_parameters={
                    candidate.model_id: model_spec_payload(candidate)
                    for candidate in candidates
                },
            )
            recorder = TrialRecorder(layout, run_id, plan)
            observation = {}
            with record_run_observations(observation):
                fit_validation_candidates(
                    fold=fold,
                    candidate_specs=candidates,
                    training_by_candidate={
                        candidate.model_id: prepared_training
                        for candidate in candidates
                    },
                    dataset_spec=dataset_spec,
                    chunks=chunks,
                    validation_input_columns=common_inputs,
                    validation_coverage=coverage,
                    trial_recorder=recorder,
                    on_validation=lambda spec, frame, fold=fold, family=family_id: (
                        on_validation(fold, family, spec, frame)
                    ),
                    on_candidate=lambda spec, metadata, fold=fold, family=family_id: (
                        on_candidate(fold, family, spec, metadata)
                    ),
                )
            on_observation(
                {
                    "phase": "shared_candidate_fit",
                    "selection_groups": [],
                    "family_id": family_id,
                    "quarter_id": quarter_id,
                    "fit_count": len(candidates),
                    "observations": observation,
                }
            )
            on_trial(fold, family_id, recorder.reference())
            last_recorders[family_id] = recorder
            last_specs[family_id] = candidates
    cutoff = pd.Timestamp(development_window["end"])
    if pd.Timestamp(time_window.end) < cutoff:
        return
    fit_windows = (TimeWindow(pd.Timestamp(formal_dataset_window["start"]), cutoff),)
    final_decisions = {
        group: select_models(group, "final", *selection_windows[(group, "final")])
        for group in ("global", "quarterly")
    }
    final_choices = {}
    for group, decision in final_decisions.items():
        for family in decision["families"]:
            key = (family["family_id"], family["selected_config_id"])
            final_choices.setdefault(key, []).append(group)
    prepared_domains.clear()
    for (family_id, selected_id), selected_groups in final_choices.items():
        groups = tuple(selected_groups)
        parents = tuple(
            sorted(
                {
                    identity
                    for group in groups
                    for identity in final_decisions[group][
                        "selection_ledger_semantic_identities"
                    ]
                }
            )
        )
        candidates = last_specs[family_id]
        selected = next(spec for spec in candidates if spec.model_id == selected_id)
        configuration = configurations[family_id]
        parameters = configuration.get(
            "parameter_candidates", [configuration["hyperparameters"]]
        )
        selected_parameters = parameters[
            tuple(spec.model_id for spec in candidates).index(selected_id)
        ]
        recorder = last_recorders[family_id]
        refit_recorder = attempt = fit_coverage = actual_spec = None

        finished = False

        def begin_refit(
            spec, training, *, recorder=recorder, groups=groups, parents=parents
        ):
            nonlocal refit_recorder, attempt, fit_coverage, actual_spec
            actual_spec = spec
            fit_coverage = _refit_sample_coverage(
                recorder.plan,
                spec,
                training,
                fitting_cutoff=cutoff,
                fit_windows=fit_windows,
            )
            refit_recorder, attempt = recorder.begin_refit(
                spec,
                common_coverage_id=fit_coverage.semantic_identity,
                fit_windows=fit_windows,
                fit_cutoff=cutoff.to_pydatetime(),
                selection_groups=groups,
                selection_ledger_semantic_identities=parents,
            )

        try:
            observation = {}
            with record_run_observations(observation):
                model_result = _fit_selected_model_result(
                    prepared_training=prepare_domain(selected, cutoff.isoformat()),
                    selected_spec=selected,
                    quarter_id="final",
                    root_seed=root_seed,
                    candidate_parameters=selected_parameters,
                    on_fit_start=begin_refit,
                )
            on_observation(
                {
                    "phase": "final_refit",
                    "selection_groups": list(groups),
                    "family_id": family_id,
                    "quarter_id": None,
                    "fit_count": 1,
                    "observations": observation,
                }
            )
            refit_recorder.finish(attempt, model_metadata=model_result.metadata)
            finished = True
            member = FinalModelResult(
                groups,
                family_id,
                actual_spec,
                model_result,
                fit_windows,
                cutoff.to_pydatetime(),
                fit_coverage.semantic_identity,
                refit_recorder.reference(),
            )
            yield member
        except BaseException as error:
            if attempt is not None and (not finished):
                try:
                    refit_recorder.finish(attempt, error=error)
                except BaseException as secondary:
                    error.add_note(
                        "refit failure record also failed: " + type(secondary).__name__
                    )
            raise


def _dataset_sessions(chunks: Iterable[pd.DataFrame]) -> pd.DatetimeIndex:
    values: set[pd.Timestamp] = set()
    with closing_iterator(iter(chunks)) as source:
        for frame in source:
            timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
            values.update(pd.DatetimeIndex(timestamps.unique()).normalize())
    if not values:
        raise ValueError("a_share_walk_forward_sessions_empty")
    return pd.DatetimeIndex(sorted(values))


def _refit_sample_coverage(
    plan, spec, training, *, fitting_cutoff, fit_windows, asset_ids=()
):
    coverage = ModelSampleCoverage(
        {
            "dataset_identity": plan.data_snapshot_id,
            "fit_windows": [
                {"start": str(window.start), "end": str(window.end)}
                for window in fit_windows
            ],
            "stage": "refit",
            "fit_cutoff": str(fitting_cutoff),
            "label": spec.label_column,
            "availability": training_availability_columns(spec.label_column)[0],
            "inputs": list(spec.input_columns),
            "input_kind": _model_algorithm_facts(spec.model_algorithm).input_kind,
            "max_train_rows": spec.max_train_rows,
            "asset_ids": list(asset_ids),
            "eligible": True,
        }
    )
    coverage.add(training)
    return coverage


def _fit_selected_model_result(
    *,
    prepared_training,
    selected_spec,
    quarter_id: str,
    root_seed: int,
    candidate_parameters,
    on_fit_start=None,
) -> fitting.ModelFitResult:
    n_refit = len(prepared_training.training)
    refit_spec = _refit_model_spec(
        selected_spec,
        n_train=n_refit,
        quarter_id=quarter_id,
        root_seed=root_seed,
        candidate_parameters=candidate_parameters,
    )
    if on_fit_start is not None:
        on_fit_start(refit_spec, prepared_training.training)
    return fitting._fit_model_result(
        prepared_training,
        refit_spec,
        estimator=build_estimator_from_model_spec(refit_spec),
    )


def _resolve_candidate_parameters(algorithm, parameters, n_train):
    values = dict(parameters)
    if "regularization_strength" in values:
        if (
            algorithm not in {"ridge_regression", "huber_regression"}
            or "alpha" in values
        ):
            raise ValueError("model_regularization_parameters_invalid")
        values["alpha"] = n_train * values.pop("regularization_strength")
    if "min_child_weight_fraction" in values:
        if algorithm != "xgboost_regressor" or "min_child_weight" in values:
            raise ValueError("model_child_weight_parameters_invalid")
        values["min_child_weight"] = n_train * values.pop("min_child_weight_fraction")
    return values


def _refit_model_spec(
    selected_spec, *, n_train, quarter_id, root_seed, candidate_parameters
):
    return replace(
        selected_spec,
        random_seed=derive_seed(
            root_seed,
            "a_share_model_refit",
            selected_spec.model_algorithm,
            selected_spec.model_id,
            quarter_id,
        ),
        hyperparameters=_resolve_candidate_parameters(
            selected_spec.model_algorithm, candidate_parameters, n_train
        ),
    )


def _selection_model_specs(base_spec, parameters, *, n_train, root_seed, quarter_id):
    return tuple(
        (
            replace(
                base_spec,
                model_id=base_spec.model_id
                if len(parameters) == 1
                else f"{base_spec.model_id}__config_{index}",
                hyperparameters=_resolve_candidate_parameters(
                    base_spec.model_algorithm, values, n_train
                ),
                random_seed=derive_seed(
                    root_seed,
                    "a_share_model_trial",
                    base_spec.model_algorithm,
                    base_spec.model_id
                    if len(parameters) == 1
                    else f"{base_spec.model_id}__config_{index}",
                    quarter_id,
                    index,
                ),
            )
            for index, values in enumerate(parameters)
        )
    )


def _prepare_training(dataset_spec, model_spec, *, chunks, train_end):
    """Prepare one selection domain, preserving validation of excluded raw rows."""
    read_columns = model_training_read_columns(
        dataset_spec.primary_key,
        model_spec.input_columns,
        label_column=model_spec.label_column,
        sample_filter_mode=model_spec.sample_filter_mode,
    )

    prepared = fitting._collect_training_frame(
        chunks(
            read_columns,
            input_columns=model_spec.input_columns,
            start=None,
            end=train_end,
        ),
        model_spec,
        fitting_cutoff=train_end,
    )
    if prepared.training.empty:
        raise ValueError("model_training_sample_empty")
    return prepared


@dataclass(frozen=True)
class CryptoFoldModelResult:
    fold: ExperimentFold
    spec: object
    result: fitting.ModelFitResult
    training_assets: tuple[str, ...]
    evaluation_assets: tuple[str, ...]
    selection_trial_reference: dict[str, str]
    refit_trial_reference: dict[str, str]
    fit_coverage_id: str


def _crypto_model_chronology(protocol_manifest, time_window):
    from quant_research_platform.research.protocols.crypto_hft import (
        research_chronology,
    )

    folds, holdout = research_chronology(protocol_manifest["research_plan"])
    return (
        tuple(
            fold
            for fold in folds
            if pd.Timestamp(fold.validation.end) >= pd.Timestamp(time_window.start)
            and pd.Timestamp(fold.validation.start) <= pd.Timestamp(time_window.end)
        ),
        holdout,
    )


def _training_domain(spec):
    return (
        spec.input_factor_columns,
        spec.input_feature_columns,
        spec.label_column,
        spec.sample_filter_mode,
        spec.max_train_rows,
        spec.training_weighting,
        _model_algorithm_facts(spec.model_algorithm).input_kind,
    )


def build_crypto_fold_models(
    *,
    layout,
    dataset_session,
    dataset_catalog,
    model_catalog,
    dataset_id,
    model_ids,
    time_window,
    protocol_manifest,
    run_id,
    research_protocol_identity,
    model_target_id,
    backend_versions,
    on_validation,
    on_selection,
):
    from quant_research_platform.research.models.trial_selection import (
        evaluate_model_trial_metrics_from_chunks,
        iter_validation_prediction_chunks,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_EXECUTION_CONTEXT_COLUMNS,
        build_hft_prediction_catalog,
    )

    def fit_prepared(prepared, spec, *, on_fit_start=None):
        result = fitting._fit_model_result(
            prepared,
            spec,
            estimator=build_estimator_from_model_spec(spec),
            on_fit_start=on_fit_start,
        )
        predictor = result.prediction_model
        if isinstance(predictor, (RankEnsemblePredictor, ICWeightedRankEnsemble)):
            result.prediction_model = replace(
                predictor,
                label_calibration=_label_calibration(
                    predictor.predict(prepared.training),
                    prepared.training[spec.label_column],
                    prepared.preprocessing[0],
                ),
            )
            result.metadata["rank_parameters"] = rank_model_parameters(
                result.prediction_model
            )
        return result

    folds, holdout = _crypto_model_chronology(protocol_manifest, time_window)
    if not folds:
        return
    dataset = dataset_catalog.get(dataset_id)
    identity = dataset_session.root.name
    for fold in folds:
        prediction_catalog = build_hft_prediction_catalog(
            configuration=protocol_manifest["research_configuration"],
            train_end=fold.train.end.isoformat(),
            test_start=fold.validation.start.isoformat(),
            test_end=fold.validation.end.isoformat(),
        )
        domains = {}
        # One source session and fold fix identity, windows and cutoffs for each group.
        for model_id in model_ids:
            spec = model_catalog.get(model_id)
            prediction = prediction_catalog.get(f"{model_id}_prediction")
            domain = (prediction.training_asset_ids, *_training_domain(spec))
            domains.setdefault(domain, []).append(model_id)
        for model_group in domains.values():
            prepared_train = prepared_refit = None
            for model_id in model_group:
                spec = replace(
                    model_catalog.get(model_id),
                    random_seed=derive_seed(
                        ROOT_SEED, "crypto_model", fold.fold_id, model_id
                    ),
                )
                prediction = prediction_catalog.get(f"{model_id}_prediction")
                plan = ExperimentPlan(
                    stage="selection",
                    experiment_id=f"experiment_{uuid4().hex}",
                    market_id=dataset.market_id,
                    research_dataset_id=dataset_id,
                    target_id=model_target_id,
                    candidate_family_id=model_id,
                    data_snapshot_id=identity,
                    input_lineage=(identity,),
                    research_protocol_identity=research_protocol_identity,
                    folds=(fold,),
                    test=holdout,
                    selection_metric="validation_mse",
                    selection_direction="minimize",
                    candidate_ids=(model_id,),
                    trial_budget=1,
                    random_seed=spec.random_seed,
                    tie_break_order=(model_id,),
                    backend_versions=backend_versions,
                    thread_settings={"blas_threads": 1},
                    candidate_parameters={model_id: model_spec_payload(spec)},
                )
                recorder = TrialRecorder(layout, run_id, plan)

                metrics = training_metadata = None

                def chunks(window, spec=spec, prediction=prediction):
                    source = _model_input_chunks(
                        dataset_session,
                        dataset_spec=dataset,
                        time_window=window,
                        columns=(
                            *model_training_read_columns(
                                dataset.primary_key,
                                spec.input_columns,
                                label_column=spec.label_column,
                                sample_filter_mode=spec.sample_filter_mode,
                            ),
                            *HFT_EXECUTION_CONTEXT_COLUMNS,
                        ),
                    )
                    with closing_iterator(
                        training_scope_chunks(
                            iter_validated_research_input_chunks(
                                source,
                                dataset_spec=dataset,
                                input_columns=spec.input_columns,
                            ),
                            asset_ids=prediction.training_asset_ids,
                        )
                    ) as delegated_chunks:
                        for yielded_value in delegated_chunks:
                            try:
                                yield yielded_value
                            except GeneratorExit:
                                return

                validation_coverage = ModelSampleCoverage(
                    crypto_validation_coverage_scope(
                        plan, spec, prediction.training_asset_ids
                    )
                )
                with closing_iterator(chunks(fold.validation)) as validation_source:
                    for frame in validation_source:
                        legal = legal_training_rows(
                            frame,
                            fitting_cutoff=fold.validation.end,
                            availability_column=training_availability_columns(
                                spec.label_column
                            )[0],
                            feature_columns=model_input_validity_columns(
                                spec.input_columns, spec.model_algorithm
                            ),
                            target_column=spec.label_column,
                        )
                        validation_coverage.add(legal)

                if validation_coverage.row_count == 0:
                    raise PortfolioTargetBlocked(
                        "crypto_model_validation_legal_rows_missing"
                    )
                attempt = recorder.begin(
                    spec, common_coverage_id=validation_coverage.semantic_identity
                )

                finished = False
                refit_recorder = refit_attempt = fit_coverage = None
                refit_finished = False

                def begin_crypto_refit(
                    actual_spec,
                    training,
                    *,
                    recorder=recorder,
                    fold=fold,
                    prediction=prediction,
                ):
                    nonlocal refit_recorder, refit_attempt, fit_coverage
                    fit_coverage = _refit_sample_coverage(
                        recorder.plan,
                        actual_spec,
                        training,
                        fitting_cutoff=fold.validation.end,
                        fit_windows=(fold.train, fold.validation),
                        asset_ids=prediction.training_asset_ids,
                    )
                    refit_recorder, refit_attempt = recorder.begin_refit(
                        actual_spec,
                        common_coverage_id=fit_coverage.semantic_identity,
                        fit_windows=(fold.train, fold.validation),
                        fit_cutoff=fold.validation.end,
                        selection_groups=(),
                        selection_ledger_semantic_identities=(
                            recorder.reference()["semantic_identity"],
                        ),
                    )

                try:
                    if prepared_train is None:
                        prepared_train = fitting._collect_training_frame(
                            chunks(fold.train), spec, fitting_cutoff=fold.train.end
                        )
                    trained = fit_prepared(prepared_train, spec)
                    training_metadata = dict(trained.metadata)
                    validation = iter_validation_prediction_chunks(
                        lambda fold=fold: chunks(fold.validation),
                        candidate_spec=spec,
                        model_result=trained,
                        validation_start=pd.Timestamp(fold.validation.start),
                        validation_end=pd.Timestamp(fold.validation.end),
                        extra_output_columns=HFT_EXECUTION_CONTEXT_COLUMNS,
                    )

                    def observed_validation(
                        validation=validation, model_id=model_id, fold=fold, spec=spec
                    ):
                        with closing_iterator(iter(validation)) as validation_source:
                            for item in validation_source:
                                frame = item
                                mature = frame.loc[
                                    label_maturity_mask(
                                        frame,
                                        fitting_cutoff=fold.validation.end,
                                        availability_column="label_available_at",
                                        target_column=spec.label_column,
                                    )
                                ]
                                on_validation(fold, spec, mature)
                                try:
                                    yield mature
                                except GeneratorExit:
                                    return

                    metrics = evaluate_model_trial_metrics_from_chunks(
                        observed_validation(),
                        model_id=model_id,
                        label_column=spec.label_column,
                        prediction_column=prediction.prediction_column,
                        expected_coverage=validation_coverage,
                        validation_axis=validation_coverage.scope["validation_axis"],
                        label_calibration=training_metadata.get(
                            "rank_parameters", {}
                        ).get("label_calibration"),
                    )
                    if metrics["validation_rows"] <= 0 or not pd.notna(
                        metrics["validation_mse"]
                    ):
                        raise ValueError("crypto_validation_sample_empty")

                    recorder.finish(
                        attempt, metrics=metrics, model_metadata=training_metadata
                    )
                    finished = True
                    selection_reference = recorder.reference()
                    on_selection(
                        fold,
                        spec,
                        training_metadata,
                        prediction.training_asset_ids,
                        prediction.evaluation_asset_ids,
                        selection_reference,
                    )
                    if (
                        fold.fold_id
                        != protocol_manifest["research_plan"]["folds"][-1]["fold_id"]
                    ):
                        continue

                    def refit_chunks(fold=fold):
                        with closing_iterator(chunks(fold.train)) as delegated_chunks:
                            for yielded_value in delegated_chunks:
                                try:
                                    yield yielded_value
                                except GeneratorExit:
                                    return
                        with closing_iterator(
                            chunks(fold.validation)
                        ) as delegated_chunks:
                            for yielded_value in delegated_chunks:
                                try:
                                    yield yielded_value
                                except GeneratorExit:
                                    return

                    if prepared_refit is None:
                        prepared_refit = fitting._collect_training_frame(
                            refit_chunks(), spec, fitting_cutoff=fold.validation.end
                        )
                    result = fit_prepared(
                        prepared_refit, spec, on_fit_start=begin_crypto_refit
                    )
                    refit_recorder.finish(refit_attempt, model_metadata=result.metadata)
                    refit_finished = True
                    member = CryptoFoldModelResult(
                        fold,
                        spec,
                        result,
                        prediction.training_asset_ids,
                        prediction.evaluation_asset_ids,
                        recorder.reference(),
                        refit_recorder.reference(),
                        fit_coverage.semantic_identity,
                    )
                    yield member
                except BaseException as error:
                    if refit_attempt is not None and (not refit_finished):
                        try:
                            refit_recorder.finish(refit_attempt, error=error)
                        except BaseException as secondary:
                            error.add_note(
                                "refit failure record also failed: "
                                + type(secondary).__name__
                            )
                    if not finished:
                        try:
                            recorder.finish(
                                attempt,
                                metrics=metrics,
                                model_metadata=training_metadata,
                                error=error,
                            )
                        except BaseException as secondary:
                            error.add_note(
                                "trial failure record also failed: "
                                f"{type(secondary).__name__}"
                            )
                    raise

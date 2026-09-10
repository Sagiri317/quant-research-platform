"""Production model selection and quarterly refit owner."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from uuid import uuid4

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.contracts.experiments import ExperimentFold, ExperimentPlan
from quant_research_platform.contracts.models import (
    _model_algorithm_facts,
    model_spec_payload,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import (
    _research_dataset_access_window,
    iter_research_dataset_chunks,
)
from quant_research_platform.research.datasets.validation import (
    iter_validated_research_input_chunks,
)
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.fitting import (
    iter_legal_training_chunks,
    model_training_read_columns,
    training_scope_chunks,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    select_best_validation_trial,
    select_prediction_model_spec,
)
from quant_research_platform.research.protocols.a_share import (
    HUBER_LAMBDAS,
    LASSO_ALPHAS,
    RF_CONFIGS,
    RIDGE_LAMBDAS,
    XGB_CONFIGS,
    model_folds,
    quarter_id_for_fold,
)
from quant_research_platform.research.randomness import ROOT_SEED, derive_seed
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    resolve_columns_from_schema,
)
from quant_research_platform.storage.layout import StorageLayout


@dataclass(frozen=True)
class QuarterlyModelResult:
    quarter: ExperimentFold
    model_id: str
    model_result: fitting.ModelFitResult
    family_id: str
    selection_metadata: Mapping[str, object]
    trial_ledgers: tuple[dict[str, str], ...] = ()


def build_model_selection_result(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    model_catalog: ModelCatalog,
    dataset_id: str,
    model_ids: tuple[str, ...],
    tie_break_order: tuple[str, ...],
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    protocol_manifest: Mapping[str, object],
    run_id: str,
    research_protocol_identity: str,
    model_target_id: str,
    backend_versions: Mapping[str, str],
    dataset_session: DatasetSession,
) -> Iterable[QuarterlyModelResult | OwnedValue[QuarterlyModelResult]]:
    if not model_ids or set(model_ids) != set(tie_break_order):
        raise ValueError("model_selection_model_ids_invalid")
    dataset_spec = dataset_catalog.get(dataset_id)
    if protocol_manifest.get("protocol_id") != "a_share_low_frequency":
        raise ValueError("a_share_protocol_required")
    chronology = protocol_manifest.get("model_chronology")
    formal_dataset_window = protocol_manifest.get("formal_dataset_window")
    development_window = protocol_manifest.get("development_window")
    if (
        type(chronology) is not dict
        or set(chronology) != {"validation_months"}
        or type(formal_dataset_window) is not dict
        or type(development_window) is not dict
    ):
        raise ValueError("model_chronology_missing")
    root_seed = protocol_manifest.get("root_seed")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("model_root_seed_missing")

    def read_chunks(read_columns: tuple[str, ...], window: TimeWindow):
        columns = resolve_columns_from_schema(
            tuple(dataset_session.arrow_schema.names),
            spec=dataset_session.spec,
            columns=read_columns,
        )
        projection = replace(
            dataset_session,
            logical_columns=columns,
            physical_columns=columns,
            owns_inventory=False,
            owns_lease=False,
        )
        window = _research_dataset_access_window(dataset_spec.market_id, window)
        with projection.filtered_view(window) as view:
            with closing_iterator(
                iter_dataset_chunks(
                    view,
                    target_chunk_bytes=max(1, resource_policy.operation_headroom_bytes),
                )
            ) as source:
                yield from source

    def chunks(
        read_columns: tuple[str, ...],
        *,
        input_columns: tuple[str, ...],
        start: object | None,
        end: object,
    ) -> Iterable[pd.DataFrame]:
        yield from iter_validated_research_input_chunks(
            read_chunks(
                read_columns,
                TimeWindow(
                    formal_dataset_window["start"] if start is None else start,
                    end,
                ),
            ),
            dataset_spec=dataset_spec,
            input_columns=input_columns,
        )

    return _build_a_share_quarterly_models(
        layout=layout,
        dataset_spec=dataset_spec,
        model_catalog=model_catalog,
        dataset_id=dataset_id,
        model_ids=model_ids,
        tie_break_order=tie_break_order,
        time_window=time_window,
        chunks=chunks,
        sessions=_dataset_sessions(
            read_chunks(
                ("timestamp",),
                TimeWindow(formal_dataset_window["start"], development_window["end"]),
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
    )


def _build_a_share_quarterly_models(
    *,
    layout: StorageLayout,
    dataset_spec,
    model_catalog: ModelCatalog,
    dataset_id: str,
    model_ids: tuple[str, ...],
    tie_break_order: tuple[str, ...],
    time_window: TimeWindow,
    chunks,
    sessions: pd.DatetimeIndex,
    chronology: Mapping[str, object],
    formal_dataset_window: Mapping[str, object],
    development_window: Mapping[str, object],
    root_seed: int,
    run_id: str,
    research_protocol_identity: str,
    model_target_id: str,
    backend_versions: Mapping[str, str],
    holdout_window: Mapping[str, object],
) -> Iterable[QuarterlyModelResult | OwnedValue[QuarterlyModelResult]]:
    quarters = model_folds(
        sessions,
        chronology=chronology,
        formal_dataset_window=formal_dataset_window,
        development_window=development_window,
    )
    requested_start = pd.Timestamp(time_window.start)
    requested_end = pd.Timestamp(time_window.end)
    selected_quarters = tuple(
        quarter
        for quarter in quarters
        if pd.Timestamp(quarter.development_test.end) >= requested_start
        and pd.Timestamp(quarter.development_test.start) <= requested_end
    )
    if not selected_quarters:
        raise ValueError("a_share_walk_forward_window_empty")

    def quarterly_results() -> Iterable[
        QuarterlyModelResult | OwnedValue[QuarterlyModelResult]
    ]:
        model_lease: MemoryLease | None = None
        try:
            for quarter in selected_quarters:
                selection_spec = PredictionSpec(
                    prediction_id=(
                        f"{dataset_id}_model_selection_{quarter_id_for_fold(quarter).lower()}"
                    ),
                    model_id=model_ids[0],
                    research_dataset_id=dataset_id,
                    train_end=time_value_isoformat(quarter.train.end, date_only=True),
                    test_start=time_value_isoformat(
                        quarter.development_test.start, date_only=True
                    ),
                    test_end=time_value_isoformat(
                        quarter.development_test.end, date_only=True
                    ),
                    candidate_model_ids=model_ids[1:],
                    model_selection_validation_start=time_value_isoformat(
                        quarter.validation.start, date_only=True
                    ),
                    model_selection_validation_end=time_value_isoformat(
                        quarter.validation.end, date_only=True
                    ),
                )
                winner_specs, selection_metadata, trial_ledgers, config_indices = (
                    _select_model_family_winners(
                        selection_spec=selection_spec,
                        dataset_spec=dataset_spec,
                        model_catalog=model_catalog,
                        chunks=chunks,
                        model_ids=model_ids,
                        tie_break_order=tie_break_order,
                        quarter_id=quarter_id_for_fold(quarter),
                        root_seed=root_seed,
                        layout=layout,
                        run_id=run_id,
                        research_protocol_identity=research_protocol_identity,
                        model_target_id=model_target_id,
                        quarter=quarter,
                        holdout_window=holdout_window,
                        backend_versions=backend_versions,
                    )
                )
                for family_id, reference in zip(model_ids, trial_ledgers, strict=True):
                    selected_spec = winner_specs[family_id]
                    model_result, model_lease = _fit_selected_model_result(
                        dataset_spec=dataset_spec,
                        selected_spec=selected_spec,
                        chunks=chunks,
                        fitting_cutoff=time_value_isoformat(
                            quarter.validation.end, date_only=True
                        ),
                        quarter_id=quarter_id_for_fold(quarter),
                        root_seed=root_seed,
                        config_index=config_indices[family_id],
                    )
                    item = QuarterlyModelResult(
                        quarter=quarter,
                        model_id=selected_spec.model_id,
                        model_result=model_result,
                        family_id=family_id,
                        selection_metadata=selection_metadata,
                        trial_ledgers=(reference,),
                    )
                    owned_item = (
                        OwnedValue(item, model_lease)
                        if model_lease is not None
                        else item
                    )
                    try:
                        yield owned_item
                    finally:
                        ownership = (
                            owned_item if isinstance(owned_item, OwnedValue) else None
                        )
                        primary = sys.exception()
                        item = None
                        model_result = None
                        model_lease = None
                        owned_item = None
                        if ownership is not None and ownership.lease.active:
                            try:
                                ownership.release(owner="research.model.artifact")
                            except BaseException as cleanup_error:
                                if primary is not None:
                                    primary.add_note(
                                        "quarterly model lease release also failed: "
                                        f"{type(cleanup_error).__name__}"
                                    )
                                else:
                                    raise
                        ownership = None
        finally:
            if model_lease is not None and model_lease.active:
                primary = sys.exception()
                try:
                    model_lease.close()
                except BaseException as cleanup_error:
                    if primary is not None:
                        primary.add_note(
                            "quarterly model lease release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    else:
                        raise

    return quarterly_results()


def _dataset_sessions(chunks: Iterable[pd.DataFrame]) -> pd.DatetimeIndex:
    values: set[pd.Timestamp] = set()
    with closing_iterator(iter(chunks)) as source:
        for frame in source:
            timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
            values.update(pd.Timestamp(value).normalize() for value in timestamps)
    if not values:
        raise ValueError("a_share_walk_forward_sessions_empty")
    return pd.DatetimeIndex(sorted(values))


def _fit_selected_model_result(
    *,
    dataset_spec,
    selected_spec,
    chunks,
    fitting_cutoff: str,
    quarter_id: str,
    root_seed: int,
    config_index: int,
) -> tuple[fitting.ModelFitResult, MemoryLease | None]:
    n_refit = _count_training_rows(
        dataset_spec, selected_spec, chunks=chunks, train_end=fitting_cutoff
    )
    refit_spec = _refit_model_spec(
        selected_spec,
        n_train=n_refit,
        quarter_id=quarter_id,
        root_seed=root_seed,
        config_index=config_index,
    )
    refit_read_columns = model_training_read_columns(
        dataset_spec.primary_key,
        refit_spec.input_columns,
        label_column=refit_spec.label_column,
        sample_filter_mode=refit_spec.sample_filter_mode,
    )
    owned_model = fitting._fit_model_result_owned_from_frames(
        training_scope_chunks(
            chunks(
                refit_read_columns,
                input_columns=refit_spec.input_columns,
                start=None,
                end=fitting_cutoff,
            ),
        ),
        refit_spec,
        estimator=build_estimator_from_model_spec(refit_spec),
        fitting_cutoff=fitting_cutoff,
    )
    model_result = (
        owned_model.value if isinstance(owned_model, OwnedValue) else owned_model
    )
    return model_result, owned_model.lease if isinstance(
        owned_model, OwnedValue
    ) else None


def _refit_model_spec(selected_spec, *, n_train, quarter_id, root_seed, config_index):
    from quant_research_platform.research.protocols.a_share import (
        resolve_trial_parameters,
    )

    return replace(
        selected_spec,
        random_seed=derive_seed(
            root_seed,
            "a_share_model_refit",
            selected_spec.model_algorithm,
            selected_spec.model_id,
            quarter_id,
        ),
        hyperparameters=resolve_trial_parameters(
            selected_spec.model_algorithm, n_train=n_train, config_index=config_index
        ),
    )


def _select_model_family_winners(
    *,
    selection_spec: PredictionSpec,
    dataset_spec,
    model_catalog: ModelCatalog,
    chunks,
    model_ids: tuple[str, ...],
    tie_break_order: tuple[str, ...],
    quarter_id: str,
    root_seed: int,
    layout: StorageLayout,
    run_id: str,
    research_protocol_identity: str,
    model_target_id: str,
    quarter: ExperimentFold,
    holdout_window: Mapping[str, object],
    backend_versions: Mapping[str, str],
):
    from quant_research_platform.research.protocols.a_share import (
        resolve_trial_parameters,
    )

    family_winners: list[dict[str, object]] = []
    trial_ledgers: list[dict[str, str]] = []
    winner_specs: dict[str, object] = {}
    config_indices: dict[str, int] = {}
    winner_metrics: list[dict[str, object]] = []
    family_validation_inputs = tuple(
        dict.fromkeys(
            column
            for family_id in model_ids
            for column in model_catalog.get(family_id).input_columns
            if _model_algorithm_facts(
                model_catalog.get(family_id).model_algorithm
            ).input_kind
            != "rank_ensemble_inputs"
        )
    )
    for family_id in model_ids:
        base_spec = model_catalog.get(family_id)
        n_train = _count_training_rows(
            dataset_spec,
            base_spec,
            chunks=chunks,
            train_end=selection_spec.train_end,
        )
        trial_specs = _a_share_trial_specs(
            base_spec,
            n_train=n_train,
            resolve_trial_parameters=resolve_trial_parameters,
            quarter_id=quarter_id,
            root_seed=root_seed,
        )
        trial_catalog = ModelCatalog(trial_specs)
        trial_ids = tuple(spec.model_id for spec in trial_specs)
        trial_order = _a_share_trial_tie_break(base_spec.model_algorithm, trial_ids)
        family_prediction = replace(
            selection_spec,
            prediction_id=f"{selection_spec.prediction_id}_{family_id}",
            model_id=trial_order[0],
            candidate_model_ids=trial_order[1:],
            selection_tie_break_order=(),
        )
        dataset_identity = layout.target_root(
            dataset_spec.data_id,
            layout.research_dataset_root(dataset_spec.market_id, dataset_spec.data_id),
        ).name
        plan = ExperimentPlan(
            experiment_id=f"experiment_{uuid4().hex}",
            market_id=dataset_spec.market_id,
            research_dataset_id=dataset_spec.data_id,
            target_id=model_target_id,
            candidate_family_id=family_id,
            data_snapshot_id=dataset_identity,
            input_lineage=(dataset_identity,),
            research_protocol_identity=research_protocol_identity,
            folds=(quarter,),
            final_holdout=TimeWindow(
                *(
                    pd.Timestamp(holdout_window[key])
                    .tz_localize(pd.Timestamp(quarter.train.start).tz)
                    .to_pydatetime()
                    if isinstance(quarter.train.start, datetime)
                    else pd.Timestamp(holdout_window[key]).date()
                    for key in ("start", "end")
                )
            ),
            selection_metric="validation_rankic",
            selection_direction="maximize",
            candidate_ids=trial_ids,
            trial_budget=len(trial_ids),
            random_seed=root_seed,
            tie_break_order=trial_order,
            backend_versions=backend_versions,
            thread_settings={"blas_threads": 1},
            candidate_parameters={
                spec.model_id: model_spec_payload(spec) for spec in trial_specs
            },
        )
        recorder = TrialRecorder(layout, run_id, plan)
        winner_spec, metadata = select_prediction_model_spec(
            prediction_spec=family_prediction,
            dataset_spec=dataset_spec,
            model_catalog=trial_catalog,
            chunks=chunks,
            validation_input_columns=family_validation_inputs,
            trial_recorder=recorder,
        )
        reference = recorder.reference()
        trial_ledgers.append(reference)
        winner_specs[family_id] = winner_spec
        config_indices[family_id] = trial_ids.index(winner_spec.model_id)
        selected_trial = next(
            item
            for item in metadata["trial_metrics"]
            if item["model_id"] == winner_spec.model_id
        )
        rankic = selected_trial.get("validation_rankic")
        family_winners.append(
            {
                "family_id": family_id,
                "config_tie_break_order": list(trial_order),
                "selected_config_id": winner_spec.model_id,
                "trial_ledger_semantic_identity": reference["semantic_identity"],
            }
        )
        winner_metrics.append(
            {
                "model_id": family_id,
                "validation_rankic": rankic,
                "validation_rankic_reason": selected_trial["validation_rankic_reason"],
                "validation_rows": selected_trial["validation_rows"],
            }
        )
    family_frame = pd.DataFrame.from_records(winner_metrics)
    selected_family = select_best_validation_trial(
        family_frame,
        trial_id_column="model_id",
        selection_metric="validation_rankic",
        maximize=True,
        validation_metric_columns=("validation_rankic",),
        test_metric_columns=(),
        tie_break_order=tie_break_order,
    )
    selected_spec = winner_specs[selected_family]
    return (
        winner_specs,
        {
            "status": "selected",
            "selection_metric": "validation_rankic",
            "selection_direction": "maximize",
            "validation_start": str(selection_spec.model_selection_validation_start),
            "validation_end": str(selection_spec.model_selection_validation_end),
            "candidate_family_ids": list(model_ids),
            "tie_break_order": list(tie_break_order),
            "selected_family_id": selected_family,
            "selected_model_id": selected_spec.model_id,
            "families": family_winners,
        },
        tuple(trial_ledgers),
        config_indices,
    )


def _a_share_trial_specs(
    base_spec,
    *,
    n_train: int,
    resolve_trial_parameters,
    quarter_id: str,
    root_seed: int,
) -> tuple[object, ...]:
    counts = {
        "ridge_regression": len(RIDGE_LAMBDAS),
        "huber_regression": len(HUBER_LAMBDAS),
        "lasso_regression": len(LASSO_ALPHAS),
        "random_forest_regressor": len(RF_CONFIGS),
        "xgboost_regressor": len(XGB_CONFIGS),
    }
    count = counts.get(base_spec.model_algorithm, 1)
    result = []
    for config_index in range(count):
        model_id = (
            base_spec.model_id
            if count == 1
            else f"{base_spec.model_id}__config_{config_index}"
        )
        trial_seed = derive_seed(
            root_seed,
            "a_share_model_trial",
            base_spec.model_algorithm,
            model_id,
            quarter_id,
            config_index,
        )
        result.append(
            replace(
                base_spec,
                model_id=model_id,
                random_seed=trial_seed,
                hyperparameters=resolve_trial_parameters(
                    base_spec.model_algorithm,
                    n_train=n_train,
                    config_index=config_index,
                ),
            )
        )
    return tuple(result)


def _a_share_trial_tie_break(
    algorithm: str,
    trial_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if algorithm in {"ridge_regression", "huber_regression", "lasso_regression"}:
        return tuple(reversed(trial_ids))
    return trial_ids


def _count_training_rows(
    dataset_spec,
    model_spec,
    *,
    chunks,
    train_end: str,
) -> int:
    read_columns = model_training_read_columns(
        dataset_spec.primary_key,
        model_spec.input_columns,
        label_column=model_spec.label_column,
        sample_filter_mode=model_spec.sample_filter_mode,
    )
    count = 0
    required_input_columns = (
        ()
        if _model_algorithm_facts(model_spec.model_algorithm).input_kind
        == "rank_ensemble_inputs"
        else model_spec.input_columns
    )
    for frame in iter_legal_training_chunks(
        chunks(
            read_columns,
            input_columns=model_spec.input_columns,
            start=None,
            end=train_end,
        ),
        train_end=pd.Timestamp(train_end),
        label_column=model_spec.label_column,
        input_columns=required_input_columns,
    ):
        count += len(frame)
    if count <= 0:
        raise ValueError("model_training_sample_empty")
    return count


@dataclass(frozen=True)
class CryptoFoldModelResult:
    fold: ExperimentFold
    spec: object
    result: fitting.ModelFitResult
    training_assets: tuple[str, ...]
    evaluation_assets: tuple[str, ...]
    trial_reference: dict[str, str]


def build_crypto_fold_models(
    *,
    layout,
    dataset_catalog,
    model_catalog,
    dataset_id,
    model_ids,
    tie_break_order,
    time_window,
    resource_policy,
    protocol_manifest,
    run_id,
    research_protocol_identity,
    model_target_id,
    backend_versions,
    on_validation=None,
):
    from quant_research_platform.research.models.trial_selection import (
        evaluate_model_trial_metrics_from_chunks,
        iter_validation_prediction_chunks,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_EXECUTION_CONTEXT_COLUMNS,
        HFT_MAKER_CALIBRATION_MODEL_IDS,
        build_hft_prediction_catalog,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        protocol as crypto_protocol,
    )

    protocol = crypto_protocol(
        dataset_id, research_plan=protocol_manifest["research_plan"]
    )
    dataset = dataset_catalog.get(dataset_id)
    identity = layout.target_root(
        dataset_id, layout.research_dataset_root(dataset.market_id, dataset_id)
    ).name
    for fold in protocol.research_plan.folds:
        if pd.Timestamp(fold.development_test.end) < pd.Timestamp(
            time_window.start
        ) or pd.Timestamp(fold.development_test.start) > pd.Timestamp(time_window.end):
            continue
        prediction_catalog = build_hft_prediction_catalog(
            train_end=fold.train.end.isoformat(),
            test_start=fold.development_test.start.isoformat(),
            test_end=fold.development_test.end.isoformat(),
        )
        for model_id in model_ids:
            spec = replace(
                model_catalog.get(model_id),
                random_seed=derive_seed(
                    ROOT_SEED, "crypto_model", fold.fold_id, model_id
                ),
            )
            prediction = prediction_catalog.get(f"{model_id}_prediction")
            plan = ExperimentPlan(
                experiment_id=f"experiment_{uuid4().hex}",
                market_id=dataset.market_id,
                research_dataset_id=dataset_id,
                target_id=model_target_id,
                candidate_family_id=model_id,
                data_snapshot_id=identity,
                input_lineage=(identity,),
                research_protocol_identity=research_protocol_identity,
                folds=(fold,),
                final_holdout=protocol.research_plan.final_holdout,
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
            attempt = recorder.begin(spec)
            train_owned = refit_owned = trained = result = None
            metrics = training_metadata = None

            def chunks(window, spec=spec, prediction=prediction):
                source = iter_research_dataset_chunks(
                    layout,
                    market_id=dataset.market_id,
                    dataset_id=dataset_id,
                    time_window=window,
                    columns=(
                        *model_training_read_columns(
                            dataset.primary_key,
                            spec.input_columns,
                            label_column=spec.label_column,
                            sample_filter_mode=spec.sample_filter_mode,
                        ),
                        *(
                            HFT_EXECUTION_CONTEXT_COLUMNS
                            if spec.model_id in HFT_MAKER_CALIBRATION_MODEL_IDS
                            else ()
                        ),
                    ),
                )
                yield from training_scope_chunks(
                    iter_validated_research_input_chunks(
                        source, dataset_spec=dataset, input_columns=spec.input_columns
                    ),
                    asset_ids=prediction.training_asset_ids,
                )

            member = None
            finished = False
            try:
                train_owned = fitting._fit_model_result_owned_from_frames(
                    chunks(fold.train),
                    spec,
                    estimator=build_estimator_from_model_spec(spec),
                    fitting_cutoff=fold.train.end,
                )
                trained = (
                    train_owned.value
                    if isinstance(train_owned, OwnedValue)
                    else train_owned
                )
                training_metadata = dict(trained.metadata)
                validation = iter_validation_prediction_chunks(
                    lambda fold=fold: chunks(fold.validation),
                    prediction_spec=replace(
                        prediction, evaluation_asset_ids=prediction.training_asset_ids
                    ),
                    candidate_spec=spec,
                    model_result=trained,
                    validation_start=pd.Timestamp(fold.validation.start),
                    validation_end=pd.Timestamp(fold.validation.end),
                    include_availability=model_id in HFT_MAKER_CALIBRATION_MODEL_IDS,
                    extra_output_columns=HFT_EXECUTION_CONTEXT_COLUMNS
                    if model_id in HFT_MAKER_CALIBRATION_MODEL_IDS
                    else (),
                )

                def observed_validation(
                    validation=validation, model_id=model_id, fold=fold, spec=spec
                ):
                    with closing_iterator(iter(validation)) as validation_source:
                        for item in validation_source:
                            if (
                                on_validation is not None
                                and model_id in HFT_MAKER_CALIBRATION_MODEL_IDS
                            ):
                                on_validation(
                                    fold,
                                    spec,
                                    item.value
                                    if isinstance(item, OwnedValue)
                                    else item,
                                )
                            yield item

                metrics = evaluate_model_trial_metrics_from_chunks(
                    observed_validation(),
                    model_id=model_id,
                    label_column=spec.label_column,
                    prediction_column=prediction.prediction_column,
                )
                if metrics["validation_rows"] <= 0 or not pd.notna(
                    metrics["validation_mse"]
                ):
                    raise ValueError("crypto_validation_sample_empty")
                trained = None
                if isinstance(train_owned, OwnedValue):
                    train_owned.release(owner="research.model.crypto.validation")
                train_owned = None

                def refit_chunks(fold=fold):
                    yield from chunks(fold.train)
                    yield from chunks(fold.validation)

                refit_owned = fitting._fit_model_result_owned_from_frames(
                    refit_chunks(),
                    spec,
                    estimator=build_estimator_from_model_spec(spec),
                    fitting_cutoff=fold.validation.end,
                )
                result = (
                    refit_owned.value
                    if isinstance(refit_owned, OwnedValue)
                    else refit_owned
                )
                recorder.finish(
                    attempt, metrics=metrics, model_metadata=training_metadata
                )
                finished = True
                member = CryptoFoldModelResult(
                    fold,
                    spec,
                    result,
                    prediction.training_asset_ids,
                    prediction.evaluation_asset_ids,
                    recorder.reference(),
                )
                yield (
                    OwnedValue(member, refit_owned.lease)
                    if isinstance(refit_owned, OwnedValue)
                    else member
                )
            except BaseException as error:
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
                from quant_research_platform.contracts.validation import (
                    clear_exception_frames,
                )

                clear_exception_frames(error)
                raise
            finally:
                trained = member = result = None
                for owner in (train_owned, refit_owned):
                    if isinstance(owner, OwnedValue) and owner.lease.active:
                        primary = sys.exception()
                        try:
                            owner.release(owner="research.model.crypto")
                        except BaseException as cleanup:
                            if primary is None:
                                raise
                            primary.add_note(
                                "crypto model release also failed: "
                                f"{type(cleanup).__name__}"
                            )
                train_owned = refit_owned = None

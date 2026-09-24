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
from quant_research_platform.contracts.experiments import (
    ExperimentFold,
    ExperimentPlan,
    legal_training_rows,
    training_availability_columns,
)
from quant_research_platform.contracts.models import (
    _model_algorithm_facts,
    model_spec_payload,
)
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
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
from quant_research_platform.research.features.transforms import (
    canonical_feature_validity_columns,
)
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
    ModelSampleCoverage,
    crypto_validation_coverage_scope,
    model_validation_coverage_scope,
    model_validation_inputs,
    select_best_validation_trial,
    select_prediction_model_spec,
)
from quant_research_platform.research.protocols.a_share import (
    model_folds,
    quarter_id_for_fold,
)
from quant_research_platform.research.randomness import ROOT_SEED, derive_seed
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    resolve_columns_from_schema,
)
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
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
    on_validation_coverage,
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
                    target_chunk_bytes=max(1, resource_policy.planning_headroom_bytes),
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
        research_configuration=protocol_manifest["research_configuration"],
        on_validation_coverage=on_validation_coverage,
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
    research_configuration,
    on_validation_coverage,
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
        training_counts = {}
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
                family_specs = tuple(
                    model_catalog.get(model_id) for model_id in model_ids
                )
                common_inputs = model_validation_inputs(family_specs)
                label = family_specs[0].label_column
                availability = training_availability_columns(label)[0]
                dataset_identity = layout.target_root(
                    dataset_spec.data_id,
                    layout.research_dataset_root(
                        dataset_spec.market_id, dataset_spec.data_id
                    ),
                ).name
                coverage = ModelSampleCoverage(
                    model_validation_coverage_scope(
                        dataset_identity, quarter, family_specs
                    )
                )

                def coverage_frames(
                    common_inputs=common_inputs,
                    label=label,
                    availability=availability,
                    quarter=quarter,
                    coverage=coverage,
                ):
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
                    source = chunks(
                        columns,
                        input_columns=common_inputs,
                        start=quarter.validation.start,
                        end=quarter.validation.end,
                    )
                    with closing_iterator(source):
                        for frame in source:
                            with current_runtime_resource_policy().lease(
                                "research.model.validation_coverage",
                                max(1, 3 * pandas_frame_payload_bytes(frame)),
                            ):
                                legal = legal_training_rows(
                                    frame,
                                    fitting_cutoff=quarter.validation.end,
                                    availability_column=availability,
                                    feature_columns=common_inputs,
                                    target_column=label,
                                )
                                if not legal.empty:
                                    keys = legal.loc[:, ["timestamp", "asset_id"]]
                                    coverage.add(keys)
                                    yield keys
                                    del keys
                                del legal

                coverage_path = on_validation_coverage(quarter, coverage_frames())
                if not coverage.row_count:
                    raise ValueError("empty_model_selection_validation_window")
                winner_specs, selection_metadata, trial_ledgers, config_indices = (
                    _select_model_family_winners(
                        selection_spec=selection_spec,
                        training_counts=training_counts,
                        validation_coverage=coverage,
                        validation_input_columns=common_inputs,
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
                        research_configuration=research_configuration,
                    )
                )
                selection_metadata["common_validation_coverage"] = {
                    "path": coverage_path,
                    "semantic_identity": coverage.semantic_identity,
                    "row_count": coverage.row_count,
                }
                for family_id, reference in zip(model_ids, trial_ledgers, strict=True):
                    selected_spec = winner_specs[family_id]
                    selected_configuration = next(
                        item
                        for item in research_configuration["models"]
                        if item["model_id"] == family_id
                    )
                    selection_plan, _selection_records = (
                        read_model_trial_ledger_reference(layout, reference)
                    )
                    refit_recorder = refit_attempt = fit_coverage = None

                    def begin_refit(
                        spec,
                        training,
                        *,
                        selection_plan=selection_plan,
                        quarter=quarter,
                        reference=reference,
                    ):
                        nonlocal refit_recorder, refit_attempt, fit_coverage
                        fit_coverage = _refit_sample_coverage(
                            selection_plan,
                            spec,
                            training,
                            fitting_cutoff=quarter.validation.end,
                        )
                        selection_recorder = TrialRecorder(
                            layout, run_id, selection_plan
                        )
                        refit_recorder, refit_attempt = selection_recorder.begin_refit(
                            spec, common_coverage_id=fit_coverage.semantic_identity
                        )

                    try:
                        model_result, model_lease = _fit_selected_model_result(
                            dataset_spec=dataset_spec,
                            selected_spec=selected_spec,
                            training_counts=training_counts,
                            chunks=chunks,
                            fitting_cutoff=time_value_isoformat(
                                quarter.validation.end, date_only=True
                            ),
                            quarter_id=quarter_id_for_fold(quarter),
                            root_seed=root_seed,
                            candidate_parameters=selected_configuration.get(
                                "parameter_candidates",
                                [selected_configuration["hyperparameters"]],
                            )[config_indices[family_id]],
                            on_fit_start=begin_refit,
                        )
                        model_result.metadata["fit_coverage_id"] = (
                            fit_coverage.semantic_identity
                        )
                        refit_recorder.finish(
                            refit_attempt, model_metadata=model_result.metadata
                        )
                    except BaseException as primary:
                        if refit_attempt is not None:
                            try:
                                refit_recorder.finish(refit_attempt, error=primary)
                            except BaseException as record_error:
                                primary.add_note(
                                    "refit trial recording also failed: "
                                    + type(record_error).__name__
                                )
                        raise
                    item = QuarterlyModelResult(
                        quarter=quarter,
                        model_id=selected_spec.model_id,
                        model_result=model_result,
                        family_id=family_id,
                        selection_metadata=selection_metadata,
                        trial_ledgers=(reference, refit_recorder.reference()),
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


def _refit_sample_coverage(plan, spec, training, *, fitting_cutoff, asset_ids=()):
    coverage = ModelSampleCoverage(
        {
            "dataset_identity": plan.data_snapshot_id,
            "fold_id": plan.folds[0].fold_id,
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
    dataset_spec,
    selected_spec,
    chunks,
    training_counts,
    fitting_cutoff: str,
    quarter_id: str,
    root_seed: int,
    candidate_parameters,
    on_fit_start=None,
) -> tuple[fitting.ModelFitResult, MemoryLease | None]:
    n_refit = _count_training_rows(
        dataset_spec,
        selected_spec,
        chunks=chunks,
        train_end=fitting_cutoff,
        cache=training_counts,
    )
    refit_spec = _refit_model_spec(
        selected_spec,
        n_train=n_refit,
        quarter_id=quarter_id,
        root_seed=root_seed,
        candidate_parameters=candidate_parameters,
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
        on_fit_start=on_fit_start,
    )
    model_result = (
        owned_model.value if isinstance(owned_model, OwnedValue) else owned_model
    )
    return model_result, owned_model.lease if isinstance(
        owned_model, OwnedValue
    ) else None


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


def _select_model_family_winners(
    *,
    selection_spec: PredictionSpec,
    training_counts,
    validation_coverage,
    validation_input_columns,
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
    research_configuration,
):
    family_winners: list[dict[str, object]] = []
    trial_ledgers: list[dict[str, str]] = []
    winner_specs: dict[str, object] = {}
    config_indices: dict[str, int] = {}
    winner_metrics: list[dict[str, object]] = []
    for family_id in model_ids:
        base_spec = model_catalog.get(family_id)
        n_train = _count_training_rows(
            dataset_spec,
            base_spec,
            chunks=chunks,
            train_end=selection_spec.train_end,
            cache=training_counts,
        )
        selected = next(
            item
            for item in research_configuration["models"]
            if item["model_id"] == family_id
        )
        parameters = selected.get("parameter_candidates", [selected["hyperparameters"]])
        trial_specs = _selection_model_specs(
            base_spec,
            parameters,
            n_train=n_train,
            root_seed=root_seed,
            quarter_id=quarter_id,
        )
        trial_catalog = ModelCatalog(trial_specs)
        trial_ids = tuple(spec.model_id for spec in trial_specs)
        trial_order = trial_ids
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
            stage="selection",
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
            validation_input_columns=validation_input_columns,
            validation_coverage=validation_coverage,
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


def _count_training_rows(
    dataset_spec,
    model_spec,
    *,
    chunks,
    train_end: str,
    cache,
) -> int:
    scope = (
        frozenset(model_spec.input_columns),
        model_spec.label_column,
        model_spec.sample_filter_mode,
        model_spec.max_train_rows,
        train_end,
        _model_algorithm_facts(model_spec.model_algorithm).input_kind,
    )
    if scope in cache:
        return cache[scope]
    read_columns = model_training_read_columns(
        dataset_spec.primary_key,
        model_spec.input_columns,
        label_column=model_spec.label_column,
        sample_filter_mode=model_spec.sample_filter_mode,
    )
    count = 0
    required_input_columns = (
        canonical_feature_validity_columns(model_spec.input_columns)
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
    cache[scope] = (
        min(count, model_spec.max_train_rows)
        if model_spec.max_train_rows is not None
        else count
    )
    return cache[scope]


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
        build_hft_prediction_catalog,
        research_chronology,
    )

    folds, holdout = research_chronology(protocol_manifest["research_plan"])

    dataset = dataset_catalog.get(dataset_id)
    identity = layout.target_root(
        dataset_id, layout.research_dataset_root(dataset.market_id, dataset_id)
    ).name
    for fold in folds:
        if pd.Timestamp(fold.development_test.end) < pd.Timestamp(
            time_window.start
        ) or pd.Timestamp(fold.development_test.start) > pd.Timestamp(time_window.end):
            continue
        prediction_catalog = build_hft_prediction_catalog(
            configuration=protocol_manifest["research_configuration"],
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
                final_holdout=holdout,
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
                        *HFT_EXECUTION_CONTEXT_COLUMNS,
                    ),
                )
                yield from training_scope_chunks(
                    iter_validated_research_input_chunks(
                        source, dataset_spec=dataset, input_columns=spec.input_columns
                    ),
                    asset_ids=prediction.training_asset_ids,
                )

            validation_coverage = ModelSampleCoverage(
                crypto_validation_coverage_scope(
                    plan, spec, prediction.training_asset_ids
                )
            )
            with closing_iterator(chunks(fold.validation)) as validation_source:
                for frame in validation_source:
                    with resource_policy.lease(
                        "research.model.validation_coverage",
                        max(1, 2 * pandas_frame_payload_bytes(frame)),
                    ):
                        legal = legal_training_rows(
                            frame,
                            fitting_cutoff=fold.validation.end,
                            availability_column=training_availability_columns(
                                spec.label_column
                            )[0],
                            feature_columns=spec.input_columns,
                            target_column=spec.label_column,
                        )
                        validation_coverage.add(legal)
                        del legal
            if validation_coverage.row_count == 0:
                raise PortfolioTargetBlocked(
                    "crypto_model_validation_legal_rows_missing"
                )
            attempt = recorder.begin(
                spec, common_coverage_id=validation_coverage.semantic_identity
            )
            member = None
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
                    asset_ids=prediction.training_asset_ids,
                )
                refit_recorder, refit_attempt = recorder.begin_refit(
                    actual_spec, common_coverage_id=fit_coverage.semantic_identity
                )

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
                    include_availability=True,
                    extra_output_columns=HFT_EXECUTION_CONTEXT_COLUMNS,
                )

                def observed_validation(
                    validation=validation, model_id=model_id, fold=fold, spec=spec
                ):
                    with closing_iterator(iter(validation)) as validation_source:
                        for item in validation_source:
                            if on_validation is not None:
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
                    expected_coverage=validation_coverage,
                )
                if metrics["validation_rows"] <= 0 or not pd.notna(
                    metrics["validation_mse"]
                ):
                    raise ValueError("crypto_validation_sample_empty")
                trained = None
                if isinstance(train_owned, OwnedValue):
                    train_owned.release(owner="research.model.crypto.validation")
                train_owned = None

                recorder.finish(
                    attempt, metrics=metrics, model_metadata=training_metadata
                )
                finished = True

                def refit_chunks(fold=fold):
                    yield from chunks(fold.train)
                    yield from chunks(fold.validation)

                refit_owned = fitting._fit_model_result_owned_from_frames(
                    refit_chunks(),
                    spec,
                    estimator=build_estimator_from_model_spec(spec),
                    fitting_cutoff=fold.validation.end,
                    on_fit_start=begin_crypto_refit,
                )
                result = (
                    refit_owned.value
                    if isinstance(refit_owned, OwnedValue)
                    else refit_owned
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
                yield (
                    OwnedValue(member, refit_owned.lease)
                    if isinstance(refit_owned, OwnedValue)
                    else member
                )
            except BaseException as error:
                if refit_attempt is not None and not refit_finished:
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

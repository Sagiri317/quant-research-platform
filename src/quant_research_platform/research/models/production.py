"""Production model selection and quarterly refit owner."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.models import _model_algorithm_facts
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.datasets.validation import (
    iter_validated_research_input_chunks,
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
    select_best_validation_trial,
    select_prediction_model_spec,
)
from quant_research_platform.research.protocols.a_share import (
    model_folds,
    quarter_id_for_fold,
)
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.layout import StorageLayout


@dataclass(frozen=True)
class QuarterlyModelResult:
    quarter: ExperimentFold
    model_id: str
    model_result: fitting.ModelFitResult


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
) -> Iterable[QuarterlyModelResult | OwnedValue[QuarterlyModelResult]]:
    if not model_ids or set(model_ids) != set(tie_break_order):
        raise ValueError("model_selection_model_ids_invalid")
    dataset_spec = dataset_catalog.get(dataset_id)
    if protocol_manifest.get("protocol_id") != "a_share_low_frequency":
        raise ValueError("a_share_protocol_required")
    chronology = protocol_manifest.get("model_chronology")
    formal_window = protocol_manifest.get("formal_window")
    development_window = protocol_manifest.get("development_window")
    if (
        type(chronology) is not dict
        or set(chronology) != {"validation_months"}
        or type(formal_window) is not dict
        or type(development_window) is not dict
    ):
        raise ValueError("model_chronology_missing")
    root_seed = protocol_manifest.get("root_seed")
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("model_root_seed_missing")
    dataset_read_window = TimeWindow(
        formal_window["start"],
        development_window["end"],
    )

    def chunks(
        read_columns: tuple[str, ...],
        *,
        input_columns: tuple[str, ...],
    ) -> Iterable[pd.DataFrame]:
        yield from iter_validated_research_input_chunks(
            iter_research_dataset_chunks(
                layout,
                market_id=dataset_spec.market_id,
                dataset_id=dataset_spec.data_id,
                time_window=dataset_read_window,
                columns=read_columns,
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
        chronology=chronology,
        formal_window=formal_window,
        development_window=development_window,
        root_seed=root_seed,
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
    chronology: Mapping[str, object],
    formal_window: Mapping[str, object],
    development_window: Mapping[str, object],
    root_seed: int,
) -> Iterable[QuarterlyModelResult | OwnedValue[QuarterlyModelResult]]:
    sessions = _dataset_sessions(
        layout,
        dataset_spec=dataset_spec,
        time_window=TimeWindow(
            formal_window["start"],
            development_window["end"],
        ),
    )
    quarters = model_folds(
        sessions,
        chronology=chronology,
        formal_window=formal_window,
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
                    train_end=_date_text(quarter.train.end),
                    test_start=_date_text(quarter.development_test.start),
                    test_end=_date_text(quarter.development_test.end),
                    candidate_model_ids=model_ids[1:],
                    model_selection_validation_start=_date_text(
                        quarter.validation.start
                    ),
                    model_selection_validation_end=_date_text(quarter.validation.end),
                )
                selected_spec, selection_metadata = _select_model_family_winners(
                    selection_spec=selection_spec,
                    dataset_spec=dataset_spec,
                    model_catalog=model_catalog,
                    chunks=chunks,
                    model_ids=model_ids,
                    tie_break_order=tie_break_order,
                    quarter_id=quarter_id_for_fold(quarter),
                    root_seed=root_seed,
                )
                model_result, model_lease = _fit_selected_model_result(
                    dataset_spec=dataset_spec,
                    selected_spec=selected_spec,
                    chunks=chunks,
                    fitting_cutoff=_date_text(quarter.validation.end),
                    quarter_id=quarter_id_for_fold(quarter),
                    root_seed=root_seed,
                    selection_metadata=dict(selection_metadata),
                )
                item = QuarterlyModelResult(
                    quarter=quarter,
                    model_id=selected_spec.model_id,
                    model_result=model_result,
                )
                owned_item = (
                    OwnedValue(item, model_lease) if model_lease is not None else item
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


def _dataset_sessions(
    layout: StorageLayout,
    *,
    dataset_spec,
    time_window: TimeWindow,
) -> pd.DatetimeIndex:
    values: set[pd.Timestamp] = set()
    for frame in iter_research_dataset_chunks(
        layout,
        market_id=dataset_spec.market_id,
        dataset_id=dataset_spec.data_id,
        time_window=time_window,
        columns=("timestamp",),
    ):
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
    selection_metadata: Mapping[str, object],
) -> tuple[fitting.ModelFitResult, MemoryLease | None]:
    refit_seed = derive_seed(
        root_seed,
        "a_share_model_refit",
        selected_spec.model_algorithm,
        selected_spec.model_id,
        quarter_id,
    )
    refit_spec = replace(selected_spec, random_seed=refit_seed)
    refit_read_columns = model_training_read_columns(
        dataset_spec.primary_key,
        refit_spec.input_columns,
        label_column=refit_spec.label_column,
        sample_filter_mode=refit_spec.sample_filter_mode,
    )
    owned_model = fitting._fit_model_result_owned_from_frames(
        training_scope_chunks(
            chunks(refit_read_columns, input_columns=refit_spec.input_columns),
        ),
        refit_spec,
        estimator=build_estimator_from_model_spec(refit_spec),
        fitting_cutoff=fitting_cutoff,
    )
    model_result = (
        owned_model.value if isinstance(owned_model, OwnedValue) else owned_model
    )
    return (
        fitting.ModelFitResult(
            coefficients=model_result.coefficients,
            metadata={
                **model_result.metadata,
                "model_selection": dict(selection_metadata),
            },
            prediction_model=model_result.prediction_model,
        ),
        owned_model.lease if isinstance(owned_model, OwnedValue) else None,
    )


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


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
):
    from quant_research_platform.research.protocols.a_share import (
        resolve_trial_parameters,
    )

    family_winners: list[dict[str, object]] = []
    winner_specs: dict[str, object] = {}
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
        winner_spec, metadata = select_prediction_model_spec(
            prediction_spec=family_prediction,
            dataset_spec=dataset_spec,
            model_catalog=trial_catalog,
            chunks=chunks,
            validation_input_columns=family_validation_inputs,
        )
        winner_specs[family_id] = winner_spec
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
                "trials": metadata["trial_metrics"],
                "validation_rows": selected_trial["validation_rows"],
                "selection_metric_value": rankic,
                "selection_metric_reason": selected_trial["validation_rankic_reason"],
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
    return selected_spec, {
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
    }


def _a_share_trial_specs(
    base_spec,
    *,
    n_train: int,
    resolve_trial_parameters,
    quarter_id: str,
    root_seed: int,
) -> tuple[object, ...]:
    counts = {
        "ridge_regression": 4,
        "huber_regression": 4,
        "lasso_regression": 4,
        "random_forest_regressor": 3,
        "xgboost_regressor": 3,
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
        chunks(read_columns, input_columns=model_spec.input_columns),
        train_end=pd.Timestamp(train_end),
        label_column=model_spec.label_column,
        input_columns=required_input_columns,
    ):
        count += len(frame)
    if count <= 0:
        raise ValueError("model_training_sample_empty")
    return count

"""Production model selection and quarterly refit owner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    select_best_validation_trial,
)
from quant_research_platform.research.predictions import runner
from quant_research_platform.research.walk_forward import (
    WalkForwardQuarter,
    a_share_walk_forward_quarters,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.layout import StorageLayout


@dataclass(frozen=True)
class ModelSelectionResult:
    model_id: str
    quarterly_results: tuple[QuarterlyModelResult, ...]


@dataclass(frozen=True)
class QuarterlyModelResult:
    quarter: WalkForwardQuarter
    model_id: str
    model_result: fitting.ModelFitResult


def build_model_selection_result(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    model_catalog: ModelCatalog,
    dataset_id: str,
    artifact_id: str,
    model_ids: tuple[str, ...],
    tie_break_order: tuple[str, ...],
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
    protocol_manifest: Mapping[str, object],
) -> ModelSelectionResult:
    if not model_ids or set(model_ids) != set(tie_break_order):
        raise ValueError("model_selection_model_ids_invalid")
    dataset_spec = dataset_catalog.get(dataset_id)
    if protocol_manifest.get("protocol_id") != "a_share_low_frequency":
        raise ValueError("a_share_protocol_required")
    dataset_read_window = TimeWindow("2014-01-01", "2023-12-31")

    def chunks(
        read_columns: tuple[str, ...],
        *,
        input_columns: tuple[str, ...],
    ) -> Iterable[pd.DataFrame]:
        yield from runner._validated_prediction_chunks(  # noqa: SLF001
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
        artifact_id=artifact_id,
        model_ids=model_ids,
        tie_break_order=tie_break_order,
        time_window=time_window,
        chunks=chunks,
    )


def _build_a_share_quarterly_models(
    *,
    layout: StorageLayout,
    dataset_spec,
    model_catalog: ModelCatalog,
    dataset_id: str,
    artifact_id: str,
    model_ids: tuple[str, ...],
    tie_break_order: tuple[str, ...],
    time_window: TimeWindow,
    chunks,
) -> ModelSelectionResult:
    sessions = _dataset_sessions(
        layout,
        dataset_spec=dataset_spec,
        time_window=TimeWindow("2014-01-01", "2023-12-31"),
    )
    quarters = a_share_walk_forward_quarters(sessions)
    requested_start = pd.Timestamp(time_window.start)
    requested_end = pd.Timestamp(time_window.end)
    selected_quarters = tuple(
        quarter
        for quarter in quarters
        if pd.Timestamp(quarter.test_window[1]) >= requested_start
        and pd.Timestamp(quarter.test_window[0]) <= requested_end
    )
    if not selected_quarters:
        raise ValueError("a_share_walk_forward_window_empty")
    results: list[QuarterlyModelResult] = []
    for quarter in selected_quarters:
        selection_spec = PredictionSpec(
            prediction_id=(
                f"{dataset_id}_model_selection_{quarter.quarter_id.lower()}"
            ),
            model_id=model_ids[0],
            research_dataset_id=dataset_id,
            train_end=_date_text(quarter.training_window[1]),
            test_start=_date_text(quarter.test_window[0]),
            test_end=_date_text(quarter.test_window[1]),
            candidate_model_ids=model_ids[1:],
            model_selection_validation_start=_date_text(quarter.validation_window[0]),
            model_selection_validation_end=_date_text(quarter.validation_window[1]),
        )
        selected_spec, selection_metadata = _select_model_family_winners(
            selection_spec=selection_spec,
            dataset_spec=dataset_spec,
            model_catalog=model_catalog,
            chunks=chunks,
            model_ids=model_ids,
            tie_break_order=tie_break_order,
        )
        model_result = _fit_selected_model_result(
            dataset_spec=dataset_spec,
            selected_spec=selected_spec,
            chunks=chunks,
            fitting_cutoff=_date_text(quarter.validation_window[1]),
            selection_metadata={
                **selection_metadata,
                "quarter_id": quarter.quarter_id,
                "training_window": _window_payload(quarter.training_window),
                "validation_window": _window_payload(quarter.validation_window),
                "prediction_window": _window_payload(quarter.test_window),
            },
        )
        results.append(
            QuarterlyModelResult(
                quarter=quarter,
                model_id=selected_spec.model_id,
                model_result=model_result,
            )
        )
    return ModelSelectionResult(
        model_id=artifact_id,
        quarterly_results=tuple(results),
    )


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
    selection_metadata: Mapping[str, object],
) -> fitting.ModelFitResult:
    refit_read_columns = runner._prediction_train_read_columns(  # noqa: SLF001
        dataset_spec.primary_key,
        selected_spec.input_columns,
        label_column=selected_spec.label_column,
        sample_filter_mode=selected_spec.sample_filter_mode,
    )
    owned_model = fitting._fit_model_result_owned_from_frames(  # noqa: SLF001
        runner._prediction_train_chunks(  # noqa: SLF001
            chunks(refit_read_columns, input_columns=selected_spec.input_columns),
            train_end=pd.Timestamp(fitting_cutoff),
            label_column=selected_spec.label_column,
            input_columns=selected_spec.input_columns,
        ),
        selected_spec,
        estimator=build_estimator_from_model_spec(selected_spec),
        fitting_cutoff=fitting_cutoff,
    )
    model_result = (
        owned_model.value if isinstance(owned_model, OwnedValue) else owned_model
    )
    if isinstance(owned_model, OwnedValue):
        owned_model.lease.close()
    return fitting.ModelFitResult(
        coefficients=model_result.coefficients,
        metadata={
            **model_result.metadata,
            "model_selection": dict(selection_metadata),
        },
        prediction_model=model_result.prediction_model,
    )


def _window_payload(window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, str]:
    return {
        "start": _date_text(window[0]),
        "end": _date_text(window[1]),
    }


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
):
    from quant_research_platform.research.protocols.a_share import (
        resolve_trial_parameters,
    )

    family_winners: list[dict[str, object]] = []
    winner_specs: dict[str, object] = {}
    winner_metrics: list[dict[str, object]] = []
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
        winner_spec, metadata = runner._select_prediction_model_spec(  # noqa: SLF001
            prediction_spec=family_prediction,
            dataset_spec=dataset_spec,
            model_catalog=trial_catalog,
            chunks=chunks,
        )
        winner_specs[family_id] = winner_spec
        selected_trial = next(
            item
            for item in metadata["trial_metrics"]
            if item["model_id"] == winner_spec.model_id
        )
        rankic = selected_trial.get("validation_rankic")
        if rankic is None:
            raise ValueError("model_family_validation_rankic_undefined")
        family_winners.append(
            {
                "family_id": family_id,
                "selected_config_id": winner_spec.model_id,
                "validation_rankic": rankic,
                "trial_metrics": metadata["trial_metrics"],
                "config_tie_break_order": list(trial_order),
            }
        )
        winner_metrics.append(
            {
                "model_id": family_id,
                "validation_rankic": rankic,
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
        "candidate_model_ids": list(model_ids),
        "selected_model_id": selected_spec.model_id,
        "selected_family_id": selected_family,
        "tie_break_order": list(tie_break_order),
        "family_winners": family_winners,
        "candidate_count": len(model_ids),
    }


def _a_share_trial_specs(
    base_spec,
    *,
    n_train: int,
    resolve_trial_parameters,
) -> tuple[object, ...]:
    counts = {
        "ridge_regression": 4,
        "huber_regression": 4,
        "lasso_regression": 4,
        "random_forest_regressor": 3,
        "xgboost_regressor": 3,
    }
    count = counts.get(base_spec.model_algorithm, 1)
    return tuple(
        replace(
            base_spec,
            model_id=(
                base_spec.model_id
                if count == 1
                else f"{base_spec.model_id}__config_{config_index}"
            ),
            hyperparameters=resolve_trial_parameters(
                base_spec.model_algorithm,
                n_train=n_train,
                config_index=config_index,
            ),
        )
        for config_index in range(count)
    )


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
    read_columns = runner._prediction_train_read_columns(  # noqa: SLF001
        dataset_spec.primary_key,
        model_spec.input_columns,
        label_column=model_spec.label_column,
        sample_filter_mode=model_spec.sample_filter_mode,
    )
    count = 0
    for frame in runner._prediction_train_chunks(  # noqa: SLF001
        chunks(read_columns, input_columns=model_spec.input_columns),
        train_end=pd.Timestamp(train_end),
        label_column=model_spec.label_column,
        input_columns=model_spec.input_columns,
    ):
        count += len(frame)
    if count <= 0:
        raise ValueError("model_training_sample_empty")
    return count

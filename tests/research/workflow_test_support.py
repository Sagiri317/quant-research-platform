from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd

from quant_research_platform.contracts.experiments import (
    ExperimentFold,
    ExperimentPlan,
    TrialRecord,
    trial_ledger_semantic_identity,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.models.fitting import ModelFitResult
from quant_research_platform.research.models.production import (
    QuarterlyModelResult,
    _refit_model_spec,
    _selection_model_specs,
)
from quant_research_platform.research.models.rank_ensemble import (
    RankEnsemblePredictor,
)
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
)
from quant_research_platform.research.models.trial_selection import (
    ModelSampleCoverage,
    model_validation_coverage_scope,
)
from quant_research_platform.research.protocols.a_share import (
    quarter_id_for_fold,
)
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.storage.experiments import (
    append_trial_record,
    read_model_trial_ledger_reference,
)
from quant_research_platform.storage.layout import StorageLayout

QUARTERLY_FIXTURE_PROTOCOL = {
    "fixture": True,
    "research_configuration": {
        "models": [
            {"model_id": algorithm, "hyperparameters": {}}
            for algorithm in ("linear_regression",)
        ]
    },
}


def _column_predictor(column: str, offset: float = 0.0) -> LinearScaledPredictor:
    return LinearScaledPredictor(
        (offset, 1.0), ModelMatrixScaler((column,), (0.0,), (1.0,))
    )


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
        "training_weighting": {"method": "equal_timestamp", "half_life": None},
        "scaler": None,
        "feature_importance": {column: 1.0 for column in columns},
        "fit_cutoff": "2018-12-31",
    }
    coverage = ModelSampleCoverage(
        {"fixture_training": model_id, "inputs": list(columns)}
    )
    coverage.add(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [result["train_start"], result["train_end"]]
                ),
                "asset_id": ["asset_0", "asset_0"],
            }
        )
    )
    result["fit_coverage_id"] = coverage.semantic_identity
    if rank_parameters is not None:
        result["rank_parameters"] = rank_parameters
    return result


def fixture_validation_keys(fold):
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([fold.validation.start] * 2),
            "asset_id": ["asset_0", "asset_1"],
        }
    )


def fixture_validation_coverage(fold, specs, data_identity="d" * 64):
    coverage = ModelSampleCoverage(
        model_validation_coverage_scope(data_identity, fold, specs)
    )
    coverage.add(fixture_validation_keys(fold))
    return {
        "path": f"models/{quarter_id_for_fold(fold)}/validation-coverage.parquet",
        "semantic_identity": coverage.semantic_identity,
        "row_count": coverage.row_count,
    }


def fixture_quarterly_results(results, *, validation_keys=None):
    def produce(*, on_validation_coverage):
        written = set()
        with closing_iterator(iter(results)) as source:
            for item in source:
                quarterly = item.value
                quarter_id = quarter_id_for_fold(quarterly.quarter)
                if quarter_id not in written:
                    on_validation_coverage(
                        quarterly.quarter,
                        iter(
                            (
                                fixture_validation_keys(quarterly.quarter)
                                if validation_keys is None
                                else validation_keys,
                            )
                        ),
                    )
                    written.add(quarter_id)
                yield item

    return produce


def fixture_refit_reference(
    layout, reference, metadata, *, candidate_parameters=None, fit_keys=None
):
    selection_plan, records = read_model_trial_ledger_reference(layout, reference)
    value = records[0].payload()["normalized_parameters"]
    spec = ModelSpec(
        **{
            **value,
            "input_factor_columns": tuple(value["input_factor_columns"]),
            "input_feature_columns": tuple(value["input_feature_columns"]),
        }
    )
    refit = _refit_model_spec(
        spec,
        candidate_parameters=(
            dict(spec.hyperparameters)
            if candidate_parameters is None
            else candidate_parameters
        ),
        n_train=metadata["rows_after_input_label_dropna"],
        quarter_id=quarter_id_for_fold(selection_plan.folds[0]),
        root_seed=selection_plan.random_seed,
    )
    coverage = ModelSampleCoverage({"fixture_refit": model_spec_payload(refit)})
    if fit_keys is None:
        fit_keys = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    metadata["train_start"],
                    metadata["train_end"],
                    periods=metadata["rows_after_input_label_dropna"],
                ),
                "asset_id": ["asset_0"] * metadata["rows_after_input_label_dropna"],
            }
        )
    coverage.add(fit_keys)
    metadata["fit_coverage_id"] = coverage.semantic_identity
    selection_recorder = TrialRecorder(layout, reference["run_id"], selection_plan)
    recorder, attempted = selection_recorder.begin_refit(
        refit, common_coverage_id=coverage.semantic_identity
    )
    recorder.finish(attempted, model_metadata=metadata)
    return recorder.reference()


def _selection_metadata(
    *,
    family_id: str,
    model_id: str,
    columns: tuple[str, ...],
) -> dict[str, object]:
    coverage = ModelSampleCoverage({"fixture_selection_inputs": list(columns)})
    coverage.add(fixture_validation_keys(_fold(1)))
    return {
        "common_validation_coverage": {
            "path": "models/2019Q1/validation-coverage.parquet",
            "semantic_identity": coverage.semantic_identity,
            "row_count": coverage.row_count,
        },
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
                "trial_ledger_semantic_identity": "0" * 64,
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


def _trial_reference(
    layout: StorageLayout,
    fold: ExperimentFold,
    *,
    family_id: str,
    model_id: str,
    target_id: str = "model-artifact",
    dataset_id: str = "fixture",
    data_identity: str = "d" * 64,
    protocol: dict[str, object] | None = None,
    rankic: float = 0.2,
    model_spec: ModelSpec | None = None,
    root_seed: int = 11,
    validation_specs=None,
) -> dict[str, str]:
    base = model_spec or ModelSpec(
        model_id=model_id,
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    trial_spec = replace(
        base,
        random_seed=derive_seed(
            root_seed,
            "a_share_model_trial",
            base.model_algorithm,
            model_id,
            quarter_id_for_fold(fold),
            0,
        ),
        hyperparameters=dict(base.hyperparameters),
    )
    plan = ExperimentPlan(
        stage="selection",
        experiment_id=f"experiment_{uuid4().hex}",
        market_id=MarketId("a_share"),
        research_dataset_id=dataset_id,
        target_id=target_id,
        candidate_family_id=family_id,
        data_snapshot_id=data_identity,
        input_lineage=(data_identity,),
        research_protocol_identity=canonical_json_hash(
            QUARTERLY_FIXTURE_PROTOCOL if protocol is None else protocol
        ),
        folds=(fold,),
        final_holdout=TimeWindow(
            f"{pd.Timestamp(fold.development_test.end).year + 1}-01-01",
            f"{pd.Timestamp(fold.development_test.end).year + 1}-12-31",
        ),
        selection_metric="validation_rankic",
        selection_direction="maximize",
        candidate_ids=(model_id,),
        trial_budget=1,
        random_seed=root_seed,
        tie_break_order=(model_id,),
        backend_versions={"internal": "test"},
        thread_settings={"native_threads": 1},
        candidate_parameters={model_id: model_spec_payload(trial_spec)},
    )
    record = TrialRecord(
        event_id=f"event_{uuid4().hex}",
        run_id="run_fixture",
        attempt_index=1,
        experiment_id=plan.experiment_id,
        protocol_hash=plan.protocol_hash,
        candidate_family_id=family_id,
        candidate_id=model_id,
        trial_id=f"trial_{uuid4().hex}",
        fold_id=fold.fold_id,
        market_id="a_share",
        research_dataset_id=dataset_id,
        target_id=target_id,
        horizon_id="label",
        common_coverage_id=fixture_validation_coverage(
            fold,
            (base,) if validation_specs is None else validation_specs,
            data_identity,
        )["semantic_identity"],
        random_seed=trial_spec.random_seed,
        normalized_parameters=model_spec_payload(trial_spec),
        status="completed",
        metrics={
            "validation_rankic": rankic,
            "validation_ic": 0.1,
            "validation_mse": 1.0,
            "validation_rows": 2,
            "n_train": 2,
            "validation_rankic_reason": None,
            "validation_ic_reason": None,
        },
        backend_versions=plan.backend_versions,
        thread_settings=plan.thread_settings,
    )
    append_trial_record(
        layout.trial_ledger_root(record.run_id, plan.experiment_id), plan, record
    )
    return {
        "run_id": record.run_id,
        "experiment_id": plan.experiment_id,
        "protocol_hash": plan.protocol_hash,
        "semantic_identity": trial_ledger_semantic_identity(plan, (record,)),
    }


def _quarterly_result(
    quarter: int,
    *,
    trial_layout: StorageLayout,
    model_spec: ModelSpec,
    protocol: dict[str, object] | None = None,
    predictor: object | None = None,
    validation_specs=None,
    validation_rankic=0.2,
) -> QuarterlyModelResult:
    validation_specs = (model_spec,) if validation_specs is None else validation_specs
    columns = model_spec.input_columns
    model_id = model_spec.model_id
    algorithm = model_spec.model_algorithm
    selected_protocol = QUARTERLY_FIXTURE_PROTOCOL if protocol is None else protocol
    configuration = next(
        item
        for item in selected_protocol["research_configuration"]["models"]
        if item["model_id"] == model_id
    )
    trial_spec = _selection_model_specs(
        model_spec,
        configuration.get("parameter_candidates", [configuration["hyperparameters"]]),
        n_train=2,
        root_seed=11,
        quarter_id=quarter_id_for_fold(_fold(quarter)),
    )[0]
    model_id = trial_spec.model_id
    model = predictor or LinearScaledPredictor(
        (0.0, 1.0, *([0.0] * (len(columns) - 1))),
        ModelMatrixScaler(columns, (0.0,) * len(columns), (1.0,) * len(columns)),
    )
    metadata = _metadata(columns, algorithm=algorithm, model_id=model_id)
    refit_spec = _refit_model_spec(
        trial_spec,
        n_train=2,
        quarter_id=quarter_id_for_fold(_fold(quarter)),
        root_seed=11,
        candidate_parameters=configuration.get(
            "parameter_candidates", [configuration["hyperparameters"]]
        )[0],
    )
    metadata.update(
        {
            key: value
            for key, value in model_spec_payload(refit_spec).items()
            if key in metadata
        }
    )
    metadata["resolved_hyperparameters"] = dict(refit_spec.hyperparameters)
    if isinstance(model, LinearScaledPredictor):
        metadata["scaler"] = model.scaler.to_payload()
        metadata["feature_importance"] = {
            column: abs(value)
            for column, value in zip(columns, model.coefficients[1:], strict=True)
        }
    else:
        metadata["feature_importance"] = (
            {column: 1 / len(model.columns) for column in model.columns}
            if isinstance(model, RankEnsemblePredictor)
            else dict(zip(model.columns, map(abs, model.weights), strict=True))
        )
    reference = _trial_reference(
        trial_layout,
        _fold(quarter),
        family_id=model_spec.model_id,
        model_id=model_id,
        model_spec=trial_spec,
        protocol=selected_protocol,
        validation_specs=validation_specs,
        rankic=validation_rankic,
    )
    selection = _selection_metadata(
        family_id=model_spec.model_id, model_id=model_id, columns=columns
    )
    selection["families"][0]["trial_ledger_semantic_identity"] = reference[
        "semantic_identity"
    ]
    selection["common_validation_coverage"] = fixture_validation_coverage(
        _fold(quarter), validation_specs
    )
    refit_reference = fixture_refit_reference(trial_layout, reference, metadata)
    return QuarterlyModelResult(
        quarter=_fold(quarter),
        model_id=model_id,
        model_result=ModelFitResult(
            metadata=metadata,
            prediction_model=model,
        ),
        family_id=model_spec.model_id,
        selection_metadata=selection,
        trial_ledgers=(reference, refit_reference),
    )


def _passthrough_session(**attributes):
    session = SimpleNamespace(**attributes)

    @contextmanager
    def filtered_view(_time_window):
        yield session

    session.filtered_view = filtered_view
    return session


def _research_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share", "a_share"],
            "asset_id": ["000001", "000002", "000001", "000002"],
            "timestamp": pd.to_datetime(
                ["2014-01-02", "2014-01-02", "2014-01-03", "2014-01-03"]
            ),
            "momentum_20d": [1.0, 2.0, 2.0, 1.0],
            "forward_return_20d": [0.01, 0.03, 0.02, 0.00],
            "eligible": [True, True, True, True],
            "can_enter": [True, True, True, True],
            "can_exit": [True, True, True, True],
            "market_type": [4, 4, 4, 4],
        }
    )


def _research_rows_for_dates(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, timestamp in enumerate(pd.to_datetime(dates), start=1):
        for asset_index, asset_id in enumerate(("000001", "000002"), start=1):
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": timestamp,
                    "momentum_20d": float(date_index + asset_index),
                    "forward_return_20d": 0.01 * (date_index + asset_index),
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "market_type": 4,
                }
            )
    return pd.DataFrame(rows)

"""Persistence boundary for selected model instances."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
)
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.research.models.fitting import ModelFitResult
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
)
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact,
    write_semantic_artifact,
)

MODEL_ARTIFACT_SCHEMA_ID = "qrp-model-artifact"


@dataclass(frozen=True)
class PersistedModelArtifact:
    model_id: str
    result: ModelFitResult
    envelope: dict[str, object]


@dataclass(frozen=True)
class QuarterlyPredictionModel:
    """Route one persisted predictor to its frozen development quarter."""

    entries: tuple[tuple[str, str, tuple[str, ...], object], ...]

    @property
    def input_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                column
                for _start, _end, columns, _predictor in self.entries
                for column in columns
            )
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
        result = np.empty(len(frame), dtype="float64")
        covered = np.zeros(len(frame), dtype=bool)
        for start, end, columns, predictor in self.entries:
            mask = (timestamps >= pd.Timestamp(start)) & (
                timestamps <= pd.Timestamp(end)
            )
            if not bool(mask.any()):
                continue
            selected = frame.loc[mask, list(columns)]
            if isinstance(predictor, (RankEnsemblePredictor, ICWeightedRankEnsemble)):
                selected = frame.loc[mask, ["timestamp", *columns]]
            values = np.asarray(predictor.predict(selected), dtype="float64")
            if values.ndim != 1 or len(values) != int(mask.sum()):
                raise ValueError("quarterly_prediction_shape_invalid")
            result[mask.to_numpy()] = values
            covered |= mask.to_numpy()
        if not bool(covered.all()):
            raise ValueError("quarterly_prediction_window_uncovered")
        if not bool(np.isfinite(result).all()):
            raise ValueError("quarterly_prediction_values_invalid")
        return result


def write_model_artifact(
    logical_root: Path,
    *,
    model_id: str,
    envelope: BusinessManifestEnvelope,
    result: ModelFitResult,
    quarterly_results: tuple[object, ...] = (),
) -> Path:
    validate_run_id(model_id)
    metadata = _validated_model_metadata(dict(result.metadata))
    coefficients = np.asarray(result.coefficients, dtype="float64")
    coefficient_payload: list[float] | None = (
        [float(value) for value in coefficients]
        if bool(np.isfinite(coefficients).all())
        else None
    )
    quarterly_payload = []
    quarterly_models: dict[str, object] = {}
    for quarterly in quarterly_results:
        quarter = quarterly.quarter
        quarterly_result = quarterly.model_result
        quarterly_payload.append(
            {
                "quarter_id": quarter.quarter_id,
                "model_id": quarterly.model_id,
                "training_window": _window_payload(quarter.training_window),
                "validation_window": _window_payload(quarter.validation_window),
                "prediction_window": _window_payload(quarter.test_window),
                "model_metadata": _validated_model_metadata(
                    dict(quarterly_result.metadata)
                ),
            }
        )
        if quarterly_result.prediction_model is None:
            raise ValueError("quarterly_prediction_model_missing")
        quarterly_models[quarter.quarter_id] = quarterly_result.prediction_model

    def write(staging: Path) -> None:
        write_internal_artifact_manifest(
            staging / "model.json",
            {
                "schema_id": MODEL_ARTIFACT_SCHEMA_ID,
                "model_id": model_id,
                "model_metadata": metadata,
                "coefficients": coefficient_payload,
                "quarterly_models": quarterly_payload,
            },
        )
        with (staging / "model.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "default": result.prediction_model,
                    "quarterly": quarterly_models,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.flush()
            os.fsync(handle.fileno())

    return write_semantic_artifact(logical_root, envelope, write)


def read_model_artifact(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
) -> PersistedModelArtifact:
    envelope = read_semantic_artifact(
        instance_root,
        expected_identity=expected_identity,
    )
    payload = read_manifest(Path(instance_root) / "model.json")
    if (
        set(payload)
        not in (
            {
                "schema_id",
                "model_id",
                "model_metadata",
                "coefficients",
            },
            {
                "schema_id",
                "model_id",
                "model_metadata",
                "coefficients",
                "quarterly_models",
            },
        )
        or payload["schema_id"] != MODEL_ARTIFACT_SCHEMA_ID
    ):
        raise ValueError("model_artifact_manifest_invalid")
    model_id = payload["model_id"]
    if type(model_id) is not str:
        raise ValueError("model_artifact_model_id_invalid")
    try:
        validate_run_id(model_id)
        metadata = _validated_model_metadata(payload["model_metadata"])
    except (TypeError, ValueError) as exc:
        raise ValueError("model_artifact_metadata_invalid") from exc
    coefficients = payload["coefficients"]
    if coefficients is None:
        coefficient_values = np.empty(0, dtype="float64")
    elif type(coefficients) is list and all(
        type(value) in {int, float} and np.isfinite(value) for value in coefficients
    ):
        coefficient_values = np.asarray(coefficients, dtype="float64")
    else:
        raise ValueError("model_artifact_coefficients_invalid")
    quarterly_payload = payload.get("quarterly_models", [])
    quarterly_entries = _validated_quarterly_payload(quarterly_payload)
    try:
        with (Path(instance_root) / "model.pkl").open("rb") as handle:
            persisted = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError) as exc:
        raise ValueError("model_artifact_model_invalid") from exc
    if type(persisted) is dict and set(persisted) == {"default", "quarterly"}:
        predictor = persisted["default"]
        quarterly_models = persisted["quarterly"]
        if type(quarterly_models) is not dict or set(quarterly_models) != {
            entry["quarter_id"] for entry in quarterly_entries
        }:
            raise ValueError("model_artifact_quarterly_invalid")
        if quarterly_entries:
            if any(
                not callable(
                    getattr(
                        quarterly_models[entry["quarter_id"]],
                        "predict",
                        None,
                    )
                )
                for entry in quarterly_entries
            ):
                raise ValueError("model_artifact_quarterly_invalid")
            routed_entries = tuple(
                (
                    entry["prediction_window"]["start"],
                    entry["prediction_window"]["end"],
                    tuple(entry["model_metadata"]["input_columns"]),
                    quarterly_models[entry["quarter_id"]],
                )
                for entry in quarterly_entries
            )
            predictor = QuarterlyPredictionModel(routed_entries)
    else:
        if quarterly_entries:
            raise ValueError("model_artifact_quarterly_invalid")
        predictor = persisted
    if not callable(getattr(predictor, "predict", None)):
        raise ValueError("model_artifact_model_invalid")
    return PersistedModelArtifact(
        model_id=model_id,
        result=ModelFitResult(
            coefficients=coefficient_values,
            metadata=metadata,
            prediction_model=predictor,
        ),
        envelope=envelope,
    )


def _validated_model_metadata(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("model_artifact_metadata_invalid")
    required = {
        "model_algorithm",
        "input_factor_columns",
        "input_feature_columns",
        "input_columns",
        "label_column",
        "sample_filter_mode",
        "hyperparameters",
        "random_seed",
        "max_train_rows",
        "dependency_status",
        "coefficient_count",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_label_dropna",
        "train_start",
        "train_end",
        "filter_columns_used",
        "filter_definition",
        "sample_weight_protocol",
        "scaler",
    }
    if not required.issubset(value):
        raise ValueError("model_artifact_metadata_invalid")
    if value["sample_weight_protocol"] != "equal_date_total_equal_sum_n_train":
        raise ValueError("model_artifact_metadata_invalid")
    for key in (
        "input_factor_columns",
        "input_feature_columns",
        "input_columns",
        "filter_columns_used",
    ):
        columns = value[key]
        if type(columns) is not list or any(
            type(column) is not str or not column for column in columns
        ):
            raise ValueError("model_artifact_metadata_invalid")
    dependency_status = value["dependency_status"]
    if (
        type(dependency_status) is not dict
        or set(dependency_status) != {"status", "package", "version"}
        or dependency_status["status"] != "available"
        or any(
            type(dependency_status[key]) is not str or not dependency_status[key]
            for key in ("package", "version")
        )
    ):
        raise ValueError("model_artifact_metadata_invalid")
    try:
        ModelSpec(
            model_id="persisted_model",
            model_algorithm=value["model_algorithm"],
            input_factor_columns=tuple(value["input_factor_columns"]),
            input_feature_columns=tuple(value["input_feature_columns"]),
            label_column=value["label_column"],
            sample_filter_mode=value["sample_filter_mode"],
            hyperparameters=value["hyperparameters"],
            random_seed=value["random_seed"],
            max_train_rows=value["max_train_rows"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("model_artifact_metadata_invalid") from exc
    if value["input_columns"] != [
        *value["input_factor_columns"],
        *value["input_feature_columns"],
    ]:
        raise ValueError("model_artifact_metadata_invalid")
    return value


def _validated_quarterly_payload(value: object) -> tuple[dict[str, object], ...]:
    if type(value) is not list:
        raise ValueError("model_artifact_quarterly_invalid")
    result: list[dict[str, object]] = []
    quarter_ids: set[str] = set()
    for item in value:
        if type(item) is not dict or set(item) != {
            "quarter_id",
            "model_id",
            "training_window",
            "validation_window",
            "prediction_window",
            "model_metadata",
        }:
            raise ValueError("model_artifact_quarterly_invalid")
        quarter_id = item["quarter_id"]
        model_id = item["model_id"]
        if (
            type(quarter_id) is not str
            or not quarter_id
            or quarter_id in quarter_ids
            or type(model_id) is not str
            or not model_id
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        for window_name in (
            "training_window",
            "validation_window",
            "prediction_window",
        ):
            window = item[window_name]
            if (
                type(window) is not dict
                or set(window) != {"start", "end"}
                or type(window["start"]) is not str
                or type(window["end"]) is not str
                or pd.Timestamp(window["start"]) > pd.Timestamp(window["end"])
            ):
                raise ValueError("model_artifact_quarterly_invalid")
        result.append(
            {
                **item,
                "model_metadata": _validated_model_metadata(item["model_metadata"]),
            }
        )
        quarter_ids.add(quarter_id)
    return tuple(result)


def _window_payload(window: tuple[object, object]) -> dict[str, str]:
    return {"start": _date_text(window[0]), "end": _date_text(window[1])}


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()

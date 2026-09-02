"""Persistence boundary for selected model instances."""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
)
from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.research.models.fitting import ModelFitResult
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
)
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact,
    write_semantic_artifact,
)

MODEL_ARTIFACT_SCHEMA_ID = "qrp-model-artifact"
_QUARTER_ID_RE = re.compile(r"^[0-9]{4}Q[1-4]$")


@dataclass(frozen=True)
class PersistedModelArtifact:
    model_id: str
    result: ModelFitResult
    envelope: dict[str, object]


@dataclass(frozen=True)
class QuarterlyPredictionModel:
    """Route one persisted predictor to its frozen development quarter."""

    entries: tuple[tuple[str, str, str, str, tuple[str, ...], object], ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("quarterly_prediction_route_empty")
        quarter_ids = tuple(entry[0] for entry in self.entries)
        if len(set(quarter_ids)) != len(quarter_ids):
            raise ValueError("quarterly_prediction_route_duplicate")
        previous_end: pd.Timestamp | None = None
        previous_quarter: str | None = None
        for quarter_id, start, end, model_id, columns, predictor in self.entries:
            if (
                type(quarter_id) is not str
                or _QUARTER_ID_RE.fullmatch(quarter_id) is None
                or type(model_id) is not str
                or not model_id
                or not columns
                or type(columns) is not tuple
                or any(type(column) is not str or not column for column in columns)
                or not callable(getattr(predictor, "predict", None))
            ):
                raise ValueError("quarterly_prediction_route_invalid")
            start_value = pd.Timestamp(start)
            end_value = pd.Timestamp(end)
            if (
                start_value > end_value
                or start_value.year < 2019
                or end_value.year > 2023
                or (previous_end is not None and start_value <= previous_end)
                or (
                    previous_quarter is not None
                    and quarter_id != _next_quarter_id(previous_quarter)
                )
            ):
                raise ValueError("quarterly_prediction_route_overlap")
            previous_end = end_value
            previous_quarter = quarter_id

    @property
    def input_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                column
                for (
                    _quarter_id,
                    _start,
                    _end,
                    _model_id,
                    columns,
                    _predictor,
                ) in self.entries
                for column in columns
            )
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
        result = np.empty(len(frame), dtype="float64")
        covered = np.zeros(len(frame), dtype=bool)
        for (
            _quarter_id,
            start,
            end,
            _model_id,
            columns,
            predictor,
        ) in self.entries:
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

    def model_ids(self, frame: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
        result = np.empty(len(frame), dtype=object)
        covered = np.zeros(len(frame), dtype=bool)
        for _quarter_id, start, end, model_id, _columns, _predictor in self.entries:
            mask = (timestamps >= pd.Timestamp(start)) & (
                timestamps <= pd.Timestamp(end)
            )
            result[mask.to_numpy()] = model_id
            covered |= mask.to_numpy()
        if not bool(covered.all()):
            raise ValueError("quarterly_prediction_window_uncovered")
        return result


def write_model_artifact(
    logical_root: Path,
    *,
    model_id: str,
    envelope: BusinessManifestEnvelope,
    quarterly_results: tuple[object, ...],
) -> Path:
    validate_run_id(model_id)
    if model_id != envelope.logical_artifact_id or not quarterly_results:
        raise ValueError("model_artifact_identity_invalid")
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
                "fit_cutoff": _date_text(quarter.validation_window[1]),
                "n_train": quarterly_result.metadata["rows_after_input_label_dropna"],
                "features": quarterly_result.metadata["input_columns"],
                "resolved_params": quarterly_result.metadata[
                    "resolved_hyperparameters"
                ],
                "random_seed": quarterly_result.metadata["random_seed"],
                "backend": quarterly_result.metadata["dependency_status"],
                "scaler": quarterly_result.metadata["scaler"],
                "rank_parameters": {
                    "directions": list(
                        getattr(quarterly_result.prediction_model, "directions", ())
                    ),
                    "weights": list(
                        getattr(quarterly_result.prediction_model, "weights", ())
                    ),
                },
                "model_metadata": _validated_model_metadata(
                    dict(quarterly_result.metadata)
                ),
            }
        )
        if quarterly_result.prediction_model is None:
            raise ValueError("quarterly_prediction_model_missing")
        quarterly_models[quarter.quarter_id] = {
            "model_id": quarterly.model_id,
            "predictor": quarterly_result.prediction_model,
        }
    quarterly_payload = list(_validated_quarterly_payload(quarterly_payload))

    def write(staging: Path) -> None:
        write_internal_artifact_manifest(
            staging / "model.json",
            {
                "schema_id": MODEL_ARTIFACT_SCHEMA_ID,
                "model_id": model_id,
                "quarterly_models": quarterly_payload,
            },
        )
        with (staging / "model.pkl").open("wb") as handle:
            pickle.dump(
                {
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
        set(payload) != {"schema_id", "model_id", "quarterly_models"}
        or payload["schema_id"] != MODEL_ARTIFACT_SCHEMA_ID
    ):
        raise ValueError("model_artifact_manifest_invalid")
    model_id = payload["model_id"]
    if type(model_id) is not str:
        raise ValueError("model_artifact_model_id_invalid")
    try:
        validate_run_id(model_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("model_artifact_model_id_invalid") from exc
    if model_id != envelope["logical_artifact_id"]:
        raise ValueError("model_artifact_identity_invalid")
    quarterly_payload = payload.get("quarterly_models", [])
    quarterly_entries = _validated_quarterly_payload(quarterly_payload)
    try:
        with open_filesystem_file(Path(instance_root) / "model.pkl") as handle:
            persisted = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ImportError) as exc:
        raise ValueError("model_artifact_model_invalid") from exc
    if type(persisted) is dict and set(persisted) == {"quarterly"}:
        quarterly_models = persisted["quarterly"]
        if type(quarterly_models) is not dict or set(quarterly_models) != {
            entry["quarter_id"] for entry in quarterly_entries
        }:
            raise ValueError("model_artifact_quarterly_invalid")
        predictors: dict[str, object] = {}
        for entry in quarterly_entries:
            binding = quarterly_models[entry["quarter_id"]]
            if (
                type(binding) is not dict
                or set(binding) != {"model_id", "predictor"}
                or binding["model_id"] != entry["model_id"]
                or not callable(getattr(binding["predictor"], "predict", None))
            ):
                raise ValueError("model_artifact_quarterly_invalid")
            predictors[entry["quarter_id"]] = binding["predictor"]
        routed_entries = tuple(
            (
                entry["quarter_id"],
                entry["prediction_window"]["start"],
                entry["prediction_window"]["end"],
                entry["model_id"],
                tuple(entry["model_metadata"]["input_columns"]),
                predictors[entry["quarter_id"]],
            )
            for entry in quarterly_entries
        )
        predictor = QuarterlyPredictionModel(routed_entries)
    else:
        raise ValueError("model_artifact_quarterly_invalid")
    if not callable(getattr(predictor, "predict", None)):
        raise ValueError("model_artifact_model_invalid")
    metadata = _aggregate_model_metadata(quarterly_entries)
    return PersistedModelArtifact(
        model_id=model_id,
        result=ModelFitResult(
            coefficients=np.empty(0, dtype="float64"),
            metadata=metadata,
            prediction_model=predictor,
        ),
        envelope=envelope,
    )


def _validated_model_metadata(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("model_artifact_metadata_invalid")
    try:
        value = parse_qrp_json_object(render_qrp_json_object(value).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ValueError("model_artifact_metadata_invalid") from exc
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
    previous_prediction_end: pd.Timestamp | None = None
    previous_quarter: str | None = None
    for item in value:
        if type(item) is not dict or set(item) != {
            "quarter_id",
            "model_id",
            "training_window",
            "validation_window",
            "prediction_window",
            "fit_cutoff",
            "n_train",
            "features",
            "resolved_params",
            "random_seed",
            "backend",
            "scaler",
            "rank_parameters",
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
            or _QUARTER_ID_RE.fullmatch(quarter_id) is None
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        try:
            validate_run_id(model_id)
        except ValueError as exc:
            raise ValueError("model_artifact_quarterly_invalid") from exc
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
        if (
            pd.Timestamp(item["training_window"]["end"])
            >= pd.Timestamp(item["validation_window"]["start"])
            or pd.Timestamp(item["validation_window"]["end"])
            >= pd.Timestamp(item["prediction_window"]["start"])
            or pd.Timestamp(item["training_window"]["start"]).year < 2014
            or pd.Timestamp(item["prediction_window"]["end"]).year > 2023
            or (
                previous_prediction_end is not None
                and pd.Timestamp(item["prediction_window"]["start"])
                <= previous_prediction_end
            )
            or (
                previous_quarter is not None
                and quarter_id != _next_quarter_id(previous_quarter)
            )
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        if type(item["fit_cutoff"]) is not str:
            raise ValueError("model_artifact_quarterly_invalid")
        try:
            fit_cutoff = pd.Timestamp(item["fit_cutoff"])
        except (TypeError, ValueError) as exc:
            raise ValueError("model_artifact_quarterly_invalid") from exc
        nested = _validated_model_metadata(item["model_metadata"])
        if (
            fit_cutoff != pd.Timestamp(item["validation_window"]["end"])
            or item["n_train"] != nested["rows_after_input_label_dropna"]
            or item["features"] != nested["input_columns"]
            or item["resolved_params"] != nested["resolved_hyperparameters"]
            or item["random_seed"] != nested["random_seed"]
            or item["backend"] != nested["dependency_status"]
            or item["scaler"] != nested["scaler"]
            or type(item["rank_parameters"]) is not dict
            or set(item["rank_parameters"]) != {"directions", "weights"}
            or any(
                type(values) is not list
                or any(
                    type(value) not in {int, float} or not np.isfinite(float(value))
                    for value in values
                )
                for values in item["rank_parameters"].values()
            )
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        result.append(
            {
                **item,
                "model_metadata": nested,
            }
        )
        quarter_ids.add(quarter_id)
        previous_prediction_end = pd.Timestamp(item["prediction_window"]["end"])
        previous_quarter = quarter_id
    return tuple(result)


def _aggregate_model_metadata(
    entries: tuple[dict[str, object], ...],
) -> dict[str, object]:
    if not entries:
        raise ValueError("model_artifact_quarterly_required")
    metadata_values = [entry["model_metadata"] for entry in entries]
    first = metadata_values[0]
    input_factor_columns = tuple(
        dict.fromkeys(
            column
            for metadata in metadata_values
            for column in metadata["input_factor_columns"]
        )
    )
    input_feature_columns = tuple(
        dict.fromkeys(
            column
            for metadata in metadata_values
            for column in metadata["input_feature_columns"]
        )
    )
    input_columns = (*input_factor_columns, *input_feature_columns)
    route = tuple(
        {
            "quarter_id": entry["quarter_id"],
            "model_id": entry["model_id"],
            "training_window": entry["training_window"],
            "validation_window": entry["validation_window"],
            "prediction_window": entry["prediction_window"],
            "fit_cutoff": entry["fit_cutoff"],
            "n_train": entry["n_train"],
            "features": entry["features"],
            "directions": entry["rank_parameters"]["directions"],
            "weights": entry["rank_parameters"]["weights"],
            "scaler": entry["scaler"],
            "resolved_params": entry["resolved_params"],
            "random_seed": entry["random_seed"],
            "backend": entry["backend"],
            "model_metadata": entry["model_metadata"],
        }
        for entry in entries
    )
    return {
        "model_algorithm": "quarterly_route",
        "sample_filter_mode": first["sample_filter_mode"],
        "filter_columns_used": first["filter_columns_used"],
        "filter_definition": first["filter_definition"],
        "rows_input": sum(metadata["rows_input"] for metadata in metadata_values),
        "rows_after_filter_mode": sum(
            metadata["rows_after_filter_mode"] for metadata in metadata_values
        ),
        "rows_after_input_label_dropna": sum(
            metadata["rows_after_input_label_dropna"] for metadata in metadata_values
        ),
        "train_start": min(metadata["train_start"] for metadata in metadata_values),
        "train_end": max(metadata["train_end"] for metadata in metadata_values),
        "test_start": min(entry["prediction_window"]["start"] for entry in entries),
        "test_end": max(entry["prediction_window"]["end"] for entry in entries),
        "oos_start": min(entry["prediction_window"]["start"] for entry in entries),
        "oos_end": max(entry["prediction_window"]["end"] for entry in entries),
        "input_factor_columns": list(input_factor_columns),
        "input_feature_columns": list(input_feature_columns),
        "input_columns": list(input_columns),
        "input_count": len(input_columns),
        "label_column": first["label_column"],
        "hyperparameters": {},
        "resolved_hyperparameters": {},
        "random_seed": None,
        "max_train_rows": None,
        "dependency_status": {
            "status": "available",
            "package": "internal",
            "version": "quarterly",
        },
        "coefficient_count": 1,
        "sample_weight_protocol": first["sample_weight_protocol"],
        "scaler": None,
        "feature_importance_status": "not_available_for_model_algorithm",
        "quarterly_route": list(route),
    }


def _window_payload(window: tuple[object, object]) -> dict[str, str]:
    return {"start": _date_text(window[0]), "end": _date_text(window[1])}


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _next_quarter_id(value: str) -> str:
    year = int(value[:4])
    quarter = int(value[-1])
    if quarter == 4:
        return f"{year + 1:04d}Q1"
    return f"{year:04d}Q{quarter + 1}"

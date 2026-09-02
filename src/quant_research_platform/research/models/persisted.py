"""Persistence boundary for selected model instances."""

from __future__ import annotations

import os
import pickle
import re
import stat
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
)
from quant_research_platform.contracts.models import (
    A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    validate_model_metadata_core,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.protocols.a_share import quarter_id_for_fold
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
)
from quant_research_platform.storage.identity import (
    native_filesystem_path,
    open_filesystem_file,
)
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_record,
    verify_semantic_artifact_member,
    write_semantic_artifact,
)

_QUARTER_ID_RE = re.compile(r"^[0-9]{4}Q[1-4]$")


@dataclass(frozen=True)
class PersistedQuarterlyModelArtifact:
    model_id: str
    quarterly_models: tuple[dict[str, object], ...]
    instance_root: Path
    envelope: dict[str, object]
    member_records: tuple[dict[str, object], ...]


def write_quarterly_model_artifact(
    logical_root: Path,
    *,
    model_id: str,
    envelope: BusinessManifestEnvelope,
    quarterly_results: Iterable[object],
) -> Path:
    validate_run_id(model_id)
    if model_id != envelope.logical_artifact_id or quarterly_results is None:
        raise ValueError("model_artifact_identity_invalid")

    def write(staging: Path) -> None:
        quarterly_payload: list[dict[str, object]] = []
        models_root = staging / "models"
        os.mkdir(native_filesystem_path(models_root))
        iterator = iter(quarterly_results)
        try:
            for yielded in iterator:
                owned_quarterly = yielded if isinstance(yielded, OwnedValue) else None
                quarterly = (
                    owned_quarterly.value if owned_quarterly is not None else yielded
                )
                quarter = None
                quarterly_result = None
                try:
                    quarter = quarterly.quarter
                    quarterly_result = quarterly.model_result
                    if quarterly_result.prediction_model is None:
                        raise ValueError("quarterly_prediction_model_missing")
                    metadata = dict(quarterly_result.metadata)
                    quarter_id = quarter_id_for_fold(quarter)
                    metadata.update(
                        {
                            "fit_cutoff": _date_text(quarter.validation.end),
                        }
                    )
                    model_algorithm = metadata["model_algorithm"]
                    if model_algorithm in {
                        "equal_weight_rank_ensemble",
                        "ic_weighted_rank_ensemble",
                    }:
                        rank_columns = tuple(
                            getattr(
                                quarterly_result.prediction_model,
                                "columns",
                                (),
                            )
                        )
                        rank_parameters = {"columns": list(rank_columns)}
                        if model_algorithm == "equal_weight_rank_ensemble":
                            rank_parameters["directions"] = list(
                                getattr(
                                    quarterly_result.prediction_model,
                                    "directions",
                                    (),
                                )
                            )
                        else:
                            rank_parameters["weights"] = list(
                                getattr(
                                    quarterly_result.prediction_model,
                                    "weights",
                                    (),
                                )
                            )
                        metadata["rank_parameters"] = rank_parameters
                    quarterly_payload.append(
                        {
                            "quarter_id": quarter_id,
                            "model_id": quarterly.model_id,
                            "training_window": _window_payload(quarter.train),
                            "validation_window": _window_payload(quarter.validation),
                            "prediction_window": _window_payload(
                                quarter.development_test
                            ),
                            "model_metadata": validate_model_metadata_core(
                                metadata,
                                quarterly=True,
                            ),
                        }
                    )
                    with open_filesystem_file(
                        models_root / f"{quarter_id}.pkl",
                        "wb",
                    ) as handle:
                        pickle.dump(
                            quarterly_result.prediction_model,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    ownership = owned_quarterly
                    primary = sys.exception()
                    quarterly = None
                    quarterly_result = None
                    owned_quarterly = None
                    if ownership is not None and ownership.lease.active:
                        try:
                            ownership.release(owner="research.model.artifact")
                        except BaseException as cleanup_error:
                            if primary is not None:
                                primary.add_note(
                                    "quarterly model ownership release also failed: "
                                    f"{type(cleanup_error).__name__}"
                                )
                            else:
                                raise
                    ownership = None
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                primary = sys.exc_info()[1]
                try:
                    close()
                except BaseException as cleanup_error:
                    if primary is not None:
                        primary.add_note(
                            "quarterly model generator cleanup also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    else:
                        raise
        if not quarterly_payload:
            raise ValueError("model_artifact_quarterly_required")
        validated_payload = list(_validated_quarterly_payload(quarterly_payload))
        write_internal_artifact_manifest(
            staging / "model.json",
            {
                "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                "model_id": model_id,
                "quarterly_models": validated_payload,
            },
        )

    return write_semantic_artifact(logical_root, envelope, write)


def read_quarterly_model_artifact(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
) -> PersistedQuarterlyModelArtifact:
    envelope, member_records = read_semantic_artifact_record(
        instance_root,
        expected_identity=expected_identity,
    )
    manifest_member = next(
        (item for item in member_records if item["path"] == "model.json"),
        None,
    )
    if manifest_member is None:
        raise ValueError("model_artifact_manifest_invalid")
    verify_semantic_artifact_member(
        instance_root,
        relative_path="model.json",
        expected=manifest_member,
    )
    payload = read_manifest(Path(instance_root) / "model.json")
    verify_semantic_artifact_member(
        instance_root,
        relative_path="model.json",
        expected=manifest_member,
    )
    if (
        set(payload) != {"schema_id", "model_id", "quarterly_models"}
        or payload["schema_id"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
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
    for entry in quarterly_entries:
        predictor_path = Path(instance_root) / "models" / f"{entry['quarter_id']}.pkl"
        try:
            predictor_stat = os.stat(
                native_filesystem_path(predictor_path),
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError("model_artifact_quarterly_invalid") from exc
        if not stat.S_ISREG(predictor_stat.st_mode):
            raise ValueError("model_artifact_quarterly_invalid")
    return PersistedQuarterlyModelArtifact(
        model_id=model_id,
        quarterly_models=quarterly_entries,
        instance_root=Path(instance_root),
        envelope=envelope,
        member_records=member_records,
    )


@contextmanager
def read_quarterly_predictor(
    artifact: PersistedQuarterlyModelArtifact,
    quarter_id: str,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
):
    """Load exactly one persisted quarter while its lease remains live."""

    entry = next(
        (
            item
            for item in artifact.quarterly_models
            if item["quarter_id"] == quarter_id
        ),
        None,
    )
    if entry is None:
        raise ValueError("model_artifact_quarter_invalid")
    path = artifact.instance_root / "models" / f"{quarter_id}.pkl"
    try:
        size = int(os.stat(native_filesystem_path(path), follow_symlinks=False).st_size)
    except OSError as exc:
        raise ValueError("model_artifact_model_invalid") from exc
    policy = resource_policy or current_runtime_resource_policy()
    lease: MemoryLease | None = None
    predictor: object | None = None
    try:
        if policy is not None:
            lease = policy.lease("research.model.predictor", max(1, size))
        relative_path = f"models/{quarter_id}.pkl"
        member = next(
            (item for item in artifact.member_records if item["path"] == relative_path),
            None,
        )
        if member is None:
            raise ValueError("model_artifact_model_invalid")
        verify_semantic_artifact_member(
            artifact.instance_root,
            relative_path=relative_path,
            expected=member,
        )
        try:
            with open_filesystem_file(path) as handle:
                predictor = pickle.load(handle)
        except (
            OSError,
            pickle.PickleError,
            EOFError,
            AttributeError,
            ImportError,
        ) as exc:
            raise ValueError("model_artifact_model_invalid") from exc
        verify_semantic_artifact_member(
            artifact.instance_root,
            relative_path=relative_path,
            expected=member,
        )
        if not callable(getattr(predictor, "predict", None)):
            raise ValueError("model_artifact_model_invalid")
        yield predictor
    finally:
        predictor = None
        if lease is not None and lease.active:
            primary = sys.exc_info()[1]
            try:
                lease.close()
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note(
                        "model predictor lease release also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
                else:
                    raise


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
        prediction_start = pd.Timestamp(item["prediction_window"]["start"])
        prediction_end = pd.Timestamp(item["prediction_window"]["end"])
        if (
            str(pd.Period(prediction_start, freq="Q")) != quarter_id
            or str(pd.Period(prediction_end, freq="Q")) != quarter_id
        ):
            raise ValueError("model_artifact_quarter_window_mismatch")
        nested = validate_model_metadata_core(
            item["model_metadata"],
            quarterly=True,
        )
        if pd.Timestamp(nested["fit_cutoff"]) != pd.Timestamp(
            item["validation_window"]["end"]
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        selection = nested.get("model_selection")
        if selection is not None and (
            type(selection) is not dict
            or selection.get("validation_start") != item["validation_window"]["start"]
            or selection.get("validation_end") != item["validation_window"]["end"]
            or selection.get("selected_model_id") != item["model_id"]
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


def _window_payload(window: TimeWindow) -> dict[str, str]:
    if not isinstance(window, TimeWindow):
        raise TypeError("model_artifact_window_invalid")
    start = window.start
    end = window.end
    return {"start": _date_text(start), "end": _date_text(end)}


def _date_text(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


def _next_quarter_id(value: str) -> str:
    year = int(value[:4])
    quarter = int(value[-1])
    if quarter == 4:
        return f"{year + 1:04d}Q1"
    return f"{year:04d}Q{quarter + 1}"

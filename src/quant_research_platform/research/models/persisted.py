"""Persistence boundary for selected model instances."""

from __future__ import annotations

import os
import pickle
import re
import stat
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.models import (
    A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
    ModelSpec,
    model_spec_payload,
    validate_a_share_model_selection_metadata,
    validate_model_metadata_core,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.models.fitting import validate_model_state
from quant_research_platform.research.models.production import _refit_model_spec
from quant_research_platform.research.models.trial_selection import (
    select_best_validation_trial,
)
from quant_research_platform.research.protocols.a_share import quarter_id_for_fold
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.datasets.writing import write_parquet_file
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
)
from quant_research_platform.storage.identity import (
    native_filesystem_path,
    open_filesystem_file,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    dataset_file_identity,
)
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_operational_metadata,
    read_semantic_artifact_record,
    verify_semantic_artifact_member,
    write_semantic_artifact,
)

_QUARTER_ID_RE = re.compile(r"^[0-9]{4}Q[1-4]$")
_CRYPTO_VALIDATION_SCHEMA = pa.schema(
    (
        ("timestamp", pa.timestamp("ns")),
        ("asset_id", pa.string()),
        ("label", pa.float64()),
        ("prediction_score", pa.float64()),
        ("label_available_at", pa.timestamp("ns")),
        ("book_segment_id", pa.string()),
        ("trade_connection_id", pa.string()),
        ("segment_id", pa.string()),
    )
)


@dataclass(frozen=True)
class PersistedQuarterlyModelArtifact:
    model_id: str
    quarterly_models: tuple[dict[str, object], ...]
    quarterly_selections: dict[str, dict[str, object]]
    instance_root: Path
    envelope: dict[str, object]
    member_records: tuple[dict[str, object], ...]
    trial_ledgers: tuple[dict[str, str], ...]

    def prediction_models(
        self, window: TimeWindow, *, access_window: str = "development"
    ) -> tuple[dict[str, object], ...]:
        """Bind observation quarters to the immutable source model quarter."""
        if access_window not in {"development", "holdout"}:
            raise ValueError("unknown_a_share_access_window")
        source_quarter = max(self.quarterly_selections)
        models = []
        for period in pd.period_range(window.start, window.end, freq="Q"):
            observed = str(period)
            source = source_quarter if access_window == "holdout" else observed
            entries = tuple(
                item for item in self.quarterly_models if item["quarter_id"] == source
            )
            if not entries:
                raise ValueError("persisted_model_prediction_window_uncovered")
            start = max(pd.Timestamp(window.start), period.start_time)
            end = min(pd.Timestamp(window.end), period.end_time.normalize())
            if access_window == "holdout" and any(
                pd.Timestamp(item["model_metadata"]["fit_cutoff"]) >= start
                for item in entries
            ):
                raise ValueError("holdout_model_cutoff_invalid")
            for item in entries:
                models.append(
                    {
                        **item,
                        "quarter_id": observed,
                        "source_quarter": source,
                        "prediction_window": item["prediction_window"]
                        if access_window == "development"
                        else {
                            "start": start.date().isoformat(),
                            "end": end.date().isoformat(),
                        },
                    }
                )
        return tuple(models)

    def prediction_lineage(
        self, window: TimeWindow, *, access_window: str = "development"
    ) -> dict[str, object]:
        models = self.prediction_models(window, access_window=access_window)
        return {
            "model_artifact_id": self.model_id,
            "model_artifact_identity": self.envelope["instance_identity"],
            "model_artifact_schema": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
            "quarterly_models": [
                {
                    key: item[key]
                    for key in (
                        "quarter_id",
                        "source_quarter",
                        "family_id",
                        "model_id",
                        "prediction_window",
                    )
                }
                for item in models
            ],
            "champion_by_quarter": {
                item["quarter_id"]: self.quarterly_selections[item["source_quarter"]][
                    "selected_family_id"
                ]
                for item in models
            },
        }


def write_quarterly_model_artifact(
    logical_root: Path,
    *,
    model_id: str,
    envelope: BusinessManifestEnvelope,
    quarterly_results: Iterable[OwnedValue],
    trial_layout: StorageLayout | None = None,
    verify_source: Callable[[], None] | None = None,
) -> Path:
    validate_run_id(model_id)
    if model_id != envelope.logical_artifact_id or quarterly_results is None:
        raise ValueError("model_artifact_identity_invalid")
    operational = {"model_trial_ledgers": []}
    quarterly_payload: list[dict[str, object]] = []

    def write(staging: Path) -> None:
        selections: dict[str, dict[str, object]] = {}
        models_root = staging / "models"
        os.mkdir(native_filesystem_path(models_root))
        iterator = iter(quarterly_results)
        try:
            for owned_quarterly in iterator:
                owned_quarterly.lease.transfer("research.model.artifact")
                quarterly = owned_quarterly.value
                quarter = None
                quarterly_result = None
                try:
                    quarter = quarterly.quarter
                    quarterly_result = quarterly.model_result
                    if quarterly_result.prediction_model is None:
                        raise ValueError("quarterly_prediction_model_missing")
                    metadata = dict(quarterly_result.metadata)
                    operational["model_trial_ledgers"].extend(quarterly.trial_ledgers)
                    quarter_id = quarter_id_for_fold(quarter)
                    family_id = quarterly.family_id
                    validate_run_id(family_id)
                    selection = dict(quarterly.selection_metadata)
                    validate_a_share_model_selection_metadata(selection)
                    if quarter_id in selections and selections[quarter_id] != selection:
                        raise ValueError("model_artifact_selection_conflict")
                    selections[quarter_id] = selection
                    quarter_root = models_root / quarter_id
                    quarter_root.mkdir(exist_ok=True)

                    cutoff = _date_text(quarter.validation.end)
                    if "fit_cutoff" in metadata and metadata["fit_cutoff"] != cutoff:
                        raise ValueError("model_fit_cutoff_mismatch")
                    metadata["fit_cutoff"] = cutoff
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
                        if (
                            "rank_parameters" in metadata
                            and metadata["rank_parameters"] != rank_parameters
                        ):
                            raise ValueError("model_state_metadata_mismatch")
                        metadata["rank_parameters"] = rank_parameters
                    quarterly_payload.append(
                        {
                            "quarter_id": quarter_id,
                            "model_id": quarterly.model_id,
                            "family_id": family_id,
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
                    validate_model_state(
                        quarterly_result.prediction_model,
                        quarterly_payload[-1]["model_metadata"],
                    )
                    _write_model_member(
                        quarter_root / f"{family_id}.pkl",
                        quarterly_result.prediction_model,
                        state_bytes=owned_quarterly.lease.reserved_bytes,
                    )
                finally:
                    ownership = owned_quarterly
                    primary = sys.exception()
                    quarterly = None
                    quarterly_result = None
                    owned_quarterly = None
                    if ownership.lease.active:
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
        validated_payload = list(
            _validated_quarterly_payload(quarterly_payload, selections)
        )
        _validate_trial_references(
            validated_payload,
            selections,
            operational["model_trial_ledgers"],
            trial_layout,
            envelope.payload(),
        )
        for quarter_id, selection in selections.items():
            write_internal_artifact_manifest(
                staging / "models" / quarter_id / "metadata.json",
                {
                    "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                    "quarter_id": quarter_id,
                    "selection": selection,
                    "models": [
                        item
                        for item in validated_payload
                        if item["quarter_id"] == quarter_id
                    ],
                },
            )
        write_internal_artifact_manifest(
            staging / "model.json",
            {
                "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                "model_id": model_id,
                "quarters": list(selections),
            },
        )

    def verify_models(root, members):
        for entry in quarterly_payload:
            relative = f"models/{entry['quarter_id']}/{entry['family_id']}.pkl"
            with _read_model_member(root, members, relative) as owned:
                validate_model_state(owned.value, entry["model_metadata"])
        if verify_source is not None:
            verify_source()

    return write_semantic_artifact(
        logical_root,
        envelope,
        write,
        operational_metadata=operational,
        staged_validator=verify_models,
    )


def read_quarterly_model_artifact(
    instance_root: Path,
    *,
    expected_identity: str | None = None,
    trial_layout: StorageLayout | None = None,
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
        set(payload) != {"schema_id", "model_id", "quarters"}
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
    quarters = payload["quarters"]
    if (
        type(quarters) is not list
        or not quarters
        or any(
            type(item) is not str or _QUARTER_ID_RE.fullmatch(item) is None
            for item in quarters
        )
        or len(quarters) != len(set(quarters))
    ):
        raise ValueError("model_artifact_quarterly_invalid")
    quarterly_payload = []
    selections = {}
    for quarter_id in quarters:
        relative = f"models/{quarter_id}/metadata.json"
        member = next(
            (item for item in member_records if item["path"] == relative), None
        )
        if member is None:
            raise ValueError("model_artifact_quarterly_invalid")
        verify_semantic_artifact_member(
            instance_root, relative_path=relative, expected=member
        )
        quarter = read_manifest(instance_root / relative)
        if (
            set(quarter) != {"schema_id", "quarter_id", "selection", "models"}
            or quarter["schema_id"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
            or quarter["quarter_id"] != quarter_id
            or any(item["quarter_id"] != quarter_id for item in quarter["models"])
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        quarterly_payload.extend(quarter["models"])
        selections[quarter_id] = quarter["selection"]
    quarterly_entries = _validated_quarterly_payload(quarterly_payload, selections)
    references = read_semantic_artifact_operational_metadata(instance_root).get(
        "model_trial_ledgers", []
    )
    _validate_trial_references(
        quarterly_entries, selections, references, trial_layout, envelope
    )
    for entry in quarterly_entries:
        predictor_path = (
            Path(instance_root)
            / "models"
            / entry["quarter_id"]
            / f"{entry['family_id']}.pkl"
        )
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
        quarterly_selections=selections,
        instance_root=Path(instance_root),
        envelope=envelope,
        member_records=member_records,
        trial_ledgers=tuple(references),
    )


def _validate_trial_references(
    quarterly_entries,
    selections,
    references,
    layout: StorageLayout | None,
    envelope: dict[str, object],
) -> None:
    if layout is None:
        raise ValueError("model_trial_layout_required")
    expected = [
        (entry, family)
        for entry in quarterly_entries
        for family in selections[entry["quarter_id"]]["families"]
        if family["family_id"] == entry["family_id"]
    ]
    if len(expected) != len(references):
        raise ValueError("model_trial_references_missing")
    contract = envelope["semantic_metadata"]["target_contract"]
    protocol_identity = canonical_json_hash(contract["protocol"])
    registered_families = {item["model_id"] for item in contract["model_specs"]}
    family_metrics = {}
    for (entry, family), reference in zip(expected, references, strict=True):
        plan, records = read_model_trial_ledger_reference(layout, reference)
        dataset_identity = envelope["upstream_identities"].get(plan.research_dataset_id)
        if (
            plan.target_id != envelope["logical_artifact_id"]
            or dataset_identity is None
            or plan.data_snapshot_id != dataset_identity
            or plan.input_lineage != (dataset_identity,)
            or plan.research_protocol_identity != protocol_identity
            or plan.candidate_family_id not in registered_families
            or len(plan.folds) != 1
            or _window_payload(plan.folds[0].train) != entry["training_window"]
            or _window_payload(plan.folds[0].validation) != entry["validation_window"]
            or _window_payload(plan.folds[0].development_test)
            != entry["prediction_window"]
        ):
            raise ValueError("model_trial_artifact_binding_invalid")
        if (
            reference["semantic_identity"] != family["trial_ledger_semantic_identity"]
            or plan.candidate_family_id != family["family_id"]
            or quarter_id_for_fold(plan.folds[0]) != entry["quarter_id"]
            or len(records) != len(plan.candidate_ids)
            or any(record.status != "completed" for record in records)
            or set(record.candidate_id for record in records) != set(plan.candidate_ids)
            or tuple(family["config_tie_break_order"]) != plan.tie_break_order
        ):
            raise ValueError("model_trial_reference_semantics_invalid")
        frame = pd.DataFrame(
            [
                {"model_id": record.candidate_id, **dict(record.validation_metrics)}
                for record in records
            ]
        )
        selected = select_best_validation_trial(
            frame,
            trial_id_column="model_id",
            selection_metric=plan.selection_metric,
            maximize=plan.selection_direction == "maximize",
            validation_metric_columns=(plan.selection_metric,),
            test_metric_columns=(),
            tie_break_order=plan.tie_break_order,
        )
        if selected != family["selected_config_id"]:
            raise ValueError("model_trial_selection_mismatch")
        winning = next(record for record in records if record.candidate_id == selected)
        parameters = winning.payload()["normalized_parameters"]
        if not plan.candidate_parameters or not parameters:
            raise ValueError("model_trial_parameters_missing")
        metadata = entry["model_metadata"]
        if (
            parameters["model_id"] != selected
            or winning.random_seed != parameters["random_seed"]
        ):
            raise ValueError("model_trial_parameters_misaligned")
        for key in (
            "model_algorithm",
            "input_factor_columns",
            "input_feature_columns",
            "label_column",
            "sample_filter_mode",
            "max_train_rows",
        ):
            if metadata[key] != parameters[key]:
                raise ValueError("model_trial_parameters_misaligned")
        selected_spec = ModelSpec(
            **{
                **parameters,
                "input_factor_columns": tuple(parameters["input_factor_columns"]),
                "input_feature_columns": tuple(parameters["input_feature_columns"]),
            }
        )
        refit = _refit_model_spec(
            selected_spec,
            n_train=metadata["rows_after_input_label_dropna"],
            quarter_id=entry["quarter_id"],
            root_seed=plan.random_seed,
            config_index=plan.candidate_ids.index(selected),
        )
        if (
            metadata["random_seed"] != refit.random_seed
            or metadata["hyperparameters"]
            != model_spec_payload(refit)["hyperparameters"]
        ):
            raise ValueError("model_trial_refit_misaligned")
        family_metrics.setdefault(entry["quarter_id"], []).append(
            {"model_id": family["family_id"], **dict(winning.validation_metrics)}
        )
    for quarter_id, selection in selections.items():
        metrics = pd.DataFrame(family_metrics[quarter_id])
        if metrics["validation_rows"].nunique() != 1:
            raise ValueError("model_trial_common_coverage_mismatch")
        champion = select_best_validation_trial(
            metrics,
            trial_id_column="model_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=tuple(selection["tie_break_order"]),
        )
        if champion != selection["selected_family_id"]:
            raise ValueError("model_trial_selection_mismatch")


@contextmanager
def read_quarterly_predictor(
    artifact: PersistedQuarterlyModelArtifact,
    quarter_id: str,
    family_id: str,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
):
    """Load exactly one persisted quarter while its lease remains live."""

    entry = next(
        (
            item
            for item in artifact.quarterly_models
            if item["quarter_id"] == quarter_id and item["family_id"] == family_id
        ),
        None,
    )
    if entry is None:
        raise ValueError("model_artifact_quarter_invalid")
    relative = f"models/{quarter_id}/{family_id}.pkl"
    with _read_model_member(
        artifact.instance_root,
        artifact.member_records,
        relative,
        resource_policy=resource_policy,
    ) as owned:
        predictor = owned.value
        try:
            validate_model_state(predictor, entry["model_metadata"])
            yield predictor
        finally:
            predictor = None


def _write_model_member(path, value, *, state_bytes):
    # Retain the model envelope plus pickle overhead without serializing a copy.
    require_free_disk_headroom(path.parent, reserve_bytes=2 * state_bytes)
    with handle_preserving_primary(open_filesystem_file(path, "wb")) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _read_model_member(
    root, members, relative, *, resource_policy=None, workspace_factor=1
):
    member = next((item for item in members if item["path"] == relative), None)
    if member is None:
        raise ValueError("model_artifact_model_invalid")
    path = root / relative
    size = os.stat(native_filesystem_path(path), follow_symlinks=False).st_size
    with runtime_resource_policy_scope(resource_policy) as policy:
        lease = policy.lease(
            "research.model.predictor", max(1, size * workspace_factor)
        )
        value = None
        owned = None
        try:
            verify_semantic_artifact_member(
                root, relative_path=relative, expected=member
            )
            try:
                with handle_preserving_primary(open_filesystem_file(path)) as handle:
                    value = pickle.load(handle)
                    if handle.read(1):
                        raise ValueError("model_artifact_model_invalid")
            except (
                OSError,
                pickle.PickleError,
                EOFError,
                AttributeError,
                ImportError,
            ) as error:
                raise ValueError("model_artifact_model_invalid") from error
            verify_semantic_artifact_member(
                root, relative_path=relative, expected=member
            )
            owned = OwnedValue(value, lease)
            value = None
            yield owned
        finally:
            value = None
            if owned is not None:
                owned.value = None
            from quant_research_platform.runtime.resources import (
                _close_lease_preserving_primary,
            )

            _close_lease_preserving_primary(lease, owner="research.model.predictor")


def _validated_quarterly_payload(
    value: object, selections: object
) -> tuple[dict[str, object], ...]:
    if type(value) is not list or type(selections) is not dict or not value:
        raise ValueError("model_artifact_quarterly_invalid")
    result = []
    quarter_families: dict[str, list[str]] = {}
    windows: dict[str, tuple[dict[str, str], ...]] = {}
    for item in value:
        if type(item) is not dict or set(item) != {
            "quarter_id",
            "family_id",
            "model_id",
            "training_window",
            "validation_window",
            "prediction_window",
            "model_metadata",
        }:
            raise ValueError("model_artifact_quarterly_invalid")
        quarter_id, family_id, model_id = (
            item["quarter_id"],
            item["family_id"],
            item["model_id"],
        )
        if (
            type(quarter_id) is not str
            or _QUARTER_ID_RE.fullmatch(quarter_id) is None
            or quarter_id not in selections
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        validate_run_id(family_id)
        validate_run_id(model_id)
        selection = selections[quarter_id]
        validate_a_share_model_selection_metadata(selection)
        for window_name in (
            "training_window",
            "validation_window",
            "prediction_window",
        ):
            window = item[window_name]
            if type(window) is not dict or set(window) != {"start", "end"}:
                raise ValueError("model_artifact_quarterly_invalid")
            TimeWindow(**window)
        current_windows = tuple(
            item[name]
            for name in ("training_window", "validation_window", "prediction_window")
        )
        if quarter_id not in quarter_families:
            if quarter_families:
                previous = next(reversed(quarter_families))
                if quarter_id != _next_quarter_id(previous) or pd.Timestamp(
                    item["prediction_window"]["start"]
                ) <= pd.Timestamp(windows[previous][2]["end"]):
                    raise ValueError("model_artifact_quarterly_invalid")
            quarter_families[quarter_id] = []
            windows[quarter_id] = current_windows
        elif (
            quarter_id != next(reversed(quarter_families))
            or current_windows != windows[quarter_id]
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        if family_id in quarter_families[quarter_id]:
            raise ValueError("model_artifact_duplicate_family")
        quarter_families[quarter_id].append(family_id)
        if pd.Timestamp(item["training_window"]["end"]) >= pd.Timestamp(
            item["validation_window"]["start"]
        ) or pd.Timestamp(item["validation_window"]["end"]) >= pd.Timestamp(
            item["prediction_window"]["start"]
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        if any(
            str(pd.Period(item["prediction_window"][bound], freq="Q")) != quarter_id
            for bound in ("start", "end")
        ):
            raise ValueError("model_artifact_quarter_window_mismatch")
        family = next(
            (
                member
                for member in selection["families"]
                if member["family_id"] == family_id
            ),
            None,
        )
        if family is None or family["selected_config_id"] != model_id:
            raise ValueError("model_artifact_family_selection_mismatch")
        nested = validate_model_metadata_core(item["model_metadata"], quarterly=True)
        if pd.Timestamp(nested["train_start"]) < pd.Timestamp(
            item["training_window"]["start"]
        ):
            raise ValueError("model_artifact_training_range_invalid")
        if (
            pd.Timestamp(nested["fit_cutoff"])
            != pd.Timestamp(item["validation_window"]["end"])
            or selection["validation_start"] != item["validation_window"]["start"]
            or selection["validation_end"] != item["validation_window"]["end"]
        ):
            raise ValueError("model_artifact_quarterly_invalid")
        result.append({**item, "model_metadata": nested})
    if set(quarter_families) != set(selections) or any(
        families != selections[quarter]["candidate_family_ids"]
        for quarter, families in quarter_families.items()
    ):
        raise ValueError("model_artifact_family_coverage_invalid")
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


@dataclass(frozen=True)
class PersistedCryptoModelArtifact:
    instance_root: Path
    envelope: dict
    member_records: tuple
    models: tuple[dict, ...]

    def validation_chunks(self, *, fold_id, model_id):
        entry = next(
            entry
            for entry in self.models
            if entry["fold_id"] == fold_id and entry["model_id"] == model_id
        )
        return _crypto_validation_chunks(self.instance_root, self.member_records, entry)

    def prediction_lineage(self, model_id):
        entries = [entry for entry in self.models if entry["model_id"] == model_id]
        return {
            "model_artifact_id": self.envelope["logical_artifact_id"],
            "model_artifact_identity": self.envelope["instance_identity"],
            "model_artifact_schema": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
            "model_id": model_id,
            "asset_ids": sorted(
                self.envelope["semantic_metadata"]["target_contract"]["protocol"][
                    "research_plan"
                ]["exposure_limits"]
            ),
            "fold_models": [
                {
                    key: entry[key]
                    for key in (
                        "fold_id",
                        "prediction_window",
                        "training_asset_ids",
                        "evaluation_asset_ids",
                    )
                }
                for entry in entries
            ],
        }


def write_crypto_model_artifact(logical_root, *, envelope, model_results, trial_layout):
    operational = {"model_trial_ledgers": []}

    def write(staging):
        folders = {}
        validation_parts = {}

        def write_validation(fold, spec, frame):
            paths = validation_parts.setdefault((fold.fold_id, spec.model_id), [])
            relative = (
                f"models/{fold.fold_id}/validation/{spec.model_id}/"
                f"part-{len(paths):06d}.parquet"
            )
            policy = current_runtime_resource_policy()
            with policy.lease(
                "research.model.validation.persist",
                max(1, 3 * pandas_frame_payload_bytes(frame)),
            ):
                table = pa.Table.from_pandas(
                    frame.rename(columns={spec.label_column: "label"}),
                    schema=_CRYPTO_VALIDATION_SCHEMA,
                    preserve_index=False,
                )
                write_parquet_file(
                    table, Path(native_filesystem_path(staging / relative))
                )
                del table
            paths.append(relative)

        with closing_iterator(
            iter(model_results(on_validation=write_validation))
        ) as source:
            for owned in source:
                owned.lease.transfer("research.model.crypto.persist")
                member = owned.value
                try:
                    fold = member.fold
                    fold_root = staging / "models" / fold.fold_id
                    fold_root.mkdir(parents=True, exist_ok=True)
                    _write_model_member(
                        fold_root / f"{member.spec.model_id}.pkl",
                        member.result.prediction_model,
                        state_bytes=owned.lease.reserved_bytes,
                    )
                    entry = {
                        "fold_id": fold.fold_id,
                        "model_id": member.spec.model_id,
                        "training_window": _intraday_window(fold.train),
                        "validation_window": _intraday_window(fold.validation),
                        "prediction_window": _intraday_window(fold.development_test),
                        "training_asset_ids": list(member.training_assets),
                        "validation_asset_ids": list(member.training_assets),
                        "evaluation_asset_ids": list(member.evaluation_assets),
                        "model_metadata": validate_model_metadata_core(
                            member.result.metadata
                        ),
                        "trial_semantic_identity": member.trial_reference[
                            "semantic_identity"
                        ],
                        "validation_prediction_parts": validation_parts.get(
                            (fold.fold_id, member.spec.model_id), []
                        ),
                    }
                    folders.setdefault(fold.fold_id, []).append(entry)
                    operational["model_trial_ledgers"].append(member.trial_reference)
                finally:
                    member = None
                    if owned.lease.active:
                        primary = sys.exception()
                        try:
                            owned.release(owner="research.model.crypto.persist")
                        except BaseException as cleanup:
                            if primary is None:
                                raise
                            primary.add_note(
                                "model member release also failed: "
                                f"{type(cleanup).__name__}"
                            )
                    owned = None
        for fold_id, entries in folders.items():
            write_internal_artifact_manifest(
                staging / "models" / fold_id / "metadata.json",
                {"schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID, "models": entries},
            )
        write_internal_artifact_manifest(
            staging / "model.json",
            {"schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID, "folds": list(folders)},
        )

    return write_semantic_artifact(
        logical_root,
        envelope,
        write,
        operational_metadata=operational,
        staged_validator=lambda root, members: _validate_crypto_model_payload(
            root,
            envelope.payload(),
            members,
            operational["model_trial_ledgers"],
            trial_layout,
        ),
    )


def _intraday_window(window):
    from quant_research_platform.contracts.time import time_value_isoformat

    return {
        "start": time_value_isoformat(window.start, preserve_datetime=True),
        "end": time_value_isoformat(window.end, preserve_datetime=True),
    }


def read_crypto_model_artifact(root, *, expected_identity=None, trial_layout=None):
    root = Path(root)
    envelope, members = read_semantic_artifact_record(
        root, expected_identity=expected_identity
    )
    refs = read_semantic_artifact_operational_metadata(root).get("model_trial_ledgers")
    return _validate_crypto_model_payload(root, envelope, members, refs, trial_layout)


def _validate_crypto_model_payload(root, envelope, members, refs, trial_layout):
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_MAKER_CALIBRATION_MODEL_IDS,
        build_hft_prediction_catalog,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        protocol as crypto_protocol,
    )

    contract = envelope["semantic_metadata"]["target_contract"]
    protocol = crypto_protocol(
        contract["protocol"]["scenario_id"],
        research_plan=contract["protocol"]["research_plan"],
    )
    window = TimeWindow(
        envelope["research_window"]["start"], envelope["research_window"]["end"]
    )
    folds = [
        fold
        for fold in protocol.research_plan.folds
        if fold.development_test.end >= window.start
        and fold.development_test.start <= window.end
    ]
    model_ids = protocol.model_ids_for_target(envelope["logical_artifact_id"])
    payload = read_manifest(root / "model.json")
    if payload != {
        "schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
        "folds": [fold.fold_id for fold in folds],
    }:
        raise ValueError("crypto_model_fold_coverage_invalid")
    if type(refs) is not list or len(refs) != len(folds) * len(model_ids):
        raise ValueError("crypto_model_trial_references_invalid")
    models = []
    required = {"model.json"}
    for fold in folds:
        relative = f"models/{fold.fold_id}/metadata.json"
        required.add(relative)
        record = read_manifest(root / relative)
        entries = record.get("models")
        if (
            record.get("schema_id") != CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
            or type(entries) is not list
            or tuple(entry.get("model_id") for entry in entries) != model_ids
        ):
            raise ValueError("crypto_model_members_invalid")
        scopes = build_hft_prediction_catalog(
            train_end=fold.train.end.isoformat(),
            test_start=fold.development_test.start.isoformat(),
            test_end=fold.development_test.end.isoformat(),
        )
        for entry in entries:
            prediction = scopes.get(f"{entry['model_id']}_prediction")
            if (
                entry["fold_id"] != fold.fold_id
                or entry["training_window"] != _intraday_window(fold.train)
                or entry["validation_window"] != _intraday_window(fold.validation)
                or entry["prediction_window"] != _intraday_window(fold.development_test)
                or tuple(entry["training_asset_ids"]) != prediction.training_asset_ids
                or tuple(entry["validation_asset_ids"]) != prediction.training_asset_ids
                or tuple(entry["evaluation_asset_ids"])
                != prediction.evaluation_asset_ids
            ):
                raise ValueError("crypto_model_fold_binding_invalid")
            validate_model_metadata_core(entry["model_metadata"])
            ref = refs[len(models)]
            if entry["trial_semantic_identity"] != ref["semantic_identity"]:
                raise ValueError("crypto_model_trial_references_invalid")
            if trial_layout is None:
                raise ValueError("model_trial_layout_required")
            plan, records = read_model_trial_ledger_reference(trial_layout, ref)
            if (
                plan.market_id != "crypto"
                or plan.research_dataset_id != protocol.scenario_id
                or plan.data_snapshot_id
                != envelope["upstream_identities"][protocol.scenario_id]
                or plan.target_id != envelope["logical_artifact_id"]
                or plan.folds != (fold,)
                or plan.candidate_ids != (entry["model_id"],)
                or plan.research_protocol_identity
                != canonical_json_hash(contract["protocol"])
                or len(records) != 1
                or records[0].status != "completed"
            ):
                raise ValueError("crypto_model_trial_binding_invalid")
            parameters = records[0].payload()["normalized_parameters"]
            metadata = entry["model_metadata"]
            for key in (
                "model_algorithm",
                "random_seed",
                "label_column",
                "input_factor_columns",
                "input_feature_columns",
                "sample_filter_mode",
                "hyperparameters",
                "max_train_rows",
            ):
                if metadata[key] != parameters[key]:
                    raise ValueError("crypto_model_parameters_misaligned")
            expected_inputs = list(parameters["input_factor_columns"]) + list(
                parameters["input_feature_columns"]
            )
            if metadata["input_columns"] != expected_inputs:
                raise ValueError("crypto_model_parameters_misaligned")
            first, last = (
                pd.Timestamp(metadata[key]) for key in ("train_start", "train_end")
            )

            def declared(value, fold=fold):
                return any(
                    pd.Timestamp(part.start) <= value <= pd.Timestamp(part.end)
                    for part in (fold.train, fold.validation)
                )

            if (
                first > last
                or not declared(first)
                or not declared(last)
                or last
                + pd.Timedelta(milliseconds=contract["protocol"]["label_horizon_ms"])
                > pd.Timestamp(fold.validation.end)
            ):
                raise ValueError("crypto_model_refit_window_invalid")
            required.add(f"models/{fold.fold_id}/{entry['model_id']}.pkl")
            paths = entry["validation_prediction_parts"]
            if (
                type(paths) is not list
                or paths
                != [
                    f"models/{fold.fold_id}/validation/{entry['model_id']}/"
                    f"part-{index:06d}.parquet"
                    for index in range(len(paths))
                ]
                or bool(paths) != (entry["model_id"] in HFT_MAKER_CALIBRATION_MODEL_IDS)
                or not set(paths).issubset({member["path"] for member in members})
            ):
                raise ValueError("crypto_model_validation_parts_invalid")
            required.update(paths)
            count = 0
            with closing_iterator(
                _crypto_validation_chunks(root, members, entry)
            ) as chunks:
                for frame in chunks:
                    count += len(frame)
            if paths and count != records[0].validation_metrics["validation_rows"]:
                raise ValueError("crypto_model_validation_count_invalid")
            models.append(entry)
    if {member["path"] for member in members} != required:
        raise ValueError("crypto_model_members_invalid")
    return PersistedCryptoModelArtifact(root, envelope, tuple(members), tuple(models))


def _crypto_validation_chunks(root, members, entry):
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_MODEL_HORIZON_MS,
        HFT_MULTI_ASSET_SCOPES,
    )

    previous = {}
    assets = entry["validation_asset_ids"] or HFT_MULTI_ASSET_SCOPES["pooled"][0]
    start, end = (
        pd.Timestamp(entry["validation_window"][key]) for key in ("start", "end")
    )
    for relative in entry["validation_prediction_parts"]:
        member = next(item for item in members if item["path"] == relative)
        path = root / relative
        identity = dataset_file_identity(path)
        verify_semantic_artifact_member(root, relative_path=relative, expected=member)
        with _opened_parquet(path, expected_identity=identity) as (parquet, _identity):
            if parquet.schema_arrow.remove_metadata() != _CRYPTO_VALIDATION_SCHEMA:
                raise ValueError("crypto_model_validation_schema_invalid")
        chunks = iter_declared_parquet_member_batches(
            path,
            target_chunk_bytes=256 * 1024,
            expected_identity=identity,
        )
        with closing_iterator(chunks):
            for frame in chunks:
                if (
                    frame.empty
                    or frame[
                        [
                            "timestamp",
                            "asset_id",
                            "label",
                            "prediction_score",
                            "label_available_at",
                        ]
                    ]
                    .isna()
                    .any()
                    .any()
                    or not np.isfinite(
                        frame[["label", "prediction_score"]].to_numpy()
                    ).all()
                ):
                    raise ValueError("crypto_model_validation_values_invalid")
                if not (
                    frame.timestamp.between(start, end).all()
                    and (frame.label_available_at <= end).all()
                    and (
                        frame.label_available_at
                        >= frame.timestamp
                        + pd.Timedelta(milliseconds=HFT_MODEL_HORIZON_MS)
                    ).all()
                    and frame.asset_id.isin(assets).all()
                ):
                    raise ValueError("crypto_model_validation_window_invalid")
                for row in frame.itertuples(index=False):
                    if (
                        row.asset_id in previous
                        and row.timestamp <= previous[row.asset_id]
                    ):
                        raise ValueError("crypto_model_validation_key_invalid")
                    previous[row.asset_id] = row.timestamp
                yield frame


@contextmanager
def read_crypto_model_member(artifact, entry, *, resource_policy):
    relative = f"models/{entry['fold_id']}/{entry['model_id']}.pkl"
    with _read_model_member(
        artifact.instance_root,
        artifact.member_records,
        relative,
        resource_policy=resource_policy,
        workspace_factor=4,
    ) as owned:
        value = owned.value
        try:
            validate_model_state(value, entry["model_metadata"])
            yield value
        finally:
            value = None

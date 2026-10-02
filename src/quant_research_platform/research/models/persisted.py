"""Persistence boundary for selected model instances."""

from __future__ import annotations

import os
import pickle
import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.experiments import label_maturity_mask
from quant_research_platform.contracts.json_codec import (
    bounded_rendered_qrp_json_object_size,
    canonical_json_hash,
    json_record_working_set_bytes,
)
from quant_research_platform.contracts.models import (
    A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
    ModelSpec,
    model_spec_payload,
    validate_a_share_model_selection_metadata,
    validate_model_metadata_core,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.models.fitting import validate_model_state
from quant_research_platform.research.models.production import (
    _crypto_model_chronology,
    _refit_model_spec,
    _selection_model_specs,
)
from quant_research_platform.research.models.trial_selection import (
    ModelSampleCoverage,
    crypto_validation_coverage_scope,
    evaluate_model_trial_metrics_from_chunks,
    model_validation_coverage_scope,
    select_best_validation_trial,
)
from quant_research_platform.research.protocols.a_share import quarter_id_for_fold
from quant_research_platform.research.randomness import ROOT_SEED, derive_seed
from quant_research_platform.runtime.execution_metrics import record_run_observations
from quant_research_platform.runtime.resources import (
    OwnedValue,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
)
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    filesystem_stat,
    open_filesystem_file,
    verified_file_handle,
)
from quant_research_platform.storage.manifests import (
    read_bounded_qrp_json,
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    _parquet_frame_reservation,
    dataset_file_identity,
    parquet_table_writer,
)
from quant_research_platform.storage.semantic_artifacts import (
    _member_record,
    read_semantic_artifact_record,
    verify_semantic_artifact_member,
    write_semantic_artifact,
)

_CRYPTO_VALIDATION_SCHEMA = pa.schema(
    (
        ("timestamp", pa.timestamp("ns")),
        ("asset_id", pa.string()),
        ("label", pa.float64()),
        ("prediction_score", pa.float64()),
        ("label_available_at", pa.timestamp("ns")),
        ("book_segment_id", pa.string()),
        ("trade_observation_id", pa.string()),
        ("segment_id", pa.string()),
    )
)


_A_SHARE_VALIDATION_SCHEMA = pa.schema(
    (
        ("timestamp", pa.timestamp("ns")),
        ("asset_id", pa.string()),
        ("label", pa.float64()),
        ("prediction_score", pa.float64()),
        ("label_available_at", pa.timestamp("ns")),
        ("score_eligible", pa.bool_()),
    )
)


@dataclass(frozen=True)
class PersistedQuarterlyModelArtifact:
    model_id: str
    candidate_models: tuple[dict, ...]
    selections: dict[str, dict[str, dict]]
    final_models: tuple[dict, ...]
    instance_root: Path
    envelope: dict
    member_records: tuple
    trial_ledgers: tuple
    member_identities: dict

    def selection(self, group, quarter_id, *, access_window):
        if group not in {"global", "quarterly"}:
            raise ValueError("a_share_selection_group_invalid")
        decision = (
            "final" if access_window == "holdout" or group == "global" else quarter_id
        )
        if decision == "final" and decision not in self.selections[group]:
            raise ValueError("model_selection_requires_complete_development")
        return self.selections[group][decision]

    def prediction_models(self, window, *, selection_group, access_window):
        if access_window not in {"development", "holdout"}:
            raise ValueError("unknown_a_share_access_window")
        protocol = self.envelope["semantic_metadata"]["target_contract"]["protocol"]
        boundary = TimeWindow(
            **protocol[
                "holdout_window" if access_window == "holdout" else "development_window"
            ]
        )
        if pd.Timestamp(window.start) < pd.Timestamp(boundary.start) or pd.Timestamp(
            window.end
        ) > pd.Timestamp(boundary.end):
            raise PermissionError("a_share_prediction_window_invalid")
        result = []
        for period in pd.period_range(window.start, window.end, freq="Q"):
            quarter_id = str(period)
            selection = self.selection(
                selection_group, quarter_id, access_window=access_window
            )
            for family in selection["families"]:
                if access_window == "development":
                    entry = next(
                        item
                        for item in self.candidate_models
                        if item["quarter_id"] == quarter_id
                        and item["family_id"] == family["family_id"]
                        and item["model_id"] == family["selected_config_id"]
                    )
                    prediction_window = entry["validation_window"]
                else:
                    entry = next(
                        item
                        for item in self.final_models
                        if item["selection_group"] == selection_group
                        and item["family_id"] == family["family_id"]
                    )
                    prediction_window = _window_payload(
                        TimeWindow(
                            max(pd.Timestamp(window.start), period.start_time),
                            min(pd.Timestamp(window.end), period.end_time.normalize()),
                        )
                    )
                result.append(
                    {
                        **entry,
                        "selection_group": selection_group,
                        "quarter_id": quarter_id,
                        "decision_id": selection["decision_id"],
                        "prediction_source": "validation"
                        if access_window == "development"
                        else "final",
                        "prediction_window": prediction_window,
                    }
                )
        return tuple(result)

    def prediction_lineage(self, window, *, selection_group, access_window):
        models = self.prediction_models(
            window, selection_group=selection_group, access_window=access_window
        )
        return {
            "model_artifact_id": self.model_id,
            "model_artifact_identity": self.envelope["instance_identity"],
            "model_artifact_schema": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
            "selection_group": selection_group,
            "quarterly_models": [
                {
                    key: item[key]
                    for key in (
                        "quarter_id",
                        "decision_id",
                        "prediction_source",
                        "family_id",
                        "model_id",
                        "prediction_window",
                    )
                }
                for item in models
            ],
            "champion_by_quarter": {
                item["quarter_id"]: self.selection(
                    selection_group, item["quarter_id"], access_window=access_window
                )["selected_family_id"]
                for item in models
            },
        }

    def validation_chunks(self, entry):
        return _a_share_validation_chunks(
            self.instance_root,
            entry,
            member_identities=self.member_identities,
            development_end=self.envelope["semantic_metadata"]["target_contract"][
                "protocol"
            ]["development_window"]["end"],
        )


def _model_metadata_bytes(value):
    return json_record_working_set_bytes(
        bounded_rendered_qrp_json_object_size(
            value, lazy_arrays={}, max_bytes=sys.maxsize
        )
    )


def _retain_model_metadata(
    value, *, lease, error="model_metadata_exceeds_memory_capacity"
):
    size = _model_metadata_bytes(value)
    current_runtime_resource_policy().require_working_set(size, error=error)
    lease.grow(size)
    return size


def _admit_model_metadata(size, *, lease):
    policy = current_runtime_resource_policy()
    cap = policy.explicit_memory_cap_bytes
    if cap is None or policy.used_memory_bytes + size <= cap:
        policy.require_working_set(size, error="model_metadata_exceeds_memory_capacity")
    lease.grow(size)


def _write_model_metadata(path, payload, *, lease):
    retained = lease.reserved_bytes
    try:
        write_internal_artifact_manifest(
            path, payload, _admit=partial(_admit_model_metadata, lease=lease)
        )
    except BaseException as error:
        clear_exception_frames(error)
        raise
    finally:
        payload = None
        lease.shrink_to(retained)


def _persisted_model_metadata(result, *, retain, fit_cutoff=None, fit_coverage_id=None):
    algorithm = result.metadata["model_algorithm"]
    rank_parameters = None
    if algorithm in {"equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"}:
        predictor = result.prediction_model
        parameter = (
            "directions" if algorithm == "equal_weight_rank_ensemble" else "weights"
        )
        rank_parameters = {
            "columns": predictor.columns,
            parameter: getattr(predictor, parameter),
        }
    cutoff = (
        None if fit_cutoff is None else time_value_isoformat(fit_cutoff, date_only=True)
    )
    retain(
        {
            "metadata": result.metadata,
            "rank_parameters": rank_parameters,
            "fit_cutoff": cutoff,
            "fit_coverage_id": fit_coverage_id,
        }
    )
    metadata = dict(result.metadata)
    if rank_parameters is not None:
        metadata["rank_parameters"] = {
            key: list(value) for key, value in rank_parameters.items()
        }
    if fit_cutoff is not None:
        metadata["fit_cutoff"] = cutoff
        metadata["fit_coverage_id"] = fit_coverage_id
    metadata = validate_model_metadata_core(metadata, quarterly=fit_cutoff is not None)
    validate_model_state(result.prediction_model, metadata)
    return metadata


def write_quarterly_model_artifact(
    logical_root,
    *,
    model_id,
    envelope,
    model_results,
    trial_layout,
    verify_source,
):
    validate_run_id(model_id)
    if model_id != envelope.logical_artifact_id:
        raise ValueError("model_artifact_identity_invalid")
    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("research.model.artifact.metadata", 0) as metadata_lease,
    ):
        operational = {
            "model_trial_ledgers": [],
            "model_phase_observations": {"environments": [], "phases": []},
        }

        admit = partial(_admit_model_metadata, lease=metadata_lease)
        retain = partial(_retain_model_metadata, lease=metadata_lease)
        write_metadata = partial(_write_model_metadata, lease=metadata_lease)
        operational_bytes = 0

        def record_observation(value):
            nonlocal operational_bytes
            operational_bytes += retain(value)
            value = dict(value)
            observation = dict(value["observations"])
            environment = dict(observation["environment"])
            created_at = environment.pop("created_at_utc")
            costs = operational["model_phase_observations"]
            if environment not in costs["environments"]:
                costs["environments"].append(environment)
            observation["environment"] = {
                "context": costs["environments"].index(environment),
                "created_at_utc": created_at,
            }
            value["observations"] = observation
            costs["phases"].append(value)

        def write(staging):
            nonlocal operational_bytes
            candidate_models = []
            final_models = []
            selections = {"global": {}, "quarterly": {}}
            candidate_files = {}
            validation_owner = ExitStack()
            validation_writer = None
            current_path = None

            def write_validation(fold, family, spec, frame):
                nonlocal validation_writer, current_path
                path = f"models/{quarter_id_for_fold(fold)}/{spec.model_id}.parquet"
                if validation_writer is None:
                    current_path = path
                    target = staging / path
                    filesystem_mkdir(target.parent, parents=True, exist_ok=True)
                    handle = validation_owner.enter_context(
                        handle_preserving_primary(open_filesystem_file(target, "xb"))
                    )
                    validation_writer = validation_owner.enter_context(
                        parquet_table_writer(
                            handle,
                            parent=target.parent,
                            compression="zstd",
                            schema=_A_SHARE_VALIDATION_SCHEMA,
                        )
                    )
                if path != current_path:
                    raise ValueError("model_validation_candidate_interleaving")
                with _parquet_frame_reservation(frame, borrowed_input=True):
                    table = None
                    try:
                        table = pa.Table.from_pandas(
                            frame.rename(columns={spec.label_column: "label"}),
                            schema=_A_SHARE_VALIDATION_SCHEMA,
                            preserve_index=False,
                        )
                        validation_writer(table)
                    finally:
                        table = None

            def write_candidate(fold, family, spec, result):
                nonlocal validation_writer, current_path
                if validation_writer is None:
                    raise ValueError("model_validation_predictions_missing")
                validation_owner.close()
                validation_writer = None
                identity = dataset_file_identity(staging / current_path)
                record = _member_record(
                    current_path, staging / current_path, identity=identity
                )
                candidate_files[current_path] = identity
                entry = {
                    "quarter_id": quarter_id_for_fold(fold),
                    "family_id": family,
                    "model_id": spec.model_id,
                    "training_window": _window_payload(fold.train),
                    "validation_window": _window_payload(fold.validation),
                    "model_metadata": None,
                    "validation_prediction": record,
                }
                retain(entry)
                entry["model_metadata"] = _persisted_model_metadata(
                    result, retain=retain
                )
                candidate_models.append(entry)
                current_path = None

            def write_trial(fold, family, reference):
                nonlocal operational_bytes
                for entry in candidate_models:
                    if (
                        entry["quarter_id"] == quarter_id_for_fold(fold)
                        and entry["family_id"] == family
                    ):
                        entry["trial_semantic_identity"] = reference[
                            "semantic_identity"
                        ]
                operational_bytes += retain(reference)
                operational["model_trial_ledgers"].append(dict(reference))

            def select_models(group, decision_id, cutoff, scoring_window):
                retained = metadata_lease.reserved_bytes
                observation = {}
                with record_run_observations(policy, observation):
                    value = _select_a_share_models(
                        staging,
                        candidate_models,
                        envelope.semantic_metadata["target_contract"],
                        group=group,
                        decision_id=decision_id,
                        cutoff=cutoff,
                        scoring_window=scoring_window,
                        member_identities=candidate_files,
                        dataset_identity=envelope.upstream_identities[
                            envelope.semantic_metadata["target_contract"]["protocol"][
                                "scenario_id"
                            ]
                        ],
                        _retain=retain,
                    )
                metadata_lease.shrink_to(retained + _model_metadata_bytes(value))
                record_observation(
                    {
                        "phase": "group_scoring",
                        "selection_group": group,
                        "family_id": None,
                        "quarter_id": decision_id,
                        "fit_count": 0,
                        "observations": observation,
                    }
                )
                selections[group][decision_id] = value
                return value

            try:
                with (
                    handle_preserving_primary(validation_owner),
                    closing_iterator(
                        iter(
                            model_results(
                                on_validation=write_validation,
                                on_candidate=write_candidate,
                                on_trial=write_trial,
                                select_models=select_models,
                                on_observation=record_observation,
                            )
                        )
                    ) as source,
                ):
                    for owned in source:
                        owned.lease.transfer("research.model.final.persist")
                        member = owned.value
                        try:
                            path = (
                                f"models/final/{member.selection_group}/"
                                f"{member.family_id}.pkl"
                            )
                            filesystem_mkdir(
                                (staging / path).parent, parents=True, exist_ok=True
                            )
                            metadata = _persisted_model_metadata(
                                member.model_result,
                                retain=retain,
                                fit_cutoff=member.fit_cutoff,
                                fit_coverage_id=member.fit_coverage_id,
                            )
                            _write_model_member(
                                staging / path,
                                member.model_result.prediction_model,
                                state_bytes=owned.lease.reserved_bytes,
                            )
                            entry = {
                                "selection_group": member.selection_group,
                                "family_id": member.family_id,
                                "model_id": member.spec.model_id,
                                "fit_windows": [
                                    _window_payload(part) for part in member.fit_windows
                                ],
                                "fit_cutoff": time_value_isoformat(
                                    member.fit_cutoff, date_only=True
                                ),
                                "model_metadata": None,
                                "refit_trial_semantic_identity": (
                                    member.refit_trial_reference["semantic_identity"]
                                ),
                            }
                            retain(entry)
                            entry["model_metadata"] = metadata
                            final_models.append(entry)
                            operational_bytes += retain(member.refit_trial_reference)
                            operational["model_trial_ledgers"].append(
                                dict(member.refit_trial_reference)
                            )
                        finally:
                            member = None
                            if owned.lease.active:
                                primary = sys.exception()
                                try:
                                    owned.release(owner="research.model.final.persist")
                                except BaseException as cleanup:
                                    if primary is None:
                                        raise
                                    primary.add_note(
                                        "model release also failed: "
                                        + type(cleanup).__name__
                                    )
                            owned = None
                quarters = tuple(
                    dict.fromkeys(entry["quarter_id"] for entry in candidate_models)
                )
                for quarter in quarters:
                    write_metadata(
                        staging / "models" / quarter / "metadata.json",
                        {
                            "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                            "quarter_id": quarter,
                            "candidate_models": [
                                entry
                                for entry in candidate_models
                                if entry["quarter_id"] == quarter
                            ],
                        },
                    )
                for group, decisions in selections.items():
                    for decision_id, decision in decisions.items():
                        target = staging / "selections" / group / f"{decision_id}.json"
                        filesystem_mkdir(target.parent, parents=True, exist_ok=True)
                        write_metadata(
                            target,
                            {
                                "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                                "selection": decision,
                            },
                        )
                final_groups = tuple(
                    dict.fromkeys(entry["selection_group"] for entry in final_models)
                )
                for group in final_groups:
                    write_metadata(
                        staging / "models" / "final" / group / "metadata.json",
                        {
                            "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                            "selection_group": group,
                            "final_models": [
                                entry
                                for entry in final_models
                                if entry["selection_group"] == group
                            ],
                        },
                    )
                write_metadata(
                    staging / "model.json",
                    {
                        "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                        "model_id": model_id,
                        "validation_quarters": list(quarters),
                        "selection_decisions": {
                            group: list(decisions)
                            for group, decisions in selections.items()
                        },
                        "final_groups": list(final_groups),
                    },
                )
            except BaseException as error:
                clear_exception_frames(error)
                raise
            finally:
                candidate_models.clear()
                final_models.clear()
                selections.clear()
                candidate_files.clear()
                entry = metadata = member = owned = decision = decisions = None
                del entry, metadata, member, owned, decision, decisions
                metadata_lease.shrink_to(operational_bytes)

        def verify_models(root, members):
            retained = metadata_lease.reserved_bytes
            try:
                admit(
                    sum(
                        2 * json_record_working_set_bytes(member["size"])
                        for member in members
                        if str(member["path"]).endswith(".json")
                    )
                )
                _validate_a_share_model_payload(
                    root,
                    envelope.payload(_admit=admit),
                    members,
                    operational["model_trial_ledgers"],
                    trial_layout,
                )
                verify_source()
            except BaseException as error:
                clear_exception_frames(error)
                raise
            finally:
                metadata_lease.shrink_to(retained)

        try:
            return write_semantic_artifact(
                logical_root,
                envelope,
                write,
                operational_metadata=operational,
                staged_validator=verify_models,
            )
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            operational.clear()


def _a_share_validation_chunks(root, entry, *, member_identities, development_end):
    record = entry["validation_prediction"]
    path = record["path"]
    identity = member_identities[path]
    with _opened_parquet(root / path, expected_identity=identity) as (
        parquet,
        _identity,
        _handle,
    ):
        if parquet.schema_arrow.remove_metadata() != _A_SHARE_VALIDATION_SCHEMA:
            raise ValueError("a_share_validation_schema_invalid")
    previous = {}
    start, end = (
        pd.Timestamp(entry["validation_window"][key]) for key in ("start", "end")
    )
    source = iter_declared_parquet_member_batches(
        root / path, target_chunk_bytes=256 * 1024, expected_identity=identity
    )
    with closing_iterator(source):
        for frame in source:
            labeled = frame["label"].notna()
            if (
                frame.empty
                or frame[
                    ["timestamp", "asset_id", "prediction_score", "score_eligible"]
                ]
                .isna()
                .any()
                .any()
                or not np.isfinite(frame["prediction_score"].to_numpy()).all()
                or not np.isfinite(frame.loc[labeled, "label"].to_numpy()).all()
                or not labeled.equals(frame["label_available_at"].notna())
                or not frame.timestamp.between(start, end).all()
                or not (
                    frame.loc[labeled, "label_available_at"]
                    > frame.loc[labeled, "timestamp"]
                ).all()
                or not (
                    frame.loc[labeled, "label_available_at"]
                    <= pd.Timestamp(development_end)
                ).all()
            ):
                raise ValueError("a_share_validation_values_invalid")
            for timestamp, asset in frame[["timestamp", "asset_id"]].itertuples(
                index=False, name=None
            ):
                if (
                    type(asset) is not str
                    or not asset
                    or (asset in previous and timestamp <= previous[asset])
                ):
                    raise ValueError("a_share_validation_keys_invalid")
                previous[asset] = timestamp
            yield frame


def _select_a_share_models(
    root,
    candidate_models,
    contract,
    *,
    group,
    decision_id,
    cutoff,
    scoring_window,
    member_identities,
    dataset_identity,
    _retain=None,
):
    if _retain is not None:
        # Borrowed entries bound the new indexes, metric rows and result containers.
        _retain({"selection_entries": candidate_models})
    protocol = contract["protocol"]
    family_ids = tuple(value["model_id"] for value in contract["model_specs"])
    eligible = [
        entry
        for entry in candidate_models
        if pd.Timestamp(entry["validation_window"]["end"]) <= pd.Timestamp(cutoff)
        and pd.Timestamp(entry["validation_window"]["end"])
        >= pd.Timestamp(scoring_window.start)
        and pd.Timestamp(entry["validation_window"]["start"])
        <= pd.Timestamp(scoring_window.end)
    ]
    if not eligible:
        raise ValueError("a_share_selection_predictions_missing")
    quarters = tuple(sorted({entry["quarter_id"] for entry in eligible}))
    source_predictions = sorted(
        (entry["validation_prediction"] for entry in eligible),
        key=lambda item: item["path"],
    )
    parents = sorted({entry["trial_semantic_identity"] for entry in eligible})
    scope = {
        "dataset_identity": dataset_identity,
        "prediction_members": source_predictions,
        "cutoff": time_value_isoformat(pd.Timestamp(cutoff), date_only=True),
        "scoring_window": _window_payload(scoring_window),
        "label": contract["model_specs"][0]["label_column"],
        "eligibility": "shared_candidate_prediction_keys",
    }
    coverage = ModelSampleCoverage(scope)
    family_rows = []
    selected_rows = []
    first_candidate = True
    for family_id in family_ids:
        candidate_ids = tuple(
            entry["model_id"]
            for entry in eligible
            if entry["quarter_id"] == quarters[0] and entry["family_id"] == family_id
        )
        if not candidate_ids or any(
            tuple(
                entry["model_id"]
                for entry in eligible
                if entry["quarter_id"] == quarter and entry["family_id"] == family_id
            )
            != candidate_ids
            for quarter in quarters
        ):
            raise ValueError("a_share_selection_candidate_coverage_missing")
        rows = []
        for candidate_id in candidate_ids:
            entries = tuple(
                entry
                for entry in eligible
                if entry["family_id"] == family_id and entry["model_id"] == candidate_id
            )

            def chunks(entries=entries, populate=first_candidate):
                for entry in entries:
                    with closing_iterator(
                        _a_share_validation_chunks(
                            root,
                            entry,
                            member_identities=member_identities,
                            development_end=protocol["development_window"]["end"],
                        )
                    ) as frames:
                        for frame in frames:
                            policy = current_runtime_resource_policy()
                            with policy.lease(
                                "research.model.selection.maturity",
                                max(1, 3 * pandas_frame_payload_bytes(frame)),
                            ):
                                mask = label_maturity_mask(
                                    frame,
                                    fitting_cutoff=cutoff,
                                    availability_column="label_available_at",
                                    target_column="label",
                                )
                                mask &= frame["score_eligible"]
                                mask &= frame.timestamp.between(
                                    pd.Timestamp(scoring_window.start),
                                    pd.Timestamp(scoring_window.end),
                                )
                                mature = frame.loc[mask]
                                if populate:
                                    coverage.add(mature)
                                yield mature

            metrics = evaluate_model_trial_metrics_from_chunks(
                chunks(),
                model_id=candidate_id,
                label_column="label",
                prediction_column="prediction_score",
                expected_coverage=coverage,
            )
            first_candidate = False
            rows.append(
                {
                    key: None
                    if isinstance(value, float) and np.isnan(value)
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                    for key, value in metrics.items()
                }
            )
        selected_id = select_best_validation_trial(
            pd.DataFrame.from_records(rows),
            trial_id_column="model_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=candidate_ids,
        )
        selected = next(row for row in rows if row["model_id"] == selected_id)
        selected_rows.append({**selected, "model_id": family_id})
        family_rows.append(
            {
                "family_id": family_id,
                "config_tie_break_order": list(candidate_ids),
                "selected_config_id": selected_id,
                "trial_metrics": rows,
            }
        )
    champion = select_best_validation_trial(
        pd.DataFrame.from_records(selected_rows),
        trial_id_column="model_id",
        selection_metric="validation_rankic",
        maximize=True,
        validation_metric_columns=("validation_rankic",),
        test_metric_columns=(),
        tie_break_order=family_ids,
    )
    result = {
        "status": "selected",
        "selection_group": group,
        "decision_id": decision_id,
        "cutoff": time_value_isoformat(pd.Timestamp(cutoff), date_only=True),
        "selection_metric": "validation_rankic",
        "selection_direction": "maximize",
        "validation_start": time_value_isoformat(scoring_window.start, date_only=True),
        "validation_end": time_value_isoformat(scoring_window.end, date_only=True),
        "candidate_family_ids": list(family_ids),
        "tie_break_order": list(family_ids),
        "selected_family_id": champion,
        "selected_model_id": next(
            family["selected_config_id"]
            for family in family_rows
            if family["family_id"] == champion
        ),
        "families": family_rows,
        "common_validation_coverage": {
            "semantic_identity": coverage.semantic_identity,
            "row_count": coverage.row_count,
        },
        "selection_ledger_semantic_identities": parents,
        "prediction_members": source_predictions,
        "label_dataset_identity": dataset_identity,
    }
    validate_a_share_model_selection_metadata(result)
    return result


def read_quarterly_model_artifact(
    instance_root,
    *,
    expected_identity=None,
    trial_layout=None,
    expected_member_identities=None,
):
    instance_root = Path(instance_root)
    member_identities = {}
    operational = {}
    envelope, members = read_semantic_artifact_record(
        instance_root,
        expected_identity=expected_identity,
        member_identities=member_identities,
        expected_member_identities=expected_member_identities,
        operational_metadata=operational,
    )
    return _validate_a_share_model_payload(
        instance_root,
        envelope,
        members,
        operational.get("model_trial_ledgers"),
        trial_layout,
        member_identities=member_identities,
    )


def _validate_a_share_model_payload(
    root,
    envelope,
    members,
    references,
    trial_layout,
    *,
    member_identities=None,
):
    if trial_layout is None:
        raise ValueError("model_trial_layout_required")
    if member_identities is None:
        member_identities = {
            member["path"]: verify_semantic_artifact_member(
                root, relative_path=member["path"], expected=member
            )
            for member in members
        }
    member_records = {member["path"]: member for member in members}
    payload = read_bounded_qrp_json(
        root / "model.json", expected_identity=member_identities["model.json"]
    )[1]
    if (
        set(payload)
        != {
            "schema_id",
            "model_id",
            "validation_quarters",
            "selection_decisions",
            "final_groups",
        }
        or payload["schema_id"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
        or payload["model_id"] != envelope["logical_artifact_id"]
        or type(payload["validation_quarters"]) is not list
        or type(payload["selection_decisions"]) is not dict
        or set(payload["selection_decisions"]) != {"global", "quarterly"}
        or type(payload["final_groups"]) is not list
        or type(references) is not list
    ):
        raise ValueError("model_artifact_manifest_invalid")
    contract = envelope["semantic_metadata"]["target_contract"]
    protocol = contract["protocol"]
    dataset_identity = envelope["upstream_identities"][protocol["scenario_id"]]
    specs = {
        item["model_id"]: ModelSpec(
            **{
                **item,
                "input_factor_columns": tuple(item["input_factor_columns"]),
                "input_feature_columns": tuple(item["input_feature_columns"]),
            }
        )
        for item in contract["model_specs"]
    }
    configurations = {
        item["model_id"]: item for item in protocol["research_configuration"]["models"]
    }
    if tuple(specs) != tuple(configurations):
        raise ValueError("model_trial_family_coverage_mismatch")
    family_ids = tuple(specs)
    quarters = tuple(
        str(period)
        for period in pd.period_range(
            protocol["model_chronology"]["warmup_window"]["start"],
            envelope["research_window"]["end"],
            freq="Q",
        )
    )
    expected_keys = []
    candidate_orders = {}
    for family_id, configuration in configurations.items():
        parameters = configuration.get(
            "parameter_candidates", [configuration["hyperparameters"]]
        )
        candidate_orders[family_id] = tuple(
            family_id if len(parameters) == 1 else f"{family_id}__config_{index}"
            for index in range(len(parameters))
        )
    for quarter in quarters:
        for family_id in family_ids:
            expected_keys.extend(
                (quarter, family_id, candidate_id)
                for candidate_id in candidate_orders[family_id]
            )
    if payload["validation_quarters"] != list(quarters):
        raise ValueError("model_candidate_matrix_incomplete")
    required = {"model.json"}

    def read_member(path):
        if path not in member_identities:
            raise ValueError("model_artifact_members_invalid")
        required.add(path)
        return read_bounded_qrp_json(
            root / path, expected_identity=member_identities[path]
        )[1]

    candidates = []
    for quarter in quarters:
        shard = read_member(f"models/{quarter}/metadata.json")
        if set(shard) != {"schema_id", "quarter_id", "candidate_models"} or (
            shard["schema_id"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
            or shard["quarter_id"] != quarter
            or type(shard["candidate_models"]) is not list
        ):
            raise ValueError("model_candidate_shard_invalid")
        candidates.extend(shard["candidate_models"])
    selections = {"global": {}, "quarterly": {}}
    for group, decisions in payload["selection_decisions"].items():
        if (
            type(decisions) is not list
            or any(type(value) is not str for value in decisions)
            or len(set(decisions)) != len(decisions)
            or any(value != "final" and value not in quarters for value in decisions)
        ):
            raise ValueError("model_selection_decisions_invalid")
        for decision_id in decisions:
            shard = read_member(f"selections/{group}/{decision_id}.json")
            if set(shard) != {"schema_id", "selection"} or (
                shard["schema_id"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
            ):
                raise ValueError("model_selection_shard_invalid")
            selections[group][decision_id] = shard["selection"]
    expected_groups = (
        ["global", "quarterly"]
        if pd.Timestamp(envelope["research_window"]["end"])
        >= pd.Timestamp(protocol["development_window"]["end"])
        else []
    )
    if payload["final_groups"] != expected_groups:
        raise ValueError("model_final_matrix_incomplete")
    final_models = []
    for group in expected_groups:
        shard = read_member(f"models/final/{group}/metadata.json")
        if set(shard) != {"schema_id", "selection_group", "final_models"} or (
            shard["schema_id"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
            or shard["selection_group"] != group
            or type(shard["final_models"]) is not list
        ):
            raise ValueError("model_final_shard_invalid")
        final_models.extend(shard["final_models"])
    if tuple(
        (entry.get("quarter_id"), entry.get("family_id"), entry.get("model_id"))
        for entry in candidates
    ) != tuple(expected_keys):
        raise ValueError("model_candidate_matrix_incomplete")
    by_identity = {
        reference["semantic_identity"]: reference for reference in references
    }
    if len(by_identity) != len(references):
        raise ValueError("model_trial_references_duplicate")
    used_references = set()
    folds = {}
    for quarter in quarters:
        for family_id in family_ids:
            entries = [
                entry
                for entry in candidates
                if entry["quarter_id"] == quarter and entry["family_id"] == family_id
            ]
            identities = {entry["trial_semantic_identity"] for entry in entries}
            if len(identities) != 1:
                raise ValueError("model_trial_reference_semantics_invalid")
            identity = next(iter(identities))
            if identity not in by_identity or identity in used_references:
                raise ValueError("model_trial_references_missing")
            used_references.add(identity)
            with read_model_trial_ledger_reference(
                trial_layout, by_identity[identity]
            ) as (plan, records):
                if (
                    plan.stage != "selection"
                    or plan.market_id != "a_share"
                    or plan.research_dataset_id != protocol["scenario_id"]
                    or plan.data_snapshot_id != dataset_identity
                    or plan.input_lineage != (dataset_identity,)
                    or plan.target_id != envelope["logical_artifact_id"]
                    or plan.research_protocol_identity != canonical_json_hash(protocol)
                    or plan.candidate_family_id != family_id
                    or len(plan.folds) != 1
                    or plan.candidate_ids != candidate_orders[family_id]
                    or plan.tie_break_order != plan.candidate_ids
                    or plan.trial_budget != len(plan.candidate_ids)
                    or plan.selection_metric != "validation_rankic"
                    or plan.selection_direction != "maximize"
                    or _window_payload(plan.test) != protocol["holdout_window"]
                    or tuple(record.candidate_id for record in records)
                    != plan.candidate_ids
                    or any(record.status != "completed" for record in records)
                ):
                    raise ValueError("model_trial_artifact_binding_invalid")
                fold = plan.folds[0]
                if (
                    quarter_id_for_fold(fold) != quarter
                    or str(pd.Period(fold.validation.start, freq="Q")) != quarter
                    or str(pd.Period(fold.validation.end, freq="Q")) != quarter
                    or pd.Timestamp(fold.train.start).year
                    != pd.Timestamp(protocol["formal_dataset_window"]["start"]).year
                    or (quarter in folds and folds[quarter] != fold)
                ):
                    raise ValueError("model_trial_fold_binding_invalid")
                folds[quarter] = fold
                configuration = configurations[family_id]
                parameters = configuration.get(
                    "parameter_candidates", [configuration["hyperparameters"]]
                )
                for entry, record in zip(entries, records, strict=True):
                    if set(entry) != {
                        "quarter_id",
                        "family_id",
                        "model_id",
                        "training_window",
                        "validation_window",
                        "model_metadata",
                        "validation_prediction",
                        "trial_semantic_identity",
                    }:
                        raise ValueError("model_candidate_metadata_invalid")
                    metadata = validate_model_metadata_core(entry["model_metadata"])
                    expected_spec = _selection_model_specs(
                        specs[family_id],
                        parameters,
                        n_train=record.metrics["n_train"],
                        root_seed=protocol["root_seed"],
                        quarter_id=quarter,
                    )[plan.candidate_ids.index(record.candidate_id)]
                    expected_parameters = model_spec_payload(expected_spec)
                    if (
                        record.payload()["normalized_parameters"] != expected_parameters
                        or plan.protocol_payload()["candidate_parameters"][
                            record.candidate_id
                        ]
                        != expected_parameters
                        or record.metrics["n_train"]
                        != metadata["rows_after_input_label_dropna"]
                        or entry["training_window"] != _window_payload(fold.train)
                        or entry["validation_window"]
                        != _window_payload(fold.validation)
                        or pd.Timestamp(metadata["train_start"])
                        < pd.Timestamp(fold.train.start)
                        or pd.Timestamp(metadata["train_end"])
                        > pd.Timestamp(fold.train.end)
                    ):
                        raise ValueError("model_trial_parameters_misaligned")
                    _validate_a_share_model_parameters(metadata, expected_parameters)
                    member = entry["validation_prediction"]
                    path = f"models/{quarter}/{entry['model_id']}.parquet"
                    if member.get("path") != path or member != member_records.get(path):
                        raise ValueError("model_validation_member_invalid")
                    required.add(path)
                    expected_coverage = ModelSampleCoverage(
                        model_validation_coverage_scope(
                            dataset_identity, fold, tuple(specs.values())
                        )
                    )

                    def scoring_frames(
                        entry=entry, fold=fold, coverage=expected_coverage
                    ):
                        with closing_iterator(
                            _a_share_validation_chunks(
                                root,
                                entry,
                                member_identities=member_identities,
                                development_end=protocol["development_window"]["end"],
                            )
                        ) as frames:
                            for frame in frames:
                                with current_runtime_resource_policy().lease(
                                    "research.model.validation_reader",
                                    max(1, 3 * pandas_frame_payload_bytes(frame)),
                                ):
                                    mature = frame.loc[
                                        label_maturity_mask(
                                            frame,
                                            fitting_cutoff=fold.validation.end,
                                            availability_column="label_available_at",
                                            target_column="label",
                                        )
                                        & frame["score_eligible"]
                                    ]
                                    coverage.add(mature)
                                    yield mature

                    metrics = evaluate_model_trial_metrics_from_chunks(
                        scoring_frames(),
                        model_id=entry["model_id"],
                        label_column="label",
                        prediction_column="prediction_score",
                        expected_coverage=expected_coverage,
                    )
                    terminal = {
                        key: None
                        if isinstance(value, float) and np.isnan(value)
                        else value.item()
                        if isinstance(value, np.generic)
                        else value
                        for key, value in metrics.items()
                        if key != "model_id"
                    }
                    terminal["n_train"] = metadata["rows_after_input_label_dropna"]
                    if (
                        dict(record.metrics) != terminal
                        or record.common_coverage_id
                        != expected_coverage.semantic_identity
                    ):
                        raise ValueError("model_validation_trial_mismatch")
    cutoff = pd.Timestamp(protocol["development_window"]["end"])
    complete = pd.Timestamp(envelope["research_window"]["end"]) >= cutoff
    expected_decisions = []
    for quarter, fold in folds.items():
        if pd.Timestamp(fold.validation.start) >= pd.Timestamp(
            protocol["development_window"]["start"]
        ):
            expected_decisions.append(
                (
                    "quarterly",
                    quarter,
                    pd.Timestamp(fold.train.end),
                    TimeWindow(
                        pd.Timestamp(fold.validation.start).to_period("Q").start_time
                        - pd.DateOffset(
                            months=protocol["model_chronology"][
                                "scoring_lookback_months"
                            ]
                        ),
                        fold.train.end,
                    ),
                )
            )
    if complete:
        expected_decisions.extend(
            (
                (
                    "global",
                    "final",
                    cutoff,
                    TimeWindow(
                        pd.Timestamp(protocol["development_window"]["start"]), cutoff
                    ),
                ),
                (
                    "quarterly",
                    "final",
                    cutoff,
                    TimeWindow(
                        cutoff
                        + pd.Timedelta(days=1)
                        - pd.DateOffset(
                            months=protocol["model_chronology"][
                                "scoring_lookback_months"
                            ]
                        ),
                        cutoff,
                    ),
                ),
            )
        )
    if any(type(value) is not dict for value in selections.values()) or {
        (group, key) for group, values in selections.items() for key in values
    } != {(group, key) for group, key, _cutoff, _window in expected_decisions}:
        raise ValueError("model_selection_decisions_incomplete")
    for group, key, decision_cutoff, window in expected_decisions:
        expected = _select_a_share_models(
            root,
            candidates,
            contract,
            group=group,
            decision_id=key,
            cutoff=decision_cutoff,
            scoring_window=window,
            member_identities=member_identities,
            dataset_identity=dataset_identity,
        )
        if selections[group][key] != expected:
            raise ValueError("model_selection_decision_mismatch")
    expected_final = (
        tuple(
            (group, family)
            for group in ("global", "quarterly")
            for family in family_ids
        )
        if complete
        else ()
    )
    if (
        tuple(
            (entry.get("selection_group"), entry.get("family_id"))
            for entry in final_models
        )
        != expected_final
    ):
        raise ValueError("model_final_matrix_incomplete")
    fit_windows = (
        TimeWindow(pd.Timestamp(protocol["formal_dataset_window"]["start"]), cutoff),
    )
    for entry in final_models:
        if set(entry) != {
            "selection_group",
            "family_id",
            "model_id",
            "fit_windows",
            "fit_cutoff",
            "model_metadata",
            "refit_trial_semantic_identity",
        }:
            raise ValueError("model_final_metadata_invalid")
        group, family_id = entry["selection_group"], entry["family_id"]
        decision = selections[group]["final"]
        family = next(
            item for item in decision["families"] if item["family_id"] == family_id
        )
        metadata = validate_model_metadata_core(entry["model_metadata"], quarterly=True)
        parameters = configurations[family_id].get(
            "parameter_candidates", [configurations[family_id]["hyperparameters"]]
        )[candidate_orders[family_id].index(family["selected_config_id"])]
        expected_spec = _refit_model_spec(
            replace(specs[family_id], model_id=family["selected_config_id"]),
            candidate_parameters=parameters,
            n_train=metadata["rows_after_input_label_dropna"],
            quarter_id="final",
            root_seed=protocol["root_seed"],
        )
        normalized = model_spec_payload(expected_spec)
        identity = entry["refit_trial_semantic_identity"]
        if identity not in by_identity or identity in used_references:
            raise ValueError("model_trial_references_missing")
        used_references.add(identity)
        with read_model_trial_ledger_reference(trial_layout, by_identity[identity]) as (
            plan,
            records,
        ):
            if (
                plan.stage != "refit"
                or plan.selection_group != group
                or plan.selection_ledger_semantic_identities
                != tuple(decision["selection_ledger_semantic_identities"])
                or plan.market_id != "a_share"
                or plan.research_dataset_id != protocol["scenario_id"]
                or plan.data_snapshot_id != dataset_identity
                or plan.input_lineage != (dataset_identity,)
                or plan.target_id != envelope["logical_artifact_id"]
                or plan.candidate_family_id != family_id
                or plan.research_protocol_identity != canonical_json_hash(protocol)
                or plan.fit_windows != fit_windows
                or pd.Timestamp(plan.fit_cutoff) != cutoff
                or plan.candidate_ids != (expected_spec.model_id,)
                or plan.random_seed != expected_spec.random_seed
                or plan.protocol_payload()["candidate_parameters"]
                != {expected_spec.model_id: normalized}
                or _window_payload(plan.test) != protocol["holdout_window"]
                or len(records) != 1
                or records[0].status != "completed"
                or records[0].payload()["normalized_parameters"] != normalized
                or records[0].common_coverage_id != metadata["fit_coverage_id"]
                or dict(records[0].metrics)
                != {"n_train": metadata["rows_after_input_label_dropna"]}
                or entry["model_id"] != expected_spec.model_id
                or entry["fit_windows"]
                != [_window_payload(part) for part in fit_windows]
                or entry["fit_cutoff"] != cutoff.date().isoformat()
                or metadata["fit_cutoff"] != entry["fit_cutoff"]
                or pd.Timestamp(metadata["train_start"])
                < pd.Timestamp(fit_windows[0].start)
                or pd.Timestamp(metadata["train_end"]) > cutoff
            ):
                raise ValueError("model_trial_refit_misaligned")
        _validate_a_share_model_parameters(metadata, normalized)
        relative = f"models/final/{group}/{family_id}.pkl"
        required.add(relative)
        with _read_model_member(
            root,
            members,
            relative,
            expected_identity=member_identities[relative],
        ) as owned:
            validate_model_state(owned.value, metadata)
    if used_references != set(by_identity) or required != set(member_records):
        raise ValueError("model_artifact_members_invalid")
    return PersistedQuarterlyModelArtifact(
        payload["model_id"],
        tuple(candidates),
        selections,
        tuple(final_models),
        root,
        envelope,
        tuple(members),
        tuple(references),
        member_identities,
    )


def _validate_a_share_model_parameters(metadata, parameters):
    for key in (
        "model_algorithm",
        "random_seed",
        "label_column",
        "input_factor_columns",
        "input_feature_columns",
        "sample_filter_mode",
        "hyperparameters",
        "max_train_rows",
        "training_weighting",
    ):
        if metadata[key] != parameters[key]:
            raise ValueError("model_trial_parameters_misaligned")


@contextmanager
def read_quarterly_predictor(
    artifact,
    selection_group,
    family_id,
    *,
    resource_policy=None,
):
    entry = next(
        item
        for item in artifact.final_models
        if item["selection_group"] == selection_group and item["family_id"] == family_id
    )
    relative = f"models/final/{selection_group}/{family_id}.pkl"
    with _read_model_member(
        artifact.instance_root,
        artifact.member_records,
        relative,
        expected_identity=artifact.member_identities[relative],
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
    root,
    members,
    relative,
    *,
    resource_policy=None,
    workspace_factor=1,
    expected_identity=None,
):
    member = next((item for item in members if item["path"] == relative), None)
    if member is None:
        raise ValueError("model_artifact_model_invalid")
    path = root / relative
    size = filesystem_stat(path, follow_symlinks=False).st_size
    with runtime_resource_policy_scope(resource_policy) as policy:
        lease = policy.lease(
            "research.model.predictor", max(1, size * workspace_factor)
        )
        value = None
        owned = None
        try:
            if expected_identity is None:
                expected_identity = verify_semantic_artifact_member(
                    root, relative_path=relative, expected=member
                )
            try:
                with handle_preserving_primary(open_filesystem_file(path)) as handle:
                    with verified_file_handle(
                        path,
                        handle,
                        expected_identity=expected_identity,
                        error="model_artifact_model_invalid",
                    ):
                        value = pickle.load(handle)
                        if handle.read(1):
                            raise ValueError("model_artifact_model_invalid")
            except (
                FileNotFoundError,
                NotADirectoryError,
                pickle.PickleError,
                EOFError,
                AttributeError,
                ImportError,
            ) as error:
                raise ValueError("model_artifact_model_invalid") from error
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


def _window_payload(window: TimeWindow) -> dict[str, str]:
    if not isinstance(window, TimeWindow):
        raise TypeError("model_artifact_window_invalid")
    start = window.start
    end = window.end
    return {
        "start": time_value_isoformat(start, date_only=True),
        "end": time_value_isoformat(end, date_only=True),
    }


@dataclass(frozen=True)
class PersistedCryptoModelArtifact:
    instance_root: Path
    envelope: dict
    member_records: tuple
    models: tuple[dict, ...]
    final_models: tuple[dict, ...]
    member_identities: dict

    def validation_chunks(self, *, fold_id, model_id):
        entry = next(
            entry
            for entry in self.models
            if entry["fold_id"] == fold_id and entry["model_id"] == model_id
        )
        return _crypto_validation_chunks(
            self.instance_root,
            entry,
            member_identities=self.member_identities,
        )

    def prediction_models(self, window, *, access_window):
        from quant_research_platform.research.protocols.crypto_hft import (
            execution_windows,
        )

        plan = self.envelope["semantic_metadata"]["target_contract"]["protocol"][
            "research_plan"
        ]
        execution_windows(plan, window, access_window=access_window)
        if not self.final_models:
            raise ValueError("crypto_final_models_missing")
        return tuple(
            {**entry, "prediction_window": _intraday_window(window)}
            for entry in self.final_models
        )

    def prediction_lineage(self, model_id, window, *, access_window):
        entries = [
            entry
            for entry in self.prediction_models(window, access_window=access_window)
            if entry["model_id"] == model_id
        ]
        return {
            "model_artifact_id": self.envelope["logical_artifact_id"],
            "model_artifact_identity": self.envelope["instance_identity"],
            "model_artifact_schema": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
            "model_id": model_id,
            "asset_ids": sorted(
                {asset for entry in entries for asset in entry["evaluation_asset_ids"]}
            ),
            "fold_models": [
                {
                    key: entry[key]
                    for key in (
                        "fold_id",
                        "prediction_window",
                        "fit_windows",
                        "fit_cutoff",
                        "training_asset_ids",
                        "evaluation_asset_ids",
                    )
                }
                for entry in entries
            ],
        }


def write_crypto_model_artifact(
    logical_root, *, envelope, model_results, trial_layout, verify_source
):
    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("research.model.artifact.metadata", 0) as metadata_lease,
    ):
        operational = {"model_trial_ledgers": []}
        reference_bytes = 0

        admit = partial(_admit_model_metadata, lease=metadata_lease)
        retain = partial(_retain_model_metadata, lease=metadata_lease)
        write_metadata = partial(_write_model_metadata, lease=metadata_lease)

        def write(staging):
            folders = {}
            final_models = []
            validation_owner = ExitStack()
            validation_writer = None

            def add_reference(reference):
                nonlocal reference_bytes
                reference_bytes += _retain_model_metadata(
                    reference,
                    lease=metadata_lease,
                    error="trial_ledger_working_set_exceeds_memory_capacity",
                )
                operational["model_trial_ledgers"].append(dict(reference))

            def write_validation(fold, spec, frame):
                nonlocal validation_writer
                if validation_writer is None:
                    path = (
                        staging
                        / f"models/{fold.fold_id}/validation/{spec.model_id}.parquet"
                    )
                    filesystem_mkdir(path.parent, parents=True, exist_ok=True)
                    handle = validation_owner.enter_context(
                        handle_preserving_primary(open_filesystem_file(path, "xb"))
                    )
                    validation_writer = validation_owner.enter_context(
                        parquet_table_writer(
                            handle,
                            parent=path.parent,
                            compression="zstd",
                            schema=_CRYPTO_VALIDATION_SCHEMA,
                        )
                    )
                with _parquet_frame_reservation(frame, borrowed_input=True):
                    table = None
                    try:
                        table = pa.Table.from_pandas(
                            frame.rename(columns={spec.label_column: "label"}),
                            schema=_CRYPTO_VALIDATION_SCHEMA,
                            preserve_index=False,
                        )
                        validation_writer(table)
                    finally:
                        table = None

            def write_selection(
                fold, spec, metadata, training_assets, evaluation_assets, reference
            ):
                nonlocal validation_writer
                validation_owner.close()
                validation_writer = None
                entry = {
                    "fold_id": fold.fold_id,
                    "model_id": spec.model_id,
                    "training_window": _intraday_window(fold.train),
                    "validation_window": _intraday_window(fold.validation),
                    "training_asset_ids": training_assets,
                    "validation_asset_ids": training_assets,
                    "evaluation_asset_ids": evaluation_assets,
                    "model_metadata": metadata,
                    "trial_semantic_identity": reference["semantic_identity"],
                    "validation_prediction_path": (
                        f"models/{fold.fold_id}/validation/{spec.model_id}.parquet"
                    ),
                }
                retain(entry)
                entry.update(
                    {
                        "training_asset_ids": list(training_assets),
                        "validation_asset_ids": list(training_assets),
                        "evaluation_asset_ids": list(evaluation_assets),
                        "model_metadata": validate_model_metadata_core(metadata),
                    }
                )
                folders.setdefault(fold.fold_id, []).append(entry)
                add_reference(reference)

            try:
                with (
                    handle_preserving_primary(validation_owner),
                    closing_iterator(
                        iter(
                            model_results(
                                on_validation=write_validation,
                                on_selection=write_selection,
                            )
                        )
                    ) as source,
                ):
                    for owned in source:
                        owned.lease.transfer("research.model.crypto.persist")
                        member = owned.value
                        try:
                            fold = member.fold
                            final_root = staging / "models" / "final"
                            filesystem_mkdir(final_root, parents=True, exist_ok=True)
                            _write_model_member(
                                final_root / f"{member.spec.model_id}.pkl",
                                member.result.prediction_model,
                                state_bytes=owned.lease.reserved_bytes,
                            )
                            entry = {
                                "fold_id": fold.fold_id,
                                "model_id": member.spec.model_id,
                                "fit_windows": [
                                    _intraday_window(part)
                                    for part in (fold.train, fold.validation)
                                ],
                                "fit_cutoff": time_value_isoformat(
                                    fold.validation.end, preserve_datetime=True
                                ),
                                "training_asset_ids": member.training_assets,
                                "evaluation_asset_ids": member.evaluation_assets,
                                "model_metadata": member.result.metadata,
                                "refit_trial_semantic_identity": (
                                    member.refit_trial_reference["semantic_identity"]
                                ),
                                "fit_coverage_id": member.fit_coverage_id,
                            }
                            retain(entry)
                            entry.update(
                                {
                                    "training_asset_ids": list(member.training_assets),
                                    "evaluation_asset_ids": list(
                                        member.evaluation_assets
                                    ),
                                    "model_metadata": validate_model_metadata_core(
                                        member.result.metadata
                                    ),
                                }
                            )
                            final_models.append(entry)
                            add_reference(member.refit_trial_reference)
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
                    write_metadata(
                        staging / "models" / fold_id / "metadata.json",
                        {
                            "schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
                            "models": entries,
                        },
                    )
                write_metadata(
                    staging / "model.json",
                    {
                        "schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
                        "folds": list(folders),
                        "final_models": final_models,
                    },
                )
            except BaseException as error:
                clear_exception_frames(error)
                raise
            finally:
                folders.clear()
                final_models.clear()
                entry = entries = member = owned = None
                del entry, entries, member, owned
                metadata_lease.shrink_to(reference_bytes)

        def verify_models(root, members):
            retained = metadata_lease.reserved_bytes
            try:
                admit(
                    sum(
                        2 * json_record_working_set_bytes(member["size"])
                        for member in members
                        if str(member["path"]).endswith(".json")
                    )
                )
                _validate_crypto_model_payload(
                    root,
                    envelope.payload(_admit=admit),
                    members,
                    operational["model_trial_ledgers"],
                    trial_layout,
                )
                verify_source()
            except BaseException as error:
                clear_exception_frames(error)
                raise
            finally:
                metadata_lease.shrink_to(retained)

        try:
            return write_semantic_artifact(
                logical_root,
                envelope,
                write,
                operational_metadata=operational,
                staged_validator=verify_models,
            )
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            operational.clear()


def _intraday_window(window):
    from quant_research_platform.contracts.time import time_value_isoformat

    return {
        "start": time_value_isoformat(window.start, preserve_datetime=True),
        "end": time_value_isoformat(window.end, preserve_datetime=True),
    }


def read_crypto_model_artifact(
    root, *, expected_identity=None, trial_layout=None, expected_member_identities=None
):
    root = Path(root)
    member_identities = {}
    operational = {}
    envelope, members = read_semantic_artifact_record(
        root,
        expected_identity=expected_identity,
        member_identities=member_identities,
        expected_member_identities=expected_member_identities,
        operational_metadata=operational,
    )
    refs = operational.get("model_trial_ledgers")
    return _validate_crypto_model_payload(
        root,
        envelope,
        members,
        refs,
        trial_layout,
        member_identities=member_identities,
    )


def _crypto_model_scope(envelope):
    protocol = envelope["semantic_metadata"]["target_contract"]["protocol"]
    folds, _holdout = _crypto_model_chronology(
        protocol,
        TimeWindow(
            envelope["research_window"]["start"], envelope["research_window"]["end"]
        ),
    )
    model_ids = tuple(
        model["model_id"] for model in protocol["research_configuration"]["models"]
    )
    return protocol, folds, model_ids


def _validate_crypto_model_payload(
    root, envelope, members, refs, trial_layout, *, member_identities=None
):
    from quant_research_platform.research.protocols.crypto_hft import (
        build_hft_prediction_catalog,
        label_horizon_ms,
    )

    if member_identities is None:
        member_identities = {
            member["path"]: verify_semantic_artifact_member(
                root, relative_path=member["path"], expected=member
            )
            for member in members
        }
    if trial_layout is None:
        raise ValueError("model_trial_layout_required")
    contract = envelope["semantic_metadata"]["target_contract"]
    protocol, folds, model_ids = _crypto_model_scope(envelope)
    test = TimeWindow(**protocol["research_plan"]["test"])
    final_fold_id = protocol["research_plan"]["folds"][-1]["fold_id"]
    final_fold = next((fold for fold in folds if fold.fold_id == final_fold_id), None)
    payload = read_bounded_qrp_json(
        root / "model.json", expected_identity=member_identities["model.json"]
    )[1]
    if (
        set(payload) != {"schema_id", "folds", "final_models"}
        or payload["schema_id"] != CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
        or payload["folds"] != [fold.fold_id for fold in folds]
        or type(payload["final_models"]) is not list
        or tuple(entry.get("model_id") for entry in payload["final_models"])
        != (model_ids if final_fold is not None else ())
    ):
        raise ValueError("crypto_model_fold_coverage_invalid")
    expected_references = len(folds) * len(model_ids) + len(payload["final_models"])
    if type(refs) is not list or len(refs) != expected_references:
        raise ValueError("crypto_model_trial_references_invalid")
    references = {ref["semantic_identity"]: ref for ref in refs}
    if len(references) != len(refs):
        raise ValueError("crypto_model_trial_references_invalid")
    registered_specs = {
        value["model_id"]: ModelSpec(
            **{
                **value,
                "input_factor_columns": tuple(value["input_factor_columns"]),
                "input_feature_columns": tuple(value["input_feature_columns"]),
            }
        )
        for value in contract["model_specs"]
    }
    if tuple(registered_specs) != model_ids:
        raise ValueError("crypto_model_members_invalid")

    def validate_parameters(metadata, parameters):
        validate_model_metadata_core(metadata)
        for key in (
            "model_algorithm",
            "random_seed",
            "label_column",
            "input_factor_columns",
            "input_feature_columns",
            "sample_filter_mode",
            "hyperparameters",
            "max_train_rows",
            "training_weighting",
        ):
            if metadata[key] != parameters[key]:
                raise ValueError("crypto_model_parameters_misaligned")
        if metadata["input_columns"] != (
            parameters["input_factor_columns"] + parameters["input_feature_columns"]
        ):
            raise ValueError("crypto_model_parameters_misaligned")

    def validate_fit_bounds(metadata, windows, cutoff):
        first, last = (
            pd.Timestamp(metadata[key]) for key in ("train_start", "train_end")
        )
        if (
            first > last
            or any(
                not any(
                    pd.Timestamp(part.start) <= value <= pd.Timestamp(part.end)
                    for part in windows
                )
                for value in (first, last)
            )
            or last
            + pd.Timedelta(milliseconds=label_horizon_ms(metadata["label_column"]))
            > pd.Timestamp(cutoff)
        ):
            raise ValueError("crypto_model_fit_window_invalid")

    models = []
    selection_references = {}
    required = {"model.json"}
    for fold in folds:
        relative = f"models/{fold.fold_id}/metadata.json"
        required.add(relative)
        record = read_bounded_qrp_json(
            root / relative, expected_identity=member_identities[relative]
        )[1]
        entries = record.get("models")
        if (
            set(record) != {"schema_id", "models"}
            or record["schema_id"] != CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
            or type(entries) is not list
            or tuple(entry.get("model_id") for entry in entries) != model_ids
        ):
            raise ValueError("crypto_model_members_invalid")
        scopes = build_hft_prediction_catalog(
            configuration=protocol["research_configuration"],
            train_end=fold.train.end.isoformat(),
            test_start=fold.validation.start.isoformat(),
            test_end=fold.validation.end.isoformat(),
        )
        for entry in entries:
            prediction = scopes.get(f"{entry['model_id']}_prediction")
            if (
                set(entry)
                != {
                    "fold_id",
                    "model_id",
                    "training_window",
                    "validation_window",
                    "training_asset_ids",
                    "validation_asset_ids",
                    "evaluation_asset_ids",
                    "model_metadata",
                    "trial_semantic_identity",
                    "validation_prediction_path",
                }
                or entry["fold_id"] != fold.fold_id
                or entry["training_window"] != _intraday_window(fold.train)
                or entry["validation_window"] != _intraday_window(fold.validation)
                or tuple(entry["training_asset_ids"]) != prediction.training_asset_ids
                or tuple(entry["validation_asset_ids"]) != prediction.training_asset_ids
                or tuple(entry["evaluation_asset_ids"])
                != prediction.evaluation_asset_ids
            ):
                raise ValueError("crypto_model_fold_binding_invalid")
            ref = references.pop(entry["trial_semantic_identity"], None)
            if ref is None:
                raise ValueError("crypto_model_trial_references_invalid")
            with read_model_trial_ledger_reference(trial_layout, ref) as (
                plan,
                records,
            ):
                expected_spec = replace(
                    registered_specs[entry["model_id"]],
                    random_seed=derive_seed(
                        ROOT_SEED, "crypto_model", fold.fold_id, entry["model_id"]
                    ),
                )
                parameters = model_spec_payload(expected_spec)
                if (
                    plan.stage != "selection"
                    or plan.market_id != "crypto"
                    or plan.research_dataset_id != protocol["scenario_id"]
                    or plan.data_snapshot_id
                    != envelope["upstream_identities"][protocol["scenario_id"]]
                    or plan.input_lineage != (plan.data_snapshot_id,)
                    or plan.candidate_family_id != entry["model_id"]
                    or plan.target_id != envelope["logical_artifact_id"]
                    or plan.folds != (fold,)
                    or plan.test != test
                    or plan.candidate_ids != (entry["model_id"],)
                    or plan.selection_metric != "validation_mse"
                    or plan.selection_direction != "minimize"
                    or plan.research_protocol_identity != canonical_json_hash(protocol)
                    or plan.protocol_payload()["candidate_parameters"]
                    != {entry["model_id"]: parameters}
                    or len(records) != 1
                    or records[0].status != "completed"
                    or records[0].payload()["normalized_parameters"] != parameters
                    or records[0].metrics["n_train"]
                    != entry["model_metadata"]["rows_after_input_label_dropna"]
                ):
                    raise ValueError("crypto_model_trial_binding_invalid")
                validate_parameters(entry["model_metadata"], parameters)
                validate_fit_bounds(
                    entry["model_metadata"], (fold.train,), fold.train.end
                )
                path = entry["validation_prediction_path"]
                if (
                    path
                    != f"models/{fold.fold_id}/validation/{entry['model_id']}.parquet"
                ):
                    raise ValueError("crypto_model_validation_path_invalid")
                required.add(path)
                validation_coverage = ModelSampleCoverage(
                    crypto_validation_coverage_scope(
                        plan, expected_spec, prediction.training_asset_ids
                    )
                )
                with closing_iterator(
                    _crypto_validation_chunks(
                        root, entry, member_identities=member_identities
                    )
                ) as chunks:
                    for frame in chunks:
                        validation_coverage.add(frame)
                if (
                    validation_coverage.row_count
                    != records[0].metrics["validation_rows"]
                    or validation_coverage.semantic_identity
                    != records[0].common_coverage_id
                ):
                    raise ValueError("crypto_model_validation_count_invalid")
            selection_references[(fold.fold_id, entry["model_id"])] = ref
            models.append(entry)

    for entry in payload["final_models"]:
        if (
            set(entry)
            != {
                "fold_id",
                "model_id",
                "fit_windows",
                "fit_cutoff",
                "training_asset_ids",
                "evaluation_asset_ids",
                "model_metadata",
                "refit_trial_semantic_identity",
                "fit_coverage_id",
            }
            or entry["fold_id"] != final_fold.fold_id
            or entry["fit_windows"]
            != [
                _intraday_window(part)
                for part in (final_fold.train, final_fold.validation)
            ]
            or entry["fit_cutoff"]
            != time_value_isoformat(final_fold.validation.end, preserve_datetime=True)
        ):
            raise ValueError("crypto_model_refit_binding_invalid")
        source = next(
            model
            for model in models
            if model["fold_id"] == final_fold.fold_id
            and model["model_id"] == entry["model_id"]
        )
        if any(
            entry[key] != source[key]
            for key in ("training_asset_ids", "evaluation_asset_ids")
        ):
            raise ValueError("crypto_model_refit_binding_invalid")
        selection_ref = selection_references[(final_fold.fold_id, entry["model_id"])]
        refit_ref = references.pop(entry["refit_trial_semantic_identity"], None)
        if refit_ref is None:
            raise ValueError("crypto_model_trial_references_invalid")
        with ExitStack() as ledgers:
            selection_plan, selection_records = ledgers.enter_context(
                read_model_trial_ledger_reference(trial_layout, selection_ref)
            )
            refit_plan, records = ledgers.enter_context(
                read_model_trial_ledger_reference(trial_layout, refit_ref)
            )
            parameters = selection_records[0].payload()["normalized_parameters"]
            metadata = entry["model_metadata"]
            if (
                refit_plan.stage != "refit"
                or refit_plan.selection_ledger_semantic_identities
                != (selection_ref["semantic_identity"],)
                or refit_plan.fit_windows != (final_fold.train, final_fold.validation)
                or refit_plan.fit_cutoff != final_fold.validation.end
                or any(
                    getattr(refit_plan, key) != getattr(selection_plan, key)
                    for key in (
                        "market_id",
                        "research_dataset_id",
                        "data_snapshot_id",
                        "input_lineage",
                        "research_protocol_identity",
                        "target_id",
                        "candidate_family_id",
                        "test",
                        "candidate_ids",
                        "backend_versions",
                        "cost_assumptions",
                        "risk_assumptions",
                    )
                )
                or refit_plan.protocol_payload()["candidate_parameters"]
                != {entry["model_id"]: parameters}
                or refit_plan.random_seed != parameters["random_seed"]
                or len(records) != 1
                or records[0].status != "completed"
                or records[0].payload()["normalized_parameters"] != parameters
                or records[0].random_seed != metadata["random_seed"]
                or records[0].common_coverage_id != entry["fit_coverage_id"]
                or re.fullmatch(r"[0-9a-f]{64}", entry["fit_coverage_id"]) is None
                or dict(records[0].metrics)
                != {"n_train": metadata["rows_after_input_label_dropna"]}
            ):
                raise ValueError("crypto_model_refit_binding_invalid")
            validate_parameters(metadata, parameters)
            validate_fit_bounds(metadata, refit_plan.fit_windows, refit_plan.fit_cutoff)
        required.add(f"models/final/{entry['model_id']}.pkl")

    if references or {member["path"] for member in members} != required:
        raise ValueError("crypto_model_members_invalid")
    return PersistedCryptoModelArtifact(
        root,
        envelope,
        tuple(members),
        tuple(models),
        tuple(payload["final_models"]),
        member_identities,
    )


def _crypto_validation_chunks(root, entry, *, member_identities):
    from quant_research_platform.research.protocols.crypto_hft import (
        label_horizon_ms,
    )

    previous = {}
    assets = entry["validation_asset_ids"]
    start, end = (
        pd.Timestamp(entry["validation_window"][key]) for key in ("start", "end")
    )
    relative = entry["validation_prediction_path"]
    path = root / relative
    identity = member_identities[relative]
    with _opened_parquet(path, expected_identity=identity) as (
        parquet,
        _identity,
        _handle,
    ):
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
                    + pd.Timedelta(
                        milliseconds=label_horizon_ms(
                            entry["model_metadata"]["label_column"]
                        )
                    )
                ).all()
                and frame.asset_id.isin(assets).all()
            ):
                raise ValueError("crypto_model_validation_window_invalid")
            for row in frame.itertuples(index=False):
                if row.asset_id in previous and row.timestamp <= previous[row.asset_id]:
                    raise ValueError("crypto_model_validation_key_invalid")
                previous[row.asset_id] = row.timestamp
            yield frame


@contextmanager
def read_crypto_model_member(artifact, entry, *, resource_policy):
    relative = f"models/final/{entry['model_id']}.pkl"
    with _read_model_member(
        artifact.instance_root,
        artifact.member_records,
        relative,
        expected_identity=artifact.member_identities[relative],
        resource_policy=resource_policy,
        workspace_factor=4,
    ) as owned:
        value = owned.value
        try:
            validate_model_state(value, entry["model_metadata"])
            yield value
        finally:
            value = None

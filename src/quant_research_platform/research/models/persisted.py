"""Persistence boundary for selected model instances."""

from __future__ import annotations

import os
import pickle
import re
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import copyfileobj

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.experiments import label_maturity_mask
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    canonical_json_hash,
)
from quant_research_platform.contracts.models import (
    A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
    ModelSpec,
    _validate_a_share_model_family_ids,
    model_spec_payload,
    validate_a_share_model_selection_metadata,
    validate_model_metadata_core,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import (
    TimeWindow,
    crypto_execution_windows,
    time_value_isoformat,
)
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
from quant_research_platform.research.protocols.a_share import (
    model_selection_windows,
    quarter_id_for_fold,
)
from quant_research_platform.research.randomness import ROOT_SEED, derive_seed
from quant_research_platform.runtime.execution_metrics import record_run_observations
from quant_research_platform.storage.datasets.reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
    verify_trial_ledger_snapshot,
)
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    open_filesystem_file,
    verified_file_handle,
)
from quant_research_platform.storage.manifests import (
    read_qrp_json_file,
    write_internal_artifact_manifest,
)
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    dataset_file_identity,
    parquet_table_writer,
)
from quant_research_platform.storage.semantic_artifacts import (
    _member_record,
    discover_semantic_artifact_instances,
    semantic_artifact_read_session,
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
_A_SHARE_MODEL_QUARTER_SCHEMA_ID = "qrp-a-share-model-quarter"


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
    operational_metadata: dict
    directory_identities: dict

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
                        and (item["model_id"] == family["selected_config_id"])
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
            development_end=self.envelope["research_window"]["end"],
        )


def _persisted_model_metadata(
    result, *, fit_cutoff=None, fit_coverage_id=None, rank_calibration=False
):
    cutoff = (
        None if fit_cutoff is None else time_value_isoformat(fit_cutoff, date_only=True)
    )
    metadata = dict(result.metadata)
    if fit_cutoff is not None:
        metadata["fit_cutoff"] = cutoff
        metadata["fit_coverage_id"] = fit_coverage_id
    metadata = validate_model_metadata_core(
        metadata, quarterly=fit_cutoff is not None, rank_calibration=rank_calibration
    )
    validate_model_state(result.prediction_model, metadata)
    return metadata


def _model_quarter_envelope(envelope, quarter):
    return build_business_manifest_envelope(
        logical_artifact_id=envelope.logical_artifact_id,
        schema_identity=canonical_json_hash(
            {
                "schema_id": _A_SHARE_MODEL_QUARTER_SCHEMA_ID,
                "model_schema_identity": envelope.schema_identity,
            }
        ),
        protocol_or_spec_identity=envelope.protocol_or_spec_identity,
        software_identity=envelope.software_identity,
        research_window=envelope.research_window,
        upstream_identities=envelope.upstream_identities,
        semantic_metadata={
            **envelope.payload()["semantic_metadata"],
            "model_instance_identity": envelope.instance_identity,
            "quarter_id": quarter,
        },
    )


def _copy_model_members(source, destination, members, member_identities):
    for member in members:
        relative = member["path"]
        target = destination / relative
        filesystem_mkdir(target.parent, parents=True, exist_ok=True)
        with closing_iterator(open_filesystem_file(source / relative)) as reader:
            with verified_file_handle(
                source / relative,
                reader,
                expected_identity=member_identities[relative],
                error="semantic_artifact_member_identity_changed",
            ):
                with closing_iterator(open_filesystem_file(target, "xb")) as writer:
                    copyfileobj(reader, writer, length=1024 * 1024)


def _validate_model_quarter(
    root,
    envelope,
    quarter,
    members,
    references,
    trial_layout,
    *,
    verify_science,
    trial_snapshots,
    member_identities=None,
):
    if member_identities is None:
        member_identities = {
            member["path"]: verify_semantic_artifact_member(
                root, relative_path=member["path"], expected=member
            )
            for member in members
        }
    if "quarter.json" not in member_identities:
        raise ValueError("model_quarter_manifest_invalid")
    payload = read_qrp_json_file(
        root / "quarter.json", expected_identity=member_identities["quarter.json"]
    )[1]
    if (
        set(payload)
        != {
            "schema_id",
            "model_instance_identity",
            "quarter_id",
            "candidate_models",
            "selection",
        }
        or payload["schema_id"] != _A_SHARE_MODEL_QUARTER_SCHEMA_ID
        or payload["model_instance_identity"] != envelope.instance_identity
        or payload["quarter_id"] != quarter
        or type(payload["candidate_models"]) is not list
        or type(references) is not list
    ):
        raise ValueError("model_quarter_manifest_invalid")
    required = {"quarter.json"}
    member_records = {member["path"]: member for member in members}
    by_identity, used, _folds, completed = _validate_a_share_candidate_quarters(
        root,
        envelope.payload(),
        payload["candidate_models"],
        (quarter,),
        references,
        trial_layout,
        verify_science=verify_science,
        member_records=member_records,
        member_identities=member_identities,
        required=required,
        trial_snapshots=trial_snapshots,
    )
    if used != set(by_identity) or required != set(member_records):
        raise ValueError("model_artifact_members_invalid")
    return payload, completed[quarter]


def write_quarterly_model_artifact(
    logical_root, *, model_id, envelope, model_results, trial_layout, verify_source
):
    validate_run_id(model_id)
    if model_id != envelope.logical_artifact_id:
        raise ValueError("model_artifact_identity_invalid")
    operational = {
        "model_trial_ledgers": [],
        "model_phase_observations": {"environments": [], "phases": []},
    }

    def record_observation(value):
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
        candidate_models = []
        final_models = []
        selections = {"global": {}, "quarterly": {}}
        candidate_files = {}
        validation_owner = ExitStack()
        validation_writer = None
        current_path = None

        def write_validation(fold, family, spec, frame):
            nonlocal validation_writer, current_path
            path = (
                f"models/{quarter_id_for_fold(fold)}/{family}/{spec.model_id}.parquet"
            )
            if validation_writer is None:
                current_path = path
                target = staging / path
                filesystem_mkdir(target.parent, parents=True, exist_ok=True)
                handle = validation_owner.enter_context(
                    closing_iterator(open_filesystem_file(target, "xb"))
                )
                validation_writer = validation_owner.enter_context(
                    parquet_table_writer(
                        handle,
                        compression="zstd",
                        schema=_A_SHARE_VALIDATION_SCHEMA,
                    )
                )
            if path != current_path:
                raise ValueError("model_validation_candidate_interleaving")

            future = frame["label_available_at"].gt(
                pd.Timestamp(envelope.research_window["end"])
            )
            if future.any():
                frame = frame.copy()
                frame.loc[future, spec.label_column] = float("nan")
                frame.loc[future, "label_available_at"] = pd.NaT
            table = pa.Table.from_pandas(
                frame.rename(columns={spec.label_column: "label"}),
                schema=_A_SHARE_VALIDATION_SCHEMA,
                preserve_index=False,
            )
            validation_writer(table)

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
            entry["model_metadata"] = _persisted_model_metadata(result)
            candidate_models.append(entry)
            current_path = None

        def write_trial(fold, family, reference):
            for entry in candidate_models:
                if (
                    entry["quarter_id"] == quarter_id_for_fold(fold)
                    and entry["family_id"] == family
                ):
                    entry["trial_semantic_identity"] = reference["semantic_identity"]
            operational["model_trial_ledgers"].append(dict(reference))

        def validate_quarter_selection(fold, decision, *, verify_science):
            quarter = quarter_id_for_fold(fold)
            contract = envelope.payload()["semantic_metadata"]["target_contract"]
            protocol = contract["protocol"]
            windows = model_selection_windows(
                ((quarter, fold.train.end, fold.validation.start),),
                chronology=protocol["model_chronology"],
                development_window=protocol["development_window"],
                complete=False,
            )
            if not windows:
                if decision is not None:
                    raise ValueError("model_selection_decisions_invalid")
                return
            group, key, cutoff, window = windows[0]
            _validate_a_share_selection_decision(
                staging,
                candidate_models,
                contract,
                decision,
                group=group,
                key=key,
                decision_cutoff=cutoff,
                window=window,
                member_identities=candidate_files,
                dataset_identity=envelope.upstream_identities[protocol["scenario_id"]],
                verify_science=verify_science,
            )

        def quarter_location(quarter):
            unit = _model_quarter_envelope(envelope, quarter)
            root = (
                Path(logical_root) / "quarters" / envelope.instance_identity / quarter
            )
            return root, unit

        def reuse_quarter(fold):
            quarter = quarter_id_for_fold(fold)
            root, unit = quarter_location(quarter)
            instances = discover_semantic_artifact_instances(
                root, manifest_scope=lambda _envelope: True
            )
            if not instances:
                return None
            expected = root / "instances" / unit.instance_identity
            if instances != (expected,):
                raise ValueError("model_quarter_identity_mismatch")
            member_identities = {}
            unit_operational = {}
            trial_snapshots = {}
            with semantic_artifact_read_session(
                expected,
                expected_identity=unit.instance_identity,
                member_identities=member_identities,
                operational_metadata=unit_operational,
            ) as (_envelope, members):
                references = unit_operational.get("model_trial_ledgers")
                payload, completed = _validate_model_quarter(
                    expected,
                    envelope,
                    quarter,
                    members,
                    references,
                    trial_layout,
                    verify_science=False,
                    member_identities=member_identities,
                    trial_snapshots=trial_snapshots,
                )
                if any(plan.folds != (fold,) for plan, _count in completed.values()):
                    raise ValueError("model_trial_fold_binding_invalid")
                prediction_members = tuple(
                    entry["validation_prediction"]
                    for entry in payload["candidate_models"]
                )
                _copy_model_members(
                    expected, staging, prediction_members, member_identities
                )
                candidate_models.extend(payload["candidate_models"])
                for member in prediction_members:
                    candidate_files[member["path"]] = dataset_file_identity(
                        staging / member["path"]
                    )
                validate_quarter_selection(
                    fold, payload["selection"], verify_science=False
                )
                if payload["selection"] is not None:
                    selections["quarterly"][quarter] = payload["selection"]
                operational["model_trial_ledgers"].extend(references)
                costs = unit_operational["model_phase_observations"]
                for phase in costs["phases"]:
                    observation = phase["observations"]
                    environment = observation["environment"]
                    record_observation(
                        {
                            **phase,
                            "observations": {
                                **observation,
                                "environment": {
                                    **costs["environments"][environment["context"]],
                                    "created_at_utc": environment["created_at_utc"],
                                },
                            },
                        }
                    )
                verify_source()
                for ledger_root, snapshot in trial_snapshots.items():
                    verify_trial_ledger_snapshot(ledger_root, snapshot)
            return completed

        def publish_quarter(fold):
            quarter = quarter_id_for_fold(fold)
            root, unit = quarter_location(quarter)
            candidates = [
                entry for entry in candidate_models if entry["quarter_id"] == quarter
            ]
            trial_ids = {entry["trial_semantic_identity"] for entry in candidates}
            references = [
                reference
                for reference in operational["model_trial_ledgers"]
                if reference["semantic_identity"] in trial_ids
            ]
            decision = selections["quarterly"].get(quarter)
            costs = operational["model_phase_observations"]
            phases = [
                phase for phase in costs["phases"] if phase["quarter_id"] == quarter
            ]
            contexts = tuple(
                dict.fromkeys(
                    phase["observations"]["environment"]["context"] for phase in phases
                )
            )
            unit_operational = {
                "model_trial_ledgers": references,
                "model_phase_observations": {
                    "environments": [
                        costs["environments"][index] for index in contexts
                    ],
                    "phases": [
                        {
                            **phase,
                            "observations": {
                                **phase["observations"],
                                "environment": {
                                    **phase["observations"]["environment"],
                                    "context": contexts.index(
                                        phase["observations"]["environment"]["context"]
                                    ),
                                },
                            },
                        }
                        for phase in phases
                    ],
                },
            }

            def write_quarter(destination):
                _copy_model_members(
                    staging,
                    destination,
                    tuple(entry["validation_prediction"] for entry in candidates),
                    candidate_files,
                )
                write_internal_artifact_manifest(
                    destination / "quarter.json",
                    {
                        "schema_id": _A_SHARE_MODEL_QUARTER_SCHEMA_ID,
                        "model_instance_identity": envelope.instance_identity,
                        "quarter_id": quarter,
                        "candidate_models": candidates,
                        "selection": decision,
                    },
                )

            def validate_quarter(destination, members):
                trial_snapshots = {}
                _validate_model_quarter(
                    destination,
                    envelope,
                    quarter,
                    members,
                    references,
                    trial_layout,
                    verify_science=True,
                    trial_snapshots=trial_snapshots,
                )
                validate_quarter_selection(fold, decision, verify_science=True)
                verify_source()
                for ledger_root, snapshot in trial_snapshots.items():
                    verify_trial_ledger_snapshot(ledger_root, snapshot)

            write_semantic_artifact(
                root,
                unit,
                write_quarter,
                operational_metadata=unit_operational,
                staged_validator=validate_quarter,
            )

        def select_models(group, decision_id, cutoff, scoring_window):
            if decision_id in selections[group]:
                return selections[group][decision_id]
            observation = {}
            with record_run_observations(observation):
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
                )
            record_observation(
                {
                    "phase": "group_scoring",
                    "selection_groups": [group],
                    "family_id": None,
                    "quarter_id": decision_id,
                    "fit_count": 0,
                    "observations": observation,
                }
            )
            selections[group][decision_id] = value
            return value

        with closing_iterator(validation_owner):
            with closing_iterator(
                iter(
                    model_results(
                        on_validation=write_validation,
                        on_candidate=write_candidate,
                        on_trial=write_trial,
                        select_models=select_models,
                        on_observation=record_observation,
                        reuse_quarter=reuse_quarter,
                        on_quarter=publish_quarter,
                    )
                )
            ) as source:
                for member_value in source:
                    member = member_value
                    path = f"models/final/{member.family_id}/{member.spec.model_id}.pkl"
                    filesystem_mkdir(
                        (staging / path).parent, parents=True, exist_ok=True
                    )
                    metadata = _persisted_model_metadata(
                        member.model_result,
                        fit_cutoff=member.fit_cutoff,
                        fit_coverage_id=member.fit_coverage_id,
                    )
                    _write_model_member(
                        staging / path, member.model_result.prediction_model
                    )
                    entry = {
                        "model_member": path,
                        "family_id": member.family_id,
                        "model_id": member.spec.model_id,
                        "fit_windows": [
                            _window_payload(part) for part in member.fit_windows
                        ],
                        "fit_cutoff": time_value_isoformat(
                            member.fit_cutoff, date_only=True
                        ),
                        "model_metadata": None,
                        "refit_trial_semantic_identity": member.refit_trial_reference[
                            "semantic_identity"
                        ],
                    }
                    entry["model_metadata"] = metadata
                    final_models.extend(
                        dict(entry, selection_group=group)
                        for group in member.selection_groups
                    )
                    operational["model_trial_ledgers"].append(
                        dict(member.refit_trial_reference)
                    )
        quarters = tuple(
            dict.fromkeys(entry["quarter_id"] for entry in candidate_models)
        )
        for quarter in quarters:
            write_internal_artifact_manifest(
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
                write_internal_artifact_manifest(
                    target,
                    {
                        "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                        "selection": decision,
                    },
                )
        family_order = {
            item["model_id"]: index
            for index, item in enumerate(
                envelope.semantic_metadata["target_contract"]["model_specs"]
            )
        }
        final_models.sort(key=lambda entry: family_order[entry["family_id"]])
        final_groups = tuple(
            dict.fromkeys(entry["selection_group"] for entry in final_models)
        )
        for group in final_groups:
            write_internal_artifact_manifest(
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
        write_internal_artifact_manifest(
            staging / "model.json",
            {
                "schema_id": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
                "model_id": model_id,
                "validation_quarters": list(quarters),
                "selection_decisions": {
                    group: list(decisions) for group, decisions in selections.items()
                },
                "final_groups": list(final_groups),
            },
        )

    def verify_models(root, members):
        _validate_a_share_model_payload(
            root,
            envelope.payload(),
            members,
            operational["model_trial_ledgers"],
            trial_layout,
            verify_science=True,
            operational_metadata=operational,
            directory_identities={},
        )
        verify_source()

    return write_semantic_artifact(
        logical_root,
        envelope,
        write,
        operational_metadata=operational,
        staged_validator=verify_models,
    )


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
        root / path, expected_identity=identity
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
                or (not np.isfinite(frame["prediction_score"].to_numpy()).all())
                or (not np.isfinite(frame.loc[labeled, "label"].to_numpy()).all())
                or (not labeled.equals(frame["label_available_at"].notna()))
                or (not frame.timestamp.between(start, end).all())
                or (
                    not (
                        frame.loc[labeled, "label_available_at"]
                        > frame.loc[labeled, "timestamp"]
                    ).all()
                )
                or (
                    not (
                        frame.loc[labeled, "label_available_at"]
                        <= pd.Timestamp(development_end)
                    ).all()
                )
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
            try:
                yield frame
            except GeneratorExit:
                return


def _eligible_a_share_predictions(candidate_models, cutoff, scoring_window):
    eligible = [
        entry
        for entry in candidate_models
        if pd.Timestamp(entry["validation_window"]["end"]) <= pd.Timestamp(cutoff)
        and pd.Timestamp(entry["validation_window"]["end"])
        >= pd.Timestamp(scoring_window.start)
        and (
            pd.Timestamp(entry["validation_window"]["start"])
            <= pd.Timestamp(scoring_window.end)
        )
    ]
    if not eligible:
        raise ValueError("a_share_selection_predictions_missing")
    return eligible


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
):
    protocol = contract["protocol"]
    family_ids = tuple(value["model_id"] for value in contract["model_specs"])
    eligible = _eligible_a_share_predictions(candidate_models, cutoff, scoring_window)
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
                            try:
                                yield mature
                            except GeneratorExit:
                                return

            metrics = evaluate_model_trial_metrics_from_chunks(
                chunks(),
                model_id=candidate_id,
                label_column="label",
                prediction_column="prediction_score",
                expected_coverage=None if first_candidate else coverage,
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
    trial_snapshots=None,
):
    instance_root = Path(instance_root)
    member_identities = {}
    operational = {}
    directories = {}
    with semantic_artifact_read_session(
        instance_root,
        expected_identity=expected_identity,
        member_identities=member_identities,
        expected_member_identities=expected_member_identities,
        operational_metadata=operational,
        directory_identities=directories,
    ) as (envelope, members):
        return _validate_a_share_model_payload(
            instance_root,
            envelope,
            members,
            operational.get("model_trial_ledgers"),
            trial_layout,
            verify_science=False,
            member_identities=member_identities,
            operational_metadata=operational,
            directory_identities=directories,
            trial_snapshots=trial_snapshots,
        )


def _validate_a_share_model_payload(
    root,
    envelope,
    members,
    references,
    trial_layout,
    *,
    verify_science,
    operational_metadata,
    directory_identities,
    member_identities=None,
    trial_snapshots=None,
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
    payload = read_qrp_json_file(
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
        or (type(payload["validation_quarters"]) is not list)
        or (type(payload["selection_decisions"]) is not dict)
        or (set(payload["selection_decisions"]) != {"global", "quarterly"})
        or (type(payload["final_groups"]) is not list)
        or (type(references) is not list)
    ):
        raise ValueError("model_artifact_manifest_invalid")
    contract = envelope["semantic_metadata"]["target_contract"]
    protocol = contract["protocol"]
    protocol_identity = canonical_json_hash(protocol)
    dataset_identity = envelope["upstream_identities"][protocol["scenario_id"]]
    specs, configurations, candidate_orders = _a_share_model_configuration(contract)
    family_ids = tuple(specs)
    quarters = tuple(
        str(period)
        for period in pd.period_range(
            protocol["model_chronology"]["warmup_window"]["start"],
            envelope["research_window"]["end"],
            freq="Q",
        )
    )
    if payload["validation_quarters"] != list(quarters):
        raise ValueError("model_candidate_matrix_incomplete")
    required = {"model.json"}

    def read_member(path):
        if path not in member_identities:
            raise ValueError("model_artifact_members_invalid")
        required.add(path)
        return read_qrp_json_file(
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
            if (
                set(shard) != {"schema_id", "selection"}
                or shard["schema_id"] != A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID
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
    by_identity, used_references, folds, _completed = (
        _validate_a_share_candidate_quarters(
            root,
            envelope,
            candidates,
            quarters,
            references,
            trial_layout,
            verify_science=verify_science,
            member_records=member_records,
            member_identities=member_identities,
            required=required,
            trial_snapshots=trial_snapshots,
        )
    )
    cutoff = pd.Timestamp(protocol["development_window"]["end"])
    complete = pd.Timestamp(envelope["research_window"]["end"]) >= cutoff
    expected_decisions = model_selection_windows(
        (
            (
                quarter,
                entry["training_window"]["end"],
                entry["validation_window"]["start"],
            )
            for quarter, entry in folds.items()
        ),
        chronology=protocol["model_chronology"],
        development_window=protocol["development_window"],
        complete=complete,
    )
    if any(type(value) is not dict for value in selections.values()) or {
        (group, key) for group, values in selections.items() for key in values
    } != {(group, key) for group, key, _cutoff, _window in expected_decisions}:
        raise ValueError("model_selection_decisions_incomplete")
    for group, key, decision_cutoff, window in expected_decisions:
        _validate_a_share_selection_decision(
            root,
            candidates,
            contract,
            selections[group][key],
            group=group,
            key=key,
            decision_cutoff=decision_cutoff,
            window=window,
            member_identities=member_identities,
            dataset_identity=dataset_identity,
            verify_science=verify_science,
        )
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
    final_references = {}
    validated_models = set()
    for entry in final_models:
        if set(entry) != {
            "selection_group",
            "family_id",
            "model_id",
            "fit_windows",
            "fit_cutoff",
            "model_metadata",
            "refit_trial_semantic_identity",
            "model_member",
        }:
            raise ValueError("model_final_metadata_invalid")
        group, family_id = (entry["selection_group"], entry["family_id"])
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
        groups = tuple(
            selected_group
            for selected_group in ("global", "quarterly")
            if any(
                item["family_id"] == family_id
                and item["selected_config_id"] == family["selected_config_id"]
                for item in selections[selected_group]["final"]["families"]
            )
        )
        parents = tuple(
            sorted(
                {
                    parent
                    for selected_group in groups
                    for parent in selections[selected_group]["final"][
                        "selection_ledger_semantic_identities"
                    ]
                }
            )
        )
        normalized = model_spec_payload(expected_spec)
        identity = entry["refit_trial_semantic_identity"]
        key = (family_id, expected_spec.model_id)
        if final_references.setdefault(key, identity) != identity:
            raise ValueError("model_trial_refit_misaligned")
        if identity not in by_identity:
            raise ValueError("model_trial_references_missing")
        used_references.add(identity)
        plan, records = read_model_trial_ledger_reference(
            trial_layout, by_identity[identity], snapshots=trial_snapshots
        )
        if (
            plan.stage != "refit"
            or plan.selection_groups != groups
            or plan.selection_ledger_semantic_identities != parents
            or (plan.market_id != "a_share")
            or (plan.research_dataset_id != protocol["scenario_id"])
            or (plan.data_snapshot_id != dataset_identity)
            or (plan.input_lineage != (dataset_identity,))
            or (plan.target_id != envelope["logical_artifact_id"])
            or (plan.candidate_family_id != family_id)
            or (plan.research_protocol_identity != protocol_identity)
            or (plan.fit_windows != fit_windows)
            or (pd.Timestamp(plan.fit_cutoff) != cutoff)
            or (plan.candidate_ids != (expected_spec.model_id,))
            or (plan.random_seed != expected_spec.random_seed)
            or (
                canonical_identity_bytes(
                    plan.protocol_payload()["candidate_parameters"]
                )
                != canonical_identity_bytes({expected_spec.model_id: normalized})
            )
            or (_window_payload(plan.test) != protocol["holdout_window"])
            or (len(records) != 1)
            or (records[0].status != "completed")
            or (
                canonical_identity_bytes(records[0].payload()["normalized_parameters"])
                != canonical_identity_bytes(normalized)
            )
            or (records[0].common_coverage_id != metadata["fit_coverage_id"])
            or (
                dict(records[0].metrics)
                != {"n_train": metadata["rows_after_input_label_dropna"]}
            )
            or (entry["model_id"] != expected_spec.model_id)
            or (entry["fit_windows"] != [_window_payload(part) for part in fit_windows])
            or (entry["fit_cutoff"] != cutoff.date().isoformat())
            or (metadata["fit_cutoff"] != entry["fit_cutoff"])
            or (
                pd.Timestamp(metadata["train_start"])
                < pd.Timestamp(fit_windows[0].start)
            )
            or (pd.Timestamp(metadata["train_end"]) > cutoff)
        ):
            raise ValueError("model_trial_refit_misaligned")
        _validate_model_parameters(
            metadata, normalized, error="model_trial_parameters_misaligned"
        )
        relative = f"models/final/{family_id}/{expected_spec.model_id}.pkl"
        if entry["model_member"] != relative:
            raise ValueError("model_final_metadata_invalid")
        required.add(relative)
        if verify_science and relative not in validated_models:
            member_value = _read_model_member(
                root, members, relative, expected_identity=member_identities[relative]
            )
            validate_model_state(member_value, metadata)
            validated_models.add(relative)
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
        operational_metadata,
        directory_identities,
    )


def _validate_a_share_selection_decision(
    root,
    candidates,
    contract,
    decision,
    *,
    group,
    key,
    decision_cutoff,
    window,
    member_identities,
    dataset_identity,
    verify_science,
):
    specs, _configurations, candidate_orders = _a_share_model_configuration(contract)
    family_ids = tuple(specs)
    validate_a_share_model_selection_metadata(decision)
    eligible = _eligible_a_share_predictions(candidates, decision_cutoff, window)
    if (
        decision["selection_group"] != group
        or decision["decision_id"] != key
        or decision["cutoff"]
        != time_value_isoformat(pd.Timestamp(decision_cutoff), date_only=True)
        or decision["validation_start"]
        != time_value_isoformat(window.start, date_only=True)
        or decision["validation_end"]
        != time_value_isoformat(window.end, date_only=True)
        or decision["candidate_family_ids"] != list(family_ids)
        or decision["label_dataset_identity"] != dataset_identity
        or decision["prediction_members"]
        != sorted(
            (entry["validation_prediction"] for entry in eligible),
            key=lambda item: item["path"],
        )
        or decision["selection_ledger_semantic_identities"]
        != sorted({entry["trial_semantic_identity"] for entry in eligible})
        or any(
            family["config_tie_break_order"]
            != list(candidate_orders[family["family_id"]])
            for family in decision["families"]
        )
    ):
        raise ValueError("model_selection_decision_mismatch")
    if verify_science:
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
        if decision != expected:
            raise ValueError("model_selection_decision_mismatch")


def _a_share_model_configuration(contract):
    protocol = contract["protocol"]
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
    _validate_a_share_model_family_ids(tuple(specs))
    candidate_orders = {}
    for family_id, configuration in configurations.items():
        parameters = configuration.get(
            "parameter_candidates", [configuration["hyperparameters"]]
        )
        candidate_orders[family_id] = tuple(
            family_id if len(parameters) == 1 else f"{family_id}__config_{index}"
            for index in range(len(parameters))
        )
    return specs, configurations, candidate_orders


def _validate_a_share_candidate_quarters(
    root,
    envelope,
    candidates,
    quarters,
    references,
    trial_layout,
    *,
    verify_science,
    member_records,
    member_identities,
    required,
    trial_snapshots=None,
):
    contract = envelope["semantic_metadata"]["target_contract"]
    protocol = contract["protocol"]
    protocol_identity = canonical_json_hash(protocol)
    dataset_identity = envelope["upstream_identities"][protocol["scenario_id"]]
    specs, configurations, candidate_orders = _a_share_model_configuration(contract)
    family_ids = tuple(specs)
    expected_keys = tuple(
        (quarter, family_id, candidate_id)
        for quarter in quarters
        for family_id in family_ids
        for candidate_id in candidate_orders[family_id]
    )
    if (
        tuple(
            (entry.get("quarter_id"), entry.get("family_id"), entry.get("model_id"))
            for entry in candidates
        )
        != expected_keys
    ):
        raise ValueError("model_candidate_matrix_incomplete")
    by_identity = {
        reference["semantic_identity"]: reference for reference in references
    }
    if len(by_identity) != len(references):
        raise ValueError("model_trial_references_duplicate")
    used_references = set()
    folds = {}
    completed = {}
    for quarter in quarters:
        completed[quarter] = {}
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
            plan, records = read_model_trial_ledger_reference(
                trial_layout, by_identity[identity], snapshots=trial_snapshots
            )
            if (
                plan.stage != "selection"
                or plan.market_id != "a_share"
                or plan.research_dataset_id != protocol["scenario_id"]
                or (plan.data_snapshot_id != dataset_identity)
                or (plan.input_lineage != (dataset_identity,))
                or (plan.target_id != envelope["logical_artifact_id"])
                or (plan.research_protocol_identity != protocol_identity)
                or (plan.candidate_family_id != family_id)
                or (len(plan.folds) != 1)
                or (plan.candidate_ids != candidate_orders[family_id])
                or (plan.tie_break_order != plan.candidate_ids)
                or (plan.trial_budget != len(plan.candidate_ids))
                or (plan.selection_metric != "validation_rankic")
                or (plan.selection_direction != "maximize")
                or (_window_payload(plan.test) != protocol["holdout_window"])
                or (
                    tuple(record.candidate_id for record in records)
                    != plan.candidate_ids
                )
                or any(record.status != "completed" for record in records)
            ):
                raise ValueError("model_trial_artifact_binding_invalid")
            fold = plan.folds[0]
            if (
                quarter_id_for_fold(fold) != quarter
                or pd.Timestamp(fold.validation.end)
                > pd.Timestamp(envelope["research_window"]["end"])
                or str(pd.Period(fold.validation.start, freq="Q")) != quarter
                or str(pd.Period(fold.validation.end, freq="Q")) != quarter
                or (
                    pd.Timestamp(fold.train.start).year
                    != pd.Timestamp(protocol["formal_dataset_window"]["start"]).year
                )
                or (
                    quarter in folds
                    and (
                        folds[quarter]["training_window"] != _window_payload(fold.train)
                        or folds[quarter]["validation_window"]
                        != _window_payload(fold.validation)
                    )
                )
            ):
                raise ValueError("model_trial_fold_binding_invalid")
            folds[quarter] = entries[0]
            completed[quarter][family_id] = (plan, records[0].metrics["n_train"])
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
                    canonical_identity_bytes(record.payload()["normalized_parameters"])
                    != canonical_identity_bytes(expected_parameters)
                    or canonical_identity_bytes(
                        plan.protocol_payload()["candidate_parameters"][
                            record.candidate_id
                        ]
                    )
                    != canonical_identity_bytes(expected_parameters)
                    or record.metrics["n_train"]
                    != metadata["rows_after_input_label_dropna"]
                    or (entry["training_window"] != _window_payload(fold.train))
                    or (entry["validation_window"] != _window_payload(fold.validation))
                    or (
                        pd.Timestamp(metadata["train_start"])
                        < pd.Timestamp(fold.train.start)
                    )
                    or (
                        pd.Timestamp(metadata["train_end"])
                        > pd.Timestamp(fold.train.end)
                    )
                ):
                    raise ValueError("model_trial_parameters_misaligned")
                _validate_model_parameters(
                    metadata,
                    expected_parameters,
                    error="model_trial_parameters_misaligned",
                )
                member = entry["validation_prediction"]
                path = f"models/{quarter}/{family_id}/{entry['model_id']}.parquet"
                if member.get("path") != path or member != member_records.get(path):
                    raise ValueError("model_validation_member_invalid")
                required.add(path)
                if verify_science:
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
                                development_end=envelope["research_window"]["end"],
                            )
                        ) as frames:
                            for frame in frames:
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
                                try:
                                    yield mature
                                except GeneratorExit:
                                    return

                    metrics = evaluate_model_trial_metrics_from_chunks(
                        scoring_frames(),
                        model_id=entry["model_id"],
                        label_column="label",
                        prediction_column="prediction_score",
                        expected_coverage=None,
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
    return by_identity, used_references, folds, completed


def _validate_model_parameters(metadata, parameters, *, error):
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
        if canonical_identity_bytes(metadata[key]) != canonical_identity_bytes(
            parameters[key]
        ):
            raise ValueError(error)


def read_quarterly_predictor(artifact, selection_group, family_id):
    entry = next(
        item
        for item in artifact.final_models
        if item["selection_group"] == selection_group and item["family_id"] == family_id
    )
    relative = entry["model_member"]
    member_value = _read_model_member(
        artifact.instance_root,
        artifact.member_records,
        relative,
        expected_identity=artifact.member_identities[relative],
    )
    predictor = member_value
    validate_model_state(predictor, entry["model_metadata"])
    return predictor


def _write_model_member(path, value):
    with closing_iterator(open_filesystem_file(path, "wb")) as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


def _read_model_member(root, members, relative, *, expected_identity=None):
    member = next((item for item in members if item["path"] == relative), None)
    if member is None:
        raise ValueError("model_artifact_model_invalid")
    path = root / relative
    if expected_identity is None:
        expected_identity = verify_semantic_artifact_member(
            root, relative_path=relative, expected=member
        )
    try:
        with closing_iterator(open_filesystem_file(path)) as handle:
            with verified_file_handle(
                path,
                handle,
                expected_identity=expected_identity,
                error="model_artifact_model_invalid",
            ):
                value = pickle.load(handle)
                if handle.read(1):
                    raise ValueError("model_artifact_model_invalid")
                return value
    except (
        FileNotFoundError,
        NotADirectoryError,
        pickle.PickleError,
        EOFError,
        AttributeError,
        ImportError,
    ) as error:
        raise ValueError("model_artifact_model_invalid") from error


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
    trial_ledgers: tuple

    def validation_chunks(self, *, fold_id, model_id):
        entry = next(
            entry
            for entry in self.models
            if entry["fold_id"] == fold_id and entry["model_id"] == model_id
        )
        return _crypto_validation_chunks(
            self.instance_root, entry, member_identities=self.member_identities
        )

    def prediction_models(self, window, *, access_window):
        plan = self.envelope["semantic_metadata"]["target_contract"]["protocol"][
            "research_plan"
        ]
        crypto_execution_windows(plan, window, access_window=access_window)
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
            "score_to_label": entries[0]["model_metadata"]["rank_parameters"][
                "label_calibration"
            ]
            if entries[0]["model_metadata"]["model_algorithm"]
            in {"equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"}
            else None,
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
    operational = {"model_trial_ledgers": []}

    def write(staging):
        folders = {}
        final_models = []
        validation_owner = ExitStack()
        validation_writer = None

        def add_reference(reference):
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
                    closing_iterator(open_filesystem_file(path, "xb"))
                )
                validation_writer = validation_owner.enter_context(
                    parquet_table_writer(
                        handle,
                        compression="zstd",
                        schema=_CRYPTO_VALIDATION_SCHEMA,
                    )
                )

            table = pa.Table.from_pandas(
                frame.rename(columns={spec.label_column: "label"}),
                schema=_CRYPTO_VALIDATION_SCHEMA,
                preserve_index=False,
            )
            validation_writer(table)

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
            entry.update(
                {
                    "training_asset_ids": list(training_assets),
                    "validation_asset_ids": list(training_assets),
                    "evaluation_asset_ids": list(evaluation_assets),
                    "model_metadata": validate_model_metadata_core(
                        metadata, rank_calibration=True
                    ),
                }
            )
            folders.setdefault(fold.fold_id, []).append(entry)
            add_reference(reference)

        with closing_iterator(validation_owner):
            with closing_iterator(
                iter(
                    model_results(
                        on_validation=write_validation, on_selection=write_selection
                    )
                )
            ) as source:
                for member_value in source:
                    member = member_value
                    fold = member.fold
                    final_root = staging / "models" / "final"
                    filesystem_mkdir(final_root, parents=True, exist_ok=True)
                    _write_model_member(
                        final_root / f"{member.spec.model_id}.pkl",
                        member.result.prediction_model,
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
                        "model_metadata": None,
                        "refit_trial_semantic_identity": member.refit_trial_reference[
                            "semantic_identity"
                        ],
                        "fit_coverage_id": member.fit_coverage_id,
                    }
                    entry.update(
                        {
                            "training_asset_ids": list(member.training_assets),
                            "evaluation_asset_ids": list(member.evaluation_assets),
                            "model_metadata": _persisted_model_metadata(
                                member.result, rank_calibration=True
                            ),
                        }
                    )
                    final_models.append(entry)
                    add_reference(member.refit_trial_reference)
        model_order = {
            spec["model_id"]: position
            for position, spec in enumerate(
                envelope.semantic_metadata["target_contract"]["model_specs"]
            )
        }
        final_models.sort(key=lambda entry: model_order[entry["model_id"]])
        for fold_id, entries in folders.items():
            entries.sort(key=lambda entry: model_order[entry["model_id"]])
            write_internal_artifact_manifest(
                staging / "models" / fold_id / "metadata.json",
                {"schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID, "models": entries},
            )
        write_internal_artifact_manifest(
            staging / "model.json",
            {
                "schema_id": CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID,
                "folds": list(folders),
                "final_models": final_models,
            },
        )

    def verify_models(root, members):
        _validate_crypto_model_payload(
            root,
            envelope.payload(),
            members,
            operational["model_trial_ledgers"],
            trial_layout,
        )
        verify_source()

    return write_semantic_artifact(
        logical_root,
        envelope,
        write,
        operational_metadata=operational,
        staged_validator=verify_models,
    )


def _intraday_window(window):
    from quant_research_platform.contracts.time import time_value_isoformat

    return {
        "start": time_value_isoformat(window.start, preserve_datetime=True),
        "end": time_value_isoformat(window.end, preserve_datetime=True),
    }


def read_crypto_model_artifact(
    root,
    *,
    expected_identity=None,
    trial_layout=None,
    expected_member_identities=None,
    trial_snapshots=None,
):
    root = Path(root)
    member_identities = {}
    operational = {}
    with semantic_artifact_read_session(
        root,
        expected_identity=expected_identity,
        member_identities=member_identities,
        expected_member_identities=expected_member_identities,
        operational_metadata=operational,
    ) as (envelope, members):
        refs = operational.get("model_trial_ledgers")
        return _validate_crypto_model_payload(
            root,
            envelope,
            members,
            refs,
            trial_layout,
            member_identities=member_identities,
            trial_snapshots=trial_snapshots,
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
    return (protocol, folds, model_ids)


def _validate_crypto_model_payload(
    root,
    envelope,
    members,
    refs,
    trial_layout,
    *,
    member_identities=None,
    trial_snapshots=None,
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
    payload = read_qrp_json_file(
        root / "model.json", expected_identity=member_identities["model.json"]
    )[1]
    if (
        set(payload) != {"schema_id", "folds", "final_models"}
        or payload["schema_id"] != CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
        or payload["folds"] != [fold.fold_id for fold in folds]
        or (type(payload["final_models"]) is not list)
        or (
            tuple(entry.get("model_id") for entry in payload["final_models"])
            != (model_ids if final_fold is not None else ())
        )
    ):
        raise ValueError("crypto_model_fold_coverage_invalid")
    expected_references = len(folds) * len(model_ids) + len(payload["final_models"])
    if type(refs) is not list or len(refs) != expected_references:
        raise ValueError("crypto_model_trial_references_invalid")
    references = {ref["semantic_identity"]: ref for ref in refs}
    trial_ledgers = []
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
        record = read_qrp_json_file(
            root / relative, expected_identity=member_identities[relative]
        )[1]
        entries = record.get("models")
        if (
            set(record) != {"schema_id", "models"}
            or record["schema_id"] != CRYPTO_FOLD_MODEL_ARTIFACT_SCHEMA_ID
            or type(entries) is not list
            or (tuple(entry.get("model_id") for entry in entries) != model_ids)
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
                or (entry["validation_window"] != _intraday_window(fold.validation))
                or (tuple(entry["training_asset_ids"]) != prediction.training_asset_ids)
                or (
                    tuple(entry["validation_asset_ids"])
                    != prediction.training_asset_ids
                )
                or (
                    tuple(entry["evaluation_asset_ids"])
                    != prediction.evaluation_asset_ids
                )
            ):
                raise ValueError("crypto_model_fold_binding_invalid")
            ref = references.pop(entry["trial_semantic_identity"], None)
            if ref is None:
                raise ValueError("crypto_model_trial_references_invalid")
            plan, records = read_model_trial_ledger_reference(
                trial_layout, ref, snapshots=trial_snapshots
            )
            trial_ledgers.append((ref, plan, records))
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
                or (
                    plan.data_snapshot_id
                    != envelope["upstream_identities"][protocol["scenario_id"]]
                )
                or (plan.input_lineage != (plan.data_snapshot_id,))
                or (plan.candidate_family_id != entry["model_id"])
                or (plan.target_id != envelope["logical_artifact_id"])
                or (plan.folds != (fold,))
                or (plan.test != test)
                or (plan.candidate_ids != (entry["model_id"],))
                or (plan.selection_metric != "validation_mse")
                or (plan.selection_direction != "minimize")
                or (plan.research_protocol_identity != canonical_json_hash(protocol))
                or (
                    canonical_identity_bytes(
                        plan.protocol_payload()["candidate_parameters"]
                    )
                    != canonical_identity_bytes({entry["model_id"]: parameters})
                )
                or (len(records) != 1)
                or (records[0].status != "completed")
                or (
                    canonical_identity_bytes(
                        records[0].payload()["normalized_parameters"]
                    )
                    != canonical_identity_bytes(parameters)
                )
                or (
                    records[0].metrics["n_train"]
                    != entry["model_metadata"]["rows_after_input_label_dropna"]
                )
            ):
                raise ValueError("crypto_model_trial_binding_invalid")
            validate_model_metadata_core(entry["model_metadata"], rank_calibration=True)
            _validate_model_parameters(
                entry["model_metadata"],
                parameters,
                error="crypto_model_parameters_misaligned",
            )
            validate_fit_bounds(entry["model_metadata"], (fold.train,), fold.train.end)
            path = entry["validation_prediction_path"]
            if path != f"models/{fold.fold_id}/validation/{entry['model_id']}.parquet":
                raise ValueError("crypto_model_validation_path_invalid")
            required.add(path)
            validation_coverage = ModelSampleCoverage(
                crypto_validation_coverage_scope(
                    plan, expected_spec, prediction.training_asset_ids
                )
            )
            time_series = (
                validation_coverage.scope["validation_axis"]
                == "single_asset_time_series"
            )
            rankic = records[0].metrics["validation_rankic"]
            rankic_reason = records[0].metrics["validation_rankic_reason"]
            if (
                time_series
                and (rankic is not None or rankic_reason != "not_applicable")
            ) or (not time_series and rankic_reason == "not_applicable"):
                raise ValueError("crypto_model_validation_axis_invalid")
            with closing_iterator(
                _crypto_validation_chunks(
                    root, entry, member_identities=member_identities
                )
            ) as chunks:
                for frame in chunks:
                    validation_coverage.add(frame)
            if (
                validation_coverage.row_count != records[0].metrics["validation_rows"]
                or validation_coverage.semantic_identity
                != records[0].common_coverage_id
            ):
                raise ValueError("crypto_model_validation_count_invalid")
            selection_references[fold.fold_id, entry["model_id"]] = ref
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
            or (
                entry["fit_cutoff"]
                != time_value_isoformat(
                    final_fold.validation.end, preserve_datetime=True
                )
            )
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
        selection_ref = selection_references[final_fold.fold_id, entry["model_id"]]
        refit_ref = references.pop(entry["refit_trial_semantic_identity"], None)
        if refit_ref is None:
            raise ValueError("crypto_model_trial_references_invalid")
        _reference, selection_plan, selection_records = next(
            item
            for item in trial_ledgers
            if item[0]["semantic_identity"] == selection_ref["semantic_identity"]
        )
        refit_plan, records = read_model_trial_ledger_reference(
            trial_layout, refit_ref, snapshots=trial_snapshots
        )
        trial_ledgers.append((refit_ref, refit_plan, records))
        parameters = selection_records[0].payload()["normalized_parameters"]
        metadata = entry["model_metadata"]
        refit_protocol, selection_protocol = (
            refit_plan.protocol_payload(),
            selection_plan.protocol_payload(),
        )
        if (
            refit_plan.stage != "refit"
            or refit_plan.selection_ledger_semantic_identities
            != (selection_ref["semantic_identity"],)
            or refit_plan.fit_windows != (final_fold.train, final_fold.validation)
            or (refit_plan.fit_cutoff != final_fold.validation.end)
            or any(
                canonical_identity_bytes(refit_protocol[key])
                != canonical_identity_bytes(selection_protocol[key])
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
            or (
                canonical_identity_bytes(
                    refit_plan.protocol_payload()["candidate_parameters"]
                )
                != canonical_identity_bytes({entry["model_id"]: parameters})
            )
            or (refit_plan.random_seed != parameters["random_seed"])
            or (len(records) != 1)
            or (records[0].status != "completed")
            or (
                canonical_identity_bytes(records[0].payload()["normalized_parameters"])
                != canonical_identity_bytes(parameters)
            )
            or (records[0].random_seed != metadata["random_seed"])
            or (records[0].common_coverage_id != entry["fit_coverage_id"])
            or (re.fullmatch("[0-9a-f]{64}", entry["fit_coverage_id"]) is None)
            or (
                dict(records[0].metrics)
                != {"n_train": metadata["rows_after_input_label_dropna"]}
            )
        ):
            raise ValueError("crypto_model_refit_binding_invalid")
        validate_model_metadata_core(metadata, rank_calibration=True)
        _validate_model_parameters(
            metadata, parameters, error="crypto_model_parameters_misaligned"
        )
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
        tuple(trial_ledgers),
    )


def _crypto_validation_chunks(root, entry, *, member_identities):
    from quant_research_platform.research.protocols.crypto_hft import label_horizon_ms

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
    chunks = iter_declared_parquet_member_batches(path, expected_identity=identity)
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
                or (
                    not np.isfinite(
                        frame[["label", "prediction_score"]].to_numpy()
                    ).all()
                )
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
            try:
                yield frame
            except GeneratorExit:
                return


def read_crypto_model_member(artifact, entry):
    relative = f"models/final/{entry['model_id']}.pkl"
    member_value = _read_model_member(
        artifact.instance_root,
        artifact.member_records,
        relative,
        expected_identity=artifact.member_identities[relative],
    )
    value = member_value
    validate_model_state(value, entry["model_metadata"])
    return value

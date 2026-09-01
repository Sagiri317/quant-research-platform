"""Immutable-record experiment ledger backed by atomic no-replace files."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from quant_research_platform.contracts.experiments import ExperimentPlan
from quant_research_platform.contracts.json_codec import (
    bounded_qrp_json_object_snapshot,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)
from quant_research_platform.research.randomness import ROOT_SEED
from quant_research_platform.storage.atomic import atomic_write_json
from quant_research_platform.storage.manifests import read_bounded_qrp_json_directory

TrialStatus = Literal["attempted", "completed", "failed", "abandoned", "invalid"]
_PROTOCOL_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_TRIAL_RECORD_BYTES = 256 * 1024


@dataclass(frozen=True, kw_only=True)
class TrialRecord:
    event_id: str
    attempt_index: int
    experiment_id: str
    protocol_hash: str
    candidate_family_id: str
    candidate_id: str
    trial_id: str
    fold_id: str
    market_id: str
    research_dataset_id: str
    target_id: str
    horizon_id: str
    common_coverage_id: str
    factor_definition_ids: tuple[str, ...] = ()
    model_definition_id: str | None = None
    portfolio_definition_id: str | None = None
    normalized_parameters: Mapping[str, object] = field(default_factory=dict)
    random_seed: int = ROOT_SEED
    backend_versions: Mapping[str, str] = field(default_factory=dict)
    thread_settings: Mapping[str, int] = field(default_factory=dict)
    status: TrialStatus
    validation_metrics: Mapping[str, object] = field(default_factory=dict)
    development_test_metrics: Mapping[str, object] = field(default_factory=dict)
    development_test_contaminated: bool = False
    undefined_reasons: Mapping[str, object] = field(default_factory=dict)
    cost_assumptions: Mapping[str, object] = field(default_factory=dict)
    risk_assumptions: Mapping[str, object] = field(default_factory=dict)
    rebalance_assumption: str = "not_applicable"
    artifact_lineage: tuple[str, ...] = ()
    holdout_accessed: bool = False
    contract_version: str = "1.0"

    def __post_init__(self) -> None:
        for value in (
            self.event_id,
            self.experiment_id,
            self.candidate_family_id,
            self.candidate_id,
            self.trial_id,
            self.fold_id,
        ):
            validate_run_id(value)
        if type(self.attempt_index) is not int or self.attempt_index <= 0:
            raise ValueError("trial_attempt_index_invalid")
        if _PROTOCOL_HASH.fullmatch(self.protocol_hash) is None:
            raise ValueError("trial_protocol_hash_invalid")
        for value in (
            self.market_id,
            self.research_dataset_id,
            self.target_id,
            self.horizon_id,
            self.common_coverage_id,
            self.rebalance_assumption,
        ):
            if type(value) is not str or not value:
                raise ValueError("trial_identifier_invalid")
        for value in (self.factor_definition_ids, self.artifact_lineage):
            if type(value) is not tuple or any(
                type(item) is not str or not item for item in value
            ):
                raise ValueError("trial_lineage_invalid")
        for value in (self.model_definition_id, self.portfolio_definition_id):
            if value is not None and (type(value) is not str or not value):
                raise ValueError("trial_definition_id_invalid")
        if type(self.random_seed) is not int:
            raise ValueError("trial_random_seed_invalid")
        if self.status not in {
            "attempted",
            "completed",
            "failed",
            "abandoned",
            "invalid",
        }:
            raise ValueError("trial_status_invalid")
        if not isinstance(self.backend_versions, Mapping) or any(
            type(key) is not str or not key or type(value) is not str or not value
            for key, value in self.backend_versions.items()
        ):
            raise ValueError("trial_backend_versions_invalid")
        if not isinstance(self.thread_settings, Mapping) or any(
            type(key) is not str or not key or type(value) is not int or value <= 0
            for key, value in self.thread_settings.items()
        ):
            raise ValueError("trial_thread_settings_invalid")
        if self.holdout_accessed is not False:
            raise ValueError("trial_holdout_access_forbidden")
        if type(self.development_test_contaminated) is not bool:
            raise ValueError("trial_development_test_contamination_invalid")
        if self.contract_version != "1.0":
            raise ValueError("trial_contract_version_invalid")
        for field_name in (
            "normalized_parameters",
            "backend_versions",
            "thread_settings",
            "validation_metrics",
            "development_test_metrics",
            "undefined_reasons",
            "cost_assumptions",
            "risk_assumptions",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError("trial_mapping_invalid")
            snapshot = bounded_qrp_json_object_snapshot(
                dict(value), max_bytes=_MAX_TRIAL_RECORD_BYTES
            )
            object.__setattr__(self, field_name, _freeze_json(snapshot))
        if evidence_json_value_unsafe_reason(self.payload()) is not None:
            raise ValueError("trial_evidence_unsafe")

    def payload(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "event_id": self.event_id,
            "attempt_index": self.attempt_index,
            "experiment_id": self.experiment_id,
            "protocol_hash": self.protocol_hash,
            "candidate_family_id": self.candidate_family_id,
            "candidate_id": self.candidate_id,
            "trial_id": self.trial_id,
            "fold_id": self.fold_id,
            "market_id": self.market_id,
            "research_dataset_id": self.research_dataset_id,
            "target_id": self.target_id,
            "horizon_id": self.horizon_id,
            "common_coverage_id": self.common_coverage_id,
            "factor_definition_ids": list(self.factor_definition_ids),
            "model_definition_id": self.model_definition_id,
            "portfolio_definition_id": self.portfolio_definition_id,
            "normalized_parameters": _thaw_json(self.normalized_parameters),
            "random_seed": self.random_seed,
            "backend_versions": _thaw_json(self.backend_versions),
            "thread_settings": _thaw_json(self.thread_settings),
            "status": self.status,
            "validation_metrics": _thaw_json(self.validation_metrics),
            "development_test_metrics": _thaw_json(self.development_test_metrics),
            "development_test_contaminated": self.development_test_contaminated,
            "undefined_reasons": _thaw_json(self.undefined_reasons),
            "cost_assumptions": _thaw_json(self.cost_assumptions),
            "risk_assumptions": _thaw_json(self.risk_assumptions),
            "rebalance_assumption": self.rebalance_assumption,
            "artifact_lineage": list(self.artifact_lineage),
            "holdout_accessed": self.holdout_accessed,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> TrialRecord:
        value = dict(payload)
        value["factor_definition_ids"] = tuple(value["factor_definition_ids"])
        value["artifact_lineage"] = tuple(value["artifact_lineage"])
        return cls(**value)  # type: ignore[arg-type]


def append_trial_record(
    root: Path,
    plan: ExperimentPlan,
    record: TrialRecord,
) -> Path:
    if not isinstance(record, TrialRecord):
        raise TypeError("trial_record_required")
    _validate_record_against_plan(plan, record)
    existing = read_trial_ledger(root, plan)
    if record.attempt_index != len(existing) + 1:
        raise ValueError("trial_attempt_not_next")
    if any(item.development_test_contaminated for item in existing):
        raise ValueError("trial_contaminated_experiment_requires_new_id")
    path = Path(root) / f"{record.attempt_index:020d}-{record.event_id}.json"
    return atomic_write_json(path, record.payload())


def read_trial_ledger(
    root: Path,
    plan: ExperimentPlan,
) -> tuple[TrialRecord, ...]:
    if not isinstance(plan, ExperimentPlan):
        raise TypeError("experiment_plan_required")
    root = Path(root)
    if not root.exists():
        return ()
    entries = read_bounded_qrp_json_directory(root, max_members=plan.trial_budget)
    records = tuple(TrialRecord.from_payload(payload) for _name, payload in entries)
    indices = tuple(record.attempt_index for record in records)
    if indices != tuple(range(1, len(records) + 1)):
        raise ValueError("trial_ledger_order_invalid")
    if any(
        name != f"{record.attempt_index:020d}-{record.event_id}.json"
        for (name, _payload), record in zip(entries, records, strict=True)
    ):
        raise ValueError("trial_ledger_filename_mismatch")
    for record in records:
        _validate_record_against_plan(plan, record)
    if any(record.development_test_contaminated for record in records[:-1]):
        raise ValueError("trial_contaminated_experiment_reused")
    return records


def _validate_record_against_plan(
    plan: ExperimentPlan,
    record: TrialRecord,
) -> None:
    if not isinstance(plan, ExperimentPlan):
        raise TypeError("experiment_plan_required")
    expected = (
        (record.experiment_id, plan.experiment_id),
        (record.protocol_hash, plan.protocol_hash),
        (record.candidate_family_id, plan.candidate_family_id),
        (record.market_id, str(plan.market_id)),
        (record.research_dataset_id, plan.research_dataset_id),
        (record.target_id, plan.target_id),
    )
    if any(actual != required for actual, required in expected):
        raise ValueError("trial_plan_mismatch")
    if record.candidate_id not in plan.candidate_ids:
        raise ValueError("trial_candidate_not_registered")
    if record.fold_id not in {fold.fold_id for fold in plan.folds}:
        raise ValueError("trial_fold_not_registered")
    if record.attempt_index > plan.trial_budget:
        raise ValueError("trial_budget_exceeded")


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value

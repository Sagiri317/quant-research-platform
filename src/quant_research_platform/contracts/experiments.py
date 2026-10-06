"""Common experiment contracts with explicit label-availability eligibility."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import date, datetime, time
from functools import cached_property
from math import isfinite
from types import MappingProxyType
from typing import Literal

import pandas as pd

from quant_research_platform.contracts.business_manifest import _freeze_json_value
from quant_research_platform.contracts.ids import MarketId, validate_market_id
from quant_research_platform.contracts.json_codec import (
    bounded_qrp_json_object_snapshot,
    canonical_json_hash,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
from quant_research_platform.contracts.validation import (
    _validate_non_empty_string,
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)

SelectionDirection = Literal["maximize", "minimize"]
ExperimentRole = Literal["train", "validation", "test"]


def _model_inference_blocker(
    *, access_window: str, selection_groups: tuple[str, ...]
) -> str | None:
    """Qualify model curves by the information used for parameter selection."""

    if access_window == "development" and "global" in selection_groups:
        return "global_development_selection_reuses_validation"
    return None


@dataclass(frozen=True, kw_only=True)
class ExperimentFold:
    fold_id: str
    train: TimeWindow
    validation: TimeWindow

    def __post_init__(self) -> None:
        validate_run_id(self.fold_id)
        if not all(
            isinstance(window, TimeWindow) for window in (self.train, self.validation)
        ):
            raise TypeError("experiment_fold_window_required")
        if (
            len(
                {
                    isinstance(bound, datetime)
                    for window in (self.train, self.validation)
                    for bound in (window.start, window.end)
                }
            )
            != 1
        ):
            raise ValueError("experiment_time_kind_mismatch")
        if not _strictly_before(self.train.end, self.validation.start):
            raise ValueError("experiment_fold_windows_overlap")


@dataclass(frozen=True, kw_only=True)
class ExperimentPlan:
    stage: Literal["selection", "refit"]
    experiment_id: str
    market_id: MarketId
    research_dataset_id: str
    target_id: str
    candidate_family_id: str
    data_snapshot_id: str
    input_lineage: tuple[str, ...]
    folds: tuple[ExperimentFold, ...]
    test: TimeWindow
    selection_metric: str | None
    selection_direction: SelectionDirection | None
    candidate_ids: tuple[str, ...]
    trial_budget: int
    random_seed: int
    tie_break_order: tuple[str, ...] = ()
    backend_versions: Mapping[str, str] = field(default_factory=dict)
    thread_settings: Mapping[str, int] = field(default_factory=dict)
    cost_assumptions: Mapping[str, object] = field(default_factory=dict)
    risk_assumptions: Mapping[str, object] = field(default_factory=dict)
    candidate_parameters: Mapping[str, object] = field(default_factory=dict)
    selection_ledger_semantic_identities: tuple[str, ...] = ()
    fit_windows: tuple[TimeWindow, ...] = ()
    fit_cutoff: date | datetime | None = None
    selection_group: Literal["global", "quarterly"] | None = None
    research_protocol_identity: str = ""
    holdout_status: Literal["LOCKED"] = "LOCKED"

    def __post_init__(self) -> None:
        validate_run_id(self.experiment_id)
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        for value, error in (
            (self.research_dataset_id, "experiment_dataset_id_required"),
            (self.target_id, "experiment_target_id_required"),
            (self.candidate_family_id, "experiment_candidate_family_id_required"),
            (self.data_snapshot_id, "experiment_data_snapshot_id_required"),
        ):
            _validate_non_empty_string(value, error)
        _validate_string_tuple(self.input_lineage, "experiment_input_lineage")
        _validate_string_tuple(self.candidate_ids, "experiment_candidate_ids")
        if self.tie_break_order:
            _validate_string_tuple(self.tie_break_order, "experiment_tie_break_order")
            if set(self.tie_break_order) != set(self.candidate_ids):
                raise ValueError("experiment_tie_break_order_mismatch")
        if type(self.folds) is not tuple or any(
            not isinstance(fold, ExperimentFold) for fold in self.folds
        ):
            raise ValueError("experiment_folds_invalid")
        if len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise ValueError("experiment_fold_ids_duplicate")
        if not isinstance(self.test, TimeWindow):
            raise TypeError("experiment_test_window_required")
        if type(self.fit_windows) is not tuple or any(
            not isinstance(window, TimeWindow) for window in self.fit_windows
        ):
            raise ValueError("experiment_fit_windows_invalid")
        parents = self.selection_ledger_semantic_identities
        if (
            type(parents) is not tuple
            or any(
                type(identity) is not str
                or re.fullmatch(r"[0-9a-f]{64}", identity) is None
                for identity in parents
            )
            or parents != tuple(sorted(set(parents)))
        ):
            raise ValueError("experiment_selection_parents_invalid")
        window_bounds = [self.test.start, self.test.end]
        for fold in self.folds:
            for window in (fold.train, fold.validation):
                window_bounds.extend((window.start, window.end))
        for window in self.fit_windows:
            window_bounds.extend((window.start, window.end))
        if self.fit_cutoff is not None:
            if not isinstance(self.fit_cutoff, date):
                raise ValueError("experiment_fit_cutoff_invalid")
            window_bounds.append(self.fit_cutoff)
        if len({isinstance(bound, datetime) for bound in window_bounds}) != 1:
            raise ValueError("experiment_time_kind_mismatch")
        if any(
            not _strictly_before(fold.validation.end, self.test.start)
            for fold in self.folds
        ):
            raise ValueError("experiment_test_overlap")
        if self.market_id == "a_share" and self.stage == "refit":
            if self.selection_group not in {"global", "quarterly"}:
                raise ValueError("experiment_selection_group_required")
        elif self.selection_group is not None:
            raise ValueError("experiment_selection_group_invalid")
        if self.stage == "selection":
            _validate_non_empty_string(
                self.selection_metric, "experiment_selection_metric_required"
            )
            if self.selection_direction not in {"maximize", "minimize"}:
                raise ValueError("experiment_selection_direction_invalid")
            if (
                not self.folds
                or parents
                or self.fit_windows
                or self.fit_cutoff is not None
            ):
                raise ValueError("experiment_selection_plan_invalid")
        elif self.stage == "refit":
            if (
                self.selection_metric is not None
                or self.selection_direction is not None
                or self.tie_break_order
                or self.folds
                or len(self.candidate_ids) != 1
                or self.trial_budget != 1
                or not parents
                or not self.fit_windows
                or self.fit_cutoff is None
                or not _strictly_before(self.fit_cutoff, self.test.start)
                or _strictly_before(self.fit_cutoff, self.fit_windows[-1].end)
                or any(
                    not _strictly_before(left.end, right.start)
                    for left, right in zip(
                        self.fit_windows, self.fit_windows[1:], strict=False
                    )
                )
            ):
                raise ValueError("experiment_refit_plan_invalid")
        else:
            raise ValueError("experiment_stage_invalid")
        if type(self.trial_budget) is not int or self.trial_budget <= 0:
            raise ValueError("experiment_trial_budget_invalid")
        if self.trial_budget < len(self.candidate_ids) * len(self.fold_ids):
            raise ValueError("experiment_trial_budget_too_small")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("experiment_random_seed_invalid")
        if self.holdout_status != "LOCKED":
            raise ValueError("experiment_holdout_must_be_locked")
        if (
            self.research_protocol_identity
            and re.fullmatch(r"[0-9a-f]{64}", self.research_protocol_identity) is None
        ):
            raise ValueError("experiment_research_protocol_identity_invalid")
        parameters = bounded_qrp_json_object_snapshot(
            _thaw_json(self.candidate_parameters), max_bytes=256 * 1024
        )
        if parameters and set(parameters) != set(self.candidate_ids):
            raise ValueError("experiment_candidate_parameters_mismatch")
        if self.stage == "refit" and (
            not parameters
            or type(parameters[self.candidate_ids[0]]) is not dict
            or parameters[self.candidate_ids[0]].get("random_seed") != self.random_seed
        ):
            raise ValueError("experiment_refit_plan_invalid")
        object.__setattr__(self, "candidate_parameters", _freeze_json_value(parameters))
        object.__setattr__(
            self, "backend_versions", _freeze_string_mapping(self.backend_versions)
        )
        object.__setattr__(
            self, "thread_settings", _freeze_int_mapping(self.thread_settings)
        )
        object.__setattr__(
            self, "cost_assumptions", _freeze_scalar_mapping(self.cost_assumptions)
        )
        object.__setattr__(
            self, "risk_assumptions", _freeze_scalar_mapping(self.risk_assumptions)
        )

    @property
    def fold_ids(self) -> tuple[str, ...]:
        return (
            tuple(fold.fold_id for fold in self.folds)
            if self.stage == "selection"
            else ("refit",)
        )

    @cached_property
    def protocol_hash(self) -> str:
        return canonical_json_hash(self.protocol_payload())

    def protocol_payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "selection_group": self.selection_group,
            "selection_ledger_semantic_identities": list(
                self.selection_ledger_semantic_identities
            ),
            "fit_windows": [_window_payload(window) for window in self.fit_windows],
            "fit_cutoff": (
                None
                if self.fit_cutoff is None
                else time_value_isoformat(self.fit_cutoff, preserve_datetime=True)
            ),
            "market_id": str(self.market_id),
            "research_dataset_id": self.research_dataset_id,
            "target_id": self.target_id,
            "candidate_family_id": self.candidate_family_id,
            "data_snapshot_id": self.data_snapshot_id,
            "input_lineage": list(self.input_lineage),
            "folds": [_fold_payload(fold) for fold in self.folds],
            "test": _window_payload(self.test),
            "selection_metric": self.selection_metric,
            "selection_direction": self.selection_direction,
            "tie_break_order": list(self.tie_break_order),
            "candidate_ids": list(self.candidate_ids),
            "trial_budget": self.trial_budget,
            "random_seed": self.random_seed,
            "backend_versions": dict(self.backend_versions),
            "thread_settings": dict(self.thread_settings),
            "cost_assumptions": dict(self.cost_assumptions),
            "risk_assumptions": dict(self.risk_assumptions),
            "candidate_parameters": _thaw_json(self.candidate_parameters),
            "research_protocol_identity": self.research_protocol_identity,
            "holdout_status": self.holdout_status,
        }

    @classmethod
    def from_payload(cls, value: dict[str, object]) -> ExperimentPlan:
        if (
            type(value) is not dict
            or set(value) != {item.name for item in fields(cls)}
            or any(
                type(value.get(key)) is not list
                for key in (
                    "folds",
                    "fit_windows",
                    "selection_ledger_semantic_identities",
                    "input_lineage",
                    "candidate_ids",
                    "tie_break_order",
                )
            )
            or any(
                type(value.get(key)) is not dict
                for key in (
                    "test",
                    "backend_versions",
                    "thread_settings",
                    "cost_assumptions",
                    "risk_assumptions",
                    "candidate_parameters",
                )
            )
        ):
            raise ValueError("experiment_plan_payload_invalid")
        payload = dict(value)
        folds = []
        for item in payload["folds"]:
            if type(item) is not dict or any(
                type(item.get(key)) is not dict for key in ("train", "validation")
            ):
                raise ValueError("experiment_plan_payload_invalid")
            fold = dict(item)
            for key in ("train", "validation"):
                fold[key] = TimeWindow(**fold[key])
            folds.append(ExperimentFold(**fold))
        payload["folds"] = tuple(folds)
        payload["test"] = TimeWindow(**payload["test"])
        payload["fit_windows"] = tuple(
            TimeWindow(**item) for item in payload["fit_windows"]
        )
        cutoff = payload["fit_cutoff"]
        payload["fit_cutoff"] = (
            None if cutoff is None else TimeWindow(cutoff, cutoff).start
        )
        for key in (
            "input_lineage",
            "candidate_ids",
            "tie_break_order",
            "selection_ledger_semantic_identities",
        ):
            payload[key] = tuple(payload[key])
        return cls(**payload)

    def window(self, role: ExperimentRole) -> tuple[TimeWindow, ...]:
        if role == "test":
            raise PermissionError("test_locked")
        if role not in {"train", "validation"}:
            raise ValueError("experiment_role_invalid")
        return tuple(getattr(fold, role) for fold in self.folds)


def label_maturity_mask(
    frame: pd.DataFrame,
    *,
    fitting_cutoff: object,
    availability_column: str | None = "label_available_at",
    target_column: str | None = None,
) -> pd.Series:
    """Select finite labels observable by the cutoff, independently of eligibility."""
    required = {"timestamp"}
    if availability_column is not None:
        required.add(availability_column)
    if target_column is not None:
        required.add(target_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("legal_training_columns_missing: " + ", ".join(missing))
    cutoff = strict_timestamp_series(
        pd.Series([fitting_cutoff]), error="invalid_fitting_cutoff"
    ).iloc[0]
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_training_timestamp"
    )
    keep = timestamps.le(cutoff)
    if availability_column is not None:
        available = strict_timestamp_series(
            frame[availability_column],
            error="invalid_label_available_at",
            allow_null=True,
        )
        keep &= available.notna() & available.le(cutoff)
    if target_column is not None:
        values = strict_finite_numeric_series(
            frame[target_column],
            error=f"invalid_training_value: {target_column}",
            allow_null=True,
        )
        keep &= values.notna()
    return keep


def legal_training_rows(
    frame: pd.DataFrame,
    *,
    fitting_cutoff: object,
    availability_column: str | None = "label_available_at",
    feature_columns: tuple[str, ...] = (),
    target_column: str | None = None,
) -> pd.DataFrame:
    """Select eligible finite inputs with mature labels."""
    missing = sorted({"eligible", *feature_columns}.difference(frame.columns))
    if missing:
        raise ValueError("legal_training_columns_missing: " + ", ".join(missing))
    eligible = frame["eligible"]
    if not pd.api.types.is_bool_dtype(eligible) or bool(eligible.isna().any()):
        raise ValueError("invalid_training_eligibility")
    keep = (
        label_maturity_mask(
            frame,
            fitting_cutoff=fitting_cutoff,
            availability_column=availability_column,
            target_column=target_column,
        )
        & eligible
    )
    for column in feature_columns:
        values = strict_finite_numeric_series(
            frame[column], error=f"invalid_training_value: {column}", allow_null=True
        )
        keep &= values.notna()
    return frame.loc[keep].copy()


def training_availability_columns(label_column: str) -> tuple[str, ...]:
    if label_column == "model_target":
        return ("forward_return_20d_available_at",)
    if label_column.startswith(
        (
            "forward_return_",
            "future_mid_return_",
            "future_mid_direction_",
            "maker_bid_sell_pressure_proxy_",
        )
    ):
        return (f"{label_column}_available_at",)
    return ("label_available_at",)


def resolve_training_availability_column(
    frame: pd.DataFrame,
    *,
    label_column: str,
) -> str:
    column = training_availability_columns(label_column)[0]
    if column not in frame.columns:
        raise ValueError("label_availability_column_required")
    return column


def _strictly_before(left: date | datetime, right: date | datetime) -> bool:
    left_value = (
        left if isinstance(left, datetime) else datetime.combine(left, time.min)
    )
    right_value = (
        right if isinstance(right, datetime) else datetime.combine(right, time.min)
    )
    return left_value < right_value


def _validate_string_tuple(value: object, error: str) -> None:
    if (
        type(value) is not tuple
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(error)


def _freeze_string_mapping(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str or not key or type(item) is not str or not item
        for key, item in value.items()
    ):
        raise ValueError("experiment_backend_versions_invalid")
    return MappingProxyType(dict(value))


def _freeze_int_mapping(value: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or any(
        type(key) is not str or not key or type(item) is not int or item <= 0
        for key, item in value.items()
    ):
        raise ValueError("experiment_thread_settings_invalid")
    return MappingProxyType(dict(value))


def _freeze_scalar_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("experiment_assumptions_invalid")
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValueError("experiment_assumptions_invalid")
        if item is None or type(item) in {bool, int, str}:
            frozen[key] = item
        elif type(item) is float and isfinite(item):
            frozen[key] = item
        else:
            raise ValueError("experiment_assumptions_invalid")
    return MappingProxyType(frozen)


def _window_payload(window: TimeWindow) -> dict[str, str]:
    return {
        "start": time_value_isoformat(window.start, preserve_datetime=True),
        "end": time_value_isoformat(window.end, preserve_datetime=True),
    }


def _fold_payload(fold: ExperimentFold) -> dict[str, object]:
    return {
        "fold_id": fold.fold_id,
        "train": _window_payload(fold.train),
        "validation": _window_payload(fold.validation),
    }


TrialStatus = Literal["attempted", "completed", "failed", "abandoned", "invalid"]
_PROTOCOL_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_TRIAL_RECORD_BYTES = 256 * 1024


@dataclass(frozen=True, kw_only=True)
class TrialRecord:
    event_id: str
    run_id: str
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
    random_seed: int
    backend_versions: Mapping[str, str] = field(default_factory=dict)
    thread_settings: Mapping[str, int] = field(default_factory=dict)
    status: TrialStatus
    metrics: Mapping[str, object] = field(default_factory=dict)
    undefined_reasons: Mapping[str, object] = field(default_factory=dict)
    cost_assumptions: Mapping[str, object] = field(default_factory=dict)
    risk_assumptions: Mapping[str, object] = field(default_factory=dict)
    rebalance_assumption: str = "not_applicable"
    artifact_lineage: tuple[str, ...] = ()
    holdout_accessed: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.event_id,
            self.run_id,
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
        for field_name in (
            "normalized_parameters",
            "backend_versions",
            "thread_settings",
            "metrics",
            "undefined_reasons",
            "cost_assumptions",
            "risk_assumptions",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ValueError("trial_mapping_invalid")
            snapshot = bounded_qrp_json_object_snapshot(
                _thaw_json(value), max_bytes=_MAX_TRIAL_RECORD_BYTES
            )
            object.__setattr__(self, field_name, _freeze_json_value(snapshot))
        if evidence_json_value_unsafe_reason(self.payload()) is not None:
            raise ValueError("trial_evidence_unsafe")

    def payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
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
            "metrics": _thaw_json(self.metrics),
            "undefined_reasons": _thaw_json(self.undefined_reasons),
            "cost_assumptions": _thaw_json(self.cost_assumptions),
            "risk_assumptions": _thaw_json(self.risk_assumptions),
            "rebalance_assumption": self.rebalance_assumption,
            "artifact_lineage": list(self.artifact_lineage),
            "holdout_accessed": self.holdout_accessed,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> TrialRecord:
        if (
            type(payload) is not dict
            or any(
                type(payload.get(key)) is not list
                for key in ("factor_definition_ids", "artifact_lineage")
            )
            or any(
                type(payload.get(key)) is not dict
                for key in (
                    "normalized_parameters",
                    "backend_versions",
                    "thread_settings",
                    "metrics",
                    "undefined_reasons",
                    "cost_assumptions",
                    "risk_assumptions",
                )
            )
        ):
            raise ValueError("trial_record_payload_invalid")
        value = dict(payload)
        value["factor_definition_ids"] = tuple(value["factor_definition_ids"])
        value["artifact_lineage"] = tuple(value["artifact_lineage"])
        return cls(**value)  # type: ignore[arg-type]


def validate_trial_record_against_plan(
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
    if plan.stage == "refit" and record.random_seed != plan.random_seed:
        raise ValueError("trial_refit_seed_mismatch")
    if (
        plan.candidate_parameters
        and record.normalized_parameters
        != plan.candidate_parameters[record.candidate_id]
    ):
        raise ValueError("trial_parameters_not_frozen")
    if record.fold_id not in plan.fold_ids:
        raise ValueError("trial_fold_not_registered")
    if record.attempt_index > plan.trial_budget:
        raise ValueError("trial_budget_exceeded")


def trial_ledger_semantic_identity(
    plan: ExperimentPlan, records: tuple[TrialRecord, ...]
) -> str:
    return canonical_json_hash(trial_ledger_semantic_payload(plan, records))


def trial_ledger_semantic_payload(
    plan: ExperimentPlan, records: tuple[TrialRecord, ...]
) -> dict[str, object]:
    """Only invocation provenance is excluded; plans and result values stay exact."""
    protocol = plan.protocol_payload()
    del protocol["experiment_id"]
    del protocol["thread_settings"]
    events = []
    for record in records:
        validate_trial_record_against_plan(plan, record)
        event = record.payload()
        for key in (
            "event_id",
            "run_id",
            "experiment_id",
            "trial_id",
            "protocol_hash",
            "thread_settings",
        ):
            del event[key]
        events.append(event)
    return {"plan": protocol, "events": events}


def validate_trial_ledger_reference(value: object) -> dict[str, str]:
    if type(value) is not dict or set(value) != {
        "run_id",
        "experiment_id",
        "protocol_hash",
        "semantic_identity",
    }:
        raise ValueError("trial_ledger_reference_invalid")
    validate_run_id(value["run_id"])
    validate_run_id(value["experiment_id"])
    for key in ("protocol_hash", "semantic_identity"):
        if type(value[key]) is not str or _PROTOCOL_HASH.fullmatch(value[key]) is None:
            raise ValueError("trial_ledger_reference_invalid")
    return dict(value)


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw_json(item) for item in value]
    return value

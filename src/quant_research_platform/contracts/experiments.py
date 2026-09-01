"""Common experiment contracts with explicit label-availability eligibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time
from math import isfinite
from types import MappingProxyType
from typing import Literal

import pandas as pd

from quant_research_platform.contracts.ids import MarketId, validate_market_id
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
from quant_research_platform.contracts.validation import (
    _validate_non_empty_string,
    strict_timestamp_series,
)

SelectionDirection = Literal["maximize", "minimize"]
ExperimentRole = Literal["train", "validation", "development_test", "final_holdout"]


@dataclass(frozen=True, kw_only=True)
class ExperimentFold:
    fold_id: str
    train: TimeWindow
    validation: TimeWindow
    development_test: TimeWindow
    step_kind: str = "explicit"
    train_end_step: int | None = None
    validation_start_step: int | None = None
    validation_end_step: int | None = None
    development_test_start_step: int | None = None
    development_test_end_step: int | None = None
    final_holdout_start_step: int | None = None
    embargo_steps: int = 0
    complete: bool = True

    def __post_init__(self) -> None:
        validate_run_id(self.fold_id)
        if not all(
            isinstance(window, TimeWindow)
            for window in (self.train, self.validation, self.development_test)
        ):
            raise TypeError("experiment_fold_window_required")
        if not _strictly_before(self.train.end, self.validation.start) or not (
            _strictly_before(self.validation.end, self.development_test.start)
        ):
            raise ValueError("experiment_fold_windows_overlap")
        if type(self.embargo_steps) is not int or self.embargo_steps < 0:
            raise ValueError("experiment_embargo_invalid")
        _validate_non_empty_string(self.step_kind, "experiment_step_kind_required")
        steps = (
            self.train_end_step,
            self.validation_start_step,
            self.validation_end_step,
            self.development_test_start_step,
            self.development_test_end_step,
            self.final_holdout_start_step,
        )
        if any(step is not None for step in steps):
            if any(type(step) is not int or step < 0 for step in steps):
                raise ValueError("experiment_boundary_step_invalid")
            assert all(step is not None for step in steps)
            if self.validation_start_step <= self.train_end_step:
                raise ValueError("experiment_boundary_step_invalid")
            if self.validation_end_step < self.validation_start_step:
                raise ValueError("experiment_boundary_step_invalid")
            if self.development_test_start_step <= self.validation_end_step:
                raise ValueError("experiment_boundary_step_invalid")
            if self.development_test_end_step < self.development_test_start_step:
                raise ValueError("experiment_boundary_step_invalid")
            if self.final_holdout_start_step <= self.development_test_end_step:
                raise ValueError("experiment_holdout_boundary_step_invalid")
            if (
                self.development_test_start_step - self.validation_end_step - 1
                != self.embargo_steps
            ):
                raise ValueError("experiment_embargo_gap_invalid")
        if type(self.complete) is not bool:
            raise ValueError("experiment_fold_complete_invalid")


@dataclass(frozen=True, kw_only=True)
class ExperimentPlan:
    experiment_id: str
    market_id: MarketId
    research_dataset_id: str
    target_id: str
    candidate_family_id: str
    data_snapshot_id: str
    input_lineage: tuple[str, ...]
    folds: tuple[ExperimentFold, ...]
    final_holdout: TimeWindow
    selection_metric: str
    selection_direction: SelectionDirection
    candidate_ids: tuple[str, ...]
    trial_budget: int
    random_seed: int
    tie_break_order: tuple[str, ...] = ()
    backend_versions: Mapping[str, str] = field(default_factory=dict)
    thread_settings: Mapping[str, int] = field(default_factory=dict)
    cost_assumptions: Mapping[str, object] = field(default_factory=dict)
    risk_assumptions: Mapping[str, object] = field(default_factory=dict)
    holdout_status: Literal["LOCKED"] = "LOCKED"

    def __post_init__(self) -> None:
        validate_run_id(self.experiment_id)
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        for value, error in (
            (self.research_dataset_id, "experiment_dataset_id_required"),
            (self.target_id, "experiment_target_id_required"),
            (self.candidate_family_id, "experiment_candidate_family_id_required"),
            (self.data_snapshot_id, "experiment_data_snapshot_id_required"),
            (self.selection_metric, "experiment_selection_metric_required"),
        ):
            _validate_non_empty_string(value, error)
        _validate_string_tuple(self.input_lineage, "experiment_input_lineage")
        _validate_string_tuple(self.candidate_ids, "experiment_candidate_ids")
        if self.tie_break_order:
            _validate_string_tuple(self.tie_break_order, "experiment_tie_break_order")
            if set(self.tie_break_order) != set(self.candidate_ids):
                raise ValueError("experiment_tie_break_order_mismatch")
        if (
            type(self.folds) is not tuple
            or not self.folds
            or any(not isinstance(fold, ExperimentFold) for fold in self.folds)
        ):
            raise ValueError("experiment_folds_invalid")
        if len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise ValueError("experiment_fold_ids_duplicate")
        if len({fold.step_kind for fold in self.folds}) != 1:
            raise ValueError("experiment_step_kind_mismatch")
        if not isinstance(self.final_holdout, TimeWindow):
            raise TypeError("experiment_final_holdout_window_required")
        if any(
            not _strictly_before(fold.development_test.end, self.final_holdout.start)
            for fold in self.folds
        ):
            raise ValueError("experiment_holdout_overlap")
        window_bounds = [self.final_holdout.start, self.final_holdout.end]
        for fold in self.folds:
            for window in (fold.train, fold.validation, fold.development_test):
                window_bounds.extend((window.start, window.end))
        if len({isinstance(bound, datetime) for bound in window_bounds}) != 1:
            raise ValueError("experiment_time_kind_mismatch")
        if self.selection_direction not in {"maximize", "minimize"}:
            raise ValueError("experiment_selection_direction_invalid")
        if type(self.trial_budget) is not int or self.trial_budget <= 0:
            raise ValueError("experiment_trial_budget_invalid")
        if self.trial_budget < len(self.candidate_ids) * len(self.folds):
            raise ValueError("experiment_trial_budget_too_small")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("experiment_random_seed_invalid")
        if self.holdout_status != "LOCKED":
            raise ValueError("experiment_holdout_must_be_locked")
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
    def protocol_hash(self) -> str:
        return canonical_json_hash(self.protocol_payload())

    def protocol_payload(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "market_id": str(self.market_id),
            "research_dataset_id": self.research_dataset_id,
            "target_id": self.target_id,
            "candidate_family_id": self.candidate_family_id,
            "data_snapshot_id": self.data_snapshot_id,
            "input_lineage": list(self.input_lineage),
            "folds": [_fold_payload(fold) for fold in self.folds],
            "final_holdout": _window_payload(self.final_holdout),
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
            "holdout_status": self.holdout_status,
        }

    def development_windows(self) -> tuple[TimeWindow, ...]:
        return tuple(fold.development_test for fold in self.folds)

    def window(self, role: ExperimentRole) -> tuple[TimeWindow, ...]:
        if role == "final_holdout":
            raise PermissionError("final_holdout_locked")
        if role not in {"train", "validation", "development_test"}:
            raise ValueError("experiment_role_invalid")
        return tuple(getattr(fold, role) for fold in self.folds)


def legal_training_rows(
    frame: pd.DataFrame,
    *,
    fitting_cutoff: object,
    availability_column: str = "label_available_at",
) -> pd.DataFrame:
    """Select rows whose label was actually available by the fitting cutoff."""

    if availability_column not in frame.columns:
        raise ValueError("label_availability_column_required")
    cutoff = strict_timestamp_series(
        pd.Series([fitting_cutoff]), error="invalid_fitting_cutoff"
    ).iloc[0]
    available = strict_timestamp_series(
        frame[availability_column], error="invalid_label_available_at", allow_null=True
    )
    return frame.loc[available.notna() & available.le(cutoff)].copy()


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
        "development_test": _window_payload(fold.development_test),
        "embargo_steps": fold.embargo_steps,
        "step_kind": fold.step_kind,
        "train_end_step": fold.train_end_step,
        "validation_start_step": fold.validation_start_step,
        "validation_end_step": fold.validation_end_step,
        "development_test_start_step": fold.development_test_start_step,
        "development_test_end_step": fold.development_test_end_step,
        "final_holdout_start_step": fold.final_holdout_start_step,
        "complete": fold.complete,
    }

"""Runtime-owned resource policy and payload diagnostics."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

MIB = 1024 * 1024
_TASK_MEMORY_CEILING_NUMERATOR = 4
_TASK_MEMORY_CEILING_DENOMINATOR = 5
_PLANNING_UNCERTAINTY_BYTES = 64 * 1024


@dataclass(slots=True)
class _ReservationLedger:
    state: str = "READY"
    scope_count: int = 0
    by_owner: dict[str, int] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(self.by_owner.values())

    def reserve(self, owner: str, reserve_bytes: int, *, budget_bytes: int) -> None:
        if self.total_bytes + reserve_bytes > budget_bytes:
            raise MemoryError("runtime_memory_budget_exceeded")
        self.by_owner[owner] = self.by_owner.get(owner, 0) + reserve_bytes

    def release(self, owner: str, reserve_bytes: int) -> None:
        remaining = self.by_owner.get(owner, 0) - reserve_bytes
        if remaining < 0:
            raise RuntimeError("runtime_memory_reservation_underflow")
        if remaining:
            self.by_owner[owner] = remaining
        else:
            self.by_owner.pop(owner, None)


@dataclass(slots=True)
class _OperationRssMeasurement:
    rss0_bytes: int
    hard_budget_bytes: int
    sampled_peak_rss_bytes: int = 0
    sample_count: int = 0
    state: str = "READY"
    error: BaseException | None = None
    _paused: bool = False

    def start(self) -> None:
        if self.state != "READY":
            raise RuntimeError("runtime_rss_sampler_already_started")
        self.state = "ACTIVE"

    def quiesce(self) -> dict[str, int | float | str]:
        """Pause after a fresh sample so a publish decision cannot race sampling."""

        if self.state != "ACTIVE" or self._paused:
            raise RuntimeError("runtime_rss_sampler_not_active")
        self._sample()
        self._paused = True
        return self._evidence()

    def resume(self) -> None:
        """Resume sampling before rejection rollback and cleanup."""

        if self.state != "ACTIVE" or not self._paused:
            raise RuntimeError("runtime_rss_sampler_not_paused")
        self._paused = False

    def checkpoint_evidence(self) -> dict[str, int | float | str]:
        if self.state != "ACTIVE":
            raise RuntimeError("runtime_rss_sampler_not_active")
        if not self._paused:
            self._sample()
        return self._evidence()

    def stop(self) -> None:
        if self.state == "RETIRED":
            return
        if self.state == "ACTIVE" and not self._paused:
            self._sample()
        self.state = "RETIRED"

    def _sample(self) -> None:
        try:
            rss = _exact_nonnegative_int(
                _process_rss_bytes(),
                error="invalid_runtime_resource_rss",
            )
        except BaseException as exc:
            self.error = exc
            return
        self.sampled_peak_rss_bytes = max(self.sampled_peak_rss_bytes, rss)
        self.sample_count += 1

    def evidence(self) -> dict[str, int | float | str]:
        if self.state != "RETIRED":
            raise RuntimeError("runtime_rss_measurement_incomplete")
        return self._evidence()

    def _evidence(self) -> dict[str, int | float | str]:
        peak = max(self.rss0_bytes, self.sampled_peak_rss_bytes)
        incremental = peak - self.rss0_bytes
        telemetry_status = (
            "AVAILABLE" if self.error is None and self.sample_count else "UNAVAILABLE"
        )
        status = (
            "PASS"
            if telemetry_status == "AVAILABLE"
            and incremental <= self.hard_budget_bytes
            else "FAIL"
        )
        return {
            "rss0_bytes": self.rss0_bytes,
            "hard_budget_bytes": self.hard_budget_bytes,
            "sampled_peak_rss_bytes": peak,
            "peak_incremental_rss_bytes": incremental,
            "measurement_metric": "process_rss_checkpoint_incremental",
            "sampling_mode": "owner_boundary_checkpoint",
            "actual_peak_proven": "NO",
            "sample_count": self.sample_count,
            "max_sample_interval_ms": 0.0,
            "cadence_status": "NOT_APPLICABLE",
            "telemetry_status": telemetry_status,
            "status": status,
        }


@dataclass(frozen=True, init=False, eq=False, slots=True)
class RuntimeResourcePolicy:
    """Immutable one-shot resource admission shared by one root operation."""

    captured_available_memory_bytes: int
    rss0_bytes: int
    task_memory_budget_bytes: int
    hard_memory_budget_bytes: int
    planning_uncertainty_bytes: int
    _ledger: _ReservationLedger
    _measurement: _OperationRssMeasurement

    def __new__(cls) -> RuntimeResourcePolicy:
        raise TypeError("runtime_resource_policy_capture_required")

    @classmethod
    def capture(
        cls,
        *,
        explicit_cap_bytes: int | None = None,
    ) -> RuntimeResourcePolicy:
        """Capture available memory and RSS exactly once before root allocation."""

        explicit_cap = (
            None
            if explicit_cap_bytes is None
            else _exact_positive_int(
                explicit_cap_bytes,
                error="invalid_runtime_resource_explicit_cap",
            )
        )
        available = _exact_positive_int(
            _available_memory_bytes(),
            error="invalid_runtime_resource_available_memory",
        )
        rss0 = _exact_nonnegative_int(
            _process_rss_bytes(),
            error="invalid_runtime_resource_rss",
        )
        hard_budget = (
            available
            * _TASK_MEMORY_CEILING_NUMERATOR
            // _TASK_MEMORY_CEILING_DENOMINATOR
        )
        if explicit_cap is not None:
            hard_budget = min(hard_budget, explicit_cap)
        planning_budget = hard_budget - _PLANNING_UNCERTAINTY_BYTES
        if planning_budget <= 0:
            raise ValueError("invalid_runtime_resource_available_memory")
        measurement = _OperationRssMeasurement(rss0, hard_budget)
        policy = object.__new__(cls)
        object.__setattr__(policy, "captured_available_memory_bytes", available)
        object.__setattr__(policy, "rss0_bytes", rss0)
        object.__setattr__(policy, "task_memory_budget_bytes", planning_budget)
        object.__setattr__(policy, "hard_memory_budget_bytes", hard_budget)
        object.__setattr__(
            policy,
            "planning_uncertainty_bytes",
            _PLANNING_UNCERTAINTY_BYTES,
        )
        object.__setattr__(policy, "_ledger", _ReservationLedger())
        object.__setattr__(policy, "_measurement", measurement)
        return policy

    @property
    def reserved_memory_bytes(self) -> int:
        return self._ledger.total_bytes

    @property
    def remaining_memory_bytes(self) -> int:
        return max(0, self.task_memory_budget_bytes - self.reserved_memory_bytes)

    @contextmanager
    def reserve(self, owner: str, reserve_bytes: int) -> Iterator[None]:
        """Reserve one owner-local upper bound against this root budget."""

        if not isinstance(owner, str) or not owner:
            raise ValueError("invalid_runtime_memory_reservation_owner")
        reserve = _exact_nonnegative_int(
            reserve_bytes,
            error="invalid_runtime_memory_reservation",
        )
        if (
            _ACTIVE_RUNTIME_RESOURCE_POLICY is not self
            or self._ledger.state != "ACTIVE"
        ):
            raise RuntimeError("runtime_resource_policy_not_active")
        self._ledger.reserve(
            owner,
            reserve,
            budget_bytes=self.task_memory_budget_bytes,
        )
        try:
            yield
        finally:
            primary = sys.exception()
            try:
                self._ledger.release(owner, reserve)
            except BaseException:
                if primary is None:
                    raise

    def to_payload(self) -> dict[str, int]:
        """Return the exact private runtime resource payload."""

        return {
            "captured_available_memory_bytes": self.captured_available_memory_bytes,
            "rss0_bytes": self.rss0_bytes,
            "task_memory_budget_bytes": self.task_memory_budget_bytes,
            "hard_memory_budget_bytes": self.hard_memory_budget_bytes,
            "planning_uncertainty_bytes": self.planning_uncertainty_bytes,
        }

    def memory_evidence(self) -> dict[str, int | float | str]:
        """Return final operation-scoped RSS acceptance evidence."""

        return self._measurement.evidence()

    def assert_memory_within_budget(self) -> None:
        """Reject publication when the active operation has already failed."""

        evidence = (
            self._measurement.evidence()
            if self._measurement.state == "RETIRED"
            else self._measurement.checkpoint_evidence()
        )
        _require_memory_acceptance(self._measurement, evidence)

    def finish_memory_measurement(
        self,
        *,
        on_reject: Callable[[], None] | None = None,
    ) -> dict[str, int | float | str]:
        """Finish and accept the sampler at an owner-controlled publish boundary."""

        if (
            _ACTIVE_RUNTIME_RESOURCE_POLICY is not self
            or self._ledger.state != "ACTIVE"
            or self._measurement.state != "ACTIVE"
        ):
            raise RuntimeError("runtime_rss_sampler_not_active")
        try:
            if self.reserved_memory_bytes:
                raise RuntimeError("runtime_memory_reservation_leaked")
            evidence = self._measurement.quiesce()
            _require_memory_acceptance(self._measurement, evidence)
            self._measurement.stop()
            evidence = self.memory_evidence()
            _require_memory_acceptance(self._measurement, evidence)
        except BaseException as primary:
            if self._measurement.state == "ACTIVE":
                try:
                    self._measurement.resume()
                except BaseException as resume_error:
                    primary.add_note(
                        "runtime memory rejection sampler resume also failed: "
                        f"{type(resume_error).__name__}"
                    )
            if on_reject is not None:
                try:
                    on_reject()
                except BaseException as rollback_error:
                    primary.add_note(
                        "runtime memory rejection rollback also failed: "
                        f"{type(rollback_error).__name__}"
                    )
            if self._measurement.state != "RETIRED":
                try:
                    self._measurement.quiesce()
                except BaseException as cleanup_evidence_error:
                    primary.add_note(
                        "runtime memory rejection cleanup evidence also failed: "
                        f"{type(cleanup_evidence_error).__name__}"
                    )
                self._measurement.stop()
            raise
        return evidence


_ACTIVE_RUNTIME_RESOURCE_POLICY: RuntimeResourcePolicy | None = None


def current_runtime_resource_policy() -> RuntimeResourcePolicy | None:
    """Return the active policy without admitting a new root operation."""

    return _ACTIVE_RUNTIME_RESOURCE_POLICY


def bind_runtime_resource_policy(
    explicit: RuntimeResourcePolicy | None,
    *,
    allow_implicit_active: bool = True,
) -> RuntimeResourcePolicy | None:
    """Bind a lazy operation to its exact creation-time policy."""

    if explicit is not None and type(explicit) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    if type(allow_implicit_active) is not bool:
        raise TypeError("invalid_runtime_resource_policy_binding")
    active = current_runtime_resource_policy()
    if active is not None and explicit is None and not allow_implicit_active:
        raise RuntimeError("runtime_resource_policy_competing_root")
    if active is not None and explicit is not None and explicit is not active:
        raise RuntimeError("runtime_resource_policy_conflict")
    return active if active is not None else explicit


def active_runtime_resource_policy() -> RuntimeResourcePolicy:
    """Return the active snapshot or discover one for the caller to capture."""

    policy = current_runtime_resource_policy()
    if policy is None:
        return RuntimeResourcePolicy.capture()
    return policy


@contextmanager
def runtime_resource_policy_scope(
    policy: RuntimeResourcePolicy,
) -> Iterator[RuntimeResourcePolicy]:
    """Expose one explicit process-local root policy to nested synchronous stages."""

    global _ACTIVE_RUNTIME_RESOURCE_POLICY
    if type(policy) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    active = current_runtime_resource_policy()
    if active is not None:
        if active is not policy:
            raise RuntimeError("runtime_resource_policy_conflict")
    else:
        if policy._ledger.state == "RETIRED":
            raise RuntimeError("runtime_resource_policy_retired")
        if policy._ledger.state != "READY":
            raise RuntimeError("runtime_resource_policy_already_active")
        policy._ledger.state = "ACTIVE"
        _ACTIVE_RUNTIME_RESOURCE_POLICY = policy
    policy._ledger.scope_count += 1
    try:
        if policy._measurement.state == "READY":
            policy._measurement.start()
        yield policy
    finally:
        primary = sys.exception()
        policy._ledger.scope_count -= 1
        if not policy._ledger.scope_count:
            _retire_runtime_resource_policy(policy, primary=primary)


def _retire_runtime_resource_policy(
    policy: RuntimeResourcePolicy,
    *,
    primary: BaseException | None,
) -> None:
    global _ACTIVE_RUNTIME_RESOURCE_POLICY
    leaked = policy.reserved_memory_bytes
    measurement_error: BaseException | None = None
    try:
        if policy._measurement.state != "RETIRED":
            try:
                policy._measurement.quiesce()
            finally:
                if policy._measurement.state != "RETIRED":
                    policy._measurement.stop()
    except BaseException as exc:
        measurement_error = exc
    policy._ledger.state = "RETIRED"
    _ACTIVE_RUNTIME_RESOURCE_POLICY = None
    if primary is not None:
        return
    if leaked:
        raise RuntimeError("runtime_memory_reservation_leaked")
    if measurement_error is not None:
        raise measurement_error
    evidence = policy.memory_evidence()
    _require_memory_acceptance(policy._measurement, evidence)


def _require_memory_acceptance(
    measurement: _OperationRssMeasurement,
    evidence: dict[str, int | float | str],
) -> None:
    if measurement.error is not None:
        raise RuntimeError("runtime_rss_sampler_failed") from None
    if evidence["status"] != "PASS":
        raise MemoryError("runtime_rss_budget_exceeded")


def pandas_frame_payload_bytes(frame: pd.DataFrame) -> int:
    """Return the sole Runtime-approved Pandas payload diagnostic."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("invalid_pandas_frame_payload")
    return int(frame.memory_usage(index=True, deep=True).sum())


def require_free_disk_headroom(path: Path, *, reserve_bytes: int) -> int:
    """Require owner-specified free-space headroom on the target volume."""

    reserve = _exact_nonnegative_int(
        reserve_bytes,
        error="invalid_runtime_resource_disk_reserve",
    )
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    free = shutil.disk_usage(candidate.resolve()).free
    if free < reserve:
        raise RuntimeError("insufficient_free_disk")
    return free


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        available = psutil.virtual_memory().available
    except Exception:
        raise RuntimeError("runtime_available_memory_discovery_failed") from None
    return _exact_positive_int(
        available,
        error="invalid_runtime_resource_available_memory",
    )


def _process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        rss = psutil.Process().memory_info().rss
    except Exception:
        raise RuntimeError("runtime_rss_discovery_failed") from None
    return _exact_nonnegative_int(rss, error="invalid_runtime_resource_rss")


def _exact_positive_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(error)
    return value


def _exact_nonnegative_int(value: object, *, error: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(error)
    return value


__all__ = [
    "RuntimeResourcePolicy",
    "active_runtime_resource_policy",
    "pandas_frame_payload_bytes",
    "require_free_disk_headroom",
    "runtime_resource_policy_scope",
]

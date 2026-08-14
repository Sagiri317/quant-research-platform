"""Runtime-owned resource policy and payload diagnostics."""

from __future__ import annotations

import mmap
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic, sleep

import pandas as pd

MIB = 1024 * 1024
_TASK_MEMORY_PLANNING_NUMERATOR = 3
_TASK_MEMORY_PLANNING_DENOMINATOR = 4
_TASK_MEMORY_CEILING_NUMERATOR = 4
_TASK_MEMORY_CEILING_DENOMINATOR = 5
_RSS_SAMPLE_INTERVAL_SECONDS = 0.005
_RSS_SAMPLER_START_TIMEOUT_SECONDS = 5.0
_RSS_SAMPLER_STOP_TIMEOUT_SECONDS = 5.0
_RSS_STATE_BYTES = 40
_RSS_SAMPLER_CODE = """
import mmap
import pathlib
import struct
import sys
import time
import psutil

process = psutil.Process(int(sys.argv[1]))
sampler = psutil.Process()
stop = pathlib.Path(sys.argv[2])
ready = pathlib.Path(sys.argv[3])
output = pathlib.Path(sys.argv[4])
interval = float(sys.argv[5])
state = pathlib.Path(sys.argv[6])
metric = sys.argv[7]
paused = pathlib.Path(sys.argv[9])
retire = pathlib.Path(sys.argv[10])
baseline = {}
for value in sys.argv[8].split(","):
    if value:
        pid, created, amount = value.split(":", 2)
        baseline[(int(pid), float(created))] = int(amount)
peak = count = max_gap = last = 0
sequence = 0
error = ""
try:
    with state.open("r+b", buffering=0) as handle:
        shared = mmap.mmap(handle.fileno(), 40)
        with shared:
            while True:
                now = time.monotonic_ns()
                incremental = 0
                for owned in (process, *process.children(recursive=True)):
                    try:
                        if owned.pid == sampler.pid:
                            continue
                        identity = (owned.pid, owned.create_time())
                        details = (
                            owned.memory_info()
                            if metric == "private"
                            else owned.memory_full_info()
                        )
                        amount = getattr(details, metric)
                        initial = baseline.get(identity)
                        incremental += amount if initial is None else max(
                            0, amount - initial
                        )
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, incremental)
                count += 1
                if last:
                    max_gap = max(max_gap, now - last)
                last = now
                sequence += 2
                struct.pack_into("<Q", shared, 0, sequence - 1)
                struct.pack_into(
                    "<QQQQ",
                    shared,
                    8,
                    peak,
                    count,
                    max_gap,
                    0,
                )
                struct.pack_into("<Q", shared, 0, sequence)
                if count == 1:
                    ready.touch(exist_ok=True)
                if retire.exists():
                    break
                if stop.exists():
                    paused.touch(exist_ok=True)
                    while stop.exists() and not retire.exists():
                        time.sleep(interval)
                    paused.unlink(missing_ok=True)
                    if retire.exists():
                        break
                time.sleep(interval)
except BaseException as exc:
    error = type(exc).__name__
output.write_text(f"{peak},{count},{max_gap},{error}", encoding="ascii")
"""


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
    max_sample_interval_ns: int = 0
    state: str = "READY"
    error: BaseException | None = None
    _temporary: tempfile.TemporaryDirectory[str] | None = None
    _process: subprocess.Popen[bytes] | None = None
    _stop_path: Path | None = None
    _ready_path: Path | None = None
    _output_path: Path | None = None
    _state_path: Path | None = None
    _paused_path: Path | None = None
    _retire_path: Path | None = None
    _metric: str = field(init=False)
    _baseline: tuple[tuple[int, float, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        self._metric, self._baseline = _process_tree_memory_baseline()

    def start(self) -> None:
        if self.state != "READY":
            raise RuntimeError("runtime_rss_sampler_already_started")
        self.state = "ACTIVE"
        self._temporary = tempfile.TemporaryDirectory(prefix="qrp-rss-")
        root = Path(self._temporary.name)
        self._stop_path = root / "stop"
        self._ready_path = root / "ready"
        self._output_path = root / "result"
        self._state_path = root / "state"
        self._paused_path = root / "paused"
        self._retire_path = root / "retire"
        descriptor = os.open(
            self._state_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            os.ftruncate(descriptor, _RSS_STATE_BYTES)
        finally:
            os.close(descriptor)
        self._process = subprocess.Popen(
            (
                sys.executable,
                "-c",
                _RSS_SAMPLER_CODE,
                str(os.getpid()),
                str(self._stop_path),
                str(self._ready_path),
                str(self._output_path),
                str(_RSS_SAMPLE_INTERVAL_SECONDS),
                str(self._state_path),
                self._metric,
                ",".join(
                    f"{pid}:{created}:{amount}"
                    for pid, created, amount in self._baseline
                ),
                str(self._paused_path),
                str(self._retire_path),
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            ),
        )
        deadline = monotonic() + _RSS_SAMPLER_START_TIMEOUT_SECONDS
        while not self._ready_path.exists():
            if self._process.poll() is not None or monotonic() >= deadline:
                self.error = RuntimeError("runtime_rss_sampler_start_failed")
                raise self.error
            sleep(_RSS_SAMPLE_INTERVAL_SECONDS)

    def quiesce(self) -> dict[str, int | float | str]:
        """Pause after a fresh sample so a publish decision cannot race sampling."""

        if (
            self.state != "ACTIVE"
            or self._stop_path is None
            or self._paused_path is None
        ):
            raise RuntimeError("runtime_rss_sampler_not_active")
        self._stop_path.touch(exist_ok=True)
        deadline = monotonic() + _RSS_SAMPLER_STOP_TIMEOUT_SECONDS
        while not self._paused_path.exists():
            process = self._process
            if process is None or process.poll() is not None or monotonic() >= deadline:
                self.error = RuntimeError("runtime_rss_sampler_pause_failed")
                raise self.error
            sleep(_RSS_SAMPLE_INTERVAL_SECONDS)
        return self.checkpoint_evidence()

    def resume(self) -> None:
        """Resume sampling before rejection rollback and cleanup."""

        if (
            self.state != "ACTIVE"
            or self._stop_path is None
            or self._paused_path is None
            or not self._paused_path.exists()
        ):
            raise RuntimeError("runtime_rss_sampler_not_paused")
        self._stop_path.unlink(missing_ok=True)
        deadline = monotonic() + _RSS_SAMPLER_STOP_TIMEOUT_SECONDS
        while self._paused_path.exists():
            process = self._process
            if process is None or process.poll() is not None or monotonic() >= deadline:
                self.error = RuntimeError("runtime_rss_sampler_resume_failed")
                raise self.error
            sleep(_RSS_SAMPLE_INTERVAL_SECONDS)

    def checkpoint_evidence(self) -> dict[str, int | float | str]:
        if self.state != "ACTIVE" or self._state_path is None:
            raise RuntimeError("runtime_rss_sampler_not_active")
        process = self._process
        if process is None or process.poll() is not None:
            self.error = RuntimeError("runtime_rss_sampler_failed")
            raise self.error
        incremental, count, max_gap, _unused = self._snapshot()
        self.sampled_peak_rss_bytes = max(
            self.sampled_peak_rss_bytes,
            self.rss0_bytes + incremental,
        )
        self.sample_count = max(self.sample_count, count)
        self.max_sample_interval_ns = max(self.max_sample_interval_ns, max_gap)
        return self._evidence()

    def stop(self) -> None:
        if self.state == "RETIRED":
            return
        try:
            if self._retire_path is not None:
                self._retire_path.touch(exist_ok=True)
            process = self._process
            if process is None:
                raise RuntimeError("runtime_rss_sampler_stop_failed")
            try:
                process.wait(timeout=_RSS_SAMPLER_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=_RSS_SAMPLER_STOP_TIMEOUT_SECONDS)
                raise RuntimeError("runtime_rss_sampler_stop_failed") from None
            if process.returncode != 0 or self._output_path is None:
                raise RuntimeError("runtime_rss_sampler_failed")
            raw = self._output_path.read_text(encoding="ascii").split(",", 3)
            if len(raw) != 4:
                raise RuntimeError("runtime_rss_sampler_failed")
            if raw[3]:
                raise RuntimeError(f"runtime_rss_sampler_failed:{raw[3]}")
            self.sampled_peak_rss_bytes = max(
                self.sampled_peak_rss_bytes,
                self.rss0_bytes + int(raw[0]),
            )
            self.sample_count = int(raw[1])
            self.max_sample_interval_ns = int(raw[2])
        except BaseException as exc:
            self.error = exc
        finally:
            self.state = "RETIRED"
            if self._temporary is not None:
                self._temporary.cleanup()

    def _snapshot(self) -> tuple[int, int, int, int]:
        if self._state_path is None:
            raise RuntimeError("runtime_rss_sampler_not_active")
        with self._state_path.open("r+b", buffering=0) as handle:
            with mmap.mmap(
                handle.fileno(),
                _RSS_STATE_BYTES,
                access=mmap.ACCESS_READ,
            ) as shared:
                for _ in range(1000):
                    sequence = struct.unpack_from("<Q", shared, 0)[0]
                    values = struct.unpack_from("<QQQQ", shared, 8)
                    if sequence == struct.unpack_from("<Q", shared, 0)[0] and not (
                        sequence & 1
                    ):
                        return values
        raise RuntimeError("runtime_rss_sampler_snapshot_failed")

    def evidence(self) -> dict[str, int | float | str]:
        if self.state != "RETIRED":
            raise RuntimeError("runtime_rss_measurement_incomplete")
        return self._evidence()

    def _evidence(self) -> dict[str, int | float | str]:
        peak = max(self.rss0_bytes, self.sampled_peak_rss_bytes)
        incremental = peak - self.rss0_bytes
        cadence_status = "PASS" if self.error is None and self.sample_count else "FAIL"
        status = (
            "PASS"
            if self.error is None
            and incremental <= self.hard_budget_bytes
            and self.sample_count
            else "FAIL"
        )
        return {
            "rss0_bytes": self.rss0_bytes,
            "hard_budget_bytes": self.hard_budget_bytes,
            "sampled_peak_rss_bytes": peak,
            "peak_incremental_rss_bytes": incremental,
            "measurement_metric": f"process_tree_{self._metric}_incremental",
            "sample_count": self.sample_count,
            "max_sample_interval_ms": self.max_sample_interval_ns / 1_000_000,
            "cadence_status": cadence_status,
            "status": status,
        }


def _operation_rss_measurement(
    rss0_bytes: int,
    hard_budget_bytes: int,
) -> _OperationRssMeasurement:
    return _OperationRssMeasurement(rss0_bytes, hard_budget_bytes)


@dataclass(frozen=True, init=False, eq=False, slots=True)
class RuntimeResourcePolicy:
    """Immutable one-shot resource admission shared by one root operation."""

    captured_available_memory_bytes: int
    rss0_bytes: int
    task_memory_budget_bytes: int
    hard_memory_budget_bytes: int
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

        available = _exact_positive_int(
            _available_memory_bytes(),
            error="invalid_runtime_resource_available_memory",
        )
        rss0 = _exact_nonnegative_int(
            _process_rss_bytes(),
            error="invalid_runtime_resource_rss",
        )
        planning_budget = (
            available
            * _TASK_MEMORY_PLANNING_NUMERATOR
            // _TASK_MEMORY_PLANNING_DENOMINATOR
        )
        hard_budget = (
            available
            * _TASK_MEMORY_CEILING_NUMERATOR
            // _TASK_MEMORY_CEILING_DENOMINATOR
        )
        if explicit_cap_bytes is not None:
            explicit_cap = _exact_positive_int(
                explicit_cap_bytes,
                error="invalid_runtime_resource_explicit_cap",
            )
            planning_budget = min(planning_budget, explicit_cap)
            hard_budget = min(hard_budget, explicit_cap)
        if hard_budget <= 0:
            raise ValueError("invalid_runtime_resource_available_memory")
        policy = object.__new__(cls)
        object.__setattr__(policy, "captured_available_memory_bytes", available)
        object.__setattr__(policy, "rss0_bytes", rss0)
        object.__setattr__(policy, "task_memory_budget_bytes", planning_budget)
        object.__setattr__(policy, "hard_memory_budget_bytes", hard_budget)
        object.__setattr__(policy, "_ledger", _ReservationLedger())
        object.__setattr__(
            policy,
            "_measurement",
            _operation_rss_measurement(rss0, hard_budget),
        )
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
            budget_bytes=self.hard_memory_budget_bytes,
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
        }

    def memory_evidence(self) -> dict[str, int | float | str]:
        """Return final operation-scoped RSS acceptance evidence."""

        return self._measurement.evidence()

    def assert_memory_within_budget(self) -> None:
        """Reject publication when the active operation has already failed."""

        evidence = self._measurement.checkpoint_evidence()
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


def _process_tree_memory_baseline() -> tuple[
    str,
    tuple[tuple[int, float, int], ...],
]:
    try:
        import psutil  # type: ignore[import-not-found]

        parent = psutil.Process()
        parent_info = parent.memory_full_info()
        parent_memory_info = parent.memory_info()
        metric = (
            "pss"
            if hasattr(parent_info, "pss")
            else "private"
            if hasattr(parent_memory_info, "private")
            else "uss"
        )
        baseline: list[tuple[int, float, int]] = []
        for owned in (parent, *parent.children(recursive=True)):
            try:
                details = (
                    owned.memory_info()
                    if metric == "private"
                    else owned.memory_full_info()
                )
                amount = getattr(details, metric)
                baseline.append((owned.pid, owned.create_time(), int(amount)))
            except psutil.NoSuchProcess:
                pass
        return metric, tuple(sorted(baseline))
    except Exception:
        raise RuntimeError("runtime_rss_discovery_failed") from None


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

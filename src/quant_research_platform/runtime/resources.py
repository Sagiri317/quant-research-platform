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
_TASK_MEMORY_CEILING_NUMERATOR = 4
_TASK_MEMORY_CEILING_DENOMINATOR = 5
_RSS_SAMPLE_INTERVAL_SECONDS = 0.001
_MAX_RSS_SAMPLE_INTERVAL_NS = 20_000_000
_RSS_SAMPLER_START_TIMEOUT_SECONDS = 5.0
_RSS_SAMPLER_STOP_TIMEOUT_SECONDS = 5.0
_RSS_SAMPLER_CODE = """
import mmap
import pathlib
import struct
import sys
import time
import psutil

process = psutil.Process(int(sys.argv[1]))
stop = pathlib.Path(sys.argv[2])
ready = pathlib.Path(sys.argv[3])
output = pathlib.Path(sys.argv[4])
interval = float(sys.argv[5])
state = pathlib.Path(sys.argv[6])
peak = count = max_gap = last = 0
sequence = 0
error = ""
timer = None
try:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateWaitableTimerExW.restype = ctypes.c_void_p
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = (ctypes.c_void_p, ctypes.c_int)
        timer = kernel32.CreateWaitableTimerExW(None, None, 2, 0x1F0003)
        due = ctypes.c_longlong(-max(1, int(interval * 10_000_000)))
        if not timer or not kernel32.SetWaitableTimer(
            timer,
            ctypes.byref(due),
            max(1, round(interval * 1000)),
            None,
            None,
            False,
        ):
            raise OSError("runtime_rss_sampler_timer_failed")
        if not kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 2):
            raise OSError("runtime_rss_sampler_priority_failed")
    with state.open("r+b", buffering=0) as handle:
        shared = mmap.mmap(handle.fileno(), 32)
        with shared:
            while True:
                now = time.monotonic_ns()
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                count += 1
                if last:
                    max_gap = max(max_gap, now - last)
                last = now
                sequence += 2
                struct.pack_into("<Q", shared, 0, sequence - 1)
                struct.pack_into("<QQQ", shared, 8, peak, count, max_gap)
                struct.pack_into("<Q", shared, 0, sequence)
                if count == 1:
                    ready.touch(exist_ok=True)
                if stop.exists():
                    break
                if timer is None:
                    time.sleep(interval)
                elif kernel32.WaitForSingleObject(timer, 0xFFFFFFFF) != 0:
                    raise OSError("runtime_rss_sampler_timer_failed")
except BaseException as exc:
    error = type(exc).__name__
finally:
    if timer is not None:
        kernel32.CloseHandle(timer)
output.write_text(f"{peak},{count},{max_gap},{error}", encoding="ascii")
"""


@dataclass(slots=True)
class _ReservationLedger:
    state: str = "READY"
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
        descriptor = os.open(
            self._state_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            os.ftruncate(descriptor, 32)
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
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "HIGH_PRIORITY_CLASS", 0)
            ),
        )
        deadline = monotonic() + _RSS_SAMPLER_START_TIMEOUT_SECONDS
        while not self._ready_path.exists():
            if self._process.poll() is not None or monotonic() >= deadline:
                self.error = RuntimeError("runtime_rss_sampler_start_failed")
                raise self.error
            sleep(_RSS_SAMPLE_INTERVAL_SECONDS)

    def checkpoint_evidence(self) -> dict[str, int | float | str]:
        if self.state != "ACTIVE" or self._state_path is None:
            raise RuntimeError("runtime_rss_sampler_not_active")
        process = self._process
        if process is None or process.poll() is not None:
            self.error = RuntimeError("runtime_rss_sampler_failed")
            raise self.error
        with self._state_path.open("r+b", buffering=0) as handle:
            with mmap.mmap(handle.fileno(), 32, access=mmap.ACCESS_READ) as shared:
                for _ in range(1000):
                    sequence = struct.unpack_from("<Q", shared, 0)[0]
                    peak, count, max_gap = struct.unpack_from("<QQQ", shared, 8)
                    if sequence == struct.unpack_from("<Q", shared, 0)[0] and not (
                        sequence & 1
                    ):
                        break
                else:
                    raise RuntimeError("runtime_rss_sampler_snapshot_failed")
        self.sampled_peak_rss_bytes = max(self.sampled_peak_rss_bytes, peak)
        self.sample_count = max(self.sample_count, count)
        self.max_sample_interval_ns = max(self.max_sample_interval_ns, max_gap)
        return self._evidence()

    def stop(self) -> None:
        if self.state == "RETIRED":
            return
        try:
            if self._stop_path is not None:
                self._stop_path.touch(exist_ok=True)
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
            if len(raw) != 4 or raw[3]:
                raise RuntimeError("runtime_rss_sampler_failed")
            self.sampled_peak_rss_bytes = int(raw[0])
            self.sample_count = int(raw[1])
            self.max_sample_interval_ns = int(raw[2])
        except BaseException as exc:
            self.error = exc
        finally:
            self.state = "RETIRED"
            if self._temporary is not None:
                self._temporary.cleanup()

    def evidence(self) -> dict[str, int | float | str]:
        if self.state != "RETIRED":
            raise RuntimeError("runtime_rss_measurement_incomplete")
        return self._evidence()

    def _evidence(self) -> dict[str, int | float | str]:
        peak = max(self.rss0_bytes, self.sampled_peak_rss_bytes)
        incremental = peak - self.rss0_bytes
        cadence_status = (
            "PASS"
            if self.error is None
            and self.max_sample_interval_ns <= _MAX_RSS_SAMPLE_INTERVAL_NS
            else "FAIL"
        )
        status = (
            "PASS"
            if self.error is None
            and incremental <= self.hard_budget_bytes
            and cadence_status == "PASS"
            else "FAIL"
        )
        return {
            "rss0_bytes": self.rss0_bytes,
            "hard_budget_bytes": self.hard_budget_bytes,
            "sampled_peak_rss_bytes": peak,
            "peak_incremental_rss_bytes": incremental,
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
        hard_budget = (
            available
            * _TASK_MEMORY_CEILING_NUMERATOR
            // _TASK_MEMORY_CEILING_DENOMINATOR
        )
        if explicit_cap_bytes is not None:
            hard_budget = min(
                hard_budget,
                _exact_positive_int(
                    explicit_cap_bytes,
                    error="invalid_runtime_resource_explicit_cap",
                ),
            )
        if hard_budget <= 0:
            raise ValueError("invalid_runtime_resource_available_memory")
        policy = object.__new__(cls)
        object.__setattr__(policy, "captured_available_memory_bytes", available)
        object.__setattr__(policy, "rss0_bytes", rss0)
        object.__setattr__(policy, "task_memory_budget_bytes", hard_budget)
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
        return self.task_memory_budget_bytes - self.reserved_memory_bytes

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
            self.assert_memory_within_budget()
            self._measurement.stop()
            evidence = self.memory_evidence()
            _require_memory_acceptance(self._measurement, evidence)
            return evidence
        except BaseException as primary:
            if on_reject is not None:
                try:
                    on_reject()
                except BaseException as rollback_error:
                    primary.add_note(
                        "runtime memory rejection rollback also failed: "
                        f"{type(rollback_error).__name__}"
                    )
            if self._measurement.state != "RETIRED":
                self._measurement.stop()
            raise

_ACTIVE_RUNTIME_RESOURCE_POLICY: RuntimeResourcePolicy | None = None


def current_runtime_resource_policy() -> RuntimeResourcePolicy | None:
    """Return the active policy without admitting a new root operation."""

    return _ACTIVE_RUNTIME_RESOURCE_POLICY


def bind_runtime_resource_policy(
    explicit: RuntimeResourcePolicy | None,
) -> RuntimeResourcePolicy | None:
    """Bind a lazy operation to its exact creation-time policy."""

    if explicit is not None and type(explicit) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    active = current_runtime_resource_policy()
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
        yield policy
        return
    if policy._ledger.state == "RETIRED":
        raise RuntimeError("runtime_resource_policy_retired")
    if policy._ledger.state != "READY":
        raise RuntimeError("runtime_resource_policy_already_active")
    policy._ledger.state = "ACTIVE"
    _ACTIVE_RUNTIME_RESOURCE_POLICY = policy
    try:
        policy._measurement.start()
        yield policy
    finally:
        primary = sys.exception()
        leaked = policy.reserved_memory_bytes
        measurement_error: BaseException | None = None
        try:
            if policy._measurement.state != "RETIRED":
                policy._measurement.stop()
        except BaseException as exc:
            measurement_error = exc
        policy._ledger.state = "RETIRED"
        _ACTIVE_RUNTIME_RESOURCE_POLICY = None
        if primary is None:
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
    if evidence["cadence_status"] != "PASS":
        raise RuntimeError(
            "runtime_rss_sampling_cadence_exceeded: "
            f"{evidence['max_sample_interval_ms']:.6f}ms"
        )
    if evidence["status"] != "PASS":
        raise MemoryError("runtime_rss_budget_exceeded")


def pandas_frame_payload_bytes(frame: pd.DataFrame) -> int:
    """Return the sole Runtime-approved Pandas payload diagnostic."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("invalid_pandas_frame_payload")
    return int(frame.memory_usage(index=True, deep=True).sum())


def require_free_disk_headroom(path: Path, *, reserve_bytes: int) -> None:
    """Require owner-specified free-space headroom on the target volume."""

    reserve = _exact_nonnegative_int(
        reserve_bytes,
        error="invalid_runtime_resource_disk_reserve",
    )
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if shutil.disk_usage(candidate.resolve()).free < reserve:
        raise RuntimeError("insufficient_free_disk")


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

from __future__ import annotations

import json
import mmap
import os
import subprocess
import sys
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from tests.helpers.runtime_resources import capture_test_policy

MIB = 1024 * 1024
GIB = 1024 * MIB


class _PreparedMeasurement:
    error = None
    state = "READY"

    def prepare(self) -> None:
        pass

    def bind_baseline(self, rss0_bytes: int, hard_budget_bytes: int) -> None:
        self.rss0_bytes = rss0_bytes
        self.hard_budget_bytes = hard_budget_bytes


def test_runtime_resource_policy_direct_construction_fails_closed() -> None:
    with pytest.raises(TypeError, match="runtime_resource_policy_capture_required"):
        RuntimeResourcePolicy()


def test_runtime_resource_policy_captures_one_shot_budget_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {"available": 0, "rss": 0}

    def available() -> int:
        observations["available"] += 1
        return 10 * GIB

    def rss() -> int:
        observations["rss"] += 1
        return 512 * MIB

    monkeypatch.setattr(resources, "_available_memory_bytes", available)
    monkeypatch.setattr(resources, "_process_rss_bytes", rss)

    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=2 * GIB)

    assert observations == {"available": 1, "rss": 1}
    assert policy.to_payload() == {
        "captured_available_memory_bytes": 10 * GIB,
        "rss0_bytes": 512 * MIB,
        "task_memory_budget_bytes": 15 * GIB // 8,
        "hard_memory_budget_bytes": 2 * GIB,
    }
    assert policy.remaining_memory_bytes == 15 * GIB // 8


def test_sampler_stabilizes_before_one_shot_budget_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Measurement:
        state = "READY"
        error = None

        def prepare(self) -> None:
            events.append("sampler_ready")

        def bind_baseline(self, rss0_bytes: int, hard_budget_bytes: int) -> None:
            assert events == ["sampler_ready", "available0", "rss0"]
            assert (rss0_bytes, hard_budget_bytes) == (100, 600)

        def stop(self) -> None:
            self.state = "RETIRED"

    measurement = Measurement()
    monkeypatch.setattr(
        resources,
        "_operation_rss_measurement",
        lambda *_args: measurement,
    )
    monkeypatch.setattr(
        resources,
        "_available_memory_bytes",
        lambda: events.append("available0") or 1000,
    )
    monkeypatch.setattr(
        resources,
        "_process_rss_bytes",
        lambda: events.append("rss0") or 100,
    )

    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=600)

    assert policy._measurement is measurement  # noqa: SLF001


def test_runtime_resource_policy_rejects_measurement_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = {"available": 0, "rss": 0}

    def available() -> int:
        observations["available"] += 1
        return 10 * GIB

    def rss() -> int:
        observations["rss"] += 1
        return GIB

    monkeypatch.setattr(resources, "_available_memory_bytes", available)
    monkeypatch.setattr(resources, "_process_rss_bytes", rss)

    RuntimeResourcePolicy.capture()

    assert observations == {"available": 1, "rss": 1}
    with pytest.raises(TypeError):
        RuntimeResourcePolicy.capture(available_memory_bytes=10**18)
    with pytest.raises(TypeError):
        RuntimeResourcePolicy.capture(rss_bytes=0)


def test_explicit_cap_can_only_lower_eighty_percent_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 10 * GIB)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: GIB)

    uncapped = RuntimeResourcePolicy.capture()
    high_cap = RuntimeResourcePolicy.capture(explicit_cap_bytes=20 * GIB)
    low_cap = RuntimeResourcePolicy.capture(explicit_cap_bytes=2 * GIB)

    assert uncapped.task_memory_budget_bytes == 15 * GIB // 2
    assert uncapped.hard_memory_budget_bytes == 8 * GIB
    assert high_cap.task_memory_budget_bytes == 15 * GIB // 2
    assert high_cap.hard_memory_budget_bytes == 8 * GIB
    assert low_cap.task_memory_budget_bytes == 15 * GIB // 8
    assert low_cap.hard_memory_budget_bytes == 2 * GIB


def test_planning_target_keeps_margin_below_hard_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 1000)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: 100)
    monkeypatch.setattr(
        resources,
        "_operation_rss_measurement",
        lambda *_args: resources._OperationRssMeasurement(100, 800),
    )

    policy = RuntimeResourcePolicy.capture()

    assert policy.task_memory_budget_bytes == 750
    assert policy.hard_memory_budget_bytes == 800
    assert policy.remaining_memory_bytes == 750


@pytest.mark.parametrize(
    ("probe", "value", "error"),
    [
        ("_available_memory_bytes", True, "invalid_runtime_resource_available_memory"),
        ("_available_memory_bytes", 0, "invalid_runtime_resource_available_memory"),
        ("_available_memory_bytes", 1.0, "invalid_runtime_resource_available_memory"),
        ("_available_memory_bytes", "1", "invalid_runtime_resource_available_memory"),
        (
            "_available_memory_bytes",
            Decimal(1),
            "invalid_runtime_resource_available_memory",
        ),
        (
            "_available_memory_bytes",
            Fraction(1, 1),
            "invalid_runtime_resource_available_memory",
        ),
        ("_process_rss_bytes", -1, "invalid_runtime_resource_rss"),
        ("_process_rss_bytes", False, "invalid_runtime_resource_rss"),
    ],
)
def test_capture_requires_exact_probe_results(
    monkeypatch: pytest.MonkeyPatch,
    probe: str,
    value: object,
    error: str,
) -> None:
    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 100)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: 10)
    monkeypatch.setattr(resources, probe, lambda: value)

    with pytest.raises(ValueError, match=error):
        RuntimeResourcePolicy.capture()


@pytest.mark.parametrize("value", (0, 1.0))
def test_capture_requires_exact_positive_explicit_cap(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_explicit_cap"):
        capture_test_policy(
            available_memory_bytes=100,
            rss_bytes=10,
            explicit_cap_bytes=value,  # type: ignore[arg-type]
        )


def test_runtime_resource_policy_uses_one_shared_reservation_ledger() -> None:
    policy = capture_test_policy(
        available_memory_bytes=1000,
        rss_bytes=100,
        explicit_cap_bytes=600,
    )

    with runtime_resource_policy_scope(policy):
        with policy.reserve("storage", 400):
            assert policy.reserved_memory_bytes == 400
            assert policy.remaining_memory_bytes == 163
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                with policy.reserve("research", 164):
                    pass
        assert policy.reserved_memory_bytes == 0
        assert policy.remaining_memory_bytes == 563


def test_same_policy_scopes_can_close_non_lifo_without_retiring_root() -> None:
    policy = capture_test_policy(
        available_memory_bytes=1000,
        rss_bytes=100,
    )
    outer = runtime_resource_policy_scope(policy)
    inner = runtime_resource_policy_scope(policy)
    outer.__enter__()
    inner.__enter__()

    outer.__exit__(None, None, None)
    assert resources.current_runtime_resource_policy() is policy
    assert policy._measurement.state == "ACTIVE"  # noqa: SLF001

    inner.__exit__(None, None, None)
    assert resources.current_runtime_resource_policy() is None
    assert policy._measurement.state == "RETIRED"  # noqa: SLF001


def test_root_scope_records_one_operation_peak_and_fails_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Measurement(_PreparedMeasurement):
        def start(self) -> None:
            self.state = "ACTIVE"

        def stop(self) -> None:
            events.append("stop")
            self.state = "RETIRED"

        def quiesce(self) -> dict[str, int | float | str]:
            events.append("quiesce")
            return self.evidence()

        def evidence(self) -> dict[str, int | float | str]:
            return {
                "rss0_bytes": 100,
                "hard_budget_bytes": 600,
                "sampled_peak_rss_bytes": 701,
                "peak_incremental_rss_bytes": 601,
                "sample_count": 2,
                "max_sample_interval_ms": 1.0,
                "cadence_status": "PASS",
                "status": "FAIL",
            }

    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 1000)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: 100)
    monkeypatch.setattr(
        resources,
        "_operation_rss_measurement",
        lambda *_args: Measurement(),
    )
    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=600)

    with pytest.raises(MemoryError, match="runtime_rss_budget_exceeded"):
        with runtime_resource_policy_scope(policy):
            with runtime_resource_policy_scope(policy):
                pass

    evidence = policy.memory_evidence()
    assert evidence["rss0_bytes"] == 100
    assert evidence["hard_budget_bytes"] == 600
    assert evidence["sampled_peak_rss_bytes"] == 701
    assert evidence["peak_incremental_rss_bytes"] == 601
    assert evidence["sample_count"] >= 2
    assert evidence["cadence_status"] == "PASS"
    assert evidence["status"] == "FAIL"
    assert events == ["quiesce", "stop"]


def test_finish_quiesces_before_retiring_the_operation_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Measurement(_PreparedMeasurement):
        def start(self) -> None:
            self.state = "ACTIVE"

        def stop(self) -> None:
            events.append("stop_sampler")
            self.state = "RETIRED"

        def quiesce(self) -> dict[str, int | float | str]:
            events.append("quiesce")
            return self.evidence()

        def resume(self) -> None:
            events.append("resume")

        def evidence(self) -> dict[str, int | float | str]:
            return {
                "rss0_bytes": 100,
                "hard_budget_bytes": 600,
                "sampled_peak_rss_bytes": 100,
                "peak_incremental_rss_bytes": 0,
                "sample_count": 2,
                "max_sample_interval_ms": 1.0,
                "cadence_status": "PASS",
                "status": "PASS",
            }

    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 1000)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: 100)
    monkeypatch.setattr(
        resources,
        "_operation_rss_measurement",
        lambda *_args: Measurement(),
    )
    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=600)

    with runtime_resource_policy_scope(policy):
        policy.finish_memory_measurement()
        assert events == ["quiesce", "stop_sampler"]

    assert events == ["quiesce", "stop_sampler"]


def test_final_rejection_samples_through_rollback_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Measurement(_PreparedMeasurement):
        def start(self) -> None:
            self.state = "ACTIVE"

        def quiesce(self) -> dict[str, int | float | str]:
            events.append("quiesce")
            return self.evidence()

        def resume(self) -> None:
            assert self.state == "ACTIVE"
            events.append("resume")

        def stop(self) -> None:
            events.append("stop_sampler")
            self.state = "RETIRED"

        def evidence(self) -> dict[str, int | float | str]:
            return {
                "rss0_bytes": 100,
                "hard_budget_bytes": 600,
                "sampled_peak_rss_bytes": 701,
                "peak_incremental_rss_bytes": 601,
                "sample_count": 2,
                "max_sample_interval_ms": 1.0,
                "cadence_status": "PASS",
                "status": "FAIL",
            }

    measurement = Measurement()
    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 1000)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: 100)
    monkeypatch.setattr(
        resources,
        "_operation_rss_measurement",
        lambda *_args: measurement,
    )
    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=600)

    def rollback() -> None:
        assert measurement.state == "ACTIVE"
        events.append("rollback_cleanup")

    with pytest.raises(MemoryError, match="runtime_rss_budget_exceeded"):
        with runtime_resource_policy_scope(policy):
            policy.finish_memory_measurement(on_reject=rollback)

    assert events == [
        "quiesce",
        "resume",
        "rollback_cleanup",
        "quiesce",
        "stop_sampler",
    ]


def test_sampler_stop_failure_still_rolls_back_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Measurement(_PreparedMeasurement):
        def start(self) -> None:
            self.state = "ACTIVE"

        def quiesce(self) -> dict[str, int | float | str]:
            events.append("quiesce")
            return self.evidence()

        def stop(self) -> None:
            events.append("stop_sampler")
            self.state = "RETIRED"
            self.error = OSError("sampler stop failed")

        def evidence(self) -> dict[str, int | float | str]:
            return {
                "rss0_bytes": 100,
                "hard_budget_bytes": 600,
                "sampled_peak_rss_bytes": 100,
                "peak_incremental_rss_bytes": 0,
                "sample_count": 2,
                "max_sample_interval_ms": 1.0,
                "cadence_status": "PASS",
                "status": "PASS",
            }

    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 1000)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: 100)
    monkeypatch.setattr(
        resources,
        "_operation_rss_measurement",
        lambda *_args: Measurement(),
    )
    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=600)

    with pytest.raises(RuntimeError, match="runtime_rss_sampler_failed"):
        with runtime_resource_policy_scope(policy):
            policy.finish_memory_measurement(
                on_reject=lambda: events.append("rollback_cleanup")
            )

    assert events == ["quiesce", "stop_sampler", "rollback_cleanup"]


def test_operation_rss_evidence_excludes_unowned_child_processes() -> None:
    code = """
import json
import subprocess
import sys
import time
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
budget = 256 * 1024 * 1024
policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=budget)
with runtime_resource_policy_scope(policy):
    child = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "import time; payload=bytearray(32*1024*1024); "
            "print('ready', flush=True); time.sleep(0.25)",
        ),
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "ready"
    time.sleep(0.05)
    child.wait(timeout=5)
print(json.dumps(policy.memory_evidence()))
"""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    evidence = json.loads(completed.stdout)

    assert evidence["status"] == "PASS"
    assert evidence["peak_incremental_rss_bytes"] < 16 * MIB
    assert evidence["peak_incremental_rss_bytes"] <= evidence["hard_budget_bytes"]
    assert evidence["measurement_metric"] == "process_rss_incremental"
    assert evidence["max_sample_interval_ms"] >= 0


def test_quiesce_acknowledges_a_post_request_sample() -> None:
    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=64 * MIB)

    with runtime_resource_policy_scope(policy):
        before = policy._measurement.checkpoint_evidence()["sample_count"]  # noqa: SLF001
        after = policy._measurement.quiesce()  # noqa: SLF001
        assert after["sample_count"] > before
        policy._measurement.resume()  # noqa: SLF001


@pytest.mark.skipif(sys.platform != "win32", reason="Windows resident metric")
def test_windows_metric_counts_resident_not_uncommitted_mapping() -> None:
    def measure(*, touch: bool) -> dict[str, object]:
        code = f"""
import json
import mmap
import time
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
size = 64 * 1024 * 1024
policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=128 * 1024 * 1024)
with runtime_resource_policy_scope(policy):
    payload = mmap.mmap(-1, size, access=mmap.ACCESS_COPY)
    if {touch!r}:
        for offset in range(0, size, 4096):
            payload[offset] = 1
    time.sleep(0.05)
print(json.dumps(policy.memory_evidence()))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        completed = subprocess.run(
            (sys.executable, "-c", code),
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        return json.loads(completed.stdout)

    untouched = measure(touch=False)
    touched = measure(touch=True)

    assert untouched["measurement_metric"] == "process_rss_incremental"
    assert untouched["peak_incremental_rss_bytes"] < 16 * MIB
    assert touched["peak_incremental_rss_bytes"] >= 48 * MIB


def test_low_cap_empty_operation_does_not_charge_sampler_process() -> None:
    budget = 16 * MIB
    code = """
import json
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
budget = 16 * 1024 * 1024
policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=budget)
with runtime_resource_policy_scope(policy):
    pass
print(json.dumps(policy.memory_evidence()))
"""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    completed = subprocess.run(
        (sys.executable, "-c", code),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    evidence = json.loads(completed.stdout)

    assert evidence["status"] == "PASS"
    assert evidence["peak_incremental_rss_bytes"] <= budget


@pytest.mark.parametrize(
    ("interval_ns", "expected_status"),
    ((20_000_000, "PASS"), (20_000_001, "FAIL")),
)
def test_sampler_cadence_is_an_authoritative_memory_gate(
    interval_ns: int,
    expected_status: str,
) -> None:
    measurement = resources._OperationRssMeasurement(100, 600)
    measurement.state = "RETIRED"
    measurement.sampled_peak_rss_bytes = 100
    measurement.sample_count = 2
    measurement.max_sample_interval_ns = interval_ns

    evidence = measurement.evidence()

    assert evidence["cadence_status"] == expected_status
    assert evidence["status"] == expected_status


def test_sampler_arms_after_prebaseline_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_available = resources._available_memory_bytes

    def available_after_transient() -> int:
        with mmap.mmap(-1, 64 * MIB, access=mmap.ACCESS_COPY) as transient:
            for offset in range(0, 64 * MIB, 4096):
                transient[offset] = 1
            time.sleep(0.02)
        return original_available()

    monkeypatch.setattr(
        resources,
        "_available_memory_bytes",
        available_after_transient,
    )
    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=32 * MIB)

    with runtime_resource_policy_scope(policy):
        pass

    assert policy.memory_evidence()["status"] == "PASS"


def test_active_primary_is_not_replaced_by_sampler_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Measurement(_PreparedMeasurement):
        def start(self) -> None:
            self.state = "ACTIVE"

        def stop(self) -> None:
            self.state = "RETIRED"
            self.error = OSError("sampler stop failed")

        def quiesce(self) -> dict[str, int | float | str]:
            return self.evidence()

        def evidence(self) -> dict[str, int | float | str]:
            return {
                "rss0_bytes": self.rss0_bytes,
                "hard_budget_bytes": self.hard_budget_bytes,
                "sampled_peak_rss_bytes": (
                    self.rss0_bytes + self.hard_budget_bytes + 1
                ),
                "peak_incremental_rss_bytes": self.hard_budget_bytes + 1,
                "sample_count": 2,
                "max_sample_interval_ms": 1.0,
                "cadence_status": "PASS",
                "status": "FAIL",
            }

    monkeypatch.setattr(resources, "_available_memory_bytes", lambda: 1000)
    monkeypatch.setattr(resources, "_process_rss_bytes", lambda: 100)
    monkeypatch.setattr(
        resources,
        "_operation_rss_measurement",
        lambda *_args: Measurement(),
    )
    policy = RuntimeResourcePolicy.capture(explicit_cap_bytes=600)

    with pytest.raises(KeyboardInterrupt):
        with runtime_resource_policy_scope(policy):
            raise KeyboardInterrupt()


def test_reservation_release_preserves_active_primary() -> None:
    policy = capture_test_policy(
        available_memory_bytes=1000,
        rss_bytes=100,
    )

    with pytest.raises(KeyboardInterrupt):
        with runtime_resource_policy_scope(policy):
            with policy.reserve("storage", 100):
                raise KeyboardInterrupt()

    assert policy.reserved_memory_bytes == 0


@pytest.mark.parametrize("owner", ["", None, 1])
def test_reservation_requires_nonempty_string_owner(owner: object) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1000,
        rss_bytes=100,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError,
            match="invalid_runtime_memory_reservation_owner",
        ):
            with policy.reserve(owner, 1):  # type: ignore[arg-type]
                pass


def test_reservation_requires_active_policy() -> None:
    policy = capture_test_policy(
        available_memory_bytes=1000,
        rss_bytes=100,
    )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        with policy.reserve("storage", 1):
            pass

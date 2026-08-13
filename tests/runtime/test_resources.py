from __future__ import annotations

import json
import os
import subprocess
import sys
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
        "task_memory_budget_bytes": 2 * GIB,
    }
    assert not hasattr(policy, "payload_target_bytes")
    assert policy.remaining_memory_bytes == 2 * GIB


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

    assert uncapped.task_memory_budget_bytes == 8 * GIB
    assert high_cap.task_memory_budget_bytes == 8 * GIB


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
            assert policy.remaining_memory_bytes == 200
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                with policy.reserve("research", 201):
                    pass
        assert policy.reserved_memory_bytes == 0
        assert policy.remaining_memory_bytes == 600


def test_root_scope_records_one_operation_peak_and_fails_over_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Measurement:
        error = None
        state = "READY"

        def start(self) -> None:
            self.state = "ACTIVE"

        def stop(self) -> None:
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


def test_finish_accepts_before_retiring_the_operation_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Measurement:
        error = None
        state = "READY"

        def start(self) -> None:
            self.state = "ACTIVE"

        def stop(self) -> None:
            events.append("stop_sampler")
            self.state = "RETIRED"

        def checkpoint_evidence(self) -> dict[str, int | float | str]:
            events.append("checkpoint")
            return self.evidence()

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
        assert events == ["checkpoint", "stop_sampler"]

    assert events == ["checkpoint", "stop_sampler"]


def test_operation_rss_evidence_includes_child_processes() -> None:
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
    assert evidence["peak_incremental_rss_bytes"] >= 32 * MIB
    assert evidence["peak_incremental_rss_bytes"] <= evidence["hard_budget_bytes"]
    assert evidence["max_sample_interval_ms"] <= 20


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

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from tests.helpers.runtime_resources import capture_test_policy

MIB = 1024 * 1024
GIB = 1024 * MIB


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
        "payload_target_bytes": 256 * MIB,
    }


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
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                with policy.reserve("research", 201):
                    pass
        assert policy.reserved_memory_bytes == 0


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

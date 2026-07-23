from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    runtime_resource_policy_scope,
)

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def test_runtime_resource_policy_defaults_to_two_resolved_quantities() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={},
    )

    assert policy.task_memory_budget_bytes == 8 * GIB
    assert policy.payload_target_bytes == 8 * GIB
    assert policy.to_payload() == {
        "task_memory_budget_bytes": 8 * GIB,
        "payload_target_bytes": 8 * GIB,
    }


def test_runtime_resource_policy_resolves_current_overrides() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": "1024",
            "QRP_PAYLOAD_TARGET_MB": "256",
        },
    )

    assert policy.task_memory_budget_bytes == 1024 * MIB
    assert policy.payload_target_bytes == 256 * MIB


def test_task_override_is_clamped_to_eighty_percent_ceiling() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "16384"},
    )

    assert policy.task_memory_budget_bytes == 8 * GIB
    assert policy.payload_target_bytes == 8 * GIB


def test_payload_target_cannot_exceed_resolved_task_budget() -> None:
    with pytest.raises(
        ValueError,
        match="runtime_payload_target_exceeds_task_memory_budget",
    ):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=10 * GIB,
            environ={
                "QRP_TASK_MEMORY_BUDGET_MB": "64",
                "QRP_PAYLOAD_TARGET_MB": "65",
            },
        )


@pytest.mark.parametrize(
    "name",
    [
        "QRP_MEMORY_BUDGET_FRACTION",
        "QRP_MEMORY_BUDGET_MB",
        "QRP_CHUNK_TARGET_MB",
        "QRP_MEMORY_SAFETY_FACTOR",
        "QRP_MIN_FREE_DISK_MB",
    ],
)
@pytest.mark.parametrize("value", ["", "1"])
def test_retired_runtime_resource_settings_fail_closed(
    name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="retired_runtime_resource_setting"):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=GIB,
            environ={name: value},
        )


@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, "1", Decimal(1), Fraction(1, 1)],
)
def test_available_memory_requires_exact_int(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="invalid_runtime_resource_available_memory",
    ):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=value,  # type: ignore[arg-type]
            environ={},
        )


@pytest.mark.parametrize("value", ["", "0", "-1", "+1", "1.0", "1e3", " 1", "１"])
@pytest.mark.parametrize(
    "name",
    ["QRP_TASK_MEMORY_BUDGET_MB", "QRP_PAYLOAD_TARGET_MB"],
)
def test_current_environment_values_require_exact_positive_integer_megabytes(
    name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_env"):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=GIB,
            environ={name: value},
        )


def test_direct_construction_is_unavailable() -> None:
    with pytest.raises(TypeError, match="runtime_resource_policy_factory_required"):
        RuntimeResourcePolicy(
            task_memory_budget_bytes=1024,
            payload_target_bytes=512,
        )


def test_class_exposes_no_resolved_value_issuance_bypass() -> None:
    assert not hasattr(RuntimeResourcePolicy, "_create")


def test_runtime_resource_policy_is_sealed() -> None:
    with pytest.raises(TypeError, match="runtime_resource_policy_subclass_forbidden"):

        class ForgedPolicy(RuntimeResourcePolicy):
            pass


def test_scope_rejects_forged_exact_instance() -> None:
    forged = object.__new__(RuntimeResourcePolicy)

    with pytest.raises(TypeError, match="invalid_runtime_resource_policy"):
        with runtime_resource_policy_scope(forged):
            pass


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_memory_budget_bytes", 10 * GIB),
        ("payload_target_bytes", 1),
    ],
)
def test_factory_issued_policy_fields_reject_mutation(
    field: str,
    value: int,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={},
    )
    expected = policy.to_payload()

    with pytest.raises(AttributeError):
        object.__setattr__(policy, field, value)

    assert policy.to_payload() == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_memory_budget_bytes", 10 * GIB),
        ("payload_target_bytes", 1),
    ],
)
def test_active_policy_fields_reject_post_entry_mutation(
    field: str,
    value: int,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={},
    )
    expected = policy.to_payload()

    with runtime_resource_policy_scope(policy):
        with pytest.raises(AttributeError):
            object.__setattr__(policy, field, value)
        assert active_runtime_resource_policy() is policy
        assert active_runtime_resource_policy().to_payload() == expected


def test_runtime_resource_policy_scope_restores_active_snapshot() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={},
    )

    with runtime_resource_policy_scope(policy):
        assert active_runtime_resource_policy() is policy

    assert active_runtime_resource_policy() is not policy


def test_scope_revalidates_active_policy_and_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={},
    )
    validate = resources._issued_policy_values
    validated: list[object] = []

    def record_validation(candidate: object) -> tuple[int, int]:
        validated.append(candidate)
        return validate(candidate)

    monkeypatch.setattr(resources, "_issued_policy_values", record_validation)

    with runtime_resource_policy_scope(policy):
        assert active_runtime_resource_policy() is policy

    assert len(validated) == 2
    assert all(candidate is policy for candidate in validated)


def test_scope_exit_does_not_revalidate_and_resets_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={},
    )
    validate = resources._issued_policy_values
    calls = 0

    def reject_redundant_validation(candidate: object) -> tuple[int, int]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("runtime_policy_revalidated_on_exit")
        return validate(candidate)

    monkeypatch.setattr(resources, "_issued_policy_values", reject_redundant_validation)

    with runtime_resource_policy_scope(policy):
        pass

    assert resources._ACTIVE_RUNTIME_RESOURCE_POLICY.get() is None
    assert calls == 1

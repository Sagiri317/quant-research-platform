from __future__ import annotations

import math
from pathlib import Path

import pytest

from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlan,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    runtime_resource_policy_scope,
)

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def test_runtime_resource_policy_uses_required_defaults() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={},
    )

    assert policy.memory_budget_fraction == 0.80
    assert policy.chunk_target_fraction == 0.30
    assert policy.chunk_target_cap_bytes == 2 * GIB
    assert policy.memory_safety_factor == 0.60
    assert policy.memory_budget_bytes == 8 * GIB
    assert policy.chunk_target_bytes == 2 * GIB
    assert policy.effective_chunk_bytes == int(2 * GIB * 0.60)


def test_runtime_resource_policy_private_payload_keeps_exact_bytes() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={"QRP_MEMORY_BUDGET_MB": "1024"},
    )

    payload = policy.to_payload()

    assert payload == {
        "available_memory_bytes": 10 * GIB,
        "memory_budget_fraction": 0.80,
        "memory_budget_bytes": 1024 * MIB,
        "memory_budget_bytes_override": 1024 * MIB,
        "chunk_target_fraction": 0.30,
        "chunk_target_cap_bytes": 2 * GIB,
        "chunk_target_bytes": int(1024 * MIB * 0.30),
        "effective_chunk_bytes": int(1024 * MIB * 0.30 * 0.60),
        "memory_safety_factor": 0.60,
        "min_free_disk_bytes": 0,
    }
    assert all("path" not in key for key in payload)


def test_runtime_resource_policy_public_payload_uses_coarse_buckets() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={"QRP_MEMORY_BUDGET_MB": "1024"},
    )

    payload = policy.to_public_summary_payload()

    assert payload == {
        "memory_budget_fraction": 0.80,
        "memory_safety_factor": 0.60,
        "available_memory_bucket": "8-16GiB",
        "memory_budget_bucket": "1-2GiB",
        "chunk_target_bucket": "256-512MiB",
        "effective_chunk_bucket": "128-256MiB",
        "has_memory_budget_override": True,
        "min_free_disk_bucket": "0",
    }
    assert not any(key.endswith("_bytes") for key in payload)
    assert 10 * GIB not in payload.values()


def test_runtime_resource_policy_scope_restores_active_snapshot() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={"QRP_MEMORY_BUDGET_MB": "1024"},
    )

    with runtime_resource_policy_scope(policy):
        assert active_runtime_resource_policy() is policy

    assert active_runtime_resource_policy() is not policy


def test_memory_budget_override_drives_derived_chunk_target() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=8 * GIB,
        environ={"QRP_MEMORY_BUDGET_MB": "1024"},
    )

    assert policy.memory_budget_bytes == 1024 * MIB
    assert policy.chunk_target_bytes == int(1024 * MIB * 0.30)
    assert policy.effective_chunk_bytes == int(1024 * MIB * 0.30 * 0.60)


def test_chunk_target_override_drives_effective_chunk_bytes() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=8 * GIB,
        environ={
            "QRP_MEMORY_BUDGET_MB": "1024",
            "QRP_CHUNK_TARGET_MB": "256",
            "QRP_MEMORY_SAFETY_FACTOR": "0.5",
            "QRP_MIN_FREE_DISK_MB": "128",
        },
    )

    assert policy.memory_budget_bytes == 1024 * MIB
    assert policy.chunk_target_bytes == 256 * MIB
    assert policy.effective_chunk_bytes == 128 * MIB
    assert policy.min_free_disk_bytes == 128 * MIB


def test_effective_chunk_bytes_cannot_exceed_memory_budget() -> None:
    with pytest.raises(ValueError, match="chunk_target_exceeds_memory_budget"):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=8 * GIB,
            environ={
                "QRP_MEMORY_BUDGET_MB": "64",
                "QRP_CHUNK_TARGET_MB": "256",
            },
        )


def test_invalid_env_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_env"):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=8 * GIB,
            environ={"QRP_MEMORY_BUDGET_MB": "0"},
        )
    with pytest.raises(ValueError, match="invalid_runtime_resource_env"):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=8 * GIB,
            environ={"QRP_CHUNK_TARGET_MB": "-1"},
        )


@pytest.mark.parametrize(
    "environ",
    [
        {"QRP_MEMORY_BUDGET_FRACTION": "1.1"},
        {"QRP_MEMORY_SAFETY_FACTOR": "1.1"},
    ],
)
def test_runtime_resource_policy_rejects_fraction_env_values_above_one(
    environ: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_env"):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=8 * GIB,
            environ=environ,
        )


def test_runtime_resource_policy_rejects_invalid_available_memory() -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_available_memory"):
        RuntimeResourcePolicy(available_memory_bytes=0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 0.0, -0.1, 1.1])
def test_runtime_resource_policy_rejects_invalid_fraction_boundaries(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_fraction"):
        RuntimeResourcePolicy(
            available_memory_bytes=8 * GIB,
            memory_budget_fraction=value,
        )


def test_runtime_resource_policy_accepts_fraction_boundary_one() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=8 * GIB,
        memory_budget_fraction=1.0,
    )

    assert policy.memory_budget_bytes == 8 * GIB


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-0.1", "1.1"])
def test_runtime_resource_policy_rejects_invalid_fraction_env_boundaries(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_env"):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=8 * GIB,
            environ={"QRP_MEMORY_BUDGET_FRACTION": value},
        )


def test_runtime_resource_policy_accepts_fraction_env_boundary_one() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=8 * GIB,
        environ={"QRP_MEMORY_BUDGET_FRACTION": "1"},
    )

    assert policy.memory_budget_bytes == 8 * GIB


def test_runtime_resource_policy_falls_back_without_psutil() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=None,
        environ={},
    )

    assert policy.memory_budget_bytes > 0
    assert policy.chunk_target_bytes > 0
    assert policy.chunk_target_bytes <= policy.memory_budget_bytes


def test_resource_chunk_planner_derives_time_session_budget_from_policy() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.80,
        chunk_target_fraction=0.30,
        chunk_target_cap_bytes=2 * GIB,
        chunk_target_bytes=3_000,
        memory_safety_factor=0.50,
        min_free_disk_bytes=0,
    )
    planner = ResourceChunkPlanner(policy)

    plan = planner.plan_sessions(
        session_count=20,
        estimated_bytes_per_session=500,
        axis=ChunkAxis.TIME,
    )

    assert isinstance(plan, ResourceChunkPlan)
    assert plan.axis is ChunkAxis.TIME
    assert plan.target_core_session_count == 3
    assert plan.chunk_count == 7
    assert plan.estimated_chunk_bytes <= policy.effective_chunk_bytes


def test_resource_chunk_planner_clamps_dynamic_target_to_session_count() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.80,
        chunk_target_fraction=0.30,
        chunk_target_cap_bytes=2 * GIB,
        chunk_target_bytes=3_000,
        memory_safety_factor=0.50,
        min_free_disk_bytes=0,
    )
    planner = ResourceChunkPlanner(policy)

    plan = planner.plan_sessions(
        session_count=3,
        estimated_bytes_per_session=100,
        axis=ChunkAxis.TIME,
    )

    assert plan.target_core_session_count == 3
    assert plan.chunk_count == 1


def test_resource_chunk_planner_does_not_require_market_calendar() -> None:
    parameters = ResourceChunkPlanner.plan_sessions.__annotations__

    assert "calendar" not in parameters
    assert "sessions" not in parameters


def test_resource_chunk_planner_does_not_plan_with_invalid_policy() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.80,
        chunk_target_fraction=0.30,
        chunk_target_cap_bytes=2 * GIB,
        chunk_target_bytes=100_000,
        memory_safety_factor=1.0,
        min_free_disk_bytes=0,
    )
    planner = ResourceChunkPlanner(policy)

    with pytest.raises(ValueError, match="chunk_target_exceeds_memory_budget"):
        planner.plan_sessions(
            session_count=10,
            estimated_bytes_per_session=500,
            axis=ChunkAxis.TIME,
        )


def test_runtime_resource_policy_checks_disk_threshold(tmp_path: Path) -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.80,
        chunk_target_fraction=0.30,
        chunk_target_cap_bytes=2 * GIB,
        chunk_target_bytes=3_000,
        memory_safety_factor=0.50,
        min_free_disk_bytes=1,
    )

    assert policy.has_minimum_free_disk(tmp_path)

    strict = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.80,
        chunk_target_fraction=0.30,
        chunk_target_cap_bytes=2 * GIB,
        chunk_target_bytes=3_000,
        memory_safety_factor=0.50,
        min_free_disk_bytes=10**30,
    )

    assert not strict.has_minimum_free_disk(tmp_path)

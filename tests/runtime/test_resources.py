from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ChunkPlanner,
    RuntimeResourcePolicy,
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


def test_runtime_resource_policy_falls_back_without_psutil() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=None,
        environ={},
    )

    assert policy.memory_budget_bytes > 0
    assert policy.chunk_target_bytes > 0
    assert policy.chunk_target_bytes <= policy.memory_budget_bytes


def test_chunk_planner_derives_asset_batches_from_policy() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.80,
        chunk_target_fraction=0.30,
        chunk_target_cap_bytes=2 * GIB,
        chunk_target_bytes=3_000,
        memory_safety_factor=0.50,
        min_free_disk_bytes=0,
    )
    planner = ChunkPlanner(policy)

    plan = planner.plan(
        axis=ChunkAxis.ASSET,
        item_count=10,
        estimated_bytes_per_item=500,
        lookback_window=252,
        lookahead_window=0,
        output_width=56,
        disjoint=True,
    )

    assert plan.axis is ChunkAxis.ASSET
    assert plan.chunk_size == 3
    assert plan.chunk_count == 4
    assert plan.lookback_window == 252
    assert plan.estimated_chunk_bytes <= policy.effective_chunk_bytes


def test_chunk_planner_does_not_plan_with_invalid_policy() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.80,
        chunk_target_fraction=0.30,
        chunk_target_cap_bytes=2 * GIB,
        chunk_target_bytes=100_000,
        memory_safety_factor=1.0,
        min_free_disk_bytes=0,
    )
    planner = ChunkPlanner(policy)

    with pytest.raises(ValueError, match="chunk_target_exceeds_memory_budget"):
        planner.plan(
            axis=ChunkAxis.ASSET,
            item_count=10,
            estimated_bytes_per_item=500,
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

from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ChunkPlanner,
    RuntimeResourcePolicy,
)


def test_runtime_resource_policy_uses_defaults_and_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QRP_MEMORY_BUDGET_MB", "1024")
    monkeypatch.setenv("QRP_CHUNK_TARGET_MB", "128")
    monkeypatch.setenv("QRP_MEMORY_SAFETY_FACTOR", "0.5")
    monkeypatch.setenv("QRP_MIN_FREE_DISK_MB", "256")

    policy = RuntimeResourcePolicy.from_environment(available_memory_bytes=8 << 30)

    assert policy.memory_budget_fraction == 0.70
    assert policy.memory_budget_bytes == 1024 * 1024 * 1024
    assert policy.chunk_target_bytes == 128 * 1024 * 1024
    assert policy.memory_safety_factor == 0.5
    assert policy.min_free_disk_bytes == 256 * 1024 * 1024


def test_runtime_resource_policy_falls_back_without_psutil() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=None,
        environ={},
    )

    assert policy.memory_budget_bytes > 0
    assert policy.chunk_target_bytes > 0
    assert policy.chunk_target_bytes < policy.memory_budget_bytes


def test_chunk_planner_derives_asset_batches_from_policy() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.70,
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


def test_runtime_resource_policy_checks_disk_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.70,
        chunk_target_bytes=3_000,
        memory_safety_factor=0.50,
        min_free_disk_bytes=1,
    )

    assert policy.has_minimum_free_disk(tmp_path)

    strict = RuntimeResourcePolicy(
        available_memory_bytes=10_000,
        memory_budget_fraction=0.70,
        chunk_target_bytes=3_000,
        memory_safety_factor=0.50,
        min_free_disk_bytes=10**30,
    )

    assert not strict.has_minimum_free_disk(tmp_path)

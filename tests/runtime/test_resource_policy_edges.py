from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_research_platform.runtime import resources
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
)


def test_runtime_resource_policy_reads_environment_overrides() -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=1024 * 1024 * 1024,
        environ={
            "QRP_MEMORY_BUDGET_FRACTION": "0.5",
            "QRP_MEMORY_BUDGET_MB": "256",
            "QRP_CHUNK_TARGET_MB": "64",
            "QRP_MEMORY_SAFETY_FACTOR": "0.5",
            "QRP_MIN_FREE_DISK_MB": "1",
        },
    )

    assert policy.memory_budget_bytes == 256 * resources.MIB
    assert policy.effective_chunk_bytes == 32 * resources.MIB
    assert policy.min_free_disk_bytes == resources.MIB


@pytest.mark.parametrize(
    ("environ", "error"),
    [
        ({"QRP_MEMORY_BUDGET_FRACTION": "0"}, "invalid_runtime_resource_env"),
        ({"QRP_MEMORY_BUDGET_MB": "-1"}, "invalid_runtime_resource_env"),
        ({"QRP_CHUNK_TARGET_MB": "0"}, "invalid_runtime_resource_env"),
        ({"QRP_MEMORY_SAFETY_FACTOR": "-0.1"}, "invalid_runtime_resource_env"),
        ({"QRP_MIN_FREE_DISK_MB": "-1"}, "invalid_runtime_resource_env"),
    ],
)
def test_runtime_resource_policy_rejects_invalid_environment(
    environ: dict[str, str],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        RuntimeResourcePolicy.from_environment(
            available_memory_bytes=1024 * 1024,
            environ=environ,
        )


def test_runtime_resource_policy_rejects_chunk_target_over_budget() -> None:
    policy = RuntimeResourcePolicy(
        available_memory_bytes=1024,
        memory_budget_fraction=0.1,
        chunk_target_bytes=1024,
        memory_safety_factor=1.0,
    )

    with pytest.raises(ValueError, match="chunk_target_exceeds_memory_budget"):
        policy.require_valid_chunk_budget()


def test_resource_chunk_planner_handles_empty_and_invalid_inputs() -> None:
    planner = ResourceChunkPlanner(
        RuntimeResourcePolicy(
            available_memory_bytes=1024 * 1024,
            chunk_target_bytes=1024,
            memory_safety_factor=1.0,
        )
    )

    empty = planner.plan_sessions(
        axis=ChunkAxis.TIME,
        session_count=0,
        estimated_bytes_per_session=10,
    )
    assert empty.chunk_count == 0
    assert empty.target_core_session_count == 1

    plan = planner.plan_sessions(
        axis=ChunkAxis.ASSET,
        session_count=25,
        estimated_bytes_per_session=100,
    )
    assert plan.chunk_count == 3
    assert plan.target_core_session_count == 10
    with pytest.raises(ValueError, match="invalid_session_count"):
        planner.plan_sessions(
            axis=ChunkAxis.TIME,
            session_count=-1,
            estimated_bytes_per_session=1,
        )
    with pytest.raises(ValueError, match="invalid_estimated_bytes_per_session"):
        planner.plan_sessions(
            axis=ChunkAxis.TIME,
            session_count=1,
            estimated_bytes_per_session=0,
        )
    with pytest.raises(ValueError, match="session_estimate_exceeds_chunk_budget"):
        planner.plan_sessions(
            axis=ChunkAxis.TIME,
            session_count=1,
            estimated_bytes_per_session=1025,
        )


def test_runtime_resource_policy_checks_disk_space(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100),
    )
    policy = RuntimeResourcePolicy(
        available_memory_bytes=1024,
        min_free_disk_bytes=101,
    )

    assert policy.has_minimum_free_disk(tmp_path / "missing" / "child") is False
    with pytest.raises(RuntimeError, match="insufficient_free_disk"):
        policy.require_minimum_free_disk(tmp_path / "missing" / "child")

    ok_policy = RuntimeResourcePolicy(
        available_memory_bytes=1024,
        min_free_disk_bytes=100,
    )
    assert ok_policy.has_minimum_free_disk(tmp_path) is True

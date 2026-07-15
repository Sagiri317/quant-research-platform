from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.runtime import resources
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
)

MIB = 1024 * 1024


def test_internal_factory_still_enforces_discovery_inputs_and_ceiling() -> None:
    policy = resources._runtime_resource_policy_factory(
        available_memory_bytes=10 * 1024 * MIB,
        environ={"QRP_TASK_MEMORY_BUDGET_MB": "16384"},
    )

    assert policy.task_memory_budget_bytes == 8 * 1024 * MIB
    assert policy.payload_target_bytes == 8 * 1024 * MIB


def test_available_memory_discovery_failure_is_not_replaced_by_fallback(
    monkeypatch,
) -> None:
    def fail_discovery() -> int:
        raise RuntimeError("runtime_available_memory_discovery_failed")

    monkeypatch.setattr(resources, "_available_memory_bytes", fail_discovery)

    with pytest.raises(
        RuntimeError,
        match="runtime_available_memory_discovery_failed",
    ):
        RuntimeResourcePolicy.from_environment(environ={})


def test_disk_headroom_is_independent_of_memory_policy(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100),
    )

    require_free_disk_headroom(tmp_path / "missing" / "child", reserve_bytes=100)
    with pytest.raises(RuntimeError, match="insufficient_free_disk"):
        require_free_disk_headroom(tmp_path, reserve_bytes=101)


@pytest.mark.parametrize("value", [True, 1.0, "1", -1])
def test_disk_headroom_requires_exact_nonnegative_int(value: object, tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_disk_reserve"):
        require_free_disk_headroom(
            tmp_path,
            reserve_bytes=value,  # type: ignore[arg-type]
        )


def test_pandas_frame_payload_bytes_is_the_exact_runtime_diagnostic() -> None:
    frame = pd.DataFrame({"value": ["a", "bb"]})

    assert pandas_frame_payload_bytes(frame) == int(
        frame.memory_usage(index=True, deep=True).sum()
    )


def test_pandas_payload_diagnostic_rejects_non_frame() -> None:
    with pytest.raises(TypeError, match="invalid_pandas_frame_payload"):
        pandas_frame_payload_bytes(pd.Series([1]))  # type: ignore[arg-type]

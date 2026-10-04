from __future__ import annotations

import gc
from contextlib import contextmanager
from unittest.mock import patch

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import RuntimeResourcePolicy


@contextmanager
def disabled_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class _TestMachineLock:
    closed = False

    def close(self) -> None:
        self.closed = True


def capture_test_policy(
    *,
    available_memory_bytes: int,
    explicit_cap_bytes: int | None = None,
) -> RuntimeResourcePolicy:
    with (
        patch.object(
            resources,
            "_available_memory_bytes",
            return_value=available_memory_bytes,
        ),
        patch.object(
            resources._MachineLock,
            "acquire",
            side_effect=lambda: _TestMachineLock(),
        ),
    ):
        return resources._capture_runtime_resource_policy(
            explicit_cap_bytes=explicit_cap_bytes
        )


def capture_storage_test_policy(
    *,
    working_headroom_bytes: int = 1024**2,
) -> RuntimeResourcePolicy:
    from quant_research_platform.storage import parquet
    from quant_research_platform.storage.datasets import writing

    return capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=(
            writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
            + parquet._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
            + working_headroom_bytes
            + 64 * 1024
        ),
    )

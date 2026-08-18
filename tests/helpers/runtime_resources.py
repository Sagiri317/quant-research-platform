from __future__ import annotations

from unittest.mock import patch

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import RuntimeResourcePolicy


class _TestMachineLock:
    closed = False

    def close(self) -> None:
        self.closed = True


def capture_test_policy(
    *,
    available_memory_bytes: int,
    rss_bytes: int = 0,
    explicit_cap_bytes: int | None = None,
) -> RuntimeResourcePolicy:
    with (
        patch.object(
            resources,
            "_available_memory_bytes",
            return_value=available_memory_bytes,
        ),
        patch.object(
            resources,
            "_acquire_machine_lock",
            side_effect=lambda: _TestMachineLock(),
        ),
    ):
        return RuntimeResourcePolicy.capture(explicit_cap_bytes=explicit_cap_bytes)


def capture_storage_test_policy(
    *,
    working_headroom_bytes: int,
) -> RuntimeResourcePolicy:
    from quant_research_platform.storage import _dataset_physical, _dataset_writing

    return capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=(
            _dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
            + _dataset_physical._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
            + working_headroom_bytes
            + 64 * 1024
        ),
    )

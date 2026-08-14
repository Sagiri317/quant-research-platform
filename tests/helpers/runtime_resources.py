from __future__ import annotations

from unittest.mock import patch

import quant_research_platform.runtime.resources as resources
from quant_research_platform.runtime.resources import RuntimeResourcePolicy


class _TestOperationRssMeasurement:
    def __init__(self, rss0_bytes: int, hard_budget_bytes: int) -> None:
        self.rss0_bytes = rss0_bytes
        self.hard_budget_bytes = hard_budget_bytes
        self.state = "READY"
        self.error = None

    def prepare(self) -> None:
        pass

    def bind_baseline(self, rss0_bytes: int, hard_budget_bytes: int) -> None:
        self.rss0_bytes = rss0_bytes
        self.hard_budget_bytes = hard_budget_bytes

    def start(self) -> None:
        self.state = "ACTIVE"

    def stop(self) -> None:
        self.state = "RETIRED"

    def quiesce(self) -> dict[str, int | float | str]:
        return self.checkpoint_evidence()

    def resume(self) -> None:
        if self.state != "ACTIVE":
            raise RuntimeError("runtime_rss_sampler_not_active")

    def checkpoint_evidence(self) -> dict[str, int | float | str]:
        if self.state != "ACTIVE":
            raise RuntimeError("runtime_rss_sampler_not_active")
        return self._payload()

    def evidence(self) -> dict[str, int | float | str]:
        return self._payload()

    def _payload(self) -> dict[str, int | float | str]:
        return {
            "rss0_bytes": self.rss0_bytes,
            "hard_budget_bytes": self.hard_budget_bytes,
            "sampled_peak_rss_bytes": self.rss0_bytes,
            "peak_incremental_rss_bytes": 0,
            "measurement_metric": "test_process_tree_incremental",
            "sample_count": 2,
            "max_sample_interval_ms": 1.0,
            "cadence_status": "PASS",
            "status": "PASS",
        }


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
        patch.object(resources, "_process_rss_bytes", return_value=rss_bytes),
        patch.object(
            resources,
            "_operation_rss_measurement",
            side_effect=_TestOperationRssMeasurement,
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

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


@pytest.mark.parametrize(
    ("owner", "error"),
    [
        ("_available_memory_bytes", "runtime_available_memory_discovery_failed"),
        ("_process_rss_bytes", "runtime_rss_discovery_failed"),
    ],
)
def test_capture_fails_closed_when_required_probe_fails(
    owner: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> int:
        raise RuntimeError(error)

    monkeypatch.setattr(resources, owner, fail)

    with pytest.raises(RuntimeError, match=error):
        RuntimeResourcePolicy.capture()


def test_disk_headroom_is_independent_of_memory_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100),
    )

    assert (
        require_free_disk_headroom(
            tmp_path / "missing" / "child", reserve_bytes=100
        )
        == 100
    )
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

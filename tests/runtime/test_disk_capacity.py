from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_research_platform.runtime import resources
from quant_research_platform.runtime.resources import require_free_disk_headroom


def test_disk_headroom_uses_existing_parent_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=100),
    )

    assert (
        require_free_disk_headroom(tmp_path / "missing" / "child", reserve_bytes=100)
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

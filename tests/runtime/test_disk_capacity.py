from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_research_platform.runtime import resources
from quant_research_platform.runtime.resources import require_free_disk_headroom
from quant_research_platform.storage.identity import native_filesystem_path


def test_disk_headroom_uses_existing_parent_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    paths = []

    def disk_usage(path):
        paths.append(path)
        return SimpleNamespace(free=100)

    monkeypatch.setattr(resources.shutil, "disk_usage", disk_usage)

    assert (
        require_free_disk_headroom(tmp_path / "missing" / "child", reserve_bytes=100)
        == 100
    )
    with pytest.raises(RuntimeError, match="insufficient_free_disk"):
        require_free_disk_headroom(tmp_path, reserve_bytes=101)
    assert paths == [native_filesystem_path(tmp_path)] * 2


@pytest.mark.parametrize("value", [True, 1.0, "1", -1])
def test_disk_headroom_requires_exact_nonnegative_int(value: object, tmp_path) -> None:
    with pytest.raises(ValueError, match="invalid_runtime_resource_disk_reserve"):
        require_free_disk_headroom(
            tmp_path,
            reserve_bytes=value,  # type: ignore[arg-type]
        )

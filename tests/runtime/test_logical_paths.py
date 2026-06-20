from __future__ import annotations

from pathlib import Path

from quant_research_platform.runtime.logical_paths import (
    dataset_layer_from_path,
    logical_path,
)


def test_logical_path_prefers_output_root_when_available(tmp_path: Path) -> None:
    output_root = tmp_path / "mnt" / "data" / "team" / "qrp" / "run"
    path = output_root / "reports" / "private_real" / "a_share" / "summary.json"

    assert logical_path(path, output_root=output_root) == (
        "reports/private_real/a_share/summary.json"
    )


def test_logical_path_uses_last_generated_marker_under_parent_data_path() -> None:
    path = Path("/mnt/data/team/qrp/reports/private_real/run/summary.json")

    assert logical_path(path) == "reports/private_real/run/summary.json"


def test_dataset_layer_from_path_uses_dataset_root_segments() -> None:
    assert (
        dataset_layer_from_path(
            Path("/mnt/data/team/qrp/data/research/a_share/demo")
        )
        == "research"
    )
    assert (
        dataset_layer_from_path(
            Path("verification-output/s1/data/processed/a_share/factor/demo")
        )
        == "processed"
    )

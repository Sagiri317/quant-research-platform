from __future__ import annotations

from pathlib import Path

from quant_research_platform.runtime.logical_paths import (
    dataset_layer_from_path,
    logical_path,
)


def test_logical_path_prefers_run_root_when_available(tmp_path: Path) -> None:
    run_root = tmp_path / "mnt" / "data" / "team" / "qrp" / "run"
    path = run_root / "outputs" / "a_share" / "summary.json"

    assert logical_path(path, run_root=run_root) == ("outputs/a_share/summary.json")


def test_logical_path_uses_last_generated_marker_under_parent_data_path() -> None:
    path = Path("sandbox-output/team/qrp/outputs/run/summary.json")

    assert logical_path(path) == "outputs/run/summary.json"


def test_dataset_layer_from_path_uses_dataset_root_segments() -> None:
    assert (
        dataset_layer_from_path(
            Path("sandbox-output/team/qrp/data/research/a_share/demo")
        )
        == "research"
    )
    assert (
        dataset_layer_from_path(
            Path("verification-output/s1/data/processed/a_share/factor/demo")
        )
        == "processed"
    )

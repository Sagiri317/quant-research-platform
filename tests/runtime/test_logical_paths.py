from __future__ import annotations

from pathlib import Path

from quant_research_platform.runtime.logical_paths import (
    DEFAULT_GENERATED_MARKERS,
    RUN_RESULT_DATASET_LAYERS,
    RUN_RESULT_LOGICAL_PATH_MARKERS,
    dataset_layer_from_path,
    logical_path,
)


def test_logical_path_prefers_run_root_when_available(tmp_path: Path) -> None:
    run_root = tmp_path / "mnt" / "data" / "team" / "qrp" / "run"
    path = run_root / "outputs" / "a_share" / "manifest.json"

    assert logical_path(path, run_root=run_root) == ("outputs/a_share/manifest.json")


def test_logical_path_uses_posix_form() -> None:
    assert logical_path(Path("plain") / "path") == "plain/path"


def test_logical_path_uses_last_generated_marker_under_parent_data_path() -> None:
    path = Path("sandbox-output/team/qrp/outputs/run/manifest.json")

    assert logical_path(path) == "outputs/run/manifest.json"


def test_dataset_layer_from_path_uses_dataset_root_segments() -> None:
    assert (
        dataset_layer_from_path(Path("local-verification-output/s1/data/raw/source"))
        == "raw"
    )
    assert (
        dataset_layer_from_path(
            Path("local-verification-output/s1/data/interim/source/demo")
        )
        == "interim"
    )
    assert (
        dataset_layer_from_path(
            Path("sandbox-output/team/qrp/data/research/a_share/demo")
        )
        == "research"
    )
    assert (
        dataset_layer_from_path(
            Path("local-verification-output/s1/data/processed/a_share/factor/demo")
        )
        == "processed"
    )
    assert (
        dataset_layer_from_path(Path("local-verification-output/s1/data/unknown"))
        is None
    )


def test_run_result_dataset_layers_are_the_exact_workflow_owner_set() -> None:
    assert RUN_RESULT_DATASET_LAYERS == frozenset(
        {"interim", "processed", "research"}
    )


def test_generated_markers_only_include_current_run_tree() -> None:
    assert DEFAULT_GENERATED_MARKERS == (
        "data",
        "outputs",
        "logs",
    )
    assert RUN_RESULT_LOGICAL_PATH_MARKERS == DEFAULT_GENERATED_MARKERS

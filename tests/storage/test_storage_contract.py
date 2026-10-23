from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.storage.datasets import (
    compare_datasets,
    fingerprint_dataset,
    write_dataset,
)
from quant_research_platform.storage.manifests import write_manifest
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_public_dataset_writer_validates_concrete_sequence_elements(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "invalid_sequence")
    with pytest.raises(TypeError, match="invalid_dataset_chunk"):
        write_dataset(
            root,
            [make_default_dataset_frame(), object()],
            spec=make_processed_dataset_spec("invalid_sequence"),
        )
    assert not root.exists()


def test_public_dataset_writer_accepts_exact_frame_sequences(tmp_path: Path) -> None:
    frame = make_default_dataset_frame()
    roots = (
        make_dataset_artifact_root(tmp_path, "list_input"),
        make_dataset_artifact_root(tmp_path, "tuple_input"),
    )
    for root, data in zip(
        roots,
        ([frame.iloc[:2], frame.iloc[2:]], (frame.iloc[:2], frame.iloc[2:])),
        strict=True,
    ):
        write_dataset(root, data, spec=make_processed_dataset_spec(root.name))
    assert (
        fingerprint_dataset(roots[0]).content_hash
        == fingerprint_dataset(roots[1]).content_hash
    )


def test_projected_exact_comparison_includes_primary_key_columns(
    tmp_path: Path,
) -> None:
    left = tmp_path / "data" / "processed" / "a_share" / "factor" / "left"
    right = tmp_path / "data" / "processed" / "a_share" / "factor" / "right"
    right_frame = make_default_dataset_frame(
        asset_ids=("asset_001",), dates=("2014-01-02",), values=(1.0,)
    ).assign(asset_id="asset_002")
    write_dataset(
        left,
        make_default_dataset_frame(
            asset_ids=("asset_001",), dates=("2014-01-02",), values=(1.0,)
        ),
        spec=make_processed_dataset_spec("left"),
    )
    write_dataset(right, right_frame, spec=make_processed_dataset_spec("right"))
    comparison = compare_datasets(left, right, columns=["value"])
    assert comparison.status == "CONTENT_MISMATCH"
    assert comparison.matches is False


def test_projected_compare_binds_only_selected_and_primary_key_schema(
    tmp_path: Path,
) -> None:
    base = make_default_dataset_frame(
        asset_ids=("asset_001",), dates=("2014-01-02",), values=(1.0,)
    )
    spec = make_processed_dataset_spec("projected_left")
    projected_roles = {**spec.column_roles, "ignored": ColumnRole.CONTEXT}
    left = make_dataset_artifact_root(tmp_path, "projected_left")
    ignored = make_dataset_artifact_root(tmp_path, "projected_ignored")
    selected = make_dataset_artifact_root(tmp_path, "projected_selected")
    primary_key = make_dataset_artifact_root(tmp_path, "projected_primary_key")
    write_dataset(
        left, base.assign(ignored=1), spec=replace(spec, column_roles=projected_roles)
    )
    for root, frame in (
        (ignored, base.assign(ignored="different")),
        (selected, base.assign(value="different", ignored=1)),
        (primary_key, base.assign(asset_id=1, ignored=1)),
    ):
        write_dataset(
            root,
            frame,
            spec=replace(
                make_processed_dataset_spec(root.name), column_roles=projected_roles
            ),
        )
    assert compare_datasets(left, ignored, columns=["value"]).status == "MATCH"
    assert (
        compare_datasets(left, selected, columns=["value"]).status == "SCHEMA_MISMATCH"
    )
    assert (
        compare_datasets(left, primary_key, columns=["value"]).status
        == "SCHEMA_MISMATCH"
    )


def test_manifest_writer_rejects_existing_target_file(tmp_path: Path) -> None:
    manifest = tmp_path / "outputs" / "run" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"existing": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="output_already_exists"):
        write_manifest(manifest, {"schema_id": "qrp-output-manifest"})
    assert manifest.read_text(encoding="utf-8") == '{"existing": true}\n'


@pytest.mark.parametrize(
    "relative_path",
    [
        "plain/project/data/processed/a_share/factor/ancestor",
        "docs/project/data/processed/a_share/factor/ancestor",
        "mydocs/project/data/processed/a_share/factor/ancestor",
        "project/docs/shared",
    ],
)
def test_storage_layout_accepts_project_ancestor_names_and_arbitrary_paths(
    tmp_path, relative_path
):
    from quant_research_platform.storage.datasets import read_dataset

    root = tmp_path / relative_path
    frame = make_default_dataset_frame()
    write_dataset(root, frame, spec=make_processed_dataset_spec("ancestor"))
    assert len(read_dataset(root)) == len(frame)

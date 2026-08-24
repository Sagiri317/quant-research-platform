from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import (
    compare_datasets,
    fingerprint_dataset,
    write_dataset,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.manifests import (
    build_layer_summary,
    build_run_data_summary,
    write_manifest,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_dataset_writer_rejects_outside_local_only_root_before_iterating_chunks(
    tmp_path: Path,
) -> None:
    frame = make_default_dataset_frame(
        asset_ids=("asset_001",),
        dates=("2014-01-02",),
        values=(1.0,),
    )

    with pytest.raises(ValueError, match="outside_local_only_boundary"):
        write_dataset(
            tmp_path / "outside-local-only" / "x",
            frame,
            spec=make_processed_dataset_spec(),
        )

    assert not (tmp_path / "public" / "x").exists()


def test_public_dataset_writer_rejects_lazy_input_before_iteration(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "lazy_input")

    class DangerousLazyInput:
        iter_called = False

        def __iter__(self) -> DangerousLazyInput:
            self.iter_called = True
            return self

    data = DangerousLazyInput()
    with pytest.raises(TypeError, match="invalid_dataset_write_input"):
        write_dataset(root, data, spec=make_processed_dataset_spec("lazy_input"))  # type: ignore[arg-type]

    assert data.iter_called is False
    assert not root.exists()


def test_public_dataset_writer_validates_concrete_sequence_elements(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "invalid_sequence")

    with pytest.raises(TypeError, match="invalid_dataset_write_input"):
        write_dataset(
            root,
            [make_default_dataset_frame(), object()],  # type: ignore[list-item]
            spec=make_processed_dataset_spec("invalid_sequence"),
        )

    assert not root.exists()


def test_public_dataset_writer_accepts_exact_frame_sequences(
    tmp_path: Path,
) -> None:
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


def test_public_dataset_writer_does_not_recharge_borrowed_frame_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "borrowed_input")
    frame = make_default_dataset_frame()
    reservation_owners: list[str] = []
    original_lease = RuntimeResourcePolicy.lease

    def record_lease(
        policy: RuntimeResourcePolicy,
        owner: str,
        reserved_bytes: int,
    ):
        reservation_owners.append(owner)
        return original_lease(policy, owner, reserved_bytes)

    monkeypatch.setattr(
        RuntimeResourcePolicy,
        "lease",
        record_lease,
    )

    write_dataset(root, frame, spec=make_processed_dataset_spec("borrowed_input"))

    assert "storage.dataset.input" not in reservation_owners
    assert any(
        owner.startswith("storage.dataset.")
        for owner in reservation_owners
        if owner != "storage.dataset.write"
    )


def test_dataset_writer_accepts_private_processed_artifact_root(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)

    write_dataset(
        root,
        make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(1.0,),
        ),
        spec=make_processed_dataset_spec(),
    )

    assert (root / "spec.json").is_file()
    assert (root / "_manifest" / "manifest.json").is_file()
    assert (root / "data" / "year=2014" / "part-000.parquet").is_file()


def test_projected_dataset_fingerprint_includes_primary_key_columns(
    tmp_path: Path,
) -> None:
    left = tmp_path / "data" / "processed" / "a_share" / "factor" / "left"
    right = tmp_path / "data" / "processed" / "a_share" / "factor" / "right"
    right_frame = make_default_dataset_frame(
        asset_ids=("asset_001",),
        dates=("2014-01-02",),
        values=(1.0,),
    ).assign(asset_id="asset_002")

    write_dataset(
        left,
        make_default_dataset_frame(
            asset_ids=("asset_001",),
            dates=("2014-01-02",),
            values=(1.0,),
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
        asset_ids=("asset_001",),
        dates=("2014-01-02",),
        values=(1.0,),
    )
    spec = make_processed_dataset_spec("projected_left")
    projected_roles = {**spec.column_roles, "ignored": ColumnRole.CONTEXT}
    left = make_dataset_artifact_root(tmp_path, "projected_left")
    ignored = make_dataset_artifact_root(tmp_path, "projected_ignored")
    selected = make_dataset_artifact_root(tmp_path, "projected_selected")
    primary_key = make_dataset_artifact_root(tmp_path, "projected_primary_key")
    write_dataset(
        left,
        base.assign(ignored=1),
        spec=replace(spec, column_roles=projected_roles),
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
                make_processed_dataset_spec(root.name),
                column_roles=projected_roles,
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


def test_output_already_exists_error_review_safe_summary() -> None:
    assert OutputAlreadyExistsError().to_review_safe_summary() == {
        "status": "failed",
        "failure_code": "output_already_exists",
    }


def test_build_layer_summary_detects_missing_dataset_manifests(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    present = layer_root / "a_share" / "factor" / "present"
    missing = layer_root / "a_share" / "label" / "missing"
    write_dataset(
        present,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("present"),
    )

    payload = build_layer_summary(
        layer_root=layer_root,
        logical_layer="processed",
        dataset_roots=(present, missing),
    )

    assert payload["dataset_count"] == 1
    assert payload["total_rows"] == 3
    assert payload["missing_dataset_manifests"] == [
        "data/processed/a_share/label/missing/_manifest/manifest.json"
    ]
    assert payload["layer_validation_status"] == "FAIL"


def test_build_run_data_summary_aggregates_dynamic_layers(tmp_path: Path) -> None:
    processed = build_layer_summary(
        layer_root=tmp_path / "data" / "processed",
        logical_layer="processed",
        dataset_roots=(),
    )
    research = build_layer_summary(
        layer_root=tmp_path / "data" / "research",
        logical_layer="research",
        dataset_roots=(),
    )

    payload = build_run_data_summary(
        layer_summaries=(processed, research),
    )

    assert payload["layer_count"] == 2
    assert payload["run_validation_status"] == "PASS"
    assert [layer["logical_layer"] for layer in payload["layers"]] == [
        "processed",
        "research",
    ]

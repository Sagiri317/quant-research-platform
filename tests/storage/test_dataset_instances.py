from __future__ import annotations

from pathlib import Path

import pytest

import quant_research_platform.storage.datasets.instances as dataset_instances
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.instances import (
    discover_dataset_instances,
)
from quant_research_platform.storage.datasets.schema import data_spec_payload
from quant_research_platform.storage.layout import StorageLayout
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def _envelope(spec, *, seed: int) -> BusinessManifestEnvelope:  # noqa: ANN001
    return build_business_manifest_envelope(
        logical_artifact_id=spec.data_id,
        schema_identity=canonical_json_hash(data_spec_payload(spec)),
        protocol_or_spec_identity="a" * 64,
        software_identity="b" * 64,
        research_window={
            "start": "2020-01-01",
            "end": "2020-12-31",
            "access": "development",
        },
        upstream_identities={"upstream_dataset": "c" * 64},
        semantic_metadata={"root_seed": seed},
    )


def _write_instance(
    logical_root: Path, spec, envelope: BusinessManifestEnvelope
) -> Path:  # noqa: ANN001
    instance_root = StorageLayout().instance_root(
        logical_root,
        envelope.instance_identity,
    )
    write_dataset(
        instance_root,
        make_default_dataset_frame(),
        spec=spec,
        instance_metadata={"common_manifest": envelope},
    )
    return instance_root


def test_two_immutable_instances_round_trip_and_discover_deterministically(
    tmp_path: Path,
) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    first = _write_instance(logical_root, spec, _envelope(spec, seed=1))
    second = _write_instance(logical_root, spec, _envelope(spec, seed=2))

    assert first != second
    assert inspect_dataset(first).manifest["common_manifest"]["status"] == "COMPLETE"
    assert len(read_dataset(first)) == len(make_default_dataset_frame())
    discovered = discover_dataset_instances(logical_root)
    assert discovered == tuple(sorted((first, second), key=lambda path: path.name))
    assert (logical_root / "instances").is_dir()


def test_discovery_rejects_bare_and_mismatched_instance_directories(
    tmp_path: Path,
) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    valid = _write_instance(logical_root, spec, _envelope(spec, seed=1))
    instances_root = logical_root / "instances"
    (instances_root / "not-an-instance").mkdir()
    with pytest.raises(ValueError, match="dataset_instance_child_invalid"):
        discover_dataset_instances(logical_root)

    (instances_root / "not-an-instance").rmdir()
    unexpected_file = instances_root / "unexpected"
    unexpected_file.write_bytes(b"not-an-instance")
    with pytest.raises(ValueError, match="dataset_instance_child_invalid"):
        discover_dataset_instances(logical_root)
    unexpected_file.unlink()

    bare = instances_root / ("d" * 64)
    bare.mkdir()
    with pytest.raises(ValueError, match="dataset_instance_invalid"):
        discover_dataset_instances(logical_root)

    bare.rmdir()
    renamed = instances_root / ("e" * 64)
    valid.rename(renamed)
    with pytest.raises(ValueError, match="dataset_instance_identity_mismatch"):
        discover_dataset_instances(logical_root)


@pytest.mark.parametrize(
    "mutation",
    ("logical_artifact_id", "schema_identity"),
)
def test_dataset_writer_rejects_common_manifest_cross_binding(
    tmp_path: Path,
    mutation: str,
) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    values = {
        "logical_artifact_id": "other_dataset",
        "schema_identity": "d" * 64,
    }
    envelope = _envelope(spec, seed=1)
    bad = build_business_manifest_envelope(
        logical_artifact_id=(
            values["logical_artifact_id"]
            if mutation == "logical_artifact_id"
            else envelope.logical_artifact_id
        ),
        schema_identity=(
            values["schema_identity"]
            if mutation == "schema_identity"
            else envelope.schema_identity
        ),
        protocol_or_spec_identity=envelope.protocol_or_spec_identity,
        software_identity=envelope.software_identity,
        research_window=dict(envelope.research_window),
        upstream_identities=dict(envelope.upstream_identities),
        semantic_metadata=dict(envelope.semantic_metadata),
    )
    instance_root = StorageLayout().instance_root(
        logical_root,
        bad.instance_identity,
    )

    with pytest.raises(ValueError, match="dataset_manifest_common_"):
        write_dataset(
            instance_root,
            make_default_dataset_frame(),
            spec=spec,
            instance_metadata={"common_manifest": bad},
        )
    assert not instance_root.exists()


def test_discovery_rejects_instances_directory_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    valid = _write_instance(logical_root, spec, _envelope(spec, seed=1))
    instances_root = logical_root / "instances"
    read_manifest = dataset_instances.read_dataset_manifest_evidence

    def mutate_directory(entry: Path) -> dict[str, object]:
        (instances_root / "late").write_bytes(b"late")
        return read_manifest(entry)

    monkeypatch.setattr(
        dataset_instances,
        "read_dataset_manifest_evidence",
        mutate_directory,
    )
    with pytest.raises(ValueError, match="dataset_instance_identity_changed"):
        discover_dataset_instances(logical_root)
    assert valid.exists()


def test_discovery_rejects_symlink_instance_child(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    _write_instance(logical_root, spec, _envelope(spec, seed=1))
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = logical_root / "instances" / ("f" * 64)
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="dataset_instance_child_invalid"):
        discover_dataset_instances(logical_root)

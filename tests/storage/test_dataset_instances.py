from __future__ import annotations

from pathlib import Path

import pytest

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
        protocol_or_spec_identity="a_share_low_frequency",
        software_identity="b" * 64,
        research_window={
            "start": "2020-01-01",
            "end": "2020-12-31",
            "access": "development",
        },
        upstream_identities=("c" * 64,),
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
    assert valid in discover_dataset_instances(logical_root)

    bare = instances_root / ("d" * 64)
    bare.mkdir()
    with pytest.raises(ValueError, match="dataset_instance_invalid"):
        discover_dataset_instances(logical_root)

    bare.rmdir()
    renamed = instances_root / ("e" * 64)
    valid.rename(renamed)
    with pytest.raises(ValueError, match="dataset_instance_identity_mismatch"):
        discover_dataset_instances(logical_root)

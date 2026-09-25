from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import quant_research_platform.storage.datasets.instances as dataset_instances
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.business_manifest import (
    BusinessManifestEnvelope,
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.instances import (
    discover_dataset_instances,
)
from quant_research_platform.storage.datasets.schema import data_spec_payload
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_is_dir,
    filesystem_mkdir,
    filesystem_rename,
    filesystem_unlink,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.research_specs import make_demo_research_dataset_spec
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
    assert filesystem_is_dir(logical_root / "instances")


@pytest.mark.parametrize("deep_root", (False, True))
def test_research_instance_round_trip_discovery_and_inspect_at_variable_roots(
    tmp_path: Path,
    deep_root: bool,
) -> None:
    project_root = tmp_path
    if deep_root:
        while len(str(project_root / "data/research/a_share/demo")) <= 260:
            project_root /= "segment_" + "a" * 32
    layout = StorageLayout(project_root)
    spec = make_demo_research_dataset_spec(
        factor_columns=("value",),
        label_columns=("label",),
        filter_columns=(),
        context_columns=(),
    )
    envelope = _envelope(spec, seed=1)
    logical_root = layout.research_dataset_root(spec.market_id, spec.data_id)
    instance_root = layout.instance_root(logical_root, envelope.instance_identity)
    if deep_root:
        assert len(str(instance_root)) > 260
    frame = make_default_dataset_frame()
    frame["label"] = [0.1, 0.2, 0.3]
    write_dataset(
        instance_root,
        frame,
        spec=spec,
        instance_metadata={"common_manifest": envelope},
    )

    assert inspect_dataset(instance_root).row_count == len(frame)
    assert len(read_dataset(instance_root)) == len(frame)
    assert discover_dataset_instances(logical_root) == (instance_root,)
    profile = replace(
        research_profile(), research_datasets=ResearchDatasetCatalog((spec,))
    )
    result = research.inspect(profile=profile, layout=layout, dataset_id=spec.data_id)
    assert result["status"] == "BLOCKED"
    assert result["blocker"] == "request_identity_required"
    assert result["instances"] == [
        {
            "instance_identity": envelope.instance_identity,
            "status": "COMPLETE",
            "schema_identity": envelope.schema_identity,
            "row_count": len(frame),
        }
    ]


def test_discovery_rejects_bare_and_mismatched_instance_directories(
    tmp_path: Path,
) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    valid = _write_instance(logical_root, spec, _envelope(spec, seed=1))
    instances_root = logical_root / "instances"
    filesystem_mkdir(instances_root / "not-an-instance")
    with pytest.raises(ValueError, match="dataset_instance_child_invalid"):
        discover_dataset_instances(logical_root)

    Path(native_filesystem_path(instances_root / "not-an-instance")).rmdir()
    unexpected_file = instances_root / "unexpected"
    filesystem_write_bytes(unexpected_file, b"not-an-instance")
    with pytest.raises(ValueError, match="dataset_instance_child_invalid"):
        discover_dataset_instances(logical_root)
    filesystem_unlink(unexpected_file)

    bare = instances_root / ("d" * 64)
    filesystem_mkdir(bare)
    with pytest.raises(ValueError, match="dataset_instance_invalid"):
        discover_dataset_instances(logical_root)

    Path(native_filesystem_path(bare)).rmdir()
    renamed = instances_root / ("e" * 64)
    filesystem_rename(valid, renamed)
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
    assert not filesystem_exists(instance_root)


@pytest.mark.parametrize("fixed_directory_identity", (False, True))
def test_discovery_rejects_instances_directory_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_directory_identity: bool,
) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    valid = _write_instance(logical_root, spec, _envelope(spec, seed=1))
    instances_root = logical_root / "instances"
    read_manifest = dataset_instances.read_dataset_manifest_evidence
    if fixed_directory_identity:
        original = dataset_instances._directory_identity
        observed = {}

        def identity(path, error):
            if path not in observed:
                observed[path] = original(path, error)
            return observed[path]

        monkeypatch.setattr(dataset_instances, "_directory_identity", identity)

    def mutate_directory(entry: Path) -> dict[str, object]:
        filesystem_write_bytes(instances_root / "late", b"late")
        return read_manifest(entry)

    monkeypatch.setattr(
        dataset_instances,
        "read_dataset_manifest_evidence",
        mutate_directory,
    )
    with pytest.raises(ValueError, match="dataset_instance_identity_changed"):
        discover_dataset_instances(logical_root)
    assert filesystem_exists(valid)


def test_discovery_rejects_symlink_instance_child(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("demo")
    logical_root = make_dataset_artifact_root(tmp_path, "demo")
    _write_instance(logical_root, spec, _envelope(spec, seed=1))
    outside = tmp_path / "outside"
    filesystem_mkdir(outside)
    alias = logical_root / "instances" / ("f" * 64)
    try:
        Path(native_filesystem_path(alias)).symlink_to(
            native_filesystem_path(outside), target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="dataset_instance_child_invalid"):
        discover_dataset_instances(logical_root)


def test_exact_requested_instance_ignores_corrupt_history_but_discovery_rejects(
    tmp_path, monkeypatch
):
    from quant_research_platform.catalog.processed import ProcessedDataCatalog

    spec = make_processed_dataset_spec("demo")
    profile = replace(research_profile(), processed=ProcessedDataCatalog((spec,)))
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    good = _write_instance(root, spec, _envelope(spec, seed=1))
    bad = _write_instance(root, spec, _envelope(spec, seed=2))
    filesystem_write_text(
        bad / "_manifest" / "manifest.json", "invalid", encoding="utf-8"
    )
    assert research._matching_instance(
        profile=profile, layout=layout, target_id="demo", requested_identity=good.name
    ) == (good.name, None)
    assert research._matching_instance(
        profile=profile, layout=layout, target_id="demo", requested_identity=bad.name
    ) == (None, "persisted_instance_invalid")
    with pytest.raises(ValueError):
        discover_dataset_instances(root)
    real_stat = research.filesystem_stat

    def denied(path, *args, **kwargs):
        if path == good:
            raise PermissionError("synthetic_permission_denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(research, "filesystem_stat", denied)
    with pytest.raises(PermissionError, match="synthetic_permission_denied"):
        research._matching_instance(
            profile=profile,
            layout=layout,
            target_id="demo",
            requested_identity=good.name,
        )

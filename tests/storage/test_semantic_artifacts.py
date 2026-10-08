from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.storage.identity import (
    filesystem_iterdir,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_replace,
    filesystem_stat,
    filesystem_write_bytes,
    native_filesystem_path,
)
from quant_research_platform.storage.semantic_artifacts import (
    discover_semantic_artifact_instances,
    read_semantic_artifact,
    write_semantic_artifact,
)


def _immediate_file_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: filesystem_read_bytes(path) for path in filesystem_iterdir(root)}


def _envelope(*, seed=None) -> object:
    return build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": "2019-03-31",
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={} if seed is None else {"seed": seed},
    )


def _write_artifact(tmp_path: Path) -> Path:

    def writer(staging: Path) -> None:
        filesystem_write_bytes(staging / "model.json", b"model")
        filesystem_write_bytes(staging / "model.pkl", b"predictor")

    return write_semantic_artifact(tmp_path / "model", _envelope(), writer)


def test_discovery_during_real_semantic_publication_keeps_published_instances(tmp_path):
    first = _write_artifact(tmp_path)
    logical_root = tmp_path / "model"
    old_bytes = _immediate_file_bytes(first)

    def writer(staging):
        assert discover_semantic_artifact_instances(logical_root) == (first,)
        filesystem_write_bytes(staging / "model.bin", b"new")

    second = write_semantic_artifact(logical_root, _envelope(seed=2), writer)
    published = tuple(sorted((first, second), key=lambda path: path.name))
    assert discover_semantic_artifact_instances(logical_root) == published
    assert _immediate_file_bytes(first) == old_bytes

    def failed_writer(staging):
        assert discover_semantic_artifact_instances(logical_root) == published
        filesystem_write_bytes(staging / "model.bin", b"aborted")
        raise RuntimeError("publication_failed")

    with pytest.raises(RuntimeError, match="^publication_failed$"):
        write_semantic_artifact(logical_root, _envelope(seed=3), failed_writer)
    assert discover_semantic_artifact_instances(logical_root) == published
    assert _immediate_file_bytes(first) == old_bytes


@pytest.mark.parametrize(
    "name", [".a-000000000000", ".a-00000000000g", ".a-000000000000-extra"]
)
def test_semantic_discovery_rejects_atomic_named_files_and_invalid_scratch_names(
    tmp_path, name
):
    _write_artifact(tmp_path)
    candidate = tmp_path / "model/instances" / name
    if name == ".a-000000000000":
        filesystem_write_bytes(candidate, b"not-a-directory")
    else:
        filesystem_mkdir(candidate)
    with pytest.raises(ValueError, match="^semantic_artifact_instance_child_invalid$"):
        discover_semantic_artifact_instances(tmp_path / "model")


def test_discovery_accepts_more_than_1024_published_semantic_instances(
    tmp_path: Path,
) -> None:
    logical_root = tmp_path / "history"
    base = _envelope()
    expected = []
    for seed in range(1025):
        envelope = build_business_manifest_envelope(
            logical_artifact_id=base.logical_artifact_id,
            schema_identity=base.schema_identity,
            protocol_or_spec_identity=base.protocol_or_spec_identity,
            software_identity=base.software_identity,
            research_window=base.research_window,
            upstream_identities=base.upstream_identities,
            semantic_metadata={"seed": seed},
        )
        expected.append(
            write_semantic_artifact(
                logical_root,
                envelope,
                lambda staging: filesystem_write_bytes(staging / "model.bin", b"x"),
            )
        )
    assert discover_semantic_artifact_instances(logical_root) == tuple(
        sorted(expected, key=lambda path: path.name)
    )


def test_semantic_artifact_empty_members_are_rejected_before_publication(
    tmp_path: Path,
) -> None:
    envelope = _envelope()
    logical_root = tmp_path / "empty"
    with pytest.raises(ValueError, match="semantic_artifact_member_manifest_invalid"):
        write_semantic_artifact(logical_root, envelope, lambda _staging: None)
    assert not (logical_root / "instances" / envelope.instance_identity).exists()


def test_semantic_artifact_member_mutation_is_rejected_by_read_and_discovery(
    tmp_path: Path,
) -> None:
    root = _write_artifact(tmp_path)
    filesystem_write_bytes(root / "model.pkl", b"changed")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_semantic_artifact(root)
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        discover_semantic_artifact_instances(tmp_path / "model")


def test_semantic_artifact_extra_member_is_rejected(tmp_path: Path) -> None:
    root = _write_artifact(tmp_path)
    filesystem_write_bytes(root / "extra.bin", b"extra")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_semantic_artifact(root)


def test_exact_comparison_keeps_attempt_provenance_outside_model_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from uuid import uuid4

    import quant_research_platform.storage.semantic_artifacts as storage
    from quant_research_platform.contracts.experiments import (
        trial_ledger_semantic_identity,
    )
    from quant_research_platform.storage.experiments import append_trial_record
    from quant_research_platform.storage.layout import StorageLayout
    from tests.research.test_experiment_ledger import _plan, _record

    layout = StorageLayout(tmp_path)

    def build(name: str, *, alpha: float = 1.0, metric: float = 0.2, threads: int = 1):
        plan = replace(
            _plan(),
            experiment_id=f"experiment_{uuid4().hex}",
            thread_settings={"native_threads": threads},
        )
        record = replace(
            _record(
                plan,
                event_id=f"event_{uuid4().hex}",
                attempt_index=1,
                status="completed",
            ),
            run_id=f"run_{name}",
            normalized_parameters={"alpha": alpha},
            metrics={"rankic": metric},
            thread_settings={"native_threads": threads},
        )
        append_trial_record(
            layout.trial_ledger_root(record.run_id, plan.experiment_id), plan, record
        )
        reference = {
            "run_id": record.run_id,
            "experiment_id": plan.experiment_id,
            "protocol_hash": plan.protocol_hash,
            "semantic_identity": trial_ledger_semantic_identity(plan, (record,)),
        }
        return storage.write_semantic_artifact(
            tmp_path / name,
            _envelope(),
            lambda staging: filesystem_write_bytes(
                staging / "model.bin", b"same model"
            ),
            operational_metadata={"model_trial_ledgers": [reference]},
        )

    old = build("first")
    same = build("retry", threads=2)
    before = filesystem_read_bytes(old / "manifest.json")
    assert (
        storage.compare_semantic_artifacts(old, same, trial_layout=layout).status
        == "MATCH"
    )
    for candidate in (
        build("parameter", alpha=2.0),
        build("typed_parameter", alpha=1),
        build("metric", metric=0.3),
    ):
        assert (
            storage.compare_semantic_artifacts(
                old, candidate, trial_layout=layout
            ).status
            == "CONTENT_MISMATCH"
        )
    assert (
        storage.compare_semantic_artifacts(
            build("positive_zero", alpha=0.0),
            build("negative_zero", alpha=-0.0),
            trial_layout=layout,
        ).status
        == "MATCH"
    )
    assert filesystem_read_bytes(old / "manifest.json") == before
    import hashlib

    real_sha256 = hashlib.sha256

    class Collision:
        def update(self, _value):
            pass

        def hexdigest(self):
            return "a" * 64

    monkeypatch.setattr(
        hashlib,
        "sha256",
        lambda value=b"": real_sha256(value) if value else Collision(),
    )
    left = storage.write_semantic_artifact(
        tmp_path / "collision_left",
        _envelope(),
        lambda staging: filesystem_write_bytes(staging / "data.bin", b"AAAA"),
    )
    right = storage.write_semantic_artifact(
        tmp_path / "collision_right",
        _envelope(),
        lambda staging: filesystem_write_bytes(staging / "data.bin", b"BBBB"),
    )
    assert storage.compare_semantic_artifacts(left, right).status == "CONTENT_MISMATCH"


@pytest.mark.parametrize("stage", ("inventory", "exact_bytes"))
def test_semantic_compare_read_failure_keeps_primary_when_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    import quant_research_platform.storage.identity as identity
    import quant_research_platform.storage.semantic_artifacts as storage

    left, right = (
        _write_artifact(tmp_path / "left"),
        _write_artifact(tmp_path / "right"),
    )
    primary = KeyboardInterrupt("test primary")
    owner = identity if stage == "inventory" else storage
    original_open = owner.open_filesystem_file
    reads = []

    class BrokenRead:
        def __init__(self, handle):
            self.handle = handle

        def fileno(self):
            return self.handle.fileno()

        def read(self, _size):
            reads.append(stage)
            raise primary

        def close(self):
            self.handle.close()
            raise RuntimeError("close failed")

    def opening(path, *args, **kwargs):
        return BrokenRead(original_open(path, *args, **kwargs))

    monkeypatch.setattr(owner, "open_filesystem_file", opening)
    with pytest.raises(KeyboardInterrupt) as caught:
        storage.compare_semantic_artifacts(left, right)
    assert caught.value is primary
    assert reads == [stage]
    assert any("close also failed" in note for note in primary.__notes__)


def test_member_limit_precedes_wide_directory_materialization_and_hashing(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import quant_research_platform.storage.semantic_artifacts as storage

    root = _write_artifact(tmp_path)
    counts = {"next": 0, "hash": 0, "close": 0}

    class Entries:
        def __iter__(self):
            return self

        def __next__(self):
            if counts["next"] == 10000:
                raise StopIteration
            counts["next"] += 1
            return SimpleNamespace(
                name=f"member-{counts['next']}.bin",
                is_symlink=lambda: False,
                is_dir=lambda **_: False,
                is_file=lambda **_: True,
            )

        def close(self):
            counts["close"] += 1

    def record(relative, _path):
        counts["hash"] += 1
        return {"path": relative, "size": 0, "sha256": "a" * 64}

    monkeypatch.setattr(storage, "filesystem_scandir", lambda _path: Entries())
    monkeypatch.setattr(storage, "_member_record", record)
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        storage.read_semantic_artifact_record(root)
    assert counts == {"next": 1, "hash": 0, "close": 1}


def test_nested_inventory_descends_before_reading_next_sibling(tmp_path, monkeypatch):
    import quant_research_platform.storage.semantic_artifacts as storage

    def writer(staging):
        for index in range(50):
            filesystem_mkdir(staging / f"empty-{index}")
        nested = staging / "z" / "inner"
        filesystem_mkdir(nested, parents=True)
        filesystem_write_bytes(nested / "model.bin", b"nested")
        filesystem_write_bytes(staging / "a.bin", b"root")

    actual_scandir = storage.filesystem_scandir
    opened = []

    class Entries:
        def __init__(self, path):
            self.entries = actual_scandir(path)
            opened.append(self)

        def __next__(self):
            assert opened[-1] is self
            return next(self.entries)

        def close(self):
            assert opened.pop() is self
            self.entries.close()

    actual_inventory = storage._member_inventory

    def inventory(root, **kwargs):
        with monkeypatch.context() as scoped:
            scoped.setattr(storage, "filesystem_scandir", Entries)
            return actual_inventory(root, **kwargs)

    monkeypatch.setattr(storage, "_member_inventory", inventory)
    root = storage.write_semantic_artifact(tmp_path / "model", _envelope(), writer)
    envelope, members = storage.read_semantic_artifact_record(root)
    assert envelope == _envelope().payload()
    assert [member["path"] for member in members] == ["a.bin", "z/inner/model.bin"]
    assert [member["size"] for member in members] == [4, 6]
    assert not opened


def test_inventory_closes_all_ancestors_without_replacing_primary(
    tmp_path, monkeypatch
):
    import quant_research_platform.storage.semantic_artifacts as storage

    def writer(staging):
        filesystem_mkdir(staging / "nested")
        filesystem_write_bytes(staging / "nested" / "bad.bin", b"bad")

    root = write_semantic_artifact(tmp_path / "model", _envelope(), writer)
    primary = KeyboardInterrupt("inventory primary")
    actual_scandir = storage.filesystem_scandir
    closed = []

    class Entries:
        def __init__(self, path):
            self.path = Path(path)
            self.entries = actual_scandir(path)

        def __next__(self):
            return next(self.entries)

        def close(self):
            self.entries.close()
            closed.append(self.path)
            raise RuntimeError("close failed")

    actual_record = storage._member_record

    def record(relative, path, **kwargs):
        if relative == "nested/bad.bin":
            raise primary
        return actual_record(relative, path, **kwargs)

    monkeypatch.setattr(storage, "filesystem_scandir", Entries)
    monkeypatch.setattr(storage, "_member_record", record)
    with pytest.raises(KeyboardInterrupt) as caught:
        storage.read_semantic_artifact_record(root)
    assert caught.value is primary
    assert closed == [root / "nested", root]
    assert len(primary.__notes__) == 2


@pytest.mark.parametrize("alias_kind", ("symlink", "junction"))
def test_inventory_rejects_child_alias_before_scanning_external_content(
    tmp_path, monkeypatch, alias_kind
):
    import os
    import subprocess

    import quant_research_platform.storage.semantic_artifacts as storage

    root = _write_artifact(tmp_path)
    external = tmp_path / "external"
    filesystem_mkdir(external)
    payload = external / "outside.bin"
    filesystem_write_bytes(payload, b"external synthetic content")
    alias = root / "alias"
    if alias_kind == "symlink":
        try:
            Path(native_filesystem_path(alias)).symlink_to(
                native_filesystem_path(external), target_is_directory=True
            )
        except OSError:
            pytest.skip("symlink_not_available")
    else:
        if os.name != "nt" or not hasattr(Path, "is_junction"):
            pytest.skip("junction_capability_required")
        created = subprocess.run(
            [
                "cmd",
                "/c",
                "mklink",
                "/J",
                native_filesystem_path(alias),
                native_filesystem_path(external),
            ],
            check=False,
            capture_output=True,
        )
        if created.returncode != 0:
            pytest.skip("junction_not_available")
    actual_scandir = storage.filesystem_scandir

    def scan(path):
        assert Path(path) == root
        return actual_scandir(path)

    monkeypatch.setattr(storage, "filesystem_scandir", scan)
    with pytest.raises(ValueError, match="semantic_artifact_member_invalid"):
        storage.read_semantic_artifact_record(root)
    assert filesystem_read_bytes(payload) == b"external synthetic content"


def test_member_inventory_has_no_total_byte_quota(tmp_path):
    root = _write_artifact(tmp_path)
    old_good = _immediate_file_bytes(root)
    candidate_fields = _envelope().payload()
    del candidate_fields["instance_identity"], candidate_fields["status"]
    candidate_fields["semantic_metadata"] = {"candidate": True}
    candidate = build_business_manifest_envelope(**candidate_fields)

    def writer(staging):
        for index in range(257):
            filesystem_write_bytes(staging / f"member-{index}.bin", b"small")

    published = write_semantic_artifact(tmp_path / "model", candidate, writer)
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_record,
    )

    _envelope_payload, members = read_semantic_artifact_record(published)
    assert len(members) == 257
    assert _immediate_file_bytes(root) == old_good


@pytest.mark.parametrize("operation", ("discovery", "publication"))
def test_constant_directory_identity_cannot_hide_added_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    import quant_research_platform.storage.semantic_artifacts as storage

    root = _write_artifact(tmp_path)
    old_good = _immediate_file_bytes(root)
    before = set(filesystem_iterdir(root.parent))
    original_identity = storage._directory_identity
    observed, calls = ({}, {})

    def identity(path):
        if path not in observed:
            observed[path] = original_identity(path)
        calls[path] = calls.get(path, 0) + 1
        if operation == "publication" and calls[path] == 2:
            filesystem_write_bytes(path / "late.bin", b"late")
        return observed[path]

    monkeypatch.setattr(storage, "_directory_identity", identity)
    if operation == "discovery":
        original_read = storage.read_semantic_artifact

        def read(path, **kwargs):
            filesystem_write_bytes(root.parent / "late", b"late")
            return original_read(path, **kwargs)

        monkeypatch.setattr(storage, "read_semantic_artifact", read)
        with pytest.raises(
            ValueError, match="semantic_artifact_instance_identity_changed"
        ):
            storage.discover_semantic_artifact_instances(tmp_path / "model")
    else:
        fields = _envelope().payload()
        del fields["instance_identity"], fields["status"]
        fields["semantic_metadata"] = {"candidate": True}
        candidate = build_business_manifest_envelope(**fields)

        def writer(staging):
            filesystem_mkdir(staging / "empty")
            filesystem_write_bytes(staging / "data.bin", b"data")

        with pytest.raises(ValueError, match="semantic_artifact_member_set_changed"):
            storage.write_semantic_artifact(tmp_path / "model", candidate, writer)
        assert set(filesystem_iterdir(root.parent)) == before
    assert _immediate_file_bytes(root) == old_good


@pytest.mark.parametrize("mutation", ("earlier_member", "member_set", "manifest"))
def test_comparison_rechecks_the_complete_round_before_match(
    tmp_path, monkeypatch, mutation
):
    import inspect

    import quant_research_platform.storage.semantic_artifacts as storage

    def writer(root):
        filesystem_write_bytes(root / "a.bin", b"AAAA")
        filesystem_write_bytes(root / "b.bin", b"BBBB")

    left = storage.write_semantic_artifact(tmp_path / "left", _envelope(), writer)
    right = storage.write_semantic_artifact(tmp_path / "right", _envelope(), writer)
    old_good = _immediate_file_bytes(left)
    original = storage.open_filesystem_file
    injected = []

    def opening(path, *args, **kwargs):
        if (
            not injected
            and path == left / "b.bin"
            and (
                inspect.currentframe().f_back.f_code.co_name
                == "compare_semantic_artifacts"
            )
        ):
            injected.append(True)
            if mutation == "member_set":
                filesystem_write_bytes(right / "late.bin", b"late")
            else:
                target = right / (
                    "a.bin" if mutation == "earlier_member" else "manifest.json"
                )
                replacement = tmp_path / "replacement"
                filesystem_write_bytes(
                    replacement,
                    b"CCCC"
                    if mutation == "earlier_member"
                    else filesystem_read_bytes(target) + b" ",
                )
                filesystem_replace(replacement, target)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(storage, "open_filesystem_file", opening)
    assert storage.compare_semantic_artifacts(left, right).status == "INSPECTION_FAILED"
    assert injected == [True]
    assert _immediate_file_bytes(left) == old_good


@pytest.mark.parametrize(
    "change", ("encoding", "row", "schema", "order", "signed_zero")
)
def test_semantic_parquet_comparison_uses_exact_logical_rows(tmp_path, change):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
    )

    table = pa.table({"id": [0, 1, 2], "value": np.array([0.0, float("nan"), 2.0])})
    candidate = table
    if change == "row":
        candidate = pa.table({"id": [0, 1, 2], "value": [0.0, float("nan"), 3.0]})
    elif change == "schema":
        candidate = table.cast(pa.schema([("id", pa.int32()), ("value", pa.float64())]))
    elif change == "order":
        candidate = table.take(pa.array([2, 1, 0]))
    elif change == "signed_zero":
        candidate = pa.table({"id": [0, 1, 2], "value": [-0.0, float("nan"), 2.0]})
    old = write_semantic_artifact(
        tmp_path / "old",
        _envelope(),
        lambda root: pq.write_table(
            table,
            native_filesystem_path(root / "table.parquet"),
            compression="NONE",
            row_group_size=3,
        ),
    )
    new = write_semantic_artifact(
        tmp_path / "new",
        _envelope(),
        lambda root: pq.write_table(
            candidate,
            native_filesystem_path(root / "table.parquet"),
            compression="zstd",
            row_group_size=1,
        ),
    )
    before = filesystem_read_bytes(old / "manifest.json")
    expected = "MATCH" if change == "encoding" else "CONTENT_MISMATCH"
    assert compare_semantic_artifacts(old, new).status == expected
    assert filesystem_read_bytes(old / "manifest.json") == before


def test_identical_parquet_rebuild_compares_logical_values(tmp_path):
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
    )

    table = pa.table({"payload": [b"x" * 4096] * 4096})
    old = write_semantic_artifact(
        tmp_path / "old",
        _envelope(),
        lambda root: pq.write_table(
            table,
            native_filesystem_path(root / "table.parquet"),
            compression="zstd",
            use_dictionary=False,
            data_page_size=32 * 1024**2,
        ),
    )
    new = write_semantic_artifact(
        tmp_path / "new",
        _envelope(),
        lambda root: shutil.copyfile(
            native_filesystem_path(old / "table.parquet"),
            native_filesystem_path(root / "table.parquet"),
        ),
    )
    assert filesystem_stat(old / "table.parquet").st_size < 1024**2
    assert compare_semantic_artifacts(old, new).status == "MATCH"


def test_member_inventory_round_trips_more_than_256_declared_members(tmp_path):
    import quant_research_platform.storage.semantic_artifacts as storage

    def writer(staging):
        for index in range(265):
            filesystem_write_bytes(staging / f"member-{index}.bin", str(index).encode())

    root = write_semantic_artifact(tmp_path / "wide", _envelope(), writer)
    _, members = storage.read_semantic_artifact_record(root)
    assert len(members) == 265
    assert {member["path"] for member in members} == {
        f"member-{i}.bin" for i in range(265)
    }
    filesystem_write_bytes(root / "member-264.bin", b"changed")
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        storage.read_semantic_artifact_record(root)


@pytest.mark.parametrize(
    "value", ["/outside.json", "C:outside.json", "C:/outside.json", "../outside.json"]
)
def test_member_manifest_rejects_windows_anchored_paths(monkeypatch, value):
    from pathlib import PureWindowsPath

    import quant_research_platform.storage.semantic_artifacts as owner

    monkeypatch.setattr(owner, "Path", PureWindowsPath)
    with pytest.raises(ValueError, match="semantic_artifact_member_manifest_invalid"):
        owner._validate_member_payload([{"path": value, "size": 0, "sha256": "0" * 64}])
    assert (
        owner._validate_member_payload(
            [{"path": "safe/file.json", "size": 0, "sha256": "0" * 64}]
        )[0]["path"]
        == "safe/file.json"
    )


@pytest.mark.parametrize("member", ("manifest.json", "model.pkl"))
def test_parent_capture_is_preserved_even_for_same_content_replacement(
    tmp_path, member
):
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_record,
    )

    root = _write_artifact(tmp_path)
    identities = {}
    read_semantic_artifact_record(root, member_identities=identities)
    assert set(identities) == {"manifest.json", "model.json", "model.pkl"}
    read_semantic_artifact_record(root, expected_member_identities=identities)
    path = root / member
    replacement = tmp_path / "replacement"
    filesystem_write_bytes(replacement, filesystem_read_bytes(path))
    filesystem_replace(replacement, path)
    with pytest.raises(ValueError, match="identity_changed"):
        read_semantic_artifact_record(root, expected_member_identities=identities)


@pytest.mark.parametrize(
    "key",
    [
        "/srv/synthetic/report",
        r"C:\synthetic\report",
        r"\\synthetic\share\report",
        "//synthetic/share/report",
        "/",
        "C:\\",
    ],
)
def test_operational_metadata_schema_rejects_absolute_keys_before_write_and_read(
    tmp_path: Path, key: str
) -> None:
    from quant_research_platform.contracts.json_codec import render_qrp_json_object
    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_operational_metadata,
    )

    root = _write_artifact(tmp_path)
    before = _immediate_file_bytes(root)
    before_instances = set(filesystem_iterdir(root.parent))
    assert read_semantic_artifact_operational_metadata(root) == {}
    with pytest.raises(
        ValueError, match="^semantic_artifact_operational_metadata_invalid$"
    ):
        write_semantic_artifact(
            tmp_path / "model",
            _envelope(seed=1),
            lambda staging: filesystem_write_bytes(staging / "model.bin", b"candidate"),
            operational_metadata={key: 1},
        )
    assert set(filesystem_iterdir(root.parent)) == before_instances
    assert _immediate_file_bytes(root) == before

    manifest_path = root / "manifest.json"
    forged = read_manifest(manifest_path)
    forged["operational_metadata"] = {key: 1}
    forged_bytes = (render_qrp_json_object(forged) + "\n").encode("utf-8")
    filesystem_write_bytes(manifest_path, forged_bytes)
    with pytest.raises(
        ValueError, match="^semantic_artifact_operational_metadata_invalid$"
    ):
        read_semantic_artifact_operational_metadata(root)
    assert filesystem_read_bytes(manifest_path) == forged_bytes


@pytest.mark.parametrize("bad", ([], {}))
@pytest.mark.parametrize(
    "path,error",
    (
        (
            ("model_phase_observations", "phases", 0, "phase"),
            "semantic_artifact_operational_metadata_invalid",
        ),
        (
            ("model_phase_observations", "phases", 0, "selection_groups"),
            "semantic_artifact_operational_metadata_invalid",
        ),
        (
            ("model_phase_observations", "phases", 0, "observations", "status"),
            "research_run_operation_evidence_invalid",
        ),
        (
            ("model_phase_observations", "environments", 0, "host_id", "status"),
            "research_run_operation_evidence_invalid",
        ),
        (
            (
                "model_phase_observations",
                "environments",
                0,
                "native_thread_env",
                "OMP_NUM_THREADS",
                "status",
            ),
            "research_run_operation_evidence_invalid",
        ),
        (
            ("final_alpha_observations", "status"),
            "research_run_operation_evidence_invalid",
        ),
    ),
)
def test_operational_metadata_rejects_wrong_enum_types_without_mutating_good(
    tmp_path, path, error, bad
):
    from copy import deepcopy

    from quant_research_platform.contracts.json_codec import render_qrp_json_object
    from quant_research_platform.runtime.execution_metrics import (
        record_run_observations,
    )
    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
        read_semantic_artifact_operational_metadata,
    )

    observations = {}
    with record_run_observations(observations):
        pass
    compressed = deepcopy(observations)
    environment = compressed.pop("environment")
    created_at = environment.pop("created_at_utc")
    compressed["environment"] = {"context": 0, "created_at_utc": created_at}
    valid = {
        "model_phase_observations": {
            "environments": [environment],
            "phases": [
                {
                    "phase": "shared_candidate_fit",
                    "selection_groups": [],
                    "family_id": "ridge",
                    "quarter_id": None,
                    "fit_count": 1,
                    "observations": compressed,
                }
            ],
        },
        "final_alpha_observations": observations,
    }

    def writer(staging):
        filesystem_write_bytes(staging / "model.bin", b"model")

    good = write_semantic_artifact(
        tmp_path / "good", _envelope(), writer, operational_metadata=valid
    )
    before = _immediate_file_bytes(good)
    assert read_semantic_artifact_operational_metadata(good) == valid
    invalid = deepcopy(valid)
    parent = invalid
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = [bad] if path[-1] == "selection_groups" else bad
    with pytest.raises(ValueError, match=f"^{error}$"):
        write_semantic_artifact(
            tmp_path / "good",
            _envelope(seed=1),
            writer,
            operational_metadata=invalid,
        )
    assert tuple(filesystem_iterdir(good.parent)) == (good,)
    assert _immediate_file_bytes(good) == before

    candidate = write_semantic_artifact(
        tmp_path / "candidate", _envelope(), writer, operational_metadata=valid
    )
    assert compare_semantic_artifacts(good, candidate).status == "MATCH"
    forged = read_manifest(candidate / "manifest.json")
    forged["operational_metadata"] = invalid
    forged_bytes = (render_qrp_json_object(forged) + "\n").encode("utf-8")
    filesystem_write_bytes(candidate / "manifest.json", forged_bytes)
    with pytest.raises(ValueError, match=f"^{error}$"):
        read_semantic_artifact_operational_metadata(candidate)
    assert compare_semantic_artifacts(good, candidate).status == "INSPECTION_FAILED"
    assert filesystem_read_bytes(candidate / "manifest.json") == forged_bytes
    assert _immediate_file_bytes(good) == before


def test_known_parquet_schema_publishes_typed_empty_file_without_explicit_rows(
    tmp_path,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.storage.parquet import parquet_table_writer

    schema = pa.schema(
        (
            pa.field("sequence", pa.int64()),
            pa.field("amount", pa.float64()),
            pa.field("asset_id", pa.string()),
            pa.field("timestamp", pa.timestamp("ns")),
        )
    )
    path = tmp_path / "empty.parquet"
    with path.open("xb") as handle:
        with parquet_table_writer(handle, compression="zstd", schema=schema):
            pass
    actual = pq.read_table(path)
    assert actual.equals(pa.Table.from_pylist([], schema=schema), check_metadata=True)


@pytest.mark.parametrize("bad", ([], {}))
@pytest.mark.parametrize(
    "field,error",
    (
        ("status", "trial_status_invalid"),
        ("selection_direction", "experiment_selection_direction_invalid"),
        ("stage", "experiment_stage_invalid"),
        ("selection_groups", "experiment_selection_group_invalid"),
    ),
)
def test_trial_enum_types_reject_in_native_ledger_and_compare_without_mutating_good(
    tmp_path, bad, field, error
):
    from dataclasses import replace

    from quant_research_platform.contracts.experiments import (
        ExperimentPlan,
        TrialRecord,
        trial_ledger_semantic_identity,
    )
    from quant_research_platform.contracts.json_codec import render_qrp_json_object
    from quant_research_platform.storage.experiments import (
        append_trial_record,
        read_trial_ledger,
    )
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
    )
    from tests.research.test_experiment_ledger import _plan, _record

    plan = _plan()
    mother = _record(plan, event_id="event", attempt_index=1, status="attempted")
    for status in ("attempted", "completed", "failed", "abandoned", "invalid"):
        record = replace(mother, status=status)
        assert TrialRecord.from_payload(record.payload()) == record
    for direction in ("maximize", "minimize"):
        selected = replace(plan, selection_direction=direction)
        assert ExperimentPlan.from_payload(selected.protocol_payload()) == selected
    decoder = (
        TrialRecord.from_payload if field == "status" else ExperimentPlan.from_payload
    )
    payload = mother.payload() if field == "status" else plan.protocol_payload()
    payload[field] = [bad] if field == "selection_groups" else bad
    with pytest.raises(ValueError, match=f"^{error}$"):
        decoder(payload)

    layout = StorageLayout(tmp_path)

    def publish(name):
        record = replace(mother, event_id=f"event_{name}", run_id=f"run_{name}")
        ledger = layout.trial_ledger_root(record.run_id, plan.experiment_id)
        append_trial_record(ledger, plan, record)
        assert read_trial_ledger(ledger) == (plan, (record,))
        reference = {
            "run_id": record.run_id,
            "experiment_id": plan.experiment_id,
            "protocol_hash": plan.protocol_hash,
            "semantic_identity": trial_ledger_semantic_identity(plan, (record,)),
        }
        artifact = write_semantic_artifact(
            tmp_path / name,
            _envelope(),
            lambda stage: filesystem_write_bytes(stage / "model.bin", b"model"),
            operational_metadata={"model_trial_ledgers": [reference]},
        )
        return artifact, ledger

    good, good_ledger = publish("good")
    candidate, candidate_ledger = publish("candidate")
    artifact_before = _immediate_file_bytes(good)
    ledger_before = {
        path.relative_to(good_ledger): path.read_bytes()
        for path in good_ledger.rglob("*")
        if path.is_file()
    }
    assert (
        compare_semantic_artifacts(good, candidate, trial_layout=layout).status
        == "MATCH"
    )
    changed = (
        candidate_ledger / "records" / "00000001.json"
        if field == "status"
        else candidate_ledger / "plan.json"
    )
    forged = read_manifest(changed)
    forged[field] = [bad] if field == "selection_groups" else bad
    forged_bytes = (render_qrp_json_object(forged) + "\n").encode("utf-8")
    filesystem_write_bytes(changed, forged_bytes)
    with pytest.raises(ValueError, match=f"^{error}$"):
        read_trial_ledger(candidate_ledger)
    assert (
        compare_semantic_artifacts(good, candidate, trial_layout=layout).status
        == "INSPECTION_FAILED"
    )
    assert filesystem_read_bytes(changed) == forged_bytes
    assert _immediate_file_bytes(good) == artifact_before
    assert {
        path.relative_to(good_ledger): path.read_bytes()
        for path in good_ledger.rglob("*")
        if path.is_file()
    } == ledger_before


def _metric_operational_metadata(value):
    from copy import deepcopy

    from quant_research_platform.runtime.execution_metrics import (
        record_run_observations,
    )

    observations = {}
    with record_run_observations(observations):
        pass
    observations["status"] = "DEFINED"
    observations["reason"] = ""
    observations["metrics"] = {
        key: {"status": "DEFINED", "value": value} for key in observations["metrics"]
    }
    compressed = deepcopy(observations)
    environment = compressed.pop("environment")
    created_at = environment.pop("created_at_utc")
    compressed["environment"] = {"context": 0, "created_at_utc": created_at}
    return {
        "model_phase_observations": {
            "environments": [environment],
            "phases": [
                {
                    "phase": "shared_candidate_fit",
                    "selection_groups": [],
                    "family_id": "ridge",
                    "quarter_id": None,
                    "fit_count": 1,
                    "observations": compressed,
                }
            ],
        },
        "final_alpha_observations": observations,
    }


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        -0.0,
        1.25,
        2**53 + 1,
        10**400 + 1,
        10**4096,
        float.fromhex("0x1.fffffffffffffp+1023"),
    ],
    ids=[
        "integer_zero",
        "float_zero",
        "signed_zero",
        "finite_float",
        "exact_integer",
        "large_integer",
        "above_former_integer_limit",
        "maximum_float",
    ],
)
def test_operational_metrics_preserve_numeric_type_and_precision_in_native_artifacts(
    tmp_path, value
):
    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
        read_semantic_artifact_operational_metadata,
    )

    metadata = _metric_operational_metadata(value)

    def writer(staging):
        filesystem_write_bytes(staging / "model.bin", b"model")

    good = write_semantic_artifact(
        tmp_path / "good", _envelope(), writer, operational_metadata=metadata
    )
    before = _immediate_file_bytes(good)
    decoded = read_semantic_artifact_operational_metadata(good)
    assert decoded == metadata
    for observations in (
        decoded["final_alpha_observations"],
        decoded["model_phase_observations"]["phases"][0]["observations"],
    ):
        for metric in observations["metrics"].values():
            assert type(metric["value"]) is type(value)
            assert metric["value"] == value
            if type(value) is float:
                assert metric["value"].hex() == value.hex()
    candidate = write_semantic_artifact(
        tmp_path / "candidate", _envelope(), writer, operational_metadata=metadata
    )
    assert compare_semantic_artifacts(good, candidate).status == "MATCH"
    assert _immediate_file_bytes(good) == before


@pytest.mark.parametrize(
    "branch", ["model_phase_observations", "final_alpha_observations"]
)
@pytest.mark.parametrize(
    "bad,error",
    [
        (True, "research_run_operation_evidence_invalid"),
        (False, "research_run_operation_evidence_invalid"),
        (-1, "research_run_operation_evidence_invalid"),
        (-0.5, "research_run_operation_evidence_invalid"),
        (-(10**400), "research_run_operation_evidence_invalid"),
        (None, "research_run_operation_evidence_invalid"),
        (float("nan"), "qrp_json_nonfinite_number"),
        (float("inf"), "qrp_json_nonfinite_number"),
        (-float("inf"), "qrp_json_nonfinite_number"),
    ],
    ids=[
        "true",
        "false",
        "negative_integer",
        "negative_float",
        "negative_large_integer",
        "null",
        "nan",
        "positive_infinity",
        "negative_infinity",
    ],
)
def test_invalid_operational_metrics_reject_before_publication_and_native_compare(
    tmp_path, branch, bad, error
):
    import json
    from copy import deepcopy

    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
        read_semantic_artifact_operational_metadata,
    )

    valid = _metric_operational_metadata(1)

    def writer(staging):
        filesystem_write_bytes(staging / "model.bin", b"model")

    good = write_semantic_artifact(
        tmp_path / "good", _envelope(), writer, operational_metadata=valid
    )
    before = _immediate_file_bytes(good)
    instances = tuple(filesystem_iterdir(good.parent))
    invalid = deepcopy(valid)
    observations = invalid[branch]
    if branch == "model_phase_observations":
        observations = observations["phases"][0]["observations"]
    observations["metrics"]["read_bytes"]["value"] = bad
    with pytest.raises(ValueError, match=f"^{error}$"):
        write_semantic_artifact(
            tmp_path / "good",
            _envelope(seed=1),
            writer,
            operational_metadata=invalid,
        )
    assert tuple(filesystem_iterdir(good.parent)) == instances
    assert _immediate_file_bytes(good) == before

    candidate = write_semantic_artifact(
        tmp_path / "candidate", _envelope(), writer, operational_metadata=valid
    )
    manifest_path = candidate / "manifest.json"
    forged = read_manifest(manifest_path)
    forged["operational_metadata"] = invalid
    raw = (json.dumps(forged, ensure_ascii=False) + "\n").encode("utf-8")
    filesystem_write_bytes(manifest_path, raw)
    with pytest.raises(ValueError, match=f"^{error}$"):
        read_semantic_artifact_operational_metadata(candidate)
    assert compare_semantic_artifacts(good, candidate).status == "INSPECTION_FAILED"
    assert filesystem_read_bytes(manifest_path) == raw
    assert _immediate_file_bytes(good) == before


def test_operational_observations_roundtrip_above_former_byte_quota(tmp_path):
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_operational_metadata,
    )

    metadata = {
        "final_alpha_observations": _metric_operational_metadata(1)[
            "final_alpha_observations"
        ]
    }
    metadata["final_alpha_observations"]["environment"]["platform"] += "x" * 65537
    good = write_semantic_artifact(
        tmp_path / "good",
        _envelope(),
        lambda staging: filesystem_write_bytes(staging / "model.bin", b"model"),
        operational_metadata=metadata,
    )
    before = _immediate_file_bytes(good)
    assert read_semantic_artifact_operational_metadata(good) == metadata
    assert _immediate_file_bytes(good) == before

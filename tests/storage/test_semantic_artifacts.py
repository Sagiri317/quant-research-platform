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


def _envelope() -> object:
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
        semantic_metadata={},
    )


def _write_artifact(tmp_path: Path) -> Path:
    def writer(staging: Path) -> None:
        filesystem_write_bytes(staging / "model.json", b"model")
        filesystem_write_bytes(staging / "model.pkl", b"predictor")

    return write_semantic_artifact(tmp_path / "model", _envelope(), writer)


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
    for candidate in (build("parameter", alpha=2.0), build("metric", metric=0.3)):
        assert (
            storage.compare_semantic_artifacts(
                old, candidate, trial_layout=layout
            ).status
            == "CONTENT_MISMATCH"
        )
    assert filesystem_read_bytes(old / "manifest.json") == before

    import hashlib

    real_sha256 = hashlib.sha256

    class Collision:
        def update(self, _value):
            pass

        def hexdigest(self):
            return "a" * 64

    # Only streaming member hashes collide; the semantic envelope hash is genuine.
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


def test_semantic_compare_read_failure_keeps_primary_when_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_research_platform.storage.semantic_artifacts as storage

    left, right = (
        _write_artifact(tmp_path / "left"),
        _write_artifact(tmp_path / "right"),
    )
    primary = KeyboardInterrupt("test primary")
    original_open = storage.open_filesystem_file

    class BrokenRead:
        def __init__(self, handle):
            self.handle = handle

        def fileno(self):
            return self.handle.fileno()

        def read(self, _size):
            raise primary

        def close(self):
            self.handle.close()
            raise RuntimeError("close failed")

    monkeypatch.setattr(
        storage, "open_filesystem_file", lambda path: BrokenRead(original_open(path))
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        storage.compare_semantic_artifacts(left, right)
    assert caught.value is primary
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
            if counts["next"] == 10_000:
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

    # Restrict the wrapper to inventory: atomic publication owns its own scans.
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
    assert closed == [
        root / "nested",
        root,
    ]
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


def test_inventory_limit_failure_preserves_published_artifact_and_parent(
    tmp_path, monkeypatch
):
    import quant_research_platform.storage.semantic_artifacts as storage
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    root = _write_artifact(tmp_path)
    old_good = _immediate_file_bytes(root)
    before = set(filesystem_iterdir(root.parent))
    candidate_fields = _envelope().payload()
    del candidate_fields["instance_identity"], candidate_fields["status"]
    candidate_fields["semantic_metadata"] = {"candidate": True}
    candidate = build_business_manifest_envelope(**candidate_fields)
    monkeypatch.setattr(storage, "MAX_QRP_JSON_BYTES", 2048)

    def writer(staging):
        for index in range(257):
            filesystem_write_bytes(staging / f"member-{index}.bin", b"small")

    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="semantic_artifact_member_limit_exceeded"):
            write_semantic_artifact(tmp_path / "model", candidate, writer)
        assert policy.used_memory_bytes == 0
    assert set(filesystem_iterdir(root.parent)) == before
    assert _immediate_file_bytes(root) == old_good


@pytest.mark.parametrize("operation", ("discovery", "publication"))
def test_constant_directory_identity_cannot_hide_added_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    import quant_research_platform.storage.semantic_artifacts as storage
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    root = _write_artifact(tmp_path)
    old_good = _immediate_file_bytes(root)
    before = set(filesystem_iterdir(root.parent))
    original_identity = storage._directory_identity
    observed, calls = {}, {}

    def identity(path):
        if path not in observed:
            observed[path] = original_identity(path)
        calls[path] = calls.get(path, 0) + 1
        if operation == "publication" and calls[path] == 2:
            filesystem_write_bytes(path / "late.bin", b"late")
        return observed[path]

    monkeypatch.setattr(storage, "_directory_identity", identity)
    with runtime_resource_policy_scope() as policy:
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

            with pytest.raises(
                ValueError, match="semantic_artifact_member_set_changed"
            ):
                storage.write_semantic_artifact(tmp_path / "model", candidate, writer)
            assert set(filesystem_iterdir(root.parent)) == before
        assert policy.used_memory_bytes == 0
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
            and inspect.currentframe().f_back.f_code.co_name
            == "compare_semantic_artifacts"
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


def test_identical_parquet_rebuild_does_not_require_decoded_payload_capacity(tmp_path):
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
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
    with runtime_resource_policy_scope(explicit_cap_bytes=1024**2) as policy:
        assert compare_semantic_artifacts(old, new).status == "MATCH"
        assert policy.used_memory_bytes == 0


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
    ("available", "cap", "error", "message"),
    (
        (512 * 1024**2, 4 * 1024**2, MemoryError, "runtime_memory_budget_exceeded"),
        (
            2 * 1024**2,
            None,
            ValueError,
            "semantic_artifact_metadata_exceeds_memory_capacity",
        ),
        (
            2 * 1024**2,
            64 * 1024**2,
            ValueError,
            "semantic_artifact_metadata_exceeds_memory_capacity",
        ),
    ),
)
def test_semantic_writer_admits_envelope_projection_before_publication(
    tmp_path, monkeypatch, available, cap, error, message
):
    import quant_research_platform.storage.semantic_artifacts as storage
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    old_root = _write_artifact(tmp_path)
    old_bytes = _immediate_file_bytes(old_root)
    payload = _envelope().payload()
    envelope = build_business_manifest_envelope(
        **{
            key: (
                {"counts": [{} for _ in range(20000)]}
                if key == "semantic_metadata"
                else value
            )
            for key, value in payload.items()
            if key not in {"instance_identity", "status"}
        }
    )
    writes = 0
    original = storage.write_manifest

    def write_manifest(path, value, **kwargs):
        nonlocal writes
        writes += 1
        return original(path, value, **kwargs)

    monkeypatch.setattr(storage, "write_manifest", write_manifest)
    policy = capture_test_policy(
        available_memory_bytes=available, explicit_cap_bytes=cap
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(error, match=message):
            storage.write_semantic_artifact(
                old_root.parent.parent,
                envelope,
                lambda staging: filesystem_write_bytes(staging / "member", b"data"),
            )
        assert policy.used_memory_bytes == 0
    assert writes == 0
    assert not (old_root.parent / envelope.instance_identity).exists()
    assert _immediate_file_bytes(old_root) == old_bytes
    assert len(envelope.semantic_metadata["counts"]) == 20000

    if cap is None:
        policy = capture_test_policy(available_memory_bytes=256 * 1024**2)
        with runtime_resource_policy_scope(policy):
            published = storage.write_semantic_artifact(
                old_root.parent.parent,
                envelope,
                lambda staging: filesystem_write_bytes(staging / "member", b"data"),
            )
            assert policy.used_memory_bytes == 0
        assert writes == 1
        assert published.name == envelope.instance_identity
        assert (
            len(read_semantic_artifact(published)["semantic_metadata"]["counts"])
            == 20000
        )
        assert _immediate_file_bytes(old_root) == old_bytes


@pytest.mark.parametrize("cap", (None, 128 * 1024**2))
def test_semantic_writer_failure_releases_metadata_with_retained_exception(
    tmp_path, monkeypatch, cap
):
    import quant_research_platform.storage.semantic_artifacts as storage
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    old_root = _write_artifact(tmp_path)
    old_bytes = _immediate_file_bytes(old_root)
    policy = capture_test_policy(
        available_memory_bytes=512 * 1024**2, explicit_cap_bytes=cap
    )
    failure = RuntimeError("semantic_manifest_boundary_primary")
    manifest_id = None

    def fail_manifest(path, payload, **kwargs):
        nonlocal manifest_id
        manifest_id = id(payload)
        assert any(
            lease.owner == "storage.semantic.write.metadata"
            and lease.reserved_bytes > 0
            for lease in policy._leases
        )
        raise failure

    monkeypatch.setattr(storage, "write_manifest", fail_manifest)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(RuntimeError) as caught:
            storage.write_semantic_artifact(
                tmp_path / "candidate",
                _envelope(),
                lambda staging: filesystem_write_bytes(staging / "member", b"data"),
            )
        assert caught.value is failure
        current = failure.__traceback__
        while current is not None:
            local_values = current.tb_frame.f_locals
            if current.tb_frame.f_code.co_name == "fail_manifest":
                assert not local_values
            assert all(id(value) != manifest_id for value in local_values.values())
            current = current.tb_next
        assert policy.used_memory_bytes == 0
        with policy.lease("test.semantic.after_failure", 64 * 1024**2):
            pass
    assert _immediate_file_bytes(old_root) == old_bytes
    assert not (
        tmp_path / "candidate" / "instances" / _envelope().instance_identity
    ).exists()

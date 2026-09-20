from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.storage.semantic_artifacts import (
    discover_semantic_artifact_instances,
    read_semantic_artifact,
    write_semantic_artifact,
)


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
        (staging / "model.json").write_bytes(b"model")
        (staging / "model.pkl").write_bytes(b"predictor")

    return write_semantic_artifact(tmp_path / "model", _envelope(), writer)


def test_semantic_artifact_member_mutation_is_rejected_by_read_and_discovery(
    tmp_path: Path,
) -> None:
    root = _write_artifact(tmp_path)
    (root / "model.pkl").write_bytes(b"changed")

    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        read_semantic_artifact(root)
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        discover_semantic_artifact_instances(tmp_path / "model")


def test_semantic_artifact_extra_member_is_rejected(tmp_path: Path) -> None:
    root = _write_artifact(tmp_path)
    (root / "extra.bin").write_bytes(b"extra")

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
            validation_metrics={"rankic": metric},
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
            lambda staging: (staging / "model.bin").write_bytes(b"same model"),
            operational_metadata={"model_trial_ledgers": [reference]},
        )

    old = build("first")
    same = build("retry", threads=2)
    before = (old / "manifest.json").read_bytes()
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
    assert (old / "manifest.json").read_bytes() == before

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
        lambda staging: (staging / "data.bin").write_bytes(b"AAAA"),
    )
    right = storage.write_semantic_artifact(
        tmp_path / "collision_right",
        _envelope(),
        lambda staging: (staging / "data.bin").write_bytes(b"BBBB"),
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

    monkeypatch.setattr(storage.os, "scandir", lambda _path: Entries())
    monkeypatch.setattr(storage, "_member_record", record)
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        storage.read_semantic_artifact_record(root)
    assert counts == {"next": 1, "hash": 0, "close": 1}


def test_nested_inventory_descends_before_reading_next_sibling(tmp_path, monkeypatch):
    import quant_research_platform.storage.semantic_artifacts as storage

    def writer(staging):
        for index in range(50):
            (staging / f"empty-{index}").mkdir()
        nested = staging / "z" / "inner"
        nested.mkdir(parents=True)
        (nested / "model.bin").write_bytes(b"nested")
        (staging / "a.bin").write_bytes(b"root")

    actual_scandir = storage.os.scandir
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
            scoped.setattr(storage.os, "scandir", Entries)
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
        (staging / "nested").mkdir()
        (staging / "nested" / "bad.bin").write_bytes(b"bad")

    root = write_semantic_artifact(tmp_path / "model", _envelope(), writer)
    primary = KeyboardInterrupt("inventory primary")
    actual_scandir = storage.os.scandir
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

    def record(relative, path):
        if relative == "nested/bad.bin":
            raise primary
        return actual_record(relative, path)

    monkeypatch.setattr(storage.os, "scandir", Entries)
    monkeypatch.setattr(storage, "_member_record", record)
    with pytest.raises(KeyboardInterrupt) as caught:
        storage.read_semantic_artifact_record(root)
    assert caught.value is primary
    assert closed == [
        Path(storage.native_filesystem_path(root / "nested")),
        Path(storage.native_filesystem_path(root)),
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
    external.mkdir()
    payload = external / "outside.bin"
    payload.write_bytes(b"external synthetic content")
    alias = root / "alias"
    if alias_kind == "symlink":
        try:
            alias.symlink_to(external, target_is_directory=True)
        except OSError:
            pytest.skip("symlink_not_available")
    else:
        if os.name != "nt" or not hasattr(Path, "is_junction"):
            pytest.skip("junction_capability_required")
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(external)],
            check=False,
            capture_output=True,
        )
        if created.returncode != 0:
            pytest.skip("junction_not_available")
    actual_scandir = storage.os.scandir

    def scan(path):
        assert Path(path) == Path(storage.native_filesystem_path(root))
        return actual_scandir(path)

    monkeypatch.setattr(storage.os, "scandir", scan)
    with pytest.raises(ValueError, match="semantic_artifact_member_invalid"):
        storage.read_semantic_artifact_record(root)
    assert payload.read_bytes() == b"external synthetic content"


def test_inventory_limit_failure_preserves_published_artifact_and_parent(
    tmp_path, monkeypatch
):
    import quant_research_platform.storage.semantic_artifacts as storage
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    root = _write_artifact(tmp_path)
    old_good = {path.name: path.read_bytes() for path in root.iterdir()}
    before = set(root.parent.iterdir())
    candidate_fields = _envelope().payload()
    del candidate_fields["instance_identity"], candidate_fields["status"]
    candidate_fields["semantic_metadata"] = {"candidate": True}
    candidate = build_business_manifest_envelope(**candidate_fields)
    monkeypatch.setattr(storage, "_MAX_QRP_JSON_BYTES", 2048)

    def writer(staging):
        for index in range(257):
            (staging / f"member-{index}.bin").write_bytes(b"small")

    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="semantic_artifact_member_limit_exceeded"):
            write_semantic_artifact(tmp_path / "model", candidate, writer)
        assert policy.used_memory_bytes == 0
    assert set(root.parent.iterdir()) == before
    assert {path.name: path.read_bytes() for path in root.iterdir()} == old_good


@pytest.mark.parametrize("operation", ("discovery", "publication"))
def test_constant_directory_identity_cannot_hide_added_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    import quant_research_platform.storage.semantic_artifacts as storage
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    root = _write_artifact(tmp_path)
    old_good = {path.name: path.read_bytes() for path in root.iterdir()}
    before = set(root.parent.iterdir())
    original_identity = storage._directory_identity
    observed, calls = {}, {}

    def identity(path):
        if path not in observed:
            observed[path] = original_identity(path)
        calls[path] = calls.get(path, 0) + 1
        if operation == "publication" and calls[path] == 2:
            (path / "late.bin").write_bytes(b"late")
        return observed[path]

    monkeypatch.setattr(storage, "_directory_identity", identity)
    with runtime_resource_policy_scope() as policy:
        if operation == "discovery":
            original_read = storage.read_semantic_artifact

            def read(path, **kwargs):
                (root.parent / "late").write_bytes(b"late")
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
                (staging / "empty").mkdir()
                (staging / "data.bin").write_bytes(b"data")

            with pytest.raises(
                ValueError, match="semantic_artifact_member_set_changed"
            ):
                storage.write_semantic_artifact(tmp_path / "model", candidate, writer)
            assert set(root.parent.iterdir()) == before
        assert policy.used_memory_bytes == 0
    assert {path.name: path.read_bytes() for path in root.iterdir()} == old_good


@pytest.mark.parametrize("mutation", ("earlier_member", "member_set", "manifest"))
def test_comparison_rechecks_the_complete_round_before_match(
    tmp_path, monkeypatch, mutation
):
    import inspect

    import quant_research_platform.storage.semantic_artifacts as storage

    def writer(root):
        (root / "a.bin").write_bytes(b"AAAA")
        (root / "b.bin").write_bytes(b"BBBB")

    left = storage.write_semantic_artifact(tmp_path / "left", _envelope(), writer)
    right = storage.write_semantic_artifact(tmp_path / "right", _envelope(), writer)
    old_good = {path.name: path.read_bytes() for path in left.iterdir()}
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
                (right / "late.bin").write_bytes(b"late")
            else:
                target = right / (
                    "a.bin" if mutation == "earlier_member" else "manifest.json"
                )
                replacement = tmp_path / "replacement"
                replacement.write_bytes(
                    b"CCCC"
                    if mutation == "earlier_member"
                    else target.read_bytes() + b" "
                )
                replacement.replace(target)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(storage, "open_filesystem_file", opening)
    assert storage.compare_semantic_artifacts(left, right).status == "INSPECTION_FAILED"
    assert injected == [True]
    assert {path.name: path.read_bytes() for path in left.iterdir()} == old_good


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
            table, root / "table.parquet", compression="NONE", row_group_size=3
        ),
    )
    new = write_semantic_artifact(
        tmp_path / "new",
        _envelope(),
        lambda root: pq.write_table(
            candidate, root / "table.parquet", compression="zstd", row_group_size=1
        ),
    )
    before = (old / "manifest.json").read_bytes()
    expected = "MATCH" if change == "encoding" else "CONTENT_MISMATCH"
    assert compare_semantic_artifacts(old, new).status == expected
    assert (old / "manifest.json").read_bytes() == before


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
            root / "table.parquet",
            compression="zstd",
            use_dictionary=False,
            data_page_size=32 * 1024**2,
        ),
    )
    new = write_semantic_artifact(
        tmp_path / "new",
        _envelope(),
        lambda root: shutil.copyfile(old / "table.parquet", root / "table.parquet"),
    )
    assert (old / "table.parquet").stat().st_size < 1024**2
    with runtime_resource_policy_scope(explicit_cap_bytes=1024**2) as policy:
        assert compare_semantic_artifacts(old, new).status == "MATCH"
        assert policy.used_memory_bytes == 0


def test_member_inventory_round_trips_more_than_256_declared_members(tmp_path):
    import quant_research_platform.storage.semantic_artifacts as storage

    def writer(staging):
        for index in range(265):
            (staging / f"member-{index}.bin").write_bytes(str(index).encode())

    root = write_semantic_artifact(tmp_path / "wide", _envelope(), writer)
    _, members = storage.read_semantic_artifact_record(root)
    assert len(members) == 265
    assert {member["path"] for member in members} == {
        f"member-{i}.bin" for i in range(265)
    }
    (root / "member-264.bin").write_bytes(b"changed")
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

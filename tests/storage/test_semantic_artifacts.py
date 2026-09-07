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

    real_sha256 = storage.hashlib.sha256

    class Collision:
        def update(self, _value):
            pass

        def hexdigest(self):
            return "a" * 64

    # Only streaming member hashes collide; the semantic envelope hash is genuine.
    monkeypatch.setattr(
        storage.hashlib,
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

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import quant_research_platform.artifacts.outputs.bundle_validation as bundle_validation
from quant_research_platform.artifacts.outputs.bundle_validation import (
    validate_output_bundle,
)
from quant_research_platform.artifacts.outputs.manifest import (
    output_manifest_payload,
    prediction_output_metadata,
)
from quant_research_platform.storage.identity import dataset_file_identity
from tests.helpers.output_frames import (
    make_prediction_report_spec,
    make_price_volume_prediction_output_frame,
    make_price_volume_prediction_spec,
)


def test_prediction_bundle_validates_and_filters_nanosecond_bounds(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.layout import StorageLayout

    frame = make_price_volume_prediction_output_frame()
    frame = frame.loc[frame.index.repeat(3)].reset_index(drop=True)
    times = pd.Timestamp("2014-01-03") + pd.to_timedelta([1, 2, 3], unit="ns")
    frame["timestamp"] = times
    frame["prediction_score"] = [1.0, 2.0, 3.0]
    spec = replace(
        make_price_volume_prediction_spec(),
        train_end="2014-01-02T00:00:00",
        test_start=times[0].isoformat(),
        test_end=times[-1].isoformat(),
    )
    model = dict(frame.attrs["model_metadata"])
    model.update(
        train_start="2014-01-02T00:00:00",
        train_end=spec.train_end,
        test_start=spec.test_start,
        test_end=spec.test_end,
        oos_start=spec.test_start,
        oos_end=spec.test_end,
    )
    metadata = dict(frame.attrs["prediction_metadata"])
    metadata.update(
        train_end=spec.train_end,
        rows_input=3,
        rows_after_filter_mode=3,
        rows_after_input_dropna=3,
        prediction_rows=3,
        test_start=spec.test_start,
        requested_test_end=spec.test_end,
        effective_test_end=spec.test_end,
    )
    with runtime_resource_policy_scope() as policy:
        root = write_prediction_outputs(
            spec=spec,
            output_spec=make_prediction_report_spec(),
            layout=StorageLayout(tmp_path),
            output_chunks=(frame,),
            model_metadata=model,
            prediction_metadata_factory=lambda: metadata,
        )
        with bundle_validation.read_prediction_output_bundle(
            root,
            expected_output_id=spec.prediction_id,
            time_window=TimeWindow(times[1], times[1]),
        ) as (_manifest, lookup):
            with lookup.values_for_frame(frame, mark_matched=True) as (values, found):
                assert list(found) == [0, 1, 0]
                assert values[1] == 2.0
        path = root / "prediction.parquet"
        with pq.ParquetFile(path) as parquet:
            original = parquet.read()
        for position, delta in ((0, -1), (2, 1)):
            changed = list(times)
            changed[position] += pd.Timedelta(delta, unit="ns")
            altered = original.set_column(
                original.schema.get_field_index("timestamp"),
                original.schema.field("timestamp"),
                pa.array(changed, type=pa.timestamp("ns")),
            )
            pq.write_table(altered, path)
            with pytest.raises(
                ValueError, match="persisted_prediction_window_mismatch"
            ):
                with bundle_validation.read_prediction_output_bundle(
                    root, expected_output_id=spec.prediction_id
                ):
                    pass
            assert policy.used_memory_bytes == 0


def test_prediction_plain_dictionary_reader_bounds_native_expansion(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    import pandas as pd
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.layout import StorageLayout

    frame = make_price_volume_prediction_output_frame()
    frame = frame.loc[frame.index.repeat(256)].reset_index(drop=True)
    times = pd.Timestamp("2014-01-03") + pd.to_timedelta(range(256), unit="ns")
    frame["timestamp"] = times
    frame["asset_id"] = "x" * (32 * 1024)
    frame["prediction_score"] = list(range(256))
    spec = replace(
        make_price_volume_prediction_spec(),
        train_end="2014-01-02T00:00:00",
        test_start=times[0].isoformat(),
        test_end=times[-1].isoformat(),
    )
    model = dict(frame.attrs["model_metadata"])
    model.update(
        train_start="2014-01-02T00:00:00",
        train_end=spec.train_end,
        test_start=spec.test_start,
        test_end=spec.test_end,
        oos_start=spec.test_start,
        oos_end=spec.test_end,
    )
    metadata = dict(frame.attrs["prediction_metadata"])
    metadata.update(
        train_end=spec.train_end,
        rows_input=256,
        rows_after_filter_mode=256,
        rows_after_input_dropna=256,
        prediction_rows=256,
        test_start=spec.test_start,
        requested_test_end=spec.test_end,
        effective_test_end=spec.test_end,
    )
    with runtime_resource_policy_scope():
        root = write_prediction_outputs(
            spec=spec,
            output_spec=make_prediction_report_spec(),
            layout=StorageLayout(tmp_path),
            output_chunks=(frame,),
            model_metadata=model,
            prediction_metadata_factory=lambda: metadata,
        )
    path = root / "prediction.parquet"
    table = pq.read_table(path)
    pq.write_table(table, path, use_dictionary=True, write_statistics=False)
    table = None
    native_batches = pq.ParquetFile.iter_batches
    pulled = []

    def observed_batches(parquet, **kwargs):
        for batch in native_batches(parquet, **kwargs):
            pulled.append(batch.get_total_buffer_size())
            assert pulled[-1] <= 4 * 1024**2
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", observed_batches)
    with runtime_resource_policy_scope(explicit_cap_bytes=4 * 1024**2) as policy:
        with bundle_validation.read_prediction_output_bundle(
            root, expected_output_id=spec.prediction_id
        ) as (_manifest, lookup):
            assert lookup.row_count == 256
            with lookup.values_for_frame(frame.iloc[[0, 255]], mark_matched=False) as (
                values,
                found,
            ):
                assert list(found) == [1, 1]
                assert list(values) == [0.0, 255.0]
        assert policy.used_memory_bytes == 0
    assert pulled and len(pulled) > 1


def _write_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    frame = make_price_volume_prediction_output_frame()
    files = ("prediction.parquet",)
    payload = output_manifest_payload(
        output_spec=make_prediction_report_spec(),
        metadata=prediction_output_metadata(
            spec=make_price_volume_prediction_spec(),
            model_metadata=frame.attrs["model_metadata"],
            prediction_metadata=frame.attrs["prediction_metadata"],
        ),
        files=files,
    )
    (root / "manifest.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    for name in files:
        (root / name).write_bytes(name.encode("utf-8"))


def test_validate_output_bundle_returns_declared_member_order(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)

    manifest, members = validate_output_bundle(
        root,
        expected_run_id="test_run",
        expected_output_id="pred-run",
    )

    assert manifest["files"] == ["prediction.parquet"]
    assert members == (
        (root / "manifest.json", dataset_file_identity(root / "manifest.json")),
        (
            root / "prediction.parquet",
            dataset_file_identity(root / "prediction.parquet"),
        ),
    )


def test_validate_output_bundle_accepts_local_only_manifest_value(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["research_dataset_id"] = "data/raw/private"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    manifest, members = validate_output_bundle(root)

    assert manifest["metadata"]["research_dataset_id"] == "data/raw/private"
    assert members[0][0] == manifest_path


def test_validate_output_bundle_rejects_alias_above_bundle_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    bundle = target / "nested" / "bundle"
    _write_bundle(bundle)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="output_bundle_root_invalid"):
        validate_output_bundle(alias / "nested" / "bundle")


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_validate_output_bundle_requires_exact_member_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    if mutation == "missing":
        (root / "prediction.parquet").unlink()
    else:
        (root / "extra.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        validate_output_bundle(root)


def test_validate_output_bundle_rejects_declared_hardlink_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["files"] = ["prediction.parquet", "prediction-copy.parquet"]
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        bundle_validation,
        "validate_output_manifest_business_envelope",
        lambda value: value,
    )
    try:
        os.link(root / "prediction.parquet", root / "prediction-copy.parquet")
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    with pytest.raises(ValueError, match="output_member_physical_identity_invalid"):
        validate_output_bundle(root)


def test_validate_output_bundle_accepts_external_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    try:
        os.link(root / "prediction.parquet", tmp_path / "external.bin")
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    manifest, members = validate_output_bundle(root)

    assert manifest["files"] == ["prediction.parquet"]
    assert members[1][0] == root / "prediction.parquet"


def test_validate_output_bundle_rejects_symlinked_manifest(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    external = tmp_path / "external-manifest.json"
    external.write_bytes((root / "manifest.json").read_bytes())
    (root / "manifest.json").unlink()
    try:
        (root / "manifest.json").symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="output_manifest_physical_identity_invalid"):
        validate_output_bundle(root)


@pytest.mark.parametrize("mutation", ("add", "delete", "rename"))
def test_validate_output_bundle_rechecks_exact_member_set_at_operation_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    validate_root = bundle_validation._validated_bundle_root
    calls = 0

    def mutate_before_final_gate(candidate: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            member = root / "prediction.parquet"
            if mutation == "add":
                (root / "late.bin").write_bytes(b"late")
            elif mutation == "delete":
                member.unlink()
            else:
                member.rename(root / "renamed.csv")
        return validate_root(candidate)

    monkeypatch.setattr(
        bundle_validation,
        "_validated_bundle_root",
        mutate_before_final_gate,
    )

    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        validate_output_bundle(root)


def test_validate_output_bundle_rejects_same_path_root_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "moved"
    _write_bundle(root)
    _write_bundle(replacement)
    validate_root = bundle_validation._validated_bundle_root
    calls = 0

    def replace_after_initial_capture(candidate: Path):  # noqa: ANN202
        nonlocal calls
        calls += 1
        validated = validate_root(candidate)
        if calls == 1:
            root.rename(moved)
            replacement.rename(root)
        return validated

    monkeypatch.setattr(
        bundle_validation,
        "_validated_bundle_root",
        replace_after_initial_capture,
    )

    with pytest.raises(ValueError, match="output_bundle_identity_changed"):
        validate_output_bundle(root)

    assert calls == 2


def test_validate_output_bundle_runs_observer_once_between_capture_and_final_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    validate_root = bundle_validation._validated_bundle_root
    require_members = bundle_validation._require_exact_member_set
    events: list[str] = []

    def observed_root(candidate: Path):  # noqa: ANN202
        events.append("root")
        return validate_root(candidate)

    def observed_members(candidate: Path, expected: tuple[str, ...]) -> None:
        events.append("members")
        require_members(candidate, expected)

    observer_hits: list[tuple[dict[str, object], tuple[object, ...]]] = []

    def observer(
        payload: dict[str, object],
        members: tuple[object, ...],
    ) -> None:
        events.append("observer")
        observer_hits.append((payload, members))

    monkeypatch.setattr(bundle_validation, "_validated_bundle_root", observed_root)
    monkeypatch.setattr(
        bundle_validation,
        "_require_exact_member_set",
        observed_members,
    )

    payload, members = validate_output_bundle(root, observer=observer)

    assert {
        "events": events,
        "observer_hits": len(observer_hits),
        "payload_is_same": observer_hits[0][0] is payload,
        "members_are_same": observer_hits[0][1] == members,
    } == {
        "events": ["root", "members", "observer", "root", "members"],
        "observer_hits": 1,
        "payload_is_same": True,
        "members_are_same": True,
    }


@pytest.mark.parametrize(
    "mutation",
    ("add", "delete", "rename", "replace", "root_replace", "parquet_replace"),
)
def test_validate_output_bundle_final_recheck_rejects_observer_window_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    if mutation == "parquet_replace":
        manifest_path = root / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["files"] = ["member.parquet"]
        manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        (root / "prediction.parquet").unlink()
        (root / "member.parquet").write_bytes(b"old")
        monkeypatch.setattr(
            bundle_validation,
            "validate_output_manifest_business_envelope",
            lambda value: value,
        )
    observer_hits = 0

    def observer(
        _payload: dict[str, object],
        _members: tuple[object, ...],
    ) -> None:
        nonlocal observer_hits
        observer_hits += 1
        member = root / (
            "member.parquet" if mutation == "parquet_replace" else "prediction.parquet"
        )
        if mutation == "add":
            (root / "late.bin").write_bytes(b"late")
        elif mutation == "delete":
            member.unlink()
        elif mutation == "rename":
            member.rename(root / "renamed.csv")
        elif mutation in {"replace", "parquet_replace"}:
            replacement = tmp_path / f"replacement-{mutation}"
            replacement.write_bytes(b"new")
            replacement.replace(member)
        else:
            moved = tmp_path / "moved"
            replacement = tmp_path / "replacement-root"
            _write_bundle(replacement)
            root.rename(moved)
            replacement.rename(root)

    with pytest.raises(ValueError) as caught:
        validate_output_bundle(root, observer=observer)

    assert {
        "observer_hits": observer_hits,
        "error": str(caught.value),
    } == {
        "observer_hits": 1,
        "error": (
            "output_member_set_mismatch"
            if mutation in {"add", "delete", "rename"}
            else "output_bundle_identity_changed"
        ),
    }


@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize(
    "secondary_type", (RuntimeError, KeyboardInterrupt, SystemExit)
)
def test_validate_output_bundle_preserves_observer_primary_over_final_secondary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
    secondary_type: type[BaseException],
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    validate_root = bundle_validation._validated_bundle_root
    primary = primary_type("observer_primary")
    secondary = secondary_type("final_secondary")
    root_hits = 0
    observer_hits = 0

    def final_secondary(candidate: Path):  # noqa: ANN202
        nonlocal root_hits
        root_hits += 1
        if root_hits == 2:
            raise secondary
        return validate_root(candidate)

    def observer(*_args: object) -> None:
        nonlocal observer_hits
        observer_hits += 1
        raise primary

    monkeypatch.setattr(
        bundle_validation,
        "_validated_bundle_root",
        final_secondary,
    )
    caught: BaseException | None = None
    try:
        validate_output_bundle(root, observer=observer)
    except BaseException as exc:
        caught = exc

    assert {
        "caught_is_primary": caught is primary,
        "observer_hits": observer_hits,
        "root_hits": root_hits,
    } == {
        "caught_is_primary": True,
        "observer_hits": 1,
        "root_hits": 2,
    }


@pytest.mark.parametrize(
    "secondary_type", (RuntimeError, KeyboardInterrupt, SystemExit)
)
def test_validate_output_bundle_propagates_final_secondary_without_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secondary_type: type[BaseException],
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    validate_root = bundle_validation._validated_bundle_root
    secondary = secondary_type("final_secondary")
    root_hits = 0
    observer_hits = 0

    def final_secondary(candidate: Path):  # noqa: ANN202
        nonlocal root_hits
        root_hits += 1
        if root_hits == 2:
            raise secondary
        return validate_root(candidate)

    def observer(*_args: object) -> None:
        nonlocal observer_hits
        observer_hits += 1

    monkeypatch.setattr(
        bundle_validation,
        "_validated_bundle_root",
        final_secondary,
    )
    caught: BaseException | None = None
    try:
        validate_output_bundle(root, observer=observer)
    except BaseException as exc:
        caught = exc

    assert {
        "caught_is_secondary": caught is secondary,
        "observer_hits": observer_hits,
        "root_hits": root_hits,
    } == {
        "caught_is_secondary": True,
        "observer_hits": 1,
        "root_hits": 2,
    }

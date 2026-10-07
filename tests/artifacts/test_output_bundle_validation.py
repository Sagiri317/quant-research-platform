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
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    native_filesystem_path,
)
from tests.helpers.output_frames import (
    make_prediction_report_spec,
    make_price_volume_prediction_output_frame,
    make_price_volume_prediction_spec,
)


@pytest.mark.parametrize("validation_only", (False, True))
@pytest.mark.parametrize("deep_root", (False, True))
def test_prediction_bundle_validates_and_filters_nanosecond_bounds(
    tmp_path: Path, monkeypatch, deep_root: bool, validation_only: bool
) -> None:
    if deep_root:
        while len(str(tmp_path)) <= 270:
            tmp_path /= "segment_" + "a" * 32
        assert len(str(tmp_path)) > 260
    from dataclasses import replace

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.time import TimeWindow
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
    root = write_prediction_outputs(
        spec=spec,
        output_spec=make_prediction_report_spec(),
        layout=StorageLayout(tmp_path),
        output_chunks=(frame,),
        model_metadata=model,
        prediction_metadata_factory=lambda: metadata,
    )
    reader = (
        bundle_validation._validate_prediction_output_bundle
        if validation_only
        else bundle_validation.read_prediction_output_bundle
    )
    index_type = bundle_validation.ExactPrimaryKeyIndex
    indexes = []

    def capture_index(**kwargs):
        index = index_type(**kwargs)
        indexes.append(index)
        return index

    monkeypatch.setattr(bundle_validation, "ExactPrimaryKeyIndex", capture_index)
    _manifest, lookup = reader(
        root,
        expected_output_id=spec.prediction_id,
        time_window=TimeWindow(times[1], times[1]),
    )
    assert len(indexes) == (1 if validation_only else 2)
    if validation_only:
        assert lookup is None
    else:
        values, found = lookup.values_for_frame(frame, mark_matched=True)
        assert list(found) == [0, 1, 0]
        assert values[1] == 2.0
    path = root / "prediction.parquet"
    if deep_root:
        assert len(str(path)) > 260
    with pq.ParquetFile(native_filesystem_path(path)) as parquet:
        original = parquet.read()
    for position, delta in ((0, -1), (2, 1)):
        changed = list(times)
        changed[position] += pd.Timedelta(delta, unit="ns")
        altered = original.set_column(
            original.schema.get_field_index("timestamp"),
            original.schema.field("timestamp"),
            pa.array(changed, type=pa.timestamp("ns")),
        )
        pq.write_table(altered, native_filesystem_path(path))
        with pytest.raises(ValueError, match="^persisted_prediction_window_mismatch$"):
            reader(
                root,
                expected_output_id=spec.prediction_id,
                time_window=TimeWindow(times[1], times[1]),
            )
            pytest.fail("out-of-window source row reached the filtered reader")
    for column, values, error in (
        (
            "timestamp",
            [times[1], times[1], times[2]],
            "duplicate_persisted_prediction_keys",
        ),
        ("prediction_score", [float("nan"), 2.0, 3.0], "invalid_exact_float_value"),
    ):
        altered = original.set_column(
            original.schema.get_field_index(column),
            original.schema.field(column),
            pa.array(values, type=original.schema.field(column).type),
        )
        pq.write_table(altered, native_filesystem_path(path))
        with pytest.raises(ValueError, match=f"^{error}$"):
            reader(
                root,
                expected_output_id=spec.prediction_id,
                time_window=TimeWindow(times[1], times[1]),
            )


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
    (root / "manifest.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    for name in files:
        (root / name).write_bytes(name.encode("utf-8"))


def test_validate_output_bundle_returns_declared_member_order(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    manifest, members = validate_output_bundle(
        root, expected_run_id="test_run", expected_output_id="pred-run"
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
    tmp_path: Path, mutation: str
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
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
        bundle_validation, "_validated_bundle_root", mutate_before_final_gate
    )
    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        validate_output_bundle(root)


def test_validate_output_bundle_rejects_same_path_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    replacement = tmp_path / "replacement"
    moved = tmp_path / "moved"
    _write_bundle(root)
    _write_bundle(replacement)
    validate_root = bundle_validation._validated_bundle_root
    calls = 0

    def replace_after_initial_capture(candidate: Path):
        nonlocal calls
        calls += 1
        validated = validate_root(candidate)
        if calls == 1:
            root.rename(moved)
            replacement.rename(root)
        return validated

    monkeypatch.setattr(
        bundle_validation, "_validated_bundle_root", replace_after_initial_capture
    )
    with pytest.raises(ValueError, match="output_bundle_identity_changed"):
        validate_output_bundle(root)
    assert calls == 2


def test_validate_output_bundle_runs_observer_once_between_capture_and_final_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    validate_root = bundle_validation._validated_bundle_root
    require_members = bundle_validation._require_exact_member_set
    events: list[str] = []

    def observed_root(candidate: Path):
        events.append("root")
        return validate_root(candidate)

    def observed_members(candidate: Path, expected: tuple[str, ...]) -> None:
        events.append("members")
        require_members(candidate, expected)

    observer_hits: list[tuple[dict[str, object], tuple[object, ...]]] = []

    def observer(payload: dict[str, object], members: tuple[object, ...]) -> None:
        events.append("observer")
        observer_hits.append((payload, members))

    monkeypatch.setattr(bundle_validation, "_validated_bundle_root", observed_root)
    monkeypatch.setattr(
        bundle_validation, "_require_exact_member_set", observed_members
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
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

    def observer(_payload: dict[str, object], _members: tuple[object, ...]) -> None:
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
    assert {"observer_hits": observer_hits, "error": str(caught.value)} == {
        "observer_hits": 1,
        "error": "output_member_set_mismatch"
        if mutation in {"add", "delete", "rename"}
        else "output_bundle_identity_changed",
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

    def final_secondary(candidate: Path):
        nonlocal root_hits
        root_hits += 1
        if root_hits == 2:
            raise secondary
        return validate_root(candidate)

    def observer(*_args: object) -> None:
        nonlocal observer_hits
        observer_hits += 1
        raise primary

    monkeypatch.setattr(bundle_validation, "_validated_bundle_root", final_secondary)
    caught: BaseException | None = None
    try:
        validate_output_bundle(root, observer=observer)
    except BaseException as exc:
        caught = exc
    assert {
        "caught_is_primary": caught is primary,
        "observer_hits": observer_hits,
        "root_hits": root_hits,
    } == {"caught_is_primary": True, "observer_hits": 1, "root_hits": 2}


@pytest.mark.parametrize(
    "secondary_type", (RuntimeError, KeyboardInterrupt, SystemExit)
)
def test_validate_output_bundle_propagates_final_secondary_without_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secondary_type: type[BaseException]
) -> None:
    root = tmp_path / "bundle"
    _write_bundle(root)
    validate_root = bundle_validation._validated_bundle_root
    secondary = secondary_type("final_secondary")
    root_hits = 0
    observer_hits = 0

    def final_secondary(candidate: Path):
        nonlocal root_hits
        root_hits += 1
        if root_hits == 2:
            raise secondary
        return validate_root(candidate)

    def observer(*_args: object) -> None:
        nonlocal observer_hits
        observer_hits += 1

    monkeypatch.setattr(bundle_validation, "_validated_bundle_root", final_secondary)
    caught: BaseException | None = None
    try:
        validate_output_bundle(root, observer=observer)
    except BaseException as exc:
        caught = exc
    assert {
        "caught_is_secondary": caught is secondary,
        "observer_hits": observer_hits,
        "root_hits": root_hits,
    } == {"caught_is_secondary": True, "observer_hits": 1, "root_hits": 2}


@pytest.mark.parametrize("validation_only", (False, True))
@pytest.mark.parametrize(
    ("boundary", "failure"),
    (
        ("decode", "memory"),
        ("decode", "io"),
        ("decode", "interrupt"),
        ("root_stat", "io"),
        ("member_stat", "io"),
        ("scandir", "io"),
    ),
)
def test_prediction_bundle_preserves_source_failure(
    tmp_path, monkeypatch, boundary, failure, validation_only
):
    import errno

    root = tmp_path / "bundle"
    _write_bundle(root)
    primary = {
        "memory": MemoryError("primary"),
        "io": OSError(errno.EIO, "primary"),
        "interrupt": KeyboardInterrupt("primary"),
    }[failure]

    def fail(*args, **kwargs):
        raise primary

    if boundary == "decode":
        monkeypatch.setattr(bundle_validation, "_opened_parquet", fail)
    else:
        operation = "scandir" if boundary == "scandir" else "stat"
        original = getattr(os, operation)
        target = root / "prediction.parquet" if boundary == "member_stat" else root

        def fail_io(candidate, *args, **kwargs):
            if Path(candidate) in (target, Path(native_filesystem_path(target))):
                raise primary
            return original(candidate, *args, **kwargs)

        monkeypatch.setattr(os, operation, fail_io)
    reader = (
        bundle_validation._validate_prediction_output_bundle
        if validation_only
        else bundle_validation.read_prediction_output_bundle
    )
    with pytest.raises(type(primary)) as caught:
        reader(root, expected_output_id="pred-run")
        pytest.fail("fatal source failure was swallowed")
    assert caught.value is primary


@pytest.mark.parametrize("validation_only", (False, True))
@pytest.mark.parametrize("intraday", (False, True))
@pytest.mark.parametrize("end_shift", (-1, 0, 1))
def test_prediction_effective_end_matches_all_rows_before_filtering(
    tmp_path, monkeypatch, intraday, end_shift, validation_only
):
    from dataclasses import replace

    import pandas as pd
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.outputs import write_prediction_outputs
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.storage.layout import StorageLayout

    step = pd.Timedelta(1, unit="ns" if intraday else "D")
    times = [pd.Timestamp("2014-01-03") + i * step for i in (1, 2, 3)]

    def stamp(value):
        return value.isoformat() if intraday else value.date().isoformat()

    frame = make_price_volume_prediction_output_frame()
    frame = frame.loc[frame.index.repeat(3)].reset_index(drop=True)
    frame["timestamp"] = times
    frame["prediction_score"] = [1.0, 2.0, 3.0]
    spec = replace(
        make_price_volume_prediction_spec(),
        train_end="2014-01-02T00:00:00" if intraday else "2014-01-02",
        test_start=stamp(times[0]),
        test_end=stamp(times[-1] + step),
    )
    model = dict(frame.attrs["model_metadata"])
    model.update(
        train_start=spec.train_end,
        train_end=spec.train_end,
        test_start=spec.test_start,
        test_end=spec.test_end,
        oos_start=spec.test_start,
        oos_end=stamp(times[-1]),
    )
    prediction = dict(frame.attrs["prediction_metadata"])
    prediction.update(
        train_end=spec.train_end,
        rows_input=3,
        rows_after_filter_mode=3,
        rows_after_input_dropna=3,
        prediction_rows=3,
        test_start=spec.test_start,
        requested_test_end=spec.test_end,
        effective_test_end=stamp(times[-1]),
    )
    root = write_prediction_outputs(
        spec=spec,
        output_spec=make_prediction_report_spec(),
        layout=StorageLayout(tmp_path),
        output_chunks=(frame,),
        model_metadata=model,
        prediction_metadata_factory=lambda: prediction,
    )
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = stamp(times[-1] + end_shift * step)
    payload["metadata"]["prediction_metadata"]["effective_test_end"] = declared
    payload["metadata"]["model_metadata"]["oos_end"] = declared
    path.write_text(json.dumps(payload), encoding="utf-8")
    original_batches = pq.ParquetFile.iter_batches
    scans = []

    def one_row_batches(parquet, **kwargs):
        scans.append(True)
        kwargs["batch_size"] = 1
        yield from original_batches(parquet, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", one_row_batches)
    reader = (
        bundle_validation._validate_prediction_output_bundle
        if validation_only
        else bundle_validation.read_prediction_output_bundle
    )
    for window in (None, TimeWindow(times[0], times[0])):
        if end_shift:
            with pytest.raises(
                ValueError, match="^persisted_prediction_effective_end_mismatch$"
            ):
                reader(root, expected_output_id=spec.prediction_id, time_window=window)
        else:
            _manifest, lookup = reader(
                root, expected_output_id=spec.prediction_id, time_window=window
            )
            if validation_only:
                assert lookup is None
            else:
                assert lookup.row_count == (3 if window is None else 1)
                values, found = lookup.values_for_frame(frame, mark_matched=False)
                assert list(found) == ([1, 1, 1] if window is None else [1, 0, 0])
                assert values[0] == 1.0
    assert len(scans) == 2


@pytest.mark.parametrize("alias", (False, True))
def test_output_destination_uses_layout_relationship_not_name(tmp_path, alias):
    from quant_research_platform.artifacts.outputs.common_validation import (
        derive_output_bundle_root,
    )
    from quant_research_platform.storage.layout import StorageLayout

    root = tmp_path / "metadata" / "raw-notes"
    root.mkdir(parents=True)
    if alias:
        data = root / "data" / "raw"
        data.mkdir(parents=True)
        (root / "outputs").symlink_to(data, target_is_directory=True)
    kwargs = dict(
        output_spec=make_prediction_report_spec(),
        layout=StorageLayout(root),
        expected_definition_id="prediction_outputs",
    )
    if alias:
        with pytest.raises(ValueError, match="output_bundle_must_not_be_dataset_root"):
            derive_output_bundle_root(**kwargs)
    else:
        assert derive_output_bundle_root(**kwargs) == root / "outputs" / "pred-run"

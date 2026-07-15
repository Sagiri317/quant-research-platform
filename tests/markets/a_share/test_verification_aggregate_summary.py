from __future__ import annotations

import inspect
import json
import os
from collections.abc import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts._evaluation_output_validation import (
    EVALUATION_ARTIFACTS,
)
from quant_research_platform.artifacts._output_manifest import (
    evaluation_output_metadata,
    output_manifest_payload,
    validate_output_manifest_business_envelope,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.markets.implementations.a_share.verification import (
    aggregates,
)
from quant_research_platform.runtime.resources import RuntimeResourcePolicy

RESOURCE_POLICY = RuntimeResourcePolicy.from_environment(
    available_memory_bytes=1024**3,
    environ={},
)


def _summary(frame: pd.DataFrame, scratch_parent) -> aggregates.AggregateSummary:
    return aggregates._summarize_frames(
        (frame,),
        scratch_parent=scratch_parent,
    )


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_b", "asset_c"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "forward_return_10d": [0.01, 0.02, None],
        }
    )


class CountingDatasetReader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.calls = 0
        self._frame = frame

    def __call__(self, *_args: object, **_kwargs: object) -> Iterator[pd.DataFrame]:
        self.calls += 1
        return iter((self._frame.copy(),))


def _write_valid_output_bundle(root) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    filter_mode = "unfiltered"
    spec = EvaluationSpec(
        run_id="synthetic",
        market_id=MarketId("a_share"),
        research_dataset_id="research_dataset",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_kind="factor",
                source_column="value",
                direction="higher_is_better",
                transform="identity",
            ),
        ),
        label_columns=("label",),
        filter_mode=filter_mode,
        n_groups=5,
    )
    diagnostics = EvaluationDiagnostics(
        filter_mode=filter_mode,
        filter_columns_used=required_filter_columns(filter_mode),
        filter_definition=filter_definition(filter_mode),
        rows_input=0,
        rows_after_filter_mode=0,
        rows_after_signal_label_dropna=0,
        metric_rows=0,
        grouped_label_value_rows=0,
        ic_summary_rows=0,
        group_spread_summary_rows=0,
        timestamp_count=0,
        min_timestamp=None,
        max_timestamp=None,
        n_groups=5,
        skipped_timestamp_count=0,
    )
    payload = output_manifest_payload(
        output_spec=OutputSpec(
            output_id=root.name,
            output_definition_id="evaluation_outputs",
            run_id=spec.run_id,
            market_id=spec.market_id,
            run_type=spec.run_type,
            scope=OutputScope.LOCAL_ONLY,
        ),
        metadata=evaluation_output_metadata(
            spec=spec,
            evaluation_diagnostics=diagnostics,
        ),
        files=EVALUATION_ARTIFACTS,
    )
    for name in EVALUATION_ARTIFACTS:
        path = root / name
        if path.suffix == ".parquet":
            pq.write_table(pa.table({"value": [1.0]}), path)
        else:
            path.write_text("summary", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_aggregates_summary_is_row_order_and_file_split_insensitive(tmp_path) -> None:
    frame = _base_frame()
    split_summary = aggregates._summarize_frames(
        (
            frame.iloc[[2]].copy(),
            frame.iloc[[0, 1]].copy(),
        ),
        scratch_parent=tmp_path,
    )
    shuffled_summary = _summary(frame.sample(frac=1.0, random_state=9), tmp_path)

    assert split_summary.aggregate_checksum == shuffled_summary.aggregate_checksum
    assert split_summary.key_value_paired_summary_hash == (
        shuffled_summary.key_value_paired_summary_hash
    )


def test_aggregates_output_frames_are_lazy(
    monkeypatch,
    tmp_path,
) -> None:
    report_root = tmp_path / "report"
    _write_valid_output_bundle(report_root)
    read_order: list[str] = []

    observed_targets: list[int] = []

    def fake_iter_parquet_file_batches(  # noqa: ANN001
        path,
        *,
        target_chunk_bytes,
    ):
        observed_targets.append(target_chunk_bytes)
        read_order.append(path.name)
        yield _base_frame().iloc[:1].copy()

    def reject_full_file_read(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("output bundle bypassed bounded physical reader")

    monkeypatch.setattr(
        aggregates,
        "iter_parquet_file_batches",
        fake_iter_parquet_file_batches,
        raising=False,
    )
    monkeypatch.setattr(aggregates.pd, "read_parquet", reject_full_file_read)
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=64 * 1024,
        environ={},
    )
    with aggregates.runtime_resource_policy_scope(policy):
        frames = aggregates._iter_output_frames(
            (report_root,),
            target_chunk_bytes=policy.payload_target_bytes,
        )

        assert read_order == []
        next(frames)
        assert read_order == ["metrics.parquet"]
        next(frames)
        assert read_order == ["metrics.parquet", "grouped_label_values.parquet"]
    assert observed_targets == [policy.payload_target_bytes] * 2


def test_aggregates_validate_all_members_before_selecting_parquet(tmp_path) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)

    assert aggregates._validated_output_parquet_paths(root) == tuple(
        root / name for name in EVALUATION_ARTIFACTS if name.endswith(".parquet")
    )


def test_aggregates_output_frames_reject_manifest_stray_parquet(tmp_path) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    table = pa.table({"value": [1.0]})
    pq.write_table(table, root / "stray.parquet")

    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


def test_aggregates_output_frames_reject_external_hardlink(tmp_path) -> None:
    external = tmp_path / "external.parquet"
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    pq.write_table(pa.table({"value": [1.0]}), external)
    (root / "metrics.parquet").unlink()
    os.link(external, root / "metrics.parquet")

    with pytest.raises(
        ValueError,
        match="output_member_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


@pytest.mark.parametrize(
    ("member_state", "expected_error"),
    (
        ("missing", "output_member_set_mismatch"),
        ("directory", "output_member_physical_identity_invalid"),
    ),
)
def test_aggregates_output_frames_validate_declared_non_parquet_members(
    tmp_path,
    member_state: str,
    expected_error: str,
) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    (root / "summary.md").unlink()
    if member_state == "directory":
        (root / "summary.md").mkdir()

    with pytest.raises(
        ValueError,
        match=expected_error,
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


def test_aggregates_output_frames_reject_declared_member_physical_duplicate(
    tmp_path,
) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    (root / "grouped_label_values.parquet").unlink()
    os.link(root / "metrics.parquet", root / "grouped_label_values.parquet")

    with pytest.raises(
        ValueError,
        match="output_member_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


def test_aggregates_output_frames_reject_normalized_manifest_alias(tmp_path) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    payload["files"] = ["metrics.parquet", "./metrics.parquet"]
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="output_file_path_invalid"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("unknown_field", "output_manifest_fields_invalid"),
        ("wrong_kind_output_id", "output_manifest_output_id_invalid"),
        ("wrong_kind_run_id", "output_manifest_run_id_invalid"),
        ("equal_ids", "output_manifest_output_id_invalid"),
        ("unknown_definition", "output_manifest_definition_id_invalid"),
        ("deep_nested_schema_id", "output_manifest_metadata_invalid"),
        ("backslash_file", "output_file_path_invalid"),
        ("casefold_duplicate_file", "output_files_invalid"),
    ),
)
def test_aggregates_explicit_output_rejects_noncurrent_manifest_envelope(
    tmp_path,
    mutation: str,
    expected_error: str,
) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    if mutation == "unknown_field":
        payload["unknown"] = True
    elif mutation == "wrong_kind_output_id":
        payload["output_id"] = 7
    elif mutation == "wrong_kind_run_id":
        payload["run_id"] = 7
    elif mutation == "equal_ids":
        payload["output_id"] = "evaluation_outputs"
    elif mutation == "unknown_definition":
        payload["output_definition_id"] = "unknown_outputs"
    elif mutation == "deep_nested_schema_id":
        payload["metadata"] = {"nested": [{"schema_id": "forbidden"}]}
    elif mutation == "backslash_file":
        payload["files"] = ["nested\\metrics.parquet"]
    elif mutation == "casefold_duplicate_file":
        payload["files"] = ["metrics.parquet", "METRICS.PARQUET"]

    with pytest.raises(ValueError, match=expected_error):
        validate_output_manifest_business_envelope(payload)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


@pytest.mark.parametrize("extra_kind", ("file", "nested"))
def test_aggregates_output_frames_reject_undeclared_non_parquet_members(
    tmp_path,
    extra_kind: str,
) -> None:
    root = tmp_path / "output"
    _write_valid_output_bundle(root)
    if extra_kind == "file":
        (root / "extra.md").write_text("summary", encoding="utf-8")
    else:
        nested = root / "nested"
        nested.mkdir()
        (nested / "summary.md").write_text("summary", encoding="utf-8")
    with pytest.raises(ValueError, match="output_member_set_mismatch"):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


def test_aggregates_output_frames_reject_manifest_symlink(tmp_path) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    (root / "manifest.json").unlink()
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    try:
        (root / "manifest.json").symlink_to(external_manifest)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(
        ValueError,
        match="output_manifest_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


def test_aggregates_output_frames_reject_manifest_external_hardlink(tmp_path) -> None:
    root = tmp_path / "output"
    payload = _write_valid_output_bundle(root)
    (root / "manifest.json").unlink()
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    try:
        os.link(external_manifest, root / "manifest.json")
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")

    with pytest.raises(
        ValueError,
        match="output_manifest_physical_identity_invalid",
    ):
        list(
            aggregates._iter_output_frames(
                (root,),
                target_chunk_bytes=1024**2,
            )
        )


def test_aggregates_summary_detects_single_value_change(tmp_path) -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    candidate.loc[0, "forward_return_10d"] = 0.03

    assert _summary(baseline, tmp_path).aggregate_checksum != (
        _summary(candidate, tmp_path).aggregate_checksum
    )


def test_aggregates_summary_detects_value_swap_between_keys(tmp_path) -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    candidate.loc[[0, 1], "forward_return_10d"] = candidate.loc[
        [1, 0],
        "forward_return_10d",
    ].to_numpy()

    left = _summary(baseline, tmp_path)
    right = _summary(candidate, tmp_path)

    assert left.content_summary == right.content_summary
    assert left.key_value_paired_summary_hash != right.key_value_paired_summary_hash
    assert left.aggregate_checksum != right.aggregate_checksum


def test_aggregates_summary_canonicalizes_numeric_signed_zero(tmp_path) -> None:
    baseline = _base_frame()
    candidate = baseline.copy()
    baseline.loc[0, "forward_return_10d"] = 0.0
    candidate.loc[0, "forward_return_10d"] = -0.0

    assert _summary(baseline, tmp_path).key_value_paired_summary_hash == (
        _summary(candidate, tmp_path).key_value_paired_summary_hash
    )


def test_aggregates_summary_detects_contract_shape_changes(tmp_path) -> None:
    baseline = _base_frame()
    extra_column = baseline.assign(extra_factor=[1.0, 2.0, 3.0])
    dtype_changed = baseline.assign(
        forward_return_10d=baseline["forward_return_10d"].astype("string")
    )
    null_changed = baseline.copy()
    null_changed.loc[1, "forward_return_10d"] = None
    duplicate_key = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)

    baseline_summary = _summary(baseline, tmp_path)

    assert baseline_summary.column_set_hash != _summary(
        extra_column,
        tmp_path,
    ).column_set_hash
    assert baseline_summary.dtype_summary_hash != (
        _summary(dtype_changed, tmp_path).dtype_summary_hash
    )
    assert baseline_summary.null_summary_hash != (
        _summary(null_changed, tmp_path).null_summary_hash
    )
    assert _summary(duplicate_key, tmp_path).duplicate_key_count > 0


def test_aggregates_summary_is_dynamic_hash_block_insensitive(
    monkeypatch,
    tmp_path,
) -> None:
    frame = pd.concat([_base_frame()] * 2000, ignore_index=True)
    baseline = _summary(frame, tmp_path)
    monkeypatch.setattr(aggregates, "HASH_BLOCK_ROWS", 7)

    assert _summary(frame, tmp_path).aggregate_checksum == baseline.aggregate_checksum


def test_aggregates_summary_ignores_dataframe_index_labels(tmp_path) -> None:
    frame = _base_frame()
    duplicate_index = frame.copy()
    duplicate_index.index = [0, 0, 0]

    assert _summary(duplicate_index, tmp_path).aggregate_checksum == (
        _summary(frame.reset_index(drop=True), tmp_path).aggregate_checksum
    )


def test_aggregate_hash_and_minmax_blocks_are_resource_contract_independent() -> None:
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            aggregates._series_hash_blocks,
            aggregates._key_hash_blocks,
            aggregates._finite_numeric_min_max,
        )
    )

    assert "payload_target" not in source
    assert "active_runtime_resource_policy" not in source
    assert "memory_usage" not in source


def test_aggregates_numeric_min_max_ignores_all_null_datetimes() -> None:
    series = pd.Series(pd.to_datetime([None, None]))

    assert (
        aggregates._finite_numeric_min_max(series)
        is None
    )


def test_aggregates_dataset_summary_cache_reuses_same_root(
    monkeypatch,
    tmp_path,
) -> None:
    reader = CountingDatasetReader(_base_frame())
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks_with_policy",
        reader,
    )
    cache = {}
    root = tmp_path / "dataset"

    first = aggregates._summarize_dataset_root(
        root,
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    second = aggregates._summarize_dataset_root(
        root,
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )

    assert first.aggregate_checksum == second.aggregate_checksum
    assert reader.calls == 1


def test_aggregates_dataset_summary_cache_keeps_year_filters_distinct(
    monkeypatch,
    tmp_path,
) -> None:
    reader = CountingDatasetReader(_base_frame())
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks_with_policy",
        reader,
    )
    cache = {}
    root = tmp_path / "dataset"

    aggregates._summarize_dataset_root(
        root,
        years=(2014,),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    aggregates._summarize_dataset_root(
        root,
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    aggregates._summarize_dataset_root(
        root,
        years=(2014,),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )

    assert reader.calls == 2


def test_aggregates_dataset_summary_cache_normalizes_year_order(
    monkeypatch,
    tmp_path,
) -> None:
    reader = CountingDatasetReader(_base_frame())
    monkeypatch.setattr(
        aggregates.materialization_io,
        "iter_dataset_chunks_with_policy",
        reader,
    )
    cache = {}
    root = tmp_path / "dataset"

    aggregates._summarize_dataset_root(
        root,
        years=(2015, 2014),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )
    aggregates._summarize_dataset_root(
        root,
        years=(2014, 2015),
        summary_cache=cache,
        resource_policy=RESOURCE_POLICY,
    )

    assert reader.calls == 1

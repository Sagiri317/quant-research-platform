"""Strict physical validation for persisted output bundles."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from quant_research_platform.artifacts.outputs.manifest import (
    validate_output_manifest_business_envelope,
)
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import require_unaliased_path
from quant_research_platform.storage.manifests import read_bounded_qrp_json
from quant_research_platform.storage.parquet import (
    DatasetFileIdentity,
    _opened_parquet,
    dataset_file_identity,
    parquet_decoded_working_set_bounds,
)


def validate_output_bundle(
    root: Path,
    *,
    expected_run_id: str | None = None,
    expected_output_id: str | None = None,
    observer: Callable[
        [
            dict[str, object],
            tuple[tuple[Path, DatasetFileIdentity], ...],
        ],
        None,
    ]
    | None = None,
) -> tuple[
    dict[str, object],
    tuple[tuple[Path, DatasetFileIdentity], ...],
]:
    """Validate one complete output bundle and retain every admitted identity.

    The optional observer runs synchronously and read-only after identity capture and
    before the operation-final recheck.  The supported caller domain is a fully
    published bundle kept quiescent for this validation window, with no concurrent
    writer.
    """

    bundle_root = Path(root)
    resolved_root, root_identity = _validated_bundle_root(bundle_root)
    manifest_path = bundle_root / "manifest.json"
    manifest_identity = _validated_regular_member(
        manifest_path,
        resolved_root=resolved_root,
        error="output_manifest_physical_identity_invalid",
    )
    try:
        _manifest_bytes, manifest = read_bounded_qrp_json(
            manifest_path,
            expected_identity=manifest_identity,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("output_manifest_unreadable") from exc
    validate_output_manifest_business_envelope(manifest)
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("output_manifest_run_id_mismatch")
    if expected_output_id is not None and manifest["output_id"] != expected_output_id:
        raise ValueError("output_manifest_output_id_mismatch")

    declared_names = tuple(manifest["files"])
    expected_names = ("manifest.json", *declared_names)
    _require_exact_member_set(bundle_root, expected_names)

    physical_identities = {manifest_identity}
    captured_identities = [(manifest_path, manifest_identity)]
    for name in declared_names:
        path = bundle_root / name
        identity = _validated_regular_member(
            path,
            resolved_root=resolved_root,
            error="output_member_physical_identity_invalid",
        )
        if identity in physical_identities:
            raise ValueError("output_member_physical_identity_invalid")
        physical_identities.add(identity)
        captured_identities.append((path, identity))
    admitted_identities = tuple(captured_identities)
    try:
        if observer is not None:
            observer(manifest, admitted_identities)
    finally:
        primary = sys.exception()
        try:
            final_resolved_root, final_root_identity = _validated_bundle_root(
                bundle_root
            )
            if final_resolved_root != resolved_root or not os.path.samestat(
                root_identity, final_root_identity
            ):
                raise ValueError("output_bundle_identity_changed")
            _require_exact_member_set(bundle_root, expected_names)
            try:
                identities_changed = any(
                    dataset_file_identity(path) != identity
                    for path, identity in admitted_identities
                )
            except (OSError, ValueError) as exc:
                raise ValueError("output_bundle_identity_changed") from exc
            if identities_changed:
                raise ValueError("output_bundle_identity_changed")
        except BaseException as verify_error:
            if primary is None:
                raise
            primary.add_note(
                f"output bundle verification also failed: {type(verify_error).__name__}"
            )
    return manifest, admitted_identities


class _PredictionLookup:
    def __init__(
        self,
        *,
        index: ExactPrimaryKeyIndex,
        primary_key: tuple[str, ...],
        prediction_column: str,
        resource_policy: RuntimeResourcePolicy,
    ) -> None:
        self._index = index
        self.primary_key = primary_key
        self.prediction_column = prediction_column
        self.resource_policy = resource_policy
        self.row_count = 0

    def add_table(self, table: pa.Table, *, payload_headroom_bytes: int) -> None:
        duplicates = self._index.add_float_value_table(
            table,
            namespace="prediction",
            value_column=self.prediction_column,
            payload_target_bytes=payload_headroom_bytes,
        )
        if duplicates:
            raise ValueError("duplicate_persisted_prediction_keys")
        self.row_count += table.num_rows

    @contextmanager
    def values_for_frame(
        self,
        frame: pd.DataFrame,
        *,
        mark_matched: bool,
    ) -> Iterator[tuple[Sequence[float], bytearray]]:
        if tuple(column for column in self.primary_key if column not in frame.columns):
            raise ValueError("persisted_prediction_join_key_invalid")
        row_count = len(frame)
        index_bytes = int(frame.index.memory_usage(deep=True))
        key_frame_bytes = index_bytes + sum(
            int(frame[column].memory_usage(index=False, deep=True))
            for column in self.primary_key
        )
        key_arrow_bytes = key_frame_bytes + 9 * row_count * len(self.primary_key)
        lookup_state_bytes = 9 * row_count
        operation_bytes = key_frame_bytes + key_arrow_bytes + lookup_state_bytes
        with self.resource_policy.lease(
            "artifacts.prediction.lookup.batch",
            operation_bytes,
        ):
            table = pa.Table.from_pandas(
                frame.loc[:, list(self.primary_key)],
                preserve_index=False,
            )
            values, found = self._index.lookup_float_values(
                table,
                source_namespace="prediction",
                matched_namespace="prediction-matched" if mark_matched else None,
                payload_target_bytes=max(1, operation_bytes),
            )
            yield values, found

    def coverage_counts(self) -> tuple[int, int]:
        return self._index.difference_counts(
            expected_namespace="prediction",
            actual_namespace="prediction-matched",
        )


@contextmanager
def read_prediction_output_bundle(
    root: Path,
    *,
    expected_output_id: str,
    expected_run_id: str | None = None,
    time_window=None,  # noqa: ANN001
) -> Iterator[tuple[dict[str, object], _PredictionLookup]]:
    lookup: _PredictionLookup | None = None
    policy = current_runtime_resource_policy()
    stack = ExitStack()

    def read_prediction(
        manifest: dict[str, object],
        members: tuple[tuple[Path, DatasetFileIdentity], ...],
    ) -> None:
        nonlocal lookup
        if manifest["output_definition_id"] != "prediction_outputs":
            raise ValueError("prediction_output_definition_mismatch")
        metadata = manifest.get("metadata")
        if type(metadata) is not dict:
            raise ValueError("persisted_prediction_metadata_mismatch")
        primary_key_value = metadata.get("primary_key")
        prediction_column = metadata.get("prediction_column")
        if (
            type(primary_key_value) is not list
            or not primary_key_value
            or any(
                type(column) is not str or not column for column in primary_key_value
            )
            or type(prediction_column) is not str
            or not prediction_column
        ):
            raise ValueError("persisted_prediction_metadata_mismatch")
        primary_key = tuple(primary_key_value)
        prediction_path, prediction_identity = next(
            (path, identity)
            for path, identity in members
            if path.name == "prediction.parquet"
        )
        try:
            key_index = ExactPrimaryKeyIndex(
                primary_key=primary_key,
                timestamp_columns=tuple(
                    column for column in primary_key if column == "timestamp"
                ),
                scratch_parent=prediction_path.parent.parent,
            )
            stack.enter_context(
                policy.lease(
                    "artifacts.prediction.lookup",
                    key_index.retained_memory_bytes,
                )
            )
            stack.enter_context(key_index)
            lookup = _PredictionLookup(
                index=key_index,
                primary_key=primary_key,
                prediction_column=prediction_column,
                resource_policy=policy,
            )
            with _opened_parquet(
                prediction_path,
                expected_identity=prediction_identity,
            ) as (parquet, _identity):
                expected_columns = (*primary_key, prediction_column)
                if tuple(parquet.schema_arrow.names) != expected_columns:
                    raise ValueError("persisted_prediction_columns_mismatch")
                test_start = metadata.get("test_start")
                test_end = metadata.get("test_end")
                if (
                    type(test_start) is not str
                    or not test_start
                    or (
                        test_end is not None
                        and (type(test_end) is not str or not test_end)
                    )
                ):
                    raise ValueError("persisted_prediction_metadata_mismatch")
                decoded_row_bytes, decode_fixed_bytes = (
                    parquet_decoded_working_set_bounds(parquet, expected_columns)
                )
                filter_row_bytes = 0 if time_window is None else decoded_row_bytes + 3
                batch_row_bytes = decoded_row_bytes + filter_row_bytes
                batch_headroom = policy.operation_headroom_bytes
                if decode_fixed_bytes + batch_row_bytes > batch_headroom:
                    raise MemoryError("runtime_memory_budget_exceeded")
                batch_size = max(
                    1,
                    (batch_headroom - decode_fixed_bytes) // batch_row_bytes,
                )
                batch_working_bytes = decode_fixed_bytes + batch_size * batch_row_bytes
                batches = iter(
                    parquet.iter_batches(
                        batch_size=batch_size,
                        use_threads=False,
                    )
                )
                while True:
                    with policy.lease(
                        "artifacts.prediction.read.batch",
                        batch_working_bytes,
                    ):
                        batch = table = timestamps = keep = None
                        try:
                            try:
                                batch = next(batches)
                            except StopIteration:
                                break
                            table = pa.Table.from_batches([batch])
                            timestamps = table.column("timestamp")
                            if bool(
                                pc.any(
                                    pc.less(
                                        timestamps,
                                        pa.scalar(pd.Timestamp(test_start)),
                                    )
                                ).as_py()
                            ) or (
                                test_end is not None
                                and bool(
                                    pc.any(
                                        pc.greater(
                                            timestamps,
                                            pa.scalar(pd.Timestamp(test_end)),
                                        )
                                    ).as_py()
                                )
                            ):
                                raise ValueError("persisted_prediction_window_mismatch")
                            duplicates = key_index.add_float_value_table(
                                table,
                                namespace="prediction-bundle-validation",
                                value_column=prediction_column,
                                payload_target_bytes=batch_working_bytes,
                            )
                            if duplicates:
                                raise ValueError("duplicate_persisted_prediction_keys")
                            if time_window is not None:
                                keep = pc.and_(
                                    pc.greater_equal(
                                        timestamps,
                                        pa.scalar(pd.Timestamp(time_window.start)),
                                    ),
                                    pc.less_equal(
                                        timestamps,
                                        pa.scalar(pd.Timestamp(time_window.end)),
                                    ),
                                )
                                table = table.filter(keep)
                            if table.num_rows:
                                lookup.add_table(
                                    table,
                                    payload_headroom_bytes=batch_working_bytes,
                                )
                        except BaseException as error:
                            clear_exception_frames(error)
                            raise
                        finally:
                            batch = table = timestamps = keep = None
                if lookup.row_count == 0:
                    raise ValueError("empty_prediction_window")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("prediction_output_unreadable") from exc

    with runtime_resource_policy_scope(policy):
        try:
            manifest, _members = validate_output_bundle(
                root,
                expected_output_id=expected_output_id,
                expected_run_id=expected_run_id,
                observer=read_prediction,
            )
            if lookup is None:
                raise ValueError("prediction_output_unreadable")
            yield manifest, lookup
        finally:
            stack.close()


def _require_exact_member_set(root: Path, expected_names: tuple[str, ...]) -> None:
    observed_names = tuple(entry.name for entry in root.iterdir())
    if (
        len({name.casefold() for name in observed_names}) != len(observed_names)
        or len(observed_names) != len(expected_names)
        or set(observed_names) != set(expected_names)
    ):
        raise ValueError("output_member_set_mismatch")


def _validated_bundle_root(root: Path) -> tuple[Path, os.stat_result]:
    try:
        identity = root.lstat()
        require_unaliased_path(root, error="output_bundle_root_invalid")
        if not stat.S_ISDIR(identity.st_mode):
            raise ValueError("output_bundle_root_invalid")
        resolved = root.resolve(strict=True)
        if not os.path.samestat(identity, root.lstat()):
            raise ValueError("output_bundle_root_invalid")
        return resolved, identity
    except OSError as exc:
        raise ValueError("output_bundle_root_invalid") from exc


def _validated_regular_member(
    path: Path,
    *,
    resolved_root: Path,
    error: str,
) -> DatasetFileIdentity:
    try:
        identity = dataset_file_identity(path)
        resolved_path = path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ValueError(error) from exc
    if resolved_path.parent != resolved_root:
        raise ValueError(error)
    return identity

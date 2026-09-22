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
from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_crypto_prediction_frame,
    validate_family_prediction_frame,
    validate_prediction_output_metadata_shape,
)
from quant_research_platform.contracts.predictions import (
    ALPHA_SCORE_COLUMN,
    prediction_output_columns,
)
from quant_research_platform.contracts.time import (
    is_datetime_value,
    time_value_isoformat,
)
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    native_filesystem_path,
    require_unaliased_path,
)
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
    except ValueError as exc:
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
            except ValueError as exc:
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
        family_selection: str | None = None,
        champions: dict[str, str] | None = None,
    ) -> None:
        self._index = index
        self.primary_key = primary_key
        self.prediction_column = prediction_column
        self.resource_policy = resource_policy
        self.family_selection = family_selection
        self.champions = champions or {}
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
        required = (
            tuple(key for key in self.primary_key if key != "family_id")
            if self.family_selection is not None
            else self.primary_key
        )
        if any(column not in frame for column in required):
            raise ValueError("persisted_prediction_join_key_invalid")
        row_count = len(frame)
        index_bytes = int(frame.index.memory_usage(deep=True))
        key_frame_bytes = index_bytes + sum(
            int(frame[column].memory_usage(index=False, deep=True))
            for column in required
        )
        operation_bytes = (
            2 * key_frame_bytes
            + 18 * row_count * len(self.primary_key)
            + (512 * row_count if self.family_selection is not None else 9 * row_count)
        )
        with self.resource_policy.lease(
            "artifacts.prediction.lookup.batch", operation_bytes
        ):
            lookup_frame = table = values = found = family = None
            try:
                lookup_frame = frame.loc[:, list(required)].copy()
                if self.family_selection is not None:
                    family = (
                        pd.to_datetime(frame["timestamp"])
                        .dt.to_period("Q")
                        .astype(str)
                        .map(self.champions)
                        if self.family_selection == "champion"
                        else pd.Series(self.family_selection, index=frame.index)
                    )
                    if family.isna().any() or (
                        "family_id" in frame and not frame["family_id"].eq(family).all()
                    ):
                        raise ValueError("persisted_prediction_family_invalid")
                    lookup_frame["family_id"] = family
                table = pa.Table.from_pandas(
                    lookup_frame.loc[:, list(self.primary_key)], preserve_index=False
                )
                values, found = self._index.lookup_float_values(
                    table,
                    source_namespace="prediction",
                    matched_namespace="prediction-matched" if mark_matched else None,
                    payload_target_bytes=max(1, operation_bytes),
                )
                yield values, found
            finally:
                lookup_frame = table = values = found = family = None

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
    family_selection: str | None = None,
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
        validate_prediction_output_metadata_shape(metadata)
        primary_key_value = metadata.get("primary_key")
        prediction_column = metadata.get("prediction_column")
        prediction_mode = metadata.get("prediction_mode")
        lookup_column = (
            ALPHA_SCORE_COLUMN
            if prediction_mode == "a_share_oos"
            else prediction_column
        )
        if family_selection is not None and (
            prediction_mode != "a_share_oos"
            or (
                family_selection != "champion"
                and family_selection
                not in {
                    item["family_id"]
                    for item in metadata["model_metadata"]["quarterly_models"]
                }
            )
        ):
            raise ValueError("persisted_prediction_family_invalid")
        primary_key = tuple(primary_key_value)
        prediction_path, prediction_identity = next(
            (path, identity)
            for path, identity in members
            if path.name == "prediction.parquet"
        )
        try:
            key_index = ExactPrimaryKeyIndex(
                primary_key=primary_key,
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
                prediction_column=lookup_column,
                resource_policy=policy,
                family_selection=family_selection,
                champions=metadata["model_metadata"].get("champion_by_quarter", {}),
            )
            with _opened_parquet(
                prediction_path,
                expected_identity=prediction_identity,
            ) as (parquet, _identity, _handle):
                expected_columns = prediction_output_columns(
                    prediction_mode,
                    prediction_column,
                    primary_key,
                )
                if tuple(parquet.schema_arrow.names) != expected_columns:
                    raise ValueError("persisted_prediction_columns_mismatch")
                if (
                    parquet.metadata.num_rows
                    != metadata["prediction_metadata"]["prediction_rows"]
                ):
                    raise ValueError("persisted_prediction_row_count_mismatch")
                test_start = metadata.get("test_start")
                test_end = metadata.get("test_end")
                effective_test_end = None
                decoded_row_bytes, decode_fixed_bytes = (
                    parquet_decoded_working_set_bounds(parquet, expected_columns)
                )
                filter_row_bytes = 0 if time_window is None else decoded_row_bytes + 3
                batch_row_bytes = 4 * decoded_row_bytes + filter_row_bytes
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
                        batch = table = timestamps = keep = frame = None
                        try:
                            try:
                                batch = next(batches)
                            except StopIteration:
                                break
                            table = pa.Table.from_batches([batch])
                            if prediction_mode == "a_share_oos":
                                frame = table.to_pandas()
                                validate_family_prediction_frame(
                                    frame, metadata["model_metadata"]
                                )
                            if prediction_mode == "crypto_oos":
                                frame = table.to_pandas()
                                validate_crypto_prediction_frame(
                                    frame, metadata["model_metadata"]
                                )
                            timestamps = table.column("timestamp")
                            if bool(
                                pc.any(
                                    pc.less(
                                        timestamps,
                                        pa.scalar(
                                            pd.Timestamp(test_start),
                                            type=pa.timestamp("ns"),
                                        ),
                                    )
                                ).as_py()
                            ) or (
                                test_end is not None
                                and bool(
                                    pc.any(
                                        pc.greater(
                                            timestamps,
                                            pa.scalar(
                                                pd.Timestamp(test_end),
                                                type=pa.timestamp("ns"),
                                            ),
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
                            if table.num_rows:
                                batch_end = pd.Timestamp(pc.max(timestamps).as_py())
                                effective_test_end = (
                                    batch_end
                                    if effective_test_end is None
                                    else max(effective_test_end, batch_end)
                                )
                            if family_selection is not None:
                                selected_families = (
                                    frame["quarter_id"].map(
                                        metadata["model_metadata"][
                                            "champion_by_quarter"
                                        ]
                                    )
                                    if family_selection == "champion"
                                    else family_selection
                                )
                                keep = pa.array(
                                    frame["family_id"].eq(selected_families).to_numpy()
                                )
                                table = table.filter(keep)
                                timestamps = table.column("timestamp")
                            frame = None
                            if time_window is not None:
                                keep = pc.and_(
                                    pc.greater_equal(
                                        timestamps,
                                        pa.scalar(
                                            pd.Timestamp(time_window.start),
                                            type=pa.timestamp("ns"),
                                        ),
                                    ),
                                    pc.less_equal(
                                        timestamps,
                                        pa.scalar(
                                            pd.Timestamp(time_window.end),
                                            type=pa.timestamp("ns"),
                                        ),
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
                            batch = table = timestamps = keep = frame = None
                if lookup.row_count == 0:
                    raise ValueError("empty_prediction_window")
                if (
                    time_value_isoformat(
                        effective_test_end,
                        date_only=not is_datetime_value(test_start),
                    )
                    != metadata["prediction_metadata"]["effective_test_end"]
                ):
                    raise ValueError("persisted_prediction_effective_end_mismatch")
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("prediction_output_unreadable") from exc

    with runtime_resource_policy_scope(policy) as policy, closing_iterator(stack):
        manifest, _members = validate_output_bundle(
            root,
            expected_output_id=expected_output_id,
            expected_run_id=expected_run_id,
            observer=read_prediction,
        )
        if lookup is None:
            raise ValueError("prediction_output_unreadable")
        yield manifest, lookup


def _require_exact_member_set(root: Path, expected_names: tuple[str, ...]) -> None:
    try:
        with os.scandir(native_filesystem_path(root)) as entries:
            observed_names = tuple(entry.name for entry in entries)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("output_member_set_mismatch") from exc
    if (
        len({name.casefold() for name in observed_names}) != len(observed_names)
        or len(observed_names) != len(expected_names)
        or set(observed_names) != set(expected_names)
    ):
        raise ValueError("output_member_set_mismatch")


def _validated_bundle_root(root: Path) -> tuple[Path, os.stat_result]:
    try:
        identity = os.stat(native_filesystem_path(root), follow_symlinks=False)
        require_unaliased_path(root, error="output_bundle_root_invalid")
        if not stat.S_ISDIR(identity.st_mode):
            raise ValueError("output_bundle_root_invalid")
        if not os.path.samestat(
            identity,
            os.stat(native_filesystem_path(root), follow_symlinks=False),
        ):
            raise ValueError("output_bundle_root_invalid")
        return root, identity
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("output_bundle_root_invalid") from exc


def _validated_regular_member(
    path: Path,
    *,
    resolved_root: Path,
    error: str,
) -> DatasetFileIdentity:
    try:
        identity = dataset_file_identity(path)
    except ValueError as exc:
        raise ValueError(error) from exc
    if path.parent != resolved_root:
        raise ValueError(error)
    return identity

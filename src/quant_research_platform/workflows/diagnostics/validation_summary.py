"""Workflow-owned validation summary payload builder."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, cast

from quant_research_platform.artifacts.outputs import validate_output_bundle
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    canonical_json_hash,
)
from quant_research_platform.policy.check_status import require_status
from quant_research_platform.policy.share_safety import (
    contains_source_local_key,
    evidence_json_value_unsafe_reason,
    evidence_text_stream_unsafe_reason,
)
from quant_research_platform.runtime.logical_paths import (
    RUN_RESULT_DATASET_LAYERS,
    RUN_RESULT_LOGICAL_PATH_MARKERS,
    dataset_layer_from_path,
    logical_path,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.inspection import (
    _inspect_dataset_session_owned,
)
from quant_research_platform.storage.datasets.reading import open_dataset_session
from quant_research_platform.storage.datasets.schema import (
    validate_dataset_manifest_consumable,
    validate_dataset_primary_key_null_summary,
)
from quant_research_platform.storage.datasets.types import DatasetInspection
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    dataset_file_identity_from_stat,
)
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

_LAYER_SUMMARY_KEYS = frozenset(
    {
        "logical_layer",
        "dataset_count",
        "datasets",
        "total_rows",
        "date_min",
        "date_max",
        "missing_dataset_manifests",
        "invalid_dataset_manifests",
        "failed_dataset_validations",
        "layer_validation_status",
        "created_at",
    }
)


def build_validation_summary(
    *,
    run_id: str,
    dataset_roots: tuple[Path, ...],
    layer_summaries: tuple[dict[str, object], ...],
    validated_output_members: tuple[tuple[Path, DatasetFileIdentity], ...],
    output_bundle_expectations: tuple[tuple[Path, str, str, str], ...],
    _retained_state: ExitStack | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, Any]:
    with (
        runtime_resource_policy_scope(resource_policy) as policy,
        ExitStack() as local_state,
    ):
        return _build_validation_summary(
            run_id=run_id,
            dataset_roots=dataset_roots,
            layer_summaries=layer_summaries,
            validated_output_members=validated_output_members,
            output_bundle_expectations=output_bundle_expectations,
            resource_policy=policy,
            retained_state=_retained_state or local_state,
        )


def _build_validation_summary(
    *,
    run_id: str,
    dataset_roots: tuple[Path, ...],
    layer_summaries: tuple[dict[str, object], ...],
    validated_output_members: tuple[tuple[Path, DatasetFileIdentity], ...],
    output_bundle_expectations: tuple[tuple[Path, str, str, str], ...],
    resource_policy: RuntimeResourcePolicy,
    retained_state: ExitStack,
) -> dict[str, Any]:
    if not dataset_roots:
        raise ValueError("validation_summary_dataset_roots_required")
    if not layer_summaries:
        raise ValueError("validation_summary_layer_summaries_required")
    if not validated_output_members:
        raise ValueError("validation_summary_generated_evidence_required")
    if not output_bundle_expectations:
        raise ValueError("validation_summary_output_bundle_expectations_required")
    if len(set(dataset_roots)) != len(dataset_roots) or len(
        {path for path, _identity in validated_output_members}
    ) != len(validated_output_members):
        raise ValueError("validation_summary_declaration_duplicate")
    if len({root for root, *_rest in output_bundle_expectations}) != len(
        output_bundle_expectations
    ):
        raise ValueError("validation_summary_output_bundle_declaration_duplicate")
    for root in dataset_roots:
        if dataset_layer_from_path(root) not in RUN_RESULT_DATASET_LAYERS:
            raise ValueError("validation_summary_dataset_layer_invalid")
    stage_directory = retained_state.enter_context(
        tempfile.TemporaryDirectory(prefix="qrp-validation-summary-")
    )
    database = sqlite3.connect(Path(stage_directory) / "summary.sqlite")
    retained_state.callback(database.close)
    database.execute("PRAGMA journal_mode=DELETE")
    database.execute("PRAGMA synchronous=NORMAL")
    database.execute("PRAGMA temp_store=FILE")
    database.execute("PRAGMA cache_size=-64")
    _stage_dataset_inspections(
        database,
        dataset_roots,
        resource_policy=resource_policy,
    )
    database.commit()

    dataset_failures: list[str] = []
    null_failures: list[str] = []
    duplicate_failures: list[str] = []
    non_finite_failures: list[str] = []
    source_local_leaks: list[str] = []
    date_ranges: list[dict[str, object]] = []
    factor_nan_profile: dict[str, int] = {}
    label_nan_profile: dict[str, int] = {}
    missing_dataset_manifests: list[str] = []
    invalid_dataset_manifests: list[str] = []
    for root, _inspection_identity, inspection_error, facts in _iter_staged_inspections(
        database,
        dataset_roots,
    ):
        logical_layer = dataset_layer_from_path(root)
        manifest_logical_path = logical_path(
            root / "_manifest" / "manifest.json",
            markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
        )
        if inspection_error is not None:
            if inspection_error == "missing":
                missing_dataset_manifests.append(manifest_logical_path)
            else:
                invalid_dataset_manifests.append(manifest_logical_path)
            continue
        assert facts is not None
        data_id = str(facts["data_id"])
        manifest_consumable = bool(facts["manifest_consumable"])
        if facts["validation_status"] != "PASS":
            dataset_failures.append(data_id)
        # This existing failure class covers an invalid required PK-null fact;
        # an absent fact is never interpreted as an observed zero count.
        if not facts["null_summary_valid"]:
            null_failures.append(data_id)
        if int(facts["duplicate_primary_key_count"]) > 0:
            duplicate_failures.append(data_id)
        non_finite = cast(Mapping[str, int], facts["non_finite_count_summary"])
        if any(count > 0 for count in non_finite.values()):
            non_finite_failures.append(data_id)
        if not manifest_consumable and data_id not in dataset_failures:
            dataset_failures.append(data_id)
        if logical_layer in {"processed", "research"} and facts["source_local_key"]:
            source_local_leaks.append(data_id)
        date_ranges.append(
            {
                "data_id": data_id,
                "date_min": facts["date_min"],
                "date_max": facts["date_max"],
                "partition_count": int(facts["partition_count"]),
            }
        )
        _merge_profile(factor_nan_profile, facts["factor_nan_profile"])
        _merge_profile(label_nan_profile, facts["label_nan_profile"])

    (
        layer_failures,
        declared_missing_manifests,
        declared_invalid_manifests,
        declared_failed_datasets,
    ) = _validate_layer_summary_graph(
        database,
        layer_summaries,
        logical_layers=tuple(
            sorted({dataset_layer_from_path(root) for root in dataset_roots})
        ),
    )
    if (
        missing_dataset_manifests != declared_missing_manifests
        or invalid_dataset_manifests != declared_invalid_manifests
        or dataset_failures != declared_failed_datasets
    ):
        raise ValueError("validation_summary_declaration_mismatch")

    safety = _share_safety_scan(
        validated_output_members,
        output_bundle_expectations=output_bundle_expectations,
    )
    failure_codes: list[str] = []
    if missing_dataset_manifests:
        failure_codes.append("dataset_manifest_missing")
    if invalid_dataset_manifests:
        failure_codes.append("dataset_manifest_invalid")
    if dataset_failures:
        failure_codes.append("dataset_validation_failed")
    if null_failures:
        failure_codes.append("primary_key_nulls")
    if duplicate_failures:
        failure_codes.append("duplicate_primary_key")
    if non_finite_failures:
        failure_codes.append("non_finite_values")
    if source_local_leaks:
        failure_codes.append("source_local_key_leakage")
    if layer_failures:
        failure_codes.append("layer_validation_failed")
    if safety["status"] != "PASS":
        failure_codes.append("share_safety_scan_failed")
    summary = {
        "run_id": run_id,
        "status": "PASS" if not failure_codes else "FAIL",
        "failure_codes": failure_codes,
        "required_columns_status": "summarized_by_dataset_manifests",
        "primary_key_null_failures": null_failures,
        "duplicate_primary_key_failures": duplicate_failures,
        "source_local_key_leakage": source_local_leaks,
        "date_ranges": date_ranges,
        "partition_coverage_status": (
            "PASS"
            if not missing_dataset_manifests and not invalid_dataset_manifests
            else "FAIL"
        ),
        "non_finite_failures": non_finite_failures,
        "factor_nan_profile": factor_nan_profile,
        "label_nan_profile": label_nan_profile,
        "dataset_manifest_completeness": {
            "missing": missing_dataset_manifests,
            "invalid": invalid_dataset_manifests,
            "checked_count": len(dataset_roots),
        },
        "data_layer_summary_completeness": {
            "missing": [],
            "checked_count": len(layer_summaries),
        },
        "share_safety_scan": safety,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
    }
    for root, initial_identity, inspection_error, _facts in _iter_staged_inspections(
        database,
        dataset_roots,
    ):
        if inspection_error is not None or initial_identity is None:
            continue
        owned_final = None
        final_inspection = None
        try:
            try:
                with open_dataset_session(
                    root,
                    columns=None,
                    time_window=None,
                    require_consumable=False,
                    resource_policy=resource_policy,
                ) as context:
                    owned_final = _inspect_dataset_session_owned(
                        context,
                        resource_policy=resource_policy,
                    )
                    final_inspection = owned_final.value
                    unchanged = (
                        _inspection_identity_hash(final_inspection)
                        == initial_identity
                    )
            except BaseException:
                if owned_final is not None:
                    final_inspection = None
                    owned_final.release(owner="workflow.validation.reinspection.result")
                raise
        except MemoryError:
            raise ValueError(
                "validation_summary_payload_exceeds_task_memory_budget"
            ) from None
        except (OSError, ValueError):
            raise ValueError("validation_summary_dataset_identity_changed") from None
        else:
            final_inspection = None
            owned_final.release(owner="workflow.validation.reinspection.result")
            if not unchanged:
                raise ValueError(
                    "validation_summary_dataset_identity_changed"
                )
    return summary


def _stage_dataset_inspections(
    database: sqlite3.Connection,
    dataset_roots: tuple[Path, ...],
    *,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    database.execute(
        "CREATE TABLE inspections ("
        "root_index INTEGER PRIMARY KEY, "
        "logical_layer TEXT NOT NULL, "
        "manifest_path TEXT NOT NULL, "
        "identity_hash TEXT, "
        "inspection_error TEXT, "
        "facts_json TEXT"
        ")"
    )
    for root_index, root in enumerate(dataset_roots):
        logical_layer = dataset_layer_from_path(root)
        manifest_logical_path = logical_path(
            root / "_manifest" / "manifest.json",
            markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
        )
        inspection_identity: str | None = None
        inspection_error: str | None = None
        facts: dict[str, object] | None = None
        inspection: DatasetInspection | None = None
        owned = None
        try:
            try:
                with open_dataset_session(
                    root,
                    columns=None,
                    time_window=None,
                    require_consumable=False,
                    resource_policy=resource_policy,
                ) as context:
                    owned = _inspect_dataset_session_owned(
                        context,
                        resource_policy=resource_policy,
                    )
                    inspection = owned.value
                    inspection_identity = _inspection_identity_hash(inspection)
                    manifest = inspection.manifest
                    try:
                        validate_dataset_manifest_consumable(manifest)
                        manifest_consumable = True
                    except ValueError:
                        manifest_consumable = False
                    try:
                        validate_dataset_primary_key_null_summary(manifest)
                        null_summary_valid = True
                    except ValueError:
                        null_summary_valid = False
                    non_finite = manifest.get("non_finite_count_summary", {})
                    if not isinstance(non_finite, Mapping):
                        non_finite = {}
                    factor_profile = manifest.get("factor_nan_profile", {})
                    if not isinstance(factor_profile, Mapping):
                        factor_profile = {}
                    label_profile = manifest.get("label_nan_profile", {})
                    if not isinstance(label_profile, Mapping):
                        label_profile = {}
                    facts = {
                        "data_id": manifest["data_id"],
                        "manifest_consumable": manifest_consumable,
                        "row_count": manifest["row_count"],
                        "validation_status": require_status(
                            manifest.get("validation_status")
                        ),
                        "null_summary_valid": null_summary_valid,
                        "duplicate_primary_key_count": manifest[
                            "duplicate_primary_key_count"
                        ],
                        "non_finite_count_summary": dict(non_finite),
                        "date_min": manifest.get("date_min"),
                        "date_max": manifest.get("date_max"),
                        "partition_count": manifest["partition_count"],
                        "source_local_key": contains_source_local_key(manifest),
                        "factor_nan_profile": dict(factor_profile),
                        "label_nan_profile": dict(label_profile),
                    }
            except (OSError, ValueError):
                try:
                    manifest = read_dataset_manifest_evidence(root)
                except FileNotFoundError:
                    inspection_identity = None
                    inspection_error = "missing"
                except (OSError, ValueError):
                    inspection_identity = None
                    inspection_error = "invalid"
                else:
                    if manifest:
                        raise ValueError(
                            "validation_summary_dataset_inspection_failed"
                        ) from None
                    inspection_identity = None
                    inspection_error = "invalid"
        finally:
            if owned is not None:
                inspection = None
                owned.release(owner="workflow.validation.inspection.result")
        database.execute(
            "INSERT INTO inspections("
            "root_index, logical_layer, manifest_path, identity_hash, "
            "inspection_error, facts_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                root_index,
                logical_layer,
                manifest_logical_path,
                inspection_identity,
                inspection_error,
                None
                if facts is None
                else json.dumps(
                    facts,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )


def _iter_staged_inspections(
    database: sqlite3.Connection,
    dataset_roots: tuple[Path, ...],
) -> Iterator[tuple[Path, str | None, str | None, dict[str, object] | None]]:
    with closing(database.execute(
        "SELECT root_index, identity_hash, inspection_error, facts_json "
        "FROM inspections ORDER BY root_index"
    )) as rows:
        for root_index, identity_hash, inspection_error, facts_json in rows:
            yield (
                dataset_roots[int(root_index)],
                identity_hash,
                inspection_error,
                None if facts_json is None else json.loads(facts_json),
            )


def _share_safety_scan(
    members: tuple[tuple[Path, DatasetFileIdentity], ...],
    *,
    output_bundle_expectations: tuple[tuple[Path, str, str, str], ...],
) -> dict[str, object]:
    final_members, failures, existing_count, missing_count = _validated_output_snapshot(
        members, output_bundle_expectations
    )
    scanned_members = tuple(
        member for member in final_members if member[0].suffix.casefold() != ".parquet"
    )
    requested_paths = [
        logical_path(path, markers=RUN_RESULT_LOGICAL_PATH_MARKERS)
        for path, _identity in scanned_members
    ]
    if len(requested_paths) != len(set(requested_paths)):
        raise ValueError("validation_summary_evidence_path_duplicate")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checked_count": existing_count,
        "requested_count": len(scanned_members),
        "existing_scanned_count": existing_count,
        "missing_count": missing_count,
        "requested_paths": requested_paths,
    }


def _validated_output_snapshot(
    expected_members: tuple[tuple[Path, DatasetFileIdentity], ...],
    expectations: tuple[tuple[Path, str, str, str], ...],
) -> tuple[
    tuple[tuple[Path, DatasetFileIdentity], ...],
    list[dict[str, str]],
    int,
    int,
]:
    observed_all: list[tuple[Path, DatasetFileIdentity]] = []
    failures: list[dict[str, str]] = []
    existing_count = 0
    missing_count = 0
    for (
        root,
        expected_run_id,
        expected_output_id,
        expected_definition_id,
    ) in expectations:
        expected = tuple(
            member for member in expected_members if member[0].parent == root
        )
        manifest_path = root / "manifest.json"
        tentative_observed: tuple[tuple[Path, DatasetFileIdentity], ...] = ()
        tentative_failures: list[dict[str, str]] = []
        tentative_existing_count = 0

        def observe_admitted_bundle(
            payload: dict[str, object],
            observed: tuple[tuple[Path, DatasetFileIdentity], ...],
            expected_definition_id: str = expected_definition_id,
            expected: tuple[tuple[Path, DatasetFileIdentity], ...] = expected,
            manifest_path: Path = manifest_path,
            tentative_failures: list[dict[str, str]] = tentative_failures,
        ) -> None:
            nonlocal tentative_existing_count, tentative_observed
            if payload["output_definition_id"] != expected_definition_id:
                raise ValueError("output_manifest_definition_id_invalid")
            if observed != expected:
                raise ValueError("output_bundle_identity_changed")

            manifest_reason = evidence_json_value_unsafe_reason(payload)
            if manifest_reason is not None:
                tentative_failures.append(
                    {
                        "path": logical_path(
                            manifest_path,
                            markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
                        ),
                        "reason": manifest_reason,
                    }
                )
            scanned = tuple(
                member
                for member in observed
                if member[0].suffix.casefold() != ".parquet"
            )
            for path, identity in scanned:
                if path == manifest_path:
                    continue
                try:
                    reason = _scan_validated_text_evidence(path, identity)
                except UnicodeDecodeError:
                    reason = "invalid_utf8"
                except (OSError, ValueError):
                    reason = "identity_changed_or_unreadable"
                if reason is not None:
                    tentative_failures.append(
                        {
                            "path": logical_path(
                                path,
                                markers=RUN_RESULT_LOGICAL_PATH_MARKERS,
                            ),
                            "reason": reason,
                        }
                    )
            tentative_observed = observed
            tentative_existing_count = len(scanned)

        try:
            validate_output_bundle(
                root,
                expected_run_id=expected_run_id,
                expected_output_id=expected_output_id,
                observer=observe_admitted_bundle,
            )
        except (OSError, ValueError):
            failures.append(
                {
                    "path": logical_path(
                        manifest_path, markers=RUN_RESULT_LOGICAL_PATH_MARKERS
                    ),
                    "reason": "identity_changed_or_unreadable",
                }
            )
            observed_all.extend(expected)
            missing_count += sum(
                path.suffix.casefold() != ".parquet" for path, _identity in expected
            )
            continue
        failures.extend(tentative_failures)
        observed_all.extend(tentative_observed)
        existing_count += tentative_existing_count
    return tuple(observed_all), failures, existing_count, missing_count


def _scan_validated_text_evidence(
    path: Path,
    expected_identity: DatasetFileIdentity,
) -> str | None:
    if dataset_file_identity(path) != expected_identity:
        raise ValueError("validation_summary_evidence_identity_changed")
    with _opened_evidence(path, expected_identity) as handle:
        try:
            reason = evidence_text_stream_unsafe_reason(
                _iter_exact_evidence_chunks(handle, expected_identity.size)
            )
        finally:
            primary = sys.exception()
            try:
                if (
                    dataset_file_identity_from_stat(os.fstat(handle.fileno()))
                    != expected_identity
                ):
                    raise ValueError("validation_summary_evidence_identity_changed")
            except BaseException as verify_error:
                if primary is None:
                    raise
                primary.add_note(
                    "validation summary evidence verification also failed: "
                    f"{type(verify_error).__name__}"
                )
    return reason


def _iter_exact_evidence_chunks(
    handle: BinaryIO,
    expected_size: int,
) -> Iterator[memoryview]:
    buffer = bytearray(min(expected_size, io.DEFAULT_BUFFER_SIZE))
    offset = 0
    while offset < expected_size:
        target = memoryview(buffer)[: min(len(buffer), expected_size - offset)]
        count = handle.readinto(target)
        if count is None or count <= 0 or count > len(target):
            raise ValueError("validation_summary_evidence_identity_changed")
        offset += count
        yield target[:count]
    if handle.read(1):
        raise ValueError("validation_summary_evidence_identity_changed")


@contextmanager
def _opened_evidence(
    path: Path,
    expected_identity: DatasetFileIdentity,
) -> Iterator[BinaryIO]:
    handle = path.open("rb")
    try:
        if (
            dataset_file_identity_from_stat(os.fstat(handle.fileno()))
            != expected_identity
        ):
            raise ValueError("validation_summary_evidence_identity_changed")
        yield handle
    except BaseException as primary:
        try:
            handle.close()
        except BaseException as close_error:
            primary.add_note(
                "validation summary evidence close also failed: "
                f"{type(close_error).__name__}"
            )
        raise
    else:
        handle.close()


def _validate_layer_summary_graph(
    database: sqlite3.Connection,
    layer_summaries: tuple[dict[str, object], ...],
    *,
    logical_layers: tuple[str, ...],
) -> tuple[list[str], list[str], list[str], list[str]]:
    if len(layer_summaries) != len(logical_layers):
        raise ValueError("validation_summary_layer_declaration_mismatch")
    layer_failures: list[str] = []
    declared_missing_all: list[str] = []
    declared_invalid_all: list[str] = []
    declared_failed_all: list[str] = []
    for summary, logical_layer in zip(layer_summaries, logical_layers, strict=True):
        if type(summary) is not dict or set(summary) != _LAYER_SUMMARY_KEYS:
            raise ValueError("validation_summary_layer_invalid")
        if type(summary["created_at"]) is not str or not summary["created_at"]:
            raise ValueError("validation_summary_layer_invalid")
        require_status(summary["layer_validation_status"])
        declared_datasets = summary["datasets"]
        declared_missing = summary["missing_dataset_manifests"]
        declared_invalid = summary["invalid_dataset_manifests"]
        declared_failed = summary["failed_dataset_validations"]
        if not all(
            type(value) is list
            for value in (
                declared_datasets,
                declared_missing,
                declared_invalid,
                declared_failed,
            )
        ):
            raise ValueError("validation_summary_layer_invalid")
        dataset_count = 0
        total_rows = 0
        date_min: str | None = None
        date_max: str | None = None
        failed_count = 0
        with closing(database.execute(
            "SELECT root_index, manifest_path, facts_json "
            "FROM inspections WHERE logical_layer = ? "
            "AND inspection_error IS NULL ORDER BY manifest_path",
            (logical_layer,),
        )) as rows:
            for _root_index, manifest_path, facts_json in rows:
                facts = json.loads(facts_json)
                item = {
                    "data_id": facts["data_id"],
                    "manifest_path": manifest_path,
                    "manifest_summary": {
                        "row_count": int(facts["row_count"]),
                        "date_min": facts["date_min"],
                        "date_max": facts["date_max"],
                        "partition_count": int(facts["partition_count"]),
                        "duplicate_primary_key_count": int(
                            facts["duplicate_primary_key_count"]
                        ),
                        "validation_status": (
                            "PASS" if facts["manifest_consumable"] else "FAIL"
                        ),
                    },
                }
                if dataset_count >= len(declared_datasets):
                    raise ValueError("validation_summary_layer_declaration_mismatch")
                try:
                    matches = canonical_identity_bytes(
                        declared_datasets[dataset_count]
                    ) == canonical_identity_bytes(item)
                except ValueError as exc:
                    raise ValueError("validation_summary_layer_invalid") from exc
                if not matches:
                    raise ValueError("validation_summary_layer_declaration_mismatch")
                dataset_count += 1
                total_rows += int(facts["row_count"])
                if facts["date_min"] is not None:
                    date_min = (
                        str(facts["date_min"])
                        if date_min is None
                        else min(date_min, str(facts["date_min"]))
                    )
                if facts["date_max"] is not None:
                    date_max = (
                        str(facts["date_max"])
                        if date_max is None
                        else max(date_max, str(facts["date_max"]))
                    )
                if not facts["manifest_consumable"]:
                    if (
                        failed_count >= len(declared_failed)
                        or declared_failed[failed_count] != str(facts["data_id"])
                    ):
                        raise ValueError(
                            "validation_summary_layer_declaration_mismatch"
                        )
                    failed_count += 1
        if dataset_count != len(declared_datasets) or failed_count != len(
            declared_failed
        ):
            raise ValueError("validation_summary_layer_declaration_mismatch")

        for error_name, declared in (
            ("missing", declared_missing),
            ("invalid", declared_invalid),
        ):
            observed_count = 0
            with closing(database.execute(
                "SELECT manifest_path FROM inspections "
                "WHERE logical_layer = ? AND inspection_error = ? "
                "ORDER BY manifest_path",
                (logical_layer, error_name),
            )) as rows:
                for manifest_path, in rows:
                    if (
                        observed_count >= len(declared)
                        or declared[observed_count] != manifest_path
                    ):
                        raise ValueError(
                            "validation_summary_layer_declaration_mismatch"
                        )
                    observed_count += 1
            if observed_count != len(declared):
                raise ValueError("validation_summary_layer_declaration_mismatch")

        expected = {
            "logical_layer": logical_layer,
            "dataset_count": dataset_count,
            "total_rows": total_rows,
            "date_min": date_min,
            "date_max": date_max,
            "missing_dataset_manifests": declared_missing,
            "invalid_dataset_manifests": declared_invalid,
            "failed_dataset_validations": declared_failed,
            "layer_validation_status": (
                "PASS"
                if not declared_missing
                and not declared_invalid
                and not declared_failed
                else "FAIL"
            ),
        }
        observed = {
            key: item
            for key, item in summary.items()
            if key not in {"created_at", "datasets"}
        }
        try:
            matches = canonical_identity_bytes(observed) == canonical_identity_bytes(
                expected
            )
        except ValueError as exc:
            raise ValueError("validation_summary_layer_invalid") from exc
        if not matches:
            raise ValueError("validation_summary_layer_declaration_mismatch")
        if expected["layer_validation_status"] != "PASS":
            layer_failures.append(logical_layer)
        declared_missing_all.extend(declared_missing)
        declared_invalid_all.extend(declared_invalid)
        declared_failed_all.extend(declared_failed)
    return (
        layer_failures,
        declared_missing_all,
        declared_invalid_all,
        declared_failed_all,
    )

def _merge_profile(target: dict[str, int], value: object) -> None:
    profile = cast(Mapping[str, int], value)
    for key, item in profile.items():
        target[key] = target.get(key, 0) + item


def _inspection_identity_hash(value: DatasetInspection) -> str:
    return canonical_json_hash(
        {
            "root": value.root.as_posix(),
            "data_root": value.data_root.as_posix(),
            "spec": value.spec,
            "columns": list(value.columns),
            "schema_hash": value.schema_hash,
            "file_count": value.file_count,
            "row_count": value.row_count,
            "partitions": list(value.partitions),
            "partition_row_counts": value.partition_row_counts,
            "date_min": value.date_min,
            "date_max": value.date_max,
            "manifest": value.manifest,
        }
    )

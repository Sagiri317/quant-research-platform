"""Compare two QRP run roots from their current contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_research_platform.policy.check_status import (  # noqa: E402
    aggregate_validation_matrix_status,
)
from quant_research_platform.policy.identifiers import (  # noqa: E402
    validate_output_identifier,
)
from quant_research_platform.policy.share_safety import (  # noqa: E402
    qrp_json_identity_reason,
)


class CompareBlockedError(ValueError):
    """Raised when compare inputs cannot be read as current QRP contracts."""


def compare_runs(
    *,
    baseline_root: Path,
    current_root: Path,
    compare_root: Path,
    pairs_file: Path | None = None,
) -> dict[str, Any]:
    baseline_root = Path(baseline_root)
    current_root = Path(current_root)
    compare_root = Path(compare_root)
    try:
        baseline_run = _read_qrp_json(
            baseline_root / "run.json",
            expected_schema_id="qrp-run",
        )
        current_run = _read_qrp_json(
            current_root / "run.json",
            expected_schema_id="qrp-run",
        )
        pairs, pairing_mode = _resolve_pairs(
            baseline_run,
            current_run,
            pairs_file=pairs_file,
        )
    except CompareBlockedError as exc:
        payload = _blocked_payload(
            baseline_root=baseline_root,
            current_root=current_root,
            reason=str(exc),
        )
        _write_compare_artifacts(compare_root, payload)
        return payload

    data_results = _compare_data_summaries(
        baseline_run.get("data"),
        current_run.get("data"),
    )
    output_results = [
        _compare_output_pair(
            baseline_root=baseline_root,
            current_root=current_root,
            baseline_run_id=str(baseline_run.get("run_id") or ""),
            current_run_id=str(current_run.get("run_id") or ""),
            baseline_output_id=pair["baseline_output_id"],
            current_output_id=pair["current_output_id"],
        )
        for pair in pairs
    ]
    statuses = [
        *(result["status"] for result in data_results),
        *(result["status"] for result in output_results),
    ]
    overall_status = aggregate_validation_matrix_status(statuses)
    status_counts = dict(sorted(Counter(statuses).items()))
    payload = {
        "schema_id": "qrp-compare",
        "inputs": {
            "baseline_root": baseline_root.name,
            "current_root": current_root.name,
        },
        "baseline_run": _run_summary(baseline_run),
        "current_run": _run_summary(current_run),
        "pairing": {
            "mode": pairing_mode,
            "pairs": pairs,
        },
        "data_results": data_results,
        "output_results": output_results,
        "status_counts": status_counts,
        "overall_status": overall_status,
    }
    _write_compare_artifacts(compare_root, payload)
    return payload


def _blocked_payload(
    *,
    baseline_root: Path,
    current_root: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_id": "qrp-compare",
        "inputs": {
            "baseline_root": baseline_root.name,
            "current_root": current_root.name,
        },
        "baseline_run": {},
        "current_run": {},
        "pairing": {"mode": "NOT_RUN", "pairs": []},
        "data_results": [],
        "output_results": [],
        "status_counts": {"BLOCKED": 1},
        "overall_status": "BLOCKED",
        "blocked_reason": reason,
    }


def _read_qrp_json(path: Path, *, expected_schema_id: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompareBlockedError(f"json_missing:{path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompareBlockedError(f"json_unreadable:{path.name}") from exc
    if not isinstance(payload, dict):
        raise CompareBlockedError(f"json_top_level_not_object:{path.name}")
    identity_reason = qrp_json_identity_reason(payload)
    if identity_reason is not None:
        raise CompareBlockedError(
            f"json_identity_invalid:{path.name}:{identity_reason}"
        )
    if payload.get("schema_id") != expected_schema_id:
        raise CompareBlockedError(f"schema_id_mismatch:{path.name}")
    return payload


def _resolve_pairs(
    baseline_run: dict[str, Any],
    current_run: dict[str, Any],
    *,
    pairs_file: Path | None,
) -> tuple[list[dict[str, str]], str]:
    if pairs_file is not None:
        payload = _read_qrp_json(
            Path(pairs_file),
            expected_schema_id="qrp-compare-pairs",
        )
        if set(payload) != {"schema_id", "pairs"}:
            raise CompareBlockedError("pairs_file_fields_invalid")
        raw_pairs = payload.get("pairs")
        if not isinstance(raw_pairs, list):
            raise CompareBlockedError("pairs_file_pairs_invalid")
        pairs = [_validate_pair(item) for item in raw_pairs]
        return pairs, "pairs_file"

    baseline_ids = _output_ids(baseline_run)
    current_ids = _output_ids(current_run)
    return (
        [
            {
                "baseline_output_id": output_id,
                "current_output_id": output_id,
            }
            for output_id in sorted(baseline_ids | current_ids)
        ],
        "output_id",
    )


def _validate_pair(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "baseline_output_id",
        "current_output_id",
    }:
        raise CompareBlockedError("pairs_file_pair_invalid")
    baseline_id = str(value["baseline_output_id"])
    current_id = str(value["current_output_id"])
    try:
        validate_output_identifier(baseline_id)
        validate_output_identifier(current_id)
    except ValueError as exc:
        raise CompareBlockedError("pairs_file_output_id_invalid") from exc
    return {
        "baseline_output_id": baseline_id,
        "current_output_id": current_id,
    }


def _output_ids(run_payload: dict[str, Any]) -> set[str]:
    outputs = run_payload.get("outputs")
    if not isinstance(outputs, list):
        raise CompareBlockedError("outputs_invalid")
    result: set[str] = set()
    for item in outputs:
        if not isinstance(item, dict):
            raise CompareBlockedError("output_index_invalid")
        output_id = item.get("output_id")
        try:
            result.add(validate_output_identifier(str(output_id)))
        except ValueError as exc:
            raise CompareBlockedError("output_id_invalid") from exc
    return result


def _compare_data_summaries(
    baseline_data: object,
    current_data: object,
) -> list[dict[str, Any]]:
    baseline_layers = _layers_by_id(baseline_data)
    current_layers = _layers_by_id(current_data)
    results: list[dict[str, Any]] = []
    for logical_layer in sorted(set(baseline_layers) | set(current_layers)):
        baseline = baseline_layers.get(logical_layer)
        current = current_layers.get(logical_layer)
        failures: list[str] = []
        if baseline is None or current is None:
            failures.append("data_layer_missing")
        elif _without_transient_fields(baseline) != _without_transient_fields(current):
            failures.append("data_summary_mismatch")
        results.append(
            {
                "logical_layer": logical_layer,
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
            }
        )
    return results


def _layers_by_id(data: object) -> dict[str, dict[str, Any]]:
    if not isinstance(data, dict):
        raise CompareBlockedError("run_data_invalid")
    layers = data.get("layers")
    if not isinstance(layers, list):
        raise CompareBlockedError("run_data_layers_invalid")
    result: dict[str, dict[str, Any]] = {}
    for layer in layers:
        if not isinstance(layer, dict) or not str(layer.get("logical_layer", "")):
            raise CompareBlockedError("run_data_layer_invalid")
        logical_layer = str(layer["logical_layer"])
        if logical_layer in result:
            raise CompareBlockedError("run_data_layer_duplicate")
        result[logical_layer] = layer
    return result


def _without_transient_fields(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_transient_fields(item)
            for key, item in sorted(value.items())
            if key != "created_at"
        }
    if isinstance(value, list):
        return [_without_transient_fields(item) for item in value]
    return value


def _compare_output_pair(
    *,
    baseline_root: Path,
    current_root: Path,
    baseline_run_id: str,
    current_run_id: str,
    baseline_output_id: str,
    current_output_id: str,
) -> dict[str, Any]:
    failures: list[str] = []
    baseline_manifest = _read_output_manifest(
        baseline_root,
        baseline_output_id,
        run_id=baseline_run_id,
        failures=failures,
    )
    current_manifest = _read_output_manifest(
        current_root,
        current_output_id,
        run_id=current_run_id,
        failures=failures,
    )
    if baseline_manifest is not None and current_manifest is not None:
        if baseline_manifest.get("output_definition_id") != current_manifest.get(
            "output_definition_id"
        ):
            failures.append("output_definition_id_mismatch")
        if baseline_manifest.get("metadata") != current_manifest.get("metadata"):
            failures.append("metadata_mismatch")
        baseline_files = baseline_manifest.get("files")
        current_files = current_manifest.get("files")
        if baseline_files != current_files:
            failures.append("file_count_or_names_mismatch")
        elif isinstance(baseline_files, list):
            if _file_hashes(
                baseline_root,
                baseline_output_id,
                baseline_files,
                failures=failures,
            ) != _file_hashes(
                current_root,
                current_output_id,
                current_files,
                failures=failures,
            ):
                failures.append("file_hash_mismatch")
        else:
            failures.append("output_files_invalid")
    return {
        "baseline_output_id": baseline_output_id,
        "current_output_id": current_output_id,
        "status": "PASS" if not failures else "FAIL",
        "failures": sorted(set(failures)),
    }


def _read_output_manifest(
    run_root: Path,
    output_id: str,
    *,
    run_id: str,
    failures: list[str],
) -> dict[str, Any] | None:
    path = run_root / "outputs" / output_id / "manifest.json"
    if not path.is_file():
        failures.append("output_manifest_missing")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        failures.append("output_manifest_unreadable")
        return None
    if not isinstance(payload, dict):
        failures.append("output_manifest_not_object")
        return None
    if payload.get("schema_id") != "qrp-output-manifest":
        failures.append("output_manifest_schema_id_mismatch")
        return None
    identity_reason = qrp_json_identity_reason(payload)
    if identity_reason is not None:
        failures.append(f"output_manifest_identity_invalid:{identity_reason}")
        return None
    if payload.get("output_id") != output_id:
        failures.append("output_manifest_output_id_mismatch")
    if payload.get("run_id") != run_id:
        failures.append("output_manifest_run_id_mismatch")
    return payload


def _file_hashes(
    run_root: Path,
    output_id: str,
    files: list[object],
    *,
    failures: list[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    output_bundle = (run_root / "outputs" / output_id).resolve(strict=False)
    for raw_name in files:
        if not isinstance(raw_name, str) or not raw_name:
            failures.append("output_file_path_invalid")
            continue
        name = raw_name.replace("\\", "/")
        path_parts = Path(name).parts
        if Path(name).is_absolute() or ".." in path_parts or "." in path_parts:
            failures.append("output_file_path_invalid")
            continue
        path = (output_bundle / name).resolve(strict=False)
        if not path.is_relative_to(output_bundle):
            failures.append("output_file_path_invalid")
            continue
        if not path.is_file():
            failures.append("output_file_missing")
            continue
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload.get("run_id"),
        "workflow_id": payload.get("workflow_id"),
        "status": payload.get("status"),
    }


def _write_compare_artifacts(compare_root: Path, payload: dict[str, Any]) -> None:
    compare_root.mkdir(parents=True, exist_ok=True)
    (compare_root / "compare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (compare_root / "summary.md").write_text(
        _render_summary(payload),
        encoding="utf-8",
    )


def _render_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        (
            "# QRP Run Comparison",
            "",
            f"- overall_status: `{payload['overall_status']}`",
            f"- data_results: `{len(payload['data_results'])}`",
            f"- output_results: `{len(payload['output_results'])}`",
            "",
        )
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--compare-root", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = compare_runs(
        baseline_root=args.baseline_root,
        current_root=args.current_root,
        compare_root=args.compare_root,
        pairs_file=args.pairs_file,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["overall_status"] in {"PASS", "NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

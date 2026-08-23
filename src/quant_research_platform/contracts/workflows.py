"""Workflow contracts."""

from __future__ import annotations

from typing import Any


def build_failed_schema_inventory_summary(
    *, failure_codes: list[str], dataset_count: int, total_row_count: int
) -> dict[str, Any]:
    value = {
        "schema_id": "qrp-schema-inventory-summary",
        "status": "FAIL",
        "failure_codes": list(failure_codes),
        "dataset_count": dataset_count,
        "total_row_count": total_row_count,
    }
    return validate_schema_inventory_summary(
        value,
        validation_status="FAIL",
        validation_failure_codes=failure_codes,
        dataset_count=dataset_count,
        total_row_count=total_row_count,
    )


def validate_schema_inventory_summary(
    value: object,
    *,
    validation_status: str,
    validation_failure_codes: list[str],
    dataset_count: int,
    total_row_count: int,
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or type(validation_failure_codes) is not list
        or value.get("schema_id") != "qrp-schema-inventory-summary"
        or type(dataset_count) is not int
        or dataset_count < 0
        or type(value.get("dataset_count")) is not int
        or value.get("dataset_count") != dataset_count
        or type(total_row_count) is not int
        or total_row_count < 0
        or type(value.get("total_row_count")) is not int
        or value.get("total_row_count") != total_row_count
    ):
        raise ValueError("schema_inventory_summary_invalid")
    if validation_status == "PASS" and not validation_failure_codes:
        schema_hash = value.get("schema_hash")
        if (
            dataset_count == 0
            or set(value)
            != {"schema_id", "dataset_count", "total_row_count", "schema_hash"}
            or type(schema_hash) is not str
            or len(schema_hash) != 64
            or any(character not in "0123456789abcdef" for character in schema_hash)
        ):
            raise ValueError("schema_inventory_summary_invalid")
    elif validation_status == "FAIL" and validation_failure_codes:
        if (
            set(value)
            != {
                "schema_id",
                "status",
                "failure_codes",
                "dataset_count",
                "total_row_count",
            }
            or value.get("status") != "FAIL"
            or value.get("failure_codes") != validation_failure_codes
            or any(
                type(code) is not str or not code for code in validation_failure_codes
            )
        ):
            raise ValueError("schema_inventory_summary_invalid")
    else:
        raise ValueError("schema_inventory_summary_invalid")
    return dict(value)

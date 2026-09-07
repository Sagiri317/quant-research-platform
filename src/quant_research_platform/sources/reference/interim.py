"""Reference-source interim target for the A-share fee schedule."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.business_manifest import (
    validate_business_manifest_envelope,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.reference.a_share_fees import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SIDECAR_NAME,
    FeeSourceRow,
    load_a_share_fee_source,
)
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout

SOURCE_ID = SourceId("reference")
A_SHARE_FEE_SCHEDULE_DATA_ID = "a_share_fee_schedule"
FEE_SCHEDULE_COLUMNS = (
    "fee_code",
    "effective_date",
    "end_date",
    "market",
    "side",
    "basis",
    "rate",
    "included_in_ledger",
    "included_in",
)
_FEE_SCHEDULE_ROLES = MappingProxyType(
    {
        "fee_code": ColumnRole.KEY,
        "effective_date": ColumnRole.TIMESTAMP,
        "end_date": ColumnRole.KEY,
        "market": ColumnRole.KEY,
        "side": ColumnRole.KEY,
        "basis": ColumnRole.KEY,
        "rate": ColumnRole.CONTEXT,
        "included_in_ledger": ColumnRole.CONTEXT,
        "included_in": ColumnRole.CONTEXT,
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceInterimImportResult:
    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]


def fee_schedule_spec() -> InterimDataSpec:
    return InterimDataSpec(
        data_id=A_SHARE_FEE_SCHEDULE_DATA_ID,
        source_id=SOURCE_ID,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.MARKET_TIMESTAMP,
        column_roles=_FEE_SCHEDULE_ROLES,
        primary_key=(
            "fee_code",
            "effective_date",
            "end_date",
            "market",
            "side",
            "basis",
        ),
        partition_columns=("year",),
    )


def interim_catalog() -> InterimDataCatalog:
    return InterimDataCatalog((fee_schedule_spec(),))


def source_content_identity(raw_root: Path, data_id: str) -> str | None:
    if data_id != A_SHARE_FEE_SCHEDULE_DATA_ID:
        raise ValueError("unknown_reference_interim_data_id")
    root = Path(raw_root)
    csv_path = root / "a_share_fee_schedule" / A_SHARE_FEE_CSV_NAME
    sidecar_path = root / "a_share_fee_schedule" / A_SHARE_FEE_SIDECAR_NAME
    if not csv_path.is_file() or not sidecar_path.is_file():
        return None
    return load_a_share_fee_source(reference_root=root).source_identity


def materialize_interim_data(
    *,
    layout: StorageLayout,
    raw_root: Path | None = None,
    data_ids: tuple[str, ...] = (A_SHARE_FEE_SCHEDULE_DATA_ID,),
    start_date: str | None = None,
    end_date: str | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
    instance_metadata: Mapping[str, object] | None = None,
) -> ReferenceInterimImportResult:
    del start_date, end_date
    if type(data_ids) is not tuple or data_ids != (A_SHARE_FEE_SCHEDULE_DATA_ID,):
        raise ValueError("reference_interim_data_ids_invalid")
    with runtime_resource_policy_scope(resource_policy) as policy:
        root = (
            Path(raw_root)
            if raw_root is not None
            else layout.project_root / "data/raw" / str(SOURCE_ID)
        )
        identity = source_content_identity(root, A_SHARE_FEE_SCHEDULE_DATA_ID)
        if (
            instance_metadata is not None
            and COMMON_MANIFEST_METADATA_KEY in instance_metadata
        ):
            envelope = validate_business_manifest_envelope(
                instance_metadata[COMMON_MANIFEST_METADATA_KEY]
            )
            if envelope["semantic_metadata"]["raw_content_identity"] != identity:
                raise ValueError("reference_raw_identity_changed")
        source = load_a_share_fee_source(reference_root=root)
        if source.source_identity != identity:
            raise ValueError("reference_raw_identity_changed")

        def verify_source(_receipt):
            if source_content_identity(root, A_SHARE_FEE_SCHEDULE_DATA_ID) != identity:
                raise ValueError("reference_raw_identity_changed")

        spec = fee_schedule_spec()
        frame = _frame_from_rows(source.rows)
        artifact_root = layout.target_root(
            spec.data_id,
            layout.interim_data_root(spec.source_id, spec.data_id),
        )
        write_dataset(
            artifact_root,
            frame,
            spec=spec,
            resource_policy=policy,
            instance_metadata=instance_metadata,
            borrowed_input=True,
            staged_validator=verify_source,
        )
        return ReferenceInterimImportResult(
            artifact_roots={spec.data_id: artifact_root},
            skipped=(),
        )


def _frame_from_rows(rows: tuple[FeeSourceRow, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "fee_code": row.fee_code,
                "effective_date": pd.Timestamp(row.effective_date),
                "end_date": pd.Timestamp(row.end_date),
                "market": row.market,
                "side": row.side,
                "basis": row.basis,
                "rate": row.rate,
                "included_in_ledger": row.included_in_ledger,
                "included_in": row.included_in,
            }
            for row in rows
        ],
        columns=FEE_SCHEDULE_COLUMNS,
    )
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame["rate"] = frame["rate"].astype("float64")
    frame["included_in_ledger"] = frame["included_in_ledger"].astype("bool")
    frame["included_in"] = frame["included_in"].astype("string")
    return frame

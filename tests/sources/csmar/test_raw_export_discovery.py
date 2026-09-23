from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.csmar import field_mappings, raw_exports


def test_requested_dataset_rejects_ambiguous_header(tmp_path: Path) -> None:
    columns = tuple(
        dict.fromkeys(
            (
                *field_mappings.RAW_TABLE_CONTRACTS["stock_master"].required_columns,
                *field_mappings.RAW_TABLE_CONTRACTS["market_cap"].required_columns,
            )
        )
    )
    (tmp_path / "unknown.csv").write_text(
        ",".join(columns) + "\n" + ",".join("1" for _ in columns) + "\n",
        encoding="utf-8",
    )

    with runtime_resource_policy_scope() as policy:
        with pytest.raises(
            ValueError, match="Cannot infer CSMAR dataset from raw header"
        ):
            tuple(
                raw_exports.dataset_files_with_identity_by_id(
                    tmp_path,
                    resource_policy=policy,
                    requested_datasets=("market_cap",),
                )
            )

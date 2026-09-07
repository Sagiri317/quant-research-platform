from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from quant_research_platform.profiles.research import research_profile
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.reference import interim
from quant_research_platform.sources.reference.a_share_fees import (
    FEE_CSV_COLUMNS,
    load_a_share_fee_source,
)
from quant_research_platform.sources.reference.interim import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SCHEDULE_DATA_ID,
    A_SHARE_FEE_SIDECAR_NAME,
)
from quant_research_platform.storage.datasets.reading import read_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.workflows import research


def _write_reference_root(root: Path, *, commission: float = 0.0003) -> Path:
    schedule_root = root / "reference" / "a_share_fee_schedule"
    schedule_root.mkdir(parents=True, exist_ok=True)
    csv_path = schedule_root / A_SHARE_FEE_CSV_NAME
    rows = [
        _row(
            "BROKER_COMMISSION",
            "2019-01-01",
            "2025-12-31",
            "沪深",
            "买入+卖出",
            commission,
        ),
        _row(
            "TRANSFER_FEE",
            "2019-01-01",
            "2025-12-31",
            "沪深",
            "买入+卖出",
            0.00001,
        ),
        _row(
            "STAMP_DUTY",
            "2019-01-01",
            "2025-12-31",
            "沪深",
            "卖出",
            0.001,
        ),
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sidecar = {
        "file_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "file_bytes": csv_path.stat().st_size,
        "row_count": len(rows),
        "fields_raw": list(FEE_CSV_COLUMNS),
    }
    (schedule_root / A_SHARE_FEE_SIDECAR_NAME).write_text(
        json.dumps(sidecar, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _row(
    fee_code: str,
    start: str,
    end: str,
    market: str,
    direction: str,
    rate: float,
) -> dict[str, str]:
    row = dict.fromkeys(FEE_CSV_COLUMNS, "")
    row.update(
        {
            "费用代码": fee_code,
            "生效日": start,
            "截止日": end,
            "市场": market,
            "方向": direction,
            "计费基数": "成交金额",
            "十进制费率": str(rate),
            "费率(‰)": str(rate * 1_000.0),
            "费率(bps)": str(rate * 10_000.0),
            "进入正式账本": "是",
        }
    )
    return row


def test_fee_interim_target_routes_build_reuse_and_identity_invalidation(
    tmp_path: Path,
) -> None:
    raw_root = _write_reference_root(tmp_path / "raw")
    raw_identity = load_a_share_fee_source(
        reference_root=raw_root / "reference"
    ).source_identity
    layout = StorageLayout(project_root=tmp_path)
    profile = research_profile()

    plan = research.plan_targets(
        profile=profile,
        action="data materialize",
        terminal_targets=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
        layout=layout,
        raw_root=raw_root,
    )
    planned = plan["targets"][0]
    assert planned["decision"] == "BUILD"
    assert planned["requested_identity"]

    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    first_target = first["target_results"][0]
    assert first_target["status"] == "BUILT"
    with runtime_resource_policy_scope(None) as policy:
        first_frame = read_dataset(
            layout.instance_root(
                layout.interim_data_root("reference", A_SHARE_FEE_SCHEDULE_DATA_ID),
                first_target["instance_identity"],
            ),
            resource_policy=policy,
        )
    manifest = read_dataset_manifest_evidence(
        layout.instance_root(
            layout.interim_data_root("reference", A_SHARE_FEE_SCHEDULE_DATA_ID),
            first_target["instance_identity"],
        )
    )
    assert (
        manifest["common_manifest"]["semantic_metadata"]["raw_content_identity"]
        == raw_identity
    )
    assert tuple(first_frame.columns) == (
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

    second = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    assert second["target_results"][0]["status"] == "REUSED"

    sidecar_path = (
        raw_root / "reference" / "a_share_fee_schedule" / A_SHARE_FEE_SIDECAR_NAME
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["notes"] = "identity mutation"
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False),
        encoding="utf-8",
    )
    third = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    assert third["target_results"][0]["status"] == "BUILT"
    assert (
        third["target_results"][0]["instance_identity"]
        != first_target["instance_identity"]
    )


@pytest.mark.parametrize("mutation", ("planned", "loaded", "staged"))
@pytest.mark.parametrize("existing_target", (False, True))
def test_fee_source_identity_race_preserves_old_good(
    tmp_path, monkeypatch, mutation, existing_target
):
    raw_root = _write_reference_root(tmp_path / "raw")
    layout = StorageLayout(project_root=tmp_path)
    built = research.data_materialize(
        profile=research_profile(),
        layout=layout,
        raw_root=raw_root,
        start_date=None,
        end_date=None,
        data_ids=(A_SHARE_FEE_SCHEDULE_DATA_ID,),
    )
    identity = built["target_results"][0]["instance_identity"]
    root = layout.instance_root(
        layout.interim_data_root("reference", A_SHARE_FEE_SCHEDULE_DATA_ID), identity
    )
    metadata = {
        "common_manifest": read_dataset_manifest_evidence(root)["common_manifest"]
    }
    old_good = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    target = (
        root
        if existing_target
        else StorageLayout(tmp_path / "candidate").interim_data_root(
            "reference", A_SHARE_FEE_SCHEDULE_DATA_ID
        )
    )
    target_layout = layout.with_target_roots({A_SHARE_FEE_SCHEDULE_DATA_ID: target})
    target.parent.mkdir(parents=True, exist_ok=True)
    unrelated = target.parent / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    previous_members = set(target.parent.iterdir())
    if mutation == "planned":
        _write_reference_root(raw_root, commission=0.0004)
    elif mutation == "loaded":
        original = interim.load_a_share_fee_source
        calls = 0

        def load(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                _write_reference_root(raw_root, commission=0.0004)
            return original(**kwargs)

        monkeypatch.setattr(interim, "load_a_share_fee_source", load)
    else:
        original_frame = interim._frame_from_rows

        def frame(rows):
            value = original_frame(rows)
            _write_reference_root(raw_root, commission=0.0004)
            return value

        monkeypatch.setattr(interim, "_frame_from_rows", frame)
    with runtime_resource_policy_scope() as policy:
        expected_error = (
            "dataset_exists"
            if existing_target and mutation == "staged"
            else "reference_raw_identity_changed"
        )
        with pytest.raises(ValueError, match=expected_error):
            interim.materialize_interim_data(
                layout=target_layout,
                raw_root=raw_root / "reference",
                instance_metadata=metadata,
                resource_policy=policy,
            )
        assert policy.used_memory_bytes == 0
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == old_good
    assert set(target.parent.iterdir()) == previous_members
    assert unrelated.read_text(encoding="utf-8") == "keep"
    if not existing_target:
        assert not target.exists()

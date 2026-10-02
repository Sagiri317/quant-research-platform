from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.artifacts.evidence import (
    read_json_evidence,
    write_json_evidence,
    write_markdown_evidence,
)
from quant_research_platform.contracts.json_codec import json_record_working_set_bytes
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.runtime_resources import capture_test_policy


def test_evidence_writers_only_persist_given_payload_or_text(tmp_path: Path) -> None:
    payload = {"schema_id": "qrp-test-evidence", "status": "PASS"}
    json_path = write_json_evidence(tmp_path / "evidence.json", payload)
    md_path = write_markdown_evidence(tmp_path / "evidence.md", "# Evidence\n")

    assert json.loads(json_path.read_text(encoding="utf-8")) == payload
    assert md_path.read_text(encoding="utf-8") == "# Evidence\n"


def test_json_evidence_requires_schema_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence_json_identity_invalid"):
        write_json_evidence(tmp_path / "evidence.json", {"status": "PASS"})


def test_json_evidence_rejects_versioned_schema_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evidence_json_identity_invalid"):
        write_json_evidence(
            tmp_path / "evidence.json",
            {"schema_id": "qrp-test/0.1", "status": "PASS"},
        )


def test_evidence_writers_reject_private_text_before_publish(tmp_path: Path) -> None:
    json_path = tmp_path / "unsafe.json"
    markdown_path = tmp_path / "unsafe.md"

    with pytest.raises(ValueError, match="unsafe_evidence: absolute_path"):
        write_json_evidence(
            json_path,
            {
                "schema_id": "qrp-test-evidence",
                "detail": "C:\\private\\raw.csv",
            },
        )
    with pytest.raises(ValueError, match="unsafe_evidence: traceback"):
        write_markdown_evidence(
            markdown_path,
            "Traceback (most recent call last):\nsynthetic failure\n",
        )

    assert not json_path.exists()
    assert not markdown_path.exists()


@pytest.mark.parametrize("operation", ["read", "borrowed_read", "write"])
@pytest.mark.parametrize("boundary", ["physical", "explicit", "soft"])
def test_json_evidence_admits_whole_workspace_before_io(
    tmp_path, monkeypatch, operation, boundary
):
    from quant_research_platform.artifacts import evidence

    payload = {"schema_id": "qrp-test-evidence", "text": "x" * 4096}
    path = write_json_evidence(tmp_path / "evidence.json", payload)
    output = tmp_path / "written.json"
    original = path.read_bytes()
    required = json_record_working_set_bytes(len(original))
    retained = required // 2 if operation == "borrowed_read" else 0
    capacity = required + retained + (1 if boundary == "soft" else -1)
    policy = capture_test_policy(
        available_memory_bytes=capacity if boundary != "explicit" else 2 * capacity,
        explicit_cap_bytes=capacity if boundary == "explicit" else None,
    )
    io_calls = []
    io_name = "atomic_write_json" if operation == "write" else "read_bounded_qrp_json"
    original_io = getattr(evidence, io_name)

    def observed_io(*args, **kwargs):
        io_calls.append(True)
        return original_io(*args, **kwargs)

    monkeypatch.setattr(evidence, io_name, observed_io)
    with runtime_resource_policy_scope(policy):
        with policy.lease("test.retained", retained) as borrowed:

            def execute():
                if operation == "write":
                    write_json_evidence(output, payload)
                else:
                    with read_json_evidence(
                        path,
                        expected_schema_id=payload["schema_id"],
                        error="synthetic_invalid",
                        lease=borrowed if operation == "borrowed_read" else None,
                    ) as decoded:
                        assert decoded == payload

            if boundary == "soft":
                assert required + retained > policy.memory_budget_bytes
                execute()
                assert io_calls == [True]
            else:
                with pytest.raises(
                    MemoryError if boundary == "explicit" else ValueError,
                    match=(
                        "runtime_memory_budget_exceeded"
                        if boundary == "explicit"
                        else "evidence_json_working_set_exceeds_capacity"
                    ),
                ):
                    execute()
                assert io_calls == []
                assert borrowed.reserved_bytes == retained
        assert policy.used_memory_bytes == 0
    assert path.read_bytes() == original
    if operation == "write":
        if boundary == "soft":
            assert output.read_bytes() == original
        else:
            assert not output.exists()

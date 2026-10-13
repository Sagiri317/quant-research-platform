"""Read-only API and query acceptance against actual published synthetic runs."""

from __future__ import annotations

import asyncio
import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from quant_research_platform.artifacts import research as report_artifact
from quant_research_platform.artifacts.evidence import (
    write_json_evidence,
    write_markdown_evidence,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.interfaces import queries
from quant_research_platform.interfaces.api import app as api_module
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    CRYPTO_PROFILE_ID,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_stat,
    filesystem_walk,
    filesystem_write_bytes,
    native_filesystem_path,
    open_filesystem_file,
)
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact,
    write_semantic_artifact,
)
from quant_research_platform.workflows import operations, research

_FORBIDDEN_SCIENTIFIC_ACTIONS = (
    "_plan_targets",
    "preflight_model_specs",
    "_default_raw_root",
)
_FORBIDDEN_OPERATIONS = (
    "plan_targets",
    "_execute_action",
    "data_materialize",
    "dataset_build",
    "research_run",
    "report_build",
)


def _forbidden(*_args, **_kwargs):
    raise AssertionError("read_only_consumer_started_work")


def _files(root):
    result = {}
    for parent, _directories, files in filesystem_walk(root):
        for name in files:
            path = parent / name
            with open_filesystem_file(path, "rb") as handle:
                result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    return result


def check_published_evidence_views(
    *,
    layout,
    profile,
    run_id,
    other_run_id,
    report_identity,
    monkeypatch,
):
    record = report_artifact.read_research_run_record(layout.research_run_path(run_id))
    target_id = f"{record.scenario_id}_report_artifact"
    logical = layout.semantic_artifact_root(profile.market_id, target_id)
    root = layout.instance_root(logical, report_identity)
    envelope = read_semantic_artifact(root, expected_identity=report_identity)
    payload = report_artifact.read_research_report_artifact(
        root, expected_identity=report_identity
    )
    previous = build_business_manifest_envelope(
        logical_artifact_id=target_id,
        schema_identity=envelope["schema_identity"],
        protocol_or_spec_identity=envelope["protocol_or_spec_identity"],
        software_identity=canonical_json_hash(
            {"synthetic_renderer": "archived_report"}
        ),
        research_window=envelope["research_window"],
        upstream_identities=envelope["upstream_identities"],
        semantic_metadata=envelope["semantic_metadata"],
    )
    archived_payload = deepcopy(payload)
    archived_payload["sections"]["Data/Coverage"]["scale diagnostics"] = {
        "results": [
            {"metric_id": "synthetic_metric", "value": 0.1} for _ in range(20000)
        ]
    }
    archived_markdown = "# Archived synthetic evidence\n\nRead the exact report.json.\n"

    def archived_writer(staging):
        write_json_evidence(staging / "report.json", archived_payload)
        write_markdown_evidence(staging / "report.md", archived_markdown)

    previous_root = layout.instance_root(logical, previous.instance_identity)
    if filesystem_exists(previous_root):
        assert (
            report_artifact.read_research_report_artifact(
                previous_root, expected_identity=previous.instance_identity
            )
            == archived_payload
        )
    else:
        write_semantic_artifact(logical, previous, archived_writer)
    assert filesystem_stat(previous_root / "report.json").st_size > 1024 * 1024
    before = _files(layout.project_root / "data")
    guarded = replace(
        profile,
        producer_bindings={
            key: replace(binding, producer=_forbidden, source_identity=_forbidden)
            for key, binding in profile.producer_bindings.items()
        },
    )
    original_profile = api_module.research_profile

    with monkeypatch.context() as patch:
        for owner, names in (
            (research, _FORBIDDEN_SCIENTIFIC_ACTIONS),
            (operations, _FORBIDDEN_OPERATIONS),
        ):
            for name in names:
                patch.setattr(owner, name, _forbidden)
        patch.setattr(report_artifact, "render_research_report", _forbidden)
        patch.setattr(
            api_module,
            "research_profile",
            lambda value: (
                guarded if value == guarded.profile_id else original_profile(value)
            ),
        )

        async def check_http():
            async with AsyncClient(
                transport=ASGITransport(
                    app=api_module.create_app(project_root=layout.project_root)
                ),
                base_url="http://testserver",
            ) as client:
                params = {"profile": profile.profile_id}
                directory = await client.get("/api/status", params=params)
                assert directory.status_code == 200 and "status" not in directory.json()
                assert run_id in {row["run_id"] for row in directory.json()["runs"]}
                selected = await client.get(
                    "/api/status", params={**params, "run_id": run_id}
                )
                assert (
                    selected.status_code == 200 and selected.json() == record.payload()
                )
                inspection = await client.get(
                    f"/api/inspect/{target_id}",
                    params={**params, "run_id": run_id},
                )
                assert inspection.status_code == 200
                assert {
                    row["instance_identity"] for row in inspection.json()["instances"]
                } >= {
                    report_identity,
                    previous.instance_identity,
                }
                assert inspection.json()["blocker"] == "report_identity_required"
                view = await client.get(
                    f"/api/runs/{run_id}/reports/{report_identity}",
                    params=params,
                )
                assert view.status_code == 200 and view.json()["report"] == payload
                archived = queries.read_report(
                    profile=guarded,
                    layout=layout,
                    run_id=run_id,
                    report_identity=previous.instance_identity,
                )
                assert archived["markdown"] == archived_markdown
                assert archived["report"] == archived_payload
                large_view = await client.get(
                    f"/api/runs/{run_id}/reports/{previous.instance_identity}",
                    params=params,
                )
                assert large_view.status_code == 200
                assert large_view.json()["report"] == archived_payload
                portfolio_target = next(
                    row.target_id
                    for row in record.target_results
                    if row.producer_kind.value == "PORTFOLIO"
                )
                curve = await client.get(
                    f"/api/runs/{run_id}/portfolios/{portfolio_target}",
                    params=params,
                )
                assert curve.status_code == 200 and curve.json()["status"] == "DEFINED"
                assert all(
                    set(point) == {"sequence", "timestamp_ns", "phase", "nav"}
                    for point in curve.json()["points"]
                )
                assert (
                    curve.json()["source_point_count"]
                    == curve.json()["displayed_point_count"]
                )
                wrong_profile = (
                    CRYPTO_PROFILE_ID
                    if profile.profile_id == A_SHARE_PROFILE_ID
                    else A_SHARE_PROFILE_ID
                )
                failures = [
                    await client.get(
                        f"/api/runs/{other_run_id}/reports/{report_identity}",
                        params=params,
                    ),
                    await client.get(
                        f"/api/runs/{run_id}/reports/{'0' * 64}", params=params
                    ),
                    await client.get(
                        f"/api/runs/{run_id}/reports/{report_identity}",
                        params={"profile": wrong_profile},
                    ),
                    await client.get(
                        f"/api/runs/{run_id}/portfolios/not_in_run",
                        params=params,
                    ),
                ]
                saved = Path(native_filesystem_path(root / "report.json")).read_bytes()
                try:
                    filesystem_write_bytes(
                        root / "report.json",
                        b"invalid_private_payload_must_not_leak",
                    )
                    failures.append(
                        await client.get(
                            f"/api/runs/{run_id}/reports/{report_identity}",
                            params=params,
                        )
                    )
                finally:
                    filesystem_write_bytes(root / "report.json", saved)
                saved_markdown = Path(
                    native_filesystem_path(root / "report.md")
                ).read_bytes()
                actual_read_markdown = report_artifact._read_utf8

                def changed_after_validation(*args, **kwargs):
                    filesystem_write_bytes(
                        root / "report.md", b"x" * len(saved_markdown)
                    )
                    return actual_read_markdown(*args, **kwargs)

                try:
                    with monkeypatch.context() as race:
                        race.setattr(
                            report_artifact,
                            "_read_utf8",
                            changed_after_validation,
                        )
                        failed = await client.get(
                            f"/api/runs/{run_id}/reports/{report_identity}",
                            params=params,
                        )
                        assert failed.status_code == 422
                finally:
                    filesystem_write_bytes(root / "report.md", saved_markdown)
                assert all(response.status_code in {404, 422} for response in failures)
                assert all(
                    "invalid_private_payload" not in response.text
                    and str(layout.project_root) not in response.text
                    for response in failures
                )

        asyncio.run(check_http())
    assert _files(layout.project_root / "data") == before

    if profile.profile_id == CRYPTO_PROFILE_ID:
        expected_nav = {
            row["candidate_id"]: row["economics"]["nav"]
            for row in payload["sections"]["portfolio"]["candidates"]
        }
    else:
        expected_nav = {
            row["candidate_id"]: next(
                metric["value"]
                for metric in row["metrics"]
                if metric["metric_id"] == "terminal_nav"
            )
            for row in payload["sections"]["portfolio"]["results"]
        }
    curves = [
        queries.portfolio_curve(
            profile=guarded,
            layout=layout,
            run_id=run_id,
            target_id=row.target_id,
        )
        for row in record.target_results
        if row.producer_kind.value == "PORTFOLIO"
    ]
    assert {curve["candidate_id"] for curve in curves} == set(expected_nav)
    for curve in curves:
        assert curve["points"][-1]["nav"] == pytest.approx(
            expected_nav[curve["candidate_id"]], rel=1e-12, abs=0
        )
        assert curve["source_point_count"] == curve["displayed_point_count"]
    assert _files(layout.project_root / "data") == before

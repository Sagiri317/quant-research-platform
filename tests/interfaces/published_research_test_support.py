"""Read-only API and notebook acceptance against actual published synthetic runs."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

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
from quant_research_platform.runtime import resources
from quant_research_platform.storage.identity import native_filesystem_path
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact,
    write_semantic_artifact,
)
from quant_research_platform.workflows import research

_FORBIDDEN_ACTIONS = (
    "plan_targets",
    "_plan_targets",
    "_execute_action",
    "preflight_model_specs",
    "_default_raw_root",
    "data_materialize",
    "dataset_build",
    "research_run",
    "report_build",
)


def _forbidden(*_args, **_kwargs):
    raise AssertionError("read_only_consumer_started_work")


def _files(root):
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
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
    root = Path(native_filesystem_path(layout.instance_root(logical, report_identity)))
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
    archived_markdown = "# Archived synthetic evidence\n\nRead the exact report.json.\n"

    def archived_writer(staging):
        write_json_evidence(staging / "report.json", payload)
        write_markdown_evidence(staging / "report.md", archived_markdown)

    previous_root = layout.instance_root(logical, previous.instance_identity)
    if previous_root.exists():
        assert (
            report_artifact.read_research_report_artifact(
                previous_root, expected_identity=previous.instance_identity
            )
            == payload
        )
    else:
        write_semantic_artifact(logical, previous, archived_writer)
    before = _files(layout.project_root / "data")
    guarded = replace(
        profile,
        producer_bindings={
            key: replace(binding, producer=_forbidden, source_identity=_forbidden)
            for key, binding in profile.producer_bindings.items()
        },
    )
    original_profile = api_module.research_profile
    original_capture = resources._capture_runtime_resource_policy
    captured = []

    def capture(*args, **kwargs):
        policy = original_capture(*args, **kwargs)
        captured.append(policy)
        return policy

    with monkeypatch.context() as patch:
        for name in _FORBIDDEN_ACTIONS:
            patch.setattr(research, name, _forbidden)
        patch.setattr(report_artifact, "_markdown", _forbidden)
        patch.setattr(resources, "_capture_runtime_resource_policy", capture)
        patch.setattr(
            api_module,
            "research_profile",
            lambda value: (
                guarded if value == guarded.profile_id else original_profile(value)
            ),
        )
        client = TestClient(api_module.create_app(project_root=layout.project_root))
        params = {"profile": profile.profile_id}
        directory = client.get("/api/0.1/status", params=params)
        assert directory.status_code == 200 and "status" not in directory.json()
        assert run_id in {row["run_id"] for row in directory.json()["runs"]}
        selected = client.get("/api/0.1/status", params={**params, "run_id": run_id})
        assert selected.status_code == 200 and selected.json() == record.payload()
        inspection = client.get(
            f"/api/0.1/inspect/{target_id}", params={**params, "run_id": run_id}
        )
        assert inspection.status_code == 200
        assert {row["instance_identity"] for row in inspection.json()["instances"]} >= {
            report_identity,
            previous.instance_identity,
        }
        assert inspection.json()["blocker"] == "report_identity_required"
        view = client.get(
            f"/api/0.1/runs/{run_id}/reports/{report_identity}", params=params
        )
        assert view.status_code == 200 and view.json()["report"] == payload
        archived = queries.read_report(
            profile=guarded,
            layout=layout,
            run_id=run_id,
            report_identity=previous.instance_identity,
        )
        assert archived["markdown"] == archived_markdown
        portfolio_target = next(
            row.target_id
            for row in record.target_results
            if row.producer_kind.value == "PORTFOLIO"
        )
        curve = client.get(
            f"/api/0.1/runs/{run_id}/portfolios/{portfolio_target}", params=params
        )
        assert curve.status_code == 200 and curve.json()["status"] == "DEFINED"
        assert all(
            set(point) == {"sequence", "timestamp_ns", "phase", "nav"}
            for point in curve.json()["points"]
        )
        assert (
            curve.json()["source_point_count"] == curve.json()["displayed_point_count"]
        )
        assert len(captured) == 6 and all(
            policy.state == "CLOSED" and policy.used_memory_bytes == 0
            for policy in captured
        )
        wrong_profile = (
            CRYPTO_PROFILE_ID
            if profile.profile_id == A_SHARE_PROFILE_ID
            else A_SHARE_PROFILE_ID
        )
        failures = [
            client.get(
                f"/api/0.1/runs/{other_run_id}/reports/{report_identity}", params=params
            ),
            client.get(f"/api/0.1/runs/{run_id}/reports/{'0' * 64}", params=params),
            client.get(
                f"/api/0.1/runs/{run_id}/reports/{report_identity}",
                params={"profile": wrong_profile},
            ),
            client.get(f"/api/0.1/runs/{run_id}/portfolios/not_in_run", params=params),
        ]
        saved = (root / "report.json").read_bytes()
        try:
            (root / "report.json").write_bytes(b"invalid_private_payload_must_not_leak")
            failures.append(
                client.get(
                    f"/api/0.1/runs/{run_id}/reports/{report_identity}", params=params
                )
            )
        finally:
            (root / "report.json").write_bytes(saved)
        saved_markdown = (root / "report.md").read_bytes()
        actual_read_report = report_artifact.read_research_report_artifact

        def changed_after_validation(*args, **kwargs):
            result = actual_read_report(*args, **kwargs)
            (root / "report.md").write_bytes(b"x" * len(saved_markdown))
            return result

        try:
            with monkeypatch.context() as race:
                race.setattr(
                    report_artifact,
                    "read_research_report_artifact",
                    changed_after_validation,
                )
                failed = client.get(
                    f"/api/0.1/runs/{run_id}/reports/{report_identity}", params=params
                )
                assert failed.status_code == 422
        finally:
            (root / "report.md").write_bytes(saved_markdown)
        assert all(response.status_code in {404, 422} for response in failures)
        assert all(
            "invalid_private_payload" not in response.text
            and str(layout.project_root) not in response.text
            for response in failures
        )
        assert all(
            policy.state == "CLOSED" and policy.used_memory_bytes == 0
            for policy in captured
        )
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
    _execute_notebook(
        layout=layout,
        profile_id=profile.profile_id,
        run_id=run_id,
        other_run_id=other_run_id,
        report_identity=report_identity,
        expected_nav=expected_nav,
        expected_status=payload["report_status"],
    )
    assert _files(layout.project_root / "data") == before


def _execute_notebook(
    *,
    layout,
    profile_id,
    run_id,
    other_run_id,
    report_identity,
    expected_nav,
    expected_status,
):
    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    repository = Path(__file__).resolve().parents[2]
    source = repository / "notebooks/research-interview-demo.ipynb"
    source_bytes = source.read_bytes()
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.kernelspec.name == "quantlab313"
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook.cells
    )
    evidence = repository / ".run/notebook-evidence" / profile_id / uuid4().hex
    kernel = evidence / "kernels/quantlab313"
    kernel.mkdir(parents=True)
    empty_cwd = evidence / "empty-cwd"
    empty_cwd.mkdir()
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Python (quantlab313)",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    parameters = next(
        cell for cell in notebook.cells if "parameters" in cell.metadata.get("tags", [])
    )
    parameters.source = (
        f"PROJECT_ROOT = {str(layout.project_root)!r}\nPROFILE_ID = {profile_id!r}\n"
        f"RUN_ID = {run_id!r}\nREPORT_IDENTITY = {report_identity!r}\n"
        "TARGET_IDS = ()\nNAV_STRIDE = 1\n"
    )
    guard = nbformat.v4.new_code_cell(
        "import sys\nfrom pathlib import Path\nfrom dataclasses import replace\n"
        "from unittest.mock import patch\n"
        "from quant_research_platform.workflows import research as guarded_workflow\n"
        "from quant_research_platform.artifacts import research as guarded_report\n"
        "from quant_research_platform.profiles import research as guarded_profiles\n"
        "def forbidden(*args, **kwargs):\n"
        "    raise AssertionError('notebook_started_work')\n"
        f"assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve()\n"
        "assert Path(sys.prefix).name == 'quantlab313'\n"
        "guards = [patch.object(guarded_workflow, name, forbidden) "
        f"for name in {_FORBIDDEN_ACTIONS!r}]\n"
        "guards.append(patch.object(guarded_report, '_markdown', forbidden))\n"
        "for guard in guards:\n    guard.start()\n"
        "actual_profile = guarded_profiles.research_profile\n"
        "def read_only_profile(profile_id):\n"
        "    result = actual_profile(profile_id)\n"
        "    return replace(result, producer_bindings={\n"
        "        key: replace(binding, producer=forbidden, source_identity=forbidden)\n"
        "        for key, binding in result.producer_bindings.items()})\n"
        "guarded_profiles.research_profile = read_only_profile\n"
    )
    notebook.cells.insert(0, guard)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import math\n"
            f"expected_nav = {expected_nav!r}\n"
            "assert {row['candidate_id'] for row in checks} == set(expected_nav)\n"
            "assert all(math.isclose(row['last_nav'], "
            "expected_nav[row['candidate_id']], rel_tol=1e-12) for row in checks)\n"
            f"assert report['report_status'] == {expected_status!r}\n"
            "assert all(curve['source_point_count'] == curve['displayed_point_count'] "
            "for curve in available)\n"
            "assert all(axis.get_xlabel() == "
            "('Time (UTC)' if is_crypto else 'Trading date') "
            "for axis in axes.flat if axis.get_visible())\n"
            "print('NOTEBOOK_SYNTHETIC_CHECKS_PASSED')\n"
        )
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(repository / "src"),
        "PYTHONUTF8": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(evidence / "mpl"),
        "IPYTHONDIR": str(evidence / "ipython"),
        "JUPYTER_RUNTIME_DIR": str(evidence / "runtime"),
    }

    def execute(value):
        manager = AsyncKernelManager(
            kernel_name="quantlab313",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]),
        )
        client = NotebookClient(
            value,
            km=manager,
            kernel_name="quantlab313",
            timeout=180,
            record_timing=False,
        )
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(
                client.async_execute(cleanup_kc=True, cwd=str(empty_cwd), env=env)
            )

    executed = execute(notebook)
    nbformat.write(executed, evidence / "executed.ipynb")
    images = []
    for cell in executed.cells:
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                images.append(base64.b64decode(encoded))
    assert images
    for index, data in enumerate(images):
        (evidence / f"figure-{index}.png").write_bytes(data)
    assert "NOTEBOOK_SYNTHETIC_CHECKS_PASSED" in json.dumps(executed)
    assert source.read_bytes() == source_bytes
    parameters.source = parameters.source.replace(
        f"RUN_ID = {run_id!r}", f"RUN_ID = {other_run_id!r}"
    )
    with pytest.raises(CellExecutionError) as mismatch:
        execute(notebook)
    assert mismatch.value.ename == "ValueError"
    assert mismatch.value.evalue == "research_report_run_lineage_mismatch"
    assert source.read_bytes() == source_bytes
    (evidence / "validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "profile_id": profile_id,
                "run_id": run_id,
                "other_run_id": other_run_id,
                "report_identity": report_identity,
                "source_notebook_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "candidate_count": len(expected_nav),
                "figure_count": len(images),
                "source_unchanged": True,
                "wrong_run_rejected": True,
                "evidence_kind": "synthetic_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

from __future__ import annotations

import ast
from pathlib import Path

from tests.architecture.governance_scan_helpers import (
    APPROVED_WRITER_MODULES,
    FORBIDDEN_WRITER_METHODS,
    RUN_OUTPUT_ARTIFACTS,
    RUNTIME_PROFILING,
    call_name,
    class_field_names,
    function_arg_names,
    is_approved_writer_path,
    is_open_call,
    is_open_write_call,
    is_write_mode,
    iter_governance_python_files,
    open_mode,
)


def test_business_logic_does_not_bypass_persistent_writers() -> None:
    hits: list[str] = []
    for path in iter_governance_python_files():
        if is_approved_writer_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            hits.append(f"{path}:{exc.lineno}: syntax_error: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called_name = call_name(node.func)
            if called_name in FORBIDDEN_WRITER_METHODS:
                hits.append(f"{path}:{node.lineno}: {called_name}")
                continue
            if is_open_write_call(node):
                hits.append(f"{path}:{node.lineno}: open(...).write")
                continue
            if is_open_call(node.func):
                mode = open_mode(node)
                if mode is None:
                    hits.append(f"{path}:{node.lineno}: open with dynamic mode")
                elif is_write_mode(mode):
                    hits.append(f"{path}:{node.lineno}: open mode={mode!r}")

    assert hits == []

def test_retired_handoff_receipt_writers_are_not_approved_writer_modules() -> None:
    retired = {
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "verify_governance.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "run_" + "package_" + "smoke.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "collect_handoff_receipts.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "run_check.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "proof_contract.py"
        ),
    }

    assert APPROVED_WRITER_MODULES.isdisjoint(retired)

def test_direct_write_gate_detects_chained_open_write() -> None:
    tree = ast.parse("open('artifact.txt', 'w').write('unsafe')\n")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(is_open_write_call(call) for call in calls)

def test_approved_writer_signatures_do_not_accept_arbitrary_physical_roots() -> None:
    writer_arg_sets = (
        function_arg_names(RUN_OUTPUT_ARTIFACTS, "write_stage_profile_outputs"),
        function_arg_names(RUN_OUTPUT_ARTIFACTS, "write_prediction_outputs"),
        function_arg_names(RUN_OUTPUT_ARTIFACTS, "write_evaluation_outputs"),
    )

    assert "output_dir" not in class_field_names(RUNTIME_PROFILING, "ProfileConfig")
    for arg_names in writer_arg_sets:
        assert "".join(("output_", "root")) not in arg_names
        assert "output_dir" not in arg_names

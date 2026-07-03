# ruff: noqa: F403,F405
from __future__ import annotations

from tests.architecture.governance_scan_helpers import *


def test_source_tree_rejects_retired_local_verification_surface_terms(
    tmp_path: Path,
) -> None:
    from governance_contract import source_tree_active_test_literal_findings

    repo = tmp_path / "repo"
    test_file = repo / "tests" / "ordinary" / "test_local_verification_surface.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "def test_ordinary_path_sample():",
                "    run_root = Path('verification-output/s1')",
                "    run_id = 'verification_output'",
                "",
                "def test_window_outside_evidence_range():",
                "    label = 'evidence_range'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)

    findings = source_tree_active_test_literal_findings(repo)

    assert {
        "retired_local_verification_output_surface",
        "retired_local_verification_evidence_range_surface",
    }.issubset({str(finding["rule"]) for finding in findings})

def test_local_verification_replaces_retired_public_names() -> None:
    expected_files = {
        Path("src/quant_research_platform/workflows/local_verification.py"),
        Path("src/quant_research_platform/reports/verification_summary.py"),
    }
    retired_files = {
        Path("src/quant_research_platform/workflows/" + "private" + "_evidence.py"),
        Path("src/quant_research_platform/reports/" + "private" + "_evidence.py"),
        Path("src/quant_research_platform/reports/evidence" + "_rendering.py"),
    }
    forbidden_terms = (
        "private" + "_evidence",
        "Private" + "Evidence",
        "run_" + "private" + "_evidence",
        "build_" + "private" + "_evidence_summary",
        "write_" + "private" + "_evidence_summary",
        "output_root_must_be_under_" + "run",
    )
    missing = sorted(str(path) for path in expected_files if not path.is_file())
    present = sorted(str(path) for path in retired_files if path.exists())
    hits: list[str] = []
    current_file = Path(__file__).resolve()
    for path in _iter_text_files():
        if path.resolve() == current_file:
            continue
        try:
            text = _public_surface_scan_text(path)
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(term in line for term in forbidden_terms):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert missing == []
    assert present == []
    assert hits == []

def test_status_aggregation_definitions_are_owned_by_policy_check_status() -> None:
    owner = Path("src/quant_research_platform/policy/check_status.py")
    forbidden_names = {
        "aggregate_component_status",
        "aggregate_validation_matrix_status",
        "overall_verification_status",
    }
    findings: list[str] = []
    for root in (Path("src"), Path(".agents")):
        for path in root.rglob("*.py"):
            if path == owner:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in forbidden_names:
                        findings.append(f"{path}:{node.lineno}:{node.name}")

    assert findings == []

def test_business_chunk_consumers_do_not_concat_unbounded_research_chunks() -> None:
    hot_paths = (Path("src/quant_research_platform/research/portfolios/runner.py"),)
    forbidden_patterns = (
        re.compile(r"pd\.concat\(\s*chunks\b"),
        re.compile(r"pd\.concat\(\s*frames\b"),
    )

    hits: list[str] = []
    for path in hot_paths:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in forbidden_patterns):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []

def test_prediction_output_builder_does_not_prederive_oos_end() -> None:
    text = Path(
        "src/quant_research_platform/research/predictions/outputs.py"
    ).read_text(encoding="utf-8")

    assert '"oos_end"' not in text

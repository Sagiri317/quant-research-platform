from __future__ import annotations

from pathlib import Path

from tests.helpers.maintenance_script_loader import load_maintenance_script_module

governance_contract = load_maintenance_script_module("governance_contract")


def _fixture_occurrence():
    return next(
        entry
        for entry in governance_contract.SOURCE_TREE_ALLOWED_OCCURRENCES
        if entry.rule_id.endswith(":sensitive_provider_marker")
    )


def _scan(repo: Path, *paths: Path) -> list[dict[str, object]]:
    scan_input = governance_contract.build_source_tree_scan_input(repo, paths)
    return governance_contract.source_tree_sensitive_negative_literal_findings(
        scan_input
    )


def test_source_tree_scan_groups_exposes_only_sensitive_boundary() -> None:
    groups = governance_contract.source_tree_scan_groups(Path("."))

    assert set(groups) == {"sensitive_negative_literals"}
    assert groups["sensitive_negative_literals"] == {
        "status": "PASS",
        "finding_count": 0,
        "findings": [],
    }


def test_sensitive_scan_rejects_unregistered_literal(tmp_path: Path) -> None:
    entry = _fixture_occurrence()
    relative = Path("docs") / "unexpected.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(entry.exact_line + "\n", encoding="utf-8")

    findings = _scan(tmp_path, relative)

    assert [(row["path"], row["rule"]) for row in findings] == [
        (relative.as_posix(), "sensitive_provider_marker")
    ]


def test_sensitive_scan_rejects_static_string_composition(tmp_path: Path) -> None:
    entry = _fixture_occurrence()
    match = governance_contract.source_tree_sensitive_negative_patterns()[
        "sensitive_provider_marker"
    ].search(entry.exact_line)
    assert match is not None
    value = match.group(0)
    relative = Path("tests") / "split_literal.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(
        f"value = {value[:4]!r} + {value[4:]!r}\n",
        encoding="utf-8",
    )

    findings = _scan(tmp_path, relative)

    assert findings == [
        {
            "path": relative.as_posix(),
            "line": 1,
            "rule": "sensitive_provider_marker",
            "reason": "static_string",
        }
    ]


def test_sensitive_fixture_allowance_is_exact_and_single_use(tmp_path: Path) -> None:
    entry = _fixture_occurrence()
    path = tmp_path / Path(entry.path)
    path.parent.mkdir(parents=True)
    path.write_text(entry.exact_line + "\n", encoding="utf-8")

    assert _scan(tmp_path, Path(entry.path)) == []

    path.write_text(entry.exact_line + "\n" + entry.exact_line + "\n", encoding="utf-8")
    findings = _scan(tmp_path, Path(entry.path))
    assert findings == [
        {
            "path": entry.path.as_posix(),
            "line": 2,
            "rule": "sensitive_provider_marker",
            "reason": "source_tree_allowed_occurrence_overmatched",
        }
    ]

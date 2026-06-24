from __future__ import annotations

from pathlib import Path

from tests.helpers.governance_fixtures import load_governance_contract

ROOTS_TO_SCAN = (
    Path("src"),
    Path(".agents/skills/quant-research-platform-maintenance/scripts"),
)
ACTIVE_DOC_ROOTS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path(".agents/briefs"),
    Path(".agents/skills/quant-research-platform-maintenance/SKILL.md"),
    Path(".agents/skills/quant-research-platform-maintenance/references"),
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/README.md"
    ),
    Path("docs/architecture"),
    Path("docs/code-map"),
    Path("docs/api"),
    Path("docs/data"),
    Path("data/README.md"),
)


def test_run_output_architecture_retired_paths_are_absent() -> None:
    contract = load_governance_contract()

    existing = [
        path
        for path in contract.retired_run_output_architecture_paths()
        if Path(path).exists()
    ]

    assert existing == []


def test_retired_run_output_names_are_absent() -> None:
    contract = load_governance_contract()
    patterns = contract.retired_run_output_architecture_name_patterns()
    hits: list[str] = []

    for path in _scan_files((*ROOTS_TO_SCAN, *ACTIVE_DOC_ROOTS)):
        if contract.is_run_output_architecture_negative_test(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            if pattern.search(text):
                hits.append(f"{path.as_posix()}:{name}")

    assert hits == []


def _scan_files(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower()
                in {".md", ".py", ".json", ".toml", ".yml", ".yaml"}
            )
    return tuple(sorted(set(files)))

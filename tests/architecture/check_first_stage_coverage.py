from __future__ import annotations

import json
import sys
from pathlib import Path

CRITICAL_FILES = (
    "src/quant_research_platform/catalog/profiles.py",
    "src/quant_research_platform/markets/implementations/a_share/processed/panels.py",
    "src/quant_research_platform/markets/implementations/a_share/processed/annotations.py",
    "src/quant_research_platform/markets/implementations/a_share/processed/materializer.py",
    "src/quant_research_platform/markets/implementations/a_share/factors/calculations.py",
    "src/quant_research_platform/markets/implementations/a_share/labels/calculations.py",
    "src/quant_research_platform/research/datasets/assembler.py",
)
THRESHOLD = 90.0


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: python tests/architecture/check_first_stage_coverage.py "
            "<coverage.json>"
        )
        return 2
    coverage_path = Path(sys.argv[1])
    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    failures: list[str] = []
    for critical_file in CRITICAL_FILES:
        entry = _coverage_entry(files, critical_file)
        if entry is None:
            failures.append(f"{critical_file}: missing")
            continue
        percent = float(entry["summary"]["percent_covered"])
        print(f"{critical_file}: {percent:.2f}%")
        if percent < THRESHOLD:
            failures.append(f"{critical_file}: {percent:.2f}% < {THRESHOLD:.2f}%")
    if failures:
        print("first_stage_coverage=FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("first_stage_coverage=PASS")
    return 0


def _coverage_entry(files: dict[str, object], target: str) -> dict[str, object] | None:
    normalized_target = target.replace("\\", "/")
    suffix = normalized_target.removeprefix("src/")
    for path, entry in files.items():
        normalized = path.replace("\\", "/")
        if normalized in {normalized_target, suffix} or normalized.endswith(
            "/" + suffix
        ):
            return entry  # type: ignore[return-value]
    return None


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build and smoke-check the package, then write a compact receipt."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

FORBIDDEN_SUFFIXES = {".parquet", ".csv", ".pkl", ".pickle", ".feather", ".arrow"}
SCANNED_ARTIFACT_KINDS = ("wheel", "sdist", "install_target")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--receipt-out", required=True)
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    receipt = run_package_smoke(repo)
    out = Path(args.receipt_out)
    if not out.is_absolute():
        out = repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


def run_package_smoke(repo: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="qrp-package-smoke-") as temp:
        root = Path(temp)
        dist = root / "dist"
        target = root / "install-target"
        build = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--outdir",
                str(dist),
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        wheel_path = first_file(dist, "*.whl")
        sdist_path = first_file(dist, "*.tar.gz")
        install_status = "NOT_RUN"
        import_status = "NOT_RUN"
        if build.returncode == 0 and wheel_path is not None:
            install = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    str(target),
                    str(wheel_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            install_status = "PASS" if install.returncode == 0 else "FAIL"
            if install.returncode == 0:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(target)
                probe = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import quant_research_platform; "
                            "print(quant_research_platform.__name__)"
                        ),
                    ],
                    cwd=repo,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                import_status = "PASS" if probe.returncode == 0 else "FAIL"
        forbidden = [
            *archive_content_scan(wheel_path, "wheel"),
            *archive_content_scan(sdist_path, "sdist"),
            *install_target_content_scan(target),
        ]
        build_status = "PASS" if build.returncode == 0 else "FAIL"
        content_status = "PASS" if not forbidden else "FAIL"
        status = aggregate_status(
            [
                build_status,
                install_status,
                import_status,
                content_status,
            ]
        )
        shutil.rmtree(target, ignore_errors=True)
        return {
            "schema_id": "qrp-package-smoke-receipt",
            "status": status,
            "build": {
                "status": build_status,
                "artifact_kinds": ["wheel", "sdist"],
            },
            "install": {
                "target_kind": "temporary_no_deps_install",
                "no_deps": True,
                "status": install_status,
            },
            "import": {
                "status": import_status,
                "module": "quant_research_platform",
            },
            "content_scan": {
                "status": content_status,
                "forbidden_member_count": len(forbidden),
                "scanned_artifact_kinds": list(SCANNED_ARTIFACT_KINDS),
            },
        }


def first_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def install_target_content_scan(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [
        f"install_target:{path.relative_to(root).as_posix()}"
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]


def archive_content_scan(path: Path | None, artifact_kind: str) -> list[str]:
    if path is None:
        return []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [
                f"{artifact_kind}:{name}"
                for name in archive.namelist()
                if Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
            ]
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return [
                f"{artifact_kind}:{member.name}"
                for member in archive.getmembers()
                if member.isfile()
                and Path(member.name).suffix.lower() in FORBIDDEN_SUFFIXES
            ]
    return []


def aggregate_status(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "NOT_RUN" in statuses:
        return "NOT_RUN"
    return "PASS"


if __name__ == "__main__":
    raise SystemExit(main())

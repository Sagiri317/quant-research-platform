#!/usr/bin/env python3
"""Build and smoke-check the package, then write a compact receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FORBIDDEN_SUFFIXES = {".parquet", ".csv", ".pkl", ".pickle", ".feather", ".arrow"}


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
        forbidden = content_scan(target if target.exists() else repo)
        content_status = "PASS" if not forbidden else "FAIL"
        status = aggregate_status(
            [
                "PASS" if build.returncode == 0 else "FAIL",
                install_status,
                import_status,
                content_status,
            ]
        )
        shutil.rmtree(target, ignore_errors=True)
        return {
            "schema_id": "qrp-package-smoke-receipt",
            "status": status,
            "wheel": file_payload(wheel_path),
            "sdist": file_payload(sdist_path),
            "install": {
                "target_kind": "temporary isolated target",
                "no_deps": True,
                "status": install_status,
            },
            "import": {"status": import_status},
            "content_scan": {
                "status": content_status,
                "forbidden_member_count": len(forbidden),
            },
        }


def first_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def file_payload(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"filename": "", "sha256": ""}
    return {"filename": path.name, "sha256": sha256_file(path)}


def content_scan(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [
        path.name
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

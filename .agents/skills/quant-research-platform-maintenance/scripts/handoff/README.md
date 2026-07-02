# Handoff Scripts

本目录保存 tracked handoff tooling。所有 handoff 构建和验证必须使用这里的脚本，不得从 `.run/` 中复制或推断工具逻辑。

## Files

```text
handoff_contract.py
receipt_contract.py
collect_handoff_receipts.py
build_handoff_package.py
verify_handoff_package.py
run_package_smoke.py
self_test.py
```

## Contract Owners

- `handoff_contract.py` owns package layout, sidecar shape, task metadata shape, validation summary rows, validation-summary evidence graph rules, allowed/required/forbidden zip entries, status summary aggregation, and exact package member sets.
- `receipt_contract.py` owns package-smoke, hard-scan, check-receipts, advisory-review receipt schemas and validators.
- `governance_contract.py` owns governance scans, source-tree scan rules, share-safety path rules, and status tokens.

## Tool Boundaries

- `collect_handoff_receipts.py` reads an input directory, optionally wraps check JSONL into `check-receipts.json`, copies required/optional receipts, and calls `receipt_contract.validate_receipt_file_set()`.
- `build_handoff_package.py` requires a clean current worktree and current branch HEAD matching task `final_head`, stages only the minimal handoff members, writes `artifact_manifest.json`, writes the external sidecar, and calls `verify_handoff_package.py` when requested.
- `verify_handoff_package.py` is the only package-level trust boundary. It verifies exact zip members, duplicate/path safety, external sidecar shape, manifest hashes, task metadata, validation summary, receipts through `receipt_contract.py`, `final-branch.bundle`, hard-scan replay, and share-safety scan.

Builder and collector do not implement package-level verification, receipt validators, share-safety regexes, or parallel status aggregation.

`handoff_evidence/validation-summary.json` is a derived view. Every `PASS`, `FAIL`, or `BLOCKED` row must be backed by a receipt graph edge in `check-receipts.json`, `package-smoke-receipt.json`, or `hard-scan-receipt.json`.

## Bundle Verification

`final-branch.bundle` must be directly verifiable:

```text
git bundle verify final-branch.bundle
git clone --branch <branch-from-sidecar> final-branch.bundle review-repo
```

The verifier clones the bundle, checks HEAD against sidecar `final_head`, checks a clean working tree, and replays hard scan from the clone. No separate reconstruction evidence file is produced or trusted.

## Output

handoff output should be written to `.run/` or a user-specified directory. `.run/` is output/evidence storage, not a tooling fact source. Reports must keep `PASS`, `FAIL`, `NOT_RUN`, and `BLOCKED` scoped to the actual command or artifact.

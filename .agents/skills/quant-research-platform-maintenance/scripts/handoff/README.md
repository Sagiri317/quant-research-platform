# QRP Handoff Tools

Tracked source for building and checking public-safe QRP handoff packages.
Generated packages do not copy these tracked files into the zip. The final zip
contains the branch bundle plus task-specific evidence; reviewers can clone the
bundle and run the tracked tools from that clone.

## Task Input

The builder takes one public-safe task JSON:

```powershell
<PYTHON_EXE> .agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py --task <task_metadata_input.json> --out-dir .run/handoffs --verify
```

The task JSON must provide `slug`, `branch`, `base`, and `final_head` plus
public-safe `validation_evidence` and optional `handoff_evidence`. In
`evidence` mode, the zip normalizes this input into:

- `task_metadata.json`
- `validation_evidence.json`
- `handoff_evidence/*.json`

`pr_closure` mode may additionally include `review_threads.json` with
public-safe PR closure metadata. Narrative PR text is outside the handoff
package contract.

Do not include private row-level data, local absolute paths, raw tracebacks,
vendor filenames, credentials, tokens, or exact private runtime payloads.

## Outputs

The builder writes exactly:

- `<slug>.zip`
- `<slug>.json`

The external JSON sidecar is not placed in the zip. It contains only the zip
filename, logical zip path, zip hash, artifact manifest hash, branch/base/head,
generation timestamp, and minimal status summary.

## Verification

From an existing checkout:

```powershell
<PYTHON_EXE> .agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py
<PYTHON_EXE> .agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py --zip .run/handoffs/<slug>.zip --json .run/handoffs/<slug>.json
<PYTHON_EXE> .agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_safety.py .run/handoffs/<slug>.zip --json .run/handoffs/<slug>.json
```

In a fresh directory with only the zip/json, first extract or copy
`final-branch.bundle` from the zip, then clone the bundle and run the tracked
tools from the cloned repository:

```bash
git clone --branch <branch-from-sidecar> final-branch.bundle repo
python repo/.agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py
python repo/.agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py --zip <slug>.zip --json <slug>.json
python repo/.agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_safety.py <slug>.zip --json <slug>.json
```

The legacy `factor_chunk_diagnostic.py`, `n2_artifact_validation.py`, and
`smoke_compare.py` files are not handoff contract requirements.

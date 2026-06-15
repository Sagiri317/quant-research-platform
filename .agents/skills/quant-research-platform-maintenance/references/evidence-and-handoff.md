# Evidence and Handoff Rules

This reference controls public-safe evidence, review bundles, and final status
claims.

## Truth Statuses

- `PASS`: command/check ran and passed. Include command or manifest evidence.
- `FAIL`: command/check ran and failed. Do not relabel as PASS without rerun
  evidence.
- `NOT_RUN`: command/check did not run. Keep as `NOT_RUN`.
- `NOT_APPLICABLE`: check does not apply to this task; explain why.

## Public-Safe Package Rules

Allowed:

- implementation summary;
- decisions and open questions;
- public-safe aggregate evidence summaries;
- smoke/compare/performance/test summaries;
- manifest and safety reports;
- git bundle when explicitly requested for branch reconstruction.

Blocked:

- raw/interim/processed private data;
- private report rows;
- parquet/csv/pkl/pickle/feather/arrow/raw vendor files;
- credentials, secrets, cookies, tokens, `.env` files;
- local absolute paths and stock-code samples;
- old handoff artifacts unless explicitly requested as evidence inputs.

## Handoff Workflow

1. Confirm task mode and branch.
2. Collect only public-safe evidence.
3. Preserve `PASS`/`FAIL`/`NOT_RUN` as recorded.
4. Run package safety scan if a package is generated.
5. Include manifest with source commit, branch, file list, and check statuses.
6. Final response must list checks run and checks not run.

## `.run` Boundary

The local run workspace is a workflow/evidence layer, not a source of
architecture. Use its official tools when available, but do not let handoff
layout drive public architecture or repo governance layout.

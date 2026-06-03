# Code Map

This directory maps the current QRP Clean Rewrite `0.1.0` implementation. It is
for code reading and onboarding, not roadmap, status, or governance.

## Files

| File | Purpose |
|---|---|
| `platform-flow.md` | Global source, API, CLI, and package ownership map |
| `a-share-data-and-pipeline.md` | A-share CSMAR data flow and migration targets |
| `reports-api.md` | Evaluation/prediction report ownership and API/CLI handoff |
| `source-csmar-contracts.md` | CSMAR source contract boundary |
| `compute-portfolio-boundary.md` | Portfolio compute split boundary and 0.1.0 limitation |

## Reading Order

1. `platform-flow.md`
2. `a-share-data-and-pipeline.md`
3. `reports-api.md`
4. `source-csmar-contracts.md`
5. `compute-portfolio-boundary.md`

## Maintenance

- Keep maps aligned with code and tests, not with old branch history.
- Do not include private absolute paths, row samples, stock-code samples, run
  IDs, commit SHAs, or debug output.
- When the 0.1 API/CLI or processed-data layout changes, update this directory
  in the same slice.

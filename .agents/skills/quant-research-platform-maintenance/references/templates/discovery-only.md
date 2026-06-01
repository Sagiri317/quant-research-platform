# Discovery Only

## Task Mode

`discovery-only`

## Recommended Reasoning Effort

Low

## Context Budget

- Start with `rg` over the requested terms, paths, or symbols.
- Read line ranges before full files.
- Do not load architecture, archive, generated output, private data, or broad code maps unless the request names them.
- Report candidate context expansion before reading beyond the requested surface.

## Allowed Scope

- Inspect:
- Summarize:
- No file edits unless the user explicitly changes the task mode.

## Forbidden Scope

- Do not edit files.
- Do not run tests or scripts unless the user asks for diagnostics.
- Do not read secrets, private row-level data, or generated private artifacts.
- Do not change branch, commit, push, merge, delete, install, or configure.

## Stop Conditions

- Stop if the next step would require edits, private data, secrets, dependency changes, deletion, or broad context expansion.
- Stop if contract, schema, `manifest.display`, API/frontend, deployment/trading, or phase-boundary meaning is unclear.

## Checks

- `git status --short --branch` only when repository state matters.
- `rg "<term>" <path>` for requested terms.

## Output Format

- Findings
- Files Read
- Not Read
- Stop Conditions Hit
- Recommended Next Step

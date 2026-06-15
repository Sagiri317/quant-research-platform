# Script Portability

Governance scripts should run on Windows 11, macOS, Linux, and WSL when Python is
available.

## Rules

- Prefer Python standard library.
- Use `pathlib`, `argparse`, `json`, `zipfile`, `hashlib`, and `subprocess` when
  needed.
- Avoid ps1-only or bash-only entrypoints.
- Do not hardcode drive letters, home directories, backslashes, or local user
  paths.
- Do not scan private roots by default.
- Provide `--help` and explicit exit codes.
- Default to check-only behavior. Dry-run mode is required when a script would
  otherwise mutate files.
- Print concise machine-readable summaries when useful; do not write outputs
  unless the task explicitly approves output paths.

## Exit Codes

- `0`: all required checks pass.
- `1`: blocking violation.
- `2`: script usage/configuration error.

## Validation

Run scripts with the project Python in the active environment. On the user's
machine this is Python `3.13.x`; scripts should avoid unnecessary version-specific
syntax so they stay portable across current agent environments.

# Governance Verification Checklist

Use this after applying governance changes.

## Required Checks

- `git status --short --branch`
- `git diff --name-only`
- `git diff --check`
- `python .agents/skills/quant-research-platform-maintenance/scripts/verify_governance.py --repo-root . --governance-only`
- `python .agents/skills/quant-research-platform-maintenance/scripts/verify_code_map_flow.py --repo-root .`
- `python .agents/skills/quant-research-platform-maintenance/scripts/package_safety_check.py --help`
- targeted architecture tests relevant to governance/docs if dependencies are
  installed.

## Review Questions

- Is `AGENTS.md` still short and hard-boundary focused?
- Are `.agents/` skills and references agent-neutral?
- Are global `.codex/` and repo `.agents/` clearly separated?
- Was repo-local `.codex/` avoided?
- Are skill triggers clear enough for progressive disclosure?
- Are task modes and context routing unambiguous?
- Are PASS/FAIL/NOT_RUN claims backed by evidence?
- Are private roots and blocked package extensions excluded from packages?
- Did the diff avoid business code and business tests?
- Did the diff avoid new public API/CLI/registry/store/facade surface?
- Are scripts Python-first and portable across Win11/macOS/Linux?
- Are stale N2/branch/next-step claims removed or updated?
- Are hooks/MCP/subagents draft-only unless explicitly approved?

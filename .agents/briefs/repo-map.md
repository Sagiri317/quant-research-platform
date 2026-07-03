# Repo Map

本文件只给 agent 提供仓库导航，不记录当前状态。

## 主要路径

```text
AGENTS.md
README.md
.agents/briefs/user-principles.md
.agents/skills/quant-research-platform-maintenance/SKILL.md
.agents/skills/quant-research-platform-maintenance/references/
.agents/skills/quant-research-platform-maintenance/scripts/
docs/architecture/
docs/code-map/
docs/api/
docs/data/
src/quant_research_platform/
tests/
```

## 源码分层

```text
policy/          share safety、identifier safety、check status aggregation
contracts/       structure contracts, enums, IDs, TimeWindow, basic validation
catalog/         generic registry class
profiles/        API/CLI/agent/workflow composition root
sources/         provider/source ingestion, raw schema, field mapping, raw/interim transform
markets/         generic market domain and session windows
markets/implementations/a_share/  A-share calendar, processed data, masks, research binding
research/        generic research result builders, no file writes
compute/         dataframe/tensor numeric kernels, portfolio math, performance
reports/         build/render evidence and report payload/text
artifacts/       write LOCAL_ONLY outputs and REVIEW_SAFE evidence files
workflows/       orchestrate research -> reports -> artifacts
storage/         dataset IO, manifest, fingerprint, compare, data layout
runtime/         resources, logging, workflow records, execution metrics payload
interfaces/      approved API, CLI, service adapters
agents/          agent-assisted research helpers
```

## 运行结果布局

```text
<run_root>/run.json
<run_root>/data/{interim,processed,research}/
<run_root>/outputs/<output_id>/manifest.json
<run_root>/logs/
<run_root>/summary.md
```

## 文档分层

```text
docs/architecture/   current architecture contracts, governance contracts, open issues
docs/code-map/       code flow maps and A-share data flow
docs/api/            API boundary
docs/data/           data directory contract
```

## agent skill 分层

```text
SKILL.md
references/current-contract.md
references/task-routing.md
references/verification-and-gates.md
references/handoff-and-evidence.md
references/tooling-and-portability.md
references/prompt-patterns.md
scripts/verify_governance.py
scripts/verify_code_map_flow.py
scripts/handoff/
schemas/handoff/
templates/handoff/
```

## 读取规则

先用 live git baseline 确认现场，再按任务选择最小文件集合。`.run/` 产物不定义当前合同。handoff 合同由 tracked `.agents` tooling、verifier 和 tests 定义。

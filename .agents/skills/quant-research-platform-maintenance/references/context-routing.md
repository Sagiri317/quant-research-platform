# Context Routing / 上下文路由

目标：少读、读准、读到 owner layer。不要为了“全面”每次加载全 repo、全 docs、全 templates 或 `.run` 原始输出。

## 通用规则

1. 先读 `AGENTS.md` 和用户任务 envelope。
2. 先 `rg` 定位，再读取窄片段。
3. 不默认读取 private data、generated outputs、old handoff zip/json、完整 `.run`、完整历史会话日志。
4. Expanded Prompt Mode（用户给了完整任务书）下，不重新读取全部模板来重建流程；只读取 named/relevant files。
5. Repo Router Mode（短 prompt）下，先读 `task-modes.md` 和 `templates/auto-task-router.md`。
6. 如果需要扩大上下文到 public API/schema/artifact/private data/dependency/跨层 owner，先触发 stop condition。

## 路由表

| 任务/表面 | 首选上下文 |
|---|---|
| 当前项目事实 | `.agents/briefs/current-qrp-status.md` |
| 用户长期偏好/架构原则 | `.agents/briefs/user-principles.md` |
| 任务分类 | `references/task-modes.md`, `references/templates/auto-task-router.md` |
| 强制 gates | `references/governance-gates.md` |
| owner layer 判定/P2/P3 路由 | `references/owner-layer-routing.md`, `docs/architecture/open-issues-register.md` |
| public API/CLI | `docs/api/README.md`, `src/quant_research_platform/interfaces/`, `tests/interfaces/` |
| contracts/schema | `src/quant_research_platform/contracts/`, `tests/contracts/`, `tests/architecture/` |
| storage/dataset IO | `src/quant_research_platform/storage/`, relevant dataset tests |
| source/interim data | `src/quant_research_platform/sources/`, `docs/data/README.md`, relevant source tests |
| A-share semantics | `src/quant_research_platform/markets/implementations/a_share/`, `docs/code-map/a-share-data-and-pipeline.md`, relevant market tests |
| research datasets/evaluation/prediction | `src/quant_research_platform/research/`, `src/quant_research_platform/reports/`, relevant tests |
| workflow | `src/quant_research_platform/workflows/`, `tests/workflows/` |
| proposal agents/factor factory | `src/quant_research_platform/agents/`, relevant tests; no external LLM API |
| docs/status sync | changed files + active docs map; avoid archive/full-history reads |
| governance changes | `AGENTS.md`, `.agents/**`, governance docs, input evidence summaries |
| handoff packaging | `.run` official local tool docs/scripts if present; package rules; safety script |
| apply-only patch | patch/zip + changed-file list + validation checklist; no redesign |

## Context budget hints

- Small fix：通常 <5 files。
- Standard implementation：owner layer files + direct tests/docs，通常 <15 files。
- Goal closure：按 slice 读取；每个 slice 只加载当前 owner surface。
- Governance redesign：读取 governance layer + evidence summaries；业务代码只读作 source-of-truth，不做 broad implementation scan。

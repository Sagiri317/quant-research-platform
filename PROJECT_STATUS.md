# Quant Research Platform 项目状态

本文记录当前 operational truth（运行性事实）和需要同步的 handoff 信号。历史 closeout、旧分支叙事、私有 run 细节、一次性调试证据不写入本文。P0/P1/P2/P3 follow-ups 以 `docs/architecture/open-issues-register.md` 为准。

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 当前阶段 | QRP Clean Rewrite `0.1.0` on Local Research Platform |
| 当前开发基线 | `feature/qrp-clean-rewrite` |
| 版本边界 | project `0.1.0`; API/contracts use `0.1` |
| 源码拓扑 | `contracts`, `storage`, `catalog`, `sources`, `markets`, `research`, `workflows`, `compute`, `agents`, `interfaces`, `reports`, `runtime` |
| 当前 API/CLI | `/api/0.1/...` factor/label catalog routes and research dataset requirements; `qrp catalog factors`, `qrp catalog labels`, `qrp research-datasets requirements` |
| 执行边界 | API/CLI is read-only for current public users; source/interim/processed materialization and private real-data execution are internal |
| 市场范围 | A-share enabled; other markets are future design only, with no empty implementation packages |
| 私有边界 | `data/raw/<internal-provider>`, `data/interim/<internal-provider>`, `data/processed/a_share`, `data/research/a_share`, `reports/private_real` |
| 交易边界 | No investment advice, paper/live trading, broker/exchange connector, deployment, scheduler, worker, or external data API |
| 治理边界 | `.agents/` is the repo governance root; global `.codex/` is read-only evidence only; repo-local `.codex/` is not created by default |

## 当前有效样例

Current clean-rewrite verification is code/test based. Private generated artifacts remain local evidence only. Previous public-safe aggregate parity and coverage evidence in `docs/reports/` is superseded by later refactors until the next private rebuild regenerates public-safe evidence.

Formal verification uses Python `>=3.13,<3.14` with conda `quantlab313`. The primary verified OS is Windows 11. macOS/Linux are cross-platform compatible targets, but PASS must only be claimed after tests run there.

Current public-safe anchors:

- 0.1 source topology tests.
- 0.1 contracts/storage/catalog/source tests.
- 0.1 A-share processed, ResearchDataset, evaluation, model, and prediction tests.
- A-share daily price-volume factors and `forward_return` labels use per-asset valid daily bars, not calendar sessions.
- 0.1 API/CLI route and command contract tests.
- Report writers derive paths through `ReportSpec`, `ReportCatalog`, `StorageLayout`, and `run_id`.
- Dataset instance metadata is writer-owned at `write_dataset()`; research dataset assembly persists `input_fingerprints` through storage rather than mutating dataset manifests after write.
- Evaluation outputs include typed `EvaluationDiagnostics` and persisted `ic_summary.parquet` / `group_spread_summary.parquet` summary artifacts. IC and RankIC summaries use explicit valid counts and keep no-valid statistics as `NaN`.
- Prediction reports require fixed-shape `model_metadata` and `prediction_metadata`, including effective train/test window values.
- Evaluation, model fitting, and prediction handoff read research datasets through chunked storage IO and use shared runtime resource policy when memory budgets require it.

## 当前 blocker / 风险

1. Public API/CLI must remain minimal until a specific task approves expansion.
2. Future-only items remain out of active public contract: `processed-data-runs`, `workflow-runs`, `report index/show`, `RunRepository`, `ReportRepository`, `private plugin`, model registry, feature store, universal artifact lifecycle API.
3. Full public-safe evidence/report regeneration remains pending until a dedicated private rebuild/evidence refresh task.
4. Governance scripts should be validated on the user's Win11 + Python 3.13.14 environment after applying governance redesign.

## 下一步

1. Apply governance-only changes on a feature branch created from `feature/qrp-clean-rewrite`, for example `feature/agent-governance-redesign`.
2. Validate with governance scripts and `git diff --check`.
3. Keep any check that was not actually executed as `NOT_RUN` or `BLOCKED`, not `PASS`.
4. Do not merge to `main` unless a separate merge gate task explicitly approves it.

## Root Handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-06-03 | status | QRP clean rewrite started on `feature/qrp-clean-rewrite`: source topology moved toward 0.1.0, API/CLI public surface changed to `/api/0.1/...`, private data boundaries changed to `data/raw`, `data/interim`, `data/processed`, and `reports/private_real`; old endpoint/command compatibility is no longer target-contract truth. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | QRP clean rewrite completed through structural local phases; private generated artifacts remain local evidence only. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-15 | governance | Agent governance redesign defines `.agents/` as project governance root, global `.codex/` as read-only evidence, and repo-local `.codex`/hooks/MCP/plugins as draft-only unless separately approved. | WORKSPACE_STATUS, PLAN_INDEX | open |

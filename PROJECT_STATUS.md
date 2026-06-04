# Quant Research Platform 项目状态

本文只记录当前 operational truth、当前 blocker / next steps、以及需要同步到
workspace root 的 handoff 信号。历史 closeout、旧分支叙事、私有 run 细节和
一次性调试证据不写入本文。

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 当前阶段 | QRP Clean Rewrite `0.1.0` on Local Research Platform |
| 当前分支 | `feature/qrp-clean-rewrite` |
| Conda 环境 | `analysis` |
| 版本边界 | project `0.1.0`; API/contracts/processed-data/research-dataset/workflow specs use `0.1` |
| 源码拓扑 | `contracts`, `storage`, `catalog`, `sources`, `markets`, `research`, `workflows`, `compute`, `agents`, `interfaces`, `reports`, `runtime` |
| 当前 API/CLI | `/api/0.1/...` and clean `qrp catalog/data-product/market/research/workflow/report` commands |
| 执行边界 | API/CLI `execution_mode` defaults to `execute`; `--dry-run` maps to `dry_run`; private real-data execution is not exposed through API/CLI |
| 市场范围 | A-share enabled; other markets are future design only, with no empty implementation packages |
| 私有边界 | `data/raw/<internal-provider>`, `data/interim/<internal-provider>`, `data/processed/a_share`, `reports/private_real` |
| 交易边界 | No investment advice, paper/live trading, broker/exchange connector, deployment, scheduler, worker, or external data API |

## 当前有效样例

Current clean-rewrite verification is code/test based. Private generated
artifacts remain local evidence only. Public-safe aggregate parity and migration
evidence is committed in `docs/reports/clean-rewrite-0.1-parity-evidence.md`.

Current public-safe anchors:

- 0.1 source topology tests.
- 0.1 contracts/storage/catalog/source tests.
- 0.1 A-share processed, ResearchDataset, evaluation, and prediction tests.
- 0.1 API/CLI route and command contract tests.
- Public-safe parity-or-explained-delta evidence committed for processed panel,
  factor values, labels, masks/universes, ResearchDataset, evaluation, and
  prediction smoke.
- P2 compute portfolio and internal provider contract split modules exist with
  compatibility facades.
- Report writers derive paths through `ReportSpec`, `ReportCatalog`,
  `StorageLayout`, and `run_id`.
- 2014-2025 historical migration completed locally for the new processed
  factor/label-supporting layout and ResearchDataset layout.

## 当前 blocker / 风险

1. Retired API/CLI compatibility tests are no longer authoritative after the
   clean public surface removed retired endpoints and commands.
2. Full historical `daily_panel` was not migrated from the old reference; the
   clean rewrite keeps the 2014 panel generated and verified by new code, while
   full 2014-2025 factor/label-supporting layers and ResearchDataset were
   migrated through spec-driven writers.
3. This branch is not merged to `main`.

## 下一步

1. Run final Phase 11 gates and review the staged diff.
2. Commit and push `feature/qrp-clean-rewrite`.
3. Do not merge `main` from this task.

## Root Handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-06-03 | status | QRP clean rewrite started on `feature/qrp-clean-rewrite`: source topology moved toward 0.1.0, API/CLI public surface changed to `/api/0.1/...`, private data boundaries changed to `data/raw`, `data/interim`, `data/processed`, and `reports/private_real`; old endpoint/command compatibility is no longer target-contract truth. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | QRP clean rewrite completed through Phase 10 locally: 2014 parity-or-explained-delta passed, 2014-2025 factor/label-supporting processed layers and ResearchDataset were migrated through spec-driven writers, and reports remain under `reports/private_real`. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | Pre-merge fix added public-safe clean rewrite parity evidence, thin API/CLI dry-run validation services, sanitizer/report/writer gates, and accepted explained-delta wording for full historical `daily_panel`. | WORKSPACE_STATUS, PLAN_INDEX | open |
| 2026-06-04 | status | Pre-merge blocker repair updated `execution_mode`, registered workflow execution, ReportSpec/ReportCatalog writer gates, P2 split modules, and public-safe aggregate evidence hash validation. | WORKSPACE_STATUS, PLAN_INDEX | open |

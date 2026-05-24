# Phase 1 Acceptance Record / Phase 1 验收记录

本文记录 `quant-research-platform` Phase 1 已完成事实。它替代旧的 Phase 1 设计文档和执行清单，作为历史验收记录保存，不再作为下一阶段实施计划。

## 验收结论

截至 2026-05-23，Phase 1 清单内代码功能已完成复核；截至 2026-05-24，`docs/code-map/` 和项目 durable docs 已按当前实现同步。

Phase 1 的目标是把单一 A 股研究项目收口为本地多市场量化研究平台的第一版结构，同时保护已跑通的 A 股主链路、artifact schema `0.5`、本地 API、下载和 React evidence workbench。

## 已落地结构

当前已落地结构：

```text
src/quant_research_platform/
  evaluations/                         EvaluationSpec、payload、runner、result、panel validation
  kernels/                             DataFrame / tensor evaluation kernels
  artifacts/                           schema 0.5 writer、loader、display、inventory、comparison
  datasets/                            generic CSV / Parquet readers
  interfaces/
    api/                               local FastAPI evidence API、downloads、pipeline、Factor Factory
    cli/                               qrp command entry
  markets/
    implementations/
      a_share/                         CSMAR-backed A-share implementation
  pipelines/
    a_share/                           snapshots、step registry、A 股 workflow helpers
  research/
    factor_factory/                    template registry、candidate batch runner、reports
  runtime/                             paths、profiling、metadata、exceptions
```

Phase 1 明确不创建 repo-level `apps/api`，不保留空 `markets/crypto` implementation，不提前创建空 Future 目录。

## 冻结合同

Phase 1 验收时保留以下合同：

| 合同 | 当前事实 |
|---|---|
| Python public API | `EvaluationSpec`, `WeightConfig`, `CostConfig`, `run_evaluation`, `make_parquet_evaluation_spec` |
| Evaluation runner | `run_evaluation(spec)` 返回 `EvaluationSetResult`；`evaluate_to_payload(spec)` 是纯计算边界 |
| Artifact schema | `artifact_schema_version == "0.5"`；`artifact_type == "evaluation_set"` |
| Display contract | `manifest.display` 是 API、下载、CSV preview、charts、frontend tables 的展示来源 |
| API run identity | API-visible run identity 是 `(source_alias, run_id)` |
| Download contract | direct download 和 ZIP download 只暴露 `manifest.display.downloads.items` |
| A 股 adapter | 私有 A 股 evaluation 使用 `markets.implementations.a_share.evaluation.run_factor_evaluation` |
| Pipeline workflow | Phase 1 保留配置 snapshot 和 step registry 边界；N1 已在此边界上实现 `start_from` executable runner 和 snapshot/run split，并已合入 `main` |
| Factor Factory | 第一版使用受控 YAML template registry 和已有 processed factor columns |

明确不恢复：

- `artifacts.xlsx`。
- 旧 `total_gross_portfolio_return` / `total_net_portfolio_return` summary 字段。
- unsupported schema 的宽松 fallback。
- 旧顶层 API 包、旧 `io` 包或旧 A 股项目目录名。

## P0-P5 完成记录

| 阶段 | 原目标 | 当前状态 |
|---|---|---|
| P0 合同冻结和文档基线 | 冻结 API/artifact/frontend 合同，写清 Phase 1 非目标 | 已完成；当前入口在 root docs、`docs/code-map/` 和本记录 |
| P1 Evaluation payload boundary | 拆出 `EvaluationRunPayload`，保留 public `run_evaluation` | 已完成 |
| P2 Artifact module split | schema、display validation、writer、loader、registry 拆分 | 已完成 |
| P3 API router/service split | `interfaces/api` route/service 分层，保留 route path | 已完成 |
| P4 A 股 pipeline boundary | 配置 snapshot、step registry 和 A 股 workflow 边界 | 已完成；Phase 1 后 N1 已实现 `start_from` runner 和 snapshot/run API split，并已合入 `main` |
| P5 Factor Factory 第一版 | registry、参数网格、批量评价、failed log、workbench 浏览 | 已完成第一版 |

## 验收证据

Phase 1 验收中已覆盖：

- 旧路径残留扫描。
- Phase 1 相关后端测试。
- 完整 `pytest`。
- `ruff check src tests`。
- 前端 `npm test -- --run --maxWorkers=1`。
- 前端 `npm run lint`。
- 前端 `npm run build`。
- CLI help。
- A 股 adapter help。
- API app import。
- architecture Markdown 链接检查。
- `git diff --check`。
- Factor Factory workbench。
- 2024 acceptance old/new 比较。
- Factor Factory smoke。

2026-05-24 文档同步中又覆盖：

- `docs/code-map/` 旧文件名和旧模块名扫描。
- code-map Markdown 链接检查。
- `platform-flow.md` 更新为 simplified overview。
- `platform-flow-interactive.html` 更新为 detailed map。
- API/contract 轻量导入和路由检查。

## 已知剩余架构债

这些不是 Phase 1 未完成项，而是 Phase 1 后续工作：

1. Eligibility / tradability 可见诊断：把覆盖数、过滤影响和不可交易原因写入 artifact。
2. Artifact ZIP packaging 下沉：把 API service 中的 ZIP 选择和打包规则抽到 `artifacts/`。
3. 前端 feature split：把当前平铺 `web/src` 拆成 Evidence、Pipeline、Factor Factory、Uploads、shared。
4. 风险、regime、成本、容量扩展：补正式研究诊断，不写成投资结论。
5. ML prediction handoff：prediction Parquet、model manifest、walk-forward 和 evaluation 接口。

后续任务见 [next-stage-roadmap.md](next-stage-roadmap.md)。

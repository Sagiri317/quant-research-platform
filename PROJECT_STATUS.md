# Quant Research Platform 项目状态

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 项目定位 | 本地多市场量化研究平台；A 股是第一版完整 market module |
| 当前分支 | `codex/a-share-build-runner` |
| Conda 环境 | `analysis` |
| 代码结构 | 主包为 `src/quant_research_platform/`；当前核心目录包括 `interfaces/api`、`interfaces/cli`、`datasets`、`markets/implementations/a_share`、`pipelines/a_share`、`research/factor_factory` |
| 数据范围 | 本地私有 CSMAR 2014-2025 raw/interim/processed 历史产物仍遵守私有边界 |
| 平台合同 | evaluation contract、artifact schema `0.5`、`manifest.display`、FastAPI evidence API、下载和 React evidence workbench 已迁入平台命名空间 |
| 前端状态 | 本地 workbench 已中文化；Factor Studio 可把表达式/模板传给 Pipeline Builder；Pipeline Builder 默认只调用 `POST /api/pipeline-snapshots` 写配置快照 |
| A 股 Pipeline | 执行入口为 `POST /api/pipeline-runs`；公开参数是 `start_from`，支持 `existing_factors`、`rebuild_factors`、`rebuild_panel`、`rebuild_interim`，并固定运行到 evaluation |
| Factor Factory | 已有 registry、批量评估、artifact `0.1`、API 和 workbench 浏览入口；候选评估通过 existing-factors start layer 调用统一 runner |
| 市场范围 | A 股 enabled；crypto / other markets planned / disabled，不代表真实支持，也不保留空实现包 |
| 私有边界 | CSMAR raw/interim/processed 和 `reports/private_real/` 不提交、不公开 |
| 交易边界 | 不做投资建议、paper trading、live trading、券商/交易所/API 连接或部署服务 |

## 已完成的当前阶段事实

- Phase 1 平台迁移、artifact/API/frontend 合同、Factor Factory 浏览入口和 P4/P5 小样本示例已完成。
- A 股专用 evaluation adapter 是真实私有 A 股 runs 的主入口；通用 `evaluate-parquet` 只用于外部/local Parquet adapter check。
- `codex/a-share-build-runner` 分支已把 A 股 pipeline 外部接口重构为 snapshot/run split 和 `start_from` 模型。
- 旧外部执行模式和双 runner 命名已从源码入口移除；旧字段只在负向测试中作为拒绝用例保留。

## 下一步

1. 完成 `codex/a-share-build-runner` 分支的全量验证和 review。
2. 继续做 N2：eligibility / tradability 诊断可视化。
3. 继续扩展前保留当前 P4/P5 smoke 命令作为回归门槛，并为新增能力写明确验收口径。
4. 后续能力如 ML prediction pipeline、风险/中性化扩展、成本/容量/组合归因需要单独设计。

## Root Handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-05-21 | planning | 项目定位从单一 A 股因子研究项目收口为 A 股量化研究平台入口；临时平台蓝图记录数据、因子、ML prediction、风险、regime、evaluation、evidence workbench 和未来模拟盘边界。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-22 |
| 2026-05-22 | migration | 项目身份迁移为 `quant-research-platform`；A 股为第一版 enabled market module；evaluation、schema `0.5` artifacts、API/download 和 React evidence workbench 成为平台内置能力。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-22 |
| 2026-05-23 | phase1 | Phase 1 目标结构已执行并完成复验；后续架构债另列。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-24 |
| 2026-05-24 | docs sync | 已重构 `docs/code-map/`，同步根文档、architecture 文档和状态文件的当前架构表述。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-24 |
| 2026-05-24 | next-stage implementation | `codex/a-share-build-runner` 分支重构 A 股 pipeline 外部接口：新增 `POST /api/pipeline-snapshots`，`POST /api/pipeline-runs` 统一为 `start_from`，统一 runner 编排 build/evaluate 到 evaluation，Factor Factory 改为 existing-factors start layer。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |

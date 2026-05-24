# Quant Research Platform 项目状态

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 项目定位 | 本地多市场量化研究平台；A 股是第一版完整 market module |
| 当前分支 | `main` |
| Conda 环境 | `analysis` |
| 代码结构 | 主包为 `src/quant_research_platform/`；当前核心目录包括 `interfaces/api`、`interfaces/cli`、`datasets`、`markets/implementations/a_share`、`pipelines/a_share`、`research/factor_factory` |
| 数据范围 | 本地私有 CSMAR 2014-2025 raw/interim/processed 历史产物仍遵守私有边界 |
| 平台合同 | evaluation contract、artifact schema `0.5`、`manifest.display`、FastAPI evidence API、下载和 React evidence workbench 已迁入平台命名空间 |
| 前端状态 | 本地 workbench 已中文化；Factor Studio 可把表达式/模板传给 Pipeline Builder；Pipeline Builder 默认只调用 `POST /api/pipeline-snapshots` 写配置快照 |
| A 股 Pipeline | 执行入口为 `POST /api/pipeline-runs`；公开参数是 `start_from`，支持 `existing_factors`、`rebuild_factors`、`rebuild_panel`、`rebuild_interim`，并固定运行到 evaluation |
| Factor Factory | 已有 registry、批量评估、artifact `0.1`、API 和 workbench 浏览入口；候选评估通过 existing-factors start layer 调用统一 runner；N7 后应适配 evaluation run 合同，而不是约束 N7 核心设计 |
| 市场范围 | A 股 enabled；crypto / other markets planned / disabled，不代表真实支持，也不保留空实现包 |
| 私有边界 | CSMAR raw/interim/processed 和 `reports/private_real/` 不提交、不公开 |
| 交易边界 | 不做投资建议、paper trading、live trading、券商/交易所/API 连接或部署服务 |

## 已完成的当前阶段事实

本轮目标是把项目身份从单一 A 股研究项目切换为 `quant-research-platform`。最终功能和架构目标已收敛在 `docs/architecture/target-architecture-and-capabilities.md`；历史迁移计划、历史迁移设计和旧 Phase 1 任务拆解已由当前验收记录替代，不再作为当前待办清单。

当前阶段已完成可提交的迁移、核心合同闭环、Factor Factory 前端浏览入口和 P4/P5 小样本示例。2026-05-23 已完成 Phase 1 代码功能复核；2026-05-24 已同步 `docs/code-map/`、`docs/architecture/` 和项目根目录文档的当前架构表述。验收结论见 `docs/architecture/phase-1-acceptance-record.md`；下一阶段任务见 `docs/architecture/next-stage-roadmap.md`。Phase 1 功能已实现并通过测试：

- 用户入口文档改为 Chinese-first 的平台说明。
- `ARCHITECTURE.md` 和 `EVALUATION_INTERFACE.md` 记录平台架构、evaluation contract、artifact schema `0.5`、`manifest.display`、API/download/frontend 边界。
- `docs/architecture/target-architecture-and-capabilities.md` 合并最终目录结构与未来能力池，保留 A 股、ML prediction、风险、regime、workbench、未来模拟盘和商业化平台边界，但不把未来能力写成当前已实现范围。
- `docs/architecture/phase-1-acceptance-record.md` 记录 Phase 1 已实现范围、冻结合同、测试证据和后续架构债。
- `docs/architecture/next-stage-roadmap.md` 把 Phase 1 后的架构债提上日程，按任务列出目标、非目标、验收口径和测试边界。
- 项目级 `AGENTS.md` 增加 `src/quant_research_platform/` 结构、Conda `analysis`、私有数据、无外部 API/凭据、无交易、exact-path deletion、frontend/API/artifact contract、旧 evaluation reference 仅作参考、`source_alias + run_id` 运行身份边界。
- `.codex/skills/quant-research-platform-maintenance/` 记录后续维护 artifact/API/download/frontend 合同时必须执行的 gates。
- React workbench 已改为中文界面，并补上“因子表达式/模板 -> 流程配置”的前端状态传递。
- React workbench 已补上 Factor Factory run browser；前端只展示 API / artifact 结果，不重新计算 IC、returns、weights 或 turnover。
- 本地 API 与前端服务已用小范围 smoke 验证；2023-2025 私有 smoke 已覆盖 tensor / dataframe 与 long-only / long-short；2026-05-23 又复跑 2024 acceptance 小样本并生成 old/new 比较，提交前仍以完整 Python 和 frontend 检查作为准入。
- A 股真实私有 runs 必须走 A 股专用 adapter：`python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`。通用 `evaluate-parquet` 只作为平台 evaluation / 外部 Parquet adapter 验收，不替代 A 股 adapter 验收。
- P4/P5 本地示例入口为 `scripts/phase1_acceptance.py`：`run` 复跑 2024 A 股 adapter acceptance runs，`compare` 低内存比较 old/new，`factory-smoke` 生成 Factor Factory candidate summary。

Phase 1 验收口径下已补齐：

1. Factor Factory 本地 workbench 浏览入口。
2. A 股 existing-factors start layer 小样本示例和 old/new 比较。
3. Factor Factory 小样本示例，生成 candidate summary 和 failed candidate log。

Phase 1 之后下一阶段进展：

1. N3 Artifact ZIP packaging 已合入 `main`：selected ZIP item validation、cache identity、file collection 和 dynamic item handling 已移入 `artifacts/artifact_package.py`。
2. N1 A 股 pipeline start-from runner 已合入 `main`：外部接口为 snapshot/run split 和 `start_from` 模型，`POST /api/pipeline-snapshots` 只保存配置，`POST /api/pipeline-runs` 执行统一 runner 并固定运行到 evaluation。
3. N2 后端 tradability artifact/API 已合入 `main`：新增 `diagnostics/tradability_diagnostics.csv`，并通过 schema `0.5`、`manifest.display`、API preview、direct download 和 ZIP download 暴露；N2 frontend 仍是后续任务。
4. N4 前端 feature split 已合入 `main`：`web/src` 已拆为 `shell/`、`features/` 和 `shared/`，未新增 diagnostics UI，未改变 API contract。
5. 退休 API 字段和双 runner 命名已从源码入口移除，并仅在负向测试中保留拒绝用例。
6. 当前推荐先做 N7 A 股 `EvaluationMatrixRun` 与 2023-2025 canonical 样例生成：一个 API-visible run 对应一个 evaluation run；计算层支持多个 feature / label / portfolio mode / kernel 一起批量或并行计算，结果层用 `combination_id` 管理 feature / label / portfolio mode / kernel 切片；schema `0.6` 按 `dimensions/combinations.csv` star schema 重构；`weights` 第一阶段直接写 Parquet；tensor 是长期主路径，dataframe 是 reference / fallback。之后再做 N2 frontend、N5 和 N6。

## 当前研究资产

历史 A 股研究资产仍是平台第一版 market module 的基础：

- `data/raw_private/csmar/`：用户手工导出的 CSMAR 原始文件。
- `data/interim_private/csmar/`：规范化、裁剪、可人工审计的 interim 层。
- `data/processed_private/csmar/panel_full/research_panel_2014_2025.parquet`：full research panel。
- `data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet`：slim factor and label table。
- `reports/private_real/factor_evaluation/full_2024_2025/`：历史 2024-2025 long-only full-sample factor evaluation；旧 schema run 已退出当前有效 run 口径。

这些产物是私有真实数据产物，不作为公开材料提交或复制。

## 当前已验证的历史事实

以下是早期 A 股 workflow 已记录的验证事实，保留为历史证据，不自动代表当前结构下所有新路径已通过：

- `conda run -n analysis python -m pytest`：53 passed。
- `conda run -n analysis python -m ruff check .`：通过。
- raw / interim / panel / factor 检查无 blocking issue。
- 2014-2025 full panel 和 full factor Parquet 已生成。
- 2024-2025 long-only full run 已生成，包含 evaluation 年度 breakdown。
- long-only 不变量曾检查通过：`portfolio_mode=long_only`、`short_count=0`、`short_return=0`，0 成本下 long/gross/net return 一致。

迁移后曾按 `src/quant_research_platform/`、API、web、pipeline 和 docs 合同完成一轮 review/fix 与验证；2026-05-23 又进一步收紧目录决策，并已执行到 `interfaces/api`、`interfaces/cli`、`datasets`、`markets/implementations/a_share`、`pipelines/a_share`。旧路径残留扫描、Phase 1 相关后端测试、完整 pytest、ruff、前端 test/lint/build、CLI help、A 股 adapter help、API app import、architecture Markdown 链接检查和 `git diff --check` 已通过。2026-05-24 又完成 code-map 重构审计、项目文档一致性同步和当前 API/contract 轻量导入检查；新的 Phase 2/扩展设计仍需单独执行。

## 下一步

1. 先做 N7：稳定 A 股 `EvaluationMatrixRun`、`combination_id` 和 tensor-native 多维评价合同，并生成 2023-2025 canonical evaluation / Factor Factory 样例；默认评价集合为 1 个 run，内部至少覆盖 `2 kernels x 2 portfolio modes x 2 labels x 3 features = 24` 个 result combinations；artifact schema `0.6` 使用 star schema，`weights` 第一阶段直接写 Parquet，frontend/API 对所有 display tables 提供 20 行 preview，`compare_evaluation_sets` 改成网页/API 可展示和可画图的低内存 summary comparison，另增只输出表格/JSON/Markdown 的 Evaluation Comparison Report，dataframe/tensor parity 使用分层 `rtol/atol` 容差。第一晚真实数据验收先跑 `reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1`，不要求和旧样例做结果一致性比较；完整 canonical root 固定为 `reports/private_real/factor_evaluation/n7_eval_2023_2025_v1`，三年样例生成后再在实现验收阶段比较新实现输出是否一致。
2. N7 后继续做 N2 前端：消费 `tradability_diagnostics` 后端 artifact/API，完成 eligibility / tradability 诊断可视化。
3. 继续扩展前保留当前 P4/P5 smoke 命令作为回归门槛，并为新增能力写明确验收口径。
4. 后续能力如 ML prediction pipeline、风险/中性化扩展、成本/容量/组合归因需要单独设计。
5. 已合并的 N1/N2/N3/N4 工作树和分支可在用户批准 exact paths 后清理。

## Root Handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-05-21 | planning | 项目定位从单一 A 股因子研究项目收口为 A 股量化研究平台入口；临时平台蓝图记录数据、因子、ML prediction、风险、regime、evaluation、evidence workbench 和未来模拟盘边界。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-22 |
| 2026-05-22 | migration | 项目身份迁移为 `quant-research-platform`；A 股为第一版 enabled market module；evaluation、schema `0.5` artifacts、API/download 和 React evidence workbench 成为平台内置能力。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-22 |
| 2026-05-23 | phase1 | Phase 1 目标结构已执行并完成复验；后续架构债另列。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-24 |
| 2026-05-24 | docs sync | 已重构 `docs/code-map/`，同步根文档、architecture 文档和状态文件的当前架构表述。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-24 |
| 2026-05-24 | architecture debt | N3 Artifact ZIP packaging 下沉已合入 main：selected ZIP item validation、cache identity、file collection 和 dynamic item handling 已移入 `artifacts/artifact_package.py`，API download service 保留 HTTP response adapter 职责；未触碰 web。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |
| 2026-05-24 | next-stage implementation | N1 A 股 start-from pipeline runner 已合入 main：新增 `POST /api/pipeline-snapshots`，`POST /api/pipeline-runs` 统一为 `start_from`，统一 runner 编排 build/evaluate 到 evaluation，Factor Factory 改为 existing-factors start layer。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |
| 2026-05-24 | next-stage implementation | N2 后端第一层 tradability artifact/API 已合入 main：新增 `diagnostics/tradability_diagnostics.csv`，通过 `manifest.display`、diagnostics preview、direct download 和 ZIP download 暴露；前端可视化等待 N7 canonical 样例。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |
| 2026-05-24 | next-stage implementation | N4 前端 feature split 已合入 main：`web/src` 拆为 `shell/`、`features/` 和 `shared/`，仅同步前端 imports、tests 和 code-map，不新增 diagnostics UI，不改变 API contract。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |
| 2026-05-25 | planning | N7 A 股 evaluation run identity 与 2023-2025 canonical 样例生成成为当前前置任务：一个 API-visible run 对应一个 `EvaluationMatrixRun`；计算层可一起跑 feature / label / portfolio mode / kernel，结果层用 `combination_id` 管理单独评价切片；schema `0.6` 按 star schema 重构，`weights` 直接 Parquet，前端/API 做 20 行 preview；Factor Factory 适配 N7；N2 frontend、N5 和 N6 后续复用该 evaluation run identity 与样例。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |

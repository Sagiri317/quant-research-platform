# Quant Research Platform 项目状态

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 项目定位 | 本地多市场量化研究平台；A 股是第一版完整 market module |
| 当前分支 | `codex/a-share-quant-platform` |
| 当前目录 | 迁移期间仍在旧本地目录内工作；最终目录改名由用户单独执行 |
| 代码结构 | 主包为 `src/quant_research_platform/`；A 股模块在 `src/quant_research_platform/markets/a_share/`；前端在 `web/` |
| Conda 环境 | `analysis` |
| 数据范围 | 本地私有 CSMAR 2014-2025 interim / full panel / full factor 已有历史产物；迁移后仍遵守私有边界 |
| 平台合同 | evaluation contract、artifact schema `0.5`、`manifest.display`、FastAPI evidence API、下载和 React evidence workbench 已迁入平台命名空间 |
| 前端状态 | 本地 workbench 已中文化；因子表达式/模板快照可从因子研究台传入流程配置页；当前流程配置仍是 snapshot，不是完整 pipeline executor |
| 市场范围 | A 股 enabled；crypto / other markets 为 planned placeholder，不代表真实支持 |
| 私有边界 | CSMAR raw/interim/processed 和 `reports/private_real/` 不提交、不公开 |
| 交易边界 | 不做投资建议、paper trading、live trading、券商/交易所/API 连接或部署服务 |

## 本轮迁移状态

本轮目标是把项目身份从单一 A 股研究项目切换为 `quant-research-platform`。长期后续功能池收敛在 `docs/architecture/future-capabilities.md`；历史迁移计划和历史迁移设计已由 Git 历史保留，不再作为当前待办清单。

当前阶段已完成可提交的迁移闭环：

- 用户入口文档改为 Chinese-first 的平台说明。
- `ARCHITECTURE.md` 和 `EVALUATION_INTERFACE.md` 记录平台架构、evaluation contract、artifact schema `0.5`、`manifest.display`、API/download/frontend 边界。
- `docs/architecture/future-capabilities.md` 收敛原 A 股平台蓝图、旧 evaluation 后续能力总结和迁移设计中的长期路线，保留 A 股、ML prediction、风险、regime、workbench 和未来模拟盘边界，但不把模拟盘写成当前范围。
- `docs/architecture/architecture-refactor-design.md` 记录下一轮 contract-first 架构重构设计，冻结 `EvaluationSpec`、`run_evaluation`、schema `0.5`、API route 和前端下载合同。
- `docs/architecture/phase-1-implementation-checklist.md` 把 Phase 1 重构拆成可执行任务、验收项、注册表分级、偏差审计和测试矩阵。
- `docs/architecture/handoff-before-directory-rename.md` 记录目录改名前和新对话启动时的交接入口。
- 项目级 `AGENTS.md` 增加 `src/quant_research_platform/` 结构、Conda `analysis`、私有数据、无外部 API/凭据、无交易、exact-path deletion、frontend/API/artifact contract、旧 evaluation reference 仅作参考、`source_alias + run_id` 运行身份边界。
- `.codex/skills/quant-research-platform-maintenance/` 记录后续维护 artifact/API/download/frontend 合同时必须执行的 gates。
- React workbench 已改为中文界面，并补上“因子表达式/模板 -> 流程配置”的前端状态传递。
- 本地 API 与前端服务已用小范围 smoke 验证；2023-2025 私有 smoke 已覆盖 tensor / dataframe 与 long-only / long-short，提交前仍以完整 Python 和 frontend 检查作为准入。
- A 股真实私有 runs 的主入口是 `python -m quant_research_platform.markets.a_share.evaluation.run_factor_evaluation`；通用 `evaluate-parquet` 只作为平台 evaluation / 外部 Parquet adapter 验收，不替代 A 股 adapter 验收。

## 当前研究资产

历史 A 股研究资产仍是平台第一版 market module 的基础：

- `data/raw_private/csmar/`：用户手工导出的 CSMAR 原始文件。
- `data/interim_private/csmar/`：规范化、裁剪、可人工审计的 interim 层。
- `data/processed_private/csmar/panel_full/research_panel_2014_2025.parquet`：full research panel。
- `data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet`：slim factor and label table。
- `reports/private_real/factor_evaluation/full_2024_2025/`：历史 2024-2025 long-only full-sample factor evaluation；旧 schema run 已退出当前有效 run 口径。

这些产物是私有真实数据产物，不作为公开材料提交或复制。

## 当前已验证的历史事实

以下是迁移前 A 股 workflow 已记录的验证事实，保留为历史证据，不自动代表迁移后所有新路径已通过：

- `conda run -n analysis python -m pytest`：53 passed。
- `conda run -n analysis python -m ruff check .`：通过。
- raw / interim / panel / factor 检查无 blocking issue。
- 2014-2025 full panel 和 full factor Parquet 已生成。
- 2024-2025 long-only full run 已生成，包含 evaluation 年度 breakdown。
- long-only 不变量曾检查通过：`portfolio_mode=long_only`、`short_count=0`、`short_return=0`，0 成本下 long/gross/net return 一致。

迁移后已按 `src/quant_research_platform/`、API、web、pipeline 和 docs 合同完成阶段性验证；当前阶段是反复 review 与修复。真实私有 A 股 2023-2025 小样本 smoke 已覆盖 tensor / dataframe 与 long-only / long-short，下一步不再是补跑 smoke，而是把 review 收口、补齐前端下载/删除语义、文档一致性和完整验证。

## 下一步

1. 迁名前先审阅 `docs/architecture/handoff-before-directory-rename.md`，提交当前文档整理，然后由用户手动把本地目录改成 `quant-research-platform`。
2. 新对话先读 `AGENTS.md`、`PROJECT_STATUS.md`、`docs/architecture/handoff-before-directory-rename.md` 和 `docs/architecture/README.md`。
3. 确认 `docs/architecture/architecture-refactor-design.md` 和 `docs/architecture/phase-1-implementation-checklist.md` 后，再进入 P0-P5 的实际代码重构。
4. 继续执行 review -> 修复 -> 验证循环，直到 artifact/display/download、API、web、A-share、文档和项目 skill 不再出现实质性问题。
5. 继续完善流程配置页，把 label、组合口径、日期、成本、run metadata 等字段全部接入真实配置快照；后续再设计长任务编排，把页面接到 A 股 build/evaluate 专用入口。
6. 在 long-only 口径下重跑成本敏感性、924 regime、行业/市值中性化、风险暴露和正式组合约束分析。
7. 基于迁移后的平台结构设计 ML prediction pipeline；建议使用 high reasoning effort，因为它会同时影响数据合同、label 口径、防泄漏、私有边界和 evaluation handoff。
8. 可清理未使用的 Vite 默认前端资源，但需要单独确认具体路径：`web/src/assets/hero.png`、`web/src/assets/react.svg`、`web/src/assets/vite.svg`。

## Root Handoff / 根目录回传信号

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-05-21 | planning | 项目定位从单一 A 股因子研究项目收口为 A 股量化研究平台入口；临时平台蓝图记录数据、因子、ML prediction、风险、regime、evaluation、evidence workbench 和未来模拟盘边界。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-22 |
| 2026-05-22 | migration | 项目身份迁移为 `quant-research-platform`；A 股为第一版 enabled market module；evaluation、schema `0.5` artifacts、API/download 和 React evidence workbench 成为平台内置能力；旧目录名等待用户后续手动改名。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-22 |
| 2026-05-22 | milestone | 平台迁移、前端中文化、小步美化、因子表达式到流程配置快照闭环已完成；2023-2025 tensor / dataframe 与 long-only / long-short 小样本 smoke 已生成并进入 review 收口。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | closed 2026-05-22 |
| 2026-05-23 | handoff | 迁名前 handoff、新对话阅读入口、Phase 1 设计和执行清单已收口；下一步是提交文档整理、用户手动改名为 `quant-research-platform`，再在新对话执行 Phase 1 重构。 | `WORKSPACE_STATUS.md`、`PLAN_INDEX.md`、`control/PROJECT_REGISTRY.md` | open |

# Quant Research Platform 项目状态

本文只记录当前 operational truth：当前状态、当前有效样例、当前 blocker / next steps、
以及需要同步到 workspace root 的 handoff 信号。历史 closeout、完成分支细节和旧
schema 验收故事不在本文展开；Phase 1 历史验收见
`docs/architecture/phase-1-acceptance-record.md`，下一阶段任务见
`docs/architecture/next-stage-roadmap.md`，实现流向见 `docs/code-map/`。

## 当前状态

| 项目 | 当前事实 |
|---|---|
| 项目定位 | 本地多市场量化研究平台；A 股是第一版完整 market module |
| 当前阶段 | Local Research Platform；Stable Core + High-Velocity Research Edge；contract-first / evidence-flow-first |
| 当前分支 | `main`；N5 MVS descriptive proxy diagnostics、N5 private real-data smoke、pipeline failed-run reliability MVS、N6 prediction handoff guardrails / synthetic smoke / local baseline、N6 prediction handoff MVS、prompt-template sync / polish 已进入 `main` |
| Conda 环境 | `analysis` |
| 代码结构 | 主包为 `src/quant_research_platform/`；核心目录包括 `interfaces/api`、`interfaces/cli`、`datasets`、`markets/implementations/a_share`、`pipelines/a_share`、`research/factor_factory` |
| 平台合同 | evaluation contract、artifact schema `0.7`、diagnostics registry、`manifest.display`、FastAPI evidence API、下载和 React evidence workbench 已迁入平台命名空间；portable metadata 使用 `run_context.json` |
| A 股 Pipeline | `POST /api/pipeline-runs` 使用 `start_from` 执行到 evaluation；failed-run reliability MVS 覆盖 persisted status fallback、failed checkpoints consistency、stable `failure_code`、sanitized failed payload 和 partial failed evaluation guard |
| Factor Factory | 已有 registry、批量评估、artifact `0.1`、API 和 workbench 浏览入口；候选评估通过 existing-factors start layer 调用统一 runner |
| 市场范围 | A 股 enabled；crypto / other markets planned / disabled，不代表真实支持，也不保留空实现包 |
| 私有边界 | CSMAR raw/interim/processed 和 `reports/private_real/` 不提交、不公开 |
| 交易边界 | 不做投资建议、paper trading、live trading、券商/交易所/API 连接或部署服务 |

## 当前有效样例

以下样例是当前可用的本地私有 evidence / regression anchors，仍属于 private
real-data outputs，不提交、不公开、不复制到聊天或公共材料中：

- `schema07_eval_2023_2025_v1`：当前 2023-2025 schema `0.7` canonical sample，含 `rev_120d`。
- `schema07_eval_2023_2025_rev120_v1`：当前 2023-2025 schema `0.7` `rev_120d` kernel / portfolio mode parity shard。
- `20260527_rev120_factory_smoke_v1`：当前 Factor Factory `rev_120d` smoke 样例。
- N5 private real-data smoke passed：run id `n5_private_smoke_2025h2_rev120_v1`；数据范围只记录为 2025H2 aggregate smoke；schema `0.7` validation、diagnostics tables、`manifest.display`、API preview 和 selected ZIP smoke 均通过。
- 当前 full research panel 与 slim factor / label table 仍在 private processed boundary 内。

Historical schema `0.5` / `0.6` runs、old 2023H1 smoke、N7 `n7_eval_*` runs
和旧 full-run outputs 只作 historical evidence，不是当前 valid run-contract examples。

## 当前 blocker / 风险

1. Full N5 risk/regime/cost/capacity capability 尚未设计完成；当前 N5 MVS 只能作为
   descriptive proxy diagnostics groundwork，不代表真实风险模型或容量模型。
2. N6 ML prediction handoff MVS 仍是 local research baseline / prediction handoff；它记录 OOS / embargo / fold metadata 并可交给 evaluation 消费 prediction，但不代表完整 ML pipeline、alpha proof、A-share prediction adapter、真实风险/容量模型、交易仿真或投资建议。
3. Pipeline retry/resume/overwrite 仍未实现；global API error envelope 也仍未迁移。
4. 单个 32-combination 2023-2025 full run 仍受 weights/remap memory path 约束；扩展前应先设计低内存写出路径，或继续按 shard 作为正式回归输入。
5. `tradability_diagnostics_summary` 暂缓；若需要全交易日平均诊断，应新增 artifact/API summary，而不是从 preview 前几行临时计算。

## 下一步

1. 单独决定下一批 N6 范围：多 fold / walk-forward 扩展、A-share prediction adapter 设计，或继续保持 local synthetic handoff；保持 training outside evaluation，evaluation 只消费 prediction。
2. 如需继续 pipeline reliability，单独做第二个 minimal vertical slice 决策审计，不和 N6 扩展混在同一批。
3. 基于当前 2023-2025 canonical / `rev_120d` evidence，保留低内存 comparison、图表分页预览和浏览器验收作为后续回归门槛。
4. 如需单个 32-combination 2023-2025 run，先处理 weights/remap 低内存写出问题。

## 已完成工作摘要

- Phase 1 已完成并由 `docs/architecture/phase-1-acceptance-record.md` 作为历史验收记录保存。
- N1/N2/N3/N4/N7/N8 已合入 `main`；详细实现和旧 schema 验收故事不再作为当前状态展开，必要时查 Git history、`docs/archive/superpowers/` historical execution records 和相关 code-map。
- N5 MVS descriptive proxy diagnostics、N5 private real-data smoke、pipeline failed-run reliability MVS、N6 initial handoff / smoke / local baseline、N6 prediction handoff MVS，以及 prompt-template sync / polish 已进入 `main`；N6 MVS 补齐 local baseline OOS / embargo / fold metadata、`ml baseline train` CLI、private-output boundary guard 和 synthetic evaluation handoff tests；当前有效 private smoke run id 为 `n5_private_smoke_2025h2_rev120_v1`。
- Retry/resume/overwrite、global API error envelope migration、full N5 capability 和 N6 handoff audit 仍是未来工作。

## Root Handoff / 根目录回传信号

子项目会话如果改变当前主题、阶段、里程碑、阻塞、下一步、公开成熟度，或
数据/API/Git/Conda/隐私边界，应在本表新增 `open` 信号。根目录同步完成后，
将状态改为 `closed` 并记录同步日期。普通实现细节、测试、notebook 调试和
局部代码变更不进入根目录，除非它们改变里程碑级状态。

| 日期 | 类型 | 变化 | 需要根目录同步 | 状态 |
|---|---|---|---|---|
| 2026-06-01 | status | N6 prediction handoff guardrails、synthetic handoff smoke、local numpy/pandas ML baseline trainer、N6 prediction handoff MVS 已进入 `main`；prompt-template sync / polish 已进入 `main`。当前仍不是完整 ML pipeline、deep learning backend、model registry、feature store、A-share prediction adapter、paper/live trading、deployment、scheduler、worker 或 external connector。下一步改为单独决定 N6 next slice 或 pipeline reliability second slice；不要直接进入 broad implementation。 | WORKSPACE_STATUS, PLAN_INDEX | closed 2026-06-01 |
| 2026-06-01 | status | N6 prediction handoff MVS 已进入 `main`：完成 local baseline OOS / embargo / fold metadata、`ml baseline train` CLI、private-output boundary guard、synthetic `evaluate-parquet --kind prediction` handoff tests；evaluation 只消费 prediction，training 保持在 `research/ml_baseline.py` 外围 research utility；仍不是 alpha proof、完整 ML pipeline、A-share prediction adapter、paper/live trading、deployment、scheduler、worker、model registry、feature store 或 external connector。 | WORKSPACE_STATUS, PLAN_INDEX | open |

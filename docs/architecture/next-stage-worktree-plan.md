# Next Stage Worktree Plan / 下一阶段分支与工作树计划

本文把 [next-stage-roadmap.md](next-stage-roadmap.md) 的近期任务拆成适合隔离开发的分支和工作树。它不是功能验收记录；功能验收仍以后续各分支的测试和 code review 为准。

## 当前基线

- `main` 已包含 N1 A 股 `start_from` runner。
- `main` 已包含 N2 backend artifact/API tradability diagnostics。
- `main` 已包含 N2 frontend diagnostics display。
- `main` 已包含 N3 Artifact ZIP packaging 下沉。
- `main` 已包含 N4 frontend feature split。
- `main` 已包含 N7 A 股 evaluation run identity、schema `0.6`、`combination_id`、kernel 维度、Factor Factory N7 artifact 读取和 public-downloadable metadata 脱敏。
- 2023-2025 smoke evaluation 与 Factor Factory 旧样例已保留为 `*_baseline_before_2023_2025_regen` 历史参考目录，不进入 N7 正式 compare contract。
- Pipeline 对外语言固定为 snapshot vs run：
  - `POST /api/pipeline-snapshots` 只保存配置。
  - `POST /api/pipeline-runs` 执行到 evaluation。
  - executable run 只用 `start_from` 控制开始层。
- 当前没有保留已完成 N1/N3 的本地工作树；N7 已合入 `main`，新任务应从干净 `main` 创建新工作树。

## 原则

- 每个分支只交付一个可 review 的主题，避免把 artifact、API、pipeline、frontend 和 docs 大范围混在一起。
- 新增用户可见 artifact、API、download、frontend table/chart 时，同步 `manifest.display`、API tests、frontend tests 和 code-map。
- 不在这些分支里接入外部 API、自动下载 CSMAR、读取凭据、部署服务、paper trading 或 live trading。
- 工作树创建前必须确认当前 checkout 干净，并确认 `.worktrees/` 被 Git 忽略。
- Vite 默认资源清理仍是独立低优先级 hygiene，不混进功能分支。

## 已完成分支

| 分支 | 路线图任务 | 状态 | 结果 |
|---|---|---|---|
| `codex/a-share-build-runner` | N1 A 股 start-from runner | merged to `main` | snapshot/run split、`start_from` runner、完整 checkpoint 链和 Factor Factory existing-factors start layer 已落地 |
| `codex/tradability-artifacts` | N2 后端 artifact/API 诊断 | merged to `main` | `diagnostics/tradability_diagnostics.csv` 最初在 schema `0.5` 期间落地，当前 schema `0.6` 合同继续通过 `manifest.display`、API preview、direct download 和 ZIP download 暴露 |
| `codex/artifact-zip-packaging` | N3 ZIP packaging 下沉 | merged to `main` | ZIP item validation、cache identity、file collection 和 dynamic item handling 已移入 `artifacts/artifact_package.py` |
| `codex/frontend-feature-split` | N4 前端 feature split | merged to `main` | `web/src` 已拆为 `shell/`、`features/` 和 `shared/`，未新增 diagnostics UI，未改变 API contract |
| `codex/n7-evaluation-run-implementation` | N7 A 股 evaluation run identity 与 2023H1 smoke | merged to `main` | schema `0.6`、`combination_id`、kernel 维度、N7 evaluation artifact、Factor Factory 读取、public-downloadable metadata 脱敏和 2023H1 private smoke 已落地；完整 2023-2025 canonical regeneration 按需执行 |
| `codex/tradability-workbench` | N2 frontend diagnostics display | merged to `main` | Evidence workbench 消费 `diagnostics/tradability_diagnostics.csv` / diagnostics preview API；诊断卡片用 `limit=1&offset=total_rows-1` 读取当前组合最后一行，CSV 明细预览仍保持 20 行；未混入 N8 横切收敛 |

## 建议后续分支

| 分支 | 路线图任务 | 范围 | 依赖 | 冲突风险 |
|---|---|---|---|---|
| `codex/contract-consumer-convergence` | N8 架构简化与合同消费收敛 | `manifest.display` preview/download helper、factor-template route/service 拆分、preview endpoint metadata、A-share service entry、CSV upload legacy/dev 降级 | 基于 N7 schema `0.6` / `combination_id`、N2 backend diagnostics、N3 artifact package 和 N4 frontend feature split | 中到高。横跨 API、artifact、pipeline、Factor Factory 和 frontend，应先写详细计划 |
| `codex/risk-regime-diagnostics` | N5 风险、regime、成本和容量扩展 | research diagnostics artifacts、API exposure、report/docs | 依赖 N7 evaluation run identity 和样例稳定；复用 N2 诊断 artifact/display 模式 | 中到高。需要先写详细计划 |
| `codex/ml-prediction-handoff` | N6 ML prediction handoff | prediction manifest、handoff schema、evaluation adapter-like input、tests/docs | 可独立设计，但应复用 N7 evaluation run identity，不能混入 N2/N5 | 中到高。合同设计优先 |

## 推荐顺序

1. N7 已完成并合入 `main`
   - 先稳定“一个 API-visible run 是一个 `EvaluationMatrixRun`”的长期语义。
   - Evaluation run 内部用 `combination_id` 管理 feature / label / portfolio mode / kernel / factor set 等结果切片。
   - `factor_set` 是一起输入计算的 feature 批次；第一阶段保持每个 feature column 单独评价。
   - Tensor 是长期主路径，应支持 matrix-native 输入和 combination-keyed 输出；dataframe 是 reference / fallback / debugging 路径。
   - 2023-2025 canonical 样例默认覆盖 tensor/dataframe、long-only/long-short、`forward_return_5d`/`forward_return_20d`，以及 `mom_20d`、`vol_20d`、`avg_amount_20d` 三个 feature。
   - 默认样例应形成 1 个 evaluation run，内部至少 24 个 result combinations：`2 kernels x 2 modes x 2 labels x 3 features`。
   - 新样例应包含当前 artifact contract 要求的 diagnostics，包括 `diagnostics/tradability_diagnostics.csv`。
   - 新样例应使用 N7 后的 `kernel` 命名和 schema `0.6` evaluation-set 合同；旧 schema `0.5` 样例不进入正式 compare contract。
   - Schema `0.6` 应按 star schema 重构：`dimensions/combinations.csv` 是维度 source of truth，fact tables 以 `combination_id` 连接，不在大表里重复塞入所有字符串维度。
   - `manifest.display` 应声明 `dimension_source`、table format、join key、downloads 和 charts；前端从 display contract 与 combinations 表生成 selectors。
   - 前端不做视觉重设计，但必须完整展示 schema `0.6` 暴露的用户可见结果；所有 display tables，包括 Parquet `weights`，都必须支持 20 行 preview。
   - `weights` 第一阶段就直接写 Parquet，不再把完整 weights CSV 作为主产物。
   - Factor Factory 应适配 N7 evaluation run 合同，用 `candidate_id + combination_id` 汇总候选表现；不要让 N7 迁就旧 candidate-run 心智。
   - Factor Factory 第一阶段只做适配，不强制 bump schema；只有持久化 artifact 结构变化时才单独 bump。
   - `kernel` 是 evaluation run 内的维度，tensor/dataframe 不再拆成用户可见散 run。
   - `compare_evaluation_sets` 应改成默认低内存的网页/API summary comparison，只读 summary、manifest、combinations 和必要 summary metrics，继续支持大样本快速比较，并作为网页 comparison cards / tables / charts 的数据源。
   - 新增更详细的低内存 Evaluation Comparison Report，只输出表格、JSON 和 Markdown 报告，不进入主网页展示；它用于比较 schema `0.6` combinations 和 kernel parity。大样本验收不得使用会读完整 weights 或大时序表的 comparison 路径。
   - dataframe / tensor 共同覆盖结果应做 parity 对照：结构 exact match；same-kernel 回归建议 `rtol=1e-12, atol=1e-12`；dataframe/tensor common metrics 建议 `rtol=1e-9, atol=1e-10`；cumulative path metrics 建议 `rtol=1e-8, atol=1e-10`。
   - 第一晚真实数据验收先跑 `reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1`，覆盖 2023-01-01 到 2023-06-30 的 24 个 result combinations；不要求和旧样例做结果一致性比较。
   - 完整 canonical root 固定为 `reports/private_real/factor_evaluation/n7_eval_2023_2025_v1`，应等 schema/unit tests、2023H1 real smoke、low-memory summary comparison、Evaluation Comparison Report、parity 和前端 20 行 preview 通过后再跑。
   - `long_short` 在 N7 中是诊断组合口径，不声明真实可融券做空或可交易策略。
   - 完整 2023-2025 样例生成后，再在实现验收阶段比较新实现输出是否一致，不打印或公开原始 CSMAR 数据。

2. `codex/contract-consumer-convergence`
   - 收敛 `manifest.display` preview/download 解析，减少 evaluation runs 与 Factor Factory 的重复 helper。
   - 将 factor-template endpoint 从 Pipeline route/service 中拆出，保持 Pipeline API 只负责 snapshot/run。
   - 让前端通过后端下发的 preview endpoint metadata 消费 table preview，减少基于 path 的 route 推导。
   - 统一组合相关但非 chart 的 display table preview 规则，让 diagnostics、N5 风险/regime 表和 N6 prediction 相关表按 metadata 声明随 `combination_id` 刷新，不继续增加前端 hardcoded table key。
   - 把 A-share build/evaluate 调用沉成 service，CLI、pipeline runner 和 Factor Factory 都调用同一 service 层。
   - 将 CSV upload 进一步标记为 legacy/dev，可用显式 dev/upload 命名或本地开关降低主路径误读。
   - 该分支不应重做视觉、不改 A-share `start_from` 语义、不引入外部 API/部署/交易能力。

3. `codex/risk-regime-diagnostics` 或 `codex/ml-prediction-handoff`
   - 两者都应先写独立计划。
   - 不建议和 N2 frontend 同时抢同一批 artifact/API/frontend 文件。
   - 若 N8 尚未完成，N5/N6 计划必须明确哪些 display/API/frontend helper 是复用现状，哪些留给 N8，避免隐性重构混入研究合同分支。

## 不建议并行的组合

- `codex/frontend-feature-split`、`codex/tradability-artifacts` 和 `codex/tradability-workbench` 已完成；后续不要在 N8/N5/N6 中重新打开这些历史分支的范围。
- N2 frontend 的最终验收应使用 N7 之后、包含 `tradability_diagnostics.csv` 的样例；旧私有样例缺少 diagnostics，不能证明真实展示路径。
- 不要把 N5 风险/regime 扩展混进 N2 tradability 诊断。两者都属于 diagnostics，但研究语义和验收口径不同。
- 不要把 N6 ML prediction handoff 混进现有 factor evaluation runner。Prediction 是新的输入合同，应单独设计。
- 不要把 N8 的 cross-cutting contract cleanup 混进 N5/N6。N8 会触碰 artifact/API/download/frontend/pipeline/Factor Factory 多个面，应该单独计划和验收。
- 不要在 N8 中同时重做 Evidence 视觉或引入新的研究诊断。N8 的目标是降低耦合和重复，不是新增研究能力。

## 工作树创建策略

N7 已合入 `main`；不要再创建旧的 A-share run identity 或 N7 implementation 工作树。后续新任务从干净 `main` 创建新工作树：

```powershell
git status --short
git branch --show-current
git worktree list
git check-ignore -q .worktrees
git worktree add ".worktrees/<task-name>" -b "codex/<task-name>"
```

如果 `.worktrees/` 未被忽略，先更新 `.gitignore` 并单独提交该治理变更，再创建工作树。不要把工作树目录加入 Git。

## 每个分支的最低验收

Artifact/API/backend 分支：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/artifacts tests/interfaces/api tests/pipelines/a_share tests/research/factor_factory
conda run -n analysis python -m ruff check src tests scripts
git diff --check
```

Frontend 分支：

```powershell
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

跨 artifact/API/frontend 合同分支应补跑：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q
conda run -n analysis python -m ruff check src tests scripts
Push-Location web
npm test -- --run --maxWorkers=1
npm run lint
npm run build
Pop-Location
git diff --check
```

需要真实浏览器 UI 抽查时，用项目 wrapper，避免生成顶层 `output/`：

```powershell
& scripts/playwright_cli_project.ps1 open http://127.0.0.1:5173
& scripts/playwright_cli_project.ps1 snapshot
& scripts/playwright_cli_project.ps1 screenshot
```

## 相关计划文件

- N7 设计 spec：[../superpowers/specs/2026-05-25-n7-a-share-run-identity-design.md](../superpowers/specs/2026-05-25-n7-a-share-run-identity-design.md)
- N1 归档计划：[../superpowers/plans/2026-05-24-a-share-build-runner.md](../superpowers/plans/2026-05-24-a-share-build-runner.md)
- N3 归档计划：[../superpowers/plans/2026-05-24-artifact-zip-packaging.md](../superpowers/plans/2026-05-24-artifact-zip-packaging.md)
- N4 计划草案：[../superpowers/plans/2026-05-24-frontend-feature-split.md](../superpowers/plans/2026-05-24-frontend-feature-split.md)

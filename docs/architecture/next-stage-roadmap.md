# Next Stage Roadmap / 下一阶段路线图

本文记录 Phase 1 之后的近期任务。它不引入部署、外部 API、券商连接、paper trading、live trading 或实盘能力。最终能力边界见 [target-architecture-and-capabilities.md](target-architecture-and-capabilities.md)。

## 优先级

| ID | 任务 | 优先级 | 状态 | 目标 |
|---|---|---:|---|---|
| N1 | A 股 start-from pipeline runner | P0 | merged to `main` | 用统一 runner 从指定数据层开始，并运行到 evaluation |
| N2 | Eligibility / tradability 诊断可视化 | P0 | backend artifact/API merged to `main`; frontend waits for N7 | 让覆盖、过滤原因和影响先进入 artifact/API，下一步由 frontend 消费 |
| N3 | Artifact ZIP packaging 下沉 | P1 | merged to `main` | 把下载打包规则从 API service 下沉到 `artifacts/` |
| N4 | 前端 feature split | P1 | merged to `main` | 拆分当前平铺 workbench，降低后续功能耦合 |
| N5 | 风险、regime、成本和容量扩展 | P1 | planned | 补研究诊断，形成正式研究扩展路径 |
| N6 | ML prediction handoff | P2 | planned | 建立 prediction output 到 evaluation 的可复现接口 |
| N7 | A 股 evaluation run identity 与 2023-2025 样例生成 | P0 | design needed | 稳定一个 API-visible `EvaluationMatrixRun`，用 `combination_id` 管理 feature / label / mode / kernel 结果切片，并生成可供前端展示和后续诊断复用的三年 A 股样例 |

## 当前推荐顺序

N7 应先于 N2 frontend、N5 和 N6 的落地实现。

```text
1. N7 A 股 evaluation run identity 与 2023-2025 样例生成
2. N2 frontend diagnostics display
3. N5 风险、regime、成本和容量扩展
4. N6 ML prediction handoff
5. Pipeline history UI
```

原因：

- N2 后端已支持 `diagnostics/tradability_diagnostics.csv`，但现有 2023-2025 私有样例是旧生成物，尚不能证明前端能展示真实 N2 diagnostics。
- N5 会继续新增 diagnostics artifacts；如果 evaluation run identity、`combination_id` 和样例未稳定，会继续叠加在不清晰的展示结构上。
- N6 的 prediction handoff 也需要稳定表达 date range、feature / signal、label、kernel、portfolio mode、factor set 和输入身份，并能在同一个 evaluation run 页面里切换结果。

## N1 A 股 Start-From Pipeline Runner

当前实现：

- Snapshot-only 入口：`POST /api/pipeline-snapshots`。
- Executable run 入口：`POST /api/pipeline-runs`。
- 公开执行参数：`start_from`。
- 结束层固定为 evaluation，不提供 build-only 或 end-layer contract。
- 统一 runner：`src/quant_research_platform/pipelines/a_share/runner.py::run_a_share_pipeline`。

执行链路：

```text
existing_factors -> evaluation
rebuild_factors  -> build_factors -> evaluation
rebuild_panel    -> build_panel -> build_factors -> evaluation
rebuild_interim  -> build_interim -> build_panel -> build_factors -> evaluation
```

非目标：

- 不自动登录 CSMAR。
- 不下载外部数据。
- 不覆盖已有正式 private runs。
- 不接交易、模拟盘或部署服务。
- 不做多 factor payload；该能力后续单独设计。

验收边界：

- 每一步有 checkpoint、status、duration；row count、warnings、blocking issues 在可取得时写入。
- Checkpoints 展示完整链路，未执行层和失败后的后续层标为 `skipped`。
- 失败后能读取最后状态并明确 failed step。
- 默认不覆盖已有 run dir。
- API 拒绝退休字段。
- Factor Factory 通过 `existing_factors` start layer 调用统一 runner。

建议测试：

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/pipelines/a_share tests/markets/implementations/a_share
conda run -n analysis python -m pytest -q tests/interfaces/api tests/research/factor_factory tests/scripts
```

## N2 Eligibility / Tradability 诊断可视化

目标：

- 把 `eligible`、`can_enter`、`can_exit` 的覆盖数、过滤影响和异常分布写成 artifact。
- 前端可以看到每个 run 的可交易性诊断。

当前后端状态：

- Backend artifact/API 已合并到 `main`。
- 新增 diagnostics 通过 schema `0.5`、`manifest.display`、API preview、direct download 和 ZIP download 暴露。
- Frontend 展示仍是后续任务，应等 N7 生成带 diagnostics 的 2023-2025 canonical 样例后再验收。

当前后端输出：

```text
diagnostics/tradability_diagnostics.csv
```

后续扩展可以单独设计按年、按 segment 的派生表；当前 N2 前端任务应先消费 `tradability_diagnostics.csv`，不要依赖尚未落地的 summary/by-year/by-segment 文件名。

非目标：

- 不把 tradability 诊断写成真实可交易保证。
- 不为缺失 private data 做静默填补。

验收：

- `manifest.display.tables` 暴露新增诊断表。
- API preview 和 direct/ZIP download 只通过 `manifest.display` 访问。
- 前端不重新计算，只展示 artifact/API 结果。

实现顺序说明：

- 前端展示应等 N7 重新生成 2023-2025 样例后再做，避免用缺少 `tradability_diagnostics.csv` 的旧样例证明 UI 路径。

## N3 Artifact ZIP Packaging 下沉

目标：

- 把 selected ZIP item validation、cache key、file collection 和 dynamic item handling 从 API service 抽到 `artifacts/`。
- API 只负责 HTTP response 和 request parsing。

建议新增模块：

```text
src/quant_research_platform/artifacts/artifact_package.py
```

验收：

- unknown item、unsafe path、missing display item fail fast。
- ZIP cache identity 继续包含 `source_alias` 和 `run_id`。
- `comparison.csv` 动态生成逻辑保持清晰边界。
- API tests 不降覆盖。

Implementation note:

- Selected ZIP packaging rules now live in `src/quant_research_platform/artifacts/artifact_package.py`.
- API download service remains responsible for HTTP responses and dynamic API-visible content such as `comparison.csv`.

## N4 前端 Feature Split

目标：

- 把 `web/src` 从平铺文件拆成更清晰的功能域。

建议结构：

```text
web/src/features/evidence/
web/src/features/pipeline/
web/src/features/factor-studio/
web/src/features/factor-factory/
web/src/features/uploads/
web/src/features/runs/
web/src/shell/
web/src/shared/api/
web/src/shared/ui/
```

非目标：

- 不改 API contract。
- 不重做视觉系统。
- 不把前端变成计算层。

验收：

- run switching / combination switching 无 stale evidence。
- download picker 和 ZIP preset 仍来自 `manifest.display`。
- Factor Factory browser 仍不暴露 private filesystem paths。

## N5 风险、Regime、成本和容量扩展

目标：

- 补充风险暴露、regime 切片、成本敏感性和流动性/容量诊断。
- 形成后续研究报告所需的稳定诊断表。

非目标：

- 不写成真实收益承诺。
- 不进入交易风控或实盘控制。
- 不把 regime 用未来收益定义。

## N6 ML Prediction Handoff

目标：

- 建立模型输出到 evaluation 的稳定接口。
- 让 prediction 像 factor 一样成为可评价输入，而不是在 evaluation layer 中训练模型。

最低字段：

- model id / version
- feature set
- label definition
- train / validation / test split
- embargo / leakage controls
- data version
- code version
- random seed
- prediction column

验收：

- prediction Parquet 可通过 `EvaluationSpec` 或 A 股 adapter-like handoff 进入 evaluation。
- model manifest 可追踪训练数据、特征、label、split 和失败实验。
- walk-forward 和 OOS 结果明确区分。

## N7 A 股 Evaluation Run Identity 与 2023-2025 样例生成

设计 spec：[../superpowers/specs/2026-05-25-n7-a-share-run-identity-design.md](../superpowers/specs/2026-05-25-n7-a-share-run-identity-design.md)

目标：

- 明确长期 run 语义：一个 API-visible evaluation run 是一个 `EvaluationMatrixRun`，内部用 `combination_id` 管理多个结果切片。
- 明确计算语义和展示语义分离：多个 feature / label / mode / kernel 可以一起批量或并行计算，但评价和前端展示仍按 `combination_id` 单独切片。
- 建立可重复生成 2023-2025 A 股样例的正式入口。
- 让 N2 frontend、N5 diagnostics 和未来 N6 prediction handoff 共享同一套 evaluation run / combination identity。

推荐 evaluation identity 至少包含：

```text
market
date range
kernel set
portfolio mode set
label set
factor set id
factor columns / feature axis
factor definition/version
weight profile/config
cost profile/config
benchmark profile set config
tradability/evaluation config when it changes behavior
```

默认展示样例：

```text
date range: 2023-01-01 -> 2025-12-31
market: a_share
kernels: tensor, dataframe
portfolio modes: long_only, long_short
labels: forward_return_5d, forward_return_20d
factors: mom_20d, vol_20d, avg_amount_20d
cost_bps: 0
```

这会形成 1 个默认 `EvaluationMatrixRun`，内部至少包含 24 个 `combination_id` 结果切片：

```text
2 kernels x 2 portfolio modes x 2 labels x 3 features = 24 result combinations
```

其中 `factor_set` 是一起输入计算的 feature 批次；第一阶段仍保持每个 feature column 单独评价。未来如果要评价多因子整体 signal，应作为新的 signal/model 结果轴加入，而不是取代单 feature 评价。

其中 `long_short` 在 N7 中是因子分组诊断口径，不等于真实可融券做空策略；如果未来要声明真实可交易 long-short，需要单独建模 borrow、short availability、margin、成交、滑点和容量约束。

代码边界：

- `markets/implementations/a_share/evaluation/` 是 A 股 processed factor table 到 matrix evaluation artifacts 的正式入口，可承载 matrix spec、`combination_id`、tensor-native 输入和 dataframe reference/fallback。
- `kernels/` 中 tensor 是长期主路径，应支持 matrix-native 输入和 combination-keyed 输出；dataframe 是 reference / fallback / debugging 路径。
- `research/factor_factory/` 是 Factor templates / candidates 到 downstream evaluation 的批量入口；它应适配 N7 evaluation run 合同，可用 `candidate_id + combination_id` 汇总候选表现，而不是让 N7 迁就旧 candidate-run 心智。
- `pipelines/a_share/` 只负责 `PipelineConfig`、`start_from`、snapshot、checkpoint 和 runtime status；不负责选择前端默认样例、样例命名或研究展示矩阵。
- `scripts/phase1_acceptance.py` 保留为 Phase 1 验收工具，不继续扩张成长期样例生成系统。

历史样例状态：

现有 2023-2025 smoke 和 Factor Factory 样例已保留为历史参考目录，不进入 N7 正式 compare contract。当前目录包括：

```text
reports/private_real/factor_evaluation/smoke_2023_2025_dataframe_long_only_baseline_before_2023_2025_regen
reports/private_real/factor_evaluation/smoke_2023_2025_dataframe_long_short_baseline_before_2023_2025_regen
reports/private_real/factor_evaluation/smoke_2023_2025_tensor_long_only_baseline_before_2023_2025_regen
reports/private_real/factor_evaluation/smoke_2023_2025_tensor_long_short_baseline_before_2023_2025_regen
reports/private_real/ff/0523_baseline_before_2023_2025_regen
reports/private_real/ff/20260523_acceptance_latest_baseline_before_2023_2025_regen
```

这些目录位于 `reports/private_real/`，仍属于私有本地 evidence，不应提交或公开。

验收边界：

- 能 plan/dry-run 出将生成的 2023-2025 evaluation run、`combination_id` 列表和输出路径。
- N7 新生成物使用 schema `0.6` evaluation-set 合同；旧 schema `0.5` run 不进入正式 compare contract。
- 新 artifacts 使用围绕 `dimensions/combinations.csv` 的 star schema；大 fact tables 以 `combination_id` 连接维度，不重复写入所有字符串维度。
- `manifest.display` 必须声明 `dimension_source`、表路径、格式、join key、downloads 和 charts；前端不得硬编码未声明表路径。
- 前端不做视觉重设计，但必须完整展示 schema `0.6` 暴露的用户可见结果；所有 display tables，包括 Parquet `weights`，都必须支持 20 行 preview。
- `weights` 第一阶段就直接写 Parquet，不再把完整 weights CSV 作为主产物。
- 新 artifacts 包含当前 artifact contract 要求的 diagnostics，包括 `diagnostics/tradability_diagnostics.csv`。
- 前端后续可以在一个 evaluation run 页面内切换 feature / label / portfolio mode / kernel 展示不同 combination result。
- dataframe / tensor 共同覆盖结果应能做 parity 对照：结构和离散字段 exact match；same-kernel 回归建议 `rtol=1e-12, atol=1e-12`；dataframe/tensor common metrics 建议 `rtol=1e-9, atol=1e-10`；cumulative returns / drawdown 等长链路指标建议 `rtol=1e-8, atol=1e-10`。
- `compare_evaluation_sets` 应改成默认低内存的网页/API summary comparison，只读 summary、manifest、combinations 和必要 summary metrics，继续支持大样本快速比较，并作为网页 comparison cards / tables / charts 的数据源。
- 新增更详细的低内存 Evaluation Comparison Report，只输出表格、JSON 和 Markdown 报告，不进入主网页展示；它用于比较 schema `0.6` combinations 和 kernel parity。大样本验收不得使用会读完整 weights 或大时序表的 comparison 路径。
- 第一晚真实数据验收先跑 `reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1`，覆盖 2023-01-01 到 2023-06-30 的 24 个 result combinations；不要求和旧样例做结果一致性比较。
- 完整 canonical root 固定为 `reports/private_real/factor_evaluation/n7_eval_2023_2025_v1`，应等 schema/unit tests、2023H1 real smoke、low-memory summary comparison、Evaluation Comparison Report、parity 和前端 20 行 preview 通过后再跑。
- 完整 2023-2025 样例生成后，再在实现验收阶段比较新实现输出是否一致。
- 对比不读取或打印原始 CSMAR 私有数据。
- 前端 root 配置后续可以指向新的 2023-2025 evaluation 展示 root。

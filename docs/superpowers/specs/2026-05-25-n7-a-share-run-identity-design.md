# N7 A 股 Evaluation Run Identity 与样例设计

状态：待实现设计草案，已按 `docs/architecture/` 复核并按最新方向重写
日期：2026-05-25
范围：N7 A 股 `EvaluationMatrixRun`、`combination_id`、`kernel` 命名、tensor-native 计算路径、Factor Factory 下游 evaluation run 引用，以及 2023-2025 canonical 样例。

## 1. 核心目标

N7 的目标不是把一个评价请求拆成一堆用户需要手动面对的散乱 runs，而是建立一个长期稳定的多维评价合同：

- 一个 API-visible run 表示一个 `EvaluationMatrixRun`；
- 一个 `EvaluationMatrixRun` 内部包含多个 `combination_id`；
- 每个 `combination_id` 表示一个完整参数组合的结果切片；
- 前端在同一个 run 页面里按 feature、label、portfolio mode、kernel、factor set 等维度切换展示结果；
- 计算层可以把多个 feature、label、portfolio mode 和 kernel 一起批量或并行计算；
- 评价和展示层仍按 `combination_id` 单独查看每个结果切片；
- tensor 是长期主计算路径，应该支持 matrix-native 输入和批量输出；
- dataframe 是 reference / fallback / debugging 路径，可以先用 per-combination 循环模拟矩阵输出；
- `manifest.display` 继续是 API、前端、下载和对比的唯一展示合同。

用户心智应是：

```text
一个评价任务 -> 一个 evaluation run -> 一个界面切换多个结果切片
```

工程心智应是：

```text
EvaluationMatrixRun
  dimensions
  combinations.csv
  combination_id-keyed tables/charts/downloads
```

## 2. 关键设计决定

### 2.1 Run 是 evaluation run，不是单组合 run

N7 之后应把 A 股 evaluation 的长期 API-visible run 语义定义为：

```text
EvaluationMatrixRun = 一个可审计、可展示、可下载、可对比的矩阵评价产物
```

它可以只包含一个 combination，也可以包含多个 combinations。

单组合只是 evaluation run 的退化情况，不应该反过来决定长期设计。

### 2.2 Combination 是结果切片

`combination_id` 是矩阵内部最重要的连接键。它把每个表格、图表、诊断、下载项和前端选择状态连接起来。

N7 第一阶段应保持现有 evaluation 的核心语义：每个 factor column 是一个可单独评价的结果轴。也就是说，`feature` 是 result axis，`factor_set` 是输入批次或研究集合，不是第一阶段唯一的评价对象。

一个 result combination 至少由这些字段定义：

```text
combination_id
factor_set_id
feature
label
kernel
portfolio_mode
weight_profile_id
cost_profile_id
benchmark_profile_set_id
universe_policy_id
eligibility_policy_id
tradability_policy_id
```

如果未来实现多因子模型、组合 signal 或 prediction score，可以把 `signal_id`、`model_id`、`prediction_id`、`risk_model_id`、`neutralization_policy_id` 等作为 future optional axes；不要把这些占位列放进 schema `0.6` 第一阶段的 required columns，也不要用它们替代第一阶段的单 feature 评价。

如果未来加入 neutralization、risk model、regime、capacity 或 prediction input，只要它们改变结果语义，也应进入 combination identity。

### 2.3 计算一起做，评价单独看

N7 需要明确区分两层语义：

```text
computation strategy = 如何计算
result identity      = 如何命名、展示、比较和下载结果
```

计算层可以把一个 factor set 里的多个 features、多个 labels、多个 portfolio modes 和多个 kernels 一起提交给 evaluation runner。tensor 路径应尽量使用 matrix-native 输入，一次性或批量产生所有结果；dataframe 路径可以用循环或并行模拟矩阵执行。

结果层必须仍然能够单独评价：

```text
feature x label x portfolio_mode x kernel x weight_profile x cost_profile x policy_set -> combination_id
```

因此，`factor_set` 的职责是组织一批一起输入计算的 feature columns；`feature` 的职责是形成可展示、可比较的单因子结果切片。未来如果要评价一个多因子整体 signal，应作为新的 signal/model 结果轴加入，而不是让第一阶段的单 feature 切片消失。

### 2.4 Tensor 是长期主路径

tensor 不应只是 dataframe 的另一个逐组合执行后端。

长期目标是：

```text
EvaluationMatrixSpec
  -> tensor layout selection/build
  -> tensor-native matrix evaluation
  -> combination-keyed outputs
```

典型 tensor 输入：

```text
date x asset x factor
date x asset x label
date x asset x eligibility_mask
date x asset x tradability_mask
date x asset x group/sector/size/liquidity fields
```

典型 tensor 输出：

```text
combination x date x metric
combination x period x metric
combination x diagnostic rows
combination x chart series
```

dataframe 可以继续保留，但定位应是：

- reference implementation；
- 小样本调试；
- tensor correctness check；
- tensor 尚未覆盖能力的 fallback。

不要让 dataframe 的 per-combination 循环形态污染长期 artifact 合同。

### 2.5 `kernel` 替代 `engine`

使用 `kernel`，不再使用 QRP 自有合同里的 `engine`，也不使用拼错的 `kernal`。

理由：

- 项目已有 `src/quant_research_platform/kernels/`；
- 量化和数值计算中，kernel 更准确表示底层计算实现；
- `engine` 太泛，且容易和 pandas / Parquet 的第三方 `engine=` 参数混淆。

注意：只改 QRP 自有 evaluation compute backend 命名。第三方库本来就叫 `engine=` 的参数不改。

## 3. 当前问题

当前 A 股 evaluation 已有可用行为，主要在：

```text
src/quant_research_platform/markets/implementations/a_share/evaluation/evaluation_adapter.py
src/quant_research_platform/markets/implementations/a_share/evaluation/run_factor_evaluation.py
src/quant_research_platform/evaluations/
src/quant_research_platform/kernels/
src/quant_research_platform/pipelines/a_share/
```

但长期边界还不够清楚：

- public contract 仍使用 `engine`，与 `kernels/` 包命名不一致；
- adapter 可以循环多个 label 并写 parent manifest，但没有正式 evaluation run 合同；
- frontend 需要在一个界面切 label / mode / kernel，而不是面对多个散 run；
- 现有 2023-2025 evaluation baseline 把 kernel、portfolio mode 和 label 分散在目录层级里，把 feature 放在各子 run 的 `combinations.csv` 里；N7 应把这些维度收进同一个 evaluation run 合同；
- Factor Factory 需要表达 candidate 的矩阵评价结果，而不是只记录一个模糊 run 目录；
- Factor Factory 应适配 N7 的 evaluation run 合同，而不是让 N7 迁就 Factor Factory 的旧 candidate-run 心智；
- tensor 的长期优势没有在合同层表达出来；
- pipeline、market evaluation、Factor Factory、artifact display 的职责需要重新拉直。

## 4. 目标对象

建议在 A 股 evaluation 模块下引入矩阵合同：

```text
src/quant_research_platform/markets/implementations/a_share/evaluation/
  matrix_spec.py
  matrix_identity.py
  matrix_runner.py
```

建议对象：

```text
AShareEvaluationMatrixSpec
AShareEvaluationMatrixIdentity
AShareEvaluationCombination
AShareEvaluationMatrixRunRef
```

其中：

- `AShareEvaluationMatrixSpec`：调用方输入的矩阵请求；
- `AShareEvaluationMatrixIdentity`：决定两个 evaluation runs 是否属于同一个研究合同；
- `AShareEvaluationCombination`：矩阵中的一个结果切片；
- `AShareEvaluationMatrixRunRef`：供 Factor Factory、pipeline、API、frontend 引用的 evaluation run 引用。

API-visible identity 仍然必须保留：

```text
(source_alias, run_id)
```

这里的 `run_id` 指 API-visible evaluation run id；内部实现可以映射到 `EvaluationMatrixRun`。

## 5. Identity 与 Lineage

必须区分 identity 和 lineage。

Identity 回答：

```text
这两个 run 是否应该被看作同一个研究合同？
```

Lineage 回答：

```text
这个 run 实际由哪些输入文件、数据版本、代码版本、环境和参数生成？
```

Matrix identity 至少包含：

```text
market
requested date range
factor_set_id
factor_definition_version
factor columns
label set
portfolio mode set
kernel set
weight profile/config
cost profile/config
benchmark profile set config
universe policy
eligibility policy
tradability policy
neutralization/risk controls when implemented
```

`benchmark_profile_set_id` 表示一组基准配置，而不是单个基准指数。第一阶段可以使用 `none`；未来应支持默认指数集合和用户自定义基准集合，例如 `cn_default_equity_indices_v1` 或 `custom_user_benchmarks_v1`。同一个 `combination_id` 内可以按 `benchmark_id` 展开多个基准比较结果，不因为多个基准指数生成更多 `combination_id`。

Lineage 至少记录：

```text
processed factor file reference
input fingerprint
actual trading-date coverage
data version or source metadata
code version
environment summary
run timestamp
warnings and missing coverage notes
```

`run_id` 可以使用紧凑别名和稳定 hash，但完整 identity 和 lineage 必须写入 config、summary、manifest 或等价 metadata。

## 6. Canonical N7 样例

N7 canonical 私有 A 股样例使用 2023-2025：

```text
start_date = 2023-01-01
end_date   = 2025-12-31
market     = a_share
```

实现和验收分两层真实数据样例：

```text
real smoke:
  output_root = reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1
  start_date  = 2023-01-01
  end_date    = 2023-06-30

canonical:
  output_root = reports/private_real/factor_evaluation/n7_eval_2023_2025_v1
  start_date  = 2023-01-01
  end_date    = 2025-12-31
```

第一晚实现 goal 应先跑 2023H1 real smoke。完整 2023-2025 canonical regeneration 必须等 schema/unit tests、2023H1 smoke、low-memory summary comparison、Evaluation Comparison Report、dataframe/tensor parity 和前端 20 行 preview 全部通过后再跑。

初始评价集合：

```text
kernels:
  - tensor
  - dataframe

portfolio_modes:
  - long_only
  - long_short

labels:
  - forward_return_5d
  - forward_return_20d

factor_set_id:
  - price_volume_core_v1

factor columns:
  - mom_20d
  - vol_20d
  - avg_amount_20d

cost_bps:
  - 0
```

这不是 8 个或 24 个用户层 run，而是 1 个 evaluation run。内部 result combinations 至少按 feature、label、portfolio mode 和 kernel 展开：

```text
2 kernels x 2 portfolio modes x 2 labels x 3 features = 24 result combinations
```

这里的 `factor_set_id = price_volume_core_v1` 表示这 3 个 feature columns 作为同一批输入一起计算；它不表示只输出一个 factor-set-level 评价结果。

`long_short` 在 N7 中是 factor spread / grouped portfolio 诊断口径，不等于真实可融券做空策略。除非未来单独建模 borrow availability、short-sale constraints、margin、execution、slippage 和 capacity，否则不得描述为真实可交易 long-short。

生成 evaluation run 前必须 dry-run processed factor file：

- 缺少 factor column 时 fail fast；
- 缺少 label column 时 fail fast；
- 缺少 eligibility / tradability column 时 fail fast 或明确 blocking issue；
- 报告 requested date range 和 actual trading-date coverage；
- 不打印原始 CSMAR 私有数据。

## 7. Artifact Schema `0.6`

N7 应把 evaluation-set artifact schema 从 `0.5` 升到 `0.6`。

主要原因：

- public contract 从 `engine` 改成 `kernel`；
- run 语义从单组合心智升级为多结果切片 evaluation run；
- artifact 表格需要以 `combination_id` 为核心连接键；
- 前端切换 label / mode / kernel 应由 display contract 驱动。

Schema `0.6` 应使用：

```json
{
  "kernel": "tensor"
}
```

不再使用：

```json
{
  "engine": "tensor"
}
```

Schema `0.6` 不应只是给旧表继续追加维度列。推荐改成围绕 `combination_id` 的 star schema。

第一阶段 required artifacts：
```text
manifest.json
summary.json
config.json
lineage.json
report.md

dimensions/
  combinations.csv

metrics/ic.csv
metrics/ic_summary.csv
metrics/grouped_returns.csv
metrics/group_spread_summary.csv
metrics/performance_summary.csv

portfolio/returns.csv
portfolio/cumulative_returns.csv
portfolio/weights.parquet

diagnostics/tradability_diagnostics.csv

input_quality/label_availability.csv
input_quality/factor_coverage.csv
input_quality/tradability_coverage.csv

summaries/evaluation_summary.csv
```

第一阶段 optional / future extension artifacts：
```text
dimensions/features.csv
dimensions/labels.csv
dimensions/kernels.csv
dimensions/portfolio_policies.csv
dimensions/weight_profiles.csv
dimensions/cost_profiles.csv
dimensions/benchmark_profiles.csv

metrics/benchmark_returns.csv
metrics/benchmark_summary.csv

portfolio/turnover.csv
portfolio/cost_breakdown.csv
portfolio/exposure_summary.csv

diagnostics/weight_diagnostics.csv
diagnostics/neutralization_diagnostics.csv
diagnostics/risk_diagnostics.csv
diagnostics/capacity_diagnostics.csv

breakdowns/year/ic_summary.csv
breakdowns/year/group_spread_summary.csv
breakdowns/year/performance_summary.csv
```

核心规则：

- `dimensions/combinations.csv` 是所有结果维度的 source of truth；
- fact tables 默认只强制包含 `combination_id` 和本表自然主键，例如 `timestamp`、`asset`、`group`、`year`；
- 大表不要重复写入 `feature`、`label`、`kernel`、`portfolio_mode`、`factor_set_id` 等字符串维度，避免体积膨胀；
- 前端、API、下载、对比通过 `combination_id` join `dimensions/combinations.csv` 获得维度信息；
- 用户可见表、可下载项和图表必须仍通过 `manifest.display` 暴露；前端不得绕过 display contract 猜测文件名；
- `weights` 第一阶段就直接写 Parquet，不再把完整 weights CSV 作为主产物；
- 旧样例中已有但 N7 暂时无法稳定保留的结果，必须在实现计划里明确列为暂缓项并说明原因；不能静默缩小结果范围。

`dimensions/combinations.csv` 至少包含：

```text
combination_id
factor_set_id
feature
label
kernel
portfolio_mode
weight_profile_id
cost_profile_id
benchmark_profile_set_id
universe_policy_id
eligibility_policy_id
tradability_policy_id
```

其中 `weight_profile_id`、`cost_profile_id`、`benchmark_profile_set_id` 表示参数配置口径，`universe_policy_id`、`eligibility_policy_id`、`tradability_policy_id` 表示样本和交易可行性规则口径。`benchmark_profile_set_id` 是一组基准，不是单个指数；一个 combination 内未来可以按 `benchmark_id` 展开多个基准比较结果。`signal_id`、`model_id` 等只作为 future optional axes，不进入第一阶段 required columns。

第一阶段 fact table 最低列约束：

```text
metrics/ic.csv:
  combination_id, timestamp, ic, rankic, n_obs

metrics/ic_summary.csv:
  combination_id, mean_ic, std_ic, icir, positive_ic_rate,
  mean_rankic, std_rankic, rankicir, positive_rankic_rate

metrics/grouped_returns.csv:
  combination_id, timestamp, group, mean_forward_return, n_obs

metrics/group_spread_summary.csv:
  combination_id, top_group_mean_return, bottom_group_mean_return,
  top_bottom_spread_mean, top_bottom_spread_hit_rate, group_monotonicity_score

metrics/performance_summary.csv:
  combination_id, periods, final_cumulative_gross_return,
  final_cumulative_net_return, mean_net_portfolio_return, mean_turnover,
  max_drawdown, total_cost

portfolio/returns.csv:
  combination_id, timestamp, portfolio_mode, long_return, short_return,
  gross_portfolio_return, turnover, cost, net_portfolio_return,
  long_count, short_count, return_source_column

portfolio/cumulative_returns.csv:
  combination_id, timestamp, gross_return, net_return,
  cost, cumulative_gross_return, cumulative_net_return, cumulative_cost,
  high_water_mark, drawdown

portfolio/weights.parquet:
  combination_id, timestamp, asset, target_weight

diagnostics/tradability_diagnostics.csv:
  combination_id, timestamp, method, direction, universe_count,
  eligible_count, can_enter_count, can_exit_count, entry_blocked_count,
  exit_blocked_count, eligible_rate, can_enter_rate, can_exit_rate,
  entry_blocked_rate, exit_blocked_rate, fallback_reason
```

未来实现基准比较时，建议补充：
```text
dimensions/benchmark_profiles.csv:
  benchmark_profile_set_id, benchmark_id, benchmark_name, benchmark_type,
  return_column, alignment_policy

metrics/benchmark_returns.csv:
  combination_id, timestamp, benchmark_id, benchmark_return,
  portfolio_excess_return

metrics/benchmark_summary.csv:
  combination_id, benchmark_id, mean_excess_return, tracking_error,
  information_ratio, beta, alpha
```

`portfolio/weights.parquet` 中的 `target_weight` 表示研究组合目标权重，不是券商真实持仓，也不是成交回放后的实际权重。未来如果引入更真实的交易约束，可以新增 `tradable_weight` 或 `realized_weight`，用于表达涨跌停、停牌、买不进、卖不掉或成交模拟后的权重。

兼容性策略不是保留旧 loader 宽容路径，而是建立清晰迁移边界：

推荐策略：

- 新 writer 只写 schema `0.6`；
- loader 拒绝旧 `0.5`，除非单独设计 migration tool；
- 旧本地样例只作为人工验收参考或一次性测试输入，不进入正式 compare contract；
- `compare_evaluation_sets` 和 Evaluation Comparison Report 只比较 schema `0.6` artifacts；
- 文档明确说明 N7 有意打破本地 schema `0.5` 兼容性。

## 8. `manifest.display` 与前端展示

前端展示应围绕一个 evaluation run 页面：

```text
Run: as_20230101_20251231_pvcore1_eval_v1

Controls:
  feature: mom_20d / vol_20d / avg_amount_20d
  label: forward_return_5d / forward_return_20d
  portfolio_mode: long_only / long_short
  kernel: tensor / dataframe
  factor_set: price_volume_core_v1

Current selection:
  combination_id = ...

Display:
  summary cards
  IC / RankIC charts
  portfolio returns
  turnover
  tradability diagnostics
  CSV preview
  downloads
```

前端不重新计算 IC、returns、turnover、weights、tradability 或 diagnostics。

前端只做：

- 读取 `manifest.display`；
- 读取 `dimensions/combinations.csv`；
- 根据用户选择找到 `combination_id`；
- 只展示当前 `combination_id` 对应的表格和图表切片；
- 支持同一个 evaluation run 内的组合对比。

`manifest.display` 在 schema `0.6` 中需要表达更多机器可读信息：

```text
display.schema_version = "0.6"
display.dimension_source = "dimensions/combinations.csv"
display.default_selection = {feature, label, portfolio_mode, kernel}
display.tables[*].path
display.tables[*].format = csv | parquet
display.tables[*].join_key = combination_id
display.tables[*].supports_combination_filter
display.tables[*].preview_rows = 20
display.charts[*].table
display.downloads.items[*].item
display.downloads.presets[*].items
```

前端可以基于 `dimension_source` 生成 selectors，但 selector 的可选值必须来自 `dimensions/combinations.csv`，不能来自硬编码 factor/label/kernel 列表。

前端不需要做视觉重设计，但必须完整展示 schema `0.6` 暴露的用户可见结果。Parquet 表也必须通过 API 提供 20 行 preview；完整下载仍通过 `manifest.display.downloads` 控制。

## 9. Tensor 与 DataFrame 执行语义

执行计划应和用户展示语义分开。

用户看到：

```text
一个 evaluation run
```

执行层可以有不同策略：

```text
tensor:
  matrix-native task
  尽量一次性或批量计算完整 matrix

dataframe:
  reference/fallback path
  可以按 combination 循环或并行执行
  输出仍必须写入同一套 evaluation artifact schema
```

因此，dataframe 可以在实现早期生成同样的 schema `0.6` 产物，但它不应决定长期结构。

长期验收重点是：

- tensor 能接收 matrix input；
- tensor 能直接产生 combination-keyed outputs；
- dataframe 和 tensor 在共同覆盖的指标上可做 correctness comparison；
- API/frontend 不需要知道内部是 tensor 一次性算完，还是 dataframe 分组合并。

## 10. 模块边界

### 10.1 `markets/implementations/a_share`

负责 A 股语义：

- date range defaults；
- A 股 factor / label naming；
- universe / eligibility / tradability policy；
- A 股 matrix spec validation；
- processed factor table 到 tensor/dataframe input 的适配；
- matrix spec 到 generic evaluation contract 的桥接。

### 10.2 `evaluations`

负责通用评价合同：

- `EvaluationSpec` 或后续 `EvaluationMatrixSpec`；
- `run_evaluation(...)` 或后续 matrix entrypoint；
- metric / portfolio / diagnostics 计算合同；
- kernel-independent orchestration；
- result / payload objects。

### 10.3 `kernels`

负责数值计算实现：

- tensor kernel；
- dataframe reference kernel；
- kernel registry；
- kernel protocol；
- matrix layout selection；
- tensor/dataframe shared output contract。

### 10.4 `pipelines/a_share`

负责 workflow 编排：

- snapshots；
- checkpoint state；
- `start_from`；
- step planning；
- 到 evaluation 阶段时调用 A 股 matrix evaluation contract。

它不负责定义 canonical sample matrix，不重新暴露 `execution_mode` 或 `rebuild_*`。

### 10.5 `research/factor_factory`

负责候选生成和批量研究：

- factor candidate definitions；
- parameter grids；
- candidate-to-matrix expansion；
- 保存 evaluation run refs；
- 通过 `candidate_id + combination_id` 汇总候选表现。

Factor Factory 是 N7 的上游调用方和下游汇总视图，不是 N7 核心合同的约束来源。N7 应先定义长期正确的 evaluation 合同；Factor Factory 再把 candidates 映射到 matrix spec，并从 N7 artifacts 派生 candidate summary。

如果 Factor Factory 持久化输出新增 evaluation run refs 或 combination summaries，应单独 bump Factor Factory artifact schema，例如 `0.1 -> 0.2`，不能和 evaluation-set schema `0.6` 混在一起。N7 第一阶段不强制 bump Factor Factory schema，除非实现确实改变了 Factor Factory 持久化 artifact。

## 11. API 与 CLI 影响

如果 N7 执行破坏性重命名：

- API payload 使用 `kernel`，不接受旧 `engine`；
- frontend shared API types 使用 `kernel`；
- upload / pipeline / Factor Factory creation surface 使用 `kernel`；
- A 股 CLI 使用 `--kernel`；
- summary / config / manifest 使用 `kernel`；
- docs 和 tests 全面改名。

不建议保留 `engine` compatibility layer。旧本地 run 直接重生成。

## 12. 实现切片

推荐顺序：

1. 定义 schema `0.6` evaluation artifact contract：`combination_id`、核心表、`manifest.display`。
2. 把 QRP 自有 evaluation compute backend 从 `engine` 改为 `kernel`。
3. 引入 A 股 `AShareEvaluationMatrixSpec`、matrix identity、combination objects，并明确 `factor_set` 是计算输入批次、`feature` 是结果轴。
4. 先让 dataframe reference path 能生成 schema `0.6` matrix artifact，作为合同和对照基线。
5. 重构 tensor path，使其接收 matrix-native 输入并输出 combination-keyed 结果。
6. 用 dataframe/tensor 对照测试验证共同覆盖结果，采用分层容差策略；如因浮点归约顺序需要放宽，必须在测试和文档中写明。
7. 更新 pipeline evaluation step，使其调用 matrix evaluation contract。
8. 更新 Factor Factory，让 candidate 适配 N7 evaluation run ref，并用 `candidate_id + combination_id` 汇总结果。
9. 如 Factor Factory 持久化结构变化，单独 bump Factor Factory schema 并更新 API/frontend tests。
10. 更新 frontend/API，使 schema `0.6` 的所有 display tables 都能完整展示或提供 20 行 preview，包含 Parquet weights。
11. 将面向网页/API 的 `compare_evaluation_sets` 改成默认低内存 summary comparison，只读 summary、manifest、combinations 和必要 summary metrics，不读取完整 weights 或大时序表；它应继续支持大样本快速比较，并作为网页 comparison cards / tables / charts 的数据源。
12. 增加低内存 Evaluation Comparison Report；它比 `compare_evaluation_sets` 更详细，用于研究比较、验收和回归，只输出表格、JSON 和 Markdown 报告，不进入主网页展示；它比较同一 evaluation run 内部或两个 schema `0.6` evaluation runs 之间的 combinations 和 kernel parity，不读取完整 weights 或大时序表。
13. 生成 2023H1 real smoke，并通过 low-memory summary comparison、Evaluation Comparison Report 和 parity 验收；2023H1 smoke 不要求和旧样例做结果一致性比较。
14. 通过后再生成或准备 2023-2025 canonical sample；三年样例生成后，再在实现验收阶段比较新实现输出是否一致。
15. 更新 docs、code map、`PROJECT_STATUS.md`。

## 13. 测试与验收

先跑窄测试：

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m pytest -q tests/kernels tests/evaluations tests/artifacts
```

实现过程中应重点覆盖：

- kernel registry；
- schema `0.6` writer / loader；
- `manifest.display` table / chart / download contract；
- `combination_id` validation；
- matrix spec validation；
- dataframe reference matrix output；
- tensor matrix output；
- dataframe/tensor parity，按分层容差策略验证；
- A 股 evaluation adapter；
- pipeline matrix evaluation step；
- Factor Factory evaluation run refs；
- frontend combination switching；
- all display tables 20-row preview, including Parquet `weights`；
- low-memory `compare_evaluation_sets` for web/API summary comparison and charts；
- low-memory Evaluation Comparison Report that outputs files only and avoids reading full weights and large time-series tables。

数值对照策略：

```text
structure:
  exact match
  covers combination_id set, primary keys, dates, assets, groups, string fields,
  integer counts, null placement, and table presence.

same_kernel_regeneration:
  rtol = 1e-12
  atol = 1e-12
  用于 tensor old/new 或 dataframe old/new 的确定性回归。

dataframe_tensor_common_metrics:
  rtol = 1e-9
  atol = 1e-10
  用于 dataframe 与 tensor 共同覆盖的 IC、RankIC、group returns、
  summary metrics、turnover、cost 等结果。

cumulative_path_metrics:
  rtol = 1e-8
  atol = 1e-10
  用于 cumulative returns、drawdown、长链路 performance 派生指标。
```

不要使用单一 `abs_diff <= 1e-9` 作为所有表的规则。接近 0 的值需要 `atol`，正常量级和累计指标需要 `rtol`，结构和离散字段必须精确一致。

前端检查：

```powershell
npm test -- --run --maxWorkers=1
npm run lint
npm run build
```

Playwright 只作为浏览器 smoke，不替代 unit / contract tests。Playwright 输出必须使用项目 wrapper，落到 `.playwright-cli/`。

canonical private sample regeneration 不是普通 unit test。它应作为合同测试通过后的本地验收步骤，且不得提交私有生成物。

真实数据执行顺序：

1. 先跑 synthetic fixture tests，验证 schema、writer/loader/display、selector 和 comparator。
2. 再跑 2023H1 A 股真实数据 smoke，覆盖 24 个 result combinations；只验证 schema、生成完整性、schema `0.6` 内部 compare/report、dataframe/tensor parity 和前端/API preview，不要求和旧样例做结果一致性比较。
3. 2023H1 smoke 通过后，才跑完整 2023-2025 canonical regeneration；三年样例生成后，再在实现验收阶段比较新实现输出是否一致。

## 14. 当前 Git 工作区处理

当前 dirty docs 仍属于文档同步 checkpoint，不是 N7 代码实现：

```text
PROJECT_STATUS.md
docs/architecture/README.md
docs/architecture/next-stage-roadmap.md
docs/architecture/next-stage-worktree-plan.md
docs/superpowers/specs/2026-05-25-n7-a-share-run-identity-design.md
```

建议先做 docs-only commit：

```text
docs: sync next-stage N7 evaluation run design
```

之后再创建 N7 实现分支：

```text
codex/n7-a-share-run-identity
```

## 15. 非目标

N7 不做：

- broker / exchange / vendor API / web API；
- paper trading / live trading / order routing；
- 发布或暴露 CSMAR 私有数据；
- repo-level `apps/api`；
- 完整 feature store；
- production scheduling；
- 真实 long-short 可交易性承诺；
- 旧 schema `0.5` 的宽松兼容。

## 16. 编码前待确认

当前已确认的方向：

- 一个 API-visible run 是 `EvaluationMatrixRun`；
- 一个 evaluation run 内部包含多个 `combination_id`；
- 计算层支持多个 features / labels / portfolio modes / kernels 一起批量或并行计算；
- 评价层按 `feature x label x portfolio_mode x kernel x cost_profile` 单独形成结果切片；
- `factor_set` 是计算输入批次，`feature` 是第一阶段的结果轴；
- 前端在一个页面切换 feature / label / portfolio mode / kernel；
- 前端不需要视觉重设计，但必须完整展示 schema `0.6` 暴露的用户可见结果；
- 所有 display tables，包括 Parquet `weights`，都必须支持 20 行 preview；
- tensor 是长期主路径，目标是 matrix-native；
- dataframe 是 reference / fallback / debugging；
- N7 新 evaluation-set artifact 使用 schema `0.6`；
- schema `0.6` 使用围绕 `dimensions/combinations.csv` 的 star schema；
- 大表通过 `combination_id` 连接维度，不重复塞入所有字符串维度；
- `weights` 第一阶段就直接写 Parquet；
- canonical root 固定为 `reports/private_real/factor_evaluation/n7_eval_2023_2025_v1`；
- 第一晚真实数据验收先跑 `reports/private_real/factor_evaluation/n7_eval_2023h1_smoke_v1`；
- 完整 2023-2025 regeneration 要等 2023H1 real smoke、low-memory summary comparison、Evaluation Comparison Report、parity 和前端 preview 通过后再跑；
- QRP 自有合同从 `engine` 改为 `kernel`；
- Factor Factory 适配 N7，不让 N7 迁就旧 Factor Factory；
- Factor Factory 第一阶段只做适配，不强制 bump schema；只有持久化 artifact 结构变化时才单独 bump；
- `kernel` 是 evaluation run 内的维度，tensor/dataframe 不再拆成用户可见散 run；
- `compare_evaluation_sets` 应改成默认低内存的网页/API summary comparison，并支持大样本快速比较和前端画图；
- 需要新增更详细的低内存 Evaluation Comparison Report，只输出表格/JSON/Markdown，不进主网页展示，用于比较 schema `0.6` combinations 和 kernel parity，不能用会读完整大 artifact 的 comparison 路径验收大样本；
- dataframe/tensor 对照是 N7 验收门槛，使用分层 `rtol/atol` 容差策略。

仍需在实现计划里细化，但不改变上述方向：

- schema `0.6` 每张表的完整列约束和 parquet/csv 写入策略；
- tensor matrix layout 的最小输入/输出合同；
- dataframe reference path 如何合并为同一 matrix artifact；
- Factor Factory schema 是否因持久化结构变化而单独 bump；
- 2023-2025 canonical sample 的实现验收对比报告格式。

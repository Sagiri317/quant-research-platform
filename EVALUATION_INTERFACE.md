# Evaluation 接口说明

## 当前合同（schema 0.7）

新写出的 evaluation-set artifact schema 为 `0.7`。一个
API-visible evaluation run 可以包含多个 result combinations；组合维度通过
`dimensions/combinations.csv` 表达，diagnostics policy 维度通过 `dimensions/diagnostic_policies.csv` 表达，组合表核心 columns 为：

```text
combination_id, feature, label, kernel, portfolio_mode
```

run-level constant 配置，包括成本、权重方法、benchmark、universe、
eligibility、tradability、生成环境和 lineage 摘要，写入 `run_context.json`，不进入
`dimensions/combinations.csv`。实际结果值仍写入 schema registry 声明的 fact tables，例如
`metrics/evaluation_metrics_summary.csv`、`portfolio/returns.csv`、
`input_quality/coverage_summary.csv`、`diagnostics/diagnostic_summary.csv`，以及
按请求 materialized 的 `portfolio/weights.parquet`。

公开命名使用 `kernel` 和 `return_source_column`。旧的 external `engine`
compatibility、`pnl_return_column`、`portfolio/pnl.csv` 和 `portfolio/positions.csv` 不属于 schema `0.7` 合同；累计收益图由 API/display chart payload 派生，不再作为落盘 artifact；`portfolio/weights.parquet` 是 conditional / materialized-on-request 输出，不是 required table。

A 股本地 CLI 第一阶段只使用显式 features，且多值参数写在同一行：

```powershell
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation `
  --processed-root data/processed_private/csmar `
  --report-root reports/private_real/factor_evaluation/schema_07_dev_smoke `
  --start-year 2023 `
  --end-year 2025 `
  --start-date 2023-01-01 `
  --end-date 2023-06-30 `
  --features mom_20d vol_20d amihud_20d `
  --labels forward_return_5d forward_return_20d `
  --kernels dataframe tensor `
  --portfolio-modes long_short long_only `
  --dry-run
```

`--input-file` 是后端/本地 CLI-only override，不暴露到 API request models、
frontend forms、Pipeline Builder payloads 或 saved snapshots。`--dry-run`
只检查输入文件、日期覆盖、required columns、组合展开和输出目录，不写 artifact。

本文是 `quant-research-platform` 内部 market module 和外部本地研究脚本接入 evaluation layer 的接口文档。它只定义 evaluation layer 需要什么输入、如何调用、会写出什么 evaluation artifacts；不绑定具体市场，也不处理真实数据许可、下载、凭据、仿真、回测系统或交易。

Evaluation layer 的职责是评价已经构造好的 factor / prediction panel，并写出标准 evaluation-set artifacts。它不下载数据、不清洗 vendor/raw data、不读取 credentials、不训练 ML model，也不判断真实市场 alpha。`EvaluationSpec` 是 evaluation artifact contract，不是 `BacktestSpec`、`SimulationSpec` 或 trading contract。

## 1. 统一 Contract

当前 public evaluation contract：

| 层级 | 名称 | 作用 |
|---|---|---|
| 输入 contract | `EvaluationSpec` | 一次 evaluation set 的完整请求，可以包含多个 feature 和多个 label，以及一个 run-level weight config 和一个 run-level cost config |
| 执行入口 | `run_evaluation(spec)` | 运行整个 evaluation set，并写出 artifacts |
| 输出结果 | `EvaluationSetResult` | 返回内存表、`combinations` 表和 artifact paths |

三类输入形态只是 input adapters，不是三套评价逻辑：

| Input adapter | 适用场景 | 入口 | 标准化结果 |
|---|---|---|---|
| DataFrame / Python | market module 内直接调用 | 手动构造 `EvaluationSpec` | `EvaluationSpec` |
| CSV upload / website | 本地网页手动上传 CSV | FastAPI `POST /api/runs` | `EvaluationSpec` |
| Parquet / Python helper | 上游 pipeline 已生成本地 panel 文件 | `make_parquet_evaluation_spec()` | `EvaluationSpec` |
| Parquet / CLI | 命令行读取本地 Parquet | `evaluate-parquet` CLI | `EvaluationSpec` |

```text
input transport -> input adapter -> EvaluationSpec
                -> run_evaluation
                -> schema 0.7 evaluation-set artifacts
```

## 2. Long Panel Schema

Panel 必须是 `timestamp x asset` long format。每行代表一个资产在一个时间点的 signal、label 和辅助列。

| 角色 | 示例列名 | 要求 |
|---|---|---|
| timestamp | `timestamp`、`trade_date`、`bar_time` | 可解析为 datetime |
| asset | `asset`、`symbol`、`stock_code` | 非空，会转为字符串 |
| feature / signal | `factor_a`、`mom_20d`、`prediction` | 至少一个，必须是 numeric |
| label | `forward_return_5d`、`target_return` | 至少一个，必须是 numeric |
| return source | `forward_return_1d` | 用于累计收益曲线；可用 `return_source_column` 指定 |

硬约束：

- `timestamp + asset` 必须唯一。
- feature、label、return source 必须是数值列。
- feature columns 和 label columns 不能重叠。
- asset code 不应因保存为数字而丢失前导零。
- 缺列、重复 key、非数值列都会 fail fast。

常用可选列：

| 角色 | 示例列名 | 用途 |
|---|---|---|
| segment | `industry`、`market_type`、`universe` | 分组诊断或 segment-aware portfolio path |
| eligibility | `eligible` | 是否进入可交易 universe |
| can enter | `can_enter` | 当前时点是否允许新开仓 |
| can exit | `can_exit` | 当前时点是否允许正常退出已有仓位 |
| risk | `risk_20d`、`volatility_60d` | 风险缩放权重 |
| neutralization group | `industry`、`sector` | group de-mean neutralization |
| exposure | `size`、`beta`、`volatility` | cross-sectional OLS neutralization |

## 3. Label 与 Return Source

`label_columns` 和 `return_source_column` 是两个不同角色。

`label_columns` 用于 label-based diagnostics：

- IC；
- RankIC；
- grouped returns；
- group spread diagnostics。

`return_source_column` 用于 evaluation portfolio-return path / diagnostic return path：

- `portfolio/returns.csv`；
- evaluation cumulative return diagnostics；
- `metrics/evaluation_metrics_summary.csv`；
- year / quarter / month breakdown 中的 performance metrics。

默认规则：

- 如果显式传入 `return_source_column`，evaluation layer 使用该列计算 evaluation portfolio returns 和 cumulative return diagnostics。
- 如果未显式传入，evaluation layer 默认查找 `forward_return_1d`。
- 如果两者都没有，evaluation layer 拒绝运行。
- 5d / 20d / 60d 等 longer-horizon label 只作为 diagnostics，不自动当作日频 evaluation portfolio-return path，更不构成真实 backtest、simulation 或 trading return path。

## 4. Python API

推荐从 package public exports 导入：

```python
from quant_research_platform import EvaluationSpec, WeightConfig, run_evaluation

spec = EvaluationSpec(
    panel=panel,
    timestamp_column="timestamp",
    asset_column="asset",
    feature_columns=("factor_a", "factor_b"),
    label_columns=("forward_return_5d", "forward_return_20d"),
    return_source_column="forward_return_1d",
    metadata={
        "run_id": "factor_eval_set",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "a_share",
        "data_stage": "local_offline_panel",
        "uses_external_data": True,
    },
    weight_configs=(WeightConfig(),),
    output_dir="reports/private_real/factor_evaluation/factor_eval_set",
    n_groups=5,
    time_breakdowns=("year", "quarter", "month"),
    kernel="dataframe",
)

result = run_evaluation(spec)
```

`metadata` 必须包含：

| 字段 | 含义 |
|---|---|
| `run_id` | 稳定 evaluation id |
| `project` | 产生该 evaluation set 的项目或模块 |
| `kind` | `factor`、`prediction` 或其他研究对象类型 |
| `domain` | 资产或研究域，例如 `a_share` |
| `data_stage` | 数据阶段，例如 `synthetic`、`local_offline_panel` |
| `uses_external_data` | 是否使用外部或真实数据 |

当 `kind="prediction"` 时，N6 MVS 仍复用 schema `0.7` /
`evaluation_set`，但必须额外提供 private-safe handoff metadata：
`prediction_handoff_version`、`prediction_columns`、`label_columns`、
`prediction_time_semantics`、`source_ref`、`model_ref`、`training_scope` 和
`prediction_boundary`。`source_ref` 和 `model_ref` 必须是 portable alias /
opaque id，不得写入 private path、机器绝对路径、row-level private data 或股票样本。
`training_scope` 固定为 `external_or_upstream_not_run_by_qrp`；boundary 必须明确
prediction evidence 不是 trading、risk、capacity、execution simulation、paper
trading 或 live trading evidence。该 MVS 不新增 artifact type、required
artifact file、schema version、`manifest.display` entry、API field 或 frontend
display。

`output_dir` 应使用 repo-relative path，不写机器绝对路径。

`run_id` 是单个 evaluation set 内的稳定 ID。通过本地 API 暴露多个 run root 时，前端、下载、comparison 和缓存使用 `(source_alias, run_id)` 作为完整身份；不要只按 `run_id` 判断同一个 run。

## 5. Parquet Helper

如果上游已经生成本地 Parquet，可用 helper 读取必要列并构造 `EvaluationSpec`。Evaluation layer 只读取显式传入的本地文件，不生成这些 Parquet。

```python
from quant_research_platform import make_parquet_evaluation_spec, run_evaluation

spec = make_parquet_evaluation_spec(
    parquet_path="data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet",
    timestamp_column="trade_date",
    asset_column="stock_code",
    feature_columns=("rev_20d", "rev_60d"),
    label_columns=("forward_return_20d",),
    return_source_column="forward_return_1d",
    metadata={
        "run_id": "a_share_reversal_smoke",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "a_share",
        "data_stage": "local_offline_panel",
        "uses_external_data": True,
    },
    output_dir="reports/private_real/factor_evaluation/a_share_reversal_smoke",
    segment_columns=("market_type",),
    kernel="dataframe",
)

result = run_evaluation(spec)
```

多个 `parquet_path` 的规则：

- 每个文件都包含本次 evaluation 所需列时，按行 concat。
- 文件共享 `timestamp/asset` key 且非 key required columns 互补时，按列 join。
- required column 重复导致语义不清时，evaluation layer 报错，不猜测。

## 6. A 股专用评估入口

A 股 market module 已有专用 adapter，真实 CSMAR processed factor run 应优先走 A 股专用入口：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation `
  --processed-root data/processed_private/csmar `
  --report-root reports/private_real/factor_evaluation/a_share_2024_dataframe_long_only `
  --start-year 2023 `
  --end-year 2025 `
  --start-date 2024-01-01 `
  --end-date 2024-12-31 `
  --features mom_20d vol_20d amihud_20d `
  --labels forward_return_5d forward_return_20d `
  --return-source-column forward_return_1d `
  --portfolio-modes long_only `
  --kernels dataframe
```

这个入口的语义是：

- `--start-year/--end-year` 用来定位 `price_volume_factors_<start>_<end>.parquet`；若精确文件不存在，会选择覆盖该年份范围的更宽 full factor 文件。
- `--start-date/--end-date` 用来对 `trade_date` 做评估阶段行过滤。
- 一个 API-visible evaluation run 内包含所有展开后的 `combination_id`；输出直接写入 `--report-root`。
- 这里会保留 A 股 adapter metadata、label availability summary、portfolio mode、kernel、date filter 和 source file basename 信息。

`evaluate-parquet` 是通用 generic evaluation adapter，适合外部本地 Parquet、临时输入验收或非 A 股市场模块；验证 A 股全流程时不要用它替代 A 股专用入口。

## 7. CLI evaluate-parquet

`evaluate-parquet` 是本地命令行入口：

先 dry-run：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.interfaces.cli evaluate-parquet `
  --dry-run `
  --parquet-path data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet `
  --timestamp-column trade_date `
  --asset-column stock_code `
  --feature rev_20d `
  --feature rev_60d `
  --label forward_return_20d `
  --return-source-column forward_return_1d `
  --segment-column market_type `
  --eligibility-column eligible `
  --can-enter-column can_enter `
  --can-exit-column can_exit `
  --evaluation-id a_share_reversal_smoke `
  --project quant-research-platform `
  --domain a_share `
  --data-stage local_offline_panel `
  --n-groups 5 `
  --cost-bps 0 `
  --cost-bps 5 `
  --output reports/private_real/factor_evaluation/a_share_reversal_smoke
```

去掉 `--dry-run` 后正式生成 artifacts。

## 8. 输出 Artifacts

一个 evaluation set 输出一个目录：

```text
manifest.json
summary.json
run_context.json
dimensions/combinations.csv
dimensions/diagnostic_policies.csv
report.md
metrics/ic.csv
metrics/grouped_returns.csv
metrics/evaluation_metrics_summary.csv
portfolio/returns.csv
input_quality/coverage_summary.csv
diagnostics/diagnostic_summary.csv
diagnostics/diagnostic_metrics.csv
```

Conditional / materialized-on-request 输出包括 `portfolio/weights.parquet`；
只有请求并生成后才通过 `manifest.display` 暴露为可预览或可下载表。

N5 diagnostics MVS 仍复用现有 schema `0.7` diagnostics 表，不新增 artifact
table。Return path risk proxy 写入 `diagnostic_metrics` 和按需的
`portfolio_risk_summary`；flat bps cost sensitivity grid 写入按需的
`scenario_metrics`，`diagnostic_metrics` 只保留 sensitivity span 摘要；liquidity /
capacity 和 market-cap / size 只使用 evaluation input 中已安全保留的 aggregate
optional context，写入 `diagnostic_metrics` 和按需的
`exposure_capacity_summary`。这些输出全部是 post-run descriptive proxy，不是
真实风控、Barra exposure、真实容量模型、交易模拟或投资建议。缺少 weights /
positions / optional context 时应写出 `partial` / `unavailable` 和
`missing_inputs`，不得为 diagnostics 强行物化大 `weights` 表。

启用 `time_breakdowns` 后，会按实际配置出现 `breakdowns/year/`、`breakdowns/quarter/` 或 `breakdowns/month/`。启用 year / month breakdown 时，`breakdowns/year/ic.csv` 和 `breakdowns/month/ic.csv` 会分别作为 `ic_breakdown_by_year` 和 `ic_breakdown_by_month` 写出 IC / RankIC breakdown。整体 time breakdown 是从完整 overall run path 中按 period 切片统计，不重新运行年度 / 季度 / 月度回测，不清仓，也不重新生成 weights 或 portfolio returns path。

`manifest.json` 和 `summary.json` 包含 `artifact_schema_version: "0.7"` 和 `artifact_type: "evaluation_set"`。Loader 会拒绝缺失或不支持 schema version 的 artifacts。`manifest.display` 同步声明前端可用的表格、下载项、下载 preset 和图表。

`run_context.json` 是 schema `0.7` 的 portable metadata：writer、direct download
和 selected ZIP packaging 都必须使用 display/download 合同允许的内容，不暴露 private data path 或本机绝对路径；portable artifacts 不再写出单独的 `config.json` 或 `lineage.json`。

## 9. Evaluation Layer 不负责的边界

Evaluation layer 不负责：

- 下载真实数据；
- 连接外部 API；
- 创建、读取或保存 credentials；
- 生成 vendor/raw/interim/processed data；
- 训练 ML model；
- 维护 market module 私有数据目录；
- 判断数据许可或公开边界；
- 发布 GitHub、部署服务、登录系统或数据库；
- 定义 `BacktestSpec`、`SimulationSpec`、订单生命周期、成交模型、滑点模型或撮合语义；
- paper trading、live trading 或 investment advice。

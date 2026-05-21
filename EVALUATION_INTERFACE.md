# Evaluation 接口说明

本文是 `quant-research-platform` 内部 market module 和外部本地研究脚本接入 core evaluation 的接口文档。它只定义 core 需要什么输入、如何调用、会写出什么结果；不绑定具体市场，也不处理真实数据许可、下载、凭据或交易。

Core 的职责是评价已经构造好的 factor / prediction panel，并写出标准 evaluation-set artifacts。Core 不下载数据、不清洗 vendor/raw data、不读取 credentials、不训练 ML model，也不判断真实市场 alpha。

## 1. 统一 Contract

当前 public evaluation contract：

| 层级 | 名称 | 作用 |
|---|---|---|
| 输入 contract | `EvaluationSpec` | 一次 evaluation set 的完整请求，可以包含多个 feature、多个 label、多个 weight config 和多个 cost config |
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
                -> schema 0.5 evaluation-set artifacts
```

## 2. Long Panel Schema

Panel 必须是 `timestamp x asset` long format。每行代表一个资产在一个时间点的 signal、label 和辅助列。

| 角色 | 示例列名 | 要求 |
|---|---|---|
| timestamp | `timestamp`、`trade_date`、`bar_time` | 可解析为 datetime |
| asset | `asset`、`symbol`、`stock_code` | 非空，会转为字符串 |
| feature / signal | `factor_a`、`mom_20d`、`prediction` | 至少一个，必须是 numeric |
| label | `forward_return_5d`、`target_return` | 至少一个，必须是 numeric |
| PnL return | `forward_return_1d` | 用于真实回测收益曲线；可用 `pnl_return_column` 指定 |

硬约束：

- `timestamp + asset` 必须唯一。
- feature、label、PnL return 必须是数值列。
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

## 3. Label 与 PnL Return

`label_columns` 和 `pnl_return_column` 是两个不同角色。

`label_columns` 用于 label-based diagnostics：

- IC；
- RankIC；
- grouped returns；
- group spread diagnostics。

`pnl_return_column` 用于真实回测收益路径：

- `portfolio/returns.csv`；
- `portfolio/pnl.csv`；
- `metrics/performance_summary.csv`；
- year / quarter / month breakdown 中的 performance metrics。

默认规则：

- 如果显式传入 `pnl_return_column`，core 使用该列计算 PnL。
- 如果未显式传入，core 默认查找 `forward_return_1d`。
- 如果两者都没有，core 拒绝运行。
- 5d / 20d / 60d 等 longer-horizon label 只作为 diagnostics，不自动当作真实日频 PnL 曲线。

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
    pnl_return_column="forward_return_1d",
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
    engine="dataframe",
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

`output_dir` 应使用 repo-relative path，不写机器绝对路径。

`run_id` 是单个 evaluation set 内的稳定 ID。通过本地 API 暴露多个 run root 时，前端、下载、comparison 和缓存使用 `(source_alias, run_id)` 作为完整身份；不要只按 `run_id` 判断同一个 run。

## 5. Parquet Helper

如果上游已经生成本地 Parquet，可用 helper 读取必要列并构造 `EvaluationSpec`。Core 只读取显式传入的本地文件，不生成这些 Parquet。

```python
from quant_research_platform import make_parquet_evaluation_spec, run_evaluation

spec = make_parquet_evaluation_spec(
    parquet_path="data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet",
    timestamp_column="trade_date",
    asset_column="stock_code",
    feature_columns=("rev_20d", "rev_60d"),
    label_columns=("forward_return_20d",),
    pnl_return_column="forward_return_1d",
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
    engine="dataframe",
)

result = run_evaluation(spec)
```

多个 `parquet_path` 的规则：

- 每个文件都包含本次 evaluation 所需列时，按行 concat。
- 文件共享 `timestamp/asset` key 且非 key required columns 互补时，按列 join。
- required column 重复导致语义不清时，core 报错，不猜测。

## 6. CLI evaluate-parquet

`evaluate-parquet` 是本地命令行入口。

先 dry-run：

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.cli evaluate-parquet `
  --dry-run `
  --parquet-path data/processed_private/csmar/price_volume_factors_full/price_volume_factors_2014_2025.parquet `
  --timestamp-column trade_date `
  --asset-column stock_code `
  --feature rev_20d `
  --feature rev_60d `
  --label forward_return_20d `
  --pnl-return-column forward_return_1d `
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

## 7. 输出 Artifacts

一个 evaluation set 输出一个目录：

```text
manifest.json
config.json
summary.json
combinations.csv
report.md
metrics/ic.csv
metrics/ic_summary.csv
metrics/grouped_returns.csv
metrics/group_spread_summary.csv
metrics/performance_summary.csv
portfolio/positions.csv
portfolio/turnover.csv
portfolio/returns.csv
portfolio/pnl.csv
portfolio/cost_breakdown.csv
portfolio/exposure_summary.csv
diagnostics/weight_diagnostics.csv
diagnostics/neutralization_diagnostics.csv
```

启用 `time_breakdowns` 后，会按实际配置出现 `metrics_by_year/`、`metrics_by_quarter/` 或 `metrics_by_month/`。这些 time breakdown 是从完整 overall run path 中按 period 切片统计，不重新运行年度 / 季度 / 月度回测，不清仓，也不重新生成 weights、positions 或 portfolio returns path。

`manifest.json` 和 `summary.json` 包含 `artifact_schema_version: "0.5"` 和 `artifact_type: "evaluation_set"`。Loader 会拒绝缺失或不支持 schema version 的 artifacts。`manifest.display` 同步声明前端可用的表格、下载项、下载 preset 和图表。

## 8. Core 不负责的边界

Core 不负责：

- 下载真实数据；
- 连接外部 API；
- 创建、读取或保存 credentials；
- 生成 vendor/raw/interim/processed data；
- 训练 ML model；
- 维护 market module 私有数据目录；
- 判断数据许可或公开边界；
- 发布 GitHub、部署服务、登录系统或数据库；
- paper trading、live trading 或 investment advice。

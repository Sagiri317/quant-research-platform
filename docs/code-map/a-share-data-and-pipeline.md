# A 股数据与 Pipeline 地图

本文按当前代码架构记录 A 股 market module、processed-existing runner、Factor Factory 和 validation/runtime 的真实接口。旧的 raw/interim、factor、validation、A 股接口说明已合并到本页。

## 范围

A 股是当前唯一 enabled market module。它使用用户手工导出的本地 CSMAR 文件，不登录 CSMAR，不自动下载，不连接外部 vendor API、券商 API、交易所 API 或任何交易系统。

当前已实现：

- CSMAR raw -> interim。
- interim -> full research panel。
- full panel -> slim factor and label table。
- processed factor table -> A 股 evaluation adapter。
- Pipeline Builder snapshot。
- `execution_mode=processed_existing` 轻量 runner。
- Factor Factory 第一版 registry / batch runner / artifacts / browser API。
- validation 和 storage space gate。

当前未实现为 Phase 1 功能：

- raw/interim/panel/factors 全链路的一键 build runner。
- crypto 或其他市场真实实现。
- 任意 Python factor execution。
- ML training pipeline、实盘、模拟盘或部署服务。

## 数据流

```text
data/raw_private/csmar/
-> csmar.build_interim
-> data/interim_private/csmar/
-> datasets.build_panel
-> data/processed_private/csmar/panel_full/research_panel_<start>_<end>.parquet
-> factors.build_factors
-> data/processed_private/csmar/price_volume_factors_full/price_volume_factors_<start>_<end>.parquet
-> evaluation.run_factor_evaluation OR pipelines.a_share.runner.run_processed_existing_pipeline
-> evaluation-set artifacts
```

## 关键模块

| 模块 | 当前职责 | 主要入口 |
|---|---|---|
| `markets/implementations/a_share/csmar/` | CSMAR raw discovery、field contracts、raw -> interim、interim inspect | `raw_exports.py`, `contracts.py`, `build_interim.py`, `inspect_raw_exports.py`, `inspect_interim.py` |
| `markets/implementations/a_share/datasets/` | universe rules、interim -> research panel、panel inspect | `universe.py`, `build_panel.py`, `inspect_panel.py` |
| `markets/implementations/a_share/factors/` | price-volume factors、forward labels、受控表达式、factor inspect | `build_factors.py`, `price_volume.py`, `labels.py`, `expressions.py`, `registry.py`, `inspect_factors.py` |
| `markets/implementations/a_share/evaluation/` | processed factor table -> `EvaluationSpec` adapter、summary、comparison | `evaluation_adapter.py`, `run_factor_evaluation.py`, `build_factor_summary.py`, `compare_factor_summaries.py` |
| `markets/implementations/a_share/validation/` | interim/panel/factor quality checks | `internal_tables.py`, `research_panel.py`, `factor_tables.py`, `quality_report.py` |
| `markets/implementations/a_share/runtime/` | private processed space gate and storage helpers | `storage.py`, `check_processed_space.py` |
| `pipelines/a_share/` | Pipeline config, snapshot, step registry, processed-existing runner | `config.py`, `snapshots.py`, `pipeline_step_registry.py`, `runner.py` |
| `research/factor_factory/` | Factor templates、candidate expansion、batch evaluation、candidate reports | `factor_registry.py`, `candidate.py`, `batch_runner.py`, `reports.py` |

## Raw / Interim 层

| 文件或对象 | 功能 |
|---|---|
| `RawExportIntakeResult` | 描述 raw export tree inspection 结果。 |
| `is_raw_export_file(path)` | 判断 raw export 文件后缀是否属于支持范围。 |
| `iter_raw_export_files(root)` | 遍历 raw export 文件。 |
| `inspect_raw_export_tree(root)` | 检查 raw 文件树，供导入前审计。 |
| `csmar/contracts.py` | 维护 CSMAR raw/interim 最小字段契约。 |
| `csmar/columns.py` | 维护标准列名、裁剪列和 slim factor base columns。 |
| `build_interim.py` | 将 raw CSV 标准化为 interim CSV。 |

边界：

- raw/interim 都是私有数据层，不进入 Git、公开报告或聊天记录。
- 字段错误、编码错误和缺文件应先通过 inspect 命令定位，不用 evaluation 层补救。

## Panel 层

核心入口：

```text
build_panel.build_research_panel()
build_panel.build_panel_full_from_interim()
```

panel builder 的主要职责：

- 从 interim yearly CSV 读取行情和基础表。
- 合并 stock master。
- `merge_asof` market cap。
- 计算 turnover。
- 计算上市天数过滤。
- 生成 `eligible`、`can_enter`、`can_exit`。
- 写出 `panel_full/research_panel_<start>_<end>.parquet`。

关键列：

| 列 | 语义 |
|---|---|
| `stock_code` | 股票代码，必须保持字符串和前导零。 |
| `trade_date` | 日频 session date。 |
| `eligible` | 当前截面是否进入可选 universe。 |
| `can_enter` | 下一交易日是否允许新开仓。 |
| `can_exit` | 下一交易日是否允许退出已有仓位。 |
| `market_type` | 市场或板块分组输入。 |
| `total_market_cap`, `float_market_cap` | 市值相关控制列。 |

常见误读：

- `can_enter=False` 不表示已有仓位必须消失；退出能力由 `can_exit` 决定。
- panel 是 full-sample processed 产物，不是按年拆分目录的 yearly run。

## Factor / Label 层

核心入口：

```text
build_factors.calculate_price_volume_factor_full()
price_volume.calculate_price_volume_factors()
price_volume.add_forward_return_labels()  # labels.py 也 re-export 该函数
expressions.parse_factor_expression()
```

当前 factor set：

- momentum。
- reversal。
- volatility。
- amount liquidity。
- turnover。
- Amihud illiquidity。
- amplitude。

默认窗口：

```text
20, 60, 120
```

默认 forward labels：

```text
1, 5, 10, 20, 60, 120 trading days
```

slim factor table 必须保留：

| 类别 | 字段 |
|---|---|
| key | `stock_code`, `trade_date` |
| tradability | `eligible`, `can_enter`, `can_exit` |
| common controls | `market_type`, `amount`, `turnover`, `total_market_cap`, `float_market_cap` |
| factors | `mom_20d`, `rev_60d`, `vol_120d`, `amihud_20d` 等 |
| labels | `forward_return_1d`, `forward_return_5d`, `forward_return_20d` 等 |

受控表达式：

- `expressions.parse_factor_expression()` 只解析白名单字段和函数。
- 不执行 Python 代码。
- `eval(...)`、未来 label 字段、未知字段和非法窗口必须被拒绝。

## A 股 Evaluation Adapter

核心入口：

```text
evaluation_adapter.run_processed_factor_evaluation()
evaluation_adapter.read_evaluation_frame()
evaluation_adapter.read_factor_frame()
evaluation_adapter.make_a_share_evaluation_spec()
run_factor_evaluation.main()
```

`run_factor_evaluation` 语义：

- `--start-year/--end-year` 用于定位 `price_volume_factors_<start>_<end>.parquet`。
- 如果精确文件不存在，选择能覆盖该年份区间的更宽文件。
- `--start-date/--end-date` 用于过滤 `trade_date`。
- 每个 label 生成一个 evaluation-set run。
- 输出 `label_availability.csv`、`evaluation_manifest.json` 和 `evaluation_runs/<run_id>/`。

不能用临时切片 Parquet + `evaluate-parquet` 替代 A 股 adapter 验收，因为那会绕过：

- 覆盖文件选择。
- label availability summary。
- A 股私有 metadata。
- source file reference。
- pipeline source alias。

## Pipeline Builder / Runner

### API payload

`src/quant_research_platform/interfaces/api/services/pipeline_service.py` 定义：

- `FactorTemplatePayload`
- `PipelineRunRequestPayload`
- `PipelineRunSummaryPayload`
- `PipelineCheckpointPayload`

`PipelineRunRequestPayload` 当前字段：

```text
market
run_id
date_start
date_end
factor_expression
factor_template_id
portfolio_mode
cost_bps
pnl_return_column
label_columns
processed_factor_column
engine
execution_mode
```

### Snapshot 模式

默认 `execution_mode="snapshot"`：

```text
PipelineConfig
-> validate_pipeline_config
-> write_run_snapshot
-> run_config.yaml
-> warnings.json
-> summary.md
```

这是前端常规默认行为，用于保存配置，不运行完整 evaluation。

### Processed-existing 模式

`execution_mode="processed_existing"`：

```text
PipelineRunRequestPayload
-> PipelineConfig
-> run_processed_existing_pipeline
-> run_processed_factor_evaluation
-> evaluation/evaluation_manifest.json
-> checkpoints.json
-> runtime_summary.json
```

真实范围：

- 只使用已有 processed factor table。
- 可以按 template 或已存在 processed factor column 解析 factor。
- 写 checkpoint、runtime summary、evaluation manifest。
- 默认不覆盖已有正式输出。
- 不重建 raw/interim/panel/factors。

## Pipeline Step Registry

`pipelines/a_share/pipeline_step_registry.py` 当前声明：

| Step | Version | Inputs | Outputs | Capabilities |
|---|---|---|---|---|
| `snapshot` | `1` | `PipelineConfig` | `run_config.yaml`, `warnings.json`, `summary.md` | `validate`, `write_snapshot` |
| `processed_evaluation` | `1` | `PipelineConfig`, `processed_factor_table` | `evaluation/evaluation_manifest.json`, `evaluation/label_availability.csv`, `evaluation/evaluation_runs/` | `evaluate`, `checkpoint` |

## Factor Factory

当前第一版由受控 registry 驱动：

```text
config/a_share/factor_templates/*.yaml
-> FactorTemplateRegistry
-> FactorCandidate
-> evaluate_processed_existing_candidates
-> run_processed_existing_pipeline
-> candidate_summary.csv / failed_candidates.json / manifest.json
-> /api/factor-factory-runs
-> web/src/FactorFactoryRuns.tsx
```

当前模板文件：

- `config/a_share/factor_templates/liquidity_20d.yaml`
- `config/a_share/factor_templates/momentum_60d.yaml`
- `config/a_share/factor_templates/reversal_20d.yaml`

`RegisteredFactorTemplate` 和 `FactorCandidate` 至少携带：

- stable id/name/version。
- expression。
- dependencies / inputs / outputs。
- capabilities。
- sample date range。
- label columns。
- parameters / grid。
- time semantics。
- availability rule。

失败记录必须区分：

- 无法映射到 existing processed factor column。
- pipeline validation failure。
- missing field。
- insufficient sample。
- unsafe expression。

## Validation / Runtime

| 层级 | 当前检查点 |
|---|---|
| raw | 文件是否存在、metadata、encoding、字段可读性 |
| interim | 日期、股票代码、关键 numeric 字段和裁剪列 |
| panel | `stock_code + trade_date`、eligible、can_enter、can_exit、market cap join |
| factor | factor columns、forward labels、PnL return column、slim columns |
| evaluation reports | schema `0.5`、`manifest.display`、time breakdown 和 label availability |
| storage | processed private data size、legacy ratio、free space ratio |

## 命令速查

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"

conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.csmar.inspect_raw_exports --help
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.csmar.build_interim --help
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.datasets.build_panel --help
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.factors.build_factors --help
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation --help
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.runtime.check_processed_space --help
```

## 当前边界总结

- A 股 full private evaluation 必须优先走 A 股 adapter。
- `evaluate-parquet` 是 generic/local Parquet adapter，不是 A 股全流程验收替代。
- Pipeline Builder 默认 snapshot；processed-existing 是 Phase 1 已实现轻量 runner。
- Factor Factory 第一版只使用 registry 和已有 processed factor columns。
- 私有数据和 private reports 不公开、不提交、不复制到聊天记录。

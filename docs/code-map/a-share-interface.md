# A 股接口地图

本页记录 A 股 market module 的当前可执行接口。它是私有 CSMAR 数据流程的入口地图，不替代 `EVALUATION_INTERFACE.md` 的 evaluation contract。

## 生产流程

```text
data/interim_private/csmar/
-> datasets.build_panel
-> data/processed_private/csmar/panel_full/research_panel_<start_year>_<end_year>.parquet
-> factors.build_factors
-> data/processed_private/csmar/price_volume_factors_full/price_volume_factors_<start_year>_<end_year>.parquet
-> evaluation.run_factor_evaluation
-> reports/private_real/factor_evaluation/<run_set>/evaluation_runs/<run_id>/
```

## 命令入口

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONPATH="src"

conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.datasets.build_panel --help
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.factors.build_factors --help
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation --help
```

`build_panel` 使用 `--start-year/--end-year` 读取 interim yearly CSV，并用 `--end-date` 选择 market-cap snapshot。

`build_factors` 使用 `--start-year/--end-year` 定位 full panel 文件并输出 slim factor table。`--windows` 和 `--horizons` 控制 price-volume factor windows 与 forward-return labels。

`run_factor_evaluation` 使用 `--start-year/--end-year` 找 processed factor Parquet。若精确年份文件不存在，`factor_full_path_covering_range()` 会选择覆盖该年份范围的更宽文件。`--start-date/--end-date` 是 evaluation 阶段的 `trade_date` 行过滤。

## 验收路径

A 股验收应区分两类入口：

- A 股全流程或私有 runs：使用 `quant_research_platform.markets.implementations.a_share.evaluation.run_factor_evaluation`。
- 通用 evaluation 或外部本地 Parquet 验收：使用 `quant_research_platform.interfaces.cli evaluate-parquet`。

不要用临时裁剪 Parquet + `evaluate-parquet` 替代 A 股全流程验收；那只能证明 generic evaluation adapter、artifact、API、frontend 链路可用，不能证明 A 股 adapter 的覆盖文件选择、日期过滤、label availability summary 和 private run metadata 正确。

`scripts/phase1_acceptance.py` 是 Phase 1 私有验收辅助脚本，边界是 acceptance run 重命名、2024 小样本 run 生成、小文件对比和 Factor Factory smoke。它会触碰 `reports/private_real/` 下的私有验证产物，不应作为日常轻量单元检查或公开数据验证命令。

## Pipeline Builder 和 API

`web/src/PipelineBuilder.tsx` 当前默认提交 `POST /api/pipeline-runs`，请求未显式设置 `execution_mode` 时后端使用 `snapshot`，只写出配置快照：

```text
run_config.yaml
warnings.json
summary.md
```

后端 `src/quant_research_platform/interfaces/api/services/pipeline_service.py` 同一入口也支持 `execution_mode=processed_existing`。该模式调用 `src/quant_research_platform/pipelines/a_share/runner.py` 的轻量 runner，在已有 processed factor table 上执行 A 股 evaluation adapter，并写出 runner checkpoints、runtime summary 与 evaluation manifest。

因此，前端文案和常规手动操作应描述为“默认 snapshot”；API/后端文档可说明存在 `processed_existing` 轻量 runner。不要把它表述为完整端到端 build/evaluate 编排，也不要暗示它会重新构建 raw/interim/processed 数据层。

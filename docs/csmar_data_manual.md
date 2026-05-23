# CSMAR 数据手册

本文档说明本项目的 CSMAR 私有数据流程。账号、密码、cookie、token、网页截图和 vendor 明细数据都不进入公开材料。

## 数据层

```text
data/raw_private/csmar/
  CSMAR 原始导出，完整保留，不改名、不删列

data/interim_private/csmar/
  raw -> interim 后的内部标准字段，CSV，便于人工核对

data/processed_private/csmar/
  panel 和 factor Parquet，供研究和平台 evaluation layer 读取
```

## 必要 CSMAR 表

| 项目数据集 | CSMAR 表 | 中文含义 | raw 放置目录 |
|---|---|---|---|
| `stock_master` | `TRDNEW_Co` | 公司文件 | `data/raw_private/csmar/stock_master/` |
| `daily_market` | `TRDNEW_Dalyr` | 日个股回报率文件 | `data/raw_private/csmar/daily_market/` |
| `market_cap` | `TRDNEW_Capchg` | 股本变动文件 | `data/raw_private/csmar/market_cap/` |
| `adjustment` | `TRD_AdjustFactor` | 股票价格复权因子表（日） | `data/raw_private/csmar/adjustment/` |
| `index_info` | `IDX_Idxinfo` | 指数基本信息文件 | `data/raw_private/csmar/index_market/` |
| `index_market` | `IDX_Idxtrd` | 国内指数日行情文件 | `data/raw_private/csmar/index_market/` |

本项目当前只从 `index_market` 生成：

```text
data/interim_private/csmar/benchmark_index_market_by_year/
```

不再生成全量 `index_market_by_year/`。

## 文件命名

推荐 raw 文件名：

```text
csmar__{dataset}__{market_scope}__{start}_{end}__export-{YYYYMMDD}.csv
```

示例：

```text
csmar__daily_market__all_a_ex_bjse__20140101_20181231__export-20260514.csv
```

## 字段映射

字段映射定义在：

```text
src/quant_research_platform/markets/implementations/a_share/csmar/contracts.py
```

示例：

| CSMAR 字段 | 内部字段 |
|---|---|
| `Stkcd` | `stock_code` |
| `Trddt` | `trade_date` |
| `Opnprc` | `open` |
| `Clsprc` | `close` |
| `Dretwd` | `return_with_dividend` |

`build_interim.py` 会继续生成内部派生列，例如：

- `return_1d`
- `adj_close`
- `adj_open`
- `float_market_cap`
- `total_market_cap`
- `is_trading`
- `is_st`
- `is_a_share_scope`
- `eligible_base`

## 列裁剪

列裁剪规则定义在：

```text
src/quant_research_platform/markets/implementations/a_share/csmar/columns.py
```

本地可选配置：

```text
config/private_column_overrides.json
```

首次运行 A 股 `build_interim.py` 时，如果配置不存在，会自动生成默认模板。JSON 使用内部列名：

```json
{
  "daily_market": {
    "keep": ["return_without_dividend"],
    "drop": [],
    "_available_optional_columns": {
      "return_without_dividend": "不含现金红利的日收益率；第一版默认不保留。"
    }
  }
}
```

实际列裁剪记录写到：

```text
data/interim_private/csmar/_manifests/column_pruning_report.md
```

## 输出约定

interim：

```text
data/interim_private/csmar/
  daily_market_by_year/daily_market_YYYY.csv
  stock_master/stock_master.csv
  market_cap/market_cap_to_20251231.csv
  adjustment/adjustment_20140101_20251231.csv
  benchmark_index_market_by_year/benchmark_index_market_YYYY.csv
  index_info/index_info.csv
```

processed：

```text
data/processed_private/csmar/
  panel_full/research_panel_2014_2025.parquet
  price_volume_factors_full/price_volume_factors_2014_2025.parquet
```

## 验证入口

```powershell
$env:PYTHONPATH="src"
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.csmar.inspect_raw_exports data/raw_private/csmar
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.csmar.inspect_interim --start-year 2014 --end-year 2025
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.datasets.inspect_panel --start-year 2014 --end-year 2025
conda run -n analysis python -m quant_research_platform.markets.implementations.a_share.factors.inspect_factors --start-year 2014 --end-year 2025
```

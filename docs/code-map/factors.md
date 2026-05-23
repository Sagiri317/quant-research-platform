# Factors 模块地图

目录：

```text
src/quant_research_platform/markets/implementations/a_share/factors/
```

## 职责

Factor 层从 full research panel 生成 slim factor table。它负责第一批 price-volume 因子、forward return labels 和第一版受控 factor expression，不负责最终评价指标。

## 关键文件

| 文件 | 作用 |
|---|---|
| `build_factors.py` | full panel -> slim full factor Parquet |
| `price_volume.py` | momentum、reversal、volatility、amount、turnover、Amihud、amplitude 因子 |
| `labels.py` | forward return labels |
| `expressions.py` | 第一版 whitelist factor expression parser |
| `inspect_factors.py` | 检查 factor table 行数、列和缺失情况 |

## Factor 主线

```text
panel_full/research_panel_2014_2025.parquet
-> read_full_panel_for_factor_build()
-> add_forward_return_labels()
-> calculate_price_volume_factors()
-> select_slim_factor_columns()
-> price_volume_factors_full/price_volume_factors_2014_2025.parquet
```

## 关键口径

- 当前完整实现仍以 `price_volume` factor set 为主。
- 默认窗口：`20, 60, 120`。
- 默认 forward label horizons：`1, 5, 10, 20, 60, 120`。
- `forward_return_1d` 是后续 PnL path 的默认 realized return 列。
- `forward_return_20d` 等 longer-horizon label 主要用于 IC / RankIC / grouped diagnostics。
- Factor expression parser 只能解析白名单字段和函数，不执行 Python 代码。

## 函数阅读路线

| 顺序 | 函数 | 阅读目标 |
|---:|---|---|
| 1 | `build_factors.panel_full_path()` | 看懂当前 full panel 的固定位置 |
| 2 | `build_factors.read_full_panel_for_factor_build()` | 看懂因子构造只读哪些必要列 |
| 3 | `labels.add_forward_return_labels()` | 看懂 forward labels 如何按 stock_code 向前 shift |
| 4 | `price_volume.calculate_price_volume_factors()` | 看懂第一批 price-volume 因子如何生成 |
| 5 | `build_factors.select_slim_factor_columns()` | 看懂为什么 factor table 是 slim table |
| 6 | `build_factors.calculate_price_volume_factor_full()` | 看懂 full factor 的写出入口 |
| 7 | `expressions.parse_factor_expression()` | 看懂受控表达式如何记录依赖而不是执行代码 |
| 8 | `inspect_factors.main()` | 看懂 factor 层验证关注哪些字段 |

## 常见误读

- full factor 文件不是“一年一个文件”；当前是一个覆盖 2014-2025 的 slim Parquet。
- factor table 保留 key、tradability、常用控制列、factor 和 label，不保留完整 OHLC 明细。
- `forward_return_20d` 不是 PnL 曲线的日收益列；正式 PnL 默认使用 `forward_return_1d`。
- 滚动窗口会制造自然缺失；判断异常时要区分 warm-up 缺失和真实数据断裂。

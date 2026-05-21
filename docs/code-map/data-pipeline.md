# Data Pipeline 模块地图

目录：

```text
src/quant_research_platform/markets/a_share/csmar/
src/quant_research_platform/markets/a_share/datasets/
```

## 职责

数据层把用户手工导出的 CSMAR CSV 转成平台内部可复用的 A 股 full research panel。它不做因子评价，也不连接外部 API、网页登录、下载器或凭据。

## 关键文件

| 文件 | 作用 |
|---|---|
| `csmar/raw_exports.py` | 描述原始导出文件、metadata 和路径发现 |
| `csmar/contracts.py` | CSMAR 原始表和 interim 表的字段契约 |
| `csmar/columns.py` | 字段命名、slim factor base columns 和列裁剪配置 |
| `csmar/inspect_raw_exports.py` | 检查 raw 导出是否齐全、字段是否满足契约 |
| `csmar/build_interim.py` | raw CSV -> normalized interim CSV |
| `csmar/inspect_interim.py` | 检查 interim 结果质量 |
| `datasets/universe.py` | A 股样本池过滤规则 |
| `datasets/build_panel.py` | interim -> `panel_full/research_panel_2014_2025.parquet` |
| `datasets/inspect_panel.py` | 检查 full panel 行数、字段和质量 |

## 数据主线

```text
data/raw_private/csmar
-> inspect_raw_exports.py
-> build_interim.py
-> data/interim_private/csmar
-> inspect_interim.py
-> build_panel.py
-> data/processed_private/csmar/panel_full/research_panel_2014_2025.parquet
-> inspect_panel.py
```

## Panel 构造主线

`build_panel.py` 的核心函数是 `build_research_panel()`：

```text
daily_market
-> merge stock_master
-> merge_asof market_cap
-> compute turnover
-> compute listing age filters
-> compute eligible
-> compute next-day tradability
-> output full panel
```

`eligible`、`can_enter`、`can_exit` 是后续 portfolio path 的关键列：

- `eligible`：当前截面是否进入可选 universe。
- `can_enter`：下一交易日是否允许新开仓。
- `can_exit`：下一交易日是否允许退出已有仓位。

## 函数阅读路线

| 顺序 | 函数 / 常量 | 阅读目标 |
|---:|---|---|
| 1 | `raw_exports.discover_raw_exports()` | 看懂项目如何发现用户手工导出的 raw 文件 |
| 2 | `contracts.*` | 看懂每张 CSMAR 表最小需要哪些字段 |
| 3 | `build_interim.main()` | 看懂 raw 到 interim 的入口参数 |
| 4 | `universe.*` | 看懂 A 股 universe 和排除规则 |
| 5 | `build_panel.build_research_panel()` | 看懂 panel 字段如何构造 |
| 6 | `build_panel.build_panel_full_from_interim()` | 看懂为什么当前只写一个 full panel |
| 7 | `inspect_panel.main()` | 看懂 panel 验证关注哪些 blocking issue |

## 常见误读

- `can_enter=False` 不等于已有仓位必须消失；已有仓位是否能退出要看 `can_exit`。
- panel 是 full-sample 产物；年份只是筛选维度，不是存储目录维度。
- `stock_code` 必须保持字符串和前导零，不能让 pandas 自动转成整数。
- raw / interim / processed 都是私有数据，不应出现在公开报告或 Git 提交中。

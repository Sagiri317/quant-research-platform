# Validation / Runtime 模块地图

目录：`code/validation/`、`code/runtime/`

## 职责

Validation 和 runtime 层用于复核私有生成产物，避免在大文件生成后才发现列缺失、行数异常、空间不足或口径漂移。

## 关键文件

| 文件 | 作用 |
|---|---|
| `code/validation/common.py` | shared validation helpers |
| `code/validation/internal_tables.py` | interim / internal table checks |
| `code/validation/research_panel.py` | research panel checks |
| `code/validation/factor_tables.py` | factor table checks |
| `code/runtime/storage.py` | 目录大小、文件大小和空间估算 |
| `code/runtime/check_processed_space.py` | processed 私有数据空间 gate |

## 验证主线

```text
inspect_raw_exports.py
-> inspect_interim.py
-> inspect_panel.py
-> inspect_factors.py
-> check_processed_space.py
-> evaluation summary / comparison checks
```

## 关键检查点

| 层级 | 重点 |
|---|---|
| raw | 文件是否存在、metadata 是否匹配、CSV 编码和字段是否可读 |
| interim | 日期、股票代码、关键 numeric 字段、字段裁剪是否符合 contract |
| panel | `stock_code + trade_date`、eligible、can_enter、can_exit、market cap join |
| factor | factor columns、forward labels、PnL return column、slim columns |
| reports | full run 是否有 `metrics_by_year`、summary 是否来自当前口径 |
| storage | full panel / full factor / reports 是否在可接受空间内 |

## 函数阅读路线

| 顺序 | 函数 / 脚本 | 阅读目标 |
|---:|---|---|
| 1 | `runtime.storage.directory_size()` | 看懂空间统计基础 |
| 2 | `check_processed_space.main()` | 看懂 processed 空间 gate |
| 3 | `validation.research_panel.*` | 看懂 panel blocking issue 的定义 |
| 4 | `validation.factor_tables.*` | 看懂 factor / label 完整性检查 |
| 5 | `inspect_panel.main()` | 看懂 panel 检查如何输出给用户 |
| 6 | `inspect_factors.main()` | 看懂 factor 检查如何输出给用户 |

## 常见误读

- 空间检查只保护 processed 私有数据，不代表 reports 目录也足够干净。
- `reports/private_real/factor_evaluation/comparisons/` 可以保留轻量证据，但大体积 diagnostic baseline 应清理。
- 删除生成目录必须先确认路径和用途；不要用通配符或递归批量清理未经确认的目录。
- 测试通过只说明代码行为没破坏，不等于研究结论已经稳健。


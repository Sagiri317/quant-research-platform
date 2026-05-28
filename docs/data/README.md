# Data Documentation

本目录记录 `quant-research-platform` 的数据文档。这里说明数据来源、字段口径、可用性和私有边界；不保存 row-level private data、CSMAR 原始明细、股票代码样本、凭据或本机绝对路径。

## 当前文档

| 文档 | 职责 |
|---|---|
| [a-share-csmar-data-manual.md](a-share-csmar-data-manual.md) | A 股 CSMAR vendor-specific 数据源说明：raw/interim/processed、字段映射、列裁剪、隐私边界和验证入口 |
| [a-share-field-inventory.md](a-share-field-inventory.md) | Source-neutral A 股字段可用性清单：full panel、slim factor table、evaluation input、N5 diagnostics readiness |

## 边界

- 字段清单解释当前可用性，不定义新的 schema、API 或 frontend contract。
- Artifact table/file contract 的机械真相仍在 `src/quant_research_platform/artifacts/schema_registry.py` 和测试中。
- A 股 CSMAR vendor-specific 字段映射仍以 `src/quant_research_platform/markets/implementations/a_share/csmar/contracts.py` 和 `src/quant_research_platform/markets/implementations/a_share/csmar/columns.py` 为准。
- CSMAR raw/interim/processed 与 `reports/private_real/` 仍是本地私有边界，不提交、不公开。

## 维护规则

- 更新 A 股 raw/interim/processed 字段映射时，同步 [a-share-csmar-data-manual.md](a-share-csmar-data-manual.md)。
- 更新 full panel、slim factor table、evaluation input 或 N5 diagnostics 字段可用性时，同步 [a-share-field-inventory.md](a-share-field-inventory.md)。
- 不在本目录记录一次性 run id、row-level 样本、完整私有路径、vendor 明细或 benchmark 数字。

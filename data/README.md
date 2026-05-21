# 私有数据区

本目录是 `quant-research-platform` 的本地私有数据边界。当前第一版完整 market module 是 A 股，数据来自用户手工导出的 CSMAR 文件。

## 目录职责

```text
reference_docs/      CSMAR PDF 说明书和版权声明
raw_private/         用户手工导出的 CSMAR 原始文件
interim_private/     raw 清洗后的中间审计层
processed_private/   研究用 Parquet 输出
```

当前三层数据都属于本地私有数据，不得提交或公开。

## 当前 processed 合同

```text
processed_private/csmar/
  panel_full/
    research_panel_2014_2025.parquet
  price_volume_factors_full/
    price_volume_factors_2014_2025.parquet
```

- interim CSV 仍按年保存，便于人工核对和局部重建。
- processed panel / factor 是 full-sample Parquet。
- panel Parquet 是完整研究主表。
- slim factor Parquet 是 A 股 evaluation 和 pipeline 输入之一。
- factor 文件保留 key、过滤列、流动性/规模辅助列、factor columns 和 forward label columns。

## 禁止用途

- 不要保存学校账号、密码、cookie、token、`.env` 或任何凭据。
- 不要提交 CSMAR raw/interim/processed 数据。
- 不要把本目录用于自动下载脚本、爬虫、API client 或浏览器自动化。
- 不要公开 `raw_private/`、`interim_private/`、`processed_private/` 或 `reports/private_real/` 下的 vendor 明细。

## 相关文档

- CSMAR 数据手册：`docs/csmar_data_manual.md`
- 平台源码阅读地图：`docs/code-map/`

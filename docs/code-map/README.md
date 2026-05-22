# Code Map 总览

本目录是 `quant-research-platform` 的源码阅读地图，不替代研究报告、数据手册或长期功能池。项目边界看 [../../README.md](../../README.md)，当前事实看 [../../PROJECT_STATUS.md](../../PROJECT_STATUS.md)，后续能力池看 [../architecture/future-capabilities.md](../architecture/future-capabilities.md)，CSMAR 字段口径看 [../csmar_data_manual.md](../csmar_data_manual.md)。

## 阅读入口

| 文档 | 说明 |
|---|---|
| [platform-flow.md](platform-flow.md) | 平台接口、入口参数、A 股核心流向和未来 crypto 高频构想图 |
| [a-share-interface.md](a-share-interface.md) | A 股 build panel / build factors / run evaluation 接口，以及前端 pipeline snapshot 边界 |
| [data-pipeline.md](data-pipeline.md) | CSMAR raw -> interim -> full research panel |
| [factors.md](factors.md) | full panel -> price-volume factors -> forward labels |
| [evaluation.md](evaluation.md) | market module -> `EvaluationSpec` -> platform evaluation / artifacts |
| [validation-runtime.md](validation-runtime.md) | validation checks、space gate 和生成产物复核 |

## 推荐源码阅读顺序

按平台数据流读：

1. [../architecture/future-capabilities.md](../architecture/future-capabilities.md)：看懂平台长期能力边界和后续研究路线。
2. [a-share-interface.md](a-share-interface.md)：看懂 A 股当前可执行入口和前端 snapshot 边界。
3. [data-pipeline.md](data-pipeline.md)：看懂 raw CSMAR 如何进入 interim，再形成 full panel。
4. [factors.md](factors.md)：看懂 panel 如何变成 slim factor table 和 forward labels。
5. [evaluation.md](evaluation.md)：看懂 A 股模块如何构造 `EvaluationSpec` 并调用平台 evaluation 入口。
6. [validation-runtime.md](validation-runtime.md)：看懂每一层产物如何检查、空间如何估算。

如果只想调试某个现象，可以反向读：

| 现象 | 先读 |
|---|---|
| 原始 CSV 列名或编码不对 | `data-pipeline.md` |
| panel 行数或 eligible / can_enter / can_exit 不对 | `data-pipeline.md` -> `validation-runtime.md` |
| factor 缺列、label 缺失、滚动窗口异常 | `factors.md` -> `validation-runtime.md` |
| evaluation run 没有 schema `0.5` 或 `manifest.display` | `evaluation.md` |
| PnL、turnover、download 或 workbench 口径不对 | `evaluation.md` -> `ARCHITECTURE.md` |
| 磁盘空间不够或误留旧产物 | `validation-runtime.md` |

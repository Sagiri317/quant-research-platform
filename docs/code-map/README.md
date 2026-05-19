# Code Map 总览

本目录是 `a-share-alpha-lab` 的源码阅读地图，不替代研究报告或数据手册。项目边界看 [../../README.md](../../README.md)，当前事实看 [../../PROJECT_STATUS.md](../../PROJECT_STATUS.md)，CSMAR 字段口径看 [../csmar_data_manual.md](../csmar_data_manual.md)。

## 阅读入口

| 文档 | 说明 |
|---|---|
| [data-pipeline.md](data-pipeline.md) | CSMAR raw -> interim -> full research panel |
| [factors.md](factors.md) | full panel -> price-volume factors -> forward labels |
| [evaluation.md](evaluation.md) | full factor table -> `alpha-research-core` evaluation / handoff |
| [validation-runtime.md](validation-runtime.md) | validation checks、space gate 和生成产物复核 |

## 推荐源码阅读顺序

按数据流读：

1. [data-pipeline.md](data-pipeline.md)：看懂 raw CSMAR 如何进入 interim，再形成 full panel。
2. [factors.md](factors.md)：看懂 panel 如何变成 slim factor table 和 forward labels。
3. [evaluation.md](evaluation.md)：看懂 ashare 如何构造 `EvaluationSpec` 并调用 core。
4. [validation-runtime.md](validation-runtime.md)：看懂每一层产物如何检查、空间如何估算。

如果只想调试某个现象，可以反向读：

| 现象 | 先读 |
|---|---|
| 原始 CSV 列名或编码不对 | `data-pipeline.md` |
| panel 行数或 eligible / can_enter / can_exit 不对 | `data-pipeline.md` -> `validation-runtime.md` |
| factor 缺列、label 缺失、滚动窗口异常 | `factors.md` -> `validation-runtime.md` |
| full evaluation 没有年度 breakdown | `evaluation.md` |
| PnL 或 turnover 口径不对 | `evaluation.md` -> core `docs/code-map/core.md` |
| 磁盘空间不够或误留旧产物 | `validation-runtime.md` |

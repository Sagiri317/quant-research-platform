# QRP 路由地图

本文件是 token-saving repo map（节约 token 的仓库索引）。它只帮助 agent 判断先读哪里，不替代 `docs/code-map/**` 或源码事实。

## 源码 owner layer

```text
contracts/   稳定 spec、ID、合同对象。改 public contract 前必须过 Public Surface Gate。
storage/     数据布局、读写、manifest、partition、checksum、storage privacy。
catalog/     可查询 catalog、research profile composition。
sources/     vendor/source 输入到 canonical interim data。
markets/     市场语义；当前 active implementation 是 a_share。
research/    factors、labels、datasets、evaluations、models、predictions、portfolios。
workflows/   本地 DAG、执行计划、checkpoint、resume。
compute/     dataframe/tensor 计算工具。
agents/      proposal-only planning/spec boundaries；不调用外部 LLM API。
interfaces/  API/CLI 适配层；public surface 最小化。
reports/     report catalog、private report artifact writers。
runtime/     privacy、sanitization、logging、profiling、errors、resources。
```

## 读取顺序建议

- Review / merge / apply-only：先读 diff，不先读源码。
- Implementation：先根据 owner layer 定位目标模块和 tests。
- Public API/CLI 相关：先看 `interfaces/`、contracts、tests，再看 docs。
- A-share 语义相关：先看 `markets/implementations/a_share/`。
- Factor/label/research dataset：先看 `research/` 里对应 owner 子层。
- Private safety / package：先看 `.agents` references 和 scripts，再看 `.run` summary/manifest。

## 不要默认读取

- 私有数据目录。
- `.run` raw logs / 大 stdout。
- 旧 handoff zip/json。
- 全部 templates。
- 全部 docs/architecture。

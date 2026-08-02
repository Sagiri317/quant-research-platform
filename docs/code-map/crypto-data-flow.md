# Crypto Data Flow

本文只描述 crypto/Binance 研究链路的长期 owner flow、资源边界和失败语义；不复制 dataset、字段、signal 或 label 清单。

## Owner Flow

```text
sources/binance/raw_exports.py
-> sources/binance/interim.py
-> markets/implementations/crypto/processed/
-> research factor/label owners
-> markets/implementations/crypto/research_profiles.py
-> markets/implementations/crypto/research_scenarios.py
```

- `sources/binance/` 负责 raw envelope、file discovery、source-local validation 与 interim normalization，不拥有 market alpha 语义。
- `markets/implementations/crypto/processed/` 负责 source-to-canonical binding、dependency windows、panel/factor/label materialization 与 market lifecycle validation。
- `research/` 负责通用 factor、label、evaluation 与 portfolio 计算，不 import crypto implementation，也不写文件。
- `storage/` 负责 physical traversal、exact spillable key validation、manifest 和 publication；crypto owner 不复制这些能力。

## Lifecycle Invariant

- 一个 operation 复用同一 `RuntimeResourcePolicy`；physical reads、known payload checks、reducer state 和 output publication 各由真实 owner admission。
- source-input IDs 必须与 profile 精确匹配；canonical asset identity、dependency identity 与 core-window coverage fail closed。
- panel、factor 和 label 在 staged artifact 上完成 direct-input identity、schema 与 exact-key validation后才发布。
- chunked factor/label 计算保持 canonical key 单调、active-asset state 有界并在 normal、error 或 early close 时释放 upstream。
- quote/trade reducer 的业务有效性由 crypto processed owner 决定；Storage 只提供过滤、顺序、key 与 IO 真相。
- 当前显式 checks 不冒充完整 process RSS 或未诊断 Python/native state；无法证明有界的 state 保持明确 ceiling。

## Boundary

该链路只服务本地研究，不新增 execution connector、生产交易、external LLM、public API/CLI、scheduler、worker 或 deployment surface。

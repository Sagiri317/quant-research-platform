# 用户长期工程原则

本文件保存稳定偏好，供任务路由和审查时按需读取。不要把本文件扩写成完整标准；详细执行规则在 `AGENTS.md`、skill references、代码和 tests 中。

## 核心原则

- Semantic closure（语义收口）：接口、命名、contracts、tests、docs、artifacts、handoff 证据必须闭合。
- Naming first-class（命名一等）：命名是架构问题，不是 cosmetic（表面修饰）。避免 `v2`、实现细节泄漏和过渡期命名。
- Replace over parallel（替代优先）：能替代就删除旧路径；能参数化就不要新增函数/类/模块；不保留 legacy alias、compatibility shim、旧 re-export、旧 wrapper。
- Minimal public API（最小公共 API）：新增 public API / CLI / registry / store / facade 默认禁止。
- Evidence over claims（证据优先）：未运行就是 `NOT_RUN`，失败就是 `FAIL`；不能为了 final report 把状态写成 `PASS`。
- Private-data safety（私有数据安全）：public-safe evidence 只能是聚合/脱敏证据，不泄露 row-level private data。
- Layer-local change standard（局部改动标准）：新功能原则上主要改 owner layer 的少量代码；跨层大改先做架构收口。
- No handoff-driven architecture（handoff 不反向驱动架构）：handoff 是证据层，不是架构源。
- Extensibility without abstraction creep（可扩展但不抽象膨胀）：未来 ML/DL、时序、factor factory、多市场、高频、前端等要预留方向，但不提前落地泛化框架。

## 执行偏好

1. 能不新增就不新增。
2. 能合并就合并。
3. 能删除就删除。
4. 旧测试要迁移到新结构，不保护旧结构。
5. 速度优先于峰值内存；全局内存预算默认 70%；中小规模分块应尽量为 1。
6. 中文输出为主；英文技术术语首次出现时给中文解释。
7. 架构 review 需要严格检查命名、路径、语义、owner layer、未来扩展兼容性和最小变更面。

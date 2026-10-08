# Crypto 高频研究

本文件定义 Binance BTC/ETH 永续高频研究的能力、输入条件和经济合同。
协议为 crypto_hft，profile 为 crypto_binance_hft，正式数据集为 binance_futures_hft_research_100ms。
这是历史研究和执行近似，不是实盘交易、交易账户或投资建议。
代码实现、合成验证、真实数据覆盖和经济有效性必须分别说明。

研究链复用单一执行场所：不可变公开事件录制或供应商历史归档 → 确定性 L2 重建 →
微观结构特征/成熟标签 → 按时间切分的模型与持久化 OOS 预测 →
具有费用、延迟、部分成交、队列近似和资金费的 maker/taker 账本 →
净收益、库存、markout、不利选择和稳健性报告。

核心是解释事件顺序、可观测信息与执行假设，并在足够完整的真实 BTC/ETH 数据上检验增量价值。
公开 L2 只提供按价格聚合深度，不能推出真实订单 ID、精确队列位置或完整撤单生命周期。
研究必须说明近似适用条件；不把短合成回放或 DataFrame 位移结果称为真实撮合证明。

通用资源与发布见[工程合同](architecture.md)；原始数据布局与安全见[数据合同](data.md)。

## 研究能力与使用边界

| 能力 | 实现入口 | 条件与边界 |
|---|---|---|
| 录制 | futures-hft 期货100ms深度与聚合成交；spot-l2 独立现货深度录制 | 公开端点、无认证下单；未录制的完整L2历史不能事后补造 |
| 回放 | 运行清单、成员身份、快照与增量、序号、连接/分段和多时钟验证 | 现货/期货独立回放，缺口不静默连续 |
| 研究数据 | 持久化100ms PANEL；微观结构、衍生品状态、现货/永续、跨资产、外部价格发现五类因子 | 按配置选择输入；来源缺失阻塞依赖分支，局部缺口和不完整窗口按各因子合同保留 NULL |
| 模型 | 配置选择模型算法、参数、标签及训练/OOS资产范围，包含 LightGBM/XGBoost 和排名集成 | 训练拟合与标定随模型冻结；proxy不是成交概率，也不能推断未观测队列 |
| 执行 | Binance BTC/ETH perpetual 的 taker/maker 候选，显式成本/延迟，continuous 与 independent_segments 两种账本模式 | 队列和流动性为公开L2近似；所需覆盖不足阻塞相应净结果，不跨缺口拼接收益 |
| 展示 | 唯一目标DAG、统计、指定run报告、CLI/API、只读Notebook | 合成与读取验证不证明真实OOS获利或充分日频覆盖 |

信息来源与执行场所分别绑定：现货/永续上下文、BTC/ETH/SOL 跨资产信息和 Bybit/OKX 外部价格发现均有生产入口；执行组合只允许 Binance BTC/ETH perpetual。SOL 可用于训练、评估和预测，不能加入执行组合。

## 数据来源、录制与回放

研究窗口以连续UTC timestamp定义；午夜不是episode或账本状态边界。book、features、labels、
position、cash、NAV和funding跨午夜延续，仅真实gap、sequence或disconnect断状态。

### 原生录制与确定性回放

每个录制根目录包含一个 `run.json`，以及该清单声明的 gzip JSONL 成员，
不会通过 glob 合并多个运行。`recording_scope` 为每个输出根目录选择一个独立录制范围：

```text
futures-hft:
  USDⓈ-M Futures 差分深度 @ 100ms
  USDⓈ-M Futures 聚合成交
  -> 由 HFT 工作流消费

spot-l2:
  现货差分深度 @ 100ms
  -> 独立现货 L2 深度采集；完整 Spot PANEL 另需成交归档
```

录制清单以 `status` 为唯一状态字段；reader 不使用旧清单的重复字段
`market_data_status`，但来源身份仍绑定完整原清单，不改写旧文件。
两个范围使用独立的根目录、清单和汇总状态。现货中断不能使完整的期货 HFT 运行降级或被拒绝。
现货与期货录制的回放保持独立。
研究 profile 的执行主线为 100ms BTC/ETH 永续，Spot、SOL 和外部场所可按配置提供研究信息；
不提供分钟级研究场景。
独立录制可以并发运行，由调用方选择机器和并行度。
提供的输出根目录必须互不相同且不存在；录制器成功时以原子方式发布。

```text
python -m quant_research_platform.sources.binance.recorder --scope futures-hft --output-root <run-root> --duration-seconds <seconds>

python -m quant_research_platform.sources.binance.recorder --scope spot-l2 --output-root <run-root> --duration-seconds <seconds>
```

记录从命令启动时开始；未捕获的时段无法回填完整 L2。

RPI 深度、最优报价推送、强制订单、标记价格和辅助统计不属于上述原生录制范围。
衍生品状态研究可通过独立 native 归档消费标记价格、OI、forceOrder 和多空比等状态；
已有消费者不表示录制器已采集这些频道，也不保证辅助归档覆盖完整。

每条原始记录包含：

```text
capture_seq
receive_time
receive_monotonic_ns
connection_id
segment_id       # 仅用于深度与快照
payload          # 无损交易所载荷
```

`capture_seq` 只有一个责任模块：串行化的运行写入器。
每个清单成员都绑定精确路径、市场、数据流、数据种类、行数、采集序号范围、
最大未压缩记录字节数、压缩字节数与 SHA-256。
回放在使用前后校验清单、精确成员集合、哈希及物理标识。
连接覆盖在验证完整来源后建立精确身份映射，各数据流区间只排序一次。
PANEL 按开始时间单调消费游标，用活动区间堆区分零、一、多条覆盖；
每格最多返回两条身份，不反复扫描请求窗口前后历史。
分段诊断由录制 session 保留至发布验证；下游消费已校验诊断。
缺失采集序号时，永不退回时间戳排序。

深度重连会创建新的 `connection_id` 和 `segment_id`。
匹配的 REST 快照绑定到相同标识；回放按采集序号单次消费快照和增量，
从不把重连分段当作连续数据。现货从 `lastUpdateId + 1` 启动；
期货从 `lastUpdateId` 启动，随后要求 `pu == previous u`。
数量为零会删除一个档位。价格标识使用整数最小价位，而不是二进制浮点相等。
缺口、交叉盘口、单边盘口或已知深度不足，都会产生明确的无效边界；
不会静默复用过期盘口。

官方录制 reader 与离线回放共用同一个 DepthReconstructor 状态转移。
reader 在收到真实快照并完成初始化后才报告就绪；实时连接的未来结束时间未知，
不因此产生未来完整覆盖。原始触发事件先写入，再由相同已知价格范围/序列规则检查有效性；
已知范围耗尽、深度不足或序列失效时关闭真实连接，新连接使用新 segment 与 REST 快照。
快照网络失败进入已有有界重连退避，磁盘/格式失败仍保留主异常并关闭连接。
这种恢复只建立新的有效段，不补造断段中的盘口、成交或持仓；发生重连的录制为 PARTIAL。

以下时钟在整个 PANEL 来源谱系中保持独立：

```text
交易所事件时间（E）
期货交易时间（T）
本地 UTC 接收时间
本地单调接收时间
采集序号
分段标识
```

录制器使用有界 `websockets` 运行时连接公开数据流，
使用 `httpx` 获取公开 REST 元数据/快照。它没有认证端点或下单 API。

### 历史归档与来源身份

Binance 官方录制与 Tardis 原生历史归档均可作为明确来源；历史归档通过切片下载与导入。
研究配置示例中的三段方案使用 2026-07-01 train、2026-08-01 validation 和 2026-09-01 test 的各自真实原件。
source_archives 是显式归档集合；source owner 按清单实际覆盖跨度与请求窗口相交选择，
逐原件验证精确身份，不能只改一个归档的日期来冒充三段。
显式 `research --raw-root` 是共同 raw 根，主 Tardis 来源及辅助来源都从其 `tardis` 子树绑定。
maker calibration 的来源身份固定使用 train 起点至 development 末端；留出 PANEL 仍按自己的访问窗选档，
冻结校准继续验证开发原件身份，不重训、不改写已发布实例。
跨日归档按实际首尾切片范围选择，date 上界与 datetime 上界复用统一 TimeWindow 合同。
不同原件可重置本地序号；持久化 observation/segment 身份带原件命名空间，
跨原件空白保持真实 gap，不拼成连续事件段。
资金费使用同一窗口路由选中日根，逐根执行 forecast/settlement 完整性及身份验证；
coverage 保存实际不重叠区间集合，缺失所需根明确阻塞，不取多段包络填补未知月份。

免费公开日期可直接下载；需要 key 时明确权限不足，不自动注册、购买或切换替代来源。
原始字节、日期、原生频道及频率、文件哈希、稳定行序、断线标记与可证明区间进入来源清单；
正式 raw 布局和来源安全由 [数据合同](data.md) 统一约束。
单次研究默认使用一个明确来源，不自动混合两类数据补洞，来源差异必须进入持久化身份。
历史入口绑定明确的连续十分钟切片范围，时长可以小于一天或跨越任意多个 UTC 日。
下载与导入分别使用同一 raw owner 的 `download_slices` 与 `import_slices`；
`date` 是偏移的 UTC 起点，`offset_minutes` 是首片偏移，切片数量不设单日上限。
供应商请求和原件文件名仍按实际 UTC 日期寻址；同一归档的累计行序、观察段和回放状态
跨午夜连续，只有真实断连或原生序列缺口才切断连续性。已建立连续盘口后，忽略额外的日初引导快照，
遵循 [Tardis 官方回放实现](https://github.com/tardis-dev/tardis-node/blob/master/src/mappers/binance.ts)；
断连后的新观察段仍须重新用快照和增量桥接。资金费查询绑定实际归档范围。
train、validation、test 按时间先后划分；最后一折的实际 train 与 validation 窗口用于最终 refit，可以跨日，不要求自然日对齐；
特征与标签不能跨越实际无效行情区间，标签成熟检查不因跨日而放宽。
不同原件日期共同进入一个来源清单和精确身份，不另设日内/跨日研究路径。
已有原件通过同一 `import_slices` 重新生成归档清单，保留原生字节及原始来源证据。
历史来源复用同一盘口、100ms PANEL、因子/标签、模型和经济账本，不建立另一套回测链。
Binance 原生字段由同一个来源校验 owner 按清单 schema 和绑定的供应商观测 UTC 判定。
官方录制要求其供应商 schema 对应字段；Tardis 历史归档按当日 schema 校验。
依据 [Binance 更新日志](https://developers.binance.com/en/docs/derivatives/change-log)，
聚合成交的非 RPI 数量 `nq` 自 2025-12-31 引入，CM/UM 的 `st`/`ps` 自
2026-06-30 引入。历史可选字段一旦出现仍严格检查其类型、身份和数量范围。
缺失的历史 `nq` 不填入原载荷，也不推断为 `q` 或零；内部成交事件保留未知。
PANEL 总成交量因子和标签消费 `q`；需要非 RPI 数量的 maker
队列消耗和校准持久化明确 BLOCKED。不能因归档成功声称 maker 所需信息齐备。

原生深度事件频率与研究网格分别声明；细增量仍按其顺序参与 OFI 等充分统计，
不能以每格首尾价位合并替代已观测的完整事件路径。

Tardis 接收 UTC 是供应商的观测时钟，不是本机接收时钟；缺失的本机单调时钟、
连接身份和连接开关事实不得通过下载时间、行号或虚构值填入。
来源中可证明的排序、快照桥接和有效区间是归档事实，不冒充本机物理连接覆盖。
缺口必须切段，恢复后的快照不能倒填缺失盘口，标签及订单路径不能跨未知段假装连续。
缺口期间的未知成交和持仓变化不能默认无损；连续净收益所需覆盖不足仍为 `BLOCKED`。

### 价格坐标与事件顺序

规范盘口使用 `price_unit` 和整数 `price_index` 作为无损价格坐标，
它们不表示交易所最小下单增量。官方原生 `tick_size` 及其来源保留，
在解码边界映射为价格单位；Tardis 按同一输入日、同一资产的全部执行价格
（快照、深度更新及聚合成交）的原始十进制精度确定固定单位，不能逐段改变。
精度推导不经过浮点或受 Decimal context 影响的四舍五入；标记价格不参与盘口坐标推导。
归档集合按本次已选原件、每资产取最细单位，metadata 与交给共享 reader 的 record 使用同一单位；
开发阶段不利用尚未选入的留出原件精度。每资产校准摘要表的 `price_unit` 是历史整数样本和事件的唯一单位声明，
同一产物全部 fold/side 必须一致，并由成员哈希与校准来源身份绑定。留出扩窗后只在消费视图中
用 Fraction 的精确整数倍率把历史 origin/book/trade 转到当前单位，再比较当前订单、状态和待生效指令；冻结文件保持不变。
执行模型比较实际观察到的价格，不以“增加一个交易所最小跳价”构造订单。
需要真实下单规则的其他能力仍须单独证明历史规则，不能将表示单位冒认为规则。

规范事件使用 `source_sequence`、`observation_id` 与原始接收 UTC。
官方相同 UTC 的事件仍按真实 `receive_monotonic_ns`、再按原生序号排序；
Tardis 相同 UTC 按切片及原生行序排序，`receive_monotonic_ns` 保持缺失。
原生成交按 symbol 与聚合成交 ID 核对重发：仅事件时间 E 改变而其余字段相同的消息保留原字节，
研究回放只采用首次观察，不重复计量成交，也不回退连续性高水位。未见过的倒序 ID 或成交字段冲突明确失败；
身份摘要用于精确去重，不参与资源准入。
观察段标识只表示来源可证明的区间，不冒充物理连接。
PANEL 的 `observed_utc`、`book_source_sequence`、`book_observation_id` 及
成交行序范围保存该身份；单调时钟与交易时钟 `transaction_time_ms` 统一以 nullable
Int64 持久化。Spot 深度事件没有交易时钟时保持 NULL，不用事件时钟或接收时钟替代。
PANEL 生产者直接输出普通 DataFrame，由唯一 dataset writer 写入，不使用所有权握手。

晚到快照之前的原生深度按观察顺序保留，快照到达后只用其推进共用重建器，
不输出历史 BookState、不提前建立覆盖。只有快照后实际收到并成功应用的深度
开始可消费覆盖；无法桥接的记录保持无效。原始接收 UTC 永不被快照时点改写。

## 特征与标签

五个因子族分别为 microstructure、derivatives_state、spot_perp、cross_asset 和 external_price_discovery；
现货/永续基差、跨资产领先滞后和 Bybit/OKX 外部价格发现均通过显式研究配置选择。

### 微观结构因子

基础 14 项微观结构因子来自同一持久化 PANEL：
spread、relative_spread、imbalance_1/5/10、microprice、microprice_deviation、
ofi_100ms、signed_trade_flow_100ms、taker_buy_volume_100ms、taker_sell_volume_100ms、
return_100ms、realized_volatility_1s、top10_depth_per_spread。
微观结构因子另提供多尺度 TFI、OFI、深度归一 OFI、深度压力与不平衡差值，另有共享的 return_500ms，族内共 30 项。
连续完整窗口在相同资产、segment 内计算，frame 边界保留滚动状态，缺口清空状态。

`book_spread` 在 PANEL 的整数价格 owner 中以
`(ask_index - bid_index) × price_unit` 计算并持久化。
FACTOR 直接消费该价差；relative_spread 和 top10_depth_per_spread 共用同一事实，
不重新相减绝对浮点报价。microprice_deviation 按
`spread × imbalance_1 / 2` 计算，microprice 为
`mid + microprice_deviation`，避免相等队列的零偏移被绝对价格抵消误差污染。

`imbalance_1/5/10` 对既有有限 float64 深度按精确有理数计算 `(bid-ask)/(bid+ask)`，
只在输出时舍入；比例不变不能因分子或分母的中间舍入变成虚假变化。
`microprice_deviation` 复用同一最优档不平衡比例。
这不恢复未保留的供应商十进制数量，也不以 epsilon、取整或全局阈值抹除真实微变。

盘口中间价在 PANEL 和执行路径都按整数价格坐标之和乘原生 price_unit 后除二，
仅在最终结果转换为 float64。交叉盘口按整数坐标判断，不把浮点舍入相等的
两个不同合法价位判为交叉，也不先把两倍中间价转换为浮点。

### 衍生品状态因子

衍生品状态由独立 derivatives-state producer 消费 native state 与 Binance 100ms PANEL，
在 producer 内按 observed_at <= decision_time 对齐，再交给精确主键 assembler。
状态分别保留 event_at、observed_at、age_ms；age 上限显式写在配置并进入 processed 规格身份。
OI 使用 base 数量，变化为当前 OI/窗口起点 OI-1，整个窗口须有效；
funding 为原始小数费率，time_to_next_funding 和 time_since_observed_force_order 为秒，
basis 为无量纲比值减一。频道名 premiumIndex 不代表供应商存在同名经济字段。

forceOrder 仅代表供应商观测到的 proxy。每条发布快照以 z（累计已成交 base 数量）加权，
BUY 为正、SELL 为负；不以订单数量 q 代替成交数量，不声称完整或去重后的全市场清算量。
无观测或分母为零保持 NULL；断连 generation、grid gap 和 episode 边界重置窗口。
执行 PANEL 的 segment/grid 边界只重置滚动窗口及预热；独立衍生品源的最新观测和观测时钟仍按自身 age 上限保留，只有该源 epoch 变化才失效。forceOrder elapsed 由同一最新观测时钟派生，窗口首个100ms只纳入该区间内的已观测成交。
双边深度和滚动流量比值在最终除法后舍入；请求的绝对 OFI 若确实超出 float64 表示范围仍明确失败，不改成零或缺失值。
缺 state 归档只阻塞依赖它的分支，微观结构因子不引入该依赖。

### 现货与永续上下文

Spot panel 复用同一个 native 回放与 panel producer，逐资产保留真实观测覆盖；
Spot/Perp context producer 在执行 panel 的主键上输出，assembler 只做精确键连接。
现货和永续共用微观结构数学核。Spot basis 为 spot_mid/perp_mid-1，收益为已成熟的
100/500ms 简单收益；OFI 以相同币种的 base 数量计，跨市场 OFI divergence 使用深度归一值。
TFI 为有符号成交量/买卖成交量之和。lagged 字段取上一格连续 Spot 100ms 信息，
不能跨 Spot segment/gap；上下游 frame 切分不改变结果。

配置 spot.max_age_ms 同时限制 Spot grid age 与实际 book observation age，
两种 age 及对应时间均保留。源不存在则依赖分支 BLOCKED；
源存在但局部缺口、过期或窗口不完整则在相同执行主键上保留 NULL，不无限前填、不补零。
现货上下文不改变 Binance BTC/ETH perpetual 的执行范围，也不把供应商观察顺序解释为共址交易延迟。

### 跨资产上下文

跨资产研究的读取行范围、模型训练/评估资产、信息资产与执行资产分别绑定。
cross_btc_eth.json 不依赖 SOL；cross_asset.json 显式加入 SOL native 归档及
SOL→BTC、BTC/ETH→SOL 的迁移选择。只有训练或评估需要 SOL 行时，研究基表才读取三资产信息 panel；
仅把 SOL 用作辅助信息时，研究基表仍为 BTC/ETH。任何配置均不能把 SOL 加入执行组合。
三资产信息 panel 按时间和资产有界归并两个输入的完整并集；保留各自独有时间、原始分段
与 nullable dtype，不按共同时间取交集，也不按资产串接造成跨批时间回退。

三资产共用 native panel、微观结构因子、完整窗口标签和模型生产器。
跨资产 context 只读取持久化的成熟 100/500ms 收益、深度归一 OFI 和 TFI；
来源资产的数值及上一格连续 100ms 的滞后值先在 producer 内按观测 grid 对齐，
记录各来源 grid 时间与 age。缺口、过期或不完整窗口保持 NULL。
同一个数学核负责 500ms 收益，不在 Spot、SOL 或跨资产层复制公式。
只依赖 BTC/ETH 的配置不引入 SOL 数据节点；选择 SOL 后缺归档的依赖分支保留 BLOCKED。

### 外部价格发现上下文

`configs/research/crypto/external_context.json` 显式选择 Bybit 与 OKX 信息因子。
两个 venue 分别拥有 processed target；未配置或缺失其中一个来源只阻塞依赖它的分支。
来源使用已有 Tardis normalized owner 的 quotes、trades，
由 producer 按 `local_timestamp <= decision_time` 对齐到 Binance BTC/ETH 100ms 主键，
随后使用现有 exact-key assembler。SOL 研究行对这些 BTC/ETH 专属信息保持 NULL。

成交 amount 的原生单位不得混用。Bybit linear 为 base 数量。OKX BTC-USDT-SWAP
旧期每张为 0.0001 BTC，ETH-USDT-SWAP 为 0.001 ETH；2020-03-20（BTC）与
2020-03-19（ETH）香港时间 16:00–16:30 调整至 0.01 BTC 与 0.1 ETH。
该半小时内的精确生效点未取得保证，相关成交明确阻塞；其他日期按成交事件时点选择已核面值。
换算事实由市场 owner 定义，
冻结进入 processed 参数身份。来源说明：
[Tardis CSV 字段](https://docs.tardis.dev/downloadable-csv-files/data-types)、
[OKX 合约面值公告](https://www.okx.com/zh-hans/help/adjustment-of-face-value-for-usdt-margined-perpetual-swap-futures-trading)。

每个 venue 输出 mid/spread（USDT）、100/500ms 简单收益、500ms observed signed
base/USDT 流量与 TFI、相对 Binance mid 的差值、两市场 mid 的相对标准差
`abs(external_mid-binance_mid)/(external_mid+binance_mid)`。
quote 的 observed/event 时间和 age，以及 trade 的对应时钟均保留为 context。
收益需要连续且未过期的完整窗口；流量需要完整 500ms 决策窗口，没有观察到成交时为 NULL。
这些是供应商观察样本，不声称完整交易流或共址延迟。九个价格发现因子不消费 ticker，
因此不要求该归档；已有 ticker 原件与 normalized 通用读取能力保留。
执行仍仅 Binance BTC/ETH perpetual。

### 标签与成熟时点

Crypto label由统一LabelDefinition/LabelCatalog拥有，horizon单位为milliseconds，
available_at、模型期限、评估binding与processed label合同由它派生。
maker报价期限由CryptoMakerPlan拥有；它与模型预测期限和markout时钟不同。

价格标签为100/500/1000/5000ms的future_mid_return和future_mid_direction；
各模型显式选择标签；baseline.json 的价格模型选择500ms收益，另有 maker_bid_sell_pressure_proxy_500ms。
每个标签有独立 available_at，短期限必须由实际事件分辨率和观测覆盖支持。

## 研究配置与模型

### 配置与数据角色

baseline.json 保存九个模型选择作为可编辑示例；运行不强制保留这九项或基础 14 项输入。
评估因子、各模型输入、训练和评估资产分别显式选择。multiscale.json 选择多尺度微观结构因子，
derivatives_state.json 加入衍生品状态因子；不做块或参数的笛卡尔搜索。
示例中的费用、延迟和资金为研究假设，不能当成已验证的历史账户费率。
数值监督模型的 scaler 只在训练期拟合；`weighted_mean_offset` 每列统一保存最简整数比 `[分子, 分母]`（正分母），极小中心偏移保留到最终标准化值舍入。旧浮点偏移 metadata 与旧 pickle 明确拒绝，受影响模型须重建；配方变化登记新的模型/试验身份，既有失败结果不覆盖。
真实研究须有合格的事件来源与资金费归档；代码验证或合成验收不能替代真实覆盖。

| 配置示例 | 选择的研究信息 | 附加来源条件 |
|---|---|---|
| [baseline.json](../configs/research/crypto/baseline.json) | 基础微观结构因子及九个示例模型 | Binance BTC/ETH 主来源 |
| [multiscale.json](../configs/research/crypto/multiscale.json) | 多尺度微观结构因子 | 同一主来源 |
| [derivatives_state.json](../configs/research/crypto/derivatives_state.json) | 微观结构与衍生品状态 | 独立 native state 归档 |
| [spot_context.json](../configs/research/crypto/spot_context.json) | 微观结构与现货/永续上下文 | Spot 深度、快照及成交归档 |
| [cross_btc_eth.json](../configs/research/crypto/cross_btc_eth.json) | BTC/ETH 跨资产上下文 | 不依赖 SOL |
| [cross_asset.json](../configs/research/crypto/cross_asset.json) | BTC/ETH/SOL 跨资产上下文及显式迁移选择 | SOL native 归档 |
| [external_context.json](../configs/research/crypto/external_context.json) | Bybit/OKX 外部价格发现 | 各场所 quotes、trades 归档 |

研究计划必须显式给定各 fold 的 train/validation、唯一锁定 test、初始资金、估值币种和暴露上限；
不为适应短录制而缩窗，不随机打乱事件行，不用排序掩盖同时间戳顺序歧义。
训练资格由成熟标签和拟合截止时间决定。额外隔离间隔若必要，属于显式实验语义，
不是第二套通用机械purge规则。特征只消费决策时已知事实，未来盘口/成交和未来派生分类不得回填。

单资产跨时间评估采用单资产时序轴；多资产横截面评估只在合法横截面上计算。
模型 trial 的验证轴由该模型的批准 `training_asset_ids` 决定：单资产训练范围在成熟验证
时间序列上计算 Pearson IC，RankIC 明确为 `not_applicable`；多资产训练范围保持横截面轴，
不会因偶发缺失只剩一资产而改轴。`evaluation_asset_ids` 不决定训练验证轴。
冻结模型的 `sample_filter_mode` 同时约束配置预测规格和最终 OOS 预测；显式 `unfiltered`
保留输入完整但不符合 eligible 的预测行。监督训练与验证仍须满足原有 eligible、成熟及输入合法性。
模型统计、OOS预测、费用/价差/延迟敏感性与经济账本分别报告。
概率模型若未来采用，必须检查校准，不能只看准确率。

数据角色只有 train、validation、test；development 是执行访问阶段。
每个开发 fold 在 train 内拟合模型并产生 validation 预测，maker calibration 只消费
该 validation 窗内截至其截止时点已成熟的结果。Crypto 不执行 parameter_candidates 自动搜索。
开发最后一个 fold 结束后，在其真实 train 与 validation 两个窗口上进行最终 refit；
两个窗口保持独立，不能用包络补齐中间空白，也不构造重叠或虚假的 fold。
development_run_id 必须指向已完成且参数完全一致的开发运行；
最终 test 复用冻结模型及 maker calibration，只构建新的预测、评估、账本、统计和报告，
并保存 holdout_evidence；禁止在 test 训练、选模型、调参或重新校准。

### 数值缩放与排名模型标定

Crypto 不使用 A 股横截面 canonical transform。数值监督模型使用 training-only ModelMatrixScaler；
排名集成直接使用排名输入，训练得到的方向或权重及标签单位标定随模型冻结。

排名模型仍发布原始无量纲排名分数。其训练同时以同一合法训练样本及权重拟合
label = intercept + slope * score，将两个系数绑定于冻结模型状态和元数据；
常量分数采用训练标签均值。taker 和 maker 的价格/proxy 输入在经济执行入口统一
转换为标签单位，maker 的验证期校准消费对应 TRAIN 模型的转换；最终执行消费
TRAIN+VALIDATION 正式重拟合后的冻结转换。验证评分及测试执行只读取已冻结系数，不以当前待评分窗口重估。

### 模型标签、maker 角色与预测期限

baseline.json 的价格分支在同一个 500 ms 收益目标上比较线性回归、LightGBM 和 XGBoost。
执行质量分支预测明确命名的 `maker_bid_sell_pressure_proxy_500ms`：
未来全部已观测卖出主动成交量除以当前展示的最优买档数量，并裁剪到 `[0, 1]`。
分子包含未来每个价格上观测到的卖出主动成交量；PANEL 不提供按价格分布的成交量，
因此不声称这些交易发生在当前买价。它以预测分数 `0.5` 为阈值控制 maker 报价日程，
与价格预测及训练/验证期冻结的成交量和收益残差配对校准共同决定 maker 数量，
并将该运行与固定报价 maker 诊断基线并列报告。
这一公开 L2 代理量不是成交概率或 MBO 队列位置，也不承诺可执行成交。

maker 的价格与压力模型角色由配置中的模型 ID 指定，不按算法名称或固定 ID 识别。
两个角色均为空表示未校准 maker；否则必须同时指定已选模型，并分别使用
500ms mid return 与 500ms observed sell-pressure proxy 标签。
该资产必须同时在模型的训练/验证与 OOS 预测范围内。
全部所选模型均保存验证预测并在读取时校验；maker 按配置引用其中的模型角色，模型改名不改变角色。

模型预测期限由所选 executable label 决定，并写入每个 model scope。
taker 只接受 future-mid-return 模型，其持有/预测期限读取该模型 scope；
direction 或压力 proxy 不得被当作收益率使用。
模型 refit 与验证产物的成熟时间校验也读取各自标签期限。

### 共同样本外评估

模型直接比较复用 evaluation catalog 与既有 prediction lookup。
每个与 `comparison_model_id` 使用同一标签的模型产生一个成对共同 OOS 评估，
DAG 先要求双方各自 OOS 评估通过完整预测覆盖校验，再按精确主键取有限预测交集，
并使用同一 fold 的标签成熟截止时间。资产或有效时间范围没有交集时，
发布零行比较产物，时间覆盖边界为 NULL，报告记为 UNDEFINED；不制造样本或零相关系数，
也不把没有共同样本误判为整个研究运行失败。
报告分别展示各模型完整评估与共同样本评估；不裁剪预测文件或交易账本，
不同标签/预测单位的模型不直接比较。该过程不是因子笛卡尔搜索，也不新增回测引擎。

## 执行、资金费与账本

### 候选、数量与成本

组合配置通过显式字段定义候选。taker 给出
`candidate_id/asset_id/model_id/latency_ms`，maker 给出
`candidate_id/asset_id/price_model_id/proxy_model_id`；执行资产仍限定 BTC/ETH。
DAG、producer binding、组合统计均由同一份解析后的选择派生。
校准 maker 的 target 随 candidate 身份生成，采样、预测与重放使用该 candidate 的资产；
报告逐组合列出校准证据，不能拿 BTC 的校准产物用于 ETH。

在硬暴露约束内，按决策时信息比较合法数量与行动，最大化计入适用成本后的预期净收益。
新增风险且零动作可行时，将零纳入比较；所有增量行动的预期净收益均不大于零时不新增交易。
已有挂单按真实剩余数量、队列、TTL 及指令延迟比较 KEEP、取消与重挂；取消不等于立即零损益，
预期净收益为负的 KEEP 仍可能优于损失更大的取消。成本可非线性，简单截断数量不自动是最优解。
定量只能用决策时订单簿，不能用延迟生效/成交时未来盘口或已实现PnL重新选量；
生效后仍允许真实部分成交。负资金费及收入保留符号，已反映在成交价格中的价差/冲击不能再扣一次。

Crypto 研究计划的 `execution` 显式提供 `taker_fee_bps`、`latency_ms`、
`markout_horizon_ms`、`latency_scenarios_ms`、`fee_scenarios_bps` 和
`additional_spread_scenarios_bps`；不填入隐含的零费用或零延迟。
`latency_scenarios_ms` 枚举使用计划默认延迟的 taker 候选，默认场景为 `[0, 100, 200]` ms。
保留显式候选延迟；按资产、模型、实际延迟去除与基准、显式候选和已派生场景重复的派生项。
新增场景复用同一候选、执行目标、账本、统计与报告链，实际延迟进入各自持久化身份。
maker 继续使用独立的 `maker.latency_ms` 及同一时钟下的冻结校准。资金和仓位上限仍来自同一计划的
`initial_capital`、`valuation_currency` 与 `exposure_limits`。

### 资金费来源、预测与结算

Tardis 市场事件只消费该日自身的深度与成交，不以官方事件补洞。
同一 Binance venue 的结算数据作为独立附属输入，明确区分官方 `fundingRate` 原件与
[XOOMAR 历史结算记录](https://xoomar.com/markets/api/funding-rates)。两种来源均原样保存完整响应至
资金费 symbol 原子目录中的 `settlements.json`，不以自行抽取的记录或上游哈希替代原响应。
`settlement_source` 绑定供应商、endpoint、symbol、请求窗、HTTP 状态、原字节哈希、实际 limit、
响应数、窗口内结算数及从响应推导的完成依据；importer 与 reader 复用同一校验责任模块。

[Binance fundingRate](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
按请求 limit 截断，默认 100、最多 1000；离线导入必须显式提供实际有效 `request_limit`。
单页导入只在响应数小于实际 limit 时认证请求窗，满页保持覆盖未证，不从小于 1000 推定。
XOOMAR 的 `data.history` 最多 5000 条、跨场所并按时间倒序；原响应必须在正确 Binance/资产
子集上同时给出请求起点之前与末端之后的真实结算，才认证中间窗口。服务 `updatedAt` 不充当
结算完整性水位。不能证明完成的输入明确 BLOCKED，保留已有产物，不增加自动分页或补洞。
其 `observations/` 子目录
仍只含严格 CSV 与元数据两个成员。规范元数据的 `settlement_source` 显式说明原件身份，
既有显式规范归档没有此原生附属成员时为 `null`，不从缺字段静默推断。
历史 `markPrice` 的 `r`/`p`/`T` 仅产生当时供应商接收 UTC 可见的预测，
不是未来最终结算；已过期的目标时点不能产生新的已知未来覆盖。
两者投影到既有资金费列与 `FundingObservations`，保持预测和实际记账责任分离。

`sources/binance/funding.py` 读取录制目录同级的
`funding/<symbol>/response.csv` 与 `response.metadata.json`，复用原始快照存储合同。
列为 `kind,time_ns,known_until_ns,rate,mark_price,cashflows`。
元数据必须声明 `schema_id=qrp-binance-funding-observations`、证券、期货市场、
`USDT`、带起止纳秒的 `coverage` 和非空来源依据 `provenance`。
连续净收益账本要求 `coverage.status=COMPLETE` 且覆盖全部 OOS/空档区间；
独立分段按每段所需结算覆盖验证，不把段外未知资金费解释为零。
缺文件或 `UNPROVEN` 只阻塞依赖该证券资金费的净收益子图；数据、模型和评估可继续。

`settlement` 记录用 `time_ns,rate,mark_price` 结算实际现金；
`forecast` 记录的 `time_ns` 是真实可得时间，`known_until_ns` 是已知覆盖末端，
`cashflows` 是按时间严格递增的 `[event_ns, signed_USDT_per_long_asset_unit]`。
两类记录的非适用字段为 `null`。预测值为 `null` 表示该笔预期现金流未知。
只有决策时已经可得、覆盖预测终点且相关现金流均已知的 forecast 才参与增量下单。
实际结算不能反向影响下单量；显式完整覆盖内的空事件表才能表示无结算，
没有输入不能当作零资金费。本模块消费外部归档，不新增联网采集端点。

### 连续与独立分段执行

研究窗口由连续UTC时点定义，午夜、文件块或普通fold交界不是账户重置边界。
execution.backtest_modes显式选择continuous、independent_segments中的非空不重复集合，
规范顺序为continuous在前。无execution的evaluation-only配置不要求此字段。
连续模式覆盖不足不自动退化；独立分段在每个最大合法连续区间从相同初始资本、空仓、
无挂单开始，复用同一执行引擎和原冻结模型，不在新段重新拟合。不得跨缺口拼连续收益。

正式连续账本要求从首个 OOS 开始至最后一个 OOS 结束始终位于一个已证明有效的盘口段，
包括折间空档；同一区间还须有唯一完整的成交观测覆盖，因为 taker 与 maker 的决策
都消费成交特征。不足时仅阻塞相应账本
及其下游，不删改训练样本门槛，不把未知期解释为无成交或提前取消。
独立分段按实际candidate所需盘口、成交和结算覆盖，在首OOS至末OOS的时间包络内求最大合法段；
决策窗口仍是原OOS窗口并集，折间空档保持合法估值/持仓/资金费但不产生新信号。
事先已知计划末端与事后观察截断严格区分：后者不能提前改变数量、TTL、KEEP、撤单或markout目标。
截断时仅保留已观察事件；待生效指令、活跃挂单和未成熟markout记截断/未知，不制造未来成交、
取消成功、强平或费用。最后可靠价格估值不等于平仓。资金费在同刻新成交前按旧仓位结算；
新段空仓不承担旧段资金费；观察stop为半开边界，事件恰好在stop不属于前段。
每段使用同一四表ledger；父semantic artifact的episodes.parquet只保存边界、原因与精确子引用，
所有子账本验证完成才原子发布父集合。失败或取消保留旧好产物，不发布半个COMPLETE集合。
报告逐段展示收益/估值、回撤与持仓、成本、覆盖及截断状态；汇总仅为覆盖/缺失范围和
同币种观察段内成本，不输出跨段总体收益、连续最大回撤、Sharpe、假年化或收益均值。
统计按candidate与mode各自依赖结果，连续BLOCKED不阻塞合格的分段结果。
执行模式在现有配置的 `research_plan.execution.backtest_modes` 中选择 `continuous`、
`independent_segments`，可以同时请求；共用同一模型和预测目标。
连续执行目标为 `{candidate_id}_ledger`，分段为 `{candidate_id}_episodes`，
各自统计为 `{candidate_id}_{mode}_statistics`，不存在所有候选共用的组合统计依赖。
统计行携带 `execution_mode`、`episode_id`；`continuous` 表示连续账本，真实独立段使用段身份，
`all` 仅表示覆盖、段数、观察段费用/资金费、成交数量和turnover汇总。
`crypto_portfolio_coverage.parquet` 从研究时间包络和已验证段索引派生半开覆盖/缺失区间，
未覆盖原因保持 `not_covered_by_qualified_episode`，不另做一次资格判定。
同资产只有段边界及执行假设一致时才可直接并列比较；覆盖不同明确
`different_coverage_not_comparable`，不按段序号配对或重拼现账本。
段间只重置经济状态，不抹掉当时合法已知forecast；价格特征窗口只由真实行情gap按原规则重置。

continuous 模式中，每个候选跨 OOS 窗口使用连续独立资金。窗口边界先处理已到时的市场事件、
资金费和生效订单，再记录边界；窗口结束取消待生效订单。空档期间不发出新信号订单，
持仓继续按实际观测价格估值和结算资金费。OOS 与空档经济结果从同一个已发布账本的
有序边界记录投影；未来盘口不能补作过去的边界估值。

### Maker 校准、队列与报价期限

`execution.maker` 显式提供 `maker_fee_bps`、`latency_ms`、
`min_calibration_samples`、`inventory_target` 和 `inventory_skew`。
`inventory_target` 和 `inventory_skew` 仅用于 fixed maker；calibrated maker 按历史净收益选量，
其计算身份和结果摘要不包含这两个未消费参数，完整配方 provenance 保留原声明。
校准使用 TRAIN 拟合后产生、在 VALIDATION 截止前成熟的价格模型和 proxy 预测；
按买卖方向保留每个样本的成交路径与价格预测残差，包括未成交样本。
预期收益计算保留成交与残差的联合关系，不把均值相乘代替配对结果。
任一必需方向样本不足只阻塞校准 maker，固定报价诊断与 taker 可继续。

maker 报价采用公开 L2 队列近似和显式延迟。生效时会穿价的 post-only 报价取消，
不会改为 taker 或读取未来盘口重新选量。KEEP 固定旧剩余数量、原队列与真实剩余期限；
真实取消重挂作为独立动作，可同价同量，但重置队列并承担显式延迟。
两者与零/取消动作按实际预期净收益比较，不把 KEEP 偷换成续期。

已生效挂单随后遇到同方向、同观察连接且覆盖合法的穿价成交时，只有已知且正的
`visible_quantity` 才触发全剩余量成交，按自身挂单价记账，不受该笔可见量封顶。
同价成交先按可见量消耗估计前方队列，再用余下可见量部分成交；零可见量既不成交
也不推进队列，未知可见量明确 BLOCKED。买卖方向使用同一规则，校准与正式执行共用 owner。
这属于固定历史路径上的小单无市场冲击假设，不是历史成交量对任意订单容量的证明。
`max_order_quantity` 与 `max_position_quantity` 只约束下单和仓位，不证明订单已足够小；
不能据此将大单模拟成交视为可实现容量。该规则与限制沿既有 `queue_assumption` /
`execution_summary` 写入账本，再由统计和报告原样读取。

每张报价最晚在其自身决策后 500ms 到期；未来盘口或成交覆盖的末端不能预先缩短
该期限或影响 KEEP、重挂、数量及预期收益。报价决策只消费当时已观测到的缺口；
事后证明的无效覆盖用于研究资格，不能冒充当时已知的取消时刻。
校准样本的完整结果覆盖只用于训练期已成熟样本资格。

已知 OOS 结束边界取消全部未完成订单。该预测期限与成交后 markout 时钟独立。

maker 使用独立的 500ms 报价/校准期限，不把这个期限强加给其他模型。
报价期限从决策时刻计时，延迟生效不会顺延截止时间，保留原挂单也不会续期；
OOS 窗口结束可以提前截断报价。该期限不是库存强制平仓时间，markout 期限独立配置。

校准样本在既有捕获生命周期内保存初始买/卖 `price_index`，读取时与其确切初始盘口事件
核对。OOS decision 复用这两个已验证原点，不为每次 KEEP/替换重放重读初始盘口。
当前订单数量、队列、待执行命令和期限仍决定各样本响应，不能将决策相关的响应
冻结为统一成交率。每个确切 fold 的校准对象在当前消费作用域内，只读取并编码一次
已验证样本与事件；买卖方向共用唯一事件历史，沿既有 event_id 和窗口坐标定位。
fold 的经济选择使用普通列表保存样本与事件，按 event_id 直接定位；样本保留方向和配对残差。
这些对象只服务本次计算，不创建长期 cache、新公开格式或另一份业务真值。
每次决策仍重算订单、队列、待执行命令、期限、费用及 funding 的精确响应，不复用经济结果。
读取结束（包括提前退出和失败）复核原 artifact 的精确身份并关闭实际文件句柄；
次生验证或清理失败不得覆盖主异常。
maker/taker 的待成熟 markout 使用普通 FIFO 队列，结果成熟后移出队列。
PANEL 与 markout 的时间采样均使用该纳秒内全部已观测盘口消息处理后的最终状态；
markout 在时钟越过目标纳秒或确认该时刻事件已耗尽后结算，不在第一条同刻盘口上结算。
实际成交、报价到期、撤换指令、资金费和断线边界继续遵循现有事件顺序；诊断取样不重排成交。
未观测到的未来时刻、覆盖缺口与段边界仍返回未知，不用尾部盘口补造未来 markout。

## 统计、报告与只读展示

固定收益/成本算法由实现和本合同定义：完整UTC日收益、365日年化、ddof=1、零收益cash基准、
真实gap的独立经济贡献、固定实际turnover的附加费用敏感性，不作为并行active manifest旋钮。

正式日频收益仅使用完整落在同一 OOS 折及同一连续账本或独立段内的 UTC 日，以实际连续 NAV 为分母，
按 365 日年化、现金零收益参考和样本标准差计算。计划内完整日若缺失边界估值，
正式日频均值与年化指标全部 `UNDEFINED`，不会只选剩余有效日。
细粒度事件 NAV、连续回撤、库存、费用、资金费和空档贡献仍保留于账本及统计产物。
报告仅从指定 `run_id` 的确切引用读取数据、模型、校准、账本和统计，
缺失子图保留 `BLOCKED`/`UNDEFINED`，运行资源观测注明按目标边界采样。

统计行按 candidate、execution_mode 和 episode_id 保留范围；独立段不拼接为连续收益。
报告展示数据覆盖、因子诊断、模型与验证、OOS 预测、共同样本比较、maker 校准、组合、执行、
费用敏感性、库存、日频统计、development/holdout 证据、readiness/blockers 和运行资源观测。
Crypto 的日频统计为经济描述，不发布 SPA、StepM、MCS、PSR、DSR 或 PBO 推断矩阵。

[只读研究 Notebook](../notebooks/research-interview-demo.ipynb) 通过明确的 profile、run_id 和 report_identity 读取已发布产物，
使用同一查询入口显示持久化统计与账本曲线。连续曲线和各独立段分别展示，Crypto 时间轴使用 UTC；
完整日不足时保留原指标的 UNDEFINED 原因，不从展示曲线重新计算年化，也不在 Notebook 内训练或发布。

## 研究操作与证据验收

研究操作沿以下依赖链执行：

```text
显式选择的录制运行或历史归档
-> 确定性的原生 L2 事件
-> 持久化 100ms PANEL 充分统计量
-> PANEL 派生的微观结构 FACTOR / LABEL 与按配置对齐的辅助信息因子
-> 标准 ResearchDataset
-> 标准评估 / 评估产物
-> 按 train/validation 拟合与冻结的 ModelSpec 及 test 持久化预测
-> 每个模型的样本外预测/目标评估
-> 配置所选 taker/maker 的 continuous 账本或 independent_segments 子账本集合
-> 所选候选统计产物与指定 run_id 报告
```

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 选择并验证BTC/ETH期货录制或历史归档 | 同时覆盖所选两币种的快照/增量/成交、分段和时间质量；保留原始字节、来源身份与清单 |
| 2 | 声明资金费与研究计划 | 实际结算及决策时可得预测分别有来源与覆盖；训练、验证、OOS、留出、资金与费用全部显式 |
| 3 | 验证真实回放与时间语义 | 无伪造缺口修复，无未来盘口回填，标签成熟与分段边界可手算复核 |
| 4 | 比较简单经济基线 | 分开报告固定报价maker、校准maker及taker的费用、库存、markout、净收益和空档贡献 |
| 5 | 按实际覆盖解释统计与容量结果 | UTC完整日覆盖真实，不挑删缺估值日期；年化须满足完整日条件，长期资源/性能及容量不能从短合成结果外推 |

录制和研究分开操作，不因读取配置自动启动网络采集或解锁留出。
新增信息场所、资产或模型需要相对简单基线的增量证据；辅助信息范围不扩大执行场所和资产。

## 高级研究候选

通用 robust/log/winsor 模型预处理属于高级研究候选；准入要求 train-fit 并随 model 冻结。
这与因子定义中已有的 log_open_interest 计算不同。

Survival/Hawkes/DeepLOB 及注意力模型属于高级研究候选，不属于正式模型合同。
需要对应真实数据和相对简单基线的增量OOS经济证据，仍复用现有DAG、试验、账本和报告，
不预留第二套平台，也不从价格变化虚构精确清算量或L3事实。

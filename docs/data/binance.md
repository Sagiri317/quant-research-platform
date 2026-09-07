# Binance 数据合同

本文描述 Binance 公开市场数据在 QRP 中的当前数据合同。
本文只记录结构和层边界，不记录真实载荷、行级样本、运行结果或本地路径。

## 规范本地封装

规范本地封装导入器只接受 JSONL。每行必须是规范封装，
所有 `data_kind` 共同必需的顶层字段为：

```text
source
market_type
data_kind
symbol
captured_at
payload
```

`source` 必须是 `binance`，`market_type` 必须是 `spot`。
`captured_at` 必须是带 `Z` 或 `+/-HH:MM` UTC 偏移的 ISO 时间戳字符串；
数值时间戳和没有时区偏移的字符串不是当前合同。当前 `data_kind` 只允许：

```text
klines
agg_trades
depth
```

条件字段：

- `klines`：`interval` 必填且必须为 `1m`。
- `agg_trades` / `depth`：`interval` 必须缺省。

严格标量规则：

- 数值/整数字段拒绝 `bool`、`NaN`、`inf`、`-inf` 和非数值。
- 整数字段拒绝小数值。
- 布尔字段只接受 JSON 布尔值，不接受字符串或数值别名。
- `captured_at`、K 线的 `open_time`/`close_time`，以及聚合成交的
  `timestamp` / `T`，使用共享的 Binance 严格时间戳校验器。
  `captured_at` 必须是带 UTC 偏移的 ISO 时间戳；市场载荷时间必须是非负毫秒时间戳。
- `1m` K 线的 `open_time` 必须整分钟对齐，`close_time` 必须严格等于
  `open_time + 59_999ms`。名义为 `1m` 但时间窗口不闭合的载荷，由原始数据责任模块明确拒绝。
- 聚合成交的 `timestamp` / `T` 在原始数据责任模块校验，
  不把非法时间延迟到中间层标准化时处理。

Binance 市场载荷的毫秒时间戳还必须落在 QRP 下游可安全表示的 pandas 纳秒时间戳范围内。
上界由 `pd.Timestamp.max` 推导；纪元零值、上界内的值返回带 UTC 时区的纳秒 `Timestamp`，
首个越界值和更大的整数都由 Binance 来源责任模块明确拒绝。
原始 K 线、原始聚合成交与中间层标准化复用同一范围合同。

`klines` 的 `payload` 是连接器返回的 K 线数组，允许连接器追加额外字段；
当前语义校验只读取已批准字段。`agg_trades` 和 `depth` 的 `payload`
是连接器返回的对象。裸载荷、非封装 JSON、非 JSONL 文件和不支持的数据种类都不是当前合同。

原始封装结构不因语义校验而改变。当前原始校验会检查 K 线时间窗口、
OHLC、成交量、成交笔数、聚合成交的时间戳/价格/数量/标识范围，以及深度的价格/数量。
K 线语义校验还要求 `taker_buy_volume <= volume`、
`taker_buy_quote_volume <= quote_volume`；数值载荷拒绝 `NaN`、`inf`、`-inf` 和非数值。
双边最优买价/卖价同时存在时必须满足 bid < ask；单边深度载荷可以作为有效原始输入保留，
后续面板层通过报价观测诊断表达报价不可用。

深度标准化在原始和中间层路径复用同一个责任模块辅助函数。
`bids` 必须按价格严格递减排列，`asks` 必须按价格严格递增排列，
每个原始档位必须恰好是 `[price, quantity]`，第 1 档表示校验后的最优买价或最优卖价。
同一资产、时间戳、方向下重复的深度档位，在市场面板检查中明确拒绝。

## 原始数据布局

原始文件从以下位置发现：

```text
data/raw/binance/spot/klines/<symbol>/1m/*.jsonl
data/raw/binance/spot/agg_trades/<symbol>/*.jsonl
data/raw/binance/spot/depth/<symbol>/*.jsonl
```

这些路径仅是规范本地封装的输入布局，不定义公开 API、公开 CLI 或持久化研究输出。
该导入器路径不进行网络采集。下面独立的录制运行合同只使用无需认证的
Binance 公开市场数据端点，并且从不读取 API 密钥或秘密。

## 录制运行合同

每个录制根目录包含一个 `run.json`，以及该清单声明的 gzip JSONL 成员，
不会通过 glob 合并多个运行。`recording_scope` 为每个输出根目录选择一个独立的当前合同：

```text
futures-hft:
  USDⓈ-M Futures 差分深度 @ 100ms
  USDⓈ-M Futures 聚合成交
  -> 由当前 HFT 工作流消费

spot-l2:
  现货差分深度 @ 100ms
  -> 独立的长期 L2 采集，用于未来现货/永续研究
```

两个范围使用独立的根目录、清单和汇总状态。现货中断不能使完整的期货 HFT 运行降级或被拒绝。
现货回放、规范本地封装输入与分钟数据处理作为独立数据能力保留。
当前研究 profile 只注册 100ms 期货 HFT，不提供分钟级研究场景。
每条命令都是一个托管的数据规模操作，并在其生命周期内持有主机本地 QRP 锁，
因此同一主机上不会并发录制 `futures-hft` 和 `spot-l2`。
请顺序运行，或使用不同主机。提供的输出根目录必须不存在；录制器成功时以原子方式发布。

```text
python -m quant_research_platform.sources.binance.recorder --scope futures-hft --output-root <run-root> --duration-seconds <seconds>

python -m quant_research_platform.sources.binance.recorder --scope spot-l2 --output-root <run-root> --duration-seconds <seconds>
```

记录从命令启动时开始；未捕获的时段无法回填完整 L2。

RPI 深度、最优报价推送、强制订单、标记价格和辅助统计不属于当前录制合同，
因为已实现的研究链不消费它们。

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
缺失采集序号时，永不退回时间戳排序。

深度重连会创建新的 `connection_id` 和 `segment_id`。
匹配的 REST 快照绑定到相同标识；回放在有界本地暂存区上建立快照索引，
从不把重连分段当作连续数据。现货从 `lastUpdateId + 1` 启动；
期货从 `lastUpdateId` 启动，随后要求 `pu == previous u`。
数量为零会删除一个档位。价格标识使用整数最小价位，而不是二进制浮点相等。
缺口、交叉盘口、单边盘口或已知深度不足，都会产生明确的无效边界；
不会静默复用过期盘口。

以下时钟在整个 PANEL 来源谱系中保持独立：

```text
交易所事件时间（E）
期货交易时间（T）
本地 UTC 接收时间
本地单调接收时间
采集序号
分段标识
```

已实现的 HFT 数据链为：

```text
一份录制运行
-> 确定性的期货 L2 事件
-> 持久化 100ms PANEL 充分统计量
-> 仅从持久化 PANEL 派生 FACTOR 和 LABEL
-> 标准 ResearchDataset
-> 标准评估 / 评估产物
-> 按时间划分训练/测试的 ModelSpec 拟合与持久化预测
-> 每个模型的样本外预测/目标评估
-> 12 个 taker 与 2 个 maker 的独立连续资金账本
-> 14 候选统计产物与指定 run_id 报告
```

价格分支在同一个 500 ms 收益目标上比较线性回归、LightGBM 和 XGBoost。
执行质量分支预测明确命名的 `maker_bid_sell_pressure_proxy_500ms`：
未来全部已观测卖出主动成交量除以当前展示的最优买档数量，并裁剪到 `[0, 1]`。
分子包含未来每个价格上观测到的卖出主动成交量；PANEL 不提供按价格分布的成交量，
因此不声称这些交易发生在当前买价。它以预测分数 `0.5` 为阈值控制 maker 报价日程，
与价格预测及训练/验证期冻结的成交量和收益残差配对校准共同决定 maker 数量，
并将该运行与固定报价 maker 诊断基线并列报告。
这一公开 L2 代理量不是成交概率或 MBO 队列位置，也不承诺可执行成交。

录制器使用有界 `websockets` 运行时连接公开数据流，
使用 `httpx` 获取公开 REST 元数据/快照。它没有认证端点或下单 API。

## 中间层数据集

`sources/binance/interim.py` 负责从原始封装到中间层的标准化：

原始文件及连续标准化批次必须已经按注册的中间层主键严格有序。
标准化保留该物理顺序，并在文件与批次之间传递最后键水位；
不会通过批次内排序掩盖回退的依赖。因此，聚合成交按
`(source_asset_id, timestamp, aggregate_trade_id)` 前进，深度档位按
`(source_asset_id, timestamp, side, level)` 前进，行情柱按
`(source_asset_id, timestamp)` 前进。重复或回退的行在任何批次大小下都会同样明确失败。
每个快照内的深度按注册的方向字典序输出，而原始买卖价格顺序仍由上述来源语义检查管理。

```text
klines      -> market_bars_1m
agg_trades  -> aggregate_trades_tick
depth       -> depth_levels_tick
```

中间层物化以有界批次读取原始封装，在标准化前校验 `data_id` 与原始 `data_kind`，
对来源时间戳应用请求的起止过滤，并通过 `storage.write_dataset` 写入来源标准化数据集。
`sources/` 只负责供应商字段映射；市场语义与研究因子由
`markets/implementations/crypto/` 和 `research/` 持有。

当原始文件存在，但请求的过滤移除所有行时，中间层物化仍会为该 `data_id` 写入具有结构的空数据集。
在窄时间窗口中，聚合成交或深度等可选数据流的空结构数据集是有效的；
没有发现原始文件的 `data_id` 仍会跳过。

## 研究 DAG 的执行与资金费输入

Crypto 研究计划的 `execution` 显式提供 `taker_fee_bps`、`latency_ms`、
`markout_horizon_ms`、`latency_scenarios_ms`、`fee_scenarios_bps` 和
`additional_spread_scenarios_bps`；不填入隐含的零费用或零延迟。
延迟候选固定为 `[0, 100, 200]` ms。资金和仓位上限仍来自同一计划的
`initial_capital`、`valuation_currency` 与 `exposure_limits`。

`sources/binance/funding.py` 读取录制目录同级的
`funding/<symbol>/response.csv` 与 `response.metadata.json`，复用原始快照存储合同。
列为 `kind,time_ns,known_until_ns,rate,mark_price,cashflows`。
元数据必须声明 `schema_id=qrp-binance-funding-observations`、证券、期货市场、
`USDT`、带起止纳秒的 `coverage` 和非空来源依据 `provenance`。
只有 `coverage.status=COMPLETE` 且覆盖全部连续 OOS/空档区间时才可发布净收益账本。
缺文件或 `UNPROVEN` 只阻塞依赖该证券资金费的净收益子图；数据、模型和评估可继续。

`settlement` 记录用 `time_ns,rate,mark_price` 结算实际现金；
`forecast` 记录的 `time_ns` 是真实可得时间，`known_until_ns` 是已知覆盖末端，
`cashflows` 是按时间严格递增的 `[event_ns, signed_USDT_per_long_asset_unit]`。
两类记录的非适用字段为 `null`。预测值为 `null` 表示该笔预期现金流未知。
只有决策时已经可得、覆盖预测终点且相关现金流均已知的 forecast 才参与增量下单。
实际结算不能反向影响下单量；显式完整覆盖内的空事件表才能表示无结算，
没有输入不能当作零资金费。本模块消费外部归档，不新增联网采集端点。

每个候选跨 OOS 窗口使用连续独立资金。窗口边界先处理已到时的市场事件、
资金费和生效订单，再记录边界；窗口结束取消待生效订单。空档期间不发出新信号订单，
持仓继续按实际观测价格估值和结算资金费。OOS 与空档经济结果从同一个已发布账本的
有序边界记录投影；未来盘口不能补作过去的边界估值。

`execution.maker` 显式提供 `maker_fee_bps`、`latency_ms`、
`min_calibration_samples`、`inventory_target` 和 `inventory_skew`。
校准使用 TRAIN 拟合后产生、在 VALIDATION 截止前成熟的价格模型和 proxy 预测；
按买卖方向保留每个样本的成交路径与价格预测残差，包括未成交样本。
预期收益计算保留成交与残差的联合关系，不把均值相乘代替配对结果。
任一必需方向样本不足只阻塞校准 maker，固定报价诊断与 taker 可继续。

maker 报价采用公开 L2 队列近似和显式延迟。生效时会穿价的 post-only 报价取消，
不会改为 taker 或读取未来盘口重新选量。KEEP 固定旧剩余数量、原队列与真实剩余期限；
真实取消重挂作为独立动作，可同价同量，但重置队列并承担显式延迟。
两者与零/取消动作按实际预期净收益比较，不把 KEEP 偷换成续期。
每张报价最晚在其自身决策后 500ms 到期，
已知 OOS 结束边界取消全部未完成订单。该预测期限与成交后 markout 时钟独立。

正式日频收益仅使用完整落在同一 OOS 折内的 UTC 日，以实际连续 NAV 为分母，
按 365 日年化、现金零收益参考和样本标准差计算。计划内完整日若缺失边界估值，
正式日频均值与年化指标全部 `UNDEFINED`，不会只选剩余有效日。
细粒度事件 NAV、连续回撤、库存、费用、资金费和空档贡献仍保留于账本及统计产物。
报告仅从指定 `run_id` 的确切引用读取数据、模型、校准、账本和统计，
缺失子图保留 `BLOCKED`/`UNDEFINED`，运行资源观测注明按目标边界采样。

## 安全边界

Binance 公开市场数据在转换为 REVIEW_SAFE 摘要或生成测试样例前，仍按本地输入处理。
不要将原始连接器载荷复制到文档、报告、最终答复或 REVIEW_SAFE 证据中。

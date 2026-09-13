# Crypto 高频研究

本文件定义 Binance BTC/ETH 主线的终局方向、当前能力、短期工作和研究合同。
协议为 crypto_hft，profile 为 crypto_binance_hft，正式数据集为 binance_futures_hft_research_100ms。
这是历史研究和执行近似，不是实盘交易、交易账户或投资建议。
代码实现、合成验证、真实数据覆盖和经济有效性必须分别说明。

## 1. 终局能力

形成可复现的单场所研究链：不可变公开事件录制 → 确定性 L2 重建 →
微观结构特征/成熟标签 → 按时间切分的模型与持久化 OOS 预测 →
具有费用、延迟、部分成交、队列近似和资金费的 maker/taker 账本 →
净收益、库存、markout、不利选择和稳健性报告。

核心是解释事件顺序、可观测信息与执行假设，并在足够完整的真实 BTC/ETH 数据上检验增量价值。
公开 L2 只提供按价格聚合深度，不能推出真实订单 ID、精确队列位置或完整撤单生命周期。
研究必须说明近似适用条件；不把短合成回放或 DataFrame 位移结果称为真实撮合证明。

## 2. 已实现功能与证明边界

| 能力 | 当前实现 | 限制 |
|---|---|---|
| 录制 | futures-hft 期货100ms深度与聚合成交；spot-l2 独立现货深度录制 | 公开端点、无认证下单；未录制的完整L2历史不能事后补造 |
| 回放 | 运行清单、成员身份、快照与增量、序号、连接/分段和多时钟验证 | 现货/期货独立回放，缺口不静默连续 |
| 研究数据 | 持久化100ms PANEL，14个微观结构特征，价格与执行质量标签 | 旧分钟线、旧独立封装及旧通用agg/depth适配已删除 |
| 模型 | LR/LightGBM/XGBoost价格模型；GBR执行质量proxy；五个Ridge跨资产范围配置 | proxy不是成交概率；模型不因未观测队列变成真实执行模型 |
| 执行 | 12个taker与2个maker候选，预期净收益定量、显式成本/延迟及连续资金账本 | 队列和流动性为公开L2近似，资金费覆盖不足则阻塞净结果 |
| 展示 | 唯一目标DAG、统计、指定run报告、CLI/API、只读Notebook | 合成与读取验证不证明真实OOS获利或充分日频覆盖 |

正式模型目录为9个配置：3个价格模型、1个proxy模型，以及 btc_independent、
eth_independent、pooled、leave_btc_out、leave_eth_out 五种Ridge范围。
当前真实验收按用户安排延期；新的合格录制与资金费归档仍是下一次真实研究的输入前提。

## 3. 短期建议

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 设计并执行新的BTC/ETH期货录制 | 同时覆盖两币种的连续运行、快照/增量/成交、分段和时间质量；保留原始字节与清单 |
| 2 | 补齐资金费与研究计划 | 实际结算及决策时可得预测分别有来源与覆盖；训练、验证、OOS、留出、资金与费用全部显式 |
| 3 | 完成真实回放与时间验收 | 无伪造缺口修复，无未来盘口回填，标签成熟与分段边界可手算复核 |
| 4 | 先验证简单经济基线 | 分开报告固定报价maker、校准maker及taker的费用、库存、markout、净收益和空档贡献 |
| 5 | 积累足够完整OOS日后做统计与容量研究 | UTC完整日覆盖真实，不挑删缺估值日期；测量长期数据资源/性能而非外推短合成结果 |

录制和研究分开操作。建议不自动启动网络采集、消耗资源或解锁留出。
优先完成当前单场所证据，不因追求特征数量而扩大场所、资产或复杂模型。

## 4. 高级扩展候选

完成单场所核心后，再分别评估：现货/永续基差和跨资产领先滞后；
Bybit/OKX跨场所价格发现；Survival/Hawkes/DeepLOB及有增量证据的注意力模型；
Deribit期权与波动率微观结构。多场所、期权、更多资产和清算研究均为候选，不是已实现功能。
需要对应真实数据和相对简单基线的增量OOS经济证据，仍复用现有DAG、试验、账本和报告，
不预留第二套平台，也不从价格变化虚构精确清算量或L3事实。

## 5. 特征、标签与研究计划

14个当前特征来自同一持久化PANEL：
spread、relative_spread、imbalance_1/5/10、microprice、microprice_deviation、
ofi_100ms、signed_trade_flow_100ms、taker_buy_volume_100ms、taker_sell_volume_100ms、
return_100ms、realized_volatility_1s、top10_depth_per_spread。
当前价格标签是100/500/1000/5000ms的future_mid_return和future_mid_direction；
模型主目标为500ms收益，另有 maker_bid_sell_pressure_proxy_500ms。
每个标签有独立 available_at，短期限必须由实际事件分辨率和观测覆盖支持。

研究计划必须显式给定滚动训练、验证、开发OOS、锁定留出窗口，初始资金、估值币种和暴露上限；
不为适应短录制而缩窗，不随机打乱事件行，不用排序掩盖同时间戳顺序歧义。
训练资格由成熟标签和拟合截止时间决定。额外隔离间隔若必要，属于显式实验语义，
不是第二套通用机械purge规则。特征只消费决策时已知事实，未来盘口/成交和未来派生分类不得回填。

单资产跨时间评估采用单资产时序轴；多资产横截面评估只在合法横截面上计算。
模型统计、OOS预测、费用/价差/延迟敏感性与经济账本分别报告。
概率模型若未来采用，必须检查校准，不能只看准确率。

在硬暴露约束内，按决策时信息比较包括零在内的合法数量与行动，最大化计入适用成本后的预期净收益；
所有合法行动的预期净收益均不大于零时不交易。成本可非线性，简单截断数量不自动是最优解。
定量只能用决策时订单簿，不能用延迟生效/成交时未来盘口或已实现PnL重新选量；
生效后仍允许真实部分成交。负资金费及收入保留符号，已反映在成交价格中的价差/冲击不能再扣一次。

## 6. 当前录制与经济合同

以下是当前正式合同，保留身份、时钟、资金费和执行限制。
通用资源与发布见 architecture.md；原始数据布局与安全见 data.md。

### 6.1 录制与回放

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
现货与期货录制的回放保持独立。
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
连接覆盖复用回放 SQLite 索引：按窗口查询，精确连接查询供 maker 消费，
不把全部历史连接物化为映射。PANEL 按开始时间单调消费游标，用已准入的活动区间堆区分零、一、多条覆盖；
每格最多返回两条身份，不反复扫描请求窗口前后历史。
分段诊断在新增条目前准入，并由录制 session 保留至发布验证；下游借用已校验诊断。
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

### 6.2 资金费、执行与账本

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

校准样本在既有捕获生命周期内保存初始买/卖价位 tick，读取时与其确切初始盘口事件
核对。OOS decision 复用这两个已验证原点，不为每次 KEEP/替换重放重读初始盘口。
当前订单数量、队列、待执行命令和期限仍决定各样本响应，不能将决策相关的响应
冻结为统一成交率；不新增另一份 sample store 或 calibration cache。


`book_spread` 在 PANEL 的整数 tick owner 中以 `(ask_tick - bid_tick) × tick_size` 计算并持久化。
FACTOR 直接消费该价差；relative_spread 和 top10_depth_per_spread 共用同一事实，
不重新相减绝对浮点报价。microprice_deviation 按
`spread × (bid_qty - ask_qty) / (2 × (bid_qty + ask_qty))` 计算，microprice 为
`mid + microprice_deviation`，避免相等队列的零偏移被绝对价格抵消误差污染。

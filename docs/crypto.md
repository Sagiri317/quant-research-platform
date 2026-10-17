# Crypto 高频研究

本文件定义 Binance BTC/ETH 永续高频研究的能力、输入条件和经济合同。
协议为 crypto_hft，只读目录 profile 为 crypto_binance_hft，正式数据集为 binance_futures_hft_research_100ms。
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
| 展示 | 独立操作、数值统计、CLI/API和只读研究图片 | 合成与读取验证不证明真实OOS获利或充分日频覆盖 |

信息来源与执行场所分别绑定：现货/永续上下文、BTC/ETH/SOL 跨资产信息和 Bybit/OKX 外部价格发现均有生产入口；执行组合只允许 Binance BTC/ETH perpetual。SOL 可用于训练、评估和预测，不能加入执行组合。

## 数据来源、录制与回放

研究窗口以连续UTC timestamp定义；午夜不是episode或账本状态边界。book、features、labels、
position、cash、NAV和funding跨午夜延续，仅真实gap、sequence或disconnect断状态。
回测源接纳保留窗口前的快照/增量初始化历史，只剔除最大 OOS 结束时点之后的归档；
校准按所有折中最大的成熟 cutoff 接纳，不能按样本窗口结束提前截掉成熟所需行情。
成交与深度回放均按纳秒闭区间结束时点裁剪；不能向下取整到毫秒而漏掉窗口尾部成交。
独立资金费根仍须按不重叠的结算覆盖区间排列；各根携带的历史预测可按 available_at
归并，重复且完全相同的事实只保留一次。同一可用时点的预测事实冲突、单根内部乱序或
结算覆盖重叠仍拒绝；不删掉后续覆盖之前已知的预测，也不据缺失构造结算时间表或零费率。

### 原生录制与确定性回放

每个录制根目录包含一个 `run.json`，以及该清单声明的 gzip JSONL 成员，
不会通过 glob 合并多个运行。`recording_scope` 为每个输出根目录选择一个独立录制范围：

```text
futures-hft:
  USDⓈ-M Futures 差分深度 @ 100ms
  USDⓈ-M Futures 聚合成交
  -> 由显式 prepare / backtest 操作消费

spot-l2:
  现货差分深度 @ 100ms
  -> 独立现货 L2 深度采集；完整 Spot PANEL 另需成交归档
```

录制清单以 `status` 为采集状态描述；普通 reader 不以它或旧字段
`market_data_status` 授予读取资格，实际消费仍检查必需字段、序号和覆盖，不改写旧文件。
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
初次采集和显式导入保留原始内容证据；普通读取核实际消费事件的结构、顺序及覆盖。
录制在结构与协议校验后保存可读原值；可选成交数量等业务异常不单独触发重连。
Tardis 可读成交序号倒退或冲突保留原始字节并记录局部诊断，只关闭受影响的成交覆盖；
重复且一致的成交去重，异常序号不成为新成交量，也不将未知区间认证为连续。
成交价格无法解释时，已知时钟、总数量和方向仍支持成交计数及基础币流量；
执行只在实际依赖该价格的同方向挂单上标为未知，不伪造成交价格。
同接收时刻的资金费预测逐字段消解：冲突字段为未知，共同期限保留，期限冲突则为零长度未知预测。
该显式未知更新替代旧预测，不沿用旧值；独立的真实结算和结算覆盖不受预测冲突影响。
未来 mid 标签与收益因子统一要求正有限价格；异常原值不改，仅依赖它的标签窗口及成熟时刻为空。

普通读取不以状态、请求 URL、文件 SHA 或压缩字节数作为通行证，不扫描未用成员或全目录，
不重读清单证明选择身份，不因关闭前缀游标而读取剩余尾部。gzip 可有正常预读；
完整消费仍核 gzip 校验、行数、序号及观察区间。来源摘要描述清单，不声称本次已认证全部原件。
同次计算不得并发覆盖已选输入；读中替换探测或全来源物理快照不属于普通消费保证。
连接覆盖通过实际解码并检查区间结构、唯一键和计数建立映射，各数据流区间只排序一次。
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
快照外的绝对数量更新（包括零删除）只证明该价格坐标；只有所有相邻表示价位
都已观察，才扩展连续已知范围。未观察价位不填零，也不从表示单位推断交易所下单规则。
档深不足期间不输出盘口；若原生序列仍连续，继续维护已有价格事实，后续恢复足够
已知档位时从该事件新开覆盖段，不回填缺失区间。真正序列缺口、数值损坏、交叉盘口
或新快照不会沿用旧段的边界外知识。

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

录制器使用 `websockets` 连接公开数据流并保留有限消息队列背压，
使用 `httpx` 获取公开 REST 元数据/快照；两者不设任意响应或单消息总字节额度。
网络超时、取消、语义验证与真实深度档位限制保持。它没有认证端点或下单 API。

### 历史归档与来源身份

Binance 官方录制与 Tardis 原生历史归档均可作为明确来源；历史归档通过切片下载与导入。
七份[科研选择参考](research-rosters/README.md)声明 2026-05-01、2026-06-01、2026-07-01 train，
2026-08-01、2026-09-01 validation，以及 2026-10-01 test；实际使用必须显式提供相应原件，
参考文件本身不证明原件已具备或研究已完成。
训练与验证窗口是上述真实首日数据的日历包络，不表示整月连续覆盖；尤其 8 月 1 日至
9 月 1 日的 validation 包络不代表连续两个月数据。执行仍使用 `continuous`，
缺口须保留真实阻塞语义，不能桥接离散日期。
原生来源通过明确 root 和 `archive_paths` 选择，不能只改一个归档的日期来冒充三段。
`prepare` 的 `source={root, archive_paths}` 指向本次实际原件；maker 校准和执行分别使用
自己的 `source` 路径与可选 `archive_paths`，不从统一 raw-root 或 scenario 目录推导。
Bybit/OKX normalized 信息准备使用 `source={root, archives}`。
有显式来源请求窗时，source owner 按真实覆盖跨度相交选择。执行与校准独立选择来源，核对 PANEL 声明的价格单位与档深坐标；
每次所选源保留自身重建所需历史前缀、顺序与连续性证据，不要求与 PANEL/PRED 的来源集合或 run_id 相同。
输入 PANEL/PRED 的局部 segment/connection ID 仅说明原样本谱系与资格；taker 在决策时刻推进实际所选源，取得当时可见的盘口段。
maker 与其校准还要求此时实际源的唯一成交覆盖。订单寿命、延迟成交、撤换、markout 与校准事件表仍严格绑定该源的连续性；
无法唯一确定覆盖的局部决策保持未定义；不会用另一来源的 UUID 判定等价经济事件失效，也不会跳过真实缺口、单位或时钟验证。
maker 校准只消费合法成熟 validation 样本；测试显式选用已保存校准并核模型、资产、
时序和价格坐标关系，不重训、不改写已发布文件，也不要求重扫全部开发原件。
跨日归档按实际首尾切片范围选择，date 上界与 datetime 上界复用统一 TimeWindow 合同。
不同原件可重置本地序号；持久化 observation/segment 身份带原件命名空间，
跨原件空白保持真实 gap，不拼成连续事件段。
资金费使用同一窗口路由选中日根，逐根执行 forecast/settlement 完整性及身份验证；
coverage 保存实际不重叠区间集合，缺失所需根明确阻塞，不取多段包络填补未知月份。

免费公开日期可直接下载；需要 key 时明确权限不足，不自动注册、购买或切换替代来源。
原始字节、日期、原生频道及频率、文件哈希、稳定行序、断线标记与可证明区间进入来源清单；
正式 raw 布局和来源安全由 [数据合同](data.md) 统一约束。
每次操作使用明确选择的来源，不自动混合两类数据补洞；真实来源差异记录在原生来源证据中。
历史入口绑定明确的连续十分钟切片范围，时长可以小于一天或跨越任意多个 UTC 日。
下载与导入分别使用同一 raw owner 的 `download_slices` 与 `import_slices`；
`date` 是偏移的 UTC 起点，`offset_minutes` 是首片偏移，切片数量不设单日上限。
供应商请求和原件文件名仍按实际 UTC 日期寻址；同一归档的累计行序、观察段和回放状态
跨午夜不按日期重置。真实断连、原生序列缺口或可定位的坏盘口数值均阻断相应重建连续性，
恢复仍须真实快照与合法桥接，派生失效不改写原始观察身份。已建立连续盘口后，忽略额外的日初引导快照，
遵循 [Tardis 官方回放实现](https://github.com/tardis-dev/tardis-node/blob/master/src/mappers/binance.ts)；
断连后的新观察段仍须重新用快照和增量桥接。已失效或尚未完成首次桥接的盘口，可以由后续
真实快照重新初始化；同一来源观察段内恢复时建立独立重建段，保持供应商原始观察身份。
只有实际收到且通过序号桥接的后续增量才能开始有效覆盖；恢复不回填缺口，标签与订单不能跨越未知区间。
资金费查询绑定实际归档范围。
train、validation、test 按时间先后划分；最后一折的实际 train 与 validation 窗口用于最终 refit，可以跨日，不要求自然日对齐；
特征与标签不能跨越实际无效行情区间，标签成熟检查不因跨日而放宽。
不同原件日期共同进入一个来源清单和精确身份，不另设日内/跨日研究路径。
已有原件通过同一 `import_slices` 重新生成归档清单，保留原生字节及原始来源证据。
历史来源复用同一盘口、100ms PANEL、因子/标签、模型和经济账本，不建立另一套回测链。
Binance 原生字段由同一个来源校验 owner 按清单 schema 和绑定的供应商观测 UTC 判定。
官方录制要求其供应商 schema 对应字段；Tardis 历史归档按当日 schema 校验。
依据 [Binance 更新日志](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/change-log)，
聚合成交的非 RPI 数量 `nq` 自 2025-12-31 引入，CM/UM 的 `st`/`ps` 自
2026-06-30 引入。采集入口校验供应商 schema、身份、时钟和协议结构；可读业务异常保留原载荷，由回放在身份、时钟和顺序可定位时按依赖解释为局部未知。
缺失的历史 `nq` 不填入原载荷，也不推断为 `q` 或零；内部成交事件保留未知。
历史回放中，若身份、时钟、顺序、价格网格、正的 `q` 与方向均合法，仅 `nq` 缺失、
不可解析或非有限时保留成交事件，`visible_quantity=null`；已知 `nq=0` 仍为真实零。
有限 `nq<0` 或 `nq>q` 属于数量矛盾，仍切断受影响成交覆盖，不能任选一个数量当真。
现代采集入口仍按当前供应商 schema 校验身份、时钟与结构；可选行情数值缺失不等于结构错误。
PANEL 总成交量因子和标签消费 `q`；需要非 RPI 数量的 maker
队列消耗在实际依赖该未知成交时使账户路径 PARTIAL；校准排除受影响的成熟结果样本，保留后续完整样本。不能因归档成功声称 maker 所需信息齐备。
旧归档若曾仅因 `nq` 未知生成成交 gap，须以未改写的原件重新生成派生清单后使用新语义；
reader 不自动忽略已有 gap 或原件校验差异。

原生深度事件频率与研究网格分别声明；细增量仍按其顺序参与 OFI 等充分统计，
不能以每格首尾价位合并替代已观测的完整事件路径。

Tardis 接收 UTC 是供应商的观测时钟，不是本机接收时钟；缺失的本机单调时钟、
连接身份和连接开关事实不得通过下载时间、行号或虚构值填入。
来源中可证明的排序、快照桥接和有效区间是归档事实，不冒充本机物理连接覆盖。
缺口必须切段，恢复后的快照不能倒填缺失盘口，标签及订单路径不能跨未知段假装连续。
缺口期间的未知成交和持仓变化不能默认无损；实际账户路径无法确定时保留 PARTIAL 精确前缀，完整净收益为未知。

### 价格坐标与事件顺序

规范盘口使用 `price_unit` 和整数 `price_index` 作为无损价格坐标，
它们不表示交易所最小下单增量。官方原生 `tick_size` 及其来源保留，
在解码边界映射为价格单位；Tardis 按同一输入日、同一资产的全部执行价格
（快照、深度更新及聚合成交）的原始十进制精度确定固定单位，不能逐段改变。
精度推导不经过浮点或受 Decimal context 影响的四舍五入；标记价格不参与盘口坐标推导。
归档集合按本次已选原件、每资产取最细单位，metadata 与交给共享 reader 的 record 使用同一单位；
开发阶段不利用尚未选入的留出原件精度。每资产校准摘要表的 `price_unit` 是历史整数样本和事件的唯一单位声明，
同一产物全部 fold/side 必须一致，由校准生产与实际消费者核单位及配对关系。
消费视图用 Fraction 计算校准单位到当前执行单位的倍率，仅倍率为整数时转换历史
origin/book/trade；不可整除时以 `hft_maker_calibration_price_unit_incompatible` 明确拒绝。
不得只取有理倍率的 numerator 误映射。PANEL 声明的单位/深度规格还须与执行来源一致，
冻结校准文件保持不变。
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
`realized_volatility_1s` 是最近 10 个连续 100ms 简单收益的样本标准差（ddof=1），
未年化；不是平方收益之和的平方根。不完整窗口保持 NULL。
微观结构因子另提供多尺度 TFI、OFI、深度归一 OFI、深度压力与不平衡差值，另有共享的 return_500ms，族内共 30 项。
BTC/ETH PANEL 的 `parameters.asset_ids` 可选其中非空子集；未提供时使用对应目录默认集合。
请求但未取得的资产在诊断中披露，不删除已取得资产的合法行；全部所需资产均缺失仍不能产出有效面板。
每个资产保留自身的 100ms 时间网格，输出按时间与资产归并并集，不因另一资产缺口删行。
`segment_id` 只表示盘口连续性；`trade_observation_id` 单独表示成交观测连续性。
盘口有效但成交未知时保留盘口行，成交计数与流量为 NULL；有认证覆盖且确实没有成交时才为零。
各因子只读取所选计算需要的列与频道，窗口按自身依赖的连续性预热；frame 边界不重置状态。
选择不存在的列或显式排除已有必需业务列是配置错误；实际源内局部缺值只影响依赖计算。

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
状态分别保留 event_at、observed_at、age_ms；age 上限写在 `prepare.parameters.max_age_ms`，
并作为本次 processed 数据规格中的实际参数记录。
`age_ms = decision_time - observed_at`，`max_age_ms` 只限制供应商接收年龄，
不保证 `event_at` 对应事件仍新鲜；较旧事件的新接收或重复轮询可以通过该门槛。
事件年龄可由已保留的 `event_at` 单独检查，不能据接收年龄声称事件新鲜，
也不因交易所事件时钟倒退而全局拒绝、重排或去重已观测状态。
OI 使用 base 数量，变化按（当前 OI-窗口起点 OI）/窗口起点 OI计算，整个窗口须有效；
funding 为原始小数费率，time_to_next_funding 和 time_since_observed_force_order 为秒，
basis 保持原有分子、分母与方向，按（分子-分母）/分母计算无量纲相对差。
频道名 premiumIndex 不代表供应商存在同名经济字段。

forceOrder 仅代表供应商观测到的 proxy。每条发布快照以 z（累计已成交 base 数量）加权，
BUY 为正、SELL 为负；不以订单数量 q 代替成交数量，不声称完整或去重后的全市场清算量。
无观测或分母为零保持 NULL；断连 generation、grid gap 和 episode 边界重置窗口。
执行 PANEL 的 segment/grid 边界只重置滚动窗口及预热；独立衍生品源的最新观测和观测时钟仍按自身 age 上限保留，只有该源 epoch 变化才失效。来源换 epoch 后，forceOrder 的完整 5s/30s 窗口从真实 observed 边界重新预热，不能回锚到当前信息时点前一个格。若收到的首条来源事件本身标记reset，同样按该真实边界预热；只有无reset的正常PANEL首格沿用首格规则。forceOrder elapsed 由同一最新观测时钟派生，窗口首个100ms只纳入该区间内的已观测成交。
双边深度和滚动流量比值在最终除法后舍入。局部聚合超出 float64 表示范围时，只有该字段为 NULL；
仍可表示的计数、盘口与比值保留，不把有符号 OFI 当成非负深度。新收到的坏状态形成未知值，不能继续沿用旧有效状态。
缺 state 归档只阻塞依赖它的分支，微观结构因子不引入该依赖。

### 现货与永续上下文

Spot panel 复用同一个 native 回放与 panel producer，逐资产保留真实观测覆盖；
Spot/Perp context producer 在执行 panel 的主键上输出，assembler 只做精确键连接。
现货和永续共用微观结构数学核。Spot basis 为 (spot_mid-perp_mid)/perp_mid，收益为已成熟的
100/500ms 简单收益；OFI 以相同币种的 base 数量计，跨市场 OFI divergence 使用深度归一值。
TFI 为有符号成交量/买卖成交量之和。lagged 字段取上一格连续 Spot 100ms 信息，
不能跨 Spot segment/gap；上下游 frame 切分不改变结果。

Spot context 的 `prepare.parameters.max_age_ms` 同时限制 Spot grid age 与实际 book observation age，
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
跨资产 context 读取持久化的成熟 100/500ms 收益、深度归一 OFI 和 TFI；information state 同时原样保留来源资产的真实 segment_id。
仅所选 lag 因子额外投影源 segment_id，要求同一信息资产的前后两格同段；不比较目标与信息资产的段名，当前单字段 NULL 也不抹除同段合法历史。
旧 information state 若未保存该字段，选择 lag 时明确失败，须由调用者重新准备；仅选当前值时仍使用最小数值依赖。
来源资产的数值及上一格连续 100ms 的滞后值先在 producer 内按观测 grid 对齐，
记录各来源 grid 时间与 age。缺口、过期或不完整窗口保持 NULL。
同一个数学核负责 500ms 收益，不在 Spot、SOL 或跨资产层复制公式。
只依赖 BTC/ETH 的配置不引入 SOL 数据节点；选择 SOL 后缺归档的依赖分支保留 BLOCKED。

### 外部价格发现上下文

`docs/research-rosters/crypto/external_context.json` 显式选择 Bybit 与 OKX 信息因子。
两个 venue 分别拥有 processed target；未配置或缺失其中一个来源只阻塞依赖它的分支。
来源使用已有 Tardis normalized owner 的 quotes、trades，
由 producer 按 `local_timestamp <= decision_time` 对齐到 Binance BTC/ETH 100ms 主键，
随后使用现有 exact-key assembler。SOL 研究行对这些 BTC/ETH 专属信息保持 NULL。

成交 amount 的原生单位不得混用。Bybit linear 为 base 数量。OKX BTC-USDT-SWAP
旧期每张为 0.0001 BTC，ETH-USDT-SWAP 为 0.001 ETH；2020-03-20（BTC）与
2020-03-19（ETH）香港时间 16:00–16:30 调整至 0.01 BTC 与 0.1 ETH。
该半小时内的精确生效点未取得保证，相关成交明确阻塞；其他日期按成交事件时点选择已核面值。
换算事实由市场 owner 定义，作为实际 processed 参数记录。来源说明：
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
maker 报价期限由 Crypto 执行 owner 的固定 500ms 合同拥有；它与模型预测期限和 markout 时钟不同。

价格标签为100/500/1000/5000ms的future_mid_return和future_mid_direction；
各模型显式选择标签；baseline.json 的价格模型选择500ms收益，另有 maker_bid_sell_pressure_proxy_500ms。
每个标签有独立 available_at，短期限必须由实际事件分辨率和观测覆盖支持。
只有有限且严格为正的 mid 与 future_mid 才能计算价格标签；非正或缺失端点使对应标签缺失。
正价格端点按 (future-mid)/mid 计算，保留相邻浮点价格的不同回报。
标签不以 epsilon 归零，原有 segment、成熟尾部 NULL 与最终不可表示域处理保持。

## 研究配置与模型

### 配置与数据角色

baseline.json 保存十个模型选择作为科研参考，包含普通决策树；agent 为实际选定模型
构造独立 fit/predict/score/select 配置，程序不直接展开该参考文件。
下列七份 JSON 是既有的可选选择参考，尚不是后文九算法、14→29 项输入及四期限的完整
实验矩阵，也不是自动执行配置或开展有限探索的前提；原有跨币、pooled 与 maker 能力保留。
评估因子、各模型输入、训练和评估资产分别显式选择。multiscale.json 选择多尺度微观结构因子，
derivatives_state.json 加入衍生品状态因子；不做块或参数的笛卡尔搜索。
示例中的费用、延迟和资金为研究假设，不能当成已验证的历史账户费率；
保留 taker 5 bp、maker 2 bp 的参考假设。
线性回归预测保留向量化快速路径；中间标准化或矩阵乘积非有限，以及次正规/下溢可能丢失最终贡献时，
局部使用已保存原点、精确偏移、scale 和系数计算完整仿射值。最终可表示值不得因中间溢出丢失，
真正不可表示的预测仍由输出 owner 明确拒绝。
排名模型的标签标定按统一二进制单位累计整数充分统计量，仅最终截距/斜率转为浮点数；
常量分数、极小非零斜率及不可表示结果的失败合同保持不变。

数值监督模型的 scaler 只在训练期拟合；`weighted_mean_offset` 每列统一保存最简整数比 `[分子, 分母]`（正分母），极小中心偏移保留到最终标准化值舍入。旧浮点偏移 metadata 与旧 pickle 明确拒绝，受影响模型须重建；配方变化显式写入新模型与真实 trial，既有失败结果不覆盖，不靠软件摘要自动失效。
真实研究须有合格的事件来源与资金费归档；代码验证或合成验收不能替代真实覆盖。

| 配置示例 | 选择的研究信息 | 附加来源条件 |
|---|---|---|
| [baseline.json](research-rosters/crypto/baseline.json) | 基础微观结构因子及十个示例模型 | Binance BTC/ETH 主来源 |
| [multiscale.json](research-rosters/crypto/multiscale.json) | 多尺度微观结构因子 | 同一主来源 |
| [derivatives_state.json](research-rosters/crypto/derivatives_state.json) | 微观结构与衍生品状态 | 独立 native state 归档 |
| [spot_context.json](research-rosters/crypto/spot_context.json) | 微观结构与现货/永续上下文 | Spot 深度、快照及成交归档 |
| [cross_btc_eth.json](research-rosters/crypto/cross_btc_eth.json) | BTC/ETH 跨资产上下文 | 不依赖 SOL |
| [cross_asset.json](research-rosters/crypto/cross_asset.json) | BTC/ETH/SOL 跨资产上下文及显式迁移选择 | SOL native 归档 |
| [external_context.json](research-rosters/crypto/external_context.json) | Bybit/OKX 外部价格发现 | 各场所 quotes、trades 归档 |

各操作只要求自己消费的输入：拟合声明历史窗口与截止时刻，预测声明实际预测窗，
校准声明样本窗与成熟截止，回测声明资金、执行假设与执行窗。
校准的 `chronology` 可补全省略的样本窗与成熟截止；显式操作窗口不受计划的 train/dev/test 名称约束。
不为适应短录制而偷偷缩窗，不随机打乱事件行，不用排序掩盖同时间戳顺序歧义。
训练资格由成熟标签和拟合截止时间决定。额外隔离间隔若必要，属于显式实验语义，
不是第二套通用机械purge规则。特征只消费决策时已知事实，未来盘口/成交和未来派生分类不得回填。

单资产跨时间评估采用单资产时序轴；多资产横截面评估只在合法横截面上计算。
直接 `fit` 的 `asset_ids` 明确训练资产，模型 metadata 的 `training_asset_ids` 记录实际训练资产；
评分采用本次显式选择的 `validation_axis`：单资产时序在成熟验证序列上计算 Pearson IC，
RankIC 明确为 `not_applicable`；截面模式按每个时点的实际样本计算，不因缺失而自动切换轴。
训练资产范围不限制后续评价轴；模型可以对训练集以外的资产产生显式预测；
执行范围由 PRED 的实际键与所声明预测资产限制，不把训练资产列表作为下游准入。
单资产时序 score 在成熟、资格和共同域筛选完成后，跨全部 chunk 只能包含一个实际资产；
该资产可以不同于训练资产。多个资产不能混成一个时序 IC，也不自动发明逐资产聚合规则；
多资产预测本身仍合法，调用者按所需统计域选择输入。
单币候选 score 矩阵须分别构造 BTC 与 ETH 的候选集合，并在对应标签输入使用既有
`row_filters` 选择同一资产。score 按 `(timestamp, asset_id)` 取候选共同键；把 BTC-only
与 ETH-only 候选放进同一矩阵会得到空共同域。跨币 pooled 模型属于另一个显式实验。
仅有 BTC/ETH 两个有效且无并列的观测时，截面 Pearson IC 和 RankIC 都只能为 ±1，
其均值等于 `2 × 排序命中率 − 1`；二者不是相互独立的效果证据，也不能区分信号幅度。
并列或缺失仍按既有有效样本数及零方差规则报告，不用时序指标替换该截面定义。
模型验证 utility 接收时间有序的 chunk，允许一个时间戳跨 chunk；合并完整截面后排名并增量归约，
跨 chunk 时间回退明确拒绝。该顺序合同同时适用于持久化模型的验证指标重算。
冻结模型的 `sample_filter_mode` 同时约束配置预测规格和最终 OOS 预测；显式 `unfiltered`
保留输入完整但不符合 eligible 的预测行。监督训练与验证仍须满足原有 eligible、成熟及输入合法性。
模型统计、OOS预测、费用/价差/延迟敏感性与经济账本分别报告。
概率模型若未来采用，必须检查校准，不能只看准确率。

`sample_role` 是可选的窗口用途说明，不赋予训练、推断或回测资格。独立预测、评估分别默认 `prediction`、`evaluation`；模型、因子及 A 股组合统计默认
`exploratory`，均可使用明确的自定义名称。Crypto 经济统计沿用输入账本的用途说明，不另设重复参数。
walk-forward 每折分别声明拟合窗口、拟合截止、预测窗口和标签成熟截止；同一日期早先用于
样本外评价，在以后标签成熟后可以进入后续拟合。各折的拟合窗可滚动或扩展，不强套永久三集身份。
模型不预先登记测试窗口；消费时只核实际选择截止、拟合事实和必要校准。重新拟合可以使用新声明且截至
拟合时刻合法的历史；离散窗口不以包络填平。已查看的测试结果不能再宣称为未参与选型的证据。
预测使用模型；后续评估、回测和统计直接消费自包含 PRED，不重新打开模型或 trial 文件。
回测按请求资产、时间窗和实际 PRED 键匹配当前执行 context，不重读历史 X，也不按当前
`sample_filter_mode` 重建预测资格；当前输入多出的键不要求补预测。所选 PRED 键仍必须有
对应的当前 context 行，缺失报 `misaligned_persisted_prediction_keys`；校准仍核当前标签的资格与成熟。
PRED 读取在同次打开中用 Parquet footer 核总行数，包括时间子窗读取；不另做准入哈希扫描。
缺新数据只阻塞真实依赖该数据的操作。

### 数值缩放与排名模型标定

Crypto 不使用 A 股横截面 canonical transform。数值监督模型使用 training-only ModelMatrixScaler；
排名集成直接使用排名输入，训练得到的方向或权重及标签单位标定随模型冻结。
线性预测保留普通矩阵乘法；当标准化值进入次正规数域而可能丢失有限仿射贡献时，
仅对受影响行按已冻结系数、中心和尺度精确求最终值。非有限结果继续走同一局部路径。
此数值边界检查不重估模型，也不把极端合成输入解释为真实收益证据。

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
直接 backtest 的空模型列表配合 `calibration=null` 表示未校准 maker；否则必须同时指定已选模型，并分别使用
500ms mid return 与 500ms observed sell-pressure proxy 标签。
maker 的 `models` 显式提供价格和 proxy 各自的自包含 PRED 路径，按实际所消费的目标与
覆盖校验；价格/proxy 按实际 `(timestamp, asset_id)` 键交集配对，同键的 fold 和执行 context
仍须一致；各自未配对的尾段也完成所选键的 context 覆盖校验。后续不重开模型或 trial。

模型预测期限由实际 `target_column` 对应的 executable label 决定；
taker 只接受 future-mid-return 模型，从同一标签 owner 取得持有/预测期限；
direction 或压力 proxy 不得被当作收益率使用。
模型 refit 与验证产物的成熟时间校验也读取各自标签期限。

### 共同样本外评估

研究者显式选择同标签模型及其预测路径，调用 `evaluate` 的同一数值 owner；
需要成对共同 OOS 评估时，明确声明双方信号和标签，不自动展开 comparison_model_id 的全研究比较。

共同评分先要求双方各自预测通过完整覆盖校验，再按精确主键取有限预测交集，
并使用同一 fold 的标签成熟截止时间。资产或有效时间范围没有交集时，
发布零行比较产物，时间覆盖边界为 NULL，报告记为 UNDEFINED；不制造样本或零相关系数，
也不把没有共同样本误判为整个研究运行失败。
报告分别展示各模型完整评估与共同样本评估；不裁剪预测文件或交易账本，
不同标签/预测单位的模型不直接比较。该过程不是因子笛卡尔搜索，也不新增回测引擎。

## 执行、资金费与账本

### 候选、数量与成本

每次 `backtest_taker` / `backtest_maker` 明确提供
`inputs`、`panel`、`source`、`asset_id`、`symbol`、`prediction_window`、
`execution_mode`、`capital`、`execution` 与新 `output`；可额外给出 `archive_paths`，
裸 Parquet 输入还需 `input_spec`。`capital` 保存 `candidate_id`、`initial_capital`、
`valuation_currency`、`valuation_method`；执行资产限定 Binance BTC/ETH，币种为 USDT，
估值方法为 `linear_perpetual`。回测仅接收显式 `prediction_window`，不接收 `chronology`。
taker 另给 `predictions` 路径；校准 maker 给按价格/proxy 顺序排列的两个
`{predictions}` 及 `calibration` 路径，
固定 maker 使用 `models=[]`、`calibration=null`。
独立回测与统计不反推 scenario 路径，不自动生成目标名字；报告逐组合列出校准证据，
不能拿 BTC 的校准产物用于 ETH。

在硬暴露约束内，按决策时信息比较合法数量与行动，最大化计入适用成本后的预期净收益。
新增风险且零动作可行时，将零纳入比较；所有增量行动的预期净收益均不大于零时不新增交易。
已有挂单按真实剩余数量、队列、TTL 及指令延迟比较 KEEP、取消与重挂；取消不等于立即零损益，
预期净收益为负的 KEEP 仍可能优于损失更大的取消。成本可非线性，简单截断数量不自动是最优解。
定量只能用决策时订单簿，不能用延迟生效/成交时未来盘口或已实现PnL重新选量；
生效后仍允许真实部分成交。负资金费及收入保留符号，已反映在成交价格中的价差/冲击不能再扣一次。

直接 taker 的 `execution` 必须提供 `order_quantity`、`position_size`、
`taker_fee_bps`、`latency_ms` 和 `markout_horizon_ms`，不填入隐含零费用或零延迟。
科研参考保留 `[0, 100, 200]` ms 延迟以及费用/附加价差敏感性选择；agent 明确决定
各次调用及新输出位置，按资产、模型、实际延迟避免重复研究，不由 backtest 展开场景。
改变执行费用/延迟后应重新执行；固定成交上的费用敏感性另由统计计算，不能混称真实重跑。
净损益由成交、费用、资金费与估值变化的共同经济事实累积，再转换为展示用float；
不能从大本金NAV的浮点差分反推微小损益。OOS与空档贡献直接按事件区间求和，不相减两个已舍入累计值。
producer 与账本汇总共用 compute 层的精确经济累计器。估值合同以 FMA 计算 quantity×mark-cost_basis，
仅在此恒等式容纳已发布成本及最终数值的独立舍入误差；预算为三个有限值各自一个 ULP 之和，
不从可能溢出的中间乘积推导容差。部分平仓先将当次已实现损益舍入为账本数值，
舍入余量留在剩余持仓成本中，使现金、持仓成本和成交金额守恒；全平或反向开仓重置成本。
数量及局部乘积抵消仍保留精度，不随历史成交不断累积平均成本的分母位数。
Taker 按实际到达的各档 take 累计成交量和成交额，不从大请求量与剩余量相减反推成交；
残量减法发生舍入损失时，仅该订单保留精确残量与末档 take，最终转为账本数值。
Maker/Taker 费用在执行 owner 统一计算；maker 接受有限有符号费率（负值为返佣），
taker 执行费率必须非负。费用按实际成交额与费率计算，非有限费用或收益明确失败。
冻结 maker 校准的候选与 KEEP 收益复用同一费用 owner；真实成交数量、现金和净损益的精度合同保留。

maker 在自己的 `execution.latency_ms` 下消费匹配的冻结校准。资金来自本次 `capital`；
taker 的 `execution.position_size` 与 maker 的 `execution.max_position` 分别给出仓位上限。

数量按小规模连续数量作研究近似，不模拟交易所最小名义金额、数量步长等下单过滤器，
不能据此声称订单在真实账户可提交。常规研究采用显式的单边 taker 5bp、maker 2bp 费用假设；
它们不是已核实的历史账户手续费，也不替代各次调用中明确提供的费用参数。
当前不模拟保证金、杠杆档位或强平；名义仓位上限不是保证金约束，也不能据此称为无杠杆回测。

研究主 NAV 始终采用当时已观察且覆盖有效的盘口中间价。实际资金费现金仍以真实结算标记价
计算；结算标记价不覆盖盘口估值状态。先按旧仓位结算同刻资金费，再处理该时刻到达的新盘口及成交，
不能使用之后才观察到的盘口补写结算时刻 NAV。
真实盘口缺口中，已知资金费保留原结算时刻和金额，暂不发布缺口中的 NAV；下一次可靠盘口到达时，
在最后可靠估值与恢复估值之间发布这些现金流，并在恢复后的任何决策或成交前结清。
若盘口始终未恢复，完整净值仍为 PARTIAL，精确前缀不前移；执行摘要单列
`unvalued_funding_cashflow` 和 `unvalued_funding_event_count`，表示已知但尚未纳入可靠估值的资金费。
缺口路径回撤仍不可识别；不把已知资金费误作未知费用，也不以结算价、旧报价或虚构平仓填补缺口。

### 资金费来源、预测与结算

Tardis 市场事件消费明确归档集合自身的深度与成交，不以官方事件补洞。
同一 Binance venue 的结算数据作为独立附属输入，明确区分官方 `fundingRate` 原件与
[XOOMAR 历史结算记录](https://xoomar.com/markets/api/funding-rates)。两种来源均原样保存完整响应至
资金费 symbol 原子目录中的 `settlements.json`，不以自行抽取的记录或上游哈希替代原响应。
`settlement_source` 绑定供应商、endpoint、symbol、请求窗、HTTP 状态、原字节哈希、实际 limit、
响应数、窗口内结算数及从响应推导的完成依据；importer 与 reader 复用同一校验责任模块。
原字节哈希与字节数作为导入来源记录保留；普通读取不以文件快照或字节哈希一致性准入，
仍解析所选原响应、核对请求窗及完成依据，并逐条核对实际结算与 CSV 投影的时间、费率和标记价格。
资金费 CSV 普通回读接受等价换行、JSON 空白和数值拼写，保留列头、类型、行数及全部经济语义校验；
显式导入验收仍核对规范字节、字节数与哈希，其他来源的 CSV 验收行为不变。

[Binance fundingRate](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
按请求 limit 截断，默认 100、最多 1000；离线导入必须显式提供实际有效 `request_limit`。
单页导入只在响应数小于实际 limit 时认证请求窗，满页保持覆盖未证，不从小于 1000 推定。
XOOMAR 的 `data.history` 最多 5000 条、跨场所并按时间倒序；原响应必须在正确 Binance/资产
子集上同时给出请求起点之前与末端之后的真实结算，才满足本地窗口包围检查。
`history_brackets_window` 的 COMPLETE 依赖 XOOMAR 返回完整历史的供应商假设；
两端包围本身不能证明中间没有漏项。允许尚未与官方逐笔结算记录核对的日期用于研究，
但必须披露此限制并继续官方记录交叉核对，不能将未核对称为独立证实完整。
探索性净收益研究可采用已与官方核对的结算费率及时间、配合第三方结算标记价格，
并分别披露来源；费率和时间已核对不代表标记价格或结算金额也获得了独立官方核验。
服务 `updatedAt` 不充当结算完整性水位。不能满足本地完成条件的输入明确 BLOCKED，
保留已有产物，不增加自动分页或按固定八小时猜测补洞。
其 `observations/` 子目录
仍只含严格 CSV 与元数据两个成员。规范元数据的 `settlement_source` 显式说明原件身份，
既有显式规范归档没有此原生附属成员时为 `null`，不从缺字段静默推断。
历史 `markPrice` 的 `r`/`p`/`T` 只产生当时供应商接收 UTC 可见的预测，不能使用未来最终结算反填。
每一条可定位的新观测都替换此前的预测知识。合法未来 T 的 rate/mark 缺失或非法时，该事件预期现金流为 null；
已知 rate 为零时金额严格为零，不依赖缺失 mark。无效、缺失或已过期的 T 保留为
`known_until_ns=time_ns, cashflows=[]` 的零未来知识观测，不沿用旧预测或补造未来时点。有效更新只恢复之后的决策。
两者投影到既有资金费列与 `FundingObservations`，保持预测和实际记账责任分离。

现有 `backtest_taker(config)` / `backtest_maker(config)` 可选 `funding_roots`，
值为按覆盖起点排列、非空且不重复的原生资金费目录路径字符串列表；每个目录直接含
`response.csv` 与 `response.metadata.json`。独立 `import_funding_history(output_root=...)`
产物使用其 `observations/` 子目录，无需复制或搬到行情归档下。只选择本资产需要的文件与日期；
各根的证券须匹配 `symbol`，覆盖不可重叠，间隔不会自动补齐。
省略字段时沿用行情来源的资金费路径选择；显式选择缺失、畸形或错误证券时直接失败，
不回退行情目录。所选路径随账本 metadata 保存；该配置不改变 PANEL、模型或独立评估输入。
回测窗口仍由 `prediction_window` 指定，结算请求覆盖保持自身真实 UTC 纳秒范围；
预测可保留窗口前最后已知更新，结算不会变成预测。费率为小数，mark 为 USDT/资产单位，
实际长仓资金现金流为 `-quantity × rate × mark_price`；时钟和未知规则见下文。

`sources/binance/funding.py` 读取录制目录同级的
`funding/<symbol>/response.csv` 与 `response.metadata.json`，复用原生来源接纳合同。
列为 `kind,time_ns,known_until_ns,rate,mark_price,cashflows`。
元数据必须声明 `schema_id=qrp-binance-funding-observations`、证券、期货市场、
`USDT`、带起止纳秒的 `coverage` 和非空来源依据 `provenance`。
完整净收益要求实际所需结算日历有可信覆盖，且有仓时所需金额已知；
独立分段按每段实际依赖检查，不把段外未知资金费解释为零。
缺文件或 `UNPROVEN` 只阻塞依赖该证券资金费的净收益账本；数据、模型和评估可继续。

`settlement` 记录用 `time_ns,rate,mark_price` 结算实际现金；
`forecast` 记录的 `time_ns` 是真实可得时间，`known_until_ns` 是已知覆盖末端，
`cashflows` 是按时间严格递增的 `[event_ns, signed_USDT_per_long_asset_unit]`。
两类记录的非适用字段为 `null`。预测值为 `null` 表示该笔预期现金流未知。
`known_until_ns` 可以等于 `time_ns`，此时现金流列表必须为空，表示没有未来覆盖，不能解释为未来资金费为零。
合法覆盖中位于当前预测期之外的未知现金流不影响当前决策。
已定位实际结算的坏 rate/mark 分别保留为 null，不删除事件或抹去另一已知字段。已知空仓时资金费为零；
有仓且实际 rate 为零时，缺失 mark 不妨碍零现金流，也不产生估值。其他有仓未知结算使相应账户后续净结果 PARTIAL。
结算时点、身份、顺序或原件不可信仍拒绝相应来源。
只有决策时已经可得、覆盖预测终点且相关现金流均已知的 forecast 才参与增量下单。
局部预测过期、覆盖末端不足或预期现金流未知时，该决策不新增订单；
使用 `causal_no_new_risk_partial_forecast_v1` 合同。taker 已发出的订单仍按原延迟到达，
不能事后撤销其承诺；校准 maker 对既有/待生效挂单发出原延迟撤单，撤单生效前及同刻的
先序市场事件仍可成交，风险预留及订单期限保持原规则。有效预测恢复后重新计算新决策。
实际持仓结算与预测资格独立，不延长预测覆盖，不用未来更新或实际结算反填预测。
`blocked_decisions` 保留缺失决策数，`funding_required_decisions` 只计市场/模型门槛合格且
确有新增或维持订单决策需要的资金预测检查；`funding_covered_decisions` 及
`funding_decision_coverage` 披露有效数及比例，无需检查时比例为 null，不能称为全覆盖。
部分检查有效则选择状态为 `PARTIAL`；全部所需检查无效也保留真实无新单账本与选择不合格诊断。
独立段分别保留选择状态与预测覆盖，STAT 同时披露逐段和全目标诊断，不拼接收益。
预测资格与账户路径完整性分别记录，不能用没有新增订单来证明已有仓位、挂单或结算已知。
执行语义变化后由研究者明确重算相应账本及其统计/图表，代码不通过软件身份自动失效。
旧严格阻塞原件不改写；没有科学依赖变化时可显式选用已有 MODEL/PRED，
不按留出收益选择缺失策略。
实际结算不能反向影响下单量；显式完整覆盖内的空事件表才能表示无结算，
没有输入不能当作零资金费。本模块消费外部归档，不新增联网采集端点。
账本summary与报告保留`funding_qualification.sources`中的coverage、settlement_source及verification。
verification区分显式归档尚未独立核对、官方响应已验证、供应商完整性假设尚未独立核对三种证据；
它们分别为`declared_archive_not_independently_verified`、`official_response_validated`、
`supplier_completeness_assumption_not_independently_verified`。没有资格记录的手工账本样例为null，不能补造独立核验证据。

### 连续与独立分段执行

直接 backtest 按实际 `prediction_window` 消费持久化预测，要求模型拟合截止早于预测窗。
dev 或探索窗口均可直接回测；显式执行窗口不会自动授予结果 test 身份。
校准 maker 仍要求全部所选校准截止早于执行开始。下文执行 OOS 指本次声明的样本外执行窗。
研究窗口由连续 UTC 时点定义，午夜或文件块不是账户重置边界。
每次 backtest 的 `execution_mode` 选择 `continuous` 或 `independent_segments`；
比较两种模式时分别调用并指定新输出。evaluate 不接收组合执行配置。
连续模式覆盖不足不自动退化；独立分段在每个预声明观察区间从相同初始资本、空仓、
无挂单开始，复用同一执行引擎和原冻结模型，不在新段重新拟合。不得跨缺口拼连续收益。

执行读取实际选定的来源覆盖，不用 PANEL 的较长时间范围代替。taker 的盘口执行不凭空要求成交源；
maker 的排队成交依赖真实成交观测。盘口缺口但没有潜在成交时，可保留已知现金与持仓，恢复后的真实估值仍可用；
有暴露的缺口路径回撤保持未知。潜在成交或有仓结算未知时，账户记为 PARTIAL，保留停止前的精确记录，
不把未知期解释为无成交、提前取消或零资金费。
独立分段在首OOS至末OOS包络内，按实际candidate所需盘口/成交区间划段，并以结算覆盖边界及未知金额事件切段。
资金费未覆盖区间不直接删除，段内按实际持仓判断结算依赖；预声明段本身不证明账户完整；
决策窗口仍是原OOS窗口并集，折间空档保持合法估值/持仓/资金费但不产生新信号。
预先划定的独立段内部新发现坏成交时，可以结束该段并保留 PARTIAL；不会自动重启一个账户或增加回放扫描，后续已声明独立段仍可继续。
事先已知计划末端与事后观察截断严格区分：后者不能提前改变数量、TTL、KEEP、撤单或markout目标。
截断时仅保留已观察事件；待生效指令、活跃挂单和未成熟markout记截断/未知，不制造未来成交、
取消成功、强平或费用。`censored_pending_command_count` 包含尚未生效的提交与撤单指令，
不能把待生效撤单记作已取消或漏出截断计数。最后可靠价格估值不等于平仓。资金费在同刻新成交前按旧仓位结算；
新段空仓不承担旧段资金费；观察stop为半开边界，事件恰好在stop不属于前段。
每段使用同一四表 ledger；父原生目录的 `episodes.parquet` 保存边界、原因与子账本路径，
实际消费时核边界、资本和账本关系。所有子账本验证完成才原子发布父集合；
失败或取消保留旧好产物，不发布半个完整集合。
报告逐段展示收益/估值、回撤与持仓、成本、覆盖及截断状态；汇总仅为覆盖/缺失范围和
同币种观察段内成本，不输出跨段总体收益、连续最大回撤、Sharpe、假年化或收益均值。
统计按candidate与mode各自依赖结果，连续BLOCKED不阻塞合格的分段结果。
两种执行模式可以显式选用同一模型与预测；账本及统计输出名称由调用者明确给出，
不存在自动生成的研究目标、所有候选必须共享的统计依赖或全研究完成门槛。
统计行携带 `execution_mode`、`episode_id`；`continuous` 表示连续账本，真实独立段使用段身份，
`all` 仅表示覆盖、段数、观察段费用/资金费、成交数量和turnover汇总。
`crypto_portfolio_coverage.parquet` 从研究时间包络和已验证段索引派生半开覆盖/缺失区间，
一般未覆盖区间记录 `not_covered_by_qualified_episode`；PARTIAL 尾部记录实际 `account_stop_reason`，
覆盖只到最后精确估值时刻加 1ns，不另做一次资格判定。
同资产只有段边界及执行假设一致时才可直接并列比较；覆盖不同明确
`different_coverage_not_comparable`，不按段序号配对或重拼现账本。
段间只重置经济状态，不抹掉当时合法已知forecast；价格特征窗口只由真实行情gap按原规则重置。

continuous 模式中，每个候选跨 OOS 窗口使用连续独立资金。窗口边界先处理已到时的市场事件、
资金费和生效订单，再记录边界；窗口结束取消待生效订单。空档期间不发出新信号订单，
持仓继续按实际观测价格估值和结算资金费。OOS 与空档经济结果从同一个已发布账本的
有序边界记录投影；未来盘口不能补作过去的边界估值。

### Maker 校准、队列与报价期限

maker 的 `execution` 提供 `order_quantity`、`max_position`、`maker_fee_bps`、
`latency_ms`、`markout_horizon_ms`、`inventory_target` 与 `inventory_skew`。
后两个字段仅供 fixed maker 使用；calibrated maker 按历史净收益选量，填 `null`，
不保留一份被忽略的固定报价设置。独立 `calibrate_maker` 的 `reference` 明确提供
`order_quantity`、`latency_ms` 与 `min_calibration_samples`，不从 backtest 配置反推。
`models` 的每项声明 `fold_id`、`inputs`、`price_predictions`、`proxy_predictions`，裸表可带
`input_spec`。不提供 `chronology` 时，每项还须给出真实 `window` 与 `cutoff`；
若提供计划，可从对应 validation 窗得到缺省值；显式值独立有效，产物保存解析后的实际窗口与截止。
`cutoff` 决定哪些标签及回放响应已经成熟，不用缺失时钟推断成熟，也不读取模型路径。
校准使用实际拟合截止之后产生、在本次校准截止前成熟的价格模型和 proxy 预测；
按买卖方向保留每个样本的成交路径与价格预测残差，包括未成交样本。
预期收益计算保留成交与残差的联合关系，不把均值相乘代替配对结果。
任一必需方向样本不足只阻塞校准 maker，固定报价诊断与 taker 可继续。

maker 报价采用公开 L2 队列近似和显式延迟。生效时会穿价的 post-only 报价取消，
不会改为 taker 或读取未来盘口重新选量。KEEP 固定旧剩余数量、原队列与真实剩余期限；
真实取消重挂作为独立动作，可同价同量，但重置队列并承担显式延迟。
两者与零/取消动作按实际预期净收益比较，不把 KEEP 偷换成续期。
盘口保留已知买侧下界和卖侧上界；明确删除的价位在该范围内是已知零数量，
不能把最后一个非零档位当成已知范围边界。输出截断真实深度时，已知范围随实际输出裁剪。
校准事件表的 `known_price_bound` 保存每侧范围，价格单位转换和 KEEP/重挂回放沿用同一事实。
生效挂价超出已知范围时，正式账户保留 PARTIAL 前缀，校准排除受影响配对样本；
历史响应不明只限制对应新选择，保留已有订单安排。真正的 post-only 穿价仍是已知拒单。

已生效挂单随后遇到同方向、同观察连接且覆盖合法的穿价成交时，只有已知且正的
`visible_quantity` 才触发全剩余量成交，按自身挂单价记账，不受该笔可见量封顶。
同价成交先按可见量消耗估计前方队列，再用余下可见量部分成交；零可见量既不成交
也不推进队列。实际活动订单依赖未知可见量时，账户保留 PARTIAL 精确前缀；校准排除受影响样本，
按剩余成熟样本判断是否足够。买卖方向使用同一规则，校准与正式执行共用 owner。
成交方向相反或价格不能触及活动报价时，其可见量缺失不影响该订单，也不阻断账户路径。
校准只排除实际依赖未知成交量的配对决策；事件带仍以 NULL 保留该笔未知量，
后续重放的 KEEP 或新报价若依赖该事件，继续明确阻塞，不能将未知量省略或填成零。
这属于固定历史路径上的小单无市场冲击假设，不是历史成交量对任意订单容量的证明。
maker 的 `execution.order_quantity` 与 `execution.max_position` 只约束下单和仓位，不证明订单已足够小；
不能据此将大单模拟成交视为可实现容量。该规则与限制沿既有 `queue_assumption` /
`execution_summary` 写入账本，再由统计和报告原样读取。

每张报价最晚在其自身决策后 500ms 到期；未来盘口或成交覆盖的末端不能预先缩短
该期限或影响 KEEP、重挂、数量及预期收益。报价决策只消费当时已观测到的缺口；
事后证明的无效覆盖用于研究资格，不能冒充当时已知的取消时刻。
校准样本的完整结果覆盖只用于训练期已成熟样本资格。
被排除样本此前已流式写入的合法事件可以保留，不要求每条事件均有最终样本引用；
仍逐事件核对主键、顺序、cutoff 与总行数，逐样本坐标、初始盘口及回放响应也照常验证。

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
仅当替换数量为零、没有活动订单且没有待执行命令时，经济响应必为零，可以跳过事件重放；
预测价格、原生历史事件顺序和样本事件坐标仍须校验。KEEP、排队撤单或 funding 敞口不走此路径。
读取结束、提前退出、阻塞或异常时关闭本次消费的迭代器及所属资源，次生清理失败不得覆盖主异常。
普通 artifact 读取只消费所选成员并核本次科学关系，不在退出时重扫全部文件身份。

Tardis 每个活动消费者拥有流式解码游标，不再保留整请求或跨操作任意重放缓存，
也没有包围多次研究操作的共享 request session。未请求的 symbol/stream 不生成回放 spool。
深度游标只在首个可用 snapshot 之前保存必要增量前缀，完成 bootstrap、观察段变化或
游标关闭时立即释放；关闭缓冲不重置仍需保留的盘口重建状态。
源集合仍决定每资产最细 price_unit、最大 snapshot_limit、累计 source_sequence 与观察/重建段关系，
本次简化不裁掉历史档案前缀。回放保来源顺序和跨档状态，不能仅凭最终 NAV 相同证明裁源等价。
独立游标可能增加 gzip 解码，是取消任意回放缓存的明确取舍；实际解码失败不能被当作正常 EOF。
必要临时文件用完即关闭，实际临时盘位由运行环境决定；不新增缓存索引、LRU、空间准入或调度器。
maker/taker 的待成熟 markout 使用普通 FIFO 队列，结果成熟后移出队列。
PANEL 与 markout 的时间采样均使用该纳秒内全部已观测盘口消息处理后的最终状态；
markout 在时钟越过目标纳秒或确认该时刻事件已耗尽后结算，不在第一条同刻盘口上结算。
实际成交、报价到期、撤换指令、资金费和断线边界继续遵循现有事件顺序；诊断取样不重排成交。
未观测到的未来时刻、覆盖缺口与段边界仍返回未知，不用尾部盘口补造未来 markout。
滑点与已成熟 markout 的平均 bps 按实际成交数量计算流式加权均值，
更新分别保留新旧数量权重。费用、现金流、每单位 markout 和未知 markout 的口径不变。

## 统计、报告与只读展示

收益/成本口径使用完整UTC日收益、ddof=1、
真实 gap 的独立经济贡献、固定实际 turnover 的附加费用敏感性；实际比较集合由调用者显式声明。

正式日频收益仅使用完整落在同一 OOS 折及同一连续账本或独立段内的 UTC 日，以实际连续 NAV 为分母，
基础 `annualized` 分析的年化按显式 `statistics.annualization_periods` 与样本标准差计算，常规 Crypto 日频研究填写 365；
实际参数随统计产物保存，不将其他系数伪称为 365 日口径。年化还要求显式的最小完整日观测数。计划内完整日若缺失边界估值，
正式日频均值与年化指标全部 `UNDEFINED`，不会只选剩余有效日。
现有参考 test 窗结束于 `2026-10-01T23:59:59.900000`，未包含次日午夜的真实估值，
不能据此宣称覆盖一个完整 UTC 日；该范围的正式日频收益与年化指标应为 `UNDEFINED`。
在各自所需输入实际有效的范围内，仍可开展有限的事件级价格响应、执行与覆盖诊断，
并披露样本和缺口，不把这些结果外推为完整日或长期表现。
细粒度事件 NAV、连续回撤、库存、费用、资金费和空档贡献仍保留于账本及统计产物。
统计直接读取指定账本、数值结果和必要输入；缺失与合法UNDEFINED分开保留，不要求完整研究run。

PARTIAL 的全窗净值、收益及依赖未知路径的指标为 UNDEFINED；已观测费用、成交量与换手等以 `known_prefix_*` 指标明确范围。
原账本 summary 只保留一份观测总量，并标注 `observed_totals_scope=known_prefix`。计划日历不缩短为最后已知记录。
零时长端点上的未知成交仍使该范围的持仓最大值未知；已知时段的时间加权矩与其他完整范围可以继续计算。
仅资金现金未知时不抹去已知持仓统计。真正不可表示的非零均值或方差保持 UNDEFINED，不伪装成零。

统计行按 candidate、execution_mode 和 episode_id 保留范围；独立段不拼接为连续收益。
报告展示数据覆盖、因子诊断、模型与验证、OOS 预测、共同样本比较、maker 校准、组合、执行、
费用敏感性、库存、日频统计、validation/test 证据、readiness/blockers 和运行资源观测。
Crypto 的基础日频统计为经济描述，不发布 SPA、StepM、MCS、PSR、DSR 或 PBO 推断矩阵。
显式选择的市场回归独立输出 HAC 推断，要求完整配对日轴；参考序列与环境比较合同见架构文档。
通用比较轴在 UTC 日、OOS 折及 episode 边界切分，首尾不足一天的区间仍保存，不能用于完整日年化或跨段拼接。
省略 analyses 只计算 performance；零收益现金假设不再作为所有组合的隐含指数比较。
通用比较的日频年化默认 365、至少 2 个完整 UTC 日，可通过 statistics.annualization_periods 与 annualization_minimum_observations 覆盖；实际参数写入 comparison_method。

研究图片通过明确输入路径引用已发布数值产物，
使用同一查询入口显示持久化统计与账本曲线。连续曲线和各独立段分别展示，Crypto 时间轴使用 UTC；
完整日不足时保留原指标的 UNDEFINED 原因，不从展示曲线重新计算年化，绘图过程不训练或发布研究产物。

## 研究操作与证据验收

agent 按研究问题显式安排下列操作及实际输入；箭头说明科学依赖，不是程序自动执行的 DAG：

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
-> 显式所选候选统计产物与独立绘图
```

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 选择并验证BTC/ETH期货录制或历史归档 | 按实际所选资产与计算依赖核快照/增量/成交及时间质量；披露缺失资产和成交覆盖，保留独立合法数据与原件证据 |
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
需要对应真实数据和相对简单基线的增量OOS经济证据，仍使用现有科学计算、试验、账本和数值结果，
不预留第二套平台，也不从价格变化虚构精确清算量或L3事实。

## 同输入模型与延迟实验约定

BTC、ETH分别训练和评价；完整输入上的 OLS、Huber、Ridge、Lasso、普通决策树、
随机森林、GBRT、XGBoost、LightGBM 使用相同输入及学习协议，OFI单变量锚点单列。
原配置的跨币/pooled及Maker候选保留；排名集成不作为单币横截面比较。
14项基础到29项多尺度，再逐个信息块增量；Bybit、OKX分别对照后才解释联合结果。
Maker只作公开L2假设敏感性附录；Taker为主结果。

可选解释中的 `permutation.axis=temporal_blocks` 按真实连续段置换；当前任一段行数
不大于 `block_size` 时，整个 explain 调用会报 `explanation_temporal_episode_too_short`，
不会自动跳过短段并输出部分结果。调用者应显式选择满足条件的连续窗口，并披露实际解释域；
不能桥接缺口或宣称覆盖被排除的短段。该可选诊断失败不等于其他研究结果不可解释。

辅助信息的 `prepare.parameters.input_delay_ms` 可配置为非负整数（默认 0），
适用于跨币、现货、衍生状态及外部场所。
信息截止时刻为决策时刻减输入延迟；不改原始观察/事件时间、目标标签或执行时钟。
信息年龄仍相对真实决策时刻计算，继续受既有 `max_age_ms` 限制；
完整流量窗口在首段及真实断段后必须等待延迟后的历史成熟。
输入延迟100/200ms与执行延迟0/100/200ms分开解释。
费用敏感性区分固定成交事后核算与改变费用后经既有边际收益决策重新执行。
默认执行延迟100ms与同资产同模型的显式100ms候选重复时只保留一份配置；历史产物身份不改写。

500ms为主目标，100/1000/5000ms为预定衰减诊断。
免费日期按数据资格选择，独立日期数、缺口与资金费资格必须随结果报告。
长成交整刻钟研究独立于L2回放，不用成交或K线替代盘口。

### 来源子集、结算窗口与费用敏感性

录制 symbols 可选当前支持资产的非空唯一子集；streams、metadata 和完整性检查针对
实际选择的集合，不为了录一只资产强制录完另一只。此选择不扩大市场或执行资产范围。
归档准备按本次输出末端排除更晚归档，保留初始化和历史状态前缀，不把输出起点当作
盘口重建起点；实际消费原件仍必须有真实来源证明。

资金费导入的请求窗口属于结算响应自身，不要求恰好等于单个行情归档全窗。
可明确 output_root；多个行情归档时必须明确输出位置。资金费覆盖、请求limit、
预测观察时点、真实结算与未知更新规则保持，窗口解耦不产生虚假的完整性证明。

固定实际换手的费用敏感性允许有限有符号费率，包括 maker 返佣；额外价差仍须非负。
它只重估同一实际成交路径的费用，不表示已经重跑订单或证明新的成交容量。

## 同一固定信号的多期限响应诊断

保留 100/500/1000/5000ms 分别拟合的独立任务；另以同一个已算好的 500ms 模型预测
评价四种未来价格收益。调用者明确选择已有 PRED、标签和评价时间窗，不重训、不按其他期限
重新选模型，不把四条曲线称为四个独立观测或交易收益。BTC/ETH 分别调用单资产 EVAL；
跨币总览只能在已算各币指标上做明确的币种等权汇总，不能混合原始分数与收益计算 IC。

现有 `evaluate(config)` 可选 `response_diagnostics`，唯一必填项是 `cutpoints`：
键为本次唯一 signal_id，值为严格递增的有限分界值列表。零至四个分界值对应一至五组。
这是数值参数，不要求 MODEL、trial、训练目录、版本证书或额外上游产物。
可选 `reference_window` 只是调用者声明的参考区间，不认证历史、也不设置查看/使用数据的门槛。
标签照常按评价的 `window`、`cutoff` 和自身成熟列取值。

训练分组可调用现有评价 owner 中的小函数 `training_quantile_cutpoints(training_scores)`：
默认五组，使用经验分布反函数分位点，重复分界去重、最大值分界删除；同值不拆分，常数
信号只有一组。输入是调用者明确选择的合法训练参考样本的预测值，通常是已拟合模型在
训练域上的样本内分数，不冒称 OOS 分数；若改用 OOS 参考分数，应明确说明不同设计。
分界只计算一次，评价期不重新分位、不按未来标签可用性调整边界；分数恰等于分界时进
较低一组。若 signal direction 为 lower_is_better，参考分数也须先取同一方向。

示意接线（路径由调用者选择；没有新增 fit 配置或计算入口）：

```python
from quant_research_platform.research.evaluations.engine import training_quantile_cutpoints

request["response_diagnostics"] = {
    "cutpoints": {"frozen_score": training_quantile_cutpoints(training_scores)},
    "reference_window": {"start": "2026-01-01", "end": "2026-01-01"},
}
# request 继续使用既有 inputs、predictions、evaluation、window、cutoff。
# evaluation_axis 为 single_asset_time_series，信号 transform 为 identity。
# 四个标签须明确声明 decimal_return；图中 bp = 比例收益 × 10000。
```

### 数值及样本口径

一次共享输入扫描和成熟标签处理，同时产生两张可选 EVAL 表：

- `response_metrics`：IC、配对 N、有限 eligible 信号数、覆盖率、最高固定组减最低固定组
  的均值差（bp），及明确的未定义原因。
- `response_groups`：每个固定组的均值、中位数（bp）、N 和未定义原因。固定空组仍有
  N=0 的行，均值/中位数为空；不会把它换成附近有样本的组。

两表均以 `period` 区分 `daily`、`pooled`、`daily_equal_weight`，以 `sample_scope`
区分 `common` 与 `own`。common 要求同一锚点的本次所有期限标签和信号均有效，为主比较；
own 只要求对应期限标签与同一信号有效，辅助披露覆盖。两种域均不插值、不补锚点、不补
标签、不按真实极端行情过滤。方向、列选择与实际数据错误沿用原有数值/来源合同。
日坐标保留该日实际最后一个输入锚点，不伪造日期中间的观测。

`daily_equal_weight.ic` 是有效每日 IC 的等权均值；`ic_std` 是这些每日 IC 的样本标准差
（ddof=1），`valid_day_count` 为有效 IC 日期数。`pooled.ic` 才是全窗合并配对样本的 IC。
`daily_equal_weight.mean_return_bp` 是各组有效每日均值的日期等权值，`mean_std_bp`
为其跨日样本标准差，`valid_day_count` 为有效组均值日期数；汇总行不计算/伪称合并中位数，
`median_return_bp` 留空、原因为 `not_aggregated`。真正合并中位数在 pooled 行中。
汇总行 N 是实际配对数之和，coverage 是这些计数之比，不是日期等权覆盖率。
日等权组差单独平均同日两端组都可定义的组差，不相减不同有效日期集合的组均值。
常数信号的单组差未定义，不能写成“高减低为零”。不足两日不计算跨日标准差；不可表示的
极大离散度留空，不中断其他可定义统计。当前只有描述统计，没有自动 bootstrap、p 值或
显著性判断；重叠未来收益不被当作独立样本，也不跨真实缺口拼接收益。

启用此选项时，主 metrics 与响应表共用批量 Pearson 结果。批量核复用现有稳定中心化和
相关结果核，以金融/数值量纲容差验证；未请求诊断时仍使用原有精确增量累积，适合真正
流式消费。不同数值算法不要求逐位相等，也没有精度参数或容差准入配置。
组差的近零浮点舍入以 1e-10 bp 绝对容差验收，不为机器精度噪声增加高精度运行框架。
均值用稳定求和，先完成 bp 换算再作最终浮点表示，避免可表示的极小 bp 被提前抹零。

精确 pooled 中位数需要保留本次明确窗口的选中数值；内存为 O(N × 选中列数)，没有训练
截断、自动采样、资源准入、额外原件扫描或跨任务缓存。现有 PRED 可以在 MODEL 和 trial
不再存在时继续消费。两张可选表同时含日表与全窗摘要，读取时不以 time_window 假装重算
摘要；要只展示某日，读取后选择 daily 行。

### 两类图

继续使用现有 `reports.plots.plot_series` 的类别轴，不新增渲染器或自动编排：
第一张选择 common 的日等权 IC（并另列 pooled/自身 N）；第二张选择同口径各固定组的
日等权均值 bp。类别按明确的 100、500、1000、5000ms 顺序排列，图中文字、图例和轴用
中文，来源和样本用途保留。中位数、N、覆盖和每日结果直接来自同一 EVAL，不能为不同图
重新读原件或重建不同标签。组差仅是价格响应诊断，不能冒充扣费、延迟和执行约束后的策略收益。

共同训练域／共同评价域的信息增量对照使用[显式实验配方](research-rosters/crypto/common-domain.md)，保留各配方自身合法域主结果，不增加模型参数或产物准入。

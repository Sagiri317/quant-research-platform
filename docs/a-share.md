# A 股低频研究

本文件说明研究目标、支持能力、研究推进顺序和量化合同。
能力说明以生产者和持久化消费者为依据；合成验证、真实数据覆盖和经济有效性分别提供证据。
正式协议是 a_share_low_frequency，只读目录 profile 为 a_share_csmar_choice；
计算操作各自接收显式配置，不通过 profile 绑定或展开整项研究。
本文件不是投资建议，不授权实盘、模拟账户或预建未来功能。

## 1. 研究目标与边界

建立一条 PIT 正确、能解释并复现研究结论的 A 股低频流水线：
CSMAR 市场/指数/行业/分析师事实及 Choice 财务事件版本 → 因子/标签/研究数据集 →
因子推断、季度滚动模型与持久化 OOS 预测 → 简单组合和指数增强 →
交易限制/成本/风险/容量/归因 → 开发与冻结留出分开的报告。

研究目标是完整真实数据下的可信研究，而不是因子或模型数量。
覆盖历史退市证券、实际首次披露及更正版本，保留输入/参数/试验/产物谱系，
能够解释亏损、阻塞、未定义指标和容量限制，不要求研究结果必然盈利。
界限是只读研究与连续权重历史执行；A 股日内数据只作为未来执行真实性研究候选。

## 2. 支持能力与证明边界

| 能力 | 支持内容 | 证明边界 |
|---|---|---|
| 数据与 PIT | CSMAR 市场事实、Choice 原生请求归档及事件版本适配、合法可用时点与修订传播 | 适配能力不表示全 A 股原始或修订历史齐备 |
| 场景与因子 | 配置选择 pv_all、pv_expanded、pv_nf_expectations、fusion_nf_core、fusion_nf_expectations；价量库50项与3项路径机制，另有共享 beta/残差波动率、Fundamental19、预期4与3项预期机制，保留 PV25/CORE6 选择 | 因子定义存在不表示真实数据齐备、正式实验完成或发现有效 alpha |
| 特征与标签 | 横截面资格、MAD/中性化/zscore、缺失指示、成熟标签、20日模型目标与辅助期限 | 缺少必要事实的真实样本仍不可用 |
| 模型与预测 | 等权/IC 排名集成及 OLS/Huber/Ridge/Lasso/决策树/RF/GBRT/XGBoost/LightGBM；研究 JSON 显式选择模型，共享季度候选预测、G/Q 选择与最终冻结预测 | 算法可配置不表示已纳入某项研究或完成真实留出验证 |
| 组合与推断 | 配置简单分位数组合或 CSI500/CSI1000 增强，选择日/周/月再平衡；复用自融资账本、成本/风险/容量/归因及预注册统计 | 配置合法不表示所有市场环境均可成交或优化可行 |
| 报告与展示 | 独立操作、显式数值产物、CLI/API 和只读研究图片 | 入口存在与合成运行不证明真实经济有效性 |

真实数据覆盖和开发完成情况以实际输入、运行结果及研究记录为准；能力清单只说明已有计算能力。

## 3. 研究推进顺序

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 按消费者盘点原始输入与缺口 | 量价、独立预期、财务版本和指数增强分别列资格；缺版本只阻塞实际依赖消费者，额度与权限明确 |
| 2 | 完成真实原始输入验收 | 字段、单位、键、生命周期、历史行业、费率和全收益基准齐备；区别范围内 PIT 适用与全量覆盖 |
| 3 | 运行开发区间的真实完整链 | 保持已冻结研究配置的窗口、模型与组合集合，报告所有阻塞和未定义项；先验证正确性，不做大规模调参 |
| 4 | 固定一份可讲解的研究报告 | 因子→预测→执行→净收益与风险/容量/归因勾稽，说明合成与真实证据的区别 |
| 5 | 安排独立测试期检验 | 事先确定模型、排名方向、预处理、Alpha 校准及重训规则，再按显式时间窗口执行2024–2025检验；测试期结果不用于调参或选择 |

推进顺序不改变正式研究协议。未取得必要原始资料，不用清洗、固定滞后或缩小历史来补造 PIT。
数据获取遵守实际访问许可与下载额度；平台按显式输入和时间窗口计算，按实际拟合截止及标签成熟时点核验资格。

## 4. 扩展研究方向

核心完成后，按明确缺口评估四类候选：更完整的会计/事件及分析师研究；
多期优化和更真实的成本/容量；NLP/LLM与深度表格/时序模型；
更高级资产定价、市场状态和动态择时。
每次只引入能相对已声明简单基线提供增量 OOS 证据的一项，复用现有数据、试验、账本和报告 owner。
它们不属于既定正式实验，也不构成新增依赖、因子搜索或接口预留的理由。

## 5. 正式研究约定

研究合同按数据与 PIT → 因子与数据集 → 模型与预测 → 组合与核算 → 统计与报告组织。
参数与公式写在所属主题；实施流水账、账户状态和操作路径不作为合同。
通用资源、身份与产物发布见 [架构约定](architecture.md)，来源布局见 [数据合同](data.md)。

### 5.1 研究目标与正式范围

QRP A股主线是：

> **只读、用于研究/回测、PIT 正确、可复现且可审计的 A 股低频系统化权益研究流水线。**

正式协议标识：

`protocol_id = a_share_low_frequency`

研究选择及实际参数随本次产物记录；语义变化后由研究者明确重算范围并使用新输出路径。

正式经济研究期：

`2014-01-01 至 2025-12-31`

预热数据按因子、标签、风险和会计的真实依赖向前读取，不以研究期起点硬截历史状态。
来源准备、加工与组装分别调用 `prepare-source`、`prepare`、`assemble`。agent 显式提供
Choice 来源的 `start_date`、`end_date` 及输出映射；`prepare-source` 不自动扩大来源窗口。
因果历史加工从显式输入已发布的 `date_min` 读取至实际输出截止，不以固定 2011 年或
请求起点裁掉已给出的历史状态。标签与执行掩码从同一给定输入读取所需未来本市场会话，
不能补造未提供的端点；processed/PIT 与研究数据集只发布精确请求窗口。
缺失会计依赖保持缺失值，不填零；缺少必需输入时不自动补跑来源准备。
Choice 的请求终点同时约束已公开信息：首次披露和修订事件晚于终点时不进入 Interim，
也不要求其未来快照；扩大窗口后，窗口内修订仍须具有完整可验证的快照。

正式市场范围：

```text
Markettype in {1, 4, 16, 32}
= 沪市 A 股主板
+ 深市 A 股主板
+ 创业板
+ 科创板
```

不含 B 股和北交所。

历史证券范围必须保留退市股票的真实生命周期；不得用当前存续证券范围回填历史。

正式研究成功只表示协议正确执行且证据完整，不表示策略必须盈利。

### 5.2 数据责任归属与真值

同一经济事实只有一个活动供应商责任模块。

主要责任来源：

```text
CSMAR
  市场 / 指数 / 行业 / 分析师结构化事实
  财务原始数据仅保留用于验证

Choice
  唯一正式财务 PIT 来源：必需报表项、普通披露、
  更正事件与不可变历史时点版本
  明确采纳后再纳入未来业绩预告/快报披露事实

交易所/公司/中证指数官方来源
  仅用于验证或填补必要缺口
```

不得在同一正式语义上无规则混用多个供应商。

规范 `data_id -> source` 真值来自类型化中间层数据规格；不维护第二份来源映射。

供应商字段名只在原始数据 -> 规范中间层边界出现；下游不得泄漏供应商专属列语义。

原始获取与历史 SDK EndDate 截止于 2025-12-31，覆盖既定研究所需的 2011–2025 连续历史及必要预热。
Choice 全 A 股及历史退市范围的原始收集不以 CSMAR 范围或正式研究股票池为上限；
正式研究范围仍按 §5.1 的市场集合约束。保留供应商原生文件及 CSMAR 验证输入。
SDK 成功不证明全市场、全部报告期或修订历史齐备。
范围内数据内容满足 PIT 可用时点/版本要求时，可认定该范围适用；质量、清洁度与后续处理单列。
代码和合成验证可先行，真实财务适配/PIT/研究须等待必需原始齐备。

### 5.3 PIT 会计与可用时点

#### 5.3.1 保守 PIT

正式财务研究使用保守 PIT：

```text
报告期
!=
普通首次披露
!=
更正披露
!=
available_at
```

日期型披露若无法证明在当日决策时点前已公开，则默认：

```text
披露日期
-> 下一市场交易日的 available_at
```

禁止：

```text
报告期 + 固定滞后
用最终修订值回填历史
将供应商最终值视为历史已知
```

无法证明可用时点的观测必须明确拒绝。

首次公开财务规则：财务 PIT 只有在 Choice STMTACTDATE
和原始报表/版本时点证据证明对应值当日已公开时，才可纳入招股书财务披露。
ACTUALISSUINGDATE 解析并归档定期报告发布日期，核对请求确实抓取该字段；
当前不据此校验招股书、首次披露和定期报告日期之间的先后关系。
日期本身不能制造历史值；未经证明的版本继续阻塞。
仅有日期的 IPO 披露，同样适用下一交易日可用规则。

#### 5.3.2 财务披露键

普通财务披露规范键：

```text
asset_id
report_period
first_disclosure_date
```

不含供应商表名或 `table_id`。

CSMAR `DeclareDate` 只表示会计差错更正披露，不得冒充普通首次披露。

#### 5.3.3 事件版本真值

财务报表真值是不可变报告/事件版本，而不是日频面板。

每个版本至少能够定位：

```text
asset_id
report_period
报表口径
业务字段/值
版本标识
首次披露可用时点
适用时的更正可用时点
```

更正不重写旧事件，而是追加新版本。

同一业务键和可用时点的数值冲突逐字段记为未知并保留诊断；独立指标与其他报告版本继续使用。

#### 5.3.4 派生会计事实

日频财务 PIT 是可以从事件版本重建的派生数据集。

流量项目先在报告事件层计算：

```text
TTM
YoY
期初/期末存量平均值
```

再按合法 `available_at` 做日频观测的时点查询。

存量项目使用 PIT 期末值。

TTM 唯一公式：

```text
TTM_t = YTD_t + FY_{t-1} - YTD_{t-1,same period}
annual report TTM = annual value  # 年报 TTM 取年度值
```

任何组成项都必须满足：

`component.available_at <= current event cutoff  # 组成项在当前事件截止时点已可用`

若后续更正改变了被后续 TTM/YoY/平均值依赖的组成项，派生会计事件必须从更正合法生效时点反映新的合法视图；不得只修旧报告期而不传播到依赖它的派生事实。

TTM、存量平均值和复合因子按完整公式以浮点数计算；多项金额相消使用稳定求和。
派生值非有限时，只将该资产、指标、报告期的相应值记为未知，其他有效事实继续计算；
因子将缺失组成项、非正分母及不可表示的最终结果记为 NaN，不截断或补零。

禁止用通用 `previous_value` 同时表达：

```text
上一季度
上年同期
期初余额
```

这些关系必须由明确的会计期间解析器定义。

#### 5.3.5 报表口径

正式非金融企业基本面因子使用合并报表。

归母与合并口径：

```text
ROE / EP / BP
  -> 归母净利润 / 归母权益

ROA / CFO / accruals 与企业经营质量
  -> 合并净利润 / 合并资产 / 合并经营现金流
```

### 5.4 市场交易日、生命周期与交易状态

#### 5.4.1 稠密市场交易日面板

面板使用：

```text
真实上市生命周期
x
每个交易所开市交易日
```

`Filling=2` 的节假日填充不属于交易日；`Filling=1` 的停牌仍属于交易日。
原始行按自身 `market_type` 的日历校验，稠密面板按当日有效的所属市场保留开市行；
另一市场开市不能把本市场的休市变为交易日。市场日历区分已知开市、已知闭市与未知；
必要日期缺少权威日历记录时PANEL明确阻塞，不能当休市删除并压缩标签或回看轴。市场归属来自逐日行情，沿日期向后保持，
包含休市填充行所记的归属变化，并跨批次延续；不将首条行情的市场固定用于整个生命周期。
消费方在依赖规划实际扩展后的 PANEL 请求范围内（含所需回看及标签覆盖），
按已证明的有效上市生命和当时市场归属，
核对每个自然日都有显式开市或闭市记录；不要求无关市场、归属前或上市前的日历。
已证明闭市的日期跳过；已证明开市但缺行情的日期保留稠密 NULL 行。
请求前仅用于上市年龄、市场归属或停牌状态初始化的旧事件，不要求完整的旧日历；
旧未知日期在内部阻断未经证明的价格沿用，后续真实有效报价可重置状态。
这些内部屏障不发布为交易行，也不计为已证明的开市会话。
这是消费所需时间轴的完整性要求，不假定供应商原件天然完整，也不推测周末或节假日状态。
CSMAR 股票主表的声明 schema 不包含 `market_type`；公司状态与上市日期按时点向后 as-of 关联。
板块改名本身不构成供应商代码迁移事实，上市年龄继续按所属市场日历累计。

每次 PANEL 请求只规范化一次完整股票主表、市值与供应商行业历史，之后按原批次计算。
公开行业快照先按真实知识窗口及资产范围选择；异常字段按依赖局部解释，冲突从实际可知时点影响对应行业前沿，未来异常不污染此前结果。
该复用仅在请求作用域内，不写长期缓存，也不裁掉必要的历史状态或未知屏障。

不得用查询窗口的第一/最后交易日推断上市或退市。
同代码的已证明非重叠上市区间按当日主表 as-of 记录保留；同一上市日的状态修订仍按生效日消解。
CSMAR主表按实际Statdt保留同一上市生命期内不同状态记录；不同状态日期不能因共用Listdt被当作主键冲突。
区间空档不生成 PANEL 行，新生命周期重新计算上市年龄，停牌价格不继承上一段；重叠或倒序区间明确失败。
下游以 PANEL 的 (asset_id, listing_date) 作为同一价格历史的分组：次日执行资格、
1/20 日等前瞻收益及其成熟时间、价量滞后与滚动窗口均不得跨上市区间衔接。
旧区间末端没有同区间未来价格时，标签与成熟时间保持缺失；新上市首日不继承旧价格。
每个生命周期保留首次真实市场归属，即使相邻两次上市的市场相同，也不能沿用旧生命事件。
同一生命周期内的停牌、闭市和市场迁移继续使用权威交易轴，不切断合法历史。
组合持仓与估值的两个价格端点也须证明同一上市身份；不能证明时 BLOCKED，
不推定退市兑付、卖出或释放现金。
财务及行业事实仍按各自公司事实和可用时点规则延续，不随价格窗口清空。

缺少合法股票主表/生命周期真值的资产，不进入正式研究。

#### 5.4.2 停牌语义

停牌交易日：

```text
复权价格沿用旧值
session return = 0  # 停牌交易日收益
volume = 0
amount = 0
turnover = 0
positive-denominator liquidity contribution such as Amihud = NaN  # 分母要求不满足时，流动性贡献记为缺失
```

复牌收益相对上一个真实有效交易价一次性实现。

仅已知非交易行沿用旧价格，原价与复权价分别使用同口径旧值；缺少有效旧原价继续缺失，
不得以复权价替代原价。正常交易行的原价/复权价缺失不回填，不覆盖其来源收益。

所有正式滚动窗口按完整市场交易日数计算，不压缩掉停牌交易日。

停牌不自动排除研究域 `eligible`；因子各自核验所需报价与观察，买卖执行独立核验 `can_enter` / `can_exit`。

#### 5.4.3 历史行业

逐资产历史行业状态由 PANEL owner 统一裁定。CSMAR 的 `STK_INDUSTRYCLASS`、
`STK_LISTEDCOINFOCHG`、`STK_INDCLASSIFYVERSION`、`STK_INDCLASSIFYSETS` 分别提供
分类事件、公告、标准版本和层级；CSRC/CAPCO 官方公开快照通过独立
`reference/a_share_industry_snapshots` 输入提供有来源证明的观察状态，不冒充 CSMAR 事件。

PANEL 只纳入 CSRC2012 或 CAPCO 分类事件；其他分类体系的事件不参与状态选择或未知屏障。
行业状态保留上述标准身份。新制度的 ImplementDate 不自动提供每只股票
的新分类，也不自动清空已公开的旧状态。分类事件（快照为 observation_date）与当前观察日期
必须处于该状态所属的同一且唯一有效版本区间 [ImplementDate, AbandonDate)；缺版本、
空区间、同标准重叠版本及明确废弃均不合格。已采用的新状态废弃后不自动退回旧标准。

CSMAR 事件须以股票、标准、原始实施日绑定唯一变更公告；ChangedItem 精确识别标准，
Value_After 须等于事件原始行业名称，不能仅凭同名拼接。公告只有日期时，在公告日之后
首个已有交易日才可用于信号，且实施日必须已到达。没有对应公告不能当作已知，
不能用后来公告的 Value_Before 回填初始状态。

官方快照分别保留观察日期、真实页面公开日期、来源标准、原分类字段和原件页码或表/行定位，
并以原件、页面及候选提取文件的精确身份绑定来源。其 timestamp 是日期型公开日的次日
可用下界，实际信号仍只在现有交易日轴上使用。Interim 主键为
`(source_asset_id, timestamp, snapshot_id)`，按 timestamp 年份分区；规范输出每份快照同一证券一行。重复证据合并，相互冲突的字段保持未知，不任意挑选赢家。
不得把季度名、统计期末、附件迁移路径
当作更早公开证据。资产须满足现有身份、正式市场范围与观察日存续条件；类别、次类和
行业码须与同标准词典父链一致。公司简称不等于行业名称，字典继续唯一解释行业层级。

已知但尚未公布的新事件，公布前保持当时已知的旧状态；仅制度变化或快照缺某资产，
不构成该资产的新行业事件。已存在而公告未知的新实施事件，从该实施日起表示旧状态
覆盖资格未证，行业四列一致 NULL；这不是声称市场当时已知新分类，不为它制造 available_at。
只有同日或更晚有效、具有真实公开证据的状态才能恢复资格。晚公布的旧事件或旧观察快照
不得覆盖较新状态、跨过未知屏障或反向改写更早的信号。首个合格公开初值之前保持未知。

同资产、标准、有效时点的公开分类冲突，以及同一可用时点和有效时点的跨标准冲突，
只使受影响的行业前沿未知，不能靠输入顺序裁决；此前已知状态及独立价格不被追溯抹除。相同事实的多份证据不重复形成经济状态。无合法映射、名称不符或
身份未证保留资格缺口，不删除资产、交易日或基础价量事实。

PANEL 保留 `industry_standard_id`、`industry_level1_code`、`industry_level1_name`、
`is_nonfinancial_industry` 的字段语义。特征中和与组合风险共用标准加代码的行业分组，
例如不同标准的同一字母不能合为一个类别；不把拼接后的分组值改写到原始 code 字段。
缺少合法历史行业时，依赖目标保留真实阻塞；已有行业行不等于模型、共同验证域或组合通过。

#### 5.4.4 无涨跌幅限制、ST、PT 与涨跌停

A 股市场责任模块必须按历史有效日处理：

```text
ST / *ST / PT
停牌
涨跌停状态
不设涨跌幅限制的例外
板块/规则变化
```

不得用单一当前规则覆盖全部历史。

CSMAR `BDT_Trdchg.Chgtype` 是两字母的前状态→后状态迁移码，由 source interim
唯一解析：A 正常、B ST、D *ST、C PT；目的状态 S/T/X（暂停、整理、终止）不推断
ST/PT，保留 nullable boolean，并继续由当日交易事实及独立生命周期 owner 判定。
interim 保留原迁移码、公告日和执行日；PANEL 只消费规范状态，在公告和执行均已到达后
使用当时已知的最新执行事件。迟到的旧执行事件不得覆盖较新的已知状态；相同执行日
的冲突从双方事实均已可知时只使对应状态字段未知；同日独立交易状态仍可提供有效证据，不抹除其他资产或价格。较新执行事件可恢复状态，相同事实不重复形成状态，不依赖原输入顺序。

### 5.5 价格、收益与市值真值

股票全收益真值：

```text
Dretwd
Adjprcwd
```

复权开盘价：

`adj_open = Opnprc * Adjprcwd / Clsprc`

价值因子的市值分母：

`Dsmvtl3 total_market_cap`

规模/流动性使用：

`Dsmvosd float_market_cap`

CSMAR 字段单位依据本地 `data/reference/股票市场交易(升级测试) 数据库说明书.pdf`
的 76 页版本：PDF 第 21–24 页（正文第 17–20 页），表三 `TRDNEW_Capchg`
及表四 `TRDNEW_Dalyr`。PDF 原件不进入 Git；本节保存处理所需的口径。

| 来源字段 | 来源单位 | QRP 规范口径 |
| --- | --- | --- |
| `Dsmvtl3` | 千元 | `total_market_cap`，在 source interim 乘 1000 一次，A 股为人民币元 |
| `Dsmvosd` | 千元 | `float_market_cap`，在 source interim 乘 1000 一次，A 股为人民币元 |
| `Nshrttl` / `Nshra` | 股 | `total_shares` / `a_float_shares`，不按手或万股缩放 |
| `Opnprc` / `Clsprc` / `Adjprcwd` | 元/股 | A 股价格以人民币元/股计；复权价不改变货币单位 |
| `Dnshrtrd` / `Ahshrtrd_D` | 股 | 正式成交量为两者之和，保持股单位 |
| `Dnvaltrd` / `Ahvaltrd_D` | 元 | 正式成交额为两者之和，A 股保持人民币元单位 |

`Dsmvtl3` 与 `Dsmvtll`、`Dsmvtl2` 不是可互换的总市值字段。
说明书的 `Dsmvtl3` 口径分别计入 A、B、H 股价格及适用汇率；
其 A 股部分的股数为 `Nshrttl - Nshrb - Nshrh - Nshroft`。
因此不能用 `Nshrttl * Clsprc` 重建这一价值分母，也不能用另一总市值列补缺。
供应商币种说明不扩展 QRP 已批准的研究市场；上述人民币单位限定于正式 A 股消费者。

供应商市值缺失时保持缺失，不用历史股数乘现价代填。股本事实保留独立口径；
规模、换手、价值分母和研究资格按各自缺失规则处理。

正式成交额 Amount：

`Amount = Dnvaltrd + Ahvaltrd_D`

已知不适用的盘后成交额为 0；适用但缺失时，盘后成交额及依赖它的总成交额保留 NULL，其他独立字段和记录继续保留。

2011–2025 来源适配边界：Markettype 32科创板自2019-07-22、16创业板自
2020-08-24适用；Filling=1/2为已知全天非交易。只有已知市场、状态及历史日期能证明
不适用时才将盘后 NULL 置 0。适用交易行缺失或适用性未知时，仅相应盘后量及总量保留 NULL；依赖该量的消费者按自身有效样本域处理，已有非 NULL 值保留。
2026以后NULL不套用旧适用表，本条不声明已支持2026市场规则。
摄取保留混合来源文件中的Markettype 64北交所记录：2025及以前属于已知盘后不适用域，
不改变正式研究CURRENT_MARKET_TYPES。依据为[北交所2025交易方式问答](https://www.bse.cn/qt/200025234.html)
及[2026新增盘后安排起草说明](https://www.bse.cn/uploads/6/file/public/202604/20260410155845_hs300rhb80.pdf)；
不将2026新规则回填历史，也不授权北交所研究执行。
依据为[CSMAR官方说明书](https://file.csmar.com/group1/M00/AA/99/CuIKV2XAnKiABeLbAB4nDGC5Ano840.pdf)
字段索引与fields_zh、[科创板开市说明](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20190821_4884321.shtml)、
[创业板实施日期](https://www.szse.cn/aboutus/trends/news/t20200821_580924.html)及
[深交所停牌盘后规则](https://www.szse.cn/www/investor/index/update/t20200902_581220.html)、
[上交所停牌盘后规则](https://edu.sse.com.cn/service/hotline/qa/c/4928890.shtml)。
以上为历史适用范围的来源依据，不将此说明扩展为 2026 以后规则支持。

### 5.6 基准真值

CSI500 / CSI1000 增强研究要求权威历史基准事实。

责任归属：

```text
IDX_Smprat
  -> 正权重成员 + 精确权重真值

IDX_Chgsmp
  -> 调样历史的验证/研究

IDX_Idxinfo
  -> 仅标识
```

`Weight` 百分比只转换为比例一次，原始供应商归档保持不变。
CSMAR《市场指数 数据库说明书》PDF 第 18 页（印刷页 13）将 `Weight` 声明为
`N10,4` 百分比，对应比例量子 `q = 1e-6`。
同一基准、同一日期实际观测到 `n` 个正权重，总和 `s` 须有限，且满足
`|s - 1| ≤ n*q + (n + 2)*float64_eps*max(1, |s|)`；机器余量仅覆盖数值转换、
除以 100 和正值求和的有限精度。供应商未声明舍入或截断方式，不能擅自采用半量子界。
这一门槛只检查源声明精度下的量化一致性，不证明成分完整；`n` 是实际观测数，
不按指数名称硬编码 500/1000，也不把 `IDX_Chgsmp` 变成第二份组合构建真值。
NULL、非正权重、重复键、精确身份不符或总和非有限／超界仍拒绝，
不按真实导出的更粗网格或最大偏差拟合更宽门槛。
通过后，由上述唯一规范化 owner 除以该组实际总和，生成单位组合权重
（浮点舍入范围内总和为 1）。这保留发布权重的相对比例，不用等权替代，
也不以归一化接受超出门限的输入。后续组合执行与归因消费这一规范权重，
不各自再次归一化或通过放宽消费者校验容忍初始持仓与 NAV 的差额。

禁止：

```text
用当前成员回填历史
对陈旧权重长期向前填充
将精确日期成员要求作为第二份组合构建真值
```

正式再平衡日缺少权威权重快照 -> 对应增强目标 BLOCKED。

基准绩效必须使用已验证的 CSI500/CSI1000 全收益序列标识，并匹配正式执行开盘到执行开盘的经济区间；不得把收盘到收盘的价格指数收益冒充正式基准收益。

全收益原生开盘值优先。仅原生开盘缺失时，现有事后组合统计 owner 可用同一来源实例、
同一交易日的已验证价格／全收益指数配对，计算
`全收益开盘 = 价格开盘 × 全收益收盘 ÷ 价格收盘`。依据
[中证股票指数计算规则（2024）](https://oss-ch.csindex.com.cn/contract/cms_add/20240726155028-%E3%80%8A%E4%B8%AD%E8%AF%81%E6%8C%87%E6%95%B0%E6%9C%89%E9%99%90%E5%85%AC%E5%8F%B8%E8%82%A1%E7%A5%A8%E6%8C%87%E6%95%B0%E8%AE%A1%E7%AE%97%E8%A7%84%E5%88%99%E3%80%8B.pdf)，
配对指数同日共享调整市值，各自除数日内固定；派生结果保留输入公布精度，
不声称还原未公布的原生数值。该计算只用于事后开盘到开盘绩效，不进入开盘决策或 PIT 特征。
必要日期或行情值缺失仍为 BLOCKED；派生所需各值须分别有限且严格为正，
不得用负值相消、跨日填充或切换指数别名补足。保留源表 NULL，不写回 raw/interim，
也不使用收盘收益字段替代这三个必要行情值。

官方基准本身不含摩擦；实际增强跟踪组合中由基准驱动的交易，仍通过真实执行/成本链。

### 5.7 正式场景与资格

正式场景由研究配置选择：

```text
pv_all
pv_expanded
pv_nf_expectations
fusion_nf_core
fusion_nf_expectations
```

不跨场景选择优胜模型；各场景分别形成独立评估、模型和组合结论。

`pv_nf_expectations` 要求完整历史行业及非金融身份，但不要求财务 PIT；融合场景继续要求合法财务 PIT。
所选财务因子及组合风险暴露共同决定必要报表依赖；未选因子的缺失表不阻塞独立消费者。

#### 科研选择与独立操作

场景提供因子和经济资格的默认组合，显式配置决定实际选择，研究 JSON 决定评估因子、模型输入与参数、比较基线、
组合集合及访问窗口。配置中未选择的能力不自动补齐；配置仍须符合实际字段、单位、标签成熟和明确冻结的科学边界。

随仓科研选择参考保留以下研究选择，因子数仅统计 `evaluation.factors`；这些文件不直接交给单操作 CLI：

| 配置 | 场景 | 评估因子 | 模型 | 组合定义 |
|---|---|---:|---:|---:|
| [core6.json](research-rosters/a_share/core6.json) | pv_all | 6 | 11 | 5 |
| [pv25.json](research-rosters/a_share/pv25.json) | pv_all | 25 | 11 | 5 |
| [fusion_nf_core.json](research-rosters/a_share/fusion_nf_core.json) | fusion_nf_core | 44 | 11 | 5 |
| [fusion_nf_expectations.json](research-rosters/a_share/fusion_nf_expectations.json) | fusion_nf_expectations | 48 | 11 | 5 |
| [pv_expanded.json](research-rosters/a_share/pv_expanded.json) | pv_expanded | 52（50价量+2暴露） | 11 | 3 |
| [pv_nf_expectations.json](research-rosters/a_share/pv_nf_expectations.json) | pv_nf_expectations | 29（PV25+预期4） | 11 | 3 |

模型列统计两种排名集成及九种监督模型；六份配置均保留完整模型集合。
组合列统计参考 JSON 中的定义数量，研究者分别为 G/Q 构造显式候选；它不是账本产物总数。
只读 profile 不生成组合或账本；各次计算由 agent 显式调用。

各单操作只接受自己消费的配置；科研选择表是供agent使用的设计参考，不由程序展开研究。
训练、预测、评分、选择、组合和统计必须独立指定输入与新输出路径。
测试集核真实冻结选择、最终模型和必要校准。

`fusion_nf_core.json` 选择 PV25 + Fundamental19，共44个因子；名字不表示只选 `FactorSet.CORE`。
配置中这44个因子继续分别做单因子FMB、RankIC/HAC/FDR及模型OOS研究。
多因子 `core_multifactor` 只用预先选中因子与 `FactorDefinition.factor_set=CORE` 的交集：
CORE6/PV25/PV-expanded各6，fusion_nf_core为12，fusion_nf_expectations为16。

研究数据集对外只有一个样本过滤真值：

`eligible`

`scenario universe` 是协议/组装内部概念，不与 `eligible` 并列持久化为第二真值。

A 股基础研究范围（`eligible`）：

```text
正式 A 股市场范围
listed >= minimum_listing_sessions  # 默认 252，本市场会话数
默认排除 ST/*ST/PT，可分别配置 exclude_st / exclude_pt
仅使用信号收盘时已知信息
```

上市会话事实由同一个市场日历 owner 生成。`listing_age_trading_days` 仅在从上市日开始的市场归属有证据时保存精确年龄；首次行情的市场不能反填到上市日，未知年龄保持 NULL。`has_252_trading_sessions` 是 nullable boolean 原始资格事实：完整历史时准确判断门槛；初始历史未知但已证实归属区间累计达到 252 个合法市场会话时为 True；尚不能证明时为 NULL。迁市场按各段真实市场日历计数，不把无报价的开市会话压缩掉。它与研究市场范围及所选 ST/PT 排除事实共同派生 `eligible`，不是另一份样本过滤结果。未知必要事实不肯定为合格，原始 nullable 事实与已知排除继续区分。停牌、价格或成交量缺失不自动改变研究范围；因子有效域与执行许可由各自消费者核验。独立 `assemble` 可设置 `minimum_listing_sessions`（默认 252）、`exclude_st` / `exclude_pt`（默认 true）；改变年龄阈值时使用已证明的 `listing_age_trading_days`，设为 0 时不要求年龄事实。关闭 ST/PT 排除时不要求对应事实，解析参数保存到原生 spec 与 metadata。首次已知归属之前的 PANEL 行保持未知市场和未知价格，不从未来借市场、补价格或默认可交易。

非金融融合场景还要求：

```text
非金融行业
合法 PIT 上下文
```

未来执行事实不参与 `eligible`。

### 5.8 执行事实与模型输入边界

执行掩码是独立市场事实：

```text
can_enter
can_exit
```

信号行 `t` 上的 A 股 `can_enter` 表示：

`t+1 市场交易日复权开盘入场是否合法`

组合消费信号 `t` 的 `can_enter[t]`，不得再次位移到 `t+1`。

执行掩码不得持久化为面向模型的研究特征/上下文列；组合通过明确连接使用执行产物。
普通 forward-return 与 ON/ID endpoint 标签只依赖 PANEL 价格及端点观测事实，不依赖 execution_mask。
涨跌停、买卖许可与组合清算规则不决定固定期限价格标签是否有效。

QRP 训练标签采用唯一的固定期限价格收益定义：两端必须是真实正常交易观测，
即 `filling_flag=0`、`is_trading=True` 且复权开盘价有限、为正。与该期限无关的更晚事实
不得反向改写已成熟或已发布的历史 model_target。未来执行事实不得进入特征或 eligible，
尚未成熟的标签不得进入拟合截止时的训练样本。

执行 owner 使用 nullable boolean：True/False 是已知允许/禁止，NULL 是 unknown。
缺失交易状态、涨跌停或必要价格不得被肯定化；已确认停牌为禁止，尾部缺少
下一 session 观测为 unknown，不能声称市场禁止交易。显式 IPO/no-limit 事件可以解除涨跌幅限制；缺少事件不是
无此例外的证据，仍需有效的当日市场事实。ST/PT 未知保留 NULL；eligible=False
只表示不具备入选证据，不能解释为已知非 ST/PT。组合对应方向的订单遇到未知执行事实不成交并明确记录原因，必要持仓估值缺失仍 BLOCKED。

### 5.9 正式因子库

#### 5.9.1 价量因子（PV25与扩展库）

正式 PV25：

```text
reversal_1d
reversal_5d
reversal_20d
momentum_60_20
momentum_120_20
momentum_252_20
volatility_20d
volatility_60d
downside_volatility_20d
downside_volatility_60d
max_return_20d
skewness_60d
drawdown_60d
amplitude_20d
log_avg_amount_20d
turnover_20d
turnover_60d
turnover_volatility_20d
amihud_20d
amihud_60d
amount_shock_20d
return_amount_corr_20d
trend_efficiency_60d
signed_amount_flow_20d
log_float_market_cap
```

CORE 价量因子：

```text
momentum_252_20
reversal_5d
volatility_20d
amihud_20d
turnover_20d
log_float_market_cap
```

正式滚动统计：

完整的是名义交易会话跨度；跨度不足时为 NaN。适用统计量的 fresh 门槛与参数见
[消费独立 PANEL 的计算时钟与 fresh 参数](#消费独立-panel-的计算时钟与-fresh-参数)。
端点、路径及极值仍遵守各自完整依赖，不使用备用阈值。
成交额流向与收益成交额集中度使用批量浮点求和和比值；分母为零保持未定义。
各因子按其有效样本、路径及量纲要求计算，独立结果使用相应数值容差核对。

选定定义：

```text
reversal_1/5/20 = -(P_t / P_{t-k} - 1)
momentum_60_20  = P_{t-20} / P_{t-60}  - 1
momentum_120_20 = P_{t-20} / P_{t-120} - 1
momentum_252_20 = P_{t-20} / P_{t-252} - 1
```

波动率使用 `Dretwd` 的样本标准差 `ddof=1`。

下行波动率：

`sqrt(mean(min(r, 0)^2))`

`skewness_60d` 使用 bias=False；常值窗口的标准化三阶矩未定义，保持 NaN，
不采用库函数返回的伪零值；统计采用完整窗口，
常值收益的样本波动率为合法的零。

`drawdown_60d` 使用复权收盘价从峰值到谷值的最小回撤。

`amplitude_20d`：

`mean((high-low)/close)`

每日换手率：

`(Dnshrtrd + Ahshrtrd_D) / ((Dsmvosd * 1000) / Clsprc)`

`amount_shock_20d` 基线排除当日。

不使用 epsilon 挽救非法分母。

##### 可选价量因子

PV25 与六项 CORE 由普通研究配置保存，不是代码中的隐式默认选择。
库通过同一 `price_volume_factor_definitions(..., factor_columns=...)` 选择以下独有16项；
没有新增实验输出组，也没有周期笛卡尔展开：

| 因子 | 定义 |
|---|---|
| momentum_20d / momentum_60d | P_t/P_{t-k}-1，不跳过近期区间；只要求真实端点 |
| upside_volatility_20d | sqrt(mean(max(r,0)^2))，完整20日 |
| min_return_20d | 完整20日收益最小值 |
| kurtosis_60d | 完整60日无偏超额峰度；常数窗保持原NULL |
| avg_amount_20d | 完整20日成交额均值 |
| amount_change_20d / volume_change_20d / turnover_change_20d | 当前值/20日前端点-1 |
| amount_volatility_20d | 完整20日成交额样本标准差ddof=1 |
| volume_shock_20d | 当前成交量/前20日成交量均值-1，排除当日 |
| return_volume_corr_20d / return_turnover_corr_20d | 名义20日窗内有效配对的Pearson相关，默认至少15对；常数或有效配对不足为NULL |
| return_amount_imbalance_20d | sum(r*amount)/sum(abs(r)*amount)，完整20日 |
| zero_amount_ratio_20d | 成交额非正值占比：mean(amount<=0)，完整20日；NULL传播 |
| log_total_market_cap | 正总市值自然对数，否则NULL |

换手率及其所有衍生量统一 `volume/(float_market_cap/close)`，不读取外部另一口径turnover。
换手率按同一浮点公式批量计算；缺失、非正值及停牌按各自事实处理。
amount_shock与volume_shock都排除当日；amplitude保持(high-low)/close。
均值、shock 和对数均值复用浮点滚动归约；按各自有效样本门槛和名义交易日窗口计算，
数值比较使用与收益、数量或金额量纲相符的容差。有符号消去满足
`abs(mean) <= window × float64 epsilon × mean(abs(source))` 时按数值零处理，相关比值沿用零分母域；
判据使用同窗绝对量级，同号小正值保留。缺失、负数及最终不可表示结果沿用各因子的合同。
零或无效分母保持NULL，其他数值符号按各因子数学定义保留。
CSMAR 摄取的 finite 数值校验不保证 amount 非负，因此非正值占比按 <=0 定义。

价量定义由同一份 definitions 及数学核提供，默认/正式/定制选择的同一项使用相同身份。
reversal_5d 预热为 5，momentum_252_20 预热为 252；只计算请求因子与核心输出窗口。
同一名称只对应一份数学定义，不以别名维护平行引擎。

##### 扩展价量选择

同一个价量 owner 还提供 share_price、avg_amount_126d、amount_cv_126d、
turnover_cv_126d、amihud_126d、skewness_20d、top5_return_mean_20d、
turnover_250d、abnormal_turnover_20_250，共 50 项价量因子。
CV 使用完整窗口样本标准差除以正均值；top5 为完整 20 日中最高五个收益的均值；
异常换手为 20 日均值除以 250 日均值。缺失窗口或非法分母保持 NULL。

beta_252d 与 residual_volatility_60d 复用 compute/market_model 的唯一风险数学核，
不复制风险引擎。pv_expanded 示例选择 50 项价量与这两项暴露；
使用 evaluation、季度模型/OOS、三个简单组合和统计/报告链。
该选择不要求 Choice 财务或增强组合。缺 PIT、分钟或 quote 事实的其他候选保留 BLOCKED，
不会以代理字段冒充已具备数据。

##### 路径机制因子

三个路径变量单独注册，不改变 CORE6、PV25、PV52 原选择。使用完整20个日收益：
`positive_return_fraction_20d = mean(r > 0)`；
`largest_daily_contribution_20d = max(abs(r)) / sum(abs(r))`；
`path_roughness_20d = 1 - abs(sum(r)) / sum(abs(r))`。
缺失窗口不输出；绝对收益和为零时后两项保持 NULL。

#### 5.9.2 基本面 19 因子

正式 Fundamental19：

```text
earnings_yield
book_to_market
sales_to_price
cashflow_to_price
roe
roa
gross_profitability
gross_margin
cfo_to_assets
accruals
asset_growth_yoy
revenue_growth_ttm_yoy
earnings_change_to_assets
parent_equity_growth_yoy
capex_to_assets
liabilities_to_assets
debt_to_assets
cash_to_assets
net_debt_to_assets
```

CORE 基本面因子：

```text
earnings_yield
book_to_market
roe
accruals
asset_growth_yoy
net_debt_to_assets
```

价值分母：信号日 `Dsmvtl3 total_market_cap`。

比率基础规则：

```text
分子为有限数
分母为有限数
denominator > 0
否则 NaN
```

不加 epsilon，也不裁剪。

流量/存量比率使用期初/期末存量平均值；存量/存量使用 PIT 期末存量。缺少上年存量 -> NaN，不退回期末值。

毛利润：

`TTM operating_revenue - TTM operating_cost`

毛利资产比：

`gross profit / average assets`

毛利率：

`gross profit / TTM revenue, revenue > 0`

增长：

```text
asset_growth_yoy
revenue_growth_ttm_yoy
parent_equity_growth_yoy
```

上述为同比比率，要求上期分母为正。

```text
earnings_change_to_assets
= (TTM parent NI_t - prior-year TTM parent NI) / AvgAssets  # 归母净利润变化除以平均资产
```

财务健康：

```text
liabilities_to_assets = total liabilities / total assets

debt = short-term borrowings
     + current portion of non-current liabilities  # 一年内到期的非流动负债
     + long-term borrowings
     + bonds payable

debt_to_assets = debt / total assets
cash_to_assets = cash / total assets
net_debt_to_assets = (debt - cash) / total assets
```

净债务可以为负，不裁剪。

质量：

```text
ROE = TTM parent NI / average parent equity  # 归母权益收益率
ROA = TTM consolidated NI / average total assets  # 合并总资产收益率
CFO/assets = TTM CFO / average total assets  # 经营现金流资产比
accruals = (TTM consolidated NI - TTM CFO) / average total assets  # 应计项目资产比
```

资本开支：

`TTM capex_cash_paid / average assets`

来源资本开支负值为非法 -> NaN；缺失不等于零。

#### 5.9.3 预期 4 因子

正式预期模块：

```text
forward_earnings_yield_fy1
forward_earnings_yield_fy2
forecast_dispersion_fy1
log_analyst_coverage
```

规则：

```text
仅使用 PIT 记录
主方案新鲜度从 declare_date 起算 90 个自然日；第90日包含，第91日排除
180 日稳健性方案
FY1/FY2 分别独立满足 >= 3 家机构，FY1 不达标不阻止 FY2 单独可用
FY1 的 forward EP、dispersion、log_analyst_coverage 仍仅按 FY1 的资格和计数定义
available_at 仅表示声明日之后首个交易日的可用门槛，不作为新鲜度起点
先筛 available_at <= 信号日 且 declare_date >= 信号日-新鲜度
同一 ReportID 先取最新可见版本；归属修改只替换该报告
每家机构按 declare_date 选最新声明日的报告
同机构/同日多个分析师 -> 机构内中位数
跨机构中位数 / MAD / 计数
FY1 = earliest not-ended annual forecast_end_date  # 第一个尚未结束的年度预测期
FY2 = FY1 forecast_end_date + 1 calendar year  # 紧邻的下一财年，缺年不跳到更远年度
forward EP = median forecast EPS / unadjusted signal-day close  # 前瞻盈利收益率定义
```

`prepare` 在 `data_id=expectations_factors` 时接受 `parameters.freshness_days`，
接受任意正整数，未显式指定时沿用该数据规格的 90 日默认值；90/180 是随仓研究选择。
解析值保存在 `ProcessedDataSpec.parameters_json.freshness_days`，约束实际来源回看和公式；
统计中的 `expectations` 研究说明应与 prepare 的实际参数一致；不另设重复的 freshness 配置字段。
180 日方案单独创建不可变研究实例，标为 `expectations.role=robustness`；保留其模型、预测与组合链，
但不将其候选或结果混入 90 日正式选型及 FDR、SPA、StepM、PBO 推断域。
180 日运行内部保留训练、验证和原排序作为稳健性诊断；基础描述、HAC、基准及归因仍按原定义计算。
正式联合检验与 PBO 标注 `expectations_robustness_excluded`，不将全部稳健性结果改成未定义。
统计合同按实际评估因子和模型输入派生此角色；只由模型消费分析师因子时仍执行相同排除。
该稳健性实例的冻结 test G/Q 保留单项 HAC 与 bootstrap；联合 FDR 因稳健性声明标为
`NOT_APPLICABLE`，不是由 test 名称本身触发。
机构日中位数、跨机构中位数及 EPS/close 比值采用普通浮点归约，落盘为 Float64。
价格缺失或非正只使 EP 与价格缩放离散度未知；原始 EPS 离散度、机构覆盖、年龄和更新比例仍按有效预测计算，
单独选择这些因子不读取 close。不可表示的派生比值只标记该值 NULL，不中断其他因子或后续日期。

预期修订、超预期、PEAD 和业绩预告属于扩展研究方向，不属于本协议的预期4模块。

##### 预期机制增量与研究准备

预期机制沿用同一 FY1、机构日最新事件、新鲜度及至少3机构门槛。
`forecast_dispersion_price_scaled_fy1` 是机构 EPS 的 MAD 除当日正收盘价；
`median_forecast_age_days_fy1` 是当前机构状态报告年龄中位数；
`newly_updated_institution_fraction_fy1` 是本可用日更新机构占当前机构的比例。
同 EPS 的新报告仍更新事件状态。原4项预期因子不改名、不替换。

主规格为90天，一次180天对照；完整模型、非金融共同样本与各自覆盖须同时报告。
固定财年修订、滚动 FY1 跳转、ON/ID分解及报告事件研究先核对原始语义和单位；
能力或配置绑定通过不代表这些机制已完成真实研究。缺少版本仅阻塞对应版本消费者。

### 5.10 特征范围与变换

正式范围：

```text
CORE
FULL
```

预期场景：

```text
CORE_Exp = CORE Fusion features + 4 expectations  # 核心融合特征加预期模块
FULL_Exp = FULL Fusion features + 4 expectations  # 完整融合特征加预期模块
```

不使用按绩效动态删除因子的选择器。

已处理事实存储经济原始因子、经济标签与市场事实；只有确定 `eligible` 后，才创建场景专属规范特征。

`assemble.feature_groups` 整批选择 `raw`（原始因子）、`processed`（当前配方处理后）、
`pre_neutral`（同样裁剪及中心/缩放，但跳过行业/规模中性化）；可同时选择三组。
有配方时默认 `["raw", "processed"]`，无配方时默认 `["raw"]`。
每组覆盖本次显式选择的全部 `factor_columns`，不提供逐因子处理开关。
`processed` 列名为 `feature_<factor>`；`pre_neutral` 为 `feature_pre_neutral_<factor>`，
各自有对应 `feature_missing_…` 与 `feature_observed_…` 伴随列。
只输出处理组时，原始因子只作为计算依赖读取，不混入输出因子列；上游原始产物不变。
组共用裁剪结果，各列/组仍独立决定有效域；只选择 `pre_neutral` 不要求行业/规模上下文。
原生配方保存 `output_groups`，操作参数保存 `feature_groups` 与实际 `input_columns`；
下游通过显式列选择使用其中一组或多组。列名冲突明确拒绝。
显式 `feature_groups` 也约束已物化 FEATURE 列的复用，按其原生配方来源识别版本；
无版本来源的 FEATURE 列归为 `processed`。例如 `["raw"]` 不会自动夹带已处理列。
显式 `feature_columns` 与组选择冲突时拒绝；省略 `feature_groups` 时按默认规则复用。
省略配方时，已有完整输出组的因子默认直接复用，只为尚无该组输出的新增因子计算；
显式指定配方/变换参数而产生同名输出时仍拒绝。`feature_recipe=null` 可明确关闭新特征生成。

`assemble.feature_recipe=null` 保留原始因子；`label_columns=[]` 可生成仅供预测的无标签数据集。
选用配方后，`feature_transform` 可分别设置 `clip`、`industry_neutralize`、`size_neutralize`、
`center`、`scale`、`impute_missing`。`clip` 支持 `none`、`mad`、`quantile_1_99`；
上述其余字段为布尔值。`size_column` 默认 `log_total_market_cap`，也可显式选择
`log_float_market_cap`；仅启用规模中性化且处理其他因子时依赖所选规模列。
配置及原生特征元数据保存实际 `size_column`。
关闭某步骤就不要求该步骤专用上下文，不自动补算未选因子。
固定顺序为裁剪、中性化、中心化、总体标准差缩放；所选规模因子不回归掉自己。
中心化不改变单位，标准化输出无量纲。仅缩放而不中心化时，零方差结果未定义；
中心化并缩放的常数截面按配方输出零。

默认规范特征流水线：

```text
符合 eligible 的横截面
-> MAD 缩尾
-> 中性化
-> 总体 zscore
```

#### 5.10.1 缩尾

`median +/- 3 * 1.4826 * MAD`

MAD=0 -> 不裁剪并记录诊断。

研究数据 manifest 的 `producer_diagnostics.canonical_features.mad_zero_by_input`
按输入列累计 MAD=0 的 eligible 日期截面数，包含辅助规模协变量；无有效观测的截面不计入。

1/99 分位仅用于稳健性。

#### 5.10.2 中性化

所有非 Size 的 alpha 因子：

```text
行业虚拟变量
+ 去均值的 log_total_market_cap（默认，可显式选流通口径）
无截距
等权 OLS 残差
```

Size 风格：

```text
仅行业虚拟变量
无截距
```

检查边界：

```text
ordinary cross-section n >= 100  # 普通横截面样本门槛
neutralization n >= max(100, 5p)  # 中性化样本门槛
要求满列秩
```

不使用 ridge/pinv 挽救失败。

行业截距与共同规模斜率使用组内中心化的等权 OLS 闭式计算，批量完成归约与投影。
残差、秩和零方差按输入尺度及运算误差判断；样本域、缺失状态与可用时点保持一致。
设有效行业数为 K：Size 自身的 p=K；非 Size 的 p=K+1，要求组内 Size 提供可识别的共同斜率。
同一组内统计由中性化 owner 复用。必要Size即使不输出自身feature，
也按同一eligible截面执行MAD缩尾再作为协变量；单因子与完整配置的同一因子输出必须一致。
辅助Size维持FACTOR角色，仅补数据依赖，不自动加入评价信号或生成无消费者feature。

#### 5.10.3 Z-score 标准化

规范 zscore 使用总体标准差 `ddof=0`。

特征零方差且 n>=2 时：

```text
observed canonical values = 0  # 零方差时已观测特征的规范值
记录诊断
```

`producer_diagnostics.canonical_features.zero_variance_by_factor` 按输出组后缀累计处理后
至少两条有效观测且总体方差为零的日期截面数；诊断在完整日期截面上计算，跨存储分块累计，
与规范特征一同由原生 writer 发布，并由 reader 读取。

目标有效值至少两个的常数截面输出零；原 NULL 保持 NULL，单有效值与全缺截面无效。

zscore 之后不再裁剪。

#### 5.10.4 缺失语义

原始经济因子的缺失继续保留 NaN。

模型矩阵中的普通输入缺失：

```text
feature value = 0  # 模型矩阵中缺失特征的填充值
feature_missing = 1
observed = 0
```

已观测有限值：

```text
feature_missing = 0
observed = 1
```

结构/上下文前提失败时，该特征/行必须明确不可用；中性化失败不得通过 `.fillna(0)` 进入模型。
三列 (feature, feature_missing, feature_observed) 的编码：已观察有效为(z,0,1)，
结构不可用或 ineligible 为(NULL,0,0)。feature_missing=1 表示变换可定义、必需上下文有效，
但原始因子普通缺失。此时，
`impute_missing=true` 输出(0,1,0)，关闭后保留(NULL,1,0)。这里的零是技术填充值，
不承诺在关闭中心化或中性化后等于截面均值；结构无效和标签缺失始终不插补。
训练、验证、OOS及预测消费者的必需PK域，共用该模型实际输入的唯一有效性规则；即使只选missing/observed
标记，也必须检查完整声明 schema 中对应的来源 feature，不能因为标记为有限数而绕过。
该关系只适用于已声明的 canonical 输出，不凭普通外部列名前缀建立依赖。raw排名模型不消费的canonical
结构状态不改变其缺失中性规则。监督模型按实际缺失策略处理普通缺值：原生 NaN 或训练期中位数插补。样本须满足每个实际消费 canonical 列族的 missing 或 observed 为1，
且至少一个实际原始输入已观察；raw、pre-neutral 与 processed 可混用，未消费列族不影响资格。原始输入和转换规格保留可复核原因。

统计和组合直接消费 PRED 的真实键与分数，不重新打开模型目录或历史 X，不重复模型输入资格判断。
仍对齐当前上下文键、研究资格及因果时点；未提供预测的资产信号为 NULL，持仓和 benchmark 会计全集不随之裁剪。
既有持仓按原交易限制退出，许可 unknown 的对应方向订单不成交并记录原因；
增强组合对可到达基准的 incomplete 资产约束主动权重为零。不能交易的真实持仓仍占预算及风险，
不得用内部零系数掩盖其风险或发布为预测，详见 §5.22.3。

单因子统计始终使用原始已观测经济值，绝不使用模型填充零。

### 5.11 标签与模型目标

A 股经济时间线：

```text
在 t 收盘生成信号
在 t+1 市场交易日复权开盘入场
在 t+1+h 市场交易日复权开盘离场
```

正式期限：

`1, 5, 10, 20, 60, 120 个市场交易日`

主模型经济标签：

`forward_return_20d`

辅助期限：

`5d / 10d / 60d`

诊断期限：

`1d / 120d`

标签两端各要求 `filling_flag=0`、`is_trading=True` 且 `adj_open` 有限、为正；
端点停牌、填充或必要事实未知时标签缺失，即使存在前值填充价格也不能视作真实端点。
`can_enter`、`can_exit` 与涨跌停限制不影响价格标签资格。

卖出限制不延长固定标签期限；标签仍定义固定期限经济收益，而不是模拟清算结果。

每个标签持久化各自的：

`forward_return_*d_available_at`

signal time 为 t 收盘；entry/exit event time 分别为 t+1/t+1+h 开盘。
对有限正价格端点，收益按`(exit-entry)/entry`局部稳定计算，避免先求比值再减1合并接近零的可分辨收益；
两端满足上述观测资格时按此计算；最终不可表示值仍缺失。该计算由标签生产者唯一实现，不改标签期限或经济开盘端点。
日线一字板判定使用入场日 OHLC，属于执行事实，不参与标签资格或成熟判断；
买卖许可分别只依赖对应方向的必要事实。已确认交易、有效开盘及无限价或非对应方向限价状态时，
无关 high/low 缺失不改变许可；有效 OHLC 已有不相等证据也可排除一字板。停牌禁止，必要交易/开盘/状态未知或一字板仍无法判定时保持未知。
经济价格事件仍为入场/离场日 Asia/Shanghai 09:30 开盘；当前 wd 复权价格口径的
调整来源依赖当日收盘，因此标签不能在离场开盘时被视作已可用。
行的日频日期使用 session 标签；`available_at` 持久化固定离场交易日的收盘时刻：
Asia/Shanghai 15:00 规范到 UTC（现代 A 股为 07:00），不使用该日零点或 09:30。
这是日频研究的 EOD 可用性约定，不声称已核实供应商盘中实际发布时点。
普通 forward-return 与 ON/ID 分别记录经济事件及所依赖价格来源的可用时钟。拟合的 `fit_cutoff` 及
`evaluate`/`score` 的 `cutoff` 按精确时点解释，日期值表示 UTC 零点，不自动扩到日末；
离场日 UTC 01:30 或 06:59 的拟合和评价不能使用该标签，07:00 才达到成熟门槛。
`statistics-factors` 数据路径和 `statistics-model` 的 `label_cutoff` 沿用窗口终点规则：
日期值包含当日全天，精确时间戳按给定时点；省略时使用所选窗口终点。不能混用两类截止字段。
不把未来端点价格或执行事实当成信号日已知信息。

#### 模型目标

`model_target` 只在场景研究数据集组装中、确定 `eligible` 之后创建：

```text
有效且满足 eligible 的 forward_return_20d
-> 横截面总体 zscore
```

一行缺失只影响该行，不得污染整个横截面。

默认 `target_transform="date_zscore"`，总体标准差分母为有效样本数；
至少两个有效值的常数截面为零，原 NULL 保留；单有效或全缺截面为 NULL。
`target_clip` 默认 `none`，可独立选择 `mad` 或 `quantile_1_99`，复用特征裁剪数学。
可选 `raw` 输出原始收益（允许单有效值）；`percentile` 使用平均并列名次除以有效数，
至少两个有效值，例如两个相同值均为 0.75；`industry_size_residual` 为等权行业截距与
`log_total_market_cap` 回归残差，保留收益单位，不再 zscore。
残差目标只使用信号日同一行的历史行业/规模事实；行业标准与代码共同标识行业，
缺少事实的行不参与裁剪及回归，不用未来行业/规模回填。沿用现有中性化样本门槛
`n >= max(100, 5p)` 及可识别性检查。其余模式不要求这些专用输入。
原始 `forward_return_20d` 始终保留且不缩尾/改写，所有标签均不插补。
目标模式、裁剪参数和实际输入列写入原生元数据；raw/残差目标单位为 `decimal_return`，
zscore/percentile 为 `dimensionless`。

每份数据集按所选模式只生成一列 `model_target`。其 `model_target_available_at` 是该截面真正参与规范化的
合格非缺失标签成熟时点的最大值；没有有效目标的行保持 NaT。训练与验证消费该独立成熟列。
无裁剪的 `raw` 目标不参与截面运算，直接沿用自身标签时钟；开启裁剪时仍采用参与截面的最晚时钟。
逐资产原始 `forward_return_20d_available_at` 保持自身价格来源可用时点，不因其他市场较晚成熟而改写。
标签端点及 execution mask 的分批 lookahead 共用正式 PANEL 的完整资产会话域：包含开市未知报价
与停牌占位，排除已知闭市，遵守历史市场迁移，不按稀疏原始报价压缩期限。

### 5.12 研究数据集语义边界

每个正式场景数据集包含：

```text
market_id
asset_id
timestamp
eligible
原始经济因子
规范特征
feature_missing 指示变量
经济前瞻收益
标签 available_at 字段
model_target
研究必需的合法 PIT 上下文
```

不包含：

```text
can_enter / can_exit 未来执行列
已拟合缩放器
已拟合模型
预测结果
组合权重
执行结果
交易成本结果
```

数据集标识依赖数据、特征和标签语义，不依赖 Ridge lambda、TE 限制或其他下游模型/组合设置。

开发与留出使用同一数据集定义和唯一生产者，按实际请求窗口明确给出各自输入与新输出。
开发标签生产者只访问开发可见末端；开发输入截止于 2023 年末。
截至最终开发 cutoff 仍未成熟、需要 2024–2025 独立 test 事实的尾部标签保持不可用，
不得为保存或重新评分而提前读取这些事实，也不得将其写入开发产物。
历史季度末未成熟、但开发冻结截止前已成熟的标签可以保留；每次选择仍按自己的截止时点重新过滤。
历史初始状态仍沿生命周期/asof owner 读取。

### 5.13 三角色、两组选择与最终留出

train、validation、test 描述本次操作中窗口的用途，不是样本的永久身份，不另设研究访问阶段。
selection 和 refit 是执行阶段；较早预测样本可在后续标签成熟后进入明确声明的拟合窗口。
以下年份与两年冻结要求属于本研究协议，不由窗口名称自动施加于所有公开操作。

- 2014–2017 提供初始训练历史。
- 2018 是验证预测预热年：全部声明候选按季度使用此前合法历史训练并预测，
  不在预热年发布冠军、组合或 Alpha 校准。
- 2019–2023 是正式 validation。每个 candidate × fold 只训练一次，
  缩放器、排名方向/权重和监督系数只由该 fold 的 train 学习。
  两组复用同一份原始 train-only 验证预测。
- 2024–2025 是最终 test，默认冻结模型及监督校准；可显式选择冻结研究规则后的季度重训，
  每次仅使用当时已成熟的训练标签，最终评价结果不反馈到规则或候选选择。

全局组 G 在开发结束时，按全部已成熟的 2019–2023 每日验证 RankIC 选择。
2018 预热行不进入 G 得分；不平均季度均值来替代每日均值。
G 的开发曲线使用期末选定配置对应的各季度原始预测，属于回顾性验证曲线。
通过 `backtest.selection_mode="retrospective"` 与显式 `selection` 产物选择；
每份原始预测的实际拟合截止仍严格早于其预测窗口，选择产物须匹配候选及科学参数。

季度组 Q 在每个正式季度开始前，只用过去 12 个月且当时已成熟的验证预测选择。
季度内固定所选配置，消费本季度该配置的原始预测。开发末期还须按最终截止时点
重新进行一次过去 12 个月选择，不直接沿用上一季度的选择。
Q 的开发曲线是当时可实现的因果路径。

训练、选择和重新校准分别执行自己的 label_available_at <= fitting_cutoff 成熟判断。
不维护第二份机械隔离间隔。两组最终选定配置均在同一合法开发历史上进行真实 final refit，
显式保存 fit_windows、fit_cutoff、组别和来源选择集合；
refit 不伪造重叠 train/validation，也不建立假的第四角色。
相同配置使用相同最终拟合规则和种子。全部获批家族与配置继续保留。

部分开发请求可以产生共享候选和已有季度选择。G 经济预测需要完整开发终局选择；
全局选择缺完整验证历史时明确拒绝；独立合法季度拟合与Q可以继续，不能虚构部分G。

最终 test 绑定明确的开发模型及各组 Alpha 校准来源。
默认方案的算法、配置、特征、系数、排名方向/权重和缩放器固定两年；
显式季度重训方案固定算法、特征和更新规则，拟合事实随每次模型保存；
经济持仓及预先声明的市场、执行、风险估计规则仍随合法可用事实演化。
留出不得用于模型/特征选择、成本或风险调参、稳健性选择。
开发与独立 test 输入分开绑定；改变独立 test 数据不能改变已冻结开发选择、模型或校准。

可选 prequential B 属于另行声明的研究设计，本协议不隐式执行。
留出后的数值修正若改变数据、预测、组合或统计，必须标为留出后修正证据。

### 5.14 正式模型层级与预处理

模型能力与一次研究选择分开定义。支持的算法由同一模型合同、拟合 owner 和持久化预测路径消费：

| 类别 | 可配置算法 |
|---|---|
| 排名集成 | 等权排名集成、IC 加权排名集成 |
| 线性与稳健回归 | OLS、Huber、Ridge、Lasso |
| 树模型 | 普通决策树、随机森林、sklearn GradientBoostingRegressor（GBRT）、XGBoost、LightGBM |

科研选择参考中的 `models`、输入列及有序 `parameter_candidates` 声明实际候选，
不由算法能力表隐式补齐，也不自动展开拟合。完整 `parameter_candidates` 与基础
`hyperparameters` 不得同时声明；不能保留一份被静默忽略的参数真值。
agent 为每个候选构造独立 `fit` 配置：`model.model_id` 标识候选，`family_id` 标识家族，
`candidate_parameters` 声明需按真实训练样本数解析的参数；使用该字段时
`model.hyperparameters` 必须为空。单次拟合不接收候选列表或自动参数搜索。
实际候选参数、输入列、`target_column`、种子、权重及拟合窗口均随模型与 trial 保存。
随仓配置的模型数量见 §5.7，六份配置均选择11种模型。普通决策树使用 sklearn 拟合路径，
这些研究参考配置明确选择深度 3、叶子最少样本比例 0.001、种子 42，不根据研究收益追加搜索；它们不替代算法 API 的默认值。
这一区分不代表任何模型已经通过真实研究验收。

单因子仅作诊断，不参与优胜模型选择。

监督模型共享 `ModelMatrixScaler` 管理训练输入。`model.training_scaling` 可显式开关：线性模型默认开启，树模型默认关闭；缺失策略与缩放独立。开启时仅在训练集拟合，作用于：

```text
规范连续特征
缺失指示变量
```

目标不缩放。

开启缩放时，训练期零方差列的冻结 scale=1，训练样本变换为 0，不删除该列。
OOS 始终使用训练期 mean/scale；新值按 (x-mean)/scale 变换，不能永久置零。
fit、持久化、reload、transform 和 predict 共用 ModelMatrixScaler。
训练加权均值以 `weighted_mean` 保存，总体尺度以 `weighted_population_scale` 保存。
模型对象与元数据使用相同的浮点值；加载后按冻结的 `(x - mean) / scale` 变换。
加权均值与方差采用批量中心化计算，独立计算结果按数值量纲和算法误差核对。

关闭缩放时保留原单位；需要的中位数插补和缺失指示器仍按训练事实处理。原生 NaN 模型不因此插补或删除全空列。排名集成和单因子不使用该缩放器。

#### 5.14.1 等时点样本权重

`equal_timestamp` 按完整timestamp分组，每个训练时点贡献相同总权重；最终权重归一化为：

`sum(sample_weight) = N_train`

不使用市值样本权重。

主方案不使用时间衰减；启用衰减的研究默认半衰期为 252 完整市场会话。

预注册稳健性通过 ModelSpec/config 的 `training_weighting={"method":"equal_timestamp","half_life_periods":756}` 执行；
按有序训练时点的年龄乘 `0.5 ** (age / half_life_periods)` 后重新归一化为 `sum=N_train`。
这里的年龄按显式完整市场会话轴计算，缺失训练观察不压缩时间，也不使用自然日。实际配置写入模型与 trial，
supervised 缩放/拟合与 rank ensemble 方向/IC 权重共用。
该 756-period 预注册对照由调用者显式声明为 robustness；其研究集合与主方案分开，
不得与其他 half-life 模型混入同一选型实例。
其内部训练、验证排序与单项HAC/bootstrap仍保留，正式FDR、SPA、StepM、MCS与PBO排除该实例，
原因使用`training_weighting_robustness_excluded`；该稳健性实例的冻结 test G/Q 单项比较保留，
联合 FDR 因稳健性声明为 `NOT_APPLICABLE`，不按 test 名称统一排除。
A 股与通用模型都接受 `half_life_periods=null` 或任意有限正数；756 是随仓研究选择，
不构成库能力上限，也不按参数值自动授予 primary/robustness 身份。
统计中的稳健性用途须显式声明；上述 756-period 研究设计的推断范围由该声明决定。

缩放器的加权均值/标准差使用最终训练样本权重。

ResearchDataset 选用特征处理时仍只接受一个具名配方；A 股默认组合为 `a_share_canonical`，
独立稳健性组合为 `a_share_quantile_1_99`；实际开关、`recipe_id` 与 `source_columns` 保存在 metadata，
可选步骤与完整截面规则见 [§5.10 特征范围与变换](#510-特征范围与变换)。
稳健性配方保留单项诊断，但排除正式联合推断，原因 `feature_recipe_robustness_excluded`。
Crypto 不接入该横截面流程。

EWMA先扣除当前有效观测的共同最小年龄，再用浮点指数权重归一化；缺失最新session不抹掉有效历史。
风险计算保留完整名义会话跨度、有效观察门槛与各项缺失原因。
FactorRiskConfig唯一拥有风险窗口、半衰期、最小观测、diagonal shrinkage、annualization与weighting；
OptimizerConfig唯一拥有active-name/TE/turnover/industry/style/SLSQP容差和迭代上限。
独立 `backtest` 的 `risk_config`、`optimizer_config` 与 `assumptions` 分别解析为上述对象；
ADV participation 由 ExecutionAssumptions 唯一拥有，不另设 CLI 覆盖值。
协方差保持 PSD 数值容差；成功优化结果不含负方差或非有限跟踪误差。
优化约束、解后检查与诊断共用风险计算。负二次型只有落在按实际乘积规模和维数计算的
浮点舍入误差界内才归零，超出误差界或非有限结果明确失败；不投影协方差或截断实质负风险。

#### 5.14.2 Ridge

协议正则强度：

`lambda in {0.01, 0.1, 1, 10}`

解析后的 sklearn 原生 alpha：

`alpha = N_train * lambda`

#### 5.14.3 Lasso

```text
alpha in {0.001, 0.003, 0.01, 0.03}
fit_intercept = True
selection = cyclic
max_iter = 20000
tol = 1e-5
positive = False
```

全零拟合合法，但没有预测变化；不收敛表示试验失败。

#### 5.14.4 Huber

```text
epsilon = 1.35
lambda in {0.01, 0.1, 1, 10}
alpha = N_train * lambda
fit_intercept = True
max_iter = 1000
tol = 1e-5
warm_start = False
```

#### 5.14.5 随机森林

试验配置：

```text
(max_depth, min_weight_fraction_leaf)
(6,  .001)
(10, .0003)
(14, .0001)
```

固定参数：

```text
n_estimators = 300
max_features = .5
bootstrap = True
max_samples = .5
实际显式传入的随机种子（省略时为 42，见 5.29）
```

运行时线程数是操作元数据，不是研究协议标识。

#### 5.14.6 XGBoost

试验配置：

```text
(max_depth, min_child_weight_fraction)
(3, .001)
(5, .0003)
(7, .0001)
```

解析参数：

`min_child_weight = N_train * fraction`

固定参数：

```text
n_estimators = 400
learning_rate = .05
subsample = .8
colsample_bytree = .8
objective = reg:squarederror
tree_method = hist
max_bin = 256
reg_alpha = 0
reg_lambda = 0
gamma = 0
device = cpu
不提前停止
```

#### 5.14.7 GBRT 与 LightGBM

GBRT 与 LightGBM 复用上述训练样本权重、独立缩放开关、确定性种子、
候选评分和持久化预测合同。原生参数及候选组合由研究 JSON 显式声明，
不因算法可用而增加未声明试验，也不以模型数量替代实际 OOS 证据。

### 5.15 模型选择

两级选择：

1. 按 §5.13 的 G 全正式验证窗或 Q 因果过去 12 个月窗，家族内以每日验证 RankIC 均值选择配置。
2. 在研究 JSON 选定的家族优胜配置中，以同一组、同一截止和同一共同域选择优胜家族。

科研选择参考的 `models[].model_id` 表示家族；直接 `fit` 使用 `family_id` 和
`model.model_id` 分开记录家族与具体候选。精确小写 `champion` 保留给组合预测选择器，
不得作为 A 股模型或家族名；拟合及评分消费同一 A 股模型合同。
A 股 `model.sample_filter_mode` 默认为 `eligible`，且仅支持该值；开发验证与最终留出使用同一资格合同。
其他合法名称按普通标识符解释。

精确并列时，按研究 JSON 中模型的声明顺序选择；候选清单只在配置中维护。
家族内并列时按显式 `parameter_candidates` 顺序选择。随仓配置保留以下保守声明顺序：

```text
Ridge / Huber -> 较大的 lambda
Lasso -> 较大的 alpha
RF / XGB -> 更浅 / 更保守的配置
```

完整有序候选集合必须保留。失败或放弃的候选可用真实终态 trial 显式登记，
记录失败原因、零验证行和未定义指标，不冒充已完成预测，也不静默删除。
选择只从已完成候选的已定义指标中进行；某家族没有可用成绩则无优胜配置，全部家族无成绩时不产生胜者。
同一季度所有参加比较的家族/配置在唯一共同有效(timestamp,asset_id)域上重算验证成绩。
先在各自完整合法截面预测，再投影共同域评分，不能先裁剪而改变rank；排名缺失中性规则不变。
只有raw因子的rank缺失保持中性；rank模型若消费canonical值或缺失/合法性标志，仍须满足来源特征的结构有效性。
agent 显式按合法训练窗调用 `fit`，再按 `prediction_window` 调用 `predict`；
部分季度不扩大到请求末端之外。每次 `score` 提供有序 `candidate_ids`、逐候选的
各 `validation_windows` 对应的预测路径、`labels`、`scoring_window`、`cutoff`、
`selection_group`、`validation_axis` 与新 `output`；不登记未来测试窗口。
`label_unit` 可从输入真实单位声明取得，未声明时须显式给出，二者不能矛盾。
G 评分窗必须覆盖所声明的完整正式验证历史，Q 选择截止必须早于实际消费季度。
模型使用各自合法输入域；PRED 只保存预测及真实键，不重复保存 label、成熟列或 score_eligible。
评分仅读取显式标签输入和 PRED，可从较宽 PRED 选择时间子窗，同一 PRED 可用于不同的离散窗。
按本次标签的 eligible、真实成熟时点和所选窗口构造评价域，再以精确主键配对；不重开历史特征或模型文件。
每候选 coverage 报告 current_mature_label_rows、paired_prediction_rows、mature_label_rows_without_prediction
和 prediction_rows_without_label；这些计数只描述当前输入，不证明原训练特征域完整。未来成熟标签不参与得分，
缺预测和缺标签只影响对应行。共同域为空时保存零样本和未定义指标，选择不产生胜者。
评分 JSON 保存候选/模型/PRED 路径、实际拟合窗、逐候选指标及 `common_rows` 等科学事实，
不维护共同域持久哈希或跨操作身份。`select` 另外声明指标、方向、完整并列次序、
`refit_windows` 与 `refit_cutoff`，保存家族优胜与 `selected_model_id`；不会自动重新拟合。
模型加载核实际拟合状态；已完成 PRED 的评分直接消费其保存的真实事实，不要求外部模型或完成 trial。
失败候选仍须有真实终态证据，选择核原评分及真实截止。
中断后由 agent 选用已正式发布且科学范围合适的模型、PRED 和选择，
只补缺失操作并给出新输出位置；不得把 staging、失败 trial 或未发布最终拟合当作成功。
trial 不可变。
每个候选同时输出每日 RankIC 的总体标准差（ddof=0）、ICIR、正值率、有效日期数及共同样本数；
ICIR 为每日均值除以该标准差，不年化；零方差和不可表示结果保留未定义。
已定义 ICIR 要求至少两个有效日期以及正的有限总体标准差；原因与有效日期数一并保存。
每次拟合只记录真实训练和实际成本。G/Q 的候选评分可明确选用同一份已保存验证 PRED；
最终模型还须满足本次冻结选择与真实拟合记录，不能仅凭参数相同替代选择关系。代码不自动搜索或合并 refit。
独立调用确实重新拟合时各记自己的 trial，不伪造第二次 attempted 或共享训练成本。
共同域只约束比较，不裁剪完整预测或经济账本；按真实键配对并披露预测缺行及覆盖，研究报告不能通过省略漏行把收缩样本冒称完整样本。

RankIC 为负的优胜模型仍是有效协议结果；不得因结果差而重新调参。

#### 排名集成

因子保留自然经济定义，不建立事后静态方向表。

等权排名方向从拟合历史的每日 RankIC 均值学习。每个正式输入都必须至少有一个
可定义的训练期 RankIC；单个输入方向不可定义即明确失败并报告该输入，不删除因子、
不补方向、不降低样本门槛。推断时某行排名缺失的中性贡献不等于训练方向可定义。

IC 排名因子权重：

`mean RankIC / sum(abs(mean RankIC))`

排名缺失贡献中性 0；不得对每只股票仅按已观测因子重新归一化。

排名模型的方向/权重与 `label_calibration` 都只在合法 TRAIN/refit 域拟合并保存。
PRED 的 `prediction_score` 仍是原始排名分数；评分的标签单位 MSE 使用已冻结的转换，
不得拿当前验证标签重估。A 股组合使用 `prediction_rank_score`，其 Alpha 校准仍按 §5.18
独立执行；Crypto 经济转换见 [Crypto 标定](crypto.md#数值缩放与排名模型标定)。

方向和权重在每个候选 fold 的 train 内学习，并在对应 validation 季度内冻结。
最终评价默认沿用冻结模型；显式选择 §5.13 的季度重训方案时，每次只用当时合法训练域
重新估计方向、权重及标签标定，随后在对应预测季度内冻结。

### 5.16 预测产物语义

组合消费显式指定的已持久化预测，并核必要模型及选择记录；等权排名基线也经同一 fit/predict 路径，
不得以活动模型对象或原始因子作为捷径。

预测行至少包含：

```text
timestamp
asset_id
prediction_score
prediction_rank_score
```

`model_id`、`family_id`、模型路径、输入路径、`sample_role`、`prediction_window`、
`effective_prediction_end` 与必要模型 metadata 保存在原生预测 metadata 中，不逐行重复。
`prediction_score` 是模型原输出，中心化排名分数为：

`prediction_rank_score = (average_rank - (n+1)/2) / n`

有效预测证券范围内，每日横截面均值为 0。

季度选择/模型产物记录：

```text
全部试验指标
家族优胜配置
优胜模型
特征列
排名方向/权重
缩放器参数
拟合截止时点
N_train
样本权重语义
已解析模型参数
实际显式传入的随机种子（省略时为 42，见 5.29）
```

### 5.17 组合候选配置

科研选择参考中的 `portfolios` 保留研究候选；直接 `backtest` 的 `candidate` 明确指定一项：

| 字段 | 责任 |
|---|---|
| `candidate_id` | 本次候选标识，由调用者为实际 G/Q 结果明确命名 |
| `selection_group` | `global` 或 `quarterly`，与所消费的选择记录一致 |
| `construction` | `quantile_equal_weight`、`quantile_equal_weight_half_step`、`quantile_float_cap` 或 `benchmark_enhanced` |
| `prediction_selector` | 已持久化模型的 `family_id`，或保留选择器 `champion`，不借组合名称推断算法 |
| `top_quantile` | 简单组合的最高分位比例，须满足 `0 < top_quantile <= 1`；增强组合为 `null` |
| `rebalance_schedule` | `daily`、`weekly-last-trading-day` 或 `monthly-last-trading-day` |
| `benchmark_id` | 增强组合的 CSI500/CSI1000 基准；简单组合为 `null` |

`prediction_selector` 精确区分大小写。调用者显式提供已经选定的预测路径；`champion`
消费该明确选择，具名 selector 额外核对所声明的模型家族；回顾模式另读取显式选择产物核对候选。
不隐式映射 `winner` 或其他拼写。模型家族与候选身份分别由配置声明，不从 `candidate_id` 推断选择器。

`quantile_equal_weight` / `quantile_float_cap` 按显式 `top_quantile` 选择最高分位；构建标识只说明加权方法。
增强组合所需 benchmark、风险及数据依赖由同一组合合同派生；配置不能省略必要事实。
只读 profile 不创建组合或账本，参考文件不作为自动展开入口。

`core6.json`、`pv25.json` 及两个融合配置各声明下列五条周频候选；
`pv_expanded.json` 声明前三条简单组合。它们是随仓配置选择，不是候选数的硬约束：

```text
1. 等权排名 -> Top20% 等权
2. 优胜模型 -> Top20% 等权
3. 优胜模型 -> Top20% 流通市值加权
4. 优胜模型 -> CSI500 增强
5. 优胜模型 -> CSI1000 增强
```

每次运行的日程与候选范围由配置冻结；成本敏感性和稳健性变体不自动扩展该试验范围。

示例中的 Top20 表示 `top_quantile=0.2`，即横截面最高20%分位，而不是前20只股票。
并列使用平均排名；边界数量可以略超过所选分位比例。

多空因子研究：

```text
多头 +0.5
空头 -0.5
净敞口 0
总敞口 1
```

多空用于 alpha 研究；没有借券数据时，不声称中国现货股票空头经济完全可投资。

### 5.18 Alpha 校准

对每种再平衡日程分别校准：

```text
最近 252 个实际交易会话
与日程匹配的下一执行开盘收益
prediction_rank_score -> 带截距、等日期权重 OLS
expected_alpha = slope * prediction_rank_score
忽略截距
```

覆盖要求：

```text
>= 80% 计划再平衡日期
and >= 6 valid dates  # 同时满足有效再平衡日期门槛
```

不使用更长窗口作为备用方案。设计矩阵（截距与 prediction_rank_score）必须满列秩；
覆盖不足或秩亏时校准为 UNDEFINED，记录原因，不发布伪零斜率及 SE/t/p/CI/R2。
开发期仍执行仅基准逻辑，但它不表示斜率已确定为零。校准状态随账本发布，
报告和 API 读取账本末期实际状态，不重新拟合。

部分 OOS 增强请求使用同一份持久化优胜模型预测真值，
按同一完整市场日历读取 252 会话校准窗口，并受显式输入范围约束；缩短输入历史不缩小计划覆盖分母。
组合账本只发布请求窗口；预热行不是第二份预测产物。

Alpha 校准使用当时全部满足资格的 A 股，
为同组、同再平衡日程的 CSI500 与 CSI1000 增强候选产生一份共享优胜模型斜率；
不同日程分别校准并冻结，留出执行按组别与日程读取匹配的斜率。
每个评分资产的信号、次日入口及成熟完成端必须绑定同一 `(asset_id, listing_date)`；
这也适用于未被组合持有的合格评分资产。退市重上市不能贡献跨生命周期收益；
同一生命周期内有可靠端点的正常停牌或本市场闭市仍按估值合同处理。

报告斜率显著性，但不将其作为交易门槛。

```text
slope > 0 -> 允许主动倾斜
slope <= 0 -> benchmark-only logic  # 非正斜率采用仅基准逻辑
```

SE/t/p/CI 使用按再平衡日期聚类的稳健推断，置信水平为 95%；
加权 R2 和覆盖度作为诊断。满秩时斜率可用性与单项诊断定义性独立：
常量收益的 R2 为 UNDEFINED；零 SE 的 t/p/CI 为 UNDEFINED，SE 保留 0。
每行拟合残差绝对值不超过 `8 × float64 epsilon × max(abs(本行实际收益), abs(本行拟合收益))` 时，
按数值精度内的零残差处理；不使用全样本尺度或 SE 阈值，斜率与截距保持等日期权重 OLS。
校准采用缩放、中心化的浮点归约，按再平衡日期归约聚类方差；数值验收使用对应量纲的容差。
每日期聚类分数在该日期残差形成及点积求和的浮点误差界内归零。采用同一缩放坐标，
误差界为未归零残差行的 `sum(abs(x_centered) × row_error_bound)` 加
`γ_n × sum(abs(score_terms))`，其中 `γ_n = n×epsilon/(1−n×epsilon)`。
诊断状态和原因随账本保存，冻结读取验证它们；
合法零斜率仍执行仅基准，不用伪造显著性或区间满足冻结要求。

Alpha OLS 校准属于监督学习。本文冻结评价方案使用开发期拟合的校准，评价期斜率固定；
显式选择模型季度重训时，组合仍使用本节声明的冻结 Alpha 校准输入。
每组在开发最终截止重新计算其实际验证路径的 Alpha 校准，纳入末日新成熟样本；不能仅冻结最后一个交易批次的旧状态。
开发增强账本保存 `final_alpha_calibration.json`；独立 `calibrate` 也可从明确的
`observations`、`scheduled_intervals`、`cutoff`、`selection_group`、`calendar` 和
`rebalance_schedule` 写入新校准 JSON。每条声明区间必须对应真实交易日历上相邻调仓的执行开盘，
不能用固定七个日历日冒充周频。calendar 的输入选择器明确拟合日历起点及所需端点范围，
覆盖率分母来自该日历中截至 cutoff 已成熟的全部计划区间，不能按实际有收益的行反推或缩小。
直接与开发最终校准均保存 `interval_semantics`，
包含调仓日程、next execution open 到 following rebalance execution open 的收益约定及 A 股统一交易日历约定。
测试 `backtest` 的 `calibration` 显式指定该文件，核合法截止、组别、校准状态及这些实际区间语义；
同组、同日程 CSI500/CSI1000 的共享校准必须一致，输入路径与参数真实记录。
缺少可用状态、覆盖不足或校准 UNDEFINED 时 BLOCKED，不能永久仅基准来冒充已冻结能力；
已合法校准的非正斜率仍可产生仅基准结果。持仓、执行、费用及预先确定的风险估计规则可合法演化，
历史市场预热不等于监督再学习。

### 5.19 风险模型

市值字段按经济用途分别使用，不设影响所有模块的全局市值开关：

- 公司规模暴露及大小盘分组默认使用总市值；已有 `log_float_market_cap` 与
  `log_total_market_cap` 原始因子各自保留含义，选择流通规模需要显式声明。
- 基本面估值分母保持现有 `total_market_cap`（CSMAR `Dsmvtl3`）口径，
  不用 A 股流通部分替代公司整体股权价值，也不重新构造另一种同名总市值。
- 换手率使用 A 股流通股数；通过市值除以价格推算时，股类、日期、单位及未复权价格必须匹配。
- 自建流通市值组合与市场收益代理继续使用 A 股流通市值。市场收益的权重使用此前可获得的值。
  普通流通市值不等于自由流通调整市值；实际指数的行业目标及比较权重直接使用其历史成分权重。
- 规模解释变量、投资基准权重和回归统计权重独立。特征中性化默认等权，
  Fama–MacBeth 保留等权与总市值加权结果；风险收益回归默认使用总市值平方根统计权重。
- 市值缺失不自动改用另一口径，只影响真正依赖该字段的计算。
  成交容量、模型日期权重与衰减、账户净值收益不因规模口径切换而改变。

风险风格：

```text
Size          = log_total_market_cap
Beta          = beta_252d
Momentum      = momentum_252_20
Residual Vol  = residual_volatility_60d
Liquidity     = amihud_60d
Value         = earnings_yield
```

风险预处理：

```text
原始风险事实
-> 3MAD 缩尾
-> 总体 zscore
```

不对风险风格应用 alpha 的行业/规模中性化。

行业暴露保持为 0/1 虚拟变量。

交易日 `s` 的因子收益估计使用 PIT 暴露、行业虚拟变量，
以及在 `s-1` 收盘观测到的上一交易日配置市值的平方根统计权重（默认总市值）；
响应为交易日 `s` 复权收盘到收盘的 `period_return`。
`t` 收盘的风险估计可以消费截至 `t` 的因子收益。

风险估计范围是所有符合资格且具有完整合法 PIT 风险输入的 A 股。
随后将得到的风险事实投影到活动基准成分股。
每个风险风格先按自身 eligible 有效样本缩尾和标准化，再选取联合回归所需的完整样本。
其他风格或响应收益的缺失不改变该风格的预处理拟合域。
若 eligible 拟合截面至少有两个有效值且总体方差为零，该期风格无法提供截面差异信息，
其有限原始值投影约定为零，包括原始值不同的活动基准资产；原始缺失保持缺失。
不足两个有效值时该风格输出缺失，不删除域外基准风险能力。

```text
beta_252d = rolling 252-session regression of stock period_return  # 个股收益的滚动回归
            on eligible All-A return  # 解释变量为符合资格的全 A 股收益
全 A 收益率 = 以前一交易日流通市值加权的股票收益率
residual_volatility_60d = population std of those model residuals over t-59:t  # 模型残差的窗口总体标准差
```

个股滚动窗口、前期市值、ADV 与特异风险历史绑定上市身份；重上市清空旧生命个股历史。
前期暴露及资格只与同生命当前收益配对，不能用新上市收益解释旧暴露。
beta 的名义窗口为252个本市场开市会话，默认至少189个股票/市场收益联合有效配对；
残差波动的名义窗口为60个会话，默认至少45个当期有效残差，按总体标准差计算。
缺失和停牌占据名义会话位置，停牌零收益不作为回归配对；当日配对缺失时仍可使用足够的历史配对估计 beta，当日残差保持缺失。
数值核按资产批量中心化并计算浮点 OLS，比较使用收益量纲与算法精度相符的容差。
因子风险的 `asset_return` 是复权收盘到收盘的 `period_return`。
窗口沿个股当日所属市场的开市会话累计；另一市场正常开市时，本市场闭市不清空历史，
也不插入个股观测或残差。复市个股收益与上次个股观测后至本次观测的 All-A 收益复利累计值配对；
All-A 每个 union 会话仍按当日 eligible 股票及其上一有效收盘流通市值计算，闭市资产的前市值保留到复市。
所属市场来自完整来源事件的 as-of 归属（含全市场闭市日事件）。开市日未知行情仍构成历史屏障；
期间任何 All-A 收益未知也使对应复市区间未知，不补零或压缩未知会话。

每日因子收益 WLS 设计：

```text
全部历史第 1 级行业虚拟变量
+ 六个风格
无截距
```

统计权重默认使用 `FactorRiskConfig.weight_column="total_market_cap"`，可显式选
`"float_market_cap"`；当前加权方式为 `weighting="sqrt_cap"`。统计权重为：

`sqrt(total_market_cap)`

若通过预乘的 OLS 实现，矩阵乘数为：

`total_market_cap ** 0.25`

样本检查：

```text
n >= max(100, 5p)
满列秩
```

因子协方差：

```text
history = 504 sessions  # 因子协方差历史长度
EWMA half-life = 126  # 因子协方差指数加权半衰期
20% 对角收缩
完整 504 会话跨度，默认 >= ceil(0.75 * 504) = 378 个 fresh 拟合日期
```

特异方差：

```text
trailing residual^2 = 252 sessions  # 特异方差使用的残差平方历史
EWMA half-life = 63  # 特异方差指数加权半衰期
完整 252 个自身市场会话跨度，默认 >= ceil(0.75 * 252) = 189 个 fresh 残差观测
```

不进行成对删除。窗口缺失日期仍占会话，不能只累计有效日期后压缩衰减或提前发布。已确认停牌的零收益可参与数值回归及残差估计，但不增加 fresh 计数；未知收益不补零。因子日期只有满足回归最低 fresh 截面样本数时才计 fresh；特异风险按资产自身市场会话衰减，已知闭市不推进该资产时钟。`min_fresh_fraction` 默认 0.75；显式 `minimum_factor_dates` / `minimum_specific_dates` 可覆盖最低 fresh 数，但不能跳过完整窗口跨度。

协方差必须有限、对称且为 PSD。

缺少特异风险时，可合法回到基准的股票目标主动权重为 0；不能交易且有主动暴露的旧持仓
仍需要真实风险依据；缺失时本次依赖风险的新目标跳过，保留真实持仓、独立退出及
可计算诊断，对相应风险标记未定义，不能伪造零风险。

因子历史不足 -> 仅基准。

### 5.20 增强组合优化器

正式目标函数：

```text
最大化预期 alpha
       - 目标交易的预期交易成本
```

预期交易成本与执行账本使用相同的历史费率及市场 `prepare_trade_cost` / `PreparedTradeCost` 责任模块；
优化器不维护平行成本计算器。

风险作为硬约束，不作为风险惩罚项。

唯一求解器：

```text
SLSQP
ftol = 1e-12
maxiter = 1000
确定性起点
```

行业、风格使用等价双边仿射约束。换手使用绝对值的精确辅助变量表示：每个股票及现金
权重对应 `u_i >= +(w_i-current_i)`、`u_i >= -(w_i-current_i)`，且
`0.5 * sum(u_i) <= target_turnover_limit`；辅助变量上界为
`adv_participation * ADV_i / NAV`，现金无 ADV 上界。消去辅助变量即恢复原始换手及容量
可行域，并使零交易点保持可用于 SLSQP 的连续约束。
固定应收不参与换手；成本目标将 `u` 与权重变化转换为独立买/卖金额，使用同一
`PreparedTradeCost.totals_and_gradients` 的真实费用与解析边际费率。线性费率和平方根冲击
保留原经济含义；求解器试探负侧金额时只作切线连续延伸，实际成交仍要求非负金额。
不对含绝对值的目标作有限差分，也不以辅助变量虚构真实交易费用。
独立解后验证仍检查原始绝对偏离、换手与参与率，不更换目标、求解器或放松边界。
每次成功求解都必须经过独立解后验证。股票和现金权重均以包含应收的全 NAV 为分母；
未支付应收作为固定预算常量保留，不能转为现金或参与买入。支付后的现金由原账本确认。
现金始终是一条显式资产，即使当前余额为零；其基准、alpha 与风险暴露均为零。
目标股票与现金权重采用费用前 NAV，费用必须由目标现金预留；真实净交易费用为 `c` 时，
成本后股票权重为 `w_stock / (1-c/NAV)`，成本后现金权重为
`(w_cash-c/NAV)/(1-c/NAV)`。单股主动边界与跟踪误差约束及成功结果诊断均使用成本后权重；
换手、参与率和原股冻结仍使用费用前真实交易金额。冻结股数不因费用改变，权重会随净 NAV 改变。
持久化目标现金为扣除执行时应收与请求股票金额后的非负余额。执行时新确认应收可能
使旧目标超出可用预算，仍保留原请求及未成交原因，不用未来权益重新决定目标。
只有非零股票持仓或非零基准参考权重进入行业/优化股票依赖域；仅剩应收的已退出资产
保留应收、生命周期与估值账本，不因其旧行业/金融分类未知阻塞无关的新股票目标。
公开 `backtest` 保存同一个实际 `OptimizerConfig` 对象的全部解析值，包括未显式覆盖的默认值；
首次实际买入的换手豁免仍作为单独政策记录。
仅基准、简单组合及中性投影的可执行目标共用这条预算：股票目标总权重至多为
`1 - unpaid_receivable / NAV`，比例缩放不改变已选股票腿构成。
新目标先扣应收与缺预测冻结旧仓的预算，半步构造再与合法旧仓平均；
不能在半步之后重新归一化补满现金。`raw_target_weight` 是预算/半步后的投影前研究目标，
`target_weight`/`target_notional` 是股票腿中性投影后的实际请求；投影保留股票总量。
仅基准分支的实际成交若因费用、现金或执行约束偏离基准，不能因分支名称认证风险满足；
风险历史不足时实际 TE 与全约束明确未定义。
决策时已知费用后的可行性复用同一执行会计核，
以当时持仓、NAV、ADV 和已知费率核算，保留冻结资产的真实持仓；不得把未来开盘价格、
权益现金或成交许可用于决定目标是否合法。

硬约束：

```text
sum stock_and_cash_weights = 1 - unpaid_receivable / NAV  # 不可交易应收保留在全 NAV 中
single-name active +/- 0.50%  # 单股主动权重边界
annualized tracking error <= 5%  # 年化跟踪误差上限
one-way target turnover <= 20%  # 单边目标换手上限
industry active exposure +/- 2 percentage points  # 行业主动暴露百分点边界
each of six style active exposures +/- 0.20  # 各风险风格主动暴露边界
ADV participation <= configured cap  # 参与率服从配置上限
```

禁止：

```text
备用求解器
动态放松约束
风险惩罚项
隐藏的软可行性挽救
```

Alpha 或风险不足时执行已定义的仅基准分支，这是合法协议结果。
全无有效预测时先确定持有与明确独立退出，不消费新目标专用的行业、风格/Size、
校准或风险估计上下文；仍校验真实数量、估值、权益、执行许可和容量事实。
行业/风格双边约束使用同一 SLSQP 的仿射不等式，目标、精度和迭代上限不变。
全 NAV 主动风险和股票腿中性暴露分别按自身非零系数核验原始事实，
仅忽略归一化浮点舍入量级的系数差，不将未知暴露补零来认证约束满足。
无法核验的目标跳过；实际 TE、股票腿暴露及约束状态分别记录可用性，
一个指标缺失不抹掉其他可计算指标。
可证明的预算、换手、容量或冻结持仓约束不可行，记录本次调仓 `SKIPPED`，
继续原持仓记账；通用求解器失败、数值失败及无法归因为经济约束的解后验证失败仍报错。

可投资集合：

```text
仅当前基准成分股
outside-index target = 0  # 指数外目标权重
金融成分股固定在基准，active=0
非金融成分股允许倾斜
缺预测旧仓保留且不加仓；风险/PIT缺失按其实际依赖标未定义，不编造零风险
```

首次 OOS 增强持仓：

```text
初始股票持仓 = 0
cash = initial_capital
```

首次实际买入正常计入成本和容量，只豁免常规换手上限；未成交不消耗此建仓标记。

### 5.21 简单组合构建

简单分位数等权/流通市值加权组合按 `top_quantile` 构建。
`neutrality` 可选 `none`（默认）、`industry`、`size`、`industry_size`。
选择中性化时，默认以完整合法研究池的流通市值权重为参照；也可通过 `benchmark_id`
与 `inputs.benchmark` 显式选择当期真实历史指数权重。指数参照不要求增强组合的风险输入或校准。
在候选合法投资域内，求与原始目标最接近的非负、总权重为 1 的目标，
使所选行业权重及标准化稳健 log 市值暴露处于基准允许偏离带内。
`industry_neutrality_limit=0.02` 为行业 ±2 个百分点，
`size_neutrality_limit=0.2` 为 Size ±0.2 个标准化单位；显式 0 表示精确匹配。
Size 顺序为 log、市值截面 MAD 缩尾、总体标准差标准化。规模暴露默认读取
`size_cap_column="total_market_cap"`，可显式选择 `"float_market_cap"`；
默认参照域及权重仍由流通市值确定；显式指数参照保持原始指数权重，均不随规模列切换。
指数成分在候选资格域之外仍参与参照暴露，但不能因此被买入；规模暴露使用候选与指数成员的
联合有效域及当期 PANEL 市值。指数行业中性不依赖市值，收益比较基准仍独立配置。
缺少当期指数权重快照时跳过该次依赖调仓，不沿用未来快照或编造指数。规模列缺失不改用另一列，
也不删除缺失成员后重新归一化基准来声称全域规模中性。仅行业中性不依赖总市值。
基准域不随预测缺失缩小；已有缺预测持仓不加仓，未持有的缺预测成员不新买入。
不能先裁掉基准成员再声称全域中性。经济约束不可行时跳过本次调仓并记录原因，
继续实际持仓、权益和估值；数值失败单独报错，不静默降级或伪造协方差。
这些简单组合不额外启用增强组合的目标换手、跟踪误差和主动权重边界。
非调仓日仅延续已知持仓时，行业或规模上下文缺失不阻断已知数量与现金记账；
相应目标/实际敞口记录 `UNDEFINED` 和具体原因，不伪造基准或零敞口。
真正构造新目标、初始化和必要维护时，仍要求所用行业/规模事实及约束可行。
已有输入中的上下文缺项只使该次新目标 `SKIPPED`，并保留缺项原因、真实持仓与明确独立退出；
输入源未提供、非法结构与数值程序失败不属于这种普通缺项。
选中股票的流通市值缺失、不可解析、非正或非有限，使依赖它的市值加权目标跳过（`target_float_market_cap_missing`），保留已证明的独立退出；
等权目标不依赖该字段，不受无关市值异常影响。
全部股票已失去资格或实际股票腿为零时，不因股票腿中性暴露无定义而阻止全现金记账或
已证明的独立退出；相应暴露保留 `UNDEFINED`，仍单独验证可算且适用的全 NAV 风险约束。

账本分开记录原始目标、约束目标、基准和实际成交权重及暴露。未来执行限制或价格变化会使实际
组合偏离目标；未知暴露、域外持仓及非正 NAV 明确记录未定义原因与可知的排除权重，不能把
部分已知暴露当作完整结果。中性暴露以股票腿为分母，现金单列；
基准未知行业权重保留，不将已知 95% 重新归一为 100%。
执行、现金、ADV 与成本共用实际经济账本。

固定预测的持仓对照可选择 `quantile_equal_weight_half_step`：换仓日先由原简单组合 owner
生成等权目标，先分配扣除应收和缺预测冻结旧仓后的可投资预算，
再与合法旧仓权重各取一半；缺预测旧仓在两侧保持原权重。首次从现金开始只建半仓；
剩余现金不重归一化；这是 `neutrality=none` 时的最终目标。启用中性化时，该半仓权重
是原始目标，随后归一股票腿做中性投影，再还原原始股票总权重，保留半步现金。失去资格的旧仓按独立退出规则处理；
仅预测缺失的旧仓保留且不加仓，实际卖出仍遵守原交易限制。
该构造在本次组合参数中记录，可显式选用原模型和预测；报告同时比较毛/净收益、风险、现金和换手，
不能把敞口差异导致的收益变化只归因为交易成本。该固定规则不根据回测收益调节。

### 5.22 组合时序与核算

主时间线：

```text
t 收盘决策
-> t+1 开盘实际执行
```

分别记录：

```text
target_weight
pre_trade_weight
realized_weight
cash_weight
```

正式 NAV 是交易日开盘执行后的路径，随实际持仓/收益演化。
`portfolio_daily.parquet` 的 `cash_post` 为执行开盘、实际成交并扣除费用后的现金金额，
`cash_weight_post = cash_post / nav_post`。均值与末次现金权重、末次现金金额进入绩效摘要，
报告保留日度表引用、候选键及末次执行日期；不以目标现金或区间末估值替代实际执行后暴露。

计划调仓日的合法空选股域产生全零股票目标，不读取不合格资产的预测。
该目标仍经过成交约束、费用和自融资核算：可卖持仓正常退出，禁止卖出或闭市持仓保留，
执行事实未知的对应方向不成交；必需估值事实未知仍阻塞。合法上下文缺少预测时信号保持 NULL，无信号旧持仓按规则保留；实际提供的非有限预测或无法对齐的键仍拒绝。


正式连续权重研究回测不模拟整数股或 100 股交易单位。

权重与名义金额用于执行；cash_equity 持仓数量使用实际原股股数，以 raw open/close 估值，现金应收及锁定股由同一账本单独保留。股数及经证据证明的权益换算采用精确有理数，金额最终舍入为有限浮点值；不得因有限金额除以极小/极大价格而溢出或抹掉权益。普通分红和送股直接消费现有 INTERIM corporate_actions，按登记股数确认；原股账户不能再用含分配收益的复权价估值。连续权重研究仍不模拟整数股和 100 股交易单位。

组合会话采用沪深本市场日历的开市日期并集。个股的真实 PANEL、因子、标签和 ADV 仍只使用该股本市场会话；组合估值视图单独处理并集会话，不向原始数据或 PANEL 写入闭市假行情。
组合会话的来源行情只投影资产、日期、市场归属与填充标志四列，按显式输入筛选保留截至计算必要末端会话的历史，
一次物化并稳定按日期排序后逐日消费，避免短日期块重复解码年度文件；内存随所选四列历史规模增长。
这仅指逐日消费阶段；前置归属/日历轴核验及市场迁移的定向历史读取仍由现有 owner 独立执行。
输出起点不截断这段既有历史，显式排除的文件、年份、日期或行仍不进入计算。

闭市时买卖权限均为禁止，未成交订单不递延。执行 mask 的下一本市场会话必须等于实际执行日期；复市首个合法开盘按当期目标和再平衡/维护规则执行。闭市不向 ADV 加入假零观测，也不丢失已有本市场历史。收益使用同一估值端点：真实开盘 100、当日真实收盘 110、随后两个本市场闭市、复市开盘 121，对应并集开盘 mark 为 100、110、110、121，区间收益为 10%、0%、10%，累计 21%；复市价差只核算一次。

组合账本与持仓诊断共享估值端点，但闭市 mark 不生成因子收益样本。归因回归只使用入口和出口均有真实 PANEL 行的配对域，保留真实确认停牌行的语义；不能识别的持仓归因保留未定义及事实不完整原因，不填零归因。已证明的历史 as-of 风险暴露可以保留到闭市当前轴。指数基准独立要求自己的合法区间端点；个股闭市不能证明跨市场指数闭市，缺指数开盘仍为 `benchmark_total_return_open_missing`。增强组合的初始持仓只能在实际执行日合法开市时建立。

组合使用唯一研究估值路径，将正常停牌、已证闭市、普通漏报价与确认退市分开。
暂估仅用于既有持仓会计，不回填原始 PANEL、执行 mask、因子、标签或 MODEL/PRED：

- 实际执行日买卖许可未知时有订单则不成交，记 `execution_fact_unknown`；已知禁止仍记 `market_blocked`。不递延订单，未成交卖出不释放现金。估值补充事实不授予任何成交许可。
- 已证明停牌/闭市可沿用同一生命周期、当时已观察到的有效价格。普通漏报价的既有持仓，在生命周期及原股单位可证明且已有可信原始报价锚点时可暂估；从该报价按当前已可用公司行动的实际生效顺序重算，迟披露的较早行动不回改已发布前缀，也不因滚动暂估时点被遗漏。停牌 PANEL 内非空沿用价不算真实新报价；真实同价报价仍重设锚点。披露原始报价时点、依据、账龄与恢复影响；不据此成交、增加现金或改写未知许可。无可信锚点或生命周期/单位不能证明时仍失败。当日收盘不能填更早的缺失开盘。
- 已确认退市且无可靠可用报价/回收事实时，账面暂按零估值，诊断为 `confirmed_delisting_zero`；这是保守研究假设，不表示股份被注销。保留精确持仓单位和原上市生命周期，不假造卖出、成交量或现金。已有可靠且时点合格的报价不能被清零覆盖。
- `a_share_equity_events` 只通过现有 reference 原件接纳和市场会话 owner 进入组合。事件标识、原权益 `(asset_id, listing_date)`、生效日、证据公开日、实际取得时间、原件哈希与定位、CNY 原股每股金额和实际股数换算证据均须明确。按实际已可用且已生效的会话确认，不倒改较早历史。日期级公开证据保守从次日可用。原股每股金额直接作用于登记或持有的实际原股股数；股数转换核生命周期、生效时点及已有权益接续证据，权益接续须有实际原股单位证据。
- 可靠转让报价只用于估值，仍禁止据它成交。实际现金回收使用独立 `recovery` 现金流与唯一权益调整记录；部分保留、注销、缩股或换股按证据数量转换，不用 principal 或假成交掩盖。事件在持久化流中只能使用一次；同原权益在同一会话聚合出多个无已证顺序的事件明确拒绝。
- 已证明的现金回收或继承转换，在同一可用且生效的财富边界完整确认：结算前旧权益的研究估值包含已证明替代现金与剩余/继承单位的可用价值；同一时间戳的下一 entry 一次转换为实际现金和单位。前一 end 与后一 before 连续，不能先将旧权益记零、下一期才补现金而制造假回撤。该替代权益估值不可用于成交；无可靠继承估值、数量或权属证据时，仅实际持有该权益的账户受影响。
- 原权益已注销后，旧生命周期的同代码报价不能恢复旧权益，也不能再按该旧生命周期买入。不同代码或上市生命周期的转换只有有证据的承接标识和精确换算比例才能产生目标权益；同一代码同时持有两个生命周期而既有键不能唯一表达时明确阻塞，不自动拼接。目标权益缺可靠估值事实时也明确阻塞，不制造报价。没有衔接证据的重新上市绝不恢复旧持仓。 已证明的衔接可支持无本地主表记录的外部后继权益；已有本地记录但生命周期未知时，不能把未知解释为已退出上市，依赖该事实的回收资格仍未证明，不影响独立已证明的原权益转换。
- 从零估值恢复按权益金额入账，数量和现金守恒由同一个原生账本及 reader 校验。整个 NAV 为零的区间仍保留金额和后续回收；零分母收益率、换手权重明确 `UNDEFINED/nonpositive_start_nav`，不填有限值、不删区间。精确成交数量汇总若不能表示为有限浮点数，仅该诊断为 `UNDEFINED/nonfinite_result`，有限现金和 NAV 不丢失。
- 账本以 `research_valuation_policy=evidenced_rights_or_confirmed_delisting_zero` 记录估值科学政策，并保存生命周期、估值依据/时点/账龄、权益事件证据及数量调整。报价缺失、零估值和非交易权益回收所影响的持有区间不伪造因子归因。预先固定的 0%/50%/100% 未报价估值压力仍是同一成交路径上的绝对 NAV，不是重新交易策略或回收承诺；已清零权益不因压力路径恢复正值；已证明的替代现金及已确认现金应收不接受未报价股票腿的折价。暂估金额、股票因子暴露、ADV 未成交金额与请求换手只使用股票价值；应收继续进入全 NAV 分母，不能当成交现金或股票头寸。零股但正应收的资产跨期仍须连续，除非有支付、核销或转换等已记录事件。

没有真实、合格的回收/换股材料时，事件输入为空只表示尚无已接纳事实。不得把合成事件测试称为补到了真实供应商资料，也不得把代码通过或 PORT 发布当作真实研究已完成。

显式复用已有模型与预测时，原生 reader 核本次实际覆盖、窗口、列序和科学关系，组合及统计写新位置。代码变化的重算范围由研究者判断；研究估值结果不能冒充严格市场事实完整的净值。

#### 5.22.1 唯一经济执行账本

简单与增强组合共享同一个自融资连续权重转换。

执行顺序：

```text
交易前漂移
-> 合法卖出 + 成本
-> 计算剩余真实现金
-> 满足 ADV 与全部买方成本的合法买入
-> 对所有受现金约束的买入按比例缩放
-> 更新持仓/现金
-> 验证自融资账本
```

性质：

```text
cash >= 0 within numerical tolerance  # 容差内要求现金非负
受阻卖出永不产生买入预算
资产排序不改变经济成交
买入定量前计入全部成本
实际换手从已执行变化量派生
```

目标换手同时包含现金腿与股票目标变化；
实际换手使用已执行股票变化及对应的成本前现金腿。

买入分配不得依赖代码迭代顺序。

#### 5.22.2 交易受阻

卖出受阻时继续持有。

买入受阻时保留现金/未分配目标缺口；未成交数量不排队延续到后续交易日。

卖出受阻不自动冻结所有买入，但买入只能使用合法卖出和成本之后实际剩余的现金。

#### 5.22.3 ADV

ADV 约束有两项职责，但只有一份参数真值：

```text
优化器 -> 事前可行性/容量
执行器 -> 实际部分成交
```

实际成交是执行真值。

本市场已确认开市但整行缺失、已有合格日线中的局部成交额缺项，或同生命周期不足
20 个本市场会话时，ADV20 保持未知；开市缺项占据原窗口的一次未知观察，不补零、
不删缺项缩短窗口，不借未来信息或其他生命周期历史。只有连续 20 个真实已知观察
进入原窗口后才恢复，已确认停牌的真实零成交额仍是已知观察，闭市不增加观察。
组合会话中的市场归属或该市场开闭事实未知时，只清空受影响资产的 ADV 历史，待随后
20 个已确认本市场观察重建；其他资产不受影响。组合轴之外的周末不制造未知会话或重置。

已经合法形成的目标中，买卖许可已知允许但 ADV20 未知的订单均不成交，记
`adv_unavailable`。卖出（包括减仓）同样需要容量与冲击成本依据，不能豁免。
保留原目标、持仓和真实现金，不向其他股票重新分配缺口，不将未成交订单递延；
其他合法订单仍按卖出优先、成本和现金比例分配规则执行。该订单无成交、无交易费用，
受阻卖出不释放现金。未知许可和未报价仍优先保留原诊断，必要持仓估值缺失仍阻塞。
已知 ADV20=0 是容量为零，记 `adv_capacity`，不同于未知；非法负值、无穷值仍拒绝。

逐资产账本对受 ADV 缺项阻止的订单披露窗口观察数与缺项数，统计分别披露买/卖未成交订单数、
执行开盘原目标与实际持仓差额对应的未成交目标金额和受影响区间数；这些是执行覆盖诊断，不是收益损失或因果影响估计。
必要来源整体不存在、身份或结构非法仍由原生准入拒绝，不能将缺源伪造成局部空行情。
已有原件中的缺项不能仅凭空值就断言为供应商缺源。

增强优化器对 ADV 未知或为零的资产固定当前真实持仓，未持有资产不新增；
不能将受阻卖出当作可用现金，也不为维持旧权重伪造成交。冻结后仍须同时满足现金、费用、
预算、单股、换手、行业/风格及跟踪误差硬约束。冻结持仓的真实暴露与特异风险继续计入；
持有风险缺失或约束不可行时，本次新目标明确跳过，保留可真实记账的持仓与独立退出，
分别披露风险、股票腿暴露与约束的未定义原因；不将风险补零或动态放宽上限。
非法负方差、无穷风险值及数值/求解程序失败仍报错，不当作普通缺项吞掉。
执行时费用导致 NAV 变化可以使未交易持仓权重自然改变，数量与金额会计不能倒过来服从旧比例。

目标换手与实现换手分开报告。

#### 5.22.4 漂移

非再平衡日，持仓随资产收益自然漂移；不得把目标权重静默复制为实际持仓。

### 5.23 交易成本与费率表

正式初始 NAV：

`RMB 1,000,000,000`

上述为科研选择参考的正式本金。直接 `backtest` 的 `assumptions.initial_nav` 明确给出
本次初始 NAV，Python 与 CLI 消费同一 ExecutionAssumptions，不存在另一层覆盖参数。
接口接受任意有限正初始 NAV；偏离预声明本金须披露为不同研究假设。现金勾稽容差随实际数值分辨率缩放，
成交后的现金由同一批原生本金现金流与各项费用分量补偿求和，避免订单数增长时累加误差侵蚀净值；区间 NAV 也按金额补偿求和。
不设固定货币单位的正数底座，也不用勾稽容差作为破产门槛。实际 NAV 为负或非有限时拒绝；NAV 为零时保留现金与权益金额账，零分母比例未定义，后续已证回收仍可继续。
持仓比例以无量纲容差独立验证；金额容差为`min(512*ulp(nav), abs(nav)*512*ulp(1))`，
不能因最小次正规金额的一格很大而放过超额持仓或负现金。合法正本金不保证每种分配都可表示；
金额舍入无法满足自融资时明确失败，不增加最低资本，也不静默改写分配或增加NAV。

主成本参数：

```text
fixed slippage = 5 bps  # 固定滑点
sqrt impact coefficient = 25 bps  # 平方根冲击系数
broker commission = 3 bps / side  # 单边经纪商佣金
ADV participation = 5%  # 主方案参与率
ADV sensitivity = 1%, 2%, 5%, 10%  # 参与率敏感性
```

历史法定印花税/过户费使用实际执行日期的费率表。

正式执行/净 NAV 目标要求完整合法费率表；费率表缺失或结构非法 -> 执行目标 BLOCKED。

不使用当前费率回填历史，不产生净收益不完整的产物，不建立独立的仅毛收益公开产品模式。

唯一正式账本可以分别提供毛结果、成本和净结果：

```text
毛经济结果
- 实际建模成本
= 净经济结果
```

机构研究不模拟散户 5-RMB 最低佣金。

#### 冲击

对交易名义金额：

```text
q = abs(delta_weight) * NAV
participation = q / ADV20
impact_rate = 25bps * sqrt(participation)
impact_cost = q * impact_rate
```

没有隐藏的最小交易阈值。

### 5.24 流动性与容量

ADV20 定义：

`mean over t-19:t of Amount  # 平均成交额的完整窗口定义`

停牌 amount=0 仍保留在窗口中。

参与率 p 下的容量：

`C_t(p) = min_i ADV20_i * p / abs(delta_w_i)`

正式组合容量结论：

`5th percentile over rebalance dates  # 容量结论采用再平衡日低分位`

容量始终使用信号日原始目标权重变化 `decision_delta_weight`，不从实际成交反推；
该事前口径与执行开盘的未成交目标金额可能因隔夜漂移而不同。任一相关目标资产 ADV20 未知时，该期容量
为 `UNDEFINED/capacity_required_adv_missing`；含该期的全年和全样本汇总同样未定义。
保留缺口日期及观察数，不只汇总已知日期冒充全期容量；完整覆盖的其他年份可独立定义。

同时报告：

```text
中位数
最小值
年度分解
瓶颈股票
```

### 5.25 再平衡语义与基准维护

每个候选的 alpha 再平衡日程由 `rebalance_schedule` 冻结，支持日频、每周最后交易日和每月最后交易日。
随仓配置采用周频；将日程作为稳健性变体时，须在独立不可变运行中声明，不自动加入已冻结试验集合。

基准调样/权重变化与 alpha 再平衡是不同概念的决定，但每笔实际组合交易都使用同一执行、成本和可交易性账本。
所选 alpha 再平衡日程之外的基准成分增删，触发仅基准维护目标；
普通每日权重漂移不触发交易。

目标换手硬约束适用于收盘决策的目标组合，不适用于未来实际执行结果。

未来市场变化/执行导致的实际约束越界属于诊断，不追溯改变目标决策。
`predicted_tracking_error`、`portfolio_factor_variance`、`portfolio_specific_variance`
保留优化目标的风险含义。实际账户另记 `realized_tracking_error`、
`realized_factor_variance`、`realized_specific_variance` 及实际最大单股、行业、风格主动暴露；
`realized_constraints_status` 区分 `SATISFIED`、`BREACHED`、`UNDEFINED`，并记录原因。
实际 NAV 非正或非零主动持仓缺少必需风险时，不编造数值，也不追溯否定原本可行的目标。

### 5.26 绩效与归因

年化：

`252 个交易日`

独立组合报告：

```text
CAGR
年化波动率
rf=0 的 Sharpe
最大回撤
换手
毛/净结果/成本拖累
```

增强组合报告：

```text
年化超额收益
跟踪误差
以净信息比率为主
最大主动回撤
换手
成本拖累
实际行业/风格暴露
```

信息比率使用每日算术超额收益。

长期主动财富使用相对财富，不使用算术超额收益的累计和。

绩效归因：

```text
基准
行业
风格
股票特异项
交易成本拖累
```

归因在实际持有区间复用风险因子定义，并与经济账本勾稽。仅登记权益、未改变股数、现金、锁定股或应收的登记日不取消原本可定义的因子归因；实际公司行动调整记 `corporate_action_adjustment`，权益回收或从零股票估值恢复记 `zero_valuation_recovery`，对应归因保持未定义。没有明确有效的时序转换，不得将收盘到收盘风险序列展示为开盘到开盘归因。未解释残差必须命名为残差/未解释项，不能自动称为选股 alpha；Brinson 归因不属于本协议。

简单组合按五风格定义、增强组合按六风格定义分别拟合持有期归因，各组具有独立拟合域。`attribution_groups` 在统计协议中绑定完整风格列表及候选成员，
组身份为完整风格列表的规范 JSON 哈希。`holding_factor_returns.attribution_group` 与
`entry_at`、`factor_id` 共同构成主键；已有候选暴露与归因行通过 candidate_id 对应所属组。
各组独立保留缺失和不可识别状态，不用候选并集隐式要求额外风格。
已不在拟合研究池的持仓仍可用已识别行业、风格和实际收益做归因投影；其市值不参与
该次回归权重，单独缺失不取消投影。拟合样本仍要求合法正权重，持仓真实依赖的风格
或行业未知仍使对应归因未定义，不填补市值或改变风险历史。

暴露 != 归因。

### 5.27 统计推断

G/Q 主比较预先声明唯一 primary_portfolio_id，方向为同一日历上的 Q−G 净收益。
缺失日期不得 inner join 或补零；固定完整组合族，包括未定义项，应用 HAC、
块 bootstrap 与 BY 校正。零差保留均值等描述结果，推断退化时明确 UNDEFINED，不伪造 p=1。
同家族每日 RankIC 的 G/Q 配对为次要证据，要求同日历、同有效证券身份。
两侧均值和 Q−G 均差均使用两侧有限的同一配对日域；完整日历、配对数量、覆盖身份及固定 FDR 家族不变。
报告按 G/Q 读取 `prediction_diagnostics`，展示冠军预测自身诊断及对应统计实例身份，
不以家族比较代替冠军诊断，也不在报告阶段重新计算。
G 回顾性开发曲线披露终局选择对验证历史的使用，Q 因果曲线披露各次真实决策截止。
两类曲线均可参与显式请求的探索性统计，候选集合和分析配置由调用者声明。
G 开发曲线及其比较不能解释为独立样本外优越性证据；独立最终评价与开发探索分开报告。
曲线查询、图表及统计继承 `selection_mode` 和真实选择事实。
共享 candidate×fold 成本只计一次；选择和最终 refit 按组分别报告实际成本，
各实际操作的时间与资源观测按已记录范围报告，边界采样峰值不冒充全进程峰值；


RankIC、FMB 系数及模型 RankIC 比较保留完整输入交易会话序列：
HAC 滞后和 bootstrap 区块不得跨未定义日期压紧时钟。任一输入日期缺少合法观测时，
对应推断为 `UNDEFINED/incomplete_session_clock`，仍保存真实有效数量和描述均值；
多模型比较保留共同证券域及逐日期覆盖，不以删掉困难日期恢复推断。
G/Q 比较保留 `calendar_count`（两组 model_losses 配对输入的完整日期网格）及 `observation_count`；
前者不表示权威交易所日历，报告从二者派生缺失数，不持久化重复计数。
产物以`input_date_count`记录分析窗内原始dataset日期数，`observation_count`或`period_count`
记录有效数；报告从两者派生缺失数并同时披露口径。原始输入日期数不冒充权威市场日历完整性证明，
整日无成熟标签或统计量未定义仍属于原始输入日期域，不能先滤掉再计数。
因子回看、标签期限和组合净收益继续使用各自完整市场时间轴；组合缺失日仍阻断，不补零、不拼接。

#### 5.27.1 单因子发现

开发发现窗口：

`2014-2023`

每日 RankIC 均值推断：

`HAC / Newey-West 滞后 19`

FDR 家族按预先选择的因子及其 `FactorDefinition.family` 冻结；随仓 PV25/基本面选择对应的示例为：

```text
PV25 一个家族
Fundamental19 一个家族
```

即使部分因子没有有效观测 p-value，每个家族的校正仍使用预注册完整家族大小。缺失的原始/调整后 p-value 保持 `UNDEFINED`，数值占位不作为观测证据；产物报告各家族计划、已定义和缺失计数。

主方案：

`Benjamini-Yekutieli q=5%`

辅助方案：

`Benjamini-Hochberg q=5%`

FDR 家族不按结果删减成员；其他显式因子选择同样使用其完整预注册家族。
BH/BY 与 SPA/StepM/MCS 按实际选择截止、完整时钟、固定候选族及探索性/稳健性声明判定资格；
`sample_role=test` 不触发统一排除，也不单独证明样本外资格。合格的冻结最终评价保留所请求的
HAC、bootstrap、FDR 与联合检验；开发曲线的选后分析披露探索性用途与选择偏差。
单候选PSR仍可定义，DSR/PBO统一为UNDEFINED/insufficient_candidate_count；候选数取真实ID数量。
FDR 永不动态改变 CORE/FULL 因子成员。

#### 5.27.2 挑战模型比较

比较基线由研究 JSON 的 `comparison_model_id` 预先声明，必须属于所选模型集合。
随仓配置以 `等权排名集成` 为基线；运行与统计产物使用该显式身份，不由算法名称补默认基线。

每个挑战模型在共同日期上比较每日 RankIC 差；开发和留出区间分开报告。

每个挑战模型是一个模型家族滚动 OOS 策略，由各次合法重新拟合时经历史验证选出的配置组成。参数配置仍是具有指标/失败原因的可识别试验，但不会各自成为正式 OOS 挑战模型。季度优胜模型作为独立策略报告；季度、家族与配置标识不得被抹平。

专用模型现实检验之外，不对模型 p-value 再叠加 BH 校正。

#### 5.27.3 SPA / StepM / MCS

使用明确请求窗口内合格预测的逐日损失，保持完整日期网格和共同证券域；
开发预测与独立最终评价分开报告，G 回顾性曲线的相关结果明确标为探索性诊断：

`loss = -RankIC`

参数：

```text
stationary block length = 20  # 平稳自助法区块长度
reps = 2000  # 默认；可显式配置为10000等其他次数
alpha = 5%
派生固定种子
```

Hansen SPA 与 StepM 是正式诊断；MCS 仅用于稳健性。

#### 5.27.4 Fama-MacBeth

因变量：

研究配置所选 `evaluation.primary_label` 的原始标签；默认是 `forward_return_20d`。
选择 `forward_return_5d` 等其他批准期限时，FMB 使用同一标签及其可用时点确定成熟样本，
输出的经济解释与该期限一致，不独立读取二十日标签或借用其成熟域。

单因子回归分别运行；多因子回归仅使用 CORE。

每日分别执行带截距等权 OLS 与总市值加权 WLS；权重为 `total_market_cap`，不是流通市值。
截距的统一系数标识为 `intercept`，该名称不用于 FMB 输入因子；其他自定义因子名按实际字段保留。
缺失、不可解析、非有限或非正市值只使对应观察退出加权回归，不影响独立等权结果；合法样本不足则该回归未定义。回归 ID 以 `equal:` / `total_market_cap:`
区分两套结果；单因子与 CORE 多因子均保留。系数时间序列使用 HAC lag19。
异方差辅助检验的残差平方若为常数，其R²无定义，直接保留诊断未定义；不执行零总平方差除法。
辅助多项式列秩亏不等于主回归秩亏；合法的投影R²与有效秩仍按统计合同使用。
HAC lag19 为协议固定统计参数，不随主标签期限自动缩短。
期间回归系数按 float64 发布，等权与加权回归共享稳定的 OLS/WLS 计算路径。
汇总消费已发布的期间系数，并遵守 §5.27.6 的 HAC 退化规则；
常量系数序列的均值推断保持 UNDEFINED，不生成显著性。

开发和留出区间保持分离。

Fama-MacBeth 用于经济解释/稳健性，不是第二选择引擎。

#### 5.27.5 组合过拟合诊断

组合过拟合诊断使用研究配置显式选择的完整候选集合，候选定义见 §5.17。
`candidate_count` 记录实际候选数；候选身份、选择模式、状态与计数写入统计产物。
G 回顾性曲线可参与所声明的探索性诊断，结果仅涵盖本次集合及其真实选择过程。

CSCV-PBO：

```text
10 个连续近等长区块
length difference <= 1  # 连续分块长度差上限
不删除任何观测
全部 C(10,5)=252 个训练/测试组合
```

PSR 基准 Sharpe = 0。

PSR / DSR / CSCV-PBO 在明确声明的候选集合与窗口内作为诊断，不作为选择门槛。
开发与冻结最终评价结果分开报告；资格取决于实际研究事实，不仅是窗口名称。

#### 5.27.6 统计退化与发布

HAC 要求至少两个有限观测且 0 <= lag < N。常量或振幅不超过
8×float64 epsilon×最大绝对值的序列，均值推断为 UNDEFINED；零/非有限 SE
或非法后端区间也为 UNDEFINED，不发布巨大伪 t 值。缺失观测按已有共同覆盖
合同处理，Inf 属于无效输入；小样本不调用后端。Bootstrap 至少两次重复，
相同观测允许有限的点区间；非有限或反序区间不得发布。
秩亏回归不发布伪系数，SPA/StepM/MCS 的常量差分、零方差与非法后端 p 值
保持 UNDEFINED。报告和研究图片 只消费已发布统计，不补算第二真相。

### 5.28 稳健性纪律

主方案与稳健性是独立的不可变运行。

稳健性每次只改变一个主假设，不进行笛卡尔积搜索。

示例：

```text
再平衡频率
MAD 替代方案
训练时间衰减
分析师新鲜度
ADV 范围/参与率
成本敏感性
```

稳健性变体保留各自独立运行内的训练、候选比较及参数选择；不进入主方案的联合模型选择、FDR、SPA、StepM 或 PBO 试验范围。不得以稳健性或留出结果反向选择主方案。

覆盖正式协议 -> 探索运行；不得标为正式证据。

### 5.29 随机性与可复现性

根随机种子：

`42`

配置解析时，省略 `model.random_seed` 默认使用 42；显式合法种子原样传给模型后端，
不因模型名、折或季度被隐式重派生。需要独立子种子时，由研究者明确选择并传入。
解释的 selection、background、permutation 各自的 `root_seed`，以及需要随机分析的
模型统计 `root_seed`，省略时同样规范化为 42；显式值覆盖默认，显式 null 不能当作省略。
非随机评价不因此新增随机参数。

解释日期、展示行、背景及置换使用确定性派生 owner 和各自稳定用途键；
不修改进程全局随机状态，也不以执行顺序或 worker 身份派生。
匹配比较使用相同根种子、数据域、样本资格和抽样/置换配置；种子相同本身不保证资格一致。
paired group 比较继续使用其声明的 bootstrap 种子。

不得使用 Python 进程随机化的 `hash()`。

无法保证跨环境精确浮点行为时，可复现合同是在批准数值容差内的持久化语义相等；不声称普遍的跨平台逐比特一致。

随机种子能够改变研究数值输出，必须随模型与 trial 记录；它不是自动复用或失效键。

### 5.30 报告语义

正式报告必须分开展示开发与留出区间：

```text
开发 OOS：2019-2023
锁定留出区间：2024-2025
```

合并的 2019-2025 曲线只能作描述性展示，不能作选择证据。

必需报告家族：

```text
数据 / 覆盖 / PIT 就绪程度
因子诊断
模型选择与 OOS 预测
统计 / 多重检验诊断
组合
成本
风险
容量
归因
开发与留出区间对照
预注册稳健性
阻塞项 / 无效目标
```

研究者/agent从显式已发布数值产物撰写解释；图片直接消费数值，不要求完整run。缺失输入、失败和合法未定义分开说明，不虚构完成。

### 5.31 正式、探索与就绪程度

正式运行要求协议锁定语义，以及每个目标完整的必需输入。

必需来源缺失只阻塞依赖操作；已有数据问题局部处理，其他独立有效研究继续。无法合法计算的结果如实失败或未定义。

字段或覆盖缺失不构成永久备用语义、虚假值、当前状态历史回填、缩短窗口或隐藏模型变化的理由。

数据获取、延期和验收进度属于操作事实，不改变本研究合同；真实运行状态从对应持久化证据读取。

## 独立 ON/ID 标签诊断

使用现有 `prepare`，选择 `data_id=on_id_decomposition`，输入 `market_panel_1d` 和显式 `trading_calendar`。
`parameters.horizons` 默认 `[20]`，可从 1/5/10/20/60/120 日中显式选择；一次结果可供多个研究配置使用。
端点复用现有 forward-return 数学 owner，不要求先提供另一份 forward_returns 文件。

对信号会话 t 和期限 h：

- pre-entry：t 收盘到 t+1 开盘的对数收益，独立列出，不属于持有期收益。
- ID：t+1 至 t+h 的 h 个开盘到收盘对数收益之和。
- ON：上述各收盘到下一开盘的 h 个对数收益之和。
- endpoint：t+1 开盘到 t+h+1 开盘的简单收益，两端要求 `filling_flag=0`、`is_trading=True` 且有限正 `adj_open`。

`expm1(ID+ON)` 与 endpoint 在科学数值误差内对账；这不要求独立浮点运算逐位一致。
pre-entry、ID、ON 与 endpoint 均不由 can_enter/can_exit 决定；完整分解 qualified 同时要求合法端点及两个分量。
ON/ID 内部已确认停牌沿用估值/零路径约定，不因此将停牌端点认作真实成交价格。
内部 close 缺失可以令分解未知而端点仍合法；缺退出 open 不能抹掉已经完整的 ID。
非正或非有限价格只影响真实依赖它的组件，不补零、不未来填充，不阻塞其他合法资产/日期。

新值列为 `on_id_pre_entry_log_return`，以及各 h 的
`on_id_holding_intraday_log_return_{h}d`、`on_id_holding_overnight_log_return_{h}d`、`on_id_endpoint_return_{h}d`。
每个值有自己的 `_available_at` 和 `_reason`，另存分解 qualified 与原因。
数值使用 LABEL role，DataSpec 明确成熟映射；对数分量单位为 `log_return`，端点为 `decimal_return`。
经济收益端点仍按上述开盘/收盘事件定义；wd 复权价格的调整来源依赖当日收盘，
故可用时点按 Asia/Shanghai 15:00 收盘规范到 UTC（现代 A 股为 07:00）：
pre-entry 为 t+1 收盘，ID 为 t+h 收盘，ON/endpoint 为 t+h+1 收盘。
普通 forward-return 与 endpoint 共用退出会话收盘的成熟门槛，经济开盘端点和期限不变。
这是日频研究 EOD 可用性约定，区别于 09:30 的经济开盘事件；不冒称供应商历史实际发布时点已经独立验证。

生产使用显式本市场日历及 PANEL 已知市场归属、上市生命周期补齐计算轴，停牌/未知报价不能压缩期限，已知休市不计会话；不跨重上市连接。
内部未知报价不移动固定端点，只影响依赖该报价的分量；普通端点收益不要求内部路径完整。
输入选择器排除的报价和 PANEL 原件内部整日缺行均保留真实日历会话，未知报价不补价；公开输出仍限制在所请求的观察键。缺少必要日历或已知市场区间的 O/C 日期证据是输入错误，与局部缺行情区分。市场归属或上市生命周期未知只使该资产当前区间不可用；不猜测归属，后续有证据区间重新积累完整窗口。
原生输入的文件、年份、日期和行筛选保留完整声明的市场/生命周期事实，由显式日历补齐计算轴，将未选价格保留为未知。
输出仅保留选中 PANEL 信号键与请求输出窗口的交集，不把用于确定时钟的未选行重新发布。
裸目录的成员列表不能独立证明完整资产/生命周期域，明确拒绝；裸单文件按调用者声明的 PANEL 资产域使用，
其行/时间筛选从未筛文件保留市场/生命周期事实，交易日期仍要求显式日历证据。

输出说明沿 selected-label 的 `producer_diagnostics.label_semantics` 经 assemble 和 evaluate 保留。
供应商复权 open/close 的代数恒等式不等于独立公司行动验证，不含费用、仓位、强制兑付或策略净利润。
信号条件研究可用现有 assemble/evaluate 评价这些不同标签及其真实覆盖，不能把分组标签均值改名为持仓 PnL。

ON/ID 准备表只通过 parameters.horizons 选择期限，各期限的分量整组输出，不接受 label_columns、factor_columns 或 base_panel。

### 消费独立 PANEL 的计算时钟与 fresh 参数

`price_volume_facts`、`market_model_factors`、`forward_returns`、`on_id_decomposition`、`execution_mask` 显式消费 `trading_calendar`；PANEL 的 `market_type`、`listing_date` 与已知 `delisting_date` 界定自身市场和生命周期。日历只填计算键，不编造行情或恢复被选择器排除的公开输出。

日历证据只覆盖请求输出及实际消费的本生命周期 lookback/lookahead；保留原件中的市场/生命周期边界事实不等于要求无关历史全部 O/C 完整。必要区间内缺 O/C 仍报错，不能压缩未知日期满足窗口。
退市事实按 PANEL 观察时点逐步可知，以首次可知观察日与有效日的较晚者终止当期轴；提前披露不提前停止，迟披露不回删此前未知的历史。终止日包含在边界内，但只有显式 O 日才产生会话；后续重上市独立计算。暂时未知的市场或上市事实仍打断数值窗口；之后明确恢复相同上市日期时，保留该生命周期此前已经知道的退市事实，不能因分段而遗忘。新的上市日期不继承旧生命周期退出事件。

`price_volume_facts` 与 `market_model_factors` 接受 `parameters.min_fresh_fraction`，默认 0.75，可显式指定 (0,1] 并保存实际参数。适用统计窗口须走满名义会话跨度，达到 ceil(fraction × w) fresh；`filling_flag=0` 才计 fresh，停牌数值和未知报价按各计算公式区别处理。端点、路径与极值不自动套用部分窗口规则。

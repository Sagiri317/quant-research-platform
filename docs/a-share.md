# A 股低频研究

本文件说明研究目标、支持能力、研究推进顺序和量化合同。
能力说明以生产者和持久化消费者为依据；合成验证、真实数据覆盖和经济有效性分别提供证据。
正式协议是 a_share_low_frequency，入口 profile 为 a_share_csmar_choice。
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
| 报告与展示 | 唯一目标 DAG、按 run_id 的报告、CLI/API 和只读 Notebook | 入口存在与合成运行不证明 Notebook 环境就绪或真实经济有效性 |

真实数据就绪、开发完成和留出资格由对应输入验收及持久化运行证据判定，不由能力清单推断。

## 3. 研究推进顺序

| 顺序 | 工作 | 完成标准 |
|---|---|---|
| 1 | 按消费者盘点原始输入与缺口 | 量价、独立预期、财务版本和指数增强分别列资格；缺版本只阻塞实际依赖消费者，额度与权限明确 |
| 2 | 完成真实原始输入验收 | 字段、单位、键、生命周期、历史行业、费率和全收益基准齐备；区别范围内 PIT 适用与全量覆盖 |
| 3 | 运行开发区间的真实完整链 | 保持已冻结研究配置的窗口、模型与组合集合，报告所有阻塞和未定义项；先验证正确性，不做大规模调参 |
| 4 | 固定一份可讲解的研究报告 | 因子→预测→执行→净收益与风险/容量/归因勾稽，说明合成与真实证据的区别 |
| 5 | 准备严格留出准入 | 开发模型、排名方向、缩放器和 Alpha 校准状态均已持久化冻结后，再决定是否启动2024–2025检验 |

推进顺序不改变正式研究协议。未取得必要原始资料，不用清洗、固定滞后或缩小历史来补造 PIT。
真实留出执行仍须满足既有访问规则；本节不自动授权下载、消耗额度或解锁留出。

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

协议发生语义变化时使用不可变协议内容/哈希区分；不得用阶段号或版本后缀制造第二命名真值。

正式经济研究期：

`2014-01-01 至 2025-12-31`

预热数据按因子、标签、风险和会计的真实依赖向前读取，不以研究期起点硬截历史状态。
公共 `data materialize` / `dataset build` 的 Choice Interim 来源窗由同一协议 owner 规划，
从 `min(2011-01-01, 请求起点)` 至请求终点；processed/PIT 与研究数据集仍只发布请求窗口。
来源窗口进入其持久化身份，缺失会计依赖保持缺失值，不填零。
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

同一业务键/时间出现多个不同值且无法证明修订顺序 -> 含糊不明 -> 明确失败。

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

有限组成项的 TTM、存量平均值和复合因子按完整公式计算，不能因中间和/差溢出而拒绝可表示结果或制造缺失。TTM 最终值确实不可表示时明确失败；因子仍按既有规则将缺失组成项、非正分母及不可表示的最终结果记为 NaN，不截断或补零。

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
CSMAR 股票主表的声明 schema 不包含 `market_type`；公司状态与上市日期按时点向后 as-of 关联。
板块改名本身不构成供应商代码迁移事实，上市年龄继续按所属市场日历累计。

每次 PANEL 请求只规范化一次完整股票主表、市值与供应商行业历史，之后按原批次计算。
未来和无关资产的原有源字段校验仍保留；公开行业快照仍先按当批资产选择，行业冲突仍按
当前窗口末端判断。该复用仅在请求作用域内，不写缓存，也不提前裁剪历史改变失败语义。

不得用查询窗口的第一/最后交易日推断上市或退市。
同代码的已证明非重叠上市区间按当日主表 as-of 记录保留；同一上市日的状态修订仍按生效日消解。
CSMAR主表按实际Statdt保留同一上市生命期内不同状态记录；不同状态日期不能因共用Listdt被当作主键冲突。
区间空档不生成 PANEL 行，新生命周期重新计算上市年龄，停牌价格不继承上一段；重叠或倒序区间明确失败。
下游以 PANEL 的 (asset_id, listing_date) 作为同一价格历史的分组：次日执行资格、
1/20 日等前瞻收益及其成熟时间、价量滞后与滚动窗口均不得跨上市区间衔接。
旧区间末端没有同区间未来价格时，标签与成熟时间保持缺失；新上市首日不继承旧价格。
每个生命周期保留首次真实市场归属，即使相邻两次上市的市场相同，也不能沿用旧生命事件。
同一生命周期内的停牌、闭市和市场迁移继续使用既有权威交易轴，不切断合法历史。
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

信号日停牌或非正常交易股票不满足 eligible。

#### 5.4.3 历史行业

逐资产历史行业状态由 PANEL owner 统一裁定。CSMAR 的 `STK_INDUSTRYCLASS`、
`STK_LISTEDCOINFOCHG`、`STK_INDCLASSIFYVERSION`、`STK_INDCLASSIFYSETS` 分别提供
分类事件、公告、标准版本和层级；CSRC/CAPCO 官方公开快照通过独立
`reference/a_share_industry_snapshots` 输入提供有来源证明的观察状态，不冒充 CSMAR 事件。

行业状态保留 CSRC2012 或 CAPCO 标准身份。新制度的 ImplementDate 不自动提供每只股票
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
`(source_asset_id, timestamp, snapshot_id)`，按 timestamp 年份分区；每份快照仍只允许同一证券一行。
不得把季度名、统计期末、附件迁移路径
当作更早公开证据。资产须满足现有身份、正式市场范围与观察日存续条件；类别、次类和
行业码须与同标准词典父链一致。公司简称不等于行业名称，原有字典继续唯一解释行业层级。

已知但尚未公布的新事件，公布前保持当时已知的旧状态；仅制度变化或快照缺某资产，
不构成该资产的新行业事件。已存在而公告未知的新实施事件，从该实施日起表示旧状态
覆盖资格未证，行业四列一致 NULL；这不是声称市场当时已知新分类，不为它制造 available_at。
只有同日或更晚有效、具有真实公开证据的状态才能恢复资格。晚公布的旧事件或旧观察快照
不得覆盖较新状态、跨过未知屏障或反向改写更早的信号。首个合格公开初值之前保持未知。

同资产、标准、有效时点的公开分类冲突明确拒绝；同一可用时点和有效时点的跨标准冲突
不能靠输入顺序裁决。相同事实的多份证据不重复形成经济状态。无合法映射、名称不符或
身份未证保留资格缺口，不删除资产、交易日或基础价量事实。

PANEL 保留 `industry_standard_id`、`industry_level1_code`、`industry_level1_name`、
`is_nonfinancial_industry` 的原有含义。特征中和与组合风险共用标准加代码的行业分组，
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
的冲突状态明确失败，相同状态取最早可用时间，不依赖原输入顺序。

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
规模、换手、价值分母和研究资格按各自既有缺失规则处理。

正式成交额 Amount：

`Amount = Dnvaltrd + Ahvaltrd_D`

不适用的盘后成交额为 0；若按市场规则应当存在却在来源中缺失，则明确失败。

2011–2025 来源适配边界：Markettype 32科创板自2019-07-22、16创业板自
2020-08-24适用；Filling=1/2为已知全天非交易。只有已知市场、状态及历史日期能证明
不适用时才将盘后NULL置0，适用交易行缺失或未知适用性稳定失败；已有非NULL值保留。
2026以后NULL不套用旧适用表，本条不声明已支持2026市场规则。
摄取保留混合来源文件中的Markettype 64北交所记录：2025及以前属于已知盘后不适用域，
不改变正式研究CURRENT_MARKET_TYPES。依据为[北交所2025交易方式问答](https://www.bse.cn/qt/200025234.html)
及[2026新增盘后安排起草说明](https://www.bse.cn/uploads/6/file/public/202604/20260410155845_hs300rhb80.pdf)；
不将2026新规则回填历史，也不授权北交所研究执行。
依据为[CSMAR官方说明书](https://file.csmar.com/group1/M00/AA/99/CuIKV2XAnKiABeLbAB4nDGC5Ano840.pdf)
字段索引与既有fields_zh、[科创板开市说明](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20190821_4884321.shtml)、
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

#### 配置、场景与执行阶段

场景限定可选因子和经济资格，研究 JSON 决定评估因子、模型输入与参数、比较基线、
组合集合及访问窗口。配置中未选择的能力不自动补齐；配置不能绕过数据、成熟度或留出准入门。

随仓配置提供以下研究选择，因子数仅统计 `evaluation.factors`：

| 配置 | 场景 | 评估因子 | 模型 | 组合定义 |
|---|---|---:|---:|---:|
| [core6.json](../configs/research/a_share/core6.json) | pv_all | 6 | 11 | 5 |
| [pv25.json](../configs/research/a_share/pv25.json) | pv_all | 25 | 11 | 5 |
| [fusion_nf_core.json](../configs/research/a_share/fusion_nf_core.json) | fusion_nf_core | 44 | 11 | 5 |
| [fusion_nf_expectations.json](../configs/research/a_share/fusion_nf_expectations.json) | fusion_nf_expectations | 48 | 11 | 5 |
| [pv_expanded.json](../configs/research/a_share/pv_expanded.json) | pv_expanded | 52（50价量+2暴露） | 11 | 3 |
| [pv_nf_expectations.json](../configs/research/a_share/pv_nf_expectations.json) | pv_nf_expectations | 29（PV25+预期4） | 11 | 3 |

模型列统计两种排名集成及九种监督模型；六份配置均保留完整模型集合。
组合列统计 JSON 中的定义数量，每条定义分别生成 G/Q 两组候选；它不是账本产物总数。
未绑定研究配置的 profile 不隐式生成组合或账本目标。

| 阶段或访问域 | 配置与产物责任 |
|---|---|
| `stage=evaluation` | 显式选择因子与标签，`models` 和 `portfolios` 必须为空；只展开因子评估终端 |
| `stage=research` | 显式选择模型及组合；按所选目标生成候选预测、选择、统计和执行账本 |
| `access_window=holdout` | 须绑定完整开发窗口的同协议 COMPLETE 记录、对应终端集合和已验证冻结产物；评估子图完成不替代开发冻结资格 |

`fusion_nf_core.json` 选择 PV25 + Fundamental19，共44个因子；名字不表示只选 `FactorSet.CORE`。
配置中这44个因子继续分别做单因子FMB、RankIC/HAC/FDR及模型OOS研究。
多因子 `core_multifactor` 只用预先选中因子与 `FactorDefinition.factor_set=CORE` 的交集：
CORE6/PV25/PV-expanded各6，fusion_nf_core为12，fusion_nf_expectations为16。

研究数据集对外只有一个样本过滤真值：

`eligible`

`scenario universe` 是协议/组装内部概念，不与 `eligible` 并列持久化为第二真值。

A 股基础信号资格：

```text
正式 A 股市场范围
listed >= 252 market sessions  # 上市交易日数量门槛
非 ST/PT
信号日正常交易 / 未停牌
满足有效且为正的流动性 / 价格要求
仅使用信号收盘时已知信息
```

上市会话事实由同一个市场日历 owner 生成。`listing_age_trading_days` 仅在从上市日开始的市场归属有证据时保存精确年龄；首次行情的市场不能反填到上市日，未知年龄保持 NULL。`has_252_trading_sessions` 是 nullable boolean 原始资格事实：完整历史时准确判断门槛；初始历史未知但已证实归属区间累计达到 252 个合法市场会话时为 True；尚不能证明时为 NULL。迁市场按各段真实市场日历计数，不把无报价的开市会话压缩掉。它与 ST、交易状态、价格和流动性事实共同派生唯一 `eligible`，不是另一份样本过滤结果；所有研究和组合资格消费者继续使用统一资格 owner。首次已知归属之前的 PANEL 行保持未知市场和未知价格，不从未来借市场、补价格或默认可交易。

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

因此信号 `t` 的标签检查 `can_enter[t]`，不得再次位移到 `t+1`。

执行掩码不得持久化为面向模型的研究特征/上下文列；标签或组合在需要时，通过明确连接使用执行产物。

QRP 训练标签采用唯一的 execution-aware realized label：构造信号 t 的标签时，
该定义必需的未来入场执行事实决定标签有效性；与该期限无关的更晚事实不得
反向改写已成熟或已发布的历史 model_target。未来执行事实不得进入特征、eligible
或拟合截止时尚未成熟的训练样本。不存在 execution-unaware 的备用 target。

执行 owner 使用 nullable boolean：True/False 是已知允许/禁止，NULL 是 unknown。
缺失交易状态、涨跌停或必要价格不得被肯定化；已确认停牌为禁止，尾部缺少
下一 session 观测为 unknown，不能声称市场禁止交易。显式 IPO/no-limit 事件可以解除涨跌幅限制；缺少事件不是
无此例外的证据，仍需有效的当日市场事实。ST/PT 未知保留 NULL；eligible=False
只表示不具备入选证据，不能解释为已知非 ST/PT。组合遇到未知执行事实明确 BLOCKED。

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

`min_periods = full window  # 必须满足完整窗口`

不使用 80%/50% 备用阈值；预热不足 -> NaN。
成交额流向与收益成交额集中度保留缩放后的向量化求和；仅在正负项严重抵消时，
按原始有限输入精确求该行比值，保留可表示的小量与排名方向。分母为零仍为未定义，
不以补零或扩大高精度计算范围改变普通路径。

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
不采用库函数返回的伪零值。完整窗口、bias=False 及原有非退化数值核不变；
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
| return_volume_corr_20d / return_turnover_corr_20d | 完整20日Pearson相关；常数/缺失窗NULL |
| return_amount_imbalance_20d | sum(r*amount)/sum(abs(r)*amount)，完整20日 |
| zero_amount_ratio_20d | 成交额非正值占比：mean(amount<=0)，完整20日；NULL传播 |
| log_total_market_cap | 正总市值自然对数，否则NULL |

换手率及其所有衍生量统一 `volume/(float_market_cap/close)`，不读取外部另一口径turnover。
有限正输入的中间股数比值若溢出、下溢或落入次正规范围，局部采用等价的稳定乘除，
不让中间舍入把可表示的最终换手率改成0或NULL；缺失、非正值及停牌合同保持。
amount_shock与volume_shock都排除当日；amplitude保持(high-low)/close。
shock 直接由精确二进制窗口和计算完整比值后舍入，避免非零历史均值先下溢为零。
对数均值在普通范围保留既有计算；只有正均值低于最小浮点数时才从精确和计算对数。
真实零、缺失、负数及最终不可表示结果沿用各因子的原有合同。
零或无效分母保持NULL，不新增epsilon/fallback或负值截零。
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
使用既有 evaluation、季度模型/OOS、三个简单组合和统计/报告链。
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
每家机构按 declare_date 选最新有效预测
同机构/同日多个分析师 -> 机构内中位数
跨机构中位数 / MAD / 计数
FY1 = earliest not-ended annual forecast_end_date  # 第一个尚未结束的年度预测期
FY2 = FY1 forecast_end_date + 1 calendar year  # 紧邻的下一财年，缺年不跳到更远年度
forward EP = median forecast EPS / unadjusted signal-day close  # 前瞻盈利收益率定义
```

公开配置 `evaluation.forecast_freshness_days` 只接受 90 或 180，省略时为 90。
该值经 `ProcessedDataSpec.parameters_json.freshness_days` 统一绑定来源回看、公式、DAG 与研究身份。
180 日方案单独创建不可变研究实例，标为 `expectations.role=robustness`；保留其模型、预测与组合链，
但不将其候选或结果混入 90 日正式选型及 FDR、SPA、StepM、PBO 推断域。
180 日运行内部保留训练、验证和原排序作为稳健性诊断；基础描述、HAC、基准及归因仍按原定义计算。
正式联合检验与 PBO 标注 `expectations_robustness_excluded`，不将全部稳健性结果改成未定义。
统计合同按实际评估因子和模型输入派生此角色；只由模型消费分析师因子时仍执行相同排除。
冻结test的G/Q保留单项HAC与bootstrap，联合FDR标为`NOT_APPLICABLE`。
机构日中位数与跨机构中位数的临时精确值保留到最终 EPS/close 比值，落盘仍为现有 Float64。

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

规范特征流水线：

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
+ 去均值的 log_float_market_cap
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

行业截距与共同规模斜率使用组内中心化的等权 OLS 闭式计算。对有限浮点输入，
中间累积及投影保留精确有理数。`rescale=False`在原单位下将最终残差转换为float；
供zscore使用的`rescale=True`先将精确残差除以共同最大绝对值，再转换为float，
避免全部残差小于float可表示范围时丢失可表示的相对差异，也避免大残差中间溢出。
精确位于设计空间内的输入残差仍为零，不以epsilon合并非零差异。共同尺度不改变OLS投影或标准分数的定义。
设有效行业数为K：Size自身的p=K；非Size的p=K+1，满秩等价于精确组内Size平方和大于0。
复用同一精确组内统计判定，不另建one-hot矩阵或浮点SVD门槛。必要Size即使不输出自身feature，
也按同一eligible截面执行MAD缩尾再作为协变量；单因子与完整配置的同一因子输出必须一致。
辅助Size维持FACTOR角色，仅补数据依赖，不自动加入评价信号或生成无消费者feature。

#### 5.10.3 Z-score 标准化

规范 zscore 使用总体标准差 `ddof=0`。

特征零方差且 n>=2 时：

```text
observed canonical values = 0  # 零方差时已观测特征的规范值
记录诊断
```

`producer_diagnostics.canonical_features.zero_variance_by_factor` 按因子累计中性化后
至少两条有效观测且总体方差为零的日期截面数；诊断在完整日期截面上计算，跨存储分块累计，
与规范特征一同由原生 writer 发布，并由 reader 读取。

目标零方差或有效样本数不足时，该日期无效。

zscore 之后不再裁剪。

#### 5.10.4 缺失语义

原始经济因子的缺失继续保留 NaN。

模型矩阵中的规范缺失：

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
现有三列 (feature, feature_missing, feature_observed) 的唯一编码：eligible且结构无效为
(NULL,1,0)，设计有效但普通raw缺失为(0,1,0)，已观察有效为(z,0,1)，ineligible为(NULL,0,0)。
训练、验证、OOS及预测消费者的必需PK域，共用该模型实际输入的唯一有效性规则；即使只选missing/observed
标记，也必须检查来源feature，不能因为标记为有限数而绕过。raw排名模型不消费的canonical
结构状态不改变其既有缺失中性规则；非rank模型还要求直接输入完整。不新增行级原因表，原始输入和转换规格保留可复核原因。

统计和组合从已验证预测谱系绑定的持久化模型派生同一有效域，域内缺失的必需预测仍失败。
`prediction_eligible` 只决定alpha及简单组合入选，域外alpha为NULL；持仓和benchmark会计全集不随之裁剪。
既有持仓按原交易限制正常退出，unknown继续BLOCKED；增强组合的incomplete资产仍固定benchmark，内部优化零系数不发布为预测。

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

若信号行 `can_enter` 为假或 unknown，所有依赖该入场的标签均缺失。

卖出限制不延长固定标签期限；标签仍定义固定期限经济收益，而不是模拟清算结果。

每个标签持久化各自的：

`forward_return_*d_available_at`

signal time 为 t 收盘；entry/exit event time 分别为 t+1/t+1+h 开盘。
对有限正价格端点，收益按`(exit-entry)/entry`局部稳定计算，避免先求比值再减1合并接近零的可分辨收益；
普通、常量及非法端点的资格规则保持，最终不可表示值仍缺失。该保证随标签生产者身份绑定，不改标签期限或成熟时点。
日线一字板判定使用入场日 OHLC，information availability 为入场日收盘；
标签 maturity 为所需执行事实与两端价格均已可得的离场交易日开盘。
日频持久化日期使用 session 标签，`available_at = exit session open`；
不把执行事实的发生时间当成信号日已知信息。

#### 模型目标

`model_target` 只在场景研究数据集组装中、确定 `eligible` 之后创建：

```text
有效且满足 eligible 的 forward_return_20d
-> 横截面总体 zscore
```

一行缺失只影响该行，不得污染整个横截面。

零方差 / 有效样本不足 -> 该日期目标无效。

只存在一份 `model_target` 真值。其 `model_target_available_at` 是该截面真正参与规范化的
合格非缺失标签成熟时点的最大值；没有有效目标的行保持 NaT。训练与验证消费该独立成熟列。
逐资产原始 `forward_return_20d_available_at` 保持自身端点时间，不因其他市场较晚成熟而改写。
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

开发与留出使用同一数据集定义和唯一生产者，按实际请求窗口绑定各自的精确输入与产物身份。
开发标签生产者只访问开发可见末端；开发输入截止于 2023 年末。
截至最终开发 cutoff 仍未成熟、需要 2024–2025 独立 test 事实的尾部标签保持不可用，
不得为保存或重新评分而提前读取这些事实，也不得将其写入开发产物。
历史季度末未成熟、但开发冻结截止前已成熟的标签可以保留；每次选择仍按自己的截止时点重新过滤。
历史初始状态仍沿既有生命周期/asof owner 读取。

### 5.13 三角色、两组选择与最终留出

数据角色只有 train、validation、test。development 是开发访问阶段，
selection 和 refit 是执行阶段，不构成额外数据角色。

- 2014–2017 提供初始训练历史。
- 2018 是验证预测预热年：全部声明候选按季度使用此前合法历史训练并预测，
  不在预热年发布冠军、组合或 Alpha 校准。
- 2019–2023 是正式 validation。每个 candidate × fold 只训练一次，
  缩放器、排名方向/权重和监督系数只由该 fold 的 train 学习。
  两组复用同一份原始 train-only 验证预测。
- 2024–2025 是冻结的最终 test；不继续季度选择、模型重训或监督校准。

全局组 G 在开发结束时，按全部已成熟的 2019–2023 每日验证 RankIC 选择。
2018 预热行不进入 G 得分；不平均季度均值来替代每日均值。
G 的开发曲线使用期末选定配置对应的各季度原始预测，属于回顾性验证曲线。

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
此前在工作流中明确 BLOCKED: global_selection_requires_complete_development，不能虚构部分 G。

最终 test 绑定明确的开发模型及各组 Alpha 校准来源。
算法、配置、特征、系数、排名方向/权重和缩放器固定两年；
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

研究 JSON 的 `models`、输入列、`hyperparameters` 和 `parameter_candidates` 冻结实际候选，
不由算法能力表隐式补齐。声明`parameter_candidates`时，季度预检、拟合、refit与reader均消费该候选序列；
被覆盖的基础`hyperparameters`只保留在完整运行provenance，不影响MODEL科学合同或开发冻结复用。
实际候选、输入、标签、种子与权重等其他冻结事实仍须完全匹配。
随仓配置的模型数量见 §5.7，六份配置均选择11种模型。普通决策树使用既有 sklearn 拟合路径，
默认预定深度3、叶子最少样本比例0.001、种子42，不根据研究收益追加搜索。
这一区分不代表任何模型已经通过真实研究验收。

单因子仅作诊断，不参与优胜模型选择。

所有监督模型共享一个仅基于训练集拟合的 `ModelMatrixScaler`，作用于：

```text
规范连续特征
缺失指示变量
```

目标不缩放。

训练期零方差列的冻结 scale=1，训练样本变换为 0，不删除该列。
OOS 始终使用训练期 mean/scale；新值按 (x-mean)/scale 变换，不能永久置零。
fit、持久化、reload、transform 和 predict 共用 ModelMatrixScaler。
加权中心以 `weighted_mean_origin + weighted_mean_offset` 保存；后者每列统一保存最简整数比 `[分子, 分母]`（正分母），模型状态使用相同精确比率。旧浮点偏移 metadata 与旧 pickle 明确拒绝，受影响模型须重建。精确偏移不能以 binary64 表示时，标准化保留精确比率直到最终输出舍入；普通列继续批量计算。
加权中心的原单位语义不变，变换使用同一相对坐标
`((x - origin) - offset) / scale`，避免相邻浮点值的均值舍入破坏零中心。
模型 reader 校验完整拆分中心与总体尺度；旧单一中心 payload 不兼容，须由已验证上游重建。

排名集成和单因子不使用该缩放器。

#### 5.14.1 等时点样本权重

`equal_timestamp` 按完整timestamp分组，每个训练时点贡献相同总权重；最终权重归一化为：

`sum(sample_weight) = N_train`

不使用市值样本权重。

主方案不使用时间衰减。

预注册稳健性通过 ModelSpec/config 的 `training_weighting={"method":"equal_timestamp","half_life":756}` 执行；
按有序训练时点的年龄乘 `0.5 ** (age / half_life)` 后重新归一化为 `sum=N_train`。
该合同进入模型与trial身份、metadata及冻结复用，supervised缩放/拟合与rank ensemble方向/IC权重共用。
756-period配置单独运行，派生`training_weighting.role=robustness`；不得与其他half-life模型混入同一选型实例。
其内部训练、验证排序与单项HAC/bootstrap仍保留，正式FDR、SPA、StepM、MCS与PBO排除该实例，
原因使用`training_weighting_robustness_excluded`；冻结test的G/Q单项比较保留，联合FDR为`NOT_APPLICABLE`。
正式 A 股协议仅接受 `half_life=null` 主方案或独立的全 `half_life=756` 稳健性方案。
由 `AShareProtocol` 统一校验；主方案与稳健性混合拒绝，其他半衰期（含 378 或 null+378）
在正式入口明确拒绝，不标为 primary，也不进入正式联合检验。通用模型与 Crypto 的半衰期
合同保留；756-period 稳健性仍保留单项诊断，不用留出结果选择模型或参数。

缩放器的加权均值/标准差使用精确的最终训练样本权重。

ResearchDataset 的特征处理只接受一个具名配方，持久化 `recipe_id` 与 `source_columns`，
输出字段由同一合同派生，不保留自由组合 DSL。A 股配置 `feature_recipe` 默认为 `a_share_canonical`，
执行 `eligible → MAD winsor → industry+size neutralize → population zscore → missing/observed encoding`。
独立稳健性配置 `a_share_quantile_1_99` 只将缩尾替换为固定 1%/99% 分位点，后续处理一致；
它保留单项诊断，但排除正式联合推断，原因 `feature_recipe_robustness_excluded`。
两个配方均按完整截面执行，原始因子不改写；assembler 不根据 scenario 绕过合同。
Crypto 不接入该横截面流程。

EWMA先扣除当前有效观测的共同最小年龄再归一化；缺失最新session不抹掉有效历史。极小半衰期下，尾部质量保留到乘上收益或残差平方之后，只对最终风险值做float64舍入；真实不可表示风险明确失败。
FactorRiskConfig唯一拥有风险窗口、半衰期、最小观测、diagonal shrinkage、annualization与weighting；
OptimizerConfig唯一拥有active-name/TE/turnover/industry/style/SLSQP容差和迭代上限。
protocol序列化同一配置，运行消费同一实例；ADV participation由ExecutionAssumptions唯一拥有。
协方差保持既有 PSD 准入容差；这不允许成功优化结果含负方差或非有限跟踪误差。
优化约束、解后检查与诊断共用风险计算。浮点二次型为负时，以输入二进制值的精确算术
区分计算舍入与真实负值；前者恢复非负结果，后者明确阻断，不投影协方差或静默截断。

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
固定派生种子
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

GBRT 与 LightGBM 复用上述训练样本权重、仅训练期拟合的缩放器、确定性种子、
候选评分和持久化预测合同。原生参数及候选组合由研究 JSON 显式声明，
不因算法可用而增加未声明试验，也不以模型数量替代实际 OOS 证据。

### 5.15 模型选择

两级选择：

1. 按 §5.13 的 G 全正式验证窗或 Q 因果过去 12 个月窗，家族内以每日验证 RankIC 均值选择配置。
2. 在研究 JSON 选定的家族优胜配置中，以同一组、同一截止和同一共同域选择优胜家族。

A 股研究 JSON 的 `models[].model_id` 是具名模型家族身份；精确小写 `champion` 保留给组合预测选择器，不能作为 A 股模型家族名。
此规则由同一 A 股模型合同约束配置绑定、生产、选择元数据及模型/预测产物读回，冲突原因为 `a_share_model_id_reserved`；预测 bundle reader 保留外层 `output_manifest_metadata_invalid` 及该原因链。
正式 A 股 `models[].sample_filter_mode` 仅支持 `eligible`；省略字段沿用既有默认。`unfiltered` 在 A 股协议准入时以 `a_share_model_sample_filter_mode_not_supported` 拒绝，开发验证与最终留出使用同一资格合同。该限制不调整既有完整合法截面排名、eligible 输出和共同域投影顺序；通用及 Crypto 模型的合法 `unfiltered` 能力保留。
`winner` 和其他拼写不被解释为优胜选择器，继续遵循各自原有身份合同；不扩展原有大小写/标识符语法，也不限制 Crypto 或通用 `ModelSpec` 的同名准入。

精确并列时，按研究 JSON 中模型的声明顺序选择；候选清单只在配置中维护。
家族内并列时按显式 `parameter_candidates` 顺序选择。随仓配置保留以下保守声明顺序：

```text
Ridge / Huber -> 较大的 lambda
Lasso -> 较大的 alpha
RF / XGB -> 更浅 / 更保守的配置
```

必需试验失败意味着正式季度选择失败；不得静默删除候选。
同一季度所有参加比较的家族/配置在唯一共同有效(timestamp,asset_id)域上重算验证成绩。
先在各自完整合法截面预测，再投影共同域评分，不能先裁剪而改变rank；排名缺失中性规则不变。
只有raw因子的rank缺失保持中性；rank模型若消费canonical值或缺失/合法性标志，仍须满足来源特征的结构有效性。
合法部分季度 MODEL 只读取截至请求结束日的会话并据此生成有效验证折；
coverage、评分标签成熟截止、trial 和发布窗口共用该折，训练仍使用独立的 fold.train.end。
发布成员不保留晚于请求结束日才成熟的标签，reader 同时约束折窗口及标签截止。
未完成整个开发期仍不生成 global 选择及最终 refit。
每个候选的原始预测成员同时保存真实键、label、label_available_at 和 score_eligible。
score_eligible 只表示共同评分资格；成员保留该候选自身合法的经济预测行，不能将其裁成共同域。
选择元数据绑定精确预测成员、标签数据集身份、cutoff、共同域 semantic_identity 和 row_count。
生产评分和发布验收逐日期核对 PK、成熟域、得分与选择结论；行数相同不代表覆盖相同。
普通 reader 核对全部成员内容身份、冻结引用、候选顺序、预测成员、截止和 trial 父引用，
不重算已由可信发布者验证的选择成绩，也不预加载每个模型；实际预测加载仍验证模型状态。
此合同不声称识别恶意发布者同时改写结果并重新计算全部哈希的伪造。
季度候选元数据、各组各次选择和最终模型元数据分别持久化，顶层仅保存唯一索引。
每个候选同时输出每日 RankIC 的总体标准差（ddof=0）、ICIR、正值率、有效日期数及共同样本数；
ICIR 为每日均值除以该标准差，不年化；零方差和不可表示结果保留未定义。
同 family/fold 只准备一次真实训练域，各候选拟合使用隔离副本。G/Q 选择同一配置、
训练窗、权重、种子与后端事实时，只实际 refit 一次，两个选择引用共享模型成员及 trial；
不同选择分别拟合，不伪造第二次 attempted 或训练成本。
共同域只约束比较，不裁剪完整预测或经济账本；无法解释的预测漏行不得用于缩小比较域。

RankIC 为负的优胜模型仍是有效协议结果；不得因结果差而重新调参。

#### 排名集成

因子保留自然经济定义，不建立事后静态方向表。

等权排名方向从拟合历史的每日 RankIC 均值学习。每个正式输入都必须至少有一个
可定义的训练期 RankIC；单个输入方向不可定义即明确失败并报告该输入，不删除因子、
不补方向、不降低样本门槛。推断时某行排名缺失的中性贡献不等于训练方向可定义。

IC 排名因子权重：

`mean RankIC / sum(abs(mean RankIC))`

排名缺失贡献中性 0；不得对每只股票仅按已观测因子重新归一化。

A 股排名模型只学习排序所需方向或权重，不拟合或持久化未被消费的标签单位
`label_calibration`。组合的 Alpha 校准仍按 §5.18 独立执行；Crypto 标签单位转换
继续在真实 TRAIN/refit 域内拟合并冻结，见 [Crypto 标定](crypto.md#数值缩放与排名模型标定)。

方向和权重在每个候选 fold 的 train 内学习，并在对应 validation 季度内冻结；
2024–2025 最终留出按§5.13固定两年，不继续更新。

### 5.16 预测产物语义

组合只能消费持久化 OOS 预测产物，不得以活动模型对象或原始因子作为捷径；唯一例外是单独定义、由等权排名基线自身正式生产的基线信号产物。

预测行至少包含：

```text
timestamp
asset_id
model_id
raw_prediction
alpha_score
```

横截面 alpha 分数：

`alpha_score = (average_rank - (n+1)/2) / n`

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
派生随机种子
```

### 5.17 组合候选配置

组合由研究 JSON 的 `portfolios` 显式声明：

| 字段 | 责任 |
|---|---|
| `candidate_id` | 配置内候选身份；协议分别派生 G/Q 两组身份 |
| `construction` | `top20_equal_weight`、`top20_equal_weight_half_step`、`top20_float_cap` 或 `benchmark_enhanced` |
| `prediction_selector` | 已持久化预测中的具名 `model_id`，或保留选择器 `champion`，不借组合名称推断算法 |
| `top_quantile` | 简单组合的最高分位比例，须满足 `0 < top_quantile <= 1`；增强组合为 `null` |
| `rebalance_schedule` | `daily`、`weekly-last-trading-day` 或 `monthly-last-trading-day` |
| `benchmark_id` | 增强组合的 CSI500/CSI1000 基准；简单组合为 `null` |

`prediction_selector` 精确区分大小写；`champion` 读取同组、同季度选择记录中的 `selected_family_id`，留出读取开发期冻结的最终优胜家族。
具名 selector 只读取所声明模型家族，不隐式映射 `winner` 或其他拼写。模型家族与候选身份分别由配置声明，不从 `candidate_id` 推断选择器。

`top20_equal_weight` / `top20_float_cap` 是构建标识，名称中的 top20 不覆盖 `top_quantile`。
增强组合所需 benchmark、风险及数据依赖由同一组合合同派生；配置不能省略必要事实。
未绑定研究配置的 profile 不隐式创建组合或账本目标。

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
过去 12 个月
与日程匹配的下一执行开盘收益
alpha_score -> 带截距、等日期权重 OLS
expected_alpha = slope * alpha_score
忽略截距
```

覆盖要求：

```text
>= 80% 计划再平衡日期
and >= 6 valid dates  # 同时满足有效再平衡日期门槛
```

不使用更长窗口作为备用方案。设计矩阵（截距与 alpha_score）必须满列秩；
覆盖不足或秩亏时校准为 UNDEFINED，记录原因，不发布伪零斜率及 SE/t/p/CI/R2。
开发期仍执行仅基准逻辑，但它不表示斜率已确定为零。校准状态随账本发布，
报告和 API 读取账本末期实际状态，不重新拟合。

部分 OOS 增强请求使用同一份持久化优胜模型预测真值，
最多读取此前 12 个自然月预热，并截断在开发窗口起点。
组合账本只发布请求窗口；预热行不是第二份预测产物。

Alpha 校准使用当时全部满足资格的 A 股，
为同组、同再平衡日程的 CSI500 与 CSI1000 增强候选产生一份共享优胜模型斜率；
不同日程分别校准并冻结，留出执行按组别与日程读取匹配的斜率。
每个评分资产的信号、次日入口及成熟完成端必须绑定同一 `(asset_id, listing_date)`；
这也适用于未被组合持有的合格评分资产。退市重上市不能贡献跨生命周期收益；
同一生命周期内有可靠端点的正常停牌或本市场闭市仍按既有估值合同处理。

报告斜率显著性，但不将其作为交易门槛。

```text
slope > 0 -> 允许主动倾斜
slope <= 0 -> benchmark-only logic  # 非正斜率采用仅基准逻辑
```

SE/t/p/CI 使用按再平衡日期聚类的稳健推断，置信水平为 95%；
加权 R2 和覆盖度作为诊断。满秩时斜率可用性与单项诊断定义性独立：
常量收益的 R2 为 UNDEFINED；零 SE 的 t/p/CI 为 UNDEFINED，SE 保留 0。
每行拟合残差绝对值不超过 `8 × float64 epsilon × max(abs(本行实际收益), abs(本行拟合收益))` 时，
按数值精度内的零残差处理；不使用全样本尺度或 SE 阈值，斜率与截距保持既有等日期权重 OLS。
诊断状态和原因随账本保存，冻结读取验证它们；
合法零斜率仍执行仅基准，不用伪造显著性或区间满足冻结要求。

2024–2025 的 Alpha OLS 校准同样属于监督学习，整个留出期不得用成熟留出收益重新拟合或更新斜率。
每组在开发最终截止重新计算其实际验证路径的 Alpha 校准，纳入末日新成熟样本；不能仅冻结最后一个交易批次的旧状态。
校准参数、组别、cutoff 及精确上游来源保存于增强账本的 final_alpha_calibration.json 语义成员，
冻结引用验证这些身份；同组、同日程 CSI500/CSI1000 的共享校准必须一致。只有耗时/资源观测进入 operational metadata。
缺少可用状态、覆盖不足或校准 UNDEFINED 时 BLOCKED，不能永久仅基准来冒充已冻结能力；
已合法校准的非正斜率仍可产生仅基准结果。持仓、执行、费用及预先确定的风险估计规则可合法演化，
历史市场预热不等于监督再学习。

### 5.19 风险模型

风险风格：

```text
Size          = log_float_market_cap
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
以及在 `s-1` 收盘观测到的上一交易日流通市值权重；
响应为交易日 `s` 复权收盘到收盘的 `period_return`。
`t` 收盘的风险估计可以消费截至 `t` 的因子收益。

风险估计范围是所有符合资格且具有完整合法 PIT 风险输入的 A 股。
随后将得到的风险事实投影到活动基准成分股。
若 eligible 拟合截面的某一风险风格总体方差为零，该期风格无法提供截面差异信息，
其全域投影约定为零，包括原始值不同的活动基准资产；不删除域外基准风险能力。

```text
beta_252d = rolling 252-session regression of stock period_return  # 个股收益的滚动回归
            on eligible All-A return  # 解释变量为符合资格的全 A 股收益
全 A 收益率 = 以前一交易日流通市值加权的股票收益率
residual_volatility_60d = population std of those model residuals over t-59:t  # 模型残差的窗口总体标准差
```

个股滚动窗口、前期市值、ADV 与特异风险历史绑定上市身份；重上市清空旧生命个股历史。
前期暴露及资格只与同生命当前收益配对，不能用新上市收益解释旧暴露。
两个滚动窗口都要求完整交易日数量。停牌零收益仍计为交易日；
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

统计权重：

`sqrt(float_market_cap)`

若通过预乘的 OLS 实现，矩阵乘数为：

`float_market_cap ** 0.25`

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
>= 252 个完整日期
```

特异方差：

```text
trailing residual^2 = 252 sessions  # 特异方差使用的残差平方历史
EWMA half-life = 63  # 特异方差指数加权半衰期
>= 126 个有效观测
```

不进行成对删除。

协方差必须有限、对称且为 PSD。

缺少特异风险 -> 股票目标主动权重 = 0。

因子历史不足 -> 仅基准。

### 5.20 增强组合优化器

正式目标函数：

```text
最大化预期 alpha
       - 目标交易的预期交易成本
```

预期交易成本与执行账本使用相同的历史费率及市场 `trade_cost` 责任模块；
优化器不维护平行成本计算器。

风险作为硬约束，不作为风险惩罚项。

唯一求解器：

```text
SLSQP
ftol = 1e-12
maxiter = 1000
确定性起点
```

每次成功求解都必须经过独立解后验证。

硬约束：

```text
sum weights = 1  # 目标权重总和
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
求解器不可行、数值失败或独立解后验证失败，均是正式目标 `FAILED`。

可投资集合：

```text
仅当前基准成分股
outside-index target = 0  # 指数外目标权重
金融成分股固定在基准，active=0
非金融成分股允许倾斜
缺少合法预测/风险/PIT 上下文 -> active=0，不删除
```

首次 OOS 增强持仓：

```text
历史基准权重
cash = 0
```

首次主动再平衡正常计入成本。

### 5.21 简单组合构建

简单分位数等权/流通市值加权组合按配置的 `top_quantile` 构建，不使用：

```text
20% 目标换手上限
TE 约束
风格约束
行业约束
```

它们直接执行规则定义的目标；执行、现金、ADV 与成本仍使用和增强组合相同的经济执行账本。

固定预测的持仓对照可选择 `top20_equal_weight_half_step`：换仓日先由原简单组合 owner
生成完整等权目标，再与仍合格且有有效预测的当前持仓权重各取一半。首次从现金开始只建半仓；
剩余现金不重归一化。失去资格或预测的旧仓目标归零，实际卖出仍遵守原交易限制。
该构造进入组合语义身份，不改变模型或预测身份；报告同时比较毛/净收益、风险、现金和换手，
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
该目标仍经过原有成交约束、费用和自融资核算：可卖持仓正常退出，禁止卖出或闭市持仓保留，
执行事实未知仍阻塞；合法域内缺失键或非有限预测仍拒绝。


正式连续权重研究回测不模拟整数股或 100 股交易单位。

权重与名义金额是研究核算单位，不声称实际股数。复权区间收益真值已经包含获批准的公司行动经济效果；账本不得再次加入同一股息/公司行动现金流。实际交易单位、整数股与公司行动权益模拟仍不属于本协议。

组合会话采用沪深本市场日历的开市日期并集。个股的真实 PANEL、因子、标签和 ADV 仍只使用该股本市场的原有会话；组合估值视图单独处理并集会话，不向原始数据或 PANEL 写入闭市假行情。

只有当个股当日生效的市场归属对应权威日历明确为闭市、已有上市/退市时点事实证明其仍在同一生命周期内、且最近真实正常交易的合法复权收盘价之后不存在未知行情/市场归属/日历事实时，已持仓的组合估值才能沿用该收盘价。该沿用是本协议的研究 mark 约定，不保证供应商真实报价或可成交公允价值。假日填充行 `Filling=2` 的价格不能成为估值锚点。普通开市缺价、缺日历行、未知停牌或归属、跨生命周期价格及退市后未知清算均继续阻塞对应组合；不删除持仓或释放其现金。

闭市时买卖权限均为禁止，未成交订单不递延。执行 mask 的下一本市场会话必须等于实际执行日期；复市首个合法开盘按当期目标和原有再平衡/维护规则执行。闭市不向 ADV 加入假零观测，也不丢失已有本市场历史。收益使用同一估值端点：真实开盘 100、当日真实收盘 110、随后两个本市场闭市、复市开盘 121，对应并集开盘 mark 为 100、110、110、121，区间收益为 10%、0%、10%，累计 21%；复市价差只核算一次。

组合账本与持仓诊断共享估值端点，但闭市 mark 不生成因子收益样本。归因回归只使用入口和出口均有真实 PANEL 行的配对域，保留真实确认停牌行的原有语义；已知闭市导致缺配对且三端 mark 合法、决策暴露原本合法或决策本身已证闭市时，持仓归因记为未定义，普通风险事实或报价缺失继续阻塞。已证明的历史 as-of 风险暴露可以保留到闭市当前轴；当真实因子配对域不能识别闭市持仓归因时，保留精确未定义或事实不完整原因，不填零归因。指数基准独立要求自己的合法区间端点；个股闭市不能证明跨市场指数闭市，缺指数开盘仍为 `benchmark_total_return_open_missing`。增强组合的初始持仓只能在实际执行日合法开市时建立。

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

目标换手与实现换手分开报告。

#### 5.22.4 漂移

非再平衡日，持仓随资产收益自然漂移；不得把目标权重静默复制为实际持仓。

### 5.23 交易成本与费率表

正式初始 NAV：

`RMB 1,000,000,000`

探索性的低层 Python 调用可通过 `build_portfolio_ledger` 的 `assumptions` 参数，
传入显式设置 `initial_nav` 的 `ExecutionAssumptions`。正式 `research_run` 配置固定使用上述 NAV，
不提供初始 NAV 覆盖参数。低层接口接受任意有限正初始NAV；现金勾稽容差随实际数值分辨率缩放，
不设固定货币单位的正数底座，也不用勾稽容差作为破产门槛。实际NAV非正或非有限才按无效资金状态处理。
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

归因在实际持有区间复用既有风险因子定义，并与经济账本勾稽。没有明确有效的时序转换，不得将收盘到收盘风险序列展示为开盘到开盘归因。未解释残差必须命名为残差/未解释项，不能自动称为选股 alpha；Brinson 归因不属于本协议。

简单组合按已有五风格定义、增强组合按已有六风格定义分别拟合持有期归因；新增另一类候选
不得改变原候选的拟合域。`attribution_groups` 在统计协议中绑定完整风格列表及候选成员，
组身份为完整风格列表的规范 JSON 哈希。`holding_factor_returns.attribution_group` 与
`entry_at`、`factor_id` 共同构成主键；已有候选暴露与归因行通过 candidate_id 对应所属组。
各组独立保留缺失和不可识别状态，不用候选并集隐式要求额外风格。

暴露 != 归因。

### 5.27 统计推断

G/Q 主比较预先声明唯一 primary_portfolio_id，方向为同一日历上的 Q−G 净收益。
缺失日期不得 inner join 或补零；固定完整组合族，包括未定义项，应用既有 HAC、
块 bootstrap 与 BY 校正。零差保留均值等描述结果，推断退化时明确 UNDEFINED，不伪造 p=1。
同家族每日 RankIC 的 G/Q 配对为次要证据，要求同日历、同有效证券身份。
两侧均值和 Q−G 均差均使用两侧有限的同一配对日域；完整日历、配对数量、覆盖身份及固定 FDR 家族不变。
报告按 G/Q 读取既有 `prediction_diagnostics`，展示冠军预测自身诊断及对应统计实例身份，
不以家族比较代替冠军诊断，也不在报告阶段重新计算。
G 在完整开发验证窗上终局选择参数，其开发曲线仅保留均值、样本数、收益和风险等描述结果。
该曲线及其家族差值的 HAC/块 bootstrap、SPA/StepM/MCS、PSR/DSR 均为
UNDEFINED/global_development_selection_reuses_validation，不作为独立 OOS 推断。
同一协议资格判定适用于单组、跨组和报告；不能仅禁止 G/Q 配对检验。
G 回顾性开发曲线与 Q 因果开发曲线的差异只作描述；正式优越性检验来自冻结 test。
Q 的因果开发预测保留既有合格推断；冻结 test 的 G/Q 保留既有 HAC/块 bootstrap 合同。
共享 candidate×fold 成本只计一次；选择和最终 refit 按组分别报告实际成本，
逐阶段 planning/时间和资源观测完整保留，边界采样峰值不冒充全进程峰值。

RankIC、FMB系数及模型RankIC比较的统计时钟统一为`ordered_defined_observations`：
HAC滞后和bootstrap区块按有定义观测的位置计数，不能解释成相同数量的市场交易日。
推断总体限定为实际可计算日期；多模型比较仍先取原合同要求的共同有限日期域。
G/Q比较保留既有`calendar_count`（两组model_losses原始日期并集）及`observation_count`；
前者不表示权威交易所日历，报告从二者派生缺失数，不持久化重复计数。
产物以`input_date_count`记录分析窗内原始dataset日期数，既有`observation_count`或`period_count`
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

`Benjamini-Hochberg q=5%`

稳健性方案：

`Benjamini-Yekutieli q=5%`

FDR 家族不按结果删减成员；其他显式因子选择同样使用其完整预注册家族。
BH/BY发现性FDR仅development；holdout只保留raw OOS/HAC，FDR记NOT_APPLICABLE。
SPA/StepM/MCS仅用于Q的因果development曲线；G开发曲线按上述资格阻塞，holdout记UNDEFINED/holdout_excluded。
单候选PSR仍可定义，DSR/PBO统一为UNDEFINED/insufficient_candidate_count；候选数取真实ID数量。
FDR 永不动态改变 CORE/FULL 因子成员。

#### 5.27.2 挑战模型比较

比较基线由研究 JSON 的 `comparison_model_id` 预先声明，必须属于所选模型集合。
随仓配置以 `等权排名集成` 为基线；运行与统计产物使用该显式身份，不由算法名称补默认基线。

每个挑战模型在共同日期上比较每日 RankIC 差；开发和留出区间分开报告。

每个挑战模型是一个模型家族滚动 OOS 策略，由各次合法重新拟合时经历史验证选出的配置组成。参数配置仍是具有指标/失败原因的可识别试验，但不会各自成为正式 OOS 挑战模型。季度优胜模型作为独立策略报告；季度、家族与配置标识不得被抹平。

专用模型现实检验之外，不对模型 p-value 再叠加 BH 校正。

#### 5.27.3 SPA / StepM / MCS

使用Q组因果预测在共同开发日期的每日损失；G开发曲线不进入该检验：

`loss = -RankIC`

参数：

```text
stationary block length = 20  # 平稳自助法区块长度
reps = 10000
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
缺市值只退出加权回归，非法非正权重明确失败。回归 ID 以 `equal:` / `total_market_cap:`
区分两套结果；单因子与 CORE 多因子均保留。系数时间序列使用 HAC lag19。
异方差辅助检验的残差平方若为常数，其R²无定义，直接保留诊断未定义；不执行零总平方差除法。
辅助多项式列秩亏不等于主回归秩亏；合法的投影R²与有效秩仍按既有统计合同使用。
HAC lag19 为协议固定统计参数，不随主标签期限自动缩短。
期间回归系数按 float64 发布：精确非零系数若小到按 float64 舍入为零，允许发布 0.0；
DEFINED 表示该期间回归可识别且输出满足现行浮点合同，不承诺保留任意小的非零值。
精确真零与可表示的次正规系数仍按各自数值发布，不用任意阈值裁零。
汇总继续消费已发布的期间系数并遵守 §5.27.6 的 HAC 退化规则；
例如舍入后为常量零的系数序列，其均值推断保持 UNDEFINED，不生成显著性。

开发和留出区间保持分离。

Fama-MacBeth 用于经济解释/稳健性，不是第二选择引擎。

#### 5.27.5 组合过拟合诊断

组合过拟合诊断使用研究 JSON 所选的完整合格 Q 候选集合，候选定义见 §5.17；不固定为五项。
G 开发曲线从 Sharpe 试验集合及 CSCV-PBO 矩阵排除，其逐组合 PSR/DSR 明确为 UNDEFINED；
不得让 G 的选后收益影响 Q 的 DSR 校正或 PBO。candidate_count 记录实际合格 Q 候选数，
候选身份、状态与该计数共同写入统计产物并由报告原样读取。

CSCV-PBO：

```text
10 个连续近等长区块
length difference <= 1  # 连续分块长度差上限
不删除任何观测
全部 C(10,5)=252 个训练/测试组合
```

PSR 基准 Sharpe = 0。

PSR / DSR / CSCV-PBO 只使用开发区间，作为诊断，不作为选择门槛。

#### 5.27.6 统计退化与发布

HAC 要求至少两个有限观测且 0 <= lag < N。常量或振幅不超过
8×float64 epsilon×最大绝对值的序列，均值推断为 UNDEFINED；零/非有限 SE
或非法后端区间也为 UNDEFINED，不发布巨大伪 t 值。缺失观测按已有共同覆盖
合同处理，Inf 属于无效输入；小样本不调用后端。Bootstrap 至少两次重复，
相同观测允许有限的点区间；非有限或反序区间不得发布。
秩亏回归不发布伪系数，SPA/StepM/MCS 的常量差分、零方差与非法后端 p 值
保持 UNDEFINED。报告和 Notebook 只消费已发布统计，不补算第二真相。

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

所有随机组件从语义组件 IDs、折或季度派生确定性子种子。

不得使用 Python 进程随机化的 `hash()`。

无法保证跨环境精确浮点行为时，可复现合同是在批准数值容差内的持久化语义相等；不声称普遍的跨平台逐比特一致。

随机种子能够改变研究数值输出时，它属于语义协议标识的一部分。

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

报告只从已发布产物派生；必需产物缺失属于阻塞/失败，绝不以 `None` 占位或虚构状态文本。

### 5.31 正式、探索与就绪程度

正式运行要求协议锁定语义，以及每个目标完整的必需输入。

数据问题只阻塞依赖子图；无关的有效研究可以继续通过目标 DAG 执行。

字段或覆盖缺失不构成永久备用语义、虚假值、当前状态历史回填、缩短窗口或隐藏模型变化的理由。

数据获取、延期和验收进度属于操作事实，不改变本研究合同；真实运行状态从对应持久化证据读取。

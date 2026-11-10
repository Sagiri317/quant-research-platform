# 全球宏观流动性

宏观源、因子和股票研究输入分别准备。数据下载和文档提取在仓库外完成；`prepare-source` 规范化结构化 raw；`prepare` 计算决策面板或因子。各操作都写新目录，不自动训练、回测或更新历史研究产物。当前数据条件、研究假说与验证边界统一见[美国流动性与 A 股研究](../reports/美国流动性与A股研究.md)。

## 来源、版本和时间

`sources.macro.catalog` 映射供应商代码，`contracts.macro.MACRO_VARIABLES` 定义源无关变量、单位和频率。美国数据优先使用 ALFRED 历史版本；其他适配包括 HKMA、BIS、BOJ、旧金山联储和东方财富的公开接口。具体来源 URL、响应原文、获取时间及 SHA256 保存在原始目录。失败来源单独记录，其他来源继续；香港分页中已成功收到的页可独立保留，但明确标记不完整。

波动率扩展使用 ALFRED 的 VXVCLS（VIX3M）、VXNCLS、VXEEMCLS、DFII10（10 年期实际国债利率），以及 Cboe 官方 VVIX、VIX9D、SKEW、OVX、GVZ 历史 CSV。VIX9D 取 `CLOSE`，其余指数取同名列，不用开高低价格代替日终指数。Cboe 原始响应保留完整，规范表按请求观测窗口裁剪。

上述波动率指数在实际存在的美国交易会话上使用纽约当地 18:00 的保守历史可用假设，随夏令时转换；不套联邦工作日日历，不为美国休市日补造观测。准确文件上线时刻及当前快照的历史版本未恢复。A 股决策仍以完整交易日历和实际 UTC 决策时刻对齐，因此美国周五、A 股长假等情形不会简单按自然日移一行。新旧代码的时间假设变更需要由调用者显式重建对应源数据。

规范表 `macro_observations` 的键是 `(series_id, observation_date, timestamp)`：

- `observation_date` 是经济观测期；`timestamp` 是该版本可用于计算的时刻，两者不能互换。
- `published_at` 只保存已证实的真实发布时间；未知时保持缺失。
- `vintage_date`、`realtime_end_date` 保留归档修订区间，另存 `retrieved_at`、`source_timezone`、来源和原始单位。
- `availability_basis=release_timestamp` 使用带时区的真实发布时刻。
- 只有发布日期时，`release_date_end` 使用来源当地下一午夜作为保守可用上界，处理夏令时；不伪称知道准确发布时间。
- ALFRED 的 `archive_date_end` 同样使用版本日期的 America/Chicago 下一午夜。后续修订按归档顺序保留；最早版本早于归档收录的历史采用下述显式推断。
- 无历史版本证据的当前快照及延迟归档的首个版本使用 `inferred_snapshot_release`：按观测期与略保守的发布规则推断历史可用时刻，`published_at` 仍留空。获取日期仅为来源事实，不再作为这类历史研究的门槛。

[ALFRED 实时区间下载](https://alfred.stlouisfed.org/help/downloaddata)保留初值及修订。一个查询跨多个下载批次时，相同版本只保留一份；同键上游更正取最近获取版本；跨来源逐序列、逐观测期选择较可信来源。HKMA 当前官方结构化 API 优先旧页面，央行政府转载原公告优先东方财富快照，其他归档修订全部保留。历史缺失值不补零，修订也不会改写已过去决策的输入。

HKMA、BOJ 当座账户、BIS 全球美元信贷、东方财富 M2 和旧金山联储政策冲击的当前下载均不提供本接口已核实的历史版本链，因此按 `inferred_snapshot_release` 导入。BIS 信贷单位为百万美元，BOJ 为亿日元；美国 ON RRP 为十亿美元，Fed 资产与 TGA 为百万美元。

[旧金山联储政策冲击](https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/)采用官方图表 CSV 的 `Surprise`，单位为基点。它是公告附近期货变化的主成分，包含事后估计权重；当前研究近似在事件日收盘后使用现有快照，这不证明 PCA 权重或该数值当时已经冻结。工作簿和图表可能属于不同数值版本，不自动拼接或替换。

上述推断是已声明的研究假设，不能消除当前修订快照或事后估计造成的潜在前视偏差。它不改变真实数值，不伪造历史版本。时间规则由 `sources/macro/publication.py` 统一负责：H.4.1 下一工作日17:30、H.10 下一周一17:15、美国隔夜利率下一工作日、M2次月第四个公布周收盘；ECB周表缓冲到下一周五18:00；中国月报次月20日18:00、香港当日末、日本月末资产次月10日、当座账户次月20日、BIS季度首日后第7个月末。已取得的逐期公告/目录可替换通用估计，但当前修订值配历史发布日期仍标推断。

官方接口的覆盖并不统一。SOFR、IORB 有各自起始日；IOER 已终止。新旧美元指数保持独立；信用利差受发布方当前历史下载范围限制。Shibor 使用 CFETS 官方历史接口，国债与 AAA 中短期票据使用中债官方日终曲线，均按来源的一年查询上限分年请求；社融增量使用商务部转载的央行数据。公开市场操作、社融存量等仍可通过明确的发布文件接入，不能用其他利率代理或补零冒充缺失指标。下载及规范表元数据记录实际覆盖，独立年度请求的成功响应可保留为不完整来源。

精确发布文件使用 `prepare-source` 的 `source_id=macro` 与 `release_csv`。CSV 必须包含 `series_id,observation_date,value,unit,frequency,source_id,source_series,source_timezone,source_url,retrieved_at`，并且只选择 `published_at` 或 `release_date` 其中一列。`published_at` 必须显式含 UTC 偏移；每次修订是独立一行。此入口也可接入有真实事前预期依据的政策惊喜，普通历史创新不能改称政策惊喜。

## 日频决策时钟

`decision_panel` 接收既有日频 `daily_dataset` 和完整 `trading_calendar`，明确设置 `mode`、`local_time` 和 `latency_seconds`。输入原始日频数据不被改写。
下一执行会话必须由输入日历的明确开／休市记录证明，周末和节假日也不能漏掉；
无法证明的信号行不生成决策，其他行继续。输入带 `market_type` 时按该市场匹配；
未带时使用调用方提供的共同 A 股日历作用域，不推断未提供市场的日历事实。
月度／季度同比按上一年同月／同季度匹配，仍只使用截至决策时已发布的版本。

- `close`：交易日已完成日线后决策，本接口要求当地不早于 15:30。
- `preopen`：以信号会话的下一交易日盘前作为实际决策时刻。上一收盘后至这个时刻的已发布信息可以使用。
- 盘前决策加延迟必须严格早于下一会话 09:25 开盘集合竞价结束，不能看到更晚信息再按已经形成的开盘价成交。

新 `timestamp` 是实际 UTC 决策时刻；`signal_session` 保留原日频会话键，`execution_session` 指向下一会话，`order_submit_at` 记录到达时刻，`return_available_at` 保留原收盘收益可用时钟。标签的经济入场、退出及成熟时钟不因重索引而变早。日频账户按 `signal_session` 推进共享股数账户，按真实决策时刻查询预测；原有收盘研究不会静默切换成盘前研究。

## 因子公式与缺失

独立因子族为 `macro_liquidity`，名称以 `ml_` 开头。完整定义来自 `research.factors.macro_liquidity.definitions.RECIPES`，通过现有因子目录查询。计算 owner 为同目录的 `calculations.py`；市场适配只负责输入投影、决策轴和持久化。

基础变换在原始 D/W/M/Q/event 观测频率上计算：水平、一期变化、二阶变化、同比差、滚动 zscore、分位、变化波动、相对历史均值的创新及标准化创新、相对历史 EWMA 的偏离。正值序列额外支持对数、增长率和同比增长；只有投放、到期、净投放及社融增量等流量变量支持窗口求和。利率水平为百分数，利率差及变化转成基点。

默认窗口分别为 60/26/24/12/20 个原频率观测；`minimum_fraction=1` 要求完整有限历史。波动率采用差分的样本标准差，创新只用当前值之前的历史。不同窗口、最少样本、EWMA 半衰期均记录为实际参数。日展开只把已经知道的状态向后带入，不把同一周值重复算成多份样本。缺成分、无方差或历史不足时该因子缺失，不阻塞独立因子。

主要组合包括：

| 组合 | 约定 |
|---|---|
| 美国净流动性 | Fed 资产 − 周三 TGA − ON RRP × 1000，单位百万美元；同期版只配相同观测日期 |
| `ml_us_net_liquidity_latest` | 各成分各自最新已知值的混频组合，独立命名，不能冒称同期资产负债表 |
| 资金与期限利差 | SOFR−IORB、EFFR−IORB/IOER、美债 10Y−2Y/3M、R007−DR007、DR007−7 天政策利率、中国期限及信用利差 |
| 中国净投放 | 同期逆回购投放减到期量；任一成分未知时不计算 |
| 资金压力 | 美国默认四项标准化指标各 1/4；中国两项资金利差各 1/2，权重固定，不随缺失重配 |
| 全球流动性 | 美、欧、日央行资产的历史 zscore 各 1/3；不直接相加不同币种金额 |
| 中美差异、储备/TGA/RRP 比率 | 按明确公式计算，最新已知比率允许混频，不隐含同步观测 |

宏观状态可与 Amihud、波动率、对数市值、Beta、换手率、动量及残差波动相互作用。先在合格股票截面上标准化个股暴露，再乘宏观状态；组装时保留宏观幅度，纯宏观及交互列不再接受股票截面 zscore。普通股票因子的原有处理保持原约定。

VIX 扩展沿用上述变换，并增加：

- `ml_us_vix3m_vix_ratio_*`、`ml_us_vix9d_vix_ratio_*`、`ml_us_vxn_vix_ratio_*`、`ml_us_vxeem_vix_ratio_*`：分子与 VIX 必须属于同一美国观测日。新会话任一成分尚未发布／缺失或 VIX 非正时为空；不会用两天价格混成一个比率。修订只在修订可用之后生效。前两项是不同期限的隐含波动率比率，后两项是不同标的的相对风险，不称期货升贴水。
- `ml_us_vix_implied_variance` 等：`(指数 / 100)^2`，以小数收益的年化方差表示。不同指数对应不同标的，VVIX 是 VIX 的波动率风险，不能当成另一份 S&P 500 方差。没有同一标的、同一期限的实际／预期方差时不生成 VRP；本次未获取 ETF 或美国股价。
- `ml_us_skew_risk_neutral_skewness`：按 Cboe 的 `SKEW = 100 − 10 × S` 还原 `S = (100 − SKEW) / 10`。SKEW 越高，该风险中性偏度越负；不是 A 股已实现偏度，也不是直接的暴跌概率。
- `ml_us_vix_skew_zscore_product`、`ml_us_vix_vvix_zscore_product`、`ml_us_liquidity_vix_zscore_product`：各自最新已知历史 zscore 的乘积。它们用于检验状态调节效应；乘积同为正可来自两项同时高或同时低，分析时必须保留两个主效应，不能把乘积直接命名为危机状态。
- `ml_<state>_x_industry_<exposure>`：先按合格股票截面计算暴露 zscore，再求当时 `(industry_standard_id, industry_level1_code)` 内等权均值，最后乘宏观状态。支持波动率、Amihud、市场 Beta；分类标准或代码未知为空，不从未来分类补回。该列描述行业风险特征，不是行业虚拟变量或行业收益率。

新增交互覆盖 VIX 变化与历史 zscore、VVIX 变化、SKEW zscore、VIX 期限比率、美元变化、实际利率变化。交互单位继承宏观状态，股票／行业标准化暴露没有量纲。VIX、VVIX 和 SKEW 的滚动简单及市场控制 Beta 复用原有流动性 Beta 路径。

滚动流动性 Beta 另需显式 `trading_calendar` 及决策面板的 `market_type`、`signal_session`、
`return_available_at`。窗口按个股所属市场的真实开市会话计数，缺行不压缩窗口，休市日不占位置；
日历必须用明确休市行覆盖周末和节假日；缺日期不能默认为休市，也不能跨未知边界差分。
缺日历事实的信号行及跨未知边界的 Beta 窗口保持未知；边界退出窗口后按原最少观测/变化门槛恢复。
缺失或晚于该行决策的收益时钟不参与配对，不按日期补造收盘时刻，也不跨缺口累计宏观冲击。
缺少必要输入时 Beta 保持缺失，产物的 `macro_beta_missing_inputs` 列明缺项，独立水平和交互列继续产出。
滚动流动性 Beta 支持简单回归及控制市场收益的二元回归；使用滚动协方差/方差充分统计，无逐窗拟合。只有截至决策时已成熟的个股收益进入估计，默认窗口 252、至少 126 个有效观测及 20 次宏观变化。按个股连续已知上市周期估计，重新上市或上市身份未知均截断历史；缺少 `listing_date` 或 `period_return` 时仅 Beta 缺失，独立宏观因子继续输出。零方差、奇异控制变量及不足历史保留缺失。它是历史暴露估计，不是因果效应识别。

## 操作顺序

[配置目录](../configs/macro-liquidity/)提供源规范化、盘前决策、因子准备和组装示例。获取工具独立于仓库，交付 raw 与来源清单后再调用这些操作。路径和窗口由调用者替换；示例不表示对应数据已具备完整 PIT 覆盖。

示例终点为 2026-10-31；实际下载可请求更晚，但月度和季度观测仍有发布滞后。样例 A 股输入路径本身不会因此自动补齐到 2026 年；真实研究前需由调用者准备对应股票面板、历史行业和完整交易日历。纯宏观状态在股票间相同：大盘时间序列分析每个时点只计一个观测，不能把广播到几千只股票的值当成几千份独立宏观样本，也不适合直接用逐日股票截面 RankIC 评价。

```text
qrp prepare-source --config configs/macro-liquidity/prepare-source.json
qrp prepare --config configs/macro-liquidity/decision-panel.json
qrp prepare --config configs/macro-liquidity/factors.json
qrp assemble --config configs/macro-liquidity/assemble.json
```

`assemble` 后继续使用原 `fit`、`predict`、`evaluate`、`calibrate` 和 `backtest`。需要盘前信息的模型必须直接拟合实际决策时间索引的数据，不能把这些值写回前一日收盘行。真实研究的重算范围和输出位置仍由调用者明确决定。

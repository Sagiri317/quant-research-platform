# 全球宏观流动性

宏观源、因子和股票研究输入分别准备。`fetch-macro` 只下载原始响应；`prepare-source` 只规范化源数据；`prepare` 才计算决策面板或因子。各操作都写新目录，不自动训练、回测或更新历史研究产物。

## 来源、版本和时间

`sources.macro.catalog` 映射供应商代码，`contracts.macro.MACRO_VARIABLES` 定义源无关变量、单位和频率。美国数据优先使用 ALFRED 历史版本；其他适配包括 HKMA、BIS、BOJ、旧金山联储和东方财富的公开接口。具体来源 URL、响应原文、获取时间及 SHA256 保存在原始目录。失败来源单独记录，其他来源继续；香港分页中已成功收到的页可独立保留，但明确标记不完整。

规范表 `macro_observations` 的键是 `(series_id, observation_date, timestamp)`：

- `observation_date` 是经济观测期；`timestamp` 是该版本可用于计算的时刻，两者不能互换。
- `published_at` 只保存已证实的真实发布时间；未知时保持缺失。
- `vintage_date`、`realtime_end_date` 保留归档修订区间，另存 `retrieved_at`、`source_timezone`、来源和原始单位。
- `availability_basis=release_timestamp` 使用带时区的真实发布时刻。
- 只有发布日期时，`release_date_end` 使用来源当地下一午夜作为保守可用上界，处理夏令时；不伪称知道准确发布时间。
- ALFRED 的 `archive_date_end` 同样使用版本日期的 America/Chicago 下一午夜。归档首次收录以前的历史值，不倒填成当时可见。
- 无历史版本证据的当前快照使用 `retrieval_only`，只自本次获取时可用。较早的观测日期不意味着可以用于较早回测。

[ALFRED 实时区间下载](https://alfred.stlouisfed.org/help/downloaddata)保留初值及修订。一个查询跨多个下载批次时，相同版本只保留一份；同键冲突值直接失败。历史缺失值不补零，修订也不会改写已过去决策的输入。

HKMA、BOJ 当座账户、BIS 全球美元信贷、东方财富 M2 和旧金山联储政策冲击的当前下载均不提供本接口已核实的历史版本链，因此按 `retrieval_only` 导入。BIS 信贷单位为百万美元，BOJ 为亿日元；美国 ON RRP 为十亿美元，Fed 资产与 TGA 为百万美元。

[旧金山联储政策冲击](https://www.frbsf.org/research-and-insights/data-and-indicators/monetary-policy-surprises/)采用官方图表 CSV 的 `Surprise`，单位为基点。它是公告附近期货变化的主成分，包含事后估计权重；不能把下载到的最新版当成公告当时已冻结的交易信号。工作簿和图表可能属于不同数值版本，不自动拼接或替换。

官方接口的覆盖并不统一。SOFR、IORB 有各自起始日；IOER 已终止。新旧美元指数保持独立；信用利差受发布方当前历史下载范围限制。中国资金利率、公开市场操作、社融和债券曲线可从有授权的官方或供应商发布文件接入，不能用其他利率代理或补零冒充缺失指标。下载及规范表元数据记录实际覆盖。

精确发布文件使用 `prepare-source` 的 `source_id=macro` 与 `release_csv`。CSV 必须包含 `series_id,observation_date,value,unit,frequency,source_id,source_series,source_timezone,source_url,retrieved_at`，并且只选择 `published_at` 或 `release_date` 其中一列。`published_at` 必须显式含 UTC 偏移；每次修订是独立一行。此入口也可接入有真实事前预期依据的政策惊喜，普通历史创新不能改称政策惊喜。

## 日频决策时钟

`decision_panel` 接收既有日频 `daily_dataset` 和完整 `trading_calendar`，明确设置 `mode`、`local_time` 和 `latency_seconds`。输入原始日频数据不被改写。

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

滚动流动性 Beta 支持简单回归及控制市场收益的二元回归；使用滚动协方差/方差充分统计，无逐窗拟合。只有截至决策时已成熟的个股收益进入估计，默认窗口 252、至少 126 个有效观测及 20 次宏观变化。按个股连续已知上市周期估计，重新上市或上市身份未知均截断历史；缺少 `listing_date` 或 `period_return` 时仅 Beta 缺失，独立宏观因子继续输出。零方差、奇异控制变量及不足历史保留缺失。它是历史暴露估计，不是因果效应识别。

## 操作顺序

[配置目录](../configs/macro-liquidity/)提供获取、源规范化、盘前决策、因子准备和组装示例。路径和窗口由调用者替换；示例不表示对应数据已具备完整 PIT 覆盖。

```text
qrp fetch-macro --config configs/macro-liquidity/fetch.json
qrp prepare-source --config configs/macro-liquidity/prepare-source.json
qrp prepare --config configs/macro-liquidity/decision-panel.json
qrp prepare --config configs/macro-liquidity/factors.json
qrp assemble --config configs/macro-liquidity/assemble.json
```

`assemble` 后继续使用原 `fit`、`predict`、`evaluate`、`calibrate` 和 `backtest`。需要盘前信息的模型必须直接拟合实际决策时间索引的数据，不能把这些值写回前一日收盘行。真实研究的重算范围和输出位置仍由调用者明确决定。

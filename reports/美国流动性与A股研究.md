# 美国流动性、波动率与偏度对 A 股的研究

本研究关注美国宏观流动性及 VIX、VVIX、SKEW 状态与 A 股大盘、行业和个股风险暴露的关系。拟研究窗口为 2014 年至 2026 年 10 月；已具备相应来源适配与因子计算能力，当前数据覆盖见下文。**尚未执行正式实证研究，以下均为待检验假说**。

来源契约、公式、缺失规则和操作方法由[宏观技术文档](../docs/macro-liquidity.md)统一说明。本报告保留研究问题、当前数据条件、方法与解释边界，不另设分轮落地或验收报告。

## 数据条件

截至 2026-11-05，现用规范源 `data/interim/macro_liquidity/consolidated-20261105` 包含 **61 个序列、706,901 条版本记录、5,087 个源 NULL 值**，其中 53 个序列有 2026 年有效观测。版本行数包含历史修订，不等于独立样本数；有 2026 年观测也不等于完整覆盖 1—10 月。

| 数据组 | 本地有效观测截止 | 主要覆盖限制 |
|---|---|---|
| Fed 资产、TGA、准备金 | 2026-10-28 | 周频；按实际可用时刻进入决策 |
| ON RRP、SOFR、EFFR、政策利率、美债名义及实际利率、ICE 利差、原 VIX | 2026-11-03—11-05，依序列而定 | SOFR 自 2018 年、IORB 自 2021 年开始；ICE 利差仅自 2023-11-06 开始 |
| NFCI、ANFCI、广义美元、主要汇率、ECB 周资产 | 2026-10-30 | 部分 10 月末观测到 11 月才可用 |
| 香港 HIBOR 隔夜／1 月、体系期初／期末总结余、基本利率 | 2026-11-04 | 2018 年后的缺口已补采；五项历史合计 20,780 条，原 API 两个 NULL 保留 |
| Shibor 1 周／3 月 | 2026-11-05 | 2026 年 1—10 月均有观测 |
| 美国 M2、中国 M2 及同比、社融存量 | 2026-09 | 月度发布有滞后；社融存量历史仅有 2025-09、2025-10 和 2026-01—09 |
| BOJ 资产／当座账户 | 2026-10／2026-09 | 10 月观测不等于 10 月末已知；未来月份 NULL 占位不计覆盖 |
| BIS 全球美元信贷／OECD 中国 3 月银行间利率 | 2026Q2／2026-08 | 各自按原频率及发布滞后使用 |
| 社融增量 | 2026-04 | 商务部返回仅至 4 月，后续央行原表访问受阻；未用累计值相减补成单月值 |

新增九个序列均有 2026 年 1—10 月及部分 11 月观测。下表为本次实际导入范围，多数从 2010 年取数以支持回看窗口，并不宣称是供应商全部可得历史。

| 序列 | 来源／代码 | 有效观测起止日 |
|---|---|---|
| VIX3M | ALFRED／VXVCLS | 2010-01-04—2026-11-03 |
| VXN | ALFRED／VXNCLS | 2010-01-04—2026-11-03 |
| VXEEM | ALFRED／VXEEMCLS | 2011-03-16—2026-11-03 |
| 10 年期实际国债利率 | ALFRED／DFII10 | 2010-01-04—2026-11-03 |
| VVIX | Cboe／VVIX | 2010-01-04—2026-11-04 |
| VIX9D | Cboe／VIX9D | 2011-01-04—2026-11-04 |
| SKEW | Cboe／SKEW | 2010-01-04—2026-11-04 |
| OVX | Cboe／OVX | 2010-01-04—2026-11-04 |
| GVZ | Cboe／GVZ | 2010-01-04—2026-11-04 |

仍完全没有有效来源的六项为 `cn_dr007`、`cn_r007`、`cn_reverse_repo`、`cn_repo_maturity`、`cn_repo_7d_rate`、`cn_treasury_10y_oecd`。中债国债 1／3／10 年和 AAA 中短期票据 3 年只到 2020-12-31，后续年度连接失败。旧金山联储政策冲击只到 2023-12-13；IOER、旧版广义美元指数、IMF 中国 M2 已停更，不与新序列直接拼接。

当前状态对齐会沿用最后已知值；这不能把长期停更或获取失败的来源变成当期数据，尤其不能把 2020 年末中债利率解释为 2026 年利率。正式研究必须按覆盖范围选变量和样本。缺失统一为 NULL/NaN，不加 missing 指示列；缺来源、缺历史或不可定义只影响依赖它的因子，标签不插补。

## 文献依据与本地构造

| 依据 | 本研究采用的启发与区别 |
|---|---|
| [Miranda-Agrippino & Rey，2020，US Monetary Policy and the Global Financial Cycle，REStud](https://www.nber.org/papers/w21722) | 美国政策与全球金融条件共同变化；美元、NFCI、利率和央行资产状态本身不是已识别的外生政策冲击 |
| [Ang、Hodrick、Xing、Zhang，2006，The Cross-Section of Volatility and Expected Returns，JF](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2006.00836.x) | 波动率风险敏感性与横截面收益有关；本研究估计 A 股的历史 VIX Beta，不照搬美国市场结论 |
| [Huang、Schlag、Shaliastovich、Thimme，2019，Volatility-of-Volatility Risk，JFQA](https://safe-frankfurt.de/de/publikationen/working-papers/details/publicationname/volatility-of-volatility-risk.html) | VVIX 补充 VIX 水平；原文的期权收益证据不证明 A 股预测有效 |
| [Bollerslev、Tauchen、Zhou，2009，Expected Stock Returns and Variance Risk Premia，RFS](https://public.econ.duke.edu/~get/wpapers/btz.pdf) | 区分隐含方差和实际方差；当前没有同标的实际方差输入，未构造 VRP |
| [Adrian、Etula、Muir，2014，Financial Intermediaries and the Cross-Section of Asset Returns，JF](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr464.pdf) | 融资约束可能影响风险定价；净流动性与 Amihud／规模／Beta 的交互是本地代理，不等于文中的券商杠杆冲击 |
| [Cboe 期限结构](https://www.cboe.com/tradable-products/vix/term-structure)、[SKEW 白皮书](https://cdn.cboe.com/resources/indices/documents/SKEWwhitepaperjan2011.pdf) | 同期比率和偏度反变换依公开定义；VXN、VXEEM 是不同标的风险，不能解释成同一标的期限结构 |

## 可检验假说

以下方向均需检验。应在看结果前指定主要变量、标签和窗口；反方向、不显著及跨期不稳定同样是有效结论。

### H1：美国流动性的边际变化比绝对规模更能解释 A 股后续变化

主要输入为 `ml_us_net_liquidity_change`、`ml_us_net_liquidity_innovation_zscore`、`ml_us_sofr_iorb_level`、`ml_us_nfci_change`、`ml_us_broad_dollar_change`、`ml_us_real_treasury_10y_change`。待检验方向是净流动性上升、资金压力下降与 A 股风险承受能力改善有关；也允许危机扩表导致流动性增加与下跌同时发生。

大盘考察未来 1／5／20 个交易日收益、已实现波动率和下行风险，行业考察相同期限收益及相对大盘收益。同期净流动性为 Fed 资产−TGA−ON RRP×1000；混频最新已知版 `ml_us_net_liquidity_latest` 独立命名，不混用。

### H2：美国流动性收紧对高风险、低流动性行业的影响更强

使用 `ml_us_net_liquidity_zscore_x_industry_amihud_60d`、`ml_us_funding_pressure_x_industry_beta_252d`、`ml_us_funding_pressure_x_industry_volatility_60d`；同名不带 `industry_` 的交互及对数市值交互研究个股差异。

行业暴露是当时合格股票的标准化风险特征均值，以“分类标准＋行业代码”识别，任一身份未知则为空。它不是行业收益率，也不预先断言某个行业更敏感。出口依赖等机制仍需真实输入，不能把行业代码当成连续特征。

### H3：VIX 上升对 A 股短期风险的解释强于对中期收益方向的解释

输入 VIX 变化、zscore、标准化创新和隐含方差。假说是 VIX 上升与后续波动、下行风险增加有关，但收益可能继续下跌，也可能因风险补偿或反转上升，应分期限检验。

`ml_us_vix_beta`、`ml_us_vix_partial_beta` 为历史风险暴露，后者控制市场收益，不是未来效应的系数真值。`ml_us_vix_change_x_beta_252d` 和 `ml_us_vix_change_x_industry_volatility_60d` 检验冲击是否集中于高 Beta／高波动资产。

### H4：短端隐含波动率相对抬升可能反映更紧迫的风险

使用 `ml_us_vix9d_vix_ratio_level`、`ml_us_vix3m_vix_ratio_level` 及变化，检验前者升高、后者降低时 A 股短期下行风险是否增加。比率只配同一美国观测日，不混用今天分子与昨天分母；它不等于期货展期收益或未来下跌概率。

### H5：SKEW 的关系取决于 VIX 状态

输入 `ml_us_skew_risk_neutral_skewness`、VIX／SKEW zscore 和 `ml_us_vix_skew_zscore_product`。SKEW 高对应更负的风险中性偏度，可能反映尾部保险需求，但不直接证明现实尾部事件将发生。

检验 VIX、SKEW 同时高位是否对下行风险有额外解释，回归同时保留两个主效应。两个 zscore 同时低也会得到正乘积，不能只按乘积正负标记危机。行业使用 `ml_us_skew_zscore_x_industry_beta_252d` 等项比较暴露差异。

### H6：VVIX 可能提供超出 VIX 的风险信息

使用 `ml_us_vvix_change`、`ml_us_vvix_zscore`、`ml_us_vix_vvix_zscore_product`，与仅含 VIX 的基线比较。待检验方向是 VVIX 升高时，A 股风险预测误差及行业分化增大；VVIX² 不解释成 S&P 500 收益方差。

### H7：流动性状态可能改变 VIX 冲击的传导强度

使用净流动性、VIX zscore 及 `ml_us_liquidity_vix_zscore_product`，检验较宽松流动性是否减弱高 VIX 与 A 股下行风险的关系，也允许“宽松由危机内生触发”导致反向结果。该乘积是本地提出的代理，不是文献标准因子的复现。

## 分析单位与识别边界

| 层次 | 一条观测 | 主要结果变量 | 必须避免的错误 |
|---|---|---|---|
| 大盘 | 一个 A 股决策时点 | 明确大盘指数或市场组合的未来收益、方差、下行风险 | 把广播给几千只股票的宏观值当成几千份独立样本；对纯宏观列做每日股票截面 RankIC |
| 行业 | 一个行业、一个决策时点 | 行业收益、相对同口径大盘收益和行业风险 | 用今天行业分类回填历史；把暴露均值当行业收益；行业等权比较中再次按股票数加权 |
| 个股 | 一只股票、一个决策时点 | 未来收益或风险，宏观状态×事前暴露 | 逐日截面标准化消掉宏观幅度；用未来估计的 Beta |

先检验每个假说的少量核心变量，再比较组合模型，不把全部注册变换一次塞入回归。优先控制当时已知的 A 股滞后收益／波动和国内资金状态；控制 VIX 后的条件关系与不控制 VIX 的总相关关系分开解释。

训练和评估沿时间推进，标准化与填补只使用合法训练及过去数据。多日重叠标签保留成熟时钟，并采用适合时间依赖的 HAC 或时间分块重采样；行业面板还要考虑同日共同冲击。报告效应大小、区间、有效日期数、缺失覆盖和跨期稳定性，不只报告 p 值或最佳窗口。

2026 年 1—10 月只有在未参与假说、模型和窗口选择时，才能称独立样本外检验。上述状态回归本身不能识别因果效应；因果分析仍需单独的识别设计。行业收益、大盘标签和实际 A 股面板准备属于后续实证工作。

## 时间假设与证据

已知发布时间保留真实时刻；未知的当前快照及延迟归档初版，沿用已认可的保守历史推断，不以本次获取日限制整段历史。`published_at` 不填猜测值，后续真实修订保留。VIX 系列按实际美国会话在纽约 18:00 推断可用，处理夏令时，再按实际 UTC 决策时刻进入 A 股；没有恢复所有初次发布值和历史快照，不能称严格 PIT。

香港新增 2,156 个有效日已逐页核对数值，但 90 页相关板块日期与页面观测日不一致，其中存在次日或跨多日更新及疑似笔误。当前仍按页面观测日日末使用当前快照，不能将其称为已证实的发布上界；其中较晚更新可能跨过下一次 A 股决策。旧金山联储政策冲击另含事后估计权重，事件日不等于该因子当时已冻结。

最终规范源已完成原生写入读回与全字段比较，独立复核确认六个输入清单、版本唯一性、新增九项原值／单位、香港逐页数值和社融原文一致。关键公式有独立手算，行业交互经过真实 `prepare → 持久化 → assemble` 链路；来源解析、真实零与 NULL、DST、观测日配对、行业身份、归档初版去重及后续修订均有定向验证。

宏观扩展版本 `709ad8e8` 的全量 `pytest -q --tb=short`：5,406 项通过，90 项子断言通过，耗时 59 分 33 秒。`ruff check .`、`ruff format --check .`（471 个文件）和 `git diff --check` 均通过。这是当时版本的验证记录，不是正式实证结论。

长期来源证据保存在数据目录，`reports/` 不依赖旧临时运行日志：

- `data/raw/macro_liquidity/`：原始响应、URL、获取时间和 SHA256。
- `data/raw/macro_liquidity/publication-audit-20261105/`：保留的逐观测期、公告、BOJ／BIS 目录及 FOMC 事件核查表，区分已证实发布时间与采用的历史推断。
- `data/interim/macro_liquidity/consolidated-20261105`：本报告对应的规范源快照。元数据保留来源事实和 raw 证据位置，报告中的行数包含历史修订版本。

来源契约与操作入口见[宏观技术文档](../docs/macro-liquidity.md)。

公开可下载不等于取得统一再分发或商业许可，须遵守 [FRED 条款](https://fred.stlouisfed.org/legal/)、[Cboe 条款](https://www.cboe.com/terms/)及各原始提供方限制。原始数据不随仓库分发。当前输入不含 ETF／美国股票价格，VRP 未构造；正式实证与服务器规模性能仍待单独验证。

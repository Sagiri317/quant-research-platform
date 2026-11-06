# A 股 Level-2 高频研究

数据来自 [Venvoo Hugging Face 数据集](https://huggingface.co/datasets/venvoo/china-a-share-l2-level2-limit-order-book-tick-data)，需要获批账号。先按明确日期下载到本地，再运行独立公开操作；接口不自动下载整库。

## 来源与时钟

`prepare-source` 的 `source_id=venvoo` 读取 `<raw_root>/<YYYYMMDD>/行情.parquet`、`逐笔委托.parquet`、`逐笔成交.parquet`，按显式证券列表过滤并分批写入原生数据集。证券列表分别指定 `wind_code`、项目 `asset_id`、`exchange`（SSE/SZSE）。`instrument_type` 支持 `stock`、`equity_etf` 和 `gold_etf`，省略时为 `stock`，并随原生来源与面板保存，供交易时段划分及账户执行使用。

ETF 示例见 [prepare-source-etf.json](../configs/high-frequency/prepare-source-etf.json)。该归档把上海 ETF 的供应商代码也写作 `.SZ`，示例中的 `510300.SZ`、`510500.SZ`、`518880.SZ` 均显式映射为 `SSE`；`159915.SZ` 映射为 `SZSE`。请按供应商实际代码筛选，并独立声明真实交易所。来源说明和单日实际覆盖均已核对。[数据集说明](https://huggingface.co/datasets/venvoo/china-a-share-l2-level2-limit-order-book-tick-data)

配置记录实际仓库 revision，输出保存所选本地文件 SHA256、证券映射与事件时间语义。证券价格除以 10000 转成 CNY，数量为股；`HHMMSSmmm` 按 Asia/Shanghai 解释后转成 UTC 无时区纳秒时间戳。筛选后源流的顺序编号用于区分同一毫秒的多个事件，不代表未筛选文件的物理行号。

供应商混用的日间 `HHMMSS` 与 `HHMMSSmmm` 编码在来源层统一解析，元数据记录各编码行数。原始 `uint64` 事件索引按可空整数读取，空值保留缺失，避免浮点转换导致索引精度丢失或跨批次类型漂移。

沪市逐笔委托从 2021-07-23 才在此归档中覆盖。早期沪市委托缺失保留 NaN，不视为零。深市成交流的 `C` 和沪市委托流的 `D` 分别视为撤单。深市撤单方向按委托索引解码，只有买方索引为有效正数且卖方为零时记为买方撤单，反向条件记为卖方撤单，缺失、哨兵或双侧索引无法唯一识别时方向保持缺失。正常成交方向使用 `bs_flag` 的 `B` / `S`。委托索引口径见[深交所行情接口规范](https://www.szse.cn/www/marketServices/technicalservice/history/P020180328468172613113.pdf)。

本次核实的清单 revision `5c609748679ac8d3ec056446b7e6bd599bc3dfb5` 有 2382 个日期，2017-01-03 至 2026-10-28；这不是 2014 年起的高频覆盖。日频 2014 起训练与该高频可用区间分开配置。

## 面板、因子和标签

`prepare` 分别生成 `hft_panel`、`hft_factors`、`hft_labels`，逐笔统计使用 `hft_event_factors`，冻结 Hawkes 强度使用 `hft_hawkes_factors`；`assemble` 按显式列组装。原 `a_share_hft` 配方仍只选择原有 18 项因子。配置示例依次为 `prepare-source-hft.json`、`prepare-hft-panel.json`、`prepare-hft-factors.json`、`prepare-hft-labels.json`、`assemble-hft.json`；扩展见 [high-frequency 配置](../configs/high-frequency/)。

面板按证券、日期、连续竞价时段分片。上午 09:30—11:30、下午 13:00 起。深圳连续竞价截止 14:57，上海股票从 2018-08-20 起截止 14:57。上海股票 ETF 和黄金 ETF 在 2026-07-06 之前截止 15:00，此后截止 14:57。终点采用左闭右开区间。午休、隔夜、非法最优盘口或超过 `max_gap_seconds`（默认 6 秒）的间隔会断开短期计算。[上海基金收盘集合竞价生效通知](https://www.sse.com.cn/lawandrules/sselawsrules2025/stocks/exchange/c/c_20260424_10816482.shtml)

原有 18 项因子包含买卖价差、相对价差、一/五/十档不平衡、微价格偏离、OFI 与深度标准化 OFI、3/30/60 秒收益、60 秒波动，以及可观察成交、委托和撤单流。流量区间为 `(前一报价, 当前报价]`。归档外或没有可确认方向的流量不补零；窗口端点不足时保留缺失。

新增快照因子包括五/十档 OFI、深度标准化 OFI、近远档差异、30 秒累计及时间 EWMA，双侧深度 HHI、前三档深度占比、累计深度相对实际价格距离的斜率及变化、双侧斜率差、同价位显示增量的补单代理。斜率单位为 `per_bp`。按档位比较的多档 OFI 和快照补单只是可观察代理，不能声称完整订单流重建。

`hft_event_factors` 直接消费 `hft_panel`、`hft_trades`、`hft_orders`。支持成交间隔、间隔均值/CV、强度与活跃比、买卖连续段及数量、固定笔数和固定成交量窗口失衡、基于纯历史分位的大小单压力、间隔与收益交互、撤单/新增比例、撤单笔数比例与压力变化。成交量窗允许只计入最早一笔的一部分；同毫秒间隔保持零，不抖动时间造出顺序。

订单年龄、撤单年龄、年龄加权压力、完整生命周期及关联比例，要求调用者明确 `order_link_field=order_id` 或 `ex_order_id`；默认 `null` 保留这些列缺失，但其他逐笔统计继续。只对已观察原始限价委托及可靠耗尽计算完整生命周期；沪市成交后剩余委托不冒称完整生命周期。午休、隔夜和缺段重置状态。跨流同毫秒采用成交在前、委托在后的显式约定，各流内部按 `source_row`；这不是共同交易所序号。

标签为 3/30/60 秒的未来 mid 收益与方向，使用同一连续片段内第一个不早于目标期限的实际报价，成熟时间就是该报价时间。不是插值价格，也不把午休或隔夜后的第一笔行情当作短期目标。

`fit-hft.json`、`predict-hft.json`、`evaluate-hft.json` 演示同日上下午隔开的最小链路；它们只是接口示例，不能证明跨日泛化。模型与日频共用 fit/predict/evaluate；高频评价使用 `single_asset_time_series`，每次通过输入 `row_filters` 明确选择一只股票，逐股票运行。

## Hawkes 与 DeepLOB

`fit-hawkes` 显式指定股票、训练窗、截止时刻、初始参数和优化参数，拟合买卖两维共享指数核：`lambda_i(t)=mu_i+sum_j alpha[i,j]*beta*sum(exp(-beta*(t-s_j)))`。时间单位为秒，`alpha` 为分支矩阵，谱半径上限必须小于 1。似然采用连续片段内积分和事件前强度，同毫秒事件共享跳变前历史；未知方向不强行归类，片段初始历史设为零并披露。参数保存至独立 `model.json`；`hft_hawkes_factors` 只能在拟合截止之后输出买/卖强度、失衡和激发比例。计算用线性递推及解析似然梯度，具体似然约定可参照[指数核推导](https://stmorse.github.io/journal/hawkes-LL.html)。

`deeplob_regressor` 和 `deeplob_classifier` 复用已有序列模型的拟合、缩放、保存、加载和预测。结构采用[DeepLOB](https://arxiv.org/abs/1808.03668)的空间卷积、Inception 与 LSTM；本项目时间卷积使用只向过去填充的适配，不声称完整复现论文训练设置。默认序列长 100、卷积通道 32、Inception 通道 64、隐藏层 64，调用者可明确调整。

模型输入必须是十档、每档依次 `ask_px,ask_vol,bid_px,bid_vol` 的固定 40 列；显式 `feature_columns` 选择原始盘口，`feature_recipe=null`。`session_date`、`segment_id` 和 `eligible` 保留为上下文。输入盘口必须完整，缺失会中断序列；跨股票、交易日和连续片段隔离历史。变长片段初始窗口可短于配置长度，不用未来值补齐。

三分类明确设置 `class_threshold`，把小数收益 `<-threshold`、闭区间 `[-threshold,threshold]`、`>threshold` 分别映射为跌/平/涨。训练使用交叉熵，原生 PRED 保存 `probability_down/flat/up`，方向评分只有一份 `prediction_score=probability_up-probability_down`。它不是收益预测；回归 MSE 与收益残差校准不适用。`score`/`select` 支持 log loss、Brier、准确率及固定三类宏平均 F1，分类候选必须使用同一阈值和目标语义。`evaluate` 在成熟且合格的自身样本上另存 `classification_metrics`；`common` 与分层接口仍只接受股票截面因子。

## 库存回测

公司行动与日频共用整数股规则：每个事件产生的股份向下取整，舍尾数量写入权益账本，估值影响由 `equity_rounding_pnl` 披露。总净值已经包含舍尾影响，不重复扣款；该规则是回测近似，见 [日频股份规则](a-share.md)。

`backtest-hft` 接收报价面板、已保存预测、原生费率表、时间窗、交易参数及新输出路径，并通过 `daily_inputs` 显式提供日频行情、证券、交易日历、公司行动和可选的特殊权益事件。来源的项目 `asset_id` 必须与日频证券一致；示例将 `000001.SZ` 的项目标识映射为 `000001`。

`daily_inputs` 服务于持仓估值、上市生命周期、交易日和权益核算。ETF 研究可独立准备这些基础事实。`predictions` 是原生预测目录列表，例如 `["results/hft/day1", "results/hft/day2"]`。各产物使用自身保存的训练截止及预测窗口，账户在预测产物切换时连续结转。调用者负责训练和选择预测产物，同一证券同一时刻存在多份预测时明确报错。`predictions=[]` 运行相同初始持仓与现金的持有基准，比较时保持窗口、估值输入及账户参数一致。

一个回测只有一个多股票账户。`initial_capital` 包含显式 `initial_inventory` 的市值，其余为可用现金；`initial_inventory`、`maximum_inventory` 都按证券指定股数。初始持仓按起始时刻已观察到的 mid 估值，没有更近报价时用当日开盘价。现金、股票、应收股息和权益锁股跨日继承。未持仓股票不需要估值；持仓缺少必要权益或估值事实时明确失败。

日频与高频共用数量规则、费用及权益处理。成交使用整数数量，股票申报的整手、零股和最大单笔数量按板块及生效日期计算。股票 ETF 和黄金 ETF 按 100 份申报、单笔最大 100 万份。默认主动执行卖出先于买入，多证券买入共享剩余现金并按比例缩量。高频账户中，股票与股票 ETF 当日买入的数量次日解锁，黄金 ETF 当日买入可再卖出；权益锁定的数量仍受约束。公司行动沿用日频现有近似，包括登记日及缺失支付日政策；若初始持仓涉及回测开始之前的登记日，须将账户起点提前到登记日。

费率表用 `etf`、`shanghai_etf`、`shenzhen_etf` 声明 ETF 规则，原生参考 CSV 对应 `沪深ETF`、`沪市ETF`、`深市ETF`。主动与被动执行按证券类型选择费率。股票和 ETF 共同回测时，费用表须分别覆盖各自市场。首轮 2023 年费用核验、佣金假设及研究窗口见 [ETF 高频周末准备](../reports/ETF高频周末准备.md)。

决策只能在严格后续、延迟已满且属于同一连续片段的报价执行；买入按 ask、卖出按 bid，数量不超过该侧一档盘口量乘参与率。统一费用包含有效期内的佣金、印花税、过户费和每笔最低佣金；超最大申报量的拆单按实际拆分计佣金。盘口和账户约束不能满足的部分保留未成交原因。

`restore_at=null` 不要求恢复底仓；例如 `restore_at="14:50:00"` 则从北京时间该时刻之后、延迟已满的报价开始尝试恢复当日开盘库存，不使用未来最后一条报价决定恢复时点。T+1、现金或流动性不足时，如实保留未完成库存并记录 `INCOMPLETE`，不会伪造收盘成交。

输出直接是一个原生 `PORTFOLIO_LEDGER`，包含交易、现金流、股数、估值、公司行动和诊断，可以直接 `inspect`、查询净值和绘图。盘中使用最近的有效 mid，完整交易日收盘时使用已公布的日频收盘估值；部分交易日截止于实际窗口内的最后报价。高频目前控制交易与单股库存，不接入行业、风格及跟踪误差限制。

`statistics-portfolio` 可直接读取该账本，示例见 `configs/portfolio-statistics-hft.json`。`performance` 按日末汇总账户：首日从实际账户起点开始，后续从前一交易日收盘接续，计入隔夜持仓收益、盘中成交费用和权益现金流。现金暴露取日末，换手为当日逐次目标请求或实际成交的单边换手金额之和除以期初净值。收益、波动、Sharpe 和回撤复用日频绩效公式；有完整交易日历、首日从开盘开始且所有日末均完整时才按每年 252 个交易日年化。部分交易日或缺日历仍输出实际收益和净值，年化指标标记 `incomplete_daily_axis`。当前高频统计支持 `performance`，不将日内账本用于日频持有区间的风险归因或 ADV 容量计算。

供应商行情约每 3 秒一条，没有接收时钟和完整队列保证。默认执行仍是下一快照的主动成交近似；下述可选被动执行也采用明确的队列假设。预测效果与费用后账户收益分开评估，不能把每个报价当成一个交易日年化。

## 被动委托、信号与执行基准

在 `parameters.passive` 显式设置 `queue_model`（`conservative` 或 `probability`）和 `cancel_latency_ms`，同时提供原生 `trades` 输入。每只股票同时至多挂一张固定价限价单，买单加入买一、卖单加入卖一；相同目标不会反复撤单抢新队列。目标变化先请求撤单，撤单延迟期间仍可部分成交；确认撤单后才重下。策略回放使用同一个账户，`PolicyDecision` 的可用现金和可卖量扣除挂单冻结。

新单按股票顺序使用当前可用现金并冻结完整义务，不预支未成交卖单的收入。申报遵守整手规则，分次成交可以是任意整数股；每张订单累计计佣金，首笔达到最低佣金后，后续只计增量费用。撤单后重下属于新单。费用重于少量卖出收入时，同样预留现金，避免首笔部分卖出形成负余额。

队列初始在当前显示量之后。保守模型只凭符合方向及价格的逐笔成交推进；概率模型把扣除成交消耗之后的净深度减少，按队前量占比均匀分配。这个概率只是取消位置假设，不是经数据识别的真实概率，公式约定参照[队列模型说明](https://hft.readthedocs.io/en/latest/reference/queue_models.html)。每笔记录只消费一次，成交量受剩余数量、越过队前后的量及参与率约束；穿过委托价格的成交可证明价格优先队列已越过，但仍只使用该笔数量。

同毫秒先处理原始成交，再确认撤单及受理新单；新单不能使用同毫秒已发生的成交。只有被同一连续片段内有效快照夹住的成交进入回放。缺段使原挂单按模拟约定终止，保留未完成量；不在未知区间补成交。完整会话末记录当日单失效，盘中截止则保留开单和冻结诊断，不强制成交补足目标。模型不改变历史盘口，依赖无自身市场反馈的小单假设；少成交不代表策略收益的保守下界。

`parameters.signal_processing` 是按顺序执行的显式管线：按秒半衰期 EWMA、按观察数滚动均值/中位数、随机游走 Kalman、阈值、软阈值 `deadband`、迟滞。Kalman 的过程方差单位是信号平方/秒，观测与初始方差为信号平方。每股票、会话及连续片段独立，计算保留执行窗口之前的当日历史预热；缺失滤波值不下目标单，已保存预测不被覆盖。

`backtest-policy` 增加 `almgren_chriss` 和 `signal_ac`，仍共用既有 `TradingEnv` 和股数账户。AC 剩余比例为 `sinh(k*(T-t))/sinh(k*T)`，`k=sqrt(risk_aversion/temporary_impact)*price_volatility`；零风险厌恶退化为 TWAP。风险厌恶单位为 1/CNY，价格波动为 CNY/股/根号秒，临时冲击系数为 CNY·秒/股²；参数 `as_of` 必须早于执行开始。信号版按买卖方向增加 `gain*signal*f*(1-f)`，累计进度不回退；这是明确的启发式调整，不宣称重新求解最优控制。费用仍由账户计入，不把调度参数的理论成本重复收费。参照[Almgren–Chriss 论文](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)。

## 显式预处理、学习诊断与日级解释

序列模型的 `sequence_group_columns` 由调用方声明。高频配置通常传入 `["session_date", "segment_id"]`，日频配置按自己的历史分组。共享模型消费这些边界列及数值输入。每次拟合的窗口、验证窗口、轮数、batch、学习率和设备均来自显式配置。

`normalize-book` 调用 `research.features.book_normalization.normalize_book`。配置明确 `inputs,output,dataset_id,sessions,lookback_sessions`。每个资产当天的 40 列盘口使用之前指定数量完整交易日的有效盘口观察计算总体均值和标准差，开盘后参数固定。某个必要历史日缺失时保持缺失，常数列使用尺度 1。输出单位为 `zscore`，元数据保存每天实际均值、尺度及观察数。后续模型设置 `training_scaling=false`。执行与经济估值读取原始盘口。

`fit.validation` 使用 `windows` 和 `cutoff` 显式指定历史验证样本。模型产物保存每轮学习曲线及实际序列长度统计。正式全历史拟合和历史验证诊断分别调用，训练域按各次真实用途保存。

`evaluate-time-series` 明确预测集合、因子、标签、交易日轴及资产，输出 `daily` 和 `inference` 两个原生数据集。`comparison_groups` 声明共同样本集合，`comparisons` 声明共同域上的左右差值，`inference_groups` 声明多重检验族。自身与共同域均保存每日配对数和输入数。HAC lag、bootstrap 块长、次数、种子、置信度及 FDR 方法来自 `statistics`。缺失交易日保留在统计轴上，合法的日级点估计继续输出。

`analyze-inventory` 明确 `ledger,benchmark,output`，核对同资本、初始持有和账户窗口，从原生账本分解盘中及隔夜库存价格损益、成交相对账本估值的差额、权益股数变化、应收股息变化、股息现金、回收现金和显性费用。输出逐日基准与增量损益、库存暴露、成交等待时间、底仓恢复状态及分解残差。股息到账与应收减少分别列示，合计反映经济收益。

Venvoo 来源入口统一处理已观察到的日间 `HHMMSS` 短编码和 `HHMMSSmmm` 毫秒编码。转换后继续检查时分秒范围，保存到 UTC 时间轴。每个 interim 来源清单记录两类编码的实际行数，修正范围由来源输入决定。

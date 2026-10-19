# 数据布局与来源边界

本文说明数据来源、目录布局、字段含义和信息可用时间。原始数据与访问凭据保存在本地。

## 原始来源

来源默认保存在 data/raw，但接纳函数只消费调用者明确指定的源根或原件集合。
来源家族由供应商/表标识及结构证明，不从目录名猜测，不探测备用根。

### CSMAR

CSMAR 提供 A 股市场、指数、行业和分析师事实；其财务只作验证。

现有 `sources/csmar/field_mappings.py` 声明支持的表和必需字段，
`sources/csmar/interim.py` 统一执行类型、枚举、日期及单位规范化。
供应商说明书中的同名字段只证明该版字段定义；输入还须绑定实际表标识、原件哈希和
适用域，不因此承诺任意导出版本、未知字段或历史完整性。股数、市值、价格、成交量/额
及百分比换算见 [A 股价格、收益与市值真值](a-share.md#55-价格收益与市值真值)。
未知枚举由对应表 owner 按消费语义处理；`corporate_actions.kind` 的保留和诊断见下文字段字典。
`index_info` 的 `Idxinfo05` 至 `Idxinfo09` 是可选分类来源字符串，未知代码原样保留，
不猜测含义、不补为空，也不据此排除整行。manifest 的 `producer_diagnostics.source_row_counts`
披露原始行数、逐字段 `unsupported_<column>_rows`、按行取并集的
`unsupported_source_code_rows` 及保留数；逐字段计数可能重叠，保留数在主键去重前统计。
这些诊断表示现有字典尚未识别，不能单独否定不依赖分类含义的下游用途。
只有实际解释分类代码的消费者才要求相应含义已知；正式基准按本次权重涉及的指数
校验代码与名称，缺少所需身份或身份不符时阻塞相应基准。
可选分类列缺失保留未知；必需来源列、键及身份由来源 owner 校验。
`index_info` 的日期转换只影响该表：空日期保持未知，不可解析或超出时间表示范围的非空日期
转为 `NaT`，按列记录 `invalid_start_date_rows`、`invalid_base_date_rows`；
`9999-12-31` 在此仅作为超出表示范围的日期处理，raw 保留原值；合法日期按既定日期规则解释。
有时序 interim 沿用 `timestamp = start_date`、缺失时取 `base_date` 的规则；
两者均不可用的行仅不进入该时序产物，并记为 `excluded_missing_timestamp_rows`，raw 完整保留。
该计数与窗口排除数分开，保留数表示时间资格筛选后、主键去重前的记录数。
全部记录均无法定时时发布带类型和诊断的空 interim；依赖指数身份的基准计算仍明确报无匹配身份，
不能将空产物视为有效研究结果。
缺失哨兵只按已经核实的表/字段规则处理，不能将某字段的 `-9666` 定义推广到其他列。`index_info.base_point` 的特殊值
规则是该字段的适配约定，其他字段使用各自核实的缺失规则。
财务验证输入不能替代正式来源的首次披露或历史版本证明。
组合会话的已确认退市证据继续使用 CSMAR 股票主表 owner，按 `(asset_id, listing_date)`
标识原上市权益。仅消费当日已生效的真实状态行，且退市日期已到达；未来状态行、
为构建历史 PANEL 补出的生命周期起点行与同代码新上市均不能提前确认旧权益退市
或证明权益衔接。该日期表示来源状态的生效时间，公告时间另需来源记录；公开补充
资料的观察/下载时点与其能证明的历史事实仍须分开保存。
CSMAR 固定投影所需研究字段，不提供可选列 override；合法额外列及长 CSV 记录不受人为额度限制，
读取核验必需字段、重复列及严格 CSV 语法，并记录实际消费字节的来源证据。
原始 reader 只解释结构并保留字符串；业务数值在各来源规范化 owner 转换一次。
可选数值非法时仅该单元格未知并记诊断；缺失必需身份或无法定位时间的行局部不发布，全部如此时输出带类型空表。
财务报表类型、机构与日期范围按实际消费依赖选择，无关行或指标的坏值不阻塞合法事实。
一次 Choice 多表准备复用本次已读取响应，不重复解码相同原件；各表分别记录实际消费原件的来源哈希，CSMAR 诊断按输出表独立构建。
CSMAR 日线与参考表共用逐字段冲突归并：同键规范化事实完全相同只保留一份，
争议字段置为 NULL，独立价格、资产及供应商直接收益仍保留；各完整来源行一致的派生值也保留，
不从互相冲突的字段拼造新值。精确重复数与逐字段冲突行数分别记录，后者按精确去重后的来源行统计。
日线归并在当前选择窗口内累积规范化帧，不假设跨块或跨文件有序；其内存随输入规模增长。
CSMAR 日期筛选范围仍按各表实际合同：行情/复权窗口、市值结束截止；要求完整历史的参考及财务表不随意裁短。
原始 CSV 检查不依赖可选 sidecar 是否存在；必需原件和正式来源身份由来源 owner 校验。
生产 reader 只读取符合标准 CSV 结构的规范输入，严格保留合法引号包裹的多行字段，
不在读取时猜测修复未转义逗号、错位列或跨物理行的损坏记录。历史损坏导出由上游一次性修复，
完成结构、逻辑记录守恒及来源核验后再进入活动 raw 根；质量证据放在活动 raw 根之外。
raw 修复只解决文件无法打开、解压或结构读取错误；可读的缺失、异常日期、未知分类及受损文本
保留来源原值，由生产 owner 按实际依赖解释、诊断或局部筛选，不能在 raw 预先剔除。
例如可正常解码且列数正确的 U+FFFD 文本保留原值，不猜恢复、不据此排除记录。
`index_info` 在类型转换前投影正式字段；合法额外说明文本不进入数值转换。
`Indexcd`、`Idxinfo01`（名称）、`Idxinfo11`、`Idxinfo02` 为必需来源字段；其余规范输出字段
缺失时保留相应可空类型，不从名称推断代码，不让无关额外列改变规范表。


CSMAR `index_membership` 是供应商指数成分事件，包含不同市场及证券类型，不能将所有
成分强制解释为六位 A 股股票代码。`constituent_stock_code` 保留供应商代码字符串；
`constituent_asset_id` 是原码、`market_code`、`security_type` 三项规范 JSON 的内容哈希，
用于区分同指数、同日期的跨市场或证券类型同码事件。缺失市场/类型仍保留缺失事实，
在该来源身份中表示为 null，不补造交易所。它不表示可交易 A 股身份。已有基准消费者
使用的 `index_weights` 仍按其 A 股代码合同处理。
CSMAR `index_weights.Weight` 与 `index_market.Idxtrd08` 的原始缺失值在源规范化表中保留未知，
不补零或删行；非有限值、非法数值与负权重局部转为未知并诊断，百分比只除以 100 一次。两列继续使用
`float64`，缺失表示为 NaN，Parquet 写为 NULL。源表可保留部分事实，不表示完整基准快照；
基准消费者仍要求所选基准权重非空、严格为正且完整，正式全收益绩效仍要求有效开盘序列。

### Choice

Choice 是现有正式财务 PIT 的来源绑定；原生响应先不可变保存，再由已有 owner 离线处理。
Raw 请求是否成功由原始响应 `ErrorCode` 判定；`status` 保存派生诊断，原始请求、响应字节及哈希保留。
资产负债表、利润表和现金流量表分别接纳，只阻塞实际依赖缺失表的因子。
绑定配置按已选因子和组合风险暴露的字段依赖裁剪报表输入；仍要求真实披露与日历，
不造空表、不以零填补缺失财务值。Raw CTR 接受官方 `IndexCode` 选项，
指数权重请求仍须保留查询日期与原始失败响应；服务错误不等于空权重。
[官方接口与指标资料](https://quantapi.eastmoney.com/AIGuide/Install?from=client)
对沪深一般企业财务 CSS 指标明确使用 `Type`：1 为合并报表，2 为母公司报表，
3 为调整后的合并报表，4 为调整后的母公司报表。现有正式消费者只接受 `Type=1/3`。
CTR 财务报表的 `ReportType` 则按 1、2、3、4 分别表示合并、调整合并、母公司、
调整母公司，不能将其数字直接用于 CSS。Raw 原样保留请求；已知财务 CSS 指标若带
`ReportType`，正式来源准备与 interim 消费明确拒绝，不默认为 `Type=1`。

`sources/choice/fields.py` 是三张报表共 14 项一般企业指标的唯一字段映射。
官方指标资料将这些数值列声明为元。按 CNY 消费的依据是境内上市一般企业的
编报财务报表域：[财会〔2006〕3 号](https://kjs.mof.gov.cn/zhengcefabu/200805/t20080522_33653.htm)
规定上市公司自 2007 年起执行企业会计准则，
[《企业会计准则第 19 号》第四、十五条](https://kjs.mof.gov.cn/zt/kjzzss/kuaijizhunzeshishi/200806/t20080618_46229.htm)
要求编报报表折算为人民币，记账本位币可以不同。这里使用报表列报币种，
不能只从“元”推定，也不移用其他接口的币种参数。现有适配器用这些指标、
CSS 类型及境内证券代码限定消费域，不执行汇率换算；境外原币报表、其他报表
类型及未知单位不属于这一合同。该适用域依据不代替逐期快照与版本资格。
`EndDate` 只按官方定义表示截止日期，参数存在或请求成功不证明历史版本可回溯。
`STMTACTDATE` 表示定期报告实际披露日期，`ACTUALISSUINGDATE` 排除招股披露，
`ITEMVALUECHANGEDATE` 表示财报科目数据变更日期；三者分别记录，也不能单独确定
对应数值当日已公开。每个被消费的版本必须由同一请求坐标和不可变原件证明。
日期字符串在现有日期 owner 归一为语义坐标，原始请求与字节身份保留。
日期型已公开信息在下一市场交易日可用；这是项目的保守研究规则，
不是供应商对日内可见性或全部历史修订的保证。规则与修订传播见 [A 股](a-share.md#53-pit-会计与可用时点)。
Choice 普通读取核验所选响应的结构及请求坐标；显式审计与 writer 按来源接纳合同验收。
同一真实版本的冲突仅将争议指标置为未知；修订缺少可证明值时在真实变更时点保留未知屏障，不改写更早已知版本。
`fundamental_pit` 保存 `balance_report_period`、`income_report_period`、
`cashflow_report_period` 三个日期上下文字段（`datetime64[ns]`，未知为 `NaT`）。
它们表示各表最新合法事实的报告截止期，不是公告时点；跨表公式按实际依赖消费。
旧 PIT 用于跨表因子时，先从财务来源重算 PIT，再重算相关因子；
独立或同表公式不要求无关期间字段。
无法定位真实披露时间的报告局部不发布并诊断，其他报告及指标保持可用。
缺少真实版本、含糊的同日冲突或覆盖不足只影响实际依赖该事实的目标；
不能补造旧原件的时间、版本或完成字段，也不等待无关的全供应商历史保证才消费已合格范围。

### Binance

Binance 原生录制 reader 仅消费录制清单声明的 gzip JSONL 成员；
历史归档、辅助状态与资金费按各自显式清单和实际消费的科学关系读取。
不提供独立 JSONL 封装或分钟线适配入口；详细来源合同见 [Crypto](crypto.md)。

### Tardis

Tardis 历史下载与导入按连续切片范围运行，保留 Binance 官方实时采集；
实现就绪状态与输入资格分别记录，不能把下载完成写成研究链已接入。
公开免费数据可直接请求；需要 API key 而没有授权时明确权限不足，不自动注册或购买。
Binance Futures 原生历史归档的不可变原始字节保存在
`data/raw/tardis/binance-futures/<archive-id>/`；清单绑定供应商、
交易场所、UTC 起始日期与连续切片范围、原生频道及频率、文件路径与字节哈希、
稳定行序和可证明的缺口/有效区间。归档和研究均不限制为一天；原件文件名中的日期仅用于寻址。
原生行是供应商 UTC 前缀与 JSON，空行表示供应商断线，因此压缩原件使用
`tardis__binance-futures__YYYY-MM-DD__minutes-HHHH-HHHH.txt.gz`，不称为 JSONL。
原生压缩字节不改写，清单、缺口与观察区间旁文件各有精确哈希。
有结束时间的 Binance 消费按可信 UTC 日期/小时分片排除完整未来成员；所选小时内仍按采集顺序
读取时钟，保留回拨后的范围内记录，包括跨整点回拨重新写入旧小时的记录。
同一流不同小时的采集序号包络可以重叠，读取按实际 `capture_seq` 归并；文件内倒序和实际重复序号仍拒绝。
范围外 payload 与 coverage 不参与本次计算，未知分片路径按现有读取规则处理。
这些哈希属于采集/导入证据；普通回读解析实际消费记录和连续性所需观察区间，
支持提前关闭游标。清单摘要只描述所选来源。
行序表示供应商归档中的位置，与本机录制的原始 capture_seq 分开记录。
归档完整性、价格表示资格、盘口连续覆盖和研究完成是不同检查，不能相互替代。
单次研究默认消费一份有明确来源身份的输入，不自动拼接另一来源补洞。
同一市场的录制与历史归档复用 L2、PANEL、因子、标签、模型和账本责任模块。

Tardis native continuity 保留原字节，包括 depth 重复/过期更新。
显式多归档集合逐原件绑定精确身份，按实际覆盖跨度及请求窗口路由。
原件的本地 source_sequence/event/segment/observation 可重新起算；
统一 source owner 为持久化 observation/segment 标识加入原件身份，并保留每段真实 gap。
按窗口选取的 funding 原件分别验证实际 coverage，不以跨月包络推断未知结算。

下载校验只推进 depth update-id 高水位，旧更新不倒退高水位、不制造新的连续性；
 Binance book engine 负责丢弃不适用的更新。
断连关闭观察区间，后续更新使用新的观察身份；前向 sequence gap 仍形成 gap。
Tardis aggTrade ID 非单调时关闭受影响的成交观察区间，记录
`native_sequence_non_monotonic` gap 并保留原生归档。诊断保存频道、slice offset、
行号和前后 ID，不输出原始价格、数量或完整 payload。
前向 ID 跳号按来源 adapter 合同处理：Tardis 原生导入记录 gap，官方录制 reader
仅验证单调性。[Binance 聚合成交流说明](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/market#aggregate-trade-streams)
定义聚合 ID；该定义本身不证明订阅消息逐 ID 加一；[Tardis 来源说明](https://docs.tardis.dev/historical-data-details/binance-futures)
也不能替代该保证。适配器的检查差异需要结合原始消息核对，现有缺口判定规则保持一致。

来源适用域按 [Tardis 历史频道说明](https://docs.tardis.dev/historical-data-details/binance-futures)
和原件清单中的频道/频率共同决定。只给出变更日期而没有精确时刻的深度、markPrice
过渡日由现有 source owner 拒绝，不能推定在 UTC 零点切换。
[OKX 面值调整公告](https://www.okx.com/zh-hans/help/adjustment-of-face-value-for-usdt-margined-perpetual-swap-futures-trading)
给出的调整半小时区间保持为不确定域；区间外才使用已经证明的合约乘数。
[Tardis 数量说明](https://docs.tardis.dev/faq/data)中的 `amount` 可能是合约数、基础资产或
报价金额，须按场所、品种、时段和实际单位归一；当前接口字段名不能证明全历史单位。
[历史合约 metadata](https://docs.tardis.dev/api/instruments-metadata-api) 的字段和变更记录
也只能在其声明的保证范围内使用，不能把一个字段的历史保证推广到所有字段。
这些限制保留不确定域的局部失败，不否定域外已通过原件、单位、身份及覆盖校验的输入。

Binance Spot 和期货辅助状态的原生归档由配置显式选择，格式和消费边界见 [Crypto](crypto.md)。
Bybit 与 OKX normalized 来源在 `data/raw/tardis/<exchange>/` 下按配置选择归档，
其中 exchange 为 `bybit` 或 `okex-swap`；每份归档保存不可变 `data.csv.gz` 与 `manifest.json`。
清单绑定场所、品种、数据类型、日期、观测范围及压缩原件哈希；只证明观测行，不证明原生消息连续性。

### 规范原始输入与归档

完全重复指字节相同。规范暂存仅保留一份，在保留文件附属元数据记录来源名称与哈希。
同表更新或经核验的修复导出成为新的活动候选，保留覆盖、结构差异及转换来源证据。
活动输入沿用 `data/raw/<provider>/<dataset>/`，无须为修复数据增加配置或备用目录。
无法满足来源 adapter 合同的已知家族标为 KEEP_RAW_ONLY；
只有无法证明表标识时才用 unclassified。两者均不以备用路径进入正式生产。

每个保留 CSV 有一份规范 UTF-8 .metadata.json，记录结构、计数、日期、来源、哈希和质量事实。
它是操作元数据，不是第二数据集真值。原始切换先暂存、读回核验再替换，失败保留旧数据。
已验证且完整保留可读来源记录与字段的结构修复文件可作为唯一活动规范输入。
切换前将旧文件完整归档到活动 raw 根之外，保留其字节、来源和修复证据；归档不参与生产扫描。
只有无法无损恢复结构读取的部分才独立保留原字节并明确报告，不能用业务缺失作为结构隔离理由。
元数据分别记录原件观察日、未知的原生导出日及真实转换日，保存原件、转换脚本与修复文件哈希，
以及足以核验结构修复的证据；不能把未合规的原文叙述标为结构通过。
独立历史费率 CSV 是研究参考输入；其精确原始内容保持，
独立外部来源核验状态与实现是否可消费分别说明，不制造第二份规范副本。

### 公开行业参考来源

公开行业快照属于 `reference` 来源，由现有 reference interim owner 读取
`data/raw/reference/a_share_industry_snapshots/metadata.json`。元数据声明机构对应的官方页面、
标准、观察日、公开日、快照身份、候选 JSON/原始附件/发布页面逻辑文件名及全部来源文件 SHA256；
候选逐行保留资产源代码、公司原名、分类原字段和原件页码或表/行定位；公司显示名及定位是诊断信息，
其缺失或格式异常不否定已有资产、公开时点与分类事实。来源 owner 读取文件并核验
声明哈希，来源证据标识实际消费字节；物化产物保持不可变。
缺少已声明文件、读入字节与声明不符或结构不合法不能以空列表替代，
失败保留已验证旧产物。公开源与 CSMAR 原件分别保留，供应商字段和原始附件保持不变。
Interim 先按知识窗口选择快照，再解释选中记录的业务字段；未来记录的业务异常不污染此前结果。
无法识别资产的记录仅局部排除；可识别资产的分类异常保留该资产时点的未知分类。
精确重复事实合并，重复记录中相互冲突的字段置为 NULL，不任意选值；排除和异常计数保留在产物诊断中。
全部记录不在窗口内或缺少必需资产身份时，仍可发布具有完整类型的空引用表，原始来源内容不变。

`a_share_industry_snapshots` 是 source_id 为 reference 的可选 PANEL 输入。纯价格、量价因子
不强制行业来源；缺少行业证据时分类与非金融标志保留 NULL，不推定为非金融或回填当前行业。
确实消费行业的中性化、适用性或风险计算按所需时点和有效域处理缺失；显式供给的行业证据按
其选中来源、PIT、未知屏障和真实冲突解释。历史取值或标准冲突只使受影响资产的有效行业前沿未知，
从相互冲突的事实均已可用时开始，到更新且可证明的观察状态恢复；不回溯污染此前已知分类，
不回填旧分类或未来修复。词典冲突只影响依赖该词典节点的分类路径。
这些未知状态不阻断独立资产或独立价格、交易事实，依赖行业的后续消费者按自身规则处理 NULL。
Interim 的 observation_date 保留观察期，publication_date 保留页面公开日，timestamp
为日期型公开日之后的次日下界；按角色窗口末端裁剪，保留窗口起点前的历史状态。
PANEL 在真实交易日轴上实施可用日规则。分类词典、历史资产身份、未知屏障与混合标准
消费仍由现有 A 股 owner 统一裁定，
见 [历史行业](a-share.md#543-历史行业)。元数据或候选完整不等于覆盖区间已获研究资格。

### 退市后权益参考事实

`reference/a_share_equity_events/metadata.json` 是现有 reference owner 的可选来源，
仅供组合权益和估值消费，不进入 PANEL、因子、标签、训练或交易许可。
原生 PORT 记录实际引用的来源路径或未提供；不存在来源不证明没有回收，也不制造空源产物。
元数据使用 `a_share_equity_events` schema，保存 `files` 原件 SHA256 和 `events`。
每个事件必须绑定原证券与上市日期、官方 SSE/SZSE/CNINFO/NEEQ 发布 URL、原件和定位、
权益衔接原件与定位、公开日、经济生效日及真实观察时刻。
日期型公开事实只在次日下界之后首个市场会话确认，且经济生效日必须已到达；
观察、下载和历史公开日期分别记录。附件哈希与 URL 只证明来源完整性，具体事实和数量换算
仍须人工核对原件；未知的公告时间、原股消灭或换股关系不可由字段齐全推定。

支持 `quote`、`cash_recovery`、`cancel`、`conversion`、`share_change` 五种明确事实。
人民币金额 `amount_per_share` 按当时实际原股计，转换比例
`successor_shares_per_old_share` 为每原股换得的后继实际股份数。
账本的 `value_per_old_unit`、`remaining_old_units_per_old_unit` 等字段中的 unit
均指实际股份。
实际证券生命周期和经济日期必须匹配。金额与数量比例共用规范精确有理数编码，避免中间溢出丢失最终
可表示的金额；原始 PANEL 行情类型保持不变。每个事件有独立身份；同一旧权益在同一
确认会话内的多个事件需要可验证的先后顺序；仅有事件 ID 时无法确定经济顺序，相关计算停止。
NEEQ 官方发布域包括 `neeq.com.cn` 与 `neeq.cc`，依据[官方信息披露入口](https://www.neeq.com.cn/disclosure/overview_information.html)
及[两网及退市公司信息披露规则](https://www.neeq.com.cn/upload/A0/B0/C1/F190.pdf)；
域名资格不证明特定旧股与转让证券的衔接，仍须逐项原件和单位证据。
单个事件身份只绑定自身声明及实际引用附件；追加未来无关附件改变整份来源身份，
不得改变已知历史事件自身的证明身份。窄窗口按事件真实的首个可用市场会话验证顺序，
不能将窗口前不同会话的事件误作窗口首日同时发生。
退市回收、注销和跨生命周期转换须有当时真实退市及权益衔接证据，
延后才生效的主表状态不能给较早回收补造依据。同生命周期 `share_change`
可表示有正式证据的强制股份变化；不要求先退市。普通 CA/SD 由 corporate_actions
消费，reference 与其重叠的同一经济事件须合并确认一次，不重复分配。
同会话不同权益的转换按已证明的因果关系推进；缺根源或循环衔接拒绝，不以代码顺序猜测。
只有报价不会产生交易许可；现金回收与卖出分开记账，注销不产生现金，同码重新上市不构成权益衔接。
参考事件经原生 reader 进入 `portfolio_sessions`，仍检查实际消费生命周期和单位关系；
可靠关闭，主异常不被清理异常替换。

## 持久化与可消费范围

输入和新输出路径由调用者显式指定，raw/interim/processed/research只是数据责任层，
不要求instances/hash或运行目录。ProcessedDataSpec的来源声明是实际消费字段关系，
允许缺省的来源只在未提供时省略；明确指定坏路径、缺必需列或文件损坏不能fallback为空。

native spec 记录有序列、角色、主键、单位、标签可用时间映射与配方；分区是物理布局，
时间主键不强制按年分区。未声明单位或可用时钟，不等于无量纲或标签已经成熟。
native manifest 记录范围、参数和必要物理信息，读取时检查所需文件和字段。
普通读取只打开指定成员、列和窗口；详见[架构](architecture.md)。
裸 Parquet 的空选择保留已声明的列类型；若文件只有物理字典类型而没有完整分类域声明，
空结果不保证携带全部未使用类别。原生数据集按其逻辑类型声明保留分类域，不为此额外扫描数据页。
正式产物不可变，计算期间不可并发覆盖；发布原子且拒绝覆盖旧结果。
PIT、标签成熟、单位、样本资格和新join覆盖由真实消费者检查，文件存在不证明科学可用。

必须来源未取得只阻塞依赖操作；已有数据问题按影响域处理，无法合法计算的结果如实失败。
NULL、未采集、损坏、缺行、UNKNOWN、真实0不合并，不从NaN猜原因。

## 本地操作与交付

采集入口、计划和 SDK 操作资料保存在本机操作记录中；
本地供应商说明书、字段参考资料和历史采集缺口清单保存在 `data/reference/`，不属于 raw 数据发现根，且不进入 Git；历史缺口清单不构成数据覆盖证明。
代码所需的已核实单位、字段口径及页码写入对应现有市场合同；
CSMAR 股数、市值、价格与成交量/额单位见 [A 股价格、收益与市值真值](a-share.md#55-价格收益与市值真值)。
库存和账户额度留在本机操作记录，不进入长期文档。
研究原始数据仍放 data/raw，不能因整理证据删除有效原始来源。
交付副本按需由仓库文档生成，不维护第二份正文。
SSH、激活信息、账户状态、供应商原始文档/表格均不进入 Git 或网页版交付。

Git 保存代码、方法文档、结构定义、合成测试、配置示例，以及 `reports/` 中的研究报告和汇总图表。
私有原始与生成行数据、供应商载荷、秘密、本地绝对路径及原始回溯不提交。
公开股票代码、提供商标识符和合成测试样例本身不属于秘密；不得借这些标识符提交关联的私有逐行观测。
共享材料使用结构、计数、哈希、状态、逻辑路径和汇总统计；配置、进程与错误记录中仅保留所需字段。

## 独立来源选择补充

CSMAR `prepare_source` 省略起止日期时消费明确输入中的完整历史，不隐式限于2014—2025。
显式日期仍遵守各来源语义：行情与复权按双端窗口；股本保留窗口前状态、仅截末端；
其他状态来源保留已有历史处理，不因接口同名而丢失因果前缀。

reference 可以在一个请求中明确多个 outputs，各表独立发布与报告缺源，不提供跨表事务。
非空 start_date 明确拒绝，避免假装已经裁剪历史；end_date 只限制行业/权益的知识时点，
费率有效区间保留。需要选择其他合法费率文件时，成对给出 `fee_csv` 与 `fee_sidecar`，
路径直接解释；省略时使用参考目录默认。内容、哈希、字节数、
行数、单位与覆盖证明仍由来源 owner 检查，改文件名不能制造新年代的合法覆盖。
CSMAR、reference 与资金费率导入按实际消费字节记录来源 SHA256；reference 核对
声明哈希、字节数与行数。输出按原子发布合同验收。

产物根可以命名为 data，实际结构及来源声明决定其含义。单独的数据子目录缺少必需结构
仍不是完整产物。schema inventory 诊断 data/{interim,processed,research} 布局；
独立计算按实际选择的输入结构读取。

## 字段字典与阶段输入输出

本节是通用字段语义及 schema 的唯一导航，不另列一份会漂移的全仓列清单。
研究合同导航见 [研究合同索引](specifications.md)；以下链接定位字段及其计算 owner。

### 通用结构、主键和时钟

- `DataSpec` 的 `data_id`、`frequency`、`grain` 描述数据表含义；`column_roles` 保存有序列及角色，`primary_key` 是真实主键，`partition_columns` 仅为物理布局。`column_units` 显式声明单位，`label_availability_columns` 将每个标签绑定其成熟列。完整定义见 [contracts/data.py](../src/quant_research_platform/contracts/data.py)。
- `market_id`、`asset_id`、`timestamp` 是平台资产时间键词汇；具体表可能只使用合同声明的子集，不能盲目给所有表添加同一主键。`source_asset_id` 是供应商代码，不能仅凭字符串直接当跨市场/跨生命周期资产身份。角色全集与键校验见 [contracts/columns.py](../src/quant_research_platform/contracts/columns.py)。
- `timestamp` 必须结合表的经济含义解释：可能是日频会话坐标、事件时间或来源除权日期，不能统一当作信息可用时间。公告、经济生效、观察/下载、本地接收和标签成熟是不同坐标。具体时区和精度以来源/市场合同为准；有时区时刻规范为 UTC，不用运行机器时区猜测无时区字段。
- `label_available_at` 或合同映射的 `<label>_available_at` 是标签可用于拟合/评分的真实下界。标签 horizon 的单位由 `horizon_unit` 声明，目前为 `market_sessions` 或 `milliseconds`；`lookahead`、`entry_rule`、`exit_rule` 和 `label_timing` 不能由列名猜测。见 [contracts/labels.py](../src/quant_research_platform/contracts/labels.py) 和 [contracts/experiments.py](../src/quant_research_platform/contracts/experiments.py)。
- A 股日频标签经济开盘事件与 EOD 复权成熟时点分别保存；退出会话 UTC 07:00 是当前现代交易时钟的收盘下界，不能写成开盘即可用。Crypto 本地可见、交易所事件和资金费结算时钟由 [Crypto 合同](crypto.md) 定义。
- `eligible` 描述所声明研究选择，`can_enter`/`can_exit` 描述相应执行条件；真实观测、缺失伴随列描述数据事实。它们互不替代，不用交易许可来证明价格标签不存在。完整 PANEL/因子/标签列见 [A 股 catalog](../src/quant_research_platform/markets/implementations/a_share/processed/catalog.py)。

### 单位与缺失

价格、数量、市值、金额和收益不能仅以 `float64` 或相同字段名判为同单位。
A 股 raw 价格为原股价格、数量为真实股数；复权价格及复权账本单位属于另一经济坐标。
总市值、流通市值、流通股数分别消费，log 变换不改变其来源口径。具体来源倍率见
[A 股价格、收益与市值真值](a-share.md#55-价格收益与市值真值)。
Crypto 数量须证明基础资产/合约数/报价金额及乘数；资金费率与已入账资金费金额不同。
收益以比例表达时 0.01 表示 1%，不得二次百分比换算。

NULL/NaN 只表示该字段没有数值，不能单独区分未采集、源缺失、结构不可计算、损坏、缺行、未知或不适用。
真实 0 不等于缺失，未知也不等于 False；来源诊断、状态和观测伴随列保存原因。
标签不可插补；模型缺失处理按 [拟合与预处理合同](a-share.md#514-正式模型层级与预处理)在合法训练域估计并冻结。`feature_`、`feature_missing_`、`feature_observed_`
的现有输出关系由 [FeatureTransformSpec](../src/quant_research_platform/contracts/features.py) 定义，
特征支持列不自动成为模型 X，模型列序由拟合合同保存。canonical 特征以 missing/observed 两位区分普通缺失与结构不可用，
模型按实际消费的完整列族判断样本资格，数值 NULL 本身不足以判断。

### CSMAR equity_changes：报表来源与科目粒度

`FN_FN046` 的同一证券、报告期、报表类型可以有多个来源和科目。
INTERIM 主键为 `source_asset_id, timestamp, report_type, data_source, subject_code`，
后两列分别原样映射 `DataSources` 和 `SubjectCode`，按字符串保留前导零。
主键须保留来源与科目，不能合并不同权益变动。
报告期不是公告可用时点；这些身份字段本身不证明每股口径或公司行动已经对齐。

### CSMAR corporate_actions：规范事实与结算未知状态

唯一来源表为 `TRDNEW_Cptl`，已有 source owner 直接产生 INTERIM `corporate_actions`。
来源映射见 [field_mappings.py](../src/quant_research_platform/sources/csmar/field_mappings.py)，
类型、保留与诊断见 [interim.py](../src/quant_research_platform/sources/csmar/interim.py) 的
`CORPORATE_ACTION_COLUMNS` / `_normalize_corporate_actions`。

| 规范字段 | 来源字段 | 含义、类型及边界 |
| --- | --- | --- |
| source_asset_id | Stkcd | 来源证券代码字符串；生命周期由消费者另核 |
| timestamp | Exdistdt | 规范化除权日期；不是公告或到账时间 |
| kind | Disttyp | 来源事件代码字符串；未知代码原样保留并计数 |
| market_type | Markettype | 可空整数市场代码；不是自动可交易资产证明 |
| announcement_date | Annodt | 可空日期；未知/不合法保留缺失并诊断，不用下载日代替 |
| settlement_date | Paydt | 可空支付/结算日期；在适用事件中核其真实经济含义 |
| settlement_date_unresolved | 派生 | 非空非法或同事件冲突付款日期为 True；与真实空白区分 |
| amount | Amount | 可空数值，单位依事件 kind：现金每原股金额或每原股新增股份等，不能跨 kind 混用 |
| subscription_price | Roprc | 可空认购价格；保留来源事实不等于策略执行认购 |

来源层对可空字段的非法值记录诊断并保留未知；除权日期和事件标识仍必须合法。
该表没有供应商登记日字段。普通 CA/SD 的登记时点按研究约定由自身市场日历推导；
付款日期真实空白时除权日开盘到账是明确研究假设；非空非法或冲突日期不能采用该假设。
同一经济事件争议字段保留未知与诊断，共同事实保留；未知金额不能当成零，实际依赖者权益无法确认。未知事件是否阻塞由实际持有人权益依赖裁定，
不是在 source 层删除整段来源。

原股账本直接消费该表；内部事件投影的 `record_at`、`effective_at`、`settlement_at`、
`available_at`、`cash_per_share`、`additional_shares_per_share`、`entitlement_proved`
由 [portfolio_sessions.py](../src/quant_research_platform/markets/implementations/a_share/portfolio_sessions.py)
定义，为原股账本的计算输入。
原股账本的价格、股数与分配须使用同一原股单位；退市后参考权益须按实际原股及生命周期解释。

### 输入选择器：只解释本次明确选择

支持共同数据集选择器的入口接受路径，或含 `path` 的对象。可选字段为 `spec`、`files`、
`columns`、`years`、`time_window`、`row_filters`，以 [现有 reader](../src/quant_research_platform/storage/datasets/reading.py)
的 `_dataset_input_options` 和各操作入口为准，不保证所有独立参数都接受此对象。

- `files` 选择成员，`columns` 保持明确列序，`years` 为离散年份集合；同时给年份与窗口取交集。
- `time_window` 由 [TimeWindow](../src/quant_research_platform/contracts/time.py) 定义；日频结束日期含全天，不能因此改写 Crypto 半开观察区间。
- `row_filters` 是现有 `==`/`in` 有限物理筛选，不是任意表达式引擎，也不证明样本科学资格。
- `spec` 为裸 Parquet 补足本操作真实需要的角色、键、单位和标签成熟映射，不能用声明替代来源证据。
- 选择输出窗口不缩短公式回看、标签端点或完整截面。排除的报价保持未知，不压紧交易会话轴。
- PK 和声明的科学伴随列由消费 owner 补齐；明确排除计算必需业务值（如分红金额）时，该依赖操作报出缺少的列。
- 有界计算只验证必要输入范围，保留回看历史与前瞻端点，范围以后的无关行不参与本次消费。

### 阶段输入、输出与唯一合同位置

| 独立操作 | 明确输入及主要选择 | 输出与 schema / owner |
| --- | --- | --- |
| prepare-source | 指定来源根/原件与表，按 source 合同选择日期 | INTERIM 规范表；[sources/prepare.py](../src/quant_research_platform/sources/prepare.py)，各来源映射及 reader |
| prepare | 已准备来源、市场、数据集及窗口 | PANEL/FACTOR/LABEL；[contracts/data.py](../src/quant_research_platform/contracts/data.py)、市场 catalog；不是自动补齐缺源 |
| assemble | 明确基表、因子批次/已有特征、标签、列投影及特征配方 | 可复用研究数据集；[ResearchDatasetSpec](../src/quant_research_platform/contracts/datasets.py)，保留列角色、成熟映射和 join 覆盖 |
| fit | 数据、列序、目标、拟合窗/截止、算法参数及 seed | 冻结 predictor 与 model.json/trial；[contracts/models.py](../src/quant_research_platform/contracts/models.py)、[production.py](../src/quant_research_platform/research/models/production.py) |
| predict | 明确已存模型及新特征窗口 | PRED 键、prediction_score、prediction_rank_score；[prediction.py](../src/quant_research_platform/artifacts/outputs/prediction.py)，不附带未来标签、不重新拟合 |
| score / select | 有序完整候选、验证窗与成熟截止；选择另声明规则和最终拟合窗 | 各自 JSON；[trial_selection.py](../src/quant_research_platform/research/models/trial_selection.py)，最终重新拟合通过显式 fit 执行 |
| explain | 冻结模型、明确样本、方法和合法背景 | 独立解释数值与元数据；[explanations.py](../src/quant_research_platform/artifacts/explanations.py)，绘图无需加载模型 |
| evaluate | 明确信号、标签、窗口、成熟截止、比较/分层选择 | EVAL 日表和摘要；[contracts/evaluations.py](../src/quant_research_platform/contracts/evaluations.py)、[evaluation.py](../src/quant_research_platform/artifacts/outputs/evaluation.py) |
| calibrate / backtest | 显式预测、账户/执行/来源、历史校准输入与回测窗口 | 校准产物或 PORT；[contracts/portfolios.py](../src/quant_research_platform/contracts/portfolios.py)、[portfolio_ledger.py](../src/quant_research_platform/artifacts/portfolio_ledger.py) |
| Crypto calibrate / backtest | 明确 episode、盘口/事件、信号、执行参数、实际依赖资金费 | 独立 episode 及同一经济账本；[crypto_hft.py](../src/quant_research_platform/artifacts/crypto_hft.py)、[Crypto 合同](crypto.md) |
| statistics-* | 已算模型/因子/组合数值、明确比较集合与 analyses | 统计表及推断状态；[statistics.py](../src/quant_research_platform/artifacts/statistics.py)，未选择分析不要求其专用输入 |
| plot | 已发布数值、series、明确用途与新图片路径 | PNG/SVG/PDF；[plots.py](../src/quant_research_platform/reports/plots.py)，不补数、不代算正式指标 |

PORT 的实际表列由 `PORTFOLIO_LEDGER_TABLE_COLUMNS` 从合同类型生成，包含 trades、cashflows、
valuations、diagnostics、rights_adjustments；不在本文手写第二套完整列。
`quantity`/`locked_quantity` 是对应账户单位数量，`receivable` 是尚未到账应收，
`cash_amount`/`amount`/`value` 为合同规定币种金额；精确数量编码与普通浮点金额分别处理。
具体现金、权益、应收和锁定状态的守恒由同一账本合同核验，不以表存在或字段非空代替。
原股账户的相邻 `end` 与 `before` 必须保持全账户财富及每项原股估值连续，金额使用
NAV 量纲的浮点舍入容差；精确股数、生命周期、应收和锁定状态仍遵循各自原合同。
即使总 NAV 相同，也不能在边界静默更换各股票的估值或股数。该不变量由原生 PORT
写入与读取共用的 transition owner 核验，统计消费者不维护另一份财富连续性规则。


### 模型预测描述指标

`model_statistics` 的 `model_losses` 分析同时发布 `model_prediction_metrics.parquet`，
主键为 `timestamp, family_id`，每行是一家族在一个实际输入日期的描述统计。
字段/schema 唯一 owner 为 [statistics.py](../src/quant_research_platform/artifacts/statistics.py)。

| 字段 | 单位与定义 |
| --- | --- |
| prediction_count / target_count | 有限预测数 / 预测与成熟主目标成对有效数 |
| mse / mse_reason / baseline_mse | 当日成对域等权平方误差及未定义原因 / 所选预测基准的平方误差，单位为主目标单位的平方 |
| oos_r2 / oos_r2_reason | `1−mse/baseline_mse`；零基准误差、无目标或不可表示结果为 NULL 并说明原因 |
| head_selected_count / head_evaluable_count | 仅由预测确定的头部人数 / 固定头部中原始收益已成熟人数 |
| head_coverage | 可评价人数除以选中人数；无选中人为 NULL，不以覆盖硬门槛过滤 |
| pool_evaluable_count | 同日 eligible 研究池中原始收益已成熟人数，不要求该资产有预测 |
| head_mean_raw_return / pool_mean_raw_return | 固定头部 / 同日研究池的等权原始 horizon 收益，比例单位、未扣成本 |
| head_excess_raw_return | 以上两均值之差；任一不可评价则 NULL |

`evaluation.head_fraction` 默认 0.2，可显式覆盖；先按 `ceil(N×fraction)` 确定预测阈值，
边界并列全部入组，再检查目标/收益可用性。`raw_return_label` 可显式指定，默认只在输入存在
`forward_return_20d` 时使用该列，否则收益指标未定义；标准化目标与原始收益使用不同单位。
元数据 `prediction_metrics` 保存解析后的头部比例、原始收益列和聚合口径。
MSE 使用标签尺度预测：rank 模型使用各次训练冻结的 `label_calibration`，普通回归输出不重复缩放；原始排序分数继续用于 IC 和头部选择。
训练从实际消费的 spec/manifest 冻结 `facts.target_definition`：源标签、目标 transform、clip 以及所选源标签的 `label_semantics`；PRED 的 `model_facts` 原样携带。
普通标签记录源列本身、不追加变换；`model_target` 记录 `forward_return_20d` 及实际 `target_transform`/`target_clip`。
不绑定路径、data ID、hash 或整份 spec，不在后来从训练源回读补造历史事实。
score 的显式 `label_unit` 可补充标签输入未声明的单位；若输入已有单位，两者仍须一致。该声明保存在评分/选择记录中，不从训练单位猜测评价单位。
只有目标定义和两侧明确单位相同，才计算 MSE 和 OOS R²；`model_target` 的 raw/none 与同源原始标签可比较。
目标不同、单位缺失或不同分别记为 `prediction_target_mismatch`、`target_unit_missing`、
`prediction_target_unit_mismatch`；冻结定义未知、对侧缺必需定义或定义不同分别记
`prediction_target_definition_missing`、`prediction_target_definition_mismatch`。MSE/R² 设为 NULL，
训练均值基准误差也为 NULL；零预测基准误差仍按评价目标计算。目标计数、RankIC 和头部描述按各自条件保留。
普通声明标签两侧均未提供额外 `label_semantics` 时按同源列定义比较；一侧有额外定义而另一侧无则未定义。
`mse_reason` 区分 `no_valid_targets` 与 `nonfinite_result`，有效损失保留标签的平方单位。
跨日 MSE 先算每日均方误差再日期等权；跨日 OOS R² 为同一可评价日期域上
`1−mean(mse)/mean(baseline_mse)`，不平均每日 R²，不按资产行数加权。
这些重叠 horizon 收益不连乘为日净值。

`evaluation.baseline` 为 `zero_prediction`（默认）或 `training_mean`。后者使用各 PRED
携带的 `training_target_mean`，由对应实际 fit/refit 的最终合法目标及训练权重估计，
包含显式时间衰减并随模型冻结。缺少冻结均值时明确失败，不从评价目标估计或回读训练源。
`prediction_metrics` 保存实际 baseline；训练均值模式的 baseline_weighting 为
`frozen_model_training_weighting`。每日及跨日误差使用同一成对样本与日期等权口径。

行业版本的 `abandon_date_unresolved` 区分真实空白（开放有效期）与非空非法或冲突终止日期。
冲突的已知终止日期仅在共同可确定的较早区间保持有效，之后该标准的行业归属未知；
未来版本的不确定性不回溯改变较早已确定区间，也不抹掉独立价格或其他标准事实。

股票生命周期非空非法退市日期保留 `delisting_date_unresolved`，不当成开放有效期，
也不删除后续状态后沿用更早状态。可定位状态时点的未知生命周期仅使依赖生命周期的事实未知；
已观测价格和独立资产保留，未知区间不凭空补出停牌观测。

分析师预期只转换实际年度/时点范围内的 EPS。已识别机构的后续 EPS 未知报告保留为未知屏障，
不能删行后复活旧预测；有报告身份但机构归属缺失或冲突时，仅该资产/预测期的机构统计未知，
直到同一报告被明确定位或新鲜度窗口届满。独立资产、其他预测期及不消费 PE 的指标不受影响。

同一 ReportID 的可见新版本仅替换该报告，机构归属修正不删除原机构的其他报告。
机构中位数在报告修订及新鲜度筛选之后，按该机构最新声明日的报告计算；
来源未提供 ReportID 时保留独立报告行，不推断跨日报告身份。

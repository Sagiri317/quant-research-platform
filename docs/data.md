# 数据布局与来源边界

本文件只记录结构与规则，不保存私有行、原始供应商载荷、凭据或本机路径。

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
未知枚举由对应表 owner 按消费语义处理；`corporate_actions.kind` 的保留和诊断见下文字段字典。`index_info` 仅保留 `Idxinfo05` 至 `Idxinfo09` 的
非空分类代码均通过既有字典的记录，不猜测映射、不把未知代码补为空。
原生 reader 保留完整记录；interim 规范化 owner 在原有结构、类型与日期校验后筛选，
通过 manifest 的 `producer_diagnostics.source_row_counts` 披露原始行数、逐列不支持行数、
按行取并集的排除数及保留数；逐列计数可能重叠，保留数在主键去重前统计。
全部记录被排除时明确失败，错误仅披露原始、排除及保留行数，不发布无有效记录的产物。
缺少必需指数仍由原消费者阻塞；主表完成不证明正式基准资格或历史完整性。
可选分类列缺失仍保留未知。
缺失哨兵只按已经核实的表/字段规则处理，不能将某字段的 `-9666` 定义推广到其他列。`index_info.base_point` 的既有特殊值
规则属于项目明确的适配约定，不能据此宣称供应商全部字段使用同一缺失值。
财务验证输入不能替代正式来源的首次披露或历史版本证明。
组合会话的已确认退市证据继续使用 CSMAR 股票主表 owner，按 `(asset_id, listing_date)`
标识原上市权益。仅消费当日已生效的真实状态行，且退市日期已到达；未来状态行、
为构建历史 PANEL 补出的生命周期起点行与同代码新上市均不能提前确认旧权益退市
或证明权益衔接。该状态日期是来源状态生效坐标，不冒称另有公告时间证明；公开补充
资料的观察/下载时点与其能证明的历史事实仍须分开保存。
CSMAR 固定投影所需研究字段，不提供可选列 override；合法额外列及长 CSV 记录不受人为额度限制，
必需字段、重复列、严格 CSV 语法、源内容身份与读取期间变更校验仍保留。
来源只校验本次规范输出实际消费的数值列；不落盘且不参与计算的可选列不做无关数值转换。
结构、必需字段及用于选择的日期/键仍先校验；能明确定位窗口外的行先筛选，再校验所选行的业务数值。
窗口外坏值不能阻塞合法窗口，但无法合法确定所属日期的结构错误不能假装已被筛除。
一次 Choice 多表准备复用本次已读取响应，不重复解码相同原件；各表分别记录实际消费原件的来源哈希，CSMAR 诊断按输出表独立构建。
CSMAR 日期筛选范围仍按各表实际合同：行情/复权窗口、市值结束截止；要求完整历史的参考及财务表不随意裁短。
原始 CSV 检查不依赖可选 sidecar 是否存在；供应商必需原件和正式来源身份校验保持不变。
`index_info` 在修复原始多行记录后、类型转换前投影正式字段；合法额外说明文本不进入数值转换。
`Indexcd`、`Idxinfo01`（名称）、`Idxinfo11`、`Idxinfo02` 为必需来源字段；其余规范输出字段
缺失时保留相应可空类型，不从名称推断代码，不让无关额外列改变规范表。


CSMAR `index_membership` 是供应商指数成分事件，包含不同市场及证券类型，不能将所有
成分强制解释为六位 A 股股票代码。`constituent_stock_code` 保留供应商代码字符串；
`constituent_asset_id` 是原码、`market_code`、`security_type` 三项规范 JSON 的内容哈希，
用于区分同指数、同日期的跨市场或证券类型同码事件。缺失市场/类型仍保留缺失事实，
在该来源身份中表示为 null，不补造交易所。它不表示可交易 A 股身份。已有基准消费者
使用的 `index_weights` 仍按其 A 股代码合同处理。
CSMAR `index_weights.Weight` 与 `index_market.Idxtrd08` 的原始缺失值在源规范化表中保留未知，
不补零或删行；非缺失值仍须有限，权重不得为负，百分比只除以 100 一次。两列继续使用
`float64`，缺失表示为 NaN，Parquet 写为 NULL。源表可保留部分事实，不表示完整基准快照；
基准消费者仍要求所选基准权重非空、严格为正且完整，正式全收益绩效仍要求有效开盘序列。

### Choice

Choice 是现有正式财务 PIT 的来源绑定；原生响应先不可变保存，再由已有 owner 离线处理。
Raw 响应状态以 `status` 为唯一事实，不再持久化重复布尔状态；原始请求、响应字节及哈希保留。
资产负债表、利润表和现金流量表分别接纳，只阻塞实际依赖缺失表的因子。
绑定配置按已选因子和组合风险暴露的字段依赖裁剪报表输入；仍要求真实披露与日历，
不造空表、不以零填补缺失财务值。Raw CTR 接受官方 `IndexCode` 选项，
指数权重请求仍须保留查询日期与原始失败响应；服务错误不等于空权重。
[官方接口与指标资料](https://quantapi.eastmoney.com/AIGuide/Install?from=client)
对沪深一般企业财务 CSS 指标明确使用 `Type`：1 为合并报表，2 为母公司报表，
3 为调整后的合并报表，4 为调整后的母公司报表。现有正式消费者只接受 `Type=1/3`。
CTR 财务报表的 `ReportType` 则按 1、2、3、4 分别表示合并、调整合并、母公司、
调整母公司，不能将其数字直接用于 CSS。Raw 原样保留请求；已知财务 CSS 指标若带
`ReportType`，正式 source identity 与 interim 消费明确拒绝，不默认为 `Type=1`，
不增加 CTR 到 CSS 的兼容映射。

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
`ITEMVALUECHANGEDATE` 表示财报科目数据变更日期；三者不相互冒充，也不单独证明
对应数值当日已公开。每个被消费的版本必须由同一请求坐标和不可变原件证明。
日期字符串在现有日期 owner 归一为语义坐标，原始请求与字节身份保留。
日期型已公开信息在下一市场交易日可用；这是项目的保守研究规则，
不是供应商对日内可见性或全部历史修订的保证。规则与修订传播见 [A 股](a-share.md#53-pit-会计与可用时点)。
缺少真实版本、含糊的同日冲突或覆盖不足只阻塞依赖该来源的目标；
不能补造旧原件的时间、版本或完成字段，也不等待无关的全供应商历史保证才消费已合格范围。

### Binance

Binance 原生录制 reader 仅消费录制清单声明的 gzip JSONL 成员；
历史归档、辅助状态与资金费按各自显式清单和身份绑定。不提供独立 JSONL 封装或分钟线适配入口；详细来源合同见 [Crypto](crypto.md)。

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
行序属于归档身份，不能冒充本机录制的原始 capture_seq。
归档完整性、价格表示资格、盘口连续覆盖和研究完成是不同检查，不能相互替代。
单次研究默认消费一份有明确来源身份的输入，不自动拼接另一来源补洞。
同一市场的录制与历史归档复用既有 L2、PANEL、因子、标签、模型和账本责任模块。

Tardis native continuity 保留原字节，包括 depth 重复/过期更新。
显式多归档集合逐原件绑定精确身份，按实际覆盖跨度及请求窗口路由。
原件的本地 source_sequence/event/segment/observation 可重新起算；
统一 source owner 为持久化 observation/segment 标识加入原件身份，并保留每段真实 gap。
按窗口选取的 funding 原件分别验证实际 coverage，不以跨月包络推断未知结算。

下载校验只推进 depth update-id 高水位，旧更新不倒退高水位、不制造新的连续性；
既有 Binance book engine 负责丢弃不适用的更新。
断连关闭观察区间，后续更新使用新的观察身份；前向 sequence gap 仍形成 gap。
aggTrade ID 非单调仍明确失败，并保存频道、slice offset、行号和前后 ID 的诊断注记，
不输出原始价格、数量或完整 payload。
前向 ID 跳号按来源 adapter 合同处理：Tardis 原生导入记录 gap，官方录制 reader
仅验证单调性。[Binance 聚合成交流说明](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/ws-streams/market#aggregate-trade-streams)
定义聚合 ID；该定义本身不证明订阅消息逐 ID 加一；[Tardis 来源说明](https://docs.tardis.dev/historical-data-details/binance-futures)
也不能替代该保证。adapter 严格度差异不构成已证实漏消息，也不授权自动放宽 gap 合同。

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
同表更新导出成为新的活动候选，保留覆盖/结构差异证据，不为修复导出改写原始字节。
无法满足来源 adapter 合同的已知家族标为 KEEP_RAW_ONLY；
只有无法证明表标识时才用 unclassified。两者均不以备用路径进入正式生产。

每个保留 CSV 有一份规范 UTF-8 .metadata.json，记录结构、计数、日期、来源、哈希和质量事实。
它是操作元数据，不是第二数据集真值。原始切换先暂存、读回核验再替换，失败保留旧数据。
已验证必需字段的派生投影可作为唯一活动规范输入；保留原件于 data/archive/<source_id>/<dataset>，
该目录不参与 raw 扫描。元数据分别记录原件观察日、未知的原生导出日及真实投影导出日，
并以原件、转换脚本和投影哈希证明来源；不能把未合规的原文叙述标为结构通过。
独立历史费率 CSV 是 QRP 研究参考输入；其精确原始内容保持，
独立外部来源核验状态与实现是否可消费分别说明，不制造第二份规范副本。

### 公开行业参考来源

公开行业快照属于 `reference` 来源，由现有 reference interim owner 读取
`data/raw/reference/a_share_industry_snapshots/metadata.json`。元数据声明机构对应的官方页面、
标准、观察日、公开日、快照身份、候选 JSON/原始附件/发布页面逻辑文件名及全部来源文件 SHA256；
候选逐行保留资产源代码、公司原名、分类原字段和原件页码或表/行定位。文件使用共同的有界读取和精确身份
owner，提交前重核完整源身份；缺少已声明文件、身份变化或结构不合法不能以空列表替代，
失败保留已验证旧产物。公开源与 CSMAR 原件分别保留，不伪造供应商字段或修改原始附件。

`a_share_industry_snapshots` 是 source_id 为 reference 的可选 PANEL 输入。纯价格、量价因子
不强制行业来源；缺少行业证据时分类与非金融标志保留 NULL，不推定为非金融或回填当前行业。
确实消费行业的中性化、适用性或风险计算按所需时点和有效域处理缺失；显式供给的行业证据仍须通过
其选中来源、PIT、未知屏障和真实冲突检查。
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
观察/下载时刻不冒充历史公开日。附件哈希与 URL 只证明来源完整性，具体事实和数量换算
仍须人工核对原件；未知的公告时间、原股消灭或换股关系不可由字段齐全推定。

支持 `quote`、`cash_recovery`、`cancel`、`conversion`、`share_change` 五种明确事实。
人民币金额 `amount_per_share` 按当时实际原股计，转换比例
`successor_shares_per_old_share` 为每原股换得的后继实际股份数。
账本的 `value_per_old_unit`、`remaining_old_units_per_old_unit` 等字段中的 unit
均指实际股份，不再换算为复权价格单位，也不接纳旧 `shares_per_old_unit` 证明结构。
实际证券生命周期和经济日期必须匹配。金额与数量比例共用规范精确有理数编码，避免中间溢出丢失最终
可表示的金额；原始 PANEL 行情类型保持不变。每个事件有独立身份；同一旧权益在同一
确认会话的多个事件若缺少可证明顺序则拒绝，不以事件 ID 排序冒充经济先后。
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
只有报价不会产生交易许可；现金回收不冒充卖出，注销不产生现金，同码重新上市不构成权益衔接。
参考事件经原生 reader 进入 `portfolio_sessions`，仍检查实际消费生命周期和单位关系；
可靠关闭，主异常不被清理异常替换。

## 持久化与可消费范围

输入和新输出路径由调用者显式指定，raw/interim/processed/research只是数据责任层，
不要求instances/hash或运行目录。ProcessedDataSpec的来源声明是实际消费字段关系，
允许缺省的来源只在未提供时省略；明确指定坏路径、缺必需列或文件损坏不能fallback为空。

native spec 记录有序列、角色、主键、单位、标签可用时间映射与配方；分区是物理布局，
时间主键不强制按年分区。未声明单位或可用时钟，不等于无量纲或标签已经成熟。
native manifest 记录范围、参数和必要物理事实；不以 PASS 标签或自我摘要作为读取通行证。
普通读取只打开指定成员、列和窗口，无全树SHA/快照或读中替换保证；详见[架构](architecture.md)。
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

Git 只保存代码、合同、结构定义、文档、合成测试和安全模板。
私有原始与生成行数据、供应商载荷、秘密、本地绝对路径及原始回溯不提交。
公开股票代码、提供商标识符和合成测试样例本身不属于秘密；不得借这些标识符提交关联的私有逐行观测。
共享检查只输出结构、计数、哈希、状态、逻辑路径及不含私有行的摘要；
在配置/进程/错误边界先选择安全字段，不能输出完整可能含令牌的内容。

## 独立来源选择补充

CSMAR `prepare_source` 省略起止日期时消费明确输入中的完整历史，不隐式限于2014—2025。
显式日期仍遵守各来源语义：行情与复权按双端窗口；股本保留窗口前状态、仅截末端；
其他状态来源保留已有历史处理，不因接口同名而丢失因果前缀。

reference 可以在一个请求中明确多个 outputs，各表独立发布与报告缺源，不提供跨表事务。
非空 start_date 明确拒绝，避免假装已经裁剪历史；end_date 只限制行业/权益的知识时点，
费率有效区间保留。需要选择其他合法费率文件时，成对给出 `fee_csv` 与 `fee_sidecar`，
路径直接解释，不再与 raw_root 拼接；省略时沿用已有参考目录默认。内容、哈希、字节数、
行数、单位与覆盖证明仍由来源 owner 检查，改文件名不能制造新年代的合法覆盖。

产物根可以命名为 data，实际结构及来源声明决定其含义。单独的数据子目录缺少必需结构
仍不是完整产物。schema inventory 仅诊断旧项目 data/{interim,processed,research} 布局，
不把该布局当作独立计算的通行条件。

## 字段字典与阶段输入输出

本节是通用字段语义及 schema 的唯一导航，不另列一份会漂移的全仓列清单。
已批准但未实现的默认与当前代码差距见 [规则状态索引](specifications.md)；以下链接指向实际合同，
不能把字段已经存在当作整条研究链已经验证发布。

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
标签不可插补；训练期填补的已批准方向见规则索引，不能把现有 `impute_missing` 布尔开关
当作训练中位数及缺失标志已实现的证明。`feature_`、`feature_missing_`、`feature_observed_`
的现有输出关系由 [FeatureTransformSpec](../src/quant_research_platform/contracts/features.py) 定义，
特征支持列不自动成为模型 X，模型列序由拟合合同保存。

### CSMAR equity_changes：报表来源与科目粒度

`FN_FN046` 的同一证券、报告期、报表类型可以有多个来源和科目。
INTERIM 主键为 `source_asset_id, timestamp, report_type, data_source, subject_code`，
后两列分别原样映射 `DataSources` 和 `SubjectCode`，按字符串保留前导零。
不得以旧三列粗键去重，也不得把不同来源、科目合并成一条权益变动。
报告期不是公告可用时点；这些身份字段本身不证明每股口径或公司行动已经对齐。

### CSMAR corporate_actions：八个规范字段

唯一来源表为 `TRDNEW_Cptl`，已有 source owner 直接产生 INTERIM `corporate_actions`。
来源映射见 [field_mappings.py](../src/quant_research_platform/sources/csmar/field_mappings.py)，
类型、保留与诊断见 [interim.py](../src/quant_research_platform/sources/csmar/interim.py) 的
`CORPORATE_ACTION_COLUMNS` / `_normalize_corporate_actions`。

| 规范字段 | 来源字段 | 含义、类型及边界 |
| --- | --- | --- |
| source_asset_id | Stkcd | 来源证券代码字符串；生命周期由消费者另核 |
| timestamp | Exdistdt | 规范化除权日期；不是公告或到账时间 |
| kind | Disttyp | 来源事件代码字符串；未知代码保留并计数，不伪造映射 |
| market_type | Markettype | 可空整数市场代码；不是自动可交易资产证明 |
| announcement_date | Annodt | 可空日期；未知/不合法保留缺失并诊断，不用下载日代替 |
| settlement_date | Paydt | 可空支付/结算日期；在适用事件中核其真实经济含义 |
| amount | Amount | 可空数值，单位依事件 kind：现金每原股金额或每原股新增股份等，不能跨 kind 混用 |
| subscription_price | Roprc | 可空认购价格；保留来源事实不等于策略执行认购 |

来源层对可空字段的非法值记录诊断并保留未知；除权日期和事件标识仍必须合法。
该表没有供应商登记日字段，不能捏造。普通 CA/SD 的登记时点按获批规则由自身市场日历推导；
付款日期缺失时除权日开盘到账是明确研究假设。未知事件是否阻塞由实际持有人权益依赖裁定，
不是在 source 层删除整段来源。

本轮原股账本直接消费该表；内部事件投影的 `record_at`、`effective_at`、`settlement_at`、
`available_at`、`cash_per_share`、`additional_shares_per_share`、`entitlement_proved`
由 [portfolio_sessions.py](../src/quant_research_platform/markets/implementations/a_share/portfolio_sessions.py)
定义，只是既有 owner 的计算输入，不新建一份公共事件表或准入框架。
原股账本与旧复权单位账本不可混用；退市后参考权益须按实际原股及生命周期解释。

### 输入选择器：只解释本次明确选择

支持共同数据集选择器的入口接受路径，或含 `path` 的对象。可选字段为 `spec`、`files`、
`columns`、`years`、`time_window`、`row_filters`，以 [现有 reader](../src/quant_research_platform/storage/datasets/reading.py)
的 `_dataset_input_options` 和各操作入口为准，不保证所有独立参数都接受此对象。

- `files` 选择成员，`columns` 保持明确列序，`years` 为离散年份集合；同时给年份与窗口取交集。
- `time_window` 由 [TimeWindow](../src/quant_research_platform/contracts/time.py) 定义；日频结束日期含全天，不能因此改写 Crypto 半开观察区间。
- `row_filters` 是现有 `==`/`in` 有限物理筛选，不是任意表达式引擎，也不证明样本科学资格。
- `spec` 为裸 Parquet 补足本操作真实需要的角色、键、单位和标签成熟映射，不能用声明替代来源证据。
- 选择输出窗口不缩短公式回看、标签端点或完整截面。排除的报价保持未知，不压紧交易会话轴。

### 阶段输入、输出与唯一合同位置

| 独立操作 | 明确输入及主要选择 | 输出与 schema / owner |
| --- | --- | --- |
| prepare-source | 指定来源根/原件与表，按 source 合同选择日期 | INTERIM 规范表；[sources/prepare.py](../src/quant_research_platform/sources/prepare.py)，各来源映射及 reader |
| prepare | 已准备来源、市场、数据集及窗口 | PANEL/FACTOR/LABEL；[contracts/data.py](../src/quant_research_platform/contracts/data.py)、市场 catalog；不是自动补齐缺源 |
| assemble | 明确基表、因子批次/已有特征、标签、列投影及特征配方 | 可复用研究数据集；[ResearchDatasetSpec](../src/quant_research_platform/contracts/datasets.py)，保留列角色、成熟映射和 join 覆盖 |
| fit | 数据、列序、目标、拟合窗/截止、算法参数及 seed | 冻结 predictor 与 model.json/trial；[contracts/models.py](../src/quant_research_platform/contracts/models.py)、[production.py](../src/quant_research_platform/research/models/production.py) |
| predict | 明确已存模型及新特征窗口 | PRED 键、prediction_score、prediction_rank_score；[prediction.py](../src/quant_research_platform/artifacts/outputs/prediction.py)，不附带未来标签、不重新拟合 |
| score / select | 有序完整候选、验证窗与成熟截止；选择另声明规则和最终拟合窗 | 各自 JSON；[trial_selection.py](../src/quant_research_platform/research/models/trial_selection.py)，不自动最终重训；新批准的默认见状态索引 |
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


### 模型预测描述指标

`model_statistics` 的 `model_losses` 分析同时发布 `model_prediction_metrics.parquet`，
主键为 `timestamp, family_id`，每行是一家族在一个实际输入日期的描述统计。
字段/schema 唯一 owner 为 [statistics.py](../src/quant_research_platform/artifacts/statistics.py)。

| 字段 | 单位与定义 |
| --- | --- |
| prediction_count / target_count | 有限预测数 / 预测与成熟主目标成对有效数 |
| mse / mse_reason / baseline_mse | 当日成对域等权平方误差及未定义原因 / 零预测基准平方误差，单位为主目标单位的平方 |
| oos_r2 / oos_r2_reason | `1−mse/baseline_mse`；零基准误差、无目标或不可表示结果为 NULL 并说明原因 |
| head_selected_count / head_evaluable_count | 仅由预测确定的头部人数 / 固定头部中原始收益已成熟人数 |
| head_coverage | 可评价人数除以选中人数；无选中人为 NULL，不以覆盖硬门槛过滤 |
| pool_evaluable_count | 同日 eligible 研究池中原始收益已成熟人数，不要求该资产有预测 |
| head_mean_raw_return / pool_mean_raw_return | 固定头部 / 同日研究池的等权原始 horizon 收益，比例单位、未扣成本 |
| head_excess_raw_return | 以上两均值之差；任一不可评价则 NULL |

`evaluation.head_fraction` 默认 0.2，可显式覆盖；先按 `ceil(N×fraction)` 确定预测阈值，
边界并列全部入组，再检查目标/收益可用性。`raw_return_label` 可显式指定，默认只在输入存在
`forward_return_20d` 时使用该列，否则收益指标未定义，不把标准化目标冒充原始收益。
元数据 `prediction_metrics` 保存解析后的头部比例、原始收益列和聚合口径。
MSE 使用标签尺度预测：rank 模型使用各次训练冻结的 `label_calibration`，普通回归输出不重复缩放；原始排序分数继续用于 IC 和头部选择。
训练从实际消费的 spec/manifest 冻结 `facts.target_definition`：源标签、目标 transform、clip 以及所选源标签的 `label_semantics`；PRED 的 `model_facts` 原样携带。
普通标签记录源列本身、不追加变换；`model_target` 记录 `forward_return_20d` 及实际 `target_transform`/`target_clip`。
不绑定路径、data ID、hash 或整份 spec，不在后来从训练源回读补造历史事实。
score 的显式 `label_unit` 可补充标签输入未声明的单位；若输入已有单位，两者仍须一致。该声明保存在评分/选择记录中，不从训练单位猜测评价单位。
只有目标定义和两侧明确单位相同，才计算 MSE 和 OOS R²；`model_target` 的 raw/none 与同源原始标签可比较。
目标不同、单位缺失或不同分别记为 `prediction_target_mismatch`、`target_unit_missing`、
`prediction_target_unit_mismatch`；冻结定义未知（含旧模型未保存定义）、对侧缺必需定义或定义不同分别记
`prediction_target_definition_missing`、`prediction_target_definition_mismatch`。仅将 MSE/R² 设为 NULL，
保留目标计数、零预测基准、RankIC 和头部描述；不猜测单位换算。普通声明标签两侧均未提供额外 `label_semantics` 时按同源列定义比较；一侧有额外定义而另一侧无则未定义。
`mse_reason` 还区分 `no_valid_targets` 与 `nonfinite_result`；非零损失下溢不得伪装成完美预测。
跨日 MSE 先算每日均方误差再日期等权；跨日 OOS R² 为同一可评价日期域上
`1−mean(mse)/mean(baseline_mse)`，不平均每日 R²，不按资产行数加权。
这些重叠 horizon 收益不连乘为日净值。训练均值基准尚无训练域来源证明，当前不提供该选项。

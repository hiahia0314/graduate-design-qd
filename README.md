# myReTest

## 输出目录与独立因子库

实验输出继续位于 `output/`；长期因子库独立位于项目根目录 `factor_library/`，不是 output 的子目录。
`--output my_run` 与 `--output output/my_run` 都指向 `output/my_run`。

```text
factor_library/
  B/
    factor_set.json            # Alpha158 全部因子 + 所有真实 LLM 因子，含完整测评历史
    factor_index.json          # 全量指标索引，不预先过滤，供训练按指标筛选
    training_candidates.json   # 当前训练筛选规则的可选视图，不是全量库
  A/                           # A 方向运行时创建/更新；B 运行不会修改 A
output/
  output_mining_.../            # 实验轨迹、因子值、完整逐日测评
  output_comparison_.../        # 对比结果
  output_labels_.../            # 标签包
```

每次真实 LLM 流水线结束（包括预算中断、异常退出），向对应 A/B 因子库追加已生成的因子及已有测评。
已生成但尚未计算的候选以 `not_evaluated` 保留，指标为空，不进入已筛选视图。
Alpha158 对比和独立因子测评也自动追加结果。所有因子均保留，包括缺少字段、覆盖不足、评分不合格的因子；
缺失指标为 null/缺省，绝不伪造为零。公式由 candidate_id 去重，同一因子多次测评追加历史记录，
旧测评不会覆盖或删除。重复导入相同记录不重复添加；记录变化保留新修订。

训练端可从 `factor_index.json` 的 `contexts[context_id]` 中按来源、分段、IC、RankIC、ICIR、覆盖率、
年度一致性及分组收益等筛选；每条包含原始测评摘要、评分、公式和因子值文件位置。
`factor_set.json` 是完整权威库，`training_candidates.json` 只是通过当前训练段筛选的派生视图。
不同数据/标签版本不混合，验证或测试测评不会被当作训练筛选证据。因子值文件仍保存于原实验目录，
因子库集中保存定义、指标及可定位的路径。入池不代表样本外有效；代数等价与相关性需另行检查。
回放与离线模拟数据不进入真实 LLM 因子库。

离线补齐历史 LLM 和 Alpha158 记录（只更新指定方向，无 API 调用）：

```bash
.venv/bin/python scripts/sync_factor_collection.py --task B
```

一个最小化的 A 股日频截面选股回测框架。策略每天收盘后同时查看所有股票，输出目标权重组合；组合层计算调仓订单，撮合层在下一交易日开盘成交。

## 数据流

```text
ParquetDailyDataSource（前复权）
  -> MarketSnapshot（当日全部股票）
  -> Strategy.on_day()
  -> TargetPortfolio
  -> TargetWeightPortfolio
  -> Order
  -> NextOpenBroker（下一交易日开盘）
  -> Fill
  -> EquityAccounting
```

## 运行

```bash
python scripts/prepare_dataset.py
python scripts/run_research_backtest.py --output output/output_research
```

可选参数：

```bash
python scripts/run_research_backtest.py --initial-cash 1000000 --top-n 2 --output output
```

运行后生成：

```text
output/output_research/result.json
output/output_research/performance.json
output/output_research/target_signals.csv
output/output_research/run_manifest.json
```

## 示例数据

`data/samples/sample_bars.csv` 包含 6 只虚构 A 股、20 个交易日的日线数据：

```text
symbol,date,open,high,low,close,volume
```

这些数据仅用于验证截面排序和调仓流程，不代表真实市场行情。

## 设计说明

关键架构决策及简短理由见 `docs/ARCHITECTURE_DECISIONS.md`。

## A/B 信号接入与研究结果导出

```bash
.venv/bin/python -m pytest -q
# 演示入口 run_signal_demo.py 已移除；A/B 接口契约见 src/strategies/signal_strategy.py
```

此示例使用虚构行情、演示股票池和固定仓位，只验证接口链路。`SignalStrategy`
接收选股 B、仓位 A、历史成分股三个 callable；信号模型版本和 ID 会保存在结果中。
真实历史股票池可以通过 `HistoricalUniverse.from_csv(path)` 接入，文件必须包含
`date,symbol`，每个研究交易日提供完整的当日有效成分股，缺日直接报错。
不要用行情中出现的股票或当前成分股名单替代真实历史成分股。

两个运行入口均可生成 `result.json` 和 `target_signals.csv`。JSON 保留信号、空仓目标、
订单、成交、每日权益、订单处理结果及期末待撮合订单；Decimal 保存为字符串。
`scripts/run_research_backtest.py` 是前复权研究入口，完整审计以 JSON 为准。

行情 CSV 可额外提供 `buy_block_reason` 和 `sell_block_reason`：非空表示该方向
在开盘时禁止成交，值会作为取消原因保存。例如数据适配器可以将停牌原因填入两个方向，
将开盘买入限制仅填入买入方向。**这些字段必须来自开盘时可知的数据**，框架不会根据
全天最高价、最低价或收盘价反推开盘交易状态；缺省空值仍是演示版“允许交易”假设。
实际历史交易规则识别和数据完整性审核仍需后续数据适配器完成。

未成交部分当天取消，下次信号重新调仓。信号 A 缺失时报错；纯选股实验显式提供固定仓位。
最后一个交易日生成的订单没有下个开盘可撮合，保留在 `pending_orders`，不虚构成交或取消。

## 沪深300真实行情数据包

年度源数据为 `data/market/daily/2017.parquet` 至 `2025.parquet`，历史成员为
`data/universe/membership.parquet`。原始行情仅用于前复权转换与预先判定交易约束。
数据文件、单位与限制见 [数据说明](data/README.md)。

## 静态风控与绩效评价

风控默认关闭，必须显式配置。演示开启单股15%、最多2只的限制：

```bash
# 演示入口 run_signal_demo.py 已移除；静态风控接口见下方 Python 示例
```

上述数值仅用于演示接口，不是推荐研究参数。风控顺序为：禁入过滤 → 按目标权重保留前K只
（同权重按代码升序）→ 单股上限 → 行业内同比例缩减至行业上限。削减部分留作现金，不补选
股票或重新归一化；不启用回撤自动降仓。原始目标保存在 `risk_decisions.input_target`，
执行目标带有 `parent_target_id` 并保持 A/B 信号关联。

Python接入：

```python
from decimal import Decimal
from src.core import RiskLimits, StaticRiskControl

risk = StaticRiskControl(RiskLimits(
    max_holdings=20,
    max_stock_weight=Decimal('0.10'),
    max_industry_weight=Decimal('0.30'),
    filter_ineligible=True,
))
# 传入 BacktestEngine(..., risk_control=risk)
```

启用行业上限或禁入过滤时，必须提供当日已知的 `MarketSnapshot.metadata`。
`SecurityMetadata.industry` 是行业名称；`exclusion_reason` 非空表示禁入，空字符串表示
明确允许，`None` 表示未知。行业或资格信息缺失时，对应规则报错，不把未知当成合格。
CSV源支持可选的 `industry` 和 `exclusion_reason` 列；它不会自动识别ST或推断历史行业。
当前下载的真实行情包尚未提供这些历史属性。

`result.json` 新增风控配置、风控前后目标、调整原因及每日实际持仓。持仓记录的
`price_is_stale` 表示当天缺少行情、估值沿用历史收盘价。目标权重限制无法保证实际仓位
时刻合规：价格变化或卖单受阻都可能使实际仓位超限，报告记录实际结果。

两个运行入口都会新增 `performance.json`：

- 总收益以第一条收盘权益为基准；年化按相邻收盘收益区间数计算，默认每年252个交易日。
- 最大回撤为正的损失比例；波动率使用日收益样本标准差，夏普默认年无风险利率为0。
- 日胜率是正收益日占所有收益观察日的比例，包含零收益日；它不是平仓交易胜率。
- 换手率为当日买卖成交金额之和除以两倍前日收盘权益；交易费用为佣金加印花税，
  滑点已体现在成交价中，不重复扣除。
- 同时保存每日仓位、持仓数、单股及行业集中度。缺少持仓/行业信息的统计为 `null`，
  不推断行业；样本不足或收益波动为零时夏普为 `null`。
- 可通过 `export_result(result, directory, benchmark={时间戳: Decimal收盘价, ...})` 传入
  与权益日期严格对齐的基准收盘价；基准缺日会报错，不前向填充。超额收益定义为策略累计
  收益减基准累计收益。没有基准时两项基准指标为 `null`。

年化参数可用 `PerformanceConfig(periods_per_year=252, annual_risk_free_rate=0.0)` 配置，
通过 `export_result(..., performance_config=config)` 传入，口径和参数会保存在报告中。
当前没有现金流入/流出模型，收益计算不适用于中途追加本金的账户。

## 真实数据接入与研究区间

当前框架统一使用前复权价格：信号、目标数量、次日开盘撮合、交易费用及收盘估值都使用相同价格口径。
`python scripts/prepare_dataset.py` 按 `config/dataset_splits.json` 准备
`data/processed/csi300_2017_2025_v5_qfq`，生成 `execution_qfq.parquet` 和 `research_qfq.parquet`。
每只股票使用全保留区间最后有效日因子作为固定锚点，所有分段共用同一锚点；这是固定快照前复权，未宣称为时点可得数据。
`qfq_anchors.parquet` 保存锚定日期与因子，文件均纳入哈希校验。

训练区间为2017—2021年，验证为2022—2023年，最终测试为2024—2025年。
最终测试仍需显式授权；收益标签现在由 `scripts/prepare_labels.py` 作为独立版本生成，见下方“默认收益标签与 DeepSeek 接入”；不自动拟合预处理。
`PreparedSplit` 只接受新前复权数据集，旧原价版本不再能进入当前回测入口。

停牌和涨跌停限制由原始行情预先判定，运行时仅传递限制原因；买入数量按100个复权单位取整，不代表实际股数。
沪深主板ST、*ST的普通限价交易日：2026-07-06之前为±5%，自该日起为±10%；不影响创业板、科创板。
年度适配器收到当日 `is_st=1` 和 `limit_reference_price_raw`（交易所当日涨跌停计算参考价，
除权除息日须用相应参考价）时按日期计算限价，四舍五入到分；明确不限价日优先。
缺参考价时沿用供应商每日限价，不从复权价或旧限价反推；限价未知仍禁止成交。
当前2017—2025年数据不受2026年规则变更影响。

全部公司行动记账和事件保护已移除：不加载分红、送转、配股、换股、现金选择权或退市事件，
不扣认购资金、不发放现金或股份、不转换代码，也不因上述事件阻塞。
无当日报价时按既有通用规则沿用最后前复权收盘价估值，缺行情订单取消；不自动清零或模拟退出。

## 用正式面板运行回测

`scripts/run_research_backtest.py` 为正式前复权数据入口。

```bash
# 默认训练段，输出目录必须为空或尚不存在
.venv/bin/python scripts/run_research_backtest.py --output output/output_research_train

# 指定训练段内区间，验证接线
.venv/bin/python scripts/run_research_backtest.py \
  --split train --start 2017-01-03 --end 2017-01-10 \
  --output output/output_research_check
```

入口默认读取 `data/processed/csi300_2017_2025_v5_qfq`，可用 `--dataset` 替换。
当前接入的是既有当日涨幅基线：用当日前复权价的 `close/open - 1` 排名，
经每日历史成员筛选后等权持有 `--top-n`（默认2）只股票，收盘决策、次日前复权开盘撮合。
这是框架接线基线，不是训练好的策略。前复权源保留退池证券用于持仓估值及退出。

`--exposure` 控制固定仓位，`--initial-cash` 控制初始资金。每次运行从现金开始，
不继承其他区间的持仓。默认双向佣金为万三（0.0003），单边不利滑点为0.2%（0.002），可通过 `--commission-rate`、
`--minimum-commission`、`--stamp-duty-rate`、`--slippage` 显式设置；滑点敏感性范围为0.1%–0.3%。
经手费、监管费不单列；最低佣金和卖出印花税仍默认0（可配置），固定费用不代表历史费率表。
买价＝开盘价×(1＋滑点)，卖价＝开盘价×(1－滑点)，佣金按成交金额计算，四舍五入到分。

成功运行导出 `result.json`、带沪深300基准比较的 `performance.json`、
`target_signals.csv` 和 `run_manifest.json`。清单记录数据版本、数据清单哈希、区间、
策略参数、费用及运行状态；最后一个交易日的待执行订单保留，不强行平仓。

正式测试段须同时指定 `--split test --allow-final-test`；仅为接线检查无需使用测试集。

## L1：A/B 模型的因子候选接入

> L1 候选目录的 CLI 入口（`prepare_factors.py`）已移除；改用下方 Python API（`build_catalog` / `export_catalog`）。

输出 `candidates.json`，包含完整 Alpha158 公式、A/B 独立候选池、输入字段、所需历史长度、
来源、假设方向、适用场景、材料证据和泄漏风险。记录提示词、模型标识、材料/响应哈希、
重试结果及目录哈希；已有文件禁止覆盖。默认离线运行，两路各有 158 个 Alpha158 和 3 个
明确标记为 `fallback` 的候选，不声称已调用真实 LLM。JSON 回放失败也会明确记录并回退。

- **B 选股**：Alpha158 与专用 LLM 因子按 `stock` 计算，输出粒度为日期×股票。
- **A 市场状态**：默认同一套 Alpha158 先逐股计算，再以当日历史成员等权聚合
  （`member_mean`），输出粒度为日期。专用 LLM 因子允许指数趋势、波动率及市场宽度。
  布尔表达式逐股计算后的均值表示成员比例。`--a-scope index` 可改为指数 Alpha158。
- `member_mean` 是后续计算层必须遵循的契约，不是对当前成员名单或原始 OHLCV 先求平均。
  L2 需逐日应用历史成员、披露有效样本覆盖率，并确定缺失处理，不能把缺失因子填为零。
- 保留全部 158 个基线定义。默认字段声明参考现有准备面板：股票 OHLCV/amount，指数 close；
  未提供 VWAP 的候选标为 `blocked`，不使用典型价替代。当前默认两路各有一个 VWAP0 受阻。
  `fields_declared` 仅说明声明的字段齐全，不代表行情已审核或已经计算成功。
- L1 不读取训练/验证/测试行情，不计算因子、不筛选方向、不拟合模型、不生成交易信号。
  L2 仍需处理复权时点、单位一致性、除零/非有限值、停牌与历史窗口；候选均为 `candidate_only`。

真实 LLM 通过 callable 接入，调用者负责供应商 SDK、密钥和超时：

```python
from src.factors import build_catalog, export_catalog

# llm_a / llm_b: (prompt: str) -> str，返回纯 JSON 文本。
# 两路可以使用同一个 LLM 服务，但收到的任务提示与输出范围分别约束。
catalog = build_catalog(
    providers={"A": llm_a, "B": llm_b},
    models={"A": "实际模型及版本", "B": "实际模型及版本"},
    research_text="仅提供研究资料，不混入最终测试集表现",
    retries=2,
)
export_catalog(catalog, "output/output_factors_live_v1")
```

可用 `build_prompt('A')` / `build_prompt('B')` 获取提示词及完整 JSON Schema。
每路最多尝试 `retries + 1` 次，格式/表达式/调用失败后重试，耗尽后确定性回退。
错误只保留类型，避免服务异常泄露密钥；原始响应只保存哈希，合格因子完整保存。
手工准备响应的最小例子（A 应使用 `index` 或 `member_mean`）：

```json
{
  "task": "B",
  "factors": [{
    "name": "MOMENTUM10",
    "expression": "$close/Ref($close,10)-1",
    "required_fields": ["close"],
    "scope": "stock",
    "rationale": "检验短期动量假设",
    "scenario": "截面选股",
    "expected_direction": "正向假设，待训练段检验",
    "leakage_risk": "复权价格必须时点可得，禁止未来回填",
    "evidence": "填写具体研究资料出处"
  }]
}
```

公式通过白名单 AST 静态校验，不执行模型输出的 Python；拒绝未来引用、任意函数、属性访问、
未知字段、非法窗口及 A/B 范围混用。表达式去重按任务与数据范围分别进行。
Schema 约束字段类型与结构，另有语义检查；这些检查不能证明 LLM 的经济解释或来源真实。
Alpha158 公式参照 [Qlib Alpha158DL](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)。

后续实验建议分别比较 Alpha158-only、LLM-only 与两者合并。共享基础公式有助于形成一致基线，
但 A/B 的标签、标准化、因子筛选和评估指标需要分别设计，并仅在训练段拟合。

前复权公式为 `供应商调整价 / 每只股票最后有效日复权因子`。已确认保留供应商调整价：
2018年11个停牌日的调整价与原价乘因子存在差异，记录在 `provider_factor_mismatches.parquet`，不改写原数据。

## 因子研究闭环：共享测评 + LLM 反馈迭代

在 L1 候选契约之上，新增独立计算、测评和研究循环。架构参考
[QuantaAlpha](https://github.com/QuantaAlpha/QuantaAlpha)，本项目未引入其运行时，
也不宣称完整复现 CoSTEER。研究循环不修改现有策略、交易撮合或记账模块。

```text
研究方向 → planning → hypothesis → expression
                                     ↓
                         确定性 DSL 实现与因子计算
                                     ↓
Alpha158 ──────────────────────→ 独立因子测评
                                     ↓
                          feedback → mutation/crossover
                                     ↓              ↓
                          因子库与实验记录 ← 后续迭代
```

模块职责：

| 模块 | 职责 |
| --- | --- |
| `src/factors/computation.py` | 执行白名单公式，按历史成员生成逐股/市场特征 |
| `src/factors/evaluation.py` | 不依赖 LLM 的因子诊断：覆盖率、IC、RankIC、分组未来收益 |
| `src/factors/runner.py` | Alpha158 与 LLM 候选使用相同数据、计算器、标签和评价配置 |
| `src/factor_mining/pipeline.py` | 规划、假设、表达式生成、反馈、父因子选择与有界迭代 |
| `src/factor_mining/library.py` | 原子写入因子库，按公式身份去重并保留各次实验 |
| `scripts/run_factor_research.py` | 正式数据的独立测评入口和研究阶段 JSON 回放入口 |

**coder 的当前实现**：LLM 输出受限 DSL，由确定性执行器计算数值，不运行 LLM 任意生成的
Python。执行器版本为 `pandas-dsl-v1`，支持现有全部 Alpha158 公式。滚动窗口要求完整观测，
标准差采用样本标准差；Rank 为窗口内平均秩百分位；IdxMax/IdxMin 为从窗口最早观测起算的
1-based 首次极值位置。该本地数值契约已测试，但尚未逐值对照 Qlib，不宣称数值完全等价。
缺失交易日保留为 NaN，不跨过缺日压缩窗口；未来行情修改不会影响过去因子值。

**A/B 评价分开，来源口径统一**：B 做按日截面相关性、非年化 ICIR、原始升序分组的未来
收益及高减低收益差；A 做按日期的时间序列相关性和非重叠标签区间相关性。
A 的历史成员均值同时记录有效成员覆盖率，默认测评要求覆盖率 100%。
缺失标签值、因子值、预热窗口均反映在覆盖率中，不静默补零；常量或样本不足返回空指标。
标签越过分段终点的样本会被剔除。非重叠诊断采用保守的互不相交标签区间，不报告未经校正的 t 检验。

分组未来收益是**因子诊断**，不等于包含实际交易约束的策略收益；当前因子测评没有策略回撤、
交易成本或状态分类 F1。后续 F1 需要明确分类标签及预测器，交易回测则需接入组合和撮合层。
这些指标没有用因子相关性代替。重复挖掘仍会过拟合训练集，最终应冻结候选后独立验证。

### 使用正式数据测评 Alpha158 或 LLM 候选

依赖：`pip install -e '.[research]'`。下述为外部标签兼容接口；本项目默认标签可直接用 `scripts/prepare_labels.py` 生成：

- B：`date,symbol,target,label_end`；A：`date,target,label_end`。
- `target` 是已定义持有期的未来收益，`label_end` 是计算标签实际使用的最后交易日。
- 必须说明入场、退出、价格口径及预测周期。自定义标签必须显式给出定义；默认口径固定于 `config/label_spec.json`。
- 正式面板通过 `PreparedSplit` 核验。CLI 只加载所选分段，段首缺少预热历史会降低覆盖率；
  Python `FactorRunner` 可显式传入更早的预热行情、交易日历及对应历史成员。
- 特征价格的时点可得性仍需数据审核；字段齐全与哈希通过不等于复权信息不存在前视问题。

```bash
# 标签文件及定义需先按自己的研究方案准备，以下路径/定义是占位说明
.venv/bin/python scripts/run_factor_research.py \
  --mode evaluate --task B --split train \
  --dataset data/processed/csi300_2017_2025_v5_qfq \
  --labels /path/to/train_b_labels.parquet \
  --label-definition '填写已确认的入场、退出、周期和价格口径' \
  --output output/output_alpha158_evaluation_v1
```

加 `--catalog output/output_factors_llm_v1/candidates.json` 即改用已有候选池，保持同一测评接口。
A 用 `--task A` 和对应的指数收益标签。`--split test` 还要求 `--allow-final-test`；
训练循环始终拒绝 validation/test。输出目录必须不存在。

结果包括总报告 `evaluation.json`，以及每个因子的 `values.csv`、`report.json`；
缺字段的因子保留 `blocked` 报告。报告记录标签定义、分段、数据标识、输入哈希和执行器版本。

### 接入真实 LLM 研究循环

```python
import pandas as pd
from src.factors.runner import prepared_runner
from src.factor_mining import MiningConfig, MiningPipeline

runner = prepared_runner(
    "data/processed/csi300_2017_2025_v5_qfq", "train",
    pd.read_parquet("/path/to/train_b_labels.parquet"),
    "填写已确认的标签定义", "B",
)

def provider(stage: str, prompt: str) -> str:
    # 调用选定供应商的 API，设置超时，返回模型回答的纯 JSON 文本。
    # stage 分别为 planning/hypothesis/expression/feedback。
    # 可按阶段选择模型，也可始终使用一个模型；密钥通过本机环境变量读取。
    return your_llm_client(prompt)

pipeline = MiningPipeline(
    MiningConfig(task="B", direction="研究量价确认下的动量与反转",
                 model="填写实际模型版本", rounds=5, directions=2, retries=2),
    provider, runner, "output/output_mining_b_v1",
    # 可选跨实验共享库；省略则在本次输出目录中建库。
    library_path="output/output_factor_library/all_factors_library.json",
)
trace = pipeline.run()
```

A 使用独立的 `task="A"`、研究方向、runner 和标签。相同供应商可以服务两路。
现在已内置 DeepSeek 官方客户端，见下方命令行接入方式；上述 callable 仍支持其他供应商。
DeepSeek 客户端从本机配置读取密钥，记录安全错误码、调用次数和 token 用量，不记录密钥或 HTTP 错误正文。

每阶段有有限重试；重试耗尽或不可恢复的技术错误会中断实验并留痕，不将离线候选伪装成 LLM 实验。
后续轮次由反馈建议 `mutate` / `crossover` / `stop`；`stop` 只记录为建议，由程序按下方固定条件决定终止。
父因子只从本次同任务训练实验中选择。B 按统一方向的年度平均 RankIC 中位数排序，并要求覆盖率与年度方向稳定性；
A 保留整段时间序列绝对 RankIC 评分。交叉要求两个不同公式；没有合格父因子时重新探索。
重复公式不重复计算，统计意义上的相关性去重尚未加入。该策略是首版可复现控制器，不是完整遗传算法。

`trajectory.json` 保存规划、提示词、响应哈希、失败阶段、假设、父因子关系、测评和反馈；
`all_factors_library.json` 保存因子及所有实验历史，包含受阻和证据不足的结果，不将它们
自动标记为有效因子。每个实验另保存数值与报告，因子库损坏时报错而非静默重建。

无 API 时，可以用手工示例验证流程（不代表真实 LLM 反馈）：

```bash
.venv/bin/python scripts/run_factor_research.py \
  --mode mine --task B --split train \
  --dataset data/processed/csi300_2017_2025_v5_qfq \
  --labels /path/to/train_b_labels.parquet \
  --label-definition '填写已确认的标签定义' \
  --direction '短期动量研究' --rounds 1 --directions 1 \
  --responses examples/mining_replay_b.json \
  --output output/output_mining_replay_v1
```

纯合成数据的模块验证无需正式标签或 API：

```bash
.venv/bin/python -m pytest -q tests/test_factor_candidates.py tests/test_factor_research.py
```

## 默认收益标签与 DeepSeek 接入

首版标签固定于 `config/label_spec.json`，这是预先指定的研究基线，不是验证集或测试集选出的最优参数。
T 日收盘计算特征；T+1 开盘入场，T+6 开盘退出，中间持有 **5 个交易日**。

| 模型 | 连续 `target` | 额外训练标签 |
| --- | --- | --- |
| B 选股 | 个股前复权开盘收益 − 同期沪深300开盘收益 | 后续可用连续超额收益做回归/排序 |
| A 状态 | 沪深300开盘收益 | `state`：低于 −1% 为 −1，高于 +1% 为 +1，其余为 0 |

A 的 IC 测评仍使用连续收益；将来三分类训练使用 `state`。分类时收益保留12位小数比较阈值，
避免浮点误差把恰好 ±1% 的样本分到涨跌类；原始连续收益不舍入。阈值尚未用于训练模型或
仓位映射。两路共享预测时段，便于组合研究；日频生成标签并不意味着每天必须换仓。

个股样本依据 **T 日**历史成员，退出成分股也保留；价格端点缺失/非正时目标为缺失值，
不后填、不断开日历重算窗口、不按未来退池情况删除样本。每段最后6个特征日没有完整的
T+6 端点，直接剔除；不从下一段读取价格来补标签。其他内部交叉验证也必须按 `label_end`
再次 purge，不能随机切分相互重叠的日频标签。

这是不扣成本的理论价格标签，不把未来停牌、涨跌停时的理论开盘价当成可成交结果。
B 比较的是前复权价格收益和价格指数收益，不宣称为严格的总回报超额收益；仍受供应商复权口径限制。
实际净收益由后续回测验证。

```bash
# 只生成训练段，输出目录必须不存在
.venv/bin/python scripts/prepare_labels.py --output output/output_labels_train_v1

# 不需要重复填写标签口径；读入前校验文件哈希、数据集身份与分段
.venv/bin/python scripts/run_factor_research.py \
  --mode evaluate --task B --split train \
  --dataset data/processed/csi300_2017_2025_v5_qfq \
  --label-bundle output/output_labels_train_v1 \
  --output output/output_alpha158_b_labels_v1
```

`scripts/prepare_labels.py` 的输出包含 `A.parquet`、`B.parquet` 和 `manifest.json`，保存价格端点日期、
连续收益、分类标签、缺失原因、口径及源文件哈希。原始指数开盘价必须与冻结数据包记录的
原始 benchmark 文件哈希一致，收盘价再与冻结面板对齐。不改写已准备行情包，其旧清单的
`labels_ready=false` 仍描述当时行情准备状态；新的标签清单才是标签就绪依据。
验证段可显式使用 `--split validation` 另建输出；最终测试需额外 `--allow-final-test`。

DeepSeek 接口依照 [官方 Chat API](https://api-docs.deepseek.com/) 接入：
固定目标为 `https://api.deepseek.com/chat/completions`，默认模型 `deepseek-flash`，关闭 thinking，
要求 JSON 对象。当前官方模型详情见 [模型文档](https://api-docs.deepseek.com/quick_start/pricing/)。
实际服务可能更新模型别名，轨迹同时记录请求模型和服务返回的模型名称，不声称别名固定快照。



单轮真实研究命令：

```bash
.venv/bin/python scripts/run_factor_research.py \
  --mode mine --provider deepseek --task B --split train \
  --dataset data/processed/csi300_2017_2025_v5_qfq \
  --label-bundle output/output_labels_train_v1 \
  --direction '研究量价关系对未来五交易日个股超额收益的解释力' \
  --rounds 1 --directions 1 --retries 1 \
  --output output/output_mining_b_deepseek_v1
```

A 将 `--task` 改为 `A`，方向改为市场趋势、波动或宽度，使用新输出目录。
API 接收研究提示、候选表达式与训练段测评摘要，不上传原始行情表；反馈摘要属于对外发送的
研究内容。默认每分支每轮候选最多2个，配置 `DEEPSEEK_MAX_CALLS` 限制每个客户端的网络尝试次数，
`DEEPSEEK_MAX_TOKENS` 限制每次回答长度。调用次数上限不是金额上限。

超时/限流/临时服务错误在研究循环中有界重试并短暂退避；认证失败、付费错误、
重定向或调用预算耗尽立即停止，不降级为伪造 LLM 结果。输出被截断会被拒绝。
`trajectory.json` 的 `api_calls` 保存安全错误码、模型名和服务返回的 token 用量。

挖掘可添加 `--reference-factors experiments/reference_alpha158_corr_b_train_v1.json`，
将 CORR30、CORR20、CORD30 的公式、训练段评分与局限作为示例传入各研究阶段。
参考文件必须声明匹配当前数据与标签的 `context_id`、任务和 `split=train`；其内容及哈希保存在轨迹中。
参考因子不计入新发现，不会改变现有停止条件。示例引导后的训练评分提升仍须检查共同有效样本、
平均 RankIC、年度稳定性及与模板的相关性，不能直接解释为样本外改善。


## 因子挖掘终止条件

停止决策由程序执行，配置保存在每次 `trajectory.json` 的 `config` 中。默认规则如下：

| 条件 | 默认值 | 命令行参数 |
| --- | --- | --- |
| 最大轮数 | 5 | `--rounds` |
| 每轮研究分支 | 2 | `--directions` |
| 每分支每轮候选数上限 | 2 | `--candidates-per-branch` |
| 全程候选提案数上限 | 20 | `--max-candidates` |
| 提供器调用次数上限（含重试） | 60 | `--max-calls` |
| 评分早停最少完整轮数 | 3 | `--min-rounds` |
| 连续停滞轮数 | 2 | `--patience` |
| 有意义的评分增量 | 0.002 | `--min-delta` |
| 连续无新增可测评候选轮数 | 2 | `--no-new-patience` |

一轮指所有规划分支各执行一次。`--rounds 1` 等硬上限优先于最少轮数；无新候选停止也可以早于第3轮。
提案数只计通过格式/表达式校验的提案，重复公式和缺字段的提案仍计入20次预算；格式错误的重试计入调用预算。
单次返回超出允许的候选数量会被拒绝，而不是静默裁掉。剩余候选预算不足2个时，会把本次上限降至剩余数量。

B 的合格评分只使用训练段：以整段平均 RankIC 的符号确定一个统一方向，乘到每个年度的平均 RankIC 上，
再取年度中位数。不逐年翻转符号，也不使用验证/测试结果决定符号。
默认要求总体覆盖率至少80%（`--min-coverage`），训练区间内每个年度至少20个有效日截面 RankIC
（`--min-year-observations`），统一方向后至少60%的年度为正（`--min-consistent-year-fraction`），
且中位数为正。缺年度不回退为全段高分。合格只代表可用于本次探索，不表示因子通过最终验证。

早停比较历史最佳合格评分与“最近一次有意义改善”的参考值；小幅提升可以累积。
例如参考值0.020，随后0.021、0.022，第三轮累计提升0.002，停滞计数归零。
如果随后两轮都未再累计提升0.002，则停止。没有任何合格评分时不触发评分停滞，仍受无新候选和硬上限约束。

“无新候选”指完整一轮没有新增可计算且能得到有限 RankIC 的唯一公式；纯重复、缺字段、常量或样本不足均不算新增可用结果。
已计算但未达到覆盖率/年度稳定性门槛的因子仍会留档，不被标记为合格父因子。
相关性去重尚未实现，因此不同表达式但经济含义高度相似的因子目前仍可能分别计数。

停止原因写入 `stop_reason`：`max_rounds`、`max_candidates`、`score_plateau`、`no_new_usable_candidates`、
`max_calls` 或安全技术错误码。`round_history` 记录每轮提案数、新增结果、重复数、最佳评分与停滞计数；
未完成的轮次明确标记 `completed=false`，不计入完整轮次早停统计。

预算中断使用 `status=interrupted`，CLI 返回码2；认证/服务重试耗尽等错误使用 `status=failed` 并抛出错误。
研究按规则结束为 `complete`，没有可测评结果为 `no_candidates`，都不代表样本外有效。
反馈调用中断前已算出的因子报告仍写入因子库。LLM 的 `stop` 保留于反馈中，并标记 `llm_stop_suggestion=true`，
不直接终止分支，也不跳过同批其他候选。

现有本机 `DEEPSEEK_MAX_CALLS` 如果低于60，仍遵守该较低限制；代码不会改写 `.env`。
实际采用的上限记录在 `effective_call_limit`，提供器调用计数为 `provider_attempts`，服务请求明细仍在 `api_calls`。
例如只运行本地回放并显式设置终止规则：

```bash
.venv/bin/python scripts/run_factor_research.py \
  --mode mine --provider replay --task B --split train \
  --dataset data/processed/csi300_2017_2025_v5_qfq \
  --label-bundle output/output_labels_train_v1 \
  --direction '终止条件本地验证' \
  --responses examples/mining_replay_b.json \
  --rounds 1 --directions 1 --max-candidates 2 --max-calls 4 \
  --output output/output_stopping_replay_v1
```

示例回放文件只有一轮记录，因此用 `--rounds 1`；多轮实验需要足够的回放记录或另行启用真实 API。

## 统一行业内排名模型训练

唯一入口为 `scripts/train_selection_model.py`，配置见 `config/ranking_training.json`。
旧收益回归、独立滚动/状态实验/固定划分入口已删除；历史 output 产物仍保留。
新运行直接读取冻结数据集、标签包和当前 B 因子库，不依赖历史实验缓存。

```bash
.venv/bin/python scripts/train_selection_model.py \
  --config config/ranking_training.json --output model_b_unified_fixed

# 同一入口：滚动验证、最多100个训练段筛选因子、组合输入和市场状态
.venv/bin/python scripts/train_selection_model.py \
  --config config/ranking_training.json --validation-mode rolling \
  --factor-scope selected --max-factors 100 --input-form combined \
  --market-state --rounds 500 --output model_b_unified_rolling
```

命令行显式参数覆盖 JSON；未知配置、无效参数报错，不静默回退。

| 配置 | 可选值及含义 |
| --- | --- |
| `validation_mode` | `fixed`：按数据集原始 train/validation 划分，一个模型预测整个验证段；`rolling`：逐年滚动 |
| `factor_scope` | `all`：全部可计算唯一公式，无IC/相关性筛选；`selected`：仅历史拟合窗口重评、去重，最多 `max_factors` 个 |
| `input_form` | `residual`：行业去均值；`percentile`：行业百分位减0.5；`combined`：二者结合 |
| `market_state` | 是否加入9个当日已知的市场/行业状态特征，状态特征不做行业去均值 |
| `max_estimators` / `--rounds` | 默认固定500轮 |
| `early_stopping` | 默认false；启用后只在历史训练段内部留出最后 `calibration_years` 年选轮数，再用完整历史训练窗口重训 |
| `training_years` | 仅滚动模式有效，默认3年 |

标签为未来五交易日超额收益在同日同行业的百分位：
`(average_rank(future_excess_return)-0.5)/valid_industry_member_count`。
LightGBM 对该标签做回归，预测值不是收益率。单股票或全相同收益的行业组不参与训练评价。
所有边界按真实 `label_end` 剔除跨界标签。固定模式不会把2022年加入训练；滚动模式预测
2023年时可以使用2022年已知标签。最终2024—2025测试集始终不读取。

因子在训练与验证的连续历史上因果计算，段间不重置预热，缺失不向未来填补。
固定模式2017年初无历史的长窗口特征保持NaN。历史行业分类从已校验源文件提取，
也可通过 `--industries` 提供。训练日期等权；全因子模式不会为了达到固定数量而编造缺失字段。

`selected` 模式按行业内RankIC重评训练段，覆盖率/有效日/年度稳定性门槛和相关性去重阈值
见配置。早停开启时因子筛选仅使用内部拟合段，校准及外层验证均不参与筛选。

输出根目录保存完整配置、输入/产物哈希、因子定义、特征质量、排名标签、汇总预测及指标。
`fixed/` 或 `fold_YYYY/` 保存模型、特征顺序、因子筛选记录、评估样本、全年预测及分年指标。
预测导出包括验证末端无标签日期；评估仅使用有效标签。指标包括每日行业RankIC均值、
行业中位数强弱准确率、覆盖率、预测并列半分准确率和排名标签RMSE。
输出目录禁止覆盖；每个模型保存后重新加载核对预测一致性。

验证段已在研究中多次查看，不代表独立最终测试；五日标签重叠，样本非独立。
本入口只评估模型，不运行策略回测。

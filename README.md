# myReTest

一个最小化的 A 股日频截面选股回测框架。策略每天收盘后同时查看所有股票，输出目标权重组合；组合层计算调仓订单，撮合层在下一交易日开盘成交。

## 数据流

```text
CsvDailyDataSource
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
python run_backtest.py
```

可选参数：

```bash
python run_backtest.py --initial-cash 1000000 --top-n 2 --output output
```

运行后生成：

```text
output/targets.csv
output/orders.csv
output/fills.csv
output/equity.csv
```

## 示例数据

`data/sample_bars.csv` 包含 6 只虚构 A 股、20 个交易日的日线数据：

```text
symbol,date,open,high,low,close,volume
```

这些数据仅用于验证截面排序和调仓流程，不代表真实市场行情。

## 设计说明

关键架构决策及简短理由见 `docs/ARCHITECTURE_DECISIONS.md`。

## A/B 信号接入与研究结果导出

```bash
.venv/bin/python run_signal_demo.py --exposure 0.5 --top-n 2 --output output_signal_demo
.venv/bin/python -m pytest -q
```

此示例使用虚构行情、演示股票池和固定仓位，只验证接口链路。`SignalStrategy`
接收选股 B、仓位 A、历史成分股三个 callable；信号模型版本和 ID 会保存在结果中。
真实历史股票池可以通过 `HistoricalUniverse.from_csv(path)` 接入，文件必须包含
`date,symbol`，每个研究交易日提供完整的当日有效成分股，缺日直接报错。
不要用行情中出现的股票或当前成分股名单替代真实历史成分股。

两个运行入口均可生成 `result.json` 和 `target_signals.csv`。JSON 保留信号、空仓目标、
订单、成交、每日权益、订单处理结果及期末待撮合订单；Decimal 保存为字符串。
原 `run_backtest.py` 继续生成原有四个 CSV，完整审计以 JSON 为准。

行情 CSV 可额外提供 `buy_block_reason` 和 `sell_block_reason`：非空表示该方向
在开盘时禁止成交，值会作为取消原因保存。例如数据适配器可以将停牌原因填入两个方向，
将开盘买入限制仅填入买入方向。**这些字段必须来自开盘时可知的数据**，框架不会根据
全天最高价、最低价或收盘价反推开盘交易状态；缺省空值仍是演示版“允许交易”假设。
实际历史交易规则识别和数据完整性审核仍需后续数据适配器完成。

未成交部分当天取消，下次信号重新调仓。信号 A 缺失时报错；纯选股实验显式提供固定仓位。
最后一个交易日生成的订单没有下个开盘可撮合，保留在 `pending_orders`，不虚构成交或取消。

## 沪深300真实行情数据包

`data/raw/akshare/` 已改用沪深300原始数据包（2016—2025年），原中证500数据归档到
`data/archive/`。数据文件、单位、下载方式与限制见 [数据说明](data/README.md)。
本数据包的股票池仍是下载时的当前成员快照，正式历史策略验证需要另补历史成分股；
演示入口默认使用的虚构 `sample_bars.csv` 保持不变。

## 静态风控与绩效评价

风控默认关闭，必须显式配置。演示开启单股15%、最多2只的限制：

```bash
.venv/bin/python run_signal_demo.py --exposure 0.5 --top-n 2 --max-stock-weight 0.15 --max-holdings 2 --output output_risk_demo
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

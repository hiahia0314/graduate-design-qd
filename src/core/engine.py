"""Daily cross-sectional backtest orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass

from src.framework.interfaces import Accounting, Broker, DataSource, Portfolio, RiskControl, Strategy
from src.framework.models import EquityRecord, ExposureSignal, Fill, HoldingRecord, Order, OrderOutcome, RiskDecision, SelectionSignal, TargetPortfolio


@dataclass(slots=True)
class BacktestResult:
    targets: list[TargetPortfolio] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    equity: list[EquityRecord] = field(default_factory=list)
    selection_signals: list[SelectionSignal] = field(default_factory=list)
    exposure_signals: list[ExposureSignal] = field(default_factory=list)
    order_outcomes: list[OrderOutcome] = field(default_factory=list)
    pending_orders: list[Order] = field(default_factory=list)
    risk_decisions: list[RiskDecision] = field(default_factory=list)
    risk_config: dict | None = None
    holdings: list[HoldingRecord] = field(default_factory=list)

    @property
    def total_return(self):
        if len(self.equity) < 2 or self.equity[0].total_equity == 0:
            return 0
        return self.equity[-1].total_equity / self.equity[0].total_equity - 1


class BacktestEngine:
    """Run one strategy evaluation and one target rebalance per trading day."""

    def __init__(
        self,
        data_source: DataSource,
        strategy: Strategy,
        portfolio: Portfolio,
        broker: Broker,
        accounting: Accounting,
        risk_control: RiskControl | None = None,
    ) -> None:
        self.data_source = data_source
        self.strategy = strategy
        self.portfolio = portfolio
        self.broker = broker
        self.accounting = accounting
        self.risk_control = risk_control

    def run(self) -> BacktestResult:
        result = BacktestResult()
        limits = getattr(self.risk_control, "limits", None)
        if limits is not None and is_dataclass(limits):
            result.risk_config = asdict(limits)
        previous_date = None

        for market in self.data_source.snapshots():
            if previous_date is not None and market.timestamp.date() <= previous_date:
                raise ValueError("Daily market dates must be strictly increasing")
            for symbol, bar in market.bars.items():
                if bar.timestamp != market.timestamp or bar.symbol != symbol:
                    raise ValueError("Bar identity and timestamp must match its market snapshot")
            previous_date = market.timestamp.date()
            fills = list(self.broker.match(market, self.accounting.cash))
            for fill in fills:
                self.accounting.on_fill(fill)
            result.fills.extend(fills)

            account = self.accounting.mark(market)
            for symbol, position in account.positions.items():
                value = account.position_values.get(symbol)
                if value is not None:
                    info = market.metadata.get(symbol)
                    result.holdings.append(HoldingRecord(
                        market.timestamp, symbol, position.quantity, value,
                        value / account.total_equity, info.industry if info else None,
                        symbol not in market.bars,
                    ))
            target = self.strategy.on_day(market)
            if self.risk_control is not None:
                decision = self.risk_control.evaluate(target, account, market)
                result.risk_decisions.append(decision)
                target = decision.output_target
            orders = list(self.portfolio.create_orders(target, account, market))
            self.broker.submit(orders)

            result.targets.append(target)
            result.orders.extend(orders)

        result.equity = list(getattr(self.accounting, "equity_records", []))
        result.selection_signals = list(getattr(self.strategy, "selection_signals", []))
        result.exposure_signals = list(getattr(self.strategy, "exposure_signals", []))
        result.order_outcomes = list(getattr(self.broker, "outcomes", []))
        result.pending_orders = list(getattr(self.broker, "pending_orders", []))
        return result

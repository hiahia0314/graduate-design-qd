"""Next-day-open execution for daily rebalance orders."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal, ROUND_HALF_UP
from uuid import uuid4

from src.framework.interfaces import Broker
from src.framework.models import CostModel, Fill, FillStatus, MarketSnapshot, Order, OrderOutcome, OrderStatus, Side


CENT = Decimal("0.01")
LOT_SIZE = 100


class NextOpenBroker(Broker):
    """Execute yesterday's sells first, then proportionally funded buys."""

    def __init__(self, costs: CostModel | None = None) -> None:
        self.costs = costs or CostModel()
        for value in (self.costs.commission_rate, self.costs.minimum_commission,
                      self.costs.stamp_duty_rate, self.costs.slippage):
            if not value.is_finite() or value < 0:
                raise ValueError("Costs must be non-negative and finite")
        if self.costs.slippage >= 1:
            raise ValueError("Slippage must be less than one")
        self._pending: list[Order] = []
        self.outcomes: list[OrderOutcome] = []
        self._order_ids: set[str] = set()

    @property
    def pending_orders(self) -> tuple[Order, ...]:
        return tuple(self._pending)

    def submit(self, orders: Sequence[Order]) -> None:
        ids = [order.order_id for order in orders]
        if len(set(ids)) != len(ids) or self._order_ids.intersection(ids):
            raise ValueError("Order IDs must be unique")
        for order in orders:
            if isinstance(order.quantity, bool) or not isinstance(order.quantity, int) or order.quantity <= 0:
                raise ValueError("Order quantity must be a positive integer")
            if order.side is Side.BUY and order.quantity % LOT_SIZE:
                raise ValueError("Buy quantity must be a multiple of the configured lot size")
        self._pending.extend(orders)
        self._order_ids.update(ids)

    def match(self, market: MarketSnapshot, available_cash: Decimal) -> list[Fill]:
        if any(order.timestamp.date() >= market.timestamp.date() for order in self._pending):
            raise ValueError("Orders can only execute on a later trading date")
        if not available_cash.is_finite() or available_cash < 0:
            raise ValueError("Available cash must be non-negative and finite")
        blocked = {}
        for order in self._pending:
            bar = market.bars.get(order.symbol)
            if bar is None:
                blocked[order.order_id] = "missing_bar"
            else:
                reason = bar.buy_block_reason if order.side is Side.BUY else bar.sell_block_reason
                if reason:
                    blocked[order.order_id] = reason
                elif not bar.open.is_finite() or bar.open <= 0:
                    raise ValueError(f"Invalid execution open price: {order.symbol}")
        pending, self._pending = self._pending, []
        sells = [order for order in pending if order.side is Side.SELL and order.order_id not in blocked]
        buys = [order for order in pending if order.side is Side.BUY and order.order_id not in blocked]
        fills: list[Fill] = []
        cash = available_cash

        for order in sells:
            bar = market.bars.get(order.symbol)
            if bar is None:
                continue
            fill = self._make_fill(order, market, order.quantity)
            fills.append(fill)
            cash += fill.price * fill.quantity - fill.total_cost

        requested_buy_cost = sum(
            self._buy_cost(order, market, order.quantity)
            for order in buys
            if order.symbol in market.bars
        )
        ratio = (
            Decimal("1")
            if requested_buy_cost <= cash or requested_buy_cost == 0
            else cash / requested_buy_cost
        )

        for order in buys:
            if order.symbol not in market.bars:
                continue
            scaled = int(Decimal(order.quantity) * ratio) // LOT_SIZE * LOT_SIZE
            quantity = self._affordable_quantity(order, market, scaled, cash)
            if quantity <= 0:
                continue
            fill = self._make_fill(order, market, quantity)
            fills.append(fill)
            cash -= fill.price * fill.quantity + fill.total_cost

        executed = {fill.order_id: fill.quantity for fill in fills}
        for order in pending:
            quantity = executed.get(order.order_id, 0)
            remainder = order.quantity - quantity
            status = OrderStatus.FILLED if remainder == 0 else (
                OrderStatus.PARTIAL_CANCELLED if quantity else OrderStatus.CANCELLED
            )
            self.outcomes.append(OrderOutcome(
                order.order_id, order.target_id, market.timestamp, status,
                order.quantity, quantity, remainder,
                blocked.get(order.order_id, "insufficient_cash_or_lot_rounding" if remainder else ""),
            ))
        return fills

    def _execution_price(self, order: Order, market: MarketSnapshot) -> Decimal:
        open_price = market.bars[order.symbol].open
        if order.side is Side.BUY:
            return open_price * (Decimal("1") + self.costs.slippage)
        return open_price * (Decimal("1") - self.costs.slippage)

    def _commission(self, gross: Decimal) -> Decimal:
        commission = max(
            self.costs.minimum_commission,
            gross * self.costs.commission_rate,
        )
        return commission.quantize(CENT, rounding=ROUND_HALF_UP)

    def _buy_cost(self, order: Order, market: MarketSnapshot, quantity: int) -> Decimal:
        price = self._execution_price(order, market)
        gross = price * quantity
        return gross + self._commission(gross)

    def _affordable_quantity(
        self,
        order: Order,
        market: MarketSnapshot,
        requested: int,
        cash: Decimal,
    ) -> int:
        quantity = requested
        while quantity > 0 and self._buy_cost(order, market, quantity) > cash:
            quantity -= LOT_SIZE
        return quantity

    def _make_fill(
        self,
        order: Order,
        market: MarketSnapshot,
        quantity: int,
    ) -> Fill:
        price = self._execution_price(order, market)
        gross = price * quantity
        commission = self._commission(gross)
        stamp_duty = (
            gross * self.costs.stamp_duty_rate
            if order.side is Side.SELL
            else Decimal("0")
        ).quantize(CENT, rounding=ROUND_HALF_UP)
        return Fill(
            fill_id=uuid4().hex,
            order_id=order.order_id,
            target_id=order.target_id,
            strategy_id=order.strategy_id,
            status=(
                FillStatus.FILLED
                if quantity == order.quantity
                else FillStatus.PARTIAL
            ),
            symbol=order.symbol,
            timestamp=market.timestamp,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            stamp_duty=stamp_duty,
        )
